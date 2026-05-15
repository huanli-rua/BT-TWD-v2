"""闭环 defer 生命周期。"""

from __future__ import annotations

from ..bucket.routing import bucket_level, parent_bucket
from ..twd.decision import decide_twd, risk_values_for_probability
from .context_update import update_decision_context
from .post_validation import is_cp_disabled, validate_post_decision


def _governance_cfg(config: dict) -> dict:
    return config.get("governance") or config.get("GOVERNANCE") or {}


def _progressive_cfg(config: dict) -> dict:
    return _governance_cfg(config).get("progressive_update", {})


def _cp_cfg(config: dict) -> dict:
    return _governance_cfg(config).get("cp", {})


def _is_progressive_disabled(config: dict) -> bool:
    governance_cfg = _governance_cfg(config)
    return bool(governance_cfg.get("ablation", {}).get("disable_progressive_update", False)) or not bool(
        governance_cfg.get("progressive_update", {}).get("enabled", True)
    )


def _root_forced_decision(aggregated_risk: dict) -> str:
    return "P" if float(aggregated_risk.get("P", 0.0)) <= float(aggregated_risk.get("N", 0.0)) else "N"


def _fallback_reliability(decision: str, config: dict, cp_result: dict | None = None) -> float:
    epsilon = float(_progressive_cfg(config).get("epsilon", 0.001))
    if decision == "BND":
        if cp_result:
            values = [float(cp_result.get("cp_p_value_0", 0.0)), float(cp_result.get("cp_p_value_1", 0.0))]
            return max(min(values), epsilon)
        return epsilon
    if is_cp_disabled(config):
        return 1.0
    if cp_result:
        if decision == "P":
            return float(cp_result.get("cp_p_value_1", epsilon))
        if decision == "N":
            return float(cp_result.get("cp_p_value_0", epsilon))
    if decision in {"P", "N"}:
        return 1.0
    return epsilon


def _validate_layer(sample_id, bucket_id, decision, posterior, risk_values, config, cp_validator):
    if decision == "BND":
        return {
            "reliable": False,
            "score": _fallback_reliability(decision, config),
            "reason": "TWD 输出 BND，直接进入 defer",
            "cp_disabled": is_cp_disabled(config),
            "cp_passed": False,
            "cp_rejected": False,
            "cp_set": [],
            "cp_p_value_0": 0.0,
            "cp_p_value_1": 0.0,
            "alpha_cp": float(_cp_cfg(config).get("alpha", 0.1)),
        }

    if cp_validator is None:
        return {
            "reliable": True,
            "score": 1.0,
            "reason": "defer lifecycle 未提供 CP validator，P/N 使用默认可靠性分数",
            "cp_disabled": is_cp_disabled(config),
            "cp_passed": True,
            "cp_rejected": False,
            "cp_set": [],
            "cp_p_value_0": 0.0,
            "cp_p_value_1": 0.0,
            "alpha_cp": float(_cp_cfg(config).get("alpha", 0.1)),
        }

    return validate_post_decision(
        sample_id=sample_id,
        bucket_id=bucket_id,
        twd_decision=decision,
        posterior=posterior,
        risk_values=risk_values,
        bucket_context={"level": bucket_level(bucket_id), "cp_validator": cp_validator},
        config=config,
    )


def _legacy_resolve(sample_id, start_bucket_id, posterior, thresholds, config, cp_validator=None):
    """progressive_update.enabled=false 时的旧闭合逻辑：parent/root 替代。"""

    costs = (config.get("THRESHOLD") or config.get("THRESHOLDS", {})).get("costs", {})
    current = start_bucket_id
    defer_path = []
    status_log = []
    context = {}

    while True:
        alpha, beta = thresholds.get(current, thresholds.get("ROOT", (0.5, 0.0)))
        risk_values = risk_values_for_probability(posterior, costs)
        decision = decide_twd(posterior, alpha, beta)
        level = bucket_level(current)
        validation = (
            {
                "reliable": True,
                "score": 1.0,
                "reason": "ROOT 不做 CP，强制闭合",
                "cp_disabled": is_cp_disabled(config),
                "cp_passed": False,
                "cp_rejected": False,
                "cp_set": [],
                "cp_p_value_0": 0.0,
                "cp_p_value_1": 0.0,
                "alpha_cp": float(_cp_cfg(config).get("alpha", 0.1)),
            }
            if current == "ROOT"
            else _validate_layer(sample_id, current, decision, posterior, risk_values, config, cp_validator)
        )
        defer_path.append(current)
        context = update_decision_context(
            context,
            {
                "bucket_id": current,
                "level": level,
                "decision": decision,
                "risk_values": risk_values,
                "reliability_score": _fallback_reliability(decision, config, validation),
            },
            config,
        )
        status_log.append(
            {
                "bucket_id": current,
                "level": level,
                "decision": decision,
                "alpha": float(alpha),
                "beta": float(beta),
                "reliability_score": float(_fallback_reliability(decision, config, validation)),
                "cp_passed": bool(validation.get("cp_passed", False)),
                "cp_rejected": bool(validation.get("cp_rejected", False)),
                "cp_disabled": bool(validation.get("cp_disabled", False)),
                "cp_set": validation.get("cp_set", []),
                "cp_p_value_0": float(validation.get("cp_p_value_0", 0.0)),
                "cp_p_value_1": float(validation.get("cp_p_value_1", 0.0)),
                "validation_reason": validation.get("reason", ""),
                "progressive_update_enabled": False,
            }
        )

        if current == "ROOT":
            forced = _root_forced_decision(context.get("aggregated_risk", risk_values))
            status_log.append({"bucket_id": "ROOT", "level": 0, "decision": forced, "forced": True})
            return {
                "final_decision": forced,
                "closure_bucket": "ROOT",
                "closure_level": 0,
                "defer_path": defer_path,
                "closed": True,
                "status_log": status_log,
                "context": context,
                "progressive_update_enabled": False,
            }

        if decision in {"P", "N"} and (bool(validation.get("reliable")) or is_cp_disabled(config)):
            return {
                "final_decision": decision,
                "closure_bucket": current,
                "closure_level": level,
                "defer_path": defer_path,
                "closed": True,
                "status_log": status_log,
                "context": context,
                "progressive_update_enabled": False,
            }

        current = parent_bucket(current)


def resolve_deferred_sample(
    sample_id,
    start_bucket_id,
    tree,
    posterior,
    thresholds,
    config,
    cp_validator=None,
    initial_decision=None,
    initial_validation=None,
):
    """沿 leaf -> parent -> root 渐进式累积风险证据并闭合 defer 样本。"""

    if _is_progressive_disabled(config):
        return _legacy_resolve(sample_id, start_bucket_id, posterior, thresholds, config, cp_validator=cp_validator)

    costs = (config.get("THRESHOLD") or config.get("THRESHOLDS", {})).get("costs", {})
    current = start_bucket_id
    defer_path = []
    status_log = []
    context = {}

    while True:
        alpha, beta = thresholds.get(current, thresholds.get("ROOT", (0.5, 0.0)))
        risk_values = risk_values_for_probability(posterior, costs)
        decision = initial_decision if current == start_bucket_id and initial_decision else decide_twd(posterior, alpha, beta)
        level = bucket_level(current)

        validation = (
            initial_validation
            if current == start_bucket_id and initial_validation is not None
            else (
                {
                    "reliable": True,
                    "score": 1.0,
                    "reason": "ROOT 不做 CP，强制闭合",
                    "cp_disabled": is_cp_disabled(config),
                    "cp_passed": False,
                    "cp_rejected": False,
                    "cp_set": [],
                    "cp_p_value_0": 0.0,
                    "cp_p_value_1": 0.0,
                    "alpha_cp": float(_cp_cfg(config).get("alpha", 0.1)),
                }
                if current == "ROOT"
                else _validate_layer(sample_id, current, decision, posterior, risk_values, config, cp_validator)
            )
        )
        reliability_score = _fallback_reliability(decision, config, validation)

        defer_path.append(current)
        context = update_decision_context(
            context,
            {
                "bucket_id": current,
                "level": level,
                "decision": decision,
                "risk_values": risk_values,
                "reliability_score": reliability_score,
            },
            config,
        )
        aggregated_decision = context["aggregated_decision"]
        aggregated_risk = context["aggregated_risk"]

        status_log.append(
            {
                "bucket_id": current,
                "level": level,
                "decision": decision,
                "alpha": float(alpha),
                "beta": float(beta),
                "reliability_score": float(reliability_score),
                "aggregated_decision": aggregated_decision,
                "aggregated_risk": aggregated_risk,
                "cp_disabled": bool(validation.get("cp_disabled", False)),
                "cp_passed": bool(validation.get("cp_passed", False)),
                "cp_rejected": bool(validation.get("cp_rejected", False)),
                "cp_set": validation.get("cp_set", []),
                "cp_p_value_0": float(validation.get("cp_p_value_0", 0.0)),
                "cp_p_value_1": float(validation.get("cp_p_value_1", 0.0)),
                "validation_reason": validation.get("reason", ""),
                "progressive_update_enabled": True,
            }
        )

        if current == "ROOT":
            forced = _root_forced_decision(aggregated_risk)
            status_log.append({"bucket_id": "ROOT", "level": 0, "decision": forced, "forced": True})
            return {
                "final_decision": forced,
                "closure_bucket": "ROOT",
                "closure_level": 0,
                "defer_path": defer_path,
                "closed": True,
                "status_log": status_log,
                "context": context,
                "progressive_update_enabled": True,
            }

        # 触发 defer 的原层只记录证据，不立即用自身聚合结果闭合；
        # 对 P/N 层，如果 CP 未通过，则继续向 parent 累积证据。
        can_close_by_aggregation = (
            current != start_bucket_id
            and aggregated_decision in {"P", "N"}
            and decision in {"P", "N"}
            and (bool(validation.get("reliable")) or is_cp_disabled(config))
        )
        if can_close_by_aggregation:
            return {
                "final_decision": aggregated_decision,
                "closure_bucket": current,
                "closure_level": level,
                "defer_path": defer_path,
                "closed": True,
                "status_log": status_log,
                "context": context,
                "progressive_update_enabled": True,
            }

        current = parent_bucket(current)


__all__ = ["resolve_deferred_sample"]
