"""BND early rescue decision rules."""

from __future__ import annotations


def _governance_cfg(config: dict | None) -> dict:
    if not config:
        return {}
    return config.get("governance") or config.get("GOVERNANCE") or {}


def _rescue_cfg(config: dict | None) -> dict:
    cfg = dict(_governance_cfg(config).get("bnd_early_rescue", {}) or {})
    cfg.setdefault("posterior_margin_threshold", 0.10)
    cfg.setdefault("risk_gap_threshold", 0.05)
    cfg.setdefault("cp_gap_threshold", 0.10)
    cfg.setdefault("cp_override_threshold", 0.20)
    cfg.setdefault("bucket_margin_threshold", 0.15)
    cfg.setdefault("min_bucket_support", 20)
    cfg.setdefault("min_conditions", 2)
    cfg.setdefault("directional_rescue_enabled", True)
    cfg.setdefault("min_aligned_conditions", 2)
    cfg.setdefault("min_direction_gap", 1)
    cfg.setdefault("directional_guard_enabled", True)
    cfg.setdefault("parent_directional_guard_enabled", True)
    cfg.setdefault("n_bucket_pos_rate_max", 0.35)
    cfg.setdefault("n_posterior_max", 0.40)
    cfg.setdefault("n_min_risk_gap", 0.08)
    cfg.setdefault("p_bucket_pos_rate_min", 0.40)
    cfg.setdefault("p_posterior_min", 0.35)
    cfg.setdefault("p_min_risk_gap", 0.08)
    cfg.setdefault("parent_n_bucket_pos_rate_max", 0.32)
    cfg.setdefault("parent_p_bucket_pos_rate_min", 0.42)
    cfg.setdefault("parent_ambiguous_posterior_band", 0.03)
    return cfg


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cp_values(cp_result: dict | None) -> tuple[float | None, float | None]:
    if not cp_result:
        return None, None
    p_pos = _optional_float(cp_result.get("cp_p_value_pos", cp_result.get("cp_p_value_1")))
    p_neg = _optional_float(cp_result.get("cp_p_value_neg", cp_result.get("cp_p_value_0")))
    if p_pos is None or p_neg is None:
        return None, None
    # Existing BND validation uses zero placeholders; those are not usable CP evidence.
    if p_pos <= 0.0 and p_neg <= 0.0:
        return None, None
    return p_pos, p_neg


def _bucket_values(bucket_context: dict | None) -> tuple[int, float | None, float | None]:
    if not bucket_context:
        return 0, None, None

    support = _optional_float(bucket_context.get("bucket_support", bucket_context.get("n_val")))
    pos_rate = _optional_float(bucket_context.get("bucket_pos_rate"))
    if pos_rate is None:
        pos_val = _optional_float(bucket_context.get("pos_val"))
        if pos_val is not None and support is not None and support > 0:
            pos_rate = pos_val / support

    if support is None or support <= 0 or pos_rate is None:
        return 0, None, None

    pos_rate = max(0.0, min(1.0, float(pos_rate)))
    return int(support), pos_rate, abs(pos_rate - 0.5)


def _cost_boundary(config: dict | None) -> float | None:
    threshold_cfg = (config or {}).get("THRESHOLD") or (config or {}).get("THRESHOLDS") or {}
    costs = threshold_cfg.get("costs", {}) or {}
    fp = _optional_float(costs.get("C_FP", costs.get("fp")))
    fn = _optional_float(costs.get("C_FN", costs.get("fn")))
    if fp is None or fn is None or fp + fn <= 0:
        return None
    return float(fp / (fp + fn))


def _add_evidence(evidence: list[dict], name: str, passed: bool, direction: str | None, strength: float | None) -> None:
    if not passed or direction not in {"P", "N"}:
        return
    evidence.append(
        {
            "name": name,
            "direction": direction,
            "strength": 0.0 if strength is None else float(strength),
        }
    )


def _support_by_direction(evidence: list[dict]) -> dict[str, list[str]]:
    support = {"P": [], "N": []}
    for item in evidence:
        direction = item.get("direction")
        name = item.get("name")
        if direction in support and name:
            support[direction].append(str(name))
    return support


def _guard_threshold(cfg: dict, key: str, parent_key: str, is_parent_layer: bool):
    if is_parent_layer and parent_key in cfg and cfg.get(parent_key) is not None:
        return cfg.get(parent_key)
    return cfg.get(key)


def _directional_guard(
    decision: str | None,
    posterior: float,
    risk_gap: float,
    bucket_pos_rate: float | None,
    cfg: dict,
    config: dict | None,
    is_parent_layer: bool,
) -> tuple[bool, str]:
    if decision not in {"P", "N"} or not bool(cfg.get("directional_guard_enabled", True)):
        return False, ""
    if is_parent_layer and not bool(cfg.get("parent_directional_guard_enabled", True)):
        return False, ""

    n_bucket_max = float(_guard_threshold(cfg, "n_bucket_pos_rate_max", "parent_n_bucket_pos_rate_max", is_parent_layer))
    p_bucket_min = float(_guard_threshold(cfg, "p_bucket_pos_rate_min", "parent_p_bucket_pos_rate_min", is_parent_layer))
    n_posterior_max = float(_guard_threshold(cfg, "n_posterior_max", "parent_n_posterior_max", is_parent_layer))
    p_posterior_min = float(_guard_threshold(cfg, "p_posterior_min", "parent_p_posterior_min", is_parent_layer))
    n_min_risk_gap = float(_guard_threshold(cfg, "n_min_risk_gap", "parent_n_min_risk_gap", is_parent_layer))
    p_min_risk_gap = float(_guard_threshold(cfg, "p_min_risk_gap", "parent_p_min_risk_gap", is_parent_layer))

    if decision == "N":
        bucket_strong = bucket_pos_rate is not None and bucket_pos_rate <= n_bucket_max
        posterior_strong = posterior <= n_posterior_max
        risk_strong = risk_gap >= n_min_risk_gap
        if not (bucket_strong or posterior_strong or risk_strong):
            return True, "weak_n_evidence"
    else:
        bucket_strong = bucket_pos_rate is not None and bucket_pos_rate >= p_bucket_min
        posterior_strong = posterior >= p_posterior_min
        risk_strong = risk_gap >= p_min_risk_gap
        if not (bucket_strong or posterior_strong or risk_strong):
            return True, "weak_p_evidence"

    if is_parent_layer:
        boundary = _cost_boundary(config)
        band = float(cfg.get("parent_ambiguous_posterior_band", 0.03) or 0.0)
        if boundary is not None and band > 0 and abs(posterior - boundary) < band and not bucket_strong:
            return True, "ambiguous_parent_cost_boundary"

    return False, ""


def _legacy_decision(
    conditions: list[str],
    risk_gap: float,
    risk_gap_threshold: float,
    risk_p: float,
    risk_n: float,
    bucket_pos_rate: float | None,
    cp_gap: float | None,
    cp_override_threshold: float,
    cp_pos: float | None,
    cp_neg: float | None,
) -> tuple[str | None, str]:
    if cp_gap is not None and cp_gap >= cp_override_threshold and cp_pos is not None and cp_neg is not None:
        return ("P" if cp_pos >= cp_neg else "N"), "cp_override"
    if risk_gap >= risk_gap_threshold:
        decision = "P" if risk_p <= risk_n else "N"
    elif bucket_pos_rate is not None and "bucket_margin" in conditions:
        decision = "P" if bucket_pos_rate >= 0.5 else "N"
    else:
        decision = "P" if risk_p <= risk_n else "N"

    if len(conditions) == 1 and conditions[0] in {"posterior_margin", "risk_gap", "bucket_margin"}:
        return decision, conditions[0]
    return decision, "combined_conditions"


def evaluate_bnd_early_rescue(
    posterior,
    risk_values,
    cp_result=None,
    bucket_context=None,
    config=None,
):
    cfg = _rescue_cfg(config)
    posterior_margin_threshold = float(cfg.get("posterior_margin_threshold", 0.10))
    risk_gap_threshold = float(cfg.get("risk_gap_threshold", 0.05))
    cp_gap_threshold = float(cfg.get("cp_gap_threshold", 0.10))
    cp_override_threshold = float(cfg.get("cp_override_threshold", 0.20))
    bucket_margin_threshold = float(cfg.get("bucket_margin_threshold", 0.15))
    min_bucket_support = int(cfg.get("min_bucket_support", 20))
    min_conditions = int(cfg.get("min_conditions", 2))
    directional_enabled = bool(cfg.get("directional_rescue_enabled", True))
    min_aligned_conditions = int(cfg.get("min_aligned_conditions", 2))
    min_direction_gap = int(cfg.get("min_direction_gap", 1))

    p = float(posterior)
    risk_p = float((risk_values or {}).get("P", 0.0))
    risk_n = float((risk_values or {}).get("N", 0.0))
    posterior_margin = abs(p - 0.5)
    risk_gap = abs(risk_p - risk_n)

    cp_pos, cp_neg = _cp_values(cp_result)
    cp_gap = abs(cp_pos - cp_neg) if cp_pos is not None and cp_neg is not None else None
    bucket_support, bucket_pos_rate, bucket_class_margin = _bucket_values(bucket_context)

    conditions = []
    evidence = []
    if posterior_margin >= posterior_margin_threshold:
        conditions.append("posterior_margin")
        _add_evidence(evidence, "posterior_margin", True, "P" if p >= 0.5 else "N", posterior_margin)
    if risk_gap >= risk_gap_threshold:
        conditions.append("risk_gap")
        _add_evidence(evidence, "risk_gap", True, "P" if risk_p <= risk_n else "N", risk_gap)
    if cp_gap is not None and cp_gap >= cp_gap_threshold:
        conditions.append("cp_gap")
        _add_evidence(evidence, "cp_gap", True, "P" if cp_pos >= cp_neg else "N", cp_gap)
    if bucket_support >= min_bucket_support and bucket_class_margin is not None and bucket_class_margin >= bucket_margin_threshold:
        conditions.append("bucket_margin")
        _add_evidence(evidence, "bucket_margin", True, "P" if bucket_pos_rate >= 0.5 else "N", bucket_class_margin)

    directional_support = _support_by_direction(evidence)
    p_support_count = len(directional_support["P"])
    n_support_count = len(directional_support["N"])
    if p_support_count >= n_support_count:
        aligned_decision = "P"
        aligned_count = p_support_count
        other_count = n_support_count
    else:
        aligned_decision = "N"
        aligned_count = n_support_count
        other_count = p_support_count
    direction_gap = abs(p_support_count - n_support_count)

    if directional_enabled:
        should_rescue = aligned_count >= min_aligned_conditions and direction_gap >= min_direction_gap
    else:
        should_rescue = len(conditions) >= min_conditions
    decision = None
    rescue_reason = "insufficient_conditions"
    blocked_by_directional_guard = False
    guard_reason = ""
    is_parent_layer = bool((bucket_context or {}).get("is_parent_rescue", False))

    if should_rescue:
        if directional_enabled:
            decision = aligned_decision
            if cp_gap is not None and cp_gap >= cp_override_threshold and "cp_gap" in directional_support[decision]:
                rescue_reason = "cp_override"
            else:
                rescue_reason = "direction_aligned"
        else:
            decision, rescue_reason = _legacy_decision(
                conditions,
                risk_gap,
                risk_gap_threshold,
                risk_p,
                risk_n,
                bucket_pos_rate,
                cp_gap,
                cp_override_threshold,
                cp_pos,
                cp_neg,
            )
        blocked_by_directional_guard, guard_reason = _directional_guard(
            decision,
            p,
            risk_gap,
            bucket_pos_rate,
            cfg,
            config,
            is_parent_layer,
        )
        if blocked_by_directional_guard:
            should_rescue = False
            decision = None
            rescue_reason = "directional_guard_blocked"

    return {
        "should_rescue": bool(should_rescue),
        "decision": decision,
        "rescue_reason": rescue_reason,
        "conditions": list(conditions),
        "directional_evidence": list(evidence),
        "directional_support": directional_support,
        "p_support_count": int(p_support_count),
        "n_support_count": int(n_support_count),
        "direction_gap": int(direction_gap),
        "blocked_by_directional_guard": bool(blocked_by_directional_guard),
        "directional_guard_reason": guard_reason,
        "is_parent_rescue": bool(is_parent_layer),
        "posterior_margin": float(posterior_margin),
        "risk_gap": float(risk_gap),
        "cp_gap": None if cp_gap is None else float(cp_gap),
        "bucket_class_margin": None if bucket_class_margin is None else float(bucket_class_margin),
        "bucket_support": int(bucket_support),
        "bucket_pos_rate": None if bucket_pos_rate is None else float(bucket_pos_rate),
    }


__all__ = ["evaluate_bnd_early_rescue"]
