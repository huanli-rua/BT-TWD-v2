"""后决策可靠性验证模块。"""

from __future__ import annotations


def _governance_cfg(config: dict) -> dict:
    return config.get("governance") or config.get("GOVERNANCE") or {}


def _cp_cfg(config: dict) -> dict:
    return _governance_cfg(config).get("cp", {})


def is_cp_disabled(config: dict) -> bool:
    governance_cfg = _governance_cfg(config)
    cp_cfg = governance_cfg.get("cp", {})
    ablation_cfg = governance_cfg.get("ablation", {})
    return bool(ablation_cfg.get("disable_cp_validation", False)) or not bool(cp_cfg.get("enabled", True))


def validate_post_decision(
    sample_id,
    bucket_id,
    twd_decision,
    posterior,
    risk_values,
    bucket_context,
    config,
):
    """验证当前 P/N/BND 决策是否可靠。

    规则：
    - TWD 输出 BND 时不调用 CP，直接进入 defer。
    - TWD 输出 P/N 时调用 split conformal prediction。
    - P 只有 CP prediction set 恰为 {1} 才可靠。
    - N 只有 CP prediction set 恰为 {0} 才可靠。
    """

    if twd_decision == "BND":
        return {
            "reliable": False,
            "score": 0.0,
            "reason": "TWD 输出 BND，直接进入 defer",
            "cp_disabled": is_cp_disabled(config),
            "cp_passed": False,
            "cp_rejected": False,
            "cp_set": [],
            "cp_p_value_0": 0.0,
            "cp_p_value_1": 0.0,
            "alpha_cp": float(_cp_cfg(config).get("alpha", 0.1)),
        }

    cp_cfg = _cp_cfg(config)
    alpha_cp = float(cp_cfg.get("alpha", 0.1))
    if is_cp_disabled(config):
        return {
            "reliable": True,
            "score": 1.0,
            "reason": "CP disabled, P/N accepted by default",
            "cp_disabled": True,
            "cp_passed": True,
            "cp_rejected": False,
            "cp_set": [],
            "cp_p_value_0": 0.0,
            "cp_p_value_1": 0.0,
            "alpha_cp": alpha_cp,
        }

    cp_validator = (bucket_context or {}).get("cp_validator")
    if cp_validator is None:
        return {
            "reliable": False,
            "score": 0.0,
            "reason": "CP validator missing, decision deferred",
            "cp_disabled": False,
            "cp_passed": False,
            "cp_rejected": True,
            "cp_set": [],
            "cp_p_value_0": 0.0,
            "cp_p_value_1": 0.0,
            "alpha_cp": alpha_cp,
        }

    cp_result = cp_validator.predict_set(posterior, alpha_cp)
    cp_set = cp_result["cp_set"]
    if twd_decision == "P":
        reliable = cp_set == [1]
        score = cp_result["cp_p_value_1"]
        expected = "{1}"
    elif twd_decision == "N":
        reliable = cp_set == [0]
        score = cp_result["cp_p_value_0"]
        expected = "{0}"
    else:
        reliable = False
        score = 0.0
        expected = "{}"

    reason = (
        f"CP prediction set={cp_set} 与 TWD 输出 {twd_decision} 匹配"
        if reliable
        else f"CP prediction set={cp_set} 未等于期望集合 {expected}，进入 defer"
    )
    return {
        "reliable": bool(reliable),
        "score": float(score),
        "reason": reason,
        "cp_disabled": False,
        "cp_passed": bool(reliable),
        "cp_rejected": not bool(reliable),
        "cp_set": cp_set,
        "cp_p_value_0": float(cp_result["cp_p_value_0"]),
        "cp_p_value_1": float(cp_result["cp_p_value_1"]),
        "alpha_cp": float(cp_result["alpha_cp"]),
    }


__all__ = ["validate_post_decision", "is_cp_disabled"]
