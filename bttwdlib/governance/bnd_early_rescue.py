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
    cfg.setdefault("min_conditions", 2)
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


def evaluate_bnd_early_rescue(
    posterior,
    risk_values,
    cp_result=None,
    config=None,
):
    cfg = _rescue_cfg(config)
    posterior_margin_threshold = float(cfg.get("posterior_margin_threshold", 0.10))
    risk_gap_threshold = float(cfg.get("risk_gap_threshold", 0.05))
    cp_gap_threshold = float(cfg.get("cp_gap_threshold", 0.10))
    cp_override_threshold = float(cfg.get("cp_override_threshold", 0.20))
    min_conditions = int(cfg.get("min_conditions", 2))

    p = float(posterior)
    risk_p = float((risk_values or {}).get("P", 0.0))
    risk_n = float((risk_values or {}).get("N", 0.0))
    posterior_margin = abs(p - 0.5)
    risk_gap = abs(risk_p - risk_n)

    cp_pos, cp_neg = _cp_values(cp_result)
    cp_gap = abs(cp_pos - cp_neg) if cp_pos is not None and cp_neg is not None else None

    conditions = []
    if posterior_margin >= posterior_margin_threshold:
        conditions.append("posterior_margin")
    if risk_gap >= risk_gap_threshold:
        conditions.append("risk_gap")
    if cp_gap is not None and cp_gap >= cp_gap_threshold:
        conditions.append("cp_gap")

    should_rescue = len(conditions) >= min_conditions
    decision = None
    rescue_reason = "insufficient_conditions"

    if should_rescue:
        if cp_gap is not None and cp_gap >= cp_override_threshold:
            decision = "P" if cp_pos >= cp_neg else "N"
            rescue_reason = "cp_override"
        else:
            decision = "P" if risk_p <= risk_n else "N"
            if len(conditions) == 1 and conditions[0] in {"posterior_margin", "risk_gap"}:
                rescue_reason = conditions[0]
            else:
                rescue_reason = "combined_conditions"

    return {
        "should_rescue": bool(should_rescue),
        "decision": decision,
        "rescue_reason": rescue_reason,
        "posterior_margin": float(posterior_margin),
        "risk_gap": float(risk_gap),
        "cp_gap": None if cp_gap is None else float(cp_gap),
    }


__all__ = ["evaluate_bnd_early_rescue"]
