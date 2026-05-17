"""样本级过程记录。"""

from __future__ import annotations

import json

from ..bucket.routing import bucket_level


def _json_dump(value):
    return json.dumps(value, ensure_ascii=False)


def build_sample_record(
    dataset_name,
    fold_id,
    sample_id,
    true_label,
    posterior,
    original_bucket_id,
    original_twd_decision,
    validation,
    defer_result,
    final_regret,
    original_regret=0.0,
    bucket_context=None,
):
    reliable = bool(validation.get("reliable"))
    final_decision = original_twd_decision if reliable else defer_result.get("final_decision")
    context = defer_result.get("context", {}) if defer_result else {}
    aggregated_risk = context.get("aggregated_risk", {}) or {}
    evidence = context.get("evidence", []) or []
    bucket_context = bucket_context or {}
    raw_bucket_id = bucket_context.get("raw_bucket_id", original_bucket_id)
    effective_bucket_id = bucket_context.get("effective_bucket_id", original_bucket_id)
    rescue_result = (defer_result or {}).get("bnd_early_rescue", {}) or {}
    rescue_attempt = rescue_result or (defer_result or {}).get("bnd_early_rescue_attempt", {}) or {}
    rescue_used = bool(rescue_result.get("should_rescue", False)) if not reliable else False
    return {
        "dataset_name": dataset_name,
        "fold_id": fold_id,
        "sample_id": sample_id,
        "true_label": int(true_label),
        "posterior": float(posterior),
        "raw_bucket_id": raw_bucket_id,
        "raw_bucket_level": bucket_level(raw_bucket_id),
        "effective_bucket_id": effective_bucket_id,
        "effective_bucket_level": bucket_level(effective_bucket_id),
        "bucket_was_weak": bool(bucket_context.get("bucket_was_weak", raw_bucket_id != effective_bucket_id)),
        "weak_reason": bucket_context.get("weak_reason", ""),
        "threshold_source_bucket": bucket_context.get("threshold_source_bucket", effective_bucket_id),
        "bucket_score": float(bucket_context.get("bucket_score", 0.0)),
        "bucket_parent_score": float(bucket_context.get("bucket_parent_score", 0.0)),
        "bucket_gain": float(bucket_context.get("bucket_gain", 0.0)),
        "original_bucket_id": original_bucket_id,
        "original_bucket_level": bucket_level(original_bucket_id),
        "original_twd_decision": original_twd_decision,
        "original_is_bnd": original_twd_decision == "BND",
        "original_regret": float(original_regret),
        "post_validation_reliable": reliable,
        "validation_score": float(validation.get("score", 0.0)),
        "validation_reason": validation.get("reason", ""),
        "cp_disabled": bool(validation.get("cp_disabled", False)),
        "cp_passed": bool(validation.get("cp_passed", False)),
        "cp_rejected": bool(validation.get("cp_rejected", False)),
        "cp_set": json.dumps(validation.get("cp_set", []), ensure_ascii=False),
        "cp_p_value_0": float(validation.get("cp_p_value_0", 0.0)),
        "cp_p_value_1": float(validation.get("cp_p_value_1", 0.0)),
        "alpha_cp": float(validation.get("alpha_cp", 0.1)),
        "defer_trigger_source": "" if reliable else ("BND" if original_twd_decision == "BND" else "post_validation"),
        "defer_path": "" if reliable else _json_dump(defer_result.get("defer_path", [])),
        "closure_bucket": original_bucket_id if reliable else defer_result.get("closure_bucket"),
        "closure_level": bucket_level(original_bucket_id) if reliable else defer_result.get("closure_level"),
        "closure_reason": "" if reliable else defer_result.get("closure_reason", ""),
        "final_decision": final_decision,
        "closed": True if reliable else bool(defer_result.get("closed")),
        "final_regret": float(final_regret),
        "bnd_early_rescue_used": rescue_used,
        "bnd_early_rescue_decision": rescue_result.get("decision", "") if rescue_used else "",
        "bnd_early_rescue_reason": rescue_result.get("rescue_reason", "") if rescue_used else "",
        "bnd_posterior_margin": rescue_attempt.get("posterior_margin"),
        "bnd_risk_gap": rescue_attempt.get("risk_gap"),
        "bnd_cp_gap": rescue_attempt.get("cp_gap"),
        "progressive_update_enabled": bool(defer_result.get("progressive_update_enabled", False)) if not reliable else False,
        "evidence_path": "" if reliable else _json_dump(evidence),
        "aggregated_risk": "" if reliable else _json_dump(aggregated_risk),
        "aggregated_risk_P": float(aggregated_risk.get("P", 0.0)) if not reliable else 0.0,
        "aggregated_risk_BND": float(aggregated_risk.get("BND", 0.0)) if not reliable else 0.0,
        "aggregated_risk_N": float(aggregated_risk.get("N", 0.0)) if not reliable else 0.0,
        "aggregated_decision": "" if reliable else context.get("aggregated_decision", ""),
        "evidence_weights": "" if reliable else _json_dump([item.get("weight", 0.0) for item in evidence]),
        "evidence_reliability_scores": ""
        if reliable
        else _json_dump([item.get("reliability_score", 0.0) for item in evidence]),
    }


__all__ = ["build_sample_record"]
