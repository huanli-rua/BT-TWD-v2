"""渐进式风险证据更新。"""

from __future__ import annotations


def _progressive_cfg(config: dict) -> dict:
    governance_cfg = config.get("governance") or config.get("GOVERNANCE") or {}
    return governance_cfg.get("progressive_update", {})


def compute_adaptive_weights(evidence, epsilon: float = 0.001):
    """根据可靠性分数自适应计算证据权重。

    不使用固定层级权重；若所有可靠性分数退化为 0，则使用均匀权重。
    """

    if not evidence:
        return []

    scores = []
    for item in evidence:
        score = item.get("reliability_score", epsilon)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = epsilon
        scores.append(max(score, 0.0))

    total = sum(scores)
    if total <= 0:
        return [1.0 / len(evidence)] * len(evidence)
    return [score / total for score in scores]


def aggregate_risks(evidence):
    """按 evidence 中的 weight 聚合 P/BND/N 风险。"""

    aggregated = {"P": 0.0, "BND": 0.0, "N": 0.0}
    for item in evidence:
        weight = float(item.get("weight", 0.0))
        risks = item.get("risk_values", {}) or {}
        aggregated["P"] += weight * float(risks.get("P", 0.0))
        aggregated["BND"] += weight * float(risks.get("BND", risks.get("B", 0.0)))
        aggregated["N"] += weight * float(risks.get("N", 0.0))
    return aggregated


def _argmin_decision(aggregated_risk: dict) -> str:
    return min(("P", "BND", "N"), key=lambda key: float(aggregated_risk.get(key, 0.0)))


def update_decision_context(previous_context, current_bucket_context, config):
    """加入当前层风险证据并更新聚合风险。

    返回字段包括：
    - evidence：路径上已激活的层级证据；
    - aggregated_risk：加权后的 P/BND/N 风险；
    - aggregated_decision：聚合风险最小的动作；
    - path：bucket 路径。
    """

    progressive_cfg = _progressive_cfg(config)
    epsilon = float(progressive_cfg.get("epsilon", 0.001))

    updated = dict(previous_context or {})
    evidence = [dict(item) for item in updated.get("evidence", [])]
    path = list(updated.get("path", []))

    bucket_id = current_bucket_context.get("bucket_id")
    if bucket_id is not None:
        path.append(bucket_id)

    evidence.append(
        {
            "bucket_id": bucket_id,
            "level": int(current_bucket_context.get("level", 0)),
            "decision": current_bucket_context.get("decision", "BND"),
            "risk_values": dict(current_bucket_context.get("risk_values", {}) or {}),
            "reliability_score": float(current_bucket_context.get("reliability_score", epsilon)),
            "weight": 0.0,
        }
    )

    weights = compute_adaptive_weights(evidence, epsilon=epsilon)
    for item, weight in zip(evidence, weights):
        item["weight"] = float(weight)

    aggregated_risk = aggregate_risks(evidence)
    updated["evidence"] = evidence
    updated["aggregated_risk"] = aggregated_risk
    updated["aggregated_decision"] = _argmin_decision(aggregated_risk)
    updated["path"] = path
    return updated


__all__ = ["update_decision_context", "compute_adaptive_weights", "aggregate_risks"]
