"""三支决策规则。"""

from __future__ import annotations

from ..metrics import predict_binary_by_cost


def decide_twd(posterior: float, alpha: float, beta: float) -> str:
    """按 alpha/beta 阈值输出 P/BND/N。"""

    if posterior >= alpha:
        return "P"
    if posterior <= beta:
        return "N"
    return "BND"


def decision_to_numeric(decision: str) -> int:
    if decision == "P":
        return 1
    if decision == "N":
        return 0
    return -1


def numeric_to_decision(value) -> str:
    if value == 1 or value == "1":
        return "P"
    if value == 0 or value == "0":
        return "N"
    return "BND"


def risk_values_for_probability(posterior: float, costs: dict) -> dict:
    """计算单样本在 P/N/BND 三种动作下的期望风险。"""

    p1 = float(posterior)
    p0 = 1.0 - p1
    return {
        "P": costs.get("C_TP", 0.0) * p1 + costs.get("C_FP", 0.0) * p0,
        "N": costs.get("C_FN", 0.0) * p1 + costs.get("C_TN", 0.0) * p0,
        "BND": costs.get("C_BP", 0.0) * p1 + costs.get("C_BN", 0.0) * p0,
    }


def force_close_by_min_risk(posterior: float, costs: dict) -> str:
    """ROOT 仍为 BND 时，在 P/N 中选择期望风险较小者强制闭合。"""

    risks = risk_values_for_probability(posterior, costs)
    return "P" if risks["P"] <= risks["N"] else "N"


__all__ = [
    "predict_binary_by_cost",
    "decide_twd",
    "decision_to_numeric",
    "numeric_to_decision",
    "risk_values_for_probability",
    "force_close_by_min_risk",
]
