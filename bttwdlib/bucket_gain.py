import warnings

import numpy as np


def compute_bucket_score(metrics: dict, score_cfg: dict) -> float:
    """
    根据桶内评估指标计算桶的综合得分，分数越大越好。

    支持的模式：
        - "bac" / "regret" / "bac_regret": 兼容旧版，继续使用 BAC 与 Regret 的线性组合。
        - "f1_regret_bnd": 按 F1 - λ1 * Regret - λ2 * BND_ratio 计算，F1 高、Regret 低、BND_ratio 小的桶得分更高。

    参数：
        metrics: dict，至少包含与当前模式对应的指标键（大小写均可，如 "F1"/"f1"）。
        score_cfg: dict，对应 cfg["SCORE"].
    """

    def _extract_metric(metric_names: list[str]) -> float | None:
        for name in metric_names:
            if name in metrics:
                return metrics.get(name)
        return None

    def _safe_value(value: float | None, name: str) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            warnings.warn(
                f"compute_bucket_score: metric '{name}' is missing or NaN, defaulting to 0.0.",
                RuntimeWarning,
            )
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            warnings.warn(
                f"compute_bucket_score: metric '{name}' is invalid ({value!r}), defaulting to 0.0.",
                RuntimeWarning,
            )
            return 0.0

    mode = str(score_cfg.get("bucket_score_mode", "f1_regret_bnd") or "").lower()

    # 兼容旧模式参数
    w_bac = float(score_cfg.get("bac_weight", 1.0))
    w_reg = float(score_cfg.get("regret_weight", 1.0))
    reg_sign = float(score_cfg.get("regret_sign", -1.0))

    # 新模式参数
    f1_weight = float(score_cfg.get("f1_weight", 1.0))
    regret_weight = float(score_cfg.get("regret_weight", 1.0))
    bnd_weight = float(score_cfg.get("bnd_weight", 1.0))

    bac = _extract_metric(["BAC", "bac"])
    regret = _extract_metric(["Regret", "regret"])

    if mode == "bac":
        return -np.inf if bac is None else w_bac * float(bac)

    if mode == "regret":
        return -np.inf if regret is None else reg_sign * w_reg * float(regret)

    if mode == "bac_regret":
        bac_val = 0.0 if bac is None else float(bac)
        regret_val = 0.0 if regret is None else float(regret)
        return w_bac * bac_val + reg_sign * w_reg * regret_val

    f1 = _safe_value(_extract_metric(["F1", "f1"]), "F1")
    regret_val = _safe_value(regret, "Regret")
    bnd_ratio = _safe_value(_extract_metric(["BND_ratio", "bnd_ratio", "BND_ratio_mean"]), "BND_ratio")

    return f1_weight * f1 - regret_weight * regret_val - bnd_weight * bnd_ratio


def compute_bucket_gain(parent_score: float, child_scores: list[float], child_weights: list[float], gamma: float) -> float:
    """
    计算桶增益：Gain = sum(w_k * S_k) - S_parent - gamma * ΔN_bucket

    ΔN_bucket = 新增桶数（子桶个数-1），gamma 用于复杂度惩罚。
    """

    if len(child_scores) != len(child_weights):
        raise ValueError("child_scores 与 child_weights 长度不一致")

    weighted_child_score = float(np.sum(np.array(child_scores) * np.array(child_weights)))
    delta_bucket = max(len(child_scores) - 1, 0)
    return weighted_child_score - float(parent_score) - float(gamma) * delta_bucket
