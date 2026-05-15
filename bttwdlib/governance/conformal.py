"""Split conformal 后决策可靠性验证工具。"""

from __future__ import annotations

import numpy as np


def _nonconformity_scores(posterior, y_true) -> np.ndarray:
    """计算二分类 nonconformity score: s(x,y)=1-p_y(x)。"""

    posterior = np.asarray(posterior, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    return np.where(y_true == 1, 1.0 - posterior, posterior)


class SplitConformalValidator:
    """基于当前 fold validation set 的 split conformal validator。

    该类只使用全局 posterior 输出，不训练局部模型，也不做按 bucket 校准。
    """

    def __init__(self, calibration_posterior, calibration_labels):
        self.calibration_scores = _nonconformity_scores(calibration_posterior, calibration_labels)
        self.n_calibration = int(len(self.calibration_scores))

    def p_value(self, candidate_score: float) -> float:
        if self.n_calibration <= 0:
            return 0.0
        count = int(np.sum(self.calibration_scores >= float(candidate_score)))
        return float((count + 1) / (self.n_calibration + 1))

    def predict_set(self, posterior: float, alpha_cp: float) -> dict:
        p = float(posterior)
        score_0 = p
        score_1 = 1.0 - p
        p_value_0 = self.p_value(score_0)
        p_value_1 = self.p_value(score_1)

        cp_set = []
        if p_value_0 > alpha_cp:
            cp_set.append(0)
        if p_value_1 > alpha_cp:
            cp_set.append(1)

        return {
            "cp_set": cp_set,
            "cp_p_value_0": p_value_0,
            "cp_p_value_1": p_value_1,
            "alpha_cp": float(alpha_cp),
        }


__all__ = ["SplitConformalValidator"]
