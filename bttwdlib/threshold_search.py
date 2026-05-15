import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_regret(y_true, y_pred_s3, costs: dict) -> float:
    """计算三支决策平均后悔值/风险成本。

    变量说明：
        POS: 接受域，编码为 1。
        NEG: 拒绝域，编码为 0。
        BND: 边界域/延迟决策，编码为 -1 或字符串 "BND"。
        regret: 样本在当前三支预测下对应成本的平均值，越小越好。
    """

    y_true = np.asarray(y_true)
    pred_arr = np.asarray(y_pred_s3)

    pred_numeric = np.where(pred_arr == "BND", -1, pred_arr)

    cost = np.zeros_like(y_true, dtype=float)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask

    cost[pos_mask & (pred_numeric == 1)] = costs.get("C_TP", 0.0)
    cost[pos_mask & (pred_numeric == 0)] = costs.get("C_FN", 0.0)
    cost[pos_mask & (pred_numeric == -1)] = costs.get("C_BP", 0.0)

    cost[neg_mask & (pred_numeric == 1)] = costs.get("C_FP", 0.0)
    cost[neg_mask & (pred_numeric == 0)] = costs.get("C_TN", 0.0)
    cost[neg_mask & (pred_numeric == -1)] = costs.get("C_BN", 0.0)

    if len(cost) == 0:
        return float("nan")
    return float(cost.mean())


def search_thresholds_with_regret(
    prob: np.ndarray,
    y_true: np.ndarray,
    alpha_grid,
    beta_grid,
    costs: dict,
    gap_min: float = 0.0,
    tol: float = 1e-12,
):
    """通过网格搜索得到局部最优阈值 alpha/beta。

    决策规则：
        p >= alpha -> POS（接受域）
        p <= beta  -> NEG（拒绝域）
        其他样本    -> BND（边界域）

    变量说明：
        opt/best: 当前网格中最优的 alpha/beta 组合。
        dval: validation decision value，可理解为验证集上的后验概率 prob。
        deff: effective decision，指最终由阈值产生的 POS/NEG/BND 决策。
        bnd_ratio: BND 样本比例。
        pos_coverage: POS 样本覆盖比例。
        regret: 成本矩阵下的平均后悔值，主优化目标。
    """

    best_alpha = None
    best_beta = None
    best_stats = None

    auc_val = float("nan")
    try:
        if np.unique(y_true).size >= 2:
            auc_val = float(roc_auc_score(y_true, prob))
    except Exception:
        auc_val = float("nan")

    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha < beta + gap_min:
                continue

            preds = np.where(prob >= alpha, 1, np.where(prob <= beta, 0, -1))
            regret_val = compute_regret(y_true, preds, costs)
            pred_binary = np.where(preds == 1, 1, 0)

            precision = precision_score(y_true, pred_binary, zero_division=0)
            recall = recall_score(y_true, pred_binary, zero_division=0)
            f1 = f1_score(y_true, pred_binary, zero_division=0)
            bnd_ratio = float(np.mean(preds == -1))
            pos_coverage = float(np.mean(preds == 1))

            pos_mask = y_true == 1
            neg_mask = ~pos_mask
            tp = np.sum((preds == 1) & pos_mask)
            tn = np.sum((preds == 0) & neg_mask)
            tpr = tp / pos_mask.sum() if pos_mask.sum() > 0 else np.nan
            tnr = tn / neg_mask.sum() if neg_mask.sum() > 0 else np.nan
            if np.isnan(tpr) and np.isnan(tnr):
                bac = np.nan
            elif np.isnan(tpr):
                bac = tnr / 2
            elif np.isnan(tnr):
                bac = tpr / 2
            else:
                bac = 0.5 * (tpr + tnr)

            stats = {
                "regret": regret_val,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "bac": float(bac) if not np.isnan(bac) else np.nan,
                "bnd_ratio": bnd_ratio,
                "pos_coverage": pos_coverage,
                "n_samples": int(len(prob)),
                "auc": auc_val,
            }

            if best_stats is None:
                best_alpha, best_beta, best_stats = alpha, beta, stats
                continue

            if regret_val + tol < best_stats["regret"]:
                best_alpha, best_beta, best_stats = alpha, beta, stats
            elif abs(regret_val - best_stats["regret"]) <= tol:
                if f1 > best_stats["f1"] + tol:
                    best_alpha, best_beta, best_stats = alpha, beta, stats
                elif abs(f1 - best_stats["f1"]) <= tol and bnd_ratio < best_stats["bnd_ratio"] - tol:
                    best_alpha, best_beta, best_stats = alpha, beta, stats

    if best_alpha is None:
        best_alpha = float(alpha_grid[0]) if len(alpha_grid) else 0.5
        best_beta = float(beta_grid[0]) if len(beta_grid) else 0.0
        best_stats = {
            "regret": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "bnd_ratio": float("nan"),
            "pos_coverage": float("nan"),
            "n_samples": 0,
        }

    return best_alpha, best_beta, best_stats

