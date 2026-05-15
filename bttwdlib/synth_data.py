"""强异质性合成数据集生成与加载入口。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils_logging import log_info


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _binary_search_bias(
    base_logit: np.ndarray,
    group_offsets: np.ndarray,
    eps: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """通过二分搜索找到满足目标正例率的全局偏置。"""

    low, high = -8.0, 8.0
    bias = 0.0
    for _ in range(max_iter):
        bias = (low + high) / 2
        prob = _sigmoid(base_logit + group_offsets + bias + eps)
        rate = prob.mean()
        if abs(rate - target_rate) < tol:
            break
        if rate < target_rate:
            low = bias
        else:
            high = bias
    return bias


def _calc_group_stats(df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for g, sub in df.groupby(group_col):
        stats[str(g)] = {
            "count": int(len(sub)),
            "positive_rate": float(sub[target_col].mean()),
        }
    return stats


def generate_synth_strong_v1(
    n: int = 200_000,
    seed: int | None = 42,
    hetero_scale: float = 1.0,
    n_groups: int = 4,
    n_x: int = 10,
    n_z: int = 5,
    eps_std: float = 0.2,
    intercepts: Tuple[float, ...] | List[float] = (-2.0, -0.5, 0.5, 2.0),
    target_rate: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    生成强异质性的二分类合成数据集。

    参数均使用中文日志描述，便于论文复现。
    """

    rng = np.random.default_rng(seed)
    if n_groups < 1:
        raise ValueError("n_groups 至少为 1")

    base_intercepts = np.array(intercepts, dtype=float)
    if n_groups > len(base_intercepts):
        # 若用户需要更多组别，使用线性插值扩展至指定组数
        base_intercepts = np.linspace(base_intercepts.min(), base_intercepts.max(), n_groups)
    else:
        base_intercepts = base_intercepts[:n_groups]

    group_labels = np.array([chr(ord("A") + i) for i in range(n_groups)], dtype=object)
    group_idx = rng.integers(0, n_groups, size=n)
    groups = group_labels[group_idx]
    scaled_intercepts = base_intercepts * hetero_scale
    group_offsets = np.take(scaled_intercepts, group_idx)

    x_features = rng.normal(0, 1, size=(n, n_x))
    z_features = rng.normal(0, 1, size=(n, n_z)) if n_z > 0 else None
    weights = rng.normal(0.0, 1.0, size=n_x)
    eps = rng.normal(0.0, eps_std, size=n)

    base_logit = x_features @ weights
    bias = _binary_search_bias(base_logit, group_offsets, eps, target_rate)
    logits = base_logit + group_offsets + bias + eps
    prob = _sigmoid(logits)
    expected_rate = float(prob.mean())
    y = rng.binomial(1, prob)

    data = {
        "target": y,
        "group": groups,
    }
    for i in range(n_x):
        data[f"x{i+1}"] = x_features[:, i]
    if n_z > 0 and z_features is not None:
        for j in range(n_z):
            data[f"z{j+1}"] = z_features[:, j]

    df = pd.DataFrame(data)
    group_stats = _calc_group_stats(df, "group", "target")
    pos_rate = float(df["target"].mean())

    meta = {
        "seed": seed,
        "n_samples": int(n),
        "n_groups": int(n_groups),
        "group_labels": group_labels.tolist(),
        "hetero_scale": float(hetero_scale),
        "base_intercepts": base_intercepts.tolist(),
        "scaled_intercepts": scaled_intercepts.tolist(),
        "n_x": int(n_x),
        "n_z": int(n_z),
        "eps_std": float(eps_std),
        "weights_mean": float(weights.mean()),
        "weights_std": float(weights.std()),
        "target_rate": pos_rate,
        "expected_rate": expected_rate,
        "target_rate_cfg": float(target_rate),
        "group_stats": group_stats,
        "bias": float(bias),
    }

    log_info(
        "【合成数据】生成完成："
        f"样本数={n}，分组={group_labels.tolist()}，全局正例率={pos_rate:.2%} (期望={expected_rate:.2%})，"
        "各组正例率如下"
    )
    for g in group_labels:
        stats = group_stats.get(g, {"count": 0, "positive_rate": float("nan")})
        log_info(
            f"组别 {g}: 样本数={stats['count']}，正例率={stats['positive_rate']:.2%}"
        )

    return df, meta


def _piecewise_values(x: np.ndarray, bins: List[float], values: List[float]) -> np.ndarray:
    if len(values) != len(bins) + 1:
        raise ValueError("values 数量应等于 bins+1")
    idx = np.digitize(x, bins, right=False)
    return np.take(values, idx)


def _binary_search_bias_general(logits_raw: np.ndarray, target_rate: float, tol: float = 1e-4, max_iter: int = 50) -> float:
    low, high = -8.0, 8.0
    bias = 0.0
    for _ in range(max_iter):
        bias = (low + high) / 2
        prob = _sigmoid(logits_raw + bias)
        rate = prob.mean()
        if abs(rate - target_rate) < tol:
            break
        if rate < target_rate:
            low = bias
        else:
            high = bias
    return bias


def generate_synth_strong_v2(
    n: int = 200_000,
    seed: int | None = 42,
    target_rate: float = 0.25,
    k1: float = 1.2,
    k2: float = 1.8,
    k_inter: float = 1.2,
    sigma_clean: float = 0.15,
    sigma_noisy: float = 0.75,
    flip_clean: float = 0.01,
    flip_noisy: float = 0.25,
    n_x: int = 10,
    n_z: int = 5,
    b_group: Tuple[float, ...] | List[float] = (-2.2, -1.0, 0.3, 1.6),
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """生成四层异质性极强的合成数据集（synth_strong_v2）。"""

    rng = np.random.default_rng(seed)
    leaf_rng = np.random.default_rng(None if seed is None else seed + 1024)
    if len(b_group) != 4:
        raise ValueError("b_group 必须提供 4 个截距，分别对应 A/B/C/D")

    group_labels = np.array(["A", "B", "C", "D"], dtype=object)
    group_idx = rng.integers(0, 4, size=n)
    groups = group_labels[group_idx]
    group_offsets = np.take(np.array(b_group, dtype=float), group_idx)
    flip_sign = np.where(np.isin(groups, ["A", "B"]), 1.0, -1.0)

    x_features = rng.normal(0, 1, size=(n, n_x))
    z_features = rng.normal(0, 1, size=(n, n_z)) if n_z > 0 else None
    x1, x2, x3, x4 = x_features[:, 0], x_features[:, 1], x_features[:, 2], x_features[:, 3]

    # 分桶规则（500 叶子：4*5*5*5）
    x1_bins = [-1.2, -0.4, 0.2, 0.9]
    x2_bins = [-1.0, -0.2, 0.4, 1.0]
    x3_bins = [-1.0, -0.1, 0.6, 1.2]

    x1_bucket = np.digitize(x1, x1_bins, right=False)
    x2_bucket = np.digitize(x2, x2_bins, right=False)
    x3_bucket = np.digitize(x3, x3_bins, right=False)

    x1_piece = _piecewise_values(x1, x1_bins, [-1.1, -0.5, 0.1, 0.6, 1.1])
    x2_piece = _piecewise_values(x2, x2_bins, [-1.0, -0.3, 0.1, 0.7, 1.2])
    x3_piece = _piecewise_values(x3, x3_bins, [-0.6, -0.2, 0.2, 0.7, 1.0])

    # 噪声口袋：最高桶 + x3>0.8 时视作 noisy
    is_noisy = (x3_bucket == 4) | (x3 > 0.8)

    # 叶子桶 ID 与类型（P/N/B）分配
    B1 = B2 = B3 = 5
    leaf_id = (((group_idx * B1) + x1_bucket) * B2 + x2_bucket) * B3 + x3_bucket
    max_leaf_id = 4 * B1 * B2 * B3
    leaf_random = leaf_rng.random(max_leaf_id)
    bucket_type_codes = np.empty(max_leaf_id, dtype="U1")
    bucket_type_codes[leaf_random < 0.35] = "P"
    bucket_type_codes[(leaf_random >= 0.35) & (leaf_random < 0.70)] = "N"
    bucket_type_codes[leaf_random >= 0.70] = "B"
    bucket_type = bucket_type_codes[leaf_id]

    # 桶级偏置，强化 P/N 方向差异
    leaf_jitter = leaf_rng.normal(0.0, 0.1, size=max_leaf_id)
    delta_P, delta_N = 0.8, 0.8
    leaf_offsets = np.take(leaf_jitter, leaf_id)
    leaf_offsets[bucket_type == "P"] += delta_P
    leaf_offsets[bucket_type == "N"] -= delta_N

    weights = rng.normal(0.0, 1.0, size=n_x)
    if n_x > 4:
        base_logit = x_features[:, 4:] @ weights[4:]
    else:
        base_logit = np.zeros(n)

    x2_flip = flip_sign * x2_piece * k2
    inter = flip_sign * (x1 * x2) * k_inter
    base_logit_components = base_logit + group_offsets + x1_piece * k1 + x2_flip + inter + x3_piece * 0.8 + leaf_offsets

    base_sigma = np.where(is_noisy, sigma_noisy, sigma_clean)
    eps_base = rng.normal(0.0, base_sigma, size=n)
    logits_raw_initial = base_logit_components + eps_base
    bias_initial = _binary_search_bias_general(logits_raw_initial, target_rate)
    prob_initial = _sigmoid(logits_raw_initial + bias_initial)
    mid_mask_initial = (prob_initial > 0.35) & (prob_initial < 0.65)

    sigma_mid_B = 0.9
    sigma_final = base_sigma.copy()
    sigma_final[(bucket_type == "B") & mid_mask_initial] = sigma_mid_B

    eps_final = rng.normal(0.0, sigma_final, size=n)
    logits_raw = base_logit_components + eps_final
    bias = _binary_search_bias_general(logits_raw, target_rate)
    prob = _sigmoid(logits_raw + bias)
    mid_mask = (prob > 0.35) & (prob < 0.65)
    target_rate_real = float(prob.mean())
    y = rng.binomial(1, prob)

    flip_prob = np.full(n, flip_clean, dtype=float)
    # B 桶：中段高翻转，其他低翻转
    flip_prob[(bucket_type == "B") & mid_mask] = max(flip_noisy, 0.35)
    flip_prob[(bucket_type == "B") & (~mid_mask)] = min(flip_clean + 0.02, 0.05)
    # P/N 桶：保持较低翻转，但允许轻微差异
    flip_prob[(bucket_type == "P") & (~mid_mask)] = min(flip_clean + 0.005, 0.03)
    flip_prob[(bucket_type == "N") & (~mid_mask)] = min(flip_clean + 0.015, 0.05)
    flips = rng.binomial(1, flip_prob)
    y = np.where(flips == 1, 1 - y, y)

    data = {"target": y, "group": groups}
    for i in range(n_x):
        data[f"x{i+1}"] = x_features[:, i]
    if n_z > 0 and z_features is not None:
        for j in range(n_z):
            data[f"z{j+1}"] = z_features[:, j]

    df = pd.DataFrame(data)
    df["group"] = df["group"].astype(str)

    group_stats = _calc_group_stats(df, "group", "target")
    noisy_rate = float(is_noisy.mean())

    bucket_stats = {}
    non_empty_leaves = 0
    for lid in range(max_leaf_id):
        mask = leaf_id == lid
        if not np.any(mask):
            continue
        non_empty_leaves += 1
        lid_mid = mid_mask[mask]
        lid_flip = flips[mask]
        mid_rate = float(lid_mid.mean()) if lid_mid.size > 0 else 0.0
        flip_mid_rate = float(lid_flip[lid_mid].mean()) if lid_mid.any() else 0.0
        bucket_stats[str(lid)] = {
            "count": int(mask.sum()),
            "positive_rate": float(df.loc[mask, "target"].mean()),
            "bucket_type": str(bucket_type_codes[lid]),
            "mid_rate": mid_rate,
            "flip_mid_rate": flip_mid_rate,
        }

    bucket_type_counts = {
        "P": int(np.sum(bucket_type_codes == "P")),
        "N": int(np.sum(bucket_type_codes == "N")),
        "B": int(np.sum(bucket_type_codes == "B")),
    }
    bucket_type_ratio = {
        k: float(v / max_leaf_id) for k, v in bucket_type_counts.items()
    }
    mid_coverage_rate = float(mid_mask.mean())
    flip_mid_rate_mean = float(flips[mid_mask].mean()) if mid_mask.any() else 0.0

    leaf_cost_profile = {}
    bucket_path_cost_profile = {}
    x1_labels = ["b1", "b2", "b3", "b4", "b5"]
    x2_labels = ["b1", "b2", "b3", "b4", "b5"]
    x3_labels = ["b1", "b2", "b3", "b4", "b5"]
    base_costs = {"C_TP": 0.0, "C_TN": 0.0, "C_FP": 1.0, "C_FN": 3.0, "C_BP": 1.5, "C_BN": 0.5}
    for lid in range(max_leaf_id):
        btype = bucket_type_codes[lid]
        jitter = float(leaf_rng.normal(0.0, 0.05))
        if btype == "P":
            costs = {
                **base_costs,
                "C_FP": max(0.2, base_costs["C_FP"] * 0.8 + jitter),
                "C_FN": base_costs["C_FN"] + 1.5 + jitter,
                "C_BP": max(0.2, base_costs["C_BP"] * 0.9 + jitter),
                "C_BN": max(0.1, base_costs["C_BN"] * 0.8 + jitter),
            }
        elif btype == "N":
            costs = {
                **base_costs,
                "C_FP": base_costs["C_FP"] + 1.6 + jitter,
                "C_FN": max(0.5, base_costs["C_FN"] * 0.8 + jitter),
                "C_BP": max(0.3, base_costs["C_BP"] * 1.1 + jitter),
                "C_BN": max(0.2, base_costs["C_BN"] * 1.1 + jitter),
            }
        else:  # B bucket
            costs = {
                **base_costs,
                "C_FP": base_costs["C_FP"] + 0.4 + jitter,
                "C_FN": base_costs["C_FN"] + 0.3 + jitter,
                "C_BP": max(0.05, base_costs["C_BP"] * 0.3 + jitter),
                "C_BN": max(0.05, base_costs["C_BN"] * 0.3 + jitter),
            }
        leaf_cost_profile[str(lid)] = {k: float(v) for k, v in costs.items()} | {"bucket_type": btype}

        g_idx = lid // (B1 * B2 * B3)
        rem = lid % (B1 * B2 * B3)
        b1_idx = rem // (B2 * B3)
        rem = rem % (B2 * B3)
        b2_idx = rem // B3
        b3_idx = rem % B3
        bucket_path = (
            f"L1_group={group_labels[g_idx]}|"
            f"L2_x1={x1_labels[b1_idx]}|"
            f"L3_x2={x2_labels[b2_idx]}|"
            f"L4_x3={x3_labels[b3_idx]}"
        )
        bucket_path_cost_profile[bucket_path] = {k: float(v) for k, v in costs.items()} | {
            "bucket_type": btype,
            "leaf_id": int(lid),
        }

    meta = {
        "version": "v2",
        "seed": seed,
        "n_samples": int(n),
        "target_rate_cfg": float(target_rate),
        "target_rate_real": target_rate_real,
        "n_x": int(n_x),
        "n_z": int(n_z),
        "k1": float(k1),
        "k2": float(k2),
        "k_inter": float(k_inter),
        "sigma_clean": float(sigma_clean),
        "sigma_noisy": float(sigma_noisy),
        "flip_clean": float(flip_clean),
        "flip_noisy": float(flip_noisy),
        "b_group": list(map(float, b_group)),
        "weights_mean": float(weights.mean()),
        "weights_std": float(weights.std()),
        "noisy_pocket_rate_real": noisy_rate,
        "group_stats": group_stats,
        "leaf_stats": bucket_stats,
        "bucket_type_ratio": bucket_type_ratio,
        "bucket_type_counts": bucket_type_counts,
        "non_empty_leaf_count": int(non_empty_leaves),
        "mid_coverage_rate": mid_coverage_rate,
        "flip_mid_rate_mean": flip_mid_rate_mean,
        "leaf_cost_profile": leaf_cost_profile,
        "bucket_path_cost_profile": bucket_path_cost_profile,
        "bucket_rules": {
            "L1_group": {
                "labels": group_labels.tolist(),
            },
            "L2_x1": {
                "bins": x1_bins,
                "labels": ["b1", "b2", "b3", "b4", "b5"],
            },
            "L3_x2": {
                "bins": x2_bins,
                "labels": ["b1", "b2", "b3", "b4", "b5"],
                "direction_flip": {
                    "positive": ["A", "B"],
                    "negative": ["C", "D"],
                },
            },
            "L4_x3": {
                "bins": x3_bins,
                "labels": ["b1", "b2", "b3", "b4", "b5"],
                "noisy_definition": "bucket==b5 or x3>0.8",
            },
        },
        "bias": float(bias),
    }

    log_info(
        "【合成数据v2】生成完成："
        + f"样本数={n}，全局正例率={target_rate_real:.2%} (目标={target_rate:.2%})，noisy比例={noisy_rate:.2%}"
        + f"，非空叶子数={non_empty_leaves}，桶类型占比 P/N/B={bucket_type_ratio['P']:.2%}/{bucket_type_ratio['N']:.2%}/{bucket_type_ratio['B']:.2%}"
    )
    for g in group_labels:
        stats = group_stats.get(str(g), {"count": 0, "positive_rate": float('nan')})
        log_info(
            f"组别 {g}: 样本数={stats['count']}，正例率={stats['positive_rate']:.2%}"
        )

    return df, meta


def save_synth_strong_v2(df: pd.DataFrame, meta: Dict[str, object], out_path: str, meta_path: str) -> None:
    out_file = Path(out_path)
    meta_file = Path(meta_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_file, index=False)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_info(f"【合成数据v2】数据已保存至 {out_file}，元数据写入 {meta_file}")


def load_synth_strong_v2(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target" not in df.columns or "group" not in df.columns:
        raise KeyError("合成数据缺少 target 或 group 列，无法加载")
    pos_rate = df["target"].mean()
    log_info(
        f"【合成数据v2加载】文件={path}，样本数={len(df)}，全局正例率={pos_rate:.2%}"
    )
    for g, sub in df.groupby("group"):
        log_info(f"组别 {g}: 样本数={len(sub)}，正例率={sub['target'].mean():.2%}")
    return df


def save_synth_strong_v1(df: pd.DataFrame, meta: Dict[str, object], out_path: str, meta_path: str) -> None:
    out_file = Path(out_path)
    meta_file = Path(meta_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_file, index=False)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_info(
        f"【合成数据】数据已保存至 {out_file}，元数据写入 {meta_file}"
    )


def load_synth_strong_v1(path: str | Path) -> pd.DataFrame:
    """读取强异质性合成数据集，并打印基础统计。"""

    df = pd.read_csv(path)
    if "target" not in df.columns or "group" not in df.columns:
        raise KeyError("合成数据缺少 target 或 group 列，无法加载")
    pos_rate = df["target"].mean()
    log_info(
        f"【合成数据加载】文件={path}，样本数={len(df)}，全局正例率={pos_rate:.2%}"
    )
    for g, sub in df.groupby("group"):
        log_info(f"组别 {g}: 样本数={len(sub)}，正例率={sub['target'].mean():.2%}")
    return df


__all__ = [
    "generate_synth_strong_v1",
    "save_synth_strong_v1",
    "load_synth_strong_v1",
    "generate_synth_strong_v2",
    "save_synth_strong_v2",
    "load_synth_strong_v2",
]
