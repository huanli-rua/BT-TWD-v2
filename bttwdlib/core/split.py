"""数据划分工具，保持第一篇实验的随机种子和分层划分语义。"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_stratified_kfold(cfg: dict) -> StratifiedKFold:
    data_cfg = cfg.get("DATA", {})
    split_cfg = data_cfg.get("split", {})
    n_splits = data_cfg.get("n_splits", 5)
    test_ratio = split_cfg.get("test_ratio")
    if test_ratio is not None:
        n_from_ratio = int(round(1.0 / test_ratio)) if test_ratio > 0 else 0
        if n_from_ratio > 0 and np.isclose(test_ratio, 1.0 / n_from_ratio, rtol=1e-3):
            n_splits = n_from_ratio
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=data_cfg.get("shuffle", True),
        random_state=data_cfg.get("random_state", 42),
    )


def split_train_validation(X, y, bucket_df, cfg: dict):
    bcfg = cfg.get("BTTWD", {})
    split_cfg = cfg.get("DATA", {}).get("split", {})
    val_ratio = split_cfg.get("val_ratio", bcfg.get("val_ratio", 0.2))
    random_state = split_cfg.get("random_state", cfg.get("DATA", {}).get("random_state", 42))
    return train_test_split(
        X,
        y,
        bucket_df,
        test_size=val_ratio,
        stratify=y,
        random_state=random_state,
    )


__all__ = ["make_stratified_kfold", "split_train_validation", "train_test_split", "StratifiedKFold"]
