from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .metrics import evaluate_baseline_by_buckets
from .threshold_search import search_thresholds_with_regret
from .utils_logging import log_info


def _format_float(val: float | int | np.floating | None) -> str:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "nan"
        return f"{float(val):.4f}"
    except Exception:
        return str(val)


def _prepare_results_dir(results_dir: str | Path) -> Path:
    path = Path(results_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_baseline_bucket_evaluation(
    X,
    y,
    bucket_df_for_split,
    bucket_tree,
    cfg,
    results_dir,
    pre_split_data: dict | None = None,
    write_outputs: bool = True,
):
    if not _XGB_AVAILABLE:
        raise ImportError("运行基线桶级评估需要安装 xgboost，请先安装该依赖或关闭相关基线。")

    bttwd_cfg = cfg.get("BTTWD", {})
    data_cfg = cfg.get("DATA", {})
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})
    cost_cfg = threshold_cfg.get("costs", {})

    xgb_cfg = bttwd_cfg.get("global_xgb", {})

    test_ratio = data_cfg.get("test_size") or data_cfg.get("test_ratio") or 0.2
    val_ratio = bttwd_cfg.get("val_ratio", 0.1)
    random_state = data_cfg.get("random_state", 42)
    stratify_flag = data_cfg.get("stratify", True)
    if isinstance(stratify_flag, str):
        stratify_flag = stratify_flag.strip().lower() in {"1", "true", "yes", "y"}

    if pre_split_data is not None:
        X_train_full = pre_split_data["X_train"]
        y_train_full = pre_split_data["y_train"]
        bucket_train_full = pre_split_data["bucket_train"]
        X_test = pre_split_data["X_test"]
        y_test = pre_split_data["y_test"]
        bucket_test = pre_split_data["bucket_test"]
    else:
        # 切分数据集
        stratify_y = y if stratify_flag else None
        X_train_full, X_test, y_train_full, y_test, bucket_train_full, bucket_test = train_test_split(
            X,
            y,
            bucket_df_for_split,
            test_size=test_ratio,
            random_state=random_state,
            stratify=stratify_y,
        )
    stratify_y_train = y_train_full if stratify_flag else None
    X_train, X_val, y_train, y_val, bucket_train, bucket_val = train_test_split(
        X_train_full,
        y_train_full,
        bucket_train_full,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_y_train,
    )

    bucket_train = bucket_train.reset_index(drop=True)
    bucket_val = bucket_val.reset_index(drop=True)
    bucket_test = bucket_test.reset_index(drop=True)

    log_info("[BASELINE] 数据拆分完成（train/val/test）")

    # 训练全局 XGB 模型
    log_info("[BASELINE] 全局 XGB 模型开始训练")
    clf = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 300),
        max_depth=xgb_cfg.get("max_depth", 4),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
        random_state=xgb_cfg.get("random_state", 42),
        n_jobs=xgb_cfg.get("n_jobs", -1),
        eval_metric="logloss",
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train)
    log_info("[BASELINE] 全局 XGB 模型训练完成")

    # 搜索最优阈值
    log_info("[BASELINE] 阈值搜索开始")
    y_score_val = clf.predict_proba(X_val)[:, 1]
    alpha_grid = threshold_cfg.get("alpha_grid", [0.5])
    beta_grid = threshold_cfg.get("beta_grid", [0.0])
    gap_min = threshold_cfg.get("gap_min", 0.0)
    alpha, beta, stats = search_thresholds_with_regret(
        y_score_val, y_val, alpha_grid, beta_grid, cost_cfg, gap_min=gap_min
    )
    log_info(
        "[BASELINE] 最佳阈值找到: "
        f"alpha={_format_float(alpha)}, beta={_format_float(beta)}, regret={_format_float(stats.get('regret'))}"
    )

    # 测试集桶映射
    bucket_ids_test = bucket_tree.assign_buckets(bucket_test)
    log_info(f"[BASELINE] 测试集桶映射完成，共 {bucket_ids_test.nunique()} 个桶")

    # 桶级指标评估
    y_score_test = clf.predict_proba(X_test)[:, 1]
    bucket_metrics = evaluate_baseline_by_buckets(
        y_true=y_test,
        y_score=y_score_test,
        bucket_series=bucket_ids_test,
        alpha=alpha,
        beta=beta,
        cost_cfg=cost_cfg,
        include_parents=True,
    )

    for rec in bucket_metrics:
        log_info(
            "[BASELINE] 桶 {bid}: BAC={bac}, Regret={regret}, Precision={prec}, Recall={recall}".format(
                bid=rec.get("bucket_id"),
                bac=_format_float(rec.get("BAC")),
                regret=_format_float(rec.get("Regret")),
                prec=_format_float(rec.get("Precision")),
                recall=_format_float(rec.get("Recall")),
            )
        )

    results_path = _prepare_results_dir(results_dir) if write_outputs else Path(results_dir)
    baseline_df = pd.DataFrame(bucket_metrics)
    if baseline_df.empty:
        baseline_df = pd.DataFrame(
            columns=[
                "bucket_id",
                "Precision",
                "Recall",
                "F1",
                "BAC",
                "AUC",
                "MCC",
                "Kappa",
                "Regret",
                "BND_ratio",
                "POS_Coverage",
                "n_samples",
                "alpha",
                "beta",
            ]
        )
    if write_outputs:
        baseline_path = results_path / "baseline_bucket_metrics.csv"
        baseline_df.to_csv(baseline_path, index=False)
        log_info("[BASELINE] baseline_bucket_metrics.csv 写出完成")

    if write_outputs:
        # 合并到 bucket_metrics_gain.csv
        gain_path = results_path / "bucket_metrics_gain.csv"
        if gain_path.exists():
            gain_df = pd.read_csv(gain_path)
        else:
            gain_df = pd.DataFrame(columns=["bucket_id"])

        baseline_merge = baseline_df[
            [
                "bucket_id",
                "Precision",
                "Recall",
                "F1",
                "BAC",
                "AUC",
                "MCC",
                "Kappa",
                "Regret",
                "BND_ratio",
                "POS_Coverage",
            ]
        ].rename(
            columns={
                "Precision": "baseline_precision",
                "Recall": "baseline_recall",
                "F1": "baseline_f1",
                "BAC": "baseline_bac",
                "AUC": "baseline_auc",
                "MCC": "baseline_mcc",
                "Kappa": "baseline_kappa",
                "Regret": "baseline_regret",
                "BND_ratio": "baseline_bnd_ratio",
                "POS_Coverage": "baseline_pos_coverage",
            }
        )

        merged_df = gain_df.merge(baseline_merge, on="bucket_id", how="left")
        merged_df.to_csv(gain_path, index=False)
        log_info("[BASELINE] baseline 指标成功合并到 bucket_metrics_gain.csv")

    return bucket_metrics
