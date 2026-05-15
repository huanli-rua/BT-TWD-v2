import os
import time
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False
from .bttwd_model import BTTWDModel
from .baselines import (
    get_decision_threshold,
    train_eval_knn,
    train_eval_logreg,
    train_eval_random_forest,
    train_eval_xgboost,
)
from .baseline_analyzer import run_baseline_bucket_evaluation
from .bucket_rules import BucketTree
from .metrics import (
    compute_binary_metrics,
    compute_s3_metrics,
    log_metrics,
    predict_binary_by_cost,
)
from .threshold_search import compute_regret
from .utils_logging import log_info

try:
    import psutil
except ImportError as exc:  # pragma: no cover - 资源评估需 psutil
    raise ImportError("需要安装 psutil 以进行训练耗时和内存评估，请先安装 psutil") from exc


def _select_baselines(cfg: dict) -> set[str]:
    """根据配置确定需要运行的基线模型集合。"""

    model_cfg = cfg.get("MODEL", {})
    baseline_list = model_cfg.get("baselines") or []
    if isinstance(baseline_list, (list, tuple)) and baseline_list:
        return {str(x).lower() for x in baseline_list}

    base_cfg = cfg.get("BASELINES", {})
    selected = set()
    if base_cfg.get("use_logreg"):
        selected.add("logreg")
    if base_cfg.get("use_random_forest"):
        selected.add("random_forest")
    if base_cfg.get("use_knn"):
        selected.add("knn")
    if base_cfg.get("use_xgboost"):
        selected.add("xgb")
    return selected


def _format_threshold_value(alpha, beta) -> str:
    """Format (alpha, beta) as a readable threshold string."""

    try:
        if alpha is None or beta is None:
            return "nan"
        if pd.isna(alpha) or pd.isna(beta):
            return "nan"
        return f"alpha={float(alpha):.6f},beta={float(beta):.6f}"
    except Exception:
        return f"{alpha},{beta}"


def _measure_training_resources(train_callable):
    """评估单次训练的耗时与内存峰值。"""

    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    result = train_callable()
    elapsed = time.perf_counter() - start_time

    mem_candidates = []
    try:
        info = process.memory_info()
        mem_candidates.extend(
            [
                info.rss,
                getattr(info, "peak_wset", 0),
                getattr(info, "peak_rss", 0),
                getattr(info, "vms", 0),
            ]
        )
    except psutil.Error:
        pass
    try:
        full_info = process.memory_full_info()
        for attr in ("rss", "uss", "peak_wset", "peak_rss"):
            mem_candidates.append(getattr(full_info, attr, 0))
    except psutil.Error:
        pass

    peak_bytes = max([m for m in mem_candidates if m], default=0)
    return elapsed, peak_bytes / (1024 * 1024), "psutil", result


def _log_training_resources(cfg: dict, elapsed: float, mem_mb: float, backend: str, context: str) -> None:
    """将资源评估结果用中文输出，关注当前配置。"""

    bcfg = cfg.get("BTTWD", {})
    parent_share = "开启" if bcfg.get("use_parent_share_rate", True) else "关闭"
    min_sample_limit = "开启" if bcfg.get("use_min_bucket_size_limit", True) else "关闭"
    fallback = "开启" if bcfg.get("use_gain_weak_backoff", True) else "关闭"
    backend_note = "" if backend == "psutil" else "（psutil 未安装，使用其他方法估算）"
    log_info(
        f"【资源评估-{context}】当前配置：父桶贡献={parent_share}，最小样本限制={min_sample_limit}，"
        f"回退机制={fallback}；训练耗时={elapsed:.3f} 秒，最大内存占用≈{mem_mb:.2f} MB{backend_note}"
    )


def _summarize_test_buckets(bucket_ids: pd.Series, y_true: np.ndarray, y_pred_s3: np.ndarray, costs: dict) -> pd.DataFrame:
    """Aggregate test-set statistics per bucket."""

    records = []
    for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
        idx_list = list(idxs)
        y_true_bucket = y_true[idx_list]
        y_pred_bucket = y_pred_s3[idx_list]
        records.append(
            {
                "bucket_id": bucket_id,
                "n_test": len(idx_list),
                "pos_rate_test": float(np.mean(y_true_bucket)) if len(idx_list) else np.nan,
                "BND_ratio_test": float(np.mean(np.isin(y_pred_bucket, [-1, "BND"]))),
                "POS_Coverage_test": float(np.mean(np.array(y_pred_bucket) == 1)),
                "regret_test": compute_regret(y_true_bucket, y_pred_bucket, costs),
            }
        )

    return pd.DataFrame(records)


def _build_baseline_estimator(model_key: str, cfg: dict):
    base_cfg = cfg.get("BASELINES", {})
    if model_key == "logreg":
        model_cfg = base_cfg.get("logreg", {})
        return LogisticRegression(max_iter=model_cfg.get("max_iter", 200), C=model_cfg.get("C", 1.0))
    if model_key == "random_forest":
        rf_cfg = base_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth"),
            random_state=rf_cfg.get("random_state", 42),
            n_jobs=cfg.get("EXP", {}).get("n_jobs", -1),
        )
    if model_key == "knn":
        knn_cfg = base_cfg.get("knn", {})
        return KNeighborsClassifier(n_neighbors=knn_cfg.get("n_neighbors", 10))
    if model_key in {"xgb", "xgboost"}:
        if not _XGB_AVAILABLE:
            raise RuntimeError("配置了 XGBoost 基线但未安装 xgboost，请先安装。")
        xgb_cfg = base_cfg.get("xgboost", {})
        return XGBClassifier(
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
    raise ValueError(f"未知的基线模型类型 {model_key}")


def _eval_baseline_holdout(model_key: str, X_train, y_train, X_test, y_test, cfg, costs: dict | None = None) -> dict:
    clf = _build_baseline_estimator(model_key, cfg)
    threshold, mode, used_custom = get_decision_threshold(model_key if model_key != "xgb" else "xgboost", cfg)
    log_info(
        f"【基线-{model_key}】使用决策阈值={threshold:.3f}，模式={mode}，"
        f"自定义阈值={'是' if used_custom else '否'}，开始在测试集评估"
    )
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
        y_score = np.zeros_like(y_pred, dtype=float)

    metrics_cfg = cfg.get("METRICS", {})
    metrics_dict = compute_binary_metrics(y_test, y_pred, y_score, metrics_cfg, costs=costs)
    metrics_dict.setdefault("BND_ratio", 0.0)
    metrics_dict.setdefault("POS_Coverage", float("nan"))
    metrics_dict["model"] = model_key
    return metrics_dict

def run_holdout_experiment(X, y, bucket_df, cfg, bucket_cols=None, bucket_tree: BucketTree | None = None, results_dir=None):
    """训练/评估单次切分的 BTTWD 模型并返回指标。"""

    repo_root = Path(__file__).resolve().parent.parent
    if results_dir is None:
        configured_results_dir = cfg.get("OUTPUT", {}).get("results_dir", "results")
        results_dir = Path(configured_results_dir)
        if not results_dir.is_absolute():
            results_dir = repo_root / results_dir

    split_cfg = cfg.get("DATA", {}).get("split", {})
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.2)
    random_state = split_cfg.get("random_state", 42)
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp, bucket_train, bucket_temp = train_test_split(
        X,
        y,
        bucket_df,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test, bucket_val, bucket_test = train_test_split(
        X_temp,
        y_temp,
        bucket_temp,
        test_size=test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0,
        stratify=y_temp,
        random_state=random_state,
    )

    # 重置分桶特征的索引，使其与对应的 X/y 数组位置对齐，避免后续按 index 访问概率时越界
    bucket_train = bucket_train.reset_index(drop=True)
    bucket_val = bucket_val.reset_index(drop=True)
    bucket_test = bucket_test.reset_index(drop=True)

    # 开发阶段的安全检查，确保长度一致
    assert len(X_train) == len(bucket_train) == len(y_train)
    assert len(X_val) == len(bucket_val) == len(y_val)
    assert len(X_test) == len(bucket_test) == len(y_test)

    log_info(
        "【数据切分】训练/验证/测试样本数 = "
        f"{len(X_train)}/{len(X_val)}/{len(X_test)}，训练正类占比={y_train.mean():.2%}"
    )

    bucket_cols = bucket_cols or bucket_df.columns.tolist()
    model = BTTWDModel.from_cfg(cfg, feature_names=bucket_cols)
    elapsed, mem_mb, backend, _ = _measure_training_resources(lambda: model.fit(X_train, y_train, bucket_train))
    _log_training_resources(cfg, elapsed, mem_mb, backend, context="Holdout")

    y_score = model.predict_proba(X_test, bucket_test)
    y_pred_s3 = model.predict(X_test, bucket_test)

    costs = (cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})).get("costs", {})
    y_pred_binary = predict_binary_by_cost(y_score, costs) if costs else np.where(y_pred_s3 == 1, 1, 0)

    metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}), costs=costs)
    metrics_binary = compute_binary_metrics(y_test, y_pred_binary, y_score, cfg.get("METRICS", {}), costs=costs)

    log_info("【测试集指标-S3】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_s3.items()]))
    log_info("【测试集指标-二分类】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_binary.items()]))

    test_bucket_ids = model.bucket_tree.assign_buckets(bucket_test)
    test_bucket_df = _summarize_test_buckets(test_bucket_ids, y_test, y_pred_s3, costs)
    if not test_bucket_df.empty:
        model.update_test_stats(test_bucket_df)
        model._export_bucket_reports()

    run_baseline_bucket_evaluation(
        X=X,
        y=y,
        bucket_df_for_split=bucket_df,
        bucket_tree=model.bucket_tree,
        cfg=cfg,
        results_dir=results_dir,
    )

    return {"metrics_s3": metrics_s3, "metrics_binary": metrics_binary}


def run_kfold_experiments(X, y, X_df_for_bucket, cfg, test_data=None, bucket_tree: BucketTree | None = None) -> dict:
    cfg = deepcopy(cfg)
    repo_root = Path(__file__).resolve().parent.parent
    configured_results_dir = cfg.get("OUTPUT", {}).get("results_dir", "results")
    results_dir = Path(configured_results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    os.makedirs(results_dir, exist_ok=True)
    cfg.setdefault("OUTPUT", {})["export_bucket_reports_on_fit"] = False

    split_cfg = cfg.get("DATA", {}).get("split", {})
    val_ratio_override = split_cfg.get("val_ratio")
    test_ratio_cfg = split_cfg.get("test_ratio")

    if val_ratio_override is not None:
        bttwd_cfg = cfg.setdefault("BTTWD", {})
        bttwd_cfg["val_ratio"] = val_ratio_override

    data_cfg = cfg.get("DATA", {})
    n_splits = data_cfg.get("n_splits", 5)
    if test_ratio_cfg is not None:
        n_splits_from_ratio = int(round(1.0 / test_ratio_cfg)) if test_ratio_cfg > 0 else 0
        if n_splits_from_ratio <= 0 or not np.isclose(test_ratio_cfg, 1.0 / n_splits_from_ratio, rtol=1e-3):
            raise ValueError("DATA.split.test_ratio 必须是 1/k 的形式，便于对齐 K 折测试集比例")
        n_splits = n_splits_from_ratio

    shuffle = data_cfg.get("shuffle", True)
    random_state = data_cfg.get("random_state", 42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    per_fold_records = []
    bucket_metrics_records = []
    threshold_log_records = []
    tree_structure_path = Path(results_dir) / "bucket_tree_structure.csv"
    if tree_structure_path.exists():
        tree_structure_path.unlink()
    threshold_costs = (cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})).get("costs", {})
    bucket_test_gain_records = []
    per_sample_test_records = []

    # 运行基线整体（使用 cross_val_predict）
    baseline_results = {}
    baseline_holdout_results = {}
    test_holdout_records: list[dict] = []
    baseline_set = _select_baselines(cfg)
    if "logreg" in baseline_set:
        baseline_results["LogReg"] = train_eval_logreg(X, y, cfg, skf, costs=threshold_costs)
    if "random_forest" in baseline_set:
        baseline_results["RandomForest"] = train_eval_random_forest(X, y, cfg, skf, costs=threshold_costs)
    if "knn" in baseline_set:
        baseline_results["KNN"] = train_eval_knn(X, y, cfg, skf, costs=threshold_costs)
    if "xgb" in baseline_set or "xgboost" in baseline_set:
        baseline_results["XGBoost"] = train_eval_xgboost(X, y, cfg, skf, costs=threshold_costs)

    fold_idx = 1
    model: BTTWDModel | None = None
    for train_idx, test_idx in skf.split(X, y):
        log_info(f"【K折实验】正在执行第 {fold_idx}/{n_splits} 折...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 重新编号训练/测试数据的索引，确保与 X_train 对齐
        X_df_train = X_df_for_bucket.iloc[train_idx].reset_index(drop=True)
        X_df_test = X_df_for_bucket.iloc[test_idx].reset_index(drop=True)

        bttwd_model = BTTWDModel.from_cfg(cfg, feature_names=X_df_for_bucket.columns.tolist())
        elapsed, mem_mb, backend, _ = _measure_training_resources(lambda: bttwd_model.fit(X_train, y_train, X_df_train))
        _log_training_resources(cfg, elapsed, mem_mb, backend, context=f"K折第{fold_idx}折")

        model = bttwd_model

        y_score = bttwd_model.predict_proba(X_test, X_df_test)
        y_pred_s3 = bttwd_model.predict(X_test, X_df_test)
        if threshold_costs:
            y_pred_binary = predict_binary_by_cost(y_score, threshold_costs)
        else:
            y_pred_binary = np.where(y_pred_s3 == 1, 1, 0)

        metrics_binary = compute_binary_metrics(
            y_test, y_pred_binary, y_score, cfg.get("METRICS", {}), costs=threshold_costs or None
        )
        metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}), costs=threshold_costs)
        log_metrics("【BTTWD】三支指标(含后悔)：", metrics_s3)

        fold_record = {"fold": fold_idx, "model": "BTTWD", **metrics_s3}
        for k, v in metrics_binary.items():
            if k not in fold_record:
                fold_record[k] = v
        per_fold_records.append(fold_record)
        bucket_df = bttwd_model.get_bucket_stats()
        if not bucket_df.empty:
            test_bucket_parts = bttwd_model.bucket_tree.assign_bucket_parts(X_df_test)
            test_bucket_ids = bttwd_model._route_bucket_ids(test_bucket_parts)
            bucket_meta = bucket_df.set_index("bucket_id").to_dict("index")
            bucket_groups = test_bucket_ids.groupby(test_bucket_ids).groups
            metrics_cfg = deepcopy(cfg.get("METRICS", {}))
            default_metrics = ["Precision", "Recall", "F1", "BAC", "AUC", "MCC", "Kappa"]
            merged_metrics = list(dict.fromkeys((metrics_cfg.get("use_metrics") or []) + default_metrics))
            metrics_cfg["use_metrics"] = merged_metrics

            baseline_metrics = run_baseline_bucket_evaluation(
                X=X_train,
                y=y_train,
                bucket_df_for_split=X_df_train,
                bucket_tree=bttwd_model.bucket_tree,
                cfg=cfg,
                results_dir=results_dir,
                pre_split_data={
                    "X_train": X_train,
                    "y_train": y_train,
                    "bucket_train": X_df_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "bucket_test": X_df_test,
                },
                write_outputs=False,
            )
            baseline_map = {rec.get("bucket_id"): rec for rec in baseline_metrics}

            for bucket_id, meta in bucket_meta.items():
                idx_list = list(bucket_groups.get(bucket_id, []))
                if idx_list:
                    y_true_bucket = y_test[idx_list]
                    y_score_bucket = y_score[idx_list]
                    y_pred_bucket = y_pred_s3[idx_list]
                    s3_metrics = compute_s3_metrics(
                        y_true_bucket, y_pred_bucket, y_score_bucket, metrics_cfg, costs=threshold_costs
                    )
                    regret_val = compute_regret(y_true_bucket, y_pred_bucket, threshold_costs)
                else:
                    s3_metrics = {}
                    regret_val = float("nan")

                baseline_rec = baseline_map.get(bucket_id, {})
                bucket_test_gain_records.append(
                    {
                        "fold": fold_idx,
                        "bucket_id": bucket_id,
                        "parent_id": meta.get("parent_bucket_id", ""),
                        "level": 0 if bucket_id == "ROOT" else len(str(bucket_id).split("|")),
                        "n_train": meta.get("n_train", 0),
                        "n_val": meta.get("n_val", 0),
                        "n_test": len(idx_list),
                        "BAC": s3_metrics.get("BAC"),
                        "F1": s3_metrics.get("F1"),
                        "AUC": s3_metrics.get("AUC"),
                        "Regret": regret_val,
                        "BND_ratio": s3_metrics.get("BND_ratio"),
                        "POS_coverage": s3_metrics.get("POS_Coverage"),
                        "is_leaf": len(bttwd_model.children_map.get(bucket_id, [])) == 0,
                        "is_weak": meta.get("is_weak", False),
                        "threshold_source_bucket": meta.get("threshold_source_bucket")
                        or meta.get("parent_with_threshold")
                        or bucket_id,
                        "threshold_used": _format_threshold_value(meta.get("alpha"), meta.get("beta")),
                        "baseline_precision": baseline_rec.get("Precision"),
                        "baseline_recall": baseline_rec.get("Recall"),
                        "baseline_f1": baseline_rec.get("F1"),
                        "baseline_bac": baseline_rec.get("BAC"),
                        "baseline_auc": baseline_rec.get("AUC"),
                        "baseline_mcc": baseline_rec.get("MCC"),
                        "baseline_kappa": baseline_rec.get("Kappa"),
                        "baseline_regret": baseline_rec.get("Regret"),
                        "baseline_bnd_ratio": baseline_rec.get("BND_ratio"),
                        "baseline_pos_coverage": baseline_rec.get("POS_Coverage"),
                    }
                )

            test_bucket_df = _summarize_test_buckets(test_bucket_ids, y_test, y_pred_s3, threshold_costs)
            if not test_bucket_df.empty:
                bttwd_model.update_test_stats(test_bucket_df)
                bucket_df = bucket_df.merge(test_bucket_df, on="bucket_id", how="left")
            else:
                bucket_df["n_test"] = np.nan
                bucket_df["pos_rate_test"] = np.nan
                bucket_df["BND_ratio_test"] = np.nan
                bucket_df["POS_Coverage_test"] = np.nan
                bucket_df["regret_test"] = np.nan

            bttwd_model._export_bucket_reports(fold=fold_idx, append_tree=True)
            bucket_df["fold"] = fold_idx
            bucket_metrics_records.append(bucket_df)

            # 逐样本预测记录：包含桶ID、阈值来源与预测结果，便于对齐排查。
            threshold_cache = {bid: bttwd_model._get_threshold_with_backoff(bid) for bid in bucket_meta.keys()}
            for local_idx, bucket_id in enumerate(test_bucket_ids.tolist()):
                alpha_beta, threshold_source = threshold_cache.get(
                    bucket_id, ((float("nan"), float("nan")), "ROOT")
                )
                global_idx = int(test_idx[local_idx])
                per_sample_test_records.append(
                    {
                        "fold": fold_idx,
                        "global_index": global_idx,
                        "bucket_id": bucket_id,
                        "threshold_source_bucket": threshold_source,
                        "alpha_used": float(alpha_beta[0]) if alpha_beta else float("nan"),
                        "beta_used": float(alpha_beta[1]) if alpha_beta else float("nan"),
                        "y_true": int(y_test[local_idx]),
                        "y_score": float(y_score[local_idx]),
                        "y_pred_s3": int(y_pred_s3[local_idx]),
                        "y_pred_binary": int(y_pred_binary[local_idx]),
                    }
                )

        th_logs = bttwd_model.get_threshold_logs()
        if not th_logs.empty:
            th_logs["fold"] = fold_idx
            threshold_log_records.append(th_logs)

        fold_idx += 1

    # 汇总 BTTWD 平均指标
    bttwd_df = pd.DataFrame(per_fold_records)
    summary_rows = []
    if not bttwd_df.empty:
        metric_cols = [c for c in bttwd_df.columns if c not in ["fold", "model"]]
        mean_series = bttwd_df[metric_cols].mean()
        std_series = bttwd_df[metric_cols].std()
        bttwd_summary = {"model": "BTTWD"}
        for col in metric_cols:
            bttwd_summary[f"{col}_mean"] = mean_series[col]
            bttwd_summary[f"{col}_std"] = std_series[col]
        summary_rows.append(bttwd_summary)

    for model_name, res in baseline_results.items():
        if res["summary"]:
            row = {"model": model_name}
            row.update(res["summary"])
            summary_rows.append(row)

        if res.get("per_fold"):
            for rec in res["per_fold"]:
                per_fold_records.append({"model": model_name, **rec})

    if test_data is not None:
        X_test, y_test, bucket_df_test = test_data
        bucket_df_test = bucket_df_test.reset_index(drop=True)
        log_info(
            f"【Holdout】检测到外部测试集，训练集 n={len(X)}, 测试集 n={len(X_test)}，开始全量训练后评估"
        )
        bttwd_final = BTTWDModel.from_cfg(cfg, feature_names=X_df_for_bucket.columns.tolist())
        elapsed, mem_mb, backend, _ = _measure_training_resources(
            lambda: bttwd_final.fit(X, y, X_df_for_bucket.reset_index(drop=True))
        )
        _log_training_resources(cfg, elapsed, mem_mb, backend, context="Holdout-外部测试集")
        y_score_final = bttwd_final.predict_proba(X_test, bucket_df_test)
        y_pred_final = bttwd_final.predict(X_test, bucket_df_test)
        if threshold_costs:
            y_pred_binary_final = predict_binary_by_cost(y_score_final, threshold_costs)
        else:
            y_pred_binary_final = np.where(y_pred_final == 1, 1, 0)

        metrics_s3_test = compute_s3_metrics(
            y_test, y_pred_final, y_score_final, cfg.get("METRICS", {}), costs=threshold_costs
        )
        metrics_binary_test = compute_binary_metrics(
            y_test, y_pred_binary_final, y_score_final, cfg.get("METRICS", {}), costs=threshold_costs or None
        )
        metrics_s3_test.update({"model": "BTTWD", "fold": "test"})
        for k, v in metrics_binary_test.items():
            if k not in metrics_s3_test:
                metrics_s3_test[k] = v
        test_holdout_records.append(metrics_s3_test)
        per_fold_records.append(metrics_s3_test)
        log_metrics("【BTTWD-测试集】", metrics_s3_test)

        for base_key in baseline_set:
            res = _eval_baseline_holdout(base_key, X, y, X_test, y_test, cfg, costs=threshold_costs or None)
            res["fold"] = "test"
            baseline_holdout_results[base_key] = res
            per_fold_records.append(res)

        test_bucket_parts_final = bttwd_final.bucket_tree.assign_bucket_parts(bucket_df_test)
        test_bucket_ids_final = bttwd_final._route_bucket_ids(test_bucket_parts_final)
        test_bucket_df_final = _summarize_test_buckets(test_bucket_ids_final, y_test, y_pred_final, threshold_costs)
        if not test_bucket_df_final.empty:
            bttwd_final.update_test_stats(test_bucket_df_final)
            bttwd_final._export_bucket_reports(fold="test", append_tree=True)

    summary_df = pd.DataFrame(summary_rows)
    per_fold_output_df = pd.DataFrame(per_fold_records)

    overview_records = []
    if test_holdout_records or baseline_holdout_results:
        overview_records.extend(test_holdout_records)
        overview_records.extend(baseline_holdout_results.values())
    else:
        for row in summary_rows:
            base_row = {"model": row.get("model")}
            for k, v in row.items():
                if k.endswith("_mean"):
                    base_row[k[:-5]] = v
            overview_records.append(base_row)

    # 写文件
    if cfg.get("OUTPUT", {}).get("save_per_fold_metrics", True):
        per_fold_output_df.to_csv(os.path.join(results_dir, "metrics_kfold_per_fold.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "metrics_kfold_summary.csv"), index=False)
    if bucket_metrics_records and cfg.get("OUTPUT", {}).get("save_bucket_metrics", True):
        all_bucket_df = pd.concat(bucket_metrics_records, ignore_index=True)
        if "pos_rate_all" in all_bucket_df.columns and "pos_rate" not in all_bucket_df.columns:
            all_bucket_df["pos_rate"] = all_bucket_df["pos_rate_all"]
        all_bucket_df.to_csv(os.path.join(results_dir, "bucket_metrics.csv"), index=False)
    if threshold_log_records and cfg.get("OUTPUT", {}).get("save_threshold_logs", True):
        th_filename = cfg.get("OUTPUT", {}).get("threshold_log_filename", "bucket_thresholds_per_fold.csv")
        pd.concat(threshold_log_records, ignore_index=True).to_csv(
            os.path.join(results_dir, th_filename), index=False
        )
    if overview_records:
        pd.DataFrame(overview_records).to_csv(os.path.join(results_dir, "metrics_overview.csv"), index=False)
    log_info("【K折实验】所有结果已写入 results 目录")

    if bucket_test_gain_records:
        bucket_test_df = pd.DataFrame(bucket_test_gain_records)
        ordered_cols = [
            "fold",
            "bucket_id",
            "parent_id",
            "level",
            "n_train",
            "n_val",
            "n_test",
            "BAC",
            "F1",
            "AUC",
            "Regret",
            "BND_ratio",
            "POS_coverage",
            "is_leaf",
            "is_weak",
            "threshold_source_bucket",
            "threshold_used",
            "baseline_precision",
            "baseline_recall",
            "baseline_f1",
            "baseline_bac",
            "baseline_auc",
            "baseline_mcc",
            "baseline_kappa",
            "baseline_regret",
            "baseline_bnd_ratio",
            "baseline_pos_coverage",
        ]
        if "is_leaf" not in bucket_test_df.columns:
            bucket_test_df["is_leaf"] = False
        bucket_test_df["is_leaf"] = bucket_test_df["is_leaf"].fillna(False).astype(bool)

        existing_cols = [c for c in ordered_cols if c in bucket_test_df.columns]
        remaining_cols = [c for c in bucket_test_df.columns if c not in existing_cols]
        bucket_test_df = bucket_test_df[existing_cols + remaining_cols]
        bucket_test_df.to_csv(Path(results_dir) / "bucket_metrics_gain_test_per_fold.csv", index=False)

    if per_sample_test_records:
        per_sample_df = pd.DataFrame(per_sample_test_records)
        per_sample_df.to_csv(Path(results_dir) / "per_sample_test_predictions.csv", index=False)

    return {"baselines": baseline_results, "bttwd": {"per_fold": per_fold_records, "summary": summary_rows}}
