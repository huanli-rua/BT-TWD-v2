import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .metrics import (
    compute_binary_metrics,
    compute_s3_metrics,
    log_metrics,
    predict_binary_by_cost,
)
from .threshold_search import search_thresholds_with_regret
from .utils_logging import log_info


def get_decision_threshold(model_key: str, cfg: dict) -> tuple[float, str, bool]:
    """
    根据 BASELINES 配置返回决策阈值。

    返回值为 (threshold, mode, used_custom)，其中 used_custom 表示是否启用了模型自定义阈值。
    """

    base_cfg = cfg.get("BASELINES", {})
    common_cfg = base_cfg.get("common", {})

    mode = common_cfg.get("threshold_mode", "fixed").strip().lower()
    global_threshold = float(common_cfg.get("fixed_threshold", 0.5))

    if mode == "fixed":
        return global_threshold, mode, False

    if mode == "per_model":
        model_cfg = base_cfg.get(model_key, {})
        use_custom = model_cfg.get("use_custom_threshold", False)
        if use_custom:
            return float(model_cfg.get("custom_threshold", global_threshold)), mode, True
        return global_threshold, mode, False

    # 预留其它模式，默认回退到全局阈值
    return global_threshold, mode, False


def _make_writable_matrix(X):
    """确保特征矩阵是可写的 numpy 数组。"""

    if sparse.issparse(X):
        # 对于随机森林，直接转换为稠密矩阵更稳妥
        X = X.toarray()
    else:
        X = np.asarray(X)

    if not X.flags.writeable:
        X = np.array(X, copy=True)
    return X


def _make_writable_vector(y):
    """确保标签向量是一维、可写的 numpy 数组。"""

    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    return arr


def _aggregate_baseline_summary(per_fold_records: list[dict]) -> dict:
    """
    将基线模型的每折指标做均值/标准差汇总。
    per_fold_records: [{'Precision': ..., 'Recall': ..., ..., 'fold': 1}, ...]
    """
    if not per_fold_records:
        return {}

    # 取出所有列名，去掉 fold
    keys = set()
    for rec in per_fold_records:
        keys.update(rec.keys())
    keys.discard("fold")

    summary: dict = {}
    for col in sorted(keys):
        values = []
        for rec in per_fold_records:
            v = rec.get(col, np.nan)
            # 避免把 dict / list 之类塞进来，这里只聚合标量数值
            if isinstance(v, (int, float, np.number)) or v is None or np.isnan(v):
                values.append(v)
            else:
                # 如果真的有非数值（一般不会有），直接跳过该列
                values = None
                break

        if values is None:
            continue

        arr = np.array(values, dtype=float)
        summary[f"{col}_mean"] = float(np.nanmean(arr))
        summary[f"{col}_std"] = float(np.nanstd(arr))

    return summary



def _run_baseline_cv(
    model_builder,
    model_name: str,
    model_key: str,
    X,
    y,
    cfg,
    cv_splitter,
    costs: dict | None = None,
) -> dict:
    X = _make_writable_matrix(X)
    y = _make_writable_vector(y)

    metrics_cfg = cfg.get("METRICS", {})
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS") or {}
    threshold, mode, used_custom = get_decision_threshold(model_key, cfg)
    costs = costs or threshold_cfg.get("costs", {})

    per_fold_records: list[dict] = []
    if isinstance(cv_splitter, StratifiedKFold):
        splitter = cv_splitter
    else:
        splitter = StratifiedKFold(
            n_splits=getattr(cv_splitter, "n_splits", 5),
            shuffle=True,
            random_state=42,
        )

    if mode == "search":
        log_info(
            f"【基线-{model_name}】阈值模式=search，将按 α/β 网格搜索最优 regret（使用验证集）"
        )
    elif mode == "per_model" and used_custom:
        log_info(f"【基线-{model_name}】使用模型自定义阈值={threshold:.3f}（per_model 模式）")
    elif mode == "per_model":
        log_info(f"【基线-{model_name}】使用通用阈值={threshold:.3f}（per_model 模式）")
    else:
        log_info(f"【基线-{model_name}】使用决策阈值={threshold:.3f}（fixed 模式）")

    fold_idx = 1
    for train_idx, test_idx in splitter.split(X, y):
        clf = model_builder()
        y_train = y[train_idx]
        if np.unique(y_train).size < 2:
            log_info(f"【基线-{model_name}】第 {fold_idx} 折训练集仅包含单一类别，跳过该折")
            fold_idx += 1
            continue
        if mode == "search":
            X_train_fold = X[train_idx]
            y_train_fold = y_train
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_fold,
                y_train_fold,
                test_size=threshold_cfg.get("val_ratio", 0.2),
                random_state=42,
                stratify=y_train_fold if np.unique(y_train_fold).size > 1 else None,
            )
            clf.fit(X_tr, y_tr)
            if not hasattr(clf, "predict_proba"):
                log_info(
                    f"【基线-{model_name}】模型不支持 predict_proba，search 模式降级为固定阈值 {threshold:.3f}"
                )
                y_pred = clf.predict(X[test_idx])
                y_score = np.zeros_like(y_pred, dtype=float)
                metrics_dict = compute_binary_metrics(
                    y[test_idx], y_pred, y_score, metrics_cfg, costs=costs
                )
                metrics_dict.setdefault("BND_ratio", 0.0)
                metrics_dict.setdefault("POS_Coverage", float("nan"))
            else:
                y_score_val = clf.predict_proba(X_val)[:, 1]
                alpha_grid = threshold_cfg.get("alpha_grid", [0.5])
                beta_grid = threshold_cfg.get("beta_grid", [0.0])
                gap_min = threshold_cfg.get("gap_min", 0.0)
                alpha_opt, beta_opt, _ = search_thresholds_with_regret(
                    y_score_val, y_val, alpha_grid, beta_grid, costs, gap_min=gap_min
                )
                y_score = clf.predict_proba(X[test_idx])[:, 1]
                y_pred = np.where(
                    y_score >= alpha_opt,
                    1,
                    np.where(y_score <= beta_opt, 0, -1),
                )
                metrics_dict = compute_s3_metrics(
                    y[test_idx], y_pred, y_score, metrics_cfg, costs=costs
                )
                metrics_dict["alpha"] = float(alpha_opt)
                metrics_dict["beta"] = float(beta_opt)
        else:
            clf.fit(X[train_idx], y[train_idx])

            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X[test_idx])[:, 1]
                y_pred = (y_score >= threshold).astype(int)
            else:
                y_pred = clf.predict(X[test_idx])
                y_score = np.zeros_like(y_pred, dtype=float)

            metrics_dict = compute_binary_metrics(
                y[test_idx], y_pred, y_score, metrics_cfg, costs=costs
            )
            metrics_dict.setdefault("BND_ratio", 0.0)
            metrics_dict.setdefault("POS_Coverage", float("nan"))
        metrics_dict["fold"] = fold_idx
        per_fold_records.append(metrics_dict)
        fold_idx += 1

    summary = _aggregate_baseline_summary(per_fold_records)
    log_metrics(f"【基线-{model_name}】整体指标：", summary)
    return {"per_fold": per_fold_records, "summary": summary}


def train_eval_logreg(X, y, cfg, cv_splitter, costs: dict | None = None) -> dict:
    model_cfg = cfg.get("BASELINES", {}).get("logreg", {})

    def _builder():
        return LogisticRegression(max_iter=model_cfg.get("max_iter", 200), C=model_cfg.get("C", 1.0))

    return _run_baseline_cv(_builder, "LogReg", "logreg", X, y, cfg, cv_splitter, costs=costs)


def train_eval_random_forest(X, y, cfg, cv_splitter, costs: dict | None = None) -> dict:
    rf_cfg = cfg.get("BASELINES", {}).get("random_forest", {})

    def _builder():
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth"),
            random_state=rf_cfg.get("random_state", 42),
            n_jobs=cfg.get("EXP", {}).get("n_jobs", -1),
        )

    return _run_baseline_cv(_builder, "RF", "random_forest", X, y, cfg, cv_splitter, costs=costs)


def train_eval_knn(X, y, cfg, cv_splitter, costs: dict | None = None) -> dict:
    """
    使用 KNN 作为全局基线模型，进行 k 折交叉验证。
    """

    knn_cfg = cfg.get("BASELINES", {}).get("knn", {})

    def _builder():
        return KNeighborsClassifier(
            n_neighbors=knn_cfg.get("n_neighbors", 10),
        )

    return _run_baseline_cv(_builder, "KNN", "knn", X, y, cfg, cv_splitter, costs=costs)


def train_eval_xgboost(X, y, cfg, cv_splitter, costs: dict | None = None) -> dict:
    """
    使用 XGBoost 作为全局基线模型，进行 k 折交叉验证。
    """

    if not _XGB_AVAILABLE:
        raise RuntimeError("配置了 use_xgboost=True 但未安装 xgboost，请先安装该库。")

    xgb_cfg = cfg.get("BASELINES", {}).get("xgboost", {})

    def _builder():
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

    return _run_baseline_cv(_builder, "XGB", "xgboost", X, y, cfg, cv_splitter, costs=costs)
