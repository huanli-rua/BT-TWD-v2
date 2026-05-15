"""全局 posterior 模型。

第二篇主流程只训练一个全局 posterior，所有 bucket 共享该后验概率，不训练局部 posterior。
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    _XGB_AVAILABLE = False


def build_global_posterior(cfg: dict):
    bcfg = cfg.get("BTTWD", {})
    name = str(bcfg.get("global_estimator", bcfg.get("posterior_estimator", "logreg"))).lower()
    if name in {"xgb", "xgboost"}:
        if not _XGB_AVAILABLE:
            raise RuntimeError("配置使用 XGBoost 全局 posterior，但当前环境未安装 xgboost。")
        xgb_cfg = bcfg.get("global_xgb", {})
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
        )
    if name in {"rf", "random_forest", "randomforest"}:
        rf_cfg = bcfg.get("global_rf", bcfg.get("bucket_rf", {}))
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth"),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=rf_cfg.get("random_state", 42),
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=bcfg.get("knn_k", 10), n_jobs=bcfg.get("knn_jobs", -1))
    if name in {"nb", "gnb", "naive_bayes"}:
        return GaussianNB()
    return LogisticRegression(max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0))


class GlobalPosterior:
    """全局后验概率模型包装器。"""

    def __init__(self, cfg: dict):
        self.model = build_global_posterior(cfg)

    def fit(self, X, y) -> "GlobalPosterior":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X).astype(float)


__all__ = ["GlobalPosterior", "build_global_posterior"]
