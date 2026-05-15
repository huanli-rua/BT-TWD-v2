import os

from .utils_logging import log_info


def load_yaml_cfg(path: str) -> dict:
    """从相对路径加载 YAML 配置。"""
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        log_info(f"【配置加载】已读取 {path}")
        return cfg
    except Exception as e:
        raise RuntimeError(f"读取配置时出错：{e}")


def show_cfg(cfg: dict) -> None:
    """按模块打印配置摘要。"""
    data_cfg = cfg.get("DATA", {})
    log_info(
        f"【配置-数据】数据集={data_cfg.get('dataset_name')}, k折={data_cfg.get('n_splits')}, 目标列={data_cfg.get('target_col')}, 正类=\"{data_cfg.get('positive_label')}\""
    )
    bttwd_cfg = cfg.get("BTTWD", {})
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})
    threshold_mode = threshold_cfg.get("mode", bttwd_cfg.get("thresholds_mode"))
    log_info(
        f"【配置-BTTWD】阈值模式={threshold_mode}, 全局模型={bttwd_cfg.get('global_estimator')}, "
        f"桶内模型={bttwd_cfg.get('bucket_estimator')}, 后验估计器(兼容字段)={bttwd_cfg.get('posterior_estimator')}"
    )
    baseline_cfg = cfg.get("BASELINES", {})
    log_info(
        f"【配置-基线】LogReg启用={baseline_cfg.get('use_logreg')}, RandomForest启用={baseline_cfg.get('use_random_forest')}, "
        f"KNN启用={baseline_cfg.get('use_knn')}, XGBoost启用={baseline_cfg.get('use_xgboost')}"
    )


def flatten_cfg_to_vars(cfg: dict) -> dict:
    """将常用配置扁平化，便于 notebook 快速查看。"""
    flat = {}
    flat.update(cfg.get("DATA", {}))
    flat.update({f"PREP_{k}": v for k, v in cfg.get("PREPROCESS", {}).items()})
    flat.update({f"BTTWD_{k}": v for k, v in cfg.get("BTTWD", {}).items()})
    flat.update({f"EXP_{k}": v for k, v in cfg.get("EXP", {}).items()})
    return flat
