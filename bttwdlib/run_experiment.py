"""通用入口：读取指定配置跑一次分层切分 + BTTWD 训练/评估。

默认使用仓库内的 airlines 延误示例配置，可通过命令行参数替换为任何数据集配置。
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib.config_loader import load_yaml_cfg, show_cfg  # noqa: E402
from bttwdlib.cv_runner import run_holdout_experiment, run_kfold_experiments  # noqa: E402
from bttwdlib.data_loader import load_dataset  # noqa: E402
from bttwdlib.preprocessing import prepare_features_and_labels  # noqa: E402
from bttwdlib.preprocessing import apply_feature_engineering_with_config  # noqa: E402
from bttwdlib.utils_logging import log_info  # noqa: E402
from bttwdlib.utils_seed import set_global_seed  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "adult_bttwd.yaml"


def _resolve_data_paths(cfg: dict, cfg_path: Path) -> None:
    """将配置中的数据路径解析为绝对路径。

    旧配置多使用 `../data/...`，语义上是相对 configs 目录。这里在入口层解析，
    避免运行目录变化影响复现结果。
    """

    data_cfg = cfg.get("DATA", {})
    for key in ("raw_path", "path", "data_path", "train_csv", "test_csv", "meta_path"):
        path_value = data_cfg.get(key)
        if not path_value:
            continue
        path = Path(path_value)
        if path.is_absolute():
            continue
        candidates = [cfg_path.parent / path, REPO_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                data_cfg[key] = str(candidate.resolve())
                break


def _build_bucket_feature_df(df_raw, cfg) -> tuple[np.ndarray, np.ndarray, object, object, object]:
    X, y, meta = prepare_features_and_labels(df_raw, cfg)
    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols: List[str] = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])
    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    for lvl in bucket_levels:
        col_name = lvl.get("col") or lvl.get("feature")
        if col_name and col_name not in bucket_cols:
            bucket_cols.append(col_name)
    df_processed = meta.get("df_processed", df_raw)
    bucket_df = df_processed[bucket_cols].reset_index(drop=True)
    return X, y, meta, bucket_df, bucket_cols


def _transform_with_pipeline(df_raw, cfg, pipeline, bucket_cols: list[str]):
    """使用已有预处理管道对新数据集进行特征转换。"""

    data_cfg = cfg.get("DATA", {})
    prep_cfg = cfg.get("PREPROCESS", {})
    target_col = data_cfg.get("target_col", "income")
    target_transform = data_cfg.get("target_transform") or {}
    target_col_for_model = target_transform.get("new_col", target_col)
    positive_label = data_cfg.get("positive_label", 1)
    negative_label = data_cfg.get("negative_label")

    drop_cols = set(prep_cfg.get("drop_cols", []))
    drop_cols.add(target_col_for_model)
    source_target_col = data_cfg.get("target_col")
    if source_target_col and source_target_col != target_col_for_model:
        drop_cols.add(source_target_col)

    df_processed = apply_feature_engineering_with_config(df_raw, cfg)
    X_raw = df_processed.drop(columns=list(drop_cols), errors="ignore")
    X = pipeline.transform(X_raw)
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)
    if not X.flags.writeable:
        X = np.array(X, copy=True)

    y = np.where(df_raw[target_col_for_model] == positive_label, 1, 0)
    if negative_label is not None:
        y = np.where(df_raw[target_col_for_model] == positive_label, 1, 0)

    bucket_df = df_processed[bucket_cols].reset_index(drop=True)
    return X, y, bucket_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run BTTWD training & eval with a YAML config.")
    parser.add_argument(
        "--config",
        "--cfg",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to the YAML configuration file. "
            "Defaults to the adult BT-TWD config."
        ),
    )
    return parser.parse_args()


def run_config(config_path: str | Path, cfg_override: dict | None = None):
    """按单个数据集配置运行第一篇 BT-TWD 实验。

    该函数保留原入口中的完整实验链路：数据加载、预处理、K 折或 holdout、
    全局 posterior、桶树路由、局部阈值搜索、BSM/weak bucket/backoff 与结果写出。
    cfg_override 仅供批量脚本覆盖输出目录等运行项，不改变算法规则。
    """

    cfg_path = Path(config_path)
    cfg = load_yaml_cfg(cfg_path)
    if cfg_override:
        for section, values in cfg_override.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
    _resolve_data_paths(cfg, cfg_path)
    show_cfg(cfg)
    set_global_seed(cfg.get("SEED", {}).get("global_seed", 42))

    df_raw, target_col = load_dataset(cfg)
    data_cfg = cfg.get("DATA", {})
    log_info(
        f"【入口】数据集={data_cfg.get('dataset_name')}，样本数={len(df_raw)}，标签列={target_col}"
    )

    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    log_info(f"【桶树层级】分裂顺序={[lvl.get('name') for lvl in bucket_levels]}")

    split_col = "split" if "split" in df_raw.columns else None
    test_data = None
    if split_col:
        df_train = df_raw[df_raw[split_col].str.lower() == "train"].reset_index(drop=True)
        df_test = df_raw[df_raw[split_col].str.lower() == "test"].reset_index(drop=True)
        log_info(f"【入口】检测到显式训练/测试划分：train={len(df_train)}，test={len(df_test)}")
        X, y, meta, bucket_df, bucket_cols = _build_bucket_feature_df(df_train, cfg)
        X_test, y_test, bucket_df_test = _transform_with_pipeline(df_test, cfg, meta["preprocess_pipeline"], bucket_cols)
        test_data = (X_test, y_test, bucket_df_test)
    else:
        X, y, meta, bucket_df, bucket_cols = _build_bucket_feature_df(df_raw, cfg)

    use_kfold = data_cfg.get("use_kfold", False)
    if isinstance(use_kfold, str):
        use_kfold = use_kfold.strip().lower() in {"1", "true", "yes", "y"}
    if use_kfold:
        log_info("【模式选择】use_kfold=true，启动K折实验...")
        return run_kfold_experiments(X, y, bucket_df, cfg, test_data=test_data)

    return run_holdout_experiment(X, y, bucket_df, cfg, bucket_cols=bucket_cols)


def main():
    args = parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
