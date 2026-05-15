"""数据加载、预处理和划分相关的兼容入口。"""

from .loader import load_adult_raw, load_dataset
from .preprocess import apply_feature_engineering_with_config, prepare_features_and_labels

__all__ = [
    "load_adult_raw",
    "load_dataset",
    "apply_feature_engineering_with_config",
    "prepare_features_and_labels",
]
