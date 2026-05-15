"""数据加载入口：复用现有 CSV/ARFF/Excel 等读取与标签处理逻辑。"""

from ..data_loader import load_adult_raw, load_dataset

__all__ = ["load_adult_raw", "load_dataset"]
