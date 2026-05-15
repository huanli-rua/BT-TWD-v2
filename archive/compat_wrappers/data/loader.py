"""数据加载模块。

当前实现复用原 `bttwdlib.data_loader`，避免改变任何数据读取、标签映射和路径解析逻辑。
"""

from ..data_loader import load_adult_raw, load_dataset

__all__ = ["load_adult_raw", "load_dataset"]
