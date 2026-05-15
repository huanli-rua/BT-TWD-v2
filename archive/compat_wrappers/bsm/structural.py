"""BSM 结构评分与增益计算。

该模块复用原 `bucket_gain.py`，用于弱桶判定和 bucket gain 记录。
"""

from ..bucket_gain import compute_bucket_gain, compute_bucket_score

__all__ = ["compute_bucket_gain", "compute_bucket_score"]
