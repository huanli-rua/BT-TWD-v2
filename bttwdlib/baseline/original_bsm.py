"""第一篇 BSM 相关旧逻辑入口。

保留 gain、weak bucket 和 ancestor threshold substitution 作为原方法 baseline 的组成部分。
"""

from ..bttwd_model import BTTWDModel
from ..bucket_gain import compute_bucket_gain, compute_bucket_score

__all__ = ["BTTWDModel", "compute_bucket_gain", "compute_bucket_score"]
