"""ancestor threshold backoff 说明。

阈值回退实现保留在 `BTTWDModel._get_threshold_with_backoff` 和预测流程中，
以保证弱桶继承父桶/ROOT 阈值的行为不变。
"""

from ..bttwd_model import BTTWDModel

__all__ = ["BTTWDModel"]
