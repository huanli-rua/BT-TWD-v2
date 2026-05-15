"""弱桶可靠性说明。

第一篇实验中 weak bucket 的判定目前保留在 `BTTWDModel.fit` 内部，
依赖验证集样本数、桶得分和 gain 等既有规则。此处仅作为规范化目录入口。
"""

from ..bttwd_model import BTTWDModel

__all__ = ["BTTWDModel"]
