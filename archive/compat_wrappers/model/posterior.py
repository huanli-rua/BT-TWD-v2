"""全局 posterior 与桶内后验模型入口。

`BTTWDModel` 内部仍负责全局模型训练、桶内模型、概率预测和回退逻辑。
"""

from ..bttwd_model import BTTWDModel

__all__ = ["BTTWDModel"]
