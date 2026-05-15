"""数据划分说明。

第一篇实验当前的 holdout 与 5-fold stratified cross-validation 逻辑保留在
`bttwdlib.cv_runner` 中，避免迁移造成划分顺序或随机种子变化。
"""

from sklearn.model_selection import StratifiedKFold, train_test_split

__all__ = ["StratifiedKFold", "train_test_split"]
