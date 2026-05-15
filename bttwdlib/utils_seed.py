import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """设置全局随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # 环境未安装 torch 时静默跳过
        pass
