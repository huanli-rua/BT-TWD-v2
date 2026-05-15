"""第一篇 BT-TWD 原方法 baseline。

该入口只用于实验对比，第二篇 governance 主流程不依赖这里的 BSM/weak bucket/backoff。
"""

from __future__ import annotations

from ..run_experiment import run_config


def run_original_bttwd(config_path, cfg_override: dict | None = None):
    return run_config(config_path, cfg_override=cfg_override)


__all__ = ["run_original_bttwd"]
