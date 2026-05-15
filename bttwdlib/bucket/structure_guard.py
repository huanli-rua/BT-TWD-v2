"""bucket tree 结构保护工具。

第二篇主流程只读取规则分桶结构，不允许在 governance 阶段触发旧 BSM 回退。
"""

from __future__ import annotations


def assert_no_bsm_dependency(config: dict) -> None:
    """显式标记第二篇主流程不依赖 original_bsm。

    当前函数只做轻量检查，后续如果配置中加入治理方法开关，可在这里集中拦截。
    """

    governance_cfg = config.get("GOVERNANCE", {})
    if governance_cfg.get("use_original_bsm_backoff", False):
        raise ValueError("第二篇 governance 主流程不允许调用 original_bsm 决策回退。")


__all__ = ["assert_no_bsm_dependency"]
