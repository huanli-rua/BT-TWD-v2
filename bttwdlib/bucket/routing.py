"""第二篇主流程使用的 bucket 路由工具。

这里不调用第一篇 `BTTWDModel` 的 split_plan、weak bucket 或 ancestor backoff。
"""

from __future__ import annotations

from ..bucket_rules import get_parent_bucket_id


def bucket_level(bucket_id: str) -> int:
    if not bucket_id or bucket_id == "ROOT":
        return 0
    return len(str(bucket_id).split("|"))


def bucket_path_to_root(bucket_id: str) -> list[str]:
    """返回从当前桶到 ROOT 的路径。"""

    path = []
    current = bucket_id
    while current:
        path.append(current)
        parent = get_parent_bucket_id(current)
        current = parent if parent is not None else "ROOT"
        if path[-1] == "ROOT":
            break
    if path[-1] != "ROOT":
        path.append("ROOT")
    return path


def parent_bucket(bucket_id: str) -> str:
    parent = get_parent_bucket_id(bucket_id)
    return parent if parent is not None else "ROOT"


__all__ = ["bucket_level", "bucket_path_to_root", "parent_bucket"]
