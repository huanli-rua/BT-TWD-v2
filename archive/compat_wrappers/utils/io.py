"""IO 路径工具。"""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。"""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


__all__ = ["ensure_dir"]
