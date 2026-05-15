"""实验运行封装。"""

from pathlib import Path

from ..run_experiment import run_config


def run_batch(config_path: str | Path) -> None:
    """调用命令式批量入口，避免复制实验编排逻辑。"""

    from scripts.run_experiments import run_batch as _run_batch

    _run_batch(Path(config_path))


__all__ = ["run_config", "run_batch"]
