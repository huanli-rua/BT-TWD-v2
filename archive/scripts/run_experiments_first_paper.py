"""命令式实验入口。

默认用 configs/default.yaml 批量运行第一篇论文的所有数据集配置。
也可以传入单个数据集配置，此时只运行该配置。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib.run_experiment import run_config  # noqa: E402
from bttwdlib.utils_logging import log_info  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return REPO_ROOT / path


def _dataset_output_dir(output_root: Path, cfg_path: Path, dataset_cfg: dict | None = None) -> Path:
    name = None
    if dataset_cfg:
        name = dataset_cfg.get("name")
    if not name:
        name = cfg_path.stem
    return output_root / str(name)


def _collect_overview(output_root: Path) -> None:
    """汇总各数据集的 overview/summary，便于论文表格复核。"""

    records: list[pd.DataFrame] = []
    for overview_path in sorted(output_root.glob("*/metrics_overview.csv")):
        df = pd.read_csv(overview_path)
        df.insert(0, "dataset", overview_path.parent.name)
        records.append(df)
    if not records:
        return

    summary_path = output_root / "metrics_all_datasets_overview.csv"
    pd.concat(records, ignore_index=True).to_csv(summary_path, index=False)
    log_info(f"【总体汇总】已写出 {summary_path}")


def run_batch(config_path: Path) -> None:
    batch_cfg = _load_yaml(config_path)
    output_root = Path(batch_cfg.get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = batch_cfg.get("datasets") or []
    if not datasets:
        run_config(config_path)
        return

    for item in datasets:
        if isinstance(item, str):
            dataset_cfg = {"config": item}
        else:
            dataset_cfg = dict(item)
        cfg_path = _resolve_config_path(dataset_cfg["config"], config_path.parent)
        results_dir = _dataset_output_dir(output_root, cfg_path, dataset_cfg)
        log_info(f"【批量实验】开始运行 {cfg_path}，输出目录={results_dir}")
        run_config(cfg_path, cfg_override={"OUTPUT": {"results_dir": str(results_dir)}})

    _collect_overview(output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 BT-TWD 第一篇论文实验")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="批量配置或单数据集 YAML 配置")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(Path(args.config))


if __name__ == "__main__":
    main()
