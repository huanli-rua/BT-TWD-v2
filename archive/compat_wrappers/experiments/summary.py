"""结果汇总工具。"""

from pathlib import Path

import pandas as pd


def collect_overview(output_root: str | Path) -> pd.DataFrame:
    """读取 outputs/*/metrics_overview.csv，生成跨数据集汇总表。"""

    root = Path(output_root)
    records = []
    for path in sorted(root.glob("*/metrics_overview.csv")):
        df = pd.read_csv(path)
        df.insert(0, "dataset", path.parent.name)
        records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


__all__ = ["collect_overview"]
