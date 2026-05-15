import pandas as pd

from .utils_logging import log_info


class BucketTree:
    def __init__(self, levels_cfg: list, feature_names: list[str]):
        self.levels_cfg = levels_cfg
        self.feature_names = feature_names

    def _assign_single_level(self, series: pd.Series, level_cfg: dict) -> pd.Series:
        level_type = str(level_cfg.get("type", "")).lower()
        if level_type in {"numeric_bin", "numeric_binning"}:
            bins = level_cfg.get("bins", [])
            labels = level_cfg.get("labels")
            cut_bins = [-float("inf")] + bins + [float("inf")]
            n_intervals = len(cut_bins) - 1
            if labels is None:
                labels = [f"bin_{i}" for i in range(n_intervals)]
            elif len(labels) != n_intervals:
                raise ValueError(
                    f"numeric_bin labels length {len(labels)} does not match interval count {n_intervals}"
                )
            return pd.cut(series, bins=cut_bins, labels=labels, include_lowest=True)
        if level_type in {"categorical_group", "category_group"}:
            other_label = (
                level_cfg.get("other_label")
                or level_cfg.get("default_group")
                or level_cfg.get("others")
                or "OTHER"
            )
            groups = level_cfg.get("groups")
            if groups:
                mapping = {}
                if isinstance(groups, dict):
                    group_items = groups.items()
                else:
                    group_items = []
                    for group_cfg in groups:
                        label = group_cfg.get("label")
                        values = group_cfg.get("values", [])
                        if label is not None:
                            group_items.append((label, values))
                for group_name, values in group_items:
                    for v in values:
                        mapping[v] = group_name
                return series.map(mapping).fillna(other_label)
            min_count = level_cfg.get("min_count")
            if min_count is not None:
                value_counts = series.value_counts()
                popular_values = set(value_counts[value_counts >= min_count].index)
                return series.apply(lambda v: v if v in popular_values else other_label)
            return series.fillna(other_label)
        return pd.Series(["unknown"] * len(series), index=series.index)

    def assign_buckets(self, X_df: pd.DataFrame) -> pd.Series:
        missing_cols = [
            lvl.get("col") or lvl.get("feature")
            for lvl in self.levels_cfg
            if (lvl.get("col") or lvl.get("feature")) not in X_df.columns
        ]
        if missing_cols:
            raise KeyError(f"分桶数据缺少以下列：{', '.join(missing_cols)}")
        bucket_parts = []
        for level_cfg in self.levels_cfg:
            col = level_cfg.get("col") or level_cfg.get("feature")
            part = self._assign_single_level(X_df[col], level_cfg)
            unknown_mask = part.isna()
            if unknown_mask.any():
                log_info(f"【桶树】列 {col} 出现未知取值，{unknown_mask.sum()} 条记录记为 unknown")
                part = part.astype(object).fillna("unknown")
            level_num = level_cfg.get("level")
            if level_num is not None:
                level_name = f"L{level_num}_{col}"
            else:
                level_name = level_cfg.get("name", col)
            bucket_parts.append(part.astype(str).apply(lambda v: f"{level_name}={v}"))
        bucket_id = bucket_parts[0]
        for idx in range(1, len(bucket_parts)):
            bucket_id = bucket_id + "|" + bucket_parts[idx]
        log_info(f"【桶树】已为样本生成桶ID，共 {bucket_id.nunique()} 个组合")
        return bucket_id

    def assign_bucket_parts(self, X_df: pd.DataFrame) -> list[pd.Series]:
        """返回每一层的桶标签序列，便于增量式分裂逻辑使用。"""

        missing_cols = [
            lvl.get("col") or lvl.get("feature")
            for lvl in self.levels_cfg
            if (lvl.get("col") or lvl.get("feature")) not in X_df.columns
        ]
        if missing_cols:
            raise KeyError(f"分桶数据缺少以下列：{', '.join(missing_cols)}")

        parts = []
        for level_cfg in self.levels_cfg:
            col = level_cfg.get("col") or level_cfg.get("feature")
            part = self._assign_single_level(X_df[col], level_cfg)
            unknown_mask = part.isna()
            if unknown_mask.any():
                log_info(f"【桶树】列 {col} 出现未知取值，{unknown_mask.sum()} 条记录记为 unknown")
                part = part.astype(object).fillna("unknown")
            level_num = level_cfg.get("level")
            if level_num is not None:
                level_name = f"L{level_num}_{col}"
            else:
                level_name = level_cfg.get("name", col)
            parts.append(part.astype(str).apply(lambda v: f"{level_name}={v}"))
        return parts

    def get_level_names(self) -> list[str]:
        names = []
        for lvl in self.levels_cfg:
            if lvl.get("name"):
                names.append(lvl["name"])
            elif lvl.get("level") is not None:
                col = lvl.get("col") or lvl.get("feature", "unknown")
                names.append(f"L{lvl['level']}_{col}")
            else:
                names.append(lvl.get("col") or lvl.get("feature") or "level")
        return names


def get_parent_bucket_id(bucket_id: str) -> str | None:
    """
    输入一个桶ID，如 'L1_age=old|L2_education=mid|L3_hours=high_hours'，
    返回其父桶ID：'L1_age=old|L2_education=mid'。
    若已是顶层（例如只有 'L1_age=old'），则返回 None。
    """

    parts = bucket_id.split("|")
    if len(parts) <= 1:
        return None
    return "|".join(parts[:-1])
