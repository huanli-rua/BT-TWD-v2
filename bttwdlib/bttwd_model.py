import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .bucket_rules import BucketTree, get_parent_bucket_id
from .utils_logging import log_bt, log_info
from .threshold_search import search_thresholds_with_regret, compute_regret
from .bucket_gain import compute_bucket_gain, compute_bucket_score


class BTTWDModel:
    def __init__(self, cfg: dict, bucket_tree):
        self.cfg = cfg
        self.bucket_tree = bucket_tree

        bcfg = cfg.get("BTTWD", {})
        data_cfg = cfg.get("DATA", {})
        thresh_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})

        self.min_bucket_size = bcfg.get("min_bucket_size", 50)
        self.use_min_bucket_size_limit = bcfg.get("use_min_bucket_size_limit", True)
        self.use_merge_small_to_residual = bcfg.get("use_merge_small_to_residual", True)
        self.min_pos_per_bucket = bcfg.get("min_pos_per_bucket", 0)
        self.max_levels = bcfg.get("max_levels", bcfg.get("max_depth", 10))
        self.min_gain_for_split = bcfg.get("min_gain_for_split", 0.0)
        self.small_bucket_threshold = bcfg.get("small_bucket_threshold", self.min_bucket_size)
        self.use_gain = bcfg.get("use_gain", True)
        self.use_gain_weak_backoff = bcfg.get("use_gain_weak_backoff", True)
        self.gamma_bucket = bcfg.get("gamma_bucket", 0.0)
        self.use_parent_share_rate = bcfg.get("use_parent_share_rate", True)
        self.parent_share_rate = bcfg.get("parent_share_rate", 0.0)
        self.min_parent_share = bcfg.get("min_parent_share", 0)
        self.val_ratio = bcfg.get("val_ratio", 0.2)
        self.min_val_samples_per_bucket = bcfg.get("min_val_samples_per_bucket", 10)
        self.use_global_backoff = bcfg.get("use_global_backoff", True)
        self.bucket_subsample = bcfg.get("bucket_subsample", 1.0)
        self.max_train_samples_per_bucket = bcfg.get("max_train_samples_per_bucket")
        self.stop_split_on_other: bool = bcfg.get("stop_split_on_other", True)
        self.stop_split_labels: set[str] = set(
            str(v)
            for v in bcfg.get(
                "stop_split_labels",
                [                   
                    "OTHER",
                    "OTHERS",
                    "Others",
                    "others",
                    "Other",
                    "other",
                    "OtherType",
                    "OTHER_CARRIER",
                    "OTHER_ORIGIN",
                    "OTHER_DEST",
                    "no_internet",
                ],
            )
        )
        self.optimize_thresholds = True
        self.threshold_mode = thresh_cfg.get("mode", bcfg.get("thresholds_mode", "bucket_wise"))
        self.threshold_objective = thresh_cfg.get("objective", "regret")
        self.threshold_grid_alpha = thresh_cfg.get("alpha_grid", [])
        self.threshold_grid_beta = thresh_cfg.get("beta_grid", [])
        self.gap_min = thresh_cfg.get("gap_min", 0.0)
        self.costs = thresh_cfg.get(
            "costs",
            {
                "C_TP": 0.0,
                "C_TN": 0.0,
                "C_FP": 1.0,
                "C_FN": 3.0,
                "C_BP": 1.5,
                "C_BN": 0.5,
            },
        )
        self.costs_per_bucket = thresh_cfg.get("costs_per_bucket") or {}
        self.global_alpha = thresh_cfg.get("alpha_init", 0.6)
        self.global_beta = thresh_cfg.get("beta_init", 0.2)
        self.min_samples_for_thresholds = thresh_cfg.get(
            "min_samples_for_thresholds", bcfg.get("min_val_samples_per_bucket", 10)
        )
        self.random_state = data_cfg.get("random_state", 42)

        self.bucket_models = {}
        self.bucket_thresholds = {}
        self.bucket_info = {}
        self.bucket_stats = {}
        self.threshold_logs = []
        self.bucket_structure_records: list[dict] = []
        self.fallback_stats: dict[str, dict] = {}
        self.results_dir = self._prepare_results_dir(cfg)
        self.children_map: dict[str, list[str]] = {}
        self.global_model = None
        self.global_pos_rate = 0.5
        self.split_plan = {}
        self.rng = np.random.default_rng(self.random_state)
        self.bucket_estimator = self._build_bucket_estimator()
        self.score_cfg = self._build_score_cfg(cfg)
        self.bucket_specific_cost_warns = 0

    @classmethod
    def from_cfg(cls, cfg: dict, feature_names: list[str] | None = None, bucket_tree=None) -> "BTTWDModel":
        """
        工厂方法：从配置构建 BucketTree 并初始化 BTTWDModel。

        feature_names: 用于分桶的特征名列表；若未提供，则默认使用预处理配置中的连续+类别列。
        bucket_tree: 允许直接传入已构建好的 BucketTree，主要用于测试或自定义场景。
        """

        if bucket_tree is None:
            bttwd_cfg = cfg.get("BTTWD", {})
            bucket_levels = bttwd_cfg.get("bucket_levels", [])
            if feature_names is None:
                prep_cfg = cfg.get("PREPROCESS", {})
                feature_names = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])
            bucket_tree = BucketTree(bucket_levels, feature_names=feature_names or [])

        return cls(cfg, bucket_tree)

    def _sample_bucket_data(self, X_bucket: np.ndarray, y_bucket: np.ndarray, bucket_id: str = ""):
        """
        对桶内样本进行裁剪和随机子采样：
        1. 如果样本数 > max_train_samples_per_bucket，则随机采样上限数量；
        2. 再按 bucket_subsample 比例继续随机采样；
        """

        subsample = float(self.bucket_subsample)
        max_samples = self.max_train_samples_per_bucket

        indices = np.arange(len(y_bucket))
        if max_samples is not None and len(indices) > int(max_samples):
            indices = self.rng.choice(indices, size=int(max_samples), replace=False)

        if subsample < 1.0:
            keep = max(1, int(len(indices) * subsample))
            indices = self.rng.choice(indices, size=keep, replace=False)

        log_bt(
            f"桶{(' ' + bucket_id) if bucket_id else ''}采样：原始样本 N={len(y_bucket)} → 使用样本 n={len(indices)}"
        )

        return X_bucket[indices], y_bucket[indices]

    def _build_bucket_estimator(self, est_name=None):
        """构建桶内局部模型（例如 KNN 或逻辑回归）。"""

        bcfg = self.cfg.get("BTTWD", {})
        if est_name is None:
            est_name = bcfg.get("bucket_estimator", bcfg.get("posterior_estimator", "logreg"))
        est_name = str(est_name).lower() if est_name is not None else "logreg"

        none_aliases = {"none", "no", "null", "disabled"}
        if est_name in none_aliases:
            return None

        if est_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=bcfg.get("knn_k", 10),
                weights=bcfg.get("knn_weights", "uniform"),
                n_jobs=bcfg.get("knn_jobs", -1),
            )

        if est_name in {"logreg", "lr", "logistic", "logistic_regression"}:
            return LogisticRegression(
                max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
            )

        if est_name in {"rf", "random_forest", "randomforest"}:
            rf_cfg = bcfg.get("bucket_rf", {})
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", None),
                n_jobs=rf_cfg.get("n_jobs", -1),
                random_state=rf_cfg.get("random_state", 42),
            )

        if est_name in {"xgb", "xgboost"}:
            if not _XGB_AVAILABLE:
                raise RuntimeError("配置了 bucket_estimator='xgb'，但未安装 xgboost 库")
            xgb_cfg = bcfg.get("bucket_xgb", {})
            return XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 3),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                n_jobs=xgb_cfg.get("n_jobs", -1),
                random_state=xgb_cfg.get("random_state", 42),
            )

        if est_name in {"nb", "gnb", "naive_bayes", "naivebayes"}:
            return GaussianNB()

        log_info(f"【BTTWD】未知的 bucket_estimator='{est_name}'，回退到 logreg")
        return LogisticRegression(
            max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
        )

    def _build_score_cfg(self, cfg: dict) -> dict:
        """构建桶评分配置，兼容旧版 score_metric 字段。"""

        score_cfg = cfg.get("SCORE")
        bcfg = cfg.get("BTTWD", {})

        if not isinstance(score_cfg, dict):
            score_metric = bcfg.get("score_metric", "f1_regret_bnd")
            return {
                "bucket_score_mode": score_metric,
                "f1_weight": 1.0,
                "regret_weight": 1.0,
                "bnd_weight": 1.0,
            }

        merged_cfg = dict(score_cfg)
        merged_cfg.setdefault(
            "bucket_score_mode", score_cfg.get("score_metric", bcfg.get("score_metric", "f1_regret_bnd"))
        )
        merged_cfg.setdefault("f1_weight", 1.0)
        merged_cfg.setdefault("bnd_weight", 1.0)
        merged_cfg.setdefault("regret_weight", 1.0)
        merged_cfg.setdefault("bac_weight", 1.0)
        merged_cfg.setdefault("regret_sign", -1.0)
        return merged_cfg

    def _prepare_results_dir(self, cfg: dict) -> Path:
        output_cfg = cfg.get("OUTPUT", {}) if isinstance(cfg, dict) else {}
        base_dir = output_cfg.get("results_dir", "results")
        results_dir = Path(base_dir)
        if not results_dir.is_absolute():
            results_dir = Path(__file__).resolve().parent.parent / results_dir

        run_name = output_cfg.get("run_name")
        if run_name:
            results_dir = results_dir / str(run_name)

        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def _build_global_estimator(self):
        """构建全局后验估计器（例如 XGB）。"""

        bcfg = self.cfg.get("BTTWD", {})
        est_name = bcfg.get("global_estimator", "logreg")
        est_name = str(est_name).lower() if est_name is not None else "logreg"

        none_aliases = {"none", "no", "null", "disabled"}
        if est_name in none_aliases:
            return None

        if est_name in {"xgb", "xgboost"}:
            if not _XGB_AVAILABLE:
                raise RuntimeError("配置了 global_estimator='xgb' 但未安装 xgboost，请先安装该库。")
            xgb_cfg = bcfg.get("global_xgb", {})
            return XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 300),
                max_depth=xgb_cfg.get("max_depth", 4),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                random_state=xgb_cfg.get("random_state", 42),
                n_jobs=xgb_cfg.get("n_jobs", -1),
                eval_metric="logloss",
                use_label_encoder=False,
            )

        if est_name in {"rf", "random_forest", "randomforest"}:
            rf_cfg = bcfg.get("global_rf", bcfg.get("bucket_rf", {}))
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", None),
                n_jobs=rf_cfg.get("n_jobs", -1),
                random_state=rf_cfg.get("random_state", 42),
            )

        if est_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=bcfg.get("knn_k", 10),
                weights=bcfg.get("knn_weights", "uniform"),
                n_jobs=bcfg.get("knn_jobs", -1),
            )

        if est_name in {"nb", "gnb", "naive_bayes", "naivebayes"}:
            return GaussianNB()

        if est_name in {"logreg", "lr", "logistic", "logistic_regression"}:
            return LogisticRegression(
                max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
            )

        log_info(f"【BTTWD】未知的 global_estimator='{est_name}'，回退到 logreg")
        return LogisticRegression(
            max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
        )

    def _find_model_with_backoff(self, bucket_id: str):
        """逐级回退查找桶模型。"""

        parts = bucket_id.split("|")
        for end in range(len(parts), 0, -1):
            candidate = "|".join(parts[:end])
            model = self.bucket_models.get(candidate)
            if model is not None:
                return model, candidate
        return None, None

    def _get_costs_for_bucket(self, bucket_id: str | None):
        if bucket_id is None:
            return self.costs, False

        if bucket_id in self.costs_per_bucket:
            return self.costs_per_bucket.get(bucket_id, self.costs), True

        if bucket_id.startswith("ROOT|"):
            trimmed = bucket_id[len("ROOT|") :]
            if trimmed in self.costs_per_bucket:
                return self.costs_per_bucket.get(trimmed, self.costs), True

        return self.costs, False

    def _search_thresholds(self, proba: np.ndarray, y_true: np.ndarray, *, bucket_id: str | None = None):
        grid_alpha = self.threshold_grid_alpha or [self.global_alpha]
        grid_beta = self.threshold_grid_beta or [self.global_beta]

        COST_KEYS = ("C_TP", "C_TN", "C_FP", "C_FN", "C_BP", "C_BN")

        costs_raw, is_bucket_specific = self._get_costs_for_bucket(bucket_id)
        costs_to_use = {k: float(costs_raw.get(k, self.costs.get(k))) for k in COST_KEYS}

        if is_bucket_specific:
            missing = [k for k in COST_KEYS if k not in costs_raw]
            if missing and self.bucket_specific_cost_warns < 5:
                log_info(
                    f"【阈值】桶 {bucket_id} 的 bucket-specific cost 缺少 {missing}，将使用全局 cost 补齐"
                )
                self.bucket_specific_cost_warns += 1

        alpha, beta, stats = search_thresholds_with_regret(
            proba,
            y_true,
            alpha_grid=grid_alpha,
            beta_grid=grid_beta,
            costs=costs_to_use,
            gap_min=self.gap_min,
        )
        return alpha, beta, stats, costs_to_use, is_bucket_specific

    def _get_threshold_with_backoff(self, bucket_id: str):
        record = self.bucket_stats.get(bucket_id, {})
        source_override = record.get("parent_with_threshold") or record.get("threshold_source_bucket")
        if source_override and source_override in self.bucket_thresholds:
            return self.bucket_thresholds[source_override], source_override

        parts = bucket_id.split("|")
        for end in range(len(parts), 0, -1):
            candidate = "|".join(parts[:end])
            if candidate in self.bucket_thresholds:
                return self.bucket_thresholds[candidate], candidate
        return (self.global_alpha, self.global_beta), "ROOT"

    def _route_bucket_ids(self, bucket_parts: list[pd.Series]) -> pd.Series:
        if not bucket_parts:
            return pd.Series(dtype="string")

        n = len(bucket_parts[0])
        bucket_ids = pd.Series(["ROOT"] * n, dtype="string")
        for i in range(n):
            current = "ROOT"
            while True:
                plan = self.split_plan.get(current)
                if plan is None:
                    break
                level = plan.get("level", 0)
                if level >= len(bucket_parts):
                    break
                part_value = bucket_parts[level].iat[i]
                allowed = plan.get("allowed_children", set())
                others_part = plan.get("others_part")
                if part_value in allowed:
                    child_part = part_value
                elif others_part is not None:
                    child_part = others_part
                else:
                    break
                next_id = self._join_bucket_id(current, child_part)
                bucket_ids.iat[i] = next_id
                if child_part == others_part:
                    break
                current = next_id
        return bucket_ids

    def _init_bucket_record(self, bucket_id, parent_id, train_idx, val_idx, y):
        return {
            "bucket_id": bucket_id,
            "layer": f"L{len(bucket_id.split('|'))}",
            "parent_bucket_id": parent_id if parent_id is not None else "",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "pos_rate_train": float(y[train_idx].mean()) if len(train_idx) else float("nan"),
            "pos_rate_val": float(y[val_idx].mean()) if len(val_idx) else float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "regret_val": float("nan"),
            "F1_val": float("nan"),
            "Precision_val": float("nan"),
            "Recall_val": float("nan"),
            "BND_ratio_val": float("nan"),
            "pos_coverage_val": float("nan"),
            "use_parent_threshold": False,
            "threshold_n_samples": 0,
        }

    def _calc_bucket_metrics(self, proba: np.ndarray, y_true: np.ndarray) -> dict:
        """
        使用当前全局阈值 alpha/beta 计算桶的结构评估指标（regret, bac）。

        注意：
        - 这里只用于“结构决策”（是否继续细分桶）；
        - 真正三支决策时，每个桶会单独进行阈值搜索，得到自己的 alpha/beta；
        - 因此这里的指标不等同于最终预测阶段使用的桶内阈值。
        """

        preds = np.where(proba >= self.global_alpha, 1, np.where(proba <= self.global_beta, 0, -1))
        regret_val = compute_regret(y_true, preds, self.costs)

        pos_mask = y_true == 1
        neg_mask = ~pos_mask
        tp = np.sum((preds == 1) & pos_mask)
        tn = np.sum((preds == 0) & neg_mask)

        tpr = tp / pos_mask.sum() if pos_mask.sum() > 0 else np.nan
        tnr = tn / neg_mask.sum() if neg_mask.sum() > 0 else np.nan
        if np.isnan(tpr) and np.isnan(tnr):
            bac = np.nan
        elif np.isnan(tpr):
            bac = tnr / 2
        elif np.isnan(tnr):
            bac = tpr / 2
        else:
            bac = 0.5 * (tpr + tnr)

        return {"regret": float(regret_val), "bac": float(bac) if not np.isnan(bac) else np.nan}

    def _join_bucket_id(self, parent_id: str | None, child_part: str) -> str:
        if parent_id in {None, "", "ROOT"}:
            return child_part
        return f"{parent_id}|{child_part}"

    def _build_bucket_tree_carve(
        self, bucket_parts: list[pd.Series], y: np.ndarray
    ) -> tuple[dict, dict, dict, dict]:
        """按最小样本数 + 正类数进行部分 carve-out 分桶，生成 others 结构。"""

        if not bucket_parts:
            return {}, {}, {}, {}

        num_levels = len(bucket_parts)
        n_samples = len(bucket_parts[0])
        leaf_index_map: dict[str, np.ndarray] = {}
        visited_parent: dict[str, str | None] = {"ROOT": None}
        bucket_index_map: dict[str, np.ndarray] = {"ROOT": np.arange(n_samples)}
        children_map: dict[str, list[str]] = {}
        split_plan: dict[str, dict] = {}

        self.bucket_structure_records = []
        root_record = {
            "bucket_id": "ROOT",
            "parent_id": "ROOT",
            "level": 0,
            "split_name": "ROOT",
            "split_type": "ROOT",
            "split_rule": "all",
            "n_samples": int(n_samples),
        }
        self.bucket_structure_records.append(root_record)
        log_bt(
            "创建桶 bucket_id=ROOT，level=0，parent_id=ROOT，split_name=ROOT，split_type=ROOT，"
            f"split_rule=\"all\"，n_samples={n_samples}"
        )

        queue = deque()
        queue.append(("ROOT", 0, np.arange(n_samples)))

        while queue:
            bucket_id, level, idx_all = queue.popleft()

            if level >= num_levels or level >= self.max_levels:
                leaf_index_map[bucket_id] = idx_all
                if level >= self.max_levels:
                    log_bt(
                        f"桶 bucket_id={bucket_id} 已达到最大层数 max_levels={self.max_levels}，不再细分"
                    )
                continue

            part_series = bucket_parts[level]

            if self.stop_split_on_other and any(tok in str(bucket_id) for tok in self.stop_split_labels):
                leaf_index_map[bucket_id] = idx_all
                log_bt(
                    f"桶 bucket_id={bucket_id} 命中 stop_split_labels={sorted(self.stop_split_labels)}，不再细分"
                )
                continue
            values = part_series.iloc[idx_all]
            groups = {cid: idxs.to_numpy() for cid, idxs in values.groupby(values).groups.items()}
            large_values = []
            merged_small = []
            skipped_small = []
            skipped_low_pos = []
            for cid, cidx in groups.items():
                is_small = len(cidx) < self.min_bucket_size
                skip_for_size = self.use_min_bucket_size_limit and is_small
                if skip_for_size:
                    skipped_small.append(cid)
                    continue

                if self.min_pos_per_bucket > 0 and np.sum(y[cidx]) < self.min_pos_per_bucket:
                    skipped_low_pos.append(cid)
                    continue

                if self.use_merge_small_to_residual and is_small:
                    merged_small.append(cid)
                    continue

                large_values.append(cid)

            if len(large_values) == 0:
                leaf_index_map[bucket_id] = idx_all
                reason_parts = []
                if self.use_min_bucket_size_limit and skipped_small:
                    reason_parts.append(
                        f"{len(skipped_small)} 个子桶样本数不足 min_bucket_size={self.min_bucket_size}"
                    )
                if self.use_merge_small_to_residual and merged_small and not skipped_small:
                    reason_parts.append(
                        f"{len(merged_small)} 个子桶按 use_merge_small_to_residual 合并到 residual"
                    )
                if skipped_low_pos:
                    reason_parts.append(
                        f"{len(skipped_low_pos)} 个子桶正样本不足 min_pos_per_bucket={self.min_pos_per_bucket}"
                    )
                reason_text = "；".join(reason_parts) or "未找到有效子桶"
                log_bt(f"桶 bucket_id={bucket_id} 不再细分：{reason_text}")
                continue

            level_name = next(iter(groups.keys())).split("=")[0] if groups else f"L{level + 1}"
            level_cfg = self.bucket_tree.levels_cfg[level] if level < len(self.bucket_tree.levels_cfg) else {}
            split_type = level_cfg.get("type", "")

            # 构造“小桶”索引列表：不在 large_values 里的子桶统统归为 residual
            if groups:
                residual_list = [groups[cid] for cid in groups if cid not in large_values]
                if residual_list:
                    residual = np.concatenate(residual_list)
                else:
                    residual = np.array([], dtype=int)
            else:
                residual = np.array([], dtype=int)

            has_residual = residual.size > 0

            child_ids = []
            for cid in large_values:
                child_id = self._join_bucket_id(bucket_id, cid)
                visited_parent[child_id] = bucket_id
                bucket_index_map[child_id] = groups[cid]
                child_ids.append(child_id)

                split_rule = cid.split("=", 1)[1] if "=" in cid else cid
                self.bucket_structure_records.append(
                    {
                        "bucket_id": child_id,
                        "parent_id": bucket_id,
                        "level": level + 1,
                        "split_name": level_name,
                        "split_type": split_type,
                        "split_rule": split_rule,
                        "n_samples": int(len(groups[cid])),
                    }
                )
                log_bt(
                    "创建桶 "
                    f"bucket_id={child_id}，level={level + 1}，parent_id={bucket_id}，"
                    f"split_name={level_name}，split_type={split_type or '-'}，split_rule=\"{split_rule}\"，"
                    f"n_samples={len(groups[cid])}"
                )

            others_part = None
            if has_residual:
                others_part = f"{level_name}=others"
                others_id = self._join_bucket_id(bucket_id, others_part)
                visited_parent[others_id] = bucket_id
                bucket_index_map[others_id] = residual
                leaf_index_map[others_id] = residual
                child_ids.append(others_id)

                self.bucket_structure_records.append(
                    {
                        "bucket_id": others_id,
                        "parent_id": bucket_id,
                        "level": level + 1,
                        "split_name": level_name,
                        "split_type": split_type,
                        "split_rule": "others",
                        "n_samples": int(len(residual)),
                    }
                )
                log_bt(
                    "创建桶 "
                    f"bucket_id={others_id}，level={level + 1}，parent_id={bucket_id}，"
                    f"split_name={level_name}，split_type={split_type or '-'}，split_rule=\"others\"，"
                    f"n_samples={len(residual)}"
                )

            children_map[bucket_id] = child_ids
            split_plan[bucket_id] = {
                "level": level,
                "allowed_children": set(large_values),
                "others_part": others_part,
            }

            next_level = level + 1
            for cid in large_values:
                child_id = self._join_bucket_id(bucket_id, cid)
                if next_level >= self.max_levels:
                    leaf_index_map[child_id] = groups[cid]
                else:
                    queue.append((child_id, next_level, groups[cid]))

        self.split_plan = split_plan
        self.children_map = children_map
        return leaf_index_map, visited_parent, bucket_index_map, children_map

    def _build_bucket_tree(self, bucket_parts: list[pd.Series], y: np.ndarray):
        return self._build_bucket_tree_carve(bucket_parts, y)

    def fit(self, X: np.ndarray, y: np.ndarray, X_df_for_bucket: pd.DataFrame):
        score_mode = self.score_cfg.get("bucket_score_mode")
        if str(score_mode).lower() == "f1_regret_bnd":
            log_info(
                "[BT] 使用桶评分配置：mode="
                f"{score_mode}, f1_weight={self.score_cfg.get('f1_weight')}, "
                f"regret_weight={self.score_cfg.get('regret_weight')}, bnd_weight={self.score_cfg.get('bnd_weight')}"
            )
        else:
            log_info(
                "[BT] 使用桶评分配置：mode="
                f"{score_mode}, bac_weight={self.score_cfg.get('bac_weight')}, "
                f"regret_weight={self.score_cfg.get('regret_weight')}, regret_sign={self.score_cfg.get('regret_sign')}"
            )
        # Step 0: 划分 inner 训练/验证集
        if self.val_ratio > 0:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_ratio, random_state=self.random_state
            )
            inner_train_idx, inner_val_idx = next(sss.split(X, y))
        else:
            inner_train_idx = np.arange(len(y))
            inner_val_idx = np.array([], dtype=int)

        self.bucket_estimator = self._build_bucket_estimator()

        inner_train_idx = np.asarray(inner_train_idx)
        inner_val_idx = np.asarray(inner_val_idx)

        self.global_pos_rate = float(np.mean(y))

        # Step 1: 预生成桶ID（逐层标签）
        bucket_parts = self.bucket_tree.assign_bucket_parts(X_df_for_bucket)

        self.bucket_stats = {}
        self.threshold_logs = []

        # Step 2: 训练全局模型 + 阈值
        X_train_inner = X[inner_train_idx]
        y_train_inner = y[inner_train_idx]

        self.global_model = self._build_global_estimator()
        if self.global_model is not None:
            self.global_model.fit(X_train_inner, y_train_inner)
            log_info("【BTTWD】全局模型训练完成，用于兜底预测")
        else:
            log_info("【BTTWD】global_estimator=none：仅使用全局正类比例作为概率")

        if self.optimize_thresholds and len(inner_val_idx) > 0:
            X_val_inner = X[inner_val_idx]
            y_val_inner = y[inner_val_idx]
            if self.global_model is None:
                proba_val_inner = np.full(len(y_val_inner), self.global_pos_rate)
            else:
                proba_val_inner = self.global_model.predict_proba(X_val_inner)[:, 1]
            (
                self.global_alpha,
                self.global_beta,
                _,
                _,
                _,
            ) = self._search_thresholds(proba_val_inner, y_val_inner, bucket_id="ROOT")

        if self.global_model is None:
            proba_all = np.full(len(y), self.global_pos_rate)
        else:
            proba_all = self.global_model.predict_proba(X)[:, 1]

        # Step 3: 构建桶树（carve-out + others）
        leaf_index_map, visited_parent, bucket_index_map, children_map = self._build_bucket_tree(bucket_parts, y)
        bucket_ids = self._route_bucket_ids(bucket_parts)

        if bucket_index_map:
            total_buckets = len(bucket_index_map)
            max_level = max(len(b.split("|")) for b in bucket_index_map.keys()) if bucket_index_map else 0
            leaf_count = len(leaf_index_map)
            log_bt(
                f"桶树构建完成：总桶数={total_buckets}，最大层数={max_level}，叶子桶数={leaf_count}"
            )

        self.bucket_info = {}
        bucket_data = {}
        for bucket_id, idx_all in bucket_index_map.items():
            parent_id = visited_parent.get(bucket_id)
            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)
            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            bucket_data[bucket_id] = {
                "all": idx_all,
                "train": train_idx_bucket,
                "val": val_idx_bucket,
                "parent": parent_id,
                "children": children_map.get(bucket_id, []),
                "is_others": str(bucket_id).endswith("=others"),
                "level": 0 if bucket_id == "ROOT" else len(bucket_id.split("|")),
            }

            self.bucket_stats[bucket_id] = {
                **self._init_bucket_record(bucket_id, parent_id, train_idx_bucket, val_idx_bucket, y),
                "n_all": int(len(idx_all)),
                "pos_rate_all": float(y[idx_all].mean()) if len(idx_all) else float("nan"),
            }

        # Step 4: 计算 parent-share 样本
        parent_train_map = defaultdict(list)
        parent_val_map = defaultdict(list)
        for parent_id, child_ids in children_map.items():
            if not self.use_parent_share_rate:
                all_child_train = (
                    np.concatenate([bucket_data[c]["train"] for c in child_ids]) if child_ids else np.array([], dtype=int)
                )
                all_child_val = (
                    np.concatenate([bucket_data[c]["val"] for c in child_ids]) if child_ids else np.array([], dtype=int)
                )
                parent_train_map[parent_id] = all_child_train.tolist()
                parent_val_map[parent_id] = all_child_val.tolist()
            else:
                for child_id in child_ids:
                    child_train = bucket_data[child_id]["train"]
                    child_val = bucket_data[child_id]["val"]

                    share_rate = 1.0 if len(child_train) <= self.small_bucket_threshold else self.parent_share_rate
                    n_share_train = min(len(child_train), int(len(child_train) * share_rate))
                    if n_share_train > 0:
                        share_idx = self.rng.choice(child_train, size=n_share_train, replace=False)
                        parent_train_map[parent_id].extend(share_idx.tolist())

                    share_rate_val = 1.0 if len(child_val) <= self.small_bucket_threshold else self.parent_share_rate
                    n_share_val = min(len(child_val), int(len(child_val) * share_rate_val))
                    if n_share_val > 0:
                        share_val_idx = self.rng.choice(child_val, size=n_share_val, replace=False)
                        parent_val_map[parent_id].extend(share_val_idx.tolist())

                if len(parent_train_map[parent_id]) < self.min_parent_share:
                    all_child_train = (
                        np.concatenate([bucket_data[c]["train"] for c in child_ids])
                        if child_ids
                        else np.array([], dtype=int)
                    )
                    parent_train_map[parent_id] = all_child_train.tolist()
                if len(parent_val_map[parent_id]) < self.min_parent_share:
                    all_child_val = (
                        np.concatenate([bucket_data[c]["val"] for c in child_ids])
                        if child_ids
                        else np.array([], dtype=int)
                    )
                    parent_val_map[parent_id] = all_child_val.tolist()

            bucket_data[parent_id]["train_share"] = np.array(parent_train_map[parent_id], dtype=int)
            bucket_data[parent_id]["val_share"] = np.array(parent_val_map[parent_id], dtype=int)

        # Step 5: 训练所有桶（包含内部节点）并判定强弱
        self.bucket_models = {}
        self.bucket_thresholds = {}
        bucket_scores = {}

        if self.bucket_estimator is None:
            log_info("【BTTWD】bucket_estimator=none：不训练桶内局部模型，仅使用全局模型概率做桶内阈值搜索")

        if len(inner_val_idx) > 0:
            global_metrics = self._calc_bucket_metrics(proba_all[inner_val_idx], y[inner_val_idx])
            global_score = compute_bucket_score(global_metrics, self.score_cfg)
        else:
            global_score = None

        def _get_bucket_dataset(bid: str):
            data = bucket_data.get(bid, {})
            if data.get("children"):
                train_idx = data.get("train_share", data.get("train", np.array([], dtype=int)))
                val_idx = data.get("val_share", data.get("val", np.array([], dtype=int)))
            else:
                train_idx = data.get("train", np.array([], dtype=int))
                val_idx = data.get("val", np.array([], dtype=int))
            return train_idx, val_idx

        ordered_buckets = sorted(bucket_data.keys(), key=lambda b: bucket_data[b]["level"])
        for bucket_id in ordered_buckets:
            data = bucket_data[bucket_id]
            parent_id = data.get("parent")

            all_idx = data.get("all", np.array([], dtype=int))
            if all_idx.size == 0:
                self.bucket_info[bucket_id] = {
                    "n_samples": 0,
                    "parent_bucket_id": parent_id,
                    "status": "weak",
                    "gain_like": None,
                }

                record = self.bucket_stats.get(bucket_id)
                if record is None:
                    record = self._init_bucket_record(
                        bucket_id,
                        parent_id,
                        np.array([], dtype=int),
                        np.array([], dtype=int),
                        y,
                    )
                    record["n_all"] = 0
                    record["pos_rate_all"] = float("nan")
                    self.bucket_stats[bucket_id] = record

                record["use_parent_threshold"] = True
                continue

            train_idx, val_idx = _get_bucket_dataset(bucket_id)
            y_train_bucket = y[train_idx]
            y_val_bucket = y[val_idx]

            record = self.bucket_stats.get(bucket_id)
            record["n_train"] = int(len(train_idx))
            record["n_val"] = int(len(val_idx))
            record["pos_rate_train"] = float(y[train_idx].mean()) if len(train_idx) else float("nan")
            record["pos_rate_val"] = float(y[val_idx].mean()) if len(val_idx) else float("nan")

            is_others = data.get("is_others", False)
            model = None
            if not is_others and self.bucket_estimator is not None and len(np.unique(y_train_bucket)) >= 2 and len(y_train_bucket) > 0:
                model = self._build_bucket_estimator()
                X_train_bucket = X[train_idx]
                X_train_bucket, y_train_bucket = self._sample_bucket_data(X_train_bucket, y_train_bucket, bucket_id)
                model.fit(X_train_bucket, y_train_bucket)
                self.bucket_models[bucket_id] = model

            enough_val = (
                len(val_idx) >= self.min_val_samples_per_bucket
                and np.unique(y_val_bucket).size >= 2
                and not is_others
            )
            bucket_score = None
            alpha = beta = None
            stats = {
                "regret": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "bnd_ratio": float("nan"),
                "pos_coverage": float("nan"),
                "bac": float("nan"),
                "auc": float("nan"),
                "n_samples": 0,
            }
            costs_used = self.costs
            used_bucket_specific = False
            threshold_data_source = "val" if enough_val else "all"

            if enough_val:
                if model is None:
                    if self.global_model is not None:
                        proba_val = self.global_model.predict_proba(X[val_idx])[:, 1]
                    else:
                        proba_val = np.full(len(val_idx), self.global_pos_rate)
                else:
                    proba_val = model.predict_proba(X[val_idx])[:, 1]
                alpha, beta, stats, costs_used, used_bucket_specific = self._search_thresholds(
                    proba_val, y_val_bucket, bucket_id=bucket_id
                )
                if self.use_gain_weak_backoff:
                    score_metrics = {
                        "regret": stats.get("regret", np.nan),
                        "bac": stats.get("bac", np.nan),
                        "f1": stats.get("f1", np.nan),
                        "BND_ratio": stats.get("bnd_ratio", np.nan),
                    }
                    bucket_score = compute_bucket_score(score_metrics, self.score_cfg)
            else:
                bucket_idx = np.asarray(data.get("all", []), dtype=int)
                y_bucket = y[bucket_idx]
                stats["n_samples"] = int(len(bucket_idx))

                if len(bucket_idx) > 0:
                    if np.unique(y_bucket).size >= 2:
                        if model is None:
                            if self.global_model is not None:
                                proba_bucket = self.global_model.predict_proba(X[bucket_idx])[:, 1]
                            else:
                                proba_bucket = np.full(len(bucket_idx), self.global_pos_rate)
                        else:
                            proba_bucket = model.predict_proba(X[bucket_idx])[:, 1]
                        alpha, beta, stats, costs_used, used_bucket_specific = self._search_thresholds(
                            proba_bucket, y_bucket, bucket_id=bucket_id
                        )
                    else:
                        unique_label = np.unique(y_bucket)
                        only_label = int(unique_label[0]) if len(unique_label) else 0
                        alpha, beta = (1.0, 1.0) if only_label == 0 else (0.0, 0.0)
                else:
                    alpha, beta = self.global_alpha, self.global_beta
                    threshold_data_source = "global_default"

            parent_score = None
            if self.use_gain_weak_backoff:
                parent_score = bucket_scores.get(parent_id) if parent_id in bucket_scores else global_score

            gain_like = None
            if self.use_gain_weak_backoff and bucket_score is not None and parent_score is not None:
                gain_like = bucket_score - parent_score

            weak_due_to_gain = (
                self.use_gain_weak_backoff
                and gain_like is not None
                and gain_like < self.min_gain_for_split
            )
            weak_due_to_eval = not enough_val

            if not self.use_gain_weak_backoff:
                status = "strong"
                use_parent_threshold = False
                threshold_source_bucket = bucket_id
            else:
                status = "strong"
                if weak_due_to_eval or weak_due_to_gain:
                    status = "weak"

                threshold_source_bucket = bucket_id if status == "strong" else (parent_id if parent_id is not None else "ROOT")
                use_parent_threshold = status == "weak"

            self.bucket_info[bucket_id] = {
                "n_samples": int(len(data.get("all", []))),
                "parent_bucket_id": parent_id,
                "status": status,
                "gain_like": gain_like,
                "bucket_score": bucket_score,
                "parent_score": parent_score,
                "use_gain_weak_backoff": self.use_gain_weak_backoff,
                "effective_bucket_id": threshold_source_bucket,
                "threshold_data_source": threshold_data_source,
            }
            if self.use_gain_weak_backoff and bucket_score is not None:
                bucket_scores[bucket_id] = bucket_score

            record.update(
                {
                    "alpha": float(alpha) if alpha is not None else float("nan"),
                    "beta": float(beta) if beta is not None else float("nan"),
                    "regret_val": float(stats.get("regret", np.nan)),
                    "F1_val": float(stats.get("f1", np.nan)),
                    "Precision_val": float(stats.get("precision", np.nan)),
                    "Recall_val": float(stats.get("recall", np.nan)),
                    "BND_ratio_val": float(stats.get("bnd_ratio", np.nan)),
                    "pos_coverage_val": float(stats.get("pos_coverage", np.nan)),
                    "threshold_n_samples": int(stats.get("n_samples", 0)),
                    "use_parent_threshold": use_parent_threshold,
                    "BAC_val": float(stats.get("bac", np.nan)),
                    "AUC_val": float(stats.get("auc", np.nan)),
                    "cost_source": "bucket_specific" if used_bucket_specific else "global_default",
                    "score_metric": self.score_cfg.get("bucket_score_mode"),
                    "score_value": float(bucket_score) if bucket_score is not None else float("nan"),
                    "parent_score_value": float(parent_score) if parent_score is not None else float("nan"),
                    "gain_value": float(gain_like) if gain_like is not None else float("nan"),
                    "is_weak": status == "weak",
                    "threshold_source_bucket": threshold_source_bucket,
                    "use_gain_weak_backoff": self.use_gain_weak_backoff,
                    "threshold_data_source": threshold_data_source,
                }
            )

            if status == "strong" and alpha is not None and beta is not None:
                self.bucket_thresholds[bucket_id] = (alpha, beta)
                log_info(
                    f"【阈值】桶 {bucket_id}（n_val={len(val_idx)}，source={threshold_data_source}) 使用本地阈值 α={alpha:.4f}, β={beta:.4f}",
                )
            elif status == "weak":
                log_info(
                    f"【阈值】桶 {bucket_id} 标记为弱桶，阈值将回退使用 {threshold_source_bucket} 的阈值"
                )

            log_bt(
                f"桶 bucket_id={bucket_id} level={data.get('level')}：\n"
                f"    n_train={len(train_idx)}, n_val={len(val_idx)},\n"
                f"    BAC={stats.get('bac', float('nan')):.3f}, F1={stats.get('f1', float('nan')):.3f}, "
                f"AUC={stats.get('auc', float('nan')):.3f},\n"
                f"    Regret={stats.get('regret', float('nan')):.3f}, BND_ratio={stats.get('bnd_ratio', float('nan')):.3f}, "
                f"POS_coverage={stats.get('pos_coverage', float('nan')):.3f},\n"
                f"    Score({self.score_cfg.get('bucket_score_mode')} )={bucket_score if bucket_score is not None else float('nan'):.3f}"
                f"，threshold_source={threshold_data_source}"
            )

            if parent_id is not None:
                log_bt(
                    f"桶 bucket_id={bucket_id}：\n"
                    f"    parent_id={parent_id}，parent_Score={(parent_score if parent_score is not None else float('nan')):.3f}, "
                    f"bucket_Score={(bucket_score if bucket_score is not None else float('nan')):.3f},\n"
                    f"    Gain={(gain_like if gain_like is not None else float('nan')):+.3f}, is_weak={status == 'weak'}"
                )

        if "ROOT" in self.bucket_info:
            self.bucket_info["ROOT"]["status"] = "strong"

        for bucket_id, data in bucket_data.items():
            record = self.bucket_stats.get(bucket_id)
            if record is None:
                continue
            train_idx, val_idx = _get_bucket_dataset(bucket_id)
            record["n_train"] = int(len(train_idx))
            record["n_val"] = int(len(val_idx))
            record["pos_rate_train"] = float(y[train_idx].mean()) if len(train_idx) else float("nan")
            record["pos_rate_val"] = float(y[val_idx].mean()) if len(val_idx) else float("nan")

        # 对未单独训练或标记为弱桶的桶补充阈值（继承父桶或全局阈值）
        self.bucket_thresholds.setdefault("ROOT", (self.global_alpha, self.global_beta))
        for bucket_id, info in self.bucket_info.items():
            parent_id = info.get("parent_bucket_id")
            status = info.get("status")
            record = self.bucket_stats.get(bucket_id)
            if record is None:
                record = self._init_bucket_record(
                    bucket_id,
                    parent_id,
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    y,
                )
                record["n_all"] = int(info.get("n_samples", 0))
                record["pos_rate_all"] = float("nan")
                self.bucket_stats[bucket_id] = record

            source_bucket = record.get("threshold_source_bucket") or (
                bucket_id if status != "weak" else (parent_id if parent_id is not None else "ROOT")
            )

            if status == "weak":
                ancestor = parent_id
                while ancestor:
                    ancestor_info = self.bucket_info.get(ancestor, {})
                    if ancestor in self.bucket_thresholds and ancestor_info.get("status") != "weak":
                        source_bucket = ancestor
                        break
                    ancestor = ancestor_info.get("parent_bucket_id")
                if source_bucket == bucket_id:
                    source_bucket = parent_id if parent_id is not None else "ROOT"
            elif status == "strong" and bucket_id in self.bucket_thresholds:
                source_bucket = bucket_id

            if source_bucket not in self.bucket_thresholds:
                ancestor = info.get("parent_bucket_id")
                while ancestor:
                    ancestor_info = self.bucket_info.get(ancestor, {})
                    if ancestor in self.bucket_thresholds and ancestor_info.get("status") != "weak":
                        source_bucket = ancestor
                        break
                    ancestor = ancestor_info.get("parent_bucket_id")

            if source_bucket not in self.bucket_thresholds:
                source_bucket = "ROOT"

            if source_bucket not in self.bucket_thresholds:
                self.bucket_thresholds[source_bucket] = (self.global_alpha, self.global_beta)

            effective_threshold = self.bucket_thresholds.get(source_bucket, (self.global_alpha, self.global_beta))
            self.bucket_thresholds[bucket_id] = effective_threshold

            record["alpha"] = float(effective_threshold[0])
            record["beta"] = float(effective_threshold[1])
            record["threshold_n_samples"] = record.get("threshold_n_samples", 0)
            record["use_parent_threshold"] = source_bucket != bucket_id
            record["parent_with_threshold"] = source_bucket if source_bucket not in {bucket_id, "ROOT"} else ""
            record["threshold_source_bucket"] = source_bucket
            info["effective_bucket_id"] = source_bucket or "ROOT"

            if source_bucket != bucket_id:
                log_bt(
                    f"桶 bucket_id={bucket_id} 样本不足或为弱桶，未单独搜索阈值，继承桶 {source_bucket} 的阈值"
                    f"(alpha={record['alpha']:.4f},beta={record['beta']:.4f})"
                )

        # 汇总阈值日志
        self.threshold_logs = []
        for record in self.bucket_stats.values():
            self.threshold_logs.append(
                {
                    "bucket_id": record.get("bucket_id"),
                    "layer": record.get("layer"),
                    "parent_bucket_id": record.get("parent_bucket_id", ""),
                    "n_train": record.get("n_train", 0),
                    "n_val": record.get("n_val", 0),
                    "pos_rate_train": record.get("pos_rate_train"),
                    "pos_rate_val": record.get("pos_rate_val"),
                    "alpha": record.get("alpha"),
                    "beta": record.get("beta"),
                    "regret_val": record.get("regret_val"),
                    "F1_val": record.get("F1_val"),
                    "Precision_val": record.get("Precision_val"),
                    "Recall_val": record.get("Recall_val"),
                    "BND_ratio_val": record.get("BND_ratio_val"),
                    "pos_coverage_val": record.get("pos_coverage_val"),
                    "threshold_n_samples": record.get("threshold_n_samples", 0),
                    "use_parent_threshold": record.get("use_parent_threshold", False),
                    "parent_with_threshold": record.get("parent_with_threshold", ""),
                    "score_metric": record.get("score_metric"),
                    "score_value": record.get("score_value"),
                    "parent_score_value": record.get("parent_score_value"),
                    "gain_value": record.get("gain_value"),
                    "threshold_source_bucket": record.get("threshold_source_bucket", ""),
                    "is_weak": record.get("is_weak", False),
                }
            )

        stats_df = self.get_bucket_stats()
        if not stats_df.empty:
            n_total = len(stats_df)
            n_non_empty = int((stats_df["n_all"] > 0).sum())
            n_strong = sum(info.get("status") == "strong" for info in self.bucket_info.values())
            n_weak = sum(info.get("status") == "weak" for info in self.bucket_info.values())
            log_info(
                f"【BTTWD】桶统计摘要：总桶数={n_total}，非空桶={n_non_empty}，强桶={n_strong}，弱桶={n_weak}"
            )

        log_info(
            "【BTTWD】共生成 "
            f"{bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（已训练局部模型）"
        )

        if self.cfg.get("OUTPUT", {}).get("export_bucket_reports_on_fit", True):
            self._export_bucket_reports()

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_parts = self.bucket_tree.assign_bucket_parts(X_df_for_bucket)
        bucket_ids = self._route_bucket_ids(bucket_parts)
        proba = np.zeros(len(X))

        if self.bucket_estimator is None:
            if self.global_model is not None:
                proba = self.global_model.predict_proba(X)[:, 1]
            else:
                proba.fill(self.global_pos_rate)
            return proba

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            model, _ = self._find_model_with_backoff(bucket_id)

            if model is None:
                if self.use_global_backoff and self.global_model is not None:
                    proba[list(idxs)] = self.global_model.predict_proba(X[list(idxs)])[:, 1]
                else:
                    proba[list(idxs)] = self.global_pos_rate
                continue

            proba[list(idxs)] = model.predict_proba(X[list(idxs)])[:, 1]
        return proba

    def predict(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X, X_df_for_bucket)
        bucket_parts = self.bucket_tree.assign_bucket_parts(X_df_for_bucket)
        bucket_ids = self._route_bucket_ids(bucket_parts)

        preds = np.zeros(len(proba))
        self.fallback_stats = {}

        def _ensure_fallback_record(bid: str):
            if bid not in self.fallback_stats:
                parent_id = self.bucket_info.get(bid, {}).get("parent_bucket_id")
                self.fallback_stats[bid] = {
                    "bucket_id": bid,
                    "level": 0 if bid == "ROOT" else len(bid.split("|")),
                    "assigned_samples": 0,
                    "used_local_decision": 0,
                    "fallback_to_parent": 0,
                    "fallback_from_children": 0,
                    "parent_id": parent_id if parent_id is not None else "ROOT",
                    "is_weak": self.bucket_info.get(bid, {}).get("status") == "weak",
                    "effective_bucket_id": self.bucket_info.get(bid, {}).get("effective_bucket_id", bid),
                }
            return self.fallback_stats[bid]

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            idx_list = list(idxs)
            (alpha, beta), source_bucket = self._get_threshold_with_backoff(bucket_id)
            effective_bucket = source_bucket if source_bucket is not None else "ROOT"

            bucket_proba = proba[idx_list]
            bucket_pred = np.where(bucket_proba >= alpha, 1, np.where(bucket_proba <= beta, 0, -1))
            preds[idx_list] = bucket_pred

            record = _ensure_fallback_record(bucket_id)
            record["assigned_samples"] += len(idx_list)

            target_record = _ensure_fallback_record(effective_bucket)
            use_local = effective_bucket == bucket_id and not record.get("is_weak", False)
            if use_local:
                record["used_local_decision"] += len(idx_list)
            else:
                record["fallback_to_parent"] += len(idx_list)
                target_record["fallback_from_children"] += len(idx_list)
                target_record["used_local_decision"] += len(idx_list)
                record["effective_bucket_id"] = effective_bucket

        for bid, rec in self.fallback_stats.items():
            log_bt(
                f"预测统计：bucket_id={bid}：\n"
                f"    assigned_samples={rec.get('assigned_samples', 0)},\n"
                f"    used_local_decision={rec.get('used_local_decision', 0)},\n"
                f"    fallback_to_parent={rec.get('fallback_to_parent', 0)},\n"
                f"    parent_id={rec.get('parent_id')},\n"
                f"    is_weak={rec.get('is_weak')}"
            )

        total_samples = len(proba)
        leaf_decisions = sum(
            rec.get("used_local_decision", 0)
            for bid, rec in self.fallback_stats.items()
            if not self.children_map.get(bid, [])
        )
        middle_decisions = sum(
            rec.get("used_local_decision", 0)
            for bid, rec in self.fallback_stats.items()
            if self.children_map.get(bid, []) and bid != "ROOT"
        )
        global_decisions = self.fallback_stats.get("ROOT", {}).get("used_local_decision", 0)

        if total_samples > 0:
            log_bt(
                "回退汇总：\n"
                f"    总样本数={total_samples},\n"
                f"    使用叶子桶决策的样本比例={leaf_decisions / total_samples:.3%},\n"
                f"    使用中间桶决策的样本比例={middle_decisions / total_samples:.3%},\n"
                f"    回退到全局桶决策的样本比例={global_decisions / total_samples:.3%}"
            )

        self._export_fallback_stats()
        return preds

    def get_bucket_stats(self) -> pd.DataFrame:
        if not self.bucket_stats:
            return pd.DataFrame()
        df = pd.DataFrame(self.bucket_stats.values())
        sort_col = "n_all" if "n_all" in df.columns else None
        if sort_col:
            df = df.sort_values(by=sort_col, ascending=False)
        return df

    def update_test_stats(self, test_bucket_df: pd.DataFrame | None = None) -> None:
        """Update stored bucket statistics with test-set counts/metrics."""

        if test_bucket_df is None or test_bucket_df.empty:
            return

        for _, row in test_bucket_df.iterrows():
            bucket_id = row.get("bucket_id")
            if pd.isna(bucket_id):
                continue

            record = self.bucket_stats.get(bucket_id)
            if record is None:
                continue

            n_test = row.get("n_test")
            if n_test is not None and not pd.isna(n_test):
                record["n_test"] = int(n_test)

            for key in ("pos_rate_test", "BND_ratio_test", "POS_Coverage_test", "regret_test"):
                value = row.get(key)
                if value is not None and not pd.isna(value):
                    record[key] = float(value)

    def get_threshold_logs(self) -> pd.DataFrame:
        if not self.threshold_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.threshold_logs)

    def _export_bucket_reports(self, fold: int | str | None = None, append_tree: bool = False) -> None:
        if not self.bucket_stats:
            return

        structure_rows = []
        seen = set()
        for rec in self.bucket_structure_records:
            bid = rec.get("bucket_id")
            seen.add(bid)
            stat = self.bucket_stats.get(bid, {})
            info = self.bucket_info.get(bid, {})
            children = self.children_map.get(bid, [])
            structure_rows.append(
                {
                    "bucket_id": bid,
                    "parent_id": rec.get("parent_id", "ROOT"),
                    "level": rec.get("level", 0),
                    "split_name": rec.get("split_name", ""),
                    "split_type": rec.get("split_type", ""),
                    "split_rule": rec.get("split_rule", ""),
                    "n_samples_total": int(stat.get("n_all", rec.get("n_samples", 0) or 0)),
                    "n_train": int(stat.get("n_train", 0)),
                    "n_val": int(stat.get("n_val", 0)),
                    "n_test": stat.get("n_test", np.nan),
                    "is_leaf": len(children) == 0,
                    "is_weak": info.get("status") == "weak",
                    "effective_bucket_id": info.get("effective_bucket_id", bid),
                }
            )

        for bid, stat in self.bucket_stats.items():
            if bid in seen:
                continue
            info = self.bucket_info.get(bid, {})
            children = self.children_map.get(bid, [])
            structure_rows.append(
                {
                    "bucket_id": bid,
                    "parent_id": stat.get("parent_bucket_id", ""),
                    "level": 0 if bid == "ROOT" else len(str(bid).split("|")),
                    "split_name": "",
                    "split_type": "",
                    "split_rule": "",
                    "n_samples_total": int(stat.get("n_all", 0)),
                    "n_train": int(stat.get("n_train", 0)),
                    "n_val": int(stat.get("n_val", 0)),
                    "n_test": stat.get("n_test", np.nan),
                    "is_leaf": len(children) == 0,
                    "is_weak": info.get("status") == "weak",
                    "effective_bucket_id": info.get("effective_bucket_id", bid),
                }
            )

        structure_df = pd.DataFrame(structure_rows)
        structure_path = self.results_dir / "bucket_tree_structure.csv"
        if fold is not None:
            structure_df.insert(0, "fold", fold)

        if append_tree:
            structure_df.to_csv(
                structure_path,
                mode="a",
                header=not structure_path.exists(),
                index=False,
            )
        else:
            structure_df.to_csv(structure_path, index=False)

        metrics_rows = []
        for bid, stat in self.bucket_stats.items():
            metrics_rows.append(
                {
                    "bucket_id": bid,
                    "parent_id": stat.get("parent_bucket_id", ""),
                    "level": 0 if bid == "ROOT" else len(str(bid).split("|")),
                    "n_train": stat.get("n_train", 0),
                    "n_val": stat.get("n_val", 0),
                    "BAC": stat.get("BAC_val"),
                    "F1": stat.get("F1_val"),
                    "AUC": stat.get("AUC_val"),
                    "Regret": stat.get("regret_val"),
                    "BND_ratio": stat.get("BND_ratio_val"),
                    "POS_coverage": stat.get("pos_coverage_val"),
                    "score_metric": stat.get("score_metric"),
                    "score_value": stat.get("score_value"),
                    "parent_score_value": stat.get("parent_score_value"),
                    "gain_value": stat.get("gain_value"),
                    "is_weak": stat.get("is_weak", False),
                    "threshold_source_bucket": stat.get("threshold_source_bucket")
                    or stat.get("parent_with_threshold")
                    or bid,
                }
            )

        pd.DataFrame(metrics_rows).to_csv(self.results_dir / "bucket_metrics_gain.csv", index=False)

        threshold_rows = []
        for bid, thresh in self.bucket_thresholds.items():
            stat = self.bucket_stats.get(bid, {})
            threshold_rows.append(
                {
                    "bucket_id": bid,
                    "alpha": float(thresh[0]),
                    "beta": float(thresh[1]),
                    "threshold_mode": self.threshold_mode,
                    "threshold_source_bucket": stat.get("threshold_source_bucket")
                    or stat.get("parent_with_threshold")
                    or bid,
                    "is_weak": stat.get("is_weak", False),
                }
            )

        pd.DataFrame(threshold_rows).to_csv(self.results_dir / "bucket_thresholds.csv", index=False)

    def _export_fallback_stats(self) -> None:
        if not self.fallback_stats:
            return

        rows = []
        for bid, rec in self.fallback_stats.items():
            rows.append(
                {
                    "bucket_id": bid,
                    "level": rec.get("level", 0),
                    "assigned_samples": rec.get("assigned_samples", 0),
                    "used_local_decision": rec.get("used_local_decision", 0),
                    "fallback_to_parent": rec.get("fallback_to_parent", 0),
                    "parent_id": rec.get("parent_id", "ROOT"),
                    "is_weak": rec.get("is_weak", False),
                    "effective_bucket_id": rec.get("effective_bucket_id", bid),
                    "fallback_from_children": rec.get("fallback_from_children", 0),
                }
            )

        pd.DataFrame(rows).to_csv(self.results_dir / "bucket_fallback_stats.csv", index=False)
