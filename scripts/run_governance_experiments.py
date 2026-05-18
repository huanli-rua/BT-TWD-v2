"""运行第二篇 governance 实验框架。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib.bucket.routing import bucket_level, bucket_path_to_root  # noqa: E402
from bttwdlib.bucket.structure_guard import assert_no_bsm_dependency  # noqa: E402
from bttwdlib.bucket.tree import BucketTree  # noqa: E402
from bttwdlib.core.data import load_dataset  # noqa: E402
from bttwdlib.core.io import ensure_dir, load_yaml  # noqa: E402
from bttwdlib.core.metrics import compute_binary_metrics  # noqa: E402
from bttwdlib.core.posterior import GlobalPosterior  # noqa: E402
from bttwdlib.core.preprocess import prepare_features_and_labels  # noqa: E402
from bttwdlib.core.seed import set_global_seed  # noqa: E402
from bttwdlib.core.split import make_stratified_kfold, split_train_validation  # noqa: E402
from bttwdlib.governance.conformal import SplitConformalValidator  # noqa: E402
from bttwdlib.governance.defer_lifecycle import resolve_deferred_sample  # noqa: E402
from bttwdlib.governance.post_validation import validate_post_decision  # noqa: E402
from bttwdlib.governance.records import build_sample_record  # noqa: E402
from bttwdlib.run_experiment import _resolve_data_paths  # noqa: E402
from bttwdlib.threshold_search import compute_regret, search_thresholds_with_regret  # noqa: E402
from bttwdlib.twd.decision import decide_twd, decision_to_numeric, risk_values_for_probability  # noqa: E402
from bttwdlib.utils_logging import log_info  # noqa: E402


def _dataset_items(config_path: Path) -> list[tuple[str, Path]]:
    cfg = load_yaml(config_path)
    datasets = cfg.get("datasets")
    if not datasets:
        return [(config_path.stem, config_path)]
    items = []
    for item in datasets:
        if isinstance(item, str):
            name = Path(item).stem
            rel_path = item
        else:
            name = item.get("name") or Path(item["config"]).stem
            rel_path = item["config"]
        path = Path(rel_path)
        if not path.is_absolute():
            candidate = config_path.parent / path
            path = candidate if candidate.exists() else REPO_ROOT / path
        items.append((name, path))
    return items


def _build_bucket_df(df_raw: pd.DataFrame, cfg: dict):
    X, y, meta = prepare_features_and_labels(df_raw, cfg)
    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols = list((prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or []))
    for lvl in cfg.get("BTTWD", {}).get("bucket_levels", []):
        col = lvl.get("col") or lvl.get("feature")
        if col and col not in bucket_cols:
            bucket_cols.append(col)
    bucket_df = meta.get("df_processed", df_raw)[bucket_cols].reset_index(drop=True)
    return X, y, bucket_df, bucket_cols


def _all_parent_ids(bucket_ids: pd.Series) -> set[str]:
    ids = {"ROOT"}
    for bid in pd.unique(bucket_ids):
        ids.update(bucket_path_to_root(str(bid)))
    return ids


def _mask_for_bucket(bucket_ids: pd.Series, bucket_id: str) -> np.ndarray:
    if bucket_id == "ROOT":
        return np.ones(len(bucket_ids), dtype=bool)
    values = bucket_ids.astype(str)
    return ((values == bucket_id) | values.str.startswith(f"{bucket_id}|")).to_numpy()


def _bucket_prefixes(bucket_id: str) -> list[str]:
    if not bucket_id or bucket_id == "ROOT":
        return []
    parts = str(bucket_id).split("|")
    return ["|".join(parts[:idx]) for idx in range(1, len(parts) + 1)]


def _bt_min_bucket_size(cfg: dict) -> int:
    bttwd_cfg = cfg.get("BTTWD", {})
    if not bool(bttwd_cfg.get("use_min_bucket_size_limit", True)):
        return 0
    return int(bttwd_cfg.get("min_bucket_size", 0) or 0)


def _weak_bucket_cfg(cfg: dict) -> dict:
    bttwd_cfg = cfg.get("BTTWD", {})
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})
    weak_cfg = dict(bttwd_cfg.get("weak_bucket", {}) or {})
    weak_cfg.setdefault("enabled", True)
    weak_cfg.setdefault("min_train_samples", _bt_min_bucket_size(cfg))
    weak_cfg.setdefault(
        "min_val_samples",
        bttwd_cfg.get("min_val_samples_per_bucket", threshold_cfg.get("min_samples_for_thresholds", 10)),
    )
    weak_cfg.setdefault("min_gain_over_parent", bttwd_cfg.get("min_gain_for_split", 0.0))
    weak_cfg.setdefault("sparse_labels", ["OTHER", "others", "unknown"])
    score_cfg = dict(weak_cfg.get("score", {}) or {})
    score_cfg.setdefault("mode", "composite")
    score_cfg.setdefault("w_f1", 1.0)
    score_cfg.setdefault("w_regret", 1.0)
    score_cfg.setdefault("w_bnd", 0.5)
    weak_cfg["score"] = score_cfg
    return weak_cfg


def _is_sparse_merged_bucket(bucket_id: str, sparse_labels: list[str]) -> bool:
    labels = {str(label).lower() for label in sparse_labels}
    for part in str(bucket_id).split("|"):
        value = part.split("=", 1)[-1].lower()
        if value in labels:
            return True
    return False


def _structural_score(stats: dict, cfg: dict) -> float:
    score_cfg = _weak_bucket_cfg(cfg).get("score", {})
    mode = str(score_cfg.get("mode", "composite")).lower()
    if mode != "composite":
        return -float(stats.get("regret", 0.0))
    return (
        float(score_cfg.get("w_f1", 1.0)) * float(stats.get("f1", 0.0))
        - float(score_cfg.get("w_regret", 1.0)) * float(stats.get("regret", 0.0))
        - float(score_cfg.get("w_bnd", 0.5)) * float(stats.get("bnd_ratio", 0.0))
    )


def _parent_bucket_id(bucket_id: str) -> str:
    path = bucket_path_to_root(bucket_id)
    return path[1] if len(path) > 1 else "ROOT"


def _nearest_threshold_ancestor(bucket_id: str, records: dict, thresholds: dict) -> str:
    for ancestor_id in bucket_path_to_root(bucket_id)[1:]:
        if ancestor_id == "ROOT":
            return "ROOT"
        if ancestor_id in records and ancestor_id in thresholds:
            return ancestor_id
    return "ROOT"


def _governance_cfg(cfg: dict) -> dict:
    return cfg.get("governance") or cfg.get("GOVERNANCE") or {}


def _bnd_rescue_cfg(cfg: dict) -> dict:
    return _governance_cfg(cfg).get("bnd_early_rescue", {}) or {}


def _weak_reason_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _nearest_rescue_only_bucket_id(
    raw_bucket_id: str,
    effective_bucket_id: str,
    bucket_meta: dict,
    cfg: dict,
) -> str | None:
    rescue_cfg = _bnd_rescue_cfg(cfg)
    if not bool(rescue_cfg.get("rescue_only_parent_enabled", True)):
        return None
    if effective_bucket_id != "ROOT" or not raw_bucket_id or raw_bucket_id == "ROOT":
        return None

    path = bucket_path_to_root(str(raw_bucket_id))
    if len(path) <= 1:
        return None
    candidates = path[1:-1] if bucket_level(raw_bucket_id) >= 3 else path[:-1]
    min_support = int(rescue_cfg.get("min_bucket_support", 20) or 0)
    allowed_weak_reasons = {
        str(item)
        for item in rescue_cfg.get("rescue_only_allowed_weak_reasons", ["insufficient_improvement_over_parent"])
        or []
    }

    for candidate_id in candidates:
        record = (bucket_meta or {}).get(candidate_id)
        if not record:
            continue
        try:
            n_val = int(float(record.get("n_val", 0) or 0))
        except (TypeError, ValueError):
            n_val = 0
        if min_support > 0 and n_val < min_support:
            continue
        reasons = set(_weak_reason_list(record.get("weak_reasons", [])))
        if reasons - allowed_weak_reasons:
            continue
        return candidate_id
    return None


def _build_allowed_bucket_ids(train_bucket_ids: pd.Series, cfg: dict) -> set[str]:
    min_bucket_size = _bt_min_bucket_size(cfg)
    all_ids = sorted(_all_parent_ids(train_bucket_ids), key=bucket_level)
    if min_bucket_size <= 0:
        return set(all_ids)

    values = train_bucket_ids.astype(str)
    counts = {"ROOT": len(values)}
    for bucket_id in all_ids:
        if bucket_id == "ROOT":
            continue
        counts[bucket_id] = int(_mask_for_bucket(values, bucket_id).sum())

    allowed = {"ROOT"}
    for bucket_id in all_ids:
        if bucket_id == "ROOT":
            continue
        parent_path = bucket_path_to_root(bucket_id)
        parent = parent_path[1] if len(parent_path) > 1 else "ROOT"
        if parent in allowed and counts.get(bucket_id, 0) >= min_bucket_size:
            allowed.add(bucket_id)
    return allowed


def _apply_bt_min_bucket_size(bucket_ids: pd.Series, allowed_bucket_ids: set[str]) -> pd.Series:
    allowed = set(allowed_bucket_ids or {"ROOT"})

    def route(bucket_id: str) -> str:
        for candidate in reversed(_bucket_prefixes(str(bucket_id))):
            if candidate in allowed:
                return candidate
        return "ROOT"

    return bucket_ids.astype(str).apply(route)


def _resolve_weak_buckets(
    y_val,
    posterior_val,
    train_bucket_ids: pd.Series,
    val_bucket_ids: pd.Series,
    cfg: dict,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame]:
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})
    alpha_grid = threshold_cfg.get("alpha_grid", [0.5])
    beta_grid = threshold_cfg.get("beta_grid", [0.0])
    gap_min = threshold_cfg.get("gap_min", 0.0)
    costs = threshold_cfg.get("costs", {})
    weak_cfg = _weak_bucket_cfg(cfg)
    weak_enabled = bool(weak_cfg.get("enabled", True))
    min_train = int(weak_cfg.get("min_train_samples", 0) or 0)
    min_val = int(weak_cfg.get("min_val_samples", 0) or 0)
    min_gain = float(weak_cfg.get("min_gain_over_parent", 0.0))
    sparse_labels = list(weak_cfg.get("sparse_labels", []))

    y_val = np.asarray(y_val)
    train_bucket_ids = train_bucket_ids.astype(str)
    val_bucket_ids = val_bucket_ids.astype(str)
    all_ids = sorted(_all_parent_ids(pd.concat([train_bucket_ids, val_bucket_ids], ignore_index=True)), key=bucket_level)

    thresholds = {}
    records = {}

    def search(bucket_id: str, mask: np.ndarray):
        alpha, beta, stats = search_thresholds_with_regret(
            posterior_val[mask],
            y_val[mask],
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            costs=costs,
            gap_min=gap_min,
        )
        stats = dict(stats or {})
        score = _structural_score(stats, cfg)
        thresholds[bucket_id] = (float(alpha), float(beta))
        return float(alpha), float(beta), stats, float(score)

    root_mask = np.ones(len(val_bucket_ids), dtype=bool)
    root_alpha, root_beta, root_stats, root_score = search("ROOT", root_mask)
    records["ROOT"] = {
        "bucket_id": "ROOT",
        "parent_bucket_id": "",
        "level": 0,
        "status": "strong",
        "weak_reasons": [],
        "effective_bucket_id": "ROOT",
        "threshold_source_bucket": "ROOT",
        "n_train": int(len(train_bucket_ids)),
        "n_val": int(len(val_bucket_ids)),
        "pos_val": int(np.sum(y_val == 1)),
        "neg_val": int(np.sum(y_val == 0)),
        "score": root_score,
        "parent_score": math.nan,
        "gain": math.nan,
        "alpha": root_alpha,
        "beta": root_beta,
        **{f"stats_{key}": value for key, value in root_stats.items()},
    }

    for bucket_id in all_ids:
        if bucket_id == "ROOT":
            continue
        parent_id = _parent_bucket_id(bucket_id)
        parent_record = records.get(parent_id, records["ROOT"])
        train_mask = _mask_for_bucket(train_bucket_ids, bucket_id)
        val_mask = _mask_for_bucket(val_bucket_ids, bucket_id)
        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        y_bucket = y_val[val_mask]
        pos_val = int(np.sum(y_bucket == 1))
        neg_val = int(np.sum(y_bucket == 0))
        weak_reasons = []

        if not weak_enabled:
            pass
        elif min_train > 0 and n_train < min_train:
            weak_reasons.append("insufficient_train_samples")
        elif min_val > 0 and n_val < min_val:
            weak_reasons.append("insufficient_validation_samples")
        elif _is_sparse_merged_bucket(bucket_id, sparse_labels):
            weak_reasons.append("merged_sparse_category")
        elif np.unique(y_bucket).size < 2:
            weak_reasons.append("invalid_structural_score")

        alpha = beta = math.nan
        stats = {}
        score = math.nan
        raw_parent_score = parent_record.get("score", math.nan)
        parent_score = float(raw_parent_score) if not pd.isna(raw_parent_score) else root_score
        gain = math.nan
        if not weak_reasons:
            alpha, beta, stats, score = search(bucket_id, val_mask)
            gain = score - parent_score
            if weak_enabled and gain < min_gain:
                weak_reasons.append("insufficient_improvement_over_parent")
                thresholds.pop(bucket_id, None)

        status = "weak" if weak_reasons else "strong"
        effective_bucket_id = bucket_id if status == "strong" else _nearest_threshold_ancestor(bucket_id, records, thresholds)
        threshold_source_bucket = bucket_id if status == "strong" else effective_bucket_id
        records[bucket_id] = {
            "bucket_id": bucket_id,
            "parent_bucket_id": parent_id,
            "level": bucket_level(bucket_id),
            "status": status,
            "weak_reasons": weak_reasons,
            "effective_bucket_id": effective_bucket_id,
            "threshold_source_bucket": threshold_source_bucket,
            "n_train": n_train,
            "n_val": n_val,
            "pos_val": pos_val,
            "neg_val": neg_val,
            "score": score,
            "parent_score": parent_score,
            "gain": gain,
            "alpha": alpha,
            "beta": beta,
            **{f"stats_{key}": value for key, value in stats.items()},
        }

    return thresholds, pd.DataFrame(records.values())


def _effective_bucket_id(raw_bucket_id: str, bucket_report: pd.DataFrame) -> str:
    if bucket_report.empty:
        return raw_bucket_id
    records = bucket_report.set_index("bucket_id").to_dict(orient="index")
    for candidate in reversed(_bucket_prefixes(str(raw_bucket_id))):
        if candidate in records:
            return str(records[candidate].get("effective_bucket_id") or "ROOT")
    return "ROOT"


def _search_thresholds_by_bucket(y_val, posterior_val, bucket_ids_val: pd.Series, cfg: dict) -> dict[str, tuple[float, float]]:
    threshold_cfg = cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})
    alpha_grid = threshold_cfg.get("alpha_grid", [0.5])
    beta_grid = threshold_cfg.get("beta_grid", [0.0])
    gap_min = threshold_cfg.get("gap_min", 0.0)
    costs = threshold_cfg.get("costs", {})
    min_samples = threshold_cfg.get(
        "min_samples_for_thresholds",
        cfg.get("BTTWD", {}).get("min_val_samples_per_bucket", 10),
    )

    thresholds = {}
    for bucket_id in sorted(_all_parent_ids(bucket_ids_val), key=bucket_level):
        mask = _mask_for_bucket(bucket_ids_val, bucket_id)
        if mask.sum() < min_samples and bucket_id != "ROOT":
            continue
        alpha, beta, _ = search_thresholds_with_regret(
            posterior_val[mask],
            y_val[mask],
            alpha_grid=alpha_grid,
            beta_grid=beta_grid,
            costs=costs,
            gap_min=gap_min,
        )
        thresholds[bucket_id] = (float(alpha), float(beta))
    thresholds.setdefault("ROOT", (float(alpha_grid[0]), float(beta_grid[0])))
    return thresholds


def _sample_regret(true_label: int, decision: str, costs: dict) -> float:
    return compute_regret([true_label], [decision_to_numeric(decision)], costs)


def _decision_to_binary_for_metrics(series: pd.Series) -> np.ndarray:
    return series.map({"P": 1, "N": 0, "BND": 0}).fillna(0).astype(int).to_numpy()


def _json_len(value: str) -> int:
    if not isinstance(value, str) or not value:
        return 0
    try:
        return len(json.loads(value))
    except Exception:
        return 0


def _json_weight_entropy(value: str) -> float:
    if not isinstance(value, str) or not value:
        return 0.0
    try:
        weights = [float(v) for v in json.loads(value)]
    except Exception:
        return 0.0
    return float(-sum(w * math.log(w) for w in weights if w > 0.0))


def _error_rate(records: pd.DataFrame, decision_col: str) -> float:
    if records.empty:
        return 0.0
    pred = _decision_to_binary_for_metrics(records[decision_col])
    return float(np.mean(pred != records["true_label"].to_numpy()))


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_stats(
    records: pd.DataFrame,
    decision_col: str,
    prefix: str,
    regret_col: str = "final_regret",
) -> dict:
    if records.empty:
        return {
            f"{prefix}_tp_count": 0,
            f"{prefix}_tn_count": 0,
            f"{prefix}_fp_count": 0,
            f"{prefix}_fn_count": 0,
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_specificity": 0.0,
            f"{prefix}_fpr": 0.0,
            f"{prefix}_fnr": 0.0,
            f"{prefix}_pred_positive_rate": 0.0,
            f"{prefix}_true_positive_rate": 0.0,
            f"{prefix}_positive_rate_gap": 0.0,
            f"{prefix}_recall_specificity_gap": 0.0,
            f"{prefix}_fp_regret_sum": 0.0,
            f"{prefix}_fn_regret_sum": 0.0,
        }

    pred = _decision_to_binary_for_metrics(records[decision_col])
    truth = records["true_label"].astype(int).to_numpy()
    tp_mask = (pred == 1) & (truth == 1)
    tn_mask = (pred == 0) & (truth == 0)
    fp_mask = (pred == 1) & (truth == 0)
    fn_mask = (pred == 0) & (truth == 1)
    tp = int(tp_mask.sum())
    tn = int(tn_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    n = int(len(records))
    precision = _safe_rate(tp, tp + fp)
    recall = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    pred_positive_rate = _safe_rate(tp + fp, n)
    true_positive_rate = _safe_rate(tp + fn, n)
    regret_values = records[regret_col].astype(float).to_numpy() if regret_col in records else np.zeros(n)
    return {
        f"{prefix}_tp_count": tp,
        f"{prefix}_tn_count": tn,
        f"{prefix}_fp_count": fp,
        f"{prefix}_fn_count": fn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_specificity": specificity,
        f"{prefix}_fpr": _safe_rate(fp, fp + tn),
        f"{prefix}_fnr": _safe_rate(fn, fn + tp),
        f"{prefix}_pred_positive_rate": pred_positive_rate,
        f"{prefix}_true_positive_rate": true_positive_rate,
        f"{prefix}_positive_rate_gap": pred_positive_rate - true_positive_rate,
        f"{prefix}_recall_specificity_gap": recall - specificity,
        f"{prefix}_fp_regret_sum": float(regret_values[fp_mask].sum()),
        f"{prefix}_fn_regret_sum": float(regret_values[fn_mask].sum()),
    }


def _json_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dump_dict(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _condition_quality(records: pd.DataFrame) -> tuple[str, str, str]:
    if records.empty or "bnd_early_rescue_conditions" not in records:
        return "{}", "{}", "{}"
    condition_series = records["bnd_early_rescue_conditions"].fillna("").replace("", "unknown")
    counts = condition_series.value_counts().sort_index().to_dict()
    error_rates = {}
    regrets = {}
    for condition_key, part in records.groupby(condition_series):
        error_rates[str(condition_key)] = _error_rate(part, "final_decision")
        regrets[str(condition_key)] = float(part["final_regret"].mean()) if len(part) else 0.0
    return _json_dump_dict(counts), _json_dump_dict(error_rates), _json_dump_dict(regrets)


def _sum_json_dict(series: pd.Series) -> dict:
    totals: dict[str, float] = {}
    for value in series:
        for key, item in _json_dict(value).items():
            totals[str(key)] = totals.get(str(key), 0.0) + float(item)
    return {key: int(value) if float(value).is_integer() else float(value) for key, value in sorted(totals.items())}


def _weighted_json_mean(value_series: pd.Series, count_series: pd.Series) -> dict:
    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for values_raw, counts_raw in zip(value_series, count_series):
        values = _json_dict(values_raw)
        counts = _json_dict(counts_raw)
        for key, value in values.items():
            weight = float(counts.get(key, 0.0))
            if weight <= 0:
                continue
            numerators[key] = numerators.get(key, 0.0) + float(value) * weight
            denominators[key] = denominators.get(key, 0.0) + weight
    return {key: float(numerators[key] / denominators[key]) for key in sorted(numerators) if denominators.get(key, 0.0) > 0}


def _refresh_classification_rates_from_counts(row: dict) -> None:
    prefixes = [key[: -len("_tp_count")] for key in row if key.endswith("_tp_count")]
    for prefix in prefixes:
        tp = float(row.get(f"{prefix}_tp_count", 0.0) or 0.0)
        tn = float(row.get(f"{prefix}_tn_count", 0.0) or 0.0)
        fp = float(row.get(f"{prefix}_fp_count", 0.0) or 0.0)
        fn = float(row.get(f"{prefix}_fn_count", 0.0) or 0.0)
        total = tp + tn + fp + fn
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        specificity = _safe_rate(tn, tn + fp)
        pred_positive_rate = _safe_rate(tp + fp, total)
        true_positive_rate = _safe_rate(tp + fn, total)
        row[f"{prefix}_precision"] = precision
        row[f"{prefix}_recall"] = recall
        row[f"{prefix}_specificity"] = specificity
        row[f"{prefix}_fpr"] = _safe_rate(fp, fp + tn)
        row[f"{prefix}_fnr"] = _safe_rate(fn, fn + tp)
        row[f"{prefix}_pred_positive_rate"] = pred_positive_rate
        row[f"{prefix}_true_positive_rate"] = true_positive_rate
        row[f"{prefix}_positive_rate_gap"] = pred_positive_rate - true_positive_rate
        row[f"{prefix}_recall_specificity_gap"] = recall - specificity


def _summarize_fold(records: pd.DataFrame, y_score: np.ndarray) -> dict:
    final_binary = _decision_to_binary_for_metrics(records["final_decision"])
    original_binary = _decision_to_binary_for_metrics(records["original_twd_decision"])
    metrics = compute_binary_metrics(records["true_label"].to_numpy(), final_binary, y_score, {"use_metrics": ["BAC", "F1"]})
    original_metrics = compute_binary_metrics(
        records["true_label"].to_numpy(), original_binary, y_score, {"use_metrics": ["BAC", "F1"]}
    )
    bnd_records = records[records["original_is_bnd"]]
    pn_records = records[~records["original_is_bnd"]]
    cp_pass_records = pn_records[pn_records["cp_passed"]]
    cp_reject_records = pn_records[pn_records["cp_rejected"]]
    cp_pass_count = int(len(cp_pass_records))
    cp_reject_count = int(len(cp_reject_records))
    root_records = records[records["closure_level"] == 0]
    leaf_records = records[records["closure_bucket"] == records["original_bucket_id"]]
    parent_records = records[(records["closure_level"] > 0) & (records["closure_bucket"] != records["original_bucket_id"])]
    bnd_root_records = bnd_records[bnd_records["closure_level"] == 0]
    bnd_parent_records = bnd_records[
        (bnd_records["closure_level"] > 0) & (bnd_records["closure_bucket"] != bnd_records["original_bucket_id"])
    ]
    bnd_rescue_records = bnd_records[bnd_records["bnd_early_rescue_used"]]
    bnd_rescue_leaf_records = bnd_rescue_records[bnd_rescue_records["bnd_early_rescue_layer"] == "leaf"]
    bnd_rescue_parent_records = bnd_rescue_records[bnd_rescue_records["bnd_early_rescue_layer"] == "parent"]
    bnd_rescue_candidate_records = (
        bnd_records[bnd_records["bnd_rescue_candidate_bucket"].fillna("") != ""]
        if "bnd_rescue_candidate_bucket" in bnd_records
        else bnd_records.iloc[0:0]
    )
    bnd_rescue_candidate_success_records = bnd_rescue_candidate_records[bnd_rescue_candidate_records["bnd_early_rescue_used"]]
    bnd_rescue_from_effective_root_records = bnd_rescue_records[bnd_rescue_records["original_bucket_id"] == "ROOT"]
    non_root_bnd_records = bnd_records[bnd_records["original_bucket_id"] != "ROOT"]
    non_root_bnd_rescue_records = non_root_bnd_records[non_root_bnd_records["bnd_early_rescue_used"]]
    root_bnd_from_effective_root_records = bnd_root_records[bnd_root_records["original_bucket_id"] == "ROOT"]
    root_bnd_from_rescue_failed_records = bnd_root_records[bnd_root_records["original_bucket_id"] != "ROOT"]
    rescue_root_error_delta = _error_rate(bnd_root_records, "final_decision") - _error_rate(
        bnd_rescue_records, "final_decision"
    )
    rescue_root_regret_delta = (
        (float(bnd_root_records["final_regret"].mean()) if len(bnd_root_records) else 0.0)
        - (float(bnd_rescue_records["final_regret"].mean()) if len(bnd_rescue_records) else 0.0)
    )
    condition_counts, condition_error_rates, condition_regrets = _condition_quality(bnd_rescue_records)
    evidence_counts = records["evidence_path"].fillna("").apply(_json_len)
    weight_entropy = records["evidence_weights"].fillna("").apply(_json_weight_entropy)
    summary = {
        "dataset_name": records["dataset_name"].iat[0],
        "fold_id": records["fold_id"].iat[0],
        "final_regret": float(records["final_regret"].mean()),
        "final_BAC": float(metrics.get("BAC", np.nan)),
        "final_F1": float(metrics.get("F1", np.nan)),
        "final_accuracy": float(np.mean(final_binary == records["true_label"].to_numpy())),
        "original_regret": float(records["original_regret"].mean()),
        "original_BAC": float(original_metrics.get("BAC", np.nan)),
        "original_F1": float(original_metrics.get("F1", np.nan)),
        "original_accuracy": float(np.mean(original_binary == records["true_label"].to_numpy())),
        "closure_rate": float(records["closed"].mean()),
        "unresolved_ratio": float((~records["closed"]).mean()),
        "average_defer_depth": float(records["defer_path"].fillna("").apply(lambda x: 0 if not x else x.count(",") + 1).mean()),
        "average_closure_level": float(records["closure_level"].mean()),
        "closed_at_leaf_count": int(len(leaf_records)),
        "closed_at_parent_count": int(len(parent_records)),
        "closed_at_root_count": int(len(root_records)),
        "original_bnd_count": int(len(bnd_records)),
        "original_bnd_ratio": float(len(bnd_records) / len(records)) if len(records) else 0.0,
        "bnd_closure_rate": float(bnd_records["closed"].mean()) if len(bnd_records) else 0.0,
        "bnd_closed_error_rate": _error_rate(bnd_records, "final_decision"),
        "bnd_closed_regret": float(bnd_records["final_regret"].mean()) if len(bnd_records) else 0.0,
        "bnd_closed_at_parent_count": int(len(bnd_parent_records)),
        "bnd_closed_at_root_count": int(len(bnd_root_records)),
        "bnd_early_rescue_count": int(len(bnd_rescue_records)),
        "bnd_early_rescue_error_rate": _error_rate(bnd_rescue_records, "final_decision"),
        "bnd_early_rescue_regret": float(bnd_rescue_records["final_regret"].mean()) if len(bnd_rescue_records) else 0.0,
        "bnd_early_rescue_positive_rate": float((bnd_rescue_records["final_decision"] == "P").mean())
        if len(bnd_rescue_records)
        else 0.0,
        "bnd_early_rescue_at_leaf_count": int(len(bnd_rescue_leaf_records)),
        "bnd_early_rescue_at_parent_count": int(len(bnd_rescue_parent_records)),
        "bnd_rescue_parent_candidate_count": int(len(bnd_rescue_candidate_records)),
        "bnd_rescue_parent_candidate_success_count": int(len(bnd_rescue_candidate_success_records)),
        "bnd_rescue_parent_candidate_success_rate": float(
            len(bnd_rescue_candidate_success_records) / len(bnd_rescue_candidate_records)
        )
        if len(bnd_rescue_candidate_records)
        else 0.0,
        "bnd_rescue_from_effective_root_count": int(len(bnd_rescue_from_effective_root_records)),
        "bnd_rescue_from_effective_root_error_rate": _error_rate(bnd_rescue_from_effective_root_records, "final_decision"),
        "root_bnd_count_after_rescue": int(len(bnd_root_records)),
        "non_root_bnd_count": int(len(non_root_bnd_records)),
        "non_root_bnd_rescue_count": int(len(non_root_bnd_rescue_records)),
        "non_root_bnd_rescue_rate": float(len(non_root_bnd_rescue_records) / len(non_root_bnd_records))
        if len(non_root_bnd_records)
        else 0.0,
        "root_bnd_reduction_rate": float(1.0 - len(root_bnd_from_rescue_failed_records) / len(non_root_bnd_records))
        if len(non_root_bnd_records)
        else 0.0,
        "rescue_vs_root_error_delta": float(rescue_root_error_delta),
        "rescue_vs_root_regret_delta": float(rescue_root_regret_delta),
        "root_bnd_from_effective_root_count": int(len(root_bnd_from_effective_root_records)),
        "root_bnd_from_rescue_failed_count": int(len(root_bnd_from_rescue_failed_records)),
        "bnd_early_rescue_condition_counts": condition_counts,
        "bnd_early_rescue_condition_error_rates": condition_error_rates,
        "bnd_early_rescue_condition_regrets": condition_regrets,
        "cp_pass_count": cp_pass_count,
        "cp_reject_count": cp_reject_count,
        "cp_reject_defer_count": int((pn_records["defer_trigger_source"] == "post_validation").sum()),
        "cp_disabled": bool(records["cp_disabled"].all()),
        "cp_alpha": float(records["alpha_cp"].dropna().iloc[0]) if "alpha_cp" in records and len(records) else 0.1,
        "cp_reject_rate": float(cp_reject_count / len(pn_records)) if len(pn_records) else 0.0,
        "cp_pass_error_rate": _error_rate(cp_pass_records, "original_twd_decision"),
        "cp_reject_original_error_rate": _error_rate(cp_reject_records, "original_twd_decision"),
        "cp_reject_final_error_rate": _error_rate(cp_reject_records, "final_decision"),
        "cp_rejected_pn_count": cp_reject_count,
        "cp_rejected_pn_ratio": float(cp_reject_count / len(pn_records)) if len(pn_records) else 0.0,
        "cp_rejected_original_error_rate": _error_rate(cp_reject_records, "original_twd_decision"),
        "cp_rejected_final_error_rate": _error_rate(cp_reject_records, "final_decision"),
        "cp_rejected_final_regret": float(cp_reject_records["final_regret"].mean()) if len(cp_reject_records) else 0.0,
        "progressive_update_enabled": bool(records["progressive_update_enabled"].any()),
        "average_evidence_count": float(evidence_counts.mean()),
        "average_weight_entropy": float(weight_entropy.mean()),
        "progressive_closed_count": int((records["closed"] & records["progressive_update_enabled"]).sum()),
    }
    for prefix, frame, decision_col, regret_col in [
        ("final", records, "final_decision", "final_regret"),
        ("original", records, "original_twd_decision", "original_regret"),
        ("bnd_closed", bnd_records, "final_decision", "final_regret"),
        ("bnd_early_rescue", bnd_rescue_records, "final_decision", "final_regret"),
        ("bnd_early_rescue_at_leaf", bnd_rescue_leaf_records, "final_decision", "final_regret"),
        ("bnd_early_rescue_at_parent", bnd_rescue_parent_records, "final_decision", "final_regret"),
        ("bnd_rescue_from_effective_root", bnd_rescue_from_effective_root_records, "final_decision", "final_regret"),
        ("root_bnd_after_rescue", bnd_root_records, "final_decision", "final_regret"),
        ("root_bnd_from_effective_root", root_bnd_from_effective_root_records, "final_decision", "final_regret"),
        ("root_bnd_from_rescue_failed", root_bnd_from_rescue_failed_records, "final_decision", "final_regret"),
    ]:
        summary.update(_classification_stats(frame, decision_col, prefix, regret_col=regret_col))
    return summary


def run_dataset(
    dataset_name: str,
    config_path: Path,
    output_root: Path,
    governance_override: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_yaml(config_path)
    if governance_override:
        local_governance = dict(cfg.get("governance") or cfg.get("GOVERNANCE") or {})
        for section, values in governance_override.items():
            if isinstance(values, dict):
                local_governance.setdefault(section, {})
                local_governance[section].update(values)
            else:
                local_governance[section] = values
        cfg["governance"] = local_governance
    _resolve_data_paths(cfg, config_path)
    assert_no_bsm_dependency(cfg)
    set_global_seed(cfg.get("SEED", {}).get("global_seed", 42))

    df_raw, _ = load_dataset(cfg)
    X, y, bucket_df, bucket_cols = _build_bucket_df(df_raw, cfg)
    tree = BucketTree(cfg.get("BTTWD", {}).get("bucket_levels", []), feature_names=bucket_cols)
    skf = make_stratified_kfold(cfg)
    costs = (cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})).get("costs", {})

    sample_records = []
    fold_summaries = []
    bucket_report_records = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        log_info(f"【governance】{dataset_name} 第 {fold_id}/{skf.n_splits} 折")
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        bucket_train_full = bucket_df.iloc[train_idx].reset_index(drop=True)
        X_test, y_test = X[test_idx], y[test_idx]
        bucket_test = bucket_df.iloc[test_idx].reset_index(drop=True)

        X_train, X_val, y_train, y_val, bucket_train, bucket_val = split_train_validation(
            X_train_full, y_train_full, bucket_train_full, cfg
        )
        posterior_model = GlobalPosterior(cfg).fit(X_train, y_train)
        posterior_val = posterior_model.predict_proba(X_val)
        cp_validator = SplitConformalValidator(posterior_val, y_val)
        raw_bucket_ids_train = tree.assign_buckets(bucket_train.reset_index(drop=True))
        raw_bucket_ids_val = tree.assign_buckets(bucket_val.reset_index(drop=True))
        thresholds, bucket_report = _resolve_weak_buckets(
            y_val,
            posterior_val,
            raw_bucket_ids_train,
            raw_bucket_ids_val,
            cfg,
        )
        bucket_report.insert(0, "fold_id", fold_id)
        bucket_report.insert(0, "dataset_name", dataset_name)
        bucket_report["weak_reasons"] = bucket_report["weak_reasons"].apply(lambda value: json.dumps(value, ensure_ascii=False))
        bucket_report_records.append(bucket_report)
        log_info(
            f"【governance】weak bucket resolver：strong="
            f"{int((bucket_report['status'] == 'strong').sum())}，"
            f"weak={int((bucket_report['status'] == 'weak').sum())}"
        )
        bucket_meta = bucket_report.set_index("bucket_id").to_dict(orient="index")

        posterior_test = posterior_model.predict_proba(X_test)
        raw_bucket_ids_test = tree.assign_buckets(bucket_test)
        bucket_ids_test = raw_bucket_ids_test.astype(str).apply(lambda bid: _effective_bucket_id(bid, bucket_report))
        fold_records = []
        for local_idx, bucket_id in enumerate(bucket_ids_test.astype(str).tolist()):
            raw_bucket_id = str(raw_bucket_ids_test.iloc[local_idx])
            bucket_info = bucket_meta.get(raw_bucket_id) or bucket_meta.get(bucket_id, {})
            p = float(posterior_test[local_idx])
            alpha, beta = thresholds.get(bucket_id, thresholds["ROOT"])
            original_decision = decide_twd(p, alpha, beta)
            rescue_start_bucket_id = (
                _nearest_rescue_only_bucket_id(raw_bucket_id, bucket_id, bucket_meta, cfg)
                if original_decision == "BND"
                else None
            )
            risks = risk_values_for_probability(p, costs)
            validation = validate_post_decision(
                sample_id=int(test_idx[local_idx]),
                bucket_id=bucket_id,
                twd_decision=original_decision,
                posterior=p,
                risk_values=risks,
                bucket_context={"level": bucket_level(bucket_id), "cp_validator": cp_validator},
                config=cfg,
            )
            defer_result = {}
            if not validation["reliable"]:
                defer_result = resolve_deferred_sample(
                    sample_id=int(test_idx[local_idx]),
                    start_bucket_id=bucket_id,
                    tree=tree,
                    posterior=p,
                    thresholds=thresholds,
                    config=cfg,
                    cp_validator=cp_validator,
                    initial_decision=original_decision,
                    initial_validation=validation,
                    bucket_meta=bucket_meta,
                    rescue_start_bucket_id=rescue_start_bucket_id,
                )
            final_decision = original_decision if validation["reliable"] else defer_result["final_decision"]
            rec = build_sample_record(
                dataset_name=dataset_name,
                fold_id=fold_id,
                sample_id=int(test_idx[local_idx]),
                true_label=int(y_test[local_idx]),
                posterior=p,
                original_bucket_id=bucket_id,
                original_twd_decision=original_decision,
                validation=validation,
                defer_result=defer_result,
                final_regret=_sample_regret(int(y_test[local_idx]), final_decision, costs),
                original_regret=_sample_regret(int(y_test[local_idx]), original_decision, costs),
                bucket_context={
                    "raw_bucket_id": raw_bucket_id,
                    "effective_bucket_id": bucket_id,
                    "bucket_was_weak": raw_bucket_id != bucket_id,
                    "weak_reason": bucket_info.get("weak_reasons", ""),
                    "threshold_source_bucket": bucket_info.get("threshold_source_bucket", bucket_id),
                    "bucket_score": bucket_info.get("score", 0.0),
                    "bucket_parent_score": bucket_info.get("parent_score", 0.0),
                    "bucket_gain": bucket_info.get("gain", 0.0),
                    "bnd_rescue_candidate_bucket": rescue_start_bucket_id or "",
                },
            )
            fold_records.append(rec)
        fold_df = pd.DataFrame(fold_records)
        sample_records.append(fold_df)
        fold_summaries.append(_summarize_fold(fold_df, posterior_test))

    sample_df = pd.concat(sample_records, ignore_index=True)
    fold_df = pd.DataFrame(fold_summaries)
    bucket_report_df = pd.concat(bucket_report_records, ignore_index=True) if bucket_report_records else pd.DataFrame()
    out_dir = ensure_dir(output_root)
    sample_df.to_csv(out_dir / "sample_records.csv", index=False)
    fold_df.to_csv(out_dir / "fold_summary.csv", index=False)
    if not bucket_report_df.empty:
        bucket_report_df.to_csv(out_dir / "weak_bucket_report.csv", index=False)
    return sample_df, fold_df


def _merge_governance_override(base: dict | None, cli_args: argparse.Namespace | None = None) -> dict:
    governance = dict(base or {})
    governance.setdefault("cp", {})
    governance.setdefault("progressive_update", {})
    governance.setdefault("bnd_early_rescue", {})
    governance.setdefault("ablation", {})
    if cli_args is not None:
        if cli_args.cp_alpha is not None:
            governance["cp"]["alpha"] = float(cli_args.cp_alpha)
        if cli_args.disable_cp_validation:
            governance["ablation"]["disable_cp_validation"] = True
        if cli_args.disable_progressive_update:
            governance["ablation"]["disable_progressive_update"] = True
    governance["cp"].setdefault("enabled", True)
    governance["cp"].setdefault("alpha", 0.1)
    governance["progressive_update"].setdefault("enabled", True)
    governance["progressive_update"].setdefault("epsilon", 0.001)
    governance["bnd_early_rescue"].setdefault("enabled", True)
    governance["bnd_early_rescue"].setdefault("posterior_margin_threshold", 0.10)
    governance["bnd_early_rescue"].setdefault("risk_gap_threshold", 0.05)
    governance["bnd_early_rescue"].setdefault("cp_gap_threshold", 0.10)
    governance["bnd_early_rescue"].setdefault("cp_override_threshold", 0.20)
    governance["bnd_early_rescue"].setdefault("bucket_margin_threshold", 0.15)
    governance["bnd_early_rescue"].setdefault("min_bucket_support", 20)
    governance["bnd_early_rescue"].setdefault("rescue_only_parent_enabled", True)
    governance["bnd_early_rescue"].setdefault(
        "rescue_only_allowed_weak_reasons", ["insufficient_improvement_over_parent"]
    )
    governance["bnd_early_rescue"].setdefault("min_conditions", 2)
    governance["ablation"].setdefault("disable_cp_validation", False)
    governance["ablation"].setdefault("disable_progressive_update", False)
    return governance


def _mode_name(governance: dict) -> str:
    no_cp = bool(governance.get("ablation", {}).get("disable_cp_validation", False)) or not bool(
        governance.get("cp", {}).get("enabled", True)
    )
    no_progressive = bool(governance.get("ablation", {}).get("disable_progressive_update", False)) or not bool(
        governance.get("progressive_update", {}).get("enabled", True)
    )
    if no_cp and no_progressive:
        key = "no_cp_no_progressive"
    elif no_cp:
        key = "no_cp"
    elif no_progressive:
        key = "no_progressive"
    else:
        key = "full"
    return str(governance.get("output", {}).get("mode_dirs", {}).get(key, key))


def _summarize_dataset(fold_df: pd.DataFrame) -> dict:
    row = {"dataset_name": fold_df["dataset_name"].iat[0]}
    json_cols = {
        "bnd_early_rescue_condition_counts",
        "bnd_early_rescue_condition_error_rates",
        "bnd_early_rescue_condition_regrets",
    }
    count_cols = [
        col
        for col in fold_df.columns
        if col.endswith("_count") or col.endswith("_count_after_rescue") or col.endswith("_regret_sum")
    ]
    for col in [c for c in fold_df.columns if c not in {"dataset_name", "fold_id"}]:
        if col in json_cols:
            continue
        if col in count_cols:
            row[col] = int(fold_df[col].sum())
        elif fold_df[col].dtype == bool:
            row[col] = bool(fold_df[col].any())
        elif pd.api.types.is_numeric_dtype(fold_df[col]):
            row[col] = float(fold_df[col].mean())
        else:
            row[col] = fold_df[col].dropna().iat[0] if len(fold_df[col].dropna()) else ""
    total_cols = {
        "total_bnd_early_rescue_count": "bnd_early_rescue_count",
        "total_bnd_early_rescue_at_leaf_count": "bnd_early_rescue_at_leaf_count",
        "total_bnd_early_rescue_at_parent_count": "bnd_early_rescue_at_parent_count",
        "total_root_bnd_after_rescue": "root_bnd_count_after_rescue",
        "total_non_root_bnd_count": "non_root_bnd_count",
        "total_non_root_bnd_rescue_count": "non_root_bnd_rescue_count",
        "total_root_bnd_from_effective_root_count": "root_bnd_from_effective_root_count",
        "total_root_bnd_from_rescue_failed_count": "root_bnd_from_rescue_failed_count",
        "total_bnd_rescue_parent_candidate_count": "bnd_rescue_parent_candidate_count",
        "total_bnd_rescue_parent_candidate_success_count": "bnd_rescue_parent_candidate_success_count",
        "total_bnd_rescue_from_effective_root_count": "bnd_rescue_from_effective_root_count",
    }
    for output_col, source_col in total_cols.items():
        if source_col in fold_df:
            row[output_col] = int(fold_df[source_col].sum())
    avg_cols = {
        "avg_bnd_early_rescue_error_rate": "bnd_early_rescue_error_rate",
        "avg_bnd_early_rescue_regret": "bnd_early_rescue_regret",
        "avg_bnd_early_rescue_positive_rate": "bnd_early_rescue_positive_rate",
        "avg_non_root_bnd_rescue_rate": "non_root_bnd_rescue_rate",
        "avg_root_bnd_reduction_rate": "root_bnd_reduction_rate",
        "avg_rescue_vs_root_error_delta": "rescue_vs_root_error_delta",
        "avg_rescue_vs_root_regret_delta": "rescue_vs_root_regret_delta",
        "avg_bnd_rescue_parent_candidate_success_rate": "bnd_rescue_parent_candidate_success_rate",
        "avg_bnd_rescue_from_effective_root_error_rate": "bnd_rescue_from_effective_root_error_rate",
    }
    for output_col, source_col in avg_cols.items():
        if source_col in fold_df:
            row[output_col] = float(fold_df[source_col].mean())
    _refresh_classification_rates_from_counts(row)
    if "bnd_early_rescue_condition_counts" in fold_df:
        condition_counts = _sum_json_dict(fold_df["bnd_early_rescue_condition_counts"])
        row["bnd_early_rescue_condition_counts"] = _json_dump_dict(condition_counts)
        if "bnd_early_rescue_condition_error_rates" in fold_df:
            row["bnd_early_rescue_condition_error_rates"] = _json_dump_dict(
                _weighted_json_mean(
                    fold_df["bnd_early_rescue_condition_error_rates"],
                    fold_df["bnd_early_rescue_condition_counts"],
                )
            )
        if "bnd_early_rescue_condition_regrets" in fold_df:
            row["bnd_early_rescue_condition_regrets"] = _json_dump_dict(
                _weighted_json_mean(
                    fold_df["bnd_early_rescue_condition_regrets"],
                    fold_df["bnd_early_rescue_condition_counts"],
                )
            )
    return row


def _upsert_dataset_summary(summary_path: Path, rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return new_df
    new_df = new_df.drop_duplicates(subset=["dataset_name"], keep="last")

    if summary_path.exists():
        try:
            existing_df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if not existing_df.empty and "dataset_name" in existing_df.columns:
        existing_df = existing_df.drop_duplicates(subset=["dataset_name"], keep="last")
        updated_names = set(new_df["dataset_name"].astype(str))
        existing_df = existing_df[~existing_df["dataset_name"].astype(str).isin(updated_names)]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined_df = new_df

    columns = ["dataset_name"] + [col for col in combined_df.columns if col != "dataset_name"]
    combined_df = combined_df[columns]
    combined_df.to_csv(summary_path, index=False)
    return combined_df


def run(config_path: Path, cli_args: argparse.Namespace | None = None) -> None:
    root_cfg = load_yaml(config_path)
    governance_override = _merge_governance_override(root_cfg.get("governance") or root_cfg.get("GOVERNANCE"), cli_args)
    root_dir = Path(governance_override.get("output", {}).get("root_dir", "outputs/governance"))
    if not root_dir.is_absolute():
        root_dir = REPO_ROOT / root_dir
    output_root = ensure_dir(root_dir / _mode_name(governance_override))
    dataset_summaries = []
    for dataset_name, dataset_cfg_path in _dataset_items(config_path):
        dataset_out = ensure_dir(output_root / dataset_name)
        _, fold_df = run_dataset(dataset_name, dataset_cfg_path, dataset_out, governance_override=governance_override)
        dataset_summaries.append(_summarize_dataset(fold_df))
    _upsert_dataset_summary(output_root / "dataset_summary.csv", dataset_summaries)
    log_info(f"【governance】结果已写入 {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行第二篇 governance 实验框架")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--disable-cp-validation", action="store_true", help="消融：关闭 CP 后决策验证")
    parser.add_argument("--disable-progressive-update", action="store_true", help="消融：关闭渐进式风险证据更新")
    parser.add_argument("--cp-alpha", type=float, default=None, help="覆盖 CP 显著性水平 alpha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.config), cli_args=args)


if __name__ == "__main__":
    main()
