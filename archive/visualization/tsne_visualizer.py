from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from .bttwd_model import BTTWDModel, _XGB_AVAILABLE
from .bucket_rules import BucketTree
from .config_loader import load_yaml_cfg
from .data_loader import load_dataset
from .preprocessing import prepare_features_and_labels
from .utils_logging import log_info
from .utils_seed import set_global_seed


def _ensure_estimators(cfg: dict, force_logreg_global: bool) -> None:
    """
    Safely set global/bucket estimators to avoid training failures caused by missing
    xgboost dependencies or implicit configuration.
    """

    bcfg = cfg.setdefault("BTTWD", {})

    bucket_est = bcfg.get("bucket_estimator") or bcfg.get("posterior_estimator")
    if bucket_est is None:
        bucket_est = "logreg"
    bucket_est_lower = str(bucket_est).lower()
    if bucket_est_lower in {"xgb", "xgboost"} and not _XGB_AVAILABLE:
        log_info("[t-SNE] xgboost not detected; bucket_estimator falls back to logreg")
        bucket_est = "logreg"
    bcfg["bucket_estimator"] = bucket_est
    bcfg["posterior_estimator"] = bucket_est

    global_est = bcfg.get("global_estimator", "logreg")
    global_est_lower = str(global_est).lower()
    if force_logreg_global or (global_est_lower in {"xgb", "xgboost"} and not _XGB_AVAILABLE):
        if global_est_lower in {"xgb", "xgboost"} and not _XGB_AVAILABLE:
            log_info("[t-SNE] xgboost not detected; global_estimator falls back to logreg")
        elif force_logreg_global and global_est_lower != "logreg":
            log_info("[t-SNE] force_logreg_global=True; global_estimator forced to logreg")
        bcfg["global_estimator"] = "logreg"


def _resolve_bucket_cols(cfg: dict, df_processed: pd.DataFrame) -> list[str]:
    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols: list[str] = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])

    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    for lvl in bucket_levels:
        col_name = lvl.get("col") or lvl.get("feature")
        if col_name and col_name not in bucket_cols:
            bucket_cols.append(col_name)

    missing_cols = [col for col in bucket_cols if col not in df_processed.columns]
    if missing_cols:
        raise KeyError(f"Missing bucket features: {', '.join(missing_cols)}")

    return bucket_cols


def _prepare_dataset(cfg: dict, sample_size: int | None, random_state: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, BucketTree]:
    df_raw, target_col = load_dataset(cfg)
    if "split" in df_raw.columns:
        train_mask = df_raw["split"].astype(str).str.lower() == "train"
        if train_mask.any():
            df_raw = df_raw[train_mask].reset_index(drop=True)
            log_info(f"[t-SNE] Detected split column; using training data for visualization, target={target_col}")

    X, y, meta = prepare_features_and_labels(df_raw, cfg)
    df_processed = meta.get("df_processed", df_raw)
    bucket_cols = _resolve_bucket_cols(cfg, df_processed)
    bucket_df = df_processed[bucket_cols].reset_index(drop=True)

    if sample_size is not None and sample_size > 0 and len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        bucket_df = bucket_df.iloc[indices].reset_index(drop=True)
        log_info(f"[t-SNE] Sampled {sample_size} rows for visualization (original N={len(df_raw)})")

    bucket_tree = BucketTree(cfg.get("BTTWD", {}).get("bucket_levels", []), feature_names=bucket_cols)
    return X, y, bucket_df, bucket_tree


def _compute_tsne_embedding(
    X: np.ndarray, perplexity: float, learning_rate: float, random_state: int
) -> np.ndarray:
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    n_samples = X.shape[0]
    max_perplexity = max(5.0, (n_samples - 1) / 3.0)
    adjusted_perplexity = min(perplexity, max_perplexity)
    if adjusted_perplexity < 1.0:
        adjusted_perplexity = 1.0

    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        init="pca",
        n_iter=1000,
        verbose=1,
    )
    embedding = tsne.fit_transform(X)
    log_info(
        f"[t-SNE] Finished embedding computation; adjusted perplexity={adjusted_perplexity:.2f}, "
        f"output shape={embedding.shape}"
    )
    return embedding


def _estimate_dbscan_eps(embedding: np.ndarray, k_neighbors: int = 10, quantile: float = 60.0) -> float:
    n_samples = len(embedding)
    if n_samples <= 1:
        return 0.5

    k = max(2, min(k_neighbors, n_samples))
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(embedding)
    distances, _ = nbrs.kneighbors(embedding)
    kth_distances = distances[:, -1]
    return float(np.percentile(kth_distances, quantile))


def _find_dense_region(df_mode: pd.DataFrame, min_samples: int = 10) -> dict[str, Any] | None:
    embedding = df_mode[["tsne_x", "tsne_y"]].to_numpy()
    n_samples = len(embedding)
    if n_samples < max(2, min_samples):
        return None

    eps = _estimate_dbscan_eps(embedding, k_neighbors=min(10, n_samples - 1))
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embedding)
    label_counts = pd.Series(labels[labels >= 0]).value_counts()
    if label_counts.empty:
        return None

    target_label = label_counts.idxmax()
    dense_mask = labels == target_label
    dense_points = embedding[dense_mask]
    x_min, y_min = dense_points.min(axis=0)
    x_max, y_max = dense_points.max(axis=0)
    padding = 0.1 * max(x_max - x_min, y_max - y_min, 1e-6)

    return {
        "mask": dense_mask,
        "xlim": (x_min - padding, x_max + padding),
        "ylim": (y_min - padding, y_max + padding),
    }


def _collect_mode_result(
    cfg: dict,
    mode_label: str,
    bucket_tree: BucketTree,
    X: np.ndarray,
    y: np.ndarray,
    bucket_df: pd.DataFrame,
    embedding: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    cfg_mode = deepcopy(cfg)
    bcfg = cfg_mode.setdefault("BTTWD", {})
    bcfg["use_gain_weak_backoff"] = mode_label == "fallback_on"
    cfg_mode.setdefault("OUTPUT", {}).update({"run_name": f"{mode_label}_tsne"})

    tree_copy = BucketTree(bucket_tree.levels_cfg, feature_names=bucket_tree.feature_names)
    model = BTTWDModel.from_cfg(cfg_mode, feature_names=bucket_tree.feature_names, bucket_tree=tree_copy)
    model.fit(X, y, bucket_df)

    bucket_ids = model.bucket_tree.assign_buckets(bucket_df).astype(str)
    y_pred_s3 = model.predict(X, bucket_df)
    y_score = model.predict_proba(X, bucket_df)

    fallback_stats = model.fallback_stats or {}
    effective_map = {bid: rec.get("effective_bucket_id", bid) for bid, rec in fallback_stats.items()}
    status_map = {bid: info.get("status") for bid, info in model.bucket_info.items()}

    df_mode = pd.DataFrame(
        {
            "mode": mode_label,
            "tsne_x": embedding[:, 0],
            "tsne_y": embedding[:, 1],
            "y_true": y,
            "y_pred": y_pred_s3,
            "y_score": y_score,
            "bucket_id": bucket_ids,
        }
    )
    df_mode["effective_bucket_id"] = df_mode["bucket_id"].map(lambda bid: effective_map.get(bid, bid))
    df_mode["bucket_status"] = df_mode["bucket_id"].map(lambda bid: status_map.get(bid, "unknown"))
    df_mode["fallback_used"] = df_mode["bucket_id"] != df_mode["effective_bucket_id"]

    csv_path = output_root / f"{mode_label}_tsne_embedding.csv"
    df_mode.to_csv(csv_path, index=False)
    log_info(f"[t-SNE] Saved {mode_label} embedding data to: {csv_path}")

    bucket_stats_df = model.get_bucket_stats()
    bucket_stats_path = output_root / f"{mode_label}_bucket_stats.csv"
    if not bucket_stats_df.empty:
        bucket_stats_df.to_csv(bucket_stats_path, index=False)
    else:
        bucket_stats_path.touch()

    fallback_stats_path = output_root / f"{mode_label}_fallback_stats.csv"
    if fallback_stats:
        pd.DataFrame(fallback_stats.values()).to_csv(fallback_stats_path, index=False)
    else:
        fallback_stats_path.touch()

    summary = {
        "mode": mode_label,
        "n_samples": int(len(df_mode)),
        "fallback_samples": int(df_mode["fallback_used"].sum()),
        "fallback_ratio": float(df_mode["fallback_used"].mean()),
        "unique_buckets": int(df_mode["bucket_id"].nunique()),
        "effective_buckets": int(df_mode["effective_bucket_id"].nunique()),
        "weak_buckets": int(sum(info.get("status") == "weak" for info in model.bucket_info.values())),
    }

    return {
        "mode": mode_label,
        "df": df_mode,
        "summary": summary,
        "bucket_stats_path": bucket_stats_path,
        "fallback_stats_path": fallback_stats_path,
    }


def _plot_tsne_modes(
    results: list[dict[str, Any]], png_path: Path, pdf_path: Path, point_size: float, dataset_name: str
) -> None:
    n_modes = len(results)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), sharex=True, sharey=True)
    if n_modes == 1:
        axes = [axes]

    local_color_negative = "#1f77b4"
    local_color_positive = "#2ca02c"
    fallback_color_negative = "#ff7f0e"
    fallback_color_positive = "#d62728"
    label_colors_local = {0: local_color_negative, 1: local_color_positive}
    label_colors_fallback = {0: fallback_color_negative, 1: fallback_color_positive}

    def _target_colors(targets: pd.Series, fallback: bool) -> list[str]:
        label_colors = label_colors_fallback if fallback else label_colors_local
        default_color = label_colors.get(0, local_color_negative)
        return targets.map(lambda value: label_colors.get(int(value), default_color)).tolist()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=local_color_negative,
            markeredgecolor=local_color_negative,
            markersize=6,
            label="Local decision, y=0 (negative)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=local_color_positive,
            markeredgecolor=local_color_positive,
            markersize=6,
            label="Local decision, y=1 (positive)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=fallback_color_negative,
            markersize=6,
            label="Backoff decision, y=0 (negative)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=fallback_color_positive,
            markersize=6,
            label="Backoff decision, y=1 (positive)",
        ),
    ]

    for ax, res in zip(axes, results):
        df_mode = res["df"]
        show_fallback = res["mode"] != "fallback_off"
        fallback_mask = df_mode["fallback_used"] if show_fallback else pd.Series(False, index=df_mode.index)

        local_scatter = ax.scatter(
            df_mode.loc[~fallback_mask, "tsne_x"],
            df_mode.loc[~fallback_mask, "tsne_y"],
            s=point_size,
            alpha=0.6,
            c=_target_colors(df_mode.loc[~fallback_mask, "y_true"], fallback=False),
        )

        if show_fallback:
            fallback_scatter = ax.scatter(
                df_mode.loc[fallback_mask, "tsne_x"],
                df_mode.loc[fallback_mask, "tsne_y"],
                s=point_size * 1.2,
                alpha=0.7,
                c=_target_colors(df_mode.loc[fallback_mask, "y_true"], fallback=True),
                marker="x",
            )

        dense_region = _find_dense_region(df_mode)
        handles = list(legend_handles)
        labels = [handle.get_label() for handle in handles]
        if dense_region:
            inset_ax = inset_axes(ax, width="40%", height="40%", loc="upper left", borderpad=1)
            inset_ax.scatter(
                df_mode.loc[~fallback_mask & dense_region["mask"], "tsne_x"],
                df_mode.loc[~fallback_mask & dense_region["mask"], "tsne_y"],
                s=point_size * 2,
                alpha=0.75,
                c=_target_colors(
                    df_mode.loc[~fallback_mask & dense_region["mask"], "y_true"], fallback=False
                ),
            )
            if show_fallback:
                inset_ax.scatter(
                    df_mode.loc[fallback_mask & dense_region["mask"], "tsne_x"],
                    df_mode.loc[fallback_mask & dense_region["mask"], "tsne_y"],
                    s=point_size * 2.4,
                    alpha=0.85,
                    c=_target_colors(
                        df_mode.loc[fallback_mask & dense_region["mask"], "y_true"], fallback=True
                    ),
                    marker="x",
                )
            inset_ax.set_xlim(*dense_region["xlim"])
            inset_ax.set_ylim(*dense_region["ylim"])
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            # inset title intentionally omitted to keep the inset clean
            rect, connector1, connector2 = mark_inset(
                ax,
                inset_ax,
                loc1=1,
                loc2=3,
                fc="none",
                ec="yellow",
                lw=2.0,
                linestyle="--",
            )
            rect.set_label("Dense region")
            handles.append(rect)
            labels.append("Dense region")

        mode_title = "Backoff On" if res["mode"] == "fallback_on" else "Backoff Off"
        ax.set_title(f"{mode_title} (backoff ratio={df_mode['fallback_used'].mean():.1%})")
        ax.set_xlabel("t-SNE dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE dimension 2", fontsize=11)
        ax.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            markerscale=0.85,
            borderpad=0.5,
            handlelength=1.8,
        )

    fig.suptitle(
        f"Visualization of the threshold backoff mechanism in t-SNE space ({dataset_name})", fontsize=14
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"[t-SNE] Comparison figure saved to: {png_path}")
    log_info(f"[t-SNE] Comparison figure saved to: {pdf_path}")


def visualize_fallback_with_tsne(
    config_path: str,
    output_dir: str = "results/tsne_fallback",
    sample_size: int | None = None,
    perplexity: float | None = None,
    learning_rate: float | None = None,
    random_state: int | None = None,
    force_logreg_global: bool = False,
) -> dict:
    # Load configuration
    cfg = load_yaml_cfg(config_path)
    tsne_cfg = cfg.get("tSNE") or cfg.get("TSNE") or {}

    effective_sample_size = sample_size if sample_size is not None else tsne_cfg.get("sample_size", 2000)
    if effective_sample_size is not None:
        effective_sample_size = int(effective_sample_size)
        if effective_sample_size <= 0:
            effective_sample_size = None

    effective_perplexity = perplexity if perplexity is not None else tsne_cfg.get("perplexity", 30.0)
    effective_learning_rate = learning_rate if learning_rate is not None else tsne_cfg.get("learning_rate", 200.0)
    effective_random_state = random_state if random_state is not None else tsne_cfg.get("random_state", 42)
    effective_point_size = float(tsne_cfg.get("point_size", 10))

    effective_perplexity = float(effective_perplexity)
    effective_learning_rate = float(effective_learning_rate)
    effective_random_state = int(effective_random_state)

    set_global_seed(effective_random_state)

    # Ensure estimators are selected correctly
    _ensure_estimators(cfg, force_logreg_global)

    # Prepare dataset
    X, y, bucket_df, bucket_tree = _prepare_dataset(cfg, effective_sample_size, effective_random_state)

    # Compute t-SNE embedding
    embedding = _compute_tsne_embedding(
        X,
        effective_perplexity,
        effective_learning_rate,
        effective_random_state,
    )

    # Create output directory
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Collect results and save
    results = []
    for mode_label in ("fallback_on", "fallback_off"):
        results.append(
            _collect_mode_result(cfg, mode_label, bucket_tree, X, y, bucket_df, embedding, output_root)
        )

    combined_df = pd.concat([res["df"] for res in results], ignore_index=True)
    combined_path = output_root / "tsne_fallback_embedding.csv"
    combined_df.to_csv(combined_path, index=False)
    log_info(f"[t-SNE] Saved t-SNE embedding with mode labels: {combined_path}")

    # Summarize results
    summary_df = pd.DataFrame([res["summary"] for res in results])
    summary_path = output_root / "tsne_fallback_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Export figure
    dataset_name = str((cfg.get("DATA") or {}).get("dataset_name", "dataset"))
    dataset_slug = re.sub(r"[^0-9A-Za-z]+", "_", dataset_name).strip("_").lower() or "dataset"
    figure_png_path = output_root / f"tsne_backoff_{dataset_slug}.png"
    figure_pdf_path = output_root / f"tsne_backoff_{dataset_slug}.pdf"
    _plot_tsne_modes(results, figure_png_path, figure_pdf_path, effective_point_size, dataset_name)

    # Return result paths
    return {
        "embedding_path": combined_path,
        "figure_path": figure_png_path,
        "figure_pdf_path": figure_pdf_path,
        "summary_path": summary_path,
        "results": results,
    }
