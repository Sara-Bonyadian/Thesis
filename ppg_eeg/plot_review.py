from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig, load_config
from .correlation import significance_symbol
from .output_layout import dataset_output_dir
from .pipeline import CORRELATIONS_FDR_FILE, MERGED_FEATURES_FILE, OBSERVATIONS_INDEX_FILE


REVIEW_SUBDIR = "review"
SCATTER_GRID_FILE = "scatter_grid.png"
CROSS_DATASET_BARS_FILE = "correlations_cross_dataset.png"
CROSS_DATASET_TABLE_FILE = "correlations_cross_dataset.csv"
PAIRS_SUMMARY_FILE = "pairs_summary.csv"


def _discover_subject_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / OBSERVATIONS_INDEX_FILE).exists()
    )


def load_merged_features(dataset_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for subject_dir in _discover_subject_dirs(dataset_dir):
        path = subject_dir / MERGED_FEATURES_FILE
        if not path.exists():
            raise FileNotFoundError(f"Missing merged features for review plots: {path}")
        dtype: dict[str, type] = {}
        header = pd.read_csv(path, nrows=0)
        if "subject_id" in header.columns:
            dtype["subject_id"] = str
        frames.append(pd.read_csv(path, dtype=dtype))
    if not frames:
        raise FileNotFoundError(
            f"No subject folders with {MERGED_FEATURES_FILE!r} found under {dataset_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def load_correlations_fdr(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / CORRELATIONS_FDR_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing correlations table for review plots: {path}")
    return pd.read_csv(path)


def select_review_pairs(
    corr_fdr: pd.DataFrame,
    *,
    primary_method: str,
    max_pairs: int,
) -> pd.DataFrame:
    if corr_fdr.empty or max_pairs <= 0:
        return corr_fdr.iloc[0:0].copy()

    focus = corr_fdr.copy()
    if "method" in focus.columns:
        method_mask = focus["method"].astype(str).str.casefold() == primary_method.casefold()
        if method_mask.any():
            focus = focus.loc[method_mask].copy()

    def _priority(row: pd.Series) -> int:
        if bool(row.get("sig_q_005")):
            return 0
        if bool(row.get("sig_q_010")):
            return 1
        if bool(row.get("sig_q_015")):
            return 2
        if bool(row.get("sig_p_005")):
            return 3
        return 4

    focus["_priority"] = focus.apply(_priority, axis=1)
    focus["_abs_r"] = pd.to_numeric(focus["correlation"], errors="coerce").abs()
    focus = focus.sort_values(["_priority", "_abs_r"], ascending=[True, False])
    return focus.drop(columns=["_priority", "_abs_r"]).head(max_pairs).reset_index(drop=True)


def _pair_slug(eeg_feature: str, ppg_feature: str) -> str:
    return f"{eeg_feature}__{ppg_feature}"


def _scatter_title(row: pd.Series, *, dataset_id: str) -> str:
    marker = significance_symbol(q_value=row.get("q_value"), p_value=row.get("p_value"))
    selected = row.get("selected_method", row.get("method", ""))
    return (
        f"{dataset_id}: {row['eeg_feature']} vs {row['ppg_feature']}\n"
        f"r={float(row['correlation']):.2f}{marker}  "
        f"p={float(row['p_value']):.3g}  q={float(row['q_value']):.3g}  "
        f"n={int(row['n'])} ({selected})"
    )


def plot_correlation_scatter(
    merged: pd.DataFrame,
    row: pd.Series,
    *,
    dataset_id: str,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    eeg_feature = str(row["eeg_feature"])
    ppg_feature = str(row["ppg_feature"])
    x = pd.to_numeric(merged[eeg_feature], errors="coerce")
    y = pd.to_numeric(merged[ppg_feature], errors="coerce")
    labels = merged["subject_id"].astype(str)

    created = ax is None
    if created:
        _fig, ax = plt.subplots(figsize=(5.0, 4.0))
    else:
        _fig = ax.figure

    valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    x_valid = x[valid]
    y_valid = y[valid]
    labels_valid = labels[valid]

    ax.scatter(x_valid, y_valid, s=55, color="#2c7bb6", edgecolors="white", linewidths=0.6, zorder=3)
    for x_val, y_val, label in zip(x_valid, y_valid, labels_valid, strict=True):
        ax.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=8)

    if len(x_valid) >= 2:
        coeffs = np.polyfit(x_valid.to_numpy(dtype=float), y_valid.to_numpy(dtype=float), 1)
        x_line = np.linspace(float(x_valid.min()), float(x_valid.max()), 50)
        ax.plot(x_line, coeffs[0] * x_line + coeffs[1], color="#d7191c", linewidth=1.2, zorder=2)

    ax.set_xlabel(eeg_feature)
    ax.set_ylabel(ppg_feature)
    ax.set_title(_scatter_title(row, dataset_id=dataset_id), fontsize=10)
    ax.grid(True, alpha=0.25)
    if created:
        _fig.tight_layout()
    return _fig, ax


def write_dataset_review_plots(
    *,
    dataset_dir: Path,
    corr_fdr: pd.DataFrame,
    merged: pd.DataFrame,
    dataset_id: str,
    primary_method: str,
    max_pairs: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    review_dir = dataset_dir / REVIEW_SUBDIR
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs = select_review_pairs(corr_fdr, primary_method=primary_method, max_pairs=max_pairs)
    pairs.to_csv(review_dir / PAIRS_SUMMARY_FILE, index=False)
    if pairs.empty:
        return []

    written: list[Path] = []
    for _, row in pairs.iterrows():
        fig, _ax = plot_correlation_scatter(merged, row, dataset_id=dataset_id)
        out_path = review_dir / f"scatter_{_pair_slug(row['eeg_feature'], row['ppg_feature'])}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = math.ceil(n_pairs / ncols)
    grid_fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), squeeze=False)
    for index, (_, row) in enumerate(pairs.iterrows()):
        row_idx, col_idx = divmod(index, ncols)
        plot_correlation_scatter(
            merged,
            row,
            dataset_id=dataset_id,
            ax=axes[row_idx][col_idx],
        )
    for index in range(n_pairs, nrows * ncols):
        row_idx, col_idx = divmod(index, ncols)
        axes[row_idx][col_idx].axis("off")

    grid_path = review_dir / SCATTER_GRID_FILE
    grid_fig.suptitle(f"{dataset_id} review scatters", fontsize=12, y=1.02)
    grid_fig.tight_layout()
    grid_fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(grid_fig)
    written.append(grid_path)
    return written


def _pairs_for_cross_dataset(corr_tables: dict[str, pd.DataFrame], *, max_pairs: int) -> list[tuple[str, str]]:
    scored: dict[tuple[str, str], float] = {}
    for corr_fdr in corr_tables.values():
        for _, row in corr_fdr.iterrows():
            key = (str(row["eeg_feature"]), str(row["ppg_feature"]))
            abs_r = abs(float(row["correlation"]))
            scored[key] = max(scored.get(key, 0.0), abs_r)
    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    return [pair for pair, _score in ranked[:max_pairs]]


def write_cross_dataset_review_plots(
    *,
    corr_tables: dict[str, pd.DataFrame],
    compare_out: Path,
    primary_method: str,
    max_pairs: int,
) -> list[Path]:
    import matplotlib.pyplot as plt

    if len(corr_tables) < 2:
        return []

    compare_out.mkdir(parents=True, exist_ok=True)
    pairs = _pairs_for_cross_dataset(corr_tables, max_pairs=max_pairs)
    if not pairs:
        return []

    rows: list[dict[str, object]] = []
    for eeg_feature, ppg_feature in pairs:
        for dataset_id, corr_fdr in corr_tables.items():
            focus = corr_fdr[
                (corr_fdr["eeg_feature"].astype(str) == eeg_feature)
                & (corr_fdr["ppg_feature"].astype(str) == ppg_feature)
            ]
            if "method" in focus.columns:
                method_mask = focus["method"].astype(str).str.casefold() == primary_method.casefold()
                if method_mask.any():
                    focus = focus.loc[method_mask]
            if focus.empty:
                continue
            row = focus.iloc[0]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "eeg_feature": eeg_feature,
                    "ppg_feature": ppg_feature,
                    "correlation": float(row["correlation"]),
                    "p_value": float(row["p_value"]),
                    "q_value": float(row.get("q_value", np.nan)),
                    "n": int(row["n"]),
                    "pair_label": f"{eeg_feature}\nvs\n{ppg_feature}",
                }
            )

    merged = pd.DataFrame(rows)
    table_path = compare_out / CROSS_DATASET_TABLE_FILE
    merged.to_csv(table_path, index=False)
    if merged.empty:
        return [table_path]

    pair_labels = [_pair_slug(e, p).replace("__", "\nvs\n") for e, p in pairs]
    dataset_ids = list(corr_tables.keys())
    x = np.arange(len(pairs))
    width = 0.8 / len(dataset_ids)

    fig, ax = plt.subplots(figsize=(max(8.0, len(pairs) * 1.6), 5.5))
    for index, dataset_id in enumerate(dataset_ids):
        subset = merged[merged["dataset_id"] == dataset_id].set_index(["eeg_feature", "ppg_feature"])
        heights = []
        for pair in pairs:
            heights.append(float(subset.loc[pair, "correlation"]) if pair in subset.index else np.nan)
        offset = (index - (len(dataset_ids) - 1) / 2) * width
        ax.bar(x + offset, heights, width=width, label=dataset_id)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Correlation (r)")
    ax.set_title(f"Cross-dataset correlation comparison ({primary_method})")
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.25)

    bars_path = compare_out / CROSS_DATASET_BARS_FILE
    fig.tight_layout()
    fig.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [table_path, bars_path]


def run_dataset_review(cfg: PipelineConfig, *, max_pairs: int) -> list[Path]:
    written: list[Path] = []
    for dataset_id in cfg.dataset_ids:
        dataset_dir = dataset_output_dir(cfg, dataset_id)
        corr_fdr = load_correlations_fdr(dataset_dir)
        merged = load_merged_features(dataset_dir)
        written.extend(
            write_dataset_review_plots(
                dataset_dir=dataset_dir,
                corr_fdr=corr_fdr,
                merged=merged,
                dataset_id=dataset_id,
                primary_method=cfg.correlation.primary_method,
                max_pairs=max_pairs,
            )
        )
    return written


def run_cross_dataset_review(configs: list[PipelineConfig], *, compare_out: Path, max_pairs: int) -> list[Path]:
    corr_tables: dict[str, pd.DataFrame] = {}
    primary_method = configs[0].correlation.primary_method
    for cfg in configs:
        for dataset_id in cfg.dataset_ids:
            dataset_dir = dataset_output_dir(cfg, dataset_id)
            corr_tables[dataset_id] = load_correlations_fdr(dataset_dir)
            if cfg.correlation.primary_method != primary_method:
                primary_method = configs[0].correlation.primary_method
    return write_cross_dataset_review_plots(
        corr_tables=corr_tables,
        compare_out=compare_out,
        primary_method=primary_method,
        max_pairs=max_pairs,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate review scatter plots and cross-dataset comparison figures from pipeline outputs.",
    )
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        help="Pipeline YAML config. Pass multiple times to add cross-dataset comparison plots.",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=6,
        help="Maximum EEG×PPG pairs to plot per dataset (default: 6).",
    )
    ap.add_argument(
        "--compare-out",
        type=str,
        default="./derivatives/review_cross_dataset",
        help="Output folder for cross-dataset plots when multiple configs are passed.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open interactive plot windows after saving PNGs.",
    )
    args = ap.parse_args()

    configs = [load_config(path) for path in args.config]
    written: list[Path] = []
    for cfg in configs:
        written.extend(run_dataset_review(cfg, max_pairs=args.max_pairs))

    if len(configs) > 1:
        written.extend(
            run_cross_dataset_review(
                configs,
                compare_out=Path(args.compare_out),
                max_pairs=args.max_pairs,
            )
        )

    for path in written:
        print(path)

    if args.show and written:
        import matplotlib.pyplot as plt

        for path in written:
            if path.suffix.lower() != ".png":
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()


if __name__ == "__main__":
    main()
