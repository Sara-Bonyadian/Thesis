from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
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
CROSS_DATASET_SCATTER_GRID_FILE = "scatter_grid_25_cross_dataset.png"
PAIRS_SUMMARY_FILE = "pairs_summary.csv"

DATASET_STYLE: dict[str, dict[str, str | float]] = {
    "ds003838": {"color": "#2c7bb6", "marker": "o"},
    "ds006848": {"color": "#d7191c", "marker": "s"},
    "hiit": {"color": "#1a9641", "marker": "^"},
}
DEFAULT_DATASET_STYLE: dict[str, str | float] = {"color": "#636363", "marker": "o"}


@dataclass(frozen=True)
class CorrPairStats:
    correlation: float
    p_value: float
    method: str
    n: int


def _format_cross_dataset_legend_label(dataset_id: str, stats: CorrPairStats) -> str:
    return (
        f"{dataset_id} | {stats.method} | "
        f"r={stats.correlation:.2f} | p={stats.p_value:.3g}"
    )


def _discover_subject_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / OBSERVATIONS_INDEX_FILE).exists()
    )


def load_merged_features(dataset_dir: Path) -> pd.DataFrame:
    combined_path = dataset_dir / MERGED_FEATURES_FILE
    if combined_path.exists():
        dtype: dict[str, type] = {}
        header = pd.read_csv(combined_path, nrows=0)
        if "subject_id" in header.columns:
            dtype["subject_id"] = str
        return pd.read_csv(combined_path, dtype=dtype)

    frames: list[pd.DataFrame] = []
    for subject_dir in _discover_subject_dirs(dataset_dir):
        path = subject_dir / MERGED_FEATURES_FILE
        if not path.exists():
            raise FileNotFoundError(f"Missing merged features for review plots: {path}")
        dtype = {}
        header = pd.read_csv(path, nrows=0)
        if "subject_id" in header.columns:
            dtype["subject_id"] = str
        frames.append(pd.read_csv(path, dtype=dtype))
    if not frames:
        raise FileNotFoundError(
            f"No {MERGED_FEATURES_FILE!r} found under {dataset_dir}. "
            f"Expected either {combined_path} or per-subject folders."
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


def all_feature_pairs(eeg_features: list[str], ppg_features: list[str]) -> list[tuple[str, str]]:
    return [(eeg, ppg) for eeg in eeg_features for ppg in ppg_features]


def _dataset_style(dataset_id: str) -> dict[str, str | float]:
    return DATASET_STYLE.get(dataset_id, DEFAULT_DATASET_STYLE)


def _short_feature_label(name: str) -> str:
    return (
        name.replace("eeg_", "")
        .replace("ppg_", "")
        .replace("_db", " (dB)")
        .replace("_uv2", "")
        .replace("_ms", " (ms)")
        .replace("_bpm", " (bpm)")
    )


def _regression_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(valid.sum()) < 2:
        return None
    x_valid = x[valid].to_numpy(dtype=float)
    y_valid = y[valid].to_numpy(dtype=float)
    coeffs = np.polyfit(x_valid, y_valid, 1)
    x_line = np.linspace(float(x_valid.min()), float(x_valid.max()), 50)
    y_line = coeffs[0] * x_line + coeffs[1]
    return x_line, y_line


def plot_cross_dataset_pair_scatter(
    ax: Any,
    *,
    merged_by_dataset: dict[str, pd.DataFrame],
    eeg_feature: str,
    ppg_feature: str,
    stats_by_dataset: dict[str, CorrPairStats] | None = None,
    show_points: bool = True,
) -> None:
    stats_by_dataset = stats_by_dataset or {}
    for dataset_id, merged in merged_by_dataset.items():
        style = _dataset_style(dataset_id)
        color = str(style["color"])
        marker = str(style["marker"])
        x = pd.to_numeric(merged[eeg_feature], errors="coerce")
        y = pd.to_numeric(merged[ppg_feature], errors="coerce")
        valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
        x_valid = x[valid]
        y_valid = y[valid]

        stats = stats_by_dataset.get(dataset_id)
        label = dataset_id
        if stats is not None:
            label = _format_cross_dataset_legend_label(dataset_id, stats)

        if show_points and not x_valid.empty:
            ax.scatter(
                x_valid,
                y_valid,
                s=18,
                alpha=0.35,
                color=color,
                marker=marker,
                edgecolors="none",
                zorder=2,
            )

        line = _regression_line(x, y)
        if line is not None:
            x_line, y_line = line
            ax.plot(x_line, y_line, color=color, linewidth=1.8, label=label, zorder=3)

    ax.set_xlabel(_short_feature_label(eeg_feature), fontsize=8)
    ax.set_ylabel(_short_feature_label(ppg_feature), fontsize=8)
    ax.set_title(f"{_short_feature_label(eeg_feature)} vs {_short_feature_label(ppg_feature)}", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            fontsize=5.5,
            loc="best",
            framealpha=0.9,
            handlelength=1.4,
            borderpad=0.3,
            labelspacing=0.25,
        )


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


def _corr_lookup(
    corr_tables: dict[str, pd.DataFrame],
    *,
    eeg_feature: str,
    ppg_feature: str,
    primary_method: str,
) -> dict[str, CorrPairStats]:
    out: dict[str, CorrPairStats] = {}
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
        selected = row.get("selected_method", row.get("method", ""))
        method = str(selected).strip() or str(row.get("method", "")).strip() or primary_method
        out[dataset_id] = CorrPairStats(
            correlation=float(row["correlation"]),
            p_value=float(row["p_value"]),
            method=method,
            n=int(row["n"]),
        )
    return out


def write_cross_dataset_scatter_grid(
    *,
    merged_by_dataset: dict[str, pd.DataFrame],
    corr_tables: dict[str, pd.DataFrame],
    eeg_features: list[str],
    ppg_features: list[str],
    compare_out: Path,
    primary_method: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    pairs = all_feature_pairs(eeg_features, ppg_features)
    if not pairs:
        return []

    compare_out.mkdir(parents=True, exist_ok=True)
    ncols = len(ppg_features)
    nrows = len(eeg_features)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for index, (eeg_feature, ppg_feature) in enumerate(pairs):
        row_idx, col_idx = divmod(index, ncols)
        stats_by_dataset = _corr_lookup(
            corr_tables,
            eeg_feature=eeg_feature,
            ppg_feature=ppg_feature,
            primary_method=primary_method,
        )
        plot_cross_dataset_pair_scatter(
            axes[row_idx][col_idx],
            merged_by_dataset=merged_by_dataset,
            eeg_feature=eeg_feature,
            ppg_feature=ppg_feature,
            stats_by_dataset=stats_by_dataset,
        )

    fig.suptitle(
        "EEG–PPG scatter grid (one regression line per dataset)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()

    grid_path = compare_out / CROSS_DATASET_SCATTER_GRID_FILE
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[dict[str, object]] = []
    for eeg_feature, ppg_feature in pairs:
        stats_by_dataset = _corr_lookup(
            corr_tables,
            eeg_feature=eeg_feature,
            ppg_feature=ppg_feature,
            primary_method=primary_method,
        )
        row: dict[str, object] = {
            "eeg_feature": eeg_feature,
            "ppg_feature": ppg_feature,
        }
        for dataset_id in merged_by_dataset:
            stats = stats_by_dataset.get(dataset_id)
            row[f"{dataset_id}_correlation"] = stats.correlation if stats else np.nan
            row[f"{dataset_id}_p_value"] = stats.p_value if stats else np.nan
            row[f"{dataset_id}_method"] = stats.method if stats else ""
            row[f"{dataset_id}_n"] = stats.n if stats else int(len(merged_by_dataset[dataset_id]))
        table_rows.append(row)

    table_path = compare_out / "scatter_grid_25_cross_dataset.csv"
    pd.DataFrame(table_rows).to_csv(table_path, index=False)
    return [grid_path, table_path]


def run_cross_dataset_scatter_grid(
    configs: list[PipelineConfig],
    *,
    compare_out: Path,
) -> list[Path]:
    merged_by_dataset: dict[str, pd.DataFrame] = {}
    corr_tables: dict[str, pd.DataFrame] = {}
    eeg_features: list[str] | None = None
    ppg_features: list[str] | None = None
    primary_method = configs[0].correlation.primary_method

    for cfg in configs:
        for dataset_id in cfg.dataset_ids:
            dataset_dir = dataset_output_dir(cfg, dataset_id)
            merged_by_dataset[dataset_id] = load_merged_features(dataset_dir)
            corr_tables[dataset_id] = load_correlations_fdr(dataset_dir)
            if eeg_features is None:
                eeg_features = list(cfg.features.eeg)
                ppg_features = list(cfg.features.ppg)

    if not merged_by_dataset or eeg_features is None or ppg_features is None:
        return []

    return write_cross_dataset_scatter_grid(
        merged_by_dataset=merged_by_dataset,
        corr_tables=corr_tables,
        eeg_features=eeg_features,
        ppg_features=ppg_features,
        compare_out=compare_out,
        primary_method=primary_method,
    )


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
        "--cross-scatter-grid",
        action="store_true",
        help=(
            "When multiple configs are passed, write a 5×5 cross-dataset scatter grid "
            "with one regression line per dataset for every EEG×PPG feature pair."
        ),
    )
    ap.add_argument(
        "--cross-scatter-grid-only",
        action="store_true",
        help=(
            "Read existing pipeline CSV outputs only and write the cross-dataset 25-panel "
            "scatter grid. Skips per-dataset review plots."
        ),
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Open interactive plot windows after saving PNGs.",
    )
    args = ap.parse_args()

    configs = [load_config(path) for path in args.config]
    written: list[Path] = []
    compare_out = Path(args.compare_out)
    write_cross_grid = len(configs) > 1 and (args.cross_scatter_grid or args.cross_scatter_grid_only)

    if not args.cross_scatter_grid_only:
        for cfg in configs:
            written.extend(run_dataset_review(cfg, max_pairs=args.max_pairs))

    if write_cross_grid:
        written.extend(run_cross_dataset_scatter_grid(configs, compare_out=compare_out))
    elif len(configs) > 1 and not args.cross_scatter_grid_only:
        written.extend(
            run_cross_dataset_review(
                configs,
                compare_out=compare_out,
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
