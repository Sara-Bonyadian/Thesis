"""Directional group cross-correlation plots: median ± bootstrap CI by peak lag sign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .config import TemporalCouplingConfig, load_config
from .cross_correlation import (
    CARDIAC_LABELS,
    PEAKS_FILENAME,
    CURVES_FILENAME,
    variable_pairs,
)
from .paths import group_output_dir, partition_key_from_row

DIRECTIONAL_GRID_PLOT = "group_directional_median_bootstrap_grid.png"
DIRECTIONAL_SUMMARY_CSV = "group_directional_median_bootstrap_summary.csv"

CARDIAC_COLOR = "#d62728"
EEG_COLOR = "#1f77b4"
DEFAULT_LAG_MIN = -30.0
DEFAULT_LAG_MAX = 30.0
DEFAULT_N_BOOT = 2000
DEFAULT_CI = 95.0


def _row_partition_key(row: pd.Series, *, dataset_id: str, hiit_partition_mode: str) -> str:
    task = str(row["task"])
    condition = str(row["condition"]) if "condition" in row.index and pd.notna(row["condition"]) else task
    observation_id = (
        str(row["observation_id"]) if "observation_id" in row.index and pd.notna(row["observation_id"]) else None
    )
    return partition_key_from_row(
        dataset_id=dataset_id,
        task=task,
        condition=condition,
        observation_id=observation_id,
        hiit_partition_mode=hiit_partition_mode,
    )


def _peak_direction_groups(peaks_df: pd.DataFrame, pair: str) -> tuple[set[str], set[str]]:
    pair_peaks = peaks_df.loc[peaks_df["pair"] == pair].copy()
    cardiac_ids: set[str] = set()
    eeg_ids: set[str] = set()
    for row in pair_peaks.itertuples(index=False):
        obs_id = str(row.observation_id)
        direction = str(getattr(row, "peak_direction", "")).casefold()
        peak_lag = getattr(row, "peak_lag_s", np.nan)
        if direction == "cardiac_leads" or (pd.isna(direction) and pd.notna(peak_lag) and float(peak_lag) > 0):
            cardiac_ids.add(obs_id)
        elif direction == "eeg_leads" or (pd.isna(direction) and pd.notna(peak_lag) and float(peak_lag) < 0):
            eeg_ids.add(obs_id)
    return cardiac_ids, eeg_ids


def _subject_curve_matrix(
    curves_df: pd.DataFrame,
    *,
    pair: str,
    observation_ids: set[str],
    lag_grid: np.ndarray,
) -> np.ndarray | None:
    if not observation_ids:
        return None
    sub = curves_df.loc[
        (curves_df["pair"] == pair) & (curves_df["observation_id"].astype(str).isin(observation_ids))
    ]
    if sub.empty:
        return None

    rows: list[np.ndarray] = []
    for obs_id in sorted(observation_ids):
        obs_curve = sub.loc[sub["observation_id"].astype(str) == obs_id].sort_values("lag_s")
        if obs_curve.empty:
            continue
        aligned = np.full(lag_grid.shape, np.nan, dtype=float)
        lag_to_r = dict(zip(obs_curve["lag_s"].astype(float), obs_curve["r"].astype(float), strict=False))
        for idx, lag in enumerate(lag_grid):
            if lag in lag_to_r:
                aligned[idx] = lag_to_r[lag]
        if np.isfinite(aligned).any():
            rows.append(aligned)
    if not rows:
        return None
    return np.vstack(rows)


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    n_boot: int,
    ci_percent: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median, lower CI, upper CI across lag columns."""
    n_subjects, _ = values.shape
    if n_subjects == 0:
        empty = np.full(values.shape[1], np.nan)
        return empty, empty, empty

    rng = np.random.default_rng(seed)
    boot_medians = np.full((n_boot, values.shape[1]), np.nan, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_subjects, size=n_subjects)
        sample = values[idx]
        boot_medians[b] = np.nanmedian(sample, axis=0)

    alpha = (100.0 - ci_percent) / 2.0
    lower_q = alpha
    upper_q = 100.0 - alpha
    median = np.nanmedian(values, axis=0)
    lower = np.nanpercentile(boot_medians, lower_q, axis=0)
    upper = np.nanpercentile(boot_medians, upper_q, axis=0)
    return median, lower, upper


def _plot_directional_panel(
    ax: plt.Axes,
    *,
    pair: str,
    curves_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    lag_grid: np.ndarray,
    n_boot: int,
    ci_percent: float,
    seed: int,
) -> dict[str, int]:
    cardiac_ids, eeg_ids = _peak_direction_groups(peaks_df, pair)
    counts = {"cardiac_leads": len(cardiac_ids), "eeg_leads": len(eeg_ids)}

    for group_name, obs_ids, color in (
        ("cardiac_leads", cardiac_ids, CARDIAC_COLOR),
        ("eeg_leads", eeg_ids, EEG_COLOR),
    ):
        matrix = _subject_curve_matrix(curves_df, pair=pair, observation_ids=obs_ids, lag_grid=lag_grid)
        if matrix is None or matrix.shape[0] == 0:
            continue
        finite_rows = np.isfinite(matrix).any(axis=1)
        matrix = matrix[finite_rows]
        if matrix.shape[0] == 0:
            continue
        median, lower, upper = bootstrap_median_ci(
            matrix,
            n_boot=n_boot,
            ci_percent=ci_percent,
            seed=seed + hash((pair, group_name)) % 10_000,
        )
        valid = np.isfinite(median)
        if not valid.any():
            continue
        x = lag_grid[valid]
        ax.plot(x, median[valid], color=color, linewidth=1.8, zorder=3)
        ci_mask = valid & np.isfinite(lower) & np.isfinite(upper)
        if ci_mask.any():
            ax.fill_between(
                x[ci_mask],
                lower[ci_mask],
                upper[ci_mask],
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )

    ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_title(
        f"{pair}\n cardiac+ n={counts['cardiac_leads']} | eeg+ n={counts['eeg_leads']}",
        fontsize=9,
    )
    return counts


def plot_directional_median_bootstrap_grid(
    curves_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    output_path: Path,
    *,
    lag_min: float = DEFAULT_LAG_MIN,
    lag_max: float = DEFAULT_LAG_MAX,
    lag_step: float = 5.0,
    n_boot: int = DEFAULT_N_BOOT,
    ci_percent: float = DEFAULT_CI,
    seed: int = 0,
    partition: str | None = None,
) -> Path:
    lag_grid = np.arange(lag_min, lag_max + 0.5 * lag_step, lag_step, dtype=float)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    summary_rows: list[dict[str, object]] = []

    for idx, pair in enumerate(variable_pairs()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        counts = _plot_directional_panel(
            ax,
            pair=pair.pair,
            curves_df=curves_df,
            peaks_df=peaks_df,
            lag_grid=lag_grid,
            n_boot=n_boot,
            ci_percent=ci_percent,
            seed=seed,
        )
        summary_rows.append(
            {
                "partition": partition or "",
                "pair": pair.pair,
                "n_cardiac_leads": counts["cardiac_leads"],
                "n_eeg_leads": counts["eeg_leads"],
                "lag_min_s": lag_min,
                "lag_max_s": lag_max,
            }
        )
        if col == 0:
            ax.set_ylabel(f"{CARDIAC_LABELS[row]}\nr")
        if row == 2:
            ax.set_xlabel("Lag (s)")
        ax.set_xlim(lag_min, lag_max)

    title_partition = f" ({partition})" if partition else ""
    fig.suptitle(
        "Group cross-correlation by peak-lag direction\n"
        f"median ± bootstrap {ci_percent:g}% CI, lag [{lag_min:g}, {lag_max:g}] s{title_partition}",
        fontsize=12,
        y=0.99,
    )
    handles = [
        Line2D([0], [0], color=CARDIAC_COLOR, linewidth=1.8, label="cardiac-leading (peak lag > 0)"),
        Line2D([0], [0], color=EEG_COLOR, linewidth=1.8, label="EEG-leading (peak lag < 0)"),
        Line2D([0], [0], color=CARDIAC_COLOR, alpha=0.20, linewidth=6, label=f"bootstrap {ci_percent:g}% CI"),
        Line2D([0], [0], color="0.7", linestyle="--", linewidth=0.8, label="lag 0"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

    summary_path = output_path.with_name(DIRECTIONAL_SUMMARY_CSV)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return output_path


def _filter_partition(
    df: pd.DataFrame,
    *,
    partition: str,
    dataset_id: str,
    hiit_partition_mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df.apply(
        lambda row: _row_partition_key(row, dataset_id=dataset_id, hiit_partition_mode=hiit_partition_mode) == partition,
        axis=1,
    )
    return df.loc[mask].copy()


def run_directional_group_plots(
    cfg: TemporalCouplingConfig,
    *,
    partitions: list[str] | None = None,
    lag_min: float = DEFAULT_LAG_MIN,
    lag_max: float = DEFAULT_LAG_MAX,
    n_boot: int = DEFAULT_N_BOOT,
    ci_percent: float = DEFAULT_CI,
    seed: int = 0,
) -> list[Path]:
    group_dir = group_output_dir(cfg)
    curves_path = group_dir / CURVES_FILENAME
    peaks_path = group_dir / PEAKS_FILENAME
    if not curves_path.is_file():
        raise FileNotFoundError(f"Missing curves file: {curves_path}. Run --stage 2 first.")
    if not peaks_path.is_file():
        raise FileNotFoundError(f"Missing peaks file: {peaks_path}. Run --stage 2 first.")

    curves_df = pd.read_csv(curves_path)
    peaks_df = pd.read_csv(peaks_path)
    lag_step = float(cfg.temporal_coupling.cross_correlation.lag_step_s)

    if partitions is None:
        partitions = sorted(
            {
                _row_partition_key(row, dataset_id=cfg.dataset_id, hiit_partition_mode=cfg.hiit_partition_mode)
                for _, row in peaks_df.iterrows()
            }
        )

    written: list[Path] = []
    for partition in partitions:
        part_curves = _filter_partition(
            curves_df,
            partition=partition,
            dataset_id=cfg.dataset_id,
            hiit_partition_mode=cfg.hiit_partition_mode,
        )
        part_peaks = _filter_partition(
            peaks_df,
            partition=partition,
            dataset_id=cfg.dataset_id,
            hiit_partition_mode=cfg.hiit_partition_mode,
        )
        if part_curves.empty or part_peaks.empty:
            print(f"[directional_plot] skipping partition={partition!r}: no data.")
            continue

        out_dir = group_dir if len(partitions) == 1 else group_output_dir(cfg, partition)
        out_path = out_dir / DIRECTIONAL_GRID_PLOT
        plot_directional_median_bootstrap_grid(
            part_curves,
            part_peaks,
            out_path,
            lag_min=lag_min,
            lag_max=lag_max,
            lag_step=lag_step,
            n_boot=n_boot,
            ci_percent=ci_percent,
            seed=seed,
            partition=partition,
        )
        written.append(out_path)
        print(f"[directional_plot] wrote {out_path}")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot median ± bootstrap CI cross-correlation curves split by peak-lag direction.",
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to exploratory temporal-coupling YAML "
            "(under exploratory-temporal-coupling/)."
        ),
    )
    ap.add_argument(
        "--partition",
        type=str,
        default="all",
        help="Task partition to plot (e.g. pre_rest) or 'all' for every partition (default).",
    )
    ap.add_argument("--lag-min", type=float, default=DEFAULT_LAG_MIN)
    ap.add_argument("--lag-max", type=float, default=DEFAULT_LAG_MAX)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--ci-percent", type=float, default=DEFAULT_CI)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        partitions = None if args.partition.casefold() == "all" else [args.partition]
        run_directional_group_plots(
            cfg,
            partitions=partitions,
            lag_min=args.lag_min,
            lag_max=args.lag_max,
            n_boot=args.n_boot,
            ci_percent=args.ci_percent,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[directional_plot] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
