"""Stage 3: group-level summary of cross-correlation peaks and mean curves."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import TemporalCouplingConfig
from .cross_correlation import (
    CARDIAC_LABELS,
    CARDIAC_VARS,
    EEG_LABELS,
    EEG_VARS,
    _load_alignment_qc,
    group_curves_output_path,
    group_peaks_output_path,
    group_permutation_p_value,
    load_subject_pair_datasets,
    variable_pairs,
)
from .paths import group_output_dir, partition_key_from_row

PEAK_SUMMARY_FILENAME = "peak_correlation_summary.csv"
SUMMARY_FILENAME = "group_cross_correlation_summary.csv"
MEAN_CURVES_FILENAME = "mean_cross_correlation_curves.csv"
MEAN_SEM_GRID_PLOT = "group_cross_correlation_mean_sem_grid.png"
INTERPRETATION_NOTES_FILENAME = "stage3_interpretation_notes.txt"

STAGE3_OUTPUT_FILENAMES = (
    PEAK_SUMMARY_FILENAME,
    SUMMARY_FILENAME,
    MEAN_CURVES_FILENAME,
    INTERPRETATION_NOTES_FILENAME,
    MEAN_SEM_GRID_PLOT,
)

AUTO_NORMALITY_ALPHA = 0.05
FDR_ALPHA = 0.05
FDR_METHOD = "fdr_bh"
MIN_N_FORMAL = 5
HIGH_EDGE_PEAK_RATE_PERCENT = 50.0

WEAK_COUPLING_THRESHOLD = 0.1
MODERATE_COUPLING_THRESHOLD = 0.3

PEAK_SUMMARY_COLUMNS = (
    "pair",
    "cardiac_var",
    "eeg_var",
    "n_subjects",
    "median_raw_peak_lag_s",
    "mean_raw_peak_lag_s",
    "sem_raw_peak_lag_s",
    "median_raw_peak_signed_r",
    "mean_raw_peak_signed_r",
    "sem_raw_peak_signed_r",
    "median_raw_peak_abs_r",
    "mean_raw_peak_abs_r",
    "n_negative_lag_peaks",
    "n_positive_lag_peaks",
    "test_peak_signed_r_method",
    "test_peak_signed_r_p",
    "test_peak_lag_method",
    "test_peak_lag_p",
    "q_value_peak_signed_r",
    "q_value_peak_lag",
    "sig_peak_signed_r_fdr",
    "sig_peak_lag_fdr",
    "n_edge_peaks",
    "percent_edge_peaks",
    "median_p_perm",
    "n_with_p_perm",
    "group_perm_stat",
    "group_perm_p",
    "group_perm_q",
    "sig_group_perm_fdr",
    "warning",
)

SUMMARY_COLUMNS = (
    "pair",
    "cardiac_var",
    "eeg_var",
    "n_subjects",
    "median_raw_peak_lag_s",
    "mean_raw_peak_lag_s",
    "mean_raw_peak_signed_r",
    "median_raw_peak_abs_r",
    "mean_raw_peak_abs_r",
    "likely_direction",
    "coupling_strength",
    "median_lag_s",
    "q_value_peak_signed_r",
    "q_value_peak_lag",
    "sig_peak_signed_r_fdr",
    "sig_peak_lag_fdr",
    "n_edge_peaks",
    "percent_edge_peaks",
    "warning",
)

MEAN_CURVES_COLUMNS = (
    "pair",
    "lag_s",
    "mean_r",
    "sem_r",
    "n_subjects",
    "total_subjects",
    "in_common_lag_range",
)


def peak_summary_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / PEAK_SUMMARY_FILENAME


def summary_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / SUMMARY_FILENAME


def mean_curves_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / MEAN_CURVES_FILENAME


def interpretation_notes_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / INTERPRETATION_NOTES_FILENAME


def stage3_output_paths(cfg: TemporalCouplingConfig) -> list[Path]:
    group_dir = group_output_dir(cfg)
    return [group_dir / name for name in STAGE3_OUTPUT_FILENAMES]


def clear_stage3_outputs(cfg: TemporalCouplingConfig) -> list[Path]:
    removed: list[Path] = []
    for path in stage3_output_paths(cfg):
        if path.is_file():
            path.unlink()
            removed.append(path)
    group_base = group_output_dir(cfg)
    if group_base.is_dir():
        for subdir in group_base.iterdir():
            if not subdir.is_dir():
                continue
            for name in STAGE3_OUTPUT_FILENAMES:
                path = subdir / name
                if path.is_file():
                    path.unlink()
                    removed.append(path)
    return removed


def _row_partition_key(row: pd.Series, *, dataset_id: str, hiit_partition_mode: str) -> str:
    task = str(row["task"])
    condition = str(row["condition"]) if "condition" in row.index and pd.notna(row["condition"]) else task
    observation_id = str(row["observation_id"]) if "observation_id" in row.index and pd.notna(row["observation_id"]) else None
    return partition_key_from_row(
        dataset_id=dataset_id,
        task=task,
        condition=condition,
        observation_id=observation_id,
        hiit_partition_mode=hiit_partition_mode,
    )


def _partition_keys(peaks_df: pd.DataFrame, *, dataset_id: str, hiit_partition_mode: str) -> list[str]:
    if peaks_df.empty:
        return []
    keys = peaks_df.apply(
        lambda row: _row_partition_key(row, dataset_id=dataset_id, hiit_partition_mode=hiit_partition_mode),
        axis=1,
    )
    return sorted(keys.unique())


def _filter_by_partition(
    df: pd.DataFrame | None,
    *,
    partition: str,
    dataset_id: str,
    hiit_partition_mode: str,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    mask = df.apply(
        lambda row: _row_partition_key(row, dataset_id=dataset_id, hiit_partition_mode=hiit_partition_mode) == partition,
        axis=1,
    )
    return df.loc[mask].copy()


def _filter_curves_by_partition(
    curves_df: pd.DataFrame | None,
    peaks_df: pd.DataFrame,
    *,
    partition: str,
    dataset_id: str,
    hiit_partition_mode: str,
) -> pd.DataFrame | None:
    if curves_df is None or curves_df.empty:
        return curves_df
    if "task" in curves_df.columns or "condition" in curves_df.columns:
        return _filter_by_partition(
            curves_df,
            partition=partition,
            dataset_id=dataset_id,
            hiit_partition_mode=hiit_partition_mode,
        )
    subject_ids = set(peaks_df["subject_id"].astype(str).unique())
    return curves_df.loc[curves_df["subject_id"].astype(str).isin(subject_ids)].copy()


def _validate_partition_inputs(
    peaks_df: pd.DataFrame,
    curves_df: pd.DataFrame | None,
    *,
    partition: str,
) -> int:
    n_subjects = int(peaks_df["subject_id"].nunique()) if not peaks_df.empty else 0
    n_peak_rows = len(peaks_df)
    expected_rows = n_subjects * len(variable_pairs())
    if n_peak_rows != expected_rows:
        raise ValueError(
            f"Stage 3 input peaks rows={n_peak_rows}, expected {expected_rows} "
            f"({n_subjects} subjects × {len(variable_pairs())} pairs) "
            f"for partition={partition!r}."
        )
    if not peaks_df.empty:
        dup = peaks_df.groupby(["subject_id", "pair"]).size()
        if (dup > 1).any():
            bad = dup[dup > 1].index.tolist()[:5]
            raise ValueError(
                f"Stage 3 duplicate subject×pair rows in partition={partition!r}: {bad}"
            )
    if curves_df is not None and not curves_df.empty:
        curve_subjects = int(curves_df["subject_id"].nunique())
        if curve_subjects != n_subjects:
            raise ValueError(
                f"Stage 3 input mismatch for partition={partition!r}: "
                f"peaks subjects={n_subjects}, curves subjects={curve_subjects}."
            )
    return n_subjects


def _validate_stage3_outputs(
    *,
    group_dir: Path,
    expected_subjects: int,
) -> None:
    peak_summary = pd.read_csv(group_dir / PEAK_SUMMARY_FILENAME)
    group_summary = pd.read_csv(group_dir / SUMMARY_FILENAME)
    notes_path = group_dir / INTERPRETATION_NOTES_FILENAME
    notes_text = notes_path.read_text(encoding="utf-8")

    if len(peak_summary) != len(variable_pairs()):
        raise ValueError(
            f"peak_correlation_summary.csv rows={len(peak_summary)}, "
            f"expected {len(variable_pairs())}."
        )
    if not (peak_summary["n_subjects"] == expected_subjects).all():
        bad = peak_summary.loc[peak_summary["n_subjects"] != expected_subjects, "pair"].tolist()
        raise ValueError(f"peak_correlation_summary.csv has unexpected n_subjects for pairs: {bad}")
    if not (group_summary["n_subjects"] == expected_subjects).all():
        bad = group_summary.loc[group_summary["n_subjects"] != expected_subjects, "pair"].tolist()
        raise ValueError(
            f"group_cross_correlation_summary.csv has unexpected n_subjects for pairs: {bad}"
        )
    if f"n_subjects_in_peaks: {expected_subjects}" not in notes_text:
        raise ValueError(
            f"stage3_interpretation_notes.txt missing n_subjects_in_peaks: {expected_subjects}."
        )

    mean_curves_path = group_dir / MEAN_CURVES_FILENAME
    if mean_curves_path.is_file():
        mean_curves = pd.read_csv(mean_curves_path)
        if int(mean_curves["total_subjects"].max()) != expected_subjects:
            raise ValueError(
                "mean_cross_correlation_curves.csv total_subjects does not match input cohort."
            )


def _shapiro_pvalue(values: np.ndarray) -> float:
    if values.size < 3:
        return float("nan")
    try:
        return float(stats.shapiro(values).pvalue)
    except Exception:
        return float("nan")


def _test_vs_zero(
    values: np.ndarray,
    *,
    prefer_wilcoxon: bool,
) -> tuple[str, float]:
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return "insufficient_data", float("nan")
    if n == 1:
        return "exploratory_n1", float("nan")

    if prefer_wilcoxon:
        method = "wilcoxon"
    else:
        shapiro_p = _shapiro_pvalue(finite)
        method = "ttest_1samp" if np.isfinite(shapiro_p) and shapiro_p > AUTO_NORMALITY_ALPHA else "wilcoxon"

    try:
        if method == "ttest_1samp":
            result = stats.ttest_1samp(finite, popmean=0.0, nan_policy="omit")
            return method, float(result.pvalue)
        result = stats.wilcoxon(finite, alternative="two-sided")
        return method, float(result.pvalue)
    except Exception:
        return method, float("nan")


def _apply_bh_fdr(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    arr = np.asarray(p_values, dtype=float)
    valid = np.isfinite(arr)
    q_values = np.full(arr.shape, np.nan, dtype=float)
    if valid.sum() == 0:
        return q_values.tolist()
    _, qvals, _, _ = multipletests(arr[valid], alpha=FDR_ALPHA, method=FDR_METHOD)
    q_values[valid] = qvals
    return q_values.tolist()


def _coupling_strength(median_abs_r: float) -> str:
    if not np.isfinite(median_abs_r):
        return "unknown"
    if median_abs_r < WEAK_COUPLING_THRESHOLD:
        return "weak"
    if median_abs_r < MODERATE_COUPLING_THRESHOLD:
        return "moderate"
    return "strong"


def _likely_direction(median_lag_s: float, *, lag_step_s: float) -> str:
    if not np.isfinite(median_lag_s):
        return "unknown"
    threshold = lag_step_s / 2.0
    if abs(median_lag_s) <= threshold:
        return "near_zero_shared"
    if median_lag_s < 0:
        return "eeg_leads"
    return "cardiac_leads"


def _direction_label(direction: str) -> str:
    return {
        "eeg_leads": "EEG leads",
        "cardiac_leads": "cardiac leads",
        "near_zero_shared": "near-zero/shared timing",
        "unknown": "unknown",
    }.get(direction, direction)


def _pair_warnings(
    *,
    n_subjects: int,
    n_negative: int,
    n_positive: int,
    n_edge: int,
    lag_step_s: float,
    n_permutations: int,
) -> str:
    warnings_out: list[str] = []
    if n_subjects < 3:
        warnings_out.append("few_subjects")
    if n_subjects < MIN_N_FORMAL:
        warnings_out.append("exploratory_small_n")

    if n_subjects > 0 and (100.0 * n_edge / n_subjects) > HIGH_EDGE_PEAK_RATE_PERCENT:
        warnings_out.append("high_edge_peak_rate")

    threshold = lag_step_s / 2.0
    if n_negative > 0 and n_positive > 0:
        warnings_out.append("mixed_peak_lag_direction")

    if n_permutations == 0:
        warnings_out.append("no_permutation_test")

    return ";".join(dict.fromkeys(warnings_out))


def _append_group_perm_warning(warning: str, *, n_group_permutations: int) -> str:
    if n_group_permutations <= 0:
        extra = "no_group_permutation_test"
        if extra in warning.split(";"):
            return warning
        return f"{warning};{extra}" if warning else extra
    return warning


def apply_group_permutation_tests(
    peak_summary_rows: list[dict[str, object]],
    peaks_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
) -> list[dict[str, object]]:
    """Add group-level circular-shift permutation p-values for each pair."""
    n_group_permutations = cfg.temporal_coupling.group.n_group_permutations
    rng = np.random.default_rng(42)
    alignment_qc = _load_alignment_qc(cfg)

    for row in peak_summary_rows:
        row["group_perm_stat"] = float("nan")
        row["group_perm_p"] = float("nan")
        row["group_perm_q"] = float("nan")
        row["sig_group_perm_fdr"] = False
        row["warning"] = _append_group_perm_warning(
            str(row.get("warning", "")),
            n_group_permutations=n_group_permutations,
        )

    if n_group_permutations <= 0:
        return peak_summary_rows

    pair_p_values: list[float] = []
    pair_indices: list[int] = []

    for idx, row in enumerate(peak_summary_rows):
        pair_name = str(row["pair"])
        pair = next(vp for vp in variable_pairs() if vp.pair == pair_name)
        pair_rows = peaks_df.loc[peaks_df["pair"] == pair_name]
        subject_ids = sorted(pair_rows["subject_id"].astype(str).unique())
        observed_stat = float(row["median_raw_peak_abs_r"])
        row["group_perm_stat"] = observed_stat

        if not subject_ids or not np.isfinite(observed_stat):
            pair_p_values.append(float("nan"))
            pair_indices.append(idx)
            continue

        subject_data = load_subject_pair_datasets(
            cfg,
            pair=pair,
            subject_ids=subject_ids,
            alignment_qc=alignment_qc,
        )
        if len(subject_data) < len(subject_ids):
            warnings.warn(
                f"[temporal_coupling] stage=3 group perm {pair_name}: "
                f"loaded {len(subject_data)}/{len(subject_ids)} aligned subjects.",
                stacklevel=2,
            )

        p_value = group_permutation_p_value(
            subject_data,
            observed_group_stat=observed_stat,
            n_group_permutations=n_group_permutations,
            cfg=cfg,
            rng=rng,
        )
        row["group_perm_p"] = float("nan") if p_value is None else float(p_value)
        pair_p_values.append(row["group_perm_p"])
        pair_indices.append(idx)

    q_values = _apply_bh_fdr(pair_p_values)
    for idx, q_value in zip(pair_indices, q_values, strict=True):
        peak_summary_rows[idx]["group_perm_q"] = q_value
        peak_summary_rows[idx]["sig_group_perm_fdr"] = bool(
            np.isfinite(q_value) and q_value <= FDR_ALPHA
        )

    return peak_summary_rows


def _sem(values: pd.Series) -> float:
    finite = values.astype(float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def _load_peaks(cfg: TemporalCouplingConfig) -> pd.DataFrame:
    peaks_path = group_peaks_output_path(cfg)
    if not peaks_path.is_file():
        raise FileNotFoundError(
            f"Missing {peaks_path.name}. Run --stage 2 first to write group peak results."
        )
    peaks_df = pd.read_csv(peaks_path)
    if peaks_df.empty:
        warnings.warn(
            "[temporal_coupling] stage=3: peak table is empty; summary will contain NaNs.",
            stacklevel=2,
        )
    return peaks_df


def _load_curves(cfg: TemporalCouplingConfig) -> pd.DataFrame | None:
    curves_path = group_curves_output_path(cfg)
    if not curves_path.is_file():
        return None
    curves_df = pd.read_csv(curves_path)
    if curves_df.empty:
        return None
    return curves_df


def build_peak_correlation_summary(
    peaks_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
) -> list[dict[str, object]]:
    lag_step_s = cfg.temporal_coupling.cross_correlation.lag_step_s
    n_permutations = cfg.temporal_coupling.cross_correlation.n_permutations
    threshold = lag_step_s / 2.0
    summary_rows: list[dict[str, object]] = []

    for pair in variable_pairs():
        pair_rows = peaks_df.loc[peaks_df["pair"] == pair.pair].copy()
        n_subjects = len(pair_rows)

        if n_subjects:
            lags = pair_rows["raw_peak_lag_s"].astype(float)
            signed_r = pair_rows["raw_peak_signed_r"].astype(float)
            abs_r = pair_rows["raw_peak_abs_r"].astype(float)
            p_perm = pair_rows["p_perm"].astype(float) if "p_perm" in pair_rows.columns else pd.Series(dtype=float)
            median_raw_peak_lag_s = float(lags.median())
            mean_raw_peak_lag_s = float(lags.mean())
            sem_raw_peak_lag_s = _sem(lags)
            median_raw_peak_signed_r = float(signed_r.median())
            mean_raw_peak_signed_r = float(signed_r.mean())
            sem_raw_peak_signed_r = _sem(signed_r)
            median_raw_peak_abs_r = float(abs_r.median())
            mean_raw_peak_abs_r = float(abs_r.mean())
            n_negative = int((lags < -threshold).sum())
            n_positive = int((lags > threshold).sum())
            n_edge = int(pair_rows["raw_peak_at_edge"].astype(bool).sum())
            percent_edge = float(100.0 * n_edge / n_subjects)
            finite_p_perm = p_perm[np.isfinite(p_perm)]
            median_p_perm = float(finite_p_perm.median()) if finite_p_perm.size else float("nan")
            n_with_p_perm = int(finite_p_perm.size)
            signed_r_values = signed_r.to_numpy(dtype=float)
            lag_values = lags.to_numpy(dtype=float)
        else:
            median_raw_peak_lag_s = float("nan")
            mean_raw_peak_lag_s = float("nan")
            sem_raw_peak_lag_s = float("nan")
            median_raw_peak_signed_r = float("nan")
            mean_raw_peak_signed_r = float("nan")
            sem_raw_peak_signed_r = float("nan")
            median_raw_peak_abs_r = float("nan")
            mean_raw_peak_abs_r = float("nan")
            n_negative = 0
            n_positive = 0
            n_edge = 0
            percent_edge = float("nan")
            median_p_perm = float("nan")
            n_with_p_perm = 0
            signed_r_values = np.array([], dtype=float)
            lag_values = np.array([], dtype=float)

        signed_r_method, signed_r_p = _test_vs_zero(signed_r_values, prefer_wilcoxon=False)
        lag_method, lag_p = _test_vs_zero(lag_values, prefer_wilcoxon=True)

        summary_rows.append(
            {
                "pair": pair.pair,
                "cardiac_var": pair.cardiac_var,
                "eeg_var": pair.eeg_var,
                "n_subjects": n_subjects,
                "median_raw_peak_lag_s": median_raw_peak_lag_s,
                "mean_raw_peak_lag_s": mean_raw_peak_lag_s,
                "sem_raw_peak_lag_s": sem_raw_peak_lag_s,
                "median_raw_peak_signed_r": median_raw_peak_signed_r,
                "mean_raw_peak_signed_r": mean_raw_peak_signed_r,
                "sem_raw_peak_signed_r": sem_raw_peak_signed_r,
                "median_raw_peak_abs_r": median_raw_peak_abs_r,
                "mean_raw_peak_abs_r": mean_raw_peak_abs_r,
                "n_negative_lag_peaks": n_negative,
                "n_positive_lag_peaks": n_positive,
                "test_peak_signed_r_method": signed_r_method,
                "test_peak_signed_r_p": signed_r_p,
                "test_peak_lag_method": lag_method,
                "test_peak_lag_p": lag_p,
                "q_value_peak_signed_r": float("nan"),
                "q_value_peak_lag": float("nan"),
                "sig_peak_signed_r_fdr": False,
                "sig_peak_lag_fdr": False,
                "n_edge_peaks": n_edge,
                "percent_edge_peaks": percent_edge,
                "median_p_perm": median_p_perm,
                "n_with_p_perm": n_with_p_perm,
                "group_perm_stat": float("nan"),
                "group_perm_p": float("nan"),
                "group_perm_q": float("nan"),
                "sig_group_perm_fdr": False,
                "warning": _pair_warnings(
                    n_subjects=n_subjects,
                    n_negative=n_negative,
                    n_positive=n_positive,
                    n_edge=n_edge,
                    lag_step_s=lag_step_s,
                    n_permutations=n_permutations,
                ),
            }
        )

    signed_r_q = _apply_bh_fdr([float(row["test_peak_signed_r_p"]) for row in summary_rows])
    lag_q = _apply_bh_fdr([float(row["test_peak_lag_p"]) for row in summary_rows])
    for idx, row in enumerate(summary_rows):
        row["q_value_peak_signed_r"] = signed_r_q[idx]
        row["q_value_peak_lag"] = lag_q[idx]
        row["sig_peak_signed_r_fdr"] = bool(
            np.isfinite(signed_r_q[idx]) and signed_r_q[idx] <= FDR_ALPHA
        )
        row["sig_peak_lag_fdr"] = bool(np.isfinite(lag_q[idx]) and lag_q[idx] <= FDR_ALPHA)

    return summary_rows


def build_group_interpretation_summary(
    peak_summary_rows: list[dict[str, object]],
    *,
    lag_step_s: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for peak in peak_summary_rows:
        median_lag = float(peak["median_raw_peak_lag_s"])
        median_abs_r = float(peak["median_raw_peak_abs_r"])
        rows.append(
            {
                "pair": peak["pair"],
                "cardiac_var": peak["cardiac_var"],
                "eeg_var": peak["eeg_var"],
                "n_subjects": peak["n_subjects"],
                "median_raw_peak_lag_s": median_lag,
                "mean_raw_peak_lag_s": peak["mean_raw_peak_lag_s"],
                "mean_raw_peak_signed_r": peak["mean_raw_peak_signed_r"],
                "median_raw_peak_abs_r": median_abs_r,
                "mean_raw_peak_abs_r": peak["mean_raw_peak_abs_r"],
                "likely_direction": _likely_direction(median_lag, lag_step_s=lag_step_s),
                "coupling_strength": _coupling_strength(median_abs_r),
                "median_lag_s": median_lag,
                "q_value_peak_signed_r": peak["q_value_peak_signed_r"],
                "q_value_peak_lag": peak["q_value_peak_lag"],
                "sig_peak_signed_r_fdr": peak["sig_peak_signed_r_fdr"],
                "sig_peak_lag_fdr": peak["sig_peak_lag_fdr"],
                "n_edge_peaks": peak["n_edge_peaks"],
                "percent_edge_peaks": peak["percent_edge_peaks"],
                "warning": peak["warning"],
            }
        )
    return rows


def _total_subjects(curves_df: pd.DataFrame) -> int:
    return int(curves_df["subject_id"].nunique())


def _common_lag_bounds(curves_df: pd.DataFrame, *, pair: str) -> tuple[float, float] | None:
    pair_curves = curves_df.loc[curves_df["pair"] == pair]
    if pair_curves.empty:
        return None
    total = int(pair_curves["subject_id"].nunique())
    lag_counts = pair_curves.groupby("lag_s")["subject_id"].nunique()
    common_lags = lag_counts[lag_counts == total].index.astype(float)
    if common_lags.empty:
        return None
    return float(common_lags.min()), float(common_lags.max())


def _global_common_lag_bounds(curves_df: pd.DataFrame) -> tuple[float, float] | None:
    bounds: list[tuple[float, float]] = []
    for pair in variable_pairs():
        pair_bounds = _common_lag_bounds(curves_df, pair=pair.pair)
        if pair_bounds is not None:
            bounds.append(pair_bounds)
    if not bounds:
        return None
    return max(lo for lo, _ in bounds), min(hi for _, hi in bounds)


def build_mean_curves(curves_df: pd.DataFrame) -> pd.DataFrame:
    total_subjects = _total_subjects(curves_df)
    rows: list[dict[str, object]] = []
    for pair in variable_pairs():
        pair_curves = curves_df.loc[curves_df["pair"] == pair.pair]
        if pair_curves.empty:
            continue
        pair_bounds = _common_lag_bounds(curves_df, pair=pair.pair)
        grouped = pair_curves.groupby("lag_s", as_index=False)
        for lag_s, lag_rows in grouped:
            values = lag_rows["r"].astype(float)
            finite = values[np.isfinite(values)]
            n_subjects = int(finite.size)
            mean_r = float(finite.mean()) if n_subjects else float("nan")
            sem_r = float(finite.std(ddof=1) / np.sqrt(n_subjects)) if n_subjects > 1 else float("nan")
            in_common = False
            if pair_bounds is not None:
                in_common = pair_bounds[0] <= float(lag_s) <= pair_bounds[1] and n_subjects == total_subjects
            rows.append(
                {
                    "pair": pair.pair,
                    "lag_s": float(lag_s),
                    "mean_r": mean_r,
                    "sem_r": sem_r,
                    "n_subjects": n_subjects,
                    "total_subjects": total_subjects,
                    "in_common_lag_range": in_common,
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(MEAN_CURVES_COLUMNS))
    return pd.DataFrame(rows).sort_values(["pair", "lag_s"]).reset_index(drop=True)


def _curve_plot_frame(
    mean_curves_df: pd.DataFrame,
    *,
    pair: str,
    plot_common_lag_only: bool,
) -> pd.DataFrame:
    pair_df = mean_curves_df.loc[mean_curves_df["pair"] == pair].sort_values("lag_s")
    if pair_df.empty:
        return pair_df
    if plot_common_lag_only:
        return pair_df.loc[pair_df["in_common_lag_range"]].copy()
    return pair_df.copy()


def _shade_partial_n_regions(
    ax: plt.Axes,
    pair_df: pd.DataFrame,
    *,
    lag_step_s: float,
) -> None:
    if pair_df.empty:
        return
    total = int(pair_df["total_subjects"].iloc[0])
    partial = pair_df.loc[pair_df["n_subjects"] < total].sort_values("lag_s")
    if partial.empty:
        return
    half_step = lag_step_s / 2.0
    for lag_s in partial["lag_s"].astype(float):
        ax.axvspan(lag_s - half_step, lag_s + half_step, color="0.85", alpha=0.55, zorder=0)


def _plot_mean_sem_grid(
    mean_curves_df: pd.DataFrame,
    output_path: Path,
    *,
    plot_common_lag_only: bool,
    lag_step_s: float,
    global_common_bounds: tuple[float, float] | None,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=False, sharey=True)
    pairs = variable_pairs()
    mode_note = (
        f"common lag range [{global_common_bounds[0]:g}, {global_common_bounds[1]:g}] s"
        if plot_common_lag_only and global_common_bounds is not None
        else "partial-n lags shaded gray"
    )

    for idx, pair in enumerate(pairs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        full_pair_df = mean_curves_df.loc[mean_curves_df["pair"] == pair.pair].sort_values("lag_s")
        plot_df = _curve_plot_frame(
            mean_curves_df,
            pair=pair.pair,
            plot_common_lag_only=plot_common_lag_only,
        )

        if plot_df.empty:
            ax.set_title(f"{pair.pair}\n(no plottable curve data)")
            ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
            continue

        if not plot_common_lag_only:
            _shade_partial_n_regions(ax, full_pair_df, lag_step_s=lag_step_s)

        x = plot_df["lag_s"].to_numpy(dtype=float)
        y = plot_df["mean_r"].to_numpy(dtype=float)
        sem = plot_df["sem_r"].to_numpy(dtype=float)
        n_subjects = plot_df["n_subjects"].astype(int)
        ax.plot(x, y, color="#1f77b4", linewidth=1.8, zorder=3)
        sem_finite = np.isfinite(sem)
        if sem_finite.any():
            ax.fill_between(
                x,
                y - sem,
                y + sem,
                where=sem_finite,
                color="#1f77b4",
                alpha=0.25,
                linewidth=0,
                zorder=2,
            )

        min_n = int(n_subjects.min())
        max_n = int(n_subjects.max())
        total_n = int(plot_df["total_subjects"].iloc[0])
        n_note = f"n={total_n}" if min_n == max_n == total_n else f"n={min_n}-{max_n}/{total_n}"
        ax.set_title(f"{pair.pair}\n{n_note}", fontsize=9)
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--", zorder=1)

        if plot_common_lag_only and global_common_bounds is not None:
            ax.set_xlim(global_common_bounds[0], global_common_bounds[1])

        if col == 0:
            ax.set_ylabel(f"{CARDIAC_LABELS[row]}\nr")
        if row == 2:
            ax.set_xlabel("Lag (s)")

    handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=1.8, label="mean r"),
        Line2D([0], [0], color="#1f77b4", alpha=0.25, linewidth=6, label="± SEM"),
        Line2D([0], [0], color="0.7", linestyle="--", linewidth=0.8, label="lag 0"),
    ]
    if not plot_common_lag_only:
        handles.append(Patch(facecolor="0.85", edgecolor="none", alpha=0.55, label="n < all subjects"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle(f"Group mean cross-correlation curves (± SEM)\n{mode_note}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _format_float(value: object, *, precision: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{precision}f}"


def _write_interpretation_notes(
    cfg: TemporalCouplingConfig,
    *,
    peak_summary_rows: list[dict[str, object]],
    interpretation_rows: list[dict[str, object]],
    global_common_bounds: tuple[float, float] | None,
    total_subjects: int | None,
    output_path: Path,
    partition: str | None = None,
) -> None:
    lines: list[str] = [
        "Stage 3 temporal coupling interpretation notes",
        "============================================",
        f"dataset_id: {cfg.dataset_id}",
    ]
    if partition:
        lines.append(f"partition: {partition}")
    lines.append(f"n_subjects_in_peaks: {total_subjects if total_subjects is not None else 'unknown'}")
    lines.extend(["", "Plot settings", f"  plot_common_lag_only: {cfg.temporal_coupling.group.plot_common_lag_only}"])
    if global_common_bounds is not None:
        lines.append(
            f"  common lag range used for mean curves: "
            f"[{global_common_bounds[0]:g}, {global_common_bounds[1]:g}] s"
        )
    else:
        lines.append("  common lag range: unavailable (subjects may have non-overlapping lag grids)")

    lines.extend(
        [
            "",
            "Statistical inference layers",
            "  1. Selected peak signed-r vs zero (group one-sample test; BH-FDR across pairs).",
            "     Does not control for selecting the maximum peak across lags.",
            "  2. Permutation-controlled group peak strength (circular-shift EEG null;",
            "     median raw peak |r| across subjects; BH-FDR across pairs).",
            "     Controls lag-search selection bias at the group level.",
            "  3. Peak lag direction vs zero (group Wilcoxon; BH-FDR across pairs).",
            "",
            "Caveats",
            "  - Negative lag means EEG leads cardiac; positive lag means cardiac leads EEG.",
            "  - Heatmaps use median RAW peak values (not preferred/interior peaks).",
            "  - Asterisks in heatmaps mark pairs with >50% raw peaks at the lag edge.",
            "  - Selected peak-r vs zero and peak-lag tests are exploratory when n is small.",
            "  - mean_cross_correlation_curves.csv may show n_subjects < cohort size at lags",
            "    outside the common lag range; use total_subjects and in_common_lag_range columns.",
            "",
            "Per-pair summary",
        ]
    )

    for peak, interp in zip(peak_summary_rows, interpretation_rows, strict=True):
        direction = _direction_label(str(interp["likely_direction"]))
        exploratory = " [EXPLORATORY]" if "exploratory_small_n" in str(peak["warning"]) else ""
        sig_r = "significant" if peak["sig_peak_signed_r_fdr"] else "not significant"
        sig_lag = "significant" if peak["sig_peak_lag_fdr"] else "not significant"
        sig_group = "significant" if peak["sig_group_perm_fdr"] else "not significant"
        lines.extend(
            [
                "",
                f"{peak['pair']}{exploratory}",
                f"  n_subjects={peak['n_subjects']}",
                f"  median raw peak lag={_format_float(peak['median_raw_peak_lag_s'])} s",
                f"  median raw peak signed r={_format_float(peak['median_raw_peak_signed_r'], precision=3)}",
                f"  coupling_strength={interp['coupling_strength']}",
                f"  likely_direction={direction}",
                f"  edge_peaks={peak['n_edge_peaks']} ({_format_float(peak['percent_edge_peaks'], precision=0)}%)",
                "  [1] selected peak-r vs zero:",
                f"      FDR q={_format_float(peak['q_value_peak_signed_r'], precision=3)} ({sig_r})",
                "  [2] permutation-controlled group peak strength:",
                f"      group_perm_stat(median |r|)={_format_float(peak['group_perm_stat'], precision=3)}",
                f"      FDR q={_format_float(peak['group_perm_q'], precision=3)} ({sig_group})",
                "  [3] peak-lag direction vs zero:",
                f"      FDR q={_format_float(peak['q_value_peak_lag'], precision=3)} ({sig_lag})",
                f"  warnings={peak['warning'] or 'none'}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary_report(interpretation_rows: list[dict[str, object]], peak_summary_rows: list[dict[str, object]]) -> None:
    peak_by_pair = {str(row["pair"]): row for row in peak_summary_rows}
    print("[temporal_coupling] stage=3 group summary:")
    for row in interpretation_rows:
        direction = _direction_label(str(row["likely_direction"]))
        exploratory = " [EXPLORATORY]" if "exploratory_small_n" in str(row["warning"]) else ""
        sig_r = "sig" if row["sig_peak_signed_r_fdr"] else "n.s."
        sig_lag = "sig" if row["sig_peak_lag_fdr"] else "n.s."
        peak_row = peak_by_pair.get(str(row["pair"]), {})
        sig_group = "sig" if peak_row.get("sig_group_perm_fdr") else "n.s."
        print(
            f"[temporal_coupling]   {row['pair']}: n={row['n_subjects']} "
            f"median_lag={_format_float(row['median_lag_s'])}s "
            f"coupling={row['coupling_strength']} "
            f"direction={direction}{exploratory} "
            f"(peak_r {sig_r} q={_format_float(row['q_value_peak_signed_r'], precision=3)}, "
            f"group_perm {sig_group} q={_format_float(peak_row.get('group_perm_q'), precision=3)}, "
            f"peak_lag {sig_lag} q={_format_float(row['q_value_peak_lag'], precision=3)})"
        )
        if row["warning"]:
            print(f"[temporal_coupling]     warning: {row['warning']}")


def _run_stage3_partition(
    cfg: TemporalCouplingConfig,
    peaks_df: pd.DataFrame,
    curves_df: pd.DataFrame | None,
    *,
    partition: str,
    group_dir: Path,
) -> list[Path]:
    lag_step_s = cfg.temporal_coupling.cross_correlation.lag_step_s
    plot_common_lag_only = cfg.temporal_coupling.group.plot_common_lag_only
    group_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = _validate_partition_inputs(peaks_df, curves_df, partition=partition)
    print(
        f"[temporal_coupling] stage=3 partition={partition!r}: subjects={n_subjects} "
        f"peak_rows={len(peaks_df)} curve_rows={0 if curves_df is None else len(curves_df)} "
        f"-> {group_dir}"
    )

    peak_summary_rows = build_peak_correlation_summary(peaks_df, cfg)
    n_group_perms = cfg.temporal_coupling.group.n_group_permutations
    if n_group_perms > 0:
        print(
            f"[temporal_coupling] stage=3 partition={partition!r}: running group permutation tests "
            f"(n_group_permutations={n_group_perms})"
        )
    peak_summary_rows = apply_group_permutation_tests(peak_summary_rows, peaks_df, cfg)
    interpretation_rows = build_group_interpretation_summary(
        peak_summary_rows,
        lag_step_s=lag_step_s,
    )

    peak_summary_path = group_dir / PEAK_SUMMARY_FILENAME
    pd.DataFrame(peak_summary_rows, columns=list(PEAK_SUMMARY_COLUMNS)).to_csv(
        peak_summary_path,
        index=False,
    )
    written: list[Path] = [peak_summary_path]
    print(f"[temporal_coupling] stage=3 wrote peak summary -> {peak_summary_path}")

    summary_path = group_dir / SUMMARY_FILENAME
    pd.DataFrame(interpretation_rows, columns=list(SUMMARY_COLUMNS)).to_csv(summary_path, index=False)
    written.append(summary_path)
    print(f"[temporal_coupling] stage=3 wrote interpretation summary -> {summary_path}")

    total_subjects = n_subjects
    global_common_bounds: tuple[float, float] | None = None
    if curves_df is not None and not curves_df.empty:
        mean_curves_df = build_mean_curves(curves_df)
        global_common_bounds = _global_common_lag_bounds(curves_df)
        mean_curves_path = group_dir / MEAN_CURVES_FILENAME
        mean_curves_df.to_csv(mean_curves_path, index=False)
        written.append(mean_curves_path)
        print(
            f"[temporal_coupling] stage=3 wrote mean curves -> {mean_curves_path} "
            f"rows={len(mean_curves_df)}"
        )
        if global_common_bounds is not None:
            print(
                f"[temporal_coupling] stage=3 partition={partition!r} common lag range: "
                f"[{global_common_bounds[0]:g}, {global_common_bounds[1]:g}] s"
            )

        if cfg.temporal_coupling.output.save_plots and not mean_curves_df.empty:
            plot_path = group_dir / MEAN_SEM_GRID_PLOT
            _plot_mean_sem_grid(
                mean_curves_df,
                plot_path,
                plot_common_lag_only=plot_common_lag_only,
                lag_step_s=lag_step_s,
                global_common_bounds=global_common_bounds,
            )
            written.append(plot_path)
            print(f"[temporal_coupling] stage=3 wrote mean SEM grid -> {plot_path}")
    else:
        print(
            f"[temporal_coupling] stage=3 partition={partition!r}: no curves; "
            "skipping mean curve outputs."
        )

    notes_path = group_dir / INTERPRETATION_NOTES_FILENAME
    _write_interpretation_notes(
        cfg,
        peak_summary_rows=peak_summary_rows,
        interpretation_rows=interpretation_rows,
        global_common_bounds=global_common_bounds,
        total_subjects=total_subjects,
        output_path=notes_path,
        partition=partition,
    )
    written.append(notes_path)
    print(f"[temporal_coupling] stage=3 wrote interpretation notes -> {notes_path}")

    if cfg.temporal_coupling.cross_correlation.n_permutations == 0:
        warnings.warn(
            f"[temporal_coupling] stage=3 partition={partition!r}: permutation null not run "
            "(n_permutations=0). Group p/q values are exploratory only.",
            stacklevel=2,
        )

    _validate_stage3_outputs(group_dir=group_dir, expected_subjects=n_subjects)
    print(
        f"[temporal_coupling] stage=3 partition={partition!r} validation OK: "
        f"{len(variable_pairs())} pair summaries, n_subjects={n_subjects}"
    )

    _print_summary_report(interpretation_rows, peak_summary_rows)
    return written


def run_stage3(cfg: TemporalCouplingConfig) -> list[Path]:
    group_base = group_output_dir(cfg)
    group_base.mkdir(parents=True, exist_ok=True)

    peaks_path = group_peaks_output_path(cfg)
    curves_path = group_curves_output_path(cfg)
    print(f"[temporal_coupling] stage=3 reading peaks -> {peaks_path}")
    print(f"[temporal_coupling] stage=3 reading curves -> {curves_path}")

    removed = clear_stage3_outputs(cfg)
    if removed:
        print(
            f"[temporal_coupling] stage=3 removed {len(removed)} prior Stage 3 outputs "
            f"from {group_base}"
        )

    peaks_df = _load_peaks(cfg)
    curves_df = _load_curves(cfg)
    partitions = _partition_keys(
        peaks_df,
        dataset_id=cfg.dataset_id,
        hiit_partition_mode=cfg.hiit_partition_mode,
    )
    if not partitions:
        print("[temporal_coupling] stage=3: no peak rows found.")
        return []

    n_partitions = len(partitions)
    if n_partitions > 1:
        print(
            f"[temporal_coupling] stage=3: {n_partitions} partitions detected "
            f"({', '.join(partitions)}); summarizing each separately."
        )

    written: list[Path] = []
    for partition in partitions:
        part_peaks = _filter_by_partition(
            peaks_df,
            partition=partition,
            dataset_id=cfg.dataset_id,
            hiit_partition_mode=cfg.hiit_partition_mode,
        )
        part_curves = _filter_curves_by_partition(
            curves_df,
            part_peaks,
            partition=partition,
            dataset_id=cfg.dataset_id,
            hiit_partition_mode=cfg.hiit_partition_mode,
        )
        out_dir = group_base if n_partitions == 1 else group_output_dir(cfg, partition)
        written.extend(
            _run_stage3_partition(
                cfg,
                part_peaks,
                part_curves,
                partition=partition,
                group_dir=out_dir,
            )
        )
    return written
