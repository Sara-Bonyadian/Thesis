"""Stage 2: lagged cross-correlation between aligned cardiac and EEG envelope series."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .cardiac_detectors import _as_auto_float, _find_peaks
from .config import TemporalCouplingConfig, TemporalCouplingCrossCorrelationConfig
from .paths import group_output_dir, hiit_condition_from_observation_id, observation_output_dir
from .resample import (
    QC_GROUP_FILENAME as ALIGNMENT_QC_FILENAME,
    aligned_output_path,
)

CURVES_FILENAME = "cross_correlation_curves.csv"
PEAKS_FILENAME = "cross_correlation_peaks.csv"
GRID_PLOT = "cross_correlation_grid_subject.png"
OVERLAY_PLOT_TEMPLATE = "cross_correlation_overlay_{pair}.png"
PEAK_LAG_DIST_RAW_PLOT = "peak_lag_distribution_raw.png"
PEAK_LAG_DIST_PREFERRED_PLOT = "peak_lag_distribution_preferred.png"
PEAK_STRENGTH_RAW_PLOT = "peak_correlation_strength_distribution_raw.png"
PEAK_STRENGTH_PREFERRED_PLOT = "peak_correlation_strength_distribution_preferred.png"
EDGE_PEAK_SUMMARY_PLOT = "edge_peak_summary.png"
QC_SUMMARY_FILENAME = "cross_correlation_qc_summary.csv"
VALIDATION_FILENAME = "cross_correlation_peak_validation.csv"
PEAK_VALIDATION_TOL = 1e-9

SMOKE_RUN_NOTE = (
    "This is a smoke run. Do not interpret biological coupling until the full cohort "
    "and permutation tests are run."
)

QC_SUMMARY_COLUMNS = (
    "pair",
    "n_subjects",
    "n_edge_peaks",
    "percent_edge_peaks",
    "median_peak_lag_s",
    "mean_peak_lag_s",
    "median_peak_abs_r",
    "mean_peak_signed_r",
    "median_preferred_peak_lag_s",
    "mean_preferred_peak_lag_s",
    "median_preferred_peak_abs_r",
    "median_interior_peak_lag_s",
    "median_interior_peak_abs_r",
    "warning",
)

CARDIAC_VARS = ("hr_z", "rmssd_z", "sdnn_z")
EEG_VARS = ("theta_env_z", "alpha_env_z", "beta_env_z")

CARDIAC_SHORT = {
    "hr_z": "hr",
    "rmssd_z": "rmssd",
    "sdnn_z": "sdnn",
}
EEG_SHORT = {
    "theta_env_z": "theta",
    "alpha_env_z": "alpha",
    "beta_env_z": "beta",
}

CARDIAC_LABELS = ("HR", "RMSSD", "SDNN")
EEG_LABELS = ("Theta", "Alpha", "Beta")

MIN_OVERLAP_POINTS = 10

PEAK_COLUMNS = (
    "dataset_id",
    "subject_id",
    "task",
    "condition",
    "observation_id",
    "pair",
    "cardiac_var",
    "eeg_var",
    "raw_peak_lag_s",
    "raw_peak_signed_r",
    "raw_peak_abs_r",
    "raw_peak_at_edge",
    "interior_peak_lag_s",
    "interior_peak_signed_r",
    "interior_peak_abs_r",
    "preferred_peak_lag_s",
    "preferred_peak_signed_r",
    "preferred_peak_abs_r",
    "preferred_peak_source",
    "peak_lag_s",
    "peak_signed_r",
    "peak_abs_r",
    "peak_direction",
    "peak_at_lag_edge",
    "min_lag_s",
    "max_lag_s",
    "n_valid_lags",
    "n_overlap_at_raw_peak",
    "n_overlap_at_preferred_peak",
    "warning",
    "p_perm",
)

VALIDATION_COLUMNS = (
    "subject_id",
    "pair",
    "expected_raw_peak_lag_s",
    "stored_raw_peak_lag_s",
    "expected_raw_peak_abs_r",
    "stored_raw_peak_abs_r",
    "passed",
    "warning",
)


@dataclass(frozen=True)
class VariablePair:
    cardiac_var: str
    eeg_var: str

    @property
    def pair(self) -> str:
        return f"{CARDIAC_SHORT[self.cardiac_var]}__{EEG_SHORT[self.eeg_var]}"


@dataclass(frozen=True)
class LagCorrelationPoint:
    lag_s: float
    r: float
    n_overlap: int


@dataclass(frozen=True)
class PeakCorrelationResult:
    raw_peak_lag_s: float
    raw_peak_signed_r: float
    raw_peak_abs_r: float
    raw_peak_at_edge: bool
    interior_peak_lag_s: float
    interior_peak_signed_r: float
    interior_peak_abs_r: float
    preferred_peak_lag_s: float
    preferred_peak_signed_r: float
    preferred_peak_abs_r: float
    preferred_peak_source: str
    peak_direction: str
    min_lag_s: float
    max_lag_s: float
    n_valid_lags: int
    n_overlap_at_raw_peak: int
    n_overlap_at_preferred_peak: int
    warning: str
    p_perm: float | None


def curves_output_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / CURVES_FILENAME


def peaks_output_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / PEAKS_FILENAME


def group_curves_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / CURVES_FILENAME


def group_peaks_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / PEAKS_FILENAME


def group_qc_summary_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / QC_SUMMARY_FILENAME


def group_validation_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / VALIDATION_FILENAME


def alignment_qc_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / ALIGNMENT_QC_FILENAME


def variable_pairs() -> tuple[VariablePair, ...]:
    return tuple(
        VariablePair(cardiac_var=cardiac, eeg_var=eeg)
        for cardiac in CARDIAC_VARS
        for eeg in EEG_VARS
    )


def build_lag_grid(*, lag_max_s: float, lag_step_s: float) -> np.ndarray:
    if lag_max_s < 0:
        raise ValueError("lag_max_s must be >= 0")
    if lag_step_s <= 0:
        raise ValueError("lag_step_s must be > 0")
    n_steps = int(round(lag_max_s / lag_step_s))
    return np.arange(-n_steps, n_steps + 1, dtype=float) * lag_step_s


def lag_grid_bounds(lag_grid_s: np.ndarray) -> tuple[float, float]:
    return float(lag_grid_s[0]), float(lag_grid_s[-1])


def _lag_to_samples(lag_s: float, fs_hz: float) -> int:
    return int(round(lag_s * fs_hz))


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _prepare_series(
    values: np.ndarray,
    xcorr_cfg: TemporalCouplingCrossCorrelationConfig,
    *,
    fs_hz: float,
) -> np.ndarray:
    if not xcorr_cfg.detrend:
        return values

    out = values.astype(float).copy()
    finite = np.isfinite(out)
    if finite.sum() < 2:
        return out

    if xcorr_cfg.rolling_detrend_window_s is not None and xcorr_cfg.rolling_detrend_window_s > 0:
        window = max(3, int(round(xcorr_cfg.rolling_detrend_window_s * fs_hz)))
        series = pd.Series(out)
        baseline = series.rolling(window=window, center=True, min_periods=1).mean()
        out = (series - baseline).to_numpy(dtype=float)
        out[~finite] = np.nan
        return out

    filled = out.copy()
    idx = np.arange(filled.size)
    filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    trend = np.polyval(np.polyfit(idx[finite], filled[finite], deg=1), idx)
    out[finite] = out[finite] - trend[finite]
    return out


def correlate_at_lag(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_s: float,
    fs_hz: float,
    min_overlap: int = MIN_OVERLAP_POINTS,
) -> LagCorrelationPoint:
    """Correlate cardiac(t) with eeg(t + lag_s).

    Negative lag: EEG leads cardiac. Positive lag: cardiac leads EEG.
    """
    lag_samples = _lag_to_samples(lag_s, fs_hz)
    n = min(cardiac.size, eeg.size)
    cardiac = cardiac[:n]
    eeg = eeg[:n]

    if lag_samples > 0:
        x_seg = cardiac[:-lag_samples]
        y_seg = eeg[lag_samples:]
    elif lag_samples < 0:
        shift = -lag_samples
        x_seg = cardiac[shift:]
        y_seg = eeg[:-shift]
    else:
        x_seg = cardiac
        y_seg = eeg

    mask = np.isfinite(x_seg) & np.isfinite(y_seg)
    n_overlap = int(mask.sum())
    if n_overlap < min_overlap:
        return LagCorrelationPoint(lag_s=lag_s, r=float("nan"), n_overlap=n_overlap)

    r = _pearson_r(x_seg[mask], y_seg[mask])
    return LagCorrelationPoint(lag_s=lag_s, r=r, n_overlap=n_overlap)


def compute_correlation_curve(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_grid_s: np.ndarray,
    fs_hz: float,
    min_overlap: int = MIN_OVERLAP_POINTS,
) -> list[LagCorrelationPoint]:
    return [
        correlate_at_lag(
            cardiac,
            eeg,
            lag_s=float(lag_s),
            fs_hz=fs_hz,
            min_overlap=min_overlap,
        )
        for lag_s in lag_grid_s
    ]


def _peak_direction(peak_lag_s: float, *, lag_step_s: float) -> str:
    if not np.isfinite(peak_lag_s):
        return "unknown"
    if abs(peak_lag_s) <= lag_step_s / 2.0:
        return "simultaneous"
    if peak_lag_s < 0:
        return "eeg_leads"
    return "cardiac_leads"


def _is_edge_lag(lag_s: float, *, min_lag_s: float, max_lag_s: float, edge_margin_s: float) -> bool:
    if edge_margin_s <= 0:
        return np.isclose(lag_s, min_lag_s) or np.isclose(lag_s, max_lag_s)
    return lag_s <= min_lag_s + edge_margin_s or lag_s >= max_lag_s - edge_margin_s


def _same_correlation_sign(a: float, b: float, *, tol: float = 1e-9) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    if abs(a) <= tol or abs(b) <= tol:
        return True
    return (a > 0) == (b > 0)


def _select_positive_max_peak(points: list[LagCorrelationPoint]) -> LagCorrelationPoint | None:
    finite = [point for point in points if np.isfinite(point.r) and point.r > 0]
    if not finite:
        return None
    max_r = max(point.r for point in finite)
    candidates = [point for point in finite if np.isclose(point.r, max_r, rtol=0.0, atol=PEAK_VALIDATION_TOL)]
    return min(candidates, key=lambda point: abs(point.lag_s))


def _auto_positive_threshold(
    values: np.ndarray,
    *,
    percentile: float,
    fallback: np.ndarray | None = None,
) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    source = positive if positive.size >= 3 else (fallback if fallback is not None else values)
    finite = source[np.isfinite(source)]
    if finite.size == 0:
        return 0.0
    return max(0.0, float(np.percentile(finite, percentile)))


def _resolved_min_peak_distance_s(
    min_peak_distance_s: float | None,
    *,
    lag_step_s: float,
) -> float:
    if min_peak_distance_s is not None and min_peak_distance_s > 0:
        return float(min_peak_distance_s)
    return float(2.0 * lag_step_s)


def _find_local_positive_peaks(
    points: list[LagCorrelationPoint],
    *,
    lag_step_s: float,
    min_peak_distance_s: float | None,
    prominence: str | float,
    height: str | float,
) -> list[LagCorrelationPoint]:
    """Find local maxima of signed r (positive correlation peaks) via cardiac-style find_peaks."""
    finite = sorted(
        (point for point in points if np.isfinite(point.r)),
        key=lambda point: point.lag_s,
    )
    if len(finite) < 3:
        return []

    r_values = np.asarray([point.r for point in finite], dtype=float)
    prominence_val = (
        _auto_positive_threshold(r_values, percentile=80)
        if str(prominence).casefold() == "auto"
        else max(0.0, float(prominence))
    )
    if str(height).casefold() == "auto":
        height_val = 0.0
    else:
        height_val = max(0.0, _auto_positive_threshold(r_values, percentile=65))
    resolved_distance = _resolved_min_peak_distance_s(min_peak_distance_s, lag_step_s=lag_step_s)
    sfreq = 1.0 / lag_step_s
    _, peak_idx = _find_peaks(
        r_values,
        sfreq,
        min_peak_distance_s=resolved_distance,
        prominence=prominence_val,
        height=height_val,
    )
    return [finite[int(idx)] for idx in peak_idx if finite[int(idx)].r > 0]


def _select_detected_peak(
    points: list[LagCorrelationPoint],
    *,
    lag_step_s: float,
    min_peak_distance_s: float | None,
    prominence: str | float,
    height: str | float,
) -> tuple[LagCorrelationPoint | None, bool]:
    local_peaks = _find_local_positive_peaks(
        points,
        lag_step_s=lag_step_s,
        min_peak_distance_s=min_peak_distance_s,
        prominence=prominence,
        height=height,
    )
    if local_peaks:
        return _select_positive_max_peak(local_peaks), False
    finite = [point for point in points if np.isfinite(point.r)]
    return _select_positive_max_peak(finite), True


def _point_from_peak(peak: LagCorrelationPoint | None) -> tuple[float, float, float, int]:
    if peak is None:
        return float("nan"), float("nan"), float("nan"), 0
    signed_r = float(peak.r)
    return float(peak.lag_s), signed_r, abs(signed_r), int(peak.n_overlap)


def extract_peaks(
    curve: list[LagCorrelationPoint],
    *,
    lag_step_s: float,
    lag_grid_s: np.ndarray,
    edge_margin_s: float,
    min_peak_distance_s: float | None = None,
    peak_prominence: str | float = "auto",
    peak_height: str | float = "auto",
) -> PeakCorrelationResult:
    min_lag_s, max_lag_s = lag_grid_bounds(lag_grid_s)
    finite = [point for point in curve if np.isfinite(point.r)]
    n_valid_lags = len(finite)

    if not finite:
        nan = float("nan")
        return PeakCorrelationResult(
            raw_peak_lag_s=nan,
            raw_peak_signed_r=nan,
            raw_peak_abs_r=nan,
            raw_peak_at_edge=False,
            interior_peak_lag_s=nan,
            interior_peak_signed_r=nan,
            interior_peak_abs_r=nan,
            preferred_peak_lag_s=nan,
            preferred_peak_signed_r=nan,
            preferred_peak_abs_r=nan,
            preferred_peak_source="raw_peak",
            peak_direction="unknown",
            min_lag_s=min_lag_s,
            max_lag_s=max_lag_s,
            n_valid_lags=0,
            n_overlap_at_raw_peak=0,
            n_overlap_at_preferred_peak=0,
            warning="no_valid_lags",
            p_perm=None,
        )

    raw_point, used_global_fallback = _select_detected_peak(
        finite,
        lag_step_s=lag_step_s,
        min_peak_distance_s=min_peak_distance_s,
        prominence=peak_prominence,
        height=peak_height,
    )
    if raw_point is None:
        nan = float("nan")
        return PeakCorrelationResult(
            raw_peak_lag_s=nan,
            raw_peak_signed_r=nan,
            raw_peak_abs_r=nan,
            raw_peak_at_edge=False,
            interior_peak_lag_s=nan,
            interior_peak_signed_r=nan,
            interior_peak_abs_r=nan,
            preferred_peak_lag_s=nan,
            preferred_peak_signed_r=nan,
            preferred_peak_abs_r=nan,
            preferred_peak_source="raw_peak",
            peak_direction="unknown",
            min_lag_s=min_lag_s,
            max_lag_s=max_lag_s,
            n_valid_lags=n_valid_lags,
            n_overlap_at_raw_peak=0,
            n_overlap_at_preferred_peak=0,
            warning="no_positive_peak",
            p_perm=None,
        )
    raw_lag, raw_signed, raw_abs, raw_overlap = _point_from_peak(raw_point)
    raw_at_edge = _is_edge_lag(
        raw_lag,
        min_lag_s=min_lag_s,
        max_lag_s=max_lag_s,
        edge_margin_s=edge_margin_s,
    )

    interior_candidates = [
        point
        for point in finite
        if not _is_edge_lag(
            point.lag_s,
            min_lag_s=min_lag_s,
            max_lag_s=max_lag_s,
            edge_margin_s=edge_margin_s,
        )
    ]
    interior_point, _ = _select_detected_peak(
        interior_candidates,
        lag_step_s=lag_step_s,
        min_peak_distance_s=min_peak_distance_s,
        prominence=peak_prominence,
        height=peak_height,
    )
    interior_lag, interior_signed, interior_abs, interior_overlap = _point_from_peak(interior_point)

    use_interior_preferred = (
        raw_at_edge
        and interior_point is not None
        and _same_correlation_sign(raw_signed, interior_signed)
    )
    if use_interior_preferred:
        preferred_lag = interior_lag
        preferred_signed = interior_signed
        preferred_abs = interior_abs
        preferred_overlap = interior_overlap
        preferred_source = "interior_peak_due_to_edge"
    else:
        preferred_lag = raw_lag
        preferred_signed = raw_signed
        preferred_abs = raw_abs
        preferred_overlap = raw_overlap
        preferred_source = "raw_peak"

    warnings_out: list[str] = []
    if used_global_fallback:
        warnings_out.append("no_prominent_local_peak")
    if raw_at_edge:
        warnings_out.append("raw_peak_at_edge")
    if raw_at_edge and interior_point is None:
        warnings_out.append("no_interior_peak")
    if raw_at_edge and interior_point is not None and not use_interior_preferred:
        warnings_out.append("opposite_sign_interior_rejected")
    if n_valid_lags < 3:
        warnings_out.append("few_valid_lags")
    if raw_overlap < MIN_OVERLAP_POINTS:
        warnings_out.append("low_overlap_at_raw_peak")
    if preferred_overlap < MIN_OVERLAP_POINTS:
        warnings_out.append("low_overlap_at_preferred_peak")
    if preferred_abs < 0.1:
        warnings_out.append("weak_peak_correlation")

    return PeakCorrelationResult(
        raw_peak_lag_s=raw_lag,
        raw_peak_signed_r=raw_signed,
        raw_peak_abs_r=raw_abs,
        raw_peak_at_edge=raw_at_edge,
        interior_peak_lag_s=interior_lag,
        interior_peak_signed_r=interior_signed,
        interior_peak_abs_r=interior_abs,
        preferred_peak_lag_s=preferred_lag,
        preferred_peak_signed_r=preferred_signed,
        preferred_peak_abs_r=preferred_abs,
        preferred_peak_source=preferred_source,
        peak_direction=_peak_direction(raw_lag, lag_step_s=lag_step_s),
        min_lag_s=min_lag_s,
        max_lag_s=max_lag_s,
        n_valid_lags=n_valid_lags,
        n_overlap_at_raw_peak=raw_overlap,
        n_overlap_at_preferred_peak=preferred_overlap,
        warning=";".join(dict.fromkeys(warnings_out)),
        p_perm=None,
    )


def extract_peak(
    curve: list[LagCorrelationPoint],
    *,
    lag_step_s: float,
    lag_grid_s: np.ndarray,
    edge_margin_s: float = 5.0,
    min_peak_distance_s: float | None = None,
    peak_prominence: str | float = "auto",
    peak_height: str | float = "auto",
) -> PeakCorrelationResult:
    return extract_peaks(
        curve,
        lag_step_s=lag_step_s,
        lag_grid_s=lag_grid_s,
        edge_margin_s=edge_margin_s,
        min_peak_distance_s=min_peak_distance_s,
        peak_prominence=peak_prominence,
        peak_height=peak_height,
    )


def _circular_shift(values: np.ndarray, shift: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return values
    if shift == 0:
        shift = int(rng.integers(1, values.size))
    out = np.roll(values, shift)
    out[~np.isfinite(values)] = np.nan
    return out


@dataclass(frozen=True)
class SubjectPairData:
    subject_id: str
    cardiac: np.ndarray
    eeg: np.ndarray
    lag_grid_s: np.ndarray


def null_peak_abs_r_from_shift(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_grid_s: np.ndarray,
    cfg: TemporalCouplingConfig,
    rng: np.random.Generator,
) -> float:
    """Circularly shift EEG, recompute lag curve, return raw peak |r| (same rules as Stage 2)."""
    xcorr_cfg = cfg.temporal_coupling.cross_correlation
    fs_hz = cfg.temporal_coupling.resample.fs_hz
    shifted = _circular_shift(eeg, 0, rng)
    curve = compute_correlation_curve(
        cardiac,
        shifted,
        lag_grid_s=lag_grid_s,
        fs_hz=fs_hz,
    )
    peak = extract_peaks(
        curve,
        lag_step_s=xcorr_cfg.lag_step_s,
        lag_grid_s=lag_grid_s,
        edge_margin_s=xcorr_cfg.edge_margin_s,
        min_peak_distance_s=xcorr_cfg.min_peak_distance_s,
        peak_prominence=xcorr_cfg.peak_prominence,
        peak_height=xcorr_cfg.peak_height,
    )
    return float(peak.raw_peak_abs_r)


def load_subject_pair_datasets(
    cfg: TemporalCouplingConfig,
    *,
    pair: VariablePair,
    subject_ids: list[str],
    alignment_qc: pd.DataFrame | None,
) -> list[SubjectPairData]:
    """Load aligned cardiac/EEG series and lag grids for one pair across subjects."""
    from .resample import aligned_output_path

    xcorr_cfg = cfg.temporal_coupling.cross_correlation
    fs_hz = cfg.temporal_coupling.resample.fs_hz
    datasets: list[SubjectPairData] = []

    for observation_id in subject_ids:
        aligned_path = aligned_output_path(cfg, observation_id)
        if not aligned_path.is_file():
            continue
        aligned_df = pd.read_csv(aligned_path)
        lag_max_s = _resolve_lag_max_s(cfg, observation_id=observation_id, alignment_qc=alignment_qc)
        lag_grid_s = build_lag_grid(lag_max_s=lag_max_s, lag_step_s=xcorr_cfg.lag_step_s)
        cardiac = _prepare_series(
            aligned_df[pair.cardiac_var].to_numpy(dtype=float),
            xcorr_cfg,
            fs_hz=fs_hz,
        )
        eeg = _prepare_series(
            aligned_df[pair.eeg_var].to_numpy(dtype=float),
            xcorr_cfg,
            fs_hz=fs_hz,
        )
        datasets.append(
            SubjectPairData(
                subject_id=str(aligned_df["subject_id"].iloc[0]),
                cardiac=cardiac,
                eeg=eeg,
                lag_grid_s=lag_grid_s,
            )
        )
    return datasets


def group_permutation_p_value(
    subject_data: list[SubjectPairData],
    *,
    observed_group_stat: float,
    n_group_permutations: int,
    cfg: TemporalCouplingConfig,
    rng: np.random.Generator,
) -> float | None:
    """Empirical p-value: fraction of null median peak |r| >= observed group median."""
    if n_group_permutations <= 0 or not np.isfinite(observed_group_stat) or not subject_data:
        return None

    null_stats: list[float] = []
    for _ in range(n_group_permutations):
        subject_nulls: list[float] = []
        for data in subject_data:
            abs_r = null_peak_abs_r_from_shift(
                data.cardiac,
                data.eeg,
                lag_grid_s=data.lag_grid_s,
                cfg=cfg,
                rng=rng,
            )
            if np.isfinite(abs_r):
                subject_nulls.append(abs_r)
        if subject_nulls:
            null_stats.append(float(np.median(subject_nulls)))

    if not null_stats:
        return None

    exceed = sum(1 for value in null_stats if value >= observed_group_stat)
    return float((exceed + 1) / (len(null_stats) + 1))


def permutation_p_value(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_grid_s: np.ndarray,
    fs_hz: float,
    observed_peak_abs_r: float,
    n_permutations: int,
    rng: np.random.Generator,
    min_overlap: int = MIN_OVERLAP_POINTS,
    lag_step_s: float,
    edge_margin_s: float = 5.0,
    min_peak_distance_s: float | None = None,
    peak_prominence: str | float = "auto",
    peak_height: str | float = "auto",
) -> float | None:
    if n_permutations <= 0 or not np.isfinite(observed_peak_abs_r):
        return None

    null_peaks: list[float] = []
    for _ in range(n_permutations):
        shifted = _circular_shift(eeg, 0, rng)
        curve = compute_correlation_curve(
            cardiac,
            shifted,
            lag_grid_s=lag_grid_s,
            fs_hz=fs_hz,
            min_overlap=min_overlap,
        )
        peak = extract_peaks(
            curve,
            lag_step_s=lag_step_s,
            lag_grid_s=lag_grid_s,
            edge_margin_s=edge_margin_s,
            min_peak_distance_s=min_peak_distance_s,
            peak_prominence=peak_prominence,
            peak_height=peak_height,
        )
        if np.isfinite(peak.raw_peak_abs_r):
            null_peaks.append(peak.raw_peak_abs_r)

    if not null_peaks:
        return None

    exceed = sum(1 for value in null_peaks if value >= observed_peak_abs_r)
    return float((exceed + 1) / (len(null_peaks) + 1))


def _resolve_lag_max_s(
    cfg: TemporalCouplingConfig,
    *,
    observation_id: str,
    alignment_qc: pd.DataFrame | None,
) -> float:
    lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s
    if alignment_qc is None or alignment_qc.empty:
        return lag_max_s

    rows = alignment_qc.loc[alignment_qc["observation_id"] == observation_id]
    if rows.empty and "subject_id" in alignment_qc.columns:
        rows = alignment_qc.loc[alignment_qc["subject_id"] == observation_id]
    if rows.empty or "recommended_xcorr_lag_s" not in rows.columns:
        return lag_max_s

    recommended = float(rows["recommended_xcorr_lag_s"].iloc[0])
    if not np.isfinite(recommended) or recommended <= 0:
        return lag_max_s
    return min(lag_max_s, recommended)


def _usable_observation_ids(alignment_qc: pd.DataFrame | None, observations: list[str]) -> set[str]:
    if alignment_qc is None or alignment_qc.empty:
        return set(observations)
    if "usable_for_xcorr" not in alignment_qc.columns:
        return set(observations)
    usable = alignment_qc.loc[
        alignment_qc["usable_for_xcorr"].astype(str).str.lower().isin({"true", "1", "yes"})
    ]
    if "observation_id" in usable.columns:
        return set(usable["observation_id"].astype(str))
    return set(usable["subject_id"].astype(str))


def _load_alignment_qc(cfg: TemporalCouplingConfig) -> pd.DataFrame | None:
    qc_path = alignment_qc_path(cfg)
    if not qc_path.is_file():
        return None
    return pd.read_csv(qc_path)


def _peak_to_row(meta: dict[str, str], pair: VariablePair, peak: PeakCorrelationResult) -> dict[str, object]:
    return {
        **meta,
        "pair": pair.pair,
        "cardiac_var": pair.cardiac_var,
        "eeg_var": pair.eeg_var,
        "raw_peak_lag_s": peak.raw_peak_lag_s,
        "raw_peak_signed_r": peak.raw_peak_signed_r,
        "raw_peak_abs_r": peak.raw_peak_abs_r,
        "raw_peak_at_edge": peak.raw_peak_at_edge,
        "interior_peak_lag_s": peak.interior_peak_lag_s,
        "interior_peak_signed_r": peak.interior_peak_signed_r,
        "interior_peak_abs_r": peak.interior_peak_abs_r,
        "preferred_peak_lag_s": peak.preferred_peak_lag_s,
        "preferred_peak_signed_r": peak.preferred_peak_signed_r,
        "preferred_peak_abs_r": peak.preferred_peak_abs_r,
        "preferred_peak_source": peak.preferred_peak_source,
        "peak_lag_s": peak.raw_peak_lag_s,
        "peak_signed_r": peak.raw_peak_signed_r,
        "peak_abs_r": peak.raw_peak_abs_r,
        "peak_direction": peak.peak_direction,
        "peak_at_lag_edge": peak.raw_peak_at_edge,
        "min_lag_s": peak.min_lag_s,
        "max_lag_s": peak.max_lag_s,
        "n_valid_lags": peak.n_valid_lags,
        "n_overlap_at_raw_peak": peak.n_overlap_at_raw_peak,
        "n_overlap_at_preferred_peak": peak.n_overlap_at_preferred_peak,
        "warning": peak.warning,
        "p_perm": peak.p_perm,
    }


def compute_observation_xcorr(
    aligned_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
    *,
    lag_max_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    xcorr_cfg = cfg.temporal_coupling.cross_correlation
    output_cfg = cfg.temporal_coupling.output
    fs_hz = cfg.temporal_coupling.resample.fs_hz
    lag_step_s = xcorr_cfg.lag_step_s
    lag_grid_s = build_lag_grid(lag_max_s=lag_max_s, lag_step_s=lag_step_s)

    row0 = aligned_df.iloc[0]
    observation_id = str(row0["observation_id"])
    task = str(row0["task"])
    if "condition" in aligned_df.columns and pd.notna(row0["condition"]) and str(row0["condition"]) != task:
        condition = str(row0["condition"])
    else:
        condition = hiit_condition_from_observation_id(observation_id) or task
    meta = {
        "dataset_id": str(row0["dataset_id"]),
        "subject_id": str(row0["subject_id"]),
        "task": task,
        "condition": condition,
        "observation_id": observation_id,
    }

    curve_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)

    for pair in variable_pairs():
        cardiac = _prepare_series(
            aligned_df[pair.cardiac_var].to_numpy(dtype=float),
            xcorr_cfg,
            fs_hz=fs_hz,
        )
        eeg = _prepare_series(
            aligned_df[pair.eeg_var].to_numpy(dtype=float),
            xcorr_cfg,
            fs_hz=fs_hz,
        )
        curve = compute_correlation_curve(
            cardiac,
            eeg,
            lag_grid_s=lag_grid_s,
            fs_hz=fs_hz,
        )
        peak = extract_peaks(
            curve,
            lag_step_s=lag_step_s,
            lag_grid_s=lag_grid_s,
            edge_margin_s=xcorr_cfg.edge_margin_s,
            min_peak_distance_s=xcorr_cfg.min_peak_distance_s,
            peak_prominence=xcorr_cfg.peak_prominence,
            peak_height=xcorr_cfg.peak_height,
        )
        if xcorr_cfg.n_permutations > 0:
            p_perm = permutation_p_value(
                cardiac,
                eeg,
                lag_grid_s=lag_grid_s,
                fs_hz=fs_hz,
                observed_peak_abs_r=peak.raw_peak_abs_r,
                n_permutations=xcorr_cfg.n_permutations,
                rng=rng,
                lag_step_s=lag_step_s,
                edge_margin_s=xcorr_cfg.edge_margin_s,
                min_peak_distance_s=xcorr_cfg.min_peak_distance_s,
                peak_prominence=xcorr_cfg.peak_prominence,
                peak_height=xcorr_cfg.peak_height,
            )
            peak = PeakCorrelationResult(
                raw_peak_lag_s=peak.raw_peak_lag_s,
                raw_peak_signed_r=peak.raw_peak_signed_r,
                raw_peak_abs_r=peak.raw_peak_abs_r,
                raw_peak_at_edge=peak.raw_peak_at_edge,
                interior_peak_lag_s=peak.interior_peak_lag_s,
                interior_peak_signed_r=peak.interior_peak_signed_r,
                interior_peak_abs_r=peak.interior_peak_abs_r,
                preferred_peak_lag_s=peak.preferred_peak_lag_s,
                preferred_peak_signed_r=peak.preferred_peak_signed_r,
                preferred_peak_abs_r=peak.preferred_peak_abs_r,
                preferred_peak_source=peak.preferred_peak_source,
                peak_direction=peak.peak_direction,
                min_lag_s=peak.min_lag_s,
                max_lag_s=peak.max_lag_s,
                n_valid_lags=peak.n_valid_lags,
                n_overlap_at_raw_peak=peak.n_overlap_at_raw_peak,
                n_overlap_at_preferred_peak=peak.n_overlap_at_preferred_peak,
                warning=peak.warning,
                p_perm=p_perm,
            )

        if output_cfg.save_curves:
            for point in curve:
                curve_rows.append(
                    {
                        **meta,
                        "pair": pair.pair,
                        "cardiac_var": pair.cardiac_var,
                        "eeg_var": pair.eeg_var,
                        "lag_s": point.lag_s,
                        "r": point.r,
                        "n_overlap": point.n_overlap,
                    }
                )

        peak_rows.append(_peak_to_row(meta, pair, peak))

    return curve_rows, peak_rows


def _panel_peak_title(peak_row: pd.Series, *, label: str) -> str:
    raw_lag = float(peak_row["raw_peak_lag_s"])
    raw_r = float(peak_row["raw_peak_signed_r"])
    raw_abs = float(peak_row["raw_peak_abs_r"])
    at_edge = bool(peak_row.get("raw_peak_at_edge", False))
    edge_note = " [EDGE]" if at_edge else ""
    lines = [f"{label}{edge_note}", f"peak |r|={raw_abs:.2f}, r={raw_r:+.2f}, lag={raw_lag:.0f}s"]
    pref_lag = float(peak_row["preferred_peak_lag_s"])
    if (
        peak_row.get("preferred_peak_source") == "interior_peak_due_to_edge"
        and np.isfinite(pref_lag)
        and not np.isclose(pref_lag, raw_lag, atol=PEAK_VALIDATION_TOL)
    ):
        pref_r = float(peak_row["preferred_peak_signed_r"])
        pref_abs = float(peak_row["preferred_peak_abs_r"])
        lines.append(f"QC interior: |r|={pref_abs:.2f}, r={pref_r:+.2f}, lag={pref_lag:.0f}s")
    return "\n".join(lines)


def _plot_peak_markers(
    ax: plt.Axes,
    *,
    peak_row: pd.Series,
    line_color: str,
) -> None:
    raw_lag = float(peak_row["raw_peak_lag_s"])
    raw_r = float(peak_row["raw_peak_signed_r"])
    at_edge = bool(peak_row["raw_peak_at_edge"])
    if np.isfinite(raw_lag) and np.isfinite(raw_r):
        ax.scatter(
            [raw_lag],
            [raw_r],
            s=90,
            zorder=5,
            marker="o",
            color=line_color,
            edgecolors="black",
            linewidths=0.6,
        )
        if at_edge:
            ax.annotate(
                "EDGE",
                xy=(raw_lag, raw_r),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold",
                color="#d62728",
                zorder=7,
            )

    pref_lag = float(peak_row["preferred_peak_lag_s"])
    pref_r = float(peak_row["preferred_peak_signed_r"])
    show_interior_qc = (
        peak_row.get("preferred_peak_source") == "interior_peak_due_to_edge"
        and np.isfinite(pref_lag)
        and np.isfinite(pref_r)
        and not np.isclose(pref_lag, raw_lag, atol=PEAK_VALIDATION_TOL)
    )
    if show_interior_qc:
        ax.scatter(
            [pref_lag],
            [pref_r],
            s=35,
            zorder=4,
            marker="o",
            facecolors="none",
            edgecolors=line_color,
            linewidths=1.0,
            alpha=0.85,
        )


def _overlay_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="subject cross-correlation curve"),
        Line2D([0], [0], color="black", linewidth=2.5, label="mean curve"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markeredgecolor="black",
            markersize=10,
            label="detected positive peak (scipy find_peaks)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="#1f77b4",
            markersize=7,
            markeredgewidth=1.0,
            label="QC interior alternative",
        ),
        Line2D([0], [0], color="0.7", linestyle="--", linewidth=0.8, label="lag 0"),
    ]


def validate_peaks_against_curves(
    curve_rows: list[dict[str, object]],
    peak_rows: list[dict[str, object]],
    *,
    tol: float = PEAK_VALIDATION_TOL,
    lag_step_s: float = 5.0,
    min_peak_distance_s: float | None = None,
    peak_prominence: str | float = "auto",
    peak_height: str | float = "auto",
) -> tuple[list[dict[str, object]], bool]:
    curves_df = pd.DataFrame(curve_rows)
    validation_rows: list[dict[str, object]] = []
    all_passed = True

    for peak in peak_rows:
        subject_id = str(peak["subject_id"])
        pair = str(peak["pair"])
        curve = curves_df.loc[
            (curves_df["subject_id"] == subject_id) & (curves_df["pair"] == pair)
        ].copy()
        finite = curve.loc[np.isfinite(curve["r"].astype(float))]
        warnings_out: list[str] = []

        if finite.empty:
            expected_lag = float("nan")
            expected_abs = float("nan")
            passed = False
            warnings_out.append("no_finite_curve_values")
            all_passed = False
        else:
            curve_points = [
                LagCorrelationPoint(
                    lag_s=float(row["lag_s"]),
                    r=float(row["r"]),
                    n_overlap=int(row.get("n_overlap", 0)),
                )
                for _, row in finite.iterrows()
            ]
            expected_point, _ = _select_detected_peak(
                curve_points,
                lag_step_s=lag_step_s,
                min_peak_distance_s=min_peak_distance_s,
                prominence=peak_prominence,
                height=peak_height,
            )
            if expected_point is None:
                expected_lag = float("nan")
                expected_abs = float("nan")
                passed = False
                warnings_out.append("no_expected_peak")
                all_passed = False
                validation_rows.append(
                    {
                        "subject_id": subject_id,
                        "pair": pair,
                        "expected_raw_peak_lag_s": expected_lag,
                        "stored_raw_peak_lag_s": peak["raw_peak_lag_s"],
                        "expected_raw_peak_abs_r": expected_abs,
                        "stored_raw_peak_abs_r": peak["raw_peak_abs_r"],
                        "passed": passed,
                        "warning": ";".join(warnings_out),
                    }
                )
                continue
            expected_lag = float(expected_point.lag_s)
            expected_abs = abs(float(expected_point.r))
            stored_lag = float(peak["raw_peak_lag_s"])
            stored_abs = float(peak["raw_peak_abs_r"])
            lag_ok = np.isclose(stored_lag, expected_lag, atol=tol)
            abs_ok = np.isclose(stored_abs, expected_abs, atol=tol)
            passed = bool(lag_ok and abs_ok)
            if not lag_ok:
                warnings_out.append("raw_peak_lag_mismatch")
            if not abs_ok:
                warnings_out.append("raw_peak_abs_r_mismatch")
            if not passed:
                all_passed = False

        validation_rows.append(
            {
                "subject_id": subject_id,
                "pair": pair,
                "expected_raw_peak_lag_s": expected_lag,
                "stored_raw_peak_lag_s": peak["raw_peak_lag_s"],
                "expected_raw_peak_abs_r": expected_abs,
                "stored_raw_peak_abs_r": peak["raw_peak_abs_r"],
                "passed": passed,
                "warning": ";".join(warnings_out),
            }
        )

    return validation_rows, all_passed


def _plot_subject_grid(
    subject_id: str,
    curves_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    pairs = variable_pairs()

    for idx, pair in enumerate(pairs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        pair_curves = curves_df.loc[curves_df["pair"] == pair.pair].sort_values("lag_s")
        pair_peak = peaks_df.loc[peaks_df["pair"] == pair.pair]
        if pair_curves.empty or pair_peak.empty:
            ax.set_title(f"{pair.pair}\n(no data)")
            ax.axvline(0.0, color="0.7", linewidth=0.8)
            continue

        line = ax.plot(pair_curves["lag_s"], pair_curves["r"], color="#1f77b4", linewidth=1.5)[0]
        ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
        peak_row = pair_peak.iloc[0]
        _plot_peak_markers(ax, peak_row=peak_row, line_color=line.get_color())

        panel_label = f"{EEG_LABELS[col]}: {pair.pair}" if row == 0 else pair.pair
        ax.set_title(_panel_peak_title(peak_row, label=panel_label), fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{CARDIAC_LABELS[row]}\nr")
        if row == 2:
            ax.set_xlabel("Lag (s)")

    fig.suptitle(f"Cross-correlation grid: {subject_id}", fontsize=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _pair_qc_warnings(
    pair_rows: pd.DataFrame,
    *,
    n_permutations: int,
    lag_step_s: float,
) -> str:
    warnings_out: list[str] = []
    n_subjects = len(pair_rows)
    if n_subjects < 3:
        warnings_out.append("few_subjects")

    n_edge = int(pair_rows["raw_peak_at_edge"].astype(bool).sum()) if n_subjects else 0
    if n_subjects > 0 and (100.0 * n_edge / n_subjects) > 50.0:
        warnings_out.append("high_edge_peak_rate")

    threshold = lag_step_s / 2.0
    lags = pair_rows["raw_peak_lag_s"].astype(float)
    has_negative = bool((lags < -threshold).any())
    has_positive = bool((lags > threshold).any())
    if has_negative and has_positive:
        warnings_out.append("mixed_peak_lag_direction")

    if n_permutations == 0:
        warnings_out.append("no_permutation_test")

    return ";".join(warnings_out)


def build_qc_summary(
    peaks_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
) -> list[dict[str, object]]:
    xcorr_cfg = cfg.temporal_coupling.cross_correlation
    summary_rows: list[dict[str, object]] = []

    for pair in variable_pairs():
        pair_rows = peaks_df.loc[peaks_df["pair"] == pair.pair]
        n_subjects = len(pair_rows)
        n_edge = int(pair_rows["raw_peak_at_edge"].astype(bool).sum()) if n_subjects else 0
        percent_edge = float(100.0 * n_edge / n_subjects) if n_subjects else float("nan")

        if n_subjects:
            median_lag = float(pair_rows["raw_peak_lag_s"].median())
            mean_lag = float(pair_rows["raw_peak_lag_s"].mean())
            median_abs_r = float(pair_rows["raw_peak_abs_r"].median())
            mean_signed_r = float(pair_rows["raw_peak_signed_r"].mean())
            median_pref_lag = float(pair_rows["preferred_peak_lag_s"].median())
            mean_pref_lag = float(pair_rows["preferred_peak_lag_s"].mean())
            median_pref_abs_r = float(pair_rows["preferred_peak_abs_r"].median())
            median_interior_lag = float(pair_rows["interior_peak_lag_s"].median())
            median_interior_abs_r = float(pair_rows["interior_peak_abs_r"].median())
        else:
            median_lag = float("nan")
            mean_lag = float("nan")
            median_abs_r = float("nan")
            mean_signed_r = float("nan")
            median_pref_lag = float("nan")
            mean_pref_lag = float("nan")
            median_pref_abs_r = float("nan")
            median_interior_lag = float("nan")
            median_interior_abs_r = float("nan")

        summary_rows.append(
            {
                "pair": pair.pair,
                "n_subjects": n_subjects,
                "n_edge_peaks": n_edge,
                "percent_edge_peaks": percent_edge,
                "median_peak_lag_s": median_lag,
                "mean_peak_lag_s": mean_lag,
                "median_peak_abs_r": median_abs_r,
                "mean_peak_signed_r": mean_signed_r,
                "median_preferred_peak_lag_s": median_pref_lag,
                "mean_preferred_peak_lag_s": mean_pref_lag,
                "median_preferred_peak_abs_r": median_pref_abs_r,
                "median_interior_peak_lag_s": median_interior_lag,
                "median_interior_peak_abs_r": median_interior_abs_r,
                "warning": _pair_qc_warnings(
                    pair_rows,
                    n_permutations=xcorr_cfg.n_permutations,
                    lag_step_s=xcorr_cfg.lag_step_s,
                ),
            }
        )

    return summary_rows


def _pair_common_lag_bounds(pair_curves: pd.DataFrame) -> tuple[float, float] | None:
    total = int(pair_curves["subject_id"].nunique())
    if total == 0:
        return None
    lag_counts = pair_curves.groupby("lag_s")["subject_id"].nunique()
    common_lags = lag_counts[lag_counts == total].index.astype(float)
    if common_lags.empty:
        return None
    return float(common_lags.min()), float(common_lags.max())


def _pair_mean_curve_in_common_range(pair_curves: pd.DataFrame) -> pd.DataFrame:
    common_bounds = _pair_common_lag_bounds(pair_curves)
    total = int(pair_curves["subject_id"].nunique())
    rows: list[dict[str, object]] = []
    for lag_s, lag_rows in pair_curves.groupby("lag_s"):
        lag_s = float(lag_s)
        if common_bounds is not None:
            lo, hi = common_bounds
            if lag_s < lo or lag_s > hi:
                continue
        finite = lag_rows["r"].astype(float)
        finite = finite[np.isfinite(finite)]
        if int(finite.size) < total:
            continue
        rows.append({"lag_s": lag_s, "r": float(finite.mean())})
    if not rows:
        return pd.DataFrame(columns=["lag_s", "r"])
    return pd.DataFrame(rows).sort_values("lag_s")


def _plot_pair_overlay(
    pair: VariablePair,
    curves_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    pair_curves = curves_df.loc[curves_df["pair"] == pair.pair]
    pair_peaks = peaks_df.loc[peaks_df["pair"] == pair.pair]
    if pair_curves.empty:
        ax.set_title(f"{pair.pair} overlay (no data)")
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return

    for subject_id, subject_curves in pair_curves.groupby("subject_id"):
        subject_curves = subject_curves.sort_values("lag_s")
        line = ax.plot(
            subject_curves["lag_s"],
            subject_curves["r"],
            linewidth=1.0,
            alpha=0.7,
            label=str(subject_id),
        )[0]
        subject_peak = pair_peaks.loc[pair_peaks["subject_id"] == subject_id]
        if subject_peak.empty:
            continue
        _plot_peak_markers(
            ax,
            peak_row=subject_peak.iloc[0],
            line_color=line.get_color(),
        )

    mean_curve = _pair_mean_curve_in_common_range(pair_curves)
    if not mean_curve.empty:
        ax.plot(mean_curve["lag_s"], mean_curve["r"], color="black", linewidth=2.5)

    ax.axvline(0.0, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Correlation r")

    common_bounds = _pair_common_lag_bounds(pair_curves)
    if common_bounds is not None:
        ax.set_xlim(common_bounds[0], common_bounds[1])

    n_subjects = int(pair_peaks["subject_id"].nunique()) if not pair_peaks.empty else 0
    n_edge = int(pair_peaks["raw_peak_at_edge"].astype(bool).sum()) if not pair_peaks.empty else 0
    median_raw_lag = float(pair_peaks["raw_peak_lag_s"].median()) if not pair_peaks.empty else float("nan")
    median_pref_lag = float(pair_peaks["preferred_peak_lag_s"].median()) if not pair_peaks.empty else float("nan")
    common_note = (
        f", common_lag=[{common_bounds[0]:g}, {common_bounds[1]:g}]s"
        if common_bounds is not None
        else ""
    )
    title_lines = [
        f"Cross-correlation overlay: {pair.pair}",
        (
            f"subjects={n_subjects}, edge_peaks={n_edge}, "
            f"median_preferred_lag={median_pref_lag:.1f}s, median_raw_lag={median_raw_lag:.1f}s"
            f"{common_note}"
        ),
    ]
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.legend(handles=_overlay_legend_handles(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_peak_lag_distribution(
    peaks_df: pd.DataFrame,
    output_path: Path,
    *,
    peak_kind: str,
) -> None:
    lag_col = "raw_peak_lag_s" if peak_kind == "raw" else "preferred_peak_lag_s"
    title_kind = "raw peak" if peak_kind == "raw" else "preferred peak"

    fig, ax = plt.subplots(figsize=(12, 5))
    pairs = [pair.pair for pair in variable_pairs()]
    x_positions = {pair: idx for idx, pair in enumerate(pairs)}

    for row in peaks_df.itertuples(index=False):
        x = x_positions[str(row.pair)]
        is_edge = bool(getattr(row, "raw_peak_at_edge", False))
        ax.scatter(
            x,
            float(getattr(row, lag_col)),
            color="#d62728" if is_edge else "#1f77b4",
            marker="X" if is_edge else "o",
            s=70 if is_edge else 50,
            linewidths=0.8,
            zorder=3 if is_edge else 2,
        )

    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel("Peak lag (s)")
    ax.set_title(f"Peak lag distribution by pair ({title_kind})")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#1f77b4",
                markersize=8,
                label=f"{title_kind} (non-edge raw)",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#d62728",
                linestyle="None",
                markersize=9,
                markeredgewidth=1.0,
                label="raw peak at edge",
            ),
        ],
        loc="best",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_peak_strength_distribution(
    peaks_df: pd.DataFrame,
    output_path: Path,
    *,
    peak_kind: str,
) -> None:
    if peak_kind == "raw":
        value_col = "raw_peak_signed_r"
        title_kind = "raw peak"
        ylabel = "raw_peak_signed_r"
    else:
        value_col = "preferred_peak_signed_r"
        title_kind = "preferred peak"
        ylabel = "preferred_peak_signed_r"

    fig, ax = plt.subplots(figsize=(12, 5))
    pairs = [pair.pair for pair in variable_pairs()]
    x_positions = {pair: idx for idx, pair in enumerate(pairs)}

    for row in peaks_df.itertuples(index=False):
        x = x_positions[str(row.pair)]
        is_edge = bool(getattr(row, "raw_peak_at_edge", False))
        ax.scatter(
            x,
            float(getattr(row, value_col)),
            color="#d62728" if is_edge else "#1f77b4",
            marker="X" if is_edge else "o",
            s=70 if is_edge else 50,
            linewidths=0.8,
            zorder=3 if is_edge else 2,
        )

    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Peak correlation strength by pair ({title_kind}; y-axis: {ylabel})")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#1f77b4",
                markersize=8,
                label=f"{title_kind} (non-edge raw)",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="#d62728",
                linestyle="None",
                markersize=9,
                markeredgewidth=1.0,
                label="raw peak at edge",
            ),
        ],
        loc="best",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_edge_peak_summary(peaks_df: pd.DataFrame, output_path: Path) -> None:
    pairs = [pair.pair for pair in variable_pairs()]
    counts: list[int] = []
    percents: list[float] = []
    for pair in pairs:
        pair_rows = peaks_df.loc[peaks_df["pair"] == pair]
        n_total = len(pair_rows)
        n_edge = int(pair_rows["raw_peak_at_edge"].astype(bool).sum()) if n_total else 0
        counts.append(n_edge)
        percents.append(100.0 * n_edge / n_total if n_total else 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pairs))
    ax.bar(x, counts, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel("Subjects with edge peak")
    ax.set_title("Edge-peak count by pair")
    for idx, (count, pct) in enumerate(zip(counts, percents, strict=True)):
        ax.text(idx, count + 0.05, f"{pct:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_qc_plots(
    cfg: TemporalCouplingConfig,
    *,
    all_curve_rows: list[dict[str, object]],
    all_peak_rows: list[dict[str, object]],
    processed_observations: list[str],
) -> list[Path]:
    if not cfg.temporal_coupling.output.save_plots:
        return []

    group_dir = group_output_dir(cfg)
    group_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    curves_df = pd.DataFrame(all_curve_rows)
    peaks_df = pd.DataFrame(all_peak_rows)

    for observation_id in processed_observations:
        obs_curves = curves_df.loc[curves_df["observation_id"] == observation_id]
        obs_peaks = peaks_df.loc[peaks_df["observation_id"] == observation_id]
        if obs_curves.empty or obs_peaks.empty:
            continue
        out_path = observation_output_dir(cfg, observation_id) / GRID_PLOT
        _plot_subject_grid(observation_id, obs_curves, obs_peaks, out_path)
        plot_paths.append(out_path)

    for pair in variable_pairs():
        out_path = group_dir / OVERLAY_PLOT_TEMPLATE.format(pair=pair.pair)
        _plot_pair_overlay(pair, curves_df, peaks_df, out_path)
        plot_paths.append(out_path)

    lag_raw_path = group_dir / PEAK_LAG_DIST_RAW_PLOT
    _plot_peak_lag_distribution(peaks_df, lag_raw_path, peak_kind="raw")
    plot_paths.append(lag_raw_path)

    lag_pref_path = group_dir / PEAK_LAG_DIST_PREFERRED_PLOT
    _plot_peak_lag_distribution(peaks_df, lag_pref_path, peak_kind="preferred")
    plot_paths.append(lag_pref_path)

    strength_raw_path = group_dir / PEAK_STRENGTH_RAW_PLOT
    _plot_peak_strength_distribution(peaks_df, strength_raw_path, peak_kind="raw")
    plot_paths.append(strength_raw_path)

    strength_pref_path = group_dir / PEAK_STRENGTH_PREFERRED_PLOT
    _plot_peak_strength_distribution(peaks_df, strength_pref_path, peak_kind="preferred")
    plot_paths.append(strength_pref_path)

    edge_path = group_dir / EDGE_PEAK_SUMMARY_PLOT
    _plot_edge_peak_summary(peaks_df, edge_path)
    plot_paths.append(edge_path)

    return plot_paths


def run_stage2(cfg: TemporalCouplingConfig) -> list[Path]:
    alignment_qc = _load_alignment_qc(cfg)
    if alignment_qc is not None and not alignment_qc.empty and "observation_id" in alignment_qc.columns:
        observation_ids = sorted(alignment_qc["observation_id"].astype(str).unique())
    elif alignment_qc is not None and not alignment_qc.empty:
        observation_ids = sorted(alignment_qc["subject_id"].astype(str).unique())
    else:
        dataset_dir = Path(cfg.paths.out_root) / cfg.dataset_id
        observation_ids = sorted(
            path.name
            for path in dataset_dir.iterdir()
            if path.is_dir() and path.name != "group" and (path / "features_temporal_aligned.csv").is_file()
        )

    if not observation_ids:
        print("[temporal_coupling] stage=2: no aligned observations found.")
        return []

    usable_ids = _usable_observation_ids(alignment_qc, observation_ids)
    written: list[Path] = []
    all_curve_rows: list[dict[str, object]] = []
    all_peak_rows: list[dict[str, object]] = []
    processed_observations: list[str] = []
    observation_lag_ranges: dict[str, tuple[float, float]] = {}
    n_ok = 0

    for observation_id in observation_ids:
        if observation_id not in usable_ids:
            warnings.warn(
                f"[temporal_coupling] stage=2 skipping {observation_id}: usable_for_xcorr=False.",
                stacklevel=2,
            )
            continue

        aligned_path = aligned_output_path(cfg, observation_id)
        if not aligned_path.is_file():
            warnings.warn(
                f"[temporal_coupling] stage=2 skipping {observation_id}: missing {aligned_path.name}. "
                "Run --stage 1c first.",
                stacklevel=2,
            )
            continue

        try:
            aligned_df = pd.read_csv(aligned_path)
            lag_max_s = _resolve_lag_max_s(cfg, observation_id=observation_id, alignment_qc=alignment_qc)
            lag_grid_s = build_lag_grid(
                lag_max_s=lag_max_s,
                lag_step_s=cfg.temporal_coupling.cross_correlation.lag_step_s,
            )
            min_lag_s, max_lag_s = lag_grid_bounds(lag_grid_s)
            observation_lag_ranges[observation_id] = (min_lag_s, max_lag_s)

            curve_rows, peak_rows = compute_observation_xcorr(
                aligned_df,
                cfg,
                lag_max_s=lag_max_s,
            )
            all_curve_rows.extend(curve_rows)
            all_peak_rows.extend(peak_rows)
            processed_observations.append(observation_id)

            out_dir = observation_output_dir(cfg, observation_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            peaks_path = peaks_output_path(cfg, observation_id)
            pd.DataFrame(peak_rows, columns=list(PEAK_COLUMNS)).to_csv(peaks_path, index=False)
            written.append(peaks_path)

            if cfg.temporal_coupling.output.save_curves and curve_rows:
                curves_path = curves_output_path(cfg, observation_id)
                pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
                written.append(curves_path)

            n_ok += 1
            task = str(aligned_df["task"].iloc[0])
            subject_id = str(aligned_df["subject_id"].iloc[0])
            n_edge = sum(1 for peak in peak_rows if peak.get("raw_peak_at_edge"))
            print(
                f"[temporal_coupling] stage=2 {subject_id} obs={observation_id} task={task}: "
                f"pairs={len(peak_rows)} lag_range=[{min_lag_s:g},{max_lag_s:g}]s "
                f"raw_edge_peaks={n_edge} -> {peaks_path}"
            )
            for peak in peak_rows:
                edge_note = " [EDGE]" if peak.get("raw_peak_at_edge") else ""
                qc_note = ""
                if peak.get("preferred_peak_source") == "interior_peak_due_to_edge":
                    qc_note = (
                        f" QC_interior_lag={peak['preferred_peak_lag_s']:.0f}s"
                        f" QC_interior_r={peak['preferred_peak_signed_r']:.3f}"
                    )
                print(
                    f"[temporal_coupling]   {peak['pair']}: "
                    f"peak_|r|={peak['raw_peak_abs_r']:.3f} "
                    f"peak_r={peak['raw_peak_signed_r']:+.3f} "
                    f"peak_lag={peak['raw_peak_lag_s']:.0f}s "
                    f"source={peak['preferred_peak_source']}"
                    f"{edge_note}{qc_note}"
                )
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=2 skipping {observation_id}: {exc}",
                stacklevel=2,
            )

    if all_peak_rows:
        group_dir = group_output_dir(cfg)
        group_dir.mkdir(parents=True, exist_ok=True)
        group_peaks_path = group_peaks_output_path(cfg)
        pd.DataFrame(all_peak_rows, columns=list(PEAK_COLUMNS)).to_csv(group_peaks_path, index=False)
        written.append(group_peaks_path)
        print(f"[temporal_coupling] stage=2 wrote group peaks -> {group_peaks_path} rows={len(all_peak_rows)}")

        if cfg.temporal_coupling.output.save_curves and all_curve_rows:
            group_curves_path = group_curves_output_path(cfg)
            pd.DataFrame(all_curve_rows).to_csv(group_curves_path, index=False)
            written.append(group_curves_path)
            print(f"[temporal_coupling] stage=2 wrote group curves -> {group_curves_path} rows={len(all_curve_rows)}")

        validation_rows, validation_passed = validate_peaks_against_curves(
            all_curve_rows,
            all_peak_rows,
            lag_step_s=cfg.temporal_coupling.cross_correlation.lag_step_s,
            min_peak_distance_s=cfg.temporal_coupling.cross_correlation.min_peak_distance_s,
            peak_prominence=cfg.temporal_coupling.cross_correlation.peak_prominence,
            peak_height=cfg.temporal_coupling.cross_correlation.peak_height,
        )
        validation_path = group_validation_output_path(cfg)
        pd.DataFrame(validation_rows, columns=list(VALIDATION_COLUMNS)).to_csv(validation_path, index=False)
        written.append(validation_path)
        n_failed = sum(1 for row in validation_rows if not row["passed"])
        print(
            f"[temporal_coupling] stage=2 wrote peak validation -> {validation_path} "
            f"passed={len(validation_rows) - n_failed}/{len(validation_rows)}"
        )
        if not validation_passed:
            warnings.warn(
                f"[temporal_coupling] stage=2 peak validation failed for {n_failed} subject/pair rows.",
                stacklevel=2,
            )

        peaks_df = pd.DataFrame(all_peak_rows)
        qc_summary_rows = build_qc_summary(peaks_df, cfg)
        qc_summary_path = group_qc_summary_output_path(cfg)
        pd.DataFrame(qc_summary_rows, columns=list(QC_SUMMARY_COLUMNS)).to_csv(qc_summary_path, index=False)
        written.append(qc_summary_path)
        print(f"[temporal_coupling] stage=2 wrote QC summary -> {qc_summary_path}")

        plot_paths = write_qc_plots(
            cfg,
            all_curve_rows=all_curve_rows,
            all_peak_rows=all_peak_rows,
            processed_observations=processed_observations,
        )
        written.extend(plot_paths)
        if plot_paths:
            print(f"[temporal_coupling] stage=2 wrote {len(plot_paths)} QC plots")
        n_subjects = peaks_df["subject_id"].nunique()
        n_edge_total = int(peaks_df["raw_peak_at_edge"].astype(bool).sum())
        print(
            f"[temporal_coupling] stage=2 QC summary: subjects={n_subjects} "
            f"group_peak_rows={len(all_peak_rows)} edge_peaks={n_edge_total}"
        )
        for observation_id in processed_observations:
            n_rows = int((peaks_df["observation_id"] == observation_id).sum())
            n_edge = int(
                peaks_df.loc[peaks_df["observation_id"] == observation_id, "raw_peak_at_edge"].astype(bool).sum()
            )
            lag_lo, lag_hi = observation_lag_ranges.get(observation_id, (float("nan"), float("nan")))
            print(
                f"[temporal_coupling]   {observation_id}: peak_rows={n_rows} "
                f"lag_range=[{lag_lo:g},{lag_hi:g}]s edge_peaks={n_edge}"
            )
        for pair in variable_pairs():
            pair_rows = peaks_df.loc[peaks_df["pair"] == pair.pair]
            n_edge = int(pair_rows["raw_peak_at_edge"].astype(bool).sum()) if not pair_rows.empty else 0
            print(f"[temporal_coupling]   {pair.pair}: raw_edge_peaks={n_edge}/{len(pair_rows)}")

        for summary in qc_summary_rows:
            if summary["warning"]:
                print(
                    f"[temporal_coupling]   QC {summary['pair']}: {summary['warning']}"
                )

    print(f"[temporal_coupling] stage=2 summary: wrote={n_ok}/{len(observation_ids)}")
    print(f"[temporal_coupling] {SMOKE_RUN_NOTE}")
    return written
