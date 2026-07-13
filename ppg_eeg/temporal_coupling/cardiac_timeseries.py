"""
Stage 1b: sliding-window cardiac time series from raw ECG/PPG.

Peak detection (see cardiac_detectors.py):
- ECG: bandpass 5-30 Hz, detrend, z-score, scipy.find_peaks with prominence + distance.
- PPG: detrend, z-score, prominence-based peaks.
- Auto mode compares detectors/channels/polarities; picks best quality_score.
- IBI cleaning: ibi_min_ms/ibi_max_ms + outlier jump rejection.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core_eeg_ppg.features_core import _read_raw
from ..core_eeg_ppg.output_layout import safe_subject_dir_name
from ..core_eeg_ppg.ppg import mean_hr_bpm, mean_rr_ms, rmssd_ms, sdnn_ms
from .cardiac_common import (
    CardiacDetectionResult,
    CardiacObservation,
    PeakAnnotation,
    resolved_debug_windows,
)
from .cardiac_detectors import (
    CHANNEL_INVENTORY_FILENAME,
    CHANNEL_PREVIEW_FILENAME,
    DETECTOR_COMPARISON_FILENAME,
    build_channel_inventory,
    compare_detectors,
    plot_channel_preview,
)
from .config import TemporalCouplingConfig
from .data_audit import audit_output_path, group_output_dir
from .paths import observation_output_dir

CARDIAC_FILENAME = "features_temporal_cardiac.csv"
PEAKS_FILENAME = "detected_peaks.csv"
QC_GROUP_FILENAME = "cardiac_qc.csv"
DEBUG_PLOT_10S_USER = "cardiac_peak_detection_debug_10s_user_window.png"
DEBUG_PLOT_10S_AUTO = "cardiac_peak_detection_debug_10s_auto_window.png"
DEBUG_PLOT_40S_OVERVIEW = "cardiac_peak_detection_debug_40s_overview.png"

HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0
MEAN_RR_MIN_MS = 300.0
MEAN_RR_MAX_MS = 1500.0


@dataclass(frozen=True)
class CardiacQcRecord:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    cardiac_file: str
    available_channels: str
    channel_used: str
    signal_type: str
    detector_used: str
    detector_polarity: str
    selection_reason: str
    cardiac_duration_s: float
    sfreq: float
    n_raw_peaks: int
    n_clean_ibis: int
    first_peak_time_s: float
    last_peak_time_s: float
    clean_ibi_coverage_s: float
    percent_valid_hr: float
    percent_valid_hrv: float
    median_hr: float
    min_hr: float
    max_hr: float
    median_mean_rr: float
    median_rmssd: float
    median_sdnn: float
    usable_for_hr: bool
    usable_for_hrv: bool
    recommended_lag_hr_s: float
    recommended_lag_hrv_s: float
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "cardiac_file": self.cardiac_file,
            "available_channels": self.available_channels,
            "channel_used": self.channel_used,
            "signal_type": self.signal_type,
            "detector_used": self.detector_used,
            "detector_polarity": self.detector_polarity,
            "selection_reason": self.selection_reason,
            "cardiac_duration_s": self.cardiac_duration_s,
            "sfreq": self.sfreq,
            "n_raw_peaks": self.n_raw_peaks,
            "n_clean_ibis": self.n_clean_ibis,
            "first_peak_time_s": self.first_peak_time_s,
            "last_peak_time_s": self.last_peak_time_s,
            "clean_ibi_coverage_s": self.clean_ibi_coverage_s,
            "percent_valid_hr": self.percent_valid_hr,
            "percent_valid_hrv": self.percent_valid_hrv,
            "median_hr": self.median_hr,
            "min_hr": self.min_hr,
            "max_hr": self.max_hr,
            "median_mean_rr": self.median_mean_rr,
            "median_rmssd": self.median_rmssd,
            "median_sdnn": self.median_sdnn,
            "usable_for_hr": self.usable_for_hr,
            "usable_for_hrv": self.usable_for_hrv,
            "recommended_lag_hr_s": self.recommended_lag_hr_s,
            "recommended_lag_hrv_s": self.recommended_lag_hrv_s,
            "warning": self.warning,
        }


def cardiac_output_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / CARDIAC_FILENAME


def cardiac_qc_group_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / QC_GROUP_FILENAME


def _clean_peak_times(detection: CardiacDetectionResult) -> np.ndarray:
    if not detection.peak_annotations:
        return np.asarray([], dtype=float)
    clean = [detection.peak_annotations[0].peak_time_s]
    for ann in detection.peak_annotations[1:]:
        if ann.is_clean_ibi:
            clean.append(ann.peak_time_s)
    rel = np.asarray(clean, dtype=float) - detection.segment_start_s
    return rel


def _window_bounds(center_s: float, window_s: float, *, duration_s: float) -> tuple[float, float]:
    half = window_s / 2.0
    return max(0.0, center_s - half), min(duration_s, center_s + half)


def _beats_and_ibis_in_window(
    peak_times_s: np.ndarray,
    *,
    window_start: float,
    window_end: float,
    ibi_min_ms: float,
    ibi_max_ms: float,
) -> tuple[np.ndarray, int]:
    # peak_times_s is monotonic; index slicing avoids full-array boolean masks per window.
    lo = int(np.searchsorted(peak_times_s, window_start, side="left"))
    hi = int(np.searchsorted(peak_times_s, window_end, side="right"))
    peak_times = peak_times_s[lo:hi]
    n_beats = int(len(peak_times))
    if n_beats < 2:
        return np.array([], dtype=float), n_beats

    ibi_ms = np.diff(peak_times) * 1000.0
    valid = ibi_ms[(ibi_ms >= ibi_min_ms) & (ibi_ms <= ibi_max_ms)]
    return valid, n_beats


def _quality_flag(
    *,
    hr: float,
    mean_rr: float,
    has_hr: bool,
    has_hrv: bool,
    has_clean_ibi: bool,
) -> str:
    flags: list[str] = []
    if not has_clean_ibi:
        flags.append("no_clean_ibi")
    if not has_hr:
        flags.append("insufficient_hr_beats")
    if not has_hrv:
        flags.append("insufficient_hrv_beats")
    if has_hr and (hr < HR_MIN_BPM or hr > HR_MAX_BPM or mean_rr < MEAN_RR_MIN_MS or mean_rr > MEAN_RR_MAX_MS):
        flags.append("outside_valid_range")
    if not flags:
        return "ok"
    return ";".join(flags)


def compute_cardiac_timeseries(detection: CardiacDetectionResult, cfg: TemporalCouplingConfig) -> pd.DataFrame:
    cardiac_cfg = cfg.temporal_coupling.cardiac
    hr_window_s = cardiac_cfg.hr_window_s
    mean_rr_window_s = cardiac_cfg.mean_rr_window_s or hr_window_s
    hrv_window_s = cardiac_cfg.hrv_window_s
    step_s = cardiac_cfg.hrv_step_s
    min_beats_hr = cardiac_cfg.min_beats_hr
    min_beats_hrv = cardiac_cfg.min_beats_hrv

    peak_times = _clean_peak_times(detection)
    if peak_times.size < 2:
        return pd.DataFrame()

    # Grid endpoints use HR/RR half-windows so overlap is not over-trimmed by HRV window size.
    half_grid = max(hr_window_s, mean_rr_window_s) / 2.0
    start_t = half_grid
    end_t = max(start_t, detection.duration_s - half_grid)
    if end_t < start_t:
        return pd.DataFrame()

    time_points = np.arange(start_t, end_t + step_s * 0.5, step_s)
    rows: list[dict[str, object]] = []

    for time_s in time_points:
        hr_start, hr_end = _window_bounds(time_s, hr_window_s, duration_s=detection.duration_s)
        rr_start, rr_end = _window_bounds(time_s, mean_rr_window_s, duration_s=detection.duration_s)
        hrv_start, hrv_end = _window_bounds(time_s, hrv_window_s, duration_s=detection.duration_s)

        ibi_hr, n_hr_beats = _beats_and_ibis_in_window(
            peak_times,
            window_start=hr_start,
            window_end=hr_end,
            ibi_min_ms=cfg.ppg.ibi_min_ms,
            ibi_max_ms=cfg.ppg.ibi_max_ms,
        )
        ibi_rr, n_rr_beats = _beats_and_ibis_in_window(
            peak_times,
            window_start=rr_start,
            window_end=rr_end,
            ibi_min_ms=cfg.ppg.ibi_min_ms,
            ibi_max_ms=cfg.ppg.ibi_max_ms,
        )
        ibi_hrv, n_hrv_beats = _beats_and_ibis_in_window(
            peak_times,
            window_start=hrv_start,
            window_end=hrv_end,
            ibi_min_ms=cfg.ppg.ibi_min_ms,
            ibi_max_ms=cfg.ppg.ibi_max_ms,
        )

        has_clean_ibi = ibi_hr.size > 0 or ibi_hrv.size > 0
        has_hr = n_hr_beats >= min_beats_hr and ibi_hr.size >= min_beats_hr - 1
        has_rr = n_rr_beats >= min_beats_hr and ibi_rr.size >= min_beats_hr - 1
        has_hrv = n_hrv_beats >= min_beats_hrv and ibi_hrv.size >= min_beats_hrv - 1

        hr = mean_hr_bpm(ibi_hr) if has_hr else float("nan")
        mean_rr = mean_rr_ms(ibi_rr) if has_rr else float("nan")
        rmssd = rmssd_ms(ibi_hrv) if has_hrv and ibi_hrv.size >= 2 else float("nan")
        sdnn = sdnn_ms(ibi_hrv) if has_hrv and ibi_hrv.size >= 2 else float("nan")

        rows.append(
            {
                "time_s": float(time_s),
                "hr": hr,
                "rmssd": rmssd,
                "sdnn": sdnn,
                "mean_rr": mean_rr,
                "n_beats_hr_window": int(n_hr_beats),
                "n_beats_hrv_window": int(n_hrv_beats),
                "quality_flag": _quality_flag(
                    hr=hr,
                    mean_rr=mean_rr,
                    has_hr=has_hr,
                    has_hrv=has_hrv,
                    has_clean_ibi=has_clean_ibi,
                ),
            }
        )

    return pd.DataFrame(rows)


def peaks_to_dataframe(obs: CardiacObservation, detection: CardiacDetectionResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ann in detection.peak_annotations:
        rows.append(
            {
                "dataset_id": obs.dataset_id,
                "subject_id": obs.subject_id,
                "task": obs.task,
                "observation_id": obs.observation_id,
                "cardiac_file": str(obs.cardiac_file),
                "channel_used": detection.channel_used,
                "signal_type": detection.signal_type,
                "peak_time_s": ann.peak_time_s,
                "peak_sample": ann.peak_sample,
                "peak_y": ann.peak_y,
                "ibi_ms": ann.ibi_ms,
                "is_clean_ibi": ann.is_clean_ibi,
                "is_accepted_peak": ann.is_accepted_peak,
                "cleaning_reason": ann.cleaning_reason,
            }
        )
    return pd.DataFrame(rows)


def _recommended_lag(coverage_s: float, lag_max_s: float) -> float:
    if coverage_s <= 0:
        return 0.0
    return float(min(lag_max_s, math.floor(coverage_s / 3.0)))


def build_cardiac_qc(
    obs: CardiacObservation,
    detection: CardiacDetectionResult,
    cardiac_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
) -> CardiacQcRecord:
    cardiac_cfg = cfg.temporal_coupling.cardiac
    lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s

    clean_times = [
        ann.peak_time_s
        for ann in detection.peak_annotations
        if ann.is_clean_ibi and ann.cleaning_reason != "first_peak"
    ]
    n_clean_ibis = sum(
        1 for ann in detection.peak_annotations if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)
    )

    if clean_times:
        first_peak = float(min(clean_times))
        last_peak = float(max(clean_times))
        clean_coverage = last_peak - first_peak
    elif detection.peak_times_s.size:
        first_peak = float(detection.peak_times_s[0])
        last_peak = float(detection.peak_times_s[-1])
        clean_coverage = last_peak - first_peak
    else:
        first_peak = float("nan")
        last_peak = float("nan")
        clean_coverage = 0.0

    n_rows = len(cardiac_df)
    hr_valid = cardiac_df["hr"].notna() if n_rows else pd.Series(dtype=bool)
    hrv_valid = cardiac_df["rmssd"].notna() if n_rows else pd.Series(dtype=bool)
    percent_valid_hr = float(100.0 * hr_valid.mean()) if n_rows else 0.0
    percent_valid_hrv = float(100.0 * hrv_valid.mean()) if n_rows else 0.0

    hr_series = cardiac_df["hr"].dropna()
    rr_series = cardiac_df["mean_rr"].dropna()
    rmssd_series = cardiac_df["rmssd"].dropna()
    sdnn_series = cardiac_df["sdnn"].dropna()

    peak_ibis = [
        ann.ibi_ms
        for ann in detection.peak_annotations
        if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)
    ]
    peak_median_hr = float(60000.0 / np.median(peak_ibis)) if peak_ibis else float("nan")
    peak_max_hr = float(60000.0 / np.min(peak_ibis)) if peak_ibis else float("nan")
    n_possible_ibis = max(0, int(detection.peak_times_s.size) - 1)
    percent_clean_ibi = 100.0 * n_clean_ibis / max(1, n_possible_ibis)

    warnings_out: list[str] = []
    if detection.peak_times_s.size == 0:
        warnings_out.append("no_peaks_detected")
    if clean_coverage < 0.7 * detection.duration_s:
        warnings_out.append("warning: poor_peak_coverage")
    if percent_clean_ibi < 70.0:
        warnings_out.append("warning: low_clean_ibi_percent")
    if percent_valid_hr < cardiac_cfg.min_valid_hr_percent:
        warnings_out.append("low_percent_valid_hr")
    if percent_valid_hrv < cardiac_cfg.min_valid_hrv_percent:
        warnings_out.append("low_percent_valid_hrv")
    if not math.isnan(peak_median_hr) and peak_median_hr > 120.0:
        warnings_out.append("warning: median_hr_high")
    if not hr_series.empty and hr_series.median() > 120.0:
        warnings_out.append("warning: median_hr_high")
    if not math.isnan(peak_max_hr) and peak_max_hr > 160.0:
        warnings_out.append("warning: max_hr_high")
    if not hr_series.empty and hr_series.max() > 160:
        warnings_out.append("warning: max_hr_high")
    if detection.channel_used and detection.signal_type == "unknown":
        warnings_out.append("warning: uncertain_channel_selection")

    polarity = "inverted" if detection.inverted else "normal"
    hr_plausible = (
        not math.isnan(peak_median_hr)
        and 45.0 <= peak_median_hr <= 120.0
        and (math.isnan(peak_max_hr) or peak_max_hr <= 160.0)
    )
    coverage_ok = clean_coverage >= 0.7 * detection.duration_s
    ibi_ok = percent_clean_ibi >= 70.0

    usable_for_hr = (
        n_clean_ibis >= cardiac_cfg.min_beats_hr
        and percent_valid_hr >= cardiac_cfg.min_valid_hr_percent
        and hr_plausible
        and coverage_ok
        and ibi_ok
    )
    usable_for_hrv = (
        n_clean_ibis >= cardiac_cfg.min_beats_hrv
        and percent_valid_hrv >= cardiac_cfg.min_valid_hrv_percent
        and hr_plausible
        and coverage_ok
        and ibi_ok
    )

    hr_coverage = clean_coverage if clean_coverage > 0 else detection.duration_s * percent_valid_hr / 100.0
    hrv_coverage = clean_coverage if clean_coverage > 0 else detection.duration_s * percent_valid_hrv / 100.0

    return CardiacQcRecord(
        dataset_id=obs.dataset_id,
        subject_id=obs.subject_id,
        task=obs.task,
        condition=obs.condition,
        observation_id=obs.observation_id,
        cardiac_file=str(obs.cardiac_file),
        available_channels=",".join(detection.available_channels),
        channel_used=detection.channel_used,
        signal_type=detection.signal_type,
        detector_used=detection.detector_name,
        detector_polarity=polarity,
        selection_reason=detection.selection_reason,
        cardiac_duration_s=detection.duration_s,
        sfreq=detection.sfreq,
        n_raw_peaks=int(detection.peak_times_s.size),
        n_clean_ibis=n_clean_ibis,
        first_peak_time_s=first_peak,
        last_peak_time_s=last_peak,
        clean_ibi_coverage_s=clean_coverage,
        percent_valid_hr=percent_valid_hr,
        percent_valid_hrv=percent_valid_hrv,
        median_hr=float(hr_series.median()) if not hr_series.empty else float("nan"),
        min_hr=float(hr_series.min()) if not hr_series.empty else float("nan"),
        max_hr=float(hr_series.max()) if not hr_series.empty else float("nan"),
        median_mean_rr=float(rr_series.median()) if not rr_series.empty else float("nan"),
        median_rmssd=float(rmssd_series.median()) if not rmssd_series.empty else float("nan"),
        median_sdnn=float(sdnn_series.median()) if not sdnn_series.empty else float("nan"),
        usable_for_hr=usable_for_hr,
        usable_for_hrv=usable_for_hrv,
        recommended_lag_hr_s=_recommended_lag(hr_coverage, lag_max_s),
        recommended_lag_hrv_s=_recommended_lag(hrv_coverage, lag_max_s),
        warning=";".join(warnings_out),
    )


def _auto_plot_window(peak_times_s: np.ndarray, *, duration_s: float, window_s: float) -> tuple[float, float]:
    if duration_s <= window_s:
        return 0.0, duration_s
    if peak_times_s.size == 0:
        return 0.0, min(window_s, duration_s)

    best_start = 0.0
    best_count = -1
    for start in np.arange(0.0, duration_s - window_s, 5.0):
        end = start + window_s
        count = int(np.sum((peak_times_s >= start) & (peak_times_s <= end)))
        if count > best_count:
            best_count = count
            best_start = float(start)
    return best_start, min(best_start + window_s, duration_s)


def _downsample_for_plot(
    times: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int = 2500,
) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_points:
        return times, values
    step = int(math.ceil(values.size / max_points))
    return times[::step], values[::step]


def _window_peak_annotations(
    detection: CardiacDetectionResult,
    *,
    rel_start: float,
    rel_end: float,
) -> list[PeakAnnotation]:
    return [
        ann
        for ann in detection.peak_annotations
        if rel_start <= (ann.peak_time_s - detection.segment_start_s) <= rel_end
    ]


def _window_stats(annotations: list[PeakAnnotation]) -> dict[str, object]:
    ibis = [ann.ibi_ms for ann in annotations if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)]
    n_peaks = len(annotations)
    median_ibi = float(np.median(ibis)) if ibis else float("nan")
    est_hr = float(60000.0 / median_ibi) if ibis else float("nan")
    return {
        "n_peaks": n_peaks,
        "median_ibi_ms": median_ibi,
        "estimated_hr_bpm": est_hr,
    }


def _detector_min_distance_s(detection: CardiacDetectionResult, cfg: TemporalCouplingConfig) -> float:
    if detection.detector_name == "ecg_rpeak":
        return cfg.temporal_coupling.cardiac.ecg.min_peak_distance_s
    return cfg.ppg.peak_min_distance_s


def _plot_title_warnings(
    stats: dict[str, object],
    *,
    qc_warnings: str,
) -> str:
    parts: list[str] = []
    est_hr = stats.get("estimated_hr_bpm", float("nan"))
    if isinstance(est_hr, (int, float)) and not math.isnan(float(est_hr)) and float(est_hr) > 120.0:
        parts.append("median_hr_high")
    for token in ("median_hr_high", "low_clean_ibi_percent", "poor_peak_coverage", "uncertain_channel_selection"):
        if token in qc_warnings:
            parts.append(token)
    if not parts:
        return ""
    return " | WARNING: " + ", ".join(parts)


def _plot_peak_debug_qc(
    obs: CardiacObservation,
    detection: CardiacDetectionResult,
    cfg: TemporalCouplingConfig,
    *,
    start_s: float,
    end_s: float,
    output_path: Path,
    qc_warnings: str = "",
) -> None:
    sfreq = detection.sfreq
    seg_start = detection.segment_start_s
    rel_start = max(0.0, start_s)
    rel_end = min(detection.duration_s, end_s)
    if rel_end <= rel_start:
        rel_end = min(detection.duration_s, rel_start + 1.0)

    i0 = int(rel_start * sfreq)
    i1 = max(i0 + 1, int(rel_end * sfreq))
    times = (np.arange(i0, i1) / sfreq) + seg_start

    raw_segment = detection.signal[i0:i1]
    processed_segment = detection.signal_norm[i0:i1]
    plot_times_raw, plot_raw = _downsample_for_plot(times, raw_segment)
    plot_times_proc, plot_proc = _downsample_for_plot(times, processed_segment)

    window_ann = _window_peak_annotations(detection, rel_start=rel_start, rel_end=rel_end)
    stats = _window_stats(window_ann)
    min_dist_s = _detector_min_distance_s(detection, cfg)

    accepted = [ann for ann in window_ann if ann.is_accepted_peak]
    rejected = [ann for ann in window_ann if not ann.is_accepted_peak]

    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    polarity = "inverted" if detection.inverted else "normal"
    signal_label = detection.signal_type.upper()
    title_warn = _plot_title_warnings(stats, qc_warnings=qc_warnings)
    fig.suptitle(
        f"{obs.subject_id} | {obs.task} | {signal_label} peak detection{title_warn}",
        fontsize=12,
        y=0.98,
        color="darkred" if title_warn else "black",
    )

    ax_raw.plot(plot_times_raw, plot_raw, color="tab:blue", linewidth=0.5, alpha=0.9, label="raw signal")
    ax_raw.set_ylabel("raw amplitude")
    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper right", fontsize=8)

    proc_label = "filtered/processed (detection signal)"
    ax_proc.plot(
        plot_times_proc,
        plot_proc,
        color="tab:blue",
        linewidth=0.5,
        alpha=0.9,
        label=proc_label,
    )
    if detection.detector_name == "simple" and detection.height_threshold is not None:
        ax_proc.axhline(
            detection.height_threshold,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"height threshold ({detection.height_threshold:g})",
        )

    if accepted:
        ax_proc.scatter(
            [ann.peak_time_s for ann in accepted],
            [ann.peak_y for ann in accepted],
            s=55,
            c="red",
            marker="o",
            edgecolors="darkred",
            linewidths=0.6,
            zorder=5,
            label=f"accepted peaks (n={len(accepted)})",
        )
    if rejected:
        ax_proc.scatter(
            [ann.peak_time_s for ann in rejected],
            [ann.peak_y for ann in rejected],
            s=55,
            c="darkorange",
            marker="x",
            linewidths=1.2,
            zorder=5,
            label=f"rejected peaks (n={len(rejected)})",
        )
    if not window_ann:
        ax_proc.text(
            0.5,
            0.5,
            "No peaks in window",
            transform=ax_proc.transAxes,
            ha="center",
            va="center",
            color="red",
        )

    ax_proc.set_xlabel("time (s)")
    ax_proc.set_ylabel("processed amplitude")
    ax_proc.grid(True, alpha=0.3)
    ax_proc.legend(loc="upper right", fontsize=8)

    bandpass_txt = (
        f"{detection.bandpass_hz[0]:.0f}-{detection.bandpass_hz[1]:.0f} Hz"
        if detection.bandpass_hz
        else "none"
    )
    threshold_txt = (
        f"prominence={detection.prominence:.2f}"
        if detection.detector_name != "simple"
        else f"height={detection.height_threshold or cfg.ppg.peak_height:.2f}"
    )
    info = (
        f"window: {rel_start:.1f}-{rel_end:.1f} s\n"
        f"Channel: {detection.channel_used} ({detection.signal_type.upper()})\n"
        f"Detector: {detection.detector_name} | polarity: {polarity}\n"
        f"Bandpass: {bandpass_txt}\n"
        f"Accepted peaks in window: {len(accepted)}\n"
        f"Median IBI: {stats['median_ibi_ms']:.0f} ms\n"
        f"Estimated HR: {stats['estimated_hr_bpm']:.0f} bpm\n"
        f"Min peak distance: {min_dist_s:.2f} s\n"
        f"Rule: {threshold_txt}"
    )
    ax_proc.text(
        0.01,
        0.98,
        info,
        transform=ax_proc.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_debug_plots(
    obs: CardiacObservation,
    detection: CardiacDetectionResult,
    cfg: TemporalCouplingConfig,
    out_dir: Path,
    *,
    qc_warnings: str = "",
) -> list[Path]:
    plot_cfg = cfg.temporal_coupling.cardiac.debug_plot
    if not plot_cfg.enabled:
        return []

    written: list[Path] = []
    user_windows = resolved_debug_windows(cfg)

    if user_windows:
        start_s, end_s = user_windows[0]
        path = out_dir / DEBUG_PLOT_10S_USER
        _plot_peak_debug_qc(
            obs, detection, cfg, start_s=start_s, end_s=end_s, output_path=path, qc_warnings=qc_warnings
        )
        written.append(path)

    if plot_cfg.also_auto_window:
        auto_start, auto_end = _auto_plot_window(
            detection.peak_times_s - detection.segment_start_s,
            duration_s=detection.duration_s,
            window_s=plot_cfg.auto_window_s,
        )
        path = out_dir / DEBUG_PLOT_10S_AUTO
        _plot_peak_debug_qc(
            obs, detection, cfg, start_s=auto_start, end_s=auto_end, output_path=path, qc_warnings=qc_warnings
        )
        written.append(path)

    if plot_cfg.save_overview and plot_cfg.overview_window_s:
        overview_s = plot_cfg.overview_window_s
        if detection.duration_s <= overview_s:
            ov_start, ov_end = 0.0, detection.duration_s
        else:
            ov_start, ov_end = _auto_plot_window(
                detection.peak_times_s - detection.segment_start_s,
                duration_s=detection.duration_s,
                window_s=overview_s,
            )
        path = out_dir / DEBUG_PLOT_40S_OVERVIEW
        _plot_peak_debug_qc(
            obs, detection, cfg, start_s=ov_start, end_s=ov_end, output_path=path, qc_warnings=qc_warnings
        )
        written.append(path)

    return written[: plot_cfg.max_plots_per_subject]


def load_usable_cardiac_observations(cfg: TemporalCouplingConfig) -> list[CardiacObservation]:
    audit_path = audit_output_path(cfg)
    if audit_path.is_file():
        audit_df = pd.read_csv(audit_path)
        if audit_df.empty:
            return []
        usable_mask = audit_df["usable"].astype(str).str.lower().isin({"true", "1", "yes"})
        rows = audit_df.loc[usable_mask]
        observations: list[CardiacObservation] = []
        for row in rows.itertuples(index=False):
            cardiac_file = Path(str(row.cardiac_file))
            if not cardiac_file.is_file():
                warnings.warn(
                    f"[temporal_coupling] skipping {row.subject_id}: audit cardiac_file missing ({cardiac_file}).",
                    stacklevel=2,
                )
                continue
            observations.append(
                CardiacObservation(
                    dataset_id=str(row.dataset_id),
                    subject_id=str(row.subject_id),
                    task=str(row.task),
                    condition=str(getattr(row, "condition", row.task)),
                    observation_id=str(row.observation_id),
                    cardiac_file=cardiac_file,
                    cardiac_format=str(getattr(row, "cardiac_format", getattr(row, "eeg_format", "eeglab"))),
                )
            )
        return observations

    warnings.warn(
        f"[temporal_coupling] {audit_path} not found; using configured observations. "
        "Run --stage 0 first for audit gating.",
        stacklevel=2,
    )
    from .data_audit import list_configured_observations

    observations = []
    for obs in list_configured_observations(cfg):
        cardiac_path = obs.eeg_path if obs.ppg_source == "embedded_eeg" else obs.ppg_path
        cardiac_format = obs.eeg_format if obs.ppg_source == "embedded_eeg" else (obs.ppg_format or "eeglab")
        if cardiac_path is None or not cardiac_path.is_file():
            warnings.warn(
                f"[temporal_coupling] skipping {obs.subject_id} condition={obs.condition_label}: cardiac file missing.",
                stacklevel=2,
            )
            continue
        observations.append(
            CardiacObservation(
                dataset_id=obs.dataset_id,
                subject_id=obs.subject_id,
                task=obs.task_label,
                condition=obs.condition_label,
                observation_id=obs.observation_id,
                cardiac_file=cardiac_path,
                cardiac_format=cardiac_format,
            )
        )
    return observations


def _print_detection_log(obs: CardiacObservation, detection: CardiacDetectionResult) -> None:
    invert_txt = "inverted" if detection.inverted else "normal"
    print(
        f"[temporal_coupling] stage=1b {obs.subject_id}: "
        f"available_channels={list(detection.available_channels)} "
        f"selected={detection.channel_used} type={detection.channel_type} "
        f"signal_type={detection.signal_type} detector={detection.detector_name} polarity={invert_txt}"
    )
    print(
        f"[temporal_coupling]   selection={detection.selection_reason} "
        f"n_raw_peaks={detection.peak_times_s.size}"
    )


def _print_qc_summary(qc: CardiacQcRecord) -> None:
    print(
        f"[temporal_coupling]   QC {qc.subject_id}: channel={qc.channel_used} "
        f"detector={qc.detector_used} polarity={qc.detector_polarity} "
        f"raw_peaks={qc.n_raw_peaks} clean_ibis={qc.n_clean_ibis} "
        f"peaks={qc.first_peak_time_s:.1f}-{qc.last_peak_time_s:.1f}s "
        f"valid_hr={qc.percent_valid_hr:.1f}% valid_hrv={qc.percent_valid_hrv:.1f}% "
        f"median_hr={qc.median_hr:.1f} hr_range=[{qc.min_hr:.1f},{qc.max_hr:.1f}] "
        f"lag_hr={qc.recommended_lag_hr_s:.0f}s lag_hrv={qc.recommended_lag_hrv_s:.0f}s "
        f"usable_hr={qc.usable_for_hr} usable_hrv={qc.usable_for_hrv}"
    )
    if qc.warning:
        warnings.warn(f"[temporal_coupling]   {qc.subject_id} QC warning: {qc.warning}", stacklevel=2)


def run_stage1b(cfg: TemporalCouplingConfig) -> list[Path]:
    observations = load_usable_cardiac_observations(cfg)
    if not observations:
        print("[temporal_coupling] stage=1b: no usable observations to process.")
        return []

    written: list[Path] = []
    qc_records: list[CardiacQcRecord] = []
    n_ok = 0

    for obs in observations:
        out_dir = observation_output_dir(cfg, obs.observation_id)
        out_path = out_dir / CARDIAC_FILENAME
        try:
            raw = _read_raw(obs.cardiac_file, obs.cardiac_format)
            out_dir.mkdir(parents=True, exist_ok=True)

            inventory_df = build_channel_inventory(raw, obs, cfg)
            inventory_df.to_csv(out_dir / CHANNEL_INVENTORY_FILENAME, index=False)
            plot_channel_preview(raw, obs, cfg, output_path=out_dir / CHANNEL_PREVIEW_FILENAME)

            comparison_df, detection = compare_detectors(raw, obs, cfg)
            comparison_df.to_csv(out_dir / DETECTOR_COMPARISON_FILENAME, index=False)
            _print_detection_log(obs, detection)

            cardiac_df = compute_cardiac_timeseries(detection, cfg)
            if cardiac_df.empty:
                raise ValueError("no_cardiac_windows")

            peaks_df = peaks_to_dataframe(obs, detection)
            qc = build_cardiac_qc(obs, detection, cardiac_df, cfg)
            qc_records.append(qc)
            _print_qc_summary(qc)

            out_df = cardiac_df.copy()
            out_df.insert(0, "usable_for_hrv", qc.usable_for_hrv)
            out_df.insert(0, "usable_for_hr", qc.usable_for_hr)
            out_df.insert(0, "observation_id", obs.observation_id)
            out_df.insert(0, "task", obs.task)
            out_df.insert(0, "condition", obs.condition)
            out_df.insert(0, "subject_id", obs.subject_id)
            out_df.insert(0, "dataset_id", obs.dataset_id)

            out_df.to_csv(out_path, index=False)
            peaks_df.to_csv(out_dir / PEAKS_FILENAME, index=False)
            plot_paths = write_debug_plots(obs, detection, cfg, out_dir, qc_warnings=qc.warning)

            written.append(out_path)
            n_ok += 1
            print(
                f"[temporal_coupling] stage=1b {obs.subject_id} task={obs.task}: "
                f"rows={len(out_df)} -> {out_path}"
            )
            for plot_path in plot_paths:
                print(f"[temporal_coupling]   debug_plot={plot_path}")
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=1b skipping {obs.subject_id} task={obs.task}: {exc}",
                stacklevel=2,
            )

    if qc_records:
        qc_path = cardiac_qc_group_path(cfg)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record.to_row() for record in qc_records]).to_csv(qc_path, index=False)
        print(f"[temporal_coupling] stage=1b wrote group QC -> {qc_path}")

    print(f"[temporal_coupling] stage=1b summary: wrote={n_ok}/{len(observations)}")
    return written
