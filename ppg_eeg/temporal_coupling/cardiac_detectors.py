"""Cardiac channel inventory, detector comparison, and R-peak / PPG peak finding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from ..ppg import detect_heartbeats, extract_ppg_signal, normalize_signal
from .cardiac_common import (
    CardiacDetectionResult,
    CardiacObservation,
    PeakAnnotation,
    annotate_peaks,
    infer_signal_type,
    list_cardiac_candidates,
    resolved_debug_windows,
    segment_bounds,
)
from .config import TemporalCouplingConfig

CHANNEL_INVENTORY_FILENAME = "cardiac_channel_inventory.csv"
DETECTOR_COMPARISON_FILENAME = "peak_detector_comparison.csv"
CHANNEL_PREVIEW_FILENAME = "cardiac_channel_preview.png"

DetectorName = Literal["simple", "ecg_rpeak", "ppg_peak"]


@dataclass(frozen=True)
class DetectorMetrics:
    n_raw_peaks: int
    n_clean_ibis: int
    median_hr: float
    min_hr: float
    max_hr: float
    ibi_cv: float
    percent_clean_ibi: float
    coverage_s: float
    end_frac: float
    quality_score: float


def _as_auto_float(value: str | float | None, signal: np.ndarray, *, percentile: float) -> float:
    if value is None or str(value).strip().casefold() == "auto":
        return float(np.percentile(signal, percentile))
    return float(value)


def _detrend(signal: np.ndarray) -> np.ndarray:
    return signal - float(np.median(signal))


def _bandpass_filter(values: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    nyq = sfreq / 2.0
    low = max(l_freq / nyq, 1e-6)
    high = min(h_freq / nyq, 0.999)
    if low >= high:
        return values.copy()
    b, a = scipy_signal.butter(4, [low, high], btype="band")
    padlen = min(3 * max(len(a), len(b)), values.size - 1)
    if padlen < 1:
        return values.copy()
    return scipy_signal.filtfilt(b, a, values, padlen=padlen)


def preprocess_ecg_for_rpeaks(
    signal: np.ndarray,
    sfreq: float,
    *,
    bandpass_hz: tuple[float, float],
) -> np.ndarray:
    detrended = _detrend(signal)
    filtered = _bandpass_filter(detrended, sfreq, bandpass_hz[0], bandpass_hz[1])
    return normalize_signal(filtered)


def preprocess_ppg_for_peaks(signal: np.ndarray) -> np.ndarray:
    return normalize_signal(_detrend(signal))


def _find_peaks(
    processed: np.ndarray,
    sfreq: float,
    *,
    min_peak_distance_s: float,
    prominence: float,
    height: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    distance = max(1, int(min_peak_distance_s * sfreq))
    kwargs: dict[str, object] = {"distance": distance, "prominence": prominence}
    if height is not None:
        kwargs["height"] = height
    peaks, _props = scipy_signal.find_peaks(processed, **kwargs)
    peak_times_s = peaks.astype(float) / sfreq
    return peak_times_s, peaks.astype(int)


def _ibi_metrics(
    peak_times_s: np.ndarray,
    *,
    duration_s: float,
    segment_start_s: float,
    sfreq: float,
    peak_samples: np.ndarray,
    signal_processed: np.ndarray,
    ibi_min_ms: float,
    ibi_max_ms: float,
) -> tuple[tuple[PeakAnnotation, ...], DetectorMetrics]:
    if peak_times_s.size == 0:
        empty = DetectorMetrics(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, -1e9)
        return tuple(), empty

    abs_times = peak_times_s + segment_start_s
    annotations = annotate_peaks(
        abs_times,
        peak_samples,
        signal_processed,
        segment_start_s=segment_start_s,
        sfreq=sfreq,
        ibi_min_ms=ibi_min_ms,
        ibi_max_ms=ibi_max_ms,
    )
    clean_ibis = [ann.ibi_ms for ann in annotations if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)]
    n_raw = int(peak_times_s.size)
    n_clean = len(clean_ibis)
    percent_clean = 100.0 * n_clean / max(1, max(0, n_raw - 1))

    if clean_ibis:
        hrs = 60000.0 / np.asarray(clean_ibis, dtype=float)
        median_hr = float(np.median(hrs))
        min_hr = float(np.min(hrs))
        max_hr = float(np.max(hrs))
        ibi_cv = float(np.std(clean_ibis) / (np.mean(clean_ibis) + 1e-12))
        coverage = float(abs_times[-1] - abs_times[0])
    else:
        median_hr = min_hr = max_hr = float("nan")
        ibi_cv = float("nan")
        coverage = 0.0

    end_frac = float(abs_times[-1] - segment_start_s) / duration_s if duration_s > 0 else 0.0
    metrics = DetectorMetrics(
        n_raw_peaks=n_raw,
        n_clean_ibis=n_clean,
        median_hr=median_hr,
        min_hr=min_hr,
        max_hr=max_hr,
        ibi_cv=ibi_cv,
        percent_clean_ibi=percent_clean,
        coverage_s=coverage,
        end_frac=end_frac,
        quality_score=_quality_score(
            median_hr=median_hr,
            max_hr=max_hr,
            ibi_cv=ibi_cv,
            percent_clean=percent_clean,
            end_frac=end_frac,
            n_clean=n_clean,
            duration_s=duration_s,
        ),
    )
    return annotations, metrics


def _quality_score(
    *,
    median_hr: float,
    max_hr: float,
    ibi_cv: float,
    percent_clean: float,
    end_frac: float,
    n_clean: int,
    duration_s: float,
) -> float:
    score = end_frac * 400.0 + n_clean * 1.5 + percent_clean * 0.5
    if not math.isnan(median_hr):
        if 50.0 <= median_hr <= 110.0:
            score += 120.0
        elif 45.0 <= median_hr <= 120.0:
            score += 60.0
        else:
            score -= 120.0
    if not math.isnan(max_hr) and max_hr > 160.0:
        score -= 100.0
    if not math.isnan(ibi_cv) and ibi_cv > 0.35:
        score -= 40.0
    if end_frac < 0.75:
        score -= 80.0
    if duration_s > 0 and n_clean < duration_s / 2.5:
        score -= 50.0
    return score


def _run_one_detector(
    *,
    detector_name: DetectorName,
    channel: str,
    ch_type: str,
    signal_raw: np.ndarray,
    inverted: bool,
    sfreq: float,
    segment_start_s: float,
    duration_s: float,
    cfg: TemporalCouplingConfig,
) -> tuple[CardiacDetectionResult | None, DetectorMetrics]:
    cardiac_cfg = cfg.temporal_coupling.cardiac
    ecg_cfg = cardiac_cfg.ecg
    oriented = -signal_raw if inverted else signal_raw
    signal_type = infer_signal_type(channel)

    prominence_val = 0.3
    height_val: float | None = None
    bandpass: tuple[float, float] | None = None
    min_dist = cfg.ppg.peak_min_distance_s

    if detector_name == "ecg_rpeak":
        bandpass = ecg_cfg.bandpass_hz
        processed = preprocess_ecg_for_rpeaks(oriented, sfreq, bandpass_hz=bandpass)
        min_dist = ecg_cfg.min_peak_distance_s
        prominence_val = _as_auto_float(ecg_cfg.prominence, processed, percentile=85)
        height_auto = _as_auto_float(ecg_cfg.height, processed, percentile=70)
        height_val = None if str(ecg_cfg.height).casefold() == "auto" else height_auto
    elif detector_name == "ppg_peak":
        processed = preprocess_ppg_for_peaks(oriented)
        min_dist = cfg.ppg.peak_min_distance_s
        prominence_val = _as_auto_float("auto", processed, percentile=80)
        height_val = _as_auto_float(cfg.ppg.peak_height, processed, percentile=65)
    else:
        processed = preprocess_ppg_for_peaks(oriented)
        min_dist = cfg.ppg.peak_min_distance_s
        height_val = cfg.ppg.peak_height
        prominence_val = 0.0
        peak_times_s, peak_idx = detect_heartbeats(
            processed,
            sampling_rate=sfreq,
            peak_min_distance_s=min_dist,
            peak_height=float(height_val),
        )
        annotations, metrics = _ibi_metrics(
            peak_times_s,
            duration_s=duration_s,
            segment_start_s=segment_start_s,
            sfreq=sfreq,
            peak_samples=peak_idx.astype(int),
            signal_processed=processed,
            ibi_min_ms=cfg.ppg.ibi_min_ms,
            ibi_max_ms=cfg.ppg.ibi_max_ms,
        )
        if peak_times_s.size < 2:
            return None, metrics
        return _build_result(
            channel=channel,
            ch_type=ch_type,
            signal_type=signal_type,
            inverted=inverted,
            detector_name=detector_name,
            bandpass=bandpass,
            prominence=prominence_val,
            height=height_val,
            oriented=oriented,
            processed=processed,
            peak_times_s=peak_times_s,
            peak_samples=peak_idx.astype(int),
            annotations=annotations,
            available=(),
            segment_start_s=segment_start_s,
            duration_s=duration_s,
            sfreq=sfreq,
            selection_reason="",
        ), metrics

    peak_times_s, peak_idx = _find_peaks(
        processed,
        sfreq,
        min_peak_distance_s=min_dist,
        prominence=prominence_val,
        height=height_val,
    )
    annotations, metrics = _ibi_metrics(
        peak_times_s,
        duration_s=duration_s,
        segment_start_s=segment_start_s,
        sfreq=sfreq,
        peak_samples=peak_idx,
        signal_processed=processed,
        ibi_min_ms=cfg.ppg.ibi_min_ms,
        ibi_max_ms=cfg.ppg.ibi_max_ms,
    )
    if peak_times_s.size < 2:
        return None, metrics

    result = _build_result(
        channel=channel,
        ch_type=ch_type,
        signal_type=signal_type,
        inverted=inverted,
        detector_name=detector_name,
        bandpass=bandpass,
        prominence=prominence_val,
        height=height_val,
        oriented=oriented,
        processed=processed,
        peak_times_s=peak_times_s,
        peak_samples=peak_idx,
        annotations=annotations,
        available=(),
        segment_start_s=segment_start_s,
        duration_s=duration_s,
        sfreq=sfreq,
        selection_reason="",
    )
    return result, metrics


def _build_result(
    *,
    channel: str,
    ch_type: str,
    signal_type: str,
    inverted: bool,
    detector_name: str,
    bandpass: tuple[float, float] | None,
    prominence: float,
    height: float | None,
    oriented: np.ndarray,
    processed: np.ndarray,
    peak_times_s: np.ndarray,
    peak_samples: np.ndarray,
    annotations: tuple[PeakAnnotation, ...],
    available: tuple[str, ...],
    segment_start_s: float,
    duration_s: float,
    sfreq: float,
    selection_reason: str,
) -> CardiacDetectionResult:
    abs_times = peak_times_s + segment_start_s
    return CardiacDetectionResult(
        available_channels=available,
        channel_used=channel,
        channel_type=ch_type,
        signal_type=signal_type,
        inverted=inverted,
        normalized=True,
        sfreq=sfreq,
        segment_start_s=segment_start_s,
        duration_s=duration_s,
        signal=oriented,
        signal_norm=processed,
        peak_times_s=abs_times,
        peak_samples=peak_samples,
        peak_annotations=annotations,
        detector_name=detector_name,
        bandpass_hz=bandpass,
        prominence=prominence,
        height_threshold=height,
        selection_reason=selection_reason,
    )


def build_channel_inventory(
    raw: mne.io.BaseRaw,
    obs: CardiacObservation,
    cfg: TemporalCouplingConfig,
) -> pd.DataFrame:
    start_s, end_s = segment_bounds(raw, cfg)
    duration_s = end_s - start_s
    sfreq = float(raw.info["sfreq"])
    rows: list[dict[str, object]] = []
    for channel in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[channel])[0]
        sig, _ = extract_ppg_signal(raw, channel, start_time_s=start_s, end_time_s=end_s)
        if sig is None or sig.size == 0:
            continue
        rows.append(
            {
                "dataset_id": obs.dataset_id,
                "subject_id": obs.subject_id,
                "task": obs.task,
                "observation_id": obs.observation_id,
                "cardiac_file": str(obs.cardiac_file),
                "channel_name": channel,
                "channel_type": ch_type,
                "sfreq": sfreq,
                "duration_s": duration_s,
                "mean": float(np.mean(sig)),
                "std": float(np.std(sig)),
                "min": float(np.min(sig)),
                "max": float(np.max(sig)),
            }
        )
    return pd.DataFrame(rows)


def plot_channel_preview(
    raw: mne.io.BaseRaw,
    obs: CardiacObservation,
    cfg: TemporalCouplingConfig,
    *,
    output_path: Path,
) -> None:
    start_s, end_s = segment_bounds(raw, cfg)
    duration_s = end_s - start_s
    sfreq = float(raw.info["sfreq"])
    windows = resolved_debug_windows(cfg)
    if windows:
        win_start, win_end = windows[0]
    else:
        win_start, win_end = 35.0, min(45.0, duration_s)
    win_start = max(0.0, min(win_start, duration_s - 1.0))
    win_end = min(duration_s, max(win_end, win_start + 1.0))

    i0 = int(win_start * sfreq)
    i1 = int(win_end * sfreq)
    times = np.arange(i0, i1) / sfreq + start_s

    n_ch = len(raw.ch_names)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2.2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for ax, channel in zip(axes, raw.ch_names, strict=True):
        sig, _ = extract_ppg_signal(raw, channel, start_time_s=start_s, end_time_s=end_s)
        if sig is None:
            continue
        seg = sig[i0:i1]
        step = max(1, int(math.ceil(seg.size / 2000)))
        ax.plot(times[::step], seg[::step], linewidth=0.5)
        ax.set_ylabel(channel, fontsize=8)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{obs.subject_id} | channel preview {win_start:.0f}-{win_end:.0f}s", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def compare_detectors(
    raw: mne.io.BaseRaw,
    obs: CardiacObservation,
    cfg: TemporalCouplingConfig,
) -> tuple[pd.DataFrame, CardiacDetectionResult]:
    cardiac_cfg = cfg.temporal_coupling.cardiac
    start_s, end_s = segment_bounds(raw, cfg)
    duration_s = end_s - start_s
    sfreq = float(raw.info["sfreq"])
    available = tuple(raw.ch_names)

    channel_pref = cardiac_cfg.channel.strip()
    if channel_pref.casefold() not in {"", "auto"}:
        lookup = {n.casefold(): n for n in raw.ch_names}
        channels = [lookup[channel_pref.casefold()]] if channel_pref.casefold() in lookup else []
    else:
        channels = list_cardiac_candidates(raw, cardiac_cfg.signal_type)

    detector_pref = cardiac_cfg.detector.strip().casefold()
    rows: list[dict[str, object]] = []
    best_result: CardiacDetectionResult | None = None
    best_score = -1e18
    best_reason = ""

    for channel in channels:
        signal_raw, _ = extract_ppg_signal(raw, channel, start_time_s=start_s, end_time_s=end_s)
        if signal_raw is None:
            continue
        ch_type = raw.get_channel_types(picks=[channel])[0]
        sig_type = infer_signal_type(channel)
        polarities = [False, True] if cardiac_cfg.ecg.test_inverted else [False]

        detectors: list[DetectorName] = []
        if detector_pref in {"", "auto"}:
            if sig_type == "ecg":
                detectors.extend(["ecg_rpeak", "simple"])
            else:
                detectors.extend(["ppg_peak", "simple"])
        elif detector_pref == "ecg_rpeak":
            detectors = ["ecg_rpeak"]
        elif detector_pref == "ppg_peak":
            detectors = ["ppg_peak"]
        else:
            detectors = ["simple"]

        for detector_name in detectors:
            if detector_name == "ecg_rpeak" and sig_type != "ecg":
                continue
            if detector_name == "ppg_peak" and sig_type != "ppg":
                continue
            for inverted in polarities:
                result, metrics = _run_one_detector(
                    detector_name=detector_name,
                    channel=channel,
                    ch_type=ch_type,
                    signal_raw=signal_raw,
                    inverted=inverted,
                    sfreq=sfreq,
                    segment_start_s=start_s,
                    duration_s=duration_s,
                    cfg=cfg,
                )
                polarity = "inverted" if inverted else "normal"
                selected = False
                reason = ""
                if metrics.quality_score > best_score and result is not None:
                    best_score = metrics.quality_score
                    best_result = result
                    best_reason = (
                        f"best quality_score={metrics.quality_score:.1f} "
                        f"detector={detector_name} channel={channel} polarity={polarity}"
                    )
                rows.append(
                    {
                        "dataset_id": obs.dataset_id,
                        "subject_id": obs.subject_id,
                        "task": obs.task,
                        "observation_id": obs.observation_id,
                        "channel_name": channel,
                        "signal_type": sig_type,
                        "detector_name": detector_name,
                        "polarity": polarity,
                        "n_raw_peaks": metrics.n_raw_peaks,
                        "n_clean_ibis": metrics.n_clean_ibis,
                        "median_hr": metrics.median_hr,
                        "min_hr": metrics.min_hr,
                        "max_hr": metrics.max_hr,
                        "ibi_cv": metrics.ibi_cv,
                        "percent_clean_ibi": metrics.percent_clean_ibi,
                        "coverage_s": metrics.coverage_s,
                        "quality_score": metrics.quality_score,
                        "selected": selected,
                        "selection_reason": "",
                    }
                )

    if best_result is None:
        raise ValueError("peak_detection_failed")

    comparison = pd.DataFrame(rows)
    if not comparison.empty:
        sel_mask = (
            (comparison["channel_name"] == best_result.channel_used)
            & (comparison["detector_name"] == best_result.detector_name)
            & (comparison["polarity"] == ("inverted" if best_result.inverted else "normal"))
        )
        comparison.loc[sel_mask, "selected"] = True
        comparison.loc[sel_mask, "selection_reason"] = best_reason

    final = CardiacDetectionResult(
        available_channels=available,
        channel_used=best_result.channel_used,
        channel_type=best_result.channel_type,
        signal_type=best_result.signal_type,
        inverted=best_result.inverted,
        normalized=best_result.normalized,
        sfreq=best_result.sfreq,
        segment_start_s=best_result.segment_start_s,
        duration_s=best_result.duration_s,
        signal=best_result.signal,
        signal_norm=best_result.signal_norm,
        peak_times_s=best_result.peak_times_s,
        peak_samples=best_result.peak_samples,
        peak_annotations=best_result.peak_annotations,
        detector_name=best_result.detector_name,
        bandpass_hz=best_result.bandpass_hz,
        prominence=best_result.prominence,
        height_threshold=best_result.height_threshold,
        selection_reason=best_reason,
    )
    return comparison, final
