"""Shared cardiac datatypes and helpers for Stage 1b."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import numpy as np

from .config import TemporalCouplingConfig, TemporalCouplingTaskTimeWindowConfig


@dataclass(frozen=True)
class CardiacObservation:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    cardiac_file: Path
    cardiac_format: str


@dataclass(frozen=True)
class PeakAnnotation:
    peak_time_s: float
    peak_sample: int
    peak_y: float
    ibi_ms: float
    is_clean_ibi: bool
    is_accepted_peak: bool
    cleaning_reason: str


@dataclass(frozen=True)
class CardiacDetectionResult:
    available_channels: tuple[str, ...]
    channel_used: str
    channel_type: str
    signal_type: str
    inverted: bool
    normalized: bool
    sfreq: float
    segment_start_s: float
    duration_s: float
    signal: np.ndarray
    signal_norm: np.ndarray
    peak_times_s: np.ndarray
    peak_samples: np.ndarray
    peak_annotations: tuple[PeakAnnotation, ...]
    detector_name: str
    bandpass_hz: tuple[float, float] | None
    prominence: float
    height_threshold: float | None
    selection_reason: str


def infer_signal_type(channel_name: str) -> str:
    low = channel_name.casefold()
    if "ecg" in low or "ekg" in low:
        return "ecg"
    if any(token in low for token in ("ppg", "photo", "optic", "pleth", "pulse", "oxi")):
        return "ppg"
    return "unknown"


def infer_signal_type_from_channel(channel_name: str, ch_type: str) -> str:
    inferred = infer_signal_type(channel_name)
    if inferred != "unknown":
        return inferred
    ctype = str(ch_type).strip().casefold()
    if ctype == "ecg":
        return "ecg"
    if ctype in {"ppg", "pleth"}:
        return "ppg"
    if ctype == "bio" and any(token in channel_name.casefold() for token in ("oxi", "ppg", "pleth", "pulse")):
        return "ppg"
    return "unknown"


def resolve_signal_type(channel_name: str, signal_type_pref: str, *, ch_type: str = "") -> str:
    pref = str(signal_type_pref).strip().casefold()
    if pref in {"ecg", "ppg"}:
        return pref
    return infer_signal_type_from_channel(channel_name, ch_type)


def list_cardiac_candidates(raw: mne.io.BaseRaw, signal_type: str) -> list[str]:
    st = signal_type.strip().casefold()
    candidates: list[str] = []
    for name in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[name])[0]
        inferred = infer_signal_type_from_channel(name, ch_type)
        if st in {"", "auto"}:
            if inferred in {"ecg", "ppg"}:
                candidates.append(name)
        elif st == "ecg" and inferred == "ecg":
            candidates.append(name)
        elif st == "ppg" and inferred == "ppg":
            candidates.append(name)
    # Dedicated single-channel cardiac recordings (e.g. paired ECG .set files).
    if not candidates and len(raw.ch_names) == 1 and st in {"", "auto", "ecg", "ppg"}:
        candidates = [raw.ch_names[0]]
    return candidates


def has_cardiac_channels(
    ch_names: list[str],
    ch_types: list[str],
    *,
    signal_type: str = "auto",
) -> bool:
    if not ch_names:
        return False
    if len(ch_names) == 1:
        return True
    st = signal_type.strip().casefold()
    for name, ch_type in zip(ch_names, ch_types, strict=False):
        inferred = infer_signal_type_from_channel(name, ch_type)
        if st in {"", "auto"}:
            if inferred in {"ecg", "ppg"}:
                return True
        elif st == "ecg" and inferred == "ecg":
            return True
        elif st == "ppg" and inferred == "ppg":
            return True
    return False


def _task_time_window(cfg: TemporalCouplingConfig, task: str | None) -> TemporalCouplingTaskTimeWindowConfig | None:
    if not task:
        return None
    return cfg.temporal_coupling.task_time_windows.get(task.casefold())


def _signal_window_bounds(
    cfg: TemporalCouplingConfig,
    *,
    task: str | None,
    signal: Literal["eeg", "cardiac"],
) -> tuple[float | None, float | None]:
    task_window = _task_time_window(cfg, task)
    if task_window is not None:
        if signal == "eeg":
            return task_window.eeg_start_time_s, task_window.eeg_end_time_s
        return task_window.cardiac_start_time_s, task_window.cardiac_end_time_s
    if signal == "cardiac":
        return cfg.ppg.start_time_s, cfg.ppg.end_time_s
    return None, None


def _window_start_end(
    duration_s: float,
    *,
    start_time_s: float | None,
    end_time_s: float | None,
) -> tuple[float, float]:
    if duration_s <= 0:
        return 0.0, 0.0
    if start_time_s is None and end_time_s is None:
        return 0.0, duration_s

    window_start = 0.0 if start_time_s is None else max(0.0, float(start_time_s))
    window_end = duration_s if end_time_s is None else min(float(end_time_s), duration_s)
    if window_end <= window_start:
        return window_start, window_start
    return window_start, window_end


def overlap_duration_s(
    cfg: TemporalCouplingConfig,
    *,
    task: str | None,
    eeg_duration_s: float,
    cardiac_duration_s: float,
) -> float:
    eeg_start_s, eeg_end_s = _signal_window_bounds(cfg, task=task, signal="eeg")
    cardiac_start_s, cardiac_end_s = _signal_window_bounds(cfg, task=task, signal="cardiac")
    eeg_start, eeg_end = _window_start_end(
        eeg_duration_s,
        start_time_s=eeg_start_s,
        end_time_s=eeg_end_s,
    )
    cardiac_start, cardiac_end = _window_start_end(
        cardiac_duration_s,
        start_time_s=cardiac_start_s,
        end_time_s=cardiac_end_s,
    )
    overlap_start = max(eeg_start, cardiac_start)
    overlap_end = min(eeg_end, cardiac_end)
    return max(0.0, overlap_end - overlap_start)


def segment_bounds(
    raw: mne.io.BaseRaw,
    cfg: TemporalCouplingConfig,
    *,
    task: str | None = None,
) -> tuple[float, float]:
    sfreq = float(raw.info["sfreq"])
    duration_s = max(0.0, float(raw.n_times) / sfreq)
    start_time_s, end_time_s = _signal_window_bounds(cfg, task=task, signal="cardiac")
    return _window_start_end(duration_s, start_time_s=start_time_s, end_time_s=end_time_s)


def eeg_segment_bounds(
    raw: mne.io.BaseRaw,
    cfg: TemporalCouplingConfig,
    *,
    task: str | None = None,
) -> tuple[float, float]:
    sfreq = float(raw.info["sfreq"])
    duration_s = max(0.0, float(raw.n_times) / sfreq)
    start_time_s, end_time_s = _signal_window_bounds(cfg, task=task, signal="eeg")
    return _window_start_end(duration_s, start_time_s=start_time_s, end_time_s=end_time_s)


def classify_ibi(
    ibi_ms: float,
    *,
    prev_ibi_ms: float | None,
    median_jump: float,
    ibi_min_ms: float,
    ibi_max_ms: float,
) -> tuple[bool, str]:
    if ibi_ms < ibi_min_ms:
        return False, "too_short"
    if ibi_ms > ibi_max_ms:
        return False, "too_long"
    if prev_ibi_ms is not None and median_jump > 0:
        if abs(ibi_ms - prev_ibi_ms) >= 3.0 * median_jump:
            return False, "outlier"
    return True, "ok"


def annotate_peaks(
    peak_times_s: np.ndarray,
    peak_samples: np.ndarray,
    signal_norm: np.ndarray,
    *,
    segment_start_s: float,
    sfreq: float,
    ibi_min_ms: float,
    ibi_max_ms: float,
) -> tuple[PeakAnnotation, ...]:
    annotations: list[PeakAnnotation] = []
    if peak_times_s.size == 0:
        return tuple()

    raw_ibis = np.diff(peak_times_s) * 1000.0 if peak_times_s.size > 1 else np.array([])
    median_jump = float(np.median(np.abs(np.diff(raw_ibis)))) if raw_ibis.size > 1 else 0.0

    for idx, peak_time in enumerate(peak_times_s):
        peak_sample = int(peak_samples[idx]) if idx < peak_samples.size else int(round((peak_time - segment_start_s) * sfreq))
        peak_y = float(signal_norm[peak_sample]) if 0 <= peak_sample < signal_norm.size else float("nan")
        if idx == 0:
            annotations.append(
                PeakAnnotation(
                    peak_time_s=float(peak_time),
                    peak_sample=peak_sample,
                    peak_y=peak_y,
                    ibi_ms=float("nan"),
                    is_clean_ibi=True,
                    is_accepted_peak=True,
                    cleaning_reason="first_peak",
                )
            )
            continue

        ibi_ms = float((peak_time - peak_times_s[idx - 1]) * 1000.0)
        prev_ibi = float((peak_times_s[idx - 1] - peak_times_s[idx - 2]) * 1000.0) if idx >= 2 else None
        is_clean, reason = classify_ibi(
            ibi_ms,
            prev_ibi_ms=prev_ibi,
            median_jump=median_jump,
            ibi_min_ms=ibi_min_ms,
            ibi_max_ms=ibi_max_ms,
        )
        annotations.append(
            PeakAnnotation(
                peak_time_s=float(peak_time),
                peak_sample=peak_sample,
                peak_y=peak_y,
                ibi_ms=ibi_ms,
                is_clean_ibi=is_clean,
                is_accepted_peak=is_clean,
                cleaning_reason=reason,
            )
        )
    return tuple(annotations)


def resolved_debug_windows(cfg: TemporalCouplingConfig) -> list[tuple[float, float]]:
    plot_cfg = cfg.temporal_coupling.cardiac.debug_plot
    if plot_cfg.windows:
        return list(plot_cfg.windows)
    if plot_cfg.start_time_s is not None and plot_cfg.end_time_s is not None:
        return [(plot_cfg.start_time_s, plot_cfg.end_time_s)]
    return []
