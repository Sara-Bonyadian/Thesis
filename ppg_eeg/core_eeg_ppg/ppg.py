from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
import mne


def find_ppg_channel(raw: mne.io.BaseRaw) -> str | None:
    """
    Prefer photosensor; fall back to optical/PPG-like channels.
    """
    priority: list[tuple[int, str]] = []
    for ch in raw.ch_names:
        low = ch.lower()
        if "photo" in low:
            priority.append((0, ch))
        elif "optic" in low or "ppg" in low:
            priority.append((1, ch))
    if not priority:
        return None
    priority.sort(key=lambda x: x[0])
    return priority[0][1]


def extract_ppg_signal(raw: mne.io.BaseRaw, channel: str, *, start_time_s: float, end_time_s: float) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if channel not in raw.ch_names:
        return None, None
    sfreq = float(raw.info["sfreq"])
    start_sample = int(start_time_s * sfreq)
    end_sample = int(end_time_s * sfreq)
    if end_sample <= start_sample or end_sample > raw.n_times:
        return None, None
    data, times = raw[channel, start_sample:end_sample]
    return data[0], times


def normalize_signal(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / std


def detect_heartbeats(
    ppg_norm: np.ndarray,
    *,
    sampling_rate: float,
    peak_min_distance_s: float = 0.4,
    peak_height: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    distance = int(peak_min_distance_s * sampling_rate)
    peaks, _properties = signal.find_peaks(ppg_norm, distance=distance, height=peak_height)
    peak_times_s = peaks / sampling_rate
    return peak_times_s, peaks


def calculate_ibi_ms(peak_times_s: np.ndarray) -> np.ndarray | None:
    if peak_times_s is None or len(peak_times_s) < 2:
        return None
    return np.diff(peak_times_s) * 1000.0


def clean_ibi_for_metrics(
    ibi_ms: np.ndarray | None,
    *,
    ibi_min_ms: float = 400.0,
    ibi_max_ms: float = 1200.0,
) -> np.ndarray | None:
    if ibi_ms is None or len(ibi_ms) < 3:
        return None

    ibi = np.array(ibi_ms, dtype=float)
    ibi = ibi[(ibi > ibi_min_ms) & (ibi < ibi_max_ms)]
    if len(ibi) < 3:
        return None

    diff_ibi = np.abs(np.diff(ibi))
    median_diff = float(np.median(diff_ibi)) or 1.0
    jumpy = diff_ibi >= 3 * median_diff
    if len(jumpy) == 0 or not np.any(jumpy):
        return ibi

    keep_indices = np.where(~jumpy)[0]
    keep_indices = np.append(keep_indices, keep_indices[-1] + 1)
    keep_indices = keep_indices[keep_indices < len(ibi)]
    cleaned = ibi[keep_indices]
    if len(cleaned) < 3:
        return None
    return cleaned


@dataclass(frozen=True)
class PpgIbiResult:
    ppg_channel: str | None
    sfreq: float
    ppg_segment_norm: np.ndarray | None
    times: np.ndarray | None
    peak_times_s: np.ndarray | None
    peaks_idx: np.ndarray | None
    ibi_ms_raw: np.ndarray | None
    ibi_ms_clean: np.ndarray | None


def extract_clean_ppg_ibi_from_raw(
    raw: mne.io.BaseRaw,
    *,
    start_time_s: float,
    end_time_s: float,
    peak_min_distance_s: float = 0.4,
    peak_height: float = 0.3,
    ibi_min_ms: float = 400.0,
    ibi_max_ms: float = 1200.0,
) -> PpgIbiResult:
    ppg_ch = find_ppg_channel(raw)
    sfreq = float(raw.info["sfreq"])
    if ppg_ch is None:
        return PpgIbiResult(None, sfreq, None, None, None, None, None, None)

    ppg, times = extract_ppg_signal(raw, ppg_ch, start_time_s=start_time_s, end_time_s=end_time_s)
    if ppg is None:
        return PpgIbiResult(ppg_ch, sfreq, None, None, None, None, None, None)

    ppg_norm = normalize_signal(ppg)
    peak_times_s, peaks_idx = detect_heartbeats(
        ppg_norm,
        sampling_rate=sfreq,
        peak_min_distance_s=peak_min_distance_s,
        peak_height=peak_height,
    )
    ibi_ms_raw = calculate_ibi_ms(peak_times_s)
    ibi_ms_clean = clean_ibi_for_metrics(ibi_ms_raw, ibi_min_ms=ibi_min_ms, ibi_max_ms=ibi_max_ms)

    return PpgIbiResult(
        ppg_channel=ppg_ch,
        sfreq=sfreq,
        ppg_segment_norm=ppg_norm,
        times=times,
        peak_times_s=peak_times_s,
        peaks_idx=peaks_idx,
        ibi_ms_raw=ibi_ms_raw,
        ibi_ms_clean=ibi_ms_clean,
    )


def rmssd_ms(ibi_ms: np.ndarray) -> float:
    diffs = np.diff(ibi_ms)
    return float(np.sqrt(np.mean(diffs**2)))


def sdnn_ms(ibi_ms: np.ndarray) -> float:
    return float(np.std(ibi_ms, ddof=1))


def mean_rr_ms(ibi_ms: np.ndarray) -> float:
    return float(np.mean(ibi_ms))


def mean_hr_bpm(ibi_ms: np.ndarray) -> float:
    return float(60000.0 / np.mean(ibi_ms))


def peak_hr_bpm(ibi_ms: np.ndarray, q: float = 0.95) -> float:
    hr = 60000.0 / ibi_ms
    return float(np.quantile(hr, q))

