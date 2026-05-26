from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import mne
from typing import Any


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Posterior montage used in prior Alpha_Analysis.ipynb PAF work.
POSTERIOR_ELECTRODES: list[str] = [
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "POz",
    "PO3",
    "PO4",
    "O1",
    "O2",
    "Oz",
]


@dataclass(frozen=True)
class EegPreprocessResult:
    raw: mne.io.BaseRaw
    bad_channels: list[str]


def detect_bad_channels_by_variance_z(raw: mne.io.BaseRaw, z_thresh: float = 3.0) -> list[str]:
    """
    Bad channels are defined as those whose variance (across time) is an outlier by z-score.
    Assumes EEG channels are already selected.
    """
    data = raw.get_data(picks="eeg")
    if data.size == 0:
        return []
    variances = np.var(data, axis=1)
    if np.allclose(variances, 0):
        return []
    z = (variances - np.mean(variances)) / (np.std(variances) + 1e-12)
    eeg_chs = np.array(raw.copy().pick("eeg").ch_names)
    bad = eeg_chs[np.abs(z) > z_thresh].tolist()
    return bad


def preprocess_eeg(
    raw: mne.io.BaseRaw,
    *,
    l_freq: float = 1.0,
    h_freq: float = 60.0,
    bad_channel_variance_z: float = 3.0,
    reference: str = "average",
    drop_bad_channels: bool = True,
) -> EegPreprocessResult:
    """
    Non-ICA preprocessing: bandpass, bad-channel detection by variance z-score, CAR.
    """
    r = raw.copy()
    r.pick_types(eeg=True, meg=False, eog=False, ecg=False, emg=False, misc=False, stim=False, exclude=[])
    r.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    bads = detect_bad_channels_by_variance_z(r, z_thresh=bad_channel_variance_z)
    r.info["bads"] = bads
    if drop_bad_channels and bads:
        r.drop_channels(bads)

    if reference == "average":
        r.set_eeg_reference("average", verbose=False)
    else:
        raise ValueError(f"Unsupported reference: {reference!r}")

    return EegPreprocessResult(raw=r, bad_channels=bads)


def compute_psd_db(
    raw: mne.io.BaseRaw,
    *,
    fmin: float = 1.0,
    fmax: float = 60.0,
    n_fft: int = 2048,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        psd_db: (n_channels, n_freqs) in dB
        freqs:  (n_freqs,)
        ch_names
    """
    r = raw.copy()
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_data = psd.get_data()
    freqs = psd.freqs

    psd_data = np.maximum(psd_data, np.finfo(float).eps)
    psd_db = 10 * np.log10(psd_data + 1e-12)
    return psd_db, freqs, r.ch_names


def band_mean(psd_db: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.full(psd_db.shape[0], np.nan)
    return np.nanmean(psd_db[:, mask], axis=1)


def global_band_value(raw: mne.io.BaseRaw, band_name: str, *, psd_fmin: float = 1.0, psd_fmax: float = 60.0) -> float:
    if band_name not in EEG_BANDS:
        raise KeyError(f"Unknown band {band_name!r}. Known: {sorted(EEG_BANDS)}")
    psd_db, freqs, _ = compute_psd_db(raw, fmin=psd_fmin, fmax=psd_fmax)
    vals = band_mean(psd_db, freqs, *EEG_BANDS[band_name])
    return float(np.nanmean(vals))


def _channel_indices(ch_names: list[str], channels: list[str]) -> np.ndarray:
    pick_set = set(channels)
    return np.array([i for i, ch in enumerate(ch_names) if ch in pick_set], dtype=int)


def compute_mean_psd_linear(
    raw: mne.io.BaseRaw,
    channels: list[str],
    *,
    fmin: float = 1.0,
    fmax: float = 30.0,
    n_fft: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD per channel, then average linear power (µV²/Hz) across channels.

    Matches the posterior-averaged spectrum used in Alpha_Analysis.ipynb.
    """
    r = raw.copy()
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_linear = psd.get_data()
    freqs = psd.freqs
    pick_idx = _channel_indices(r.ch_names, channels)
    if pick_idx.size == 0:
        return freqs, np.full(freqs.shape[0], np.nan)
    mean_power = psd_linear[pick_idx].mean(axis=0)
    return freqs, mean_power


def alpha_peak_from_spectrum(
    freqs: np.ndarray,
    power_linear: np.ndarray,
    *,
    alpha_range: tuple[float, float] = (5.0, 16.0),
    smooth_sigma: float = 1.0,
    min_height_ratio: float = 0.5,
) -> tuple[float, float]:
    """
    Detect PAF on a 1D spectrum: Gaussian-smooth alpha band, find_peaks with
    height >= min_height_ratio * max(smoothed), return highest surviving peak.

    Returns (frequency_hz, power_at_peak_uv2_hz). NaNs if no peak is found.
    """
    alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
    if alpha_idx.size == 0:
        return float("nan"), float("nan")

    alpha_freqs = freqs[alpha_idx]
    alpha_power = power_linear[alpha_idx]
    if not np.all(np.isfinite(alpha_power)):
        return float("nan"), float("nan")

    smoothed = gaussian_filter1d(alpha_power, sigma=smooth_sigma)
    threshold = float(np.max(smoothed)) * min_height_ratio
    peaks, _ = find_peaks(smoothed, height=threshold)
    if peaks.size == 0:
        return float("nan"), float("nan")

    best_peak_idx = peaks[np.argmax(smoothed[peaks])]
    return float(alpha_freqs[best_peak_idx]), float(alpha_power[best_peak_idx])


@dataclass(frozen=True)
class AlphaPeakResult:
    frequency_hz: float
    power_uv2_hz: float


def alpha_peak(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    alpha_range: tuple[float, float] = (5.0, 16.0),
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
    smooth_sigma: float = 1.0,
    min_height_ratio: float = 0.5,
) -> AlphaPeakResult:
    """
    Posterior-averaged PAF (default channels: POSTERIOR_ELECTRODES).

    Uses smoothed peak detection on the channel-mean linear PSD, following
    Alpha_Analysis.ipynb rather than per-channel argmax.
    """
    chs = channels if channels is not None else POSTERIOR_ELECTRODES
    freqs, mean_power = compute_mean_psd_linear(
        raw, chs, fmin=psd_fmin, fmax=psd_fmax, n_fft=n_fft
    )
    freq_hz, power = alpha_peak_from_spectrum(
        freqs,
        mean_power,
        alpha_range=alpha_range,
        smooth_sigma=smooth_sigma,
        min_height_ratio=min_height_ratio,
    )
    return AlphaPeakResult(frequency_hz=freq_hz, power_uv2_hz=power)


def alpha_peak_frequency(raw: mne.io.BaseRaw, **kwargs) -> float:
    """Convenience wrapper returning only PAF in Hz."""
    return alpha_peak(raw, **kwargs).frequency_hz


@dataclass(frozen=True)
class AlphaPeakPlotResult:
    fig: Any
    ax: Any
    peak_frequency_hz: float
    peak_power_uv2_hz: float


def plot_alpha_peak(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    alpha_range: tuple[float, float] = (5.0, 16.0),
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
    smooth_sigma: float = 1.0,
    min_height_ratio: float = 0.5,
    annotate_peaks: bool = True,
) -> AlphaPeakPlotResult:
    """
    Plot posterior mean PSD (linear µV²/Hz) and mark detected alpha peaks and the chosen PAF.
    """
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

    import matplotlib.pyplot as plt

    chs = channels if channels is not None else POSTERIOR_ELECTRODES
    freqs, mean_power = compute_mean_psd_linear(raw, chs, fmin=psd_fmin, fmax=psd_fmax, n_fft=n_fft)

    alpha_idx = np.where((freqs >= alpha_range[0]) & (freqs <= alpha_range[1]))[0]
    peak_f, peak_p = float("nan"), float("nan")
    smoothed = None
    peaks = np.array([], dtype=int)
    threshold = float("nan")

    if alpha_idx.size > 0 and np.all(np.isfinite(mean_power[alpha_idx])):
        alpha_power = mean_power[alpha_idx]
        smoothed = gaussian_filter1d(alpha_power, sigma=smooth_sigma)
        threshold = float(np.max(smoothed)) * min_height_ratio
        peaks, _ = find_peaks(smoothed, height=threshold)
        if peaks.size > 0:
            best_peak_idx = peaks[np.argmax(smoothed[peaks])]
            peak_f = float(freqs[alpha_idx][best_peak_idx])
            peak_p = float(alpha_power[best_peak_idx])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, mean_power, color="black", linewidth=2, label="Posterior mean PSD (linear)")

    # Alpha-band overlays
    if alpha_idx.size > 0:
        ax.axvspan(alpha_range[0], alpha_range[1], color="orange", alpha=0.12, label="Alpha search band")
        if smoothed is not None:
            ax.plot(freqs[alpha_idx], smoothed, color="tab:blue", linewidth=1.5, label="Smoothed (alpha band)")
            ax.axhline(threshold, color="tab:blue", linestyle="--", linewidth=1, alpha=0.7, label="Peak threshold")

        if peaks.size > 0:
            ax.scatter(
                freqs[alpha_idx][peaks],
                mean_power[alpha_idx][peaks],
                color="red",
                s=35,
                zorder=5,
                label=f"Peaks (n={len(peaks)})",
            )

    # Chosen PAF marker
    if np.isfinite(peak_f):
        ax.axvline(peak_f, color="red", linewidth=1.5, alpha=0.9, label=f"PAF = {peak_f:.2f} Hz")
        ax.scatter([peak_f], [peak_p], color="red", s=60, zorder=6)
        if annotate_peaks:
            ax.annotate(
                f"{peak_f:.2f} Hz",
                xy=(peak_f, peak_p),
                xytext=(peak_f + 0.5, peak_p),
                fontsize=9,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

    ax.set_title("Alpha peak frequency (PAF) — posterior mean spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz)")
    ax.set_xlim(psd_fmin, psd_fmax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    return AlphaPeakPlotResult(fig=fig, ax=ax, peak_frequency_hz=peak_f, peak_power_uv2_hz=peak_p)

