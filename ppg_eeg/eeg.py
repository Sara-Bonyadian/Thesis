from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import mne
from typing import Any, Sequence


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Standard frontal pairs for Frontal Alpha Asymmetry (left, right).
FRONTAL_PAIRS_DEFAULT: list[tuple[str, str]] = [
    ("F3", "F4"),
    ("F7", "F8"),
]

# Standard frontal-midline channels for FM-theta.
FRONTAL_MIDLINE_CHANNELS_DEFAULT: list[str] = [
    "Fz",
    "FCz",
    "Cz",
]

# Standard frontal channels for frontal-beta.
FRONTAL_BETA_CHANNELS_DEFAULT: list[str] = [
    "F3",
    "F4",
    "F7",
    "F8",
    "Fz",
    "FCz",
]

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


def _trapezoid_integral(y: np.ndarray, *, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    NumPy 2.x removed np.trapz in favor of np.trapezoid.
    Keep compatibility with both old and new NumPy versions.
    """
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def _integrate_band_power(band_power: np.ndarray, band_freqs: np.ndarray) -> np.ndarray:
    """
    Integrate PSD across a frequency band.
    If only one PSD bin falls in-band (coarse n_fft), use that bin value.
    """
    if band_power.shape[1] == 1:
        return band_power[:, 0]
    return _trapezoid_integral(band_power, x=band_freqs, axis=1)


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


def channel_band_powers(
    raw: mne.io.BaseRaw,
    *,
    band_names: Sequence[str] = ("theta", "alpha", "beta"),
    psd_fmin: float = 1.0,
    psd_fmax: float = 60.0,
    n_fft: int = 2048,
) -> dict[str, dict[str, float]]:
    """
    Compute per-channel band powers in both dB and linear integrated units.

    Returns:
        {
            "<channel_name>": {
                "power_theta": <mean dB>,
                "theta_power_uv2": <integrated linear µV²>,
                ...
            },
            ...
        }
    """
    normalized_bands = [str(name).casefold() for name in band_names]
    unknown = [band for band in normalized_bands if band not in EEG_BANDS]
    if unknown:
        raise KeyError(f"Unknown bands {unknown!r}. Known: {sorted(EEG_BANDS)}")

    r = raw.copy()
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_linear = np.maximum(psd.get_data(), np.finfo(float).eps)
    freqs = psd.freqs
    psd_db = 10 * np.log10(psd_linear + 1e-12)

    band_arrays: dict[str, np.ndarray] = {
        band: band_mean(psd_db, freqs, *EEG_BANDS[band]) for band in normalized_bands
    }
    linear_band_arrays: dict[str, np.ndarray] = {}
    for band in normalized_bands:
        band_lo, band_hi = EEG_BANDS[band]
        mask = (freqs >= band_lo) & (freqs <= band_hi)
        if not np.any(mask):
            linear_band_arrays[band] = np.full(psd_linear.shape[0], np.nan)
            continue
        linear_band_arrays[band] = _integrate_band_power(psd_linear[:, mask], freqs[mask])

    out: dict[str, dict[str, float]] = {}
    for ch_idx, ch_name in enumerate(r.ch_names):
        row: dict[str, float] = {}
        for band in normalized_bands:
            row[f"power_{band}"] = float(band_arrays[band][ch_idx])
            row[f"{band}_power_uv2"] = float(linear_band_arrays[band][ch_idx])
        out[ch_name] = row
    return out


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


@dataclass(frozen=True)
class FaaResult:
    """
    Frontal Alpha Asymmetry result.

    per_pair       : {"<right>-<left>": ln(P_alpha[right]) - ln(P_alpha[left])}
                     NaN for pairs that could not be computed.
    mean_faa       : mean across pairs with finite values (NaN if none).
    alpha_band_used: (lo_hz, hi_hz) actually used for integration.
    missing_pairs  : pairs skipped because one/both electrodes were absent
                     from `raw` (e.g. dropped as bad during preprocessing).
    """

    per_pair: dict[str, float]
    mean_faa: float
    alpha_band_used: tuple[float, float]
    missing_pairs: list[tuple[str, str]]


def _alpha_power_per_channel(
    raw: mne.io.BaseRaw,
    channels: list[str],
    *,
    alpha_band: tuple[float, float],
    psd_fmin: float,
    psd_fmax: float,
    n_fft: int,
) -> dict[str, float]:
    """
    Welch PSD per channel (linear µV²/Hz), then integrate over `alpha_band`
    with the trapezoidal rule. Returns {channel_name: alpha_power_uv2}.
    Channels not present in `raw` are silently skipped.
    """
    present = [ch for ch in channels if ch in raw.ch_names]
    if not present:
        return {}

    r = raw.copy().pick(present)
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_linear = psd.get_data()  # (n_channels, n_freqs)
    freqs = psd.freqs

    mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    if not np.any(mask):
        return {ch: float("nan") for ch in r.ch_names}

    band_freqs = freqs[mask]
    band_power = psd_linear[:, mask]
    alpha_power = _integrate_band_power(band_power, band_freqs)  # µV²
    return {ch: float(p) for ch, p in zip(r.ch_names, alpha_power)}


def faa(
    raw: mne.io.BaseRaw,
    *,
    pairs: list[tuple[str, str]] | None = None,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    use_individual_alpha: bool = False,
    iaf_half_width: float = 2.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FaaResult:
    """
    Frontal Alpha Asymmetry (FAA).

    For each (left, right) pair, FAA is defined as

        FAA = ln(P_alpha[right]) - ln(P_alpha[left])

    where P_alpha is the alpha-band power (µV²) obtained by integrating the
    Welch PSD (linear, µV²/Hz) over `alpha_band` via the trapezoidal rule.

    Sign convention: positive FAA -> *less* alpha on the left frontal site
    -> relatively greater left-frontal cortical activity -> typically
    interpreted as approach motivation / positive affect. Negative FAA ->
    withdrawal / negative affect.

    Notes on reference: FAA depends on the EEG reference. This function
    assumes `raw` has already been re-referenced (e.g. average reference via
    `preprocess_eeg`). Switching references will change the FAA values.

    Parameters
    ----------
    raw : preprocessed EEG.
    pairs : list of (left, right) electrode names.
        Defaults to FRONTAL_PAIRS_DEFAULT = [("F3","F4"), ("F7","F8")].
    alpha_band : explicit alpha band in Hz (default 8-13).
        Ignored if `use_individual_alpha` is True and PAF was found.
    use_individual_alpha : if True, center the alpha band on the
        participant's PAF (computed posteriorly) with total width
        2 * iaf_half_width. Falls back to `alpha_band` if PAF is NaN.
    iaf_half_width : half-width (Hz) for the individualized alpha band.
    psd_fmin, psd_fmax, n_fft : Welch PSD settings.
    """
    if pairs is None:
        pairs = list(FRONTAL_PAIRS_DEFAULT)

    if use_individual_alpha:
        paf = alpha_peak_frequency(
            raw,
            psd_fmin=psd_fmin,
            psd_fmax=psd_fmax,
            n_fft=n_fft,
        )
        if np.isfinite(paf):
            band_used: tuple[float, float] = (
                float(paf - iaf_half_width),
                float(paf + iaf_half_width),
            )
        else:
            band_used = (float(alpha_band[0]), float(alpha_band[1]))
    else:
        band_used = (float(alpha_band[0]), float(alpha_band[1]))

    seen: set[str] = set()
    unique_channels: list[str] = []
    for L, R in pairs:
        for ch in (L, R):
            if ch not in seen:
                seen.add(ch)
                unique_channels.append(ch)

    alpha_per_ch = _alpha_power_per_channel(
        raw,
        unique_channels,
        alpha_band=band_used,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )

    per_pair: dict[str, float] = {}
    missing: list[tuple[str, str]] = []
    for L, R in pairs:
        key = f"{R}-{L}"
        p_l = alpha_per_ch.get(L)
        p_r = alpha_per_ch.get(R)
        if (
            p_l is None
            or p_r is None
            or not np.isfinite(p_l)
            or not np.isfinite(p_r)
            or p_l <= 0.0
            or p_r <= 0.0
        ):
            per_pair[key] = float("nan")
            missing.append((L, R))
            continue
        per_pair[key] = float(np.log(p_r) - np.log(p_l))

    finite_vals = [v for v in per_pair.values() if np.isfinite(v)]
    mean_faa = float(np.mean(finite_vals)) if finite_vals else float("nan")

    return FaaResult(
        per_pair=per_pair,
        mean_faa=mean_faa,
        alpha_band_used=band_used,
        missing_pairs=missing,
    )


def frontal_alpha_asymmetry(raw: mne.io.BaseRaw, **kwargs) -> float:
    """Convenience wrapper returning only the mean FAA across pairs."""
    return faa(raw, **kwargs).mean_faa


@dataclass(frozen=True)
class FmThetaResult:
    """
    Frontal-midline theta (FM-theta) result.

    per_channel_power : {"<ch>": integrated theta power in µV²}
                       (NaN for channels that are missing or invalid).
    mean_theta_power  : mean theta power across channels with finite positive
                       values (NaN if none).
    fm_theta          : final FM-theta value. If log_transform=True,
                       fm_theta = ln(mean_theta_power); otherwise it equals
                       mean_theta_power.
    theta_band_used   : (lo_hz, hi_hz) used for integration.
    missing_channels  : requested channels absent from `raw`.
    """

    per_channel_power: dict[str, float]
    mean_theta_power: float
    fm_theta: float
    theta_band_used: tuple[float, float]
    missing_channels: list[str]


def _theta_power_per_channel(
    raw: mne.io.BaseRaw,
    channels: list[str],
    *,
    theta_band: tuple[float, float],
    psd_fmin: float,
    psd_fmax: float,
    n_fft: int,
) -> dict[str, float]:
    """
    Welch PSD per channel (linear µV²/Hz), then integrate over `theta_band`
    with the trapezoidal rule. Returns {channel_name: theta_power_uv2}.
    Channels not present in `raw` are silently skipped.
    """
    present = [ch for ch in channels if ch in raw.ch_names]
    if not present:
        return {}

    r = raw.copy().pick(present)
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_linear = psd.get_data()  # (n_channels, n_freqs)
    freqs = psd.freqs

    mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    if not np.any(mask):
        return {ch: float("nan") for ch in r.ch_names}

    band_freqs = freqs[mask]
    band_power = psd_linear[:, mask]
    theta_power = _integrate_band_power(band_power, band_freqs)  # µV²
    return {ch: float(p) for ch, p in zip(r.ch_names, theta_power)}


def fm_theta(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    theta_band: tuple[float, float] = (4.0, 7.0),
    log_transform: bool = True,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FmThetaResult:
    """
    Frontal-midline theta (FM-theta).

    Computes linear theta power (µV²) per selected frontal-midline channel by
    integrating Welch PSD over `theta_band`, then averages across valid
    channels. Optionally applies a natural-log transform to the mean.

    Parameters
    ----------
    raw : preprocessed EEG.
    channels : frontal-midline channels to include.
        Defaults to FRONTAL_MIDLINE_CHANNELS_DEFAULT = ["Fz", "FCz", "Cz"].
    theta_band : theta range in Hz (default 4-7).
    log_transform : if True, return ln(mean_theta_power) as fm_theta.
    psd_fmin, psd_fmax, n_fft : Welch PSD settings.
    """
    requested = channels if channels is not None else FRONTAL_MIDLINE_CHANNELS_DEFAULT

    # Keep order stable while removing duplicates.
    seen: set[str] = set()
    unique_channels: list[str] = []
    for ch in requested:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)

    band_used = (float(theta_band[0]), float(theta_band[1]))
    theta_per_ch = _theta_power_per_channel(
        raw,
        unique_channels,
        theta_band=band_used,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )

    per_channel: dict[str, float] = {}
    missing_channels: list[str] = []
    for ch in unique_channels:
        val = theta_per_ch.get(ch)
        if val is None:
            per_channel[ch] = float("nan")
            missing_channels.append(ch)
        else:
            per_channel[ch] = float(val)

    finite_positive_vals = [
        v for v in per_channel.values() if np.isfinite(v) and v > 0.0
    ]
    mean_theta_power = (
        float(np.mean(finite_positive_vals)) if finite_positive_vals else float("nan")
    )

    if log_transform:
        fm_val = (
            float(np.log(mean_theta_power))
            if np.isfinite(mean_theta_power) and mean_theta_power > 0.0
            else float("nan")
        )
    else:
        fm_val = mean_theta_power

    return FmThetaResult(
        per_channel_power=per_channel,
        mean_theta_power=mean_theta_power,
        fm_theta=fm_val,
        theta_band_used=band_used,
        missing_channels=missing_channels,
    )


def frontal_midline_theta(raw: mne.io.BaseRaw, **kwargs) -> float:
    """Convenience wrapper returning only the FM-theta value."""
    return fm_theta(raw, **kwargs).fm_theta


@dataclass(frozen=True)
class FrontalBetaResult:
    """
    Frontal beta result.

    per_channel_power : {"<ch>": integrated beta power in µV²}
                       (NaN for channels that are missing or invalid).
    mean_beta_power   : mean beta power across channels with finite positive
                       values (NaN if none).
    frontal_beta      : final frontal-beta value. If log_transform=True,
                       frontal_beta = ln(mean_beta_power); otherwise it equals
                       mean_beta_power.
    beta_band_used    : (lo_hz, hi_hz) used for integration.
    missing_channels  : requested channels absent from `raw`.
    """

    per_channel_power: dict[str, float]
    mean_beta_power: float
    frontal_beta: float
    beta_band_used: tuple[float, float]
    missing_channels: list[str]


def _beta_power_per_channel(
    raw: mne.io.BaseRaw,
    channels: list[str],
    *,
    beta_band: tuple[float, float],
    psd_fmin: float,
    psd_fmax: float,
    n_fft: int,
) -> dict[str, float]:
    """
    Welch PSD per channel (linear µV²/Hz), then integrate over `beta_band`
    with the trapezoidal rule. Returns {channel_name: beta_power_uv2}.
    Channels not present in `raw` are silently skipped.
    """
    present = [ch for ch in channels if ch in raw.ch_names]
    if not present:
        return {}

    r = raw.copy().pick(present)
    r.apply_function(lambda x: x * 1e6)  # V -> µV
    psd = r.compute_psd(
        method="welch",
        fmin=psd_fmin,
        fmax=psd_fmax,
        n_fft=min(n_fft, r.n_times),
        verbose=False,
    )
    psd_linear = psd.get_data()  # (n_channels, n_freqs)
    freqs = psd.freqs

    mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
    if not np.any(mask):
        return {ch: float("nan") for ch in r.ch_names}

    band_freqs = freqs[mask]
    band_power = psd_linear[:, mask]
    beta_power = _integrate_band_power(band_power, band_freqs)  # µV²
    return {ch: float(p) for ch, p in zip(r.ch_names, beta_power)}


def frontal_beta(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    beta_band: tuple[float, float] = (13.0, 30.0),
    log_transform: bool = True,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FrontalBetaResult:
    """
    Frontal beta.

    Computes linear beta power (µV²) per selected frontal channel by
    integrating Welch PSD over `beta_band`, then averages across valid
    channels. Optionally applies a natural-log transform to the mean.

    Parameters
    ----------
    raw : preprocessed EEG.
    channels : frontal channels to include.
        Defaults to FRONTAL_BETA_CHANNELS_DEFAULT.
    beta_band : beta range in Hz (default 13-30).
    log_transform : if True, return ln(mean_beta_power) as frontal_beta.
    psd_fmin, psd_fmax, n_fft : Welch PSD settings.
    """
    requested = channels if channels is not None else FRONTAL_BETA_CHANNELS_DEFAULT

    # Keep order stable while removing duplicates.
    seen: set[str] = set()
    unique_channels: list[str] = []
    for ch in requested:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)

    band_used = (float(beta_band[0]), float(beta_band[1]))
    beta_per_ch = _beta_power_per_channel(
        raw,
        unique_channels,
        beta_band=band_used,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )

    per_channel: dict[str, float] = {}
    missing_channels: list[str] = []
    for ch in unique_channels:
        val = beta_per_ch.get(ch)
        if val is None:
            per_channel[ch] = float("nan")
            missing_channels.append(ch)
        else:
            per_channel[ch] = float(val)

    finite_positive_vals = [
        v for v in per_channel.values() if np.isfinite(v) and v > 0.0
    ]
    mean_beta_power = (
        float(np.mean(finite_positive_vals)) if finite_positive_vals else float("nan")
    )

    if log_transform:
        frontal_beta_val = (
            float(np.log(mean_beta_power))
            if np.isfinite(mean_beta_power) and mean_beta_power > 0.0
            else float("nan")
        )
    else:
        frontal_beta_val = mean_beta_power

    return FrontalBetaResult(
        per_channel_power=per_channel,
        mean_beta_power=mean_beta_power,
        frontal_beta=frontal_beta_val,
        beta_band_used=band_used,
        missing_channels=missing_channels,
    )


def frontal_beta_value(raw: mne.io.BaseRaw, **kwargs) -> float:
    """Convenience wrapper returning only the frontal-beta value."""
    return frontal_beta(raw, **kwargs).frontal_beta


@dataclass(frozen=True)
class FrontalBetaPlotResult:
    fig: Any
    ax: Any
    result: FrontalBetaResult


def plot_frontal_beta(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    beta_band: tuple[float, float] = (13.0, 30.0),
    log_transform: bool = True,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FrontalBetaPlotResult:
    """
    Plot per-channel beta power (µV²) for frontal channels used in
    frontal-beta, with a horizontal line for the mean beta power.
    """
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

    import matplotlib.pyplot as plt

    result = frontal_beta(
        raw,
        channels=channels,
        beta_band=beta_band,
        log_transform=log_transform,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )

    requested = channels if channels is not None else FRONTAL_BETA_CHANNELS_DEFAULT
    # Keep order stable while removing duplicates.
    seen: set[str] = set()
    unique_channels: list[str] = []
    for ch in requested:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)

    x = np.arange(len(unique_channels), dtype=float)
    y = np.array([result.per_channel_power.get(ch, float("nan")) for ch in unique_channels], dtype=float)
    valid_mask = np.isfinite(y) & (y > 0.0)

    fig, ax = plt.subplots(figsize=(8.0, 4.0))

    # Plot valid and missing/invalid channels separately to keep NaNs visible.
    if np.any(valid_mask):
        ax.bar(
            x[valid_mask],
            y[valid_mask],
            color="tab:purple",
            alpha=0.85,
            label="Beta power (valid channels)",
        )
    if np.any(~valid_mask):
        ax.bar(
            x[~valid_mask],
            np.zeros(np.sum(~valid_mask)),
            color="lightgray",
            edgecolor="gray",
            hatch="//",
            label="Missing/invalid channel",
        )
        for xi in x[~valid_mask]:
            ax.text(xi, 0.0, "NaN", ha="center", va="bottom", fontsize=8, color="gray")

    if np.isfinite(result.mean_beta_power):
        ax.axhline(
            result.mean_beta_power,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean beta power = {result.mean_beta_power:.4f} µV²",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(unique_channels)
    ax.set_ylabel("Beta power (µV²)")
    ax.set_xlabel("Frontal channels")
    ax.grid(True, axis="y", alpha=0.25)

    fb_str = f"{result.frontal_beta:.4f}" if np.isfinite(result.frontal_beta) else "NaN"
    transform_label = "log" if log_transform else "linear"
    band = result.beta_band_used
    ax.set_title(
        f"Frontal beta ({transform_label}) = {fb_str}\n"
        f"Beta band {band[0]:.1f}-{band[1]:.1f} Hz"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    return FrontalBetaPlotResult(fig=fig, ax=ax, result=result)


@dataclass(frozen=True)
class FmThetaPlotResult:
    fig: Any
    ax: Any
    result: FmThetaResult


def plot_fm_theta(
    raw: mne.io.BaseRaw,
    *,
    channels: list[str] | None = None,
    theta_band: tuple[float, float] = (4.0, 7.0),
    log_transform: bool = True,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FmThetaPlotResult:
    """
    Plot per-channel theta power (µV²) for frontal-midline channels used in
    FM-theta, with a horizontal line for the mean theta power.
    """
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

    import matplotlib.pyplot as plt

    result = fm_theta(
        raw,
        channels=channels,
        theta_band=theta_band,
        log_transform=log_transform,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )

    requested = channels if channels is not None else FRONTAL_MIDLINE_CHANNELS_DEFAULT
    # Keep order stable while removing duplicates.
    seen: set[str] = set()
    unique_channels: list[str] = []
    for ch in requested:
        if ch not in seen:
            seen.add(ch)
            unique_channels.append(ch)

    x = np.arange(len(unique_channels), dtype=float)
    y = np.array([result.per_channel_power.get(ch, float("nan")) for ch in unique_channels], dtype=float)
    valid_mask = np.isfinite(y) & (y > 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # Plot valid and missing/invalid channels separately to keep NaNs visible.
    if np.any(valid_mask):
        ax.bar(
            x[valid_mask],
            y[valid_mask],
            color="tab:blue",
            alpha=0.85,
            label="Theta power (valid channels)",
        )
    if np.any(~valid_mask):
        ax.bar(
            x[~valid_mask],
            np.zeros(np.sum(~valid_mask)),
            color="lightgray",
            edgecolor="gray",
            hatch="//",
            label="Missing/invalid channel",
        )
        for xi in x[~valid_mask]:
            ax.text(xi, 0.0, "NaN", ha="center", va="bottom", fontsize=8, color="gray")

    if np.isfinite(result.mean_theta_power):
        ax.axhline(
            result.mean_theta_power,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean theta power = {result.mean_theta_power:.4f} µV²",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(unique_channels)
    ax.set_ylabel("Theta power (µV²)")
    ax.set_xlabel("Frontal-midline channels")
    ax.grid(True, axis="y", alpha=0.25)

    fm_str = f"{result.fm_theta:.4f}" if np.isfinite(result.fm_theta) else "NaN"
    transform_label = "log" if log_transform else "linear"
    band = result.theta_band_used
    ax.set_title(
        f"FM-theta ({transform_label}) = {fm_str}\n"
        f"Theta band {band[0]:.1f}-{band[1]:.1f} Hz"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    return FmThetaPlotResult(fig=fig, ax=ax, result=result)


@dataclass(frozen=True)
class FaaPlotResult:
    fig: Any
    axes: Any  # list of matplotlib Axes, one per pair
    result: FaaResult


def plot_faa(
    raw: mne.io.BaseRaw,
    *,
    pairs: list[tuple[str, str]] | None = None,
    alpha_band: tuple[float, float] = (8.0, 13.0),
    use_individual_alpha: bool = False,
    iaf_half_width: float = 2.0,
    psd_fmin: float = 1.0,
    psd_fmax: float = 30.0,
    n_fft: int = 2048,
) -> FaaPlotResult:
    """
    Plot per-pair linear PSDs for the frontal electrodes used in FAA, with the
    alpha integration band shaded and the per-pair FAA annotated. One subplot
    per pair; figure suptitle reports the mean FAA and the alpha band used.
    """
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

    import matplotlib.pyplot as plt

    if pairs is None:
        pairs = list(FRONTAL_PAIRS_DEFAULT)

    result = faa(
        raw,
        pairs=pairs,
        alpha_band=alpha_band,
        use_individual_alpha=use_individual_alpha,
        iaf_half_width=iaf_half_width,
        psd_fmin=psd_fmin,
        psd_fmax=psd_fmax,
        n_fft=n_fft,
    )
    band_used = result.alpha_band_used

    # One Welch PSD pass over all needed channels (those present in raw).
    seen: set[str] = set()
    needed: list[str] = []
    for L, R in pairs:
        for ch in (L, R):
            if ch not in seen and ch in raw.ch_names:
                seen.add(ch)
                needed.append(ch)

    if needed:
        r = raw.copy().pick(needed)
        r.apply_function(lambda x: x * 1e6)  # V -> µV
        psd = r.compute_psd(
            method="welch",
            fmin=psd_fmin,
            fmax=psd_fmax,
            n_fft=min(n_fft, r.n_times),
            verbose=False,
        )
        psd_linear = psd.get_data()
        freqs = psd.freqs
        psd_by_ch = {ch: psd_linear[i] for i, ch in enumerate(r.ch_names)}
    else:
        freqs = np.array([])
        psd_by_ch = {}

    n_pairs = len(pairs)
    fig, axes = plt.subplots(
        1, n_pairs, figsize=(5.0 * n_pairs, 4.0), squeeze=False, sharey=True
    )
    axes_flat = list(axes[0])

    for ax, (L, R) in zip(axes_flat, pairs):
        pair_key = f"{R}-{L}"
        faa_val = result.per_pair.get(pair_key, float("nan"))
        ax.axvspan(
            band_used[0], band_used[1], color="orange", alpha=0.15, label=f"Alpha {band_used[0]:.1f}-{band_used[1]:.1f} Hz"
        )
        plotted_any = False
        for ch, color, side in ((L, "tab:blue", "L"), (R, "tab:red", "R")):
            if ch in psd_by_ch:
                ax.plot(freqs, psd_by_ch[ch], color=color, linewidth=1.8, label=f"{ch} ({side})")
                plotted_any = True
            else:
                ax.plot([], [], color=color, linewidth=1.8, label=f"{ch} ({side}) — missing")

        title = f"{R} vs {L}"
        if np.isfinite(faa_val):
            title += f"\nFAA = {faa_val:+.3f}"
        else:
            title += "\nFAA = NaN"
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(psd_fmin, psd_fmax)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        if not plotted_any:
            ax.text(
                0.5, 0.5, "both electrodes missing",
                transform=ax.transAxes, ha="center", va="center", color="gray",
            )

    axes_flat[0].set_ylabel("Power (µV²/Hz)")

    mean_str = f"{result.mean_faa:+.3f}" if np.isfinite(result.mean_faa) else "NaN"
    fig.suptitle(
        f"Frontal Alpha Asymmetry — mean FAA = {mean_str}  "
        f"(alpha band {band_used[0]:.1f}-{band_used[1]:.1f} Hz, sign: ln(R) − ln(L))"
    )
    fig.tight_layout()

    return FaaPlotResult(fig=fig, axes=axes_flat, result=result)

