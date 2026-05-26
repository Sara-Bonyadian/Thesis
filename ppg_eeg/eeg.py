from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mne


EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


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

