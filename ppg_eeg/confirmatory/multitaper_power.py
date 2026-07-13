"""DPSS multitaper EEG band-power extraction for confirmatory analysis."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np
from scipy.signal import filtfilt, iirnotch
from scipy.signal.windows import dpss

from ..core_eeg_ppg.eeg import preprocess_eeg
from ..core_eeg_ppg.features_core import _read_raw

FEATURES_FILENAME = "features_multitaper_power.csv"
QC_FILENAME = "multitaper_qc.csv"
WINDOW_S = 2.0
STEP_S = 1.0
TIME_BANDWIDTH = 3.0
N_TAPERS = 5
ROBUST_MEDIAN_CHANNEL = "__robust_median__"

BANDS_HZ: dict[str, tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 29.0),
    "low_gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class MultitaperFeature:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    window_index: int
    window_start_s: float
    window_center_s: float
    window_end_s: float
    channel: str
    aggregation: str
    band: str
    band_low_hz: float
    band_high_hz: float
    absolute_power: float
    absolute_log10_power: float

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "window_index": self.window_index,
            "window_start_s": self.window_start_s,
            "window_center_s": self.window_center_s,
            "window_end_s": self.window_end_s,
            "channel": self.channel,
            "aggregation": self.aggregation,
            "band": self.band,
            "band_low_hz": self.band_low_hz,
            "band_high_hz": self.band_high_hz,
            "absolute_power": self.absolute_power,
            "absolute_log10_power": self.absolute_log10_power,
        }


@dataclass(frozen=True)
class MultitaperQC:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    eeg_file: str
    sfreq_hz: float
    duration_s: float
    window_s: float
    step_s: float
    time_bandwidth: float
    n_tapers: int
    n_input_channels: int
    requested_channels: str
    usable_channels: str
    rejected_channels: str
    missing_channels: str
    n_usable_channels: int
    line_frequency_hz: float | None
    line_noise_handling: str
    line_noise_ratio_before: float | None
    line_noise_ratio_after: float | None
    n_windows: int
    n_channel_band_rows: int
    n_nonfinite_power_rows: int
    nyquist_hz: float
    frequency_resolution_hz: float
    theta_frequency_bins: int
    alpha_frequency_bins: int
    beta_frequency_bins: int
    low_gamma_frequency_bins: int
    all_bands_supported: bool
    spectral_qc_passed: bool
    status: str
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "eeg_file": self.eeg_file,
            "sfreq_hz": self.sfreq_hz,
            "duration_s": self.duration_s,
            "window_s": self.window_s,
            "step_s": self.step_s,
            "time_bandwidth": self.time_bandwidth,
            "n_tapers": self.n_tapers,
            "n_input_channels": self.n_input_channels,
            "requested_channels": self.requested_channels,
            "usable_channels": self.usable_channels,
            "rejected_channels": self.rejected_channels,
            "missing_channels": self.missing_channels,
            "n_usable_channels": self.n_usable_channels,
            "line_frequency_hz": self.line_frequency_hz,
            "line_noise_handling": self.line_noise_handling,
            "line_noise_ratio_before": self.line_noise_ratio_before,
            "line_noise_ratio_after": self.line_noise_ratio_after,
            "n_windows": self.n_windows,
            "n_channel_band_rows": self.n_channel_band_rows,
            "n_nonfinite_power_rows": self.n_nonfinite_power_rows,
            "nyquist_hz": self.nyquist_hz,
            "frequency_resolution_hz": self.frequency_resolution_hz,
            "theta_frequency_bins": self.theta_frequency_bins,
            "alpha_frequency_bins": self.alpha_frequency_bins,
            "beta_frequency_bins": self.beta_frequency_bins,
            "low_gamma_frequency_bins": self.low_gamma_frequency_bins,
            "all_bands_supported": self.all_bands_supported,
            "spectral_qc_passed": self.spectral_qc_passed,
            "status": self.status,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class MultitaperResult:
    features: tuple[MultitaperFeature, ...]
    qc: MultitaperQC


def _identity(identity: Mapping[str, object] | None) -> dict[str, str]:
    values = identity or {}
    return {
        "dataset_id": str(values.get("dataset_id", "")).strip().casefold(),
        "subject_id": str(values.get("subject_id", "")).strip().casefold(),
        "task": str(values.get("task", "")).strip().casefold(),
        "condition": str(values.get("condition", "")).strip().casefold(),
        "observation_id": str(values.get("observation_id", "")).strip(),
    }


def _resolve_channels(
    ch_names: Sequence[str],
    requested: Sequence[str] | None,
) -> tuple[list[int], list[str], list[str]]:
    lookup = {name.casefold(): (index, name) for index, name in enumerate(ch_names)}
    requested_names = list(requested) if requested is not None else list(ch_names)
    indices: list[int] = []
    usable: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in requested_names:
        normalized = str(name).strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        match = lookup.get(normalized)
        if match is None:
            missing.append(str(name))
        else:
            indices.append(match[0])
            usable.append(match[1])
    return indices, usable, missing


def _line_noise_ratio(
    data: np.ndarray,
    *,
    sfreq: float,
    line_frequency_hz: float | None,
) -> float | None:
    if line_frequency_hz is None or line_frequency_hz <= 0:
        return None
    nyquist = sfreq / 2.0
    if line_frequency_hz >= nyquist or data.shape[1] < 4:
        return None
    centered = data - np.mean(data, axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)
    line_mask = np.abs(freqs - line_frequency_hz) <= 1.0
    broadband_mask = (freqs >= 1.0) & (freqs <= nyquist)
    denominator = float(np.sum(spectrum[:, broadband_mask]))
    if denominator <= 0 or not np.any(line_mask):
        return 0.0
    return float(np.sum(spectrum[:, line_mask]) / denominator)


def _notch_line_noise(
    data: np.ndarray,
    *,
    sfreq: float,
    line_frequency_hz: float | None,
) -> tuple[np.ndarray, str]:
    if line_frequency_hz is None or line_frequency_hz <= 0:
        return data.copy(), "not_applied_line_frequency_unknown"
    if line_frequency_hz >= sfreq / 2.0:
        return data.copy(), "not_applied_line_frequency_at_or_above_nyquist"
    b, a = iirnotch(line_frequency_hz, Q=30.0, fs=sfreq)
    if data.shape[1] <= 3 * max(len(a), len(b)):
        return data.copy(), "not_applied_recording_too_short"
    return filtfilt(b, a, data, axis=1), f"notch_{line_frequency_hz:g}_hz_q30"


def _integrate_band(
    psd: np.ndarray,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return np.full(psd.shape[0], np.nan, dtype=float)
    selected = psd[:, mask]
    selected_freqs = freqs[mask]
    if selected_freqs.size == 1:
        return selected[:, 0]
    return np.trapezoid(selected, x=selected_freqs, axis=1)


def _multitaper_psd(
    window: np.ndarray,
    *,
    sfreq: float,
    time_bandwidth: float,
    n_tapers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = window.shape[1]
    tapers, ratios = dpss(
        n_samples,
        NW=time_bandwidth,
        Kmax=n_tapers,
        sym=False,
        norm=2,
        return_ratios=True,
    )
    centered = window - np.mean(window, axis=1, keepdims=True)
    tapered = centered[:, np.newaxis, :] * tapers[np.newaxis, :, :]
    spectra = np.fft.rfft(tapered, axis=2)
    psd_by_taper = np.abs(spectra) ** 2 / sfreq
    if n_samples > 1:
        if n_samples % 2 == 0:
            psd_by_taper[:, :, 1:-1] *= 2.0
        else:
            psd_by_taper[:, :, 1:] *= 2.0
    weights = ratios / np.sum(ratios)
    psd = np.sum(psd_by_taper * weights[np.newaxis, :, np.newaxis], axis=1)
    return psd, np.fft.rfftfreq(n_samples, d=1.0 / sfreq)


def compute_multitaper_power(
    data: np.ndarray,
    *,
    sfreq: float,
    ch_names: Sequence[str],
    clean_channels: Sequence[str] | None = None,
    rejected_channels: Sequence[str] = (),
    line_frequency_hz: float | None = None,
    apply_line_notch: bool = True,
    identity: Mapping[str, object] | None = None,
    eeg_file: str = "",
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
    time_bandwidth: float = TIME_BANDWIDTH,
    n_tapers: int = N_TAPERS,
) -> MultitaperResult:
    """Compute channel and robust-median log10 absolute band power."""
    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples).")
    if data.shape[0] != len(ch_names):
        raise ValueError("ch_names length must match data channel count.")
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be finite and positive.")
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_s and step_s must be positive.")
    if time_bandwidth <= 0 or n_tapers <= 0:
        raise ValueError("time_bandwidth and n_tapers must be positive.")
    if not np.all(np.isfinite(data)):
        raise ValueError("EEG data must contain only finite values.")

    ids = _identity(identity)
    indices, usable_channels, missing_channels = _resolve_channels(
        ch_names, clean_channels
    )
    requested_channels = (
        list(clean_channels) if clean_channels is not None else list(ch_names)
    )
    sfreq_hz = float(sfreq)
    duration_s = data.shape[1] / sfreq_hz
    nyquist_hz = sfreq_hz / 2.0
    all_bands_supported = nyquist_hz >= max(high for _, high in BANDS_HZ.values())
    warnings: list[str] = []
    if missing_channels:
        warnings.append("missing_requested_channels")
    if not all_bands_supported:
        warnings.append("nyquist_below_low_gamma_upper_edge")

    selected = data[indices] if indices else np.empty((0, data.shape[1]))
    line_ratio_before = _line_noise_ratio(
        selected, sfreq=sfreq_hz, line_frequency_hz=line_frequency_hz
    )
    if apply_line_notch and selected.size:
        processed, line_handling = _notch_line_noise(
            selected,
            sfreq=sfreq_hz,
            line_frequency_hz=line_frequency_hz,
        )
    else:
        processed = selected.copy()
        line_handling = (
            "not_applied_disabled"
            if not apply_line_notch
            else "not_applied_no_usable_channels"
        )
    line_ratio_after = _line_noise_ratio(
        processed, sfreq=sfreq_hz, line_frequency_hz=line_frequency_hz
    )

    window_samples = int(round(window_s * sfreq_hz))
    step_samples = int(round(step_s * sfreq_hz))
    if window_samples < 2 or step_samples < 1:
        raise ValueError("window_s/step_s are too small for the sampling rate.")
    if not usable_channels:
        status = "no_usable_channels"
        warnings.append(status)
        starts: list[int] = []
    elif data.shape[1] < window_samples:
        status = "short_recording"
        warnings.append(status)
        starts = []
    else:
        status = "ok"
        starts = list(range(0, data.shape[1] - window_samples + 1, step_samples))

    window_freqs = np.fft.rfftfreq(window_samples, d=1.0 / sfreq_hz)
    band_bin_counts = {
        band: int(np.sum((window_freqs >= low_hz) & (window_freqs <= high_hz)))
        for band, (low_hz, high_hz) in BANDS_HZ.items()
    }

    features: list[MultitaperFeature] = []
    epsilon = np.finfo(float).tiny
    for window_index, start in enumerate(starts):
        stop = start + window_samples
        psd, freqs = _multitaper_psd(
            processed[:, start:stop],
            sfreq=sfreq_hz,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
        )
        start_s = start / sfreq_hz
        end_s = stop / sfreq_hz
        center_s = start_s + window_s / 2.0
        for band, (low_hz, high_hz) in BANDS_HZ.items():
            powers = _integrate_band(psd, freqs, low_hz, high_hz)
            logs = np.log10(np.maximum(powers, epsilon))
            for channel_index, channel in enumerate(usable_channels):
                features.append(
                    MultitaperFeature(
                        **ids,
                        window_index=window_index,
                        window_start_s=start_s,
                        window_center_s=center_s,
                        window_end_s=end_s,
                        channel=channel,
                        aggregation="channel",
                        band=band,
                        band_low_hz=low_hz,
                        band_high_hz=high_hz,
                        absolute_power=float(powers[channel_index]),
                        absolute_log10_power=float(logs[channel_index]),
                    )
                )
            features.append(
                MultitaperFeature(
                    **ids,
                    window_index=window_index,
                    window_start_s=start_s,
                    window_center_s=center_s,
                    window_end_s=end_s,
                    channel=ROBUST_MEDIAN_CHANNEL,
                    aggregation="robust_median",
                    band=band,
                    band_low_hz=low_hz,
                    band_high_hz=high_hz,
                    absolute_power=float(np.nanmedian(powers)),
                    absolute_log10_power=float(np.nanmedian(logs)),
                )
            )

    nonfinite_rows = sum(
        not math.isfinite(feature.absolute_log10_power) for feature in features
    )
    if nonfinite_rows:
        warnings.append("nonfinite_band_power")
        if status == "ok":
            status = "spectral_qc_failed"
    residual_line_noise = (
        line_ratio_after is not None and line_ratio_after > 0.10
    )
    if residual_line_noise:
        warnings.append("residual_line_noise_high")
        if status == "ok":
            status = "spectral_qc_failed"
    spectral_qc_passed = (
        status == "ok"
        and all_bands_supported
        and nonfinite_rows == 0
        and not residual_line_noise
        and all(count > 0 for count in band_bin_counts.values())
    )
    qc = MultitaperQC(
        **ids,
        eeg_file=eeg_file,
        sfreq_hz=sfreq_hz,
        duration_s=duration_s,
        window_s=window_s,
        step_s=step_s,
        time_bandwidth=time_bandwidth,
        n_tapers=n_tapers,
        n_input_channels=data.shape[0],
        requested_channels=";".join(requested_channels),
        usable_channels=";".join(usable_channels),
        rejected_channels=";".join(rejected_channels),
        missing_channels=";".join(missing_channels),
        n_usable_channels=len(usable_channels),
        line_frequency_hz=line_frequency_hz,
        line_noise_handling=line_handling,
        line_noise_ratio_before=line_ratio_before,
        line_noise_ratio_after=line_ratio_after,
        n_windows=len(starts),
        n_channel_band_rows=len(features),
        n_nonfinite_power_rows=nonfinite_rows,
        nyquist_hz=nyquist_hz,
        frequency_resolution_hz=sfreq_hz / window_samples,
        theta_frequency_bins=band_bin_counts["theta"],
        alpha_frequency_bins=band_bin_counts["alpha"],
        beta_frequency_bins=band_bin_counts["beta"],
        low_gamma_frequency_bins=band_bin_counts["low_gamma"],
        all_bands_supported=all_bands_supported,
        spectral_qc_passed=spectral_qc_passed,
        status=status,
        warning=";".join(warnings),
    )
    return MultitaperResult(features=tuple(features), qc=qc)


def extract_multitaper_from_raw(
    raw: mne.io.BaseRaw,
    *,
    clean_channels: Sequence[str] | None = None,
    identity: Mapping[str, object] | None = None,
    eeg_file: str = "",
    line_frequency_hz: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 60.0,
    bad_channel_variance_z: float = 3.0,
    reference: str = "average",
) -> MultitaperResult:
    """Reuse repository EEG preprocessing, then compute confirmatory power."""
    nyquist = float(raw.info["sfreq"]) / 2.0
    effective_h_freq = min(h_freq, np.nextafter(nyquist, 0.0))
    prepared = preprocess_eeg(
        raw,
        l_freq=l_freq,
        h_freq=effective_h_freq,
        bad_channel_variance_z=bad_channel_variance_z,
        reference=reference,
        drop_bad_channels=True,
    )
    cleaned = prepared.raw
    resolved_line_frequency = line_frequency_hz
    if resolved_line_frequency is None:
        info_line_frequency = raw.info.get("line_freq")
        if info_line_frequency is not None and math.isfinite(float(info_line_frequency)):
            resolved_line_frequency = float(info_line_frequency)
    return compute_multitaper_power(
        cleaned.get_data(),
        sfreq=float(cleaned.info["sfreq"]),
        ch_names=cleaned.ch_names,
        clean_channels=clean_channels,
        rejected_channels=prepared.bad_channels,
        line_frequency_hz=resolved_line_frequency,
        apply_line_notch=True,
        identity=identity,
        eeg_file=eeg_file,
    )


def _write_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_multitaper_outputs(
    result: MultitaperResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    feature_fields = list(
        MultitaperFeature(
            dataset_id="",
            subject_id="",
            task="",
            condition="",
            observation_id="",
            window_index=0,
            window_start_s=0.0,
            window_center_s=1.0,
            window_end_s=2.0,
            channel="",
            aggregation="channel",
            band="theta",
            band_low_hz=4.0,
            band_high_hz=7.0,
            absolute_power=0.0,
            absolute_log10_power=0.0,
        ).to_row()
    )
    features_path = output_path / FEATURES_FILENAME
    _write_rows(
        features_path,
        (feature.to_row() for feature in result.features),
        feature_fields,
    )
    qc_path = output_path / QC_FILENAME
    qc_row = result.qc.to_row()
    _write_rows(qc_path, [qc_row], list(qc_row))
    return features_path, qc_path


def extract_multitaper_file(
    eeg_path: str | Path,
    eeg_format: str,
    output_dir: str | Path,
    *,
    clean_channels: Sequence[str] | None = None,
    identity: Mapping[str, object] | None = None,
    line_frequency_hz: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 60.0,
    bad_channel_variance_z: float = 3.0,
    reference: str = "average",
) -> tuple[Path, Path]:
    """Read EEG through the existing loader and write M3 outputs."""
    resolved_path = Path(eeg_path).expanduser().resolve()
    raw = _read_raw(resolved_path, eeg_format)
    result = extract_multitaper_from_raw(
        raw,
        clean_channels=clean_channels,
        identity=identity,
        eeg_file=str(resolved_path),
        line_frequency_hz=line_frequency_hz,
        l_freq=l_freq,
        h_freq=h_freq,
        bad_channel_variance_z=bad_channel_variance_z,
        reference=reference,
    )
    return write_multitaper_outputs(result, output_dir)


__all__ = [
    "BANDS_HZ",
    "FEATURES_FILENAME",
    "N_TAPERS",
    "QC_FILENAME",
    "ROBUST_MEDIAN_CHANNEL",
    "STEP_S",
    "TIME_BANDWIDTH",
    "WINDOW_S",
    "MultitaperFeature",
    "MultitaperQC",
    "MultitaperResult",
    "compute_multitaper_power",
    "extract_multitaper_file",
    "extract_multitaper_from_raw",
    "write_multitaper_outputs",
]
