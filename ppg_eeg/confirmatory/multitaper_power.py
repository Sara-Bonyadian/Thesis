"""DPSS multitaper EEG band-power extraction for confirmatory analysis."""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mne
import numpy as np
from scipy.signal import filtfilt, iirnotch
from scipy.signal.windows import dpss

from ..core_eeg_ppg.eeg import preprocess_eeg
from ..core_eeg_ppg.features_core import _read_raw
from ..core_eeg_ppg.output_layout import safe_subject_dir_name
from ..datasets import CanonicalObservation
from .reason_codes import (
    INSUFFICIENT_DURATION,
    INPUT_DISCOVERY_FAILED,
    MISSING_REQUIRED_BAND,
    MISSING_REQUIRED_MODALITY,
    STRUCTURED_NC_FIELDS,
    attach_structured_reason,
    map_exclusion_to_reason_code,
)
from .parallel_util import (
    atomic_write_csv_rows,
    atomic_write_json,
    configure_blas_threads,
    estimate_c1a_mem_per_worker_gb,
    file_identity,
    prepare_obs_checkpoint_dir,
    report_progress,
    resolve_c1a_n_jobs,
    total_ram_bytes,
)

FEATURES_FILENAME = "features_multitaper_power.csv"
QC_FILENAME = "multitaper_qc.csv"
WINDOW_S = 2.0
STEP_S = 1.0
TIME_BANDWIDTH = 3.0
N_TAPERS = 5
ROBUST_MEDIAN_CHANNEL = "__robust_median__"
CHECKPOINT_SCHEMA_VERSION = "c1a_checkpoint_v1"
CHECKPOINT_DIRNAME = "_obs_checkpoints"
COMPLETE_MARKER_FILENAME = "C1a_COMPLETE.json"

# Process-local DPSS cache: identical tapers reused across equal window lengths.
_DPSS_CACHE: dict[tuple[int, float, int], tuple[np.ndarray, np.ndarray]] = {}

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


def _dpss_tapers(
    n_samples: int,
    *,
    time_bandwidth: float,
    n_tapers: int,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return DPSS tapers/ratios, optionally cached for bit-identical reuse."""
    key = (int(n_samples), float(time_bandwidth), int(n_tapers))
    if cache and key in _DPSS_CACHE:
        return _DPSS_CACHE[key]
    tapers, ratios = dpss(
        int(n_samples),
        NW=float(time_bandwidth),
        Kmax=int(n_tapers),
        sym=False,
        norm=2,
        return_ratios=True,
    )
    if cache:
        _DPSS_CACHE[key] = (tapers, ratios)
    return tapers, ratios


def clear_dpss_cache() -> None:
    """Clear the process-local DPSS cache (tests / isolation)."""
    _DPSS_CACHE.clear()


def _multitaper_psd(
    window: np.ndarray,
    *,
    sfreq: float,
    time_bandwidth: float,
    n_tapers: int,
    tapers: np.ndarray | None = None,
    ratios: np.ndarray | None = None,
    cache_dpss: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = window.shape[1]
    if tapers is None or ratios is None:
        tapers, ratios = _dpss_tapers(
            n_samples,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            cache=cache_dpss,
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
    cache_dpss: bool = True,
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

    tapers: np.ndarray | None = None
    ratios: np.ndarray | None = None
    if starts:
        tapers, ratios = _dpss_tapers(
            window_samples,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            cache=cache_dpss,
        )

    features: list[MultitaperFeature] = []
    epsilon = np.finfo(float).tiny
    for window_index, start in enumerate(starts):
        stop = start + window_samples
        psd, freqs = _multitaper_psd(
            processed[:, start:stop],
            sfreq=sfreq_hz,
            time_bandwidth=time_bandwidth,
            n_tapers=n_tapers,
            tapers=tapers,
            ratios=ratios,
            cache_dpss=cache_dpss,
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
    cache_dpss: bool = True,
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
        cache_dpss=cache_dpss,
    )


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
    atomic_write_csv_rows(
        features_path,
        [feature.to_row() for feature in result.features],
        feature_fields,
    )
    qc_path = output_path / QC_FILENAME
    qc_row = _enrich_c1a_qc_row(result.qc.to_row())
    atomic_write_csv_rows(
        qc_path,
        [qc_row],
        list(dict.fromkeys([*qc_row.keys(), *STRUCTURED_NC_FIELDS])),
    )
    return features_path, qc_path


def _enrich_c1a_qc_row(row: Mapping[str, object]) -> dict[str, object]:
    """Attach StructuredReason fields at the C1a QC decision site."""
    status = str(row.get("status") or "").strip().casefold()
    warning = str(row.get("warning") or "").strip()
    eligible = status == "ok" and bool(row.get("spectral_qc_passed", True))
    if status in {"", "ok"} and eligible:
        exclusion = ""
    elif status in {"no_usable_channels", "short_recording", "spectral_qc_failed"}:
        exclusion = status
    elif warning:
        exclusion = warning.split(";")[0].strip() or status or INPUT_DISCOVERY_FAILED
    else:
        exclusion = status or INPUT_DISCOVERY_FAILED
    code = map_exclusion_to_reason_code(exclusion)
    if not code:
        if "channel" in exclusion:
            code = MISSING_REQUIRED_MODALITY
        elif "short" in exclusion or "duration" in exclusion:
            code = INSUFFICIENT_DURATION
        elif "band" in exclusion or "nyquist" in exclusion:
            code = MISSING_REQUIRED_BAND
        elif exclusion:
            code = INPUT_DISCOVERY_FAILED
    return attach_structured_reason(
        {
            **dict(row),
            "participant_id": str(row.get("subject_id") or row.get("participant_id") or ""),
            "session_id": str(row.get("session_id") or ""),
            "exclusion_reason": exclusion,
            "reason_code": code,
        },
        stage="C1a",
        eligible=eligible,
        reason_code=code,
        reason=exclusion.replace("_", " ") if exclusion else "",
        required_evidence=(
            "usable EEG channels with spectral QC pass and supported bands"
            if not eligible
            else ""
        ),
        observed_evidence=(
            f"status={status}; usable_channels={row.get('n_usable_channels')}; "
            f"spectral_qc_passed={row.get('spectral_qc_passed')}; warning={warning}"
            if not eligible
            else ""
        ),
    )


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
    cache_dpss: bool = True,
) -> tuple[Path, Path]:
    """Read EEG through the existing loader and write M3 outputs.

    Use ``Path.absolute()`` rather than ``resolve()`` so git-annex symlinks
    (OpenNeuro BrainVision .vhdr/.vmrk/.eeg triples) are not followed. Resolving
    the symlink makes MNE look for companion files beside the content-hashed
    annex object name, which fails even when all three files are present.
    """
    # absolute() keeps BIDS/annex symlink paths intact; resolve() does not.
    load_path = Path(eeg_path).expanduser().absolute()
    raw = _read_raw(load_path, eeg_format)
    result = extract_multitaper_from_raw(
        raw,
        clean_channels=clean_channels,
        identity=identity,
        eeg_file=str(load_path),
        line_frequency_hz=line_frequency_hz,
        l_freq=l_freq,
        h_freq=h_freq,
        bad_channel_variance_z=bad_channel_variance_z,
        reference=reference,
        cache_dpss=cache_dpss,
    )
    return write_multitaper_outputs(result, output_dir)


def _c1a_param_fingerprint(
    *,
    line_frequency_hz: float | None,
    l_freq: float,
    h_freq: float,
    bad_channel_variance_z: float,
    reference: str,
    window_s: float,
    step_s: float,
    time_bandwidth: float,
    n_tapers: int,
    cache_dpss: bool,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "line_frequency_hz": line_frequency_hz,
        "l_freq": float(l_freq),
        "h_freq": float(h_freq),
        "bad_channel_variance_z": float(bad_channel_variance_z),
        "reference": str(reference),
        "window_s": float(window_s),
        "step_s": float(step_s),
        "time_bandwidth": float(time_bandwidth),
        "n_tapers": int(n_tapers),
        "bands_hz": {name: list(bounds) for name, bounds in BANDS_HZ.items()},
        "cache_dpss": bool(cache_dpss),
        "features_filename": FEATURES_FILENAME,
        "qc_filename": QC_FILENAME,
    }


def _c1a_obs_fingerprint(
    obs: CanonicalObservation,
    *,
    params: Mapping[str, object],
) -> dict[str, object]:
    eeg_path = Path(obs.eeg_path).expanduser().absolute()
    return {
        **dict(params),
        "observation_id": obs.observation_id,
        "eeg_format": obs.eeg_format,
        "eeg_file": file_identity(eeg_path),
    }


def _checkpoint_path(checkpoint_dir: Path, observation_id: str) -> Path:
    digest = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:32]
    return checkpoint_dir / f"obs_{digest}.json"


def _required_c1a_outputs(obs_dir: Path) -> tuple[Path, Path]:
    return obs_dir / FEATURES_FILENAME, obs_dir / QC_FILENAME


def _c1a_outputs_complete(obs_dir: Path) -> bool:
    features, qc = _required_c1a_outputs(obs_dir)
    return (
        features.is_file()
        and qc.is_file()
        and features.stat().st_size > 0
        and qc.stat().st_size > 0
    )


def _load_c1a_checkpoint(
    checkpoint_dir: Path,
    *,
    observation_id: str,
    fingerprint: Mapping[str, object],
    obs_dir: Path,
) -> bool:
    path = _checkpoint_path(checkpoint_dir, observation_id)
    if not path.is_file() or not _c1a_outputs_complete(obs_dir):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return False
    if payload.get("fingerprint") != dict(fingerprint):
        return False
    if payload.get("status") != "ok":
        return False
    return True


def _write_c1a_checkpoint(
    checkpoint_dir: Path,
    *,
    observation_id: str,
    fingerprint: Mapping[str, object],
    obs_dir: Path,
) -> None:
    features, qc = _required_c1a_outputs(obs_dir)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "observation_id": observation_id,
        "status": "ok",
        "fingerprint": dict(fingerprint),
        "outputs": {
            "features": str(features),
            "qc": str(qc),
            "features_size": int(features.stat().st_size),
            "qc_size": int(qc.stat().st_size),
        },
    }
    atomic_write_json(_checkpoint_path(checkpoint_dir, observation_id), payload)


def _process_one_c1a_observation(
    obs: CanonicalObservation,
    obs_dir: Path,
    *,
    line_frequency_hz: float | None,
    l_freq: float,
    h_freq: float,
    bad_channel_variance_z: float,
    reference: str,
    cache_dpss: bool,
) -> str:
    configure_blas_threads(1)
    obs_dir.mkdir(parents=True, exist_ok=True)
    extract_multitaper_file(
        obs.eeg_path,
        obs.eeg_format,
        obs_dir,
        identity={
            "dataset_id": obs.dataset_id,
            "subject_id": obs.subject_id,
            "task": obs.task_label,
            "condition": obs.condition_label,
            "observation_id": obs.observation_id,
        },
        line_frequency_hz=line_frequency_hz,
        l_freq=l_freq,
        h_freq=h_freq,
        bad_channel_variance_z=bad_channel_variance_z,
        reference=reference,
        cache_dpss=cache_dpss,
    )
    return obs.observation_id


def _worker_c1a(
    payload: tuple[
        int,
        CanonicalObservation,
        str,
        float | None,
        float,
        float,
        float,
        str,
        bool,
    ],
) -> tuple[int, str | None, str | None]:
    (
        index,
        obs,
        obs_dir_s,
        line_frequency_hz,
        l_freq,
        h_freq,
        bad_channel_variance_z,
        reference,
        cache_dpss,
    ) = payload
    try:
        _process_one_c1a_observation(
            obs,
            Path(obs_dir_s),
            line_frequency_hz=line_frequency_hz,
            l_freq=l_freq,
            h_freq=h_freq,
            bad_channel_variance_z=bad_channel_variance_z,
            reference=reference,
            cache_dpss=cache_dpss,
        )
        return index, obs.observation_id, None
    except Exception as exc:  # noqa: BLE001
        return index, None, f"{obs.observation_id}: {type(exc).__name__}: {exc}"


def run_confirmatory_multitaper(
    observations: Sequence[CanonicalObservation],
    stage_root: str | Path,
    *,
    line_frequency_hz: float | None = None,
    l_freq: float = 1.0,
    h_freq: float = 60.0,
    bad_channel_variance_z: float = 3.0,
    reference: str = "average",
    n_jobs: int | None = -1,
    progress: bool = True,
    cache_dpss: bool = True,
) -> dict[str, object]:
    """Extract multitaper features for all observations (serial or parallel)."""
    output_path = Path(stage_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    complete_marker = output_path / COMPLETE_MARKER_FILENAME
    if complete_marker.is_file():
        try:
            complete_marker.unlink()
        except OSError:
            pass

    obs_list = sorted(observations, key=lambda o: o.observation_id)
    params = _c1a_param_fingerprint(
        line_frequency_hz=line_frequency_hz,
        l_freq=l_freq,
        h_freq=h_freq,
        bad_channel_variance_z=bad_channel_variance_z,
        reference=reference,
        window_s=WINDOW_S,
        step_s=STEP_S,
        time_bandwidth=TIME_BANDWIDTH,
        n_tapers=N_TAPERS,
        cache_dpss=cache_dpss,
    )
    fingerprints = [
        _c1a_obs_fingerprint(obs, params=params) for obs in obs_list
    ]
    obs_dirs = [
        output_path / safe_subject_dir_name(obs.observation_id) for obs in obs_list
    ]
    unit_keys = [obs.observation_id for obs in obs_list]
    ckpt_dir = prepare_obs_checkpoint_dir(
        output_path / CHECKPOINT_DIRNAME,
        manifest={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": "C1a",
            "unit_keys": unit_keys,
            "n_observations": len(unit_keys),
            "params": params,
        },
    )

    workers = resolve_c1a_n_jobs(
        n_jobs,
        eeg_paths=[obs.eeg_path for obs in obs_list],
    )
    mem_est = estimate_c1a_mem_per_worker_gb([obs.eeg_path for obs in obs_list])
    total_ram = total_ram_bytes()
    pending: list[int] = []
    written: list[str] = []
    errors: list[str] = []
    for index, obs in enumerate(obs_list):
        if _load_c1a_checkpoint(
            ckpt_dir,
            observation_id=obs.observation_id,
            fingerprint=fingerprints[index],
            obs_dir=obs_dirs[index],
        ):
            written.append(obs.observation_id)
        else:
            pending.append(index)

    total = len(obs_list)
    done = total - len(pending)
    start_time = time.perf_counter()
    last_report = 0.0
    if progress:
        ram_gb = None if total_ram is None else total_ram / float(1024**3)
        print(
            f"[confirmatory] C1a multitaper: {total} observations "
            f"(n_jobs={workers}, pending={len(pending)}, resumed={done}, "
            f"est_mem/worker≈{mem_est:.1f} GB"
            + (f", host_RAM≈{ram_gb:.1f} GB" if ram_gb is not None else "")
            + ")...",
            flush=True,
        )
        if workers == 1 and mem_est >= 4.0:
            print(
                "[confirmatory] C1a: large EEG payloads detected — forcing serial "
                "workers to avoid out-of-memory crashes. Use --n-jobs 1 explicitly "
                "on laptops; raise n_jobs only if you have ample free RAM.",
                flush=True,
            )
        if done:
            last_report = report_progress(
                label="C1a",
                done=done,
                total=total,
                start_time=start_time,
                last_report=last_report,
                force=True,
            )

    def _mark_ok(index: int) -> None:
        nonlocal done, last_report
        obs = obs_list[index]
        _write_c1a_checkpoint(
            ckpt_dir,
            observation_id=obs.observation_id,
            fingerprint=fingerprints[index],
            obs_dir=obs_dirs[index],
        )
        written.append(obs.observation_id)
        done += 1
        if progress:
            last_report = report_progress(
                label="C1a",
                done=done,
                total=total,
                start_time=start_time,
                last_report=last_report,
                force=(done == total),
            )

    if pending:
        if workers == 1 or len(pending) == 1:
            configure_blas_threads(1)
            for index in pending:
                try:
                    _process_one_c1a_observation(
                        obs_list[index],
                        obs_dirs[index],
                        line_frequency_hz=line_frequency_hz,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        bad_channel_variance_z=bad_channel_variance_z,
                        reference=reference,
                        cache_dpss=cache_dpss,
                    )
                    _mark_ok(index)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{obs_list[index].observation_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    done += 1
                    if progress:
                        last_report = report_progress(
                            label="C1a",
                            done=done,
                            total=total,
                            start_time=start_time,
                            last_report=last_report,
                            force=(done == total),
                        )
        else:
            payloads = [
                (
                    index,
                    obs_list[index],
                    str(obs_dirs[index]),
                    line_frequency_hz,
                    float(l_freq),
                    float(h_freq),
                    float(bad_channel_variance_z),
                    str(reference),
                    bool(cache_dpss),
                )
                for index in pending
            ]
            max_workers = min(workers, len(pending))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=configure_blas_threads,
                initargs=(1,),
            ) as executor:
                futures = {
                    executor.submit(_worker_c1a, payload): payload[0]
                    for payload in payloads
                }
                for future in as_completed(futures):
                    index, _ok_id, err = future.result()
                    if err is None:
                        _mark_ok(index)
                    else:
                        errors.append(err)
                        done += 1
                        if progress:
                            last_report = report_progress(
                                label="C1a",
                                done=done,
                                total=total,
                                start_time=start_time,
                                last_report=last_report,
                                force=(done == total),
                            )

    written_sorted = sorted(set(written))
    if not written_sorted:
        raise RuntimeError(f"C1a wrote no observations. Errors: {errors[:5]}")

    wall_time_s = time.perf_counter() - start_time
    if len(written_sorted) == total and not errors:
        atomic_write_json(
            complete_marker,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage": "C1a",
                "n_observations": total,
                "observation_ids": written_sorted,
                "n_jobs": workers,
                "wall_time_s": wall_time_s,
            },
        )

    return {
        "n_ok": len(written_sorted),
        "n_error": len(errors),
        "errors": errors[:20],
        "n_jobs": workers,
        "wall_time_s": wall_time_s,
        "complete": complete_marker.is_file(),
    }


__all__ = [
    "BANDS_HZ",
    "CHECKPOINT_DIRNAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPLETE_MARKER_FILENAME",
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
    "clear_dpss_cache",
    "compute_multitaper_power",
    "extract_multitaper_file",
    "extract_multitaper_from_raw",
    "run_confirmatory_multitaper",
    "write_multitaper_outputs",
]
