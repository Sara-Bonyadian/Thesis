from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, welch

from ..eeg import preprocess_eeg
from ..features_core import _read_raw
from .config import TemporalCouplingConfig
from .data_audit import audit_output_path, group_output_dir
from .paths import observation_output_dir

ENVELOPE_FILENAME = "features_temporal_eeg_envelope.csv"
QC_GROUP_FILENAME = "eeg_envelope_qc.csv"
DEBUG_USER = "eeg_envelope_debug_user_window.png"
DEBUG_AUTO = "eeg_envelope_debug_auto_window.png"
TIMESERIES_PLOT = "eeg_envelope_timeseries.png"
PSD_PLOT = "eeg_psd_debug.png"

BAND_COLUMNS: dict[str, str] = {
    "theta": "theta_env",
    "alpha": "alpha_env",
    "beta": "beta_env",
}

BAND_LABELS: dict[str, str] = {
    "theta": "theta 4–8 Hz",
    "alpha": "alpha 8–13 Hz",
    "beta": "beta 13–30 Hz",
}


def _envelope_working_sfreq_hz(cfg: TemporalCouplingConfig) -> float:
    bands = cfg.temporal_coupling.eeg.bands
    max_band_hf = max(bands.theta[1], bands.alpha[1], bands.beta[1])
    # 8× the highest band edge is enough for bandpass + Hilbert; floor keeps high-fs EDF exports tractable.
    return max(250.0, float(max_band_hf) * 8.0)


def _prepare_raw_for_envelope_pipeline(raw: mne.io.BaseRaw, cfg: TemporalCouplingConfig) -> mne.io.BaseRaw:
    native_sfreq = float(raw.info["sfreq"])
    target_sfreq = _envelope_working_sfreq_hz(cfg)
    if native_sfreq <= target_sfreq:
        return raw
    prepared = raw.copy()
    prepared.resample(target_sfreq, verbose=False)
    return prepared


@dataclass(frozen=True)
class UsableObservation:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    eeg_file: Path
    eeg_format: str


@dataclass(frozen=True)
class BandEnvelopeResult:
    time_s: np.ndarray
    envelopes: dict[str, np.ndarray]
    roi_channels_used: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...]
    sfreq: float
    duration_s: float


@dataclass(frozen=True)
class EegEnvelopeQcRecord:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    eeg_file: str
    sfreq: float
    duration_s: float
    envelope_output_fs_hz: float | None
    theta_roi_channels_used: str
    alpha_roi_channels_used: str
    beta_roi_channels_used: str
    theta_nan_percent: float
    alpha_nan_percent: float
    beta_nan_percent: float
    theta_min: float
    theta_max: float
    theta_median: float
    alpha_min: float
    alpha_max: float
    alpha_median: float
    beta_min: float
    beta_max: float
    beta_median: float
    usable_eeg_envelope: bool
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "eeg_file": self.eeg_file,
            "sfreq": self.sfreq,
            "duration_s": self.duration_s,
            "envelope_output_fs_hz": self.envelope_output_fs_hz,
            "theta_roi_channels_used": self.theta_roi_channels_used,
            "alpha_roi_channels_used": self.alpha_roi_channels_used,
            "beta_roi_channels_used": self.beta_roi_channels_used,
            "theta_nan_percent": self.theta_nan_percent,
            "alpha_nan_percent": self.alpha_nan_percent,
            "beta_nan_percent": self.beta_nan_percent,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "theta_median": self.theta_median,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_median": self.alpha_median,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "beta_median": self.beta_median,
            "usable_eeg_envelope": self.usable_eeg_envelope,
            "warning": self.warning,
        }


def envelope_output_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / ENVELOPE_FILENAME


def envelope_qc_group_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / QC_GROUP_FILENAME


def _smooth_envelope(envelope: np.ndarray, *, sfreq: float, smooth_s: float) -> np.ndarray:
    if smooth_s <= 0 or envelope.size == 0:
        return envelope
    sigma_samples = max(1.0, smooth_s * sfreq / 4.0)
    return gaussian_filter1d(envelope, sigma=sigma_samples, axis=-1)


def resolve_roi_channels(ch_names: list[str], requested: tuple[str, ...]) -> tuple[list[str], list[str]]:
    lookup = {name.casefold(): name for name in ch_names}
    available: list[str] = []
    missing: list[str] = []
    for channel in requested:
        actual = lookup.get(channel.casefold())
        if actual is None:
            missing.append(channel)
        else:
            available.append(actual)
    return available, missing


def configured_roi_channels(cfg: TemporalCouplingConfig) -> tuple[str, ...]:
    rois = cfg.temporal_coupling.eeg.rois
    return tuple(dict.fromkeys((*rois.theta, *rois.alpha, *rois.beta)))


def has_usable_eeg_channels(ch_names: list[str], requested_roi: tuple[str, ...]) -> bool:
    available, _ = resolve_roi_channels(ch_names, requested_roi)
    return bool(available)


def resolve_debug_channel(
    ch_names: list[str],
    requested: str,
    roi_fallback: tuple[str, ...],
) -> tuple[str, list[str]]:
    available, missing = resolve_roi_channels(ch_names, (requested,))
    if available:
        return available[0], missing
    roi_available, roi_missing = resolve_roi_channels(ch_names, roi_fallback)
    if roi_available:
        return roi_available[0], [requested, *roi_missing]
    fallback = _fallback_roi_channels(ch_names)
    if fallback:
        return fallback[0], [requested, *roi_missing, *roi_missing]
    raise ValueError(f"no debug channel available for requested={requested!r}")


def _fallback_roi_channels(ch_names: list[str]) -> list[str]:
    """
    Some datasets (e.g., ds004511 EDF exports) use numeric EEG labels instead of
    canonical montage names (Fz/Pz/Cz...). Prefer numeric scalp channels when present.
    """
    numeric = [name for name in ch_names if name.isdigit()]
    if len(numeric) >= 16:
        return numeric
    return list(ch_names)


def compute_band_envelope(
    signal_1d: np.ndarray,
    *,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    smooth_s: float,
) -> np.ndarray:
    if signal_1d.ndim != 1:
        raise ValueError("signal_1d must be one-dimensional.")

    info = mne.create_info(ch_names=["ch"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(signal_1d[np.newaxis, :], info, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    filtered = raw.get_data()[0]
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    return _smooth_envelope(envelope, sfreq=sfreq, smooth_s=smooth_s)


def compute_band_pipeline_single_channel(
    signal_1d: np.ndarray,
    *,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    smooth_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = mne.create_info(ch_names=["ch"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(signal_1d[np.newaxis, :], info, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    filtered = raw.get_data()[0]
    envelope = _smooth_envelope(np.abs(hilbert(filtered)), sfreq=sfreq, smooth_s=smooth_s)
    return signal_1d, filtered, envelope


def _compute_roi_band_envelope(
    preprocessed: mne.io.BaseRaw,
    *,
    roi_channels: list[str],
    l_freq: float,
    h_freq: float,
    smooth_s: float,
) -> np.ndarray:
    band_raw = preprocessed.copy().pick(roi_channels)
    band_raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    sfreq = float(band_raw.info["sfreq"])
    data = band_raw.get_data()
    envelopes = np.abs(hilbert(data, axis=1))
    smoothed = _smooth_envelope(envelopes, sfreq=sfreq, smooth_s=smooth_s)
    return np.mean(smoothed, axis=0)


def extract_eeg_envelopes(
    raw: mne.io.BaseRaw,
    cfg: TemporalCouplingConfig,
) -> BandEnvelopeResult:
    tc_eeg = cfg.temporal_coupling.eeg
    bands = tc_eeg.bands
    rois = tc_eeg.rois
    smooth_s = tc_eeg.envelope_smooth_s

    preprocessed = preprocess_eeg(
        raw,
        l_freq=cfg.eeg.l_freq,
        h_freq=cfg.eeg.h_freq,
        bad_channel_variance_z=cfg.eeg.bad_channel_variance_z,
        reference=cfg.eeg.reference,
    ).raw

    sfreq = float(preprocessed.info["sfreq"])
    n_times = int(preprocessed.n_times)
    time_s = np.arange(n_times, dtype=float) / sfreq
    duration_s = float(time_s[-1]) if time_s.size else 0.0

    warnings_out: list[str] = []
    envelopes: dict[str, np.ndarray] = {}
    roi_used: dict[str, tuple[str, ...]] = {}
    band_specs = {
        "theta": (bands.theta, rois.theta),
        "alpha": (bands.alpha, rois.alpha),
        "beta": (bands.beta, rois.beta),
    }

    for band_name, (limits, requested_roi) in band_specs.items():
        available, missing = resolve_roi_channels(preprocessed.ch_names, requested_roi)
        if missing:
            warnings_out.append(
                f"warning: missing_roi_channel {band_name}={missing} (requested={list(requested_roi)})"
            )
        if not available:
            fallback = _fallback_roi_channels(preprocessed.ch_names)
            if not fallback:
                raise ValueError(f"{band_name}: no ROI channels available after matching montage.")
            available = fallback
            warnings_out.append(
                f"warning: roi_fallback_all_eeg_channels {band_name}=using_{len(available)}_channels"
            )

        roi_used[band_name] = tuple(available)
        l_freq, h_freq = limits
        envelopes[band_name] = _compute_roi_band_envelope(
            preprocessed,
            roi_channels=available,
            l_freq=l_freq,
            h_freq=h_freq,
            smooth_s=smooth_s,
        )

    return BandEnvelopeResult(
        time_s=time_s,
        envelopes=envelopes,
        roi_channels_used=roi_used,
        warnings=tuple(warnings_out),
        sfreq=sfreq,
        duration_s=duration_s,
    )


def downsample_envelope_result(
    result: BandEnvelopeResult,
    *,
    output_fs_hz: float | None,
) -> BandEnvelopeResult:
    if output_fs_hz is None or output_fs_hz <= 0 or result.time_s.size == 0:
        return result

    native_fs = 1.0 / float(np.median(np.diff(result.time_s)))
    if output_fs_hz >= native_fs:
        return result

    duration_s = float(result.time_s[-1])
    n_out = int(np.floor(duration_s * output_fs_hz)) + 1
    time_out = np.arange(n_out, dtype=float) / output_fs_hz
    time_out = time_out[time_out <= duration_s]

    downsampled = {
        band: np.interp(time_out, result.time_s, envelope)
        for band, envelope in result.envelopes.items()
    }
    return BandEnvelopeResult(
        time_s=time_out,
        envelopes=downsampled,
        roi_channels_used=result.roi_channels_used,
        warnings=result.warnings,
        sfreq=output_fs_hz,
        duration_s=duration_s,
    )


def envelopes_to_dataframe(
    *,
    dataset_id: str,
    subject_id: str,
    task: str,
    condition: str,
    observation_id: str,
    result: BandEnvelopeResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "subject_id": subject_id,
            "task": task,
            "condition": condition,
            "observation_id": observation_id,
            "time_s": result.time_s,
            "theta_env": result.envelopes["theta"],
            "alpha_env": result.envelopes["alpha"],
            "beta_env": result.envelopes["beta"],
        }
    )


def _band_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 100.0
    nan_frac = float(100.0 * np.mean(np.isnan(values)))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), nan_frac
    return float(np.min(finite)), float(np.max(finite)), float(np.median(finite)), nan_frac


def build_eeg_envelope_qc(
    obs: UsableObservation,
    native_result: BandEnvelopeResult,
    output_result: BandEnvelopeResult,
    cfg: TemporalCouplingConfig,
) -> EegEnvelopeQcRecord:
    tc_eeg = cfg.temporal_coupling.eeg
    warnings_out = list(native_result.warnings)

    stats: dict[str, tuple[float, float, float, float]] = {}
    for band in ("theta", "alpha", "beta"):
        stats[band] = _band_stats(output_result.envelopes[band])

    for band in ("theta", "alpha", "beta"):
        vmin, vmax, vmed, nan_pct = stats[band]
        if nan_pct > 0:
            warnings_out.append(f"warning: envelope_contains_nans {band}")
        if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) < 1e-9:
            warnings_out.append(f"warning: envelope_flat {band}")
        if np.isfinite(vmed) and np.isfinite(vmax) and vmed > 0 and vmax > 20.0 * vmed:
            warnings_out.append(f"warning: envelope_extreme_values {band}")

    output_fs = tc_eeg.envelope_output_fs_hz
    if output_fs and output_result.time_s.size > 0:
        expected_n = int(np.floor(native_result.duration_s * output_fs)) + 1
        if output_result.time_s.size < max(10, int(0.5 * expected_n)):
            warnings_out.append("warning: too_few_samples_after_downsampling")
        if abs(float(output_result.time_s[-1]) - native_result.duration_s) > 1.0 / max(output_fs, 1.0):
            warnings_out.append("warning: output_duration_mismatch")

    usable = (
        all(stats[b][3] == 0.0 for b in ("theta", "alpha", "beta"))
        and all(np.isfinite(stats[b][2]) for b in ("theta", "alpha", "beta"))
        and all((stats[b][1] - stats[b][0]) > 1e-9 for b in ("theta", "alpha", "beta"))
        and not any("too_few_samples" in w for w in warnings_out)
    )

    return EegEnvelopeQcRecord(
        dataset_id=obs.dataset_id,
        subject_id=obs.subject_id,
        task=obs.task,
        condition=obs.condition,
        observation_id=obs.observation_id,
        eeg_file=str(obs.eeg_file),
        sfreq=native_result.sfreq,
        duration_s=native_result.duration_s,
        envelope_output_fs_hz=output_fs,
        theta_roi_channels_used=",".join(native_result.roi_channels_used["theta"]),
        alpha_roi_channels_used=",".join(native_result.roi_channels_used["alpha"]),
        beta_roi_channels_used=",".join(native_result.roi_channels_used["beta"]),
        theta_nan_percent=stats["theta"][3],
        alpha_nan_percent=stats["alpha"][3],
        beta_nan_percent=stats["beta"][3],
        theta_min=stats["theta"][0],
        theta_max=stats["theta"][1],
        theta_median=stats["theta"][2],
        alpha_min=stats["alpha"][0],
        alpha_max=stats["alpha"][1],
        alpha_median=stats["alpha"][2],
        beta_min=stats["beta"][0],
        beta_max=stats["beta"][1],
        beta_median=stats["beta"][2],
        usable_eeg_envelope=usable,
        warning=";".join(dict.fromkeys(warnings_out)),
    )


def _resolved_user_window(cfg: TemporalCouplingConfig, duration_s: float) -> tuple[float, float] | None:
    plot_cfg = cfg.temporal_coupling.eeg.debug_plot
    if plot_cfg.start_time_s is not None and plot_cfg.end_time_s is not None:
        start_s = max(0.0, float(plot_cfg.start_time_s))
        end_s = min(duration_s, float(plot_cfg.end_time_s))
        if end_s > start_s:
            return start_s, end_s
    return None


def _auto_plot_window(duration_s: float, *, window_s: float) -> tuple[float, float]:
    if duration_s <= window_s:
        return 0.0, duration_s
    start_s = max(0.0, (duration_s - window_s) / 2.0)
    return start_s, min(duration_s, start_s + window_s)


def _downsample_for_plot(times: np.ndarray, values: np.ndarray, *, max_points: int = 2500) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_points:
        return times, values
    step = int(math.ceil(values.size / max_points))
    return times[::step], values[::step]


def _plot_envelope_debug_window(
    obs: UsableObservation,
    preprocessed: mne.io.BaseRaw,
    cfg: TemporalCouplingConfig,
    *,
    start_s: float,
    end_s: float,
    output_path: Path,
    plot_warnings: tuple[str, ...],
) -> None:
    tc_eeg = cfg.temporal_coupling.eeg
    plot_cfg = tc_eeg.debug_plot
    bands = tc_eeg.bands
    rois = tc_eeg.rois
    smooth_s = tc_eeg.envelope_smooth_s
    sfreq = float(preprocessed.info["sfreq"])
    duration_s = float(preprocessed.n_times) / sfreq

    start_s = max(0.0, min(start_s, duration_s - 0.1))
    end_s = min(duration_s, max(end_s, start_s + 0.1))
    i0 = int(start_s * sfreq)
    i1 = max(i0 + 1, int(end_s * sfreq))
    times = np.arange(i0, i1, dtype=float) / sfreq

    band_order = ("theta", "alpha", "beta")
    band_limits = {"theta": bands.theta, "alpha": bands.alpha, "beta": bands.beta}
    band_rois = {"theta": rois.theta, "alpha": rois.alpha, "beta": rois.beta}
    debug_channels = {
        "theta": plot_cfg.channels.theta,
        "alpha": plot_cfg.channels.alpha,
        "beta": plot_cfg.channels.beta,
    }

    fig, axes = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    row_labels = ("raw EEG", "bandpass-filtered", "Hilbert envelope")
    extra_warnings: list[str] = list(plot_warnings)

    for band_idx, band in enumerate(band_order):
        channel, missing = resolve_debug_channel(
            preprocessed.ch_names,
            debug_channels[band],
            band_rois[band],
        )
        if missing:
            extra_warnings.append(f"warning: missing_roi_channel {band}={missing}")

        raw_signal = preprocessed.get_data(picks=[channel])[0]
        segment = raw_signal[i0:i1]
        l_freq, h_freq = band_limits[band]
        _raw, filtered, envelope = compute_band_pipeline_single_channel(
            segment,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            smooth_s=smooth_s,
        )

        for row_idx, (values, label) in enumerate(
            zip((segment, filtered, envelope), row_labels, strict=True)
        ):
            ax = axes[band_idx * 3 + row_idx]
            plot_t, plot_v = _downsample_for_plot(times, values)
            ax.plot(plot_t, plot_v, linewidth=0.6, color="tab:blue")
            ax.set_ylabel(f"{band}\n{label}", fontsize=8)
            ax.grid(True, alpha=0.25)
            if row_idx == 0:
                ax.set_title(
                    f"{BAND_LABELS[band]} | channel: {channel}",
                    fontsize=9,
                    loc="left",
                )

    axes[-1].set_xlabel("time (s)")
    warn_txt = "; ".join(extra_warnings) if extra_warnings else "ok"
    fig.suptitle(
        f"{obs.subject_id} | {obs.task} | EEG envelope debug {start_s:.1f}-{end_s:.1f}s\n{warn_txt}",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_envelope_timeseries(
    obs: UsableObservation,
    result: BandEnvelopeResult,
    *,
    output_path: Path,
) -> None:
    times = result.time_s
    plot_t, _ = _downsample_for_plot(times, result.envelopes["theta"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = {"theta": "tab:purple", "alpha": "tab:green", "beta": "tab:orange"}
    for ax, band in zip(axes, ("theta", "alpha", "beta"), strict=True):
        _, plot_v = _downsample_for_plot(times, result.envelopes[band])
        ax.plot(plot_t, plot_v, linewidth=0.7, color=colors[band], label=f"{band}_env")
        ax.set_ylabel(f"{band}_env")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fs_note = f" @ {result.sfreq:.1f} Hz" if result.sfreq < 500 else ""
    fig.suptitle(
        f"{obs.subject_id} | ROI-averaged envelopes (full recording{fs_note})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_eeg_psd_debug(
    obs: UsableObservation,
    preprocessed: mne.io.BaseRaw,
    cfg: TemporalCouplingConfig,
    *,
    output_path: Path,
) -> None:
    tc_eeg = cfg.temporal_coupling.eeg
    plot_cfg = tc_eeg.debug_plot
    bands = tc_eeg.bands
    rois = tc_eeg.rois
    sfreq = float(preprocessed.info["sfreq"])

    band_order = ("theta", "alpha", "beta")
    debug_channels = {
        "theta": plot_cfg.channels.theta,
        "alpha": plot_cfg.channels.alpha,
        "beta": plot_cfg.channels.beta,
    }
    band_rois = {"theta": rois.theta, "alpha": rois.alpha, "beta": rois.beta}

    fig, ax = plt.subplots(figsize=(10, 5))
    for band in band_order:
        channel, _ = resolve_debug_channel(
            preprocessed.ch_names,
            debug_channels[band],
            band_rois[band],
        )
        signal = preprocessed.get_data(picks=[channel])[0]
        freqs, psd = welch(signal, fs=sfreq, nperseg=min(2048, signal.size))
        ax.semilogy(freqs, psd, linewidth=0.9, label=channel)

    band_colors = {"theta": "gold", "alpha": "lightgreen", "beta": "lightsalmon"}
    for band, limits in (("theta", bands.theta), ("alpha", bands.alpha), ("beta", bands.beta)):
        ax.axvspan(limits[0], limits[1], color=band_colors[band], alpha=0.25, label=f"{band} band")

    ax.set_xlim(0, min(45, sfreq / 2))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(f"{obs.subject_id} | PSD sanity check (debug channels)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def write_debug_plots(
    obs: UsableObservation,
    preprocessed: mne.io.BaseRaw,
    native_result: BandEnvelopeResult,
    output_result: BandEnvelopeResult,
    cfg: TemporalCouplingConfig,
    out_dir: Path,
    *,
    qc_warnings: str,
) -> list[Path]:
    plot_cfg = cfg.temporal_coupling.eeg.debug_plot
    if not plot_cfg.enabled:
        return []

    written: list[Path] = []
    plot_warnings = tuple(qc_warnings.split(";")) if qc_warnings else ()

    user_window = _resolved_user_window(cfg, native_result.duration_s)
    if user_window:
        path = out_dir / DEBUG_USER
        _plot_envelope_debug_window(
            obs,
            preprocessed,
            cfg,
            start_s=user_window[0],
            end_s=user_window[1],
            output_path=path,
            plot_warnings=plot_warnings,
        )
        written.append(path)

    if plot_cfg.also_auto_window:
        auto_start, auto_end = _auto_plot_window(
            native_result.duration_s,
            window_s=plot_cfg.auto_window_s,
        )
        if user_window is None or (auto_start, auto_end) != user_window:
            path = out_dir / DEBUG_AUTO
            _plot_envelope_debug_window(
                obs,
                preprocessed,
                cfg,
                start_s=auto_start,
                end_s=auto_end,
                output_path=path,
                plot_warnings=plot_warnings,
            )
            written.append(path)

    ts_path = out_dir / TIMESERIES_PLOT
    plot_envelope_timeseries(obs, output_result, output_path=ts_path)
    written.append(ts_path)

    if plot_cfg.save_psd:
        psd_path = out_dir / PSD_PLOT
        plot_eeg_psd_debug(obs, preprocessed, cfg, output_path=psd_path)
        written.append(psd_path)

    return written


def load_usable_observations(cfg: TemporalCouplingConfig) -> list[UsableObservation]:
    audit_path = audit_output_path(cfg)
    if audit_path.is_file():
        audit_df = pd.read_csv(audit_path)
        if audit_df.empty:
            return []
        usable_mask = audit_df["usable"].astype(str).str.lower().isin({"true", "1", "yes"})
        rows = audit_df.loc[usable_mask]
        observations: list[UsableObservation] = []
        for row in rows.itertuples(index=False):
            eeg_file = Path(str(row.eeg_file))
            if not eeg_file.is_file():
                warnings.warn(
                    f"[temporal_coupling] skipping {row.subject_id}: audit eeg_file missing ({eeg_file}).",
                    stacklevel=2,
                )
                continue
            observations.append(
                UsableObservation(
                    dataset_id=str(row.dataset_id),
                    subject_id=str(row.subject_id),
                    task=str(row.task),
                    condition=str(getattr(row, "condition", row.task)),
                    observation_id=str(row.observation_id),
                    eeg_file=eeg_file,
                    eeg_format=str(getattr(row, "eeg_format", "eeglab")),
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
        if not obs.eeg_path.is_file():
            warnings.warn(
                f"[temporal_coupling] skipping {obs.subject_id} condition={obs.condition_label}: EEG file missing.",
                stacklevel=2,
            )
            continue
        observations.append(
            UsableObservation(
                dataset_id=obs.dataset_id,
                subject_id=obs.subject_id,
                task=obs.task_label,
                condition=obs.condition_label,
                observation_id=obs.observation_id,
                eeg_file=obs.eeg_path,
                eeg_format=obs.eeg_format,
            )
        )
    return observations


def _preprocess_raw(raw: mne.io.BaseRaw, cfg: TemporalCouplingConfig) -> mne.io.BaseRaw:
    return preprocess_eeg(
        raw,
        l_freq=cfg.eeg.l_freq,
        h_freq=cfg.eeg.h_freq,
        bad_channel_variance_z=cfg.eeg.bad_channel_variance_z,
        reference=cfg.eeg.reference,
    ).raw


def run_stage1a(cfg: TemporalCouplingConfig) -> list[Path]:
    observations = load_usable_observations(cfg)
    if not observations:
        print("[temporal_coupling] stage=1a: no usable observations to process.")
        return []

    written: list[Path] = []
    qc_records: list[EegEnvelopeQcRecord] = []
    n_ok = 0
    n_total = len(observations)
    print(f"[temporal_coupling] stage=1a: processing {n_total} observations")

    for idx, obs in enumerate(observations, start=1):
        out_dir = observation_output_dir(cfg, obs.observation_id)
        out_path = out_dir / ENVELOPE_FILENAME
        try:
            print(
                f"[temporal_coupling] stage=1a [{idx}/{n_total}] "
                f"{obs.observation_id} task={obs.task}: loading {obs.eeg_file.name}",
                flush=True,
            )
            raw = _read_raw(obs.eeg_file, obs.eeg_format)
            native_sfreq = float(raw.info["sfreq"])
            working = _prepare_raw_for_envelope_pipeline(raw, cfg)
            working_sfreq = float(working.info["sfreq"])
            if working_sfreq < native_sfreq:
                print(
                    f"[temporal_coupling] stage=1a [{idx}/{n_total}] "
                    f"downsampled {native_sfreq:.0f} -> {working_sfreq:.0f} Hz before envelope extraction",
                    flush=True,
                )
            native_result = extract_eeg_envelopes(working, cfg)
            output_result = downsample_envelope_result(
                native_result,
                output_fs_hz=cfg.temporal_coupling.eeg.envelope_output_fs_hz,
            )
            for message in native_result.warnings:
                warnings.warn(f"[temporal_coupling] {obs.subject_id} {message}", stacklevel=2)

            preprocessed = _preprocess_raw(working, cfg)
            qc = build_eeg_envelope_qc(obs, native_result, output_result, cfg)
            qc_records.append(qc)
            if qc.warning:
                warnings.warn(f"[temporal_coupling] {obs.subject_id} EEG envelope QC: {qc.warning}", stacklevel=2)

            df = envelopes_to_dataframe(
                dataset_id=obs.dataset_id,
                subject_id=obs.subject_id,
                task=obs.task,
                condition=obs.condition,
                observation_id=obs.observation_id,
                result=output_result,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            written.append(out_path)
            n_ok += 1
            duration_s = float(df["time_s"].iloc[-1]) if not df.empty else 0.0
            print(
                f"[temporal_coupling] stage=1a {obs.subject_id} task={obs.task}: "
                f"rows={len(df)} duration_s={duration_s:.1f} usable={qc.usable_eeg_envelope} -> {out_path}"
            )
            try:
                plot_paths = write_debug_plots(
                    obs,
                    preprocessed,
                    native_result,
                    output_result,
                    cfg,
                    out_dir,
                    qc_warnings=qc.warning,
                )
                for plot_path in plot_paths:
                    print(f"[temporal_coupling]   eeg_plot={plot_path}")
            except Exception as plot_exc:
                warnings.warn(
                    f"[temporal_coupling] stage=1a {obs.subject_id} task={obs.task}: "
                    f"debug plots skipped: {plot_exc}",
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=1a skipping {obs.subject_id} task={obs.task}: {exc}",
                stacklevel=2,
            )

    if qc_records:
        qc_path = envelope_qc_group_path(cfg)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record.to_row() for record in qc_records]).to_csv(qc_path, index=False)
        print(f"[temporal_coupling] stage=1a wrote group QC -> {qc_path}")

    print(f"[temporal_coupling] stage=1a summary: wrote={n_ok}/{len(observations)}")
    return written
