from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from ..eeg import preprocess_eeg
from ..features_core import _read_raw
from ..output_layout import safe_subject_dir_name
from .config import TemporalCouplingConfig
from .data_audit import audit_output_path

ENVELOPE_FILENAME = "features_temporal_eeg_envelope.csv"

BAND_COLUMNS: dict[str, str] = {
    "theta": "theta_env",
    "alpha": "alpha_env",
    "beta": "beta_env",
}


@dataclass(frozen=True)
class UsableObservation:
    dataset_id: str
    subject_id: str
    task: str
    observation_id: str
    eeg_file: Path


@dataclass(frozen=True)
class BandEnvelopeResult:
    time_s: np.ndarray
    envelopes: dict[str, np.ndarray]
    warnings: tuple[str, ...]


def subject_output_dir(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return Path(cfg.paths.out_root) / cfg.dataset_id / safe_subject_dir_name(subject_id)


def envelope_output_path(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return subject_output_dir(cfg, subject_id) / ENVELOPE_FILENAME


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


def compute_band_envelope(
    signal_1d: np.ndarray,
    *,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    smooth_s: float,
) -> np.ndarray:
    """
    Bandpass-filter one channel, Hilbert envelope, and smooth.
    Intended for unit tests and low-level reuse.
    """
    if signal_1d.ndim != 1:
        raise ValueError("signal_1d must be one-dimensional.")

    info = mne.create_info(ch_names=["ch"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(signal_1d[np.newaxis, :], info, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    filtered = raw.get_data()[0]
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    return _smooth_envelope(envelope, sfreq=sfreq, smooth_s=smooth_s)


def _compute_roi_band_envelope(
    preprocessed: mne.io.BaseRaw,
    *,
    roi_channels: list[str],
    l_freq: float,
    h_freq: float,
    smooth_s: float,
) -> np.ndarray:
    band_raw = preprocessed.copy().pick_channels(roi_channels)
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

    warnings_out: list[str] = []
    envelopes: dict[str, np.ndarray] = {}
    band_specs = {
        "theta": (bands.theta, rois.theta),
        "alpha": (bands.alpha, rois.alpha),
        "beta": (bands.beta, rois.beta),
    }

    for band_name, (limits, requested_roi) in band_specs.items():
        available, missing = resolve_roi_channels(preprocessed.ch_names, requested_roi)
        if missing:
            warnings_out.append(
                f"{band_name}: missing ROI channels {missing} (requested={list(requested_roi)})"
            )
        if not available:
            raise ValueError(f"{band_name}: no ROI channels available after matching montage.")

        l_freq, h_freq = limits
        envelopes[band_name] = _compute_roi_band_envelope(
            preprocessed,
            roi_channels=available,
            l_freq=l_freq,
            h_freq=h_freq,
            smooth_s=smooth_s,
        )

    return BandEnvelopeResult(time_s=time_s, envelopes=envelopes, warnings=tuple(warnings_out))


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
        warnings=result.warnings,
    )


def envelopes_to_dataframe(
    *,
    dataset_id: str,
    subject_id: str,
    task: str,
    observation_id: str,
    result: BandEnvelopeResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "subject_id": subject_id,
            "task": task,
            "observation_id": observation_id,
            "time_s": result.time_s,
            "theta_env": result.envelopes["theta"],
            "alpha_env": result.envelopes["alpha"],
            "beta_env": result.envelopes["beta"],
        }
    )


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
                    observation_id=str(row.observation_id),
                    eeg_file=eeg_file,
                )
            )
        return observations

    warnings.warn(
        f"[temporal_coupling] {audit_path} not found; using all configured subject/task pairs. "
        "Run --stage 0 first for audit gating.",
        stacklevel=2,
    )
    from .data_audit import _planned_subject_tasks, _resolve_dataset_root, _find_signal_file

    dataset_root = _resolve_dataset_root(cfg.paths.raw_root, cfg.dataset_id)
    observations: list[UsableObservation] = []
    for subject_id, task in _planned_subject_tasks(cfg):
        eeg_path = None if dataset_root is None else _find_signal_file(dataset_root, subject_id, task, "eeg")
        if eeg_path is None or not eeg_path.is_file():
            warnings.warn(
                f"[temporal_coupling] skipping {subject_id} task={task}: EEG file missing.",
                stacklevel=2,
            )
            continue
        observations.append(
            UsableObservation(
                dataset_id=cfg.dataset_id,
                subject_id=subject_id,
                task=task,
                observation_id=f"{cfg.dataset_id}-{subject_id}-task-{task}",
                eeg_file=eeg_path,
            )
        )
    return observations


def run_stage1a(cfg: TemporalCouplingConfig) -> list[Path]:
    observations = load_usable_observations(cfg)
    if not observations:
        print("[temporal_coupling] stage=1a: no usable observations to process.")
        return []

    written: list[Path] = []
    n_ok = 0
    for obs in observations:
        out_path = envelope_output_path(cfg, obs.subject_id)
        try:
            raw = _read_raw(obs.eeg_file, "eeglab")
            result = extract_eeg_envelopes(raw, cfg)
            result = downsample_envelope_result(
                result,
                output_fs_hz=cfg.temporal_coupling.eeg.envelope_output_fs_hz,
            )
            for message in result.warnings:
                warnings.warn(f"[temporal_coupling] {obs.subject_id} {message}", stacklevel=2)

            df = envelopes_to_dataframe(
                dataset_id=obs.dataset_id,
                subject_id=obs.subject_id,
                task=obs.task,
                observation_id=obs.observation_id,
                result=result,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            written.append(out_path)
            n_ok += 1
            duration_s = float(df["time_s"].iloc[-1]) if not df.empty else 0.0
            print(
                f"[temporal_coupling] stage=1a {obs.subject_id} task={obs.task}: "
                f"rows={len(df)} duration_s={duration_s:.1f} -> {out_path}"
            )
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=1a skipping {obs.subject_id} task={obs.task}: {exc}",
                stacklevel=2,
            )

    print(f"[temporal_coupling] stage=1a summary: wrote={n_ok}/{len(observations)}")
    return written
