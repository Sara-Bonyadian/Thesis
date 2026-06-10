"""Stage 1c: align EEG envelopes and cardiac time series, resample, z-score."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..output_layout import safe_subject_dir_name
from .cardiac_timeseries import CARDIAC_FILENAME
from .config import TemporalCouplingConfig
from .data_audit import audit_output_path
from .eeg_envelope import ENVELOPE_FILENAME, UsableObservation

ALIGNED_FILENAME = "features_temporal_aligned.csv"

EEG_ENVELOPE_COLUMNS = ("theta_env", "alpha_env", "beta_env")
CARDIAC_COLUMNS = ("hr", "rmssd", "sdnn", "mean_rr")
RAW_COLUMNS = (*CARDIAC_COLUMNS, *EEG_ENVELOPE_COLUMNS)
Z_COLUMNS = (
    "hr_z",
    "rmssd_z",
    "sdnn_z",
    "mean_rr_z",
    "theta_env_z",
    "alpha_env_z",
    "beta_env_z",
)


@dataclass(frozen=True)
class OverlapRange:
    start_s: float
    end_s: float


def subject_output_dir(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return Path(cfg.paths.out_root) / cfg.dataset_id / safe_subject_dir_name(subject_id)


def aligned_output_path(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return subject_output_dir(cfg, subject_id) / ALIGNED_FILENAME


def envelope_input_path(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return subject_output_dir(cfg, subject_id) / ENVELOPE_FILENAME


def cardiac_input_path(cfg: TemporalCouplingConfig, subject_id: str) -> Path:
    return subject_output_dir(cfg, subject_id) / CARDIAC_FILENAME


def overlap_time_range(eeg_df: pd.DataFrame, cardiac_df: pd.DataFrame) -> OverlapRange:
    eeg_start = float(eeg_df["time_s"].min())
    eeg_end = float(eeg_df["time_s"].max())
    car_start = float(cardiac_df["time_s"].min())
    car_end = float(cardiac_df["time_s"].max())

    start_s = max(eeg_start, car_start)
    end_s = min(eeg_end, car_end)
    if end_s <= start_s:
        raise ValueError("no_temporal_overlap")
    return OverlapRange(start_s=start_s, end_s=end_s)


def build_time_grid(start_s: float, end_s: float, fs_hz: float) -> np.ndarray:
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if end_s < start_s:
        raise ValueError("invalid_time_range")
    step = 1.0 / fs_hz
    n = int(np.floor((end_s - start_s) / step)) + 1
    grid = start_s + np.arange(n, dtype=float) * step
    return grid[grid <= end_s + step * 0.25]


def interp_to_grid(times: np.ndarray, values: np.ndarray, grid_times: np.ndarray) -> np.ndarray:
    if times.size == 0 or values.size == 0:
        return np.full(grid_times.shape, np.nan, dtype=float)
    order = np.argsort(times)
    t_sorted = times[order].astype(float)
    v_sorted = values[order].astype(float)
    finite = np.isfinite(v_sorted)
    if finite.sum() == 0:
        return np.full(grid_times.shape, np.nan, dtype=float)
    t_sorted = t_sorted[finite]
    v_sorted = v_sorted[finite]
    if t_sorted.size == 1:
        out = np.full(grid_times.shape, np.nan, dtype=float)
        out[np.isclose(grid_times, t_sorted[0])] = v_sorted[0]
        return out
    return np.interp(grid_times, t_sorted, v_sorted, left=np.nan, right=np.nan)


def interp_cardiac_limited_gap(
    times: np.ndarray,
    values: np.ndarray,
    grid_times: np.ndarray,
    *,
    max_gap_s: float,
    fs_hz: float,
) -> np.ndarray:
    del fs_hz  # gap limit is defined in seconds between source samples
    out = np.full(grid_times.shape, np.nan, dtype=float)
    if times.size == 0:
        return out

    order = np.argsort(times)
    t_sorted = times[order].astype(float)
    v_sorted = values[order].astype(float)
    finite = np.isfinite(v_sorted)
    t_sorted = t_sorted[finite]
    v_sorted = v_sorted[finite]
    if t_sorted.size == 0:
        return out

    for ti, vi in zip(t_sorted, v_sorted, strict=True):
        out[np.isclose(grid_times, ti)] = vi

    for idx in range(t_sorted.size - 1):
        t0 = float(t_sorted[idx])
        t1 = float(t_sorted[idx + 1])
        gap = t1 - t0
        if gap <= 0 or gap > max_gap_s:
            continue
        v0 = float(v_sorted[idx])
        v1 = float(v_sorted[idx + 1])
        mask = (grid_times >= t0) & (grid_times <= t1)
        if not np.any(mask):
            continue
        frac = (grid_times[mask] - t0) / gap
        out[mask] = v0 + frac * (v1 - v0)

    return out


def zscore_within_observation(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size < 2:
        return out
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0))
    if std <= 0:
        out[finite_mask] = 0.0
        return out
    out[finite_mask] = (values[finite_mask] - mean) / std
    return out


def align_observation(
    eeg_df: pd.DataFrame,
    cardiac_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
) -> pd.DataFrame:
    resample_cfg = cfg.temporal_coupling.resample
    overlap = overlap_time_range(eeg_df, cardiac_df)
    grid_times = build_time_grid(overlap.start_s, overlap.end_s, resample_cfg.fs_hz)

    row0 = eeg_df.iloc[0]
    aligned: dict[str, object] = {
        "dataset_id": row0["dataset_id"],
        "subject_id": row0["subject_id"],
        "task": row0["task"],
        "observation_id": row0["observation_id"],
        "time_s": grid_times,
    }

    eeg_times = eeg_df["time_s"].to_numpy(dtype=float)
    for col in EEG_ENVELOPE_COLUMNS:
        aligned[col] = interp_to_grid(eeg_times, eeg_df[col].to_numpy(dtype=float), grid_times)

    car_times = cardiac_df["time_s"].to_numpy(dtype=float)
    for col in CARDIAC_COLUMNS:
        aligned[col] = interp_cardiac_limited_gap(
            car_times,
            cardiac_df[col].to_numpy(dtype=float),
            grid_times,
            max_gap_s=resample_cfg.interpolate_max_gap_s,
            fs_hz=resample_cfg.fs_hz,
        )

    if resample_cfg.z_score:
        for raw_col, z_col in zip(RAW_COLUMNS, Z_COLUMNS, strict=True):
            aligned[z_col] = zscore_within_observation(np.asarray(aligned[raw_col], dtype=float))
    else:
        for z_col in Z_COLUMNS:
            aligned[z_col] = np.full(grid_times.shape, np.nan, dtype=float)

    columns = [
        "dataset_id",
        "subject_id",
        "task",
        "observation_id",
        "time_s",
        *CARDIAC_COLUMNS,
        *EEG_ENVELOPE_COLUMNS,
        *Z_COLUMNS,
    ]
    return pd.DataFrame(aligned)[columns]


def load_usable_observations(cfg: TemporalCouplingConfig) -> list[UsableObservation]:
    audit_path = audit_output_path(cfg)
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"{audit_path} not found. Run --stage 0 first, or ensure Stage 1a/1b outputs exist."
        )

    audit_df = pd.read_csv(audit_path)
    if audit_df.empty:
        return []

    usable_mask = audit_df["usable"].astype(str).str.lower().isin({"true", "1", "yes"})
    rows = audit_df.loc[usable_mask]
    observations: list[UsableObservation] = []
    for row in rows.itertuples(index=False):
        eeg_file = Path(str(row.eeg_file))
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


def run_stage1c(cfg: TemporalCouplingConfig) -> list[Path]:
    observations = load_usable_observations(cfg)
    if not observations:
        print("[temporal_coupling] stage=1c: no usable observations to process.")
        return []

    written: list[Path] = []
    n_ok = 0

    for obs in observations:
        eeg_path = envelope_input_path(cfg, obs.subject_id)
        cardiac_path = cardiac_input_path(cfg, obs.subject_id)
        out_path = aligned_output_path(cfg, obs.subject_id)

        if not eeg_path.is_file():
            warnings.warn(
                f"[temporal_coupling] stage=1c skipping {obs.subject_id}: missing {eeg_path.name}. "
                "Run --stage 1a first.",
                stacklevel=2,
            )
            continue
        if not cardiac_path.is_file():
            warnings.warn(
                f"[temporal_coupling] stage=1c skipping {obs.subject_id}: missing {cardiac_path.name}. "
                "Run --stage 1b first.",
                stacklevel=2,
            )
            continue

        try:
            eeg_df = pd.read_csv(eeg_path)
            cardiac_df = pd.read_csv(cardiac_path)
            aligned_df = align_observation(eeg_df, cardiac_df, cfg)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(out_path, index=False)
            written.append(out_path)
            n_ok += 1

            overlap = overlap_time_range(eeg_df, cardiac_df)
            print(
                f"[temporal_coupling] stage=1c {obs.subject_id} task={obs.task}: "
                f"overlap={overlap.start_s:.1f}-{overlap.end_s:.1f}s rows={len(aligned_df)} "
                f"fs={cfg.temporal_coupling.resample.fs_hz:g}Hz -> {out_path}"
            )
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=1c skipping {obs.subject_id} task={obs.task}: {exc}",
                stacklevel=2,
            )

    print(f"[temporal_coupling] stage=1c summary: wrote={n_ok}/{len(observations)}")
    return written
