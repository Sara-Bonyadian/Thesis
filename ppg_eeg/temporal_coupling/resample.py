"""Stage 1c: align EEG envelopes and cardiac time series, resample, z-score."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cardiac_timeseries import CARDIAC_FILENAME, QC_GROUP_FILENAME as CARDIAC_QC_FILENAME
from .config import TemporalCouplingConfig
from .data_audit import audit_output_path, group_output_dir, recommended_max_lag_s
from .eeg_envelope import ENVELOPE_FILENAME, QC_GROUP_FILENAME as EEG_QC_FILENAME, UsableObservation, load_usable_observations
from .paths import observation_output_dir

ALIGNED_FILENAME = "features_temporal_aligned.csv"
QC_GROUP_FILENAME = "alignment_qc.csv"

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


@dataclass(frozen=True)
class AlignmentQcRecord:
    subject_id: str
    observation_id: str
    aligned_start_s: float
    aligned_end_s: float
    aligned_duration_s: float
    n_rows: int
    fs_hz: float
    missing_percent_hr: float
    missing_percent_rmssd: float
    missing_percent_sdnn: float
    missing_percent_theta: float
    missing_percent_alpha: float
    missing_percent_beta: float
    recommended_xcorr_lag_s: float
    usable_for_xcorr: bool
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "observation_id": self.observation_id,
            "aligned_start_s": self.aligned_start_s,
            "aligned_end_s": self.aligned_end_s,
            "aligned_duration_s": self.aligned_duration_s,
            "n_rows": self.n_rows,
            "fs_hz": self.fs_hz,
            "missing_percent_hr": self.missing_percent_hr,
            "missing_percent_rmssd": self.missing_percent_rmssd,
            "missing_percent_sdnn": self.missing_percent_sdnn,
            "missing_percent_theta": self.missing_percent_theta,
            "missing_percent_alpha": self.missing_percent_alpha,
            "missing_percent_beta": self.missing_percent_beta,
            "recommended_xcorr_lag_s": self.recommended_xcorr_lag_s,
            "usable_for_xcorr": self.usable_for_xcorr,
            "warning": self.warning,
        }


def aligned_output_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / ALIGNED_FILENAME


def envelope_input_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / ENVELOPE_FILENAME


def cardiac_input_path(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return observation_output_dir(cfg, observation_id) / CARDIAC_FILENAME


def alignment_qc_group_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / QC_GROUP_FILENAME


def _missing_percent(series: pd.Series) -> float:
    if series.empty:
        return 100.0
    return float(100.0 * series.isna().mean())


def _load_upstream_qc_flags(cfg: TemporalCouplingConfig) -> dict[str, tuple[bool | None, bool | None]]:
    """Map observation_id -> (usable_for_hr, usable_eeg_envelope)."""
    group_dir = group_output_dir(cfg)
    flags: dict[str, tuple[bool | None, bool | None]] = {}

    cardiac_path = group_dir / CARDIAC_QC_FILENAME
    if cardiac_path.is_file():
        cardiac_df = pd.read_csv(cardiac_path)
        for row in cardiac_df.itertuples(index=False):
            obs_id = str(row.observation_id)
            usable_hr = bool(row.usable_for_hr) if hasattr(row, "usable_for_hr") else None
            prev = flags.get(obs_id, (None, None))
            flags[obs_id] = (usable_hr, prev[1])

    eeg_path = group_dir / EEG_QC_FILENAME
    if eeg_path.is_file():
        eeg_df = pd.read_csv(eeg_path)
        for row in eeg_df.itertuples(index=False):
            obs_id = str(row.observation_id)
            usable_eeg = bool(row.usable_eeg_envelope) if hasattr(row, "usable_eeg_envelope") else None
            prev = flags.get(obs_id, (None, None))
            flags[obs_id] = (prev[0], usable_eeg)

    return flags


def build_alignment_qc(
    obs: UsableObservation,
    aligned_df: pd.DataFrame,
    overlap: OverlapRange,
    cfg: TemporalCouplingConfig,
    *,
    upstream_usable_for_hr: bool | None = None,
    upstream_usable_eeg_envelope: bool | None = None,
) -> AlignmentQcRecord:
    resample_cfg = cfg.temporal_coupling.resample
    cardiac_cfg = cfg.temporal_coupling.cardiac
    audit_cfg = cfg.temporal_coupling.audit
    lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s

    aligned_start_s = float(overlap.start_s)
    aligned_end_s = float(overlap.end_s)
    aligned_duration_s = max(0.0, aligned_end_s - aligned_start_s)
    n_rows = len(aligned_df)
    fs_hz = float(resample_cfg.fs_hz)

    missing_hr = _missing_percent(aligned_df["hr"])
    missing_rmssd = _missing_percent(aligned_df["rmssd"])
    missing_sdnn = _missing_percent(aligned_df["sdnn"])
    missing_theta = _missing_percent(aligned_df["theta_env"])
    missing_alpha = _missing_percent(aligned_df["alpha_env"])
    missing_beta = _missing_percent(aligned_df["beta_env"])

    recommended_lag = recommended_max_lag_s(
        overlap_duration_s=aligned_duration_s,
        requested_lag_max_s=lag_max_s,
    )

    warnings_out: list[str] = []
    if aligned_duration_s < audit_cfg.min_overlap_s:
        warnings_out.append(
            f"short_aligned_duration ({aligned_duration_s:.1f}s < {audit_cfg.min_overlap_s:.1f}s)"
        )
    if recommended_lag <= 0:
        warnings_out.append("no_recommended_xcorr_lag")
    if missing_hr > 100.0 - cardiac_cfg.min_valid_hr_percent:
        warnings_out.append("high_missing_hr")
    if missing_rmssd > 100.0 - cardiac_cfg.min_valid_hrv_percent:
        warnings_out.append("high_missing_rmssd")
    if missing_sdnn > 100.0 - cardiac_cfg.min_valid_hrv_percent:
        warnings_out.append("high_missing_sdnn")
    for band, pct in (
        ("theta", missing_theta),
        ("alpha", missing_alpha),
        ("beta", missing_beta),
    ):
        if pct > 0:
            warnings_out.append(f"high_missing_{band}")

    if upstream_usable_for_hr is False:
        warnings_out.append("upstream_not_usable_for_hr")
    if upstream_usable_eeg_envelope is False:
        warnings_out.append("upstream_not_usable_eeg_envelope")

    if resample_cfg.z_score:
        for z_col in ("hr_z", "rmssd_z", "sdnn_z", "theta_env_z", "alpha_env_z", "beta_env_z"):
            if z_col not in aligned_df.columns:
                continue
            finite = aligned_df[z_col].dropna()
            if finite.size < 2:
                warnings_out.append(f"insufficient_finite_{z_col}")
            elif float(finite.std(ddof=0)) <= 0:
                warnings_out.append(f"flat_{z_col}")

    usable = (
        aligned_duration_s >= audit_cfg.min_overlap_s
        and recommended_lag > 0
        and missing_hr <= 100.0 - cardiac_cfg.min_valid_hr_percent
        and missing_rmssd <= 100.0 - cardiac_cfg.min_valid_hrv_percent
        and missing_sdnn <= 100.0 - cardiac_cfg.min_valid_hrv_percent
        and missing_theta == 0.0
        and missing_alpha == 0.0
        and missing_beta == 0.0
        and n_rows >= 10
        and (upstream_usable_for_hr is not False)
        and (upstream_usable_eeg_envelope is not False)
        and not any(w.startswith("insufficient_finite_") or w.startswith("flat_") for w in warnings_out)
    )

    return AlignmentQcRecord(
        subject_id=obs.subject_id,
        observation_id=obs.observation_id,
        aligned_start_s=aligned_start_s,
        aligned_end_s=aligned_end_s,
        aligned_duration_s=aligned_duration_s,
        n_rows=n_rows,
        fs_hz=fs_hz,
        missing_percent_hr=missing_hr,
        missing_percent_rmssd=missing_rmssd,
        missing_percent_sdnn=missing_sdnn,
        missing_percent_theta=missing_theta,
        missing_percent_alpha=missing_alpha,
        missing_percent_beta=missing_beta,
        recommended_xcorr_lag_s=recommended_lag,
        usable_for_xcorr=usable,
        warning=";".join(dict.fromkeys(warnings_out)),
    )


def _print_alignment_qc_summary(qc: AlignmentQcRecord) -> None:
    print(
        f"[temporal_coupling]   QC {qc.subject_id}: "
        f"aligned={qc.aligned_start_s:.1f}-{qc.aligned_end_s:.1f}s "
        f"rows={qc.n_rows} missing_hr={qc.missing_percent_hr:.1f}% "
        f"missing_rmssd={qc.missing_percent_rmssd:.1f}% "
        f"lag={qc.recommended_xcorr_lag_s:.0f}s usable_for_xcorr={qc.usable_for_xcorr}"
    )
    if qc.warning:
        warnings.warn(f"[temporal_coupling]   {qc.subject_id} alignment QC warning: {qc.warning}", stacklevel=2)


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
    car0 = cardiac_df.iloc[0]
    if "condition" in eeg_df.columns and pd.notna(row0["condition"]):
        condition = str(row0["condition"])
    elif "condition" in cardiac_df.columns and pd.notna(car0["condition"]):
        condition = str(car0["condition"])
    else:
        condition = str(row0["task"])
    aligned: dict[str, object] = {
        "dataset_id": row0["dataset_id"],
        "subject_id": row0["subject_id"],
        "task": row0["task"],
        "condition": condition,
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
        "condition",
        "observation_id",
        "time_s",
        *CARDIAC_COLUMNS,
        *EEG_ENVELOPE_COLUMNS,
        *Z_COLUMNS,
    ]
    return pd.DataFrame(aligned)[columns]


def run_stage1c(cfg: TemporalCouplingConfig) -> list[Path]:
    observations = load_usable_observations(cfg)
    if not observations:
        print("[temporal_coupling] stage=1c: no usable observations to process.")
        return []

    written: list[Path] = []
    qc_records: list[AlignmentQcRecord] = []
    upstream_flags = _load_upstream_qc_flags(cfg)
    n_ok = 0

    for obs in observations:
        eeg_path = envelope_input_path(cfg, obs.observation_id)
        cardiac_path = cardiac_input_path(cfg, obs.observation_id)
        out_path = aligned_output_path(cfg, obs.observation_id)

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
            overlap = overlap_time_range(eeg_df, cardiac_df)
            upstream_hr, upstream_eeg = upstream_flags.get(obs.observation_id, (None, None))
            qc = build_alignment_qc(
                obs,
                aligned_df,
                overlap,
                cfg,
                upstream_usable_for_hr=upstream_hr,
                upstream_usable_eeg_envelope=upstream_eeg,
            )
            qc_records.append(qc)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(out_path, index=False)
            written.append(out_path)
            n_ok += 1

            print(
                f"[temporal_coupling] stage=1c {obs.subject_id} task={obs.task}: "
                f"overlap={overlap.start_s:.1f}-{overlap.end_s:.1f}s rows={len(aligned_df)} "
                f"fs={cfg.temporal_coupling.resample.fs_hz:g}Hz usable_for_xcorr={qc.usable_for_xcorr} "
                f"-> {out_path}"
            )
            _print_alignment_qc_summary(qc)
        except Exception as exc:
            warnings.warn(
                f"[temporal_coupling] stage=1c skipping {obs.subject_id} task={obs.task}: {exc}",
                stacklevel=2,
            )

    if qc_records:
        qc_path = alignment_qc_group_path(cfg)
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record.to_row() for record in qc_records]).to_csv(qc_path, index=False)
        print(f"[temporal_coupling] stage=1c wrote group QC -> {qc_path}")

    print(f"[temporal_coupling] stage=1c summary: wrote={n_ok}/{len(observations)}")
    return written
