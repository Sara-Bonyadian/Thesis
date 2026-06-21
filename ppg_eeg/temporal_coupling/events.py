"""Stage 4: event-triggered averages around HR and EEG envelope events."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TemporalCouplingConfig, TemporalCouplingEventsConfig
from .cross_correlation import CARDIAC_LABELS, CARDIAC_VARS, EEG_LABELS, EEG_VARS
from .paths import group_output_dir, partition_key_from_row
from .resample import (
    QC_GROUP_FILENAME as ALIGNMENT_QC_FILENAME,
    aligned_output_path,
)

EVENT_COUNTS_FILENAME = "event_counts.csv"
EVENT_LIST_FILENAME = "event_list.csv"
EVENT_QC_FILENAME = "event_qc.csv"
HR_TO_EEG_FILENAME = "event_triggered_hr_to_eeg.csv"
EEG_TO_HR_FILENAME = "event_triggered_eeg_to_hr.csv"
HR_TO_EEG_PLOT = "event_triggered_hr_to_eeg.png"
EEG_TO_HR_PLOT = "event_triggered_eeg_to_hr.png"
INTERPRETATION_NOTES_FILENAME = "stage4_interpretation_notes.txt"

AVERAGING_METHOD = "subject_mean_then_group_mean"

STAGE4_OUTPUT_FILENAMES = (
    EVENT_COUNTS_FILENAME,
    EVENT_LIST_FILENAME,
    EVENT_QC_FILENAME,
    HR_TO_EEG_FILENAME,
    EEG_TO_HR_FILENAME,
    HR_TO_EEG_PLOT,
    EEG_TO_HR_PLOT,
    INTERPRETATION_NOTES_FILENAME,
)

HR_INCREASE = "hr_increase"
HR_DECREASE = "hr_decrease"
THETA_BURST = "theta_burst"
ALPHA_SUPPRESSION = "alpha_suppression"
BETA_BURST = "beta_burst"

EEG_EVENT_TYPES = (THETA_BURST, ALPHA_SUPPRESSION, BETA_BURST)
HR_EVENT_TYPES = (HR_INCREASE, HR_DECREASE)
ALL_EVENT_TYPES = (*HR_EVENT_TYPES, *EEG_EVENT_TYPES)

HR_ROW_LABELS = {
    HR_INCREASE: "HR increase events → EEG envelopes",
    HR_DECREASE: "HR decrease events → EEG envelopes",
}

EEG_ROW_LABELS = {
    THETA_BURST: "Theta burst → HR / RMSSD / SDNN",
    ALPHA_SUPPRESSION: "Alpha suppression → HR / RMSSD / SDNN",
    BETA_BURST: "Beta burst → HR / RMSSD / SDNN",
}

EXPLORATORY_NOTE = (
    "Exploratory event-triggered averages — not for biological interpretation."
)

EVENT_COUNTS_COLUMNS = (
    "subject_id",
    "hr_increase",
    "hr_decrease",
    "theta_burst",
    "alpha_suppression",
    "beta_burst",
    "warning",
)

EVENT_LIST_COLUMNS = (
    "dataset_id",
    "subject_id",
    "task",
    "observation_id",
    "event_type",
    "event_time_s",
    "event_value",
    "epoch_start_s",
    "epoch_end_s",
    "accepted",
    "rejection_reason",
)

EVENT_QC_COLUMNS = (
    "event_type",
    "total_events",
    "n_subjects",
    "mean_events_per_subject",
    "min_events_per_subject",
    "max_events_per_subject",
    "usable_for_group_plot",
    "averaging_method",
    "subject_balanced_average",
    "min_total_events_required",
    "min_subjects_required",
    "warning",
)

HR_TO_EEG_COLUMNS = (
    "event_type",
    "variable",
    "time_s",
    "mean_z",
    "sem_z",
    "n_subjects",
    "n_events",
)

EEG_TO_HR_COLUMNS = (
    "event_type",
    "variable",
    "time_s",
    "mean_z",
    "sem_z",
    "n_subjects",
    "n_events",
)


@dataclass(frozen=True)
class SubjectEventCounts:
    subject_id: str
    hr_increase: int
    hr_decrease: int
    theta_burst: int
    alpha_suppression: int
    beta_burst: int
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "hr_increase": self.hr_increase,
            "hr_decrease": self.hr_decrease,
            "theta_burst": self.theta_burst,
            "alpha_suppression": self.alpha_suppression,
            "beta_burst": self.beta_burst,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class EventListRecord:
    dataset_id: str
    subject_id: str
    task: str
    observation_id: str
    event_type: str
    event_time_s: float
    event_value: float
    epoch_start_s: float
    epoch_end_s: float
    accepted: bool
    rejection_reason: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "observation_id": self.observation_id,
            "event_type": self.event_type,
            "event_time_s": self.event_time_s,
            "event_value": self.event_value,
            "epoch_start_s": self.epoch_start_s,
            "epoch_end_s": self.epoch_end_s,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class EventTypeQc:
    event_type: str
    total_events: int
    n_subjects: int
    mean_events_per_subject: float
    min_events_per_subject: int
    max_events_per_subject: int
    usable_for_group_plot: bool
    averaging_method: str
    subject_balanced_average: bool
    min_total_events_required: int
    min_subjects_required: int
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "total_events": self.total_events,
            "n_subjects": self.n_subjects,
            "mean_events_per_subject": self.mean_events_per_subject,
            "min_events_per_subject": self.min_events_per_subject,
            "max_events_per_subject": self.max_events_per_subject,
            "usable_for_group_plot": self.usable_for_group_plot,
            "averaging_method": self.averaging_method,
            "subject_balanced_average": self.subject_balanced_average,
            "min_total_events_required": self.min_total_events_required,
            "min_subjects_required": self.min_subjects_required,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class SubjectEventResult:
    counts: SubjectEventCounts
    hr_to_eeg: dict[str, dict[str, np.ndarray]]
    eeg_to_hr: dict[str, dict[str, np.ndarray]]
    event_records: tuple[EventListRecord, ...]


def event_counts_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / EVENT_COUNTS_FILENAME


def event_list_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / EVENT_LIST_FILENAME


def event_qc_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / EVENT_QC_FILENAME


def interpretation_notes_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / INTERPRETATION_NOTES_FILENAME


def hr_to_eeg_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / HR_TO_EEG_FILENAME


def eeg_to_hr_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / EEG_TO_HR_FILENAME


def hr_to_eeg_plot_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / HR_TO_EEG_PLOT


def eeg_to_hr_plot_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / EEG_TO_HR_PLOT


def stage4_output_paths(cfg: TemporalCouplingConfig) -> list[Path]:
    group_dir = group_output_dir(cfg)
    return [group_dir / name for name in STAGE4_OUTPUT_FILENAMES]


def _alignment_qc_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / ALIGNMENT_QC_FILENAME


def _load_alignment_qc(cfg: TemporalCouplingConfig) -> pd.DataFrame | None:
    qc_path = _alignment_qc_path(cfg)
    if not qc_path.is_file():
        return None
    return pd.read_csv(qc_path)


def _discover_observation_ids(
    cfg: TemporalCouplingConfig,
    alignment_qc: pd.DataFrame | None,
) -> list[str]:
    if alignment_qc is not None and not alignment_qc.empty and "observation_id" in alignment_qc.columns:
        return sorted(alignment_qc["observation_id"].astype(str).unique())
    dataset_dir = Path(cfg.paths.out_root) / cfg.dataset_id
    if not dataset_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in dataset_dir.iterdir()
        if path.is_dir()
        and path.name != "group"
        and (path / "features_temporal_aligned.csv").is_file()
    )


def _usable_observation_ids(
    alignment_qc: pd.DataFrame | None,
    observation_ids: list[str],
) -> set[str]:
    if alignment_qc is None or alignment_qc.empty:
        return set(observation_ids)
    if "usable_for_xcorr" not in alignment_qc.columns:
        return set(observation_ids)
    usable = alignment_qc.loc[
        alignment_qc["usable_for_xcorr"].astype(str).str.lower().isin({"true", "1", "yes"})
    ]
    if "observation_id" in usable.columns:
        return set(usable["observation_id"].astype(str))
    return set(usable["subject_id"].astype(str))


def _resolve_event_percentile(events_cfg: TemporalCouplingEventsConfig) -> float:
    if events_cfg.event_percentile is not None:
        return float(events_cfg.event_percentile)
    return float(100.0 - events_cfg.hr_percentile)


def _build_epoch_times(events_cfg: TemporalCouplingEventsConfig, fs_hz: float) -> np.ndarray:
    pre = float(events_cfg.epoch_pre_s)
    post = float(events_cfg.epoch_post_s)
    step = 1.0 / fs_hz
    n_pre = int(round(pre * fs_hz))
    n_post = int(round(post * fs_hz))
    return np.arange(-n_pre, n_post + 1) * step


def _compute_delta_hr_z(
    time_s: np.ndarray,
    hr_z: np.ndarray,
    hr_delta_window_s: float,
) -> np.ndarray:
    delta = np.full(len(time_s), np.nan, dtype=float)
    if len(time_s) < 2:
        return delta

    for idx in range(len(time_s)):
        target_time = float(time_s[idx]) - hr_delta_window_s
        past_idx = int(np.searchsorted(time_s, target_time, side="left"))
        if past_idx >= idx:
            continue
        if not np.isfinite(hr_z[idx]) or not np.isfinite(hr_z[past_idx]):
            continue
        delta[idx] = float(hr_z[idx] - hr_z[past_idx])
    return delta


def _valid_event_indices(
    time_s: np.ndarray,
    *,
    epoch_pre_s: float,
    epoch_post_s: float,
    hr_delta_window_s: float,
) -> np.ndarray:
    if time_s.size == 0:
        return np.array([], dtype=int)
    t_min = float(time_s[0]) + max(epoch_pre_s, hr_delta_window_s)
    t_max = float(time_s[-1]) - epoch_post_s
    if t_max < t_min:
        return np.array([], dtype=int)
    return np.flatnonzero((time_s >= t_min) & (time_s <= t_max))


def _identify_hr_events(
    delta_hr_z: np.ndarray,
    valid_indices: np.ndarray,
    events_cfg: TemporalCouplingEventsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    valid_delta = delta_hr_z[valid_indices]
    finite_mask = np.isfinite(valid_delta)
    if not np.any(finite_mask):
        return np.array([], dtype=int), np.array([], dtype=int)

    finite_delta = valid_delta[finite_mask]
    high_pct = _resolve_event_percentile(events_cfg)
    low_pct = 100.0 - high_pct
    increase_thr = float(np.percentile(finite_delta, high_pct))
    decrease_thr = float(np.percentile(finite_delta, low_pct))

    increase_mask = finite_mask & (valid_delta >= increase_thr)
    decrease_mask = finite_mask & (valid_delta <= decrease_thr)

    return valid_indices[increase_mask], valid_indices[decrease_mask]


def _contiguous_run_onsets(mask: np.ndarray, min_len: int) -> np.ndarray:
    if mask.size == 0 or min_len <= 0:
        return np.array([], dtype=int)

    onsets: list[int] = []
    run_start: int | None = None
    run_len = 0
    for idx, active in enumerate(mask):
        if active:
            if run_start is None:
                run_start = idx
            run_len += 1
        elif run_start is not None:
            if run_len >= min_len:
                onsets.append(run_start)
            run_start = None
            run_len = 0
    if run_start is not None and run_len >= min_len:
        onsets.append(run_start)
    return np.asarray(onsets, dtype=int)


def _filter_by_min_distance(
    indices: np.ndarray,
    time_s: np.ndarray,
    min_distance_s: float,
    scores: np.ndarray,
) -> np.ndarray:
    if indices.size == 0:
        return indices
    if min_distance_s <= 0:
        return np.asarray(sorted(indices.astype(int)), dtype=int)

    order = np.argsort(-scores[indices])
    picked: list[int] = []
    for pos in order:
        idx = int(indices[pos])
        t = float(time_s[idx])
        if all(abs(t - float(time_s[prev])) >= min_distance_s for prev in picked):
            picked.append(idx)
    return np.asarray(sorted(picked), dtype=int)


def _identify_fixed_z_events(
    series: np.ndarray,
    valid_indices: np.ndarray,
    time_s: np.ndarray,
    *,
    above: bool,
    threshold_z: float,
    min_samples: int,
    min_distance_s: float,
) -> np.ndarray:
    if valid_indices.size == 0:
        return np.array([], dtype=int)

    full_mask = np.zeros(len(series), dtype=bool)
    finite = np.isfinite(series[valid_indices])
    if above:
        full_mask[valid_indices] = finite & (series[valid_indices] > threshold_z)
        scores = series
    else:
        full_mask[valid_indices] = finite & (series[valid_indices] < -threshold_z)
        scores = -series

    onsets = _contiguous_run_onsets(full_mask, min_samples)
    return _filter_by_min_distance(onsets, time_s, min_distance_s, scores)


def _identify_percentile_events(
    series: np.ndarray,
    valid_indices: np.ndarray,
    time_s: np.ndarray,
    *,
    percentile: float,
    tail: str,
    min_samples: int,
    min_distance_s: float,
) -> np.ndarray:
    if valid_indices.size == 0:
        return np.array([], dtype=int)

    values = series[valid_indices]
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.array([], dtype=int)

    finite_values = values[finite_mask]
    threshold = float(np.percentile(finite_values, percentile))

    full_mask = np.zeros(len(series), dtype=bool)
    if tail == "high":
        full_mask[valid_indices] = finite_mask & (values >= threshold)
        scores = series
    else:
        full_mask[valid_indices] = finite_mask & (values <= threshold)
        scores = -series

    onsets = _contiguous_run_onsets(full_mask, min_samples)
    return _filter_by_min_distance(onsets, time_s, min_distance_s, scores)


def _normalize_method(method: str) -> str:
    return str(method).strip().casefold().replace("-", "_")


def _identify_band_events(
    series: np.ndarray,
    valid_indices: np.ndarray,
    time_s: np.ndarray,
    events_cfg: TemporalCouplingEventsConfig,
    *,
    method: str,
    percentile: float,
    tail: str,
    min_samples: int,
) -> np.ndarray:
    normalized = _normalize_method(method)
    if normalized == "fixed_z":
        above = tail == "high"
        return _identify_fixed_z_events(
            series,
            valid_indices,
            time_s,
            above=above,
            threshold_z=float(events_cfg.eeg_event_threshold_z),
            min_samples=min_samples,
            min_distance_s=float(events_cfg.min_event_distance_s),
        )
    return _identify_percentile_events(
        series,
        valid_indices,
        time_s,
        percentile=percentile,
        tail=tail,
        min_samples=min_samples,
        min_distance_s=float(events_cfg.min_event_distance_s),
    )


def _epoch_rejection_reason(
    values: np.ndarray,
    event_idx: int,
    epoch_times: np.ndarray,
    fs_hz: float,
) -> str | None:
    n_pre = int(round(abs(float(epoch_times[0])) * fs_hz))
    start_idx = int(event_idx) - n_pre
    end_idx = start_idx + len(epoch_times)
    if start_idx < 0:
        return "epoch_before_recording_start"
    if end_idx > len(values):
        return "epoch_after_recording_end"
    segment = values[start_idx:end_idx]
    if not np.all(np.isfinite(segment)):
        return "non_finite_epoch_values"
    return None


def _split_accepted_events(
    event_indices: np.ndarray,
    values: np.ndarray,
    epoch_times: np.ndarray,
    fs_hz: float,
) -> tuple[np.ndarray, dict[int, str]]:
    if event_indices.size == 0:
        return np.array([], dtype=int), {}

    accepted: list[int] = []
    rejected: dict[int, str] = {}
    for event_idx in event_indices:
        reason = _epoch_rejection_reason(values, int(event_idx), epoch_times, fs_hz)
        if reason is None:
            accepted.append(int(event_idx))
        else:
            rejected[int(event_idx)] = reason
    return np.asarray(accepted, dtype=int), rejected


def _build_event_records(
    meta: dict[str, str],
    event_type: str,
    event_indices: np.ndarray,
    event_values: np.ndarray,
    time_s: np.ndarray,
    values: np.ndarray,
    epoch_times: np.ndarray,
    fs_hz: float,
    epoch_pre_s: float,
    epoch_post_s: float,
) -> tuple[np.ndarray, tuple[EventListRecord, ...]]:
    accepted_idx, rejected = _split_accepted_events(event_indices, values, epoch_times, fs_hz)
    records: list[EventListRecord] = []

    for event_idx in event_indices:
        idx = int(event_idx)
        event_time = float(time_s[idx])
        value = float(event_values[idx]) if np.isfinite(event_values[idx]) else np.nan
        rejection_reason = rejected.get(idx, "")
        records.append(
            EventListRecord(
                dataset_id=meta["dataset_id"],
                subject_id=meta["subject_id"],
                task=meta["task"],
                observation_id=meta["observation_id"],
                event_type=event_type,
                event_time_s=event_time,
                event_value=value,
                epoch_start_s=event_time - epoch_pre_s,
                epoch_end_s=event_time + epoch_post_s,
                accepted=idx not in rejected,
                rejection_reason=rejection_reason,
            )
        )
    return accepted_idx, tuple(records)


def _extract_epochs(
    values: np.ndarray,
    event_indices: np.ndarray,
    epoch_times: np.ndarray,
    fs_hz: float,
) -> np.ndarray:
    if event_indices.size == 0:
        return np.empty((0, len(epoch_times)), dtype=float)

    epochs: list[np.ndarray] = []
    n_pre = int(round(abs(float(epoch_times[0])) * fs_hz))

    for event_idx in event_indices:
        start_idx = int(event_idx) - n_pre
        end_idx = start_idx + len(epoch_times)
        if start_idx < 0 or end_idx > len(values):
            continue
        segment = values[start_idx:end_idx]
        if not np.all(np.isfinite(segment)):
            continue
        epochs.append(segment.astype(float))

    if not epochs:
        return np.empty((0, len(epoch_times)), dtype=float)
    return np.vstack(epochs)


def _mean_epoch(epochs: np.ndarray) -> np.ndarray:
    if epochs.size == 0:
        return np.full(0, np.nan)
    return np.nanmean(epochs, axis=0)


def _sem_across_subjects(subject_means: list[np.ndarray]) -> np.ndarray:
    if not subject_means:
        return np.array([], dtype=float)
    stacked = np.vstack(subject_means)
    n = stacked.shape[0]
    if n <= 1:
        return np.zeros(stacked.shape[1], dtype=float)
    return np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n)


def _process_subject(
    aligned_df: pd.DataFrame,
    cfg: TemporalCouplingConfig,
    epoch_times: np.ndarray,
) -> SubjectEventResult:
    events_cfg = cfg.temporal_coupling.events
    fs_hz = float(cfg.temporal_coupling.resample.fs_hz)
    subject_id = str(aligned_df["subject_id"].iloc[0])
    meta = {
        "dataset_id": str(aligned_df["dataset_id"].iloc[0]),
        "subject_id": subject_id,
        "task": str(aligned_df["task"].iloc[0]),
        "observation_id": str(aligned_df["observation_id"].iloc[0]),
    }

    time_s = aligned_df["time_s"].to_numpy(dtype=float)
    hr_z = aligned_df["hr_z"].to_numpy(dtype=float)
    theta_z = aligned_df["theta_env_z"].to_numpy(dtype=float)
    alpha_z = aligned_df["alpha_env_z"].to_numpy(dtype=float)
    beta_z = aligned_df["beta_env_z"].to_numpy(dtype=float)

    valid_indices = _valid_event_indices(
        time_s,
        epoch_pre_s=events_cfg.epoch_pre_s,
        epoch_post_s=events_cfg.epoch_post_s,
        hr_delta_window_s=events_cfg.hr_delta_window_s,
    )

    delta_hr_z = _compute_delta_hr_z(time_s, hr_z, events_cfg.hr_delta_window_s)
    hr_increase_candidates, hr_decrease_candidates = _identify_hr_events(
        delta_hr_z, valid_indices, events_cfg
    )

    min_samples = max(1, int(round(events_cfg.min_event_duration_s * fs_hz)))

    theta_candidates = _identify_band_events(
        theta_z,
        valid_indices,
        time_s,
        events_cfg,
        method=events_cfg.theta_burst_method,
        percentile=float(events_cfg.theta_burst_percentile),
        tail="high",
        min_samples=min_samples,
    )
    alpha_candidates = _identify_band_events(
        alpha_z,
        valid_indices,
        time_s,
        events_cfg,
        method=events_cfg.alpha_suppression_method,
        percentile=float(events_cfg.alpha_suppression_percentile),
        tail="low",
        min_samples=min_samples,
    )
    beta_candidates = _identify_band_events(
        beta_z,
        valid_indices,
        time_s,
        events_cfg,
        method=events_cfg.beta_burst_method,
        percentile=float(events_cfg.beta_burst_percentile),
        tail="high",
        min_samples=min_samples,
    )

    event_records: list[EventListRecord] = []
    hr_increase_idx, hr_increase_records = _build_event_records(
        meta,
        HR_INCREASE,
        hr_increase_candidates,
        delta_hr_z,
        time_s,
        hr_z,
        epoch_times,
        fs_hz,
        events_cfg.epoch_pre_s,
        events_cfg.epoch_post_s,
    )
    event_records.extend(hr_increase_records)

    hr_decrease_idx, hr_decrease_records = _build_event_records(
        meta,
        HR_DECREASE,
        hr_decrease_candidates,
        delta_hr_z,
        time_s,
        hr_z,
        epoch_times,
        fs_hz,
        events_cfg.epoch_pre_s,
        events_cfg.epoch_post_s,
    )
    event_records.extend(hr_decrease_records)

    theta_idx, theta_records = _build_event_records(
        meta,
        THETA_BURST,
        theta_candidates,
        theta_z,
        time_s,
        theta_z,
        epoch_times,
        fs_hz,
        events_cfg.epoch_pre_s,
        events_cfg.epoch_post_s,
    )
    event_records.extend(theta_records)

    alpha_idx, alpha_records = _build_event_records(
        meta,
        ALPHA_SUPPRESSION,
        alpha_candidates,
        alpha_z,
        time_s,
        alpha_z,
        epoch_times,
        fs_hz,
        events_cfg.epoch_pre_s,
        events_cfg.epoch_post_s,
    )
    event_records.extend(alpha_records)

    beta_idx, beta_records = _build_event_records(
        meta,
        BETA_BURST,
        beta_candidates,
        beta_z,
        time_s,
        beta_z,
        epoch_times,
        fs_hz,
        events_cfg.epoch_pre_s,
        events_cfg.epoch_post_s,
    )
    event_records.extend(beta_records)

    warning_parts: list[str] = []
    if valid_indices.size == 0:
        warning_parts.append("no_valid_epoch_window")

    counts = SubjectEventCounts(
        subject_id=subject_id,
        hr_increase=len(hr_increase_idx),
        hr_decrease=len(hr_decrease_idx),
        theta_burst=len(theta_idx),
        alpha_suppression=len(alpha_idx),
        beta_burst=len(beta_idx),
        warning="; ".join(warning_parts),
    )

    hr_to_eeg: dict[str, dict[str, np.ndarray]] = {}
    for event_type, event_indices in (
        (HR_INCREASE, hr_increase_idx),
        (HR_DECREASE, hr_decrease_idx),
    ):
        hr_to_eeg[event_type] = {}
        for var in EEG_VARS:
            epochs = _extract_epochs(
                aligned_df[var].to_numpy(dtype=float),
                event_indices,
                epoch_times,
                fs_hz,
            )
            hr_to_eeg[event_type][var] = _mean_epoch(epochs)

    eeg_to_hr: dict[str, dict[str, np.ndarray]] = {}
    for event_type, event_indices in (
        (THETA_BURST, theta_idx),
        (ALPHA_SUPPRESSION, alpha_idx),
        (BETA_BURST, beta_idx),
    ):
        eeg_to_hr[event_type] = {}
        for var in CARDIAC_VARS:
            epochs = _extract_epochs(
                aligned_df[var].to_numpy(dtype=float),
                event_indices,
                epoch_times,
                fs_hz,
            )
            eeg_to_hr[event_type][var] = _mean_epoch(epochs)

    return SubjectEventResult(
        counts=counts,
        hr_to_eeg=hr_to_eeg,
        eeg_to_hr=eeg_to_hr,
        event_records=tuple(event_records),
    )


def _event_count_for_type(counts: SubjectEventCounts, event_type: str) -> int:
    return int(getattr(counts, event_type))


def _build_event_qc(
    counts_df: pd.DataFrame,
    events_cfg: TemporalCouplingEventsConfig,
) -> list[EventTypeQc]:
    min_total = int(events_cfg.min_total_events)
    min_subjects = int(events_cfg.min_contributing_subjects)
    summaries: list[EventTypeQc] = []

    for event_type in ALL_EVENT_TYPES:
        per_subject = counts_df[event_type].astype(int)
        contributing = per_subject[per_subject > 0]
        total_events = int(per_subject.sum())
        n_subjects = int(len(contributing))
        usable = total_events >= min_total and n_subjects >= min_subjects

        warning_parts: list[str] = []
        if total_events == 0:
            warning_parts.append("zero_events")
        if total_events < min_total:
            warning_parts.append("exploratory_insufficient_events")
        if n_subjects < min_subjects:
            warning_parts.append("exploratory_insufficient_subjects")
        if not usable:
            warning_parts.append("exploratory")

        if contributing.empty:
            mean_events = 0.0
            min_events = 0
            max_events = 0
        else:
            mean_events = float(contributing.mean())
            min_events = int(contributing.min())
            max_events = int(contributing.max())

        summaries.append(
            EventTypeQc(
                event_type=event_type,
                total_events=total_events,
                n_subjects=n_subjects,
                mean_events_per_subject=mean_events,
                min_events_per_subject=min_events,
                max_events_per_subject=max_events,
                usable_for_group_plot=usable,
                averaging_method=AVERAGING_METHOD,
                subject_balanced_average=True,
                min_total_events_required=min_total,
                min_subjects_required=min_subjects,
                warning="; ".join(warning_parts),
            )
        )
    return summaries


def _qc_lookup(qc_rows: list[EventTypeQc]) -> dict[str, EventTypeQc]:
    return {row.event_type: row for row in qc_rows}


def _format_subplot_title(
    left_label: str,
    right_label: str,
    qc: EventTypeQc,
    events_cfg: TemporalCouplingEventsConfig,
) -> str:
    parts = [f"events={qc.total_events}", f"subjects={qc.n_subjects}"]
    if qc.total_events == 0:
        parts.append("zero events")
    if qc.total_events < events_cfg.min_total_events:
        parts.append("low events")
    if qc.n_subjects < events_cfg.min_contributing_subjects:
        parts.append("low subjects")
    if not qc.usable_for_group_plot:
        parts.append("exploratory")
    return f"{left_label} → {right_label} ({', '.join(parts)})"


def _build_hr_to_eeg_rows(
    subject_results: list[SubjectEventResult],
    epoch_times: np.ndarray,
) -> list[dict[str, object]]:
    """Average events within each subject, then average subject trajectories (balanced)."""
    rows: list[dict[str, object]] = []
    for event_type in HR_EVENT_TYPES:
        for var in EEG_VARS:
            subject_means: list[np.ndarray] = []
            n_events = 0
            for result in subject_results:
                n_events += _event_count_for_type(result.counts, event_type)
                mean_curve = result.hr_to_eeg[event_type][var]
                if mean_curve.size != len(epoch_times) or not np.any(np.isfinite(mean_curve)):
                    continue
                subject_means.append(mean_curve)

            if not subject_means:
                continue

            group_mean = np.nanmean(np.vstack(subject_means), axis=0)
            group_sem = _sem_across_subjects(subject_means)
            for t, m, s in zip(epoch_times, group_mean, group_sem):
                rows.append(
                    {
                        "event_type": event_type,
                        "variable": var,
                        "time_s": float(t),
                        "mean_z": float(m),
                        "sem_z": float(s) if np.isfinite(s) else np.nan,
                        "n_subjects": len(subject_means),
                        "n_events": n_events,
                    }
                )
    return rows


def _build_eeg_to_hr_rows(
    subject_results: list[SubjectEventResult],
    epoch_times: np.ndarray,
) -> list[dict[str, object]]:
    """Average events within each subject, then average subject trajectories (balanced)."""
    rows: list[dict[str, object]] = []
    for event_type in EEG_EVENT_TYPES:
        for var in CARDIAC_VARS:
            subject_means: list[np.ndarray] = []
            n_events = 0
            for result in subject_results:
                n_events += _event_count_for_type(result.counts, event_type)
                mean_curve = result.eeg_to_hr[event_type][var]
                if mean_curve.size != len(epoch_times) or not np.any(np.isfinite(mean_curve)):
                    continue
                subject_means.append(mean_curve)

            if not subject_means:
                continue

            group_mean = np.nanmean(np.vstack(subject_means), axis=0)
            group_sem = _sem_across_subjects(subject_means)
            for t, m, s in zip(epoch_times, group_mean, group_sem):
                rows.append(
                    {
                        "event_type": event_type,
                        "variable": var,
                        "time_s": float(t),
                        "mean_z": float(m),
                        "sem_z": float(s) if np.isfinite(s) else np.nan,
                        "n_subjects": len(subject_means),
                        "n_events": n_events,
                    }
                )
    return rows


def _plot_hr_to_eeg(
    hr_to_eeg_df: pd.DataFrame,
    qc_lookup: dict[str, EventTypeQc],
    output_path: Path,
    events_cfg: TemporalCouplingEventsConfig,
    *,
    epoch_pre_s: float,
    epoch_post_s: float,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    event_rows = [(HR_INCREASE, 0), (HR_DECREASE, 1)]

    for event_type, row_idx in event_rows:
        qc = qc_lookup[event_type]
        row_label = HR_ROW_LABELS[event_type]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nEnvelope z")

        for col_idx, (var, label) in enumerate(zip(EEG_VARS, EEG_LABELS)):
            ax = axes[row_idx, col_idx]
            subset = hr_to_eeg_df.loc[
                (hr_to_eeg_df["event_type"] == event_type) & (hr_to_eeg_df["variable"] == var)
            ]
            title = _format_subplot_title(row_label.split(" → ")[0], label, qc, events_cfg)
            if subset.empty:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no curve", ha="center", va="center", transform=ax.transAxes)
                ax.axvline(0.0, color="0.4", linewidth=0.8)
                ax.axhline(0.0, color="0.4", linewidth=0.8)
                ax.set_xlim(-epoch_pre_s, epoch_post_s)
                continue

            ax.plot(subset["time_s"], subset["mean_z"], color="#1f77b4", linewidth=1.5)
            ax.fill_between(
                subset["time_s"],
                subset["mean_z"] - subset["sem_z"],
                subset["mean_z"] + subset["sem_z"],
                color="#1f77b4",
                alpha=0.25,
                linewidth=0,
            )
            ax.axvline(0.0, color="0.4", linewidth=0.8)
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-epoch_pre_s, epoch_post_s)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time relative to HR event (s)")

    fig.suptitle(f"{EXPLORATORY_NOTE}\nHR change events → EEG envelope z-scores", y=1.03, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_eeg_to_hr(
    eeg_to_hr_df: pd.DataFrame,
    qc_lookup: dict[str, EventTypeQc],
    output_path: Path,
    events_cfg: TemporalCouplingEventsConfig,
    *,
    epoch_pre_s: float,
    epoch_post_s: float,
) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    event_labels = {
        THETA_BURST: "Theta burst",
        ALPHA_SUPPRESSION: "Alpha suppression",
        BETA_BURST: "Beta burst",
    }

    for row_idx, event_type in enumerate(EEG_EVENT_TYPES):
        qc = qc_lookup[event_type]
        row_label = EEG_ROW_LABELS[event_type]
        axes[row_idx, 0].set_ylabel(f"{row_label}\nCardiac z")

        for col_idx, (var, label) in enumerate(zip(CARDIAC_VARS, CARDIAC_LABELS)):
            ax = axes[row_idx, col_idx]
            subset = eeg_to_hr_df.loc[
                (eeg_to_hr_df["event_type"] == event_type) & (eeg_to_hr_df["variable"] == var)
            ]
            title = _format_subplot_title(event_labels[event_type], label, qc, events_cfg)
            if subset.empty:
                ax.set_title(title, fontsize=9)
                ax.text(0.5, 0.5, "no curve", ha="center", va="center", transform=ax.transAxes)
                ax.axvline(0.0, color="0.4", linewidth=0.8)
                ax.axhline(0.0, color="0.4", linewidth=0.8)
                ax.set_xlim(-epoch_pre_s, epoch_post_s)
                continue

            ax.plot(subset["time_s"], subset["mean_z"], color="#2ca02c", linewidth=1.5)
            ax.fill_between(
                subset["time_s"],
                subset["mean_z"] - subset["sem_z"],
                subset["mean_z"] + subset["sem_z"],
                color="#2ca02c",
                alpha=0.25,
                linewidth=0,
            )
            ax.axvline(0.0, color="0.4", linewidth=0.8)
            ax.axhline(0.0, color="0.4", linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-epoch_pre_s, epoch_post_s)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time relative to EEG event (s)")

    fig.suptitle(
        f"{EXPLORATORY_NOTE}\nEEG envelope events → cardiac z-score trajectories",
        y=1.03,
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _qc_status_label(qc: EventTypeQc) -> str:
    if not qc.usable_for_group_plot:
        return "exploratory/insufficient"
    if qc.event_type == BETA_BURST:
        return "usable (cautious)"
    return "usable"


def _write_stage4_notes(
    cfg: TemporalCouplingConfig,
    qc_rows: list[EventTypeQc],
    n_subjects_processed: int,
    output_path: Path,
    *,
    partition: str | None = None,
) -> None:
    qc_lookup = _qc_lookup(qc_rows)
    lines: list[str] = [
        "Stage 4 temporal coupling interpretation notes",
        "============================================",
        f"dataset_id: {cfg.dataset_id}",
    ]
    if partition:
        lines.append(f"partition: {partition}")
    lines.extend(
        [
        f"n_subjects_processed: {n_subjects_processed}",
        "",
        "Averaging method",
        f"  averaging_method: {AVERAGING_METHOD}",
        "  subject_balanced_average: true",
        "  description: events are averaged within each subject first;",
        "    group curves are the mean of subject-level trajectories (one per subject).",
        "  note: subjects with many events do not dominate the group average.",
        "",
        "Usability thresholds",
        f"  min_total_events_required: {cfg.temporal_coupling.events.min_total_events}",
        f"  min_subjects_required: {cfg.temporal_coupling.events.min_contributing_subjects}",
        "",
        "Event-type status (this run)",
        ]
    )

    for event_type in ALL_EVENT_TYPES:
        qc = qc_lookup[event_type]
        status = _qc_status_label(qc)
        lines.append(
            f"  {event_type}: events={qc.total_events} subjects={qc.n_subjects} status={status}"
        )
        if qc.warning:
            lines.append(f"    warning: {qc.warning}")

    lines.extend(
        [
            "",
            "Interpretation guidance",
            "  - HR-triggered EEG plots are the most reliable Stage 4 result in this validation run.",
            "  - EEG-triggered cardiac plots are exploratory, especially theta burst and alpha suppression.",
            "  - Beta burst meets minimum counts but should be interpreted cautiously.",
            "  - No biological conclusion should be made until the full cohort is run.",
            "",
            "Caveats",
            f"  - {EXPLORATORY_NOTE}",
            "  - Event-triggered curves show z-scored aligned features only.",
            "  - Short rest recordings limit epoch windows and event yield.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_stage4_partition(
    cfg: TemporalCouplingConfig,
    subject_results: list[SubjectEventResult],
    *,
    group_dir: Path,
    epoch_times: np.ndarray,
    events_cfg: TemporalCouplingEventsConfig,
    partition: str | None = None,
) -> list[Path]:
    group_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    counts_rows = [result.counts.to_row() for result in subject_results]
    counts_df = pd.DataFrame(counts_rows, columns=list(EVENT_COUNTS_COLUMNS))
    counts_path = group_dir / EVENT_COUNTS_FILENAME
    counts_df.to_csv(counts_path, index=False)
    written.append(counts_path)

    all_event_records = [record for result in subject_results for record in result.event_records]
    event_list_df = pd.DataFrame(
        [record.to_row() for record in all_event_records],
        columns=list(EVENT_LIST_COLUMNS),
    )
    event_list_path = group_dir / EVENT_LIST_FILENAME
    event_list_df.to_csv(event_list_path, index=False)
    written.append(event_list_path)

    qc_rows = _build_event_qc(counts_df, events_cfg)
    qc_lookup = _qc_lookup(qc_rows)
    qc_df = pd.DataFrame([row.to_row() for row in qc_rows], columns=list(EVENT_QC_COLUMNS))
    qc_path = group_dir / EVENT_QC_FILENAME
    qc_df.to_csv(qc_path, index=False)
    written.append(qc_path)

    hr_to_eeg_rows = _build_hr_to_eeg_rows(subject_results, epoch_times)
    hr_to_eeg_df = pd.DataFrame(hr_to_eeg_rows, columns=list(HR_TO_EEG_COLUMNS))
    hr_to_eeg_path = group_dir / HR_TO_EEG_FILENAME
    hr_to_eeg_df.to_csv(hr_to_eeg_path, index=False)
    written.append(hr_to_eeg_path)

    eeg_to_hr_rows = _build_eeg_to_hr_rows(subject_results, epoch_times)
    eeg_to_hr_df = pd.DataFrame(eeg_to_hr_rows, columns=list(EEG_TO_HR_COLUMNS))
    eeg_to_hr_path = group_dir / EEG_TO_HR_FILENAME
    eeg_to_hr_df.to_csv(eeg_to_hr_path, index=False)
    written.append(eeg_to_hr_path)

    label = f"partition={partition!r} " if partition else ""
    print(f"[temporal_coupling] stage=4 {label}event QC:")
    print(
        f"[temporal_coupling]   averaging: {AVERAGING_METHOD} "
        f"(subject_balanced_average=true)"
    )
    for qc in qc_rows:
        status = _qc_status_label(qc)
        print(
            f"[temporal_coupling]   {qc.event_type}: events={qc.total_events} "
            f"subjects={qc.n_subjects} {status}"
            + (f" warning={qc.warning}" if qc.warning else "")
        )

    notes_path = group_dir / INTERPRETATION_NOTES_FILENAME
    _write_stage4_notes(
        cfg,
        qc_rows,
        len(subject_results),
        notes_path,
        partition=partition,
    )
    written.append(notes_path)
    print(f"[temporal_coupling] stage=4 {label}wrote interpretation notes -> {notes_path}")

    if cfg.temporal_coupling.output.save_plots:
        hr_plot_path = _plot_hr_to_eeg(
            hr_to_eeg_df,
            qc_lookup,
            group_dir / HR_TO_EEG_PLOT,
            events_cfg,
            epoch_pre_s=events_cfg.epoch_pre_s,
            epoch_post_s=events_cfg.epoch_post_s,
        )
        written.append(hr_plot_path)

        eeg_plot_path = _plot_eeg_to_hr(
            eeg_to_hr_df,
            qc_lookup,
            group_dir / EEG_TO_HR_PLOT,
            events_cfg,
            epoch_pre_s=events_cfg.epoch_pre_s,
            epoch_post_s=events_cfg.epoch_post_s,
        )
        written.append(eeg_plot_path)

    return written


def run_stage4(cfg: TemporalCouplingConfig) -> list[Path]:
    events_cfg = cfg.temporal_coupling.events
    fs_hz = float(cfg.temporal_coupling.resample.fs_hz)
    epoch_times = _build_epoch_times(events_cfg, fs_hz)

    alignment_qc = _load_alignment_qc(cfg)
    observation_ids = _discover_observation_ids(cfg, alignment_qc)

    if not observation_ids:
        print("[temporal_coupling] stage=4: no aligned observations found.")
        return []

    usable_ids = _usable_observation_ids(alignment_qc, observation_ids)
    partitioned_results: dict[str, list[SubjectEventResult]] = {}

    for observation_id in observation_ids:
        if observation_id not in usable_ids:
            warnings.warn(
                f"[temporal_coupling] stage=4 skipping {observation_id}: usable_for_xcorr=False.",
                stacklevel=2,
            )
            continue

        aligned_path = aligned_output_path(cfg, observation_id)
        if not aligned_path.is_file():
            warnings.warn(
                f"[temporal_coupling] stage=4 skipping {observation_id}: missing {aligned_path.name}. "
                "Run --stage 1c first.",
                stacklevel=2,
            )
            continue

        aligned_df = pd.read_csv(aligned_path)
        task = str(aligned_df["task"].iloc[0])
        condition = (
            str(aligned_df["condition"].iloc[0])
            if "condition" in aligned_df.columns and pd.notna(aligned_df["condition"].iloc[0])
            else task
        )
        partition = partition_key_from_row(
            dataset_id=cfg.dataset_id,
            task=task,
            condition=condition,
            observation_id=observation_id,
        )
        result = _process_subject(aligned_df, cfg, epoch_times)
        partitioned_results.setdefault(partition, []).append(result)
        print(
            f"[temporal_coupling] stage=4 {observation_id} (partition={partition}): "
            f"hr_increase={result.counts.hr_increase} hr_decrease={result.counts.hr_decrease} "
            f"theta_burst={result.counts.theta_burst} "
            f"alpha_suppression={result.counts.alpha_suppression} "
            f"beta_burst={result.counts.beta_burst}"
        )

    if not partitioned_results:
        print("[temporal_coupling] stage=4: no observations processed.")
        return []

    group_base = group_output_dir(cfg)
    n_partitions = len(partitioned_results)
    if n_partitions > 1:
        print(
            f"[temporal_coupling] stage=4: {n_partitions} partitions detected "
            f"({', '.join(sorted(partitioned_results))}); summarizing each separately."
        )

    written: list[Path] = []
    for partition in sorted(partitioned_results):
        out_dir = group_base if n_partitions == 1 else group_output_dir(cfg, partition)
        written.extend(
            _run_stage4_partition(
                cfg,
                partitioned_results[partition],
                group_dir=out_dir,
                epoch_times=epoch_times,
                events_cfg=events_cfg,
                partition=partition if n_partitions > 1 else None,
            )
        )

    print(f"[temporal_coupling] stage=4: {EXPLORATORY_NOTE}")
    return written
