"""Combine HIIT rest + tetris aligned segments into PRE/POST observations per protocol."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TemporalCouplingConfig
from .data_audit import audit_output_path, group_output_dir
from ..datasets.hiit import hiit_session_subject_id
from .eeg_envelope import UsableObservation
from .resample import (
    ALIGNED_FILENAME,
    CARDIAC_COLUMNS,
    EEG_ENVELOPE_COLUMNS,
    RAW_COLUMNS,
    Z_COLUMNS,
    AlignmentQcRecord,
    OverlapRange,
    aligned_output_path,
    alignment_qc_group_path,
    build_alignment_qc,
    zscore_within_observation,
)

AUDIT_COLUMNS = (
    "dataset_id",
    "subject_id",
    "task",
    "condition",
    "observation_id",
    "eeg_file",
    "eeg_format",
    "cardiac_file",
    "cardiac_format",
    "ppg_source",
    "eeg_exists",
    "cardiac_exists",
    "eeg_duration_s",
    "cardiac_duration_s",
    "overlap_duration_s",
    "eeg_sfreq",
    "cardiac_sfreq",
    "usable",
    "skip_reason",
    "recommended_max_lag_s",
)


@dataclass(frozen=True)
class HiitCombinedPartition:
    condition: str
    modality: str
    timepoint: str
    rest_suffix: str
    tetris_suffix: str


HIIT_COMBINED_PARTITIONS: tuple[HiitCombinedPartition, ...] = (
    HiitCombinedPartition("ps_pre", "ps", "pre", "ps-pre-rest", "ps-pre-tetris"),
    HiitCombinedPartition("ps_post", "ps", "post", "ps-post-rest", "ps-post-tetris"),
    HiitCombinedPartition("ph_pre", "ph", "pre", "ph-pre-rest", "ph-pre-tetris"),
    HiitCombinedPartition("ph_post", "ph", "post", "ph-post-rest", "ph-post-tetris"),
)


def combined_observation_id(*, subject_id: str, modality: str, timepoint: str) -> str:
    return f"hiit-{subject_id}-{modality}-{timepoint}"


def source_observation_id(*, subject_id: str, suffix: str) -> str:
    return f"hiit-{subject_id}-{suffix}"


def source_aligned_path(source_root: Path, dataset_id: str, observation_id: str) -> Path:
    return source_root / dataset_id / observation_id / ALIGNED_FILENAME


def combine_aligned_dataframes(
    rest_df: pd.DataFrame,
    tetris_df: pd.DataFrame,
    *,
    combined_observation_id: str,
    combined_condition: str,
    timepoint: str,
    session_subject_id: str,
    fs_hz: float,
    z_score: bool,
) -> pd.DataFrame:
    rest = rest_df.sort_values("time_s").reset_index(drop=True)
    tetris = tetris_df.sort_values("time_s").reset_index(drop=True)
    combined_raw = pd.concat(
        [rest[list(RAW_COLUMNS)], tetris[list(RAW_COLUMNS)]],
        ignore_index=True,
    )
    n_rows = len(combined_raw)
    time_s = np.arange(n_rows, dtype=float) / fs_hz

    row0 = rest.iloc[0]
    aligned: dict[str, object] = {
        "dataset_id": row0["dataset_id"],
        "subject_id": session_subject_id,
        "task": timepoint,
        "condition": combined_condition,
        "observation_id": combined_observation_id,
        "time_s": time_s,
    }
    for col in RAW_COLUMNS:
        aligned[col] = combined_raw[col].to_numpy(dtype=float)

    if z_score:
        for raw_col, z_col in zip(RAW_COLUMNS, Z_COLUMNS, strict=True):
            aligned[z_col] = zscore_within_observation(np.asarray(aligned[raw_col], dtype=float))
    else:
        for z_col in Z_COLUMNS:
            aligned[z_col] = np.full(n_rows, np.nan, dtype=float)

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


def _source_alignment_qc_path(source_root: Path, dataset_id: str) -> Path:
    return source_root / dataset_id / "group" / "alignment_qc.csv"


def _usable_source_ids(source_qc: pd.DataFrame) -> set[str]:
    usable = source_qc.loc[
        source_qc["usable_for_xcorr"].astype(str).str.lower().isin({"true", "1", "yes"})
    ]
    return set(usable["observation_id"].astype(str))


def _subjects_from_source(source_qc: pd.DataFrame) -> list[str]:
    participant_ids: set[str] = set()
    for oid in source_qc["observation_id"].astype(str):
        if not oid.startswith("hiit-"):
            continue
        parts = oid.split("-")
        if len(parts) >= 2:
            participant_ids.add(parts[1])
    return sorted(participant_ids)


def _build_audit_row(
    *,
    cfg: TemporalCouplingConfig,
    subject_id: str,
    partition: HiitCombinedPartition,
    observation_id: str,
    aligned_df: pd.DataFrame,
    qc: AlignmentQcRecord,
) -> dict[str, object]:
    aligned_path = aligned_output_path(cfg, observation_id)
    return {
        "dataset_id": cfg.dataset_id,
        "subject_id": subject_id,
        "task": partition.timepoint,
        "condition": partition.condition,
        "observation_id": observation_id,
        "eeg_file": str(aligned_path),
        "eeg_format": "combined_aligned",
        "cardiac_file": str(aligned_path),
        "cardiac_format": "combined_aligned",
        "ppg_source": "embedded_eeg",
        "eeg_exists": True,
        "cardiac_exists": True,
        "eeg_duration_s": qc.aligned_duration_s,
        "cardiac_duration_s": qc.aligned_duration_s,
        "overlap_duration_s": qc.aligned_duration_s,
        "eeg_sfreq": qc.fs_hz,
        "cardiac_sfreq": qc.fs_hz,
        "usable": qc.usable_for_xcorr,
        "skip_reason": qc.warning if not qc.usable_for_xcorr else "",
        "recommended_max_lag_s": qc.recommended_xcorr_lag_s,
    }


def run_stage1d(cfg: TemporalCouplingConfig) -> list[Path]:
    if cfg.dataset_id.casefold() != "hiit":
        raise ValueError("stage=1d HIIT combine is only supported for dataset_id='hiit'.")

    source_root = cfg.paths.source_out_root
    if source_root is None:
        raise ValueError("paths.source_out_root is required for stage=1d HIIT combine.")

    source_qc_path = _source_alignment_qc_path(source_root, cfg.dataset_id)
    if not source_qc_path.is_file():
        raise FileNotFoundError(
            f"Missing source alignment QC: {source_qc_path}. "
            "Run the full HIIT pipeline (stages 1a–1c) on source_out_root first."
        )

    source_qc = pd.read_csv(source_qc_path)
    usable_source = _usable_source_ids(source_qc)
    subjects = _subjects_from_source(source_qc)
    if cfg.subjects:
        subjects = [sid for sid in subjects if sid in set(cfg.subjects)]

    fs_hz = float(cfg.temporal_coupling.resample.fs_hz)
    z_score = bool(cfg.temporal_coupling.resample.z_score)
    written: list[Path] = []
    qc_records: list[AlignmentQcRecord] = []
    audit_rows: list[dict[str, object]] = []
    n_ok = 0

    group_dir = group_output_dir(cfg)
    group_dir.mkdir(parents=True, exist_ok=True)

    for participant_id in subjects:
        for partition in HIIT_COMBINED_PARTITIONS:
            if cfg.conditions and partition.condition not in cfg.conditions:
                continue

            session_subject_id = hiit_session_subject_id(participant_id, partition.modality)
            rest_id = source_observation_id(subject_id=participant_id, suffix=partition.rest_suffix)
            tetris_id = source_observation_id(subject_id=participant_id, suffix=partition.tetris_suffix)
            obs_id = combined_observation_id(
                subject_id=participant_id,
                modality=partition.modality,
                timepoint=partition.timepoint,
            )

            if rest_id not in usable_source or tetris_id not in usable_source:
                warnings.warn(
                    f"[temporal_coupling] stage=1d skipping {obs_id}: "
                    f"source segment not usable ({rest_id}, {tetris_id}).",
                    stacklevel=2,
                )
                continue

            rest_path = source_aligned_path(source_root, cfg.dataset_id, rest_id)
            tetris_path = source_aligned_path(source_root, cfg.dataset_id, tetris_id)
            if not rest_path.is_file() or not tetris_path.is_file():
                warnings.warn(
                    f"[temporal_coupling] stage=1d skipping {obs_id}: missing aligned segment file.",
                    stacklevel=2,
                )
                continue

            rest_df = pd.read_csv(rest_path)
            tetris_df = pd.read_csv(tetris_path)
            aligned_df = combine_aligned_dataframes(
                rest_df,
                tetris_df,
                combined_observation_id=obs_id,
                combined_condition=partition.condition,
                timepoint=partition.timepoint,
                session_subject_id=session_subject_id,
                fs_hz=fs_hz,
                z_score=z_score,
            )

            out_path = aligned_output_path(cfg, obs_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(out_path, index=False)
            written.append(out_path)

            duration_s = float(aligned_df["time_s"].iloc[-1] - aligned_df["time_s"].iloc[0])
            overlap = OverlapRange(start_s=float(aligned_df["time_s"].iloc[0]), end_s=float(aligned_df["time_s"].iloc[-1]))
            obs = UsableObservation(
                dataset_id=cfg.dataset_id,
                subject_id=session_subject_id,
                task=partition.timepoint,
                condition=partition.condition,
                observation_id=obs_id,
                eeg_file=out_path,
                eeg_format="combined_aligned",
            )
            qc = build_alignment_qc(obs, aligned_df, overlap, cfg)
            qc_records.append(qc)
            audit_rows.append(
                _build_audit_row(
                    cfg=cfg,
                    subject_id=session_subject_id,
                    partition=partition,
                    observation_id=obs_id,
                    aligned_df=aligned_df,
                    qc=qc,
                )
            )
            n_ok += 1
            print(
                f"[temporal_coupling] stage=1d {obs_id}: "
                f"rest_rows={len(rest_df)} tetris_rows={len(tetris_df)} "
                f"combined_rows={len(aligned_df)} duration_s={duration_s:.1f} "
                f"usable={qc.usable_for_xcorr} -> {out_path}"
            )

    if audit_rows:
        audit_path = audit_output_path(cfg)
        pd.DataFrame(audit_rows, columns=list(AUDIT_COLUMNS)).to_csv(audit_path, index=False)
        written.append(audit_path)
        print(f"[temporal_coupling] stage=1d wrote audit -> {audit_path}")

    if qc_records:
        qc_path = alignment_qc_group_path(cfg)
        pd.DataFrame([record.to_row() for record in qc_records]).to_csv(qc_path, index=False)
        written.append(qc_path)
        print(f"[temporal_coupling] stage=1d wrote alignment QC -> {qc_path}")

    print(
        f"[temporal_coupling] stage=1d summary: wrote={n_ok} combined observations "
        f"({len(HIIT_COMBINED_PARTITIONS)} partitions × {len(subjects)} subjects max)"
    )
    return written
