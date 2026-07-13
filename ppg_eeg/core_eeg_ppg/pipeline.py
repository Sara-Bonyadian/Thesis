from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement, display_correlation_heatmaps, write_correlation_heatmap
from ..datasets import build_observations
from .features_core import (
    FeatureExtractionResult,
    base_eeg_power_columns,
    base_ppg_ibi_columns,
    derive_core_feature_tables_from_base_tables,
    extract_core_feature_tables,
)
from .output_layout import dataset_output_dir, subject_output_dir

OBSERVATIONS_INDEX_FILE = "observations_index.csv"
EEG_BASE_FILE = "features_base_eeg_power.csv"
PPG_IBI_BASE_FILE = "features_base_ppg_ibi.csv"
EEG_FEATURES_FILE = "features_eeg.csv"
PPG_FEATURES_FILE = "features_ppg.csv"
MERGED_FEATURES_FILE = "features_merged.csv"
CORRELATIONS_RAW_FILE = "correlations_raw.csv"
CORRELATIONS_FDR_FILE = "correlations_fdr.csv"
CORRELATIONS_HEATMAP_FILE = "correlations_heatmap.png"

STAGE1_DATASET_FILES: tuple[str, ...] = ()
STAGE1_SUBJECT_FILES: tuple[str, ...] = (
    OBSERVATIONS_INDEX_FILE,
    EEG_BASE_FILE,
    PPG_IBI_BASE_FILE,
)
STAGE2_DATASET_FILES: tuple[str, ...] = (
    CORRELATIONS_RAW_FILE,
    CORRELATIONS_FDR_FILE,
)
STAGE2_SUBJECT_FILES: tuple[str, ...] = (
    EEG_FEATURES_FILE,
    PPG_FEATURES_FILE,
    MERGED_FEATURES_FILE,
)
OBSERVATION_INDEX_COLUMNS: list[str] = [
    "dataset_id",
    "observation_id",
    "subject_id",
    "task_label",
    "condition_label",
    "eeg_path",
    "eeg_format",
    "ppg_source",
    "ppg_path",
    "ppg_format",
    "session_label",
    "modality",
    "timepoint",
    "state",
    "is_usable",
    "notes",
]


@dataclass(frozen=True)
class DatasetStage1Artifacts:
    dataset_id: str
    observations: pd.DataFrame
    eeg_base_features: pd.DataFrame
    ppg_ibi_features: pd.DataFrame


@dataclass(frozen=True)
class PipelineStage1Artifacts:
    per_dataset: dict[str, DatasetStage1Artifacts]


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    observations: pd.DataFrame
    eeg_features: pd.DataFrame
    eeg_base_features: pd.DataFrame
    ppg_features: pd.DataFrame
    ppg_ibi_features: pd.DataFrame
    merged_features: pd.DataFrame
    correlations_raw: pd.DataFrame
    correlations_fdr: pd.DataFrame


@dataclass(frozen=True)
class PipelineArtifacts:
    per_dataset: dict[str, DatasetArtifacts]
    trend_agreement: pd.DataFrame
    trend_summary: dict[str, Any]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _dataset_output_dir(cfg: PipelineConfig, dataset_id: str) -> Path:
    return dataset_output_dir(cfg, dataset_id)


def _subject_ids_from_observations(observations_df: pd.DataFrame) -> list[str]:
    if observations_df.empty or "subject_id" not in observations_df.columns:
        return []
    return sorted({str(value) for value in observations_df["subject_id"].dropna().astype(str).unique()})


def _filter_rows_for_subject(
    frame: pd.DataFrame,
    subject_id: str,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    subject_observations = observations_df[observations_df["subject_id"].astype(str) == str(subject_id)]
    if not subject_observations.empty and "observation_id" in frame.columns and "observation_id" in subject_observations.columns:
        observation_ids = set(subject_observations["observation_id"].astype(str))
        return frame[frame["observation_id"].astype(str).isin(observation_ids)].copy()

    if "subject_id" not in frame.columns:
        return frame.copy()
    return frame[frame["subject_id"].astype(str) == str(subject_id)].copy()


def _write_per_subject_csvs(
    cfg: PipelineConfig,
    dataset_id: str,
    artifacts_by_filename: dict[str, pd.DataFrame],
    *,
    observations_df: pd.DataFrame,
) -> None:
    observations_df = _ensure_columns(observations_df, OBSERVATION_INDEX_COLUMNS)
    subject_ids = _subject_ids_from_observations(observations_df)
    for subject_id in subject_ids:
        subject_dir = subject_output_dir(cfg, dataset_id, subject_id)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for filename, frame in artifacts_by_filename.items():
            _write_csv(
                _filter_rows_for_subject(frame, subject_id, observations_df),
                subject_dir / filename,
            )


def _discover_subject_dirs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and (path / OBSERVATIONS_INDEX_FILE).exists()
    )


def _concat_csvs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_stage1_subject_csvs(
    cfg: PipelineConfig,
    dataset_id: str,
    observations_df: pd.DataFrame,
    eeg_base_df: pd.DataFrame,
    ppg_ibi_df: pd.DataFrame,
) -> None:
    observations_df = _ensure_columns(observations_df, OBSERVATION_INDEX_COLUMNS)
    eeg_base_df = _prepare_base_eeg_df(eeg_base_df)
    ppg_ibi_df = _prepare_base_ppg_ibi_df(ppg_ibi_df)
    _write_per_subject_csvs(
        cfg,
        dataset_id,
        {
            OBSERVATIONS_INDEX_FILE: observations_df,
            EEG_BASE_FILE: eeg_base_df,
            PPG_IBI_BASE_FILE: ppg_ibi_df,
        },
        observations_df=observations_df,
    )


def write_stage1_dataset_csvs(
    cfg: PipelineConfig,
    dataset_id: str,
    observations_df: pd.DataFrame,
    eeg_base_df: pd.DataFrame,
    ppg_ibi_df: pd.DataFrame,
) -> None:
    """Write one combined Stage 1 CSV per table at the dataset folder level."""
    dataset_dir = _dataset_output_dir(cfg, dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    observations_df = _ensure_columns(observations_df, OBSERVATION_INDEX_COLUMNS)
    eeg_base_df = _prepare_base_eeg_df(eeg_base_df)
    ppg_ibi_df = _prepare_base_ppg_ibi_df(ppg_ibi_df)
    _write_csv(observations_df, dataset_dir / OBSERVATIONS_INDEX_FILE)
    _write_csv(eeg_base_df, dataset_dir / EEG_BASE_FILE)
    _write_csv(ppg_ibi_df, dataset_dir / PPG_IBI_BASE_FILE)


def load_stage1_tables_from_subject_dirs(
    cfg: PipelineConfig,
    dataset_id: str,
    *,
    dataset_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = dataset_dir or _dataset_output_dir(cfg, dataset_id)
    subject_dirs = _discover_subject_dirs(dataset_dir)
    if not subject_dirs:
        raise FileNotFoundError(
            f"No Stage 1 subject folders found under {dataset_dir}. "
            f"Expected subdirectories containing {OBSERVATIONS_INDEX_FILE!r}."
        )

    observations_frames: list[pd.DataFrame] = []
    eeg_base_frames: list[pd.DataFrame] = []
    ppg_ibi_frames: list[pd.DataFrame] = []
    for subject_dir in subject_dirs:
        observations_frames.append(_read_required_csv(subject_dir / OBSERVATIONS_INDEX_FILE))
        eeg_base_frames.append(_read_required_csv(subject_dir / EEG_BASE_FILE))
        ppg_ibi_frames.append(_read_required_csv(subject_dir / PPG_IBI_BASE_FILE))

    return (
        _concat_csvs(observations_frames),
        _concat_csvs(eeg_base_frames),
        _concat_csvs(ppg_ibi_frames),
    )


def _prepare_base_eeg_df(eeg_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = base_eeg_power_columns("export")
    return _ensure_columns(_subset_columns(eeg_df, base_columns), base_columns)


def _prepare_base_ppg_ibi_df(ppg_ibi_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = base_ppg_ibi_columns()
    return _ensure_columns(_subset_columns(ppg_ibi_df, base_columns), base_columns)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline input is missing: {path}")
    dtype: dict[str, type] = {}
    header = pd.read_csv(path, nrows=0)
    if "subject_id" in header.columns:
        dtype["subject_id"] = str
    return pd.read_csv(path, dtype=dtype)


def _subset_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in dict.fromkeys(columns) if col in df.columns]
    return df.loc[:, existing].copy()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.loc[:, columns].copy()


def _write_base_eeg_csv(dataset_dir: Path, eeg_df: pd.DataFrame) -> None:
    _write_csv(_prepare_base_eeg_df(eeg_df), dataset_dir / EEG_BASE_FILE)


def _write_base_ppg_ibi_csv(dataset_dir: Path, ppg_ibi_df: pd.DataFrame) -> None:
    _write_csv(_prepare_base_ppg_ibi_df(ppg_ibi_df), dataset_dir / PPG_IBI_BASE_FILE)


def _run_stage1_single_dataset(dataset_id: str, cfg: PipelineConfig) -> DatasetStage1Artifacts:
    observations = build_observations(
        dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects,
        tasks=cfg.tasks,
        conditions=cfg.conditions,
        sessions=cfg.sessions,
        subject_tasks=cfg.subject_tasks,
        subject_conditions=cfg.subject_conditions,
    )
    observations_df = _ensure_columns(
        pd.DataFrame([obs.to_record() for obs in observations]),
        OBSERVATION_INDEX_COLUMNS,
    )
    extracted = extract_core_feature_tables(observations, cfg)
    return DatasetStage1Artifacts(
        dataset_id=dataset_id,
        observations=observations_df,
        eeg_base_features=extracted.eeg_base_features,
        ppg_ibi_features=extracted.ppg_ibi_features,
    )


def run_stage1(cfg: PipelineConfig) -> PipelineStage1Artifacts:
    return PipelineStage1Artifacts(
        per_dataset={
            dataset_id: _run_stage1_single_dataset(dataset_id, cfg)
            for dataset_id in cfg.dataset_ids
        }
    )


def write_stage1_artifacts(cfg: PipelineConfig, artifacts: PipelineStage1Artifacts) -> None:
    Path(cfg.paths.out_root).mkdir(parents=True, exist_ok=True)
    for dataset_id, data in artifacts.per_dataset.items():
        _dataset_output_dir(cfg, dataset_id).mkdir(parents=True, exist_ok=True)
        write_stage1_subject_csvs(
            cfg,
            dataset_id,
            _ensure_columns(data.observations, OBSERVATION_INDEX_COLUMNS),
            data.eeg_base_features,
            data.ppg_ibi_features,
        )


def _dataset_artifacts_from_feature_result(
    *,
    dataset_id: str,
    observations_df: pd.DataFrame,
    features: FeatureExtractionResult,
    cfg: PipelineConfig,
) -> DatasetArtifacts:
    corr_raw = compute_pairwise_correlations(
        features.merged_features,
        eeg_features=cfg.features.eeg,
        ppg_features=cfg.features.ppg,
        cfg=cfg,
    )
    corr_fdr = apply_fdr(corr_raw, cfg=cfg)

    return DatasetArtifacts(
        dataset_id=dataset_id,
        observations=observations_df,
        eeg_features=features.eeg_features,
        eeg_base_features=features.eeg_base_features,
        ppg_features=features.ppg_features,
        ppg_ibi_features=features.ppg_ibi_features,
        merged_features=features.merged_features,
        correlations_raw=corr_raw,
        correlations_fdr=corr_fdr,
    )


def _build_pipeline_artifacts(per_dataset: dict[str, DatasetArtifacts], cfg: PipelineConfig) -> PipelineArtifacts:
    all_corr_fdr = [data.correlations_fdr for data in per_dataset.values()]
    if all_corr_fdr:
        all_corr_fdr_df = pd.concat(all_corr_fdr, ignore_index=True)
    else:
        all_corr_fdr_df = pd.DataFrame()
    trend_df, trend_summary = compute_trend_agreement(all_corr_fdr_df, cfg=cfg)
    return PipelineArtifacts(
        per_dataset=per_dataset,
        trend_agreement=trend_df,
        trend_summary=trend_summary,
    )


def run_stage2_from_stage1(
    cfg: PipelineConfig,
    stage1_artifacts: PipelineStage1Artifacts,
) -> PipelineArtifacts:
    per_dataset: dict[str, DatasetArtifacts] = {}
    for dataset_id, data in stage1_artifacts.per_dataset.items():
        features = derive_core_feature_tables_from_base_tables(
            data.observations,
            data.eeg_base_features,
            data.ppg_ibi_features,
            cfg,
        )
        per_dataset[dataset_id] = _dataset_artifacts_from_feature_result(
            dataset_id=dataset_id,
            observations_df=data.observations,
            features=features,
            cfg=cfg,
        )
    return _build_pipeline_artifacts(per_dataset, cfg)


def _run_stage2_single_dataset_from_csvs(dataset_id: str, cfg: PipelineConfig) -> DatasetArtifacts:
    observations_df, eeg_base_df, ppg_ibi_df = load_stage1_tables_from_subject_dirs(cfg, dataset_id)
    features = derive_core_feature_tables_from_base_tables(
        observations_df,
        eeg_base_df,
        ppg_ibi_df,
        cfg,
    )
    return _dataset_artifacts_from_feature_result(
        dataset_id=dataset_id,
        observations_df=observations_df,
        features=features,
        cfg=cfg,
    )


def run_stage2_from_base_csvs(cfg: PipelineConfig) -> PipelineArtifacts:
    per_dataset = {
        dataset_id: _run_stage2_single_dataset_from_csvs(dataset_id, cfg)
        for dataset_id in cfg.dataset_ids
    }
    return _build_pipeline_artifacts(per_dataset, cfg)


def write_stage2_artifacts(cfg: PipelineConfig, artifacts: PipelineArtifacts) -> dict[str, Any]:
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    heatmap_figures: dict[str, Any] = {}

    for dataset_id, data in artifacts.per_dataset.items():
        dataset_dir = _dataset_output_dir(cfg, dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(data.correlations_raw, dataset_dir / CORRELATIONS_RAW_FILE)
        _write_csv(data.correlations_fdr, dataset_dir / CORRELATIONS_FDR_FILE)
        if cfg.output.save_heatmap:
            heatmap_path = dataset_dir / CORRELATIONS_HEATMAP_FILE
            fig = write_correlation_heatmap(
                data.correlations_fdr,
                heatmap_path,
                dataset_id=dataset_id,
                method=cfg.correlation.primary_method,
            )
            if fig is not None:
                heatmap_figures[dataset_id] = fig
        _write_per_subject_csvs(
            cfg,
            dataset_id,
            {
                EEG_FEATURES_FILE: data.eeg_features,
                PPG_FEATURES_FILE: data.ppg_features,
                MERGED_FEATURES_FILE: data.merged_features,
            },
            observations_df=data.observations,
        )

    cross_dir = out_root / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts.trend_agreement, cross_dir / "trend_agreement.csv")
    if cfg.output.save_summary_json:
        summary_path = cross_dir / "trend_agreement_summary.json"
        summary_path.write_text(json.dumps(artifacts.trend_summary, indent=2))

    return heatmap_figures


def run_pipeline(cfg: PipelineConfig) -> PipelineArtifacts:
    """Run both stages in memory and return all artifacts."""

    stage1_artifacts = run_stage1(cfg)
    return run_stage2_from_stage1(cfg, stage1_artifacts)


def run_two_stage_pipeline(cfg: PipelineConfig) -> PipelineArtifacts:
    """Run Stage 1 to CSV, reload those base CSVs, then run and write Stage 2."""

    stage1_artifacts = run_stage1(cfg)
    write_stage1_artifacts(cfg, stage1_artifacts)
    artifacts = run_stage2_from_base_csvs(cfg)
    write_stage2_artifacts(cfg, artifacts)
    return artifacts


def write_artifacts(cfg: PipelineConfig, artifacts: PipelineArtifacts) -> None:
    stage1_artifacts = PipelineStage1Artifacts(
        per_dataset={
            dataset_id: DatasetStage1Artifacts(
                dataset_id=dataset_id,
                observations=data.observations,
                eeg_base_features=data.eeg_base_features,
                ppg_ibi_features=data.ppg_ibi_features,
            )
            for dataset_id, data in artifacts.per_dataset.items()
        }
    )
    write_stage1_artifacts(cfg, stage1_artifacts)
    write_stage2_artifacts(cfg, artifacts)
