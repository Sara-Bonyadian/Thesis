from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement
from .datasets import build_observations
from .features_core import (
    FeatureExtractionResult,
    base_eeg_power_columns,
    base_ppg_ibi_columns,
    derive_core_feature_tables_from_base_tables,
    extract_core_feature_tables,
)

OBSERVATIONS_INDEX_FILE = "observations_index.csv"
EEG_BASE_FILE = "features_base_eeg_power.csv"
PPG_IBI_BASE_FILE = "features_base_ppg_ibi.csv"
EEG_FEATURES_FILE = "features_eeg.csv"
PPG_FEATURES_FILE = "features_ppg.csv"
MERGED_FEATURES_FILE = "features_merged.csv"
CORRELATIONS_RAW_FILE = "correlations_raw.csv"
CORRELATIONS_FDR_FILE = "correlations_fdr.csv"

STAGE1_DATASET_FILES: tuple[str, ...] = (
    OBSERVATIONS_INDEX_FILE,
    EEG_BASE_FILE,
    PPG_IBI_BASE_FILE,
)
STAGE2_DATASET_FILES: tuple[str, ...] = (
    EEG_FEATURES_FILE,
    PPG_FEATURES_FILE,
    MERGED_FEATURES_FILE,
    CORRELATIONS_RAW_FILE,
    CORRELATIONS_FDR_FILE,
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
    return Path(cfg.paths.out_root) / dataset_id


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline input is missing: {path}")
    return pd.read_csv(path)


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
    base_columns = base_eeg_power_columns("export")
    base_rows = _ensure_columns(_subset_columns(eeg_df, base_columns), base_columns)
    _write_csv(base_rows, dataset_dir / EEG_BASE_FILE)


def _write_base_ppg_ibi_csv(dataset_dir: Path, ppg_ibi_df: pd.DataFrame) -> None:
    base_columns = base_ppg_ibi_columns()
    base_rows = _ensure_columns(_subset_columns(ppg_ibi_df, base_columns), base_columns)
    _write_csv(base_rows, dataset_dir / PPG_IBI_BASE_FILE)


def _run_stage1_single_dataset(dataset_id: str, cfg: PipelineConfig) -> DatasetStage1Artifacts:
    observations = build_observations(
        dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects,
        tasks=cfg.tasks,
        conditions=cfg.conditions,
        sessions=cfg.sessions,
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
        dataset_dir = _dataset_output_dir(cfg, dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(data.observations, dataset_dir / OBSERVATIONS_INDEX_FILE)
        _write_base_eeg_csv(dataset_dir, data.eeg_base_features)
        _write_base_ppg_ibi_csv(dataset_dir, data.ppg_ibi_features)


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
    dataset_dir = _dataset_output_dir(cfg, dataset_id)
    observations_df = _read_required_csv(dataset_dir / OBSERVATIONS_INDEX_FILE)
    eeg_base_df = _read_required_csv(dataset_dir / EEG_BASE_FILE)
    ppg_ibi_df = _read_required_csv(dataset_dir / PPG_IBI_BASE_FILE)
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


def write_stage2_artifacts(cfg: PipelineConfig, artifacts: PipelineArtifacts) -> None:
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset_id, data in artifacts.per_dataset.items():
        dataset_dir = _dataset_output_dir(cfg, dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(data.eeg_features, dataset_dir / EEG_FEATURES_FILE)
        _write_csv(data.ppg_features, dataset_dir / PPG_FEATURES_FILE)
        _write_csv(data.merged_features, dataset_dir / MERGED_FEATURES_FILE)
        _write_csv(data.correlations_raw, dataset_dir / CORRELATIONS_RAW_FILE)
        _write_csv(data.correlations_fdr, dataset_dir / CORRELATIONS_FDR_FILE)

    cross_dir = out_root / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts.trend_agreement, cross_dir / "trend_agreement.csv")
    if cfg.output.save_summary_json:
        summary_path = cross_dir / "trend_agreement_summary.json"
        summary_path.write_text(json.dumps(artifacts.trend_summary, indent=2))


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
