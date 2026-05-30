from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement
from .datasets import build_observations
from .features_core import FeatureExtractionResult, extract_core_feature_tables


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    observations: pd.DataFrame
    eeg_features: pd.DataFrame
    ppg_features: pd.DataFrame
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


def _run_single_dataset(dataset_id: str, cfg: PipelineConfig) -> DatasetArtifacts:
    observations = build_observations(
        dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects,
        tasks=cfg.tasks,
        conditions=cfg.conditions,
        sessions=cfg.sessions,
    )
    observations_df = pd.DataFrame([obs.to_record() for obs in observations])

    features: FeatureExtractionResult = extract_core_feature_tables(observations, cfg)
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
        ppg_features=features.ppg_features,
        merged_features=features.merged_features,
        correlations_raw=corr_raw,
        correlations_fdr=corr_fdr,
    )


def run_pipeline(cfg: PipelineConfig) -> PipelineArtifacts:
    per_dataset: dict[str, DatasetArtifacts] = {}
    all_corr_fdr: list[pd.DataFrame] = []

    for dataset_id in cfg.dataset_ids:
        artifacts = _run_single_dataset(dataset_id, cfg)
        per_dataset[dataset_id] = artifacts
        all_corr_fdr.append(artifacts.correlations_fdr)

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


def write_artifacts(cfg: PipelineConfig, artifacts: PipelineArtifacts) -> None:
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset_id, data in artifacts.per_dataset.items():
        dataset_dir = _dataset_output_dir(cfg, dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if cfg.output.save_observation_index:
            _write_csv(data.observations, dataset_dir / "observations_index.csv")
        _write_csv(data.eeg_features, dataset_dir / "features_core_eeg.csv")
        _write_csv(data.ppg_features, dataset_dir / "features_core_ppg.csv")
        _write_csv(data.merged_features, dataset_dir / "features_core_merged.csv")
        _write_csv(data.correlations_raw, dataset_dir / "correlations_raw.csv")
        _write_csv(data.correlations_fdr, dataset_dir / "correlations_fdr.csv")

    cross_dir = out_root / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts.trend_agreement, cross_dir / "trend_agreement.csv")
    if cfg.output.save_summary_json:
        summary_path = cross_dir / "trend_agreement_summary.json"
        summary_path.write_text(json.dumps(artifacts.trend_summary, indent=2))
