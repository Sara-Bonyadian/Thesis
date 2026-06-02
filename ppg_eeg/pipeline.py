from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement
from .datasets import build_observations
from .features_core import FeatureExtractionResult, base_eeg_power_columns, extract_core_feature_tables

EEG_FEATURE_ROW_KEYS: list[str] = ["dataset_id", "observation_id", "channel"]


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset_id: str
    observations: pd.DataFrame
    eeg_features: pd.DataFrame
    eeg_base_features: pd.DataFrame
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


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
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


def _eeg_feature_extension_columns(cfg: PipelineConfig, eeg_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for feature_name in cfg.features.eeg:
        if feature_name in eeg_df.columns:
            cols.append(feature_name)
        rz_col = f"{feature_name}_rz"
        if rz_col in eeg_df.columns:
            cols.append(rz_col)
    return list(dict.fromkeys(cols))


def _upsert_feature_columns(existing_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    for key in EEG_FEATURE_ROW_KEYS:
        if key not in existing_df.columns:
            raise ValueError(f"Existing EEG base CSV is missing key column: {key}")
        if key not in incoming_df.columns:
            raise ValueError(f"Incoming EEG base rows are missing key column: {key}")

    existing = existing_df.drop_duplicates(subset=EEG_FEATURE_ROW_KEYS, keep="last").copy()
    incoming = incoming_df.drop_duplicates(subset=EEG_FEATURE_ROW_KEYS, keep="last").copy()
    existing_idx = existing.set_index(EEG_FEATURE_ROW_KEYS)
    incoming_idx = incoming.set_index(EEG_FEATURE_ROW_KEYS)

    merged = existing_idx.reindex(existing_idx.index.union(incoming_idx.index))
    for col in incoming_idx.columns:
        merged[col] = incoming_idx[col]
    return merged.reset_index()


def _write_base_eeg_csv(cfg: PipelineConfig, dataset_dir: Path, eeg_df: pd.DataFrame) -> None:
    base_path = dataset_dir / "features_base_eeg_power.csv"
    base_export_columns = base_eeg_power_columns("export")
    base_rows = _ensure_columns(_subset_columns(eeg_df, base_export_columns), base_export_columns)
    mode = cfg.output.eeg_base_csv_mode

    if mode == "base_only":
        _write_csv(base_rows, base_path)
        return

    extension_cols = _eeg_feature_extension_columns(cfg, eeg_df)

    if mode == "append_columns":
        existing = _read_csv_if_exists(base_path)
        if existing is None:
            _write_csv(base_rows, base_path)
            existing = base_rows
        incoming_columns = [*base_export_columns, *extension_cols]
        incoming_rows = _ensure_columns(_subset_columns(eeg_df, incoming_columns), incoming_columns)
        merged = _upsert_feature_columns(existing, incoming_rows)
        _write_csv(merged, base_path)
        return

    if mode == "new_file":
        _write_csv(base_rows, base_path)
        if extension_cols:
            new_path = dataset_dir / cfg.output.eeg_base_csv_new_file_name
            new_columns = [*base_export_columns, *extension_cols]
            new_rows = _ensure_columns(_subset_columns(eeg_df, new_columns), new_columns)
            _write_csv(new_rows, new_path)
        return

    raise ValueError(f"Unsupported output.eeg_base_csv_mode: {mode!r}")


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
    dataset_dir = _dataset_output_dir(cfg, dataset_id)

    eeg_feature_cache: pd.DataFrame | None = None
    ppg_feature_cache: pd.DataFrame | None = None
    if cfg.features.reuse_eeg_features_csv:
        eeg_feature_cache = _read_csv_if_exists(dataset_dir / "features_core_eeg.csv")
        if eeg_feature_cache is None:
            eeg_feature_cache = _read_csv_if_exists(dataset_dir / cfg.output.eeg_base_csv_new_file_name)
        if eeg_feature_cache is None:
            eeg_feature_cache = _read_csv_if_exists(dataset_dir / "features_base_eeg_power.csv")
    if cfg.features.reuse_ppg_features_csv:
        ppg_feature_cache = _read_csv_if_exists(dataset_dir / "features_core_ppg.csv")

    features: FeatureExtractionResult = extract_core_feature_tables(
        observations,
        cfg,
        eeg_feature_cache=eeg_feature_cache,
        ppg_feature_cache=ppg_feature_cache,
    )
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
        _write_base_eeg_csv(cfg, dataset_dir, data.eeg_base_features)
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
