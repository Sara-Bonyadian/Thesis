"""Shared output path helpers for temporal coupling (observation- and partition-aware)."""

from __future__ import annotations

from pathlib import Path

from ..datasets import CanonicalObservation
from ..output_layout import safe_subject_dir_name
from .config import TemporalCouplingConfig


def observation_output_dir(cfg: TemporalCouplingConfig, observation_id: str) -> Path:
    return Path(cfg.paths.out_root) / cfg.dataset_id / safe_subject_dir_name(observation_id)


def group_output_dir(cfg: TemporalCouplingConfig, partition: str | None = None) -> Path:
    base = Path(cfg.paths.out_root) / cfg.dataset_id / "group"
    if partition:
        return base / safe_subject_dir_name(partition)
    return base


def hiit_condition_from_observation_id(observation_id: str) -> str | None:
    """Parse hiit-01-ph-pre-rest -> ph_pre_rest."""
    parts = observation_id.split("-")
    if len(parts) >= 5 and parts[0].casefold() == "hiit":
        return f"{parts[2]}_{parts[3]}_{parts[4]}"
    return None


def analysis_partition_key(*, dataset_id: str, task: str, condition: str) -> str:
    if dataset_id.casefold() == "hiit":
        return condition
    return task or condition or "all"


def observation_partition_key(obs: CanonicalObservation) -> str:
    return analysis_partition_key(
        dataset_id=obs.dataset_id,
        task=obs.task_label,
        condition=obs.condition_label,
    )


def partition_key_from_row(
    *,
    dataset_id: str,
    task: str,
    condition: str | None = None,
    observation_id: str | None = None,
) -> str:
    if dataset_id.casefold() == "hiit" and observation_id:
        derived = hiit_condition_from_observation_id(observation_id)
        if derived:
            return derived
    return analysis_partition_key(
        dataset_id=dataset_id,
        task=task,
        condition=condition or task,
    )
