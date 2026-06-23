from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter
from .ds003838 import DS003838Adapter
from .ds006848 import DS006848Adapter
from .hiit import HIITAdapter


ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "hiit": HIITAdapter,
    "ds003838": DS003838Adapter,
    "ds006848": DS006848Adapter,
}


def get_adapter(dataset_id: str) -> DatasetAdapter:
    key = dataset_id.casefold()
    if key not in ADAPTER_REGISTRY:
        known = ", ".join(sorted(ADAPTER_REGISTRY))
        raise KeyError(f"Unknown dataset_id={dataset_id!r}. Known: {known}")
    return ADAPTER_REGISTRY[key]()


def _apply_subject_task_assignments(
    observations: list[CanonicalObservation],
    subject_tasks: dict[str, str] | None,
) -> list[CanonicalObservation]:
    if not subject_tasks:
        return observations

    assignments = {
        str(subject_id).strip().casefold(): str(task_label).strip().casefold()
        for subject_id, task_label in subject_tasks.items()
        if str(subject_id).strip() and str(task_label).strip()
    }
    if not assignments:
        return observations

    filtered: list[CanonicalObservation] = []
    for observation in observations:
        assigned_task = assignments.get(observation.subject_id.casefold())
        if assigned_task is None or observation.task_label.casefold() == assigned_task:
            filtered.append(observation)
    return filtered


def _apply_subject_condition_assignments(
    observations: list[CanonicalObservation],
    subject_conditions: dict[str, str] | None,
) -> list[CanonicalObservation]:
    if not subject_conditions:
        return observations

    assignments = {
        str(subject_id).strip().casefold(): str(condition_label).strip().casefold()
        for subject_id, condition_label in subject_conditions.items()
        if str(subject_id).strip() and str(condition_label).strip()
    }
    if not assignments:
        return observations

    filtered: list[CanonicalObservation] = []
    for observation in observations:
        assigned_condition = assignments.get(observation.subject_id.casefold())
        if assigned_condition is None or observation.condition_label.casefold() == assigned_condition:
            filtered.append(observation)
    return filtered


def build_observations(
    dataset_id: str,
    raw_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
    subject_tasks: dict[str, str] | None = None,
    subject_conditions: dict[str, str] | None = None,
    hiit_partition_mode: str = "protocol_task",
) -> list[CanonicalObservation]:
    adapter = get_adapter(dataset_id)
    build_kwargs: dict[str, object] = {
        "subjects": subjects,
        "tasks": tasks,
        "conditions": conditions,
        "sessions": sessions,
    }
    if dataset_id.casefold() == "hiit":
        build_kwargs["hiit_partition_mode"] = hiit_partition_mode
    observations = adapter.build_observations(raw_root, **build_kwargs)
    observations = _apply_subject_task_assignments(observations, subject_tasks)
    return _apply_subject_condition_assignments(observations, subject_conditions)


__all__ = [
    "ADAPTER_REGISTRY",
    "CanonicalObservation",
    "DatasetAdapter",
    "build_observations",
    "get_adapter",
]
