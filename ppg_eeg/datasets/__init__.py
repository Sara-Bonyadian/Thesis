from __future__ import annotations

import inspect
from pathlib import Path
from typing import Sequence

from .base import (
    CanonicalObservation,
    CanonicalObservationIdentity,
    DatasetAdapter,
    canonical_observation_identity,
    validate_observation_identities,
)
from .ds003690 import DS003690Adapter
from .ds003816 import DS003816Adapter
from .ds003838 import DS003838Adapter
from .ds004511 import DS004511Adapter
from .ds004582 import DS004582Adapter
from .ds004587 import DS004587Adapter
from .ds006848 import DS006848Adapter
from .hiit import HIITAdapter
from .mindfulness import MindfulnessAdapter


ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "hiit": HIITAdapter,
    "mindfulness": MindfulnessAdapter,
    "ds003690": DS003690Adapter,
    "ds003816": DS003816Adapter,
    "ds003838": DS003838Adapter,
    "ds004511": DS004511Adapter,
    "ds004582": DS004582Adapter,
    "ds004587": DS004587Adapter,
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
        subject_key = observation.subject_id.casefold()
        participant_key = str(observation.participant_id or "").strip().casefold()
        # Allow assignments keyed by biological participant (``01``) or
        # session-qualified subject (``01_ph``).
        assigned_condition = assignments.get(subject_key)
        if assigned_condition is None and participant_key:
            assigned_condition = assignments.get(participant_key)
        if assigned_condition is None and "_" in subject_key:
            assigned_condition = assignments.get(subject_key.rsplit("_", 1)[0])
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
    build_params = inspect.signature(adapter.build_observations).parameters
    if "hiit_partition_mode" in build_params:
        build_kwargs["hiit_partition_mode"] = hiit_partition_mode
    observations = adapter.build_observations(raw_root, **build_kwargs)
    observations = _apply_subject_task_assignments(observations, subject_tasks)
    observations = _apply_subject_condition_assignments(observations, subject_conditions)
    validate_observation_identities(observations)
    return observations


__all__ = [
    "ADAPTER_REGISTRY",
    "CanonicalObservation",
    "CanonicalObservationIdentity",
    "DatasetAdapter",
    "build_observations",
    "canonical_observation_identity",
    "get_adapter",
    "validate_observation_identities",
]
