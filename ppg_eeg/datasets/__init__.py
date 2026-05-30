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


def build_observations(
    dataset_id: str,
    raw_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
) -> list[CanonicalObservation]:
    adapter = get_adapter(dataset_id)
    return adapter.build_observations(
        raw_root,
        subjects=subjects,
        tasks=tasks,
        conditions=conditions,
        sessions=sessions,
    )


__all__ = [
    "ADAPTER_REGISTRY",
    "CanonicalObservation",
    "DatasetAdapter",
    "build_observations",
    "get_adapter",
]
