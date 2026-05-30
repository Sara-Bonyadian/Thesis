from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CanonicalObservation:
    """
    Dataset-agnostic record describing one analyzable EEG/PPG observation.
    """

    dataset_id: str
    observation_id: str
    subject_id: str
    task_label: str
    condition_label: str
    eeg_path: Path
    eeg_format: str
    ppg_source: str
    ppg_path: Path | None = None
    ppg_format: str | None = None
    session_label: str | None = None
    modality: str | None = None
    timepoint: str | None = None
    state: str | None = None
    is_usable: bool = True
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "observation_id": self.observation_id,
            "subject_id": self.subject_id,
            "task_label": self.task_label,
            "condition_label": self.condition_label,
            "eeg_path": str(self.eeg_path),
            "eeg_format": self.eeg_format,
            "ppg_source": self.ppg_source,
            "ppg_path": str(self.ppg_path) if self.ppg_path is not None else "",
            "ppg_format": self.ppg_format or "",
            "session_label": self.session_label or "",
            "modality": self.modality or "",
            "timepoint": self.timepoint or "",
            "state": self.state or "",
            "is_usable": bool(self.is_usable),
            "notes": self.notes,
        }


class DatasetAdapter(ABC):
    dataset_id: str
    dataset_folder: str

    @abstractmethod
    def build_observations(
        self,
        raw_root: Path,
        *,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        conditions: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
    ) -> list[CanonicalObservation]:
        raise NotImplementedError

    def resolve_dataset_root(self, raw_root: Path) -> Path:
        raw_root = Path(raw_root)
        if raw_root.name == self.dataset_folder and raw_root.exists():
            return raw_root

        candidate = raw_root / self.dataset_folder
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            f"Dataset root for {self.dataset_id!r} not found. "
            f"Expected {candidate} or provided root to be dataset folder itself."
        )


def normalized_set(values: Sequence[str] | None, *, casefold: bool = True) -> set[str] | None:
    if values is None:
        return None
    cleaned = [v.strip() for v in values if str(v).strip()]
    if not cleaned:
        return None
    if casefold:
        return {v.casefold() for v in cleaned}
    return set(cleaned)


def include_if_in_filter(value: str | None, allowed: set[str] | None) -> bool:
    if allowed is None:
        return True
    if value is None:
        return False
    return value.casefold() in allowed


def unique_sorted_paths(paths: Iterable[Path]) -> list[Path]:
    deduped = {Path(p) for p in paths}
    return sorted(deduped, key=lambda p: str(p))
