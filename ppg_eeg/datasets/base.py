from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_BIDS_ENTITY_RE = re.compile(r"(?:^|[-_])(sub|ses|run)-?([A-Za-z0-9]+)")


def _identity_token(value: str | None, default: str) -> str:
    text = str(value or "").strip().casefold()
    return text or default


@dataclass(frozen=True)
class CanonicalObservationIdentity:
    """Stable, dataset-scoped identity used by every confirmatory stage.

    ``pairing_id`` deliberately excludes ``condition_id``: it identifies the
    observation unit against which low/high condition observations are paired.
    Dataset adapters or YAML protocol mappings provide the component fields;
    this object never guesses from a dataset name.
    """

    dataset_id: str
    participant_id: str
    session_id: str
    run_id: str
    condition_id: str
    observation_id: str
    pairing_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "dataset_id": self.dataset_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "observation_id": self.observation_id,
            "pairing_id": self.pairing_id,
        }


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
    participant_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    condition_id: str | None = None
    pairing_id: str | None = None
    normalized_state: str | None = None
    normalized_time: str | None = None
    cardiac_modality: str | None = None
    cardiac_event_type: str | None = None
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
            "participant_id": self.participant_id or "",
            "session_id": self.session_id or "",
            "run_id": self.run_id or "",
            "condition_id": self.condition_id or "",
            "pairing_id": self.pairing_id or "",
            "normalized_state": self.normalized_state or "",
            "normalized_time": self.normalized_time or "",
            "cardiac_modality": self.cardiac_modality or "",
            "cardiac_event_type": self.cardiac_event_type or "",
            "is_usable": bool(self.is_usable),
            "notes": self.notes,
        }

    @property
    def identity(self) -> CanonicalObservationIdentity:
        return canonical_observation_identity(self)


def canonical_observation_identity(
    observation: CanonicalObservation,
    *,
    participant_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    condition_id: str | None = None,
) -> CanonicalObservationIdentity:
    """Construct the one normalized identity representation for an observation."""
    dataset = _identity_token(observation.dataset_id, "unknown")
    participant = _identity_token(
        participant_id if participant_id is not None else observation.participant_id or observation.subject_id,
        "unknown",
    )
    session = _identity_token(
        session_id if session_id is not None else observation.session_id or observation.session_label,
        "single",
    )
    inferred_run = observation.run_id
    if not inferred_run:
        entities = {name: value for name, value in _BIDS_ENTITY_RE.findall(observation.observation_id)}
        inferred_run = entities.get("run")
    run = _identity_token(run_id if run_id is not None else inferred_run, "single")
    condition = _identity_token(
        condition_id if condition_id is not None else observation.condition_id or observation.condition_label,
        "unknown",
    )
    observation_id = _identity_token(observation.observation_id, "unknown")
    pairing = _identity_token(
        observation.pairing_id,
        f"{dataset}::{participant}::{session}::{run}",
    )
    return CanonicalObservationIdentity(
        dataset_id=dataset,
        participant_id=participant,
        session_id=session,
        run_id=run,
        condition_id=condition,
        observation_id=observation_id,
        pairing_id=pairing,
    )


def validate_observation_identities(
    observations: Iterable[CanonicalObservation],
) -> None:
    """Fail fast when two raw observations collapse to one canonical identity."""
    seen: dict[tuple[str, str], CanonicalObservationIdentity] = {}
    for observation in observations:
        identity = observation.identity
        key = (identity.dataset_id, identity.observation_id)
        previous = seen.get(key)
        if previous is not None and previous != identity:
            raise ValueError(
                "Canonical observation identity collision for "
                f"{identity.dataset_id}/{identity.observation_id}: "
                f"{previous.to_record()} != {identity.to_record()}"
            )
        seen[key] = identity


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
    """Deduplicate and sort paths, dropping macOS AppleDouble ``._*`` files.

    External volumes often store Finder metadata as ``._sub-..._eeg.set``.
    Those names still match ``*_eeg.set`` globs and parse as BIDS entities
    (``._sub-AB10_...`` splits to ``sub-AB10``), so they must be filtered here.
    """
    deduped = {Path(p) for p in paths if not Path(p).name.startswith("._")}
    return sorted(deduped, key=lambda p: str(p))
