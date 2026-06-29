from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths


def _parse_bids_eeg_stem(stem: str) -> dict[str, str]:
    """
    Parse BIDS-like stem components from names such as:
    sub-S210317_ses-01_task-Rest_run-01_eeg
    """
    entities: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key and value:
            entities[key] = value
    return entities


class DS004511Adapter(DatasetAdapter):
    dataset_id = "ds004511"
    dataset_folder = "ds004511"

    def build_observations(
        self,
        raw_root: Path,
        *,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        conditions: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
    ) -> list[CanonicalObservation]:
        dataset_root = self.resolve_dataset_root(raw_root)

        subject_filter = normalized_set(subjects)
        task_filter = normalized_set(tasks)
        condition_filter = normalized_set(conditions)
        session_filter = normalized_set(sessions)

        rows: list[CanonicalObservation] = []
        for eeg_json in unique_sorted_paths(dataset_root.glob("sub-*/ses-*/eeg/*_eeg.json")):
            entities = _parse_bids_eeg_stem(eeg_json.stem)
            subject = entities.get("sub")
            task = entities.get("task")
            session = entities.get("ses")
            run = entities.get("run")
            if subject is None or task is None:
                continue

            task_label = task.casefold()
            session_label = session.casefold() if session else "single"
            condition_label = task_label

            if not include_if_in_filter(subject, subject_filter):
                continue
            if not include_if_in_filter(task_label, task_filter):
                continue
            if not include_if_in_filter(condition_label, condition_filter):
                continue
            if not include_if_in_filter(session_label, session_filter):
                continue

            eeg_path = eeg_json.with_suffix(".edf")
            if not eeg_path.exists():
                continue

            run_suffix = f"-run-{run.casefold()}" if run else ""
            subject_instance = subject.casefold()
            if session:
                subject_instance = f"{subject_instance}_ses-{session.casefold()}"
            if run:
                subject_instance = f"{subject_instance}_run-{run.casefold()}"
            obs_id = (
                f"{self.dataset_id}-{subject.casefold()}-ses-{session_label}"
                f"-task-{task_label}{run_suffix}"
            )
            rows.append(
                CanonicalObservation(
                    dataset_id=self.dataset_id,
                    observation_id=obs_id,
                    subject_id=subject_instance,
                    task_label=task_label,
                    condition_label=condition_label,
                    eeg_path=eeg_path,
                    eeg_format="edf",
                    ppg_source="embedded_eeg",
                    ppg_path=None,
                    ppg_format=None,
                    session_label=session_label,
                    modality="eeg_ecg",
                    timepoint="na",
                    state=task_label,
                    is_usable=True,
                    notes="ECG expected as embedded channel in EDF EEG files.",
                )
            )

        return rows
