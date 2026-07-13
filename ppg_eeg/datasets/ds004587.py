from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths


def _parse_bids_eeg_stem(stem: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key and value:
            entities[key] = value
    return entities


def _physio_path_for_eeg(vhdr: Path) -> Path:
    return vhdr.parent.parent / "beh" / f"{vhdr.stem.replace('_eeg', '_physio')}.tsv.gz"


def _rest_physio_path_for_eeg(vhdr: Path) -> Path:
    entities = _parse_bids_eeg_stem(vhdr.stem)
    subject = entities.get("sub")
    session = entities.get("ses")
    if subject is None or session is None:
        return vhdr.parent.parent / "beh" / "missing_task-rest_physio.tsv.gz"
    return vhdr.parent.parent / "beh" / f"sub-{subject}_ses-{session}_task-rest_physio.tsv.gz"


def _append_ig_observation(
    rows: list[CanonicalObservation],
    *,
    vhdr: Path,
    subject: str,
    task_label: str,
    session_label: str,
    run: str | None,
) -> None:
    eeg_bin = vhdr.with_suffix(".eeg")
    marker = vhdr.with_suffix(".vmrk")
    physio_path = _physio_path_for_eeg(vhdr)
    if not eeg_bin.is_file() or not marker.is_file() or not physio_path.is_file():
        return

    run_suffix = f"-run-{run.casefold()}" if run else ""
    subject_instance = subject.casefold()
    if session_label != "single":
        subject_instance = f"{subject_instance}_ses-{session_label}"
    if run:
        subject_instance = f"{subject_instance}_run-{run.casefold()}"
    obs_id = f"ds004587-{subject.casefold()}-ses-{session_label}-task-{task_label}{run_suffix}"
    rows.append(
        CanonicalObservation(
            dataset_id="ds004587",
            observation_id=obs_id,
            subject_id=subject_instance,
            task_label=task_label,
            condition_label=task_label,
            eeg_path=vhdr,
            eeg_format="brainvision",
            ppg_source="external_file",
            ppg_path=physio_path,
            ppg_format="bids_physio",
            session_label=session_label,
            modality="eeg_physio",
            timepoint="na",
            state=task_label,
            is_usable=True,
            notes="Cardiac from beh/*_task-IG*_physio.tsv.gz (ECGBIT/OXIBIT); HR computed via peak detection.",
        )
    )


def _append_rest_observation(
    rows: list[CanonicalObservation],
    *,
    vhdr: Path,
    subject: str,
    session_label: str,
) -> None:
    eeg_bin = vhdr.with_suffix(".eeg")
    marker = vhdr.with_suffix(".vmrk")
    rest_physio_path = _rest_physio_path_for_eeg(vhdr)
    if not eeg_bin.is_file() or not marker.is_file() or not rest_physio_path.is_file():
        return

    task_label = "rest"
    subject_instance = subject.casefold()
    if session_label != "single":
        subject_instance = f"{subject_instance}_ses-{session_label}"
    obs_id = f"ds004587-{subject.casefold()}-ses-{session_label}-task-{task_label}"
    rows.append(
        CanonicalObservation(
            dataset_id="ds004587",
            observation_id=obs_id,
            subject_id=subject_instance,
            task_label=task_label,
            condition_label=task_label,
            eeg_path=vhdr,
            eeg_format="brainvision",
            ppg_source="external_file",
            ppg_path=rest_physio_path,
            ppg_format="bids_physio",
            session_label=session_label,
            modality="eeg_physio",
            timepoint="na",
            state=task_label,
            is_usable=True,
            notes=(
                "Rest eyes-closed segment from session-start IG EEG paired with "
                "beh/*_task-rest_physio.tsv.gz; crop EEG to first 8 min via task_time_windows.rest."
            ),
        )
    )


class DS004587Adapter(DatasetAdapter):
    dataset_id = "ds004587"
    dataset_folder = "ds004587"

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

        include_ig = include_if_in_filter("ig", task_filter)
        include_rest = include_if_in_filter("rest", task_filter)

        rows: list[CanonicalObservation] = []
        for vhdr in unique_sorted_paths(dataset_root.glob("sub-*/ses-*/eeg/*_eeg.vhdr")):
            entities = _parse_bids_eeg_stem(vhdr.stem)
            subject = entities.get("sub")
            task = entities.get("task")
            session = entities.get("ses")
            run = entities.get("run")
            if subject is None or task is None:
                continue

            task_label = task.casefold()
            session_label = session.casefold() if session else "single"

            if not include_if_in_filter(subject, subject_filter):
                continue
            if not include_if_in_filter(session_label, session_filter):
                continue

            if task_label != "ig":
                continue
            if run is not None and run.casefold() != "01":
                continue

            if include_ig and include_if_in_filter(task_label, task_filter) and include_if_in_filter(
                task_label, condition_filter
            ):
                _append_ig_observation(
                    rows,
                    vhdr=vhdr,
                    subject=subject,
                    task_label=task_label,
                    session_label=session_label,
                    run=run,
                )

            if include_rest and include_if_in_filter("rest", condition_filter):
                _append_rest_observation(
                    rows,
                    vhdr=vhdr,
                    subject=subject,
                    session_label=session_label,
                )

        return rows
