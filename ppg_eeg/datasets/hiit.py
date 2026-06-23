from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths


HIIT_STEM_RE = re.compile(
    r"^HIIT_(?P<subject>\d{2})_(?P<modality>PH|PS)_(?P<timepoint>PRE|POST)(?P<tetris>_T)?$"
)


def hiit_session_subject_id(participant_id: str, modality: str) -> str:
    """Participant + protocol session id, e.g. participant 01 PS -> '01_ps'."""
    return f"{participant_id}_{modality.casefold()}"


def hiit_task_condition_label(timepoint: str, state: str) -> str:
    """Task-only condition label, e.g. pre + rest -> pre_rest."""
    return f"{timepoint.casefold()}_{state.casefold()}"


def hiit_task_partition_from_observation_id(observation_id: str) -> str | None:
    """Parse hiit-01-ph-pre-rest -> pre_rest (ignores protocol)."""
    parts = observation_id.split("-")
    if len(parts) >= 5 and parts[0].casefold() == "hiit":
        return f"{parts[3]}_{parts[4]}"
    return None


def hiit_participant_id_from_session_subject(session_subject_id: str) -> str:
    """Bare participant id from a session subject id, e.g. '01_ps' -> '01'."""
    if "_" in session_subject_id:
        return session_subject_id.rsplit("_", 1)[0]
    return session_subject_id


class HIITAdapter(DatasetAdapter):
    dataset_id = "hiit"
    dataset_folder = "HIIT"

    def build_observations(
        self,
        raw_root: Path,
        *,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        conditions: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
        hiit_partition_mode: str = "protocol_task",
    ) -> list[CanonicalObservation]:
        dataset_root = self.resolve_dataset_root(raw_root)

        subject_filter = normalized_set(subjects, casefold=False)
        task_filter = normalized_set(tasks)
        condition_filter = normalized_set(conditions)
        session_filter = normalized_set(sessions)

        rows: list[CanonicalObservation] = []
        vhdr_files = unique_sorted_paths(dataset_root.glob("HIIT_*_*/*.vhdr"))
        for vhdr in vhdr_files:
            m = HIIT_STEM_RE.match(vhdr.stem)
            if not m:
                continue

            eeg_bin = vhdr.with_suffix(".eeg")
            marker = vhdr.with_suffix(".vmrk")
            if not eeg_bin.exists() or not marker.exists():
                continue

            subject = m.group("subject")
            if subject_filter is not None and subject not in subject_filter:
                continue

            modality = m.group("modality").casefold()
            timepoint = m.group("timepoint").casefold()
            state = "tetris" if m.group("tetris") else "rest"

            task_label = state
            if hiit_partition_mode == "task_only":
                condition_label = hiit_task_condition_label(timepoint, state)
            else:
                condition_label = f"{modality}_{timepoint}_{state}"
            session_label = modality
            session_subject_id = hiit_session_subject_id(subject, modality)

            if not include_if_in_filter(task_label, task_filter):
                continue
            if not include_if_in_filter(condition_label, condition_filter):
                continue
            if not include_if_in_filter(session_label, session_filter):
                continue

            rows.append(
                CanonicalObservation(
                    dataset_id=self.dataset_id,
                    observation_id=f"hiit-{subject}-{modality}-{timepoint}-{state}",
                    subject_id=session_subject_id,
                    task_label=task_label,
                    condition_label=condition_label,
                    eeg_path=vhdr,
                    eeg_format="brainvision",
                    ppg_source="embedded_eeg",
                    ppg_path=None,
                    ppg_format=None,
                    session_label=session_label,
                    modality=modality,
                    timepoint=timepoint,
                    state=state,
                    is_usable=True,
                    notes="PPG expected in photosensor/optical channels.",
                )
            )

        return rows
