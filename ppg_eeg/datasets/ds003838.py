from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths


EEG_STEM_RE = re.compile(r"^(?P<subject>sub-\d+)_task-(?P<task>[a-zA-Z0-9]+)_eeg$")
ECG_STEM_RE = re.compile(r"^(?P<subject>sub-\d+)_task-(?P<task>[a-zA-Z0-9]+)_ecg$")


class DS003838Adapter(DatasetAdapter):
    dataset_id = "ds003838"
    dataset_folder = "ds003838"

    def build_observations(
        self,
        raw_root: Path,
        *,
        subjects: Sequence[str] | None = None,
        tasks: Sequence[str] | None = None,
        conditions: Sequence[str] | None = None,
        sessions: Sequence[str] | None = None,
    ) -> list[CanonicalObservation]:
        # sessions are not present in this dataset but accepted for interface parity.
        _ = sessions

        dataset_root = self.resolve_dataset_root(raw_root)

        subject_filter = normalized_set(subjects)
        task_filter = normalized_set(tasks)
        condition_filter = normalized_set(conditions)

        eeg_by_key: dict[tuple[str, str], Path] = {}
        for eeg_set in unique_sorted_paths(dataset_root.glob("sub-*/eeg/*_eeg.set")):
            m = EEG_STEM_RE.match(eeg_set.stem)
            if not m:
                continue
            key = (m.group("subject"), m.group("task").casefold())
            eeg_by_key[key] = eeg_set

        ecg_by_key: dict[tuple[str, str], Path] = {}
        for ecg_set in unique_sorted_paths(dataset_root.glob("sub-*/ecg/*_ecg.set")):
            m = ECG_STEM_RE.match(ecg_set.stem)
            if not m:
                continue
            key = (m.group("subject"), m.group("task").casefold())
            ecg_by_key[key] = ecg_set

        rows: list[CanonicalObservation] = []
        for key in sorted(set(eeg_by_key).intersection(ecg_by_key)):
            subject, task = key
            if not include_if_in_filter(subject, subject_filter):
                continue
            if not include_if_in_filter(task, task_filter):
                continue
            if not include_if_in_filter(task, condition_filter):
                continue

            eeg_path = eeg_by_key[key]
            ppg_path = ecg_by_key[key]
            rows.append(
                CanonicalObservation(
                    dataset_id=self.dataset_id,
                    observation_id=f"ds003838-{subject}-task-{task}",
                    subject_id=subject,
                    task_label=task,
                    condition_label=task,
                    eeg_path=eeg_path,
                    eeg_format="eeglab",
                    ppg_source="external_file",
                    ppg_path=ppg_path,
                    ppg_format="eeglab",
                    session_label="single",
                    modality="eeg_ecg_split",
                    timepoint="na",
                    state=task,
                    is_usable=True,
                    notes="PPG expected in paired ECG/physio recording.",
                )
            )

        return rows
