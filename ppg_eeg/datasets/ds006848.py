from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths


EEG_STEM_RE = re.compile(r"^(?P<subject>sub-\d+)_task-(?P<task>[a-zA-Z0-9]+)_eeg$")


class DS006848Adapter(DatasetAdapter):
    dataset_id = "ds006848"
    dataset_folder = "ds006848"

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

        rows: list[CanonicalObservation] = []
        for vhdr in unique_sorted_paths(dataset_root.glob("sub-*/eeg/*_eeg.vhdr")):
            m = EEG_STEM_RE.match(vhdr.stem)
            if not m:
                continue
            subject = m.group("subject")
            task = m.group("task").casefold()

            if not include_if_in_filter(subject, subject_filter):
                continue
            if not include_if_in_filter(task, task_filter):
                continue
            if not include_if_in_filter(task, condition_filter):
                continue

            eeg_bin = vhdr.with_suffix(".eeg")
            marker = vhdr.with_suffix(".vmrk")
            if not eeg_bin.exists() or not marker.exists():
                continue

            rows.append(
                CanonicalObservation(
                    dataset_id=self.dataset_id,
                    observation_id=f"ds006848-{subject}-task-{task}",
                    subject_id=subject,
                    task_label=task,
                    condition_label=task,
                    eeg_path=vhdr,
                    eeg_format="brainvision",
                    ppg_source="embedded_eeg",
                    ppg_path=None,
                    ppg_format=None,
                    session_label=None,
                    modality=None,
                    timepoint=None,
                    state=None,
                    is_usable=True,
                    notes="PPG expected in embedded MISC channel.",
                )
            )

        return rows
