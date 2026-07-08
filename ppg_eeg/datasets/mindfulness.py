from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import CanonicalObservation, DatasetAdapter, include_if_in_filter, normalized_set, unique_sorted_paths

_STEP_RE = re.compile(r"(?:p2_)?step(?P<step>[123])$", re.IGNORECASE)


def _mbd_folder(vhdr: Path) -> str | None:
    for parent in vhdr.parents:
        name = parent.name
        if name.startswith("mbd-"):
            return name
    return None


def _session_label(vhdr: Path) -> str:
    for part in vhdr.parts:
        low = part.casefold()
        if low in {"part2", "sub1_part2"} or low.endswith("_part2"):
            return "part2"
        if low == "part1":
            return "part1"
    if re.search(r"(?:^|_)p2_step[123]$", vhdr.stem, re.IGNORECASE):
        return "part2"
    return "part1"


def _task_label(stem: str) -> str | None:
    match = _STEP_RE.search(stem)
    if match is None:
        return None
    return f"step{match.group('step')}"


class MindfulnessAdapter(DatasetAdapter):
    dataset_id = "mindfulness"
    dataset_folder = "Mindfulness project"

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
        vhdr_files = unique_sorted_paths(dataset_root.glob("mbd-*/**/*.vhdr"))
        for vhdr in vhdr_files:
            mbd = _mbd_folder(vhdr)
            task_label = _task_label(vhdr.stem)
            if mbd is None or task_label is None:
                continue

            session_label = _session_label(vhdr)
            condition_label = task_label

            if not include_if_in_filter(mbd, subject_filter):
                continue
            if not include_if_in_filter(task_label, task_filter):
                continue
            if not include_if_in_filter(condition_label, condition_filter):
                continue
            if not include_if_in_filter(session_label, session_filter):
                continue

            eeg_bin = vhdr.with_suffix(".eeg")
            marker = vhdr.with_suffix(".vmrk")
            if not eeg_bin.is_file() or not marker.is_file():
                continue

            subject_instance = f"{mbd.casefold()}_{session_label}_{task_label}"
            obs_id = f"{self.dataset_id}-{mbd.casefold()}-{session_label}-task-{task_label}"
            rows.append(
                CanonicalObservation(
                    dataset_id=self.dataset_id,
                    observation_id=obs_id,
                    subject_id=subject_instance,
                    task_label=task_label,
                    condition_label=condition_label,
                    eeg_path=vhdr,
                    eeg_format="brainvision",
                    ppg_source="embedded_eeg",
                    ppg_path=None,
                    ppg_format=None,
                    session_label=session_label,
                    modality="eeg_ecg",
                    timepoint=session_label,
                    state=task_label,
                    is_usable=True,
                    notes="Photosensor PPG embedded in BrainVision EEG; mindfulness steps part1/part2.",
                )
            )

        return rows
