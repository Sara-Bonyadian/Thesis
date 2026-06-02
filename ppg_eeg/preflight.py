from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig
from .datasets import build_observations
from .output_layout import dataset_output_dir, subject_output_dir

PREFLIGHT_DIRNAME = "preflight"
SUBJECT_MANIFEST_SUFFIX = ".preflight.json"
SUMMARY_FILENAME = "preflight_summary.csv"


@dataclass(frozen=True)
class SubjectPreflightRecord:
    dataset_id: str
    subject_id: str
    ready: bool
    n_observations: int
    missing_files: tuple[str, ...]
    output_dir: Path
    output_dir_writable: bool
    subject_manifest_path: Path
    observations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "ready": self.ready,
            "n_observations": self.n_observations,
            "missing_files": list(self.missing_files),
            "output_dir": str(self.output_dir),
            "output_dir_writable": self.output_dir_writable,
            "subject_manifest_path": str(self.subject_manifest_path),
            "observations": list(self.observations),
        }


@dataclass(frozen=True)
class PreflightReport:
    records: tuple[SubjectPreflightRecord, ...]
    summary_path: Path

    @property
    def ready(self) -> bool:
        return all(record.ready for record in self.records)

    @property
    def n_ready(self) -> int:
        return sum(1 for record in self.records if record.ready)

    @property
    def n_total(self) -> int:
        return len(self.records)


def _safe_subject_filename(subject_id: str) -> str:
    from .output_layout import safe_subject_dir_name

    return safe_subject_dir_name(subject_id)


def _required_paths_for_observation(observation: CanonicalObservation) -> list[Path]:
    paths = [observation.eeg_path]
    if observation.eeg_format.casefold() == "brainvision":
        paths.extend(
            [
                observation.eeg_path.with_suffix(".eeg"),
                observation.eeg_path.with_suffix(".vmrk"),
            ]
        )
    if observation.ppg_path is not None:
        paths.append(observation.ppg_path)
    return paths


def _check_readable_file(path: Path) -> str | None:
    if not path.exists():
        return f"missing: {path}"
    if not path.is_file():
        return f"not a file: {path}"
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        return f"unreadable: {path} ({exc})"
    return None


def _check_writable_directory(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"not writable: {path} ({exc})"
    return True, None


def _configured_subject_ids(cfg: PipelineConfig, dataset_id: str) -> list[str]:
    if cfg.subject_conditions:
        return [subject_id for subject_id in cfg.subject_conditions if str(subject_id).strip()]
    if cfg.subject_tasks:
        return [subject_id for subject_id in cfg.subject_tasks if str(subject_id).strip()]
    if cfg.subjects:
        return list(cfg.subjects)
    return []


def _observations_for_dataset(cfg: PipelineConfig, dataset_id: str) -> list[CanonicalObservation]:
    return build_observations(
        dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects or None,
        tasks=cfg.tasks or None,
        conditions=cfg.conditions or None,
        sessions=cfg.sessions or None,
        subject_tasks=cfg.subject_tasks or None,
        subject_conditions=cfg.subject_conditions or None,
    )


def run_preflight(cfg: PipelineConfig) -> PreflightReport:
    preflight_root = Path(cfg.paths.out_root) / PREFLIGHT_DIRNAME
    records: list[SubjectPreflightRecord] = []

    for dataset_id in cfg.dataset_ids:
        observations = [obs for obs in _observations_for_dataset(cfg, dataset_id) if obs.is_usable]
        obs_by_subject: dict[str, list[CanonicalObservation]] = {}
        for observation in observations:
            obs_by_subject.setdefault(observation.subject_id, []).append(observation)

        configured_subjects = _configured_subject_ids(cfg, dataset_id)
        subject_ids = configured_subjects or sorted(obs_by_subject)
        dataset_output_dir_path = dataset_output_dir(cfg, dataset_id)
        dataset_writable, dataset_write_error = _check_writable_directory(dataset_output_dir_path)

        for subject_id in subject_ids:
            subject_observations = obs_by_subject.get(subject_id, [])
            missing_files: list[str] = []
            observation_records: list[dict[str, Any]] = []

            for observation in subject_observations:
                observation_paths = _required_paths_for_observation(observation)
                observation_missing = [
                    issue
                    for path in observation_paths
                    if (issue := _check_readable_file(path)) is not None
                ]
                missing_files.extend(observation_missing)
                observation_records.append(
                    {
                        "observation_id": observation.observation_id,
                        "task_label": observation.task_label,
                        "condition_label": observation.condition_label,
                        "session_label": observation.session_label or "",
                        "eeg_path": str(observation.eeg_path),
                        "ppg_path": str(observation.ppg_path) if observation.ppg_path is not None else "",
                        "missing_files": observation_missing,
                    }
                )

            if not subject_observations:
                missing_files.append(f"no observations matched config filters for subject_id={subject_id!r}")

            subject_output_dir_path = subject_output_dir(cfg, dataset_id, subject_id)
            subject_writable, subject_write_error = _check_writable_directory(subject_output_dir_path)
            if subject_write_error:
                missing_files.append(subject_write_error)
            if dataset_write_error:
                missing_files.append(dataset_write_error)

            manifest_dir = preflight_root / dataset_id
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{_safe_subject_filename(subject_id)}{SUBJECT_MANIFEST_SUFFIX}"

            ready = (
                bool(subject_observations)
                and not missing_files
                and subject_writable
                and dataset_writable
            )
            record = SubjectPreflightRecord(
                dataset_id=dataset_id,
                subject_id=subject_id,
                ready=ready,
                n_observations=len(subject_observations),
                missing_files=tuple(dict.fromkeys(missing_files)),
                output_dir=subject_output_dir_path,
                output_dir_writable=subject_writable and dataset_writable,
                subject_manifest_path=manifest_path,
                observations=tuple(observation_records),
            )
            manifest_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            records.append(record)

    summary_rows = [
        {
            "dataset_id": record.dataset_id,
            "subject_id": record.subject_id,
            "ready": record.ready,
            "n_observations": record.n_observations,
            "output_dir": str(record.output_dir),
            "output_dir_writable": record.output_dir_writable,
            "missing_files": " | ".join(record.missing_files),
            "manifest_path": str(record.subject_manifest_path),
        }
        for record in records
    ]
    summary_path = preflight_root / SUMMARY_FILENAME
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    return PreflightReport(records=tuple(records), summary_path=summary_path)


def print_preflight_report(report: PreflightReport) -> None:
    print(f"preflight_ready={report.ready} ({report.n_ready}/{report.n_total} subjects)")
    print(f"preflight_summary={report.summary_path}")
    for record in report.records:
        status = "OK" if record.ready else "FAIL"
        print(
            f"[{record.dataset_id}] subject={record.subject_id} status={status} "
            f"observations={record.n_observations} manifest={record.subject_manifest_path}"
        )
        for issue in record.missing_files:
            print(f"  - {issue}")
