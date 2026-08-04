"""Configuration and capability validation for confirmatory dataset runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..datasets import CanonicalObservation
from ..temporal_coupling.data_audit import read_signal_file_info
from .config import ConfirmatoryDatasetConfig
from .reason_codes import (
    CONFIGURATION_VALIDATION_FAILED,
    INSUFFICIENT_DURATION,
    MISSING_CONDITION_MAPPING,
    MISSING_DATASET_ROOT,
    MISSING_EVENT_SERIES,
    MISSING_REQUIRED_MODALITY,
    TOPOGRAPHY_NOT_SUPPORTED,
)

VALIDATION_REPORT_FILENAME = "configuration_validation_report.txt"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"


def validate_dataset_configuration(
    dataset: ConfirmatoryDatasetConfig,
    observations: Iterable[CanonicalObservation],
) -> list[ValidationIssue]:
    """Validate dataset YAML semantics against discovered observations."""
    obs = list(observations)
    issues: list[ValidationIssue] = []
    raw_root = Path(dataset.paths.raw_root).expanduser()
    if not raw_root.exists():
        issues.append(
            ValidationIssue(
                code=MISSING_DATASET_ROOT,
                message=(
                    f"{dataset.dataset_id}: dataset root missing or unreadable: {raw_root}"
                ),
            )
        )
        return issues
    if not obs:
        issues.append(
            ValidationIssue(
                code=CONFIGURATION_VALIDATION_FAILED,
                message=f"{dataset.dataset_id}: no observations discovered after selection filters.",
            )
        )
        return issues

    conditions = {row.condition_label.casefold() for row in obs}
    tasks = {row.task_label.casefold() for row in obs}
    sessions = {(row.session_label or "single").casefold() for row in obs}
    configured_conditions = {item.casefold() for item in dataset.selection.conditions}
    configured_tasks = {item.casefold() for item in dataset.selection.tasks}
    configured_sessions = {item.casefold() for item in dataset.selection.sessions}

    missing_conditions = sorted(configured_conditions - conditions)
    missing_tasks = sorted(configured_tasks - tasks)
    missing_sessions = sorted(configured_sessions - sessions)
    if missing_conditions:
        issues.append(
            ValidationIssue(
                code=MISSING_CONDITION_MAPPING,
                message=(
                    f"{dataset.dataset_id}: configured conditions missing in data: "
                    + ", ".join(missing_conditions)
                ),
            )
        )
    if missing_tasks:
        issues.append(
            ValidationIssue(
                code=CONFIGURATION_VALIDATION_FAILED,
                message=(
                    f"{dataset.dataset_id}: configured tasks missing in data: "
                    + ", ".join(missing_tasks)
                ),
            )
        )
    if missing_sessions:
        issues.append(
            ValidationIssue(
                code=CONFIGURATION_VALIDATION_FAILED,
                message=(
                    f"{dataset.dataset_id}: configured sessions missing in data: "
                    + ", ".join(missing_sessions)
                ),
            )
        )

    capability = dataset.capabilities
    if not capability.has_hr:
        issues.append(
            ValidationIssue(
                code=MISSING_REQUIRED_MODALITY,
                message=f"{dataset.dataset_id}: has_hr=false; cardiac-dependent analyses become not_computable.",
                severity="warning",
            )
        )
    if capability.cardiac_modality in {"ecg", "both"} and not capability.has_ecg_r_peaks:
        issues.append(
            ValidationIssue(
                code=MISSING_EVENT_SERIES,
                message=f"{dataset.dataset_id}: ECG modality declared without ECG peak capability.",
                severity="warning",
            )
        )
    if capability.cardiac_modality in {"ppg", "both"} and not capability.has_ppg_peaks:
        issues.append(
            ValidationIssue(
                code=MISSING_EVENT_SERIES,
                message=f"{dataset.dataset_id}: PPG modality declared without PPG peak capability.",
                severity="warning",
            )
        )
    if not capability.supports_d240:
        issues.append(
            ValidationIssue(
                code=INSUFFICIENT_DURATION,
                message=f"{dataset.dataset_id}: supports_d240=false; D240 estimand exported as not_computable.",
                severity="warning",
            )
        )
    if not capability.supports_topography:
        issues.append(
            ValidationIssue(
                code=TOPOGRAPHY_NOT_SUPPORTED,
                message=f"{dataset.dataset_id}: supports_topography=false; Panel F exported as not_computable.",
                severity="warning",
            )
        )

    # Practical channel presence check from first readable EEG file.
    eeg_channels: set[str] = set()
    for row in obs:
        if not row.eeg_path.is_file():
            continue
        info = read_signal_file_info(row.eeg_path, data_format=row.eeg_format)
        if info is None:
            continue
        eeg_channels = {name.casefold() for name in info.ch_names}
        break
    if dataset.capabilities.supports_topography and not eeg_channels:
        issues.append(
            ValidationIssue(
                code=CONFIGURATION_VALIDATION_FAILED,
                message=f"{dataset.dataset_id}: unable to read EEG channels for montage validation.",
            )
        )
    return issues


def write_validation_report(
    *,
    dataset: ConfirmatoryDatasetConfig,
    issues: list[ValidationIssue],
    output_dir: str | Path,
) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / VALIDATION_REPORT_FILENAME
    lines = [
        f"dataset_id: {dataset.dataset_id}",
        f"role: {dataset.role}",
        f"config: {dataset.source_path}",
        f"validation_status: {'failed' if any(i.severity == 'error' for i in issues) else 'passed'}",
        "",
    ]
    if not issues:
        lines.append("No configuration or capability issues detected.")
    else:
        for issue in issues:
            lines.append(f"- [{issue.severity}] {issue.code}: {issue.message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

