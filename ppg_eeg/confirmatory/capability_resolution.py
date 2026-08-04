"""Resolve declared YAML capabilities against observed evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..datasets import CanonicalObservation
from .config import ConfirmatoryDatasetConfig
from .dataset_contracts import DatasetCapabilities
from .reason_codes import (
    INSUFFICIENT_DURATION,
    MISSING_CONDITION_MAPPING,
    MISSING_EVENT_SERIES,
    MISSING_PAIRED_OBSERVATION,
    MISSING_REQUIRED_MODALITY,
    TOPOGRAPHY_NOT_SUPPORTED,
)

CAPABILITY_RESOLUTION_FILENAME = "capability_resolution.csv"


@dataclass(frozen=True)
class CapabilityResolutionRow:
    capability: str
    declared_value: str
    observed_evidence: str
    effective_value: str
    reason_code: str
    affected_stage: str

    def to_row(self) -> dict[str, str]:
        return asdict(self)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def resolve_effective_capabilities(
    dataset: ConfirmatoryDatasetConfig,
    observations: Sequence[CanonicalObservation],
    *,
    data_audit_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[DatasetCapabilities, list[CapabilityResolutionRow]]:
    """Derive effective capabilities from declared YAML + observed evidence.

    A YAML declaration alone never makes an analysis computable.
    """
    declared = dataset.capabilities
    obs = list(observations)
    rows: list[CapabilityResolutionRow] = []

    has_eeg_files = any(o.eeg_path.is_file() for o in obs) if obs else False
    has_hr_files = False
    if obs:
        for o in obs:
            if o.ppg_source == "embedded_eeg" and o.eeg_path.is_file():
                has_hr_files = True
                break
            if o.ppg_path is not None and o.ppg_path.is_file():
                has_hr_files = True
                break
    if data_audit_rows:
        has_eeg_files = has_eeg_files or any(
            str(r.get("eeg_exists", "")).casefold() in {"1", "true", "yes"}
            for r in data_audit_rows
        )
        has_hr_files = has_hr_files or any(
            str(r.get("cardiac_exists", "")).casefold() in {"1", "true", "yes"}
            for r in data_audit_rows
        )

    conditions = {o.condition_label.casefold() for o in obs}
    semantics = {
        k.casefold(): v for k, v in dataset.normalization.condition_semantics.items()
    }
    has_low = any(
        str(v.get("state", "")).casefold() == "state_low" for v in semantics.values()
    ) or any("rest" in c or "passive" in c or c == "step1" for c in conditions)
    has_high = any(
        str(v.get("state", "")).casefold() == "state_high" for v in semantics.values()
    ) or any(
        any(tok in c for tok in ("tetris", "memory", "wm", "go", "ig", "step2", "step3"))
        for c in conditions
    )
    has_pre = any(
        str(v.get("time", "")).casefold() == "time_pre" for v in semantics.values()
    ) or any("pre" in c for c in conditions)
    has_post = any(
        str(v.get("time", "")).casefold() == "time_post" for v in semantics.values()
    ) or any("post" in c for c in conditions)

    signal_types = {
        str(r.get("cardiac_signal_type", "")).casefold()
        for r in data_audit_rows
        if str(r.get("cardiac_signal_type", "")).strip()
    }
    observed_modality = declared.cardiac_modality
    if "ppg" in signal_types and "ecg" in signal_types:
        observed_modality = "both"
    elif "ppg" in signal_types:
        observed_modality = "ppg"
    elif "ecg" in signal_types:
        observed_modality = "ecg"
    elif not has_hr_files:
        observed_modality = "neither"

    overlaps = [
        float(r.get("raw_overlap_s") or r.get("overlap_duration_s") or 0)
        for r in data_audit_rows
        if str(r.get("raw_overlap_s") or r.get("overlap_duration_s") or "").strip()
    ]
    max_overlap = max(overlaps) if overlaps else None
    supports_d240_obs = max_overlap is not None and max_overlap >= 240
    supports_d180_obs = max_overlap is not None and max_overlap >= 180

    effective = DatasetCapabilities(
        has_eeg=bool(declared.has_eeg and has_eeg_files),
        has_hr=bool(declared.has_hr and has_hr_files),
        cardiac_modality=observed_modality if has_hr_files else "neither",
        has_ecg_r_peaks=bool(
            declared.has_ecg_r_peaks and observed_modality in {"ecg", "both"}
        ),
        has_ppg_peaks=bool(
            declared.has_ppg_peaks and observed_modality in {"ppg", "both"}
        ),
        has_low_high_task_pair=bool(
            declared.has_low_high_task_pair and has_low and has_high
        ),
        has_pre_post_pair=bool(declared.has_pre_post_pair and has_pre and has_post),
        has_behavior=bool(declared.has_behavior),
        supports_d180=bool(declared.supports_d180 and (supports_d180_obs if max_overlap is not None else False)),
        supports_d240=bool(declared.supports_d240 and (supports_d240_obs if max_overlap is not None else False)),
        supports_topography=bool(declared.supports_topography and has_eeg_files),
        supports_gamma=bool(declared.supports_gamma and has_eeg_files),
        sensitivity_only=bool(declared.sensitivity_only),
        has_artifact_controls=bool(declared.has_artifact_controls and has_hr_files),
    )

    def add(
        capability: str,
        declared_value: object,
        observed: str,
        effective_value: object,
        reason_code: str,
        stage: str,
    ) -> None:
        rows.append(
            CapabilityResolutionRow(
                capability=capability,
                declared_value=str(declared_value).casefold()
                if isinstance(declared_value, bool)
                else str(declared_value),
                observed_evidence=observed,
                effective_value=str(effective_value).casefold()
                if isinstance(effective_value, bool)
                else str(effective_value),
                reason_code=reason_code,
                affected_stage=stage,
            )
        )

    add(
        "has_eeg",
        declared.has_eeg,
        f"eeg_files_present={_bool_str(has_eeg_files)}; n_obs={len(obs)}",
        effective.has_eeg,
        "" if effective.has_eeg else MISSING_REQUIRED_MODALITY,
        "C0/C1a",
    )
    add(
        "has_hr",
        declared.has_hr,
        f"cardiac_files_present={_bool_str(has_hr_files)}; signal_types={sorted(signal_types)}",
        effective.has_hr,
        "" if effective.has_hr else MISSING_REQUIRED_MODALITY,
        "C0/C1b",
    )
    add(
        "cardiac_modality",
        declared.cardiac_modality,
        f"observed_signal_types={sorted(signal_types) or ['none']}",
        effective.cardiac_modality,
        "" if effective.has_hr else MISSING_REQUIRED_MODALITY,
        "C1b/PanelD",
    )
    add(
        "has_ecg_r_peaks",
        declared.has_ecg_r_peaks,
        f"modality={effective.cardiac_modality}",
        effective.has_ecg_r_peaks,
        "" if effective.has_ecg_r_peaks or not declared.has_ecg_r_peaks else MISSING_EVENT_SERIES,
        "PanelD",
    )
    add(
        "has_ppg_peaks",
        declared.has_ppg_peaks,
        f"modality={effective.cardiac_modality}",
        effective.has_ppg_peaks,
        "" if effective.has_ppg_peaks or not declared.has_ppg_peaks else MISSING_EVENT_SERIES,
        "PanelD",
    )
    add(
        "has_low_high_task_pair",
        declared.has_low_high_task_pair,
        f"conditions={sorted(conditions)}; semantics_keys={sorted(semantics)}",
        effective.has_low_high_task_pair,
        ""
        if effective.has_low_high_task_pair or not declared.has_low_high_task_pair
        else MISSING_CONDITION_MAPPING,
        "C5/Figure2-3",
    )
    add(
        "has_pre_post_pair",
        declared.has_pre_post_pair,
        f"has_pre={_bool_str(has_pre)}; has_post={_bool_str(has_post)}",
        effective.has_pre_post_pair,
        ""
        if effective.has_pre_post_pair or not declared.has_pre_post_pair
        else MISSING_PAIRED_OBSERVATION,
        "C5",
    )
    add(
        "supports_d240",
        declared.supports_d240,
        f"max_raw_overlap_s={max_overlap}",
        effective.supports_d240,
        ""
        if effective.supports_d240 or not declared.supports_d240
        else INSUFFICIENT_DURATION,
        "C0/C1c/Figure3C",
    )
    add(
        "supports_d180",
        declared.supports_d180,
        f"max_raw_overlap_s={max_overlap}",
        effective.supports_d180,
        ""
        if effective.supports_d180 or not declared.supports_d180
        else INSUFFICIENT_DURATION,
        "C0/C1c/Figure3C",
    )
    add(
        "supports_topography",
        declared.supports_topography,
        f"has_eeg={_bool_str(effective.has_eeg)}",
        effective.supports_topography,
        ""
        if effective.supports_topography or not declared.supports_topography
        else TOPOGRAPHY_NOT_SUPPORTED,
        "PanelF",
    )
    add(
        "supports_gamma",
        declared.supports_gamma,
        f"has_eeg={_bool_str(effective.has_eeg)}",
        effective.supports_gamma,
        "" if effective.supports_gamma or not declared.supports_gamma else "missing_required_band",
        "PanelF",
    )

    return effective, rows


def write_capability_resolution(
    rows: Iterable[CapabilityResolutionRow],
    output_dir: str | Path,
) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / CAPABILITY_RESOLUTION_FILENAME
    payload = [row.to_row() for row in rows]
    fieldnames = list(payload[0].keys()) if payload else [
        "capability",
        "declared_value",
        "observed_evidence",
        "effective_value",
        "reason_code",
        "affected_stage",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)
    return path
