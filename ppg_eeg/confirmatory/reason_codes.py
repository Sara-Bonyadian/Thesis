"""Structured not-computable / configuration reason codes for confirmatory runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Precise taxonomy — prefer these over generic "insufficient_data".
MISSING_DATASET_ROOT = "missing_dataset_root"
INPUT_DISCOVERY_FAILED = "input_discovery_failed"
MISSING_REQUIRED_MODALITY = "missing_required_modality"
MISSING_EVENT_SERIES = "missing_event_series"
UNSUPPORTED_CONTROL_FOR_MODALITY = "unsupported_control_for_modality"
INSUFFICIENT_DURATION = "insufficient_duration"
INSUFFICIENT_LAG_SUPPORT = "insufficient_lag_support"
INSUFFICIENT_COMMON_SUPPORT = "insufficient_common_support"
MISSING_CONDITION_MAPPING = "missing_condition_mapping"
MISSING_PAIRED_OBSERVATION = "missing_paired_observation"
TOPOGRAPHY_NOT_SUPPORTED = "topography_not_supported_by_dataset"
MISSING_SENSOR_LOCATIONS = "missing_sensor_locations"
INSUFFICIENT_COMMON_MONTAGE = "insufficient_common_montage"
MISSING_REQUIRED_BAND = "missing_required_band"
NO_CHANNEL_LEVEL_ENDPOINT_VALUES = "no_channel_level_endpoint_values"
ARTIFACT_CONTROL_NOT_AVAILABLE = "artifact_control_not_available"
UNDEFINED_NULL_VARIANCE = "undefined_null_variance"
CONFIGURATION_VALIDATION_FAILED = "configuration_validation_failed"
STANDARD_ZLPI_NOT_APPLICABLE = "standard_zlpi_not_applicable"
ENDPOINT_CONTRACT_NON_ZLPI = "endpoint_contract_non_zlpi"

REASON_CODES: tuple[str, ...] = (
    MISSING_DATASET_ROOT,
    INPUT_DISCOVERY_FAILED,
    MISSING_REQUIRED_MODALITY,
    MISSING_EVENT_SERIES,
    UNSUPPORTED_CONTROL_FOR_MODALITY,
    INSUFFICIENT_DURATION,
    INSUFFICIENT_LAG_SUPPORT,
    INSUFFICIENT_COMMON_SUPPORT,
    MISSING_CONDITION_MAPPING,
    MISSING_PAIRED_OBSERVATION,
    TOPOGRAPHY_NOT_SUPPORTED,
    MISSING_SENSOR_LOCATIONS,
    INSUFFICIENT_COMMON_MONTAGE,
    MISSING_REQUIRED_BAND,
    NO_CHANNEL_LEVEL_ENDPOINT_VALUES,
    ARTIFACT_CONTROL_NOT_AVAILABLE,
    UNDEFINED_NULL_VARIANCE,
    CONFIGURATION_VALIDATION_FAILED,
    STANDARD_ZLPI_NOT_APPLICABLE,
    ENDPOINT_CONTRACT_NON_ZLPI,
)


@dataclass(frozen=True)
class StructuredReason:
    status: str
    reason_code: str
    reason: str
    required_evidence: str
    observed_evidence: str

    def to_row(self) -> dict[str, str]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "required_evidence": self.required_evidence,
            "observed_evidence": self.observed_evidence,
        }


def structured_reason(
    *,
    status: str,
    reason_code: str,
    reason: str,
    required_evidence: str = "",
    observed_evidence: str = "",
) -> StructuredReason:
    return StructuredReason(
        status=status,
        reason_code=reason_code,
        reason=reason,
        required_evidence=required_evidence,
        observed_evidence=observed_evidence,
    )


def merge_reason_fields(row: Mapping[str, object], reason: StructuredReason) -> dict[str, object]:
    out = dict(row)
    out.update(reason.to_row())
    return out
