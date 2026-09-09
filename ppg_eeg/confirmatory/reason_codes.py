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
OPTION_C_PEAK_NOT_APPLICABLE = "option_c_peak_not_applicable"
EXCLUDED_BY_MANUSCRIPT_DESIGN = "excluded_by_manuscript_design"

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
    OPTION_C_PEAK_NOT_APPLICABLE,
    EXCLUDED_BY_MANUSCRIPT_DESIGN,
)

# Legacy / stage-local exclusion strings → taxonomy codes.
_EXCLUSION_ALIASES: dict[str, str] = {
    "missing_dataset_root": MISSING_DATASET_ROOT,
    "input_discovery_failed": INPUT_DISCOVERY_FAILED,
    "data_not_supplied": INPUT_DISCOVERY_FAILED,
    "no_curve_rows": INPUT_DISCOVERY_FAILED,
    "missing_c1a_or_c1b_features": INPUT_DISCOVERY_FAILED,
    "missing_mismatch_pool": INPUT_DISCOVERY_FAILED,
    "derangement_impossible": INPUT_DISCOVERY_FAILED,
    "mismatch_surrogate_count": CONFIGURATION_VALIDATION_FAILED,
    "missing_required_modality": MISSING_REQUIRED_MODALITY,
    "missing_eeg": MISSING_REQUIRED_MODALITY,
    "missing_cardiac_data": MISSING_REQUIRED_MODALITY,
    "missing_event_series": MISSING_EVENT_SERIES,
    "unsupported_control_for_modality": UNSUPPORTED_CONTROL_FOR_MODALITY,
    "insufficient_duration": INSUFFICIENT_DURATION,
    "insufficient_raw_duration": INSUFFICIENT_DURATION,
    "insufficient_beat_span": INSUFFICIENT_DURATION,
    "insufficient_clean_support": INSUFFICIENT_DURATION,
    "insufficient_length": INSUFFICIENT_DURATION,
    "insufficient_blocks": INSUFFICIENT_DURATION,
    "d240_segment_not_available": INSUFFICIENT_DURATION,
    "excluded_by_manuscript_design": EXCLUDED_BY_MANUSCRIPT_DESIGN,
    "insufficient_lag_support": INSUFFICIENT_LAG_SUPPORT,
    "incomplete_lag_grid": INSUFFICIENT_LAG_SUPPORT,
    "incomplete_endpoint_windows": INSUFFICIENT_LAG_SUPPORT,
    "insufficient_common_support_for_lag_grid": INSUFFICIENT_LAG_SUPPORT,
    "insufficient_common_support": INSUFFICIENT_COMMON_SUPPORT,
    "no_common_support": INSUFFICIENT_COMMON_SUPPORT,
    "no_aligned_samples": INSUFFICIENT_COMMON_SUPPORT,
    "nonconstant_overlap": INSUFFICIENT_COMMON_SUPPORT,
    "nonfinite_required_correlations": INSUFFICIENT_COMMON_SUPPORT,
    "insufficient_overlap_after_masking": INSUFFICIENT_COMMON_SUPPORT,
    "insufficient_template_support": INSUFFICIENT_COMMON_SUPPORT,
    "missing_condition_mapping": MISSING_CONDITION_MAPPING,
    "protocol_mismatch": MISSING_CONDITION_MAPPING,
    "missing_paired_observation": MISSING_PAIRED_OBSERVATION,
    "missing_paired_state": MISSING_PAIRED_OBSERVATION,
    "unresolved_pairing": MISSING_PAIRED_OBSERVATION,
    "both_endpoint_ineligible": MISSING_PAIRED_OBSERVATION,
    "low_endpoint_ineligible": MISSING_PAIRED_OBSERVATION,
    "effort_endpoint_ineligible": MISSING_PAIRED_OBSERVATION,
    "missing_low_demand": MISSING_PAIRED_OBSERVATION,
    "missing_effort": MISSING_PAIRED_OBSERVATION,
    "no_observations": MISSING_PAIRED_OBSERVATION,
    "topography_not_supported_by_dataset": TOPOGRAPHY_NOT_SUPPORTED,
    "missing_sensor_locations": MISSING_SENSOR_LOCATIONS,
    "insufficient_common_montage": INSUFFICIENT_COMMON_MONTAGE,
    "missing_required_band": MISSING_REQUIRED_BAND,
    "no_channel_level_endpoint_values": NO_CHANNEL_LEVEL_ENDPOINT_VALUES,
    "missing_rest_or_task_channel_zlpi": NO_CHANNEL_LEVEL_ENDPOINT_VALUES,
    "artifact_control_not_available": ARTIFACT_CONTROL_NOT_AVAILABLE,
    "control_unavailable": ARTIFACT_CONTROL_NOT_AVAILABLE,
    "skipped_not_requested": ARTIFACT_CONTROL_NOT_AVAILABLE,
    "skipped_not_applicable": ARTIFACT_CONTROL_NOT_AVAILABLE,
    "skipped_unavailable": ARTIFACT_CONTROL_NOT_AVAILABLE,
    "undefined_null_variance": UNDEFINED_NULL_VARIANCE,
    "no_finite_surrogates": UNDEFINED_NULL_VARIANCE,
    "configuration_validation_failed": CONFIGURATION_VALIDATION_FAILED,
    "standard_zlpi_not_applicable": STANDARD_ZLPI_NOT_APPLICABLE,
    "endpoint_contract_non_zlpi": ENDPOINT_CONTRACT_NON_ZLPI,
    "option_c_peak_not_applicable": OPTION_C_PEAK_NOT_APPLICABLE,
    "option_c_peak_requires_d180_d240": OPTION_C_PEAK_NOT_APPLICABLE,
    "insufficient_option_c_flank_support": INSUFFICIENT_LAG_SUPPORT,
}


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


STRUCTURED_NC_FIELDS: tuple[str, ...] = (
    "status",
    "reason_code",
    "reason",
    "required_evidence",
    "observed_evidence",
    "dataset_id",
    "stage",
    "specification_id",
    "participant_id",
    "session_id",
    "observation_id",
)


def map_exclusion_to_reason_code(exclusion: object) -> str:
    """Map a stage-local exclusion / status string onto the locked taxonomy."""
    text = str(exclusion or "").strip()
    if not text:
        return ""
    key = text.casefold()
    if key in _EXCLUSION_ALIASES:
        return _EXCLUSION_ALIASES[key]
    # Prefer exact taxonomy tokens embedded in compound notes.
    for code in REASON_CODES:
        if code in key:
            return code
    # Heuristic fragments for free-text control / topography notes.
    if "montage" in key and ("common" in key or "empty" in key or "insufficient" in key):
        return INSUFFICIENT_COMMON_MONTAGE
    if "sensor" in key and "loc" in key:
        return MISSING_SENSOR_LOCATIONS
    if "topograph" in key:
        return TOPOGRAPHY_NOT_SUPPORTED
    if "channel-level" in key or "channel level" in key:
        return NO_CHANNEL_LEVEL_ENDPOINT_VALUES
    if "null" in key and ("variance" in key or "std" in key or "sd" in key):
        return UNDEFINED_NULL_VARIANCE
    if "artifact" in key and ("unavailable" in key or "not available" in key):
        return ARTIFACT_CONTROL_NOT_AVAILABLE
    if "modality" in key and ("unsupported" in key or "ppg" in key or "ecg" in key):
        return UNSUPPORTED_CONTROL_FOR_MODALITY
    if "event" in key and ("missing" in key or "absent" in key):
        return MISSING_EVENT_SERIES
    if "dataset" in key and "root" in key:
        return MISSING_DATASET_ROOT
    return text


def decision_status_from_flags(
    *,
    eligible: bool | None = None,
    exclusion_reason: object = "",
    raw_status: object = "",
) -> str:
    """Normalize producer eligibility flags into computed/excluded/not_computable."""
    text = str(raw_status or "").strip().casefold()
    if text in {"eligible", "computed", "ok", "estimable", "computable", "primary", "sensitivity_only"}:
        return "computed"
    if text in {"excluded", "ineligible"}:
        return "excluded"
    if text in {
        "not_available",
        "not_identifiable",
        "not_supplied",
        "not_computable",
        "control_unavailable",
        "insufficient_data",
        "unavailable",
    }:
        return "not_computable"
    if text:
        # Keep specialized null-QC statuses as not_computable unless ok-like.
        return "not_computable"
    if eligible is True and not str(exclusion_reason or "").strip():
        return "computed"
    if eligible is False:
        return "excluded"
    if str(exclusion_reason or "").strip():
        return "not_computable"
    return "computed"


def attach_structured_reason(
    row: Mapping[str, object],
    *,
    stage: str,
    status: str = "",
    reason_code: str = "",
    reason: str = "",
    required_evidence: str = "",
    observed_evidence: str = "",
    specification_id: str = "",
    eligible: bool | None = None,
) -> dict[str, object]:
    """Attach a full structured NC envelope at the producer decision site.

    Existing explicit fields always win. Identity fields are nullable only when
    genuinely absent from the source row.
    """
    out = dict(row)
    exclusion = (
        reason_code
        or out.get("reason_code")
        or out.get("endpoint_reason_code")
        or out.get("exclusion_reason")
        or out.get("exclusion_code")
        or out.get("not_computable_reason")
        or out.get("computability_reason")
        or ""
    )
    mapped_code = map_exclusion_to_reason_code(exclusion) if exclusion else ""
    resolved_status = status or decision_status_from_flags(
        eligible=eligible if eligible is not None else out.get("eligible"),
        exclusion_reason=exclusion,
        raw_status=(
            out.get("status")
            or out.get("computability_status")
            or out.get("endpoint_status")
            or out.get("eligibility_status")
            or out.get("eligibility")
            or out.get("pairing_status")
            or ""
        ),
    )
    if resolved_status == "computed":
        mapped_code = ""
    elif not mapped_code:
        mapped_code = INPUT_DISCOVERY_FAILED

    human_reason = str(reason or out.get("reason") or "").strip()
    if resolved_status != "computed" and not human_reason:
        human_reason = str(
            out.get("notes")
            or out.get("exclusion_reason")
            or out.get("not_computable_reason")
            or mapped_code.replace("_", " ")
        ).strip()

    participant = str(
        out.get("participant_id")
        or out.get("biological_participant_id")
        or ""
    ).strip()
    session = str(out.get("session_id") or out.get("session_label") or "").strip()
    observation = str(out.get("observation_id") or "").strip()
    dataset = str(out.get("dataset_id") or "").strip()

    out.update(
        {
            "status": resolved_status,
            "reason_code": mapped_code if resolved_status != "computed" else "",
            "reason": human_reason if resolved_status != "computed" else "",
            "required_evidence": str(
                out.get("required_evidence") or required_evidence or ""
            ),
            "observed_evidence": str(
                out.get("observed_evidence") or observed_evidence or ""
            ),
            "dataset_id": dataset,
            "stage": str(out.get("stage") or stage),
            "specification_id": str(
                out.get("specification_id") or specification_id or ""
            ),
            "participant_id": participant,
            "session_id": session,
            "observation_id": observation,
        }
    )
    # Preserve producer-local eligibility / exclusion columns when present.
    if "exclusion_reason" in row and not str(out.get("exclusion_reason") or "").strip():
        out["exclusion_reason"] = mapped_code
    return out


def with_structured_nc_fields(
    row: Mapping[str, object],
    *,
    stage: str,
    specification_id: str = "",
    required_evidence: str = "",
    observed_evidence: str = "",
) -> dict[str, object]:
    """Return an export row with a complete structured NC envelope.

    Existing explicit fields always win. Computed rows receive an empty reason
    envelope; unavailable rows retain missing numeric values and get a
    deterministic human-readable reason rather than a plotted surrogate zero.
    """
    return attach_structured_reason(
        row,
        stage=stage,
        specification_id=specification_id,
        required_evidence=required_evidence,
        observed_evidence=observed_evidence,
    )
