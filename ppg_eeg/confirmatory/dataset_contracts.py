"""Dataset-normalized semantics and capability contracts for confirmatory runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


STATE_LOW = "state_low"
STATE_HIGH = "state_high"
STATE_OTHER = "state_other"

TIME_PRE = "time_pre"
TIME_POST = "time_post"
TIME_NA = "time_na"


@dataclass(frozen=True)
class ConditionSemantic:
    """Normalized semantics for one dataset-native condition label."""

    condition_label: str
    normalized_state: str
    normalized_time: str = TIME_NA
    session_alias: str = ""


@dataclass(frozen=True)
class DatasetCapabilities:
    """Capability gates used by shared pipeline stages."""

    has_eeg: bool
    has_hr: bool
    cardiac_modality: str
    has_ecg_r_peaks: bool
    has_ppg_peaks: bool
    has_low_high_task_pair: bool
    has_pre_post_pair: bool
    has_behavior: bool
    supports_d120: bool = True
    supports_d180: bool = True
    supports_d240: bool = True
    supports_topography: bool = True
    supports_gamma: bool = True
    sensitivity_only: bool = False
    has_artifact_controls: bool = True


@dataclass(frozen=True)
class DatasetContracts:
    """Resolved semantic and capability metadata for one dataset config."""

    dataset_id: str
    participant_id_from: str
    session_id_from: str
    cardiac_event_type: str
    condition_semantics: dict[str, ConditionSemantic]
    capabilities: DatasetCapabilities

    def semantic_for_condition(self, condition_label: str) -> ConditionSemantic | None:
        return self.condition_semantics.get(condition_label.casefold())


def resolve_condition_semantics(
    *,
    condition_semantics: Mapping[str, Mapping[str, str]] | None,
    low_demand_conditions: tuple[str, ...],
    high_demand_conditions: tuple[str, ...],
) -> dict[str, ConditionSemantic]:
    """Resolve condition semantics from YAML mapping or protocol defaults."""
    resolved: dict[str, ConditionSemantic] = {}
    if condition_semantics:
        for label, payload in condition_semantics.items():
            key = str(label).strip().casefold()
            state = str(payload.get("state", "")).strip().casefold()
            time_role = str(payload.get("time", TIME_NA)).strip().casefold() or TIME_NA
            if state not in {STATE_LOW, STATE_HIGH, STATE_OTHER}:
                raise ValueError(
                    f"condition_semantics[{label!r}].state must be one of "
                    f"{STATE_LOW!r}, {STATE_HIGH!r}, {STATE_OTHER!r}."
                )
            if time_role not in {TIME_PRE, TIME_POST, TIME_NA}:
                raise ValueError(
                    f"condition_semantics[{label!r}].time must be one of "
                    f"{TIME_PRE!r}, {TIME_POST!r}, {TIME_NA!r}."
                )
            resolved[key] = ConditionSemantic(
                condition_label=key,
                normalized_state=state,
                normalized_time=time_role,
                session_alias=str(payload.get("session", "")).strip().casefold(),
            )

    if resolved:
        return resolved

    for condition in low_demand_conditions:
        key = condition.casefold()
        resolved[key] = ConditionSemantic(
            condition_label=key,
            normalized_state=STATE_LOW,
            normalized_time=_infer_time_role(key),
            session_alias=_infer_session_alias(key),
        )
    for condition in high_demand_conditions:
        key = condition.casefold()
        resolved[key] = ConditionSemantic(
            condition_label=key,
            normalized_state=STATE_HIGH,
            normalized_time=_infer_time_role(key),
            session_alias=_infer_session_alias(key),
        )
    return resolved


def _infer_time_role(text: str) -> str:
    value = text.casefold()
    if "pre" in value:
        return TIME_PRE
    if "post" in value:
        return TIME_POST
    return TIME_NA


def _infer_session_alias(text: str) -> str:
    value = f"_{text.casefold()}_"
    if "_ph_" in value:
        return "ph"
    if "_ps_" in value:
        return "ps"
    return ""

