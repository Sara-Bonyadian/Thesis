"""Centralized dataset-role metadata and eligibility gates.

Two distinct questions must never be conflated:

``eligible_for_dataset_display``
    May this dataset's own estimates appear in a dataset-specific panel? True for
    every configured scientific role. Role never suppresses display.

``eligible_for_primary_meta_pooling``
    May this dataset×contrast contribute to the pooled random-effects primary
    meta-analysis (pooled diamond, prediction interval, ``k``, I², τ²)? Only
    primary-role paired state-dependent datasets on a prespecified
    ``PRIMARY_META_CONTRASTS`` pair.

Binary ``primary`` / ``sensitivity`` in ``master.yaml`` controls output-path
routing. Scientific identity is carried by ``analysis_family`` and the
eligibility flags below — a single primary/sensitivity bit is not sufficient
to answer panel, pooling, duration, or external-generalization questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

ROLE_PRIMARY = "primary"
ROLE_SENSITIVITY = "sensitivity"
ROLE_UNKNOWN = "unknown"

ROLE_TAGS = {ROLE_PRIMARY: "P", ROLE_SENSITIVITY: "S"}
UNKNOWN_ROLE_TAG = "?"

# Scientific analysis families (orthogonal to path-routing role).
FAMILY_PAIRED_STATE_DEPENDENT = "paired_state_dependent"
FAMILY_EXERCISE_STATE_MODERATION = "exercise_state_moderation"
FAMILY_EXTERNAL_GENERALIZATION = "external_generalization"
FAMILY_INTERNAL_ATTENTION = "internal_attention_generalization"
FAMILY_DURATION_SENSITIVITY = "duration_sensitivity"

# Why a dataset-specific paired panel is empty. Distinguishes a genuine estimand
# gap from role-based filtering, which must never empty a panel on its own.
NC_NO_PRESPECIFIED_CONTRAST = "no_prespecified_low_high_contrast"
NC_NO_ELIGIBLE_PAIRS = "no_eligible_paired_observations"
NC_NOT_APPLICABLE_ESTIMAND = "estimand_not_applicable_for_dataset"
NC_NOT_PRIMARY_ELIGIBLE = "not_primary_eligible"
NC_INSUFFICIENT_DURATION = "insufficient_duration"
NC_UNSUPPORTED_MODALITY = "unsupported_modality"
NC_UNMATCHED_STATE_STRUCTURE = "unmatched_state_structure"
NC_MISSING_UPSTREAM_INPUT = "missing_upstream_input"

# Panel / export status tokens (section 9).
PANEL_STATUS_NOT_COMPUTABLE = "not_computable"
PANEL_STATUS_NOT_APPLICABLE = "not_applicable"
PANEL_STATUS_NOT_PRIMARY_ELIGIBLE = "not_primary_eligible"
PANEL_STATUS_MISSING_UPSTREAM = "missing_upstream_input"
PANEL_STATUS_INSUFFICIENT_DURATION = "insufficient_duration"
PANEL_STATUS_INSUFFICIENT_COMMON_SUPPORT = "insufficient_common_support"
PANEL_STATUS_UNMATCHED_STATE = "unmatched_state_structure"
PANEL_STATUS_UNSUPPORTED_MODALITY = "unsupported_modality"

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "zero-lag-reanalysis-repo"


def _norm(value: object) -> str:
    return str(value or "").strip().casefold()


@dataclass(frozen=True)
class DatasetScientificProfile:
    """Locked manuscript identity and eligibility for one confirmatory dataset."""

    dataset_id: str
    path_role: str  # primary | sensitivity (output routing)
    analysis_family: str
    sensor_modality: str
    available_states: tuple[str, ...]
    state_order: tuple[str, ...]
    pairing_structure: str
    participant_key: str
    session_key: str
    repeated_measure_key: str
    eligible_durations_s: tuple[int, ...]
    primary_endpoint_eligible: bool
    primary_pool_eligible: bool
    sensitivity_eligible: bool
    external_generalization_eligible: bool
    preserve_condition_identity: bool
    supports_state_contrast: bool
    supports_graded_demand: bool
    supports_duration_analysis: bool
    supports_topography: bool
    display_eligible: bool = True
    # Runtime production status (orthogonal to scientific role).
    runtime_status: str = "ok"  # ok | blocked
    block_reason: str = ""


# Locked registry — single source of scientific truth for architecture audits.
DATASET_SCIENTIFIC_PROFILES: dict[str, DatasetScientificProfile] = {
    "ds003690": DatasetScientificProfile(
        dataset_id="ds003690",
        path_role=ROLE_PRIMARY,
        analysis_family=FAMILY_PAIRED_STATE_DEPENDENT,
        sensor_modality="ecg",
        available_states=("passive", "simplert", "gonogo"),
        state_order=("passive", "simplert", "gonogo"),
        pairing_structure="intersecting_subjects_within_session",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="run_id",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=True,
        primary_pool_eligible=True,
        sensitivity_eligible=True,
        external_generalization_eligible=False,
        preserve_condition_identity=True,
        supports_state_contrast=True,
        supports_graded_demand=True,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "ds003838": DatasetScientificProfile(
        dataset_id="ds003838",
        path_role=ROLE_PRIMARY,
        analysis_family=FAMILY_PAIRED_STATE_DEPENDENT,
        sensor_modality="ecg",
        available_states=("rest", "memory"),
        state_order=("rest", "memory"),
        pairing_structure="intersecting_subjects",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=True,
        primary_pool_eligible=True,
        sensitivity_eligible=True,
        external_generalization_eligible=False,
        preserve_condition_identity=True,
        supports_state_contrast=True,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "ds006848": DatasetScientificProfile(
        dataset_id="ds006848",
        path_role=ROLE_PRIMARY,
        analysis_family=FAMILY_PAIRED_STATE_DEPENDENT,
        sensor_modality="ecg",
        available_states=("rest", "verbalwm"),
        state_order=("rest", "verbalwm"),
        pairing_structure="intersecting_subjects",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=True,
        primary_pool_eligible=True,
        sensitivity_eligible=True,
        external_generalization_eligible=False,
        preserve_condition_identity=True,
        supports_state_contrast=True,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "hiit": DatasetScientificProfile(
        dataset_id="hiit",
        path_role=ROLE_SENSITIVITY,
        analysis_family=FAMILY_EXERCISE_STATE_MODERATION,
        sensor_modality="ppg",
        available_states=(
            "ph_pre_rest",
            "ph_pre_tetris",
            "ph_post_rest",
            "ph_post_tetris",
            "ps_pre_rest",
            "ps_pre_tetris",
            "ps_post_rest",
            "ps_post_tetris",
        ),
        state_order=(
            "ph_pre_rest",
            "ph_pre_tetris",
            "ph_post_rest",
            "ph_post_tetris",
            "ps_pre_rest",
            "ps_pre_tetris",
            "ps_post_rest",
            "ps_post_tetris",
        ),
        pairing_structure="pre_with_pre_post_with_post_within_session",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="time_pre_post",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=False,
        primary_pool_eligible=False,
        sensitivity_eligible=True,
        external_generalization_eligible=False,
        preserve_condition_identity=True,
        supports_state_contrast=True,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "ds004582": DatasetScientificProfile(
        dataset_id="ds004582",
        path_role=ROLE_SENSITIVITY,
        analysis_family=FAMILY_EXTERNAL_GENERALIZATION,
        sensor_modality="ecg",
        available_states=("ff",),
        state_order=("ff",),
        pairing_structure="single_condition_no_state_pair",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=False,
        primary_pool_eligible=False,
        sensitivity_eligible=True,
        external_generalization_eligible=True,
        preserve_condition_identity=True,
        supports_state_contrast=False,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "ds004587": DatasetScientificProfile(
        dataset_id="ds004587",
        path_role=ROLE_SENSITIVITY,
        analysis_family=FAMILY_EXTERNAL_GENERALIZATION,
        sensor_modality="ecg",
        available_states=("rest", "ig"),
        state_order=("rest", "ig"),
        pairing_structure="single_condition_no_state_pair",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=False,
        primary_pool_eligible=False,
        sensitivity_eligible=True,
        external_generalization_eligible=True,
        preserve_condition_identity=True,
        supports_state_contrast=False,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "mindfulness": DatasetScientificProfile(
        dataset_id="mindfulness",
        path_role=ROLE_SENSITIVITY,
        analysis_family=FAMILY_INTERNAL_ATTENTION,
        sensor_modality="ppg",
        available_states=("step1", "step2", "step3"),
        state_order=("step1", "step2", "step3"),
        pairing_structure="within_participant_ordered_steps",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="step",
        eligible_durations_s=(60, 120, 180, 240),
        primary_endpoint_eligible=False,
        primary_pool_eligible=False,
        sensitivity_eligible=True,
        external_generalization_eligible=True,
        preserve_condition_identity=True,
        supports_state_contrast=True,
        supports_graded_demand=True,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
    "ds003816": DatasetScientificProfile(
        dataset_id="ds003816",
        path_role=ROLE_SENSITIVITY,
        analysis_family=FAMILY_DURATION_SENSITIVITY,
        sensor_modality="ecg",
        available_states=(
            "preresting",
            "postresting",
            "lkmself",
            "lkmother",
            "visualizeself",
            "visualizeother",
        ),
        state_order=(
            "preresting",
            "postresting",
            "lkmself",
            "lkmother",
            "visualizeself",
            "visualizeother",
        ),
        pairing_structure="no_confirmatory_state_contrast",
        participant_key="participant_id",
        session_key="session_id",
        repeated_measure_key="run_id",
        # Locked: duration-matched D60 sensitivity only (SWPI), never D120–D240.
        eligible_durations_s=(60,),
        primary_endpoint_eligible=False,
        primary_pool_eligible=False,
        sensitivity_eligible=True,
        external_generalization_eligible=False,
        preserve_condition_identity=True,
        supports_state_contrast=False,
        supports_graded_demand=False,
        supports_duration_analysis=True,
        supports_topography=True,
    ),
}


def scientific_profile(dataset_id: str) -> DatasetScientificProfile | None:
    return DATASET_SCIENTIFIC_PROFILES.get(_norm(dataset_id))


def is_runtime_blocked(dataset_id: str) -> bool:
    profile = scientific_profile(dataset_id)
    return bool(profile and profile.runtime_status == "blocked")


def runtime_block_reason(dataset_id: str) -> str:
    profile = scientific_profile(dataset_id)
    return "" if profile is None else profile.block_reason


def analysis_family_for(dataset_id: str) -> str:
    profile = scientific_profile(dataset_id)
    return profile.analysis_family if profile is not None else ROLE_UNKNOWN


def eligible_durations_for(dataset_id: str) -> frozenset[int]:
    profile = scientific_profile(dataset_id)
    if profile is None:
        return frozenset({60, 120, 180, 240})
    return frozenset(profile.eligible_durations_s)


def family_display_label(analysis_family: str) -> str:
    """Human-readable role label for figure titles / legends."""
    labels = {
        FAMILY_PAIRED_STATE_DEPENDENT: "Primary paired state-dependent",
        FAMILY_EXERCISE_STATE_MODERATION: "Exercise-state moderation (sensitivity)",
        FAMILY_EXTERNAL_GENERALIZATION: "External generalization",
        FAMILY_INTERNAL_ATTENTION: "Internal-attention generalization",
        FAMILY_DURATION_SENSITIVITY: "Duration sensitivity",
    }
    return labels.get(_norm(analysis_family), _norm(analysis_family) or "unknown")


@lru_cache(maxsize=1)
def configured_dataset_roles() -> Mapping[str, str]:
    """``dataset_id -> path role`` from ``master.yaml`` (routing source of truth)."""
    try:
        from .config import load_master_config

        master = load_master_config(_CONFIG_ROOT / "master.yaml")
    except Exception:  # noqa: BLE001 - figures must still render without configs
        return {
            ds: profile.path_role
            for ds, profile in DATASET_SCIENTIFIC_PROFILES.items()
        }
    roles: dict[str, str] = {}
    for dataset_id in master.dataset_roles.primary:
        roles[_norm(dataset_id)] = ROLE_PRIMARY
    for dataset_id in master.dataset_roles.sensitivity:
        roles[_norm(dataset_id)] = ROLE_SENSITIVITY
    return roles


def resolve_dataset_role(
    dataset_id: str,
    *,
    protocol_rows: Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Path-routing role for ``dataset_id``, preferring C0 rows then ``master.yaml``."""
    key = _norm(dataset_id)
    if not key:
        return ROLE_UNKNOWN
    for row in protocol_rows or ():
        if _norm(row.get("dataset_id")) != key:
            continue
        role = _norm(row.get("dataset_role"))
        if role in {ROLE_PRIMARY, ROLE_SENSITIVITY}:
            return role
    configured = configured_dataset_roles().get(key)
    if configured is not None:
        return configured
    profile = scientific_profile(key)
    return profile.path_role if profile is not None else ROLE_UNKNOWN


def is_sensitivity_dataset(
    dataset_id: str,
    *,
    protocol_rows: Sequence[Mapping[str, object]] | None = None,
) -> bool:
    return (
        resolve_dataset_role(dataset_id, protocol_rows=protocol_rows)
        == ROLE_SENSITIVITY
    )


def is_primary_dataset(
    dataset_id: str,
    *,
    protocol_rows: Sequence[Mapping[str, object]] | None = None,
) -> bool:
    return resolve_dataset_role(dataset_id, protocol_rows=protocol_rows) == ROLE_PRIMARY


def is_external_generalization_dataset(dataset_id: str) -> bool:
    profile = scientific_profile(dataset_id)
    return bool(profile and profile.external_generalization_eligible)


def sensitivity_dataset_ids() -> frozenset[str]:
    return frozenset(
        ds for ds, role in configured_dataset_roles().items() if role == ROLE_SENSITIVITY
    )


def primary_dataset_ids() -> frozenset[str]:
    return frozenset(
        ds for ds, role in configured_dataset_roles().items() if role == ROLE_PRIMARY
    )


def role_tag(role: str) -> str:
    """Heatmap/axis tag: ``P`` primary, ``S`` sensitivity, ``?`` unresolved."""
    return ROLE_TAGS.get(_norm(role), UNKNOWN_ROLE_TAG)


def role_tagged_label(base_label: str, role: str) -> str:
    return f"{base_label} [{role_tag(role)}]"


def family_tagged_label(base_label: str, dataset_id: str) -> str:
    """Label that identifies scientific family, not only P/S path role."""
    profile = scientific_profile(dataset_id)
    if profile is None:
        return base_label
    return f"{base_label} [{family_display_label(profile.analysis_family)}]"


# --------------------------------------------------------------------------
# Eligibility gates
# --------------------------------------------------------------------------


def eligible_for_dataset_display(
    dataset_id: str,
    *,
    protocol_rows: Sequence[Mapping[str, object]] | None = None,
) -> bool:
    """May this dataset appear in its own dataset-specific panel?

    Role-independent by design. Only an unresolvable dataset is withheld.
    """
    profile = scientific_profile(dataset_id)
    if profile is not None:
        return bool(profile.display_eligible)
    return resolve_dataset_role(dataset_id, protocol_rows=protocol_rows) in {
        ROLE_PRIMARY,
        ROLE_SENSITIVITY,
    }


def eligible_for_primary_meta_pooling(dataset_id: str, contrast_id: str) -> bool:
    """May this dataset×contrast enter the pooled primary random-effects meta?"""
    from .inference import META_EXCLUDED_DATASETS, PRIMARY_META_CONTRASTS

    ds = _norm(dataset_id)
    profile = scientific_profile(ds)
    if profile is not None and not profile.primary_pool_eligible:
        return False
    if not is_primary_dataset(ds):
        return False
    if ds in META_EXCLUDED_DATASETS:
        return False
    return (ds, _norm(contrast_id)) in PRIMARY_META_CONTRASTS


def eligible_for_sensitivity_analysis(dataset_id: str) -> bool:
    profile = scientific_profile(dataset_id)
    if profile is not None:
        return bool(profile.sensitivity_eligible)
    return is_sensitivity_dataset(dataset_id)


def eligible_for_external_generalization(dataset_id: str) -> bool:
    return is_external_generalization_dataset(dataset_id)


def eligible_for_duration_analysis(dataset_id: str, duration_s: int) -> bool:
    return int(duration_s) in eligible_durations_for(dataset_id)


def panel_reason_for_empty_paired(
    dataset_id: str,
) -> tuple[str, str]:
    """Return ``(panel_status, reason_code)`` when a paired panel cannot render."""
    profile = scientific_profile(dataset_id)
    if profile is None:
        return PANEL_STATUS_NOT_COMPUTABLE, NC_MISSING_UPSTREAM_INPUT
    if not profile.supports_state_contrast or not has_prespecified_contrast(dataset_id):
        if profile.external_generalization_eligible:
            return PANEL_STATUS_NOT_APPLICABLE, NC_NOT_APPLICABLE_ESTIMAND
        return PANEL_STATUS_NOT_COMPUTABLE, NC_NO_PRESPECIFIED_CONTRAST
    return PANEL_STATUS_NOT_COMPUTABLE, NC_NO_ELIGIBLE_PAIRS


def eligibility_export_fields(dataset_id: str) -> dict[str, object]:
    """Metadata fields required on source-data exports."""
    profile = scientific_profile(dataset_id)
    role = resolve_dataset_role(dataset_id)
    if profile is None:
        return {
            "dataset_id": _norm(dataset_id),
            "dataset_role": role,
            "analysis_family": ROLE_UNKNOWN,
            "can_display_dataset_result": eligible_for_dataset_display(dataset_id),
            "eligible_for_primary_pool": False,
            "eligible_for_sensitivity_analysis": role == ROLE_SENSITIVITY,
            "eligible_for_external_generalization": False,
            "eligible_durations_s": "60;120;180;240",
            "supports_state_contrast": False,
        }
    return {
        "dataset_id": profile.dataset_id,
        "dataset_role": role,
        "analysis_family": profile.analysis_family,
        "can_display_dataset_result": profile.display_eligible,
        "eligible_for_primary_pool": profile.primary_pool_eligible,
        "eligible_for_sensitivity_analysis": profile.sensitivity_eligible,
        "eligible_for_external_generalization": profile.external_generalization_eligible,
        "eligible_durations_s": ";".join(str(d) for d in profile.eligible_durations_s),
        "supports_state_contrast": profile.supports_state_contrast,
        "supports_graded_demand": profile.supports_graded_demand,
        "sensor_modality": profile.sensor_modality,
        "pairing_structure": profile.pairing_structure,
    }


# --------------------------------------------------------------------------
# Protocol-derived contrast / condition sets (replace hard-coded HIIT sets)
# --------------------------------------------------------------------------


def _spec(dataset_id: str):
    from .protocol_audit import protocol_spec

    try:
        return protocol_spec(dataset_id)
    except KeyError:
        return None


def dataset_contrast_ids(dataset_id: str) -> frozenset[str]:
    """Prespecified low-demand vs high-demand contrast ids for this dataset."""
    spec = _spec(dataset_id)
    if spec is None:
        return frozenset()
    return frozenset(_norm(c.contrast_id) for c in spec.contrasts)


def dataset_low_demand_conditions(dataset_id: str) -> frozenset[str]:
    spec = _spec(dataset_id)
    if spec is None:
        return frozenset()
    return frozenset(_norm(c) for c in spec.low_demand_conditions)


def has_prespecified_contrast(dataset_id: str) -> bool:
    """False means paired ΔZLPI panels are genuinely not computable / N/A."""
    profile = scientific_profile(dataset_id)
    if profile is not None and not profile.supports_state_contrast:
        return False
    return bool(dataset_contrast_ids(dataset_id))


def session_alias_for_condition(dataset_id: str, condition: str) -> str:
    """Protocol session token for a condition (``ph``/``ps``/``part1``/...)."""
    label = _norm(condition)
    if not label:
        return ""
    try:
        from .config import load_dataset_config, load_master_config

        master = load_master_config(_CONFIG_ROOT / "master.yaml")
        config = load_dataset_config(
            _CONFIG_ROOT / "datasets" / f"{_norm(dataset_id)}.yaml", master=master
        )
        semantic = config.contracts.semantic_for_condition(label)
        if semantic is not None and semantic.session_alias:
            return semantic.session_alias
    except Exception:  # noqa: BLE001 - fall through to protocol inference
        pass
    from .protocol_audit import condition_semantics_for

    _state, _time, session = condition_semantics_for(dataset_id, label)
    return "" if session in {"", "single"} else session


def session_unit_key(row: Mapping[str, object], *, dataset_id: str = "") -> str:
    """Analysis unit for a sensitivity display row (session subject when nested)."""
    subject = _norm(row.get("subject_id"))
    if subject:
        return subject
    participant = _norm(row.get("participant_id"))
    session = _norm(row.get("session_id")) or "single"
    if session not in {"", "single"}:
        return f"{participant}_{session}" if participant else session
    dataset = _norm(dataset_id) or _norm(row.get("dataset_id"))
    condition = _norm(row.get("condition") or row.get("contrast_id"))
    alias = session_alias_for_condition(dataset, condition) if condition else ""
    if not alias and condition:
        alias = session_alias_for_condition(dataset, condition.split("__", 1)[0])
    if alias:
        return f"{participant}_{alias}" if participant else alias
    return participant


__all__ = [
    "DATASET_SCIENTIFIC_PROFILES",
    "DatasetScientificProfile",
    "FAMILY_DURATION_SENSITIVITY",
    "FAMILY_EXERCISE_STATE_MODERATION",
    "FAMILY_EXTERNAL_GENERALIZATION",
    "FAMILY_INTERNAL_ATTENTION",
    "FAMILY_PAIRED_STATE_DEPENDENT",
    "NC_INSUFFICIENT_DURATION",
    "NC_MISSING_UPSTREAM_INPUT",
    "NC_NOT_APPLICABLE_ESTIMAND",
    "NC_NOT_PRIMARY_ELIGIBLE",
    "NC_NO_ELIGIBLE_PAIRS",
    "NC_NO_PRESPECIFIED_CONTRAST",
    "NC_UNMATCHED_STATE_STRUCTURE",
    "NC_UNSUPPORTED_MODALITY",
    "PANEL_STATUS_INSUFFICIENT_COMMON_SUPPORT",
    "PANEL_STATUS_INSUFFICIENT_DURATION",
    "PANEL_STATUS_MISSING_UPSTREAM",
    "PANEL_STATUS_NOT_APPLICABLE",
    "PANEL_STATUS_NOT_COMPUTABLE",
    "PANEL_STATUS_NOT_PRIMARY_ELIGIBLE",
    "PANEL_STATUS_UNMATCHED_STATE",
    "PANEL_STATUS_UNSUPPORTED_MODALITY",
    "ROLE_PRIMARY",
    "ROLE_SENSITIVITY",
    "ROLE_TAGS",
    "ROLE_UNKNOWN",
    "UNKNOWN_ROLE_TAG",
    "analysis_family_for",
    "configured_dataset_roles",
    "dataset_contrast_ids",
    "dataset_low_demand_conditions",
    "eligibility_export_fields",
    "eligible_durations_for",
    "eligible_for_dataset_display",
    "eligible_for_duration_analysis",
    "eligible_for_external_generalization",
    "eligible_for_primary_meta_pooling",
    "eligible_for_sensitivity_analysis",
    "family_display_label",
    "family_tagged_label",
    "has_prespecified_contrast",
    "is_external_generalization_dataset",
    "is_primary_dataset",
    "is_runtime_blocked",
    "is_sensitivity_dataset",
    "panel_reason_for_empty_paired",
    "primary_dataset_ids",
    "resolve_dataset_role",
    "role_tag",
    "role_tagged_label",
    "runtime_block_reason",
    "scientific_profile",
    "sensitivity_dataset_ids",
    "session_alias_for_condition",
    "session_unit_key",
]
