"""Protocol metadata, pairing, and duration eligibility for confirmatory C0.

C0 also runs an observation-level raw-data audit (see ``data_audit.py``).
This module still does not derive cardiac beats; ``clean_beat_span_s`` remains
unknown until a later stage supplies it.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..datasets import (
    CanonicalObservation,
    canonical_observation_identity,
    build_observations,
    validate_observation_identities,
)
from .config import (
    EXPECTED_DURATIONS_S,
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
)
from .dataset_contracts import (
    STATE_OTHER,
    TIME_NA,
    DatasetContracts,
    resolve_condition_semantics,
)
from .duration_contracts import (
    contract_for_duration,
    standard_zlpi_is_computable,
)
from .reason_codes import (
    ENDPOINT_CONTRACT_NON_ZLPI,
    EXCLUDED_BY_MANUSCRIPT_DESIGN,
    INSUFFICIENT_DURATION,
    INSUFFICIENT_LAG_SUPPORT,
    INPUT_DISCOVERY_FAILED,
    MISSING_CONDITION_MAPPING,
    MISSING_PAIRED_OBSERVATION,
    MISSING_REQUIRED_MODALITY,
    STANDARD_ZLPI_NOT_APPLICABLE,
    STRUCTURED_NC_FIELDS,
    map_exclusion_to_reason_code,
)

PROTOCOL_AUDIT_FILENAME = "protocol_audit.csv"
PAIRED_SUBJECT_SETS_FILENAME = "paired_subject_sets.json"
ELIGIBILITY_BY_DURATION_FILENAME = "eligibility_by_duration.csv"
ELIGIBILITY_QC_SUMMARY_FILENAME = "eligibility_qc_summary.csv"

ELIGIBILITY_STATUSES = ("eligible", "ineligible", "not_computable", "not_supplied")
EXCLUSION_CODES = (
    "missing_paired_state",
    "insufficient_raw_duration",
    "insufficient_beat_span",
    "missing_eeg",
    "missing_cardiac_data",
    "unresolved_pairing",
    "protocol_mismatch",
    "data_not_supplied",
    "excluded_by_manuscript_design",
)


@dataclass(frozen=True)
class ContrastSpec:
    contrast_id: str
    low_demand_condition: str
    cognitive_effort_condition: str
    pair_within: tuple[str, ...] = ("participant_id", "session_id")


@dataclass(frozen=True)
class ProtocolSpec:
    dataset_id: str
    low_demand_conditions: tuple[str, ...]
    cognitive_effort_conditions: tuple[str, ...]
    contrasts: tuple[ContrastSpec, ...]
    cardiac_modality: str
    cardiac_source: str
    eye_state: str
    posture: str
    task_timing: str
    nuisance_signals: tuple[str, ...]
    run_pairing_policy: str
    unresolved_assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class EligibilityMetadata:
    """Metadata needed to evaluate one observation at fixed durations.

    ``None`` means unknown/not supplied.  Unknown source fields produce
    ``not_supplied`` or ``not_computable`` outcomes, never ``ineligible``.
    """

    dataset_id: str
    observation_id: str
    participant_id: str
    condition: str
    contrast_id: str = ""
    source_data_supplied: bool = True
    eeg_exists: bool | None = None
    cardiac_exists: bool | None = None
    raw_overlap_s: float | None = None
    clean_beat_span_s: float | None = None
    requires_paired_state: bool = False
    paired_state_available: bool | None = None
    pairing_resolved: bool | None = True
    protocol_match: bool | None = True
    notes: str = ""


@dataclass(frozen=True)
class EligibilityDecision:
    dataset_id: str
    observation_id: str
    participant_id: str
    condition: str
    contrast_id: str
    duration_s: int
    status: str
    exclusion_code: str
    raw_overlap_s: float | None
    clean_beat_span_s: float | None
    source_data_supplied: bool
    eeg_exists: bool | None
    cardiac_exists: bool | None
    requires_paired_state: bool
    paired_state_available: bool | None
    pairing_resolved: bool | None
    protocol_match: bool | None
    notes: str
    # Explicit duration / endpoint contract fields (disambiguate "eligible").
    endpoint_name: str = ""
    endpoint_alias: str = ""
    is_standard_zlpi: bool = False
    has_required_duration: bool | None = None
    has_required_lag_support: bool | None = None
    endpoint_computable: bool = False
    endpoint_status: str = "not_computable"
    endpoint_reason_code: str = ""
    standard_zlpi_computable: bool = False
    standard_zlpi_reason_code: str = ""

    def to_row(self) -> dict[str, object]:
        required = (
            f"duration_s>={self.duration_s}; "
            f"eeg+cardiac present; pairing resolved when required"
        )
        observed = (
            f"raw_overlap_s={self.raw_overlap_s}; "
            f"clean_beat_span_s={self.clean_beat_span_s}; "
            f"eeg_exists={self.eeg_exists}; cardiac_exists={self.cardiac_exists}; "
            f"pairing_resolved={self.pairing_resolved}; "
            f"paired_state_available={self.paired_state_available}"
        )
        reason_code = map_exclusion_to_reason_code(
            self.endpoint_reason_code or self.exclusion_code
        )
        # Preserve protocol eligibility status vocabulary (eligible/ineligible/
        # not_computable/not_supplied). Structured reason fields ride alongside.
        return {
            "dataset_id": self.dataset_id,
            "observation_id": self.observation_id,
            "participant_id": self.participant_id,
            "session_id": "",
            "condition": self.condition,
            "contrast_id": self.contrast_id,
            "duration_s": self.duration_s,
            "duration_role": (
                "primary" if self.duration_s == EXPECTED_DURATIONS_S[0] else "sensitivity"
            ),
            "status": self.status,
            "exclusion_code": self.exclusion_code,
            "raw_overlap_s": self.raw_overlap_s,
            "clean_beat_span_s": self.clean_beat_span_s,
            "source_data_supplied": self.source_data_supplied,
            "eeg_exists": self.eeg_exists,
            "cardiac_exists": self.cardiac_exists,
            "requires_paired_state": self.requires_paired_state,
            "paired_state_available": self.paired_state_available,
            "pairing_resolved": self.pairing_resolved,
            "protocol_match": self.protocol_match,
            "notes": self.notes,
            "endpoint_name": self.endpoint_name,
            "endpoint_alias": self.endpoint_alias,
            "is_standard_zlpi": self.is_standard_zlpi,
            "has_required_duration": self.has_required_duration,
            "has_required_lag_support": self.has_required_lag_support,
            "endpoint_computable": self.endpoint_computable,
            "endpoint_status": self.endpoint_status,
            "endpoint_reason_code": self.endpoint_reason_code,
            "standard_zlpi_computable": self.standard_zlpi_computable,
            "standard_zlpi_reason_code": self.standard_zlpi_reason_code,
            "reason_code": reason_code if self.status != "eligible" or not self.endpoint_computable else "",
            "reason": (
                self.notes
                or (reason_code.replace("_", " ") if reason_code else "")
            ),
            "required_evidence": required if self.status != "eligible" or not self.endpoint_computable else "",
            "observed_evidence": observed if self.status != "eligible" or not self.endpoint_computable else "",
            "stage": "C1",
            "specification_id": f"{self.endpoint_name or 'endpoint'}_d{self.duration_s}",
        }


PROTOCOL_SPECS: dict[str, ProtocolSpec] = {
    "ds003838": ProtocolSpec(
        dataset_id="ds003838",
        low_demand_conditions=("rest",),
        cognitive_effort_conditions=("memory",),
        contrasts=(ContrastSpec("rest__memory", "rest", "memory"),),
        cardiac_modality="ECG",
        cardiac_source="separate ECG EEGLAB recording",
        eye_state="unknown",
        posture="unknown",
        task_timing="rest and memory are separate recordings",
        nuisance_signals=("ECG",),
        run_pairing_policy="single recording per participant and condition",
        unresolved_assumptions=(
            "Confirm eye state and posture from source documentation.",
        ),
    ),
    "ds006848": ProtocolSpec(
        dataset_id="ds006848",
        low_demand_conditions=("rest",),
        cognitive_effort_conditions=("verbalwm",),
        contrasts=(ContrastSpec("rest__verbalwm", "rest", "verbalwm"),),
        cardiac_modality="ECG;PPG",
        cardiac_source=(
            "ECG and PPG embedded in BrainVision EEG; select one primary "
            "modality for instantaneous HR (ECG preferred when QC passes)"
        ),
        eye_state="unknown",
        posture="unknown",
        task_timing="rest and verbal working-memory are separate recordings",
        nuisance_signals=("embedded ECG", "embedded PPG/MISC"),
        run_pairing_policy="single recording per participant and condition",
        unresolved_assumptions=(
            "Confirm eye state, posture, and exact ECG/PPG channel labels.",
        ),
    ),
    "ds003690": ProtocolSpec(
        dataset_id="ds003690",
        low_demand_conditions=("passive",),
        cognitive_effort_conditions=("simplert", "gonogo"),
        contrasts=(
            ContrastSpec("passive__simplert", "passive", "simplert"),
            ContrastSpec("passive__gonogo", "passive", "gonogo"),
        ),
        cardiac_modality="ECG",
        cardiac_source="EKG channel embedded in EEGLAB EEG",
        eye_state="unknown",
        posture="unknown",
        task_timing="passive, simple reaction-time, and go/no-go runs",
        nuisance_signals=("EKG", "VEO", "HEO"),
        run_pairing_policy="pair by participant/session; retain run and aggregate duplicate runs later",
        unresolved_assumptions=(
            "Confirm eye-state instruction and posture.",
            "Endpoint aggregation across duplicate runs is deferred beyond M1b.",
        ),
    ),
    "ds004587": ProtocolSpec(
        dataset_id="ds004587",
        low_demand_conditions=("rest", "ig"),
        cognitive_effort_conditions=(),
        contrasts=(),
        cardiac_modality="ECG",
        cardiac_source="ECGBIT in external BIDS physio (OXIBIT also available)",
        eye_state="rest eyes closed; IG task eye state not explicitly recorded",
        posture="unknown",
        task_timing=(
            "rest and IG retained as separate single-state recordings for "
            "external lag-zero replication; no confirmatory paired contrast"
        ),
        nuisance_signals=("OXIBIT", "device HR"),
        run_pairing_policy=(
            "single-condition external generalization; do not force rest–IG "
            "into a primary paired state contrast"
        ),
        unresolved_assumptions=(
            "Confirm posture and IG eye-state description.",
            "Treat ECGBIT as primary cardiac source for HR; OXIBIT is inventory-only.",
        ),
    ),
    "ds004582": ProtocolSpec(
        dataset_id="ds004582",
        low_demand_conditions=("ff",),
        cognitive_effort_conditions=(),
        contrasts=(),
        cardiac_modality="ECG",
        cardiac_source="ECGBIT in external BIDS physio (OXIBIT also available)",
        eye_state="unknown",
        posture="unknown",
        task_timing="single FF condition",
        nuisance_signals=("OXIBIT", "device HR", "respiration"),
        run_pairing_policy="single-state dataset; no state pairing",
        unresolved_assumptions=(
            "FF is treated as a low-demand single-state replication; verify task interpretation.",
            "Confirm eye state and posture.",
        ),
    ),
    "ds003816": ProtocolSpec(
        dataset_id="ds003816",
        low_demand_conditions=("preresting", "postresting"),
        cognitive_effort_conditions=(
            "lkmself",
            "lkmother",
            "visualizeself",
            "visualizeother",
        ),
        contrasts=(),
        cardiac_modality="ECG",
        cardiac_source="ECG embedded in BrainVision EEG",
        eye_state="unknown",
        posture="unknown",
        task_timing="resting and short meditation/visualization task recordings",
        nuisance_signals=("embedded ECG",),
        run_pairing_policy="preserve participant/session/run; no confirmatory contrast prespecified",
        unresolved_assumptions=(
            "No cognitive-effort contrast is prespecified for confirmatory inference.",
            "Confirm task semantics, eye state, posture, and any additional task label.",
        ),
    ),
    "hiit": ProtocolSpec(
        dataset_id="hiit",
        low_demand_conditions=(
            "ph_pre_rest",
            "ph_post_rest",
            "ps_pre_rest",
            "ps_post_rest",
        ),
        cognitive_effort_conditions=(
            "ph_pre_tetris",
            "ph_post_tetris",
            "ps_pre_tetris",
            "ps_post_tetris",
        ),
        contrasts=(
            ContrastSpec("ph_pre_rest__tetris", "ph_pre_rest", "ph_pre_tetris"),
            ContrastSpec("ph_post_rest__tetris", "ph_post_rest", "ph_post_tetris"),
            ContrastSpec("ps_pre_rest__tetris", "ps_pre_rest", "ps_pre_tetris"),
            ContrastSpec("ps_post_rest__tetris", "ps_post_rest", "ps_post_tetris"),
        ),
        cardiac_modality="PPG;ECG",
        cardiac_source=(
            "embedded PPG is primary cardiac source for instantaneous HR; "
            "inventory co-recorded ECG in C0 when channel QC passes"
        ),
        eye_state="rest=eyes_closed; tetris=eyes_open",
        posture="rest=seated; tetris=seated",
        task_timing=(
            "PRE and POST rest/Tetris recordings in separate PH and PS "
            "randomized-crossover modality sessions"
        ),
        nuisance_signals=("photosensor", "ECG", "respiration", "protocol modality PH/PS"),
        run_pairing_policy=(
            "PH and PS remain separate crossover sessions; pair Rest–Tetris "
            "within participant, PH/PS session, and PRE/POST timepoint"
        ),
        unresolved_assumptions=(),
    ),
    "mindfulness": ProtocolSpec(
        dataset_id="mindfulness",
        low_demand_conditions=("step1",),
        cognitive_effort_conditions=("step2", "step3"),
        contrasts=(
            ContrastSpec("step1__step2", "step1", "step2"),
            ContrastSpec("step1__step3", "step1", "step3"),
        ),
        cardiac_modality="PPG;ECG",
        cardiac_source=(
            "embedded PPG is primary cardiac source for instantaneous HR; "
            "inventory co-recorded ECG in C0 when complete"
        ),
        eye_state="unknown",
        posture="unknown",
        task_timing="ordered steps 1-3 within part1/part2 sessions",
        nuisance_signals=("photosensor",),
        run_pairing_policy="strip task suffix and pair by underlying participant/session",
        unresolved_assumptions=(
            "Steps are treated as graded states; confirm cognitive-effort interpretation.",
            "Confirm eye state, posture, and whether part1/part2 are directly comparable.",
        ),
    ),
}

_RUN_RE = re.compile(r"(?:^|[-_])run[-_]?([a-zA-Z0-9]+)(?:$|[-_])", re.IGNORECASE)


def protocol_spec(dataset_id: str) -> ProtocolSpec:
    key = dataset_id.strip().casefold()
    try:
        return PROTOCOL_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(PROTOCOL_SPECS))
        raise KeyError(f"Unknown confirmatory dataset {dataset_id!r}. Known: {known}") from exc


def condition_semantics_for(dataset_id: str, condition: str) -> tuple[str, str, str]:
    """Return normalized (state, time, session) roles for condition labels."""
    label = condition.strip().casefold()
    try:
        spec = protocol_spec(dataset_id)
    except KeyError:
        spec = None
    if spec is not None and label in spec.low_demand_conditions:
        state = "state_low"
    elif spec is not None and label in spec.cognitive_effort_conditions:
        state = "state_high"
    elif "rest" in label or "passive" in label or label == "step1":
        state = "state_low"
    elif "tetris" in label or "memory" in label or "wm" in label or "go" in label:
        state = "state_high"
    else:
        state = "state_other"
    time_role = "time_pre" if "pre" in label else ("time_post" if "post" in label else "time_na")
    session_role = "ph" if "_ph_" in f"_{label}_" else ("ps" if "_ps_" in f"_{label}_" else "single")
    return state, time_role, session_role


def _protocol_participant_id(
    observation: CanonicalObservation,
    participant_id_from: str,
) -> str:
    """Resolve participant identity using a YAML-declared source strategy."""
    source = participant_id_from.strip().casefold() or "auto"
    if source in {"participant_id", "column"} and observation.participant_id:
        return observation.participant_id.casefold()
    if source in {"subject_id", "subject"}:
        return observation.subject_id.casefold()
    if source in {"observation_id", "bids", "auto"}:
        obs_id = observation.observation_id.casefold()
        dataset_prefix = f"{observation.dataset_id.casefold()}-"
        remainder = obs_id.removeprefix(dataset_prefix)
        if "-ses-" in remainder:
            return remainder.split("-ses-", 1)[0]
        if "-task-" in remainder:
            return remainder.split("-task-", 1)[0]
        token = remainder.split("-", 1)[0]
        if token:
            return token
    return (observation.participant_id or observation.subject_id).casefold()


def _protocol_session_id(
    observation: CanonicalObservation,
    session_id_from: str,
) -> str:
    """Resolve session identity using a YAML-declared source strategy."""
    source = session_id_from.strip().casefold() or "auto"
    if source in {"session_id", "column"} and observation.session_id:
        return observation.session_id.casefold()
    if source in {"session_label", "bids", "auto"} and observation.session_label:
        return observation.session_label.casefold()
    if source == "subject_suffix":
        subject = observation.subject_id.casefold()
        return subject.rsplit("_", 1)[-1] if "_" in subject else "single"
    if source == "observation_id":
        match = _SES_RE.search(observation.observation_id)
        if match:
            return match.group(1).casefold()
    return "single"


def _participant_id(observation: CanonicalObservation) -> str:
    """Biological / protocol participant identity (not session-qualified subject).

    Prefer an already-normalized ``participant_id``. Otherwise apply the generic
    observation-id / BIDS parsing strategy before falling back to ``subject_id``.
    Session-qualified subjects such as ``01_ph`` remain available via
    ``subject_id`` for Panel-B analysis units.
    """
    if observation.participant_id:
        return str(observation.participant_id).strip().casefold()
    return _protocol_participant_id(observation, "auto")


def _run_id(observation: CanonicalObservation) -> str:
    return canonical_observation_identity(observation).run_id


def _session_id(observation: CanonicalObservation) -> str:
    if observation.session_id:
        return str(observation.session_id).strip().casefold()
    if observation.session_label:
        return str(observation.session_label).strip().casefold()
    # Generic session-qualified subject suffix (e.g. ``01_ph`` → ``ph``).
    subject = str(observation.subject_id or "").strip().casefold()
    if "_" in subject:
        suffix = subject.rsplit("_", 1)[-1]
        if suffix and not suffix.isdigit():
            return suffix
    return canonical_observation_identity(observation).session_id


def _dataset_contracts(
    config: ConfirmatoryDatasetConfig,
    spec: ProtocolSpec,
) -> DatasetContracts:
    semantics = resolve_condition_semantics(
        condition_semantics=config.normalization.condition_semantics,
        low_demand_conditions=spec.low_demand_conditions,
        high_demand_conditions=spec.cognitive_effort_conditions,
    )
    return DatasetContracts(
        dataset_id=config.dataset_id,
        participant_id_from=config.protocol.participant_id_from,
        session_id_from=config.protocol.session_id_from,
        cardiac_event_type=config.protocol.cardiac_event_type,
        condition_semantics=semantics,
        capabilities=config.capabilities,
    )


def _normalize_observation_with_contracts(
    observation: CanonicalObservation,
    *,
    contracts: DatasetContracts,
) -> CanonicalObservation:
    participant = _protocol_participant_id(
        observation, contracts.participant_id_from
    )
    session = _protocol_session_id(observation, contracts.session_id_from)
    semantic = contracts.semantic_for_condition(observation.condition_label.casefold())
    normalized_state = semantic.normalized_state if semantic else STATE_OTHER
    normalized_time = semantic.normalized_time if semantic else TIME_NA
    if semantic and semantic.session_alias:
        session = semantic.session_alias
    identity = canonical_observation_identity(
        observation,
        participant_id=participant,
        session_id=session,
        condition_id=observation.condition_label,
    )
    return replace(
        observation,
        participant_id=identity.participant_id,
        session_id=identity.session_id,
        run_id=identity.run_id,
        condition_id=identity.condition_id,
        pairing_id=identity.pairing_id,
        normalized_state=normalized_state,
        normalized_time=normalized_time,
        cardiac_modality=contracts.capabilities.cardiac_modality,
        cardiac_event_type=contracts.cardiac_event_type,
    )


def _pair_key(
    observation: CanonicalObservation,
    contrast: ContrastSpec,
) -> tuple[str, ...]:
    identity = observation.identity
    values = {
        "participant_id": identity.participant_id,
        "session_id": identity.session_id,
        "run_id": identity.run_id,
    }
    return tuple(values[field] for field in contrast.pair_within)


def _key_payload(contrast: ContrastSpec, key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(contrast.pair_within, key, strict=True))


def _observations_for_condition(
    observations: Sequence[CanonicalObservation],
    condition: str,
) -> list[CanonicalObservation]:
    target = condition.casefold()
    return [obs for obs in observations if obs.condition_label.casefold() == target]


def build_paired_subject_sets(
    observations_by_dataset: Mapping[str, Sequence[CanonicalObservation]],
) -> dict[str, object]:
    """Return deterministic paired participant/session sets for each contrast."""
    datasets: dict[str, object] = {}
    for dataset_id in sorted(PROTOCOL_SPECS):
        spec = PROTOCOL_SPECS[dataset_id]
        observations = tuple(observations_by_dataset.get(dataset_id, ()))
        discovery_status = (
            "not_supplied"
            if dataset_id not in observations_by_dataset
            else ("observed" if observations else "no_observations")
        )
        contrasts: dict[str, object] = {}
        for contrast in spec.contrasts:
            low_rows = _observations_for_condition(
                observations, contrast.low_demand_condition
            )
            effort_rows = _observations_for_condition(
                observations, contrast.cognitive_effort_condition
            )
            low_by_key: dict[tuple[str, ...], list[str]] = {}
            effort_by_key: dict[tuple[str, ...], list[str]] = {}
            for obs in low_rows:
                low_by_key.setdefault(_pair_key(obs, contrast), []).append(
                    obs.observation_id
                )
            for obs in effort_rows:
                effort_by_key.setdefault(_pair_key(obs, contrast), []).append(
                    obs.observation_id
                )

            low_keys = set(low_by_key)
            effort_keys = set(effort_by_key)
            paired_keys = sorted(low_keys & effort_keys)
            contrasts[contrast.contrast_id] = {
                "low_demand_condition": contrast.low_demand_condition,
                "cognitive_effort_condition": contrast.cognitive_effort_condition,
                "pair_within": list(contrast.pair_within),
                "n_low_demand": len(low_keys),
                "n_cognitive_effort": len(effort_keys),
                "n_paired": len(paired_keys),
                "paired_keys": [_key_payload(contrast, key) for key in paired_keys],
                "low_demand_only_keys": [
                    _key_payload(contrast, key)
                    for key in sorted(low_keys - effort_keys)
                ],
                "cognitive_effort_only_keys": [
                    _key_payload(contrast, key)
                    for key in sorted(effort_keys - low_keys)
                ],
                "observations_by_pair": [
                    {
                        **_key_payload(contrast, key),
                        "low_demand_observation_ids": sorted(low_by_key[key]),
                        "cognitive_effort_observation_ids": sorted(
                            effort_by_key[key]
                        ),
                    }
                    for key in paired_keys
                ],
            }

        datasets[dataset_id] = {
            "has_prespecified_pairing": bool(spec.contrasts),
            "n_observations": len(observations),
            "observation_discovery_status": discovery_status,
            "contrasts": contrasts,
            "unresolved_assumptions": list(spec.unresolved_assumptions),
        }
    return {"schema_version": 1, "datasets": datasets}


def _audit_rows(
    observations_by_dataset: Mapping[str, Sequence[CanonicalObservation]],
    paired_sets: Mapping[str, object],
    dataset_roles: Mapping[str, str],
) -> list[dict[str, object]]:
    datasets_payload = paired_sets["datasets"]
    assert isinstance(datasets_payload, Mapping)
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(PROTOCOL_SPECS):
        spec = PROTOCOL_SPECS[dataset_id]
        observations = tuple(observations_by_dataset.get(dataset_id, ()))
        conditions = spec.low_demand_conditions + spec.cognitive_effort_conditions
        contrast_by_condition: dict[str, list[str]] = {}
        for contrast in spec.contrasts:
            contrast_by_condition.setdefault(
                contrast.low_demand_condition, []
            ).append(contrast.contrast_id)
            contrast_by_condition.setdefault(
                contrast.cognitive_effort_condition, []
            ).append(contrast.contrast_id)

        dataset_payload = datasets_payload[dataset_id]
        assert isinstance(dataset_payload, Mapping)
        contrasts_payload = dataset_payload["contrasts"]
        assert isinstance(contrasts_payload, Mapping)
        for condition in conditions:
            condition_rows = _observations_for_condition(observations, condition)
            participants = sorted({_participant_id(obs) for obs in condition_rows})
            contrast_ids = contrast_by_condition.get(condition, [])
            pair_counts = [
                int(contrasts_payload[contrast_id]["n_paired"])
                for contrast_id in contrast_ids
            ]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": dataset_roles.get(dataset_id, ""),
                    "condition": condition,
                    "demand_class": (
                        "low_demand"
                        if condition in spec.low_demand_conditions
                        else "cognitive_effort"
                    ),
                    "contrast_ids": ";".join(contrast_ids),
                    "subject_pairing_key": "participant_id",
                    "session_pairing_key": "session_id",
                    "run_pairing_key": spec.run_pairing_policy,
                    "cardiac_modality": spec.cardiac_modality,
                    "cardiac_source": spec.cardiac_source,
                    "eye_state": spec.eye_state,
                    "posture": spec.posture,
                    "task_timing": spec.task_timing,
                    "nuisance_signals": ";".join(spec.nuisance_signals),
                    "observation_discovery_status": dataset_payload[
                        "observation_discovery_status"
                    ],
                    "n_observations": len(condition_rows),
                    "n_participants": len(participants),
                    "n_paired_participants": min(pair_counts) if pair_counts else 0,
                    "participant_ids": ";".join(participants),
                    "unresolved_assumptions": " | ".join(
                        spec.unresolved_assumptions
                    ),
                }
            )
    return rows


def _known_nonnegative(value: float | None, *, field: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number or None.")
    return number


def evaluate_duration_eligibility(
    metadata: EligibilityMetadata,
    duration_s: int,
) -> EligibilityDecision:
    """Evaluate one fixed duration without interpreting absent data as failure.

    ``status`` reflects extraction-window / protocol eligibility for the nested
    duration segment. Endpoint computability is reported separately via
    ``endpoint_*`` and ``standard_zlpi_*`` fields so D60/D120 are never mistaken
    for computable standard-ZLPI endpoints.
    """
    if duration_s not in EXPECTED_DURATIONS_S:
        raise ValueError(
            f"duration_s must be one of {EXPECTED_DURATIONS_S}, got {duration_s}."
        )
    contract = contract_for_duration(duration_s)
    raw_overlap_s = _known_nonnegative(
        metadata.raw_overlap_s, field="raw_overlap_s"
    )
    clean_beat_span_s = _known_nonnegative(
        metadata.clean_beat_span_s, field="clean_beat_span_s"
    )

    status = "eligible"
    exclusion_code = ""
    # Locked scientific registry: e.g. ds003816 may enter D60 only.
    from .dataset_roles import eligible_for_duration_analysis

    if metadata.dataset_id and not eligible_for_duration_analysis(
        metadata.dataset_id, duration_s
    ):
        status, exclusion_code = "ineligible", "excluded_by_manuscript_design"
    elif not metadata.source_data_supplied:
        status, exclusion_code = "not_supplied", "data_not_supplied"
    elif metadata.protocol_match is False:
        status, exclusion_code = "ineligible", "protocol_mismatch"
    elif metadata.protocol_match is None:
        status, exclusion_code = "not_computable", "data_not_supplied"
    elif metadata.pairing_resolved is False:
        status, exclusion_code = "ineligible", "unresolved_pairing"
    elif metadata.pairing_resolved is None:
        status, exclusion_code = "not_computable", "unresolved_pairing"
    elif (
        metadata.requires_paired_state
        and metadata.paired_state_available is False
    ):
        status, exclusion_code = "ineligible", "missing_paired_state"
    elif (
        metadata.requires_paired_state
        and metadata.paired_state_available is None
    ):
        status, exclusion_code = "not_computable", "unresolved_pairing"
    elif metadata.eeg_exists is False:
        status, exclusion_code = "not_computable", "missing_eeg"
    elif metadata.cardiac_exists is False:
        status, exclusion_code = "not_computable", "missing_cardiac_data"
    elif metadata.eeg_exists is None or metadata.cardiac_exists is None:
        status, exclusion_code = "not_computable", "data_not_supplied"
    elif raw_overlap_s is not None and raw_overlap_s < duration_s:
        status, exclusion_code = "ineligible", "insufficient_raw_duration"
    elif clean_beat_span_s is not None and clean_beat_span_s < duration_s:
        status, exclusion_code = "ineligible", "insufficient_beat_span"
    elif raw_overlap_s is None:
        # Raw overlap is required for duration gates; clean beat span is optional
        # at C0 (beat detection is deferred).
        status, exclusion_code = "not_computable", "data_not_supplied"

    notes = metadata.notes
    if (
        status == "eligible"
        and clean_beat_span_s is None
        and "clean_beat_span_not_computed" not in notes
    ):
        suffix = "clean_beat_span_not_computed"
        notes = f"{notes} {suffix}".strip() if notes else suffix

    has_required_duration: bool | None
    if raw_overlap_s is None:
        has_required_duration = None
    else:
        has_required_duration = raw_overlap_s >= float(duration_s)

    # Lag-support for THIS duration's own endpoint contract (ZLPI/MWPI/SWPI).
    required_overlap_for_contract = float(
        contract.expected_constant_overlap_if_fully_finite
    )
    has_required_lag_support: bool | None
    if raw_overlap_s is None:
        has_required_lag_support = None
    else:
        # Need enough contiguous samples to extract duration_s AND leave the
        # contract's common-support overlap under its lag envelope.
        has_required_lag_support = (
            has_required_duration is True
            and required_overlap_for_contract > 0
            and raw_overlap_s >= float(duration_s)
        )

    endpoint_computable = status == "eligible" and has_required_lag_support is True
    exclusion_to_reason = {
        "missing_paired_state": MISSING_PAIRED_OBSERVATION,
        "unresolved_pairing": MISSING_PAIRED_OBSERVATION,
        "insufficient_raw_duration": INSUFFICIENT_DURATION,
        "insufficient_beat_span": INSUFFICIENT_DURATION,
        "missing_eeg": MISSING_REQUIRED_MODALITY,
        "missing_cardiac_data": MISSING_REQUIRED_MODALITY,
        "protocol_mismatch": MISSING_CONDITION_MAPPING,
        "data_not_supplied": INPUT_DISCOVERY_FAILED,
        "excluded_by_manuscript_design": EXCLUDED_BY_MANUSCRIPT_DESIGN,
    }
    if status == "not_supplied":
        endpoint_status = "not_supplied"
        endpoint_reason_code = exclusion_to_reason.get(
            exclusion_code, exclusion_code or INPUT_DISCOVERY_FAILED
        )
    elif status == "ineligible":
        endpoint_status = "ineligible"
        endpoint_reason_code = exclusion_to_reason.get(
            exclusion_code, exclusion_code or INSUFFICIENT_DURATION
        )
    elif not endpoint_computable:
        endpoint_status = "not_computable"
        if has_required_duration is False:
            endpoint_reason_code = INSUFFICIENT_DURATION
        elif has_required_lag_support is False:
            endpoint_reason_code = INSUFFICIENT_LAG_SUPPORT
        else:
            endpoint_reason_code = exclusion_to_reason.get(
                exclusion_code, exclusion_code or INPUT_DISCOVERY_FAILED
            )
    else:
        endpoint_status = "computable"
        endpoint_reason_code = ""

    # Standard ZLPI is never applicable at D60/D120 under the locked contract.
    if not contract.is_standard_zlpi:
        standard_zlpi_computable = False
        standard_zlpi_reason_code = ENDPOINT_CONTRACT_NON_ZLPI
    elif not standard_zlpi_is_computable(duration_s):
        standard_zlpi_computable = False
        standard_zlpi_reason_code = STANDARD_ZLPI_NOT_APPLICABLE
    elif endpoint_computable:
        standard_zlpi_computable = True
        standard_zlpi_reason_code = ""
    else:
        standard_zlpi_computable = False
        standard_zlpi_reason_code = endpoint_reason_code or INSUFFICIENT_LAG_SUPPORT

    return EligibilityDecision(
        dataset_id=metadata.dataset_id.casefold(),
        observation_id=metadata.observation_id,
        participant_id=metadata.participant_id.casefold(),
        condition=metadata.condition.casefold(),
        contrast_id=metadata.contrast_id.casefold(),
        duration_s=duration_s,
        status=status,
        exclusion_code=exclusion_code,
        raw_overlap_s=raw_overlap_s,
        clean_beat_span_s=clean_beat_span_s,
        source_data_supplied=metadata.source_data_supplied,
        eeg_exists=metadata.eeg_exists,
        cardiac_exists=metadata.cardiac_exists,
        requires_paired_state=metadata.requires_paired_state,
        paired_state_available=metadata.paired_state_available,
        pairing_resolved=metadata.pairing_resolved,
        protocol_match=metadata.protocol_match,
        notes=notes,
        endpoint_name=contract.endpoint_name,
        endpoint_alias=contract.endpoint_alias,
        is_standard_zlpi=bool(contract.is_standard_zlpi),
        has_required_duration=has_required_duration,
        has_required_lag_support=has_required_lag_support,
        endpoint_computable=endpoint_computable,
        endpoint_status=endpoint_status,
        endpoint_reason_code=endpoint_reason_code,
        standard_zlpi_computable=standard_zlpi_computable,
        standard_zlpi_reason_code=standard_zlpi_reason_code,
    )


def evaluate_all_durations(
    metadata_rows: Iterable[EligibilityMetadata],
) -> list[EligibilityDecision]:
    """Evaluate D240/D180/D120/D60 in deterministic observation order."""
    ordered = sorted(
        metadata_rows,
        key=lambda row: (
            row.dataset_id.casefold(),
            row.participant_id.casefold(),
            row.observation_id,
            row.condition.casefold(),
        ),
    )
    return [
        evaluate_duration_eligibility(metadata, duration_s)
        for metadata in ordered
        for duration_s in EXPECTED_DURATIONS_S
    ]


def _eligibility_qc_rows(
    decisions: Sequence[EligibilityDecision],
) -> list[dict[str, object]]:
    participant_counts: dict[tuple[str, int, str, str], set[str]] = {}
    observation_counts: dict[tuple[str, int, str, str], set[str]] = {}
    row_counts: dict[tuple[str, int, str, str], int] = {}
    for decision in decisions:
        key = (
            decision.dataset_id,
            decision.duration_s,
            decision.status,
            decision.exclusion_code,
        )
        row_counts[key] = row_counts.get(key, 0) + 1
        if decision.participant_id:
            participant_counts.setdefault(key, set()).add(decision.participant_id)
        else:
            participant_counts.setdefault(key, set())
        if decision.observation_id:
            observation_counts.setdefault(key, set()).add(decision.observation_id)
        else:
            observation_counts.setdefault(key, set())
    return [
        {
            "dataset_id": dataset_id,
            "duration_s": duration_s,
            "duration_role": "primary" if duration_s == 240 else "sensitivity",
            "status": status,
            "exclusion_code": exclusion_code,
            "n_records": row_counts[
                (dataset_id, duration_s, status, exclusion_code)
            ],
            "n_observations": len(
                observation_counts[
                    (dataset_id, duration_s, status, exclusion_code)
                ]
            ),
            "n_participants": len(
                participant_counts[
                    (dataset_id, duration_s, status, exclusion_code)
                ]
            ),
        }
        for dataset_id, duration_s, status, exclusion_code in sorted(row_counts)
    ]


def write_duration_eligibility(
    metadata_rows: Iterable[EligibilityMetadata],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write duration-level decisions and their exclusion/QC counts."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    decisions = evaluate_all_durations(metadata_rows)

    sample_row = EligibilityDecision(
        dataset_id="",
        observation_id="",
        participant_id="",
        condition="",
        contrast_id="",
        duration_s=240,
        status="not_supplied",
        exclusion_code="data_not_supplied",
        raw_overlap_s=None,
        clean_beat_span_s=None,
        source_data_supplied=False,
        eeg_exists=None,
        cardiac_exists=None,
        requires_paired_state=False,
        paired_state_available=None,
        pairing_resolved=None,
        protocol_match=None,
        notes="",
    ).to_row()
    eligibility_columns = list(
        dict.fromkeys([*sample_row.keys(), *STRUCTURED_NC_FIELDS])
    )
    eligibility_path = output_path / ELIGIBILITY_BY_DURATION_FILENAME
    with eligibility_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=eligibility_columns)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {field: decision.to_row().get(field, "") for field in eligibility_columns}
            )

    summary_columns = [
        "dataset_id",
        "duration_s",
        "duration_role",
        "status",
        "exclusion_code",
        "n_records",
        "n_observations",
        "n_participants",
    ]
    summary_path = output_path / ELIGIBILITY_QC_SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(_eligibility_qc_rows(decisions))
    return eligibility_path, summary_path


def write_protocol_audit(
    observations_by_dataset: Mapping[str, Sequence[CanonicalObservation]],
    output_dir: str | Path,
    *,
    dataset_roles: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write protocol declarations and participant-pair intersections."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    roles = {
        key.casefold(): value.casefold()
        for key, value in (dataset_roles or {}).items()
    }
    paired_sets = build_paired_subject_sets(observations_by_dataset)
    rows = _audit_rows(observations_by_dataset, paired_sets, roles)

    csv_path = output_path / PROTOCOL_AUDIT_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_path / PAIRED_SUBJECT_SETS_FILENAME
    json_path.write_text(
        json.dumps(paired_sets, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def observations_from_dataset_config(
    config: ConfirmatoryDatasetConfig,
) -> list[CanonicalObservation]:
    """Discover observations using the existing dataset adapters."""
    observations = build_observations(
        config.dataset_id,
        config.paths.raw_root,
        subjects=config.selection.subjects or None,
        tasks=config.selection.tasks or None,
        conditions=config.selection.conditions or None,
        sessions=config.selection.sessions or None,
        hiit_partition_mode=config.protocol.hiit_partition_mode,
    )
    spec = protocol_spec(config.dataset_id)
    contracts = _dataset_contracts(config, spec)
    normalized = [
        _normalize_observation_with_contracts(obs, contracts=contracts)
        for obs in observations
    ]
    validate_observation_identities(normalized)
    return normalized


def eligibility_metadata_from_observations(
    observations_by_dataset: Mapping[str, Sequence[CanonicalObservation]],
) -> list[EligibilityMetadata]:
    """Create structural M1c inputs; duration fields remain unknown.

    Raw overlap and clean beat span must be supplied by their metadata
    producers.  This helper intentionally does not read signals or derive
    beats.
    """
    paired_sets = build_paired_subject_sets(observations_by_dataset)
    datasets_payload = paired_sets["datasets"]
    assert isinstance(datasets_payload, Mapping)
    result: list[EligibilityMetadata] = []
    for dataset_id in sorted(PROTOCOL_SPECS):
        observations = tuple(observations_by_dataset.get(dataset_id, ()))
        if not observations:
            result.append(
                EligibilityMetadata(
                    dataset_id=dataset_id,
                    observation_id="",
                    participant_id="",
                    condition="",
                    source_data_supplied=False,
                    eeg_exists=None,
                    cardiac_exists=None,
                    pairing_resolved=None,
                    protocol_match=None,
                    notes="No observation metadata supplied for this dataset.",
                )
            )
            continue

        spec = PROTOCOL_SPECS[dataset_id]
        dataset_payload = datasets_payload[dataset_id]
        assert isinstance(dataset_payload, Mapping)
        contrasts_payload = dataset_payload["contrasts"]
        assert isinstance(contrasts_payload, Mapping)
        paired_ids: set[str] = set()
        unpaired_ids: set[str] = set()
        contrast_for_id: dict[str, str] = {}
        for contrast_id, raw_contrast in contrasts_payload.items():
            assert isinstance(raw_contrast, Mapping)
            for pair in raw_contrast["observations_by_pair"]:
                for field in (
                    "low_demand_observation_ids",
                    "cognitive_effort_observation_ids",
                ):
                    for observation_id in pair[field]:
                        paired_ids.add(observation_id)
                        contrast_for_id[observation_id] = str(contrast_id)
            for field in (
                "low_demand_only_keys",
                "cognitive_effort_only_keys",
            ):
                keys = raw_contrast[field]
                key_tuples = {
                    tuple(str(key[name]) for name in raw_contrast["pair_within"])
                    for key in keys
                }
                contrast_spec = next(
                    item
                    for item in spec.contrasts
                    if item.contrast_id == contrast_id
                )
                for observation in observations:
                    if _pair_key(observation, contrast_spec) in key_tuples:
                        unpaired_ids.add(observation.observation_id)
                        contrast_for_id[observation.observation_id] = str(
                            contrast_id
                        )

        paired_conditions = {
            condition
            for contrast in spec.contrasts
            for condition in (
                contrast.low_demand_condition,
                contrast.cognitive_effort_condition,
            )
        }
        for observation in observations:
            eeg_exists = observation.eeg_path.is_file()
            cardiac_exists = (
                eeg_exists
                if observation.ppg_source == "embedded_eeg"
                else (
                    observation.ppg_path is not None
                    and observation.ppg_path.is_file()
                )
            )
            requires_pairing = observation.condition_label.casefold() in paired_conditions
            paired_available: bool | None = None
            if requires_pairing:
                paired_available = observation.observation_id in paired_ids
                if observation.observation_id not in paired_ids | unpaired_ids:
                    paired_available = None
            result.append(
                EligibilityMetadata(
                    dataset_id=dataset_id,
                    observation_id=observation.observation_id,
                    participant_id=_participant_id(observation),
                    condition=observation.condition_label,
                    contrast_id=contrast_for_id.get(observation.observation_id, ""),
                    source_data_supplied=True,
                    eeg_exists=eeg_exists,
                    cardiac_exists=cardiac_exists,
                    raw_overlap_s=None,
                    clean_beat_span_s=None,
                    requires_paired_state=requires_pairing,
                    paired_state_available=paired_available,
                    pairing_resolved=True,
                    protocol_match=(
                        observation.condition_label.casefold()
                        in spec.low_demand_conditions
                        + spec.cognitive_effort_conditions
                    ),
                    notes="clean_beat_span_not_computed",
                )
            )
    return result


def _optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return _known_nonnegative(float(value), field="CSV duration metadata")


def _csv_rows_by_observation(
    paths: Iterable[str | Path],
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset_id = str(row.get("dataset_id", "")).strip().casefold()
                observation_id = str(row.get("observation_id", "")).strip()
                if not dataset_id or not observation_id:
                    continue
                key = (dataset_id, observation_id)
                if key in result:
                    raise ValueError(
                        "Duplicate metadata row for "
                        f"dataset={dataset_id!r}, observation={observation_id!r}."
                    )
                result[key] = dict(row)
    return result


def enrich_eligibility_metadata_from_csv(
    metadata_rows: Iterable[EligibilityMetadata],
    *,
    raw_audit_paths: Iterable[str | Path] = (),
    cardiac_qc_paths: Iterable[str | Path] = (),
) -> list[EligibilityMetadata]:
    """Join C0 raw-overlap audit and optional cardiac clean-span metadata.

    Missing files or rows leave fields unknown; this function does not run
    signal processing or peak detection. Raw overlap is never copied into
    ``clean_beat_span_s``.
    """
    raw_by_key = _csv_rows_by_observation(raw_audit_paths)
    cardiac_by_key = _csv_rows_by_observation(cardiac_qc_paths)
    enriched: list[EligibilityMetadata] = []
    for metadata in metadata_rows:
        key = (metadata.dataset_id.casefold(), metadata.observation_id)
        raw = raw_by_key.get(key)
        cardiac = cardiac_by_key.get(key)
        notes = [metadata.notes] if metadata.notes else []
        if raw is not None:
            notes.append("Raw overlap loaded from C0 data_audit.csv.")
        if cardiac is not None:
            notes.append("Clean beat span loaded from cardiac QC metadata.")
        elif "clean_beat_span_not_computed" not in " ".join(notes):
            notes.append("clean_beat_span_not_computed")
        raw_overlap = metadata.raw_overlap_s
        if raw is not None:
            # Prefer explicit raw_overlap_s; fall back to Stage-0-style column.
            raw_overlap = _optional_float(
                raw.get("raw_overlap_s") or raw.get("overlap_duration_s")
            )
        enriched.append(
            replace(
                metadata,
                eeg_exists=(
                    _optional_bool(raw.get("eeg_exists"))
                    if raw is not None
                    else metadata.eeg_exists
                ),
                cardiac_exists=(
                    _optional_bool(raw.get("cardiac_exists"))
                    if raw is not None
                    else metadata.cardiac_exists
                ),
                raw_overlap_s=raw_overlap,
                clean_beat_span_s=(
                    _optional_float(cardiac.get("clean_ibi_coverage_s"))
                    if cardiac is not None
                    else metadata.clean_beat_span_s
                ),
                notes=" ".join(part for part in notes if part),
            )
        )
    return enriched


def run_protocol_audit(
    master: ConfirmatoryMasterConfig,
    dataset_configs: Iterable[ConfirmatoryDatasetConfig],
    *,
    output_dir: str | Path | None = None,
    eligibility_metadata: Iterable[EligibilityMetadata] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Discover observations and write M1b/M1c audit outputs."""
    observations_by_dataset: dict[str, Sequence[CanonicalObservation]] = {}
    roles: dict[str, str] = {}
    for config in dataset_configs:
        if config.dataset_id in observations_by_dataset:
            raise ValueError(
                f"Duplicate dataset config for {config.dataset_id!r}; "
                "run full and smoke audits separately."
            )
        observations_by_dataset[config.dataset_id] = observations_from_dataset_config(
            config
        )
        roles[config.dataset_id] = config.role
    target = Path(output_dir) if output_dir is not None else master.output_root / "audit"
    protocol_paths = write_protocol_audit(
        observations_by_dataset,
        target,
        dataset_roles=roles,
    )
    metadata_rows = (
        list(eligibility_metadata)
        if eligibility_metadata is not None
        else eligibility_metadata_from_observations(observations_by_dataset)
    )
    eligibility_paths = write_duration_eligibility(metadata_rows, target)
    return (*protocol_paths, *eligibility_paths)


__all__ = [
    "PAIRED_SUBJECT_SETS_FILENAME",
    "PROTOCOL_AUDIT_FILENAME",
    "ELIGIBILITY_BY_DURATION_FILENAME",
    "ELIGIBILITY_QC_SUMMARY_FILENAME",
    "ELIGIBILITY_STATUSES",
    "EXCLUSION_CODES",
    "PROTOCOL_SPECS",
    "ContrastSpec",
    "EligibilityDecision",
    "EligibilityMetadata",
    "ProtocolSpec",
    "build_paired_subject_sets",
    "condition_semantics_for",
    "enrich_eligibility_metadata_from_csv",
    "eligibility_metadata_from_observations",
    "evaluate_all_durations",
    "evaluate_duration_eligibility",
    "observations_from_dataset_config",
    "protocol_spec",
    "run_protocol_audit",
    "write_duration_eligibility",
    "write_protocol_audit",
]
