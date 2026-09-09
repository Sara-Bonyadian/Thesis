"""Participant-level observed − null ΔZLPI inference for Figure 3 Panel A.

Independent unit (prespecified)
--------------------------------
The inferential unit is ``dataset_id::participant_id``. For HIIT, that ID is
the **session subject_id** (e.g. ``01_ph``, ``01_ps``), matching Figure 1 Panel B.
Other datasets use biological participant_id (protocol-normalized).

Primary estimand (prespecified slice)
-------------------------------------
For each eligible matched observation *i* in the slice
``D240 x absolute_log10 x theta x circular_shift x zlpi``:

    delta_i = observed_endpoint_index_i - null_mean_i

For each unit *p* and condition *c*:

    Delta_{p,c} = mean{delta_i : i in (p, c)}

For each unit *p*:

    Delta_p = mean{Delta_{p,c} : c in p}

Group (dataset) inference uses the unweighted mean of {Delta_p} with a Student-t
interval (df = n_participants - 1). Units with more conditions or observations
do **not** receive greater weight.

Main Figure 3 Panel A plots **one row per dataset**. Dataset-specific participant
forests are **internal QC** artifacts (under ``figures/internal_qc/``), not
manuscript or supplementary exports unless explicitly requested.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from ppg_eeg.confirmatory.group_tables import normalize_keys
from ppg_eeg.confirmatory.inference import FDR_ALPHA, bh_fdr
from ppg_eeg.confirmatory.nulls import (
    DEFAULT_N_SURROGATES,
    NULL_TYPE_AR1_INNOVATIONS,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_CROSS_SUBJECT_MISMATCH,
    NULL_TYPE_PHASE_RANDOMIZATION,
    SMOKE_N_SURROGATES,
)
from ppg_eeg.confirmatory.paired_delta_inference import observation_unit_id

PRIMARY_DURATION_S = 240
PRIMARY_REPRESENTATION = "absolute_log10"
PRIMARY_BAND = "theta"
PRIMARY_NULL_TYPE = NULL_TYPE_CIRCULAR_SHIFT
PRIMARY_ENDPOINT = "zlpi"

SECONDARY_NULL_TYPES: tuple[str, ...] = (
    NULL_TYPE_PHASE_RANDOMIZATION,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CROSS_SUBJECT_MISMATCH,
    NULL_TYPE_AR1_INNOVATIONS,
)

# Preferred display / export order for multi-dataset forests.
DATASET_DISPLAY_ORDER: tuple[str, ...] = (
    "hiit",
    "ds003838",
    "ds006848",
    "ds003690",
    "ds004587",
    "ds004582",
    "ds003816",
    "mindfulness",
)

INTERPRETATION_EXCEEDS = "observed significantly exceeds null"
INTERPRETATION_DIRECTIONAL = "directionally positive but inconclusive"
INTERPRETATION_NO_EVIDENCE = "no evidence of difference"
INTERPRETATION_LOWER = "observed is lower than null"
INTERPRETATION_INSUFFICIENT = "insufficient participant-level data"

INDEPENDENT_UNIT_VERDICT = (
    "Independent unit = dataset_id::participant_id. For HIIT, participant_id is "
    "the session subject_id (e.g. 01_ph, 01_ps), matching Figure 1 Panel B — PH/PS "
    "are separate units. External BIDS datasets use protocol-normalized "
    "biological participant_id (subject / ses-stripped ID)."
)


@dataclass(frozen=True)
class MatchedNullObservation:
    dataset_id: str
    subject_id: str
    participant_id: str
    participant_unit_id: str
    session_id: str
    observation_id: str
    condition: str
    modality: str
    band: str
    null_type: str
    power_representation: str
    duration_s: int
    endpoint_name: str
    n_surrogates_requested: int
    n_surrogates_finite: int
    observed_endpoint_index: float
    null_mean: float
    delta_obs_minus_null: float
    rng_seed_u64: str


@dataclass(frozen=True)
class ParticipantConditionNullDelta:
    """Condition-level mean δ within one biological participant."""

    dataset_id: str
    participant_id: str
    participant_unit_id: str
    session_id: str
    condition: str
    display_label: str
    band: str
    null_type: str
    n_observations: int
    mean_observed: float
    mean_null: float
    delta_pc: float
    n_surrogates_requested_min: int
    n_surrogates_requested_max: int


@dataclass(frozen=True)
class ParticipantNullDelta:
    """One equal-weight Δ_p per biological participant."""

    dataset_id: str
    participant_id: str
    participant_unit_id: str
    display_label: str
    band: str
    null_type: str
    n_observations: int
    n_conditions: int
    n_sessions: int
    mean_observed: float
    mean_null: float
    delta_p: float
    n_surrogates_requested_min: int
    n_surrogates_requested_max: int


@dataclass(frozen=True)
class NullDeltaInference:
    slice_label: str
    band: str
    null_type: str
    is_primary_slice: bool
    dataset_id: str
    n_observations: int
    n_participant_conditions: int
    n_participants: int
    mean_delta: float
    se_delta: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value_two_sided: float
    p_value_greater: float
    df: int
    cohen_dz: float
    ci_method: str
    interpretation: str
    run_class: str
    n_surrogates_requested_min: int
    n_surrogates_requested_max: int
    notes: str
    sample_size_label: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NullSliceAnalysis:
    matched: list[MatchedNullObservation]
    condition_deltas: list[ParticipantConditionNullDelta]
    participants: list[ParticipantNullDelta]
    dataset_inferences: list[NullDeltaInference]
    pooled_inference: NullDeltaInference


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = _as_str(value).casefold()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return default


def canonicalize_participant_id(participant_id: str) -> str:
    """Stable biological ID; collapse digit padding (``01`` → ``1``)."""
    text = _as_str(participant_id).casefold()
    if text.isdigit():
        return str(int(text))
    return text


def resolve_biological_keys(row: Mapping[str, object]) -> dict[str, str]:
    """Resolve dataset-scoped participant and session keys.

    Independent-unit rules (metadata-driven, no dataset-name branch):

    1. Declared ``participant_unit_id`` / ``analysis_unit_id`` wins when present.
    2. Session-qualified ``subject_id`` values (``{id}_{session}``, e.g.
       ``01_ph``) are the locked Panel-B analysis unit and are kept verbatim.
    3. Otherwise the biological participant ID is used (digit padding collapsed).
    """
    keys = normalize_keys(row)
    dataset_id = _as_str(keys["dataset_id"]).casefold()
    session_id = _as_str(keys["session_id"], "single").casefold() or "single"
    biological = canonicalize_participant_id(keys["participant_id"])
    subject_id = _as_str(keys["subject_id"] or row.get("subject_id")).casefold()
    declared_unit = _as_str(
        row.get("participant_unit_id") or row.get("analysis_unit_id")
    ).casefold()
    if declared_unit:
        participant_id = declared_unit
    elif subject_id and "_" in subject_id:
        # Session-qualified subject token is the independent unit.
        participant_id = subject_id
        if session_id in {"", "single"}:
            _stem, suffix = subject_id.rsplit("_", 1)
            if suffix and not suffix.isdigit():
                session_id = suffix
    elif subject_id:
        participant_id = canonicalize_participant_id(subject_id)
    elif session_id not in {"", "single"} and biological:
        participant_id = f"{biological}_{session_id}"
    else:
        participant_id = biological

    unit_row = {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
    }
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "subject_id": subject_id or _as_str(row.get("subject_id")),
        "condition": _as_str(keys["condition"] or row.get("condition")),
        "observation_id": _as_str(keys["observation_id"] or row.get("observation_id")),
        "participant_unit_id": observation_unit_id(unit_row, unit_field="participant_id"),
    }


def anonymized_display_label(participant_id: str, *, index: int) -> str:
    """Stable anonymized label ``P01``, ``P02``, … (1-based index)."""
    _ = participant_id
    return f"P{index:02d}"


def sample_size_annotation(*, n_participants: int, n_participant_conditions: int) -> str:
    """Explicit wording that separates biological n from condition estimates."""
    if n_participants <= 0:
        return "n=0 participants"
    if n_participant_conditions <= n_participants:
        return f"n={n_participants} participants"
    return (
        f"{n_participant_conditions} condition estimates from "
        f"{n_participants} participants"
    )


def classify_run_surrogates(n_requested_values: Sequence[int]) -> str:
    """Label smoke vs production from surrogate request counts."""
    finite = [int(v) for v in n_requested_values if int(v) > 0]
    if not finite:
        return "unknown_surrogate_count"
    minimum = min(finite)
    maximum = max(finite)
    if maximum < DEFAULT_N_SURROGATES:
        return "smoke_diagnostic"
    if minimum >= DEFAULT_N_SURROGATES:
        return "production"
    return "mixed_surrogate_count"


def classify_primary_interpretation(
    *,
    n_participants: int,
    mean_delta: float,
    ci_low: float,
    ci_high: float,
) -> str:
    """Map participant-level CI to a confirmatory interpretation category."""
    if n_participants < 2 or not all(
        math.isfinite(v) for v in (mean_delta, ci_low, ci_high)
    ):
        return INTERPRETATION_INSUFFICIENT
    if ci_low > 0.0:
        return INTERPRETATION_EXCEEDS
    if ci_high < 0.0:
        return INTERPRETATION_LOWER
    if mean_delta > 0.0:
        return INTERPRETATION_DIRECTIONAL
    return INTERPRETATION_NO_EVIDENCE


def filter_matched_null_observations(
    rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int = PRIMARY_DURATION_S,
    power_representation: str = PRIMARY_REPRESENTATION,
    band: str | None = None,
    null_type: str | None = None,
    endpoint_name: str = PRIMARY_ENDPOINT,
) -> list[MatchedNullObservation]:
    """Keep eligible rows with finite matched observed and null_mean."""
    out: list[MatchedNullObservation] = []
    for row in rows:
        if _as_str(row.get("endpoint_name"), endpoint_name).casefold() != endpoint_name:
            continue
        if _as_int(row.get("duration_s"), duration_s) != int(duration_s):
            continue
        if (
            _as_str(row.get("power_representation"), power_representation).casefold()
            != power_representation.casefold()
        ):
            continue
        row_band = _as_str(row.get("band")).casefold()
        row_null = _as_str(row.get("null_type")).casefold()
        if band is not None and row_band != band.casefold():
            continue
        if null_type is not None and row_null != null_type.casefold():
            continue
        if not _as_bool(row.get("observed_eligible"), True):
            continue
        observed = _as_float(row.get("observed_endpoint_index"))
        null_mean = _as_float(row.get("null_mean"))
        if not (math.isfinite(observed) and math.isfinite(null_mean)):
            continue
        keys = resolve_biological_keys(row)
        if not keys["participant_id"]:
            continue
        out.append(
            MatchedNullObservation(
                dataset_id=keys["dataset_id"],
                subject_id=keys["subject_id"],
                participant_id=keys["participant_id"],
                participant_unit_id=keys["participant_unit_id"],
                session_id=keys["session_id"],
                observation_id=keys["observation_id"],
                condition=keys["condition"],
                modality=_as_str(row.get("modality"), "default"),
                band=row_band,
                null_type=row_null,
                power_representation=power_representation.casefold(),
                duration_s=int(duration_s),
                endpoint_name=endpoint_name,
                n_surrogates_requested=_as_int(row.get("n_surrogates_requested")),
                n_surrogates_finite=_as_int(row.get("n_surrogates_finite")),
                observed_endpoint_index=observed,
                null_mean=null_mean,
                delta_obs_minus_null=float(observed - null_mean),
                rng_seed_u64=_as_str(row.get("rng_seed_u64")),
            )
        )
    return out


def condition_deltas_from_matched(
    matched: Sequence[MatchedNullObservation],
) -> list[ParticipantConditionNullDelta]:
    """One Δ_{p,c} per biological participant × condition."""
    buckets: dict[tuple[str, str], list[MatchedNullObservation]] = {}
    for row in matched:
        key = (row.participant_unit_id, _as_str(row.condition).casefold() or "unknown")
        buckets.setdefault(key, []).append(row)

    # Stable anonymized labels within each dataset.
    by_dataset: dict[str, list[str]] = {}
    for rows in buckets.values():
        first = rows[0]
        by_dataset.setdefault(first.dataset_id, [])
        if first.participant_id not in by_dataset[first.dataset_id]:
            by_dataset[first.dataset_id].append(first.participant_id)
    label_map: dict[str, str] = {}
    for dataset_id, pids in by_dataset.items():
        for index, pid in enumerate(sorted(pids), start=1):
            label_map[f"{dataset_id}::{pid}"] = anonymized_display_label(pid, index=index)

    out: list[ParticipantConditionNullDelta] = []
    for (_unit, _cond), rows in sorted(
        buckets.items(),
        key=lambda item: (
            item[1][0].dataset_id,
            item[1][0].participant_id,
            item[0][1],
        ),
    ):
        deltas = [r.delta_obs_minus_null for r in rows]
        obs = [r.observed_endpoint_index for r in rows]
        nulls = [r.null_mean for r in rows]
        n_req = [r.n_surrogates_requested for r in rows]
        first = rows[0]
        out.append(
            ParticipantConditionNullDelta(
                dataset_id=first.dataset_id,
                participant_id=first.participant_id,
                participant_unit_id=first.participant_unit_id,
                session_id=first.session_id,
                condition=first.condition,
                display_label=label_map[first.participant_unit_id],
                band=first.band,
                null_type=first.null_type,
                n_observations=len(rows),
                mean_observed=float(np.mean(obs)),
                mean_null=float(np.mean(nulls)),
                delta_pc=float(np.mean(deltas)),
                n_surrogates_requested_min=int(min(n_req)),
                n_surrogates_requested_max=int(max(n_req)),
            )
        )
    return out


def participant_deltas_from_condition(
    condition_rows: Sequence[ParticipantConditionNullDelta],
) -> list[ParticipantNullDelta]:
    """One Δ_p per biological participant = mean of condition-level means."""
    buckets: dict[str, list[ParticipantConditionNullDelta]] = {}
    for row in condition_rows:
        buckets.setdefault(row.participant_unit_id, []).append(row)
    out: list[ParticipantNullDelta] = []
    for unit_id, rows in sorted(
        buckets.items(),
        key=lambda item: (item[1][0].dataset_id, item[1][0].participant_id),
    ):
        deltas = [r.delta_pc for r in rows]
        obs = [r.mean_observed for r in rows]
        nulls = [r.mean_null for r in rows]
        n_req = [r.n_surrogates_requested_min for r in rows] + [
            r.n_surrogates_requested_max for r in rows
        ]
        sessions = {_as_str(r.session_id) for r in rows if _as_str(r.session_id)}
        first = rows[0]
        out.append(
            ParticipantNullDelta(
                dataset_id=first.dataset_id,
                participant_id=first.participant_id,
                participant_unit_id=unit_id,
                display_label=first.display_label,
                band=first.band,
                null_type=first.null_type,
                n_observations=int(sum(int(r.n_observations) for r in rows)),
                n_conditions=len(rows),
                n_sessions=len(sessions),
                mean_observed=float(np.mean(obs)),
                mean_null=float(np.mean(nulls)),
                delta_p=float(np.mean(deltas)),
                n_surrogates_requested_min=int(min(n_req)) if n_req else 0,
                n_surrogates_requested_max=int(max(n_req)) if n_req else 0,
            )
        )
    return out


def participant_deltas_from_matched(
    matched: Sequence[MatchedNullObservation],
) -> list[ParticipantNullDelta]:
    """One equal-weight Δ_p per biological participant."""
    return participant_deltas_from_condition(condition_deltas_from_matched(matched))


def _slice_label(band: str, null_type: str) -> str:
    return (
        f"D{PRIMARY_DURATION_S}|{PRIMARY_REPRESENTATION}|{band}|{null_type}|{PRIMARY_ENDPOINT}"
    )


def _estimand_notes(*, run_class: str) -> str:
    notes = (
        "Estimand = unweighted mean of biological-participant Δ_p where "
        "Δ_p = mean of condition-level mean(observed − null_mean); "
        "equal participant weight; Student-t CI df=n_participants-1. "
        f"{INDEPENDENT_UNIT_VERDICT}"
    )
    if run_class == "smoke_diagnostic":
        notes += (
            f" Surrogate count < {DEFAULT_N_SURROGATES} "
            f"(smoke uses {SMOKE_N_SURROGATES}); smoke diagnostic only."
        )
    return notes


def infer_participant_null_deltas(
    participant_rows: Sequence[ParticipantNullDelta],
    *,
    band: str,
    null_type: str,
    is_primary_slice: bool,
    dataset_id: str = "",
    n_participant_conditions: int | None = None,
) -> NullDeltaInference:
    """Student-*t* inference on equal-weight biological-participant Δ_p values."""
    deltas = [float(r.delta_p) for r in participant_rows if math.isfinite(float(r.delta_p))]
    n = len(deltas)
    n_obs = int(sum(int(r.n_observations) for r in participant_rows))
    n_cond = (
        int(n_participant_conditions)
        if n_participant_conditions is not None
        else int(sum(int(r.n_conditions) for r in participant_rows))
    )
    n_req = [int(r.n_surrogates_requested_min) for r in participant_rows] + [
        int(r.n_surrogates_requested_max) for r in participant_rows
    ]
    run_class = classify_run_surrogates(n_req)
    notes = _estimand_notes(run_class=run_class)
    size_label = sample_size_annotation(
        n_participants=n,
        n_participant_conditions=n_cond,
    )
    slice_label = _slice_label(band, null_type)
    resolved_dataset = _as_str(dataset_id)
    if not resolved_dataset:
        datasets = sorted({_as_str(r.dataset_id) for r in participant_rows if _as_str(r.dataset_id)})
        resolved_dataset = datasets[0] if len(datasets) == 1 else "pooled"

    def _empty(*, n_participants: int, mean: float, extra_note: str = "") -> NullDeltaInference:
        return NullDeltaInference(
            slice_label=slice_label,
            band=band,
            null_type=null_type,
            is_primary_slice=is_primary_slice,
            dataset_id=resolved_dataset,
            n_observations=n_obs,
            n_participant_conditions=n_cond,
            n_participants=n_participants,
            mean_delta=mean,
            se_delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            t_stat=float("nan"),
            p_value_two_sided=float("nan"),
            p_value_greater=float("nan"),
            df=0,
            cohen_dz=float("nan"),
            ci_method="student_t_participant_means",
            interpretation=INTERPRETATION_INSUFFICIENT,
            run_class=run_class,
            n_surrogates_requested_min=min(n_req) if n_req else 0,
            n_surrogates_requested_max=max(n_req) if n_req else 0,
            notes=notes + extra_note,
            sample_size_label=sample_size_annotation(
                n_participants=n_participants,
                n_participant_conditions=n_cond,
            ),
        )

    if n == 0:
        return _empty(n_participants=0, mean=float("nan"))

    arr = np.asarray(deltas, dtype=float)
    mean = float(np.mean(arr))
    if n == 1:
        return _empty(
            n_participants=1,
            mean=mean,
            extra_note=" Need ≥2 participants for a Student-t CI.",
        )

    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    t_stat, p_two = stats.ttest_1samp(arr, popmean=0.0)
    t_stat_f = float(t_stat)
    p_two_f = float(p_two)
    if math.isfinite(t_stat_f):
        p_greater = float(stats.t.sf(t_stat_f, df=df))
    else:
        p_greater = float("nan")
    cohen = float(mean / sd) if sd > 0 else float("nan")
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    interpretation = classify_primary_interpretation(
        n_participants=n,
        mean_delta=mean,
        ci_low=ci_low,
        ci_high=ci_high,
    )
    return NullDeltaInference(
        slice_label=slice_label,
        band=band,
        null_type=null_type,
        is_primary_slice=is_primary_slice,
        dataset_id=resolved_dataset,
        n_observations=n_obs,
        n_participant_conditions=n_cond,
        n_participants=n,
        mean_delta=mean,
        se_delta=se,
        ci_low=ci_low,
        ci_high=ci_high,
        t_stat=t_stat_f,
        p_value_two_sided=p_two_f,
        p_value_greater=p_greater,
        df=df,
        cohen_dz=cohen,
        ci_method="student_t_participant_means",
        interpretation=interpretation,
        run_class=run_class,
        n_surrogates_requested_min=min(n_req) if n_req else 0,
        n_surrogates_requested_max=max(n_req) if n_req else 0,
        notes=notes,
        sample_size_label=size_label,
    )


def infer_dataset_null_deltas(
    participant_rows: Sequence[ParticipantNullDelta],
    *,
    band: str,
    null_type: str,
    is_primary_slice: bool,
) -> list[NullDeltaInference]:
    """One Student-*t* inference per dataset from biological-participant Δ_p."""
    by_dataset: dict[str, list[ParticipantNullDelta]] = {}
    for row in participant_rows:
        by_dataset.setdefault(row.dataset_id, []).append(row)

    def _sort_key(dataset_id: str) -> tuple[int, str]:
        try:
            return (DATASET_DISPLAY_ORDER.index(dataset_id), dataset_id)
        except ValueError:
            return (len(DATASET_DISPLAY_ORDER), dataset_id)

    out: list[NullDeltaInference] = []
    for dataset_id in sorted(by_dataset, key=_sort_key):
        rows = by_dataset[dataset_id]
        out.append(
            infer_participant_null_deltas(
                rows,
                band=band,
                null_type=null_type,
                is_primary_slice=is_primary_slice,
                dataset_id=dataset_id,
                n_participant_conditions=int(sum(int(r.n_conditions) for r in rows)),
            )
        )
    return out


def leave_one_participant_out(
    participant_rows: Sequence[ParticipantNullDelta],
    *,
    band: str,
    null_type: str,
) -> list[dict[str, object]]:
    """Recompute dataset mean Δ omitting each participant once (within dataset)."""
    rows = list(participant_rows)
    out: list[dict[str, object]] = []
    for omitted in rows:
        kept = [
            r
            for r in rows
            if r.participant_unit_id != omitted.participant_unit_id
            and r.dataset_id == omitted.dataset_id
        ]
        # Include other-dataset participants only for within-dataset LOO scope.
        inference = infer_participant_null_deltas(
            kept,
            band=band,
            null_type=null_type,
            is_primary_slice=True,
            dataset_id=omitted.dataset_id,
        )
        out.append(
            {
                "omitted_participant_unit_id": omitted.participant_unit_id,
                "omitted_dataset_id": omitted.dataset_id,
                "omitted_participant_id": omitted.participant_id,
                "omitted_subject_id": omitted.participant_id,
                "omitted_display_label": omitted.display_label,
                "omitted_delta_p": omitted.delta_p,
                "n_participants_remaining": inference.n_participants,
                "mean_delta": inference.mean_delta,
                "ci_low": inference.ci_low,
                "ci_high": inference.ci_high,
                "p_value_two_sided": inference.p_value_two_sided,
                "interpretation": inference.interpretation,
                "sample_size_label": inference.sample_size_label,
            }
        )
    return out


def analyze_null_slice(
    null_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    null_type: str,
    is_primary_slice: bool,
) -> tuple[list[ParticipantNullDelta], NullDeltaInference, list[MatchedNullObservation]]:
    """Analyze one band×null slice (backward-compatible 3-tuple).

    ``NullDeltaInference`` is the pooled biological-participant inference across
    all datasets in the slice (used by secondary FDR). Prefer
    :func:`analyze_null_slice_full` for production Panel A exports.
    """
    analysis = analyze_null_slice_full(
        null_rows,
        band=band,
        null_type=null_type,
        is_primary_slice=is_primary_slice,
    )
    return analysis.participants, analysis.pooled_inference, analysis.matched


def analyze_null_slice_full(
    null_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    null_type: str,
    is_primary_slice: bool,
) -> NullSliceAnalysis:
    """Full primary-slice analysis with dataset-level and participant-level layers."""
    matched = filter_matched_null_observations(
        null_rows,
        band=band,
        null_type=null_type,
    )
    condition_deltas = condition_deltas_from_matched(matched)
    participants = participant_deltas_from_condition(condition_deltas)
    dataset_inferences = infer_dataset_null_deltas(
        participants,
        band=band,
        null_type=null_type,
        is_primary_slice=is_primary_slice,
    )
    pooled = infer_participant_null_deltas(
        participants,
        band=band,
        null_type=null_type,
        is_primary_slice=is_primary_slice,
        dataset_id="pooled" if len(dataset_inferences) != 1 else (
            dataset_inferences[0].dataset_id if dataset_inferences else ""
        ),
        n_participant_conditions=len(condition_deltas),
    )
    # Single-dataset smoke: keep dataset_id on the pooled summary for clarity.
    if len(dataset_inferences) == 1:
        pooled = replace(
            pooled,
            dataset_id=dataset_inferences[0].dataset_id,
            notes=(
                pooled.notes
                + " No cross-dataset pooled estimate plotted (single dataset)."
            ),
        )
    else:
        pooled = replace(
            pooled,
            dataset_id="pooled",
            notes=(
                pooled.notes
                + " Cross-dataset mean is not plotted on the main panel "
                "(no prespecified multi-dataset meta-analytic weighting for Panel A)."
            ),
        )
    return NullSliceAnalysis(
        matched=matched,
        condition_deltas=condition_deltas,
        participants=participants,
        dataset_inferences=dataset_inferences,
        pooled_inference=pooled,
    )


def secondary_band_null_fdr_table(
    null_rows: Sequence[Mapping[str, object]],
    *,
    bands: Sequence[str] = ("theta", "alpha", "beta"),
    null_types: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Biological-participant inference for non-primary band×null cells + BH-FDR.

    Primary ``theta × circular_shift`` is excluded from the FDR family but may
    be listed with ``in_fdr_family=False`` for completeness when present.
    """
    types = list(null_types) if null_types is not None else [
        PRIMARY_NULL_TYPE,
        *SECONDARY_NULL_TYPES,
    ]
    records: list[dict[str, object]] = []
    for band in bands:
        for null_type in types:
            is_primary = (
                band.casefold() == PRIMARY_BAND
                and null_type.casefold() == PRIMARY_NULL_TYPE
            )
            _participants, inference, _matched = analyze_null_slice(
                null_rows,
                band=band,
                null_type=null_type,
                is_primary_slice=is_primary,
            )
            records.append(
                {
                    **inference.as_dict(),
                    "in_fdr_family": (not is_primary),
                    "fdr_alpha": FDR_ALPHA,
                    "q_value": float("nan"),
                    "reject_fdr": False,
                }
            )

    family_idx = [i for i, r in enumerate(records) if bool(r["in_fdr_family"])]
    family_p = [float(records[i]["p_value_two_sided"]) for i in family_idx]
    q_values = bh_fdr(family_p, alpha=FDR_ALPHA)
    for idx, q in zip(family_idx, q_values, strict=True):
        records[idx]["q_value"] = q
        p_raw = float(records[idx]["p_value_two_sided"])
        records[idx]["reject_fdr"] = bool(
            math.isfinite(q) and q <= FDR_ALPHA and math.isfinite(p_raw)
        )
    return records


def independent_unit_verdict_rows(
    *,
    dataset_ids: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Human-readable independent-unit verdict per confirmatory dataset."""
    rows = [
        {
            "dataset_id": "hiit",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "01_ph, 01_ps",
            "verdict": (
                "PH and PS are separate session subject_ids (Panel B-aligned); "
                "treat as dataset::session_subject (e.g. hiit::01_ph), not "
                "collapsed biological participant alone."
            ),
        },
        {
            "dataset_id": "ds003838",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "BIDS subject_id",
            "verdict": "One biological participant per subject_id.",
        },
        {
            "dataset_id": "ds006848",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "BIDS subject_id",
            "verdict": "One biological participant per subject_id.",
        },
        {
            "dataset_id": "ds003690",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "ses-stripped observation stem",
            "verdict": (
                "Multiple runs/conditions within participant are repeated "
                "measures; collapse to biological participant."
            ),
        },
        {
            "dataset_id": "ds004587",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "ses-stripped observation stem",
            "verdict": "Repeated conditions collapse to biological participant.",
        },
        {
            "dataset_id": "ds004582",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "ses-stripped observation stem",
            "verdict": "Single-state dataset; participant_id is the unit.",
        },
        {
            "dataset_id": "ds003816",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "ses-stripped observation stem",
            "verdict": "Repeated task labels collapse to biological participant.",
        },
        {
            "dataset_id": "mindfulness",
            "independent_unit": "dataset_id::participant_id",
            "example_raw_subject_ids": "mbd-##",
            "verdict": (
                "part1/part2 sessions and graded steps are repeated measures "
                "within mbd participant."
            ),
        },
    ]
    if dataset_ids is None:
        return rows
    wanted = {str(ds).strip().casefold() for ds in dataset_ids if str(ds).strip()}
    return [
        row
        for row in rows
        if str(row.get("dataset_id", "")).strip().casefold() in wanted
    ]


__all__ = [
    "DATASET_DISPLAY_ORDER",
    "INDEPENDENT_UNIT_VERDICT",
    "PRIMARY_BAND",
    "PRIMARY_DURATION_S",
    "PRIMARY_ENDPOINT",
    "PRIMARY_NULL_TYPE",
    "PRIMARY_REPRESENTATION",
    "SECONDARY_NULL_TYPES",
    "MatchedNullObservation",
    "NullDeltaInference",
    "NullSliceAnalysis",
    "ParticipantConditionNullDelta",
    "ParticipantNullDelta",
    "analyze_null_slice",
    "analyze_null_slice_full",
    "anonymized_display_label",
    "canonicalize_participant_id",
    "classify_primary_interpretation",
    "classify_run_surrogates",
    "condition_deltas_from_matched",
    "filter_matched_null_observations",
    "independent_unit_verdict_rows",
    "infer_dataset_null_deltas",
    "infer_participant_null_deltas",
    "leave_one_participant_out",
    "participant_deltas_from_condition",
    "participant_deltas_from_matched",
    "resolve_biological_keys",
    "sample_size_annotation",
    "secondary_band_null_fdr_table",
]
