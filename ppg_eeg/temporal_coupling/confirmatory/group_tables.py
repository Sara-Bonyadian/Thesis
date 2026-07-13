"""Participant-level tables and within-subject paired contrasts (M8).

Combines M6 endpoint metrics with M7 peak-fit parameters, normalizes
dataset-specific identity keys, and emits task-minus-low-demand contrasts
only for valid within-subject pairings. ZLPI, MWPI, and SWPI stay separate.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ...datasets import CanonicalObservation
from .endpoints import METRICS_TEMPLATE
from .peak_model import PARAMS_FILENAME
from .protocol_audit import (
    PROTOCOL_SPECS,
    ContrastSpec,
    ProtocolSpec,
    _participant_id,
    _run_id,
    _session_id,
    protocol_spec,
)

SUBJECT_LEVEL_FILENAME = "subject_level_metrics.csv"
PAIRED_CONTRASTS_FILENAME = "paired_contrasts.csv"
PAIRING_QC_FILENAME = "pairing_qc.csv"

_SES_RE = re.compile(r"(?:^|[-_])ses[-_]?([a-zA-Z0-9]+)(?:$|[-_])", re.IGNORECASE)

JOIN_FIELDS = (
    "dataset_id",
    "observation_id",
    "duration_s",
    "endpoint_name",
    "band",
    "power_representation",
)

SUBJECT_GRAIN_FIELDS = (
    "dataset_id",
    "participant_id",
    "session_id",
    "condition",
    "duration_s",
    "endpoint_name",
    "band",
    "power_representation",
)

SUBJECT_LEVEL_FIELDS = (
    "dataset_id",
    "participant_id",
    "session_id",
    "condition",
    "duration_s",
    "duration_role",
    "endpoint_name",
    "endpoint_alias",
    "is_standard_zlpi",
    "pool_with_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "n_runs",
    "run_ids",
    "observation_ids",
    "endpoint_eligible",
    "endpoint_exclusion_reason",
    "endpoint_index",
    "local_prominence",
    "r0",
    "z0",
    "n_common_support",
    "peak_converged",
    "has_identifiable_peak",
    "report_timing_shift",
    "baseline_C",
    "peak_height_A",
    "peak_center_mu_s",
    "sigma_s",
    "fwhm_s",
    "peak_exclusion_reason",
)

PAIRED_CONTRAST_FIELDS = (
    "dataset_id",
    "contrast_id",
    "participant_id",
    "session_id",
    "pair_within",
    "low_demand_condition",
    "cognitive_effort_condition",
    "duration_s",
    "duration_role",
    "endpoint_name",
    "is_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_representation",
    "n_low_runs",
    "n_effort_runs",
    "low_observation_ids",
    "effort_observation_ids",
    "delta_endpoint_index",
    "delta_local_prominence",
    "delta_peak_height_A",
    "delta_peak_center_mu_s",
    "delta_fwhm_s",
    "mu_contrast_eligible",
    "low_endpoint_eligible",
    "effort_endpoint_eligible",
    "low_has_identifiable_peak",
    "effort_has_identifiable_peak",
    "low_endpoint_index",
    "effort_endpoint_index",
    "low_local_prominence",
    "effort_local_prominence",
    "low_peak_height_A",
    "effort_peak_height_A",
    "low_peak_center_mu_s",
    "effort_peak_center_mu_s",
    "low_fwhm_s",
    "effort_fwhm_s",
)

PAIRING_QC_FIELDS = (
    "dataset_id",
    "contrast_id",
    "participant_id",
    "session_id",
    "pair_within",
    "low_demand_condition",
    "cognitive_effort_condition",
    "pairing_status",
    "n_low_keys",
    "n_effort_keys",
    "n_paired_keys",
    "n_low_runs",
    "n_effort_runs",
    "low_observation_ids",
    "effort_observation_ids",
    "duplicate_runs",
    "notes",
)


@dataclass(frozen=True)
class GroupTableResult:
    subject_level_rows: tuple[dict[str, object], ...]
    paired_contrast_rows: tuple[dict[str, object], ...]
    pairing_qc_rows: tuple[dict[str, object], ...]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return bool(value)


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = _as_str(value)
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _as_int(value: object, default: int = 0) -> int:
    if value is None or _as_str(value) == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _semicolon_join(values: Sequence[str]) -> str:
    return ";".join(sorted({v for v in values if v}))


def infer_session_label(row: Mapping[str, object]) -> str:
    """Best-effort session label when M6/M7 rows omit ``session_label``."""
    for key in ("session_id", "session_label"):
        if key in row and _as_str(row.get(key)):
            return _as_str(row.get(key)).casefold()

    observation_id = _as_str(row.get("observation_id")).casefold()
    match = _SES_RE.search(observation_id)
    if match:
        return match.group(1).casefold()

    dataset_id = _as_str(row.get("dataset_id")).casefold()
    subject_id = _as_str(row.get("subject_id")).casefold()
    if dataset_id == "hiit" and "_" in subject_id:
        return subject_id.rsplit("_", 1)[-1].casefold()
    if dataset_id == "mindfulness":
        # mindfulness-{participant}-{session}-task-{condition}
        parts = observation_id.split("-")
        if len(parts) >= 4 and parts[0] == "mindfulness":
            # mbd-01 may occupy two tokens; session follows participant block.
            task_idx = observation_id.find("-task-")
            if task_idx > 0:
                head = observation_id[:task_idx]
                # head = mindfulness-mbd-01-part1
                tokens = head.split("-")
                if len(tokens) >= 4:
                    return tokens[-1].casefold()
    return "single"


def normalize_keys(row: Mapping[str, object]) -> dict[str, str]:
    """Normalize participant/session/run keys for one metric or peak row."""
    dataset_id = _as_str(row.get("dataset_id")).casefold()
    observation_id = _as_str(row.get("observation_id")).casefold()
    subject_id = _as_str(row.get("subject_id")).casefold()
    condition = _as_str(row.get("condition") or row.get("task")).casefold()
    session_label = infer_session_label(row)

    if _as_str(row.get("participant_id")):
        participant_id = _as_str(row.get("participant_id")).casefold()
        run_id = _as_str(row.get("run_id"), "single").casefold()
        session_id = (
            _as_str(row.get("session_id") or row.get("session_label"), session_label)
            .casefold()
        )
        return {
            "dataset_id": dataset_id,
            "participant_id": participant_id,
            "session_id": session_id,
            "run_id": run_id or "single",
            "condition": condition,
            "observation_id": observation_id,
            "subject_id": subject_id,
        }

    obs = CanonicalObservation(
        dataset_id=dataset_id or "unknown",
        observation_id=observation_id or "unknown",
        subject_id=subject_id or "unknown",
        task_label=condition or "unknown",
        condition_label=condition or "unknown",
        eeg_path=Path("."),
        eeg_format="n/a",
        ppg_source="n/a",
        session_label=session_label,
    )
    return {
        "dataset_id": dataset_id,
        "participant_id": _participant_id(obs),
        "session_id": _session_id(obs),
        "run_id": _run_id(obs),
        "condition": condition,
        "observation_id": observation_id,
        "subject_id": subject_id,
    }


def _join_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        _as_str(row.get(field)).casefold()
        if field != "duration_s"
        else str(_as_int(row.get("duration_s")))
        for field in JOIN_FIELDS
    )


def _subject_grain(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(_as_str(row.get(field)).casefold() for field in SUBJECT_GRAIN_FIELDS)


def _pair_key_from_subject_row(
    row: Mapping[str, object],
    contrast: ContrastSpec,
) -> tuple[str, ...]:
    values = {
        "participant_id": _as_str(row.get("participant_id")).casefold(),
        "session_id": _as_str(row.get("session_id")).casefold(),
        "run_id": _as_str(row.get("run_ids") or row.get("run_id"), "single")
        .casefold()
        .split(";")[0],
    }
    return tuple(values[field] for field in contrast.pair_within)


def combine_endpoint_and_peak_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
    peak_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Left-join peak fits onto endpoint metrics on the shared analysis key."""
    peaks_by_key: dict[tuple[str, ...], Mapping[str, object]] = {}
    for peak in peak_rows:
        peaks_by_key[_join_key(peak)] = peak

    combined: list[dict[str, object]] = []
    for endpoint in endpoint_rows:
        keys = normalize_keys(endpoint)
        peak = peaks_by_key.get(_join_key(endpoint), {})
        duration_s = _as_int(endpoint.get("duration_s"))
        combined.append(
            {
                **keys,
                "duration_s": str(duration_s),
                "duration_role": _as_str(endpoint.get("duration_role")),
                "endpoint_name": _as_str(endpoint.get("endpoint_name")).casefold(),
                "endpoint_alias": _as_str(endpoint.get("endpoint_alias")),
                "is_standard_zlpi": _as_bool(endpoint.get("is_standard_zlpi")),
                "pool_with_standard_zlpi": _as_bool(
                    endpoint.get("pool_with_standard_zlpi")
                ),
                "band": _as_str(endpoint.get("band")).casefold(),
                "power_representation": _as_str(
                    endpoint.get("power_representation")
                ).casefold(),
                "is_primary_representation": _as_bool(
                    endpoint.get("is_primary_representation")
                ),
                "pair": _as_str(endpoint.get("pair")),
                "endpoint_eligible": _as_bool(endpoint.get("eligible")),
                "endpoint_exclusion_reason": _as_str(
                    endpoint.get("exclusion_reason")
                ),
                "endpoint_index": _as_float(endpoint.get("endpoint_index")),
                "local_prominence": _as_float(endpoint.get("local_prominence")),
                "r0": _as_float(endpoint.get("r0")),
                "z0": _as_float(endpoint.get("z0")),
                "n_common_support": _as_float(endpoint.get("n_common_support")),
                "peak_converged": _as_bool(peak.get("converged")) if peak else False,
                "has_identifiable_peak": (
                    _as_bool(peak.get("has_identifiable_peak")) if peak else False
                ),
                "report_timing_shift": (
                    _as_bool(peak.get("report_timing_shift")) if peak else False
                ),
                "baseline_C": _as_float(peak.get("baseline_C")) if peak else float("nan"),
                "peak_height_A": (
                    _as_float(peak.get("peak_height_A")) if peak else float("nan")
                ),
                "peak_center_mu_s": (
                    _as_float(peak.get("peak_center_mu_s")) if peak else float("nan")
                ),
                "sigma_s": _as_float(peak.get("sigma_s")) if peak else float("nan"),
                "fwhm_s": _as_float(peak.get("fwhm_s")) if peak else float("nan"),
                "peak_exclusion_reason": (
                    _as_str(peak.get("exclusion_reason")) if peak else "missing_peak_fit"
                ),
            }
        )
    return combined


def build_subject_level_metrics(
    endpoint_rows: Sequence[Mapping[str, object]],
    peak_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """One canonical row per dataset×participant×session×condition×duration×band×repr×endpoint."""
    combined = combine_endpoint_and_peak_rows(endpoint_rows, peak_rows)
    buckets: dict[tuple[str, ...], list[dict[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    for row in combined:
        grain = _subject_grain(row)
        if grain not in buckets:
            buckets[grain] = []
            order.append(grain)
        buckets[grain].append(row)

    subject_rows: list[dict[str, object]] = []
    for grain in order:
        members = buckets[grain]
        eligible_members = [m for m in members if m["endpoint_eligible"]]
        peak_members = [m for m in members if m["has_identifiable_peak"]]
        endpoint_source = eligible_members or members
        peak_source = peak_members or members

        first = members[0]
        subject_rows.append(
            {
                "dataset_id": first["dataset_id"],
                "participant_id": first["participant_id"],
                "session_id": first["session_id"],
                "condition": first["condition"],
                "duration_s": _as_int(first["duration_s"]),
                "duration_role": first["duration_role"],
                "endpoint_name": first["endpoint_name"],
                "endpoint_alias": first["endpoint_alias"],
                "is_standard_zlpi": first["is_standard_zlpi"],
                "pool_with_standard_zlpi": first["pool_with_standard_zlpi"],
                "band": first["band"],
                "power_representation": first["power_representation"],
                "is_primary_representation": first["is_primary_representation"],
                "pair": first["pair"],
                "n_runs": len(members),
                "run_ids": _semicolon_join(
                    [_as_str(m.get("run_id"), "single") for m in members]
                ),
                "observation_ids": _semicolon_join(
                    [_as_str(m.get("observation_id")) for m in members]
                ),
                "endpoint_eligible": all(bool(m["endpoint_eligible"]) for m in members),
                "endpoint_exclusion_reason": _semicolon_join(
                    [
                        _as_str(m.get("endpoint_exclusion_reason"))
                        for m in members
                        if _as_str(m.get("endpoint_exclusion_reason"))
                    ]
                ),
                "endpoint_index": _mean(
                    [float(m["endpoint_index"]) for m in endpoint_source]
                ),
                "local_prominence": _mean(
                    [float(m["local_prominence"]) for m in endpoint_source]
                ),
                "r0": _mean([float(m["r0"]) for m in endpoint_source]),
                "z0": _mean([float(m["z0"]) for m in endpoint_source]),
                "n_common_support": _mean(
                    [float(m["n_common_support"]) for m in endpoint_source]
                ),
                "peak_converged": all(bool(m["peak_converged"]) for m in members),
                "has_identifiable_peak": all(
                    bool(m["has_identifiable_peak"]) for m in members
                ),
                "report_timing_shift": all(
                    bool(m["report_timing_shift"]) for m in members
                ),
                "baseline_C": _mean([float(m["baseline_C"]) for m in peak_source]),
                "peak_height_A": _mean(
                    [float(m["peak_height_A"]) for m in peak_source]
                ),
                "peak_center_mu_s": _mean(
                    [float(m["peak_center_mu_s"]) for m in peak_members]
                )
                if peak_members and len(peak_members) == len(members)
                else float("nan"),
                "sigma_s": _mean([float(m["sigma_s"]) for m in peak_source]),
                "fwhm_s": _mean([float(m["fwhm_s"]) for m in peak_source]),
                "peak_exclusion_reason": _semicolon_join(
                    [
                        _as_str(m.get("peak_exclusion_reason"))
                        for m in members
                        if _as_str(m.get("peak_exclusion_reason"))
                    ]
                ),
            }
        )
    return subject_rows


def _analysis_slice_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        _as_str(row.get("duration_s")),
        _as_str(row.get("endpoint_name")).casefold(),
        _as_str(row.get("band")).casefold(),
        _as_str(row.get("power_representation")).casefold(),
    )


def _delta(effort: float, low: float) -> float:
    if not (math.isfinite(effort) and math.isfinite(low)):
        return float("nan")
    return float(effort - low)


def build_paired_contrasts(
    subject_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Task-minus-low-demand contrasts for valid within-subject pairings only."""
    by_dataset: dict[str, list[Mapping[str, object]]] = {}
    for row in subject_rows:
        by_dataset.setdefault(_as_str(row.get("dataset_id")).casefold(), []).append(row)

    contrast_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for dataset_id in sorted(set(PROTOCOL_SPECS) | set(by_dataset)):
        rows = by_dataset.get(dataset_id, [])
        try:
            spec: ProtocolSpec = protocol_spec(dataset_id)
        except KeyError:
            qc_rows.append(
                {
                    "dataset_id": dataset_id,
                    "contrast_id": "",
                    "participant_id": "",
                    "session_id": "",
                    "pair_within": "",
                    "low_demand_condition": "",
                    "cognitive_effort_condition": "",
                    "pairing_status": "unknown_dataset",
                    "n_low_keys": 0,
                    "n_effort_keys": 0,
                    "n_paired_keys": 0,
                    "n_low_runs": 0,
                    "n_effort_runs": 0,
                    "low_observation_ids": "",
                    "effort_observation_ids": "",
                    "duplicate_runs": False,
                    "notes": "Dataset has subject-level rows but no protocol spec.",
                }
            )
            continue

        if not spec.contrasts:
            qc_rows.append(
                {
                    "dataset_id": dataset_id,
                    "contrast_id": "",
                    "participant_id": "",
                    "session_id": "",
                    "pair_within": "",
                    "low_demand_condition": "",
                    "cognitive_effort_condition": "",
                    "pairing_status": "no_prespecified_contrast",
                    "n_low_keys": 0,
                    "n_effort_keys": 0,
                    "n_paired_keys": 0,
                    "n_low_runs": 0,
                    "n_effort_runs": 0,
                    "low_observation_ids": "",
                    "effort_observation_ids": "",
                    "duplicate_runs": False,
                    "notes": "; ".join(spec.unresolved_assumptions),
                }
            )
            continue

        for contrast in spec.contrasts:
            low_cond = contrast.low_demand_condition.casefold()
            effort_cond = contrast.cognitive_effort_condition.casefold()
            low_rows = [
                r for r in rows if _as_str(r.get("condition")).casefold() == low_cond
            ]
            effort_rows = [
                r
                for r in rows
                if _as_str(r.get("condition")).casefold() == effort_cond
            ]

            # Index by pair_within × analysis slice (duration/endpoint/band/repr).
            low_index: dict[
                tuple[tuple[str, ...], tuple[str, ...]], Mapping[str, object]
            ] = {}
            effort_index: dict[
                tuple[tuple[str, ...], tuple[str, ...]], Mapping[str, object]
            ] = {}
            for row in low_rows:
                key = (_pair_key_from_subject_row(row, contrast), _analysis_slice_key(row))
                low_index[key] = row
            for row in effort_rows:
                key = (_pair_key_from_subject_row(row, contrast), _analysis_slice_key(row))
                effort_index[key] = row

            # Pairing inventory at participant/session (ignore analysis slices).
            low_pair_keys = {
                _pair_key_from_subject_row(r, contrast) for r in low_rows
            }
            effort_pair_keys = {
                _pair_key_from_subject_row(r, contrast) for r in effort_rows
            }
            paired_pair_keys = sorted(low_pair_keys & effort_pair_keys)
            low_only = sorted(low_pair_keys - effort_pair_keys)
            effort_only = sorted(effort_pair_keys - low_pair_keys)

            def _key_fields(pair_key: tuple[str, ...]) -> dict[str, str]:
                payload = dict(zip(contrast.pair_within, pair_key, strict=True))
                return {
                    "participant_id": payload.get("participant_id", ""),
                    "session_id": payload.get("session_id", ""),
                }

            for pair_key in paired_pair_keys:
                # Collect runs from any slice to summarize QC.
                low_examples = [
                    r
                    for r in low_rows
                    if _pair_key_from_subject_row(r, contrast) == pair_key
                ]
                effort_examples = [
                    r
                    for r in effort_rows
                    if _pair_key_from_subject_row(r, contrast) == pair_key
                ]
                n_low_runs = max((_as_int(r.get("n_runs"), 1) for r in low_examples), default=0)
                n_effort_runs = max(
                    (_as_int(r.get("n_runs"), 1) for r in effort_examples), default=0
                )
                duplicate = n_low_runs > 1 or n_effort_runs > 1
                fields = _key_fields(pair_key)
                qc_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "contrast_id": contrast.contrast_id,
                        "participant_id": fields["participant_id"],
                        "session_id": fields["session_id"],
                        "pair_within": ";".join(contrast.pair_within),
                        "low_demand_condition": contrast.low_demand_condition,
                        "cognitive_effort_condition": contrast.cognitive_effort_condition,
                        "pairing_status": "paired",
                        "n_low_keys": len(low_pair_keys),
                        "n_effort_keys": len(effort_pair_keys),
                        "n_paired_keys": len(paired_pair_keys),
                        "n_low_runs": n_low_runs,
                        "n_effort_runs": n_effort_runs,
                        "low_observation_ids": _semicolon_join(
                            [
                                oid
                                for r in low_examples
                                for oid in _as_str(r.get("observation_ids")).split(";")
                                if oid
                            ]
                        ),
                        "effort_observation_ids": _semicolon_join(
                            [
                                oid
                                for r in effort_examples
                                for oid in _as_str(r.get("observation_ids")).split(";")
                                if oid
                            ]
                        ),
                        "duplicate_runs": duplicate,
                        "notes": (
                            "Duplicate runs aggregated within pairing key."
                            if duplicate
                            else ""
                        ),
                    }
                )

            for pair_key in low_only:
                fields = _key_fields(pair_key)
                examples = [
                    r
                    for r in low_rows
                    if _pair_key_from_subject_row(r, contrast) == pair_key
                ]
                qc_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "contrast_id": contrast.contrast_id,
                        "participant_id": fields["participant_id"],
                        "session_id": fields["session_id"],
                        "pair_within": ";".join(contrast.pair_within),
                        "low_demand_condition": contrast.low_demand_condition,
                        "cognitive_effort_condition": contrast.cognitive_effort_condition,
                        "pairing_status": "missing_cognitive_effort",
                        "n_low_keys": len(low_pair_keys),
                        "n_effort_keys": len(effort_pair_keys),
                        "n_paired_keys": len(paired_pair_keys),
                        "n_low_runs": max(
                            (_as_int(r.get("n_runs"), 1) for r in examples), default=0
                        ),
                        "n_effort_runs": 0,
                        "low_observation_ids": _semicolon_join(
                            [
                                oid
                                for r in examples
                                for oid in _as_str(r.get("observation_ids")).split(";")
                                if oid
                            ]
                        ),
                        "effort_observation_ids": "",
                        "duplicate_runs": False,
                        "notes": "Low-demand observation lacks matching effort condition.",
                    }
                )

            for pair_key in effort_only:
                fields = _key_fields(pair_key)
                examples = [
                    r
                    for r in effort_rows
                    if _pair_key_from_subject_row(r, contrast) == pair_key
                ]
                qc_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "contrast_id": contrast.contrast_id,
                        "participant_id": fields["participant_id"],
                        "session_id": fields["session_id"],
                        "pair_within": ";".join(contrast.pair_within),
                        "low_demand_condition": contrast.low_demand_condition,
                        "cognitive_effort_condition": contrast.cognitive_effort_condition,
                        "pairing_status": "missing_low_demand",
                        "n_low_keys": len(low_pair_keys),
                        "n_effort_keys": len(effort_pair_keys),
                        "n_paired_keys": len(paired_pair_keys),
                        "n_low_runs": 0,
                        "n_effort_runs": max(
                            (_as_int(r.get("n_runs"), 1) for r in examples), default=0
                        ),
                        "low_observation_ids": "",
                        "effort_observation_ids": _semicolon_join(
                            [
                                oid
                                for r in examples
                                for oid in _as_str(r.get("observation_ids")).split(";")
                                if oid
                            ]
                        ),
                        "duplicate_runs": False,
                        "notes": "Effort observation lacks matching low-demand condition.",
                    }
                )

            if not low_pair_keys and not effort_pair_keys:
                qc_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "contrast_id": contrast.contrast_id,
                        "participant_id": "",
                        "session_id": "",
                        "pair_within": ";".join(contrast.pair_within),
                        "low_demand_condition": contrast.low_demand_condition,
                        "cognitive_effort_condition": contrast.cognitive_effort_condition,
                        "pairing_status": "no_observations",
                        "n_low_keys": 0,
                        "n_effort_keys": 0,
                        "n_paired_keys": 0,
                        "n_low_runs": 0,
                        "n_effort_runs": 0,
                        "low_observation_ids": "",
                        "effort_observation_ids": "",
                        "duplicate_runs": False,
                        "notes": "No subject-level rows for this contrast.",
                    }
                )

            # Emit contrast rows only for exact intersections at analysis grain.
            for key in sorted(set(low_index) & set(effort_index)):
                pair_key, _slice = key
                low = low_index[key]
                effort = effort_index[key]
                # Never mix endpoint families: already encoded in analysis slice.
                assert _as_str(low.get("endpoint_name")).casefold() == _as_str(
                    effort.get("endpoint_name")
                ).casefold()
                fields = _key_fields(pair_key)
                mu_eligible = bool(low["has_identifiable_peak"]) and bool(
                    effort["has_identifiable_peak"]
                )
                contrast_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "contrast_id": contrast.contrast_id,
                        "participant_id": fields["participant_id"],
                        "session_id": fields["session_id"],
                        "pair_within": ";".join(contrast.pair_within),
                        "low_demand_condition": contrast.low_demand_condition,
                        "cognitive_effort_condition": contrast.cognitive_effort_condition,
                        "duration_s": _as_int(low.get("duration_s")),
                        "duration_role": _as_str(low.get("duration_role")),
                        "endpoint_name": _as_str(low.get("endpoint_name")).casefold(),
                        "is_standard_zlpi": _as_bool(low.get("is_standard_zlpi")),
                        "band": _as_str(low.get("band")).casefold(),
                        "power_representation": _as_str(
                            low.get("power_representation")
                        ).casefold(),
                        "is_primary_representation": _as_bool(
                            low.get("is_primary_representation")
                        ),
                        "n_low_runs": _as_int(low.get("n_runs"), 1),
                        "n_effort_runs": _as_int(effort.get("n_runs"), 1),
                        "low_observation_ids": _as_str(low.get("observation_ids")),
                        "effort_observation_ids": _as_str(
                            effort.get("observation_ids")
                        ),
                        "delta_endpoint_index": _delta(
                            float(effort["endpoint_index"]),
                            float(low["endpoint_index"]),
                        ),
                        "delta_local_prominence": _delta(
                            float(effort["local_prominence"]),
                            float(low["local_prominence"]),
                        ),
                        "delta_peak_height_A": _delta(
                            float(effort["peak_height_A"]),
                            float(low["peak_height_A"]),
                        ),
                        "delta_peak_center_mu_s": (
                            _delta(
                                float(effort["peak_center_mu_s"]),
                                float(low["peak_center_mu_s"]),
                            )
                            if mu_eligible
                            else float("nan")
                        ),
                        "delta_fwhm_s": _delta(
                            float(effort["fwhm_s"]), float(low["fwhm_s"])
                        ),
                        "mu_contrast_eligible": mu_eligible,
                        "low_endpoint_eligible": bool(low["endpoint_eligible"]),
                        "effort_endpoint_eligible": bool(effort["endpoint_eligible"]),
                        "low_has_identifiable_peak": bool(
                            low["has_identifiable_peak"]
                        ),
                        "effort_has_identifiable_peak": bool(
                            effort["has_identifiable_peak"]
                        ),
                        "low_endpoint_index": float(low["endpoint_index"]),
                        "effort_endpoint_index": float(effort["endpoint_index"]),
                        "low_local_prominence": float(low["local_prominence"]),
                        "effort_local_prominence": float(effort["local_prominence"]),
                        "low_peak_height_A": float(low["peak_height_A"]),
                        "effort_peak_height_A": float(effort["peak_height_A"]),
                        "low_peak_center_mu_s": float(low["peak_center_mu_s"])
                        if mu_eligible
                        else float("nan"),
                        "effort_peak_center_mu_s": float(effort["peak_center_mu_s"])
                        if mu_eligible
                        else float("nan"),
                        "low_fwhm_s": float(low["fwhm_s"]),
                        "effort_fwhm_s": float(effort["fwhm_s"]),
                    }
                )

    return contrast_rows, qc_rows


def build_group_tables(
    endpoint_rows: Sequence[Mapping[str, object]],
    peak_rows: Sequence[Mapping[str, object]],
) -> GroupTableResult:
    subject_rows = build_subject_level_metrics(endpoint_rows, peak_rows)
    contrast_rows, qc_rows = build_paired_contrasts(subject_rows)
    return GroupTableResult(
        subject_level_rows=tuple(subject_rows),
        paired_contrast_rows=tuple(contrast_rows),
        pairing_qc_rows=tuple(qc_rows),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                elif isinstance(value, bool):
                    payload[field] = str(value)
                else:
                    payload[field] = value
            writer.writerow(payload)


def write_group_table_outputs(
    result: GroupTableResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    subject_path = output_path / SUBJECT_LEVEL_FILENAME
    contrast_path = output_path / PAIRED_CONTRASTS_FILENAME
    qc_path = output_path / PAIRING_QC_FILENAME
    _write_csv(subject_path, result.subject_level_rows, SUBJECT_LEVEL_FIELDS)
    _write_csv(contrast_path, result.paired_contrast_rows, PAIRED_CONTRAST_FIELDS)
    _write_csv(qc_path, result.pairing_qc_rows, PAIRING_QC_FIELDS)
    return {
        "subject_level_metrics": subject_path,
        "paired_contrasts": contrast_path,
        "pairing_qc": qc_path,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_endpoint_metric_tables(
    input_dir: str | Path,
    *,
    durations: Sequence[int] = (240, 180, 120, 60),
) -> dict[int, Path]:
    root = Path(input_dir).expanduser().resolve()
    found: dict[int, Path] = {}
    for duration in durations:
        path = root / METRICS_TEMPLATE.format(duration_s=duration)
        if path.is_file():
            found[int(duration)] = path
    return found


def load_endpoint_and_peak_rows(
    endpoints_dir: str | Path,
    peaks_dir: str | Path | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    endpoint_root = Path(endpoints_dir).expanduser().resolve()
    peak_root = (
        Path(peaks_dir).expanduser().resolve()
        if peaks_dir is not None
        else endpoint_root
    )
    endpoint_rows: list[dict[str, str]] = []
    for path in discover_endpoint_metric_tables(endpoint_root).values():
        endpoint_rows.extend(read_csv_rows(path))
    peak_path = peak_root / PARAMS_FILENAME
    peak_rows = read_csv_rows(peak_path) if peak_path.is_file() else []
    return endpoint_rows, peak_rows


def run_confirmatory_group_tables(
    endpoints_dir: str | Path,
    output_dir: str | Path,
    *,
    peaks_dir: str | Path | None = None,
) -> GroupTableResult:
    endpoint_rows, peak_rows = load_endpoint_and_peak_rows(
        endpoints_dir, peaks_dir=peaks_dir
    )
    result = build_group_tables(endpoint_rows, peak_rows)
    write_group_table_outputs(result, output_dir)
    return result


__all__ = [
    "PAIRING_QC_FILENAME",
    "PAIRING_QC_FIELDS",
    "PAIRED_CONTRASTS_FILENAME",
    "PAIRED_CONTRAST_FIELDS",
    "SUBJECT_LEVEL_FILENAME",
    "SUBJECT_LEVEL_FIELDS",
    "GroupTableResult",
    "build_group_tables",
    "build_paired_contrasts",
    "build_subject_level_metrics",
    "combine_endpoint_and_peak_rows",
    "infer_session_label",
    "normalize_keys",
    "run_confirmatory_group_tables",
    "write_group_table_outputs",
]
