"""Fisher-z confirmatory endpoints (ZLPI, MWPI, SWPI) and local prominence.

Table identity
--------------
Each metrics/QC row carries both ``endpoint_name`` (canonical ID used for
joins/filters) and ``endpoint_alias`` (short display acronym). See
``DurationAnalysisContract`` for the distinction.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .correlation import CURVES_TEMPLATE, IDENTITY_FIELDS, identity_from_row
from .duration_contracts import (
    DurationAnalysisContract,
    EXPECTED_DURATIONS_S,
    contract_for_duration,
)

FISHER_R_CLIP = 0.999999
# Plan floor for standard/mid-window endpoints; D60 uses its full common-support length.
MIN_COMMON_SUPPORT_STANDARD = 60

METRICS_TEMPLATE = "confirmatory_endpoint_metrics_D{duration_s}.csv"
QC_TEMPLATE = "confirmatory_endpoint_qc_D{duration_s}.csv"

METRICS_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "lag_analysis_role",
    "endpoint_name",
    "endpoint_alias",
    "is_standard_zlpi",
    "pool_with_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "eligible",
    "exclusion_reason",
    "r0",
    "z0",
    "endpoint_index",
    "local_prominence",
    "negative_flank_mean_z",
    "positive_flank_mean_z",
    "combined_flank_mean_z",
    "negative_shoulder_mean_z",
    "positive_shoulder_mean_z",
    "flank_inner_s",
    "flank_outer_s",
    "shoulders_inner_s",
    "shoulders_outer_s",
    "n_common_support",
    "n_negative_flank_lags",
    "n_positive_flank_lags",
    "n_combined_flank_lags",
    "n_negative_shoulder_lags",
    "n_positive_shoulder_lags",
)

QC_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "lag_analysis_role",
    "endpoint_name",
    "endpoint_alias",
    "is_standard_zlpi",
    "pool_with_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "eligible",
    "exclusion_reason",
    "n_lags_observed",
    "n_lags_expected",
    "lag_grid_complete",
    "required_windows_complete",
    "overlap_is_constant",
    "n_common_support",
    "min_common_support_required",
    "r0_finite",
    "all_required_r_finite",
)


@dataclass(frozen=True)
class EndpointResult:
    duration_s: int
    metrics_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def fisher_z(r: float, *, clip: float = FISHER_R_CLIP) -> float:
    """Fisher z-transform with confirmatory clipping at ±clip before atanh."""
    if not math.isfinite(r):
        return float("nan")
    if not (0.0 < clip < 1.0):
        raise ValueError("clip must be in (0, 1).")
    clipped = float(np.clip(float(r), -clip, clip))
    return float(np.arctanh(clipped))


def min_common_support_required(contract: DurationAnalysisContract) -> int:
    """Minimum constant ``n_overlap`` required for endpoint eligibility."""
    fully_finite = contract.expected_constant_overlap_if_fully_finite
    if contract.is_standard_zlpi or contract.duration_s == 120:
        return min(MIN_COMMON_SUPPORT_STANDARD, fully_finite)
    return fully_finite


def negative_flank_lags(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return tuple(range(-contract.flank_outer_s, -contract.flank_inner_s + 1))


def positive_flank_lags(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return tuple(range(contract.flank_inner_s, contract.flank_outer_s + 1))


def combined_flank_lags(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return negative_flank_lags(contract) + positive_flank_lags(contract)


def negative_shoulder_lags(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return tuple(range(-contract.shoulders_outer_s, -contract.shoulders_inner_s + 1))


def positive_shoulder_lags(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return tuple(range(contract.shoulders_inner_s, contract.shoulders_outer_s + 1))


def expected_lag_grid(contract: DurationAnalysisContract) -> tuple[int, ...]:
    return tuple(range(contract.lag_min_s, contract.lag_max_s + 1, contract.lag_step_s))


def _as_float(value: object) -> float:
    return float(value)


def _mean_z(lag_to_r: Mapping[int, float], lags: Sequence[int]) -> float:
    values = [fisher_z(lag_to_r[int(lag)]) for lag in lags]
    if not values or any(not math.isfinite(value) for value in values):
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)))


def _identity_from_row(row: Mapping[str, object]) -> dict[str, str]:
    return identity_from_row(row)


def _group_curve_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, str], list[Mapping[str, object]]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    meta: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        identity = _identity_from_row(row)
        band = str(row.get("band", "")).strip().casefold()
        representation = str(row.get("power_representation", "")).strip()
        key = tuple(identity[field] for field in IDENTITY_FIELDS) + (
            band,
            representation,
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
            meta[key] = identity
        grouped[key].append(row)
    return [(meta[key], grouped[key]) for key in order]


def _empty_rows(
    identity: Mapping[str, str],
    contract: DurationAnalysisContract,
    *,
    duration_role: str,
    band: str,
    representation: str,
    pair: str,
    is_primary: bool,
    exclusion_reason: str,
) -> tuple[dict[str, object], dict[str, object]]:
    metrics = {
        **identity,
        "duration_s": contract.duration_s,
        "duration_role": duration_role,
        "lag_analysis_role": contract.lag_analysis_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": contract.is_standard_zlpi,
        "pool_with_standard_zlpi": contract.pool_with_standard_zlpi,
        "band": band,
        "power_representation": representation,
        "is_primary_representation": is_primary,
        "pair": pair,
        "eligible": False,
        "exclusion_reason": exclusion_reason,
        "r0": float("nan"),
        "z0": float("nan"),
        "endpoint_index": float("nan"),
        "local_prominence": float("nan"),
        "negative_flank_mean_z": float("nan"),
        "positive_flank_mean_z": float("nan"),
        "combined_flank_mean_z": float("nan"),
        "negative_shoulder_mean_z": float("nan"),
        "positive_shoulder_mean_z": float("nan"),
        "flank_inner_s": contract.flank_inner_s,
        "flank_outer_s": contract.flank_outer_s,
        "shoulders_inner_s": contract.shoulders_inner_s,
        "shoulders_outer_s": contract.shoulders_outer_s,
        "n_common_support": 0,
        "n_negative_flank_lags": len(negative_flank_lags(contract)),
        "n_positive_flank_lags": len(positive_flank_lags(contract)),
        "n_combined_flank_lags": len(combined_flank_lags(contract)),
        "n_negative_shoulder_lags": len(negative_shoulder_lags(contract)),
        "n_positive_shoulder_lags": len(positive_shoulder_lags(contract)),
    }
    qc = {
        **identity,
        "duration_s": contract.duration_s,
        "duration_role": duration_role,
        "lag_analysis_role": contract.lag_analysis_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": contract.is_standard_zlpi,
        "pool_with_standard_zlpi": contract.pool_with_standard_zlpi,
        "band": band,
        "power_representation": representation,
        "is_primary_representation": is_primary,
        "pair": pair,
        "eligible": False,
        "exclusion_reason": exclusion_reason,
        "n_lags_observed": 0,
        "n_lags_expected": contract.n_lags,
        "lag_grid_complete": False,
        "required_windows_complete": False,
        "overlap_is_constant": False,
        "n_common_support": 0,
        "min_common_support_required": min_common_support_required(contract),
        "r0_finite": False,
        "all_required_r_finite": False,
    }
    return metrics, qc


def evaluate_endpoint_curve(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one observation × band × representation lag curve."""
    contract = contract_for_duration(duration_s)
    if not curve_rows:
        return _empty_rows(
            {field: "" for field in IDENTITY_FIELDS},
            contract,
            duration_role="",
            band="",
            representation="",
            pair="",
            is_primary=False,
            exclusion_reason="no_curve_rows",
        )

    first = curve_rows[0]
    identity = _identity_from_row(first)
    duration_role = str(first.get("duration_role", "")).strip()
    band = str(first.get("band", "")).strip().casefold()
    representation = str(first.get("power_representation", "")).strip()
    pair = str(first.get("pair", "")).strip()
    is_primary = bool(first.get("is_primary_representation", False))

    lag_to_r: dict[int, float] = {}
    lag_to_overlap: dict[int, int] = {}
    for row in curve_rows:
        lag = int(round(_as_float(row["lag_s"])))
        lag_to_r[lag] = _as_float(row["r"])
        lag_to_overlap[lag] = int(round(_as_float(row.get("n_overlap", 0))))

    expected = expected_lag_grid(contract)
    observed = tuple(sorted(lag_to_r))
    lag_grid_complete = observed == expected
    overlaps = [lag_to_overlap[lag] for lag in observed if lag in lag_to_overlap]
    overlap_is_constant = len(set(overlaps)) <= 1 if overlaps else False
    if overlaps and overlap_is_constant:
        n_common_support = int(overlaps[0])
    elif overlaps:
        n_common_support = int(min(overlaps))
    else:
        n_common_support = 0
    min_support = min_common_support_required(contract)

    neg_flank = negative_flank_lags(contract)
    pos_flank = positive_flank_lags(contract)
    combined = combined_flank_lags(contract)
    neg_shoulder = negative_shoulder_lags(contract)
    pos_shoulder = positive_shoulder_lags(contract)
    required_unique = tuple(
        dict.fromkeys((0,) + combined + neg_shoulder + pos_shoulder)
    )

    required_windows_complete = all(lag in lag_to_r for lag in required_unique)
    all_required_r_finite = required_windows_complete and all(
        math.isfinite(lag_to_r[lag]) for lag in required_unique
    )
    r0 = lag_to_r.get(0, float("nan"))
    r0_finite = math.isfinite(r0)

    exclusion_reason = ""
    if not lag_grid_complete:
        exclusion_reason = "incomplete_lag_grid"
    elif not overlap_is_constant:
        exclusion_reason = "nonconstant_overlap"
    elif n_common_support < min_support:
        exclusion_reason = "insufficient_common_support"
    elif not required_windows_complete:
        exclusion_reason = "incomplete_endpoint_windows"
    elif not all_required_r_finite:
        exclusion_reason = "nonfinite_required_correlations"

    eligible = exclusion_reason == ""
    z0 = fisher_z(r0) if r0_finite else float("nan")
    if eligible:
        neg_flank_z = _mean_z(lag_to_r, neg_flank)
        pos_flank_z = _mean_z(lag_to_r, pos_flank)
        combined_z = _mean_z(lag_to_r, combined)
        neg_shoulder_z = _mean_z(lag_to_r, neg_shoulder)
        pos_shoulder_z = _mean_z(lag_to_r, pos_shoulder)
        endpoint_index = float(z0 - combined_z)
        local_prominence = float(z0 - max(neg_shoulder_z, pos_shoulder_z))
    else:
        neg_flank_z = pos_flank_z = combined_z = float("nan")
        neg_shoulder_z = pos_shoulder_z = float("nan")
        endpoint_index = local_prominence = float("nan")

    # Never pool or flag MWPI/SWPI as standard ZLPI.
    is_standard_zlpi = bool(contract.is_standard_zlpi)
    pool_with_standard_zlpi = bool(contract.pool_with_standard_zlpi and is_standard_zlpi)

    metrics = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": duration_role,
        "lag_analysis_role": contract.lag_analysis_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": is_standard_zlpi,
        "pool_with_standard_zlpi": pool_with_standard_zlpi,
        "band": band,
        "power_representation": representation,
        "is_primary_representation": is_primary,
        "pair": pair,
        "eligible": eligible,
        "exclusion_reason": exclusion_reason,
        "r0": float(r0),
        "z0": float(z0),
        "endpoint_index": endpoint_index,
        "local_prominence": local_prominence,
        "negative_flank_mean_z": neg_flank_z,
        "positive_flank_mean_z": pos_flank_z,
        "combined_flank_mean_z": combined_z,
        "negative_shoulder_mean_z": neg_shoulder_z,
        "positive_shoulder_mean_z": pos_shoulder_z,
        "flank_inner_s": contract.flank_inner_s,
        "flank_outer_s": contract.flank_outer_s,
        "shoulders_inner_s": contract.shoulders_inner_s,
        "shoulders_outer_s": contract.shoulders_outer_s,
        "n_common_support": n_common_support,
        "n_negative_flank_lags": len(neg_flank),
        "n_positive_flank_lags": len(pos_flank),
        "n_combined_flank_lags": len(combined),
        "n_negative_shoulder_lags": len(neg_shoulder),
        "n_positive_shoulder_lags": len(pos_shoulder),
    }
    qc = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": duration_role,
        "lag_analysis_role": contract.lag_analysis_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": is_standard_zlpi,
        "pool_with_standard_zlpi": pool_with_standard_zlpi,
        "band": band,
        "power_representation": representation,
        "is_primary_representation": is_primary,
        "pair": pair,
        "eligible": eligible,
        "exclusion_reason": exclusion_reason,
        "n_lags_observed": len(observed),
        "n_lags_expected": len(expected),
        "lag_grid_complete": lag_grid_complete,
        "required_windows_complete": required_windows_complete,
        "overlap_is_constant": overlap_is_constant,
        "n_common_support": n_common_support,
        "min_common_support_required": min_support,
        "r0_finite": r0_finite,
        "all_required_r_finite": all_required_r_finite,
    }
    return metrics, qc


def compute_endpoints_from_curves(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
) -> EndpointResult:
    """Compute duration-specific named endpoints for M5 curve rows."""
    if duration_s not in EXPECTED_DURATIONS_S:
        raise ValueError(f"Unsupported duration_s={duration_s}.")
    metrics_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for _identity, rows in _group_curve_rows(curve_rows):
        metrics, qc = evaluate_endpoint_curve(rows, duration_s=duration_s)
        metrics_rows.append(metrics)
        qc_rows.append(qc)
    return EndpointResult(
        duration_s=int(duration_s),
        metrics_rows=tuple(metrics_rows),
        qc_rows=tuple(qc_rows),
    )


def read_curve_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_endpoints_from_csv(
    curves_csv: str | Path,
    *,
    duration_s: int,
) -> EndpointResult:
    return compute_endpoints_from_curves(
        read_curve_csv(curves_csv),
        duration_s=duration_s,
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
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_endpoint_outputs(
    result: EndpointResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / METRICS_TEMPLATE.format(duration_s=result.duration_s)
    qc_path = output_path / QC_TEMPLATE.format(duration_s=result.duration_s)
    _write_csv(metrics_path, result.metrics_rows, METRICS_FIELDS)
    _write_csv(qc_path, result.qc_rows, QC_FIELDS)
    return {"metrics": metrics_path, "qc": qc_path}


def discover_curve_tables(
    input_dir: str | Path,
    *,
    durations: Sequence[int] = EXPECTED_DURATIONS_S,
) -> dict[int, Path]:
    root = Path(input_dir).expanduser().resolve()
    found: dict[int, Path] = {}
    for duration in durations:
        path = root / CURVES_TEMPLATE.format(duration_s=duration)
        if path.is_file():
            found[int(duration)] = path
    return found


def run_confirmatory_endpoints(
    curves_dir: str | Path,
    output_dir: str | Path,
    *,
    durations: Sequence[int] | None = None,
) -> dict[int, EndpointResult]:
    tables = discover_curve_tables(
        curves_dir,
        durations=tuple(durations) if durations is not None else EXPECTED_DURATIONS_S,
    )
    results: dict[int, EndpointResult] = {}
    for duration, path in sorted(tables.items()):
        result = compute_endpoints_from_csv(path, duration_s=duration)
        write_endpoint_outputs(result, output_dir)
        results[duration] = result
    return results


__all__ = [
    "FISHER_R_CLIP",
    "METRICS_TEMPLATE",
    "MIN_COMMON_SUPPORT_STANDARD",
    "QC_TEMPLATE",
    "EndpointResult",
    "combined_flank_lags",
    "compute_endpoints_from_csv",
    "compute_endpoints_from_curves",
    "discover_curve_tables",
    "evaluate_endpoint_curve",
    "expected_lag_grid",
    "fisher_z",
    "min_common_support_required",
    "negative_flank_lags",
    "negative_shoulder_lags",
    "positive_flank_lags",
    "positive_shoulder_lags",
    "read_curve_csv",
    "run_confirmatory_endpoints",
    "write_endpoint_outputs",
]
