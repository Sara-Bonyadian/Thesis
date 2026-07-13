"""Signed instantaneous-HR × EEG-power lag correlations for confirmatory analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ppg_eeg.temporal_coupling.cross_correlation import (
    build_lag_grid,
    compute_correlation_curve,
)

from .config import (
    EXPECTED_BANDS_HZ,
    EXPECTED_DURATIONS_S,
    EXPECTED_LAG_RANGE_S,
    EXPECTED_LAG_STEP_S,
)
from .harmonize import ALIGNED_FEATURES_TEMPLATE

FS_HZ = 1.0
DEFAULT_LAG_MIN_S = EXPECTED_LAG_RANGE_S[0]
DEFAULT_LAG_MAX_S = EXPECTED_LAG_RANGE_S[1]
DEFAULT_LAG_STEP_S = EXPECTED_LAG_STEP_S
DEFAULT_MIN_OVERLAP = 2  # Pearson needs ≥2; ZLPI overlap gates are applied in M6
BAND_ORDER = tuple(EXPECTED_BANDS_HZ)

# (representation_id, aligned-column suffix after "{band}_", is_primary)
POWER_REPRESENTATIONS: tuple[tuple[str, str, bool], ...] = (
    ("absolute_log10", "absolute_log10_power_z", True),
    ("relative", "relative_power_z", False),
    ("broadband_residualized", "broadband_residualized_log10_z", False),
)
PRIMARY_POWER_REPRESENTATION = "absolute_log10"

CURVES_TEMPLATE = "confirmatory_cross_correlation_curves_D{duration_s}.csv"
QC_TEMPLATE = "confirmatory_cross_correlation_qc_D{duration_s}.csv"

IDENTITY_FIELDS = (
    "dataset_id",
    "subject_id",
    "task",
    "condition",
    "observation_id",
)

CURVE_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "lag_s",
    "r",
    "n_overlap",
)

QC_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "n_samples",
    "n_lags",
    "lag_min_s",
    "lag_max_s",
    "lag_step_s",
    "n_finite_r",
    "n_nonfinite_r",
    "min_n_overlap",
    "max_n_overlap",
    "n_overlap_at_zero",
    "r_at_zero",
    "exclusion_reason",
)


@dataclass(frozen=True)
class ConfirmatoryCorrelationResult:
    duration_s: int
    lag_grid_s: tuple[float, ...]
    curve_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def expected_lag_count(
    *,
    lag_min_s: float = DEFAULT_LAG_MIN_S,
    lag_max_s: float = DEFAULT_LAG_MAX_S,
    lag_step_s: float = DEFAULT_LAG_STEP_S,
) -> int:
    if lag_step_s <= 0:
        raise ValueError("lag_step_s must be > 0.")
    if lag_max_s < lag_min_s:
        raise ValueError("lag_max_s must be >= lag_min_s.")
    if abs(lag_min_s + lag_max_s) > 1e-12:
        raise ValueError("Confirmatory lag grids must be symmetric about zero.")
    return int(round(lag_max_s / lag_step_s)) * 2 + 1


def _pair_name(band: str, representation: str) -> str:
    return f"hr_x_{band}_{representation}"


def _eeg_column(band: str, suffix: str) -> str:
    return f"{band}_{suffix}"


def _as_float_series(values: Sequence[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def _identity_from_row(row: Mapping[str, object]) -> dict[str, str]:
    return {field: str(row.get(field, "")).strip() for field in IDENTITY_FIELDS}


def _group_aligned_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, str], str, list[Mapping[str, object]]]]:
    """Group by observation identity; preserve first-seen observation order."""
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    meta: dict[tuple[str, ...], tuple[dict[str, str], str]] = {}
    for row in rows:
        identity = _identity_from_row(row)
        key = tuple(identity[field] for field in IDENTITY_FIELDS)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
            meta[key] = (identity, str(row.get("duration_role", "")).strip())
        grouped[key].append(row)
    return [(meta[key][0], meta[key][1], grouped[key]) for key in order]


def read_aligned_features_csv(path: str | Path) -> list[dict[str, str]]:
    """Read an M4 aligned duration table."""
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_signed_lag_curves(
    aligned_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
    lag_max_s: float = DEFAULT_LAG_MAX_S,
    lag_step_s: float = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    bands: Sequence[str] = BAND_ORDER,
    representations: Sequence[tuple[str, str, bool]] = POWER_REPRESENTATIONS,
) -> ConfirmatoryCorrelationResult:
    """Compute signed HR_z × EEG-power_z lag curves for every band/representation.

    Convention (inherited from Stage 2 ``correlate_at_lag``):
    correlates ``hr(t)`` with ``eeg(t + lag_s)``.
    Positive lag ⇒ EEG follows HR; negative lag ⇒ EEG precedes HR.
    """
    if duration_s not in EXPECTED_DURATIONS_S:
        raise ValueError(f"Unsupported duration_s={duration_s}.")
    if lag_max_s < 0:
        raise ValueError("lag_max_s must be >= 0.")
    if lag_step_s <= 0:
        raise ValueError("lag_step_s must be > 0.")
    if min_overlap < 2:
        raise ValueError("min_overlap must be >= 2.")

    lag_grid = build_lag_grid(lag_max_s=float(lag_max_s), lag_step_s=float(lag_step_s))
    lag_grid_t = tuple(float(value) for value in lag_grid)
    n_lags = int(lag_grid.size)
    unknown_bands = [band for band in bands if band not in EXPECTED_BANDS_HZ]
    if unknown_bands:
        raise ValueError(f"Unknown EEG bands: {unknown_bands}.")

    curve_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for identity, duration_role, obs_rows in _group_aligned_rows(aligned_rows):
        ordered = sorted(obs_rows, key=lambda row: float(row["time_s"]))
        if not ordered:
            continue
        times = _as_float_series([row["time_s"] for row in ordered])
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(
                "Aligned times must be strictly increasing for "
                f"observation_id={identity['observation_id']!r}."
            )
        hr_z = _as_float_series([row["hr_z"] for row in ordered])
        n_samples = int(hr_z.size)
        missing = [
            _eeg_column(band, suffix)
            for band in bands
            for _, suffix, _ in representations
            if _eeg_column(band, suffix) not in ordered[0]
        ]
        if missing:
            raise KeyError(
                "Aligned features missing required columns: "
                f"{sorted(set(missing))}."
            )

        for band in bands:
            for representation, suffix, is_primary in representations:
                column = _eeg_column(band, suffix)
                eeg_z = _as_float_series([row[column] for row in ordered])
                pair = _pair_name(band, representation)
                exclusion_reason = ""
                if n_samples == 0:
                    exclusion_reason = "no_aligned_samples"
                    curve = []
                else:
                    curve = compute_correlation_curve(
                        hr_z,
                        eeg_z,
                        lag_grid_s=lag_grid,
                        fs_hz=FS_HZ,
                        min_overlap=min_overlap,
                    )

                finite_rs = [float(point.r) for point in curve if np.isfinite(point.r)]
                overlaps = [int(point.n_overlap) for point in curve]
                zero_point = next(
                    (point for point in curve if float(point.lag_s) == 0.0),
                    None,
                )
                for point in curve:
                    curve_rows.append(
                        {
                            **identity,
                            "duration_s": int(duration_s),
                            "duration_role": duration_role,
                            "band": band,
                            "power_representation": representation,
                            "is_primary_representation": bool(is_primary),
                            "pair": pair,
                            "lag_s": float(point.lag_s),
                            "r": float(point.r),
                            "n_overlap": int(point.n_overlap),
                        }
                    )

                qc_rows.append(
                    {
                        **identity,
                        "duration_s": int(duration_s),
                        "duration_role": duration_role,
                        "band": band,
                        "power_representation": representation,
                        "is_primary_representation": bool(is_primary),
                        "pair": pair,
                        "n_samples": n_samples,
                        "n_lags": n_lags,
                        "lag_min_s": float(lag_grid_t[0]) if lag_grid_t else float("nan"),
                        "lag_max_s": float(lag_grid_t[-1]) if lag_grid_t else float("nan"),
                        "lag_step_s": float(lag_step_s),
                        "n_finite_r": len(finite_rs),
                        "n_nonfinite_r": n_lags - len(finite_rs),
                        "min_n_overlap": int(min(overlaps)) if overlaps else 0,
                        "max_n_overlap": int(max(overlaps)) if overlaps else 0,
                        "n_overlap_at_zero": (
                            int(zero_point.n_overlap) if zero_point is not None else 0
                        ),
                        "r_at_zero": (
                            float(zero_point.r) if zero_point is not None else float("nan")
                        ),
                        "exclusion_reason": exclusion_reason,
                    }
                )

    return ConfirmatoryCorrelationResult(
        duration_s=int(duration_s),
        lag_grid_s=lag_grid_t,
        curve_rows=tuple(curve_rows),
        qc_rows=tuple(qc_rows),
    )


def compute_signed_lag_curves_from_csv(
    aligned_csv: str | Path,
    *,
    duration_s: int,
    lag_max_s: float = DEFAULT_LAG_MAX_S,
    lag_step_s: float = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> ConfirmatoryCorrelationResult:
    """Load an M4 aligned duration CSV and compute confirmatory lag curves."""
    return compute_signed_lag_curves(
        read_aligned_features_csv(aligned_csv),
        duration_s=duration_s,
        lag_max_s=lag_max_s,
        lag_step_s=lag_step_s,
        min_overlap=min_overlap,
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


def write_correlation_outputs(
    result: ConfirmatoryCorrelationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write per-duration lag curves and lag-correlation QC tables."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    curves_path = output_path / CURVES_TEMPLATE.format(duration_s=result.duration_s)
    qc_path = output_path / QC_TEMPLATE.format(duration_s=result.duration_s)
    _write_csv(curves_path, result.curve_rows, CURVE_FIELDS)
    _write_csv(qc_path, result.qc_rows, QC_FIELDS)
    return {"curves": curves_path, "qc": qc_path}


def discover_aligned_duration_tables(
    input_dir: str | Path,
    *,
    durations: Sequence[int] = EXPECTED_DURATIONS_S,
) -> dict[int, Path]:
    """Map duration → existing M4 aligned feature table under ``input_dir``."""
    root = Path(input_dir).expanduser().resolve()
    found: dict[int, Path] = {}
    for duration in durations:
        path = root / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
        if path.is_file():
            found[int(duration)] = path
    return found


def run_confirmatory_correlations(
    aligned_dir: str | Path,
    output_dir: str | Path,
    *,
    durations: Sequence[int] | None = None,
    lag_max_s: float = DEFAULT_LAG_MAX_S,
    lag_step_s: float = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> dict[int, ConfirmatoryCorrelationResult]:
    """Compute and write confirmatory lag curves for available M4 tables."""
    tables = discover_aligned_duration_tables(
        aligned_dir,
        durations=tuple(durations) if durations is not None else EXPECTED_DURATIONS_S,
    )
    results: dict[int, ConfirmatoryCorrelationResult] = {}
    for duration, path in sorted(tables.items()):
        result = compute_signed_lag_curves_from_csv(
            path,
            duration_s=duration,
            lag_max_s=lag_max_s,
            lag_step_s=lag_step_s,
            min_overlap=min_overlap,
        )
        write_correlation_outputs(result, output_dir)
        results[duration] = result
    return results


__all__ = [
    "BAND_ORDER",
    "CURVES_TEMPLATE",
    "DEFAULT_LAG_MAX_S",
    "DEFAULT_LAG_MIN_S",
    "DEFAULT_LAG_STEP_S",
    "DEFAULT_MIN_OVERLAP",
    "FS_HZ",
    "POWER_REPRESENTATIONS",
    "PRIMARY_POWER_REPRESENTATION",
    "QC_TEMPLATE",
    "ConfirmatoryCorrelationResult",
    "compute_signed_lag_curves",
    "compute_signed_lag_curves_from_csv",
    "discover_aligned_duration_tables",
    "expected_lag_count",
    "read_aligned_features_csv",
    "run_confirmatory_correlations",
    "write_correlation_outputs",
]
