"""Signed instantaneous-HR × EEG-power lag correlations for confirmatory analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ppg_eeg.temporal_coupling.cross_correlation import (
    LagCorrelationPoint,
    build_lag_grid,
)

from .config import EXPECTED_BANDS_HZ, EXPECTED_DURATIONS_S
from .duration_contracts import (
    ANALYSIS_ROLE_MID_WINDOW_SENSITIVITY,
    ANALYSIS_ROLE_SHORT_WINDOW_SENSITIVITY,
    ANALYSIS_ROLE_STANDARD_ZLPI,
    EXPECTED_LAG_STEP_S,
    MID_WINDOW_DURATION_S,
    MWPI_LAG_MAX_S,
    SHORT_WINDOW_DURATION_S,
    STANDARD_ZLPI_DURATIONS_S,
    STANDARD_ZLPI_LAG_MAX_S,
    SWPI_LAG_MAX_S,
    contract_for_duration,
)
from .harmonize import ALIGNED_FEATURES_TEMPLATE
from .reason_codes import (
    STRUCTURED_NC_FIELDS,
    attach_structured_reason,
    map_exclusion_to_reason_code,
)

FS_HZ = 1.0
DEFAULT_LAG_STEP_S = EXPECTED_LAG_STEP_S
DEFAULT_MIN_OVERLAP = 2  # Pearson needs ≥2; ZLPI overlap gates are applied in M6

# Re-exports / aliases for duration-specific lag envelopes.
STANDARD_ZLPI_N_LAGS = STANDARD_ZLPI_LAG_MAX_S * 2 + 1  # 121
MID_WINDOW_N_LAGS = MWPI_LAG_MAX_S * 2 + 1  # 61
SHORT_WINDOW_N_LAGS = SWPI_LAG_MAX_S * 2 + 1  # 41
STANDARD_ANALYSIS_ROLE = ANALYSIS_ROLE_STANDARD_ZLPI
MID_WINDOW_ANALYSIS_ROLE = ANALYSIS_ROLE_MID_WINDOW_SENSITIVITY
SHORT_WINDOW_ANALYSIS_ROLE = ANALYSIS_ROLE_SHORT_WINDOW_SENSITIVITY
DEFAULT_LAG_MAX_S = STANDARD_ZLPI_LAG_MAX_S
DEFAULT_LAG_MIN_S = -STANDARD_ZLPI_LAG_MAX_S

BAND_ORDER = tuple(EXPECTED_BANDS_HZ)

# (representation_id, aligned-column suffix after "{band}_", is_primary)
POWER_REPRESENTATIONS: tuple[tuple[str, str, bool], ...] = (
    ("absolute_log10", "absolute_log10_power_z", True),
    ("relative", "relative_power_z", False),
    ("broadband_residualized", "broadband_residualized_log10_z", False),
)
PRIMARY_POWER_REPRESENTATION = "absolute_log10"
DEFAULT_CARDIAC_VARIABLE = "instantaneous_hr"

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
    "lag_analysis_role",
    "endpoint_name",
    "endpoint_alias",
    "is_standard_zlpi",
    "pool_with_standard_zlpi",
    "flank_inner_s",
    "flank_outer_s",
    "shoulders_inner_s",
    "shoulders_outer_s",
    "cardiac_variable",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "lag_s",
    "r",
    "n_overlap",
)

QC_FIELDS = tuple(
    dict.fromkeys(
        IDENTITY_FIELDS
        + (
            "duration_s",
            "duration_role",
            "lag_analysis_role",
            "endpoint_name",
            "endpoint_alias",
            "is_standard_zlpi",
            "pool_with_standard_zlpi",
            "flank_inner_s",
            "flank_outer_s",
            "shoulders_inner_s",
            "shoulders_outer_s",
            "cardiac_variable",
            "band",
            "power_representation",
            "is_primary_representation",
            "pair",
            "n_samples",
            "n_lags",
            "lag_min_s",
            "lag_max_s",
            "lag_step_s",
            "n_common_support",
            "overlap_is_constant",
            "n_finite_r",
            "n_nonfinite_r",
            "min_n_overlap",
            "max_n_overlap",
            "n_overlap_at_zero",
            "r_at_zero",
            "peak_r",
            "peak_abs_r",
            "peak_lag_s",
            "exclusion_reason",
        )
        + STRUCTURED_NC_FIELDS
    )
)


@dataclass(frozen=True)
class DurationLagSpec:
    duration_s: int
    lag_min_s: int
    lag_max_s: int
    lag_step_s: int
    n_lags: int
    lag_analysis_role: str
    endpoint_name: str
    endpoint_alias: str
    flank_inner_s: int
    flank_outer_s: int
    shoulders_inner_s: int
    shoulders_outer_s: int
    is_standard_zlpi: bool
    pool_with_standard_zlpi: bool

    @property
    def is_zlpi_input(self) -> bool:
        """Backward-compatible alias for ``is_standard_zlpi``."""
        return self.is_standard_zlpi


@dataclass(frozen=True)
class ConfirmatoryCorrelationResult:
    duration_s: int
    lag_grid_s: tuple[float, ...]
    lag_spec: DurationLagSpec
    curve_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def lag_spec_for_duration(
    duration_s: int,
    *,
    lag_step_s: int = DEFAULT_LAG_STEP_S,
) -> DurationLagSpec:
    """Return the frozen lag + endpoint contract for a confirmatory duration."""
    if lag_step_s != DEFAULT_LAG_STEP_S:
        raise ValueError(
            f"Confirmatory lag_step_s must be {DEFAULT_LAG_STEP_S}, got {lag_step_s}."
        )
    contract = contract_for_duration(duration_s)
    return DurationLagSpec(
        duration_s=contract.duration_s,
        lag_min_s=contract.lag_min_s,
        lag_max_s=contract.lag_max_s,
        lag_step_s=contract.lag_step_s,
        n_lags=contract.n_lags,
        lag_analysis_role=contract.lag_analysis_role,
        endpoint_name=contract.endpoint_name,
        endpoint_alias=contract.endpoint_alias,
        flank_inner_s=contract.flank_inner_s,
        flank_outer_s=contract.flank_outer_s,
        shoulders_inner_s=contract.shoulders_inner_s,
        shoulders_outer_s=contract.shoulders_outer_s,
        is_standard_zlpi=contract.is_standard_zlpi,
        pool_with_standard_zlpi=contract.pool_with_standard_zlpi,
    )


def expected_lag_count(duration_s: int | None = None, **kwargs: float) -> int:
    """Return planned lag count for a duration, or for an explicit symmetric grid."""
    if duration_s is not None:
        if kwargs:
            raise ValueError("Pass either duration_s or explicit lag bounds, not both.")
        return lag_spec_for_duration(int(duration_s)).n_lags
    lag_min_s = float(kwargs.get("lag_min_s", DEFAULT_LAG_MIN_S))
    lag_max_s = float(kwargs.get("lag_max_s", DEFAULT_LAG_MAX_S))
    lag_step_s = float(kwargs.get("lag_step_s", DEFAULT_LAG_STEP_S))
    if lag_step_s <= 0:
        raise ValueError("lag_step_s must be > 0.")
    if lag_max_s < lag_min_s:
        raise ValueError("lag_max_s must be >= lag_min_s.")
    if abs(lag_min_s + lag_max_s) > 1e-12:
        raise ValueError("Confirmatory lag grids must be symmetric about zero.")
    return int(round(lag_max_s / lag_step_s)) * 2 + 1


def _contract_row_fields(spec: DurationLagSpec) -> dict[str, object]:
    return {
        "lag_analysis_role": spec.lag_analysis_role,
        "endpoint_name": spec.endpoint_name,
        "endpoint_alias": spec.endpoint_alias,
        "is_standard_zlpi": bool(spec.is_standard_zlpi),
        "pool_with_standard_zlpi": bool(spec.pool_with_standard_zlpi),
        "flank_inner_s": int(spec.flank_inner_s),
        "flank_outer_s": int(spec.flank_outer_s),
        "shoulders_inner_s": int(spec.shoulders_inner_s),
        "shoulders_outer_s": int(spec.shoulders_outer_s),
    }


def _pair_name(band: str, representation: str) -> str:
    return f"hr_x_{band}_{representation}"


def _eeg_column(band: str, suffix: str) -> str:
    return f"{band}_{suffix}"


def _as_float_series(values: Sequence[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=float)


def _condition_from_observation_id(observation_id: str, dataset_id: str) -> str:
    """Best-effort condition recovery when aligned tables omit the field."""
    oid = str(observation_id).strip().casefold()
    dataset = str(dataset_id).strip().casefold()
    if not oid:
        return ""
    if dataset == "hiit":
        # hiit-01-ph-post-rest → ph_post_rest
        parts = oid.split("-")
        if len(parts) >= 5 and parts[0] == "hiit":
            return "_".join(parts[2:])
    return ""


def identity_from_row(
    row: Mapping[str, object],
    *,
    condition_by_observation: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Extract IDENTITY_FIELDS, recovering blank ``condition`` when possible."""
    identity = {field: str(row.get(field, "")).strip() for field in IDENTITY_FIELDS}
    if identity["condition"]:
        identity["condition"] = identity["condition"].casefold()
        return identity
    observation_id = identity["observation_id"]
    if condition_by_observation:
        for key in (observation_id, observation_id.casefold()):
            if key in condition_by_observation and condition_by_observation[key]:
                identity["condition"] = str(condition_by_observation[key]).strip().casefold()
                return identity
    recovered = _condition_from_observation_id(
        observation_id, identity["dataset_id"]
    )
    if recovered:
        identity["condition"] = recovered
    return identity


def _identity_from_row(
    row: Mapping[str, object],
    *,
    condition_by_observation: Mapping[str, str] | None = None,
) -> dict[str, str]:
    return identity_from_row(
        row, condition_by_observation=condition_by_observation
    )


def _group_aligned_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    condition_by_observation: Mapping[str, str] | None = None,
) -> list[tuple[dict[str, str], str, list[Mapping[str, object]]]]:
    """Group by observation identity; preserve first-seen observation order."""
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    meta: dict[tuple[str, ...], tuple[dict[str, str], str]] = {}
    for row in rows:
        identity = _identity_from_row(
            row, condition_by_observation=condition_by_observation
        )
        key = tuple(identity[field] for field in IDENTITY_FIELDS)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
            meta[key] = (identity, str(row.get("duration_role", "")).strip())
        grouped[key].append(row)
    return [(meta[key][0], meta[key][1], grouped[key]) for key in order]


def _peak_summary_from_curve(
    curve: Sequence[LagCorrelationPoint],
) -> tuple[float, float, float]:
    """Return (peak_r, peak_abs_r, peak_lag_s) from max-|r| on the lag curve.

    On |r| ties, prefer the lag closest to zero, then the more negative lag.
    """
    candidates = [
        (float(point.lag_s), float(point.r))
        for point in curve
        if np.isfinite(point.r)
    ]
    if not candidates:
        return float("nan"), float("nan"), float("nan")
    lag_s, peak_r = min(
        candidates,
        key=lambda item: (-abs(item[1]), abs(item[0]), item[0]),
    )
    return peak_r, abs(peak_r), lag_s


def read_aligned_features_csv(path: str | Path) -> list[dict[str, str]]:
    """Read an M4 aligned duration table."""
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(float) - float(np.mean(x))
    y = y.astype(float) - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def common_support_anchor_indices(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_max_samples: int,
) -> np.ndarray:
    """Return anchors usable at every lag in [-lag_max, +lag_max].

    Anchor ``i`` is kept only when ``cardiac[i]`` and ``eeg[i + τ]`` are finite
    for every integer lag ``τ`` in the configured range. Using only these anchors
    makes ``n_overlap`` identical across lags.
    """
    if lag_max_samples < 0:
        raise ValueError("lag_max_samples must be >= 0.")
    n = min(int(cardiac.size), int(eeg.size))
    cardiac = np.asarray(cardiac, dtype=float)[:n]
    eeg = np.asarray(eeg, dtype=float)[:n]
    if n <= 2 * lag_max_samples:
        return np.asarray([], dtype=int)

    anchors: list[int] = []
    for index in range(lag_max_samples, n - lag_max_samples):
        if not np.isfinite(cardiac[index]):
            continue
        window = eeg[index - lag_max_samples : index + lag_max_samples + 1]
        if window.size != 2 * lag_max_samples + 1:
            continue
        if np.all(np.isfinite(window)):
            anchors.append(index)
    return np.asarray(anchors, dtype=int)


def correlate_at_lag_common_support(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_s: float,
    anchor_indices: np.ndarray,
    fs_hz: float = FS_HZ,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> LagCorrelationPoint:
    """Correlate cardiac(t) with eeg(t + lag_s) on a fixed common-support set.

    Negative lag: EEG leads cardiac. Positive lag: cardiac leads EEG.
    """
    lag_samples = int(round(float(lag_s) * float(fs_hz)))
    if anchor_indices.size == 0:
        return LagCorrelationPoint(lag_s=float(lag_s), r=float("nan"), n_overlap=0)

    cardiac = np.asarray(cardiac, dtype=float)
    eeg = np.asarray(eeg, dtype=float)
    x = cardiac[anchor_indices]
    y = eeg[anchor_indices + lag_samples]
    mask = np.isfinite(x) & np.isfinite(y)
    n_overlap = int(mask.sum())
    # By construction of common-support anchors, mask should be all-True.
    if n_overlap < min_overlap:
        return LagCorrelationPoint(
            lag_s=float(lag_s), r=float("nan"), n_overlap=n_overlap
        )
    return LagCorrelationPoint(
        lag_s=float(lag_s),
        r=_pearson_r(x[mask], y[mask]),
        n_overlap=n_overlap,
    )


def compute_correlation_curve_common_support(
    cardiac: np.ndarray,
    eeg: np.ndarray,
    *,
    lag_grid_s: np.ndarray,
    lag_max_s: float,
    fs_hz: float = FS_HZ,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> tuple[list[LagCorrelationPoint], np.ndarray]:
    """Compute a lag curve with identical ``n_overlap`` at every lag."""
    lag_max_samples = int(round(float(lag_max_s) * float(fs_hz)))
    anchors = common_support_anchor_indices(
        cardiac, eeg, lag_max_samples=lag_max_samples
    )
    curve = [
        correlate_at_lag_common_support(
            cardiac,
            eeg,
            lag_s=float(lag_s),
            anchor_indices=anchors,
            fs_hz=fs_hz,
            min_overlap=min_overlap,
        )
        for lag_s in lag_grid_s
    ]
    return curve, anchors


def compute_signed_lag_curves(
    aligned_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
    lag_step_s: int = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    bands: Sequence[str] = BAND_ORDER,
    representations: Sequence[tuple[str, str, bool]] = POWER_REPRESENTATIONS,
    cardiac_variable: str = DEFAULT_CARDIAC_VARIABLE,
    condition_by_observation: Mapping[str, str] | None = None,
) -> ConfirmatoryCorrelationResult:
    """Compute signed HR_z × EEG-power_z lag curves for every band/representation.

    Duration selects the frozen lag + endpoint contract:
    - D240/D180 → ±60 s (121 lags), standard ``zlpi``
    - D120 → ±30 s (61 lags), ``mid_window_proximal_index`` (not ZLPI)
    - D60 → ±20 s (41 lags), ``short_window_proximal_index`` (not ZLPI)

    Each curve is cropped to common temporal support so ``n_overlap`` is identical
    across lags. Convention: ``corr(hr(t), eeg(t + lag))``; positive lag means
    EEG follows HR.
    """
    if min_overlap < 2:
        raise ValueError("min_overlap must be >= 2.")

    spec = lag_spec_for_duration(duration_s, lag_step_s=lag_step_s)
    lag_grid = build_lag_grid(
        lag_max_s=float(spec.lag_max_s), lag_step_s=float(spec.lag_step_s)
    )
    lag_grid_t = tuple(float(value) for value in lag_grid)
    if int(lag_grid.size) != spec.n_lags:
        raise RuntimeError(
            f"Lag grid length {lag_grid.size} does not match planned {spec.n_lags}."
        )

    unknown_bands = [band for band in bands if band not in EXPECTED_BANDS_HZ]
    if unknown_bands:
        raise ValueError(f"Unknown EEG bands: {unknown_bands}.")
    cardiac_variable_name = str(cardiac_variable).strip() or DEFAULT_CARDIAC_VARIABLE

    curve_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for identity, duration_role, obs_rows in _group_aligned_rows(
        aligned_rows,
        condition_by_observation=condition_by_observation,
    ):
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
                anchors = np.asarray([], dtype=int)
                if n_samples == 0:
                    exclusion_reason = "no_aligned_samples"
                    curve: list[LagCorrelationPoint] = []
                else:
                    curve, anchors = compute_correlation_curve_common_support(
                        hr_z,
                        eeg_z,
                        lag_grid_s=lag_grid,
                        lag_max_s=float(spec.lag_max_s),
                        fs_hz=FS_HZ,
                        min_overlap=min_overlap,
                    )
                    if anchors.size == 0:
                        exclusion_reason = "insufficient_common_support_for_lag_grid"

                finite_rs = [float(point.r) for point in curve if np.isfinite(point.r)]
                overlaps = [int(point.n_overlap) for point in curve]
                overlap_is_constant = (
                    len(set(overlaps)) <= 1 if overlaps else True
                )
                n_common_support = int(overlaps[0]) if overlaps else 0
                zero_point = next(
                    (point for point in curve if float(point.lag_s) == 0.0),
                    None,
                )
                peak_r, peak_abs_r, peak_lag_s = _peak_summary_from_curve(curve)
                for point in curve:
                    curve_rows.append(
                        {
                            **identity,
                            "duration_s": int(duration_s),
                            "duration_role": duration_role,
                            **_contract_row_fields(spec),
                            "cardiac_variable": cardiac_variable_name,
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
                    attach_structured_reason(
                        {
                            **identity,
                            "duration_s": int(duration_s),
                            "duration_role": duration_role,
                            **_contract_row_fields(spec),
                            "cardiac_variable": cardiac_variable_name,
                            "band": band,
                            "power_representation": representation,
                            "is_primary_representation": bool(is_primary),
                            "pair": pair,
                            "n_samples": n_samples,
                            "n_lags": int(spec.n_lags),
                            "lag_min_s": int(spec.lag_min_s),
                            "lag_max_s": int(spec.lag_max_s),
                            "lag_step_s": int(spec.lag_step_s),
                            "n_common_support": n_common_support,
                            "overlap_is_constant": bool(overlap_is_constant),
                            "n_finite_r": len(finite_rs),
                            "n_nonfinite_r": int(spec.n_lags) - len(finite_rs),
                            "min_n_overlap": int(min(overlaps)) if overlaps else 0,
                            "max_n_overlap": int(max(overlaps)) if overlaps else 0,
                            "n_overlap_at_zero": (
                                int(zero_point.n_overlap) if zero_point is not None else 0
                            ),
                            "r_at_zero": (
                                float(zero_point.r) if zero_point is not None else float("nan")
                            ),
                            "peak_r": peak_r,
                            "peak_abs_r": peak_abs_r,
                            "peak_lag_s": peak_lag_s,
                            "exclusion_reason": exclusion_reason,
                            "reason_code": map_exclusion_to_reason_code(exclusion_reason),
                        },
                        stage="C2",
                        eligible=not bool(exclusion_reason),
                        reason_code=map_exclusion_to_reason_code(exclusion_reason),
                        reason=exclusion_reason.replace("_", " ") if exclusion_reason else "",
                        required_evidence=(
                            f"lag_grid anchors with min_overlap>={min_overlap}"
                            if exclusion_reason
                            else ""
                        ),
                        observed_evidence=(
                            f"n_samples={n_samples}; n_common_support={n_common_support}"
                            if exclusion_reason
                            else ""
                        ),
                        specification_id=str(spec.endpoint_name or pair),
                    )
                )

    return ConfirmatoryCorrelationResult(
        duration_s=int(duration_s),
        lag_grid_s=lag_grid_t,
        lag_spec=spec,
        curve_rows=tuple(curve_rows),
        qc_rows=tuple(qc_rows),
    )


def compute_signed_lag_curves_from_csv(
    aligned_csv: str | Path,
    *,
    duration_s: int,
    lag_step_s: int = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    cardiac_variable: str = DEFAULT_CARDIAC_VARIABLE,
    condition_by_observation: Mapping[str, str] | None = None,
) -> ConfirmatoryCorrelationResult:
    """Load an M4 aligned duration CSV and compute confirmatory lag curves."""
    return compute_signed_lag_curves(
        read_aligned_features_csv(aligned_csv),
        duration_s=duration_s,
        lag_step_s=lag_step_s,
        min_overlap=min_overlap,
        cardiac_variable=cardiac_variable,
        condition_by_observation=condition_by_observation,
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
    lag_step_s: int = DEFAULT_LAG_STEP_S,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    cardiac_variable: str = DEFAULT_CARDIAC_VARIABLE,
    condition_by_observation: Mapping[str, str] | None = None,
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
            lag_step_s=lag_step_s,
            min_overlap=min_overlap,
            cardiac_variable=cardiac_variable,
            condition_by_observation=condition_by_observation,
        )
        write_correlation_outputs(result, output_dir)
        results[duration] = result
    return results


__all__ = [
    "BAND_ORDER",
    "CURVES_TEMPLATE",
    "DEFAULT_CARDIAC_VARIABLE",
    "DEFAULT_LAG_MAX_S",
    "DEFAULT_LAG_MIN_S",
    "DEFAULT_LAG_STEP_S",
    "DEFAULT_MIN_OVERLAP",
    "FS_HZ",
    "IDENTITY_FIELDS",
    "identity_from_row",
    "MID_WINDOW_ANALYSIS_ROLE",
    "MID_WINDOW_DURATION_S",
    "MID_WINDOW_N_LAGS",
    "POWER_REPRESENTATIONS",
    "PRIMARY_POWER_REPRESENTATION",
    "QC_TEMPLATE",
    "SHORT_WINDOW_ANALYSIS_ROLE",
    "SHORT_WINDOW_DURATION_S",
    "SHORT_WINDOW_N_LAGS",
    "STANDARD_ANALYSIS_ROLE",
    "STANDARD_ZLPI_DURATIONS_S",
    "STANDARD_ZLPI_LAG_MAX_S",
    "STANDARD_ZLPI_N_LAGS",
    "ConfirmatoryCorrelationResult",
    "DurationLagSpec",
    "common_support_anchor_indices",
    "compute_correlation_curve_common_support",
    "compute_signed_lag_curves",
    "compute_signed_lag_curves_from_csv",
    "correlate_at_lag_common_support",
    "discover_aligned_duration_tables",
    "expected_lag_count",
    "lag_spec_for_duration",
    "read_aligned_features_csv",
    "run_confirmatory_correlations",
    "write_correlation_outputs",
]
