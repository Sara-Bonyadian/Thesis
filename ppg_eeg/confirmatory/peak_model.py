"""Subject-level Gaussian peak fits on Fisher-z lag curves.

Model
-----
``z(τ) = C + A · exp(-(τ - μ)² / (2σ²))``

Constraints (confirmatory protocol)
-----------------------------------
- ``A ≥ 0``
- ``μ ∈ [-20, +20]`` s
- ``σ ∈ [1, 60]`` s
- ``FWHM = 2.355 · σ``

Fitting uses weighted nonlinear least squares with lag ``n_overlap`` as weights.
This module performs **subject-level** fits only; group hierarchical inference is M10+.

Exclusion reasons (``exclusion_reason``)
----------------------------------------
- ``gaussian_fit_not_attempted``: too few finite weighted lag points to call the optimizer
- ``fit_failed``: optimizer was invoked but did not converge
- ``no_identifiable_positive_peak``: fit converged, but amplitude criteria failed
- empty string: identifiable positive peak (timing shift may be reported)

``endpoint_name`` is the canonical duration-contract ID (e.g. ``zlpi``);
``endpoint_alias`` is the short display acronym (``ZLPI`` / ``MWPI`` / ``SWPI``).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import curve_fit

from .correlation import CURVES_TEMPLATE, IDENTITY_FIELDS, identity_from_row
from .duration_contracts import EXPECTED_DURATIONS_S, contract_for_duration
from .endpoints import fisher_z

PARAMS_FILENAME = "peak_fit_params.csv"
QC_FILENAME = "peak_fit_qc.csv"

MU_BOUND_S = 20.0
SIGMA_MIN_S = 1.0
SIGMA_MAX_S = 60.0
FWHM_FACTOR = 2.355
BOUNDARY_ATOL = 1e-6
MIN_IDENTIFIABLE_A = 1e-3
MIN_FINITE_POINTS = 4
DEFAULT_SIGMA0_S = 10.0

EXCLUSION_FIT_NOT_ATTEMPTED = "gaussian_fit_not_attempted"
EXCLUSION_FIT_FAILED = "fit_failed"
EXCLUSION_NO_IDENTIFIABLE_PEAK = "no_identifiable_positive_peak"

PARAMS_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "endpoint_name",
    "endpoint_alias",
    "is_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_representation",
    "pair",
    "converged",
    "has_identifiable_peak",
    "report_timing_shift",
    "baseline_C",
    "peak_height_A",
    "peak_center_mu_s",
    "sigma_s",
    "fwhm_s",
    "peak_r",
    "peak_lag_observed_s",
    "peak_lag_fitted_s",
    "se_baseline_C",
    "se_peak_height_A",
    "se_peak_center_mu_s",
    "se_sigma_s",
    "rmse",
    "weighted_rss",
    "n_lags_fit",
    "n_common_support",
    "A_at_lower_bound",
    "mu_at_bound",
    "sigma_at_bound",
    "exclusion_reason",
)

QC_FIELDS = IDENTITY_FIELDS + (
    "duration_s",
    "duration_role",
    "endpoint_name",
    "endpoint_alias",
    "band",
    "power_representation",
    "pair",
    "converged",
    "has_identifiable_peak",
    "report_timing_shift",
    "fit_success",
    "optimizer_message",
    "n_lags_input",
    "n_lags_fit",
    "n_common_support",
    "overlap_is_constant",
    "init_baseline_C",
    "init_peak_height_A",
    "init_peak_center_mu_s",
    "init_sigma_s",
    "boundary_hit",
    "exclusion_reason",
)


@dataclass(frozen=True)
class PeakFitResult:
    params_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def gaussian_peak(
    lag_s: np.ndarray | float,
    baseline_C: float,
    peak_height_A: float,
    peak_center_mu_s: float,
    sigma_s: float,
) -> np.ndarray | float:
    """Confirmatory Gaussian peak model on Fisher-z units."""
    lag = np.asarray(lag_s, dtype=float)
    if sigma_s <= 0:
        raise ValueError("sigma_s must be > 0.")
    values = baseline_C + peak_height_A * np.exp(
        -0.5 * ((lag - peak_center_mu_s) / sigma_s) ** 2
    )
    if np.isscalar(lag_s):
        return float(values)
    return values


def fwhm_from_sigma(sigma_s: float) -> float:
    if not math.isfinite(sigma_s) or sigma_s <= 0:
        return float("nan")
    return float(FWHM_FACTOR * sigma_s)


def _identity_from_row(row: Mapping[str, object]) -> dict[str, str]:
    return identity_from_row(row)


def _observed_peak_from_r(
    lags: Sequence[float] | np.ndarray,
    r_values: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return (peak_r, peak_lag_observed_s) via max-|r| (same rule as C2 QC)."""
    candidates = [
        (float(lag), float(r))
        for lag, r in zip(
            np.asarray(lags, dtype=float),
            np.asarray(r_values, dtype=float),
            strict=True,
        )
        if np.isfinite(lag) and np.isfinite(r)
    ]
    if not candidates:
        return float("nan"), float("nan")
    lag_s, peak_r = min(
        candidates,
        key=lambda item: (-abs(item[1]), abs(item[0]), item[0]),
    )
    return peak_r, lag_s


def _group_curve_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[list[Mapping[str, object]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        identity = _identity_from_row(row)
        key = tuple(identity[field] for field in IDENTITY_FIELDS) + (
            str(row.get("band", "")).strip().casefold(),
            str(row.get("power_representation", "")).strip(),
            str(row.get("duration_s", "")).strip(),
        )
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    return [grouped[key] for key in order]


def _initial_guess(
    lags: np.ndarray,
    z_values: np.ndarray,
) -> tuple[float, float, float, float]:
    baseline = float(np.median(z_values))
    peak_idx = int(np.argmax(z_values))
    peak_height = float(max(0.0, z_values[peak_idx] - baseline))
    mu0 = float(np.clip(lags[peak_idx], -MU_BOUND_S, MU_BOUND_S))
    sigma0 = DEFAULT_SIGMA0_S
    if peak_height <= 0:
        mu0 = 0.0
    return baseline, peak_height, mu0, sigma0


def _near_bound(value: float, lo: float, hi: float, *, atol: float = BOUNDARY_ATOL) -> bool:
    if not math.isfinite(value):
        return False
    return abs(value - lo) <= atol or abs(value - hi) <= atol


def fit_gaussian_peak(
    lags_s: Sequence[float] | np.ndarray,
    z_values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one weighted Gaussian peak to a Fisher-z lag curve."""
    lags = np.asarray(lags_s, dtype=float)
    z = np.asarray(z_values, dtype=float)
    if lags.shape != z.shape:
        raise ValueError("lags_s and z_values must have the same shape.")
    if weights is None:
        w = np.ones(lags.shape, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != lags.shape:
            raise ValueError("weights must match lags_s.")

    mask = np.isfinite(lags) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    lags_f = lags[mask]
    z_f = z[mask]
    w_f = w[mask]
    n_fit = int(lags_f.size)

    empty = {
        "converged": False,
        "has_identifiable_peak": False,
        "report_timing_shift": False,
        "baseline_C": float("nan"),
        "peak_height_A": float("nan"),
        "peak_center_mu_s": float("nan"),
        "sigma_s": float("nan"),
        "fwhm_s": float("nan"),
        "peak_lag_fitted_s": float("nan"),
        "se_baseline_C": float("nan"),
        "se_peak_height_A": float("nan"),
        "se_peak_center_mu_s": float("nan"),
        "se_sigma_s": float("nan"),
        "rmse": float("nan"),
        "weighted_rss": float("nan"),
        "n_lags_fit": n_fit,
        "A_at_lower_bound": False,
        "mu_at_bound": False,
        "sigma_at_bound": False,
        "boundary_hit": False,
        "fit_success": False,
        "optimizer_message": "",
        "init_baseline_C": float("nan"),
        "init_peak_height_A": float("nan"),
        "init_peak_center_mu_s": float("nan"),
        "init_sigma_s": float("nan"),
        "exclusion_reason": "",
    }

    if n_fit < MIN_FINITE_POINTS:
        empty["exclusion_reason"] = EXCLUSION_FIT_NOT_ATTEMPTED
        empty["optimizer_message"] = (
            "Gaussian fit not attempted: need at least 4 finite weighted lag points."
        )
        return empty

    c0, a0, mu0, sigma0 = _initial_guess(lags_f, z_f)
    empty.update(
        {
            "init_baseline_C": c0,
            "init_peak_height_A": a0,
            "init_peak_center_mu_s": mu0,
            "init_sigma_s": sigma0,
        }
    )

    # curve_fit sigma is y-std; larger weight → smaller sigma.
    y_sigma = 1.0 / np.sqrt(w_f)
    bounds_lower = [-np.inf, 0.0, -MU_BOUND_S, SIGMA_MIN_S]
    bounds_upper = [np.inf, np.inf, MU_BOUND_S, SIGMA_MAX_S]

    try:
        popt, pcov = curve_fit(
            gaussian_peak,
            lags_f,
            z_f,
            p0=(c0, a0, mu0, sigma0),
            bounds=(bounds_lower, bounds_upper),
            sigma=y_sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        message = "curve_fit converged"
        success = True
    except (RuntimeError, ValueError) as exc:
        empty["exclusion_reason"] = EXCLUSION_FIT_FAILED
        empty["optimizer_message"] = str(exc)
        return empty

    baseline_C, peak_height_A, peak_center_mu_s, sigma_s = (float(v) for v in popt)
    se = np.full(4, np.nan, dtype=float)
    if pcov is not None:
        diag = np.diag(np.asarray(pcov, dtype=float))
        if diag.shape == (4,) and np.all(np.isfinite(diag)) and np.all(diag >= 0):
            se = np.sqrt(diag)

    fitted = gaussian_peak(lags_f, baseline_C, peak_height_A, peak_center_mu_s, sigma_s)
    resid = z_f - fitted
    weighted_rss = float(np.sum(w_f * resid * resid))
    rmse = float(np.sqrt(np.mean(resid * resid)))

    a_at_lower = peak_height_A <= BOUNDARY_ATOL
    mu_at_bound = _near_bound(peak_center_mu_s, -MU_BOUND_S, MU_BOUND_S)
    sigma_at_bound = _near_bound(sigma_s, SIGMA_MIN_S, SIGMA_MAX_S)
    boundary_hit = bool(a_at_lower or mu_at_bound or sigma_at_bound)

    # Positive identifiable peak: height above absolute floor and residual noise.
    amplitude_ok = peak_height_A >= MIN_IDENTIFIABLE_A and peak_height_A >= (
        2.0 * rmse
    )
    se_A = float(se[1])
    if math.isfinite(se_A) and se_A > 0:
        amplitude_ok = amplitude_ok and peak_height_A >= (2.0 * se_A)
    # Reject weak edge artifacts when mu sits on the ±20 s wall.
    if mu_at_bound and peak_height_A < 0.2:
        amplitude_ok = False

    has_peak = bool(success and amplitude_ok)
    report_timing = bool(has_peak)
    reported_mu = peak_center_mu_s if report_timing else float("nan")
    reported_se_mu = (
        float(se[2]) if report_timing and math.isfinite(float(se[2])) else float("nan")
    )

    return {
        "converged": True,
        "has_identifiable_peak": has_peak,
        "report_timing_shift": report_timing,
        "baseline_C": baseline_C,
        "peak_height_A": peak_height_A,
        # Timing-report μ remains NaN when the peak is not identifiable.
        "peak_center_mu_s": reported_mu,
        "sigma_s": sigma_s,
        "fwhm_s": fwhm_from_sigma(sigma_s),
        # Always retain the optimizer μ for comparison with the observed peak.
        "peak_lag_fitted_s": peak_center_mu_s,
        "se_baseline_C": float(se[0]),
        "se_peak_height_A": float(se[1]),
        "se_peak_center_mu_s": reported_se_mu,
        "se_sigma_s": float(se[3]),
        "rmse": rmse,
        "weighted_rss": weighted_rss,
        "n_lags_fit": n_fit,
        "A_at_lower_bound": a_at_lower,
        "mu_at_bound": mu_at_bound,
        "sigma_at_bound": sigma_at_bound,
        "boundary_hit": boundary_hit,
        "fit_success": True,
        "optimizer_message": message,
        "init_baseline_C": c0,
        "init_peak_height_A": a0,
        "init_peak_center_mu_s": mu0,
        "init_sigma_s": sigma0,
        "exclusion_reason": (
            "" if has_peak else EXCLUSION_NO_IDENTIFIABLE_PEAK
        ),
    }


def evaluate_peak_curve(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit a peak model for one observation × band × representation curve."""
    if duration_s not in EXPECTED_DURATIONS_S:
        raise ValueError(f"Unsupported duration_s={duration_s}.")
    contract = contract_for_duration(duration_s)
    fit_param_keys = (
        "converged",
        "has_identifiable_peak",
        "report_timing_shift",
        "baseline_C",
        "peak_height_A",
        "peak_center_mu_s",
        "sigma_s",
        "fwhm_s",
        "peak_lag_fitted_s",
        "se_baseline_C",
        "se_peak_height_A",
        "se_peak_center_mu_s",
        "se_sigma_s",
        "rmse",
        "weighted_rss",
        "n_lags_fit",
        "A_at_lower_bound",
        "mu_at_bound",
        "sigma_at_bound",
        "exclusion_reason",
    )
    qc_fit_keys = (
        "converged",
        "has_identifiable_peak",
        "report_timing_shift",
        "fit_success",
        "optimizer_message",
        "n_lags_fit",
        "init_baseline_C",
        "init_peak_height_A",
        "init_peak_center_mu_s",
        "init_sigma_s",
        "boundary_hit",
        "exclusion_reason",
    )
    if not curve_rows:
        identity = {field: "" for field in IDENTITY_FIELDS}
        fit = fit_gaussian_peak([], [])
        params = {
            **identity,
            "duration_s": duration_s,
            "duration_role": "",
            "endpoint_name": contract.endpoint_name,
            "endpoint_alias": contract.endpoint_alias,
            "is_standard_zlpi": contract.is_standard_zlpi,
            "band": "",
            "power_representation": "",
            "is_primary_representation": False,
            "pair": "",
            "peak_r": float("nan"),
            "peak_lag_observed_s": float("nan"),
            "n_common_support": 0,
            **{key: fit[key] for key in fit_param_keys},
        }
        qc = {
            **identity,
            "duration_s": duration_s,
            "duration_role": "",
            "endpoint_name": contract.endpoint_name,
            "endpoint_alias": contract.endpoint_alias,
            "band": "",
            "power_representation": "",
            "pair": "",
            "n_lags_input": 0,
            "n_common_support": 0,
            "overlap_is_constant": False,
            **{key: fit[key] for key in qc_fit_keys},
        }
        return params, qc

    first = curve_rows[0]
    identity = _identity_from_row(first)
    ordered = sorted(curve_rows, key=lambda row: float(row["lag_s"]))
    lags = np.asarray([float(row["lag_s"]) for row in ordered], dtype=float)
    r_values = np.asarray([float(row["r"]) for row in ordered], dtype=float)
    overlaps = np.asarray([float(row.get("n_overlap", 0)) for row in ordered], dtype=float)
    z_values = np.asarray([fisher_z(float(r)) for r in r_values], dtype=float)
    peak_r, peak_lag_observed_s = _observed_peak_from_r(lags, r_values)

    finite_overlap = overlaps[np.isfinite(overlaps) & (overlaps > 0)]
    overlap_is_constant = (
        bool(finite_overlap.size)
        and float(np.max(finite_overlap) - np.min(finite_overlap)) <= 1e-12
    )
    n_common_support = int(round(float(np.min(finite_overlap)))) if finite_overlap.size else 0

    fit = fit_gaussian_peak(lags, z_values, weights=overlaps)
    params = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": str(first.get("duration_role", "")).strip(),
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": bool(contract.is_standard_zlpi),
        "band": str(first.get("band", "")).strip().casefold(),
        "power_representation": str(first.get("power_representation", "")).strip(),
        "is_primary_representation": bool(first.get("is_primary_representation", False)),
        "pair": str(first.get("pair", "")).strip(),
        "peak_r": peak_r,
        "peak_lag_observed_s": peak_lag_observed_s,
        "n_common_support": n_common_support,
        **{key: fit[key] for key in fit_param_keys},
    }
    qc = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": str(first.get("duration_role", "")).strip(),
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "band": str(first.get("band", "")).strip().casefold(),
        "power_representation": str(first.get("power_representation", "")).strip(),
        "pair": str(first.get("pair", "")).strip(),
        "n_lags_input": int(lags.size),
        "n_common_support": n_common_support,
        "overlap_is_constant": overlap_is_constant,
        **{key: fit[key] for key in qc_fit_keys},
    }
    return params, qc


def compute_peak_fits_from_curves(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
) -> PeakFitResult:
    """Fit subject-level peaks for all curves at one confirmatory duration."""
    params_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for group in _group_curve_rows(curve_rows):
        params, qc = evaluate_peak_curve(group, duration_s=duration_s)
        params_rows.append(params)
        qc_rows.append(qc)
    return PeakFitResult(params_rows=tuple(params_rows), qc_rows=tuple(qc_rows))


def read_curve_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_peak_fits_from_csv(
    curves_csv: str | Path,
    *,
    duration_s: int,
) -> PeakFitResult:
    return compute_peak_fits_from_curves(
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


def write_peak_fit_outputs(
    result: PeakFitResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write ``peak_fit_params.csv`` and ``peak_fit_qc.csv``."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    params_path = output_path / PARAMS_FILENAME
    qc_path = output_path / QC_FILENAME
    _write_csv(params_path, result.params_rows, PARAMS_FIELDS)
    _write_csv(qc_path, result.qc_rows, QC_FIELDS)
    return {"params": params_path, "qc": qc_path}


def append_peak_fit_outputs(
    result: PeakFitResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Append rows into shared peak_fit_*.csv files (multi-duration runs)."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    params_path = output_path / PARAMS_FILENAME
    qc_path = output_path / QC_FILENAME

    def _append(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    _append(params_path, result.params_rows, PARAMS_FIELDS)
    _append(qc_path, result.qc_rows, QC_FIELDS)
    return {"params": params_path, "qc": qc_path}


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


def run_confirmatory_peak_fits(
    curves_dir: str | Path,
    output_dir: str | Path,
    *,
    durations: Sequence[int] | None = None,
) -> PeakFitResult:
    """Fit peaks for available duration curve tables into shared output CSVs."""
    tables = discover_curve_tables(
        curves_dir,
        durations=tuple(durations) if durations is not None else EXPECTED_DURATIONS_S,
    )
    all_params: list[dict[str, object]] = []
    all_qc: list[dict[str, object]] = []
    for duration, path in sorted(tables.items()):
        result = compute_peak_fits_from_csv(path, duration_s=duration)
        all_params.extend(result.params_rows)
        all_qc.extend(result.qc_rows)
    combined = PeakFitResult(params_rows=tuple(all_params), qc_rows=tuple(all_qc))
    write_peak_fit_outputs(combined, output_dir)
    return combined


__all__ = [
    "DEFAULT_SIGMA0_S",
    "EXCLUSION_FIT_FAILED",
    "EXCLUSION_FIT_NOT_ATTEMPTED",
    "EXCLUSION_NO_IDENTIFIABLE_PEAK",
    "FWHM_FACTOR",
    "MIN_IDENTIFIABLE_A",
    "MU_BOUND_S",
    "PARAMS_FILENAME",
    "QC_FILENAME",
    "SIGMA_MAX_S",
    "SIGMA_MIN_S",
    "PeakFitResult",
    "append_peak_fit_outputs",
    "compute_peak_fits_from_csv",
    "compute_peak_fits_from_curves",
    "discover_curve_tables",
    "evaluate_peak_curve",
    "fit_gaussian_peak",
    "fwhm_from_sigma",
    "gaussian_peak",
    "read_curve_csv",
    "run_confirmatory_peak_fits",
    "write_peak_fit_outputs",
]
