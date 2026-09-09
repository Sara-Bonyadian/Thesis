"""Subject-level near-zero Gaussian peak fits on Fisher-z lag curves.

Model
-----
Primary production fit (near-zero central peak):
1) Estimate distant-lag linear baseline on flanks ``20 ≤ |τ| ≤ 60``::

    B(τ) = b0 + b1·τ

2) Baseline-adjust central window ``|τ| ≤ 20``::

    z_adj(τ) = z(τ) - B(τ)

3) Fit nonnegative Gaussian component on central window only::

    z_adj(τ) = A · exp(-(τ - μ)² / (2σ²))

For diagnostics and output compatibility, ``baseline_C`` stores ``b0``.
The helper ``gaussian_peak()`` remains available as
``C + A·exp(...)`` for synthetic tests and historical tooling.

Constraints (confirmatory protocol)
-----------------------------------
- ``A ≥ 0``
- ``μ ∈ [-20, +20]`` s
- ``σ ∈ [1, 60]`` s
- ``FWHM = 2.355 · σ``

Identifiability (positive peak; all must hold)
----------------------------------------------
- ``A ≥ MIN_IDENTIFIABLE_A`` (``1e-3``)
- ``A ≥ IDENTIFIABLE_A_OVER_RMSE × RMSE`` (production: ``1.8×RMSE``)
- ``A ≥ 2 × SE(A)`` when ``SE(A)`` is finite and positive
- weak-edge: reject when ``μ`` is at the ±20 s bound **and** ``A < 0.2``

After Option C baseline subtraction, ``RMSE`` and ``SE(A)`` are computed on the
baseline-adjusted central-window residuals (``|τ| ≤ 20``), with the flank
baseline held fixed. Threshold constants are unchanged; their reference noise
is therefore the central residual scale, not the full ±60 s residual scale.

Fitting uses weighted nonlinear least squares with lag ``n_overlap`` as weights.
This module performs **subject-level** fits only; group hierarchical inference is M10+.

Exclusion reasons (``exclusion_reason``)
----------------------------------------
- ``option_c_peak_requires_d180_d240``: duration is not standard ZLPI (D60/D120);
  Option C is not computed; A/μ/FWHM are not exported under this estimand
- ``insufficient_option_c_flank_support``: D180/D240 curve lacks ≥2 finite
  weighted lags in ``20 ≤ |τ| ≤ 60`` (no median-z or reduced-flank fallback)
- ``gaussian_fit_not_attempted``: too few finite weighted lag points to call the optimizer
- ``fit_failed``: optimizer was invoked but did not converge
- ``no_identifiable_positive_peak``: fit converged, but amplitude criteria failed
- empty string: identifiable positive peak (timing shift may be reported)

Option C is **only** the locked D180/D240 estimand. D60/D120 rows are written
as ``not_computable`` with ``OPTION_C_PEAK_NOT_APPLICABLE``.

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
from .reason_codes import (
    attach_structured_reason,
    map_exclusion_to_reason_code,
)

PARAMS_FILENAME = "peak_fit_params.csv"
QC_FILENAME = "peak_fit_qc.csv"

MU_BOUND_S = 20.0
SIGMA_MIN_S = 1.0
SIGMA_MAX_S = 60.0
FWHM_FACTOR = 2.355
BOUNDARY_ATOL = 1e-6
MIN_IDENTIFIABLE_A = 1e-3
# Amplitude vs residual noise: production Panel F / peak-timing gate.
IDENTIFIABLE_A_OVER_RMSE = 1.8
MIN_FINITE_POINTS = 4
DEFAULT_SIGMA0_S = 10.0
CENTRAL_LAG_MAX_S = 20.0
FLANK_INNER_S = 20.0
FLANK_OUTER_S = 60.0

EXCLUSION_FIT_NOT_ATTEMPTED = "gaussian_fit_not_attempted"
EXCLUSION_FIT_FAILED = "fit_failed"
EXCLUSION_NO_IDENTIFIABLE_PEAK = "no_identifiable_positive_peak"
EXCLUSION_OPTION_C_NOT_APPLICABLE = "option_c_peak_requires_d180_d240"
EXCLUSION_INSUFFICIENT_FLANKS = "insufficient_option_c_flank_support"
MIN_FLANK_POINTS = 2

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
    "status",
    "reason_code",
    "reason",
    "required_evidence",
    "observed_evidence",
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
    "status",
    "reason_code",
    "reason",
    "required_evidence",
    "observed_evidence",
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


def _gaussian_component(
    lag_s: np.ndarray | float,
    peak_height_A: float,
    peak_center_mu_s: float,
    sigma_s: float,
) -> np.ndarray | float:
    """Gaussian component without baseline (used on baseline-adjusted central data)."""
    lag = np.asarray(lag_s, dtype=float)
    if sigma_s <= 0:
        raise ValueError("sigma_s must be > 0.")
    values = peak_height_A * np.exp(-0.5 * ((lag - peak_center_mu_s) / sigma_s) ** 2)
    if np.isscalar(lag_s):
        return float(values)
    return values


def fwhm_from_sigma(sigma_s: float) -> float:
    if not math.isfinite(sigma_s) or sigma_s <= 0:
        return float("nan")
    return float(FWHM_FACTOR * sigma_s)


def _empty_fit_result(*, n_fit: int = 0, exclusion_reason: str = "") -> dict[str, object]:
    return {
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
        "exclusion_reason": exclusion_reason,
    }


def _with_peak_reason(
    row: Mapping[str, object],
    *,
    exclusion_reason: str,
    required_evidence: str = "",
    observed_evidence: str = "",
) -> dict[str, object]:
    """Attach C3 peak-model NC envelope."""
    if not exclusion_reason:
        status = "computed"
        eligible = True
    elif exclusion_reason in {
        EXCLUSION_OPTION_C_NOT_APPLICABLE,
        EXCLUSION_INSUFFICIENT_FLANKS,
    }:
        status = "not_computable"
        eligible = False
    else:
        status = "excluded"
        eligible = False
    code = map_exclusion_to_reason_code(exclusion_reason)
    reason_text = exclusion_reason.replace("_", " ") if exclusion_reason else ""
    return attach_structured_reason(
        {**dict(row), "exclusion_reason": exclusion_reason, "reason_code": code},
        stage="C3",
        status=status,
        eligible=eligible,
        reason_code=code,
        reason=reason_text,
        required_evidence=required_evidence,
        observed_evidence=observed_evidence,
        specification_id="option_c_near_zero_peak",
    )


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
) -> tuple[float, float, float]:
    """Initial (A, mu, sigma) for central baseline-adjusted Gaussian component."""
    peak_idx = int(np.argmax(z_values))
    peak_height = float(max(0.0, z_values[peak_idx]))
    mu0 = float(np.clip(lags[peak_idx], -CENTRAL_LAG_MAX_S, CENTRAL_LAG_MAX_S))
    sigma0 = DEFAULT_SIGMA0_S
    if peak_height <= 0:
        mu0 = 0.0
    return peak_height, mu0, sigma0


def _fit_weighted_linear_baseline(
    lags: np.ndarray,
    z_values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Weighted least-squares fit of b0 + b1*lag on flank rows."""
    if lags.size < 2:
        return float(np.median(z_values)), 0.0
    X = np.column_stack((np.ones_like(lags), lags))
    sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))
    Xw = X * sqrt_w[:, None]
    yw = z_values * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    b0 = float(beta[0])
    b1 = float(beta[1]) if beta.shape[0] > 1 else 0.0
    return b0, b1


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
    """Fit one weighted near-zero Gaussian peak to a Fisher-z lag curve."""
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

    empty = _empty_fit_result(n_fit=n_fit)

    if n_fit < MIN_FINITE_POINTS:
        empty["exclusion_reason"] = EXCLUSION_FIT_NOT_ATTEMPTED
        empty["optimizer_message"] = (
            "Gaussian fit not attempted: need at least 4 finite weighted lag points."
        )
        return empty

    # Baseline from distant flanks: 20 <= |tau| <= 60. No reduced-flank or
    # median-z fallback — Option C is undefined without this support.
    flank_mask = (
        np.abs(lags_f) >= FLANK_INNER_S
    ) & (
        np.abs(lags_f) <= FLANK_OUTER_S
    )
    n_flank = int(np.sum(flank_mask))
    if n_flank < MIN_FLANK_POINTS:
        empty["exclusion_reason"] = EXCLUSION_INSUFFICIENT_FLANKS
        empty["optimizer_message"] = (
            "Option C not attempted: need at least "
            f"{MIN_FLANK_POINTS} finite weighted flank lags with "
            f"{FLANK_INNER_S:.0f}<=|tau|<={FLANK_OUTER_S:.0f}."
        )
        return empty
    b0, b1 = _fit_weighted_linear_baseline(
        lags_f[flank_mask],
        z_f[flank_mask],
        w_f[flank_mask],
    )

    # Central window fit domain: |tau| <= 20.
    central_mask = np.abs(lags_f) <= CENTRAL_LAG_MAX_S
    lags_c = lags_f[central_mask]
    z_c = z_f[central_mask]
    w_c = w_f[central_mask]
    n_central = int(lags_c.size)
    if n_central < MIN_FINITE_POINTS:
        empty["n_lags_fit"] = n_central
        empty["baseline_C"] = b0
        empty["init_baseline_C"] = b0
        empty["exclusion_reason"] = EXCLUSION_FIT_NOT_ATTEMPTED
        empty["optimizer_message"] = (
            "Gaussian fit not attempted: need at least 4 finite weighted central lag points."
        )
        return empty

    baseline_c = b0 + b1 * lags_c
    z_center_adjusted = z_c - baseline_c

    a0, mu0, sigma0 = _initial_guess(lags_c, z_center_adjusted)
    empty.update(
        {
            "init_baseline_C": b0,
            "init_peak_height_A": a0,
            "init_peak_center_mu_s": mu0,
            "init_sigma_s": sigma0,
            "baseline_C": b0,
            "n_lags_fit": n_central,
        }
    )

    # curve_fit sigma is y-std; larger weight → smaller sigma.
    y_sigma = 1.0 / np.sqrt(np.clip(w_c, 1e-12, None))
    bounds_lower = [0.0, -MU_BOUND_S, SIGMA_MIN_S]
    bounds_upper = [np.inf, MU_BOUND_S, SIGMA_MAX_S]

    try:
        popt, pcov = curve_fit(
            _gaussian_component,
            lags_c,
            z_center_adjusted,
            p0=(a0, mu0, sigma0),
            bounds=(bounds_lower, bounds_upper),
            sigma=y_sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
        message = (
            "curve_fit converged (flank baseline on 20<=|tau|<=60; "
            "Gaussian on |tau|<=20 baseline-adjusted data)"
        )
        success = True
    except (RuntimeError, ValueError) as exc:
        empty["exclusion_reason"] = EXCLUSION_FIT_FAILED
        empty["optimizer_message"] = str(exc)
        return empty

    peak_height_A, peak_center_mu_s, sigma_s = (float(v) for v in popt)
    se = np.full(3, np.nan, dtype=float)
    if pcov is not None:
        diag = np.diag(np.asarray(pcov, dtype=float))
        if diag.shape == (3,) and np.all(np.isfinite(diag)) and np.all(diag >= 0):
            se = np.sqrt(diag)

    fitted = _gaussian_component(lags_c, peak_height_A, peak_center_mu_s, sigma_s)
    resid = z_center_adjusted - fitted
    weighted_rss = float(np.sum(w_c * resid * resid))
    rmse = float(np.sqrt(np.mean(resid * resid)))

    a_at_lower = peak_height_A <= BOUNDARY_ATOL
    mu_at_bound = _near_bound(peak_center_mu_s, -MU_BOUND_S, MU_BOUND_S)
    sigma_at_bound = _near_bound(sigma_s, SIGMA_MIN_S, SIGMA_MAX_S)
    boundary_hit = bool(a_at_lower or mu_at_bound or sigma_at_bound)

    # Positive identifiable peak: height above absolute floor and residual noise.
    amplitude_ok = peak_height_A >= MIN_IDENTIFIABLE_A and peak_height_A >= (
        IDENTIFIABLE_A_OVER_RMSE * rmse
    )
    se_A = float(se[0])
    if math.isfinite(se_A) and se_A > 0:
        amplitude_ok = amplitude_ok and peak_height_A >= (2.0 * se_A)
    # Reject weak edge artifacts when mu sits on the ±20 s wall.
    if mu_at_bound and peak_height_A < 0.2:
        amplitude_ok = False

    has_peak = bool(success and amplitude_ok)
    report_timing = bool(has_peak)
    reported_mu = peak_center_mu_s if report_timing else float("nan")
    reported_se_mu = float(se[1]) if report_timing and math.isfinite(float(se[1])) else float("nan")

    return {
        "converged": True,
        "has_identifiable_peak": has_peak,
        "report_timing_shift": report_timing,
        "baseline_C": b0,
        "peak_height_A": peak_height_A if has_peak else float("nan"),
        # Timing-report μ remains NaN when the peak is not identifiable.
        "peak_center_mu_s": reported_mu,
        "sigma_s": sigma_s if has_peak else float("nan"),
        "fwhm_s": fwhm_from_sigma(sigma_s) if has_peak else float("nan"),
        # Always retain the optimizer μ for comparison with the observed peak.
        "peak_lag_fitted_s": peak_center_mu_s,
        # Baseline slope/intercept uncertainty is not currently exported.
        "se_baseline_C": float("nan"),
        "se_peak_height_A": float(se[0]),
        "se_peak_center_mu_s": reported_se_mu,
        "se_sigma_s": float(se[2]),
        "rmse": rmse,
        "weighted_rss": weighted_rss,
        "n_lags_fit": n_central,
        "A_at_lower_bound": a_at_lower,
        "mu_at_bound": mu_at_bound,
        "sigma_at_bound": sigma_at_bound,
        "boundary_hit": boundary_hit,
        "fit_success": True,
        "optimizer_message": message,
        "init_baseline_C": b0,
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
    """Fit a peak model for one observation × band × representation curve.

    Option C (20≤|τ|≤60 baseline + |τ|≤20 Gaussian) is computed only for
    standard ZLPI durations (D180/D240). Other durations are exported as
    ``not_computable`` without A/μ/FWHM.
    """
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

    if curve_rows:
        first = curve_rows[0]
        identity = _identity_from_row(first)
        duration_role = str(first.get("duration_role", "")).strip()
        band = str(first.get("band", "")).strip().casefold()
        representation = str(first.get("power_representation", "")).strip()
        pair = str(first.get("pair", "")).strip()
        is_primary = bool(first.get("is_primary_representation", False))
        ordered = sorted(curve_rows, key=lambda row: float(row["lag_s"]))
        lags = np.asarray([float(row["lag_s"]) for row in ordered], dtype=float)
        r_values = np.asarray([float(row["r"]) for row in ordered], dtype=float)
        overlaps = np.asarray(
            [float(row.get("n_overlap", 0)) for row in ordered], dtype=float
        )
        peak_r, peak_lag_observed_s = _observed_peak_from_r(lags, r_values)
        finite_overlap = overlaps[np.isfinite(overlaps) & (overlaps > 0)]
        overlap_is_constant = (
            bool(finite_overlap.size)
            and float(np.max(finite_overlap) - np.min(finite_overlap)) <= 1e-12
        )
        n_common_support = (
            int(round(float(np.min(finite_overlap)))) if finite_overlap.size else 0
        )
        n_lags_input = int(lags.size)
    else:
        identity = {field: "" for field in IDENTITY_FIELDS}
        duration_role = ""
        band = ""
        representation = ""
        pair = ""
        is_primary = False
        lags = np.asarray([], dtype=float)
        r_values = np.asarray([], dtype=float)
        overlaps = np.asarray([], dtype=float)
        peak_r, peak_lag_observed_s = float("nan"), float("nan")
        overlap_is_constant = False
        n_common_support = 0
        n_lags_input = 0

    if not contract.is_standard_zlpi:
        fit = _empty_fit_result(exclusion_reason=EXCLUSION_OPTION_C_NOT_APPLICABLE)
        fit["optimizer_message"] = (
            "Option C peak fit is defined only for standard ZLPI durations "
            "(D180/D240) with 20<=|tau|<=60 flank baseline; "
            f"D{duration_s} {contract.endpoint_alias} is not this estimand."
        )
        required = (
            "duration_s in {180, 240}; Option C flanks 20<=|tau|<=60; "
            "A, mu, FWHM under this estimand"
        )
        observed = (
            f"duration_s={duration_s}; endpoint_name={contract.endpoint_name}; "
            f"lag_max_s={contract.lag_max_s}"
        )
    else:
        z_values = np.asarray([fisher_z(float(r)) for r in r_values], dtype=float)
        fit = fit_gaussian_peak(lags, z_values, weights=overlaps if lags.size else None)
        exclusion = str(fit.get("exclusion_reason") or "")
        if exclusion == EXCLUSION_INSUFFICIENT_FLANKS:
            required = f"at least {MIN_FLANK_POINTS} finite weighted lags with 20<=|tau|<=60"
            observed = f"duration_s={duration_s}; n_lags_input={n_lags_input}"
        elif exclusion:
            required = "identifiable Option C peak on D180/D240 Fisher-z curve"
            observed = (
                f"duration_s={duration_s}; exclusion={exclusion}; "
                f"n_lags_fit={fit.get('n_lags_fit', 0)}"
            )
        else:
            required = ""
            observed = ""

    params = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": duration_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "is_standard_zlpi": bool(contract.is_standard_zlpi),
        "band": band,
        "power_representation": representation,
        "is_primary_representation": is_primary,
        "pair": pair,
        "peak_r": peak_r,
        "peak_lag_observed_s": peak_lag_observed_s,
        "n_common_support": n_common_support,
        **{key: fit[key] for key in fit_param_keys},
    }
    # Keep optimizer diagnostics in QC; manuscript-facing params stay blank when
    # the peak is not identifiable.
    if not bool(fit.get("has_identifiable_peak")):
        params["peak_lag_fitted_s"] = float("nan")
        params["se_peak_height_A"] = float("nan")
        params["se_peak_center_mu_s"] = float("nan")
        params["se_sigma_s"] = float("nan")
    qc = {
        **identity,
        "duration_s": int(duration_s),
        "duration_role": duration_role,
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": contract.endpoint_alias,
        "band": band,
        "power_representation": representation,
        "pair": pair,
        "n_lags_input": n_lags_input,
        "n_common_support": n_common_support,
        "overlap_is_constant": overlap_is_constant,
        **{key: fit[key] for key in qc_fit_keys},
    }
    exclusion_reason = str(fit.get("exclusion_reason") or "")
    params = _with_peak_reason(
        params,
        exclusion_reason=exclusion_reason,
        required_evidence=required,
        observed_evidence=observed,
    )
    qc = _with_peak_reason(
        qc,
        exclusion_reason=exclusion_reason,
        required_evidence=required,
        observed_evidence=observed,
    )
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


def _csv_cell(value: object) -> object:
    """Blank non-finite floats so unsupported A/μ/FWHM are not exported as ``nan``."""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value if value is not None else ""


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_cell(row.get(field, "")) for field in fieldnames}
            )


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
                writer.writerow(
                    {field: _csv_cell(row.get(field, "")) for field in fields}
                )

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
    "EXCLUSION_INSUFFICIENT_FLANKS",
    "EXCLUSION_NO_IDENTIFIABLE_PEAK",
    "EXCLUSION_OPTION_C_NOT_APPLICABLE",
    "FWHM_FACTOR",
    "IDENTIFIABLE_A_OVER_RMSE",
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
