"""Cluster-aware inference for nested paired ΔZLPI (Figure 2 Panel A).

Observation-level points (e.g. participant × contrast) may be plotted, but
summary means and confidence intervals must use the independent sampling unit
(typically participant). Repeated contrasts/sessions nested within a participant
are never treated as independent draws.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

DEFAULT_CLUSTER_BOOTSTRAP_DRAWS = 2000
# Mixed models for a single-dataset Panel A slice need a realistic unit count.
MIN_UNITS_FOR_MIXED_MODEL = 6
MIN_WITHIN_UNIT_OBS_FOR_MIXED = 2


@dataclass(frozen=True)
class IndependentUnitDecision:
    """Automatic choice of the independent sampling unit for a row set."""

    unit_field: str
    unit_label: str
    n_observations: int
    n_units: int
    nested_repeated_measures: bool
    reason: str


@dataclass(frozen=True)
class UnitSummary:
    unit_id: str
    n_observations: int
    mean_delta: float


@dataclass(frozen=True)
class ClusterAwarePairedInference:
    """Primary participant/unit-level inference plus diagnostics."""

    estimand: str
    unit_field: str
    unit_label: str
    n_observations: int
    n_units: int
    nested_repeated_measures: bool
    unit_detection_reason: str
    mean_delta: float
    se_delta: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    df: int
    ci_method: str
    cluster_bootstrap_mean: float
    cluster_bootstrap_ci_low: float
    cluster_bootstrap_ci_high: float
    cluster_bootstrap_n_draws: int
    bootstrap_agrees_with_unit_ci: bool
    mixed_model_appropriate: bool
    mixed_model_recommendation: str
    mixed_model_status: str
    mixed_model_mean: float
    mixed_model_ci_low: float
    mixed_model_ci_high: float
    mixed_model_p_value: float
    notes: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


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


def observation_unit_id(row: Mapping[str, object], *, unit_field: str) -> str:
    """Stable independent-unit key (dataset-scoped when dataset_id is present)."""
    unit = _as_str(row.get(unit_field))
    if not unit:
        # Last-resort identity so missing IDs cannot silently collapse units.
        unit = (
            f"row::{_as_str(row.get('participant_id'))}::"
            f"{_as_str(row.get('contrast_id'))}::{_as_str(row.get('session_id'))}"
        )
    dataset = _as_str(row.get("dataset_id")).casefold()
    return f"{dataset}::{unit}" if dataset else unit


def infer_independent_sampling_unit(
    rows: Sequence[Mapping[str, object]],
) -> IndependentUnitDecision:
    """Choose the independent sampling unit without dataset-specific branches.

    Confirmatory paired contrasts sample *participants*. Multiple contrasts or
    sessions contributed by the same participant are repeated measures. If each
    participant contributes exactly one finite Δ, the observation is already
    the independent unit.
    """
    finite = [
        r
        for r in rows
        if math.isfinite(_as_float(r.get("delta_endpoint_index")))
    ]
    n_obs = len(finite)
    if n_obs == 0:
        return IndependentUnitDecision(
            unit_field="participant_id",
            unit_label="participant",
            n_observations=0,
            n_units=0,
            nested_repeated_measures=False,
            reason="no_finite_deltas",
        )

    n_participants = len(
        {
            observation_unit_id(r, unit_field="participant_id")
            for r in finite
            if _as_str(r.get("participant_id"))
        }
    )
    # If participant_id is missing on all rows, fall back to row identity.
    if n_participants == 0:
        return IndependentUnitDecision(
            unit_field="observation",
            unit_label="observation",
            n_observations=n_obs,
            n_units=n_obs,
            nested_repeated_measures=False,
            reason="missing_participant_id_fallback_to_rows",
        )

    nested = n_obs > n_participants
    reason = (
        "multiple_observations_per_participant"
        if nested
        else "one_observation_per_participant"
    )
    # Sessions nested in participants still do not create independent units for
    # population inference; participant remains the sampling unit.
    n_sessions = len(
        {
            (
                observation_unit_id(r, unit_field="participant_id"),
                _as_str(r.get("session_id"), "single"),
            )
            for r in finite
        }
    )
    if nested and n_sessions > n_participants:
        reason = "contrasts_and_or_sessions_nested_in_participants"

    return IndependentUnitDecision(
        unit_field="participant_id",
        unit_label="participant",
        n_observations=n_obs,
        n_units=n_participants,
        nested_repeated_measures=nested,
        reason=reason,
    )


def summarize_by_independent_unit(
    rows: Sequence[Mapping[str, object]],
    *,
    unit_field: str = "participant_id",
) -> list[UnitSummary]:
    """Mean Δ within each independent unit.

    Downstream inference uses the **unweighted mean of these unit means**, so
    every participant (or other independent unit) receives equal inferential
    weight even when the number of nested contrasts differs. Observation-count
    weighting is not applied unless a caller explicitly chooses
    ``estimand='mean_of_observations'`` for a diagnostic bootstrap.
    """
    buckets: dict[str, list[float]] = {}
    for row in rows:
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        unit_id = observation_unit_id(row, unit_field=unit_field)
        buckets.setdefault(unit_id, []).append(delta)
    summaries = [
        UnitSummary(
            unit_id=unit_id,
            n_observations=len(values),
            mean_delta=float(np.mean(values)),
        )
        for unit_id, values in sorted(buckets.items())
    ]
    return summaries


def _student_t_mean_ci(
    values: Sequence[float],
) -> tuple[float, float, float, float, float, int]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    n = int(arr.size)
    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )
    mean = float(np.mean(arr))
    if n == 1:
        return mean, float("nan"), float("nan"), float("nan"), float("nan"), 0
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    t_stat, p_value = stats.ttest_1samp(arr, popmean=0.0)
    return mean, se, mean - t_crit * se, mean + t_crit * se, float(t_stat), df


def cluster_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    unit_field: str = "participant_id",
    n_draws: int = DEFAULT_CLUSTER_BOOTSTRAP_DRAWS,
    seed: int = 0,
    estimand: str = "mean_of_unit_means",
) -> tuple[float, float, float]:
    """Cluster bootstrap CI by resampling independent units with replacement.

    Nested observations belonging to a drawn unit are retained. The default
    estimand matches primary inference: mean of within-unit means (equal unit
    weight). ``mean_of_observations`` weights units by their nested row counts.
    """
    summaries = summarize_by_independent_unit(rows, unit_field=unit_field)
    if not summaries:
        return float("nan"), float("nan"), float("nan")

    unit_means = np.asarray([s.mean_delta for s in summaries], dtype=float)
    unit_ns = np.asarray([s.n_observations for s in summaries], dtype=float)
    n_units = int(unit_means.size)
    if estimand == "mean_of_observations":
        point = float(np.average(unit_means, weights=unit_ns))
    else:
        point = float(np.mean(unit_means))
    if n_units < 2:
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    boots = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        idx = rng.integers(0, n_units, size=n_units)
        if estimand == "mean_of_observations":
            means = unit_means[idx]
            weights = unit_ns[idx]
            boots[i] = float(np.average(means, weights=weights))
        else:
            boots[i] = float(np.mean(unit_means[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _cis_agree(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
    *,
    atol: float = 1e-9,
    rtol_width: float = 0.5,
) -> bool:
    """Loose agreement: both include/exclude 0 the same way and overlap substantially."""
    if not all(math.isfinite(v) for v in (low_a, high_a, low_b, high_b)):
        return False
    covers_zero_a = low_a <= 0.0 <= high_a
    covers_zero_b = low_b <= 0.0 <= high_b
    if covers_zero_a != covers_zero_b:
        return False
    # Interval overlap relative to the wider width.
    overlap = max(0.0, min(high_a, high_b) - max(low_a, low_b))
    width = max(high_a - low_a, high_b - low_b, atol)
    return overlap + atol >= rtol_width * width


def evaluate_mixed_model_for_panel_a(
    rows: Sequence[Mapping[str, object]],
    *,
    unit_field: str = "participant_id",
) -> dict[str, object]:
    """Optionally fit a random-intercept model for nested Δ rows.

    Used only when nested repeated measures exist and the unit count is large
    enough to support a random effect. Small smoke cohorts are skipped.
    """
    decision = infer_independent_sampling_unit(rows)
    base = {
        "mixed_model_appropriate": False,
        "mixed_model_recommendation": "skip",
        "mixed_model_status": "not_attempted",
        "mixed_model_mean": float("nan"),
        "mixed_model_ci_low": float("nan"),
        "mixed_model_ci_high": float("nan"),
        "mixed_model_p_value": float("nan"),
        "notes": "",
    }
    if decision.n_units < MIN_UNITS_FOR_MIXED_MODEL:
        base["mixed_model_recommendation"] = "insufficient_units"
        base["mixed_model_status"] = "skipped_insufficient_units"
        base["notes"] = (
            f"Need ≥{MIN_UNITS_FOR_MIXED_MODEL} independent "
            f"{decision.unit_label}s for Panel A mixed models."
        )
        return base
    if not decision.nested_repeated_measures:
        base["mixed_model_recommendation"] = "not_needed_no_nesting"
        base["mixed_model_status"] = "skipped_no_nesting"
        base["notes"] = "One observation per independent unit; unit-mean t-test is sufficient."
        return base

    summaries = summarize_by_independent_unit(rows, unit_field=unit_field)
    if not any(s.n_observations >= MIN_WITHIN_UNIT_OBS_FOR_MIXED for s in summaries):
        base["mixed_model_recommendation"] = "insufficient_within_unit_replication"
        base["mixed_model_status"] = "skipped_no_within_unit_replication"
        return base

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        base["mixed_model_status"] = "skipped_missing_statsmodels"
        base["notes"] = "statsmodels unavailable."
        return base

    records = []
    for row in rows:
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        records.append(
            {
                "delta": delta,
                "unit_id": observation_unit_id(row, unit_field=unit_field),
                "contrast_id": _as_str(row.get("contrast_id"), "contrast"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    try:
        # Random intercept for independent units; fixed intercept = mean Δ.
        model = smf.mixedlm("delta ~ 1", data=frame, groups=frame["unit_id"])
        fitted = model.fit(method=["lbfgs"], reml=True, maxiter=200, disp=False)
        mean = float(fitted.fe_params["Intercept"])
        # Prefer Wald CI on the intercept when available.
        conf = fitted.conf_int()
        lo = float(conf.loc["Intercept", 0])
        hi = float(conf.loc["Intercept", 1])
        p_value = float(fitted.pvalues["Intercept"])
        base.update(
            {
                "mixed_model_appropriate": True,
                "mixed_model_recommendation": "optional_sensitivity",
                "mixed_model_status": (
                    "ok" if bool(getattr(fitted, "converged", False)) else "fit_unconverged"
                ),
                "mixed_model_mean": mean,
                "mixed_model_ci_low": lo,
                "mixed_model_ci_high": hi,
                "mixed_model_p_value": p_value,
                "notes": "Random-intercept MixedLM on observation-level Δ; groups=independent units.",
            }
        )
    except Exception as exc:  # noqa: BLE001
        base["mixed_model_status"] = f"failed:{type(exc).__name__}"
        base["notes"] = str(exc)
    return base


def infer_paired_deltas_cluster_aware(
    rows: Sequence[Mapping[str, object]],
    *,
    n_bootstrap: int = DEFAULT_CLUSTER_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = 0,
) -> tuple[ClusterAwarePairedInference, list[UnitSummary]]:
    """Primary mean/CI from independent-unit means; bootstrap for confirmation."""
    decision = infer_independent_sampling_unit(rows)
    unit_field = "participant_id"
    working_rows: list[Mapping[str, object]] = list(rows)
    if decision.unit_field == "observation":
        # Synthetic unit per row using a stable composite key.
        enriched: list[dict[str, object]] = []
        for idx, row in enumerate(rows):
            item = dict(row)
            item["participant_id"] = (
                _as_str(row.get("participant_id"))
                or f"obs{idx}_{_as_str(row.get('contrast_id'))}"
            )
            enriched.append(item)
        working_rows = enriched

    summaries = summarize_by_independent_unit(working_rows, unit_field=unit_field)
    unit_means = [s.mean_delta for s in summaries]
    mean, se, lo, hi, t_stat, df = _student_t_mean_ci(unit_means)
    p_value = float("nan")
    if len(unit_means) >= 2:
        _, p_value = stats.ttest_1samp(
            np.asarray(unit_means, dtype=float), popmean=0.0
        )
        p_value = float(p_value)

    boot_mean, boot_lo, boot_hi = cluster_bootstrap_mean_ci(
        working_rows,
        unit_field=unit_field,
        n_draws=n_bootstrap,
        seed=bootstrap_seed,
        estimand="mean_of_unit_means",
    )
    mixed = evaluate_mixed_model_for_panel_a(working_rows, unit_field=unit_field)
    # Percentile cluster bootstrap undercovers with very small n_units; Student-t
    # on unit means remains primary. Agreement is only decisive for n_units≥5.
    if decision.n_units < 5:
        agrees = True
        bootstrap_note = (
            "bootstrap_vs_t: not decisive for n_units<5 "
            "(percentile cluster bootstrap often narrower than t CI)."
        )
    else:
        agrees = _cis_agree(lo, hi, boot_lo, boot_hi)
        bootstrap_note = (
            "bootstrap_vs_t: intervals agree on zero-coverage and overlap"
            if agrees
            else "bootstrap_vs_t: intervals disagree — inspect design carefully"
        )

    notes = [
        decision.reason,
        (
            "Primary estimand = mean of within-unit mean Δ "
            f"(equal weight per {decision.unit_label})."
        ),
        (
            "SE/CI use n_units="
            f"{decision.n_units}, not n_observations={decision.n_observations}."
        ),
        bootstrap_note,
    ]
    if mixed.get("notes"):
        notes.append(str(mixed["notes"]))

    result = ClusterAwarePairedInference(
        estimand="mean_of_unit_means",
        unit_field=decision.unit_field,
        unit_label=decision.unit_label,
        n_observations=decision.n_observations,
        n_units=decision.n_units,
        nested_repeated_measures=decision.nested_repeated_measures,
        unit_detection_reason=decision.reason,
        mean_delta=mean,
        se_delta=se,
        ci_low=lo,
        ci_high=hi,
        t_stat=t_stat if len(unit_means) >= 2 else float("nan"),
        p_value=p_value,
        df=df,
        ci_method="student_t_on_independent_unit_means",
        cluster_bootstrap_mean=boot_mean,
        cluster_bootstrap_ci_low=boot_lo,
        cluster_bootstrap_ci_high=boot_hi,
        cluster_bootstrap_n_draws=int(n_bootstrap) if decision.n_units >= 2 else 0,
        bootstrap_agrees_with_unit_ci=bool(agrees),
        mixed_model_appropriate=bool(mixed["mixed_model_appropriate"]),
        mixed_model_recommendation=str(mixed["mixed_model_recommendation"]),
        mixed_model_status=str(mixed["mixed_model_status"]),
        mixed_model_mean=float(mixed["mixed_model_mean"]),
        mixed_model_ci_low=float(mixed["mixed_model_ci_low"]),
        mixed_model_ci_high=float(mixed["mixed_model_ci_high"]),
        mixed_model_p_value=float(mixed["mixed_model_p_value"]),
        notes="; ".join(notes),
    )
    return result, summaries


__all__ = [
    "DEFAULT_CLUSTER_BOOTSTRAP_DRAWS",
    "MIN_UNITS_FOR_MIXED_MODEL",
    "ClusterAwarePairedInference",
    "IndependentUnitDecision",
    "UnitSummary",
    "cluster_bootstrap_mean_ci",
    "evaluate_mixed_model_for_panel_a",
    "infer_independent_sampling_unit",
    "infer_paired_deltas_cluster_aware",
    "observation_unit_id",
    "summarize_by_independent_unit",
]
