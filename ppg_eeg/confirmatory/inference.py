"""Confirmatory statistical inference (M10).

Primary ZLPI mixed models, paired dataset effects, random-effects
meta-analysis (with leave-one-dataset-out), peak-center TOST equivalence,
and BH-FDR across prespecified families. ZLPI, MWPI, and SWPI stay separate;
D120/D60/MWPI/SWPI never promote or rescue primary ZLPI decisions.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW

from .duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
)
from .group_tables import (
    PAIRED_CONTRASTS_FILENAME,
    SUBJECT_LEVEL_FILENAME,
)
from .protocol_audit import PROTOCOL_SPECS

MIXED_MODEL_RESULTS_FILENAME = "mixed_model_results.csv"
DATASET_EFFECTS_FILENAME = "dataset_effects.csv"
META_ANALYSIS_RESULTS_FILENAME = "meta_analysis_results.csv"
LEAVE_ONE_DATASET_OUT_FILENAME = "leave_one_dataset_out.csv"
PEAK_CENTER_EQUIVALENCE_FILENAME = "peak_center_equivalence.csv"
MULTIPLICITY_RESULTS_FILENAME = "multiplicity_results.csv"
INFERENCE_QC_FILENAME = "inference_qc.csv"

PRIMARY_POWER_REPRESENTATION = "absolute_log10"
FDR_ALPHA = 0.05
FDR_METHOD = "fdr_bh"
BAND_ORDER = ("delta", "theta", "alpha", "beta")

# Unpaired / single-state datasets excluded from task-attenuation meta-analysis.
META_EXCLUDED_DATASETS = frozenset({"ds003816", "ds004582"})

# Prespecified primary paired datasets and state-attenuation contrasts.
PRIMARY_PAIRED_DATASETS = (
    "ds003838",
    "ds006848",
    "ds003690",
    "ds004587",
)
PRIMARY_STATE_CONTRASTS = (
    "rest__memory",
    "rest__verbalwm",
    "passive__simplert",
    "passive__gonogo",
    "rest__ig",
)

FAMILY_PRIMARY_LOW_DEMAND_ZLPI = "primary_low_demand_zlpi"
FAMILY_PRIMARY_STATE_ATTENUATION = "primary_state_attenuation"
FAMILY_META_BAND = "meta_analytic_band_effects"
FAMILY_MU_EQUIVALENCE = "mu_equivalence"
FAMILY_LOCAL_PROMINENCE = "local_prominence"

MU_LOW = -float(EXPECTED_PEAK_CENTER_EQUIVALENCE_S)
MU_UPP = float(EXPECTED_PEAK_CENTER_EQUIVALENCE_S)

MODEL_USED_RANDOM_EFFECTS = "random_effects"
MODEL_USED_FIXED_EFFECTS_FALLBACK = "fixed_effects_fallback"

ANALYSIS_STATUS_COMPLETED = "completed"
ANALYSIS_STATUS_SKIPPED_INSUFFICIENT = "skipped_insufficient_datasets"

MIXED_MODEL_FIELDS = (
    "endpoint_name",
    "duration_s",
    "power_representation",
    "is_primary_analysis",
    "model_backend",
    "model_used",
    "converged",
    "term",
    "coef",
    "stderr",
    "z_or_t",
    "p_value",
    "ci_low",
    "ci_high",
    "n_obs",
    "n_groups",
    "notes",
)

DATASET_EFFECT_FIELDS = (
    "dataset_id",
    "contrast_id",
    "duration_s",
    "endpoint_name",
    "is_standard_zlpi",
    "band",
    "power_representation",
    "is_primary_analysis",
    "n_pairs",
    "effect_mean",
    "effect_sd",
    "effect_se",
    "effect_var",
    "t_stat",
    "p_value",
    "ci_low",
    "ci_high",
    "enters_meta",
)

META_FIELDS = (
    "endpoint_name",
    "duration_s",
    "band",
    "power_representation",
    "is_primary_analysis",
    "analysis_status",
    "estimator",
    "n_datasets",
    "pooled_effect",
    "ci_low",
    "ci_high",
    "prediction_low",
    "prediction_high",
    "q",
    "tau2",
    "i2",
    "p_value",
    "dataset_ids",
    "notes",
)

LOO_FIELDS = (
    "endpoint_name",
    "duration_s",
    "band",
    "power_representation",
    "analysis_status",
    "omitted_dataset_id",
    "estimator",
    "n_datasets",
    "pooled_effect",
    "ci_low",
    "ci_high",
    "prediction_low",
    "prediction_high",
    "q",
    "tau2",
    "i2",
    "p_value",
    "delta_vs_full",
    "notes",
)

EQUIVALENCE_FIELDS = (
    "dataset_id",
    "endpoint_name",
    "duration_s",
    "band",
    "power_representation",
    "condition_role",
    "n",
    "mean_mu",
    "se_mu",
    "ci_low",
    "ci_high",
    "bound_low",
    "bound_high",
    "tost_p_lower",
    "tost_p_upper",
    "tost_p",
    "equivalent",
    "notes",
)

MULTIPLICITY_FIELDS = (
    "family_id",
    "endpoint_name",
    "duration_s",
    "band",
    "power_representation",
    "dataset_id",
    "contrast_id",
    "test_id",
    "raw_p",
    "q_value",
    "reject_fdr",
    "effect",
    "notes",
)

QC_FIELDS = (
    "component",
    "endpoint_name",
    "duration_s",
    "status",
    "model_backend",
    "converged",
    "n_obs",
    "n_groups",
    "notes",
)


@dataclass(frozen=True)
class InferenceResult:
    mixed_model_rows: tuple[dict[str, object], ...]
    dataset_effect_rows: tuple[dict[str, object], ...]
    meta_rows: tuple[dict[str, object], ...]
    loo_rows: tuple[dict[str, object], ...]
    equivalence_rows: tuple[dict[str, object], ...]
    multiplicity_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return bool(value)


def _as_int(value: object, default: int = 0) -> int:
    if value is None or _as_str(value) == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def bh_fdr(p_values: Sequence[float], *, alpha: float = FDR_ALPHA) -> list[float]:
    """Benjamini–Hochberg q-values; non-finite inputs stay NaN."""
    arr = np.asarray(list(p_values), dtype=float)
    q_values = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if int(valid.sum()) == 0:
        return q_values.tolist()
    _, qvals, _, _ = multipletests(arr[valid], alpha=alpha, method=FDR_METHOD)
    q_values[valid] = qvals
    return q_values.tolist()


def prediction_interval(
    pooled_effect: float,
    var_pooled: float,
    tau2: float,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Approximate 100(1-α)% prediction interval for a new study effect."""
    if not (math.isfinite(pooled_effect) and math.isfinite(var_pooled) and math.isfinite(tau2)):
        return float("nan"), float("nan")
    se = math.sqrt(max(0.0, float(tau2)) + max(0.0, float(var_pooled)))
    if se <= 0:
        return float(pooled_effect), float(pooled_effect)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(pooled_effect - z * se), float(pooled_effect + z * se)


def model_used_from_backend(backend: str) -> str:
    """Coarse model class for reporting (distinct from technical ``model_backend``)."""
    text = _as_str(backend).casefold()
    if text.startswith("mixedlm"):
        return MODEL_USED_RANDOM_EFFECTS
    if text.startswith("ols"):
        return MODEL_USED_FIXED_EFFECTS_FALLBACK
    return ""


def clip_i2(i2: float) -> float:
    """Clip statsmodels I² (may be negative when Q < df) to [0, 1]."""
    if not math.isfinite(i2):
        return float("nan")
    # statsmodels returns I² on a proportion-like scale that can be negative.
    return float(min(1.0, max(0.0, i2)))


def random_effects_meta(
    effects: Sequence[float],
    variances: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Random-effects meta-analysis via Paule–Mandel (iterated), DL fallback."""
    effect = np.asarray(list(effects), dtype=float)
    variance = np.asarray(list(variances), dtype=float)
    mask = np.isfinite(effect) & np.isfinite(variance) & (variance > 0)
    effect = effect[mask]
    variance = variance[mask]
    names = None
    if labels is not None:
        names = [str(labels[i]) for i, keep in enumerate(mask) if keep]
    if effect.size < 2:
        return {
            "estimator": "insufficient_studies",
            "analysis_status": ANALYSIS_STATUS_SKIPPED_INSUFFICIENT,
            "n_datasets": int(effect.size),
            "pooled_effect": float(effect[0]) if effect.size == 1 else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "prediction_low": float("nan"),
            "prediction_high": float("nan"),
            "q": float("nan"),
            "tau2": float("nan"),
            "i2": float("nan"),
            "p_value": float("nan"),
            "var_pooled": float("nan"),
            "dataset_ids": ";".join(names or []),
            "notes": "Need ≥2 studies with finite positive sampling variances.",
        }

    estimator = "paule_mandel"
    notes = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = combine_effects(
                effect,
                variance,
                method_re="iterated",
                row_names=names,
                alpha=alpha,
            )
    except Exception as exc:  # noqa: BLE001 — deterministic fallback path
        estimator = "der_simonian_laird"
        notes = f"Paule-Mandel failed ({exc}); using DerSimonian-Laird."
        result = combine_effects(
            effect,
            variance,
            method_re="chi2",
            row_names=names,
            alpha=alpha,
        )

    pooled = float(result.mean_effect_re)
    var_pooled = float(result.var_eff_w_re)
    sd_pooled = float(result.sd_eff_w_re)
    z = pooled / sd_pooled if sd_pooled > 0 else float("nan")
    p_value = (
        float(2.0 * stats.norm.sf(abs(z))) if math.isfinite(z) else float("nan")
    )
    cis = result.conf_int(alpha=alpha)
    # conf_int returns (fe_normal, re_normal, fe_t, re_t) pairs of [low, high].
    re_ci = cis[1]
    ci_low, ci_high = float(re_ci[0]), float(re_ci[1])
    tau2 = float(result.tau2)
    pred_low, pred_high = prediction_interval(pooled, var_pooled, tau2, alpha=alpha)
    return {
        "estimator": estimator,
        "analysis_status": ANALYSIS_STATUS_COMPLETED,
        "n_datasets": int(effect.size),
        "pooled_effect": pooled,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": pred_low,
        "prediction_high": pred_high,
        "q": float(result.q),
        "tau2": tau2,
        "i2": clip_i2(float(result.i2)),
        "p_value": p_value,
        "var_pooled": var_pooled,
        "dataset_ids": ";".join(names or []),
        "notes": notes,
    }


def _condition_role(dataset_id: str, condition: str) -> str:
    """Map a condition label to low_demand / cognitive_effort / other."""
    key = dataset_id.casefold()
    cond = condition.casefold()
    spec = PROTOCOL_SPECS.get(key)
    if spec is None:
        return "other"
    if cond in {c.casefold() for c in spec.low_demand_conditions}:
        return "low_demand"
    if cond in {c.casefold() for c in spec.cognitive_effort_conditions}:
        return "cognitive_effort"
    return "other"


def _is_primary_analysis(
    *,
    endpoint_name: str,
    duration_s: int,
    power_representation: str,
) -> bool:
    return (
        endpoint_name.casefold() == ENDPOINT_ZLPI
        and int(duration_s) == int(EXPECTED_PRIMARY_DURATION_S)
        and power_representation.casefold() == PRIMARY_POWER_REPRESENTATION
    )


def _prepare_subject_frame(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    endpoint_name: str,
    duration_s: int | None = None,
    power_representation: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in subject_rows:
        ep = _as_str(raw.get("endpoint_name")).casefold()
        if ep != endpoint_name.casefold():
            continue
        dur = _as_int(raw.get("duration_s"))
        if duration_s is not None and dur != int(duration_s):
            continue
        repr_name = _as_str(raw.get("power_representation")).casefold()
        if (
            power_representation is not None
            and repr_name != power_representation.casefold()
        ):
            continue
        if not _as_bool(raw.get("endpoint_eligible", True)):
            continue
        dataset_id = _as_str(raw.get("dataset_id")).casefold()
        condition = _as_str(raw.get("condition")).casefold()
        role = _as_str(raw.get("state") or raw.get("condition_role"))
        if not role:
            role = _condition_role(dataset_id, condition)
        if role not in {"low_demand", "cognitive_effort"}:
            # Allow explicit binary coding already present.
            if condition in {"low_demand", "cognitive_effort", "rest", "task"}:
                role = (
                    "low_demand"
                    if condition in {"low_demand", "rest"}
                    else "cognitive_effort"
                )
            else:
                continue
        mean_hr = _as_float(raw.get("mean_hr"))
        if not math.isfinite(mean_hr):
            mean_hr = _as_float(raw.get("mean_hr_bpm"))
        if not math.isfinite(mean_hr):
            mean_hr = 0.0
        modality = _as_str(raw.get("modality") or raw.get("sensor_modality"), "eeg")
        participant = _as_str(raw.get("participant_id") or raw.get("subject_id"))
        rows.append(
            {
                "dataset_id": dataset_id,
                "participant_id": participant,
                "participant_uid": f"{dataset_id}::{participant}",
                "session_id": _as_str(raw.get("session_id"), "single"),
                "condition": condition,
                "state": role,
                "modality": modality.casefold(),
                "mean_hr": float(mean_hr),
                "band": _as_str(raw.get("band")).casefold(),
                "endpoint_index": _as_float(raw.get("endpoint_index")),
                "local_prominence": _as_float(raw.get("local_prominence")),
                "peak_center_mu_s": _as_float(raw.get("peak_center_mu_s")),
                "has_identifiable_peak": _as_bool(raw.get("has_identifiable_peak")),
                "duration_s": dur,
                "endpoint_name": ep,
                "power_representation": repr_name,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame[np.isfinite(frame["endpoint_index"].to_numpy(dtype=float))].copy()
    return frame


def fit_mixed_model(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    endpoint_name: str = ENDPOINT_ZLPI,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
    power_representation: str = PRIMARY_POWER_REPRESENTATION,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Fit participant-level mixed model for one endpoint family.

    Fixed effects: state, band, state×band, modality, mean_hr.
    Random: participant (+ dataset VC), with deterministic fallbacks.
    """
    is_primary = _is_primary_analysis(
        endpoint_name=endpoint_name,
        duration_s=duration_s,
        power_representation=power_representation,
    )
    # Never allow MWPI/SWPI rows into a ZLPI model call — filter is by endpoint_name.
    frame = _prepare_subject_frame(
        subject_rows,
        endpoint_name=endpoint_name,
        duration_s=duration_s,
        power_representation=power_representation,
    )
    qc: dict[str, object] = {
        "component": "mixed_model",
        "endpoint_name": endpoint_name,
        "duration_s": int(duration_s),
        "status": "ok",
        "model_backend": "",
        "converged": False,
        "n_obs": int(len(frame)),
        "n_groups": 0,
        "notes": "",
    }
    if frame.empty or frame["participant_uid"].nunique() < 2:
        qc["status"] = "insufficient_data"
        qc["notes"] = "Need eligible rows from ≥2 participants."
        return [], qc

    # Ensure categorical levels exist.
    if frame["state"].nunique() < 2:
        qc["status"] = "insufficient_state_levels"
        qc["notes"] = "Need both low_demand and cognitive_effort observations."
        return [], qc
    if frame["band"].nunique() < 1:
        qc["status"] = "missing_bands"
        return [], qc

    formula_core = "endpoint_index ~ C(state) * C(band) + C(modality) + mean_hr"
    notes: list[str] = []
    fitted = None
    backend = ""

    # Fallback 1: subject RE + dataset variance component.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                formula_core,
                data=frame,
                groups=frame["participant_uid"],
                vc_formula={"dataset": "0 + C(dataset_id)"},
            )
            fitted = model.fit(method=["lbfgs"], reml=True, maxiter=200, disp=False)
        if bool(getattr(fitted, "converged", False)):
            backend = "mixedlm_subject_re_dataset_vc"
        else:
            notes.append("dataset_vc_not_converged")
            fitted = None
    except Exception as exc:  # noqa: BLE001
        notes.append(f"dataset_vc_failed:{type(exc).__name__}")
        fitted = None

    # Fallback 2: subject RE + dataset fixed effect.
    if fitted is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    formula_core + " + C(dataset_id)",
                    data=frame,
                    groups=frame["participant_uid"],
                )
                fitted = model.fit(method=["lbfgs"], reml=True, maxiter=200, disp=False)
            if bool(getattr(fitted, "converged", False)):
                backend = "mixedlm_subject_re_dataset_fe"
            else:
                notes.append("dataset_fe_mixedlm_not_converged")
                fitted = None
        except Exception as exc:  # noqa: BLE001
            notes.append(f"dataset_fe_mixedlm_failed:{type(exc).__name__}")
            fitted = None

    # Fallback 3: OLS with participant-clustered covariance.
    if fitted is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = smf.ols(
                    formula_core + " + C(dataset_id)",
                    data=frame,
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": frame["participant_uid"]},
                )
            backend = "ols_cluster_participant"
            notes.append("fell_back_to_ols_cluster")
        except Exception as exc:  # noqa: BLE001
            qc["status"] = "fit_failed"
            qc["notes"] = ";".join(notes + [f"ols_failed:{type(exc).__name__}:{exc}"])
            return [], qc

    qc["model_backend"] = backend
    qc["converged"] = bool(getattr(fitted, "converged", True))
    qc["n_groups"] = int(frame["participant_uid"].nunique())
    qc["notes"] = ";".join(notes)
    if backend.startswith("ols"):
        qc["status"] = "fallback_ols"
    elif not qc["converged"]:
        qc["status"] = "not_converged"
    else:
        qc["status"] = "ok"

    rows: list[dict[str, object]] = []
    params = fitted.params
    bse = fitted.bse
    pvalues = fitted.pvalues
    # Prefer conf_int when available.
    try:
        conf = fitted.conf_int()
    except Exception:  # noqa: BLE001
        conf = None
    for term in params.index:
        ci_low = float("nan")
        ci_high = float("nan")
        if conf is not None and term in conf.index:
            ci_low = float(conf.loc[term, 0])
            ci_high = float(conf.loc[term, 1])
        stderr = float(bse.get(term, float("nan")))
        coef = float(params.get(term, float("nan")))
        stat = coef / stderr if math.isfinite(stderr) and stderr > 0 else float("nan")
        rows.append(
            {
                "endpoint_name": endpoint_name,
                "duration_s": int(duration_s),
                "power_representation": power_representation,
                "is_primary_analysis": is_primary,
                "model_backend": backend,
                "model_used": model_used_from_backend(backend),
                "converged": bool(qc["converged"]),
                "term": str(term),
                "coef": coef,
                "stderr": stderr,
                "z_or_t": float(stat),
                "p_value": float(pvalues.get(term, float("nan"))),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_obs": int(len(frame)),
                "n_groups": int(qc["n_groups"]),
                "notes": qc["notes"],
            }
        )
    return rows, qc


def estimate_dataset_effects(
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Paired task-minus-low-demand effects and sampling variances per dataset cell."""
    buckets: dict[tuple[str, ...], list[float]] = {}
    meta_flags: dict[tuple[str, ...], dict[str, object]] = {}
    for raw in paired_rows:
        dataset_id = _as_str(raw.get("dataset_id")).casefold()
        contrast_id = _as_str(raw.get("contrast_id")).casefold()
        duration_s = _as_int(raw.get("duration_s"))
        endpoint_name = _as_str(raw.get("endpoint_name")).casefold()
        band = _as_str(raw.get("band")).casefold()
        representation = _as_str(raw.get("power_representation")).casefold()
        delta = _as_float(raw.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        key = (
            dataset_id,
            contrast_id,
            str(duration_s),
            endpoint_name,
            band,
            representation,
        )
        buckets.setdefault(key, []).append(float(delta))
        meta_flags[key] = {
            "is_standard_zlpi": _as_bool(raw.get("is_standard_zlpi"))
            or endpoint_name == ENDPOINT_ZLPI,
        }

    effects: list[dict[str, object]] = []
    for key in sorted(buckets):
        values = np.asarray(buckets[key], dtype=float)
        n = int(values.size)
        mean = float(np.mean(values))
        if n >= 2:
            sd = float(np.std(values, ddof=1))
            se = sd / math.sqrt(n)
            t_stat, p_value = stats.ttest_1samp(values, popmean=0.0)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
        else:
            sd = float("nan")
            se = float("nan")
            t_stat = float("nan")
            p_value = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
        dataset_id, contrast_id, duration_s_s, endpoint_name, band, representation = key
        duration_s = int(duration_s_s)
        is_primary = _is_primary_analysis(
            endpoint_name=endpoint_name,
            duration_s=duration_s,
            power_representation=representation,
        )
        enters_meta = (
            is_primary
            and dataset_id not in META_EXCLUDED_DATASETS
            and n >= 2
            and math.isfinite(se)
            and se > 0
        )
        effects.append(
            {
                "dataset_id": dataset_id,
                "contrast_id": contrast_id,
                "duration_s": duration_s,
                "endpoint_name": endpoint_name,
                "is_standard_zlpi": bool(meta_flags[key]["is_standard_zlpi"]),
                "band": band,
                "power_representation": representation,
                "is_primary_analysis": is_primary,
                "n_pairs": n,
                "effect_mean": mean,
                "effect_sd": sd,
                "effect_se": se,
                "effect_var": float(se * se) if math.isfinite(se) else float("nan"),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "enters_meta": bool(enters_meta),
            }
        )
    return effects


def run_meta_analysis(
    dataset_effects: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Random-effects meta-analysis per endpoint×band (primary ZLPI cells only for primary flag)."""
    buckets: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in dataset_effects:
        if not _as_bool(row.get("enters_meta")):
            # Still allow explicitly separated sensitivity metas when requested via
            # non-primary rows that were marked enters_meta; default path uses primary.
            continue
        key = (
            _as_str(row.get("endpoint_name")).casefold(),
            str(_as_int(row.get("duration_s"))),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation")).casefold(),
        )
        buckets.setdefault(key, []).append(row)

    metas: list[dict[str, object]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        # Collapse multiple contrasts within a dataset (e.g. ds003690) by inverse-variance
        # weighted mean so each dataset contributes once to the meta-analysis.
        by_dataset: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            by_dataset.setdefault(_as_str(row.get("dataset_id")).casefold(), []).append(
                row
            )
        effects: list[float] = []
        variances: list[float] = []
        labels: list[str] = []
        for dataset_id, dset_rows in sorted(by_dataset.items()):
            if len(dset_rows) == 1:
                effects.append(_as_float(dset_rows[0].get("effect_mean")))
                variances.append(_as_float(dset_rows[0].get("effect_var")))
            else:
                ws = []
                es = []
                for r in dset_rows:
                    v = _as_float(r.get("effect_var"))
                    if math.isfinite(v) and v > 0:
                        ws.append(1.0 / v)
                        es.append(_as_float(r.get("effect_mean")))
                if not ws:
                    continue
                w = np.asarray(ws, dtype=float)
                e = np.asarray(es, dtype=float)
                mean = float(np.sum(w * e) / np.sum(w))
                var = float(1.0 / np.sum(w))
                effects.append(mean)
                variances.append(var)
            labels.append(dataset_id)
        endpoint_name, duration_s_s, band, representation = key
        duration_s = int(duration_s_s)
        meta = random_effects_meta(effects, variances, labels=labels)
        metas.append(
            {
                "endpoint_name": endpoint_name,
                "duration_s": duration_s,
                "band": band,
                "power_representation": representation,
                "is_primary_analysis": _is_primary_analysis(
                    endpoint_name=endpoint_name,
                    duration_s=duration_s,
                    power_representation=representation,
                ),
                "analysis_status": meta["analysis_status"],
                "estimator": meta["estimator"],
                "n_datasets": meta["n_datasets"],
                "pooled_effect": meta["pooled_effect"],
                "ci_low": meta["ci_low"],
                "ci_high": meta["ci_high"],
                "prediction_low": meta["prediction_low"],
                "prediction_high": meta["prediction_high"],
                "q": meta["q"],
                "tau2": meta["tau2"],
                "i2": meta["i2"],
                "p_value": meta["p_value"],
                "dataset_ids": meta["dataset_ids"],
                "notes": meta["notes"],
            }
        )
    return metas


def leave_one_dataset_out(
    dataset_effects: Sequence[Mapping[str, object]],
    full_meta_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Leave-one-dataset-out random-effects meta-analysis."""
    full_by_key: dict[tuple[str, ...], float] = {}
    if full_meta_rows is not None:
        for row in full_meta_rows:
            full_by_key[
                (
                    _as_str(row.get("endpoint_name")).casefold(),
                    str(_as_int(row.get("duration_s"))),
                    _as_str(row.get("band")).casefold(),
                    _as_str(row.get("power_representation")).casefold(),
                )
            ] = _as_float(row.get("pooled_effect"))

    buckets: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in dataset_effects:
        if not _as_bool(row.get("enters_meta")):
            continue
        key = (
            _as_str(row.get("endpoint_name")).casefold(),
            str(_as_int(row.get("duration_s"))),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation")).casefold(),
        )
        buckets.setdefault(key, []).append(row)

    loo_rows: list[dict[str, object]] = []
    for key in sorted(buckets):
        rows = buckets[key]
        by_dataset: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            by_dataset.setdefault(_as_str(row.get("dataset_id")).casefold(), []).append(
                row
            )
        dataset_ids = sorted(by_dataset)
        endpoint_name, duration_s_s, band, representation = key
        if len(dataset_ids) < 3:
            # LOO needs at least 2 remaining studies after omission.
            loo_rows.append(
                {
                    "endpoint_name": endpoint_name,
                    "duration_s": int(duration_s_s),
                    "band": band,
                    "power_representation": representation,
                    "analysis_status": ANALYSIS_STATUS_SKIPPED_INSUFFICIENT,
                    "omitted_dataset_id": "",
                    "estimator": "",
                    "n_datasets": len(dataset_ids),
                    "pooled_effect": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "prediction_low": float("nan"),
                    "prediction_high": float("nan"),
                    "q": float("nan"),
                    "tau2": float("nan"),
                    "i2": float("nan"),
                    "p_value": float("nan"),
                    "delta_vs_full": float("nan"),
                    "notes": (
                        "LOO skipped: requires ≥3 datasets "
                        "(so ≥2 remain after omission)."
                    ),
                }
            )
            continue
        duration_s = int(duration_s_s)
        full_effect = full_by_key.get(key, float("nan"))

        def _dataset_effect(dset_rows: Sequence[Mapping[str, object]]) -> tuple[float, float]:
            if len(dset_rows) == 1:
                return (
                    _as_float(dset_rows[0].get("effect_mean")),
                    _as_float(dset_rows[0].get("effect_var")),
                )
            ws = []
            es = []
            for r in dset_rows:
                v = _as_float(r.get("effect_var"))
                if math.isfinite(v) and v > 0:
                    ws.append(1.0 / v)
                    es.append(_as_float(r.get("effect_mean")))
            w = np.asarray(ws, dtype=float)
            e = np.asarray(es, dtype=float)
            return float(np.sum(w * e) / np.sum(w)), float(1.0 / np.sum(w))

        for omitted in dataset_ids:
            effects = []
            variances = []
            labels = []
            for dataset_id in dataset_ids:
                if dataset_id == omitted:
                    continue
                eff, var = _dataset_effect(by_dataset[dataset_id])
                effects.append(eff)
                variances.append(var)
                labels.append(dataset_id)
            meta = random_effects_meta(effects, variances, labels=labels)
            pooled = _as_float(meta["pooled_effect"])
            loo_rows.append(
                {
                    "endpoint_name": endpoint_name,
                    "duration_s": duration_s,
                    "band": band,
                    "power_representation": representation,
                    "analysis_status": _as_str(
                        meta.get("analysis_status"), ANALYSIS_STATUS_COMPLETED
                    ),
                    "omitted_dataset_id": omitted,
                    "estimator": meta["estimator"],
                    "n_datasets": meta["n_datasets"],
                    "pooled_effect": pooled,
                    "ci_low": meta["ci_low"],
                    "ci_high": meta["ci_high"],
                    "prediction_low": meta["prediction_low"],
                    "prediction_high": meta["prediction_high"],
                    "q": meta["q"],
                    "tau2": meta["tau2"],
                    "i2": meta["i2"],
                    "p_value": meta["p_value"],
                    "delta_vs_full": (
                        float(pooled - full_effect)
                        if math.isfinite(pooled) and math.isfinite(full_effect)
                        else float("nan")
                    ),
                    "notes": _as_str(meta.get("notes")),
                }
            )
    return loo_rows


def tost_peak_center_equivalence(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    bound: float = EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
) -> list[dict[str, object]]:
    """TOST equivalence of low-demand peak centers to 0 within ±bound seconds."""
    buckets: dict[tuple[str, ...], list[float]] = {}
    for raw in subject_rows:
        role = _as_str(raw.get("state") or raw.get("condition_role"))
        dataset_id = _as_str(raw.get("dataset_id")).casefold()
        condition = _as_str(raw.get("condition")).casefold()
        if not role:
            role = _condition_role(dataset_id, condition)
        if role != "low_demand":
            continue
        if not _as_bool(raw.get("has_identifiable_peak")):
            continue
        mu = _as_float(raw.get("peak_center_mu_s"))
        if not math.isfinite(mu):
            continue
        key = (
            dataset_id,
            _as_str(raw.get("endpoint_name")).casefold(),
            str(_as_int(raw.get("duration_s"))),
            _as_str(raw.get("band")).casefold(),
            _as_str(raw.get("power_representation")).casefold(),
        )
        buckets.setdefault(key, []).append(float(mu))

    rows: list[dict[str, object]] = []
    low, upp = -float(bound), float(bound)
    for key in sorted(buckets):
        values = np.asarray(buckets[key], dtype=float)
        n = int(values.size)
        mean_mu = float(np.mean(values))
        if n >= 2:
            se = float(np.std(values, ddof=1) / math.sqrt(n))
            dstats = DescrStatsW(values)
            tost_p, lower_res, upper_res = dstats.ttost_mean(low, upp)
            # lower_res / upper_res: (tstat, pvalue, df)
            p_lower = float(lower_res[1])
            p_upper = float(upper_res[1])
            ci = dstats.tconfint_mean()
            ci_low, ci_high = float(ci[0]), float(ci[1])
            notes = ""
        elif n == 1:
            se = float("nan")
            tost_p = float("nan")
            p_lower = float("nan")
            p_upper = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            notes = "Need ≥2 identifiable low-demand peaks for TOST."
        else:
            continue
        dataset_id, endpoint_name, duration_s_s, band, representation = key
        rows.append(
            {
                "dataset_id": dataset_id,
                "endpoint_name": endpoint_name,
                "duration_s": int(duration_s_s),
                "band": band,
                "power_representation": representation,
                "condition_role": "low_demand",
                "n": n,
                "mean_mu": mean_mu,
                "se_mu": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bound_low": low,
                "bound_high": upp,
                "tost_p_lower": p_lower,
                "tost_p_upper": p_upper,
                "tost_p": float(tost_p) if math.isfinite(float(tost_p)) else float("nan"),
                "equivalent": bool(
                    math.isfinite(float(tost_p)) and float(tost_p) < FDR_ALPHA
                ),
                "notes": notes,
            }
        )
    return rows


def _low_demand_zlpi_tests(
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Per dataset×band one-sample tests that low-demand primary ZLPI ≠ 0."""
    buckets: dict[tuple[str, str], list[float]] = {}
    for raw in subject_rows:
        if not _is_primary_analysis(
            endpoint_name=_as_str(raw.get("endpoint_name")),
            duration_s=_as_int(raw.get("duration_s")),
            power_representation=_as_str(raw.get("power_representation")),
        ):
            continue
        dataset_id = _as_str(raw.get("dataset_id")).casefold()
        if dataset_id not in PRIMARY_PAIRED_DATASETS:
            continue
        role = _as_str(raw.get("state") or raw.get("condition_role"))
        if not role:
            role = _condition_role(dataset_id, _as_str(raw.get("condition")))
        if role != "low_demand":
            continue
        if not _as_bool(raw.get("endpoint_eligible", True)):
            continue
        value = _as_float(raw.get("endpoint_index"))
        if not math.isfinite(value):
            continue
        band = _as_str(raw.get("band")).casefold()
        buckets.setdefault((dataset_id, band), []).append(value)

    rows: list[dict[str, object]] = []
    for (dataset_id, band), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            p_value = float("nan")
            effect = float(np.mean(arr)) if arr.size else float("nan")
        else:
            effect = float(np.mean(arr))
            _t, p_value = stats.ttest_1samp(arr, popmean=0.0)
            p_value = float(p_value)
        rows.append(
            {
                "family_id": FAMILY_PRIMARY_LOW_DEMAND_ZLPI,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "band": band,
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "dataset_id": dataset_id,
                "contrast_id": "",
                "test_id": f"low_demand_zlpi::{dataset_id}::{band}",
                "raw_p": p_value,
                "effect": effect,
                "notes": "",
            }
        )
    return rows


def apply_multiplicity(
    *,
    subject_rows: Sequence[Mapping[str, object]],
    dataset_effects: Sequence[Mapping[str, object]],
    meta_rows: Sequence[Mapping[str, object]],
    equivalence_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """BH-FDR within each prespecified, non-overlapping family."""
    candidates: list[dict[str, object]] = []
    candidates.extend(_low_demand_zlpi_tests(subject_rows))

    # Primary state attenuation: primary ZLPI dataset contrasts.
    for row in dataset_effects:
        if not _as_bool(row.get("is_primary_analysis")):
            continue
        contrast_id = _as_str(row.get("contrast_id")).casefold()
        if contrast_id not in PRIMARY_STATE_CONTRASTS:
            continue
        candidates.append(
            {
                "family_id": FAMILY_PRIMARY_STATE_ATTENUATION,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "band": _as_str(row.get("band")),
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "dataset_id": _as_str(row.get("dataset_id")),
                "contrast_id": contrast_id,
                "test_id": (
                    f"state_atten::{row.get('dataset_id')}::{contrast_id}::"
                    f"{row.get('band')}"
                ),
                "raw_p": _as_float(row.get("p_value")),
                "effect": _as_float(row.get("effect_mean")),
                "notes": "",
            }
        )

    for row in meta_rows:
        if not _as_bool(row.get("is_primary_analysis")):
            continue
        candidates.append(
            {
                "family_id": FAMILY_META_BAND,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "band": _as_str(row.get("band")),
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "dataset_id": "",
                "contrast_id": "",
                "test_id": f"meta_band::{row.get('band')}",
                "raw_p": _as_float(row.get("p_value")),
                "effect": _as_float(row.get("pooled_effect")),
                "notes": "",
            }
        )

    for row in equivalence_rows:
        if _as_str(row.get("endpoint_name")).casefold() != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s")) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            _as_str(row.get("power_representation")).casefold()
            != PRIMARY_POWER_REPRESENTATION
        ):
            continue
        candidates.append(
            {
                "family_id": FAMILY_MU_EQUIVALENCE,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "band": _as_str(row.get("band")),
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "dataset_id": _as_str(row.get("dataset_id")),
                "contrast_id": "",
                "test_id": f"mu_eq::{row.get('dataset_id')}::{row.get('band')}",
                "raw_p": _as_float(row.get("tost_p")),
                "effect": _as_float(row.get("mean_mu")),
                "notes": "TOST p (max of one-sided); FDR is descriptive within family.",
            }
        )

    # Local prominence attenuation family (secondary frequency family).
    prom_buckets: dict[tuple[str, ...], list[float]] = {}
    for raw in subject_rows:
        # Only primary ZLPI paired contrasts' low/effort difference via subject table
        # is awkward; use paired rows through dataset_effects-style contrasts of
        # local prominence when present on subject-level task pairs is not available.
        # Derive from paired contrast rows passed indirectly: store from subject by
        # computing low-demand local prominence tests per band×dataset.
        if not _is_primary_analysis(
            endpoint_name=_as_str(raw.get("endpoint_name")),
            duration_s=_as_int(raw.get("duration_s")),
            power_representation=_as_str(raw.get("power_representation")),
        ):
            continue
        dataset_id = _as_str(raw.get("dataset_id")).casefold()
        if dataset_id not in PRIMARY_PAIRED_DATASETS:
            continue
        role = _as_str(raw.get("state") or raw.get("condition_role"))
        if not role:
            role = _condition_role(dataset_id, _as_str(raw.get("condition")))
        if role != "low_demand":
            continue
        value = _as_float(raw.get("local_prominence"))
        if not math.isfinite(value):
            continue
        band = _as_str(raw.get("band")).casefold()
        prom_buckets.setdefault((dataset_id, band), []).append(value)
    for (dataset_id, band), values in sorted(prom_buckets.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            p_value = float("nan")
            effect = float(np.mean(arr)) if arr.size else float("nan")
        else:
            effect = float(np.mean(arr))
            _t, p_value = stats.ttest_1samp(arr, popmean=0.0)
            p_value = float(p_value)
        candidates.append(
            {
                "family_id": FAMILY_LOCAL_PROMINENCE,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "band": band,
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "dataset_id": dataset_id,
                "contrast_id": "",
                "test_id": f"local_prom::{dataset_id}::{band}",
                "raw_p": p_value,
                "effect": effect,
                "notes": "Secondary frequency family; cannot rescue ZLPI primary.",
            }
        )

    # Apply BH within each family independently.
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in candidates:
        by_family.setdefault(str(row["family_id"]), []).append(row)

    out: list[dict[str, object]] = []
    for family_id, rows in sorted(by_family.items()):
        order = sorted(rows, key=lambda r: str(r["test_id"]))
        q_values = bh_fdr([_as_float(r.get("raw_p")) for r in order])
        for row, q_value in zip(order, q_values, strict=True):
            raw_p = _as_float(row.get("raw_p"))
            out.append(
                {
                    "family_id": family_id,
                    "endpoint_name": row["endpoint_name"],
                    "duration_s": row["duration_s"],
                    "band": row["band"],
                    "power_representation": row["power_representation"],
                    "dataset_id": row["dataset_id"],
                    "contrast_id": row["contrast_id"],
                    "test_id": row["test_id"],
                    "raw_p": raw_p,
                    "q_value": q_value,
                    "reject_fdr": bool(
                        math.isfinite(q_value) and q_value <= FDR_ALPHA
                    ),
                    "effect": _as_float(row.get("effect")),
                    "notes": _as_str(row.get("notes")),
                }
            )
    return out


def run_confirmatory_inference(
    subject_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    *,
    fit_endpoints: Sequence[str] = (
        ENDPOINT_ZLPI,
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ),
) -> InferenceResult:
    """Run the full M10 inference stack with endpoint families kept separate."""
    mixed_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    endpoint_duration = {
        ENDPOINT_ZLPI: EXPECTED_PRIMARY_DURATION_S,
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: 120,
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: 60,
    }
    for endpoint_name in fit_endpoints:
        duration_s = endpoint_duration.get(endpoint_name, EXPECTED_PRIMARY_DURATION_S)
        # ZLPI may also be requested for D180 sensitivity — primary path is D240.
        coef_rows, qc = fit_mixed_model(
            subject_rows,
            endpoint_name=endpoint_name,
            duration_s=duration_s,
            power_representation=PRIMARY_POWER_REPRESENTATION,
        )
        mixed_rows.extend(coef_rows)
        qc_rows.append(qc)
        # Explicit guardrail: D120/D60 never marked primary.
        if endpoint_name != ENDPOINT_ZLPI:
            qc_rows.append(
                {
                    "component": "endpoint_guardrail",
                    "endpoint_name": endpoint_name,
                    "duration_s": duration_s,
                    "status": "sensitivity_only",
                    "model_backend": "",
                    "converged": "",
                    "n_obs": "",
                    "n_groups": "",
                    "notes": (
                        "MWPI/SWPI kept separate; cannot rescue failed primary ZLPI."
                    ),
                }
            )

    dataset_effects = estimate_dataset_effects(paired_rows)
    # Meta only from enters_meta (primary ZLPI); MWPI/SWPI effects stay in dataset table
    # but do not feed primary meta.
    meta_rows = run_meta_analysis(dataset_effects)
    loo_rows = leave_one_dataset_out(dataset_effects, meta_rows)
    equivalence_rows = tost_peak_center_equivalence(subject_rows)
    multiplicity_rows = apply_multiplicity(
        subject_rows=subject_rows,
        dataset_effects=dataset_effects,
        meta_rows=meta_rows,
        equivalence_rows=equivalence_rows,
    )

    # QC: confirm no MWPI/SWPI effects entered primary meta.
    for row in meta_rows:
        if _as_str(row.get("endpoint_name")).casefold() != ENDPOINT_ZLPI:
            qc_rows.append(
                {
                    "component": "meta_analysis",
                    "endpoint_name": row.get("endpoint_name"),
                    "duration_s": row.get("duration_s"),
                    "status": "error_endpoint_mix",
                    "model_backend": "",
                    "converged": "",
                    "n_obs": "",
                    "n_groups": "",
                    "notes": "Non-ZLPI endpoint entered meta; should never happen.",
                }
            )
        elif not _as_bool(row.get("is_primary_analysis")):
            qc_rows.append(
                {
                    "component": "meta_analysis",
                    "endpoint_name": row.get("endpoint_name"),
                    "duration_s": row.get("duration_s"),
                    "status": "non_primary_meta",
                    "model_backend": row.get("estimator"),
                    "converged": True,
                    "n_obs": row.get("n_datasets"),
                    "n_groups": "",
                    "notes": "Non-primary meta cell retained separately.",
                }
            )
        else:
            qc_rows.append(
                {
                    "component": "meta_analysis",
                    "endpoint_name": row.get("endpoint_name"),
                    "duration_s": row.get("duration_s"),
                    "status": "ok",
                    "model_backend": row.get("estimator"),
                    "converged": True,
                    "n_obs": row.get("n_datasets"),
                    "n_groups": "",
                    "notes": _as_str(row.get("notes")),
                }
            )

    return InferenceResult(
        mixed_model_rows=tuple(mixed_rows),
        dataset_effect_rows=tuple(dataset_effects),
        meta_rows=tuple(meta_rows),
        loo_rows=tuple(loo_rows),
        equivalence_rows=tuple(equivalence_rows),
        multiplicity_rows=tuple(multiplicity_rows),
        qc_rows=tuple(qc_rows),
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
            payload: dict[str, object] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                elif isinstance(value, bool):
                    payload[field] = str(value)
                else:
                    payload[field] = value
            writer.writerow(payload)


def write_inference_outputs(
    result: InferenceResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "mixed_model_results": output_path / MIXED_MODEL_RESULTS_FILENAME,
        "dataset_effects": output_path / DATASET_EFFECTS_FILENAME,
        "meta_analysis_results": output_path / META_ANALYSIS_RESULTS_FILENAME,
        "leave_one_dataset_out": output_path / LEAVE_ONE_DATASET_OUT_FILENAME,
        "peak_center_equivalence": output_path / PEAK_CENTER_EQUIVALENCE_FILENAME,
        "multiplicity_results": output_path / MULTIPLICITY_RESULTS_FILENAME,
        "inference_qc": output_path / INFERENCE_QC_FILENAME,
    }
    loo_rows = list(result.loo_rows)
    if not loo_rows:
        loo_rows = [
            {
                "endpoint_name": "",
                "duration_s": "",
                "band": "",
                "power_representation": "",
                "analysis_status": ANALYSIS_STATUS_SKIPPED_INSUFFICIENT,
                "omitted_dataset_id": "",
                "estimator": "",
                "n_datasets": 0,
                "pooled_effect": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "prediction_low": float("nan"),
                "prediction_high": float("nan"),
                "q": float("nan"),
                "tau2": float("nan"),
                "i2": float("nan"),
                "p_value": float("nan"),
                "delta_vs_full": float("nan"),
                "notes": (
                    "LOO table empty: no primary meta cells with ≥3 datasets "
                    "(requires ≥2 remaining after omission)."
                ),
            }
        ]
    _write_csv(paths["mixed_model_results"], result.mixed_model_rows, MIXED_MODEL_FIELDS)
    _write_csv(paths["dataset_effects"], result.dataset_effect_rows, DATASET_EFFECT_FIELDS)
    _write_csv(paths["meta_analysis_results"], result.meta_rows, META_FIELDS)
    _write_csv(paths["leave_one_dataset_out"], loo_rows, LOO_FIELDS)
    _write_csv(
        paths["peak_center_equivalence"], result.equivalence_rows, EQUIVALENCE_FIELDS
    )
    _write_csv(paths["multiplicity_results"], result.multiplicity_rows, MULTIPLICITY_FIELDS)
    _write_csv(paths["inference_qc"], result.qc_rows, QC_FIELDS)
    return paths


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_confirmatory_inference_from_dir(
    group_tables_dir: str | Path,
    output_dir: str | Path,
) -> InferenceResult:
    """Load M8 tables and write M10 inference outputs."""
    root = Path(group_tables_dir).expanduser().resolve()
    subject_rows = read_csv_rows(root / SUBJECT_LEVEL_FILENAME)
    paired_rows = read_csv_rows(root / PAIRED_CONTRASTS_FILENAME)
    result = run_confirmatory_inference(subject_rows, paired_rows)
    write_inference_outputs(result, output_dir)
    return result


__all__ = [
    "DATASET_EFFECTS_FILENAME",
    "FAMILY_META_BAND",
    "FAMILY_MU_EQUIVALENCE",
    "FAMILY_PRIMARY_LOW_DEMAND_ZLPI",
    "FAMILY_PRIMARY_STATE_ATTENUATION",
    "FDR_ALPHA",
    "INFERENCE_QC_FILENAME",
    "LEAVE_ONE_DATASET_OUT_FILENAME",
    "META_ANALYSIS_RESULTS_FILENAME",
    "META_EXCLUDED_DATASETS",
    "MIXED_MODEL_RESULTS_FILENAME",
    "MULTIPLICITY_RESULTS_FILENAME",
    "PEAK_CENTER_EQUIVALENCE_FILENAME",
    "InferenceResult",
    "apply_multiplicity",
    "bh_fdr",
    "estimate_dataset_effects",
    "fit_mixed_model",
    "leave_one_dataset_out",
    "prediction_interval",
    "random_effects_meta",
    "run_confirmatory_inference",
    "run_confirmatory_inference_from_dir",
    "run_meta_analysis",
    "tost_peak_center_equivalence",
    "write_inference_outputs",
]
