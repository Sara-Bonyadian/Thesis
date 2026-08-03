"""Figure 3 Panel E: nuisance and modality robustness (standalone sheet).

Primary analysis uses within-pair Tetris−Rest nuisance *changes* so the OLS
intercept is the Rest–Tetris ΔZLPI at zero nuisance change. The legacy
pair-average + centered-covariate construction is retained only as an internal
audit (algebraically forces intercept = unadjusted mean).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from .duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S
from .paired_delta_inference import DEFAULT_CLUSTER_BOOTSTRAP_DRAWS

PANEL_E_BAND = "alpha"
PANEL_E_DURATION_S = int(EXPECTED_PRIMARY_DURATION_S)
PANEL_E_ENDPOINT = ENDPOINT_ZLPI
PANEL_E_REPRESENTATION = "absolute_log10"
PANEL_E_STEM = "figure3_panel_e_nuisance_modality"
PANEL_E_FIGURE_TITLE = "Figure 3E | Nuisance and modality robustness"
PANEL_E_SUBTITLE = (
    "D240 alpha ZLPI Rest–Tetris contrast; participant-clustered 95% CIs; "
    "HIIT sensitivity dataset"
)

BOOTSTRAP_DRAWS = int(DEFAULT_CLUSTER_BOOTSTRAP_DRAWS)
BOOTSTRAP_SEED = 19
NEAR_ZERO_VARIANCE = 1e-12
RANK_DEFICIENT_COND = 1e12

STATUS_COMPUTED = "computed"
STATUS_NOT_AVAILABLE = "not_available"
STATUS_NOT_COMPUTABLE = "not_computable"
STATUS_NOT_IDENTIFIABLE = "not_identifiable"
STATUS_RANK_DEFICIENT = "rank_deficient"

# Primary manuscript display (estimable rows only on numeric axes).
# Preferred broadband nuisance excludes alpha (avoids circularity with alpha ZLPI).
ESTIMABLE_SPEC_ORDER: tuple[str, ...] = (
    "baseline",
    "delta_mean_hr",
    "delta_broadband_power_ex_alpha",
    "delta_hr_broadband_ex_alpha",
)

# Sensitivity-only: alpha-inclusive broadband (partially circular with alpha ZLPI).
SENSITIVITY_SPEC_ORDER: tuple[str, ...] = (
    "delta_broadband_power",
    "delta_hr_broadband",
)

# Observation-level sensitivity specs (same Δ-nuisance scientific content).
OBSLEVEL_SPEC_ORDER: tuple[str, ...] = (
    "obs_baseline",
    "obs_mean_hr",
    "obs_broadband_power",
    "obs_hr_broadband",
)

# Locked bootstrap seed offsets so reordering primary/sensitivity does not alter fits.
SPEC_SEED_OFFSET: dict[str, int] = {
    "baseline": 1,
    "delta_mean_hr": 2,
    "delta_broadband_power": 3,
    "delta_hr_broadband": 4,
    "delta_broadband_power_ex_alpha": 10,
    "delta_hr_broadband_ex_alpha": 11,
}

SPEC_DISPLAY_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "delta_mean_hr": "Mean HR change",
    "delta_broadband_power_ex_alpha": "Broadband change, excluding alpha",
    "delta_hr_broadband_ex_alpha": "HR + broadband change, excluding alpha",
    "delta_broadband_power": "Broadband change, including alpha (sensitivity)",
    "delta_hr_broadband": "HR + broadband change, including alpha (sensitivity)",
    "obs_baseline": "Obs-level: state (unadjusted)",
    "obs_mean_hr": "Obs-level: state + mean HR",
    "obs_broadband_power": "Obs-level: state + broadband power",
    "obs_hr_broadband": "Obs-level: state + HR + broadband",
}

SPEC_COVARIATES: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "delta_mean_hr": ("delta_mean_hr",),
    "delta_broadband_power": ("delta_broadband_power",),
    "delta_hr_broadband": ("delta_mean_hr", "delta_broadband_power"),
    "delta_broadband_power_ex_alpha": ("delta_broadband_power_ex_alpha",),
    "delta_hr_broadband_ex_alpha": ("delta_mean_hr", "delta_broadband_power_ex_alpha"),
}

# Compact planned-but-unavailable table.
UNAVAILABLE_TABLE: tuple[dict[str, str], ...] = (
    {
        "planned_adjustment": "Respiration",
        "status": "Not available",
        "reason": "No retained observation-level respiration measure",
    },
    {
        "planned_adjustment": "Motion",
        "status": "Not available",
        "reason": "No retained motion or accelerometer summary",
    },
    {
        "planned_adjustment": "Bad-window rate",
        "status": "Not available",
        "reason": "No observation-level rejected-window proportion",
    },
    {
        "planned_adjustment": "EOG",
        "status": "Not available",
        "reason": "No retained ocular-artifact summary",
    },
    {
        "planned_adjustment": "Eye state",
        "status": "Not identifiable",
        "reason": "Fixed by Rest versus Tetris condition",
    },
    {
        "planned_adjustment": "EMG",
        "status": "Not available",
        "reason": "No retained muscle-artifact summary",
    },
    {
        "planned_adjustment": "ECG/PPG modality",
        "status": "Not identifiable",
        "reason": "All HIIT observations use PPG",
    },
)

BROADBAND_DEFINITION = {
    "primary": {
        "bands": ["theta", "beta", "low_gamma"],
        "frequency_hz": {"theta": "4–8", "beta": "12–30", "low_gamma": "30–45"},
        "channels": "confirmatory multitaper channel set (whole-scalp usable montage)",
        "transform": (
            "per-band absolute_log10_power, then nanmean across theta/beta/low_gamma "
            "at each 1 Hz sample, then mean over D240; Tetris−Rest difference"
        ),
        "includes_alpha": False,
        "role": "preferred_manuscript_nuisance",
        "independence_note": (
            "Excludes alpha so the composite is not constructed from the same band "
            "that enters the alpha ZLPI outcome."
        ),
    },
    "alpha_inclusive_sensitivity": {
        "bands": ["theta", "alpha", "beta", "low_gamma"],
        "frequency_hz": {
            "theta": "4–8",
            "alpha": "8–12",
            "beta": "12–30",
            "low_gamma": "30–45",
        },
        "channels": "confirmatory multitaper channel set (whole-scalp usable montage)",
        "transform": (
            "per-band absolute_log10_power, then nanmean across bands at each 1 Hz "
            "sample, then mean over D240; Tetris−Rest difference"
        ),
        "includes_alpha": True,
        "role": "sensitivity_only_potentially_circular",
        "independence_note": (
            "Includes alpha log-power and therefore partially overlaps the alpha "
            "series that enters ZLPI. Not an independent nuisance control; retained "
            "only as internal sensitivity. Ex-alpha and alpha-inclusive Δβ can differ "
            "in sign because removing the circular alpha component changes the "
            "within-pair power change that is partialled out."
        ),
    },
}


@dataclass(frozen=True)
class PanelEResult:
    observation_rows: tuple[dict[str, object], ...]
    specification_rows: tuple[dict[str, object], ...]
    common_sample_rows: tuple[dict[str, object], ...]
    observation_level_rows: tuple[dict[str, object], ...]
    diagnostic_rows: tuple[dict[str, object], ...]
    missingness_rows: tuple[dict[str, object], ...]
    audit_old_vs_new_rows: tuple[dict[str, object], ...]
    nuisance_state_summary_rows: tuple[dict[str, object], ...]
    metadata: dict[str, object]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    return text in {"1", "true", "yes", "y", "t"}


def _as_int_duration(value: object) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return -1


def _read_csv(path: Path | None) -> list[dict[str, object]]:
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                elif isinstance(value, bool):
                    payload[field] = "True" if value else "False"
                else:
                    payload[field] = value
            writer.writerow(payload)


def _percentile(arr: np.ndarray, q: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _summarize_numeric(values: Sequence[float]) -> dict[str, object]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "n_total": int(arr.size),
        "n_finite": int(finite.size),
        "n_missing": int(arr.size - finite.size),
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        "p05": _percentile(finite, 5),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
    }


def _paired_ci(deltas: np.ndarray) -> tuple[float, float, float]:
    finite = deltas[np.isfinite(deltas)]
    n = int(finite.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(np.std(finite, ddof=1) / math.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * se, mean + tcrit * se


def _observation_feature_summaries(
    aligned_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """Per-observation mean HR and broadband composites from 1 Hz aligned features."""
    hr_b: dict[str, list[float]] = defaultdict(list)
    bb_b: dict[str, list[float]] = defaultdict(list)
    bb_ex_b: dict[str, list[float]] = defaultdict(list)
    for row in aligned_rows:
        obs = _as_str(row.get("observation_id"))
        if not obs:
            continue
        hr = _as_float(row.get("hr_bpm"))
        if math.isfinite(hr):
            hr_b[obs].append(hr)
        band_vals = {
            band: _as_float(row.get(f"{band}_absolute_log10_power"))
            for band in ("theta", "alpha", "beta", "low_gamma")
        }
        all_finite = [v for v in band_vals.values() if math.isfinite(v)]
        ex_alpha = [
            band_vals[b] for b in ("theta", "beta", "low_gamma") if math.isfinite(band_vals[b])
        ]
        if all_finite:
            bb_b[obs].append(float(np.mean(all_finite)))
        if ex_alpha:
            bb_ex_b[obs].append(float(np.mean(ex_alpha)))
    out: dict[str, dict[str, float]] = {}
    for obs in sorted(set(hr_b) | set(bb_b) | set(bb_ex_b)):
        out[obs] = {
            "mean_hr": float(np.mean(hr_b[obs])) if hr_b.get(obs) else float("nan"),
            "broadband_power": float(np.mean(bb_b[obs])) if bb_b.get(obs) else float("nan"),
            "broadband_power_ex_alpha": (
                float(np.mean(bb_ex_b[obs])) if bb_ex_b.get(obs) else float("nan")
            ),
        }
    return out


def _modality_lookup(
    data_audit_rows: Sequence[Mapping[str, object]],
    peak_qc_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in data_audit_rows:
        obs = _as_str(row.get("observation_id"))
        sig = _as_str(row.get("cardiac_signal_type")).casefold()
        if obs and sig:
            out[obs] = sig
    for row in peak_qc_rows:
        obs = _as_str(row.get("observation_id"))
        sig = _as_str(row.get("signal_type")).casefold()
        if obs and sig and obs not in out:
            out[obs] = sig
    return out


def build_panel_e_contrast_rows(
    *,
    paired_rows: Sequence[Mapping[str, object]],
    aligned_rows: Sequence[Mapping[str, object]],
    data_audit_rows: Sequence[Mapping[str, object]] = (),
    peak_qc_rows: Sequence[Mapping[str, object]] = (),
    protocol_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build paired analysis rows with within-pair Tetris−Rest nuisance changes."""
    del protocol_rows  # retained for API compatibility; eye-state handled as non-identifiable
    features = _observation_feature_summaries(aligned_rows)
    modality = _modality_lookup(data_audit_rows, peak_qc_rows)
    rows_out: list[dict[str, object]] = []

    for row in paired_rows:
        if _as_int_duration(row.get("duration_s")) != PANEL_E_DURATION_S:
            continue
        if _as_str(row.get("endpoint_name")).casefold() != PANEL_E_ENDPOINT:
            continue
        if _as_str(row.get("power_representation")).casefold() != PANEL_E_REPRESENTATION:
            continue
        if _as_str(row.get("band")).casefold() != PANEL_E_BAND:
            continue
        if not _as_bool(row.get("contrast_eligible")):
            continue
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue

        low_obs = _as_str(row.get("low_observation_ids")).split(";")[0].strip()
        effort_obs = _as_str(row.get("effort_observation_ids")).split(";")[0].strip()
        rest = features.get(low_obs, {})
        tetris = features.get(effort_obs, {})

        def _get(side: Mapping[str, float], key: str) -> float:
            return _as_float(side.get(key))

        rest_hr = _get(rest, "mean_hr")
        tet_hr = _get(tetris, "mean_hr")
        rest_bb = _get(rest, "broadband_power")
        tet_bb = _get(tetris, "broadband_power")
        rest_bb_ex = _get(rest, "broadband_power_ex_alpha")
        tet_bb_ex = _get(tetris, "broadband_power_ex_alpha")

        # Legacy pair-average (audit only).
        legacy_hr = (
            float(np.mean([v for v in (rest_hr, tet_hr) if math.isfinite(v)]))
            if any(math.isfinite(v) for v in (rest_hr, tet_hr))
            else float("nan")
        )
        legacy_bb = (
            float(np.mean([v for v in (rest_bb, tet_bb) if math.isfinite(v)]))
            if any(math.isfinite(v) for v in (rest_bb, tet_bb))
            else float("nan")
        )

        sigs = [modality.get(low_obs, ""), modality.get(effort_obs, "")]
        sigs = [s for s in sigs if s]
        if any(s == "ecg" for s in sigs):
            cardiac_signal = "ecg" if all(s == "ecg" for s in sigs) else "mixed"
            modality_ecg = 1.0
        elif any(s == "ppg" for s in sigs):
            cardiac_signal = "ppg"
            modality_ecg = 0.0
        else:
            cardiac_signal = ""
            modality_ecg = float("nan")

        rows_out.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")).casefold(),
                "participant_id": _as_str(row.get("participant_id")).casefold(),
                "session_id": _as_str(row.get("session_id"), "single").casefold(),
                "contrast_id": _as_str(row.get("contrast_id")),
                "observation_id": f"{low_obs}__{effort_obs}",
                "low_observation_id": low_obs,
                "effort_observation_id": effort_obs,
                "band": PANEL_E_BAND,
                "duration_s": PANEL_E_DURATION_S,
                "endpoint_name": PANEL_E_ENDPOINT,
                "power_representation": PANEL_E_REPRESENTATION,
                "delta_endpoint_index": delta,
                # Within-pair changes (primary predictors).
                "rest_mean_hr": rest_hr,
                "tetris_mean_hr": tet_hr,
                "delta_mean_hr": (tet_hr - rest_hr)
                if math.isfinite(tet_hr) and math.isfinite(rest_hr)
                else float("nan"),
                "rest_broadband_power": rest_bb,
                "tetris_broadband_power": tet_bb,
                "delta_broadband_power": (tet_bb - rest_bb)
                if math.isfinite(tet_bb) and math.isfinite(rest_bb)
                else float("nan"),
                "rest_broadband_power_ex_alpha": rest_bb_ex,
                "tetris_broadband_power_ex_alpha": tet_bb_ex,
                "delta_broadband_power_ex_alpha": (tet_bb_ex - rest_bb_ex)
                if math.isfinite(tet_bb_ex) and math.isfinite(rest_bb_ex)
                else float("nan"),
                # Legacy pair averages (audit only).
                "legacy_pairavg_mean_hr": legacy_hr,
                "legacy_pairavg_broadband_power": legacy_bb,
                "cardiac_signal_type": cardiac_signal,
                "modality_ecg": modality_ecg,
                "modality_ppg": 1.0 - modality_ecg if math.isfinite(modality_ecg) else float("nan"),
            }
        )
    return rows_out


def build_observation_level_rows(
    contrast_rows: Sequence[Mapping[str, object]],
    endpoint_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Expand each contrast into Rest/Tetris observation rows for sensitivity models."""
    ep_lookup: dict[str, float] = {}
    for row in endpoint_rows:
        if _as_int_duration(row.get("duration_s")) != PANEL_E_DURATION_S:
            continue
        if _as_str(row.get("endpoint_name")).casefold() != PANEL_E_ENDPOINT:
            continue
        if _as_str(row.get("power_representation")).casefold() != PANEL_E_REPRESENTATION:
            continue
        if _as_str(row.get("band")).casefold() != PANEL_E_BAND:
            continue
        if not _as_bool(row.get("eligible")):
            continue
        obs = _as_str(row.get("observation_id"))
        val = _as_float(row.get("endpoint_index"))
        if obs and math.isfinite(val):
            ep_lookup[obs] = val

    out: list[dict[str, object]] = []
    for row in contrast_rows:
        for state, obs_key, hr_key, bb_key in (
            ("rest", "low_observation_id", "rest_mean_hr", "rest_broadband_power"),
            ("tetris", "effort_observation_id", "tetris_mean_hr", "tetris_broadband_power"),
        ):
            obs = _as_str(row.get(obs_key))
            y = ep_lookup.get(obs, float("nan"))
            if not math.isfinite(y):
                continue
            out.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "participant_id": row.get("participant_id"),
                    "session_id": row.get("session_id"),
                    "contrast_id": row.get("contrast_id"),
                    "observation_id": obs,
                    "pair_id": row.get("observation_id"),
                    "state": state,
                    "state_tetris": 1.0 if state == "tetris" else 0.0,
                    "endpoint_index": y,
                    "mean_hr": _as_float(row.get(hr_key)),
                    "broadband_power": _as_float(row.get(bb_key)),
                    "cardiac_signal_type": row.get("cardiac_signal_type"),
                }
            )
    return out


def assess_modality_identifiability(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ds = _as_str(row.get("dataset_id")).casefold()
        sig = _as_str(row.get("cardiac_signal_type")).casefold()
        if not ds:
            continue
        if sig in {"ecg", "ppg"}:
            by_dataset[ds].add(sig)
        elif sig == "mixed":
            by_dataset[ds].update({"ecg", "ppg"})
    all_signals = sorted({s for sigs in by_dataset.values() for s in sigs})
    within = {ds: sorted(sigs) for ds, sigs in by_dataset.items() if len(sigs) > 1}
    identifiable = bool(within)
    if not identifiable and len(all_signals) <= 1:
        reason = (
            f"cardiac modality is constant ({all_signals[0] if all_signals else 'unknown'}) "
            "in the analysis sample; modality coefficient has zero variance"
        )
    elif not identifiable:
        reason = (
            "ECG/PPG varies only between datasets and is perfectly confounded with "
            "dataset identity; independent modality coefficient is not identifiable"
        )
    else:
        reason = "within-dataset ECG/PPG variation present"
    return {
        "identifiable": identifiable,
        "reason": reason,
        "signals_present": all_signals,
        "within_dataset_variation": within,
        "n_ecg": sum(1 for r in rows if _as_str(r.get("cardiac_signal_type")).casefold() == "ecg"),
        "n_ppg": sum(1 for r in rows if _as_str(r.get("cardiac_signal_type")).casefold() == "ppg"),
    }


def _design_matrix_uncentered(
    rows: Sequence[Mapping[str, object]],
    covariates: Sequence[str],
    *,
    outcome_key: str = "delta_endpoint_index",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Design matrix with intercept; continuous predictors UNCENTERED (zero = no change)."""
    y = np.asarray([_as_float(r.get(outcome_key)) for r in rows], dtype=float)
    kept: list[str] = []
    cols: list[np.ndarray] = []
    for name in covariates:
        arr = np.asarray([_as_float(r.get(name)) for r in rows], dtype=float)
        if not np.any(np.isfinite(arr)):
            continue
        if float(np.nanvar(arr)) <= NEAR_ZERO_VARIANCE:
            continue
        kept.append(name)
        cols.append(arr)
    if cols:
        x = np.column_stack([np.ones(len(rows), dtype=float), *cols])
    else:
        x = np.ones((len(rows), 1), dtype=float)
    return y, x, kept


def _design_matrix_centered_legacy(
    rows: Sequence[Mapping[str, object]],
    covariates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    """Legacy centered design (audit only)."""
    y = np.asarray([_as_float(r.get("delta_endpoint_index")) for r in rows], dtype=float)
    kept: list[str] = []
    cols: list[np.ndarray] = []
    means: dict[str, float] = {}
    for name in covariates:
        arr = np.asarray([_as_float(r.get(name)) for r in rows], dtype=float)
        if not np.any(np.isfinite(arr)):
            continue
        if float(np.nanvar(arr)) <= NEAR_ZERO_VARIANCE:
            continue
        mu = float(np.nanmean(arr))
        kept.append(name)
        cols.append(arr - mu)
        means[name] = mu
    if cols:
        x = np.column_stack([np.ones(len(rows), dtype=float), *cols])
    else:
        x = np.ones((len(rows), 1), dtype=float)
    return y, x, kept, means


def _ols_fit(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n = int(mask.sum())
    p = int(x.shape[1])
    if n < p + 1:
        return {
            "ok": False,
            "reason": f"insufficient_rows_for_ols n={n} p={p}",
            "beta": np.full(p, np.nan),
            "condition_number": float("nan"),
            "rank_deficient": True,
            "n": n,
            "mask": mask,
        }
    xm = x[mask]
    ym = y[mask]
    try:
        cond = float(np.linalg.cond(xm))
    except np.linalg.LinAlgError:
        cond = float("inf")
    rank = int(np.linalg.matrix_rank(xm))
    rank_deficient = bool(rank < p or (math.isfinite(cond) and cond >= RANK_DEFICIENT_COND))
    if rank_deficient:
        return {
            "ok": False,
            "reason": f"rank_deficient rank={rank} p={p} cond={cond:.3g}",
            "beta": np.full(p, np.nan),
            "condition_number": cond,
            "rank_deficient": True,
            "n": n,
            "mask": mask,
        }
    beta, *_ = np.linalg.lstsq(xm, ym, rcond=None)
    return {
        "ok": True,
        "reason": "",
        "beta": np.asarray(beta, dtype=float),
        "condition_number": cond,
        "rank_deficient": False,
        "n": n,
        "mask": mask,
    }


def _unit_index_map(rows: Sequence[Mapping[str, object]]) -> tuple[list[str], dict[str, list[int]]]:
    by_unit: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_unit[_as_str(row.get("participant_id"))].append(idx)
    units = sorted(by_unit)
    return units, by_unit


def fit_paired_intercept_cluster_boot(
    rows: Sequence[Mapping[str, object]],
    covariates: Sequence[str],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    center: bool = False,
    legacy_cov_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fit ΔZLPI ~ 1 + predictors; return intercept with participant-clustered bootstrap CI.

    Primary path uses uncentered within-pair changes (center=False).
    """
    if not rows:
        return {
            "ok": False,
            "reason": "no_rows",
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "p_value": float("nan"),
            "condition_number": float("nan"),
            "rank_deficient": True,
            "kept_covariates": [],
            "n_observations": 0,
            "n_participants": 0,
            "boot_intercepts": np.asarray([], dtype=float),
        }

    if center:
        keys = list(legacy_cov_keys or covariates)
        y, x, kept, _means = _design_matrix_centered_legacy(rows, keys)
    else:
        y, x, kept = _design_matrix_uncentered(rows, covariates)

    fit = _ols_fit(y, x)
    if not fit["ok"]:
        return {
            "ok": False,
            "reason": fit["reason"],
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "p_value": float("nan"),
            "condition_number": fit["condition_number"],
            "rank_deficient": fit["rank_deficient"],
            "kept_covariates": kept,
            "n_observations": int(fit["n"]),
            "n_participants": len({_as_str(r.get("participant_id")) for r in rows}),
            "boot_intercepts": np.asarray([], dtype=float),
        }

    intercept = float(fit["beta"][0])
    units, by_unit = _unit_index_map(rows)
    n_units = len(units)
    if n_units < 2:
        return {
            "ok": True,
            "reason": "",
            "estimate": intercept,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "p_value": float("nan"),
            "condition_number": fit["condition_number"],
            "rank_deficient": False,
            "kept_covariates": kept,
            "n_observations": int(fit["n"]),
            "n_participants": n_units,
            "boot_intercepts": np.asarray([], dtype=float),
        }

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    boots = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        draw_units = [units[j] for j in rng.integers(0, n_units, size=n_units)]
        idx = [k for u in draw_units for k in by_unit[u]]
        fit_b = _ols_fit(y[idx], x[idx])
        boots[i] = float(fit_b["beta"][0]) if fit_b["ok"] else np.nan
    finite = boots[np.isfinite(boots)]
    if finite.size < max(10, int(0.5 * n_draws)):
        return {
            "ok": False,
            "reason": "bootstrap_failed",
            "estimate": intercept,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "p_value": float("nan"),
            "condition_number": fit["condition_number"],
            "rank_deficient": True,
            "kept_covariates": kept,
            "n_observations": int(fit["n"]),
            "n_participants": n_units,
            "boot_intercepts": boots,
        }
    lo, hi = np.quantile(finite, [0.025, 0.975])
    se = float(np.std(finite, ddof=1))
    p_value = float(2.0 * stats.norm.sf(abs(intercept / se))) if se > 0 else float("nan")
    return {
        "ok": True,
        "reason": "",
        "estimate": intercept,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "standard_error": se,
        "p_value": p_value,
        "condition_number": fit["condition_number"],
        "rank_deficient": False,
        "kept_covariates": kept,
        "n_observations": int(fit["n"]),
        "n_participants": n_units,
        "boot_intercepts": boots,
    }


def fit_obslevel_state_cluster_boot(
    obs_rows: Sequence[Mapping[str, object]],
    nuisance_vars: Sequence[str],
    *,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Observation-level: endpoint ~ state + nuisance + participant FE; extract state coef.

    Equivalent (no interaction) to paired ΔY ~ 1 + ΔN when nuisance enters additively.
    """
    if not obs_rows:
        return {
            "ok": False,
            "reason": "no_rows",
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "n_observations": 0,
            "n_participants": 0,
            "boot_estimates": np.asarray([], dtype=float),
        }

    participants = sorted({_as_str(r.get("participant_id")) for r in obs_rows})
    part_index = {p: i for i, p in enumerate(participants)}
    n = len(obs_rows)
    # Columns: intercept, state, nuisances..., participant FE (drop first)
    cols: list[np.ndarray] = [
        np.ones(n, dtype=float),
        np.asarray([_as_float(r.get("state_tetris")) for r in obs_rows], dtype=float),
    ]
    kept_nui: list[str] = []
    for name in nuisance_vars:
        arr = np.asarray([_as_float(r.get(name)) for r in obs_rows], dtype=float)
        if np.sum(np.isfinite(arr)) < 3 or float(np.nanvar(arr)) <= NEAR_ZERO_VARIANCE:
            continue
        # Leave uncentered so state coef matches Δ-model at absolute nuisance scale;
        # still identified with participant FE.
        kept_nui.append(name)
        cols.append(arr)
    if len(participants) > 1:
        fe = np.zeros((n, len(participants) - 1), dtype=float)
        for i, row in enumerate(obs_rows):
            j = part_index[_as_str(row.get("participant_id"))]
            if j > 0:
                fe[i, j - 1] = 1.0
        x = np.column_stack([*cols, fe])
    else:
        x = np.column_stack(cols)
    y = np.asarray([_as_float(r.get("endpoint_index")) for r in obs_rows], dtype=float)
    fit = _ols_fit(y, x)
    if not fit["ok"]:
        return {
            "ok": False,
            "reason": fit["reason"],
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "n_observations": int(fit["n"]),
            "n_participants": len(participants),
            "kept_nuisance": kept_nui,
            "boot_estimates": np.asarray([], dtype=float),
        }
    state_coef = float(fit["beta"][1])  # column 1 = state

    # Cluster bootstrap by participant (resample all obs for drawn participants).
    by_unit: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(obs_rows):
        by_unit[_as_str(row.get("participant_id"))].append(idx)
    units = sorted(by_unit)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    boots = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        draw = [units[j] for j in rng.integers(0, len(units), size=len(units))]
        idx = [k for u in draw for k in by_unit[u]]
        # Rebuild FE for drawn set
        sub_rows = [obs_rows[k] for k in idx]
        sub = fit_obslevel_state_cluster_boot(
            sub_rows, nuisance_vars, n_draws=1, seed=seed + i + 1
        ) if False else None  # placeholder to avoid recursion
        del sub
        yb = y[idx]
        # Reconstruct design for idx with FE among unique drawn participants
        drawn_parts = sorted({_as_str(obs_rows[k].get("participant_id")) for k in idx})
        pmap = {p: i for i, p in enumerate(drawn_parts)}
        base_cols = [np.ones(len(idx)), np.asarray([_as_float(obs_rows[k].get("state_tetris")) for k in idx])]
        for name in kept_nui:
            base_cols.append(np.asarray([_as_float(obs_rows[k].get(name)) for k in idx]))
        if len(drawn_parts) > 1:
            fe_b = np.zeros((len(idx), len(drawn_parts) - 1))
            for ii, k in enumerate(idx):
                j = pmap[_as_str(obs_rows[k].get("participant_id"))]
                if j > 0:
                    fe_b[ii, j - 1] = 1.0
            xb = np.column_stack([*base_cols, fe_b])
        else:
            xb = np.column_stack(base_cols)
        fit_b = _ols_fit(yb, xb)
        boots[i] = float(fit_b["beta"][1]) if fit_b["ok"] and fit_b["beta"].size > 1 else np.nan

    finite = boots[np.isfinite(boots)]
    if finite.size < max(10, int(0.5 * n_draws)):
        return {
            "ok": True,
            "reason": "bootstrap_unstable",
            "estimate": state_coef,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "standard_error": float("nan"),
            "n_observations": int(fit["n"]),
            "n_participants": len(participants),
            "kept_nuisance": kept_nui,
            "boot_estimates": boots,
        }
    lo, hi = np.quantile(finite, [0.025, 0.975])
    se = float(np.std(finite, ddof=1))
    return {
        "ok": True,
        "reason": "",
        "estimate": state_coef,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "standard_error": se,
        "n_observations": int(fit["n"]),
        "n_participants": len(participants),
        "kept_nuisance": kept_nui,
        "boot_estimates": boots,
    }


def _delta_beta_from_boots(
    base_boots: np.ndarray,
    adj_boots: np.ndarray,
    base_est: float,
    adj_est: float,
) -> dict[str, float]:
    d = adj_est - base_est
    if base_boots.size and adj_boots.size and base_boots.shape == adj_boots.shape:
        diff = adj_boots - base_boots
        finite = diff[np.isfinite(diff)]
        if finite.size >= 10:
            lo, hi = np.quantile(finite, [0.025, 0.975])
            se = float(np.std(finite, ddof=1))
            return {
                "change_from_baseline": float(d),
                "change_ci_lower": float(lo),
                "change_ci_upper": float(hi),
                "change_standard_error": se,
                "change_ci_method": "paired_cluster_bootstrap_coefficient_difference",
            }
    return {
        "change_from_baseline": float(d) if math.isfinite(d) else float("nan"),
        "change_ci_lower": float("nan"),
        "change_ci_upper": float("nan"),
        "change_standard_error": float("nan"),
        "change_ci_method": "point_difference_only",
    }


def _spec_row(
    *,
    spec_id: str,
    sample_scheme: str,
    fit: Mapping[str, object],
    baseline_fit: Mapping[str, object] | None,
    plot_order: int,
    analysis_family: str,
) -> dict[str, object]:
    status = STATUS_COMPUTED if fit.get("ok") else STATUS_NOT_COMPUTABLE
    if not fit.get("ok") and "rank" in _as_str(fit.get("reason")).casefold():
        status = STATUS_RANK_DEFICIENT
    est = _as_float(fit.get("estimate"))
    base_est = _as_float((baseline_fit or {}).get("estimate")) if baseline_fit else float("nan")
    change_info = (
        _delta_beta_from_boots(
            np.asarray((baseline_fit or {}).get("boot_intercepts", []), dtype=float),
            np.asarray(fit.get("boot_intercepts", fit.get("boot_estimates", [])), dtype=float),
            base_est,
            est,
        )
        if baseline_fit and fit.get("ok")
        else {
            "change_from_baseline": 0.0 if spec_id.endswith("baseline") or spec_id == "baseline" else float("nan"),
            "change_ci_lower": float("nan"),
            "change_ci_upper": float("nan"),
            "change_standard_error": float("nan"),
            "change_ci_method": "",
        }
    )
    if status != STATUS_COMPUTED:
        est = float("nan")
    return {
        "specification_id": spec_id,
        "display_label": SPEC_DISPLAY_LABELS.get(spec_id, spec_id),
        "analysis_family": analysis_family,
        "sample_scheme": sample_scheme,
        "adjustment_variables": ",".join(fit.get("kept_covariates") or fit.get("kept_nuisance") or []),
        "endpoint_name": PANEL_E_ENDPOINT,
        "duration_s": PANEL_E_DURATION_S,
        "band": PANEL_E_BAND,
        "power_representation": PANEL_E_REPRESENTATION,
        "coefficient_definition": (
            "ols_intercept_at_zero_nuisance_change"
            if analysis_family == "paired_delta"
            else "obslevel_state_coefficient_with_participant_fe"
        ),
        "estimate": est,
        "ci_lower": _as_float(fit.get("ci_lower")) if status == STATUS_COMPUTED else float("nan"),
        "ci_upper": _as_float(fit.get("ci_upper")) if status == STATUS_COMPUTED else float("nan"),
        "standard_error": _as_float(fit.get("standard_error")) if status == STATUS_COMPUTED else float("nan"),
        "p_value": _as_float(fit.get("p_value")) if status == STATUS_COMPUTED else float("nan"),
        "n_observations": int(fit.get("n_observations") or 0),
        "n_participants": int(fit.get("n_participants") or 0),
        "n_ecg_observations": int(fit.get("n_ecg") or 0),
        "n_ppg_observations": int(fit.get("n_ppg") or 0),
        **change_info,
        "direction_preserved": (
            "yes"
            if status == STATUS_COMPUTED
            and math.isfinite(est)
            and math.isfinite(base_est)
            and base_est != 0
            and math.copysign(1, est) == math.copysign(1, base_est)
            else ("baseline" if spec_id in {"baseline", "obs_baseline"} else "")
        ),
        "convergence_status": "converged" if status == STATUS_COMPUTED else "not_fit",
        "rank_deficient": bool(fit.get("rank_deficient")),
        "condition_number": _as_float(fit.get("condition_number")),
        "computability_status": status,
        "computability_reason": _as_str(fit.get("reason")),
        "short_reason_code": "" if status == STATUS_COMPUTED else "not computable",
        "plotted": status == STATUS_COMPUTED and analysis_family == "paired_delta" and spec_id in ESTIMABLE_SPEC_ORDER,
        "plot_order": plot_order,
        "predictors_centered": False,
        "zero_nuisance_interpretation": "no Rest–Tetris nuisance change",
    }


def _nuisance_state_summaries(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for var, rest_key, tet_key, delta_key in (
        ("mean_hr_bpm", "rest_mean_hr", "tetris_mean_hr", "delta_mean_hr"),
        ("broadband_power", "rest_broadband_power", "tetris_broadband_power", "delta_broadband_power"),
        (
            "broadband_power_ex_alpha",
            "rest_broadband_power_ex_alpha",
            "tetris_broadband_power_ex_alpha",
            "delta_broadband_power_ex_alpha",
        ),
    ):
        rest = np.asarray([_as_float(r.get(rest_key)) for r in rows], dtype=float)
        tet = np.asarray([_as_float(r.get(tet_key)) for r in rows], dtype=float)
        delta = np.asarray([_as_float(r.get(delta_key)) for r in rows], dtype=float)
        mask = np.isfinite(rest) & np.isfinite(tet)
        corr = float(np.corrcoef(rest[mask], tet[mask])[0, 1]) if int(mask.sum()) >= 3 else float("nan")
        d_mean, d_lo, d_hi = _paired_ci(delta)
        for state, arr in (("rest", rest), ("tetris", tet)):
            summary = _summarize_numeric(arr)
            out.append(
                {
                    "variable": var,
                    "state": state,
                    **summary,
                    "paired_correlation_rest_tetris": corr,
                    "mean_paired_change": d_mean,
                    "paired_change_ci_lower": d_lo,
                    "paired_change_ci_upper": d_hi,
                }
            )
    return out


def _audit_centered_identity(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Demonstrate legacy centered pair-average intercept equals unadjusted mean."""
    y = np.asarray([_as_float(r.get("delta_endpoint_index")) for r in rows], dtype=float)
    raw_mean = float(np.mean(y[np.isfinite(y)])) if np.any(np.isfinite(y)) else float("nan")
    audit: list[dict[str, object]] = []

    legacy_map = {
        "delta_mean_hr": ("legacy_pairavg_mean_hr", "mean_hr"),
        "delta_broadband_power": ("legacy_pairavg_broadband_power", "broadband_power"),
        "delta_hr_broadband": (
            ("legacy_pairavg_mean_hr", "legacy_pairavg_broadband_power"),
            ("mean_hr", "broadband_power"),
        ),
    }

    # Baseline
    base = fit_paired_intercept_cluster_boot(rows, (), seed=BOOTSTRAP_SEED)
    audit.append(
        {
            "specification_id": "baseline",
            "old_pairavg_nuisance": "",
            "new_within_pair_nuisance_change": "",
            "old_centered_intercept": base.get("estimate"),
            "corrected_uncentered_delta_intercept": base.get("estimate"),
            "corrected_ci_lower": base.get("ci_lower"),
            "corrected_ci_upper": base.get("ci_upper"),
            "delta_beta_from_baseline": 0.0,
            "raw_mean_delta_zlpi": raw_mean,
            "abs_old_intercept_minus_raw_mean": abs(_as_float(base.get("estimate")) - raw_mean),
            "explanation": (
                "Baseline has no covariates; intercept equals mean paired ΔZLPI by construction."
            ),
            "zero_delta_beta_was": "n/a",
        }
    )

    for spec_id, (legacy_keys, _label) in (
        ("delta_mean_hr", (("legacy_pairavg_mean_hr",), "mean_hr")),
        ("delta_broadband_power", (("legacy_pairavg_broadband_power",), "broadband_power")),
        (
            "delta_hr_broadband",
            (("legacy_pairavg_mean_hr", "legacy_pairavg_broadband_power"), "joint"),
        ),
    ):
        new_covs = SPEC_COVARIATES[spec_id]
        # Map legacy keys for centered audit fit
        legacy_rows = []
        for r in rows:
            item = dict(r)
            # expose legacy keys under short names for centered design
            if "legacy_pairavg_mean_hr" in item:
                item["mean_hr"] = item["legacy_pairavg_mean_hr"]
            if "legacy_pairavg_broadband_power" in item:
                item["broadband_power"] = item["legacy_pairavg_broadband_power"]
            legacy_rows.append(item)
        legacy_cov_names = {
            "delta_mean_hr": ("mean_hr",),
            "delta_broadband_power": ("broadband_power",),
            "delta_hr_broadband": ("mean_hr", "broadband_power"),
        }[spec_id]

        old = fit_paired_intercept_cluster_boot(
            legacy_rows,
            legacy_cov_names,
            seed=BOOTSTRAP_SEED + 3,
            center=True,
            legacy_cov_keys=legacy_cov_names,
        )
        new = fit_paired_intercept_cluster_boot(rows, new_covs, seed=BOOTSTRAP_SEED + 5)
        y2, x_c, kept, means = _design_matrix_centered_legacy(legacy_rows, legacy_cov_names)
        col_means = x_c[:, 1:].mean(axis=0) if x_c.shape[1] > 1 else np.asarray([])
        abs_diff = abs(_as_float(old.get("estimate")) - raw_mean)
        explanation = (
            "Centered pair-average covariates make X'1 orthogonal for covariate columns; "
            f"normal equations give b0 = mean(y). |b0-mean(y)|={abs_diff:.3e}. "
            f"Covariate means before centering={means}; design col means after centering="
            f"{[float(v) for v in col_means]}. "
            "Corrected model uses uncentered Tetris−Rest changes; intercept is E[ΔZLPI|ΔN=0]."
        )
        audit.append(
            {
                "specification_id": spec_id,
                "old_pairavg_nuisance": ",".join(legacy_keys),
                "new_within_pair_nuisance_change": ",".join(new_covs),
                "old_centered_intercept": old.get("estimate"),
                "corrected_uncentered_delta_intercept": new.get("estimate"),
                "corrected_ci_lower": new.get("ci_lower"),
                "corrected_ci_upper": new.get("ci_upper"),
                "delta_beta_from_baseline": (
                    _as_float(new.get("estimate")) - _as_float(base.get("estimate"))
                ),
                "raw_mean_delta_zlpi": raw_mean,
                "abs_old_intercept_minus_raw_mean": abs_diff,
                "old_covariate_means_before_centering": json.dumps(means),
                "old_design_col_means_after_centering": json.dumps([float(v) for v in col_means]),
                "explanation": explanation,
                "zero_delta_beta_was": (
                    "algebraically_guaranteed_by_centered_covariate_ols"
                    if abs_diff < 1e-10
                    else "empirical_or_mixed"
                ),
            }
        )
    return audit


def compute_panel_e_nuisance_modality(
    *,
    paired_rows: Sequence[Mapping[str, object]],
    aligned_rows: Sequence[Mapping[str, object]],
    data_audit_rows: Sequence[Mapping[str, object]] = (),
    peak_qc_rows: Sequence[Mapping[str, object]] = (),
    protocol_rows: Sequence[Mapping[str, object]] = (),
    endpoint_rows: Sequence[Mapping[str, object]] = (),
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> PanelEResult:
    """Compute corrected Panel E paired Δ-nuisance robustness analysis."""
    del bootstrap_draws
    contrast_rows = build_panel_e_contrast_rows(
        paired_rows=paired_rows,
        aligned_rows=aligned_rows,
        data_audit_rows=data_audit_rows,
        peak_qc_rows=peak_qc_rows,
        protocol_rows=protocol_rows,
    )
    modality_info = assess_modality_identifiability(contrast_rows)

    # Common sample: finite ΔZLPI and finite primary Δ-nuisances (HR + ex-alpha broadband).
    # Also require alpha-inclusive broadband finite so sensitivity specs share the same n.
    common_rows = [
        r
        for r in contrast_rows
        if math.isfinite(_as_float(r.get("delta_endpoint_index")))
        and math.isfinite(_as_float(r.get("delta_mean_hr")))
        and math.isfinite(_as_float(r.get("delta_broadband_power_ex_alpha")))
        and math.isfinite(_as_float(r.get("delta_broadband_power")))
    ]

    n_ecg = sum(1 for r in common_rows if _as_str(r.get("cardiac_signal_type")) == "ecg")
    n_ppg = sum(1 for r in common_rows if _as_str(r.get("cardiac_signal_type")) == "ppg")

    def _annotate(fit: dict[str, object]) -> dict[str, object]:
        out = dict(fit)
        out["n_ecg"] = n_ecg
        out["n_ppg"] = n_ppg
        return out

    def _fit_spec(spec_id: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return _annotate(
            fit_paired_intercept_cluster_boot(
                rows,
                SPEC_COVARIATES[spec_id],
                seed=bootstrap_seed + int(SPEC_SEED_OFFSET[spec_id]),
            )
        )

    # Primary paired fits on common sample (ex-alpha broadband preferred).
    fits: dict[str, dict[str, object]] = {}
    for spec_id in ESTIMABLE_SPEC_ORDER:
        fits[spec_id] = _fit_spec(spec_id, common_rows)

    # Alpha-inclusive broadband retained as sensitivity-only (potentially circular).
    for spec_id in SENSITIVITY_SPEC_ORDER:
        fits[spec_id] = _fit_spec(spec_id, common_rows)

    base_fit = fits["baseline"]
    specification_rows: list[dict[str, object]] = []
    common_sample_rows: list[dict[str, object]] = []
    for order, spec_id in enumerate(ESTIMABLE_SPEC_ORDER, start=1):
        row = _spec_row(
            spec_id=spec_id,
            sample_scheme="common_sample",
            fit=fits[spec_id],
            baseline_fit=base_fit,
            plot_order=order,
            analysis_family="paired_delta",
        )
        specification_rows.append(row)
        common_sample_rows.append(dict(row))
    for order, spec_id in enumerate(SENSITIVITY_SPEC_ORDER, start=5):
        row = _spec_row(
            spec_id=spec_id,
            sample_scheme="common_sample_alpha_inclusive_sensitivity",
            fit=fits[spec_id],
            baseline_fit=base_fit,
            plot_order=order,
            analysis_family="paired_delta_sensitivity_circular",
        )
        row["plotted"] = False
        row["circularity_warning"] = (
            "Alpha-inclusive broadband partially overlaps the alpha ZLPI outcome; "
            "sensitivity only, not an independent nuisance control."
        )
        specification_rows.append(row)

    # Observation-level sensitivity.
    obs_rows = build_observation_level_rows(common_rows, endpoint_rows)
    obs_level_out: list[dict[str, object]] = []
    obs_specs = [
        ("obs_baseline", ()),
        ("obs_mean_hr", ("mean_hr",)),
        ("obs_broadband_power", ("broadband_power",)),
        ("obs_hr_broadband", ("mean_hr", "broadband_power")),
    ]
    obs_base = None
    for order, (spec_id, nui) in enumerate(obs_specs, start=1):
        fit = fit_obslevel_state_cluster_boot(
            obs_rows, nui, seed=bootstrap_seed + 50 + order
        )
        fit = _annotate(fit)
        if spec_id == "obs_baseline":
            obs_base = fit
            # Align boot key name for Δβ helper
            fit["boot_intercepts"] = fit.get("boot_estimates", np.asarray([]))
        else:
            fit["boot_intercepts"] = fit.get("boot_estimates", np.asarray([]))
        if obs_base is not None and "boot_intercepts" not in obs_base:
            obs_base["boot_intercepts"] = obs_base.get("boot_estimates", np.asarray([]))
        row = _spec_row(
            spec_id=spec_id,
            sample_scheme="observation_level_sensitivity",
            fit=fit,
            baseline_fit=obs_base,
            plot_order=order,
            analysis_family="observation_level",
        )
        row["plotted"] = False
        obs_level_out.append(row)

    audit_rows = _audit_centered_identity(common_rows)
    nuisance_summaries = _nuisance_state_summaries(common_rows)

    # Missingness / availability summary for planned variables.
    missingness_rows = [
        {
            "variable": "delta_mean_hr",
            "available": True,
            "n_finite": sum(1 for r in contrast_rows if math.isfinite(_as_float(r.get("delta_mean_hr")))),
            "n_total": len(contrast_rows),
            "definition": "mean_hr_tetris - mean_hr_rest (bpm); D240 segment means from hr_bpm",
        },
        {
            "variable": "delta_broadband_power_ex_alpha",
            "available": True,
            "n_finite": sum(
                1
                for r in contrast_rows
                if math.isfinite(_as_float(r.get("delta_broadband_power_ex_alpha")))
            ),
            "n_total": len(contrast_rows),
            "definition": json.dumps(BROADBAND_DEFINITION["primary"]),
        },
        {
            "variable": "delta_broadband_power",
            "available": True,
            "n_finite": sum(
                1 for r in contrast_rows if math.isfinite(_as_float(r.get("delta_broadband_power")))
            ),
            "n_total": len(contrast_rows),
            "definition": json.dumps(BROADBAND_DEFINITION["alpha_inclusive_sensitivity"]),
        },
        {
            "variable": "motion",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "not retained",
        },
        {
            "variable": "eog",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "not retained",
        },
        {
            "variable": "emg",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "not retained",
        },
        {
            "variable": "bad_window_rate",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "not retained",
        },
        {
            "variable": "respiration",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "protocol-declared only",
        },
        {
            "variable": "eye_state",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": "not identifiable: determined by Rest/Tetris",
        },
        {
            "variable": "modality_ecg_ppg",
            "available": False,
            "n_finite": 0,
            "n_total": len(contrast_rows),
            "definition": modality_info.get("reason"),
        },
    ]

    diagnostic_rows = [
        {
            "diagnostic_type": "centered_covariate_identity",
            "raw_mean_delta_zlpi": audit_rows[0]["raw_mean_delta_zlpi"] if audit_rows else float("nan"),
            "max_abs_old_intercept_minus_mean": max(
                (_as_float(r.get("abs_old_intercept_minus_raw_mean")) for r in audit_rows),
                default=float("nan"),
            ),
            "conclusion": (
                "Previous zero Δβ values were algebraically guaranteed by centered "
                "pair-average covariate OLS, not empirical coefficient stability."
            ),
        },
        {
            "diagnostic_type": "modality_identifiability",
            **{k: modality_info.get(k) for k in ("identifiable", "reason", "signals_present", "n_ecg", "n_ppg")},
        },
        {
            "diagnostic_type": "broadband_definition_primary_ex_alpha",
            **BROADBAND_DEFINITION["primary"],
        },
        {
            "diagnostic_type": "broadband_definition_alpha_inclusive_sensitivity",
            **BROADBAND_DEFINITION["alpha_inclusive_sensitivity"],
        },
        {
            "diagnostic_type": "ex_alpha_vs_alpha_inclusive_difference",
            "conclusion": (
                "Preferred broadband excludes alpha. Alpha-inclusive broadband is "
                "sensitivity-only because it partially overlaps the alpha ZLPI outcome; "
                "the two adjustments can shift Δβ in opposite directions and the "
                "alpha-inclusive model must not be characterized as an independent "
                "nuisance control."
            ),
        },
    ]

    metadata = {
        "schema_version": "figure3_panel_e_nuisance_modality_v2_1_ex_alpha_primary",
        "title": PANEL_E_FIGURE_TITLE,
        "subtitle": PANEL_E_SUBTITLE,
        "stem": PANEL_E_STEM,
        "primary_analysis": "paired_delta_uncentered_within_pair_nuisance_change_ex_alpha_broadband",
        "sensitivity_analysis": (
            "observation_level_state_plus_nuisance_with_participant_fe; "
            "alpha_inclusive_broadband_paired_delta_circularity_sensitivity"
        ),
        "locked_estimand": {
            "duration_s": PANEL_E_DURATION_S,
            "endpoint_name": PANEL_E_ENDPOINT,
            "band": PANEL_E_BAND,
            "power_representation": PANEL_E_REPRESENTATION,
            "outcome": "paired Rest–Tetris delta_endpoint_index (Tetris − Rest)",
            "coefficient_definition": (
                "OLS intercept of ΔZLPI ~ 1 + ΔNuisance(s), evaluated at zero "
                "within-pair nuisance change (predictors not mean-centered)"
            ),
            "preferred_broadband": "theta+beta+low_gamma absolute_log10 (excludes alpha)",
            "clustering_unit": "participant_id",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": bootstrap_seed,
            "spec_seed_offsets": dict(SPEC_SEED_OFFSET),
        },
        "n_contrasts_available": len(contrast_rows),
        "n_contrasts_common_sample": len(common_rows),
        "n_participants_common_sample": len(
            {_as_str(r.get("participant_id")) for r in common_rows}
        ),
        "n_ecg": n_ecg,
        "n_ppg": n_ppg,
        "modality_identifiability": modality_info,
        "broadband_definition": BROADBAND_DEFINITION,
        "unavailable_table": list(UNAVAILABLE_TABLE),
        "estimable_spec_order": list(ESTIMABLE_SPEC_ORDER),
        "sensitivity_spec_order": list(SENSITIVITY_SPEC_ORDER),
        "no_imputation_policy": "Non-finite values never replaced with zero; no proxies.",
        "legacy_method_status": (
            "pair-average + centered covariates retained only in "
            "figure3_panel_e_nuisance_modality_audit_old_vs_new.csv; not used for manuscript conclusion"
        ),
        "algebraic_audit_conclusion": (
            "Original zero Δβ values were a consequence of centered-covariate OLS algebra "
            "(intercept identically equals unadjusted mean ΔZLPI), not an empirical stability finding."
        ),
        "hiit_robustness_scope": (
            "HIIT alone cannot support full nuisance/modality robustness claims: most planned "
            "nuisance variables are unavailable and ECG/PPG modality has no variation. "
            "Broader multi-dataset execution is required to test ECG-versus-PPG modality."
        ),
        "common_sample_note": (
            f"Common sample: {len(common_rows)} paired contrasts from "
            f"{len({_as_str(r.get('participant_id')) for r in common_rows})} participants; "
            f"PPG {n_ppg}, ECG {n_ecg}"
        ),
    }

    obs_export_fields = [
        "dataset_id",
        "participant_id",
        "session_id",
        "contrast_id",
        "observation_id",
        "low_observation_id",
        "effort_observation_id",
        "delta_endpoint_index",
        "rest_mean_hr",
        "tetris_mean_hr",
        "delta_mean_hr",
        "rest_broadband_power",
        "tetris_broadband_power",
        "delta_broadband_power",
        "rest_broadband_power_ex_alpha",
        "tetris_broadband_power_ex_alpha",
        "delta_broadband_power_ex_alpha",
        "legacy_pairavg_mean_hr",
        "legacy_pairavg_broadband_power",
        "cardiac_signal_type",
    ]
    observation_export = [{k: r.get(k, "") for k in obs_export_fields} for r in contrast_rows]

    return PanelEResult(
        observation_rows=tuple(observation_export),
        specification_rows=tuple(specification_rows),
        common_sample_rows=tuple(common_sample_rows),
        observation_level_rows=tuple(obs_level_out),
        diagnostic_rows=tuple(diagnostic_rows),
        missingness_rows=tuple(missingness_rows),
        audit_old_vs_new_rows=tuple(audit_rows),
        nuisance_state_summary_rows=tuple(nuisance_summaries),
        metadata=metadata,
    )


def write_panel_e_exports(result: PanelEResult, source_dir: Path) -> dict[str, Path]:
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _dump(name: str, rows: Sequence[Mapping[str, object]]) -> Path:
        path = source_dir / f"{PANEL_E_STEM}_{name}.csv"
        fields = sorted({k for r in rows for k in r.keys()}) if rows else ["placeholder"]
        _write_csv(path, rows, fields)
        return path

    paths["observations"] = _dump("observation_level", result.observation_rows)
    paths["specifications"] = _dump("specifications", result.specification_rows)
    paths["common_sample"] = _dump("common_sample", result.common_sample_rows)
    paths["observation_level_models"] = _dump(
        "observation_level_models", result.observation_level_rows
    )
    paths["diagnostics"] = _dump("diagnostics", result.diagnostic_rows)
    paths["missingness"] = _dump("missingness", result.missingness_rows)
    paths["audit_old_vs_new"] = _dump("audit_old_vs_new", result.audit_old_vs_new_rows)
    paths["nuisance_state_summary"] = _dump(
        "nuisance_state_summary", result.nuisance_state_summary_rows
    )
    paths["metadata"] = source_dir / f"{PANEL_E_STEM}_metadata.json"
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return paths


def panel_e_caption(result: PanelEResult) -> str:
    meta = result.metadata
    n_pairs = meta.get("n_contrasts_common_sample")
    n_part = meta.get("n_participants_common_sample")
    lines = [
        PANEL_E_FIGURE_TITLE,
        "",
        PANEL_E_SUBTITLE + ".",
        "",
        f"Estimates use {n_pairs} common-sample paired contrasts from {n_part} participants "
        f"(PPG {meta.get('n_ppg')}, ECG {meta.get('n_ecg')}). "
        "Nuisance predictors are uncentered Tetris−Rest changes; the plotted coefficient "
        "is the OLS intercept (expected ΔZLPI at zero nuisance change) with "
        "participant-clustered bootstrap 95% CIs. Δβ uncertainty uses paired bootstrap "
        "coefficient differences. "
        "Mean HR adjustment produces negligible coefficient change. "
        "Broadband excluding alpha (theta/beta/low-gamma absolute log10 power) is the "
        "preferred power adjustment; alpha-inclusive broadband is sensitivity-only because "
        "it partially overlaps the alpha series entering ZLPI and is not an independent "
        "nuisance control. "
        "Unavailable variables were not proxied or imputed. "
        "Eye state and ECG/PPG modality are non-identifiable in this HIIT-only dataset; "
        "broader datasets are required to assess ECG-versus-PPG robustness. "
        "Combinations of missing channels (for example EOG with eye state, or modality "
        "with other physiology) remain untestable for the same reasons.",
    ]
    return "\n".join(lines) + "\n"


def verify_panel_e_integrity(result: PanelEResult) -> list[str]:
    failures: list[str] = []
    plotted = [r for r in result.common_sample_rows if r.get("plotted")]
    for row in plotted:
        if _as_str(row.get("endpoint_name")) != PANEL_E_ENDPOINT:
            failures.append("plotted row estimand mismatch")
        if bool(row.get("predictors_centered")):
            failures.append(f"{row.get('specification_id')}: predictors unexpectedly centered")
        if row.get("computability_status") != STATUS_COMPUTED:
            failures.append(f"{row.get('specification_id')}: non-computed plotted")
        if not math.isfinite(_as_float(row.get("estimate"))):
            failures.append(f"{row.get('specification_id')}: plotted without finite estimate")
        sid = _as_str(row.get("specification_id"))
        if sid not in ESTIMABLE_SPEC_ORDER:
            failures.append(f"{sid}: plotted but not in primary estimable order")
        if "including alpha" in _as_str(row.get("display_label")).casefold():
            failures.append(f"{sid}: alpha-inclusive shown as preferred manuscript row")

    for row in list(result.specification_rows) + list(result.common_sample_rows):
        if row.get("computability_status") != STATUS_COMPUTED:
            est = _as_float(row.get("estimate"))
            if math.isfinite(est) and est == 0.0:
                failures.append(f"{row.get('specification_id')}: NC assigned numerical zero")

    plotted_ids = {_as_str(r.get("specification_id")) for r in plotted}
    if plotted_ids != set(ESTIMABLE_SPEC_ORDER):
        failures.append(
            f"plotted ids {sorted(plotted_ids)} != primary {list(ESTIMABLE_SPEC_ORDER)}"
        )
    if "delta_broadband_power" in plotted_ids or "delta_hr_broadband" in plotted_ids:
        failures.append("alpha-inclusive broadband plotted as preferred manuscript adjustment")

    for row in result.common_sample_rows:
        sid = _as_str(row.get("specification_id"))
        if sid == "baseline":
            continue
        adj = _as_str(row.get("adjustment_variables"))
        if adj and "delta_" not in adj:
            failures.append(f"{sid}: predictors are not within-pair deltas")

    base = next((r for r in result.common_sample_rows if r.get("specification_id") == "baseline"), None)
    if base:
        base_est = _as_float(base.get("estimate"))
        for row in result.common_sample_rows:
            if row.get("specification_id") == "baseline":
                continue
            if row.get("computability_status") != STATUS_COMPUTED:
                continue
            if bool(row.get("predictors_centered")) and abs(
                _as_float(row.get("estimate")) - base_est
            ) < 1e-12:
                failures.append(
                    f"{row.get('specification_id')}: centered construction forced equality"
                )

    return failures


def _wrap_table_text(text: str, width: int = 58) -> str:
    words = str(text).split()
    if not words:
        return ""
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def render_panel_e_figure(
    result: PanelEResult,
    output_dir: Path,
    *,
    sample_scheme: str = "common_sample",
    include_internal_qc: bool = True,
) -> dict[str, Path]:
    """Render polished Panel E: estimable forest + unavailable table (display-only)."""
    from .figures import (
        AXIS_LINE_WIDTH,
        FS_AXIS,
        FS_TICK,
        PALETTE,
        _configure_publication_style,
        save_figure_trio,
    )
    from matplotlib.ticker import FixedLocator, FuncFormatter

    del sample_scheme
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    export_paths = write_panel_e_exports(result, source_dir)
    caption_path = output_dir / f"{PANEL_E_STEM}_caption.txt"
    caption_path.write_text(panel_e_caption(result), encoding="utf-8")

    integrity = verify_panel_e_integrity(result)
    if integrity:
        raise ValueError("Panel E integrity failed: " + "; ".join(integrity))

    rows = [
        r
        for r in result.common_sample_rows
        if r.get("specification_id") in ESTIMABLE_SPEC_ORDER and r.get("plotted")
    ]
    by_id = {_as_str(r.get("specification_id")): r for r in rows}
    ordered = [by_id[s] for s in ESTIMABLE_SPEC_ORDER if s in by_id]

    n_obs_set = {int(r.get("n_observations") or 0) for r in ordered}
    n_part_set = {int(r.get("n_participants") or 0) for r in ordered}
    sample_identical = len(n_obs_set) == 1 and len(n_part_set) == 1

    _configure_publication_style()
    fig = plt.figure(figsize=(14.4, 8.4), constrained_layout=False)
    gs = GridSpec(
        2,
        2 if sample_identical else 3,
        figure=fig,
        height_ratios=[1.65, 1.0],
        width_ratios=[2.55, 1.35] if sample_identical else [2.3, 1.15, 1.05],
        hspace=0.55,
        wspace=0.22,
        left=0.28,
        right=0.985,
        top=0.78,
        bottom=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_info = None if sample_identical else fig.add_subplot(gs[0, 2], sharey=ax)
    ax_table = fig.add_subplot(gs[1, :])

    n = len(ordered)
    y_pos = np.arange(n, dtype=float)[::-1]
    baseline_est = float("nan")
    for r in ordered:
        if r.get("specification_id") == "baseline":
            baseline_est = _as_float(r.get("estimate"))

    x_vals = [0.0]
    for yi, row in zip(y_pos, ordered, strict=True):
        est = _as_float(row.get("estimate"))
        lo = _as_float(row.get("ci_lower"))
        hi = _as_float(row.get("ci_upper"))
        is_base = row.get("specification_id") == "baseline"
        color = PALETTE["dark_gray"] if is_base else PALETTE["blue"]
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.0, solid_capstyle="round", zorder=2)
            x_vals.extend([lo, hi])
        ax.plot(
            est,
            yi,
            marker="D" if is_base else "o",
            color=color,
            markersize=9 if is_base else 8,
            zorder=3,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.4,
        )
        x_vals.append(est)

        d = _as_float(row.get("change_from_baseline"))
        dlo = _as_float(row.get("change_ci_lower"))
        dhi = _as_float(row.get("change_ci_upper"))
        if is_base:
            d = 0.0
        if math.isfinite(dlo) and math.isfinite(dhi) and not is_base:
            ax_delta.plot([dlo, dhi], [yi, yi], color=PALETTE["blue"], lw=1.6, zorder=2)
        ax_delta.plot(
            d,
            yi,
            marker="o",
            color=PALETTE["blue"] if not is_base else PALETTE["dark_gray"],
            markersize=7,
            zorder=3,
        )

        if ax_info is not None:
            ax_info.text(
                0.02,
                yi,
                f"n={int(row.get('n_observations') or 0)}   "
                f"Participants={int(row.get('n_participants') or 0)}",
                va="center",
                ha="left",
                fontsize=FS_TICK - 2,
                color=PALETTE["dark_gray"],
                transform=ax_info.get_yaxis_transform(),
            )

    ax.axvline(0.0, color="#444444", ls="-", lw=1.2, zorder=1)
    if math.isfinite(baseline_est):
        ax.axvline(baseline_est, color=PALETTE["orange"], ls="--", lw=1.35, zorder=1)
    ax_delta.axvline(0.0, color="#444444", ls="-", lw=1.15, zorder=1)

    labels = [_as_str(r.get("display_label")) for r in ordered]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax_delta.tick_params(axis="y", left=False, labelleft=False)
    if ax_info is not None:
        ax_info.set_axis_off()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
        ax_delta.spines[spine].set_visible(False)

    finite_x = [v for v in x_vals if math.isfinite(v)]
    if finite_x:
        lo_v, hi_v = min(finite_x), max(finite_x)
        span = hi_v - lo_v if hi_v > lo_v else max(abs(hi_v), 0.05)
        pad = 0.18 * span
        ax.set_xlim(lo_v - pad, hi_v + pad)

    dvals = []
    for r in ordered:
        dvals.append(_as_float(r.get("change_from_baseline")))
        dvals.append(_as_float(r.get("change_ci_lower")))
        dvals.append(_as_float(r.get("change_ci_upper")))
    dfin = [v for v in dvals if math.isfinite(v)]
    if dfin:
        m = max(abs(min(dfin)), abs(max(dfin)), 0.05)
        lim = max(math.ceil(m * 20) / 20.0, 0.05)
        ax_delta.set_xlim(-lim, lim)
        tick_candidates = [-lim, -0.05, 0.0, 0.05, lim]
        ticks = sorted({round(t, 10) for t in tick_candidates if abs(t) <= lim + 1e-12})
        ax_delta.xaxis.set_major_locator(FixedLocator(ticks))

        def _fmt_tick2(v: float, _pos: int | None = None) -> str:
            if abs(v) < 1e-12:
                return "0"
            if abs(v - 0.05) < 1e-12:
                return "+0.05"
            if abs(v + 0.05) < 1e-12:
                return "−0.05"
            if abs(v - 0.10) < 1e-12:
                return "+0.10"
            if abs(v + 0.10) < 1e-12:
                return "−0.10"
            return f"{v:+.2f}"

        ax_delta.xaxis.set_major_formatter(FuncFormatter(_fmt_tick2))

    ax.set_ylim(-0.55, n - 0.45)
    ax_delta.set_ylim(-0.55, n - 0.45)
    ax.set_title("Adjusted estimate, 95% CI", fontsize=FS_TICK - 1, pad=8, loc="left")
    ax_delta.set_title("Change from baseline, 95% CI", fontsize=FS_TICK - 1, pad=8, loc="left")
    ax.set_xlabel("Adjusted Rest–Tetris ΔZLPI\nAlpha, Fisher z", fontsize=FS_AXIS - 3)
    ax_delta.set_xlabel("Δβ from baseline\nPaired-bootstrap 95% CI", fontsize=FS_AXIS - 3)
    for a in (ax, ax_delta):
        a.grid(False)
        for yi in y_pos:
            a.axhline(yi, color="#F2F2F2", lw=0.8, zorder=0)
        for spine in ("bottom", "left"):
            a.spines[spine].set_linewidth(AXIS_LINE_WIDTH)

    handles = [
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=PALETTE["dark_gray"], markeredgecolor=PALETTE["dark_gray"],
            markersize=8, label="Baseline",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=PALETTE["blue"], markeredgecolor=PALETTE["dark_gray"],
            markersize=8, label="Adjusted",
        ),
        Line2D([0], [0], color=PALETTE["orange"], ls="--", lw=1.35, label="Baseline estimate"),
        Line2D([0], [0], color="#444444", ls="-", lw=1.2, label="Null"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.845),
        ncol=4,
        frameon=False,
        fontsize=FS_TICK - 2,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.suptitle(PANEL_E_FIGURE_TITLE, fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, PANEL_E_SUBTITLE, ha="center", va="top", fontsize=FS_TICK - 1, color="#555555")
    fig.text(0.5, 0.905, "A. Estimable specifications", ha="center", va="top", fontsize=FS_TICK - 1, color="#333333")
    sample_note = _as_str(
        result.metadata.get("common_sample_note"),
        (
            f"Common sample: {result.metadata.get('n_contrasts_common_sample')} paired "
            f"contrasts from {result.metadata.get('n_participants_common_sample')} "
            f"participants; PPG {result.metadata.get('n_ppg')}, "
            f"ECG {result.metadata.get('n_ecg')}"
        ),
    )
    fig.text(0.5, 0.875, sample_note, ha="center", va="top", fontsize=FS_TICK - 2, color="#555555")

    ax_table.set_axis_off()
    ax_table.set_title("B. Planned but unavailable checks", loc="left", fontsize=FS_TICK, pad=14)
    table_data = [
        [r["planned_adjustment"], r["status"], _wrap_table_text(r["reason"], width=58)]
        for r in UNAVAILABLE_TABLE
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Planned check", "Status", "Reason"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.02, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FS_TICK - 2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        cell.PAD = 0.02
        cell.set_text_props(ha="left", va="center")
        if col == 0:
            cell.set_width(0.18)
        elif col == 1:
            cell.set_width(0.16)
        else:
            cell.set_width(0.66)
        if row == 0:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(weight="bold", ha="left", va="center")
            cell.set_height(0.10)
        else:
            cell.set_facecolor("white")
            n_lines = max(1, table_data[row - 1][2].count("\n") + 1)
            cell.set_height(0.10 + 0.035 * (n_lines - 1))

    trio = save_figure_trio(fig, output_dir, PANEL_E_STEM, bbox_inches=None, pad_inches=0.35)
    out: dict[str, Path] = {
        "pdf": trio[0],
        "svg": trio[1],
        "png": trio[2],
        "caption": caption_path,
        **export_paths,
    }

    if include_internal_qc:
        qc_dir = output_dir / "internal_qc" / "panel_e_nuisance_modality"
        qc_dir.mkdir(parents=True, exist_ok=True)
        sens = [
            r for r in result.specification_rows
            if _as_str(r.get("specification_id")) in SENSITIVITY_SPEC_ORDER
        ]
        note = qc_dir / "panel_e_qc_notes.txt"
        note.write_text(
            "\n".join(
                [
                    "Panel E internal QC (v2.1 ex-alpha primary)",
                    str(result.metadata.get("algebraic_audit_conclusion", "")),
                    str(result.metadata.get("hiit_robustness_scope", "")),
                    "",
                    "Primary (manuscript) coefficients:",
                    *[
                        f"{r.get('specification_id')}: {r.get('estimate')} "
                        f"CI=[{r.get('ci_lower')},{r.get('ci_upper')}] "
                        f"Δβ={r.get('change_from_baseline')}"
                        for r in result.common_sample_rows
                    ],
                    "",
                    "Alpha-inclusive broadband sensitivity (NOT independent nuisance control):",
                    *[
                        f"{r.get('specification_id')}: {r.get('estimate')} "
                        f"CI=[{r.get('ci_lower')},{r.get('ci_upper')}] "
                        f"Δβ={r.get('change_from_baseline')} | {r.get('circularity_warning', '')}"
                        for r in sens
                    ],
                    "",
                    "Observation-level sensitivity:",
                    *[
                        f"{r.get('specification_id')}: {r.get('estimate')} "
                        f"CI=[{r.get('ci_lower')},{r.get('ci_upper')}]"
                        for r in result.observation_level_rows
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out["qc_notes"] = note

    plt.close(fig)
    return out


__all__ = [
    "PANEL_E_BAND",
    "PANEL_E_DURATION_S",
    "PANEL_E_ENDPOINT",
    "PANEL_E_REPRESENTATION",
    "PANEL_E_STEM",
    "PANEL_E_FIGURE_TITLE",
    "ESTIMABLE_SPEC_ORDER",
    "SENSITIVITY_SPEC_ORDER",
    "SPEC_SEED_OFFSET",
    "PanelEResult",
    "assess_modality_identifiability",
    "build_panel_e_contrast_rows",
    "compute_panel_e_nuisance_modality",
    "panel_e_caption",
    "write_panel_e_exports",
    "verify_panel_e_integrity",
    "render_panel_e_figure",
    "fit_paired_intercept_cluster_boot",
    "BROADBAND_DEFINITION",
]
