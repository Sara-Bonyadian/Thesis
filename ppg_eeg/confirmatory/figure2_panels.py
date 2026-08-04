"""Figure 2 six-panel presentation layout (C7 only).

Does not alter C0–C6 analyses. Panels A/B/E use exact C5 paired-intersection
participants linked to C2 lag curves via observation IDs when PRIMARY_META pairs
are present; sensitivity-only runs fall back to display-only
low-demand–high-demand sensitivity. No unpaired fallback. Panel F remains
ds003690-graded only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

from .duration_contracts import (
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    contract_for_duration,
)
from .endpoints import fisher_z
from .forest_display import (
    FOREST_EXPORT_FIELDS,
    HIIT_ALL_CONTRASTS,
    HIIT_DATASET_ID,
    ROW_TYPE_SENSITIVITY_DISPLAY,
    _hiit_session_unit_key,
    build_alpha_forest_export,
    draw_alpha_meta_forest,
    hiit_session_sensitivity_forest_rows,
    primary_meta_alpha_forest_rows,
)
from .inference import (
    MIXED_MODEL_CONTRAST_FIELDS,
    MIXED_MODEL_MARGINAL_FIELDS,
    PRIMARY_META_CONTRASTS,
    fit_mixed_model,
)
from .manifest import FigurePanelSource
from .protocol_audit import PROTOCOL_SPECS

GRADED_DS003690_CONTRASTS = frozenset({"passive__simplert", "passive__gonogo"})
GRADED_DATASET_ID = "ds003690"
HIIT_COMBINED_CONTRAST_ID = "hiit_combined_ph_ps_pre_post_mean"
HIIT_SESSION_LAG_AGGREGATION = "session_subject_mean_of_available_pre_post_curves"
HIIT_MATCHED_PAIR_AGGREGATION = "matched_observation_pairs"
HIIT_CLUSTER_BOOTSTRAP_CI_METHOD = "session_subject_cluster_bootstrap"

PANEL_D_DISPLAY_BANDS = ("theta", "alpha", "beta", "gamma")
PANEL_D_STATE_SHORT = {
    "low cognitive demand": "Low",
    "high cognitive demand": "High",
}
PANEL_D_MARGINAL_EXPORT_FIELDS = MIXED_MODEL_MARGINAL_FIELDS
PANEL_D_CONTRAST_EXPORT_FIELDS = MIXED_MODEL_CONTRAST_FIELDS
# Contrast-strip layout (display only; top → bottom).
PANEL_D_ALPHA_CONTRAST_ORDER = ("gamma", "beta", "theta")
PANEL_D_CONTRAST_ROW_SPACING = 1.75
PANEL_D_MAIN_CONTRAST_HEIGHT_RATIOS = (2.15, 1.85)
PANEL_D_MAIN_CONTRAST_HSPACE = 0.72


def _fig():
    from . import figures as f

    return f


def _split_observation_ids(raw: object) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parts: list[str] = []
    for chunk in text.replace(",", ";").split(";"):
        token = chunk.strip()
        if token:
            parts.append(token)
    return parts


def _primary_meta_key(dataset_id: str, contrast_id: str) -> tuple[str, str]:
    return (dataset_id.casefold(), contrast_id.casefold())


def filter_primary_meta_paired_rows(
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Exact C5 pairs for PRIMARY_META contrasts (one contrast per dataset)."""
    f = _fig()
    selected: list[dict[str, object]] = []
    seen_participant: set[tuple[str, str, str]] = set()
    for row in paired_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        contrast_id = f._as_str(row.get("contrast_id")).casefold()
        if _primary_meta_key(dataset_id, contrast_id) not in PRIMARY_META_CONTRASTS:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
            EXPECTED_PRIMARY_DURATION_S
        ):
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if "contrast_eligible" in row and not f._as_bool(row.get("contrast_eligible")):
            continue
        participant_id = f._as_str(row.get("participant_id"))
        band = f._as_str(row.get("band")).casefold()
        unit_key = (dataset_id, participant_id, band)
        # One contrast per dataset is already enforced by PRIMARY_META; still
        # guard against duplicate rows for the same participant×band.
        if unit_key in seen_participant:
            continue
        seen_participant.add(unit_key)
        selected.append(dict(row))
    return selected


def filter_hiit_sensitivity_paired_rows(
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """C5 HIIT low–high demand pairs for display-only Panel A/B sensitivity curves.

    Uses the locked HIIT YAML contrast set and primary ZLPI / D240 / absolute_log10
    slice. Does not admit the sensitivity cohort into PRIMARY_META.
    """
    f = _fig()
    selected: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in paired_rows:
        if f._as_str(row.get("dataset_id")).casefold() != HIIT_DATASET_ID:
            continue
        contrast_id = f._as_str(row.get("contrast_id")).casefold()
        if contrast_id not in HIIT_ALL_CONTRASTS:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
            EXPECTED_PRIMARY_DURATION_S
        ):
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if "contrast_eligible" in row and not f._as_bool(row.get("contrast_eligible")):
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        participant_id = f._as_str(row.get("participant_id"))
        session_id = f._as_str(row.get("session_id"), "single")
        key = (participant_id, session_id, contrast_id, band)
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(row))
    return selected


def collapse_hiit_session_lag_series(
    series_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Average available pre/post lag curves within each HIIT session condition.

    Locked hierarchy (matches Figure 1 Panel E / Figure 2 Panel C): protocol session
    conditions are separate analysis units; within a session, mean available
    pre-/post-intervention low–high demand curves at each lag. ``participant_id``
    is rewritten to the
    session-unit key so existing paired bootstrap resamples session subjects.
    """
    f = _fig()
    # (session_unit, band, contrast) -> lag -> (z_low, z_effort)
    by_contrast: dict[tuple[str, str, str], dict[int, tuple[float, float]]] = {}
    meta: dict[tuple[str, str], dict[str, str]] = {}
    for row in series_rows:
        if f._as_str(row.get("dataset_id")).casefold() != HIIT_DATASET_ID:
            continue
        contrast = f._as_str(row.get("contrast_id")).casefold()
        if contrast not in HIIT_ALL_CONTRASTS:
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        unit = _hiit_session_unit_key(row)
        if not unit:
            continue
        z_low = f._as_float(row.get("z_low"))
        z_effort = f._as_float(row.get("z_effort"))
        if not (math.isfinite(z_low) and math.isfinite(z_effort)):
            continue
        lag = int(round(f._as_float(row.get("lag_s"))))
        by_contrast.setdefault((unit, band, contrast), {})[lag] = (z_low, z_effort)
        meta[(unit, band)] = {
            "dataset_id": HIIT_DATASET_ID,
            "session_id": f._as_str(row.get("session_id"), "ph|ps"),
        }

    collapsed: list[dict[str, object]] = []
    for (unit, band), info in sorted(meta.items()):
        # lag -> lists of (z_low, z_effort) across available pre/post contrasts
        buckets: dict[int, list[tuple[float, float]]] = {}
        for contrast in sorted(HIIT_ALL_CONTRASTS):
            lag_map = by_contrast.get((unit, band, contrast))
            if not lag_map:
                continue
            for lag, pair in lag_map.items():
                buckets.setdefault(lag, []).append(pair)
        for lag in sorted(buckets):
            pairs = buckets[lag]
            z_lows = [p[0] for p in pairs]
            z_efforts = [p[1] for p in pairs]
            z_low = float(np.mean(z_lows))
            z_effort = float(np.mean(z_efforts))
            collapsed.append(
                {
                    "dataset_id": info["dataset_id"],
                    "participant_id": unit,
                    "session_id": info["session_id"],
                    "contrast_id": HIIT_COMBINED_CONTRAST_ID,
                    "band": band,
                    "lag_s": lag,
                    "z_low": z_low,
                    "z_effort": z_effort,
                    "delta_z": z_effort - z_low,
                    "duration_s": EXPECTED_PRIMARY_DURATION_S,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "power_representation": f.PRIMARY_REPRESENTATION,
                    "aggregation": HIIT_SESSION_LAG_AGGREGATION,
                    "row_type": ROW_TYPE_SENSITIVITY_DISPLAY,
                }
            )
    return collapsed


def build_hiit_sensitivity_panel_a_series(
    paired_rows: Sequence[Mapping[str, object]],
    curve_index: Mapping[tuple[str, str], Mapping[int, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Reconstruct matched HIIT lag curves for Panel A/B observation-level fallback.

    Each C5 pre-/post-intervention pair is retained separately (no pre/post or
    protocol-session averaging). ``cluster_id`` marks the session condition for
    clustered bootstrap CIs. Series rows include ``z_low``, ``z_effort``, and
    ``delta_z`` for Panel A (states) and Panel B (difference).
    """
    f = _fig()
    hiit_pairs = filter_hiit_sensitivity_paired_rows(paired_rows)
    series, gaps = reconstruct_matched_pair_curves(hiit_pairs, curve_index)
    annotated: list[dict[str, object]] = []
    for row in series:
        out = dict(row)
        cluster = _hiit_session_unit_key(out)
        contrast = f._as_str(out.get("contrast_id")).casefold()
        out["cluster_id"] = cluster
        out["pair_id"] = f"{cluster}::{contrast}" if cluster and contrast else cluster
        out["aggregation"] = HIIT_MATCHED_PAIR_AGGREGATION
        out["row_type"] = ROW_TYPE_SENSITIVITY_DISPLAY
        annotated.append(out)
    return annotated, gaps


PANEL_E_STATE_LONG_FIELDS = (
    "dataset_id",
    "dataset_role",
    "participant_id",
    "session_id",
    "period",
    "contrast_id",
    "state",
    "band",
    "endpoint",
    "duration_s",
    "representation",
    "peak_identifiable",
    "peak_center_mu_s",
    "peak_fwhm_s",
    "fit_status",
    "exclusion_reason",
    "amplitude",
    "rmse",
    "amplitude_to_rmse",
    "amplitude_se",
    "r_squared",
    "row_type",
)

PANEL_E_SUMMARY_FIELDS = (
    "dataset_id",
    "dataset_role",
    "band",
    "state",
    "parameter",
    "estimate",
    "ci_lower_95",
    "ci_upper_95",
    "candidate_n",
    "identifiable_n",
    "identifiable_percent",
    "unique_session_n",
    "unique_participant_n",
    "bootstrap_draws",
    "bootstrap_seed",
    "bootstrap_cluster_field",
    "bootstrap_estimand",
    "ci_method",
    "ci_note",
)

PANEL_E_BOOTSTRAP_ESTIMAND = "mean_of_participant_means"
PANEL_E_BOOTSTRAP_CLUSTER_FIELD = "participant_id"
PANEL_E_CI_METHOD = "percentile_cluster_bootstrap"


def _panel_e_period_from_contrast(contrast_id: str) -> str:
    text = str(contrast_id or "").casefold()
    if "pre" in text:
        return "pre"
    if "post" in text:
        return "post"
    return "single"


def _panel_e_dataset_role(*, hiit_sensitivity: bool, dataset_id: str) -> str:
    if hiit_sensitivity or dataset_id.casefold() == HIIT_DATASET_ID:
        return "sensitivity"
    return "primary_meta"


def build_panel_e_peak_export(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    hiit_sensitivity: bool = False,
    peak_params_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """State-long Panel E rows: one row per matched pair × band × state.

    μ and FWHM are state-specific (shown when that state's peak is identifiable).
    Joint identifiability is not required to display one state.
    """
    f = _fig()
    peak_by_obs: dict[str, Mapping[str, object]] = {}
    for prow in peak_params_rows or ():
        obs = f._as_str(prow.get("observation_id"))
        if obs:
            peak_by_obs[obs] = prow

    export: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    for row in paired_rows:
        dataset_id = f._as_str(row.get("dataset_id"))
        participant_id = f._as_str(row.get("participant_id"))
        session_id = f._as_str(row.get("session_id"), "single")
        contrast_id = f._as_str(row.get("contrast_id"))
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        if hiit_sensitivity:
            key: tuple[str, ...] = (
                dataset_id.casefold(),
                participant_id,
                session_id.casefold(),
                contrast_id.casefold(),
                band,
            )
        else:
            key = (dataset_id.casefold(), participant_id, band)
        if key in seen:
            continue
        seen.add(key)

        period = _panel_e_period_from_contrast(contrast_id)
        dataset_role = _panel_e_dataset_role(
            hiit_sensitivity=hiit_sensitivity, dataset_id=dataset_id
        )
        endpoint = f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold()
        duration_s = f._as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S)
        representation = f._as_str(
            row.get("power_representation"), f.PRIMARY_REPRESENTATION
        ).casefold()
        row_type = (
            ROW_TYPE_SENSITIVITY_DISPLAY if hiit_sensitivity else "primary"
        )

        state_specs = (
            (
                "rest",
                f._as_bool(row.get("low_has_identifiable_peak")),
                f._as_float(row.get("low_peak_center_mu_s")),
                f._as_float(row.get("low_fwhm_s")),
                f._as_float(row.get("low_peak_height_A")),
                f._as_str(row.get("low_observation_ids")),
            ),
            (
                "task",
                f._as_bool(row.get("effort_has_identifiable_peak")),
                f._as_float(row.get("effort_peak_center_mu_s")),
                f._as_float(row.get("effort_fwhm_s")),
                f._as_float(row.get("effort_peak_height_A")),
                f._as_str(row.get("effort_observation_ids")),
            ),
        )
        for state, identifiable, mu, fwhm, amplitude, obs_ids in state_specs:
            first_obs = obs_ids.split(";")[0].strip() if obs_ids else ""
            peak = peak_by_obs.get(first_obs, {})
            rmse = f._as_float(peak.get("rmse")) if peak else float("nan")
            amp_se = f._as_float(peak.get("se_peak_height_A")) if peak else float("nan")
            amp_to_rmse = (
                float(amplitude / rmse)
                if math.isfinite(amplitude) and math.isfinite(rmse) and rmse > 0
                else float("nan")
            )
            exclusion = ""
            fit_status = "identifiable" if identifiable else "not_identifiable"
            if identifiable:
                exclusion = ""
            elif peak:
                exclusion = f._as_str(peak.get("exclusion_reason")) or (
                    "no_identifiable_positive_peak"
                )
            else:
                exclusion = "missing_peak_fit"
            export.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": dataset_role,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "period": period,
                    "contrast_id": contrast_id,
                    "state": state,
                    "band": band,
                    "endpoint": endpoint,
                    "duration_s": duration_s,
                    "representation": representation,
                    "peak_identifiable": identifiable,
                    "peak_center_mu_s": mu if identifiable else float("nan"),
                    "peak_fwhm_s": fwhm if identifiable else float("nan"),
                    "fit_status": fit_status,
                    "exclusion_reason": exclusion,
                    "amplitude": amplitude,
                    "rmse": rmse,
                    "amplitude_to_rmse": amp_to_rmse,
                    "amplitude_se": amp_se,
                    "r_squared": float("nan"),
                    "row_type": row_type,
                }
            )
    return export


def _panel_e_point_estimates_match(
    current: Sequence[Mapping[str, object]],
    prior: Sequence[Mapping[str, object]],
) -> bool:
    """True when dataset×band×state×parameter point estimates agree."""
    f = _fig()

    def _key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
        return (
            f._as_str(row.get("dataset_id")),
            f._as_str(row.get("band")).casefold(),
            f._as_str(row.get("state")).casefold(),
            f._as_str(row.get("parameter")).casefold(),
        )

    current_map = {_key(r): r for r in current}
    prior_map = {_key(r): r for r in prior}
    if set(current_map) != set(prior_map):
        return False
    for key, crow in current_map.items():
        prow = prior_map[key]
        if int(crow.get("identifiable_n") or 0) != int(prow.get("identifiable_n") or 0):
            return False
        if int(crow.get("unique_participant_n") or 0) != int(
            prow.get("unique_participant_n") or 0
        ):
            return False
        c_est = f._as_float(crow.get("estimate"))
        p_est = f._as_float(prow.get("estimate"))
        if math.isfinite(c_est) != math.isfinite(p_est):
            return False
        if math.isfinite(c_est) and abs(c_est - p_est) > 1e-12:
            return False
    return True


def _panel_e_participant_mean_bootstrap(
    values_by_participant: Mapping[str, Sequence[float]],
    *,
    n_draws: int,
    seed: int,
) -> tuple[float, float, float, str]:
    """Mean of participant means with percentile cluster bootstrap CI."""
    unit_ids = sorted(values_by_participant)
    unit_means = np.asarray(
        [float(np.mean(values_by_participant[uid])) for uid in unit_ids],
        dtype=float,
    )
    n_units = int(unit_means.size)
    if n_units == 0:
        return float("nan"), float("nan"), float("nan"), "no_identifiable_observations"
    point = float(np.mean(unit_means))
    if n_units < 2:
        return point, float("nan"), float("nan"), "n_participants_lt_2"
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    boots = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        idx = rng.integers(0, n_units, size=n_units)
        boots[i] = float(np.mean(unit_means[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi), ""


def build_panel_e_state_summaries(
    state_rows: Sequence[Mapping[str, object]],
    *,
    n_bootstrap: int | None = None,
    seed: int | None = None,
) -> list[dict[str, object]]:
    """Dataset×band×state×parameter summaries (mean of participant means)."""
    f = _fig()
    n_bootstrap = int(n_bootstrap if n_bootstrap is not None else f.FIGURE1_BOOTSTRAP_N)
    seed = int(seed if seed is not None else f.FIGURE1_BOOTSTRAP_SEED)

    groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in state_rows:
        dataset_id = f._as_str(row.get("dataset_id"))
        band = f._as_str(row.get("band")).casefold()
        state = f._as_str(row.get("state")).casefold()
        if band not in f.BAND_ORDER or state not in {"rest", "task"}:
            continue
        key = (dataset_id, band, state, f._as_str(row.get("dataset_role"), "primary_meta"))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (dataset_id, band, state, dataset_role), rows in sorted(groups.items()):
        candidate_n = len(rows)
        for parameter, value_field in (
            ("mu", "peak_center_mu_s"),
            ("fwhm", "peak_fwhm_s"),
        ):
            by_participant: dict[str, list[float]] = {}
            id_sessions: set[tuple[str, str]] = set()
            identifiable_n = 0
            for row in rows:
                if not bool(row.get("peak_identifiable")):
                    continue
                value = f._as_float(row.get(value_field))
                if not math.isfinite(value):
                    continue
                identifiable_n += 1
                pid = f._as_str(row.get("participant_id"))
                by_participant.setdefault(pid, []).append(value)
                id_sessions.add((pid, f._as_str(row.get("session_id"), "single")))
            estimate, ci_lo, ci_hi, ci_note = _panel_e_participant_mean_bootstrap(
                by_participant,
                n_draws=n_bootstrap,
                seed=seed + abs(hash((dataset_id, band, state, parameter))) % 10_000,
            )
            summaries.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": dataset_role,
                    "band": band,
                    "state": state,
                    "parameter": parameter,
                    "estimate": estimate,
                    "ci_lower_95": ci_lo,
                    "ci_upper_95": ci_hi,
                    "candidate_n": candidate_n,
                    "identifiable_n": identifiable_n,
                    "identifiable_percent": (
                        100.0 * identifiable_n / candidate_n if candidate_n else float("nan")
                    ),
                    "unique_session_n": len(id_sessions),
                    "unique_participant_n": len(by_participant),
                    "bootstrap_draws": n_bootstrap,
                    "bootstrap_seed": seed,
                    "bootstrap_cluster_field": PANEL_E_BOOTSTRAP_CLUSTER_FIELD,
                    "bootstrap_estimand": PANEL_E_BOOTSTRAP_ESTIMAND,
                    "ci_method": PANEL_E_CI_METHOD,
                    "ci_note": ci_note,
                }
            )
    return summaries



def build_curve_lag_index(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    power_representation: str | None = None,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
) -> dict[tuple[str, str], dict[int, float]]:
    """Map (observation_id, band) -> {lag_s: Fisher-z} for primary representation."""
    f = _fig()
    power_representation = power_representation or f.PRIMARY_REPRESENTATION
    index: dict[tuple[str, str], dict[int, float]] = {}
    for row in curve_rows:
        if f._as_int(row.get("duration_s"), duration_s) != duration_s:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != power_representation.casefold()
        ):
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        obs_id = f._as_str(row.get("observation_id"))
        if not obs_id:
            continue
        lag = int(round(f._as_float(row.get("lag_s"))))
        r = f._as_float(row.get("r"))
        z = fisher_z(r) if math.isfinite(r) else float("nan")
        if not math.isfinite(z):
            continue
        index.setdefault((obs_id, band), {})[lag] = z
    return index


def _mean_lag_map(
    lag_maps: Sequence[Mapping[int, float]],
) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for lag_map in lag_maps:
        for lag, z in lag_map.items():
            if math.isfinite(float(z)):
                buckets.setdefault(int(lag), []).append(float(z))
    return {lag: float(np.mean(vals)) for lag, vals in buckets.items() if vals}


def reconstruct_matched_pair_curves(
    paired_rows: Sequence[Mapping[str, object]],
    curve_index: Mapping[tuple[str, str], Mapping[int, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build matched low/effort lag series for C5 pairs.

    Returns (series_rows, wiring_gap_rows). Missing curve IDs → wiring gap;
    unpaired fallback is never used.
    """
    f = _fig()
    series: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    for row in paired_rows:
        dataset_id = f._as_str(row.get("dataset_id"))
        participant_id = f._as_str(row.get("participant_id"))
        session_id = f._as_str(row.get("session_id"), "single")
        contrast_id = f._as_str(row.get("contrast_id"))
        band = f._as_str(row.get("band")).casefold()
        low_ids = _split_observation_ids(row.get("low_observation_ids"))
        effort_ids = _split_observation_ids(row.get("effort_observation_ids"))
        if not low_ids or not effort_ids:
            gaps.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "contrast_id": contrast_id,
                    "band": band,
                    "status": "wiring_gap",
                    "reason": "missing_observation_id_fields",
                    "low_observation_ids": ";".join(low_ids),
                    "effort_observation_ids": ";".join(effort_ids),
                    "missing_low": "",
                    "missing_effort": "",
                }
            )
            continue
        low_maps = [dict(curve_index.get((oid, band), {})) for oid in low_ids]
        effort_maps = [dict(curve_index.get((oid, band), {})) for oid in effort_ids]
        missing_low = [oid for oid, m in zip(low_ids, low_maps) if not m]
        missing_effort = [oid for oid, m in zip(effort_ids, effort_maps) if not m]
        if missing_low or missing_effort:
            gaps.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "contrast_id": contrast_id,
                    "band": band,
                    "status": "wiring_gap",
                    "reason": "observation_ids_missing_from_curves",
                    "low_observation_ids": ";".join(low_ids),
                    "effort_observation_ids": ";".join(effort_ids),
                    "missing_low": ";".join(missing_low),
                    "missing_effort": ";".join(missing_effort),
                }
            )
            continue
        low_mean = _mean_lag_map(low_maps)
        effort_mean = _mean_lag_map(effort_maps)
        lags = sorted(set(low_mean) | set(effort_mean))
        for lag in lags:
            z_low = float(low_mean.get(lag, float("nan")))
            z_effort = float(effort_mean.get(lag, float("nan")))
            delta = (
                z_effort - z_low
                if math.isfinite(z_low) and math.isfinite(z_effort)
                else float("nan")
            )
            series.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "contrast_id": contrast_id,
                    "band": band,
                    "lag_s": lag,
                    "z_low": z_low,
                    "z_effort": z_effort,
                    "delta_z": delta,
                    "duration_s": EXPECTED_PRIMARY_DURATION_S,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "power_representation": f.PRIMARY_REPRESENTATION,
                }
            )
    return series, gaps


def paired_participant_bootstrap_ci(
    series_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    value_field: str,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    ci_percent: float | None = None,
) -> list[dict[str, object]]:
    """Mean ± paired participant-within-dataset bootstrap CI for a lag field.

    Resampling a participant preserves both low and effort (the matched pair),
    because ``series_rows`` already stores the pair-level quantity.
    """
    f = _fig()
    n_bootstrap = int(n_bootstrap if n_bootstrap is not None else f.FIGURE1_BOOTSTRAP_N)
    seed = int(seed if seed is not None else f.FIGURE1_BOOTSTRAP_SEED)
    ci_percent = float(
        ci_percent if ci_percent is not None else f.FIGURE1_BOOTSTRAP_CI_PERCENT
    )
    band_key = band.casefold()

    # dataset -> participant -> lag -> value
    by_ds: dict[str, dict[str, dict[int, float]]] = {}
    for row in series_rows:
        if f._as_str(row.get("band")).casefold() != band_key:
            continue
        value = f._as_float(row.get(value_field))
        if not math.isfinite(value):
            continue
        dataset = f._as_str(row.get("dataset_id"), "unknown")
        participant = f._as_str(row.get("participant_id"), "unknown")
        lag = int(round(f._as_float(row.get("lag_s"))))
        by_ds.setdefault(dataset, {}).setdefault(participant, {})[lag] = value

    if not by_ds:
        return []

    all_lags = sorted(
        {
            lag
            for subjects in by_ds.values()
            for lag_map in subjects.values()
            for lag in lag_map
        }
    )
    point: dict[int, list[float]] = {lag: [] for lag in all_lags}
    n_participants = 0
    for subjects in by_ds.values():
        n_participants += len(subjects)
        for lag_map in subjects.values():
            for lag, value in lag_map.items():
                point[lag].append(value)
    mean_by_lag = {
        lag: float(np.mean(vals)) for lag, vals in point.items() if vals
    }

    rng = np.random.default_rng(seed)
    alpha = (100.0 - ci_percent) / 2.0
    boot_means = {lag: np.empty(n_bootstrap, dtype=float) for lag in all_lags}
    dataset_subjects = {ds: list(subjects.keys()) for ds, subjects in by_ds.items()}

    for b in range(n_bootstrap):
        collected: dict[int, list[float]] = {lag: [] for lag in all_lags}
        for ds, subjects in dataset_subjects.items():
            if not subjects:
                continue
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            for participant in draw:
                for lag, value in by_ds[ds][participant].items():
                    collected[lag].append(value)
        for lag in all_lags:
            vals = collected[lag]
            boot_means[lag][b] = float(np.mean(vals)) if vals else float("nan")

    rows: list[dict[str, object]] = []
    for lag in all_lags:
        mean = mean_by_lag.get(lag, float("nan"))
        samples = boot_means[lag]
        finite = samples[np.isfinite(samples)]
        if finite.size >= 2:
            ci_low = float(np.percentile(finite, alpha))
            ci_high = float(np.percentile(finite, 100.0 - alpha))
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                "lag_s": lag,
                "band": band_key,
                "value_field": value_field,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "endpoint_name": contract_for_duration(
                    EXPECTED_PRIMARY_DURATION_S
                ).endpoint_name,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "n_participants": n_participants,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": "paired_participant_within_dataset_bootstrap",
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def hiit_matched_pair_cluster_bootstrap_ci(
    series_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    value_field: str,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    ci_percent: float | None = None,
) -> list[dict[str, object]]:
    """Observation-level mean with session-condition cluster bootstrap.

    Point estimate: equal-weight mean across matched C5 pairs (pre and post
    each contribute). Uncertainty: resample session subjects (``01_ph`` /
    ``01_ps``); when a session is drawn, include all of its matched pairs.
    Pairs are never treated as independent bootstrap units.
    """
    f = _fig()
    n_bootstrap = int(n_bootstrap if n_bootstrap is not None else f.FIGURE1_BOOTSTRAP_N)
    seed = int(seed if seed is not None else f.FIGURE1_BOOTSTRAP_SEED)
    ci_percent = float(
        ci_percent if ci_percent is not None else f.FIGURE1_BOOTSTRAP_CI_PERCENT
    )
    band_key = band.casefold()

    # pair_id -> {lag -> value}; cluster -> list[pair_id]
    pair_lags: dict[str, dict[int, float]] = {}
    by_cluster: dict[str, list[str]] = {}
    for row in series_rows:
        if f._as_str(row.get("band")).casefold() != band_key:
            continue
        value = f._as_float(row.get(value_field))
        if not math.isfinite(value):
            continue
        cluster = f._as_str(row.get("cluster_id")) or _hiit_session_unit_key(row)
        if not cluster:
            continue
        pair_id = f._as_str(row.get("pair_id"))
        if not pair_id:
            contrast = f._as_str(row.get("contrast_id")).casefold()
            pair_id = f"{cluster}::{contrast}" if contrast else cluster
        lag = int(round(f._as_float(row.get("lag_s"))))
        pair_lags.setdefault(pair_id, {})[lag] = value
        if pair_id not in by_cluster.setdefault(cluster, []):
            by_cluster[cluster].append(pair_id)

    if not pair_lags:
        return []

    all_lags = sorted({lag for lag_map in pair_lags.values() for lag in lag_map})
    point: dict[int, list[float]] = {lag: [] for lag in all_lags}
    for lag_map in pair_lags.values():
        for lag, value in lag_map.items():
            point[lag].append(value)
    mean_by_lag = {lag: float(np.mean(vals)) for lag, vals in point.items() if vals}

    n_matched_pairs = len(pair_lags)
    n_session_clusters = len(by_cluster)
    clusters = list(by_cluster.keys())
    rng = np.random.default_rng(seed)
    alpha = (100.0 - ci_percent) / 2.0
    boot_means = {lag: np.empty(n_bootstrap, dtype=float) for lag in all_lags}

    for b in range(n_bootstrap):
        collected: dict[int, list[float]] = {lag: [] for lag in all_lags}
        if clusters:
            draw = rng.choice(clusters, size=len(clusters), replace=True)
            for cluster in draw:
                for pair_id in by_cluster[cluster]:
                    for lag, value in pair_lags[pair_id].items():
                        collected[lag].append(value)
        for lag in all_lags:
            vals = collected[lag]
            boot_means[lag][b] = float(np.mean(vals)) if vals else float("nan")

    rows: list[dict[str, object]] = []
    for lag in all_lags:
        mean = mean_by_lag.get(lag, float("nan"))
        samples = boot_means[lag]
        finite = samples[np.isfinite(samples)]
        if finite.size >= 2:
            ci_low = float(np.percentile(finite, alpha))
            ci_high = float(np.percentile(finite, 100.0 - alpha))
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                "lag_s": lag,
                "band": band_key,
                "value_field": value_field,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "endpoint_name": contract_for_duration(
                    EXPECTED_PRIMARY_DURATION_S
                ).endpoint_name,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "n_matched_pairs": n_matched_pairs,
                "n_session_clusters": n_session_clusters,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": HIIT_CLUSTER_BOOTSTRAP_CI_METHOD,
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _cardiac_modality(dataset_id: str, protocol_rows: Sequence[Mapping[str, object]]) -> str:
    f = _fig()
    key = f._as_str(dataset_id).casefold()
    for row in protocol_rows:
        if f._as_str(row.get("dataset_id")).casefold() != key:
            continue
        modality = f._as_str(row.get("cardiac_modality"))
        if modality:
            return modality.split(";")[0].strip() or modality
    spec = PROTOCOL_SPECS.get(key) or PROTOCOL_SPECS.get(f._as_str(dataset_id))
    if spec is not None:
        modality = f._as_str(spec.cardiac_modality)
        return modality.split(";")[0].strip() or modality
    return ""


def _short_mixedlm_term_label(term: str) -> str:
    """Display-only shortening of patsy MixedLM/OLS term strings."""
    f = _fig()
    text = str(term).strip()
    if not text:
        return text
    if text.casefold() == "intercept":
        return "Intercept"
    if text.casefold() in {"mean_hr", "mean_hr_bpm"}:
        return "Mean HR"

    band_tokens = {
        "theta": "θ",
        "alpha": "α",
        "beta": "β",
        "low_gamma": "low-γ",
        "gamma": "γ",
    }
    state_tokens = {
        "low_demand": "low",
        "cognitive_effort": "effort",
        "effort": "effort",
    }

    def _factor_level(fragment: str) -> tuple[str, str] | None:
        # C(factor)[T.level] or C(factor)[level]
        frag = fragment.strip()
        if not frag.startswith("C(") or "]" not in frag:
            return None
        try:
            inside = frag[frag.index("(") + 1 : frag.index(")")]
            bracket = frag[frag.index("[") + 1 : frag.index("]")]
        except ValueError:
            return None
        level = bracket[2:] if bracket.startswith("T.") else bracket
        return inside.casefold(), level

    if ":" in text:
        left, right = text.split(":", 1)
        parts = [_factor_level(left), _factor_level(right)]
        if all(parts):
            (f0, l0), (f1, l1) = parts  # type: ignore[misc]
            levels: list[str] = []
            for factor, level in ((f0, l0), (f1, l1)):
                key = level.casefold()
                if "band" in factor:
                    levels.append(band_tokens.get(key, level))
                elif "state" in factor:
                    levels.append(state_tokens.get(key, level))
                else:
                    levels.append(level)
            return f"State×band: {'×'.join(levels)}"

    parsed = _factor_level(text)
    if parsed is not None:
        factor, level = parsed
        key = level.casefold()
        if "state" in factor:
            return f"State: {state_tokens.get(key, level)}"
        if "band" in factor:
            return f"Band: {band_tokens.get(key, f._band_display(key) if key in f.BAND_ORDER else level)}"
        if "dataset" in factor:
            return f"Dataset: {f._dataset_display(level)}"
        if "modality" in factor:
            return f"Modality: {level}"
        return f"{factor}: {level}"
    return text


def _endpoint_label() -> str:
    f = _fig()
    return f._endpoint_display(ENDPOINT_ZLPI)


def _filter_primary_panel_d_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    f = _fig()
    selected: list[dict[str, object]] = []
    for row in rows:
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if str(row.get("is_primary_analysis", "true")).lower() not in {
            "true",
            "1",
            "yes",
            "",
        }:
            continue
        selected.append(dict(row))
    return selected


def _load_panel_d_tables(
    inputs: Mapping[str, Path | None],
    *,
    panel_a_hiit_sensitivity: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Load or compute Panel D marginal estimates, contrasts, and coefficient rows."""
    f = _fig()
    mixed = f.read_csv_rows(inputs.get("mixed_model"))
    marginal = _filter_primary_panel_d_rows(
        f.read_csv_rows(inputs.get("mixed_model_marginal"))
    )
    contrasts = _filter_primary_panel_d_rows(
        f.read_csv_rows(inputs.get("mixed_model_contrasts"))
    )
    coef_rows: list[dict[str, object]] = []
    for row in mixed:
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if str(row.get("is_primary_analysis", "true")).lower() not in {
            "true",
            "1",
            "yes",
            "",
        }:
            continue
        term = f._as_str(row.get("term"))
        coef = f._as_float(row.get("coef"))
        if term and math.isfinite(coef):
            coef_rows.append(dict(row))

    if not marginal:
        subject_rows = f.read_csv_rows(inputs.get("subject_level"))
        if subject_rows:
            coef_rows, qc = fit_mixed_model(subject_rows, endpoint_name=ENDPOINT_ZLPI)
            marginal = list(qc.get("panel_d_marginal_rows") or [])
            contrasts = list(qc.get("panel_d_contrast_rows") or [])
            if panel_a_hiit_sensitivity and marginal:
                for row in marginal:
                    row["dataset_scope"] = "HIIT_sensitivity_display"
                for row in contrasts:
                    row["dataset_scope"] = "HIIT_sensitivity_display"

    return marginal, contrasts, coef_rows


def _panel_d_band_index(band: str) -> int:
    key = str(band).strip().casefold()
    try:
        return PANEL_D_DISPLAY_BANDS.index(key)
    except ValueError:
        return len(PANEL_D_DISPLAY_BANDS)


def _panel_d_alpha_contrast_other_band(contrast_name: str) -> str:
    return (
        str(contrast_name)
        .replace("alpha_minus_", "")
        .replace("_state_effect", "")
        .strip()
        .casefold()
    )


def _panel_d_alpha_contrast_sort_key(row: Mapping[str, object]) -> int:
    other = _panel_d_alpha_contrast_other_band(str(row.get("contrast_name", "")))
    try:
        return PANEL_D_ALPHA_CONTRAST_ORDER.index(other)
    except ValueError:
        return len(PANEL_D_ALPHA_CONTRAST_ORDER)


def _render_panel_d_estimation_plot(
    ax_main: plt.Axes,
    ax_contrast: plt.Axes,
    marginal_rows: Sequence[Mapping[str, object]],
    contrast_rows: Sequence[Mapping[str, object]],
    *,
    panel_d_hiit_sensitivity: bool,
) -> tuple[str, list[plt.Line2D]]:
    """Draw Panel D estimates/contrasts. Title/legend are placed by the caller."""
    f = _fig()
    title = (
        f.sensitivity_display_title(
            "band × state interaction", dataset_id="hiit"
        )
        if panel_d_hiit_sensitivity
        else "Band × state interaction"
    )
    legend_handles = [
        plt.Line2D([0], [0], color=f.PALETTE["blue"], marker="o", linestyle="", label="Low demand"),
        plt.Line2D(
            [0], [0], color=f.PALETTE["vermillion"], marker="o", linestyle="", label="High demand"
        ),
    ]

    if not marginal_rows:
        f._mark_empty_panel(
            ax_main,
            "No model-estimated Fisher-z ZLPI for primary D240 ZLPI.",
            xlabel="Frequency band",
            ylabel=f.FIGURE2_PANEL_D_OUTCOME_LABEL,
        )
        ax_contrast.set_visible(False)
        return title, legend_handles

    low_color = f.PALETTE["blue"]
    high_color = f.PALETTE["vermillion"]
    alpha_fill = (*plt.matplotlib.colors.to_rgb(f.PALETTE["orange"]), 0.12)
    x_positions = np.arange(len(PANEL_D_DISPLAY_BANDS), dtype=float)
    offset = 0.16

    for row in sorted(
        marginal_rows,
        key=lambda r: (
            _panel_d_band_index(str(r.get("band", ""))),
            0 if "low" in str(r.get("state", "")).casefold() else 1,
        ),
    ):
        band = f._as_str(row.get("band")).casefold()
        state = f._as_str(row.get("state")).casefold()
        if band not in PANEL_D_DISPLAY_BANDS:
            continue
        xi = float(PANEL_D_DISPLAY_BANDS.index(band))
        is_low = "low" in state
        xpos = xi - offset if is_low else xi + offset
        est = f._as_float(row.get("estimated_zlpi"))
        lo = f._as_float(row.get("ci_low"))
        hi = f._as_float(row.get("ci_high"))
        color = low_color if is_low else high_color
        lw = f.LINE_WIDTH + (0.6 if band == "alpha" else 0.0)
        ms = f.MARKER_SIZE + (1 if band == "alpha" else 0)
        alpha_m = 1.0 if band == "alpha" else 0.88
        if math.isfinite(lo) and math.isfinite(hi):
            yerr = [[est - lo], [hi - est]]
        else:
            yerr = None
        ax_main.errorbar(
            xpos,
            est,
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=ms,
            capsize=3,
            elinewidth=lw,
            alpha=alpha_m,
            zorder=4 if band == "alpha" else 3,
        )

    alpha_idx = PANEL_D_DISPLAY_BANDS.index("alpha")
    ax_main.axvspan(
        alpha_idx - 0.45,
        alpha_idx + 0.45,
        color=alpha_fill,
        zorder=1,
    )
    ax_main.set_xticks(x_positions)
    ax_main.set_xticklabels(
        [b.capitalize() if b != "gamma" else "Gamma" for b in PANEL_D_DISPLAY_BANDS],
        fontsize=f.FS_TICK - 1,
    )
    # Omit bottom xlabel when the contrast strip is present — it collides with
    # the contrast subtitle; band names on the ticks are already sufficient.
    ax_main.set_ylabel(f.FIGURE2_PANEL_D_OUTCOME_LABEL, fontsize=f.FS_AXIS - 2)
    f._style_axes(ax_main)

    alpha_contrasts = [
        row
        for row in contrast_rows
        if f._as_str(row.get("contrast_type")) == "alpha_vs_other_state_effect"
    ]
    alpha_contrasts = sorted(alpha_contrasts, key=_panel_d_alpha_contrast_sort_key)
    if not alpha_contrasts:
        ax_main.set_xlabel("Frequency band", fontsize=f.FS_AXIS - 2)
        ax_contrast.set_visible(False)
        return title, legend_handles

    n_contrast = len(alpha_contrasts)
    y_positions = [
        (n_contrast - 1 - i) * PANEL_D_CONTRAST_ROW_SPACING
        for i in range(n_contrast)
    ]
    ci_highs: list[float] = []
    p_value_labels: list[tuple[float, float, float]] = []
    for ypos, row in zip(y_positions, alpha_contrasts, strict=True):
        est = f._as_float(row.get("estimate"))
        lo = f._as_float(row.get("ci_low"))
        hi = f._as_float(row.get("ci_high"))
        p_value = f._as_float(row.get("p_value"))
        xerr = None
        if math.isfinite(lo) and math.isfinite(hi):
            xerr = [[est - lo], [hi - est]]
            ci_highs.append(hi)
        ax_contrast.errorbar(
            est,
            ypos,
            xerr=xerr,
            fmt="D",
            color=f.PALETTE["orange"],
            markersize=f.MARKER_SIZE - 2,
            capsize=3,
            elinewidth=f.LINE_WIDTH,
            zorder=3,
        )
        if math.isfinite(p_value):
            p_value_labels.append((ypos, hi if math.isfinite(hi) else est, p_value))
    f._ref_vline(ax_contrast, 0.0)
    ax_contrast.set_yticks(y_positions)
    ax_contrast.set_yticklabels(
        [
            f._as_str(r.get("contrast_name"))
            .replace("alpha_minus_", "α − ")
            .replace("_state_effect", "")
            .replace("_", " ")
            for r in alpha_contrasts
        ],
        fontsize=f.FS_TICK - 4,
    )
    ax_contrast.set_xlabel("Contrast (95% CI)", fontsize=f.FS_AXIS - 3, labelpad=6)
    # Keep title pad small; main-panel "Frequency band" xlabel is omitted above
    # so this caption no longer collides with the upper axis label.
    ax_contrast.set_title(
        "Alpha vs other-band state-effect contrasts\n"
        "(negative = stronger alpha attenuation)",
        fontsize=f.FS_TICK - 3,
        loc="left",
        pad=8,
    )
    f._style_axes(ax_contrast)
    y_pad = 0.70 * PANEL_D_CONTRAST_ROW_SPACING
    ax_contrast.set_ylim(
        -y_pad,
        (n_contrast - 1) * PANEL_D_CONTRAST_ROW_SPACING + y_pad,
    )
    xmin, xmax = ax_contrast.get_xlim()
    data_xmax = max(ci_highs) if ci_highs else xmax
    x_span = max(xmax - xmin, 1e-6)
    ax_contrast.set_xlim(xmin, max(xmax, data_xmax + 0.32 * x_span))
    xmin, xmax = ax_contrast.get_xlim()
    x_span = max(xmax - xmin, 1e-6)
    for ypos, anchor, p_value in p_value_labels:
        text_x = anchor + 0.12 * x_span
        ax_contrast.text(
            text_x,
            ypos,
            f"p={p_value:.3g}",
            va="center",
            ha="left",
            fontsize=f.FS_TICK - 5,
            color=f.PALETTE["dark_gray"],
            clip_on=False,
        )
    return title, legend_handles


def render_figure2(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
):
    """Six-panel manuscript Figure 2 (presentation layout only)."""
    f = _fig()
    f._configure_publication_style()
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    # Drop legacy Figure 2 exports from the prior 2×2 layout.
    for stale in (
        "figure2_panel_a_paired_deltas.csv",
        "figure2_panel_a_unit_summaries.csv",
        "figure2_panel_a_inference.csv",
        "figure2_panel_b_meta_forest.csv",
        "figure2_panel_c_d180_sensitivity.csv",
        "figure2_panel_d_mu_equivalence.csv",
    ):
        path = source_dir / stale
        if path.is_file():
            path.unlink()

    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    curves = f.read_csv_rows(inputs.get("curves_d240"))
    paired = f.read_csv_rows(inputs.get("paired_contrasts"))
    effects = f.read_csv_rows(inputs.get("dataset_effects"))
    meta = f.read_csv_rows(inputs.get("meta_analysis"))
    mixed = f.read_csv_rows(inputs.get("mixed_model"))
    equivalence = f.read_csv_rows(inputs.get("peak_equivalence"))
    protocol = f.read_csv_rows(inputs.get("protocol_audit"))

    meta_pairs = filter_primary_meta_paired_rows(paired)
    curve_index = build_curve_lag_index(curves)
    # Panels C–F keep PRIMARY_META-only inputs. Panels A/B may fall back to HIIT.
    series_rows, wiring_gaps = reconstruct_matched_pair_curves(meta_pairs, curve_index)
    panel_a_series = series_rows
    panel_a_gaps = wiring_gaps
    panel_a_hiit_sensitivity = False
    if not meta_pairs:
        hiit_series, hiit_gaps = build_hiit_sensitivity_panel_a_series(
            paired, curve_index
        )
        if hiit_series:
            panel_a_series = hiit_series
            panel_a_gaps = hiit_gaps
            panel_a_hiit_sensitivity = True
        else:
            panel_a_gaps = hiit_gaps
    # Panel B uses the same matched series as Panel A (PRIMARY_META or HIIT).
    panel_b_series = panel_a_series
    panel_b_gaps = panel_a_gaps
    panel_b_hiit_sensitivity = panel_a_hiit_sensitivity

    fig = plt.figure(figsize=(17.0, 17.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, hspace=0.68, wspace=0.38)

    # ----- Panel A: matched low vs effort lag curves -----
    gs_a_wrap = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 0], height_ratios=[0.18, 1.0], hspace=0.06
    )
    ax_a_title = fig.add_subplot(gs_a_wrap[0, 0])
    ax_a_title.axis("off")
    gs_a = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_a_wrap[1, 0], hspace=0.40, wspace=0.28
    )
    panel_a_export: list[dict[str, object]] = []
    has_a = False
    panel_a_expected_na = False
    panel_a_na_detail = ""
    ax_a0: plt.Axes | None = None
    panel_a_title = (
        f.sensitivity_display_title(
            "matched low vs high-demand curves", dataset_id="hiit"
        )
        if panel_a_hiit_sensitivity
        else "Matched low vs effort curves"
    )
    ax_a_title.text(
        0.0,
        0.40,
        panel_a_title,
        transform=ax_a_title.transAxes,
        ha="left",
        va="center",
        fontsize=f.FS_PANEL_TITLE - 1,
        fontweight="bold",
        color="black",
    )
    if panel_a_series:
        for bi, band in enumerate(f.BAND_ORDER):
            ax = fig.add_subplot(gs_a[bi // 2, bi % 2])
            if bi == 0:
                ax_a0 = ax
            if panel_a_hiit_sensitivity:
                low_ci = hiit_matched_pair_cluster_bootstrap_ci(
                    panel_a_series, band=band, value_field="z_low"
                )
                effort_ci = hiit_matched_pair_cluster_bootstrap_ci(
                    panel_a_series, band=band, value_field="z_effort"
                )
            else:
                low_ci = paired_participant_bootstrap_ci(
                    panel_a_series, band=band, value_field="z_low"
                )
                effort_ci = paired_participant_bootstrap_ci(
                    panel_a_series, band=band, value_field="z_effort"
                )
            for row in low_ci:
                export = dict(row)
                export["state"] = "low_demand"
                panel_a_export.append(export)
            for row in effort_ci:
                export = dict(row)
                export["state"] = "cognitive_effort"
                panel_a_export.append(export)
            if low_ci:
                lags = np.asarray([r["lag_s"] for r in low_ci], dtype=float)
                # Plot the full 1 s lag grid so σ=1 s display smooth is visible
                # (FIGURE1 2 s thinning hides most of a 1-sample smooth).
                mask = f.display_lag_mask(lags, 1)
                mean = np.asarray([r["mean"] for r in low_ci], dtype=float)
                lo = np.asarray([r["ci_low"] for r in low_ci], dtype=float)
                hi = np.asarray([r["ci_high"] for r in low_ci], dtype=float)
                # Display-only smooth after CI computation; exports stay unsmoothed.
                mean_d = f.gaussian_smooth_display_series(
                    mean, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                lo_d = f.gaussian_smooth_display_series(
                    lo, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                hi_d = f.gaussian_smooth_display_series(
                    hi, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                ax.fill_between(
                    lags[mask],
                    lo_d[mask],
                    hi_d[mask],
                    color=f.PALETTE["green"],
                    alpha=f.FIGURE1_PANEL_B_CI_ALPHA,
                    linewidth=0,
                )
                ax.plot(
                    lags[mask],
                    mean_d[mask],
                    color=f.PALETTE["green"],
                    lw=f.LINE_WIDTH,
                    label="Low-demand",
                )
            if effort_ci:
                lags = np.asarray([r["lag_s"] for r in effort_ci], dtype=float)
                mask = f.display_lag_mask(lags, 1)
                mean = np.asarray([r["mean"] for r in effort_ci], dtype=float)
                lo = np.asarray([r["ci_low"] for r in effort_ci], dtype=float)
                hi = np.asarray([r["ci_high"] for r in effort_ci], dtype=float)
                mean_d = f.gaussian_smooth_display_series(
                    mean, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                lo_d = f.gaussian_smooth_display_series(
                    lo, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                hi_d = f.gaussian_smooth_display_series(
                    hi, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                ax.fill_between(
                    lags[mask],
                    lo_d[mask],
                    hi_d[mask],
                    color=f.PALETTE["vermillion"],
                    alpha=f.FIGURE1_PANEL_B_CI_ALPHA,
                    linewidth=0,
                )
                ax.plot(
                    lags[mask],
                    mean_d[mask],
                    color=f.PALETTE["vermillion"],
                    lw=f.LINE_WIDTH,
                    label="Effort",
                )
            f._ref_vline(ax, 0.0)
            ax.set_xlim(-60, 60)
            ax.set_title(f._band_display(band), fontsize=f.FS_TICK - 1, pad=3)
            if bi >= 2:
                ax.set_xlabel(f.LAG_XLABEL, fontsize=f.FS_AXIS - 4)
            if bi % 2 == 0:
                ax.set_ylabel(f.Z_YLABEL, fontsize=f.FS_AXIS - 4)
            f._style_axes(ax)
            if bi == 0:
                ax.legend(
                    fontsize=f.FS_LEGEND - 3,
                    loc="upper right",
                    frameon=True,
                    fancybox=False,
                    edgecolor="#CCCCCC",
                    framealpha=0.95,
                    borderpad=0.35,
                    handletextpad=0.35,
                    labelspacing=0.25,
                )
        assert ax_a0 is not None
        f._add_panel_label(ax_a0, "A")
        # Exterior title placed after subplots_adjust (see below).
        has_a = True
    else:
        ax_a = fig.add_subplot(gs[0, 0])
        if panel_a_gaps:
            msg = f.FIGURE2_WIRING_GAP_NOTE
        else:
            panel_a_na_detail = (
                "No PRIMARY_META C5 pairs with reconstructable C2 curves."
            )
            panel_a_expected_na = True
            msg = f.primary_meta_expected_na_message(panel_a_na_detail)
        f._mark_empty_panel(ax_a, msg, xlabel=f.LAG_XLABEL, ylabel=f.Z_YLABEL)
        f._set_panel_title(ax_a, panel_a_title)
        f._add_panel_label(ax_a, "A")
        has_a = False
        ax_a0 = None

    gap_csv = source_dir / "figure2_panel_a_wiring_gaps.csv"
    f.write_source_csv(
        gap_csv,
        panel_a_gaps,
        (
            "dataset_id",
            "participant_id",
            "session_id",
            "contrast_id",
            "band",
            "status",
            "reason",
            "low_observation_ids",
            "effort_observation_ids",
            "missing_low",
            "missing_effort",
        ),
    )
    source_paths.append(gap_csv)
    panel_a_csv = source_dir / "figure2_panel_a_matched_lag_curves.csv"
    panel_a_fields = (
        (
            "lag_s",
            "band",
            "state",
            "value_field",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n_matched_pairs",
            "n_session_clusters",
            "mean",
            "ci_low",
            "ci_high",
            "ci_method",
            "n_bootstrap",
        )
        if panel_a_hiit_sensitivity
        else (
            "lag_s",
            "band",
            "state",
            "value_field",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n_participants",
            "mean",
            "ci_low",
            "ci_high",
            "ci_method",
            "n_bootstrap",
        )
    )
    f.write_source_csv(
        panel_a_csv,
        panel_a_export,
        panel_a_fields,
    )
    source_paths.append(panel_a_csv)
    if panel_a_hiit_sensitivity:
        panel_a_keys = [
            "cohort=HIIT_sensitivity_matched_observation_pairs",
            f"ci={HIIT_CLUSTER_BOOTSTRAP_CI_METHOD}",
            "unpaired_fallback=false",
            f"aggregation={HIIT_MATCHED_PAIR_AGGREGATION}",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
            "enters_primary_meta=false",
        ]
        panel_a_notes = f.FIGURE2_PANEL_A_SENSITIVITY_NOTE
        panel_a_panel_title = f.sensitivity_display_title(
            "matched low vs high-demand lag curves", dataset_id="hiit"
        )
    elif panel_a_expected_na:
        panel_a_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "ci=paired_participant_within_dataset_bootstrap",
            "unpaired_fallback=false",
            f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
        ]
        panel_a_notes = f.annotate_expected_not_applicable(
            "Exact C5 paired-intersection participants linked via "
            "low/effort_observation_ids → C2 observation_id.",
            detail=panel_a_na_detail,
        )
        panel_a_panel_title = "Matched low vs effort lag curves (C5 pairs)"
    else:
        panel_a_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "ci=paired_participant_within_dataset_bootstrap",
            "unpaired_fallback=false",
        ]
        panel_a_notes = (
            "Exact C5 paired-intersection participants linked via "
            "low/effort_observation_ids → C2 observation_id. "
            + (f.FIGURE2_WIRING_GAP_NOTE if panel_a_gaps and not has_a else "")
        )
        panel_a_panel_title = "Matched low vs effort lag curves (C5 pairs)"
    panel_a_keys = [
        *panel_a_keys,
        f"display_smooth=gaussian_sigma_{f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g}s",
        "source_data=unsmoothed",
    ]
    panel_a_notes = (
        f"{panel_a_notes.rstrip()} {f.LAG_CURVE_DISPLAY_SMOOTH_NOTE}"
    ).strip()
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="matched_low_effort_curves",
            title=panel_a_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("paired_contrasts") or ""),
                str(inputs.get("curves_d240") or ""),
            ],
            source_data_csv=str(panel_a_csv),
            analysis_keys=panel_a_keys,
            notes=panel_a_notes,
        )
    )

    # ----- Panel B: matched lag-difference curves -----
    gs_b_wrap = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1], height_ratios=[0.18, 1.0], hspace=0.06
    )
    ax_b_title = fig.add_subplot(gs_b_wrap[0, 0])
    ax_b_title.axis("off")
    gs_b = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_b_wrap[1, 0], hspace=0.40, wspace=0.28
    )
    panel_b_export: list[dict[str, object]] = []
    ax_b0: plt.Axes | None = None
    has_b_nested = False
    panel_b_title = (
        f.sensitivity_display_title(
            "matched lag-difference curves", dataset_id="hiit"
        )
        if panel_b_hiit_sensitivity
        else "Matched lag-difference curves"
    )
    ax_b_title.text(
        0.0,
        0.40,
        panel_b_title,
        transform=ax_b_title.transAxes,
        ha="left",
        va="center",
        fontsize=f.FS_PANEL_TITLE - 1,
        fontweight="bold",
        color="black",
    )
    if panel_b_series:
        for bi, band in enumerate(f.BAND_ORDER):
            ax = fig.add_subplot(gs_b[bi // 2, bi % 2])
            if bi == 0:
                ax_b0 = ax
            if panel_b_hiit_sensitivity:
                delta_ci = hiit_matched_pair_cluster_bootstrap_ci(
                    panel_b_series, band=band, value_field="delta_z"
                )
            else:
                delta_ci = paired_participant_bootstrap_ci(
                    panel_b_series, band=band, value_field="delta_z"
                )
            panel_b_export.extend(delta_ci)
            if delta_ci:
                lags = np.asarray([r["lag_s"] for r in delta_ci], dtype=float)
                # Full 1 s lag grid so σ=1 s display smooth is visible.
                mask = f.display_lag_mask(lags, 1)
                mean = np.asarray([r["mean"] for r in delta_ci], dtype=float)
                lo = np.asarray([r["ci_low"] for r in delta_ci], dtype=float)
                hi = np.asarray([r["ci_high"] for r in delta_ci], dtype=float)
                # Display-only smooth after CI computation; exports stay unsmoothed.
                mean_d = f.gaussian_smooth_display_series(
                    mean, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                lo_d = f.gaussian_smooth_display_series(
                    lo, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                hi_d = f.gaussian_smooth_display_series(
                    hi, sigma_s=f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
                )
                color = f._band_color(band)
                ax.fill_between(
                    lags[mask],
                    lo_d[mask],
                    hi_d[mask],
                    color=color,
                    alpha=f.FIGURE1_PANEL_B_CI_ALPHA,
                    linewidth=0,
                )
                ax.plot(lags[mask], mean_d[mask], color=color, lw=f.LINE_WIDTH)
            f._ref_vline(ax, 0.0)
            f._ref_hline(ax, 0.0)
            ax.set_xlim(-60, 60)
            ax.set_title(f._band_display(band), fontsize=f.FS_TICK - 1, pad=3)
            if bi >= 2:
                ax.set_xlabel(f.LAG_XLABEL, fontsize=f.FS_AXIS - 4)
            if bi % 2 == 0:
                ax.set_ylabel(f"Δ {f.Z_YLABEL}", fontsize=f.FS_AXIS - 4)
            f._style_axes(ax)
        assert ax_b0 is not None
        f._add_panel_label(ax_b0, "B")
        has_b_nested = True
        panel_b_expected_na = False
        panel_b_na_detail = ""
    else:
        ax_b = fig.add_subplot(gs[0, 1])
        if panel_b_gaps:
            msg = f.FIGURE2_WIRING_GAP_NOTE
            panel_b_expected_na = False
            panel_b_na_detail = ""
        else:
            panel_b_na_detail = (
                "No PRIMARY_META C5 pairs with reconstructable C2 curves."
            )
            panel_b_expected_na = True
            msg = f.primary_meta_expected_na_message(panel_b_na_detail)
        f._mark_empty_panel(ax_b, msg, xlabel=f.LAG_XLABEL, ylabel=f"Δ {f.Z_YLABEL}")
        f._set_panel_title(ax_b, panel_b_title)
        f._add_panel_label(ax_b, "B")
        has_b_nested = False
        ax_b0 = None

    panel_b_csv = source_dir / "figure2_panel_b_lag_difference_curves.csv"
    panel_b_fields = (
        (
            "lag_s",
            "band",
            "value_field",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n_matched_pairs",
            "n_session_clusters",
            "mean",
            "ci_low",
            "ci_high",
            "ci_method",
            "n_bootstrap",
        )
        if panel_b_hiit_sensitivity
        else (
            "lag_s",
            "band",
            "value_field",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n_participants",
            "mean",
            "ci_low",
            "ci_high",
            "ci_method",
            "n_bootstrap",
        )
    )
    f.write_source_csv(
        panel_b_csv,
        panel_b_export,
        panel_b_fields,
    )
    source_paths.append(panel_b_csv)
    if panel_b_hiit_sensitivity:
        panel_b_keys = [
            "cohort=HIIT_sensitivity_matched_observation_pairs",
            f"ci={HIIT_CLUSTER_BOOTSTRAP_CI_METHOD}",
            "cluster_permutation=false",
            f"aggregation={HIIT_MATCHED_PAIR_AGGREGATION}",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
            "enters_primary_meta=false",
        ]
        panel_b_notes = f.FIGURE2_PANEL_B_SENSITIVITY_NOTE
        panel_b_panel_title = f.sensitivity_display_title(
            "matched lag-difference curves", dataset_id="hiit"
        )
    elif panel_b_expected_na:
        panel_b_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "ci=paired_participant_within_dataset_bootstrap",
            "cluster_permutation=false",
            f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
        ]
        panel_b_notes = f.annotate_expected_not_applicable(
            "Display-only matched Δz(τ). Formal lag-0 attenuation uses "
            "paired ΔZLPI / PRIMARY_META / C4 surrogates (Panels C/elsewhere). "
            "No cluster-permutation testing.",
            detail=panel_b_na_detail,
        )
        panel_b_panel_title = "Matched task−rest lag-difference curves"
    else:
        panel_b_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "ci=paired_participant_within_dataset_bootstrap",
            "cluster_permutation=false",
        ]
        panel_b_notes = (
            "Display-only matched Δz(τ). Formal lag-0 attenuation uses "
            "paired ΔZLPI / PRIMARY_META / C4 surrogates (Panels C/elsewhere). "
            "No cluster-permutation testing."
        )
        panel_b_panel_title = "Matched task−rest lag-difference curves"
    panel_b_keys = [
        *panel_b_keys,
        f"display_smooth=gaussian_sigma_{f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g}s",
        "source_data=unsmoothed",
    ]
    panel_b_notes = (
        f"{panel_b_notes.rstrip()} {f.LAG_CURVE_DISPLAY_SMOOTH_NOTE}"
    ).strip()
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="matched_lag_difference_curves",
            title=panel_b_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("paired_contrasts") or ""),
                str(inputs.get("curves_d240") or ""),
            ],
            source_data_csv=str(panel_b_csv),
            analysis_keys=panel_b_keys,
            notes=panel_b_notes,
        )
    )

    # ----- Panel C: alpha PRIMARY_META absolute ΔZLPI forest -----
    ax_c = fig.add_subplot(gs[1, 0])
    studies, pooled = primary_meta_alpha_forest_rows(
        effects,
        meta,
        protocol,
        cardiac_modality_fn=_cardiac_modality,
    )
    sensitivity_studies = hiit_session_sensitivity_forest_rows(paired)
    forest_export = build_alpha_forest_export(
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
    )
    panel_c_hiit_only = bool(sensitivity_studies) and not studies
    panel_c_title = (
        f.sensitivity_display_title("Alpha ΔZLPI", dataset_id="hiit")
        if panel_c_hiit_only
        else "Alpha PRIMARY_META ΔZLPI"
    )
    drawn = draw_alpha_meta_forest(
        ax_c,
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
        dataset_display_fn=f._dataset_display,
        band_color_fn=f._band_color,
        ref_vline_fn=f._ref_vline,
        style_axes_fn=f._style_axes,
        set_panel_title_fn=f._set_panel_title,
        panel_title=panel_c_title,
        xlabel=f"Δ {_endpoint_label()} ({f.ZLPI_METRIC})",
        marker_size=f.MARKER_SIZE,
        line_width=f.LINE_WIDTH,
        tick_fontsize=f.FS_TICK,
        axis_fontsize=f.FS_AXIS,
        palette=f.PALETTE,
    )
    if drawn:
        panel_c_expected_na = False
        panel_c_na_detail = ""
    else:
        panel_c_expected_na = True
        panel_c_na_detail = "No PRIMARY_META alpha study effects in this run."
        f._mark_empty_panel(
            ax_c,
            f.primary_meta_expected_na_message(panel_c_na_detail),
            xlabel=f"Δ ZLPI ({f.CI_95_LABEL})",
            ylabel="Dataset",
        )
        f._set_panel_title(ax_c, panel_c_title)
    f._add_panel_label(ax_c, "C")
    panel_c_csv = source_dir / "figure2_panel_c_alpha_meta_forest.csv"
    f.write_source_csv(
        panel_c_csv,
        forest_export,
        FOREST_EXPORT_FIELDS,
    )
    source_paths.append(panel_c_csv)
    if panel_c_hiit_only:
        panel_c_keys = [
            "band=alpha",
            "endpoint=absolute_delta_zlpi",
            "percent_attenuation=false",
            "cohort=HIIT_sensitivity_display",
            "hiit_sensitivity_display=combined_ph_ps_within_participant_mean",
            "enters_primary_meta=false",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
        ]
        panel_c_notes = (
            "Sensitivity display (display-only): one combined alpha ΔZLPI estimate "
            "(protocol session conditions counted separately). Excluded from "
            "PRIMARY_META RE pooling; not a primary confirmatory claim."
        )
        panel_c_panel_title = f.sensitivity_display_title(
            "Alpha absolute ΔZLPI", dataset_id="hiit"
        )
    elif panel_c_expected_na:
        panel_c_keys = [
            "band=alpha",
            "enters_meta=true",
            "endpoint=absolute_delta_zlpi",
            "percent_attenuation=false",
            "hiit_sensitivity_display=combined_ph_ps_within_participant_mean",
            f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
        ]
        panel_c_notes = f.annotate_expected_not_applicable(
            f.FIGURE2_PANEL_C_NOTE,
            detail=panel_c_na_detail,
        )
        panel_c_panel_title = "Alpha PRIMARY_META absolute ΔZLPI"
    else:
        panel_c_keys = [
            "band=alpha",
            "enters_meta=true",
            "endpoint=absolute_delta_zlpi",
            "percent_attenuation=false",
            "hiit_sensitivity_display=combined_ph_ps_within_participant_mean",
        ]
        panel_c_notes = f.FIGURE2_PANEL_C_NOTE
        panel_c_panel_title = "Alpha PRIMARY_META absolute ΔZLPI"
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="alpha_primary_meta_forest",
            title=panel_c_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("dataset_effects") or ""),
                str(inputs.get("meta_analysis") or ""),
                str(inputs.get("protocol_audit") or ""),
                str(inputs.get("paired_contrasts") or ""),
            ],
            source_data_csv=str(panel_c_csv),
            analysis_keys=panel_c_keys,
            notes=panel_c_notes,
        )
    )

    # ----- Panel D: band × state model-estimated ZLPI -----
    # Title/legend placed after subplots_adjust so "D" + title share one baseline
    # and the legend sits directly under the title (left-aligned).
    gs_d = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[1, 1],
        height_ratios=list(PANEL_D_MAIN_CONTRAST_HEIGHT_RATIOS),
        hspace=PANEL_D_MAIN_CONTRAST_HSPACE,
    )
    ax_d_main = fig.add_subplot(gs_d[0, 0])
    ax_d_contrast = fig.add_subplot(gs_d[1, 0])
    marginal_rows, contrast_rows, coef_rows = _load_panel_d_tables(
        inputs,
        panel_a_hiit_sensitivity=panel_a_hiit_sensitivity,
    )
    panel_d_hiit_sensitivity = bool(marginal_rows) and panel_a_hiit_sensitivity
    panel_d_title, panel_d_legend_handles = _render_panel_d_estimation_plot(
        ax_d_main,
        ax_d_contrast,
        marginal_rows,
        contrast_rows,
        panel_d_hiit_sensitivity=panel_d_hiit_sensitivity,
    )
    panel_d_marginal_csv = source_dir / "figure2_panel_d_marginal_estimates.csv"
    panel_d_contrast_csv = source_dir / "figure2_panel_d_contrasts.csv"
    panel_d_coef_csv = source_dir / "figure2_panel_d_mixedlm_coefficients.csv"
    f.write_source_csv(
        panel_d_marginal_csv,
        marginal_rows,
        PANEL_D_MARGINAL_EXPORT_FIELDS,
    )
    f.write_source_csv(
        panel_d_contrast_csv,
        contrast_rows,
        PANEL_D_CONTRAST_EXPORT_FIELDS,
    )
    coef_export = [
        {
            "term": f._as_str(row.get("term")),
            "coef": f._as_float(row.get("coef")),
            "stderr": f._as_float(row.get("stderr")),
            "ci_low": f._as_float(row.get("ci_low")),
            "ci_high": f._as_float(row.get("ci_high")),
            "p_value": f._as_float(row.get("p_value")),
            "model_backend": f._as_str(row.get("model_backend")),
            "converged": f._as_str(row.get("converged")),
            "n_obs": f._as_int(row.get("n_obs")),
            "n_groups": f._as_int(row.get("n_groups")),
            "notes": f._as_str(row.get("notes")),
        }
        for row in coef_rows
    ]
    f.write_source_csv(
        panel_d_coef_csv,
        coef_export,
        (
            "term",
            "coef",
            "stderr",
            "ci_low",
            "ci_high",
            "p_value",
            "model_backend",
            "converged",
            "n_obs",
            "n_groups",
            "notes",
        ),
    )
    source_paths.extend([panel_d_marginal_csv, panel_d_contrast_csv, panel_d_coef_csv])
    if panel_d_hiit_sensitivity:
        panel_d_keys = [
            "model=absolute_pooled_state",
            "display=model_estimated_zlpi",
            "vcov_available=true",
            "cohort=HIIT_sensitivity_display",
            "enters_primary_meta=false",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
        ]
        panel_d_notes = (
            "Sensitivity display (display-only): model-estimated Fisher-z ZLPI under "
            "low and high cognitive demand by band from this sensitivity run. "
            "Excluded from PRIMARY_META; not a primary confirmatory claim."
        )
        panel_d_panel_title = f.sensitivity_display_title(
            "band × state interaction", dataset_id="hiit"
        )
    else:
        panel_d_keys = [
            "model=absolute_pooled_state",
            "display=model_estimated_zlpi",
            "vcov_available=true",
        ]
        panel_d_notes = f.FIGURE2_PANEL_D_NOTE
        panel_d_panel_title = "Band × state interaction"
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="mixedlm_band_state_estimation",
            title=panel_d_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("mixed_model_marginal") or inputs.get("mixed_model") or ""),
                str(inputs.get("mixed_model_contrasts") or ""),
                str(inputs.get("subject_level") or ""),
            ],
            source_data_csv=str(panel_d_marginal_csv),
            analysis_keys=panel_d_keys,
            notes=panel_d_notes,
        )
    )

    # ----- Panel E: state-specific μ and FWHM (visualization layout) -----
    gs_e_wrap = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[2, 0],
        height_ratios=[0.40, 1.0],
        hspace=0.30,
    )
    ax_e_title = fig.add_subplot(gs_e_wrap[0, 0])
    ax_e_title.axis("off")
    # μ | shared sample-size column | FWHM — avoids n-label / outlier collisions.
    gs_e = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs_e_wrap[1, 0],
        width_ratios=[1.2, 0.40, 1.2],
        wspace=0.30,
    )
    ax_e_mu = fig.add_subplot(gs_e[0, 0])
    ax_e_n = fig.add_subplot(gs_e[0, 1], sharey=ax_e_mu)
    ax_e_fwhm = fig.add_subplot(gs_e[0, 2], sharey=ax_e_mu)
    panel_e_hiit_sensitivity = False
    peak_params = f.read_csv_rows(inputs.get("peak_params"))
    if meta_pairs:
        peak_export = build_panel_e_peak_export(
            meta_pairs,
            hiit_sensitivity=False,
            peak_params_rows=peak_params,
        )
    else:
        hiit_peak_pairs = filter_hiit_sensitivity_paired_rows(paired)
        peak_export = build_panel_e_peak_export(
            hiit_peak_pairs,
            hiit_sensitivity=True,
            peak_params_rows=peak_params,
        )
        panel_e_hiit_sensitivity = bool(peak_export)
    panel_e_summaries = build_panel_e_state_summaries(peak_export)
    # Visualization regenerations must not churn bootstrap CIs when the
    # underlying state-long observations and point estimates are unchanged.
    prior_summary_csv = source_dir / "figure2_panel_e_state_summaries.csv"
    if prior_summary_csv.is_file():
        prior_summaries = f.read_csv_rows(prior_summary_csv)
        if prior_summaries and _panel_e_point_estimates_match(
            panel_e_summaries, prior_summaries
        ):
            panel_e_summaries = prior_summaries

    panel_e_mu_title = "Peak center, μ"
    panel_e_fwhm_title = "Peak width, FWHM"
    panel_e_rest_color = f.PALETTE["green"]
    panel_e_task_color = f.PALETTE["vermillion"]
    panel_e_legend_handles: list[object] = []

    def _panel_e_summary_lookup(
        band: str, state: str, parameter: str
    ) -> dict[str, object] | None:
        for row in panel_e_summaries:
            if (
                f._as_str(row.get("band")).casefold() == band
                and f._as_str(row.get("state")).casefold() == state
                and f._as_str(row.get("parameter")).casefold() == parameter
            ):
                return row
        return None

    def _plot_panel_e_parameter(
        ax: plt.Axes,
        *,
        parameter: str,
        value_field: str,
        xlabel: str,
        show_zero: bool,
        show_ylabels: bool,
    ) -> None:
        from matplotlib.ticker import MaxNLocator

        rest_color = panel_e_rest_color
        task_color = panel_e_task_color
        all_values: list[float] = []
        y_rest, y_task = -0.22, 0.22

        # Soft band separators (behind everything).
        for bi in range(len(f.BAND_ORDER)):
            if bi % 2 == 1:
                ax.axhspan(
                    bi - 0.48,
                    bi + 0.48,
                    facecolor="#F3F3F3",
                    edgecolor="none",
                    zorder=0,
                    alpha=1.0,
                )

        # Pass 1: faint observation cloud.
        for bi, band in enumerate(f.BAND_ORDER):
            for state, y_off, color in (
                ("rest", y_rest, rest_color),
                ("task", y_task, task_color),
            ):
                values = [
                    float(r[value_field])
                    for r in peak_export
                    if r["band"] == band
                    and r["state"] == state
                    and bool(r.get("peak_identifiable"))
                    and math.isfinite(float(r[value_field]))
                ]
                all_values.extend(values)
                rng = np.random.default_rng(bi + (21 if state == "rest" else 41))
                if values:
                    jitter = (rng.random(len(values)) - 0.5) * 0.08
                    ax.scatter(
                        values,
                        np.full(len(values), bi) + y_off + jitter,
                        color=color,
                        s=5,
                        alpha=0.12,
                        edgecolors="none",
                        zorder=2,
                    )

        # Pass 2: dominant summary diamonds + CIs.
        for bi, band in enumerate(f.BAND_ORDER):
            for state, y_off, color in (
                ("rest", y_rest, rest_color),
                ("task", y_task, task_color),
            ):
                summary = _panel_e_summary_lookup(band, state, parameter)
                if summary is None:
                    continue
                est = f._as_float(summary.get("estimate"))
                lo = f._as_float(summary.get("ci_lower_95"))
                hi = f._as_float(summary.get("ci_upper_95"))
                if not math.isfinite(est):
                    continue
                if math.isfinite(lo) and math.isfinite(hi):
                    ax.errorbar(
                        est,
                        bi + y_off,
                        xerr=[[est - lo], [hi - est]],
                        fmt="D",
                        color=color,
                        markersize=5.5,
                        markeredgecolor="black",
                        markeredgewidth=0.6,
                        ecolor=color,
                        elinewidth=f.LINE_WIDTH + 0.6,
                        capsize=3.5,
                        capthick=f.LINE_WIDTH + 0.3,
                        zorder=8,
                        clip_on=False,
                    )
                else:
                    ax.scatter(
                        [est],
                        [bi + y_off],
                        color=color,
                        s=55,
                        marker="D",
                        edgecolors="black",
                        linewidths=0.9,
                        zorder=8,
                        clip_on=False,
                    )

        if all_values:
            lo_v = float(min(all_values))
            hi_v = float(max(all_values))
            span = max(hi_v - lo_v, 1.0)
            pad = 0.04 * span
            ax.set_xlim(lo_v - pad, hi_v + pad)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.set_yticks(range(len(f.BAND_ORDER)))
        if show_ylabels:
            ax.set_yticklabels([f._band_display(b) for b in f.BAND_ORDER])
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        # Manuscript band order top → bottom: Theta, Alpha, Beta, Low gamma.
        ax.set_ylim(len(f.BAND_ORDER) - 0.55, -0.55)
        ax.set_xlabel(xlabel, fontsize=f.FS_AXIS - 2)
        f._style_axes(ax, grid=True)
        if show_zero:
            # Distinct from light gridlines; not a paired-state null label.
            ax.axvline(
                0.0,
                color=f.PALETTE["dark_gray"],
                lw=1.6,
                ls=(0, (4, 2.5)),
                zorder=1,
                alpha=0.85,
            )

    def _plot_panel_e_sample_sizes(ax: plt.Axes) -> None:
        """Single shared n column: identifiable obs / biological participants."""
        y_rest, y_task = -0.22, 0.22
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(len(f.BAND_ORDER) - 0.55, -0.55)
        for bi in range(len(f.BAND_ORDER)):
            if bi % 2 == 1:
                ax.axhspan(
                    bi - 0.48,
                    bi + 0.48,
                    facecolor="#F3F3F3",
                    edgecolor="none",
                    zorder=0,
                    alpha=1.0,
                )
        # Fixed left edge so R/T strings align across bands.
        count_x = 0.12
        for bi, band in enumerate(f.BAND_ORDER):
            for state, y_off, color, prefix in (
                ("rest", y_rest, panel_e_rest_color, "L"),
                ("task", y_task, panel_e_task_color, "H"),
            ):
                summary = _panel_e_summary_lookup(band, state, "mu")
                if summary is None:
                    summary = _panel_e_summary_lookup(band, state, "fwhm")
                if summary is None:
                    continue
                n_obs = int(summary.get("identifiable_n") or 0)
                n_part = int(summary.get("unique_participant_n") or 0)
                ax.text(
                    count_x,
                    bi + y_off,
                    f"{prefix} {n_obs}/{n_part}",
                    ha="left",
                    va="center",
                    fontsize=f.FS_TICK - 6,
                    color=color,
                    fontfamily="monospace",
                    clip_on=False,
                    zorder=3,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Place caption below the axis so it does not crowd low-gamma counts.
        ax.set_xlabel("")
        ax.text(
            0.5,
            -0.12,
            "n obs/part",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=f.FS_TICK - 6,
            color=f.PALETTE["dark_gray"],
            clip_on=False,
        )
        ax.tick_params(left=False, bottom=False, labelleft=False)

    if peak_export:
        _plot_panel_e_parameter(
            ax_e_mu,
            parameter="mu",
            value_field="peak_center_mu_s",
            xlabel="Peak μ (s)",
            show_zero=True,
            show_ylabels=True,
        )
        _plot_panel_e_sample_sizes(ax_e_n)
        _plot_panel_e_parameter(
            ax_e_fwhm,
            parameter="fwhm",
            value_field="peak_fwhm_s",
            xlabel="FWHM (s)",
            show_zero=False,
            show_ylabels=False,
        )
        ax_e_fwhm.ticklabel_format(axis="x", useOffset=False, style="plain")
        panel_e_expected_na = False
        panel_e_na_detail = ""
        from matplotlib.lines import Line2D

        panel_e_legend_handles = [
            Line2D(
                [0],
                [0],
                color=panel_e_rest_color,
                marker="D",
                linestyle="",
                markersize=4.5,
                markeredgecolor="black",
                markeredgewidth=0.45,
                label=f.format_role_display("low_demand", raw_label="rest"),
            ),
            Line2D(
                [0],
                [0],
                color=panel_e_task_color,
                marker="D",
                linestyle="",
                markersize=4.5,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=f.format_role_display("high_demand", raw_label="task"),
            ),
        ]
    else:
        panel_e_expected_na = True
        panel_e_na_detail = "No PRIMARY_META paired peak rows."
        msg = f.primary_meta_expected_na_message(panel_e_na_detail)
        ax_e_n.axis("off")
        f._mark_empty_panel(
            ax_e_mu,
            msg,
            xlabel="Peak μ (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
        f._mark_empty_panel(
            ax_e_fwhm,
            msg,
            xlabel="FWHM (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
    f._set_panel_title(ax_e_mu, panel_e_mu_title, fontsize=f.FS_PANEL_TITLE - 5, pad=6)
    f._set_panel_title(
        ax_e_fwhm, panel_e_fwhm_title, fontsize=f.FS_PANEL_TITLE - 5, pad=6
    )
    # Keep the middle column title blank so subplot titles stay aligned.
    ax_e_n.set_title(" ", fontsize=f.FS_PANEL_TITLE - 5, pad=6)
    panel_e_csv = source_dir / "figure2_panel_e_state_peaks.csv"
    panel_e_summary_csv = source_dir / "figure2_panel_e_state_summaries.csv"
    # Drop legacy wide paired-peaks export from prior Panel E layout.
    for stale in (
        source_dir / "figure2_panel_e_paired_peaks.csv",
    ):
        if stale.exists():
            stale.unlink()
    f.write_source_csv(panel_e_csv, peak_export, PANEL_E_STATE_LONG_FIELDS)
    f.write_source_csv(panel_e_summary_csv, panel_e_summaries, PANEL_E_SUMMARY_FIELDS)
    source_paths.extend([panel_e_csv, panel_e_summary_csv])
    if panel_e_hiit_sensitivity:
        panel_e_keys = [
            "cohort=HIIT_sensitivity_state_specific_peaks",
            "sampling_unit=biological_participant",
            "estimand=mean_of_participant_means",
            "eligibility=state_specific_identifiable_peak",
            "enters_primary_meta=false",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
        ]
        panel_e_notes = f.FIGURE2_PANEL_E_SENSITIVITY_NOTE
        panel_e_panel_title = f.sensitivity_display_title(
            "Peak center and width by state", dataset_id="hiit"
        )
    elif panel_e_expected_na:
        panel_e_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "sampling_unit=biological_participant",
            "estimand=mean_of_participant_means",
            f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
        ]
        panel_e_notes = f.annotate_expected_not_applicable(
            f.FIGURE2_PANEL_E_NOTE,
            detail=panel_e_na_detail,
        )
        panel_e_panel_title = "Peak center and width by state"
    else:
        panel_e_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "sampling_unit=biological_participant",
            "estimand=mean_of_participant_means",
            "eligibility=state_specific_identifiable_peak",
        ]
        panel_e_notes = f.FIGURE2_PANEL_E_NOTE
        panel_e_panel_title = "Peak center and width by state"
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="state_peaks_mu_fwhm",
            title=panel_e_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("paired_contrasts") or "")],
            source_data_csv=str(panel_e_csv),
            analysis_keys=panel_e_keys,
            notes=panel_e_notes,
        )
    )

    # ----- Panel F: prespecified graded ds003690 -----
    ax_f = fig.add_subplot(gs[2, 1])
    graded: list[dict[str, object]] = []
    for row in effects:
        if f._as_str(row.get("dataset_id")).casefold() != GRADED_DATASET_ID:
            continue
        contrast = f._as_str(row.get("contrast_id")).casefold()
        if contrast not in GRADED_DS003690_CONTRASTS:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        graded.append(
            {
                "dataset_id": GRADED_DATASET_ID,
                "contrast_id": contrast,
                "band": band,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "effect_mean": f._as_float(row.get("effect_mean")),
                "ci_low": f._as_float(row.get("ci_low")),
                "ci_high": f._as_float(row.get("ci_high")),
                "n_pairs": f._as_int(row.get("n_pairs")),
                "enters_meta": f._as_bool(row.get("enters_meta")),
            }
        )

    if graded:
        # Group by contrast then band for y positions.
        contrasts = ["passive__simplert", "passive__gonogo"]
        y_labels: list[str] = []
        positions: list[float] = []
        y = 0.0
        for contrast in contrasts:
            for band in f.BAND_ORDER:
                match = next(
                    (
                        r
                        for r in graded
                        if r["contrast_id"] == contrast and r["band"] == band
                    ),
                    None,
                )
                if match is None:
                    continue
                pe = float(match["effect_mean"])
                lo = float(match["ci_low"])
                hi = float(match["ci_high"])
                xerr = None
                if math.isfinite(pe) and math.isfinite(lo) and math.isfinite(hi):
                    xerr = [[pe - lo], [hi - pe]]
                ax_f.errorbar(
                    pe if math.isfinite(pe) else 0.0,
                    y,
                    xerr=xerr,
                    fmt=f._band_marker(band),
                    color=f._band_color(band),
                    markersize=f.MARKER_SIZE,
                    capsize=3,
                    elinewidth=f.LINE_WIDTH,
                    markeredgecolor=f.PALETTE["dark_gray"],
                    markeredgewidth=0.5,
                    zorder=3,
                )
                short = "simplert" if "simplert" in contrast else "gonogo"
                y_labels.append(f"{short} · {f._band_display(band)}")
                positions.append(y)
                y += 1
        f._ref_vline(ax_f, 0.0)
        ax_f.set_yticks(positions)
        ax_f.set_yticklabels(y_labels, fontsize=f.FS_TICK - 3)
        ax_f.set_xlabel(
            f"Δ {_endpoint_label()} ({f.ZLPI_METRIC}; {f.CI_95_LABEL})",
            fontsize=f.FS_AXIS,
        )
        f._style_axes(ax_f)
        f._set_panel_title(ax_f, "Graded contrasts (ds003690)")
        panel_f_expected_na = False
        panel_f_na_detail = ""
    else:
        panel_f_expected_na = True
        panel_f_na_detail = "ds003690 graded contrasts not present in this run."
        f._mark_empty_panel(
            ax_f,
            (
                "Expected not applicable\n"
                "(ds003690 graded contrasts absent)\n"
                "Empty by design for this run."
            ),
            xlabel=f"Δ ZLPI ({f.CI_95_LABEL})",
            ylabel="Contrast · band",
        )
        f._set_panel_title(ax_f, "Graded contrasts (ds003690)")
    f._add_panel_label(ax_f, "F")
    panel_f_csv = source_dir / "figure2_panel_f_graded_ds003690.csv"
    f.write_source_csv(
        panel_f_csv,
        graded,
        (
            "dataset_id",
            "contrast_id",
            "band",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "effect_mean",
            "ci_low",
            "ci_high",
            "n_pairs",
            "enters_meta",
        ),
    )
    source_paths.append(panel_f_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="graded_ds003690",
            title="Prespecified graded ds003690 contrasts",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("dataset_effects") or "")],
            source_data_csv=str(panel_f_csv),
            analysis_keys=[
                "dataset=ds003690",
                "contrasts=passive__simplert,passive__gonogo",
                "dose_response=false",
                "behavioral=false",
                *(
                    [f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}"]
                    if panel_f_expected_na
                    else []
                ),
            ],
            notes=(
                f.annotate_expected_not_applicable(
                    f.FIGURE2_PANEL_F_NOTE,
                    detail=panel_f_na_detail,
                )
                if panel_f_expected_na
                else f.FIGURE2_PANEL_F_NOTE
            ),
        )
    )

    fig.suptitle(f.FIGURE2_TITLE, fontsize=f.FS_SUPTITLE - 2, fontweight="bold", y=0.992)
    # Compact footer below axes; keep detailed prose in caption/changelog.
    fig.text(
        0.5,
        0.010,
        "\n".join(
            (
                f.LAG_CONVENTION_NOTE,
                "A/B: C5 pairs; paired bootstrap; no unpaired fallback; no cluster permutation.",
                "C: absolute α PRIMARY_META ΔZLPI (no % attenuation). "
                "D: Fisher-z ZLPI band×state interaction (low/high demand by band).",
                "F: prespecified ds003690 graded contrasts only.",
            )
        ),
        ha="center",
        va="bottom",
        fontsize=f.FS_TICK - 6,
        color=f.PALETTE["dark_gray"],
        linespacing=1.40,
    )
    # Extra top margin for the figure title; A/B panel titles live in reserved rows.
    fig.subplots_adjust(left=0.11, right=0.98, top=0.945, bottom=0.145)
    # Panel D: place "D" + title on one baseline above the axes; legend under title.
    pos_d = gs[1, 1].get_position(fig)
    title_y = pos_d.y1 + 0.034
    # loc="lower left" keeps the legend box above this anchor (outside the axes).
    legend_y = pos_d.y1 + 0.006
    fig.text(
        pos_d.x0 - 0.024,
        title_y,
        "D",
        ha="left",
        va="bottom",
        fontsize=f.FS_PANEL_LABEL,
        fontweight="bold",
        fontfamily="sans-serif",
        color=f.PALETTE["dark_gray"],
        clip_on=False,
        zorder=20,
    )
    fig.text(
        pos_d.x0,
        title_y,
        panel_d_title,
        ha="left",
        va="bottom",
        fontsize=f.FS_PANEL_TITLE - 1,
        fontweight="bold",
        color="black",
        clip_on=False,
        zorder=20,
    )
    fig.legend(
        handles=panel_d_legend_handles,
        loc="lower left",
        bbox_to_anchor=(pos_d.x0, legend_y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=f.FS_TICK - 2,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    ax_e_title.text(
        0.0,
        0.78,
        panel_e_panel_title,
        transform=ax_e_title.transAxes,
        ha="left",
        va="center",
        fontsize=f.FS_PANEL_TITLE - 2,
        fontweight="bold",
        color="black",
    )
    # Panel letter aligned with the Panel E title row (not subplot titles).
    ax_e_title.annotate(
        "E",
        xy=(0.0, 0.78),
        xycoords="axes fraction",
        xytext=(-32, 0),
        textcoords="offset points",
        fontsize=f.FS_PANEL_LABEL,
        fontweight="bold",
        fontfamily="sans-serif",
        va="center",
        ha="left",
        color=f.PALETTE["dark_gray"],
        clip_on=False,
        annotation_clip=False,
        zorder=20,
    )
    if panel_e_legend_handles:
        # Center shared low-/high-demand legend across the full Panel E width.
        ax_e_title.legend(
            handles=panel_e_legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            fontsize=f.FS_TICK - 2,
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.35,
            columnspacing=1.4,
        )

    if panel_a_hiit_sensitivity:
        panel_a_caption = (
            "A: Sensitivity display (display-only) — matched low-demand vs high-demand "
            "Fisher-z lag curves from C5 pairs linked to C2; each pre- and "
            "post-intervention pair contributes separately (no pre/post or "
            "protocol-session averaging). "
            "Point estimate = mean across matched pairs; CI = session-condition "
            "cluster bootstrap. Excluded from PRIMARY_META; not a primary confirmatory "
            "claim. Report n_matched_pairs and n_session_clusters.\n"
        )
    else:
        panel_a_caption = (
            "A: Matched low-demand vs high-demand Fisher-z lag curves for "
            "exact C5 PRIMARY_META paired participants (observation_id link to C2); "
            "paired participant-within-dataset bootstrap CIs. "
            "Wiring-gap note if matched curves cannot be reconstructed "
            "(no unpaired fallback).\n"
        )
    if panel_b_hiit_sensitivity:
        panel_b_caption = (
            "B: Sensitivity display (display-only) — matched low-demand–high-demand Δ "
            "Fisher-z lag-difference curves Δz(τ)=z_high−z_low on the same C5 pairs "
            "as Panel A; each pre- and post-intervention pair contributes separately. "
            "Point estimate = mean across matched pairs; CI = session-condition "
            "cluster bootstrap. Excluded from PRIMARY_META; no cluster-permutation "
            "testing. Report n_matched_pairs and n_session_clusters.\n"
        )
    else:
        panel_b_caption = (
            "B: Matched high-demand − low-demand lag-difference curves on the same "
            "pairs; display only — no cluster-permutation testing. Formal lag-0 "
            "attenuation via paired ΔZLPI / PRIMARY_META / C4.\n"
        )
    if panel_c_hiit_only:
        panel_c_caption = (
            "C: Sensitivity display (display-only) — one combined alpha ΔZLPI estimate "
            "(protocol session conditions counted separately). Excluded from "
            "PRIMARY_META RE pooling.\n"
        )
    else:
        panel_c_caption = f"C: {f.FIGURE2_PANEL_C_NOTE}\n"
    if panel_d_hiit_sensitivity:
        panel_d_caption = (
            "D: Sensitivity display (display-only) — band×state interaction; "
            "model-estimated Fisher-z ZLPI under low and high cognitive demand by "
            "band (endpoint_index on the Fisher-z reporting scale); "
            "alpha-versus-other-band state-effect contrasts below main panel "
            "(negative ⇒ stronger alpha attenuation); not PRIMARY_META.\n"
        )
    else:
        panel_d_caption = f"D: {f.FIGURE2_PANEL_D_NOTE}\n"
    if panel_e_hiit_sensitivity:
        panel_e_caption = (
            "E: Sensitivity display (display-only) — Gaussian peak centers (μ) and "
            "widths (FWHM) shown separately for low-demand and high-demand states by "
            "band; summaries and 95% CIs conditional on an identifiable peak in that "
            "state (sample sizes may differ); participant-clustered uncertainty; "
            "descriptive only, not a formal paired low-demand–high-demand test. The "
            "center column (L / H) reports identifiable observations / unique "
            "biological participants once for both μ and FWHM.\n"
        )
    else:
        panel_e_caption = f"E: {f.FIGURE2_PANEL_E_NOTE}\n"
    caption_path = output_dir / "figure2_caption.txt"
    caption_path.write_text(
        (
            f"{f.FIGURE2_TITLE}\n\n"
            f"{panel_a_caption}"
            f"{panel_b_caption}"
            f"{f.LAG_CURVE_DISPLAY_SMOOTH_NOTE}\n"
            f"{panel_c_caption}"
            f"{panel_d_caption}"
            f"{panel_e_caption}"
            f"F: {f.FIGURE2_PANEL_F_NOTE}\n"
        ),
        encoding="utf-8",
    )
    changelog_path = output_dir / "figure2_changelog.md"
    hiit_fallback = (
        panel_a_hiit_sensitivity
        or panel_b_hiit_sensitivity
        or panel_c_hiit_only
        or panel_d_hiit_sensitivity
        or panel_e_hiit_sensitivity
    )
    changelog_extra = (
        "- Sensitivity-only display for Panels A/B/C/D/E: matched "
        "low-demand–high-demand pairs (pre-/post-intervention each contribute where "
        "applicable); excluded from PRIMARY_META. Panel F remains ds003690-only "
        "(expected empty without that dataset).\n"
        if hiit_fallback
        else ""
    )
    changelog_path.write_text(
        (
            "# Figure 2 changelog\n\n"
            "- Six-panel manuscript layout (presentation only; C0–C6 frozen).\n"
            "- Panels A/B: exact C5 pairs → C2 curves via observation IDs; "
            "paired bootstrap; no unpaired fallback; no cluster permutation.\n"
            f"- Panels A/B display-only Gaussian smooth "
            f"(σ = {f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g} s) on plotted mean "
            "curves and ribbon boundaries; source-data CSVs remain unsmoothed; "
            "all CIs/tests from original data.\n"
            f"{changelog_extra}"
            "- Panel C: absolute PRIMARY_META α ΔZLPI forest; no percent attenuation; "
            "one combined sensitivity display-only row never enters RE pooling.\n"
            "- Panel D: Fisher-z ZLPI band×state interaction (low/high demand); "
            "alpha-versus-other-band state-effect contrasts "
            "(negative ⇒ stronger alpha attenuation).\n"
            "- Panel E: state-specific low-/high-demand μ and FWHM (not joint "
            "complete-case); separate state summaries = mean of "
            "biological-participant means with participant-clustered 95% "
            "percentile bootstrap CIs; no Δμ/ΔFWHM or linked state lines; "
            "descriptive only (not a formal paired test). Paired Δ fields "
            "retained upstream in C5 only.\n"
            "- Panel F: prespecified ds003690 graded contrasts only.\n"
            "- Replaces prior 2×2 layout (paired scatter / all-band meta / "
            "D180 / μ-TOST).\n"
            "- Viz refinement: shorter panel titles; Panel D estimation plot with "
            "alpha emphasis; wider Panel E μ–FWHM gap; footer cleared from axes.\n"
        ),
        encoding="utf-8",
    )

    pdf, svg, png = f.save_figure_trio(fig, output_dir, f.FIGURE2_STEM)
    plt.close(fig)
    return f.FigureArtifacts(
        figure_id="figure2",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(source_paths),
        panels=tuple(panel_sources),
    )


__all__ = [
    "PANEL_E_STATE_LONG_FIELDS",
    "PANEL_E_SUMMARY_FIELDS",
    "build_curve_lag_index",
    "build_hiit_sensitivity_panel_a_series",
    "build_panel_e_peak_export",
    "build_panel_e_state_summaries",
    "collapse_hiit_session_lag_series",
    "filter_hiit_sensitivity_paired_rows",
    "filter_primary_meta_paired_rows",
    "hiit_matched_pair_cluster_bootstrap_ci",
    "paired_participant_bootstrap_ci",
    "reconstruct_matched_pair_curves",
    "render_figure2",
]
