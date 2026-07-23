"""Figure 2 six-panel presentation layout (C7 only).

Does not alter C0–C6 analyses. Panels A/B/E use exact C5 paired-intersection
participants linked to C2 lag curves via observation IDs when PRIMARY_META pairs
are present; HIIT-only runs fall back to display-only Rest–Tetris sensitivity.
No unpaired fallback. Panel F remains ds003690-graded only.
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
from .inference import PRIMARY_META_CONTRASTS
from .manifest import FigurePanelSource
from .protocol_audit import PROTOCOL_SPECS

GRADED_DS003690_CONTRASTS = frozenset({"passive__simplert", "passive__gonogo"})
GRADED_DATASET_ID = "ds003690"
HIIT_COMBINED_CONTRAST_ID = "hiit_combined_ph_ps_pre_post_mean"
HIIT_SESSION_LAG_AGGREGATION = "session_subject_mean_of_available_pre_post_curves"
HIIT_MATCHED_PAIR_AGGREGATION = "matched_observation_pairs"
HIIT_CLUSTER_BOOTSTRAP_CI_METHOD = "session_subject_cluster_bootstrap"


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
    """C5 HIIT Rest–Tetris pairs for display-only Panel A/B sensitivity curves.

    Uses the locked HIIT contrast set and primary ZLPI / D240 / absolute_log10
    slice. Does not admit HIIT into PRIMARY_META.
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
    """Average available PRE/POST lag curves within each HIIT session subject.

    Locked hierarchy (matches Figure 1 Panel E / Figure 2 Panel C): PH and PS are
    separate session subjects; within a session, mean available PRE/POST
    Rest–Tetris curves at each lag. ``participant_id`` is rewritten to the
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
        # lag -> lists of (z_low, z_effort) across available PRE/POST contrasts
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

    Each C5 PRE/POST Rest–Tetris pair is retained separately (no PRE/POST or
    PH/PS averaging). ``cluster_id`` marks the PH/PS session subject for
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


def build_panel_e_peak_export(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    hiit_sensitivity: bool = False,
) -> list[dict[str, object]]:
    """Paired low/effort peak μ and FWHM rows for Figure 2 Panel E.

    PRIMARY_META: one row per (dataset, participant, band).
    HIIT sensitivity: one row per matched Rest–Tetris pair
    (participant × session × contrast × band); no PRE/POST or PH/PS collapse.
    """
    f = _fig()
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
        low_peak = f._as_bool(row.get("low_has_identifiable_peak"))
        effort_peak = f._as_bool(row.get("effort_has_identifiable_peak"))
        low_mu = f._as_float(row.get("low_peak_center_mu_s"))
        effort_mu = f._as_float(row.get("effort_peak_center_mu_s"))
        low_fwhm = f._as_float(row.get("low_fwhm_s"))
        effort_fwhm = f._as_float(row.get("effort_fwhm_s"))
        export.append(
            {
                "dataset_id": dataset_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "contrast_id": contrast_id,
                "band": band,
                "low_has_identifiable_peak": low_peak,
                "effort_has_identifiable_peak": effort_peak,
                "low_peak_center_mu_s": low_mu if low_peak else float("nan"),
                "effort_peak_center_mu_s": effort_mu if effort_peak else float("nan"),
                "low_fwhm_s": low_fwhm if low_peak else float("nan"),
                "effort_fwhm_s": effort_fwhm if effort_peak else float("nan"),
                "row_type": (
                    ROW_TYPE_SENSITIVITY_DISPLAY if hiit_sensitivity else "primary"
                ),
            }
        )
    return export


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
    """Observation-level mean with PH/PS session-subject cluster bootstrap.

    Point estimate: equal-weight mean across matched C5 pairs (PRE and POST
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

    fig = plt.figure(figsize=(17.0, 16.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.36)

    # ----- Panel A: matched low vs effort lag curves -----
    gs_a = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0], hspace=0.35, wspace=0.28)
    panel_a_export: list[dict[str, object]] = []
    has_a = False
    panel_a_expected_na = False
    panel_a_na_detail = ""
    ax_a0: plt.Axes | None = None
    panel_a_title = (
        "HIIT Sensitivity: matched low vs effort curves"
        if panel_a_hiit_sensitivity
        else "Matched low vs effort curves"
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
            ax.set_title(f._band_display(band), fontsize=f.FS_TICK)
            if bi >= 2:
                ax.set_xlabel(f.LAG_XLABEL, fontsize=f.FS_AXIS - 4)
            if bi % 2 == 0:
                ax.set_ylabel(f.Z_YLABEL, fontsize=f.FS_AXIS - 4)
            f._style_axes(ax)
            if bi == 0:
                ax.legend(fontsize=f.FS_LEGEND - 4, loc="upper right", frameon=False)
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
        panel_a_notes = f.FIGURE2_PANEL_A_HIIT_SENSITIVITY_NOTE
        panel_a_panel_title = "HIIT Sensitivity: matched low vs effort lag curves"
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
    gs_b = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 1], hspace=0.35, wspace=0.28)
    panel_b_export: list[dict[str, object]] = []
    ax_b0: plt.Axes | None = None
    has_b_nested = False
    panel_b_title = (
        "HIIT Sensitivity: matched lag-difference curves"
        if panel_b_hiit_sensitivity
        else "Matched lag-difference curves"
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
            ax.set_title(f._band_display(band), fontsize=f.FS_TICK)
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
        panel_b_notes = f.FIGURE2_PANEL_B_HIIT_SENSITIVITY_NOTE
        panel_b_panel_title = "HIIT Sensitivity: matched lag-difference curves"
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
        "HIIT Sensitivity: Alpha ΔZLPI"
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
            "HIIT Sensitivity (display-only): one combined alpha ΔZLPI estimate "
            "(PH/PS session subjects counted separately). Excluded from PRIMARY_META "
            "RE pooling; not a primary confirmatory claim."
        )
        panel_c_panel_title = "HIIT Sensitivity: Alpha absolute ΔZLPI"
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

    # ----- Panel D: MixedLM coefficient forest -----
    ax_d = fig.add_subplot(gs[1, 1])
    coef_rows: list[dict[str, object]] = []
    backends: set[str] = set()
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
        if not term or not math.isfinite(coef):
            continue
        backends.add(f._as_str(row.get("model_backend"), "unknown"))
        coef_rows.append(
            {
                "term": term,
                "coef": coef,
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
        )
    # Prefer interaction / state / band terms first for readability.
    def _term_sort(term: str) -> tuple[int, str]:
        t = term.casefold()
        if t == "intercept":
            return (0, t)
        if "state" in t and "band" in t:
            return (1, t)
        if "state" in t:
            return (2, t)
        if "band" in t:
            return (3, t)
        return (4, t)

    coef_rows = sorted(coef_rows, key=lambda r: _term_sort(str(r["term"])))
    panel_d_hiit_sensitivity = bool(coef_rows) and panel_a_hiit_sensitivity
    panel_d_title = (
        "HIIT Sensitivity: absolute state coefficients"
        if panel_d_hiit_sensitivity
        else "Absolute state coefficients"
    )
    if coef_rows:
        for i, row in enumerate(coef_rows):
            pe = float(row["coef"])
            lo = float(row["ci_low"])
            hi = float(row["ci_high"])
            xerr = None
            if math.isfinite(lo) and math.isfinite(hi):
                xerr = [[pe - lo], [hi - pe]]
            ax_d.errorbar(
                pe,
                i,
                xerr=xerr,
                fmt="s",
                color=f.PALETTE["dark_gray"],
                markersize=f.MARKER_SIZE - 1,
                capsize=3,
                elinewidth=f.LINE_WIDTH,
                zorder=3,
            )
        f._ref_vline(ax_d, 0.0)
        ax_d.set_yticks(range(len(coef_rows)))
        ax_d.set_yticklabels(
            [_short_mixedlm_term_label(str(r["term"])) for r in coef_rows],
            fontsize=f.FS_TICK - 3,
        )
        ax_d.set_xlabel("Coefficient (95% CI)", fontsize=f.FS_AXIS - 2)
        f._style_axes(ax_d)
        backend_note = ";".join(sorted(backends)) if backends else "unknown"
        f._set_panel_title(ax_d, panel_d_title)
        ax_d.text(
            0.98,
            0.02,
            backend_note,
            transform=ax_d.transAxes,
            ha="right",
            va="bottom",
            fontsize=f.FS_TICK - 5,
            color=f.PALETTE["dark_gray"],
            style="italic",
        )
    else:
        f._mark_empty_panel(
            ax_d,
            "No mixed_model_results for primary D240 ZLPI.",
            xlabel=f"Coefficient ({f.CI_95_LABEL})",
            ylabel="Term",
        )
        f._set_panel_title(ax_d, panel_d_title)
    f._add_panel_label(ax_d, "D")
    panel_d_csv = source_dir / "figure2_panel_d_mixedlm_coefficients.csv"
    f.write_source_csv(
        panel_d_csv,
        coef_rows,
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
    source_paths.append(panel_d_csv)
    if panel_d_hiit_sensitivity:
        panel_d_keys = [
            "model=absolute_pooled_state",
            "display=coefficients_not_marginal_means",
            "vcov_available=false",
            "cohort=HIIT_sensitivity_display",
            "enters_primary_meta=false",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
        ]
        panel_d_notes = (
            "HIIT Sensitivity (display-only): absolute pooled state model "
            "coefficients from this sensitivity run. Excluded from PRIMARY_META; "
            "not a primary confirmatory claim."
        )
        panel_d_panel_title = "HIIT Sensitivity: absolute state coefficients"
    else:
        panel_d_keys = [
            "model=absolute_pooled_state",
            "display=coefficients_not_marginal_means",
            "vcov_available=false",
        ]
        panel_d_notes = f.FIGURE2_PANEL_D_NOTE
        panel_d_panel_title = "Absolute pooled state model coefficients"
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="mixedlm_coefficients",
            title=panel_d_panel_title,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("mixed_model") or "")],
            source_data_csv=str(panel_d_csv),
            analysis_keys=panel_d_keys,
            notes=panel_d_notes,
        )
    )

    # ----- Panel E: paired low vs effort μ and FWHM -----
    gs_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 0], wspace=0.75)
    ax_e_mu = fig.add_subplot(gs_e[0, 0])
    ax_e_fwhm = fig.add_subplot(gs_e[0, 1])
    panel_e_hiit_sensitivity = False
    if meta_pairs:
        peak_export = build_panel_e_peak_export(meta_pairs, hiit_sensitivity=False)
    else:
        hiit_peak_pairs = filter_hiit_sensitivity_paired_rows(paired)
        peak_export = build_panel_e_peak_export(
            hiit_peak_pairs, hiit_sensitivity=True
        )
        panel_e_hiit_sensitivity = bool(peak_export)

    eq_export: list[dict[str, object]] = []
    for row in equivalence:
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        # Equivalence TOST is PRIMARY_META display context; skip on HIIT-only path.
        if panel_e_hiit_sensitivity:
            continue
        eq_export.append(
            {
                "dataset_id": f._as_str(row.get("dataset_id")),
                "band": f._as_str(row.get("band")).casefold(),
                "mean_mu": f._as_float(row.get("mean_mu")),
                "ci_low": f._as_float(row.get("ci_low")),
                "ci_high": f._as_float(row.get("ci_high")),
                "equivalent": f._as_str(row.get("equivalent")),
                "tost_p": f._as_float(row.get("tost_p")),
            }
        )

    panel_e_mu_title = (
        "HIIT Sensitivity: Peak μ (paired)"
        if panel_e_hiit_sensitivity
        else "Peak μ (paired)"
    )
    panel_e_fwhm_title = (
        "HIIT Sensitivity: FWHM (descriptive)"
        if panel_e_hiit_sensitivity
        else "FWHM (descriptive)"
    )
    if peak_export or eq_export:
        ax_e_mu.axvspan(
            -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            facecolor=f.PALETTE["orange"],
            alpha=0.25,
            zorder=0,
            label=f.MU_EQUIVALENCE_LABEL,
        )
        for bi, band in enumerate(f.BAND_ORDER):
            low_mus = [
                float(r["low_peak_center_mu_s"])
                for r in peak_export
                if r["band"] == band
                and math.isfinite(float(r["low_peak_center_mu_s"]))
            ]
            effort_mus = [
                float(r["effort_peak_center_mu_s"])
                for r in peak_export
                if r["band"] == band
                and math.isfinite(float(r["effort_peak_center_mu_s"]))
            ]
            rng = np.random.default_rng(bi + 21)
            if low_mus:
                jitter = (rng.random(len(low_mus)) - 0.5) * 0.15
                ax_e_mu.scatter(
                    low_mus,
                    np.full(len(low_mus), bi) - 0.15 + jitter,
                    color=f.PALETTE["green"],
                    s=28,
                    alpha=0.55,
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.4,
                    zorder=2,
                )
                ax_e_mu.scatter(
                    [float(np.mean(low_mus))],
                    [bi - 0.15],
                    color=f.PALETTE["green"],
                    s=80,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
            if effort_mus:
                jitter = (rng.random(len(effort_mus)) - 0.5) * 0.15
                ax_e_mu.scatter(
                    effort_mus,
                    np.full(len(effort_mus), bi) + 0.15 + jitter,
                    color=f.PALETTE["vermillion"],
                    s=28,
                    alpha=0.55,
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.4,
                    zorder=2,
                )
                ax_e_mu.scatter(
                    [float(np.mean(effort_mus))],
                    [bi + 0.15],
                    color=f.PALETTE["vermillion"],
                    s=80,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
        ax_e_mu.set_yticks(range(len(f.BAND_ORDER)))
        ax_e_mu.set_yticklabels([f._band_display(b) for b in f.BAND_ORDER])
        ax_e_mu.set_xlabel("Peak μ (s)", fontsize=f.FS_AXIS - 2)
        f._style_axes(ax_e_mu)
        f._ref_vline(ax_e_mu, 0.0)

        for bi, band in enumerate(f.BAND_ORDER):
            low_f = [
                float(r["low_fwhm_s"])
                for r in peak_export
                if r["band"] == band and math.isfinite(float(r["low_fwhm_s"]))
            ]
            effort_f = [
                float(r["effort_fwhm_s"])
                for r in peak_export
                if r["band"] == band and math.isfinite(float(r["effort_fwhm_s"]))
            ]
            rng = np.random.default_rng(bi + 41)
            if low_f:
                jitter = (rng.random(len(low_f)) - 0.5) * 0.15
                ax_e_fwhm.scatter(
                    low_f,
                    np.full(len(low_f), bi) - 0.15 + jitter,
                    color=f.PALETTE["green"],
                    s=28,
                    alpha=0.55,
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.4,
                    zorder=2,
                )
            if effort_f:
                jitter = (rng.random(len(effort_f)) - 0.5) * 0.15
                ax_e_fwhm.scatter(
                    effort_f,
                    np.full(len(effort_f), bi) + 0.15 + jitter,
                    color=f.PALETTE["vermillion"],
                    s=28,
                    alpha=0.55,
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.4,
                    zorder=2,
                )
        ax_e_fwhm.set_yticks(range(len(f.BAND_ORDER)))
        ax_e_fwhm.set_yticklabels([f._band_display(b) for b in f.BAND_ORDER])
        ax_e_fwhm.set_xlabel("FWHM (s)", fontsize=f.FS_AXIS - 2)
        ax_e_fwhm.ticklabel_format(axis="x", useOffset=False, style="plain")
        f._style_axes(ax_e_fwhm)
        panel_e_expected_na = False
        panel_e_na_detail = ""
    else:
        panel_e_expected_na = True
        panel_e_na_detail = "No PRIMARY_META paired peak rows."
        msg = f.primary_meta_expected_na_message(panel_e_na_detail)
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
    f._set_panel_title(ax_e_mu, panel_e_mu_title)
    f._set_panel_title(ax_e_fwhm, panel_e_fwhm_title)
    f._add_panel_label(ax_e_mu, "E")
    panel_e_csv = source_dir / "figure2_panel_e_paired_peaks.csv"
    f.write_source_csv(
        panel_e_csv,
        peak_export,
        (
            "dataset_id",
            "participant_id",
            "session_id",
            "contrast_id",
            "band",
            "low_has_identifiable_peak",
            "effort_has_identifiable_peak",
            "low_peak_center_mu_s",
            "effort_peak_center_mu_s",
            "low_fwhm_s",
            "effort_fwhm_s",
            "row_type",
        ),
    )
    source_paths.append(panel_e_csv)
    if panel_e_hiit_sensitivity:
        panel_e_keys = [
            "cohort=HIIT_sensitivity_matched_observation_pairs",
            "sampling_unit=matched_rest_tetris_pair",
            "aggregation=matched_observation_pairs",
            "fwhm=descriptive",
            "enters_primary_meta=false",
            f"panel_status={f.PANEL_STATUS_SENSITIVITY_DISPLAY}",
        ]
        panel_e_notes = f.FIGURE2_PANEL_E_HIIT_SENSITIVITY_NOTE
        panel_e_panel_title = "HIIT Sensitivity: paired low vs effort μ and FWHM"
    elif panel_e_expected_na:
        panel_e_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "sampling_unit=dataset_participant",
            "fwhm=descriptive",
            f"panel_status={f.PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
        ]
        panel_e_notes = f.annotate_expected_not_applicable(
            f.FIGURE2_PANEL_E_NOTE,
            detail=panel_e_na_detail,
        )
        panel_e_panel_title = "Paired low vs effort μ and FWHM"
    else:
        panel_e_keys = [
            "cohort=PRIMARY_META_C5_pairs",
            "sampling_unit=dataset_participant",
            "fwhm=descriptive",
        ]
        panel_e_notes = f.FIGURE2_PANEL_E_NOTE
        panel_e_panel_title = "Paired low vs effort μ and FWHM"
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="paired_peaks_mu_fwhm",
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
                f"Expected not applicable — {panel_f_na_detail} "
                "Panel F requires the prespecified ds003690 graded contrasts "
                "and is empty by design when that dataset is absent."
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

    fig.suptitle(f.FIGURE2_TITLE, fontsize=f.FS_SUPTITLE - 2, fontweight="bold", y=0.988)
    # Compact footer below axes; keep detailed prose in caption/changelog.
    fig.text(
        0.5,
        0.012,
        "\n".join(
            (
                f.LAG_CONVENTION_NOTE,
                "A/B: C5 pairs; paired bootstrap; no unpaired fallback; no cluster permutation. "
                "C: absolute α PRIMARY_META ΔZLPI (no % attenuation). "
                "D: absolute-model coefficients (not marginal means). "
                "F: prespecified ds003690 graded contrasts only.",
            )
        ),
        ha="center",
        va="bottom",
        fontsize=f.FS_TICK - 6,
        color=f.PALETTE["dark_gray"],
        linespacing=1.35,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.945, bottom=0.165)
    # Nested A/B titles after layout so positions clear band subplot titles.
    if has_a and ax_a0 is not None:
        pos_a = gs[0, 0].get_position(fig)
        fig.text(
            pos_a.x0,
            min(pos_a.y1 + 0.010, 0.96),
            panel_a_title,
            ha="left",
            va="bottom",
            fontsize=f.FS_PANEL_TITLE - 2,
            color=f.PALETTE["dark_gray"],
        )
    if has_b_nested and ax_b0 is not None:
        pos_b = gs[0, 1].get_position(fig)
        fig.text(
            pos_b.x0,
            min(pos_b.y1 + 0.010, 0.96),
            panel_b_title,
            ha="left",
            va="bottom",
            fontsize=f.FS_PANEL_TITLE - 2,
            color=f.PALETTE["dark_gray"],
        )

    if panel_a_hiit_sensitivity:
        panel_a_caption = (
            "A: HIIT Sensitivity (display-only) — matched low-demand vs cognitive-effort "
            "Fisher-z lag curves from C5 Rest–Tetris pairs linked to C2; each PRE and "
            "POST pair contributes separately (no PRE/POST or PH/PS averaging). "
            "Point estimate = mean across matched pairs; CI = PH/PS session-subject "
            "cluster bootstrap. Excluded from PRIMARY_META; not a primary confirmatory "
            "claim. Report n_matched_pairs and n_session_clusters.\n"
        )
    else:
        panel_a_caption = (
            "A: Matched low-demand vs cognitive-effort Fisher-z lag curves for "
            "exact C5 PRIMARY_META paired participants (observation_id link to C2); "
            "paired participant-within-dataset bootstrap CIs. "
            "Wiring-gap note if matched curves cannot be reconstructed "
            "(no unpaired fallback).\n"
        )
    if panel_b_hiit_sensitivity:
        panel_b_caption = (
            "B: HIIT Sensitivity (display-only) — matched Rest–Tetris Δ Fisher-z "
            "lag-difference curves Δz(τ)=z_effort−z_low on the same C5 pairs as Panel A; "
            "each PRE and POST pair contributes separately. Point estimate = mean "
            "across matched pairs; CI = PH/PS session-subject cluster bootstrap. "
            "Excluded from PRIMARY_META; no cluster-permutation testing. "
            "Report n_matched_pairs and n_session_clusters.\n"
        )
    else:
        panel_b_caption = (
            "B: Matched task−rest lag-difference curves on the same pairs; "
            "display only — no cluster-permutation testing. Formal lag-0 "
            "attenuation via paired ΔZLPI / PRIMARY_META / C4.\n"
        )
    if panel_c_hiit_only:
        panel_c_caption = (
            "C: HIIT Sensitivity (display-only) — one combined alpha ΔZLPI estimate "
            "(PH/PS session subjects counted separately). Excluded from PRIMARY_META "
            "RE pooling.\n"
        )
    else:
        panel_c_caption = f"C: {f.FIGURE2_PANEL_C_NOTE}\n"
    if panel_d_hiit_sensitivity:
        panel_d_caption = (
            "D: HIIT Sensitivity (display-only) — absolute pooled state model "
            "coefficients from this sensitivity run; not PRIMARY_META.\n"
        )
    else:
        panel_d_caption = f"D: {f.FIGURE2_PANEL_D_NOTE}\n"
    if panel_e_hiit_sensitivity:
        panel_e_caption = (
            "E: HIIT Sensitivity (display-only) — paired Rest–Tetris peak μ and FWHM "
            "from C5 matched pairs (PRE/POST each contribute; no PRE/POST or PH/PS "
            "collapse); FWHM descriptive only.\n"
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
        "- HIIT-only sensitivity display for Panels A/B/C/D/E: matched Rest–Tetris "
        "pairs (PRE/POST each contribute where applicable); excluded from "
        "PRIMARY_META. Panel F remains ds003690-only (expected empty without that "
        "dataset).\n"
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
            "one combined HIIT display-only sensitivity row never enters RE pooling.\n"
            "- Panel D: MixedLM coefficient forest (not marginal means).\n"
            "- Panel E: paired low/effort μ and FWHM; FWHM descriptive.\n"
            "- Panel F: prespecified ds003690 graded contrasts only.\n"
            "- Replaces prior 2×2 layout (paired scatter / all-band meta / "
            "D180 / μ-TOST).\n"
            "- Viz refinement: shorter panel titles; shortened Panel D coefficient "
            "display labels; wider Panel E μ–FWHM gap; footer cleared from axes.\n"
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
    "build_curve_lag_index",
    "build_hiit_sensitivity_panel_a_series",
    "build_panel_e_peak_export",
    "collapse_hiit_session_lag_series",
    "filter_hiit_sensitivity_paired_rows",
    "filter_primary_meta_paired_rows",
    "hiit_matched_pair_cluster_bootstrap_ci",
    "paired_participant_bootstrap_ci",
    "reconstruct_matched_pair_curves",
    "render_figure2",
]
