"""Figure 1 six-panel presentation layout (C7 only).

Does not alter C0–C6 analyses. Reuses frozen confirmatory tables and existing
definitions for ZLPI, lag windows, PRIMARY_META membership, and μ TOST.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .duration_contracts import (
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    contract_for_duration,
)
from .endpoints import fisher_z
from .dataset_roles import (
    ROLE_PRIMARY,
    ROLE_SENSITIVITY,
    ROLE_UNKNOWN,
    is_runtime_blocked,
    resolve_dataset_role,
    role_tag,
)
from .forest_display import (
    FOREST_EXPORT_FIELDS,
    build_alpha_forest_export,
    draw_alpha_meta_forest,
)
from .manifest import FigurePanelSource
from .inference import low_demand_alpha_replication
from .null_delta_inference import PRIMARY_NULL_TYPE

# Imported after figures constants exist; callers invoke via deferred import.
def _fig():
    from . import figures as f

    return f


def bootstrap_mean_ci_by_lag(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    condition_role: str | None = "low_demand",
    power_representation: str | None = None,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    ci_percent: float | None = None,
) -> list[dict[str, object]]:
    """Mean Fisher-z with participant-within-dataset bootstrap percentile CIs.

    Within each dataset, participants are resampled with replacement. When a
    participant is drawn, all of that participant's repeated observations are
    retained (with multiplicity if drawn more than once). Presentation only.
    """
    f = _fig()
    power_representation = power_representation or f.PRIMARY_REPRESENTATION
    n_bootstrap = int(n_bootstrap if n_bootstrap is not None else f.FIGURE1_BOOTSTRAP_N)
    seed = int(seed if seed is not None else f.FIGURE1_BOOTSTRAP_SEED)
    ci_percent = float(
        ci_percent if ci_percent is not None else f.FIGURE1_BOOTSTRAP_CI_PERCENT
    )

    # observation_id -> {lag -> z}; also participant index.
    obs_lags: dict[str, dict[int, float]] = {}
    # dataset -> list of (participant_key, list[observation_id])
    by_dataset: dict[str, dict[str, list[str]]] = {}
    participants: set[tuple[str, str]] = set()

    for row in curve_rows:
        if f._as_int(row.get("duration_s"), duration_s) != duration_s:
            continue
        if f._as_str(row.get("band")).casefold() != band.casefold():
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != power_representation.casefold()
        ):
            continue
        condition = f._as_str(row.get("condition") or row.get("task")).casefold()
        role = f._as_str(row.get("condition_role") or row.get("state")).casefold()
        if condition_role == "low_demand":
            normalized_state = f._as_str(
                row.get("normalized_state") or row.get("state_role")
            ).casefold()
            if role not in {"low_demand", "state_low"} and normalized_state != "state_low":
                dataset_id = f._as_str(row.get("dataset_id")).casefold()
                state_role = ""
                if dataset_id and condition:
                    state_role, _time, _session = f.condition_semantics_for(
                        dataset_id, condition
                    )
                if state_role != "state_low":
                    continue
        lag = int(round(f._as_float(row.get("lag_s"))))
        r = f._as_float(row.get("r"))
        z = fisher_z(r) if math.isfinite(r) else float("nan")
        if not math.isfinite(z):
            continue
        dataset = f._as_str(row.get("dataset_id"), "unknown")
        subject = f._as_str(
            row.get("subject_id") or row.get("participant_id") or row.get("observation_id"),
            "unknown",
        )
        obs_id = f._as_str(row.get("observation_id"), f"{dataset}:{subject}:default")
        obs_lags.setdefault(obs_id, {})[lag] = z
        by_dataset.setdefault(dataset, {}).setdefault(subject, [])
        if obs_id not in by_dataset[dataset][subject]:
            by_dataset[dataset][subject].append(obs_id)
        participants.add((dataset, subject))

    if not obs_lags:
        return []

    all_lags = sorted({lag for lags in obs_lags.values() for lag in lags})
    # Point estimate: pool all observations (matches historic mean_ci_by_lag mean).
    buckets: dict[int, list[float]] = {lag: [] for lag in all_lags}
    for lag_map in obs_lags.values():
        for lag, z in lag_map.items():
            buckets[lag].append(z)
    mean_by_lag = {lag: float(np.mean(vals)) for lag, vals in buckets.items() if vals}

    # Bootstrap: participant-within-dataset with observation multiplicity.
    rng = np.random.default_rng(seed)
    alpha = (100.0 - ci_percent) / 2.0
    boot_means = {lag: np.empty(n_bootstrap, dtype=float) for lag in all_lags}
    dataset_subjects = {
        ds: list(subjects.keys()) for ds, subjects in by_dataset.items()
    }

    for b in range(n_bootstrap):
        collected: dict[int, list[float]] = {lag: [] for lag in all_lags}
        for ds, subjects in dataset_subjects.items():
            if not subjects:
                continue
            draw = rng.choice(subjects, size=len(subjects), replace=True)
            for subject in draw:
                for obs_id in by_dataset[ds][subject]:
                    for lag, z in obs_lags[obs_id].items():
                        collected[lag].append(z)
        for lag in all_lags:
            vals = collected[lag]
            boot_means[lag][b] = float(np.mean(vals)) if vals else float("nan")

    n_participants = len(participants)
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
                "band": band,
                "duration_s": duration_s,
                "endpoint_name": contract_for_duration(duration_s).endpoint_name,
                "power_representation": power_representation,
                "n": n_participants,
                "n_observations": sum(len(v) for v in buckets.values()) // max(len(all_lags), 1),
                "mean_z": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": "participant_within_dataset_bootstrap",
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _dataset_role(dataset_id: str, protocol_rows: Sequence[Mapping[str, object]]) -> str:
    """Centralized role lookup: C0 rows first, then ``master.yaml``.

    The config fallback matters when a run's C0 ``protocol_audit`` table is
    missing or stale; without it a configured sensitivity dataset rendered as an
    unknown role.
    """
    role = resolve_dataset_role(dataset_id, protocol_rows=protocol_rows)
    return "other" if role == ROLE_UNKNOWN else role


def _dataset_role_from_config(dataset_id: str) -> str:
    """Read the configured role when source rows do not carry it."""
    role = resolve_dataset_role(dataset_id)
    return "other" if role == ROLE_UNKNOWN else role


def _blocked_sensitivity_datasets(
    dataset_ids: set[str],
    protocol_rows: Sequence[Mapping[str, object]],
) -> set[str]:
    blocked: set[str] = set()
    for dataset_id in dataset_ids:
        ds = str(dataset_id).strip().casefold()
        if not ds:
            continue
        role = resolve_dataset_role(ds, protocol_rows=protocol_rows)
        if role == ROLE_SENSITIVITY and is_runtime_blocked(ds):
            blocked.add(ds)
    return blocked


def _drop_blocked_sensitivity_rows(
    rows: Sequence[Mapping[str, object]],
    blocked_dataset_ids: set[str],
) -> list[dict[str, object]]:
    if not blocked_dataset_ids:
        return [dict(row) for row in rows]
    f = _fig()
    out: list[dict[str, object]] = []
    for row in rows:
        ds = f._as_str(row.get("dataset_id")).casefold()
        if ds in blocked_dataset_ids:
            continue
        out.append(dict(row))
    return out


def _cardiac_modality(dataset_id: str, protocol_rows: Sequence[Mapping[str, object]]) -> str:
    f = _fig()
    key = f._as_str(dataset_id).casefold()
    for row in protocol_rows:
        if f._as_str(row.get("dataset_id")).casefold() != key:
            continue
        modality = f._as_str(row.get("cardiac_modality"))
        if modality:
            # Metadata annotation: prefer primary token before semicolon.
            return modality.split(";")[0].strip() or modality
    return ""


def participant_lag_category_rows(
    endpoint_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Participant means of lag-0 / shoulder / flank Fisher-z (display aggregation)."""
    f = _fig()
    buckets: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for row in endpoint_rows:
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if "eligible" in row and not f._as_bool(row.get("eligible"), True):
            continue
        if not f._is_low_demand_condition(row):
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        z0 = f._as_float(row.get("z0"))
        neg_s = f._as_float(row.get("negative_shoulder_mean_z"))
        pos_s = f._as_float(row.get("positive_shoulder_mean_z"))
        flank = f._as_float(row.get("combined_flank_mean_z"))
        if not all(math.isfinite(v) for v in (z0, neg_s, pos_s, flank)):
            continue
        shoulder = max(neg_s, pos_s)
        key = (
            f._as_str(row.get("dataset_id")),
            f._as_str(row.get("subject_id") or row.get("participant_id")),
            band,
        )
        buckets.setdefault(key, []).append(
            {
                "z0": z0,
                "shoulder_ref_z": shoulder,
                "flank_z": flank,
                "endpoint_index": f._as_float(row.get("endpoint_index")),
                "local_prominence": f._as_float(row.get("local_prominence")),
            }
        )
    out: list[dict[str, object]] = []
    for (dataset_id, subject_id, band), members in sorted(buckets.items()):
        out.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "band": band,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "endpoint_name": ENDPOINT_ZLPI,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "z0": float(np.mean([m["z0"] for m in members])),
                "shoulder_ref_z": float(np.mean([m["shoulder_ref_z"] for m in members])),
                "flank_z": float(np.mean([m["flank_z"] for m in members])),
                "endpoint_index": float(
                    np.nanmean([m["endpoint_index"] for m in members])
                ),
                "local_prominence": float(
                    np.nanmean([m["local_prominence"] for m in members])
                ),
                "n_observations": len(members),
            }
        )
    return out


def subject_level_mean_zlpi_cells(
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Dataset × band mean low-demand ZLPI from subject_level_metrics.

    Observations with an explicit normalized session are first averaged within
    participant-session. This provides the former session-sensitive display
    behavior without relying on a dataset identity.
    """
    f = _fig()
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for row in subject_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if "endpoint_eligible" in row and not f._as_bool(row.get("endpoint_eligible"), True):
            continue
        if not f._is_low_demand_condition(row):
            continue
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        value = f._as_float(row.get("endpoint_index"))
        if not math.isfinite(value):
            continue
        participant = f._as_str(row.get("participant_id") or row.get("subject_id"))
        session = f._as_str(row.get("session_id"), "single").casefold() or "single"
        unit = f"{participant}::{session}" if participant else f._as_str(
            row.get("observation_id"), "unknown"
        )
        key = (f._as_str(row.get("dataset_id")), band, unit)
        buckets.setdefault(key, []).append(value)
    by_cell: dict[tuple[str, str], list[float]] = {}
    for (dataset_id, band, _unit), values in buckets.items():
        by_cell.setdefault((dataset_id, band), []).append(float(np.mean(values)))
    return [
        {
            "dataset_id": ds,
            "source_dataset_id": ds,
            "session_id": "",
            "display_label": str(ds).upper(),
            "dataset_role": _dataset_role_from_config(str(ds)),
            "band": band,
            "mean_zlpi": float(np.mean(vals)),
            "n_participants": len(vals),
            "n_participant_sessions": len(vals),
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "power_representation": f.PRIMARY_REPRESENTATION,
            "aggregation": "session_subject_mean_of_available_low_demand_zlpi",
        }
        for (ds, band), vals in sorted(by_cell.items())
    ]


def surrogate_significance_marks(
    null_summary_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], bool]:
    """Prespecified display rule: median_empirical_p < α for PRIMARY_NULL_TYPE.

    Default: median of condition-level median_empirical_p within (dataset, band).
    Sensitivity cohorts with multi-session protocols may use one combined mark
    per band from all low-demand pre-/post-intervention condition-level p-values
    (see ``hiit_session_surrogate_significance_marks`` for the retained HIIT
    display aggregation).
    """
    f = _fig()
    by_cell: dict[tuple[str, str], list[float]] = {}
    for row in null_summary_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if f._as_str(row.get("null_type")).casefold() != PRIMARY_NULL_TYPE.casefold():
            continue
        if not f._is_low_demand_condition(row) and f._as_str(row.get("condition")):
            # null_summary uses condition; allow empty condition.
            condition = f._as_str(row.get("condition")).casefold()
            if condition and condition not in f.LOW_DEMAND_CONDITION_LABELS:
                if not any(
                    token in condition for token in ("rest", "passive", "low")
                ):
                    continue
        p = f._as_float(row.get("median_empirical_p"))
        if not math.isfinite(p):
            continue
        key = (f._as_str(row.get("dataset_id")), f._as_str(row.get("band")).casefold())
        by_cell.setdefault(key, []).append(p)
    marks = {
        key: float(np.median(ps)) < f.FIGURE1_PANEL_D_SURROGATE_ALPHA
        for key, ps in by_cell.items()
    }
    return marks


def alpha_replication_forest_rows(
    subject_rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
    low_demand_effect_rows: Sequence[Mapping[str, object]] | None = None,
    low_demand_meta_rows: Sequence[Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object] | None]:
    """Low-demand D240 alpha ZLPI rows + pooled RE for Figure 1 Panel E."""
    f = _fig()
    computed_effects, computed_meta, _computed_loo = low_demand_alpha_replication(
        subject_rows
    )
    effect_rows = list(low_demand_effect_rows or computed_effects)
    if low_demand_meta_rows:
        meta_row = dict(low_demand_meta_rows[0])
    else:
        meta_row = dict(computed_meta)

    studies: list[dict[str, object]] = []
    for row in effect_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        if not dataset_id:
            continue
        if _dataset_role(dataset_id, protocol_rows) != ROLE_PRIMARY:
            continue
        studies.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": "primary",
                "analysis_family": "low_demand_alpha_replication",
                "contrast_id": "low_demand_mean_zlpi",
                "band": "alpha",
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "effect_mean": f._as_float(row.get("effect_mean")),
                "ci_low": f._as_float(row.get("ci_low")),
                "ci_high": f._as_float(row.get("ci_high")),
                "n_pairs": f._as_int(row.get("n_participants")),
                "n_participants": f._as_int(row.get("n_participants")),
                "enters_meta": f._as_bool(row.get("enters_meta"), True),
                "cardiac_modality": _cardiac_modality(dataset_id, protocol_rows),
                "prediction_low": "",
                "prediction_high": "",
                "n_datasets": "",
                "row_type": "primary",
                "display_label": "",
            }
        )
    studies.sort(key=lambda r: f._as_str(r["dataset_id"]).casefold())
    pooled: dict[str, object] | None = None
    if (
        f._as_str(meta_row.get("analysis_status")).casefold() == "completed"
        and f._as_int(meta_row.get("n_datasets")) >= 2
    ):
        pooled = {
            "band": "alpha",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "power_representation": f.PRIMARY_REPRESENTATION,
            "pooled_effect": f._as_float(meta_row.get("pooled_effect")),
            "ci_low": f._as_float(meta_row.get("ci_low")),
            "ci_high": f._as_float(meta_row.get("ci_high")),
            "prediction_low": f._as_float(meta_row.get("prediction_low")),
            "prediction_high": f._as_float(meta_row.get("prediction_high")),
            "n_datasets": f._as_int(meta_row.get("n_datasets")),
            "analysis_status": f._as_str(meta_row.get("analysis_status")),
            "i2": f._as_float(meta_row.get("i2")),
            "tau2": f._as_float(meta_row.get("tau2")),
        }
    return studies, [], pooled


def _forest_unavailable_message(
    studies: Sequence[Mapping[str, object]],
) -> str:
    if not studies:
        return "No eligible low-demand D240 alpha ZLPI observations for primary datasets."
    return "No pooled random-effects estimate available (need at least two eligible datasets)."


def _panel_c_group_summaries(
    cat_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Band × category means and participant SEM for Panel C display."""
    f = _fig()
    categories = ("z0", "shoulder_ref_z", "flank_z")
    out: list[dict[str, object]] = []
    for band in f.BAND_ORDER:
        band_rows = [r for r in cat_rows if f._as_str(r.get("band")) == band]
        if not band_rows:
            continue
        for cat in categories:
            vals = np.asarray(
                [float(r[cat]) for r in band_rows if math.isfinite(float(r[cat]))],
                dtype=float,
            )
            if vals.size == 0:
                continue
            sem = (
                float(np.std(vals, ddof=1) / math.sqrt(vals.size))
                if vals.size >= 2
                else float("nan")
            )
            out.append(
                {
                    "band": band,
                    "category": cat,
                    "mean_z": float(np.mean(vals)),
                    "sem_z": sem,
                    "n_participants": int(vals.size),
                }
            )
    return out


def _mu_axis_limits(
    participant_peak_rows: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    """Focus μ axis on equivalence band and identifiable participant peaks."""
    mus = [
        float(r["peak_center_mu_s"])
        for r in participant_peak_rows
        if math.isfinite(float(r.get("peak_center_mu_s", float("nan"))))
    ]
    pad = 3.0
    equiv = float(EXPECTED_PEAK_CENTER_EQUIVALENCE_S)
    if mus:
        lo = min(-equiv - pad, float(np.percentile(mus, 5)) - pad)
        hi = max(equiv + pad, float(np.percentile(mus, 95)) + pad)
    else:
        lo, hi = -equiv - pad, equiv + pad
    if hi - lo > 24.0:
        center = float(np.median(mus)) if mus else 0.0
        lo = center - 12.0
        hi = center + 12.0
    return lo, hi


def _fwhm_axis_limits(
    participant_peak_rows: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    """FWHM axis from participant values; cap display span for outliers."""
    fwhms = [
        float(r["fwhm_s"])
        for r in participant_peak_rows
        if math.isfinite(float(r.get("fwhm_s", float("nan"))))
    ]
    if not fwhms:
        return 0.0, 20.0
    hi = float(np.percentile(fwhms, 90)) * 1.25
    hi = max(hi, 12.0)
    hi = min(hi, 40.0)
    return 0.0, hi


def _figure1_footer_lines() -> tuple[str, ...]:
    f = _fig()
    return (
        f.LAG_CONVENTION_NOTE,
        "Panel B ribbons: participant-within-dataset percentile bootstrap (95%).",
        "Alpha is confirmatory; other bands are secondary. μ ±2 s equivalence not established.",
    )


def _fig1_band_short(band: str) -> str:
    key = str(band).casefold()
    return {
        "theta": "Theta",
        "alpha": "Alpha",
        "beta": "Beta",
        "low_gamma": "Low-γ",
    }.get(key, _fig()._band_display(band))


def _fig1_dataset_id_label(dataset_id: str) -> str:
    """Keep accession IDs traceable (ds003690 / ds003838 / ds006848)."""
    key = _fig()._as_str(dataset_id)
    if key.casefold().startswith("ds"):
        return key.casefold()
    return _fig()._dataset_display(key)


def _draw_schematic(ax: plt.Axes) -> None:
    f = _fig()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.70, 0.22, 0.24, "ECG or PPG\n→ instantaneous HR"),
        (0.28, 0.70, 0.22, 0.24, "EEG multitaper\nθ / α / β / low-γ"),
        (0.54, 0.70, 0.22, 0.24, "Lag-resolved r\n→ Fisher-z"),
        (0.80, 0.70, 0.18, 0.24, "ZLPI &\npeak metrics"),
        (0.28, 0.28, 0.22, 0.24, "Near-zero peak\n(μ, FWHM)"),
        (0.54, 0.28, 0.44, 0.24, "Replication &\ninference"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=f.PALETTE["dark_gray"],
            facecolor="#F7F7F7",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=f.FS_TICK - 2,
            color=f.PALETTE["dark_gray"],
        )
    arrows = [
        ((0.24, 0.82), (0.28, 0.82)),
        ((0.50, 0.82), (0.54, 0.82)),
        ((0.76, 0.82), (0.80, 0.82)),
        ((0.65, 0.70), (0.39, 0.52)),
        ((0.65, 0.70), (0.76, 0.52)),
        ((0.50, 0.40), (0.54, 0.40)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color=f.PALETTE["dark_gray"],
            )
        )
    ax.text(
        0.5,
        0.10,
        "Implemented confirmatory pipeline (not ECG–vs–PPG).",
        ha="center",
        va="center",
        fontsize=f.FS_TICK - 3,
        color=f.PALETTE["dark_gray"],
        style="italic",
    )
    f._set_panel_title(ax, "Analysis schematic")
    f._add_panel_label(ax, "A")


def render_figure1(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
):
    """Six-panel manuscript Figure 1 (presentation layout only)."""
    f = _fig()
    f._configure_publication_style()
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    # Drop legacy Figure 1 per-band mean±SE exports from prior layout.
    for stale in source_dir.glob("figure1_panel_*_mean_ci.csv"):
        stale.unlink(missing_ok=True)
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    curves = f.read_csv_rows(inputs.get("curves_d240"))
    endpoints = f.read_csv_rows(inputs.get("endpoints_d240"))
    subjects = f.read_csv_rows(inputs.get("subject_level"))
    null_summary = f.read_csv_rows(inputs.get("null_summary"))
    low_demand_alpha_effects = f.read_csv_rows(inputs.get("low_demand_alpha_effects"))
    low_demand_alpha_meta = f.read_csv_rows(inputs.get("low_demand_alpha_meta"))
    peaks = f.read_csv_rows(inputs.get("peak_params"))
    equivalence = f.read_csv_rows(inputs.get("peak_equivalence"))
    peak_hier = f.read_csv_rows(inputs.get("peak_hierarchical"))
    protocol = f.read_csv_rows(inputs.get("protocol_audit"))
    dataset_ids_in_run = {
        f._as_str(row.get("dataset_id")).casefold()
        for table in (curves, endpoints, subjects, peaks, protocol)
        for row in table
        if f._as_str(row.get("dataset_id"))
    }
    blocked_sensitivity_ids = _blocked_sensitivity_datasets(dataset_ids_in_run, protocol)
    curves = _drop_blocked_sensitivity_rows(curves, blocked_sensitivity_ids)
    endpoints = _drop_blocked_sensitivity_rows(endpoints, blocked_sensitivity_ids)
    subjects = _drop_blocked_sensitivity_rows(subjects, blocked_sensitivity_ids)
    null_summary = _drop_blocked_sensitivity_rows(null_summary, blocked_sensitivity_ids)
    low_demand_alpha_effects = _drop_blocked_sensitivity_rows(
        low_demand_alpha_effects, blocked_sensitivity_ids
    )
    peaks = _drop_blocked_sensitivity_rows(peaks, blocked_sensitivity_ids)
    equivalence = _drop_blocked_sensitivity_rows(equivalence, blocked_sensitivity_ids)
    peak_hier = _drop_blocked_sensitivity_rows(peak_hier, blocked_sensitivity_ids)

    fig = plt.figure(figsize=(17.0, 15.4))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=0.075,
        right=0.975,
        top=0.935,
        bottom=0.145,
        wspace=0.26,
        hspace=0.36,
    )

    # ----- Panel A: schematic -----
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_schematic(ax_a)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="schematic",
            title="Analysis schematic",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[],
            source_data_csv="",
            analysis_keys=["panel=A", "schematic=implemented_pipeline"],
            notes="Artist schematic of implemented C1–C6 workflow.",
        )
    )

    # ----- Panel B: multi-band lag curves + bootstrap CI -----
    ax_b = fig.add_subplot(gs[0, 1])
    primary_contract = contract_for_duration(EXPECTED_PRIMARY_DURATION_S)
    series_all: list[dict[str, object]] = []
    plotted: dict[str, tuple[object, str]] = {}
    any_series = False
    n_participants_note = 0
    series_by_band: dict[str, list[dict[str, object]]] = {}
    for band in f.BAND_ORDER:
        series = bootstrap_mean_ci_by_lag(curves, band=band, condition_role="low_demand")
        series_all.extend(series)
        if series:
            series_by_band[band] = series
            any_series = True
            n_participants_note = int(series[0]["n"])
    # Draw non-alpha first; alpha last so confirmatory band sits on top.
    band_draw_order = [b for b in f.BAND_ORDER if b != "alpha"] + (
        ["alpha"] if "alpha" in series_by_band else []
    )
    for band in band_draw_order:
        series = series_by_band.get(band)
        if not series:
            continue
        lags = np.asarray([r["lag_s"] for r in series], dtype=float)
        mean = np.asarray([r["mean_z"] for r in series], dtype=float)
        lo = np.asarray([r["ci_low"] for r in series], dtype=float)
        hi = np.asarray([r["ci_high"] for r in series], dtype=float)
        # Full 1 s lag grid so σ display smooth is visible (same as Figure 2 A/B).
        mask = f.display_lag_mask(lags, 1)
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
        is_alpha = band == "alpha"
        ax_b.fill_between(
            lags[mask],
            lo_d[mask],
            hi_d[mask],
            color=color,
            alpha=f.FIGURE1_PANEL_B_CI_ALPHA if is_alpha else max(
                0.06, f.FIGURE1_PANEL_B_CI_ALPHA - 0.04
            ),
            linewidth=0,
            zorder=3 if is_alpha else 2,
        )
        (line,) = ax_b.plot(
            lags[mask],
            mean_d[mask],
            color=color,
            lw=f.LINE_WIDTH + (0.7 if is_alpha else 0.0),
            ls=f._band_linestyle(band),
            label=_fig1_band_short(band),
            zorder=5 if is_alpha else 4,
        )
        plotted[band] = (line, _fig1_band_short(band))
    if any_series:
        legend_handles = [plotted[b][0] for b in f.BAND_ORDER if b in plotted]
        legend_labels = [plotted[b][1] for b in f.BAND_ORDER if b in plotted]
        f._shade_flanks(ax_b, EXPECTED_PRIMARY_DURATION_S)
        f._set_lag_axes(ax_b, EXPECTED_PRIMARY_DURATION_S)
        ax_b.set_ylabel("Fisher-z lag correlation", fontsize=f.FS_AXIS)
        # Stronger zero-lag reference (display only).
        ax_b.axvline(0.0, color=f.PALETTE["dark_gray"], lw=1.6, ls="-", zorder=1, alpha=0.9)
        f._annotate_lag_regions(ax_b, EXPECTED_PRIMARY_DURATION_S, enabled=True)
        ax_b.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=f.FS_LEGEND - 1,
            frameon=False,
            ncol=1,
            borderaxespad=0.2,
        )
        ax_b.text(
            0.98,
            0.02,
            f"N = {n_participants_note}",
            transform=ax_b.transAxes,
            ha="right",
            va="bottom",
            fontsize=f.FS_TICK - 1,
            color=f.PALETTE["dark_gray"],
        )
        # Sparse x ticks for manuscript scale.
        ax_b.set_xticks([-60, -40, -20, 0, 20, 40, 60])
        f._set_panel_title(ax_b, "Low-demand lag curves")
    else:
        f._mark_empty_panel(
            ax_b,
            f.MSG_NOT_INCLUDED,
            xlabel=f.LAG_XLABEL,
            ylabel="Fisher-z lag correlation",
            xlim=(
                float(primary_contract.lag_min_s),
                float(primary_contract.lag_max_s),
            ),
            ylim=(-0.2, 0.2),
        )
        f._set_panel_title(ax_b, "Low-demand lag curves")
    f._add_panel_label(ax_b, "B")
    panel_b_csv = source_dir / "figure1_panel_b_lag_curves_bootstrap_ci.csv"
    f.write_source_csv(
        panel_b_csv,
        series_all,
        (
            "lag_s",
            "band",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n",
            "n_observations",
            "mean_z",
            "ci_low",
            "ci_high",
            "ci_method",
            "n_bootstrap",
        ),
    )
    source_paths.append(panel_b_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="lag_curves_bootstrap",
            title="Low-demand lag curves",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("curves_d240") or "")],
            source_data_csv=str(panel_b_csv),
            analysis_keys=[
                f"duration={EXPECTED_PRIMARY_DURATION_S}",
                f"endpoint={ENDPOINT_ZLPI}",
                f"representation={f.PRIMARY_REPRESENTATION}",
                "ci=participant_within_dataset_bootstrap",
                f"display_smooth=gaussian_sigma_{f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g}s",
                "source_data=unsmoothed",
            ],
            notes=(
                f"{f.CI_95_METHOD_NOTE}. {f.FIGURE1_DISPLAY_GRID_DISCLOSURE} "
                f"Four-band overlay with lighter bootstrap ribbons "
                f"(α={f.FIGURE1_PANEL_B_CI_ALPHA}); small multiples not used. "
                f"{f.LAG_CURVE_DISPLAY_SMOOTH_NOTE}"
            ),
        )
    )

    # ----- Panel C: lag-category (group means ± SEM; faint participant dots) -----
    ax_c = fig.add_subplot(gs[1, 0])
    cat_rows = participant_lag_category_rows(endpoints)
    categories = ("z0", "shoulder_ref_z", "flank_z")
    category_labels = ("Lag 0", "Shoulders", "Distant flanks")
    group_summaries = _panel_c_group_summaries(cat_rows)
    if cat_rows:
        x_base = np.arange(1, 4, dtype=float)
        for bi, band in enumerate(f.BAND_ORDER):
            band_rows = [r for r in cat_rows if f._as_str(r["band"]) == band]
            if not band_rows:
                continue
            offset = (bi - 1.5) * 0.06
            color = f._band_color(band)
            for cat_idx, cat in enumerate(categories):
                vals = [
                    float(r[cat])
                    for r in band_rows
                    if math.isfinite(float(r[cat]))
                ]
                if not vals:
                    continue
                jitter = (np.random.default_rng(bi + cat_idx).random(len(vals)) - 0.5) * 0.05
                ax_c.scatter(
                    np.full(len(vals), x_base[cat_idx] + offset) + jitter,
                    vals,
                    color=color,
                    s=12,
                    alpha=0.16,
                    edgecolors="none",
                    zorder=2,
                )
            means = [
                float(np.mean([float(r[cat]) for r in band_rows if math.isfinite(float(r[cat]))]))
                for cat in categories
            ]
            sems = []
            for cat in categories:
                vals = np.asarray(
                    [float(r[cat]) for r in band_rows if math.isfinite(float(r[cat]))],
                    dtype=float,
                )
                if vals.size >= 2:
                    sems.append(float(np.std(vals, ddof=1) / math.sqrt(vals.size)))
                else:
                    sems.append(float("nan"))
            yerr = np.asarray(
                [
                    s if math.isfinite(s) else 0.0
                    for s in sems
                ],
                dtype=float,
            )
            ax_c.errorbar(
                x_base + offset,
                means,
                yerr=yerr,
                color=color,
                marker=f._band_marker(band),
                lw=f.LINE_WIDTH,
                markersize=f.MARKER_SIZE - 1,
                capsize=3,
                elinewidth=1.2,
                label=_fig1_band_short(band),
                zorder=4,
            )
        ax_c.set_xticks(x_base)
        ax_c.set_xticklabels(category_labels, fontsize=f.FS_TICK)
        ax_c.set_ylabel("Fisher-z lag correlation", fontsize=f.FS_AXIS)
        ax_c.legend(
            fontsize=f.FS_LEGEND - 1,
            frameon=False,
            loc="upper right",
            ncol=1,
        )
        f._style_axes(ax_c)
        f._set_panel_title(ax_c, "Lag-category structure")
    else:
        f._mark_empty_panel(
            ax_c,
            f.MSG_NOT_INCLUDED,
            xlabel="Lag category",
            ylabel="Fisher-z lag correlation",
        )
        f._set_panel_title(ax_c, "Lag-category structure")
    f._add_panel_label(ax_c, "C")
    panel_c_csv = source_dir / "figure1_panel_c_lag_categories.csv"
    f.write_source_csv(
        panel_c_csv,
        cat_rows,
        (
            "dataset_id",
            "subject_id",
            "band",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "z0",
            "shoulder_ref_z",
            "flank_z",
            "endpoint_index",
            "local_prominence",
            "n_observations",
        ),
    )
    summary_c_csv = source_dir / "figure1_panel_c_group_summaries.csv"
    f.write_source_csv(
        summary_c_csv,
        group_summaries,
        ("band", "category", "mean_z", "sem_z", "n_participants"),
    )
    source_paths.append(panel_c_csv)
    source_paths.append(summary_c_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="lag_categories",
            title="Participant lag-category comparisons",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("endpoints_d240") or "")],
            source_data_csv=str(panel_c_csv),
            analysis_keys=[
                "categories=z0,max_shoulder,combined_flank",
                "aggregation=participant_mean",
            ],
            notes=(
                "Shoulder reference = max(neg,pos) shoulder mean-z (matches local "
                "prominence); flank = combined_flank_mean_z (matches ZLPI). "
                "Panel shows group means ± participant SEM with faint participant dots."
            ),
        )
    )

    # ----- Panel D: heatmap -----
    ax_d = fig.add_subplot(gs[1, 1])
    cells = subject_level_mean_zlpi_cells(subjects)
    marks = surrogate_significance_marks(null_summary)
    for cell in cells:
        key = (f._as_str(cell["dataset_id"]), f._as_str(cell["band"]))
        cell["surrogate_significant"] = bool(marks.get(key, False))
        if not f._as_str(cell.get("dataset_role")):
            cell["dataset_role"] = _dataset_role(f._as_str(cell["dataset_id"]), protocol)
        if not f._as_str(cell.get("display_label")):
            cell["display_label"] = f._dataset_display(f._as_str(cell["dataset_id"]))

    primary_ds = sorted(
        {
            f._as_str(c["dataset_id"])
            for c in cells
            if f._as_str(c["dataset_role"]) == "primary"
        },
        key=str.casefold,
    )
    sens_ds = sorted(
        {
            f._as_str(c["dataset_id"])
            for c in cells
            if f._as_str(c["dataset_role"]) == "sensitivity"
        },
        key=str.casefold,
    )
    has_sensitivity_cells = bool(sens_ds)
    other_ds = sorted(
        {
            f._as_str(c["dataset_id"])
            for c in cells
            if f._as_str(c["dataset_role"]) not in {"primary", "sensitivity"}
        },
        key=str.casefold,
    )
    row_datasets = primary_ds + (["—"] if primary_ds and sens_ds else []) + sens_ds + other_ds
    if cells and row_datasets:
        value_map = {
            (f._as_str(c["dataset_id"]), f._as_str(c["band"])): float(c["mean_zlpi"])
            for c in cells
        }
        sig_map = {
            (f._as_str(c["dataset_id"]), f._as_str(c["band"])): bool(
                c["surrogate_significant"]
            )
            for c in cells
        }
        matrix = np.full((len(row_datasets), len(f.BAND_ORDER)), np.nan)
        for i, ds in enumerate(row_datasets):
            if ds == "—":
                continue
            for j, band in enumerate(f.BAND_ORDER):
                matrix[i, j] = value_map.get((ds, band), np.nan)
        finite = matrix[np.isfinite(matrix)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        im = ax_d.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        any_surrogate_mark = False
        for i, ds in enumerate(row_datasets):
            if ds == "—":
                ax_d.axhline(i, color="white", lw=3)
                continue
            for j, band in enumerate(f.BAND_ORDER):
                val = matrix[i, j]
                if not np.isfinite(val):
                    continue
                # Emphasize confirmatory alpha column with a light edge.
                if band == "alpha":
                    ax_d.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor=f.PALETTE["dark_gray"],
                            linewidth=1.4,
                            zorder=3,
                        )
                    )
                text_color = "white" if abs(val) > 0.55 * vmax else f.PALETTE["dark_gray"]
                cell_text = f"{val:+.3f}"
                if sig_map.get((ds, band)):
                    any_surrogate_mark = True
                    cell_text = f"{cell_text}*"
                ax_d.text(
                    j,
                    i,
                    cell_text,
                    ha="center",
                    va="center",
                    fontsize=f.FS_TICK - 1,
                    color=text_color,
                    fontweight="bold" if band == "alpha" else "normal",
                    zorder=4,
                )
        ax_d.set_xticks(range(len(f.BAND_ORDER)))
        xticks = []
        for band in f.BAND_ORDER:
            label = _fig1_band_short(band)
            if band == "alpha":
                label = f"{label}†"
            xticks.append(label)
        ax_d.set_xticklabels(xticks, fontsize=f.FS_TICK)
        ylabels = []
        for ds in row_datasets:
            if ds == "—":
                ylabels.append("")
            else:
                role = _dataset_role(ds, protocol)
                tag = role_tag(role)
                ylabels.append(f"{_fig1_dataset_id_label(ds)} [{tag}]")
        ax_d.set_yticks(range(len(row_datasets)))
        ax_d.set_yticklabels(ylabels, fontsize=f.FS_TICK)
        cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
        cbar.set_label("Mean ZLPI (Fisher z)", fontsize=f.FS_TICK)
        cbar.ax.tick_params(labelsize=f.FS_TICK - 1)
        f._set_panel_title(ax_d, "ZLPI by dataset and band")
    else:
        any_surrogate_mark = False
        f._mark_empty_panel(
            ax_d,
            f.MSG_NOT_INCLUDED,
            xlabel="EEG band",
            ylabel="Dataset",
        )
        f._set_panel_title(ax_d, "ZLPI by dataset and band")
    footnote_bits = ["† confirmatory band; others secondary"]
    if any_surrogate_mark:
        footnote_bits.append(f.FIGURE1_PANEL_D_ASTERISK_LABEL)
    ax_d.text(
        0.5,
        -0.18,
        " · ".join(footnote_bits),
        transform=ax_d.transAxes,
        ha="center",
        va="top",
        fontsize=f.FS_TICK - 2,
        color=f.PALETTE["dark_gray"],
        style="italic",
    )
    f._add_panel_label(ax_d, "D")
    panel_d_csv = source_dir / "figure1_panel_d_zlpi_heatmap.csv"
    f.write_source_csv(
        panel_d_csv,
        cells,
        (
            "dataset_id",
            "source_dataset_id",
            "session_id",
            "display_label",
            "dataset_role",
            "band",
            "mean_zlpi",
            "n_participants",
            "n_participant_sessions",
            "surrogate_significant",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "aggregation",
        ),
    )
    source_paths.append(panel_d_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="zlpi_heatmap",
            title="Dataset/session × band ZLPI heatmap",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("subject_level") or ""),
                str(inputs.get("null_summary") or ""),
                str(inputs.get("protocol_audit") or ""),
            ],
            source_data_csv=str(panel_d_csv),
            analysis_keys=[
                "value=subject_level_mean_endpoint_index",
                "hiit_display=combined_ph_ps_within_participant_mean",
                f"surrogate_rule=median_empirical_p<{f.FIGURE1_PANEL_D_SURROGATE_ALPHA}",
                f"null_type={PRIMARY_NULL_TYPE}",
                "hiit_surrogate=combined_ph_ps_low_demand_median_p",
            ],
            notes=f.FIGURE1_PANEL_D_NOTE,
        )
    )

    # ----- Panel E: alpha replication forest -----
    ax_e = fig.add_subplot(gs[2, 0])
    studies, sensitivity_studies, pooled = alpha_replication_forest_rows(
        subjects,
        protocol,
        low_demand_effect_rows=low_demand_alpha_effects,
        low_demand_meta_rows=low_demand_alpha_meta,
    )
    forest_export = build_alpha_forest_export(
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
    )
    # Display-only labels after source export (values unchanged).
    for study in studies:
        ds = f._as_str(study.get("dataset_id"))
        n_pairs = study.get("n_pairs")
        n_txt = (
            f"N = {int(n_pairs)}"
            if n_pairs not in {"", None} and str(n_pairs).strip() != ""
            else ""
        )
        study["display_label"] = (
            f"{_fig1_dataset_id_label(ds)} ({n_txt})" if n_txt else _fig1_dataset_id_label(ds)
        )
    panel_e_title = "Alpha ZLPI replication"
    drawn = draw_alpha_meta_forest(
        ax_e,
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
        dataset_display_fn=lambda dataset_id, n_pairs=None: _fig1_dataset_id_label(
            dataset_id
        ),
        band_color_fn=f._band_color,
        ref_vline_fn=f._ref_vline,
        style_axes_fn=f._style_axes,
        set_panel_title_fn=f._set_panel_title,
        panel_title=panel_e_title,
        xlabel=f"ZLPI (Fisher z; {f.CI_95_LABEL})",
        marker_size=f.MARKER_SIZE,
        line_width=f.LINE_WIDTH,
        tick_fontsize=f.FS_TICK,
        axis_fontsize=f.FS_AXIS,
        palette=f.PALETTE,
        pooled_section_label="",
        prediction_interval_lw=8.0,
        pooled_marker_size_delta=1.5,
        ytick_fontsize_delta=-1.0,
        header_fontsize_delta=-2.0,
    )
    if not drawn:
        f._mark_empty_panel(
            ax_e,
            _forest_unavailable_message(studies),
            xlabel=f"ZLPI (Fisher z; {f.CI_95_LABEL})",
            ylabel="Dataset",
        )
        f._set_panel_title(ax_e, panel_e_title)
    f._add_panel_label(ax_e, "E")
    panel_e_csv = source_dir / "figure1_panel_e_alpha_replication_forest.csv"
    f.write_source_csv(
        panel_e_csv,
        forest_export,
        FOREST_EXPORT_FIELDS,
    )
    source_paths.append(panel_e_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="alpha_replication_forest",
            title="Low-demand alpha ZLPI replication forest",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("low_demand_alpha_effects") or ""),
                str(inputs.get("low_demand_alpha_meta") or ""),
                str(inputs.get("protocol_audit") or ""),
                str(inputs.get("subject_level") or ""),
            ],
            source_data_csv=str(panel_e_csv),
            analysis_keys=[
                "band=alpha",
                "estimand=low_demand_mean_zlpi",
                "meta=random_effects_paule_mandel",
                "primary_datasets_only=true",
            ],
            notes=f.FIGURE1_PANEL_E_NOTE,
        )
    )

    # ----- Panel F: μ and FWHM -----
    gs_f = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1], wspace=0.35)
    ax_f_mu = fig.add_subplot(gs_f[0, 0])
    ax_f_fwhm = fig.add_subplot(gs_f[0, 1])
    peak_export: list[dict[str, object]] = []
    participant_peak_rows: list[dict[str, object]] = []
    eligible_units_by_cell: dict[tuple[str, str], set[str]] = {}
    for row in subjects:
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if "endpoint_eligible" in row and not f._as_bool(row.get("endpoint_eligible"), True):
            continue
        if not f._is_low_demand_condition(row):
            continue
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        participant = f._as_str(row.get("participant_id") or row.get("subject_id"))
        session = f._as_str(row.get("session_id"), "single").casefold() or "single"
        unit = (
            f"{participant}::{session}"
            if participant
            else f._as_str(row.get("observation_id"), "unknown")
        )
        eligible_units_by_cell.setdefault((dataset_id, band), set()).add(unit)
    for row in peaks:
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        if not f._as_bool(row.get("has_identifiable_peak")):
            continue
        if not f._is_low_demand_condition(row):
            # peak_fit_params may lack condition; keep if condition blank
            if f._as_str(row.get("condition") or row.get("task")):
                continue
        mu = f._as_float(row.get("peak_center_mu_s"))
        fwhm = f._as_float(row.get("fwhm_s"))
        band = f._as_str(row.get("band")).casefold()
        if band not in f.BAND_ORDER:
            continue
        if not math.isfinite(mu) and not math.isfinite(fwhm):
            continue
        participant_peak_rows.append(
            {
                "dataset_id": f._as_str(row.get("dataset_id")),
                "subject_id": f._as_str(
                    row.get("subject_id") or row.get("participant_id")
                ),
                "band": band,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "peak_center_mu_s": mu,
                "fwhm_s": fwhm,
                "has_identifiable_peak": True,
            }
        )
        peak_export.append(participant_peak_rows[-1])

    # μ TOST / hierarchical group rows.
    identifiable_units_by_cell: dict[tuple[str, str], set[str]] = {}
    for row in participant_peak_rows:
        key = (
            f._as_str(row.get("dataset_id")).casefold(),
            f._as_str(row.get("band")).casefold(),
        )
        identifiable_units_by_cell.setdefault(key, set()).add(
            f._as_str(row.get("subject_id"))
        )

    eq_export: list[dict[str, object]] = []
    for row in equivalence:
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            not in {"", f.PRIMARY_REPRESENTATION}
            and f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        band = f._as_str(row.get("band")).casefold()
        n_identifiable = len(identifiable_units_by_cell.get((dataset_id, band), set()))
        n_eligible = len(eligible_units_by_cell.get((dataset_id, band), set()))
        identifiability_rate = (
            float(n_identifiable / n_eligible) if n_eligible > 0 else float("nan")
        )
        status = "computed"
        reason = ""
        if (
            math.isfinite(identifiability_rate)
            and identifiability_rate < f.FIGURE1_PANEL_F_IDENTIFIABILITY_MIN
        ):
            status = "suppressed_low_identifiability"
            reason = (
                f"identifiability_rate={identifiability_rate:.3f}<"
                f"{f.FIGURE1_PANEL_F_IDENTIFIABILITY_MIN:.2f}"
            )
        eq_export.append(
            {
                "dataset_id": dataset_id,
                "band": band,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": f.PRIMARY_REPRESENTATION,
                "mean_mu": (
                    f._as_float(row.get("mean_mu"))
                    if status == "computed"
                    else float("nan")
                ),
                "ci_low": (
                    f._as_float(row.get("ci_low"))
                    if status == "computed"
                    else float("nan")
                ),
                "ci_high": (
                    f._as_float(row.get("ci_high"))
                    if status == "computed"
                    else float("nan")
                ),
                "equivalent": (
                    f._as_str(row.get("equivalent")) if status == "computed" else ""
                ),
                "tost_p": (
                    f._as_float(row.get("tost_p"))
                    if status == "computed"
                    else float("nan")
                ),
                "n_identifiable": n_identifiable,
                "n_eligible": n_eligible,
                "identifiability_rate": identifiability_rate,
                "status": status,
                "reason": reason,
            }
        )

    # Hierarchical FWHM summaries (back-transformed log-FWHM MixedLM).
    fwhm_hier: dict[str, dict[str, float]] = {}
    for row in peak_hier:
        if f._as_str(row.get("parameter")) != "fwhm":
            continue
        if f._as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if f._as_int(row.get("duration_s"), 240) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if (
            f._as_str(row.get("power_representation"), f.PRIMARY_REPRESENTATION).casefold()
            != f.PRIMARY_REPRESENTATION
        ):
            continue
        band = f._as_str(row.get("band")).casefold()
        fwhm_hier[band] = {
            "mean": f._as_float(row.get("mean")),
            "ci_low": f._as_float(row.get("ci_low")),
            "ci_high": f._as_float(row.get("ci_high")),
        }

    if participant_peak_rows or eq_export:
        ax_f_mu.axvspan(
            -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            facecolor=f.PALETTE["orange"],
            alpha=0.22,
            zorder=0,
            label=f.MU_EQUIVALENCE_LABEL,
        )
        # Explicit ±2 s bound markers.
        for bound in (
            -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
        ):
            ax_f_mu.axvline(
                bound,
                color=f.PALETTE["orange"],
                lw=1.4,
                ls="--",
                zorder=1,
                alpha=0.95,
            )
        for bi, band in enumerate(f.BAND_ORDER):
            mus = [
                float(r["peak_center_mu_s"])
                for r in participant_peak_rows
                if f._as_str(r["band"]) == band
                and math.isfinite(float(r["peak_center_mu_s"]))
            ]
            if mus:
                jitter = (np.random.default_rng(bi + 1).random(len(mus)) - 0.5) * 0.18
                ax_f_mu.scatter(
                    mus,
                    np.full(len(mus), bi) + jitter,
                    color=f._band_color(band),
                    s=22,
                    alpha=0.45,
                    marker=f._band_marker(band),
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.35,
                    zorder=2,
                )
            for idx, erow in enumerate(
                [e for e in eq_export if f._as_str(e["band"]) == band and f._as_str(e.get("status")) == "computed"]
            ):
                mean_mu = float(erow["mean_mu"])
                if not math.isfinite(mean_mu):
                    continue
                y_offset = bi + (idx - 0.5) * 0.18
                # Hierarchical MixedLM mean (diamond) + CI.
                ax_f_mu.scatter(
                    [mean_mu],
                    [y_offset],
                    color=f._band_color(band),
                    s=90,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
                lo = float(erow["ci_low"])
                hi = float(erow["ci_high"])
                xerr = None
                if math.isfinite(lo) and math.isfinite(hi):
                    xerr = [[mean_mu - lo], [hi - mean_mu]]
                ax_f_mu.errorbar(
                    mean_mu,
                    y_offset,
                    xerr=xerr,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=3,
                    zorder=4,
                )
        ax_f_mu.set_yticks(range(len(f.BAND_ORDER)))
        ax_f_mu.set_yticklabels([_fig1_band_short(b) for b in f.BAND_ORDER], fontsize=f.FS_TICK)
        ax_f_mu.set_xlabel("Peak center μ (s)", fontsize=f.FS_AXIS - 1)
        mu_lo, mu_hi = _mu_axis_limits(participant_peak_rows)
        ax_f_mu.set_xlim(mu_lo, mu_hi)
        f._style_axes(ax_f_mu)
        f._ref_vline(ax_f_mu, 0.0)
        ax_f_mu.text(
            0.98,
            0.02,
            "±2 s bounds",
            transform=ax_f_mu.transAxes,
            ha="right",
            va="bottom",
            fontsize=f.FS_TICK - 2,
            color=f.PALETTE["orange"],
        )
        for artist in ax_f_mu.get_children():
            if hasattr(artist, "set_clip_on"):
                artist.set_clip_on(True)

        for bi, band in enumerate(f.BAND_ORDER):
            fwhms = [
                float(r["fwhm_s"])
                for r in participant_peak_rows
                if f._as_str(r["band"]) == band and math.isfinite(float(r["fwhm_s"]))
            ]
            if not fwhms:
                continue
            jitter = (np.random.default_rng(bi + 11).random(len(fwhms)) - 0.5) * 0.18
            ax_f_fwhm.scatter(
                fwhms,
                np.full(len(fwhms), bi) + jitter,
                color=f._band_color(band),
                s=22,
                alpha=0.45,
                marker=f._band_marker(band),
                edgecolors=f.PALETTE["dark_gray"],
                linewidths=0.35,
                zorder=2,
            )
            hier = fwhm_hier.get(band)
            if hier and math.isfinite(float(hier["mean"])):
                mean_f = float(hier["mean"])
                ax_f_fwhm.scatter(
                    [mean_f],
                    [bi],
                    color=f._band_color(band),
                    s=90,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
                lo = float(hier["ci_low"])
                hi = float(hier["ci_high"])
                if math.isfinite(lo) and math.isfinite(hi):
                    ax_f_fwhm.errorbar(
                        mean_f,
                        bi,
                        xerr=[[mean_f - lo], [hi - mean_f]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.5,
                        capsize=3,
                        zorder=4,
                    )
            else:
                ax_f_fwhm.scatter(
                    [float(np.mean(fwhms))],
                    [bi],
                    color=f._band_color(band),
                    s=90,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
        ax_f_fwhm.set_yticks(range(len(f.BAND_ORDER)))
        ax_f_fwhm.set_yticklabels([_fig1_band_short(b) for b in f.BAND_ORDER], fontsize=f.FS_TICK)
        ax_f_fwhm.set_xlabel("FWHM (s)", fontsize=f.FS_AXIS - 1)
        fwhm_lo, fwhm_hi = _fwhm_axis_limits(participant_peak_rows)
        ax_f_fwhm.set_xlim(fwhm_lo, fwhm_hi)
        f._style_axes(ax_f_fwhm)
        # Identifiable vs eligible N for alpha (confirmatory) — below axis.
        alpha_rows = [
            e
            for e in eq_export
            if f._as_str(e.get("band")) == "alpha" and f._as_str(e.get("status")) == "computed"
        ]
        if alpha_rows:
            n_id = sum(int(e.get("n_identifiable") or 0) for e in alpha_rows)
            n_el = sum(int(e.get("n_eligible") or 0) for e in alpha_rows)
            ax_f_fwhm.text(
                0.5,
                -0.22,
                f"Alpha identifiable / eligible N = {n_id}/{n_el}",
                transform=ax_f_fwhm.transAxes,
                ha="center",
                va="top",
                fontsize=f.FS_TICK - 2,
                color=f.PALETTE["dark_gray"],
            )
    else:
        f._mark_empty_panel(
            ax_f_mu,
            f.MSG_NOT_INCLUDED,
            xlabel="Peak center μ (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
        f._mark_empty_panel(
            ax_f_fwhm,
            f.MSG_NOT_INCLUDED,
            xlabel="FWHM (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
    f._set_panel_title(ax_f_mu, "Peak center μ")
    f._set_panel_title(ax_f_fwhm, "FWHM")
    f._add_panel_label(ax_f_mu, "F")
    panel_f_peaks_csv = source_dir / "figure1_panel_f_participant_peaks.csv"
    f.write_source_csv(
        panel_f_peaks_csv,
        peak_export,
        (
            "dataset_id",
            "subject_id",
            "band",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "peak_center_mu_s",
            "fwhm_s",
            "has_identifiable_peak",
        ),
    )
    source_paths.append(panel_f_peaks_csv)
    computed_eq = [e for e in eq_export if f._as_str(e.get("status")) == "computed"]
    any_equivalent = any(f._as_bool(e.get("equivalent"), False) for e in computed_eq)
    suppressed_eq_rows = [r for r in eq_export if f._as_str(r.get("status")) != "computed"]
    f_notes: list[str] = []
    if computed_eq and not any_equivalent:
        f_notes.append("μ equivalence (±2 s) not established")
    if suppressed_eq_rows:
        summary = ", ".join(
            f"{_fig1_dataset_id_label(f._as_str(r.get('dataset_id')))} "
            f"{_fig1_band_short(f._as_str(r.get('band')))}"
            for r in suppressed_eq_rows
        )
        f_notes.append(
            f"suppressed (identifiability < {f.FIGURE1_PANEL_F_IDENTIFIABILITY_MIN:.2f}): {summary}"
        )
    if f_notes:
        ax_f_mu.text(
            0.02,
            -0.22,
            "; ".join(f_notes),
            transform=ax_f_mu.transAxes,
            ha="left",
            va="top",
            fontsize=f.FS_TICK - 2,
            color=f.PALETTE["dark_gray"],
            style="italic",
        )

    panel_f_eq_csv = source_dir / "figure1_panel_f_mu_tost.csv"
    f.write_source_csv(
        panel_f_eq_csv,
        eq_export,
        (
            "dataset_id",
            "band",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "mean_mu",
            "ci_low",
            "ci_high",
            "equivalent",
            "tost_p",
            "n_identifiable",
            "n_eligible",
            "identifiability_rate",
            "status",
            "reason",
        ),
    )
    source_paths.append(panel_f_eq_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure1",
            panel_id="peaks_mu_fwhm",
            title="Peak μ and FWHM",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("peak_params") or ""),
                str(inputs.get("peak_equivalence") or ""),
            ],
            source_data_csv=str(panel_f_peaks_csv),
            analysis_keys=[
                "peaks=identifiable_only",
                "mu_tost=existing_peak_center_equivalence",
                "fwhm=descriptive_only",
            ],
            notes=f.FIGURE1_PANEL_F_NOTE,
        )
    )

    fig.suptitle(f.FIGURE1_TITLE, fontsize=f.FS_SUPTITLE, fontweight="bold", y=0.982)
    footer_lines = _figure1_footer_lines()
    footer_text = "\n".join(footer_lines)
    fig.text(
        0.5,
        0.028,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=f.FS_TICK - 3,
        color=f.PALETTE["dark_gray"],
        linespacing=1.35,
        wrap=True,
    )

    caption_path = output_dir / "figure1_caption.txt"
    panel_d_role_clause = (
        "Primary vs sensitivity cohorts are visually separated. "
        if has_sensitivity_cells
        else "Primary cohorts only (ds003690, ds003838, ds006848). "
    )
    blocked_clause = (
        ""
        if not blocked_sensitivity_ids
        else (
            "Blocked sensitivity datasets were suppressed from manuscript-facing "
            "display rows in this render. "
        )
    )
    caption_path.write_text(
        (
            f"{f.FIGURE1_TITLE}\n\n"
            "A: Analysis schematic of the implemented confirmatory pipeline.\n"
            "B: Low-demand lag curves by band (Fisher-z lag correlation vs lag τ); "
            f"ribbons are participant-within-dataset percentile bootstrap 95% CIs. "
            f"{f.FIGURE1_DISPLAY_GRID_DISCLOSURE} {f.LAG_CURVE_DISPLAY_SMOOTH_NOTE}\n"
            "C: Lag-category structure (lag 0, shoulders, distant flanks); "
            "group means ± SEM with faint participant points.\n"
            "D: Dataset × band mean ZLPI (Fisher z); α is confirmatory (†). "
            f"{panel_d_role_clause}{blocked_clause}"
            "Cell values are subject-level means; surrogate marks appear only when "
            "the circular-shift rule is met.\n"
            "E: Alpha ZLPI replication forest across primary datasets with one pooled "
            "random-effects estimate and prediction interval. Low-demand alpha point "
            "estimates are positive; the pooled CI includes zero.\n"
            "F: Peak center μ (with ±2 s equivalence bounds) and descriptive FWHM; "
            "diamonds are hierarchical group means with 95% CIs. μ equivalence is "
            "not established. Identifiable peaks only; N shown as identifiable/eligible.\n"
        ),
        encoding="utf-8",
    )
    changelog_path = output_dir / "figure1_changelog.md"
    changelog_path.write_text(
        (
            "# Figure 1 changelog\n\n"
            "- Six-panel manuscript layout (A–F) with display-only visual polish.\n"
            "- Panel B CIs: participant-within-dataset percentile bootstrap "
            "(not parametric SE); four-band overlay retained with lighter ribbons "
            f"(α={f.FIGURE1_PANEL_B_CI_ALPHA}); confirmatory alpha emphasized.\n"
            f"- Panel B display-only Gaussian smooth "
            f"(σ = {f.LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g} s); source data unsmoothed.\n"
            "- Panel C: group means ± participant SEM; faint dots (no spaghetti lines).\n"
            "- Panel D values: subject-level mean ZLPI; "
            + (
                "sensitivity display rows are shown only when present and not runtime-blocked; "
                if has_sensitivity_cells
                else "no sensitivity display rows are present in this render; "
            )
            + "surrogate mark uses "
            + f"`median_empirical_p < {f.FIGURE1_PANEL_D_SURROGATE_ALPHA}` for "
            + f"`{PRIMARY_NULL_TYPE}` (circular-shift significance only).\n"
            "- Panel E: low-demand alpha ZLPI replication (D240 standard ZLPI, "
            "absolute log10) with one independent study per primary dataset and "
            "one pooled random-effects row (with prediction interval).\n"
            "- Panel F: Option C near-zero central peak (flank baseline + "
            "baseline-adjusted Gaussian on |τ|≤20); participant-nested MixedLM "
            "μ/FWHM (protocol session conditions as units) + hierarchical μ TOST; "
            "data-driven axis limits; FWHM hierarchical CI (log-scale fit).\n"
            "- Footer: compact lag convention and interpretation caveats.\n"
            "- No ECG–vs–PPG comparison; no max-|r| / argmax metrics.\n"
        ),
        encoding="utf-8",
    )

    pdf, svg, png = f.save_figure_trio(fig, output_dir, f.FIGURE1_STEM)
    return f.FigureArtifacts(
        figure_id="figure1",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(source_paths),
        panels=tuple(panel_sources),
    )


def _endpoint_label() -> str:
    f = _fig()
    return f._endpoint_display(ENDPOINT_ZLPI)


__all__ = [
    "alpha_replication_forest_rows",
    "bootstrap_mean_ci_by_lag",
    "participant_lag_category_rows",
    "render_figure1",
    "subject_level_mean_zlpi_cells",
    "surrogate_significance_marks",
]
