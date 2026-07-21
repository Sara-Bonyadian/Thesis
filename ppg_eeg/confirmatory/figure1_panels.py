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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .duration_contracts import (
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    contract_for_duration,
)
from .endpoints import fisher_z
from .forest_display import (
    FOREST_EXPORT_FIELDS,
    build_alpha_forest_export,
    draw_alpha_meta_forest,
    hiit_session_mean_zlpi_cells,
    hiit_session_sensitivity_forest_rows,
    hiit_session_surrogate_significance_marks,
    primary_meta_alpha_forest_rows,
)
from .manifest import FigurePanelSource
from .null_delta_inference import PRIMARY_NULL_TYPE
from .protocol_audit import PROTOCOL_SPECS

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

    low_labels = f.LOW_DEMAND_CONDITION_LABELS
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
            if role and role != "low_demand" and condition not in low_labels:
                continue
            if not role and condition and condition not in low_labels:
                if (
                    "rest" not in condition
                    and "passive" not in condition
                    and "low" not in condition
                ):
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
    f = _fig()
    key = f._as_str(dataset_id).casefold()
    for row in protocol_rows:
        if f._as_str(row.get("dataset_id")).casefold() != key:
            continue
        role = f._as_str(row.get("dataset_role")).casefold()
        if role in {"primary", "sensitivity"}:
            return role
    if key in {d.casefold() for d in f.FIGURE1_PRIMARY_DATASETS}:
        return "primary"
    if key in {d.casefold() for d in f.FIGURE1_SENSITIVITY_DATASETS}:
        return "sensitivity"
    return "other"


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
    spec = PROTOCOL_SPECS.get(key) or PROTOCOL_SPECS.get(f._as_str(dataset_id))
    if spec is not None:
        modality = f._as_str(spec.cardiac_modality)
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

    Non-HIIT datasets: one cell per dataset × band (participant-level means as
    stored in subject_level_metrics). HIIT is a single combined sensitivity row
    via within-participant averaging of available PH/PS × PRE/POST low-demand
    ZLPI (see ``hiit_session_mean_zlpi_cells``).
    """
    f = _fig()
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in subject_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        if dataset_id == "hiit":
            # Handled by hiit_session_mean_zlpi_cells (within-participant combine).
            continue
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
        key = (f._as_str(row.get("dataset_id")), band)
        buckets.setdefault(key, []).append(value)
    out: list[dict[str, object]] = [
        {
            "dataset_id": ds,
            "source_dataset_id": ds,
            "session_id": "",
            "display_label": "",
            "dataset_role": "",
            "band": band,
            "mean_zlpi": float(np.mean(vals)),
            "n_participants": len(vals),
            "n_participant_sessions": len(vals),
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "power_representation": f.PRIMARY_REPRESENTATION,
            "aggregation": "subject_level_endpoint_index",
        }
        for (ds, band), vals in sorted(buckets.items())
    ]
    out.extend(
        hiit_session_mean_zlpi_cells(
            subject_rows,
            band_order=f.BAND_ORDER,
            primary_representation=f.PRIMARY_REPRESENTATION,
        )
    )
    return out


def surrogate_significance_marks(
    null_summary_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], bool]:
    """Prespecified display rule: median_empirical_p < α for PRIMARY_NULL_TYPE.

    Non-HIIT: median of condition-level median_empirical_p within (dataset, band).
    HIIT: one combined mark per band from all low-demand PH/PS × PRE/POST
    condition-level p-values (see ``hiit_session_surrogate_significance_marks``).
    """
    f = _fig()
    by_cell: dict[tuple[str, str], list[float]] = {}
    for row in null_summary_rows:
        dataset_id = f._as_str(row.get("dataset_id")).casefold()
        if dataset_id == "hiit":
            continue
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
    marks.update(
        hiit_session_surrogate_significance_marks(
            null_summary_rows,
            band_order=f.BAND_ORDER,
            primary_representation=f.PRIMARY_REPRESENTATION,
            null_type=PRIMARY_NULL_TYPE,
            alpha=f.FIGURE1_PANEL_D_SURROGATE_ALPHA,
            low_demand_labels=set(f.LOW_DEMAND_CONDITION_LABELS),
        )
    )
    return marks


def alpha_replication_forest_rows(
    dataset_effects: Sequence[Mapping[str, object]],
    meta_rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object] | None]:
    """PRIMARY_META alpha rows + display-only combined HIIT sensitivity + pooled.

    Sensitivity rows never set ``enters_meta`` and are not used for pooling.
    """
    studies, pooled = primary_meta_alpha_forest_rows(
        dataset_effects,
        meta_rows,
        protocol_rows,
        cardiac_modality_fn=_cardiac_modality,
    )
    sensitivity = hiit_session_sensitivity_forest_rows(paired_rows or [])
    return studies, sensitivity, pooled


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
        f.CI_95_METHOD_NOTE,
        f.FIGURE1_PANEL_D_SURROGATE_RULE_NOTE,
        f.FIGURE1_PANEL_E_NOTE,
    )


def _draw_schematic(ax: plt.Axes) -> None:
    f = _fig()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.70, 0.22, 0.24, "ECG or PPG\n(dataset choice)\n→ instant. HR"),
        (0.28, 0.70, 0.22, 0.24, "EEG → multitaper\nθ/α/β/low-γ\n(channel median)"),
        (0.54, 0.70, 0.22, 0.24, "Common support\nlag-resolved r\n→ Fisher-z"),
        (0.80, 0.70, 0.18, 0.24, "Region means\nZLPI /\nprominence"),
        (0.28, 0.28, 0.22, 0.24, "Gaussian peak\nA, μ, FWHM"),
        (0.54, 0.28, 0.44, 0.24, "Confirmatory inference\n(surrogates, pairs,\nPRIMARY_META, μ TOST)"),
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
            fontsize=f.FS_TICK - 3,
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
        0.12,
        "Matches implemented confirmatory pipeline (not ECG–vs–PPG comparison).",
        ha="center",
        va="center",
        fontsize=f.FS_TICK - 4,
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
    effects = f.read_csv_rows(inputs.get("dataset_effects"))
    meta = f.read_csv_rows(inputs.get("meta_analysis"))
    peaks = f.read_csv_rows(inputs.get("peak_params"))
    equivalence = f.read_csv_rows(inputs.get("peak_equivalence"))
    protocol = f.read_csv_rows(inputs.get("protocol_audit"))
    paired = f.read_csv_rows(inputs.get("paired_contrasts"))

    fig = plt.figure(figsize=(17.2, 16.0))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.19,
        wspace=0.28,
        hspace=0.40,
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
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    any_series = False
    n_participants_note = 0
    for band in f.BAND_ORDER:
        series = bootstrap_mean_ci_by_lag(curves, band=band, condition_role="low_demand")
        series_all.extend(series)
        if not series:
            continue
        any_series = True
        n_participants_note = int(series[0]["n"])
        lags = np.asarray([r["lag_s"] for r in series], dtype=float)
        mean = np.asarray([r["mean_z"] for r in series], dtype=float)
        lo = np.asarray([r["ci_low"] for r in series], dtype=float)
        hi = np.asarray([r["ci_high"] for r in series], dtype=float)
        mask = f.display_lag_mask(lags, f.FIGURE1_DISPLAY_LAG_STEP_S)
        color = f._band_color(band)
        ax_b.fill_between(
            lags[mask],
            lo[mask],
            hi[mask],
            color=color,
            alpha=f.FIGURE1_PANEL_B_CI_ALPHA,
            linewidth=0,
            zorder=2,
        )
        (line,) = ax_b.plot(
            lags[mask],
            mean[mask],
            color=color,
            lw=f.LINE_WIDTH,
            ls=f._band_linestyle(band),
            label=f._band_display(band),
            zorder=3,
        )
        legend_handles.append(line)
        legend_labels.append(f._band_display(band))
    if any_series:
        f._shade_flanks(ax_b, EXPECTED_PRIMARY_DURATION_S)
        f._set_lag_axes(ax_b, EXPECTED_PRIMARY_DURATION_S)
        f._annotate_lag_regions(ax_b, EXPECTED_PRIMARY_DURATION_S, enabled=True)
        ax_b.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            fontsize=f.FS_LEGEND - 2,
            frameon=False,
        )
        f._set_panel_title(
            ax_b,
            f"Low-demand Fisher-z lag curves (n = {n_participants_note} participants)",
        )
    else:
        f._mark_empty_panel(
            ax_b,
            f.MSG_NOT_INCLUDED,
            xlabel=f.LAG_XLABEL,
            ylabel=f.Z_YLABEL,
            xlim=(
                float(primary_contract.lag_min_s),
                float(primary_contract.lag_max_s),
            ),
            ylim=(-0.2, 0.2),
        )
        f._set_panel_title(ax_b, "Low-demand Fisher-z lag curves")
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
            title="Low-demand Fisher-z lag curves",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("curves_d240") or "")],
            source_data_csv=str(panel_b_csv),
            analysis_keys=[
                f"duration={EXPECTED_PRIMARY_DURATION_S}",
                f"endpoint={ENDPOINT_ZLPI}",
                f"representation={f.PRIMARY_REPRESENTATION}",
                "ci=participant_within_dataset_bootstrap",
            ],
            notes=(
                f"{f.CI_95_METHOD_NOTE}. {f.FIGURE1_DISPLAY_GRID_DISCLOSURE} "
                f"Four-band overlay with lighter bootstrap ribbons "
                f"(α={f.FIGURE1_PANEL_B_CI_ALPHA}); small multiples not used."
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
                    s=14,
                    alpha=0.22,
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
                label=f._band_display(band),
                zorder=4,
            )
        ax_c.set_xticks(x_base)
        ax_c.set_xticklabels(category_labels)
        ax_c.set_ylabel(f.Z_YLABEL, fontsize=f.FS_AXIS)
        ax_c.legend(fontsize=f.FS_LEGEND - 2, frameon=False, loc="best")
        f._style_axes(ax_c)
        f._set_panel_title(ax_c, "Participant lag-category Fisher-z (means ± SEM)")
    else:
        f._mark_empty_panel(
            ax_c,
            f.MSG_NOT_INCLUDED,
            xlabel="Lag category",
            ylabel=f.Z_YLABEL,
        )
        f._set_panel_title(ax_c, "Participant lag-category Fisher-z")
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
        label_map = {
            f._as_str(c["dataset_id"]): f._as_str(c.get("display_label"))
            or f._dataset_display(f._as_str(c["dataset_id"]))
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
        for i, ds in enumerate(row_datasets):
            if ds == "—":
                ax_d.axhline(i, color="white", lw=3)
                continue
            for j, band in enumerate(f.BAND_ORDER):
                if sig_map.get((ds, band)):
                    ax_d.text(
                        j,
                        i,
                        "*",
                        ha="center",
                        va="center",
                        fontsize=f.FS_PANEL_TITLE,
                        color="black",
                        fontweight="bold",
                    )
        ax_d.set_xticks(range(len(f.BAND_ORDER)))
        ax_d.set_xticklabels([f._band_display(b) for b in f.BAND_ORDER])
        ylabels = []
        for ds in row_datasets:
            if ds == "—":
                ylabels.append("")
            else:
                role = _dataset_role(ds, protocol)
                tag = "P" if role == "primary" else ("S" if role == "sensitivity" else "?")
                base = label_map.get(ds) or f._dataset_display(ds)
                ylabels.append(f"{base} [{tag}]")
        ax_d.set_yticks(range(len(row_datasets)))
        ax_d.set_yticklabels(ylabels, fontsize=f.FS_TICK - 2)
        cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
        cbar.set_label("Mean ZLPI (Fisher z)", fontsize=f.FS_TICK - 1)
        f._set_panel_title(ax_d, "Dataset/session × band ZLPI (primary | sensitivity)")
    else:
        f._mark_empty_panel(
            ax_d,
            f.MSG_NOT_INCLUDED,
            xlabel="EEG band",
            ylabel="Dataset",
        )
        f._set_panel_title(ax_d, "Dataset/session × band ZLPI")
    ax_d.text(
        0.5,
        -0.20,
        f.FIGURE1_PANEL_D_ASTERISK_LABEL,
        transform=ax_d.transAxes,
        ha="center",
        va="top",
        fontsize=f.FS_TICK - 3,
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
        effects, meta, protocol, paired
    )
    forest_export = build_alpha_forest_export(
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
    )
    drawn = draw_alpha_meta_forest(
        ax_e,
        primary_studies=studies,
        sensitivity_studies=sensitivity_studies,
        pooled=pooled,
        dataset_display_fn=f._dataset_display,
        band_color_fn=f._band_color,
        ref_vline_fn=f._ref_vline,
        style_axes_fn=f._style_axes,
        set_panel_title_fn=f._set_panel_title,
        panel_title="Alpha replication (PRIMARY_META display)",
        xlabel=f"Δ {_endpoint_label()} ({f.ZLPI_METRIC}; {f.CI_95_LABEL})",
        marker_size=f.MARKER_SIZE,
        line_width=f.LINE_WIDTH,
        tick_fontsize=f.FS_TICK,
        axis_fontsize=f.FS_AXIS,
        palette=f.PALETTE,
    )
    if not drawn:
        f._mark_empty_panel(
            ax_e,
            "No PRIMARY_META alpha study effects in this run.",
            xlabel=f"Δ ZLPI ({f.CI_95_LABEL})",
            ylabel="Dataset",
        )
        f._set_panel_title(ax_e, "Alpha replication (PRIMARY_META display)")
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
            title="Alpha PRIMARY_META replication display",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("dataset_effects") or ""),
                str(inputs.get("meta_analysis") or ""),
                str(inputs.get("protocol_audit") or ""),
                str(inputs.get("paired_contrasts") or ""),
            ],
            source_data_csv=str(panel_e_csv),
            analysis_keys=[
                "band=alpha",
                "enters_meta=true",
                "display_only_band_filter=true",
                "hiit_sensitivity_display=combined_ph_ps_within_participant_mean",
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
                "peak_center_mu_s": mu,
                "fwhm_s": fwhm,
                "has_identifiable_peak": True,
            }
        )
        peak_export.append(participant_peak_rows[-1])

    # μ TOST group rows (existing table).
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

    if participant_peak_rows or eq_export:
        ax_f_mu.axvspan(
            -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            facecolor=f.PALETTE["orange"],
            alpha=0.25,
            zorder=0,
            label=f.MU_EQUIVALENCE_LABEL,
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
                    s=28,
                    alpha=0.55,
                    marker=f._band_marker(band),
                    edgecolors=f.PALETTE["dark_gray"],
                    linewidths=0.4,
                    zorder=2,
                )
                ax_f_mu.scatter(
                    [float(np.mean(mus))],
                    [bi],
                    color=f._band_color(band),
                    s=90,
                    marker="D",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
            for erow in eq_export:
                if f._as_str(erow["band"]) != band:
                    continue
                mean_mu = float(erow["mean_mu"])
                if not math.isfinite(mean_mu):
                    continue
                lo = float(erow["ci_low"])
                hi = float(erow["ci_high"])
                xerr = None
                if math.isfinite(lo) and math.isfinite(hi):
                    xerr = [[mean_mu - lo], [hi - mean_mu]]
                ax_f_mu.errorbar(
                    mean_mu,
                    bi + 0.28,
                    xerr=xerr,
                    fmt="s",
                    color=f.PALETTE["dark_gray"],
                    markersize=6,
                    capsize=3,
                    elinewidth=1.5,
                    zorder=4,
                )
        ax_f_mu.set_yticks(range(len(f.BAND_ORDER)))
        ax_f_mu.set_yticklabels([f._band_display(b) for b in f.BAND_ORDER])
        ax_f_mu.set_xlabel(f"Peak μ (s; {f.CI_95_LABEL})", fontsize=f.FS_AXIS - 2)
        mu_lo, mu_hi = _mu_axis_limits(participant_peak_rows)
        ax_f_mu.set_xlim(mu_lo, mu_hi)
        f._style_axes(ax_f_mu)
        f._ref_vline(ax_f_mu, 0.0)
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
                s=28,
                alpha=0.55,
                marker=f._band_marker(band),
                edgecolors=f.PALETTE["dark_gray"],
                linewidths=0.4,
                zorder=2,
            )
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
        ax_f_fwhm.set_yticklabels([f._band_display(b) for b in f.BAND_ORDER])
        ax_f_fwhm.set_xlabel("FWHM (s)", fontsize=f.FS_AXIS - 2)
        fwhm_lo, fwhm_hi = _fwhm_axis_limits(participant_peak_rows)
        ax_f_fwhm.set_xlim(fwhm_lo, fwhm_hi)
        f._style_axes(ax_f_fwhm)
    else:
        f._mark_empty_panel(
            ax_f_mu,
            f.MSG_NOT_INCLUDED,
            xlabel="Peak μ (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
        f._mark_empty_panel(
            ax_f_fwhm,
            f.MSG_NOT_INCLUDED,
            xlabel="FWHM (s)",
            ylabel=f.EEG_BAND_YLABEL,
        )
    f._set_panel_title(ax_f_mu, "Peak center μ")
    f._set_panel_title(ax_f_fwhm, "FWHM (descriptive)")
    f._add_panel_label(ax_f_mu, "F")
    panel_f_peaks_csv = source_dir / "figure1_panel_f_participant_peaks.csv"
    f.write_source_csv(
        panel_f_peaks_csv,
        peak_export,
        (
            "dataset_id",
            "subject_id",
            "band",
            "peak_center_mu_s",
            "fwhm_s",
            "has_identifiable_peak",
        ),
    )
    source_paths.append(panel_f_peaks_csv)
    panel_f_eq_csv = source_dir / "figure1_panel_f_mu_tost.csv"
    f.write_source_csv(
        panel_f_eq_csv,
        eq_export,
        (
            "dataset_id",
            "band",
            "mean_mu",
            "ci_low",
            "ci_high",
            "equivalent",
            "tost_p",
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

    fig.suptitle(f.FIGURE1_TITLE, fontsize=f.FS_SUPTITLE, fontweight="bold", y=0.985)
    footer_lines = _figure1_footer_lines()
    footer_text = "\n".join(footer_lines)
    fig.text(
        0.5,
        0.035,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=f.FS_TICK - 5,
        color=f.PALETTE["dark_gray"],
        linespacing=1.3,
        wrap=True,
    )

    caption_path = output_dir / "figure1_caption.txt"
    caption_path.write_text(
        (
            f"{f.FIGURE1_TITLE}\n\n"
            "A: Implemented analysis schematic.\n"
            f"B: Low-demand D240 Fisher-z lag curves by band (single overlay; "
            f"lighter bootstrap ribbons for full-dataset readability); "
            f"{f.CI_95_METHOD_NOTE}. {f.FIGURE1_DISPLAY_GRID_DISCLOSURE}\n"
            "C: Participant-level lag-category Fisher-z (lag 0 vs max shoulder vs "
            "combined distant flank); group means ± participant SEM with faint dots.\n"
            f"D: Dataset/session × band subject-level mean ZLPI; {f.FIGURE1_PANEL_D_ASTERISK_LABEL}. "
            "Primary vs sensitivity cohorts are visually separated. "
            f"{f.FIGURE1_PANEL_D_NOTE}\n"
            f"E: {f.FIGURE1_PANEL_E_NOTE} Includes study CIs, pooled RE CI, and "
            "prediction interval when present; cardiac modality is metadata only.\n"
            f"F: {f.FIGURE1_PANEL_F_NOTE} μ axis focuses on identifiable participant "
            "peaks and ±2 s equivalence (wide group TOST CIs may clip).\n"
        ),
        encoding="utf-8",
    )
    changelog_path = output_dir / "figure1_changelog.md"
    changelog_path.write_text(
        (
            "# Figure 1 changelog\n\n"
            "- Six-panel manuscript layout (A–F).\n"
            "- Panel B CIs: participant-within-dataset percentile bootstrap "
            "(not parametric SE); four-band overlay retained with lighter ribbons "
            f"(α={f.FIGURE1_PANEL_B_CI_ALPHA}) instead of small multiples.\n"
            "- Panel C: group means ± participant SEM; faint dots (no spaghetti lines).\n"
            "- Panel D values: subject-level mean ZLPI; HIIT shown as one combined "
            "sensitivity row (within-participant mean of available PH/PS × PRE/POST "
            "low-demand ZLPI); surrogate mark uses "
            f"`median_empirical_p < {f.FIGURE1_PANEL_D_SURROGATE_ALPHA}` for "
            f"`{PRIMARY_NULL_TYPE}` (circular-shift significance only; HIIT marks "
            "from combined low-demand condition p-values).\n"
            "- Panel E: alpha band display of equal four-band PRIMARY_META "
            "(not an alpha-only hierarchy); one combined HIIT display-only "
            "sensitivity row (within-participant mean of available contrasts) "
            "never enters RE pooling.\n"
            "- Panel F: participant μ/FWHM + existing μ TOST; data-driven axis "
            "limits on identifiable peaks; FWHM descriptive only.\n"
            "- Footer: multi-line below panels E/F to avoid overlap.\n"
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
