"""Publication figures for confirmatory zero-lag analyses (M12).

Reads only frozen confirmatory tables. No hard-coded scientific results.
Figures 1–3 are written as PDF, SVG, and 300-dpi PNG with companion
source-data CSVs and a figure-source manifest.
"""

from __future__ import annotations

import csv
import json
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .correlation import read_aligned_features_csv
from .config import EXPECTED_BANDS_HZ
from .duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_DURATIONS_S,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    EXPECTED_SHOULDERS_S,
    ZLPI_FLANKS_S,
    contract_for_duration,
)
from .endpoints import evaluate_endpoint_curve, fisher_z, min_common_support_required
from .manifest import (
    FigurePanelSource,
    build_figure_source_manifest,
    hash_directory_files,
    sha256_file,
)
from .null_delta_inference import (
    DATASET_DISPLAY_ORDER,
    INDEPENDENT_UNIT_VERDICT,
    PRIMARY_BAND,
    PRIMARY_DURATION_S,
    PRIMARY_NULL_TYPE,
    SECONDARY_NULL_TYPES,
    analyze_null_slice,
    analyze_null_slice_full,
    independent_unit_verdict_rows,
    leave_one_participant_out,
    secondary_band_null_fdr_table,
)
from .nulls import (
    DEFAULT_N_SURROGATES,
    NULL_SURROGATE_VALUES_FILENAME,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_PHASE_RANDOMIZATION,
    SeriesUnit,
    ar1_innovations,
    circular_shift_series,
    compute_endpoint_index_from_series,
    deterministic_seed,
    lag1_autocorrelation,
    seeded_derangement,
    series_units_from_aligned_rows,
    surrogate_effect_size,
    valid_circular_shifts,
)
from .paired_delta_inference import infer_paired_deltas_cluster_aware
from .panel_d_cardiac_controls import (
    CONTROL_BASELINE,
    CONTROL_BEAT_COUNT,
    CONTROL_ECG_CHANNELS,
    CONTROL_ICA_TEMPLATE,
    CONTROL_ORDER,
    CONTROL_RPEAK_MASK,
    PANEL_D_PLOT_CONTROL_ORDER,
    compute_panel_d_cardiac_controls,
    compute_panel_d_from_observation_controls,
    panel_d_short_display_label,
    short_not_computable_reason_code,
    verify_panel_d_summary_integrity,
    write_panel_d_cardiac_control_exports,
)
from .panel_e_nuisance_modality import (
    PANEL_E_STEM,
    compute_panel_e_nuisance_modality,
    render_panel_e_figure,
)
from .panel_f_topography_gamma import (
    PANEL_F_STEM,
    compute_panel_f_topography,
    render_panel_f_figure,
)
from .protocol_audit import condition_semantics_for
from .reason_codes import STRUCTURED_NC_FIELDS, with_structured_nc_fields

FIGURE_DPI = 300

# Colorblind-safe (Okabe–Ito–style) palette.
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "dark_gray": "#4D4D4D",
    "light_gray": "#D0D0D0",
    "flank": "#E8E8E8",
}

# Match confirmatory band set (θ/α/β/low-γ); presentation order only.
BAND_ORDER = tuple(EXPECTED_BANDS_HZ)
BAND_COLORS: dict[str, str] = {
    "delta": PALETTE["dark_gray"],
    "theta": PALETTE["blue"],
    "alpha": PALETTE["orange"],
    "beta": PALETTE["green"],
    "low_gamma": PALETTE["vermillion"],
}
# Marker + linestyle cues keep bands distinguishable in grayscale.
BAND_MARKERS: dict[str, str] = {
    "delta": "o",
    "theta": "o",
    "alpha": "s",
    "beta": "o",
    "low_gamma": "s",
}
BAND_LINESTYLES: dict[str, str | tuple] = {
    "delta": ":",
    "theta": "-",
    "alpha": "-",
    "beta": "-",
    "low_gamma":"-",
}
# Horizontal jitter within each duration column so bands do not overplot.
BAND_DURATION_X_OFFSET: dict[str, float] = {
    "delta": 0.0,
    "theta": -7.5,
    "alpha": -2.5,
    "beta": 2.5,
    "low_gamma": 7.5,
}
FIGURE3_DURATION_XLIM = (42.0, 258.0)
DATASET_DISPLAY: dict[str, str] = {
    "hiit": "HIIT",
    "hiit_ph": "HIIT PH",
    "hiit_ps": "HIIT PS",
}
ENDPOINT_DISPLAY: dict[str, str] = {
    ENDPOINT_ZLPI: "ZLPI",
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: "MWPI",
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: "SWPI",
}
PRIMARY_REPRESENTATION = "absolute_log10"
PANEL_A_REQUIRED_NULL_TYPES: tuple[str, ...] = (
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_PHASE_RANDOMIZATION,
    NULL_TYPE_BLOCK_SHUFFLE,
)

# Typography (pt).
FS_SUPTITLE = 22
FS_PANEL_TITLE = 20
FS_PANEL_LABEL = 22
FS_AXIS = 18
FS_TICK = 15
FS_LEGEND = 14
FS_EMPTY = 15
LINE_WIDTH = 2.4
AXIS_LINE_WIDTH = 1.5
CI_ALPHA = 0.25
MARKER_SIZE = 9.0
SCATTER_SIZE = 48.0
GRID_COLOR = PALETTE["light_gray"]
REF_LINE_COLOR = PALETTE["dark_gray"]

LAG_XLABEL = "Lag τ (s)"
LAG_CONVENTION_NOTE = (
    "Lag convention: corr(HR(t), EEG(t+τ)); "
    "negative τ = EEG leads HR; positive τ = HR leads EEG"
)
Z_YLABEL = "Fisher z"
CI_95_LABEL = "Pointwise 95% CI"
# Figure 1 Panel B: participant-within-dataset bootstrap (presentation only).
CI_95_METHOD_NOTE = (
    "Pointwise 95% CI = percentile bootstrap of mean Fisher-z "
    "(resample participants within each dataset; preserve all repeated "
    "observations of each drawn participant)"
)
ZLPI_METRIC = "Fisher z"
EEG_BAND_YLABEL = "EEG frequency band"
EQUIVALENCE_REGION_LABEL = f"±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S} s"
MEDIAN_PEAK_CENTER_LABEL = "Median fitted peak μ (≠ group-mean max)"
MU_CLARIFICATION_NOTE = (
    "Orange dotted line: median of participant-specific near-zero Gaussian peak "
    "centers (μ; flank baseline + central-window fit); this is not necessarily "
    "the location of the maximum of the group-average Fisher-z curve"
)
GROUP_MEAN_LABEL_TEMPLATE = "Group mean Fisher z (n = {n} participants)"
MU_EQUIVALENCE_LABEL = f"μ equivalence ({EQUIVALENCE_REGION_LABEL})"
MSG_NOT_INCLUDED = "Not included in the confirmatory analysis."
MSG_NOT_APPLICABLE = "Not applicable for this dataset"
LOW_DEMAND_CONDITION_LABELS = {"rest", "passive", "step1", "low_demand"}

# Master.yaml dataset_roles mirrored for figure cohort splitting when audit omits role.
FIGURE1_PRIMARY_DATASETS = frozenset(
    {"ds003838", "ds006848", "ds003690", "ds004587"}
)
FIGURE1_SENSITIVITY_DATASETS = frozenset(
    {"ds004582", "ds003816", "hiit", "mindfulness"}
)
FIGURE1_BOOTSTRAP_N = 2000
FIGURE1_BOOTSTRAP_SEED = 20260715
FIGURE1_BOOTSTRAP_CI_PERCENT = 95.0
# Panel B: lighter ribbons keep four-band overlay readable at full-dataset scale.
FIGURE1_PANEL_B_CI_ALPHA = 0.12
FIGURE1_PANEL_B_USE_SMALL_MULTIPLES = False
# Display-only Gaussian smooth for lag curves (Figure 1 Panel B; Figure 2 Panels A/B).
# Applied after bootstrap CIs are computed; source-data exports remain unsmoothed.
LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S = 2.0
LAG_CURVE_DISPLAY_SMOOTH_NOTE = (
    "For visualization only, the displayed curves were lightly smoothed using a "
    f"Gaussian kernel (σ = {LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S:g} s). "
    "All statistical analyses, hypothesis tests, and "
    "confidence intervals were computed from the original unsmoothed data."
)
# Back-compat aliases used by Figure 2 A/B.
FIGURE2_PANEL_AB_DISPLAY_SMOOTH_SIGMA_S = LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S
FIGURE2_PANEL_AB_DISPLAY_SMOOTH_NOTE = LAG_CURVE_DISPLAY_SMOOTH_NOTE
# Panel D: prespecified surrogate mark (display only; not a new inferential family).
FIGURE1_PANEL_D_SURROGATE_ALPHA = 0.05
FIGURE1_PANEL_D_SURROGATE_RULE_NOTE = (
    f"Surrogate mark (*): median_empirical_p < {FIGURE1_PANEL_D_SURROGATE_ALPHA:g} "
    f"for {PRIMARY_NULL_TYPE} on D{EXPECTED_PRIMARY_DURATION_S} "
    f"{PRIMARY_REPRESENTATION} ZLPI (circular-shift significance only; "
    "prespecified display α; not a new FDR family)."
)
FIGURE1_PANEL_D_NOTE = (
    FIGURE1_PANEL_D_SURROGATE_RULE_NOTE
    + " Sensitivity datasets may contribute one descriptive estimate. Protocol "
    "session conditions are analysis units (Panel B-aligned; counted separately). "
    "Within each session condition, available pre-/post-intervention low-demand "
    "ZLPI are averaged before the group mean. Sensitivity rows are summaries, "
    "not independent primary datasets. Surrogate marks use the median of all "
    "low-demand condition-level median_empirical_p values for that band within "
    "the sensitivity cohort."
)
FIGURE1_PANEL_D_ASTERISK_LABEL = (
    f"* = circular-shift surrogate p < {FIGURE1_PANEL_D_SURROGATE_ALPHA:g} only"
)
FIGURE1_PANEL_E_NOTE = (
    "Alpha-focused replication display of the equal four-band primary meta "
    "(one prespecified contrast per dataset); not an alpha-only confirmatory hierarchy. "
    "Sensitivity datasets may contribute one descriptive estimate. Protocol session "
    "conditions are analysis units (Panel B-aligned; counted separately). Within each "
    "session condition, available low-demand–high-demand ΔZLPI contrasts are averaged "
    "before group summarization. Sensitivity rows are excluded from the pooled "
    "random-effects meta-analysis."
)
FIGURE1_PANEL_F_NOTE = (
    "Near-zero central peak: linear baseline from distant flanks (20≤|τ|≤60 s), "
    "nonnegative Gaussian on baseline-adjusted |τ|≤20 s; identifiable peaks only "
    "(A ≥ 1.8×RMSE, SE(A), weak-edge). Group μ/FWHM: mean of per-session-condition "
    "means (Panel B-aligned; protocol session conditions counted as separate units) "
    "with hierarchical CI; μ TOST / ±2 s on that mean. FWHM on log scale "
    "(back-transformed CI)."
)

FIGURE1_TITLE = "Confirmatory EEG–cardiac coupling: structure, replication, and peaks"
FIGURE2_TITLE = (
    "State-dependent attenuation: lag structure, meta-replication, and peaks"
)
FIGURE3_TITLE = "Temporal specificity and core robustness"
FIGURE2_PANEL_C_NOTE = (
    "Alpha PRIMARY_META absolute paired ΔZLPI (high-demand − low-demand); "
    "no percent attenuation. Sensitivity datasets may contribute one descriptive "
    "estimate (protocol session conditions counted separately, Panel B-aligned) "
    "and are excluded from the pooled random-effects meta-analysis."
)
FIGURE2_PANEL_D_OUTCOME_LABEL = "Model-estimated Fisher-z ZLPI"
FIGURE2_PANEL_D_NOTE = (
    "Band×state interaction — model-estimated Fisher-z ZLPI (endpoint_index on "
    "the Fisher-z reporting scale; absolute log10 power representation) under low "
    "and high cognitive demand for theta, alpha, beta, and gamma; estimates and "
    "95% CIs from the fixed-effects covariance matrix with mean HR held at the "
    "sample mean; alpha-versus-other-band state-effect contrasts shown in-panel "
    "(negative contrast ⇒ stronger alpha attenuation under high demand)."
)
FIGURE2_PANEL_E_NOTE = (
    "State-specific Gaussian peak centers (μ) and widths (FWHM) for low-demand "
    "and high-demand states by dataset and frequency band. Estimates and "
    "participant-clustered 95% CIs are conditional on an identifiable peak in the "
    "corresponding state; low- and high-demand states may have different sample "
    "sizes. Sample-size labels (L / H) report identifiable observations / unique "
    "biological participants. Descriptive only — not a formal paired "
    "low-demand–high-demand test."
)
FIGURE2_PANEL_F_NOTE = (
    "Prespecified graded ds003690 contrasts (passive__simplert, "
    "passive__gonogo) from existing dataset_effects; not a formal "
    "dose-response or behavioral analysis."
)
FIGURE2_WIRING_GAP_NOTE = (
    "Wiring gap: C5 low/effort observation_ids could not be matched to C2 "
    "lag curves; unpaired fallback is not used."
)
PANEL_STATUS_EXPECTED_NOT_APPLICABLE = "expected_not_applicable"
PANEL_STATUS_SENSITIVITY_DISPLAY = "sensitivity_display"
FIGURE2_EXPECTED_NOT_APPLICABLE_NOTE = (
    "Expected not applicable: PRIMARY_META panels require primary-cohort "
    "meta inputs and are empty by design for sensitivity-only runs."
)
FIGURE2_PANEL_A_SENSITIVITY_NOTE = (
    "Sensitivity display (display-only): matched low-demand–high-demand Fisher-z "
    "lag curves from C5 pairs (pre- and post-intervention each contribute; no "
    "pre/post or protocol-session averaging before the group mean). Point "
    "estimate = mean across matched pairs; 95% CI = session-condition cluster "
    "bootstrap (pairs within a drawn session condition are retained together). "
    "Excluded from PRIMARY_META; not a primary confirmatory claim. Report "
    "n_matched_pairs and n_session_clusters (not participant n)."
)
FIGURE2_PANEL_B_SENSITIVITY_NOTE = (
    "Sensitivity display (display-only): matched low-demand–high-demand Δ Fisher-z "
    "lag curves Δz(τ)=z_high(τ)−z_low(τ) from the same C5 pairs as Panel A "
    "(pre- and post-intervention each contribute; no pre/post or protocol-session "
    "averaging before the group mean). Point estimate = mean across matched pairs; "
    "95% CI = session-condition cluster bootstrap. Excluded from PRIMARY_META; "
    "display only — no cluster-permutation testing. Report n_matched_pairs and "
    "n_session_clusters (not participant n)."
)
FIGURE2_PANEL_E_SENSITIVITY_NOTE = (
    "Sensitivity display (display-only): state-specific low-demand and high-demand "
    "Gaussian peak μ and FWHM from C5 matched pairs (pre-/post-intervention each "
    "contribute; no pre/post or protocol-session collapse). Suppress μ/FWHM "
    "independently when that state's peak is not identifiable. Summaries = mean "
    "of biological-participant means with participant-clustered percentile "
    "bootstrap 95% CIs. Sample-size labels (L / H) report identifiable "
    "observations / unique biological participants. Descriptive only; excluded "
    "from PRIMARY_META."
)
# Deprecated aliases retained for callers/tests that still import the HIIT names.
FIGURE2_PANEL_A_HIIT_SENSITIVITY_NOTE = FIGURE2_PANEL_A_SENSITIVITY_NOTE  # deprecated
FIGURE2_PANEL_B_HIIT_SENSITIVITY_NOTE = FIGURE2_PANEL_B_SENSITIVITY_NOTE  # deprecated
FIGURE2_PANEL_E_HIIT_SENSITIVITY_NOTE = FIGURE2_PANEL_E_SENSITIVITY_NOTE  # deprecated


def primary_meta_expected_na_message(detail: str) -> str:
    """On-figure notice for empty PRIMARY_META / sensitivity-only panels."""
    detail = detail.strip()
    return (
        f"Expected not applicable — {detail} "
        "PRIMARY_META panels are empty by design for sensitivity-only runs."
    )


def annotate_expected_not_applicable(notes: str, *, detail: str) -> str:
    """Append a panel_status annotation for empty sensitivity-only panels."""
    prefix = (
        f"Expected not applicable "
        f"(panel_status={PANEL_STATUS_EXPECTED_NOT_APPLICABLE}); {detail.strip()}"
    )
    base = notes.strip()
    return f"{prefix} {base}".strip() if base else prefix

FIGURE1_STEM = "figure1_lag_resolved_zero_lag"
FIGURE2_STEM = "figure2_state_attenuation_replication"
FIGURE3_STEM = "figure3_temporal_artifact_specificity"
FIGURE3_SUPPLEMENT_STEM = "figure3_supplement_null_diagnostics"
FIGURE3_SUPPLEMENT_LABEL = "Descriptive nested-observation diagnostic"
# Dataset-specific participant forests are internal QC only (not manuscript/supplement).
FIGURE3_QC_PARTICIPANT_FOREST_STEM = "figure3_qc_participant_null_forests"
FIGURE3_PARTICIPANT_FOREST_STEM = FIGURE3_QC_PARTICIPANT_FOREST_STEM  # compat alias
FIGURE3_INTERNAL_QC_SUBDIR = "internal_qc"
FIGURE3_PARTICIPANT_FOREST_MAX_ROWS = 24
FIGURE_EXPORT_CATEGORIES_FILENAME = "figure_export_categories.csv"
EXPORT_CATEGORY_MANUSCRIPT = "manuscript"
EXPORT_CATEGORY_SUPPLEMENTARY = "supplementary"
EXPORT_CATEGORY_INTERNAL_QC = "internal_qc"
FIGURE3_PANEL_B_ENCODING_NOTE = (
    "Color = EEG frequency band · Marker shape = endpoint index "
    "(ZLPI at 240/180 s; MWPI at 120 s; SWPI at 60 s)"
)
FIGURE3_PANEL_B_FOOTNOTE_SHORT = (
    "B: color = band · shape = index (ZLPI 240/180; MWPI 120; SWPI 60)"
)
FIGURE3_FIGSIZE = (21.2, 15.4)
FIGURE3_SUBPLOT_ADJUST = {
    "left": 0.075,
    "right": 0.988,
    "top": 0.925,
    "bottom": 0.108,
    "wspace": 0.32,
    "hspace": 0.40,
}
# Mild right bias so Panel D has room without starving A/C.
FIGURE3_GRID_WIDTH_RATIOS = (1.0, 1.32)
FIGURE3_GRID_HEIGHT_RATIOS = (1.05, 1.0)
# Nested Panel D: compact labels/status, widened coefficient and Δ axes.
# No sharey between columns — sharey reserves empty left gutters on plot axes.
FIGURE3_PANEL_D_WIDTH_RATIOS = (0.95, 1.78, 1.58, 1.18)
FIGURE3_PANEL_D_WSPACE = 0.10
FIGURE3_PANEL_D_ROW_LABELS = {
    "beat_count_adjusted": "Beat-count\nadjusted",
    "ppg_pulse_locked_template_subtraction": "PPG template\nsubtraction",
    "ppg_systolic_peak_mask": "PPG event mask",
    "ecg_cardiac_template_subtraction": "ECG template\nsubtraction",
    "ecg_prone_channels_removed": "ECG-prone\nchannels rem.",
}
FIGURE3_PANEL_D_STATUS_HEADER = "Computable / baseline-eligible\nobservation-band rows"
FIGURE3_PANEL_D_TITLE = "Cardiac-field and pulse-synchronous controls"
FIGURE3_PANEL_D_SUBTITLE = "Controlled estimates and paired changes from baseline"
PANEL_A_XLABEL = "Standardized ZLPI\nrelative to observation-specific null"
PANEL_A_TITLE = "Observed theta ZLPI relative to autocorrelation-preserving nulls"
FIGURE3_SUPPLEMENT_FIGSIZE = (14.5, 13.8)
FIGURE3_SUPPLEMENT_SUBPLOT_ADJUST = {
    "left": 0.15,
    "right": 0.97,
    "top": 0.915,
    "bottom": 0.085,
    "wspace": 0.34,
    "hspace": 0.36,
}
FIGURE3_CI_LINEWIDTH = 1.6
FIGURE3_MARKER_SIZE = 7.5
FIGURE3_SCATTER_SIZE = 36.0
FIGURE3_REF_LINEWIDTH = 1.0
# Display-only: analysis remains on the 1-s lag grid; plotting uses every Nth lag.
FIGURE1_DISPLAY_LAG_STEP_S = 2
FIGURE1_DISPLAY_GRID_DISCLOSURE = (
    "Correlations and all inferential analyses used the predefined 1-s lag grid; "
    "for visual clarity, the displayed mean curve and pointwise confidence band "
    "show every second lag value."
)

_STYLE_CONFIGURED = False


def _configure_publication_style() -> None:
    """Apply journal-style matplotlib defaults once per process."""
    global _STYLE_CONFIGURED
    if _STYLE_CONFIGURED:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "Source Sans Pro",
                "Liberation Sans",
                "sans-serif",
            ],
            "axes.titlesize": FS_PANEL_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FS_AXIS,
            "axes.labelpad": 12,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["dark_gray"],
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "legend.fontsize": FS_LEGEND,
            "legend.frameon": False,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "figure.titlesize": FS_SUPTITLE,
            "figure.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            # Keep SVG text editable (not converted to paths).
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    _STYLE_CONFIGURED = True


def _style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.minorticks_off()
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, width=1.2, length=4.5, pad=6)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(AXIS_LINE_WIDTH)
        ax.spines[spine].set_color(PALETTE["dark_gray"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if grid:
        ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.6, alpha=0.75, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _add_panel_label(ax: plt.Axes, letter: str) -> None:
    """Place a bold panel letter clearly left of the title (no overlap)."""
    # Multi-character supplement labels (S1–S5) need a slightly larger left offset.
    x_off = -40 if len(letter) > 1 else -32
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_off, 10),
        textcoords="offset points",
        fontsize=FS_PANEL_LABEL,
        fontweight="bold",
        fontfamily="sans-serif",
        va="bottom",
        ha="left",
        color=PALETTE["dark_gray"],
        clip_on=False,
        annotation_clip=False,
        zorder=20,
    )


def _set_panel_title(ax: plt.Axes, title: str, *, fontsize: float | None = None, pad: float = 12) -> None:
    # Left-aligned title with extra pad so it clears the panel letter.
    ax.set_title(
        title,
        fontsize=FS_PANEL_TITLE if fontsize is None else fontsize,
        fontweight="normal",
        pad=pad,
        loc="left",
    )


def _legend_inside(
    ax: plt.Axes,
    handles: Sequence[object] | None = None,
    labels: Sequence[str] | None = None,
    *,
    loc: str = "upper right",
    ncol: int = 1,
) -> None:
    kwargs = {
        "fontsize": FS_LEGEND - 1,
        "frameon": True,
        "fancybox": False,
        "edgecolor": PALETTE["light_gray"],
        "framealpha": 0.92,
        "loc": loc,
        "borderaxespad": 0.6,
        "handlelength": 1.8,
        "ncol": ncol,
        "columnspacing": 1.0,
        "labelspacing": 0.4,
    }
    if handles is not None and labels is not None:
        ax.legend(handles, labels, **kwargs)
    else:
        ax.legend(**kwargs)


def _band_duration_x(duration_s: int, band: str) -> float:
    """Nominal duration tick plus band-specific horizontal offset."""
    return float(duration_s) + BAND_DURATION_X_OFFSET.get(_as_str(band).casefold(), 0.0)


def _participant_forest_labels(rows: Sequence[object]) -> list[str]:
    """Compact participant labels; drop redundant dataset prefix when uniform."""
    datasets = {_as_str(getattr(r, "dataset_id", "")) for r in rows}  # type: ignore[arg-type]
    single_dataset = len({d for d in datasets if d}) == 1
    labels: list[str] = []
    for row in rows:
        label = _as_str(getattr(row, "display_label", ""))  # type: ignore[arg-type]
        dataset = _as_str(getattr(row, "dataset_id", ""))  # type: ignore[arg-type]
        if single_dataset and label:
            labels.append(label)
        elif dataset and label:
            labels.append(f"{dataset}:{label}")
        else:
            labels.append(label or dataset or "participant")
    return labels


def _clipped_vertical_errorbar(
    ax: plt.Axes,
    x: float,
    y: float,
    lo: float,
    hi: float,
    *,
    y_lo: float,
    y_hi: float,
    **kwargs: object,
) -> None:
    """Plot error bars, truncating at axis limits with limit indicators."""
    low_err = y - lo
    high_err = hi - y
    lolims = uplims = False
    if math.isfinite(lo) and lo < y_lo:
        low_err = max(0.0, y - y_lo)
        lolims = True
    if math.isfinite(hi) and hi > y_hi:
        high_err = max(0.0, y_hi - y)
        uplims = True
    if not (math.isfinite(low_err) and math.isfinite(high_err)):
        ax.scatter([x], [y], zorder=3, **{k: v for k, v in kwargs.items() if k != "yerr"})
        return
    ax.errorbar(
        x,
        y,
        yerr=[[low_err], [high_err]],
        lolims=lolims,
        uplims=uplims,
        zorder=3,
        **kwargs,
    )


def _legend_dual_encoding(
    ax: plt.Axes,
    band_handles: Sequence[object],
    band_labels: Sequence[str],
    index_handles: Sequence[object],
    index_labels: Sequence[str],
    *,
    loc: str = "upper right",
) -> None:
    """Two stacked in-panel legends: band color and endpoint marker shape."""
    legend_kw = {
        "fontsize": FS_LEGEND - 2,
        "frameon": True,
        "fancybox": False,
        "edgecolor": PALETTE["light_gray"],
        "framealpha": 0.92,
        "borderaxespad": 0.3,
        "handlelength": 1.3,
        "labelspacing": 0.22,
        "columnspacing": 0.7,
        "handletextpad": 0.35,
        "title_fontsize": FS_LEGEND - 1,
    }
    band_legend = ax.legend(
        band_handles,
        band_labels,
        title="Band",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        **legend_kw,
    )
    ax.add_artist(band_legend)
    ax.legend(
        index_handles,
        index_labels,
        title="Index",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        ncol=1,
        **legend_kw,
    )


def _legend_outside(
    ax: plt.Axes,
    handles: Sequence[object] | None = None,
    labels: Sequence[str] | None = None,
    *,
    ncol: int = 1,
) -> None:
    # Prefer inside legends for multipanel figures to avoid collisions.
    _legend_inside(ax, handles, labels, loc="best", ncol=ncol)


def _ref_hline(ax: plt.Axes, y: float = 0.0) -> None:
    ax.axhline(y, color=REF_LINE_COLOR, lw=FIGURE3_REF_LINEWIDTH, ls="--", zorder=1)


def _ref_vline(ax: plt.Axes, x: float = 0.0) -> None:
    ax.axvline(x, color=REF_LINE_COLOR, lw=FIGURE3_REF_LINEWIDTH, ls="--", zorder=1)


def _panel_d_summary_lookup(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    """Index summaries by control, including fallback alias IDs onto plot rows."""
    by_control: dict[str, Mapping[str, object]] = {
        _as_str(row.get("control")): row for row in summaries if _as_str(row.get("control"))
    }
    # Fallback / modality-specific IDs can occupy the locked display slots when the
    # preferred upstream control is absent or not computable.
    alias_pairs = (
        ("cardiac_template_subtraction", "ppg_pulse_locked_template_subtraction"),
        ("cardiac_event_masking", "ppg_systolic_peak_mask"),
        ("ecg_r_peak_mask", "ppg_systolic_peak_mask"),
    )
    for src, dst in alias_pairs:
        if src not in by_control:
            continue
        src_row = by_control[src]
        dst_row = by_control.get(dst)
        src_ok = _as_str(src_row.get("computability_status")).casefold() == "computed"
        dst_ok = (
            dst_row is not None
            and _as_str(dst_row.get("computability_status")).casefold() == "computed"
        )
        if src_ok and not dst_ok:
            by_control[dst] = src_row
        elif dst not in by_control:
            by_control[dst] = src_row
    return by_control


def _panel_d_figure_reason(reason: str) -> str:
    """Figure-only short reason; full text stays in caption/metadata."""
    code = short_not_computable_reason_code(reason)
    if code == "channel reaggregation unavailable":
        return "reagg. unavailable"
    if code == "1 Hz mask incompatible":
        return "1 Hz mask incompatible"
    if code == "insufficient support":
        return "insufficient support"
    return code


def _panel_d_row_label(control: str) -> str:
    return FIGURE3_PANEL_D_ROW_LABELS.get(control, panel_d_short_display_label(control))


def _render_figure3_panel_d_paired_axes(
    fig: plt.Figure,
    subplot_spec,
    *,
    summaries: Sequence[Mapping[str, object]],
) -> tuple[plt.Axes, plt.Axes, plt.Axes, list[str]]:
    """Draw Panel D as aligned coefficient, Δ, and status columns."""
    from matplotlib.lines import Line2D

    by_control = _panel_d_summary_lookup(summaries)
    baseline_row = by_control.get(CONTROL_BASELINE, {})
    baseline_level = _as_float(baseline_row.get("estimate"))
    plot_controls = list(PANEL_D_PLOT_CONTROL_ORDER)
    integrity = verify_panel_d_summary_integrity(
        [
            by_control[c]
            for c in ([CONTROL_BASELINE] + plot_controls)
            if c in by_control
        ]
    )
    if integrity:
        raise ValueError("Panel D visualization integrity failed: " + "; ".join(integrity))

    # Row 0 = plot body; row 1 = legend / NC note (keeps them off the axes).
    gs_d = subplot_spec.subgridspec(
        2,
        4,
        height_ratios=[1.0, 0.20],
        width_ratios=list(FIGURE3_PANEL_D_WIDTH_RATIOS),
        wspace=FIGURE3_PANEL_D_WSPACE,
        hspace=0.18,
    )
    # Do not sharey: shared y-axes reserve invisible tick gutters between columns.
    ax_lab = fig.add_subplot(gs_d[0, 0])
    ax_coef = fig.add_subplot(gs_d[0, 1])
    ax_delta = fig.add_subplot(gs_d[0, 2])
    ax_status = fig.add_subplot(gs_d[0, 3])
    ax_leg = fig.add_subplot(gs_d[1, 1:3])
    ax_note = fig.add_subplot(gs_d[1, 3])
    ax_leg.set_axis_off()
    ax_note.set_axis_off()

    n_rows = len(plot_controls)
    # Uneven y with an explicit gap between statistical and signal-level groups.
    y_positions = np.asarray([5.05, 3.20, 2.10, 1.00, -0.10], dtype=float)
    if n_rows != len(y_positions):
        y_positions = np.arange(n_rows, dtype=float)[::-1]
    coef_color = PALETTE["dark_gray"]
    delta_color = PALETTE["blue"]
    baseline_marker_color = PALETTE["orange"]
    nc_color = "#6E6E6E"
    group_color = "#5E5E5E"
    zero_color = "#7A7A7A"
    status_x = 0.0
    guide_color = "#F0F0F0"

    x_coef_vals: list[float] = []
    x_delta_vals: list[float] = []
    for yi, control in zip(y_positions, plot_controls, strict=True):
        row = by_control.get(control, {})
        status = _as_str(row.get("computability_status")).casefold()
        estimate = _as_float(row.get("estimate"))
        lo = _as_float(row.get("ci_lower"))
        hi = _as_float(row.get("ci_upper"))
        change = _as_float(row.get("change_from_baseline"))
        clo = _as_float(row.get("change_ci_lower"))
        chi = _as_float(row.get("change_ci_upper"))
        denom = _as_str(row.get("denominator_label")) or ""

        if status == "computed" and math.isfinite(estimate):
            if math.isfinite(baseline_level):
                ax_coef.plot(
                    [baseline_level, estimate],
                    [yi, yi],
                    color="#B8B8B8",
                    linewidth=1.15,
                    zorder=2,
                    solid_capstyle="round",
                )
                ax_coef.scatter(
                    [baseline_level],
                    [yi],
                    marker="D",
                    s=36,
                    color=baseline_marker_color,
                    edgecolors=coef_color,
                    linewidths=0.55,
                    zorder=3,
                )
            xerr = None
            if math.isfinite(lo) and math.isfinite(hi):
                xerr = [[max(0.0, estimate - lo)], [max(0.0, hi - estimate)]]
                x_coef_vals.extend([lo, hi, estimate])
            else:
                x_coef_vals.append(estimate)
            ax_coef.errorbar(
                estimate,
                yi,
                xerr=xerr,
                fmt="o",
                color=coef_color,
                markersize=FIGURE3_MARKER_SIZE,
                capsize=2.5,
                elinewidth=FIGURE3_CI_LINEWIDTH,
                markeredgecolor=coef_color,
                markeredgewidth=0.45,
                zorder=4,
            )
            if math.isfinite(change):
                dxerr = None
                if math.isfinite(clo) and math.isfinite(chi):
                    dxerr = [[max(0.0, change - clo)], [max(0.0, chi - change)]]
                    x_delta_vals.extend([clo, chi, change])
                else:
                    x_delta_vals.append(change)
                ax_delta.errorbar(
                    change,
                    yi,
                    xerr=dxerr,
                    fmt="o",
                    color=delta_color,
                    markersize=FIGURE3_MARKER_SIZE,
                    capsize=2.5,
                    elinewidth=FIGURE3_CI_LINEWIDTH,
                    markeredgecolor=delta_color,
                    markeredgewidth=0.45,
                    zorder=4,
                )
            ax_status.text(
                status_x,
                yi,
                f"Computable  {denom}",
                ha="left",
                va="center",
                fontsize=FS_TICK - 3,
                color=coef_color,
                clip_on=False,
            )
        else:
            reason = _panel_d_figure_reason(
                _as_str(row.get("short_reason_code")) or _as_str(row.get("computability_reason"))
            )
            ax_status.text(
                status_x,
                yi + (0.16 if reason else 0.0),
                f"NC  {denom or '0'}",
                ha="left",
                va="center",
                fontsize=FS_TICK - 3,
                color=nc_color,
                fontweight="bold",
                clip_on=False,
            )
            if reason:
                ax_status.text(
                    status_x,
                    yi - 0.22,
                    reason,
                    ha="left",
                    va="center",
                    fontsize=FS_TICK - 4,
                    color=nc_color,
                    clip_on=False,
                )

    if math.isfinite(baseline_level):
        ax_coef.axvline(
            baseline_level,
            color=baseline_marker_color,
            linestyle=":",
            linewidth=1.55,
            alpha=0.98,
            zorder=1,
        )
    for ax in (ax_coef, ax_delta):
        ax.axvline(0.0, color=zero_color, lw=1.25, ls="--", zorder=1)

    y_labels = [_panel_d_row_label(c) for c in plot_controls]
    y_lo, y_hi = -0.55, 5.95
    for ax in (ax_lab, ax_coef, ax_delta, ax_status):
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(list(y_positions))
        ax.tick_params(axis="y", which="both", left=False, labelleft=False, length=0)
        ax.set_yticklabels([])

    # Flush labels to the right edge of the label column (avoids empty gutter from
    # external yticklabels hanging left of the axes box).
    for yi, lab in zip(y_positions, y_labels, strict=True):
        ax_lab.text(
            1.0,
            yi,
            lab,
            ha="right",
            va="center",
            fontsize=FS_TICK - 1,
            linespacing=1.05,
            color=PALETTE["dark_gray"],
            clip_on=False,
            transform=ax_lab.get_yaxis_transform(),
        )
    ax_lab.tick_params(axis="x", bottom=False, labelbottom=False, length=0)
    ax_lab.set_xlim(0.0, 1.0)
    for spine in ("top", "right", "bottom", "left"):
        ax_lab.spines[spine].set_visible(False)
    ax_lab.patch.set_alpha(0.0)
    ax_lab.grid(False)
    ax_status.set_xticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax_status.spines[spine].set_visible(False)
    ax_status.set_xlim(0.0, 1.0)
    ax_status.patch.set_alpha(0.0)
    ax_status.grid(False)

    for i, line in enumerate(FIGURE3_PANEL_D_STATUS_HEADER.split("\n")):
        ax_status.text(
            status_x,
            1.048 - 0.038 * i,
            line,
            transform=ax_status.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_TICK - 4,
            color=coef_color,
            clip_on=False,
        )

    if n_rows >= 2:
        sep_y = 4.05
        for ax in (ax_coef, ax_delta):
            ax.axhline(sep_y, color="#DDDDDD", linewidth=0.85, zorder=0)
        ax_lab.text(
            1.0,
            float(y_positions[0]) + 0.62,
            "STATISTICAL SENSITIVITY",
            ha="right",
            va="bottom",
            fontsize=FS_TICK - 4,
            fontweight="bold",
            color=group_color,
            clip_on=False,
        )
        ax_lab.text(
            1.0,
            float(y_positions[1]) + 0.62,
            "SIGNAL-LEVEL CONTROLS",
            ha="right",
            va="bottom",
            fontsize=FS_TICK - 4,
            fontweight="bold",
            color=group_color,
            clip_on=False,
        )

    def _coef_limits(values: list[float]) -> tuple[float, float]:
        finite = [v for v in values if math.isfinite(v)]
        finite.append(0.0)
        if math.isfinite(baseline_level):
            finite.append(baseline_level)
        if not finite:
            return (-0.05, 0.05)
        lo_v = min(finite)
        hi_v = max(finite)
        span = hi_v - lo_v if hi_v > lo_v else max(abs(hi_v), 0.05)
        pad = 0.14 * span
        return lo_v - pad, hi_v + pad

    def _delta_limits(values: list[float]) -> tuple[float, float]:
        finite = [abs(v) for v in values if math.isfinite(v)]
        finite.append(0.0)
        m = max(finite) if finite else 0.05
        pad = max(0.14 * m, 0.002)
        return -(m + pad), (m + pad)

    ax_coef.set_xlim(*_coef_limits(x_coef_vals))
    ax_delta.set_xlim(*_delta_limits(x_delta_vals))
    ax_coef.set_xlabel("Controlled coefficient\n(Fisher z)", fontsize=FS_AXIS - 5, labelpad=5)
    ax_delta.set_xlabel("Change from baseline\nΔ Fisher z", fontsize=FS_AXIS - 5, labelpad=5)
    for ax in (ax_coef, ax_delta):
        ax.minorticks_off()
        ax.tick_params(axis="x", pad=1.5, labelsize=FS_TICK - 3, width=1.0, length=3.2)
        ax.tick_params(axis="y", length=0)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_linewidth(AXIS_LINE_WIDTH)
            ax.spines[spine].set_color(PALETTE["dark_gray"])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(False)
        ax.set_axisbelow(True)
        for y in y_positions:
            ax.axhline(y, color=guide_color, linewidth=0.75, zorder=0)

    _set_panel_title(
        ax_lab,
        FIGURE3_PANEL_D_TITLE,
        fontsize=FS_PANEL_TITLE - 1,
        pad=20,
    )
    ax_lab.text(
        0.0,
        1.012,
        FIGURE3_PANEL_D_SUBTITLE,
        transform=ax_lab.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_TICK - 2,
        color="#6A6A6A",
        clip_on=False,
    )
    _add_panel_label(ax_lab, "D")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=baseline_marker_color,
            markeredgecolor=coef_color,
            markersize=6.5,
            label="Baseline",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=coef_color,
            markeredgecolor=coef_color,
            markersize=6.5,
            label="Controlled",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=delta_color,
            markeredgecolor=delta_color,
            markersize=6.5,
            label="Δ from baseline",
        ),
    ]
    ax_leg.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=FS_TICK - 4,
        handletextpad=0.35,
        columnspacing=1.25,
        borderaxespad=0.0,
    )
    ax_note.text(
        0.0,
        0.50,
        "NC = not computable;\nfull reasons in caption/metadata.",
        ha="left",
        va="center",
        fontsize=FS_TICK - 5,
        color=nc_color,
        linespacing=1.15,
        transform=ax_note.transAxes,
        clip_on=False,
    )
    return ax_coef, ax_delta, ax_status, plot_controls



def _spec_display(control_id: str) -> str:
    """Map technical specification IDs to short publication labels."""
    key = _as_str(control_id).casefold()
    mapping = {
        "primary_d240_absolute_zlpi": "Primary 240 s",
        "duration_d180_zlpi": "ZLPI 180 s",
        "duration_d120_mwpi": "MWPI 120 s",
        "duration_d60_swpi": "SWPI 60 s",
        "broadband_residualized": "Broadband",
        "relative_power": "Relative",
        "absolute_log10": "Absolute",
    }
    if key in mapping:
        return mapping[key]
    text = _as_str(control_id).replace("_", " ").strip()
    if not text:
        return "Spec"
    # Keep unmapped IDs short so they do not collide with panel C.
    short = text[:1].upper() + text[1:]
    return short if len(short) <= 14 else short[:13] + "…"


def _finish_layout(fig: plt.Figure) -> None:
    """Leave room for panel letters and axis labels without collisions."""
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.10, wspace=0.36, hspace=0.42)


@dataclass(frozen=True)
class FigureArtifacts:
    figure_id: str
    pdf: Path
    svg: Path
    png: Path
    source_csvs: tuple[Path, ...]
    panels: tuple[FigurePanelSource, ...]


@dataclass(frozen=True)
class FiguresResult:
    figure1: FigureArtifacts
    figure2: FigureArtifacts
    figure3: FigureArtifacts
    figure3_supplement: FigureArtifacts | None
    figure3_internal_qc: tuple[Path, ...]
    figure_export_categories: Path
    figure_source_manifest: Path
    panel_records: tuple[FigurePanelSource, ...]


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


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return default


def _band_color(band: str) -> str:
    return BAND_COLORS.get(_as_str(band).casefold(), PALETTE["blue"])


def _band_marker(band: str) -> str:
    return BAND_MARKERS.get(_as_str(band).casefold(), "o")


def _band_linestyle(band: str) -> str | tuple:
    return BAND_LINESTYLES.get(_as_str(band).casefold(), "-")


def _band_display(band: str) -> str:
    text = _as_str(band).replace("_", " ").strip().casefold()
    if not text:
        return "Band"
    return text[:1].upper() + text[1:]


def _dataset_display(dataset_id: str, *, n_pairs: int | None = None) -> str:
    key = _as_str(dataset_id)
    label = DATASET_DISPLAY.get(key.casefold())
    if label is None:
        label = key.upper() if key == key.casefold() else key
    if n_pairs is not None and n_pairs >= 0:
        return f"{label} (n = {n_pairs} pairs)"
    return label


STATE_ROLE_DISPLAY: dict[str, str] = {
    "state_low": "Low-demand state",
    "low_demand": "Low-demand state",
    "rest": "Low-demand state",
    "state_high": "High-demand state",
    "high_demand": "High-demand state",
    "cognitive_effort": "High-demand state",
    "tetris": "High-demand state",
    "task": "High-demand state",
}
TIME_ROLE_DISPLAY: dict[str, str] = {
    "time_pre": "Pre-intervention",
    "pre": "Pre-intervention",
    "time_post": "Post-intervention",
    "post": "Post-intervention",
}


def format_role_display(
    role: str,
    *,
    raw_label: str | None = None,
    role_map: Mapping[str, str] | None = None,
) -> str:
    """General role label with optional YAML condition token in parentheses.

    Example: ``format_role_display("state_low", raw_label="rest")`` →
    ``"Low-demand state (rest)"``.
    """
    key = _as_str(role).casefold()
    mapping = role_map or STATE_ROLE_DISPLAY
    base = mapping.get(key, key.replace("_", " ").strip().title() or "Condition")
    raw = _as_str(raw_label).casefold()
    if raw and raw not in {key, base.casefold()} and raw not in mapping:
        return f"{base} ({raw})"
    if raw and raw in mapping and mapping[raw] == base and raw != key:
        return f"{base} ({raw})"
    return base


def sensitivity_display_title(body: str, *, dataset_id: str | None = None) -> str:
    """Manuscript panel title for sensitivity-only display (metadata-driven)."""
    body = body.strip()
    if dataset_id:
        return f"Sensitivity display ({_dataset_display(dataset_id)}): {body}"
    return f"Sensitivity display: {body}"


def _endpoint_display(endpoint_name: str) -> str:
    key = _as_str(endpoint_name, ENDPOINT_ZLPI)
    return ENDPOINT_DISPLAY.get(key, key.upper() if key else "ZLPI")


def _band_not_analyzed_message(band: str) -> str:
    return f"{_band_display(band)} band not analyzed"


def _mark_empty_panel(
    ax: plt.Axes,
    message: str,
    *,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Overlay a notice while retaining labeled x/y axes with numeric scales."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=FS_AXIS)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    _style_axes(ax, grid=True)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=FS_EMPTY,
        wrap=True,
        zorder=5,
        color=PALETTE["dark_gray"],
    )


def _n_datasets_from_rows(rows: Sequence[Mapping[str, object]]) -> int:
    counts = [
        _as_int(r.get("n_datasets"))
        for r in rows
        if _as_int(r.get("n_datasets")) > 0
    ]
    if counts:
        return max(counts)
    datasets = {
        _as_str(r.get("dataset_id")).casefold()
        for r in rows
        if _as_str(r.get("dataset_id"))
    }
    return len(datasets)


def _meta_or_single_title(*, n_datasets: int, endpoint_label: str = "ZLPI") -> str:
    del endpoint_label  # reserved for future panel subtitles
    if n_datasets >= 2:
        return "Random-effects meta-analysis"
    return "Single-dataset effect estimate"


def _band_sort_key(band: str) -> tuple[int, str]:
    key = _as_str(band).casefold()
    try:
        return (BAND_ORDER.index(key), key)
    except ValueError:
        return (len(BAND_ORDER), key)


def _cleanup_svg(path: Path) -> None:
    """Light SVG tidy: drop empty groups while preserving editable text."""
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        def _strip_empty(element: ET.Element) -> None:
            for child in list(element):
                _strip_empty(child)
                if child.tag == f"{ns}g" and len(child) == 0 and not (child.text or "").strip():
                    element.remove(child)

        _strip_empty(root)
        tree.write(path, encoding="utf-8", xml_declaration=True)
    except Exception:
        return


def save_figure_trio(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.45,
) -> tuple[Path, Path, Path]:
    _configure_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    save_kwargs: dict[str, object] = {"pad_inches": pad_inches}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(pdf, **save_kwargs)
    fig.savefig(svg, **save_kwargs)
    _cleanup_svg(svg)
    fig.savefig(png, dpi=FIGURE_DPI, **save_kwargs)
    plt.close(fig)
    return pdf, svg, png


def _shade_flanks(ax: plt.Axes, duration_s: int) -> None:
    """Shade distant flanks and (more lightly) local shoulders."""
    contract = contract_for_duration(duration_s)
    inner, outer = contract.flank_inner_s, contract.flank_outer_s
    sh_in, sh_out = contract.shoulders_inner_s, contract.shoulders_outer_s
    # Distant flanks |τ| ∈ [inner, outer]
    ax.axvspan(-outer, -inner, color=PALETTE["flank"], alpha=0.85, zorder=0, linewidth=0)
    ax.axvspan(inner, outer, color=PALETTE["flank"], alpha=0.85, zorder=0, linewidth=0)
    # Local shoulders |τ| ∈ [sh_in, sh_out] — subtle so curves remain readable
    ax.axvspan(-sh_out, -sh_in, color="#F0F0F0", alpha=0.9, zorder=0, linewidth=0)
    ax.axvspan(sh_in, sh_out, color="#F0F0F0", alpha=0.9, zorder=0, linewidth=0)
    _ref_vline(ax, 0.0)


def _annotate_lag_regions(ax: plt.Axes, duration_s: int, *, enabled: bool) -> None:
    """Add sparse lag-category labels once; positions match analysis contracts."""
    if not enabled:
        return
    contract = contract_for_duration(duration_s)
    # Guard: figure annotations must match confirmatory lag-category definitions.
    if (contract.flank_inner_s, contract.flank_outer_s) != ZLPI_FLANKS_S:
        raise ValueError(
            f"Figure flank annotation {contract.flanks_s} does not match ZLPI_FLANKS_S={ZLPI_FLANKS_S}"
        )
    if (contract.shoulders_inner_s, contract.shoulders_outer_s) != EXPECTED_SHOULDERS_S:
        raise ValueError(
            "Figure shoulder annotation does not match EXPECTED_SHOULDERS_S="
            f"{EXPECTED_SHOULDERS_S}"
        )
    y0, y1 = ax.get_ylim()
    y_span = y1 - y0
    # Distant labels stay near the bottom; shoulder labels sit above the curve.
    y_flank = y0 + 0.04 * y_span
    y_shoulder = y1 - 0.06 * y_span
    base_style = {
        "color": PALETTE["dark_gray"],
        "ha": "center",
        "clip_on": True,
        "alpha": 0.95,
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.12},
    }
    flank_style = {**base_style, "fontsize": FS_TICK - 3, "va": "bottom"}
    # Smaller type for narrow shoulder bands.
    shoulder_style = {**base_style, "fontsize": FS_TICK - 5, "va": "top"}
    flank_mid = 0.5 * (contract.flank_inner_s + contract.flank_outer_s)
    shoulder_mid = 0.5 * (contract.shoulders_inner_s + contract.shoulders_outer_s)
    ax.text(-flank_mid, y_flank, "distant -", **flank_style)
    ax.text(-shoulder_mid, y_shoulder, "shoulder -", **shoulder_style)
    ax.text(shoulder_mid, y_shoulder, "shoulder +", **shoulder_style)
    ax.text(flank_mid, y_flank, "distant +", **flank_style)


def _set_lag_axes(ax: plt.Axes, duration_s: int) -> None:
    contract = contract_for_duration(duration_s)
    ax.set_xlim(contract.lag_min_s, contract.lag_max_s)
    ax.set_xlabel(LAG_XLABEL, fontsize=FS_AXIS)
    ax.set_ylabel(Z_YLABEL, fontsize=FS_AXIS)
    _style_axes(ax)


def _is_low_demand_condition(row: Mapping[str, object]) -> bool:
    dataset_id = _as_str(row.get("dataset_id")).casefold()
    condition = _as_str(row.get("condition") or row.get("task")).casefold()
    role = _as_str(row.get("condition_role") or row.get("state")).casefold()
    if role == "low_demand":
        return True
    if dataset_id:
        state_role, _time_role, _session = condition_semantics_for(dataset_id, condition)
        if state_role == "state_low":
            return True
    return condition in LOW_DEMAND_CONDITION_LABELS


def read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_named_file(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def resolve_reporting_inputs(confirmatory_root: str | Path) -> dict[str, Path | None]:
    """Locate frozen confirmatory tables by canonical filename under ``root``."""
    root = Path(confirmatory_root).expanduser().resolve()
    names = {
        "curves_d240": "confirmatory_cross_correlation_curves_D240.csv",
        "curves_d180": "confirmatory_cross_correlation_curves_D180.csv",
        "curves_d120": "confirmatory_cross_correlation_curves_D120.csv",
        "curves_d60": "confirmatory_cross_correlation_curves_D60.csv",
        "endpoints_d240": "confirmatory_endpoint_metrics_D240.csv",
        "subject_level": "subject_level_metrics.csv",
        "paired_contrasts": "paired_contrasts.csv",
        "dataset_effects": "dataset_effects.csv",
        "meta_analysis": "meta_analysis_results.csv",
        "leave_one_out": "leave_one_dataset_out.csv",
        "peak_params": "peak_fit_params.csv",
        "peak_equivalence": "peak_center_equivalence.csv",
        "peak_hierarchical": "peak_hierarchical_summaries.csv",
        "null_subject": "null_subject_results.csv",
        "null_surrogate_values": NULL_SURROGATE_VALUES_FILENAME,
        "null_summary": "null_summary.csv",
        "protocol_audit": "protocol_audit.csv",
        "cardiac_peak_qc": "cardiac_peak_qc.csv",
        "cardiac_controls_observation": "cardiac_controls_observation_level.csv",
        "cardiac_controls_dataset_qc": "cardiac_controls_dataset_qc.csv",
        "cardiac_controls_metadata": "cardiac_controls_metadata.json",
        "sensitivity": "sensitivity_results.csv",
        "specification_matrix": "specification_matrix.csv",
        "duration_sensitivity": "duration_sensitivity.csv",
        "eligibility": "eligibility_by_duration.csv",
        "mixed_model": "mixed_model_results.csv",
        "mixed_model_marginal": "mixed_model_marginal_estimates.csv",
        "mixed_model_contrasts": "mixed_model_contrasts.csv",
        "aligned_d240": "features_confirmatory_aligned_D240.csv",
        "data_audit": "data_audit.csv",
    }
    return {key: discover_named_file(root, name) for key, name in names.items()}


def write_source_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys([*fieldnames, *STRUCTURED_NC_FIELDS]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row = with_structured_nc_fields(
                row,
                stage="C7",
                specification_id=str(row.get("specification_id") or path.stem),
            )
            payload = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                else:
                    payload[field] = value
            writer.writerow(payload)
    return path


def mean_ci_by_lag(
    curve_rows: Sequence[Mapping[str, object]],
    *,
    band: str,
    condition_role: str | None = None,
    power_representation: str = PRIMARY_REPRESENTATION,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
) -> list[dict[str, object]]:
    """Aggregate Fisher-z curves to mean ± 95% CI per lag (low-demand by default)."""
    buckets: dict[int, list[float]] = {}
    for row in curve_rows:
        if _as_int(row.get("duration_s"), duration_s) != duration_s:
            continue
        if _as_str(row.get("band")).casefold() != band.casefold():
            continue
        if (
            _as_str(row.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
            != power_representation.casefold()
        ):
            continue
        condition = _as_str(row.get("condition") or row.get("task")).casefold()
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        role = _as_str(row.get("condition_role") or row.get("state")).casefold()
        if condition_role == "low_demand":
            state_role = ""
            if dataset_id and condition:
                state_role, _time_role, _session = condition_semantics_for(
                    dataset_id, condition
                )
            is_low = (
                role in {"low_demand", "state_low"}
                or state_role == "state_low"
                or (
                    not role
                    and not state_role
                    and condition in LOW_DEMAND_CONDITION_LABELS
                )
            )
            if condition and not is_low:
                continue
        lag = int(round(_as_float(row.get("lag_s"))))
        r = _as_float(row.get("r"))
        z = fisher_z(r) if math.isfinite(r) else float("nan")
        if math.isfinite(z):
            buckets.setdefault(lag, []).append(z)

    rows: list[dict[str, object]] = []
    for lag in sorted(buckets):
        values = np.asarray(buckets[lag], dtype=float)
        n = int(values.size)
        mean = float(np.mean(values))
        if n >= 2:
            se = float(np.std(values, ddof=1) / math.sqrt(n))
            half = 1.959963984540054 * se
            ci_low, ci_high = mean - half, mean + half
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                "lag_s": lag,
                "band": band,
                "duration_s": duration_s,
                "endpoint_name": contract_for_duration(duration_s).endpoint_name,
                "power_representation": power_representation,
                "n": n,
                "mean_z": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def display_lag_mask(lags: np.ndarray, display_lag_step_s: int) -> np.ndarray:
    """Boolean mask for display-only lag thinning (analysis values unchanged)."""
    step = int(display_lag_step_s)
    if step <= 1:
        return np.ones(np.asarray(lags).shape, dtype=bool)
    lags_i = np.rint(np.asarray(lags, dtype=float)).astype(int)
    return (lags_i % step) == 0


def gaussian_smooth_display_series(
    values: np.ndarray | Sequence[float],
    *,
    sigma_s: float | None = None,
    lag_step_s: float = 1.0,
) -> np.ndarray:
    """Light 1-D Gaussian smooth for plot series only (not for inference).

    ``sigma_s`` is in seconds; with a 1 s lag grid this equals σ in samples.
    Defaults to ``LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S``. NaN positions are
    preserved. Does not mutate the input array.
    """
    from scipy.ndimage import gaussian_filter1d

    if sigma_s is None:
        sigma_s = float(LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S)
    arr = np.asarray(values, dtype=float).copy()
    if arr.size == 0:
        return arr
    step = float(lag_step_s) if float(lag_step_s) > 0.0 else 1.0
    sigma_samples = float(sigma_s) / step
    if not math.isfinite(sigma_samples) or sigma_samples <= 0.0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    if bool(finite.all()):
        return gaussian_filter1d(arr, sigma=sigma_samples, mode="nearest")
    filled = arr.copy()
    idx = np.arange(arr.size)
    filled[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    smoothed = gaussian_filter1d(filled, sigma=sigma_samples, mode="nearest")
    smoothed[~finite] = np.nan
    return smoothed


def render_figure1(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 1: six-panel manuscript layout (presentation only; C0–C6 frozen)."""
    from .figure1_panels import render_figure1 as _render_figure1_panels

    return _render_figure1_panels(inputs, output_dir)


def _paired_points_for_dataset(
    subject_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    *,
    dataset_id: str,
    band: str = "theta",
) -> list[dict[str, object]]:
    """Collect observation-level Δ rows for Panel A.

    Missing / ineligible contrasts are dropped before participant means:
    - non-primary representation / wrong duration / band / endpoint skipped;
    - ``contrast_eligible`` False skipped when that column is present;
    - non-finite ``delta_endpoint_index`` skipped.
    Participants with no remaining finite eligible contrasts contribute no unit
    mean (and therefore no inferential weight). Participants with unequal numbers
    of remaining contrasts still receive equal weight via within-participant
    averaging before the across-participant mean.
    """
    rows: list[dict[str, object]] = []
    for row in paired_rows:
        if _as_str(row.get("dataset_id")).casefold() != dataset_id.casefold():
            continue
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), 240) != 240:
            continue
        if _as_str(row.get("band")).casefold() != band.casefold():
            continue
        if (
            _as_str(row.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
            != PRIMARY_REPRESENTATION
        ):
            continue
        # When eligibility is recorded, keep only eligible contrasts.
        if "contrast_eligible" in row and not _as_bool(row.get("contrast_eligible")):
            continue
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "participant_id": _as_str(row.get("participant_id")),
                "session_id": _as_str(row.get("session_id"), "single"),
                "contrast_id": _as_str(row.get("contrast_id")),
                "band": band,
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "power_representation": PRIMARY_REPRESENTATION,
                "sampling_unit": "participant_x_contrast",
                "low_endpoint_index": _as_float(row.get("low_endpoint_index")),
                "effort_endpoint_index": _as_float(row.get("effort_endpoint_index")),
                "delta_endpoint_index": delta,
            }
        )
    # If paired table lacks absolute sides, recover from subject metrics.
    if rows and all(
        not math.isfinite(float(r["low_endpoint_index"]))
        and not math.isfinite(float(r["effort_endpoint_index"]))
        for r in rows
    ):
        by_key: dict[tuple[str, str, str], float] = {}
        for s in subject_rows:
            if _as_str(s.get("dataset_id")).casefold() != dataset_id.casefold():
                continue
            if _as_str(s.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
                continue
            if _as_int(s.get("duration_s"), 240) != 240:
                continue
            if _as_str(s.get("band")).casefold() != band:
                continue
            key = (
                _as_str(s.get("participant_id")),
                _as_str(s.get("session_id"), "single"),
                _as_str(s.get("condition")).casefold(),
            )
            by_key[key] = _as_float(s.get("endpoint_index"))
        for row in rows:
            # Best-effort: leave deltas if sides missing.
            _ = by_key
    return rows


def render_figure2(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 2: six-panel manuscript layout (presentation only; C0–C6 frozen)."""
    from .figure2_panels import render_figure2 as _render_figure2_panels

    return _render_figure2_panels(inputs, output_dir)


def _plot_dataset_null_forest(
    ax: plt.Axes,
    dataset_inferences: Sequence[object],
    *,
    title: str,
    panel_label: str | None = "A",
) -> None:
    """Main Panel A: one row per dataset (participant-mean Δ ± Student-t CI)."""
    rows = list(dataset_inferences)
    if not rows:
        _mark_empty_panel(
            ax,
            MSG_NOT_INCLUDED,
            xlabel=f"Participant mean Δ ({ZLPI_METRIC})",
            ylabel="Dataset",
        )
        if panel_label:
            _add_panel_label(ax, panel_label)
        _set_panel_title(ax, title, fontsize=FS_PANEL_TITLE - 2, pad=10)
        return

    y_pos = np.arange(len(rows), dtype=float)
    labels: list[str] = []
    for idx, row in enumerate(rows):
        dataset = _as_str(getattr(row, "dataset_id", ""))
        n_part = int(getattr(row, "n_participants", 0))
        labels.append(f"{_dataset_display(dataset)} (n={n_part})")
        mean = float(getattr(row, "mean_delta"))
        ci_l = float(getattr(row, "ci_low"))
        ci_h = float(getattr(row, "ci_high"))
        y = float(idx)
        if math.isfinite(mean) and math.isfinite(ci_l) and math.isfinite(ci_h):
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - ci_l], [ci_h - mean]],
                fmt="D",
                color=PALETTE["orange"],
                markersize=FIGURE3_MARKER_SIZE,
                capsize=3.5,
                elinewidth=FIGURE3_CI_LINEWIDTH,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
                zorder=4,
            )
        elif math.isfinite(mean):
            ax.scatter(
                [mean],
                [y],
                s=FIGURE3_SCATTER_SIZE + 10,
                color=PALETTE["orange"],
                marker="D",
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.6,
                zorder=4,
            )
    _ref_vline(ax, 0.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=FS_TICK - 1)
    ax.set_xlabel(f"Participant mean Δ ({ZLPI_METRIC})", fontsize=FS_AXIS - 2, labelpad=6)
    ax.set_ylabel("Dataset", fontsize=FS_AXIS - 2, labelpad=6)
    # Keep in-panel text minimal; run-class / estimand notes go to caption.
    interps = {_as_str(getattr(r, "interpretation", "")) for r in rows}
    interp_text = next(iter(interps)) if len(interps) == 1 else ""
    if interp_text and "significantly" in interp_text.casefold():
        # Compact category only (no smoke/run badges).
        short = "exceeds null" if "exceeds" in interp_text.casefold() else interp_text
        ax.text(
            0.02,
            0.06,
            short,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_TICK - 4,
            color=PALETTE["dark_gray"],
            zorder=6,
        )
    _style_axes(ax)
    _set_panel_title(ax, title, fontsize=FS_PANEL_TITLE - 2, pad=10)
    if panel_label:
        _add_panel_label(ax, panel_label)


def _null_display_label(null_type: str) -> str:
    mapping = {
        NULL_TYPE_CIRCULAR_SHIFT: "Circular shift",
        NULL_TYPE_PHASE_RANDOMIZATION: "Phase randomized",
        NULL_TYPE_BLOCK_SHUFFLE: "Block shuffled",
    }
    return mapping.get(_as_str(null_type).casefold(), _as_str(null_type).replace("_", " "))


def _dataset_role_lookup(protocol_rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in protocol_rows:
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        role = _as_str(row.get("dataset_role")).casefold()
        if dataset_id and role:
            out[dataset_id] = role
    return out


def _panel_a_slice_null_row(row: Mapping[str, object]) -> bool:
    return (
        _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() == ENDPOINT_ZLPI
        and _as_int(row.get("duration_s"), PRIMARY_DURATION_S) == PRIMARY_DURATION_S
        and _as_str(row.get("band")).casefold() == PRIMARY_BAND
        and _as_str(row.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
        == PRIMARY_REPRESENTATION
    )


def _panel_a_slice_surrogate_row(row: Mapping[str, object]) -> bool:
    return (
        _as_str(row.get("endpoint"), ENDPOINT_ZLPI).casefold() == ENDPOINT_ZLPI
        and _as_int(row.get("duration"), PRIMARY_DURATION_S) == PRIMARY_DURATION_S
        and _as_str(row.get("band")).casefold() == PRIMARY_BAND
        and _as_str(row.get("representation"), PRIMARY_REPRESENTATION).casefold()
        == PRIMARY_REPRESENTATION
    )


def _panel_a_surrogate_counts_complete(
    surrogate_rows: Sequence[Mapping[str, object]],
) -> bool:
    """True when each observation×null block has the full requested surrogate count."""
    from .nulls import surrogate_export_is_complete

    sliced = [
        r
        for r in surrogate_rows
        if _panel_a_slice_surrogate_row(r)
        and _as_str(r.get("null_type")).casefold() in PANEL_A_REQUIRED_NULL_TYPES
    ]
    return surrogate_export_is_complete(sliced)


def _biological_participant_standardized_means(
    observation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate standardized observed ZLPI to biological participants.

    Hierarchy (equal weight at each level):
    observation → condition → participant-session → biological participant.
    """
    # observation-level rows keyed by bio/session/condition
    obs_buckets: dict[tuple[str, str, str, str], list[float]] = {}
    meta: dict[tuple[str, str], dict[str, object]] = {}
    for row in observation_rows:
        bio = _as_str(row.get("biological_participant_id")).casefold()
        session = _as_str(row.get("session_id"), "single").casefold() or "single"
        condition = _as_str(row.get("condition"), "unknown").casefold() or "unknown"
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        val = _as_float(row.get("standardized_observed_value"))
        if not bio or not math.isfinite(val):
            continue
        key = (dataset_id, bio, session, condition)
        obs_buckets.setdefault(key, []).append(val)
        meta[(dataset_id, bio)] = {
            "dataset_id": dataset_id,
            "dataset_role": _as_str(row.get("dataset_role")),
            "biological_participant_id": bio,
            "null_type": _as_str(row.get("null_type")).casefold(),
            "band": _as_str(row.get("band")).casefold(),
        }

    # condition means within participant-session
    cond_means: dict[tuple[str, str, str], list[float]] = {}
    for (dataset_id, bio, session, _condition), vals in obs_buckets.items():
        cond_means.setdefault((dataset_id, bio, session), []).append(float(np.mean(vals)))

    # session means within biological participant
    session_means: dict[tuple[str, str], list[float]] = {}
    for (dataset_id, bio, _session), vals in cond_means.items():
        session_means.setdefault((dataset_id, bio), []).append(float(np.mean(vals)))

    out: list[dict[str, object]] = []
    for (dataset_id, bio), vals in sorted(session_means.items()):
        info = meta.get((dataset_id, bio), {})
        out.append(
            {
                **info,
                "dataset_id": dataset_id,
                "biological_participant_id": bio,
                "n_sessions": len(vals),
                "standardized_observed_value": float(np.mean(vals)),
                "aggregation_level": "biological_participant",
                "aggregation_order": (
                    "observation->condition->session->biological_participant"
                ),
            }
        )
    return out


def _surrogate_rows_fallback_from_null_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Legacy fallback when C4 surrogate-value export is unavailable."""
    from .group_tables import normalize_keys

    out: list[dict[str, object]] = []
    for row in rows:
        if not _panel_a_slice_null_row(row):
            continue
        null_type = _as_str(row.get("null_type")).casefold()
        if null_type not in PANEL_A_REQUIRED_NULL_TYPES:
            continue
        keys = normalize_keys(
            {
                "dataset_id": row.get("dataset_id"),
                "subject_id": row.get("subject_id"),
                "observation_id": row.get("observation_id"),
                "condition": row.get("condition"),
                "task": row.get("task"),
                "participant_id": row.get("participant_id"),
                "session_id": row.get("session_id"),
            }
        )
        out.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")).casefold(),
                "dataset_role": "",
                "biological_participant_id": _as_str(keys.get("participant_id")),
                "analysis_unit_id": (
                    f"{_as_str(keys.get('dataset_id')).casefold()}|"
                    f"{_as_str(keys.get('participant_id'))}|"
                    f"{_as_str(keys.get('session_id'), 'single')}"
                ),
                "subject_id": _as_str(row.get("subject_id")),
                "session_id": _as_str(keys.get("session_id"), "single"),
                "observation_id": _as_str(row.get("observation_id")),
                "condition": _as_str(row.get("condition")),
                "period": "",
                "state": "",
                "task": _as_str(row.get("task")),
                "band": _as_str(row.get("band")).casefold(),
                "duration": _as_int(row.get("duration_s"), PRIMARY_DURATION_S),
                "representation": PRIMARY_REPRESENTATION,
                "endpoint": ENDPOINT_ZLPI,
                "null_type": null_type,
                "surrogate_index": 0,
                "surrogate_endpoint_index": _as_float(row.get("null_mean")),
                "observed_endpoint_index": _as_float(row.get("observed_endpoint_index")),
                "standardized_surrogate_value": 0.0,
                "standardized_observed_value": _as_float(row.get("effect_size_surrogate_z")),
                "null_mean": _as_float(row.get("null_mean")),
                "null_median": _as_float(row.get("null_median")),
                "null_std": _as_float(row.get("null_std")),
                "empirical_p": _as_float(row.get("empirical_p")),
                "rng_seed_u64": _as_str(row.get("rng_seed_u64")),
                "n_surrogates": _as_int(row.get("n_surrogates_requested"), 1),
                "eligibility_status": (
                    "eligible" if _as_bool(row.get("observed_eligible"), True) else "ineligible"
                ),
                "qc_status": "fallback_summary_only",
            }
        )
    return out


def _plot_panel_a_empirical_nulls(
    ax: plt.Axes,
    surrogate_rows: Sequence[Mapping[str, object]],
    *,
    dataset_roles: Mapping[str, str],
    panel_label: str = "A",
    full_surrogate_distributions: bool = True,
) -> list[dict[str, object]]:
    """Plot pooled surrogate densities with biological-participant overlays.

    Returns plot-layer audit rows describing each geometry.
    """
    layer_rows: list[dict[str, object]] = []
    rows = [r for r in surrogate_rows if _panel_a_slice_surrogate_row(r)]
    rows = [
        r
        for r in rows
        if _as_str(r.get("null_type")).casefold() in PANEL_A_REQUIRED_NULL_TYPES
    ]
    title = PANEL_A_TITLE
    xlabel = PANEL_A_XLABEL
    if not rows:
        _mark_empty_panel(ax, MSG_NOT_INCLUDED, xlabel=xlabel, ylabel="")
        _add_panel_label(ax, panel_label)
        ax.text(
            0.0,
            1.06,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_PANEL_TITLE - 4,
            color=PALETTE["dark_gray"],
            clip_on=False,
        )
        return layer_rows

    datasets = sorted(
        {_as_str(r.get("dataset_id")).casefold() for r in rows},
        key=lambda d: (_as_str(dataset_roles.get(d)) != "primary", d),
    )
    single_dataset = len(datasets) == 1
    null_order = list(PANEL_A_REQUIRED_NULL_TYPES)
    labels: list[str] = []
    distributions: list[np.ndarray] = []
    bio_point_sets: list[np.ndarray] = []
    dataset_means: list[float] = []
    row_meta: list[dict[str, object]] = []

    for dataset_id in datasets:
        role = _as_str(dataset_roles.get(dataset_id), "unknown")
        for null_type in null_order:
            cell = [
                r
                for r in rows
                if _as_str(r.get("dataset_id")).casefold() == dataset_id
                and _as_str(r.get("null_type")).casefold() == null_type
            ]
            if not cell:
                continue
            sur = np.asarray(
                [_as_float(r.get("standardized_surrogate_value")) for r in cell],
                dtype=float,
            )
            sur = sur[np.isfinite(sur)]
            if sur.size == 0:
                continue

            # One standardized observed value per observation (first surrogate row).
            obs_by_id: dict[str, dict[str, object]] = {}
            for r in cell:
                oid = _as_str(r.get("observation_id")).casefold()
                if not oid or oid in obs_by_id:
                    continue
                obs_by_id[oid] = {
                    "dataset_id": dataset_id,
                    "dataset_role": role,
                    "biological_participant_id": _as_str(
                        r.get("biological_participant_id")
                    ).casefold(),
                    "session_id": _as_str(r.get("session_id"), "single").casefold()
                    or "single",
                    "condition": _as_str(r.get("condition"), "unknown").casefold()
                    or "unknown",
                    "null_type": null_type,
                    "band": PRIMARY_BAND,
                    "standardized_observed_value": _as_float(
                        r.get("standardized_observed_value")
                    ),
                }
            bio_rows = _biological_participant_standardized_means(list(obs_by_id.values()))
            bio_vals = np.asarray(
                [_as_float(r.get("standardized_observed_value")) for r in bio_rows],
                dtype=float,
            )
            bio_vals = bio_vals[np.isfinite(bio_vals)]
            ds_mean = float(np.mean(bio_vals)) if bio_vals.size else float("nan")

            n_obs = len(obs_by_id)
            n_surr_req = {
                _as_int(r.get("n_surrogates"))
                for r in cell
                if _as_int(r.get("n_surrogates")) > 0
            }
            n_surr_per = int(min(n_surr_req)) if n_surr_req else 0
            n_sessions = len(
                {
                    _as_str(r.get("session_id")).casefold()
                    for r in cell
                    if _as_str(r.get("session_id"))
                }
            )
            # Prefer analysis_unit count when present; else unique bio×session.
            analysis_units = {
                _as_str(r.get("analysis_unit_id")).casefold()
                for r in cell
                if _as_str(r.get("analysis_unit_id"))
            }
            if not analysis_units:
                analysis_units = {
                    f"{_as_str(r.get('biological_participant_id')).casefold()}|"
                    f"{_as_str(r.get('session_id'), 'single').casefold()}"
                    for r in cell
                    if _as_str(r.get("biological_participant_id"))
                }

            distributions.append(sur)
            bio_point_sets.append(bio_vals)
            dataset_means.append(ds_mean)
            label = _null_display_label(null_type) if single_dataset else (
                f"{_null_display_label(null_type)}"
            )
            labels.append(label)
            row_meta.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": role,
                    "null_type": null_type,
                    "n_biological_participants": int(bio_vals.size),
                    "n_analysis_units": len(analysis_units),
                    "n_sessions": n_sessions,
                    "n_eligible_observations": n_obs,
                    "n_surrogates_per_observation": n_surr_per,
                    "n_surrogate_values": int(sur.size),
                    "full_distributions": bool(
                        full_surrogate_distributions and n_surr_per >= 2 and sur.size == n_obs * n_surr_per
                    ),
                }
            )

            layer_rows.extend(
                [
                    {
                        "dataset_id": dataset_id,
                        "null_type": null_type,
                        "layer": "null_density_violin",
                        "source_file": "figure3_panel_a_null_distributions.csv",
                        "source_column": "standardized_surrogate_value",
                        "aggregation_level": "pooled_observation_specific_surrogates",
                        "n_rows_available": int(sur.size),
                        "n_rows_plotted": int(sur.size),
                        "weighting": "unweighted_pooled_descriptive",
                        "description": (
                            "Horizontal half-violin / density of all standardized "
                            "surrogate ZLPI values pooled across eligible observations "
                            "(descriptive; not a dataset-level sampling distribution)."
                        ),
                    },
                    {
                        "dataset_id": dataset_id,
                        "null_type": null_type,
                        "layer": "participant_observed_points",
                        "source_file": "figure3_panel_a_null_distributions.csv",
                        "source_column": "standardized_observed_value",
                        "aggregation_level": "biological_participant",
                        "n_rows_available": int(bio_vals.size),
                        "n_rows_plotted": int(bio_vals.size),
                        "weighting": (
                            "observation->condition->session->"
                            "biological_participant equal-weight means"
                        ),
                        "description": (
                            "Jittered points: biological-participant summaries of "
                            "standardized observed ZLPI (not session units)."
                        ),
                    },
                    {
                        "dataset_id": dataset_id,
                        "null_type": null_type,
                        "layer": "dataset_mean_diamond",
                        "source_file": "figure3_panel_a_null_distributions.csv",
                        "source_column": "standardized_observed_value",
                        "aggregation_level": "dataset",
                        "n_rows_available": int(bio_vals.size),
                        "n_rows_plotted": 1 if math.isfinite(ds_mean) else 0,
                        "weighting": "equal_weight_across_biological_participants",
                        "description": (
                            "Orange diamond: equal-weight mean of biological-participant "
                            "standardized observed values."
                        ),
                    },
                    {
                        "dataset_id": dataset_id,
                        "null_type": null_type,
                        "layer": "zero_reference_line",
                        "source_file": "",
                        "source_column": "",
                        "aggregation_level": "observation_specific_null_center",
                        "n_rows_available": 1,
                        "n_rows_plotted": 1,
                        "weighting": "n/a",
                        "description": (
                            "Vertical dashed line at 0 = center of each "
                            "observation-specific null on the standardized scale."
                        ),
                    },
                ]
            )

    if not distributions:
        _mark_empty_panel(ax, MSG_NOT_INCLUDED, xlabel=xlabel, ylabel="")
        _add_panel_label(ax, panel_label)
        ax.text(
            0.0,
            1.06,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=FS_PANEL_TITLE - 4,
            color=PALETTE["dark_gray"],
            clip_on=False,
        )
        return layer_rows

    y = np.arange(len(distributions), dtype=float)
    # Full empirical null geometry (all surrogate values; no downsampling).
    vio = ax.violinplot(
        distributions,
        positions=y,
        orientation="horizontal",
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.78,
    )
    for body, meta in zip(vio["bodies"], row_meta, strict=True):
        null_type = _as_str(meta.get("null_type")).casefold()
        if null_type == NULL_TYPE_CIRCULAR_SHIFT:
            color = PALETTE["blue"]
        elif null_type == NULL_TYPE_PHASE_RANDOMIZATION:
            color = PALETTE["orange"]
        else:
            color = PALETTE["green"]
        body.set_facecolor(color)
        body.set_edgecolor(PALETTE["dark_gray"])
        body.set_alpha(0.28)
        body.set_linewidth(0.5)
        # Keep the lower half of each horizontal violin so observed points remain readable.
        path = body.get_paths()[0]
        vertices = path.vertices
        y_center = float(np.mean(vertices[:, 1]))
        vertices[:, 1] = np.clip(vertices[:, 1], None, y_center)

    for idx, bio_vals in enumerate(bio_point_sets):
        if bio_vals.size:
            rng = np.random.default_rng(
                int(
                    hashlib.md5(
                        f"{row_meta[idx]['dataset_id']}|{row_meta[idx]['null_type']}".encode()
                    ).hexdigest()[:8],
                    16,
                )
            )
            jitter = rng.uniform(-0.12, 0.12, size=bio_vals.size)
            ax.scatter(
                bio_vals,
                y[idx] + jitter,
                s=22,
                color=PALETTE["dark_gray"],
                alpha=0.75,
                linewidths=0.0,
                zorder=4,
                label="Participant observed" if idx == 0 else None,
            )
        if math.isfinite(dataset_means[idx]):
            ax.scatter(
                [dataset_means[idx]],
                [y[idx]],
                s=42,
                marker="D",
                color=PALETTE["vermillion"],
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.45,
                zorder=5,
                label="Dataset mean" if idx == 0 else None,
            )

    _ref_vline(ax, 0.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=FS_TICK - 2)
    ax.set_xlabel(
        xlabel,
        fontsize=FS_AXIS - 3,
        labelpad=8,
        linespacing=1.15,
    )
    ax.set_ylabel("")
    ax.tick_params(axis="x", pad=3)
    ax.tick_params(axis="y", pad=2)
    ax.margins(x=0.04)

    # Subtitle with hierarchy counts (session-condition units when single dataset).
    first = row_meta[0]
    role_disp = _as_str(first.get("dataset_role"), "unknown").title()
    if single_dataset:
        subtitle = (
            f"{_dataset_display(_as_str(first.get('dataset_id')))} — {role_disp} | "
            f"D{PRIMARY_DURATION_S} | "
            f"{int(first['n_biological_participants'])} biological participants | "
            f"{int(first['n_analysis_units'])} sessions | "
            f"{int(first['n_eligible_observations'])} observations | "
            f"{int(first['n_surrogates_per_observation'])} surrogates/observation"
        )
    else:
        subtitle = (
            f"Primary + sensitivity | D{PRIMARY_DURATION_S} | "
            "rows = null method; see eligibility/hierarchy exports for per-dataset counts"
        )
    if not full_surrogate_distributions or not all(
        bool(m.get("full_distributions")) for m in row_meta
    ):
        subtitle += " | incomplete surrogate export"

    _style_axes(ax)
    _add_panel_label(ax, panel_label)
    ax.text(
        0.0,
        1.10,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_PANEL_TITLE - 4,
        color=PALETTE["dark_gray"],
        clip_on=False,
    )
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_TICK - 4,
        color=PALETTE["dark_gray"],
        alpha=0.88,
        clip_on=False,
    )
    # Legend explaining layers (proxy artists when first-row labels missing).
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=PALETTE["light_gray"],
            edgecolor=PALETTE["dark_gray"],
            alpha=0.55,
            label="Null distribution",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["dark_gray"],
            markersize=5.5,
            label="Participant observed",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=PALETTE["vermillion"],
            markeredgecolor=PALETTE["dark_gray"],
            markersize=6.5,
            label="Dataset mean",
        ),
        Line2D(
            [0],
            [0],
            color=REF_LINE_COLOR,
            lw=FIGURE3_REF_LINEWIDTH,
            ls="--",
            label="Null center (0)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=FS_LEGEND - 3,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    return layer_rows


def _fit_ar1_phi_for_panel(values: np.ndarray) -> float:
    """OLS AR(1) coefficient used for per-observation diagnostics."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    x = arr[:-1]
    y = arr[1:]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x_m = x[mask]
    y_m = y[mask]
    x_mean = float(np.mean(x_m))
    y_mean = float(np.mean(y_m))
    denom = float(np.sum((x_m - x_mean) ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum((x_m - x_mean) * (y_m - y_mean)) / denom)


def _panel_b_parse_state_period(dataset_id: str, condition: str) -> tuple[str, str, str]:
    text = _as_str(condition).casefold()
    state_role, time_role, session_type = condition_semantics_for(dataset_id, text)
    state = (
        "rest"
        if state_role == "state_low"
        else ("task" if state_role == "state_high" else "")
    )
    period = (
        "pre"
        if time_role == "time_pre"
        else ("post" if time_role == "time_post" else "")
    )
    return state, period, session_type


def _panel_b_series_units(
    aligned_path: Path | None,
    *,
    band: str = PRIMARY_BAND,
) -> list[SeriesUnit]:
    if aligned_path is None or not aligned_path.is_file():
        return []
    units = series_units_from_aligned_rows(
        read_aligned_features_csv(aligned_path),
        duration_s=PRIMARY_DURATION_S,
        bands=(band,),
    )
    return [
        unit
        for unit in units
        if unit.power_representation.casefold() == PRIMARY_REPRESENTATION
    ]


def _null_standardize_effect(
    observed: float,
    null_values: Sequence[float],
    *,
    required_draws: int,
) -> tuple[float, float, float, float, str]:
    """Return null-standardized effect plus diagnostics.

    The standardization is valid only when the null has enough finite draws and
    non-degenerate dispersion at floating-point resolution for this magnitude.
    """
    finite = [float(v) for v in null_values if math.isfinite(float(v))]
    if len(finite) < int(required_draws):
        return float("nan"), float("nan"), float("nan"), float("nan"), "insufficient_finite_null_draws"
    null_mean = float(np.mean(finite))
    null_sd = float(np.std(np.asarray(finite, dtype=float), ddof=1)) if len(finite) > 1 else float("nan")
    null_span = float(np.max(finite) - np.min(finite))
    scale_ref = max(1.0, abs(null_mean), abs(float(observed)) if math.isfinite(float(observed)) else 1.0)
    # Numerical lower bound: below one floating-point spacing at this scale,
    # null variance is computationally indistinguishable from zero.
    denom_floor = float(np.spacing(scale_ref))
    if (not math.isfinite(null_sd)) or null_sd <= denom_floor:
        return float("nan"), null_mean, null_sd, denom_floor, "degenerate_null_distribution"
    if null_span <= denom_floor:
        return float("nan"), null_mean, null_sd, denom_floor, "degenerate_null_distribution"
    z_val = surrogate_effect_size(float(observed), finite)
    if not math.isfinite(z_val):
        return float("nan"), null_mean, null_sd, denom_floor, "degenerate_null_distribution"
    return float(z_val), null_mean, null_sd, denom_floor, ""


def _panel_b_unit_keys(unit: SeriesUnit) -> dict[str, str]:
    from .group_tables import normalize_keys

    keys = normalize_keys(
        {
            "dataset_id": unit.dataset_id,
            "subject_id": unit.subject_id,
            "observation_id": unit.observation_id,
            "condition": unit.condition,
            "task": unit.task,
        }
    )
    state, period, session_type = _panel_b_parse_state_period(unit.dataset_id, unit.condition)
    return {
        "dataset_id": _as_str(keys.get("dataset_id"), unit.dataset_id).casefold(),
        "biological_participant_id": _as_str(keys.get("participant_id")).casefold(),
        "session_id": _as_str(keys.get("session_id"), "single").casefold() or "single",
        "state": state,
        "period": period,
        "session_type": session_type,
    }


def _seeded_bio_derangement(
    bio_ids: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray | None:
    n = len(bio_ids)
    if n < 2:
        return None
    base = np.arange(n, dtype=int)
    for _ in range(2000):
        perm = rng.permutation(n)
        if np.any(perm == base):
            continue
        if any(_as_str(bio_ids[i]) == _as_str(bio_ids[int(perm[i])]) for i in range(n)):
            continue
        return perm
    return None


def _panel_b_cross_subject_and_innovation(
    units: Sequence[SeriesUnit],
    *,
    n_null_draws: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Observation-level cross-subject HR mismatch and innovations analyses."""
    if not units:
        return [], [], []

    unit_meta = {unit.observation_id: _panel_b_unit_keys(unit) for unit in units}
    pools: dict[tuple[str, ...], list[SeriesUnit]] = defaultdict(list)
    for unit in units:
        meta = unit_meta[unit.observation_id]
        pools[
            (
                meta["dataset_id"],
                unit.condition.casefold(),
                unit.modality.casefold(),
                str(int(unit.duration_s)),
                unit.band.casefold(),
                unit.power_representation.casefold(),
            )
        ].append(unit)

    donor_draw_map: dict[str, list[str]] = {}
    ineligible_cross_reason: dict[str, str] = {}
    cross_draw_rows: list[dict[str, object]] = []

    for pool_key, pool_units in pools.items():
        bios = [unit_meta[unit.observation_id]["biological_participant_id"] for unit in pool_units]
        unique_bios = {bio for bio in bios if bio}
        if len(unique_bios) < 2:
            for unit in pool_units:
                ineligible_cross_reason[unit.observation_id] = "insufficient_biological_donor_pool"
            continue
        seed = deterministic_seed(
            "|".join(pool_key),
            f"panel_b_cross_hr|draws={n_null_draws}",
            "cross_subject_mismatch",
        )
        rng = np.random.default_rng(seed)
        draw_assignments: list[np.ndarray] = []
        for _draw in range(int(n_null_draws) + 1):
            perm = _seeded_bio_derangement(bios, rng)
            if perm is None:
                draw_assignments = []
                break
            draw_assignments.append(perm)
        if not draw_assignments:
            for unit in pool_units:
                ineligible_cross_reason[unit.observation_id] = "no_valid_biological_derangement"
            continue
        for focal_idx, focal in enumerate(pool_units):
            donors = [pool_units[int(perm[focal_idx])].observation_id for perm in draw_assignments]
            donor_draw_map[focal.observation_id] = donors
            for draw_idx, donor_obs in enumerate(donors):
                donor_unit = next(u for u in pool_units if u.observation_id == donor_obs)
                cross_draw_rows.append(
                    {
                        "dataset_id": unit_meta[focal.observation_id]["dataset_id"],
                        "observation_id": focal.observation_id,
                        "biological_participant_id": unit_meta[focal.observation_id]["biological_participant_id"],
                        "session_id": unit_meta[focal.observation_id]["session_id"],
                        "condition": focal.condition,
                        "state": unit_meta[focal.observation_id]["state"],
                        "period": unit_meta[focal.observation_id]["period"],
                        "session_type": unit_meta[focal.observation_id]["session_type"],
                        "band": focal.band,
                        "duration_s": int(focal.duration_s),
                        "endpoint_name": contract_for_duration(focal.duration_s).endpoint_name,
                        "draw_index": int(draw_idx),
                        "is_cross_observed_draw": bool(draw_idx == 0),
                        "donor_observation_id": donor_obs,
                        "donor_biological_participant_id": unit_meta[donor_obs]["biological_participant_id"],
                        "donor_session_id": unit_meta[donor_obs]["session_id"],
                        "donor_period": unit_meta[donor_obs]["period"],
                        "donor_state": unit_meta[donor_obs]["state"],
                    }
                )

    observation_rows: list[dict[str, object]] = []
    innovation_diag_rows: list[dict[str, object]] = []

    for unit in units:
        meta = unit_meta[unit.observation_id]
        endpoint_name = contract_for_duration(unit.duration_s).endpoint_name
        identity = {
            "dataset_id": unit.dataset_id,
            "subject_id": unit.subject_id,
            "task": unit.task,
            "condition": unit.condition,
            "observation_id": unit.observation_id,
        }
        observed = compute_endpoint_index_from_series(
            unit.hr_z,
            unit.eeg_z,
            duration_s=unit.duration_s,
            identity=identity,
            band=unit.band,
            power_representation=unit.power_representation,
            duration_role=unit.duration_role,
            is_primary_representation=unit.is_primary_representation,
            pair=unit.pair,
        )
        e_correct = _as_float(observed.get("endpoint_index"))
        observed_eligible = bool(observed.get("eligible"))

        exclusion_reason = ""
        cross_values: list[float] = []
        donor_obs_ids = donor_draw_map.get(unit.observation_id, [])
        if not observed_eligible:
            exclusion_reason = "observed_endpoint_ineligible"
        elif unit.observation_id in ineligible_cross_reason:
            exclusion_reason = ineligible_cross_reason[unit.observation_id]
        elif not donor_obs_ids:
            exclusion_reason = "missing_donor_assignments"
        else:
            donor_by_obs = {u.observation_id: u for u in units}
            for donor_obs in donor_obs_ids:
                donor = donor_by_obs.get(donor_obs)
                if donor is None:
                    cross_values.append(float("nan"))
                    continue
                if len(donor.hr_z) != len(unit.eeg_z):
                    cross_values.append(float("nan"))
                    continue
                cross_metrics = compute_endpoint_index_from_series(
                    donor.hr_z,
                    unit.eeg_z,
                    duration_s=unit.duration_s,
                    identity=identity,
                    band=unit.band,
                    power_representation=unit.power_representation,
                    duration_role=unit.duration_role,
                    is_primary_representation=unit.is_primary_representation,
                    pair=unit.pair,
                )
                cross_values.append(_as_float(cross_metrics.get("endpoint_index")))
            if sum(1 for value in cross_values if math.isfinite(value)) < int(n_null_draws):
                exclusion_reason = "insufficient_finite_cross_draws"

        e_cross_obs = float("nan")
        cross_null = []
        mu_cross = float("nan")
        sd_cross = float("nan")
        z_correct = float("nan")
        z_cross = float("nan")
        delta_z = float("nan")
        if not exclusion_reason and cross_values:
            e_cross_obs = _as_float(cross_values[0])
            cross_null = [float(value) for value in cross_values[1:] if math.isfinite(float(value))]
            z_correct, mu_cross, sd_cross, _cross_floor, cross_reason = _null_standardize_effect(
                e_correct,
                cross_null,
                required_draws=int(n_null_draws),
            )
            z_cross, _mu_cross2, _sd_cross2, _cross_floor2, cross_reason2 = _null_standardize_effect(
                e_cross_obs,
                cross_null,
                required_draws=int(n_null_draws),
            )
            if cross_reason and not exclusion_reason:
                exclusion_reason = cross_reason
            if cross_reason2 and not exclusion_reason:
                exclusion_reason = cross_reason2
            if math.isfinite(z_correct) and math.isfinite(z_cross):
                delta_z = z_correct - z_cross
            else:
                delta_z = float("nan")

        # AR(1) innovations (observed endpoint from innovations, not raw series).
        hr_phi = _fit_ar1_phi_for_panel(unit.hr_z)
        eeg_phi = _fit_ar1_phi_for_panel(unit.eeg_z)
        # AR(1) innovations are defined for t>=1; sample 0 is undefined.
        # Trimming index 0 avoids circularly shifting a structural NaN through
        # the common-support anchor window, which can collapse the innovation
        # null distribution to a single eligible shift.
        hr_inn_full = ar1_innovations(unit.hr_z)
        eeg_inn_full = ar1_innovations(unit.eeg_z)
        hr_inn = np.asarray(hr_inn_full[1:], dtype=float)
        eeg_inn = np.asarray(eeg_inn_full[1:], dtype=float)
        ac_hr_before = lag1_autocorrelation(unit.hr_z)
        ac_eeg_before = lag1_autocorrelation(unit.eeg_z)
        ac_hr_after = lag1_autocorrelation(hr_inn)
        ac_eeg_after = lag1_autocorrelation(eeg_inn)
        innov_eff_n = int(np.sum(np.isfinite(hr_inn) & np.isfinite(eeg_inn)))
        innovation_failure = ""
        innovation_observed = float("nan")
        innovation_null_vals: list[float] = []
        innovation_null_finite_draws = 0
        innovation_null_mean = float("nan")
        innovation_null_sd = float("nan")
        z_innovation = float("nan")
        shifts = valid_circular_shifts(len(eeg_inn))
        if innov_eff_n < 3:
            innovation_failure = "insufficient_innovation_length"
        elif shifts.size == 0:
            innovation_failure = "no_valid_innovation_circular_shift"
        else:
            innov_obs_metrics = compute_endpoint_index_from_series(
                hr_inn,
                eeg_inn,
                duration_s=unit.duration_s,
                identity=identity,
                band=unit.band,
                power_representation=unit.power_representation,
                duration_role=unit.duration_role,
                is_primary_representation=unit.is_primary_representation,
                pair=unit.pair,
            )
            innovation_observed = _as_float(innov_obs_metrics.get("endpoint_index"))
            innov_seed = deterministic_seed(
                unit.observation_id,
                f"{unit.duration_s}|{unit.band}|{unit.power_representation}|innovation",
                "circular_shift",
            )
            innov_rng = np.random.default_rng(innov_seed)
            for _ in range(int(n_null_draws)):
                shift = int(innov_rng.choice(shifts))
                eeg_shift = circular_shift_series(eeg_inn, shift)
                null_metrics = compute_endpoint_index_from_series(
                    hr_inn,
                    eeg_shift,
                    duration_s=unit.duration_s,
                    identity=identity,
                    band=unit.band,
                    power_representation=unit.power_representation,
                    duration_role=unit.duration_role,
                    is_primary_representation=unit.is_primary_representation,
                    pair=unit.pair,
                )
                innovation_null_vals.append(_as_float(null_metrics.get("endpoint_index")))
            finite_innovation_null = [
                float(value)
                for value in innovation_null_vals
                if math.isfinite(float(value))
            ]
            innovation_null_finite_draws = len(finite_innovation_null)
            if innovation_null_finite_draws < int(n_null_draws):
                innovation_failure = "insufficient_finite_innovation_draws"
            else:
                (
                    z_innovation,
                    innovation_null_mean,
                    innovation_null_sd,
                    _innovation_floor,
                    innovation_reason,
                ) = _null_standardize_effect(
                    innovation_observed,
                    finite_innovation_null,
                    required_draws=int(n_null_draws),
                )
                if innovation_reason:
                    innovation_failure = innovation_reason

        innovation_diag_rows.append(
            {
                "dataset_id": meta["dataset_id"],
                "observation_id": unit.observation_id,
                "biological_participant_id": meta["biological_participant_id"],
                "session_id": meta["session_id"],
                "condition": unit.condition,
                "state": meta["state"],
                "period": meta["period"],
                "session_type": meta["session_type"],
                "band": unit.band,
                "duration_s": int(unit.duration_s),
                "endpoint_name": endpoint_name,
                "lag1_hr_before": ac_hr_before,
                "lag1_eeg_before": ac_eeg_before,
                "lag1_hr_after": ac_hr_after,
                "lag1_eeg_after": ac_eeg_after,
                "ar1_phi_hr": hr_phi,
                "ar1_phi_eeg": eeg_phi,
                "innovation_effective_length": innov_eff_n,
                "innovation_null_finite_draws": innovation_null_finite_draws,
                "innovation_null_nonfinite_draws": int(n_null_draws)
                - innovation_null_finite_draws,
                "model_failure_flag": bool(innovation_failure),
                "model_failure_reason": innovation_failure,
            }
        )

        observation_rows.append(
            {
                "dataset_id": meta["dataset_id"],
                "biological_participant_id": meta["biological_participant_id"],
                "session_id": meta["session_id"],
                "condition": unit.condition,
                "period": meta["period"],
                "state": meta["state"],
                "session_type": meta["session_type"],
                "observation_id": unit.observation_id,
                "band": unit.band,
                "duration_s": int(unit.duration_s),
                "endpoint_name": endpoint_name,
                "representation": unit.power_representation,
                "correct_endpoint": e_correct,
                "cross_observed_endpoint": e_cross_obs,
                "mean_cross_subject_endpoint": mu_cross,
                "cross_subject_null_mean": mu_cross,
                "cross_subject_null_sd": sd_cross,
                "correct_null_normalized_effect": z_correct,
                "cross_subject_null_normalized_effect": z_cross,
                "paired_specificity_contrast_delta_z": delta_z,
                "innovation_endpoint": innovation_observed,
                "innovation_null_mean": innovation_null_mean,
                "innovation_null_sd": innovation_null_sd,
                "innovation_null_normalized_effect": z_innovation,
                "innovation_null_finite_draws": innovation_null_finite_draws,
                "innovation_null_nonfinite_draws": int(n_null_draws)
                - innovation_null_finite_draws,
                "n_cross_subject_draws": len(cross_null),
                "n_innovation_null_draws": len(innovation_null_vals),
                "eligibility_flag": (not exclusion_reason),
                "exclusion_reason": exclusion_reason,
            }
        )
    return observation_rows, cross_draw_rows, innovation_diag_rows


def _panel_b_participant_aggregate(
    observation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    metrics = (
        "correct_null_normalized_effect",
        "cross_subject_null_normalized_effect",
        "paired_specificity_contrast_delta_z",
        "innovation_null_normalized_effect",
    )
    buckets: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in observation_rows:
        if not _as_bool(row.get("eligibility_flag"), False):
            continue
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("biological_participant_id")).casefold(),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("endpoint_name")).casefold(),
        )
        buckets[key].append(row)

    out: list[dict[str, object]] = []
    for (dataset_id, bio, band, endpoint_name), rows in sorted(buckets.items()):
        value_map: dict[str, float] = {}
        for metric in metrics:
            by_session: dict[str, list[float]] = defaultdict(list)
            by_sess_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
            for row in rows:
                session = _as_str(row.get("session_id"), "single").casefold() or "single"
                condition = _as_str(row.get("condition"), "unknown").casefold() or "unknown"
                val = _as_float(row.get(metric))
                if math.isfinite(val):
                    by_sess_cond[(session, condition)].append(val)
            for (session, _condition), vals in by_sess_cond.items():
                by_session[session].append(float(np.mean(vals)))
            sess_means = [float(np.mean(vals)) for vals in by_session.values() if vals]
            value_map[metric] = float(np.mean(sess_means)) if sess_means else float("nan")

        out.append(
            {
                "dataset_id": dataset_id,
                "biological_participant_id": bio,
                "band": band,
                "endpoint_name": endpoint_name,
                "n_contributing_observations": len(rows),
                "n_sessions": len(
                    {
                        _as_str(row.get("session_id"), "single").casefold() or "single"
                        for row in rows
                    }
                ),
                "aggregation_rule": "observation->condition->session->biological_participant",
                **value_map,
            }
        )
    return out


def _panel_b_group_summaries(
    participant_rows: Sequence[Mapping[str, object]],
    observation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    estimands = (
        "correct_null_normalized_effect",
        "cross_subject_null_normalized_effect",
        "paired_specificity_contrast_delta_z",
        "innovation_null_normalized_effect",
    )
    out: list[dict[str, object]] = []
    for estimand in estimands:
        vals = np.asarray(
            [
                _as_float(row.get(estimand))
                for row in participant_rows
                if math.isfinite(_as_float(row.get(estimand)))
            ],
            dtype=float,
        )
        n = int(vals.size)
        mean = float(np.mean(vals)) if n else float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        if n >= 2:
            se = float(np.std(vals, ddof=1) / math.sqrt(n))
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
        out.append(
            {
                "estimand": estimand,
                "band": PRIMARY_BAND,
                "endpoint_name": ENDPOINT_ZLPI,
                "estimate": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "inferential_n": n,
                "biological_participant_n": len(
                    {
                        _as_str(row.get("biological_participant_id")).casefold()
                        for row in participant_rows
                    }
                ),
                "session_unit_n": len(
                    {
                        (
                            _as_str(row.get("biological_participant_id")).casefold(),
                            _as_str(row.get("session_id"), "single").casefold() or "single",
                        )
                        for row in observation_rows
                        if _as_bool(row.get("eligibility_flag"), False)
                    }
                ),
                "method": "student_t_participant_means",
            }
        )
    return out


def _plot_panel_b_cross_subject_innovations(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    participant_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
) -> None:
    if not participant_rows:
        _mark_empty_panel(
            ax_top,
            MSG_NOT_INCLUDED,
            xlabel="Pairing condition",
            ylabel="Null-normalized endpoint effect",
        )
        _mark_empty_panel(
            ax_bottom,
            MSG_NOT_INCLUDED,
            xlabel="Control type",
            ylabel="Null-normalized endpoint effect",
        )
        _set_panel_title(ax_top, "Cross-subject specificity and innovations", fontsize=FS_PANEL_TITLE - 2, pad=10)
        _add_panel_label(ax_top, "B")
        return

    by_part = sorted(
        participant_rows,
        key=lambda r: (_as_str(r.get("dataset_id")), _as_str(r.get("biological_participant_id"))),
    )
    x_cross = 0.0
    x_correct = 1.0
    x_diff = 2.0
    for row in by_part:
        y_cross = _as_float(row.get("cross_subject_null_normalized_effect"))
        y_corr = _as_float(row.get("correct_null_normalized_effect"))
        if math.isfinite(y_cross) and math.isfinite(y_corr):
            ax_top.plot(
                [x_cross, x_correct],
                [y_cross, y_corr],
                color=PALETTE["light_gray"],
                lw=0.8,
                alpha=0.65,
                zorder=2,
            )
        if math.isfinite(y_cross):
            ax_top.scatter([x_cross], [y_cross], s=20, color=PALETTE["dark_gray"], alpha=0.8, zorder=3)
        if math.isfinite(y_corr):
            ax_top.scatter([x_correct], [y_corr], s=20, color=PALETTE["blue"], alpha=0.85, zorder=3)

    group_by_est = { _as_str(r.get("estimand")): r for r in group_rows }
    for x_pos, estimand, color in (
        (x_cross, "cross_subject_null_normalized_effect", PALETTE["dark_gray"]),
        (x_correct, "correct_null_normalized_effect", PALETTE["blue"]),
        (x_diff, "paired_specificity_contrast_delta_z", PALETTE["vermillion"]),
    ):
        row = group_by_est.get(estimand)
        if row is None:
            continue
        est = _as_float(row.get("estimate"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        if math.isfinite(est) and math.isfinite(lo) and math.isfinite(hi):
            ax_top.errorbar(
                x_pos,
                est,
                yerr=[[est - lo], [hi - est]],
                fmt="D",
                color=color,
                markersize=6.5,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.4,
                capsize=2.5,
                lw=1.1,
                zorder=4,
            )

    _ref_hline(ax_top, 0.0)
    ax_top.set_xticks([x_cross, x_correct, x_diff])
    ax_top.set_xticklabels(
        ["Cross-subject HR", "Correct simultaneous", "Correct − Cross"],
        fontsize=FS_TICK - 4,
    )
    # Single shared y-label on the lower axes avoids mid-panel collision.
    ax_top.set_ylabel("")
    _style_axes(ax_top)
    _set_panel_title(ax_top, "Cross-subject specificity and innovations", fontsize=FS_PANEL_TITLE - 2, pad=10)
    _add_panel_label(ax_top, "B")

    ctrl_order = [
        ("correct_null_normalized_effect", "Correct simultaneous", PALETTE["blue"], "o"),
        ("cross_subject_null_normalized_effect", "Cross-subject HR", PALETTE["dark_gray"], "s"),
        ("innovation_null_normalized_effect", "AR(1) innovations", PALETTE["purple"], "D"),
    ]
    for idx, (metric, label, color, marker) in enumerate(ctrl_order):
        vals = np.asarray(
            [_as_float(row.get(metric)) for row in by_part if math.isfinite(_as_float(row.get(metric)))],
            dtype=float,
        )
        if vals.size:
            jitter_rng = np.random.default_rng(42 + idx)
            x_vals = idx + jitter_rng.uniform(-0.10, 0.10, size=vals.size)
            ax_bottom.scatter(x_vals, vals, s=18, color=color, alpha=0.65, zorder=3)
        g = group_by_est.get(metric)
        if g:
            est = _as_float(g.get("estimate"))
            lo = _as_float(g.get("ci_low"))
            hi = _as_float(g.get("ci_high"))
            if math.isfinite(est) and math.isfinite(lo) and math.isfinite(hi):
                ax_bottom.errorbar(
                    idx,
                    est,
                    yerr=[[est - lo], [hi - est]],
                    fmt=marker,
                    color=color,
                    markersize=7,
                    markeredgecolor=PALETTE["dark_gray"],
                    markeredgewidth=0.4,
                    capsize=3,
                    lw=1.2,
                    zorder=4,
                )
    _ref_hline(ax_bottom, 0.0)
    ax_bottom.set_xticks([0, 1, 2])
    ax_bottom.set_xticklabels(
        ["Correct", "Cross-subject", "AR(1) innovations"],
        fontsize=FS_TICK - 4,
    )
    ax_bottom.set_ylabel(
        "Null-normalized\nendpoint effect (Znull)",
        fontsize=FS_AXIS - 5,
        labelpad=6,
    )
    _style_axes(ax_bottom)
    ax_bottom.set_xlabel("Control type", fontsize=FS_AXIS - 4)


def _panel_c_dataset_role(dataset_id: str, dataset_roles: Mapping[str, str]) -> str:
    ds = _as_str(dataset_id).casefold()
    role = _as_str(dataset_roles.get(ds)).casefold()
    if role:
        return role
    if ds in FIGURE1_PRIMARY_DATASETS:
        return "primary"
    if ds in FIGURE1_SENSITIVITY_DATASETS:
        return "sensitivity"
    return "unknown"


def _panel_c_supported_durations(dataset_id: str) -> frozenset[int]:
    """Return duration set allowed for Panel C from YAML capabilities.

    Declared capability alone never computes an endpoint; this only gates which
    duration contracts may appear for a dataset. Missing YAML → all locked
    durations (eligibility still decides computability).
    """
    ds = _as_str(dataset_id).casefold()
    if not ds:
        return frozenset(EXPECTED_DURATIONS_S)
    try:
        import yaml

        from .config import load_dataset_config, load_master_config

        repo_root = Path(__file__).resolve().parents[2] / "zero-lag-reanalysis-repo"
        dataset_yaml = repo_root / "datasets" / f"{ds}.yaml"
        master_yaml = repo_root / "master.yaml"
        if not dataset_yaml.is_file() or not master_yaml.is_file():
            # Lightweight parse when master is unavailable.
            if dataset_yaml.is_file():
                raw = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
                caps = raw.get("capabilities") or {}
                allowed = {60}
                if bool(caps.get("supports_d120", True)):
                    allowed.add(120)
                if bool(caps.get("supports_d180", True)):
                    allowed.add(180)
                if bool(caps.get("supports_d240", True)):
                    allowed.add(240)
                return frozenset(allowed)
            return frozenset(EXPECTED_DURATIONS_S)
        master = load_master_config(master_yaml)
        cfg = load_dataset_config(dataset_yaml, master=master)
        caps = cfg.capabilities
        allowed = {60}
        if caps.supports_d120:
            allowed.add(120)
        if caps.supports_d180:
            allowed.add(180)
        if caps.supports_d240:
            allowed.add(240)
        return frozenset(allowed)
    except Exception:  # noqa: BLE001
        return frozenset(EXPECTED_DURATIONS_S)


def _split_observation_ids(raw_ids: object) -> list[str]:
    text = _as_str(raw_ids)
    if not text:
        return []
    return [token.strip() for token in text.split(";") if token.strip()]


def _panel_c_contract_identity(duration_s: int) -> dict[str, object]:
    """Prespecified duration-contract identity fields for Panel C."""
    contract = contract_for_duration(int(duration_s))
    alias = _as_str(contract.endpoint_alias)
    formula = (
        f"{alias} = z(r0) - mean_z(flanks); "
        f"lag±{int(contract.lag_max_s)}; "
        f"flanks |τ|∈[{int(contract.flank_inner_s)},{int(contract.flank_outer_s)}]"
    )
    return {
        "endpoint_name": contract.endpoint_name,
        "endpoint_alias": alias,
        "lag_min_s": int(contract.lag_min_s),
        "lag_max_s": int(contract.lag_max_s),
        "zero_lag_window_s": 0,
        "flank_inner_s": int(contract.flank_inner_s),
        "flank_outer_s": int(contract.flank_outer_s),
        "duration_contract_id": contract.lag_analysis_role,
        "n_overlap_expected_at_lag_max": int(
            contract.expected_constant_overlap_if_fully_finite
        ),
        "is_standard_zlpi": bool(contract.is_standard_zlpi),
        "endpoint_formula": formula,
    }


def _panel_c_expected_endpoint_name(duration_s: int) -> str:
    return contract_for_duration(int(duration_s)).endpoint_name


def _panel_c_curve_endpoint_rows(inputs: Mapping[str, Path | None]) -> list[dict[str, object]]:
    """Recompute duration-specific proximal endpoints from frozen lag curves.

    Mixed locked design: D60=SWPI, D120=MWPI, D180/D240=ZLPI. Endpoint identity is
    taken from ``evaluate_endpoint_curve`` / the frozen duration contract and is
    never overwritten to ``zlpi``.
    """
    out: list[dict[str, object]] = []
    for duration_s in EXPECTED_DURATIONS_S:
        key = f"curves_d{int(duration_s)}"
        curve_rows = read_csv_rows(inputs.get(key))
        if not curve_rows:
            continue
        contract = contract_for_duration(int(duration_s))
        identity = _panel_c_contract_identity(int(duration_s))
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in curve_rows:
            g_key = (
                _as_str(row.get("dataset_id")).casefold(),
                _as_str(row.get("subject_id")).casefold(),
                _as_str(row.get("task")).casefold(),
                _as_str(row.get("condition")).casefold(),
                _as_str(row.get("observation_id")).casefold(),
                _as_str(row.get("band")).casefold(),
                _as_str(row.get("power_representation")).casefold(),
                _as_str(row.get("pair")).casefold(),
            )
            grouped[g_key].append(row)
        for group_key, rows in grouped.items():
            metrics, _qc = evaluate_endpoint_curve(rows, duration_s=int(duration_s))
            returned_name = _as_str(metrics.get("endpoint_name")).casefold()
            returned_alias = _as_str(metrics.get("endpoint_alias"), identity["endpoint_alias"])
            is_standard = _as_bool(metrics.get("is_standard_zlpi"), False)
            if returned_name != _as_str(contract.endpoint_name).casefold():
                raise RuntimeError(
                    "Panel C endpoint identity mismatch from evaluate_endpoint_curve "
                    f"at D{duration_s}: got {returned_name!r}, expected "
                    f"{contract.endpoint_name!r}."
                )
            if is_standard != bool(contract.is_standard_zlpi):
                raise RuntimeError(
                    f"Panel C is_standard_zlpi mismatch at D{duration_s}: "
                    f"got {is_standard}, expected {contract.is_standard_zlpi}."
                )
            # Never relabel SWPI/MWPI as ZLPI.
            if (
                returned_name
                in {
                    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
                    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                }
                and returned_name == ENDPOINT_ZLPI
            ):
                raise RuntimeError("Panel C refused SWPI/MWPI labeled as ZLPI.")
            out.append(
                {
                    "dataset_id": group_key[0],
                    "subject_id": group_key[1],
                    "task": group_key[2],
                    "condition": group_key[3],
                    "observation_id": group_key[4],
                    "band": group_key[5],
                    "power_representation": group_key[6],
                    "pair": group_key[7],
                    "duration_s": int(duration_s),
                    "endpoint_name": returned_name,
                    "endpoint_alias": returned_alias,
                    "is_standard_zlpi": is_standard,
                    "lag_min_s": int(contract.lag_min_s),
                    "lag_max_s": int(contract.lag_max_s),
                    "zero_lag_window_s": 0,
                    "flank_inner_s": _as_int(
                        metrics.get("flank_inner_s"), int(contract.flank_inner_s)
                    ),
                    "flank_outer_s": _as_int(
                        metrics.get("flank_outer_s"), int(contract.flank_outer_s)
                    ),
                    "duration_contract_id": _as_str(
                        metrics.get("lag_analysis_role"), contract.lag_analysis_role
                    ),
                    "n_overlap_expected_at_lag_max": identity[
                        "n_overlap_expected_at_lag_max"
                    ],
                    "endpoint_index": _as_float(metrics.get("endpoint_index")),
                    "eligible": _as_bool(metrics.get("eligible"), False),
                    "exclusion_reason": _as_str(metrics.get("exclusion_reason")),
                    "n_common_support": _as_int(metrics.get("n_common_support")),
                    "min_common_support_required": int(
                        min_common_support_required(contract)
                    ),
                    "endpoint_formula": identity["endpoint_formula"],
                }
            )
    return out


def _panel_c_observation_level_rows(
    paired_rows: Sequence[Mapping[str, object]],
    endpoint_rows: Sequence[Mapping[str, object]],
    *,
    dataset_roles: Mapping[str, str],
) -> list[dict[str, object]]:
    """Recompute duration-specific effort−low-demand proximal coupling contrasts."""
    lookup: dict[tuple[str, int, str, str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in endpoint_rows:
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_int(row.get("duration_s")),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation")).casefold(),
            _as_str(row.get("observation_id")).casefold(),
        )
        lookup[key].append(row)

    out: list[dict[str, object]] = []
    for row in paired_rows:
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        duration_s = _as_int(row.get("duration_s"))
        band = _as_str(row.get("band")).casefold()
        representation = _as_str(
            row.get("power_representation"), PRIMARY_REPRESENTATION
        ).casefold()
        if duration_s not in EXPECTED_DURATIONS_S:
            continue
        if representation != PRIMARY_REPRESENTATION:
            continue

        participant_id = _as_str(row.get("participant_id")).casefold()
        session_id = _as_str(row.get("session_id"), "single").casefold() or "single"
        # Prefer explicit session-unit fields when present; otherwise compose
        # participant_id + session_id. Some datasets store protocol session
        # conditions in subject_id (YAML session_id_from=subject_suffix).
        biological_participant_id = participant_id or _as_str(
            row.get("subject_id")
        ).casefold()
        subject_id = _as_str(row.get("subject_id")).casefold()
        if subject_id and ("_" in subject_id) and session_id in {"", "single"}:
            # Session-suffixed subject IDs (e.g. 01_ph) act as session units.
            session_unit_id = subject_id
            biological_participant_id = subject_id.rsplit("_", 1)[0]
        elif subject_id and session_id not in {"", "single"} and subject_id.endswith(
            f"_{session_id}"
        ):
            session_unit_id = subject_id
        else:
            session_unit_id = (
                f"{biological_participant_id}_{session_id}"
                if session_id not in {"", "single"}
                else biological_participant_id
            )

        identity = _panel_c_contract_identity(int(duration_s))
        expected_name = _as_str(identity["endpoint_name"]).casefold()
        low_ids = _split_observation_ids(row.get("low_observation_ids"))
        effort_ids = _split_observation_ids(row.get("effort_observation_ids"))
        base = {
            "dataset_id": dataset_id,
            "dataset_role": _panel_c_dataset_role(dataset_id, dataset_roles),
            "duration_s": int(duration_s),
            "endpoint_name": expected_name,
            "endpoint_alias": identity["endpoint_alias"],
            "is_standard_zlpi": identity["is_standard_zlpi"],
            "lag_min_s": identity["lag_min_s"],
            "lag_max_s": identity["lag_max_s"],
            "zero_lag_window_s": identity["zero_lag_window_s"],
            "flank_inner_s": identity["flank_inner_s"],
            "flank_outer_s": identity["flank_outer_s"],
            "duration_contract_id": identity["duration_contract_id"],
            "n_overlap_expected_at_lag_max": identity["n_overlap_expected_at_lag_max"],
            "endpoint_formula": identity["endpoint_formula"],
            "band": band,
            "contrast_id": _as_str(row.get("contrast_id")).casefold(),
            "participant_id": participant_id,
            "biological_participant_id": biological_participant_id,
            "session_id": session_id,
            "session_unit_id": session_unit_id,
            "low_observation_ids": ";".join(low_ids),
            "effort_observation_ids": ";".join(effort_ids),
            "n_low_observations_used": 0,
            "n_effort_observations_used": 0,
            "low_endpoint": float("nan"),
            "effort_endpoint": float("nan"),
            "delta_endpoint": float("nan"),
            # Compat aliases retained for downstream readers.
            "low_zlpi": float("nan"),
            "effort_zlpi": float("nan"),
            "delta_zlpi": float("nan"),
            "estimand": "mean_effort_minus_low_demand_proximal_endpoint",
            "contrast_direction": "effort_endpoint-low_demand_endpoint",
            "segment_selection_rule": (
                "center-anchored nested segment from harmonized windows"
            ),
        }

        if int(duration_s) not in _panel_c_supported_durations(dataset_id):
            out.append(
                {
                    **base,
                    "eligibility_status": "excluded",
                    "exclusion_reason": "duration_not_supported_by_dataset_capabilities",
                }
            )
            continue

        low_vals: list[float] = []
        effort_vals: list[float] = []
        exclusion_reasons: list[str] = []
        matched_identity: Mapping[str, object] | None = None
        for obs_id in low_ids:
            items = lookup.get(
                (dataset_id, int(duration_s), band, representation, obs_id.casefold()),
                [],
            )
            vals = []
            for item in items:
                if not _as_bool(item.get("eligible"), False):
                    continue
                item_name = _as_str(item.get("endpoint_name")).casefold()
                if item_name != expected_name:
                    exclusion_reasons.append(
                        f"endpoint_identity_mismatch_low:{obs_id}:{item_name}"
                    )
                    continue
                value = _as_float(item.get("endpoint_index"))
                if math.isfinite(value):
                    vals.append(value)
                    matched_identity = item
            if vals:
                low_vals.append(float(np.mean(vals)))
            else:
                exclusion_reasons.append(f"missing_or_ineligible_low:{obs_id}")
        for obs_id in effort_ids:
            items = lookup.get(
                (dataset_id, int(duration_s), band, representation, obs_id.casefold()),
                [],
            )
            vals = []
            for item in items:
                if not _as_bool(item.get("eligible"), False):
                    continue
                item_name = _as_str(item.get("endpoint_name")).casefold()
                if item_name != expected_name:
                    exclusion_reasons.append(
                        f"endpoint_identity_mismatch_effort:{obs_id}:{item_name}"
                    )
                    continue
                value = _as_float(item.get("endpoint_index"))
                if math.isfinite(value):
                    vals.append(value)
                    matched_identity = item
            if vals:
                effort_vals.append(float(np.mean(vals)))
            else:
                exclusion_reasons.append(f"missing_or_ineligible_effort:{obs_id}")

        low_z = float(np.mean(low_vals)) if low_vals else float("nan")
        effort_z = float(np.mean(effort_vals)) if effort_vals else float("nan")
        delta_z = (
            float(effort_z - low_z)
            if math.isfinite(effort_z) and math.isfinite(low_z)
            else float("nan")
        )
        eligible = math.isfinite(delta_z)
        exclusion_reason = ";".join(exclusion_reasons)
        if matched_identity is not None:
            base["endpoint_name"] = _as_str(
                matched_identity.get("endpoint_name"), expected_name
            )
            base["endpoint_alias"] = _as_str(
                matched_identity.get("endpoint_alias"), identity["endpoint_alias"]
            )
            base["is_standard_zlpi"] = _as_bool(
                matched_identity.get("is_standard_zlpi"),
                bool(identity["is_standard_zlpi"]),
            )
            base["lag_min_s"] = _as_int(
                matched_identity.get("lag_min_s"), int(identity["lag_min_s"])
            )
            base["lag_max_s"] = _as_int(
                matched_identity.get("lag_max_s"), int(identity["lag_max_s"])
            )
            base["zero_lag_window_s"] = _as_int(
                matched_identity.get("zero_lag_window_s"), 0
            )
            base["flank_inner_s"] = _as_int(
                matched_identity.get("flank_inner_s"), int(identity["flank_inner_s"])
            )
            base["flank_outer_s"] = _as_int(
                matched_identity.get("flank_outer_s"), int(identity["flank_outer_s"])
            )
            base["duration_contract_id"] = _as_str(
                matched_identity.get("duration_contract_id"),
                identity["duration_contract_id"],
            )
            base["endpoint_formula"] = _as_str(
                matched_identity.get("endpoint_formula"), identity["endpoint_formula"]
            )

        out.append(
            {
                **base,
                "n_low_observations_used": len(low_vals),
                "n_effort_observations_used": len(effort_vals),
                "low_endpoint": low_z,
                "effort_endpoint": effort_z,
                "delta_endpoint": delta_z,
                "low_zlpi": low_z,
                "effort_zlpi": effort_z,
                "delta_zlpi": delta_z,
                "eligibility_status": "eligible" if eligible else "excluded",
                "exclusion_reason": exclusion_reason,
            }
        )
    return out


def _panel_c_summary_rows(
    observation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Dataset-specific duration summaries with participant-cluster bootstrap."""
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in observation_rows:
        if _as_str(row.get("eligibility_status")).casefold() != "eligible":
            continue
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("dataset_role")).casefold(),
            _as_int(row.get("duration_s")),
            _as_str(row.get("band")).casefold(),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for (dataset_id, dataset_role, duration_s, band), rows in sorted(grouped.items()):
        identity = _panel_c_contract_identity(int(duration_s))
        first = rows[0]
        by_bio: dict[str, list[float]] = defaultdict(list)
        session_units: set[str] = set()
        all_obs: set[str] = set()
        for row in rows:
            bio = _as_str(row.get("biological_participant_id")).casefold()
            delta = _as_float(row.get("delta_endpoint"))
            if not math.isfinite(delta):
                delta = _as_float(row.get("delta_zlpi"))
            if bio and math.isfinite(delta):
                by_bio[bio].append(delta)
            session_units.add(_as_str(row.get("session_unit_id")).casefold())
            all_obs.update(_split_observation_ids(row.get("low_observation_ids")))
            all_obs.update(_split_observation_ids(row.get("effort_observation_ids")))
        participant_means = np.asarray(
            [float(np.mean(vals)) for vals in by_bio.values() if vals], dtype=float
        )
        estimate = (
            float(np.mean(participant_means))
            if int(participant_means.size) > 0
            else float("nan")
        )
        ci_low = float("nan")
        ci_high = float("nan")
        if int(participant_means.size) >= 2:
            seed = int(
                hashlib.sha256(
                    f"{dataset_id}|{duration_s}|{band}|panel_c_cluster_boot".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16],
                16,
            )
            rng = np.random.default_rng(seed)
            boot = np.empty(2000, dtype=float)
            for i in range(2000):
                draw = rng.integers(
                    0, int(participant_means.size), int(participant_means.size)
                )
                boot[i] = float(np.mean(participant_means[draw]))
            ci_low = float(np.percentile(boot, 2.5))
            ci_high = float(np.percentile(boot, 97.5))

        eligible = int(participant_means.size) > 0 and math.isfinite(estimate)
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": dataset_role,
                "duration_s": int(duration_s),
                "endpoint_name": _as_str(
                    first.get("endpoint_name"), identity["endpoint_name"]
                ),
                "endpoint_alias": _as_str(
                    first.get("endpoint_alias"), identity["endpoint_alias"]
                ),
                "is_standard_zlpi": _as_bool(
                    first.get("is_standard_zlpi"), bool(identity["is_standard_zlpi"])
                ),
                "lag_min_s": _as_int(
                    first.get("lag_min_s"), int(identity["lag_min_s"])
                ),
                "lag_max_s": _as_int(
                    first.get("lag_max_s"), int(identity["lag_max_s"])
                ),
                "zero_lag_window_s": _as_int(first.get("zero_lag_window_s"), 0),
                "flank_inner_s": _as_int(
                    first.get("flank_inner_s"), int(identity["flank_inner_s"])
                ),
                "flank_outer_s": _as_int(
                    first.get("flank_outer_s"), int(identity["flank_outer_s"])
                ),
                "duration_contract_id": _as_str(
                    first.get("duration_contract_id"), identity["duration_contract_id"]
                ),
                "n_overlap_expected_at_lag_max": _as_int(
                    first.get("n_overlap_expected_at_lag_max"),
                    int(identity["n_overlap_expected_at_lag_max"]),
                ),
                "band": band,
                "effect_estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_observations": len(all_obs),
                "n_pairs": len(rows),
                "n_session_units": len({s for s in session_units if s}),
                "n_biological_participants": int(participant_means.size),
                "estimand": "mean_effort_minus_low_demand_proximal_endpoint",
                "contrast_direction": "effort_endpoint-low_demand_endpoint",
                "endpoint_formula": _as_str(
                    first.get("endpoint_formula"), identity["endpoint_formula"]
                ),
                "eligibility_status": "eligible" if eligible else "insufficient",
                "exclusion_reason": "",
                "uncertainty_method": "participant_cluster_bootstrap_2000",
                "plotted": bool(eligible),
            }
        )
    return summary_rows


def _validate_panel_c_rows(rows: Sequence[Mapping[str, object]]) -> None:
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        duration_s = _as_int(row.get("duration_s"))
        endpoint_name = _as_str(row.get("endpoint_name")).casefold()
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        band = _as_str(row.get("band")).casefold()
        status = _as_str(row.get("eligibility_status")).casefold()
        is_standard = _as_bool(row.get("is_standard_zlpi"), False)
        lag_max = _as_int(row.get("lag_max_s"))
        flank_inner = _as_int(row.get("flank_inner_s"))
        flank_outer = _as_int(row.get("flank_outer_s"))
        if duration_s not in EXPECTED_DURATIONS_S:
            raise RuntimeError(f"Panel C duration outside locked set: {duration_s}")
        if not dataset_id:
            raise RuntimeError("Panel C summary row has empty dataset_id.")
        supported = _panel_c_supported_durations(dataset_id)
        if duration_s not in supported and status == "eligible":
            raise RuntimeError(
                f"Panel C eligible row for {dataset_id} at D{duration_s} "
                f"outside capability-supported durations {sorted(supported)}."
            )
        contract = contract_for_duration(duration_s)
        expected_name = _as_str(contract.endpoint_name).casefold()
        if endpoint_name != expected_name:
            raise RuntimeError(
                f"Panel C endpoint identity error at D{duration_s}: "
                f"got {endpoint_name!r}, expected {expected_name!r}."
            )
        if duration_s == 60 and endpoint_name != ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX:
            raise RuntimeError("Panel C D60 must be SWPI.")
        if duration_s == 120 and endpoint_name != ENDPOINT_MID_WINDOW_PROXIMAL_INDEX:
            raise RuntimeError("Panel C D120 must be MWPI.")
        if duration_s in {180, 240} and endpoint_name != ENDPOINT_ZLPI:
            raise RuntimeError(f"Panel C D{duration_s} must be ZLPI.")
        if duration_s in {60, 120} and endpoint_name == ENDPOINT_ZLPI:
            raise RuntimeError(
                f"Panel C forbids labeling D{duration_s} as ZLPI (got SWPI/MWPI contract)."
            )
        if duration_s in {180, 240} and endpoint_name in {
            ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
            ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
        }:
            raise RuntimeError(
                f"Panel C forbids labeling D{duration_s} as SWPI/MWPI."
            )
        if lag_max != int(contract.lag_max_s) or (flank_inner, flank_outer) != tuple(
            contract.flanks_s
        ):
            raise RuntimeError(
                "Panel C lag/flank fields differ from duration contract "
                f"(duration={duration_s}, lag_max={lag_max}, "
                f"flanks=({flank_inner},{flank_outer}), "
                f"expected lag_max={contract.lag_max_s}, flanks={contract.flanks_s})."
            )
        if is_standard != bool(contract.is_standard_zlpi):
            raise RuntimeError(
                f"Panel C is_standard_zlpi mismatch at D{duration_s}."
            )
        if status == "eligible" and not math.isfinite(_as_float(row.get("effect_estimate"))):
            raise RuntimeError("Panel C eligible row has non-finite effect_estimate.")
        key = (dataset_id, duration_s, band)
        if key in seen:
            raise RuntimeError(f"Duplicate Panel C dataset-duration-band row: {key}")
        seen.add(key)


def _plot_panel_c_dataset_trajectories(
    gs_cell: matplotlib.gridspec.SubplotSpec,
    fig: plt.Figure,
    summary_rows: Sequence[Mapping[str, object]],
) -> list[plt.Axes]:
    """Plot dataset trajectories for mixed duration-specific proximal endpoints.

    Visualization-only layout: shared title/axes, duration×endpoint key beneath
    the facet grid, and a compact dataset legend (no right-edge trajectory labels).
    """
    plot_rows = [
        r
        for r in summary_rows
        if _as_str(r.get("eligibility_status")).casefold() == "eligible"
        and math.isfinite(_as_float(r.get("effect_estimate")))
        and _as_bool(r.get("plotted"), True)
    ]
    bands = [
        band
        for band in BAND_ORDER
        if any(_as_str(r.get("band")).casefold() == band for r in plot_rows)
    ]
    panel_title = "Duration-specific proximal coupling sensitivity"
    y_label = "Mean effort − low-demand\nproximal coupling effect"
    endpoint_styles = {
        ENDPOINT_ZLPI: ("o", "-"),
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: ("s", "--"),
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: ("^", "-."),
    }
    duration_endpoint_key = (
        (60, "SWPI"),
        (120, "MWPI"),
        (180, "ZLPI"),
        (240, "ZLPI"),
    )

    if not bands:
        ax = fig.add_subplot(gs_cell)
        _mark_empty_panel(
            ax,
            MSG_NOT_INCLUDED,
            xlabel="Duration (s)",
            ylabel=y_label,
        )
        _set_panel_title(ax, panel_title, fontsize=FS_PANEL_TITLE - 5, pad=8)
        _add_panel_label(ax, "C")
        return [ax]

    datasets = sorted({_as_str(r.get("dataset_id")).casefold() for r in plot_rows})
    cmap = plt.get_cmap("tab10")
    colors = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}
    # Small symmetric x-offsets only when multiple datasets share a duration.
    n_ds = len(datasets)
    if n_ds <= 1:
        x_offsets = {ds: 0.0 for ds in datasets}
    else:
        span = 10.0
        x_offsets = {
            ds: -0.5 * span + i * (span / (n_ds - 1)) for i, ds in enumerate(datasets)
        }

    # Shared y-limits across facets from estimates and CI ends.
    y_vals: list[float] = []
    for row in plot_rows:
        yy = _as_float(row.get("effect_estimate"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        if math.isfinite(yy):
            y_vals.append(yy)
        if math.isfinite(lo):
            y_vals.append(lo)
        if math.isfinite(hi):
            y_vals.append(hi)
    if y_vals:
        y_min = float(min(y_vals))
        y_max = float(max(y_vals))
        y_span = y_max - y_min if y_max > y_min else 0.1
        y_pad = 0.18 * y_span
        shared_ylim = (y_min - y_pad, y_max + y_pad)
    else:
        shared_ylim = (-0.1, 0.1)

    gs_wrap = gs_cell.subgridspec(
        4,
        2,
        height_ratios=[0.20, 1.0, 0.28, 0.30],
        width_ratios=[0.16, 1.0],
        hspace=0.28,
        wspace=0.06,
    )

    # Header: panel letter + title only.
    ax_header = fig.add_subplot(gs_wrap[0, :])
    ax_header.set_axis_off()
    ax_header.text(
        0.0,
        0.35,
        "C",
        transform=ax_header.transAxes,
        fontsize=FS_PANEL_LABEL,
        fontweight="bold",
        fontfamily="sans-serif",
        va="center",
        ha="left",
        color=PALETTE["dark_gray"],
        clip_on=False,
    )
    ax_header.text(
        0.065,
        0.35,
        panel_title,
        transform=ax_header.transAxes,
        fontsize=FS_PANEL_TITLE - 4,
        fontweight="normal",
        va="center",
        ha="left",
        color=PALETTE["dark_gray"],
        clip_on=False,
    )

    # Shared y-axis label (farther left of tick labels).
    ax_ylab = fig.add_subplot(gs_wrap[1, 0])
    ax_ylab.set_axis_off()
    ax_ylab.text(
        0.05,
        0.5,
        y_label,
        transform=ax_ylab.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=FS_AXIS - 5,
        color=PALETTE["dark_gray"],
        clip_on=False,
    )

    gs_facets = gs_wrap[1, 1].subgridspec(2, 2, hspace=0.55, wspace=0.28)
    axes: list[plt.Axes] = []
    for idx, band in enumerate(bands[:4]):
        ax = fig.add_subplot(gs_facets[idx // 2, idx % 2])
        axes.append(ax)
        band_rows = [r for r in plot_rows if _as_str(r.get("band")).casefold() == band]
        for ds in datasets:
            ds_rows = [
                r for r in band_rows if _as_str(r.get("dataset_id")).casefold() == ds
            ]
            if not ds_rows:
                continue
            ds_rows = sorted(ds_rows, key=lambda r: _as_int(r.get("duration_s")))
            x_plot = [
                _as_int(r.get("duration_s")) + x_offsets[ds] for r in ds_rows
            ]
            y_plot = [_as_float(r.get("effect_estimate")) for r in ds_rows]
            role = _as_str(ds_rows[0].get("dataset_role")).casefold()
            ls = "-" if role == "primary" else "--"
            # Connect only within-dataset supported durations (no interpolation).
            if len(x_plot) >= 2 and ds != "ds003816":
                ax.plot(
                    x_plot,
                    y_plot,
                    color=colors[ds],
                    lw=1.35,
                    linestyle=ls,
                    alpha=0.9,
                    zorder=3,
                )
            for row, xx, yy in zip(ds_rows, x_plot, y_plot):
                lo = _as_float(row.get("ci_low"))
                hi = _as_float(row.get("ci_high"))
                endpoint_name = _as_str(row.get("endpoint_name")).casefold()
                marker, _ = endpoint_styles.get(endpoint_name, ("o", "-"))
                yerr = None
                if math.isfinite(lo) and math.isfinite(hi):
                    yerr = [[yy - lo], [hi - yy]]
                ax.errorbar(
                    xx,
                    yy,
                    yerr=yerr,
                    fmt=marker,
                    color=colors[ds],
                    markersize=5.2,
                    capsize=2.2,
                    elinewidth=0.95,
                    markeredgecolor=PALETTE["dark_gray"],
                    markeredgewidth=0.4,
                    zorder=4,
                    clip_on=False,
                )
        _ref_hline(ax, 0.0)
        ax.set_xticks([60, 120, 180, 240])
        ax.set_xlim(*FIGURE3_DURATION_XLIM)
        ax.set_ylim(*shared_ylim)
        ax.set_title(_band_display(band), fontsize=FS_PANEL_TITLE - 7, pad=8)
        ax.tick_params(axis="both", labelsize=FS_TICK - 3, pad=2)
        # Shared scales: y ticks only on left column; x ticks only on bottom row.
        if idx % 2 == 1:
            ax.tick_params(labelleft=False)
        if idx // 2 == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels(["60", "120", "180", "240"], fontsize=FS_TICK - 3)
        _style_axes(ax)

    # Shared duration × endpoint key (once below the full facet grid).
    ax_key = fig.add_subplot(gs_wrap[2, 1])
    ax_key.set_axis_off()
    ax_key.set_xlim(0.0, 1.0)
    ax_key.set_ylim(0.0, 1.0)
    # Align key columns with the four duration ticks conceptually.
    key_xs = (0.125, 0.375, 0.625, 0.875)
    for x_frac, (dur, alias) in zip(key_xs, duration_endpoint_key):
        ax_key.text(
            x_frac,
            0.72,
            str(dur),
            transform=ax_key.transAxes,
            ha="center",
            va="center",
            fontsize=FS_TICK - 2,
            color=PALETTE["dark_gray"],
            clip_on=False,
        )
        ax_key.text(
            x_frac,
            0.28,
            alias,
            transform=ax_key.transAxes,
            ha="center",
            va="center",
            fontsize=FS_TICK - 4,
            color=PALETTE["dark_gray"],
            clip_on=False,
        )

    # Dataset legend (left) + shared x-axis label (centered).
    ax_footer = fig.add_subplot(gs_wrap[3, :])
    ax_footer.set_axis_off()
    dataset_handles = []
    dataset_labels = []
    for ds in datasets:
        row = next(
            (r for r in plot_rows if _as_str(r.get("dataset_id")).casefold() == ds),
            None,
        )
        if row is None:
            continue
        role = _as_str(row.get("dataset_role")).casefold()
        ls = "-" if role == "primary" else "--"
        handle, = ax_footer.plot(
            [],
            [],
            color=colors[ds],
            linestyle=ls,
            marker="o",
            markersize=4.0,
            linewidth=1.25,
        )
        dataset_handles.append(handle)
        dataset_labels.append(_dataset_display(ds))

    if dataset_handles:
        n_leg = len(dataset_handles)
        ncol = 1 if n_leg <= 2 else (2 if n_leg <= 6 else 3)
        ax_footer.legend(
            dataset_handles,
            dataset_labels,
            title="Dataset",
            loc="upper left",
            bbox_to_anchor=(0.02, 1.05),
            fontsize=FS_LEGEND - 5,
            title_fontsize=FS_LEGEND - 4,
            frameon=False,
            ncol=ncol,
            handlelength=1.6,
            labelspacing=0.2,
            columnspacing=0.9,
            handletextpad=0.35,
            borderaxespad=0.0,
        )
    ax_footer.text(
        0.55,
        0.15,
        "Duration (s)",
        transform=ax_footer.transAxes,
        fontsize=FS_AXIS - 4,
        color=PALETTE["dark_gray"],
        va="center",
        ha="center",
        clip_on=False,
    )
    return axes


def _plot_participant_null_forest(
    ax: plt.Axes,
    participant_rows: Sequence[object],
    inference: object,
    *,
    title: str,
    panel_label: str | None = "A",
    ylabel: str = "Participant",
    show_run_badge: bool = True,
) -> None:
    """Supplemental horizontal forest of biological-participant Δ_p ± mean CI."""
    rows = list(participant_rows)
    if not rows:
        _mark_empty_panel(
            ax,
            MSG_NOT_INCLUDED,
            xlabel=f"Observed − null mean ({ZLPI_METRIC})",
            ylabel=ylabel,
        )
        if panel_label:
            _add_panel_label(ax, panel_label)
        _set_panel_title(ax, title, fontsize=FS_PANEL_TITLE - 3, pad=8)
        return

    y_labels = _participant_forest_labels(rows)
    y_pos = np.arange(len(rows), dtype=float)
    deltas = np.asarray([float(r.delta_p) for r in rows], dtype=float)  # type: ignore[attr-defined]
    ax.scatter(
        deltas,
        y_pos,
        s=FIGURE3_SCATTER_SIZE,
        color=PALETTE["blue"],
        marker="o",
        edgecolors=PALETTE["dark_gray"],
        linewidths=0.5,
        zorder=3,
    )
    summary_y = float(len(rows))
    mean = float(getattr(inference, "mean_delta"))
    lo = float(getattr(inference, "ci_low"))
    hi = float(getattr(inference, "ci_high"))
    if math.isfinite(mean):
        if math.isfinite(lo) and math.isfinite(hi):
            ax.errorbar(
                mean,
                summary_y,
                xerr=[[mean - lo], [hi - mean]],
                fmt="D",
                color=PALETTE["orange"],
                markersize=FIGURE3_MARKER_SIZE,
                capsize=3.5,
                elinewidth=FIGURE3_CI_LINEWIDTH,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
                zorder=4,
            )
        else:
            ax.scatter(
                [mean],
                [summary_y],
                s=FIGURE3_SCATTER_SIZE + 12,
                color=PALETTE["orange"],
                marker="D",
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.6,
                zorder=4,
            )
    _ref_vline(ax, 0.0)
    size_label = _as_str(getattr(inference, "sample_size_label", ""))
    n_part = int(getattr(inference, "n_participants", 0) or 0)
    if not size_label:
        mean_ytick = f"Mean (n={n_part})" if n_part else "Mean"
    else:
        # Prefer compact n_participants; avoid mangled "12 6 part." strings.
        mean_ytick = f"Mean (n={n_part})" if n_part else "Mean"
    ax.set_yticks(list(y_pos) + [summary_y])
    ax.set_yticklabels(
        y_labels + [mean_ytick],
        fontsize=FS_TICK - 2,
    )
    ax.set_xlabel(f"Participant Δ ({ZLPI_METRIC})", fontsize=FS_AXIS - 2, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS - 2, labelpad=4)
    if show_run_badge:
        run_class = _as_str(getattr(inference, "run_class", ""))
        interp = _as_str(getattr(inference, "interpretation", ""))
        badge = "Smoke" if run_class == "smoke_diagnostic" else run_class.replace("_", " ")
        note = " · ".join(p for p in (badge, interp) if p)
        if note:
            ax.text(
                0.02,
                0.98,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=FS_TICK - 4,
                color=PALETTE["dark_gray"],
            )
    _style_axes(ax)
    _set_panel_title(ax, title, fontsize=FS_PANEL_TITLE - 3, pad=8)
    if panel_label:
        _add_panel_label(ax, panel_label)


def _chunked(items: Sequence[object], size: int) -> list[list[object]]:
    if size <= 0:
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _cleanup_legacy_figure3_qc_stems(figures_dir: Path) -> None:
    """Remove legacy root-level QC stems that previously looked like supplements."""
    legacy_prefixes = (
        "figure3_supplement_participant_null_forests",
        FIGURE3_QC_PARTICIPANT_FOREST_STEM,
    )
    for prefix in legacy_prefixes:
        for path in figures_dir.glob(f"{prefix}*"):
            if path.is_file() and path.parent == figures_dir:
                path.unlink(missing_ok=True)


def _render_figure3_participant_forest_qc(
    analysis: object,
    output_dir: Path,
    *,
    include_internal_qc: bool = True,
) -> tuple[list[tuple[Path, Path, Path]], list[Path], list[FigurePanelSource]]:
    """Dataset-specific participant forests — internal QC only (not manuscript/supplement)."""
    source_dir = output_dir / "source_data"
    source_paths: list[Path] = []
    panel_sources: list[FigurePanelSource] = []
    trios: list[tuple[Path, Path, Path]] = []
    _cleanup_legacy_figure3_qc_stems(output_dir)
    if not include_internal_qc:
        return trios, source_paths, panel_sources

    qc_dir = output_dir / FIGURE3_INTERNAL_QC_SUBDIR
    qc_dir.mkdir(parents=True, exist_ok=True)

    participants = list(getattr(analysis, "participants"))
    dataset_inferences = {
        _as_str(getattr(inf, "dataset_id")): inf
        for inf in getattr(analysis, "dataset_inferences")
    }
    by_dataset: dict[str, list[object]] = {}
    for row in participants:
        by_dataset.setdefault(_as_str(getattr(row, "dataset_id")), []).append(row)

    page_payloads: list[tuple[str, list[object], object, int]] = []

    def _dataset_sort_key(dataset_id: str) -> tuple[int, str]:
        try:
            return (DATASET_DISPLAY_ORDER.index(dataset_id), dataset_id)
        except ValueError:
            return (len(DATASET_DISPLAY_ORDER), dataset_id)

    for dataset_id in sorted(by_dataset, key=_dataset_sort_key):
        rows = sorted(
            by_dataset[dataset_id],
            key=lambda r: _as_str(getattr(r, "display_label")),
        )
        chunks = _chunked(rows, FIGURE3_PARTICIPANT_FOREST_MAX_ROWS)
        inference = dataset_inferences.get(dataset_id)
        if inference is None:
            continue
        for page_i, chunk in enumerate(chunks, start=1):
            page_payloads.append((dataset_id, chunk, inference, page_i if len(chunks) > 1 else 0))

    if not page_payloads:
        return trios, source_paths, panel_sources

    # One figure page per dataset chunk so labels stay readable.
    for page_idx, (dataset_id, chunk, inference, page_i) in enumerate(page_payloads, start=1):
        n_rows = len(chunk)
        fig_h = max(4.5, 0.42 * (n_rows + 2) + 1.2)
        _configure_publication_style()
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        page_note = f" (page {page_i})" if page_i else ""
        size_label = _as_str(getattr(inference, "sample_size_label", ""))
        _plot_participant_null_forest(
            ax,
            chunk,
            inference,
            title=(
                f"{dataset_id}: biological-participant Δ vs circular-shift null"
                f"{page_note}"
            ),
            panel_label=None,
            ylabel="Participant",
        )
        fig.suptitle(
            "Figure 3 internal QC — participant null forests",
            fontsize=FS_SUPTITLE - 2,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.01,
            (
                f"{size_label}; each row = one unique biological participant. "
                "Internal QC only — not a manuscript or supplementary figure. "
                "Main Panel A aggregates to one row per dataset."
            ),
            ha="center",
            va="bottom",
            fontsize=FS_TICK - 2,
            color=PALETTE["dark_gray"],
        )
        fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.10)
        stem = (
            f"{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_{dataset_id}"
            if len(page_payloads) == 1
            else f"{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_{dataset_id}_p{page_idx:02d}"
        )
        trio = save_figure_trio(fig, qc_dir, stem)
        trios.append(trio)
        plt.close(fig)

    readme = qc_dir / "README.md"
    readme.write_text(
        (
            "# Figure 3 internal QC artifacts\n\n"
            "Dataset-specific participant null forests "
            f"(`{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_*`) are **internal QC** only.\n\n"
            "- **Not** manuscript Figure 3 (`figure3_temporal_artifact_specificity`).\n"
            "- **Not** supplementary material (`figure3_supplement_null_diagnostics`).\n"
            "- Exclude from manuscript and supplementary exports unless explicitly requested.\n"
        ),
        encoding="utf-8",
    )
    source_paths.append(readme)

    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3_internal_qc",
            panel_id="participant_null_forests",
            title="Biological-participant null forests by dataset (internal QC)",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[],
            source_data_csv=str(source_dir / "figure3_panel_a_participant_deltas.csv"),
            analysis_keys=[
                f"export_category={EXPORT_CATEGORY_INTERNAL_QC}",
                "role=internal_qc",
                "unit=biological_participant",
                f"max_rows_per_page={FIGURE3_PARTICIPANT_FOREST_MAX_ROWS}",
                "include_in_manuscript_export=false",
                "include_in_supplementary_export=false",
            ],
            notes=(
                "Internal QC participant forests faceted by dataset and paginated "
                f"when n_participants > {FIGURE3_PARTICIPANT_FOREST_MAX_ROWS}. "
                "Excluded from manuscript and supplementary exports by default. "
                + INDEPENDENT_UNIT_VERDICT
            ),
            export_category=EXPORT_CATEGORY_INTERNAL_QC,
        )
    )
    return trios, source_paths, panel_sources



def _render_figure3_null_supplement(
    null_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    secondary_records: Sequence[Mapping[str, object]],
) -> tuple[tuple[Path, Path, Path], list[Path], list[FigurePanelSource]]:
    """Supplemental nested scatter + secondary-null participant forests."""
    source_dir = output_dir / "source_data"
    source_paths: list[Path] = []
    panel_sources: list[FigurePanelSource] = []
    null_type_styles: dict[str, tuple[str, str]] = {
        "circular_shift": (PALETTE["blue"], "o"),
        "phase_randomization": (PALETTE["orange"], "s"),
        "block_shuffle": (PALETTE["green"], "^"),
        "cross_subject_mismatch": (PALETTE["vermillion"], "D"),
        "ar1_innovations": (PALETTE["purple"], "v"),
    }
    scatter_source = []
    for row in null_rows:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), PRIMARY_DURATION_S) != PRIMARY_DURATION_S:
            continue
        if (
            _as_str(row.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
            != PRIMARY_REPRESENTATION
        ):
            continue
        scatter_source.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")),
                "subject_id": _as_str(row.get("subject_id")),
                "observation_id": _as_str(row.get("observation_id")),
                "condition": _as_str(row.get("condition")),
                "band": _as_str(row.get("band")),
                "power_representation": PRIMARY_REPRESENTATION,
                "null_type": _as_str(row.get("null_type")),
                "endpoint_name": ENDPOINT_ZLPI,
                "n_surrogates_requested": _as_int(row.get("n_surrogates_requested")),
                "n_surrogates_finite": _as_int(row.get("n_surrogates_finite")),
                "observed_endpoint_index": _as_float(row.get("observed_endpoint_index")),
                "null_mean": _as_float(row.get("null_mean")),
                "empirical_p": _as_float(row.get("empirical_p")),
                "effect_size_surrogate_z": _as_float(row.get("effect_size_surrogate_z")),
                "diagnostic_label": FIGURE3_SUPPLEMENT_LABEL,
            }
        )

    _configure_publication_style()
    fig = plt.figure(figsize=FIGURE3_SUPPLEMENT_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.0, 1.0],
        hspace=FIGURE3_SUPPLEMENT_SUBPLOT_ADJUST["hspace"],
        wspace=FIGURE3_SUPPLEMENT_SUBPLOT_ADJUST["wspace"],
    )

    ax_scatter = fig.add_subplot(gs[0, :])
    if scatter_source:
        observed = np.asarray(
            [r["observed_endpoint_index"] for r in scatter_source], dtype=float
        )
        null_mean = np.asarray([r["null_mean"] for r in scatter_source], dtype=float)
        for null_type, (color, marker) in null_type_styles.items():
            mask = np.asarray(
                [r["null_type"] == null_type for r in scatter_source], dtype=bool
            )
            if not np.any(mask):
                continue
            ax_scatter.scatter(
                null_mean[mask],
                observed[mask],
                s=FIGURE3_SCATTER_SIZE,
                alpha=0.55,
                color=color,
                marker=marker,
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.4,
                zorder=2,
                label=null_type.replace("_", " "),
            )
        lims = [
            float(np.nanmin([null_mean.min(), observed.min()])),
            float(np.nanmax([null_mean.max(), observed.max()])),
        ]
        ax_scatter.plot(
            lims,
            lims,
            color=REF_LINE_COLOR,
            ls="--",
            lw=FIGURE3_REF_LINEWIDTH,
            zorder=1,
        )
        ax_scatter.legend(
            loc="lower right",
            fontsize=FS_LEGEND - 3,
            frameon=True,
            fancybox=False,
            edgecolor=PALETTE["light_gray"],
            framealpha=0.92,
            title="Null type",
            title_fontsize=FS_LEGEND - 2,
            markerscale=0.9,
            handlelength=1.2,
            labelspacing=0.3,
        )
    else:
        _mark_empty_panel(
            ax_scatter,
            MSG_NOT_INCLUDED,
            xlabel=f"Null mean endpoint ({ZLPI_METRIC})",
            ylabel=f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
        )
    ax_scatter.set_xlabel(f"Null mean ({ZLPI_METRIC})", fontsize=FS_AXIS - 2)
    ax_scatter.set_ylabel(
        f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
        fontsize=FS_AXIS - 2,
    )
    _style_axes(ax_scatter)
    _set_panel_title(
        ax_scatter,
        "Observed vs null mean (nested)",
        fontsize=FS_PANEL_TITLE - 2,
        pad=8,
    )
    _add_panel_label(ax_scatter, "S1")

    null_display = {
        "phase_randomization": "Phase randomization",
        "block_shuffle": "Block shuffle",
        "cross_subject_mismatch": "Cross-subject",
        "ar1_innovations": "AR(1) innovations",
    }
    # Secondary null forests at theta (participant-level), excluding primary.
    for idx, null_type in enumerate(SECONDARY_NULL_TYPES):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        participants, inference, _matched = analyze_null_slice(
            null_rows,
            band=PRIMARY_BAND,
            null_type=null_type,
            is_primary_slice=False,
        )
        short = null_display.get(null_type, null_type.replace("_", " "))
        _plot_participant_null_forest(
            ax,
            participants,
            inference,
            title=f"{short} (θ)",
            panel_label=f"S{idx + 2}",
            show_run_badge=False,
        )

    fig.suptitle(
        "Figure 3 supplement — temporal null diagnostics",
        fontsize=FS_SUPTITLE - 2,
        fontweight="bold",
        y=0.978,
    )
    fig.text(
        0.5,
        0.022,
        (
            "S1 nested (not independent). S2–S5: participant Δ vs secondary nulls (θ). "
            "Main Panel A uses circular-shift only."
        ),
        ha="center",
        va="bottom",
        fontsize=FS_TICK - 5,
        color=PALETTE["dark_gray"],
    )
    fig.subplots_adjust(**FIGURE3_SUPPLEMENT_SUBPLOT_ADJUST)
    trio = save_figure_trio(fig, output_dir, FIGURE3_SUPPLEMENT_STEM)

    caption_path = output_dir / "figure3_supplement_caption.txt"
    caption_path.write_text(
        (
            "Figure 3 supplement — temporal null diagnostics\n\n"
            "S1: Nested observed-vs-null scatter (observation × band × null type). "
            "Points are not independent; do not infer from point density or the "
            "identity line.\n"
            "S2–S5: Biological-participant Δ (observed − null mean) for secondary "
            "C4 nulls at the theta spotlight slice, with orange diamonds showing "
            "the participant-mean Δ and Student-t 95% CI.\n"
            "Main Figure 3 Panel A reports the circular-shift confirmatory forest; "
            "these panels are diagnostic only.\n"
            "Source data: figures/source_data/figure3_supplement_nested_null_scatter.csv "
            "and figure3_panel_a_participant_deltas.csv.\n"
        ),
        encoding="utf-8",
    )

    scatter_csv = source_dir / "figure3_supplement_nested_null_scatter.csv"
    write_source_csv(
        scatter_csv,
        scatter_source,
        (
            "dataset_id",
            "subject_id",
            "observation_id",
            "condition",
            "band",
            "power_representation",
            "null_type",
            "endpoint_name",
            "n_surrogates_requested",
            "n_surrogates_finite",
            "observed_endpoint_index",
            "null_mean",
            "empirical_p",
            "effect_size_surrogate_z",
            "diagnostic_label",
        ),
    )
    source_paths.append(scatter_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3_supplement",
            panel_id="nested_null_scatter_diagnostic",
            title=FIGURE3_SUPPLEMENT_LABEL,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[],
            source_data_csv=str(scatter_csv),
            analysis_keys=[
                f"export_category={EXPORT_CATEGORY_SUPPLEMENTARY}",
                "role=supplementary",
                "endpoint=zlpi",
                f"duration={PRIMARY_DURATION_S}",
                f"representation={PRIMARY_REPRESENTATION}",
                "include_in_manuscript_export=false",
                "include_in_supplementary_export=true",
                "not_for_confirmatory_inference=true",
            ],
            notes=(
                "Supplementary nested-observation diagnostic (S1–S5). Each S1 point "
                "is observation×band×null type; points are not independent; no "
                "inference from density or proportion above y=x. Statistical "
                "inference uses participant-level observed-minus-null Δ. "
                "Export category: supplementary (not manuscript main Figure 3)."
            ),
            export_category=EXPORT_CATEGORY_SUPPLEMENTARY,
        )
    )
    # Retain secondary_records reference in notes via export already written by caller.
    _ = secondary_records
    return trio, source_paths, panel_sources


@dataclass(frozen=True)
class Figure3RenderResult:
    """Figure 3 render outputs classified by export category."""

    manuscript: FigureArtifacts
    supplement: FigureArtifacts | None
    internal_qc_paths: tuple[Path, ...]
    panels: tuple[FigurePanelSource, ...]


def render_figure3(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
    *,
    include_internal_qc: bool = True,
) -> Figure3RenderResult:
    """Figure 3: nulls, cross-subject checks, duration, and cardiac controls."""
    null_rows = read_csv_rows(inputs.get("null_subject"))
    protocol_rows = read_csv_rows(inputs.get("protocol_audit"))
    surrogate_rows = read_csv_rows(inputs.get("null_surrogate_values"))
    used_surrogate_fallback = False
    if not surrogate_rows:
        surrogate_rows = _surrogate_rows_fallback_from_null_rows(null_rows)
        used_surrogate_fallback = True
    full_surrogate_distributions = (
        not used_surrogate_fallback and _panel_a_surrogate_counts_complete(surrogate_rows)
    )
    dataset_roles = _dataset_role_lookup(protocol_rows)
    duration = read_csv_rows(inputs.get("duration_sensitivity"))
    paired_rows = read_csv_rows(inputs.get("paired_contrasts"))
    subject_rows = read_csv_rows(inputs.get("subject_level"))
    cardiac_qc_rows = read_csv_rows(inputs.get("cardiac_peak_qc"))
    sensitivity_detail = read_csv_rows(inputs.get("sensitivity"))
    sensitivity = read_csv_rows(inputs.get("specification_matrix")) or sensitivity_detail
    loo = read_csv_rows(inputs.get("leave_one_out"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    manuscript_source_paths: list[Path] = []
    supplement_source_paths: list[Path] = []
    qc_paths: list[Path] = []
    # Alias used by the manuscript panel body below.
    source_paths = manuscript_source_paths

    _configure_publication_style()
    fig = plt.figure(figsize=FIGURE3_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=list(FIGURE3_GRID_WIDTH_RATIOS),
        height_ratios=list(FIGURE3_GRID_HEIGHT_RATIOS),
        hspace=FIGURE3_SUBPLOT_ADJUST["hspace"],
        wspace=FIGURE3_SUBPLOT_ADJUST["wspace"],
    )

    # Panel A: observed-over-empirical-null distributions (required null triplet).
    ax_a = fig.add_subplot(gs[0, 0])
    primary_analysis = analyze_null_slice_full(
        null_rows,
        band=PRIMARY_BAND,
        null_type=PRIMARY_NULL_TYPE,
        is_primary_slice=True,
    )
    primary_participants = primary_analysis.participants
    primary_inference = (
        primary_analysis.dataset_inferences[0]
        if len(primary_analysis.dataset_inferences) == 1
        else primary_analysis.pooled_inference
    )
    primary_matched = primary_analysis.matched
    panel_a_layer_rows = _plot_panel_a_empirical_nulls(
        ax_a,
        surrogate_rows,
        dataset_roles=dataset_roles,
        panel_label="A",
        full_surrogate_distributions=full_surrogate_distributions,
    )

    delta_rows = [
        {
            "dataset_id": p.dataset_id,
            "participant_id": p.participant_id,
            "subject_id": p.participant_id,
            "participant_unit_id": p.participant_unit_id,
            "display_label": p.display_label,
            "band": p.band,
            "null_type": p.null_type,
            "power_representation": PRIMARY_REPRESENTATION,
            "duration_s": PRIMARY_DURATION_S,
            "endpoint_name": ENDPOINT_ZLPI,
            "n_observations": p.n_observations,
            "n_conditions": p.n_conditions,
            "n_sessions": p.n_sessions,
            "mean_observed": p.mean_observed,
            "mean_null": p.mean_null,
            "delta_p": p.delta_p,
            "n_surrogates_requested_min": p.n_surrogates_requested_min,
            "n_surrogates_requested_max": p.n_surrogates_requested_max,
        }
        for p in primary_participants
    ]
    delta_csv = source_dir / "figure3_panel_a_participant_deltas.csv"
    write_source_csv(
        delta_csv,
        delta_rows,
        (
            "dataset_id",
            "participant_id",
            "subject_id",
            "participant_unit_id",
            "display_label",
            "band",
            "null_type",
            "power_representation",
            "duration_s",
            "endpoint_name",
            "n_observations",
            "n_conditions",
            "n_sessions",
            "mean_observed",
            "mean_null",
            "delta_p",
            "n_surrogates_requested_min",
            "n_surrogates_requested_max",
        ),
    )
    source_paths.append(delta_csv)

    condition_rows = [
        {
            "dataset_id": c.dataset_id,
            "participant_id": c.participant_id,
            "participant_unit_id": c.participant_unit_id,
            "session_id": c.session_id,
            "condition": c.condition,
            "display_label": c.display_label,
            "band": c.band,
            "null_type": c.null_type,
            "n_observations": c.n_observations,
            "mean_observed": c.mean_observed,
            "mean_null": c.mean_null,
            "delta_pc": c.delta_pc,
            "n_surrogates_requested_min": c.n_surrogates_requested_min,
            "n_surrogates_requested_max": c.n_surrogates_requested_max,
        }
        for c in primary_analysis.condition_deltas
    ]
    condition_csv = source_dir / "figure3_panel_a_participant_condition_deltas.csv"
    write_source_csv(
        condition_csv,
        condition_rows,
        (
            "dataset_id",
            "participant_id",
            "participant_unit_id",
            "session_id",
            "condition",
            "display_label",
            "band",
            "null_type",
            "n_observations",
            "mean_observed",
            "mean_null",
            "delta_pc",
            "n_surrogates_requested_min",
            "n_surrogates_requested_max",
        ),
    )
    source_paths.append(condition_csv)

    dataset_inference_rows = [inf.as_dict() for inf in primary_analysis.dataset_inferences]
    dataset_csv = source_dir / "figure3_panel_a_dataset_inference.csv"
    write_source_csv(
        dataset_csv,
        dataset_inference_rows,
        tuple(dataset_inference_rows[0].keys())
        if dataset_inference_rows
        else ("dataset_id", "n_participants", "mean_delta"),
    )
    source_paths.append(dataset_csv)

    count_rows = []
    for inf in primary_analysis.dataset_inferences:
        count_rows.append(
            {
                "dataset_id": inf.dataset_id,
                "n_biological_participants": inf.n_participants,
                "n_participant_conditions": inf.n_participant_conditions,
                "n_observations": inf.n_observations,
                "sample_size_label": inf.sample_size_label,
                "pooled_estimate_plotted": False,
            }
        )
    for p in primary_participants:
        count_rows.append(
            {
                "dataset_id": p.dataset_id,
                "participant_id": p.participant_id,
                "participant_unit_id": p.participant_unit_id,
                "display_label": p.display_label,
                "n_biological_participants": 1,
                "n_participant_conditions": p.n_conditions,
                "n_observations": p.n_observations,
                "n_sessions": p.n_sessions,
                "sample_size_label": "",
                "pooled_estimate_plotted": False,
            }
        )
    counts_csv = source_dir / "figure3_panel_a_unit_counts.csv"
    write_source_csv(
        counts_csv,
        count_rows,
        (
            "dataset_id",
            "participant_id",
            "participant_unit_id",
            "display_label",
            "n_biological_participants",
            "n_participant_conditions",
            "n_observations",
            "n_sessions",
            "sample_size_label",
            "pooled_estimate_plotted",
        ),
    )
    source_paths.append(counts_csv)

    unit_verdict_csv = source_dir / "figure3_panel_a_independent_unit_verdict.csv"
    write_source_csv(
        unit_verdict_csv,
        independent_unit_verdict_rows(),
        ("dataset_id", "independent_unit", "example_raw_subject_ids", "verdict"),
    )
    source_paths.append(unit_verdict_csv)

    inference_row = {
        **primary_inference.as_dict(),
        "pooled_estimate_plotted": False,
        "alternative_claim": (
            "Does observed theta ZLPI systematically exceed the circular-shift "
            "null at the biological-participant level within each dataset?"
        ),
    }
    inference_csv = source_dir / "figure3_panel_a_inference.csv"
    write_source_csv(
        inference_csv,
        [inference_row],
        tuple(inference_row.keys()),
    )
    source_paths.append(inference_csv)

    loo_rows = leave_one_participant_out(
        primary_participants,
        band=PRIMARY_BAND,
        null_type=PRIMARY_NULL_TYPE,
    )
    loo_csv = source_dir / "figure3_panel_a_leave_one_out.csv"
    write_source_csv(
        loo_csv,
        loo_rows,
        (
            "omitted_participant_unit_id",
            "omitted_dataset_id",
            "omitted_participant_id",
            "omitted_subject_id",
            "omitted_display_label",
            "omitted_delta_p",
            "n_participants_remaining",
            "mean_delta",
            "ci_low",
            "ci_high",
            "p_value_two_sided",
            "interpretation",
            "sample_size_label",
        ),
    )
    source_paths.append(loo_csv)

    secondary_records = secondary_band_null_fdr_table(null_rows)
    secondary_csv = source_dir / "figure3_secondary_nulls_fdr.csv"
    write_source_csv(
        secondary_csv,
        secondary_records,
        tuple(secondary_records[0].keys()) if secondary_records else ("band", "null_type"),
    )
    source_paths.append(secondary_csv)

    # Observation-level matched rows for the primary slice (audit trail).
    matched_csv = source_dir / "figure3_panel_a_matched_observations.csv"
    write_source_csv(
        matched_csv,
        [
            {
                "dataset_id": m.dataset_id,
                "subject_id": m.subject_id,
                "participant_id": m.participant_id,
                "participant_unit_id": m.participant_unit_id,
                "session_id": m.session_id,
                "observation_id": m.observation_id,
                "condition": m.condition,
                "modality": m.modality,
                "band": m.band,
                "null_type": m.null_type,
                "power_representation": m.power_representation,
                "duration_s": m.duration_s,
                "endpoint_name": m.endpoint_name,
                "observed_endpoint_index": m.observed_endpoint_index,
                "null_mean": m.null_mean,
                "delta_obs_minus_null": m.delta_obs_minus_null,
                "n_surrogates_requested": m.n_surrogates_requested,
                "rng_seed_u64": m.rng_seed_u64,
            }
            for m in primary_matched
        ],
        (
            "dataset_id",
            "subject_id",
            "participant_id",
            "participant_unit_id",
            "session_id",
            "observation_id",
            "condition",
            "modality",
            "band",
            "null_type",
            "power_representation",
            "duration_s",
            "endpoint_name",
            "observed_endpoint_index",
            "null_mean",
            "delta_obs_minus_null",
            "n_surrogates_requested",
            "rng_seed_u64",
        ),
    )
    source_paths.append(matched_csv)

    # Panel A distribution exports (all datasets, required null methods only).
    required_nulls = set(PANEL_A_REQUIRED_NULL_TYPES)
    panel_a_null_rows = [
        r
        for r in null_rows
        if _panel_a_slice_null_row(r)
        and _as_str(r.get("null_type")).casefold() in required_nulls
    ]
    panel_a_surrogates = [
        r
        for r in surrogate_rows
        if _panel_a_slice_surrogate_row(r)
        and _as_str(r.get("null_type")).casefold() in required_nulls
    ]
    all_dataset_ids = sorted(
        set(dataset_roles) | {_as_str(r.get("dataset_id")).casefold() for r in panel_a_null_rows}
    )
    eligibility_rows: list[dict[str, object]] = []
    hierarchy_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    dataset_summary_rows: list[dict[str, object]] = []

    for dataset_id in all_dataset_ids:
        ds_role = _as_str(dataset_roles.get(dataset_id), "unknown")
        ds_rows = [
            r
            for r in panel_a_null_rows
            if _as_str(r.get("dataset_id")).casefold() == dataset_id
        ]
        ds_sur = [
            r
            for r in panel_a_surrogates
            if _as_str(r.get("dataset_id")).casefold() == dataset_id
        ]
        nulls_present = {
            _as_str(r.get("null_type")).casefold()
            for r in ds_rows
            if _as_bool(r.get("observed_eligible"), True)
        }
        required_nulls_available = all(n in nulls_present for n in required_nulls)
        by_null_obs: dict[str, set[str]] = {}
        for r in ds_rows:
            if not _as_bool(r.get("observed_eligible"), True):
                continue
            nt = _as_str(r.get("null_type")).casefold()
            by_null_obs.setdefault(nt, set()).add(_as_str(r.get("observation_id")))
        obs_intersection: set[str] = set()
        if by_null_obs:
            obs_intersection = set.intersection(*by_null_obs.values()) if len(by_null_obs) >= 3 else set()
        included = required_nulls_available and bool(obs_intersection)
        exclusion_reason = ""
        if not ds_rows:
            exclusion_reason = "No D240/absolute_log10/theta/ZLPI rows."
        elif not required_nulls_available:
            missing = sorted(required_nulls - nulls_present)
            exclusion_reason = f"Missing required null methods: {', '.join(missing)}."
        elif not obs_intersection:
            exclusion_reason = "No observation overlap across required null methods."

        eligibility_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": ds_role,
                "included": included,
                "duration_available": bool(ds_rows),
                "endpoint_available": bool(ds_rows),
                "band_available": bool(ds_rows),
                "required_nulls_available": required_nulls_available,
                "n_eligible_observations": len(obs_intersection),
                "exclusion_reason": exclusion_reason,
            }
        )

        bio_ids = {
            _as_str(r.get("biological_participant_id")).casefold()
            for r in ds_sur
            if _as_str(r.get("biological_participant_id"))
        }
        analysis_ids = {
            _as_str(r.get("analysis_unit_id")).casefold()
            for r in ds_sur
            if _as_str(r.get("analysis_unit_id"))
        }
        session_ids = {
            _as_str(r.get("session_id")).casefold()
            for r in ds_sur
            if _as_str(r.get("session_id"))
        }
        obs_ids = {_as_str(r.get("observation_id")).casefold() for r in ds_rows}
        hierarchy_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": ds_role,
                "n_biological_participants": len(bio_ids),
                "n_analysis_units": len(analysis_ids),
                "n_sessions": len(session_ids),
                "n_observations": len(obs_ids),
                "n_eligible_observations": len(obs_intersection),
                "participant_identifier": "biological_participant_id",
                "session_identifier": "session_id",
                "condition_identifier": "condition",
                "aggregation_order": "observation->condition->session->biological_participant->dataset",
            }
        )

        for row in ds_rows:
            from .group_tables import normalize_keys

            keys = normalize_keys(
                {
                    "dataset_id": row.get("dataset_id"),
                    "subject_id": row.get("subject_id"),
                    "observation_id": row.get("observation_id"),
                    "condition": row.get("condition"),
                    "task": row.get("task"),
                    "participant_id": row.get("participant_id"),
                    "session_id": row.get("session_id"),
                }
            )
            null_type = _as_str(row.get("null_type")).casefold()
            percentile = float("nan")
            p = _as_float(row.get("empirical_p"))
            if math.isfinite(p):
                percentile = max(0.0, min(1.0, 1.0 - p))
            obs_id = _as_str(row.get("observation_id")).casefold()
            std_obs = float("nan")
            if ds_sur:
                for s in ds_sur:
                    if (
                        _as_str(s.get("observation_id")).casefold() == obs_id
                        and _as_str(s.get("null_type")).casefold() == null_type
                    ):
                        std_obs = _as_float(s.get("standardized_observed_value"))
                        break
            observation_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": ds_role,
                    "biological_participant_id": _as_str(keys.get("participant_id")),
                    "analysis_unit_id": (
                        f"{_as_str(keys.get('dataset_id')).casefold()}|"
                        f"{_as_str(keys.get('participant_id'))}|"
                        f"{_as_str(keys.get('session_id'), 'single')}"
                    ),
                    "session_id": _as_str(keys.get("session_id"), "single"),
                    "observation_id": _as_str(row.get("observation_id")),
                    "condition": _as_str(row.get("condition")),
                    "period": "",
                    "state": "",
                    "band": _as_str(row.get("band")).casefold(),
                    "null_type": null_type,
                    "observed_endpoint_index": _as_float(row.get("observed_endpoint_index")),
                    "null_mean": _as_float(row.get("null_mean")),
                    "null_median": _as_float(row.get("null_median")),
                    "null_std": _as_float(row.get("null_std")),
                    "delta_obs_minus_null": _as_float(row.get("observed_endpoint_index"))
                    - _as_float(row.get("null_mean")),
                    "standardized_observed_value": std_obs,
                    "empirical_percentile": percentile,
                    "empirical_p": p,
                    "n_surrogates": _as_int(row.get("n_surrogates_requested")),
                    "eligibility_status": (
                        "eligible" if _as_bool(row.get("observed_eligible"), True) else "ineligible"
                    ),
                    "exclusion_reason": exclusion_reason if not included else "",
                }
            )

        for null_type in PANEL_A_REQUIRED_NULL_TYPES:
            ds_obs_rows = [
                r
                for r in observation_rows
                if r["dataset_id"] == dataset_id and _as_str(r["null_type"]).casefold() == null_type
            ]
            ds_sur_rows = [
                r
                for r in ds_sur
                if _as_str(r.get("null_type")).casefold() == null_type
            ]
            bio_rows = _biological_participant_standardized_means(ds_obs_rows)
            bio_means = np.asarray(
                [
                    _as_float(r.get("standardized_observed_value"))
                    for r in bio_rows
                    if math.isfinite(_as_float(r.get("standardized_observed_value")))
                ],
                dtype=float,
            )
            estimate = float(np.mean(bio_means)) if bio_means.size else float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
            p_value = float("nan")
            if bio_means.size >= 2:
                sd = float(np.std(bio_means, ddof=1))
                se = sd / math.sqrt(float(bio_means.size))
                t_crit = float(stats.t.ppf(0.975, df=int(bio_means.size - 1)))
                ci_low = estimate - t_crit * se
                ci_high = estimate + t_crit * se
                _t_stat, p2 = stats.ttest_1samp(bio_means, popmean=0.0)
                p_value = float(p2)
            n_surrogates_values = {
                _as_int(r.get("n_surrogates")) for r in ds_sur_rows if _as_int(r.get("n_surrogates")) > 0
            }
            n_surr_per = min(n_surrogates_values) if n_surrogates_values else 0
            n_eligible = sum(
                1 for r in ds_obs_rows if _as_str(r.get("eligibility_status")) == "eligible"
            )
            dataset_summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": ds_role,
                    "null_type": null_type,
                    "band": PRIMARY_BAND,
                    "n_biological_participants": len(bio_rows),
                    "n_analysis_units": len(analysis_ids),
                    "n_sessions": len(session_ids),
                    "n_observations": len(ds_obs_rows),
                    "n_eligible_observations": n_eligible,
                    "n_surrogates_per_observation": n_surr_per,
                    "n_total_surrogate_values": len(ds_sur_rows),
                    "display_scale": "standardized_observation_specific_null",
                    "aggregation_method": (
                        "biological_participant_mean_of_standardized_observed_"
                        "observation_condition_session_hierarchy"
                    ),
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "inference_method": "student_t_on_biological_participant_means",
                }
            )

    # Inject dataset_role into full surrogate export.
    for row in panel_a_surrogates:
        ds = _as_str(row.get("dataset_id")).casefold()
        if not _as_str(row.get("dataset_role")):
            row["dataset_role"] = _as_str(dataset_roles.get(ds), "unknown")

    null_dist_csv = source_dir / "figure3_panel_a_null_distributions.csv"
    write_source_csv(
        null_dist_csv,
        panel_a_surrogates,
        (
            "dataset_id",
            "dataset_role",
            "biological_participant_id",
            "analysis_unit_id",
            "subject_id",
            "session_id",
            "observation_id",
            "condition",
            "period",
            "state",
            "task",
            "band",
            "duration",
            "representation",
            "endpoint",
            "null_type",
            "surrogate_index",
            "surrogate_endpoint_index",
            "observed_endpoint_index",
            "standardized_surrogate_value",
            "standardized_observed_value",
            "null_mean",
            "null_median",
            "null_std",
            "empirical_p",
            "rng_seed_u64",
            "n_surrogates",
            "eligibility_status",
            "qc_status",
        ),
    )
    source_paths.append(null_dist_csv)

    # Preferred parquet; always keep CSV as fallback.
    null_dist_parquet = source_dir / "figure3_panel_a_null_distributions.parquet"
    global_parquet = output_dir.parent / "null_surrogate_values.parquet"
    try:
        import pandas as pd  # type: ignore

        frame = pd.DataFrame(panel_a_surrogates)
        frame.to_parquet(null_dist_parquet, index=False)
        frame.to_parquet(global_parquet, index=False)
        source_paths.append(null_dist_parquet)
        source_paths.append(global_parquet)
    except Exception:
        pass

    obs_summary_csv = source_dir / "figure3_panel_a_observation_summary.csv"
    write_source_csv(
        obs_summary_csv,
        observation_rows,
        (
            "dataset_id",
            "dataset_role",
            "biological_participant_id",
            "analysis_unit_id",
            "session_id",
            "observation_id",
            "condition",
            "period",
            "state",
            "band",
            "null_type",
            "observed_endpoint_index",
            "null_mean",
            "null_median",
            "null_std",
            "delta_obs_minus_null",
            "standardized_observed_value",
            "empirical_percentile",
            "empirical_p",
            "n_surrogates",
            "eligibility_status",
            "exclusion_reason",
        ),
    )
    source_paths.append(obs_summary_csv)

    dataset_summary_csv = source_dir / "figure3_panel_a_dataset_summary.csv"
    write_source_csv(
        dataset_summary_csv,
        dataset_summary_rows,
        (
            "dataset_id",
            "dataset_role",
            "null_type",
            "band",
            "n_biological_participants",
            "n_analysis_units",
            "n_sessions",
            "n_observations",
            "n_eligible_observations",
            "n_surrogates_per_observation",
            "n_total_surrogate_values",
            "display_scale",
            "aggregation_method",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "inference_method",
        ),
    )
    source_paths.append(dataset_summary_csv)

    eligibility_csv = source_dir / "figure3_panel_a_dataset_eligibility.csv"
    write_source_csv(
        eligibility_csv,
        eligibility_rows,
        (
            "dataset_id",
            "dataset_role",
            "included",
            "duration_available",
            "endpoint_available",
            "band_available",
            "required_nulls_available",
            "n_eligible_observations",
            "exclusion_reason",
        ),
    )
    source_paths.append(eligibility_csv)

    hierarchy_csv = source_dir / "figure3_panel_a_dataset_hierarchy.csv"
    write_source_csv(
        hierarchy_csv,
        hierarchy_rows,
        (
            "dataset_id",
            "dataset_role",
            "n_biological_participants",
            "n_analysis_units",
            "n_sessions",
            "n_observations",
            "n_eligible_observations",
            "participant_identifier",
            "session_identifier",
            "condition_identifier",
            "aggregation_order",
        ),
    )
    source_paths.append(hierarchy_csv)

    # Plot-layer audit + distribution completeness validation.
    if not panel_a_layer_rows:
        # Ensure required audit table exists even for empty panels.
        for dataset_id in all_dataset_ids:
            for null_type in PANEL_A_REQUIRED_NULL_TYPES:
                panel_a_layer_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "null_type": null_type,
                        "layer": "missing",
                        "source_file": "",
                        "source_column": "",
                        "aggregation_level": "",
                        "n_rows_available": 0,
                        "n_rows_plotted": 0,
                        "weighting": "",
                        "description": "No Panel A geometry rendered for this cell.",
                    }
                )
    plot_layers_csv = source_dir / "figure3_panel_a_plot_layers.csv"
    write_source_csv(
        plot_layers_csv,
        panel_a_layer_rows,
        (
            "dataset_id",
            "null_type",
            "layer",
            "source_file",
            "source_column",
            "aggregation_level",
            "n_rows_available",
            "n_rows_plotted",
            "weighting",
            "description",
        ),
    )
    source_paths.append(plot_layers_csv)

    distribution_validation_rows: list[dict[str, object]] = []
    for dataset_id in all_dataset_ids:
        for null_type in PANEL_A_REQUIRED_NULL_TYPES:
            ds_obs = [
                r
                for r in observation_rows
                if r["dataset_id"] == dataset_id
                and _as_str(r["null_type"]).casefold() == null_type
                and _as_str(r.get("eligibility_status")) == "eligible"
            ]
            ds_sur = [
                r
                for r in panel_a_surrogates
                if _as_str(r.get("dataset_id")).casefold() == dataset_id
                and _as_str(r.get("null_type")).casefold() == null_type
            ]
            n_eligible = len({_as_str(r.get("observation_id")).casefold() for r in ds_obs})
            n_surr_vals = {
                _as_int(r.get("n_surrogates")) for r in ds_sur if _as_int(r.get("n_surrogates")) > 0
            }
            n_per = int(min(n_surr_vals)) if n_surr_vals else 0
            expected = int(n_eligible * n_per) if n_per > 0 else 0
            available = len(ds_sur)
            complete = (
                full_surrogate_distributions
                and n_per >= 2
                and available == expected
                and expected > 0
            )
            distribution_validation_rows.append(
                {
                    "dataset_id": dataset_id,
                    "null_type": null_type,
                    "n_eligible_observations": n_eligible,
                    "n_surrogates_per_observation": n_per,
                    "expected_surrogate_values": expected,
                    "available_surrogate_values": available,
                    "values_used_for_density": available if complete else available,
                    "complete": complete,
                    "used_fallback_summary": used_surrogate_fallback,
                    "downsampling": "none",
                }
            )
    dist_val_csv = source_dir / "figure3_panel_a_distribution_validation.csv"
    write_source_csv(
        dist_val_csv,
        distribution_validation_rows,
        (
            "dataset_id",
            "null_type",
            "n_eligible_observations",
            "n_surrogates_per_observation",
            "expected_surrogate_values",
            "available_surrogate_values",
            "values_used_for_density",
            "complete",
            "used_fallback_summary",
            "downsampling",
        ),
    )
    source_paths.append(dist_val_csv)

    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="null_dataset_forest",
            title="Observed theta ZLPI relative to autocorrelation-preserving nulls",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("null_subject") or ""),
                str(inputs.get("null_surrogate_values") or ""),
            ],
            source_data_csv=str(null_dist_csv),
            analysis_keys=[
                "endpoint=zlpi",
                f"duration={PRIMARY_DURATION_S}",
                f"representation={PRIMARY_REPRESENTATION}",
                f"band={PRIMARY_BAND}",
                "null_types=circular_shift;phase_randomization;block_shuffle",
                "display_scale=standardized_observation_specific_null",
                "null_geometry=standardized_surrogate_value_violin",
                "observed_overlay=biological_participant_means",
                "dataset_mean=equal_weight_biological_participants",
                f"full_surrogate_distributions={str(full_surrogate_distributions).lower()}",
                "dataset_roles=primary_and_sensitivity",
            ],
            notes=(
                "Panel A: gray/colored violin = pooled standardized_surrogate_value "
                "(descriptive pooled observation-specific nulls, not a dataset-level "
                "sampling distribution); gray points = biological-participant means of "
                "standardized_observed_value "
                "(observation→condition→session→biological participant); orange diamond "
                "= equal-weight dataset mean across biological participants; dashed "
                "zero = observation-specific null center. All three nulls use the "
                "configured surrogate count per observation when the C4 surrogate "
                "export is complete."
            ),
        )
    )

    # Panel B: cross-subject specificity + innovations (two stacked subpanels).
    gs_b = gs[0, 1].subgridspec(2, 1, hspace=0.48, height_ratios=[1.05, 1.0])
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[1, 0])
    panel_b_units = _panel_b_series_units(inputs.get("aligned_d240"), band=PRIMARY_BAND)
    n_panel_b_draws = DEFAULT_N_SURROGATES
    if null_rows:
        req = {
            _as_int(row.get("n_surrogates_requested"))
            for row in null_rows
            if _panel_a_slice_null_row(row)
            and _as_str(row.get("null_type")).casefold() == NULL_TYPE_CIRCULAR_SHIFT
            and _as_int(row.get("n_surrogates_requested")) > 1
        }
        if req:
            n_panel_b_draws = int(min(req))
    b_obs_rows, b_draw_rows, b_diag_rows = _panel_b_cross_subject_and_innovation(
        panel_b_units,
        n_null_draws=n_panel_b_draws,
    )
    b_participant_rows = _panel_b_participant_aggregate(b_obs_rows)
    b_group_rows = _panel_b_group_summaries(b_participant_rows, b_obs_rows)
    _plot_panel_b_cross_subject_innovations(ax_b1, ax_b2, b_participant_rows, b_group_rows)

    panel_b_footnote = ""
    if b_group_rows:
        first = b_group_rows[0]
        panel_b_footnote = (
            f"B: paired participant-level cross-subject specificity and AR(1) innovations; "
            f"n_bio={_as_int(first.get('biological_participant_n'))}, "
            f"n_session={_as_int(first.get('session_unit_n'))}, "
            f"draws/obs={n_panel_b_draws}"
        )

    panel_b_obs_csv = source_dir / "figure3_panel_b_observation_estimates.csv"
    write_source_csv(
        panel_b_obs_csv,
        b_obs_rows,
        (
            "dataset_id",
            "biological_participant_id",
            "session_id",
            "condition",
            "period",
            "state",
            "session_type",
            "observation_id",
            "band",
            "duration_s",
            "endpoint_name",
            "representation",
            "correct_endpoint",
            "mean_cross_subject_endpoint",
            "cross_subject_null_mean",
            "cross_subject_null_sd",
            "correct_null_normalized_effect",
            "cross_subject_null_normalized_effect",
            "paired_specificity_contrast_delta_z",
            "innovation_endpoint",
            "innovation_null_mean",
            "innovation_null_sd",
            "innovation_null_normalized_effect",
            "innovation_null_finite_draws",
            "innovation_null_nonfinite_draws",
            "eligibility_flag",
            "exclusion_reason",
        ),
    )
    source_paths.append(panel_b_obs_csv)
    panel_b_part_csv = source_dir / "figure3_panel_b_participant_estimates.csv"
    write_source_csv(
        panel_b_part_csv,
        b_participant_rows,
        (
            "dataset_id",
            "biological_participant_id",
            "band",
            "endpoint_name",
            "aggregation_rule",
            "n_contributing_observations",
            "n_sessions",
            "correct_null_normalized_effect",
            "cross_subject_null_normalized_effect",
            "paired_specificity_contrast_delta_z",
            "innovation_null_normalized_effect",
        ),
    )
    source_paths.append(panel_b_part_csv)
    panel_b_draw_csv = source_dir / "figure3_panel_b_cross_subject_draws.csv"
    write_source_csv(
        panel_b_draw_csv,
        b_draw_rows,
        (
            "dataset_id",
            "observation_id",
            "biological_participant_id",
            "session_id",
            "condition",
            "state",
            "period",
            "session_type",
            "band",
            "duration_s",
            "endpoint_name",
            "draw_index",
            "is_cross_observed_draw",
            "donor_observation_id",
            "donor_biological_participant_id",
            "donor_session_id",
            "donor_period",
            "donor_state",
        ),
    )
    source_paths.append(panel_b_draw_csv)
    panel_b_diag_csv = source_dir / "figure3_panel_b_innovation_diagnostics.csv"
    write_source_csv(
        panel_b_diag_csv,
        b_diag_rows,
        (
            "dataset_id",
            "observation_id",
            "biological_participant_id",
            "session_id",
            "condition",
            "state",
            "period",
            "session_type",
            "band",
            "duration_s",
            "endpoint_name",
            "lag1_hr_before",
            "lag1_eeg_before",
            "lag1_hr_after",
            "lag1_eeg_after",
            "ar1_phi_hr",
            "ar1_phi_eeg",
            "innovation_effective_length",
            "innovation_null_finite_draws",
            "innovation_null_nonfinite_draws",
            "model_failure_flag",
            "model_failure_reason",
        ),
    )
    source_paths.append(panel_b_diag_csv)
    panel_b_group_csv = source_dir / "figure3_panel_b_group_summaries.csv"
    write_source_csv(
        panel_b_group_csv,
        b_group_rows,
        (
            "estimand",
            "band",
            "endpoint_name",
            "estimate",
            "ci_low",
            "ci_high",
            "inferential_n",
            "biological_participant_n",
            "session_unit_n",
            "method",
        ),
    )
    source_paths.append(panel_b_group_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="cross_subject_and_innovations",
            title="Cross-subject specificity and AR(1) innovations",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[
                str(inputs.get("aligned_d240") or ""),
                str(inputs.get("null_subject") or ""),
            ],
            source_data_csv=str(panel_b_group_csv),
            analysis_keys=[
                "panel_b_primary_band=theta",
                "cross_subject=donor_hr_to_focal_eeg",
                f"n_cross_subject_draws={n_panel_b_draws}",
                "innovation_observed=endpoint_on_hr_and_eeg_innovations",
                "innovation_null=circular_shift_eeg_innovations",
                "display_scale=null_normalized_z",
                "unit=biological_participant",
            ],
            notes=(
                "B1 paired plot: participant-level correct simultaneous vs cross-subject HR "
                "mismatch (same dataset/condition/session-type/period/duration/band; donor "
                "must be different biological participant). B2: null-normalized participant "
                "effects for correct simultaneous, cross-subject HR, and AR(1) innovations."
            ),
        )
    )

    # Panel C: mixed duration-specific proximal endpoints (SWPI/MWPI/ZLPI).
    panel_c_endpoint_rows = _panel_c_curve_endpoint_rows(inputs)
    panel_c_observation_rows = _panel_c_observation_level_rows(
        paired_rows,
        panel_c_endpoint_rows,
        dataset_roles=dataset_roles,
    )
    panel_c_summary_rows = _panel_c_summary_rows(panel_c_observation_rows)
    _validate_panel_c_rows(panel_c_summary_rows)
    _plot_panel_c_dataset_trajectories(gs[1, 0], fig, panel_c_summary_rows)

    panel_c_summary_fields = (
        "dataset_id",
        "dataset_role",
        "duration_s",
        "endpoint_name",
        "endpoint_alias",
        "is_standard_zlpi",
        "lag_min_s",
        "lag_max_s",
        "zero_lag_window_s",
        "flank_inner_s",
        "flank_outer_s",
        "duration_contract_id",
        "n_overlap_expected_at_lag_max",
        "band",
        "effect_estimate",
        "ci_low",
        "ci_high",
        "n_observations",
        "n_pairs",
        "n_session_units",
        "n_biological_participants",
        "estimand",
        "contrast_direction",
        "endpoint_formula",
        "eligibility_status",
        "exclusion_reason",
        "uncertainty_method",
        "plotted",
    )
    duration_csv = source_dir / "figure3_panel_c_duration_sensitivity.csv"
    write_source_csv(
        duration_csv,
        panel_c_summary_rows,
        panel_c_summary_fields,
    )
    source_paths.append(duration_csv)
    # Backward-compatible alias retained for existing consumers.
    legacy_duration_csv = source_dir / "figure3_panel_c_duration.csv"
    write_source_csv(
        legacy_duration_csv,
        panel_c_summary_rows,
        panel_c_summary_fields,
    )
    source_paths.append(legacy_duration_csv)
    legacy_panel_b_duration_csv = source_dir / "figure3_panel_b_duration.csv"
    write_source_csv(
        legacy_panel_b_duration_csv,
        panel_c_summary_rows,
        panel_c_summary_fields,
    )
    source_paths.append(legacy_panel_b_duration_csv)
    panel_c_observation_csv = source_dir / "figure3_panel_c_duration_observation_level.csv"
    write_source_csv(
        panel_c_observation_csv,
        panel_c_observation_rows,
        (
            "dataset_id",
            "dataset_role",
            "duration_s",
            "endpoint_name",
            "endpoint_alias",
            "is_standard_zlpi",
            "lag_min_s",
            "lag_max_s",
            "zero_lag_window_s",
            "flank_inner_s",
            "flank_outer_s",
            "duration_contract_id",
            "n_overlap_expected_at_lag_max",
            "band",
            "contrast_id",
            "participant_id",
            "biological_participant_id",
            "session_id",
            "session_unit_id",
            "low_observation_ids",
            "effort_observation_ids",
            "n_low_observations_used",
            "n_effort_observations_used",
            "low_endpoint",
            "effort_endpoint",
            "delta_endpoint",
            "low_zlpi",
            "effort_zlpi",
            "delta_zlpi",
            "estimand",
            "contrast_direction",
            "endpoint_formula",
            "segment_selection_rule",
            "eligibility_status",
            "exclusion_reason",
        ),
    )
    source_paths.append(panel_c_observation_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="duration_sensitivity",
            title="Duration-specific proximal coupling sensitivity",
            endpoint_name="mixed_duration_proximal_endpoints",
            duration_s=0,
            input_tables=[
                str(inputs.get("paired_contrasts") or ""),
                str(inputs.get("curves_d60") or ""),
                str(inputs.get("curves_d120") or ""),
                str(inputs.get("curves_d180") or ""),
                str(inputs.get("curves_d240") or ""),
            ],
            source_data_csv=str(duration_csv),
            analysis_keys=[
                "panel_moved_to=C",
                "resolution=B_mixed_duration_endpoints",
                "d60=swpi",
                "d120=mwpi",
                "d180=zlpi",
                "d240=zlpi",
                "ds003816_only_at_60_swpi=true",
                "unit=biological_participant_clustered",
                "uncertainty=participant_cluster_bootstrap_2000",
            ],
            notes=(
                "Panel C recomputes effort-minus-low-demand proximal coupling under "
                "prespecified duration contracts: SWPI at D60, MWPI at D120, and "
                "standard ZLPI at D180/D240. Endpoint identity is retained; trajectories "
                "are not a single-ZLPI duration effect."
            ),
        )
    )

    # Supplemental export: broadband residualization remains available as
    # a sensitivity table, but is no longer the manuscript Panel D display.
    broadband_source = []
    broadband_rows = [
        row
        for row in (sensitivity_detail or sensitivity)
        if _as_str(row.get("control_id")).casefold() == "broadband_residualized"
    ]
    for row in broadband_rows:
        effect = _as_float(row.get("effect_estimate") or row.get("pooled_effect"))
        status = _as_str(row.get("status")).casefold()
        if status == "control_unavailable":
            continue
        if not math.isfinite(effect):
            continue
        broadband_source.append(
            {
                "control_id": "broadband_residualized",
                "dataset_id": _as_str(row.get("dataset_id"), "pooled"),
                "band": _as_str(row.get("band"), "pooled"),
                "contrast_id": _as_str(row.get("contrast_id")),
                "endpoint_name": _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
                "duration_s": _as_int(row.get("duration_s"), 240),
                "power_representation": _as_str(
                    row.get("power_representation"), "broadband_residualized"
                ),
                "effect_estimate": effect,
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n")),
                "status": _as_str(row.get("status")),
                "can_rescue_primary": False,
            }
        )
    broadband_csv = source_dir / "figure3_sensitivity_broadband_residualization.csv"
    write_source_csv(
        broadband_csv,
        broadband_source,
        (
            "control_id",
            "dataset_id",
            "band",
            "contrast_id",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "effect_estimate",
            "ci_low",
            "ci_high",
            "n",
            "status",
            "can_rescue_primary",
        ),
    )
    source_paths.append(broadband_csv)
    legacy_broadband_csv = source_dir / "figure3_panel_c_broadband.csv"
    write_source_csv(
        legacy_broadband_csv,
        broadband_source,
        (
            "control_id",
            "dataset_id",
            "band",
            "contrast_id",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "effect_estimate",
            "ci_low",
            "ci_high",
            "n",
            "status",
            "can_rescue_primary",
        ),
    )
    source_paths.append(legacy_broadband_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3_supplement",
            panel_id="broadband_residualized_sensitivity",
            title="Broadband residualization sensitivity (supplementary)",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[
                str(inputs.get("sensitivity") or ""),
                str(inputs.get("specification_matrix") or ""),
            ],
            source_data_csv=str(broadband_csv),
            analysis_keys=[
                f"export_category={EXPORT_CATEGORY_SUPPLEMENTARY}",
                "role=sensitivity",
                "control_id=broadband_residualized",
                "can_rescue_primary=false",
            ],
            notes=(
                "Default confirmatory representation sensitivity retained as a "
                "supplementary/sensitivity export. Not displayed as manuscript "
                "Figure 3 Panel D."
            ),
            export_category=EXPORT_CATEGORY_SUPPLEMENTARY,
        )
    )

    # Panel D: locked cardiac-field controls (paired coefficient + Δ display).
    precomputed_obs = read_csv_rows(inputs.get("cardiac_controls_observation"))
    precomputed_dataset_qc = read_csv_rows(inputs.get("cardiac_controls_dataset_qc"))
    precomputed_metadata: dict[str, object] = {}
    metadata_path = inputs.get("cardiac_controls_metadata")
    if metadata_path is not None and metadata_path.is_file():
        try:
            precomputed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            precomputed_metadata = {}

    if precomputed_obs:
        cardiac_panel = compute_panel_d_from_observation_controls(
            precomputed_obs,
            dataset_qc_rows=precomputed_dataset_qc,
            metadata=precomputed_metadata,
        )
    else:
        cardiac_panel = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows,
            protocol_rows=protocol_rows,
            cardiac_qc_rows=cardiac_qc_rows,
        )
    cardiac_paths = write_panel_d_cardiac_control_exports(cardiac_panel, source_dir)
    source_paths.extend(
        [
            cardiac_paths["observations"],
            cardiac_paths["summaries"],
            cardiac_paths["dataset_qc"],
            cardiac_paths["metadata"],
        ]
    )
    _ax_coef, _ax_delta, _ax_status, _plot_controls = _render_figure3_panel_d_paired_axes(
        fig,
        gs[1, 1],
        summaries=cardiac_panel.summaries,
    )
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="cardiac_field_controls",
            title="Cardiac-field and pulse-synchronous controls",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[
                str(inputs.get("cardiac_controls_observation") or ""),
                str(inputs.get("cardiac_controls_dataset_qc") or ""),
                str(inputs.get("cardiac_controls_metadata") or ""),
            ],
            source_data_csv=str(cardiac_paths["summaries"]),
            analysis_keys=[
                "panel=D",
                "control_family=cardiac_field",
                "endpoint=d240_absolute_log10_zlpi",
                "paired_change_display=true",
                "layout=horizontal_paired_estimation",
                "uncertainty=biological_participant_clustered_bootstrap",
                f"plot_control_order={','.join(_plot_controls)}",
            ],
            notes=(
                "Panel D shows locked D240 absolute-log10 ZLPI controlled coefficients "
                "(left) and paired Δ = control − baseline (right) with participant-"
                "clustered bootstrap CIs. Beat-count adjustment is statistical "
                "sensitivity only. PPG controls assess pulse-synchronous contamination "
                "and do not directly test ECG electrical-field leakage. NC rows leave "
                "the plotting area empty and are not treated as zero."
            ),
        )
    )

    fig.suptitle(FIGURE3_TITLE, fontsize=FS_SUPTITLE - 1, fontweight="bold", y=0.978)
    fig.subplots_adjust(**FIGURE3_SUBPLOT_ADJUST)
    if panel_b_footnote:
        # Anchor under Panel B so it does not collide with Panel D annotations.
        fig.text(
            0.72,
            0.022,
            panel_b_footnote,
            ha="center",
            va="bottom",
            fontsize=FS_TICK - 5,
            color=PALETTE["dark_gray"],
        )
    caption_path = output_dir / "figure3_caption.txt"
    caption_path.write_text(
        (
            f"{FIGURE3_TITLE}\n\n"
            "A: Observed theta ZLPI relative to autocorrelation-preserving nulls "
            "(circular shift, phase randomized, block shuffled) at D240 absolute-log10.\n"
            "  - Violin/density = pooled standardized_surrogate_value across eligible "
            "observations (descriptive pooled observation-specific nulls; not a "
            "dataset-level sampling distribution).\n"
            "  - Gray points = biological-participant summaries of "
            "standardized_observed_value (observation→condition→session→biological "
            "participant); not participant-session units.\n"
            "  - Orange diamond = equal-weight mean across biological participants.\n"
            "  - Dashed zero = center of each observation-specific null.\n"
            "  - When the C4 surrogate export is complete, each null uses the "
            "prespecified number of surrogates per eligible observation "
            "(observation count × draws per null method).\n"
            "B: Cross-subject specificity and AR(1) innovations (D240/theta/ZLPI, "
            "absolute-log10). Correct pairing uses simultaneous HR_i with EEG_i. "
            "Cross-subject mismatch substitutes another participant's HR_j (same "
            "dataset/condition/period/session-type/duration; j != i) against EEG_i; "
            "null-normalized effects use observation-specific mismatch draws. "
            "Innovations fit AR(1) separately to HR and EEG and compute endpoint on "
            "aligned residual series; innovation null uses circular shifts of EEG "
            "innovations. Participant aggregation: observation→condition→session→"
            "biological participant; CIs are Student-t on participant means.\n"
            "C: Duration-specific proximal coupling sensitivity by dataset. SWPI was "
            "recomputed from 60-second segments, MWPI from 120-second segments, and "
            "standard ZLPI from 180- and 240-second segments under their "
            "prespecified lag-support and reference-flank contracts. Points show "
            "dataset-specific effort-minus-low-demand endpoint estimates with 95% "
            "biological-participant-clustered bootstrap confidence intervals, and "
            "lines connect supported durations within the same dataset only. "
            "Endpoint identity is shown beneath the duration axis: D60 = SWPI, "
            "D120 = MWPI, and D180/D240 = ZLPI. "
            "ds003816 contributes only to the fully duration-matched 60-second SWPI "
            "analysis and is excluded at 120, 180, and 240 seconds. Because the "
            "endpoint lag support and reference windows differ across durations, "
            "changes across the trajectory reflect both analysis duration and "
            "endpoint definition and should not be interpreted as a pure duration "
            "effect.\n"
            "D: Cardiac-field and pulse-synchronous controls. The left axis shows "
            "controlled D240 absolute_log10 ZLPI coefficients on the Fisher-z scale "
            "with 95% biological-participant-clustered bootstrap confidence intervals; "
            "baseline is marked by an orange diamond on computable rows and a shared "
            "vertical reference. The right axis shows paired changes "
            "Δ = controlled − baseline with the same clustering procedure and a zero "
            "reference. Beat-count adjustment is a statistical sensitivity analysis "
            "and does not remove cardiac artifact. PPG controls assess pulse-synchronous "
            "contamination and do not directly test ECG electrical-field leakage. "
            "The PPG template-subtraction Δ confidence interval includes zero, so "
            "there is no clear evidence that it differs from baseline. ECG-specific "
            "controls are marked NC when no usable ECG signal is available or when "
            "channel reaggregation is unavailable. NC values are missing and are not "
            "represented as zero. Denominators are observation-band rows "
            "(computable / baseline-eligible), not biological participants "
            "(for example 612/628). All values, confidence intervals, and "
            "computability decisions are unchanged for this visualization-only "
            "revision. Similarity across tested controls indicates robustness to "
            "those procedures only and does not establish neural origin or eliminate "
            "all cardiac contamination.\n"
            "Legacy participant-mean observed-minus-null forest is retained as internal "
            "QC and supplementary diagnostics only.\n"
            "Source data: figures/source_data/figure3_panel_*.csv, "
            "figure3_panel_a_plot_layers.csv, "
            "figure3_panel_a_distribution_validation.csv, and "
            "figure3_panel_a_null_distributions.parquet.\n"
        ),
        encoding="utf-8",
    )
    pdf, svg, png = save_figure_trio(
        fig,
        output_dir,
        FIGURE3_STEM,
        bbox_inches=None,
        pad_inches=0.12,
    )

    _supp_trio, supp_paths, supp_panels = _render_figure3_null_supplement(
        null_rows,
        output_dir,
        secondary_records=secondary_records,
    )
    supplement_source_paths.extend(supp_paths)
    panel_sources.extend(supp_panels)
    for path in _supp_trio:
        supplement_source_paths.append(path)

    _part_trios, part_paths, part_panels = _render_figure3_participant_forest_qc(
        primary_analysis,
        output_dir,
        include_internal_qc=include_internal_qc,
    )
    qc_paths.extend(part_paths)
    panel_sources.extend(part_panels)
    for trio in _part_trios:
        qc_paths.extend(trio)

    audit_note = source_dir / "figure3_panel_a_AUDIT_NOTE.md"
    audit_note.write_text(
        "\n".join(
            [
                "# Figure 3 Panel A — estimand note",
                "",
                "## Main panel display",
                "Main Figure 3 Panel A displays pooled observation-specific empirical null",
                "distributions (standardized_surrogate_value) for circular_shift,",
                "phase_randomization, and block_shuffle (D240/theta/absolute_log10/ZLPI).",
                "Observed overlays are biological-participant means of",
                "standardized_observed_value; the orange diamond is the equal-weight",
                "mean across biological participants. Pooled surrogate densities are",
                "descriptive and are not dataset-level sampling distributions.",
                "See `figure3_panel_a_plot_layers.csv` and",
                "`figure3_panel_a_distribution_validation.csv`.",
                "",
                "## Retained contrast forest (not main Panel A)",
                "The prior observed-minus-null participant-mean contrast forest is retained in",
                f"`{FIGURE3_INTERNAL_QC_SUBDIR}/{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_*` and supplementary diagnostics,",
                "and remains valid for its own estimand.",
                "",
                "## Independent unit (final verdict)",
                INDEPENDENT_UNIT_VERDICT,
                "",
                "See `figure3_panel_a_independent_unit_verdict.csv` for per-dataset notes.",
                "",
                "## Estimand",
                "For the prespecified slice D240 / absolute_log10 / theta / "
                "circular_shift / zlpi, each eligible matched observation "
                "contributes δ = observed_endpoint_index − null_mean.",
                "Condition-level Δ_{p,c} = mean(δ) within biological participant × condition.",
                "Biological participant Δ_p = unweighted mean of that participant's Δ_{p,c}.",
                "Main Panel A: one row per dataset = unweighted mean of Δ_p with "
                "Student-t 95% CI (df = n_participants − 1).",
                "",
                "## Sample size wording",
                f"- This render: **{primary_inference.sample_size_label}**",
                "- When protocol session conditions are separate analysis units "
                "(Panel B-aligned), n uses session-condition identifiers rather "
                "than biological participants alone.",
                "",
                "## CI method",
                "Unweighted mean of biological-participant Δ_p within dataset; "
                "Student-t 95% CI, df = n_participants − 1.",
                "No cross-dataset pooled estimate is plotted (no prespecified "
                "Panel A meta-analytic weighting).",
                "",
                "## Interpretation (this render)",
                f"- run_class: `{primary_inference.run_class}`",
                f"- n_participants (biological): {primary_inference.n_participants}",
                f"- n_participant_conditions: {primary_inference.n_participant_conditions}",
                f"- mean Δ: {primary_inference.mean_delta}",
                f"- 95% CI: [{primary_inference.ci_low}, {primary_inference.ci_high}]",
                f"- category: **{primary_inference.interpretation}**",
                "",
                "## Export categories",
                f"- Manuscript main figure: `{FIGURE3_STEM}`",
                f"- Supplementary: `{FIGURE3_SUPPLEMENT_STEM}`",
                f"- Internal QC (excluded from manuscript/supplement exports): "
                f"`{FIGURE3_INTERNAL_QC_SUBDIR}/{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_*`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manuscript_source_paths.append(audit_note)

    # Tag / order panels by export category.
    ordered_panels: list[FigurePanelSource] = []
    for panel in panel_sources:
        if panel.figure_id == "figure3":
            keys = list(panel.analysis_keys)
            flag = f"export_category={EXPORT_CATEGORY_MANUSCRIPT}"
            if flag not in keys:
                keys.append(flag)
            ordered_panels.append(
                replace(
                    panel,
                    export_category=EXPORT_CATEGORY_MANUSCRIPT,
                    analysis_keys=keys,
                )
            )
        else:
            ordered_panels.append(panel)

    manuscript = FigureArtifacts(
        figure_id="figure3",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(manuscript_source_paths),
        panels=tuple(p for p in ordered_panels if p.export_category == EXPORT_CATEGORY_MANUSCRIPT),
    )
    supplement = FigureArtifacts(
        figure_id="figure3_supplement",
        pdf=_supp_trio[0],
        svg=_supp_trio[1],
        png=_supp_trio[2],
        source_csvs=tuple(supplement_source_paths),
        panels=tuple(
            p for p in ordered_panels if p.export_category == EXPORT_CATEGORY_SUPPLEMENTARY
        ),
    )
    return Figure3RenderResult(
        manuscript=manuscript,
        supplement=supplement,
        internal_qc_paths=tuple(qc_paths),
        panels=tuple(ordered_panels),
    )


def write_figure_export_categories(output_dir: Path) -> Path:
    """Write manuscript / supplementary / internal-QC figure export map."""
    rows = [
        {
            "stem": FIGURE1_STEM,
            "figure_id": "figure1",
            "export_category": EXPORT_CATEGORY_MANUSCRIPT,
            "include_in_manuscript_export": True,
            "include_in_supplementary_export": False,
            "relative_path_glob": f"{FIGURE1_STEM}.*",
            "notes": "Main manuscript Figure 1",
        },
        {
            "stem": FIGURE2_STEM,
            "figure_id": "figure2",
            "export_category": EXPORT_CATEGORY_MANUSCRIPT,
            "include_in_manuscript_export": True,
            "include_in_supplementary_export": False,
            "relative_path_glob": f"{FIGURE2_STEM}.*",
            "notes": "Main manuscript Figure 2",
        },
        {
            "stem": FIGURE3_STEM,
            "figure_id": "figure3",
            "export_category": EXPORT_CATEGORY_MANUSCRIPT,
            "include_in_manuscript_export": True,
            "include_in_supplementary_export": False,
            "relative_path_glob": f"{FIGURE3_STEM}.*",
            "notes": "Only main manuscript Figure 3 (panels A–D)",
        },
        {
            "stem": "figure3_panel_e_nuisance_modality",
            "figure_id": "figure3_panel_e",
            "export_category": EXPORT_CATEGORY_MANUSCRIPT,
            "include_in_manuscript_export": True,
            "include_in_supplementary_export": False,
            "relative_path_glob": "figure3_panel_e_nuisance_modality.*",
            "notes": "Standalone Figure 3 Panel E nuisance/modality robustness sheet",
        },
        {
            "stem": PANEL_F_STEM,
            "figure_id": "figure3_panel_f",
            "export_category": EXPORT_CATEGORY_MANUSCRIPT,
            "include_in_manuscript_export": True,
            "include_in_supplementary_export": False,
            "relative_path_glob": f"{PANEL_F_STEM}.*",
            "notes": "Standalone Figure 3 Panel F topography and gamma specificity sheet",
        },
        {
            "stem": FIGURE3_SUPPLEMENT_STEM,
            "figure_id": "figure3_supplement",
            "export_category": EXPORT_CATEGORY_SUPPLEMENTARY,
            "include_in_manuscript_export": False,
            "include_in_supplementary_export": True,
            "relative_path_glob": f"{FIGURE3_SUPPLEMENT_STEM}.*",
            "notes": "Supplementary null diagnostics (S1–S5)",
        },
        {
            "stem": FIGURE3_QC_PARTICIPANT_FOREST_STEM,
            "figure_id": "figure3_internal_qc",
            "export_category": EXPORT_CATEGORY_INTERNAL_QC,
            "include_in_manuscript_export": False,
            "include_in_supplementary_export": False,
            "relative_path_glob": f"{FIGURE3_INTERNAL_QC_SUBDIR}/{FIGURE3_QC_PARTICIPANT_FOREST_STEM}_*",
            "notes": (
                "Dataset-specific participant null forests; internal QC only; "
                "exclude from manuscript and supplementary exports unless "
                "explicitly requested"
            ),
        },
    ]
    path = output_dir / FIGURE_EXPORT_CATEGORIES_FILENAME
    write_source_csv(
        path,
        rows,
        (
            "stem",
            "figure_id",
            "export_category",
            "include_in_manuscript_export",
            "include_in_supplementary_export",
            "relative_path_glob",
            "notes",
        ),
    )
    return path


def render_figure3_panel_e(
    inputs: Mapping[str, Path | None],
    output_dir: str | Path,
    *,
    include_internal_qc: bool = True,
) -> dict[str, object]:
    """Render standalone Figure 3 Panel E nuisance/modality robustness sheet."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paired_rows = read_csv_rows(inputs.get("paired_contrasts"))
    aligned_rows = read_csv_rows(inputs.get("aligned_d240"))
    data_audit_rows = read_csv_rows(inputs.get("data_audit"))
    peak_qc_rows = read_csv_rows(inputs.get("cardiac_peak_qc"))
    protocol_rows = read_csv_rows(inputs.get("protocol_audit"))
    endpoint_rows = read_csv_rows(inputs.get("endpoints_d240"))
    result = compute_panel_e_nuisance_modality(
        paired_rows=paired_rows,
        aligned_rows=aligned_rows,
        data_audit_rows=data_audit_rows,
        peak_qc_rows=peak_qc_rows,
        protocol_rows=protocol_rows,
        endpoint_rows=endpoint_rows,
    )
    paths = render_panel_e_figure(
        result,
        out,
        sample_scheme="common_sample",
        include_internal_qc=include_internal_qc,
    )
    panel_source = FigurePanelSource(
        figure_id="figure3_panel_e",
        panel_id="nuisance_modality_robustness",
        title="Figure 3E | Nuisance and modality robustness",
        endpoint_name="zlpi",
        duration_s=240,
        source_data_csv=str(paths.get("specifications") or ""),
        input_tables=[
            str(inputs.get("paired_contrasts") or ""),
            str(inputs.get("aligned_d240") or ""),
            str(inputs.get("data_audit") or ""),
            str(inputs.get("cardiac_peak_qc") or ""),
            str(inputs.get("protocol_audit") or ""),
            str(inputs.get("endpoints_d240") or ""),
        ],
        notes=(
            f"export_category={EXPORT_CATEGORY_MANUSCRIPT}; "
            "standalone sheet; within-pair Δ-nuisance primary display"
        ),
        export_category=EXPORT_CATEGORY_MANUSCRIPT,
    )
    paths["panel_source"] = panel_source
    paths["result"] = result
    return paths


def render_figure3_panel_f(
    inputs: Mapping[str, Path | None],
    output_dir: str | Path,
    *,
    confirmatory_root: str | Path | None = None,
    include_internal_qc: bool = True,
    force_recompute_channel_zlpi: bool = False,
) -> dict[str, object]:
    """Render standalone Figure 3 Panel F topography / gamma specificity sheet."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paired_rows = read_csv_rows(inputs.get("paired_contrasts"))
    root = Path(confirmatory_root).expanduser().resolve() if confirmatory_root else out.parent
    # Prefer C7 root when output is .../C7/figures
    if root.name == "figures":
        root = root.parent
    result = compute_panel_f_topography(
        confirmatory_root=root,
        paired_rows=paired_rows,
        force_recompute_channel_zlpi=force_recompute_channel_zlpi,
    )
    paths = render_panel_f_figure(
        result,
        out,
        include_internal_qc=include_internal_qc,
    )
    panel_source = FigurePanelSource(
        figure_id="figure3_panel_f",
        panel_id="topography_gamma_specificity",
        title="Figure 3F | Topography and gamma specificity",
        endpoint_name="zlpi",
        duration_s=240,
        source_data_csv=str(paths.get("summary") or ""),
        input_tables=[
            str(inputs.get("paired_contrasts") or ""),
        ],
        notes=(
            f"export_category={EXPORT_CATEGORY_MANUSCRIPT}; "
            "standalone sheet; channel-level ZLPI topography from C1a+C1b"
        ),
        export_category=EXPORT_CATEGORY_MANUSCRIPT,
    )
    paths["panel_source"] = panel_source
    paths["result"] = result
    return paths


def generate_confirmatory_figures(
    confirmatory_root: str | Path,
    output_dir: str | Path,
    *,
    include_internal_qc: bool = True,
) -> FiguresResult:
    """Generate Figures 1–3 and the figure-source manifest from frozen outputs."""
    root = Path(confirmatory_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = resolve_reporting_inputs(root)

    figure1 = render_figure1(inputs, out)
    figure2 = render_figure2(inputs, out)
    figure3_bundle = render_figure3(
        inputs, out, include_internal_qc=include_internal_qc
    )
    figure3 = figure3_bundle.manuscript
    panel_e_paths = render_figure3_panel_e(
        inputs, out, include_internal_qc=include_internal_qc
    )
    panel_f_paths = render_figure3_panel_f(
        inputs,
        out,
        confirmatory_root=root,
        include_internal_qc=include_internal_qc,
    )
    panels = figure1.panels + figure2.panels + figure3_bundle.panels
    panel_e_source = panel_e_paths.get("panel_source")
    if panel_e_source is not None:
        panels = panels + (panel_e_source,)
    panel_f_source = panel_f_paths.get("panel_source")
    if panel_f_source is not None:
        panels = panels + (panel_f_source,)
    export_categories = write_figure_export_categories(out)

    # Hash the frozen inputs actually referenced by panels.
    input_hash_map: dict[str, str] = {}
    for panel in panels:
        for table in panel.input_tables:
            if not table:
                continue
            path = Path(table)
            if path.is_file():
                input_hash_map[str(path)] = sha256_file(path)
    # Also hash discovered confirmatory CSVs for completeness.
    for record in hash_directory_files(root, patterns=("*.csv",)):
        input_hash_map.setdefault(record.relative_path, record.sha256)

    manifest_payload = build_figure_source_manifest(
        panels,
        input_hashes=input_hash_map,
        output_dir=out,
    )
    return FiguresResult(
        figure1=figure1,
        figure2=figure2,
        figure3=figure3,
        figure3_supplement=figure3_bundle.supplement,
        figure3_internal_qc=figure3_bundle.internal_qc_paths,
        figure_export_categories=export_categories,
        figure_source_manifest=Path(manifest_payload["manifest_path"]),
        panel_records=panels,
    )


__all__ = [
    "EXPORT_CATEGORY_INTERNAL_QC",
    "EXPORT_CATEGORY_MANUSCRIPT",
    "EXPORT_CATEGORY_SUPPLEMENTARY",
    "FIGURE1_STEM",
    "FIGURE2_STEM",
    "FIGURE3_STEM",
    "FIGURE3_SUPPLEMENT_STEM",
    "FIGURE3_QC_PARTICIPANT_FOREST_STEM",
    "FIGURE3_PARTICIPANT_FOREST_STEM",
    "FIGURE3_INTERNAL_QC_SUBDIR",
    "FIGURE_EXPORT_CATEGORIES_FILENAME",
    "FIGURE_DPI",
    "FIGURE2_PANEL_AB_DISPLAY_SMOOTH_NOTE",
    "FIGURE2_PANEL_AB_DISPLAY_SMOOTH_SIGMA_S",
    "FIGURE2_PANEL_A_SENSITIVITY_NOTE",
    "FIGURE2_PANEL_B_SENSITIVITY_NOTE",
    "FIGURE2_PANEL_E_SENSITIVITY_NOTE",
    "FIGURE2_PANEL_A_HIIT_SENSITIVITY_NOTE",  # deprecated alias
    "FIGURE2_PANEL_B_HIIT_SENSITIVITY_NOTE",  # deprecated alias
    "FIGURE2_PANEL_E_HIIT_SENSITIVITY_NOTE",  # deprecated alias
    "LAG_CURVE_DISPLAY_SMOOTH_NOTE",
    "LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S",
    "FigureArtifacts",
    "Figure3RenderResult",
    "FiguresResult",
    "display_lag_mask",
    "format_role_display",
    "gaussian_smooth_display_series",
    "generate_confirmatory_figures",
    "mean_ci_by_lag",
    "render_figure1",
    "render_figure3",
    "render_figure3_panel_e",
    "render_figure3_panel_f",
    "resolve_reporting_inputs",
    "save_figure_trio",
    "sensitivity_display_title",
    "write_figure_export_categories",
]
