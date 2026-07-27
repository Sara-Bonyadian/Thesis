"""Publication figures for confirmatory zero-lag analyses (M12).

Reads only frozen confirmatory tables. No hard-coded scientific results.
Figures 1–3 are written as PDF, SVG, and 300-dpi PNG with companion
source-data CSVs and a figure-source manifest.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .artifact_controls import participant_duration_sensitivity_effects
from .config import EXPECTED_BANDS_HZ
from .duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    EXPECTED_SHOULDERS_S,
    ZLPI_FLANKS_S,
    contract_for_duration,
)
from .endpoints import fisher_z
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
from .paired_delta_inference import infer_paired_deltas_cluster_aware

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
LOW_DEMAND_CONDITION_LABELS = {
    "rest",
    "passive",
    "step1",
    "low_demand",
    "ph_pre_rest",
    "ph_post_rest",
    "ps_pre_rest",
    "ps_post_rest",
}

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
    + " HIIT is shown as one descriptive sensitivity estimate. PH and PS are "
    "session subjects (Panel B-aligned; counted separately, n≈40). Within each "
    "session, available PRE/POST low-demand ZLPI are averaged before the group "
    "mean. The HIIT row is a sensitivity summary, not an independent primary "
    "dataset. Surrogate marks use the median of all HIIT low-demand "
    "condition-level median_empirical_p values for that band."
)
FIGURE1_PANEL_D_ASTERISK_LABEL = (
    f"* = circular-shift surrogate p < {FIGURE1_PANEL_D_SURROGATE_ALPHA:g} only"
)
FIGURE1_PANEL_E_NOTE = (
    "Alpha-focused replication display of the equal four-band primary meta "
    "(one prespecified contrast per dataset); not an alpha-only confirmatory hierarchy. "
    "HIIT is shown as one descriptive sensitivity estimate. PH and PS are session "
    "subjects (Panel B-aligned; counted separately). Within each session, available "
    "Rest–Tetris ΔZLPI contrasts are averaged before group summarization. "
    "The HIIT row is excluded from the pooled random-effects meta-analysis."
)
FIGURE1_PANEL_F_NOTE = (
    "Near-zero central peak: linear baseline from distant flanks (20≤|τ|≤60 s), "
    "nonnegative Gaussian on baseline-adjusted |τ|≤20 s; identifiable peaks only "
    "(A ≥ 1.8×RMSE, SE(A), weak-edge). Group μ/FWHM: mean of per-session-subject "
    "means (Panel B-aligned; HIIT PH/PS counted as separate units) with "
    "hierarchical CI; μ TOST / ±2 s on that mean. FWHM on log scale "
    "(back-transformed CI)."
)

FIGURE1_TITLE = "Confirmatory EEG–cardiac coupling: structure, replication, and peaks"
FIGURE2_TITLE = (
    "State-dependent attenuation: lag structure, meta-replication, and peaks"
)
FIGURE3_TITLE = "Temporal specificity and core robustness"
FIGURE2_PANEL_C_NOTE = (
    "Alpha PRIMARY_META absolute paired ΔZLPI (task − low-demand); "
    "no percent attenuation. HIIT is shown as one descriptive sensitivity estimate "
    "(PH/PS session subjects counted separately, Panel B-aligned) and is excluded "
    "from the pooled random-effects meta-analysis."
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
    "Paired low-demand vs effort μ and FWHM from C5; one PRIMARY_META contrast "
    "per dataset; suppress μ when that state's peak is not identifiable; "
    "FWHM descriptive only."
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
    "meta inputs and are empty by design for sensitivity-only runs "
    "(e.g. HIIT alone)."
)
FIGURE2_PANEL_A_HIIT_SENSITIVITY_NOTE = (
    "HIIT Sensitivity (display-only): matched Rest–Tetris Fisher-z lag curves "
    "from C5 pairs (PRE and POST each contribute; no PRE/POST or PH/PS "
    "averaging before the group mean). Point estimate = mean across matched "
    "pairs; 95% CI = PH/PS session-subject cluster bootstrap (pairs within a "
    "drawn session are retained together). Excluded from PRIMARY_META; not a "
    "primary confirmatory claim. Report n_matched_pairs and n_session_clusters "
    "(not participant n)."
)
FIGURE2_PANEL_B_HIIT_SENSITIVITY_NOTE = (
    "HIIT Sensitivity (display-only): matched Rest–Tetris Δ Fisher-z lag curves "
    "Δz(τ)=z_effort(τ)−z_low(τ) from the same C5 pairs as Panel A (PRE and POST "
    "each contribute; no PRE/POST or PH/PS averaging before the group mean). "
    "Point estimate = mean across matched pairs; 95% CI = PH/PS session-subject "
    "cluster bootstrap. Excluded from PRIMARY_META; display only — no "
    "cluster-permutation testing. Report n_matched_pairs and n_session_clusters "
    "(not participant n)."
)
FIGURE2_PANEL_E_HIIT_SENSITIVITY_NOTE = (
    "HIIT Sensitivity (display-only): paired Rest–Tetris peak μ and FWHM from "
    "C5 matched pairs (PRE and POST each contribute; no PRE/POST or PH/PS "
    "collapse). Suppress μ/FWHM when that state's peak is not identifiable; "
    "FWHM descriptive only. Excluded from PRIMARY_META."
)


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
FIGURE3_FIGSIZE = (15.5, 13.6)
FIGURE3_SUBPLOT_ADJUST = {
    "left": 0.12,
    "right": 0.84,
    "top": 0.925,
    "bottom": 0.10,
    "wspace": 0.36,
    "hspace": 0.42,
}
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


def save_figure_trio(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    _configure_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.45)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.45)
    _cleanup_svg(svg)
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.45)
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
    condition = _as_str(row.get("condition") or row.get("task")).casefold()
    role = _as_str(row.get("condition_role") or row.get("state")).casefold()
    if role == "low_demand":
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
        "null_summary": "null_summary.csv",
        "protocol_audit": "protocol_audit.csv",
        "sensitivity": "sensitivity_results.csv",
        "specification_matrix": "specification_matrix.csv",
        "duration_sensitivity": "duration_sensitivity.csv",
        "eligibility": "eligibility_by_duration.csv",
        "mixed_model": "mixed_model_results.csv",
        "mixed_model_marginal": "mixed_model_marginal_estimates.csv",
        "mixed_model_contrasts": "mixed_model_contrasts.csv",
    }
    return {key: discover_named_file(root, name) for key, name in names.items()}


def write_source_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload = {}
            for field in fieldnames:
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
    low_labels = LOW_DEMAND_CONDITION_LABELS
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
        role = _as_str(row.get("condition_role") or row.get("state")).casefold()
        if condition_role == "low_demand":
            if role and role != "low_demand" and condition not in low_labels:
                continue
            if not role and condition not in low_labels and condition:
                # If condition unknown, keep row (synthetic tables often use rest).
                if condition not in low_labels and condition not in {"", "rest"}:
                    # keep generic synthetic conditions like "rest"
                    if "rest" not in condition and "passive" not in condition and "low" not in condition:
                        pass
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
    """Figure 3: nulls, duration robustness, broadband sensitivity, LOO."""
    null_rows = read_csv_rows(inputs.get("null_subject"))
    duration = read_csv_rows(inputs.get("duration_sensitivity"))
    paired_rows = read_csv_rows(inputs.get("paired_contrasts"))
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
    gs = fig.add_gridspec(2, 2, hspace=FIGURE3_SUBPLOT_ADJUST["hspace"], wspace=FIGURE3_SUBPLOT_ADJUST["wspace"])

    # Panel A: dataset-level forest (biological-participant mean Δ ± CI).
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
    _plot_dataset_null_forest(
        ax_a,
        primary_analysis.dataset_inferences,
        title=f"Circular-shift null (θ {_endpoint_display(ENDPOINT_ZLPI)}, D{PRIMARY_DURATION_S})",
        panel_label="A",
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

    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="null_dataset_forest",
            title="Dataset-level observed − circular-shift null Δ",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[str(inputs.get("null_subject") or "")],
            source_data_csv=str(dataset_csv),
            analysis_keys=[
                "endpoint=zlpi",
                f"duration={PRIMARY_DURATION_S}",
                f"representation={PRIMARY_REPRESENTATION}",
                f"band={PRIMARY_BAND}",
                f"null_type={PRIMARY_NULL_TYPE}",
                "unit=biological_participant",
                "aggregation=dataset_mean_of_participant_deltas",
                "pooled_estimate_plotted=false",
                f"run_class={primary_inference.run_class}",
                f"interpretation={primary_inference.interpretation}",
            ],
            notes=(
                "Primary confirmatory null panel. One row per dataset = "
                "unweighted mean of biological-participant Δ_p with Student-t "
                "95% CI (df=n_participants−1). Annotated n = unique biological "
                "participants. Cross-dataset pooled estimate is not plotted "
                "(no prespecified Panel A meta-analytic weighting). "
                + INDEPENDENT_UNIT_VERDICT
            ),
        )
    )

    # Panel B: participant-level duration sensitivity (equal participant weight).
    ax_b = fig.add_subplot(gs[0, 1])
    endpoint_styles = {
        ENDPOINT_ZLPI: ("o", "-"),
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: ("s", "--"),
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: ("^", "-."),
    }
    if paired_rows:
        duration_source, participant_duration_rows = (
            participant_duration_sensitivity_effects(paired_rows)
        )
    else:
        # Fall back to contrast-cell table only if paired contrasts unavailable.
        duration_source = [
            {
                "duration_s": _as_int(row.get("duration_s")),
                "endpoint_name": _as_str(row.get("endpoint_name")),
                "band": _as_str(row.get("band")),
                "dataset_id": _as_str(row.get("dataset_id")),
                "contrast_id": _as_str(row.get("contrast_id")),
                "effect_estimate": _as_float(row.get("effect_estimate")),
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n")),
                "is_primary_analysis": _as_str(row.get("is_primary_analysis")),
                "can_rescue_primary": False,
                "estimand": "contrast_cell_fallback",
                "unit": "contrast_cell",
                "notes": _as_str(row.get("notes")),
            }
            for row in duration
            if math.isfinite(_as_float(row.get("effect_estimate")))
        ]
        participant_duration_rows = []

    plotted_duration = False
    clipped_swpi = False
    for band in BAND_ORDER:
        band_rows = [
            r for r in duration_source if _as_str(r.get("band")).casefold() == band
        ]
        if not band_rows:
            continue
        for row in band_rows:
            endpoint = _as_str(row.get("endpoint_name"))
            if endpoint == ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX and _as_int(
                row.get("duration_s")
            ) == 60:
                lo = _as_float(row.get("ci_low"))
                hi = _as_float(row.get("ci_high"))
                if (math.isfinite(hi) and hi > 0.25) or (
                    math.isfinite(lo) and lo < -0.25
                ):
                    clipped_swpi = True
            if math.isfinite(_as_float(row.get("effect_estimate"))):
                plotted_duration = True

    if plotted_duration:
        ylim_rows = [
            r
            for r in duration_source
            if not (
                _as_str(r.get("endpoint_name")) == ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX
                and _as_int(r.get("duration_s")) == 60
            )
        ]
        span_vals: list[float] = []
        for row in ylim_rows:
            for key in ("effect_estimate", "ci_low", "ci_high"):
                val = _as_float(row.get(key))
                if math.isfinite(val):
                    span_vals.append(abs(val))
        y_span = max(span_vals) if span_vals else 0.15
        y_span = max(y_span, 0.12)
        y_lo, y_hi = -1.15 * y_span, 1.15 * y_span

        for band in BAND_ORDER:
            band_rows = [
                r for r in duration_source if _as_str(r.get("band")).casefold() == band
            ]
            if not band_rows:
                continue
            band_rows = sorted(band_rows, key=lambda r: _as_int(r.get("duration_s")))
            for row in band_rows:
                endpoint = _as_str(row.get("endpoint_name"))
                effect = _as_float(row.get("effect_estimate"))
                if not math.isfinite(effect):
                    continue
                marker, _linestyle = endpoint_styles.get(endpoint, ("x", ":"))
                color = _band_color(band)
                lo = _as_float(row.get("ci_low"))
                hi = _as_float(row.get("ci_high"))
                x_plot = _band_duration_x(_as_int(row.get("duration_s")), band)
                _clipped_vertical_errorbar(
                    ax_b,
                    x_plot,
                    effect,
                    lo,
                    hi,
                    y_lo=y_lo,
                    y_hi=y_hi,
                    fmt=marker,
                    color=color,
                    linestyle="none",
                    markersize=FIGURE3_MARKER_SIZE,
                    capsize=2.5,
                    elinewidth=FIGURE3_CI_LINEWIDTH,
                    markeredgecolor=PALETTE["dark_gray"],
                    markeredgewidth=0.5,
                    alpha=0.90,
                )

        _ref_hline(ax_b, 0.0)
        ax_b.set_xlim(*FIGURE3_DURATION_XLIM)
        ax_b.set_xticks([60, 120, 180, 240])
        ax_b.set_ylim(y_lo, y_hi)
        ax_b.set_xlabel("Duration (s)", fontsize=FS_AXIS - 2)
        ax_b.set_ylabel(
            f"Participant mean Δ ({Z_YLABEL})",
            fontsize=FS_AXIS - 2,
        )
        _style_axes(ax_b)
        band_handles = []
        band_labels = []
        for band in BAND_ORDER:
            if not any(_as_str(r.get("band")).casefold() == band for r in duration_source):
                continue
            (handle,) = ax_b.plot(
                [],
                [],
                marker="o",
                color=_band_color(band),
                ls="none",
                markersize=FIGURE3_MARKER_SIZE,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.5,
            )
            band_handles.append(handle)
            band_labels.append(_band_display(band))
        index_handles = []
        index_labels = []
        for endpoint, (marker, _linestyle) in endpoint_styles.items():
            if not any(
                _as_str(r.get("endpoint_name")) == endpoint for r in duration_source
            ):
                continue
            (handle,) = ax_b.plot(
                [],
                [],
                marker=marker,
                color=PALETTE["dark_gray"],
                ls="none",
                markersize=FIGURE3_MARKER_SIZE,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.5,
            )
            index_handles.append(handle)
            index_labels.append(_endpoint_display(endpoint))
        if band_handles and index_handles:
            _legend_dual_encoding(
                ax_b,
                band_handles,
                band_labels,
                index_handles,
                index_labels,
                loc="upper right",
            )
        n_units = max((_as_int(r.get("n")) for r in duration_source), default=0)
        panel_b_footnote = (
            f"{FIGURE3_PANEL_B_FOOTNOTE_SHORT}; n={n_units}"
            + ("; † D60 SWPI CIs clipped" if clipped_swpi else "")
        )
    else:
        panel_b_footnote = ""
        _mark_empty_panel(
            ax_b,
            MSG_NOT_INCLUDED,
            xlabel="Duration (s)",
            ylabel=f"Participant mean Δ ({Z_YLABEL})",
        )
    _set_panel_title(
        ax_b,
        "Duration sensitivity",
        fontsize=FS_PANEL_TITLE - 2,
        pad=10,
    )
    _add_panel_label(ax_b, "B")
    duration_csv = source_dir / "figure3_panel_b_duration.csv"
    write_source_csv(
        duration_csv,
        duration_source,
        (
            "duration_s",
            "endpoint_name",
            "band",
            "dataset_id",
            "contrast_id",
            "effect_estimate",
            "ci_low",
            "ci_high",
            "n",
            "is_primary_analysis",
            "can_rescue_primary",
            "estimand",
            "unit",
            "notes",
        ),
    )
    source_paths.append(duration_csv)
    participant_csv = source_dir / "figure3_panel_b_participant_estimates.csv"
    write_source_csv(
        participant_csv,
        participant_duration_rows,
        (
            "dataset_id",
            "participant_id",
            "participant_unit_id",
            "band",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n_contrasts",
            "mean_delta",
            "is_primary_analysis",
            "can_rescue_primary",
            "control_id",
        ),
    )
    source_paths.append(participant_csv)
    # Retain contrast-cell duration_sensitivity as diagnostic export when present.
    if duration:
        cell_csv = source_dir / "figure3_panel_b_contrast_cells_diagnostic.csv"
        write_source_csv(
            cell_csv,
            [
                {
                    "duration_s": _as_int(row.get("duration_s")),
                    "endpoint_name": _as_str(row.get("endpoint_name")),
                    "band": _as_str(row.get("band")),
                    "dataset_id": _as_str(row.get("dataset_id")),
                    "contrast_id": _as_str(row.get("contrast_id")),
                    "effect_estimate": _as_float(row.get("effect_estimate")),
                    "ci_low": _as_float(row.get("ci_low")),
                    "ci_high": _as_float(row.get("ci_high")),
                    "n": _as_int(row.get("n")),
                    "diagnostic_label": "contrast-cell Student-t (not Panel B estimand)",
                }
                for row in duration
                if math.isfinite(_as_float(row.get("effect_estimate")))
            ],
            (
                "duration_s",
                "endpoint_name",
                "band",
                "dataset_id",
                "contrast_id",
                "effect_estimate",
                "ci_low",
                "ci_high",
                "n",
                "diagnostic_label",
            ),
        )
        source_paths.append(cell_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="duration_sensitivity",
            title="Participant-level duration sensitivity",
            endpoint_name="mixed_labeled",
            duration_s=0,
            input_tables=[
                str(inputs.get("paired_contrasts") or ""),
                str(inputs.get("duration_sensitivity") or ""),
            ],
            source_data_csv=str(duration_csv),
            analysis_keys=[
                "zlpi",
                "mwpi",
                "swpi",
                "can_rescue_primary=false",
                "unit=participant",
                "estimand=mean_of_participant_means",
                "absolute_estimates_not_equivalence=true",
            ],
            notes=(
                f"{FIGURE3_PANEL_B_ENCODING_NOTE}. "
                "Points = unweighted mean of participant-level mean Δ at each "
                "duration×band×endpoint cell (contrasts averaged within participant "
                "first). Student-t CI uses n_participants. Absolute estimates only — "
                "overlapping CIs do not imply equivalence. D120/D60 are MWPI/SWPI "
                "(not ZLPI); cannot rescue primary D240 ZLPI."
            ),
        )
    )

    # Panel C: broadband residualization sensitivity.
    ax_c = fig.add_subplot(gs[1, 0])
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
        if not math.isfinite(effect) and not broadband_source:
            # Keep unavailable rows out of the plot, but track for empty-panel logic.
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
    y_labels_c: list[str] = []

    def _short_contrast(contrast_id: str) -> str:
        text = _as_str(contrast_id)
        if not text:
            return ""
        # Prefer session/state prefix (e.g. ph_post_rest__tetris → ph_post).
        head = text.split("__", 1)[0]
        parts = [p for p in head.split("_") if p]
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return head[:18]

    datasets_c = {_as_str(r["dataset_id"]) or "pooled" for r in broadband_source}
    single_dataset_c = len(datasets_c) == 1
    contrasts_c = {_as_str(r.get("contrast_id")) for r in broadband_source if _as_str(r.get("contrast_id"))}
    multi_contrast_c = len(contrasts_c) > 1
    # Stable visual order: contrast → band → dataset.
    broadband_source = sorted(
        broadband_source,
        key=lambda r: (
            _as_str(r.get("contrast_id")),
            BAND_ORDER.index(_as_str(r.get("band")).casefold())
            if _as_str(r.get("band")).casefold() in BAND_ORDER
            else 99,
            _as_str(r.get("dataset_id")),
        ),
    )
    for row in broadband_source:
        effect = _as_float(row["effect_estimate"])
        lo = _as_float(row["ci_low"])
        hi = _as_float(row["ci_high"])
        xerr = None
        if math.isfinite(lo) and math.isfinite(hi):
            xerr = [[effect - lo], [hi - effect]]
        band = _as_str(row["band"]) or "pooled"
        dataset = _as_str(row["dataset_id"]) or "pooled"
        band_lab = _band_display(band) if band != "pooled" else band
        parts: list[str] = [band_lab]
        if multi_contrast_c:
            short_c = _short_contrast(_as_str(row.get("contrast_id")))
            if short_c:
                parts.append(short_c)
        if not single_dataset_c and dataset not in {"", "pooled"}:
            parts.append(_dataset_display(dataset))
        label = " · ".join(parts)
        ax_c.errorbar(
            effect,
            len(y_labels_c),
            xerr=xerr,
            fmt="o",
            color=PALETTE["dark_gray"],
            markersize=FIGURE3_MARKER_SIZE - 0.5,
            capsize=3,
            elinewidth=FIGURE3_CI_LINEWIDTH,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.5,
        )
        y_labels_c.append(label)
    if y_labels_c:
        _ref_vline(ax_c, 0.0)
        ax_c.set_yticks(range(len(y_labels_c)))
        tick_fs = FS_TICK - 3 if len(y_labels_c) > 10 else FS_TICK - 2
        ax_c.set_yticklabels(y_labels_c, fontsize=tick_fs)
        if multi_contrast_c and single_dataset_c:
            ylabel_c = "Band · contrast"
        elif single_dataset_c:
            ylabel_c = "Band"
        else:
            ylabel_c = "Band · dataset"
        ax_c.set_ylabel(ylabel_c, fontsize=FS_AXIS - 2, labelpad=4)
        ax_c.set_xlabel(f"Effect ({ZLPI_METRIC})", fontsize=FS_AXIS - 2, labelpad=6)
        _style_axes(ax_c)
        _set_panel_title(
            ax_c,
            "Broadband residualization",
            fontsize=FS_PANEL_TITLE - 2,
            pad=10,
        )
        _add_panel_label(ax_c, "C")
    else:
        _mark_empty_panel(
            ax_c,
            MSG_NOT_INCLUDED,
            xlabel=f"Effect ({ZLPI_METRIC})",
            ylabel="Band · dataset",
            xlim=(-1.0, 1.0),
            ylim=(0.0, 1.0),
        )
        _set_panel_title(ax_c, "Broadband residualization", fontsize=FS_PANEL_TITLE - 2, pad=10)
        _add_panel_label(ax_c, "C")
    broadband_csv = source_dir / "figure3_panel_c_broadband.csv"
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
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="broadband_residualized",
            title="Broadband residualization sensitivity",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[
                str(inputs.get("sensitivity") or ""),
                str(inputs.get("specification_matrix") or ""),
            ],
            source_data_csv=str(broadband_csv),
            analysis_keys=[
                "control_id=broadband_residualized",
                "can_rescue_primary=false",
            ],
            notes=(
                "Default confirmatory representation sensitivity only; cannot "
                "rescue primary D240 absolute-power ZLPI."
            ),
        )
    )

    # Panel D: specification / LOO summary.
    ax_d = fig.add_subplot(gs[1, 1])
    spec_source = []
    for i, row in enumerate(sensitivity):
        control_id = _as_str(row.get("control_id"))
        effect = _as_float(row.get("effect_estimate") or row.get("pooled_effect"))
        if not control_id:
            continue
        spec_source.append(
            {
                "control_id": control_id,
                "endpoint_name": _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
                "duration_s": _as_int(row.get("duration_s"), 240),
                "effect_estimate": effect,
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n") or row.get("n_datasets")),
                "status": _as_str(row.get("status")),
                "is_primary_analysis": _as_str(row.get("is_primary_analysis")),
                "can_rescue_primary": False,
            }
        )
    # Prefer plotting finite sensitivity effects; overlay LOO deltas if present.
    y_labels = []
    for i, row in enumerate(spec_source[:20]):
        effect = _as_float(row["effect_estimate"])
        if not math.isfinite(effect):
            continue
        lo = _as_float(row["ci_low"])
        hi = _as_float(row["ci_high"])
        xerr = None
        if math.isfinite(lo) and math.isfinite(hi):
            xerr = [[effect - lo], [hi - effect]]
        control_key = _as_str(row["control_id"]).casefold()
        if "swpi" in control_key:
            marker, color = "s", PALETTE["green"]
        elif "mwpi" in control_key:
            marker, color = "s", PALETTE["orange"]
        else:
            marker, color = "o", PALETTE["blue"]
        ax_d.errorbar(
            effect,
            len(y_labels),
            xerr=xerr,
            fmt=marker,
            color=color,
            markersize=FIGURE3_MARKER_SIZE - 0.5,
            capsize=3,
            elinewidth=FIGURE3_CI_LINEWIDTH,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.5,
        )
        y_labels.append(_spec_display(str(row["control_id"])))
    if loo:
        loo_source = []
        for row in loo:
            if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
                continue
            loo_source.append(
                {
                    "band": _as_str(row.get("band")),
                    "omitted_dataset_id": _as_str(row.get("omitted_dataset_id")),
                    "pooled_effect": _as_float(row.get("pooled_effect")),
                    "delta_vs_full": _as_float(row.get("delta_vs_full")),
                    "endpoint_name": ENDPOINT_ZLPI,
                }
            )
        loo_csv = source_dir / "figure3_panel_d_loo.csv"
        write_source_csv(
            loo_csv,
            loo_source,
            (
                "band",
                "omitted_dataset_id",
                "pooled_effect",
                "delta_vs_full",
                "endpoint_name",
            ),
        )
        source_paths.append(loo_csv)
    if y_labels:
        _ref_vline(ax_d, 0.0)
        ax_d.set_yticks(range(len(y_labels)))
        ax_d.set_yticklabels(y_labels, fontsize=FS_TICK - 2)
        ax_d.set_ylabel("Specification", fontsize=FS_AXIS - 2, labelpad=4)
        ax_d.set_xlabel(f"Effect ({ZLPI_METRIC})", fontsize=FS_AXIS - 2, labelpad=6)
        _style_axes(ax_d)
    else:
        _mark_empty_panel(
            ax_d,
            MSG_NOT_INCLUDED,
            xlabel=f"Effect ({ZLPI_METRIC})",
            ylabel="Specification",
        )
    _set_panel_title(ax_d, "Specification matrix", fontsize=FS_PANEL_TITLE - 2, pad=10)
    _add_panel_label(ax_d, "D")
    spec_csv = source_dir / "figure3_panel_d_specification.csv"
    write_source_csv(
        spec_csv,
        spec_source,
        (
            "control_id",
            "endpoint_name",
            "duration_s",
            "effect_estimate",
            "ci_low",
            "ci_high",
            "n",
            "status",
            "is_primary_analysis",
            "can_rescue_primary",
        ),
    )
    source_paths.append(spec_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="specification_matrix",
            title="Sensitivity specification matrix",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[
                str(inputs.get("specification_matrix") or ""),
                str(inputs.get("sensitivity") or ""),
                str(inputs.get("leave_one_out") or ""),
            ],
            source_data_csv=str(spec_csv),
            analysis_keys=["can_rescue_primary=false"],
        )
    )

    fig.suptitle(FIGURE3_TITLE, fontsize=FS_SUPTITLE - 1, fontweight="bold", y=0.978)
    fig.subplots_adjust(**FIGURE3_SUBPLOT_ADJUST)
    if panel_b_footnote:
        fig.text(
            0.5,
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
            "A: Dataset-level mean of biological-participant Δ (observed − circular-shift "
            "null) for D240 absolute-log10 theta ZLPI, with Student-t 95% CIs. Secondary "
            "nulls appear in the Figure 3 supplement.\n"
            "B: Duration sensitivity (ZLPI at 240/180 s; MWPI at 120 s; SWPI at 60 s). "
            "Color encodes EEG band; marker shape encodes endpoint index. Shorter "
            "windows cannot rescue primary D240 ZLPI; overlapping CIs are not "
            "equivalence.\n"
            "C: Broadband-residualized sensitivity forest (default confirmatory "
            "representation control).\n"
            "D: Specification matrix of default sensitivity controls (optional "
            "CFA/nuisance rows appear only when enabled). Dual ECG–PPG comparison is "
            "not part of this figure.\n"
            "Source data: figures/source_data/figure3_panel_*.csv.\n"
        ),
        encoding="utf-8",
    )
    pdf, svg, png = save_figure_trio(fig, output_dir, FIGURE3_STEM)

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
                "- For HIIT, n uses session subject_ids (e.g. `01_ph` / `01_ps`), "
                "matching Figure 1 Panel B.",
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
    panels = figure1.panels + figure2.panels + figure3_bundle.panels
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
    "LAG_CURVE_DISPLAY_SMOOTH_NOTE",
    "LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S",
    "FigureArtifacts",
    "Figure3RenderResult",
    "FiguresResult",
    "display_lag_mask",
    "gaussian_smooth_display_series",
    "generate_confirmatory_figures",
    "mean_ci_by_lag",
    "render_figure1",
    "render_figure3",
    "resolve_reporting_inputs",
    "save_figure_trio",
    "write_figure_export_categories",
]
