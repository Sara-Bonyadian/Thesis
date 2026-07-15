"""Publication figures for confirmatory zero-lag analyses (M12).

Reads only frozen confirmatory tables. No hard-coded scientific results.
Figures 1–3 are written as PDF, SVG, and 300-dpi PNG with companion
source-data CSVs and a figure-source manifest.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
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
CI_95_METHOD_NOTE = (
    "Pointwise 95% CI = mean ± 1.96×SE in Fisher-z space (not bootstrap)"
)
ZLPI_METRIC = "Fisher z"
EEG_BAND_YLABEL = "EEG frequency band"
EQUIVALENCE_REGION_LABEL = f"±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S} s"
MEDIAN_PEAK_CENTER_LABEL = "Median fitted peak μ (≠ group-mean max)"
MU_CLARIFICATION_NOTE = (
    "Orange dotted line: median of participant-specific Gaussian-fitted peak centers (μ); "
    "this is not necessarily the location of the maximum of the group-average Fisher-z curve"
)
GROUP_MEAN_LABEL_TEMPLATE = "Group mean Fisher z (n = {n})"
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

FIGURE1_TITLE = "Lag-resolved EEG–cardiac coupling"
FIGURE2_TITLE = "State-dependent attenuation of coupling"
FIGURE3_TITLE = "Temporal specificity and artifact controls"

FIGURE1_STEM = "figure1_lag_resolved_zero_lag"
FIGURE2_STEM = "figure2_state_attenuation_replication"
FIGURE3_STEM = "figure3_temporal_artifact_specificity"
FIGURE3_SUPPLEMENT_STEM = "figure3_supplement_null_diagnostics"
FIGURE3_SUPPLEMENT_LABEL = "Descriptive nested-observation diagnostic"
FIGURE3_PARTICIPANT_FOREST_STEM = "figure3_supplement_participant_null_forests"
FIGURE3_PARTICIPANT_FOREST_MAX_ROWS = 24
FIGURE3_PANEL_B_ENCODING_NOTE = (
    "Color = EEG frequency band · Marker shape = endpoint index "
    "(ZLPI at 240/180 s; MWPI at 120 s; SWPI at 60 s)"
)
FIGURE3_PANEL_B_FOOTNOTE_SHORT = (
    "Panel B encoding: color = band · shape = index "
    "(ZLPI 240/180; MWPI 120; SWPI 60); Student-t 95% CIs; "
    "absolute estimates (not equivalence)"
)
FIGURE3_FIGSIZE = (15.2, 12.6)
FIGURE3_SUBPLOT_ADJUST = {
    "left": 0.08,
    "right": 0.86,
    "top": 0.91,
    "bottom": 0.09,
    "wspace": 0.45,
    "hspace": 0.50,
}
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
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-32, 10),
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
    ax.axhline(y, color=REF_LINE_COLOR, lw=1.0, ls="--", zorder=1)


def _ref_vline(ax: plt.Axes, x: float = 0.0) -> None:
    ax.axvline(x, color=REF_LINE_COLOR, lw=1.0, ls="--", zorder=1)


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
        "subject_level": "subject_level_metrics.csv",
        "paired_contrasts": "paired_contrasts.csv",
        "dataset_effects": "dataset_effects.csv",
        "meta_analysis": "meta_analysis_results.csv",
        "leave_one_out": "leave_one_dataset_out.csv",
        "peak_params": "peak_fit_params.csv",
        "peak_equivalence": "peak_center_equivalence.csv",
        "null_subject": "null_subject_results.csv",
        "null_summary": "null_summary.csv",
        "sensitivity": "sensitivity_results.csv",
        "specification_matrix": "specification_matrix.csv",
        "modality": "modality_comparison.csv",
        "duration_sensitivity": "duration_sensitivity.csv",
        "eligibility": "eligibility_by_duration.csv",
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


def render_figure1(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 1: lag-resolved zero-lag structure (D240 ZLPI Fisher-z).

    Aggregation uses the full 1-s lag grid. Plotting may thin to every
    ``FIGURE1_DISPLAY_LAG_STEP_S`` lag for readability (no interpolation/smoothing).
    """
    _configure_publication_style()
    curves = read_csv_rows(inputs.get("curves_d240"))
    peak_rows = read_csv_rows(inputs.get("peak_params"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 12.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes_flat = list(axes.ravel())
    panel_letters = ("A", "B", "C", "D")
    shared_handles: list[object] = []
    shared_labels: list[str] = []
    primary_contract = contract_for_duration(EXPECTED_PRIMARY_DURATION_S)
    for ax, band, letter in zip(axes_flat, BAND_ORDER, panel_letters, strict=True):
        series = mean_ci_by_lag(curves, band=band, condition_role="low_demand")
        fields = (
            "lag_s",
            "band",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "n",
            "mean_z",
            "ci_low",
            "ci_high",
        )
        csv_path = source_dir / f"figure1_panel_{band}_mean_ci.csv"
        # Source CSV retains the full 1-s analysis grid.
        write_source_csv(csv_path, series, fields)
        source_paths.append(csv_path)
        color = _band_color(band)
        if series:
            _shade_flanks(ax, EXPECTED_PRIMARY_DURATION_S)
            lags = np.asarray([r["lag_s"] for r in series], dtype=float)
            mean = np.asarray([r["mean_z"] for r in series], dtype=float)
            lo = np.asarray([r["ci_low"] for r in series], dtype=float)
            hi = np.asarray([r["ci_high"] for r in series], dtype=float)
            mask = display_lag_mask(lags, FIGURE1_DISPLAY_LAG_STEP_S)
            lags_d, mean_d, lo_d, hi_d = lags[mask], mean[mask], lo[mask], hi[mask]
            n = int(series[0]["n"]) if series else 0
            ax.fill_between(
                lags_d,
                lo_d,
                hi_d,
                color=color,
                alpha=CI_ALPHA,
                linewidth=0,
                label=CI_95_LABEL,
                zorder=2,
            )
            ax.plot(
                lags_d,
                mean_d,
                color=color,
                lw=LINE_WIDTH,
                ls=_band_linestyle(band),
                label=GROUP_MEAN_LABEL_TEMPLATE.format(n=n),
                zorder=3,
            )
            # Peak μ overlay: low-demand identifiable peaks only (match curve state).
            mus = [
                _as_float(r.get("peak_center_mu_s"))
                for r in peak_rows
                if _as_str(r.get("band")).casefold() == band
                and _as_int(r.get("duration_s"), 240) == 240
                and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
                and _as_str(r.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
                == PRIMARY_REPRESENTATION
                and str(r.get("has_identifiable_peak", "")).lower() in {"true", "1", "yes"}
                and _is_low_demand_condition(r)
            ]
            mus = [m for m in mus if math.isfinite(m)]
            if mus:
                ax.axvspan(
                    -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                    facecolor=PALETTE["orange"],
                    alpha=0.14,
                    zorder=1,
                    linewidth=0,
                    label=MU_EQUIVALENCE_LABEL,
                )
                ax.axvline(
                    float(np.median(mus)),
                    color=PALETTE["orange"],
                    ls=":",
                    lw=LINE_WIDTH,
                    label=MEDIAN_PEAK_CENTER_LABEL,
                    zorder=4,
                )
            _set_lag_axes(ax, EXPECTED_PRIMARY_DURATION_S)
            _set_panel_title(ax, f"{_band_display(band)} · low-demand (n = {n})")
            _add_panel_label(ax, letter)
            if letter == "A":
                shared_handles, shared_labels = ax.get_legend_handles_labels()
                # Keep legend order: mean, CI, median μ, equivalence (if present).
                preferred = [
                    GROUP_MEAN_LABEL_TEMPLATE.format(n=n),
                    CI_95_LABEL,
                    MEDIAN_PEAK_CENTER_LABEL,
                    MU_EQUIVALENCE_LABEL,
                ]
                ordered = []
                for lab in preferred:
                    for h, L in zip(shared_handles, shared_labels, strict=False):
                        if L == lab:
                            ordered.append((h, L))
                            break
                if ordered:
                    shared_handles, shared_labels = map(list, zip(*ordered, strict=False))
        else:
            _set_panel_title(ax, _band_display(band))
            _add_panel_label(ax, letter)
            _mark_empty_panel(
                ax,
                _band_not_analyzed_message(band),
                xlabel=LAG_XLABEL,
                ylabel=Z_YLABEL,
                xlim=(
                    float(contract_for_duration(EXPECTED_PRIMARY_DURATION_S).lag_min_s),
                    float(contract_for_duration(EXPECTED_PRIMARY_DURATION_S).lag_max_s),
                ),
                ylim=(-0.2, 0.2),
            )
        panel_sources.append(
            FigurePanelSource(
                figure_id="figure1",
                panel_id=f"band_{band}",
                title=f"Low-demand Fisher-z lag curve ({band})",
                endpoint_name=ENDPOINT_ZLPI,
                duration_s=EXPECTED_PRIMARY_DURATION_S,
                input_tables=[
                    str(inputs.get("curves_d240") or ""),
                    str(inputs.get("peak_params") or ""),
                ],
                source_data_csv=str(csv_path),
                analysis_keys=[
                    f"duration={EXPECTED_PRIMARY_DURATION_S}",
                    f"endpoint={ENDPOINT_ZLPI}",
                    f"band={band}",
                    f"representation={PRIMARY_REPRESENTATION}",
                ],
                notes=(
                    f"Low-demand D{EXPECTED_PRIMARY_DURATION_S} absolute_log10 curves; "
                    f"distant flanks |τ|∈[{primary_contract.flank_inner_s},"
                    f"{primary_contract.flank_outer_s}] s; "
                    f"local shoulders |τ|∈[{primary_contract.shoulders_inner_s},"
                    f"{primary_contract.shoulders_outer_s}] s; "
                    f"orange band = {EQUIVALENCE_REGION_LABEL} μ equivalence when "
                    f"low-demand peaks exist. Ribbon is {CI_95_METHOD_NOTE}. "
                    f"Analysis lag step=1 s; display lag step={FIGURE1_DISPLAY_LAG_STEP_S} s "
                    "(plot thinning only)."
                ),
            )
        )

    # Annotate lag regions after shared y-limits are established.
    for ax in axes_flat:
        _annotate_lag_regions(ax, EXPECTED_PRIMARY_DURATION_S, enabled=True)
        # sharex=True hides tick numbers on the top row; show them on A–D.
        ax.tick_params(axis="x", labelbottom=True)
    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="lower center",
            ncol=min(4, len(shared_labels)),
            fontsize=FS_LEGEND - 2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.025),
        )
    fig.suptitle(FIGURE1_TITLE, fontsize=FS_SUPTITLE, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        f"{LAG_CONVENTION_NOTE}.  {CI_95_METHOD_NOTE}.  {FIGURE1_DISPLAY_GRID_DISCLOSURE}",
        ha="center",
        va="bottom",
        fontsize=FS_TICK - 2,
        color=PALETTE["dark_gray"],
    )
    # Bottom room for C/D "Lag τ (s)" above legend; hspace keeps A/B labels off C/D titles.
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.20, wspace=0.18, hspace=0.44)
    pdf, svg, png = save_figure_trio(fig, output_dir, FIGURE1_STEM)
    return FigureArtifacts(
        figure_id="figure1",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(source_paths),
        panels=tuple(panel_sources),
    )


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
    """Figure 2: state attenuation, meta forest, D180 sensitivity, μ equivalence."""
    subjects = read_csv_rows(inputs.get("subject_level"))
    paired = read_csv_rows(inputs.get("paired_contrasts"))
    effects = read_csv_rows(inputs.get("dataset_effects"))
    meta = read_csv_rows(inputs.get("meta_analysis"))
    equivalence = read_csv_rows(inputs.get("peak_equivalence"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    _configure_publication_style()
    fig = plt.figure(figsize=(13.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.38)

    # Panel A: paired deltas by dataset (theta ZLPI D240).
    # Points = observation-level participant×contrast; mean/CI from independent units.
    ax_a = fig.add_subplot(gs[0, 0])
    datasets = sorted(
        {
            _as_str(r.get("dataset_id")).casefold()
            for r in paired
            if _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
            and _as_int(r.get("duration_s"), 240) == 240
            and _as_str(r.get("band")).casefold() == "theta"
        }
    )
    paired_source: list[dict[str, object]] = []
    unit_source: list[dict[str, object]] = []
    inference_source: list[dict[str, object]] = []
    plotted_any = False
    dataset_tick_labels: list[str] = []
    for idx, dataset_id in enumerate(datasets):
        points = _paired_points_for_dataset(
            subjects, paired, dataset_id=dataset_id, band="theta"
        )
        paired_source.extend(points)
        ys = np.asarray(
            [_as_float(p["delta_endpoint_index"]) for p in points], dtype=float
        )
        finite_mask = np.isfinite(ys)
        ys_plot = ys[finite_mask]
        base = _dataset_display(dataset_id)
        seed = int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16)
        inference, unit_summaries = infer_paired_deltas_cluster_aware(
            points,
            bootstrap_seed=seed,
        )
        for summary in unit_summaries:
            unit_source.append(
                {
                    "dataset_id": dataset_id,
                    "unit_id": summary.unit_id,
                    "unit_field": inference.unit_field,
                    "n_observations": summary.n_observations,
                    "mean_delta": summary.mean_delta,
                    "band": "theta",
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                }
            )
        inference_row = {
            "dataset_id": dataset_id,
            "band": "theta",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            **inference.as_dict(),
        }
        inference_source.append(inference_row)
        dataset_tick_labels.append(
            f"{base}\n"
            f"({inference.n_observations} obs, "
            f"{inference.n_units} {inference.unit_label}s)"
        )
        if ys_plot.size == 0:
            continue
        plotted_any = True
        x = np.full(ys_plot.shape, idx, dtype=float) + 0.05 * np.random.default_rng(
            seed
        ).normal(size=ys_plot.size)
        ax_a.scatter(
            x,
            ys_plot,
            s=SCATTER_SIZE,
            alpha=0.75,
            color=_band_color("theta"),
            marker=_band_marker("theta"),
            edgecolors=PALETTE["dark_gray"],
            linewidths=0.6,
            label=None,
            zorder=2,
        )
        # Inferential marker: mean of independent-unit means ± Student-t CI.
        mean_u = float(inference.mean_delta)
        if (
            math.isfinite(mean_u)
            and math.isfinite(float(inference.ci_low))
            and math.isfinite(float(inference.ci_high))
        ):
            yerr = [
                [mean_u - float(inference.ci_low)],
                [float(inference.ci_high) - mean_u],
            ]
        else:
            yerr = 0.0
        ax_a.errorbar(
            idx,
            mean_u if math.isfinite(mean_u) else 0.0,
            yerr=yerr,
            fmt="s",
            color=PALETTE["vermillion"],
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
            zorder=3,
        )
    if plotted_any:
        _ref_hline(ax_a, 0.0)
        ax_a.set_xticks(range(len(datasets)))
        ax_a.set_xticklabels(dataset_tick_labels, rotation=0, ha="center")
        ax_a.set_xlabel("Dataset", fontsize=FS_AXIS, labelpad=8)
        ax_a.set_ylabel(
            f"Δ {_endpoint_display(ENDPOINT_ZLPI)} (Fisher z)\n"
            f"task − low-demand",
            fontsize=FS_AXIS,
            labelpad=10,
        )
        _style_axes(ax_a)
    else:
        _mark_empty_panel(
            ax_a,
            MSG_NOT_APPLICABLE,
            xlabel="Dataset",
            ylabel=(
                f"Δ {_endpoint_display(ENDPOINT_ZLPI)} "
                f"(task − low-demand; {ZLPI_METRIC})"
            ),
        )
    _set_panel_title(ax_a, f"Paired 240 s {_endpoint_display(ENDPOINT_ZLPI)} (theta)")
    _add_panel_label(ax_a, "A")
    paired_csv = source_dir / "figure2_panel_a_paired_deltas.csv"
    write_source_csv(
        paired_csv,
        paired_source,
        (
            "dataset_id",
            "participant_id",
            "session_id",
            "contrast_id",
            "band",
            "duration_s",
            "endpoint_name",
            "power_representation",
            "sampling_unit",
            "low_endpoint_index",
            "effort_endpoint_index",
            "delta_endpoint_index",
        ),
    )
    unit_csv = source_dir / "figure2_panel_a_unit_summaries.csv"
    write_source_csv(
        unit_csv,
        unit_source,
        (
            "dataset_id",
            "unit_id",
            "unit_field",
            "n_observations",
            "mean_delta",
            "band",
            "duration_s",
            "endpoint_name",
        ),
    )
    inference_csv = source_dir / "figure2_panel_a_inference.csv"
    inference_fields = (
        "dataset_id",
        "band",
        "duration_s",
        "endpoint_name",
        "estimand",
        "unit_field",
        "unit_label",
        "n_observations",
        "n_units",
        "nested_repeated_measures",
        "unit_detection_reason",
        "mean_delta",
        "se_delta",
        "ci_low",
        "ci_high",
        "t_stat",
        "p_value",
        "df",
        "ci_method",
        "cluster_bootstrap_mean",
        "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high",
        "cluster_bootstrap_n_draws",
        "bootstrap_agrees_with_unit_ci",
        "mixed_model_appropriate",
        "mixed_model_recommendation",
        "mixed_model_status",
        "mixed_model_mean",
        "mixed_model_ci_low",
        "mixed_model_ci_high",
        "mixed_model_p_value",
        "notes",
    )
    write_source_csv(inference_csv, inference_source, inference_fields)
    source_paths.extend([paired_csv, unit_csv, inference_csv])
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="paired_deltas",
            title="Paired task-minus-low-demand ZLPI",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[
                str(inputs.get("paired_contrasts") or ""),
                str(inputs.get("subject_level") or ""),
            ],
            source_data_csv=str(paired_csv),
            analysis_keys=[
                "endpoint=zlpi",
                "duration=240",
                "band=theta",
                "inference=independent_unit_means",
            ],
            notes=(
                f"n_datasets={len(datasets)}; each plotted point is one "
                "participant×contrast ΔZLPI (task − low-demand). Orange marker = "
                "mean of participant-level mean Δ with Student-t 95% CI; SE uses "
                "n_participants, not n_contrast rows. Companion CSVs: unit "
                "summaries + cluster-bootstrap / mixed-model diagnostics."
            ),
        )
    )

    # Panel B: forest / meta (primary ZLPI by band).
    ax_b = fig.add_subplot(gs[0, 1])
    meta_rows = [
        r
        for r in meta
        if _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
        and _as_int(r.get("duration_s"), 240) == 240
        and str(r.get("is_primary_analysis", "true")).lower() in {"true", "1", "yes", ""}
    ]
    forest_source = []
    for band in BAND_ORDER:
        row = next((r for r in meta_rows if _as_str(r.get("band")).casefold() == band), None)
        if row is None:
            continue
        effect = _as_float(row.get("pooled_effect"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        if not math.isfinite(effect):
            continue
        forest_source.append(
            {
                "band": band,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": 240,
                "pooled_effect": effect,
                "ci_low": lo,
                "ci_high": hi,
                "n_datasets": _as_int(row.get("n_datasets")),
                "i2": _as_float(row.get("i2")),
                "tau2": _as_float(row.get("tau2")),
            }
        )
    n_meta_datasets = _n_datasets_from_rows(forest_source) or _n_datasets_from_rows(meta_rows)
    for i, row in enumerate(forest_source):
        effect = float(row["pooled_effect"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        band = str(row["band"])
        xerr = None
        if math.isfinite(lo) and math.isfinite(hi):
            xerr = [[effect - lo], [hi - effect]]
        ax_b.errorbar(
            effect,
            i,
            xerr=xerr,
            fmt=_band_marker(band),
            color=_band_color(band),
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            linestyle=_band_linestyle(band),
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
        )
    if forest_source:
        _ref_vline(ax_b, 0.0)
        ax_b.set_yticks(range(len(forest_source)))
        ax_b.set_yticklabels([_band_display(str(r["band"])) for r in forest_source])
        ax_b.set_ylabel(EEG_BAND_YLABEL, fontsize=FS_AXIS, labelpad=10)
        ax_b.set_xlabel(
            f"Pooled Δ {_endpoint_display(ENDPOINT_ZLPI)} "
            f"({ZLPI_METRIC}; {CI_95_LABEL})",
            fontsize=FS_AXIS,
            labelpad=8,
        )
        _style_axes(ax_b)
    else:
        _mark_empty_panel(
            ax_b,
            MSG_NOT_INCLUDED,
            xlabel=(
                f"Pooled Δ {_endpoint_display(ENDPOINT_ZLPI)} "
                f"({ZLPI_METRIC}; {CI_95_LABEL})"
            ),
            ylabel=EEG_BAND_YLABEL,
        )
    _set_panel_title(
        ax_b,
        _meta_or_single_title(
            n_datasets=n_meta_datasets,
            endpoint_label=_endpoint_display(ENDPOINT_ZLPI),
        ),
    )
    _add_panel_label(ax_b, "B")
    forest_csv = source_dir / "figure2_panel_b_meta_forest.csv"
    write_source_csv(
        forest_csv,
        forest_source,
        (
            "band",
            "endpoint_name",
            "duration_s",
            "pooled_effect",
            "ci_low",
            "ci_high",
            "n_datasets",
            "i2",
            "tau2",
        ),
    )
    source_paths.append(forest_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="meta_forest",
            title=(
                "Random-effects meta-analysis"
                if n_meta_datasets >= 2
                else "Single-dataset effect estimate"
            ),
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("meta_analysis") or "")],
            source_data_csv=str(forest_csv),
            analysis_keys=["endpoint=zlpi", "duration=240", "is_primary_analysis=true"],
            notes=f"n_datasets={n_meta_datasets}; error bars are {CI_95_LABEL}.",
        )
    )

    # Panel C: absolute D180 ZLPI state-contrast effects (sensitivity only).
    # This is NOT the D180−D240 difference. Restrict to absolute_log10 so the
    # band means are not contaminated by relative / broadband-residualized rows.
    ax_c = fig.add_subplot(gs[1, 0])
    d180 = [
        r
        for r in effects
        if _as_int(r.get("duration_s"), 0) == 180
        and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
        and _as_str(r.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
        == PRIMARY_REPRESENTATION
    ]
    if not d180:
        d180 = [
            r
            for r in paired
            if _as_int(r.get("duration_s"), 0) == 180
            and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
            and _as_str(r.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
            == PRIMARY_REPRESENTATION
        ]
    d180_source = []
    by_band: dict[str, list[float]] = {}
    for row in d180:
        band = _as_str(row.get("band")).casefold() or "theta"
        value = _as_float(row.get("effect_mean", row.get("delta_endpoint_index")))
        if math.isfinite(value):
            by_band.setdefault(band, []).append(value)
            d180_source.append(
                {
                    "band": band,
                    "dataset_id": _as_str(row.get("dataset_id")),
                    "contrast_id": _as_str(row.get("contrast_id")),
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": 180,
                    "power_representation": PRIMARY_REPRESENTATION,
                    "effect": value,
                    "is_primary_analysis": False,
                    "can_rescue_primary": False,
                }
            )
    plotted_bands = [band for band in BAND_ORDER if by_band.get(band)]
    for i, band in enumerate(plotted_bands):
        vals = np.asarray(by_band.get(band, []), dtype=float)
        if vals.size == 0:
            continue
        ax_c.errorbar(
            float(np.mean(vals)),
            i,
            xerr=1.959963984540054 * float(np.std(vals, ddof=1) / math.sqrt(vals.size))
            if vals.size >= 2
            else 0.0,
            fmt=_band_marker(band),
            color=_band_color(band),
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            linestyle=_band_linestyle(band),
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
        )
    if plotted_bands:
        _ref_vline(ax_c, 0.0)
        ax_c.set_yticks(range(len(plotted_bands)))
        ax_c.set_yticklabels([_band_display(b) for b in plotted_bands])
        ax_c.set_ylabel(EEG_BAND_YLABEL, fontsize=FS_AXIS, labelpad=10)
        ax_c.set_xlabel(
            f"180 s Δ {_endpoint_display(ENDPOINT_ZLPI)} "
            f"(task − low; {ZLPI_METRIC}; {CI_95_LABEL}; not D180−D240)",
            fontsize=FS_AXIS,
            labelpad=8,
        )
        _style_axes(ax_c)
    else:
        _mark_empty_panel(
            ax_c,
            MSG_NOT_INCLUDED,
            xlabel=(
                f"180 s Δ {_endpoint_display(ENDPOINT_ZLPI)} "
                f"(task − low; {ZLPI_METRIC}; {CI_95_LABEL}; not D180−D240)"
            ),
            ylabel=EEG_BAND_YLABEL,
        )
    _set_panel_title(
        ax_c,
        f"180 s {_endpoint_display(ENDPOINT_ZLPI)} sensitivity (absolute effects)",
    )
    _add_panel_label(ax_c, "C")
    d180_csv = source_dir / "figure2_panel_c_d180_sensitivity.csv"
    write_source_csv(
        d180_csv,
        d180_source,
        (
            "band",
            "dataset_id",
            "contrast_id",
            "endpoint_name",
            "duration_s",
            "power_representation",
            "effect",
            "is_primary_analysis",
            "can_rescue_primary",
        ),
    )
    source_paths.append(d180_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="d180_sensitivity",
            title="D180 ZLPI sensitivity (absolute_log10 state-contrast effects)",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=180,
            input_tables=[str(inputs.get("dataset_effects") or ""), str(inputs.get("paired_contrasts") or "")],
            source_data_csv=str(d180_csv),
            analysis_keys=[
                "endpoint=zlpi",
                "duration=180",
                f"representation={PRIMARY_REPRESENTATION}",
                "can_rescue_primary=false",
            ],
            notes=(
                "Absolute D180 state-contrast ΔZLPI (not D180−D240). "
                "Displayed separately from primary D240; not a rescue pathway."
            ),
        )
    )

    # Panel D: peak-center TOST / ±2 s equivalence (primary D240 ZLPI slice).
    ax_d = fig.add_subplot(gs[1, 1])
    eq_candidates = []
    for row in equivalence:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), 240) != 240:
            continue
        if (
            _as_str(row.get("power_representation"), PRIMARY_REPRESENTATION).casefold()
            != PRIMARY_REPRESENTATION
        ):
            continue
        mean_mu = _as_float(row.get("mean_mu"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        if not math.isfinite(mean_mu):
            continue
        eq_candidates.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")),
                "band": _as_str(row.get("band")).casefold(),
                "endpoint_name": ENDPOINT_ZLPI,
                "mean_mu": mean_mu,
                "ci_low": lo,
                "ci_high": hi,
                "bound_low": -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                "bound_high": EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                "equivalent": _as_str(row.get("equivalent")),
            }
        )
    eq_source = sorted(
        eq_candidates,
        key=lambda r: (_as_str(r["dataset_id"]).casefold(), _band_sort_key(str(r["band"]))),
    )
    n_eq_datasets = len({_as_str(r["dataset_id"]).casefold() for r in eq_source})
    # For a single-dataset figure, reserve a row for every confirmatory band.
    if n_eq_datasets <= 1 and eq_source:
        by_band = {str(r["band"]).casefold(): r for r in eq_source}
        dataset_id = _as_str(eq_source[0]["dataset_id"])
        expanded = []
        for band in BAND_ORDER:
            if band in by_band:
                expanded.append(by_band[band])
            else:
                expanded.append(
                    {
                        "dataset_id": dataset_id,
                        "band": band,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "mean_mu": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "bound_low": -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                        "bound_high": EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                        "equivalent": "",
                    }
                )
        eq_source = expanded
    for i, row in enumerate(eq_source):
        mean_mu = float(row["mean_mu"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        band = str(row["band"])
        if not math.isfinite(mean_mu):
            ax_d.text(
                0.02,
                i,
                "Not included",
                va="center",
                ha="left",
                fontsize=FS_TICK - 1,
                color=PALETTE["dark_gray"],
                transform=ax_d.get_yaxis_transform(),
            )
            continue
        xerr = None
        if math.isfinite(lo) and math.isfinite(hi):
            xerr = [[mean_mu - lo], [hi - mean_mu]]
        ax_d.errorbar(
            mean_mu,
            i,
            xerr=xerr,
            fmt=_band_marker(band),
            color=_band_color(band),
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            linestyle=_band_linestyle(band),
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
        )
    if eq_source:
        ax_d.axvspan(
            -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
            facecolor=PALETTE["orange"],
            alpha=0.35,
            hatch="////",
            edgecolor=PALETTE["vermillion"],
            linewidth=0.8,
            label=EQUIVALENCE_REGION_LABEL,
            zorder=0,
        )
        ax_d.axvline(-EXPECTED_PEAK_CENTER_EQUIVALENCE_S, color=PALETTE["vermillion"], lw=1.2, ls=":", zorder=1)
        ax_d.axvline(EXPECTED_PEAK_CENTER_EQUIVALENCE_S, color=PALETTE["vermillion"], lw=1.2, ls=":", zorder=1)
        _ref_vline(ax_d, 0.0)
        ax_d.set_yticks(range(len(eq_source)))
        if n_eq_datasets <= 1:
            y_labels = [_band_display(str(r["band"])) for r in eq_source]
        else:
            y_labels = [
                f"{r['dataset_id']} · {_band_display(str(r['band']))}" for r in eq_source
            ]
        ax_d.set_yticklabels(y_labels, fontsize=FS_TICK)
        ax_d.set_ylabel(
            EEG_BAND_YLABEL if n_eq_datasets <= 1 else "Dataset · EEG frequency band",
            fontsize=FS_AXIS,
            labelpad=10,
        )
        ax_d.set_xlabel(f"Peak center μ (s; {CI_95_LABEL})", fontsize=FS_AXIS, labelpad=8)
        _style_axes(ax_d)
        handles, labels = ax_d.get_legend_handles_labels()
        eq_items = [
            (h, lab)
            for h, lab in zip(handles, labels, strict=False)
            if lab == EQUIVALENCE_REGION_LABEL
        ]
        if eq_items:
            _legend_inside(ax_d, [eq_items[0][0]], [eq_items[0][1]], loc="upper right")
    else:
        _mark_empty_panel(
            ax_d,
            MSG_NOT_INCLUDED,
            xlabel=f"Peak center μ (s; {CI_95_LABEL})",
            ylabel=EEG_BAND_YLABEL,
        )
    # Title: ±2 s is a prespecified TOST reference region; do not imply TOST passed.
    _set_panel_title(
        ax_d,
        f"Peak-center μ vs ±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S} s reference",
    )
    _add_panel_label(ax_d, "D")
    eq_csv = source_dir / "figure2_panel_d_mu_equivalence.csv"
    write_source_csv(
        eq_csv,
        eq_source,
        (
            "dataset_id",
            "band",
            "endpoint_name",
            "mean_mu",
            "ci_low",
            "ci_high",
            "bound_low",
            "bound_high",
            "equivalent",
        ),
    )
    source_paths.append(eq_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="mu_equivalence",
            title=(
                f"Peak-center μ estimates with ±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S} s "
                "TOST reference region"
            ),
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("peak_equivalence") or "")],
            source_data_csv=str(eq_csv),
            analysis_keys=[
                "endpoint=zlpi",
                f"tost_bounds=±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S}s",
                "duration=240",
                f"representation={PRIMARY_REPRESENTATION}",
            ],
            notes=(
                "Primary D240 absolute_log10 ZLPI, low-demand identifiable peaks only. "
                f"Square = mean μ; bars = Student-t {CI_95_LABEL}. "
                "Orange band is the TOST reference region, not a claim that TOST passed. "
                f"equivalent column is TOST at α={0.05}."
            ),
        )
    )

    fig.suptitle(FIGURE2_TITLE, fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.10, wspace=0.45, hspace=0.42)
    pdf, svg, png = save_figure_trio(fig, output_dir, FIGURE2_STEM)
    return FigureArtifacts(
        figure_id="figure2",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(source_paths),
        panels=tuple(panel_sources),
    )


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
        _set_panel_title(ax, title)
        return

    y_pos = np.arange(len(rows), dtype=float)
    labels: list[str] = []
    for idx, row in enumerate(rows):
        dataset = _as_str(getattr(row, "dataset_id", ""))
        n_part = int(getattr(row, "n_participants", 0))
        labels.append(f"{dataset}  (n={n_part})")
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
                markersize=MARKER_SIZE + 1.0,
                capsize=5,
                elinewidth=LINE_WIDTH,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.7,
                zorder=4,
            )
        elif math.isfinite(mean):
            ax.scatter(
                [mean],
                [y],
                s=SCATTER_SIZE + 16,
                color=PALETTE["orange"],
                marker="D",
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.7,
                zorder=4,
            )
    _ref_vline(ax, 0.0)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=FS_TICK - 1)
    ax.set_xlabel(
        f"Participant mean Δ (observed − null; {ZLPI_METRIC})",
        fontsize=FS_AXIS - 1,
        labelpad=8,
    )
    ax.set_ylabel("Dataset", fontsize=FS_AXIS - 1, labelpad=6)
    run_classes = {_as_str(getattr(r, "run_class", "")) for r in rows}
    interps = {_as_str(getattr(r, "interpretation", "")) for r in rows}
    run_class = next(iter(run_classes)) if len(run_classes) == 1 else "mixed"
    badge = "SMOKE DIAGNOSTIC" if run_class == "smoke_diagnostic" else run_class
    interp_text = next(iter(interps)) if len(interps) == 1 else "see source data"
    ax.text(
        0.02,
        0.08,
        f"{badge}\n{interp_text}\nNo pooled row",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_TICK - 3,
        color=PALETTE["dark_gray"],
        linespacing=1.2,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 1.5,
        },
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
        _set_panel_title(ax, title)
        return

    y_labels = _participant_forest_labels(rows)
    y_pos = np.arange(len(rows), dtype=float)
    deltas = np.asarray([float(r.delta_p) for r in rows], dtype=float)  # type: ignore[attr-defined]
    ax.scatter(
        deltas,
        y_pos,
        s=SCATTER_SIZE,
        color=PALETTE["blue"],
        marker="o",
        edgecolors=PALETTE["dark_gray"],
        linewidths=0.6,
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
                markersize=MARKER_SIZE + 1.5,
                capsize=5,
                elinewidth=LINE_WIDTH,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.7,
                zorder=4,
            )
        else:
            ax.scatter(
                [mean],
                [summary_y],
                s=SCATTER_SIZE + 20,
                color=PALETTE["orange"],
                marker="D",
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.7,
                zorder=4,
            )
    _ref_vline(ax, 0.0)
    size_label = _as_str(getattr(inference, "sample_size_label", ""))
    if not size_label:
        size_label = f"n={int(getattr(inference, 'n_participants'))} participants"
    ax.set_yticks(list(y_pos) + [summary_y])
    ax.set_yticklabels(
        y_labels + [f"Mean Δ ({size_label})"],
        fontsize=FS_TICK - 1,
    )
    ax.set_xlabel(
        f"Participant Δ = observed − null mean ({ZLPI_METRIC})",
        fontsize=FS_AXIS,
        labelpad=8,
    )
    ax.set_ylabel(ylabel, fontsize=FS_AXIS, labelpad=6)
    run_class = _as_str(getattr(inference, "run_class", ""))
    interp = _as_str(getattr(inference, "interpretation", ""))
    badge = "SMOKE DIAGNOSTIC" if run_class == "smoke_diagnostic" else run_class
    ax.text(
        0.02,
        0.98,
        f"{badge}\n{interp}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FS_TICK - 2,
        color=PALETTE["dark_gray"],
    )
    _style_axes(ax)
    _set_panel_title(ax, title)
    if panel_label:
        _add_panel_label(ax, panel_label)


def _chunked(items: Sequence[object], size: int) -> list[list[object]]:
    if size <= 0:
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _render_figure3_participant_forest_supplement(
    analysis: object,
    output_dir: Path,
) -> tuple[list[tuple[Path, Path, Path]], list[Path], list[FigurePanelSource]]:
    """Faceted / paginated biological-participant forests (supplement)."""
    source_dir = output_dir / "source_data"
    source_paths: list[Path] = []
    panel_sources: list[FigurePanelSource] = []
    trios: list[tuple[Path, Path, Path]] = []

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
            "Figure 3 supplement — participant null forests",
            fontsize=FS_SUPTITLE - 2,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.01,
            (
                f"{size_label}; each row = one unique biological participant. "
                "Main Panel A aggregates to one row per dataset."
            ),
            ha="center",
            va="bottom",
            fontsize=FS_TICK - 2,
            color=PALETTE["dark_gray"],
        )
        fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.10)
        stem = (
            f"{FIGURE3_PARTICIPANT_FOREST_STEM}_{dataset_id}"
            if len(page_payloads) == 1
            else f"{FIGURE3_PARTICIPANT_FOREST_STEM}_{dataset_id}_p{page_idx:02d}"
        )
        trio = save_figure_trio(fig, output_dir, stem)
        trios.append(trio)
        plt.close(fig)

    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3_supplement",
            panel_id="participant_null_forests",
            title="Biological-participant null forests by dataset",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=PRIMARY_DURATION_S,
            input_tables=[],
            source_data_csv=str(source_dir / "figure3_panel_a_participant_deltas.csv"),
            analysis_keys=[
                "role=supplement",
                "unit=biological_participant",
                f"max_rows_per_page={FIGURE3_PARTICIPANT_FOREST_MAX_ROWS}",
            ],
            notes=(
                "Supplemental participant forests faceted by dataset and paginated "
                f"when n_participants > {FIGURE3_PARTICIPANT_FOREST_MAX_ROWS}. "
                + INDEPENDENT_UNIT_VERDICT
            ),
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
    fig = plt.figure(figsize=(14.0, 14.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.40, height_ratios=[1.15, 1.0, 1.0])

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
                s=SCATTER_SIZE,
                alpha=0.65,
                color=color,
                marker=marker,
                edgecolors=PALETTE["dark_gray"],
                linewidths=0.5,
                zorder=2,
                label=null_type.replace("_", " "),
            )
        lims = [
            float(np.nanmin([null_mean.min(), observed.min()])),
            float(np.nanmax([null_mean.max(), observed.max()])),
        ]
        ax_scatter.plot(lims, lims, color=REF_LINE_COLOR, ls="--", lw=1.0, zorder=1)
        ax_scatter.legend(
            loc="lower right",
            fontsize=FS_LEGEND - 2,
            frameon=True,
            title="null type",
        )
    else:
        _mark_empty_panel(
            ax_scatter,
            MSG_NOT_INCLUDED,
            xlabel=f"Null mean endpoint ({ZLPI_METRIC})",
            ylabel=f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
        )
    ax_scatter.set_xlabel(f"Null mean endpoint ({ZLPI_METRIC})", fontsize=FS_AXIS)
    ax_scatter.set_ylabel(
        f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
        fontsize=FS_AXIS,
    )
    _style_axes(ax_scatter)
    _set_panel_title(ax_scatter, FIGURE3_SUPPLEMENT_LABEL)
    _add_panel_label(ax_scatter, "S1")
    ax_scatter.text(
        0.01,
        0.98,
        (
            "Each point = observation × band × null type (nested).\n"
            "Not independent; no inference from density / diagonal.\n"
            "Confirmatory inference: participant-level Δ elsewhere."
        ),
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        fontsize=FS_TICK - 2,
        color=PALETTE["dark_gray"],
    )

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
        _plot_participant_null_forest(
            ax,
            participants,
            inference,
            title=f"Secondary: {null_type.replace('_', ' ')} (theta)",
            panel_label=f"S{idx + 2}",
        )

    fig.suptitle(
        "Figure 3 supplement — null diagnostics (not confirmatory Panel A)",
        fontsize=FS_SUPTITLE - 2,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.06, wspace=0.45, hspace=0.42)
    trio = save_figure_trio(fig, output_dir, FIGURE3_SUPPLEMENT_STEM)

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
                "role=supplement_diagnostic",
                "endpoint=zlpi",
                f"duration={PRIMARY_DURATION_S}",
                f"representation={PRIMARY_REPRESENTATION}",
                "not_for_confirmatory_inference=true",
            ],
            notes=(
                "Descriptive nested-observation diagnostic. Each point is "
                "observation×band×null type; points are not independent; no "
                "inference from density or proportion above y=x. All statistical "
                "inference comes from participant-level observed-minus-null Δ."
            ),
        )
    )
    # Retain secondary_records reference in notes via export already written by caller.
    _ = secondary_records
    return trio, source_paths, panel_sources


def render_figure3(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 3: nulls, modality, duration robustness, sensitivity matrix, LOO."""
    null_rows = read_csv_rows(inputs.get("null_subject"))
    modality = read_csv_rows(inputs.get("modality"))
    duration = read_csv_rows(inputs.get("duration_sensitivity"))
    paired_rows = read_csv_rows(inputs.get("paired_contrasts"))
    sensitivity = read_csv_rows(inputs.get("specification_matrix")) or read_csv_rows(
        inputs.get("sensitivity")
    )
    loo = read_csv_rows(inputs.get("leave_one_out"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

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
        title=(
            "Dataset Δ vs circular-shift null\n"
            f"(theta {_endpoint_display(ENDPOINT_ZLPI)}, D{PRIMARY_DURATION_S})"
        ),
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
                    markersize=MARKER_SIZE,
                    capsize=3,
                    elinewidth=1.4,
                    markeredgecolor=PALETTE["dark_gray"],
                    markeredgewidth=0.6,
                    alpha=0.92,
                )

        _ref_hline(ax_b, 0.0)
        ax_b.set_xlim(*FIGURE3_DURATION_XLIM)
        ax_b.set_xticks([60, 120, 180, 240])
        ax_b.set_ylim(y_lo, y_hi)
        ax_b.set_xlabel("Duration (s)", fontsize=FS_AXIS - 1)
        ax_b.set_ylabel(
            f"Participant mean Δ ({Z_YLABEL})",
            fontsize=FS_AXIS - 1,
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
                markersize=MARKER_SIZE,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
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
                markersize=MARKER_SIZE,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
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
            f"{FIGURE3_PANEL_B_FOOTNOTE_SHORT}\n"
            f"n = {n_units} participants"
            + (
                "; † D60 SWPI CIs clipped (source data)"
                if clipped_swpi
                else ""
            )
        )
    else:
        panel_b_footnote = ""
        _mark_empty_panel(
            ax_b,
            MSG_NOT_INCLUDED,
            xlabel="Duration (s)",
            ylabel=f"Participant mean Δ ({Z_YLABEL}; Student-t 95% CI)",
        )
    _set_panel_title(
        ax_b,
        "Duration sensitivity\n(participant-level; band × index)",
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

    # Panel C: matched ECG vs PPG.
    ax_c = fig.add_subplot(gs[1, 0])
    modality_source = []
    for row in modality:
        if str(row.get("matched", "")).lower() not in {"true", "1", "yes"}:
            continue
        delta = _as_float(row.get("delta_ecg_minus_ppg"))
        modality_source.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")),
                "participant_id": _as_str(row.get("participant_id")),
                "band": _as_str(row.get("band")),
                "endpoint_name": _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
                "delta_ecg_minus_ppg": delta,
            }
        )
    if modality_source:
        vals = np.asarray([r["delta_ecg_minus_ppg"] for r in modality_source], dtype=float)
        ax_c.hist(
            vals[np.isfinite(vals)],
            bins=15,
            color=PALETTE["purple"],
            alpha=0.85,
            edgecolor=PALETTE["dark_gray"],
            linewidth=0.6,
        )
        _ref_vline(ax_c, 0.0)
        ax_c.set_xlabel(f"ECG − PPG endpoint difference ({ZLPI_METRIC})", fontsize=FS_AXIS)
        ax_c.set_ylabel("Count", fontsize=FS_AXIS)
        _style_axes(ax_c)
        _set_panel_title(ax_c, f"Matched ECG−PPG (n = {int(np.isfinite(vals).sum())})", fontsize=FS_PANEL_TITLE - 2, pad=10)
        _add_panel_label(ax_c, "C")
    else:
        _mark_empty_panel(
            ax_c,
            MSG_NOT_APPLICABLE,
            xlabel=f"ECG − PPG endpoint difference ({ZLPI_METRIC})",
            ylabel="Count",
            xlim=(-1.0, 1.0),
            ylim=(0.0, 1.0),
        )
        _set_panel_title(ax_c, "Matched ECG−PPG", fontsize=FS_PANEL_TITLE - 2, pad=10)
        _add_panel_label(ax_c, "C")
    modality_csv = source_dir / "figure3_panel_c_modality.csv"
    write_source_csv(
        modality_csv,
        modality_source,
        (
            "dataset_id",
            "participant_id",
            "band",
            "endpoint_name",
            "delta_ecg_minus_ppg",
        ),
    )
    source_paths.append(modality_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="modality_ecg_ppg",
            title="Matched ECG versus PPG",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("modality") or "")],
            source_data_csv=str(modality_csv),
            analysis_keys=["matched_only=true"],
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
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
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
        ax_d.set_yticklabels(y_labels, fontsize=FS_TICK - 1)
        ax_d.set_ylabel("Spec.", fontsize=FS_AXIS, labelpad=6)
        ax_d.set_xlabel(
            f"Effect estimate ({ZLPI_METRIC}; {CI_95_LABEL})",
            fontsize=FS_AXIS,
            labelpad=8,
        )
        _style_axes(ax_d)
    else:
        _mark_empty_panel(
            ax_d,
            MSG_NOT_INCLUDED,
            xlabel=f"Effect estimate ({ZLPI_METRIC}; {CI_95_LABEL})",
            ylabel="Spec.",
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

    fig.suptitle(FIGURE3_TITLE, fontsize=FS_SUPTITLE, fontweight="bold", y=0.985)
    fig.subplots_adjust(**FIGURE3_SUBPLOT_ADJUST)
    if panel_b_footnote:
        # Figure footer keeps the A/B–C/D gutter free of colliding annotations.
        fig.text(
            0.5,
            0.035,
            panel_b_footnote.split("\n")[0],
            ha="center",
            va="bottom",
            fontsize=FS_TICK - 3,
            color=PALETTE["dark_gray"],
        )
        extra = " · ".join(
            line.strip()
            for line in panel_b_footnote.split("\n")[1:]
            if line.strip()
        )
        if extra:
            fig.text(
                0.5,
                0.012,
                extra,
                ha="center",
                va="bottom",
                fontsize=FS_TICK - 3,
                color=PALETTE["dark_gray"],
            )
    pdf, svg, png = save_figure_trio(fig, output_dir, FIGURE3_STEM)

    _supp_trio, supp_paths, supp_panels = _render_figure3_null_supplement(
        null_rows,
        output_dir,
        secondary_records=secondary_records,
    )
    source_paths.extend(supp_paths)
    panel_sources.extend(supp_panels)
    for path in _supp_trio:
        source_paths.append(path)

    _part_trios, part_paths, part_panels = _render_figure3_participant_forest_supplement(
        primary_analysis,
        output_dir,
    )
    source_paths.extend(part_paths)
    panel_sources.extend(part_panels)
    for trio in _part_trios:
        source_paths.extend(trio)

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
                "- Do not label n from protocol/session subject_ids (e.g. HIIT "
                "`01_ph` / `01_ps`).",
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
                "## Supplements",
                f"- Nested observed-vs-null scatter: `{FIGURE3_SUPPLEMENT_STEM}`",
                f"- Participant forests by dataset: `{FIGURE3_PARTICIPANT_FOREST_STEM}_*`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_paths.append(audit_note)

    return FigureArtifacts(
        figure_id="figure3",
        pdf=pdf,
        svg=svg,
        png=png,
        source_csvs=tuple(source_paths),
        panels=tuple(panel_sources),
    )


def generate_confirmatory_figures(
    confirmatory_root: str | Path,
    output_dir: str | Path,
) -> FiguresResult:
    """Generate Figures 1–3 and the figure-source manifest from frozen outputs."""
    root = Path(confirmatory_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = resolve_reporting_inputs(root)

    figure1 = render_figure1(inputs, out)
    figure2 = render_figure2(inputs, out)
    figure3 = render_figure3(inputs, out)
    panels = figure1.panels + figure2.panels + figure3.panels

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
        figure_source_manifest=Path(manifest_payload["manifest_path"]),
        panel_records=panels,
    )


__all__ = [
    "FIGURE1_STEM",
    "FIGURE2_STEM",
    "FIGURE3_STEM",
    "FIGURE3_SUPPLEMENT_STEM",
    "FIGURE3_PARTICIPANT_FOREST_STEM",
    "FIGURE_DPI",
    "FigureArtifacts",
    "FiguresResult",
    "generate_confirmatory_figures",
    "mean_ci_by_lag",
    "resolve_reporting_inputs",
    "save_figure_trio",
]
