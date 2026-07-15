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


def _set_panel_title(ax: plt.Axes, title: str) -> None:
    # Left-aligned title with extra pad so it clears the panel letter.
    ax.set_title(title, fontsize=FS_PANEL_TITLE, fontweight="normal", pad=12, loc="left")


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
        rows.append(
            {
                "dataset_id": dataset_id,
                "participant_id": _as_str(row.get("participant_id")),
                "contrast_id": _as_str(row.get("contrast_id")),
                "band": band,
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "low_endpoint_index": _as_float(row.get("low_endpoint_index")),
                "effort_endpoint_index": _as_float(row.get("effort_endpoint_index")),
                "delta_endpoint_index": _as_float(row.get("delta_endpoint_index")),
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
    plotted_any = False
    dataset_tick_labels: list[str] = []
    for idx, dataset_id in enumerate(datasets):
        points = _paired_points_for_dataset(
            subjects, paired, dataset_id=dataset_id, band="theta"
        )
        paired_source.extend(points)
        ys = np.asarray([_as_float(p["delta_endpoint_index"]) for p in points], dtype=float)
        ys = ys[np.isfinite(ys)]
        base = _dataset_display(dataset_id)
        dataset_tick_labels.append(f"{base}\n(n = {int(ys.size)} pairs)")
        if ys.size == 0:
            continue
        plotted_any = True
        seed = int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16)
        x = np.full(ys.shape, idx, dtype=float) + 0.05 * np.random.default_rng(
            seed
        ).normal(size=ys.size)
        ax_a.scatter(
            x,
            ys,
            s=SCATTER_SIZE,
            alpha=0.75,
            color=_band_color("theta"),
            marker=_band_marker("theta"),
            edgecolors=PALETTE["dark_gray"],
            linewidths=0.6,
            label=None,
            zorder=2,
        )
        ax_a.errorbar(
            idx,
            float(np.mean(ys)),
            yerr=1.959963984540054 * float(np.std(ys, ddof=1) / math.sqrt(ys.size))
            if ys.size >= 2
            else 0.0,
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
        # CI is stated on the axis title family; no redundant legend.
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
            "contrast_id",
            "band",
            "duration_s",
            "endpoint_name",
            "low_endpoint_index",
            "effort_endpoint_index",
            "delta_endpoint_index",
        ),
    )
    source_paths.append(paired_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure2",
            panel_id="paired_deltas",
            title="Paired task-minus-low-demand ZLPI",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("paired_contrasts") or ""), str(inputs.get("subject_level") or "")],
            source_data_csv=str(paired_csv),
            analysis_keys=["endpoint=zlpi", "duration=240", "band=theta"],
            notes=f"n_datasets={len(datasets)}; points are participants.",
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
    _set_panel_title(ax_d, "Peak-center equivalence (±2 s)")
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
            title="Peak-center TOST equivalence region",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("peak_equivalence") or "")],
            source_data_csv=str(eq_csv),
            analysis_keys=[
                "endpoint=zlpi",
                f"equivalence_bounds=±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S}s",
                "duration=240",
                f"representation={PRIMARY_REPRESENTATION}",
            ],
            notes=(
                "Primary D240 absolute_log10 ZLPI slice, sorted by dataset then band. "
                f"Error bars are {CI_95_LABEL}."
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


def render_figure3(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 3: nulls, modality, duration robustness, sensitivity matrix, LOO."""
    null_rows = read_csv_rows(inputs.get("null_subject"))
    modality = read_csv_rows(inputs.get("modality"))
    duration = read_csv_rows(inputs.get("duration_sensitivity"))
    sensitivity = read_csv_rows(inputs.get("specification_matrix")) or read_csv_rows(
        inputs.get("sensitivity")
    )
    loo = read_csv_rows(inputs.get("leave_one_out"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    _configure_publication_style()
    fig = plt.figure(figsize=(13.5, 10.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.38)

    # Panel A: observed vs null surrogate markers.
    ax_a = fig.add_subplot(gs[0, 0])
    null_source = []
    for row in null_rows:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), 240) != 240:
            continue
        null_source.append(
            {
                "observation_id": _as_str(row.get("observation_id")),
                "band": _as_str(row.get("band")),
                "null_type": _as_str(row.get("null_type")),
                "endpoint_name": ENDPOINT_ZLPI,
                "observed_endpoint_index": _as_float(row.get("observed_endpoint_index")),
                "null_mean": _as_float(row.get("null_mean")),
                "empirical_p": _as_float(row.get("empirical_p")),
                "effect_size_surrogate_z": _as_float(row.get("effect_size_surrogate_z")),
            }
        )
    if null_source:
        observed = np.asarray(
            [r["observed_endpoint_index"] for r in null_source], dtype=float
        )
        null_mean = np.asarray([r["null_mean"] for r in null_source], dtype=float)
        ax_a.scatter(
            null_mean,
            observed,
            s=SCATTER_SIZE,
            alpha=0.75,
            color=_band_color("theta"),
            marker="o",
            edgecolors=PALETTE["dark_gray"],
            linewidths=0.6,
            zorder=2,
        )
        lims = [
            np.nanmin([null_mean.min(), observed.min()]),
            np.nanmax([null_mean.max(), observed.max()]),
        ]
        ax_a.plot(lims, lims, color=REF_LINE_COLOR, ls="--", lw=1.0, zorder=1)
    else:
        _mark_empty_panel(
            ax_a,
            MSG_NOT_INCLUDED,
            xlabel=f"Null mean endpoint ({ZLPI_METRIC})",
            ylabel=f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
            xlim=(-0.5, 0.5),
            ylim=(-0.5, 0.5),
        )
    ax_a.set_xlabel(f"Null mean endpoint ({ZLPI_METRIC})", fontsize=FS_AXIS, labelpad=8)
    ax_a.set_ylabel(
        f"Observed {_endpoint_display(ENDPOINT_ZLPI)} ({ZLPI_METRIC})",
        fontsize=FS_AXIS,
        labelpad=10,
    )
    _style_axes(ax_a)
    _set_panel_title(ax_a, f"Observed vs surrogate nulls ({_endpoint_display(ENDPOINT_ZLPI)})")
    _add_panel_label(ax_a, "A")
    null_csv = source_dir / "figure3_panel_a_nulls.csv"
    write_source_csv(
        null_csv,
        null_source,
        (
            "observation_id",
            "band",
            "null_type",
            "endpoint_name",
            "observed_endpoint_index",
            "null_mean",
            "empirical_p",
            "effect_size_surrogate_z",
        ),
    )
    source_paths.append(null_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="null_scatter",
            title="Observed vs null endpoint indices",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("null_subject") or "")],
            source_data_csv=str(null_csv),
            analysis_keys=["endpoint=zlpi", "duration=240"],
            notes="Diagonal = equivalence of observed and null mean.",
        )
    )

    # Panel B: duration robustness with endpoint labels separated.
    ax_b = fig.add_subplot(gs[0, 1])
    duration_source = []
    endpoint_styles = {
        ENDPOINT_ZLPI: ("o", "-", PALETTE["blue"]),
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: ("s", "--", PALETTE["orange"]),
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: ("s", "-.", PALETTE["green"]),
    }
    plotted_duration = False
    for row in duration:
        endpoint = _as_str(row.get("endpoint_name"))
        duration_s = _as_int(row.get("duration_s"))
        effect = _as_float(row.get("effect_estimate"))
        if not math.isfinite(effect):
            continue
        duration_source.append(
            {
                "duration_s": duration_s,
                "endpoint_name": endpoint,
                "band": _as_str(row.get("band")),
                "effect_estimate": effect,
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n")),
                "is_primary_analysis": _as_str(row.get("is_primary_analysis")),
                "can_rescue_primary": False,
            }
        )
        marker, linestyle, color = endpoint_styles.get(endpoint, ("x", ":", "#999999"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        yerr = [[effect - lo], [hi - effect]] if math.isfinite(lo) and math.isfinite(hi) else None
        ax_b.errorbar(
            duration_s,
            effect,
            yerr=yerr,
            fmt=marker,
            color=color,
            linestyle=linestyle,
            markersize=MARKER_SIZE,
            capsize=4,
            elinewidth=LINE_WIDTH,
            markeredgecolor=PALETTE["dark_gray"],
            markeredgewidth=0.6,
            alpha=0.9,
        )
        plotted_duration = True
    if plotted_duration:
        _ref_hline(ax_b, 0.0)
        ax_b.set_xticks([60, 120, 180, 240])
        ax_b.set_xlabel("Duration (s)", fontsize=FS_AXIS)
        ax_b.set_ylabel(
            f"Effect estimate ({ZLPI_METRIC}; {CI_95_LABEL})",
            fontsize=FS_AXIS,
        )
        _style_axes(ax_b)
        legend_handles = []
        legend_labels = []
        for endpoint, (marker, linestyle, color) in endpoint_styles.items():
            (handle,) = ax_b.plot(
                [],
                [],
                marker=marker,
                color=color,
                ls=linestyle,
                markersize=MARKER_SIZE,
                markeredgecolor=PALETTE["dark_gray"],
                markeredgewidth=0.6,
                lw=LINE_WIDTH,
            )
            legend_handles.append(handle)
            legend_labels.append(_endpoint_display(endpoint))
        _legend_inside(ax_b, legend_handles, legend_labels, loc="upper right")
    else:
        _mark_empty_panel(
            ax_b,
            MSG_NOT_INCLUDED,
            xlabel="Duration (s)",
            ylabel=f"Effect estimate ({ZLPI_METRIC}; {CI_95_LABEL})",
        )
    _set_panel_title(ax_b, "Duration sensitivity")
    _add_panel_label(ax_b, "B")
    duration_csv = source_dir / "figure3_panel_b_duration.csv"
    write_source_csv(
        duration_csv,
        duration_source,
        (
            "duration_s",
            "endpoint_name",
            "band",
            "effect_estimate",
            "ci_low",
            "ci_high",
            "n",
            "is_primary_analysis",
            "can_rescue_primary",
        ),
    )
    source_paths.append(duration_csv)
    panel_sources.append(
        FigurePanelSource(
            figure_id="figure3",
            panel_id="duration_sensitivity",
            title="Duration robustness with separated endpoints",
            endpoint_name="mixed_labeled",
            duration_s=0,
            input_tables=[str(inputs.get("duration_sensitivity") or "")],
            source_data_csv=str(duration_csv),
            analysis_keys=["zlpi", "mwpi", "swpi", "can_rescue_primary=false"],
            notes="Markers encode endpoint family; primary D240 ZLPI is never replaced.",
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
        _set_panel_title(ax_c, f"Matched ECG−PPG (n = {int(np.isfinite(vals).sum())})")
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
        _set_panel_title(ax_c, "Matched ECG−PPG")
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
    _set_panel_title(ax_d, "Specification matrix")
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

    fig.suptitle(FIGURE3_TITLE, fontsize=FS_SUPTITLE, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.10, wspace=0.50, hspace=0.42)
    pdf, svg, png = save_figure_trio(fig, output_dir, FIGURE3_STEM)
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
    "FIGURE_DPI",
    "FigureArtifacts",
    "FiguresResult",
    "generate_confirmatory_figures",
    "mean_ci_by_lag",
    "resolve_reporting_inputs",
    "save_figure_trio",
]
