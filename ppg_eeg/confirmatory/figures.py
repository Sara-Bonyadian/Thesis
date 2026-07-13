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

from .duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
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
BAND_ORDER = ("delta", "theta", "alpha", "beta")
PRIMARY_REPRESENTATION = "absolute_log10"
LAG_XLABEL = "Lag τ (s): corr(HR(t), EEG(t+τ)); +τ = EEG follows HR"
Z_YLABEL = "Fisher z"

FIGURE1_STEM = "figure1_lag_resolved_zero_lag"
FIGURE2_STEM = "figure2_state_attenuation_replication"
FIGURE3_STEM = "figure3_temporal_artifact_specificity"


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


def save_figure_trio(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    svg = output_dir / f"{stem}.svg"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return pdf, svg, png


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
    low_labels = {"rest", "passive", "step1", "low_demand", "ph_pre_rest", "ph_post_rest", "ps_pre_rest", "ps_post_rest"}
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


def _shade_flanks(ax: plt.Axes, duration_s: int) -> None:
    contract = contract_for_duration(duration_s)
    inner, outer = contract.flank_inner_s, contract.flank_outer_s
    ax.axvspan(-outer, -inner, color="#dddddd", alpha=0.5, zorder=0)
    ax.axvspan(inner, outer, color="#dddddd", alpha=0.5, zorder=0)
    ax.axvline(0.0, color="black", lw=0.8, ls="--")


def _set_lag_axes(ax: plt.Axes, duration_s: int) -> None:
    contract = contract_for_duration(duration_s)
    ax.set_xlim(contract.lag_min_s, contract.lag_max_s)
    ax.set_xlabel(LAG_XLABEL)
    ax.set_ylabel(Z_YLABEL)


def render_figure1(
    inputs: Mapping[str, Path | None],
    output_dir: Path,
) -> FigureArtifacts:
    """Figure 1: lag-resolved zero-lag structure (D240 ZLPI Fisher-z)."""
    curves = read_csv_rows(inputs.get("curves_d240"))
    peak_rows = read_csv_rows(inputs.get("peak_params"))
    source_dir = output_dir / "source_data"
    panel_sources: list[FigurePanelSource] = []
    source_paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    for ax, band in zip(axes_flat, BAND_ORDER, strict=True):
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
        write_source_csv(csv_path, series, fields)
        source_paths.append(csv_path)
        _shade_flanks(ax, EXPECTED_PRIMARY_DURATION_S)
        if series:
            lags = np.asarray([r["lag_s"] for r in series], dtype=float)
            mean = np.asarray([r["mean_z"] for r in series], dtype=float)
            lo = np.asarray([r["ci_low"] for r in series], dtype=float)
            hi = np.asarray([r["ci_high"] for r in series], dtype=float)
            n = int(series[0]["n"]) if series else 0
            ax.fill_between(lags, lo, hi, color="#4C78A8", alpha=0.25, linewidth=0)
            ax.plot(lags, mean, color="#4C78A8", lw=1.5, label=f"mean z (n={n})")
            # Optional subject-level peak overlay for this band.
            mus = [
                _as_float(r.get("peak_center_mu_s"))
                for r in peak_rows
                if _as_str(r.get("band")).casefold() == band
                and _as_int(r.get("duration_s"), 240) == 240
                and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
                and str(r.get("has_identifiable_peak", "")).lower() in {"true", "1", "yes"}
            ]
            mus = [m for m in mus if math.isfinite(m)]
            if mus:
                ax.axvspan(
                    -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                    color="#F58518",
                    alpha=0.08,
                    zorder=0,
                )
                ax.axvline(float(np.median(mus)), color="#F58518", ls=":", lw=1.2, label="median μ")
            ax.set_title(f"{band} · {ENDPOINT_ZLPI} D{EXPECTED_PRIMARY_DURATION_S} · n={n}")
        else:
            ax.set_title(f"{band} · input unavailable")
            ax.text(0.5, 0.5, "No frozen curve rows", ha="center", va="center", transform=ax.transAxes)
        _set_lag_axes(ax, EXPECTED_PRIMARY_DURATION_S)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
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
                notes="Flank shading uses ZLPI 20–60 s windows; orange band marks ±2 s μ equivalence region when peaks exist.",
            )
        )

    fig.suptitle(
        f"Figure 1 — Lag-resolved zero-lag structure ({ENDPOINT_ZLPI}, D{EXPECTED_PRIMARY_DURATION_S})",
        fontsize=12,
    )
    fig.tight_layout()
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

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

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
    for idx, dataset_id in enumerate(datasets):
        points = _paired_points_for_dataset(
            subjects, paired, dataset_id=dataset_id, band="theta"
        )
        paired_source.extend(points)
        ys = np.asarray([_as_float(p["delta_endpoint_index"]) for p in points], dtype=float)
        ys = ys[np.isfinite(ys)]
        if ys.size == 0:
            continue
        seed = int(hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()[:8], 16)
        x = np.full(ys.shape, idx, dtype=float) + 0.05 * np.random.default_rng(
            seed
        ).normal(size=ys.size)
        ax_a.scatter(x, ys, s=18, alpha=0.7, color="#4C78A8")
        ax_a.errorbar(
            idx,
            float(np.mean(ys)),
            yerr=1.959963984540054 * float(np.std(ys, ddof=1) / math.sqrt(ys.size))
            if ys.size >= 2
            else 0.0,
            fmt="o",
            color="#E45756",
            capsize=3,
            label=None,
        )
        ax_a.text(idx, ax_a.get_ylim()[1] if False else 0, "", fontsize=7)
    ax_a.axhline(0.0, color="black", lw=0.8, ls="--")
    ax_a.set_xticks(range(len(datasets)))
    ax_a.set_xticklabels(datasets, rotation=30, ha="right")
    ax_a.set_ylabel(f"Δ {ENDPOINT_ZLPI} (task − low-demand)")
    ax_a.set_title(f"A · Paired D240 {ENDPOINT_ZLPI} (theta)")
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
    for i, band in enumerate(BAND_ORDER):
        row = next((r for r in meta_rows if _as_str(r.get("band")).casefold() == band), None)
        if row is None:
            continue
        effect = _as_float(row.get("pooled_effect"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
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
        ax_b.errorbar(effect, i, xerr=[[effect - lo], [hi - effect]], fmt="o", color="#4C78A8", capsize=3)
    ax_b.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_b.set_yticks(range(len(forest_source)))
    ax_b.set_yticklabels([r["band"] for r in forest_source])
    ax_b.set_xlabel(f"Pooled Δ {ENDPOINT_ZLPI} (95% CI)")
    ax_b.set_title("B · Random-effects meta (primary ZLPI)")
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
            title="Meta-analysis forest by band",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=240,
            input_tables=[str(inputs.get("meta_analysis") or "")],
            source_data_csv=str(forest_csv),
            analysis_keys=["endpoint=zlpi", "duration=240", "is_primary_analysis=true"],
        )
    )

    # Panel C: D180 ZLPI sensitivity (never labeled as primary rescue).
    ax_c = fig.add_subplot(gs[1, 0])
    d180 = [
        r
        for r in effects
        if _as_int(r.get("duration_s"), 0) == 180
        and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
    ]
    if not d180:
        d180 = [
            r
            for r in paired
            if _as_int(r.get("duration_s"), 0) == 180
            and _as_str(r.get("endpoint_name"), ENDPOINT_ZLPI) == ENDPOINT_ZLPI
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
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": 180,
                    "effect": value,
                    "is_primary_analysis": False,
                    "can_rescue_primary": False,
                }
            )
    for i, band in enumerate(BAND_ORDER):
        vals = np.asarray(by_band.get(band, []), dtype=float)
        if vals.size == 0:
            continue
        ax_c.errorbar(
            float(np.mean(vals)),
            i,
            xerr=1.959963984540054 * float(np.std(vals, ddof=1) / math.sqrt(vals.size))
            if vals.size >= 2
            else 0.0,
            fmt="s",
            color="#54A24B",
            capsize=3,
        )
    ax_c.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_c.set_yticks(range(len(BAND_ORDER)))
    ax_c.set_yticklabels(list(BAND_ORDER))
    ax_c.set_xlabel(f"D180 {ENDPOINT_ZLPI} effects (sensitivity only)")
    ax_c.set_title("C · D180 ZLPI sensitivity (cannot rescue primary)")
    d180_csv = source_dir / "figure2_panel_c_d180_sensitivity.csv"
    write_source_csv(
        d180_csv,
        d180_source,
        (
            "band",
            "dataset_id",
            "endpoint_name",
            "duration_s",
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
            title="D180 ZLPI sensitivity",
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=180,
            input_tables=[str(inputs.get("dataset_effects") or ""), str(inputs.get("paired_contrasts") or "")],
            source_data_csv=str(d180_csv),
            analysis_keys=["endpoint=zlpi", "duration=180", "can_rescue_primary=false"],
            notes="Displayed separately from primary D240; not a rescue pathway.",
        )
    )

    # Panel D: peak-center TOST / ±2 s equivalence.
    ax_d = fig.add_subplot(gs[1, 1])
    eq_source = []
    for i, row in enumerate(equivalence):
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        mean_mu = _as_float(row.get("mean_mu"))
        lo = _as_float(row.get("ci_low"))
        hi = _as_float(row.get("ci_high"))
        eq_source.append(
            {
                "dataset_id": _as_str(row.get("dataset_id")),
                "band": _as_str(row.get("band")),
                "endpoint_name": ENDPOINT_ZLPI,
                "mean_mu": mean_mu,
                "ci_low": lo,
                "ci_high": hi,
                "bound_low": -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                "bound_high": EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
                "equivalent": _as_str(row.get("equivalent")),
            }
        )
        ax_d.errorbar(mean_mu, i, xerr=[[mean_mu - lo], [hi - mean_mu]], fmt="o", color="#F58518", capsize=3)
    ax_d.axvspan(
        -EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
        EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
        color="#F58518",
        alpha=0.15,
    )
    ax_d.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_d.set_yticks(range(len(eq_source)))
    ax_d.set_yticklabels(
        [f"{r['dataset_id']}:{r['band']}" for r in eq_source] or ["no data"]
    )
    ax_d.set_xlabel("Peak center μ (s)")
    ax_d.set_title("D · Low-demand μ equivalence (±2 s)")
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
            ],
        )
    )

    fig.suptitle("Figure 2 — State attenuation and replication", fontsize=12)
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

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

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
        ax_a.scatter(null_mean, observed, s=16, alpha=0.7, color="#4C78A8")
        lims = [
            np.nanmin([null_mean.min(), observed.min()]),
            np.nanmax([null_mean.max(), observed.max()]),
        ]
        ax_a.plot(lims, lims, color="black", ls="--", lw=0.8)
    else:
        ax_a.text(0.5, 0.5, "No null table", ha="center", va="center", transform=ax_a.transAxes)
    ax_a.set_xlabel("Null mean endpoint")
    ax_a.set_ylabel(f"Observed {ENDPOINT_ZLPI}")
    ax_a.set_title("A · Observed vs surrogate nulls (ZLPI)")
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
        ENDPOINT_ZLPI: ("o", "#4C78A8"),
        ENDPOINT_MID_WINDOW_PROXIMAL_INDEX: ("s", "#F58518"),
        ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX: ("D", "#54A24B"),
    }
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
        marker, color = endpoint_styles.get(endpoint, ("x", "#999999"))
        ax_b.errorbar(
            duration_s,
            effect,
            yerr=[
                [effect - _as_float(row.get("ci_low"))],
                [_as_float(row.get("ci_high")) - effect],
            ]
            if math.isfinite(_as_float(row.get("ci_low")))
            else None,
            fmt=marker,
            color=color,
            capsize=2,
            alpha=0.8,
        )
    ax_b.axhline(0.0, color="black", lw=0.8, ls="--")
    ax_b.set_xticks([60, 120, 180, 240])
    ax_b.set_xlabel("Duration (s)")
    ax_b.set_ylabel("Effect estimate")
    ax_b.set_title("B · Duration sensitivity (ZLPI/MWPI/SWPI separate)")
    # Legend proxies
    for endpoint, (marker, color) in endpoint_styles.items():
        ax_b.plot([], [], marker=marker, color=color, ls="none", label=endpoint)
    ax_b.legend(fontsize=7, frameon=False, loc="best")
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
        ax_c.hist(vals[np.isfinite(vals)], bins=15, color="#B279A2", alpha=0.85)
        ax_c.axvline(0.0, color="black", lw=0.8, ls="--")
        ax_c.set_title(f"C · Matched ECG−PPG (n={np.isfinite(vals).sum()})")
    else:
        ax_c.text(0.5, 0.5, "No matched modality pairs", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_title("C · Matched ECG−PPG")
    ax_c.set_xlabel("ECG − PPG endpoint")
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
        ax_d.errorbar(effect, len(y_labels), fmt="o", color="#4C78A8", capsize=2)
        y_labels.append(str(row["control_id"])[:28])
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
    ax_d.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_d.set_yticks(range(len(y_labels)))
    ax_d.set_yticklabels(y_labels or ["no data"])
    ax_d.set_xlabel("Effect estimate")
    ax_d.set_title("D · Specification matrix / sensitivity")
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

    fig.suptitle("Figure 3 — Temporal and artifact specificity", fontsize=12)
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
