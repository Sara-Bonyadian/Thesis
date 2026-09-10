"""ds003816 D60 SWPI descriptive/QC figures (sensitivity only).

This package is intentionally separate from manuscript Figures 1–3.
ds003816 has no prespecified paired contrast, does not enter the meta-analysis,
and does not support D240 ZLPI, C4 temporal-null, or Option-C peak timing.

Rebuilds from frozen C0/C2/C3/C5 (C1b optional for the analysis funnel).
Does not rerun C0–C6 and does not emit paired-Δ, μ/FWHM, ±2 s, or null claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .duration_contracts import contract_for_duration
from .endpoints import fisher_z
from .figures import (
    BAND_COLORS,
    BAND_LINESTYLES,
    BAND_MARKERS,
    BAND_ORDER,
    FIGURE_DPI,
    FS_AXIS,
    FS_LEGEND,
    FS_PANEL_TITLE,
    FS_SUPTITLE,
    PALETTE,
    PRIMARY_REPRESENTATION,
    _as_bool,
    _as_float,
    _as_int,
    _as_str,
    _band_display,
    _configure_publication_style,
    save_figure_trio,
    write_source_csv,
)

DATASET_ID = "ds003816"
DURATION_S = 60
ENDPOINT_ALIAS = "SWPI"
ROLE_BANNER = (
    "Sensitivity / descriptive only  ·  Not pooled  ·  No prespecified paired contrast"
)
ROLE_SUBTITLE = (
    "ds003816 · D60 only · endpoint = SWPI · enters_meta = false"
)
Z_95 = 1.959963984540054

STEM_LAG = "ds003816_d60_lag_resolved_swpi"
STEM_BAND = "ds003816_d60_band_summary"
STEM_ALPHA = "ds003816_alpha_d60_descriptive_summary"
STEM_QC = "ds003816_duration_common_support_qc"
REQUIRED_STEMS = (STEM_LAG, STEM_BAND, STEM_ALPHA, STEM_QC)

_FROZEN_SOURCES = (
    "C2/confirmatory_cross_correlation_curves_D60.csv",
    "C2/confirmatory_cross_correlation_qc_D60.csv",
    "C3/confirmatory_endpoint_metrics_D60.csv",
    "C3/confirmatory_endpoint_qc_D60.csv",
    "C5/subject_level_metrics.csv",
    "C5/pairing_qc.csv",
    "C0/data_audit.csv",
    "C0/eligibility_by_duration.csv",
)


def _sha256_prefix(path: Path, *, n: int = 16) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n]


def _write_caption(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    return path


def _mean_ci(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean = float(np.mean(arr))
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    se = float(np.std(arr, ddof=1) / math.sqrt(n))
    half = Z_95 * se
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
    }


def observation_id_from_row(row: Mapping[str, object]) -> str:
    for key in ("observation_id", "observation_ids"):
        raw = _as_str(row.get(key)).strip()
        if raw:
            return raw.split(",")[0].strip()
    return ""


def biological_id_from_row(row: Mapping[str, object]) -> str:
    pid = _as_str(row.get("participant_id")).strip()
    if pid:
        return pid
    observation_id = observation_id_from_row(row)
    parts = observation_id.split("-")
    if len(parts) >= 2 and parts[0].casefold() == DATASET_ID:
        return parts[1]
    subject_id = _as_str(row.get("subject_id"))
    if subject_id:
        return subject_id.split("_")[0]
    return observation_id


def _is_primary_abs_swpi_d60(row: Mapping[str, object]) -> bool:
    if _as_str(row.get("dataset_id")).casefold() not in {"", DATASET_ID}:
        return False
    if _as_int(row.get("duration_s"), DURATION_S) != DURATION_S:
        return False
    representation = _as_str(
        row.get("power_representation"), PRIMARY_REPRESENTATION
    ).casefold()
    if representation != PRIMARY_REPRESENTATION:
        return False
    alias = _as_str(row.get("endpoint_alias") or row.get("endpoint_name")).casefold()
    return alias in {"swpi", "short_window_proximal_index"}


def _eligible_row(row: Mapping[str, object]) -> bool:
    if "eligible" in row:
        return _as_bool(row.get("eligible"), False)
    if "endpoint_eligible" in row:
        return _as_bool(row.get("endpoint_eligible"), False)
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _apply_role_banner(fig: plt.Figure, *, y: float = 0.985) -> None:
    fig.suptitle(ROLE_BANNER, fontsize=FS_SUPTITLE - 6, fontweight="bold", y=y, color=PALETTE["vermillion"])
    fig.text(
        0.5,
        y - 0.035,
        ROLE_SUBTITLE,
        ha="center",
        va="top",
        fontsize=FS_AXIS - 4,
        color=PALETTE["dark_gray"],
    )


def _shade_swpi_flanks(ax: plt.Axes) -> None:
    contract = contract_for_duration(DURATION_S)
    inner, outer = contract.flank_inner_s, contract.flank_outer_s
    ax.axvspan(-outer, -inner, color=PALETTE["flank"], alpha=0.85, zorder=0, linewidth=0)
    ax.axvspan(inner, outer, color=PALETTE["flank"], alpha=0.85, zorder=0, linewidth=0)
    ax.axvline(0.0, color=PALETTE["light_gray"], linewidth=1.0, zorder=1)


def _load_subject_swpi(c5_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _read_csv(c5_path):
        if not _is_primary_abs_swpi_d60(row) or not _eligible_row(row):
            continue
        value = _as_float(row.get("endpoint_index"))
        if not math.isfinite(value):
            continue
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "observation_id": observation_id_from_row(row),
                "participant_id": biological_id_from_row(row),
                "condition": _as_str(row.get("condition") or row.get("task")),
                "band": _as_str(row.get("band")).casefold(),
                "duration_s": DURATION_S,
                "endpoint_alias": ENDPOINT_ALIAS,
                "power_representation": PRIMARY_REPRESENTATION,
                "n_common_support": _as_int(row.get("n_common_support")),
                "swpi": value,
                "estimand": "observation_level_swpi",
                "scientific_role": "sensitivity_descriptive_only",
                "enters_meta": False,
                "prespecified_paired_contrast": False,
            }
        )
    return rows


def _participant_means(rows: Sequence[Mapping[str, object]], *, band: str) -> list[float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if _as_str(row.get("band")).casefold() != band.casefold():
            continue
        pid = _as_str(row.get("participant_id"))
        value = _as_float(row.get("swpi"))
        if pid and math.isfinite(value):
            buckets[pid].append(value)
    return [float(np.mean(values)) for values in buckets.values() if values]


def _aggregate_lag_curves(curves_path: Path) -> list[dict[str, object]]:
    """Participant-clustered mean Fisher-z at each D60 lag, by band."""
    per_bio: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_obs: dict[str, set[str]] = defaultdict(set)
    with curves_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not _is_primary_abs_swpi_d60(row):
                continue
            band = _as_str(row.get("band")).casefold()
            if band not in BAND_ORDER:
                continue
            r = _as_float(row.get("r"))
            if not math.isfinite(r):
                continue
            z = fisher_z(r)
            if not math.isfinite(z):
                continue
            lag = int(round(_as_float(row.get("lag_s"))))
            pid = biological_id_from_row(row)
            per_bio[(band, lag)][pid].append(z)
            n_obs[band].add(observation_id_from_row(row))
    out: list[dict[str, object]] = []
    contract = contract_for_duration(DURATION_S)
    for band in BAND_ORDER:
        lags = sorted({lag for (b, lag) in per_bio if b == band})
        for lag in lags:
            bio_means = [
                float(np.mean(values))
                for values in per_bio[(band, lag)].values()
                if values
            ]
            stats = _mean_ci(bio_means)
            out.append(
                {
                    "dataset_id": DATASET_ID,
                    "band": band,
                    "lag_s": lag,
                    "duration_s": DURATION_S,
                    "endpoint_alias": ENDPOINT_ALIAS,
                    "power_representation": PRIMARY_REPRESENTATION,
                    "estimand": "participant_mean_fisher_z",
                    "n_participants": stats["n"],
                    "n_observations": len(n_obs[band]),
                    "mean_z": stats["mean"],
                    "se": stats["se"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "flank_inner_s": contract.flank_inner_s,
                    "flank_outer_s": contract.flank_outer_s,
                    "scientific_role": "sensitivity_descriptive_only",
                    "enters_meta": False,
                    "prespecified_paired_contrast": False,
                    "pooled": False,
                }
            )
    return out


def _render_lag_figure(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_publication_style()
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    _shade_swpi_flanks(ax)
    for band in BAND_ORDER:
        band_rows = [row for row in rows if _as_str(row.get("band")) == band]
        if not band_rows:
            continue
        lags = np.asarray([_as_float(row["lag_s"]) for row in band_rows], dtype=float)
        mean = np.asarray([_as_float(row["mean_z"]) for row in band_rows], dtype=float)
        lo = np.asarray([_as_float(row["ci_low"]) for row in band_rows], dtype=float)
        hi = np.asarray([_as_float(row["ci_high"]) for row in band_rows], dtype=float)
        color = BAND_COLORS.get(band, PALETTE["blue"])
        ax.fill_between(lags, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(
            lags,
            mean,
            color=color,
            linestyle=BAND_LINESTYLES.get(band, "-"),
            marker=BAND_MARKERS.get(band, "o"),
            markevery=4,
            label=_band_display(band),
            zorder=3,
        )
    n_part = next((_as_int(row.get("n_participants")) for row in rows), 0)
    n_obs = next((_as_int(row.get("n_observations")) for row in rows), 0)
    ax.set_xlim(-20.5, 20.5)
    ax.set_xlabel("Lag τ (s)")
    ax.set_ylabel("Mean Fisher-z (participant-clustered)")
    ax.set_title(
        f"D60 lag-resolved SWPI  ·  all conditions  ·  N={n_part} participants / {n_obs} observations",
        fontsize=FS_PANEL_TITLE - 4,
        pad=10,
    )
    ax.legend(frameon=False, loc="upper right", fontsize=FS_LEGEND)
    ax.text(
        0.02,
        0.04,
        "Shaded |τ|∈[10, 20] s = SWPI distant flanks.\n"
        "Descriptive mean ± 95% CI across participants.\n"
        "Not a paired contrast; not pooled with ZLPI.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=FS_LEGEND - 2,
        color=PALETTE["dark_gray"],
    )
    _apply_role_banner(fig)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.98)
    pdf, svg, png = save_figure_trio(fig, output_dir, STEM_LAG)
    source = write_source_csv(
        output_dir / "source_data" / f"{STEM_LAG}.csv",
        rows,
        fieldnames=[
            "dataset_id",
            "band",
            "lag_s",
            "duration_s",
            "endpoint_alias",
            "power_representation",
            "estimand",
            "n_participants",
            "n_observations",
            "mean_z",
            "se",
            "ci_low",
            "ci_high",
            "flank_inner_s",
            "flank_outer_s",
            "scientific_role",
            "enters_meta",
            "prespecified_paired_contrast",
            "pooled",
        ],
    )
    caption = _write_caption(
        output_dir / f"{STEM_LAG}_caption.txt",
        (
            f"{ROLE_BANNER}\n{ROLE_SUBTITLE}\n\n"
            "D60 lag-resolved SWPI summary. Mean Fisher-z lag curves by band "
            "(absolute_log10) with participant-clustered 95% CIs. All available "
            "conditions are pooled only as a descriptive display; this is not a "
            "prespecified paired contrast and is not pooled with D240 ZLPI or the "
            "primary meta-analysis. Distant flanks are the SWPI window |τ|∈[10, 20] s. "
            "No μ/FWHM timing, ±2 s equivalence, or temporal-null claims."
        ),
    )
    return {"pdf": pdf, "svg": svg, "png": png, "source": source, "caption": caption}


def _band_summary_rows(swpi_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for band in BAND_ORDER:
        obs = [
            _as_float(row["swpi"])
            for row in swpi_rows
            if _as_str(row.get("band")) == band
        ]
        bio = _participant_means(swpi_rows, band=band)
        for estimand, values in (
            ("observation_level_swpi", obs),
            ("participant_mean_swpi", bio),
        ):
            stats = _mean_ci(values)
            out.append(
                {
                    "dataset_id": DATASET_ID,
                    "band": band,
                    "duration_s": DURATION_S,
                    "endpoint_alias": ENDPOINT_ALIAS,
                    "power_representation": PRIMARY_REPRESENTATION,
                    "estimand": estimand,
                    "n": stats["n"],
                    "mean_swpi": stats["mean"],
                    "se": stats["se"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "scientific_role": "sensitivity_descriptive_only",
                    "enters_meta": False,
                    "prespecified_paired_contrast": False,
                    "pooled": False,
                }
            )
    return out


def _render_band_figure(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_publication_style()
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    bio_rows = [row for row in rows if row.get("estimand") == "participant_mean_swpi"]
    y_pos = np.arange(len(BAND_ORDER))
    for idx, band in enumerate(BAND_ORDER):
        match = next((row for row in bio_rows if row.get("band") == band), None)
        if match is None:
            continue
        mean = _as_float(match["mean_swpi"])
        lo = _as_float(match["ci_low"])
        hi = _as_float(match["ci_high"])
        color = BAND_COLORS.get(band, PALETTE["blue"])
        ax.plot([lo, hi], [idx, idx], color=color, linewidth=3.0, solid_capstyle="round")
        ax.plot(mean, idx, marker="o", color=color, markersize=9)
        ax.text(
            1.01,
            idx,
            f"N={_as_int(match['n'])}",
            transform=ax.get_yaxis_transform(),
            fontsize=FS_LEGEND - 2,
            color=PALETTE["dark_gray"],
            va="center",
            ha="left",
        )
    ax.axvline(0.0, color=PALETTE["light_gray"], linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_band_display(band) for band in BAND_ORDER])
    ax.set_xlabel("Participant-mean SWPI (95% CI)")
    ax.set_title(
        "D60 band summary  ·  descriptive SWPI  ·  not a paired Δ forest",
        fontsize=FS_PANEL_TITLE - 4,
        pad=10,
    )
    ax.invert_yaxis()
    ax.text(
        0.98,
        0.04,
        "Each point is the mean of per-participant means.\n"
        "Not pooled. No prespecified paired contrast.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FS_LEGEND - 2,
        color=PALETTE["dark_gray"],
    )
    _apply_role_banner(fig)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.16, right=0.90)
    pdf, svg, png = save_figure_trio(fig, output_dir, STEM_BAND)
    source = write_source_csv(
        output_dir / "source_data" / f"{STEM_BAND}.csv",
        rows,
        fieldnames=[
            "dataset_id",
            "band",
            "duration_s",
            "endpoint_alias",
            "power_representation",
            "estimand",
            "n",
            "mean_swpi",
            "se",
            "ci_low",
            "ci_high",
            "scientific_role",
            "enters_meta",
            "prespecified_paired_contrast",
            "pooled",
        ],
    )
    caption = _write_caption(
        output_dir / f"{STEM_BAND}_caption.txt",
        (
            f"{ROLE_BANNER}\n{ROLE_SUBTITLE}\n\n"
            "D60 theta/alpha/beta/low-gamma SWPI descriptive summary. Displayed "
            "estimand is the mean of per-participant means (absolute_log10) with "
            "a 95% CI. Observation-level means are in the source CSV. This is not "
            "a paired-contrast forest and is not pooled."
        ),
    )
    return {"pdf": pdf, "svg": svg, "png": png, "source": source, "caption": caption}


def _alpha_summary_rows(swpi_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    alpha = [row for row in swpi_rows if _as_str(row.get("band")) == "alpha"]
    bio_buckets: dict[str, list[float]] = defaultdict(list)
    for row in alpha:
        bio_buckets[_as_str(row.get("participant_id"))].append(_as_float(row["swpi"]))
    bio_rows = [
        {
            "dataset_id": DATASET_ID,
            "participant_id": pid,
            "band": "alpha",
            "duration_s": DURATION_S,
            "endpoint_alias": ENDPOINT_ALIAS,
            "n_observations": len(values),
            "participant_mean_swpi": float(np.mean(values)),
            "estimand": "participant_mean_swpi",
            "scientific_role": "sensitivity_descriptive_only",
            "enters_meta": False,
            "prespecified_paired_contrast": False,
            "pooled": False,
        }
        for pid, values in sorted(bio_buckets.items())
        if values
    ]
    obs_stats = _mean_ci([_as_float(row["swpi"]) for row in alpha])
    bio_stats = _mean_ci([_as_float(row["participant_mean_swpi"]) for row in bio_rows])
    summary = [
        {
            "dataset_id": DATASET_ID,
            "band": "alpha",
            "duration_s": DURATION_S,
            "endpoint_alias": ENDPOINT_ALIAS,
            "power_representation": PRIMARY_REPRESENTATION,
            "estimand": "observation_level_swpi",
            "n": obs_stats["n"],
            "mean_swpi": obs_stats["mean"],
            "se": obs_stats["se"],
            "ci_low": obs_stats["ci_low"],
            "ci_high": obs_stats["ci_high"],
            "scientific_role": "sensitivity_descriptive_only",
            "enters_meta": False,
            "prespecified_paired_contrast": False,
            "pooled": False,
        },
        {
            "dataset_id": DATASET_ID,
            "band": "alpha",
            "duration_s": DURATION_S,
            "endpoint_alias": ENDPOINT_ALIAS,
            "power_representation": PRIMARY_REPRESENTATION,
            "estimand": "participant_mean_swpi",
            "n": bio_stats["n"],
            "mean_swpi": bio_stats["mean"],
            "se": bio_stats["se"],
            "ci_low": bio_stats["ci_low"],
            "ci_high": bio_stats["ci_high"],
            "scientific_role": "sensitivity_descriptive_only",
            "enters_meta": False,
            "prespecified_paired_contrast": False,
            "pooled": False,
        },
    ]
    return summary, bio_rows


def _render_alpha_figure(
    swpi_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    bio_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), sharey=False)
    alpha_obs = [
        _as_float(row["swpi"])
        for row in swpi_rows
        if _as_str(row.get("band")) == "alpha"
    ]
    obs_stats = next(row for row in summary_rows if row["estimand"] == "observation_level_swpi")
    bio_stats = next(row for row in summary_rows if row["estimand"] == "participant_mean_swpi")

    axes[0].hist(
        alpha_obs,
        bins=24,
        color=BAND_COLORS["alpha"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].axvline(0.0, color=PALETTE["light_gray"], linewidth=1.0)
    axes[0].axvline(_as_float(obs_stats["mean_swpi"]), color=PALETTE["dark_gray"], linewidth=1.8)
    axes[0].set_xlabel("Observation-level alpha SWPI")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"Observation distribution  ·  N={_as_int(obs_stats['n'])}",
        fontsize=FS_PANEL_TITLE - 6,
        pad=8,
    )

    bio_values = [_as_float(row["participant_mean_swpi"]) for row in bio_rows]
    jitter = np.linspace(-0.18, 0.18, num=max(len(bio_values), 1))
    axes[1].scatter(
        np.zeros(len(bio_values)) + jitter[: len(bio_values)],
        bio_values,
        s=28,
        color=BAND_COLORS["alpha"],
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    mean = _as_float(bio_stats["mean_swpi"])
    lo = _as_float(bio_stats["ci_low"])
    hi = _as_float(bio_stats["ci_high"])
    axes[1].plot([0.55, 0.55], [lo, hi], color=PALETTE["dark_gray"], linewidth=3.0, zorder=3)
    axes[1].plot(0.55, mean, marker="s", color=PALETTE["dark_gray"], markersize=8, zorder=4)
    axes[1].axhline(0.0, color=PALETTE["light_gray"], linewidth=1.0)
    axes[1].set_xlim(-0.5, 1.0)
    axes[1].set_xticks([0.0, 0.55])
    axes[1].set_xticklabels(["Participants", "Mean ± 95% CI"])
    axes[1].set_ylabel("Participant-mean alpha SWPI")
    axes[1].set_title(
        f"Biological-participant means  ·  N={_as_int(bio_stats['n'])}",
        fontsize=FS_PANEL_TITLE - 6,
        pad=8,
    )
    fig.text(
        0.5,
        0.02,
        "Descriptive only. Repeated observations are nested in participants. "
        "Not a confirmatory paired effect and not pooled.",
        ha="center",
        fontsize=FS_LEGEND - 1,
        color=PALETTE["dark_gray"],
    )
    _apply_role_banner(fig)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.98, wspace=0.28)
    pdf, svg, png = save_figure_trio(fig, output_dir, STEM_ALPHA)
    source = write_source_csv(
        output_dir / "source_data" / f"{STEM_ALPHA}.csv",
        list(summary_rows) + list(bio_rows),
        fieldnames=[
            "dataset_id",
            "participant_id",
            "band",
            "duration_s",
            "endpoint_alias",
            "power_representation",
            "estimand",
            "n",
            "n_observations",
            "mean_swpi",
            "participant_mean_swpi",
            "se",
            "ci_low",
            "ci_high",
            "scientific_role",
            "enters_meta",
            "prespecified_paired_contrast",
            "pooled",
        ],
    )
    caption = _write_caption(
        output_dir / f"{STEM_ALPHA}_caption.txt",
        (
            f"{ROLE_BANNER}\n{ROLE_SUBTITLE}\n\n"
            "Alpha D60 descriptive SWPI summary. Left: observation-level distribution. "
            "Right: per-biological-participant means and the mean of those means with "
            "a 95% CI. This is not a confirmatory paired contrast and is not pooled."
        ),
    )
    return {"pdf": pdf, "svg": svg, "png": png, "source": source, "caption": caption}


def _qc_rows(
    *,
    audit_rows: Sequence[Mapping[str, str]],
    eligibility_rows: Sequence[Mapping[str, str]],
    c1b_rows: Sequence[Mapping[str, str]],
    swpi_rows: Sequence[Mapping[str, object]],
    pairing_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    audit = [row for row in audit_rows if _as_str(row.get("dataset_id")).casefold() in {"", DATASET_ID}]
    d60 = [
        row
        for row in eligibility_rows
        if _as_str(row.get("dataset_id")).casefold() in {"", DATASET_ID}
        and _as_int(row.get("duration_s"), DURATION_S) == DURATION_S
    ]
    usable = sum(1 for row in audit if _as_bool(row.get("usable"), False))
    eligible = sum(1 for row in d60 if _as_str(row.get("status")).casefold() == "eligible")
    ineligible = len(d60) - eligible
    c1b_usable = sum(1 for row in c1b_rows if _as_bool(row.get("usable"), False))
    c1b_missing = sum(
        1
        for row in c1b_rows
        if _as_str(row.get("reason_code")).casefold() == "missing_event_series"
    )
    n_common = sorted({_as_int(row.get("n_common_support")) for row in swpi_rows})
    pairing_status = next(
        (
            _as_str(row.get("pairing_status") or row.get("reason_code"))
            for row in pairing_rows
        ),
        "no_prespecified_contrast",
    )
    durations = [
        _as_float(row.get("eeg_duration_s") or row.get("raw_overlap_s"))
        for row in audit
        if math.isfinite(_as_float(row.get("eeg_duration_s") or row.get("raw_overlap_s")))
    ]
    return [
        {
            "dataset_id": DATASET_ID,
            "item": "c0_audit_observations",
            "n": len(audit),
            "note": "C0 data_audit rows",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "c0_usable",
            "n": usable,
            "note": "usable=True in C0 data_audit",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "d60_eligible",
            "n": eligible,
            "note": "C0 eligibility_by_duration status=eligible at D60/SWPI",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "d60_ineligible",
            "n": ineligible,
            "note": "C0 D60 ineligible (typically insufficient_raw_duration)",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "c1b_usable_cardiac",
            "n": c1b_usable,
            "note": "C1b usable cardiac event series",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "c1b_missing_event_series",
            "n": c1b_missing,
            "note": "C1b excluded: missing_event_series",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "c3_d60_swpi_observations",
            "n": len({_as_str(row.get("observation_id")) for row in swpi_rows}),
            "note": "Unique observations with eligible D60 absolute_log10 SWPI",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "c3_d60_swpi_participants",
            "n": len({_as_str(row.get("participant_id")) for row in swpi_rows}),
            "note": "Unique biological participants in D60 SWPI",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "n_common_support_unique_values",
            "n": len(n_common),
            "note": ",".join(str(v) for v in n_common) or "none",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "pairing_status",
            "n": 0,
            "note": pairing_status or "no_prespecified_contrast",
        },
        {
            "dataset_id": DATASET_ID,
            "item": "eeg_duration_s_n",
            "n": len(durations),
            "note": (
                f"min={min(durations):.3f};median={float(np.median(durations)):.3f};"
                f"max={max(durations):.3f}"
                if durations
                else "no_durations"
            ),
        },
        {
            "dataset_id": DATASET_ID,
            "item": "scientific_role",
            "n": 0,
            "note": ROLE_BANNER,
        },
    ]


def _render_qc_figure(
    qc_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, str]],
    swpi_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.0))
    lookup = {str(row["item"]): row for row in qc_rows}
    funnel_keys = [
        ("c0_audit_observations", "C0 audit"),
        ("c0_usable", "C0 usable"),
        ("d60_eligible", "D60 eligible"),
        ("c1b_usable_cardiac", "C1b cardiac"),
        ("c3_d60_swpi_observations", "C3 SWPI"),
    ]
    values = [_as_int(lookup[key]["n"]) for key, _label in funnel_keys if key in lookup]
    labels = [label for key, label in funnel_keys if key in lookup]
    axes[0].barh(np.arange(len(values)), values, color=PALETTE["blue"], alpha=0.85)
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(values) * 1.28 if values else 1)
    axes[0].set_xlabel("N observations")
    axes[0].set_title("Duration / analysis funnel", fontsize=FS_PANEL_TITLE - 6, pad=8)
    for idx, value in enumerate(values):
        axes[0].text(
            value + max(values) * 0.02,
            idx,
            str(value),
            va="center",
            fontsize=FS_LEGEND - 1,
            clip_on=False,
        )

    durations = [
        _as_float(row.get("eeg_duration_s"))
        for row in audit_rows
        if _as_str(row.get("dataset_id")).casefold() in {"", DATASET_ID}
        and math.isfinite(_as_float(row.get("eeg_duration_s")))
    ]
    if durations:
        axes[1].hist(durations, bins=30, color=PALETTE["green"], alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1].axvline(60.0, color=PALETTE["vermillion"], linewidth=1.8, label="D60 threshold")
    axes[1].set_xlabel("EEG duration (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("C0 duration eligibility", fontsize=FS_PANEL_TITLE - 6, pad=8)
    axes[1].legend(frameon=False, loc="upper left")

    support_by_obs = {
        _as_str(row.get("observation_id")): _as_int(row.get("n_common_support"))
        for row in swpi_rows
        if _as_str(row.get("observation_id"))
    }
    support = list(support_by_obs.values())
    if support:
        axes[2].hist(
            support,
            bins=np.arange(min(support) - 0.5, max(support) + 1.5, 1.0),
            color=PALETTE["orange"],
            edgecolor="white",
        )
    axes[2].axvline(20.0, color=PALETTE["dark_gray"], linewidth=1.4, linestyle="--", label="SWPI n=20")
    axes[2].set_xlabel("n_common_support")
    axes[2].set_ylabel("Observations")
    axes[2].set_title("D60 common-support QC", fontsize=FS_PANEL_TITLE - 6, pad=8)
    pairing = str(lookup.get("pairing_status", {}).get("note", "no_prespecified_contrast"))
    axes[2].text(
        0.5,
        -0.22,
        f"Pairing: {pairing}\nNot pooled. No prespecified paired contrast.",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=FS_LEGEND - 2,
        color=PALETTE["dark_gray"],
    )
    _apply_role_banner(fig)
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.10, right=0.98, wspace=0.32)
    pdf, svg, png = save_figure_trio(fig, output_dir, STEM_QC)
    source = write_source_csv(
        output_dir / "source_data" / f"{STEM_QC}.csv",
        qc_rows,
        fieldnames=["dataset_id", "item", "n", "note"],
    )
    caption = _write_caption(
        output_dir / f"{STEM_QC}_caption.txt",
        (
            f"{ROLE_BANNER}\n{ROLE_SUBTITLE}\n\n"
            "Duration eligibility and common-support QC. Funnel shows C0 audit → "
            "usable → D60-eligible → C1b usable cardiac → C3 SWPI observations. "
            "The C0-to-C3 drop is dominated by C1b missing_event_series (not a "
            "duration-threshold failure). Center: raw EEG durations vs the 60 s "
            "threshold. Right: C3 n_common_support for eligible D60 absolute_log10 "
            "SWPI (contract = 20). Pairing QC remains no_prespecified_contrast. "
            "Not pooled."
        ),
    )
    return {"pdf": pdf, "svg": svg, "png": png, "source": source, "caption": caption}


def _completeness_rows() -> list[dict[str, object]]:
    return [
        {
            "display": "D60 alpha SWPI descriptive distribution",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_ALPHA,
            "reason": "Required descriptive display; was missing from manuscript C7.",
        },
        {
            "display": "D60 theta/alpha/beta/low-gamma summaries",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_BAND,
            "reason": "Required descriptive display; was missing from manuscript C7.",
        },
        {
            "display": "subject/bio-level descriptive summary",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_ALPHA,
            "reason": "Participant-mean panel is part of the alpha descriptive figure.",
        },
        {
            "display": "duration eligibility / common-support QC",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_QC,
            "reason": "Required QC display; C0/C1b/C3 support it without C0–C4 rerun.",
        },
        {
            "display": "lag curves at D60",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_LAG,
            "reason": "Required descriptive display from frozen C2 D60 curves.",
        },
        {
            "display": "endpoint support / common-support QC",
            "class": "A",
            "status": "rebuilt",
            "stem": STEM_QC,
            "reason": "Shown as n_common_support in the QC figure.",
        },
        {
            "display": "optional topography",
            "class": "C",
            "status": "not_applicable",
            "stem": "",
            "reason": "supports_gamma=false; no D60 gamma topography or paired Δ map.",
        },
        {
            "display": "paired contrast forest",
            "class": "C",
            "status": "not_applicable",
            "stem": "",
            "reason": "PROTOCOL_SPECS contrasts=(); pairing_status=no_prespecified_contrast.",
        },
        {
            "display": "temporal-null panel",
            "class": "C",
            "status": "not_applicable",
            "stem": "",
            "reason": "C4 is D240-only; ds003816 has no D240 units. C4 not extended.",
        },
        {
            "display": "peak timing / ±2 s equivalence panel",
            "class": "C",
            "status": "not_applicable",
            "stem": "",
            "reason": "Option C peak timing requires D180/D240; D60 is excluded.",
        },
        {
            "display": "manuscript Figures 1–3 on this sensitivity tree",
            "class": "B",
            "status": "stale_empty_sources_left_in_place",
            "stem": "figure1_lag_resolved_zero_lag;figure2_state_attenuation_replication;figure3_temporal_artifact_specificity",
            "reason": "Wrong estimand for this cohort; not regenerated; not the sensitivity package.",
        },
    ]


def render_ds003816_descriptive_qc(
    dataset_root: str | Path,
    c7_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(dataset_root).expanduser().resolve()
    report_dir = Path(c7_dir).expanduser().resolve() if c7_dir is not None else root / "C7"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir = figure_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        rel: _sha256_prefix(root / rel) for rel in _FROZEN_SOURCES
    }
    c2_curves = root / "C2" / "confirmatory_cross_correlation_curves_D60.csv"
    c3 = root / "C3" / "confirmatory_endpoint_metrics_D60.csv"
    c5 = root / "C5" / "subject_level_metrics.csv"
    if not c2_curves.is_file() or not c5.is_file():
        raise FileNotFoundError(
            "ds003816 descriptive QC requires frozen C2 D60 curves and C5 "
            f"subject_level_metrics under {root}"
        )

    swpi_rows = _load_subject_swpi(c5)
    if not swpi_rows and c3.is_file():
        # Fallback if C5 is empty but C3 exists.
        for row in _read_csv(c3):
            if not _is_primary_abs_swpi_d60(row) or not _eligible_row(row):
                continue
            value = _as_float(row.get("endpoint_index"))
            if not math.isfinite(value):
                continue
            swpi_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "observation_id": observation_id_from_row(row),
                    "participant_id": biological_id_from_row(row),
                    "condition": _as_str(row.get("condition") or row.get("task")),
                    "band": _as_str(row.get("band")).casefold(),
                    "duration_s": DURATION_S,
                    "endpoint_alias": ENDPOINT_ALIAS,
                    "power_representation": PRIMARY_REPRESENTATION,
                    "n_common_support": _as_int(row.get("n_common_support")),
                    "swpi": value,
                    "estimand": "observation_level_swpi",
                    "scientific_role": "sensitivity_descriptive_only",
                    "enters_meta": False,
                    "prespecified_paired_contrast": False,
                }
            )
    if not swpi_rows:
        raise ValueError("No eligible D60 absolute_log10 SWPI rows found in C5/C3.")

    lag_rows = _aggregate_lag_curves(c2_curves)
    band_rows = _band_summary_rows(swpi_rows)
    alpha_summary, alpha_bio = _alpha_summary_rows(swpi_rows)
    qc_rows = _qc_rows(
        audit_rows=_read_csv(root / "C0" / "data_audit.csv"),
        eligibility_rows=_read_csv(root / "C0" / "eligibility_by_duration.csv"),
        c1b_rows=_read_csv(root / "C1b" / "cardiac_peak_qc.csv"),
        swpi_rows=swpi_rows,
        pairing_rows=_read_csv(root / "C5" / "pairing_qc.csv"),
    )

    paths: dict[str, Path] = {}
    for prefix, rendered in (
        ("lag", _render_lag_figure(lag_rows, figure_dir)),
        ("band", _render_band_figure(band_rows, figure_dir)),
        ("alpha", _render_alpha_figure(swpi_rows, alpha_summary, alpha_bio, figure_dir)),
        ("qc", _render_qc_figure(qc_rows, _read_csv(root / "C0" / "data_audit.csv"), swpi_rows, figure_dir)),
    ):
        for key, path in rendered.items():
            paths[f"{prefix}_{key}"] = path

    completeness = _completeness_rows()
    completeness_path = source_dir / "ds003816_descriptive_qc_completeness.csv"
    completeness_path.parent.mkdir(parents=True, exist_ok=True)
    with completeness_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["display", "class", "status", "stem", "reason"],
        )
        writer.writeheader()
        writer.writerows(completeness)
    inventory = {
        "dataset_id": DATASET_ID,
        "scientific_role": "sensitivity / duration_sensitivity / D60 only / SWPI",
        "enters_meta": False,
        "prespecified_paired_contrast": False,
        "rebuild_boundary": "A_C7_only",
        "required_stems": list(REQUIRED_STEMS),
        "frozen_source_sha256_16": source_hashes,
        "primary_figures_1_3_changed": False,
        "generated_paired_forest": False,
        "generated_mu_fwhm_timing": False,
        "generated_pm2s_equivalence": False,
        "generated_temporal_null_claims": False,
        "completeness": completeness,
    }
    inventory_path = figure_dir / "ds003816_descriptive_qc_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    paths["inventory"] = inventory_path
    paths["completeness"] = source_dir / "ds003816_descriptive_qc_completeness.csv"
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("derivatives/confirmatory_temporal_coupling/sensitivity/ds003816"),
        help="Dataset confirmatory output root containing C0/C2/C3/C5.",
    )
    parser.add_argument("--c7-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = render_ds003816_descriptive_qc(args.root, args.c7_dir)
    for key, path in sorted(paths.items()):
        print(f"{key}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
