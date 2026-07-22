"""Display-only forest helpers for Figure 1 Panel E / Figure 2 Panel C.

Presentation layer only. Does not alter ``enters_meta``, PRIMARY_META membership,
or random-effects pooling.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from .duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S
from .inference import PRIMARY_POWER_REPRESENTATION

ROW_TYPE_PRIMARY = "primary"
ROW_TYPE_SENSITIVITY_DISPLAY = "sensitivity_display"
ROW_TYPE_POOLED = "pooled"

HIIT_DATASET_ID = "hiit"
# All Rest–Tetris ΔZLPI contrasts (PH/PS × PRE/POST) for combined forest display.
HIIT_ALL_CONTRASTS = frozenset(
    {
        "ph_pre_rest__tetris",
        "ph_post_rest__tetris",
        "ps_pre_rest__tetris",
        "ps_post_rest__tetris",
    }
)
# All low-demand absolute-ZLPI conditions for combined Panel D display.
HIIT_ALL_LOW_DEMAND = frozenset(
    {
        "ph_pre_rest",
        "ph_post_rest",
        "ps_pre_rest",
        "ps_post_rest",
    }
)
# Retained for tests / callers that still reference session tokens.
HIIT_PH_CONTRASTS = frozenset({"ph_pre_rest__tetris", "ph_post_rest__tetris"})
HIIT_PS_CONTRASTS = frozenset({"ps_pre_rest__tetris", "ps_post_rest__tetris"})
HIIT_PH_LOW_DEMAND = frozenset({"ph_pre_rest", "ph_post_rest"})
HIIT_PS_LOW_DEMAND = frozenset({"ps_pre_rest", "ps_post_rest"})

SENSITIVITY_SECTION_LABEL = "Sensitivity (not pooled)"
HIIT_FOREST_DISPLAY_LABEL = "HIIT [S]"


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return default


def student_t_effect_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Mean ± Student-t 95% CI from participant-level values (same as dataset effects)."""
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "n_pairs": 0,
            "effect_mean": float("nan"),
            "effect_sd": float("nan"),
            "effect_se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean = float(np.mean(arr))
    if n >= 2:
        sd = float(np.std(arr, ddof=1))
        se = sd / math.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        sd = float("nan")
        se = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "n_pairs": n,
        "effect_mean": mean,
        "effect_sd": sd,
        "effect_se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _primary_slice_ok(row: Mapping[str, object], *, band: str = "alpha") -> bool:
    if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
        return False
    if _as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
        EXPECTED_PRIMARY_DURATION_S
    ):
        return False
    if _as_str(row.get("band")).casefold() != band.casefold():
        return False
    if (
        _as_str(row.get("power_representation"), PRIMARY_POWER_REPRESENTATION).casefold()
        != PRIMARY_POWER_REPRESENTATION
    ):
        return False
    if "contrast_eligible" in row and not _as_bool(row.get("contrast_eligible"), True):
        return False
    return True


def _hiit_session_unit_key(row: Mapping[str, object]) -> str:
    """Panel-B-aligned HIIT unit: session subject_id (PH/PS separate).

    Prefers ``subject_id`` (``01_ph``). Falls back to ``participant_id`` +
    ``session_id`` or protocol token from ``condition`` / ``contrast_id``.
    """
    subject = _as_str(row.get("subject_id")).casefold()
    if subject:
        return subject
    participant = _as_str(row.get("participant_id")).casefold()
    session = _as_str(row.get("session_id"), "single").casefold() or "single"
    if session not in {"", "single"}:
        return f"{participant}_{session}" if participant else session
    condition = _as_str(row.get("condition") or row.get("contrast_id")).casefold()
    if condition.startswith("ph_"):
        return f"{participant}_ph" if participant else "ph"
    if condition.startswith("ps_"):
        return f"{participant}_ps" if participant else "ps"
    return participant


def hiit_session_sensitivity_forest_rows(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    band: str = "alpha",
) -> list[dict[str, object]]:
    """One combined HIIT sensitivity forest row (display-only).

    Panel-B-aligned: each PH/PS session subject is one unit. Within a session,
    average available Rest–Tetris ΔZLPI contrasts for that session, then
    Student-t mean/CI across session subjects (n≈40, not biological n≈20).
    """
    # session_unit -> {contrast_id: delta}
    buckets: dict[str, dict[str, float]] = {}
    for row in paired_rows:
        if _as_str(row.get("dataset_id")).casefold() != HIIT_DATASET_ID:
            continue
        if not _primary_slice_ok(row, band=band):
            continue
        contrast = _as_str(row.get("contrast_id")).casefold()
        if contrast not in HIIT_ALL_CONTRASTS:
            continue
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        unit = _hiit_session_unit_key(row)
        if not unit:
            continue
        buckets.setdefault(unit, {})[contrast] = delta

    session_avgs: list[float] = []
    for _unit, by_contrast in sorted(buckets.items()):
        vals = [
            by_contrast[c]
            for c in sorted(HIIT_ALL_CONTRASTS)
            if c in by_contrast and math.isfinite(by_contrast[c])
        ]
        if not vals:
            continue
        session_avgs.append(float(np.mean(np.asarray(vals, dtype=float))))

    summary = student_t_effect_summary(session_avgs)
    n_sessions = int(summary["n_pairs"])
    if n_sessions < 1:
        return []
    return [
        {
            "dataset_id": HIIT_DATASET_ID,
            "contrast_id": "hiit_combined_ph_ps_pre_post_mean",
            "band": band.casefold(),
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "effect_mean": summary["effect_mean"],
            "effect_sd": summary["effect_sd"],
            "effect_se": summary["effect_se"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "n_pairs": n_sessions,
            "n_participants": n_sessions,
            "n_participant_sessions": n_sessions,
            "enters_meta": False,
            "cardiac_modality": "",
            "prediction_low": "",
            "prediction_high": "",
            "n_datasets": "",
            "row_type": ROW_TYPE_SENSITIVITY_DISPLAY,
            "display_label": HIIT_FOREST_DISPLAY_LABEL,
            "session_id": "ph|ps",
            "aggregation": "session_subject_mean_of_available_pre_post_delta",
        }
    ]


def hiit_session_mean_zlpi_cells(
    subject_rows: Sequence[Mapping[str, object]],
    *,
    band_order: Sequence[str],
    primary_representation: str,
) -> list[dict[str, object]]:
    """Figure 1 Panel D: one combined HIIT [S] row.

    Panel-B-aligned: each PH/PS session subject is one unit. Within a session,
    average available low-demand ZLPI for that session, then mean across
    session subjects (n≈40).
    """
    # (session_unit, band) -> {condition: z}
    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for row in subject_rows:
        if _as_str(row.get("dataset_id")).casefold() != HIIT_DATASET_ID:
            continue
        if _as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
            EXPECTED_PRIMARY_DURATION_S
        ):
            continue
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if (
            _as_str(row.get("power_representation"), primary_representation).casefold()
            != primary_representation.casefold()
        ):
            continue
        if "endpoint_eligible" in row and not _as_bool(
            row.get("endpoint_eligible"), True
        ):
            continue
        condition = _as_str(row.get("condition")).casefold()
        if condition not in HIIT_ALL_LOW_DEMAND:
            continue
        band = _as_str(row.get("band")).casefold()
        if band not in {b.casefold() for b in band_order}:
            continue
        value = _as_float(row.get("endpoint_index"))
        if not math.isfinite(value):
            continue
        unit = _hiit_session_unit_key(row)
        if not unit:
            continue
        # Restrict conditions to the session's protocol.
        if unit.endswith("_ph") and not condition.startswith("ph_"):
            continue
        if unit.endswith("_ps") and not condition.startswith("ps_"):
            continue
        buckets.setdefault((unit, band), {})[condition] = value

    out: list[dict[str, object]] = []
    for band in band_order:
        band_key = band.casefold()
        session_avgs: list[float] = []
        for (unit, b), by_cond in sorted(buckets.items()):
            if b != band_key:
                continue
            allowed = (
                HIIT_PH_LOW_DEMAND
                if unit.endswith("_ph")
                else HIIT_PS_LOW_DEMAND
                if unit.endswith("_ps")
                else HIIT_ALL_LOW_DEMAND
            )
            vals = [
                by_cond[c]
                for c in sorted(allowed)
                if c in by_cond and math.isfinite(by_cond[c])
            ]
            if not vals:
                continue
            session_avgs.append(float(np.mean(np.asarray(vals, dtype=float))))
        if not session_avgs:
            continue
        out.append(
            {
                "dataset_id": HIIT_DATASET_ID,
                "source_dataset_id": HIIT_DATASET_ID,
                "session_id": "ph|ps",
                "display_label": "HIIT",
                "dataset_role": "sensitivity",
                "band": band_key,
                "mean_zlpi": float(np.mean(np.asarray(session_avgs, dtype=float))),
                "n_participants": len(session_avgs),
                "n_participant_sessions": len(session_avgs),
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": primary_representation,
                "aggregation": "session_subject_mean_of_available_low_demand_zlpi",
            }
        )
    return out


def hiit_session_surrogate_significance_marks(
    null_summary_rows: Sequence[Mapping[str, object]],
    *,
    band_order: Sequence[str],
    primary_representation: str,
    null_type: str,
    alpha: float,
    low_demand_labels: set[str],
) -> dict[tuple[str, str], bool]:
    """Combined HIIT Panel D surrogate marks (one mark per band).

    Collects circular-shift ``median_empirical_p`` from all HIIT low-demand
    conditions (PH/PS × PRE/POST), takes their median per band, and marks if
    median < alpha. Does not copy separate PH/PS marks.
    """
    by_band: dict[str, list[float]] = {}
    for row in null_summary_rows:
        if _as_str(row.get("dataset_id")).casefold() != HIIT_DATASET_ID:
            continue
        if _as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
            EXPECTED_PRIMARY_DURATION_S
        ):
            continue
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if (
            _as_str(row.get("power_representation"), primary_representation).casefold()
            != primary_representation.casefold()
        ):
            continue
        if _as_str(row.get("null_type")).casefold() != null_type.casefold():
            continue
        condition = _as_str(row.get("condition")).casefold()
        if condition not in HIIT_ALL_LOW_DEMAND:
            continue
        if condition not in low_demand_labels and condition not in HIIT_ALL_LOW_DEMAND:
            continue
        band = _as_str(row.get("band")).casefold()
        if band not in {b.casefold() for b in band_order}:
            continue
        p = _as_float(row.get("median_empirical_p"))
        if not math.isfinite(p):
            continue
        by_band.setdefault(band, []).append(p)
    return {
        (HIIT_DATASET_ID, band): float(np.median(np.asarray(ps, dtype=float)))
        < float(alpha)
        for band, ps in by_band.items()
    }


def primary_meta_alpha_forest_rows(
    dataset_effects: Sequence[Mapping[str, object]],
    meta_rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
    *,
    cardiac_modality_fn,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    """PRIMARY_META alpha study rows + pooled meta (unchanged confirmatory gate)."""
    studies: list[dict[str, object]] = []
    for row in dataset_effects:
        if not _as_bool(row.get("enters_meta")):
            continue
        if not _primary_slice_ok(row, band="alpha"):
            continue
        # Figure 2 additionally requires primary representation; _primary_slice_ok
        # already enforces absolute_log10.
        dataset_id = _as_str(row.get("dataset_id"))
        studies.append(
            {
                "dataset_id": dataset_id,
                "contrast_id": _as_str(row.get("contrast_id")),
                "band": "alpha",
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "effect_mean": _as_float(row.get("effect_mean")),
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n_pairs": _as_int(row.get("n_pairs")),
                "enters_meta": True,
                "cardiac_modality": cardiac_modality_fn(dataset_id, protocol_rows),
                "prediction_low": "",
                "prediction_high": "",
                "n_datasets": "",
                "row_type": ROW_TYPE_PRIMARY,
                "display_label": "",
            }
        )
    studies.sort(key=lambda r: _as_str(r["dataset_id"]).casefold())

    pooled: dict[str, object] | None = None
    for row in meta_rows:
        if _as_str(row.get("band")).casefold() != "alpha":
            continue
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != (
            EXPECTED_PRIMARY_DURATION_S
        ):
            continue
        if (
            _as_str(
                row.get("power_representation"), PRIMARY_POWER_REPRESENTATION
            ).casefold()
            != PRIMARY_POWER_REPRESENTATION
        ):
            continue
        # Figure 2 also checks is_primary_analysis; accept missing/true.
        flag = str(row.get("is_primary_analysis", "true")).strip().casefold()
        if flag not in {"true", "1", "yes", ""}:
            continue
        pooled = {
            "band": "alpha",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "pooled_effect": _as_float(row.get("pooled_effect")),
            "ci_low": _as_float(row.get("ci_low")),
            "ci_high": _as_float(row.get("ci_high")),
            "prediction_low": _as_float(row.get("prediction_low")),
            "prediction_high": _as_float(row.get("prediction_high")),
            "n_datasets": _as_int(row.get("n_datasets")),
            "i2": _as_float(row.get("i2")),
            "tau2": _as_float(row.get("tau2")),
        }
        break
    return studies, pooled


def build_alpha_forest_export(
    *,
    primary_studies: Sequence[Mapping[str, object]],
    sensitivity_studies: Sequence[Mapping[str, object]],
    pooled: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Flatten forest rows for source CSV (primary, sensitivity_display, pooled)."""
    export: list[dict[str, object]] = []
    for row in primary_studies:
        payload = dict(row)
        payload.setdefault("row_type", ROW_TYPE_PRIMARY)
        payload.setdefault("enters_meta", True)
        export.append(payload)
    for row in sensitivity_studies:
        payload = dict(row)
        payload["row_type"] = ROW_TYPE_SENSITIVITY_DISPLAY
        payload["enters_meta"] = False
        export.append(payload)
    if pooled is not None and math.isfinite(
        float(pooled.get("pooled_effect", float("nan")))
    ):
        export.append(
            {
                "dataset_id": "POOLED",
                "contrast_id": "",
                "band": "alpha",
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "effect_mean": pooled["pooled_effect"],
                "ci_low": pooled["ci_low"],
                "ci_high": pooled["ci_high"],
                "n_pairs": "",
                "enters_meta": True,
                "cardiac_modality": "",
                "prediction_low": pooled.get("prediction_low", ""),
                "prediction_high": pooled.get("prediction_high", ""),
                "n_datasets": pooled.get("n_datasets", ""),
                "row_type": ROW_TYPE_POOLED,
                "display_label": "",
            }
        )
    return export


def draw_alpha_meta_forest(
    ax,
    *,
    primary_studies: Sequence[Mapping[str, object]],
    sensitivity_studies: Sequence[Mapping[str, object]],
    pooled: Mapping[str, object] | None,
    dataset_display_fn,
    band_color_fn,
    ref_vline_fn,
    style_axes_fn,
    set_panel_title_fn,
    panel_title: str,
    xlabel: str,
    marker_size: float,
    line_width: float,
    tick_fontsize: float,
    axis_fontsize: float,
    palette: Mapping[str, str],
) -> bool:
    """Draw PRIMARY_META + optional HIIT sensitivity rows + pooled diamond.

    Returns True if anything was drawn. Y-axis is inverted so the first primary
    study is at the top (standard forest orientation).
    """
    has_pooled = pooled is not None and math.isfinite(
        float(pooled.get("pooled_effect", float("nan")))
    )
    if not primary_studies and not sensitivity_studies and not has_pooled:
        return False

    y_labels: list[str] = []
    positions: list[float] = []
    header_positions: list[tuple[float, str]] = []
    y_pos = 0.0

    def _draw_study(
        study: Mapping[str, object],
        *,
        fmt: str,
        color: str,
        markerfacecolor: str | None = None,
    ) -> None:
        nonlocal y_pos
        pe = float(study["effect_mean"])
        lo = float(study["ci_low"])
        hi = float(study["ci_high"])
        xerr = None
        if math.isfinite(pe) and math.isfinite(lo) and math.isfinite(hi):
            xerr = [[pe - lo], [hi - pe]]
        ax.errorbar(
            pe if math.isfinite(pe) else 0.0,
            y_pos,
            xerr=xerr,
            fmt=fmt,
            color=color,
            markersize=marker_size,
            capsize=4,
            elinewidth=line_width,
            markeredgecolor=palette["dark_gray"],
            markeredgewidth=0.6,
            markerfacecolor=markerfacecolor if markerfacecolor is not None else color,
            zorder=3,
        )
        label = _as_str(study.get("display_label"))
        if not label:
            modality = _as_str(study.get("cardiac_modality"))
            n_pairs = study.get("n_pairs")
            label = dataset_display_fn(
                _as_str(study["dataset_id"]),
                n_pairs=int(n_pairs) if n_pairs not in {"", None} else None,
            )
            if modality:
                label = f"{label} · {modality}"
        y_labels.append(label)
        positions.append(float(y_pos))
        y_pos += 1.0

    def _section_break(title: str | None = None) -> None:
        nonlocal y_pos
        if positions or header_positions:
            y_pos += 0.25
            ax.axhline(
                y_pos - 0.35,
                color=palette.get("light_gray", "#BDBDBD"),
                lw=0.8,
                ls="--",
                zorder=1,
            )
            y_pos += 0.35
        if title:
            header_positions.append((float(y_pos), title))
            y_pos += 0.75

    for study in primary_studies:
        _draw_study(study, fmt="o", color=band_color_fn("alpha"))

    if sensitivity_studies:
        # Section header only when separating from PRIMARY_META rows. On
        # sensitivity-only runs the floating label sat at the top of Panel E
        # (clip_on=False) and visually leaked into Panel C above.
        if primary_studies:
            _section_break(SENSITIVITY_SECTION_LABEL)
        for study in sensitivity_studies:
            _draw_study(
                study,
                fmt="o",
                color=palette["dark_gray"],
                markerfacecolor="white",
            )

    if has_pooled:
        assert pooled is not None
        _section_break("Pooled PRIMARY_META")
        pe = float(pooled["pooled_effect"])
        plo = float(pooled["ci_low"])
        phi = float(pooled["ci_high"])
        pred_lo = float(pooled.get("prediction_low", float("nan")))
        pred_hi = float(pooled.get("prediction_high", float("nan")))
        if math.isfinite(pred_lo) and math.isfinite(pred_hi):
            ax.plot(
                [pred_lo, pred_hi],
                [y_pos, y_pos],
                color=palette["light_gray"],
                lw=6,
                solid_capstyle="butt",
                zorder=2,
                label="Prediction interval",
            )
        xerr = None
        if math.isfinite(plo) and math.isfinite(phi):
            xerr = [[pe - plo], [phi - pe]]
        ax.errorbar(
            pe,
            y_pos,
            xerr=xerr,
            fmt="D",
            color=palette["dark_gray"],
            markersize=marker_size,
            capsize=4,
            elinewidth=line_width,
            zorder=3,
            label="Pooled RE",
        )
        y_labels.append(
            f"Pooled RE (k = {int(pooled.get('n_datasets') or len(primary_studies))})"
        )
        positions.append(float(y_pos))

    ref_vline_fn(ax, 0.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(y_labels, fontsize=tick_fontsize - 2)
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    style_axes_fn(ax)
    set_panel_title_fn(ax, panel_title)
    ax.invert_yaxis()
    # Section headers stay inside / beside this axes only (never bleed upward).
    for y_h, title in header_positions:
        ax.text(
            0.02,
            y_h,
            title,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=tick_fontsize - 3,
            color=palette["dark_gray"],
            fontstyle="italic",
            clip_on=True,
            zorder=1,
        )
    return True


FOREST_EXPORT_FIELDS = (
    "dataset_id",
    "contrast_id",
    "band",
    "endpoint_name",
    "duration_s",
    "effect_mean",
    "ci_low",
    "ci_high",
    "n_pairs",
    "n_participants",
    "n_participant_sessions",
    "enters_meta",
    "cardiac_modality",
    "prediction_low",
    "prediction_high",
    "n_datasets",
    "row_type",
    "display_label",
    "session_id",
    "aggregation",
)

__all__ = [
    "FOREST_EXPORT_FIELDS",
    "HIIT_ALL_CONTRASTS",
    "HIIT_ALL_LOW_DEMAND",
    "HIIT_FOREST_DISPLAY_LABEL",
    "HIIT_PH_CONTRASTS",
    "HIIT_PH_LOW_DEMAND",
    "HIIT_PS_CONTRASTS",
    "HIIT_PS_LOW_DEMAND",
    "ROW_TYPE_POOLED",
    "ROW_TYPE_PRIMARY",
    "ROW_TYPE_SENSITIVITY_DISPLAY",
    "SENSITIVITY_SECTION_LABEL",
    "build_alpha_forest_export",
    "draw_alpha_meta_forest",
    "hiit_session_mean_zlpi_cells",
    "hiit_session_sensitivity_forest_rows",
    "hiit_session_surrogate_significance_marks",
    "primary_meta_alpha_forest_rows",
    "student_t_effect_summary",
]
