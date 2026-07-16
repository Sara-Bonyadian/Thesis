"""Read-only post-run QC reporting for confirmatory C0–C7 outputs.

Writes exclusively under ``{dataset_output_root}/QC/``. Never modifies stage
outputs, eligibility, statistics, or figures. Heuristic thresholds prioritize
human review only and do not change analytical decisions.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .duration_contracts import EXPECTED_DURATIONS_S, EXPECTED_PRIMARY_DURATION_S
from .manifest import git_commit_hash, sha256_file

SCHEMA_VERSION = "confirmatory_qc_report_v1"
QC_DIRNAME = "QC"
EXTENSIONS_DIRNAME = "extensions"

PIPELINE_FACT = "pipeline_fact"
DERIVED_METRIC = "derived_metric"
HEURISTIC_REVIEW_SUGGESTION = "heuristic_review_suggestion"
ANALYTICAL_EXCLUSION = "analytical_exclusion"
NOT_ASSESSED = "not_assessed"
NO_AUTOMATED_CONCERN = "no_automated_concern_detected"
FLAGGED_FOR_REVIEW = "flagged_for_review"

DISCLAIMER = (
    "QC report thresholds are not prespecified analytical exclusions, are "
    "non-binding, are intended only to prioritize human review, and cannot "
    "change C0–C7 eligibility or inference. Absence of automated flags is not "
    "evidence of scientific correctness."
)

PRIMARY_POWER_REPRESENTATION = "absolute_log10"

HEURISTIC_PARAM_KEYS = (
    "hr_plausible_min_bpm",
    "hr_plausible_max_bpm",
    "hr_abs_diff_bpm",
    "hr_outside_band_frac",
    "gap_masked_pct",
    "polarity_score_gap",
    "polarity_score_rel_gap",
    "eeg_channel_reject_pct",
    "visual_review_n_extremes",
    "visual_review_n_random",
    "visual_review_seed",
)


@dataclass(frozen=True)
class QcReportParams:
    """Configurable heuristic thresholds for review prioritization only."""

    hr_plausible_min_bpm: float = 40.0
    hr_plausible_max_bpm: float = 180.0
    hr_abs_diff_bpm: float = 20.0
    hr_outside_band_frac: float = 0.01
    gap_masked_pct: float = 5.0
    polarity_score_gap: float = 50.0
    polarity_score_rel_gap: float = 0.05
    eeg_channel_reject_pct: float = 25.0
    visual_review_n_extremes: int = 10
    visual_review_n_random: int = 10
    visual_review_seed: int = 0
    schema_version: str = SCHEMA_VERSION
    aligned_to_c1b_usable_hr_band: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disclaimer"] = DISCLAIMER
        payload["heuristic_parameter_keys"] = list(HEURISTIC_PARAM_KEYS)
        payload["notes"] = {
            "hr_plausible_band": (
                "Default [40, 180] aligns with C1b usable median_hr gate; "
                "still a heuristic review aid when applied to IHR samples."
            ),
            "binding": False,
            "affects_c0_c7_eligibility": False,
            "affects_inference": False,
            "prespecified_analytical_exclusion": False,
        }
        return payload


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _as_text(value: Any, default: str = "") -> str:
    if _is_missing(value):
        return default
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "<na>"}:
        return default
    return text


def _as_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f", ""}:
        return False
    return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _disposition_rank(label: str) -> int:
    order = {
        FLAGGED_FOR_REVIEW: 3,
        ANALYTICAL_EXCLUSION: 2,
        NOT_ASSESSED: 1,
        NO_AUTOMATED_CONCERN: 0,
    }
    return order.get(label, 0)


def _combine_dispositions(*labels: str) -> str:
    present = [label for label in labels if label]
    if not present:
        return NOT_ASSESSED
    return max(present, key=_disposition_rank)


def _join_rules(*parts: str) -> str:
    return ";".join(part for part in parts if part)


def _hash_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "relative_hint": path.name,
        "sha256": sha256_file(path),
        "nbytes": int(path.stat().st_size),
    }


def _summary_get(summary: pd.DataFrame, key: str) -> Any:
    if summary.empty:
        return 0
    return summary.iloc[0].get(key, 0)


def _polarity_metrics(comparison: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "selected_polarity": "",
        "selected_quality_score": float("nan"),
        "best_alternative_quality_score": float("nan"),
        "polarity_score_gap_abs": float("nan"),
        "polarity_score_gap_rel": float("nan"),
        "percent_clean_ibi_selected": float("nan"),
        "polarity_metric_source": NOT_ASSESSED,
    }
    if comparison.empty or "quality_score" not in comparison.columns:
        return empty
    selected = comparison[comparison["selected"].map(_as_bool)]
    if selected.empty:
        return empty
    sel = selected.iloc[0]
    sel_score = _as_float(sel.get("quality_score"))
    others = comparison[~comparison["selected"].map(_as_bool)]
    if others.empty:
        best_alt = float("nan")
        gap_abs = float("nan")
        gap_rel = float("nan")
    else:
        best_alt = float(np.nanmax(others["quality_score"].map(_as_float).to_numpy()))
        gap_abs = (
            sel_score - best_alt
            if math.isfinite(sel_score) and math.isfinite(best_alt)
            else float("nan")
        )
        denom = max(abs(sel_score), 1e-12) if math.isfinite(sel_score) else float("nan")
        gap_rel = (
            gap_abs / denom
            if math.isfinite(gap_abs) and math.isfinite(denom)
            else float("nan")
        )
    return {
        "selected_polarity": str(sel.get("polarity", "")),
        "selected_quality_score": sel_score,
        "best_alternative_quality_score": best_alt,
        "polarity_score_gap_abs": gap_abs,
        "polarity_score_gap_rel": gap_rel,
        "percent_clean_ibi_selected": _as_float(sel.get("percent_clean_ibi")),
        "polarity_metric_source": DERIVED_METRIC,
    }


def _ihr_derived_metrics(
    series: pd.DataFrame,
    ihr_qc: Mapping[str, Any] | None,
    params: QcReportParams,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_valid_hr_samples": _as_int((ihr_qc or {}).get("n_valid_hr_samples")),
        "n_gap_masked_samples": _as_int((ihr_qc or {}).get("n_gap_masked_samples")),
        "n_grid_samples": _as_int((ihr_qc or {}).get("n_grid_samples")),
        "max_beat_gap_s": _as_float((ihr_qc or {}).get("max_beat_gap_s")),
        "ihr_status": str((ihr_qc or {}).get("status", "")),
        "ihr_warning": str((ihr_qc or {}).get("warning", "")),
        "pct_gap_masked": float("nan"),
        "hr_min_bpm": float("nan"),
        "hr_p05_bpm": float("nan"),
        "hr_median_bpm": float("nan"),
        "hr_p95_bpm": float("nan"),
        "hr_max_bpm": float("nan"),
        "n_hr_outside_plausible_band": 0,
        "frac_hr_outside_plausible_band": float("nan"),
        "n_hr_abs_diff_gt_threshold": 0,
        "ihr_derived_metric_source": NOT_ASSESSED,
    }
    if ihr_qc:
        n_grid = max(out["n_grid_samples"], 1)
        out["pct_gap_masked"] = 100.0 * out["n_gap_masked_samples"] / n_grid
        out["ihr_derived_metric_source"] = DERIVED_METRIC
    if series.empty or "instant_hr_bpm" not in series.columns:
        return out
    valid = series
    if "is_valid_hr" in series.columns:
        valid = series[series["is_valid_hr"].map(_as_bool)]
    hr = valid["instant_hr_bpm"].map(_as_float).to_numpy(dtype=float)
    hr = hr[np.isfinite(hr)]
    if hr.size == 0:
        return out
    out.update(
        {
            "hr_min_bpm": float(np.min(hr)),
            "hr_p05_bpm": float(np.percentile(hr, 5)),
            "hr_median_bpm": float(np.median(hr)),
            "hr_p95_bpm": float(np.percentile(hr, 95)),
            "hr_max_bpm": float(np.max(hr)),
            "n_valid_hr_samples": int(hr.size),
            "ihr_derived_metric_source": DERIVED_METRIC,
        }
    )
    outside = (hr < params.hr_plausible_min_bpm) | (hr > params.hr_plausible_max_bpm)
    out["n_hr_outside_plausible_band"] = int(np.sum(outside))
    out["frac_hr_outside_plausible_band"] = float(np.mean(outside))
    if hr.size >= 2:
        diffs = np.abs(np.diff(hr))
        out["n_hr_abs_diff_gt_threshold"] = int(np.sum(diffs > params.hr_abs_diff_bpm))
    return out


def _eeg_reject_pct(row: Mapping[str, Any]) -> float:
    n_input = _as_int(row.get("n_input_channels"))
    n_usable = _as_int(row.get("n_usable_channels"))
    if n_input <= 0:
        rejected = str(row.get("rejected_channels", "") or "")
        n_rej = len([p for p in rejected.split(";") if p.strip()]) if rejected else 0
        total = n_usable + n_rej
        if total <= 0:
            return float("nan")
        return 100.0 * n_rej / total
    return 100.0 * max(n_input - n_usable, 0) / n_input


def apply_observation_flags(
    row: dict[str, Any],
    params: QcReportParams,
) -> dict[str, Any]:
    """Attach per-domain dispositions and heuristic triggering rules."""
    cardiac_rules: list[str] = []
    polarity_rules: list[str] = []
    ihr_rules: list[str] = []
    eeg_rules: list[str] = []

    if row.get("c1b_present"):
        if not _as_bool(row.get("c1b_usable"), default=True):
            cardiac_disp = ANALYTICAL_EXCLUSION
            cardiac_rules.append("c1b_usable_false")
        else:
            cardiac_disp = NO_AUTOMATED_CONCERN
            warning = _as_text(row.get("c1b_warning"))
            if warning:
                cardiac_rules.append("c1b_warning_present")
                cardiac_disp = FLAGGED_FOR_REVIEW
            if not _as_text(row.get("signal_type")):
                cardiac_rules.append("signal_type_missing")
                cardiac_disp = FLAGGED_FOR_REVIEW
            if not _as_text(row.get("channel_used")):
                cardiac_rules.append("channel_used_missing")
                cardiac_disp = FLAGGED_FOR_REVIEW
    else:
        cardiac_disp = NOT_ASSESSED

    if row.get("polarity_metric_source") == DERIVED_METRIC:
        gap_abs = _as_float(row.get("polarity_score_gap_abs"))
        gap_rel = _as_float(row.get("polarity_score_gap_rel"))
        polarity_disp = NO_AUTOMATED_CONCERN
        if math.isfinite(gap_abs) and gap_abs < params.polarity_score_gap:
            polarity_rules.append(f"polarity_abs_gap<{params.polarity_score_gap:g}")
            polarity_disp = FLAGGED_FOR_REVIEW
        if math.isfinite(gap_rel) and gap_rel < params.polarity_score_rel_gap:
            polarity_rules.append(f"polarity_rel_gap<{params.polarity_score_rel_gap:g}")
            polarity_disp = FLAGGED_FOR_REVIEW
    else:
        polarity_disp = NOT_ASSESSED

    if row.get("ihr_derived_metric_source") == DERIVED_METRIC:
        ihr_disp = NO_AUTOMATED_CONCERN
        pct_gap = _as_float(row.get("pct_gap_masked"))
        if math.isfinite(pct_gap) and pct_gap > params.gap_masked_pct:
            ihr_rules.append(f"pct_gap_masked>{params.gap_masked_pct:g}")
            ihr_disp = FLAGGED_FOR_REVIEW
        frac_out = _as_float(row.get("frac_hr_outside_plausible_band"))
        if math.isfinite(frac_out) and frac_out > params.hr_outside_band_frac:
            ihr_rules.append(
                f"frac_hr_outside_plausible_band>{params.hr_outside_band_frac:g}"
            )
            ihr_disp = FLAGGED_FOR_REVIEW
        if _as_int(row.get("n_hr_abs_diff_gt_threshold")) > 0:
            ihr_rules.append(f"n_hr_abs_diff_gt_{params.hr_abs_diff_bpm:g}>0")
            ihr_disp = FLAGGED_FOR_REVIEW
        status = str(row.get("ihr_status", "")).casefold()
        if status and status not in {"ok", ""}:
            ihr_rules.append(f"ihr_status={status}")
            ihr_disp = FLAGGED_FOR_REVIEW
    else:
        ihr_disp = NOT_ASSESSED

    if row.get("c1a_present"):
        eeg_disp = NO_AUTOMATED_CONCERN
        pct_rej = _as_float(row.get("pct_channels_rejected"))
        if math.isfinite(pct_rej) and pct_rej > params.eeg_channel_reject_pct:
            eeg_rules.append(
                f"pct_channels_rejected>{params.eeg_channel_reject_pct:g}"
            )
            eeg_disp = FLAGGED_FOR_REVIEW
        if not _as_bool(row.get("spectral_qc_passed"), default=True):
            eeg_rules.append("spectral_qc_passed_false")
            eeg_disp = FLAGGED_FOR_REVIEW
        status = str(row.get("multitaper_status", "")).casefold()
        if status and status not in {"ok", ""}:
            eeg_rules.append(f"multitaper_status={status}")
            eeg_disp = FLAGGED_FOR_REVIEW
    else:
        eeg_disp = NOT_ASSESSED

    if row.get("c0_present"):
        c0_disp = (
            ANALYTICAL_EXCLUSION
            if not _as_bool(row.get("c0_usable"), default=True)
            else NO_AUTOMATED_CONCERN
        )
        if c0_disp == ANALYTICAL_EXCLUSION:
            cardiac_rules.append("c0_usable_false")
    else:
        c0_disp = NOT_ASSESSED

    overall = _combine_dispositions(
        cardiac_disp, polarity_disp, ihr_disp, eeg_disp, c0_disp
    )
    row.update(
        {
            "review_cardiac_disposition": cardiac_disp,
            "review_cardiac_triggering_rule": _join_rules(*cardiac_rules),
            "review_polarity_disposition": polarity_disp,
            "review_polarity_triggering_rule": _join_rules(*polarity_rules),
            "review_ihr_disposition": ihr_disp,
            "review_ihr_triggering_rule": _join_rules(*ihr_rules),
            "review_eeg_disposition": eeg_disp,
            "review_eeg_triggering_rule": _join_rules(*eeg_rules),
            "observation_review_disposition": overall,
            "observation_triggering_rules": _join_rules(
                *cardiac_rules, *polarity_rules, *ihr_rules, *eeg_rules
            ),
        }
    )
    return row


def _load_multitaper_by_obs(c1a_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not c1a_dir.is_dir():
        return out
    for path in sorted(c1a_dir.glob("*/multitaper_qc.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        oid = str(row.get("observation_id") or path.parent.name)
        out[oid] = row
    return out


def _load_alignment_qc(c1c_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not c1c_dir.is_dir():
        return pd.DataFrame()
    for path in sorted(c1c_dir.glob("*/alignment_qc_confirmatory.csv")):
        df = _safe_read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_stage_qc_glob(stage_dir: Path, pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not stage_dir.is_dir():
        return pd.DataFrame()
    for path in sorted(stage_dir.glob(pattern)):
        df = _safe_read_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _figure_source_observation_ids(c7_dir: Path) -> set[str]:
    ids: set[str] = set()
    source_root = c7_dir / "figures" / "source_data"
    if not source_root.is_dir():
        return ids
    for path in sorted(source_root.rglob("*.csv")):
        df = _safe_read_csv(path)
        if df.empty:
            continue
        for col in ("observation_id", "low_observation_ids", "effort_observation_ids"):
            if col not in df.columns:
                continue
            for value in df[col].dropna().astype(str):
                for part in value.replace("|", ";").split(";"):
                    part = part.strip()
                    if part:
                        ids.add(part)
    return ids


def _append_hash(
    input_hashes: list[dict[str, Any]],
    path: Path,
    relative_hint: str,
    *,
    n_files: int | None = None,
) -> None:
    rec = _hash_if_exists(path)
    if rec is None:
        return
    rec["relative_hint"] = relative_hint
    if n_files is not None:
        rec["n_files"] = n_files
    input_hashes.append(rec)


def _build_observation_qc(
    root: Path,
    params: QcReportParams,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    c0_path = root / "C0" / "data_audit.csv"
    c1b_path = root / "C1b" / "cardiac_peak_qc.csv"
    c0 = _safe_read_csv(c0_path)
    c1b = _safe_read_csv(c1b_path)
    for path, label in (
        (c0_path, "C0/data_audit.csv"),
        (c1b_path, "C1b/cardiac_peak_qc.csv"),
    ):
        if path.is_file():
            _append_hash(input_hashes, path, label)
        else:
            missing_inputs.append(label)

    multitaper = _load_multitaper_by_obs(root / "C1a")
    mt_files = sorted((root / "C1a").glob("*/multitaper_qc.csv")) if (root / "C1a").is_dir() else []
    if not mt_files:
        missing_inputs.append("C1a/*/multitaper_qc.csv")
    else:
        _append_hash(
            input_hashes,
            mt_files[0],
            "C1a/*/multitaper_qc.csv",
            n_files=len(mt_files),
        )

    figure_ids = _figure_source_observation_ids(root / "C7")

    obs_ids: list[str] = []
    if not c0.empty and "observation_id" in c0.columns:
        obs_ids.extend(c0["observation_id"].astype(str).tolist())
    if not c1b.empty and "observation_id" in c1b.columns:
        obs_ids.extend(c1b["observation_id"].astype(str).tolist())
    obs_ids.extend(multitaper.keys())
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for oid in obs_ids:
        if oid not in seen:
            seen.add(oid)
            ordered_ids.append(oid)
    ordered_ids = sorted(ordered_ids)

    c0_by = (
        {str(r["observation_id"]): r for _, r in c0.iterrows()}
        if not c0.empty and "observation_id" in c0.columns
        else {}
    )
    c1b_by = (
        {str(r["observation_id"]): r for _, r in c1b.iterrows()}
        if not c1b.empty and "observation_id" in c1b.columns
        else {}
    )

    rows: list[dict[str, Any]] = []
    for oid in ordered_ids:
        c0_row = c0_by.get(oid)
        c1b_row = c1b_by.get(oid)
        mt_row = multitaper.get(oid)

        row: dict[str, Any] = {
            "observation_id": oid,
            "dataset_id": "",
            "participant_id": "",
            "subject_id": "",
            "session_id": "",
            "condition": "",
            "task": "",
            "c0_present": c0_row is not None,
            "c1a_present": mt_row is not None,
            "c1b_present": c1b_row is not None,
            "listed_in_figure_source_tables": oid in figure_ids,
        }
        if c0_row is not None:
            row.update(
                {
                    "dataset_id": str(c0_row.get("dataset_id", "")),
                    "participant_id": str(c0_row.get("participant_id", "")),
                    "subject_id": str(c0_row.get("subject_id", "")),
                    "session_id": str(c0_row.get("session_id", "")),
                    "condition": str(c0_row.get("condition", "")),
                    "task": str(c0_row.get("task", "")),
                    "c0_usable": _as_bool(c0_row.get("usable")),
                    "c0_exclusion_reason": str(c0_row.get("exclusion_reason", "") or ""),
                    "c0_skip_reason": str(c0_row.get("skip_reason", "") or ""),
                    "cardiac_exists": _as_bool(c0_row.get("cardiac_exists")),
                    "eeg_exists": _as_bool(c0_row.get("eeg_exists")),
                    "available_cardiac_channel": str(
                        c0_row.get("available_cardiac_channel", "") or ""
                    ),
                    "c0_cardiac_signal_type": str(
                        c0_row.get("cardiac_signal_type", "") or ""
                    ),
                }
            )
        if c1b_row is not None:
            row.update(
                {
                    "dataset_id": row["dataset_id"]
                    or str(c1b_row.get("dataset_id", "")),
                    "subject_id": row["subject_id"]
                    or str(c1b_row.get("subject_id", "")),
                    "condition": row["condition"]
                    or str(c1b_row.get("condition", "")),
                    "task": row["task"] or str(c1b_row.get("task", "")),
                    "channel_used": str(c1b_row.get("channel_used", "") or ""),
                    "signal_type": str(c1b_row.get("signal_type", "") or ""),
                    "detector_used": str(c1b_row.get("detector_used", "") or ""),
                    "detector_polarity": str(
                        c1b_row.get("detector_polarity", "") or ""
                    ),
                    "selection_reason": str(c1b_row.get("selection_reason", "") or ""),
                    "n_raw_peaks": _as_int(c1b_row.get("n_raw_peaks")),
                    "n_clean_ibis": _as_int(c1b_row.get("n_clean_ibis")),
                    "n_accepted_peaks": _as_int(c1b_row.get("n_accepted_peaks")),
                    "clean_ibi_coverage_s": _as_float(
                        c1b_row.get("clean_ibi_coverage_s")
                    ),
                    "median_hr_bpm_peaks": _as_float(c1b_row.get("median_hr_bpm")),
                    "c1b_usable": _as_bool(c1b_row.get("usable")),
                    "c1b_warning": _as_text(c1b_row.get("warning")),
                }
            )
        else:
            row.setdefault("signal_type", row.get("c0_cardiac_signal_type", ""))
            row.setdefault("channel_used", row.get("available_cardiac_channel", ""))

        obs_c1b = root / "C1b" / oid
        comparison = _safe_read_csv(obs_c1b / "peak_detector_comparison.csv")
        row.update(_polarity_metrics(comparison))

        ihr_qc_df = _safe_read_csv(obs_c1b / "instant_hr_qc.csv")
        ihr_qc = ihr_qc_df.iloc[0].to_dict() if not ihr_qc_df.empty else None
        ihr_series = _safe_read_csv(obs_c1b / "features_instant_hr.csv")
        row.update(_ihr_derived_metrics(ihr_series, ihr_qc, params))

        if mt_row is not None:
            row.update(
                {
                    "n_input_channels": _as_int(mt_row.get("n_input_channels")),
                    "n_usable_channels": _as_int(mt_row.get("n_usable_channels")),
                    "rejected_channels": str(mt_row.get("rejected_channels", "") or ""),
                    "pct_channels_rejected": _eeg_reject_pct(mt_row),
                    "spectral_qc_passed": _as_bool(mt_row.get("spectral_qc_passed")),
                    "multitaper_status": str(mt_row.get("status", "") or ""),
                    "multitaper_warning": str(mt_row.get("warning", "") or ""),
                    "n_windows": _as_int(mt_row.get("n_windows")),
                    "n_nonfinite_power_rows": _as_int(
                        mt_row.get("n_nonfinite_power_rows")
                    ),
                }
            )

        rows.append(apply_observation_flags(row, params))

    ihr_files = (
        sorted((root / "C1b").glob("*/instant_hr_qc.csv"))
        if (root / "C1b").is_dir()
        else []
    )
    if not ihr_files:
        missing_inputs.append("C1b/*/instant_hr_qc.csv")
    else:
        _append_hash(
            input_hashes,
            ihr_files[0],
            "C1b/*/instant_hr_qc.csv",
            n_files=len(ihr_files),
        )

    return pd.DataFrame(rows)


def _build_duration_qc(
    root: Path,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    elig_path = root / "C0" / "eligibility_by_duration.csv"
    elig = _safe_read_csv(elig_path)
    if elig_path.is_file():
        _append_hash(input_hashes, elig_path, "C0/eligibility_by_duration.csv")
    else:
        missing_inputs.append("C0/eligibility_by_duration.csv")

    align = _load_alignment_qc(root / "C1c")
    align_files = (
        sorted((root / "C1c").glob("*/alignment_qc_confirmatory.csv"))
        if (root / "C1c").is_dir()
        else []
    )
    if not align_files:
        missing_inputs.append("C1c/*/alignment_qc_confirmatory.csv")
    else:
        _append_hash(
            input_hashes,
            align_files[0],
            "C1c/*/alignment_qc_confirmatory.csv",
            n_files=len(align_files),
        )

    c3 = _load_stage_qc_glob(root / "C3", "confirmatory_endpoint_qc_D*.csv")
    c3_files = (
        sorted((root / "C3").glob("confirmatory_endpoint_qc_D*.csv"))
        if (root / "C3").is_dir()
        else []
    )
    if not c3_files:
        missing_inputs.append("C3/confirmatory_endpoint_qc_D*.csv")
    else:
        for path in c3_files:
            _append_hash(input_hashes, path, f"C3/{path.name}")

    keys: list[tuple[str, int]] = []
    if not elig.empty and "observation_id" in elig.columns:
        for _, r in elig.iterrows():
            keys.append((str(r["observation_id"]), _as_int(r["duration_s"])))
    if not align.empty and "observation_id" in align.columns:
        for _, r in align.iterrows():
            keys.append((str(r["observation_id"]), _as_int(r["duration_s"])))
    uniq = sorted(set(keys), key=lambda x: (x[0], x[1]))

    elig_by = (
        {
            (str(r["observation_id"]), _as_int(r["duration_s"])): r
            for _, r in elig.iterrows()
        }
        if not elig.empty and "observation_id" in elig.columns
        else {}
    )
    align_by = (
        {
            (str(r["observation_id"]), _as_int(r["duration_s"])): r
            for _, r in align.iterrows()
        }
        if not align.empty and "observation_id" in align.columns
        else {}
    )

    primary_c3 = pd.DataFrame()
    if not c3.empty:
        mask = (
            c3["power_representation"].astype(str) == PRIMARY_POWER_REPRESENTATION
            if "power_representation" in c3.columns
            else pd.Series(False, index=c3.index)
        )
        if "is_primary_representation" in c3.columns:
            mask = mask | c3["is_primary_representation"].map(_as_bool)
        primary_c3 = c3[mask] if mask.any() else c3

    rows: list[dict[str, Any]] = []
    for oid, duration_s in uniq:
        e = elig_by.get((oid, duration_s))
        a = align_by.get((oid, duration_s))
        row: dict[str, Any] = {
            "observation_id": oid,
            "duration_s": duration_s,
            "duration_role": "",
            "dataset_id": "",
            "contrast_id": "",
            "c0_status": NOT_ASSESSED,
            "c0_exclusion_code": "",
            "c1c_eligible": NOT_ASSESSED,
            "c1c_exclusion_reason": "",
            "n_common_support": float("nan"),
            "available_support_s": float("nan"),
            "primary_rep_any_endpoint_ineligible": NOT_ASSESSED,
            "primary_rep_n_ineligible": float("nan"),
            "duration_review_disposition": NOT_ASSESSED,
            "duration_triggering_rule": "",
        }
        rules: list[str] = []
        disp = NO_AUTOMATED_CONCERN
        if e is not None:
            status = str(e.get("status", "") or "")
            row["c0_status"] = status
            row["c0_exclusion_code"] = str(e.get("exclusion_code", "") or "")
            row["dataset_id"] = str(e.get("dataset_id", "") or "")
            row["contrast_id"] = str(e.get("contrast_id", "") or "")
            if status.casefold() not in {"eligible", "ok", ""}:
                disp = ANALYTICAL_EXCLUSION
                rules.append(f"c0_status={status}")
        if a is not None:
            eligible = _as_bool(a.get("eligible"))
            row["c1c_eligible"] = eligible
            row["c1c_exclusion_reason"] = str(a.get("exclusion_reason", "") or "")
            row["duration_role"] = str(a.get("duration_role", "") or "")
            row["n_common_support"] = _as_int(a.get("n_common_support"))
            row["available_support_s"] = _as_float(a.get("available_support_s"))
            if not eligible:
                disp = ANALYTICAL_EXCLUSION
                rules.append("c1c_eligible_false")
        if not primary_c3.empty:
            sub = primary_c3[
                (primary_c3["observation_id"].astype(str) == oid)
                & (primary_c3["duration_s"].map(_as_int) == duration_s)
            ]
            if not sub.empty and "eligible" in sub.columns:
                n_inelig = int((~sub["eligible"].map(_as_bool)).sum())
                row["primary_rep_n_ineligible"] = n_inelig
                row["primary_rep_any_endpoint_ineligible"] = n_inelig > 0
                if n_inelig > 0:
                    disp = _combine_dispositions(disp, ANALYTICAL_EXCLUSION)
                    rules.append("c3_primary_rep_ineligible")
            else:
                row["primary_rep_any_endpoint_ineligible"] = NOT_ASSESSED
        if e is None and a is None:
            disp = NOT_ASSESSED
        row["duration_review_disposition"] = disp
        row["duration_triggering_rule"] = _join_rules(*rules)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_coupling_unit_qc(
    root: Path,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    c2 = _load_stage_qc_glob(root / "C2", "confirmatory_cross_correlation_qc_D*.csv")
    c3 = _load_stage_qc_glob(root / "C3", "confirmatory_endpoint_qc_D*.csv")
    c2_files = (
        sorted((root / "C2").glob("confirmatory_cross_correlation_qc_D*.csv"))
        if (root / "C2").is_dir()
        else []
    )
    if not c2_files:
        missing_inputs.append("C2/confirmatory_cross_correlation_qc_D*.csv")
    else:
        for path in c2_files:
            _append_hash(input_hashes, path, f"C2/{path.name}")
    if c3.empty and not (root / "C3").is_dir():
        missing_inputs.append("C3/confirmatory_endpoint_qc_D*.csv")

    peak = _safe_read_csv(root / "C3" / "peak_fit_qc.csv")
    peak_path = root / "C3" / "peak_fit_qc.csv"
    if peak_path.is_file():
        _append_hash(input_hashes, peak_path, "C3/peak_fit_qc.csv")
    elif (root / "C3").is_dir():
        missing_inputs.append("C3/peak_fit_qc.csv")

    if c3.empty and c2.empty:
        return pd.DataFrame()

    base = c3 if not c3.empty else c2
    key_cols = [
        c
        for c in (
            "observation_id",
            "duration_s",
            "band",
            "power_representation",
            "endpoint_name",
        )
        if c in base.columns
    ]
    rows: list[dict[str, Any]] = []
    for _, r in base.iterrows():
        oid = str(r.get("observation_id", ""))
        duration_s = _as_int(r.get("duration_s"))
        band = str(r.get("band", ""))
        rep = str(r.get("power_representation", ""))
        endpoint = str(r.get("endpoint_name", ""))
        row: dict[str, Any] = {
            "observation_id": oid,
            "duration_s": duration_s,
            "band": band,
            "power_representation": rep,
            "endpoint_name": endpoint,
            "is_primary_representation": (
                _as_bool(r.get("is_primary_representation"))
                if "is_primary_representation" in r.index
                else (rep == PRIMARY_POWER_REPRESENTATION)
            ),
            "c2_exclusion_reason": "",
            "c3_eligible": NOT_ASSESSED,
            "c3_exclusion_reason": "",
            "n_common_support": float("nan"),
            "peak_fit_converged": NOT_ASSESSED,
            "has_identifiable_peak": NOT_ASSESSED,
            "coupling_review_disposition": NOT_ASSESSED,
            "coupling_triggering_rule": "",
        }
        rules: list[str] = []
        disp = NO_AUTOMATED_CONCERN
        if not c2.empty:
            c2_sub = c2[
                (c2["observation_id"].astype(str) == oid)
                & (c2["duration_s"].map(_as_int) == duration_s)
                & (c2["band"].astype(str) == band)
                & (c2["power_representation"].astype(str) == rep)
            ]
            if not c2_sub.empty:
                excl = str(c2_sub.iloc[0].get("exclusion_reason", "") or "")
                row["c2_exclusion_reason"] = excl
                if "n_common_support" in c2_sub.columns:
                    row["n_common_support"] = _as_int(
                        c2_sub.iloc[0].get("n_common_support")
                    )
                if excl.strip():
                    disp = ANALYTICAL_EXCLUSION
                    rules.append("c2_exclusion_reason_present")
        if not c3.empty:
            eligible = _as_bool(r.get("eligible"), default=True)
            row["c3_eligible"] = eligible
            row["c3_exclusion_reason"] = str(r.get("exclusion_reason", "") or "")
            if "n_common_support" in r.index and not math.isfinite(
                _as_float(row["n_common_support"])
            ):
                row["n_common_support"] = _as_int(r.get("n_common_support"))
            if not eligible:
                disp = ANALYTICAL_EXCLUSION
                rules.append("c3_eligible_false")
        if not peak.empty:
            pmask = (
                (peak["observation_id"].astype(str) == oid)
                & (peak["duration_s"].map(_as_int) == duration_s)
                & (peak["band"].astype(str) == band)
                & (peak["power_representation"].astype(str) == rep)
            )
            if "endpoint_name" in peak.columns:
                pmask = pmask & (peak["endpoint_name"].astype(str) == endpoint)
            psub = peak[pmask]
            if not psub.empty:
                prow = psub.iloc[0]
                row["peak_fit_converged"] = _as_bool(prow.get("converged"), default=True)
                row["has_identifiable_peak"] = _as_bool(
                    prow.get("has_identifiable_peak")
                )
                if not row["peak_fit_converged"]:
                    disp = FLAGGED_FOR_REVIEW
                    rules.append("peak_fit_converged_false")
        row["coupling_review_disposition"] = disp
        row["coupling_triggering_rule"] = _join_rules(*rules)
        for col in ("dataset_id", "subject_id", "condition", "task", "pair"):
            if col in r.index:
                row[col] = r.get(col)
        rows.append(row)
    df = pd.DataFrame(rows)
    if key_cols and not df.empty:
        df = df.sort_values(key_cols).reset_index(drop=True)
    return df


def _build_contrast_qc(
    root: Path,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    path = root / "C5" / "paired_contrasts.csv"
    df = _safe_read_csv(path)
    if path.is_file():
        _append_hash(input_hashes, path, "C5/paired_contrasts.csv")
    else:
        missing_inputs.append("C5/paired_contrasts.csv")
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        eligible = _as_bool(r.get("contrast_eligible"), default=True)
        rules: list[str] = []
        disp = NO_AUTOMATED_CONCERN
        if not eligible:
            disp = ANALYTICAL_EXCLUSION
            rules.append("contrast_eligible_false")
        rows.append(
            {
                "dataset_id": str(r.get("dataset_id", "")),
                "contrast_id": str(r.get("contrast_id", "")),
                "participant_id": str(r.get("participant_id", "")),
                "session_id": str(r.get("session_id", "")),
                "duration_s": _as_int(r.get("duration_s")),
                "band": str(r.get("band", "")),
                "power_representation": str(r.get("power_representation", "")),
                "endpoint_name": str(r.get("endpoint_name", "")),
                "is_primary_representation": _as_bool(
                    r.get("is_primary_representation")
                ),
                "contrast_eligible": eligible,
                "contrast_exclusion_reason": str(
                    r.get("contrast_exclusion_reason", "") or ""
                ),
                "mu_contrast_eligible": _as_bool(r.get("mu_contrast_eligible")),
                "low_observation_ids": str(r.get("low_observation_ids", "") or ""),
                "effort_observation_ids": str(
                    r.get("effort_observation_ids", "") or ""
                ),
                "delta_endpoint_index": _as_float(r.get("delta_endpoint_index")),
                "contrast_review_disposition": disp,
                "contrast_triggering_rule": _join_rules(*rules),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [
                "contrast_id",
                "participant_id",
                "session_id",
                "duration_s",
                "band",
                "power_representation",
                "endpoint_name",
            ]
        ).reset_index(drop=True)
    return out


def _build_null_unit_qc(
    root: Path,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    path = root / "C4" / "null_qc.csv"
    df = _safe_read_csv(path)
    if path.is_file():
        _append_hash(input_hashes, path, "C4/null_qc.csv")
    else:
        missing_inputs.append("C4/null_qc.csv")
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        status = str(r.get("status", "") or "")
        rules: list[str] = []
        disp = NO_AUTOMATED_CONCERN
        if status.casefold() not in {"ok", "complete", "success", ""}:
            disp = FLAGGED_FOR_REVIEW
            rules.append(f"null_status={status}")
        n_req = _as_int(r.get("n_surrogates_requested"))
        n_fin = _as_int(r.get("n_surrogates_finite"))
        if n_req > 0 and n_fin < n_req:
            disp = FLAGGED_FOR_REVIEW
            rules.append("n_surrogates_finite<n_requested")
        rows.append(
            {
                "observation_id": str(r.get("observation_id", "")),
                "duration_s": _as_int(r.get("duration_s")),
                "band": str(r.get("band", "")),
                "power_representation": str(r.get("power_representation", "")),
                "endpoint_name": str(r.get("endpoint_name", "")),
                "null_type": str(r.get("null_type", "")),
                "status": status,
                "n_surrogates_requested": n_req,
                "n_surrogates_finite": n_fin,
                "notes": str(r.get("notes", "") or ""),
                "null_review_disposition": disp,
                "null_triggering_rule": _join_rules(*rules),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [
                "observation_id",
                "duration_s",
                "band",
                "power_representation",
                "null_type",
            ]
        ).reset_index(drop=True)
    return out


def _build_model_qc(
    root: Path,
    missing_inputs: list[str],
    input_hashes: list[dict[str, Any]],
) -> pd.DataFrame:
    path = root / "C6" / "inference_qc.csv"
    df = _safe_read_csv(path)
    if path.is_file():
        _append_hash(input_hashes, path, "C6/inference_qc.csv")
    else:
        missing_inputs.append("C6/inference_qc.csv")
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        status = str(r.get("status", "") or "")
        converged = _as_bool(r.get("converged"), default=True)
        rules: list[str] = []
        disp = NO_AUTOMATED_CONCERN
        if not converged:
            disp = FLAGGED_FOR_REVIEW
            rules.append("converged_false")
        if status.casefold() in {"failed", "error"}:
            disp = FLAGGED_FOR_REVIEW
            rules.append(f"inference_status={status}")
        rows.append(
            {
                "component": str(r.get("component", "")),
                "endpoint_name": str(r.get("endpoint_name", "")),
                "duration_s": _as_int(r.get("duration_s")),
                "status": status,
                "model_backend": str(r.get("model_backend", "") or ""),
                "converged": converged,
                "n_obs": _as_int(r.get("n_obs")),
                "n_groups": _as_int(r.get("n_groups")),
                "notes": str(r.get("notes", "") or ""),
                "model_review_disposition": disp,
                "model_triggering_rule": _join_rules(*rules),
            }
        )
    return pd.DataFrame(rows)


def _build_visual_review_list(
    observation_qc: pd.DataFrame,
    params: QcReportParams,
) -> pd.DataFrame:
    columns = [
        "grain",
        "observation_id",
        "priority",
        "reason",
        "triggering_rule",
        "disposition",
        "metric_value",
    ]
    if observation_qc.empty:
        return pd.DataFrame(columns=columns)
    entries: list[dict[str, Any]] = []
    flagged = observation_qc[
        observation_qc["observation_review_disposition"] == FLAGGED_FOR_REVIEW
    ]
    for _, r in flagged.iterrows():
        entries.append(
            {
                "grain": "observation",
                "observation_id": r["observation_id"],
                "priority": "flagged",
                "reason": "heuristic_or_warning_flag",
                "triggering_rule": r.get("observation_triggering_rules", ""),
                "disposition": FLAGGED_FOR_REVIEW,
                "metric_value": float("nan"),
            }
        )

    def _add_extremes(
        df: pd.DataFrame,
        column: str,
        *,
        ascending: bool,
        reason: str,
    ) -> None:
        if column not in df.columns:
            return
        sub = df.copy()
        sub["_v"] = sub[column].map(_as_float)
        sub = sub[np.isfinite(sub["_v"].to_numpy(dtype=float))]
        if sub.empty:
            return
        sub = sub.sort_values(["_v", "observation_id"], ascending=[ascending, True]).head(
            params.visual_review_n_extremes
        )
        for _, r in sub.iterrows():
            entries.append(
                {
                    "grain": "observation",
                    "observation_id": r["observation_id"],
                    "priority": "extreme",
                    "reason": reason,
                    "triggering_rule": f"extreme:{column}",
                    "disposition": r.get(
                        "observation_review_disposition", NO_AUTOMATED_CONCERN
                    ),
                    "metric_value": r["_v"],
                }
            )

    _add_extremes(
        observation_qc,
        "polarity_score_gap_abs",
        ascending=True,
        reason="smallest_polarity_score_gap",
    )
    _add_extremes(
        observation_qc,
        "pct_gap_masked",
        ascending=False,
        reason="largest_pct_gap_masked",
    )
    _add_extremes(
        observation_qc,
        "pct_channels_rejected",
        ascending=False,
        reason="largest_pct_channels_rejected",
    )

    already = {e["observation_id"] for e in entries}
    pool = sorted(
        oid
        for oid in observation_qc["observation_id"].astype(str).tolist()
        if oid not in already
    )
    rng = random.Random(params.visual_review_seed)
    rng.shuffle(pool)
    for oid in pool[: params.visual_review_n_random]:
        entries.append(
            {
                "grain": "observation",
                "observation_id": oid,
                "priority": "random",
                "reason": "random_sample",
                "triggering_rule": f"visual_review_seed={params.visual_review_seed}",
                "disposition": NO_AUTOMATED_CONCERN,
                "metric_value": float("nan"),
            }
        )

    out = pd.DataFrame(entries)
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = out.drop_duplicates(subset=["observation_id", "reason"], keep="first")
    out = out.sort_values(
        ["priority", "reason", "observation_id"]
    ).reset_index(drop=True)
    return out


def _build_dataset_summary(
    *,
    dataset_id: str,
    observation_qc: pd.DataFrame,
    duration_qc: pd.DataFrame,
    coupling_unit_qc: pd.DataFrame,
    contrast_qc: pd.DataFrame,
    null_unit_qc: pd.DataFrame,
    model_qc: pd.DataFrame,
    missing_inputs: Sequence[str],
) -> pd.DataFrame:
    def _count_disp(df: pd.DataFrame, col: str, label: str) -> int:
        if df.empty or col not in df.columns:
            return 0
        return int((df[col] == label).sum())

    domains_not_assessed = []
    if observation_qc.empty:
        domains_not_assessed.append("observation")
    if duration_qc.empty:
        domains_not_assessed.append("duration")
    if coupling_unit_qc.empty:
        domains_not_assessed.append("coupling_unit")
    if contrast_qc.empty:
        domains_not_assessed.append("contrast")
    if null_unit_qc.empty:
        domains_not_assessed.append("null_unit")
    if model_qc.empty:
        domains_not_assessed.append("model")

    row = {
        "dataset_id": dataset_id,
        "n_observations": int(len(observation_qc)),
        "n_obs_flagged_for_review": _count_disp(
            observation_qc, "observation_review_disposition", FLAGGED_FOR_REVIEW
        ),
        "n_obs_analytical_exclusion": _count_disp(
            observation_qc, "observation_review_disposition", ANALYTICAL_EXCLUSION
        ),
        "n_obs_no_automated_concern": _count_disp(
            observation_qc, "observation_review_disposition", NO_AUTOMATED_CONCERN
        ),
        "n_obs_not_assessed": _count_disp(
            observation_qc, "observation_review_disposition", NOT_ASSESSED
        ),
        "n_review_cardiac_flagged": _count_disp(
            observation_qc, "review_cardiac_disposition", FLAGGED_FOR_REVIEW
        ),
        "n_review_polarity_flagged": _count_disp(
            observation_qc, "review_polarity_disposition", FLAGGED_FOR_REVIEW
        ),
        "n_review_ihr_flagged": _count_disp(
            observation_qc, "review_ihr_disposition", FLAGGED_FOR_REVIEW
        ),
        "n_review_eeg_flagged": _count_disp(
            observation_qc, "review_eeg_disposition", FLAGGED_FOR_REVIEW
        ),
        "n_duration_rows": int(len(duration_qc)),
        "n_coupling_unit_rows": int(len(coupling_unit_qc)),
        "n_contrast_rows": int(len(contrast_qc)),
        "n_null_unit_rows": int(len(null_unit_qc)),
        "n_model_rows": int(len(model_qc)),
        "domains_not_assessed": ";".join(domains_not_assessed) or "(none)",
        "missing_inputs": ";".join(missing_inputs) or "(none)",
        "primary_duration_s": EXPECTED_PRIMARY_DURATION_S,
        "expected_durations_s": ";".join(str(d) for d in EXPECTED_DURATIONS_S),
    }
    return pd.DataFrame([row])


def _write_markdown_report(
    path: Path,
    *,
    dataset_id: str,
    params: QcReportParams,
    summary: pd.DataFrame,
    missing_inputs: Sequence[str],
    observation_qc: pd.DataFrame,
) -> None:
    s = summary.iloc[0].to_dict() if not summary.empty else {}
    lines = [
        f"# Confirmatory QC report — `{dataset_id}`",
        "",
        "## Disclaimer",
        "",
        DISCLAIMER,
        "",
        "This layer does **not** validate scientific truth. Labels such as "
        f"`{NO_AUTOMATED_CONCERN}` mean only that no configured automated "
        "review heuristic fired.",
        "",
        "## Parameters",
        "",
        "Resolved heuristic thresholds are written to `qc_params.json`. "
        "They are review aids only.",
        "",
        "```json",
        json.dumps(params.to_json_dict(), indent=2, sort_keys=True),
        "```",
        "",
        "## Dataset summary",
        "",
        f"- Observations: **{s.get('n_observations', 0)}**",
        f"- `{FLAGGED_FOR_REVIEW}`: **{s.get('n_obs_flagged_for_review', 0)}**",
        f"- `{ANALYTICAL_EXCLUSION}`: **{s.get('n_obs_analytical_exclusion', 0)}**",
        f"- `{NO_AUTOMATED_CONCERN}`: **{s.get('n_obs_no_automated_concern', 0)}**",
        f"- `{NOT_ASSESSED}`: **{s.get('n_obs_not_assessed', 0)}**",
        f"- review_cardiac flagged: **{s.get('n_review_cardiac_flagged', 0)}**",
        f"- review_polarity flagged: **{s.get('n_review_polarity_flagged', 0)}**",
        f"- review_ihr flagged: **{s.get('n_review_ihr_flagged', 0)}**",
        f"- review_eeg flagged: **{s.get('n_review_eeg_flagged', 0)}**",
        f"- Domains `{NOT_ASSESSED}`: `{s.get('domains_not_assessed', '')}`",
        "",
        "## Missing inputs",
        "",
    ]
    if missing_inputs:
        for item in missing_inputs:
            lines.append(f"- `{item}`")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Observation flags (flagged only)", ""])
    if observation_qc.empty:
        lines.append("_No observation rows._")
    else:
        flagged = observation_qc[
            observation_qc["observation_review_disposition"] == FLAGGED_FOR_REVIEW
        ]
        if flagged.empty:
            lines.append(f"_No observations with `{FLAGGED_FOR_REVIEW}`._")
        else:
            lines.append("| observation_id | disposition | triggering_rules |")
            lines.append("|---|---|---|")
            for _, r in flagged.sort_values("observation_id").iterrows():
                lines.append(
                    f"| `{r['observation_id']}` | `{r['observation_review_disposition']}` | "
                    f"`{r.get('observation_triggering_rules', '')}` |"
                )
    lines.extend(
        [
            "",
            "## Grain tables",
            "",
            "- `observation_qc.csv`",
            "- `duration_qc.csv`",
            "- `coupling_unit_qc.csv`",
            "- `contrast_qc.csv`",
            "- `null_unit_qc.csv`",
            "- `model_qc.csv`",
            "- `dataset_qc_summary.csv`",
            "- `visual_review_list.csv`",
            "",
            "Model / contrast / null grains are separate; see those CSVs rather "
            "than collapsing into observation rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("", encoding="utf-8")
        return
    df.to_csv(path, index=False)


def generate_qc_report(
    dataset_output_root: str | Path,
    *,
    qc_params: QcReportParams | None = None,
    qc_root: str | Path | None = None,
    dataset_id: str | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Generate read-only QC tables under ``QC/`` (or ``qc_root``).

    Never opens C0–C7 paths for write. Reruns overwrite only files inside the
    QC directory.
    """
    root = Path(dataset_output_root).expanduser().resolve()
    params = qc_params or QcReportParams()
    out_dir = (
        Path(qc_root).expanduser().resolve()
        if qc_root is not None
        else root / QC_DIRNAME
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EXTENSIONS_DIRNAME).mkdir(parents=True, exist_ok=True)

    missing_inputs: list[str] = []
    input_hashes: list[dict[str, Any]] = []

    for stage in ("C0", "C1a", "C1b", "C1c", "C2", "C3", "C4", "C5", "C6", "C7"):
        if not (root / stage).is_dir():
            missing_inputs.append(f"{stage}/ (stage directory absent)")

    observation_qc = _build_observation_qc(root, params, missing_inputs, input_hashes)
    duration_qc = _build_duration_qc(root, missing_inputs, input_hashes)
    coupling_unit_qc = _build_coupling_unit_qc(root, missing_inputs, input_hashes)
    contrast_qc = _build_contrast_qc(root, missing_inputs, input_hashes)
    null_unit_qc = _build_null_unit_qc(root, missing_inputs, input_hashes)
    model_qc = _build_model_qc(root, missing_inputs, input_hashes)
    visual = _build_visual_review_list(observation_qc, params)

    resolved_dataset_id = dataset_id or ""
    if not resolved_dataset_id and not observation_qc.empty:
        resolved_dataset_id = str(observation_qc.iloc[0].get("dataset_id", "") or "")
    if not resolved_dataset_id:
        resolved_dataset_id = root.name

    seen_m: set[str] = set()
    missing_unique: list[str] = []
    for item in missing_inputs:
        if item not in seen_m:
            seen_m.add(item)
            missing_unique.append(item)

    summary = _build_dataset_summary(
        dataset_id=resolved_dataset_id,
        observation_qc=observation_qc,
        duration_qc=duration_qc,
        coupling_unit_qc=coupling_unit_qc,
        contrast_qc=contrast_qc,
        null_unit_qc=null_unit_qc,
        model_qc=model_qc,
        missing_inputs=missing_unique,
    )

    outputs = {
        "observation_qc.csv": observation_qc,
        "duration_qc.csv": duration_qc,
        "coupling_unit_qc.csv": coupling_unit_qc,
        "contrast_qc.csv": contrast_qc,
        "null_unit_qc.csv": null_unit_qc,
        "model_qc.csv": model_qc,
        "dataset_qc_summary.csv": summary,
        "visual_review_list.csv": visual,
    }
    generated_files: list[str] = []
    for name, df in outputs.items():
        _write_csv(out_dir / name, df)
        generated_files.append(name)

    params_path = out_dir / "qc_params.json"
    params_path.write_text(
        json.dumps(params.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generated_files.append("qc_params.json")

    _write_markdown_report(
        out_dir / "QC_REPORT.md",
        dataset_id=resolved_dataset_id,
        params=params,
        summary=summary,
        missing_inputs=missing_unique,
        observation_qc=observation_qc,
    )
    generated_files.append("QC_REPORT.md")

    timestamp = datetime.now(timezone.utc).isoformat()
    commit = (
        git_commit_hash(repo_root) if repo_root is not None else git_commit_hash()
    )
    manifest = {
        "schema_version": params.schema_version,
        "dataset_id": resolved_dataset_id,
        "dataset_output_root": str(root),
        "qc_root": str(out_dir),
        "generated_at_utc": timestamp,
        "git_commit": commit,
        "disclaimer": DISCLAIMER,
        "qc_params": params.to_json_dict(),
        "input_files": input_hashes,
        "missing_stages_or_inputs": missing_unique,
        "generated_output_files": generated_files + [f"{EXTENSIONS_DIRNAME}/"],
    }
    (out_dir / "qc_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generated_files.append("qc_manifest.json")

    print(f"[confirmatory] QC report wrote {out_dir}")
    print(
        "[confirmatory] QC summary "
        f"flagged={_summary_get(summary, 'n_obs_flagged_for_review')} "
        f"obs={_summary_get(summary, 'n_observations')} "
        f"missing={len(missing_unique)}"
    )
    return out_dir


def run_qc_report_from_config(
    config_path: str | Path,
    *,
    master_config: str | Path | None = None,
    config_dir: str | Path | None = None,
    qc_root: str | Path | None = None,
    qc_params: QcReportParams | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve dataset YAML and write QC under its ``output_root``."""
    from .run import discover_config_dir, resolve_master_and_dataset

    _master, dataset, _mp, _dp = resolve_master_and_dataset(
        config_path,
        master_config=master_config,
        config_dir=config_dir,
    )
    resolved_repo = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else discover_config_dir(config_dir).parent
    )
    return generate_qc_report(
        dataset.output_root,
        qc_params=qc_params,
        qc_root=qc_root,
        dataset_id=dataset.dataset_id,
        repo_root=resolved_repo,
    )


__all__ = [
    "ANALYTICAL_EXCLUSION",
    "DERIVED_METRIC",
    "DISCLAIMER",
    "FLAGGED_FOR_REVIEW",
    "HEURISTIC_REVIEW_SUGGESTION",
    "NO_AUTOMATED_CONCERN",
    "NOT_ASSESSED",
    "PIPELINE_FACT",
    "QcReportParams",
    "SCHEMA_VERSION",
    "apply_observation_flags",
    "generate_qc_report",
    "run_qc_report_from_config",
]
