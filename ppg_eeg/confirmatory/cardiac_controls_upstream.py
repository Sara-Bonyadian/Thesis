"""Upstream observation-level cardiac controls for Figure 3 Panel D.

This module computes event-locked control estimates before C7, using aligned
observation-level time series and C1b detector outputs. C7 must only summarize.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..core_eeg_ppg.output_layout import safe_subject_dir_name
from .correlation import (
    BAND_ORDER,
    POWER_REPRESENTATIONS,
    PRIMARY_POWER_REPRESENTATION,
    compute_signed_lag_curves,
)
from .duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S, contract_for_duration
from .endpoints import compute_endpoints_from_curves, min_common_support_required
from .panel_f_topography_gamma import (
    compute_observation_channel_zlpi,
    load_ecg_prone_channels,
)
from .reason_codes import (
    ARTIFACT_CONTROL_NOT_AVAILABLE,
    INSUFFICIENT_COMMON_MONTAGE,
    INSUFFICIENT_COMMON_SUPPORT,
    MISSING_EVENT_SERIES,
    MISSING_REQUIRED_MODALITY,
    UNSUPPORTED_CONTROL_FOR_MODALITY,
)

OBSERVATION_CONTROLS_FILENAME = "cardiac_controls_observation_level.csv"
DATASET_QC_FILENAME = "cardiac_controls_dataset_qc.csv"
MASK_DIAGNOSTICS_FILENAME = "cardiac_controls_mask_diagnostics.csv"
METADATA_FILENAME = "cardiac_controls_metadata.json"

CONTROL_BASELINE = "baseline"
CONTROL_ECG_MASK = "ecg_r_peak_mask"
CONTROL_PPG_MASK = "ppg_systolic_peak_mask"
CONTROL_ECG_TEMPLATE = "ecg_cardiac_template_subtraction"
CONTROL_PPG_TEMPLATE = "ppg_pulse_locked_template_subtraction"
CONTROL_BEAT_ADJUST = "beat_count_adjusted"
CONTROL_ECG_CHANNELS = "ecg_prone_channels_removed"

MASK_WINDOWS_S = {
    "ecg": (0.05, 0.05),
    "ppg": (0.15, 0.15),
}
PPG_TEMPLATE_HALF_WINDOW_S = 1.0
ECG_TEMPLATE_HALF_WINDOW_S = 1.0
MIN_RETAINED_COMMON_MONTAGE_CHANNELS = 8

OBS_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "participant_id",
    "session_id",
    "observation_id",
    "condition",
    "band",
    "control_type",
    "cardiac_signal_type",
    "cardiac_event_type",
    "source_channel",
    "detector_name",
    "detector_parameters",
    "event_mask_window_pre_s",
    "event_mask_window_post_s",
    "n_events_detected",
    "n_events_accepted",
    "n_events_rejected",
    "event_rate_per_min",
    "median_inter_event_interval_s",
    "alignment_status",
    "baseline_zlpi",
    "controlled_zlpi",
    "delta_vs_baseline",
    "n_valid_samples_baseline",
    "n_valid_samples_control",
    "n_channels_original",
    "n_channels_removed",
    "n_channels_retained",
    "removed_channels",
    "minimum_lag_overlap",
    "computable",
    "reason_code",
    "not_computable_reason",
    "not_computable_explanation",
    "pipeline_stage",
    "code_version",
)

DATASET_QC_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "protocol_declared_cardiac_modalities",
    "source_file_modalities_found",
    "selected_cardiac_signal_type",
    "selected_source_channel",
    "cardiac_event_type",
    "detector_name",
    "detector_parameters",
    "detector_execution_status",
    "n_observations",
    "n_observations_computable",
)

MASK_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "participant_id",
    "session_id",
    "observation_id",
    "condition",
    "band",
    "total_samples_before_mask",
    "masked_samples",
    "masked_fraction",
    "n_events_detected",
    "adjacent_mask_window_overlaps",
    "longest_uninterrupted_valid_segment_samples",
    "remaining_valid_samples",
    "valid_pairs_min_across_lags",
    "valid_pairs_median_across_lags",
    "valid_pairs_max_across_lags",
    "valid_pairs_lag0",
    "valid_pairs_lag_plus60",
    "valid_pairs_lag_minus60",
    "strict_common_support_anchors",
    "minimum_lag_overlap_required",
    "first_failing_rule",
    "aligned_sampling_hz",
)


@dataclass(frozen=True)
class CardiacControlResult:
    observation_rows: tuple[dict[str, object], ...]
    dataset_qc_rows: tuple[dict[str, object], ...]
    mask_diagnostic_rows: tuple[dict[str, object], ...]
    metadata: dict[str, object]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    text = _as_str(value)
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _as_bool(value: object) -> bool:
    return _as_str(value).casefold() in {"1", "true", "yes"}


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                else:
                    payload[field] = value
            writer.writerow(payload)


def _declared_modalities(protocol_rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in protocol_rows:
        dataset = _as_str(row.get("dataset_id")).casefold()
        if not dataset:
            continue
        out[dataset] = _as_str(row.get("cardiac_modality"), "unknown")
    return out


def _parse_modality_tokens(value: str) -> set[str]:
    text = _as_str(value).casefold()
    if not text:
        return set()
    tokens = {
        piece.strip()
        for raw in text.replace("/", ";").replace(",", ";").split(";")
        for piece in raw.split()
        if piece.strip()
    }
    out: set[str] = set()
    for token in tokens:
        if token in {"ecg", "ppg", "both"}:
            if token == "both":
                out.update({"ecg", "ppg"})
            else:
                out.add(token)
    return out


def _allowed_modalities_for_dataset(declared_modality: str) -> set[str]:
    allowed = _parse_modality_tokens(declared_modality)
    if allowed:
        return allowed
    return {"ecg", "ppg"}


def _group_aligned_d240(aligned_rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aligned_rows:
        if int(float(row.get("duration_s", 0) or 0)) != EXPECTED_PRIMARY_DURATION_S:
            continue
        out[_as_str(row.get("observation_id"))].append(dict(row))
    for key in out:
        out[key].sort(key=lambda item: _as_float(item.get("time_s")))
    return out


def _baseline_endpoint_lookup(endpoint_rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], Mapping[str, object]]:
    lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in endpoint_rows:
        if _as_str(row.get("endpoint_name")).casefold() != ENDPOINT_ZLPI:
            continue
        if int(float(row.get("duration_s", 0) or 0)) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if _as_str(row.get("power_representation")) != PRIMARY_POWER_REPRESENTATION:
            continue
        obs = _as_str(row.get("observation_id"))
        band = _as_str(row.get("band")).casefold()
        if not obs or not band:
            continue
        lookup[(obs, band)] = row
    return lookup


def _peaks_for_observation(c1b_dir: Path, observation_id: str) -> list[dict[str, str]]:
    obs_dir = c1b_dir / safe_subject_dir_name(observation_id)
    return _read_csv(obs_dir / "detected_peaks.csv")


def _accepted_event_times(peaks_rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    times: list[float] = []
    for row in peaks_rows:
        if _as_bool(row.get("is_accepted_peak")):
            value = _as_float(row.get("peak_time_s"))
            if math.isfinite(value):
                times.append(value)
    times.sort()
    return np.asarray(times, dtype=float)


def _alignment_status(
    event_times: np.ndarray,
    sample_times: np.ndarray,
) -> tuple[str, str]:
    if event_times.size == 0:
        return "fail", "no_accepted_events"
    if np.any(np.diff(event_times) <= 0):
        return "fail", "non_monotonic_or_duplicate_events"
    start = float(sample_times[0])
    end = float(sample_times[-1])
    in_bounds = event_times[(event_times >= start) & (event_times <= end)]
    if in_bounds.size == 0:
        return "fail", "all_events_outside_observation_bounds"
    rate = 60.0 * float(in_bounds.size) / max(end - start + 1e-9, 1e-9)
    if rate < 25.0 or rate > 240.0:
        return "fail", "implausible_event_rate"
    warning = []
    if in_bounds.size < 30:
        warning.append("few_events")
    if in_bounds.size != event_times.size:
        warning.append("out_of_bounds_events_dropped")
    if warning:
        return "warning", ";".join(warning)
    return "pass", ""


def _apply_event_mask(
    rows: Sequence[Mapping[str, object]],
    event_times: np.ndarray,
    *,
    pre_s: float,
    post_s: float,
) -> tuple[list[dict[str, object]], int]:
    if not rows:
        return [], 0
    masked: list[dict[str, object]] = []
    n_masked = 0
    sample_times = np.asarray([_as_float(r.get("time_s")) for r in rows], dtype=float)
    for row, time_s in zip(rows, sample_times, strict=True):
        item = dict(row)
        # Mark as invalid if sample sits in any event-centered window.
        left = np.searchsorted(event_times, time_s - post_s, side="left")
        right = np.searchsorted(event_times, time_s + pre_s, side="right")
        masked_now = right > left
        if masked_now:
            n_masked += 1
            for band in BAND_ORDER:
                key = f"{band}_absolute_log10_power_z"
                if key in item:
                    item[key] = float("nan")
        masked.append(item)
    return masked, n_masked


def _control_curves_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    result = compute_signed_lag_curves(
        rows,
        duration_s=EXPECTED_PRIMARY_DURATION_S,
        representations=(
            next(r for r in POWER_REPRESENTATIONS if r[0] == PRIMARY_POWER_REPRESENTATION),
        ),
    )
    endpoints = compute_endpoints_from_curves(
        result.curve_rows,
        duration_s=EXPECTED_PRIMARY_DURATION_S,
    )
    endpoint_by_band: dict[str, dict[str, object]] = {}
    for row in endpoints.metrics_rows:
        endpoint_by_band[_as_str(row.get("band")).casefold()] = dict(row)
    qc_by_band: dict[str, dict[str, object]] = {}
    for row in endpoints.qc_rows:
        qc_by_band[_as_str(row.get("band")).casefold()] = dict(row)
    return endpoint_by_band, qc_by_band


def _valid_pair_counts_by_lag(
    hr: np.ndarray,
    eeg_valid: np.ndarray,
    *,
    lag_min: int = -60,
    lag_max: int = 60,
) -> dict[int, int]:
    counts: dict[int, int] = {}
    n = int(min(hr.size, eeg_valid.size))
    valid_hr = np.isfinite(hr[:n])
    valid_eeg = np.asarray(eeg_valid[:n], dtype=bool)
    for tau in range(lag_min, lag_max + 1):
        count = 0
        for idx in range(n):
            j = idx + tau
            if j < 0 or j >= n:
                continue
            if valid_hr[idx] and valid_eeg[j]:
                count += 1
        counts[tau] = count
    return counts


def _strict_common_support_anchor_count(
    hr: np.ndarray,
    eeg_valid: np.ndarray,
    *,
    lag_max: int = 60,
) -> int:
    n = int(min(hr.size, eeg_valid.size))
    valid_hr = np.isfinite(hr[:n])
    valid_eeg = np.asarray(eeg_valid[:n], dtype=bool)
    anchors = 0
    for idx in range(lag_max, n - lag_max):
        if not valid_hr[idx]:
            continue
        window = valid_eeg[idx - lag_max : idx + lag_max + 1]
        if window.size == 2 * lag_max + 1 and np.all(window):
            anchors += 1
    return anchors


def _apply_event_locked_template_subtraction(
    rows: Sequence[Mapping[str, object]],
    event_times: np.ndarray,
    *,
    half_window_s: float,
) -> tuple[list[dict[str, object]], int, int]:
    """Subtract a simple pulse-locked template on aligned band-power series."""
    if not rows or event_times.size == 0:
        return [], 0, 0
    sample_times = np.asarray([_as_float(r.get("time_s")) for r in rows], dtype=float)
    nearest_event_idx = np.searchsorted(event_times, sample_times, side="left")
    nearest_dist = np.full(sample_times.shape, np.inf, dtype=float)
    nearest_delta = np.full(sample_times.shape, np.nan, dtype=float)
    for direction in (0, -1):
        idx = nearest_event_idx + direction
        valid = (idx >= 0) & (idx < event_times.size)
        if not np.any(valid):
            continue
        deltas = sample_times[valid] - event_times[idx[valid]]
        absd = np.abs(deltas)
        improve = absd < nearest_dist[valid]
        if np.any(improve):
            loc = np.flatnonzero(valid)[improve]
            nearest_dist[loc] = absd[improve]
            nearest_delta[loc] = deltas[improve]

    use_mask = np.isfinite(nearest_delta) & (nearest_dist <= half_window_s)
    if int(np.sum(use_mask)) < 20:
        return [], 0, int(event_times.size)
    bin_values = np.rint(nearest_delta[use_mask]).astype(int)
    valid_bins = np.isin(bin_values, np.asarray([-1, 0, 1], dtype=int))
    if not np.any(valid_bins):
        return [], 0, int(event_times.size)
    use_idx = np.flatnonzero(use_mask)[valid_bins]
    template_bins = np.rint(nearest_delta[use_idx]).astype(int)

    cleaned = [dict(r) for r in rows]
    for band in BAND_ORDER:
        key = f"{band}_absolute_log10_power_z"
        series = np.asarray([_as_float(r.get(key)) for r in rows], dtype=float)
        templates: dict[int, float] = {}
        for b in (-1, 0, 1):
            vals = series[use_idx[template_bins == b]]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            templates[b] = float(np.mean(vals))
        if len(templates) < 2:
            return [], 0, int(event_times.size)
        for idx_row in use_idx:
            b = int(round(nearest_delta[idx_row]))
            if b not in templates or not math.isfinite(series[idx_row]):
                continue
            cleaned[idx_row][key] = float(series[idx_row] - templates[b])

    used_events = int(len({int(np.argmin(np.abs(event_times - sample_times[i]))) for i in use_idx}))
    rejected_events = max(0, int(event_times.size) - used_events)
    return cleaned, used_events, rejected_events


def _detector_row_for_obs(c1b_qc_rows: Sequence[Mapping[str, object]], observation_id: str) -> Mapping[str, object]:
    for row in c1b_qc_rows:
        if _as_str(row.get("observation_id")) == observation_id:
            return row
    return {}


def _event_type_for_signal(signal_type: str) -> str:
    if signal_type == "ecg":
        return "r_peak"
    if signal_type == "ppg":
        return "ppg_systolic_peak"
    return "unknown"


def _signal_type_label(signal_type: str) -> str:
    if signal_type == "ecg":
        return "ECG"
    if signal_type == "ppg":
        return "PPG"
    return "UNKNOWN"


def _build_dataset_qc(
    obs_rows: Sequence[Mapping[str, object]],
    *,
    declared_modalities: Mapping[str, str],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in obs_rows:
        grouped[_as_str(row.get("dataset_id")).casefold()].append(row)

    rows: list[dict[str, object]] = []
    for dataset_id, members in sorted(grouped.items()):
        source_mods = sorted({_as_str(m.get("cardiac_signal_type")) for m in members if _as_str(m.get("cardiac_signal_type"))})
        selected = source_mods[0] if source_mods else "UNKNOWN"
        channels = sorted({_as_str(m.get("source_channel")) for m in members if _as_str(m.get("source_channel"))})
        events = sorted({_as_str(m.get("cardiac_event_type")) for m in members if _as_str(m.get("cardiac_event_type"))})
        detectors = sorted({_as_str(m.get("detector_name")) for m in members if _as_str(m.get("detector_name"))})
        computable = [m for m in members if _as_bool(m.get("computable"))]
        rows.append(
            {
                "dataset_id": dataset_id,
                "protocol_declared_cardiac_modalities": declared_modalities.get(dataset_id, "unknown"),
                "source_file_modalities_found": ";".join(source_mods),
                "selected_cardiac_signal_type": selected,
                "selected_source_channel": ";".join(channels),
                "cardiac_event_type": ";".join(events),
                "detector_name": ";".join(detectors),
                "detector_parameters": "see observation-level detector_parameters",
                "detector_execution_status": "ok" if computable else "not_computable",
                "n_observations": len({_as_str(m.get("observation_id")) for m in members}),
                "n_observations_computable": len({_as_str(m.get("observation_id")) for m in computable}),
            }
        )
    return rows


def _beat_count_adjust(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    # Observation-level statistical sensitivity: y ~ beat_count + mean_hr + event_rate.
    # We residualize y at the observation level within each band.
    out: list[dict[str, object]] = []
    by_band: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["control_type"] != CONTROL_BASELINE:
            continue
        by_band[_as_str(row.get("band")).casefold()].append(row)

    for band, members in by_band.items():
        y = np.asarray([_as_float(r.get("baseline_zlpi")) for r in members], dtype=float)
        beat = np.asarray([_as_float(r.get("n_events_accepted")) for r in members], dtype=float)
        mean_hr = np.asarray([_as_float(r.get("event_rate_per_min")) for r in members], dtype=float)
        event_rate = np.asarray([_as_float(r.get("event_rate_per_min")) for r in members], dtype=float)
        finite = np.isfinite(y) & np.isfinite(beat) & np.isfinite(mean_hr) & np.isfinite(event_rate)
        if int(np.sum(finite)) < 6:
            for row in members:
                item = dict(row)
                item["control_type"] = CONTROL_BEAT_ADJUST
                _mark_not_computable(
                    item,
                    reason_code=ARTIFACT_CONTROL_NOT_AVAILABLE,
                    explanation="insufficient rows for beat-count adjustment",
                )
                out.append(item)
            continue
        yv = y[finite]
        beatv = beat[finite] - float(np.mean(beat[finite]))
        hrv = mean_hr[finite] - float(np.mean(mean_hr[finite]))
        ratev = event_rate[finite] - float(np.mean(event_rate[finite]))
        x = np.column_stack((np.ones(yv.size), beatv, hrv, ratev))
        coef, *_ = np.linalg.lstsq(x, yv, rcond=None)
        fitted = x @ coef
        adj = coef[0] + (yv - fitted)
        adj_iter = iter(adj.tolist())
        for i, row in enumerate(members):
            item = dict(row)
            item["control_type"] = CONTROL_BEAT_ADJUST
            if finite[i]:
                val = float(next(adj_iter))
                item["controlled_zlpi"] = val
                item["delta_vs_baseline"] = val - _as_float(row.get("baseline_zlpi"))
                item["computable"] = True
                item["reason_code"] = ""
                item["not_computable_reason"] = ""
                item["not_computable_explanation"] = ""
            else:
                _mark_not_computable(
                    item,
                    reason_code=ARTIFACT_CONTROL_NOT_AVAILABLE,
                    explanation="missing covariates for beat-count adjustment",
                )
            out.append(item)
    return out


def _mark_not_computable(
    row: dict[str, object],
    *,
    reason_code: str,
    explanation: str,
) -> dict[str, object]:
    row.update(
        {
            "controlled_zlpi": "",
            "delta_vs_baseline": "",
            "n_valid_samples_control": "",
            "computable": False,
            "reason_code": reason_code,
            "not_computable_reason": explanation,
            "not_computable_explanation": explanation,
        }
    )
    return row


def _compute_ecg_channel_removal_control(
    *,
    observation_id: str,
    identity: Mapping[str, object],
    c1a_dir: Path,
    c1b_dir: Path,
    ecg_prone_channels: Sequence[str],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    channel_rows = compute_observation_channel_zlpi(
        observation_id=observation_id,
        c1a_dir=c1a_dir,
        c1b_dir=c1b_dir,
        identity=identity,
        bands=BAND_ORDER,
    )
    by_band: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in channel_rows:
        band = _as_str(row.get("band")).casefold()
        if band:
            by_band[band].append(row)
    prone_set = {_as_str(ch) for ch in ecg_prone_channels if _as_str(ch)}
    for band in BAND_ORDER:
        members = by_band.get(band, [])
        if not members:
            out[band] = {
                "computable": False,
                "reason_code": ARTIFACT_CONTROL_NOT_AVAILABLE,
                "reason": "channel-level ZLPI rows are unavailable for this observation",
            }
            continue
        finite = [
            m
            for m in members
            if _as_bool(m.get("computable")) and math.isfinite(_as_float(m.get("zpli_value")))
        ]
        if not finite:
            out[band] = {
                "computable": False,
                "reason_code": ARTIFACT_CONTROL_NOT_AVAILABLE,
                "reason": "no finite channel-level ZLPI values after spectral/HR QC",
            }
            continue
        original_channels = sorted({_as_str(m.get("channel")) for m in finite if _as_str(m.get("channel"))})
        retained = [m for m in finite if _as_str(m.get("channel")) not in prone_set]
        retained_channels = sorted({_as_str(m.get("channel")) for m in retained if _as_str(m.get("channel"))})
        removed_channels = sorted(ch for ch in original_channels if ch not in retained_channels)
        if len(retained_channels) < MIN_RETAINED_COMMON_MONTAGE_CHANNELS:
            out[band] = {
                "computable": False,
                "reason_code": INSUFFICIENT_COMMON_MONTAGE,
                "reason": (
                    "retained channel montage below minimum "
                    f"({len(retained_channels)} < {MIN_RETAINED_COMMON_MONTAGE_CHANNELS})"
                ),
                "n_channels_original": len(original_channels),
                "n_channels_removed": len(removed_channels),
                "n_channels_retained": len(retained_channels),
                "removed_channels": ";".join(removed_channels),
            }
            continue
        values = np.asarray([_as_float(m.get("zpli_value")) for m in retained], dtype=float)
        if values.size == 0 or not np.any(np.isfinite(values)):
            out[band] = {
                "computable": False,
                "reason_code": ARTIFACT_CONTROL_NOT_AVAILABLE,
                "reason": "retained channels have no finite ZLPI values",
                "n_channels_original": len(original_channels),
                "n_channels_removed": len(removed_channels),
                "n_channels_retained": len(retained_channels),
                "removed_channels": ";".join(removed_channels),
            }
            continue
        out[band] = {
            "computable": True,
            "controlled_zlpi": float(np.mean(values[np.isfinite(values)])),
            "n_channels_original": len(original_channels),
            "n_channels_removed": len(removed_channels),
            "n_channels_retained": len(retained_channels),
            "removed_channels": ";".join(removed_channels),
            "reason_code": "",
            "reason": "",
        }
    return out


def run_confirmatory_cardiac_controls_upstream(
    *,
    c0_dir: Path,
    c1a_dir: Path | None = None,
    c1b_dir: Path,
    c1c_dir: Path,
    c3_dir: Path,
    output_dir: Path,
    code_version: str = "cardiac_controls_upstream_v1",
) -> CardiacControlResult:
    protocol_rows = _read_csv(c0_dir / "protocol_audit.csv")
    declared = _declared_modalities(protocol_rows)
    aligned_rows = _read_csv(c1c_dir / "features_confirmatory_aligned_D240.csv")
    grouped_aligned = _group_aligned_d240(aligned_rows)
    endpoint_rows = _read_csv(c3_dir / "confirmatory_endpoint_metrics_D240.csv")
    baseline_lookup = _baseline_endpoint_lookup(endpoint_rows)
    c1b_qc_rows = _read_csv(c1b_dir / "cardiac_peak_qc.csv")
    try:
        ecg_prone_set_id, ecg_prone_channels = load_ecg_prone_channels()
    except Exception:
        ecg_prone_set_id, ecg_prone_channels = "ecg_prone_unavailable", ()

    observation_rows: list[dict[str, object]] = []
    mask_diagnostic_rows: list[dict[str, object]] = []

    for observation_id, obs_rows in sorted(grouped_aligned.items()):
        detector_row = _detector_row_for_obs(c1b_qc_rows, observation_id)
        signal_type = _as_str(detector_row.get("signal_type")).casefold()
        signal_label = _signal_type_label(signal_type)
        event_type = _event_type_for_signal(signal_type)
        source_channel = _as_str(detector_row.get("channel_used"))
        detector_name = _as_str(detector_row.get("detector_used"))
        detector_parameters = json.dumps(
            {
                "detector_polarity": _as_str(detector_row.get("detector_polarity")),
                "selection_reason": _as_str(detector_row.get("selection_reason")),
                "sampling_rate_hz": _as_float(detector_row.get("sfreq")),
            },
            sort_keys=True,
        )
        dataset_hint = _as_str(obs_rows[0].get("dataset_id")).casefold()
        declared_modality = _as_str(declared.get(dataset_hint)).casefold()
        allowed_modalities = _allowed_modalities_for_dataset(declared_modality)

        peaks_rows = _peaks_for_observation(c1b_dir, observation_id)
        event_times = _accepted_event_times(peaks_rows)
        sample_times = np.asarray([_as_float(r.get("time_s")) for r in obs_rows], dtype=float)
        align_status, align_reason = _alignment_status(event_times, sample_times)
        in_bounds_events = event_times[(event_times >= sample_times[0]) & (event_times <= sample_times[-1])] if sample_times.size else np.asarray([], dtype=float)
        n_detected = sum(1 for r in peaks_rows if math.isfinite(_as_float(r.get("peak_time_s"))))
        n_accepted = int(in_bounds_events.size)
        n_rejected = max(0, n_detected - n_accepted)
        span_s = max(sample_times[-1] - sample_times[0], 1.0) if sample_times.size else 1.0
        event_rate = 60.0 * n_accepted / span_s
        median_iei = float(np.median(np.diff(in_bounds_events))) if in_bounds_events.size >= 2 else float("nan")

        # Compute masked control endpoints once per observation.
        endpoints_by_control: dict[str, tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], int]] = {}
        template_event_stats: tuple[int, int] | None = None
        mask_valid_by_band: dict[str, np.ndarray] = {}
        channel_control_by_band: dict[str, dict[str, object]] = {}

        if (
            signal_type in {"ecg", "ppg"}
            and signal_type in allowed_modalities
            and in_bounds_events.size
            and align_status != "fail"
        ):
            pre_s, post_s = MASK_WINDOWS_S[signal_type]
            masked_rows, n_masked = _apply_event_mask(
                obs_rows,
                in_bounds_events,
                pre_s=pre_s,
                post_s=post_s,
            )
            try:
                endpoint_map, qc_map = _control_curves_from_rows(masked_rows)
                ctrl_name = CONTROL_ECG_MASK if signal_type == "ecg" else CONTROL_PPG_MASK
                endpoints_by_control[ctrl_name] = (endpoint_map, qc_map, n_masked)
            except Exception:
                pass
            for band in BAND_ORDER:
                key = f"{band}_absolute_log10_power_z"
                mask_valid_by_band[band] = np.asarray(
                    [math.isfinite(_as_float(row.get(key))) for row in masked_rows],
                    dtype=bool,
                )
            if signal_type == "ppg":
                cleaned_rows, n_used_template, n_rejected_template = _apply_event_locked_template_subtraction(
                    obs_rows,
                    in_bounds_events,
                    half_window_s=PPG_TEMPLATE_HALF_WINDOW_S,
                )
                if cleaned_rows:
                    try:
                        endpoint_map, qc_map = _control_curves_from_rows(cleaned_rows)
                        endpoints_by_control[CONTROL_PPG_TEMPLATE] = (endpoint_map, qc_map, 0)
                        template_event_stats = (n_used_template, n_rejected_template)
                    except Exception:
                        template_event_stats = None
            if signal_type == "ecg":
                cleaned_rows, n_used_template, n_rejected_template = _apply_event_locked_template_subtraction(
                    obs_rows,
                    in_bounds_events,
                    half_window_s=ECG_TEMPLATE_HALF_WINDOW_S,
                )
                if cleaned_rows:
                    try:
                        endpoint_map, qc_map = _control_curves_from_rows(cleaned_rows)
                        endpoints_by_control[CONTROL_ECG_TEMPLATE] = (endpoint_map, qc_map, 0)
                        template_event_stats = (n_used_template, n_rejected_template)
                    except Exception:
                        template_event_stats = None
        if c1a_dir is not None and signal_type == "ecg" and "ecg" in allowed_modalities:
            identity = {
                "dataset_id": _as_str(obs_rows[0].get("dataset_id")).casefold(),
                "participant_id": _as_str(obs_rows[0].get("subject_id")).casefold(),
                "subject_id": _as_str(obs_rows[0].get("subject_id")).casefold(),
                "session_id": "single",
                "condition": _as_str(obs_rows[0].get("condition")).casefold(),
            }
            channel_control_by_band = _compute_ecg_channel_removal_control(
                observation_id=observation_id,
                identity=identity,
                c1a_dir=c1a_dir,
                c1b_dir=c1b_dir,
                ecg_prone_channels=ecg_prone_channels,
            )

        # Emit per-band control rows.
        for band in BAND_ORDER:
            base = baseline_lookup.get((observation_id, band))
            dataset_id = _as_str((base or {}).get("dataset_id") or obs_rows[0].get("dataset_id")).casefold()
            participant_id = _as_str((base or {}).get("participant_id") or (base or {}).get("subject_id") or obs_rows[0].get("subject_id")).casefold()
            session_id = _as_str((base or {}).get("session_id"), "single").casefold()
            condition = _as_str((base or {}).get("condition") or obs_rows[0].get("condition")).casefold()
            baseline_z = _as_float((base or {}).get("endpoint_index"))
            n_common = _as_float((base or {}).get("n_common_support"))
            base_eligible = _as_bool((base or {}).get("eligible"))
            min_overlap = float(min_common_support_required(contract_for_duration(EXPECTED_PRIMARY_DURATION_S)))

            common = {
                "dataset_id": dataset_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "observation_id": observation_id,
                "condition": condition,
                "band": band,
                "cardiac_signal_type": signal_label,
                "cardiac_event_type": event_type,
                "source_channel": source_channel,
                "detector_name": detector_name,
                "detector_parameters": detector_parameters,
                "event_mask_window_pre_s": MASK_WINDOWS_S.get(signal_type, (float("nan"), float("nan")))[0],
                "event_mask_window_post_s": MASK_WINDOWS_S.get(signal_type, (float("nan"), float("nan")))[1],
                "n_events_detected": float(n_detected),
                "n_events_accepted": float(n_accepted),
                "n_events_rejected": float(n_rejected),
                "event_rate_per_min": float(event_rate),
                "median_inter_event_interval_s": median_iei,
                "alignment_status": align_status,
                "baseline_zlpi": baseline_z,
                "n_valid_samples_baseline": n_common,
                "n_channels_original": "",
                "n_channels_removed": "",
                "n_channels_retained": "",
                "removed_channels": "",
                "minimum_lag_overlap": min_overlap,
                "reason_code": "",
                "not_computable_explanation": "",
                "pipeline_stage": "C6",
                "code_version": code_version,
            }

            baseline_row = dict(common)
            baseline_row.update(
                {
                    "control_type": CONTROL_BASELINE,
                    "controlled_zlpi": baseline_z,
                    "delta_vs_baseline": 0.0,
                    "n_valid_samples_control": n_common,
                    "reason_code": "",
                    "not_computable_explanation": "",
                    "computable": bool(base_eligible and math.isfinite(baseline_z)),
                    "not_computable_reason": "" if (base_eligible and math.isfinite(baseline_z)) else "baseline endpoint not eligible",
                }
            )
            observation_rows.append(baseline_row)

            # Event mask control, with modality policy deciding which family applies.
            if allowed_modalities == {"ecg"}:
                ctrl_name = CONTROL_ECG_MASK
            elif allowed_modalities == {"ppg"}:
                ctrl_name = CONTROL_PPG_MASK
            else:
                ctrl_name = CONTROL_ECG_MASK if signal_type == "ecg" else CONTROL_PPG_MASK
            ctrl_row = dict(common)
            ctrl_row["control_type"] = ctrl_name
            if not base_eligible or not math.isfinite(baseline_z):
                _mark_not_computable(
                    ctrl_row,
                    reason_code=INSUFFICIENT_COMMON_SUPPORT,
                    explanation="baseline endpoint not eligible for paired comparison",
                )
            elif signal_type not in {"ecg", "ppg"}:
                _mark_not_computable(
                    ctrl_row,
                    reason_code=MISSING_REQUIRED_MODALITY,
                    explanation="no ECG/PPG detector row selected for this observation",
                )
            elif signal_type not in allowed_modalities:
                _mark_not_computable(
                    ctrl_row,
                    reason_code=UNSUPPORTED_CONTROL_FOR_MODALITY,
                    explanation=(
                        f"declared dataset modality {declared_modality!r} forbids {signal_type.upper()} "
                        "fallback for event-mask control"
                    ),
                )
            elif align_status == "fail":
                _mark_not_computable(
                    ctrl_row,
                    reason_code=MISSING_EVENT_SERIES,
                    explanation=f"event alignment failed: {align_reason}",
                )
            elif ctrl_name not in endpoints_by_control:
                _mark_not_computable(
                    ctrl_row,
                    reason_code=ARTIFACT_CONTROL_NOT_AVAILABLE,
                    explanation="event-mask control could not be computed from available aligned/event series",
                )
            else:
                endpoint_map, _qc_map, _n_masked = endpoints_by_control[ctrl_name]
                endpoint_row = endpoint_map.get(band)
                controlled = _as_float((endpoint_row or {}).get("endpoint_index"))
                n_valid_control = _as_float((endpoint_row or {}).get("n_common_support"))
                eligible = _as_bool((endpoint_row or {}).get("eligible"))
                reason_code = ""
                reason_text = ""
                # Keep the structural PPG-mask NC explicit on 1 Hz locked support.
                if (not eligible) and band in mask_valid_by_band:
                    hr = np.asarray([_as_float(r.get("hr_z")) for r in obs_rows], dtype=float)
                    eeg_valid = mask_valid_by_band[band]
                    lag_counts = _valid_pair_counts_by_lag(hr, eeg_valid)
                    anchors = _strict_common_support_anchor_count(hr, eeg_valid)
                    aligned_fs = float("nan")
                    if sample_times.size >= 2:
                        deltas = np.diff(sample_times)
                        if deltas.size and np.all(np.isfinite(deltas)) and float(np.median(deltas)) > 0:
                            aligned_fs = 1.0 / float(np.median(deltas))
                    pre_s = float(common["event_mask_window_pre_s"])
                    post_s = float(common["event_mask_window_post_s"])
                    subsecond_mask_on_1hz = (
                        math.isfinite(aligned_fs)
                        and abs(aligned_fs - 1.0) < 0.05
                        and math.isfinite(pre_s)
                        and math.isfinite(post_s)
                        and (pre_s + post_s) < 1.0
                    )
                    if (
                        subsecond_mask_on_1hz
                        and anchors < int(min_overlap)
                        and lag_counts
                        and min(lag_counts.values()) >= int(min_overlap)
                    ):
                        reason_code = INSUFFICIENT_COMMON_SUPPORT
                        reason_text = (
                            "1Hz grid + sub-second event-mask windows destroy strict common-support "
                            "anchors required by locked D240 ZLPI"
                        )
                if not reason_code and (not eligible or not math.isfinite(controlled)):
                    reason_code = INSUFFICIENT_COMMON_SUPPORT
                    reason_text = _as_str((endpoint_row or {}).get("exclusion_reason"), "insufficient overlap after masking")
                if eligible and math.isfinite(controlled):
                    ctrl_row.update(
                        {
                            "controlled_zlpi": controlled,
                            "delta_vs_baseline": controlled - baseline_z,
                            "n_valid_samples_control": n_valid_control,
                            "computable": True,
                            "reason_code": "",
                            "not_computable_reason": "",
                            "not_computable_explanation": "",
                        }
                    )
                else:
                    _mark_not_computable(
                        ctrl_row,
                        reason_code=reason_code or INSUFFICIENT_COMMON_SUPPORT,
                        explanation=reason_text or "insufficient support after event masking",
                    )
            observation_rows.append(ctrl_row)

            # Event-locked template subtraction (PPG or ECG per modality policy).
            if allowed_modalities == {"ecg"}:
                template_control = CONTROL_ECG_TEMPLATE
            elif allowed_modalities == {"ppg"}:
                template_control = CONTROL_PPG_TEMPLATE
            else:
                template_control = CONTROL_ECG_TEMPLATE if signal_type == "ecg" else CONTROL_PPG_TEMPLATE
            tmpl_row = dict(common)
            tmpl_row["control_type"] = template_control
            if not base_eligible or not math.isfinite(baseline_z):
                _mark_not_computable(
                    tmpl_row,
                    reason_code=INSUFFICIENT_COMMON_SUPPORT,
                    explanation="baseline endpoint not eligible for paired comparison",
                )
            elif signal_type not in {"ecg", "ppg"}:
                _mark_not_computable(
                    tmpl_row,
                    reason_code=MISSING_REQUIRED_MODALITY,
                    explanation="no ECG/PPG detector row selected for this observation",
                )
            elif signal_type not in allowed_modalities:
                _mark_not_computable(
                    tmpl_row,
                    reason_code=UNSUPPORTED_CONTROL_FOR_MODALITY,
                    explanation=(
                        f"declared dataset modality {declared_modality!r} forbids {signal_type.upper()} "
                        "fallback for template subtraction control"
                    ),
                )
            elif template_control not in endpoints_by_control:
                _mark_not_computable(
                    tmpl_row,
                    reason_code=ARTIFACT_CONTROL_NOT_AVAILABLE,
                    explanation="event-locked template subtraction unavailable from current aligned/event inputs",
                )
            else:
                endpoint_map, _qc_map, _ = endpoints_by_control[template_control]
                endpoint_row = endpoint_map.get(band)
                controlled = _as_float((endpoint_row or {}).get("endpoint_index"))
                n_valid_control = _as_float((endpoint_row or {}).get("n_common_support"))
                eligible = _as_bool((endpoint_row or {}).get("eligible"))
                if eligible and math.isfinite(controlled):
                    tmpl_row.update(
                        {
                            "n_events_accepted": float(template_event_stats[0]) if template_event_stats else float(n_accepted),
                            "n_events_rejected": float(template_event_stats[1]) if template_event_stats else float(n_rejected),
                            "controlled_zlpi": controlled,
                            "delta_vs_baseline": controlled - baseline_z,
                            "n_valid_samples_control": n_valid_control,
                            "computable": True,
                            "reason_code": "",
                            "not_computable_reason": "",
                            "not_computable_explanation": "",
                        }
                    )
                else:
                    _mark_not_computable(
                        tmpl_row,
                        reason_code=INSUFFICIENT_COMMON_SUPPORT,
                        explanation=_as_str((endpoint_row or {}).get("exclusion_reason"), "insufficient template support"),
                    )
            observation_rows.append(tmpl_row)

            # ECG-prone channel removal control from channel-level D240 ZLPI re-aggregation.
            ch_row = dict(common)
            ch_row["control_type"] = CONTROL_ECG_CHANNELS
            if not base_eligible or not math.isfinite(baseline_z):
                _mark_not_computable(
                    ch_row,
                    reason_code=INSUFFICIENT_COMMON_SUPPORT,
                    explanation="baseline endpoint not eligible for paired comparison",
                )
            elif "ecg" not in allowed_modalities:
                _mark_not_computable(
                    ch_row,
                    reason_code=UNSUPPORTED_CONTROL_FOR_MODALITY,
                    explanation=f"declared dataset modality {declared_modality!r} does not define ECG-prone channel control",
                )
            elif signal_type != "ecg":
                _mark_not_computable(
                    ch_row,
                    reason_code=MISSING_REQUIRED_MODALITY,
                    explanation="ECG-prone channel control requires ECG-selected observation events",
                )
            elif c1a_dir is None:
                _mark_not_computable(
                    ch_row,
                    reason_code=ARTIFACT_CONTROL_NOT_AVAILABLE,
                    explanation="C1a channel-level features directory not provided",
                )
            else:
                channel_payload = channel_control_by_band.get(band, {})
                if _as_bool(channel_payload.get("computable")):
                    controlled = _as_float(channel_payload.get("controlled_zlpi"))
                    ch_row.update(
                        {
                            "controlled_zlpi": controlled,
                            "delta_vs_baseline": controlled - baseline_z,
                            "n_valid_samples_control": n_common,
                            "n_channels_original": int(channel_payload.get("n_channels_original", 0)),
                            "n_channels_removed": int(channel_payload.get("n_channels_removed", 0)),
                            "n_channels_retained": int(channel_payload.get("n_channels_retained", 0)),
                            "removed_channels": _as_str(channel_payload.get("removed_channels")),
                            "computable": True,
                            "reason_code": "",
                            "not_computable_reason": "",
                            "not_computable_explanation": "",
                        }
                    )
                else:
                    ch_row.update(
                        {
                            "n_channels_original": int(channel_payload.get("n_channels_original", 0) or 0),
                            "n_channels_removed": int(channel_payload.get("n_channels_removed", 0) or 0),
                            "n_channels_retained": int(channel_payload.get("n_channels_retained", 0) or 0),
                            "removed_channels": _as_str(channel_payload.get("removed_channels")),
                        }
                    )
                    _mark_not_computable(
                        ch_row,
                        reason_code=_as_str(channel_payload.get("reason_code"), ARTIFACT_CONTROL_NOT_AVAILABLE),
                        explanation=_as_str(
                            channel_payload.get("reason"),
                            "channel-level re-aggregation unavailable for ECG-prone channel exclusion control",
                        ),
                    )
            observation_rows.append(ch_row)

            # Diagnostic row for this observation-band masked control path.
            if signal_type in {"ecg", "ppg"} and in_bounds_events.size and band in mask_valid_by_band:
                hr_series = np.asarray([_as_float(r.get("hr_z")) for r in obs_rows], dtype=float)
                eeg_valid = mask_valid_by_band[band]
                lag_counts = _valid_pair_counts_by_lag(hr_series, eeg_valid)
                anchors = _strict_common_support_anchor_count(hr_series, eeg_valid)
                # Longest uninterrupted valid segment for masked EEG series.
                longest = 0
                cur = 0
                for v in eeg_valid:
                    if v:
                        cur += 1
                        longest = max(longest, cur)
                    else:
                        cur = 0
                # Overlaps in adjacent event windows in event-time domain.
                overlaps = 0
                if in_bounds_events.size >= 2:
                    gaps = np.diff(in_bounds_events)
                    overlaps = int(np.sum(gaps < (common["event_mask_window_pre_s"] + common["event_mask_window_post_s"])))
                aligned_fs = float("nan")
                if sample_times.size >= 2:
                    deltas = np.diff(sample_times)
                    if deltas.size and np.all(np.isfinite(deltas)) and float(np.median(deltas)) > 0:
                        aligned_fs = 1.0 / float(np.median(deltas))
                first_rule = "none"
                if anchors < min_overlap:
                    pre_s = float(common["event_mask_window_pre_s"])
                    post_s = float(common["event_mask_window_post_s"])
                    if (
                        math.isfinite(aligned_fs)
                        and abs(aligned_fs - 1.0) < 0.05
                        and math.isfinite(pre_s)
                        and math.isfinite(post_s)
                        and (pre_s + post_s) < 1.0
                        and lag_counts
                        and min(lag_counts.values()) >= int(min_overlap)
                    ):
                        first_rule = (
                            "1hz_envelope_cannot_support_event_centered_masking_"
                            "for_strict_common_support_zlpi"
                        )
                    else:
                        first_rule = "insufficient_common_support"
                mask_diagnostic_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "observation_id": observation_id,
                        "condition": condition,
                        "band": band,
                        "total_samples_before_mask": int(sample_times.size),
                        "masked_samples": int(sample_times.size - int(np.sum(eeg_valid))),
                        "masked_fraction": float((sample_times.size - int(np.sum(eeg_valid))) / max(int(sample_times.size), 1)),
                        "n_events_detected": float(n_accepted),
                        "adjacent_mask_window_overlaps": overlaps,
                        "longest_uninterrupted_valid_segment_samples": longest,
                        "remaining_valid_samples": int(np.sum(eeg_valid)),
                        "valid_pairs_min_across_lags": int(min(lag_counts.values())),
                        "valid_pairs_median_across_lags": float(np.median(np.asarray(list(lag_counts.values()), dtype=float))),
                        "valid_pairs_max_across_lags": int(max(lag_counts.values())),
                        "valid_pairs_lag0": int(lag_counts.get(0, 0)),
                        "valid_pairs_lag_plus60": int(lag_counts.get(60, 0)),
                        "valid_pairs_lag_minus60": int(lag_counts.get(-60, 0)),
                        "strict_common_support_anchors": anchors,
                        "minimum_lag_overlap_required": min_overlap,
                        "first_failing_rule": first_rule,
                        "aligned_sampling_hz": aligned_fs,
                    }
                )

    # Statistical beat-count adjustment on baseline outcomes.
    observation_rows.extend(_beat_count_adjust(observation_rows))

    dataset_qc = _build_dataset_qc(observation_rows, declared_modalities=declared)

    metadata = {
        "schema_version": "panel_d_cardiac_controls_upstream_v1",
        "duration_s": EXPECTED_PRIMARY_DURATION_S,
        "endpoint_name": ENDPOINT_ZLPI,
        "power_representation": PRIMARY_POWER_REPRESENTATION,
        "lag_range_s": [-60, 60],
        "flanks_s": [20, 60],
        "mask_windows_s": {
            "ecg_r_peak_mask": {"pre": MASK_WINDOWS_S["ecg"][0], "post": MASK_WINDOWS_S["ecg"][1]},
            "ppg_systolic_peak_mask": {"pre": MASK_WINDOWS_S["ppg"][0], "post": MASK_WINDOWS_S["ppg"][1]},
        },
        "ecg_prone_channel_set_id": ecg_prone_set_id,
        "ecg_prone_channels": list(ecg_prone_channels),
        "minimum_retained_common_montage_channels": MIN_RETAINED_COMMON_MONTAGE_CHANNELS,
        "alignment_status_levels": ["pass", "warning", "fail"],
        "event_semantics": {
            "ECG": "r_peak",
            "PPG": "ppg_systolic_peak",
        },
        "modality_policy_note": (
            "Dataset-declared cardiac modality gates control eligibility. ECG-declared datasets "
            "do not silently substitute PPG-derived events for ECG controls."
        ),
        "ppg_interpretation_note": (
            "PPG pulse-event controls assess pulse-synchronous contamination and do not "
            "directly test ECG electrical-field leakage."
        ),
        "event_mask_computability_note": (
            "Locked D240 ZLPI uses strict common-support anchors on the 1 Hz analysis grid "
            "(≥60 anchors with continuous ±60 s EEG validity). Sub-second pulse-event masks "
            f"(PPG ±{MASK_WINDOWS_S['ppg'][0]:g} s) punch periodic holes (~1 event / IBI), "
            "so anchors are structurally zero even when lag-wise pairwise overlaps remain "
            "≥60. This is reported as not computable; it is not relabeled computable and does "
            "not relax the estimand or duration gate. PPG template subtraction remains "
            "computable because it preserves finite samples."
        ),
        "code_version": code_version,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / OBSERVATION_CONTROLS_FILENAME, observation_rows, OBS_COLUMNS)
    _write_csv(output_dir / DATASET_QC_FILENAME, dataset_qc, DATASET_QC_COLUMNS)
    _write_csv(output_dir / MASK_DIAGNOSTICS_FILENAME, mask_diagnostic_rows, MASK_DIAGNOSTIC_COLUMNS)
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return CardiacControlResult(
        observation_rows=tuple(observation_rows),
        dataset_qc_rows=tuple(dataset_qc),
        mask_diagnostic_rows=tuple(mask_diagnostic_rows),
        metadata=metadata,
    )


__all__ = [
    "CONTROL_BASELINE",
    "CONTROL_BEAT_ADJUST",
    "CONTROL_ECG_CHANNELS",
    "CONTROL_ECG_MASK",
    "CONTROL_ECG_TEMPLATE",
    "CONTROL_PPG_MASK",
    "CONTROL_PPG_TEMPLATE",
    "DATASET_QC_COLUMNS",
    "DATASET_QC_FILENAME",
    "METADATA_FILENAME",
    "MASK_DIAGNOSTIC_COLUMNS",
    "MASK_DIAGNOSTICS_FILENAME",
    "OBS_COLUMNS",
    "OBSERVATION_CONTROLS_FILENAME",
    "CardiacControlResult",
    "run_confirmatory_cardiac_controls_upstream",
]
