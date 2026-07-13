"""Align instantaneous HR and multitaper EEG onto nested confirmatory segments."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import EXPECTED_DURATIONS_S, EXPECTED_PRIMARY_DURATION_S
from .multitaper_power import BANDS_HZ, ROBUST_MEDIAN_CHANNEL

MAX_GAP_S = 5.0
NESTED_DURATIONS_S = (240, 180, 120)
SENSITIVITY_DURATION_S = 60
ALIGNED_FEATURES_TEMPLATE = "features_confirmatory_aligned_D{duration_s}.csv"
SEGMENT_MANIFEST_FILENAME = "segment_manifest.json"
ALIGNMENT_QC_FILENAME = "alignment_qc_confirmatory.csv"
BAND_ORDER = tuple(BANDS_HZ)


@dataclass(frozen=True)
class ContiguousBlock:
    start_s: float
    end_s: float
    time_s: tuple[float, ...]

    @property
    def n_samples(self) -> int:
        return len(self.time_s)

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s + 1.0)


@dataclass(frozen=True)
class DurationSegment:
    duration_s: int
    role: str
    start_s: float
    end_s: float
    center_s: float
    time_s: tuple[float, ...]
    eligible: bool
    exclusion_reason: str

    @property
    def n_samples(self) -> int:
        return len(self.time_s)


@dataclass(frozen=True)
class HarmonizeResult:
    identity: dict[str, str]
    selected_block: ContiguousBlock | None
    center_s: float | None
    segments: dict[int, DurationSegment]
    features_by_duration: dict[int, tuple[dict[str, object], ...]]
    qc_rows: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return array


def contiguous_blocks(
    time_s: Sequence[float] | np.ndarray,
    *,
    max_gap_s: float = MAX_GAP_S,
) -> list[ContiguousBlock]:
    """Split sorted common-support times wherever consecutive gaps exceed max_gap_s."""
    times = _as_float_array(time_s)
    if times.size == 0:
        return []
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be unique.")
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive.")

    blocks: list[ContiguousBlock] = []
    start_index = 0
    for index in range(1, times.size):
        if times[index] - times[index - 1] > max_gap_s:
            chunk = tuple(float(value) for value in times[start_index:index])
            blocks.append(
                ContiguousBlock(
                    start_s=chunk[0],
                    end_s=chunk[-1],
                    time_s=chunk,
                )
            )
            start_index = index
    chunk = tuple(float(value) for value in times[start_index:])
    blocks.append(
        ContiguousBlock(start_s=chunk[0], end_s=chunk[-1], time_s=chunk)
    )
    return blocks


def select_longest_block(blocks: Sequence[ContiguousBlock]) -> ContiguousBlock | None:
    """Prefer maximum sample count; break ties by earliest start."""
    if not blocks:
        return None
    return min(
        blocks,
        key=lambda block: (-block.n_samples, block.start_s),
    )


def place_centered_segment(
    block: ContiguousBlock,
    duration_s: int,
    *,
    center_s: float | None = None,
) -> DurationSegment:
    """Place a fixed-length 1-Hz segment at the shared or block-local center."""
    role = (
        "primary"
        if duration_s == EXPECTED_PRIMARY_DURATION_S
        else (
            "nested_sensitivity"
            if duration_s in NESTED_DURATIONS_S
            else "separate_sensitivity"
        )
    )
    if duration_s not in EXPECTED_DURATIONS_S:
        raise ValueError(f"Unsupported duration_s={duration_s}.")
    if block.n_samples < duration_s:
        return DurationSegment(
            duration_s=duration_s,
            role=role,
            start_s=float("nan"),
            end_s=float("nan"),
            center_s=float("nan") if center_s is None else float(center_s),
            time_s=(),
            eligible=False,
            exclusion_reason="insufficient_clean_support",
        )

    times = np.asarray(block.time_s, dtype=float)
    if center_s is None:
        leftover = block.n_samples - duration_s
        start_index = leftover // 2  # odd leftover goes to the right (earlier start)
        end_index = start_index + duration_s - 1
        selected = times[start_index : end_index + 1]
        local_center = float(selected[0] + selected[-1]) / 2.0
        return DurationSegment(
            duration_s=duration_s,
            role=role,
            start_s=float(selected[0]),
            end_s=float(selected[-1]),
            center_s=local_center,
            time_s=tuple(float(value) for value in selected),
            eligible=True,
            exclusion_reason="",
        )

    # Shared center: choose integer indices whose midpoint is as close as possible
    # and never extends outside the block. Prefer earlier start on residual ties.
    half = duration_s / 2.0
    ideal_start = center_s - half + 0.5
    candidates: list[tuple[float, float, int]] = []
    for start_index in range(0, block.n_samples - duration_s + 1):
        end_index = start_index + duration_s - 1
        seg_center = float(times[start_index] + times[end_index]) / 2.0
        candidates.append(
            (abs(seg_center - center_s), times[start_index], start_index)
        )
    _distance, _start, start_index = min(candidates)
    selected = times[start_index : start_index + duration_s]
    return DurationSegment(
        duration_s=duration_s,
        role=role,
        start_s=float(selected[0]),
        end_s=float(selected[-1]),
        center_s=float(selected[0] + selected[-1]) / 2.0,
        time_s=tuple(float(value) for value in selected),
        eligible=True,
        exclusion_reason="",
    )


def build_duration_segments(
    block: ContiguousBlock | None,
) -> tuple[float | None, dict[int, DurationSegment]]:
    """Create nested 240/180/120 segments and a separate 60-s sensitivity segment."""
    if block is None:
        return None, {
            duration: DurationSegment(
                duration_s=duration,
                role=(
                    "primary"
                    if duration == EXPECTED_PRIMARY_DURATION_S
                    else (
                        "nested_sensitivity"
                        if duration in NESTED_DURATIONS_S
                        else "separate_sensitivity"
                    )
                ),
                start_s=float("nan"),
                end_s=float("nan"),
                center_s=float("nan"),
                time_s=(),
                eligible=False,
                exclusion_reason="no_common_support",
            )
            for duration in EXPECTED_DURATIONS_S
        }

    nested_available = [d for d in NESTED_DURATIONS_S if block.n_samples >= d]
    nested_center: float | None = None
    segments: dict[int, DurationSegment] = {}
    if nested_available:
        primary_nested = nested_available[0]
        anchor = place_centered_segment(block, primary_nested, center_s=None)
        nested_center = anchor.center_s
        for duration in NESTED_DURATIONS_S:
            segments[duration] = place_centered_segment(
                block, duration, center_s=nested_center
            )
        segments[SENSITIVITY_DURATION_S] = place_centered_segment(
            block, SENSITIVITY_DURATION_S, center_s=nested_center
        )
    else:
        for duration in NESTED_DURATIONS_S:
            segments[duration] = place_centered_segment(block, duration)
        # D60 remains available as its own sensitivity placement when nested
        # family support is absent.
        segments[SENSITIVITY_DURATION_S] = place_centered_segment(
            block, SENSITIVITY_DURATION_S, center_s=None
        )
        if segments[SENSITIVITY_DURATION_S].eligible:
            nested_center = segments[SENSITIVITY_DURATION_S].center_s
    return nested_center, segments


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if not np.any(finite):
        return result
    subset = values[finite]
    std = float(np.std(subset, ddof=0))
    mean = float(np.mean(subset))
    if std <= 0 or not math.isfinite(std):
        result[finite] = 0.0
        return result
    result[finite] = (subset - mean) / std
    return result


def _relative_and_residualized(
    absolute_power: Mapping[str, np.ndarray],
    absolute_log10_power: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    stack = np.vstack([absolute_power[band] for band in BAND_ORDER])
    totals = np.sum(stack, axis=0)
    relative = {
        band: np.divide(
            absolute_power[band],
            totals,
            out=np.full(totals.shape, np.nan, dtype=float),
            where=np.isfinite(totals) & (totals > 0),
        )
        for band in BAND_ORDER
    }

    log_stack = np.vstack([absolute_log10_power[band] for band in BAND_ORDER])
    broadband = np.nanmean(log_stack, axis=0)
    residualized: dict[str, np.ndarray] = {}
    finite = np.isfinite(broadband)
    for band in BAND_ORDER:
        y = absolute_log10_power[band]
        residuals = np.full(y.shape, np.nan, dtype=float)
        both = finite & np.isfinite(y)
        if int(np.sum(both)) >= 2:
            x = broadband[both]
            yy = y[both]
            x_design = np.column_stack((np.ones(x.size), x))
            coef, *_ = np.linalg.lstsq(x_design, yy, rcond=None)
            residuals[both] = yy - x_design @ coef
        elif int(np.sum(both)) == 1:
            residuals[both] = 0.0
        residualized[band] = residuals
    return relative, residualized


def intersect_valid_times(
    hr_time_s: Sequence[float] | np.ndarray,
    hr_valid: Sequence[bool] | np.ndarray,
    eeg_time_s: Sequence[float] | np.ndarray,
    eeg_valid: Sequence[bool] | np.ndarray,
) -> np.ndarray:
    """Return sorted times that are marked valid independently in both series."""
    hr_times = _as_float_array(hr_time_s)
    eeg_times = _as_float_array(eeg_time_s)
    hr_mask = np.asarray(hr_valid, dtype=bool)
    eeg_mask = np.asarray(eeg_valid, dtype=bool)
    if hr_times.shape != hr_mask.shape or eeg_times.shape != eeg_mask.shape:
        raise ValueError("Validity masks must match their time vectors.")
    hr_set = set(np.round(hr_times[hr_mask], 6).tolist())
    eeg_set = set(np.round(eeg_times[eeg_mask], 6).tolist())
    common = sorted(hr_set.intersection(eeg_set))
    return np.asarray(common, dtype=float)


def _lookup_series(
    time_s: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    source_times = _as_float_array(time_s)
    source_values = _as_float_array(values)
    if source_times.size != source_values.size:
        raise ValueError("time_s and values must have equal length.")
    order = np.argsort(source_times, kind="mergesort")
    source_times = source_times[order]
    source_values = source_values[order]
    indexed = {
        round(float(time_value), 6): float(value)
        for time_value, value in zip(source_times, source_values, strict=True)
    }
    return np.asarray(
        [indexed.get(round(float(time_value), 6), float("nan")) for time_value in query_times],
        dtype=float,
    )


def _identity(values: Mapping[str, object] | None) -> dict[str, str]:
    payload = values or {}
    return {
        "dataset_id": str(payload.get("dataset_id", "")).strip().casefold(),
        "subject_id": str(payload.get("subject_id", "")).strip().casefold(),
        "task": str(payload.get("task", "")).strip().casefold(),
        "condition": str(payload.get("condition", "")).strip().casefold(),
        "observation_id": str(payload.get("observation_id", "")).strip(),
    }


def harmonize_observation(
    *,
    hr_time_s: Sequence[float] | np.ndarray,
    hr_bpm: Sequence[float] | np.ndarray,
    hr_valid: Sequence[bool] | np.ndarray,
    eeg_time_s: Sequence[float] | np.ndarray,
    eeg_absolute_power: Mapping[str, Sequence[float] | np.ndarray],
    eeg_absolute_log10_power: Mapping[str, Sequence[float] | np.ndarray],
    eeg_valid: Sequence[bool] | np.ndarray | None = None,
    identity: Mapping[str, object] | None = None,
    max_gap_s: float = MAX_GAP_S,
) -> HarmonizeResult:
    """Harmonize one observation onto nested confirmatory duration segments."""
    ids = _identity(identity)
    missing_bands = [band for band in BAND_ORDER if band not in eeg_absolute_power]
    missing_log = [
        band for band in BAND_ORDER if band not in eeg_absolute_log10_power
    ]
    if missing_bands or missing_log:
        raise ValueError(
            "eeg power mappings must include all confirmatory bands "
            f"{BAND_ORDER}; missing={sorted(set(missing_bands + missing_log))}."
        )

    eeg_times = _as_float_array(eeg_time_s)
    if eeg_valid is None:
        stacked = np.vstack(
            [
                _as_float_array(eeg_absolute_log10_power[band])
                for band in BAND_ORDER
            ]
        )
        eeg_mask = np.all(np.isfinite(stacked), axis=0)
    else:
        eeg_mask = np.asarray(eeg_valid, dtype=bool)

    common_times = intersect_valid_times(
        hr_time_s, hr_valid, eeg_times, eeg_mask
    )
    n_hr_valid = int(np.sum(np.asarray(hr_valid, dtype=bool)))
    n_eeg_valid = int(np.sum(eeg_mask))
    n_common = int(common_times.size)
    blocks = contiguous_blocks(common_times, max_gap_s=max_gap_s)
    selected = select_longest_block(blocks)
    center_s, segments = build_duration_segments(selected)

    features_by_duration: dict[int, tuple[dict[str, object], ...]] = {}
    qc_rows: list[dict[str, object]] = []
    for duration in EXPECTED_DURATIONS_S:
        segment = segments[duration]
        if not segment.eligible:
            features_by_duration[duration] = ()
            qc_rows.append(
                {
                    **ids,
                    "duration_s": duration,
                    "duration_role": segment.role,
                    "eligible": False,
                    "exclusion_reason": segment.exclusion_reason,
                    "selected_block_start_s": (
                        selected.start_s if selected is not None else None
                    ),
                    "selected_block_end_s": (
                        selected.end_s if selected is not None else None
                    ),
                    "selected_block_n_samples": (
                        selected.n_samples if selected is not None else 0
                    ),
                    "n_hr_valid": n_hr_valid,
                    "n_eeg_valid": n_eeg_valid,
                    "n_common_support": n_common,
                    "n_blocks": len(blocks),
                    "segment_start_s": None,
                    "segment_end_s": None,
                    "segment_center_s": center_s,
                    "segment_n_samples": 0,
                    "n_missing_within_segment": 0,
                    "available_support_s": (
                        selected.duration_s if selected is not None else 0.0
                    ),
                }
            )
            continue

        query = np.asarray(segment.time_s, dtype=float)
        hr_seg = _lookup_series(hr_time_s, hr_bpm, query)
        absolute = {
            band: _lookup_series(eeg_time_s, eeg_absolute_power[band], query)
            for band in BAND_ORDER
        }
        abs_log = {
            band: _lookup_series(
                eeg_time_s, eeg_absolute_log10_power[band], query
            )
            for band in BAND_ORDER
        }
        relative, residualized = _relative_and_residualized(absolute, abs_log)
        hr_z = _zscore(hr_seg)
        abs_log_z = {band: _zscore(abs_log[band]) for band in BAND_ORDER}
        relative_z = {band: _zscore(relative[band]) for band in BAND_ORDER}
        residual_z = {band: _zscore(residualized[band]) for band in BAND_ORDER}

        rows: list[dict[str, object]] = []
        for index, time_value in enumerate(query):
            row: dict[str, object] = {
                **ids,
                "duration_s": duration,
                "duration_role": segment.role,
                "time_s": float(time_value),
                "hr_bpm": float(hr_seg[index]),
                "hr_z": float(hr_z[index]),
            }
            for band in BAND_ORDER:
                row[f"{band}_absolute_power"] = float(absolute[band][index])
                row[f"{band}_absolute_log10_power"] = float(abs_log[band][index])
                row[f"{band}_relative_power"] = float(relative[band][index])
                row[f"{band}_broadband_residualized_log10"] = float(
                    residualized[band][index]
                )
                row[f"{band}_absolute_log10_power_z"] = float(
                    abs_log_z[band][index]
                )
                row[f"{band}_relative_power_z"] = float(relative_z[band][index])
                row[f"{band}_broadband_residualized_log10_z"] = float(
                    residual_z[band][index]
                )
            rows.append(row)
        features_by_duration[duration] = tuple(rows)

        expected_count = duration
        missing_within = expected_count - len(rows)
        qc_rows.append(
            {
                **ids,
                "duration_s": duration,
                "duration_role": segment.role,
                "eligible": True,
                "exclusion_reason": "",
                "selected_block_start_s": selected.start_s if selected else None,
                "selected_block_end_s": selected.end_s if selected else None,
                "selected_block_n_samples": (
                    selected.n_samples if selected else 0
                ),
                "n_hr_valid": n_hr_valid,
                "n_eeg_valid": n_eeg_valid,
                "n_common_support": n_common,
                "n_blocks": len(blocks),
                "segment_start_s": segment.start_s,
                "segment_end_s": segment.end_s,
                "segment_center_s": segment.center_s,
                "segment_n_samples": segment.n_samples,
                "n_missing_within_segment": missing_within,
                "available_support_s": (
                    selected.duration_s if selected is not None else 0.0
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "identity": ids,
        "max_gap_s": max_gap_s,
        "selected_block": (
            None
            if selected is None
            else {
                "start_s": selected.start_s,
                "end_s": selected.end_s,
                "n_samples": selected.n_samples,
                "duration_s": selected.duration_s,
            }
        ),
        "center_s": center_s,
        "n_common_support": n_common,
        "n_blocks": len(blocks),
        "blocks": [
            {
                "start_s": block.start_s,
                "end_s": block.end_s,
                "n_samples": block.n_samples,
            }
            for block in blocks
        ],
        "segments": {
            str(duration): {
                "duration_s": segment.duration_s,
                "role": segment.role,
                "eligible": segment.eligible,
                "exclusion_reason": segment.exclusion_reason,
                "start_s": segment.start_s if segment.eligible else None,
                "end_s": segment.end_s if segment.eligible else None,
                "center_s": segment.center_s if math.isfinite(segment.center_s) else None,
                "n_samples": segment.n_samples,
            }
            for duration, segment in segments.items()
        },
    }
    return HarmonizeResult(
        identity=ids,
        selected_block=selected,
        center_s=center_s,
        segments=segments,
        features_by_duration=features_by_duration,
        qc_rows=tuple(qc_rows),
        manifest=manifest,
    )


def multitaper_timeseries_from_features(
    rows: Iterable[Mapping[str, object]],
    *,
    aggregation: str = "robust_median",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Pivot multitaper feature rows onto a common window-center timeline."""
    selected = [
        row
        for row in rows
        if str(row.get("aggregation", "")).strip().casefold()
        == aggregation.casefold()
        or (
            aggregation == "robust_median"
            and str(row.get("channel", "")).strip() == ROBUST_MEDIAN_CHANNEL
        )
    ]
    by_time: dict[float, dict[str, dict[str, float]]] = {}
    for row in selected:
        time_s = float(row["window_center_s"])
        band = str(row["band"]).strip().casefold()
        by_time.setdefault(time_s, {})[band] = {
            "absolute_power": float(row["absolute_power"]),
            "absolute_log10_power": float(row["absolute_log10_power"]),
        }
    times = np.asarray(sorted(by_time), dtype=float)
    absolute = {
        band: np.asarray(
            [
                by_time[time_s].get(band, {}).get("absolute_power", float("nan"))
                for time_s in times
            ],
            dtype=float,
        )
        for band in BAND_ORDER
    }
    abs_log = {
        band: np.asarray(
            [
                by_time[time_s]
                .get(band, {})
                .get("absolute_log10_power", float("nan"))
                for time_s in times
            ],
            dtype=float,
        )
        for band in BAND_ORDER
    }
    valid = np.ones(times.shape, dtype=bool)
    for band in BAND_ORDER:
        valid &= np.isfinite(absolute[band]) & np.isfinite(abs_log[band])
    return times, absolute, abs_log, valid


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_harmonize_outputs(
    result: HarmonizeResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write per-duration feature tables, segment manifest, and alignment QC."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    example_fields = [
        "dataset_id",
        "subject_id",
        "task",
        "condition",
        "observation_id",
        "duration_s",
        "duration_role",
        "time_s",
        "hr_bpm",
        "hr_z",
    ]
    for band in BAND_ORDER:
        example_fields.extend(
            [
                f"{band}_absolute_power",
                f"{band}_absolute_log10_power",
                f"{band}_relative_power",
                f"{band}_broadband_residualized_log10",
                f"{band}_absolute_log10_power_z",
                f"{band}_relative_power_z",
                f"{band}_broadband_residualized_log10_z",
            ]
        )

    for duration, rows in result.features_by_duration.items():
        path = output_path / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
        fieldnames = list(rows[0].keys()) if rows else example_fields
        _write_csv(path, rows, fieldnames)
        written[f"features_D{duration}"] = path

    manifest_path = output_path / SEGMENT_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["segment_manifest"] = manifest_path

    qc_path = output_path / ALIGNMENT_QC_FILENAME
    qc_fields = list(result.qc_rows[0].keys()) if result.qc_rows else [
        "dataset_id",
        "subject_id",
        "task",
        "condition",
        "observation_id",
        "duration_s",
        "duration_role",
        "eligible",
        "exclusion_reason",
        "selected_block_start_s",
        "selected_block_end_s",
        "selected_block_n_samples",
        "n_hr_valid",
        "n_eeg_valid",
        "n_common_support",
        "n_blocks",
        "segment_start_s",
        "segment_end_s",
        "segment_center_s",
        "segment_n_samples",
        "n_missing_within_segment",
        "available_support_s",
    ]
    _write_csv(qc_path, result.qc_rows, qc_fields)
    written["alignment_qc"] = qc_path
    return written


__all__ = [
    "ALIGNMENT_QC_FILENAME",
    "ALIGNED_FEATURES_TEMPLATE",
    "MAX_GAP_S",
    "NESTED_DURATIONS_S",
    "SEGMENT_MANIFEST_FILENAME",
    "SENSITIVITY_DURATION_S",
    "ContiguousBlock",
    "DurationSegment",
    "HarmonizeResult",
    "build_duration_segments",
    "contiguous_blocks",
    "harmonize_observation",
    "intersect_valid_times",
    "multitaper_timeseries_from_features",
    "place_centered_segment",
    "select_longest_block",
    "write_harmonize_outputs",
]
