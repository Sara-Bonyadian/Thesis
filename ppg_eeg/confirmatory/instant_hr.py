"""Instantaneous heart-rate reconstruction from accepted detected beats."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

FEATURES_FILENAME = "features_instant_hr.csv"
QC_FILENAME = "instant_hr_qc.csv"
OUTPUT_FS_HZ = 1.0
MAX_BEAT_GAP_S = 5.0
# Documented reconstruction settings (algorithm unchanged).
HR_SAMPLING_RATE_HZ = OUTPUT_FS_HZ
HR_INTERPOLATION_METHOD = "pchip"


@dataclass(frozen=True)
class AcceptedBeat:
    source_beat_index: int
    peak_time_s: float
    peak_sample: int | None


@dataclass(frozen=True)
class InstantHRFeature:
    dataset_id: str
    subject_id: str
    task: str
    observation_id: str
    time_s: float
    instant_hr_bpm: float | None
    is_valid_hr: bool
    is_interpolated: bool
    is_gap_masked: bool
    source_beat_index_left: int | None
    source_beat_index_right: int | None
    source_peak_sample_left: int | None
    source_peak_sample_right: int | None

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "observation_id": self.observation_id,
            "time_s": self.time_s,
            "instant_hr_bpm": self.instant_hr_bpm,
            "is_valid_hr": self.is_valid_hr,
            "is_interpolated": self.is_interpolated,
            "is_gap_masked": self.is_gap_masked,
            "source_beat_index_left": self.source_beat_index_left,
            "source_beat_index_right": self.source_beat_index_right,
            "source_peak_sample_left": self.source_peak_sample_left,
            "source_peak_sample_right": self.source_peak_sample_right,
        }


@dataclass(frozen=True)
class InstantHRQC:
    dataset_id: str
    subject_id: str
    task: str
    observation_id: str
    peaks_file: str
    n_source_rows: int
    n_accepted_beats: int
    n_rejected_beats: int
    n_hr_beat_samples: int
    first_accepted_beat_s: float | None
    last_accepted_beat_s: float | None
    clean_beat_span_s: float | None
    hr_support_start_s: float | None
    hr_support_end_s: float | None
    n_grid_samples: int
    n_valid_hr_samples: int
    n_interpolated_samples: int
    hr_sampling_rate_hz: float
    hr_interpolation_method: str
    n_gap_masked_samples: int
    n_long_gaps: int
    max_beat_gap_s: float | None
    status: str
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "observation_id": self.observation_id,
            "peaks_file": self.peaks_file,
            "n_source_rows": self.n_source_rows,
            "n_accepted_beats": self.n_accepted_beats,
            "n_rejected_beats": self.n_rejected_beats,
            "n_hr_beat_samples": self.n_hr_beat_samples,
            "first_accepted_beat_s": self.first_accepted_beat_s,
            "last_accepted_beat_s": self.last_accepted_beat_s,
            "clean_beat_span_s": self.clean_beat_span_s,
            "hr_support_start_s": self.hr_support_start_s,
            "hr_support_end_s": self.hr_support_end_s,
            "n_grid_samples": self.n_grid_samples,
            "n_valid_hr_samples": self.n_valid_hr_samples,
            "n_interpolated_samples": self.n_interpolated_samples,
            "hr_sampling_rate_hz": self.hr_sampling_rate_hz,
            "hr_interpolation_method": self.hr_interpolation_method,
            "n_gap_masked_samples": self.n_gap_masked_samples,
            "n_long_gaps": self.n_long_gaps,
            "max_beat_gap_s": self.max_beat_gap_s,
            "status": self.status,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class InstantHRResult:
    features: tuple[InstantHRFeature, ...]
    qc: InstantHRQC


def _parse_bool(value: object, *, field: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be boolean, got {value!r}.")


def _optional_int(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _identity(rows: list[Mapping[str, object]]) -> dict[str, str]:
    first = rows[0] if rows else {}
    return {
        "dataset_id": str(first.get("dataset_id", "")).strip().casefold(),
        "subject_id": str(first.get("subject_id", "")).strip().casefold(),
        "task": str(first.get("task", "")).strip().casefold(),
        "observation_id": str(first.get("observation_id", "")).strip(),
    }


def accepted_beats_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[AcceptedBeat], int]:
    """Select accepted beats and retain their original CSV row indices."""
    materialized = list(rows)
    beats: list[AcceptedBeat] = []
    for row_index, row in enumerate(materialized):
        if "peak_time_s" not in row or "is_accepted_peak" not in row:
            raise ValueError(
                "detected_peaks.csv requires peak_time_s and is_accepted_peak."
            )
        if not _parse_bool(row["is_accepted_peak"], field="is_accepted_peak"):
            continue
        peak_time_s = float(row["peak_time_s"])
        if not math.isfinite(peak_time_s):
            raise ValueError("Accepted peak_time_s values must be finite.")
        source_index = _optional_int(row.get("source_beat_index", ""))
        beats.append(
            AcceptedBeat(
                source_beat_index=(
                    row_index if source_index is None else source_index
                ),
                peak_time_s=peak_time_s,
                peak_sample=_optional_int(row.get("peak_sample", "")),
            )
        )
    beats.sort(key=lambda beat: beat.peak_time_s)
    times = np.asarray([beat.peak_time_s for beat in beats], dtype=float)
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("Accepted beat times must be unique and strictly increasing.")
    return beats, len(materialized)


def _contiguous_true_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ranges of true values in a one-dimensional mask."""
    if mask.size == 0:
        return []
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if value and index == mask.size - 1 else index - 1
            ranges.append((start, end))
            start = None
    return ranges


def _empty_qc(
    identity: Mapping[str, str],
    *,
    peaks_file: str,
    n_source_rows: int,
    beats: list[AcceptedBeat],
    status: str,
    warning: str,
    max_beat_gap_s: float,
) -> InstantHRQC:
    gaps = np.diff([beat.peak_time_s for beat in beats])
    return InstantHRQC(
        **identity,
        peaks_file=peaks_file,
        n_source_rows=n_source_rows,
        n_accepted_beats=len(beats),
        n_rejected_beats=n_source_rows - len(beats),
        n_hr_beat_samples=max(0, len(beats) - 1),
        first_accepted_beat_s=beats[0].peak_time_s if beats else None,
        last_accepted_beat_s=beats[-1].peak_time_s if beats else None,
        clean_beat_span_s=(
            _max_contiguous_beat_span_s(beats, max_beat_gap_s)
            if len(beats) >= 2 else None
        ),
        hr_support_start_s=None,
        hr_support_end_s=None,
        n_grid_samples=0,
        n_valid_hr_samples=0,
        n_interpolated_samples=0,
        hr_sampling_rate_hz=HR_SAMPLING_RATE_HZ,
        hr_interpolation_method=HR_INTERPOLATION_METHOD,
        n_gap_masked_samples=0,
        n_long_gaps=int(np.sum(gaps > max_beat_gap_s)),
        max_beat_gap_s=float(np.max(gaps)) if gaps.size else None,
        status=status,
        warning=warning,
    )


def _max_contiguous_beat_span_s(
    beats: list[AcceptedBeat],
    max_beat_gap_s: float,
) -> float:
    times = np.asarray([beat.peak_time_s for beat in beats], dtype=float)
    if times.size < 2:
        return 0.0
    split_after = np.flatnonzero(np.diff(times) > max_beat_gap_s)
    boundaries = np.concatenate(([-1], split_after, [times.size - 1]))
    return float(
        max(
            times[int(boundaries[index + 1])]
            - times[int(boundaries[index] + 1)]
            for index in range(len(boundaries) - 1)
        )
    )


def reconstruct_instant_hr(
    rows: Iterable[Mapping[str, object]],
    *,
    peaks_file: str = "",
    max_beat_gap_s: float = MAX_BEAT_GAP_S,
) -> InstantHRResult:
    """Reconstruct PCHIP instantaneous HR on an integer-second grid."""
    if not math.isfinite(max_beat_gap_s) or max_beat_gap_s <= 0:
        raise ValueError("max_beat_gap_s must be finite and positive.")
    materialized = list(rows)
    identity = _identity(materialized)
    beats, n_source_rows = accepted_beats_from_rows(materialized)
    if len(beats) < 3:
        qc = _empty_qc(
            identity,
            peaks_file=peaks_file,
            n_source_rows=n_source_rows,
            beats=beats,
            status="insufficient_beats",
            warning="At least three accepted beats are required for PCHIP HR.",
            max_beat_gap_s=max_beat_gap_s,
        )
        return InstantHRResult(features=(), qc=qc)

    beat_times = np.asarray([beat.peak_time_s for beat in beats], dtype=float)
    beat_gaps = np.diff(beat_times)
    valid_intervals = beat_gaps <= max_beat_gap_s
    hr_times = beat_times[1:]
    hr_bpm = 60.0 / beat_gaps
    later_beats = beats[1:]

    # A PCHIP segment needs at least two consecutive valid IBI-derived samples.
    segment_ranges = [
        (start, end)
        for start, end in _contiguous_true_ranges(valid_intervals)
        if end - start + 1 >= 2
    ]
    if not segment_ranges:
        qc = _empty_qc(
            identity,
            peaks_file=peaks_file,
            n_source_rows=n_source_rows,
            beats=beats,
            status="insufficient_contiguous_beats",
            warning="No segment contains two consecutive valid IBI samples.",
            max_beat_gap_s=max_beat_gap_s,
        )
        return InstantHRResult(features=(), qc=qc)

    support_start = float(hr_times[segment_ranges[0][0]])
    support_end = float(hr_times[segment_ranges[-1][1]])
    grid_start = int(math.ceil(support_start))
    grid_end = int(math.floor(support_end))
    if grid_start > grid_end:
        qc = _empty_qc(
            identity,
            peaks_file=peaks_file,
            n_source_rows=n_source_rows,
            beats=beats,
            status="no_1hz_support",
            warning="Valid beat support contains no integer-second sample.",
            max_beat_gap_s=max_beat_gap_s,
        )
        return InstantHRResult(features=(), qc=qc)

    grid = np.arange(grid_start, grid_end + 1, dtype=float)
    values = np.full(grid.shape, np.nan, dtype=float)
    valid_mask = np.zeros(grid.shape, dtype=bool)
    interpolated_mask = np.zeros(grid.shape, dtype=bool)
    gap_mask = np.ones(grid.shape, dtype=bool)
    left_indices = np.full(grid.shape, -1, dtype=int)
    right_indices = np.full(grid.shape, -1, dtype=int)
    left_samples = np.full(grid.shape, -1, dtype=int)
    right_samples = np.full(grid.shape, -1, dtype=int)

    for start, end in segment_ranges:
        sample_times = hr_times[start : end + 1]
        sample_values = hr_bpm[start : end + 1]
        segment_grid_mask = (grid >= sample_times[0]) & (grid <= sample_times[-1])
        segment_grid = grid[segment_grid_mask]
        if segment_grid.size == 0:
            continue
        values[segment_grid_mask] = PchipInterpolator(
            sample_times, sample_values, extrapolate=False
        )(segment_grid)
        valid_mask[segment_grid_mask] = True
        gap_mask[segment_grid_mask] = False

        segment_positions = np.flatnonzero(segment_grid_mask)
        for output_position, time_s in zip(
            segment_positions, segment_grid, strict=True
        ):
            exact = np.flatnonzero(np.isclose(sample_times, time_s, atol=1e-12))
            if exact.size:
                local_left = local_right = int(exact[0])
                interpolated_mask[output_position] = False
            else:
                local_right = int(np.searchsorted(sample_times, time_s))
                local_left = local_right - 1
                interpolated_mask[output_position] = True
            left_beat = later_beats[start + local_left]
            right_beat = later_beats[start + local_right]
            left_indices[output_position] = left_beat.source_beat_index
            right_indices[output_position] = right_beat.source_beat_index
            left_samples[output_position] = (
                -1 if left_beat.peak_sample is None else left_beat.peak_sample
            )
            right_samples[output_position] = (
                -1 if right_beat.peak_sample is None else right_beat.peak_sample
            )

    features = tuple(
        InstantHRFeature(
            **identity,
            time_s=float(time_s),
            instant_hr_bpm=float(values[index]) if valid_mask[index] else None,
            is_valid_hr=bool(valid_mask[index]),
            is_interpolated=bool(interpolated_mask[index]),
            is_gap_masked=bool(gap_mask[index]),
            source_beat_index_left=(
                int(left_indices[index]) if left_indices[index] >= 0 else None
            ),
            source_beat_index_right=(
                int(right_indices[index]) if right_indices[index] >= 0 else None
            ),
            source_peak_sample_left=(
                int(left_samples[index]) if left_samples[index] >= 0 else None
            ),
            source_peak_sample_right=(
                int(right_samples[index]) if right_samples[index] >= 0 else None
            ),
        )
        for index, time_s in enumerate(grid)
    )
    n_long_gaps = int(np.sum(beat_gaps > max_beat_gap_s))
    qc = InstantHRQC(
        **identity,
        peaks_file=peaks_file,
        n_source_rows=n_source_rows,
        n_accepted_beats=len(beats),
        n_rejected_beats=n_source_rows - len(beats),
        n_hr_beat_samples=len(beats) - 1,
        first_accepted_beat_s=float(beat_times[0]),
        last_accepted_beat_s=float(beat_times[-1]),
        clean_beat_span_s=_max_contiguous_beat_span_s(
            beats, max_beat_gap_s
        ),
        hr_support_start_s=support_start,
        hr_support_end_s=support_end,
        n_grid_samples=len(features),
        n_valid_hr_samples=int(np.sum(valid_mask)),
        n_interpolated_samples=int(np.sum(interpolated_mask)),
        hr_sampling_rate_hz=HR_SAMPLING_RATE_HZ,
        hr_interpolation_method=HR_INTERPOLATION_METHOD,
        n_gap_masked_samples=int(np.sum(gap_mask)),
        n_long_gaps=n_long_gaps,
        max_beat_gap_s=float(np.max(beat_gaps)),
        status="ok" if np.any(valid_mask) else "no_valid_grid_samples",
        warning=(
            f"{n_long_gaps} beat gap(s) longer than {max_beat_gap_s:g} s masked."
            if n_long_gaps
            else ""
        ),
    )
    return InstantHRResult(features=features, qc=qc)


def read_detected_peaks(path: str | Path) -> list[dict[str, str]]:
    peaks_path = Path(path).expanduser().resolve()
    if not peaks_path.is_file():
        raise FileNotFoundError(f"detected_peaks.csv not found: {peaks_path}")
    with peaks_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_instant_hr_file(
    peaks_path: str | Path,
    output_dir: str | Path,
    *,
    max_beat_gap_s: float = MAX_BEAT_GAP_S,
) -> tuple[Path, Path]:
    """Read one detected-peaks file and write feature/QC outputs."""
    resolved_peaks = Path(peaks_path).expanduser().resolve()
    result = reconstruct_instant_hr(
        read_detected_peaks(resolved_peaks),
        peaks_file=str(resolved_peaks),
        max_beat_gap_s=max_beat_gap_s,
    )
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    feature_fields = list(
        InstantHRFeature(
            dataset_id="",
            subject_id="",
            task="",
            observation_id="",
            time_s=0.0,
            instant_hr_bpm=None,
            is_valid_hr=False,
            is_interpolated=False,
            is_gap_masked=False,
            source_beat_index_left=None,
            source_beat_index_right=None,
            source_peak_sample_left=None,
            source_peak_sample_right=None,
        ).to_row()
    )
    features_path = output_path / FEATURES_FILENAME
    _write_rows(
        features_path,
        (feature.to_row() for feature in result.features),
        feature_fields,
    )

    qc_path = output_path / QC_FILENAME
    qc_row = result.qc.to_row()
    _write_rows(qc_path, [qc_row], list(qc_row))
    return features_path, qc_path


__all__ = [
    "FEATURES_FILENAME",
    "MAX_BEAT_GAP_S",
    "OUTPUT_FS_HZ",
    "QC_FILENAME",
    "AcceptedBeat",
    "InstantHRFeature",
    "InstantHRQC",
    "InstantHRResult",
    "accepted_beats_from_rows",
    "read_detected_peaks",
    "reconstruct_instant_hr",
    "reconstruct_instant_hr_file",
]
