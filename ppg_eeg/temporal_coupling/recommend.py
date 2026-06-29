from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .config import TemporalCouplingConfig

ALIGNMENT_QC_FILENAME = "alignment_qc.csv"
EEG_FEATURES_FILENAME = "features_temporal_eeg_envelope.csv"
CARDIAC_FEATURES_FILENAME = "features_temporal_cardiac.csv"


@dataclass(frozen=True)
class OverlapRecommendation:
    min_overlap_s: int
    max_usable_count: int
    chosen_usable_count: int
    chosen_usable_rate: float


@dataclass(frozen=True)
class WindowRecommendation:
    hr_window_s: float
    mean_rr_window_s: float
    hrv_window_s: float
    no_overlap_rows: int
    no_overlap_recoverable_fraction: float


def _as_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _time_range(csv_path: Path) -> tuple[float, float, int] | None:
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    values = [_as_float(row.get("time_s")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return min(finite), max(finite), len(finite)


def recommend_min_overlap(cfg: TemporalCouplingConfig) -> OverlapRecommendation:
    group_dir = Path(cfg.paths.out_root) / cfg.dataset_id / "group"
    alignment_path = group_dir / ALIGNMENT_QC_FILENAME
    if not alignment_path.is_file():
        raise FileNotFoundError(f"missing {alignment_path}; run --stage 1c first.")

    with alignment_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{alignment_path} is empty.")

    pairs: list[tuple[float, bool]] = []
    for row in rows:
        duration = _as_float(row.get("aligned_duration_s"))
        if not math.isfinite(duration):
            continue
        pairs.append((duration, _as_bool(row.get("usable_for_xcorr"))))
    if not pairs:
        raise ValueError("alignment_qc has no finite aligned_duration_s values.")

    thresholds = (10, 15, 20, 25, 30, 40, 50, 60)
    stats: list[tuple[int, int, int, float]] = []
    for threshold in thresholds:
        kept = [usable for duration, usable in pairs if duration >= threshold]
        if not kept:
            continue
        usable_count = sum(1 for value in kept if value)
        usable_rate = usable_count / len(kept)
        stats.append((threshold, len(kept), usable_count, usable_rate))
    if not stats:
        raise ValueError("no overlap-threshold statistics could be computed.")

    max_usable_count = max(usable_count for _, _, usable_count, _ in stats)
    chosen = None
    for threshold, _kept, usable_count, usable_rate in stats:
        if usable_rate >= 0.90 and usable_count >= int(round(0.95 * max_usable_count)):
            chosen = (threshold, usable_count, usable_rate)
            break
    if chosen is None:
        best = max(stats, key=lambda item: (item[2], item[3], -item[0]))
        chosen = (best[0], best[2], best[3])

    return OverlapRecommendation(
        min_overlap_s=chosen[0],
        max_usable_count=max_usable_count,
        chosen_usable_count=chosen[1],
        chosen_usable_rate=chosen[2],
    )


def recommend_cardiac_windows(cfg: TemporalCouplingConfig) -> WindowRecommendation:
    dataset_dir = Path(cfg.paths.out_root) / cfg.dataset_id
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"missing dataset output directory: {dataset_dir}")

    no_overlap_eeg_ends: list[float] = []
    for path in dataset_dir.iterdir():
        if not path.is_dir() or path.name == "group":
            continue
        if (path / "features_temporal_aligned.csv").is_file():
            continue
        eeg_range = _time_range(path / EEG_FEATURES_FILENAME)
        cardiac_range = _time_range(path / CARDIAC_FEATURES_FILENAME)
        if eeg_range is None or cardiac_range is None:
            continue
        eeg_start, eeg_end, _ = eeg_range
        cardiac_start, cardiac_end, _ = cardiac_range
        overlap_start = max(eeg_start, cardiac_start)
        overlap_end = min(eeg_end, cardiac_end)
        if overlap_end <= overlap_start:
            no_overlap_eeg_ends.append(eeg_end)

    if not no_overlap_eeg_ends:
        current = cfg.temporal_coupling.cardiac
        return WindowRecommendation(
            hr_window_s=current.hr_window_s,
            mean_rr_window_s=current.mean_rr_window_s or current.hr_window_s,
            hrv_window_s=current.hrv_window_s,
            no_overlap_rows=0,
            no_overlap_recoverable_fraction=0.0,
        )

    candidate_half_windows = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0)
    chosen_half = candidate_half_windows[-1]
    recoverable_fraction = 0.0
    n_rows = len(no_overlap_eeg_ends)
    for half in candidate_half_windows:
        recoverable = sum(1 for eeg_end in no_overlap_eeg_ends if eeg_end > half)
        fraction = recoverable / n_rows
        if fraction >= 0.5:
            chosen_half = half
            recoverable_fraction = fraction
            break
        if fraction > recoverable_fraction:
            recoverable_fraction = fraction
            chosen_half = half

    hr_window = round(2.0 * chosen_half, 1)
    mean_rr_window = hr_window
    hrv_window = max(20.0, round(hr_window * 2.0, 1))
    return WindowRecommendation(
        hr_window_s=hr_window,
        mean_rr_window_s=mean_rr_window,
        hrv_window_s=hrv_window,
        no_overlap_rows=n_rows,
        no_overlap_recoverable_fraction=recoverable_fraction,
    )


def print_recommendations(cfg: TemporalCouplingConfig) -> None:
    overlap = recommend_min_overlap(cfg)
    windows = recommend_cardiac_windows(cfg)

    print("[temporal_coupling] data-driven config recommendations")
    print(f"[temporal_coupling] dataset_id={cfg.dataset_id}")
    print(
        "[temporal_coupling] overlap rationale: "
        f"recommended min_overlap_s={overlap.min_overlap_s} "
        f"(usable={overlap.chosen_usable_count}/{overlap.max_usable_count} max, "
        f"usable_rate={overlap.chosen_usable_rate:.3f})"
    )
    print(
        "[temporal_coupling] cardiac-window rationale: "
        f"no_overlap_rows={windows.no_overlap_rows}, "
        f"recoverable_with_half_window={windows.no_overlap_recoverable_fraction:.3f}"
    )
    print("[temporal_coupling] suggested YAML:")
    print("temporal_coupling:")
    print("  audit:")
    print(f"    min_overlap_s: {overlap.min_overlap_s}")
    print("  cardiac:")
    print(f"    hr_window_s: {windows.hr_window_s}")
    print(f"    mean_rr_window_s: {windows.mean_rr_window_s}")
    print(f"    hrv_window_s: {windows.hrv_window_s}")
    print("[temporal_coupling] rerun --stage 1b then --stage 1c after applying.")
