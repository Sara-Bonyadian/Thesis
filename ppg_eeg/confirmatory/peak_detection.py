"""Confirmatory C1b peak detection (reuses exploratory cardiac detectors).

Writes peaks and cardiac QC under the confirmatory ``C1b/`` tree. Does not
require ``temporal_coupling --stage 1b`` or a separate beats YAML.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..core_eeg_ppg.features_core import _read_raw
from ..datasets import CanonicalObservation
from ..temporal_coupling.cardiac_common import CardiacObservation
from ..temporal_coupling.cardiac_detectors import (
    CHANNEL_INVENTORY_FILENAME,
    CHANNEL_PREVIEW_FILENAME,
    DETECTOR_COMPARISON_FILENAME,
    build_channel_inventory,
    compare_detectors,
    plot_channel_preview,
)
from ..temporal_coupling.cardiac_timeseries import peaks_to_dataframe
from ..temporal_coupling.config import (
    TemporalCouplingAuditConfig,
    TemporalCouplingCardiacConfig,
    TemporalCouplingCardiacDebugPlotConfig,
    TemporalCouplingCardiacEcgConfig,
    TemporalCouplingConfig,
    TemporalCouplingCrossCorrelationConfig,
    TemporalCouplingEegBandsConfig,
    TemporalCouplingEegRoiConfig,
    TemporalCouplingEegSectionConfig,
    TemporalCouplingEventsConfig,
    TemporalCouplingGroupConfig,
    TemporalCouplingOutputConfig,
    TemporalCouplingResampleConfig,
    TemporalCouplingSectionConfig,
    TemporalEegConfig,
    TemporalPathsConfig,
    TemporalPpgConfig,
)
from .config import ConfirmatoryDatasetConfig, ConfirmatoryMasterConfig

PEAKS_FILENAME = "detected_peaks.csv"
PEAK_QC_FILENAME = "cardiac_peak_qc.csv"
GROUP_PEAK_QC_FILENAME = "cardiac_peak_qc.csv"


@dataclass(frozen=True)
class PeakDetectionQc:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    cardiac_file: str
    channel_used: str
    signal_type: str
    detector_used: str
    detector_polarity: str
    selection_reason: str
    n_raw_peaks: int
    n_clean_ibis: int
    n_accepted_peaks: int
    first_peak_time_s: float
    last_peak_time_s: float
    clean_ibi_coverage_s: float
    median_hr_bpm: float
    usable: bool
    warning: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "cardiac_file": self.cardiac_file,
            "channel_used": self.channel_used,
            "signal_type": self.signal_type,
            "detector_used": self.detector_used,
            "detector_polarity": self.detector_polarity,
            "selection_reason": self.selection_reason,
            "n_raw_peaks": self.n_raw_peaks,
            "n_clean_ibis": self.n_clean_ibis,
            "n_accepted_peaks": self.n_accepted_peaks,
            "first_peak_time_s": self.first_peak_time_s,
            "last_peak_time_s": self.last_peak_time_s,
            "clean_ibi_coverage_s": self.clean_ibi_coverage_s,
            "median_hr_bpm": self.median_hr_bpm,
            "usable": self.usable,
            "warning": self.warning,
        }


def canonical_to_cardiac_observation(obs: CanonicalObservation) -> CardiacObservation:
    if obs.ppg_source == "embedded_eeg":
        cardiac_path = obs.eeg_path
        cardiac_format = obs.eeg_format
    else:
        if obs.ppg_path is None:
            raise FileNotFoundError(
                f"No cardiac file for observation {obs.observation_id!r}."
            )
        cardiac_path = obs.ppg_path
        cardiac_format = obs.ppg_format or "eeglab"
    if not cardiac_path.is_file():
        raise FileNotFoundError(f"Cardiac file missing: {cardiac_path}")
    return CardiacObservation(
        dataset_id=obs.dataset_id,
        subject_id=obs.subject_id,
        task=obs.task_label,
        condition=obs.condition_label,
        observation_id=obs.observation_id,
        cardiac_file=cardiac_path,
        cardiac_format=cardiac_format,
    )


def tc_config_for_peak_detection(
    dataset: ConfirmatoryDatasetConfig,
    master: ConfirmatoryMasterConfig,
    *,
    out_root: Path,
) -> TemporalCouplingConfig:
    """Build a minimal TemporalCouplingConfig for detector reuse only."""
    cardiac = dataset.cardiac
    lag_max_s = float(master.lag.max_s)
    return TemporalCouplingConfig(
        dataset_id=dataset.dataset_id,
        paths=TemporalPathsConfig(
            raw_root=dataset.paths.raw_root,
            out_root=out_root,
            base_root=out_root,
        ),
        subjects=list(dataset.selection.subjects),
        tasks=list(dataset.selection.tasks),
        conditions=list(dataset.selection.conditions),
        sessions=list(dataset.selection.sessions),
        eeg=TemporalEegConfig(),
        ppg=TemporalPpgConfig(
            start_time_s=cardiac.start_time_s,
            end_time_s=cardiac.end_time_s,
            peak_min_distance_s=cardiac.peak_min_distance_s,
            peak_height=cardiac.peak_height,
            ibi_min_ms=cardiac.ibi_min_ms,
            ibi_max_ms=cardiac.ibi_max_ms,
        ),
        temporal_coupling=TemporalCouplingSectionConfig(
            audit=TemporalCouplingAuditConfig(min_overlap_s=60.0, min_clean_beats=30),
            eeg=TemporalCouplingEegSectionConfig(
                bands=TemporalCouplingEegBandsConfig(),
                envelope_smooth_s=2.0,
                envelope_output_fs_hz=10.0,
                rois=TemporalCouplingEegRoiConfig(
                    theta=("Fz",),
                    alpha=("Pz",),
                    beta=("Fz",),
                ),
            ),
            cardiac=TemporalCouplingCardiacConfig(
                channel=cardiac.channel,
                signal_type=cardiac.signal_type,
                detector=cardiac.detector,
                ecg=TemporalCouplingCardiacEcgConfig(),
                debug_plot=TemporalCouplingCardiacDebugPlotConfig(
                    enabled=cardiac.debug_plot,
                    save_overview=False,
                    max_plots_per_subject=1,
                ),
            ),
            resample=TemporalCouplingResampleConfig(fs_hz=1.0),
            cross_correlation=TemporalCouplingCrossCorrelationConfig(
                lag_max_s=lag_max_s,
                lag_step_s=float(master.lag.step_s),
                n_permutations=0,
            ),
            events=TemporalCouplingEventsConfig(enabled=False),
            output=TemporalCouplingOutputConfig(save_curves=False, save_plots=False),
            group=TemporalCouplingGroupConfig(n_group_permutations=0),
        ),
        hiit_partition_mode="protocol_task",
    )


def _qc_from_detection(
    cardiac_obs: CardiacObservation,
    detection,
) -> PeakDetectionQc:
    clean_times = [
        ann.peak_time_s
        for ann in detection.peak_annotations
        if ann.is_clean_ibi and ann.cleaning_reason != "first_peak"
    ]
    n_clean_ibis = sum(
        1
        for ann in detection.peak_annotations
        if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)
    )
    n_accepted = sum(1 for ann in detection.peak_annotations if ann.is_accepted_peak)
    if clean_times:
        first_peak = float(min(clean_times))
        last_peak = float(max(clean_times))
        coverage = last_peak - first_peak
    elif detection.peak_times_s.size:
        first_peak = float(detection.peak_times_s[0])
        last_peak = float(detection.peak_times_s[-1])
        coverage = last_peak - first_peak
    else:
        first_peak = float("nan")
        last_peak = float("nan")
        coverage = 0.0

    peak_ibis = [
        ann.ibi_ms
        for ann in detection.peak_annotations
        if ann.is_clean_ibi and not math.isnan(ann.ibi_ms)
    ]
    median_hr = (
        float(60000.0 / float(pd.Series(peak_ibis).median())) if peak_ibis else float("nan")
    )
    warnings_out: list[str] = []
    if detection.peak_times_s.size == 0:
        warnings_out.append("no_peaks_detected")
    if coverage < 0.7 * max(detection.duration_s, 1e-9):
        warnings_out.append("poor_peak_coverage")
    if n_clean_ibis < 30:
        warnings_out.append("few_clean_ibis")
    if detection.signal_type == "unknown":
        warnings_out.append("uncertain_channel_selection")
    usable = (
        n_accepted >= 10
        and n_clean_ibis >= 30
        and coverage >= 60.0
        and (math.isnan(median_hr) or 40.0 <= median_hr <= 180.0)
    )
    return PeakDetectionQc(
        dataset_id=cardiac_obs.dataset_id,
        subject_id=cardiac_obs.subject_id,
        task=cardiac_obs.task,
        condition=cardiac_obs.condition,
        observation_id=cardiac_obs.observation_id,
        cardiac_file=str(cardiac_obs.cardiac_file),
        channel_used=detection.channel_used,
        signal_type=detection.signal_type,
        detector_used=detection.detector_name,
        detector_polarity="inverted" if detection.inverted else "normal",
        selection_reason=detection.selection_reason,
        n_raw_peaks=int(detection.peak_times_s.size),
        n_clean_ibis=n_clean_ibis,
        n_accepted_peaks=n_accepted,
        first_peak_time_s=first_peak,
        last_peak_time_s=last_peak,
        clean_ibi_coverage_s=coverage,
        median_hr_bpm=median_hr,
        usable=usable,
        warning=";".join(warnings_out),
    )


def detect_peaks_for_observation(
    obs: CanonicalObservation,
    cfg: TemporalCouplingConfig,
    out_dir: Path,
) -> tuple[Path, PeakDetectionQc]:
    """Detect peaks for one observation and write C1b peak artifacts."""
    cardiac_obs = canonical_to_cardiac_observation(obs)
    raw = _read_raw(cardiac_obs.cardiac_file, cardiac_obs.cardiac_format)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_df = build_channel_inventory(raw, cardiac_obs, cfg)
    inventory_df.to_csv(out_dir / CHANNEL_INVENTORY_FILENAME, index=False)
    plot_channel_preview(raw, cardiac_obs, cfg, output_path=out_dir / CHANNEL_PREVIEW_FILENAME)

    comparison_df, detection = compare_detectors(raw, cardiac_obs, cfg)
    comparison_df.to_csv(out_dir / DETECTOR_COMPARISON_FILENAME, index=False)

    peaks_df = peaks_to_dataframe(cardiac_obs, detection)
    peaks_path = out_dir / PEAKS_FILENAME
    peaks_df.to_csv(peaks_path, index=False)

    qc = _qc_from_detection(cardiac_obs, detection)
    pd.DataFrame([qc.to_row()]).to_csv(out_dir / PEAK_QC_FILENAME, index=False)
    return peaks_path, qc


def run_confirmatory_peak_detection(
    observations: Sequence[CanonicalObservation],
    dataset: ConfirmatoryDatasetConfig,
    master: ConfirmatoryMasterConfig,
    stage_root: Path,
) -> tuple[list[Path], list[PeakDetectionQc], list[str]]:
    """Detect peaks for all observations under ``stage_root`` (C1b)."""
    from ..core_eeg_ppg.output_layout import safe_subject_dir_name

    cfg = tc_config_for_peak_detection(dataset, master, out_root=stage_root)
    peaks_paths: list[Path] = []
    qc_rows: list[PeakDetectionQc] = []
    errors: list[str] = []
    n_total = len(observations)
    if n_total:
        print(
            f"[confirmatory] C1b peak detection: {n_total} observations "
            f"(channel={dataset.cardiac.channel}, "
            f"signal_type={dataset.cardiac.signal_type}, "
            f"detector={dataset.cardiac.detector})..."
        )
    for idx, obs in enumerate(observations, start=1):
        obs_out = stage_root / safe_subject_dir_name(obs.observation_id)
        try:
            peaks_path, qc = detect_peaks_for_observation(obs, cfg, obs_out)
            peaks_paths.append(peaks_path)
            qc_rows.append(qc)
            print(
                f"[confirmatory] C1b {obs.observation_id}: "
                f"channel={qc.channel_used} type={qc.signal_type} "
                f"peaks={qc.n_raw_peaks} clean_ibis={qc.n_clean_ibis} "
                f"span={qc.clean_ibi_coverage_s:.1f}s usable={qc.usable}"
            )
        except Exception as exc:  # noqa: BLE001 — continue other observations
            errors.append(f"{obs.observation_id}: {type(exc).__name__}: {exc}")
            print(f"[confirmatory] C1b error {obs.observation_id}: {exc}")
        if idx == 1 or idx == n_total or idx % 10 == 0:
            print(f"[confirmatory] C1b peak progress: {idx}/{n_total}")

    if qc_rows:
        group_path = stage_root / GROUP_PEAK_QC_FILENAME
        pd.DataFrame([row.to_row() for row in qc_rows]).to_csv(group_path, index=False)
        print(f"[confirmatory] C1b wrote group QC -> {group_path}")

    return peaks_paths, qc_rows, errors


__all__ = [
    "GROUP_PEAK_QC_FILENAME",
    "PEAKS_FILENAME",
    "PEAK_QC_FILENAME",
    "PeakDetectionQc",
    "canonical_to_cardiac_observation",
    "detect_peaks_for_observation",
    "run_confirmatory_peak_detection",
    "tc_config_for_peak_detection",
]
