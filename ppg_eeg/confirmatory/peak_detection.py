"""Confirmatory C1b peak detection (reuses exploratory cardiac detectors).

Writes peaks and cardiac QC under the confirmatory ``C1b/`` tree. Does not
require ``temporal_coupling --stage 1b`` or a separate beats YAML.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from ..core_eeg_ppg.features_core import _read_raw
from ..core_eeg_ppg.output_layout import safe_subject_dir_name
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
from .instant_hr import (
    FEATURES_FILENAME as INSTANT_HR_FEATURES,
    MAX_BEAT_GAP_S,
    QC_FILENAME as INSTANT_HR_QC,
    reconstruct_instant_hr_file,
)
from .parallel_util import (
    atomic_write_csv_rows,
    atomic_write_json,
    configure_blas_threads,
    file_identity,
    prepare_obs_checkpoint_dir,
    report_progress,
    resolve_n_jobs,
)

PEAKS_FILENAME = "detected_peaks.csv"
PEAK_QC_FILENAME = "cardiac_peak_qc.csv"
GROUP_PEAK_QC_FILENAME = "cardiac_peak_qc.csv"
CHECKPOINT_SCHEMA_VERSION = "c1b_checkpoint_v1"
CHECKPOINT_DIRNAME = "_obs_checkpoints"
COMPLETE_MARKER_FILENAME = "C1b_COMPLETE.json"


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


def _atomic_dataframe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


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
    _atomic_dataframe_to_csv(inventory_df, out_dir / CHANNEL_INVENTORY_FILENAME)

    debug_enabled = bool(cfg.temporal_coupling.cardiac.debug_plot.enabled)
    if debug_enabled:
        preview_path = out_dir / CHANNEL_PREVIEW_FILENAME
        tmp_preview = out_dir / f"{CHANNEL_PREVIEW_FILENAME}.tmp.png"
        plot_channel_preview(raw, cardiac_obs, cfg, output_path=tmp_preview)
        os.replace(tmp_preview, preview_path)

    comparison_df, detection = compare_detectors(raw, cardiac_obs, cfg)
    _atomic_dataframe_to_csv(comparison_df, out_dir / DETECTOR_COMPARISON_FILENAME)

    peaks_df = peaks_to_dataframe(cardiac_obs, detection)
    peaks_path = out_dir / PEAKS_FILENAME
    _atomic_dataframe_to_csv(peaks_df, peaks_path)

    qc = _qc_from_detection(cardiac_obs, detection)
    atomic_write_csv_rows(
        out_dir / PEAK_QC_FILENAME,
        [qc.to_row()],
        list(qc.to_row()),
    )
    return peaks_path, qc


def _c1b_param_fingerprint(
    dataset: ConfirmatoryDatasetConfig,
    *,
    max_beat_gap_s: float,
) -> dict[str, object]:
    cardiac = dataset.cardiac
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "channel": cardiac.channel,
        "signal_type": cardiac.signal_type,
        "detector": cardiac.detector,
        "peak_min_distance_s": float(cardiac.peak_min_distance_s),
        "peak_height": float(cardiac.peak_height),
        "ibi_min_ms": float(cardiac.ibi_min_ms),
        "ibi_max_ms": float(cardiac.ibi_max_ms),
        "debug_plot": bool(cardiac.debug_plot),
        "start_time_s": cardiac.start_time_s,
        "end_time_s": cardiac.end_time_s,
        "max_beat_gap_s": float(max_beat_gap_s),
        "test_inverted": True,
        "peaks_filename": PEAKS_FILENAME,
        "peak_qc_filename": PEAK_QC_FILENAME,
        "instant_hr_features": INSTANT_HR_FEATURES,
        "instant_hr_qc": INSTANT_HR_QC,
    }


def _c1b_obs_fingerprint(
    obs: CanonicalObservation,
    *,
    params: Mapping[str, object],
) -> dict[str, object]:
    cardiac_obs = canonical_to_cardiac_observation(obs)
    return {
        **dict(params),
        "observation_id": obs.observation_id,
        "cardiac_format": cardiac_obs.cardiac_format,
        "cardiac_file": file_identity(Path(cardiac_obs.cardiac_file)),
    }


def _checkpoint_path(checkpoint_dir: Path, observation_id: str) -> Path:
    digest = hashlib.sha256(observation_id.encode("utf-8")).hexdigest()[:32]
    return checkpoint_dir / f"obs_{digest}.json"


def _required_c1b_files(obs_dir: Path, *, debug_plot: bool) -> list[Path]:
    required = [
        obs_dir / CHANNEL_INVENTORY_FILENAME,
        obs_dir / DETECTOR_COMPARISON_FILENAME,
        obs_dir / PEAKS_FILENAME,
        obs_dir / PEAK_QC_FILENAME,
        obs_dir / INSTANT_HR_FEATURES,
        obs_dir / INSTANT_HR_QC,
    ]
    if debug_plot:
        required.append(obs_dir / CHANNEL_PREVIEW_FILENAME)
    return required


def _c1b_outputs_complete(obs_dir: Path, *, debug_plot: bool) -> bool:
    for path in _required_c1b_files(obs_dir, debug_plot=debug_plot):
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _read_peak_qc_file(path: Path) -> PeakDetectionQc | None:
    if not path.is_file():
        return None
    try:
        rows = pd.read_csv(path).to_dict(orient="records")
    except (OSError, pd.errors.EmptyDataError, ValueError):
        return None
    if not rows:
        return None
    row = rows[0]
    try:
        return PeakDetectionQc(
            dataset_id=str(row["dataset_id"]),
            subject_id=str(row["subject_id"]),
            task=str(row["task"]),
            condition=str(row["condition"]),
            observation_id=str(row["observation_id"]),
            cardiac_file=str(row["cardiac_file"]),
            channel_used=str(row["channel_used"]),
            signal_type=str(row["signal_type"]),
            detector_used=str(row["detector_used"]),
            detector_polarity=str(row["detector_polarity"]),
            selection_reason=str(row.get("selection_reason", "")),
            n_raw_peaks=int(row["n_raw_peaks"]),
            n_clean_ibis=int(row["n_clean_ibis"]),
            n_accepted_peaks=int(row["n_accepted_peaks"]),
            first_peak_time_s=float(row["first_peak_time_s"]),
            last_peak_time_s=float(row["last_peak_time_s"]),
            clean_ibi_coverage_s=float(row["clean_ibi_coverage_s"]),
            median_hr_bpm=float(row["median_hr_bpm"]),
            usable=bool(row["usable"]),
            warning=str(row.get("warning", "") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_c1b_checkpoint(
    checkpoint_dir: Path,
    *,
    observation_id: str,
    fingerprint: Mapping[str, object],
    obs_dir: Path,
    debug_plot: bool,
) -> PeakDetectionQc | None:
    path = _checkpoint_path(checkpoint_dir, observation_id)
    if not path.is_file() or not _c1b_outputs_complete(obs_dir, debug_plot=debug_plot):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if payload.get("fingerprint") != dict(fingerprint):
        return None
    if payload.get("status") != "ok":
        return None
    return _read_peak_qc_file(obs_dir / PEAK_QC_FILENAME)


def _write_c1b_checkpoint(
    checkpoint_dir: Path,
    *,
    observation_id: str,
    fingerprint: Mapping[str, object],
    obs_dir: Path,
    qc: PeakDetectionQc,
    debug_plot: bool,
) -> None:
    del qc  # QC is validated from on-disk peak_qc.csv
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "observation_id": observation_id,
        "status": "ok",
        "fingerprint": dict(fingerprint),
        "outputs": {
            str(path.name): int(path.stat().st_size)
            for path in _required_c1b_files(obs_dir, debug_plot=debug_plot)
        },
    }
    atomic_write_json(_checkpoint_path(checkpoint_dir, observation_id), payload)


def process_one_c1b_observation(
    obs: CanonicalObservation,
    cfg: TemporalCouplingConfig,
    out_dir: Path,
    *,
    max_beat_gap_s: float = MAX_BEAT_GAP_S,
) -> PeakDetectionQc:
    """Peaks + instantaneous HR for one observation."""
    configure_blas_threads(1)
    peaks_path, qc = detect_peaks_for_observation(obs, cfg, out_dir)
    reconstruct_instant_hr_file(
        peaks_path, out_dir, max_beat_gap_s=max_beat_gap_s
    )
    return qc


def _worker_c1b(
    payload: tuple[int, CanonicalObservation, str, TemporalCouplingConfig, float],
) -> tuple[int, dict[str, object] | None, str | None]:
    index, obs, out_dir_s, cfg, max_beat_gap_s = payload
    try:
        qc = process_one_c1b_observation(
            obs,
            cfg,
            Path(out_dir_s),
            max_beat_gap_s=max_beat_gap_s,
        )
        return index, qc.to_row(), None
    except Exception as exc:  # noqa: BLE001
        return index, None, f"{obs.observation_id}: {type(exc).__name__}: {exc}"


def run_confirmatory_c1b(
    observations: Sequence[CanonicalObservation],
    dataset: ConfirmatoryDatasetConfig,
    master: ConfirmatoryMasterConfig,
    stage_root: Path,
    *,
    n_jobs: int | None = -1,
    progress: bool = True,
    max_beat_gap_s: float = MAX_BEAT_GAP_S,
) -> dict[str, object]:
    """Run C1b peak detection + instantaneous HR (serial or parallel)."""
    output_path = Path(stage_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    complete_marker = output_path / COMPLETE_MARKER_FILENAME
    if complete_marker.is_file():
        try:
            complete_marker.unlink()
        except OSError:
            pass

    obs_list = sorted(observations, key=lambda o: o.observation_id)
    cfg = tc_config_for_peak_detection(dataset, master, out_root=output_path)
    debug_plot = bool(dataset.cardiac.debug_plot)
    params = _c1b_param_fingerprint(dataset, max_beat_gap_s=max_beat_gap_s)
    fingerprints = [
        _c1b_obs_fingerprint(obs, params=params) for obs in obs_list
    ]
    obs_dirs = [
        output_path / safe_subject_dir_name(obs.observation_id) for obs in obs_list
    ]
    unit_keys = [obs.observation_id for obs in obs_list]
    ckpt_dir = prepare_obs_checkpoint_dir(
        output_path / CHECKPOINT_DIRNAME,
        manifest={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": "C1b",
            "unit_keys": unit_keys,
            "n_observations": len(unit_keys),
            "params": params,
        },
    )

    workers = resolve_n_jobs(n_jobs)
    pending: list[int] = []
    qc_by_index: dict[int, PeakDetectionQc] = {}
    errors: list[str] = []
    for index, obs in enumerate(obs_list):
        loaded = _load_c1b_checkpoint(
            ckpt_dir,
            observation_id=obs.observation_id,
            fingerprint=fingerprints[index],
            obs_dir=obs_dirs[index],
            debug_plot=debug_plot,
        )
        if loaded is not None:
            qc_by_index[index] = loaded
        else:
            pending.append(index)

    total = len(obs_list)
    done = total - len(pending)
    start_time = time.perf_counter()
    last_report = 0.0
    if progress:
        print(
            f"[confirmatory] C1b peaks+IHR: {total} observations "
            f"(channel={dataset.cardiac.channel}, "
            f"signal_type={dataset.cardiac.signal_type}, "
            f"detector={dataset.cardiac.detector}, "
            f"debug_plot={debug_plot}, n_jobs={workers}, "
            f"pending={len(pending)}, resumed={done})...",
            flush=True,
        )
        if done:
            last_report = report_progress(
                label="C1b",
                done=done,
                total=total,
                start_time=start_time,
                last_report=last_report,
                force=True,
            )

    def _mark_ok(index: int, qc: PeakDetectionQc) -> None:
        nonlocal done, last_report
        _write_c1b_checkpoint(
            ckpt_dir,
            observation_id=obs_list[index].observation_id,
            fingerprint=fingerprints[index],
            obs_dir=obs_dirs[index],
            qc=qc,
            debug_plot=debug_plot,
        )
        qc_by_index[index] = qc
        done += 1
        if progress:
            last_report = report_progress(
                label="C1b",
                done=done,
                total=total,
                start_time=start_time,
                last_report=last_report,
                force=(done == total),
            )

    if pending:
        if workers == 1 or len(pending) == 1:
            configure_blas_threads(1)
            for index in pending:
                try:
                    qc = process_one_c1b_observation(
                        obs_list[index],
                        cfg,
                        obs_dirs[index],
                        max_beat_gap_s=max_beat_gap_s,
                    )
                    _mark_ok(index, qc)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        f"{obs_list[index].observation_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    done += 1
                    if progress:
                        last_report = report_progress(
                            label="C1b",
                            done=done,
                            total=total,
                            start_time=start_time,
                            last_report=last_report,
                            force=(done == total),
                        )
        else:
            payloads = [
                (
                    index,
                    obs_list[index],
                    str(obs_dirs[index]),
                    cfg,
                    float(max_beat_gap_s),
                )
                for index in pending
            ]
            max_workers = min(workers, len(pending))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=configure_blas_threads,
                initargs=(1,),
            ) as executor:
                futures = {
                    executor.submit(_worker_c1b, payload): payload[0]
                    for payload in payloads
                }
                for future in as_completed(futures):
                    index, qc_row, err = future.result()
                    if err is None and qc_row is not None:
                        _mark_ok(index, PeakDetectionQc(**qc_row))  # type: ignore[arg-type]
                    else:
                        errors.append(err or f"{obs_list[index].observation_id}: unknown")
                        done += 1
                        if progress:
                            last_report = report_progress(
                                label="C1b",
                                done=done,
                                total=total,
                                start_time=start_time,
                                last_report=last_report,
                                force=(done == total),
                            )

    qc_rows = [qc_by_index[i] for i in range(total) if i in qc_by_index]
    qc_rows.sort(key=lambda row: row.observation_id)
    peaks_paths = [
        obs_dirs[i] / PEAKS_FILENAME
        for i in range(total)
        if i in qc_by_index and (obs_dirs[i] / PEAKS_FILENAME).is_file()
    ]

    if qc_rows:
        group_path = output_path / GROUP_PEAK_QC_FILENAME
        _atomic_dataframe_to_csv(
            pd.DataFrame([row.to_row() for row in qc_rows]),
            group_path,
        )
        if progress:
            print(f"[confirmatory] C1b wrote group QC -> {group_path}", flush=True)

    wall_time_s = time.perf_counter() - start_time
    if len(qc_rows) == total and not errors:
        atomic_write_json(
            complete_marker,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage": "C1b",
                "n_observations": total,
                "observation_ids": [q.observation_id for q in qc_rows],
                "n_jobs": workers,
                "debug_plot": debug_plot,
                "wall_time_s": wall_time_s,
            },
        )

    return {
        "n_ok": len(qc_rows),
        "n_peaks": len(peaks_paths),
        "n_peak_qc": len(qc_rows),
        "n_error": len(errors),
        "errors": errors[:20],
        "n_jobs": workers,
        "wall_time_s": wall_time_s,
        "complete": complete_marker.is_file(),
        "peaks_paths": peaks_paths,
        "qc_rows": qc_rows,
    }


def run_confirmatory_peak_detection(
    observations: Sequence[CanonicalObservation],
    dataset: ConfirmatoryDatasetConfig,
    master: ConfirmatoryMasterConfig,
    stage_root: Path,
    *,
    n_jobs: int | None = 1,
    progress: bool = True,
) -> tuple[list[Path], list[PeakDetectionQc], list[str]]:
    """Detect peaks (+ IHR) for all observations under ``stage_root`` (C1b).

    Kept for callers that expect the legacy tuple return. Prefer
    :func:`run_confirmatory_c1b` for full stage orchestration.
    """
    result = run_confirmatory_c1b(
        observations,
        dataset,
        master,
        stage_root,
        n_jobs=n_jobs,
        progress=progress,
    )
    return (
        list(result["peaks_paths"]),  # type: ignore[arg-type]
        list(result["qc_rows"]),  # type: ignore[arg-type]
        list(result["errors"]),  # type: ignore[arg-type]
    )


__all__ = [
    "CHECKPOINT_DIRNAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPLETE_MARKER_FILENAME",
    "GROUP_PEAK_QC_FILENAME",
    "PEAKS_FILENAME",
    "PEAK_QC_FILENAME",
    "PeakDetectionQc",
    "canonical_to_cardiac_observation",
    "detect_peaks_for_observation",
    "process_one_c1b_observation",
    "run_confirmatory_c1b",
    "run_confirmatory_peak_detection",
    "tc_config_for_peak_detection",
]
