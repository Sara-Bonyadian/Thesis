from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..datasets import CanonicalObservation, build_observations
from .config import TemporalCouplingConfig
from .paths import group_output_dir  # re-exported for stage modules

AUDIT_FILENAME = "data_audit.csv"

__all__ = [
    "AUDIT_FILENAME",
    "audit_output_path",
    "compute_overlap_duration_s",
    "group_output_dir",
    "list_configured_observations",
    "read_signal_metadata",
    "recommended_max_lag_s",
    "run_stage0_audit",
]


@dataclass(frozen=True)
class SignalMetadata:
    duration_s: float
    sfreq: float
    source: str


@dataclass(frozen=True)
class AuditRecord:
    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    eeg_file: str
    eeg_format: str
    cardiac_file: str
    cardiac_format: str
    ppg_source: str
    eeg_exists: bool
    cardiac_exists: bool
    eeg_duration_s: float
    cardiac_duration_s: float
    overlap_duration_s: float
    eeg_sfreq: float
    cardiac_sfreq: float
    usable: bool
    skip_reason: str
    recommended_max_lag_s: float

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "task": self.task,
            "condition": self.condition,
            "observation_id": self.observation_id,
            "eeg_file": self.eeg_file,
            "eeg_format": self.eeg_format,
            "cardiac_file": self.cardiac_file,
            "cardiac_format": self.cardiac_format,
            "ppg_source": self.ppg_source,
            "eeg_exists": self.eeg_exists,
            "cardiac_exists": self.cardiac_exists,
            "eeg_duration_s": self.eeg_duration_s,
            "cardiac_duration_s": self.cardiac_duration_s,
            "overlap_duration_s": self.overlap_duration_s,
            "eeg_sfreq": self.eeg_sfreq,
            "cardiac_sfreq": self.cardiac_sfreq,
            "usable": self.usable,
            "skip_reason": self.skip_reason,
            "recommended_max_lag_s": self.recommended_max_lag_s,
        }


def audit_output_path(cfg: TemporalCouplingConfig) -> Path:
    return group_output_dir(cfg) / AUDIT_FILENAME


def _resolve_dataset_root(raw_root: Path, dataset_id: str) -> Path | None:
    raw_root = Path(raw_root)
    if raw_root.name.casefold() == dataset_id.casefold() and raw_root.is_dir():
        return raw_root
    candidate = raw_root / dataset_id
    if candidate.is_dir():
        return candidate
    if raw_root.is_dir():
        return raw_root
    return None


def _find_signal_file(dataset_root: Path, subject_id: str, task: str, modality: str) -> Path | None:
    """BIDS EEGLAB fallback used by ds003838."""
    modality_dir = dataset_root / subject_id / modality
    if not modality_dir.is_dir():
        return None

    exact = modality_dir / f"{subject_id}_task-{task}_{modality}.set"
    if exact.is_file():
        return exact

    matches = sorted(modality_dir.glob(f"*_task-{task}_{modality}.set"))
    return matches[0] if matches else None


def _read_bids_sidecar_metadata(path: Path) -> SignalMetadata | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    duration = payload.get("RecordingDuration")
    sfreq = payload.get("SamplingFrequency")
    if duration is None or sfreq is None:
        return None

    duration_s = float(duration)
    sfreq_hz = float(sfreq)
    if duration_s <= 0 or sfreq_hz <= 0:
        return None
    return SignalMetadata(duration_s=duration_s, sfreq=sfreq_hz, source="bids_json")


def _read_mne_metadata(path: Path, data_format: str | None = None) -> SignalMetadata | None:
    try:
        import mne
    except ImportError:
        return None

    fmt = (data_format or "").casefold()
    suffix = path.suffix.casefold()
    if not fmt:
        if suffix == ".vhdr":
            fmt = "brainvision"
        elif suffix == ".set":
            fmt = "eeglab"
        elif suffix == ".edf":
            fmt = "edf"

    try:
        if fmt == "brainvision":
            raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose=False)
        elif fmt == "edf":
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
        else:
            raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose=False)
    except Exception:
        return None

    sfreq = float(raw.info["sfreq"])
    duration_s = float(raw.n_times) / sfreq if sfreq > 0 else 0.0
    if duration_s <= 0 or sfreq <= 0:
        return None
    return SignalMetadata(duration_s=duration_s, sfreq=sfreq, source="mne_header")


def read_signal_metadata(path: Path | None, *, data_format: str | None = None) -> SignalMetadata | None:
    if path is None or not path.is_file():
        return None
    return _read_bids_sidecar_metadata(path) or _read_mne_metadata(path, data_format)


def _overlap_window(cfg: TemporalCouplingConfig, eeg_duration_s: float, cardiac_duration_s: float) -> tuple[float, float]:
    max_duration = min(eeg_duration_s, cardiac_duration_s)
    if max_duration <= 0:
        return 0.0, 0.0

    start_s = cfg.ppg.start_time_s
    end_s = cfg.ppg.end_time_s
    if start_s is None and end_s is None:
        return 0.0, max_duration

    window_start = 0.0 if start_s is None else max(0.0, float(start_s))
    window_end = max_duration if end_s is None else min(float(end_s), max_duration)
    if window_end <= window_start:
        return window_start, window_start
    return window_start, window_end


def compute_overlap_duration_s(
    cfg: TemporalCouplingConfig,
    *,
    eeg_duration_s: float,
    cardiac_duration_s: float,
) -> float:
    start_s, end_s = _overlap_window(cfg, eeg_duration_s, cardiac_duration_s)
    return max(0.0, end_s - start_s)


def recommended_max_lag_s(
    *,
    overlap_duration_s: float,
    requested_lag_max_s: float,
) -> float:
    if overlap_duration_s <= 0:
        return 0.0
    adaptive = math.floor(overlap_duration_s / 3.0)
    return float(min(requested_lag_max_s, adaptive))


def _evaluate_usability(
    cfg: TemporalCouplingConfig,
    *,
    eeg_exists: bool,
    cardiac_exists: bool,
    eeg_meta: SignalMetadata | None,
    cardiac_meta: SignalMetadata | None,
    overlap_duration_s: float,
) -> tuple[bool, str]:
    if not eeg_exists:
        return False, "eeg_file_missing"
    if not cardiac_exists:
        return False, "cardiac_file_missing"
    if eeg_meta is None:
        return False, "eeg_metadata_unreadable"
    if cardiac_meta is None:
        return False, "cardiac_metadata_unreadable"
    if eeg_meta.duration_s <= 0:
        return False, "eeg_duration_invalid"
    if cardiac_meta.duration_s <= 0:
        return False, "cardiac_duration_invalid"
    if overlap_duration_s <= 0:
        return False, "no_temporal_overlap"
    min_overlap_s = cfg.temporal_coupling.audit.min_overlap_s
    if overlap_duration_s < min_overlap_s:
        return False, f"overlap_too_short ({overlap_duration_s:.1f}s < {min_overlap_s:.1f}s)"
    return True, ""


def _cardiac_paths_for_observation(obs: CanonicalObservation) -> tuple[Path | None, str]:
    if obs.ppg_source == "embedded_eeg":
        return obs.eeg_path, obs.eeg_format
    if obs.ppg_path is not None and obs.ppg_path.is_file():
        return obs.ppg_path, obs.ppg_format or "eeglab"
    return None, ""


def audit_canonical_observation(cfg: TemporalCouplingConfig, obs: CanonicalObservation) -> AuditRecord:
    requested_lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s

    eeg_path = obs.eeg_path
    cardiac_path, cardiac_format = _cardiac_paths_for_observation(obs)
    eeg_exists = eeg_path.is_file()
    cardiac_exists = cardiac_path is not None and cardiac_path.is_file()

    eeg_meta = read_signal_metadata(eeg_path, data_format=obs.eeg_format) if eeg_exists else None
    cardiac_meta = (
        read_signal_metadata(cardiac_path, data_format=cardiac_format) if cardiac_exists else None
    )

    eeg_duration_s = eeg_meta.duration_s if eeg_meta is not None else float("nan")
    cardiac_duration_s = cardiac_meta.duration_s if cardiac_meta is not None else float("nan")
    eeg_sfreq = eeg_meta.sfreq if eeg_meta is not None else float("nan")
    cardiac_sfreq = cardiac_meta.sfreq if cardiac_meta is not None else float("nan")

    overlap_duration_s = compute_overlap_duration_s(
        cfg,
        eeg_duration_s=eeg_duration_s if eeg_meta is not None else 0.0,
        cardiac_duration_s=cardiac_duration_s if cardiac_meta is not None else 0.0,
    )
    rec_lag = recommended_max_lag_s(
        overlap_duration_s=overlap_duration_s,
        requested_lag_max_s=requested_lag_max_s,
    )
    usable, skip_reason = _evaluate_usability(
        cfg,
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_meta=eeg_meta,
        cardiac_meta=cardiac_meta,
        overlap_duration_s=overlap_duration_s,
    )

    return AuditRecord(
        dataset_id=obs.dataset_id,
        subject_id=obs.subject_id,
        task=obs.task_label,
        condition=obs.condition_label,
        observation_id=obs.observation_id,
        eeg_file=str(eeg_path),
        eeg_format=obs.eeg_format,
        cardiac_file=str(cardiac_path) if cardiac_path is not None else "",
        cardiac_format=cardiac_format,
        ppg_source=obs.ppg_source,
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_duration_s=eeg_duration_s,
        cardiac_duration_s=cardiac_duration_s,
        overlap_duration_s=overlap_duration_s,
        eeg_sfreq=eeg_sfreq,
        cardiac_sfreq=cardiac_sfreq,
        usable=usable,
        skip_reason=skip_reason,
        recommended_max_lag_s=rec_lag,
    )


def audit_observation(
    cfg: TemporalCouplingConfig,
    *,
    subject_id: str,
    task: str,
    dataset_root: Path | None,
) -> AuditRecord:
    """Legacy BIDS-only audit path retained for ds003838 compatibility."""
    dataset_id = cfg.dataset_id
    observation_id = f"{dataset_id}-{subject_id}-task-{task}"
    requested_lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s

    eeg_path: Path | None = None
    cardiac_path: Path | None = None
    if dataset_root is not None:
        eeg_path = _find_signal_file(dataset_root, subject_id, task, "eeg")
        cardiac_path = _find_signal_file(dataset_root, subject_id, task, "ecg")

    eeg_exists = eeg_path is not None and eeg_path.is_file()
    cardiac_exists = cardiac_path is not None and cardiac_path.is_file()

    eeg_meta = read_signal_metadata(eeg_path, data_format="eeglab") if eeg_exists else None
    cardiac_meta = read_signal_metadata(cardiac_path, data_format="eeglab") if cardiac_exists else None

    eeg_duration_s = eeg_meta.duration_s if eeg_meta is not None else float("nan")
    cardiac_duration_s = cardiac_meta.duration_s if cardiac_meta is not None else float("nan")
    eeg_sfreq = eeg_meta.sfreq if eeg_meta is not None else float("nan")
    cardiac_sfreq = cardiac_meta.sfreq if cardiac_meta is not None else float("nan")

    overlap_duration_s = compute_overlap_duration_s(
        cfg,
        eeg_duration_s=eeg_duration_s if eeg_meta is not None else 0.0,
        cardiac_duration_s=cardiac_duration_s if cardiac_meta is not None else 0.0,
    )
    rec_lag = recommended_max_lag_s(
        overlap_duration_s=overlap_duration_s,
        requested_lag_max_s=requested_lag_max_s,
    )
    usable, skip_reason = _evaluate_usability(
        cfg,
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_meta=eeg_meta,
        cardiac_meta=cardiac_meta,
        overlap_duration_s=overlap_duration_s,
    )

    return AuditRecord(
        dataset_id=dataset_id,
        subject_id=subject_id,
        task=task,
        condition=task,
        observation_id=observation_id,
        eeg_file=str(eeg_path) if eeg_path is not None else "",
        eeg_format="eeglab",
        cardiac_file=str(cardiac_path) if cardiac_path is not None else "",
        cardiac_format="eeglab",
        ppg_source="separate_ecg",
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_duration_s=eeg_duration_s,
        cardiac_duration_s=cardiac_duration_s,
        overlap_duration_s=overlap_duration_s,
        eeg_sfreq=eeg_sfreq,
        cardiac_sfreq=cardiac_sfreq,
        usable=usable,
        skip_reason=skip_reason,
        recommended_max_lag_s=rec_lag,
    )


def list_configured_observations(cfg: TemporalCouplingConfig) -> list[CanonicalObservation]:
    return build_observations(
        cfg.dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects or None,
        tasks=cfg.tasks or None,
        conditions=cfg.conditions or None,
        sessions=cfg.sessions or None,
        hiit_partition_mode=cfg.hiit_partition_mode,
    )


def run_stage0_audit(cfg: TemporalCouplingConfig) -> Path:
    observations = list_configured_observations(cfg)
    records: list[AuditRecord] = []

    if observations:
        for obs in observations:
            records.append(audit_canonical_observation(cfg, obs))
    else:
        dataset_root = _resolve_dataset_root(cfg.paths.raw_root, cfg.dataset_id)
        tasks = cfg.tasks or ["rest"]
        subjects = cfg.subjects or []
        if subjects:
            pairs = [(subject_id, task) for subject_id in subjects for task in tasks]
        else:
            pairs = []
        for subject_id, task in pairs:
            records.append(
                audit_observation(cfg, subject_id=subject_id, task=task, dataset_root=dataset_root)
            )

    out_path = audit_output_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record.to_row() for record in records])
    df.to_csv(out_path, index=False)

    n_usable = int(df["usable"].sum()) if not df.empty else 0
    n_total = len(df)
    print(f"[temporal_coupling] stage=0 wrote {out_path}")
    print(f"[temporal_coupling] audit summary: usable={n_usable}/{n_total}")
    for record in records:
        status = "OK" if record.usable else f"SKIP ({record.skip_reason})"
        label = record.condition if record.condition != record.task else record.task
        print(
            f"[temporal_coupling]   {record.subject_id} task={record.task} condition={label}: "
            f"overlap={record.overlap_duration_s:.1f}s "
            f"recommended_max_lag_s={record.recommended_max_lag_s:.0f} -> {status}"
        )
    return out_path
