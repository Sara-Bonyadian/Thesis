from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..datasets import CanonicalObservation, build_observations
from .cardiac_common import has_cardiac_channels, overlap_duration_s as _task_overlap_duration_s
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
class SignalFileInfo:
    duration_s: float
    sfreq: float
    source: str
    ch_names: tuple[str, ...]
    ch_types: tuple[str, ...]


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


def _channels_tsv_path(signal_path: Path) -> Path:
    stem = signal_path.stem
    for suffix in ("_eeg", "_ecg"):
        if stem.endswith(suffix):
            return signal_path.parent / f"{stem[: -len(suffix)]}_channels.tsv"
    return signal_path.parent / f"{stem}_channels.tsv"


def _normalize_bids_channel_type(channel_type: str) -> str:
    return str(channel_type).strip().casefold()


def _read_bids_channels_tsv(signal_path: Path) -> tuple[list[str], list[str]] | None:
    tsv_path = _channels_tsv_path(signal_path)
    if not tsv_path.is_file():
        return None
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except (OSError, ValueError):
        return None
    if "name" not in df.columns:
        return None
    ch_names = [str(name) for name in df["name"].tolist()]
    if "type" in df.columns:
        ch_types = [_normalize_bids_channel_type(value) for value in df["type"].tolist()]
    else:
        ch_types = [""] * len(ch_names)
    return ch_names, ch_types


def _read_bids_sidecar_payload(path: Path) -> dict[str, object] | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_bids_sidecar_metadata(path: Path) -> SignalMetadata | None:
    payload = _read_bids_sidecar_payload(path)
    if payload is None:
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


def _bids_sidecar_ecg_channel_count(path: Path) -> int | None:
    payload = _read_bids_sidecar_payload(path)
    if payload is None:
        return None
    count = payload.get("ECGChannelCount")
    if count is None:
        return None
    try:
        return int(count)
    except (TypeError, ValueError):
        return None


def _read_mne_metadata(path: Path, data_format: str | None = None) -> SignalMetadata | None:
    info = _read_mne_channel_info(path, data_format)
    if info is None:
        return None
    ch_names, ch_types, sfreq, duration_s = info
    _ = ch_names
    _ = ch_types
    return SignalMetadata(duration_s=duration_s, sfreq=sfreq, source="mne_header")


def _read_mne_channel_info(
    path: Path,
    data_format: str | None = None,
) -> tuple[list[str], list[str], float, float] | None:
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
    ch_names = list(raw.ch_names)
    ch_types = list(raw.get_channel_types())
    return ch_names, ch_types, sfreq, duration_s


def read_signal_file_info(
    path: Path | None,
    *,
    data_format: str | None = None,
    cache: dict[tuple[str, str], SignalFileInfo | None] | None = None,
) -> SignalFileInfo | None:
    if path is None or not path.is_file():
        return None

    cache_key = (str(path.resolve()), (data_format or "").casefold())
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    meta = _read_bids_sidecar_metadata(path)
    channel_info = _read_bids_channels_tsv(path)
    if meta is not None and channel_info is not None:
        ch_names, ch_types = channel_info
        info = SignalFileInfo(
            duration_s=meta.duration_s,
            sfreq=meta.sfreq,
            source="bids_sidecar",
            ch_names=tuple(ch_names),
            ch_types=tuple(ch_types),
        )
        if cache is not None:
            cache[cache_key] = info
        return info

    fmt = (data_format or "").casefold()
    if fmt == "bids_physio" or path.name.endswith("_physio.tsv.gz"):
        from ..core_eeg_ppg.bids_physio import read_physio_channel_info

        physio_info = read_physio_channel_info(path)
        if physio_info is not None:
            ch_names, ch_types, sfreq, duration_s = physio_info
            info = SignalFileInfo(
                duration_s=duration_s,
                sfreq=sfreq,
                source="bids_physio",
                ch_names=tuple(ch_names),
                ch_types=tuple(ch_types),
            )
            if cache is not None:
                cache[cache_key] = info
            return info

    mne_info = _read_mne_channel_info(path, data_format)
    if mne_info is None:
        if cache is not None:
            cache[cache_key] = None
        return None
    ch_names, ch_types, sfreq, duration_s = mne_info
    info = SignalFileInfo(
        duration_s=duration_s,
        sfreq=sfreq,
        source="mne_header",
        ch_names=tuple(ch_names),
        ch_types=tuple(ch_types),
    )
    if cache is not None:
        cache[cache_key] = info
    return info


def read_channel_info(path: Path | None, *, data_format: str | None = None) -> tuple[list[str], list[str]] | None:
    info = read_signal_file_info(path, data_format=data_format)
    if info is None:
        return None
    return list(info.ch_names), list(info.ch_types)


def read_signal_metadata(path: Path | None, *, data_format: str | None = None) -> SignalMetadata | None:
    if path is None or not path.is_file():
        return None
    info = read_signal_file_info(path, data_format=data_format)
    if info is not None:
        return SignalMetadata(duration_s=info.duration_s, sfreq=info.sfreq, source=info.source)
    return _read_bids_sidecar_metadata(path) or _read_mne_metadata(path, data_format)


def compute_overlap_duration_s(
    cfg: TemporalCouplingConfig,
    *,
    eeg_duration_s: float,
    cardiac_duration_s: float,
    task: str | None = None,
) -> float:
    return _task_overlap_duration_s(
        cfg,
        task=task,
        eeg_duration_s=eeg_duration_s,
        cardiac_duration_s=cardiac_duration_s,
    )


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
    eeg_ch_names: list[str] | None = None,
    cardiac_ch_names: list[str] | None = None,
    cardiac_ch_types: list[str] | None = None,
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
    if eeg_ch_names is not None:
        from .eeg_envelope import configured_roi_channels, has_usable_eeg_channels

        if not has_usable_eeg_channels(eeg_ch_names, configured_roi_channels(cfg)):
            return False, "no_eeg_channels"
    if cardiac_ch_names is not None and cardiac_ch_types is not None:
        if not has_cardiac_channels(
            cardiac_ch_names,
            cardiac_ch_types,
            signal_type=cfg.temporal_coupling.cardiac.signal_type,
        ):
            return False, "no_cardiac_channels"
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


def audit_canonical_observation(
    cfg: TemporalCouplingConfig,
    obs: CanonicalObservation,
    *,
    file_info_cache: dict[tuple[str, str], SignalFileInfo | None] | None = None,
) -> AuditRecord:
    requested_lag_max_s = cfg.temporal_coupling.cross_correlation.lag_max_s

    eeg_path = obs.eeg_path
    cardiac_path, cardiac_format = _cardiac_paths_for_observation(obs)
    eeg_exists = eeg_path.is_file()
    cardiac_exists = cardiac_path is not None and cardiac_path.is_file()

    eeg_info = read_signal_file_info(eeg_path, data_format=obs.eeg_format, cache=file_info_cache) if eeg_exists else None
    if cardiac_path is not None and eeg_exists and cardiac_path.resolve() == eeg_path.resolve():
        cardiac_info = eeg_info
    else:
        cardiac_info = (
            read_signal_file_info(cardiac_path, data_format=cardiac_format, cache=file_info_cache)
            if cardiac_exists
            else None
        )

    eeg_ch_names = list(eeg_info.ch_names) if eeg_info is not None else None
    cardiac_ch_names = list(cardiac_info.ch_names) if cardiac_info is not None else None
    cardiac_ch_types = list(cardiac_info.ch_types) if cardiac_info is not None else None
    if cardiac_ch_names is not None and cardiac_ch_types is not None:
        ecg_count = _bids_sidecar_ecg_channel_count(eeg_path) if eeg_exists else None
        if ecg_count == 0 and obs.ppg_source == "embedded_eeg":
            cardiac_ch_names = []
            cardiac_ch_types = []

    eeg_duration_s = eeg_info.duration_s if eeg_info is not None else float("nan")
    cardiac_duration_s = cardiac_info.duration_s if cardiac_info is not None else float("nan")
    eeg_sfreq = eeg_info.sfreq if eeg_info is not None else float("nan")
    cardiac_sfreq = cardiac_info.sfreq if cardiac_info is not None else float("nan")

    overlap_duration_s = compute_overlap_duration_s(
        cfg,
        eeg_duration_s=eeg_duration_s if eeg_info is not None else 0.0,
        cardiac_duration_s=cardiac_duration_s if cardiac_info is not None else 0.0,
        task=obs.task_label,
    )
    rec_lag = recommended_max_lag_s(
        overlap_duration_s=overlap_duration_s,
        requested_lag_max_s=requested_lag_max_s,
    )
    usable, skip_reason = _evaluate_usability(
        cfg,
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_meta=(
            SignalMetadata(duration_s=eeg_info.duration_s, sfreq=eeg_info.sfreq, source=eeg_info.source)
            if eeg_info is not None
            else None
        ),
        cardiac_meta=(
            SignalMetadata(
                duration_s=cardiac_info.duration_s,
                sfreq=cardiac_info.sfreq,
                source=cardiac_info.source,
            )
            if cardiac_info is not None
            else None
        ),
        overlap_duration_s=overlap_duration_s,
        eeg_ch_names=eeg_ch_names,
        cardiac_ch_names=cardiac_ch_names,
        cardiac_ch_types=cardiac_ch_types,
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
    eeg_channel_info = read_channel_info(eeg_path, data_format="eeglab") if eeg_exists else None
    cardiac_channel_info = read_channel_info(cardiac_path, data_format="eeglab") if cardiac_exists else None
    eeg_ch_names = eeg_channel_info[0] if eeg_channel_info is not None else None
    cardiac_ch_names = cardiac_channel_info[0] if cardiac_channel_info is not None else None
    cardiac_ch_types = cardiac_channel_info[1] if cardiac_channel_info is not None else None

    eeg_duration_s = eeg_meta.duration_s if eeg_meta is not None else float("nan")
    cardiac_duration_s = cardiac_meta.duration_s if cardiac_meta is not None else float("nan")
    eeg_sfreq = eeg_meta.sfreq if eeg_meta is not None else float("nan")
    cardiac_sfreq = cardiac_meta.sfreq if cardiac_meta is not None else float("nan")

    overlap_duration_s = compute_overlap_duration_s(
        cfg,
        eeg_duration_s=eeg_duration_s if eeg_meta is not None else 0.0,
        cardiac_duration_s=cardiac_duration_s if cardiac_meta is not None else 0.0,
        task=task,
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
        eeg_ch_names=eeg_ch_names,
        cardiac_ch_names=cardiac_ch_names,
        cardiac_ch_types=cardiac_ch_types,
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
    file_info_cache: dict[tuple[str, str], SignalFileInfo | None] = {}

    if observations:
        n_total = len(observations)
        print(f"[temporal_coupling] stage=0: auditing {n_total} observations...")
        for idx, obs in enumerate(observations, start=1):
            records.append(
                audit_canonical_observation(cfg, obs, file_info_cache=file_info_cache)
            )
            if idx == 1 or idx == n_total or idx % 25 == 0:
                print(f"[temporal_coupling] stage=0 progress: {idx}/{n_total}")
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
