"""Observation-level raw-data audit for confirmatory Stage C0.

Discovers file presence, EEG/cardiac durations, raw overlap, cardiac channel
availability, and recommended max lag. Does not run beat detection and does
not write to exploratory temporal-coupling derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..datasets import CanonicalObservation
from ..temporal_coupling.cardiac_common import (
    has_cardiac_channels,
    infer_signal_type_from_channel,
)
from ..temporal_coupling.data_audit import (
    SignalFileInfo,
    SignalMetadata,
    read_signal_file_info,
    recommended_max_lag_s,
)
from .protocol_audit import _participant_id, _session_id, protocol_spec

DATA_AUDIT_FILENAME = "data_audit.csv"

# Preferred cardiac channel names when multiple PPG/ECG candidates exist.
_PREFERRED_CARDIAC_CHANNELS = (
    "photosensor",
    "ppg",
    "ecg",
    "ekg",
)


@dataclass(frozen=True)
class ConfirmatoryAuditRecord:
    """One observation's C0 raw-data audit result."""

    dataset_id: str
    observation_id: str
    participant_id: str
    session_id: str
    subject_id: str
    condition: str
    task: str
    eeg_file: str
    eeg_format: str
    cardiac_file: str
    cardiac_format: str
    ppg_source: str
    eeg_exists: bool
    cardiac_exists: bool
    eeg_duration_s: float
    cardiac_duration_s: float
    raw_overlap_s: float
    overlap_duration_s: float
    eeg_sfreq: float
    cardiac_sfreq: float
    available_cardiac_channel: str
    cardiac_signal_type: str
    recommended_max_lag_s: float
    usable: bool
    exclusion_reason: str

    def to_row(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "observation_id": self.observation_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "condition": self.condition,
            "task": self.task,
            "eeg_file": self.eeg_file,
            "eeg_format": self.eeg_format,
            "cardiac_file": self.cardiac_file,
            "cardiac_format": self.cardiac_format,
            "ppg_source": self.ppg_source,
            "eeg_exists": self.eeg_exists,
            "cardiac_exists": self.cardiac_exists,
            "eeg_duration_s": self.eeg_duration_s,
            "cardiac_duration_s": self.cardiac_duration_s,
            "raw_overlap_s": self.raw_overlap_s,
            "overlap_duration_s": self.overlap_duration_s,
            "eeg_sfreq": self.eeg_sfreq,
            "cardiac_sfreq": self.cardiac_sfreq,
            "available_cardiac_channel": self.available_cardiac_channel,
            "cardiac_signal_type": self.cardiac_signal_type,
            "recommended_max_lag_s": self.recommended_max_lag_s,
            "usable": self.usable,
            "exclusion_reason": self.exclusion_reason,
            # Compatibility with Stage 0 / enrich_eligibility_metadata_from_csv.
            "skip_reason": self.exclusion_reason,
        }


def _cardiac_paths_for_observation(obs: CanonicalObservation) -> tuple[Path | None, str]:
    if obs.ppg_source == "embedded_eeg":
        return obs.eeg_path, obs.eeg_format
    if obs.ppg_path is not None and obs.ppg_path.is_file():
        return obs.ppg_path, obs.ppg_format or "eeglab"
    return None, ""


def _signal_type_preference(dataset_id: str) -> str:
    modality = protocol_spec(dataset_id).cardiac_modality.casefold()
    tokens = {part.strip() for part in modality.replace(";", ",").split(",") if part.strip()}
    if "ppg" in tokens and "ecg" not in tokens:
        return "ppg"
    if "ecg" in tokens and "ppg" not in tokens:
        return "ecg"
    if "ppg" in tokens:
        return "ppg"
    if "ecg" in tokens:
        return "ecg"
    return "auto"


def _raw_overlap_s(*, eeg_duration_s: float, cardiac_duration_s: float) -> float:
    if not math.isfinite(eeg_duration_s) or not math.isfinite(cardiac_duration_s):
        return 0.0
    if eeg_duration_s <= 0 or cardiac_duration_s <= 0:
        return 0.0
    return float(min(eeg_duration_s, cardiac_duration_s))


def _list_cardiac_candidates(
    ch_names: Sequence[str],
    ch_types: Sequence[str],
    *,
    signal_type: str,
) -> list[tuple[str, str]]:
    st = signal_type.strip().casefold()
    matches: list[tuple[str, str]] = []
    for name, ch_type in zip(ch_names, ch_types, strict=False):
        inferred = infer_signal_type_from_channel(name, ch_type)
        if st in {"", "auto"}:
            if inferred in {"ecg", "ppg"}:
                matches.append((name, inferred))
        elif st == inferred:
            matches.append((name, inferred))
    if not matches and len(ch_names) == 1 and st in {"", "auto", "ecg", "ppg"}:
        alone = ch_names[0]
        alone_type = ch_types[0] if ch_types else ""
        inferred = infer_signal_type_from_channel(alone, alone_type)
        if inferred == "unknown":
            inferred = st if st in {"ecg", "ppg"} else "unknown"
        matches.append((alone, inferred))
    return matches


def _select_cardiac_channel(
    ch_names: Sequence[str] | None,
    ch_types: Sequence[str] | None,
    *,
    signal_type: str,
) -> tuple[str, str]:
    if not ch_names or ch_types is None:
        return "", ""
    candidates = _list_cardiac_candidates(ch_names, ch_types, signal_type=signal_type)
    if not candidates:
        return "", ""
    preferred = {name.casefold() for name in _PREFERRED_CARDIAC_CHANNELS}
    for name, inferred in candidates:
        if name.casefold() in preferred:
            return name, inferred
    return candidates[0]


def _evaluate_usability(
    *,
    eeg_exists: bool,
    cardiac_exists: bool,
    eeg_meta: SignalMetadata | None,
    cardiac_meta: SignalMetadata | None,
    overlap_duration_s: float,
    eeg_ch_names: list[str] | None,
    cardiac_ch_names: list[str] | None,
    cardiac_ch_types: list[str] | None,
    signal_type: str,
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
    if eeg_ch_names is not None and not eeg_ch_names:
        return False, "no_eeg_channels"
    if cardiac_ch_names is not None and cardiac_ch_types is not None:
        if not has_cardiac_channels(
            cardiac_ch_names,
            cardiac_ch_types,
            signal_type=signal_type,
        ):
            return False, "no_cardiac_channels"
    if overlap_duration_s <= 0:
        return False, "no_temporal_overlap"
    return True, ""


def audit_observation(
    obs: CanonicalObservation,
    *,
    lag_max_s: float,
    signal_type: str | None = None,
    file_info_cache: dict[tuple[str, str], SignalFileInfo | None] | None = None,
) -> ConfirmatoryAuditRecord:
    """Inspect one observation's EEG and cardiac files for C0."""
    preferred_type = signal_type or _signal_type_preference(obs.dataset_id)
    eeg_path = obs.eeg_path
    cardiac_path, cardiac_format = _cardiac_paths_for_observation(obs)
    eeg_exists = eeg_path.is_file()
    cardiac_exists = cardiac_path is not None and cardiac_path.is_file()

    eeg_info = (
        read_signal_file_info(eeg_path, data_format=obs.eeg_format, cache=file_info_cache)
        if eeg_exists
        else None
    )
    if (
        cardiac_path is not None
        and eeg_exists
        and cardiac_path.resolve() == eeg_path.resolve()
    ):
        cardiac_info = eeg_info
    else:
        cardiac_info = (
            read_signal_file_info(
                cardiac_path, data_format=cardiac_format, cache=file_info_cache
            )
            if cardiac_exists
            else None
        )

    eeg_ch_names = list(eeg_info.ch_names) if eeg_info is not None else None
    cardiac_ch_names = list(cardiac_info.ch_names) if cardiac_info is not None else None
    cardiac_ch_types = list(cardiac_info.ch_types) if cardiac_info is not None else None

    eeg_duration_s = eeg_info.duration_s if eeg_info is not None else float("nan")
    cardiac_duration_s = (
        cardiac_info.duration_s if cardiac_info is not None else float("nan")
    )
    eeg_sfreq = eeg_info.sfreq if eeg_info is not None else float("nan")
    cardiac_sfreq = cardiac_info.sfreq if cardiac_info is not None else float("nan")

    overlap = _raw_overlap_s(
        eeg_duration_s=eeg_duration_s if eeg_info is not None else 0.0,
        cardiac_duration_s=cardiac_duration_s if cardiac_info is not None else 0.0,
    )
    channel, inferred_type = _select_cardiac_channel(
        cardiac_ch_names,
        cardiac_ch_types,
        signal_type=preferred_type,
    )
    rec_lag = recommended_max_lag_s(
        overlap_duration_s=overlap,
        requested_lag_max_s=float(lag_max_s),
    )
    usable, exclusion_reason = _evaluate_usability(
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_meta=(
            SignalMetadata(
                duration_s=eeg_info.duration_s,
                sfreq=eeg_info.sfreq,
                source=eeg_info.source,
            )
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
        overlap_duration_s=overlap,
        eeg_ch_names=eeg_ch_names,
        cardiac_ch_names=cardiac_ch_names,
        cardiac_ch_types=cardiac_ch_types,
        signal_type=preferred_type,
    )

    return ConfirmatoryAuditRecord(
        dataset_id=obs.dataset_id,
        observation_id=obs.observation_id,
        participant_id=_participant_id(obs),
        session_id=_session_id(obs),
        subject_id=obs.subject_id,
        condition=obs.condition_label,
        task=obs.task_label,
        eeg_file=str(eeg_path),
        eeg_format=obs.eeg_format,
        cardiac_file=str(cardiac_path) if cardiac_path is not None else "",
        cardiac_format=cardiac_format,
        ppg_source=obs.ppg_source,
        eeg_exists=eeg_exists,
        cardiac_exists=cardiac_exists,
        eeg_duration_s=eeg_duration_s,
        cardiac_duration_s=cardiac_duration_s,
        raw_overlap_s=overlap,
        overlap_duration_s=overlap,
        eeg_sfreq=eeg_sfreq,
        cardiac_sfreq=cardiac_sfreq,
        available_cardiac_channel=channel,
        cardiac_signal_type=inferred_type,
        recommended_max_lag_s=rec_lag,
        usable=usable,
        exclusion_reason=exclusion_reason,
    )


def run_confirmatory_data_audit(
    observations: Iterable[CanonicalObservation],
    output_dir: str | Path,
    *,
    lag_max_s: float,
) -> tuple[Path, list[ConfirmatoryAuditRecord]]:
    """Write ``data_audit.csv`` under the confirmatory C0 directory."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    out_path = target / DATA_AUDIT_FILENAME
    records: list[ConfirmatoryAuditRecord] = []
    cache: dict[tuple[str, str], SignalFileInfo | None] = {}
    obs_list = list(observations)
    n_total = len(obs_list)
    if n_total:
        print(f"[confirmatory] C0 raw-data audit: {n_total} observations...")
    for idx, obs in enumerate(obs_list, start=1):
        records.append(
            audit_observation(obs, lag_max_s=lag_max_s, file_info_cache=cache)
        )
        if idx == 1 or idx == n_total or idx % 25 == 0:
            print(f"[confirmatory] C0 raw-data audit progress: {idx}/{n_total}")

    df = pd.DataFrame([record.to_row() for record in records])
    df.to_csv(out_path, index=False)
    n_usable = int(df["usable"].sum()) if not df.empty else 0
    print(f"[confirmatory] C0 wrote {out_path} usable={n_usable}/{n_total}")
    return out_path, records


__all__ = [
    "DATA_AUDIT_FILENAME",
    "ConfirmatoryAuditRecord",
    "audit_observation",
    "run_confirmatory_data_audit",
]
