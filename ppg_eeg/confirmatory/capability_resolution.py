"""Resolve declared YAML capabilities against observed evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..datasets import CanonicalObservation
from ..temporal_coupling.data_audit import read_signal_file_info
from .config import EXPECTED_BANDS_HZ, ConfirmatoryDatasetConfig
from .dataset_contracts import DatasetCapabilities
from .reason_codes import (
    INSUFFICIENT_COMMON_MONTAGE,
    INSUFFICIENT_DURATION,
    MISSING_CONDITION_MAPPING,
    MISSING_EVENT_SERIES,
    MISSING_PAIRED_OBSERVATION,
    MISSING_REQUIRED_BAND,
    MISSING_REQUIRED_MODALITY,
    MISSING_SENSOR_LOCATIONS,
    STRUCTURED_NC_FIELDS,
    TOPOGRAPHY_NOT_SUPPORTED,
    with_structured_nc_fields,
)

CAPABILITY_RESOLUTION_FILENAME = "capability_resolution.csv"
# Panel F spatial maps need more than "an EEG file exists": enough scalp
# channels must join MNE standard_1020 so a common montage can be drawn.
MIN_STANDARD_1020_CHANNELS = 8
_EEG_CHANNEL_TYPES = frozenset({"eeg"})
_LOW_GAMMA_HIGH_HZ = float(EXPECTED_BANDS_HZ["low_gamma"][1])
_STANDARD_1020_KEYS: frozenset[str] | None = None


@dataclass(frozen=True)
class PanelFChannelEvidence:
    n_eeg_channels: int
    n_standard_1020: int
    sfreq_hz: float | None
    nyquist_hz: float | None
    source: str
    supports_topography: bool
    supports_gamma: bool
    topography_reason_code: str
    gamma_reason_code: str

    def evidence_str(self) -> str:
        sfreq = "none" if self.sfreq_hz is None else f"{self.sfreq_hz:g}"
        nyquist = "none" if self.nyquist_hz is None else f"{self.nyquist_hz:g}"
        return (
            f"n_eeg_channels={self.n_eeg_channels}; "
            f"n_standard_1020={self.n_standard_1020}; "
            f"sfreq_hz={sfreq}; nyquist_hz={nyquist}; "
            f"source={self.source or 'none'}"
        )


@dataclass(frozen=True)
class CapabilityResolutionRow:
    capability: str
    declared_value: str
    observed_evidence: str
    effective_value: str
    reason_code: str
    affected_stage: str

    def to_row(self) -> dict[str, str]:
        return asdict(self)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _standard_1020_keys() -> frozenset[str]:
    global _STANDARD_1020_KEYS
    if _STANDARD_1020_KEYS is not None:
        return _STANDARD_1020_KEYS
    try:
        import mne

        pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        _STANDARD_1020_KEYS = frozenset(str(name).casefold() for name in pos)
    except Exception:
        _STANDARD_1020_KEYS = frozenset()
    return _STANDARD_1020_KEYS


def _eeg_channel_names(
    ch_names: Sequence[str],
    ch_types: Sequence[str],
) -> tuple[str, ...]:
    names = [str(name).strip() for name in ch_names]
    types = [str(typ).strip().casefold() for typ in ch_types]
    if len(types) < len(names):
        types.extend([""] * (len(names) - len(types)))
    out: list[str] = []
    montage = _standard_1020_keys()
    for name, typ in zip(names, types):
        if not name:
            continue
        if typ in _EEG_CHANNEL_TYPES:
            out.append(name)
            continue
        # Sidecars sometimes omit type; 10-20 membership is still scalp EEG.
        if not typ and name.casefold() in montage:
            out.append(name)
    return tuple(out)


def _n_standard_1020(channel_names: Sequence[str]) -> int:
    keys = _standard_1020_keys()
    if not keys:
        return 0
    return sum(1 for name in channel_names if str(name).casefold() in keys)


def _as_existing_path(value: object) -> Path | None:
    if not isinstance(value, Path):
        return None
    try:
        if value.name.startswith("._"):
            return None
        if not value.is_file():
            return None
    except OSError:
        return None
    return value


def inspect_panel_f_channel_evidence(
    observations: Sequence[CanonicalObservation],
) -> PanelFChannelEvidence:
    """Inspect one readable EEG recording for Panel F topography/gamma gates."""
    info = None
    for observation in observations:
        path = _as_existing_path(getattr(observation, "eeg_path", None))
        if path is None:
            continue
        try:
            info = read_signal_file_info(
                path,
                data_format=getattr(observation, "eeg_format", None),
            )
        except (OSError, TypeError, ValueError):
            info = None
        if info is not None:
            break

    if info is None:
        return PanelFChannelEvidence(
            n_eeg_channels=0,
            n_standard_1020=0,
            sfreq_hz=None,
            nyquist_hz=None,
            source="",
            supports_topography=False,
            supports_gamma=False,
            topography_reason_code=MISSING_SENSOR_LOCATIONS,
            gamma_reason_code=MISSING_REQUIRED_BAND,
        )

    eeg_names = _eeg_channel_names(info.ch_names, info.ch_types)
    n_1020 = _n_standard_1020(eeg_names)
    sfreq = float(info.sfreq) if info.sfreq and info.sfreq > 0 else None
    nyquist = (sfreq / 2.0) if sfreq is not None else None
    has_topo = n_1020 >= MIN_STANDARD_1020_CHANNELS
    has_gamma = bool(eeg_names) and nyquist is not None and nyquist >= _LOW_GAMMA_HIGH_HZ
    if has_topo:
        topo_reason = ""
    elif n_1020 > 0:
        topo_reason = INSUFFICIENT_COMMON_MONTAGE
    else:
        topo_reason = MISSING_SENSOR_LOCATIONS
    gamma_reason = "" if has_gamma else MISSING_REQUIRED_BAND
    return PanelFChannelEvidence(
        n_eeg_channels=len(eeg_names),
        n_standard_1020=n_1020,
        sfreq_hz=sfreq,
        nyquist_hz=nyquist,
        source=str(info.source or ""),
        supports_topography=has_topo,
        supports_gamma=has_gamma,
        topography_reason_code=topo_reason,
        gamma_reason_code=gamma_reason,
    )


def resolve_effective_capabilities(
    dataset: ConfirmatoryDatasetConfig,
    observations: Sequence[CanonicalObservation],
    *,
    data_audit_rows: Sequence[Mapping[str, object]] = (),
) -> tuple[DatasetCapabilities, list[CapabilityResolutionRow]]:
    """Derive effective capabilities from declared YAML + observed evidence.

    A YAML declaration alone never makes an analysis computable.
    """
    declared = dataset.capabilities
    obs = list(observations)
    rows: list[CapabilityResolutionRow] = []

    has_eeg_files = any(o.eeg_path.is_file() for o in obs) if obs else False
    has_hr_files = False
    if obs:
        for o in obs:
            if o.ppg_source == "embedded_eeg" and o.eeg_path.is_file():
                has_hr_files = True
                break
            if o.ppg_path is not None and o.ppg_path.is_file():
                has_hr_files = True
                break
    if data_audit_rows:
        has_eeg_files = has_eeg_files or any(
            str(r.get("eeg_exists", "")).casefold() in {"1", "true", "yes"}
            for r in data_audit_rows
        )
        has_hr_files = has_hr_files or any(
            str(r.get("cardiac_exists", "")).casefold() in {"1", "true", "yes"}
            for r in data_audit_rows
        )

    semantics = {
        k.casefold(): v for k, v in dataset.normalization.condition_semantics.items()
    }
    observed_semantics = [
        semantics.get(o.condition_label.casefold(), {}) for o in obs
    ]
    has_low = any(
        str(v.get("state", "")).casefold() == "state_low"
        for v in observed_semantics
    )
    has_high = any(
        str(v.get("state", "")).casefold() == "state_high"
        for v in observed_semantics
    )
    has_pre = any(
        str(v.get("time", "")).casefold() == "time_pre"
        for v in observed_semantics
    )
    has_post = any(
        str(v.get("time", "")).casefold() == "time_post"
        for v in observed_semantics
    )

    signal_types = {
        str(r.get("cardiac_signal_type", "")).casefold()
        for r in data_audit_rows
        if str(r.get("cardiac_signal_type", "")).strip()
    }
    observed_modality = declared.cardiac_modality
    if "ppg" in signal_types and "ecg" in signal_types:
        observed_modality = "both"
    elif "ppg" in signal_types:
        observed_modality = "ppg"
    elif "ecg" in signal_types:
        observed_modality = "ecg"
    elif not has_hr_files:
        observed_modality = "neither"

    overlaps = [
        float(r.get("raw_overlap_s") or r.get("overlap_duration_s") or 0)
        for r in data_audit_rows
        if str(r.get("raw_overlap_s") or r.get("overlap_duration_s") or "").strip()
    ]
    max_overlap = max(overlaps) if overlaps else None
    supports_d240_obs = max_overlap is not None and max_overlap >= 240
    supports_d180_obs = max_overlap is not None and max_overlap >= 180
    panel_f = inspect_panel_f_channel_evidence(obs)

    effective = DatasetCapabilities(
        has_eeg=bool(declared.has_eeg and has_eeg_files),
        has_hr=bool(declared.has_hr and has_hr_files),
        cardiac_modality=observed_modality if has_hr_files else "neither",
        has_ecg_r_peaks=bool(
            declared.has_ecg_r_peaks and observed_modality in {"ecg", "both"}
        ),
        has_ppg_peaks=bool(
            declared.has_ppg_peaks and observed_modality in {"ppg", "both"}
        ),
        has_low_high_task_pair=bool(
            declared.has_low_high_task_pair and has_low and has_high
        ),
        has_pre_post_pair=bool(declared.has_pre_post_pair and has_pre and has_post),
        has_behavior=bool(declared.has_behavior),
        supports_d120=bool(
            declared.supports_d120
            and (max_overlap is None or max_overlap >= 120)
        ),
        supports_d180=bool(declared.supports_d180 and (supports_d180_obs if max_overlap is not None else False)),
        supports_d240=bool(declared.supports_d240 and (supports_d240_obs if max_overlap is not None else False)),
        supports_topography=bool(
            declared.supports_topography and has_eeg_files and panel_f.supports_topography
        ),
        supports_gamma=bool(
            declared.supports_gamma and has_eeg_files and panel_f.supports_gamma
        ),
        sensitivity_only=bool(declared.sensitivity_only),
        has_artifact_controls=bool(declared.has_artifact_controls and has_hr_files),
    )

    def add(
        capability: str,
        declared_value: object,
        observed: str,
        effective_value: object,
        reason_code: str,
        stage: str,
    ) -> None:
        rows.append(
            CapabilityResolutionRow(
                capability=capability,
                declared_value=str(declared_value).casefold()
                if isinstance(declared_value, bool)
                else str(declared_value),
                observed_evidence=observed,
                effective_value=str(effective_value).casefold()
                if isinstance(effective_value, bool)
                else str(effective_value),
                reason_code=reason_code,
                affected_stage=stage,
            )
        )

    add(
        "has_eeg",
        declared.has_eeg,
        f"eeg_files_present={_bool_str(has_eeg_files)}; n_obs={len(obs)}",
        effective.has_eeg,
        "" if effective.has_eeg else MISSING_REQUIRED_MODALITY,
        "C0/C1a",
    )
    add(
        "has_hr",
        declared.has_hr,
        f"cardiac_files_present={_bool_str(has_hr_files)}; signal_types={sorted(signal_types)}",
        effective.has_hr,
        "" if effective.has_hr else MISSING_REQUIRED_MODALITY,
        "C0/C1b",
    )
    add(
        "cardiac_modality",
        declared.cardiac_modality,
        f"observed_signal_types={sorted(signal_types) or ['none']}",
        effective.cardiac_modality,
        "" if effective.has_hr else MISSING_REQUIRED_MODALITY,
        "C1b/PanelD",
    )
    add(
        "has_ecg_r_peaks",
        declared.has_ecg_r_peaks,
        f"modality={effective.cardiac_modality}",
        effective.has_ecg_r_peaks,
        "" if effective.has_ecg_r_peaks or not declared.has_ecg_r_peaks else MISSING_EVENT_SERIES,
        "PanelD",
    )
    add(
        "has_ppg_peaks",
        declared.has_ppg_peaks,
        f"modality={effective.cardiac_modality}",
        effective.has_ppg_peaks,
        "" if effective.has_ppg_peaks or not declared.has_ppg_peaks else MISSING_EVENT_SERIES,
        "PanelD",
    )
    add(
        "has_low_high_task_pair",
        declared.has_low_high_task_pair,
        f"observed_conditions={sorted(o.condition_label.casefold() for o in obs)}; semantics_keys={sorted(semantics)}",
        effective.has_low_high_task_pair,
        ""
        if effective.has_low_high_task_pair or not declared.has_low_high_task_pair
        else MISSING_CONDITION_MAPPING,
        "C5/Figure2-3",
    )
    add(
        "has_pre_post_pair",
        declared.has_pre_post_pair,
        f"has_pre={_bool_str(has_pre)}; has_post={_bool_str(has_post)}",
        effective.has_pre_post_pair,
        ""
        if effective.has_pre_post_pair or not declared.has_pre_post_pair
        else MISSING_PAIRED_OBSERVATION,
        "C5",
    )
    add(
        "supports_d240",
        declared.supports_d240,
        f"max_raw_overlap_s={max_overlap}",
        effective.supports_d240,
        ""
        if effective.supports_d240 or not declared.supports_d240
        else INSUFFICIENT_DURATION,
        "C0/C1c/Figure3C",
    )
    add(
        "supports_d180",
        declared.supports_d180,
        f"max_raw_overlap_s={max_overlap}",
        effective.supports_d180,
        ""
        if effective.supports_d180 or not declared.supports_d180
        else INSUFFICIENT_DURATION,
        "C0/C1c/Figure3C",
    )
    if effective.supports_topography or not declared.supports_topography:
        topo_reason = ""
    elif not has_eeg_files:
        topo_reason = TOPOGRAPHY_NOT_SUPPORTED
    else:
        topo_reason = panel_f.topography_reason_code or TOPOGRAPHY_NOT_SUPPORTED
    add(
        "supports_topography",
        declared.supports_topography,
        f"has_eeg={_bool_str(has_eeg_files)}; {panel_f.evidence_str()}",
        effective.supports_topography,
        topo_reason,
        "PanelF",
    )
    gamma_reason = (
        ""
        if effective.supports_gamma or not declared.supports_gamma
        else (panel_f.gamma_reason_code or MISSING_REQUIRED_BAND)
    )
    add(
        "supports_gamma",
        declared.supports_gamma,
        f"has_eeg={_bool_str(has_eeg_files)}; {panel_f.evidence_str()}",
        effective.supports_gamma,
        gamma_reason,
        "PanelF",
    )

    return effective, rows


def write_capability_resolution(
    rows: Iterable[CapabilityResolutionRow],
    output_dir: str | Path,
) -> Path:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / CAPABILITY_RESOLUTION_FILENAME
    payload = [
        with_structured_nc_fields(
            {
                **row.to_row(),
                "status": "computed" if not row.reason_code else "not_computable",
                "reason": row.reason_code.replace("_", " ") if row.reason_code else "",
                "required_evidence": f"declared:{row.capability}={row.declared_value}",
                "observed_evidence": row.observed_evidence,
                "specification_id": row.capability,
                "dataset_id": "",
                "participant_id": "",
                "session_id": "",
                "observation_id": "",
            },
            stage=str(row.affected_stage or "C0"),
            specification_id=row.capability,
        )
        for row in rows
    ]
    fieldnames = list(
        dict.fromkeys(
            [
                *(list(payload[0].keys()) if payload else []),
                *STRUCTURED_NC_FIELDS,
                "capability",
                "declared_value",
                "effective_value",
                "affected_stage",
            ]
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)
    return path
