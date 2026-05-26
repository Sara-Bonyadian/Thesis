from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PsdConfig:
    fmin: float = 1.0
    fmax: float = 60.0
    n_fft: int = 2048


@dataclass(frozen=True)
class EegConfig:
    l_freq: float = 1.0
    h_freq: float = 60.0
    bad_channel_variance_z: float = 3.0
    reference: str = "average"
    psd: PsdConfig = PsdConfig()


@dataclass(frozen=True)
class PpgConfig:
    start_time_s: float = 175.0
    end_time_s: float = 235.0
    peak_min_distance_s: float = 0.4
    peak_height: float = 0.3
    ibi_min_ms: float = 400.0
    ibi_max_ms: float = 1200.0


@dataclass(frozen=True)
class CorrelationConfig:
    alpha: float = 0.05
    min_n: int = 5
    fdr_method: str = "fdr_bh"


@dataclass(frozen=True)
class PathsConfig:
    raw_root: Path
    out_root: Path


@dataclass(frozen=True)
class PipelineConfig:
    dataset_id: str
    paths: PathsConfig
    subjects: list[str]
    conditions: list[str]
    sessions: list[str]
    eeg: EegConfig = EegConfig()
    ppg: PpgConfig = PpgConfig()
    correlation: CorrelationConfig = CorrelationConfig()


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at top level.")

    paths = data.get("paths") or {}
    raw_root = Path(paths.get("raw_root", "./data/raw"))
    out_root = Path(paths.get("out_root", "./derivatives"))

    eeg = data.get("eeg") or {}
    psd = eeg.get("psd") or {}

    cfg = PipelineConfig(
        dataset_id=str(data.get("dataset_id", "unknown")),
        paths=PathsConfig(raw_root=raw_root, out_root=out_root),
        subjects=list(data.get("subjects") or []),
        conditions=list(data.get("conditions") or []),
        sessions=list(data.get("sessions") or []),
        eeg=EegConfig(
            l_freq=float(_get(eeg, "l_freq", 1.0)),
            h_freq=float(_get(eeg, "h_freq", 60.0)),
            bad_channel_variance_z=float(_get(eeg, "bad_channel_variance_z", 3.0)),
            reference=str(_get(eeg, "reference", "average")),
            psd=PsdConfig(
                fmin=float(_get(psd, "fmin", 1.0)),
                fmax=float(_get(psd, "fmax", 60.0)),
                n_fft=int(_get(psd, "n_fft", 2048)),
            ),
        ),
        ppg=PpgConfig(
            start_time_s=float(_get(data.get("ppg") or {}, "start_time_s", 175.0)),
            end_time_s=float(_get(data.get("ppg") or {}, "end_time_s", 235.0)),
            peak_min_distance_s=float(_get(data.get("ppg") or {}, "peak_min_distance_s", 0.4)),
            peak_height=float(_get(data.get("ppg") or {}, "peak_height", 0.3)),
            ibi_min_ms=float(_get(data.get("ppg") or {}, "ibi_min_ms", 400.0)),
            ibi_max_ms=float(_get(data.get("ppg") or {}, "ibi_max_ms", 1200.0)),
        ),
        correlation=CorrelationConfig(
            alpha=float(_get(data.get("correlation") or {}, "alpha", 0.05)),
            min_n=int(_get(data.get("correlation") or {}, "min_n", 5)),
            fdr_method=str(_get(data.get("correlation") or {}, "fdr_method", "fdr_bh")),
        ),
    )

    return cfg

