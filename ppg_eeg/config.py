from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CORE_EEG_FEATURES: list[str] = [
    "eeg_fm_theta",
    "eeg_frontal_beta",
    "eeg_faa",
    "eeg_global_alpha_db",
    "eeg_global_beta_db",
]

DEFAULT_CORE_PPG_FEATURES: list[str] = [
    "ppg_mean_hr_bpm",
    "ppg_rmssd_ms",
    "ppg_sdnn_ms",
    "ppg_mean_rr_ms",
    "ppg_peak_hr_bpm",
]


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
    methods: list[str] = field(default_factory=lambda: ["spearman"])
    primary_method: str = "spearman"
    fdr_method: str = "fdr_bh"


@dataclass(frozen=True)
class PathsConfig:
    raw_root: Path
    out_root: Path


@dataclass(frozen=True)
class FeaturesConfig:
    eeg: list[str] = field(default_factory=lambda: list(DEFAULT_CORE_EEG_FEATURES))
    ppg: list[str] = field(default_factory=lambda: list(DEFAULT_CORE_PPG_FEATURES))
    include_robust_z: bool = True
    reuse_eeg_features_csv: bool = False
    reuse_ppg_features_csv: bool = False


@dataclass(frozen=True)
class OutputConfig:
    save_observation_index: bool = True
    save_summary_json: bool = True
    eeg_base_csv_mode: str = "base_only"
    eeg_base_csv_new_file_name: str = "features_base_eeg_power_extended.csv"


@dataclass(frozen=True)
class PipelineConfig:
    dataset_ids: list[str]
    dataset_id: str
    paths: PathsConfig
    subjects: list[str]
    tasks: list[str]
    conditions: list[str]
    sessions: list[str]
    eeg: EegConfig = EegConfig()
    ppg: PpgConfig = PpgConfig()
    features: FeaturesConfig = FeaturesConfig()
    correlation: CorrelationConfig = CorrelationConfig()
    output: OutputConfig = OutputConfig()


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at top level.")

    paths = data.get("paths") or {}
    raw_root = Path(paths.get("raw_root", "./data/raw"))
    out_root = Path(paths.get("out_root", "./derivatives"))

    dataset_ids = _as_str_list(data.get("dataset_ids"))
    if not dataset_ids:
        dataset_ids = _as_str_list(data.get("dataset_id")) or ["unknown"]
    dataset_id = str(data.get("dataset_id", dataset_ids[0]))

    eeg = data.get("eeg") or {}
    psd = eeg.get("psd") or {}
    ppg = data.get("ppg") or {}
    features = data.get("features") or {}
    correlation = data.get("correlation") or {}
    output = data.get("output") or {}

    corr_methods = [m.casefold() for m in (_as_str_list(_get(correlation, "methods", ["spearman"])) or ["spearman"])]
    corr_primary = str(_get(correlation, "primary_method", corr_methods[0])).casefold()
    if corr_primary not in corr_methods:
        corr_methods.insert(0, corr_primary)

    feature_eeg = _as_str_list(features.get("eeg")) or list(DEFAULT_CORE_EEG_FEATURES)
    feature_ppg = _as_str_list(features.get("ppg")) or list(DEFAULT_CORE_PPG_FEATURES)

    cfg = PipelineConfig(
        dataset_ids=dataset_ids,
        dataset_id=dataset_id,
        paths=PathsConfig(raw_root=raw_root, out_root=out_root),
        subjects=_as_str_list(data.get("subjects")),
        tasks=_as_str_list(data.get("tasks")),
        conditions=_as_str_list(data.get("conditions")),
        sessions=_as_str_list(data.get("sessions")),
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
            start_time_s=float(_get(ppg, "start_time_s", 175.0)),
            end_time_s=float(_get(ppg, "end_time_s", 235.0)),
            peak_min_distance_s=float(_get(ppg, "peak_min_distance_s", 0.4)),
            peak_height=float(_get(ppg, "peak_height", 0.3)),
            ibi_min_ms=float(_get(ppg, "ibi_min_ms", 400.0)),
            ibi_max_ms=float(_get(ppg, "ibi_max_ms", 1200.0)),
        ),
        features=FeaturesConfig(
            eeg=feature_eeg,
            ppg=feature_ppg,
            include_robust_z=bool(_get(features, "include_robust_z", True)),
            reuse_eeg_features_csv=bool(_get(features, "reuse_eeg_features_csv", False)),
            reuse_ppg_features_csv=bool(_get(features, "reuse_ppg_features_csv", False)),
        ),
        correlation=CorrelationConfig(
            alpha=float(_get(correlation, "alpha", 0.05)),
            min_n=int(_get(correlation, "min_n", 5)),
            methods=corr_methods,
            primary_method=corr_primary,
            fdr_method=str(_get(correlation, "fdr_method", "fdr_bh")),
        ),
        output=OutputConfig(
            save_observation_index=bool(_get(output, "save_observation_index", True)),
            save_summary_json=bool(_get(output, "save_summary_json", True)),
            eeg_base_csv_mode=str(_get(output, "eeg_base_csv_mode", "base_only")).casefold(),
            eeg_base_csv_new_file_name=str(
                _get(output, "eeg_base_csv_new_file_name", "features_base_eeg_power_extended.csv")
            ),
        ),
    )

    allowed_modes = {"base_only", "append_columns", "new_file"}
    if cfg.output.eeg_base_csv_mode not in allowed_modes:
        allowed_text = ", ".join(sorted(allowed_modes))
        raise ValueError(
            f"Invalid output.eeg_base_csv_mode={cfg.output.eeg_base_csv_mode!r}. "
            f"Expected one of: {allowed_text}."
        )
    if not cfg.output.eeg_base_csv_new_file_name.strip():
        raise ValueError("output.eeg_base_csv_new_file_name must be a non-empty filename.")

    return cfg

