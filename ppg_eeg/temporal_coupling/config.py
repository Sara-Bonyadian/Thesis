from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_STAGES: frozenset[str] = frozenset({"0", "1", "1a", "1b", "1c", "2", "3", "4", "all"})


@dataclass(frozen=True)
class TemporalPathsConfig:
    raw_root: Path
    out_root: Path
    base_root: Path | None = None


@dataclass(frozen=True)
class TemporalEegConfig:
    l_freq: float = 1.0
    h_freq: float = 60.0
    bad_channel_variance_z: float = 3.0
    reference: str = "average"


@dataclass(frozen=True)
class TemporalPpgConfig:
    start_time_s: float | None = None
    end_time_s: float | None = None
    peak_min_distance_s: float = 0.4
    peak_height: float = 0.3
    ibi_min_ms: float = 400.0
    ibi_max_ms: float = 1200.0


@dataclass(frozen=True)
class TemporalCouplingEegBandsConfig:
    theta: tuple[float, float] = (4.0, 8.0)
    alpha: tuple[float, float] = (8.0, 13.0)
    beta: tuple[float, float] = (13.0, 30.0)


@dataclass(frozen=True)
class TemporalCouplingEegRoiConfig:
    theta: tuple[str, ...]
    alpha: tuple[str, ...]
    beta: tuple[str, ...]


@dataclass(frozen=True)
class TemporalCouplingEegDebugPlotChannelsConfig:
    theta: str = "Fz"
    alpha: str = "Pz"
    beta: str = "Fz"


@dataclass(frozen=True)
class TemporalCouplingEegDebugPlotConfig:
    enabled: bool = False
    start_time_s: float | None = None
    end_time_s: float | None = None
    auto_window_s: float = 20.0
    also_auto_window: bool = True
    save_psd: bool = False
    channels: TemporalCouplingEegDebugPlotChannelsConfig = TemporalCouplingEegDebugPlotChannelsConfig()


@dataclass(frozen=True)
class TemporalCouplingEegSectionConfig:
    bands: TemporalCouplingEegBandsConfig
    envelope_smooth_s: float
    rois: TemporalCouplingEegRoiConfig
    envelope_output_fs_hz: float | None = None
    debug_plot: TemporalCouplingEegDebugPlotConfig = TemporalCouplingEegDebugPlotConfig()


@dataclass(frozen=True)
class TemporalCouplingCardiacEcgConfig:
    bandpass_hz: tuple[float, float] = (5.0, 30.0)
    min_peak_distance_s: float = 0.45
    prominence: str | float = "auto"
    height: str | float = "auto"
    test_inverted: bool = True


@dataclass(frozen=True)
class TemporalCouplingCardiacDebugPlotConfig:
    enabled: bool = False
    windows: tuple[tuple[float, float], ...] = ()
    start_time_s: float | None = None
    end_time_s: float | None = None
    auto_window_s: float = 10.0
    overview_window_s: float | None = 40.0
    save_overview: bool = True
    also_auto_window: bool = True
    max_plots_per_subject: int = 5


@dataclass(frozen=True)
class TemporalCouplingCardiacConfig:
    channel: str = "auto"
    signal_type: str = "auto"
    detector: str = "auto"
    ecg: TemporalCouplingCardiacEcgConfig = TemporalCouplingCardiacEcgConfig()
    hr_window_s: float = 15.0
    mean_rr_window_s: float | None = None
    hrv_window_s: float = 60.0
    hrv_step_s: float = 1.0
    min_beats_hr: int = 5
    min_beats_hrv: int = 20
    min_valid_hr_percent: float = 50.0
    min_valid_hrv_percent: float = 30.0
    debug_plot: TemporalCouplingCardiacDebugPlotConfig = TemporalCouplingCardiacDebugPlotConfig()


@dataclass(frozen=True)
class TemporalCouplingResampleConfig:
    fs_hz: float
    z_score: bool = True
    interpolate_max_gap_s: float = 5.0


@dataclass(frozen=True)
class TemporalCouplingCrossCorrelationConfig:
    lag_max_s: float
    lag_step_s: float
    n_permutations: int = 0


@dataclass(frozen=True)
class TemporalCouplingEventsConfig:
    enabled: bool = False
    hr_percentile: float = 10.0
    eeg_threshold_sd: float = 2.0
    epoch_pre_s: float = 60.0
    epoch_post_s: float = 60.0


@dataclass(frozen=True)
class TemporalCouplingOutputConfig:
    save_curves: bool = True
    save_plots: bool = True


@dataclass(frozen=True)
class TemporalCouplingAuditConfig:
    min_overlap_s: float = 120.0
    min_clean_beats: int = 30


@dataclass(frozen=True)
class TemporalCouplingSectionConfig:
    audit: TemporalCouplingAuditConfig
    eeg: TemporalCouplingEegSectionConfig
    cardiac: TemporalCouplingCardiacConfig
    resample: TemporalCouplingResampleConfig
    cross_correlation: TemporalCouplingCrossCorrelationConfig
    events: TemporalCouplingEventsConfig
    output: TemporalCouplingOutputConfig


@dataclass(frozen=True)
class TemporalCouplingConfig:
    dataset_id: str
    paths: TemporalPathsConfig
    subjects: list[str]
    tasks: list[str]
    conditions: list[str]
    sessions: list[str]
    eeg: TemporalEegConfig
    ppg: TemporalPpgConfig
    temporal_coupling: TemporalCouplingSectionConfig


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_band_limits(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"temporal_coupling.eeg.bands.{name} must be [low_hz, high_hz].")
    low, high = float(value[0]), float(value[1])
    if low <= 0 or high <= low:
        raise ValueError(f"temporal_coupling.eeg.bands.{name} must satisfy 0 < low < high.")
    return (low, high)


def _as_plot_windows(value: Any) -> tuple[tuple[float, float], ...]:
    if not value:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("temporal_coupling.cardiac.debug_plot.windows must be a list of [start_s, end_s] pairs.")
    windows: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each debug_plot window must be [start_s, end_s].")
        start_s, end_s = float(item[0]), float(item[1])
        if end_s <= start_s:
            raise ValueError(f"Invalid debug_plot window: end_s must exceed start_s ({start_s}, {end_s}).")
        windows.append((start_s, end_s))
    return tuple(windows)


def _as_roi_channels(value: Any, *, name: str) -> tuple[str, ...]:
    channels = tuple(ch.strip() for ch in _as_str_list(value))
    if not channels:
        raise ValueError(f"temporal_coupling.eeg.rois.{name} must list at least one channel.")
    return channels


def validate_stage(stage: str) -> str:
    normalized = str(stage).strip().casefold()
    if normalized not in VALID_STAGES:
        known = ", ".join(sorted(VALID_STAGES, key=lambda s: (len(s), s)))
        raise ValueError(f"Invalid --stage={stage!r}. Allowed values: {known}")
    return normalized


def load_config(path: str | Path) -> TemporalCouplingConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at top level.")

    dataset_id = str(_get(data, "dataset_id", "")).strip()
    if not dataset_id:
        dataset_ids = _as_str_list(data.get("dataset_ids"))
        dataset_id = dataset_ids[0] if dataset_ids else ""
    if not dataset_id:
        raise ValueError("Config must set dataset_id (or dataset_ids).")

    paths_raw = data.get("paths") or {}
    raw_root = Path(str(_get(paths_raw, "raw_root", "./data/raw")))
    out_root = Path(str(_get(paths_raw, "out_root", "./derivatives")))
    base_root_value = paths_raw.get("base_root")
    base_root = Path(str(base_root_value)) if base_root_value else None

    eeg_raw = data.get("eeg") or {}
    ppg_raw = data.get("ppg") or {}
    tc_raw = data.get("temporal_coupling")
    if not isinstance(tc_raw, dict):
        raise ValueError("Config must include a temporal_coupling mapping.")

    audit_raw = tc_raw.get("audit") or {}
    eeg_tc_raw = tc_raw.get("eeg") or {}
    eeg_debug_raw = eeg_tc_raw.get("debug_plot") or {}
    eeg_debug_channels_raw = eeg_debug_raw.get("channels") or {}
    cardiac_raw = tc_raw.get("cardiac") or {}
    ecg_raw = cardiac_raw.get("ecg") or {}
    debug_plot_raw = cardiac_raw.get("debug_plot") or {}
    ecg_bandpass = ecg_raw.get("bandpass_hz") or [5.0, 30.0]
    resample_raw = tc_raw.get("resample") or {}
    xcorr_raw = tc_raw.get("cross_correlation") or {}
    events_raw = tc_raw.get("events") or {}
    output_raw = tc_raw.get("output") or {}

    bands_raw = eeg_tc_raw.get("bands") or {}
    rois_raw = eeg_tc_raw.get("rois") or eeg_tc_raw.get("channels") or {}

    fs_hz = float(_get(resample_raw, "fs_hz", 0.0))
    if fs_hz <= 0:
        raise ValueError("temporal_coupling.resample.fs_hz must be > 0.")

    lag_max_s = float(_get(xcorr_raw, "lag_max_s", 0.0))
    lag_step_s = float(_get(xcorr_raw, "lag_step_s", 0.0))
    if lag_max_s <= 0:
        raise ValueError("temporal_coupling.cross_correlation.lag_max_s must be > 0.")
    if lag_step_s <= 0:
        raise ValueError("temporal_coupling.cross_correlation.lag_step_s must be > 0.")

    cfg = TemporalCouplingConfig(
        dataset_id=dataset_id,
        paths=TemporalPathsConfig(raw_root=raw_root, out_root=out_root, base_root=base_root),
        subjects=_as_str_list(data.get("subjects")),
        tasks=_as_str_list(data.get("tasks")),
        conditions=_as_str_list(data.get("conditions")),
        sessions=_as_str_list(data.get("sessions")),
        eeg=TemporalEegConfig(
            l_freq=float(_get(eeg_raw, "l_freq", 1.0)),
            h_freq=float(_get(eeg_raw, "h_freq", 60.0)),
            bad_channel_variance_z=float(_get(eeg_raw, "bad_channel_variance_z", 3.0)),
            reference=str(_get(eeg_raw, "reference", "average")),
        ),
        ppg=TemporalPpgConfig(
            start_time_s=_as_optional_float(ppg_raw.get("start_time_s")),
            end_time_s=_as_optional_float(ppg_raw.get("end_time_s")),
            peak_min_distance_s=float(_get(ppg_raw, "peak_min_distance_s", 0.4)),
            peak_height=float(_get(ppg_raw, "peak_height", 0.3)),
            ibi_min_ms=float(_get(ppg_raw, "ibi_min_ms", 400.0)),
            ibi_max_ms=float(_get(ppg_raw, "ibi_max_ms", 1200.0)),
        ),
        temporal_coupling=TemporalCouplingSectionConfig(
            audit=TemporalCouplingAuditConfig(
                min_overlap_s=float(_get(audit_raw, "min_overlap_s", 120.0)),
                min_clean_beats=int(_get(audit_raw, "min_clean_beats", 30)),
            ),
            eeg=TemporalCouplingEegSectionConfig(
                bands=TemporalCouplingEegBandsConfig(
                    theta=_as_band_limits(_get(bands_raw, "theta", [4.0, 8.0]), name="theta"),
                    alpha=_as_band_limits(_get(bands_raw, "alpha", [8.0, 13.0]), name="alpha"),
                    beta=_as_band_limits(_get(bands_raw, "beta", [13.0, 30.0]), name="beta"),
                ),
                envelope_smooth_s=float(_get(eeg_tc_raw, "envelope_smooth_s", 2.0)),
                envelope_output_fs_hz=_as_optional_float(eeg_tc_raw.get("envelope_output_fs_hz")),
                rois=TemporalCouplingEegRoiConfig(
                    theta=_as_roi_channels(_get(rois_raw, "theta", []), name="theta"),
                    alpha=_as_roi_channels(_get(rois_raw, "alpha", []), name="alpha"),
                    beta=_as_roi_channels(_get(rois_raw, "beta", []), name="beta"),
                ),
                debug_plot=TemporalCouplingEegDebugPlotConfig(
                    enabled=bool(_get(eeg_debug_raw, "enabled", False)),
                    start_time_s=_as_optional_float(eeg_debug_raw.get("start_time_s")),
                    end_time_s=_as_optional_float(eeg_debug_raw.get("end_time_s")),
                    auto_window_s=float(_get(eeg_debug_raw, "auto_window_s", 20.0)),
                    also_auto_window=bool(_get(eeg_debug_raw, "also_auto_window", True)),
                    save_psd=bool(_get(eeg_debug_raw, "save_psd", False)),
                    channels=TemporalCouplingEegDebugPlotChannelsConfig(
                        theta=str(_get(eeg_debug_channels_raw, "theta", "Fz")),
                        alpha=str(_get(eeg_debug_channels_raw, "alpha", "Pz")),
                        beta=str(_get(eeg_debug_channels_raw, "beta", "Fz")),
                    ),
                ),
            ),
            cardiac=TemporalCouplingCardiacConfig(
                channel=str(_get(cardiac_raw, "channel", "auto")),
                signal_type=str(_get(cardiac_raw, "signal_type", "auto")),
                detector=str(_get(cardiac_raw, "detector", "auto")),
                ecg=TemporalCouplingCardiacEcgConfig(
                    bandpass_hz=(
                        float(ecg_bandpass[0]),
                        float(ecg_bandpass[1]),
                    ),
                    min_peak_distance_s=float(_get(ecg_raw, "min_peak_distance_s", 0.45)),
                    prominence=ecg_raw.get("prominence", "auto"),
                    height=ecg_raw.get("height", "auto"),
                    test_inverted=bool(_get(ecg_raw, "test_inverted", True)),
                ),
                hr_window_s=float(_get(cardiac_raw, "hr_window_s", 15.0)),
                mean_rr_window_s=_as_optional_float(cardiac_raw.get("mean_rr_window_s")),
                hrv_window_s=float(_get(cardiac_raw, "hrv_window_s", 60.0)),
                hrv_step_s=float(_get(cardiac_raw, "hrv_step_s", 1.0)),
                min_beats_hr=int(_get(cardiac_raw, "min_beats_hr", 5)),
                min_beats_hrv=int(_get(cardiac_raw, "min_beats_hrv", 20)),
                min_valid_hr_percent=float(_get(cardiac_raw, "min_valid_hr_percent", 50.0)),
                min_valid_hrv_percent=float(_get(cardiac_raw, "min_valid_hrv_percent", 30.0)),
                debug_plot=TemporalCouplingCardiacDebugPlotConfig(
                    enabled=bool(_get(debug_plot_raw, "enabled", False)),
                    windows=_as_plot_windows(debug_plot_raw.get("windows")),
                    start_time_s=_as_optional_float(debug_plot_raw.get("start_time_s")),
                    end_time_s=_as_optional_float(debug_plot_raw.get("end_time_s")),
                    auto_window_s=float(_get(debug_plot_raw, "auto_window_s", 10.0)),
                    overview_window_s=_as_optional_float(debug_plot_raw.get("overview_window_s")),
                    save_overview=bool(_get(debug_plot_raw, "save_overview", True)),
                    also_auto_window=bool(_get(debug_plot_raw, "also_auto_window", True)),
                    max_plots_per_subject=int(_get(debug_plot_raw, "max_plots_per_subject", 5)),
                ),
            ),
            resample=TemporalCouplingResampleConfig(
                fs_hz=fs_hz,
                z_score=bool(_get(resample_raw, "z_score", True)),
                interpolate_max_gap_s=float(_get(resample_raw, "interpolate_max_gap_s", 5.0)),
            ),
            cross_correlation=TemporalCouplingCrossCorrelationConfig(
                lag_max_s=lag_max_s,
                lag_step_s=lag_step_s,
                n_permutations=int(_get(xcorr_raw, "n_permutations", 0)),
            ),
            events=TemporalCouplingEventsConfig(
                enabled=bool(_get(events_raw, "enabled", False)),
                hr_percentile=float(_get(events_raw, "hr_percentile", 10.0)),
                eeg_threshold_sd=float(_get(events_raw, "eeg_threshold_sd", 2.0)),
                epoch_pre_s=float(_get(events_raw, "epoch_pre_s", 60.0)),
                epoch_post_s=float(_get(events_raw, "epoch_post_s", 60.0)),
            ),
            output=TemporalCouplingOutputConfig(
                save_curves=bool(_get(output_raw, "save_curves", True)),
                save_plots=bool(_get(output_raw, "save_plots", True)),
            ),
        ),
    )
    return cfg
