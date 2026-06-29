from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import mne
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .datasets import CanonicalObservation
from .eeg import (
    FRONTAL_BETA_CHANNELS_DEFAULT,
    FRONTAL_MIDLINE_CHANNELS_DEFAULT,
    FRONTAL_PAIRS_DEFAULT,
    channel_band_powers,
    preprocess_eeg,
)
from .ppg import (
    PpgIbiResult,
    extract_clean_ppg_ibi_from_raw,
    mean_hr_bpm,
    mean_rr_ms,
    peak_hr_bpm,
    rmssd_ms,
    sdnn_ms,
)


CORE_EEG_FEATURES: list[str] = [
    "eeg_fm_theta",
    "eeg_frontal_beta",
    "eeg_faa",
    "eeg_global_alpha_db",
    "eeg_global_beta_db",
    "theta_power_uv2",
    "alpha_power_uv2",
    "beta_power_uv2",
]

CORE_PPG_FEATURES: list[str] = [
    "ppg_mean_hr_bpm",
    "ppg_rmssd_ms",
    "ppg_sdnn_ms",
    "ppg_mean_rr_ms",
    "ppg_peak_hr_bpm",
]

OBSERVATION_KEY_COLUMNS: list[str] = [
    "dataset_id",
    "observation_id",
    "subject_id",
    "task_label",
    "condition_label",
    "session_label",
    "modality",
    "timepoint",
    "state",
]

EEG_POWER_SCHEMA_VERSION = "eeg_power_schema_v2"
PPG_IBI_SCHEMA_VERSION = "ppg_ibi_schema_v1"


LINEAR_BAND_POWER_COLUMNS: list[str] = [
    "theta_power_uv2",
    "alpha_power_uv2",
    "beta_power_uv2",
]


EEG_PROCESSING_METADATA_COLUMNS: list[str] = [
    "l_freq",
    "h_freq",
    "reference",
    "psd_fmin",
    "psd_fmax",
    "n_fft",
    "processing_version",
]


PPG_IBI_COLUMNS: list[str] = [
    *OBSERVATION_KEY_COLUMNS,
    "eeg_path",
    "eeg_format",
    "ppg_source",
    "ppg_path",
    "ppg_format",
    "ppg_error",
    "ppg_channel",
    "ppg_segment_start_s",
    "ppg_segment_end_s",
    "sfreq",
    "peak_min_distance_s",
    "peak_height",
    "ibi_min_ms",
    "ibi_max_ms",
    "n_peaks",
    "n_ibi_raw",
    "n_ibi_clean",
    "ibi_index",
    "peak_time_relative_s",
    "peak_time_absolute_s",
    "peak_index",
    "ibi_ms_clean",
    "n_ibi_raw_invalid",
    "ibi_ms_raw_min",
    "ibi_ms_raw_max",
    "processing_version",
]


def base_eeg_power_columns(kind: str = "export") -> list[str]:
    power_feature_columns = list(LINEAR_BAND_POWER_COLUMNS)

    normalized = kind.casefold()
    if normalized in {"feature", "features"}:
        return list(power_feature_columns)
    if normalized == "export":
        return [
            "dataset_id",
            "observation_id",
            "subject_id",
            "task_label",
            "condition_label",
            "session_label",
            "modality",
            "timepoint",
            "state",
            "eeg_format",
            "eeg_path",
            "n_bad_channels",
            "eeg_error",
            "channel",
            *power_feature_columns,
            *EEG_PROCESSING_METADATA_COLUMNS,
        ]
    raise ValueError(f"Unsupported base EEG column kind: {kind!r}")


def base_ppg_ibi_columns() -> list[str]:
    return list(PPG_IBI_COLUMNS)


@dataclass(frozen=True)
class FeatureExtractionResult:
    eeg_features: pd.DataFrame
    eeg_base_features: pd.DataFrame
    ppg_features: pd.DataFrame
    ppg_ibi_features: pd.DataFrame
    merged_features: pd.DataFrame


def _infer_channel_type(ch_name: str) -> str:
    low = ch_name.casefold()
    if "ecg" in low:
        return "ecg"
    if any(token in low for token in ("ppg", "photo", "optic", "pleth", "pulse")):
        return "misc"
    return "eeg"


def _extract_channel_names_from_eeglab(eeg_obj: dict[str, object], n_channels: int) -> list[str]:
    chanlocs = eeg_obj.get("chanlocs")
    if isinstance(chanlocs, dict):
        labels = chanlocs.get("labels")
        if isinstance(labels, str):
            out = [labels]
        elif isinstance(labels, (list, tuple, np.ndarray)):
            out = [str(lbl) for lbl in labels]
        else:
            out = []
        if len(out) == n_channels and all(name.strip() for name in out):
            return out
    return [f"EEG{idx + 1:03d}" for idx in range(n_channels)]


def _read_raw_eeglab_v73(path: Path) -> mne.io.BaseRaw:
    from pymatreader import read_mat

    loaded = read_mat(str(path))
    eeg_obj = loaded.get("EEG", loaded)
    if not isinstance(eeg_obj, dict):
        raise ValueError(f"Unexpected EEGLAB payload type in {path}: {type(eeg_obj)!r}")

    data = np.asarray(eeg_obj.get("data"), dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEGLAB data array in {path}, got shape {data.shape!r}")

    sfreq = float(eeg_obj.get("srate", 1.0))
    if sfreq <= 0:
        raise ValueError(f"Invalid sampling rate in {path}: {sfreq!r}")

    ch_names = _extract_channel_names_from_eeglab(eeg_obj, n_channels=data.shape[0])
    ch_types = [_infer_channel_type(ch_name) for ch_name in ch_names]

    # EEGLAB numeric arrays are typically stored in micro-units.
    data = data * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)


def _read_raw(path: Path, data_format: str) -> mne.io.BaseRaw:
    fmt = data_format.casefold()
    if fmt == "brainvision":
        return mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
    if fmt == "eeglab":
        try:
            return mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
        except NotImplementedError as exc:
            if "matlab v7.3" in str(exc).lower() or "hdf reader" in str(exc).lower():
                return _read_raw_eeglab_v73(path)
            raise
    if fmt == "edf":
        return mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    raise ValueError(f"Unsupported data format: {data_format!r}")


def _safe_eval(func: Callable[[], float]) -> float:
    try:
        return float(func())
    except Exception:
        return float("nan")


def _segment_bounds_for_ppg(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> tuple[float, float]:
    sfreq = float(raw.info["sfreq"])
    duration_s = max(0.0, (float(raw.n_times) - 1.0) / sfreq)
    if duration_s <= 0.0:
        return 0.0, 0.0

    start_s = max(0.0, float(cfg.ppg.start_time_s))
    end_s = min(float(cfg.ppg.end_time_s), duration_s)
    if end_s <= start_s:
        return 0.0, duration_s
    return start_s, end_s


def _extract_ppg_ibi_with_fallback(
    raw: mne.io.BaseRaw,
    cfg: PipelineConfig,
) -> tuple[PpgIbiResult, float, float]:
    start_s, end_s = _segment_bounds_for_ppg(raw, cfg)
    result = extract_clean_ppg_ibi_from_raw(
        raw,
        start_time_s=start_s,
        end_time_s=end_s,
        peak_min_distance_s=cfg.ppg.peak_min_distance_s,
        peak_height=cfg.ppg.peak_height,
        ibi_min_ms=cfg.ppg.ibi_min_ms,
        ibi_max_ms=cfg.ppg.ibi_max_ms,
    )

    if result.ibi_ms_clean is not None and len(result.ibi_ms_clean) >= 3:
        return result, start_s, end_s

    sfreq = float(raw.info["sfreq"])
    duration_s = max(0.0, (float(raw.n_times) - 1.0) / sfreq)
    if start_s > 0.0 and duration_s > 0.0:
        fallback = extract_clean_ppg_ibi_from_raw(
            raw,
            start_time_s=0.0,
            end_time_s=duration_s,
            peak_min_distance_s=cfg.ppg.peak_min_distance_s,
            peak_height=cfg.ppg.peak_height,
            ibi_min_ms=cfg.ppg.ibi_min_ms,
            ibi_max_ms=cfg.ppg.ibi_max_ms,
        )
        if fallback.ibi_ms_clean is not None and len(fallback.ibi_ms_clean) >= 3:
            return fallback, 0.0, duration_s

    return result, start_s, end_s


def _robust_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    valid = values[np.isfinite(values)]
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype=float)

    return (values - med) / scale


def add_dataset_local_robust_zscores(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        z_col = f"{col}_rz"
        if col not in out.columns:
            out[z_col] = np.nan
            continue
        out[z_col] = (
            out.groupby("dataset_id", dropna=False)[col]
            .transform(_robust_zscore)
            .astype(float)
        )
    return out


def _base_observation_record(obs: CanonicalObservation) -> dict[str, object]:
    return {
        "dataset_id": obs.dataset_id,
        "observation_id": obs.observation_id,
        "subject_id": obs.subject_id,
        "task_label": obs.task_label,
        "condition_label": obs.condition_label,
        "session_label": obs.session_label or "",
        "modality": obs.modality or "",
        "timepoint": obs.timepoint or "",
        "state": obs.state or "",
    }


def _eeg_processing_metadata(cfg: PipelineConfig) -> dict[str, object]:
    return {
        "l_freq": float(cfg.eeg.l_freq),
        "h_freq": float(cfg.eeg.h_freq),
        "reference": cfg.eeg.reference,
        "psd_fmin": float(cfg.eeg.psd.fmin),
        "psd_fmax": float(cfg.eeg.psd.fmax),
        "n_fft": int(cfg.eeg.psd.n_fft),
        "processing_version": EEG_POWER_SCHEMA_VERSION,
    }


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric[np.isfinite(numeric)].to_numpy(dtype=float)


def _mean_finite(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _mean_positive(values: pd.Series) -> float:
    finite = _finite_values(values)
    positive = finite[finite > 0.0]
    return float(np.mean(positive)) if positive.size else float("nan")


def _channel_power_map(df: pd.DataFrame, power_col: str) -> dict[str, float]:
    if "channel" not in df.columns or power_col not in df.columns:
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        channel = str(row.get("channel", "")).strip()
        if not channel:
            continue
        value = _coerce_float(row.get(power_col))
        if np.isfinite(value):
            out[channel] = value
    return out


def _derive_eeg_feature_values_from_channel_rows(channel_rows: Sequence[dict[str, object]]) -> dict[str, float]:
    df = pd.DataFrame(channel_rows)
    values = {name: float("nan") for name in CORE_EEG_FEATURES}
    values.update({name: float("nan") for name in LINEAR_BAND_POWER_COLUMNS})
    if df.empty:
        return values

    for col in LINEAR_BAND_POWER_COLUMNS:
        if col in df.columns:
            values[col] = _mean_positive(df[col])

    alpha_power = values.get("alpha_power_uv2", float("nan"))
    if np.isfinite(alpha_power) and alpha_power > 0.0:
        values["eeg_global_alpha_db"] = float(10 * np.log10(alpha_power))
    beta_power = values.get("beta_power_uv2", float("nan"))
    if np.isfinite(beta_power) and beta_power > 0.0:
        values["eeg_global_beta_db"] = float(10 * np.log10(beta_power))

    theta_by_channel = _channel_power_map(df, "theta_power_uv2")
    theta_vals = [
        theta_by_channel[ch]
        for ch in FRONTAL_MIDLINE_CHANNELS_DEFAULT
        if ch in theta_by_channel and np.isfinite(theta_by_channel[ch]) and theta_by_channel[ch] > 0.0
    ]
    if theta_vals:
        values["eeg_fm_theta"] = float(np.log(np.mean(theta_vals)))

    beta_by_channel = _channel_power_map(df, "beta_power_uv2")
    beta_vals = [
        beta_by_channel[ch]
        for ch in FRONTAL_BETA_CHANNELS_DEFAULT
        if ch in beta_by_channel and np.isfinite(beta_by_channel[ch]) and beta_by_channel[ch] > 0.0
    ]
    if beta_vals:
        values["eeg_frontal_beta"] = float(np.log(np.mean(beta_vals)))

    alpha_by_channel = _channel_power_map(df, "alpha_power_uv2")
    faa_vals: list[float] = []
    for left, right in FRONTAL_PAIRS_DEFAULT:
        p_left = alpha_by_channel.get(left)
        p_right = alpha_by_channel.get(right)
        if (
            p_left is None
            or p_right is None
            or not np.isfinite(p_left)
            or not np.isfinite(p_right)
            or p_left <= 0.0
            or p_right <= 0.0
        ):
            continue
        faa_vals.append(float(np.log(p_right) - np.log(p_left)))
    if faa_vals:
        values["eeg_faa"] = float(np.mean(faa_vals))

    return values


def _ppg_processing_metadata(cfg: PipelineConfig) -> dict[str, object]:
    return {
        "peak_min_distance_s": float(cfg.ppg.peak_min_distance_s),
        "peak_height": float(cfg.ppg.peak_height),
        "ibi_min_ms": float(cfg.ppg.ibi_min_ms),
        "ibi_max_ms": float(cfg.ppg.ibi_max_ms),
        "processing_version": PPG_IBI_SCHEMA_VERSION,
    }


def _ppg_feature_values_from_ibi(ibi_ms: np.ndarray | Sequence[float] | pd.Series | None) -> dict[str, float]:
    values = {name: float("nan") for name in CORE_PPG_FEATURES}
    if ibi_ms is None:
        return values

    ibi = pd.to_numeric(pd.Series(ibi_ms), errors="coerce").astype(float)
    clean = ibi[np.isfinite(ibi)].to_numpy(dtype=float)
    if len(clean) < 3:
        return values

    values.update(
        {
            "ppg_mean_hr_bpm": _safe_eval(lambda: mean_hr_bpm(clean)),
            "ppg_rmssd_ms": _safe_eval(lambda: rmssd_ms(clean)),
            "ppg_sdnn_ms": _safe_eval(lambda: sdnn_ms(clean)),
            "ppg_mean_rr_ms": _safe_eval(lambda: mean_rr_ms(clean)),
            "ppg_peak_hr_bpm": _safe_eval(lambda: peak_hr_bpm(clean)),
        }
    )
    return values


def _ppg_feature_values(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> tuple[dict[str, float], dict[str, object], PpgIbiResult, float, float]:
    result, start_s, end_s = _extract_ppg_ibi_with_fallback(raw, cfg)
    ibi_ms = result.ibi_ms_clean
    qc: dict[str, object] = {
        "ppg_channel": result.ppg_channel or "",
        "ppg_segment_start_s": start_s,
        "ppg_segment_end_s": end_s,
        "n_ibi_clean": int(len(ibi_ms)) if ibi_ms is not None else 0,
    }

    return _ppg_feature_values_from_ibi(ibi_ms), qc, result, start_s, end_s


def _ppg_ibi_rows_from_result(
    *,
    base: dict[str, object],
    obs: CanonicalObservation,
    cfg: PipelineConfig,
    result: PpgIbiResult,
    start_s: float,
    end_s: float,
    ppg_error: str,
) -> list[dict[str, object]]:
    raw_ibi = result.ibi_ms_raw if result.ibi_ms_raw is not None else np.array([], dtype=float)
    clean_ibi = result.ibi_ms_clean if result.ibi_ms_clean is not None else np.array([], dtype=float)
    peak_times = result.peak_times_s if result.peak_times_s is not None else np.array([], dtype=float)
    peak_indices = result.peaks_idx if result.peaks_idx is not None else np.array([], dtype=float)
    if len(clean_ibi) == 0:
        return []

    raw_ibi_finite = raw_ibi[np.isfinite(raw_ibi)] if len(raw_ibi) else np.array([], dtype=float)
    invalid_raw = raw_ibi_finite[(raw_ibi_finite <= cfg.ppg.ibi_min_ms) | (raw_ibi_finite >= cfg.ppg.ibi_max_ms)]
    raw_min = float(np.min(raw_ibi_finite)) if len(raw_ibi_finite) else np.nan
    raw_max = float(np.max(raw_ibi_finite)) if len(raw_ibi_finite) else np.nan
    rows: list[dict[str, object]] = []
    metadata = _ppg_processing_metadata(cfg)
    for idx, ibi_clean in enumerate(clean_ibi):
        peak_time_relative = float(peak_times[idx + 1]) if idx + 1 < len(peak_times) else np.nan
        row = dict(base)
        row.update(
            {
                "eeg_path": str(obs.eeg_path),
                "eeg_format": obs.eeg_format,
                "ppg_source": obs.ppg_source,
                "ppg_path": str(obs.ppg_path) if obs.ppg_path is not None else "",
                "ppg_format": obs.ppg_format or "",
                "ppg_error": ppg_error,
                "ppg_channel": result.ppg_channel or "",
                "ppg_segment_start_s": start_s,
                "ppg_segment_end_s": end_s,
                "sfreq": float(result.sfreq),
                "n_peaks": int(len(peak_times)),
                "n_ibi_raw": int(len(raw_ibi)),
                "n_ibi_clean": int(len(clean_ibi)),
                "ibi_index": idx,
                "peak_time_relative_s": peak_time_relative,
                "peak_time_absolute_s": peak_time_relative + start_s if np.isfinite(peak_time_relative) else np.nan,
                "peak_index": int(peak_indices[idx + 1]) if idx + 1 < len(peak_indices) else pd.NA,
                "ibi_ms_clean": float(ibi_clean),
                "n_ibi_raw_invalid": int(len(invalid_raw)),
                "ibi_ms_raw_min": raw_min,
                "ibi_ms_raw_max": raw_max,
                **metadata,
            }
        )
        rows.append(row)
    return rows


def _derive_ppg_from_ibi_rows(ibi_rows: Sequence[dict[str, object]]) -> tuple[dict[str, float], dict[str, object], str]:
    df = pd.DataFrame(ibi_rows)
    values = {name: float("nan") for name in CORE_PPG_FEATURES}
    qc: dict[str, object] = {
        "ppg_channel": "",
        "ppg_segment_start_s": np.nan,
        "ppg_segment_end_s": np.nan,
        "n_ibi_clean": 0,
    }
    if df.empty:
        return values, qc, "cached"

    clean_ibi = pd.to_numeric(df.get("ibi_ms_clean"), errors="coerce").astype(float)
    clean_ibi = clean_ibi[np.isfinite(clean_ibi)]
    values = _ppg_feature_values_from_ibi(clean_ibi)
    first = df.iloc[0]
    qc = {
        "ppg_channel": _coerce_str(first.get("ppg_channel"), default=""),
        "ppg_segment_start_s": _coerce_float(first.get("ppg_segment_start_s")),
        "ppg_segment_end_s": _coerce_float(first.get("ppg_segment_end_s")),
        "n_ibi_clean": int(len(clean_ibi)),
    }
    return values, qc, _coerce_str(first.get("ppg_error"), default="cached")


def _build_feature_cache_lookup(
    cache_df: pd.DataFrame | None,
    *,
    required_feature_columns: Sequence[str],
    table_name: str,
) -> dict[str, dict[str, object]]:
    if cache_df is None or cache_df.empty:
        return {}

    if "observation_id" not in cache_df.columns:
        raise ValueError(f"{table_name} cache must contain an 'observation_id' column.")

    missing_cols = [col for col in required_feature_columns if col not in cache_df.columns]
    if missing_cols:
        needed = ", ".join(sorted(missing_cols))
        raise ValueError(f"{table_name} cache is missing required feature columns: {needed}")

    lookup: dict[str, dict[str, object]] = {}
    for record in cache_df.to_dict(orient="records"):
        obs_id = str(record.get("observation_id", "")).strip()
        if not obs_id:
            continue
        lookup[obs_id] = record
    return lookup


def _has_channel_power_schema(cache_df: pd.DataFrame) -> bool:
    required = {"observation_id", "channel", *LINEAR_BAND_POWER_COLUMNS}
    return required.issubset(set(cache_df.columns))


def _build_eeg_cache_lookups(
    cache_df: pd.DataFrame | None,
    *,
    required_feature_columns: Sequence[str],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    if cache_df is None or cache_df.empty:
        return {}, {}

    if "observation_id" not in cache_df.columns:
        raise ValueError("EEG feature cache must contain an 'observation_id' column.")

    channel_lookup: dict[str, list[dict[str, object]]] = {}
    if _has_channel_power_schema(cache_df):
        for obs_id, group in cache_df.groupby("observation_id", dropna=False):
            obs_key = str(obs_id).strip()
            if not obs_key:
                continue
            channel_lookup[obs_key] = group.to_dict(orient="records")

        feature_lookup: dict[str, dict[str, object]] = {}
        for obs_key, channel_rows in channel_lookup.items():
            first = dict(channel_rows[0])
            first["channel"] = "|".join(
                str(row.get("channel", "")).strip()
                for row in channel_rows
                if str(row.get("channel", "")).strip()
            )
            first.update(_derive_eeg_feature_values_from_channel_rows(channel_rows))
            feature_lookup[obs_key] = first

        missing_cols = [
            col
            for col in required_feature_columns
            if all(col not in record for record in feature_lookup.values())
        ]
        if missing_cols:
            needed = ", ".join(sorted(missing_cols))
            raise ValueError(f"EEG channel-level cache cannot derive required feature columns: {needed}")
        return feature_lookup, channel_lookup

    missing_cols = [col for col in required_feature_columns if col not in cache_df.columns]
    if missing_cols:
        needed = ", ".join(sorted(missing_cols))
        raise ValueError(f"EEG feature cache is missing required feature columns: {needed}")

    feature_lookup = {}
    for record in cache_df.to_dict(orient="records"):
        obs_id = str(record.get("observation_id", "")).strip()
        if not obs_id:
            continue
        feature_lookup[obs_id] = record
    return feature_lookup, {}


def _has_ppg_ibi_schema(cache_df: pd.DataFrame) -> bool:
    required = {"observation_id", "ibi_ms_clean"}
    return required.issubset(set(cache_df.columns))


def _build_ppg_cache_lookups(
    cache_df: pd.DataFrame | None,
    *,
    required_feature_columns: Sequence[str],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    if cache_df is None or cache_df.empty:
        return {}, {}

    if "observation_id" not in cache_df.columns:
        raise ValueError("PPG feature cache must contain an 'observation_id' column.")

    ibi_lookup: dict[str, list[dict[str, object]]] = {}
    if _has_ppg_ibi_schema(cache_df):
        for obs_id, group in cache_df.groupby("observation_id", dropna=False):
            obs_key = str(obs_id).strip()
            if not obs_key:
                continue
            ibi_lookup[obs_key] = group.to_dict(orient="records")

        feature_lookup: dict[str, dict[str, object]] = {}
        for obs_key, ibi_rows in ibi_lookup.items():
            values, qc, ppg_error = _derive_ppg_from_ibi_rows(ibi_rows)
            first = dict(ibi_rows[0])
            first.update(values)
            first.update(qc)
            first["ppg_error"] = ppg_error
            feature_lookup[obs_key] = first

        missing_cols = [
            col
            for col in required_feature_columns
            if all(col not in record for record in feature_lookup.values())
        ]
        if missing_cols:
            needed = ", ".join(sorted(missing_cols))
            raise ValueError(f"PPG IBI cache cannot derive required feature columns: {needed}")
        return feature_lookup, ibi_lookup

    missing_cols = [col for col in required_feature_columns if col not in cache_df.columns]
    if missing_cols:
        needed = ", ".join(sorted(missing_cols))
        raise ValueError(f"PPG feature cache is missing required feature columns: {needed}")

    feature_lookup = {}
    for record in cache_df.to_dict(orient="records"):
        obs_id = str(record.get("observation_id", "")).strip()
        if not obs_id:
            continue
        feature_lookup[obs_id] = record
    return feature_lookup, {}


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, *, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        return float(pd.to_numeric(value, errors="coerce"))
    except (TypeError, ValueError):
        return default


def _coerce_str(value: object, *, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return text if text.strip() else default


def _truthy_csv_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return text not in {"", "0", "false", "no", "nan", "none"}


def _observation_base_from_record(record: dict[str, object]) -> dict[str, object]:
    return {col: record.get(col, "") for col in OBSERVATION_KEY_COLUMNS}


def _records_by_observation_id(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    if df.empty or "observation_id" not in df.columns:
        return {}

    grouped: dict[str, list[dict[str, object]]] = {}
    for obs_id, group in df.groupby("observation_id", dropna=False):
        obs_key = str(obs_id).strip()
        if obs_key:
            grouped[obs_key] = group.to_dict(orient="records")
    return grouped


def _ordered_feature_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    ordered = list(dict.fromkeys(columns))
    extra = [col for col in out.columns if col not in ordered]
    return out.loc[:, [*ordered, *extra]].copy()


def derive_core_feature_tables_from_base_tables(
    observations_df: pd.DataFrame,
    eeg_base_df: pd.DataFrame,
    ppg_ibi_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> FeatureExtractionResult:
    """Derive analysis feature tables from the required Stage 1 CSV tables."""

    selected_eeg_features = list(cfg.features.eeg)
    selected_ppg_features = list(cfg.features.ppg)
    base_eeg_feature_columns = base_eeg_power_columns("feature")
    eeg_by_observation = _records_by_observation_id(eeg_base_df)
    ppg_by_observation = _records_by_observation_id(ppg_ibi_df)

    eeg_records: list[dict[str, object]] = []
    ppg_records: list[dict[str, object]] = []

    for obs_record in observations_df.to_dict(orient="records"):
        if "is_usable" in obs_record and not _truthy_csv_value(obs_record.get("is_usable")):
            continue

        obs_id = str(obs_record.get("observation_id", "")).strip()
        if not obs_id:
            continue

        base = _observation_base_from_record(obs_record)
        eeg_row = dict(base)
        eeg_row.update(
            {
                "eeg_path": _coerce_str(obs_record.get("eeg_path")),
                "eeg_format": _coerce_str(obs_record.get("eeg_format")),
                "ppg_source": _coerce_str(obs_record.get("ppg_source")),
                "n_bad_channels": 0,
                "eeg_error": "missing_base_eeg_power",
                "channel": "",
            }
        )
        eeg_values = {name: float("nan") for name in [*CORE_EEG_FEATURES, *LINEAR_BAND_POWER_COLUMNS]}
        eeg_rows = eeg_by_observation.get(obs_id, [])
        if eeg_rows:
            first_eeg = eeg_rows[0]
            eeg_values.update(_derive_eeg_feature_values_from_channel_rows(eeg_rows))
            eeg_row.update(
                {
                    "eeg_path": _coerce_str(first_eeg.get("eeg_path"), default=eeg_row["eeg_path"]),
                    "eeg_format": _coerce_str(first_eeg.get("eeg_format"), default=eeg_row["eeg_format"]),
                    "n_bad_channels": _coerce_int(first_eeg.get("n_bad_channels"), default=0),
                    "eeg_error": _coerce_str(first_eeg.get("eeg_error"), default="cached"),
                    "channel": "|".join(
                        str(row.get("channel", "")).strip()
                        for row in eeg_rows
                        if str(row.get("channel", "")).strip()
                    ),
                }
            )
        for name in base_eeg_feature_columns:
            eeg_row[name] = float(eeg_values.get(name, np.nan))
        for name in selected_eeg_features:
            eeg_row[name] = float(eeg_values.get(name, np.nan))
        eeg_records.append(eeg_row)

        ppg_row = dict(base)
        ppg_row.update(
            {
                "eeg_path": _coerce_str(obs_record.get("eeg_path")),
                "eeg_format": _coerce_str(obs_record.get("eeg_format")),
                "ppg_source": _coerce_str(obs_record.get("ppg_source")),
                "ppg_path": _coerce_str(obs_record.get("ppg_path")),
                "ppg_format": _coerce_str(obs_record.get("ppg_format")),
                "ppg_error": "missing_base_ppg_ibi",
                "ppg_channel": "",
                "ppg_segment_start_s": np.nan,
                "ppg_segment_end_s": np.nan,
                "n_ibi_clean": 0,
            }
        )
        ppg_values = {name: float("nan") for name in CORE_PPG_FEATURES}
        ppg_rows = ppg_by_observation.get(obs_id, [])
        if ppg_rows:
            values, qc, ppg_error = _derive_ppg_from_ibi_rows(ppg_rows)
            first_ppg = ppg_rows[0]
            ppg_values.update(values)
            ppg_row.update(
                {
                    "eeg_path": _coerce_str(first_ppg.get("eeg_path"), default=ppg_row["eeg_path"]),
                    "eeg_format": _coerce_str(first_ppg.get("eeg_format"), default=ppg_row["eeg_format"]),
                    "ppg_source": _coerce_str(first_ppg.get("ppg_source"), default=ppg_row["ppg_source"]),
                    "ppg_path": _coerce_str(first_ppg.get("ppg_path"), default=ppg_row["ppg_path"]),
                    "ppg_format": _coerce_str(first_ppg.get("ppg_format"), default=ppg_row["ppg_format"]),
                    "ppg_error": ppg_error,
                    **qc,
                }
            )
        for name in selected_ppg_features:
            ppg_row[name] = float(ppg_values.get(name, np.nan))
        ppg_records.append(ppg_row)

    eeg_columns = [
        *OBSERVATION_KEY_COLUMNS,
        "eeg_path",
        "eeg_format",
        "ppg_source",
        "n_bad_channels",
        "eeg_error",
        "channel",
        *base_eeg_feature_columns,
        *selected_eeg_features,
    ]
    ppg_columns = [
        *OBSERVATION_KEY_COLUMNS,
        "eeg_path",
        "eeg_format",
        "ppg_source",
        "ppg_path",
        "ppg_format",
        "ppg_error",
        "ppg_channel",
        "ppg_segment_start_s",
        "ppg_segment_end_s",
        "n_ibi_clean",
        *selected_ppg_features,
    ]

    eeg_df = _ordered_feature_table(pd.DataFrame(eeg_records), eeg_columns)
    ppg_df = _ordered_feature_table(pd.DataFrame(ppg_records), ppg_columns)

    if cfg.features.include_robust_z:
        eeg_df = add_dataset_local_robust_zscores(eeg_df, selected_eeg_features)
        ppg_df = add_dataset_local_robust_zscores(ppg_df, selected_ppg_features)

    merge_cols = OBSERVATION_KEY_COLUMNS
    if eeg_df.empty or ppg_df.empty:
        merged_df = pd.DataFrame()
    else:
        ppg_merge_columns = [
            *merge_cols,
            *selected_ppg_features,
            *[f"{f}_rz" for f in selected_ppg_features if f"{f}_rz" in ppg_df.columns],
            "ppg_error",
            "ppg_channel",
            "ppg_segment_start_s",
            "ppg_segment_end_s",
            "n_ibi_clean",
        ]
        ppg_merge_columns = [col for col in dict.fromkeys(ppg_merge_columns) if col in ppg_df.columns]
        merged_df = eeg_df.merge(ppg_df[ppg_merge_columns], on=merge_cols, how="inner")

    return FeatureExtractionResult(
        eeg_features=eeg_df,
        eeg_base_features=eeg_base_df.copy(),
        ppg_features=ppg_df,
        ppg_ibi_features=ppg_ibi_df.copy(),
        merged_features=merged_df,
    )


def extract_core_feature_tables(
    observations: Sequence[CanonicalObservation],
    cfg: PipelineConfig,
    *,
    eeg_feature_cache: pd.DataFrame | None = None,
    ppg_feature_cache: pd.DataFrame | None = None,
) -> FeatureExtractionResult:
    eeg_records: list[dict[str, object]] = []
    base_eeg_records: list[dict[str, object]] = []
    ppg_records: list[dict[str, object]] = []
    ppg_ibi_records: list[dict[str, object]] = []

    selected_eeg_features = list(cfg.features.eeg)
    selected_ppg_features = list(cfg.features.ppg)
    base_eeg_feature_columns = base_eeg_power_columns("feature")
    eeg_processing_metadata = _eeg_processing_metadata(cfg)
    eeg_cache_lookup, eeg_channel_cache_lookup = _build_eeg_cache_lookups(
        eeg_feature_cache,
        required_feature_columns=selected_eeg_features,
    )
    ppg_cache_lookup, ppg_ibi_cache_lookup = _build_ppg_cache_lookups(
        ppg_feature_cache,
        required_feature_columns=selected_ppg_features,
    )

    for obs in observations:
        if not obs.is_usable:
            continue

        base = _base_observation_record(obs)
        eeg_row = dict(base)
        ppg_row = dict(base)
        eeg_row.update(
            {
                "eeg_path": str(obs.eeg_path),
                "eeg_format": obs.eeg_format,
                "ppg_source": obs.ppg_source,
                "channel": "",
            }
        )
        ppg_row.update(
            {
                "eeg_path": str(obs.eeg_path),
                "eeg_format": obs.eeg_format,
                "ppg_source": obs.ppg_source,
                "ppg_path": str(obs.ppg_path) if obs.ppg_path is not None else "",
                "ppg_format": obs.ppg_format or "",
            }
        )

        eeg_values = {name: float("nan") for name in CORE_EEG_FEATURES}
        ppg_values = {name: float("nan") for name in CORE_PPG_FEATURES}
        eeg_error = "ok"
        ppg_error = "ok"
        n_bad_channels = 0
        eeg_channel = ""
        eeg_raw: mne.io.BaseRaw | None = None
        eeg_load_error: Exception | None = None

        eeg_cached = eeg_cache_lookup.get(str(obs.observation_id))
        channel_band_rows: list[dict[str, object]] = []
        if eeg_cached is not None:
            eeg_error = _coerce_str(eeg_cached.get("eeg_error"), default="cached")
            n_bad_channels = _coerce_int(eeg_cached.get("n_bad_channels"), default=0)
            eeg_channel = _coerce_str(eeg_cached.get("channel"), default="")
            for name in base_eeg_feature_columns:
                eeg_row[name] = _coerce_float(eeg_cached.get(name))
            for name in selected_eeg_features:
                eeg_row[name] = _coerce_float(eeg_cached.get(name))
            cached_base_row = dict(base)
            cached_base_row.update(
                {
                    "eeg_format": obs.eeg_format,
                    "eeg_path": str(obs.eeg_path),
                    "n_bad_channels": n_bad_channels,
                    "eeg_error": eeg_error,
                    "channel": eeg_channel,
                    **eeg_processing_metadata,
                }
            )
            cached_channel_rows = eeg_channel_cache_lookup.get(str(obs.observation_id), [])
            if cached_channel_rows:
                for cached_channel_row in cached_channel_rows:
                    ch_row = dict(base)
                    for name in base_eeg_power_columns("export"):
                        if name in OBSERVATION_KEY_COLUMNS:
                            continue
                        if name in cached_channel_row:
                            ch_row[name] = cached_channel_row[name]
                    ch_row.update(
                        {
                            "eeg_format": obs.eeg_format,
                            "eeg_path": str(obs.eeg_path),
                            "n_bad_channels": n_bad_channels,
                            "eeg_error": eeg_error,
                            **eeg_processing_metadata,
                        }
                    )
                    channel_band_rows.append(ch_row)
            else:
                for name in base_eeg_feature_columns:
                    cached_base_row[name] = _coerce_float(eeg_cached.get(name))
                channel_band_rows.append(cached_base_row)
        else:
            try:
                eeg_raw = _read_raw(obs.eeg_path, obs.eeg_format)
                prep = preprocess_eeg(
                    eeg_raw,
                    l_freq=cfg.eeg.l_freq,
                    h_freq=cfg.eeg.h_freq,
                    bad_channel_variance_z=cfg.eeg.bad_channel_variance_z,
                    reference=cfg.eeg.reference,
                )
                n_bad_channels = len(prep.bad_channels)
                eeg_channel = "|".join(prep.raw.ch_names)
                per_channel_band = channel_band_powers(
                    prep.raw,
                    band_names=("theta", "alpha", "beta"),
                    psd_fmin=cfg.eeg.psd.fmin,
                    psd_fmax=cfg.eeg.psd.fmax,
                    n_fft=cfg.eeg.psd.n_fft,
                )
                for ch_name, band_values in per_channel_band.items():
                    ch_row = dict(base)
                    ch_row.update(
                        {
                            "eeg_format": obs.eeg_format,
                            "eeg_path": str(obs.eeg_path),
                            "n_bad_channels": n_bad_channels,
                            "eeg_error": eeg_error,
                            "channel": ch_name,
                            **eeg_processing_metadata,
                        }
                    )
                    for name in base_eeg_feature_columns:
                        ch_row[name] = float(band_values.get(name, np.nan))
                    channel_band_rows.append(ch_row)
                eeg_values = _derive_eeg_feature_values_from_channel_rows(channel_band_rows)
            except Exception as exc:
                eeg_error = f"{type(exc).__name__}: {exc}"
                eeg_load_error = exc

        ppg_qc: dict[str, object] = {
            "ppg_channel": "",
            "ppg_segment_start_s": np.nan,
            "ppg_segment_end_s": np.nan,
            "n_ibi_clean": 0,
        }
        ppg_cached = ppg_cache_lookup.get(str(obs.observation_id))
        ppg_ibi_rows: list[dict[str, object]] = []
        if ppg_cached is not None:
            ppg_error = _coerce_str(ppg_cached.get("ppg_error"), default="cached")
            ppg_qc = {
                "ppg_channel": _coerce_str(ppg_cached.get("ppg_channel"), default=""),
                "ppg_segment_start_s": _coerce_float(ppg_cached.get("ppg_segment_start_s")),
                "ppg_segment_end_s": _coerce_float(ppg_cached.get("ppg_segment_end_s")),
                "n_ibi_clean": _coerce_int(ppg_cached.get("n_ibi_clean"), default=0),
            }
            for name in selected_ppg_features:
                ppg_row[name] = _coerce_float(ppg_cached.get(name))
            cached_ibi_rows = ppg_ibi_cache_lookup.get(str(obs.observation_id), [])
            if cached_ibi_rows:
                for cached_ibi_row in cached_ibi_rows:
                    ibi_row = dict(base)
                    for name in base_ppg_ibi_columns():
                        if name in OBSERVATION_KEY_COLUMNS:
                            continue
                        if name in cached_ibi_row:
                            ibi_row[name] = cached_ibi_row[name]
                    ibi_row.update(
                        {
                            "eeg_path": str(obs.eeg_path),
                            "eeg_format": obs.eeg_format,
                            "ppg_source": obs.ppg_source,
                            "ppg_path": str(obs.ppg_path) if obs.ppg_path is not None else "",
                            "ppg_format": obs.ppg_format or "",
                            "ppg_error": ppg_error,
                            "ppg_channel": ppg_qc["ppg_channel"],
                        }
                    )
                    ppg_ibi_rows.append(ibi_row)
        else:
            try:
                if obs.ppg_source == "embedded_eeg":
                    if eeg_raw is None:
                        if eeg_load_error is not None:
                            raise RuntimeError("EEG recording failed to load; cannot access embedded PPG.") from eeg_load_error
                        eeg_raw = _read_raw(obs.eeg_path, obs.eeg_format)
                    ppg_raw = eeg_raw
                elif obs.ppg_source == "external_file":
                    if obs.ppg_path is None or obs.ppg_format is None:
                        raise ValueError("External PPG source requires both ppg_path and ppg_format.")
                    ppg_raw = _read_raw(obs.ppg_path, obs.ppg_format)
                else:
                    raise ValueError(f"Unsupported ppg_source={obs.ppg_source!r}")

                ppg_values, ppg_qc, ppg_ibi_result, ppg_start_s, ppg_end_s = _ppg_feature_values(ppg_raw, cfg)
                ppg_ibi_rows = _ppg_ibi_rows_from_result(
                    base=base,
                    obs=obs,
                    cfg=cfg,
                    result=ppg_ibi_result,
                    start_s=ppg_start_s,
                    end_s=ppg_end_s,
                    ppg_error=ppg_error,
                )
            except Exception as exc:
                ppg_error = f"{type(exc).__name__}: {exc}"

        eeg_row["n_bad_channels"] = n_bad_channels
        eeg_row["eeg_error"] = eeg_error
        eeg_row["channel"] = eeg_channel
        for name in base_eeg_feature_columns:
            if name in eeg_row:
                continue
            eeg_row[name] = float(eeg_values.get(name, np.nan))
        for name in selected_eeg_features:
            if name in eeg_row:
                continue
            eeg_row[name] = float(eeg_values.get(name, np.nan))

        if not channel_band_rows:
            fallback_base_row = dict(base)
            fallback_base_row.update(
                {
                    "eeg_format": obs.eeg_format,
                    "eeg_path": str(obs.eeg_path),
                    "n_bad_channels": n_bad_channels,
                    "eeg_error": eeg_error,
                    "channel": eeg_channel,
                    **eeg_processing_metadata,
                }
            )
            for name in base_eeg_feature_columns:
                fallback_base_row[name] = float(eeg_row.get(name, eeg_values.get(name, np.nan)))
            channel_band_rows.append(fallback_base_row)

        ppg_row["ppg_error"] = ppg_error
        ppg_row.update(ppg_qc)
        for name in selected_ppg_features:
            if name in ppg_row:
                continue
            ppg_row[name] = float(ppg_values.get(name, np.nan))

        eeg_records.append(eeg_row)
        base_eeg_records.extend(channel_band_rows)
        ppg_records.append(ppg_row)
        ppg_ibi_records.extend(ppg_ibi_rows)

    eeg_df = pd.DataFrame(eeg_records)
    base_eeg_df = pd.DataFrame(base_eeg_records)
    ppg_df = pd.DataFrame(ppg_records)
    ppg_ibi_df = pd.DataFrame(ppg_ibi_records)

    if eeg_df.empty:
        eeg_columns = OBSERVATION_KEY_COLUMNS + [
            "eeg_path",
            "eeg_format",
            "ppg_source",
            "n_bad_channels",
            "eeg_error",
            "channel",
            *base_eeg_feature_columns,
            *selected_eeg_features,
        ]
        eeg_df = pd.DataFrame(columns=list(dict.fromkeys(eeg_columns)))
    if base_eeg_df.empty:
        base_eeg_df = pd.DataFrame(columns=base_eeg_power_columns("export"))
    else:
        ordered_base_cols = base_eeg_power_columns("export")
        for col in ordered_base_cols:
            if col not in base_eeg_df.columns:
                base_eeg_df[col] = np.nan
        base_eeg_df = base_eeg_df[ordered_base_cols]
    if ppg_df.empty:
        ppg_df = pd.DataFrame(columns=OBSERVATION_KEY_COLUMNS + ["eeg_path", "eeg_format", "ppg_source", "ppg_path", "ppg_format", "ppg_error", "ppg_channel", "ppg_segment_start_s", "ppg_segment_end_s", "n_ibi_clean"] + selected_ppg_features)
    if ppg_ibi_df.empty:
        ppg_ibi_df = pd.DataFrame(columns=base_ppg_ibi_columns())
    else:
        ordered_ppg_ibi_cols = base_ppg_ibi_columns()
        for col in ordered_ppg_ibi_cols:
            if col not in ppg_ibi_df.columns:
                ppg_ibi_df[col] = np.nan
        ppg_ibi_df = ppg_ibi_df[ordered_ppg_ibi_cols]

    if cfg.features.include_robust_z:
        eeg_df = add_dataset_local_robust_zscores(eeg_df, selected_eeg_features)
        ppg_df = add_dataset_local_robust_zscores(ppg_df, selected_ppg_features)

    merge_cols = OBSERVATION_KEY_COLUMNS
    merged_df = eeg_df.merge(
        ppg_df[[*merge_cols, *selected_ppg_features, *[f"{f}_rz" for f in selected_ppg_features if f"{f}_rz" in ppg_df.columns], "ppg_error", "ppg_channel", "ppg_segment_start_s", "ppg_segment_end_s", "n_ibi_clean"]],
        on=merge_cols,
        how="inner",
    )

    return FeatureExtractionResult(
        eeg_features=eeg_df,
        eeg_base_features=base_eeg_df,
        ppg_features=ppg_df,
        ppg_ibi_features=ppg_ibi_df,
        merged_features=merged_df,
    )
