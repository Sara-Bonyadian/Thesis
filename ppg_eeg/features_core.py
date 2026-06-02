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
    channel_band_powers,
    frontal_alpha_asymmetry,
    frontal_beta_value,
    frontal_midline_theta,
    global_band_value,
    preprocess_eeg,
)
from .ppg import extract_clean_ppg_ibi_from_raw, mean_hr_bpm, mean_rr_ms, peak_hr_bpm, rmssd_ms, sdnn_ms


CORE_EEG_FEATURES: list[str] = [
    "eeg_fm_theta",
    "eeg_frontal_beta",
    "eeg_faa",
    "eeg_global_alpha_db",
    "eeg_global_beta_db",
    "power_theta",
    "power_alpha",
    "power_beta",
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

def base_eeg_power_columns(kind: str = "export") -> list[str]:
    power_feature_columns = [
        "power_theta",
        "power_alpha",
        "power_beta",
    ]

    normalized = kind.casefold()
    if normalized in {"feature", "features"}:
        return list(power_feature_columns)
    if normalized == "export":
        return [
            "dataset_id",
            "observation_id",
            "subject_id",
            "task_label",
            "modality",
            "state",
            "eeg_format",
            "eeg_path",
            "n_bad_channels",
            "eeg_error",
            "channel",
            *power_feature_columns,
        ]
    raise ValueError(f"Unsupported base EEG column kind: {kind!r}")


@dataclass(frozen=True)
class FeatureExtractionResult:
    eeg_features: pd.DataFrame
    eeg_base_features: pd.DataFrame
    ppg_features: pd.DataFrame
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
) -> tuple[np.ndarray | None, str | None, float, float]:
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
        return result.ibi_ms_clean, result.ppg_channel, start_s, end_s

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
            return fallback.ibi_ms_clean, fallback.ppg_channel, 0.0, duration_s

    return None, result.ppg_channel, start_s, end_s


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


def _eeg_feature_values(preprocessed_raw: mne.io.BaseRaw, cfg: PipelineConfig) -> dict[str, float]:
    psd_kwargs = dict(
        psd_fmin=cfg.eeg.psd.fmin,
        psd_fmax=cfg.eeg.psd.fmax,
        n_fft=cfg.eeg.psd.n_fft,
    )

    band_power_values = {
        "power_theta": _safe_eval(
            lambda: global_band_value(preprocessed_raw, "theta", psd_fmin=cfg.eeg.psd.fmin, psd_fmax=cfg.eeg.psd.fmax)
        ),
        "power_alpha": _safe_eval(
            lambda: global_band_value(preprocessed_raw, "alpha", psd_fmin=cfg.eeg.psd.fmin, psd_fmax=cfg.eeg.psd.fmax)
        ),
        "power_beta": _safe_eval(
            lambda: global_band_value(preprocessed_raw, "beta", psd_fmin=cfg.eeg.psd.fmin, psd_fmax=cfg.eeg.psd.fmax)
        ),
    }

    return {
        **band_power_values,
        "eeg_fm_theta": _safe_eval(lambda: frontal_midline_theta(preprocessed_raw, **psd_kwargs)),
        "eeg_frontal_beta": _safe_eval(lambda: frontal_beta_value(preprocessed_raw, **psd_kwargs)),
        "eeg_faa": _safe_eval(lambda: frontal_alpha_asymmetry(preprocessed_raw, **psd_kwargs)),
        "eeg_global_alpha_db": _safe_eval(
            lambda: global_band_value(preprocessed_raw, "alpha", psd_fmin=cfg.eeg.psd.fmin, psd_fmax=cfg.eeg.psd.fmax)
        ),
        "eeg_global_beta_db": _safe_eval(
            lambda: global_band_value(preprocessed_raw, "beta", psd_fmin=cfg.eeg.psd.fmin, psd_fmax=cfg.eeg.psd.fmax)
        ),
    }


def _ppg_feature_values(raw: mne.io.BaseRaw, cfg: PipelineConfig) -> tuple[dict[str, float], dict[str, object]]:
    ibi_ms, ppg_channel, start_s, end_s = _extract_ppg_ibi_with_fallback(raw, cfg)
    qc: dict[str, object] = {
        "ppg_channel": ppg_channel or "",
        "ppg_segment_start_s": start_s,
        "ppg_segment_end_s": end_s,
        "n_ibi_clean": int(len(ibi_ms)) if ibi_ms is not None else 0,
    }

    if ibi_ms is None or len(ibi_ms) < 3:
        values = {name: float("nan") for name in CORE_PPG_FEATURES}
        return values, qc

    values = {
        "ppg_mean_hr_bpm": _safe_eval(lambda: mean_hr_bpm(ibi_ms)),
        "ppg_rmssd_ms": _safe_eval(lambda: rmssd_ms(ibi_ms)),
        "ppg_sdnn_ms": _safe_eval(lambda: sdnn_ms(ibi_ms)),
        "ppg_mean_rr_ms": _safe_eval(lambda: mean_rr_ms(ibi_ms)),
        "ppg_peak_hr_bpm": _safe_eval(lambda: peak_hr_bpm(ibi_ms)),
    }
    return values, qc


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


def _coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_str(value: object, *, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value)
    return text if text.strip() else default


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

    selected_eeg_features = list(cfg.features.eeg)
    selected_ppg_features = list(cfg.features.ppg)
    base_eeg_feature_columns = base_eeg_power_columns("feature")
    eeg_cache_lookup = _build_feature_cache_lookup(
        eeg_feature_cache,
        required_feature_columns=selected_eeg_features,
        table_name="EEG feature",
    )
    ppg_cache_lookup = _build_feature_cache_lookup(
        ppg_feature_cache,
        required_feature_columns=selected_ppg_features,
        table_name="PPG feature",
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
                eeg_row[name] = float(pd.to_numeric(eeg_cached.get(name), errors="coerce"))
            for name in selected_eeg_features:
                eeg_row[name] = float(pd.to_numeric(eeg_cached.get(name), errors="coerce"))
            cached_base_row = dict(base)
            cached_base_row.update(
                {
                    "eeg_format": obs.eeg_format,
                    "eeg_path": str(obs.eeg_path),
                    "n_bad_channels": n_bad_channels,
                    "eeg_error": eeg_error,
                    "channel": eeg_channel,
                }
            )
            for name in base_eeg_feature_columns:
                cached_base_row[name] = float(pd.to_numeric(eeg_cached.get(name), errors="coerce"))
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
                eeg_values = _eeg_feature_values(prep.raw, cfg)
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
                        }
                    )
                    for name in base_eeg_feature_columns:
                        ch_row[name] = float(band_values.get(name, np.nan))
                    channel_band_rows.append(ch_row)
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
        if ppg_cached is not None:
            ppg_error = _coerce_str(ppg_cached.get("ppg_error"), default="cached")
            ppg_qc = {
                "ppg_channel": _coerce_str(ppg_cached.get("ppg_channel"), default=""),
                "ppg_segment_start_s": float(pd.to_numeric(ppg_cached.get("ppg_segment_start_s"), errors="coerce")),
                "ppg_segment_end_s": float(pd.to_numeric(ppg_cached.get("ppg_segment_end_s"), errors="coerce")),
                "n_ibi_clean": _coerce_int(ppg_cached.get("n_ibi_clean"), default=0),
            }
            for name in selected_ppg_features:
                ppg_row[name] = float(pd.to_numeric(ppg_cached.get(name), errors="coerce"))
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

                ppg_values, ppg_qc = _ppg_feature_values(ppg_raw, cfg)
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

    eeg_df = pd.DataFrame(eeg_records)
    base_eeg_df = pd.DataFrame(base_eeg_records)
    ppg_df = pd.DataFrame(ppg_records)

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
        merged_features=merged_df,
    )
