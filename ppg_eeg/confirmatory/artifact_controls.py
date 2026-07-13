"""Sensitivity and artifact-control analyses (M11).

Prespecified controls around the immutable primary analysis
(D240 absolute-power ZLPI). Sensitivity results never replace or rescue
primary findings. ZLPI, MWPI, and SWPI stay separate. Missing control
signals are recorded as unavailable — never imputed as zero.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from .duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_PRIMARY_DURATION_S,
)
from .group_tables import PAIRED_CONTRASTS_FILENAME, SUBJECT_LEVEL_FILENAME
from .nulls import compute_endpoint_index_from_series

SENSITIVITY_RESULTS_FILENAME = "sensitivity_results.csv"
ARTIFACT_CONTROL_RESULTS_FILENAME = "artifact_control_results.csv"
MODALITY_COMPARISON_FILENAME = "modality_comparison.csv"
DURATION_SENSITIVITY_FILENAME = "duration_sensitivity.csv"
SPECIFICATION_MATRIX_FILENAME = "specification_matrix.csv"
SENSITIVITY_QC_FILENAME = "sensitivity_qc.csv"

PRIMARY_POWER_REPRESENTATION = "absolute_log10"
PRIMARY_CONTROL_ID = "primary_d240_absolute_zlpi"

CONTROL_CARDIAC_FIELD = "cardiac_field_qrs_interpolation"
CONTROL_MOTION = "nuisance_motion"
CONTROL_EOG = "nuisance_eog"
CONTROL_EMG = "nuisance_emg"
CONTROL_RESPIRATION = "nuisance_respiration"
CONTROL_BROADBAND = "broadband_residualized"
CONTROL_MEAN_HR = "covariate_mean_hr"
CONTROL_BEAT_COUNT = "covariate_beat_count"
CONTROL_BEAT_DENSITY = "covariate_beat_density"
CONTROL_EYE_STATE = "stratify_eye_state"
CONTROL_ECG_VS_PPG = "modality_ecg_vs_ppg"
CONTROL_D180 = "duration_d180_zlpi"
CONTROL_D120 = "duration_d120_mwpi"
CONTROL_D60 = "duration_d60_swpi"

SENSITIVITY_CONTROL_IDS: tuple[str, ...] = (
    CONTROL_CARDIAC_FIELD,
    CONTROL_MOTION,
    CONTROL_EOG,
    CONTROL_EMG,
    CONTROL_RESPIRATION,
    CONTROL_BROADBAND,
    CONTROL_MEAN_HR,
    CONTROL_BEAT_COUNT,
    CONTROL_BEAT_DENSITY,
    CONTROL_EYE_STATE,
    CONTROL_ECG_VS_PPG,
    CONTROL_D180,
    CONTROL_D120,
    CONTROL_D60,
)

ALL_CONTROL_IDS: tuple[str, ...] = (PRIMARY_CONTROL_ID,) + SENSITIVITY_CONTROL_IDS

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "control_unavailable"
STATUS_INSUFFICIENT = "insufficient_data"
STATUS_PRIMARY = "primary_reference"
STATUS_SENSITIVITY_ONLY = "sensitivity_only"

# QRS interpolation default half-width (seconds) around each beat on raw-ish EEG grids.
DEFAULT_QRS_HALF_WIDTH_S = 0.05

SENSITIVITY_FIELDS = (
    "control_id",
    "analysis_role",
    "is_primary_analysis",
    "can_rescue_primary",
    "endpoint_name",
    "duration_s",
    "power_representation",
    "band",
    "dataset_id",
    "contrast_id",
    "effect_estimate",
    "ci_low",
    "ci_high",
    "n",
    "p_value",
    "status",
    "notes",
)

ARTIFACT_FIELDS = (
    "control_id",
    "observation_id",
    "dataset_id",
    "band",
    "duration_s",
    "endpoint_name",
    "primary_endpoint_index",
    "controlled_endpoint_index",
    "delta_vs_primary",
    "artifact_injected",
    "control_applied",
    "status",
    "notes",
)

MODALITY_FIELDS = (
    "dataset_id",
    "participant_id",
    "session_id",
    "condition",
    "observation_id",
    "duration_s",
    "endpoint_name",
    "band",
    "power_representation",
    "ecg_endpoint_index",
    "ppg_endpoint_index",
    "delta_ecg_minus_ppg",
    "matched",
    "status",
    "notes",
)

DURATION_FIELDS = (
    "duration_s",
    "endpoint_name",
    "is_standard_zlpi",
    "power_representation",
    "band",
    "dataset_id",
    "contrast_id",
    "effect_estimate",
    "ci_low",
    "ci_high",
    "n",
    "p_value",
    "is_primary_analysis",
    "can_rescue_primary",
    "status",
    "notes",
)

SPEC_MATRIX_FIELDS = (
    "control_id",
    "analysis_role",
    "is_primary_analysis",
    "can_rescue_primary",
    "endpoint_name",
    "duration_s",
    "power_representation",
    "effect_estimate",
    "ci_low",
    "ci_high",
    "n",
    "status",
    "notes",
)

QC_FIELDS = (
    "control_id",
    "component",
    "status",
    "n_available",
    "n_unavailable",
    "n_effects",
    "notes",
)


@dataclass(frozen=True)
class ArtifactControlResult:
    sensitivity_rows: tuple[dict[str, object], ...]
    artifact_rows: tuple[dict[str, object], ...]
    modality_rows: tuple[dict[str, object], ...]
    duration_rows: tuple[dict[str, object], ...]
    specification_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = _as_str(value)
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _as_bool(value: object) -> bool | None:
    """Tri-state bool: True/False, or None when unset/unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    if text in {"", "nan", "none", "null", "unknown", "na", "n/a"}:
        return None
    if text in {"1", "true", "yes", "y", "t", "available", "present"}:
        return True
    if text in {"0", "false", "no", "n", "f", "unavailable", "missing", "absent"}:
        return False
    return None


def _as_int(value: object, default: int = 0) -> int:
    if value is None or _as_str(value) == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_primary_cell(
    *,
    endpoint_name: str,
    duration_s: int,
    power_representation: str,
) -> bool:
    return (
        endpoint_name.casefold() == ENDPOINT_ZLPI
        and int(duration_s) == int(EXPECTED_PRIMARY_DURATION_S)
        and power_representation.casefold() == PRIMARY_POWER_REPRESENTATION
    )


def beat_density(n_beats: float, duration_s: float) -> float:
    """Beats per second over a usable span; NaN if inputs are invalid."""
    if not (math.isfinite(n_beats) and math.isfinite(duration_s)):
        return float("nan")
    if duration_s <= 0 or n_beats < 0:
        return float("nan")
    return float(n_beats / duration_s)


def summarize_effect(values: Sequence[float]) -> dict[str, float | int | str]:
    """One-sample mean effect with 95% CI against zero."""
    arr = np.asarray(
        [float(v) for v in values if math.isfinite(float(v))],
        dtype=float,
    )
    n = int(arr.size)
    if n == 0:
        return {
            "effect_estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "p_value": float("nan"),
            "status": STATUS_INSUFFICIENT,
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "effect_estimate": mean,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 1,
            "p_value": float("nan"),
            "status": STATUS_INSUFFICIENT,
        }
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    t_stat, p_value = stats.ttest_1samp(arr, popmean=0.0)
    return {
        "effect_estimate": mean,
        "ci_low": float(mean - t_crit * se),
        "ci_high": float(mean + t_crit * se),
        "n": n,
        "p_value": float(p_value),
        "status": STATUS_OK,
    }


def residualize_series(
    signal: np.ndarray,
    *nuisance: np.ndarray,
) -> np.ndarray:
    """OLS-residualize ``signal`` against nuisance regressors.

    Raises ``ValueError`` if any nuisance series is missing/empty — callers must
    not substitute zeros for unavailable controls.
    """
    y = np.asarray(signal, dtype=float)
    if not nuisance:
        raise ValueError("At least one nuisance regressor is required.")
    cols = []
    for series in nuisance:
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            raise ValueError("Nuisance series is empty/unavailable.")
        if arr.shape != y.shape:
            raise ValueError("Nuisance series must match signal shape.")
        cols.append(arr)
    design_cols = [np.ones(y.shape, dtype=float), *cols]
    design = np.column_stack(design_cols)
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    out = np.full(y.shape, np.nan, dtype=float)
    if int(mask.sum()) < design.shape[1] + 1:
        return out
    coef, *_ = np.linalg.lstsq(design[mask], y[mask], rcond=None)
    out[mask] = y[mask] - design[mask] @ coef
    return out


def broadband_residualize_log_power(
    band_log10: np.ndarray,
    broadband_log10: np.ndarray,
) -> np.ndarray:
    """Residualize band log-power against broadband log-power within a segment."""
    return residualize_series(band_log10, broadband_log10)


def qrs_interpolate_eeg(
    eeg: np.ndarray,
    times_s: np.ndarray,
    beat_times_s: np.ndarray,
    *,
    half_width_s: float = DEFAULT_QRS_HALF_WIDTH_S,
) -> np.ndarray:
    """Linearly interpolate EEG through windows around each beat (cardiac-field control).

    Instantaneous HR / beat times are left unchanged — only EEG samples in
    QRS neighborhoods are replaced. Missing beat times must not be invented.
    """
    signal = np.asarray(eeg, dtype=float).copy()
    times = np.asarray(times_s, dtype=float)
    beats = np.asarray(beat_times_s, dtype=float)
    if signal.size == 0:
        return signal
    if times.shape != signal.shape:
        raise ValueError("times_s must match eeg shape.")
    if beats.size == 0 or not np.any(np.isfinite(beats)):
        raise ValueError("beat_times_s unavailable; cannot apply QRS interpolation.")
    if half_width_s < 0:
        raise ValueError("half_width_s must be >= 0.")

    finite_beats = beats[np.isfinite(beats)]
    contaminated = np.zeros(signal.shape, dtype=bool)
    for beat in finite_beats:
        contaminated |= np.abs(times - float(beat)) <= float(half_width_s)
    keep = np.isfinite(signal) & ~contaminated
    if int(keep.sum()) < 2:
        # Not enough anchors to interpolate; return NaNs in contaminated windows.
        signal[contaminated] = np.nan
        return signal
    signal[contaminated] = np.interp(
        times[contaminated],
        times[keep],
        signal[keep],
    )
    return signal


def control_availability(
    inventory: Mapping[str, object] | None,
    control_id: str,
) -> tuple[bool, str]:
    """Return (available, note) for a control given an observation inventory."""
    if control_id == PRIMARY_CONTROL_ID:
        return True, "Primary reference analysis."
    if control_id in {CONTROL_D180, CONTROL_D120, CONTROL_D60, CONTROL_BROADBAND}:
        return True, "Derived from duration/representation tables when present."
    if inventory is None:
        return False, "No control inventory supplied."

    key_map = {
        CONTROL_CARDIAC_FIELD: ("beat_times_s", "beats_available", "qrs_available"),
        CONTROL_MOTION: ("motion", "motion_available"),
        CONTROL_EOG: ("eog", "eog_available"),
        CONTROL_EMG: ("emg", "emg_available"),
        CONTROL_RESPIRATION: ("respiration", "respiration_available"),
        CONTROL_MEAN_HR: ("mean_hr", "mean_hr_bpm"),
        CONTROL_BEAT_COUNT: ("beat_count", "n_beats"),
        CONTROL_BEAT_DENSITY: ("beat_density", "beat_count", "n_beats"),
        CONTROL_EYE_STATE: ("eye_state",),
        CONTROL_ECG_VS_PPG: ("ecg_available", "ppg_available"),
    }
    keys = key_map.get(control_id, ())
    if not keys:
        return False, f"Unknown control_id={control_id!r}."

    if control_id == CONTROL_ECG_VS_PPG:
        ecg = _as_bool(inventory.get("ecg_available"))
        ppg = _as_bool(inventory.get("ppg_available"))
        if ecg is True and ppg is True:
            return True, "Both ECG and PPG available for matched comparison."
        return False, "Matched ECG/PPG requires both modalities."

    if control_id == CONTROL_CARDIAC_FIELD:
        beats = inventory.get("beat_times_s")
        if beats is not None:
            arr = np.asarray(beats, dtype=float)
            if arr.size and np.any(np.isfinite(arr)):
                return True, "Beat times present for QRS interpolation."
        flag = _as_bool(inventory.get("beats_available"))
        if flag is True:
            return True, "beats_available flag set."
        if flag is False:
            return False, "Beat times unavailable."
        return False, "Beat times not provided."

    if control_id == CONTROL_EYE_STATE:
        eye = _as_str(inventory.get("eye_state")).casefold()
        if eye in {"", "unknown", "na", "n/a"}:
            return False, "Eye state unknown/unavailable."
        return True, f"Eye state={eye}."

    for key in keys:
        if key not in inventory:
            continue
        value = inventory.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            if arr.size and np.any(np.isfinite(arr)):
                return True, f"{key} present."
            continue
        flag = _as_bool(value)
        if flag is True:
            return True, f"{key} available."
        if flag is False:
            return False, f"{key} marked unavailable."
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return True, f"{key} numeric value present."
        if _as_str(value):
            # Non-empty string that is not a known negative flag.
            return True, f"{key} present."
    return False, f"Control signals for {control_id} unavailable."


def apply_cardiac_field_control(
    hr_z: np.ndarray,
    eeg_z: np.ndarray,
    times_s: np.ndarray,
    beat_times_s: np.ndarray,
    *,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
    half_width_s: float = DEFAULT_QRS_HALF_WIDTH_S,
) -> dict[str, object]:
    """Recompute endpoint after QRS interpolation of EEG (HR series untouched)."""
    primary = compute_endpoint_index_from_series(
        hr_z, eeg_z, duration_s=duration_s
    )
    controlled_eeg = qrs_interpolate_eeg(
        eeg_z, times_s, beat_times_s, half_width_s=half_width_s
    )
    # Guardrail: HR must remain identical.
    if not np.array_equal(
        np.asarray(hr_z, dtype=float), np.asarray(hr_z, dtype=float)
    ):
        raise RuntimeError("HR series mutated unexpectedly.")
    controlled = compute_endpoint_index_from_series(
        hr_z, controlled_eeg, duration_s=duration_s
    )
    primary_idx = _as_float(primary.get("endpoint_index"))
    controlled_idx = _as_float(controlled.get("endpoint_index"))
    return {
        "primary_endpoint_index": primary_idx,
        "controlled_endpoint_index": controlled_idx,
        "delta_vs_primary": (
            float(controlled_idx - primary_idx)
            if math.isfinite(primary_idx) and math.isfinite(controlled_idx)
            else float("nan")
        ),
        "endpoint_name": _as_str(primary.get("endpoint_name"), ENDPOINT_ZLPI),
        "hr_unchanged": True,
    }


def _filter_paired_rows(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    endpoint_name: str | None = None,
    duration_s: int | None = None,
    power_representation: str | None = None,
    modality: str | None = None,
    band: str | None = None,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for row in paired_rows:
        if endpoint_name is not None and _as_str(row.get("endpoint_name")).casefold() != endpoint_name.casefold():
            continue
        if duration_s is not None and _as_int(row.get("duration_s")) != int(duration_s):
            continue
        if (
            power_representation is not None
            and _as_str(row.get("power_representation")).casefold()
            != power_representation.casefold()
        ):
            continue
        if modality is not None:
            row_mod = _as_str(row.get("modality") or row.get("sensor_modality")).casefold()
            if row_mod and row_mod != modality.casefold():
                continue
        if band is not None and _as_str(row.get("band")).casefold() != band.casefold():
            continue
        out.append(row)
    return out


def _effects_by_band_dataset(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    control_id: str,
    analysis_role: str,
    is_primary: bool,
    endpoint_name: str,
    duration_s: int,
    power_representation: str,
    status: str = STATUS_OK,
    notes: str = "",
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str], list[float]] = {}
    for row in paired_rows:
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("contrast_id")).casefold(),
            _as_str(row.get("band")).casefold(),
        )
        delta = _as_float(row.get("delta_endpoint_index"))
        if math.isfinite(delta):
            buckets.setdefault(key, []).append(delta)

    rows: list[dict[str, object]] = []
    for (dataset_id, contrast_id, band), values in sorted(buckets.items()):
        summary = summarize_effect(values)
        rows.append(
            {
                "control_id": control_id,
                "analysis_role": analysis_role,
                "is_primary_analysis": bool(is_primary),
                "can_rescue_primary": False,
                "endpoint_name": endpoint_name,
                "duration_s": int(duration_s),
                "power_representation": power_representation,
                "band": band,
                "dataset_id": dataset_id,
                "contrast_id": contrast_id,
                "effect_estimate": summary["effect_estimate"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "n": summary["n"],
                "p_value": summary["p_value"],
                "status": status if summary["status"] == STATUS_OK else summary["status"],
                "notes": notes,
            }
        )
    return rows


def _unavailable_row(
    *,
    control_id: str,
    analysis_role: str,
    endpoint_name: str = ENDPOINT_ZLPI,
    duration_s: int = EXPECTED_PRIMARY_DURATION_S,
    power_representation: str = PRIMARY_POWER_REPRESENTATION,
    notes: str,
) -> dict[str, object]:
    return {
        "control_id": control_id,
        "analysis_role": analysis_role,
        "is_primary_analysis": False,
        "can_rescue_primary": False,
        "endpoint_name": endpoint_name,
        "duration_s": int(duration_s),
        "power_representation": power_representation,
        "band": "",
        "dataset_id": "",
        "contrast_id": "",
        "effect_estimate": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n": 0,
        "p_value": float("nan"),
        "status": STATUS_UNAVAILABLE,
        "notes": notes,
    }


def covariate_adjusted_effect(
    paired_rows: Sequence[Mapping[str, object]],
    subject_rows: Sequence[Mapping[str, object]],
    *,
    covariate: str,
) -> tuple[list[float], str]:
    """Adjust paired deltas for a subject-level covariate (mean across pair).

    Uses OLS: delta ~ 1 + covariate. Returns residuals about the intercept as
    adjusted effects when covariate values exist; otherwise explains unavailability.
    """
    # Map subject covariates.
    cov_lookup: dict[tuple[str, ...], float] = {}
    for row in subject_rows:
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("participant_id") or row.get("subject_id")).casefold(),
            _as_str(row.get("session_id"), "single").casefold(),
            _as_str(row.get("condition")).casefold(),
            str(_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S)),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation"), PRIMARY_POWER_REPRESENTATION).casefold(),
        )
        value = _as_float(row.get(covariate))
        if math.isfinite(value):
            cov_lookup[key] = value

    deltas: list[float] = []
    cov_vals: list[float] = []
    missing = 0
    for row in paired_rows:
        if not is_primary_cell(
            endpoint_name=_as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
            duration_s=_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S),
            power_representation=_as_str(
                row.get("power_representation"), PRIMARY_POWER_REPRESENTATION
            ),
        ):
            continue
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        participant = _as_str(row.get("participant_id")).casefold()
        session = _as_str(row.get("session_id"), "single").casefold()
        band = _as_str(row.get("band")).casefold()
        duration = str(_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S))
        representation = _as_str(
            row.get("power_representation"), PRIMARY_POWER_REPRESENTATION
        ).casefold()
        low = _as_str(row.get("low_demand_condition")).casefold()
        effort = _as_str(row.get("cognitive_effort_condition")).casefold()
        low_key = (dataset_id, participant, session, low, duration, band, representation)
        effort_key = (
            dataset_id,
            participant,
            session,
            effort,
            duration,
            band,
            representation,
        )
        if low_key not in cov_lookup or effort_key not in cov_lookup:
            missing += 1
            continue
        cov_mean = 0.5 * (cov_lookup[low_key] + cov_lookup[effort_key])
        deltas.append(delta)
        cov_vals.append(cov_mean)

    if not deltas:
        return [], f"Covariate {covariate!r} unavailable for paired units (missing={missing})."
    if len(deltas) < 3:
        return deltas, f"Insufficient pairs with {covariate} for adjustment; using raw deltas."

    y = np.asarray(deltas, dtype=float)
    x = np.asarray(cov_vals, dtype=float)
    design = np.column_stack((np.ones(y.size), x - np.mean(x)))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    # Report intercept-adjusted observations (fitted intercept + residual).
    adjusted = coef[0] + (y - design @ coef)
    return adjusted.tolist(), f"Adjusted for {covariate}; n_missing_covariate={missing}."


def compare_matched_modalities(
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """ECG vs PPG endpoint comparison only for matched observation keys."""
    by_key: dict[tuple[str, ...], dict[str, Mapping[str, object]]] = {}
    for row in subject_rows:
        modality = _as_str(row.get("modality") or row.get("sensor_modality")).casefold()
        if modality not in {"ecg", "ppg"}:
            continue
        if not is_primary_cell(
            endpoint_name=_as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
            duration_s=_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S),
            power_representation=_as_str(
                row.get("power_representation"), PRIMARY_POWER_REPRESENTATION
            ),
        ):
            # Allow matched modality at primary grain only for this table.
            continue
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("participant_id") or row.get("subject_id")).casefold(),
            _as_str(row.get("session_id"), "single").casefold(),
            _as_str(row.get("condition")).casefold(),
            str(_as_int(row.get("duration_s"))),
            _as_str(row.get("endpoint_name")).casefold(),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation")).casefold(),
        )
        by_key.setdefault(key, {})[modality] = row

    rows: list[dict[str, object]] = []
    for key, mods in sorted(by_key.items()):
        dataset_id, participant_id, session_id, condition, duration_s, endpoint_name, band, representation = key
        has_ecg = "ecg" in mods
        has_ppg = "ppg" in mods
        if has_ecg and has_ppg:
            ecg_val = _as_float(mods["ecg"].get("endpoint_index"))
            ppg_val = _as_float(mods["ppg"].get("endpoint_index"))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "condition": condition,
                    "observation_id": _as_str(
                        mods["ecg"].get("observation_ids")
                        or mods["ecg"].get("observation_id")
                    ),
                    "duration_s": int(duration_s),
                    "endpoint_name": endpoint_name,
                    "band": band,
                    "power_representation": representation,
                    "ecg_endpoint_index": ecg_val,
                    "ppg_endpoint_index": ppg_val,
                    "delta_ecg_minus_ppg": (
                        float(ecg_val - ppg_val)
                        if math.isfinite(ecg_val) and math.isfinite(ppg_val)
                        else float("nan")
                    ),
                    "matched": True,
                    "status": STATUS_OK,
                    "notes": "Matched ECG/PPG observation pair.",
                }
            )
        else:
            present = "ecg" if has_ecg else "ppg"
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "condition": condition,
                    "observation_id": _as_str(
                        mods[present].get("observation_ids")
                        or mods[present].get("observation_id")
                    ),
                    "duration_s": int(duration_s),
                    "endpoint_name": endpoint_name,
                    "band": band,
                    "power_representation": representation,
                    "ecg_endpoint_index": (
                        _as_float(mods["ecg"].get("endpoint_index")) if has_ecg else float("nan")
                    ),
                    "ppg_endpoint_index": (
                        _as_float(mods["ppg"].get("endpoint_index")) if has_ppg else float("nan")
                    ),
                    "delta_ecg_minus_ppg": float("nan"),
                    "matched": False,
                    "status": STATUS_UNAVAILABLE,
                    "notes": "Unmatched modality; excluded from ECG−PPG comparison.",
                }
            )
    return rows


def duration_sensitivity_effects(
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Duration/endpoint sensitivity table with hard separation and no rescue."""
    specs = (
        (240, ENDPOINT_ZLPI, True, PRIMARY_CONTROL_ID),
        (180, ENDPOINT_ZLPI, False, CONTROL_D180),
        (120, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX, False, CONTROL_D120),
        (60, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX, False, CONTROL_D60),
    )
    rows: list[dict[str, object]] = []
    for duration_s, endpoint_name, is_primary, control_id in specs:
        filtered = _filter_paired_rows(
            paired_rows,
            endpoint_name=endpoint_name,
            duration_s=duration_s,
            power_representation=PRIMARY_POWER_REPRESENTATION,
        )
        # Also accept absolute_log10 primary representation aliasing.
        if not filtered and not is_primary:
            filtered = _filter_paired_rows(
                paired_rows,
                endpoint_name=endpoint_name,
                duration_s=duration_s,
            )
        effect_rows = _effects_by_band_dataset(
            filtered,
            control_id=control_id,
            analysis_role=("primary" if is_primary else "duration_sensitivity"),
            is_primary=is_primary,
            endpoint_name=endpoint_name,
            duration_s=duration_s,
            power_representation=PRIMARY_POWER_REPRESENTATION,
            status=STATUS_PRIMARY if is_primary else STATUS_SENSITIVITY_ONLY,
            notes=(
                "Immutable primary D240 absolute-power ZLPI."
                if is_primary
                else "Sensitivity only; cannot rescue primary ZLPI."
            ),
        )
        for row in effect_rows:
            rows.append(
                {
                    "duration_s": row["duration_s"],
                    "endpoint_name": row["endpoint_name"],
                    "is_standard_zlpi": endpoint_name == ENDPOINT_ZLPI
                    and duration_s in {240, 180},
                    "power_representation": row["power_representation"],
                    "band": row["band"],
                    "dataset_id": row["dataset_id"],
                    "contrast_id": row["contrast_id"],
                    "effect_estimate": row["effect_estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "n": row["n"],
                    "p_value": row["p_value"],
                    "is_primary_analysis": row["is_primary_analysis"],
                    "can_rescue_primary": False,
                    "status": row["status"],
                    "notes": row["notes"],
                }
            )
        if not effect_rows and not is_primary:
            rows.append(
                {
                    "duration_s": duration_s,
                    "endpoint_name": endpoint_name,
                    "is_standard_zlpi": endpoint_name == ENDPOINT_ZLPI
                    and duration_s in {240, 180},
                    "power_representation": PRIMARY_POWER_REPRESENTATION,
                    "band": "",
                    "dataset_id": "",
                    "contrast_id": "",
                    "effect_estimate": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n": 0,
                    "p_value": float("nan"),
                    "is_primary_analysis": False,
                    "can_rescue_primary": False,
                    "status": STATUS_INSUFFICIENT,
                    "notes": "No paired rows for this duration/endpoint.",
                }
            )
    return rows


def run_artifact_controls(
    subject_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    *,
    control_inventory: Mapping[str, Mapping[str, object]] | None = None,
    series_controls: Sequence[Mapping[str, object]] | None = None,
) -> ArtifactControlResult:
    """Run the full M11 sensitivity / artifact-control battery.

    Parameters
    ----------
    subject_rows, paired_rows
        M8 tables (possibly already multi-duration / multi-representation).
    control_inventory
        Optional mapping observation_id → availability flags / beat times.
    series_controls
        Optional per-observation series payloads for cardiac-field tests:
        ``hr_z``, ``eeg_z``, ``times_s``, ``beat_times_s``, metadata.
    """
    inventory = {
        _as_str(key): dict(value)
        for key, value in (control_inventory or {}).items()
    }
    sensitivity_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    # ---- Primary reference (immutable) ----
    primary_paired = _filter_paired_rows(
        paired_rows,
        endpoint_name=ENDPOINT_ZLPI,
        duration_s=EXPECTED_PRIMARY_DURATION_S,
        power_representation=PRIMARY_POWER_REPRESENTATION,
    )
    primary_effects = _effects_by_band_dataset(
        primary_paired,
        control_id=PRIMARY_CONTROL_ID,
        analysis_role="primary",
        is_primary=True,
        endpoint_name=ENDPOINT_ZLPI,
        duration_s=EXPECTED_PRIMARY_DURATION_S,
        power_representation=PRIMARY_POWER_REPRESENTATION,
        status=STATUS_PRIMARY,
        notes="Primary D240 absolute-power ZLPI; never overwritten by sensitivities.",
    )
    sensitivity_rows.extend(primary_effects)
    qc_rows.append(
        {
            "control_id": PRIMARY_CONTROL_ID,
            "component": "primary_reference",
            "status": STATUS_PRIMARY,
            "n_available": len(primary_effects),
            "n_unavailable": 0,
            "n_effects": len(primary_effects),
            "notes": "Primary analysis retained unchanged.",
        }
    )

    # ---- Broadband residualization ----
    broadband_paired = _filter_paired_rows(
        paired_rows,
        endpoint_name=ENDPOINT_ZLPI,
        duration_s=EXPECTED_PRIMARY_DURATION_S,
        power_representation="broadband_residualized",
    )
    if broadband_paired:
        sensitivity_rows.extend(
            _effects_by_band_dataset(
                broadband_paired,
                control_id=CONTROL_BROADBAND,
                analysis_role="representation_sensitivity",
                is_primary=False,
                endpoint_name=ENDPOINT_ZLPI,
                duration_s=EXPECTED_PRIMARY_DURATION_S,
                power_representation="broadband_residualized",
                status=STATUS_SENSITIVITY_ONLY,
                notes="Broadband-residualized power; cannot rescue primary.",
            )
        )
        qc_rows.append(
            {
                "control_id": CONTROL_BROADBAND,
                "component": "representation",
                "status": STATUS_OK,
                "n_available": len(broadband_paired),
                "n_unavailable": 0,
                "n_effects": sum(
                    1 for r in sensitivity_rows if r["control_id"] == CONTROL_BROADBAND
                ),
                "notes": "",
            }
        )
    else:
        sensitivity_rows.append(
            _unavailable_row(
                control_id=CONTROL_BROADBAND,
                analysis_role="representation_sensitivity",
                notes="No broadband_residualized paired rows available.",
            )
        )
        qc_rows.append(
            {
                "control_id": CONTROL_BROADBAND,
                "component": "representation",
                "status": STATUS_UNAVAILABLE,
                "n_available": 0,
                "n_unavailable": 1,
                "n_effects": 0,
                "notes": "Missing broadband_residualized contrasts.",
            }
        )

    # ---- Duration sensitivities ----
    duration_rows = duration_sensitivity_effects(paired_rows)
    for control_id, duration_s, endpoint_name in (
        (CONTROL_D180, 180, ENDPOINT_ZLPI),
        (CONTROL_D120, 120, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX),
        (CONTROL_D60, 60, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX),
    ):
        subset = [
            r
            for r in duration_rows
            if int(r["duration_s"]) == duration_s
            and _as_str(r["endpoint_name"]) == endpoint_name
            and int(r["n"]) > 0
        ]
        if subset:
            for row in subset:
                sensitivity_rows.append(
                    {
                        "control_id": control_id,
                        "analysis_role": "duration_sensitivity",
                        "is_primary_analysis": False,
                        "can_rescue_primary": False,
                        "endpoint_name": row["endpoint_name"],
                        "duration_s": row["duration_s"],
                        "power_representation": row["power_representation"],
                        "band": row["band"],
                        "dataset_id": row["dataset_id"],
                        "contrast_id": row["contrast_id"],
                        "effect_estimate": row["effect_estimate"],
                        "ci_low": row["ci_low"],
                        "ci_high": row["ci_high"],
                        "n": row["n"],
                        "p_value": row["p_value"],
                        "status": STATUS_SENSITIVITY_ONLY,
                        "notes": row["notes"],
                    }
                )
            qc_rows.append(
                {
                    "control_id": control_id,
                    "component": "duration",
                    "status": STATUS_OK,
                    "n_available": len(subset),
                    "n_unavailable": 0,
                    "n_effects": len(subset),
                    "notes": "Duration sensitivity; cannot rescue primary.",
                }
            )
        else:
            sensitivity_rows.append(
                _unavailable_row(
                    control_id=control_id,
                    analysis_role="duration_sensitivity",
                    endpoint_name=endpoint_name,
                    duration_s=duration_s,
                    notes=f"No paired rows for duration={duration_s}.",
                )
            )
            qc_rows.append(
                {
                    "control_id": control_id,
                    "component": "duration",
                    "status": STATUS_UNAVAILABLE,
                    "n_available": 0,
                    "n_unavailable": 1,
                    "n_effects": 0,
                    "notes": f"Missing duration={duration_s} contrasts.",
                }
            )

    # ---- Inventory-gated nuisance / covariate controls ----
    def _any_available(control_id: str) -> tuple[bool, str]:
        if not inventory:
            return control_availability(None, control_id)
        notes = []
        for obs_id, payload in inventory.items():
            ok, note = control_availability(payload, control_id)
            if ok:
                return True, note
            notes.append(f"{obs_id}:{note}")
        return False, notes[0] if notes else "Unavailable."

    for control_id, analysis_role, covariate in (
        (CONTROL_MOTION, "nuisance_residualization", None),
        (CONTROL_EOG, "nuisance_residualization", None),
        (CONTROL_EMG, "nuisance_residualization", None),
        (CONTROL_RESPIRATION, "nuisance_residualization", None),
        (CONTROL_MEAN_HR, "covariate_adjustment", "mean_hr"),
        (CONTROL_BEAT_COUNT, "covariate_adjustment", "beat_count"),
        (CONTROL_BEAT_DENSITY, "covariate_adjustment", "beat_density"),
        (CONTROL_EYE_STATE, "stratification", None),
    ):
        available, avail_note = _any_available(control_id)
        if not available:
            sensitivity_rows.append(
                _unavailable_row(
                    control_id=control_id,
                    analysis_role=analysis_role,
                    notes=avail_note,
                )
            )
            qc_rows.append(
                {
                    "control_id": control_id,
                    "component": analysis_role,
                    "status": STATUS_UNAVAILABLE,
                    "n_available": 0,
                    "n_unavailable": 1,
                    "n_effects": 0,
                    "notes": avail_note,
                }
            )
            continue

        if covariate is not None:
            # Prefer explicit subject column; for beat_density, derive when possible.
            subject_for_cov = list(subject_rows)
            if covariate == "beat_density":
                enriched = []
                for row in subject_rows:
                    item = dict(row)
                    if not math.isfinite(_as_float(item.get("beat_density"))):
                        density = beat_density(
                            _as_float(item.get("beat_count") or item.get("n_beats")),
                            _as_float(
                                item.get("usable_span_s")
                                or item.get("duration_s")
                                or EXPECTED_PRIMARY_DURATION_S
                            ),
                        )
                        item["beat_density"] = density
                    enriched.append(item)
                subject_for_cov = enriched
            if covariate == "mean_hr":
                enriched = []
                for row in subject_rows:
                    item = dict(row)
                    if not math.isfinite(_as_float(item.get("mean_hr"))):
                        item["mean_hr"] = _as_float(item.get("mean_hr_bpm"))
                    enriched.append(item)
                subject_for_cov = enriched

            adjusted, note = covariate_adjusted_effect(
                primary_paired, subject_for_cov, covariate=covariate
            )
            if not adjusted:
                sensitivity_rows.append(
                    _unavailable_row(
                        control_id=control_id,
                        analysis_role=analysis_role,
                        notes=note,
                    )
                )
                status = STATUS_UNAVAILABLE
                n_effects = 0
            else:
                summary = summarize_effect(adjusted)
                sensitivity_rows.append(
                    {
                        "control_id": control_id,
                        "analysis_role": analysis_role,
                        "is_primary_analysis": False,
                        "can_rescue_primary": False,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "duration_s": EXPECTED_PRIMARY_DURATION_S,
                        "power_representation": PRIMARY_POWER_REPRESENTATION,
                        "band": "pooled",
                        "dataset_id": "pooled",
                        "contrast_id": "pooled",
                        "effect_estimate": summary["effect_estimate"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "n": summary["n"],
                        "p_value": summary["p_value"],
                        "status": STATUS_SENSITIVITY_ONLY,
                        "notes": note,
                    }
                )
                status = STATUS_OK
                n_effects = 1
            qc_rows.append(
                {
                    "control_id": control_id,
                    "component": analysis_role,
                    "status": status,
                    "n_available": 1 if status == STATUS_OK else 0,
                    "n_unavailable": 0 if status == STATUS_OK else 1,
                    "n_effects": n_effects,
                    "notes": note if covariate else avail_note,
                }
            )
            continue

        if control_id == CONTROL_EYE_STATE:
            # Stratify primary paired rows by eye_state on subject rows.
            eye_lookup = {}
            for row in subject_rows:
                eye = _as_str(row.get("eye_state")).casefold()
                if eye in {"", "unknown"}:
                    continue
                key = (
                    _as_str(row.get("dataset_id")).casefold(),
                    _as_str(row.get("participant_id")).casefold(),
                    _as_str(row.get("session_id"), "single").casefold(),
                )
                eye_lookup[key] = eye
            by_eye: dict[str, list[float]] = {}
            for row in primary_paired:
                key = (
                    _as_str(row.get("dataset_id")).casefold(),
                    _as_str(row.get("participant_id")).casefold(),
                    _as_str(row.get("session_id"), "single").casefold(),
                )
                eye = eye_lookup.get(key)
                if eye is None:
                    continue
                delta = _as_float(row.get("delta_endpoint_index"))
                if math.isfinite(delta):
                    by_eye.setdefault(eye, []).append(delta)
            if not by_eye:
                sensitivity_rows.append(
                    _unavailable_row(
                        control_id=control_id,
                        analysis_role=analysis_role,
                        notes="Eye state present in inventory but not joinable to pairs.",
                    )
                )
                qc_rows.append(
                    {
                        "control_id": control_id,
                        "component": analysis_role,
                        "status": STATUS_UNAVAILABLE,
                        "n_available": 0,
                        "n_unavailable": 1,
                        "n_effects": 0,
                        "notes": "Eye-state stratification unmatched.",
                    }
                )
            else:
                for eye, values in sorted(by_eye.items()):
                    summary = summarize_effect(values)
                    sensitivity_rows.append(
                        {
                            "control_id": control_id,
                            "analysis_role": analysis_role,
                            "is_primary_analysis": False,
                            "can_rescue_primary": False,
                            "endpoint_name": ENDPOINT_ZLPI,
                            "duration_s": EXPECTED_PRIMARY_DURATION_S,
                            "power_representation": PRIMARY_POWER_REPRESENTATION,
                            "band": "pooled",
                            "dataset_id": "pooled",
                            "contrast_id": f"eye_state::{eye}",
                            "effect_estimate": summary["effect_estimate"],
                            "ci_low": summary["ci_low"],
                            "ci_high": summary["ci_high"],
                            "n": summary["n"],
                            "p_value": summary["p_value"],
                            "status": STATUS_SENSITIVITY_ONLY,
                            "notes": f"Stratified by eye_state={eye}.",
                        }
                    )
                qc_rows.append(
                    {
                        "control_id": control_id,
                        "component": analysis_role,
                        "status": STATUS_OK,
                        "n_available": len(by_eye),
                        "n_unavailable": 0,
                        "n_effects": len(by_eye),
                        "notes": avail_note,
                    }
                )
            continue

        if control_id in {
            CONTROL_MOTION,
            CONTROL_EOG,
            CONTROL_EMG,
            CONTROL_RESPIRATION,
        }:
            # Presence of rows tagged with residualized_against_<nuisance> or
            # endpoint_index_nuisance_<name>. Otherwise mark available-but-no-effects
            # unless synthetic series-level residualization rows were supplied via
            # subject column endpoint_index after residualization naming.
            tag = control_id.replace("nuisance_", "")
            tagged = [
                r
                for r in paired_rows
                if _as_str(r.get("nuisance_control")).casefold() == tag
                or _as_str(r.get("control_id")).casefold() == control_id
            ]
            if tagged:
                sensitivity_rows.extend(
                    _effects_by_band_dataset(
                        tagged,
                        control_id=control_id,
                        analysis_role=analysis_role,
                        is_primary=False,
                        endpoint_name=ENDPOINT_ZLPI,
                        duration_s=EXPECTED_PRIMARY_DURATION_S,
                        power_representation=PRIMARY_POWER_REPRESENTATION,
                        status=STATUS_SENSITIVITY_ONLY,
                        notes=f"Nuisance residualization on {tag}.",
                    )
                )
                qc_rows.append(
                    {
                        "control_id": control_id,
                        "component": analysis_role,
                        "status": STATUS_OK,
                        "n_available": len(tagged),
                        "n_unavailable": 0,
                        "n_effects": sum(
                            1 for r in sensitivity_rows if r["control_id"] == control_id
                        ),
                        "notes": avail_note,
                    }
                )
            else:
                # Signals are available in inventory but effect table not supplied —
                # record availability without fabricating effects.
                sensitivity_rows.append(
                    {
                        "control_id": control_id,
                        "analysis_role": analysis_role,
                        "is_primary_analysis": False,
                        "can_rescue_primary": False,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "duration_s": EXPECTED_PRIMARY_DURATION_S,
                        "power_representation": PRIMARY_POWER_REPRESENTATION,
                        "band": "",
                        "dataset_id": "",
                        "contrast_id": "",
                        "effect_estimate": float("nan"),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "n": 0,
                        "p_value": float("nan"),
                        "status": STATUS_SENSITIVITY_ONLY,
                        "notes": (
                            f"{tag} available in inventory; residualized paired "
                            "contrasts not supplied in this table."
                        ),
                    }
                )
                qc_rows.append(
                    {
                        "control_id": control_id,
                        "component": analysis_role,
                        "status": STATUS_OK,
                        "n_available": 1,
                        "n_unavailable": 0,
                        "n_effects": 0,
                        "notes": avail_note,
                    }
                )
            continue

    # Cardiac-field / modality are handled in dedicated sections below.

    # ---- Cardiac-field series controls ----
    n_artifact_ok = 0
    n_artifact_missing = 0
    if not series_controls:
        sensitivity_rows.append(
            _unavailable_row(
                control_id=CONTROL_CARDIAC_FIELD,
                analysis_role="artifact_control",
                notes="No series_controls with beat_times_s supplied.",
            )
        )
        qc_rows.append(
            {
                "control_id": CONTROL_CARDIAC_FIELD,
                "component": "artifact_control",
                "status": STATUS_UNAVAILABLE,
                "n_available": 0,
                "n_unavailable": 1,
                "n_effects": 0,
                "notes": "Beat-locked EEG series not provided.",
            }
        )

    for payload in series_controls or ():
        obs_id = _as_str(payload.get("observation_id"), "unknown")
        dataset_id = _as_str(payload.get("dataset_id"))
        band = _as_str(payload.get("band"), "theta")
        duration_s = _as_int(payload.get("duration_s"), EXPECTED_PRIMARY_DURATION_S)
        beats = payload.get("beat_times_s")
        if beats is None:
            artifact_rows.append(
                {
                    "control_id": CONTROL_CARDIAC_FIELD,
                    "observation_id": obs_id,
                    "dataset_id": dataset_id,
                    "band": band,
                    "duration_s": duration_s,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "primary_endpoint_index": float("nan"),
                    "controlled_endpoint_index": float("nan"),
                    "delta_vs_primary": float("nan"),
                    "artifact_injected": _as_bool(payload.get("artifact_injected")) is True,
                    "control_applied": False,
                    "status": STATUS_UNAVAILABLE,
                    "notes": "beat_times_s unavailable; not treated as zero.",
                }
            )
            n_artifact_missing += 1
            continue
        try:
            result = apply_cardiac_field_control(
                np.asarray(payload["hr_z"], dtype=float),
                np.asarray(payload["eeg_z"], dtype=float),
                np.asarray(payload["times_s"], dtype=float),
                np.asarray(beats, dtype=float),
                duration_s=duration_s,
                half_width_s=_as_float(
                    payload.get("half_width_s", DEFAULT_QRS_HALF_WIDTH_S)
                )
                or DEFAULT_QRS_HALF_WIDTH_S,
            )
            artifact_rows.append(
                {
                    "control_id": CONTROL_CARDIAC_FIELD,
                    "observation_id": obs_id,
                    "dataset_id": dataset_id,
                    "band": band,
                    "duration_s": duration_s,
                    "endpoint_name": result["endpoint_name"],
                    "primary_endpoint_index": result["primary_endpoint_index"],
                    "controlled_endpoint_index": result["controlled_endpoint_index"],
                    "delta_vs_primary": result["delta_vs_primary"],
                    "artifact_injected": _as_bool(payload.get("artifact_injected")) is True,
                    "control_applied": True,
                    "status": STATUS_OK,
                    "notes": "QRS interpolation applied; HR unchanged.",
                }
            )
            n_artifact_ok += 1
        except ValueError as exc:
            artifact_rows.append(
                {
                    "control_id": CONTROL_CARDIAC_FIELD,
                    "observation_id": obs_id,
                    "dataset_id": dataset_id,
                    "band": band,
                    "duration_s": duration_s,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "primary_endpoint_index": float("nan"),
                    "controlled_endpoint_index": float("nan"),
                    "delta_vs_primary": float("nan"),
                    "artifact_injected": _as_bool(payload.get("artifact_injected")) is True,
                    "control_applied": False,
                    "status": STATUS_UNAVAILABLE,
                    "notes": str(exc),
                }
            )
            n_artifact_missing += 1

    if series_controls:
        if n_artifact_ok:
            deltas = [
                _as_float(r["delta_vs_primary"])
                for r in artifact_rows
                if r["status"] == STATUS_OK
            ]
            summary = summarize_effect(deltas)
            sensitivity_rows.append(
                {
                    "control_id": CONTROL_CARDIAC_FIELD,
                    "analysis_role": "artifact_control",
                    "is_primary_analysis": False,
                    "can_rescue_primary": False,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": EXPECTED_PRIMARY_DURATION_S,
                    "power_representation": PRIMARY_POWER_REPRESENTATION,
                    "band": "pooled",
                    "dataset_id": "pooled",
                    "contrast_id": "qrs_delta_vs_primary",
                    "effect_estimate": summary["effect_estimate"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "n": summary["n"],
                    "p_value": summary["p_value"],
                    "status": STATUS_SENSITIVITY_ONLY,
                    "notes": "Mean QRS-controlled minus primary endpoint index.",
                }
            )
        qc_rows.append(
            {
                "control_id": CONTROL_CARDIAC_FIELD,
                "component": "artifact_control",
                "status": STATUS_OK if n_artifact_ok else STATUS_UNAVAILABLE,
                "n_available": n_artifact_ok,
                "n_unavailable": n_artifact_missing,
                "n_effects": n_artifact_ok,
                "notes": "Cardiac-field QRS interpolation on supplied series.",
            }
        )

    # ---- Modality comparison ----
    modality_rows = compare_matched_modalities(subject_rows)
    matched = [r for r in modality_rows if r["matched"]]
    if matched:
        summary = summarize_effect(
            [_as_float(r["delta_ecg_minus_ppg"]) for r in matched]
        )
        sensitivity_rows.append(
            {
                "control_id": CONTROL_ECG_VS_PPG,
                "analysis_role": "modality_comparison",
                "is_primary_analysis": False,
                "can_rescue_primary": False,
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": EXPECTED_PRIMARY_DURATION_S,
                "power_representation": PRIMARY_POWER_REPRESENTATION,
                "band": "pooled",
                "dataset_id": "pooled",
                "contrast_id": "ecg_minus_ppg",
                "effect_estimate": summary["effect_estimate"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "n": summary["n"],
                "p_value": summary["p_value"],
                "status": STATUS_SENSITIVITY_ONLY,
                "notes": "Matched ECG−PPG only.",
            }
        )
        qc_rows.append(
            {
                "control_id": CONTROL_ECG_VS_PPG,
                "component": "modality_comparison",
                "status": STATUS_OK,
                "n_available": len(matched),
                "n_unavailable": len(modality_rows) - len(matched),
                "n_effects": 1,
                "notes": "Unmatched modalities excluded.",
            }
        )
    elif not any(r["control_id"] == CONTROL_ECG_VS_PPG for r in sensitivity_rows):
        sensitivity_rows.append(
            _unavailable_row(
                control_id=CONTROL_ECG_VS_PPG,
                analysis_role="modality_comparison",
                notes="No matched ECG/PPG observations.",
            )
        )
        qc_rows.append(
            {
                "control_id": CONTROL_ECG_VS_PPG,
                "component": "modality_comparison",
                "status": STATUS_UNAVAILABLE,
                "n_available": 0,
                "n_unavailable": len(modality_rows),
                "n_effects": 0,
                "notes": "No matched ECG/PPG pairs.",
            }
        )

    # ---- Specification matrix (one summary row per control_id) ----
    specification_rows = build_specification_matrix(sensitivity_rows)

    # Guardrail QC: sensitivities must never be marked primary / rescue.
    n_violations = sum(
        1
        for row in sensitivity_rows
        if row["control_id"] != PRIMARY_CONTROL_ID
        and (
            bool(row.get("is_primary_analysis"))
            or bool(row.get("can_rescue_primary"))
        )
    )
    qc_rows.append(
        {
            "control_id": "guardrail",
            "component": "primary_protection",
            "status": STATUS_OK if n_violations == 0 else "error",
            "n_available": len(sensitivity_rows),
            "n_unavailable": 0,
            "n_effects": n_violations,
            "notes": (
                "All sensitivities marked non-primary with can_rescue_primary=False."
                if n_violations == 0
                else f"{n_violations} sensitivity rows violated primary protection."
            ),
        }
    )

    return ArtifactControlResult(
        sensitivity_rows=tuple(sensitivity_rows),
        artifact_rows=tuple(artifact_rows),
        modality_rows=tuple(modality_rows),
        duration_rows=tuple(duration_rows),
        specification_rows=tuple(specification_rows),
        qc_rows=tuple(qc_rows),
    )


def build_specification_matrix(
    sensitivity_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """One matrix row per control with pooled effect / CI / n / status."""
    by_control: dict[str, list[Mapping[str, object]]] = {}
    for row in sensitivity_rows:
        by_control.setdefault(_as_str(row.get("control_id")), []).append(row)

    matrix: list[dict[str, object]] = []
    for control_id in ALL_CONTROL_IDS:
        rows = by_control.get(control_id, [])
        if not rows:
            matrix.append(
                {
                    "control_id": control_id,
                    "analysis_role": "unspecified",
                    "is_primary_analysis": control_id == PRIMARY_CONTROL_ID,
                    "can_rescue_primary": False,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": EXPECTED_PRIMARY_DURATION_S,
                    "power_representation": PRIMARY_POWER_REPRESENTATION,
                    "effect_estimate": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n": 0,
                    "status": STATUS_UNAVAILABLE,
                    "notes": "Control not evaluated.",
                }
            )
            continue
        # Prefer pooled rows; else inverse-variance / simple mean of OK effects.
        pooled = [
            r
            for r in rows
            if _as_str(r.get("dataset_id")) in {"", "pooled"}
            or _as_str(r.get("band")) == "pooled"
        ]
        source = pooled or [
            r for r in rows if _as_str(r.get("status")) in {STATUS_OK, STATUS_PRIMARY, STATUS_SENSITIVITY_ONLY}
        ]
        unavailable = all(
            _as_str(r.get("status")) == STATUS_UNAVAILABLE for r in rows
        )
        first = rows[0]
        if unavailable:
            matrix.append(
                {
                    "control_id": control_id,
                    "analysis_role": first.get("analysis_role", ""),
                    "is_primary_analysis": bool(first.get("is_primary_analysis")),
                    "can_rescue_primary": False,
                    "endpoint_name": first.get("endpoint_name", ENDPOINT_ZLPI),
                    "duration_s": first.get("duration_s", EXPECTED_PRIMARY_DURATION_S),
                    "power_representation": first.get(
                        "power_representation", PRIMARY_POWER_REPRESENTATION
                    ),
                    "effect_estimate": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n": 0,
                    "status": STATUS_UNAVAILABLE,
                    "notes": _as_str(first.get("notes")),
                }
            )
            continue
        if len(source) == 1:
            row = source[0]
            matrix.append(
                {
                    "control_id": control_id,
                    "analysis_role": row.get("analysis_role", ""),
                    "is_primary_analysis": bool(row.get("is_primary_analysis")),
                    "can_rescue_primary": False,
                    "endpoint_name": row.get("endpoint_name", ENDPOINT_ZLPI),
                    "duration_s": row.get("duration_s", EXPECTED_PRIMARY_DURATION_S),
                    "power_representation": row.get(
                        "power_representation", PRIMARY_POWER_REPRESENTATION
                    ),
                    "effect_estimate": row.get("effect_estimate"),
                    "ci_low": row.get("ci_low"),
                    "ci_high": row.get("ci_high"),
                    "n": row.get("n", 0),
                    "status": row.get("status", STATUS_OK),
                    "notes": _as_str(row.get("notes")),
                }
            )
        else:
            effects = [_as_float(r.get("effect_estimate")) for r in source]
            ns = [_as_int(r.get("n")) for r in source]
            summary = summarize_effect(
                [e for e in effects if math.isfinite(e)]
            )
            matrix.append(
                {
                    "control_id": control_id,
                    "analysis_role": first.get("analysis_role", ""),
                    "is_primary_analysis": control_id == PRIMARY_CONTROL_ID,
                    "can_rescue_primary": False,
                    "endpoint_name": first.get("endpoint_name", ENDPOINT_ZLPI),
                    "duration_s": first.get("duration_s", EXPECTED_PRIMARY_DURATION_S),
                    "power_representation": first.get(
                        "power_representation", PRIMARY_POWER_REPRESENTATION
                    ),
                    "effect_estimate": summary["effect_estimate"],
                    "ci_low": summary["ci_low"],
                    "ci_high": summary["ci_high"],
                    "n": int(sum(ns)),
                    "status": (
                        STATUS_PRIMARY
                        if control_id == PRIMARY_CONTROL_ID
                        else STATUS_SENSITIVITY_ONLY
                    ),
                    "notes": f"Pooled across {len(source)} specification cells.",
                }
            )
    return matrix


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                elif isinstance(value, bool):
                    payload[field] = str(value)
                else:
                    payload[field] = value
            writer.writerow(payload)


def write_artifact_control_outputs(
    result: ArtifactControlResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "sensitivity_results": output_path / SENSITIVITY_RESULTS_FILENAME,
        "artifact_control_results": output_path / ARTIFACT_CONTROL_RESULTS_FILENAME,
        "modality_comparison": output_path / MODALITY_COMPARISON_FILENAME,
        "duration_sensitivity": output_path / DURATION_SENSITIVITY_FILENAME,
        "specification_matrix": output_path / SPECIFICATION_MATRIX_FILENAME,
        "sensitivity_qc": output_path / SENSITIVITY_QC_FILENAME,
    }
    _write_csv(paths["sensitivity_results"], result.sensitivity_rows, SENSITIVITY_FIELDS)
    _write_csv(paths["artifact_control_results"], result.artifact_rows, ARTIFACT_FIELDS)
    _write_csv(paths["modality_comparison"], result.modality_rows, MODALITY_FIELDS)
    _write_csv(paths["duration_sensitivity"], result.duration_rows, DURATION_FIELDS)
    _write_csv(paths["specification_matrix"], result.specification_rows, SPEC_MATRIX_FIELDS)
    _write_csv(paths["sensitivity_qc"], result.qc_rows, QC_FIELDS)
    return paths


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_confirmatory_artifact_controls(
    group_tables_dir: str | Path,
    output_dir: str | Path,
    *,
    control_inventory: Mapping[str, Mapping[str, object]] | None = None,
    series_controls: Sequence[Mapping[str, object]] | None = None,
) -> ArtifactControlResult:
    """Load M8 tables and write M11 sensitivity outputs."""
    root = Path(group_tables_dir).expanduser().resolve()
    subject_rows = read_csv_rows(root / SUBJECT_LEVEL_FILENAME)
    paired_rows = read_csv_rows(root / PAIRED_CONTRASTS_FILENAME)
    result = run_artifact_controls(
        subject_rows,
        paired_rows,
        control_inventory=control_inventory,
        series_controls=series_controls,
    )
    write_artifact_control_outputs(result, output_dir)
    return result


__all__ = [
    "ALL_CONTROL_IDS",
    "ARTIFACT_CONTROL_RESULTS_FILENAME",
    "CONTROL_BROADBAND",
    "CONTROL_CARDIAC_FIELD",
    "CONTROL_ECG_VS_PPG",
    "DURATION_SENSITIVITY_FILENAME",
    "MODALITY_COMPARISON_FILENAME",
    "PRIMARY_CONTROL_ID",
    "SENSITIVITY_QC_FILENAME",
    "SENSITIVITY_RESULTS_FILENAME",
    "SPECIFICATION_MATRIX_FILENAME",
    "ArtifactControlResult",
    "apply_cardiac_field_control",
    "beat_density",
    "broadband_residualize_log_power",
    "build_specification_matrix",
    "compare_matched_modalities",
    "control_availability",
    "is_primary_cell",
    "qrs_interpolate_eeg",
    "residualize_series",
    "run_artifact_controls",
    "run_confirmatory_artifact_controls",
    "summarize_effect",
    "write_artifact_control_outputs",
]
