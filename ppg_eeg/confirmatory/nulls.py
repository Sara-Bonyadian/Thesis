"""Temporal and cross-subject null analyses for confirmatory endpoints (M9).

Null types
----------
- ``circular_shift``: integer EEG circular shifts ≥60 s from identity
- ``phase_randomization``: preserve EEG amplitude spectrum + conjugate symmetry
- ``block_shuffle``: permute non-overlapping 30-s EEG blocks (reject identity)
- ``cross_subject_mismatch``: seeded derangements within
  dataset × state × modality × duration
- ``ar1_innovations``: AR(1) residual series for HR and EEG

Each duration keeps its M6 endpoint contract (ZLPI / MWPI / SWPI). Null
distributions are never pooled across endpoint types.

Surrogate counts
----------------
- Production / confirmatory default: ``DEFAULT_N_SURROGATES`` (500)
- Smoke tests: ``SMOKE_N_SURROGATES`` (20), typically via dataset YAML
  ``n_surrogates: 20``

Statistical contract note (500 vs prior 1000)
---------------------------------------------
``DEFAULT_N_SURROGATES`` was reduced from 1000 to 500 as an explicit
Monte Carlo contract change (not a bit-identical optimization). With the
add-one empirical p-value, the finest attainable p is ``1/(n+1)``:
``1/501 ≈ 0.001996`` at n=500 (was ``1/1001 ≈ 0.000999`` at n=1000).
Monte Carlo SE for a given p scales roughly as ``sqrt(p(1-p)/n)``, so
uncertainty increases by about ``sqrt(2) ≈ 1.41×`` relative to n=1000.
Results at 500 surrogates are not identical to results at 1000.

Reproducibility
---------------
Each subject×null-type row records ``rng_seed_u64``: the uint64 seed passed
to ``numpy.random.default_rng`` for that unit. Seeds are derived from SHA-256
of ``observation_id``, analysis key, and null type (not Python ``hash()``).
The seed-generation method is unchanged when ``n_surrogates`` changes; only
the number of draws from that RNG stream changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ppg_eeg.temporal_coupling.cross_correlation import build_lag_grid

from .correlation import (
    DEFAULT_MIN_OVERLAP,
    FS_HZ,
    IDENTITY_FIELDS,
    POWER_REPRESENTATIONS,
    _eeg_column,
    _group_aligned_rows,
    compute_correlation_curve_common_support,
    discover_aligned_duration_tables,
    read_aligned_features_csv,
)
from .duration_contracts import EXPECTED_DURATIONS_S, contract_for_duration
from .endpoints import evaluate_endpoint_curve
from .protocol_audit import condition_semantics_for
from .reason_codes import (
    CONFIGURATION_VALIDATION_FAILED,
    STRUCTURED_NC_FIELDS,
    UNDEFINED_NULL_VARIANCE,
    attach_structured_reason,
    map_exclusion_to_reason_code,
)

NULL_SUBJECT_RESULTS_FILENAME = "null_subject_results.csv"
NULL_SUMMARY_FILENAME = "null_summary.csv"
NULL_QC_FILENAME = "null_qc.csv"
NULL_SURROGATE_VALUES_FILENAME = "null_surrogate_values.csv"

NULL_TYPE_CIRCULAR_SHIFT = "circular_shift"
NULL_TYPE_PHASE_RANDOMIZATION = "phase_randomization"
NULL_TYPE_BLOCK_SHUFFLE = "block_shuffle"
NULL_TYPE_CROSS_SUBJECT_MISMATCH = "cross_subject_mismatch"
NULL_TYPE_AR1_INNOVATIONS = "ar1_innovations"

NULL_TYPES: tuple[str, ...] = (
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_PHASE_RANDOMIZATION,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CROSS_SUBJECT_MISMATCH,
    NULL_TYPE_AR1_INNOVATIONS,
)

WITHIN_SERIES_NULL_TYPES: tuple[str, ...] = (
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_PHASE_RANDOMIZATION,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_AR1_INNOVATIONS,
)

SMOKE_N_SURROGATES = 20
DEFAULT_N_SURROGATES = 500
MIN_CIRCULAR_SHIFT_S = 60
BLOCK_LENGTH_S = 30

CHECKPOINT_SCHEMA_VERSION = "c4_checkpoint_v2"
CHECKPOINT_DIRNAME = "_unit_checkpoints"
RUN_MANIFEST_FILENAME = "run_manifest.json"
COMPLETE_MARKER_FILENAME = "C4_COMPLETE.json"

QC_FIELDS = tuple(
    dict.fromkeys(
        (
            "dataset_id",
            "subject_id",
            "condition",
            "observation_id",
            "modality",
            "duration_s",
            "endpoint_name",
            "band",
            "power_representation",
            "null_type",
            "null_status",
            "n_surrogates_requested",
            "n_surrogates_finite",
            "notes",
            *STRUCTURED_NC_FIELDS,
        )
    )
)

SUBJECT_RESULT_FIELDS = tuple(
    dict.fromkeys(
        (
            "dataset_id",
            "subject_id",
            "task",
            "condition",
            "observation_id",
            "modality",
            "duration_s",
            "duration_role",
            "endpoint_name",
            "endpoint_alias",
            "is_standard_zlpi",
            "band",
            "power_representation",
            "is_primary_representation",
            "pair",
            "null_type",
            "n_surrogates_requested",
            "n_surrogates_finite",
            "observed_endpoint_index",
            "observed_eligible",
            "null_mean",
            "null_std",
            "null_median",
            "empirical_p",
            "effect_size_surrogate_z",
            "rng_seed_u64",
            "analysis_key",
            *STRUCTURED_NC_FIELDS,
        )
    )
)

SUMMARY_FIELDS = (
    "dataset_id",
    "condition",
    "modality",
    "duration_s",
    "endpoint_name",
    "is_standard_zlpi",
    "band",
    "power_representation",
    "null_type",
    "n_units",
    "n_eligible_observed",
    "median_empirical_p",
    "mean_empirical_p",
    "mean_effect_size_surrogate_z",
    "median_effect_size_surrogate_z",
    "median_observed_endpoint_index",
)

SURROGATE_VALUE_FIELDS = (
    "dataset_id",
    "dataset_role",
    "biological_participant_id",
    "analysis_unit_id",
    "subject_id",
    "session_id",
    "observation_id",
    "condition",
    "period",
    "state",
    "task",
    "band",
    "duration",
    "representation",
    "endpoint",
    "null_type",
    "surrogate_index",
    "surrogate_endpoint_index",
    "observed_endpoint_index",
    "standardized_surrogate_value",
    "standardized_observed_value",
    "null_mean",
    "null_median",
    "null_std",
    "empirical_p",
    "rng_seed_u64",
    "n_surrogates",
    "eligibility_status",
    "qc_status",
)


@dataclass(frozen=True)
class SeriesUnit:
    """One HR×EEG analysis unit for null generation."""

    dataset_id: str
    subject_id: str
    task: str
    condition: str
    observation_id: str
    modality: str
    duration_s: int
    duration_role: str
    band: str
    power_representation: str
    is_primary_representation: bool
    pair: str
    hr_z: np.ndarray
    eeg_z: np.ndarray


@dataclass(frozen=True)
class NullBatteryResult:
    subject_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    qc_rows: tuple[dict[str, object], ...]
    surrogate_rows: tuple[dict[str, object], ...] = ()


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


def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _median(values: Sequence[float]) -> float:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return float("nan")
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float(0.5 * (finite[mid - 1] + finite[mid]))


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return float("nan")
    return float(np.std(np.asarray(finite, dtype=float), ddof=1))


def analysis_key(
    *,
    duration_s: int,
    endpoint_name: str,
    band: str,
    power_representation: str,
    modality: str = "default",
) -> str:
    """Stable analysis identifier used in deterministic seeding."""
    return "|".join(
        [
            str(int(duration_s)),
            _as_str(endpoint_name).casefold(),
            _as_str(band).casefold(),
            _as_str(power_representation).casefold(),
            _as_str(modality, "default").casefold(),
        ]
    )


def deterministic_seed(
    observation_id: str,
    analysis_key_value: str,
    null_type: str,
) -> int:
    """Derive the ``rng_seed_u64`` value for one null unit.

    Returns a uint64 suitable for ``numpy.random.default_rng``. The digest is
    SHA-256 of observation ID, analysis key, and null type — not Python's
    process-randomized ``hash()``.
    """
    payload = (
        f"{_as_str(observation_id)}\0{_as_str(analysis_key_value)}\0"
        f"{_as_str(null_type).casefold()}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def empirical_p_value(
    observed: float,
    null_values: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """Add-one smoothed empirical p-value against surrogate statistics."""
    if not math.isfinite(observed):
        return float("nan")
    finite = [float(v) for v in null_values if math.isfinite(float(v))]
    if not finite:
        return float("nan")
    alt = alternative.casefold()
    if alt == "greater":
        exceed = sum(1 for value in finite if value >= observed)
    elif alt == "less":
        exceed = sum(1 for value in finite if value <= observed)
    elif alt == "two_sided":
        exceed = sum(1 for value in finite if abs(value) >= abs(observed))
    else:
        raise ValueError(
            "alternative must be 'greater', 'less', or 'two_sided', "
            f"got {alternative!r}."
        )
    return float((exceed + 1) / (len(finite) + 1))


def surrogate_effect_size(
    observed: float,
    null_values: Sequence[float],
) -> float:
    """Surrogate-normalized effect size: (obs − mean_null) / sd_null."""
    if not math.isfinite(observed):
        return float("nan")
    center = _mean(null_values)
    scale = _std(null_values)
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        return float("nan")
    return float((observed - center) / scale)


def valid_circular_shifts(
    n_samples: int,
    *,
    min_abs_shift_s: int = MIN_CIRCULAR_SHIFT_S,
    fs_hz: float = FS_HZ,
) -> np.ndarray:
    """Integer shifts at least ``min_abs_shift_s`` from identity (circular)."""
    if n_samples <= 1:
        return np.asarray([], dtype=int)
    min_shift = int(round(float(min_abs_shift_s) * float(fs_hz)))
    if min_shift < 1:
        min_shift = 1
    if n_samples <= 2 * min_shift:
        return np.asarray([], dtype=int)
    # Shifts k with min(k, n-k) >= min_shift, excluding 0.
    shifts = [
        int(k)
        for k in range(1, n_samples)
        if min(int(k), int(n_samples - k)) >= min_shift
    ]
    return np.asarray(shifts, dtype=int)


def circular_shift_series(values: np.ndarray, shift: int) -> np.ndarray:
    """Circularly shift a series (NaNs travel with the values)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    return np.roll(arr, int(shift))


def phase_randomize_series(
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize phases while preserving amplitude spectrum and real conjugate symmetry."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr.copy()
    work = arr.copy()
    fill = float(np.nanmean(work)) if np.any(finite) else 0.0
    if not math.isfinite(fill):
        fill = 0.0
    work[~finite] = fill

    spectrum = np.fft.rfft(work)
    amplitudes = np.abs(spectrum)
    n = int(work.size)
    n_bins = int(amplitudes.size)
    phases = np.zeros(n_bins, dtype=float)
    if n_bins > 1:
        # DC phase stays 0 (real). Randomize positive frequencies.
        # For even n, Nyquist bin must remain real (phase 0 or π).
        if n % 2 == 0 and n_bins >= 2:
            mid = n_bins - 1
            if mid > 1:
                phases[1:mid] = rng.uniform(0.0, 2.0 * np.pi, size=mid - 1)
            phases[mid] = float(rng.choice([0.0, np.pi]))
        else:
            phases[1:] = rng.uniform(0.0, 2.0 * np.pi, size=n_bins - 1)
    randomized = amplitudes * np.exp(1j * phases)
    out = np.fft.irfft(randomized, n=n)
    out = np.asarray(out, dtype=float)
    out[~finite] = np.nan
    return out


def amplitude_spectrum(values: np.ndarray) -> np.ndarray:
    """Amplitude spectrum used for phase-randomization QC."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    work = arr.copy()
    fill = float(np.nanmean(work)) if np.any(finite) else 0.0
    if not math.isfinite(fill):
        fill = 0.0
    work[~finite] = fill
    return np.abs(np.fft.rfft(work))


def block_shuffle_series(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    block_length_s: int = BLOCK_LENGTH_S,
    fs_hz: float = FS_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute non-overlapping blocks; keep incomplete terminal block unmoved.

    Returns ``(shuffled, block_order)``. Rejects the identity permutation when
    more than one complete block exists.
    """
    arr = np.asarray(values, dtype=float)
    block_len = int(round(float(block_length_s) * float(fs_hz)))
    if block_len < 1:
        raise ValueError("block_length_s must yield block_len >= 1.")
    n = int(arr.size)
    n_complete = (n // block_len) * block_len
    if n_complete < block_len:
        return arr.copy(), np.asarray([], dtype=int)

    n_blocks = n_complete // block_len
    blocks = arr[:n_complete].reshape(n_blocks, block_len)
    tail = arr[n_complete:]
    order = np.arange(n_blocks, dtype=int)
    if n_blocks == 1:
        shuffled_head = blocks.reshape(-1)
    else:
        for _ in range(10_000):
            candidate = rng.permutation(n_blocks)
            if not np.array_equal(candidate, order):
                order = candidate
                break
        else:
            # Deterministic non-identity fallback.
            order = np.roll(np.arange(n_blocks, dtype=int), 1)
        shuffled_head = blocks[order].reshape(-1)
    out = np.concatenate([shuffled_head, tail]) if tail.size else shuffled_head
    return np.asarray(out, dtype=float), order


def complete_blocks(
    values: np.ndarray,
    *,
    block_length_s: int = BLOCK_LENGTH_S,
    fs_hz: float = FS_HZ,
) -> np.ndarray:
    """Return complete non-overlapping blocks as a ``(n_blocks, block_len)`` array."""
    arr = np.asarray(values, dtype=float)
    block_len = int(round(float(block_length_s) * float(fs_hz)))
    if block_len < 1:
        raise ValueError("block_length_s must yield block_len >= 1.")
    n_complete = (int(arr.size) // block_len) * block_len
    if n_complete < block_len:
        return np.zeros((0, block_len), dtype=float)
    return arr[:n_complete].reshape(-1, block_len)


def fit_ar1_phi(values: np.ndarray) -> tuple[float, float]:
    """Return ``(phi, intercept)`` for an AR(1) fit on consecutive finite pairs."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0, 0.0
    y = arr[1:]
    x = arr[:-1]
    mask = np.isfinite(y) & np.isfinite(x)
    if int(mask.sum()) < 2:
        return 0.0, float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
    x_m = x[mask]
    y_m = y[mask]
    x_mean = float(np.mean(x_m))
    y_mean = float(np.mean(y_m))
    var_x = float(np.sum((x_m - x_mean) ** 2))
    if var_x <= 0:
        return 0.0, y_mean
    phi = float(np.sum((x_m - x_mean) * (y_m - y_mean)) / var_x)
    # Keep stationary-ish; do not clip hard-zero noise series oddly.
    phi = float(np.clip(phi, -0.999, 0.999))
    intercept = float(y_mean - phi * x_mean)
    return phi, intercept


def ar1_innovations(values: np.ndarray) -> np.ndarray:
    """Return one-step AR(1) innovations (NaN at the first sample)."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.size < 2:
        return out
    phi, intercept = fit_ar1_phi(arr)
    for index in range(1, int(arr.size)):
        if np.isfinite(arr[index]) and np.isfinite(arr[index - 1]):
            out[index] = float(arr[index] - intercept - phi * arr[index - 1])
    return out


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation over consecutive finite pairs."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    x = arr[:-1]
    y = arr[1:]
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x_m = x[mask] - float(np.mean(x[mask]))
    y_m = y[mask] - float(np.mean(y[mask]))
    denom = float(np.sqrt(np.sum(x_m * x_m) * np.sum(y_m * y_m)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x_m * y_m) / denom)


def seeded_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a derangement of ``0..n-1`` (no fixed points)."""
    if n < 2:
        raise ValueError("Derangement requires n >= 2.")
    index = np.arange(n, dtype=int)
    for _ in range(10_000):
        perm = rng.permutation(n)
        if not np.any(perm == index):
            return perm
    # Cycle shift is always a derangement for n >= 2.
    return np.roll(index, 1)


def compute_endpoint_index_from_series(
    hr_z: np.ndarray,
    eeg_z: np.ndarray,
    *,
    duration_s: int,
    identity: Mapping[str, str] | None = None,
    band: str = "theta",
    power_representation: str = "absolute_log10",
    duration_role: str = "",
    is_primary_representation: bool = True,
    pair: str = "",
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> dict[str, object]:
    """Recompute the duration-contracted M6 endpoint from HR/EEG series."""
    contract = contract_for_duration(duration_s)
    lag_grid = build_lag_grid(
        lag_max_s=float(contract.lag_max_s),
        lag_step_s=float(contract.lag_step_s),
    )
    curve, _anchors = compute_correlation_curve_common_support(
        np.asarray(hr_z, dtype=float),
        np.asarray(eeg_z, dtype=float),
        lag_grid_s=lag_grid,
        lag_max_s=float(contract.lag_max_s),
        fs_hz=FS_HZ,
        min_overlap=min_overlap,
    )
    base_identity = {
        field: "" for field in IDENTITY_FIELDS
    }
    if identity is not None:
        base_identity.update({k: _as_str(v) for k, v in identity.items()})
    rows = [
        {
            **base_identity,
            "duration_s": int(duration_s),
            "duration_role": duration_role,
            "band": band,
            "power_representation": power_representation,
            "is_primary_representation": bool(is_primary_representation),
            "pair": pair or f"hr_x_{band}_{power_representation}",
            "lag_s": float(point.lag_s),
            "r": float(point.r),
            "n_overlap": int(point.n_overlap),
        }
        for point in curve
    ]
    metrics, _qc = evaluate_endpoint_curve(rows, duration_s=duration_s)
    return metrics


def _surrogate_eeg(
    null_type: str,
    eeg_z: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if null_type == NULL_TYPE_CIRCULAR_SHIFT:
        shifts = valid_circular_shifts(int(eeg_z.size))
        if shifts.size == 0:
            return np.full(eeg_z.shape, np.nan, dtype=float)
        shift = int(rng.choice(shifts))
        return circular_shift_series(eeg_z, shift)
    if null_type == NULL_TYPE_PHASE_RANDOMIZATION:
        return phase_randomize_series(eeg_z, rng)
    if null_type == NULL_TYPE_BLOCK_SHUFFLE:
        shuffled, _order = block_shuffle_series(eeg_z, rng)
        return shuffled
    raise ValueError(f"Unsupported within-EEG null type: {null_type!r}")


def _permute_finite(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute finite samples (NaN mask preserved)."""
    out = np.asarray(values, dtype=float).copy()
    idx = np.flatnonzero(np.isfinite(out))
    if idx.size < 2:
        return out
    out[idx] = out[idx][rng.permutation(int(idx.size))]
    return out


def _null_statistics_for_unit(
    unit: SeriesUnit,
    *,
    null_type: str,
    n_surrogates: int,
    partner_eeg_by_surrogate: Sequence[np.ndarray] | None = None,
    observed_stat: float | None = None,
    observed_eligible: bool | None = None,
) -> tuple[dict[str, object], dict[str, object], list[float]]:
    contract = contract_for_duration(unit.duration_s)
    key = analysis_key(
        duration_s=unit.duration_s,
        endpoint_name=contract.endpoint_name,
        band=unit.band,
        power_representation=unit.power_representation,
        modality=unit.modality,
    )
    seed = deterministic_seed(unit.observation_id, key, null_type)
    rng = np.random.default_rng(seed)

    identity = {
        "dataset_id": unit.dataset_id,
        "subject_id": unit.subject_id,
        "task": unit.task,
        "condition": unit.condition,
        "observation_id": unit.observation_id,
    }
    if observed_stat is None or observed_eligible is None:
        observed = compute_endpoint_index_from_series(
            unit.hr_z,
            unit.eeg_z,
            duration_s=unit.duration_s,
            identity=identity,
            band=unit.band,
            power_representation=unit.power_representation,
            duration_role=unit.duration_role,
            is_primary_representation=unit.is_primary_representation,
            pair=unit.pair,
        )
        observed_stat = _as_float(observed.get("endpoint_index"))
        observed_eligible = bool(observed.get("eligible"))
    else:
        observed_stat = float(observed_stat)
        observed_eligible = bool(observed_eligible)

    null_vals: list[float] = []
    notes = ""
    status = "ok"

    if null_type == NULL_TYPE_CIRCULAR_SHIFT:
        if valid_circular_shifts(int(unit.eeg_z.size)).size == 0:
            status = "insufficient_length"
            notes = (
                f"Need length > {2 * MIN_CIRCULAR_SHIFT_S} for "
                f"±{MIN_CIRCULAR_SHIFT_S}s circular shifts."
            )
    elif null_type == NULL_TYPE_BLOCK_SHUFFLE:
        blocks = complete_blocks(unit.eeg_z)
        if blocks.shape[0] < 2:
            status = "insufficient_blocks"
            notes = f"Need ≥2 complete {BLOCK_LENGTH_S}s blocks."
    elif null_type == NULL_TYPE_CROSS_SUBJECT_MISMATCH:
        if partner_eeg_by_surrogate is None:
            status = "missing_mismatch_pool"
            notes = "Cross-subject pool unavailable."
        elif len(partner_eeg_by_surrogate) == 0:
            status = "derangement_impossible"
            notes = "Need ≥2 observations in dataset×state×modality×duration pool."
        elif len(partner_eeg_by_surrogate) != int(n_surrogates):
            status = "mismatch_surrogate_count"
            notes = "Partner EEG list length must equal n_surrogates."

    if status == "ok":
        hr_inn: np.ndarray | None = None
        eeg_inn: np.ndarray | None = None
        if null_type == NULL_TYPE_AR1_INNOVATIONS:
            hr_inn = ar1_innovations(unit.hr_z)
            eeg_inn = ar1_innovations(unit.eeg_z)

        for surrogate_i in range(int(n_surrogates)):
            if null_type == NULL_TYPE_AR1_INNOVATIONS:
                assert hr_inn is not None and eeg_inn is not None
                # Whitened series; destroy residual contemporaneous coupling by
                # independently permuting EEG innovations each surrogate.
                hr_surr = hr_inn
                eeg_surr = _permute_finite(eeg_inn, rng)
            elif null_type == NULL_TYPE_CROSS_SUBJECT_MISMATCH:
                assert partner_eeg_by_surrogate is not None
                hr_surr = unit.hr_z
                eeg_surr = np.asarray(
                    partner_eeg_by_surrogate[surrogate_i], dtype=float
                )
            else:
                hr_surr = unit.hr_z
                eeg_surr = _surrogate_eeg(null_type, unit.eeg_z, rng)

            metrics = compute_endpoint_index_from_series(
                hr_surr,
                eeg_surr,
                duration_s=unit.duration_s,
                identity=identity,
                band=unit.band,
                power_representation=unit.power_representation,
                duration_role=unit.duration_role,
                is_primary_representation=unit.is_primary_representation,
                pair=unit.pair,
            )
            null_vals.append(_as_float(metrics.get("endpoint_index")))

    n_finite = sum(1 for value in null_vals if math.isfinite(value))
    if status == "ok" and n_finite == 0:
        status = "no_finite_surrogates"
        notes = "All surrogate endpoint statistics were non-finite."

    p_value = empirical_p_value(observed_stat, null_vals, alternative="greater")
    effect = surrogate_effect_size(observed_stat, null_vals)
    null_std = _std(null_vals)
    reason_status = status
    reason_code = ""
    reason_text = notes
    if status == "ok" and n_finite > 0 and (not math.isfinite(null_std) or null_std <= 0):
        reason_status = UNDEFINED_NULL_VARIANCE
        reason_code = UNDEFINED_NULL_VARIANCE
        reason_text = "Null surrogate variance is undefined or zero; standardized effect not computable."
    elif status != "ok":
        reason_code = map_exclusion_to_reason_code(status)

    subject_row = attach_structured_reason(
        {
            **identity,
            "modality": unit.modality,
            "duration_s": int(unit.duration_s),
            "duration_role": unit.duration_role,
            "endpoint_name": contract.endpoint_name,
            "endpoint_alias": contract.endpoint_alias,
            "is_standard_zlpi": bool(contract.is_standard_zlpi),
            "band": unit.band,
            "power_representation": unit.power_representation,
            "is_primary_representation": bool(unit.is_primary_representation),
            "pair": unit.pair,
            "null_type": null_type,
            "n_surrogates_requested": int(n_surrogates),
            "n_surrogates_finite": int(n_finite),
            "observed_endpoint_index": observed_stat,
            "observed_eligible": observed_eligible,
            "null_mean": _mean(null_vals),
            "null_std": null_std,
            "null_median": _median(null_vals),
            "empirical_p": p_value,
            "effect_size_surrogate_z": effect,
            "rng_seed_u64": int(seed),
            "analysis_key": key,
            "notes": reason_text,
        },
        stage="C4",
        status="computed" if reason_status == "ok" else "not_computable",
        reason_code=reason_code,
        reason=reason_text,
        required_evidence=(
            f"n_surrogates={n_surrogates}; finite null variance"
            if reason_status != "ok"
            else ""
        ),
        observed_evidence=(
            f"status={status}; n_surrogates_finite={n_finite}; null_std={null_std}"
            if reason_status != "ok"
            else ""
        ),
        specification_id=f"{null_type}:{contract.endpoint_name}",
    )
    qc_row = attach_structured_reason(
        {
            "dataset_id": unit.dataset_id,
            "subject_id": unit.subject_id,
            "condition": unit.condition,
            "observation_id": unit.observation_id,
            "modality": unit.modality,
            "duration_s": int(unit.duration_s),
            "endpoint_name": contract.endpoint_name,
            "band": unit.band,
            "power_representation": unit.power_representation,
            "null_type": null_type,
            "null_status": status if reason_status == "ok" else reason_status,
            "n_surrogates_requested": int(n_surrogates),
            "n_surrogates_finite": int(n_finite),
            "notes": reason_text,
        },
        stage="C4",
        status="computed" if reason_status == "ok" else "not_computable",
        reason_code=reason_code,
        reason=reason_text,
        required_evidence=(
            f"n_surrogates={n_surrogates}; finite null variance"
            if reason_status != "ok"
            else ""
        ),
        observed_evidence=(
            f"status={status}; n_surrogates_finite={n_finite}; null_std={null_std}"
            if reason_status != "ok"
            else ""
        ),
        specification_id=f"{null_type}:{contract.endpoint_name}",
    )
    # Preserve legacy null-QC status vocabulary for downstream filters.
    qc_row["status"] = status if reason_status == "ok" else reason_status
    return subject_row, qc_row, list(null_vals)


def series_units_from_aligned_rows(
    aligned_rows: Sequence[Mapping[str, object]],
    *,
    duration_s: int,
    bands: Sequence[str] | None = None,
    representations: Sequence[tuple[str, str, bool]] = POWER_REPRESENTATIONS,
) -> list[SeriesUnit]:
    """Build analysis units from M4 aligned feature rows."""
    representation_lookup = {
        name: (suffix, is_primary) for name, suffix, is_primary in representations
    }
    units: list[SeriesUnit] = []
    for identity, duration_role, obs_rows in _group_aligned_rows(aligned_rows):
        ordered = sorted(obs_rows, key=lambda row: float(row["time_s"]))
        if not ordered:
            continue
        hr_z = np.asarray([_as_float(row["hr_z"]) for row in ordered], dtype=float)
        modality = _as_str(
            ordered[0].get("modality") or ordered[0].get("cardiac_modality"),
            "default",
        ).casefold()
        selected_bands = list(bands) if bands is not None else []
        if not selected_bands:
            # Discover bands present via primary absolute representation columns.
            for band_candidate in (
                "delta",
                "theta",
                "alpha",
                "beta",
            ):
                column = _eeg_column(band_candidate, "absolute_log10_power_z")
                if column in ordered[0]:
                    selected_bands.append(band_candidate)
        for band in selected_bands:
            for representation, suffix, is_primary in representations:
                if representation not in representation_lookup:
                    continue
                column = _eeg_column(band, suffix)
                if column not in ordered[0]:
                    continue
                eeg_z = np.asarray(
                    [_as_float(row[column]) for row in ordered], dtype=float
                )
                units.append(
                    SeriesUnit(
                        dataset_id=_as_str(identity["dataset_id"]).casefold(),
                        subject_id=_as_str(identity["subject_id"]),
                        task=_as_str(identity["task"]),
                        condition=_as_str(identity["condition"]).casefold(),
                        observation_id=_as_str(identity["observation_id"]),
                        modality=modality,
                        duration_s=int(duration_s),
                        duration_role=_as_str(duration_role) or _as_str(
                            ordered[0].get("duration_role")
                        ),
                        band=_as_str(band).casefold(),
                        power_representation=_as_str(representation).casefold(),
                        is_primary_representation=bool(is_primary),
                        pair=f"hr_x_{band}_{representation}",
                        hr_z=hr_z,
                        eeg_z=eeg_z,
                    )
                )
    return units


def _pool_key(unit: SeriesUnit) -> tuple[str, ...]:
    return (
        unit.dataset_id.casefold(),
        unit.condition.casefold(),
        unit.modality.casefold(),
        str(int(unit.duration_s)),
    )


def _observation_key(unit: SeriesUnit) -> str:
    return unit.observation_id


def unit_analysis_key(unit: SeriesUnit) -> str:
    """Stable unit identity for checkpoints and manifests."""
    return "|".join(
        [
            _as_str(unit.observation_id),
            str(int(unit.duration_s)),
            _as_str(unit.band).casefold(),
            _as_str(unit.power_representation).casefold(),
            _as_str(unit.modality, "default").casefold(),
        ]
    )


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve worker count: ``None``/``-1`` → all CPUs; ``>=1`` → that many."""
    if n_jobs is None or int(n_jobs) == -1:
        return max(1, int(os.cpu_count() or 1))
    resolved = int(n_jobs)
    if resolved < 1:
        raise ValueError("n_jobs must be >= 1 or -1 (all CPUs).")
    return resolved


def _format_hms(seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _build_pool_derangements(
    units: Sequence[SeriesUnit],
    *,
    n_surrogates: int,
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    pools: dict[tuple[str, ...], list[SeriesUnit]] = {}
    for unit in units:
        pools.setdefault(_pool_key(unit), []).append(unit)

    pool_derangements: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for pool, pool_units in pools.items():
        obs_ids = sorted({_observation_key(unit) for unit in pool_units})
        if len(obs_ids) < 2:
            pool_derangements[pool] = []
            continue
        pool_seed = deterministic_seed(
            "|".join(pool),
            "cross_subject_pool",
            NULL_TYPE_CROSS_SUBJECT_MISMATCH,
        )
        pool_rng = np.random.default_rng(pool_seed)
        mappings: list[dict[str, str]] = []
        for _ in range(int(n_surrogates)):
            perm = seeded_derangement(len(obs_ids), pool_rng)
            mappings.append(
                {obs_ids[i]: obs_ids[int(perm[i])] for i in range(len(obs_ids))}
            )
        pool_derangements[pool] = mappings
    return pool_derangements


def _by_obs_band_index(
    units: Sequence[SeriesUnit],
) -> dict[tuple[str, str, str, int], SeriesUnit]:
    by_obs_band: dict[tuple[str, str, str, int], SeriesUnit] = {}
    for unit in units:
        by_obs_band[
            (
                unit.observation_id,
                unit.band,
                unit.power_representation,
                int(unit.duration_s),
            )
        ] = unit
    return by_obs_band


def _resolve_cross_subject_partners(
    unit: SeriesUnit,
    *,
    pool_derangements: Mapping[tuple[str, ...], list[dict[str, str]]],
    by_obs_band: Mapping[tuple[str, str, str, int], SeriesUnit],
) -> tuple[list[np.ndarray], bool]:
    """Return ``(partners, missing_partner)``.

    Empty ``partners`` with ``missing_partner=False`` means derangement impossible
    (same as legacy empty-list path). ``missing_partner=True`` forces the
    ``missing_partner_series`` QC override after statistics are computed.
    """
    mappings = pool_derangements.get(_pool_key(unit), [])
    if not mappings:
        return [], False
    partners: list[np.ndarray] = []
    for mapping in mappings:
        partner_id = mapping[unit.observation_id]
        partner_unit = by_obs_band.get(
            (
                partner_id,
                unit.band,
                unit.power_representation,
                int(unit.duration_s),
            )
        )
        if partner_unit is None:
            return [], True
        partners.append(partner_unit.eeg_z)
    return partners, False


def _compute_observed_for_unit(unit: SeriesUnit) -> tuple[float, bool]:
    identity = {
        "dataset_id": unit.dataset_id,
        "subject_id": unit.subject_id,
        "task": unit.task,
        "condition": unit.condition,
        "observation_id": unit.observation_id,
    }
    observed = compute_endpoint_index_from_series(
        unit.hr_z,
        unit.eeg_z,
        duration_s=unit.duration_s,
        identity=identity,
        band=unit.band,
        power_representation=unit.power_representation,
        duration_role=unit.duration_role,
        is_primary_representation=unit.is_primary_representation,
        pair=unit.pair,
    )
    return _as_float(observed.get("endpoint_index")), bool(observed.get("eligible"))


def _surrogate_identity_fields(unit: SeriesUnit) -> dict[str, str]:
    """Resolve dataset/participant/session identifiers for surrogate rows."""
    # Reuse canonical key normalization already used by C5/C7 aggregation logic.
    from .group_tables import normalize_keys

    keys = normalize_keys(
        {
            "dataset_id": unit.dataset_id,
            "subject_id": unit.subject_id,
            "observation_id": unit.observation_id,
            "condition": unit.condition,
            "task": unit.task,
        }
    )
    return {
        "dataset_id": _as_str(keys.get("dataset_id"), unit.dataset_id).casefold(),
        "biological_participant_id": _as_str(keys.get("participant_id")).casefold(),
        "session_id": _as_str(keys.get("session_id"), "single").casefold() or "single",
        "analysis_unit_id": "|".join(
            [
                _as_str(keys.get("dataset_id"), unit.dataset_id).casefold(),
                _as_str(keys.get("participant_id")).casefold(),
                _as_str(keys.get("session_id"), "single").casefold() or "single",
            ]
        ),
    }


def _surrogate_rows_for_unit(
    *,
    unit: SeriesUnit,
    null_type: str,
    observed_endpoint_index: float,
    null_values: Sequence[float],
    null_mean: float,
    null_median: float,
    null_std: float,
    empirical_p: float,
    rng_seed_u64: object,
    n_surrogates: int,
    observed_eligible: bool,
    qc_status: str,
) -> list[dict[str, object]]:
    """Expand one unit's surrogate vector into row-wise long-form records."""
    ids = _surrogate_identity_fields(unit)
    state_role, time_role, _session = condition_semantics_for(
        unit.dataset_id,
        _as_str(unit.condition),
    )
    state = "rest" if state_role == "state_low" else ("task" if state_role == "state_high" else "")
    period = "pre" if time_role == "time_pre" else ("post" if time_role == "time_post" else "")
    obs_z = float("nan")
    if math.isfinite(null_std) and null_std > 0 and math.isfinite(observed_endpoint_index):
        obs_z = float((observed_endpoint_index - null_mean) / null_std)

    rows: list[dict[str, object]] = []
    for surrogate_i, surrogate_value in enumerate(null_values):
        surrogate_z = float("nan")
        if math.isfinite(null_std) and null_std > 0 and math.isfinite(float(surrogate_value)):
            surrogate_z = float((float(surrogate_value) - null_mean) / null_std)
        rows.append(
            {
                "dataset_id": ids["dataset_id"],
                "dataset_role": "",
                "biological_participant_id": ids["biological_participant_id"],
                "analysis_unit_id": ids["analysis_unit_id"],
                "subject_id": unit.subject_id,
                "session_id": ids["session_id"],
                "observation_id": unit.observation_id,
                "condition": unit.condition,
                "period": period,
                "state": state,
                "task": unit.task,
                "band": unit.band,
                "duration": int(unit.duration_s),
                "representation": unit.power_representation,
                "endpoint": contract_for_duration(unit.duration_s).endpoint_name,
                "null_type": null_type,
                "surrogate_index": int(surrogate_i),
                "surrogate_endpoint_index": float(surrogate_value),
                "observed_endpoint_index": observed_endpoint_index,
                "standardized_surrogate_value": surrogate_z,
                "standardized_observed_value": obs_z,
                "null_mean": null_mean,
                "null_median": null_median,
                "null_std": null_std,
                "empirical_p": empirical_p,
                "rng_seed_u64": _as_str(rng_seed_u64),
                "n_surrogates": int(n_surrogates),
                "eligibility_status": "eligible" if observed_eligible else "ineligible",
                "qc_status": qc_status,
            }
        )
    return rows


def process_one_unit(
    unit: SeriesUnit,
    *,
    null_types: Sequence[str],
    n_surrogates: int,
    pool_derangements: Mapping[tuple[str, ...], list[dict[str, str]]],
    by_obs_band: Mapping[tuple[str, str, str, int], SeriesUnit],
    cache_observed: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run all null types for one analysis unit (observed optionally cached)."""
    observed_stat: float | None = None
    observed_eligible: bool | None = None
    if cache_observed:
        observed_stat, observed_eligible = _compute_observed_for_unit(unit)

    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    surrogate_rows: list[dict[str, object]] = []
    for null_type in null_types:
        partner_eeg_by_surrogate: list[np.ndarray] | None = None
        missing_partner = False
        if null_type == NULL_TYPE_CROSS_SUBJECT_MISMATCH:
            partner_eeg_by_surrogate, missing_partner = _resolve_cross_subject_partners(
                unit,
                pool_derangements=pool_derangements,
                by_obs_band=by_obs_band,
            )

        subject_row, qc_row, null_values = _null_statistics_for_unit(
            unit,
            null_type=null_type,
            n_surrogates=n_surrogates,
            partner_eeg_by_surrogate=partner_eeg_by_surrogate,
            observed_stat=observed_stat,
            observed_eligible=observed_eligible,
        )
        if missing_partner:
            qc_row = {
                **qc_row,
                "status": "missing_partner_series",
                "notes": "Partner observation lacks matching band/representation.",
            }
        subject_rows.append(subject_row)
        qc_rows.append(qc_row)
        surrogate_rows.extend(
            _surrogate_rows_for_unit(
                unit=unit,
                null_type=null_type,
                observed_endpoint_index=_as_float(subject_row.get("observed_endpoint_index")),
                null_values=null_values,
                null_mean=_as_float(subject_row.get("null_mean")),
                null_median=_as_float(subject_row.get("null_median")),
                null_std=_as_float(subject_row.get("null_std")),
                empirical_p=_as_float(subject_row.get("empirical_p")),
                rng_seed_u64=subject_row.get("rng_seed_u64", ""),
                n_surrogates=int(subject_row.get("n_surrogates_requested", n_surrogates)),
                observed_eligible=bool(subject_row.get("observed_eligible")),
                qc_status=_as_str(qc_row.get("status"), "unknown"),
            )
        )
    return subject_rows, qc_rows, surrogate_rows


def _worker_process_unit(
    payload: tuple[
        int,
        SeriesUnit,
        tuple[str, ...],
        int,
        dict[tuple[str, ...], list[dict[str, str]]],
        dict[tuple[str, str, str, int], SeriesUnit],
        bool,
    ],
) -> tuple[int, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    (
        unit_index,
        unit,
        null_types,
        n_surrogates,
        pool_derangements,
        by_obs_band,
        cache_observed,
    ) = payload
    subject_rows, qc_rows, surrogate_rows = process_one_unit(
        unit,
        null_types=null_types,
        n_surrogates=n_surrogates,
        pool_derangements=pool_derangements,
        by_obs_band=by_obs_band,
        cache_observed=cache_observed,
    )
    return unit_index, subject_rows, qc_rows, surrogate_rows


def _checkpoint_path(checkpoint_dir: Path, unit_key: str) -> Path:
    digest = hashlib.sha256(unit_key.encode("utf-8")).hexdigest()[:32]
    return checkpoint_dir / f"unit_{digest}.json"


def _run_manifest_payload(
    *,
    n_surrogates: int,
    null_types: Sequence[str],
    unit_keys: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "n_surrogates": int(n_surrogates),
        "null_types": [str(name) for name in null_types],
        "unit_keys": list(unit_keys),
        "n_units": len(unit_keys),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _validate_checkpoint_payload(
    payload: Mapping[str, object],
    *,
    unit_key: str,
    n_surrogates: int,
    null_types: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
    if _as_str(payload.get("schema_version")) != CHECKPOINT_SCHEMA_VERSION:
        return None
    if _as_str(payload.get("unit_key")) != unit_key:
        return None
    if int(payload.get("n_surrogates", -1)) != int(n_surrogates):
        return None
    stored_types = [_as_str(v) for v in list(payload.get("null_types") or [])]
    if stored_types != [str(v) for v in null_types]:
        return None
    subject_rows = list(payload.get("subject_rows") or [])
    qc_rows = list(payload.get("qc_rows") or [])
    surrogate_rows = list(payload.get("surrogate_rows") or [])
    if len(subject_rows) != len(null_types) or len(qc_rows) != len(null_types):
        return None
    for row in subject_rows:
        if "rng_seed_u64" not in row or "n_surrogates_requested" not in row:
            return None
        if int(row.get("n_surrogates_requested", -1)) != int(n_surrogates):
            return None
    if surrogate_rows:
        expected_rows = len(null_types) * int(n_surrogates)
        if len(surrogate_rows) != expected_rows:
            return None
    return subject_rows, qc_rows, surrogate_rows


def _load_unit_checkpoint(
    checkpoint_dir: Path,
    *,
    unit_key: str,
    n_surrogates: int,
    null_types: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]] | None:
    path = _checkpoint_path(checkpoint_dir, unit_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _validate_checkpoint_payload(
        payload,
        unit_key=unit_key,
        n_surrogates=n_surrogates,
        null_types=null_types,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def _json_safe_row(row: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_safe(val) for key, val in row.items()}


def _write_unit_checkpoint(
    checkpoint_dir: Path,
    *,
    unit_key: str,
    n_surrogates: int,
    null_types: Sequence[str],
    subject_rows: Sequence[Mapping[str, object]],
    qc_rows: Sequence[Mapping[str, object]],
    surrogate_rows: Sequence[Mapping[str, object]],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "unit_key": unit_key,
        "n_surrogates": int(n_surrogates),
        "null_types": [str(name) for name in null_types],
        "subject_rows": [_json_safe_row(row) for row in subject_rows],
        "qc_rows": [_json_safe_row(row) for row in qc_rows],
        "surrogate_rows": [_json_safe_row(row) for row in surrogate_rows],
    }
    _atomic_write_json(_checkpoint_path(checkpoint_dir, unit_key), payload)


def _prepare_checkpoint_dir(
    checkpoint_dir: Path | None,
    *,
    n_surrogates: int,
    null_types: Sequence[str],
    unit_keys: Sequence[str],
) -> Path | None:
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    for tmp in path.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass
    manifest_path = path / RUN_MANIFEST_FILENAME
    expected = _run_manifest_payload(
        n_surrogates=n_surrogates,
        null_types=null_types,
        unit_keys=unit_keys,
    )
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing != expected:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, expected)
    return path


def _report_progress(
    *,
    done: int,
    total: int,
    start_time: float,
    last_report: float,
    force: bool = False,
) -> float:
    now = time.perf_counter()
    if not force and done not in (1, total) and (now - last_report) < 1.0:
        return last_report
    elapsed = now - start_time
    rate = elapsed / done if done else 0.0
    remaining = rate * (total - done) if done else 0.0
    print(
        f"[confirmatory] C4 progress: {done}/{total} units | "
        f"elapsed={_format_hms(elapsed)} | ETA={_format_hms(remaining)}",
        flush=True,
    )
    return now


def run_null_battery(
    units: Sequence[SeriesUnit],
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    null_types: Sequence[str] = NULL_TYPES,
    n_jobs: int | None = 1,
    checkpoint_dir: str | Path | None = None,
    progress: bool = False,
    cache_observed: bool = True,
) -> NullBatteryResult:
    """Run the confirmatory null battery for analysis units.

    Parameters
    ----------
    n_jobs:
        ``1`` = serial; ``-1``/``None`` = all CPUs; ``>1`` = that many workers.
    checkpoint_dir:
        When set, write/resume atomic per-unit checkpoints under this directory.
    cache_observed:
        Compute the observed endpoint once per unit and reuse across null types
        (bit-identical to recomputing for each null type).
    """
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1.")
    unknown = [name for name in null_types if name not in NULL_TYPES]
    if unknown:
        raise ValueError(f"Unknown null types: {unknown}.")

    unit_list = list(units)
    null_type_tuple = tuple(null_types)
    unit_keys = [unit_analysis_key(unit) for unit in unit_list]
    workers = resolve_n_jobs(n_jobs)

    ckpt_dir = _prepare_checkpoint_dir(
        Path(checkpoint_dir) if checkpoint_dir is not None else None,
        n_surrogates=n_surrogates,
        null_types=null_type_tuple,
        unit_keys=unit_keys,
    )

    pool_derangements = _build_pool_derangements(unit_list, n_surrogates=n_surrogates)
    by_obs_band = _by_obs_band_index(unit_list)

    results_by_index: dict[
        int,
        tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]],
    ] = {}
    pending_indices: list[int] = []
    for index, unit in enumerate(unit_list):
        if ckpt_dir is not None:
            loaded = _load_unit_checkpoint(
                ckpt_dir,
                unit_key=unit_keys[index],
                n_surrogates=n_surrogates,
                null_types=null_type_tuple,
            )
            if loaded is not None:
                results_by_index[index] = loaded
                continue
        pending_indices.append(index)

    total = len(unit_list)
    done = total - len(pending_indices)
    start_time = time.perf_counter()
    last_report = 0.0
    if progress and done:
        last_report = _report_progress(
            done=done, total=total, start_time=start_time, last_report=last_report, force=True
        )

    def _store_result(
        index: int,
        subject_rows: list[dict[str, object]],
        qc_rows: list[dict[str, object]],
        surrogate_rows: list[dict[str, object]],
    ) -> None:
        nonlocal done, last_report
        results_by_index[index] = (subject_rows, qc_rows, surrogate_rows)
        if ckpt_dir is not None:
            _write_unit_checkpoint(
                ckpt_dir,
                unit_key=unit_keys[index],
                n_surrogates=n_surrogates,
                null_types=null_type_tuple,
                subject_rows=subject_rows,
                qc_rows=qc_rows,
                surrogate_rows=surrogate_rows,
            )
        done += 1
        if progress:
            last_report = _report_progress(
                done=done,
                total=total,
                start_time=start_time,
                last_report=last_report,
                force=(done == total),
            )

    if not pending_indices:
        if progress and total:
            _report_progress(
                done=total, total=total, start_time=start_time, last_report=last_report, force=True
            )
    elif workers == 1 or len(pending_indices) == 1:
        for index in pending_indices:
            subject_rows, qc_rows, surrogate_rows = process_one_unit(
                unit_list[index],
                null_types=null_type_tuple,
                n_surrogates=n_surrogates,
                pool_derangements=pool_derangements,
                by_obs_band=by_obs_band,
                cache_observed=cache_observed,
            )
            _store_result(index, subject_rows, qc_rows, surrogate_rows)
    else:
        max_workers = min(workers, len(pending_indices))
        payloads = [
            (
                index,
                unit_list[index],
                null_type_tuple,
                int(n_surrogates),
                pool_derangements,
                by_obs_band,
                bool(cache_observed),
            )
            for index in pending_indices
        ]
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_worker_process_unit, payload): payload[0]
                    for payload in payloads
                }
                for future in as_completed(futures):
                    index, subject_rows, qc_rows, surrogate_rows = future.result()
                    _store_result(index, subject_rows, qc_rows, surrogate_rows)
        except PermissionError:
            # Some sandboxed environments disallow process semaphores; fall back
            # to deterministic serial execution.
            for index in pending_indices:
                subject_rows, qc_rows, surrogate_rows = process_one_unit(
                    unit_list[index],
                    null_types=null_type_tuple,
                    n_surrogates=n_surrogates,
                    pool_derangements=pool_derangements,
                    by_obs_band=by_obs_band,
                    cache_observed=cache_observed,
                )
                _store_result(index, subject_rows, qc_rows, surrogate_rows)

    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    surrogate_rows: list[dict[str, object]] = []
    for index in range(total):
        unit_subject, unit_qc, unit_surrogate = results_by_index[index]
        subject_rows.extend(unit_subject)
        qc_rows.extend(unit_qc)
        surrogate_rows.extend(unit_surrogate)

    summary_rows = summarize_null_subject_results(subject_rows)
    return NullBatteryResult(
        subject_rows=tuple(subject_rows),
        summary_rows=tuple(summary_rows),
        qc_rows=tuple(qc_rows),
        surrogate_rows=tuple(surrogate_rows),
    )


def summarize_null_subject_results(
    subject_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate null results without mixing endpoint types."""
    buckets: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    order: list[tuple[str, ...]] = []
    for row in subject_rows:
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("condition")).casefold(),
            _as_str(row.get("modality")).casefold(),
            str(int(_as_float(row.get("duration_s")))),
            _as_str(row.get("endpoint_name")).casefold(),
            _as_str(row.get("band")).casefold(),
            _as_str(row.get("power_representation")).casefold(),
            _as_str(row.get("null_type")).casefold(),
        )
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)

    summaries: list[dict[str, object]] = []
    for key in order:
        rows = buckets[key]
        first = rows[0]
        p_vals = [_as_float(r.get("empirical_p")) for r in rows]
        effects = [_as_float(r.get("effect_size_surrogate_z")) for r in rows]
        observed = [_as_float(r.get("observed_endpoint_index")) for r in rows]
        summaries.append(
            {
                "dataset_id": first["dataset_id"],
                "condition": first["condition"],
                "modality": first["modality"],
                "duration_s": int(_as_float(first.get("duration_s"))),
                "endpoint_name": first["endpoint_name"],
                "is_standard_zlpi": bool(first.get("is_standard_zlpi")),
                "band": first["band"],
                "power_representation": first["power_representation"],
                "null_type": first["null_type"],
                "n_units": len(rows),
                "n_eligible_observed": sum(
                    1 for r in rows if bool(r.get("observed_eligible"))
                ),
                "median_empirical_p": _median(p_vals),
                "mean_empirical_p": _mean(p_vals),
                "mean_effect_size_surrogate_z": _mean(effects),
                "median_effect_size_surrogate_z": _median(effects),
                "median_observed_endpoint_index": _median(observed),
            }
        )
    return summaries


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


def _read_null_result_csvs(output_path: Path) -> NullBatteryResult | None:
    subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
    summary_path = output_path / NULL_SUMMARY_FILENAME
    qc_path = output_path / NULL_QC_FILENAME
    surrogate_path = output_path / NULL_SURROGATE_VALUES_FILENAME
    if not (subject_path.is_file() and summary_path.is_file() and qc_path.is_file()):
        return None
    with subject_path.open(encoding="utf-8", newline="") as handle:
        subject_rows = list(csv.DictReader(handle))
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with qc_path.open(encoding="utf-8", newline="") as handle:
        qc_rows = list(csv.DictReader(handle))
    surrogate_rows: list[dict[str, str]] = []
    if surrogate_path.is_file():
        with surrogate_path.open(encoding="utf-8", newline="") as handle:
            surrogate_rows = list(csv.DictReader(handle))
    return NullBatteryResult(
        subject_rows=tuple(subject_rows),
        summary_rows=tuple(summary_rows),
        qc_rows=tuple(qc_rows),
        surrogate_rows=tuple(surrogate_rows),
    )


def _validate_complete_marker(
    output_path: Path,
    *,
    n_surrogates: int,
    n_units: int,
    null_types: Sequence[str],
    configuration_hash_sha256: str | None = None,
    root_seed: int | None = None,
) -> bool:
    marker = output_path / COMPLETE_MARKER_FILENAME
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if int(payload.get("n_surrogates", -1)) != int(n_surrogates):
        return False
    if int(payload.get("n_units", -1)) != int(n_units):
        return False
    stored_types = [_as_str(v) for v in list(payload.get("null_types") or [])]
    if stored_types != [str(v) for v in null_types]:
        return False
    if configuration_hash_sha256 is not None:
        stored_hash = _as_str(payload.get("configuration_hash_sha256"))
        if stored_hash and stored_hash != str(configuration_hash_sha256):
            return False
    if root_seed is not None and "root_seed" in payload:
        if int(payload.get("root_seed", -1)) != int(root_seed):
            return False
    return True


class StaleC4CacheError(RuntimeError):
    """Raised when cached C4 artifacts do not match the current run contract."""


def read_c4_cache_fingerprint(c4_dir: str | Path) -> dict[str, object]:
    """Read C4 complete-marker / subject-table fingerprint for cache validation."""
    output_path = Path(c4_dir).expanduser().resolve()
    marker = output_path / COMPLETE_MARKER_FILENAME
    fingerprint: dict[str, object] = {
        "c4_dir": str(output_path),
        "marker_present": marker.is_file(),
        "n_surrogates": None,
        "configuration_hash_sha256": "",
        "root_seed": None,
        "null_types": (),
        "n_units": None,
    }
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            fingerprint["n_surrogates"] = (
                int(payload["n_surrogates"])
                if payload.get("n_surrogates") is not None
                else None
            )
            fingerprint["configuration_hash_sha256"] = _as_str(
                payload.get("configuration_hash_sha256")
            )
            if payload.get("root_seed") is not None:
                fingerprint["root_seed"] = int(payload.get("root_seed"))
            fingerprint["null_types"] = tuple(
                _as_str(v) for v in list(payload.get("null_types") or [])
            )
            if payload.get("n_units") is not None:
                fingerprint["n_units"] = int(payload.get("n_units"))
    if fingerprint["n_surrogates"] is None:
        subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
        if subject_path.is_file():
            with subject_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            values = {
                int(_as_float(r.get("n_surrogates_requested")))
                for r in rows
                if math.isfinite(_as_float(r.get("n_surrogates_requested")))
            }
            if values:
                fingerprint["n_surrogates"] = int(min(values))
    return fingerprint


def validate_c4_cache_for_run(
    c4_dir: str | Path,
    *,
    expected_n_surrogates: int,
    configuration_hash_sha256: str | None = None,
    root_seed: int | None = None,
    require_present: bool = False,
) -> dict[str, object]:
    """Reject stale C4 caches when surrogate count / config fingerprint diverge.

    Typical failure: smoke ``n_surrogates=20`` artifacts left under a production
    tree that now requests ``n_surrogates=500``. Downstream C5/C6/C7 stages that
    consume C4 must call this helper before reading null exports.
    """
    output_path = Path(c4_dir).expanduser().resolve()
    fingerprint = read_c4_cache_fingerprint(output_path)
    marker = output_path / COMPLETE_MARKER_FILENAME
    subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
    present = marker.is_file() or subject_path.is_file()
    if not present:
        if require_present:
            raise StaleC4CacheError(
                f"C4 artifacts required under {output_path} but none were found "
                f"(expected n_surrogates={expected_n_surrogates})."
            )
        return {**fingerprint, "validated": False, "present": False}

    observed = fingerprint.get("n_surrogates")
    if observed is None:
        raise StaleC4CacheError(
            f"C4 cache under {output_path} is missing n_surrogates fingerprint; "
            f"refusing to consume potentially stale nulls "
            f"(expected n_surrogates={expected_n_surrogates})."
        )
    if int(observed) != int(expected_n_surrogates):
        raise StaleC4CacheError(
            f"Stale C4 cache under {output_path}: "
            f"n_surrogates={observed} but run expects {expected_n_surrogates}. "
            f"Re-run C4 (reason_code={CONFIGURATION_VALIDATION_FAILED})."
        )
    stored_hash = _as_str(fingerprint.get("configuration_hash_sha256"))
    if (
        configuration_hash_sha256
        and stored_hash
        and stored_hash != str(configuration_hash_sha256)
    ):
        raise StaleC4CacheError(
            f"Stale C4 cache under {output_path}: configuration_hash mismatch "
            f"(cached={stored_hash}, expected={configuration_hash_sha256}). "
            f"Re-run C4 (reason_code={CONFIGURATION_VALIDATION_FAILED})."
        )
    stored_seed = fingerprint.get("root_seed")
    if (
        root_seed is not None
        and stored_seed is not None
        and int(stored_seed) != int(root_seed)
    ):
        raise StaleC4CacheError(
            f"Stale C4 cache under {output_path}: root_seed mismatch "
            f"(cached={stored_seed}, expected={root_seed}). "
            f"Re-run C4 (reason_code={CONFIGURATION_VALIDATION_FAILED})."
        )
    return {**fingerprint, "validated": True, "present": True}


def _delete_final_outputs(output_path: Path) -> None:
    for name in (
        NULL_SUBJECT_RESULTS_FILENAME,
        NULL_SUMMARY_FILENAME,
        NULL_QC_FILENAME,
        NULL_SURROGATE_VALUES_FILENAME,
        COMPLETE_MARKER_FILENAME,
    ):
        path = output_path / name
        if path.is_file():
            path.unlink()
        tmp = output_path / f"{name}.tmp"
        if tmp.is_file():
            tmp.unlink()


def write_null_outputs(
    result: NullBatteryResult,
    output_dir: str | Path,
    *,
    n_surrogates: int | None = None,
    n_units: int | None = None,
    null_types: Sequence[str] = NULL_TYPES,
    wall_time_s: float | None = None,
    configuration_hash_sha256: str | None = None,
    root_seed: int | None = None,
) -> dict[str, Path]:
    """Write null CSVs atomically, then the ``C4_COMPLETE.json`` marker."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
    summary_path = output_path / NULL_SUMMARY_FILENAME
    qc_path = output_path / NULL_QC_FILENAME
    surrogate_path = output_path / NULL_SURROGATE_VALUES_FILENAME

    subject_tmp = output_path / f"{NULL_SUBJECT_RESULTS_FILENAME}.tmp"
    summary_tmp = output_path / f"{NULL_SUMMARY_FILENAME}.tmp"
    qc_tmp = output_path / f"{NULL_QC_FILENAME}.tmp"
    surrogate_tmp = output_path / f"{NULL_SURROGATE_VALUES_FILENAME}.tmp"
    _write_csv(subject_tmp, result.subject_rows, SUBJECT_RESULT_FIELDS)
    _write_csv(summary_tmp, result.summary_rows, SUMMARY_FIELDS)
    _write_csv(qc_tmp, result.qc_rows, QC_FIELDS)
    _write_csv(surrogate_tmp, result.surrogate_rows, SURROGATE_VALUE_FIELDS)
    os.replace(subject_tmp, subject_path)
    os.replace(summary_tmp, summary_path)
    os.replace(qc_tmp, qc_path)
    os.replace(surrogate_tmp, surrogate_path)

    resolved_n_surr = (
        int(n_surrogates)
        if n_surrogates is not None
        else (
            int(result.subject_rows[0]["n_surrogates_requested"])
            if result.subject_rows
            else DEFAULT_N_SURROGATES
        )
    )
    resolved_n_units = (
        int(n_units)
        if n_units is not None
        else (
            len(result.subject_rows) // max(1, len(null_types))
            if result.subject_rows
            else 0
        )
    )
    marker_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "n_surrogates": resolved_n_surr,
        "n_units": resolved_n_units,
        "null_types": [str(name) for name in null_types],
        "n_subject_rows": len(result.subject_rows),
        "n_qc_rows": len(result.qc_rows),
        "wall_time_s": None if wall_time_s is None else float(wall_time_s),
        "configuration_hash_sha256": str(configuration_hash_sha256 or ""),
        "root_seed": None if root_seed is None else int(root_seed),
    }
    _atomic_write_json(output_path / COMPLETE_MARKER_FILENAME, marker_payload)
    return {
        "null_subject_results": subject_path,
        "null_summary": summary_path,
        "null_qc": qc_path,
        "null_surrogate_values": surrogate_path,
        "complete_marker": output_path / COMPLETE_MARKER_FILENAME,
    }


PANEL_A_NULL_TYPES: tuple[str, ...] = (
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_PHASE_RANDOMIZATION,
    NULL_TYPE_BLOCK_SHUFFLE,
)


def surrogate_export_is_complete(
    surrogate_rows: Sequence[Mapping[str, object]],
    *,
    expected_n_surrogates: int | None = None,
) -> bool:
    """True when each observation×null block has the configured surrogate count."""
    if not surrogate_rows:
        return False
    buckets: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in surrogate_rows:
        key = (
            _as_str(row.get("observation_id")).casefold(),
            _as_str(row.get("null_type")).casefold(),
            _as_str(row.get("band")).casefold(),
        )
        buckets.setdefault(key, []).append(row)
    for rows in buckets.values():
        n_req = expected_n_surrogates
        if n_req is None:
            n_req = int(_as_float(rows[0].get("n_surrogates")))
        if n_req < 2:
            return False
        if len(rows) != int(n_req):
            return False
        idxs = sorted(int(_as_float(r.get("surrogate_index"))) for r in rows)
        if idxs != list(range(int(n_req))):
            return False
    return True


def export_panel_a_surrogate_values(
    aligned_dir: str | Path,
    output_dir: str | Path,
    *,
    n_surrogates: int | None = None,
    duration_s: int = 240,
    band: str = "theta",
    representation: str = "absolute_log10",
    null_types: Sequence[str] = PANEL_A_NULL_TYPES,
    n_jobs: int | None = -1,
    progress: bool = True,
) -> Path:
    """Regenerate full Panel A surrogate rows without rewriting C4 summaries.

    Uses the same deterministic seeds as C4 so ``null_mean`` / ``empirical_p``
    remain bit-identical to existing ``null_subject_results.csv`` when the
    surrogate count matches the prior run.
    """
    root = Path(aligned_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
    if n_surrogates is None:
        if not subject_path.is_file():
            raise FileNotFoundError(
                f"Cannot infer n_surrogates: missing {subject_path}."
            )
        with subject_path.open(encoding="utf-8", newline="") as handle:
            subject_rows = list(csv.DictReader(handle))
        n_vals = [
            int(_as_float(r.get("n_surrogates_requested")))
            for r in subject_rows
            if _as_str(r.get("band")).casefold() == band.casefold()
            and _as_str(r.get("null_type")).casefold()
            in {t.casefold() for t in null_types}
            and int(_as_float(r.get("duration_s"))) == int(duration_s)
        ]
        if not n_vals:
            raise ValueError("No matching null_subject rows for Panel A slice.")
        n_surrogates = int(min(n_vals))

    tables = discover_aligned_duration_tables(root, durations=(duration_s,))
    if int(duration_s) not in tables:
        raise FileNotFoundError(
            f"Aligned features for D{duration_s} not found under {root}."
        )
    units = [
        unit
        for unit in series_units_from_aligned_rows(
            read_aligned_features_csv(tables[int(duration_s)]),
            duration_s=int(duration_s),
            bands=(band,),
        )
        if unit.power_representation.casefold() == representation.casefold()
    ]
    if not units:
        raise ValueError(
            f"No SeriesUnits for Panel A slice D{duration_s}/{band}/{representation}."
        )

    if progress:
        print(
            f"[confirmatory] Panel A surrogate export: {len(units)} units × "
            f"{len(null_types)} nulls × {n_surrogates} surrogates",
            flush=True,
        )
    # Dedicated checkpoint tree so Panel A export can resume without touching
    # the production C4 unit checkpoints (which may predate surrogate rows).
    panel_a_ckpt = output_path / "_panel_a_surrogate_checkpoints"
    result = run_null_battery(
        units,
        n_surrogates=int(n_surrogates),
        null_types=tuple(null_types),
        n_jobs=n_jobs,
        checkpoint_dir=panel_a_ckpt,
        progress=progress,
        cache_observed=True,
    )
    surrogate_path = output_path / NULL_SURROGATE_VALUES_FILENAME
    tmp = output_path / f"{NULL_SURROGATE_VALUES_FILENAME}.tmp"
    _write_csv(tmp, result.surrogate_rows, SURROGATE_VALUE_FIELDS)
    os.replace(tmp, surrogate_path)
    if not surrogate_export_is_complete(
        result.surrogate_rows, expected_n_surrogates=int(n_surrogates)
    ):
        raise RuntimeError("Panel A surrogate export incomplete after regeneration.")
    return surrogate_path


def run_confirmatory_nulls(
    aligned_dir: str | Path,
    output_dir: str | Path,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    durations: Sequence[int] = EXPECTED_DURATIONS_S,
    null_types: Sequence[str] = NULL_TYPES,
    n_jobs: int | None = -1,
    progress: bool = True,
    cache_observed: bool = True,
    configuration_hash_sha256: str | None = None,
    root_seed: int | None = None,
) -> NullBatteryResult:
    """Load aligned M4 tables, run nulls, and write result CSVs."""
    root = Path(aligned_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    null_type_tuple = tuple(null_types)

    tables = discover_aligned_duration_tables(root, durations=durations)
    units: list[SeriesUnit] = []
    for duration_s, path in sorted(tables.items()):
        rows = read_aligned_features_csv(path)
        units.extend(series_units_from_aligned_rows(rows, duration_s=duration_s))

    if _validate_complete_marker(
        output_path,
        n_surrogates=n_surrogates,
        n_units=len(units),
        null_types=null_type_tuple,
        configuration_hash_sha256=configuration_hash_sha256,
        root_seed=root_seed,
    ):
        existing = _read_null_result_csvs(output_path)
        if (
            existing is not None
            and len(existing.subject_rows) == len(units) * len(null_type_tuple)
            and surrogate_export_is_complete(
                existing.surrogate_rows, expected_n_surrogates=n_surrogates
            )
        ):
            if progress:
                print(
                    f"[confirmatory] C4 complete marker valid "
                    f"({len(units)} units, n_surrogates={n_surrogates}); skipping.",
                    flush=True,
                )
            return existing

    # Final CSVs without a valid complete marker are treated as corrupt.
    if any(
        (output_path / name).is_file()
        for name in (
            NULL_SUBJECT_RESULTS_FILENAME,
            NULL_SUMMARY_FILENAME,
            NULL_QC_FILENAME,
            NULL_SURROGATE_VALUES_FILENAME,
        )
    ) and not _validate_complete_marker(
        output_path,
        n_surrogates=n_surrogates,
        n_units=len(units),
        null_types=null_type_tuple,
        configuration_hash_sha256=configuration_hash_sha256,
        root_seed=root_seed,
    ):
        _delete_final_outputs(output_path)

    checkpoint_dir = output_path / CHECKPOINT_DIRNAME
    t0 = time.perf_counter()
    result = run_null_battery(
        units,
        n_surrogates=n_surrogates,
        null_types=null_type_tuple,
        n_jobs=n_jobs,
        checkpoint_dir=checkpoint_dir,
        progress=progress,
        cache_observed=cache_observed,
    )
    wall = time.perf_counter() - t0
    write_null_outputs(
        result,
        output_path,
        n_surrogates=n_surrogates,
        n_units=len(units),
        null_types=null_type_tuple,
        wall_time_s=wall,
        configuration_hash_sha256=configuration_hash_sha256,
        root_seed=root_seed,
    )
    return result


__all__ = [
    "BLOCK_LENGTH_S",
    "CHECKPOINT_DIRNAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPLETE_MARKER_FILENAME",
    "DEFAULT_N_SURROGATES",
    "MIN_CIRCULAR_SHIFT_S",
    "NULL_QC_FILENAME",
    "NULL_SUBJECT_RESULTS_FILENAME",
    "NULL_SURROGATE_VALUES_FILENAME",
    "NULL_SUMMARY_FILENAME",
    "NULL_TYPES",
    "PANEL_A_NULL_TYPES",
    "SMOKE_N_SURROGATES",
    "SURROGATE_VALUE_FIELDS",
    "SUBJECT_RESULT_FIELDS",
    "StaleC4CacheError",
    "read_c4_cache_fingerprint",
    "validate_c4_cache_for_run",
    "NullBatteryResult",
    "SeriesUnit",
    "amplitude_spectrum",
    "analysis_key",
    "ar1_innovations",
    "block_shuffle_series",
    "circular_shift_series",
    "complete_blocks",
    "compute_endpoint_index_from_series",
    "deterministic_seed",
    "empirical_p_value",
    "export_panel_a_surrogate_values",
    "lag1_autocorrelation",
    "phase_randomize_series",
    "process_one_unit",
    "resolve_n_jobs",
    "run_confirmatory_nulls",
    "run_null_battery",
    "seeded_derangement",
    "series_units_from_aligned_rows",
    "surrogate_effect_size",
    "surrogate_export_is_complete",
    "unit_analysis_key",
    "valid_circular_shifts",
    "write_null_outputs",
]
