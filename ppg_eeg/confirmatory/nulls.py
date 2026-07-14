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
- Production / confirmatory default: ``DEFAULT_N_SURROGATES`` (1000)
- Smoke tests: ``SMOKE_N_SURROGATES`` (20), typically via dataset YAML
  ``n_surrogates: 20``

Reproducibility
---------------
Each subject×null-type row records ``rng_seed_u64``: the uint64 seed passed
to ``numpy.random.default_rng`` for that unit. Seeds are derived from SHA-256
of ``observation_id``, analysis key, and null type (not Python ``hash()``).
"""

from __future__ import annotations

import csv
import hashlib
import math
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

NULL_SUBJECT_RESULTS_FILENAME = "null_subject_results.csv"
NULL_SUMMARY_FILENAME = "null_summary.csv"
NULL_QC_FILENAME = "null_qc.csv"

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
DEFAULT_N_SURROGATES = 1000
MIN_CIRCULAR_SHIFT_S = 60
BLOCK_LENGTH_S = 30

SUBJECT_RESULT_FIELDS = (
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

QC_FIELDS = (
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
    "status",
    "n_surrogates_requested",
    "n_surrogates_finite",
    "notes",
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


def complete_blocks(values: np.ndarray, *, block_length_s: int = BLOCK_LENGTH_S) -> np.ndarray:
    """Return complete non-overlapping blocks as a ``(n_blocks, block_len)`` array."""
    arr = np.asarray(values, dtype=float)
    block_len = int(block_length_s)
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
) -> tuple[dict[str, object], dict[str, object]]:
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

    subject_row = {
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
        "null_std": _std(null_vals),
        "null_median": _median(null_vals),
        "empirical_p": p_value,
        "effect_size_surrogate_z": effect,
        "rng_seed_u64": int(seed),
        "analysis_key": key,
    }
    qc_row = {
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
        "status": status,
        "n_surrogates_requested": int(n_surrogates),
        "n_surrogates_finite": int(n_finite),
        "notes": notes,
    }
    return subject_row, qc_row


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


def run_null_battery(
    units: Sequence[SeriesUnit],
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    null_types: Sequence[str] = NULL_TYPES,
) -> NullBatteryResult:
    """Run the confirmatory null battery for analysis units."""
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1.")
    unknown = [name for name in null_types if name not in NULL_TYPES]
    if unknown:
        raise ValueError(f"Unknown null types: {unknown}.")

    subject_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    pools: dict[tuple[str, ...], list[SeriesUnit]] = {}
    for unit in units:
        pools.setdefault(_pool_key(unit), []).append(unit)

    # One seeded derangement per surrogate within each pool.
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

    for unit in units:
        for null_type in null_types:
            partner_eeg_by_surrogate: list[np.ndarray] | None = None
            if null_type == NULL_TYPE_CROSS_SUBJECT_MISMATCH:
                mappings = pool_derangements.get(_pool_key(unit), [])
                if not mappings:
                    subject_row, qc_row = _null_statistics_for_unit(
                        unit,
                        null_type=null_type,
                        n_surrogates=n_surrogates,
                        partner_eeg_by_surrogate=[],
                    )
                    subject_rows.append(subject_row)
                    qc_rows.append(qc_row)
                    continue
                partners: list[np.ndarray] = []
                missing = False
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
                        missing = True
                        break
                    partners.append(partner_unit.eeg_z)
                if missing:
                    subject_row, qc_row = _null_statistics_for_unit(
                        unit,
                        null_type=null_type,
                        n_surrogates=n_surrogates,
                        partner_eeg_by_surrogate=[],
                    )
                    qc_row = {
                        **qc_row,
                        "status": "missing_partner_series",
                        "notes": "Partner observation lacks matching band/representation.",
                    }
                    subject_rows.append(subject_row)
                    qc_rows.append(qc_row)
                    continue
                partner_eeg_by_surrogate = partners

            subject_row, qc_row = _null_statistics_for_unit(
                unit,
                null_type=null_type,
                n_surrogates=n_surrogates,
                partner_eeg_by_surrogate=partner_eeg_by_surrogate,
            )
            subject_rows.append(subject_row)
            qc_rows.append(qc_row)

    summary_rows = summarize_null_subject_results(subject_rows)
    return NullBatteryResult(
        subject_rows=tuple(subject_rows),
        summary_rows=tuple(summary_rows),
        qc_rows=tuple(qc_rows),
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


def write_null_outputs(
    result: NullBatteryResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    subject_path = output_path / NULL_SUBJECT_RESULTS_FILENAME
    summary_path = output_path / NULL_SUMMARY_FILENAME
    qc_path = output_path / NULL_QC_FILENAME
    _write_csv(subject_path, result.subject_rows, SUBJECT_RESULT_FIELDS)
    _write_csv(summary_path, result.summary_rows, SUMMARY_FIELDS)
    _write_csv(qc_path, result.qc_rows, QC_FIELDS)
    return {
        "null_subject_results": subject_path,
        "null_summary": summary_path,
        "null_qc": qc_path,
    }


def run_confirmatory_nulls(
    aligned_dir: str | Path,
    output_dir: str | Path,
    *,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    durations: Sequence[int] = EXPECTED_DURATIONS_S,
    null_types: Sequence[str] = NULL_TYPES,
) -> NullBatteryResult:
    """Load aligned M4 tables, run nulls, and write result CSVs."""
    root = Path(aligned_dir).expanduser().resolve()
    tables = discover_aligned_duration_tables(root, durations=durations)
    units: list[SeriesUnit] = []
    for duration_s, path in sorted(tables.items()):
        rows = read_aligned_features_csv(path)
        units.extend(series_units_from_aligned_rows(rows, duration_s=duration_s))
    result = run_null_battery(
        units,
        n_surrogates=n_surrogates,
        null_types=null_types,
    )
    write_null_outputs(result, output_dir)
    return result


__all__ = [
    "BLOCK_LENGTH_S",
    "DEFAULT_N_SURROGATES",
    "MIN_CIRCULAR_SHIFT_S",
    "NULL_QC_FILENAME",
    "NULL_SUBJECT_RESULTS_FILENAME",
    "NULL_SUMMARY_FILENAME",
    "NULL_TYPES",
    "SMOKE_N_SURROGATES",
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
    "lag1_autocorrelation",
    "phase_randomize_series",
    "run_confirmatory_nulls",
    "run_null_battery",
    "seeded_derangement",
    "series_units_from_aligned_rows",
    "surrogate_effect_size",
    "valid_circular_shifts",
    "write_null_outputs",
]
