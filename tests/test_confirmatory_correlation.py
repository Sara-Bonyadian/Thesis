from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.temporal_coupling.confirmatory.correlation import (
    BAND_ORDER,
    CURVES_TEMPLATE,
    MID_WINDOW_ANALYSIS_ROLE,
    MID_WINDOW_DURATION_S,
    MID_WINDOW_N_LAGS,
    POWER_REPRESENTATIONS,
    PRIMARY_POWER_REPRESENTATION,
    QC_TEMPLATE,
    SHORT_WINDOW_ANALYSIS_ROLE,
    SHORT_WINDOW_DURATION_S,
    SHORT_WINDOW_N_LAGS,
    STANDARD_ANALYSIS_ROLE,
    STANDARD_ZLPI_DURATIONS_S,
    STANDARD_ZLPI_N_LAGS,
    compute_signed_lag_curves,
    expected_lag_count,
    lag_spec_for_duration,
    write_correlation_outputs,
)
from ppg_eeg.temporal_coupling.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    MWPI_FLANKS_S,
    SWPI_FLANKS_S,
    ZLPI_FLANKS_S,
    contract_for_duration,
)


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if int(np.sum(finite)) < 2:
        return result
    subset = values[finite]
    std = float(np.std(subset, ddof=0))
    mean = float(np.mean(subset))
    if std <= 0:
        result[finite] = 0.0
        return result
    result[finite] = (subset - mean) / std
    return result


def _shift(signal: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(signal.shape, np.nan, dtype=float)
    if lag > 0:
        out[lag:] = signal[:-lag]
    elif lag < 0:
        out[:lag] = signal[-lag:]
    else:
        out[:] = signal
    return out


def _noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def _aligned_rows_from_z(
    hr_z: np.ndarray,
    band_z: dict[str, np.ndarray],
    *,
    duration_s: int = 240,
    start_s: float = 0.0,
    duration_role: str = "primary",
) -> list[dict[str, object]]:
    n = int(hr_z.size)
    times = start_s + np.arange(n, dtype=float)
    rows: list[dict[str, object]] = []
    for index, time_s in enumerate(times):
        row: dict[str, object] = {
            "dataset_id": "ds_test",
            "subject_id": "sub-01",
            "task": "rest",
            "condition": "rest",
            "observation_id": "obs-01",
            "duration_s": duration_s,
            "duration_role": duration_role,
            "time_s": float(time_s),
            "hr_bpm": float(hr_z[index]) if np.isfinite(hr_z[index]) else float("nan"),
            "hr_z": float(hr_z[index]),
        }
        for band in BAND_ORDER:
            z_value = float(band_z[band][index])
            raw = z_value if math.isfinite(z_value) else float("nan")
            row[f"{band}_absolute_power"] = raw
            row[f"{band}_absolute_log10_power"] = raw
            row[f"{band}_relative_power"] = 0.25
            row[f"{band}_broadband_residualized_log10"] = raw
            row[f"{band}_absolute_log10_power_z"] = z_value
            row[f"{band}_relative_power_z"] = z_value
            row[f"{band}_broadband_residualized_log10_z"] = z_value
        rows.append(row)
    return rows


def _aligned_rows(
    hr: np.ndarray,
    band_signals: dict[str, np.ndarray],
    *,
    duration_s: int = 240,
    duration_role: str = "primary",
) -> list[dict[str, object]]:
    hr_z = _zscore(hr.astype(float))
    band_z = {band: _zscore(signal.astype(float)) for band, signal in band_signals.items()}
    return _aligned_rows_from_z(
        hr_z, band_z, duration_s=duration_s, duration_role=duration_role
    )


def _curve_map(
    result,
    *,
    band: str = "theta",
    representation: str = PRIMARY_POWER_REPRESENTATION,
) -> dict[float, tuple[float, int]]:
    out: dict[float, tuple[float, int]] = {}
    for row in result.curve_rows:
        if row["band"] != band or row["power_representation"] != representation:
            continue
        out[float(row["lag_s"])] = (float(row["r"]), int(row["n_overlap"]))
    return out


def _peak_lag(curve: dict[float, tuple[float, int]]) -> float:
    return max(
        curve.items(),
        key=lambda item: item[1][0] if math.isfinite(item[1][0]) else -np.inf,
    )[0]


def _assert_constant_overlap(curve: dict[float, tuple[float, int]], expected: int) -> None:
    overlaps = {n_overlap for _, n_overlap in curve.values()}
    assert overlaps == {expected}, overlaps


class TestConfirmatoryCorrelation(unittest.TestCase):
    def test_duration_lag_and_endpoint_contracts(self) -> None:
        for duration in STANDARD_ZLPI_DURATIONS_S:
            spec = lag_spec_for_duration(duration)
            contract = contract_for_duration(duration)
            self.assertEqual(expected_lag_count(duration), STANDARD_ZLPI_N_LAGS)
            self.assertEqual(spec.lag_max_s, 60)
            self.assertEqual(spec.n_lags, 121)
            self.assertTrue(spec.is_standard_zlpi)
            self.assertTrue(spec.pool_with_standard_zlpi)
            self.assertEqual(spec.endpoint_name, ENDPOINT_ZLPI)
            self.assertEqual(spec.flank_inner_s, ZLPI_FLANKS_S[0])
            self.assertEqual(spec.flank_outer_s, ZLPI_FLANKS_S[1])
            self.assertEqual(spec.lag_analysis_role, STANDARD_ANALYSIS_ROLE)
            self.assertEqual(contract.expected_constant_overlap_if_fully_finite, duration - 120)

        mid = lag_spec_for_duration(MID_WINDOW_DURATION_S)
        self.assertEqual(expected_lag_count(MID_WINDOW_DURATION_S), MID_WINDOW_N_LAGS)
        self.assertEqual(mid.lag_max_s, 30)
        self.assertEqual(mid.n_lags, 61)
        self.assertFalse(mid.is_standard_zlpi)
        self.assertFalse(mid.pool_with_standard_zlpi)
        self.assertEqual(mid.endpoint_name, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertEqual((mid.flank_inner_s, mid.flank_outer_s), MWPI_FLANKS_S)
        self.assertEqual(mid.lag_analysis_role, MID_WINDOW_ANALYSIS_ROLE)

        short = lag_spec_for_duration(SHORT_WINDOW_DURATION_S)
        self.assertEqual(expected_lag_count(SHORT_WINDOW_DURATION_S), SHORT_WINDOW_N_LAGS)
        self.assertEqual(short.lag_max_s, 20)
        self.assertEqual(short.n_lags, 41)
        self.assertFalse(short.is_standard_zlpi)
        self.assertFalse(short.pool_with_standard_zlpi)
        self.assertEqual(short.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
        self.assertEqual((short.flank_inner_s, short.flank_outer_s), SWPI_FLANKS_S)
        self.assertEqual(short.lag_analysis_role, SHORT_WINDOW_ANALYSIS_ROLE)

    def test_standard_zlpi_durations_have_constant_overlap(self) -> None:
        for duration, expected_overlap in ((240, 120), (180, 60)):
            hr = _noise(duration, seed=duration)
            bands = {band: hr.copy() for band in BAND_ORDER}
            result = compute_signed_lag_curves(
                _aligned_rows(hr, bands, duration_s=duration),
                duration_s=duration,
            )
            self.assertEqual(len(result.lag_grid_s), 121)
            self.assertEqual(result.lag_grid_s[0], -60.0)
            self.assertEqual(result.lag_grid_s[-1], 60.0)
            self.assertTrue(result.lag_spec.is_standard_zlpi)
            curve = _curve_map(result)
            _assert_constant_overlap(curve, expected_overlap)
            for qc in result.qc_rows:
                self.assertTrue(qc["overlap_is_constant"])
                self.assertEqual(qc["n_common_support"], expected_overlap)
                self.assertEqual(qc["endpoint_name"], ENDPOINT_ZLPI)
                self.assertTrue(qc["is_standard_zlpi"])
                self.assertTrue(qc["pool_with_standard_zlpi"])

    def test_d120_mwpi_61_lags_constant_overlap_not_pooled(self) -> None:
        n = 120
        hr_z = _zscore(_noise(n, seed=120))
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(
                hr_z,
                {band: hr_z.copy() for band in BAND_ORDER},
                duration_s=120,
                duration_role="nested_sensitivity",
            ),
            duration_s=120,
        )
        self.assertEqual(len(result.lag_grid_s), 61)
        self.assertEqual(result.lag_grid_s[0], -30.0)
        self.assertEqual(result.lag_grid_s[-1], 30.0)
        self.assertEqual(len(result.curve_rows), 4 * 3 * 61)
        self.assertFalse(result.lag_spec.is_standard_zlpi)
        self.assertFalse(result.lag_spec.pool_with_standard_zlpi)
        self.assertEqual(
            result.lag_spec.endpoint_name, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX
        )
        curve = _curve_map(result)
        _assert_constant_overlap(curve, 60)
        for row in result.curve_rows:
            self.assertEqual(row["endpoint_name"], ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
            self.assertFalse(row["is_standard_zlpi"])
            self.assertFalse(row["pool_with_standard_zlpi"])
            self.assertEqual(row["lag_analysis_role"], MID_WINDOW_ANALYSIS_ROLE)

    def test_d60_swpi_41_lags_not_zlpi(self) -> None:
        n = 60
        hr_z = _zscore(_noise(n, seed=60))
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(
                hr_z,
                {band: hr_z.copy() for band in BAND_ORDER},
                duration_s=60,
                duration_role="separate_sensitivity",
            ),
            duration_s=60,
        )
        self.assertEqual(len(result.lag_grid_s), 41)
        self.assertEqual(result.lag_grid_s[0], -20.0)
        self.assertEqual(result.lag_grid_s[-1], 20.0)
        self.assertFalse(result.lag_spec.is_standard_zlpi)
        self.assertEqual(
            result.lag_spec.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX
        )
        _assert_constant_overlap(_curve_map(result), 20)

    def test_zero_lag_coupling(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=2))
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, {band: hr_z.copy() for band in BAND_ORDER}),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertAlmostEqual(curve[0.0][0], 1.0, places=10)
        self.assertEqual(_peak_lag(curve), 0.0)
        _assert_constant_overlap(curve, 120)

    def test_known_positive_lag_eeg_follows_hr(self) -> None:
        n = 240
        lag = 12
        hr_z = _zscore(_noise(n, seed=3))
        eeg_z = _shift(hr_z, lag)
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, {band: eeg_z.copy() for band in BAND_ORDER}),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertEqual(_peak_lag(curve), float(lag))
        self.assertGreater(curve[float(lag)][0], 0.99)

    def test_known_negative_lag_eeg_precedes_hr(self) -> None:
        n = 240
        lag = 9
        hr_z = _zscore(_noise(n, seed=4))
        eeg_z = _shift(hr_z, -lag)
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, {band: eeg_z.copy() for band in BAND_ORDER}),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertEqual(_peak_lag(curve), float(-lag))
        self.assertGreater(curve[float(-lag)][0], 0.99)

    def test_lag_sign_convention_documented(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=5))
        follow_curve = _curve_map(
            compute_signed_lag_curves(
                _aligned_rows_from_z(
                    hr_z, {band: _shift(hr_z, 5).copy() for band in BAND_ORDER}
                ),
                duration_s=240,
            )
        )
        precede_curve = _curve_map(
            compute_signed_lag_curves(
                _aligned_rows_from_z(
                    hr_z, {band: _shift(hr_z, -5).copy() for band in BAND_ORDER}
                ),
                duration_s=240,
            )
        )
        self.assertGreater(follow_curve[5.0][0], follow_curve[-5.0][0])
        self.assertGreater(precede_curve[-5.0][0], precede_curve[5.0][0])

    def test_constant_signals_yield_nan_correlation(self) -> None:
        n = 180
        result = compute_signed_lag_curves(
            _aligned_rows(
                np.full(n, 70.0),
                {band: np.full(n, 1.0) for band in BAND_ORDER},
                duration_s=180,
            ),
            duration_s=180,
        )
        curve = _curve_map(result)
        _assert_constant_overlap(curve, 60)
        for lag_s, (r, n_overlap) in curve.items():
            self.assertTrue(math.isnan(r), msg=f"lag={lag_s}")
            self.assertEqual(n_overlap, 60)

    def test_missing_values_keep_constant_overlap_on_common_support(self) -> None:
        n = 240
        hr = _noise(n, seed=6)
        rows = _aligned_rows(hr, {band: hr.copy() for band in BAND_ORDER})
        for index in range(100, 110):
            rows[index]["hr_z"] = float("nan")
        for index in range(150, 160):
            rows[index]["theta_absolute_log10_power_z"] = float("nan")
        result = compute_signed_lag_curves(rows, duration_s=240)
        curve = _curve_map(result)
        overlaps = {n_overlap for _, n_overlap in curve.values()}
        self.assertEqual(len(overlaps), 1)
        self.assertLess(next(iter(overlaps)), 120)

    def test_bands_and_representations_are_all_processed(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=7))
        bands = {
            "theta": hr_z.copy(),
            "alpha": _shift(hr_z, 3),
            "beta": _shift(hr_z, -4),
            "low_gamma": _shift(hr_z, 7),
        }
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, bands),
            duration_s=240,
        )
        pairs = {(row["band"], row["power_representation"]) for row in result.qc_rows}
        expected = {
            (band, representation)
            for band in BAND_ORDER
            for representation, _, _ in POWER_REPRESENTATIONS
        }
        self.assertEqual(pairs, expected)
        self.assertEqual(_peak_lag(_curve_map(result, band="alpha")), 3.0)
        self.assertEqual(_peak_lag(_curve_map(result, band="beta")), -4.0)

    def test_does_not_select_largest_absolute_peak(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=8))
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, {band: (-hr_z).copy() for band in BAND_ORDER}),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertAlmostEqual(curve[0.0][0], -1.0, places=10)
        self.assertNotIn("peak_lag_s", result.curve_rows[0])

    def test_deterministic_outputs(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=9))
        rows = _aligned_rows_from_z(hr_z, {band: _shift(hr_z, 2) for band in BAND_ORDER})
        first = compute_signed_lag_curves(rows, duration_s=240)
        second = compute_signed_lag_curves(rows, duration_s=240)
        self.assertEqual(first.curve_rows, second.curve_rows)
        self.assertEqual(first.qc_rows, second.qc_rows)
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = write_correlation_outputs(first, out)
            with paths["curves"].open(encoding="utf-8", newline="") as handle:
                curve_csv = list(csv.DictReader(handle))
            with paths["qc"].open(encoding="utf-8", newline="") as handle:
                qc_csv = list(csv.DictReader(handle))
            self.assertEqual(len(curve_csv), 4 * 3 * 121)
            self.assertIn("endpoint_name", curve_csv[0])
            self.assertIn("is_standard_zlpi", curve_csv[0])
            self.assertIn("pool_with_standard_zlpi", curve_csv[0])
            self.assertIn("n_common_support", qc_csv[0])
            self.assertEqual(paths["qc"].name, QC_TEMPLATE.format(duration_s=240))
            self.assertEqual(paths["curves"].name, CURVES_TEMPLATE.format(duration_s=240))


if __name__ == "__main__":
    unittest.main()
