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
    POWER_REPRESENTATIONS,
    PRIMARY_POWER_REPRESENTATION,
    QC_TEMPLATE,
    compute_signed_lag_curves,
    expected_lag_count,
    write_correlation_outputs,
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
    """Causal shift with NaN padding (no circular wrap)."""
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
    """Build M4-like rows from already standardized series (used by lag tests)."""
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
    start_s: float = 0.0,
    duration_role: str = "primary",
) -> list[dict[str, object]]:
    hr_z = _zscore(hr.astype(float))
    band_z = {band: _zscore(signal.astype(float)) for band, signal in band_signals.items()}
    return _aligned_rows_from_z(
        hr_z,
        band_z,
        duration_s=duration_s,
        start_s=start_s,
        duration_role=duration_role,
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
    return max(curve.items(), key=lambda item: item[1][0] if math.isfinite(item[1][0]) else -np.inf)[0]


class TestConfirmatoryCorrelation(unittest.TestCase):
    def test_exact_lag_count_and_overlap_counts(self) -> None:
        n = 240
        hr = _noise(n, seed=1)
        bands = {band: hr.copy() for band in BAND_ORDER}
        result = compute_signed_lag_curves(
            _aligned_rows(hr, bands),
            duration_s=240,
        )
        self.assertEqual(expected_lag_count(), 121)
        self.assertEqual(len(result.lag_grid_s), 121)
        self.assertEqual(result.lag_grid_s[0], -60.0)
        self.assertEqual(result.lag_grid_s[-1], 60.0)
        # 4 bands × 3 representations × 121 lags
        self.assertEqual(len(result.curve_rows), 4 * 3 * 121)
        self.assertEqual(len(result.qc_rows), 4 * 3)

        curve = _curve_map(result)
        self.assertEqual(curve[0.0][1], 240)
        self.assertEqual(curve[10.0][1], 230)
        self.assertEqual(curve[-10.0][1], 230)
        self.assertEqual(curve[60.0][1], 180)
        self.assertEqual(curve[-60.0][1], 180)

    def test_zero_lag_coupling(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=2))
        bands = {band: hr_z.copy() for band in BAND_ORDER}
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, bands),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertAlmostEqual(curve[0.0][0], 1.0, places=10)
        self.assertEqual(_peak_lag(curve), 0.0)

    def test_known_positive_lag_eeg_follows_hr(self) -> None:
        n = 240
        lag = 12
        hr_z = _zscore(_noise(n, seed=3))
        eeg_z = _shift(hr_z, lag)  # EEG[t] = HR[t - lag] ⇒ EEG follows HR
        bands = {band: eeg_z.copy() for band in BAND_ORDER}
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, bands),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertEqual(_peak_lag(curve), float(lag))
        self.assertGreater(curve[float(lag)][0], 0.99)

    def test_known_negative_lag_eeg_precedes_hr(self) -> None:
        n = 240
        lag = 9
        hr_z = _zscore(_noise(n, seed=4))
        eeg_z = _shift(hr_z, -lag)  # EEG[t] = HR[t + lag] ⇒ EEG precedes HR
        bands = {band: eeg_z.copy() for band in BAND_ORDER}
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, bands),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertEqual(_peak_lag(curve), float(-lag))
        self.assertGreater(curve[float(-lag)][0], 0.99)

    def test_lag_sign_convention_documented(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=5))
        follow = _shift(hr_z, 5)
        precede = _shift(hr_z, -5)
        follow_curve = _curve_map(
            compute_signed_lag_curves(
                _aligned_rows_from_z(
                    hr_z, {band: follow.copy() for band in BAND_ORDER}
                ),
                duration_s=240,
            )
        )
        precede_curve = _curve_map(
            compute_signed_lag_curves(
                _aligned_rows_from_z(
                    hr_z, {band: precede.copy() for band in BAND_ORDER}
                ),
                duration_s=240,
            )
        )
        # Positive lag: EEG follows HR. Negative lag: EEG precedes HR.
        self.assertGreater(follow_curve[5.0][0], follow_curve[-5.0][0])
        self.assertGreater(precede_curve[-5.0][0], precede_curve[5.0][0])

    def test_constant_signals_yield_nan_correlation(self) -> None:
        n = 180
        hr = np.full(n, 70.0)
        bands = {band: np.full(n, 1.0) for band in BAND_ORDER}
        result = compute_signed_lag_curves(
            _aligned_rows(hr, bands, duration_s=180, duration_role="nested_sensitivity"),
            duration_s=180,
        )
        curve = _curve_map(result)
        for lag_s, (r, n_overlap) in curve.items():
            self.assertTrue(math.isnan(r), msg=f"lag={lag_s}")
            self.assertGreaterEqual(n_overlap, 2)

    def test_missing_values_reduce_overlap_and_preserve_finite_support(self) -> None:
        n = 240
        hr = _noise(n, seed=6)
        eeg = hr.copy()
        rows = _aligned_rows(hr, {band: eeg.copy() for band in BAND_ORDER})
        # Punch holes in HR and EEG that are non-overlapping at zero lag.
        for index in range(10, 20):
            rows[index]["hr_z"] = float("nan")
        for index in range(40, 55):
            rows[index]["theta_absolute_log10_power_z"] = float("nan")

        result = compute_signed_lag_curves(rows, duration_s=240)
        curve = _curve_map(result)
        self.assertEqual(curve[0.0][1], 240 - 10 - 15)
        # Far positive lag should lose additional HR samples at the trail and EEG at head.
        self.assertLess(curve[5.0][1], curve[0.0][1] + 5)
        self.assertTrue(math.isfinite(curve[0.0][0]))

    def test_bands_and_representations_are_all_processed(self) -> None:
        n = 120
        hr_z = _zscore(_noise(n, seed=7))
        bands = {
            "theta": hr_z.copy(),
            "alpha": _shift(hr_z, 3),
            "beta": _shift(hr_z, -4),
            "low_gamma": _shift(hr_z, 7),
        }
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(
                hr_z, bands, duration_s=120, duration_role="nested_sensitivity"
            ),
            duration_s=120,
        )
        pairs = {(row["band"], row["power_representation"]) for row in result.qc_rows}
        expected = {
            (band, representation)
            for band in BAND_ORDER
            for representation, _, _ in POWER_REPRESENTATIONS
        }
        self.assertEqual(pairs, expected)
        primary = [
            row
            for row in result.qc_rows
            if row["power_representation"] == PRIMARY_POWER_REPRESENTATION
        ]
        self.assertTrue(all(row["is_primary_representation"] is True for row in primary))
        sensitivity = [
            row
            for row in result.qc_rows
            if row["power_representation"] != PRIMARY_POWER_REPRESENTATION
        ]
        self.assertTrue(all(row["is_primary_representation"] is False for row in sensitivity))
        self.assertEqual(_peak_lag(_curve_map(result, band="alpha")), 3.0)
        self.assertEqual(_peak_lag(_curve_map(result, band="beta")), -4.0)

    def test_does_not_select_largest_absolute_peak(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=8))
        # Strong negative correlation at lag 0, weaker positive elsewhere stored as-is.
        eeg_z = -hr_z
        result = compute_signed_lag_curves(
            _aligned_rows_from_z(hr_z, {band: eeg_z.copy() for band in BAND_ORDER}),
            duration_s=240,
        )
        curve = _curve_map(result)
        self.assertAlmostEqual(curve[0.0][0], -1.0, places=10)
        signed_values = [r for r, _ in curve.values() if math.isfinite(r)]
        self.assertTrue(any(value < 0 for value in signed_values))
        # No peak columns are emitted on curve or QC rows.
        self.assertNotIn("peak_lag_s", result.curve_rows[0])
        self.assertNotIn("peak_signed_r", result.qc_rows[0])

    def test_deterministic_outputs(self) -> None:
        n = 240
        hr_z = _zscore(_noise(n, seed=9))
        bands = {band: _shift(hr_z, 2) for band in BAND_ORDER}
        rows = _aligned_rows_from_z(hr_z, bands)
        first = compute_signed_lag_curves(rows, duration_s=240)
        second = compute_signed_lag_curves(rows, duration_s=240)
        self.assertEqual(first.curve_rows, second.curve_rows)
        self.assertEqual(first.qc_rows, second.qc_rows)

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths_a = write_correlation_outputs(first, out)
            bytes_a = {
                key: path.read_bytes() for key, path in paths_a.items()
            }
            paths_b = write_correlation_outputs(second, out)
            for key, path in paths_b.items():
                self.assertEqual(path.read_bytes(), bytes_a[key])

            curves_path = (out / CURVES_TEMPLATE.format(duration_s=240)).resolve()
            qc_path = (out / QC_TEMPLATE.format(duration_s=240)).resolve()
            self.assertEqual(paths_a["curves"], curves_path)
            self.assertEqual(paths_a["qc"], qc_path)

            with curves_path.open(encoding="utf-8", newline="") as handle:
                curve_csv = list(csv.DictReader(handle))
            with qc_path.open(encoding="utf-8", newline="") as handle:
                qc_csv = list(csv.DictReader(handle))
            self.assertEqual(len(curve_csv), 4 * 3 * 121)
            self.assertEqual(len(qc_csv), 4 * 3)
            self.assertIn("n_overlap", curve_csv[0])
            self.assertIn("r_at_zero", qc_csv[0])
            self.assertIn("min_n_overlap", qc_csv[0])


if __name__ == "__main__":
    unittest.main()
