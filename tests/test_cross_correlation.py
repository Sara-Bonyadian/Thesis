from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.config import load_config
from ppg_eeg.temporal_coupling.cross_correlation import (
    LagCorrelationPoint,
    QC_SUMMARY_COLUMNS,
    build_lag_grid,
    build_qc_summary,
    compute_correlation_curve,
    compute_observation_xcorr,
    correlate_at_lag,
    extract_peaks,
    validate_peaks_against_curves,
    variable_pairs,
)


class TestCrossCorrelation(unittest.TestCase):
    def test_build_lag_grid_smoke(self) -> None:
        grid = build_lag_grid(lag_max_s=60.0, lag_step_s=5.0)
        self.assertEqual(grid[0], -60.0)
        self.assertEqual(grid[-1], 60.0)
        self.assertEqual(len(grid), 25)

    def test_raw_peak_uses_scipy_find_peaks_on_positive_r(self) -> None:
        curve = [
            LagCorrelationPoint(lag_s=-30.0, r=0.2, n_overlap=20),
            LagCorrelationPoint(lag_s=-20.0, r=0.9, n_overlap=20),
            LagCorrelationPoint(lag_s=-10.0, r=0.3, n_overlap=20),
            LagCorrelationPoint(lag_s=0.0, r=0.1, n_overlap=20),
            LagCorrelationPoint(lag_s=10.0, r=0.15, n_overlap=20),
            LagCorrelationPoint(lag_s=20.0, r=0.25, n_overlap=20),
            LagCorrelationPoint(lag_s=30.0, r=0.4, n_overlap=20),
        ]
        lag_grid = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
        peak = extract_peaks(
            curve,
            lag_step_s=10.0,
            lag_grid_s=lag_grid,
            edge_margin_s=0.0,
            min_peak_distance_s=10.0,
            peak_prominence=0.1,
            peak_height="auto",
        )
        self.assertAlmostEqual(peak.raw_peak_lag_s, -20.0)
        self.assertAlmostEqual(peak.raw_peak_abs_r, 0.9)

    def test_raw_peak_ignores_negative_local_maxima(self) -> None:
        curve = [
            LagCorrelationPoint(lag_s=-40.0, r=-0.46, n_overlap=87),
            LagCorrelationPoint(lag_s=-20.0, r=-0.22, n_overlap=107),
            LagCorrelationPoint(lag_s=-10.0, r=-0.04, n_overlap=117),
            LagCorrelationPoint(lag_s=0.0, r=-0.09, n_overlap=127),
            LagCorrelationPoint(lag_s=35.0, r=0.22, n_overlap=92),
            LagCorrelationPoint(lag_s=40.0, r=0.27, n_overlap=87),
        ]
        lag_grid = np.array([-40.0, -20.0, -10.0, 0.0, 35.0, 40.0])
        peak = extract_peaks(
            curve,
            lag_step_s=5.0,
            lag_grid_s=lag_grid,
            edge_margin_s=0.0,
            min_peak_distance_s=10.0,
            peak_prominence="auto",
            peak_height="auto",
        )
        self.assertGreater(peak.raw_peak_signed_r, 0.0)
        self.assertAlmostEqual(peak.raw_peak_lag_s, 40.0)
        self.assertTrue(peak.raw_peak_at_edge)

    def test_raw_peak_uses_max_positive_r_with_tie_to_zero(self) -> None:
        curve = [
            LagCorrelationPoint(lag_s=-10.0, r=0.5, n_overlap=20),
            LagCorrelationPoint(lag_s=0.0, r=-0.5, n_overlap=20),
            LagCorrelationPoint(lag_s=10.0, r=0.5, n_overlap=20),
        ]
        lag_grid = np.array([-10.0, 0.0, 10.0])
        peak = extract_peaks(curve, lag_step_s=5.0, lag_grid_s=lag_grid, edge_margin_s=0.0)
        self.assertAlmostEqual(peak.raw_peak_lag_s, -10.0)
        self.assertAlmostEqual(peak.raw_peak_signed_r, 0.5)
        self.assertAlmostEqual(peak.raw_peak_abs_r, 0.5)

    def test_preferred_peak_keeps_raw_when_only_interior_is_negative(self) -> None:
        curve = [
            LagCorrelationPoint(lag_s=-20.0, r=0.9, n_overlap=20),
            LagCorrelationPoint(lag_s=0.0, r=-0.8, n_overlap=20),
            LagCorrelationPoint(lag_s=20.0, r=-0.2, n_overlap=20),
        ]
        lag_grid = np.array([-20.0, 0.0, 20.0])
        peak = extract_peaks(curve, lag_step_s=5.0, lag_grid_s=lag_grid, edge_margin_s=5.0)
        self.assertTrue(peak.raw_peak_at_edge)
        self.assertAlmostEqual(peak.raw_peak_lag_s, -20.0)
        self.assertAlmostEqual(peak.preferred_peak_lag_s, -20.0)
        self.assertEqual(peak.preferred_peak_source, "raw_peak")
        self.assertIn("no_interior_peak", peak.warning)

    def test_preferred_peak_uses_interior_when_raw_at_edge(self) -> None:
        curve = [
            LagCorrelationPoint(lag_s=-20.0, r=0.9, n_overlap=20),
            LagCorrelationPoint(lag_s=0.0, r=0.4, n_overlap=20),
            LagCorrelationPoint(lag_s=20.0, r=0.2, n_overlap=20),
        ]
        lag_grid = np.array([-20.0, 0.0, 20.0])
        peak = extract_peaks(curve, lag_step_s=5.0, lag_grid_s=lag_grid, edge_margin_s=5.0)
        self.assertTrue(peak.raw_peak_at_edge)
        self.assertAlmostEqual(peak.raw_peak_lag_s, -20.0)
        self.assertAlmostEqual(peak.preferred_peak_lag_s, 0.0)
        self.assertEqual(peak.preferred_peak_source, "interior_peak_due_to_edge")

    def test_known_shift_detected_at_negative_lag(self) -> None:
        fs_hz = 1.0
        n = 300
        true_lag_s = -15.0
        lag_step_s = 5.0
        lag_grid_s = build_lag_grid(lag_max_s=60.0, lag_step_s=lag_step_s)

        eeg = np.sin(2 * np.pi * 0.03 * np.arange(n, dtype=float))
        cardiac = np.full(n, np.nan, dtype=float)
        shift = int(-true_lag_s)
        cardiac[shift:] = eeg[:-shift]

        curve = compute_correlation_curve(
            cardiac,
            eeg,
            lag_grid_s=lag_grid_s,
            fs_hz=fs_hz,
            min_overlap=20,
        )
        peak = extract_peaks(curve, lag_step_s=lag_step_s, lag_grid_s=lag_grid_s, edge_margin_s=5.0)

        self.assertTrue(np.isfinite(peak.raw_peak_signed_r))
        self.assertGreaterEqual(peak.raw_peak_abs_r, 0.9)
        self.assertAlmostEqual(peak.raw_peak_lag_s, true_lag_s, delta=lag_step_s)
        self.assertFalse(peak.raw_peak_at_edge)

    def test_correlate_at_lag_requires_minimum_overlap(self) -> None:
        cardiac = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eeg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        point = correlate_at_lag(cardiac, eeg, lag_s=0.0, fs_hz=1.0, min_overlap=10)
        self.assertTrue(np.isnan(point.r))
        self.assertEqual(point.n_overlap, 5)

    def test_compute_observation_xcorr_nine_pairs(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        n = 200
        times = np.arange(n, dtype=float)
        aligned_df = pd.DataFrame(
            {
                "dataset_id": "ds-test",
                "subject_id": "sub-001",
                "task": "rest",
                "observation_id": "ds-test-sub-001-task-rest",
                "time_s": times,
                "hr": np.sin(times),
                "rmssd": np.cos(times),
                "sdnn": np.sin(2 * times),
                "mean_rr": np.cos(2 * times),
                "theta_env": np.sin(0.5 * times),
                "alpha_env": np.cos(0.5 * times),
                "beta_env": np.sin(0.25 * times),
                "hr_z": np.sin(times),
                "rmssd_z": np.cos(times),
                "sdnn_z": np.sin(2 * times),
                "mean_rr_z": np.cos(2 * times),
                "theta_env_z": np.sin(0.5 * times),
                "alpha_env_z": np.cos(0.5 * times),
                "beta_env_z": np.sin(0.25 * times),
            }
        )

        curve_rows, peak_rows = compute_observation_xcorr(aligned_df, cfg, lag_max_s=60.0)
        self.assertEqual(len(variable_pairs()), 9)
        self.assertEqual(len(peak_rows), 9)
        self.assertGreater(len(curve_rows), 0)

        for peak in peak_rows:
            self.assertGreaterEqual(peak["raw_peak_abs_r"], 0.0)
            self.assertLessEqual(peak["raw_peak_abs_r"], 1.0)
            self.assertIn("preferred_peak_source", peak)
            self.assertEqual(peak["peak_lag_s"], peak["raw_peak_lag_s"])
            self.assertEqual(peak["peak_signed_r"], peak["raw_peak_signed_r"])
            self.assertEqual(peak["peak_abs_r"], peak["raw_peak_abs_r"])
            self.assertEqual(peak["peak_at_lag_edge"], peak["raw_peak_at_edge"])

        xcorr_cfg = cfg.temporal_coupling.cross_correlation
        validation_rows, passed = validate_peaks_against_curves(
            curve_rows,
            peak_rows,
            lag_step_s=xcorr_cfg.lag_step_s,
            min_peak_distance_s=xcorr_cfg.min_peak_distance_s,
            peak_prominence=xcorr_cfg.peak_prominence,
            peak_height=xcorr_cfg.peak_height,
        )
        self.assertTrue(passed)
        self.assertEqual(len(validation_rows), 9)

        if cfg.temporal_coupling.output.save_curves:
            self.assertEqual(len({row["pair"] for row in curve_rows}), 9)

    def test_build_qc_summary_warnings(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        peaks_df = pd.DataFrame(
            [
                {
                    "pair": "hr__theta",
                    "subject_id": "sub-001",
                    "raw_peak_lag_s": -40.0,
                    "raw_peak_signed_r": -0.4,
                    "raw_peak_abs_r": 0.4,
                    "preferred_peak_lag_s": -10.0,
                    "preferred_peak_signed_r": -0.2,
                    "preferred_peak_abs_r": 0.2,
                    "interior_peak_lag_s": -10.0,
                    "interior_peak_abs_r": 0.2,
                    "raw_peak_at_edge": True,
                },
                {
                    "pair": "hr__theta",
                    "subject_id": "sub-002",
                    "raw_peak_lag_s": 30.0,
                    "raw_peak_signed_r": 0.3,
                    "raw_peak_abs_r": 0.3,
                    "preferred_peak_lag_s": 30.0,
                    "preferred_peak_signed_r": 0.3,
                    "preferred_peak_abs_r": 0.3,
                    "interior_peak_lag_s": 30.0,
                    "interior_peak_abs_r": 0.3,
                    "raw_peak_at_edge": False,
                },
            ]
        )
        summary = build_qc_summary(peaks_df, cfg)
        hr_theta = next(row for row in summary if row["pair"] == "hr__theta")
        self.assertEqual(hr_theta["n_subjects"], 2)
        self.assertEqual(hr_theta["n_edge_peaks"], 1)
        self.assertEqual(hr_theta["percent_edge_peaks"], 50.0)
        self.assertIn("few_subjects", hr_theta["warning"])
        self.assertIn("mixed_peak_lag_direction", hr_theta["warning"])
        self.assertIn("no_permutation_test", hr_theta["warning"])
        self.assertEqual(list(summary[0].keys()), list(QC_SUMMARY_COLUMNS))


if __name__ == "__main__":
    unittest.main()
