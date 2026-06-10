from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.config import load_config
from ppg_eeg.temporal_coupling.eeg_envelope import UsableObservation
from ppg_eeg.temporal_coupling.resample import (
    align_observation,
    build_alignment_qc,
    build_time_grid,
    interp_cardiac_limited_gap,
    overlap_time_range,
    zscore_within_observation,
)


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    times_eeg = np.arange(0.0, 100.1, 0.1)
    eeg_df = pd.DataFrame(
        {
            "dataset_id": "ds-test",
            "subject_id": "sub-001",
            "task": "rest",
            "observation_id": "ds-test-sub-001-task-rest",
            "time_s": times_eeg,
            "theta_env": np.sin(2 * np.pi * 0.05 * times_eeg) + 1.0,
            "alpha_env": np.cos(2 * np.pi * 0.03 * times_eeg) + 1.0,
            "beta_env": np.sin(2 * np.pi * 0.08 * times_eeg) + 1.0,
        }
    )
    times_car = np.arange(20.0, 80.1, 2.0)
    cardiac_df = pd.DataFrame(
        {
            "dataset_id": "ds-test",
            "subject_id": "sub-001",
            "task": "rest",
            "observation_id": "ds-test-sub-001-task-rest",
            "time_s": times_car,
            "hr": 70.0 + 5.0 * np.sin(2 * np.pi * 0.02 * times_car),
            "rmssd": 50.0 + 10.0 * np.cos(2 * np.pi * 0.02 * times_car),
            "sdnn": 45.0 + 8.0 * np.sin(2 * np.pi * 0.015 * times_car),
            "mean_rr": 850.0 + 40.0 * np.cos(2 * np.pi * 0.02 * times_car),
        }
    )
    return eeg_df, cardiac_df


class TestResample(unittest.TestCase):
    def test_overlap_time_range(self) -> None:
        eeg_df, cardiac_df = _synthetic_frames()
        overlap = overlap_time_range(eeg_df, cardiac_df)
        self.assertAlmostEqual(overlap.start_s, 20.0)
        self.assertAlmostEqual(overlap.end_s, 80.0)

    def test_build_time_grid_1hz(self) -> None:
        grid = build_time_grid(20.0, 24.0, 1.0)
        np.testing.assert_allclose(grid, np.array([20.0, 21.0, 22.0, 23.0, 24.0]))

    def test_zscore_mean_and_std(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore_within_observation(values)
        self.assertAlmostEqual(float(np.nanmean(z)), 0.0, places=6)
        self.assertAlmostEqual(float(np.nanstd(z)), 1.0, places=6)

    def test_interp_cardiac_does_not_fill_long_gap(self) -> None:
        times = np.array([0.0, 1.0, 12.0, 13.0])
        values = np.array([10.0, 11.0, 20.0, 21.0])
        grid = np.arange(0.0, 14.0, 1.0)
        out = interp_cardiac_limited_gap(times, values, grid, max_gap_s=5.0, fs_hz=1.0)
        self.assertTrue(np.isfinite(out[0]))
        self.assertTrue(np.isfinite(out[1]))
        self.assertTrue(np.isnan(out[6]))
        self.assertTrue(np.isnan(out[7]))
        self.assertTrue(np.isfinite(out[12]))

    def test_align_observation_zscore_and_overlap(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        eeg_df, cardiac_df = _synthetic_frames()
        aligned = align_observation(eeg_df, cardiac_df, cfg)

        self.assertEqual(len(aligned), 61)
        self.assertAlmostEqual(float(aligned["time_s"].iloc[0]), 20.0)
        self.assertAlmostEqual(float(aligned["time_s"].iloc[-1]), 80.0)

        for z_col in (
            "hr_z",
            "rmssd_z",
            "sdnn_z",
            "mean_rr_z",
            "theta_env_z",
            "alpha_env_z",
            "beta_env_z",
        ):
            z = aligned[z_col].to_numpy(dtype=float)
            finite = z[np.isfinite(z)]
            self.assertGreater(finite.size, 10)
            self.assertAlmostEqual(float(np.mean(finite)), 0.0, places=5)
            self.assertAlmostEqual(float(np.std(finite, ddof=0)), 1.0, places=5)

    def test_build_alignment_qc(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        eeg_df, _cardiac_df = _synthetic_frames()
        times_car = np.arange(20.0, 150.1, 2.0)
        cardiac_df = pd.DataFrame(
            {
                "dataset_id": _cardiac_df["dataset_id"].iloc[0],
                "subject_id": _cardiac_df["subject_id"].iloc[0],
                "task": _cardiac_df["task"].iloc[0],
                "observation_id": _cardiac_df["observation_id"].iloc[0],
                "time_s": times_car,
                "hr": 70.0 + 5.0 * np.sin(2 * np.pi * 0.02 * times_car),
                "rmssd": 50.0 + 10.0 * np.cos(2 * np.pi * 0.02 * times_car),
                "sdnn": 45.0 + 8.0 * np.sin(2 * np.pi * 0.015 * times_car),
                "mean_rr": 850.0 + 40.0 * np.cos(2 * np.pi * 0.02 * times_car),
            }
        )
        aligned = align_observation(eeg_df, cardiac_df, cfg)
        overlap = overlap_time_range(eeg_df, cardiac_df)
        obs = UsableObservation(
            dataset_id="ds-test",
            subject_id="sub-001",
            task="rest",
            observation_id="ds-test-sub-001-task-rest",
            eeg_file=Path("dummy.set"),
        )
        qc = build_alignment_qc(
            obs,
            aligned,
            overlap,
            cfg,
            upstream_usable_for_hr=True,
            upstream_usable_eeg_envelope=True,
        )

        self.assertEqual(qc.subject_id, "sub-001")
        self.assertEqual(qc.observation_id, "ds-test-sub-001-task-rest")
        self.assertAlmostEqual(qc.aligned_start_s, 20.0)
        self.assertAlmostEqual(qc.aligned_end_s, 100.0)
        self.assertAlmostEqual(qc.aligned_duration_s, 80.0)
        self.assertEqual(qc.n_rows, 81)
        self.assertAlmostEqual(qc.fs_hz, 1.0)
        self.assertEqual(qc.missing_percent_hr, 0.0)
        self.assertEqual(qc.missing_percent_theta, 0.0)
        self.assertGreater(qc.recommended_xcorr_lag_s, 0.0)
        self.assertFalse(qc.usable_for_xcorr)
        self.assertIn("short_aligned_duration", qc.warning)

        row = qc.to_row()
        self.assertEqual(
            list(row.keys()),
            [
                "subject_id",
                "observation_id",
                "aligned_start_s",
                "aligned_end_s",
                "aligned_duration_s",
                "n_rows",
                "fs_hz",
                "missing_percent_hr",
                "missing_percent_rmssd",
                "missing_percent_sdnn",
                "missing_percent_theta",
                "missing_percent_alpha",
                "missing_percent_beta",
                "recommended_xcorr_lag_s",
                "usable_for_xcorr",
                "warning",
            ],
        )

    def test_build_alignment_qc_usable_when_overlap_long_enough(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        times_eeg = np.arange(0.0, 200.1, 0.1)
        eeg_df = pd.DataFrame(
            {
                "dataset_id": "ds-test",
                "subject_id": "sub-001",
                "task": "rest",
                "observation_id": "ds-test-sub-001-task-rest",
                "time_s": times_eeg,
                "theta_env": np.sin(2 * np.pi * 0.05 * times_eeg) + 1.0,
                "alpha_env": np.cos(2 * np.pi * 0.03 * times_eeg) + 1.0,
                "beta_env": np.sin(2 * np.pi * 0.08 * times_eeg) + 1.0,
            }
        )
        times_car = np.arange(20.0, 180.1, 2.0)
        cardiac_df = pd.DataFrame(
            {
                "dataset_id": "ds-test",
                "subject_id": "sub-001",
                "task": "rest",
                "observation_id": "ds-test-sub-001-task-rest",
                "time_s": times_car,
                "hr": 70.0 + 5.0 * np.sin(2 * np.pi * 0.02 * times_car),
                "rmssd": 50.0 + 10.0 * np.cos(2 * np.pi * 0.02 * times_car),
                "sdnn": 45.0 + 8.0 * np.sin(2 * np.pi * 0.015 * times_car),
                "mean_rr": 850.0 + 40.0 * np.cos(2 * np.pi * 0.02 * times_car),
            }
        )
        aligned = align_observation(eeg_df, cardiac_df, cfg)
        overlap = overlap_time_range(eeg_df, cardiac_df)
        obs = UsableObservation(
            dataset_id="ds-test",
            subject_id="sub-001",
            task="rest",
            observation_id="ds-test-sub-001-task-rest",
            eeg_file=Path("dummy.set"),
        )
        qc = build_alignment_qc(
            obs,
            aligned,
            overlap,
            cfg,
            upstream_usable_for_hr=True,
            upstream_usable_eeg_envelope=True,
        )
        self.assertGreaterEqual(qc.aligned_duration_s, cfg.temporal_coupling.audit.min_overlap_s)
        self.assertTrue(qc.usable_for_xcorr)
        self.assertEqual(qc.warning, "")


if __name__ == "__main__":
    unittest.main()
