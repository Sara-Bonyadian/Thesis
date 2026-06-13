from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.config import load_config
from ppg_eeg.temporal_coupling.events import (
    _build_epoch_times,
    _build_event_qc,
    _compute_delta_hr_z,
    _filter_by_min_distance,
    _identify_band_events,
    _identify_hr_events,
    _process_subject,
    _valid_event_indices,
)


def _synthetic_aligned_df(
    *,
    duration_s: int = 200,
    hr_step: float = 0.0,
    theta_level: float = 0.0,
) -> pd.DataFrame:
    time_s = np.arange(30, 30 + duration_s, dtype=float)
    hr_z = np.linspace(0.0, hr_step * len(time_s), len(time_s))
    theta_env_z = np.full(len(time_s), theta_level, dtype=float)
    return pd.DataFrame(
        {
            "dataset_id": "ds003838",
            "subject_id": "sub-test",
            "task": "rest",
            "observation_id": "ds003838-sub-test-task-rest",
            "time_s": time_s,
            "hr": hr_z,
            "rmssd": np.zeros(len(time_s)),
            "sdnn": np.zeros(len(time_s)),
            "mean_rr": np.zeros(len(time_s)),
            "theta_env": np.zeros(len(time_s)),
            "alpha_env": np.zeros(len(time_s)),
            "beta_env": np.zeros(len(time_s)),
            "hr_z": hr_z,
            "rmssd_z": np.zeros(len(time_s)),
            "sdnn_z": np.zeros(len(time_s)),
            "mean_rr_z": np.zeros(len(time_s)),
            "theta_env_z": theta_env_z,
            "alpha_env_z": np.zeros(len(time_s)),
            "beta_env_z": np.zeros(len(time_s)),
        }
    )


class TestEvents(unittest.TestCase):
    def test_epoch_window_skips_recording_edges(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        events_cfg = cfg.temporal_coupling.events
        time_s = np.arange(30, 230, dtype=float)
        valid = _valid_event_indices(
            time_s,
            epoch_pre_s=events_cfg.epoch_pre_s,
            epoch_post_s=events_cfg.epoch_post_s,
            hr_delta_window_s=events_cfg.hr_delta_window_s,
        )
        self.assertGreater(len(valid), 0)
        self.assertGreater(valid[0], 0)
        self.assertLess(valid[-1], len(time_s) - 1)
        epoch_times = _build_epoch_times(events_cfg, cfg.temporal_coupling.resample.fs_hz)
        self.assertEqual(epoch_times[0], -60.0)
        self.assertEqual(epoch_times[-1], 60.0)

    def test_hr_increase_and_decrease_detection(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        events_cfg = cfg.temporal_coupling.events
        time_s = np.arange(30, 230, dtype=float)
        hr_z = np.zeros(len(time_s))
        hr_z[120:140] = 3.0
        hr_z[160:180] = -3.0
        delta = _compute_delta_hr_z(time_s, hr_z, events_cfg.hr_delta_window_s)
        valid = _valid_event_indices(
            time_s,
            epoch_pre_s=events_cfg.epoch_pre_s,
            epoch_post_s=events_cfg.epoch_post_s,
            hr_delta_window_s=events_cfg.hr_delta_window_s,
        )
        increase_idx, decrease_idx = _identify_hr_events(delta, valid, events_cfg)
        self.assertGreater(len(increase_idx), 0)
        self.assertGreater(len(decrease_idx), 0)

    def test_fixed_z_burst_requires_minimum_duration(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        events_cfg = cfg.temporal_coupling.events
        time_s = np.arange(30, 230, dtype=float)
        theta = np.zeros(len(time_s))
        theta[100:101] = 3.0
        theta[120:125] = 3.0
        valid = _valid_event_indices(
            time_s,
            epoch_pre_s=events_cfg.epoch_pre_s,
            epoch_post_s=events_cfg.epoch_post_s,
            hr_delta_window_s=events_cfg.hr_delta_window_s,
        )
        min_samples = int(round(events_cfg.min_event_duration_s * cfg.temporal_coupling.resample.fs_hz))
        bursts = _identify_band_events(
            theta,
            valid,
            time_s,
            events_cfg,
            method="fixed_z",
            percentile=90.0,
            tail="high",
            min_samples=min_samples,
        )
        self.assertEqual(len(bursts), 1)
        self.assertEqual(int(bursts[0]), 120)

    def test_min_event_distance_filters_nearby_events(self) -> None:
        time_s = np.arange(0, 100, dtype=float)
        scores = np.zeros(len(time_s))
        indices = np.array([10, 15, 40, 55], dtype=int)
        filtered = _filter_by_min_distance(indices, time_s, min_distance_s=10.0, scores=scores)
        self.assertEqual(len(filtered), 3)
        self.assertNotIn(15, filtered)

    def test_event_qc_flags_exploratory_counts(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        counts_df = pd.DataFrame(
            [
                {
                    "subject_id": "sub-001",
                    "hr_increase": 2,
                    "hr_decrease": 2,
                    "theta_burst": 1,
                    "alpha_suppression": 0,
                    "beta_burst": 0,
                    "warning": "",
                },
                {
                    "subject_id": "sub-002",
                    "hr_increase": 3,
                    "hr_decrease": 3,
                    "theta_burst": 0,
                    "alpha_suppression": 0,
                    "beta_burst": 1,
                    "warning": "",
                },
            ]
        )
        qc_rows = _build_event_qc(counts_df, cfg.temporal_coupling.events)
        hr_qc = next(row for row in qc_rows if row.event_type == "hr_increase")
        alpha_qc = next(row for row in qc_rows if row.event_type == "alpha_suppression")
        self.assertFalse(hr_qc.usable_for_group_plot)
        self.assertIn("exploratory", hr_qc.warning)
        self.assertEqual(alpha_qc.total_events, 0)
        self.assertIn("zero_events", alpha_qc.warning)

    def test_event_qc_includes_averaging_metadata(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        counts_df = pd.DataFrame(
            [
                {
                    "subject_id": "sub-001",
                    "hr_increase": 2,
                    "hr_decrease": 2,
                    "theta_burst": 1,
                    "alpha_suppression": 0,
                    "beta_burst": 0,
                    "warning": "",
                },
            ]
        )
        qc_rows = _build_event_qc(counts_df, cfg.temporal_coupling.events)
        hr_qc = next(row for row in qc_rows if row.event_type == "hr_increase")
        self.assertEqual(hr_qc.averaging_method, "subject_mean_then_group_mean")
        self.assertTrue(hr_qc.subject_balanced_average)
        self.assertEqual(hr_qc.min_total_events_required, 10)
        self.assertEqual(hr_qc.min_subjects_required, 4)

    def test_process_subject_reports_counts(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        aligned_df = _synthetic_aligned_df(duration_s=200, hr_step=0.05, theta_level=3.0)
        epoch_times = _build_epoch_times(
            cfg.temporal_coupling.events,
            cfg.temporal_coupling.resample.fs_hz,
        )
        result = _process_subject(aligned_df, cfg, epoch_times)
        self.assertGreater(result.counts.hr_increase, 0)
        self.assertGreater(result.counts.theta_burst, 0)


if __name__ == "__main__":
    unittest.main()
