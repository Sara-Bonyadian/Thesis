from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.config import load_config
from ppg_eeg.temporal_coupling.cross_correlation import PEAK_COLUMNS, variable_pairs
from ppg_eeg.temporal_coupling.group_summary import (
    MEAN_CURVES_COLUMNS,
    PEAK_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    _global_common_lag_bounds,
    build_group_interpretation_summary,
    build_mean_curves,
    build_peak_correlation_summary,
    run_stage3,
    summary_output_path,
)


def _synthetic_peaks(n_subjects: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subject_idx in range(n_subjects):
        subject_id = f"sub-{subject_idx:03d}"
        for pair_idx, pair in enumerate(variable_pairs()):
            lag = float(-20 + pair_idx * 5 + subject_idx * 2)
            signed_r = float(0.15 + 0.02 * pair_idx)
            rows.append(
                {
                    "dataset_id": "ds003838",
                    "subject_id": subject_id,
                    "task": "rest",
                    "observation_id": f"ds003838-{subject_id}-task-rest",
                    "pair": pair.pair,
                    "cardiac_var": pair.cardiac_var,
                    "eeg_var": pair.eeg_var,
                    "raw_peak_lag_s": lag,
                    "raw_peak_signed_r": signed_r,
                    "raw_peak_abs_r": abs(signed_r),
                    "raw_peak_at_edge": False,
                    "interior_peak_lag_s": lag,
                    "interior_peak_signed_r": signed_r,
                    "interior_peak_abs_r": abs(signed_r),
                    "preferred_peak_lag_s": lag,
                    "preferred_peak_signed_r": signed_r,
                    "preferred_peak_abs_r": abs(signed_r),
                    "preferred_peak_source": "raw_peak",
                    "peak_lag_s": lag,
                    "peak_signed_r": signed_r,
                    "peak_abs_r": abs(signed_r),
                    "peak_direction": "eeg_leads",
                    "peak_at_lag_edge": False,
                    "min_lag_s": -60.0,
                    "max_lag_s": 60.0,
                    "n_valid_lags": 25,
                    "n_overlap_at_raw_peak": 100,
                    "n_overlap_at_preferred_peak": 100,
                    "warning": "",
                    "p_perm": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(PEAK_COLUMNS))


class TestGroupSummary(unittest.TestCase):
    def test_build_peak_summary_counts_and_direction(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        peaks_df = _synthetic_peaks(n_subjects=3)
        peak_rows = build_peak_correlation_summary(peaks_df, cfg)
        interp_rows = build_group_interpretation_summary(
            peak_rows,
            lag_step_s=cfg.temporal_coupling.cross_correlation.lag_step_s,
        )
        self.assertEqual(len(peak_rows), len(variable_pairs()))
        self.assertEqual(list(peak_rows[0].keys())[:4], list(PEAK_SUMMARY_COLUMNS[:4]))

        hr_theta_peak = next(row for row in peak_rows if row["pair"] == "hr__theta")
        hr_theta_interp = next(row for row in interp_rows if row["pair"] == "hr__theta")
        self.assertEqual(hr_theta_peak["n_subjects"], 3)
        self.assertEqual(hr_theta_peak["n_negative_lag_peaks"], 3)
        self.assertEqual(hr_theta_peak["n_positive_lag_peaks"], 0)
        self.assertEqual(hr_theta_interp["likely_direction"], "eeg_leads")
        self.assertIn("exploratory_small_n", str(hr_theta_peak["warning"]))
        self.assertIn("no_permutation_test", str(hr_theta_peak["warning"]))

    def test_build_mean_curves_sem_and_common_lag(self) -> None:
        curve_rows: list[dict[str, object]] = []
        for subject_idx in range(3):
            subject_id = f"sub-{subject_idx:03d}"
            for pair in variable_pairs()[:1]:
                for lag_s in (-40.0, -20.0, 0.0, 20.0, 40.0):
                    if subject_idx == 0 and abs(lag_s) > 30:
                        continue
                    curve_rows.append(
                        {
                            "subject_id": subject_id,
                            "pair": pair.pair,
                            "lag_s": lag_s,
                            "r": 0.1 + 0.01 * subject_idx,
                        }
                    )
        curves_df = pd.DataFrame(curve_rows)
        mean_curves = build_mean_curves(curves_df)
        self.assertEqual(list(mean_curves.columns), list(MEAN_CURVES_COLUMNS))
        zero_row = mean_curves.loc[mean_curves["lag_s"] == 0.0].iloc[0]
        self.assertAlmostEqual(float(zero_row["mean_r"]), 0.11)
        self.assertAlmostEqual(float(zero_row["sem_r"]), 0.01 / np.sqrt(3), places=6)
        self.assertEqual(int(zero_row["n_subjects"]), 3)
        self.assertTrue(bool(zero_row["in_common_lag_range"]))
        bounds = _global_common_lag_bounds(curves_df)
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertEqual(bounds, (-20.0, 20.0))

    def test_run_stage3_writes_outputs_without_raw_data(self) -> None:
        peaks_df = _synthetic_peaks(n_subjects=3)
        curve_rows: list[dict[str, object]] = []
        for subject_idx in range(3):
            subject_id = f"sub-{subject_idx:03d}"
            for pair in variable_pairs()[:2]:
                for lag_s in np.arange(-10.0, 11.0, 5.0):
                    curve_rows.append(
                        {
                            "subject_id": subject_id,
                            "pair": pair.pair,
                            "lag_s": float(lag_s),
                            "r": 0.1,
                        }
                    )

        with TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir)
            cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
            cfg = replace(
                cfg,
                paths=replace(
                    cfg.paths,
                    raw_root=Path("/nonexistent/raw"),
                    out_root=out_root,
                ),
            )
            group_dir = out_root / cfg.dataset_id / "group"
            group_dir.mkdir(parents=True)
            peaks_df.to_csv(group_dir / "cross_correlation_peaks.csv", index=False)
            pd.DataFrame(curve_rows).to_csv(group_dir / "cross_correlation_curves.csv", index=False)

            written = run_stage3(cfg)
            summary_path = summary_output_path(cfg)
            self.assertTrue(summary_path.is_file())
            self.assertIn(summary_path, written)

            summary_df = pd.read_csv(summary_path)
            self.assertEqual(len(summary_df), len(variable_pairs()))
            self.assertEqual(list(summary_df.columns), list(SUMMARY_COLUMNS))
            self.assertTrue((group_dir / "peak_correlation_summary.csv").is_file())
            self.assertTrue((group_dir / "mean_cross_correlation_curves.csv").is_file())
            self.assertTrue((group_dir / "group_cross_correlation_mean_sem_grid.png").is_file())
            self.assertTrue((group_dir / "group_peak_summary_heatmap.png").is_file())
            self.assertTrue((group_dir / "group_edge_peak_rate_heatmap.png").is_file())
            self.assertTrue((group_dir / "peak_lag_distribution.png").is_file())
            self.assertTrue((group_dir / "peak_r_distribution.png").is_file())
            self.assertTrue((group_dir / "stage3_interpretation_notes.txt").is_file())
