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
    _resolve_plot_lag_plan,
    apply_group_permutation_tests,
    build_group_interpretation_summary,
    build_mean_curves,
    build_peak_correlation_summary,
    run_stage3,
    summary_output_path,
)
from ppg_eeg.temporal_coupling.cross_correlation import (
    SubjectPairData,
    group_permutation_p_value,
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
        self.assertIn("group_perm_stat", peak_rows[0])
        self.assertIn("sig_group_perm_fdr", peak_rows[0])

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

    def test_common_lag_bounds_ignore_non_finite_r(self) -> None:
        curve_rows: list[dict[str, object]] = []
        for subject_idx in range(3):
            subject_id = f"sub-{subject_idx:03d}"
            for lag_s in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
                r = float("nan") if subject_idx == 0 and abs(lag_s) > 1 else 0.1
                curve_rows.append(
                    {
                        "subject_id": subject_id,
                        "pair": "hr__theta",
                        "lag_s": lag_s,
                        "r": r,
                    }
                )
        curves_df = pd.DataFrame(curve_rows)
        mean_curves = build_mean_curves(curves_df)
        common = mean_curves.loc[mean_curves["in_common_lag_range"], "lag_s"].astype(float)
        self.assertEqual(common.min(), -1.0)
        self.assertEqual(common.max(), 1.0)
        bounds = _global_common_lag_bounds(curves_df)
        self.assertEqual(bounds, (-1.0, 1.0))

    def test_resolve_plot_lag_plan_uses_fractional_when_strict_common_narrow(self) -> None:
        curve_rows: list[dict[str, object]] = []
        for subject_idx in range(10):
            subject_id = f"sub-{subject_idx:03d}"
            for lag_s in range(-10, 11):
                r = float("nan") if subject_idx == 0 and abs(lag_s) > 1 else 0.1
                curve_rows.append(
                    {
                        "subject_id": subject_id,
                        "pair": "hr__theta",
                        "lag_s": float(lag_s),
                        "r": r,
                    }
                )
        mean_curves = build_mean_curves(pd.DataFrame(curve_rows))
        plan = _resolve_plot_lag_plan(
            plot_common_lag_only=True,
            mean_curves_df=mean_curves,
            strict_common_bounds=(-1.0, 1.0),
        )
        self.assertEqual(plan.filter_mode, "fractional")
        self.assertFalse(plan.shade_partial_n)
        self.assertIsNotNone(plan.display_bounds)
        assert plan.display_bounds is not None
        self.assertGreaterEqual(plan.display_bounds[1] - plan.display_bounds[0], 5.0)

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
                temporal_coupling=replace(
                    cfg.temporal_coupling,
                    group=replace(cfg.temporal_coupling.group, n_group_permutations=0),
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
            self.assertTrue((group_dir / "stage3_interpretation_notes.txt").is_file())
            notes = (group_dir / "stage3_interpretation_notes.txt").read_text(encoding="utf-8")
            self.assertIn("Statistical inference layers", notes)
            self.assertIn("permutation-controlled group peak strength", notes)

    def test_group_permutation_stronger_than_null(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        cfg = replace(
            cfg,
            temporal_coupling=replace(
                cfg.temporal_coupling,
                group=replace(cfg.temporal_coupling.group, n_group_permutations=200),
            ),
        )
        n = 150
        lag_grid_s = np.arange(-20.0, 21.0, 5.0)
        rng = np.random.default_rng(123)

        subject_data: list[SubjectPairData] = []
        observed_peaks: list[float] = []
        for idx in range(5):
            cardiac = rng.standard_normal(n)
            eeg = cardiac.copy()
            subject_data.append(
                SubjectPairData(
                    subject_id=f"sub-{idx:03d}",
                    cardiac=cardiac,
                    eeg=eeg,
                    lag_grid_s=lag_grid_s,
                )
            )
            from ppg_eeg.temporal_coupling.cross_correlation import compute_correlation_curve, extract_peaks

            curve = compute_correlation_curve(
                cardiac,
                eeg,
                lag_grid_s=lag_grid_s,
                fs_hz=cfg.temporal_coupling.resample.fs_hz,
            )
            peak = extract_peaks(
                curve,
                lag_step_s=cfg.temporal_coupling.cross_correlation.lag_step_s,
                lag_grid_s=lag_grid_s,
                edge_margin_s=cfg.temporal_coupling.cross_correlation.edge_margin_s,
                min_peak_distance_s=cfg.temporal_coupling.cross_correlation.min_peak_distance_s,
                peak_prominence=cfg.temporal_coupling.cross_correlation.peak_prominence,
                peak_height=cfg.temporal_coupling.cross_correlation.peak_height,
            )
            observed_peaks.append(float(peak.raw_peak_abs_r))

        observed = float(np.median(observed_peaks))
        self.assertGreater(observed, 0.9)

        p_value = group_permutation_p_value(
            subject_data,
            observed_group_stat=observed,
            n_group_permutations=200,
            cfg=cfg,
            rng=np.random.default_rng(0),
        )
        self.assertIsNotNone(p_value)
        assert p_value is not None
        self.assertLess(p_value, 0.05)

    def test_apply_group_permutation_adds_columns(self) -> None:
        cfg = load_config("config.smoke.ds003838.temporal_coupling.yaml")
        cfg = replace(
            cfg,
            temporal_coupling=replace(
                cfg.temporal_coupling,
                group=replace(cfg.temporal_coupling.group, n_group_permutations=0),
            ),
        )
        peaks_df = _synthetic_peaks(n_subjects=3)
        peak_rows = build_peak_correlation_summary(peaks_df, cfg)
        enriched = apply_group_permutation_tests(peak_rows, peaks_df, cfg)
        self.assertEqual(len(enriched), len(variable_pairs()))
        self.assertTrue(all("no_group_permutation_test" in str(row["warning"]) for row in enriched))
        self.assertTrue(all(not row["sig_group_perm_fdr"] for row in enriched))
