"""Figure 2 Panel E: state-specific μ/FWHM (no joint complete-case μ)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.figure2_panels import (
    build_panel_e_peak_export,
    build_panel_e_state_summaries,
)
from ppg_eeg.confirmatory.figures import read_csv_rows, render_figure2
from ppg_eeg.confirmatory.group_tables import build_group_tables
from tests.test_group_tables import _ds003838_pair


def _paired_peak_row(
    *,
    participant_id: str,
    session_id: str,
    contrast_id: str,
    band: str,
    rest_id: bool,
    task_id: bool,
    rest_mu: float,
    task_mu: float,
    rest_fwhm: float = 10.0,
    task_fwhm: float = 12.0,
) -> dict[str, object]:
    return {
        "dataset_id": "hiit",
        "participant_id": participant_id,
        "session_id": session_id,
        "contrast_id": contrast_id,
        "band": band,
        "endpoint_name": "zlpi",
        "duration_s": 240,
        "power_representation": "absolute_log10",
        "contrast_eligible": True,
        "low_has_identifiable_peak": rest_id,
        "effort_has_identifiable_peak": task_id,
        "low_peak_center_mu_s": rest_mu if rest_id else float("nan"),
        "effort_peak_center_mu_s": task_mu if task_id else float("nan"),
        "low_fwhm_s": rest_fwhm if rest_id else float("nan"),
        "effort_fwhm_s": task_fwhm if task_id else float("nan"),
        "low_peak_height_A": 0.5 if rest_id else 0.01,
        "effort_peak_height_A": 0.4 if task_id else 0.01,
        "low_observation_ids": f"hiit-{participant_id}-{session_id}-rest",
        "effort_observation_ids": f"hiit-{participant_id}-{session_id}-tetris",
        "mu_contrast_eligible": rest_id and task_id,
        "delta_peak_center_mu_s": (
            (task_mu - rest_mu) if (rest_id and task_id) else float("nan")
        ),
        "delta_fwhm_s": (
            (task_fwhm - rest_fwhm) if (rest_id and task_id) else float("nan")
        ),
    }


class Figure2PanelEStateSpecificTests(unittest.TestCase):
    def test_group_tables_rest_only_keeps_rest_fields(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-001",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=True,
            memory_identifiable=False,
            rest_mu=-1.5,
            memory_mu=9.0,
        )
        row = build_group_tables(endpoints, peaks).paired_contrast_rows[0]
        self.assertFalse(row["mu_contrast_eligible"])
        self.assertTrue(math.isnan(float(row["delta_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["delta_fwhm_s"])))
        self.assertAlmostEqual(float(row["low_peak_center_mu_s"]), -1.5, places=12)
        self.assertTrue(math.isfinite(float(row["low_fwhm_s"])))
        self.assertTrue(math.isnan(float(row["effort_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["effort_fwhm_s"])))

    def test_group_tables_task_only_keeps_task_fields(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-002",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=False,
            memory_identifiable=True,
            rest_mu=-1.5,
            memory_mu=2.25,
        )
        row = build_group_tables(endpoints, peaks).paired_contrast_rows[0]
        self.assertFalse(row["mu_contrast_eligible"])
        self.assertTrue(math.isnan(float(row["low_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["low_fwhm_s"])))
        self.assertAlmostEqual(float(row["effort_peak_center_mu_s"]), 2.25, places=12)
        self.assertTrue(math.isfinite(float(row["effort_fwhm_s"])))

    def test_group_tables_both_and_neither(self) -> None:
        both_e, both_p = _ds003838_pair(
            "sub-003",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=True,
            memory_identifiable=True,
            rest_mu=1.0,
            memory_mu=3.0,
        )
        both = build_group_tables(both_e, both_p).paired_contrast_rows[0]
        self.assertTrue(both["mu_contrast_eligible"])
        self.assertAlmostEqual(float(both["delta_peak_center_mu_s"]), 2.0, places=12)
        self.assertTrue(math.isfinite(float(both["delta_fwhm_s"])))

        none_e, none_p = _ds003838_pair(
            "sub-004",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=False,
            memory_identifiable=False,
        )
        none = build_group_tables(none_e, none_p).paired_contrast_rows[0]
        self.assertFalse(none["mu_contrast_eligible"])
        for key in (
            "low_peak_center_mu_s",
            "effort_peak_center_mu_s",
            "low_fwhm_s",
            "effort_fwhm_s",
            "delta_peak_center_mu_s",
            "delta_fwhm_s",
        ):
            self.assertTrue(math.isnan(float(none[key])))

    def test_panel_e_means_use_all_state_identifiable_not_joint(self) -> None:
        rows = [
            # Joint-identifiable pair
            _paired_peak_row(
                participant_id="01",
                session_id="ph",
                contrast_id="ph_pre_rest__tetris",
                band="alpha",
                rest_id=True,
                task_id=True,
                rest_mu=-10.0,
                task_mu=10.0,
            ),
            # Rest-only
            _paired_peak_row(
                participant_id="02",
                session_id="ph",
                contrast_id="ph_pre_rest__tetris",
                band="alpha",
                rest_id=True,
                task_id=False,
                rest_mu=4.0,
                task_mu=99.0,
            ),
            # Task-only
            _paired_peak_row(
                participant_id="03",
                session_id="ph",
                contrast_id="ph_pre_rest__tetris",
                band="alpha",
                rest_id=False,
                task_id=True,
                rest_mu=-99.0,
                task_mu=-2.0,
            ),
        ]
        long_rows = build_panel_e_peak_export(rows, hiit_sensitivity=True)
        rest_mu = [
            float(r["peak_center_mu_s"])
            for r in long_rows
            if r["state"] == "rest" and r["peak_identifiable"]
        ]
        task_mu = [
            float(r["peak_center_mu_s"])
            for r in long_rows
            if r["state"] == "task" and r["peak_identifiable"]
        ]
        self.assertEqual(sorted(rest_mu), [-10.0, 4.0])
        self.assertEqual(sorted(task_mu), [-2.0, 10.0])

        summaries = build_panel_e_state_summaries(long_rows, n_bootstrap=200, seed=1)
        rest_summary = next(
            s
            for s in summaries
            if s["band"] == "alpha" and s["state"] == "rest" and s["parameter"] == "mu"
        )
        task_summary = next(
            s
            for s in summaries
            if s["band"] == "alpha" and s["state"] == "task" and s["parameter"] == "mu"
        )
        # Mean of participant means with one obs each: (-10+4)/2 and (10-2)/2.
        self.assertAlmostEqual(float(rest_summary["estimate"]), -3.0, places=12)
        self.assertAlmostEqual(float(task_summary["estimate"]), 4.0, places=12)
        self.assertEqual(int(rest_summary["identifiable_n"]), 2)
        self.assertEqual(int(task_summary["identifiable_n"]), 2)
        # Joint-only means would be -10 and +10.
        self.assertNotAlmostEqual(float(rest_summary["estimate"]), -10.0, places=6)
        self.assertNotAlmostEqual(float(task_summary["estimate"]), 10.0, places=6)

    def test_render_exports_state_long_and_summary(self) -> None:
        paired = [
            _paired_peak_row(
                participant_id="01",
                session_id="ph",
                contrast_id="ph_pre_rest__tetris",
                band=band,
                rest_id=True,
                task_id=(band != "theta"),
                rest_mu=1.0,
                task_mu=2.0,
            )
            for band in ("theta", "alpha", "beta", "low_gamma")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pub = root / "publish"
            pub.mkdir()
            # Minimal paired_contrasts CSV for figure inputs.
            import csv

            fields = list(paired[0].keys())
            with (pub / "paired_contrasts.csv").open("w", newline="", encoding="utf-8") as h:
                writer = csv.DictWriter(h, fieldnames=fields)
                writer.writeheader()
                writer.writerows(paired)
            # Empty companions so resolve/render does not fail hard.
            for name in (
                "confirmatory_cross_correlation_curves_D240.csv",
                "dataset_effects.csv",
                "meta_analysis_results.csv",
                "mixed_model_results.csv",
                "mixed_model_marginal_estimates.csv",
                "mixed_model_contrasts.csv",
                "peak_center_equivalence.csv",
                "protocol_audit_summary.csv",
            ):
                (pub / name).write_text("dataset_id\n", encoding="utf-8")

            from ppg_eeg.confirmatory.figures import resolve_reporting_inputs

            inputs = resolve_reporting_inputs(pub)
            out = root / "figures"
            render_figure2(inputs, out)
            state_csv = out / "source_data" / "figure2_panel_e_state_peaks.csv"
            summary_csv = out / "source_data" / "figure2_panel_e_state_summaries.csv"
            self.assertTrue(state_csv.is_file())
            self.assertTrue(summary_csv.is_file())
            self.assertFalse(
                (out / "source_data" / "figure2_panel_e_paired_peaks.csv").exists()
            )
            states = read_csv_rows(state_csv)
            self.assertTrue(any(r["state"] == "rest" and r["band"] == "theta" for r in states))
            # Rest-only theta must keep finite Rest μ.
            theta_rest = next(
                r for r in states if r["band"] == "theta" and r["state"] == "rest"
            )
            self.assertTrue(str(theta_rest["peak_identifiable"]).lower() in {"true", "1"})
            self.assertTrue(math.isfinite(float(theta_rest["peak_center_mu_s"])))
            theta_task = next(
                r for r in states if r["band"] == "theta" and r["state"] == "task"
            )
            self.assertFalse(
                str(theta_task["peak_identifiable"]).lower() in {"true", "1"}
            )
            self.assertFalse(math.isfinite(float(theta_task["peak_center_mu_s"] or "nan")))


if __name__ == "__main__":
    unittest.main()
