"""Validation for Figure 2 Panel A cluster-aware paired Δ inference."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import PRIMARY_REPRESENTATION, read_csv_rows, render_figure2
from ppg_eeg.confirmatory.paired_delta_inference import (
    cluster_bootstrap_mean_ci,
    infer_independent_sampling_unit,
    infer_paired_deltas_cluster_aware,
    summarize_by_independent_unit,
)


def _row(
    *,
    participant_id: str,
    contrast_id: str,
    delta: float,
    dataset_id: str = "demo",
    session_id: str = "single",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "contrast_id": contrast_id,
        "band": "theta",
        "duration_s": 240,
        "endpoint_name": ENDPOINT_ZLPI,
        "power_representation": PRIMARY_REPRESENTATION,
        "low_endpoint_index": 0.0,
        "effort_endpoint_index": float(delta),
        "delta_endpoint_index": float(delta),
    }


class IndependentUnitDetectionTests(unittest.TestCase):
    def test_detects_nested_repeated_measures(self) -> None:
        rows = [
            _row(participant_id="01", contrast_id="a", delta=0.1),
            _row(participant_id="01", contrast_id="b", delta=0.2),
            _row(participant_id="02", contrast_id="a", delta=-0.1),
            _row(participant_id="02", contrast_id="b", delta=0.0),
        ]
        decision = infer_independent_sampling_unit(rows)
        self.assertEqual(decision.unit_field, "participant_id")
        self.assertEqual(decision.n_observations, 4)
        self.assertEqual(decision.n_units, 2)
        self.assertTrue(decision.nested_repeated_measures)

    def test_one_observation_per_participant_not_nested(self) -> None:
        rows = [
            _row(participant_id="01", contrast_id="a", delta=0.1),
            _row(participant_id="02", contrast_id="a", delta=0.2),
            _row(participant_id="03", contrast_id="a", delta=0.3),
        ]
        decision = infer_independent_sampling_unit(rows)
        self.assertFalse(decision.nested_repeated_measures)
        self.assertEqual(decision.n_units, 3)
        self.assertEqual(decision.n_observations, 3)


class ClusterAwareInferenceTests(unittest.TestCase):
    def test_ci_uses_n_units_not_n_rows(self) -> None:
        # Two participants, four nested contrasts. Naïve row-level SE uses n=4;
        # correct unit-level SE uses n=2.
        rows = [
            _row(participant_id="01", contrast_id="c1", delta=1.0),
            _row(participant_id="01", contrast_id="c2", delta=1.0),
            _row(participant_id="02", contrast_id="c1", delta=-1.0),
            _row(participant_id="02", contrast_id="c2", delta=-1.0),
        ]
        result, summaries = infer_paired_deltas_cluster_aware(rows, n_bootstrap=200, bootstrap_seed=1)
        self.assertEqual(result.n_observations, 4)
        self.assertEqual(result.n_units, 2)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(result.df, 1)  # n_units - 1
        self.assertAlmostEqual(result.mean_delta, 0.0, places=12)

        ys = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=float)
        naive_se = float(np.std(ys, ddof=1) / math.sqrt(ys.size))
        unit_means = np.asarray([1.0, -1.0], dtype=float)
        unit_se = float(np.std(unit_means, ddof=1) / math.sqrt(unit_means.size))
        self.assertAlmostEqual(result.se_delta, unit_se, places=12)
        self.assertGreater(result.se_delta, naive_se)
        # CI must not be computed with df = n_rows - 1.
        self.assertNotEqual(result.df, 3)

    def test_duplicated_participants_cannot_inflate_n(self) -> None:
        rows = [
            _row(participant_id="pA", contrast_id=f"c{i}", delta=0.5 + 0.01 * i)
            for i in range(8)
        ] + [
            _row(participant_id="pB", contrast_id=f"c{i}", delta=-0.4 + 0.01 * i)
            for i in range(8)
        ]
        result, _ = infer_paired_deltas_cluster_aware(rows, n_bootstrap=100, bootstrap_seed=2)
        self.assertEqual(result.n_observations, 16)
        self.assertEqual(result.n_units, 2)
        self.assertTrue(result.nested_repeated_measures)

    def test_cluster_bootstrap_resamples_units(self) -> None:
        rows = [
            _row(participant_id="01", contrast_id="a", delta=2.0),
            _row(participant_id="01", contrast_id="b", delta=2.0),
            _row(participant_id="02", contrast_id="a", delta=0.0),
            _row(participant_id="02", contrast_id="b", delta=0.0),
            _row(participant_id="03", contrast_id="a", delta=0.0),
            _row(participant_id="03", contrast_id="b", delta=0.0),
        ]
        mean, lo, hi = cluster_bootstrap_mean_ci(
            rows, n_draws=1000, seed=123, estimand="mean_of_unit_means"
        )
        self.assertAlmostEqual(mean, 2.0 / 3.0, places=12)
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_unit_summaries_equal_weight_per_participant(self) -> None:
        rows = [
            _row(participant_id="01", contrast_id="a", delta=10.0),
            _row(participant_id="01", contrast_id="b", delta=10.0),
            _row(participant_id="01", contrast_id="c", delta=10.0),
            _row(participant_id="02", contrast_id="a", delta=0.0),
        ]
        summaries = summarize_by_independent_unit(rows)
        means = {s.unit_id: s.mean_delta for s in summaries}
        # demo::01 and demo::02
        self.assertEqual(len(summaries), 2)
        self.assertAlmostEqual(float(np.mean(list(means.values()))), 5.0, places=12)

    def test_unbalanced_contrasts_do_not_inflate_participant_weight(self) -> None:
        """Participant with many contrast rows must not dominate the CI mean."""
        # Participant A: 1 contrast at +1.0
        # Participant B: 9 contrasts all at 0.0
        # Equal-participant mean = (1.0 + 0.0) / 2 = 0.5
        # Row-weighted mean would be 1/10 = 0.1
        rows = [_row(participant_id="A", contrast_id="only", delta=1.0)] + [
            _row(participant_id="B", contrast_id=f"c{i}", delta=0.0) for i in range(9)
        ]
        result, summaries = infer_paired_deltas_cluster_aware(
            rows, n_bootstrap=200, bootstrap_seed=7
        )
        self.assertEqual(result.n_units, 2)
        self.assertEqual(result.n_observations, 10)
        self.assertEqual(result.estimand, "mean_of_unit_means")
        self.assertAlmostEqual(result.mean_delta, 0.5, places=12)

        by_n = {s.unit_id: s.n_observations for s in summaries}
        self.assertEqual(sorted(by_n.values()), [1, 9])

        row_weighted = float(
            np.average(
                [s.mean_delta for s in summaries],
                weights=[s.n_observations for s in summaries],
            )
        )
        self.assertAlmostEqual(row_weighted, 0.1, places=12)
        self.assertNotAlmostEqual(result.mean_delta, row_weighted, places=12)

        # Diagnostic bootstrap estimand can use observation weights, but primary
        # cluster bootstrap matches equal unit weight.
        boot_unit, _, _ = cluster_bootstrap_mean_ci(
            rows, n_draws=50, seed=1, estimand="mean_of_unit_means"
        )
        boot_obs, _, _ = cluster_bootstrap_mean_ci(
            rows, n_draws=50, seed=1, estimand="mean_of_observations"
        )
        self.assertAlmostEqual(boot_unit, 0.5, places=12)
        self.assertAlmostEqual(boot_obs, 0.1, places=12)

    def test_ineligible_and_nonfinite_contrasts_are_excluded_from_unit_means(self) -> None:
        from ppg_eeg.confirmatory.figures import _paired_points_for_dataset

        paired = [
            {
                **_row(participant_id="01", contrast_id="ok", delta=0.4),
                "contrast_eligible": True,
            },
            {
                **_row(participant_id="01", contrast_id="bad", delta=99.0),
                "contrast_eligible": False,
            },
            {
                **_row(participant_id="02", contrast_id="ok", delta=0.2),
                "contrast_eligible": True,
            },
            {
                **_row(participant_id="02", contrast_id="nan", delta=float("nan")),
                "contrast_eligible": True,
                "delta_endpoint_index": float("nan"),
            },
        ]
        points = _paired_points_for_dataset([], paired, dataset_id="demo", band="theta")
        self.assertEqual(len(points), 2)
        result, summaries = infer_paired_deltas_cluster_aware(points, n_bootstrap=20)
        self.assertEqual(result.n_units, 2)
        self.assertEqual(result.n_observations, 2)
        self.assertAlmostEqual(result.mean_delta, 0.3, places=12)
        self.assertTrue(all(s.n_observations == 1 for s in summaries))


class Figure2PanelARenderTests(unittest.TestCase):
    def test_render_exports_unit_inference_and_correct_annotation_counts(self) -> None:
        paired = []
        for pid, deltas in {
            "01": (0.2, 0.1, -0.05, 0.0),
            "02": (-0.1, 0.05, 0.0, 0.1),
            "03": (0.0, -0.2, 0.15, 0.05),
        }.items():
            for i, delta in enumerate(deltas):
                paired.append(
                    {
                        "dataset_id": "hiit",
                        "participant_id": pid,
                        "session_id": "s1",
                        "contrast_id": f"contrast_{i}",
                        "band": "theta",
                        "duration_s": 240,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "power_representation": PRIMARY_REPRESENTATION,
                        "low_endpoint_index": 0.0,
                        "effort_endpoint_index": float(delta),
                        "delta_endpoint_index": float(delta),
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path = root / "paired.csv"
            with paired_path.open("w", encoding="utf-8") as handle:
                fields = list(paired[0].keys())
                handle.write(",".join(fields) + "\n")
                for row in paired:
                    handle.write(",".join(str(row[f]) for f in fields) + "\n")
            out = root / "figures"
            render_figure2(
                {
                    "subject_level": None,
                    "paired_contrasts": paired_path,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                },
                out,
            )
            points = read_csv_rows(out / "source_data" / "figure2_panel_a_paired_deltas.csv")
            units = read_csv_rows(out / "source_data" / "figure2_panel_a_unit_summaries.csv")
            inference = read_csv_rows(out / "source_data" / "figure2_panel_a_inference.csv")
            self.assertEqual(len(points), 12)
            self.assertEqual(len(units), 3)
            self.assertEqual(len(inference), 1)
            self.assertEqual(int(float(inference[0]["n_observations"])), 12)
            self.assertEqual(int(float(inference[0]["n_units"])), 3)
            self.assertEqual(str(inference[0]["nested_repeated_measures"]).lower(), "true")
            self.assertEqual(inference[0]["ci_method"], "student_t_on_independent_unit_means")
            # SE based on 3 units, not 12 rows.
            self.assertEqual(int(float(inference[0]["df"])), 2)
            self.assertEqual(points[0]["sampling_unit"], "participant_x_contrast")


if __name__ == "__main__":
    unittest.main()
