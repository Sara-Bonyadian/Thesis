"""Figure 3 Panel A participant-forest and supplement diagnostics."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import (
    FIGURE3_PARTICIPANT_FOREST_STEM,
    FIGURE3_STEM,
    FIGURE3_SUPPLEMENT_LABEL,
    FIGURE3_SUPPLEMENT_STEM,
    PRIMARY_REPRESENTATION,
    render_figure3,
)
from ppg_eeg.confirmatory.null_delta_inference import (
    PRIMARY_BAND,
    PRIMARY_NULL_TYPE,
    analyze_null_slice,
    analyze_null_slice_full,
    classify_primary_interpretation,
    infer_dataset_null_deltas,
    infer_participant_null_deltas,
    participant_deltas_from_matched,
    resolve_biological_keys,
    sample_size_annotation,
    secondary_band_null_fdr_table,
)
from ppg_eeg.confirmatory.nulls import (
    NULL_TYPE_AR1_INNOVATIONS,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_CROSS_SUBJECT_MISMATCH,
    NULL_TYPE_PHASE_RANDOMIZATION,
    SeriesUnit,
    amplitude_spectrum,
    analysis_key,
    circular_shift_series,
    complete_blocks,
    compute_endpoint_index_from_series,
    deterministic_seed,
    phase_randomize_series,
    valid_circular_shifts,
    _null_statistics_for_unit,
)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr - np.mean(arr)) / np.std(arr)


def _null_row(
    *,
    dataset_id: str,
    subject_id: str,
    observation_id: str,
    condition: str,
    band: str,
    null_type: str,
    observed: float,
    null_mean: float,
    power_representation: str = "absolute_log10",
    duration_s: int = 240,
    n_surrogates: int = 20,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "observation_id": observation_id,
        "condition": condition,
        "modality": "default",
        "endpoint_name": ENDPOINT_ZLPI,
        "duration_s": duration_s,
        "band": band,
        "power_representation": power_representation,
        "null_type": null_type,
        "n_surrogates_requested": n_surrogates,
        "n_surrogates_finite": n_surrogates,
        "observed_endpoint_index": observed,
        "observed_eligible": eligible,
        "null_mean": null_mean,
        "empirical_p": 0.5,
        "effect_size_surrogate_z": 0.0,
        "rng_seed_u64": "1",
    }


def _write_null_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestParticipantNullDeltaEstimand(unittest.TestCase):
    def test_each_participant_one_delta_equal_weight(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="o1",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.30,
                null_mean=0.10,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="o2",
                condition="task",
                band="theta",
                null_type="circular_shift",
                observed=0.50,
                null_mean=0.10,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="02",
                observation_id="o3",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.20,
                null_mean=0.00,
            ),
        ]
        participants, inference, matched = analyze_null_slice(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(participants), 2)
        by_id = {p.participant_id: p for p in participants}
        # Participant 01 → canonical "1": mean((0.2)+(0.4))/2 = 0.3
        self.assertAlmostEqual(by_id["1"].delta_p, 0.30, places=12)
        self.assertEqual(by_id["1"].n_observations, 2)
        self.assertAlmostEqual(by_id["2"].delta_p, 0.20, places=12)
        # Equal weight: (0.30 + 0.20) / 2 despite unequal observation counts.
        self.assertAlmostEqual(inference.mean_delta, 0.25, places=12)
        self.assertEqual(inference.n_participants, 2)
        self.assertEqual(inference.n_observations, 3)

    def test_ineligible_and_wrong_slice_excluded_symmetrically(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="keep",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.4,
                null_mean=0.1,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="relative",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=9.0,
                null_mean=0.0,
                power_representation="relative",
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="alpha",
                condition="rest",
                band="alpha",
                null_type="circular_shift",
                observed=9.0,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="phase",
                condition="rest",
                band="theta",
                null_type="phase_randomization",
                observed=9.0,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="bad",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=9.0,
                null_mean=0.0,
                eligible=False,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="02",
                observation_id="p2",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.2,
                null_mean=0.0,
            ),
        ]
        participants, inference, matched = analyze_null_slice(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual({m.observation_id for m in matched}, {"keep", "p2"})
        self.assertEqual(inference.n_participants, 2)
        self.assertAlmostEqual(inference.mean_delta, 0.25, places=12)

    def test_dataset_scoped_participant_ids(self) -> None:
        rows = [
            _null_row(
                dataset_id="ds_a",
                subject_id="01",
                observation_id="a",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.1,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="ds_b",
                subject_id="01",
                observation_id="b",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.3,
                null_mean=0.0,
            ),
        ]
        participants, inference, _ = analyze_null_slice(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual(len(participants), 2)
        self.assertEqual(
            {p.participant_unit_id for p in participants},
            {"ds_a::1", "ds_b::1"},
        )
        self.assertEqual(inference.n_participants, 2)

    def test_secondary_fdr_excludes_primary_slice(self) -> None:
        rows = []
        for subject in ("01", "02", "03"):
            for null_type in (
                NULL_TYPE_CIRCULAR_SHIFT,
                NULL_TYPE_PHASE_RANDOMIZATION,
                NULL_TYPE_BLOCK_SHUFFLE,
            ):
                for band in ("theta", "alpha"):
                    rows.append(
                        _null_row(
                            dataset_id="hiit",
                            subject_id=subject,
                            observation_id=f"{subject}_{band}_{null_type}",
                            condition="rest",
                            band=band,
                            null_type=null_type,
                            observed=0.1,
                            null_mean=0.0,
                        )
                    )
        table = secondary_band_null_fdr_table(
            rows,
            bands=("theta", "alpha"),
            null_types=(
                NULL_TYPE_CIRCULAR_SHIFT,
                NULL_TYPE_PHASE_RANDOMIZATION,
                NULL_TYPE_BLOCK_SHUFFLE,
            ),
        )
        primary = [
            r
            for r in table
            if r["band"] == "theta" and r["null_type"] == "circular_shift"
        ]
        self.assertEqual(len(primary), 1)
        self.assertFalse(primary[0]["in_fdr_family"])
        self.assertTrue(math.isnan(float(primary[0]["q_value"])))
        family = [r for r in table if r["in_fdr_family"]]
        self.assertGreaterEqual(len(family), 1)
        self.assertTrue(all(math.isfinite(float(r["q_value"])) for r in family))


class TestFigure3PanelAForest(unittest.TestCase):
    def _base_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for subject_i, subject in enumerate(("01", "02", "03")):
            for cond_i, condition in enumerate(("rest", "task")):
                rows.append(
                    _null_row(
                        dataset_id="hiit",
                        subject_id=subject,
                        observation_id=f"{subject}_{condition}",
                        condition=condition,
                        band="theta",
                        null_type="circular_shift",
                        observed=0.10 + 0.01 * subject_i,
                        null_mean=0.02,
                        n_surrogates=20,
                    )
                )
                # Nested extras that must not enter the primary forest.
                rows.append(
                    _null_row(
                        dataset_id="hiit",
                        subject_id=subject,
                        observation_id=f"{subject}_{condition}_alpha",
                        condition=condition,
                        band="alpha",
                        null_type="circular_shift",
                        observed=0.9,
                        null_mean=0.0,
                    )
                )
                rows.append(
                    _null_row(
                        dataset_id="hiit",
                        subject_id=subject,
                        observation_id=f"{subject}_{condition}_phase",
                        condition=condition,
                        band="theta",
                        null_type="phase_randomization",
                        observed=0.9,
                        null_mean=0.0,
                    )
                )
                rows.append(
                    _null_row(
                        dataset_id="hiit",
                        subject_id=subject,
                        observation_id=f"{subject}_{condition}_rel",
                        condition=condition,
                        band="theta",
                        null_type="circular_shift",
                        observed=0.9,
                        null_mean=0.0,
                        power_representation="relative",
                    )
                )
        return rows

    def test_forest_exports_and_primary_slice_only(self) -> None:
        rows = self._base_rows()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            null_path = root / "null_subject_results.csv"
            _write_null_csv(null_path, rows)
            out = root / "figures"
            artifacts = render_figure3({"null_subject": null_path}, out)
            self.assertEqual(artifacts.manuscript.figure_id, "figure3")
            self.assertIsNotNone(artifacts.supplement)
            self.assertTrue(artifacts.manuscript.png.is_file())

            self.assertTrue((out / f"{FIGURE3_STEM}.png").is_file())
            self.assertTrue((out / f"{FIGURE3_SUPPLEMENT_STEM}.png").is_file())
            # Distinct stems so smoke/main cannot overwrite supplement.
            self.assertNotEqual(FIGURE3_STEM, FIGURE3_SUPPLEMENT_STEM)

            delta_csv = out / "source_data" / "figure3_panel_a_participant_deltas.csv"
            inference_csv = out / "source_data" / "figure3_panel_a_inference.csv"
            loo_csv = out / "source_data" / "figure3_panel_a_leave_one_out.csv"
            secondary_csv = out / "source_data" / "figure3_secondary_nulls_fdr.csv"
            scatter_csv = (
                out / "source_data" / "figure3_supplement_nested_null_scatter.csv"
            )
            for path in (delta_csv, inference_csv, loo_csv, secondary_csv, scatter_csv):
                self.assertTrue(path.is_file(), msg=str(path))

            with delta_csv.open(encoding="utf-8") as handle:
                deltas = list(csv.DictReader(handle))
            self.assertEqual(len(deltas), 3)
            self.assertEqual({r["participant_id"] for r in deltas}, {"1", "2", "3"})
            self.assertEqual({r["subject_id"] for r in deltas}, {"1", "2", "3"})
            self.assertTrue(all(r["band"] == "theta" for r in deltas))
            self.assertTrue(all(r["null_type"] == "circular_shift" for r in deltas))
            self.assertTrue(
                all(r["power_representation"] == PRIMARY_REPRESENTATION for r in deltas)
            )
            # Duplicate nested rows cannot inflate participant count.
            self.assertEqual(len(deltas), len({r["participant_unit_id"] for r in deltas}))

            dataset_csv = out / "source_data" / "figure3_panel_a_dataset_inference.csv"
            self.assertTrue(dataset_csv.is_file())
            with dataset_csv.open(encoding="utf-8") as handle:
                dataset_rows = list(csv.DictReader(handle))
            self.assertEqual(len(dataset_rows), 1)
            self.assertEqual(dataset_rows[0]["dataset_id"], "hiit")
            self.assertEqual(int(float(dataset_rows[0]["n_participants"])), 3)

            condition_csv = (
                out / "source_data" / "figure3_panel_a_participant_condition_deltas.csv"
            )
            self.assertTrue(condition_csv.is_file())
            with condition_csv.open(encoding="utf-8") as handle:
                condition_rows = list(csv.DictReader(handle))
            # 3 participants × 2 conditions
            self.assertEqual(len(condition_rows), 6)

            # Internal QC participant forests (not manuscript/supplement).
            part_forest = list(
                (out / "internal_qc").glob(f"{FIGURE3_PARTICIPANT_FOREST_STEM}*.png")
            )
            self.assertGreaterEqual(len(part_forest), 1)
            # Must not land in the manuscript/supplement figures root.
            self.assertEqual(
                list(out.glob("figure3_supplement_participant_null_forests*.png")),
                [],
            )
            self.assertEqual(
                list(out.glob(f"{FIGURE3_PARTICIPANT_FOREST_STEM}*.png")),
                [],
            )

            with inference_csv.open(encoding="utf-8") as handle:
                inference = list(csv.DictReader(handle))[0]
            self.assertEqual(int(float(inference["n_participants"])), 3)
            self.assertEqual(
                inference["sample_size_label"],
                "6 condition estimates from 3 participants",
            )
            self.assertEqual(inference["run_class"], "smoke_diagnostic")
            self.assertEqual(str(inference.get("pooled_estimate_plotted")).casefold(), "false")
            recomputed = infer_participant_null_deltas(
                participant_deltas_from_matched(
                    analyze_null_slice(
                        rows,
                        band=PRIMARY_BAND,
                        null_type=PRIMARY_NULL_TYPE,
                        is_primary_slice=True,
                    )[2]
                ),
                band=PRIMARY_BAND,
                null_type=PRIMARY_NULL_TYPE,
                is_primary_slice=True,
            )
            self.assertAlmostEqual(
                float(inference["mean_delta"]), recomputed.mean_delta, places=12
            )
            self.assertAlmostEqual(
                float(inference["ci_low"]), recomputed.ci_low, places=12
            )
            self.assertAlmostEqual(
                float(inference["ci_high"]), recomputed.ci_high, places=12
            )

            with scatter_csv.open(encoding="utf-8") as handle:
                scatter = list(csv.DictReader(handle))
            self.assertTrue(
                all(r["power_representation"] == PRIMARY_REPRESENTATION for r in scatter)
            )
            self.assertTrue(
                all(r["diagnostic_label"] == FIGURE3_SUPPLEMENT_LABEL for r in scatter)
            )
            # Supplemental scatter still includes all null types / bands (nested).
            self.assertGreater(len(scatter), len(deltas))
            self.assertIn(
                "nested_null_scatter_diagnostic",
                {p.panel_id for p in artifacts.panels},
            )
            self.assertIn(
                "null_dataset_forest",
                {p.panel_id for p in artifacts.panels},
            )

            with loo_csv.open(encoding="utf-8") as handle:
                loo = list(csv.DictReader(handle))
            self.assertEqual(len(loo), 3)

    def test_unequal_observations_do_not_change_weights(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="rich",
                observation_id=f"r{i}",
                condition=f"c{i}",
                band="theta",
                null_type="circular_shift",
                observed=1.0,
                null_mean=0.0,
            )
            for i in range(5)
        ] + [
            _null_row(
                dataset_id="hiit",
                subject_id="poor",
                observation_id="p0",
                condition="c0",
                band="theta",
                null_type="circular_shift",
                observed=0.0,
                null_mean=0.0,
            )
        ]
        _participants, inference, _ = analyze_null_slice(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        # Equal weight (1.0 + 0.0) / 2 = 0.5, not observation-weighted 5/6.
        self.assertAlmostEqual(inference.mean_delta, 0.5, places=12)


class TestBiologicalParticipantUnit(unittest.TestCase):
    def test_hiit_ph_ps_are_same_biological_participant(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="01_ph",
                observation_id="hiit-01-ph-pre-rest",
                condition="ph_pre_rest",
                band="theta",
                null_type="circular_shift",
                observed=0.40,
                null_mean=0.10,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="01_ps",
                observation_id="hiit-01-ps-pre-rest",
                condition="ps_pre_rest",
                band="theta",
                null_type="circular_shift",
                observed=0.20,
                null_mean=0.00,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="02_ph",
                observation_id="hiit-02-ph-pre-rest",
                condition="ph_pre_rest",
                band="theta",
                null_type="circular_shift",
                observed=0.10,
                null_mean=0.00,
            ),
        ]
        keys_ph = resolve_biological_keys(rows[0])
        keys_ps = resolve_biological_keys(rows[1])
        self.assertEqual(keys_ph["participant_id"], keys_ps["participant_id"])
        self.assertEqual(keys_ph["participant_unit_id"], "hiit::1")
        self.assertNotEqual(keys_ph["session_id"], keys_ps["session_id"])

        analysis = analyze_null_slice_full(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual(len(analysis.participants), 2)
        self.assertEqual(
            {p.participant_id for p in analysis.participants},
            {"1", "2"},
        )
        # Participant 1 averages two protocol conditions (Δ=0.3 and Δ=0.2).
        by_id = {p.participant_id: p for p in analysis.participants}
        self.assertAlmostEqual(by_id["1"].delta_p, 0.25, places=12)
        self.assertEqual(by_id["1"].n_conditions, 2)
        self.assertEqual(by_id["1"].n_sessions, 2)
        self.assertEqual(analysis.pooled_inference.n_participants, 2)
        self.assertEqual(analysis.pooled_inference.n_participant_conditions, 3)
        self.assertEqual(
            analysis.pooled_inference.sample_size_label,
            "3 condition estimates from 2 participants",
        )
        # Dataset mean from participant-level values: (0.25 + 0.10) / 2.
        self.assertAlmostEqual(analysis.dataset_inferences[0].mean_delta, 0.175, places=12)
        self.assertEqual(analysis.dataset_inferences[0].n_participants, 2)

    def test_unequal_conditions_do_not_change_participant_weights(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="01_ph",
                observation_id=f"hiit-01-ph-c{i}",
                condition=f"ph_c{i}",
                band="theta",
                null_type="circular_shift",
                observed=1.0,
                null_mean=0.0,
            )
            for i in range(4)
        ] + [
            _null_row(
                dataset_id="hiit",
                subject_id="02_ph",
                observation_id="hiit-02-ph-c0",
                condition="ph_c0",
                band="theta",
                null_type="circular_shift",
                observed=0.0,
                null_mean=0.0,
            )
        ]
        participants, inference, _ = analyze_null_slice(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual(len(participants), 2)
        # Participant with 4 conditions still contributes one unit: mean = 0.5.
        self.assertAlmostEqual(inference.mean_delta, 0.5, places=12)

    def test_dataset_means_from_participant_values_multi_dataset(self) -> None:
        rows = [
            _null_row(
                dataset_id="hiit",
                subject_id="01",
                observation_id="hiit-01-ph-rest",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.20,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="hiit",
                subject_id="02",
                observation_id="hiit-02-ph-rest",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.40,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="ds003838",
                subject_id="sub-01",
                observation_id="ds003838-sub-01",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.10,
                null_mean=0.0,
            ),
            _null_row(
                dataset_id="ds003838",
                subject_id="sub-02",
                observation_id="ds003838-sub-02",
                condition="rest",
                band="theta",
                null_type="circular_shift",
                observed=0.30,
                null_mean=0.0,
            ),
        ]
        analysis = analyze_null_slice_full(
            rows,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        by_ds = {d.dataset_id: d for d in analysis.dataset_inferences}
        self.assertAlmostEqual(by_ds["hiit"].mean_delta, 0.30, places=12)
        self.assertAlmostEqual(by_ds["ds003838"].mean_delta, 0.20, places=12)
        self.assertEqual(by_ds["hiit"].n_participants, 2)
        self.assertEqual(by_ds["ds003838"].n_participants, 2)
        # Pooled is available for secondary use but not a plotted main-panel row.
        self.assertEqual(analysis.pooled_inference.dataset_id, "pooled")
        inferred = infer_dataset_null_deltas(
            analysis.participants,
            band=PRIMARY_BAND,
            null_type=PRIMARY_NULL_TYPE,
            is_primary_slice=True,
        )
        self.assertEqual([d.dataset_id for d in inferred], ["hiit", "ds003838"])

    def test_sample_size_annotation_wording(self) -> None:
        self.assertEqual(
            sample_size_annotation(n_participants=3, n_participant_conditions=3),
            "n=3 participants",
        )
        self.assertEqual(
            sample_size_annotation(n_participants=3, n_participant_conditions=6),
            "6 condition estimates from 3 participants",
        )


class TestInterpretationCategories(unittest.TestCase):
    def test_categories(self) -> None:
        self.assertEqual(
            classify_primary_interpretation(
                n_participants=10, mean_delta=0.2, ci_low=0.05, ci_high=0.35
            ),
            "observed significantly exceeds null",
        )
        self.assertEqual(
            classify_primary_interpretation(
                n_participants=10, mean_delta=0.05, ci_low=-0.02, ci_high=0.12
            ),
            "directionally positive but inconclusive",
        )
        self.assertEqual(
            classify_primary_interpretation(
                n_participants=10, mean_delta=-0.01, ci_low=-0.1, ci_high=0.08
            ),
            "no evidence of difference",
        )
        self.assertEqual(
            classify_primary_interpretation(
                n_participants=10, mean_delta=-0.2, ci_low=-0.3, ci_high=-0.05
            ),
            "observed is lower than null",
        )
        self.assertEqual(
            classify_primary_interpretation(
                n_participants=1, mean_delta=0.2, ci_low=float("nan"), ci_high=float("nan")
            ),
            "insufficient participant-level data",
        )


class TestSurrogateContracts(unittest.TestCase):
    def test_circular_shift_preserves_values_and_destroys_zero_lag_alignment(self) -> None:
        rng = np.random.default_rng(11)
        n = 240
        t = np.arange(n, dtype=float)
        pulse = np.exp(-0.5 * ((t - 120.0) / 8.0) ** 2)
        hr = _zscore(pulse + 0.05 * rng.normal(size=n))
        eeg = _zscore(pulse + 0.05 * rng.normal(size=n))
        shifts = valid_circular_shifts(n)
        shift = int(shifts[len(shifts) // 2])
        eeg_s = circular_shift_series(eeg, shift)
        np.testing.assert_allclose(np.sort(eeg_s), np.sort(eeg))
        self.assertGreater(float(np.corrcoef(hr, eeg)[0, 1]), 0.8)
        self.assertLess(abs(float(np.corrcoef(hr, eeg_s)[0, 1])), 0.35)

    def test_phase_randomization_preserves_amplitude_spectrum(self) -> None:
        rng = np.random.default_rng(5)
        signal = rng.normal(size=256)
        surrogate = phase_randomize_series(signal, np.random.default_rng(99))
        np.testing.assert_allclose(
            amplitude_spectrum(surrogate),
            amplitude_spectrum(signal),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_complete_blocks_uses_fs_hz(self) -> None:
        values = np.arange(240, dtype=float)
        self.assertEqual(complete_blocks(values, block_length_s=30, fs_hz=1.0).shape, (8, 30))
        self.assertEqual(complete_blocks(values, block_length_s=30, fs_hz=2.0).shape, (4, 60))

    def test_null_mean_is_arithmetic_mean_and_seed_reproducible(self) -> None:
        rng = np.random.default_rng(3)
        hr = _zscore(rng.normal(size=240))
        eeg = _zscore(hr + 0.2 * rng.normal(size=240))
        unit = SeriesUnit(
            dataset_id="ds",
            subject_id="s1",
            task="rest",
            condition="rest",
            observation_id="obs-1",
            modality="default",
            duration_s=240,
            duration_role="primary",
            band="theta",
            power_representation="absolute_log10",
            is_primary_representation=True,
            pair="hr_x_theta_absolute_log10",
            hr_z=hr,
            eeg_z=eeg,
        )
        row_a, _ = _null_statistics_for_unit(
            unit, null_type=NULL_TYPE_CIRCULAR_SHIFT, n_surrogates=20
        )
        row_b, _ = _null_statistics_for_unit(
            unit, null_type=NULL_TYPE_CIRCULAR_SHIFT, n_surrogates=20
        )
        self.assertEqual(row_a["null_mean"], row_b["null_mean"])
        key = analysis_key(
            duration_s=240,
            endpoint_name=ENDPOINT_ZLPI,
            band="theta",
            power_representation="absolute_log10",
            modality="default",
        )
        seed = deterministic_seed("obs-1", key, NULL_TYPE_CIRCULAR_SHIFT)
        local_rng = np.random.default_rng(seed)
        null_vals = []
        for _ in range(20):
            shift = int(local_rng.choice(valid_circular_shifts(240)))
            metrics = compute_endpoint_index_from_series(
                hr,
                circular_shift_series(eeg, shift),
                duration_s=240,
                identity={"observation_id": "obs-1"},
                band="theta",
                power_representation="absolute_log10",
            )
            null_vals.append(float(metrics["endpoint_index"]))
        self.assertAlmostEqual(float(row_a["null_mean"]), float(np.mean(null_vals)), places=12)

    def test_secondary_null_type_constants(self) -> None:
        self.assertIn(NULL_TYPE_PHASE_RANDOMIZATION, (
            NULL_TYPE_PHASE_RANDOMIZATION,
            NULL_TYPE_BLOCK_SHUFFLE,
            NULL_TYPE_CROSS_SUBJECT_MISMATCH,
            NULL_TYPE_AR1_INNOVATIONS,
        ))


if __name__ == "__main__":
    unittest.main()
