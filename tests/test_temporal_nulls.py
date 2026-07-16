from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    contract_for_duration,
)
from ppg_eeg.confirmatory.nulls import (
    BLOCK_LENGTH_S,
    DEFAULT_N_SURROGATES,
    NULL_QC_FILENAME,
    NULL_SUBJECT_RESULTS_FILENAME,
    NULL_SUMMARY_FILENAME,
    NULL_TYPE_AR1_INNOVATIONS,
    NULL_TYPE_BLOCK_SHUFFLE,
    NULL_TYPE_CIRCULAR_SHIFT,
    NULL_TYPE_CROSS_SUBJECT_MISMATCH,
    NULL_TYPE_PHASE_RANDOMIZATION,
    SMOKE_N_SURROGATES,
    SUBJECT_RESULT_FIELDS,
    SeriesUnit,
    amplitude_spectrum,
    analysis_key,
    ar1_innovations,
    block_shuffle_series,
    complete_blocks,
    compute_endpoint_index_from_series,
    deterministic_seed,
    empirical_p_value,
    lag1_autocorrelation,
    phase_randomize_series,
    run_null_battery,
    seeded_derangement,
    surrogate_effect_size,
    write_null_outputs,
)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.mean(values)) / np.std(values)


def _make_unit(
    *,
    observation_id: str,
    subject_id: str,
    hr_z: np.ndarray,
    eeg_z: np.ndarray,
    duration_s: int = 240,
    condition: str = "rest",
    band: str = "theta",
    dataset_id: str = "ds_test",
    modality: str = "default",
) -> SeriesUnit:
    contract = contract_for_duration(duration_s)
    return SeriesUnit(
        dataset_id=dataset_id,
        subject_id=subject_id,
        task=condition,
        condition=condition,
        observation_id=observation_id,
        modality=modality,
        duration_s=duration_s,
        duration_role="primary" if duration_s == 240 else "sensitivity",
        band=band,
        power_representation="absolute_log10",
        is_primary_representation=True,
        pair=f"hr_x_{band}_absolute_log10",
        hr_z=np.asarray(hr_z, dtype=float),
        eeg_z=np.asarray(eeg_z, dtype=float),
    )


class TestDeterministicSeeds(unittest.TestCase):
    def test_sha256_seed_is_stable_and_not_builtin_hash(self) -> None:
        key = analysis_key(
            duration_s=240,
            endpoint_name=ENDPOINT_ZLPI,
            band="theta",
            power_representation="absolute_log10",
        )
        a = deterministic_seed("obs-01", key, NULL_TYPE_CIRCULAR_SHIFT)
        b = deterministic_seed("obs-01", key, NULL_TYPE_CIRCULAR_SHIFT)
        c = deterministic_seed("obs-01", key, NULL_TYPE_PHASE_RANDOMIZATION)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertIsInstance(a, int)
        self.assertGreaterEqual(a, 0)

    def test_null_battery_is_deterministic(self) -> None:
        rng = np.random.default_rng(123)
        base = _zscore(rng.normal(size=240))
        hr = base
        eeg = _zscore(np.roll(base, 0) + 0.05 * rng.normal(size=240))
        unit = _make_unit(
            observation_id="obs-01",
            subject_id="sub-01",
            hr_z=hr,
            eeg_z=eeg,
        )
        first = run_null_battery(
            [unit],
            n_surrogates=SMOKE_N_SURROGATES,
            null_types=(NULL_TYPE_CIRCULAR_SHIFT, NULL_TYPE_PHASE_RANDOMIZATION),
        )
        second = run_null_battery(
            [unit],
            n_surrogates=SMOKE_N_SURROGATES,
            null_types=(NULL_TYPE_CIRCULAR_SHIFT, NULL_TYPE_PHASE_RANDOMIZATION),
        )
        self.assertEqual(first.subject_rows, second.subject_rows)
        self.assertEqual(first.qc_rows, second.qc_rows)


class TestPhaseSpectrumPreservation(unittest.TestCase):
    def test_phase_randomization_preserves_amplitude_spectrum(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.normal(size=256)
        surrogate = phase_randomize_series(signal, np.random.default_rng(99))
        original_amp = amplitude_spectrum(signal)
        surrogate_amp = amplitude_spectrum(surrogate)
        np.testing.assert_allclose(surrogate_amp, original_amp, rtol=1e-10, atol=1e-10)
        # Should not be identical in time domain almost surely.
        self.assertFalse(np.allclose(surrogate, signal))


class TestDerangements(unittest.TestCase):
    def test_seeded_derangement_has_no_fixed_points(self) -> None:
        rng = np.random.default_rng(0)
        for n in (2, 3, 5, 10, 20):
            for _ in range(50):
                perm = seeded_derangement(n, rng)
                self.assertEqual(sorted(perm.tolist()), list(range(n)))
                self.assertFalse(np.any(perm == np.arange(n)))

    def test_cross_subject_mismatch_never_self_pairs(self) -> None:
        rng = np.random.default_rng(11)
        units = []
        for i in range(4):
            base = _zscore(rng.normal(size=240))
            units.append(
                _make_unit(
                    observation_id=f"obs-{i}",
                    subject_id=f"sub-{i}",
                    hr_z=base,
                    eeg_z=_zscore(np.roll(base, i + 1)),
                )
            )
        result = run_null_battery(
            units,
            n_surrogates=10,
            null_types=(NULL_TYPE_CROSS_SUBJECT_MISMATCH,),
        )
        qc_ok = [row for row in result.qc_rows if row["status"] == "ok"]
        self.assertEqual(len(qc_ok), 4)
        for row in result.subject_rows:
            self.assertEqual(row["n_surrogates_finite"], 10)


class TestBlockPreservation(unittest.TestCase):
    def test_block_shuffle_preserves_block_multiset_and_rejects_identity(self) -> None:
        rng = np.random.default_rng(3)
        values = np.arange(240, dtype=float)
        original_blocks = complete_blocks(values, block_length_s=BLOCK_LENGTH_S)
        shuffled, order = block_shuffle_series(
            values, rng, block_length_s=BLOCK_LENGTH_S
        )
        shuffled_blocks = complete_blocks(shuffled, block_length_s=BLOCK_LENGTH_S)
        self.assertFalse(np.array_equal(order, np.arange(order.size)))
        # Same multiset of blocks.
        original_sorted = np.sort(original_blocks, axis=0)
        shuffled_sorted = np.sort(shuffled_blocks, axis=0)
        # Sort rows lexicographically for multiset compare.
        orig_rows = np.array(sorted(map(tuple, original_blocks.tolist())))
        shuf_rows = np.array(sorted(map(tuple, shuffled_blocks.tolist())))
        np.testing.assert_array_equal(orig_rows, shuf_rows)
        # Incomplete tail preserved when present.
        values_tail = np.arange(250, dtype=float)
        shuffled_tail, _ = block_shuffle_series(
            values_tail, np.random.default_rng(5), block_length_s=BLOCK_LENGTH_S
        )
        np.testing.assert_array_equal(shuffled_tail[240:], values_tail[240:])
        self.assertEqual(original_sorted.shape, shuffled_sorted.shape)


class TestAR1Innovations(unittest.TestCase):
    def test_innovations_reduce_lag1_autocorrelation(self) -> None:
        rng = np.random.default_rng(21)
        n = 400
        eps = rng.normal(size=n)
        series = np.zeros(n, dtype=float)
        phi = 0.8
        for t in range(1, n):
            series[t] = phi * series[t - 1] + eps[t]
        original_ac = lag1_autocorrelation(series)
        innovations = ar1_innovations(series)
        innov_ac = lag1_autocorrelation(innovations)
        self.assertGreater(original_ac, 0.5)
        self.assertLess(abs(innov_ac), 0.15)


class TestEndpointSeparation(unittest.TestCase):
    def test_duration_contracts_keep_endpoint_names_separate(self) -> None:
        rng = np.random.default_rng(42)
        base = _zscore(rng.normal(size=240))
        units = [
            _make_unit(
                observation_id="obs-240",
                subject_id="sub-01",
                hr_z=base,
                eeg_z=base,
                duration_s=240,
            ),
            _make_unit(
                observation_id="obs-120",
                subject_id="sub-01",
                hr_z=base[:120],
                eeg_z=base[:120],
                duration_s=120,
            ),
            _make_unit(
                observation_id="obs-60",
                subject_id="sub-01",
                hr_z=base[:60],
                eeg_z=base[:60],
                duration_s=60,
            ),
        ]
        result = run_null_battery(
            units,
            n_surrogates=5,
            null_types=(NULL_TYPE_PHASE_RANDOMIZATION,),
        )
        by_duration = {
            int(row["duration_s"]): row["endpoint_name"]
            for row in result.subject_rows
        }
        self.assertEqual(by_duration[240], ENDPOINT_ZLPI)
        self.assertEqual(by_duration[120], ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(by_duration[60], ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)

        # Summary keys never pool distinct endpoint names together.
        summary_keys = {
            (
                int(row["duration_s"]),
                row["endpoint_name"],
                row["null_type"],
            )
            for row in result.summary_rows
        }
        self.assertEqual(len(summary_keys), 3)
        endpoint_names = {key[1] for key in summary_keys}
        self.assertEqual(
            endpoint_names,
            {
                ENDPOINT_ZLPI,
                ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
            },
        )

    def test_observed_statistic_matches_m6_endpoint_helper(self) -> None:
        rng = np.random.default_rng(5)
        hr = _zscore(rng.normal(size=240))
        eeg = _zscore(hr + 0.1 * rng.normal(size=240))
        metrics = compute_endpoint_index_from_series(
            hr,
            eeg,
            duration_s=240,
            identity={
                "dataset_id": "ds",
                "subject_id": "s",
                "task": "rest",
                "condition": "rest",
                "observation_id": "o",
            },
        )
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_ZLPI)
        self.assertTrue(bool(metrics["eligible"]))
        self.assertTrue(math.isfinite(float(metrics["endpoint_index"])))


class TestEmpiricalPValue(unittest.TestCase):
    def test_empirical_p_and_effect_size_formulas(self) -> None:
        nulls = [0.0, 0.1, 0.2, 0.3, 0.4]
        observed = 0.35
        # greater: values >= 0.35 → 0.4 only → (1+1)/(5+1)=2/6
        self.assertAlmostEqual(
            empirical_p_value(observed, nulls, alternative="greater"),
            2.0 / 6.0,
            places=12,
        )
        # two_sided on |·|: |null| >= 0.35 → 0.4 → same 2/6
        self.assertAlmostEqual(
            empirical_p_value(observed, nulls, alternative="two_sided"),
            2.0 / 6.0,
            places=12,
        )
        mean_null = sum(nulls) / len(nulls)
        std_null = float(np.std(np.asarray(nulls, dtype=float), ddof=1))
        expected_effect = (observed - mean_null) / std_null
        self.assertAlmostEqual(
            surrogate_effect_size(observed, nulls), expected_effect, places=12
        )


class TestWriteOutputs(unittest.TestCase):
    def test_writes_required_null_csv_files(self) -> None:
        rng = np.random.default_rng(9)
        base = _zscore(rng.normal(size=240))
        unit = _make_unit(
            observation_id="obs-01",
            subject_id="sub-01",
            hr_z=base,
            eeg_z=base,
        )
        result = run_null_battery(
            [unit],
            n_surrogates=5,
            null_types=(NULL_TYPE_BLOCK_SHUFFLE, NULL_TYPE_AR1_INNOVATIONS),
        )
        self.assertIn("rng_seed_u64", result.subject_rows[0])
        self.assertNotIn("seed_u64", result.subject_rows[0])
        self.assertIn("rng_seed_u64", SUBJECT_RESULT_FIELDS)
        with TemporaryDirectory() as tmp:
            paths = write_null_outputs(result, tmp)
            self.assertEqual(
                paths["null_subject_results"].name, NULL_SUBJECT_RESULTS_FILENAME
            )
            self.assertEqual(paths["null_summary"].name, NULL_SUMMARY_FILENAME)
            self.assertEqual(paths["null_qc"].name, NULL_QC_FILENAME)
            for path in paths.values():
                self.assertTrue(path.is_file())


class TestSurrogateCountDefaults(unittest.TestCase):
    def test_smoke_and_production_surrogate_counts(self) -> None:
        self.assertEqual(SMOKE_N_SURROGATES, 20)
        self.assertEqual(DEFAULT_N_SURROGATES, 500)
        self.assertNotEqual(DEFAULT_N_SURROGATES, 1000)
        self.assertGreaterEqual(DEFAULT_N_SURROGATES, 500)

    def test_add_one_p_resolution_at_production_n(self) -> None:
        # Finest attainable add-one p when no null meets/exceeds observed.
        n = DEFAULT_N_SURROGATES
        nulls = [0.0] * n
        p = empirical_p_value(1.0, nulls, alternative="greater")
        self.assertAlmostEqual(p, 1.0 / (n + 1), places=12)
        self.assertAlmostEqual(p, 1.0 / 501.0, places=12)


class TestParallelIdentityAndResume(unittest.TestCase):
    def _units(self, n: int = 3) -> list[SeriesUnit]:
        rng = np.random.default_rng(99)
        units: list[SeriesUnit] = []
        for i in range(n):
            base = _zscore(rng.normal(size=240))
            units.append(
                _make_unit(
                    observation_id=f"obs-{i}",
                    subject_id=f"sub-{i}",
                    hr_z=base,
                    eeg_z=_zscore(np.roll(base, i + 1) + 0.05 * rng.normal(size=240)),
                )
            )
        return units

    def test_serial_matches_parallel_same_surrogate_count(self) -> None:
        units = self._units(3)
        null_types = (
            NULL_TYPE_CIRCULAR_SHIFT,
            NULL_TYPE_PHASE_RANDOMIZATION,
            NULL_TYPE_BLOCK_SHUFFLE,
            NULL_TYPE_CROSS_SUBJECT_MISMATCH,
            NULL_TYPE_AR1_INNOVATIONS,
        )
        serial = run_null_battery(
            units,
            n_surrogates=5,
            null_types=null_types,
            n_jobs=1,
            progress=False,
        )
        parallel = run_null_battery(
            units,
            n_surrogates=5,
            null_types=null_types,
            n_jobs=2,
            progress=False,
        )
        self.assertEqual(serial.subject_rows, parallel.subject_rows)
        self.assertEqual(serial.qc_rows, parallel.qc_rows)
        self.assertEqual(serial.summary_rows, parallel.summary_rows)
        # Ordering: unit-major then null_type order.
        self.assertEqual(
            [row["observation_id"] for row in serial.subject_rows],
            [f"obs-{i // len(null_types)}" for i in range(len(units) * len(null_types))],
        )
        self.assertEqual(
            [row["null_type"] for row in serial.subject_rows[: len(null_types)]],
            list(null_types),
        )

    def test_observed_cache_bit_identical(self) -> None:
        units = self._units(2)
        null_types = (
            NULL_TYPE_CIRCULAR_SHIFT,
            NULL_TYPE_PHASE_RANDOMIZATION,
            NULL_TYPE_AR1_INNOVATIONS,
        )
        cached = run_null_battery(
            units,
            n_surrogates=5,
            null_types=null_types,
            n_jobs=1,
            cache_observed=True,
            progress=False,
        )
        uncached = run_null_battery(
            units,
            n_surrogates=5,
            null_types=null_types,
            n_jobs=1,
            cache_observed=False,
            progress=False,
        )
        self.assertEqual(cached.subject_rows, uncached.subject_rows)
        self.assertEqual(cached.qc_rows, uncached.qc_rows)

    def test_checkpoint_resume_matches_fresh_run(self) -> None:
        units = self._units(3)
        null_types = (NULL_TYPE_CIRCULAR_SHIFT, NULL_TYPE_PHASE_RANDOMIZATION)
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpts"
            fresh = run_null_battery(
                units,
                n_surrogates=5,
                null_types=null_types,
                n_jobs=1,
                checkpoint_dir=ckpt,
                progress=False,
            )
            unit_files = sorted(ckpt.glob("unit_*.json"))
            self.assertGreaterEqual(len(unit_files), 3)
            unit_files[-1].unlink()
            resumed = run_null_battery(
                units,
                n_surrogates=5,
                null_types=null_types,
                n_jobs=1,
                checkpoint_dir=ckpt,
                progress=False,
            )
            self.assertEqual(resumed.subject_rows, fresh.subject_rows)
            self.assertEqual(resumed.qc_rows, fresh.qc_rows)

    def test_mismatched_checkpoint_contract_is_rejected(self) -> None:
        units = self._units(2)
        null_types = (NULL_TYPE_CIRCULAR_SHIFT,)
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpts"
            run_null_battery(
                units,
                n_surrogates=5,
                null_types=null_types,
                n_jobs=1,
                checkpoint_dir=ckpt,
                progress=False,
            )
            # Manifest mismatch (different n_surrogates) wipes and recomputes.
            recomputed = run_null_battery(
                units,
                n_surrogates=7,
                null_types=null_types,
                n_jobs=1,
                checkpoint_dir=ckpt,
                progress=False,
            )
            self.assertTrue(
                all(int(r["n_surrogates_requested"]) == 7 for r in recomputed.subject_rows)
            )

    def test_corrupt_final_csvs_without_marker_are_ignored(self) -> None:
        from ppg_eeg.confirmatory.nulls import (
            COMPLETE_MARKER_FILENAME,
            NULL_SUBJECT_RESULTS_FILENAME,
            write_null_outputs,
        )

        units = self._units(1)
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "C4"
            out.mkdir()
            # Stale partial CSV without complete marker.
            (out / NULL_SUBJECT_RESULTS_FILENAME).write_text(
                "dataset_id\nbogus\n", encoding="utf-8"
            )
            result = run_null_battery(
                units,
                n_surrogates=3,
                null_types=(NULL_TYPE_CIRCULAR_SHIFT,),
                n_jobs=1,
                progress=False,
            )
            write_null_outputs(
                result,
                out,
                n_surrogates=3,
                n_units=1,
                null_types=(NULL_TYPE_CIRCULAR_SHIFT,),
            )
            self.assertTrue((out / COMPLETE_MARKER_FILENAME).is_file())
            self.assertNotIn("bogus", (out / NULL_SUBJECT_RESULTS_FILENAME).read_text())


if __name__ == "__main__":
    unittest.main()
