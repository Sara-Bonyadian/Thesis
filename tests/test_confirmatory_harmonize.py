from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.temporal_coupling.confirmatory.harmonize import (
    ALIGNMENT_QC_FILENAME,
    ALIGNED_FEATURES_TEMPLATE,
    MAX_GAP_S,
    NESTED_DURATIONS_S,
    SEGMENT_MANIFEST_FILENAME,
    SENSITIVITY_DURATION_S,
    contiguous_blocks,
    harmonize_observation,
    select_longest_block,
    write_harmonize_outputs,
)
from ppg_eeg.temporal_coupling.confirmatory.multitaper_power import BANDS_HZ


BANDS = tuple(BANDS_HZ)


def _series(n: int, start: float = 0.0) -> np.ndarray:
    return start + np.arange(n, dtype=float)


def _eeg_maps(
    n: int,
    *,
    theta: float = 4.0,
    alpha: float = 3.0,
    beta: float = 2.0,
    low_gamma: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    absolute = {
        "theta": np.full(n, theta, dtype=float),
        "alpha": np.full(n, alpha, dtype=float),
        "beta": np.full(n, beta, dtype=float),
        "low_gamma": np.full(n, low_gamma, dtype=float),
    }
    abs_log = {band: np.log10(values) for band, values in absolute.items()}
    return absolute, abs_log


def _identity() -> dict[str, str]:
    return {
        "dataset_id": "ds_test",
        "subject_id": "sub-01",
        "task": "rest",
        "condition": "rest",
        "observation_id": "obs-01",
    }


class TestConfirmatoryHarmonize(unittest.TestCase):
    def test_exact_alignment_uses_only_common_valid_times(self) -> None:
        hr_time = _series(10)
        eeg_time = _series(10)
        hr_valid = np.array(
            [True, True, False, True, True, True, True, True, True, True]
        )
        eeg_valid = np.array(
            [True, True, True, True, False, True, True, True, True, True]
        )
        absolute, abs_log = _eeg_maps(10)
        result = harmonize_observation(
            hr_time_s=hr_time,
            hr_bpm=np.linspace(60.0, 69.0, 10),
            hr_valid=hr_valid,
            eeg_time_s=eeg_time,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            eeg_valid=eeg_valid,
            identity=_identity(),
        )
        self.assertEqual(result.manifest["n_common_support"], 8)
        self.assertIsNotNone(result.selected_block)
        assert result.selected_block is not None
        self.assertNotIn(2.0, result.selected_block.time_s)
        self.assertNotIn(4.0, result.selected_block.time_s)

    def test_nested_segments_share_center(self) -> None:
        n = 300
        times = _series(n, start=100.0)
        absolute, abs_log = _eeg_maps(n)
        hr = 70.0 + 0.01 * np.arange(n)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=hr,
            hr_valid=np.ones(n, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        anchors = [result.segments[duration].center_s for duration in NESTED_DURATIONS_S]
        self.assertTrue(all(np.isclose(center, anchors[0]) for center in anchors))
        self.assertTrue(result.segments[240].eligible)
        self.assertTrue(result.segments[180].eligible)
        self.assertTrue(result.segments[120].eligible)
        self.assertTrue(result.segments[SENSITIVITY_DURATION_S].eligible)
        self.assertEqual(result.segments[SENSITIVITY_DURATION_S].role, "separate_sensitivity")
        self.assertEqual(result.segments[180].role, "nested_sensitivity")
        # Nested lengths are exact and lie inside the selected block.
        for duration in (240, 180, 120, 60):
            segment = result.segments[duration]
            self.assertEqual(segment.n_samples, duration)
            self.assertGreaterEqual(segment.start_s, result.selected_block.start_s)  # type: ignore[union-attr]
            self.assertLessEqual(segment.end_s, result.selected_block.end_s)  # type: ignore[union-attr]
            self.assertEqual(len(result.features_by_duration[duration]), duration)

    def test_gap_split_and_longest_block_selection(self) -> None:
        left = _series(100, start=0.0)
        right = _series(150, start=200.0)
        times = np.concatenate([left, right])
        blocks = contiguous_blocks(times, max_gap_s=MAX_GAP_S)
        self.assertEqual(len(blocks), 2)
        selected = select_longest_block(blocks)
        assert selected is not None
        self.assertEqual(selected.start_s, 200.0)
        self.assertEqual(selected.n_samples, 150)

        absolute, abs_log = _eeg_maps(times.size)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=np.full(times.size, 72.0),
            hr_valid=np.ones(times.size, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        self.assertEqual(result.selected_block.start_s, 200.0)  # type: ignore[union-attr]
        self.assertEqual(result.manifest["n_blocks"], 2)

    def test_earliest_tie_break(self) -> None:
        first = _series(80, start=0.0)
        second = _series(80, start=200.0)
        times = np.concatenate([first, second])
        selected = select_longest_block(contiguous_blocks(times))
        assert selected is not None
        self.assertEqual(selected.start_s, 0.0)

        absolute, abs_log = _eeg_maps(times.size)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=np.full(times.size, 65.0),
            hr_valid=np.ones(times.size, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        self.assertEqual(result.selected_block.start_s, 0.0)  # type: ignore[union-attr]
        # Support 80 s: nested 240/180/120 ineligible; D60 sensitivity still ok.
        self.assertFalse(result.segments[240].eligible)
        self.assertFalse(result.segments[180].eligible)
        self.assertFalse(result.segments[120].eligible)
        self.assertTrue(result.segments[60].eligible)
        self.assertEqual(
            result.segments[240].exclusion_reason, "insufficient_clean_support"
        )

    def test_insufficient_duration_excludes_all_segments(self) -> None:
        n = 50
        times = _series(n)
        absolute, abs_log = _eeg_maps(n)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=np.full(n, 60.0),
            hr_valid=np.ones(n, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        for duration in (240, 180, 120, 60):
            self.assertFalse(result.segments[duration].eligible)
            self.assertEqual(result.features_by_duration[duration], ())
            qc = next(row for row in result.qc_rows if row["duration_s"] == duration)
            self.assertFalse(qc["eligible"])
            self.assertEqual(qc["exclusion_reason"], "insufficient_clean_support")

    def test_relative_and_residualized_representations_and_zscore(self) -> None:
        n = 240
        times = _series(n)
        t = np.arange(n, dtype=float)
        absolute = {
            "theta": 2.0 + 0.5 * np.sin(t / 11.0) + 0.2 * np.cos(t / 7.0),
            "alpha": 3.0 + 0.4 * np.cos(t / 13.0),
            "beta": 4.0 + 0.3 * np.sin(t / 17.0),
            "low_gamma": 5.0 + 0.25 * np.cos(t / 19.0),
        }
        abs_log = {band: np.log10(values) for band, values in absolute.items()}
        hr = 60.0 + 0.1 * np.arange(n)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=hr,
            hr_valid=np.ones(n, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        rows = result.features_by_duration[240]
        self.assertEqual(len(rows), 240)
        relative_sums = [
            sum(float(row[f"{band}_relative_power"]) for band in BANDS) for row in rows
        ]
        self.assertTrue(np.allclose(relative_sums, 1.0))
        hr_z = np.asarray([float(row["hr_z"]) for row in rows], dtype=float)
        self.assertAlmostEqual(float(np.mean(hr_z)), 0.0, places=10)
        self.assertAlmostEqual(float(np.std(hr_z, ddof=0)), 1.0, places=10)
        theta_resid = np.asarray(
            [float(row["theta_broadband_residualized_log10"]) for row in rows],
            dtype=float,
        )
        theta_log = np.asarray(
            [float(row["theta_absolute_log10_power"]) for row in rows],
            dtype=float,
        )
        broadband = np.mean(
            np.vstack([np.log10(absolute[band]) for band in BANDS]),
            axis=0,
        )
        self.assertGreater(float(np.std(theta_resid, ddof=0)), 1e-6)
        self.assertFalse(np.allclose(theta_resid, theta_log))
        # Residualization is fit within the selected segment times.
        corr = np.corrcoef(theta_resid, broadband)[0, 1]
        self.assertAlmostEqual(float(corr), 0.0, places=8)

    def test_deterministic_outputs_and_schemas(self) -> None:
        n = 260
        times = _series(n, start=10.0)
        absolute, abs_log = _eeg_maps(n, theta=8.0, alpha=4.0, beta=2.0, low_gamma=1.0)
        kwargs = dict(
            hr_time_s=times,
            hr_bpm=70.0 + 0.05 * np.arange(n),
            hr_valid=np.ones(n, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        first = harmonize_observation(**kwargs)
        second = harmonize_observation(**kwargs)
        self.assertEqual(first.manifest, second.manifest)
        self.assertEqual(first.features_by_duration[240], second.features_by_duration[240])

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = write_harmonize_outputs(first, out)
            for duration in (240, 180, 120, 60):
                path = (out / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)).resolve()
                self.assertTrue(path.exists())
                self.assertEqual(paths[f"features_D{duration}"], path)
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), duration)
                self.assertIn("hr_z", rows[0])
                self.assertIn("theta_absolute_log10_power", rows[0])
                self.assertIn("theta_relative_power", rows[0])
                self.assertIn("theta_broadband_residualized_log10", rows[0])
                self.assertIn("theta_absolute_log10_power_z", rows[0])

            manifest_path = out / SEGMENT_MANIFEST_FILENAME
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("segments", manifest)
            self.assertIn("selected_block", manifest)
            self.assertEqual(manifest["segments"]["240"]["n_samples"], 240)

            qc_path = out / ALIGNMENT_QC_FILENAME
            self.assertTrue(qc_path.exists())
            with qc_path.open(encoding="utf-8", newline="") as handle:
                qc_rows = list(csv.DictReader(handle))
            self.assertEqual(len(qc_rows), 4)
            self.assertIn("available_support_s", qc_rows[0])
            self.assertIn("n_missing_within_segment", qc_rows[0])
            self.assertIn("exclusion_reason", qc_rows[0])

            again = write_harmonize_outputs(second, out)
            for key, path in paths.items():
                self.assertEqual(
                    path.read_bytes(),
                    again[key].read_bytes(),
                )

    def test_no_common_support_records_exclusion(self) -> None:
        times = _series(120)
        absolute, abs_log = _eeg_maps(120)
        result = harmonize_observation(
            hr_time_s=times,
            hr_bpm=np.full(120, 60.0),
            hr_valid=np.zeros(120, dtype=bool),
            eeg_time_s=times,
            eeg_absolute_power=absolute,
            eeg_absolute_log10_power=abs_log,
            identity=_identity(),
        )
        self.assertIsNone(result.selected_block)
        for duration in (240, 180, 120, 60):
            self.assertEqual(
                result.segments[duration].exclusion_reason, "no_common_support"
            )


if __name__ == "__main__":
    unittest.main()
