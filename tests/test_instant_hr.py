from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.instant_hr import (
    reconstruct_instant_hr,
    reconstruct_instant_hr_file,
)


def _rows(
    times: list[float],
    *,
    accepted: list[bool] | None = None,
) -> list[dict[str, object]]:
    flags = accepted if accepted is not None else [True] * len(times)
    return [
        {
            "dataset_id": "synthetic",
            "subject_id": "sub-001",
            "task": "rest",
            "observation_id": "synthetic-sub-001-task-rest",
            "peak_time_s": time_s,
            "peak_sample": round(time_s * 100),
            "is_clean_ibi": is_accepted,
            "is_accepted_peak": is_accepted,
            "cleaning_reason": "ok" if is_accepted else "rejected",
        }
        for time_s, is_accepted in zip(times, flags, strict=True)
    ]


class TestInstantHRReconstruction(unittest.TestCase):
    def test_constant_hr_is_60_bpm_at_later_beats(self) -> None:
        result = reconstruct_instant_hr(_rows([0, 1, 2, 3, 4, 5]))

        self.assertEqual(result.qc.status, "ok")
        self.assertEqual(result.qc.n_accepted_beats, 6)
        self.assertEqual([row.time_s for row in result.features], [1, 2, 3, 4, 5])
        np.testing.assert_allclose(
            [row.instant_hr_bpm for row in result.features],
            np.full(5, 60.0),
        )
        first = result.features[0]
        self.assertEqual(first.source_beat_index_left, 1)
        self.assertEqual(first.source_beat_index_right, 1)
        self.assertFalse(first.is_interpolated)

    def test_changing_hr_matches_consecutive_ibi_values_at_beats(self) -> None:
        result = reconstruct_instant_hr(_rows([0, 1, 3, 4, 5]))
        by_time = {row.time_s: row for row in result.features}

        self.assertAlmostEqual(by_time[1].instant_hr_bpm or 0.0, 60.0)
        self.assertAlmostEqual(by_time[3].instant_hr_bpm or 0.0, 30.0)
        self.assertAlmostEqual(by_time[4].instant_hr_bpm or 0.0, 60.0)
        self.assertAlmostEqual(by_time[5].instant_hr_bpm or 0.0, 60.0)

    def test_rejected_beats_are_removed_and_source_indices_preserved(self) -> None:
        result = reconstruct_instant_hr(
            _rows(
                [0, 1, 1.5, 2, 3],
                accepted=[True, True, False, True, True],
            )
        )
        by_time = {row.time_s: row for row in result.features}

        self.assertEqual(result.qc.n_source_rows, 5)
        self.assertEqual(result.qc.n_accepted_beats, 4)
        self.assertEqual(result.qc.n_rejected_beats, 1)
        self.assertAlmostEqual(by_time[2].instant_hr_bpm or 0.0, 60.0)
        self.assertEqual(by_time[2].source_beat_index_left, 3)
        self.assertEqual(by_time[2].source_peak_sample_left, 200)

    def test_long_beat_gap_is_masked_without_pchip_bridging(self) -> None:
        result = reconstruct_instant_hr(_rows([0, 1, 2, 10, 11, 12]))
        by_time = {row.time_s: row for row in result.features}

        self.assertEqual(result.qc.n_long_gaps, 1)
        self.assertEqual(result.qc.n_gap_masked_samples, 8)
        self.assertEqual(result.qc.clean_beat_span_s, 2.0)
        for time_s in range(3, 11):
            self.assertTrue(by_time[time_s].is_gap_masked)
            self.assertFalse(by_time[time_s].is_valid_hr)
            self.assertIsNone(by_time[time_s].instant_hr_bpm)
            self.assertIsNone(by_time[time_s].source_beat_index_left)
        self.assertAlmostEqual(by_time[2].instant_hr_bpm or 0.0, 60.0)
        self.assertAlmostEqual(by_time[11].instant_hr_bpm or 0.0, 60.0)

    def test_insufficient_beats_produces_qc_and_no_features(self) -> None:
        result = reconstruct_instant_hr(_rows([0, 1]))

        self.assertEqual(result.features, ())
        self.assertEqual(result.qc.status, "insufficient_beats")
        self.assertEqual(result.qc.n_accepted_beats, 2)
        self.assertEqual(result.qc.n_grid_samples, 0)

    def test_grid_never_extrapolates_beyond_hr_support(self) -> None:
        result = reconstruct_instant_hr(_rows([0.2, 1.2, 2.2, 3.2]))

        self.assertEqual([row.time_s for row in result.features], [2.0, 3.0])
        self.assertGreaterEqual(
            min(row.time_s for row in result.features),
            result.qc.hr_support_start_s or 0.0,
        )
        self.assertLessEqual(
            max(row.time_s for row in result.features),
            result.qc.hr_support_end_s or float("inf"),
        )
        self.assertTrue(all(row.is_interpolated for row in result.features))
        interpolated = result.features[0]
        self.assertEqual(interpolated.source_beat_index_left, 1)
        self.assertEqual(interpolated.source_beat_index_right, 2)


class TestInstantHRFiles(unittest.TestCase):
    def test_reads_detected_peaks_and_writes_feature_and_qc_schemas(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            peaks_path = root / "detected_peaks.csv"
            rows = _rows([0, 1, 2, 3])
            with peaks_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            features_path, qc_path = reconstruct_instant_hr_file(
                peaks_path, root / "confirmatory"
            )

            self.assertEqual(features_path.name, "features_instant_hr.csv")
            self.assertEqual(qc_path.name, "instant_hr_qc.csv")
            with features_path.open(newline="", encoding="utf-8") as handle:
                features = list(csv.DictReader(handle))
            self.assertEqual(len(features), 3)
            self.assertTrue(
                {
                    "time_s",
                    "instant_hr_bpm",
                    "is_valid_hr",
                    "is_interpolated",
                    "is_gap_masked",
                    "source_beat_index_left",
                    "source_beat_index_right",
                }.issubset(features[0])
            )
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc = list(csv.DictReader(handle))
            self.assertEqual(len(qc), 1)
            self.assertEqual(qc[0]["status"], "ok")
            self.assertEqual(qc[0]["n_accepted_beats"], "4")

    def test_insufficient_beats_still_writes_header_and_qc(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            peaks_path = root / "detected_peaks.csv"
            rows = _rows([0, 1])
            with peaks_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            features_path, qc_path = reconstruct_instant_hr_file(
                peaks_path, root / "confirmatory"
            )
            with features_path.open(newline="", encoding="utf-8") as handle:
                features = list(csv.DictReader(handle))
            self.assertEqual(features, [])
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc = list(csv.DictReader(handle))
            self.assertEqual(qc[0]["status"], "insufficient_beats")


if __name__ == "__main__":
    unittest.main()
