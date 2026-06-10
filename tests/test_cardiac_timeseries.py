from __future__ import annotations

import unittest

import numpy as np

from ppg_eeg.ppg import rmssd_ms, sdnn_ms
from ppg_eeg.temporal_coupling.cardiac_common import (
    annotate_peaks,
    classify_ibi,
    infer_signal_type,
)
from ppg_eeg.temporal_coupling.cardiac_timeseries import _beats_and_ibis_in_window


class TestCardiacTimeseries(unittest.TestCase):
    def test_rmssd_and_sdnn_on_synthetic_ibi(self) -> None:
        ibi_ms = np.array([800.0, 820.0, 790.0, 810.0, 805.0], dtype=float)
        diffs = np.diff(ibi_ms)
        expected_rmssd = float(np.sqrt(np.mean(diffs**2)))
        expected_sdnn = float(np.std(ibi_ms, ddof=1))

        self.assertAlmostEqual(rmssd_ms(ibi_ms), expected_rmssd, places=6)
        self.assertAlmostEqual(sdnn_ms(ibi_ms), expected_sdnn, places=6)

    def test_beats_and_ibis_in_window_filters_out_of_range(self) -> None:
        peak_times_s = np.array([1.0, 1.8, 2.5, 3.0, 10.0, 10.5], dtype=float)

        ibi_ms, n_beats = _beats_and_ibis_in_window(
            peak_times_s,
            window_start=0.5,
            window_end=3.5,
            ibi_min_ms=400.0,
            ibi_max_ms=1200.0,
        )
        self.assertEqual(n_beats, 4)
        np.testing.assert_allclose(ibi_ms, np.array([800.0, 700.0, 500.0]))

    def test_classify_ibi_reasons(self) -> None:
        self.assertEqual(classify_ibi(300, prev_ibi_ms=None, median_jump=10, ibi_min_ms=400, ibi_max_ms=1200)[1], "too_short")
        self.assertEqual(classify_ibi(1300, prev_ibi_ms=None, median_jump=10, ibi_min_ms=400, ibi_max_ms=1200)[1], "too_long")
        ok, reason = classify_ibi(800, prev_ibi_ms=810, median_jump=10, ibi_min_ms=400, ibi_max_ms=1200)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_annotate_peaks_marks_first_peak(self) -> None:
        peaks = np.array([1.0, 1.8, 2.6], dtype=float)
        peak_samples = np.array([1000, 1800, 2600], dtype=int)
        signal_norm = np.linspace(0.0, 1.0, 3000)
        annotations = annotate_peaks(
            peaks,
            peak_samples,
            signal_norm,
            segment_start_s=0.0,
            sfreq=1000.0,
            ibi_min_ms=400.0,
            ibi_max_ms=1200.0,
        )
        self.assertEqual(len(annotations), 3)
        self.assertEqual(annotations[0].cleaning_reason, "first_peak")
        self.assertAlmostEqual(annotations[1].peak_y, signal_norm[1800])

    def test_infer_signal_type(self) -> None:
        self.assertEqual(infer_signal_type("ECG"), "ecg")
        self.assertEqual(infer_signal_type("PPG"), "ppg")


if __name__ == "__main__":
    unittest.main()
