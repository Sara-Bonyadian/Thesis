from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.multitaper_power import (
    N_TAPERS,
    TIME_BANDWIDTH,
    compute_multitaper_power,
    write_multitaper_outputs,
)


def _sine(
    frequency_hz: float,
    *,
    sfreq: float,
    duration_s: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    time = np.arange(round(sfreq * duration_s), dtype=float) / sfreq
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def _median_features(result: object) -> dict[tuple[int, str], float]:
    return {
        (feature.window_index, feature.band): feature.absolute_log10_power
        for feature in result.features  # type: ignore[attr-defined]
        if feature.aggregation == "robust_median"
    }


class TestMultitaperPower(unittest.TestCase):
    def test_known_sine_mixtures_peak_in_expected_bands(self) -> None:
        sfreq = 200.0
        duration = 8.0
        data = np.vstack(
            [
                _sine(5.0, sfreq=sfreq, duration_s=duration, amplitude=3.0)
                + _sine(10.0, sfreq=sfreq, duration_s=duration, amplitude=0.2),
                _sine(10.0, sfreq=sfreq, duration_s=duration, amplitude=3.0)
                + _sine(20.0, sfreq=sfreq, duration_s=duration, amplitude=0.2),
                _sine(20.0, sfreq=sfreq, duration_s=duration, amplitude=3.0)
                + _sine(35.0, sfreq=sfreq, duration_s=duration, amplitude=0.2),
                _sine(35.0, sfreq=sfreq, duration_s=duration, amplitude=3.0),
            ]
        )
        result = compute_multitaper_power(
            data,
            sfreq=sfreq,
            ch_names=["Theta", "Alpha", "Beta", "Gamma"],
            line_frequency_hz=None,
        )

        first_window = {
            (feature.channel, feature.band): feature.absolute_log10_power
            for feature in result.features
            if feature.window_index == 0 and feature.aggregation == "channel"
        }
        self.assertGreater(
            first_window[("Theta", "theta")],
            first_window[("Theta", "alpha")],
        )
        self.assertGreater(
            first_window[("Alpha", "alpha")],
            first_window[("Alpha", "theta")],
        )
        self.assertGreater(
            first_window[("Beta", "beta")],
            first_window[("Beta", "alpha")],
        )
        self.assertGreater(
            first_window[("Gamma", "low_gamma")],
            first_window[("Gamma", "beta")],
        )
        self.assertEqual(result.qc.time_bandwidth, TIME_BANDWIDTH)
        self.assertEqual(result.qc.n_tapers, N_TAPERS)
        self.assertTrue(result.qc.all_bands_supported)
        self.assertTrue(result.qc.spectral_qc_passed)
        self.assertEqual(result.qc.frequency_resolution_hz, 0.5)

    def test_missing_channels_are_recorded_and_not_fabricated(self) -> None:
        sfreq = 200.0
        data = np.vstack(
            [
                _sine(10, sfreq=sfreq, duration_s=4),
                _sine(20, sfreq=sfreq, duration_s=4),
            ]
        )
        result = compute_multitaper_power(
            data,
            sfreq=sfreq,
            ch_names=["Fz", "Pz"],
            clean_channels=["Fz", "Cz"],
        )

        self.assertEqual(result.qc.usable_channels, "Fz")
        self.assertEqual(result.qc.missing_channels, "Cz")
        self.assertIn("missing_requested_channels", result.qc.warning)
        channels = {feature.channel for feature in result.features}
        self.assertEqual(channels, {"Fz", "__robust_median__"})

    def test_short_recording_has_qc_but_no_features(self) -> None:
        sfreq = 200.0
        result = compute_multitaper_power(
            np.vstack([_sine(10, sfreq=sfreq, duration_s=1.99)]),
            sfreq=sfreq,
            ch_names=["Fz"],
        )
        self.assertEqual(result.features, ())
        self.assertEqual(result.qc.status, "short_recording")
        self.assertEqual(result.qc.n_windows, 0)

    def test_line_noise_notch_reduces_recorded_contamination(self) -> None:
        sfreq = 250.0
        signal = _sine(10, sfreq=sfreq, duration_s=8, amplitude=1.0)
        contaminated = signal + _sine(
            60, sfreq=sfreq, duration_s=8, amplitude=5.0
        )
        result = compute_multitaper_power(
            contaminated[np.newaxis, :],
            sfreq=sfreq,
            ch_names=["Fz"],
            line_frequency_hz=60.0,
            apply_line_notch=True,
        )

        self.assertEqual(result.qc.line_noise_handling, "notch_60_hz_q30")
        self.assertIsNotNone(result.qc.line_noise_ratio_before)
        self.assertIsNotNone(result.qc.line_noise_ratio_after)
        self.assertLess(
            result.qc.line_noise_ratio_after or 1.0,
            (result.qc.line_noise_ratio_before or 0.0) * 0.1,
        )

    def test_two_second_windows_with_one_second_step_use_center_times(self) -> None:
        sfreq = 200.0
        data = np.vstack(
            [
                _sine(10, sfreq=sfreq, duration_s=6),
                _sine(12, sfreq=sfreq, duration_s=6),
            ]
        )
        result = compute_multitaper_power(
            data,
            sfreq=sfreq,
            ch_names=["Fz", "Pz"],
        )

        self.assertEqual(result.qc.n_windows, 5)
        medians = [
            feature
            for feature in result.features
            if feature.aggregation == "robust_median"
            and feature.band == "alpha"
        ]
        self.assertEqual(
            [feature.window_start_s for feature in medians],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [feature.window_center_s for feature in medians],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            [feature.window_end_s for feature in medians],
            [2, 3, 4, 5, 6],
        )

    def test_robust_median_matches_channel_log_power_median(self) -> None:
        sfreq = 200.0
        data = np.vstack(
            [
                _sine(10, sfreq=sfreq, duration_s=4, amplitude=1),
                _sine(10, sfreq=sfreq, duration_s=4, amplitude=2),
                _sine(10, sfreq=sfreq, duration_s=4, amplitude=10),
            ]
        )
        result = compute_multitaper_power(
            data,
            sfreq=sfreq,
            ch_names=["Fz", "Cz", "Pz"],
        )
        channels = [
            feature.absolute_log10_power
            for feature in result.features
            if feature.window_index == 0
            and feature.band == "alpha"
            and feature.aggregation == "channel"
        ]
        median = _median_features(result)[(0, "alpha")]
        self.assertAlmostEqual(median, float(np.median(channels)))


class TestMultitaperOutputs(unittest.TestCase):
    def test_writes_feature_and_qc_schemas(self) -> None:
        sfreq = 200.0
        result = compute_multitaper_power(
            np.vstack(
                [
                    _sine(10, sfreq=sfreq, duration_s=3),
                    _sine(20, sfreq=sfreq, duration_s=3),
                ]
            ),
            sfreq=sfreq,
            ch_names=["Fz", "Pz"],
            rejected_channels=["Fp1"],
            identity={
                "dataset_id": "synthetic",
                "subject_id": "sub-001",
                "task": "rest",
                "condition": "rest",
                "observation_id": "synthetic-sub-001-rest",
            },
        )
        with TemporaryDirectory() as tmp:
            features_path, qc_path = write_multitaper_outputs(result, tmp)
            self.assertEqual(
                features_path.name, "features_multitaper_power.csv"
            )
            self.assertEqual(qc_path.name, "multitaper_qc.csv")

            with features_path.open(newline="", encoding="utf-8") as handle:
                features = list(csv.DictReader(handle))
            self.assertTrue(features)
            self.assertTrue(
                {
                    "window_start_s",
                    "window_center_s",
                    "window_end_s",
                    "channel",
                    "aggregation",
                    "band",
                    "absolute_power",
                    "absolute_log10_power",
                }.issubset(features[0])
            )
            with qc_path.open(newline="", encoding="utf-8") as handle:
                qc = list(csv.DictReader(handle))
            self.assertEqual(qc[0]["rejected_channels"], "Fp1")
            self.assertEqual(qc[0]["window_s"], "2.0")
            self.assertEqual(qc[0]["step_s"], "1.0")


if __name__ == "__main__":
    unittest.main()
