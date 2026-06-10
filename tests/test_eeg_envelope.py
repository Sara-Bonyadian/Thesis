from __future__ import annotations

import unittest

import numpy as np

from ppg_eeg.temporal_coupling.eeg_envelope import (
    BandEnvelopeResult,
    compute_band_envelope,
    downsample_envelope_result,
    resolve_roi_channels,
)


class TestEegEnvelope(unittest.TestCase):
    def test_resolve_roi_channels_case_insensitive(self) -> None:
        available, missing = resolve_roi_channels(["Fz", "fcz", "Cz"], ("FZ", "FCz", "CPz"))
        self.assertEqual(available, ["Fz", "fcz"])
        self.assertEqual(missing, ["CPz"])

    def test_compute_band_envelope_tracks_amplitude_modulation(self) -> None:
        sfreq = 256.0
        duration_s = 8.0
        n_samples = int(sfreq * duration_s)
        t = np.arange(n_samples, dtype=float) / sfreq

        carrier_hz = 6.0
        mod_hz = 0.5
        amplitude = 1.0 + 0.8 * np.sin(2.0 * np.pi * mod_hz * t)
        signal = amplitude * np.sin(2.0 * np.pi * carrier_hz * t)

        envelope = compute_band_envelope(
            signal,
            sfreq=sfreq,
            l_freq=4.0,
            h_freq=8.0,
            smooth_s=0.5,
        )

        # Drop filter edge transients before comparing to the known AM envelope.
        valid = slice(int(2 * sfreq), int(6 * sfreq))
        env_valid = envelope[valid]
        amp_valid = amplitude[valid]
        corr = float(np.corrcoef(env_valid, amp_valid)[0, 1])
        self.assertGreater(corr, 0.85)
        self.assertGreater(float(np.nanmax(env_valid)), float(np.nanmin(env_valid)))

    def test_downsample_envelope_result_reduces_sampling_rate(self) -> None:
        sfreq = 100.0
        duration_s = 10.0
        n_samples = int(sfreq * duration_s) + 1
        time_s = np.arange(n_samples, dtype=float) / sfreq
        envelopes = {
            "theta": np.sin(2.0 * np.pi * 0.5 * time_s),
            "alpha": np.cos(2.0 * np.pi * 0.5 * time_s),
            "beta": np.sin(2.0 * np.pi * 1.0 * time_s),
        }
        result = BandEnvelopeResult(time_s=time_s, envelopes=envelopes, warnings=())

        downsampled = downsample_envelope_result(result, output_fs_hz=10.0)
        step = float(np.median(np.diff(downsampled.time_s)))
        self.assertAlmostEqual(step, 0.1, places=2)
        self.assertEqual(len(downsampled.time_s), 101)
        self.assertAlmostEqual(float(downsampled.time_s[-1]), duration_s, places=2)
        for band in ("theta", "alpha", "beta"):
            self.assertFalse(np.any(np.isnan(downsampled.envelopes[band])))


if __name__ == "__main__":
    unittest.main()
