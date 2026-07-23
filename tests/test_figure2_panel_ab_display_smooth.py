"""Figure 2 Panels A/B display-only Gaussian smoothing guards."""

from __future__ import annotations

import unittest

import numpy as np

from ppg_eeg.confirmatory.figures import (
    FIGURE2_PANEL_AB_DISPLAY_SMOOTH_SIGMA_S,
    LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S,
    gaussian_smooth_display_series,
)


class DisplaySmoothTests(unittest.TestCase):
    def test_sigma_matches_configured_display_smooth(self) -> None:
        self.assertEqual(LAG_CURVE_DISPLAY_SMOOTH_SIGMA_S, 2.0)
        self.assertEqual(FIGURE2_PANEL_AB_DISPLAY_SMOOTH_SIGMA_S, 2.0)

    def test_smooth_reduces_high_frequency_jitter(self) -> None:
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 4 * np.pi, 121)) + 0.4 * rng.normal(size=121)
        y = gaussian_smooth_display_series(x, sigma_s=1.0, lag_step_s=1.0)
        self.assertEqual(y.shape, x.shape)
        self.assertLess(float(np.std(np.diff(y))), float(np.std(np.diff(x))))

    def test_does_not_mutate_input(self) -> None:
        x = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
        x_copy = x.copy()
        _ = gaussian_smooth_display_series(x, sigma_s=1.0)
        np.testing.assert_array_equal(x, x_copy)

    def test_preserves_nan_positions(self) -> None:
        x = np.asarray([0.0, np.nan, 1.0, 0.5, np.nan], dtype=float)
        y = gaussian_smooth_display_series(x, sigma_s=1.0)
        self.assertTrue(np.isnan(y[1]))
        self.assertTrue(np.isnan(y[4]))
        self.assertTrue(np.isfinite(y[0]))
        self.assertTrue(np.isfinite(y[2]))


if __name__ == "__main__":
    unittest.main()
