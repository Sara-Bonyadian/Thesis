from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.endpoints import fisher_z
from ppg_eeg.confirmatory.peak_model import (
    FWHM_FACTOR,
    MIN_IDENTIFIABLE_A,
    MU_BOUND_S,
    PARAMS_FILENAME,
    QC_FILENAME,
    SIGMA_MAX_S,
    SIGMA_MIN_S,
    compute_peak_fits_from_curves,
    fit_gaussian_peak,
    fwhm_from_sigma,
    gaussian_peak,
    write_peak_fit_outputs,
)


def _lags(duration_s: int = 240) -> np.ndarray:
    lag_max = {240: 60, 180: 60, 120: 30, 60: 20}[duration_s]
    return np.arange(-lag_max, lag_max + 1, dtype=float)


def _curve_rows(
    lags: np.ndarray,
    r_values: np.ndarray,
    *,
    duration_s: int = 240,
    n_overlap: int = 120,
    band: str = "theta",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lag, r in zip(lags, r_values, strict=True):
        rows.append(
            {
                "dataset_id": "ds_test",
                "subject_id": "sub-01",
                "task": "rest",
                "condition": "rest",
                "observation_id": "obs-01",
                "duration_s": duration_s,
                "duration_role": "primary",
                "endpoint_name": "zlpi",
                "band": band,
                "power_representation": "absolute_log10",
                "is_primary_representation": True,
                "pair": f"hr_x_{band}_absolute_log10",
                "lag_s": float(lag),
                "r": float(r),
                "n_overlap": int(n_overlap),
            }
        )
    return rows


def _r_from_z_peak(
    lags: np.ndarray,
    *,
    baseline_C: float,
    peak_height_A: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    z = gaussian_peak(lags, baseline_C, peak_height_A, mu, sigma)
    # Invert Fisher-z approximately for synthetic input rows (r = tanh(z)).
    return np.tanh(np.asarray(z, dtype=float))


class TestGaussianPeakHelpers(unittest.TestCase):
    def test_fwhm_factor(self) -> None:
        self.assertAlmostEqual(fwhm_from_sigma(10.0), FWHM_FACTOR * 10.0, places=12)


class TestPeakModelFits(unittest.TestCase):
    def test_recovers_known_zero_centered_peak(self) -> None:
        lags = _lags(240)
        z = gaussian_peak(lags, 0.05, 0.40, 0.0, 8.0)
        # Fit directly in z-space to isolate optimizer recovery.
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        self.assertTrue(fit["has_identifiable_peak"])
        self.assertTrue(fit["report_timing_shift"])
        self.assertAlmostEqual(float(fit["peak_height_A"]), 0.40, places=2)
        self.assertAlmostEqual(float(fit["peak_center_mu_s"]), 0.0, places=2)
        self.assertAlmostEqual(float(fit["sigma_s"]), 8.0, places=2)
        self.assertAlmostEqual(
            float(fit["fwhm_s"]), FWHM_FACTOR * float(fit["sigma_s"]), places=10
        )

    def test_recovers_shifted_peak(self) -> None:
        lags = _lags(240)
        z = gaussian_peak(lags, 0.0, 0.35, 7.0, 6.0)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 100.0))
        self.assertTrue(fit["report_timing_shift"])
        self.assertAlmostEqual(float(fit["peak_center_mu_s"]), 7.0, places=2)
        self.assertAlmostEqual(float(fit["peak_height_A"]), 0.35, places=2)

    def test_recovers_broad_peak(self) -> None:
        lags = _lags(240)
        z = gaussian_peak(lags, -0.02, 0.25, -3.0, 25.0)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        self.assertAlmostEqual(float(fit["sigma_s"]), 25.0, delta=1.0)
        self.assertAlmostEqual(float(fit["peak_center_mu_s"]), -3.0, delta=0.5)

    def test_flat_curve_has_no_reportable_timing(self) -> None:
        lags = _lags(180)
        z = np.full(lags.shape, 0.12)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 60.0))
        self.assertTrue(fit["converged"])
        self.assertLess(float(fit["peak_height_A"]), MIN_IDENTIFIABLE_A)
        self.assertFalse(fit["has_identifiable_peak"])
        self.assertFalse(fit["report_timing_shift"])
        self.assertTrue(math.isnan(float(fit["peak_center_mu_s"])))
        self.assertEqual(fit["exclusion_reason"], "no_identifiable_positive_peak")

    def test_noisy_curve_still_recovers_approximate_center(self) -> None:
        rng = np.random.default_rng(123)
        lags = _lags(240)
        z = gaussian_peak(lags, 0.0, 0.5, -5.0, 7.0) + rng.normal(0.0, 0.02, size=lags.shape)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        self.assertTrue(fit["has_identifiable_peak"])
        self.assertAlmostEqual(float(fit["peak_center_mu_s"]), -5.0, delta=1.0)
        self.assertGreater(float(fit["peak_height_A"]), 0.3)

    def test_failed_fit_insufficient_points(self) -> None:
        fit = fit_gaussian_peak([0.0, 1.0], [0.1, 0.2], weights=[10.0, 10.0])
        self.assertFalse(fit["converged"])
        self.assertEqual(fit["exclusion_reason"], "insufficient_finite_lags")
        self.assertFalse(fit["report_timing_shift"])

    def test_mu_boundary_case(self) -> None:
        lags = _lags(240)
        # Peak far outside allowed mu window; optimizer should hit ±20 bound.
        z = gaussian_peak(lags, 0.0, 0.8, 40.0, 5.0)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        if fit["has_identifiable_peak"]:
            self.assertLessEqual(abs(float(fit["peak_center_mu_s"])), MU_BOUND_S + 1e-8)
            self.assertTrue(fit["mu_at_bound"] or abs(float(fit["peak_center_mu_s"])) > 15.0)

    def test_sigma_bounds_respected(self) -> None:
        lags = _lags(240)
        z = gaussian_peak(lags, 0.0, 0.5, 0.0, 0.2)  # extremely sharp
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        self.assertGreaterEqual(float(fit["sigma_s"]), SIGMA_MIN_S - 1e-8)
        self.assertLessEqual(float(fit["sigma_s"]), SIGMA_MAX_S + 1e-8)

    def test_negative_amplitude_not_allowed(self) -> None:
        lags = _lags(240)
        # Deep trough; constrained A>=0 should not produce negative height.
        z = -gaussian_peak(lags, 0.0, 0.6, 0.0, 6.0)
        fit = fit_gaussian_peak(lags, z, weights=np.full(lags.shape, 120.0))
        self.assertTrue(fit["converged"])
        self.assertGreaterEqual(float(fit["peak_height_A"]), -1e-10)
        self.assertFalse(fit["report_timing_shift"])

    def test_curve_rows_pipeline_and_outputs(self) -> None:
        lags = _lags(240)
        r = _r_from_z_peak(lags, baseline_C=0.0, peak_height_A=0.3, mu=2.0, sigma=9.0)
        rows = _curve_rows(lags, r, duration_s=240, n_overlap=120)
        result = compute_peak_fits_from_curves(rows, duration_s=240)
        self.assertEqual(len(result.params_rows), 1)
        params = result.params_rows[0]
        self.assertTrue(params["converged"])
        self.assertTrue(params["has_identifiable_peak"])
        self.assertAlmostEqual(float(params["peak_center_mu_s"]), 2.0, delta=1.0)
        self.assertIn("se_peak_height_A", params)
        self.assertIn("weighted_rss", params)

        with TemporaryDirectory() as tmp:
            paths = write_peak_fit_outputs(result, tmp)
            self.assertEqual(paths["params"].name, PARAMS_FILENAME)
            self.assertEqual(paths["qc"].name, QC_FILENAME)
            self.assertTrue(paths["params"].is_file())
            self.assertTrue(paths["qc"].is_file())

    def test_fisher_z_path_matches_direct_z_fit_qualitatively(self) -> None:
        lags = _lags(240)
        true_z = gaussian_peak(lags, 0.0, 0.4, 0.0, 10.0)
        r = np.tanh(true_z)
        z_from_r = np.asarray([fisher_z(float(value)) for value in r], dtype=float)
        # Rounding through tanh/atanh clip should still recover near-zero center.
        fit = fit_gaussian_peak(lags, z_from_r, weights=np.full(lags.shape, 120.0))
        self.assertAlmostEqual(float(fit["peak_center_mu_s"]), 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
