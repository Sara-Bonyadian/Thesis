"""Figure 2 Panel D: model-estimated ZLPI and prespecified contrasts."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S
from ppg_eeg.confirmatory.figure2_panels import (
    PANEL_D_DISPLAY_BANDS,
    _load_panel_d_tables,
    _render_panel_d_estimation_plot,
    render_figure2,
)
from ppg_eeg.confirmatory.figures import resolve_reporting_inputs
from ppg_eeg.confirmatory.inference import (
    MIXED_MODEL_FORMULA_CORE,
    PANEL_D_BAND_EXPORT_LABELS,
    PANEL_D_STATE_EXPORT_LABELS,
    _align_exog_to_params,
    _linear_contrast_inference,
    compute_mixed_model_panel_d_estimates,
    fit_mixed_model,
    run_confirmatory_inference,
    write_inference_outputs,
)
from tests.test_confirmatory_inference import (
    _synthetic_paired_from_subjects,
)


class Figure2PanelDMarginalEstimatesTests(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_confirmatory_inference import _subject_row

        rng = np.random.default_rng(11)
        bands = ("theta", "alpha", "beta", "low_gamma")
        datasets = (
            ("ds003838", "rest", "memory"),
            ("ds006848", "rest", "verbalwm"),
            ("ds003690", "passive", "gonogo"),
        )
        self.subjects: list[dict[str, object]] = []
        for dataset_id, low, effort in datasets:
            for i in range(10):
                pid = f"p{i:02d}"
                for band in bands:
                    base_val = 0.40 + 0.02 * bands.index(band)
                    self.subjects.append(
                        _subject_row(
                            dataset_id=dataset_id,
                            participant_id=pid,
                            condition=low,
                            state="low_demand",
                            band=band,
                            endpoint_index=base_val + float(rng.normal(0.0, 0.05)),
                            mean_hr=65 + float(rng.normal(0, 2)),
                        )
                    )
                    self.subjects.append(
                        _subject_row(
                            dataset_id=dataset_id,
                            participant_id=pid,
                            condition=effort,
                            state="cognitive_effort",
                            band=band,
                            endpoint_index=base_val - 0.35 + float(rng.normal(0.0, 0.05)),
                            mean_hr=72 + float(rng.normal(0, 2)),
                        )
                    )
        self.paired = _synthetic_paired_from_subjects(self.subjects)

    def test_panel_d_produces_eight_band_state_estimates(self) -> None:
        _, qc = fit_mixed_model(self.subjects, endpoint_name=ENDPOINT_ZLPI)
        marginal = list(qc.get("panel_d_marginal_rows") or [])
        self.assertEqual(len(marginal), 8)
        bands = {row["band"] for row in marginal}
        states = {row["state"] for row in marginal}
        self.assertEqual(bands, set(PANEL_D_DISPLAY_BANDS))
        self.assertEqual(states, set(PANEL_D_STATE_EXPORT_LABELS.values()))
        for row in marginal:
            self.assertTrue(math.isfinite(float(row["estimated_zlpi"])))
            self.assertTrue(math.isfinite(float(row["standard_error"])))
            self.assertTrue(math.isfinite(float(row["ci_low"])))
            self.assertTrue(math.isfinite(float(row["ci_high"])))
            self.assertNotIn("low-γ", str(row["band"]))
            self.assertNotIn("low-γ", str(row["state"]))

    def test_panel_d_contrast_tables(self) -> None:
        _, qc = fit_mixed_model(self.subjects, endpoint_name=ENDPOINT_ZLPI)
        contrasts = list(qc.get("panel_d_contrast_rows") or [])
        within = [r for r in contrasts if r["contrast_type"] == "within_band_state_effect"]
        alpha_vs = [r for r in contrasts if r["contrast_type"] == "alpha_vs_other_state_effect"]
        self.assertEqual(len(within), 4)
        self.assertEqual(len(alpha_vs), 3)
        self.assertEqual(
            {r["contrast_name"] for r in within},
            {
                "theta_high_minus_low",
                "alpha_high_minus_low",
                "beta_high_minus_low",
                "gamma_high_minus_low",
            },
        )
        self.assertEqual(
            {r["contrast_name"] for r in alpha_vs},
            {
                "alpha_minus_theta_state_effect",
                "alpha_minus_beta_state_effect",
                "alpha_minus_gamma_state_effect",
            },
        )
        for row in contrasts:
            self.assertTrue(math.isfinite(float(row["estimate"])))
            self.assertTrue(math.isfinite(float(row["standard_error"])))
            self.assertTrue(math.isfinite(float(row["p_value"])))

    def test_manual_validation_design_vectors(self) -> None:
        import statsmodels.formula.api as smf
        from unittest.mock import patch

        from ppg_eeg.confirmatory.inference import (
            _panel_d_prediction_frame,
            _prepare_subject_frame,
        )

        with patch(
            "ppg_eeg.confirmatory.inference.smf.mixedlm",
            side_effect=RuntimeError("force_ols"),
        ):
            _, qc = fit_mixed_model(self.subjects, endpoint_name=ENDPOINT_ZLPI)
        marginal = list(qc.get("panel_d_marginal_rows") or [])
        contrasts = list(qc.get("panel_d_contrast_rows") or [])

        frame = _prepare_subject_frame(
            self.subjects,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            power_representation="absolute_log10",
        )
        fitted = smf.ols(
            MIXED_MODEL_FORMULA_CORE + " + C(dataset_id)",
            data=frame,
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["participant_uid"]})

        pred_frame = _panel_d_prediction_frame(frame)
        exog = _align_exog_to_params(fitted, pred_frame)
        param_names = list(fitted.params.index)
        row_index = {
            (str(r["band"]).casefold(), str(r["state"]).casefold()): i
            for i, (_, r) in enumerate(pred_frame.iterrows())
        }

        alpha_low_row = next(
            r for r in marginal if r["band"] == "alpha" and "low" in str(r["state"])
        )
        alpha_low_idx = row_index[("alpha", "low_demand")]
        manual_est, manual_se, _, _, _ = _linear_contrast_inference(
            fitted,
            exog[alpha_low_idx, :],
            param_names=param_names,
        )
        self.assertAlmostEqual(float(alpha_low_row["estimated_zlpi"]), manual_est, places=6)
        self.assertAlmostEqual(float(alpha_low_row["standard_error"]), manual_se, places=6)

        alpha_within = next(r for r in contrasts if r["contrast_name"] == "alpha_high_minus_low")
        alpha_high_idx = row_index[("alpha", "cognitive_effort")]
        within_vec = exog[alpha_high_idx, :] - exog[alpha_low_idx, :]
        manual_within, _, _, _, manual_p = _linear_contrast_inference(
            fitted, within_vec, param_names=param_names
        )
        self.assertAlmostEqual(float(alpha_within["estimate"]), manual_within, places=6)
        self.assertAlmostEqual(float(alpha_within["p_value"]), manual_p, places=6)

        alpha_vs_theta = next(
            r for r in contrasts if r["contrast_name"] == "alpha_minus_theta_state_effect"
        )
        theta_low_idx = row_index[("theta", "low_demand")]
        theta_high_idx = row_index[("theta", "cognitive_effort")]
        alpha_state_vec = exog[alpha_high_idx, :] - exog[alpha_low_idx, :]
        theta_state_vec = exog[theta_high_idx, :] - exog[theta_low_idx, :]
        interaction_vec = alpha_state_vec - theta_state_vec
        manual_interaction, _, _, _, _ = _linear_contrast_inference(
            fitted, interaction_vec, param_names=param_names
        )
        self.assertAlmostEqual(float(alpha_vs_theta["estimate"]), manual_interaction, places=6)
        self.assertIn("negative", str(alpha_vs_theta["contrast_direction"]).casefold())
        self.assertIn("attenuation", str(alpha_vs_theta["contrast_direction"]).casefold())

    def test_reference_invariance(self) -> None:
        from ppg_eeg.confirmatory.inference import (
            _panel_d_bands_in_frame,
            _prepare_subject_frame,
        )
        import statsmodels.formula.api as smf

        frame = _prepare_subject_frame(
            self.subjects,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            power_representation="absolute_log10",
        )
        default_formula = MIXED_MODEL_FORMULA_CORE + " + C(dataset_id)"
        alt_formula = (
            "endpoint_index ~ "
            "C(state, Treatment(reference='low_demand')) * "
            "C(band, Treatment(reference='theta')) + C(modality) + mean_hr + C(dataset_id)"
        )
        fitted_default = smf.ols(default_formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["participant_uid"]},
        )
        fitted_alt = smf.ols(alt_formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["participant_uid"]},
        )
        marginal_a, contrasts_a = compute_mixed_model_panel_d_estimates(
            fitted_default,
            frame,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            power_representation="absolute_log10",
            is_primary=True,
            backend="ols_cluster_participant",
            qc_notes="",
        )
        marginal_b, contrasts_b = compute_mixed_model_panel_d_estimates(
            fitted_alt,
            frame,
            endpoint_name=ENDPOINT_ZLPI,
            duration_s=EXPECTED_PRIMARY_DURATION_S,
            power_representation="absolute_log10",
            is_primary=True,
            backend="ols_cluster_participant",
            qc_notes="",
        )
        self.assertEqual(len(marginal_a), 8)
        self.assertEqual(len(marginal_b), 8)
        for row_a, row_b in zip(
            sorted(marginal_a, key=lambda r: (r["band"], r["state"])),
            sorted(marginal_b, key=lambda r: (r["band"], r["state"])),
            strict=True,
        ):
            self.assertAlmostEqual(
                float(row_a["estimated_zlpi"]),
                float(row_b["estimated_zlpi"]),
                places=5,
            )
        for row_a, row_b in zip(
            sorted(contrasts_a, key=lambda r: r["contrast_name"]),
            sorted(contrasts_b, key=lambda r: r["contrast_name"]),
            strict=True,
        ):
            self.assertAlmostEqual(float(row_a["estimate"]), float(row_b["estimate"]), places=5)

    def test_inference_exports_and_figure_match(self) -> None:
        result = run_confirmatory_inference(
            self.subjects,
            self.paired,
            fit_endpoints=(ENDPOINT_ZLPI,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_dir = root / "infer"
            fig_dir = root / "figures"
            paths = write_inference_outputs(result, infer_dir)
            self.assertTrue(paths["mixed_model_marginal_estimates"].is_file())
            self.assertTrue(paths["mixed_model_contrasts"].is_file())
            inputs = resolve_reporting_inputs(infer_dir)
            render_figure2(inputs, fig_dir)
            marginal_csv = fig_dir / "source_data" / "figure2_panel_d_marginal_estimates.csv"
            contrast_csv = fig_dir / "source_data" / "figure2_panel_d_contrasts.csv"
            self.assertTrue(marginal_csv.is_file())
            self.assertTrue(contrast_csv.is_file())
            marginal, contrasts, _ = _load_panel_d_tables(inputs, panel_a_hiit_sensitivity=False)
            self.assertEqual(len(marginal), 8)
            self.assertEqual(len(contrasts), 7)

    def test_covariate_method_documents_sample_mean_hr(self) -> None:
        _, qc = fit_mixed_model(self.subjects, endpoint_name=ENDPOINT_ZLPI)
        marginal = list(qc.get("panel_d_marginal_rows") or [])
        self.assertTrue(marginal)
        method = str(marginal[0]["covariate_prediction_method"])
        self.assertIn("mean_hr=sample_mean", method)
        self.assertIn("modality=sample_mode", method)


if __name__ == "__main__":
    unittest.main()
