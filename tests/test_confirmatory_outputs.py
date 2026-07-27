from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
)
from ppg_eeg.confirmatory.figures import (
    FIGURE_DPI,
    generate_confirmatory_figures,
    mean_ci_by_lag,
    resolve_reporting_inputs,
)
from ppg_eeg.confirmatory.manifest import (
    FIGURE_SOURCE_MANIFEST_FILENAME,
    RUN_MANIFEST_FILENAME,
    build_figure_source_manifest,
    build_run_manifest,
    config_hash,
    sha256_file,
    sha256_text,
    software_versions,
)
from ppg_eeg.confirmatory.report import (
    RESULTS_BUNDLE_FILENAME,
    generate_confirmatory_report,
    run_confirmatory_reporting,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _seed_frozen_outputs(root: Path) -> None:
    lags = list(range(-60, 61))
    curve_rows = []
    # Matched low (rest) and effort (memory / verbalwm) curves for PRIMARY_META pairs.
    for band_i, band in enumerate(("theta", "alpha", "beta", "low_gamma")):
        for ds, low_cond, effort_cond in (
            ("ds003838", "rest", "memory"),
            ("ds006848", "rest", "verbalwm"),
        ):
            for obs in range(6):
                for state, cond in (("low", low_cond), ("effort", effort_cond)):
                    amp = 0.22 if state == "low" else 0.10
                    oid = f"{ds}-p{obs}-{cond}"
                    for lag in lags:
                        r = amp * math.exp(-0.5 * ((lag - 0) / 12.0) ** 2) + 0.01 * band_i
                        curve_rows.append(
                            {
                                "dataset_id": ds,
                                "subject_id": f"p{obs}",
                                "participant_id": f"p{obs}",
                                "task": cond,
                                "condition": cond,
                                "observation_id": oid,
                                "duration_s": 240,
                                "endpoint_name": ENDPOINT_ZLPI,
                                "band": band,
                                "power_representation": "absolute_log10",
                                "lag_s": lag,
                                "r": r,
                                "n_overlap": 120,
                            }
                        )
    _write_csv(root / "confirmatory_cross_correlation_curves_D240.csv", curve_rows)

    paired = []
    subjects = []
    for ds, contrast, low, effort in (
        ("ds003838", "rest__memory", "rest", "memory"),
        ("ds006848", "rest__verbalwm", "rest", "verbalwm"),
    ):
        for band in ("theta", "alpha", "beta", "low_gamma"):
            for i in range(6):
                paired.append(
                    {
                        "dataset_id": ds,
                        "contrast_id": contrast,
                        "participant_id": f"p{i}",
                        "session_id": "single",
                        "duration_s": 240,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "band": band,
                        "power_representation": "absolute_log10",
                        "low_endpoint_index": 0.4,
                        "effort_endpoint_index": 0.15,
                        "delta_endpoint_index": -0.25 + 0.01 * i,
                        "is_standard_zlpi": True,
                        "contrast_eligible": True,
                        "low_observation_ids": f"{ds}-p{i}-{low}",
                        "effort_observation_ids": f"{ds}-p{i}-{effort}",
                        "low_has_identifiable_peak": True,
                        "effort_has_identifiable_peak": i % 5 != 0,
                        "low_peak_center_mu_s": 0.4,
                        "effort_peak_center_mu_s": 0.6 if i % 5 != 0 else "",
                        "low_fwhm_s": 12.0,
                        "effort_fwhm_s": 14.0 if i % 5 != 0 else "",
                    }
                )
        for i in range(6):
            paired.append(
                {
                    "dataset_id": ds,
                    "contrast_id": contrast,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "duration_s": 180,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "delta_endpoint_index": -0.10,
                    "is_standard_zlpi": True,
                }
            )
            paired.append(
                {
                    "dataset_id": ds,
                    "contrast_id": contrast,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "duration_s": 120,
                    "endpoint_name": ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "delta_endpoint_index": -0.40,
                    "is_standard_zlpi": False,
                }
            )
            paired.append(
                {
                    "dataset_id": ds,
                    "contrast_id": contrast,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "duration_s": 60,
                    "endpoint_name": ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "delta_endpoint_index": -0.10,
                    "is_standard_zlpi": False,
                }
            )
            subjects.append(
                {
                    "dataset_id": ds,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "condition": low,
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "endpoint_index": 0.4,
                    "endpoint_eligible": True,
                }
            )
            subjects.append(
                {
                    "dataset_id": ds,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "condition": effort,
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "endpoint_index": 0.15,
                    "endpoint_eligible": True,
                }
            )
    _write_csv(root / "paired_contrasts.csv", paired)
    _write_csv(root / "subject_level_metrics.csv", subjects)

    effects = [
        {
            "dataset_id": "ds003838",
            "contrast_id": "rest__memory",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "theta",
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "n_pairs": 6,
            "effect_mean": -0.25,
            "ci_low": -0.35,
            "ci_high": -0.15,
            "p_value": 0.01,
            "enters_meta": True,
        },
        {
            "dataset_id": "ds003838",
            "contrast_id": "rest__memory",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "alpha",
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "n_pairs": 6,
            "effect_mean": -0.20,
            "ci_low": -0.30,
            "ci_high": -0.10,
            "p_value": 0.02,
            "enters_meta": True,
        },
        {
            "dataset_id": "ds006848",
            "contrast_id": "rest__verbalwm",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "alpha",
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "n_pairs": 6,
            "effect_mean": -0.18,
            "ci_low": -0.28,
            "ci_high": -0.08,
            "p_value": 0.03,
            "enters_meta": True,
        },
    ]
    for band in ("theta", "alpha", "beta", "low_gamma"):
        for contrast, effect in (("passive__simplert", -0.08), ("passive__gonogo", -0.16)):
            effects.append(
                {
                    "dataset_id": "ds003690",
                    "contrast_id": contrast,
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": band,
                    "power_representation": "absolute_log10",
                    "is_primary_analysis": True,
                    "n_pairs": 5,
                    "effect_mean": effect,
                    "ci_low": effect - 0.05,
                    "ci_high": effect + 0.05,
                    "p_value": 0.04,
                    "enters_meta": contrast == "passive__gonogo" and band == "alpha",
                }
            )
    _write_csv(root / "dataset_effects.csv", effects)

    meta = [
        {
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "band": band,
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "pooled_effect": -0.22,
            "ci_low": -0.30,
            "ci_high": -0.14,
            "prediction_low": -0.40,
            "prediction_high": -0.05,
            "n_datasets": 2,
            "i2": 0.1,
            "tau2": 0.01,
            "p_value": 0.001,
            "notes": "",
        }
        for band in ("theta", "alpha", "beta", "low_gamma")
    ]
    _write_csv(root / "meta_analysis_results.csv", meta)

    mixed_rows = [
        {
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "model_backend": "ols_cluster_participant",
            "converged": True,
            "term": term,
            "coef": coef,
            "stderr": 0.05,
            "z_or_t": coef / 0.05,
            "p_value": 0.1,
            "ci_low": coef - 0.1,
            "ci_high": coef + 0.1,
            "n_obs": 48,
            "n_groups": 12,
            "notes": "seed",
        }
        for term, coef in (
            ("Intercept", 0.05),
            ("C(state)[T.low_demand]", 0.10),
            ("C(band)[T.theta]", 0.02),
            ("C(state)[T.low_demand]:C(band)[T.theta]", 0.03),
        )
    ]
    _write_csv(root / "mixed_model_results.csv", mixed_rows)

    marginal_rows = []
    for band in ("theta", "alpha", "beta", "gamma"):
        for state, est in (
            ("low cognitive demand", 0.40),
            ("high cognitive demand", 0.25),
        ):
            marginal_rows.append(
                {
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": 240,
                    "power_representation": "absolute_log10",
                    "is_primary_analysis": True,
                    "dataset_scope": "PRIMARY_META_pooled",
                    "band": band,
                    "state": state,
                    "estimated_zlpi": est + (0.02 if band == "alpha" else 0.0),
                    "standard_error": 0.04,
                    "ci_low": est - 0.08,
                    "ci_high": est + 0.08,
                    "n_participants": 12,
                    "n_observations": 48,
                    "model_formula": (
                        "endpoint_index ~ C(state) * C(band) + C(modality) + mean_hr"
                    ),
                    "covariance_method": "participant_cluster_robust",
                    "covariate_prediction_method": "mean_hr=sample_mean;modality=sample_mode",
                    "model_backend": "ols_cluster_participant",
                    "converged": True,
                    "notes": "seed",
                }
            )
    _write_csv(root / "mixed_model_marginal_estimates.csv", marginal_rows)

    contrast_rows = [
        {
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "power_representation": "absolute_log10",
            "is_primary_analysis": True,
            "dataset_scope": "PRIMARY_META_pooled",
            "contrast_type": "within_band_state_effect",
            "contrast_name": f"{band}_high_minus_low",
            "contrast_direction": f"estimated ZLPI({band}, high) minus low",
            "estimate": -0.15,
            "standard_error": 0.03,
            "ci_low": -0.21,
            "ci_high": -0.09,
            "p_value": 0.01,
            "covariance_method": "participant_cluster_robust",
            "model_backend": "ols_cluster_participant",
            "converged": True,
            "notes": "seed",
        }
        for band in ("theta", "alpha", "beta", "gamma")
    ]
    for other in ("theta", "beta", "gamma"):
        contrast_rows.append(
            {
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": 240,
                "power_representation": "absolute_log10",
                "is_primary_analysis": True,
                "dataset_scope": "PRIMARY_META_pooled",
                "contrast_type": "alpha_vs_other_state_effect",
                "contrast_name": f"alpha_minus_{other}_state_effect",
                "contrast_direction": f"(alpha high−low) minus ({other} high−low)",
                "estimate": -0.02,
                "standard_error": 0.02,
                "ci_low": -0.06,
                "ci_high": 0.02,
                "p_value": 0.4,
                "covariance_method": "participant_cluster_robust",
                "model_backend": "ols_cluster_participant",
                "converged": True,
                "notes": "seed",
            }
        )
    _write_csv(root / "mixed_model_contrasts.csv", contrast_rows)

    equivalence = [
        {
            "dataset_id": "ds003838",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "band": "theta",
            "power_representation": "absolute_log10",
            "mean_mu": 0.2,
            "ci_low": -0.3,
            "ci_high": 0.7,
            "n": 10,
            "tost_p": 0.01,
            "equivalent": True,
        }
    ]
    _write_csv(root / "peak_center_equivalence.csv", equivalence)

    peak_params = [
        {
            "dataset_id": "ds003838",
            "subject_id": "sub-0",
            "condition": "rest",
            "band": "theta",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "power_representation": "absolute_log10",
            "has_identifiable_peak": True,
            "peak_center_mu_s": 0.5,
            "fwhm_s": 12.0,
        },
        {
            "dataset_id": "ds003838",
            "subject_id": "sub-1",
            "condition": "rest",
            "band": "alpha",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "power_representation": "absolute_log10",
            "has_identifiable_peak": True,
            "peak_center_mu_s": 0.2,
            "fwhm_s": 10.0,
        },
    ]
    _write_csv(root / "peak_fit_params.csv", peak_params)


    # Expand subject-level with all bands for heatmap cells.
    subjects_heatmap = list(subjects)
    for ds in ("ds003838", "ds006848"):
        for band in ("alpha", "beta", "low_gamma"):
            for i in range(3):
                subjects_heatmap.append(
                    {
                        "dataset_id": ds,
                        "participant_id": f"p{i}",
                        "session_id": "single",
                        "condition": "rest",
                        "duration_s": 240,
                        "endpoint_name": ENDPOINT_ZLPI,
                        "band": band,
                        "power_representation": "absolute_log10",
                        "endpoint_index": 0.35 - 0.02 * i,
                        "endpoint_eligible": True,
                    }
                )
    # Sensitivity cohort example.
    for band in ("theta", "alpha", "beta", "low_gamma"):
        subjects_heatmap.append(
            {
                "dataset_id": "hiit",
                "participant_id": "h0",
                "session_id": "ph",
                "condition": "ph_post_rest",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": band,
                "power_representation": "absolute_log10",
                "endpoint_index": 0.10,
                "endpoint_eligible": True,
            }
        )
    _write_csv(root / "subject_level_metrics.csv", subjects_heatmap)

    endpoint_rows = []
    for band in ("theta", "alpha", "beta", "low_gamma"):
        for obs in range(3):
            endpoint_rows.append(
                {
                    "dataset_id": "ds003838",
                    "subject_id": f"sub-{obs}",
                    "condition": "rest",
                    "observation_id": f"obs-{obs}",
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": band,
                    "power_representation": "absolute_log10",
                    "eligible": True,
                    "z0": 0.25,
                    "negative_shoulder_mean_z": 0.05,
                    "positive_shoulder_mean_z": 0.08,
                    "combined_flank_mean_z": 0.02,
                    "endpoint_index": 0.23,
                    "local_prominence": 0.17,
                }
            )
    _write_csv(root / "confirmatory_endpoint_metrics_D240.csv", endpoint_rows)

    null_summary = []
    for ds, role_band_p in (
        ("ds003838", 0.02),
        ("ds006848", 0.20),
        ("hiit", 0.40),
    ):
        for band in ("theta", "alpha", "beta", "low_gamma"):
            null_summary.append(
                {
                    "dataset_id": ds,
                    "condition": "rest" if ds != "hiit" else "ph_post_rest",
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": band,
                    "power_representation": "absolute_log10",
                    "null_type": "circular_shift",
                    "median_empirical_p": role_band_p,
                    "median_observed_endpoint_index": 0.2,
                }
            )
    _write_csv(root / "null_summary.csv", null_summary)

    protocol = [
        {
            "dataset_id": "ds003838",
            "dataset_role": "primary",
            "cardiac_modality": "ECG",
        },
        {
            "dataset_id": "ds006848",
            "dataset_role": "primary",
            "cardiac_modality": "ECG",
        },
        {
            "dataset_id": "hiit",
            "dataset_role": "sensitivity",
            "cardiac_modality": "PPG",
        },
    ]
    _write_csv(root / "protocol_audit.csv", protocol)

    null_rows = [
        {
            "dataset_id": "ds003838",
            "participant_id": f"p{i}",
            "subject_id": f"sub-{i:02d}",
            "session_id": "single",
            "condition": "rest" if i % 2 == 0 else "memory",
            "observation_id": f"obs-{i}",
            "band": "theta",
            "null_type": "circular_shift",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "observed_endpoint_index": 0.3,
            "null_mean": 0.05,
            "empirical_p": 0.04,
            "effect_size_surrogate_z": 2.0,
            "n_surrogates_requested": 20,
            "n_surrogates_finite": 20,
            "observed_eligible": True,
        }
        for i in range(5)
    ]
    _write_csv(root / "null_subject_results.csv", null_rows)

    duration = [
        {
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "theta",
            "effect_estimate": -0.25,
            "ci_low": -0.35,
            "ci_high": -0.15,
            "n": 6,
            "is_primary_analysis": True,
            "can_rescue_primary": False,
        },
        {
            "duration_s": 120,
            "endpoint_name": ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
            "band": "theta",
            "effect_estimate": -0.40,
            "ci_low": -0.55,
            "ci_high": -0.25,
            "n": 6,
            "is_primary_analysis": False,
            "can_rescue_primary": False,
        },
        {
            "duration_s": 60,
            "endpoint_name": ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
            "band": "theta",
            "effect_estimate": -0.10,
            "ci_low": -0.20,
            "ci_high": 0.0,
            "n": 6,
            "is_primary_analysis": False,
            "can_rescue_primary": False,
        },
    ]
    _write_csv(root / "duration_sensitivity.csv", duration)

    sensitivity = [
        {
            "control_id": "broadband_residualized",
            "dataset_id": "hiit",
            "band": "theta",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "power_representation": "broadband_residualized",
            "effect_estimate": -0.18,
            "ci_low": -0.28,
            "ci_high": -0.08,
            "n": 12,
            "status": "sensitivity_only",
            "is_primary_analysis": False,
            "can_rescue_primary": False,
        }
    ]
    _write_csv(root / "sensitivity_results.csv", sensitivity)

    spec = [
        {
            "control_id": "primary_d240_absolute_zlpi",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "effect_estimate": -0.25,
            "ci_low": -0.35,
            "ci_high": -0.15,
            "n": 12,
            "status": "primary_reference",
            "is_primary_analysis": True,
            "can_rescue_primary": False,
        },
        {
            "control_id": "broadband_residualized",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "effect_estimate": -0.18,
            "ci_low": -0.28,
            "ci_high": -0.08,
            "n": 12,
            "status": "sensitivity_only",
            "is_primary_analysis": False,
            "can_rescue_primary": False,
        },
    ]
    _write_csv(root / "specification_matrix.csv", spec)

    loo = [
        {
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "theta",
            "omitted_dataset_id": "ds003838",
            "pooled_effect": -0.20,
            "delta_vs_full": 0.02,
        }
    ]
    _write_csv(root / "leave_one_dataset_out.csv", loo)

    eligibility = [
        {"status": "eligible"},
        {"status": "eligible"},
        {"status": "ineligible"},
    ]
    _write_csv(root / "eligibility_by_duration.csv", eligibility)


class TestManifest(unittest.TestCase):
    def test_hashes_and_manifests_are_deterministic_for_same_inputs(self) -> None:
        self.assertEqual(sha256_text("abc"), sha256_text("abc"))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.yaml"
            cfg.write_text("primary_duration_s: 240\n", encoding="utf-8")
            h1 = config_hash([cfg])
            h2 = config_hash([cfg])
            self.assertEqual(h1, h2)
            versions = software_versions()
            self.assertIn("python", versions)
            self.assertIn("numpy", versions)

            panels = []
            from ppg_eeg.confirmatory.manifest import FigurePanelSource

            panels.append(
                FigurePanelSource(
                    figure_id="figure1",
                    panel_id="band_theta",
                    title="theta",
                    endpoint_name=ENDPOINT_ZLPI,
                    duration_s=240,
                    input_tables=[str(cfg)],
                    source_data_csv="source.csv",
                    analysis_keys=["endpoint=zlpi"],
                )
            )
            out1 = root / "out1"
            out2 = root / "out2"
            m1 = build_figure_source_manifest(
                panels, input_hashes={str(cfg): sha256_file(cfg)}, output_dir=out1
            )
            m2 = build_figure_source_manifest(
                panels, input_hashes={str(cfg): sha256_file(cfg)}, output_dir=out2
            )
            # Panel payload equal modulo timestamps.
            self.assertEqual(m1["panels"], m2["panels"])
            self.assertEqual(m1["input_file_sha256"], m2["input_file_sha256"])
            run = build_run_manifest(
                output_dir=root / "manifests",
                confirmatory_root=root,
                config_paths=[cfg],
                seeds={"null_surrogates": 20},
                inclusion_counts={"eligible": 2},
            )
            self.assertTrue(Path(run["manifest_path"]).name == RUN_MANIFEST_FILENAME)
            self.assertEqual(run["seeds"]["null_surrogates"], 20)


class TestFigures(unittest.TestCase):
    def test_mean_ci_and_figure_contracts(self) -> None:
        from ppg_eeg.confirmatory import figures as fig_mod

        # Keep figure generation fast in unit tests; production uses FIGURE1_BOOTSTRAP_N.
        fig_mod.FIGURE1_BOOTSTRAP_N = 40
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "frozen"
            out = Path(tmp) / "figures"
            _seed_frozen_outputs(root)
            resolved = resolve_reporting_inputs(root)
            self.assertIsNotNone(resolved["curves_d240"])

            curves_path = resolved["curves_d240"]
            assert curves_path is not None
            with curves_path.open(encoding="utf-8", newline="") as handle:
                curves = list(csv.DictReader(handle))
            series = mean_ci_by_lag(curves, band="theta", condition_role="low_demand")
            self.assertGreater(len(series), 50)
            self.assertEqual(series[0]["endpoint_name"], ENDPOINT_ZLPI)

            result = generate_confirmatory_figures(root, out)
            for artifacts in (result.figure1, result.figure2, result.figure3):
                self.assertTrue(artifacts.pdf.is_file())
                self.assertTrue(artifacts.svg.is_file())
                self.assertTrue(artifacts.png.is_file())
                self.assertTrue(artifacts.source_csvs)
            self.assertIsNotNone(result.figure3_supplement)
            assert result.figure3_supplement is not None
            self.assertTrue(result.figure3_supplement.png.is_file())
            self.assertTrue(result.figure_export_categories.is_file())
            with result.figure_export_categories.open(encoding="utf-8") as handle:
                export_rows = list(csv.DictReader(handle))
            by_stem = {r["stem"]: r for r in export_rows}
            self.assertEqual(by_stem["figure3_temporal_artifact_specificity"]["export_category"], "manuscript")
            self.assertEqual(by_stem["figure3_supplement_null_diagnostics"]["export_category"], "supplementary")
            self.assertEqual(by_stem["figure3_qc_participant_null_forests"]["export_category"], "internal_qc")
            self.assertEqual(
                by_stem["figure3_qc_participant_null_forests"]["include_in_supplementary_export"].lower(),
                "false",
            )
            self.assertTrue((out / "internal_qc").is_dir())
            self.assertEqual(result.figure_source_manifest.name, FIGURE_SOURCE_MANIFEST_FILENAME)
            payload = json.loads(result.figure_source_manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["panels"]), 8)
            # Figure 1 six-panel sources.
            for name in (
                "figure1_panel_b_lag_curves_bootstrap_ci.csv",
                "figure1_panel_c_lag_categories.csv",
                "figure1_panel_d_zlpi_heatmap.csv",
                "figure1_panel_e_alpha_replication_forest.csv",
                "figure1_panel_f_participant_peaks.csv",
            ):
                self.assertTrue((out / "source_data" / name).is_file())
            boot_csv = out / "source_data" / "figure1_panel_b_lag_curves_bootstrap_ci.csv"
            with boot_csv.open(encoding="utf-8", newline="") as handle:
                boot_rows = list(csv.DictReader(handle))
            self.assertTrue(boot_rows)
            self.assertEqual(boot_rows[0]["ci_method"], "participant_within_dataset_bootstrap")
            heat_csv = out / "source_data" / "figure1_panel_d_zlpi_heatmap.csv"
            with heat_csv.open(encoding="utf-8", newline="") as handle:
                heat_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row.get("surrogate_significant", "").lower() == "true" for row in heat_rows))
            forest_csv = out / "source_data" / "figure1_panel_e_alpha_replication_forest.csv"
            with forest_csv.open(encoding="utf-8", newline="") as handle:
                forest_rows = list(csv.DictReader(handle))
            self.assertTrue(any(row.get("band") == "alpha" for row in forest_rows))
            self.assertTrue(any(row.get("dataset_id") == "POOLED" for row in forest_rows))
            # Figure 2 six-panel sources.
            for name in (
                "figure2_panel_a_matched_lag_curves.csv",
                "figure2_panel_a_wiring_gaps.csv",
                "figure2_panel_b_lag_difference_curves.csv",
                "figure2_panel_c_alpha_meta_forest.csv",
                "figure2_panel_d_marginal_estimates.csv",
                "figure2_panel_d_contrasts.csv",
                "figure2_panel_d_mixedlm_coefficients.csv",
                "figure2_panel_e_paired_peaks.csv",
                "figure2_panel_f_graded_ds003690.csv",
            ):
                self.assertTrue((out / "source_data" / name).is_file())
            f2_a = out / "source_data" / "figure2_panel_a_matched_lag_curves.csv"
            with f2_a.open(encoding="utf-8", newline="") as handle:
                f2_a_rows = list(csv.DictReader(handle))
            self.assertTrue(f2_a_rows)
            self.assertEqual(
                f2_a_rows[0]["ci_method"],
                "paired_participant_within_dataset_bootstrap",
            )
            with (out / "source_data" / "figure2_panel_a_wiring_gaps.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                gap_rows = list(csv.DictReader(handle))
            self.assertEqual(gap_rows, [])
            with (out / "source_data" / "figure2_panel_c_alpha_meta_forest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                f2_c = list(csv.DictReader(handle))
            self.assertTrue(any(row.get("band") == "alpha" for row in f2_c))
            self.assertTrue(any(row.get("dataset_id") == "POOLED" for row in f2_c))
            caption2 = (out / "figure2_caption.txt").read_text(encoding="utf-8").casefold()
            self.assertIn("no percent attenuation", caption2)
            self.assertIn("no cluster-permutation", caption2)
            self.assertNotIn("% attenuation", caption2)
            with (out / "source_data" / "figure2_panel_d_marginal_estimates.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                f2_d = list(csv.DictReader(handle))
            self.assertEqual(len(f2_d), 8)
            self.assertEqual({row["band"] for row in f2_d}, {"theta", "alpha", "beta", "gamma"})
            with (out / "source_data" / "figure2_panel_d_contrasts.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                f2_d_contrasts = list(csv.DictReader(handle))
            self.assertEqual(len(f2_d_contrasts), 7)
            with (out / "source_data" / "figure2_panel_f_graded_ds003690.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                f2_f = list(csv.DictReader(handle))
            self.assertTrue(
                {row["contrast_id"] for row in f2_f}
                <= {"passive__simplert", "passive__gonogo"}
            )
            self.assertTrue(all(row["dataset_id"] == "ds003690" for row in f2_f))
            self.assertIsNotNone(resolved["mixed_model"])

            # Endpoint separation present in figure3 duration source.
            duration_csv = out / "source_data" / "figure3_panel_b_duration.csv"
            with duration_csv.open(encoding="utf-8", newline="") as handle:
                duration_rows = list(csv.DictReader(handle))
            endpoints = {row["endpoint_name"] for row in duration_rows}
            self.assertIn(ENDPOINT_ZLPI, endpoints)
            self.assertIn(ENDPOINT_MID_WINDOW_PROXIMAL_INDEX, endpoints)
            self.assertIn(ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX, endpoints)
            # No hard-coded rescue: can_rescue_primary is false in source.
            for row in duration_rows:
                self.assertEqual(str(row["can_rescue_primary"]).lower(), "false")
            broadband_csv = out / "source_data" / "figure3_panel_c_broadband.csv"
            self.assertTrue(broadband_csv.is_file())
            self.assertFalse((out / "source_data" / "figure3_panel_c_modality.csv").exists())
            self.assertEqual(FIGURE_DPI, 300)

            meta_panel = next(
                p for p in payload["panels"] if p["panel_id"] == "alpha_primary_meta_forest"
            )
            self.assertIn("absolute", meta_panel["title"].casefold())

            from ppg_eeg.confirmatory.figures import (
                FIGURE1_TITLE,
                FIGURE2_TITLE,
                MSG_NOT_APPLICABLE,
                _band_not_analyzed_message,
                _endpoint_display,
                _meta_or_single_title,
            )

            self.assertEqual(_endpoint_display(ENDPOINT_ZLPI), "ZLPI")
            self.assertEqual(_endpoint_display(ENDPOINT_MID_WINDOW_PROXIMAL_INDEX), "MWPI")
            self.assertEqual(_endpoint_display(ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX), "SWPI")
            self.assertIn("not analyzed", _band_not_analyzed_message("delta").casefold())
            self.assertEqual(_meta_or_single_title(n_datasets=1), "Single-dataset effect estimate")
            self.assertEqual(
                _meta_or_single_title(n_datasets=2), "Random-effects meta-analysis"
            )
            self.assertEqual(MSG_NOT_APPLICABLE, "Not applicable for this dataset")
            self.assertEqual(
                FIGURE1_TITLE,
                "Confirmatory EEG–cardiac coupling: structure, replication, and peaks",
            )
            self.assertIn("State-dependent attenuation", FIGURE2_TITLE)


class TestFigure1Bootstrap(unittest.TestCase):
    def test_participant_within_dataset_bootstrap_ci(self) -> None:
        from ppg_eeg.confirmatory.figure1_panels import bootstrap_mean_ci_by_lag

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "frozen"
            _seed_frozen_outputs(root)
            with (root / "confirmatory_cross_correlation_curves_D240.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                curves = list(csv.DictReader(handle))
            series = bootstrap_mean_ci_by_lag(
                curves, band="theta", condition_role="low_demand", n_bootstrap=50, seed=1
            )
            self.assertGreater(len(series), 50)
            self.assertEqual(series[0]["ci_method"], "participant_within_dataset_bootstrap")
            self.assertEqual(int(series[0]["n"]), 12)
            self.assertTrue(math.isfinite(float(series[0]["ci_low"])))
            self.assertTrue(math.isfinite(float(series[0]["ci_high"])))
            self.assertLessEqual(float(series[0]["ci_low"]), float(series[0]["mean_z"]))
            self.assertGreaterEqual(float(series[0]["ci_high"]), float(series[0]["mean_z"]))


class TestReport(unittest.TestCase):
    def test_report_bundle_reads_frozen_outputs_only(self) -> None:
        from ppg_eeg.confirmatory import figures as fig_mod

        fig_mod.FIGURE1_BOOTSTRAP_N = 40
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "frozen"
            out = Path(tmp) / "reports"
            _seed_frozen_outputs(root)
            report = generate_confirmatory_report(root, out)
            self.assertTrue(report.methods_summary.is_file())
            self.assertTrue(report.results_summary.is_file())
            self.assertTrue(report.results_bundle.is_file())
            bundle = json.loads(report.results_bundle.read_text(encoding="utf-8"))
            self.assertEqual(bundle["primary_endpoint"], ENDPOINT_ZLPI)
            self.assertTrue(bundle["input_tables"])
            # Estimates come from frozen meta table.
            meta_estimates = [
                row["estimate"]
                for row in bundle["results_summary"]
                if row["family"] == "meta_analysis"
            ]
            self.assertTrue(meta_estimates)
            self.assertTrue(all(math.isfinite(float(v)) for v in meta_estimates))

            paths = run_confirmatory_reporting(
                root,
                Path(tmp) / "full",
                config_paths=[],
                seeds={"smoke_n_surrogates": 20},
            )
            self.assertTrue(paths["figure1_png"].is_file())
            self.assertTrue(paths["run_manifest"].is_file())
            self.assertEqual(paths["results_bundle"].name, RESULTS_BUNDLE_FILENAME)


if __name__ == "__main__":
    unittest.main()
