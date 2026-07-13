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
    for band_i, band in enumerate(("delta", "theta", "alpha", "beta")):
        for obs in range(3):
            for lag in lags:
                # Smooth zero-centered bump in Fisher-r space.
                r = 0.2 * math.exp(-0.5 * ((lag - 0) / 12.0) ** 2) + 0.01 * band_i
                curve_rows.append(
                    {
                        "dataset_id": "ds003838",
                        "subject_id": f"sub-{obs}",
                        "task": "rest",
                        "condition": "rest",
                        "observation_id": f"obs-{obs}",
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
        for i in range(6):
            paired.append(
                {
                    "dataset_id": ds,
                    "contrast_id": contrast,
                    "participant_id": f"p{i}",
                    "session_id": "single",
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "low_endpoint_index": 0.4,
                    "effort_endpoint_index": 0.15,
                    "delta_endpoint_index": -0.25 + 0.01 * i,
                    "is_standard_zlpi": True,
                }
            )
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
            "n_datasets": 2,
            "i2": 0.1,
            "tau2": 0.01,
            "p_value": 0.001,
            "notes": "",
        }
        for band in ("delta", "theta", "alpha", "beta")
    ]
    _write_csv(root / "meta_analysis_results.csv", meta)

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
        }
    ]
    _write_csv(root / "dataset_effects.csv", effects)

    equivalence = [
        {
            "dataset_id": "ds003838",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "band": "theta",
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
            "band": "theta",
            "duration_s": 240,
            "endpoint_name": ENDPOINT_ZLPI,
            "has_identifiable_peak": True,
            "peak_center_mu_s": 0.5,
        }
    ]
    _write_csv(root / "peak_fit_params.csv", peak_params)

    null_rows = [
        {
            "observation_id": f"obs-{i}",
            "band": "theta",
            "null_type": "circular_shift",
            "endpoint_name": ENDPOINT_ZLPI,
            "duration_s": 240,
            "observed_endpoint_index": 0.3,
            "null_mean": 0.05,
            "empirical_p": 0.04,
            "effect_size_surrogate_z": 2.0,
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

    modality = [
        {
            "dataset_id": "hiit",
            "participant_id": "p01",
            "band": "theta",
            "endpoint_name": ENDPOINT_ZLPI,
            "matched": True,
            "delta_ecg_minus_ppg": 0.05,
        }
    ]
    _write_csv(root / "modality_comparison.csv", modality)

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
            self.assertEqual(result.figure_source_manifest.name, FIGURE_SOURCE_MANIFEST_FILENAME)
            payload = json.loads(result.figure_source_manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["panels"]), 8)
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
            self.assertEqual(FIGURE_DPI, 300)


class TestReport(unittest.TestCase):
    def test_report_bundle_reads_frozen_outputs_only(self) -> None:
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
