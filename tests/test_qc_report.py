"""Tests for read-only confirmatory QC reporting."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ppg_eeg.confirmatory.qc_report import (
    DERIVED_METRIC,
    FLAGGED_FOR_REVIEW,
    NO_AUTOMATED_CONCERN,
    NOT_ASSESSED,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    SEVERITY_SEVERE,
    QcReportParams,
    apply_observation_flags,
    assign_qc_review_severity,
    generate_qc_report,
)
from ppg_eeg.confirmatory.run import STAGE_ORDER, expand_stages, main as confirmatory_main


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path, *, exclude_qc: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if exclude_qc and (rel == "QC" or rel.startswith("QC/")):
            continue
        out[rel] = _sha256(path)
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _seed_minimal_tree(root: Path, *, with_c4_c6: bool = True) -> None:
    _write_csv(
        root / "C0" / "data_audit.csv",
        [
            {
                "dataset_id": "toy",
                "observation_id": "obs-a",
                "participant_id": "p1",
                "session_id": "s1",
                "subject_id": "p1",
                "condition": "rest",
                "task": "rest",
                "usable": True,
                "exclusion_reason": "",
                "skip_reason": "",
                "cardiac_exists": True,
                "eeg_exists": True,
                "available_cardiac_channel": "ECG",
                "cardiac_signal_type": "ecg",
            },
            {
                "dataset_id": "toy",
                "observation_id": "obs-b",
                "participant_id": "p1",
                "session_id": "s1",
                "subject_id": "p1",
                "condition": "task",
                "task": "task",
                "usable": True,
                "exclusion_reason": "",
                "skip_reason": "",
                "cardiac_exists": True,
                "eeg_exists": True,
                "available_cardiac_channel": "ECG",
                "cardiac_signal_type": "ecg",
            },
        ],
    )
    _write_csv(
        root / "C0" / "eligibility_by_duration.csv",
        [
            {
                "dataset_id": "toy",
                "observation_id": "obs-a",
                "contrast_id": "rest__task",
                "duration_s": 240,
                "status": "eligible",
                "exclusion_code": "",
            },
            {
                "dataset_id": "toy",
                "observation_id": "obs-b",
                "contrast_id": "rest__task",
                "duration_s": 240,
                "status": "eligible",
                "exclusion_code": "",
            },
        ],
    )
    for oid in ("obs-a", "obs-b"):
        _write_csv(
            root / "C1a" / oid / "multitaper_qc.csv",
            [
                {
                    "dataset_id": "toy",
                    "observation_id": oid,
                    "n_input_channels": 64,
                    "n_usable_channels": 60 if oid == "obs-a" else 40,
                    "rejected_channels": "X1;X2;X3;X4",
                    "spectral_qc_passed": True,
                    "status": "ok",
                    "warning": "",
                    "n_windows": 100,
                    "n_nonfinite_power_rows": 0,
                }
            ],
        )
        _write_csv(
            root / "C1c" / oid / "alignment_qc_confirmatory.csv",
            [
                {
                    "dataset_id": "toy",
                    "observation_id": oid,
                    "duration_s": 240,
                    "duration_role": "primary",
                    "eligible": True,
                    "exclusion_reason": "",
                    "n_common_support": 300,
                    "available_support_s": 300.0,
                }
            ],
        )

    _write_csv(
        root / "C1b" / "cardiac_peak_qc.csv",
        [
            {
                "dataset_id": "toy",
                "observation_id": "obs-a",
                "subject_id": "p1",
                "condition": "rest",
                "task": "rest",
                "channel_used": "ECG",
                "signal_type": "ecg",
                "detector_used": "neurokit",
                "detector_polarity": "normal",
                "selection_reason": "best",
                "n_raw_peaks": 100,
                "n_clean_ibis": 90,
                "n_accepted_peaks": 95,
                "clean_ibi_coverage_s": 200.0,
                "median_hr_bpm": 70.0,
                "usable": True,
                "warning": "",
            },
            {
                "dataset_id": "toy",
                "observation_id": "obs-b",
                "subject_id": "p1",
                "condition": "task",
                "task": "task",
                "channel_used": "ECG",
                "signal_type": "ecg",
                "detector_used": "neurokit",
                "detector_polarity": "normal",
                "selection_reason": "best",
                "n_raw_peaks": 100,
                "n_clean_ibis": 90,
                "n_accepted_peaks": 95,
                "clean_ibi_coverage_s": 200.0,
                "median_hr_bpm": 75.0,
                "usable": True,
                "warning": "",
            },
        ],
    )
    # Close polarity gap on obs-a → flagged; wide gap on obs-b.
    _write_csv(
        root / "C1b" / "obs-a" / "peak_detector_comparison.csv",
        [
            {
                "polarity": "normal",
                "quality_score": 100.0,
                "selected": True,
                "percent_clean_ibi": 90.0,
            },
            {
                "polarity": "inverted",
                "quality_score": 95.0,
                "selected": False,
                "percent_clean_ibi": 80.0,
            },
        ],
    )
    _write_csv(
        root / "C1b" / "obs-b" / "peak_detector_comparison.csv",
        [
            {
                "polarity": "normal",
                "quality_score": 200.0,
                "selected": True,
                "percent_clean_ibi": 90.0,
            },
            {
                "polarity": "inverted",
                "quality_score": 50.0,
                "selected": False,
                "percent_clean_ibi": 80.0,
            },
        ],
    )
    for oid, gap_n in (("obs-a", 40), ("obs-b", 0)):
        _write_csv(
            root / "C1b" / oid / "instant_hr_qc.csv",
            [
                {
                    "observation_id": oid,
                    "n_valid_hr_samples": 100,
                    "n_gap_masked_samples": gap_n,
                    "n_grid_samples": 100,
                    "max_beat_gap_s": 2.0,
                    "status": "ok",
                    "warning": "",
                }
            ],
        )
        hrs = [70.0] * 100
        if oid == "obs-a":
            hrs[10] = 100.0  # abs diff > 20
        _write_csv(
            root / "C1b" / oid / "features_instant_hr.csv",
            [
                {"time_s": float(i), "instant_hr_bpm": hrs[i], "is_valid_hr": True}
                for i in range(100)
            ],
        )

    _write_csv(
        root / "C2" / "confirmatory_cross_correlation_qc_D240.csv",
        [
            {
                "dataset_id": "toy",
                "observation_id": oid,
                "duration_s": 240,
                "band": "theta",
                "power_representation": "absolute_log10",
                "endpoint_name": "zlpi",
                "exclusion_reason": "",
                "n_common_support": 120,
            }
            for oid in ("obs-a", "obs-b")
        ],
    )
    _write_csv(
        root / "C3" / "confirmatory_endpoint_qc_D240.csv",
        [
            {
                "dataset_id": "toy",
                "observation_id": oid,
                "duration_s": 240,
                "band": "theta",
                "power_representation": "absolute_log10",
                "is_primary_representation": True,
                "endpoint_name": "zlpi",
                "eligible": True,
                "exclusion_reason": "",
                "n_common_support": 120,
            }
            for oid in ("obs-a", "obs-b")
        ],
    )
    _write_csv(
        root / "C3" / "peak_fit_qc.csv",
        [
            {
                "observation_id": oid,
                "duration_s": 240,
                "band": "theta",
                "power_representation": "absolute_log10",
                "endpoint_name": "zlpi",
                "converged": True,
                "has_identifiable_peak": True,
            }
            for oid in ("obs-a", "obs-b")
        ],
    )

    if with_c4_c6:
        _write_csv(
            root / "C4" / "null_qc.csv",
            [
                {
                    "observation_id": "obs-a",
                    "duration_s": 240,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "endpoint_name": "zlpi",
                    "null_type": "circular_shift",
                    "status": "ok",
                    "n_surrogates_requested": 20,
                    "n_surrogates_finite": 20,
                    "notes": "",
                }
            ],
        )
        _write_csv(
            root / "C5" / "paired_contrasts.csv",
            [
                {
                    "dataset_id": "toy",
                    "contrast_id": "rest__task",
                    "participant_id": "p1",
                    "session_id": "s1",
                    "duration_s": 240,
                    "band": "theta",
                    "power_representation": "absolute_log10",
                    "endpoint_name": "zlpi",
                    "is_primary_representation": True,
                    "contrast_eligible": True,
                    "contrast_exclusion_reason": "",
                    "mu_contrast_eligible": True,
                    "low_observation_ids": "obs-a",
                    "effort_observation_ids": "obs-b",
                    "delta_endpoint_index": 0.1,
                }
            ],
        )
        _write_csv(
            root / "C6" / "inference_qc.csv",
            [
                {
                    "component": "mixed_model",
                    "endpoint_name": "zlpi",
                    "duration_s": 240,
                    "status": "ok",
                    "model_backend": "ols",
                    "converged": True,
                    "n_obs": 2,
                    "n_groups": 1,
                    "notes": "",
                }
            ],
        )


class QcReportTests(unittest.TestCase):
    def test_stage_all_excludes_qc(self) -> None:
        stages = expand_stages("all")
        self.assertEqual(stages, list(STAGE_ORDER))
        self.assertNotIn("QC", STAGE_ORDER)
        self.assertNotIn("qc-report", STAGE_ORDER)

    def test_heuristic_flags_record_triggering_rules(self) -> None:
        params = QcReportParams(
            polarity_score_gap=50.0,
            polarity_score_rel_gap=0.05,
            gap_masked_pct=5.0,
            hr_abs_diff_bpm=20.0,
            hr_outside_band_frac=0.01,
            eeg_channel_reject_pct=25.0,
        )
        row = apply_observation_flags(
            {
                "c0_present": True,
                "c0_usable": True,
                "c1b_present": True,
                "c1b_usable": True,
                "c1b_warning": "",
                "signal_type": "ecg",
                "channel_used": "ECG",
                "polarity_metric_source": DERIVED_METRIC,
                "polarity_score_gap_abs": 5.0,
                "polarity_score_gap_rel": 0.05,
                "ihr_derived_metric_source": DERIVED_METRIC,
                "pct_gap_masked": 40.0,
                "frac_hr_outside_plausible_band": 0.0,
                "n_hr_abs_diff_gt_threshold": 1,
                "ihr_status": "ok",
                "c1a_present": True,
                "pct_channels_rejected": 37.5,
                "spectral_qc_passed": True,
                "multitaper_status": "ok",
            },
            params,
        )
        self.assertEqual(row["review_polarity_disposition"], FLAGGED_FOR_REVIEW)
        self.assertIn("polarity_abs_gap<50", row["review_polarity_triggering_rule"])
        self.assertEqual(row["review_ihr_disposition"], FLAGGED_FOR_REVIEW)
        self.assertIn("pct_gap_masked>5", row["review_ihr_triggering_rule"])
        self.assertIn("n_hr_abs_diff_gt_20>0", row["review_ihr_triggering_rule"])
        self.assertEqual(row["review_eeg_disposition"], FLAGGED_FOR_REVIEW)
        self.assertIn("pct_channels_rejected>25", row["review_eeg_triggering_rule"])
        self.assertEqual(row["observation_review_disposition"], FLAGGED_FOR_REVIEW)
        self.assertEqual(row["qc_review_severity"], SEVERITY_SEVERE)

    def test_severity_ranking_preserves_binary_flags(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            # Make obs-b severely out-of-band without changing binary schema.
            hrs = [70.0] * 50 + [20.0] * 50  # 50% outside [40,180]
            _write_csv(
                root / "C1b" / "obs-b" / "features_instant_hr.csv",
                [
                    {
                        "time_s": float(i),
                        "instant_hr_bpm": hrs[i],
                        "is_valid_hr": True,
                    }
                    for i in range(100)
                ],
            )
            qc_dir = generate_qc_report(root, dataset_id="toy")
            obs = pd.read_csv(qc_dir / "observation_qc.csv")
            for col in (
                "frac_hr_outside",
                "ihr_jump_rate_per_min",
                "max_consecutive_hr_jump",
                "qc_review_severity",
                "review_ihr_disposition",
                "observation_review_disposition",
            ):
                self.assertIn(col, obs.columns)
            # Binary IHR flags still fire on any >20 bpm jump (obs-a).
            a = obs.loc[obs["observation_id"] == "obs-a"].iloc[0]
            self.assertEqual(a["review_ihr_disposition"], FLAGGED_FOR_REVIEW)
            self.assertIn("n_hr_abs_diff_gt_20>0", str(a["review_ihr_triggering_rule"]))
            # Severity ranks the out-of-band case as severe.
            b = obs.loc[obs["observation_id"] == "obs-b"].iloc[0]
            self.assertEqual(b["qc_review_severity"], SEVERITY_SEVERE)
            self.assertGreater(float(b["frac_hr_outside"]), 0.10)

            visual = pd.read_csv(qc_dir / "visual_review_list.csv")
            self.assertIn("qc_review_severity", visual.columns)
            self.assertIn("review_rank", visual.columns)
            # Ranked list must not dump every binary flag as equal priority.
            self.assertFalse((visual["priority"] == "flagged").any())
            severity_rows = visual[visual["priority"] == "severity"]
            self.assertFalse(severity_rows.empty)
            # First severity row should be severe before mild.
            first = severity_rows.sort_values("review_rank").iloc[0]
            self.assertEqual(first["qc_review_severity"], SEVERITY_SEVERE)

            params = json.loads((qc_dir / "qc_params.json").read_text(encoding="utf-8"))
            self.assertTrue(params["notes"]["thresholds_are_descriptive"])
            summary = pd.read_csv(qc_dir / "dataset_qc_summary.csv").iloc[0]
            self.assertGreaterEqual(int(summary["n_obs_severity_severe"]), 1)

    def test_assign_qc_review_severity_tiers(self) -> None:
        params = QcReportParams()
        mild = assign_qc_review_severity(
            {
                "review_ihr_disposition": FLAGGED_FOR_REVIEW,
                "observation_review_disposition": FLAGGED_FOR_REVIEW,
                "frac_hr_outside": 0.02,
                "ihr_jump_rate_per_min": 1.0,
                "max_consecutive_hr_jump": 25.0,
                "pct_gap_masked": 0.0,
            },
            params,
        )
        self.assertEqual(mild, SEVERITY_MILD)
        moderate = assign_qc_review_severity(
            {
                "review_ihr_disposition": FLAGGED_FOR_REVIEW,
                "frac_hr_outside": 0.06,
                "ihr_jump_rate_per_min": 6.0,
                "max_consecutive_hr_jump": 45.0,
                "pct_gap_masked": 0.0,
            },
            params,
        )
        self.assertEqual(moderate, SEVERITY_MODERATE)
        none = assign_qc_review_severity(
            {
                "review_ihr_disposition": NO_AUTOMATED_CONCERN,
                "observation_review_disposition": NO_AUTOMATED_CONCERN,
                "frac_hr_outside": 0.0,
                "ihr_jump_rate_per_min": 0.0,
                "max_consecutive_hr_jump": 5.0,
                "pct_gap_masked": 0.0,
            },
            params,
        )
        self.assertEqual(none, SEVERITY_NONE)

    def test_immutability_and_qc_only_writes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            before = _hash_tree(root)
            qc_dir = generate_qc_report(root, dataset_id="toy")
            after_stage = _hash_tree(root, exclude_qc=True)
            self.assertEqual(before, after_stage)
            self.assertTrue(qc_dir.is_dir())
            # Only QC/ should be new relative to pre-QC tree.
            after_all = _hash_tree(root)
            new_paths = sorted(set(after_all) - set(before))
            self.assertTrue(new_paths)
            self.assertTrue(all(p.startswith("QC/") for p in new_paths))
            for name in (
                "observation_qc.csv",
                "duration_qc.csv",
                "coupling_unit_qc.csv",
                "contrast_qc.csv",
                "null_unit_qc.csv",
                "model_qc.csv",
                "dataset_qc_summary.csv",
                "visual_review_list.csv",
                "QC_REPORT.md",
                "qc_params.json",
                "qc_manifest.json",
            ):
                self.assertTrue((qc_dir / name).is_file(), name)

    def test_grains_match_source_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            qc_dir = generate_qc_report(root, dataset_id="toy")
            c3 = pd.read_csv(root / "C3" / "confirmatory_endpoint_qc_D240.csv")
            coupling = pd.read_csv(qc_dir / "coupling_unit_qc.csv")
            c3_keys = set(
                zip(
                    c3["observation_id"].astype(str),
                    c3["duration_s"].astype(int),
                    c3["band"].astype(str),
                    c3["power_representation"].astype(str),
                )
            )
            coup_keys = set(
                zip(
                    coupling["observation_id"].astype(str),
                    coupling["duration_s"].astype(int),
                    coupling["band"].astype(str),
                    coupling["power_representation"].astype(str),
                )
            )
            self.assertEqual(c3_keys, coup_keys)

            c5 = pd.read_csv(root / "C5" / "paired_contrasts.csv")
            contrast = pd.read_csv(qc_dir / "contrast_qc.csv")
            self.assertEqual(len(c5), len(contrast))

            inf = pd.read_csv(root / "C6" / "inference_qc.csv")
            model = pd.read_csv(qc_dir / "model_qc.csv")
            self.assertEqual(len(inf), len(model))

    def test_missing_c4_c6_does_not_crash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=False)
            qc_dir = generate_qc_report(root, dataset_id="toy")
            manifest = json.loads((qc_dir / "qc_manifest.json").read_text(encoding="utf-8"))
            missing = " ".join(manifest["missing_stages_or_inputs"])
            self.assertIn("C4", missing)
            self.assertIn("C5", missing)
            self.assertIn("C6", missing)
            summary = pd.read_csv(qc_dir / "dataset_qc_summary.csv")
            domains = str(summary.iloc[0]["domains_not_assessed"])
            self.assertIn("contrast", domains)
            self.assertIn("null_unit", domains)
            self.assertIn("model", domains)
            # Empty grain files exist.
            self.assertEqual((qc_dir / "contrast_qc.csv").read_text(encoding="utf-8"), "")
            self.assertEqual((qc_dir / "null_unit_qc.csv").read_text(encoding="utf-8"), "")
            self.assertEqual((qc_dir / "model_qc.csv").read_text(encoding="utf-8"), "")

    def test_deterministic_tables_except_manifest_timestamp(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            qc1 = generate_qc_report(root, dataset_id="toy")
            tables = [
                "observation_qc.csv",
                "duration_qc.csv",
                "coupling_unit_qc.csv",
                "contrast_qc.csv",
                "null_unit_qc.csv",
                "model_qc.csv",
                "dataset_qc_summary.csv",
                "visual_review_list.csv",
                "qc_params.json",
            ]
            first = {name: (qc1 / name).read_bytes() for name in tables}
            first_md = (qc1 / "QC_REPORT.md").read_text(encoding="utf-8")
            first_manifest = json.loads((qc1 / "qc_manifest.json").read_text(encoding="utf-8"))
            qc2 = generate_qc_report(root, dataset_id="toy")
            for name in tables:
                self.assertEqual(first[name], (qc2 / name).read_bytes(), name)
            self.assertEqual(first_md, (qc2 / "QC_REPORT.md").read_text(encoding="utf-8"))
            second_manifest = json.loads((qc2 / "qc_manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(
                first_manifest["generated_at_utc"],
                second_manifest["generated_at_utc"],
            )
            first_manifest.pop("generated_at_utc")
            second_manifest.pop("generated_at_utc")
            self.assertEqual(first_manifest, second_manifest)

    def test_cli_qc_report_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            # Direct API covers CLI body; ensure mode wiring rejects stage combo.
            code = confirmatory_main(["--mode", "qc-report", "--stage", "C0"])
            self.assertEqual(code, 2)

    def test_params_disclaimer_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            _seed_minimal_tree(root, with_c4_c6=True)
            qc_dir = generate_qc_report(root, dataset_id="toy")
            params = json.loads((qc_dir / "qc_params.json").read_text(encoding="utf-8"))
            self.assertFalse(params["notes"]["affects_c0_c7_eligibility"])
            self.assertFalse(params["notes"]["affects_inference"])
            self.assertIn("non-binding", params["disclaimer"])
            report = (qc_dir / "QC_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("non-binding", report)
            obs = pd.read_csv(qc_dir / "observation_qc.csv")
            a = obs.loc[obs["observation_id"] == "obs-a"].iloc[0]
            self.assertEqual(a["review_polarity_disposition"], FLAGGED_FOR_REVIEW)
            self.assertTrue(str(a["review_polarity_triggering_rule"]))
            # obs-b should not be polarity-flagged (gap 150).
            b = obs.loc[obs["observation_id"] == "obs-b"].iloc[0]
            self.assertEqual(b["review_polarity_disposition"], NO_AUTOMATED_CONCERN)
            self.assertNotIn("n_rejected_ibi_est", obs.columns)


if __name__ == "__main__":
    unittest.main()
