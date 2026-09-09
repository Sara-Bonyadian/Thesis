from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import render_figure3
from ppg_eeg.confirmatory.panel_d_cardiac_controls import (
    CONTROL_BASELINE,
    CONTROL_BEAT_COUNT,
    CONTROL_ECG_CHANNELS,
    CONTROL_ICA_TEMPLATE,
    CONTROL_RPEAK_MASK,
    compute_panel_d_cardiac_controls,
)


def _paired_row(pid: str, *, contrast_id: str = "rest__memory", delta: float = -0.2) -> dict[str, object]:
    return {
        "dataset_id": "ds_test",
        "participant_id": pid,
        "session_id": "single",
        "contrast_id": contrast_id,
        "duration_s": 240,
        "endpoint_name": ENDPOINT_ZLPI,
        "band": "theta",
        "power_representation": "absolute_log10",
        "delta_endpoint_index": delta,
        "contrast_eligible": True,
    }


def _subject_row(
    pid: str,
    condition: str,
    *,
    beat_count: float,
    mean_hr: float,
) -> dict[str, object]:
    return {
        "dataset_id": "ds_test",
        "participant_id": pid,
        "session_id": "single",
        "condition": condition,
        "duration_s": 240,
        "endpoint_name": ENDPOINT_ZLPI,
        "band": "theta",
        "power_representation": "absolute_log10",
        "beat_count": beat_count,
        "mean_hr": mean_hr,
    }


def _cardiac_qc_row(
    dataset_id: str,
    *,
    signal_type: str,
    channel_used: str,
    detector_used: str,
    detector_polarity: str = "normal",
    sfreq: float = 500.0,
    n_raw_peaks: int = 260,
    n_clean_ibis: int = 240,
    n_accepted_peaks: int = 241,
    clean_ibi_coverage_s: float = 240.0,
    median_hr_bpm: float = 60.0,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "signal_type": signal_type,
        "channel_used": channel_used,
        "detector_used": detector_used,
        "detector_polarity": detector_polarity,
        "sfreq": sfreq,
        "n_raw_peaks": n_raw_peaks,
        "n_clean_ibis": n_clean_ibis,
        "n_accepted_peaks": n_accepted_peaks,
        "clean_ibi_coverage_s": clean_ibi_coverage_s,
        "median_hr_bpm": median_hr_bpm,
        "selection_reason": "unit-test",
    }


class TestFigure3PanelDCardiacControls(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _minimal_inputs(self, root: Path) -> dict[str, Path | None]:
        null_subject = root / "null_subject_results.csv"
        self._write_csv(
            null_subject,
            [
                {
                    "dataset_id": "ds_test",
                    "participant_id": "p0",
                    "subject_id": "p0",
                    "session_id": "single",
                    "condition": "rest",
                    "observation_id": "obs-0",
                    "band": "theta",
                    "null_type": "circular_shift",
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": 240,
                    "observed_endpoint_index": 0.3,
                    "null_mean": 0.1,
                    "empirical_p": 0.05,
                    "effect_size_surrogate_z": 1.2,
                    "n_surrogates_requested": 20,
                    "n_surrogates_finite": 20,
                    "observed_eligible": True,
                }
            ],
        )
        protocol = root / "protocol_audit.csv"
        self._write_csv(
            protocol,
            [{"dataset_id": "ds_test", "dataset_role": "primary", "cardiac_modality": "ECG"}],
        )
        duration = root / "duration_sensitivity.csv"
        self._write_csv(
            duration,
            [
                {
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "is_standard_zlpi": True,
                    "power_representation": "absolute_log10",
                    "band": "theta",
                    "dataset_id": "ds_test",
                    "contrast_id": "rest__memory",
                    "effect_estimate": -0.2,
                    "ci_low": -0.3,
                    "ci_high": -0.1,
                    "n": 4,
                    "p_value": 0.02,
                    "is_primary_analysis": True,
                    "can_rescue_primary": False,
                    "status": "primary_reference",
                    "notes": "",
                    "eligibility_status": "eligible",
                }
            ],
        )
        paired = root / "paired_contrasts.csv"
        self._write_csv(
            paired,
            [
                _paired_row("p0", delta=-0.22),
                _paired_row("p1", delta=-0.18),
                _paired_row("p2", delta=-0.21),
                _paired_row("p3", delta=-0.15),
            ],
        )
        subject = root / "subject_level_metrics.csv"
        subj_rows: list[dict[str, object]] = []
        for i, pid in enumerate(("p0", "p1", "p2", "p3")):
            subj_rows.append(_subject_row(pid, "rest", beat_count=250 + i, mean_hr=62 + i))
            subj_rows.append(_subject_row(pid, "memory", beat_count=258 + i, mean_hr=67 + i))
        self._write_csv(subject, subj_rows)
        sensitivity = root / "sensitivity_results.csv"
        self._write_csv(
            sensitivity,
            [
                {
                    "control_id": "broadband_residualized",
                    "dataset_id": "ds_test",
                    "band": "theta",
                    "contrast_id": "rest__memory",
                    "endpoint_name": ENDPOINT_ZLPI,
                    "duration_s": 240,
                    "power_representation": "broadband_residualized",
                    "effect_estimate": -0.17,
                    "ci_low": -0.30,
                    "ci_high": -0.05,
                    "n": 4,
                    "status": "sensitivity_only",
                }
            ],
        )
        spec = root / "specification_matrix.csv"
        self._write_csv(
            spec,
            [
                {
                    "control_id": "primary_d240_absolute_zlpi",
                    "effect_estimate": -0.2,
                    "ci_low": -0.3,
                    "ci_high": -0.1,
                    "n": 4,
                    "status": "primary_reference",
                }
            ],
        )
        return {
            "null_subject": null_subject,
            "protocol_audit": protocol,
            "cardiac_peak_qc": None,
            "cardiac_controls_observation": None,
            "cardiac_controls_dataset_qc": None,
            "cardiac_controls_metadata": None,
            "null_surrogate_values": None,
            "duration_sensitivity": duration,
            "paired_contrasts": paired,
            "subject_level": subject,
            "sensitivity": sensitivity,
            "specification_matrix": spec,
            "leave_one_out": None,
            "curves_d60": None,
            "curves_d120": None,
            "curves_d180": None,
            "curves_d240": None,
        }

    def test_controls_are_paired_and_locked_to_d240_zlpi(self) -> None:
        paired_rows = [
            _paired_row("p0", delta=-0.22),
            _paired_row("p1", delta=-0.18),
            _paired_row("p2", delta=-0.21),
            _paired_row("p3", delta=-0.15),
        ]
        subject_rows = []
        for i, pid in enumerate(("p0", "p1", "p2", "p3")):
            subject_rows.append(_subject_row(pid, "rest", beat_count=250 + i, mean_hr=62 + i))
            subject_rows.append(_subject_row(pid, "memory", beat_count=258 + i, mean_hr=67 + i))

        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows,
            protocol_rows=[{"dataset_id": "ds_test", "cardiac_modality": "ECG"}],
            cardiac_qc_rows=[_cardiac_qc_row("ds_test", signal_type="ecg", channel_used="ECG", detector_used="ecg_rpeak")],
        )
        rows = list(result.observations)
        baseline = [r for r in rows if r["control"] == CONTROL_BASELINE]
        self.assertEqual(len(baseline), 4)
        self.assertTrue(all(r["endpoint_name"] == ENDPOINT_ZLPI for r in baseline))
        self.assertTrue(all(int(r["duration_s"]) == 240 for r in baseline))

        for control in (
            CONTROL_ICA_TEMPLATE,
            CONTROL_RPEAK_MASK,
            CONTROL_BEAT_COUNT,
            CONTROL_ECG_CHANNELS,
        ):
            control_rows = [r for r in rows if r["control"] == control]
            self.assertEqual(len(control_rows), 4)
            by_key = {
                (r["dataset_id"], r["participant_id"], r["session_id"], r["state"], r["band"]): r
                for r in control_rows
            }
            for b in baseline:
                key = (b["dataset_id"], b["participant_id"], b["session_id"], b["state"], b["band"])
                self.assertIn(key, by_key)
                self.assertEqual(by_key[key]["baseline_endpoint_value"], b["endpoint_value"])

        beat_summary = next(s for s in result.summaries if s["control"] == CONTROL_BEAT_COUNT)
        self.assertEqual(beat_summary["computability_status"], "computed")

    def test_not_computable_controls_are_explicit(self) -> None:
        paired_rows = [_paired_row("p0"), _paired_row("p1"), _paired_row("p2")]
        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows=[],
            protocol_rows=[{"dataset_id": "ds_test", "cardiac_modality": "ECG"}],
            cardiac_qc_rows=[],
        )
        summary = {r["control"]: r for r in result.summaries}
        self.assertEqual(summary[CONTROL_ICA_TEMPLATE]["computability_status"], "not_computable")
        self.assertEqual(summary[CONTROL_RPEAK_MASK]["computability_status"], "not_computable")
        self.assertEqual(summary[CONTROL_ECG_CHANNELS]["computability_status"], "not_computable")
        self.assertIn("high-rate", str(summary[CONTROL_ICA_TEMPLATE]["computability_reason"]))
        self.assertIn("R-peak", str(summary[CONTROL_RPEAK_MASK]["computability_reason"]))

    def test_dataset_qc_ecg_and_ppg_semantics(self) -> None:
        paired_rows = [
            dict(_paired_row("p0"), dataset_id="ecg_set"),
            dict(_paired_row("p1"), dataset_id="ppg_set"),
        ]
        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows=[],
            protocol_rows=[
                {"dataset_id": "ecg_set", "cardiac_modality": "ECG"},
                {"dataset_id": "ppg_set", "cardiac_modality": "PPG"},
            ],
            cardiac_qc_rows=[
                _cardiac_qc_row("ecg_set", signal_type="ecg", channel_used="ECG", detector_used="ecg_rpeak"),
                _cardiac_qc_row("ppg_set", signal_type="ppg", channel_used="photosensor", detector_used="ppg_peak"),
            ],
        )
        qc = {row["dataset_id"]: row for row in result.dataset_qc}
        self.assertEqual(qc["ecg_set"]["cardiac_event_type"], "r_peak")
        self.assertEqual(qc["ppg_set"]["cardiac_event_type"], "ppg_systolic_peak")
        labels = result.metadata["control_display_labels"]
        self.assertEqual(labels[CONTROL_RPEAK_MASK], "Cardiac-event mask")

    def test_dataset_qc_uses_ecg_when_both_present_and_tracks_polarity(self) -> None:
        paired_rows = [dict(_paired_row("p0"), dataset_id="mixed_set")]
        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows=[],
            protocol_rows=[{"dataset_id": "mixed_set", "cardiac_modality": "ECG;PPG"}],
            cardiac_qc_rows=[
                _cardiac_qc_row("mixed_set", signal_type="ppg", channel_used="photosensor", detector_used="ppg_peak"),
                _cardiac_qc_row(
                    "mixed_set",
                    signal_type="ecg",
                    channel_used="ECG",
                    detector_used="ecg_rpeak",
                    detector_polarity="inverted",
                ),
            ],
        )
        self.assertEqual(len(result.dataset_qc), 1)
        row = result.dataset_qc[0]
        self.assertEqual(row["cardiac_signal_type"], "ECG")
        self.assertEqual(row["cardiac_event_type"], "r_peak")
        self.assertIn("inverted", str(row["detector_parameters"]))

    def test_dataset_qc_missing_signal_and_alignment_status(self) -> None:
        paired_rows = [dict(_paired_row("p0"), dataset_id="missing_set")]
        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows=[],
            protocol_rows=[{"dataset_id": "missing_set", "cardiac_modality": "ECG"}],
            cardiac_qc_rows=[],
        )
        row = result.dataset_qc[0]
        self.assertFalse(bool(row["computable"]))
        self.assertIn("unavailable", str(row["not_computable_reason"]))
        self.assertEqual(row["alignment_status"], "not_verifiable_from_frozen_outputs")

    def test_ecg_required_does_not_silently_substitute_ppg(self) -> None:
        paired_rows = [dict(_paired_row("p0"), dataset_id="ecg_required_set")]
        result = compute_panel_d_cardiac_controls(
            paired_rows,
            subject_rows=[],
            protocol_rows=[{"dataset_id": "ecg_required_set", "cardiac_modality": "ECG"}],
            cardiac_qc_rows=[
                _cardiac_qc_row(
                    "ecg_required_set",
                    signal_type="ppg",
                    channel_used="photosensor",
                    detector_used="ppg_peak",
                )
            ],
        )
        row = result.dataset_qc[0]
        self.assertFalse(bool(row["computable"]))
        self.assertIn("not valid substitutes", str(row["not_computable_reason"]))

    def test_rendered_panel_d_exports_and_caption_are_cardiac(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "frozen"
            out = Path(tmp) / "figures"
            root.mkdir(parents=True, exist_ok=True)
            result = render_figure3(self._minimal_inputs(root), out)
            self.assertTrue(result.manuscript.png.is_file())

            obs_csv = out / "source_data" / "figure3_panel_d_cardiac_controls_observations.csv"
            summary_csv = out / "source_data" / "figure3_panel_d_cardiac_controls_summaries.csv"
            dataset_qc_csv = out / "source_data" / "figure3_panel_d_cardiac_controls_dataset_qc.csv"
            metadata_json = out / "source_data" / "figure3_panel_d_cardiac_controls_metadata.json"
            self.assertTrue(obs_csv.is_file())
            self.assertTrue(summary_csv.is_file())
            self.assertTrue(dataset_qc_csv.is_file())
            self.assertTrue(metadata_json.is_file())
            self.assertTrue((out / "source_data" / "figure3_sensitivity_broadband_residualization.csv").is_file())
            self.assertFalse((out / "source_data" / "figure3_panel_d_broadband.csv").exists())

            with summary_csv.open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            controls = {r["control"] for r in summary_rows}
            self.assertEqual(
                controls,
                {
                    CONTROL_BASELINE,
                    CONTROL_ICA_TEMPLATE,
                    CONTROL_RPEAK_MASK,
                    CONTROL_BEAT_COUNT,
                    CONTROL_ECG_CHANNELS,
                },
            )
            by_control = {r["control"]: r for r in summary_rows}
            self.assertEqual(by_control[CONTROL_ICA_TEMPLATE]["computability_status"], "not_computable")
            self.assertEqual(by_control[CONTROL_RPEAK_MASK]["computability_status"], "not_computable")

            caption = (out / "figure3_caption.txt").read_text(encoding="utf-8").casefold()
            self.assertIn("cardiac-field and pulse-synchronous controls", caption)
            self.assertIn("are not represented as zero", caption)
            self.assertIn("does not establish neural origin", caption)
            self.assertIn("pulse-synchronous contamination", caption)
            self.assertIn("statistical sensitivity", caption)
            self.assertIn("observation-band rows", caption)
            self.assertIn("controlled", caption)
            self.assertIn("baseline", caption)
            self.assertIn("includes zero", caption)
            self.assertIn("unchanged", caption)
            self.assertNotIn("artifact-free", caption)
            self.assertNotIn("cardiac contamination ruled out", caption)
            self.assertNotIn("the inset reports", caption)

            metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["duration_s"], 240)
            self.assertEqual(metadata["bootstrap_clustering_unit"], "biological_participant_id")
            self.assertEqual(metadata["bootstrap_replicates"], 2000)

    def test_render_figure3_prefers_precomputed_upstream_controls(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "frozen"
            out = Path(tmp) / "figures"
            root.mkdir(parents=True, exist_ok=True)
            inputs = self._minimal_inputs(root)
            precomputed_obs = root / "cardiac_controls_observation_level.csv"
            precomputed_qc = root / "cardiac_controls_dataset_qc.csv"
            precomputed_meta = root / "cardiac_controls_metadata.json"
            self._write_csv(
                precomputed_obs,
                [
                    {
                        "dataset_id": "ds_test",
                        "participant_id": "p0",
                        "session_id": "single",
                        "observation_id": "obs-0",
                        "condition": "rest",
                        "band": "theta",
                        "control_type": "baseline",
                        "cardiac_signal_type": "ECG",
                        "cardiac_event_type": "r_peak",
                        "source_channel": "ECG",
                        "detector_name": "ecg_rpeak",
                        "detector_parameters": "{}",
                        "event_mask_window_pre_s": 0.05,
                        "event_mask_window_post_s": 0.05,
                        "n_events_detected": 200,
                        "n_events_accepted": 190,
                        "n_events_rejected": 10,
                        "event_rate_per_min": 60.0,
                        "median_inter_event_interval_s": 1.0,
                        "alignment_status": "pass",
                        "baseline_zlpi": -0.2,
                        "controlled_zlpi": -0.2,
                        "delta_vs_baseline": 0.0,
                        "n_valid_samples_baseline": 180,
                        "n_valid_samples_control": 180,
                        "minimum_lag_overlap": 60,
                        "computable": True,
                        "not_computable_reason": "",
                        "pipeline_stage": "C6",
                        "code_version": "test",
                    },
                    {
                        "dataset_id": "ds_test",
                        "participant_id": "p0",
                        "session_id": "single",
                        "observation_id": "obs-0",
                        "condition": "rest",
                        "band": "theta",
                        "control_type": "ecg_r_peak_mask",
                        "cardiac_signal_type": "ECG",
                        "cardiac_event_type": "r_peak",
                        "source_channel": "ECG",
                        "detector_name": "ecg_rpeak",
                        "detector_parameters": "{}",
                        "event_mask_window_pre_s": 0.05,
                        "event_mask_window_post_s": 0.05,
                        "n_events_detected": 200,
                        "n_events_accepted": 190,
                        "n_events_rejected": 10,
                        "event_rate_per_min": 60.0,
                        "median_inter_event_interval_s": 1.0,
                        "alignment_status": "pass",
                        "baseline_zlpi": -0.2,
                        "controlled_zlpi": -0.1,
                        "delta_vs_baseline": 0.1,
                        "n_valid_samples_baseline": 180,
                        "n_valid_samples_control": 170,
                        "minimum_lag_overlap": 60,
                        "computable": True,
                        "not_computable_reason": "",
                        "pipeline_stage": "C6",
                        "code_version": "test",
                    },
                ],
            )
            self._write_csv(
                precomputed_qc,
                [
                    {
                        "dataset_id": "ds_test",
                        "protocol_declared_cardiac_modalities": "ECG",
                        "source_file_modalities_found": "ECG",
                        "selected_cardiac_signal_type": "ECG",
                        "selected_source_channel": "ECG",
                        "cardiac_event_type": "r_peak",
                        "detector_name": "ecg_rpeak",
                        "detector_parameters": "",
                        "detector_execution_status": "ok",
                        "n_observations": 1,
                        "n_observations_computable": 1,
                    }
                ],
            )
            precomputed_meta.write_text(json.dumps({"schema_version": "test"}, indent=2), encoding="utf-8")
            inputs["cardiac_controls_observation"] = precomputed_obs
            inputs["cardiac_controls_dataset_qc"] = precomputed_qc
            inputs["cardiac_controls_metadata"] = precomputed_meta

            _result = render_figure3(inputs, out)
            summary_csv = out / "source_data" / "figure3_panel_d_cardiac_controls_summaries.csv"
            with summary_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            controls = {row["control"] for row in rows}
            self.assertIn("ecg_r_peak_mask", controls)


    def test_panel_d_visualization_row_order_and_nc_not_zero(self) -> None:
        from ppg_eeg.confirmatory.panel_d_cardiac_controls import (
            PANEL_D_PLOT_CONTROL_ORDER,
            compute_panel_d_from_observation_controls,
            verify_panel_d_summary_integrity,
        )

        rows = [
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "baseline",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": 0.10,
                "delta_vs_baseline": 0.0,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "baseline",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": 0.20,
                "delta_vs_baseline": 0.0,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "beat_count_adjusted",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": 0.12,
                "delta_vs_baseline": 0.02,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "beat_count_adjusted",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": 0.21,
                "delta_vs_baseline": 0.01,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "ppg_pulse_locked_template_subtraction",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": 0.15,
                "delta_vs_baseline": 0.05,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "ppg_pulse_locked_template_subtraction",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": 0.22,
                "delta_vs_baseline": 0.02,
                "computable": True,
                "not_computable_reason": "",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "ppg_systolic_peak_mask",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "strict_global_common_support_after_masking_at_1hz",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "ppg_systolic_peak_mask",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "strict_global_common_support_after_masking_at_1hz",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "ecg_cardiac_template_subtraction",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "ppg_only_no_ecg_template_control",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "ecg_cardiac_template_subtraction",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "ppg_only_no_ecg_template_control",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p0",
                "session_id": "single",
                "observation_id": "obs-0",
                "band": "theta",
                "control_type": "ecg_prone_channels_removed",
                "baseline_zlpi": 0.10,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "channel_level_reaggregation_not_computed_in_current_pipeline",
            },
            {
                "dataset_id": "ds",
                "participant_id": "p1",
                "session_id": "single",
                "observation_id": "obs-1",
                "band": "theta",
                "control_type": "ecg_prone_channels_removed",
                "baseline_zlpi": 0.20,
                "controlled_zlpi": "",
                "delta_vs_baseline": "",
                "computable": False,
                "not_computable_reason": "channel_level_reaggregation_not_computed_in_current_pipeline",
            },
        ]
        result = compute_panel_d_from_observation_controls(rows)
        summary_controls = [r["control"] for r in result.summaries]
        # Baseline first, then locked plot order.
        self.assertEqual(summary_controls[0], "baseline")
        self.assertEqual(summary_controls[1:6], list(PANEL_D_PLOT_CONTROL_ORDER))
        by_control = {r["control"]: r for r in result.summaries}
        self.assertEqual(by_control["beat_count_adjusted"]["control_family"], "statistical_sensitivity")
        self.assertEqual(
            by_control["ppg_pulse_locked_template_subtraction"]["control_family"],
            "signal_level",
        )
        self.assertEqual(by_control["ppg_systolic_peak_mask"]["computability_status"], "not_computable")
        for key in ("estimate", "ci_lower", "ci_upper", "change_from_baseline"):
            self.assertFalse(math.isfinite(float(by_control["ppg_systolic_peak_mask"][key])))
        self.assertEqual(by_control["ppg_pulse_locked_template_subtraction"]["denominator_label"], "2/2")
        # Δ = control − baseline
        ppg = by_control["ppg_pulse_locked_template_subtraction"]
        self.assertAlmostEqual(
            float(ppg["change_from_baseline"]),
            float(ppg["controlled_estimate"]) - float(ppg["baseline_estimate"]),
        )
        self.assertEqual(verify_panel_d_summary_integrity(result.summaries), [])

    def test_panel_d_dual_axis_layout_has_no_inset(self) -> None:
        from matplotlib.axes import Axes

        from ppg_eeg.confirmatory.figures import _render_figure3_panel_d_paired_axes
        from ppg_eeg.confirmatory.panel_d_cardiac_controls import (
            PANEL_D_PLOT_CONTROL_ORDER,
            compute_panel_d_from_observation_controls,
        )

        rows = []
        for pid, base, ctl in (("p0", 0.1, 0.12), ("p1", 0.2, 0.18)):
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "baseline",
                    "baseline_zlpi": base,
                    "controlled_zlpi": base,
                    "delta_vs_baseline": 0.0,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "beat_count_adjusted",
                    "baseline_zlpi": base,
                    "controlled_zlpi": ctl,
                    "delta_vs_baseline": ctl - base,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "ppg_pulse_locked_template_subtraction",
                    "baseline_zlpi": base,
                    "controlled_zlpi": ctl + 0.01,
                    "delta_vs_baseline": (ctl + 0.01) - base,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
        for control, reason in (
            ("ppg_systolic_peak_mask", "strict_global_common_support_after_masking_at_1hz"),
            ("ecg_cardiac_template_subtraction", "ppg_only_no_ecg_template_control"),
            ("ecg_prone_channels_removed", "channel_level_reaggregation_not_computed_in_current_pipeline"),
        ):
            for pid, base in (("p0", 0.1), ("p1", 0.2)):
                rows.append(
                    {
                        "dataset_id": "ds",
                        "participant_id": pid,
                        "session_id": "single",
                        "observation_id": f"obs-{pid}",
                        "band": "theta",
                        "control_type": control,
                        "baseline_zlpi": base,
                        "controlled_zlpi": "",
                        "delta_vs_baseline": "",
                        "computable": False,
                        "not_computable_reason": reason,
                    }
                )
        result = compute_panel_d_from_observation_controls(rows)
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 1)
        ax_coef, ax_delta, ax_status, plot_controls = _render_figure3_panel_d_paired_axes(
            fig, gs[0, 0], summaries=result.summaries
        )
        self.assertEqual(plot_controls, list(PANEL_D_PLOT_CONTROL_ORDER))
        self.assertIsInstance(ax_coef, Axes)
        self.assertIsInstance(ax_delta, Axes)
        self.assertIsInstance(ax_status, Axes)
        # No inset axes overlapping the coefficient panel.
        self.assertEqual(len(ax_coef.child_axes), 0)
        # Y labels live on the dedicated left label axis (text annotations, not ticks).
        ax_lab = next(
            ax
            for ax in fig.axes
            if ax not in (ax_coef, ax_delta, ax_status) and ax.get_ylabel() == "" and ax.axison
        )
        # Prefer the leftmost plot-body axis (label column).
        plot_body = [ax for ax in fig.axes if ax.get_position().y1 > 0.3]
        ax_lab = min(plot_body, key=lambda a: a.get_position().x0)
        for label in ax_coef.get_xticklabels():
            self.assertEqual(float(label.get_rotation()), 0.0)
        self.assertIn("Controlled coefficient", ax_coef.get_xlabel())
        self.assertIn("Change from baseline", ax_delta.get_xlabel())
        # Title hierarchy on the label axis.
        from ppg_eeg.confirmatory.figures import FIGURE3_PANEL_D_SUBTITLE, FIGURE3_PANEL_D_TITLE

        self.assertIn(FIGURE3_PANEL_D_TITLE, ax_lab.get_title(loc="left"))
        self.assertTrue(any(FIGURE3_PANEL_D_SUBTITLE in t.get_text() for t in ax_lab.texts))
        # NC status text present; no NC marker at x=0 on coefficient axis.
        status_texts = [t.get_text() for t in ax_status.texts]
        self.assertTrue(any(t.startswith("NC") for t in status_texts))
        self.assertTrue(any("observation-band" in t for t in status_texts))
        self.assertTrue(any(t.startswith("Computable") for t in status_texts))
        self.assertTrue(
            any("reagg. unavailable" in t for t in status_texts)
            or any("reaggregation unavailable" in t for t in status_texts)
            or any("artifact control unavailable" in t for t in status_texts)
        )
        self.assertTrue(
            any("1 Hz mask incompatible" in t for t in status_texts)
            or any(t == "not computable" for t in status_texts)
        )
        self.assertFalse(any("reaggregation unavailable" == t for t in status_texts))
        self.assertFalse(any(t == "insufficient support" for t in status_texts))
        # NC rows must not place numerical markers in the coefficient axis.
        n_errorbars = sum(1 for line in ax_coef.lines if len(line.get_xdata()) == 1)
        # Reference lines and connectors are allowed; status-only NC rows add no markers.
        nc_collections = []
        for coll in ax_coef.collections:
            offsets = np.asarray(coll.get_offsets())
            if offsets.size:
                nc_collections.append(offsets)
        # Only computable rows (2) should have baseline diamonds + controlled points.
        # Exactly two diamond+circle pairs expected in this fixture.
        self.assertGreaterEqual(len(ax_coef.collections) + n_errorbars, 1)
        plotted_x = []
        for line in ax_coef.lines:
            plotted_x.extend([float(v) for v in line.get_xdata() if math.isfinite(float(v))])
        for coll in ax_coef.collections:
            offsets = getattr(coll, "get_offsets", lambda: [])()
            for xy in offsets:
                if len(xy) >= 1 and math.isfinite(float(xy[0])):
                    plotted_x.append(float(xy[0]))
        # Zero reference line is allowed; NC must not invent a spurious estimate at 0
        # as a marker-only series without baseline pairing.
        expected_labels = [
            "Beat-count\nadjusted",
            "PPG template\nsubtraction",
            "PPG event mask",
            "ECG template\nsubtraction",
            "ECG-prone\nchannels rem.",
        ]
        row_label_texts = [
            t.get_text()
            for t in ax_lab.texts
            if t.get_text() in expected_labels
            or any(t.get_text() == lab for lab in expected_labels)
        ]
        # Preserve top-to-bottom plot order via y positions of label texts.
        labeled = sorted(
            (
                (float(np.asarray(t.get_position())[1]), t.get_text())
                for t in ax_lab.texts
                if t.get_text() in expected_labels
            ),
            key=lambda item: -item[0],
        )
        self.assertEqual([lab for _, lab in labeled], expected_labels)
        # Identical row order / y coordinates across axes.
        self.assertEqual(list(ax_coef.get_yticks()), list(ax_delta.get_yticks()))
        self.assertEqual(list(ax_coef.get_yticks()), list(ax_status.get_yticks()))
        self.assertEqual(list(ax_coef.get_yticks()), list(ax_lab.get_yticks()))
        plt.close(fig)

    def test_panel_d_layout_spacing_and_status_bounds(self) -> None:
        from ppg_eeg.confirmatory.figures import (
            FIGURE3_PANEL_D_WIDTH_RATIOS,
            FIGURE3_PANEL_D_WSPACE,
            _render_figure3_panel_d_paired_axes,
        )
        from ppg_eeg.confirmatory.panel_d_cardiac_controls import (
            compute_panel_d_from_observation_controls,
        )

        rows = []
        for pid, base, ctl in (("p0", 0.1, 0.12), ("p1", 0.2, 0.18)):
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "baseline",
                    "baseline_zlpi": base,
                    "controlled_zlpi": base,
                    "delta_vs_baseline": 0.0,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "beat_count_adjusted",
                    "baseline_zlpi": base,
                    "controlled_zlpi": ctl,
                    "delta_vs_baseline": ctl - base,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
            rows.append(
                {
                    "dataset_id": "ds",
                    "participant_id": pid,
                    "session_id": "single",
                    "observation_id": f"obs-{pid}",
                    "band": "theta",
                    "control_type": "ppg_pulse_locked_template_subtraction",
                    "baseline_zlpi": base,
                    "controlled_zlpi": ctl + 0.01,
                    "delta_vs_baseline": (ctl + 0.01) - base,
                    "computable": True,
                    "not_computable_reason": "",
                }
            )
        for control, reason in (
            ("ppg_systolic_peak_mask", "strict_global_common_support_after_masking_at_1hz"),
            ("ecg_cardiac_template_subtraction", "ppg_only_no_ecg_template_control"),
            (
                "ecg_prone_channels_removed",
                "channel_level_reaggregation_not_computed_in_current_pipeline",
            ),
        ):
            for pid, base in (("p0", 0.1), ("p1", 0.2)):
                rows.append(
                    {
                        "dataset_id": "ds",
                        "participant_id": pid,
                        "session_id": "single",
                        "observation_id": f"obs-{pid}",
                        "band": "theta",
                        "control_type": control,
                        "baseline_zlpi": base,
                        "controlled_zlpi": "",
                        "delta_vs_baseline": "",
                        "computable": False,
                        "not_computable_reason": reason,
                    }
                )
        result = compute_panel_d_from_observation_controls(rows)
        fig = plt.figure(figsize=(14.0, 5.2))
        gs = fig.add_gridspec(1, 1)
        ax_coef, ax_delta, ax_status, _plot_controls = _render_figure3_panel_d_paired_axes(
            fig, gs[0, 0], summaries=result.summaries
        )
        fig.subplots_adjust(left=0.28, right=0.985, bottom=0.24, top=0.78)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # X-axis labels must remain horizontally separated.
        coef_label = ax_coef.xaxis.label
        delta_label = ax_delta.xaxis.label
        coef_bbox = coef_label.get_window_extent(renderer=renderer)
        delta_bbox = delta_label.get_window_extent(renderer=renderer)
        self.assertLess(coef_bbox.x1, delta_bbox.x0)

        # Status column starts to the right of the Δ axes.
        coef_pos = ax_coef.get_position()
        delta_pos = ax_delta.get_position()
        status_pos = ax_status.get_position()
        self.assertLess(coef_pos.x1, delta_pos.x0)
        self.assertLess(delta_pos.x1, status_pos.x0)

        # Row status/denom/reason text must not invade the Δ plotting region.
        delta_right = delta_pos.x1 * fig.bbox.width
        for text in ax_status.texts:
            raw = text.get_text()
            if raw.startswith("NC =") or "observation-band" in raw or "baseline-eligible" in raw:
                continue
            if raw == "Status | n/N | reason":
                continue
            tb = text.get_window_extent(renderer=renderer)
            self.assertGreaterEqual(tb.x0, delta_right - 2.0)
            self.assertLessEqual(tb.x1, fig.bbox.x1 + 12.0)

        # Shared y coordinates across all three axes.
        self.assertEqual(list(ax_coef.get_yticks()), list(ax_delta.get_yticks()))
        self.assertEqual(list(ax_coef.get_yticks()), list(ax_status.get_yticks()))
        self.assertEqual(ax_coef.get_ylim(), ax_delta.get_ylim())
        self.assertEqual(ax_coef.get_ylim(), ax_status.get_ylim())

        # Nested layout: wider coefficient/Δ axes, compact labels/status.
        self.assertEqual(FIGURE3_PANEL_D_WIDTH_RATIOS, (0.95, 1.78, 1.58, 1.18))
        self.assertLessEqual(FIGURE3_PANEL_D_WSPACE, 0.12)
        self.assertEqual(len(FIGURE3_PANEL_D_WIDTH_RATIOS), 4)
        self.assertGreater(FIGURE3_PANEL_D_WIDTH_RATIOS[1], FIGURE3_PANEL_D_WIDTH_RATIOS[0])
        self.assertGreater(FIGURE3_PANEL_D_WIDTH_RATIOS[2], FIGURE3_PANEL_D_WIDTH_RATIOS[3])
        status_texts = " ".join(t.get_text() for t in ax_status.texts)
        self.assertIn("observation-band", status_texts)
        # Legend lives in the dedicated footer row below the plot axes.
        legends = [
            child
            for ax in fig.axes
            for child in ax.get_children()
            if child.__class__.__name__ == "Legend"
        ]
        self.assertTrue(legends)
        fig.canvas.draw()
        legend_bbox = legends[0].get_window_extent(renderer=fig.canvas.get_renderer())
        self.assertGreater(legend_bbox.y0, fig.bbox.y0 + 2.0)
        # Legend must sit below the coefficient axis so it cannot overlap x-labels.
        coef_bottom = ax_coef.get_position().y0 * fig.bbox.height
        self.assertLessEqual(legend_bbox.y1, coef_bottom + 8.0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
