from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.cardiac_controls_upstream import (
    CONTROL_ECG_MASK,
    CONTROL_PPG_MASK,
    OBSERVATION_CONTROLS_FILENAME,
    _apply_event_mask,
    _strict_common_support_anchor_count,
    _valid_pair_counts_by_lag,
    run_confirmatory_cardiac_controls_upstream,
)
from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI


class TestCardiacControlsUpstream(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _aligned_rows(self, *, dataset_id: str, observation_id: str) -> list[dict[str, object]]:
        times = np.arange(0, 240, dtype=float)
        hr = np.sin(2 * np.pi * times / 20.0)
        rows: list[dict[str, object]] = []
        for t, h in zip(times, hr, strict=True):
            row: dict[str, object] = {
                "dataset_id": dataset_id,
                "subject_id": "p01",
                "task": "rest",
                "condition": "rest",
                "observation_id": observation_id,
                "duration_s": 240,
                "duration_role": "primary",
                "time_s": float(t),
                "hr_z": float(h),
            }
            for band in ("theta", "alpha", "beta", "low_gamma"):
                row[f"{band}_absolute_log10_power_z"] = float(np.sin(2 * np.pi * (t + 2) / 20.0))
            rows.append(row)
        return rows

    def _baseline_rows(self, *, dataset_id: str, observation_id: str) -> list[dict[str, object]]:
        out = []
        for band in ("theta", "alpha", "beta", "low_gamma"):
            out.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": "p01",
                    "session_id": "single",
                    "observation_id": observation_id,
                    "condition": "rest",
                    "duration_s": 240,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "power_representation": "absolute_log10",
                    "band": band,
                    "eligible": True,
                    "endpoint_index": 0.12,
                    "n_common_support": 180,
                }
            )
        return out

    def _peak_rows(self, signal_type: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        peak_rows: list[dict[str, object]] = []
        for i in range(20, 220, 2):
            peak_rows.append(
                {
                    "dataset_id": "ds_test",
                    "subject_id": "p01",
                    "task": "rest",
                    "observation_id": "obs-1",
                    "peak_time_s": float(i) + 0.3,
                    "is_accepted_peak": True,
                    "peak_sample": i,
                }
            )
        qc_rows = [
            {
                "dataset_id": "ds_test",
                "observation_id": "obs-1",
                "signal_type": signal_type,
                "channel_used": "ECG" if signal_type == "ecg" else "photosensor",
                "detector_used": "ecg_rpeak" if signal_type == "ecg" else "ppg_peak",
                "detector_polarity": "normal",
                "sfreq": 500,
                "selection_reason": "unit-test",
            }
        ]
        return peak_rows, qc_rows

    def test_ecg_mask_control_computed_upstream(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0, c1b, c1c, c3, out = [root / x for x in ("C0", "C1b", "C1c", "C3", "C6")]
            self._write_csv(c0 / "protocol_audit.csv", [{"dataset_id": "ds_test", "cardiac_modality": "ECG"}])
            aligned = self._aligned_rows(dataset_id="ds_test", observation_id="obs-1")
            self._write_csv(c1c / "features_confirmatory_aligned_D240.csv", aligned)
            self._write_csv(c3 / "confirmatory_endpoint_metrics_D240.csv", self._baseline_rows(dataset_id="ds_test", observation_id="obs-1"))
            peaks, qc = self._peak_rows("ecg")
            self._write_csv(c1b / "obs-1" / "detected_peaks.csv", peaks)
            self._write_csv(c1b / "cardiac_peak_qc.csv", qc)

            result = run_confirmatory_cardiac_controls_upstream(
                c0_dir=c0,
                c1b_dir=c1b,
                c1c_dir=c1c,
                c3_dir=c3,
                output_dir=out,
            )
            self.assertTrue((out / OBSERVATION_CONTROLS_FILENAME).is_file())
            mask_rows = [r for r in result.observation_rows if r["control_type"] == CONTROL_ECG_MASK and r["band"] == "theta"]
            self.assertTrue(mask_rows)
            self.assertTrue(any(bool(r["computable"]) for r in mask_rows))

    def test_ppg_mask_named_and_not_r_peak(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0, c1b, c1c, c3, out = [root / x for x in ("C0", "C1b", "C1c", "C3", "C6")]
            self._write_csv(c0 / "protocol_audit.csv", [{"dataset_id": "ds_test", "cardiac_modality": "PPG"}])
            self._write_csv(c1c / "features_confirmatory_aligned_D240.csv", self._aligned_rows(dataset_id="ds_test", observation_id="obs-1"))
            self._write_csv(c3 / "confirmatory_endpoint_metrics_D240.csv", self._baseline_rows(dataset_id="ds_test", observation_id="obs-1"))
            peaks, qc = self._peak_rows("ppg")
            self._write_csv(c1b / "obs-1" / "detected_peaks.csv", peaks)
            self._write_csv(c1b / "cardiac_peak_qc.csv", qc)

            result = run_confirmatory_cardiac_controls_upstream(
                c0_dir=c0,
                c1b_dir=c1b,
                c1c_dir=c1c,
                c3_dir=c3,
                output_dir=out,
            )
            rows = [r for r in result.observation_rows if r["control_type"] == CONTROL_PPG_MASK and r["band"] == "theta"]
            self.assertTrue(rows)
            self.assertEqual(rows[0]["cardiac_event_type"], "ppg_systolic_peak")

    def test_declared_ecg_without_ecg_does_not_substitute_ppg(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c0, c1b, c1c, c3, out = [root / x for x in ("C0", "C1b", "C1c", "C3", "C6")]
            self._write_csv(c0 / "protocol_audit.csv", [{"dataset_id": "ds_test", "cardiac_modality": "ECG"}])
            self._write_csv(c1c / "features_confirmatory_aligned_D240.csv", self._aligned_rows(dataset_id="ds_test", observation_id="obs-1"))
            self._write_csv(c3 / "confirmatory_endpoint_metrics_D240.csv", self._baseline_rows(dataset_id="ds_test", observation_id="obs-1"))
            peaks, qc = self._peak_rows("ppg")
            self._write_csv(c1b / "obs-1" / "detected_peaks.csv", peaks)
            self._write_csv(c1b / "cardiac_peak_qc.csv", qc)

            result = run_confirmatory_cardiac_controls_upstream(
                c0_dir=c0,
                c1b_dir=c1b,
                c1c_dir=c1c,
                c3_dir=c3,
                output_dir=out,
            )
            ppg_rows = [r for r in result.observation_rows if r["control_type"] == CONTROL_PPG_MASK and r["band"] == "theta"]
            self.assertTrue(ppg_rows)
            self.assertFalse(bool(ppg_rows[0]["computable"]))
            self.assertIn("not valid substitutes", str(ppg_rows[0]["not_computable_reason"]))

    def test_ppg_event_mask_1hz_hits_integer_second_without_rounding(self) -> None:
        rows = [{"time_s": float(t), "theta_absolute_log10_power_z": 1.0} for t in range(10)]
        # At 1 Hz, +/-0.15s around 3.12 should only invalidate second 3.
        masked, n_masked = _apply_event_mask(rows, np.asarray([3.12], dtype=float), pre_s=0.15, post_s=0.15)
        self.assertEqual(n_masked, 1)
        invalid = [i for i, r in enumerate(masked) if not np.isfinite(float(r["theta_absolute_log10_power_z"]))]
        self.assertEqual(invalid, [3])

    def test_event_mask_high_rate_spans_multiple_samples(self) -> None:
        rows = []
        for t in np.arange(0.0, 2.0, 0.01):
            rows.append({"time_s": float(t), "theta_absolute_log10_power_z": 1.0})
        masked, n_masked = _apply_event_mask(rows, np.asarray([1.0], dtype=float), pre_s=0.15, post_s=0.15)
        # With 10 ms sampling in this construction, 30 samples are invalidated.
        self.assertEqual(n_masked, 30)
        self.assertTrue(np.isfinite(float(masked[0]["theta_absolute_log10_power_z"])))
        self.assertTrue(np.isfinite(float(masked[-1]["theta_absolute_log10_power_z"])))

    def test_event_mask_boundary_events_only_touch_available_samples(self) -> None:
        rows = [{"time_s": float(t), "theta_absolute_log10_power_z": 1.0} for t in range(240)]
        masked, n_masked = _apply_event_mask(rows, np.asarray([0.05, 239.95], dtype=float), pre_s=0.15, post_s=0.15)
        # At 1 Hz, the late event has no in-window sample to invalidate.
        self.assertEqual(n_masked, 1)
        invalid = [i for i, r in enumerate(masked) if not np.isfinite(float(r["theta_absolute_log10_power_z"]))]
        self.assertEqual(invalid, [0])

    def test_lagwise_pairs_can_remain_while_strict_common_support_is_zero(self) -> None:
        n = 240
        hr = np.ones(n, dtype=float)
        eeg_valid = np.ones(n, dtype=bool)
        # Invalidate every fourth second to mimic event-level sparse masking.
        eeg_valid[::4] = False
        counts = _valid_pair_counts_by_lag(hr, eeg_valid, lag_min=-60, lag_max=60)
        anchors = _strict_common_support_anchor_count(hr, eeg_valid, lag_max=60)
        self.assertGreater(min(counts.values()), 0)
        self.assertEqual(anchors, 0)

    def test_ppg_mask_on_1hz_is_structurally_incompatible_with_strict_zlpi(self) -> None:
        """Periodic ±0.15 s masks on 1 Hz grids zero anchors while pairwise support remains."""
        from ppg_eeg.confirmatory.panel_d_cardiac_controls import short_not_computable_reason_code

        n = 240
        times = np.arange(n, dtype=float)
        # ~75 bpm → events every 0.8 s, phase such that many integer seconds are hit.
        events = np.arange(0.12, 239.0, 0.8, dtype=float)
        rows = [{"time_s": float(t), "theta_absolute_log10_power_z": 1.0} for t in times]
        masked, n_masked = _apply_event_mask(rows, events, pre_s=0.15, post_s=0.15)
        self.assertGreater(n_masked, 40)
        eeg_valid = np.asarray(
            [np.isfinite(float(r["theta_absolute_log10_power_z"])) for r in masked],
            dtype=bool,
        )
        hr = np.ones(n, dtype=float)
        counts = _valid_pair_counts_by_lag(hr, eeg_valid, lag_min=-60, lag_max=60)
        anchors = _strict_common_support_anchor_count(hr, eeg_valid, lag_max=60)
        self.assertGreaterEqual(min(counts.values()), 60)
        self.assertEqual(anchors, 0)
        code = short_not_computable_reason_code(
            "1hz_envelope_cannot_support_event_centered_masking_for_strict_common_support_zlpi"
        )
        self.assertEqual(code, "1 Hz mask incompatible")
        # Legacy alias still maps to the same short code.
        self.assertEqual(
            short_not_computable_reason_code("strict_global_common_support_after_masking_at_1hz"),
            "1 Hz mask incompatible",
        )

    def test_adjacent_event_windows_mask_union_without_double_count(self) -> None:
        rows = [{"time_s": float(t), "theta_absolute_log10_power_z": 1.0} for t in np.arange(0.0, 3.0, 0.1)]
        # Adjacent windows overlap heavily around 1.0 and 1.1 seconds.
        masked, n_masked = _apply_event_mask(rows, np.asarray([1.0, 1.1], dtype=float), pre_s=0.15, post_s=0.15)
        invalid = [i for i, r in enumerate(masked) if not np.isfinite(float(r["theta_absolute_log10_power_z"]))]
        self.assertEqual(n_masked, len(invalid))
        self.assertGreater(n_masked, 0)

    def test_sparse_hr_validity_reduces_lag_pair_counts(self) -> None:
        hr = np.ones(240, dtype=float)
        hr[::3] = np.nan
        eeg_valid = np.ones(240, dtype=bool)
        counts = _valid_pair_counts_by_lag(hr, eeg_valid, lag_min=-60, lag_max=60)
        self.assertLess(min(counts.values()), max(counts.values()))


if __name__ == "__main__":
    unittest.main()
