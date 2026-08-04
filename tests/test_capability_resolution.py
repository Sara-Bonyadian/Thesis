"""Declared YAML capabilities must not alone make analyses computable."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ppg_eeg.confirmatory.capability_resolution import (
    resolve_effective_capabilities,
    write_capability_resolution,
)
from ppg_eeg.confirmatory.config import DatasetNormalizationConfig
from ppg_eeg.confirmatory.dataset_contracts import DatasetCapabilities


def _dataset(*, caps: DatasetCapabilities) -> MagicMock:
    ds = MagicMock()
    ds.capabilities = caps
    ds.normalization = DatasetNormalizationConfig(
        condition_semantics={
            "rest": {"state": "state_low", "time": "time_na", "session": "single"},
            "task": {"state": "state_high", "time": "time_na", "session": "single"},
        }
    )
    return ds


class TestCapabilityResolution(unittest.TestCase):
    def test_declared_true_without_files_is_effectively_false(self) -> None:
        caps = DatasetCapabilities(
            has_eeg=True,
            has_hr=True,
            cardiac_modality="ecg",
            has_ecg_r_peaks=True,
            has_ppg_peaks=False,
            has_low_high_task_pair=True,
            has_pre_post_pair=False,
            has_behavior=False,
            supports_d180=True,
            supports_d240=True,
            supports_topography=True,
            supports_gamma=True,
            sensitivity_only=False,
            has_artifact_controls=True,
        )
        effective, rows = resolve_effective_capabilities(_dataset(caps=caps), [])
        self.assertFalse(effective.has_eeg)
        self.assertFalse(effective.has_hr)
        self.assertFalse(effective.supports_d240)
        self.assertEqual(effective.cardiac_modality, "neither")
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["has_eeg"].declared_value, "true")
        self.assertEqual(by_cap["has_eeg"].effective_value, "false")
        self.assertTrue(by_cap["has_eeg"].reason_code)

    def test_observed_overlap_gates_duration_support(self) -> None:
        caps = DatasetCapabilities(
            has_eeg=True,
            has_hr=True,
            cardiac_modality="ecg",
            has_ecg_r_peaks=True,
            has_ppg_peaks=False,
            has_low_high_task_pair=True,
            has_pre_post_pair=False,
            has_behavior=False,
            supports_d180=True,
            supports_d240=True,
            supports_topography=True,
            supports_gamma=True,
            sensitivity_only=False,
            has_artifact_controls=True,
        )
        audit = [
            {
                "eeg_exists": True,
                "cardiac_exists": True,
                "cardiac_signal_type": "ECG",
                "raw_overlap_s": 200,
            }
        ]
        obs = MagicMock()
        obs.eeg_path = MagicMock()
        obs.eeg_path.is_file.return_value = True
        obs.ppg_path = MagicMock()
        obs.ppg_path.is_file.return_value = True
        obs.ppg_source = "external"
        obs.condition_label = "rest"
        effective, rows = resolve_effective_capabilities(
            _dataset(caps=caps), [obs], data_audit_rows=audit
        )
        self.assertTrue(effective.supports_d180)
        self.assertFalse(effective.supports_d240)
        by_cap = {r.capability: r for r in rows}
        self.assertIn("200", by_cap["supports_d240"].observed_evidence)

    def test_write_capability_resolution_export(self) -> None:
        caps = DatasetCapabilities(
            has_eeg=False,
            has_hr=False,
            cardiac_modality="neither",
            has_ecg_r_peaks=False,
            has_ppg_peaks=False,
            has_low_high_task_pair=False,
            has_pre_post_pair=False,
            has_behavior=False,
            supports_d180=False,
            supports_d240=False,
            supports_topography=False,
            supports_gamma=False,
            sensitivity_only=True,
            has_artifact_controls=False,
        )
        _effective, rows = resolve_effective_capabilities(_dataset(caps=caps), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_capability_resolution(rows, tmp)
            self.assertTrue(Path(path).is_file())
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("capability,declared_value,observed_evidence,effective_value", text)


if __name__ == "__main__":
    unittest.main()
