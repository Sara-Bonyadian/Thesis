"""Declared YAML capabilities must not alone make analyses computable."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from ppg_eeg.confirmatory.capability_resolution import (
    MIN_STANDARD_1020_CHANNELS,
    resolve_effective_capabilities,
    write_capability_resolution,
)
from ppg_eeg.confirmatory.config import DatasetNormalizationConfig
from ppg_eeg.confirmatory.dataset_contracts import DatasetCapabilities
from ppg_eeg.confirmatory.reason_codes import (
    INSUFFICIENT_COMMON_MONTAGE,
    MISSING_REQUIRED_BAND,
    MISSING_SENSOR_LOCATIONS,
    TOPOGRAPHY_NOT_SUPPORTED,
)
from ppg_eeg.datasets.base import CanonicalObservation

_STANDARD_1020_EEG = (
    ("Fp1", "EEG"),
    ("Fz", "EEG"),
    ("Cz", "EEG"),
    ("Pz", "EEG"),
    ("O1", "EEG"),
    ("O2", "EEG"),
    ("C3", "EEG"),
    ("C4", "EEG"),
)


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


def _declared_panel_f_caps() -> DatasetCapabilities:
    return DatasetCapabilities(
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


def _write_bids_eeg(
    tmp: Path,
    *,
    channels: tuple[tuple[str, str], ...],
    sfreq_hz: float,
) -> Path:
    eeg_path = tmp / "sub-01_task-rest_eeg.set"
    eeg_path.write_bytes(b"placeholder")
    (tmp / "sub-01_task-rest_eeg.json").write_text(
        json.dumps({"SamplingFrequency": sfreq_hz, "RecordingDuration": 300.0}),
        encoding="utf-8",
    )
    lines = ["name\ttype\tunits"]
    lines.extend(f"{name}\t{typ}\tuV" for name, typ in channels)
    (tmp / "sub-01_task-rest_channels.tsv").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return eeg_path


def _obs_for_eeg(eeg_path: Path) -> CanonicalObservation:
    return CanonicalObservation(
        dataset_id="ds003690",
        observation_id="obs-01",
        subject_id="sub-01",
        task_label="rest",
        condition_label="rest",
        eeg_path=eeg_path,
        eeg_format="eeglab",
        ppg_source="embedded_eeg",
        session_label="single",
    )


class TestCapabilityResolution(unittest.TestCase):
    def test_declared_true_without_files_is_effectively_false(self) -> None:
        caps = _declared_panel_f_caps()
        effective, rows = resolve_effective_capabilities(_dataset(caps=caps), [])
        self.assertFalse(effective.has_eeg)
        self.assertFalse(effective.has_hr)
        self.assertFalse(effective.supports_d240)
        self.assertFalse(effective.supports_topography)
        self.assertFalse(effective.supports_gamma)
        self.assertEqual(effective.cardiac_modality, "neither")
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["has_eeg"].declared_value, "true")
        self.assertEqual(by_cap["has_eeg"].effective_value, "false")
        self.assertTrue(by_cap["has_eeg"].reason_code)
        self.assertEqual(by_cap["supports_topography"].reason_code, TOPOGRAPHY_NOT_SUPPORTED)
        self.assertEqual(by_cap["supports_gamma"].reason_code, MISSING_REQUIRED_BAND)

    def test_eeg_file_presence_alone_does_not_grant_topography_or_gamma(self) -> None:
        caps = _declared_panel_f_caps()
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
        self.assertFalse(effective.supports_topography)
        self.assertFalse(effective.supports_gamma)
        by_cap = {r.capability: r for r in rows}
        self.assertIn("200", by_cap["supports_d240"].observed_evidence)
        self.assertEqual(by_cap["supports_topography"].reason_code, MISSING_SENSOR_LOCATIONS)
        self.assertEqual(by_cap["supports_gamma"].reason_code, MISSING_REQUIRED_BAND)
        self.assertIn("n_standard_1020=0", by_cap["supports_topography"].observed_evidence)

    def test_standard_1020_montage_and_nyquist_enable_panel_f(self) -> None:
        self.assertGreaterEqual(len(_STANDARD_1020_EEG), MIN_STANDARD_1020_CHANNELS)
        with tempfile.TemporaryDirectory() as tmp:
            eeg = _write_bids_eeg(
                Path(tmp), channels=_STANDARD_1020_EEG, sfreq_hz=500.0
            )
            effective, rows = resolve_effective_capabilities(
                _dataset(caps=_declared_panel_f_caps()),
                [_obs_for_eeg(eeg)],
                data_audit_rows=[{"eeg_exists": True, "raw_overlap_s": 300}],
            )
        self.assertTrue(effective.supports_topography)
        self.assertTrue(effective.supports_gamma)
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["supports_topography"].reason_code, "")
        self.assertEqual(by_cap["supports_gamma"].reason_code, "")
        self.assertIn("n_standard_1020=8", by_cap["supports_topography"].observed_evidence)
        self.assertIn("sfreq_hz=500", by_cap["supports_gamma"].observed_evidence)
        self.assertIn("nyquist_hz=250", by_cap["supports_gamma"].observed_evidence)

    def test_ecg_only_channels_do_not_support_topography_or_gamma(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eeg = _write_bids_eeg(
                Path(tmp), channels=(("EKG", "ECG"),), sfreq_hz=500.0
            )
            effective, rows = resolve_effective_capabilities(
                _dataset(caps=_declared_panel_f_caps()),
                [_obs_for_eeg(eeg)],
                data_audit_rows=[{"eeg_exists": True, "raw_overlap_s": 300}],
            )
        self.assertFalse(effective.supports_topography)
        self.assertFalse(effective.supports_gamma)
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["supports_topography"].reason_code, MISSING_SENSOR_LOCATIONS)
        self.assertEqual(by_cap["supports_gamma"].reason_code, MISSING_REQUIRED_BAND)

    def test_too_few_standard_1020_channels_is_insufficient_montage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eeg = _write_bids_eeg(
                Path(tmp),
                channels=_STANDARD_1020_EEG[:3],
                sfreq_hz=500.0,
            )
            effective, rows = resolve_effective_capabilities(
                _dataset(caps=_declared_panel_f_caps()),
                [_obs_for_eeg(eeg)],
                data_audit_rows=[{"eeg_exists": True, "raw_overlap_s": 300}],
            )
        self.assertFalse(effective.supports_topography)
        self.assertTrue(effective.supports_gamma)
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["supports_topography"].reason_code, INSUFFICIENT_COMMON_MONTAGE)
        self.assertIn("n_standard_1020=3", by_cap["supports_topography"].observed_evidence)

    def test_nyquist_below_low_gamma_blocks_gamma_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eeg = _write_bids_eeg(
                Path(tmp), channels=_STANDARD_1020_EEG, sfreq_hz=80.0
            )
            effective, rows = resolve_effective_capabilities(
                _dataset(caps=_declared_panel_f_caps()),
                [_obs_for_eeg(eeg)],
                data_audit_rows=[{"eeg_exists": True, "raw_overlap_s": 300}],
            )
        self.assertTrue(effective.supports_topography)
        self.assertFalse(effective.supports_gamma)
        by_cap = {r.capability: r for r in rows}
        self.assertEqual(by_cap["supports_gamma"].reason_code, MISSING_REQUIRED_BAND)
        self.assertIn("nyquist_hz=40", by_cap["supports_gamma"].observed_evidence)

    def test_observed_overlap_gates_duration_support(self) -> None:
        caps = _declared_panel_f_caps()
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
