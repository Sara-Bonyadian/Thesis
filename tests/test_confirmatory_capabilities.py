from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.config import (
    DatasetCardiacConfig,
    DatasetCapabilities,
    DatasetNormalizationConfig,
    DatasetPathsConfig,
    DatasetProtocolConfig,
    DatasetSelectionConfig,
    ConfirmatoryDatasetConfig,
)
from ppg_eeg.confirmatory.validation import validate_dataset_configuration
from ppg_eeg.datasets.base import CanonicalObservation


def _obs(*, dataset_id: str, condition: str, task: str = "", session: str = "single") -> CanonicalObservation:
    return CanonicalObservation(
        dataset_id=dataset_id,
        observation_id=f"{dataset_id}-obs-01",
        subject_id="sub-01",
        task_label=task or condition,
        condition_label=condition,
        eeg_path=Path("/tmp/missing_eeg.fif"),
        eeg_format="fif",
        ppg_source="embedded_eeg",
        session_label=session,
    )


def _cfg(*, dataset_id: str, capabilities: DatasetCapabilities) -> ConfirmatoryDatasetConfig:
    return ConfirmatoryDatasetConfig(
        schema_version=1,
        source_path=Path("dataset.yaml"),
        dataset_id=dataset_id,
        role="primary",
        paths=DatasetPathsConfig(raw_root=Path("."), output_subdir=Path("out")),
        selection=DatasetSelectionConfig(
            subjects=(),
            tasks=("rest", "task"),
            conditions=("rest", "task"),
            sessions=("single",),
        ),
        output_root=Path(TemporaryDirectory().name),
        cardiac=DatasetCardiacConfig(),
        protocol=DatasetProtocolConfig(),
        normalization=DatasetNormalizationConfig(condition_semantics={}),
        capabilities=capabilities,
    )


class TestConfirmatoryCapabilityValidation(unittest.TestCase):
    def test_missing_configured_condition_is_error(self) -> None:
        cfg = _cfg(
            dataset_id="ds003838",
            capabilities=DatasetCapabilities(
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
                supports_topography=False,
                supports_gamma=True,
            ),
        )
        issues = validate_dataset_configuration(cfg, [_obs(dataset_id="ds003838", condition="rest")])
        self.assertTrue(any(issue.code == "missing_condition_mapping" for issue in issues))

    def test_missing_hr_capability_becomes_warning(self) -> None:
        cfg = _cfg(
            dataset_id="mindfulness",
            capabilities=DatasetCapabilities(
                has_eeg=True,
                has_hr=False,
                cardiac_modality="ppg",
                has_ecg_r_peaks=False,
                has_ppg_peaks=False,
                has_low_high_task_pair=True,
                has_pre_post_pair=False,
                has_behavior=True,
                supports_d180=True,
                supports_d240=False,
                supports_topography=False,
                supports_gamma=False,
                sensitivity_only=True,
            ),
        )
        issues = validate_dataset_configuration(
            cfg,
            [_obs(dataset_id="mindfulness", condition="rest"), _obs(dataset_id="mindfulness", condition="task")],
        )
        self.assertTrue(any(issue.code == "missing_required_modality" for issue in issues))
        self.assertFalse(any(issue.severity == "error" and issue.code == "missing_required_modality" for issue in issues))


if __name__ == "__main__":
    unittest.main()

