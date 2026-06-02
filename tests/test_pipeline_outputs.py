from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ppg_eeg.config import FeaturesConfig, OutputConfig, PathsConfig, PipelineConfig
from ppg_eeg.features_core import base_eeg_power_columns, base_ppg_ibi_columns
from ppg_eeg.pipeline import (
    CORRELATIONS_FDR_FILE,
    CORRELATIONS_RAW_FILE,
    DatasetStage1Artifacts,
    EEG_BASE_FILE,
    EEG_FEATURES_FILE,
    MERGED_FEATURES_FILE,
    OBSERVATIONS_INDEX_FILE,
    PPG_FEATURES_FILE,
    PPG_IBI_BASE_FILE,
    PipelineStage1Artifacts,
    run_stage2_from_base_csvs,
    write_stage1_artifacts,
    write_stage2_artifacts,
)


def _cfg_for_tmp_out(out_root: Path) -> PipelineConfig:
    return PipelineConfig(
        dataset_ids=["ds003838"],
        dataset_id="ds003838",
        paths=PathsConfig(raw_root=Path("."), out_root=out_root),
        subjects=[],
        tasks=[],
        conditions=[],
        sessions=[],
        features=FeaturesConfig(
            eeg=["theta_power_uv2", "alpha_power_uv2", "beta_power_uv2"],
            ppg=["ppg_mean_hr_bpm"],
            include_robust_z=False,
        ),
        output=OutputConfig(save_summary_json=False),
    )


def _sample_observations_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_id": "ds003838",
                "observation_id": "obs-01",
                "subject_id": "01",
                "task_label": "rest",
                "condition_label": "rest",
                "eeg_path": "/tmp/fake.set",
                "eeg_format": "eeglab",
                "ppg_source": "external_file",
                "ppg_path": "/tmp/fake-ecg.set",
                "ppg_format": "eeglab",
                "session_label": "single",
                "modality": "eeg_ecg_split",
                "timepoint": "",
                "state": "rest",
                "is_usable": True,
                "notes": "",
            }
        ]
    )


def _sample_eeg_base_df() -> pd.DataFrame:
    rows = []
    for channel, offset in [("F3", 0.0), ("F4", 1.0), ("Fz", 2.0)]:
        rows.append(
            {
                "dataset_id": "ds003838",
                "observation_id": "obs-01",
                "subject_id": "01",
                "task_label": "rest",
                "condition_label": "rest",
                "session_label": "single",
                "modality": "eeg_ecg_split",
                "timepoint": "",
                "state": "rest",
                "eeg_format": "eeglab",
                "eeg_path": "/tmp/fake.set",
                "n_bad_channels": 0,
                "eeg_error": "ok",
                "channel": channel,
                "theta_power_uv2": 1.0 + offset,
                "alpha_power_uv2": 2.0 + offset,
                "beta_power_uv2": 3.0 + offset,
                "l_freq": 1.0,
                "h_freq": 60.0,
                "reference": "average",
                "psd_fmin": 1.0,
                "psd_fmax": 60.0,
                "n_fft": 512,
                "processing_version": "eeg_power_schema_v2",
            }
        )
    return pd.DataFrame(rows, columns=base_eeg_power_columns("export"))


def _sample_ppg_ibi_df() -> pd.DataFrame:
    rows = []
    for idx, ibi in enumerate([800.0, 900.0, 1000.0]):
        rows.append(
            {
                "dataset_id": "ds003838",
                "observation_id": "obs-01",
                "subject_id": "01",
                "task_label": "rest",
                "condition_label": "rest",
                "session_label": "single",
                "modality": "eeg_ecg_split",
                "timepoint": "",
                "state": "rest",
                "eeg_path": "/tmp/fake.set",
                "eeg_format": "eeglab",
                "ppg_source": "external_file",
                "ppg_path": "/tmp/fake-ecg.set",
                "ppg_format": "eeglab",
                "ppg_error": "ok",
                "ppg_channel": "ECG",
                "ppg_segment_start_s": 0.0,
                "ppg_segment_end_s": 10.0,
                "sfreq": 1000.0,
                "peak_min_distance_s": 0.4,
                "peak_height": 0.3,
                "ibi_min_ms": 400.0,
                "ibi_max_ms": 1200.0,
                "n_peaks": 4,
                "n_ibi_raw": 3,
                "n_ibi_clean": 3,
                "ibi_index": idx,
                "peak_time_relative_s": float(idx + 1),
                "peak_time_absolute_s": float(idx + 1),
                "peak_index": idx,
                "ibi_ms_clean": ibi,
                "n_ibi_raw_invalid": 0,
                "ibi_ms_raw_min": 800.0,
                "ibi_ms_raw_max": 1000.0,
                "processing_version": "ppg_ibi_schema_v1",
            }
        )
    return pd.DataFrame(rows, columns=base_ppg_ibi_columns())


class TestPipelineOutputs(unittest.TestCase):
    def test_stage1_writes_only_required_base_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            cfg = _cfg_for_tmp_out(out_root)
            artifacts = PipelineStage1Artifacts(
                per_dataset={
                    "ds003838": DatasetStage1Artifacts(
                        dataset_id="ds003838",
                        observations=_sample_observations_df(),
                        eeg_base_features=_sample_eeg_base_df(),
                        ppg_ibi_features=_sample_ppg_ibi_df(),
                    )
                }
            )

            write_stage1_artifacts(cfg, artifacts)

            dataset_dir = out_root / "ds003838"
            self.assertTrue((dataset_dir / OBSERVATIONS_INDEX_FILE).exists())
            self.assertTrue((dataset_dir / EEG_BASE_FILE).exists())
            self.assertTrue((dataset_dir / PPG_IBI_BASE_FILE).exists())
            self.assertFalse((dataset_dir / EEG_FEATURES_FILE).exists())
            self.assertFalse((dataset_dir / PPG_FEATURES_FILE).exists())
            self.assertFalse((dataset_dir / MERGED_FEATURES_FILE).exists())
            self.assertFalse((dataset_dir / CORRELATIONS_RAW_FILE).exists())
            self.assertFalse((dataset_dir / CORRELATIONS_FDR_FILE).exists())

    def test_stage2_derives_outputs_from_base_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            cfg = _cfg_for_tmp_out(out_root)
            stage1 = PipelineStage1Artifacts(
                per_dataset={
                    "ds003838": DatasetStage1Artifacts(
                        dataset_id="ds003838",
                        observations=_sample_observations_df(),
                        eeg_base_features=_sample_eeg_base_df(),
                        ppg_ibi_features=_sample_ppg_ibi_df(),
                    )
                }
            )
            write_stage1_artifacts(cfg, stage1)

            artifacts = run_stage2_from_base_csvs(cfg)
            write_stage2_artifacts(cfg, artifacts)

            dataset_dir = out_root / "ds003838"
            eeg_df = pd.read_csv(dataset_dir / EEG_FEATURES_FILE)
            ppg_df = pd.read_csv(dataset_dir / PPG_FEATURES_FILE)
            merged_df = pd.read_csv(dataset_dir / MERGED_FEATURES_FILE)

            self.assertAlmostEqual(float(eeg_df.loc[0, "theta_power_uv2"]), 2.0)
            self.assertAlmostEqual(float(ppg_df.loc[0, "ppg_mean_hr_bpm"]), 60000.0 / 900.0)
            self.assertEqual(len(merged_df), 1)
            self.assertTrue((dataset_dir / CORRELATIONS_RAW_FILE).exists())
            self.assertTrue((dataset_dir / CORRELATIONS_FDR_FILE).exists())


if __name__ == "__main__":
    unittest.main()
