from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ppg_eeg.config import FeaturesConfig, OutputConfig, PathsConfig, PipelineConfig
from ppg_eeg.features_core import base_eeg_power_columns
from ppg_eeg.pipeline import _write_base_eeg_csv


def _cfg_for_output_mode(mode: str, *, new_file_name: str = "features_extended.csv") -> PipelineConfig:
    return PipelineConfig(
        dataset_ids=["ds003838"],
        dataset_id="ds003838",
        paths=PathsConfig(raw_root=Path("."), out_root=Path("./derivatives")),
        subjects=[],
        tasks=[],
        conditions=[],
        sessions=[],
        features=FeaturesConfig(
            eeg=["power_theta", "power_alpha", "power_beta", "eeg_custom"],
            ppg=["ppg_mean_hr_bpm"],
            include_robust_z=False,
        ),
        output=OutputConfig(
            save_observation_index=False,
            save_summary_json=False,
            eeg_base_csv_mode=mode,
            eeg_base_csv_new_file_name=new_file_name,
        ),
    )


def _sample_eeg_df(custom_value: float) -> pd.DataFrame:
    row = {
        "dataset_id": "ds003838",
        "observation_id": "obs-01",
        "subject_id": "01",
        "task_label": "rest",
        "modality": "eeg_ecg_split",
        "state": "rest",
        "eeg_format": "eeglab",
        "eeg_path": "/tmp/fake.set",
        "n_bad_channels": 0,
        "eeg_error": "ok",
        "channel": "F3|F4|Fz",
        "power_theta": 0.1,
        "power_alpha": 0.2,
        "power_beta": 0.3,
        "eeg_custom": custom_value,
    }
    return pd.DataFrame([row])


class TestPipelineOutputs(unittest.TestCase):
    def test_append_columns_mode_adds_new_feature_column(self) -> None:
        cfg = _cfg_for_output_mode("append_columns")

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds003838"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            _write_base_eeg_csv(cfg, dataset_dir, _sample_eeg_df(1.23))

            out = pd.read_csv(dataset_dir / "features_base_eeg_power.csv")
            self.assertIn("eeg_custom", out.columns)
            self.assertAlmostEqual(float(out.loc[0, "eeg_custom"]), 1.23)
            for col in base_eeg_power_columns("export"):
                self.assertIn(col, out.columns)

    def test_new_file_mode_keeps_base_and_writes_extended(self) -> None:
        cfg = _cfg_for_output_mode("new_file", new_file_name="features_plus_new.csv")

        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "ds003838"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            _write_base_eeg_csv(cfg, dataset_dir, _sample_eeg_df(2.34))

            base_df = pd.read_csv(dataset_dir / "features_base_eeg_power.csv")
            ext_df = pd.read_csv(dataset_dir / "features_plus_new.csv")

            self.assertNotIn("eeg_custom", base_df.columns)
            self.assertIn("eeg_custom", ext_df.columns)
            self.assertAlmostEqual(float(ext_df.loc[0, "eeg_custom"]), 2.34)


if __name__ == "__main__":
    unittest.main()
