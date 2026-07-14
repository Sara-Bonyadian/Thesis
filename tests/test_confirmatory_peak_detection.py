from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.confirmatory.config import load_dataset_config, load_master_config
from ppg_eeg.confirmatory.peak_detection import tc_config_for_peak_detection
from ppg_eeg.confirmatory.production import master_config_path


class TestConfirmatoryPeakDetectionConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.config_dir = cls.repo / "zero-lag-reanalysis-repo"
        cls.master = load_master_config(master_config_path(cls.config_dir))

    def test_hiit_smoke_builds_photosensor_tc_config(self) -> None:
        dataset = load_dataset_config(
            self.config_dir / "smoke" / "hiit" / "confirmatory.yaml",
            master=self.master,
        )
        cfg = tc_config_for_peak_detection(
            dataset, self.master, out_root=Path("/tmp/c1b_test")
        )
        self.assertEqual(cfg.temporal_coupling.cardiac.channel, "photosensor")
        self.assertEqual(cfg.temporal_coupling.cardiac.signal_type, "ppg")
        self.assertEqual(cfg.temporal_coupling.cardiac.detector, "ppg_peak")
        self.assertEqual(cfg.ppg.ibi_max_ms, 1500.0)
        self.assertEqual(cfg.ppg.peak_min_distance_s, 0.4)

    def test_ecg_smoke_defaults_to_auto_cardiac(self) -> None:
        dataset = load_dataset_config(
            self.config_dir / "smoke" / "ds003838.yaml",
            master=self.master,
        )
        self.assertEqual(dataset.cardiac.channel, "auto")
        self.assertEqual(dataset.cardiac.signal_type, "auto")
        self.assertEqual(dataset.cardiac.detector, "auto")


if __name__ == "__main__":
    unittest.main()
