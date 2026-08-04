from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.confirmatory.config import load_dataset_config, load_master_config
from ppg_eeg.confirmatory.production import master_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "zero-lag-reanalysis-repo"
MASTER_PATH = master_config_path(CONFIG_DIR)


class TestAllConfirmatoryDatasetConfigs(unittest.TestCase):
    def test_all_dataset_configs_load(self) -> None:
        master = load_master_config(MASTER_PATH)
        dataset_dir = CONFIG_DIR / "datasets"
        yaml_paths = sorted(dataset_dir.glob("*.yaml"))
        self.assertGreaterEqual(len(yaml_paths), 8)
        for path in yaml_paths:
            with self.subTest(path=path.name):
                cfg = load_dataset_config(path, master=master)
                self.assertTrue(cfg.dataset_id)
                self.assertIn(cfg.role, {"primary", "sensitivity"})
                self.assertIsInstance(cfg.normalization.condition_semantics, dict)
                self.assertTrue(cfg.capabilities.has_eeg)


if __name__ == "__main__":
    unittest.main()

