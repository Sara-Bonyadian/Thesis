from __future__ import annotations

import json
import unittest
from pathlib import Path

from ppg_eeg.temporal_coupling.confirmatory.config import (
    load_dataset_config,
    load_master_config,
)
from ppg_eeg.temporal_coupling.confirmatory.production import (
    iter_dataset_config_paths,
    iter_smoke_config_paths,
    master_config_path,
)
from ppg_eeg.temporal_coupling.confirmatory.protocol_audit import protocol_spec


class TestM13bHiitSmokeConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.config_dir = cls.repo / "zero-lag-reanalysis-repo"
        cls.hiit_smoke = cls.config_dir / "smoke" / "hiit"
        cls.master = load_master_config(master_config_path(cls.config_dir))
        cls.smoke_path = cls.hiit_smoke / "confirmatory.yaml"
        cls.verification_path = cls.hiit_smoke / "verification.json"
        cls.beats_path = cls.hiit_smoke / "beats.yaml"

    def test_layout_keeps_hiit_smoke_in_one_folder(self) -> None:
        self.assertTrue(self.smoke_path.is_file())
        self.assertTrue(self.verification_path.is_file())
        self.assertTrue(self.beats_path.is_file())
        smoke_paths = iter_smoke_config_paths(self.config_dir)
        self.assertIn(self.smoke_path.resolve(), [p.resolve() for p in smoke_paths])
        dataset_paths = iter_dataset_config_paths(self.config_dir)
        self.assertTrue(any(p.name == "hiit.yaml" for p in dataset_paths))

    def test_smoke_config_loads_with_protocol_task_pairs(self) -> None:
        cfg = load_dataset_config(self.smoke_path, master=self.master)
        self.assertEqual(cfg.dataset_id, "hiit")
        self.assertEqual(cfg.role, "sensitivity")
        self.assertEqual(cfg.selection.subjects, ("01", "02", "03"))
        self.assertEqual(cfg.selection.sessions, ("ph", "ps"))
        self.assertEqual(
            set(cfg.selection.conditions),
            {
                "ph_pre_rest",
                "ph_pre_tetris",
                "ph_post_rest",
                "ph_post_tetris",
                "ps_pre_rest",
                "ps_pre_tetris",
                "ps_post_rest",
                "ps_post_tetris",
            },
        )
        self.assertEqual(cfg.paths.output_subdir, Path("smoke/hiit_m13b"))

    def test_smoke_conditions_match_protocol_contrasts(self) -> None:
        cfg = load_dataset_config(self.smoke_path, master=self.master)
        spec = protocol_spec("hiit")
        contrast_conditions = {
            condition
            for contrast in spec.contrasts
            for condition in (
                contrast.low_demand_condition,
                contrast.cognitive_effort_condition,
            )
        }
        self.assertTrue(set(cfg.selection.conditions).issubset(contrast_conditions))

    def test_verification_companion_locks_ppg_photosensor(self) -> None:
        payload = json.loads(self.verification_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["primary_cardiac_modality"], "ppg")
        self.assertEqual(payload["primary_cardiac_channel"], "photosensor")
        self.assertEqual(payload["selection"]["participants"], ["01", "02", "03"])
        self.assertEqual(payload["duration_gates"]["check_durations_s"], [240, 180])
        self.assertIn(
            "never pool across protocols",
            " ".join(payload["pairing"]["notes"]).casefold(),
        )


if __name__ == "__main__":
    unittest.main()
