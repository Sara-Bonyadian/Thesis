from __future__ import annotations

import json
import unittest
from pathlib import Path

from ppg_eeg.temporal_coupling.confirmatory.config import (
    load_dataset_config,
    load_master_config,
)
from ppg_eeg.temporal_coupling.confirmatory.protocol_audit import protocol_spec


class TestM13bHiitSmokeConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.config_dir = cls.repo / "zero-lag-reanalysis-repo"
        cls.master = load_master_config(
            cls.config_dir / "config.confirmatory.master.yaml"
        )
        cls.smoke_path = cls.config_dir / "config.confirmatory.smoke.hiit.yaml"
        cls.verification_path = (
            cls.config_dir / "m13b.hiit.smoke.verification.json"
        )

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
        self.assertEqual(
            {c.contrast_id for c in spec.contrasts},
            {
                "ph_pre_rest__tetris",
                "ph_post_rest__tetris",
                "ps_pre_rest__tetris",
                "ps_post_rest__tetris",
            },
        )

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
