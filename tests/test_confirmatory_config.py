from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from ppg_eeg.temporal_coupling.confirmatory.config import (
    EXPECTED_BANDS_HZ,
    EXPECTED_DURATIONS_S,
    load_dataset_config,
    load_master_config,
)
from ppg_eeg.temporal_coupling.confirmatory.production import master_config_path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "zero-lag-reanalysis-repo"
MASTER_PATH = master_config_path(CONFIG_DIR)
SMOKE_PATH = CONFIG_DIR / "smoke" / "ds003838.yaml"


def _master_payload() -> dict[str, object]:
    payload = yaml.safe_load(MASTER_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestConfirmatoryMasterConfig(unittest.TestCase):
    def test_repository_master_config_has_fixed_protocol(self) -> None:
        cfg = load_master_config(MASTER_PATH)

        self.assertEqual(
            (cfg.bands.theta.low_hz, cfg.bands.theta.high_hz),
            EXPECTED_BANDS_HZ["theta"],
        )
        self.assertEqual(
            (cfg.bands.alpha.low_hz, cfg.bands.alpha.high_hz),
            EXPECTED_BANDS_HZ["alpha"],
        )
        self.assertEqual(
            (cfg.bands.beta.low_hz, cfg.bands.beta.high_hz),
            EXPECTED_BANDS_HZ["beta"],
        )
        self.assertEqual(
            (cfg.bands.low_gamma.low_hz, cfg.bands.low_gamma.high_hz),
            EXPECTED_BANDS_HZ["low_gamma"],
        )
        self.assertEqual(cfg.durations.all_s, EXPECTED_DURATIONS_S)
        self.assertEqual(cfg.durations.primary_s, 240)
        self.assertEqual(cfg.lag.step_s, 1)
        self.assertEqual((cfg.lag.min_s, cfg.lag.max_s), (-60, 60))
        self.assertEqual(
            (cfg.lag.by_duration[240].min_s, cfg.lag.by_duration[240].max_s),
            (-60, 60),
        )
        self.assertEqual(
            (cfg.lag.by_duration[120].min_s, cfg.lag.by_duration[120].max_s),
            (-30, 30),
        )
        self.assertEqual(
            (cfg.lag.by_duration[60].min_s, cfg.lag.by_duration[60].max_s),
            (-20, 20),
        )
        self.assertEqual(cfg.endpoints.zlpi_flanks_s, (20, 60))
        self.assertEqual(cfg.endpoints.shoulders_s, (5, 15))
        self.assertEqual(cfg.endpoints.peak_center_equivalence_s, 2)
        self.assertEqual(cfg.endpoints.by_duration[240].name, "zlpi")
        self.assertEqual(
            cfg.endpoints.by_duration[120].name, "mid_window_proximal_index"
        )
        self.assertEqual(
            cfg.endpoints.by_duration[60].name, "short_window_proximal_index"
        )
        self.assertFalse(cfg.endpoints.by_duration[120].is_standard_zlpi)
        self.assertFalse(cfg.endpoints.by_duration[120].pool_with_standard_zlpi)
        self.assertFalse(cfg.endpoints.by_duration[60].pool_with_standard_zlpi)
        self.assertTrue(cfg.endpoints.by_duration[180].pool_with_standard_zlpi)
        self.assertEqual(cfg.cardiac.hr_mode, "instantaneous")
        self.assertGreaterEqual(cfg.root_seed, 0)
        self.assertIn("confirmatory", str(cfg.output_root).casefold())
        self.assertEqual(
            cfg.dataset_roles.primary,
            ("ds003838", "ds006848", "ds003690", "ds004587"),
        )
        self.assertEqual(
            cfg.dataset_roles.sensitivity,
            ("ds004582", "ds003816", "hiit", "mindfulness"),
        )

    def test_unknown_key_rejected_at_top_level_and_nested_level(self) -> None:
        for path, key in [
            ((), "unexpected"),
            (("confirmatory",), "unexpected"),
            (("confirmatory", "bands"), "delta"),
            (("confirmatory", "endpoints"), "other_window"),
        ]:
            with self.subTest(path=path, key=key), TemporaryDirectory() as tmp:
                payload = _master_payload()
                target: dict[str, object] = payload
                for part in path:
                    value = target[part]
                    self.assertIsInstance(value, dict)
                    target = value  # type: ignore[assignment]
                target[key] = 1
                config_path = Path(tmp) / "master.yaml"
                config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "Unknown key"):
                    load_master_config(config_path)

    def test_protocol_constants_cannot_be_overridden(self) -> None:
        mutations = [
            (("confirmatory", "bands", "theta"), [4, 8]),
            (("confirmatory", "bands", "low_gamma"), [30, 44]),
            (("confirmatory", "durations", "all_s"), [240, 180, 120]),
            (("confirmatory", "durations", "primary_s"), 180),
            (("confirmatory", "lag", "step_s"), 5),
            (("confirmatory", "lag", "by_duration", 120, "max_s"), 60),
            (("confirmatory", "endpoints", "shoulders_s"), [5, 10]),
            (("confirmatory", "endpoints", "peak_center_equivalence_s"), 3),
            (
                ("confirmatory", "endpoints", "by_duration", 120, "name"),
                "zlpi",
            ),
            (
                ("confirmatory", "endpoints", "by_duration", 120, "pool_with_standard_zlpi"),
                True,
            ),
            (
                ("confirmatory", "endpoints", "by_duration", 240, "flanks_s"),
                [15, 60],
            ),
            (("confirmatory", "cardiac", "hr_mode"), "windowed"),
        ]
        for path, replacement in mutations:
            with self.subTest(path=path), TemporaryDirectory() as tmp:
                payload = _master_payload()
                target: dict[str, object] = payload
                for part in path[:-1]:
                    value = target[part]
                    self.assertIsInstance(value, dict)
                    target = value  # type: ignore[assignment]
                target[path[-1]] = replacement
                config_path = Path(tmp) / "master.yaml"
                config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

                with self.assertRaises(ValueError):
                    load_master_config(config_path)

    def test_dataset_roles_must_be_disjoint(self) -> None:
        payload = _master_payload()
        confirmatory = payload["confirmatory"]
        self.assertIsInstance(confirmatory, dict)
        roles = confirmatory["dataset_roles"]  # type: ignore[index]
        self.assertIsInstance(roles, dict)
        roles["sensitivity"].append("ds003838")  # type: ignore[index, union-attr]

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "master.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "both primary and sensitivity"):
                load_master_config(config_path)


class TestConfirmatoryDatasetConfig(unittest.TestCase):
    def test_repository_smoke_config_loads_against_master(self) -> None:
        master = load_master_config(MASTER_PATH)
        cfg = load_dataset_config(SMOKE_PATH, master=master)

        self.assertEqual(cfg.dataset_id, "ds003838")
        self.assertEqual(cfg.role, "primary")
        self.assertEqual(cfg.selection.tasks, ("rest", "memory"))
        self.assertEqual(
            cfg.selection.subjects, ("sub-032", "sub-033", "sub-034")
        )
        self.assertEqual(cfg.output_root, master.output_root / "smoke" / "ds003838")
        self.assertNotEqual(cfg.output_root, cfg.paths.raw_root)

    def test_dataset_role_must_match_master(self) -> None:
        master = load_master_config(MASTER_PATH)
        payload = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
        payload["role"] = "sensitivity"

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "dataset.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assigned role"):
                load_dataset_config(config_path, master=master)

    def test_unknown_dataset_key_rejected(self) -> None:
        master = load_master_config(MASTER_PATH)
        payload = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
        payload["selection"]["pairing"] = "not-yet-supported"

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "dataset.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown key"):
                load_dataset_config(config_path, master=master)

    def test_output_subdir_cannot_escape_confirmatory_root(self) -> None:
        master = load_master_config(MASTER_PATH)
        payload = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
        payload["paths"]["output_subdir"] = "../legacy"

        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "dataset.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain"):
                load_dataset_config(config_path, master=master)


if __name__ == "__main__":
    unittest.main()
