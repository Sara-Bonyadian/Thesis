from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.config import PathsConfig, PipelineConfig
from ppg_eeg.preflight import run_preflight


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


class TestPreflight(unittest.TestCase):
    def test_preflight_writes_subject_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            out_root = Path(tmp) / "derivatives"
            dataset_root = raw_root / "ds003838"
            _touch(dataset_root / "sub-001/eeg/sub-001_task-rest_eeg.set")
            _touch(dataset_root / "sub-001/ecg/sub-001_task-rest_ecg.set")

            cfg = PipelineConfig(
                dataset_ids=["ds003838"],
                dataset_id="ds003838",
                paths=PathsConfig(raw_root=raw_root, out_root=out_root),
                subjects=["sub-001"],
                tasks=["rest"],
                conditions=["rest"],
                sessions=["single"],
            )
            report = run_preflight(cfg)
            self.assertTrue(report.ready)
            self.assertEqual(report.n_total, 1)
            manifest = out_root / "preflight" / "ds003838" / "sub-001.preflight.json"
            self.assertTrue(manifest.exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["ready"])
            self.assertTrue((out_root / "ds003838" / "sub-001").exists())

    def test_preflight_marks_missing_subject_as_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            out_root = Path(tmp) / "derivatives"
            (raw_root / "ds003838").mkdir(parents=True)
            cfg = PipelineConfig(
                dataset_ids=["ds003838"],
                dataset_id="ds003838",
                paths=PathsConfig(raw_root=raw_root, out_root=out_root),
                subjects=["sub-999"],
                tasks=["rest"],
                conditions=["rest"],
                sessions=["single"],
            )
            report = run_preflight(cfg)
            self.assertFalse(report.ready)
            self.assertEqual(report.n_ready, 0)


if __name__ == "__main__":
    unittest.main()
