from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ppg_eeg.datasets.ds003838 import DS003838Adapter
from ppg_eeg.datasets.ds006848 import DS006848Adapter
from ppg_eeg.datasets.hiit import HIITAdapter


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


class TestDatasetAdapters(unittest.TestCase):
    def test_hiit_adapter_parses_only_strict_canonical_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HIIT"
            folder = root / "HIIT_01_PH"
            _touch(folder / "HIIT_01_PH_PRE.vhdr")
            _touch(folder / "HIIT_01_PH_PRE.eeg")
            _touch(folder / "HIIT_01_PH_PRE.vmrk")

            # malformed/orphan files should be ignored
            _touch(folder / "HIIT_01_PH_PRE_.vhdr")
            _touch(folder / "HIIT_01_PH_PRE_.vmrk")
            _touch(folder / "H.vhdr")

            adapter = HIITAdapter()
            rows = adapter.build_observations(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].condition_label, "ph_pre_rest")

    def test_ds003838_requires_eeg_ecg_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds003838"
            _touch(root / "sub-001/eeg/sub-001_task-rest_eeg.set")
            _touch(root / "sub-001/ecg/sub-001_task-rest_ecg.set")
            _touch(root / "sub-001/eeg/sub-001_task-memory_eeg.set")
            # memory task lacks ecg pair and should be dropped

            adapter = DS003838Adapter()
            rows = adapter.build_observations(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].task_label, "rest")
            self.assertEqual(rows[0].ppg_source, "external_file")

    def test_ds006848_embedded_ppg_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds006848"
            stem = root / "sub-001/eeg/sub-001_task-rest_eeg"
            _touch(stem.with_suffix(".vhdr"))
            _touch(stem.with_suffix(".eeg"))
            _touch(stem.with_suffix(".vmrk"))

            adapter = DS006848Adapter()
            rows = adapter.build_observations(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].task_label, "rest")
            self.assertEqual(rows[0].ppg_source, "embedded_eeg")


if __name__ == "__main__":
    unittest.main()
