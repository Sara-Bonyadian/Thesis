from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ppg_eeg.datasets import build_observations
from ppg_eeg.datasets.ds003690 import DS003690Adapter
from ppg_eeg.datasets.ds003838 import DS003838Adapter
from ppg_eeg.datasets.ds006848 import DS006848Adapter
from ppg_eeg.datasets.hiit import HIITAdapter
from ppg_eeg.temporal_coupling.data_audit import _read_bids_sidecar_payload


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
            self.assertEqual(rows[0].subject_id, "01_ph")

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
            self.assertEqual(rows[0].session_label, "single")
            self.assertEqual(rows[0].modality, "eeg_ecg_split")
            self.assertEqual(rows[0].timepoint, "na")
            self.assertEqual(rows[0].state, "rest")

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
            self.assertEqual(rows[0].session_label, "single")
            self.assertEqual(rows[0].modality, "eeg_ppg")
            self.assertEqual(rows[0].timepoint, "na")
            self.assertEqual(rows[0].state, "rest")

    def test_build_observations_subject_tasks_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds003838"
            for subject, task in [("sub-001", "rest"), ("sub-001", "memory"), ("sub-002", "rest")]:
                _touch(root / f"{subject}/eeg/{subject}_task-{task}_eeg.set")
                _touch(root / f"{subject}/ecg/{subject}_task-{task}_ecg.set")

            rows = build_observations(
                "ds003838",
                Path(tmp),
                subjects=["sub-001", "sub-002"],
                tasks=["rest", "memory"],
                subject_tasks={"sub-001": "memory", "sub-002": "rest"},
            )
            self.assertEqual(
                sorted((row.subject_id, row.task_label) for row in rows),
                [("sub-001", "memory"), ("sub-002", "rest")],
            )

    def test_build_observations_subject_conditions_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HIIT"
            folder = root / "HIIT_01_PH"
            for stem in ["HIIT_01_PH_PRE", "HIIT_01_PH_POST", "HIIT_01_PH_PRE_T"]:
                _touch(folder / f"{stem}.vhdr")
                _touch(folder / f"{stem}.eeg")
                _touch(folder / f"{stem}.vmrk")

            rows = build_observations(
                "hiit",
                Path(tmp),
                subjects=["01"],
                tasks=["rest", "tetris"],
                conditions=["ph_pre_rest", "ph_post_rest", "ph_pre_tetris"],
                sessions=["ph"],
                subject_conditions={"01": "ph_post_rest"},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].condition_label, "ph_post_rest")

    def test_ds003690_skips_macos_appledouble_set_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            eeg = Path(tmp) / "ds003690" / "sub-AB10" / "eeg"
            _touch(eeg / "sub-AB10_task-gonogo_run-1_eeg.set")
            _touch(eeg / "._sub-AB10_task-gonogo_run-1_eeg.set")
            rows = DS003690Adapter().build_observations(Path(tmp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].eeg_path.name, "sub-AB10_task-gonogo_run-1_eeg.set")

    def test_bids_sidecar_ignores_non_utf8_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "sub-AB10_task-gonogo_run-1_eeg.json"
            sidecar.write_bytes(b'{"SamplingFrequency": 50' + bytes([0xB0]) + b"}")
            payload = _read_bids_sidecar_payload(sidecar.with_suffix(".set"))
            self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
