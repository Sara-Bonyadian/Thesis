from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.bids_physio import physio_channel_type, read_physio_channel_info, read_physio_raw
from ppg_eeg.temporal_coupling.cardiac_common import infer_signal_type, list_cardiac_candidates


class TestBidsPhysio(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.physio_path = Path(
            "data/raw/ds004582/sub-FFE149/ses-01/beh/sub-FFE149_ses-01_task-FF_run-01_physio.tsv.gz"
        )
        if not cls.physio_path.is_file():
            raise unittest.SkipTest("ds004582 physio sample not available locally")

    def test_channel_types(self) -> None:
        self.assertEqual(physio_channel_type("ECGBIT"), "ecg")
        self.assertEqual(physio_channel_type("OXIBIT"), "bio")
        self.assertEqual(infer_signal_type("ECGBIT"), "ecg")
        self.assertEqual(infer_signal_type("OXIBIT"), "ppg")

    def test_read_physio_channel_info(self) -> None:
        info = read_physio_channel_info(self.physio_path)
        self.assertIsNotNone(info)
        assert info is not None
        ch_names, ch_types, sfreq, duration_s = info
        self.assertIn("ECGBIT", ch_names)
        self.assertIn("OXIBIT", ch_names)
        self.assertEqual(sfreq, 1000.0)
        self.assertGreater(duration_s, 1000.0)
        self.assertEqual(ch_types[ch_names.index("ECGBIT")], "ecg")
        self.assertEqual(ch_types[ch_names.index("OXIBIT")], "bio")

    def test_peak_detection_candidates_include_ecg_and_ppg(self) -> None:
        raw = read_physio_raw(self.physio_path)
        auto = list_cardiac_candidates(raw, "auto")
        self.assertEqual(set(auto), {"ECGBIT", "OXIBIT"})
        self.assertEqual(list_cardiac_candidates(raw, "ecg"), ["ECGBIT"])
        self.assertEqual(list_cardiac_candidates(raw, "ppg"), ["OXIBIT"])


if __name__ == "__main__":
    unittest.main()
