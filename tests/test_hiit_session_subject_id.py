from __future__ import annotations

import unittest

from ppg_eeg.datasets.hiit import (
    hiit_participant_id_from_session_subject,
    hiit_session_subject_id,
)


class TestHiitSessionSubjectId(unittest.TestCase):
    def test_session_subject_id(self) -> None:
        self.assertEqual(hiit_session_subject_id("01", "PS"), "01_ps")
        self.assertEqual(hiit_session_subject_id("20", "ph"), "20_ph")

    def test_participant_id_from_session_subject(self) -> None:
        self.assertEqual(hiit_participant_id_from_session_subject("01_ps"), "01")
        self.assertEqual(hiit_participant_id_from_session_subject("20_ph"), "20")


if __name__ == "__main__":
    unittest.main()
