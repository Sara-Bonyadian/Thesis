from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.figures import resolve_reporting_inputs


class TestFigure3ExportContract(unittest.TestCase):
    def test_resolve_reporting_inputs_discovers_panel_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {
                "null_subject": "null_subject_results.csv",
                "duration_sensitivity": "duration_sensitivity.csv",
                "cardiac_controls_observation": "cardiac_controls_observation_level.csv",
                "aligned_d240": "features_confirmatory_aligned_D240.csv",
            }
            for filename in expected.values():
                (root / filename).write_text("a,b\n1,2\n", encoding="utf-8")

            inputs = resolve_reporting_inputs(root)
            for key, filename in expected.items():
                self.assertIsNotNone(inputs[key])
                self.assertEqual(inputs[key].name, filename)


if __name__ == "__main__":
    unittest.main()

