"""HIIT regression baseline: exact identifiers + tight numeric tolerances."""

from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.confirmatory.hiit_regression import (
    DEFAULT_SNAPSHOT_DIR,
    compare_csv_to_snapshot,
    compare_hiit_root_to_snapshot,
)

HIIT_ROOT = Path(
    "derivatives/confirmatory_temporal_coupling/sensitivity/hiit"
)


@unittest.skipUnless(
    (DEFAULT_SNAPSHOT_DIR / "manifest.json").is_file(),
    "HIIT regression snapshot not frozen yet",
)
class TestHiitRegressionBaseline(unittest.TestCase):
    def test_snapshot_manifest_present(self) -> None:
        manifest = DEFAULT_SNAPSHOT_DIR / "manifest.json"
        self.assertTrue(manifest.is_file())
        text = manifest.read_text(encoding="utf-8")
        self.assertIn("configuration_hash", text)
        self.assertIn("code_commit", text)
        self.assertIn("specification_version", text)

    @unittest.skipUnless(HIIT_ROOT.is_dir(), "live HIIT derivatives absent")
    def test_live_outputs_match_frozen_snapshot(self) -> None:
        failures = compare_hiit_root_to_snapshot(HIIT_ROOT)
        self.assertEqual(
            failures,
            [],
            "HIIT regression drift detected. Do not overwrite the snapshot; "
            "explain every difference first.\n" + "\n".join(failures[:40]),
        )

    def test_eligibility_standard_zlpi_fields_in_snapshot(self) -> None:
        path = (
            DEFAULT_SNAPSHOT_DIR
            / "data"
            / "C0"
            / "eligibility_by_duration.csv"
        )
        if not path.is_file():
            self.skipTest("eligibility snapshot absent")
        # Self-compare to exercise exact categorical columns.
        self.assertEqual(compare_csv_to_snapshot(path, path), [])


if __name__ == "__main__":
    unittest.main()
