"""Stable invariants for the real ds004587 Phase 2 integration run."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path("derivatives/confirmatory_temporal_coupling/primary/ds004587")


@unittest.skipUnless(ROOT.is_dir(), "ds004587 Phase 2 derivatives absent")
class TestDs004587Phase2Invariants(unittest.TestCase):
    def test_c4_uses_production_surrogate_count_and_has_all_null_exports(self) -> None:
        status = json.loads((ROOT / "stage_status.json").read_text(encoding="utf-8"))
        c4 = status["stages"]["C4"]
        complete_path = ROOT / "C4" / "C4_COMPLETE.json"
        # Mid-rerun trees may still advertise the prior integration detail until
        # production C4 finishes and rewrites stage_status.
        if not complete_path.is_file():
            self.skipTest("production C4_COMPLETE.json not written yet")
        self.assertEqual(c4["status"], "ok")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        self.assertEqual(int(complete.get("n_surrogates", -1)), 500)
        for name in (
            "null_qc.csv",
            "null_subject_results.csv",
            "null_summary.csv",
            "null_surrogate_values.csv",
        ):
            self.assertTrue((ROOT / "C4" / name).is_file(), name)

    def test_aggregation_manifest_keeps_endpoint_families_separate(self) -> None:
        path = ROOT / "C5" / "aggregation_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("endpoint_family", reader.fieldnames or [])
            self.assertIn("status", reader.fieldnames or [])
            rows = list(reader)
        # ds004587 currently has no C5 paired rows after its run-normalized
        # pairing contract; an empty manifest is an explicit no-pooling result.
        if not rows:
            self.assertTrue((ROOT / "C5" / "pairing_qc.csv").is_file())
            return
        families = {r["endpoint_family"] for r in rows}
        self.assertIn("standard_zlpi", families)
        self.assertIn("mwpi", families)
        self.assertIn("swpi", families)
        for row in rows:
            self.assertTrue(row["specification_id"])
            self.assertTrue(row["units"])
            self.assertIn(row["status"], {"computed", "excluded"})

    def test_c7_exports_figure_source_data_and_nc_topography_slot(self) -> None:
        status = json.loads((ROOT / "stage_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["stages"]["C7"]["status"], "ok")
        source = ROOT / "C7" / "figures" / "source_data"
        self.assertTrue((source / "figure3_panel_c_duration_sensitivity.csv").is_file())
        with (source / "figure3_panel_f_topography_gamma_summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            topography = list(csv.DictReader(handle))
        # Empty common montage is an explicit NC display state, never zero maps.
        for row in topography:
            if row.get("status") != "computed":
                self.assertNotEqual(row.get("estimate"), "0")


if __name__ == "__main__":
    unittest.main()
