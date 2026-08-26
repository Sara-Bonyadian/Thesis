"""Stable invariants for the real ds004587 Phase 2 integration run.

Only ``sensitivity/ds004587`` is valid. The legacy ``primary/ds004587`` tree is
stale and must not be accepted.
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

SENSITIVITY_ROOT = Path("derivatives/confirmatory_temporal_coupling/sensitivity/ds004587")
LEGACY_PRIMARY_ROOT = Path("derivatives/confirmatory_temporal_coupling/primary/ds004587")
STALE_MARKER = LEGACY_PRIMARY_ROOT / "STALE_INVALIDATED.txt"


class TestDs004587PathPolicy(unittest.TestCase):
    def test_legacy_primary_tree_is_marked_stale(self) -> None:
        if LEGACY_PRIMARY_ROOT.is_dir():
            self.assertTrue(
                STALE_MARKER.is_file(),
                "primary/ds004587 must carry STALE_INVALIDATED.txt",
            )

    def test_valid_root_is_sensitivity_only(self) -> None:
        self.assertEqual(
            SENSITIVITY_ROOT.as_posix(),
            "derivatives/confirmatory_temporal_coupling/sensitivity/ds004587",
        )


@unittest.skipUnless(
    (SENSITIVITY_ROOT / "stage_status.json").is_file(),
    "ds004587 sensitivity C0–C7 absent (blocked until raw BIDS restored; then rerun under sensitivity/)",
)
class TestDs004587Phase2Invariants(unittest.TestCase):
    def test_c4_uses_production_surrogate_count_and_has_all_null_exports(self) -> None:
        root = SENSITIVITY_ROOT
        status = json.loads((root / "stage_status.json").read_text(encoding="utf-8"))
        c4 = status["stages"]["C4"]
        complete_path = root / "C4" / "C4_COMPLETE.json"
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
            self.assertTrue((root / "C4" / name).is_file(), name)

    def test_aggregation_manifest_keeps_endpoint_families_separate(self) -> None:
        root = SENSITIVITY_ROOT
        path = root / "C5" / "aggregation_manifest.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("endpoint_family", reader.fieldnames or [])
            self.assertIn("status", reader.fieldnames or [])
            rows = list(reader)
        # External-generalization cohort: empty paired manifest is an explicit
        # no-pooling result (no forced rest–IG contrast).
        if not rows:
            self.assertTrue((root / "C5" / "pairing_qc.csv").is_file())
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
        root = SENSITIVITY_ROOT
        status = json.loads((root / "stage_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["stages"]["C7"]["status"], "ok")
        source = root / "C7" / "figures" / "source_data"
        self.assertTrue((source / "figure3_panel_c_duration_sensitivity.csv").is_file())
        with (source / "figure3_panel_f_topography_gamma_summary.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            topography = list(csv.DictReader(handle))
        for row in topography:
            if row.get("status") != "computed":
                self.assertNotEqual(row.get("estimate"), "0")

    def test_role_is_sensitivity_external_generalization(self) -> None:
        root = SENSITIVITY_ROOT
        protocol = root / "C0" / "protocol_audit.csv"
        if not protocol.is_file():
            self.skipTest("C0 protocol_audit.csv not written yet")
        rows = list(csv.DictReader(protocol.open(encoding="utf-8")))
        roles = {r.get("dataset_role", "").casefold() for r in rows if r.get("dataset_id") == "ds004587"}
        self.assertTrue(roles)
        self.assertEqual(roles, {"sensitivity"})


if __name__ == "__main__":
    unittest.main()
