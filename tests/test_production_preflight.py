from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.temporal_coupling.confirmatory.production import (
    PRODUCTION_PREFLIGHT_FILENAME,
    PRODUCTION_RUN_PLAN_FILENAME,
    SMOKE_RUN_MANIFEST_FILENAME,
    STAGE_EXECUTION_LOG_FILENAME,
    STAGE_ORDER,
    build_synthetic_aligned_tables,
    primary_blockers_summary,
    run_production,
    validate_csv_schema,
)


class TestProductionPreflightAndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo = Path(__file__).resolve().parents[1]
        cls.config_dir = repo / "zero-lag-reanalysis-repo"
        cls.repo_root = repo
        assert (cls.config_dir / "config.confirmatory.master.yaml").is_file()

    def test_synthetic_aligned_tables_cover_durations(self) -> None:
        with TemporaryDirectory() as tmp:
            written = build_synthetic_aligned_tables(tmp, n_participants=2)
            self.assertEqual(len(written), 4)
            for path in written.values():
                self.assertTrue(path.is_file())
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreater(len(rows), 0)
                self.assertIn("hr_z", rows[0])
                self.assertIn("theta_absolute_log10_power_z", rows[0])

    def test_preflight_marks_missing_raw_as_blocker_not_exclusion(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_production(
                mode="preflight",
                config_dir=self.config_dir,
                production_root=tmp,
                repo_root=self.repo_root,
            )
            self.assertTrue(result.preflight_path.is_file())
            self.assertTrue(result.run_plan_path.is_file())
            with result.preflight_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            raw_fails = [
                row
                for row in rows
                if row["check_id"] == "raw_data_available" and row["status"] == "fail"
            ]
            self.assertGreater(len(raw_fails), 0)
            for row in raw_fails:
                self.assertEqual(row["severity_class"], "blocker")
                self.assertIn("not a participant exclusion", row["detail"])
            plan = json.loads(result.run_plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["mode"], "preflight")
            self.assertFalse(plan["execute_stages"])

    def test_primary_mode_refuses_full_cohort_when_raw_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_production(
                mode="primary",
                config_dir=self.config_dir,
                production_root=tmp,
                repo_root=self.repo_root,
            )
            plan = json.loads(result.run_plan_path.read_text(encoding="utf-8"))
            self.assertFalse(plan["execute_stages"])
            self.assertIn("refusing full real-cohort", plan["refuse_full_cohort_reason"])
            self.assertFalse(result.ready_for_primary)
            blockers = primary_blockers_summary(result)
            self.assertTrue(any("raw_data_available" in line for line in blockers))
            with result.stage_log_path.open(encoding="utf-8", newline="") as handle:
                logs = list(csv.DictReader(handle))
            statuses = {row["status"] for row in logs}
            self.assertIn("blocked", statuses)

    def test_smoke_end_to_end_writes_manifests_and_stage_log(self) -> None:
        with TemporaryDirectory() as tmp:
            result = run_production(
                mode="smoke",
                config_dir=self.config_dir,
                production_root=tmp,
                repo_root=self.repo_root,
            )
            for name in (
                PRODUCTION_PREFLIGHT_FILENAME,
                PRODUCTION_RUN_PLAN_FILENAME,
                STAGE_EXECUTION_LOG_FILENAME,
                SMOKE_RUN_MANIFEST_FILENAME,
            ):
                self.assertTrue((Path(tmp) / name).is_file(), msg=name)

            self.assertIsNotNone(result.smoke_manifest_path)
            assert result.smoke_manifest_path is not None
            manifest = json.loads(result.smoke_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "confirmatory_smoke_run_manifest_v1")
            self.assertTrue(manifest["input_availability"]["synthetic_fixture"])
            self.assertIn("code_commit", manifest)
            self.assertIn("config_hash", manifest)

            with result.stage_log_path.open(encoding="utf-8", newline="") as handle:
                logs = list(csv.DictReader(handle))
            stage_ids = [row["stage_id"] for row in logs]
            for stage_id, _ in STAGE_ORDER:
                self.assertIn(stage_id, stage_ids)
            failures = [
                row
                for row in logs
                if row["status"] in {"error", "schema_fail"}
            ]
            self.assertEqual(failures, [], msg=failures)

            # Sample schema check on a primary correlation output path from smoke work root.
            curves = list((Path(tmp) / "smoke_e2e" / "m5_curves").glob("*.csv"))
            self.assertGreater(len(curves), 0)

    def test_validate_csv_schema_reports_missing_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            ok, message = validate_csv_schema(path, ("a", "c"))
            self.assertFalse(ok)
            self.assertIn("missing columns", message)


if __name__ == "__main__":
    unittest.main()
