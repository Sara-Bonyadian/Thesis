from __future__ import annotations

import math
import unittest
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.group_tables import (
    PAIRING_QC_FILENAME,
    PAIRED_CONTRASTS_FILENAME,
    SUBJECT_LEVEL_FILENAME,
    build_group_tables,
    build_paired_contrasts,
    build_subject_level_metrics,
    normalize_keys,
    write_group_table_outputs,
)


def _endpoint_row(
    *,
    dataset_id: str,
    observation_id: str,
    subject_id: str,
    condition: str,
    endpoint_index: float,
    local_prominence: float = 0.1,
    duration_s: int = 240,
    endpoint_name: str = "zlpi",
    band: str = "theta",
    power_representation: str = "absolute_log10",
    eligible: bool = True,
    session_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "task": condition,
        "condition": condition,
        "observation_id": observation_id,
        "duration_s": duration_s,
        "duration_role": "primary" if duration_s == 240 else "sensitivity",
        "endpoint_name": endpoint_name,
        "endpoint_alias": endpoint_name,
        "is_standard_zlpi": endpoint_name == "zlpi" and duration_s in {240, 180},
        "pool_with_standard_zlpi": endpoint_name == "zlpi" and duration_s in {240, 180},
        "band": band,
        "power_representation": power_representation,
        "is_primary_representation": power_representation == "absolute_log10",
        "pair": f"hr_x_{band}_{power_representation}",
        "eligible": eligible,
        "exclusion_reason": "" if eligible else "test_reject",
        "r0": 0.2,
        "z0": 0.203,
        "endpoint_index": endpoint_index,
        "local_prominence": local_prominence,
        "n_common_support": 120,
    }
    if session_id is not None:
        row["session_id"] = session_id
    return row


def _peak_row(
    *,
    dataset_id: str,
    observation_id: str,
    subject_id: str,
    condition: str,
    peak_height_A: float,
    peak_center_mu_s: float,
    fwhm_s: float = 23.55,
    has_identifiable_peak: bool = True,
    duration_s: int = 240,
    endpoint_name: str = "zlpi",
    band: str = "theta",
    power_representation: str = "absolute_log10",
    session_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "task": condition,
        "condition": condition,
        "observation_id": observation_id,
        "duration_s": duration_s,
        "duration_role": "primary" if duration_s == 240 else "sensitivity",
        "endpoint_name": endpoint_name,
        "is_standard_zlpi": endpoint_name == "zlpi" and duration_s in {240, 180},
        "band": band,
        "power_representation": power_representation,
        "is_primary_representation": power_representation == "absolute_log10",
        "pair": f"hr_x_{band}_{power_representation}",
        "converged": True,
        "has_identifiable_peak": has_identifiable_peak,
        "report_timing_shift": has_identifiable_peak,
        "baseline_C": 0.0,
        "peak_height_A": peak_height_A,
        "peak_center_mu_s": peak_center_mu_s if has_identifiable_peak else float("nan"),
        "sigma_s": fwhm_s / 2.355,
        "fwhm_s": fwhm_s,
        "exclusion_reason": "" if has_identifiable_peak else "no_identifiable_peak",
    }
    if session_id is not None:
        row["session_id"] = session_id
    return row


def _ds003838_pair(
    participant: str,
    *,
    rest_index: float,
    memory_index: float,
    rest_A: float = 0.4,
    memory_A: float = 0.2,
    rest_mu: float = 0.0,
    memory_mu: float = 1.0,
    rest_identifiable: bool = True,
    memory_identifiable: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rest_obs = f"ds003838-{participant}-ses-single-task-rest"
    mem_obs = f"ds003838-{participant}-ses-single-task-memory"
    endpoints = [
        _endpoint_row(
            dataset_id="ds003838",
            observation_id=rest_obs,
            subject_id=participant,
            condition="rest",
            endpoint_index=rest_index,
            session_id="single",
        ),
        _endpoint_row(
            dataset_id="ds003838",
            observation_id=mem_obs,
            subject_id=participant,
            condition="memory",
            endpoint_index=memory_index,
            local_prominence=0.05,
            session_id="single",
        ),
    ]
    peaks = [
        _peak_row(
            dataset_id="ds003838",
            observation_id=rest_obs,
            subject_id=participant,
            condition="rest",
            peak_height_A=rest_A,
            peak_center_mu_s=rest_mu,
            has_identifiable_peak=rest_identifiable,
            session_id="single",
        ),
        _peak_row(
            dataset_id="ds003838",
            observation_id=mem_obs,
            subject_id=participant,
            condition="memory",
            peak_height_A=memory_A,
            peak_center_mu_s=memory_mu,
            fwhm_s=20.0,
            has_identifiable_peak=memory_identifiable,
            session_id="single",
        ),
    ]
    return endpoints, peaks


class TestNormalizeKeys(unittest.TestCase):
    def test_normalizes_ds003690_participant_session_run(self) -> None:
        keys = normalize_keys(
            {
                "dataset_id": "ds003690",
                "subject_id": "ab4_ses-single-run-1",
                "observation_id": "ds003690-ab4-ses-single-task-gonogo-run-1",
                "condition": "gonogo",
                "session_id": "single",
            }
        )
        self.assertEqual(keys["participant_id"], "ab4")
        self.assertEqual(keys["session_id"], "single")
        self.assertEqual(keys["run_id"], "1")


class TestExactPairing(unittest.TestCase):
    def test_exact_within_subject_pairing_and_deltas(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-001",
            rest_index=0.50,
            memory_index=0.20,
            rest_A=0.40,
            memory_A=0.10,
            rest_mu=0.0,
            memory_mu=2.0,
        )
        result = build_group_tables(endpoints, peaks)
        self.assertEqual(len(result.subject_level_rows), 2)
        self.assertEqual(len(result.paired_contrast_rows), 1)
        row = result.paired_contrast_rows[0]
        self.assertEqual(row["contrast_id"], "rest__memory")
        self.assertEqual(row["participant_id"], "sub-001")
        self.assertAlmostEqual(float(row["delta_endpoint_index"]), -0.30, places=12)
        self.assertAlmostEqual(float(row["delta_peak_height_A"]), -0.30, places=12)
        self.assertAlmostEqual(float(row["delta_peak_center_mu_s"]), 2.0, places=12)
        self.assertTrue(row["mu_contrast_eligible"])
        self.assertAlmostEqual(float(row["delta_fwhm_s"]), 20.0 - 23.55, places=12)

        paired_qc = [
            q for q in result.pairing_qc_rows if q["pairing_status"] == "paired"
        ]
        self.assertEqual(len(paired_qc), 1)


class TestMissingConditions(unittest.TestCase):
    def test_missing_effort_produces_no_contrast(self) -> None:
        endpoints = [
            _endpoint_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-001-ses-single-task-rest",
                subject_id="sub-001",
                condition="rest",
                endpoint_index=0.4,
                session_id="single",
            )
        ]
        peaks = [
            _peak_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-001-ses-single-task-rest",
                subject_id="sub-001",
                condition="rest",
                peak_height_A=0.3,
                peak_center_mu_s=0.0,
                session_id="single",
            )
        ]
        result = build_group_tables(endpoints, peaks)
        self.assertEqual(result.paired_contrast_rows, ())
        statuses = {q["pairing_status"] for q in result.pairing_qc_rows}
        self.assertIn("missing_cognitive_effort", statuses)


class TestDuplicateRuns(unittest.TestCase):
    def test_duplicate_runs_aggregate_and_pair_once(self) -> None:
        endpoints = [
            _endpoint_row(
                dataset_id="ds003690",
                observation_id="ds003690-ab4-ses-single-task-passive-run-1",
                subject_id="ab4_ses-single-run-1",
                condition="passive",
                endpoint_index=0.40,
                session_id="single",
            ),
            _endpoint_row(
                dataset_id="ds003690",
                observation_id="ds003690-ab4-ses-single-task-passive-run-2",
                subject_id="ab4_ses-single-run-2",
                condition="passive",
                endpoint_index=0.60,
                session_id="single",
            ),
            _endpoint_row(
                dataset_id="ds003690",
                observation_id="ds003690-ab4-ses-single-task-gonogo-run-3",
                subject_id="ab4_ses-single-run-3",
                condition="gonogo",
                endpoint_index=0.20,
                session_id="single",
            ),
        ]
        peaks = [
            _peak_row(
                dataset_id="ds003690",
                observation_id=r["observation_id"],
                subject_id=r["subject_id"],
                condition=r["condition"],
                peak_height_A=0.5 if r["condition"] == "passive" else 0.1,
                peak_center_mu_s=0.0,
                session_id="single",
            )
            for r in endpoints
        ]
        subject_rows = build_subject_level_metrics(endpoints, peaks)
        passive = [
            r for r in subject_rows if r["condition"] == "passive"
        ]
        self.assertEqual(len(passive), 1)
        self.assertEqual(passive[0]["n_runs"], 2)
        self.assertAlmostEqual(float(passive[0]["endpoint_index"]), 0.50, places=12)

        contrasts, qc = build_paired_contrasts(subject_rows)
        gonogo = [c for c in contrasts if c["contrast_id"] == "passive__gonogo"]
        self.assertEqual(len(gonogo), 1)
        self.assertAlmostEqual(
            float(gonogo[0]["delta_endpoint_index"]), -0.30, places=12
        )
        self.assertEqual(gonogo[0]["n_low_runs"], 2)
        paired = [q for q in qc if q["pairing_status"] == "paired" and q["contrast_id"] == "passive__gonogo"]
        self.assertEqual(len(paired), 1)
        self.assertTrue(paired[0]["duplicate_runs"])


class TestSessionMismatches(unittest.TestCase):
    def test_never_pairs_across_sessions(self) -> None:
        endpoints = [
            _endpoint_row(
                dataset_id="mindfulness",
                observation_id="mindfulness-mbd-01-part1-task-step1",
                subject_id="mbd-01_part1_step1",
                condition="step1",
                endpoint_index=0.5,
                session_id="part1",
            ),
            _endpoint_row(
                dataset_id="mindfulness",
                observation_id="mindfulness-mbd-01-part2-task-step2",
                subject_id="mbd-01_part2_step2",
                condition="step2",
                endpoint_index=0.2,
                session_id="part2",
            ),
        ]
        peaks = [
            _peak_row(
                dataset_id="mindfulness",
                observation_id=r["observation_id"],
                subject_id=r["subject_id"],
                condition=r["condition"],
                peak_height_A=0.3,
                peak_center_mu_s=0.0,
                session_id=r["session_id"],
            )
            for r in endpoints
        ]
        result = build_group_tables(endpoints, peaks)
        step_contrast = [
            c for c in result.paired_contrast_rows if c["contrast_id"] == "step1__step2"
        ]
        self.assertEqual(step_contrast, [])
        statuses = {
            (q["contrast_id"], q["pairing_status"]) for q in result.pairing_qc_rows
        }
        self.assertIn(("step1__step2", "missing_cognitive_effort"), statuses)
        self.assertIn(("step1__step2", "missing_low_demand"), statuses)


class TestHiitPrePostPairing(unittest.TestCase):
    def test_hiit_pairs_within_modality_and_timepoint_only(self) -> None:
        endpoints = [
            _endpoint_row(
                dataset_id="hiit",
                observation_id="hiit-01-ph-ph-pre-rest",
                subject_id="01_ph",
                condition="ph_pre_rest",
                endpoint_index=0.40,
                session_id="ph",
            ),
            _endpoint_row(
                dataset_id="hiit",
                observation_id="hiit-01-ph-ph-pre-tetris",
                subject_id="01_ph",
                condition="ph_pre_tetris",
                endpoint_index=0.10,
                session_id="ph",
            ),
            _endpoint_row(
                dataset_id="hiit",
                observation_id="hiit-01-ph-ph-post-rest",
                subject_id="01_ph",
                condition="ph_post_rest",
                endpoint_index=0.35,
                session_id="ph",
            ),
            _endpoint_row(
                dataset_id="hiit",
                observation_id="hiit-01-ps-ps-pre-rest",
                subject_id="01_ps",
                condition="ps_pre_rest",
                endpoint_index=0.30,
                session_id="ps",
            ),
        ]
        peaks = [
            _peak_row(
                dataset_id="hiit",
                observation_id=r["observation_id"],
                subject_id=r["subject_id"],
                condition=r["condition"],
                peak_height_A=0.2,
                peak_center_mu_s=0.0,
                session_id=r["session_id"],
            )
            for r in endpoints
        ]
        result = build_group_tables(endpoints, peaks)
        by_contrast = {
            c["contrast_id"]: c for c in result.paired_contrast_rows
        }
        self.assertIn("ph_pre_rest__tetris", by_contrast)
        self.assertNotIn("ph_post_rest__tetris", by_contrast)
        self.assertNotIn("ps_pre_rest__tetris", by_contrast)
        pre = by_contrast["ph_pre_rest__tetris"]
        self.assertAlmostEqual(float(pre["delta_endpoint_index"]), -0.30, places=12)
        # Never pair pre-rest with post-rest as a state contrast.
        self.assertEqual(pre["low_demand_condition"], "ph_pre_rest")
        self.assertEqual(pre["cognitive_effort_condition"], "ph_pre_tetris")


class TestDs003690Contrasts(unittest.TestCase):
    def test_ds003690_emits_both_prespecified_contrasts(self) -> None:
        endpoints = []
        peaks = []
        for condition, index, A in (
            ("passive", 0.50, 0.40),
            ("simplert", 0.30, 0.20),
            ("gonogo", 0.10, 0.05),
        ):
            obs = f"ds003690-ab4-ses-single-task-{condition}-run-1"
            endpoints.append(
                _endpoint_row(
                    dataset_id="ds003690",
                    observation_id=obs,
                    subject_id=f"ab4_ses-single-run-1",
                    condition=condition,
                    endpoint_index=index,
                    session_id="single",
                )
            )
            peaks.append(
                _peak_row(
                    dataset_id="ds003690",
                    observation_id=obs,
                    subject_id=f"ab4_ses-single-run-1",
                    condition=condition,
                    peak_height_A=A,
                    peak_center_mu_s=1.0 if condition != "passive" else 0.0,
                    session_id="single",
                )
            )
        result = build_group_tables(endpoints, peaks)
        ids = {c["contrast_id"] for c in result.paired_contrast_rows}
        self.assertEqual(ids, {"passive__simplert", "passive__gonogo"})
        by_id = {c["contrast_id"]: c for c in result.paired_contrast_rows}
        self.assertAlmostEqual(
            float(by_id["passive__simplert"]["delta_endpoint_index"]), -0.20, places=12
        )
        self.assertAlmostEqual(
            float(by_id["passive__gonogo"]["delta_endpoint_index"]), -0.40, places=12
        )


class TestNoCrossSubjectPairing(unittest.TestCase):
    def test_never_pairs_across_participants(self) -> None:
        endpoints_a, peaks_a = _ds003838_pair(
            "sub-001", rest_index=0.5, memory_index=0.2
        )
        # Only rest for sub-002 and only memory for sub-003.
        endpoints = [
            endpoints_a[0],
            _endpoint_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-002-ses-single-task-rest",
                subject_id="sub-002",
                condition="rest",
                endpoint_index=0.4,
                session_id="single",
            ),
            _endpoint_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-003-ses-single-task-memory",
                subject_id="sub-003",
                condition="memory",
                endpoint_index=0.1,
                session_id="single",
            ),
        ]
        peaks = [
            peaks_a[0],
            _peak_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-002-ses-single-task-rest",
                subject_id="sub-002",
                condition="rest",
                peak_height_A=0.3,
                peak_center_mu_s=0.0,
                session_id="single",
            ),
            _peak_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-003-ses-single-task-memory",
                subject_id="sub-003",
                condition="memory",
                peak_height_A=0.1,
                peak_center_mu_s=1.0,
                session_id="single",
            ),
        ]
        # Incomplete pair for sub-001 (memory missing) plus mismatched subjects.
        result = build_group_tables(endpoints, peaks)
        self.assertEqual(result.paired_contrast_rows, ())
        participants = {
            q["participant_id"]
            for q in result.pairing_qc_rows
            if q["pairing_status"] in {"missing_cognitive_effort", "missing_low_demand"}
        }
        self.assertEqual(participants, {"sub-001", "sub-002", "sub-003"})

    def test_complete_pair_only_same_participant(self) -> None:
        e1, p1 = _ds003838_pair("sub-001", rest_index=0.5, memory_index=0.2)
        e2, p2 = _ds003838_pair("sub-002", rest_index=0.6, memory_index=0.1)
        result = build_group_tables(e1 + e2, p1 + p2)
        self.assertEqual(len(result.paired_contrast_rows), 2)
        participants = {c["participant_id"] for c in result.paired_contrast_rows}
        self.assertEqual(participants, {"sub-001", "sub-002"})
        # No hybrid participant ids.
        for row in result.paired_contrast_rows:
            self.assertIn(row["participant_id"], {"sub-001", "sub-002"})


class TestMuAndEndpointSeparation(unittest.TestCase):
    def test_mu_contrast_requires_identifiable_peaks_on_both_sides(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-001",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=True,
            memory_identifiable=False,
        )
        result = build_group_tables(endpoints, peaks)
        row = result.paired_contrast_rows[0]
        self.assertFalse(row["mu_contrast_eligible"])
        self.assertTrue(math.isnan(float(row["delta_peak_center_mu_s"])))
        # Other contrasts still computed.
        self.assertAlmostEqual(float(row["delta_endpoint_index"]), -0.30, places=12)
        self.assertAlmostEqual(float(row["delta_peak_height_A"]), -0.20, places=12)

    def test_never_pools_zlpi_mwpi_swpi(self) -> None:
        endpoints = [
            _endpoint_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-001-ses-single-task-rest",
                subject_id="sub-001",
                condition="rest",
                endpoint_index=0.5,
                duration_s=240,
                endpoint_name="zlpi",
                session_id="single",
            ),
            _endpoint_row(
                dataset_id="ds003838",
                observation_id="ds003838-sub-001-ses-single-task-memory",
                subject_id="sub-001",
                condition="memory",
                endpoint_index=0.2,
                duration_s=120,
                endpoint_name="mid_window_proximal_index",
                session_id="single",
            ),
        ]
        peaks = [
            _peak_row(
                dataset_id="ds003838",
                observation_id=r["observation_id"],
                subject_id=r["subject_id"],
                condition=r["condition"],
                peak_height_A=0.3,
                peak_center_mu_s=0.0,
                duration_s=int(r["duration_s"]),
                endpoint_name=str(r["endpoint_name"]),
                session_id="single",
            )
            for r in endpoints
        ]
        result = build_group_tables(endpoints, peaks)
        self.assertEqual(result.paired_contrast_rows, ())


class TestWriteOutputs(unittest.TestCase):
    def test_writes_three_canonical_csv_files(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-001", rest_index=0.5, memory_index=0.2
        )
        result = build_group_tables(endpoints, peaks)
        with TemporaryDirectory() as tmp:
            paths = write_group_table_outputs(result, tmp)
            self.assertTrue(paths["subject_level_metrics"].is_file())
            self.assertTrue(paths["paired_contrasts"].is_file())
            self.assertTrue(paths["pairing_qc"].is_file())
            self.assertEqual(
                paths["subject_level_metrics"].name, SUBJECT_LEVEL_FILENAME
            )
            self.assertEqual(
                paths["paired_contrasts"].name, PAIRED_CONTRASTS_FILENAME
            )
            self.assertEqual(paths["pairing_qc"].name, PAIRING_QC_FILENAME)


if __name__ == "__main__":
    unittest.main()
