from __future__ import annotations

import math
import unittest
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.group_tables import (
    PAIRING_QC_FILENAME,
    PAIRED_CONTRASTS_FILENAME,
    SUBJECT_LEVEL_FIELDS,
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
        "z0": math.atanh(0.2),
        "negative_flank_mean_z": 0.01,
        "positive_flank_mean_z": 0.02,
        "combined_flank_mean_z": 0.015,
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

    def test_hiit_session_qualified_subject_stays_biological_participant(self) -> None:
        keys = normalize_keys(
            {
                "dataset_id": "hiit",
                "subject_id": "01_ph",
                "observation_id": "hiit-01-ph-post-rest",
                "condition": "ph_post_rest",
                "participant_id": "",
                "session_id": "",
            }
        )
        self.assertEqual(keys["participant_id"], "01")
        self.assertEqual(keys["session_id"], "ph")
        self.assertEqual(keys["condition"], "ph_post_rest")

    def test_mindfulness_recovers_biological_participant_and_part_session(self) -> None:
        keys = normalize_keys(
            {
                "dataset_id": "mindfulness",
                "subject_id": "mbd-01_part1_step1",
                "observation_id": "mindfulness-mbd-01-part1-task-step1",
                "condition": "step1",
                "participant_id": "",
                "session_id": "",
            }
        )
        self.assertEqual(keys["participant_id"], "mbd-01")
        self.assertEqual(keys["session_id"], "part1")
        self.assertEqual(keys["condition"], "step1")

    def test_mindfulness_ignores_nan_session_and_parses_observation_id(self) -> None:
        keys = normalize_keys(
            {
                "dataset_id": "mindfulness",
                "subject_id": "mbd-01",
                "observation_id": "mindfulness-mbd-01-part2-task-step3",
                "condition": "step3",
                "participant_id": float("nan"),
                "session_id": float("nan"),
            }
        )
        self.assertEqual(keys["participant_id"], "mbd-01")
        self.assertEqual(keys["session_id"], "part2")
        self.assertEqual(keys["condition"], "step3")

    def test_mindfulness_does_not_infer_session_from_condition(self) -> None:
        keys = normalize_keys(
            {
                "dataset_id": "mindfulness",
                "subject_id": "mbd-01",
                "observation_id": "mindfulness-mbd-01-part1-task-step2",
                "condition": "step2",
                "session_id": "step2",
            }
        )
        self.assertEqual(keys["session_id"], "part1")
        self.assertEqual(keys["condition"], "step2")


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
        self.assertTrue(row["contrast_eligible"])
        self.assertEqual(row["contrast_exclusion_reason"], "")
        self.assertTrue(row["mu_contrast_eligible"])
        self.assertAlmostEqual(float(row["delta_fwhm_s"]), 20.0 - 23.55, places=12)

        paired_qc = [
            q for q in result.pairing_qc_rows if q["pairing_status"] == "paired"
        ]
        self.assertEqual(len(paired_qc), 1)
        # Dataset-scoped QC: no protocol-wide placeholder rows for empty datasets.
        datasets = {q["dataset_id"] for q in result.pairing_qc_rows}
        self.assertEqual(datasets, {"ds003838"})


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

    def test_multi_run_r0_is_tanh_of_mean_z0_not_mean_r(self) -> None:
        z_a = 0.10
        z_b = 0.40
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
        ]
        endpoints[0]["z0"] = z_a
        endpoints[0]["r0"] = math.tanh(z_a)
        endpoints[1]["z0"] = z_b
        endpoints[1]["r0"] = math.tanh(z_b)
        endpoints[0]["negative_flank_mean_z"] = -0.05
        endpoints[1]["negative_flank_mean_z"] = 0.07
        peaks = [
            _peak_row(
                dataset_id="ds003690",
                observation_id=r["observation_id"],
                subject_id=r["subject_id"],
                condition=r["condition"],
                peak_height_A=0.5,
                peak_center_mu_s=0.0,
                session_id="single",
            )
            for r in endpoints
        ]
        row = build_subject_level_metrics(endpoints, peaks)[0]
        mean_z = 0.5 * (z_a + z_b)
        self.assertAlmostEqual(float(row["z0"]), mean_z, places=12)
        self.assertAlmostEqual(float(row["r0"]), math.tanh(mean_z), places=12)
        self.assertNotAlmostEqual(
            float(row["r0"]),
            0.5 * (math.tanh(z_a) + math.tanh(z_b)),
            places=8,
        )
        self.assertAlmostEqual(
            float(row["negative_flank_mean_z"]), 0.5 * (-0.05 + 0.07), places=12
        )
        self.assertEqual(row["participant_id"], "ab4")
        self.assertEqual(row["session_id"], "single")


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
            rest_mu=-1.5,
            memory_mu=2.0,
        )
        result = build_group_tables(endpoints, peaks)
        row = result.paired_contrast_rows[0]
        self.assertTrue(row["contrast_eligible"])
        self.assertEqual(row["contrast_exclusion_reason"], "")
        self.assertFalse(row["mu_contrast_eligible"])
        self.assertTrue(math.isnan(float(row["delta_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["delta_fwhm_s"])))
        # State-specific display: Rest remains available when only Rest is identifiable.
        self.assertTrue(row["low_has_identifiable_peak"])
        self.assertFalse(row["effort_has_identifiable_peak"])
        self.assertAlmostEqual(float(row["low_peak_center_mu_s"]), -1.5, places=12)
        self.assertTrue(math.isfinite(float(row["low_fwhm_s"])))
        self.assertTrue(math.isnan(float(row["effort_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["effort_fwhm_s"])))
        # Other contrasts still computed.
        self.assertAlmostEqual(float(row["delta_endpoint_index"]), -0.30, places=12)
        self.assertTrue(math.isnan(float(row["delta_peak_height_A"])))

    def test_task_only_identifiable_keeps_task_state_fields(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-002",
            rest_index=0.5,
            memory_index=0.2,
            rest_identifiable=False,
            memory_identifiable=True,
            rest_mu=-1.5,
            memory_mu=2.25,
        )
        result = build_group_tables(endpoints, peaks)
        row = result.paired_contrast_rows[0]
        self.assertFalse(row["mu_contrast_eligible"])
        self.assertTrue(math.isnan(float(row["delta_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["delta_fwhm_s"])))
        self.assertFalse(row["low_has_identifiable_peak"])
        self.assertTrue(row["effort_has_identifiable_peak"])
        self.assertTrue(math.isnan(float(row["low_peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(row["low_fwhm_s"])))
        self.assertAlmostEqual(float(row["effort_peak_center_mu_s"]), 2.25, places=12)
        self.assertTrue(math.isfinite(float(row["effort_fwhm_s"])))

    def test_nonidentifiable_peak_leaves_shape_fields_blank(self) -> None:
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
                peak_height_A=0.01,
                peak_center_mu_s=0.0,
                fwhm_s=23.55,
                has_identifiable_peak=False,
                session_id="single",
            )
        ]
        subject = build_subject_level_metrics(endpoints, peaks)[0]
        self.assertFalse(subject["has_identifiable_peak"])
        self.assertTrue(math.isnan(float(subject["peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(subject["sigma_s"])))
        self.assertTrue(math.isnan(float(subject["fwhm_s"])))
        self.assertTrue(math.isnan(float(subject["peak_height_A"])))

    def test_contrast_exclusion_when_endpoint_ineligible(self) -> None:
        endpoints, peaks = _ds003838_pair(
            "sub-001",
            rest_index=0.5,
            memory_index=0.2,
        )
        for row in endpoints:
            if row["condition"] == "memory":
                row["eligible"] = False
                row["exclusion_reason"] = "test_reject"
        result = build_group_tables(endpoints, peaks)
        row = result.paired_contrast_rows[0]
        self.assertFalse(row["contrast_eligible"])
        self.assertEqual(row["contrast_exclusion_reason"], "effort_endpoint_ineligible")

    def test_pairing_qc_restricted_to_dataset_ids(self) -> None:
        e1, p1 = _ds003838_pair("sub-001", rest_index=0.5, memory_index=0.2)
        e2 = [
            _endpoint_row(
                dataset_id="hiit",
                observation_id="hiit-01-ph-pre-rest",
                subject_id="01_ph",
                condition="ph_pre_rest",
                endpoint_index=0.4,
                session_id="ph",
            )
        ]
        p2 = [
            _peak_row(
                dataset_id="hiit",
                observation_id="hiit-01-ph-pre-rest",
                subject_id="01_ph",
                condition="ph_pre_rest",
                peak_height_A=0.3,
                peak_center_mu_s=0.0,
                session_id="ph",
            )
        ]
        result = build_group_tables(e1 + e2, p1 + p2, dataset_ids=("ds003838",))
        datasets = {q["dataset_id"] for q in result.pairing_qc_rows}
        self.assertEqual(datasets, {"ds003838"})
        subject_datasets = {r["dataset_id"] for r in result.subject_level_rows}
        self.assertEqual(subject_datasets, {"ds003838"})

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
        d120 = [
            r
            for r in result.subject_level_rows
            if int(r["duration_s"]) == 120
        ][0]
        self.assertTrue(math.isnan(float(d120["peak_height_A"])))
        self.assertTrue(math.isnan(float(d120["peak_center_mu_s"])))
        self.assertTrue(math.isnan(float(d120["fwhm_s"])))
        self.assertFalse(d120["has_identifiable_peak"])
        self.assertEqual(
            d120["peak_exclusion_reason"], "option_c_peak_requires_d180_d240"
        )
        d240 = [
            r
            for r in result.subject_level_rows
            if int(r["duration_s"]) == 240
        ][0]
        self.assertAlmostEqual(float(d240["peak_height_A"]), 0.3, places=12)
        self.assertIn("negative_flank_mean_z", d240)
        self.assertAlmostEqual(float(d240["combined_flank_mean_z"]), 0.015, places=12)


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
            header = paths["subject_level_metrics"].read_text(encoding="utf-8").splitlines()[0]
            for field in (
                "negative_flank_mean_z",
                "positive_flank_mean_z",
                "combined_flank_mean_z",
                "participant_id",
                "session_id",
            ):
                self.assertIn(field, header.split(","))
            self.assertTrue(
                SUBJECT_LEVEL_FIELDS.index("negative_flank_mean_z")
                > SUBJECT_LEVEL_FIELDS.index("z0")
            )


def _mindfulness_step(
    participant: str,
    session: str,
    step: str,
    *,
    endpoint_index: float,
    observation_id: str | None = None,
    subject_id: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    oid = observation_id or f"mindfulness-{participant}-{session}-task-{step}"
    sid = subject_id or participant
    return (
        [
            _endpoint_row(
                dataset_id="mindfulness",
                observation_id=oid,
                subject_id=sid,
                condition=step,
                endpoint_index=endpoint_index,
            )
        ],
        [
            _peak_row(
                dataset_id="mindfulness",
                observation_id=oid,
                subject_id=sid,
                condition=step,
                peak_height_A=0.2,
                peak_center_mu_s=0.0,
            )
        ],
    )


class TestMindfulnessSessionPairing(unittest.TestCase):
    def test_pairs_within_participant_session_not_across_parts(self) -> None:
        endpoints: list[dict[str, object]] = []
        peaks: list[dict[str, object]] = []
        for session, step1, step2 in (("part1", 0.50, 0.20), ("part2", 0.40, 0.10)):
            e1, p1 = _mindfulness_step("mbd-01", session, "step1", endpoint_index=step1)
            e2, p2 = _mindfulness_step("mbd-01", session, "step2", endpoint_index=step2)
            endpoints.extend(e1 + e2)
            peaks.extend(p1 + p2)
        result = build_group_tables(endpoints, peaks)
        pairs = [
            r
            for r in result.paired_contrast_rows
            if r["contrast_id"] == "step1__step2"
            and r["band"] == "theta"
            and int(r["duration_s"]) == 240
        ]
        self.assertEqual(len(pairs), 2)
        sessions = {r["session_id"] for r in pairs}
        self.assertEqual(sessions, {"part1", "part2"})
        self.assertTrue(all(r["participant_id"] == "mbd-01" for r in pairs))
        qc = [
            q
            for q in result.pairing_qc_rows
            if q["contrast_id"] == "step1__step2" and q["pairing_status"] == "paired"
        ]
        self.assertEqual({q["session_id"] for q in qc}, {"part1", "part2"})
        self.assertEqual(int(qc[0]["n_paired_keys"]), 2)

    def test_aggregates_duplicate_runs_within_same_part_and_drops_legacy(self) -> None:
        endpoints: list[dict[str, object]] = []
        peaks: list[dict[str, object]] = []
        for oid, index in (
            ("mindfulness-mbd-15-part1-task-step1", 0.10),
            ("mindfulness-mbd-15-part1-task-step1-run-01", 0.30),
            ("mindfulness-mbd-15-part1-task-step1-run-02", 0.50),
        ):
            e, p = _mindfulness_step(
                "mbd-15",
                "part1",
                "step1",
                endpoint_index=index,
                observation_id=oid,
                subject_id="mbd-15_part1_step1" if "run" not in oid else "mbd-15",
            )
            endpoints.extend(e)
            peaks.extend(p)
        e2, p2 = _mindfulness_step("mbd-15", "part1", "step2", endpoint_index=0.20)
        endpoints.extend(e2)
        peaks.extend(p2)
        result = build_group_tables(endpoints, peaks)
        step1 = [
            r
            for r in result.subject_level_rows
            if r["condition"] == "step1"
            and r["band"] == "theta"
            and int(r["duration_s"]) == 240
        ]
        self.assertEqual(len(step1), 1)
        self.assertEqual(int(step1[0]["n_runs"]), 2)
        self.assertEqual(step1[0]["participant_id"], "mbd-15")
        self.assertEqual(step1[0]["session_id"], "part1")
        obs_ids = set(str(step1[0]["observation_ids"]).split(";"))
        self.assertEqual(
            obs_ids,
            {
                "mindfulness-mbd-15-part1-task-step1-run-01",
                "mindfulness-mbd-15-part1-task-step1-run-02",
            },
        )
        self.assertAlmostEqual(float(step1[0]["endpoint_index"]), 0.40, places=12)


if __name__ == "__main__":
    unittest.main()
