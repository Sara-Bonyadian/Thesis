from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.datasets import CanonicalObservation
from ppg_eeg.confirmatory.protocol_audit import (
    EXCLUSION_CODES,
    PROTOCOL_SPECS,
    EligibilityMetadata,
    build_paired_subject_sets,
    enrich_eligibility_metadata_from_csv,
    evaluate_all_durations,
    evaluate_duration_eligibility,
    write_duration_eligibility,
    write_protocol_audit,
)


def _observation(
    dataset_id: str,
    participant: str,
    condition: str,
    *,
    session: str = "single",
    run: str | None = None,
) -> CanonicalObservation:
    if dataset_id == "hiit":
        observation_id = f"hiit-{participant}-{session}-{condition.replace('_', '-')}"
        subject_id = f"{participant}_{session}"
    elif dataset_id == "mindfulness":
        observation_id = (
            f"mindfulness-{participant}-{session}-task-{condition}"
        )
        subject_id = f"{participant}_{session}_{condition}"
    else:
        run_suffix = f"-run-{run}" if run is not None else ""
        observation_id = (
            f"{dataset_id}-{participant}-ses-{session}"
            f"-task-{condition}{run_suffix}"
        )
        subject_id = participant
        if dataset_id in {
            "ds003690",
            "ds003816",
            "ds004582",
            "ds004587",
        }:
            subject_id = f"{participant}_ses-{session}{run_suffix}"
    return CanonicalObservation(
        dataset_id=dataset_id,
        observation_id=observation_id,
        subject_id=subject_id,
        task_label=condition,
        condition_label=condition,
        eeg_path=Path(f"{observation_id}.vhdr"),
        eeg_format="brainvision",
        ppg_source="embedded_eeg",
        session_label=session,
        modality=session if dataset_id == "hiit" else "eeg_cardiac",
        state=condition,
        participant_id=participant,
        session_id=session,
        run_id=run or "single",
        condition_id=condition,
    )


class TestProtocolMetadata(unittest.TestCase):
    def test_all_confirmatory_datasets_have_protocol_specs(self) -> None:
        self.assertEqual(
            set(PROTOCOL_SPECS),
            {
                "ds003690",
                "ds003816",
                "ds003838",
                "ds004582",
                "ds004587",
                "ds006848",
                "hiit",
                "mindfulness",
            },
        )
        for dataset_id, spec in PROTOCOL_SPECS.items():
            with self.subTest(dataset_id=dataset_id):
                self.assertTrue(spec.low_demand_conditions)
                self.assertTrue(
                    set(spec.cardiac_modality.split(";")).issubset(
                        {"ECG", "PPG"}
                    )
                )
                self.assertTrue(spec.eye_state)
                self.assertTrue(spec.posture)
                self.assertTrue(spec.task_timing)
                self.assertTrue(spec.run_pairing_policy)

    def test_single_state_and_nonprespecified_datasets_have_no_contrast(self) -> None:
        self.assertEqual(PROTOCOL_SPECS["ds004582"].contrasts, ())
        self.assertEqual(PROTOCOL_SPECS["ds003816"].contrasts, ())


class TestParticipantPairing(unittest.TestCase):
    def test_pairing_is_intersection_not_union(self) -> None:
        observations = {
            "ds003838": [
                _observation("ds003838", "sub-001", "rest"),
                _observation("ds003838", "sub-001", "memory"),
                _observation("ds003838", "sub-002", "rest"),
                _observation("ds003838", "sub-003", "memory"),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrast = payload["datasets"]["ds003838"]["contrasts"]["rest__memory"]

        self.assertEqual(contrast["n_low_demand"], 2)
        self.assertEqual(contrast["n_cognitive_effort"], 2)
        self.assertEqual(contrast["n_paired"], 1)
        self.assertEqual(
            contrast["paired_keys"],
            [{"participant_id": "sub-001", "session_id": "single"}],
        )
        self.assertEqual(
            contrast["low_demand_only_keys"],
            [{"participant_id": "sub-002", "session_id": "single"}],
        )
        self.assertEqual(
            contrast["cognitive_effort_only_keys"],
            [{"participant_id": "sub-003", "session_id": "single"}],
        )

    def test_pairing_never_crosses_sessions(self) -> None:
        observations = {
            "mindfulness": [
                _observation(
                    "mindfulness", "mbd-01", "step1", session="part1"
                ),
                _observation(
                    "mindfulness", "mbd-01", "step2", session="part2"
                ),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrast = payload["datasets"]["mindfulness"]["contrasts"][
            "step1__step2"
        ]
        self.assertEqual(contrast["n_paired"], 0)
        self.assertEqual(contrast["n_low_demand"], 1)
        self.assertEqual(contrast["n_cognitive_effort"], 1)

    def test_ds003690_retains_runs_but_pairs_at_participant_session_level(self) -> None:
        observations = {
            "ds003690": [
                _observation(
                    "ds003690", "ab4", "passive", session="single", run="01"
                ),
                _observation(
                    "ds003690", "ab4", "passive", session="single", run="02"
                ),
                _observation(
                    "ds003690", "ab4", "gonogo", session="single", run="03"
                ),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrast = payload["datasets"]["ds003690"]["contrasts"][
            "passive__gonogo"
        ]

        self.assertEqual(contrast["n_paired"], 1)
        pair = contrast["observations_by_pair"][0]
        self.assertEqual(len(pair["low_demand_observation_ids"]), 2)
        self.assertEqual(len(pair["cognitive_effort_observation_ids"]), 1)

    def test_hiit_pairs_within_modality_session_and_timepoint_contrast(self) -> None:
        observations = {
            "hiit": [
                _observation(
                    "hiit", "01", "ph_pre_rest", session="ph"
                ),
                _observation(
                    "hiit", "01", "ph_pre_tetris", session="ph"
                ),
                _observation(
                    "hiit", "01", "ps_pre_rest", session="ps"
                ),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrasts = payload["datasets"]["hiit"]["contrasts"]
        self.assertEqual(contrasts["ph_pre_rest__tetris"]["n_paired"], 1)
        self.assertEqual(contrasts["ps_pre_rest__tetris"]["n_paired"], 0)


class TestProtocolAuditOutputs(unittest.TestCase):
    def test_writes_required_outputs_without_duration_eligibility(self) -> None:
        observations = {
            "ds003838": [
                _observation("ds003838", "sub-001", "rest"),
                _observation("ds003838", "sub-001", "memory"),
            ]
        }
        with TemporaryDirectory() as tmp:
            csv_path, json_path = write_protocol_audit(
                observations,
                tmp,
                dataset_roles={"ds003838": "primary"},
            )
            self.assertEqual(csv_path.name, "protocol_audit.csv")
            self.assertEqual(json_path.name, "paired_subject_sets.json")

            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            columns = set(rows[0])
            self.assertTrue(
                {
                    "dataset_id",
                    "condition",
                    "demand_class",
                    "subject_pairing_key",
                    "session_pairing_key",
                    "run_pairing_key",
                    "cardiac_modality",
                    "eye_state",
                    "posture",
                    "task_timing",
                    "nuisance_signals",
                    "n_paired_participants",
                }.issubset(columns)
            )
            self.assertFalse(any("duration" in column for column in columns))
            self.assertFalse(any("eligible" in column for column in columns))

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["datasets"]["ds003838"]["contrasts"]["rest__memory"][
                    "n_paired"
                ],
                1,
            )

    def test_output_is_deterministic(self) -> None:
        rows = [
            _observation("ds003838", "sub-002", "memory"),
            _observation("ds003838", "sub-001", "rest"),
            _observation("ds003838", "sub-001", "memory"),
        ]
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            first_paths = write_protocol_audit({"ds003838": rows}, first)
            second_paths = write_protocol_audit(
                {"ds003838": list(reversed(rows))}, second
            )
            self.assertEqual(
                first_paths[0].read_text(encoding="utf-8"),
                second_paths[0].read_text(encoding="utf-8"),
            )
            self.assertEqual(
                first_paths[1].read_text(encoding="utf-8"),
                second_paths[1].read_text(encoding="utf-8"),
            )


class TestDurationEligibility(unittest.TestCase):
    def _complete(self, **overrides: object) -> EligibilityMetadata:
        values: dict[str, object] = {
            "dataset_id": "ds003838",
            "observation_id": "ds003838-sub-001-task-rest",
            "participant_id": "sub-001",
            "condition": "rest",
            "source_data_supplied": True,
            "eeg_exists": True,
            "cardiac_exists": True,
            "raw_overlap_s": 240.0,
            "clean_beat_span_s": 240.0,
            "requires_paired_state": True,
            "paired_state_available": True,
            "pairing_resolved": True,
            "protocol_match": True,
        }
        values.update(overrides)
        return EligibilityMetadata(**values)  # type: ignore[arg-type]

    def test_exact_duration_boundaries_are_eligible(self) -> None:
        decisions = evaluate_all_durations([self._complete()])
        self.assertEqual(
            [(decision.duration_s, decision.status) for decision in decisions],
            [
                (240, "eligible"),
                (180, "eligible"),
                (120, "eligible"),
                (60, "eligible"),
            ],
        )

    def test_raw_overlap_boundary_and_nested_durations(self) -> None:
        decisions = evaluate_all_durations(
            [self._complete(raw_overlap_s=179.999, clean_beat_span_s=300.0)]
        )
        by_duration = {decision.duration_s: decision for decision in decisions}
        self.assertEqual(by_duration[240].status, "ineligible")
        self.assertEqual(
            by_duration[240].exclusion_code, "insufficient_raw_duration"
        )
        self.assertEqual(by_duration[180].status, "ineligible")
        self.assertEqual(by_duration[120].status, "eligible")
        self.assertEqual(by_duration[60].status, "eligible")

    def test_clean_beat_span_boundary(self) -> None:
        decision = evaluate_duration_eligibility(
            self._complete(clean_beat_span_s=239.999),
            240,
        )
        self.assertEqual(decision.status, "ineligible")
        self.assertEqual(decision.exclusion_code, "insufficient_beat_span")

    def test_absent_source_data_is_not_ineligible(self) -> None:
        decisions = evaluate_all_durations(
            [
                self._complete(
                    source_data_supplied=False,
                    eeg_exists=None,
                    cardiac_exists=None,
                    raw_overlap_s=None,
                    clean_beat_span_s=None,
                )
            ]
        )
        self.assertTrue(
            all(decision.status == "not_supplied" for decision in decisions)
        )
        self.assertTrue(
            all(
                decision.exclusion_code == "data_not_supplied"
                for decision in decisions
            )
        )

    def test_missing_metadata_is_not_computable_not_ineligible(self) -> None:
        cases = [
            ({"eeg_exists": False}, "missing_eeg"),
            ({"cardiac_exists": False}, "missing_cardiac_data"),
            (
                {"raw_overlap_s": None, "clean_beat_span_s": 240.0},
                "data_not_supplied",
            ),
            ({"pairing_resolved": None}, "unresolved_pairing"),
        ]
        for overrides, expected_code in cases:
            with self.subTest(overrides=overrides):
                decision = evaluate_duration_eligibility(
                    self._complete(**overrides), 240
                )
                self.assertEqual(decision.status, "not_computable")
                self.assertEqual(decision.exclusion_code, expected_code)

    def test_missing_clean_beat_span_is_deferred_not_blocking(self) -> None:
        decision = evaluate_duration_eligibility(
            self._complete(raw_overlap_s=240.0, clean_beat_span_s=None),
            240,
        )
        self.assertEqual(decision.status, "eligible")
        self.assertEqual(decision.exclusion_code, "")
        self.assertIsNone(decision.clean_beat_span_s)
        self.assertIn("clean_beat_span_not_computed", decision.notes)

    def test_known_protocol_and_pairing_exclusions_are_ineligible(self) -> None:
        cases = [
            ({"paired_state_available": False}, "missing_paired_state"),
            ({"pairing_resolved": False}, "unresolved_pairing"),
            ({"protocol_match": False}, "protocol_mismatch"),
        ]
        for overrides, expected_code in cases:
            with self.subTest(overrides=overrides):
                decision = evaluate_duration_eligibility(
                    self._complete(**overrides), 240
                )
                self.assertEqual(decision.status, "ineligible")
                self.assertEqual(decision.exclusion_code, expected_code)

    def test_all_required_exclusion_codes_are_exposed(self) -> None:
        self.assertEqual(
            set(EXCLUSION_CODES),
            {
                "missing_paired_state",
                "insufficient_raw_duration",
                "insufficient_beat_span",
                "missing_eeg",
                "missing_cardiac_data",
                "unresolved_pairing",
                "protocol_mismatch",
                "data_not_supplied",
            },
        )

    def test_writes_eligibility_and_qc_summary_tables(self) -> None:
        metadata = [
            self._complete(),
            self._complete(
                observation_id="ds003838-sub-002-task-rest",
                participant_id="sub-002",
                raw_overlap_s=100.0,
            ),
        ]
        with TemporaryDirectory() as tmp:
            eligibility_path, summary_path = write_duration_eligibility(
                metadata, tmp
            )
            self.assertEqual(
                eligibility_path.name, "eligibility_by_duration.csv"
            )
            self.assertEqual(
                summary_path.name, "eligibility_qc_summary.csv"
            )

            with eligibility_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                {row["duration_s"] for row in rows},
                {"240", "180", "120", "60"},
            )
            self.assertNotIn("hr", rows[0])
            self.assertNotIn("eeg_power", rows[0])

            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertTrue(summary)
            self.assertTrue(
                {
                    "dataset_id",
                    "duration_s",
                    "status",
                    "exclusion_code",
                    "n_observations",
                    "n_participants",
                }.issubset(summary[0])
            )

    def test_enriches_from_existing_raw_and_cardiac_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_path = root / "data_audit.csv"
            raw_path.write_text(
                "dataset_id,observation_id,eeg_exists,cardiac_exists,"
                "overlap_duration_s\n"
                "ds003838,ds003838-sub-001-task-rest,true,true,240\n",
                encoding="utf-8",
            )
            cardiac_path = root / "cardiac_qc.csv"
            cardiac_path.write_text(
                "dataset_id,observation_id,clean_ibi_coverage_s\n"
                "ds003838,ds003838-sub-001-task-rest,180\n",
                encoding="utf-8",
            )
            base = self._complete(
                raw_overlap_s=None,
                clean_beat_span_s=None,
                eeg_exists=None,
                cardiac_exists=None,
            )
            enriched = enrich_eligibility_metadata_from_csv(
                [base],
                raw_audit_paths=[raw_path],
                cardiac_qc_paths=[cardiac_path],
            )

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].raw_overlap_s, 240.0)
        self.assertEqual(enriched[0].clean_beat_span_s, 180.0)
        self.assertTrue(enriched[0].eeg_exists)
        self.assertTrue(enriched[0].cardiac_exists)
        self.assertEqual(
            evaluate_duration_eligibility(enriched[0], 180).status,
            "eligible",
        )
        decision_240 = evaluate_duration_eligibility(enriched[0], 240)
        self.assertEqual(decision_240.status, "ineligible")
        self.assertEqual(
            decision_240.exclusion_code, "insufficient_beat_span"
        )


if __name__ == "__main__":
    unittest.main()
