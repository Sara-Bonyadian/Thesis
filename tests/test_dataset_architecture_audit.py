"""Architecture audit: dataset roles, pairing, duration, display vs pooling."""

from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.confirmatory.dataset_roles import (
    DATASET_SCIENTIFIC_PROFILES,
    FAMILY_DURATION_SENSITIVITY,
    FAMILY_EXERCISE_STATE_MODERATION,
    FAMILY_EXTERNAL_GENERALIZATION,
    FAMILY_INTERNAL_ATTENTION,
    FAMILY_PAIRED_STATE_DEPENDENT,
    NC_NOT_APPLICABLE_ESTIMAND,
    PANEL_STATUS_NOT_APPLICABLE,
    analysis_family_for,
    eligible_durations_for,
    eligible_for_dataset_display,
    eligible_for_duration_analysis,
    eligible_for_external_generalization,
    eligible_for_primary_meta_pooling,
    eligible_for_sensitivity_analysis,
    has_prespecified_contrast,
    is_external_generalization_dataset,
    is_primary_dataset,
    is_sensitivity_dataset,
    panel_reason_for_empty_paired,
    primary_dataset_ids,
    scientific_profile,
    sensitivity_dataset_ids,
    session_unit_key,
)
from ppg_eeg.confirmatory.inference import (
    META_EXCLUDED_DATASETS,
    PRIMARY_META_CONTRASTS,
    PRIMARY_PAIRED_DATASETS,
    PRIMARY_STATE_CONTRASTS,
)
from ppg_eeg.confirmatory.protocol_audit import PROTOCOL_SPECS, build_paired_subject_sets
from ppg_eeg.datasets import CanonicalObservation


def _obs(
    dataset_id: str,
    participant: str,
    condition: str,
    *,
    session: str = "single",
) -> CanonicalObservation:
    return CanonicalObservation(
        dataset_id=dataset_id,
        observation_id=f"{dataset_id}-{participant}-{session}-{condition}",
        subject_id=participant,
        task_label=condition,
        condition_label=condition,
        eeg_path=Path(f"{dataset_id}-{participant}-{condition}.vhdr"),
        eeg_format="brainvision",
        ppg_source="embedded_eeg",
        session_label=session,
        modality="eeg_cardiac",
        state=condition,
        participant_id=participant,
        session_id=session,
        run_id="single",
        condition_id=condition,
    )


class TestCentralizedDatasetRegistry(unittest.TestCase):
    def test_all_locked_datasets_are_registered(self) -> None:
        expected = {
            "ds003690",
            "ds003838",
            "ds006848",
            "hiit",
            "ds004582",
            "ds004587",
            "mindfulness",
            "ds003816",
        }
        self.assertEqual(set(DATASET_SCIENTIFIC_PROFILES), expected)

    def test_path_roles_match_master_routing(self) -> None:
        self.assertEqual(
            primary_dataset_ids(),
            frozenset({"ds003838", "ds006848", "ds003690"}),
        )
        self.assertEqual(
            sensitivity_dataset_ids(),
            frozenset({"ds004582", "ds004587", "ds003816", "hiit", "mindfulness"}),
        )

    def test_analysis_families(self) -> None:
        self.assertEqual(analysis_family_for("ds003690"), FAMILY_PAIRED_STATE_DEPENDENT)
        self.assertEqual(analysis_family_for("ds003838"), FAMILY_PAIRED_STATE_DEPENDENT)
        self.assertEqual(analysis_family_for("ds006848"), FAMILY_PAIRED_STATE_DEPENDENT)
        self.assertEqual(analysis_family_for("hiit"), FAMILY_EXERCISE_STATE_MODERATION)
        self.assertEqual(analysis_family_for("ds004582"), FAMILY_EXTERNAL_GENERALIZATION)
        self.assertEqual(analysis_family_for("ds004587"), FAMILY_EXTERNAL_GENERALIZATION)
        self.assertEqual(analysis_family_for("mindfulness"), FAMILY_INTERNAL_ATTENTION)
        self.assertEqual(analysis_family_for("ds003816"), FAMILY_DURATION_SENSITIVITY)

    def test_display_independent_of_pooling(self) -> None:
        for dataset_id in DATASET_SCIENTIFIC_PROFILES:
            self.assertTrue(eligible_for_dataset_display(dataset_id), dataset_id)
        for dataset_id in ("hiit", "mindfulness", "ds004582", "ds004587", "ds003816"):
            self.assertTrue(eligible_for_sensitivity_analysis(dataset_id))
            self.assertFalse(
                any(
                    eligible_for_primary_meta_pooling(dataset_id, contrast)
                    for contrast in ("rest__memory", "x", "")
                )
            )


class TestDs003690GradedDemand(unittest.TestCase):
    def test_preserves_passive_simplert_gonogo(self) -> None:
        profile = scientific_profile("ds003690")
        assert profile is not None
        self.assertEqual(profile.available_states, ("passive", "simplert", "gonogo"))
        self.assertTrue(profile.supports_graded_demand)
        self.assertTrue(profile.preserve_condition_identity)
        self.assertEqual(
            eligible_durations_for("ds003690"),
            frozenset({60, 120, 180, 240}),
        )
        spec = PROTOCOL_SPECS["ds003690"]
        self.assertEqual(spec.low_demand_conditions, ("passive",))
        self.assertEqual(spec.cognitive_effort_conditions, ("simplert", "gonogo"))
        contrast_ids = {c.contrast_id for c in spec.contrasts}
        self.assertEqual(contrast_ids, {"passive__simplert", "passive__gonogo"})

    def test_intersecting_participants_for_contrasts(self) -> None:
        observations = {
            "ds003690": [
                _obs("ds003690", "s1", "passive"),
                _obs("ds003690", "s1", "simplert"),
                _obs("ds003690", "s1", "gonogo"),
                _obs("ds003690", "s2", "passive"),
                _obs("ds003690", "s2", "simplert"),
                _obs("ds003690", "s3", "gonogo"),
            ]
        }
        payload = build_paired_subject_sets(observations)
        simplert = payload["datasets"]["ds003690"]["contrasts"]["passive__simplert"]
        gonogo = payload["datasets"]["ds003690"]["contrasts"]["passive__gonogo"]
        self.assertEqual(simplert["n_paired"], 2)
        self.assertEqual(gonogo["n_paired"], 1)
        self.assertEqual(gonogo["n_low_demand"], 2)
        self.assertEqual(gonogo["n_cognitive_effort"], 2)


class TestPairedPrimaryDatasets(unittest.TestCase):
    def test_ds003838_rest_memory_pairing(self) -> None:
        observations = {
            "ds003838": [
                _obs("ds003838", "a", "rest"),
                _obs("ds003838", "a", "memory"),
                _obs("ds003838", "b", "rest"),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrast = payload["datasets"]["ds003838"]["contrasts"]["rest__memory"]
        self.assertEqual(contrast["n_paired"], 1)
        self.assertEqual(contrast["n_low_demand"], 2)
        self.assertEqual(contrast["n_cognitive_effort"], 1)

    def test_ds006848_rest_verbalwm_pairing(self) -> None:
        observations = {
            "ds006848": [
                _obs("ds006848", "a", "rest"),
                _obs("ds006848", "a", "verbalwm"),
                _obs("ds006848", "b", "verbalwm"),
            ]
        }
        payload = build_paired_subject_sets(observations)
        contrast = payload["datasets"]["ds006848"]["contrasts"]["rest__verbalwm"]
        self.assertEqual(contrast["n_paired"], 1)
        self.assertEqual(contrast["n_low_demand"], 1)
        self.assertEqual(contrast["n_cognitive_effort"], 2)


class TestHiitPairing(unittest.TestCase):
    def test_pre_with_pre_and_post_with_post(self) -> None:
        spec = PROTOCOL_SPECS["hiit"]
        pairs = {
            (c.low_demand_condition, c.cognitive_effort_condition) for c in spec.contrasts
        }
        self.assertEqual(
            pairs,
            {
                ("ph_pre_rest", "ph_pre_tetris"),
                ("ph_post_rest", "ph_post_tetris"),
                ("ps_pre_rest", "ps_pre_tetris"),
                ("ps_post_rest", "ps_post_tetris"),
            },
        )
        for low, high in pairs:
            self.assertEqual("pre" in low, "pre" in high)
            self.assertEqual("post" in low, "post" in high)
            self.assertEqual(low.split("_")[0], high.split("_")[0])

    def test_sessions_are_not_independent_participants(self) -> None:
        row_ph = {
            "dataset_id": "hiit",
            "participant_id": "01",
            "session_id": "ph",
            "condition": "ph_pre_rest",
        }
        row_ps = {
            "dataset_id": "hiit",
            "participant_id": "01",
            "session_id": "ps",
            "condition": "ps_pre_rest",
        }
        self.assertNotEqual(
            session_unit_key(row_ph, dataset_id="hiit"),
            session_unit_key(row_ps, dataset_id="hiit"),
        )
        self.assertFalse(eligible_for_primary_meta_pooling("hiit", "ph_pre_rest__tetris"))


class TestExternalGeneralization(unittest.TestCase):
    def test_ds004582_not_forced_into_paired_contrast(self) -> None:
        self.assertEqual(PROTOCOL_SPECS["ds004582"].contrasts, ())
        self.assertFalse(has_prespecified_contrast("ds004582"))
        self.assertTrue(is_external_generalization_dataset("ds004582"))
        status, code = panel_reason_for_empty_paired("ds004582")
        self.assertEqual(status, PANEL_STATUS_NOT_APPLICABLE)
        self.assertEqual(code, NC_NOT_APPLICABLE_ESTIMAND)

    def test_ds004587_not_forced_into_paired_contrast(self) -> None:
        self.assertEqual(PROTOCOL_SPECS["ds004587"].contrasts, ())
        self.assertFalse(has_prespecified_contrast("ds004587"))
        self.assertTrue(is_sensitivity_dataset("ds004587"))
        self.assertTrue(is_external_generalization_dataset("ds004587"))
        self.assertNotIn("ds004587", PRIMARY_PAIRED_DATASETS)
        self.assertNotIn(("ds004587", "rest__ig"), PRIMARY_META_CONTRASTS)
        self.assertNotIn("rest__ig", PRIMARY_STATE_CONTRASTS)
        self.assertIn("ds004587", META_EXCLUDED_DATASETS)
        status, code = panel_reason_for_empty_paired("ds004587")
        self.assertEqual(status, PANEL_STATUS_NOT_APPLICABLE)
        self.assertEqual(code, NC_NOT_APPLICABLE_ESTIMAND)
        self.assertEqual(
            set(PROTOCOL_SPECS["ds004587"].low_demand_conditions),
            {"rest", "ig"},
        )


class TestMindfulnessSteps(unittest.TestCase):
    def test_step_identity_preserved_in_protocol(self) -> None:
        profile = scientific_profile("mindfulness")
        assert profile is not None
        self.assertEqual(profile.available_states, ("step1", "step2", "step3"))
        self.assertTrue(profile.preserve_condition_identity)
        spec = PROTOCOL_SPECS["mindfulness"]
        self.assertEqual(spec.low_demand_conditions, ("step1",))
        self.assertEqual(spec.cognitive_effort_conditions, ("step2", "step3"))
        self.assertEqual(
            {c.contrast_id for c in spec.contrasts},
            {"step1__step2", "step1__step3"},
        )

    def test_mindfulness_is_blocked_pending_participant_id_fix(self) -> None:
        from ppg_eeg.confirmatory.dataset_roles import (
            is_runtime_blocked,
            runtime_block_reason,
        )

        self.assertTrue(is_runtime_blocked("mindfulness"))
        reason = runtime_block_reason("mindfulness")
        self.assertIn("participant", reason.casefold())
        self.assertIn("C0", reason)
        self.assertFalse(eligible_for_primary_meta_pooling("mindfulness", "step1__step2"))

    def test_current_derivatives_have_zero_paired_contrasts(self) -> None:
        from pathlib import Path
        import csv

        paired = Path(
            "derivatives/confirmatory_temporal_coupling/sensitivity/mindfulness/"
            "C5/paired_contrasts.csv"
        )
        qc = Path(
            "derivatives/confirmatory_temporal_coupling/sensitivity/mindfulness/"
            "C5/pairing_qc.csv"
        )
        if not paired.is_file() or not qc.is_file():
            self.skipTest("mindfulness C5 derivatives absent")
        with paired.open(encoding="utf-8") as handle:
            paired_rows = list(csv.DictReader(handle))
        self.assertEqual(paired_rows, [])
        with qc.open(encoding="utf-8") as handle:
            qc_rows = list(csv.DictReader(handle))
        self.assertTrue(qc_rows)
        self.assertTrue(all(int(float(r.get("n_paired_keys") or 0)) == 0 for r in qc_rows))
        # Diagnostic: pairing identity embeds the step token.
        sample = qc_rows[0]
        pid = str(sample.get("participant_id") or "")
        self.assertTrue(
            any(tok in pid for tok in ("step1", "step2", "step3", "_step")),
            msg=f"expected step token in pairing participant_id, got {pid!r}",
        )


class TestDs003816Duration(unittest.TestCase):
    def test_excluded_from_d240_and_only_d60(self) -> None:
        self.assertEqual(eligible_durations_for("ds003816"), frozenset({60}))
        self.assertTrue(eligible_for_duration_analysis("ds003816", 60))
        self.assertFalse(eligible_for_duration_analysis("ds003816", 120))
        self.assertFalse(eligible_for_duration_analysis("ds003816", 180))
        self.assertFalse(eligible_for_duration_analysis("ds003816", 240))
        self.assertIn("ds003816", META_EXCLUDED_DATASETS)
        self.assertFalse(has_prespecified_contrast("ds003816"))


class TestPrimaryPoolingGate(unittest.TestCase):
    def test_only_prespecified_primary_pairs_enter_meta(self) -> None:
        self.assertEqual(
            PRIMARY_META_CONTRASTS,
            frozenset(
                {
                    ("ds003838", "rest__memory"),
                    ("ds006848", "rest__verbalwm"),
                    ("ds003690", "passive__gonogo"),
                }
            ),
        )
        for ds, contrast in PRIMARY_META_CONTRASTS:
            self.assertTrue(is_primary_dataset(ds))
            self.assertTrue(eligible_for_primary_meta_pooling(ds, contrast))
        self.assertFalse(
            eligible_for_primary_meta_pooling("ds003690", "passive__simplert")
        )
        for ds in META_EXCLUDED_DATASETS:
            self.assertFalse(eligible_for_primary_meta_pooling(ds, "any"))

    def test_sensitivity_display_not_gated_by_hiit_only(self) -> None:
        self.assertTrue(has_prespecified_contrast("mindfulness"))
        self.assertTrue(eligible_for_dataset_display("mindfulness"))
        self.assertTrue(is_sensitivity_dataset("mindfulness"))
        self.assertTrue(eligible_for_dataset_display("ds003816"))
        self.assertTrue(eligible_for_external_generalization("ds004587"))


if __name__ == "__main__":
    unittest.main()
