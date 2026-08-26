"""Output-level gates for primary meta membership and duration-matched D60."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_PRIMARY_DURATION_S,
)
from ppg_eeg.confirmatory.figures import _panel_c_supported_durations
from ppg_eeg.confirmatory.harmonize import ContiguousBlock, build_duration_segments
from ppg_eeg.confirmatory.inference import (
    META_EXCLUDED_DATASETS,
    PRIMARY_META_CONTRASTS,
    PRIMARY_POWER_REPRESENTATION,
    estimate_dataset_effects,
    run_meta_analysis,
)
from ppg_eeg.confirmatory.protocol_audit import (
    EligibilityMetadata,
    evaluate_duration_eligibility,
)
from ppg_eeg.confirmatory.reason_codes import EXCLUDED_BY_MANUSCRIPT_DESIGN
from tests.test_confirmatory_inference import (
    _paired_row,
    _synthetic_paired_from_subjects,
    _synthetic_state_effect_subjects,
)


PRIMARY_META_DATASETS = frozenset({"ds003690", "ds003838", "ds006848"})
EXCLUDED_FROM_PRIMARY_META = frozenset(
    {"ds004587", "hiit", "mindfulness", "ds004582", "ds003816"}
)


class TestPrimaryMetaMembershipOutput(unittest.TestCase):
    def test_constants_match_locked_primary_set(self) -> None:
        self.assertEqual(
            {ds for ds, _ in PRIMARY_META_CONTRASTS},
            PRIMARY_META_DATASETS,
        )
        self.assertTrue(EXCLUDED_FROM_PRIMARY_META <= META_EXCLUDED_DATASETS)

    def test_meta_analysis_inputs_only_primary_three(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.30, noise=0.04, seed=3)
        paired = _synthetic_paired_from_subjects(subjects)
        # Inject excluded cohorts with finite paired deltas.
        for i in range(6):
            pid = f"x{i:02d}"
            for dataset_id, contrast in (
                ("ds004587", "rest__ig"),
                ("hiit", "ph_pre_rest__tetris"),
                ("mindfulness", "step1__step2"),
                ("ds004582", "ff__ff"),
                ("ds003816", "preresting__lkmself"),
            ):
                paired.append(
                    _paired_row(
                        dataset_id=dataset_id,
                        contrast_id=contrast,
                        participant_id=pid,
                        band="alpha",
                        delta=-0.2,
                    )
                )
        effects = estimate_dataset_effects(paired)
        alpha_primary = [
            e
            for e in effects
            if e.get("enters_meta")
            and str(e.get("band")).casefold() == "alpha"
            and int(e.get("duration_s") or 0) == EXPECTED_PRIMARY_DURATION_S
            and str(e.get("endpoint_name")).casefold() == ENDPOINT_ZLPI
        ]
        datasets = {str(e["dataset_id"]).casefold() for e in alpha_primary}
        self.assertEqual(datasets, PRIMARY_META_DATASETS)
        for banned in EXCLUDED_FROM_PRIMARY_META:
            self.assertNotIn(banned, datasets)

        metas = run_meta_analysis(effects)
        alpha_meta = [
            m
            for m in metas
            if str(m.get("band")).casefold() == "alpha"
            and bool(m.get("is_primary_analysis"))
        ]
        self.assertTrue(alpha_meta)
        for row in alpha_meta:
            self.assertLessEqual(int(row["n_datasets"]), len(PRIMARY_META_DATASETS))
            self.assertEqual(int(row["n_datasets"]), len(datasets))

    def test_write_dataset_effects_export_membership(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.25, noise=0.03, seed=5)
        paired = _synthetic_paired_from_subjects(subjects)
        effects = estimate_dataset_effects(paired)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_effects.csv"
            fields = list(effects[0].keys())
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(effects)
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
        entering = {
            r["dataset_id"].casefold()
            for r in rows
            if str(r.get("enters_meta")).strip().casefold() in {"true", "1", "yes"}
            and r.get("band", "").casefold() == "alpha"
            and r.get("power_representation", "").casefold()
            == PRIMARY_POWER_REPRESENTATION
        }
        self.assertEqual(entering, PRIMARY_META_DATASETS)
        for banned in EXCLUDED_FROM_PRIMARY_META:
            self.assertNotIn(banned, entering)


class TestDs003816DurationMatchedD60(unittest.TestCase):
    def test_registry_and_panel_c_only_d60(self) -> None:
        self.assertEqual(_panel_c_supported_durations("ds003816"), frozenset({60}))
        self.assertEqual(_panel_c_supported_durations("ds003838"), frozenset({60, 120, 180, 240}))

    def test_c0_eligibility_excludes_non_d60_by_manuscript_design(self) -> None:
        meta = EligibilityMetadata(
            dataset_id="ds003816",
            observation_id="ds003816-sub-001-preresting",
            participant_id="sub-001",
            condition="preresting",
            contrast_id="",
            raw_overlap_s=400.0,
            clean_beat_span_s=None,
            source_data_supplied=True,
            eeg_exists=True,
            cardiac_exists=True,
            requires_paired_state=False,
            paired_state_available=True,
            pairing_resolved=True,
            protocol_match=True,
            notes="",
        )
        d60 = evaluate_duration_eligibility(meta, 60)
        self.assertEqual(d60.status, "eligible")
        self.assertEqual(d60.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(d60.endpoint_alias, "SWPI")
        for duration in (120, 180, 240):
            decision = evaluate_duration_eligibility(meta, duration)
            self.assertEqual(decision.status, "ineligible")
            self.assertEqual(decision.exclusion_code, "excluded_by_manuscript_design")
            self.assertEqual(decision.endpoint_reason_code, EXCLUDED_BY_MANUSCRIPT_DESIGN)
            self.assertFalse(decision.endpoint_computable)

    def test_harmonize_segments_only_allow_d60_for_ds003816(self) -> None:
        block = ContiguousBlock(
            start_s=0.0,
            end_s=499.0,
            time_s=tuple(float(i) for i in range(500)),
        )
        _center, segments = build_duration_segments(
            block, allowed_durations_s=frozenset({60})
        )
        self.assertTrue(segments[60].eligible)
        for duration in (120, 180, 240):
            self.assertFalse(segments[duration].eligible)
            self.assertEqual(
                segments[duration].exclusion_reason, "excluded_by_manuscript_design"
            )

    def test_duration_matched_comparators_use_same_swpi_labels(self) -> None:
        # All datasets admitted to a D60 comparison must share the SWPI estimand.
        for dataset_id in ("ds003816", "ds003838", "hiit", "ds006848"):
            meta = EligibilityMetadata(
                dataset_id=dataset_id,
                observation_id=f"{dataset_id}-p1",
                participant_id="p1",
                condition="rest",
                contrast_id="",
                raw_overlap_s=300.0,
                clean_beat_span_s=None,
                source_data_supplied=True,
                eeg_exists=True,
                cardiac_exists=True,
                requires_paired_state=False,
                paired_state_available=True,
                pairing_resolved=True,
                protocol_match=True,
                notes="",
            )
            decision = evaluate_duration_eligibility(meta, 60)
            if dataset_id == "ds003816" or 60 in _panel_c_supported_durations(dataset_id):
                self.assertEqual(decision.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
                self.assertEqual(decision.endpoint_alias, "SWPI")
                self.assertFalse(decision.is_standard_zlpi)


class TestStaleDs004587Path(unittest.TestCase):
    def test_legacy_primary_path_marked_stale(self) -> None:
        legacy = Path("derivatives/confirmatory_temporal_coupling/primary/ds004587")
        if legacy.is_dir():
            self.assertTrue((legacy / "STALE_INVALIDATED.txt").is_file())


if __name__ == "__main__":
    unittest.main()
