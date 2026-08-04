"""Duration eligibility must not confuse extraction eligibility with standard ZLPI."""

from __future__ import annotations

import unittest

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
)
from ppg_eeg.confirmatory.figures import _validate_panel_c_rows
from ppg_eeg.confirmatory.protocol_audit import (
    EligibilityMetadata,
    evaluate_duration_eligibility,
)
from ppg_eeg.confirmatory.reason_codes import ENDPOINT_CONTRACT_NON_ZLPI


def _meta(*, raw_overlap_s: float = 300.0) -> EligibilityMetadata:
    return EligibilityMetadata(
        dataset_id="hiit",
        observation_id="obs-1",
        participant_id="p01",
        condition="rest_pre_ph",
        contrast_id="rest__tetris",
        raw_overlap_s=raw_overlap_s,
        clean_beat_span_s=None,
        source_data_supplied=True,
        eeg_exists=True,
        cardiac_exists=True,
        requires_paired_state=True,
        paired_state_available=True,
        pairing_resolved=True,
        protocol_match=True,
        notes="",
    )


class TestDurationEligibilitySemantics(unittest.TestCase):
    def test_d60_and_d120_never_standard_zlpi_computable(self) -> None:
        meta = _meta(raw_overlap_s=400.0)
        d60 = evaluate_duration_eligibility(meta, 60)
        d120 = evaluate_duration_eligibility(meta, 120)

        self.assertEqual(d60.status, "eligible")
        self.assertEqual(d120.status, "eligible")
        self.assertTrue(d60.endpoint_computable)
        self.assertTrue(d120.endpoint_computable)
        self.assertEqual(d60.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(d120.endpoint_name, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertFalse(d60.is_standard_zlpi)
        self.assertFalse(d120.is_standard_zlpi)
        self.assertFalse(d60.standard_zlpi_computable)
        self.assertFalse(d120.standard_zlpi_computable)
        self.assertEqual(d60.standard_zlpi_reason_code, ENDPOINT_CONTRACT_NON_ZLPI)
        self.assertEqual(d120.standard_zlpi_reason_code, ENDPOINT_CONTRACT_NON_ZLPI)
        self.assertNotEqual(d60.endpoint_name, ENDPOINT_ZLPI)
        self.assertNotEqual(d120.endpoint_name, ENDPOINT_ZLPI)

    def test_d180_d240_standard_zlpi_when_support_ok(self) -> None:
        meta = _meta(raw_overlap_s=250.0)
        d180 = evaluate_duration_eligibility(meta, 180)
        d240 = evaluate_duration_eligibility(meta, 240)
        self.assertTrue(d180.standard_zlpi_computable)
        self.assertTrue(d240.standard_zlpi_computable)
        self.assertEqual(d180.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(d240.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(d180.endpoint_status, "computable")
        self.assertEqual(d240.endpoint_status, "computable")

    def test_insufficient_duration_blocks_endpoint(self) -> None:
        meta = _meta(raw_overlap_s=100.0)
        d180 = evaluate_duration_eligibility(meta, 180)
        self.assertEqual(d180.status, "ineligible")
        self.assertFalse(d180.endpoint_computable)
        self.assertFalse(d180.standard_zlpi_computable)
        self.assertFalse(d180.has_required_duration)
        self.assertEqual(d180.endpoint_status, "ineligible")

    def test_panel_c_rejects_d60_d120_labeled_as_zlpi(self) -> None:
        bad_d60 = {
            "dataset_id": "hiit",
            "duration_s": 60,
            "band": "alpha",
            "endpoint_name": ENDPOINT_ZLPI,
            "is_standard_zlpi": True,
            "lag_max_s": 20,
            "flank_inner_s": 10,
            "flank_outer_s": 20,
            "eligibility_status": "eligible",
            "effect_estimate": 0.1,
        }
        with self.assertRaisesRegex(
            RuntimeError,
            r"endpoint identity error at D60|forbids labeling D60 as ZLPI|must be SWPI",
        ):
            _validate_panel_c_rows([bad_d60])

        bad_d120 = {
            "dataset_id": "hiit",
            "duration_s": 120,
            "band": "alpha",
            "endpoint_name": ENDPOINT_ZLPI,
            "is_standard_zlpi": True,
            "lag_max_s": 30,
            "flank_inner_s": 20,
            "flank_outer_s": 30,
            "eligibility_status": "eligible",
            "effect_estimate": 0.1,
        }
        with self.assertRaisesRegex(
            RuntimeError,
            r"endpoint identity error at D120|forbids labeling D120 as ZLPI|must be MWPI",
        ):
            _validate_panel_c_rows([bad_d120])

    def test_panel_c_accepts_swpi_mwpi_not_as_standard_zlpi(self) -> None:
        rows = [
            {
                "dataset_id": "hiit",
                "duration_s": 60,
                "band": "alpha",
                "endpoint_name": ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
                "is_standard_zlpi": False,
                "lag_max_s": 20,
                "flank_inner_s": 10,
                "flank_outer_s": 20,
                "eligibility_status": "eligible",
                "effect_estimate": 0.1,
            },
            {
                "dataset_id": "hiit",
                "duration_s": 120,
                "band": "alpha",
                "endpoint_name": ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                "is_standard_zlpi": False,
                "lag_max_s": 30,
                "flank_inner_s": 20,
                "flank_outer_s": 30,
                "eligibility_status": "eligible",
                "effect_estimate": 0.2,
            },
        ]
        _validate_panel_c_rows(rows)


if __name__ == "__main__":
    unittest.main()
