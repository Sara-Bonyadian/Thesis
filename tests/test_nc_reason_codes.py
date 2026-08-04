"""Expand NC reason-code taxonomy coverage."""

from __future__ import annotations

import unittest

from ppg_eeg.confirmatory.panel_d_cardiac_controls import short_not_computable_reason_code
from ppg_eeg.confirmatory.reason_codes import (
    ARTIFACT_CONTROL_NOT_AVAILABLE,
    CONFIGURATION_VALIDATION_FAILED,
    ENDPOINT_CONTRACT_NON_ZLPI,
    INSUFFICIENT_COMMON_MONTAGE,
    INSUFFICIENT_COMMON_SUPPORT,
    INSUFFICIENT_DURATION,
    INSUFFICIENT_LAG_SUPPORT,
    INPUT_DISCOVERY_FAILED,
    MISSING_CONDITION_MAPPING,
    MISSING_DATASET_ROOT,
    MISSING_EVENT_SERIES,
    MISSING_PAIRED_OBSERVATION,
    MISSING_REQUIRED_BAND,
    MISSING_REQUIRED_MODALITY,
    MISSING_SENSOR_LOCATIONS,
    NO_CHANNEL_LEVEL_ENDPOINT_VALUES,
    REASON_CODES,
    STANDARD_ZLPI_NOT_APPLICABLE,
    TOPOGRAPHY_NOT_SUPPORTED,
    UNDEFINED_NULL_VARIANCE,
    UNSUPPORTED_CONTROL_FOR_MODALITY,
    structured_reason,
)


class TestNcReasonCodes(unittest.TestCase):
    def test_specific_reason_codes_are_stable(self) -> None:
        self.assertEqual(
            short_not_computable_reason_code("insufficient support for paired estimate"),
            "insufficient support",
        )
        self.assertEqual(
            short_not_computable_reason_code("missing ecg signal"),
            "not computable",
        )
        self.assertNotEqual(
            short_not_computable_reason_code("undefined null variance"),
            "",
        )

    def test_taxonomy_includes_required_codes(self) -> None:
        required = {
            MISSING_DATASET_ROOT,
            INPUT_DISCOVERY_FAILED,
            MISSING_REQUIRED_MODALITY,
            MISSING_EVENT_SERIES,
            UNSUPPORTED_CONTROL_FOR_MODALITY,
            INSUFFICIENT_DURATION,
            INSUFFICIENT_LAG_SUPPORT,
            INSUFFICIENT_COMMON_SUPPORT,
            MISSING_CONDITION_MAPPING,
            MISSING_PAIRED_OBSERVATION,
            TOPOGRAPHY_NOT_SUPPORTED,
            MISSING_SENSOR_LOCATIONS,
            INSUFFICIENT_COMMON_MONTAGE,
            MISSING_REQUIRED_BAND,
            NO_CHANNEL_LEVEL_ENDPOINT_VALUES,
            ARTIFACT_CONTROL_NOT_AVAILABLE,
            UNDEFINED_NULL_VARIANCE,
            CONFIGURATION_VALIDATION_FAILED,
            STANDARD_ZLPI_NOT_APPLICABLE,
            ENDPOINT_CONTRACT_NON_ZLPI,
        }
        self.assertTrue(required.issubset(set(REASON_CODES)))

    def test_structured_reason_fields(self) -> None:
        row = structured_reason(
            status="not_computable",
            reason_code=INSUFFICIENT_LAG_SUPPORT,
            reason="D180 lacks ±60-s flank support",
            required_evidence="raw_overlap_s >= 180 with lag_max=60",
            observed_evidence="raw_overlap_s=120",
        )
        payload = row.to_row()
        self.assertEqual(payload["status"], "not_computable")
        self.assertEqual(payload["reason_code"], INSUFFICIENT_LAG_SUPPORT)
        self.assertIn("flank", payload["reason"])
        self.assertTrue(payload["required_evidence"])
        self.assertTrue(payload["observed_evidence"])


if __name__ == "__main__":
    unittest.main()
