"""Aggregation manifests preserve endpoint-family compatibility and NC reasons."""

from __future__ import annotations

import unittest

from ppg_eeg.confirmatory.group_tables import build_aggregation_manifest


class TestAggregationManifest(unittest.TestCase):
    def test_endpoint_families_are_explicit_and_not_interchangeable(self) -> None:
        rows = [
            {
                "dataset_id": "primary",
                "endpoint_name": "zlpi",
                "duration_s": 240,
                "power_representation": "absolute_log10",
                "contrast_eligible": True,
            },
            {
                "dataset_id": "sensitivity",
                "endpoint_name": "short_window_proximal_index",
                "duration_s": 60,
                "power_representation": "absolute_log10",
                "contrast_eligible": False,
                "exclusion_reason": "insufficient_duration",
            },
        ]
        manifest = build_aggregation_manifest(
            rows, dataset_roles={"primary": "primary", "sensitivity": "sensitivity"}
        )
        self.assertEqual(manifest[0]["endpoint_family"], "standard_zlpi")
        self.assertEqual(manifest[1]["endpoint_family"], "swpi")
        self.assertEqual(manifest[1]["status"], "excluded")
        self.assertEqual(manifest[1]["reason_code"], "insufficient_duration")
        self.assertTrue(manifest[1]["reason"])


if __name__ == "__main__":
    unittest.main()
