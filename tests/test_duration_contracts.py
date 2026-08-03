from __future__ import annotations

import unittest

from ppg_eeg.confirmatory.duration_contracts import (
    DURATION_ANALYSIS_CONTRACTS,
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_DURATIONS_S,
    EXPECTED_SHOULDERS_S,
    MWPI_FLANKS_S,
    STANDARD_ZLPI_DURATIONS_S,
    SWPI_FLANKS_S,
    ZLPI_FLANKS_S,
    assert_contracts_internally_consistent,
    contract_for_duration,
    standard_zlpi_expected_n_overlap,
    standard_zlpi_is_computable,
    standard_zlpi_pool_durations,
)


class TestDurationContracts(unittest.TestCase):
    def test_documented_endpoint_names_and_windows(self) -> None:
        assert_contracts_internally_consistent()
        self.assertEqual(tuple(DURATION_ANALYSIS_CONTRACTS), EXPECTED_DURATIONS_S)
        self.assertEqual(standard_zlpi_pool_durations(), STANDARD_ZLPI_DURATIONS_S)

        d240 = contract_for_duration(240)
        self.assertEqual(d240.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(d240.flanks_s, ZLPI_FLANKS_S)
        self.assertEqual(d240.shoulders_s, EXPECTED_SHOULDERS_S)
        self.assertEqual((d240.lag_min_s, d240.lag_max_s, d240.n_lags), (-60, 60, 121))
        self.assertTrue(d240.is_standard_zlpi)
        self.assertTrue(d240.pool_with_standard_zlpi)
        self.assertEqual(d240.expected_constant_overlap_if_fully_finite, 120)

        d180 = contract_for_duration(180)
        self.assertEqual(d180.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(d180.flanks_s, ZLPI_FLANKS_S)
        self.assertEqual((d180.lag_min_s, d180.lag_max_s, d180.n_lags), (-60, 60, 121))
        self.assertEqual(d180.expected_constant_overlap_if_fully_finite, 60)

        d120 = contract_for_duration(120)
        self.assertEqual(d120.endpoint_name, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(d120.endpoint_alias, "MWPI")
        self.assertEqual(d120.flanks_s, MWPI_FLANKS_S)
        self.assertEqual(d120.shoulders_s, EXPECTED_SHOULDERS_S)
        self.assertEqual((d120.lag_min_s, d120.lag_max_s, d120.n_lags), (-30, 30, 61))
        self.assertFalse(d120.is_standard_zlpi)
        self.assertFalse(d120.pool_with_standard_zlpi)
        self.assertEqual(d120.expected_constant_overlap_if_fully_finite, 60)

        d60 = contract_for_duration(60)
        self.assertEqual(d60.endpoint_name, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(d60.endpoint_alias, "SWPI")
        self.assertEqual(d60.flanks_s, SWPI_FLANKS_S)
        self.assertEqual(d60.shoulders_s, EXPECTED_SHOULDERS_S)
        self.assertEqual((d60.lag_min_s, d60.lag_max_s, d60.n_lags), (-20, 20, 41))
        self.assertFalse(d60.is_standard_zlpi)
        self.assertFalse(d60.pool_with_standard_zlpi)
        self.assertEqual(d60.expected_constant_overlap_if_fully_finite, 20)

    def test_non_zlpi_endpoints_are_never_named_zlpi(self) -> None:
        for duration in (120, 60):
            contract = contract_for_duration(duration)
            self.assertNotEqual(contract.endpoint_name, ENDPOINT_ZLPI)
            self.assertFalse(contract.is_standard_zlpi)
            self.assertFalse(contract.pool_with_standard_zlpi)

    def test_standard_zlpi_computability_by_overlap(self) -> None:
        self.assertEqual(standard_zlpi_expected_n_overlap(60), -60)
        self.assertEqual(standard_zlpi_expected_n_overlap(120), 0)
        self.assertEqual(standard_zlpi_expected_n_overlap(180), 60)
        self.assertEqual(standard_zlpi_expected_n_overlap(240), 120)
        self.assertFalse(standard_zlpi_is_computable(60))
        self.assertFalse(standard_zlpi_is_computable(120))
        self.assertTrue(standard_zlpi_is_computable(180))
        self.assertTrue(standard_zlpi_is_computable(240))


if __name__ == "__main__":
    unittest.main()
