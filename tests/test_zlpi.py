from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    contract_for_duration,
)
from ppg_eeg.confirmatory.endpoints import (
    FISHER_R_CLIP,
    METRICS_TEMPLATE,
    QC_TEMPLATE,
    combined_flank_lags,
    compute_endpoints_from_curves,
    evaluate_endpoint_curve,
    expected_lag_grid,
    fisher_z,
    min_common_support_required,
    negative_flank_lags,
    negative_shoulder_lags,
    positive_flank_lags,
    positive_shoulder_lags,
    write_endpoint_outputs,
)


def _base_identity(**overrides: object) -> dict[str, object]:
    row = {
        "dataset_id": "ds_test",
        "subject_id": "sub-01",
        "task": "rest",
        "condition": "rest",
        "observation_id": "obs-01",
        "duration_role": "primary",
        "band": "theta",
        "power_representation": "absolute_log10",
        "is_primary_representation": True,
        "pair": "hr_x_theta_absolute_log10",
    }
    row.update(overrides)
    return row


def _curve_from_lag_map(
    lag_to_r: dict[int, float],
    *,
    duration_s: int,
    n_overlap: int | None = None,
    **identity_overrides: object,
) -> list[dict[str, object]]:
    contract = contract_for_duration(duration_s)
    if n_overlap is None:
        n_overlap = contract.expected_constant_overlap_if_fully_finite
    identity = _base_identity(**identity_overrides)
    rows: list[dict[str, object]] = []
    for lag in expected_lag_grid(contract):
        rows.append(
            {
                **identity,
                "duration_s": duration_s,
                "lag_analysis_role": contract.lag_analysis_role,
                "endpoint_name": contract.endpoint_name,
                "endpoint_alias": contract.endpoint_alias,
                "is_standard_zlpi": contract.is_standard_zlpi,
                "pool_with_standard_zlpi": contract.pool_with_standard_zlpi,
                "flank_inner_s": contract.flank_inner_s,
                "flank_outer_s": contract.flank_outer_s,
                "shoulders_inner_s": contract.shoulders_inner_s,
                "shoulders_outer_s": contract.shoulders_outer_s,
                "lag_s": float(lag),
                "r": float(lag_to_r[lag]),
                "n_overlap": int(n_overlap),
            }
        )
    return rows


class TestFisherZ(unittest.TestCase):
    def test_clipping_near_unit_correlation(self) -> None:
        z_clip = fisher_z(1.0)
        self.assertAlmostEqual(z_clip, math.atanh(FISHER_R_CLIP), places=12)
        self.assertAlmostEqual(fisher_z(-1.0), -math.atanh(FISHER_R_CLIP), places=12)
        self.assertAlmostEqual(fisher_z(0.5), math.atanh(0.5), places=12)
        self.assertTrue(math.isnan(fisher_z(float("nan"))))


class TestEndpointWindows(unittest.TestCase):
    def test_zlpi_flank_and_shoulder_windows(self) -> None:
        contract = contract_for_duration(240)
        self.assertEqual(negative_flank_lags(contract)[0], -60)
        self.assertEqual(negative_flank_lags(contract)[-1], -20)
        self.assertEqual(positive_flank_lags(contract)[0], 20)
        self.assertEqual(positive_flank_lags(contract)[-1], 60)
        self.assertEqual(len(combined_flank_lags(contract)), 82)
        self.assertEqual(negative_shoulder_lags(contract), tuple(range(-15, -4)))
        self.assertEqual(positive_shoulder_lags(contract), tuple(range(5, 16)))

    def test_mwpi_and_swpi_windows(self) -> None:
        mwpi = contract_for_duration(120)
        self.assertEqual(negative_flank_lags(mwpi), tuple(range(-30, -19)))
        self.assertEqual(positive_flank_lags(mwpi), tuple(range(20, 31)))
        swpi = contract_for_duration(60)
        self.assertEqual(negative_flank_lags(swpi), tuple(range(-20, -9)))
        self.assertEqual(positive_flank_lags(swpi), tuple(range(10, 21)))


class TestZlpiEndpoints(unittest.TestCase):
    def test_hand_calculated_zlpi_and_prominence(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.0 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.5
        for lag in combined_flank_lags(contract):
            lag_to_r[lag] = 0.1
        for lag in negative_shoulder_lags(contract):
            lag_to_r[lag] = 0.2
        for lag in positive_shoulder_lags(contract):
            lag_to_r[lag] = 0.3

        metrics, qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=240),
            duration_s=240,
        )
        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_ZLPI)
        self.assertTrue(metrics["is_standard_zlpi"])
        self.assertTrue(metrics["pool_with_standard_zlpi"])

        z0 = fisher_z(0.5)
        flank_z = fisher_z(0.1)
        neg_s = fisher_z(0.2)
        pos_s = fisher_z(0.3)
        self.assertAlmostEqual(float(metrics["r0"]), 0.5, places=12)
        self.assertAlmostEqual(float(metrics["z0"]), z0, places=12)
        self.assertAlmostEqual(float(metrics["combined_flank_mean_z"]), flank_z, places=12)
        self.assertAlmostEqual(float(metrics["negative_flank_mean_z"]), flank_z, places=12)
        self.assertAlmostEqual(float(metrics["positive_flank_mean_z"]), flank_z, places=12)
        self.assertAlmostEqual(float(metrics["endpoint_index"]), z0 - flank_z, places=12)
        self.assertAlmostEqual(float(metrics["negative_shoulder_mean_z"]), neg_s, places=12)
        self.assertAlmostEqual(float(metrics["positive_shoulder_mean_z"]), pos_s, places=12)
        self.assertAlmostEqual(
            float(metrics["local_prominence"]), z0 - max(neg_s, pos_s), places=12
        )
        self.assertTrue(qc["lag_grid_complete"])
        self.assertTrue(qc["all_required_r_finite"])

    def test_flat_curve_gives_zero_endpoint_and_prominence(self) -> None:
        contract = contract_for_duration(180)
        lag_to_r = {lag: 0.25 for lag in expected_lag_grid(contract)}
        metrics, _qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=180, duration_role="nested_sensitivity"),
            duration_s=180,
        )
        self.assertTrue(metrics["eligible"])
        self.assertAlmostEqual(float(metrics["endpoint_index"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["local_prominence"]), 0.0, places=12)
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_ZLPI)

    def test_asymmetric_flanks_are_recorded_separately(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.0 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.4
        for lag in negative_flank_lags(contract):
            lag_to_r[lag] = 0.05
        for lag in positive_flank_lags(contract):
            lag_to_r[lag] = 0.20
        for lag in negative_shoulder_lags(contract) + positive_shoulder_lags(contract):
            lag_to_r[lag] = 0.10

        metrics, _qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=240),
            duration_s=240,
        )
        neg = fisher_z(0.05)
        pos = fisher_z(0.20)
        combined = 0.5 * (neg + pos)
        self.assertAlmostEqual(float(metrics["negative_flank_mean_z"]), neg, places=12)
        self.assertAlmostEqual(float(metrics["positive_flank_mean_z"]), pos, places=12)
        self.assertAlmostEqual(float(metrics["combined_flank_mean_z"]), combined, places=12)
        self.assertAlmostEqual(
            float(metrics["endpoint_index"]),
            fisher_z(0.4) - combined,
            places=12,
        )

    def test_missing_lag_rejects_incomplete_grid(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.1 for lag in expected_lag_grid(contract)}
        del lag_to_r[30]
        # Build incomplete rows manually.
        rows = [
            row
            for row in _curve_from_lag_map(
                {**lag_to_r, 30: 0.1}, duration_s=240
            )
            if int(row["lag_s"]) != 30
        ]
        metrics, qc = evaluate_endpoint_curve(rows, duration_s=240)
        self.assertFalse(metrics["eligible"])
        self.assertEqual(metrics["exclusion_reason"], "incomplete_lag_grid")
        self.assertFalse(qc["lag_grid_complete"])
        self.assertTrue(math.isnan(float(metrics["endpoint_index"])))

    def test_nonfinite_required_correlation_rejected(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.1 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.4
        lag_to_r[-40] = float("nan")
        metrics, qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=240),
            duration_s=240,
        )
        self.assertFalse(metrics["eligible"])
        self.assertEqual(metrics["exclusion_reason"], "nonfinite_required_correlations")
        self.assertFalse(qc["all_required_r_finite"])

    def test_insufficient_common_support_rejected(self) -> None:
        contract = contract_for_duration(180)
        lag_to_r = {lag: 0.2 for lag in expected_lag_grid(contract)}
        metrics, qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=180, n_overlap=40),
            duration_s=180,
        )
        self.assertEqual(min_common_support_required(contract), 60)
        self.assertFalse(metrics["eligible"])
        self.assertEqual(metrics["exclusion_reason"], "insufficient_common_support")
        self.assertEqual(qc["min_common_support_required"], 60)

    def test_condition_recovered_from_hiit_observation_id(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.1 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.4
        rows = _curve_from_lag_map(
            lag_to_r,
            duration_s=240,
            condition="",
            dataset_id="hiit",
            observation_id="hiit-01-ph-pre-rest",
        )
        metrics, qc = evaluate_endpoint_curve(rows, duration_s=240)
        self.assertEqual(metrics["condition"], "ph_pre_rest")
        self.assertEqual(qc["condition"], "ph_pre_rest")
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_ZLPI)
        self.assertEqual(metrics["endpoint_alias"], "ZLPI")

    def test_clipping_propagates_into_endpoint(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.0 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 1.0
        for lag in combined_flank_lags(contract):
            lag_to_r[lag] = -1.0
        for lag in negative_shoulder_lags(contract) + positive_shoulder_lags(contract):
            lag_to_r[lag] = 0.0
        metrics, _qc = evaluate_endpoint_curve(
            _curve_from_lag_map(lag_to_r, duration_s=240),
            duration_s=240,
        )
        expected = fisher_z(1.0) - fisher_z(-1.0)
        self.assertAlmostEqual(float(metrics["endpoint_index"]), expected, places=12)
        self.assertAlmostEqual(float(metrics["z0"]), fisher_z(1.0), places=12)


class TestDurationSpecificNames(unittest.TestCase):
    def test_d120_is_mwpi_not_zlpi(self) -> None:
        contract = contract_for_duration(120)
        lag_to_r = {lag: 0.15 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.4
        metrics, _qc = evaluate_endpoint_curve(
            _curve_from_lag_map(
                lag_to_r,
                duration_s=120,
                duration_role="nested_sensitivity",
            ),
            duration_s=120,
        )
        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(metrics["endpoint_alias"], "MWPI")
        self.assertFalse(metrics["is_standard_zlpi"])
        self.assertFalse(metrics["pool_with_standard_zlpi"])
        self.assertEqual(metrics["flank_inner_s"], 20)
        self.assertEqual(metrics["flank_outer_s"], 30)

    def test_d60_is_swpi_not_zlpi(self) -> None:
        contract = contract_for_duration(60)
        lag_to_r = {lag: 0.1 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.3
        metrics, _qc = evaluate_endpoint_curve(
            _curve_from_lag_map(
                lag_to_r,
                duration_s=60,
                duration_role="separate_sensitivity",
                n_overlap=20,
            ),
            duration_s=60,
        )
        self.assertTrue(metrics["eligible"])
        self.assertEqual(metrics["endpoint_name"], ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(metrics["endpoint_alias"], "SWPI")
        self.assertFalse(metrics["is_standard_zlpi"])
        self.assertFalse(metrics["pool_with_standard_zlpi"])
        self.assertEqual(metrics["flank_inner_s"], 10)
        self.assertEqual(metrics["flank_outer_s"], 20)

    def test_batch_outputs_and_schemas(self) -> None:
        contract = contract_for_duration(240)
        lag_to_r = {lag: 0.05 for lag in expected_lag_grid(contract)}
        lag_to_r[0] = 0.35
        rows = _curve_from_lag_map(lag_to_r, duration_s=240)
        # Second band/representation.
        rows.extend(
            _curve_from_lag_map(
                lag_to_r,
                duration_s=240,
                band="alpha",
                power_representation="relative",
                is_primary_representation=False,
                pair="hr_x_alpha_relative",
            )
        )
        result = compute_endpoints_from_curves(rows, duration_s=240)
        self.assertEqual(len(result.metrics_rows), 2)
        with TemporaryDirectory() as tmp:
            paths = write_endpoint_outputs(result, tmp)
            self.assertEqual(
                paths["metrics"].name, METRICS_TEMPLATE.format(duration_s=240)
            )
            self.assertEqual(paths["qc"].name, QC_TEMPLATE.format(duration_s=240))
            self.assertTrue(paths["metrics"].is_file())
            self.assertTrue(paths["qc"].is_file())


if __name__ == "__main__":
    unittest.main()
