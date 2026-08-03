"""Figure 3 Panel C mixed duration-specific proximal endpoint guards."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
    EXPECTED_DURATIONS_S,
    MWPI_ALIAS,
    MWPI_FLANKS_S,
    MWPI_LAG_MAX_S,
    STANDARD_ZLPI_LAG_MAX_S,
    SWPI_ALIAS,
    SWPI_FLANKS_S,
    SWPI_LAG_MAX_S,
    ZLPI_FLANKS_S,
    contract_for_duration,
)
from ppg_eeg.confirmatory.endpoints import evaluate_endpoint_curve
from ppg_eeg.confirmatory.figures import (
    _panel_c_curve_endpoint_rows,
    _validate_panel_c_rows,
    render_figure3,
)
from ppg_eeg.confirmatory.harmonize import CENTER_SELECTION, ContiguousBlock, build_duration_segments


EXPECTED_BY_DURATION = {
    60: ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    120: ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    180: ENDPOINT_ZLPI,
    240: ENDPOINT_ZLPI,
}


def _paired_row(
    *,
    participant_id: str,
    contrast_id: str,
    band: str,
    duration_s: int,
    low_observation_id: str,
    effort_observation_id: str,
    dataset_id: str = "hiit",
    eligible: bool = True,
    representation: str = "absolute_log10",
    session_id: str = "single",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "contrast_id": contrast_id,
        "band": band,
        "duration_s": duration_s,
        "endpoint_name": "legacy_mixed_endpoint",
        "power_representation": representation,
        "contrast_eligible": eligible,
        "low_observation_ids": low_observation_id,
        "effort_observation_ids": effort_observation_id,
    }


def _curve_rows_for_observation(
    *,
    dataset_id: str,
    participant_id: str,
    condition: str,
    observation_id: str,
    duration_s: int,
    band: str,
    base_r: float,
    zero_lag_bump: float,
) -> list[dict[str, object]]:
    contract = contract_for_duration(duration_s)
    rows: list[dict[str, object]] = []
    for lag in range(contract.lag_min_s, contract.lag_max_s + 1):
        r = base_r + (zero_lag_bump if lag == 0 else 0.0)
        rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": participant_id,
                "task": "task",
                "condition": condition,
                "observation_id": observation_id,
                "duration_s": duration_s,
                "duration_role": "sensitivity",
                "band": band,
                "power_representation": "absolute_log10",
                "pair": "default",
                "lag_s": lag,
                "r": max(-0.95, min(0.95, r)),
                "n_overlap": contract.expected_constant_overlap_if_fully_finite,
                "is_primary_representation": True,
            }
        )
    return rows


class TestPanelCDurationSensitivity(unittest.TestCase):
    def test_midpoint_nested_windows_are_subsets(self) -> None:
        self.assertEqual(CENTER_SELECTION, "midpoint_of_longest_common_support")
        times = tuple(float(t) for t in range(0, 300))
        block = ContiguousBlock(start_s=0.0, end_s=299.0, time_s=times)
        _center, segments = build_duration_segments(block)
        sets = {d: set(segments[d].time_s) for d in (240, 180, 120, 60)}
        self.assertTrue(sets[60] <= sets[120] <= sets[180] <= sets[240])

    def test_evaluate_endpoint_identity_preserved_per_duration(self) -> None:
        for duration, expected in EXPECTED_BY_DURATION.items():
            rows = _curve_rows_for_observation(
                dataset_id="hiit",
                participant_id="01",
                condition="rest",
                observation_id=f"obs-{duration}",
                duration_s=duration,
                band="theta",
                base_r=0.01,
                zero_lag_bump=0.05,
            )
            metrics, _qc = evaluate_endpoint_curve(rows, duration_s=duration)
            contract = contract_for_duration(duration)
            self.assertEqual(metrics["endpoint_name"], expected)
            self.assertEqual(bool(metrics["is_standard_zlpi"]), contract.is_standard_zlpi)
            self.assertEqual(int(metrics["flank_inner_s"]), contract.flank_inner_s)
            self.assertEqual(int(metrics["flank_outer_s"]), contract.flank_outer_s)

    def test_panel_c_builder_preserves_contract_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs: dict[str, Path] = {}
            for duration in EXPECTED_DURATIONS_S:
                rows = _curve_rows_for_observation(
                    dataset_id="hiit",
                    participant_id="01",
                    condition="rest",
                    observation_id=f"hiit-01-{duration}",
                    duration_s=duration,
                    band="theta",
                    base_r=0.01,
                    zero_lag_bump=0.04,
                )
                path = root / f"curves_D{duration}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                inputs[f"curves_d{duration}"] = path
            built = _panel_c_curve_endpoint_rows(inputs)
            by_duration = {int(r["duration_s"]): r for r in built}
            self.assertEqual(sorted(by_duration), [60, 120, 180, 240])
            self.assertEqual(
                by_duration[60]["endpoint_name"], ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX
            )
            self.assertEqual(by_duration[60]["endpoint_alias"], SWPI_ALIAS)
            self.assertFalse(bool(by_duration[60]["is_standard_zlpi"]))
            self.assertEqual(int(by_duration[60]["lag_max_s"]), SWPI_LAG_MAX_S)
            self.assertEqual(
                (int(by_duration[60]["flank_inner_s"]), int(by_duration[60]["flank_outer_s"])),
                tuple(SWPI_FLANKS_S),
            )
            self.assertEqual(
                by_duration[120]["endpoint_name"], ENDPOINT_MID_WINDOW_PROXIMAL_INDEX
            )
            self.assertEqual(by_duration[120]["endpoint_alias"], MWPI_ALIAS)
            self.assertFalse(bool(by_duration[120]["is_standard_zlpi"]))
            self.assertEqual(int(by_duration[120]["lag_max_s"]), MWPI_LAG_MAX_S)
            self.assertEqual(
                (
                    int(by_duration[120]["flank_inner_s"]),
                    int(by_duration[120]["flank_outer_s"]),
                ),
                tuple(MWPI_FLANKS_S),
            )
            for duration in (180, 240):
                self.assertEqual(by_duration[duration]["endpoint_name"], ENDPOINT_ZLPI)
                self.assertTrue(bool(by_duration[duration]["is_standard_zlpi"]))
                self.assertEqual(
                    int(by_duration[duration]["lag_max_s"]), STANDARD_ZLPI_LAG_MAX_S
                )
                self.assertEqual(
                    (
                        int(by_duration[duration]["flank_inner_s"]),
                        int(by_duration[duration]["flank_outer_s"]),
                    ),
                    tuple(ZLPI_FLANKS_S),
                )

    def test_validate_rejects_false_zlpi_relabel(self) -> None:
        bad_d60 = [
            {
                "dataset_id": "hiit",
                "duration_s": 60,
                "band": "theta",
                "endpoint_name": ENDPOINT_ZLPI,
                "is_standard_zlpi": True,
                "lag_max_s": SWPI_LAG_MAX_S,
                "flank_inner_s": SWPI_FLANKS_S[0],
                "flank_outer_s": SWPI_FLANKS_S[1],
                "eligibility_status": "eligible",
                "effect_estimate": 0.1,
                "exclusion_reason": "",
            }
        ]
        with self.assertRaises(RuntimeError):
            _validate_panel_c_rows(bad_d60)
        bad_d180 = [
            {
                "dataset_id": "hiit",
                "duration_s": 180,
                "band": "theta",
                "endpoint_name": ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                "is_standard_zlpi": False,
                "lag_max_s": STANDARD_ZLPI_LAG_MAX_S,
                "flank_inner_s": ZLPI_FLANKS_S[0],
                "flank_outer_s": ZLPI_FLANKS_S[1],
                "eligibility_status": "eligible",
                "effect_estimate": 0.1,
                "exclusion_reason": "",
            }
        ]
        with self.assertRaises(RuntimeError):
            _validate_panel_c_rows(bad_d180)

    def test_panel_c_exports_mixed_endpoints_and_ds003816_rule(self) -> None:
        rows = []
        curve_files: dict[int, list[dict[str, object]]] = {
            d: [] for d in EXPECTED_DURATIONS_S
        }
        for duration in EXPECTED_DURATIONS_S:
            for participant in ("01", "02"):
                low_obs = f"hiit-{participant}-ph-{duration}-low"
                eff_obs = f"hiit-{participant}-ph-{duration}-eff"
                rows.append(
                    _paired_row(
                        participant_id=participant,
                        contrast_id="ph_pre_rest__tetris",
                        band="theta",
                        duration_s=duration,
                        low_observation_id=low_obs,
                        effort_observation_id=eff_obs,
                        dataset_id="hiit",
                        session_id="ph",
                    )
                )
                curve_files[duration].extend(
                    _curve_rows_for_observation(
                        dataset_id="hiit",
                        participant_id=participant,
                        condition="ph_pre_rest",
                        observation_id=low_obs,
                        duration_s=duration,
                        band="theta",
                        base_r=0.01,
                        zero_lag_bump=0.01,
                    )
                )
                curve_files[duration].extend(
                    _curve_rows_for_observation(
                        dataset_id="hiit",
                        participant_id=participant,
                        condition="ph_pre_tetris",
                        observation_id=eff_obs,
                        duration_s=duration,
                        band="theta",
                        base_r=0.01,
                        zero_lag_bump=0.06,
                    )
                )

            low_obs = f"ds003816-p1-{duration}-low"
            eff_obs = f"ds003816-p1-{duration}-eff"
            rows.append(
                _paired_row(
                    participant_id="p1",
                    contrast_id="meditation",
                    band="theta",
                    duration_s=duration,
                    low_observation_id=low_obs,
                    effort_observation_id=eff_obs,
                    dataset_id="ds003816",
                )
            )
            curve_files[duration].extend(
                _curve_rows_for_observation(
                    dataset_id="ds003816",
                    participant_id="p1",
                    condition="preresting",
                    observation_id=low_obs,
                    duration_s=duration,
                    band="theta",
                    base_r=0.01,
                    zero_lag_bump=0.01,
                )
            )
            curve_files[duration].extend(
                _curve_rows_for_observation(
                    dataset_id="ds003816",
                    participant_id="p1",
                    condition="lkmself",
                    observation_id=eff_obs,
                    duration_s=duration,
                    band="theta",
                    base_r=0.01,
                    zero_lag_bump=0.05,
                )
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path = root / "paired_contrasts.csv"
            fields = list(rows[0].keys())
            with paired_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            curve_inputs: dict[str, Path] = {}
            for duration in EXPECTED_DURATIONS_S:
                path = root / f"confirmatory_cross_correlation_curves_D{duration}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    fields = list(curve_files[duration][0].keys())
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(curve_files[duration])
                curve_inputs[f"curves_d{duration}"] = path
            protocol = root / "protocol_audit.csv"
            with protocol.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("dataset_id", "dataset_role")
                )
                writer.writeheader()
                writer.writerow({"dataset_id": "hiit", "dataset_role": "sensitivity"})
                writer.writerow(
                    {"dataset_id": "ds003816", "dataset_role": "sensitivity"}
                )
            out = root / "figures"
            artifacts = render_figure3(
                {
                    "paired_contrasts": paired_path,
                    "protocol_audit": protocol,
                    **curve_inputs,
                },
                out,
            )
            panel_csv = out / "source_data" / "figure3_panel_c_duration_sensitivity.csv"
            obs_csv = (
                out / "source_data" / "figure3_panel_c_duration_observation_level.csv"
            )
            self.assertTrue(panel_csv.is_file())
            with panel_csv.open(encoding="utf-8") as handle:
                plotted = list(csv.DictReader(handle))
            with obs_csv.open(encoding="utf-8") as handle:
                obs_rows = list(csv.DictReader(handle))

            eligible = [r for r in plotted if r["eligibility_status"] == "eligible"]
            self.assertTrue(eligible)
            self.assertTrue(all(r["dataset_id"] for r in eligible))
            self.assertNotIn("not_computable", {r["eligibility_status"] for r in plotted})

            hiit = [r for r in eligible if r["dataset_id"] == "hiit"]
            self.assertEqual(
                sorted({int(float(r["duration_s"])) for r in hiit}),
                [60, 120, 180, 240],
            )
            for r in hiit:
                duration = int(float(r["duration_s"]))
                self.assertEqual(r["endpoint_name"], EXPECTED_BY_DURATION[duration])
                contract = contract_for_duration(duration)
                self.assertEqual(int(float(r["lag_max_s"])), contract.lag_max_s)
                self.assertEqual(int(float(r["flank_inner_s"])), contract.flank_inner_s)
                self.assertEqual(int(float(r["flank_outer_s"])), contract.flank_outer_s)
                self.assertEqual(
                    str(r["is_standard_zlpi"]).casefold() in {"1", "true"},
                    contract.is_standard_zlpi,
                )

            # No SWPI/MWPI falsely labeled ZLPI.
            for r in eligible:
                if int(float(r["duration_s"])) in {60, 120}:
                    self.assertNotEqual(r["endpoint_name"], ENDPOINT_ZLPI)

            ds816 = [r for r in eligible if r["dataset_id"] == "ds003816"]
            self.assertEqual(
                sorted({int(float(r["duration_s"])) for r in ds816}),
                [60],
            )
            self.assertTrue(
                all(r["endpoint_name"] == ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX for r in ds816)
            )
            self.assertTrue(
                all(r["dataset_id"] != "ds003816" or int(float(r["duration_s"])) == 60 for r in plotted)
            )

            # Observation-level identity matches summary.
            for duration, expected in EXPECTED_BY_DURATION.items():
                obs_hiit = [
                    r
                    for r in obs_rows
                    if r["dataset_id"] == "hiit"
                    and int(float(r["duration_s"])) == duration
                    and r["eligibility_status"] == "eligible"
                ]
                self.assertTrue(obs_hiit)
                self.assertTrue(all(r["endpoint_name"] == expected for r in obs_hiit))

            caption = (out / "figure3_caption.txt").read_text(encoding="utf-8")
            self.assertIn("SWPI was recomputed from 60-second segments", caption)
            self.assertIn("MWPI from 120-second segments", caption)
            self.assertIn("standard ZLPI from 180- and 240-second segments", caption)
            self.assertIn("should not be interpreted as a pure duration effect", caption)
            self.assertNotIn("Mean ΔZLPI", caption)
            self.assertNotIn("providing sufficient lag overlap", caption)

            notes = next(
                p.notes for p in artifacts.panels if p.panel_id == "duration_sensitivity"
            )
            self.assertIn("SWPI at D60", notes)
            self.assertIn("MWPI at D120", notes)
            panel = next(p for p in artifacts.panels if p.panel_id == "duration_sensitivity")
            self.assertEqual(
                panel.title, "Duration-specific proximal coupling sensitivity"
            )


if __name__ == "__main__":
    unittest.main()
