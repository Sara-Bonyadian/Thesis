"""Regression tests for Figure 2 Panel C (D180 absolute ZLPI sensitivity)."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import (
    PRIMARY_REPRESENTATION,
    read_csv_rows,
    render_figure2,
)
from ppg_eeg.confirmatory.harmonize import (
    CENTER_SELECTION,
    ContiguousBlock,
    build_duration_segments,
    place_centered_segment,
)


def _effect_row(
    *,
    duration_s: int,
    band: str,
    effect_mean: float,
    power_representation: str = PRIMARY_REPRESENTATION,
    dataset_id: str = "hiit",
    contrast_id: str = "ph_pre_rest__tetris",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "contrast_id": contrast_id,
        "duration_s": duration_s,
        "endpoint_name": ENDPOINT_ZLPI,
        "is_standard_zlpi": True,
        "band": band,
        "power_representation": power_representation,
        "is_primary_analysis": duration_s == 240
        and power_representation == PRIMARY_REPRESENTATION,
        "n_pairs": 3,
        "effect_mean": effect_mean,
        "effect_sd": 0.1,
        "effect_se": 0.05,
        "effect_var": 0.0025,
        "t_stat": 1.0,
        "p_value": 0.2,
        "ci_low": effect_mean - 0.1,
        "ci_high": effect_mean + 0.1,
        "enters_meta": duration_s == 240
        and power_representation == PRIMARY_REPRESENTATION,
    }


class Figure2PanelCDurationSensitivityTests(unittest.TestCase):
    def test_panel_c_excludes_non_absolute_representations(self) -> None:
        """Panel C must not mix relative / residualized effects into D180 means."""
        effects = [
            _effect_row(duration_s=180, band="theta", effect_mean=0.50),
            _effect_row(
                duration_s=180,
                band="theta",
                effect_mean=-0.50,
                power_representation="relative",
            ),
            _effect_row(
                duration_s=180,
                band="theta",
                effect_mean=-0.50,
                power_representation="broadband_residualized",
            ),
            _effect_row(duration_s=240, band="theta", effect_mean=0.10),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            effects_path = out / "dataset_effects.csv"
            fieldnames = list(effects[0].keys())
            with effects_path.open("w", encoding="utf-8") as handle:
                handle.write(",".join(fieldnames) + "\n")
                for row in effects:
                    handle.write(
                        ",".join(str(row[name]) for name in fieldnames) + "\n"
                    )
            render_figure2(
                {
                    "subject_level": None,
                    "paired_contrasts": None,
                    "dataset_effects": effects_path,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                },
                out,
            )
            panel_c = read_csv_rows(out / "source_data" / "figure2_panel_c_d180_sensitivity.csv")
        self.assertEqual(len(panel_c), 1)
        self.assertEqual(panel_c[0]["band"], "theta")
        self.assertEqual(int(float(panel_c[0]["duration_s"])), 180)
        self.assertEqual(panel_c[0]["power_representation"], PRIMARY_REPRESENTATION)
        self.assertAlmostEqual(float(panel_c[0]["effect"]), 0.50)
        # Contaminated mean would be (0.5-0.5-0.5)/3 = -0.166...
        contaminated = (0.50 - 0.50 - 0.50) / 3.0
        self.assertNotAlmostEqual(float(panel_c[0]["effect"]), contaminated)

    def test_panel_c_does_not_read_240_columns(self) -> None:
        effects = [
            _effect_row(duration_s=240, band="theta", effect_mean=0.99),
            _effect_row(duration_s=180, band="theta", effect_mean=0.11),
            _effect_row(duration_s=180, band="alpha", effect_mean=0.22),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            effects_path = out / "dataset_effects.csv"
            fieldnames = list(effects[0].keys())
            with effects_path.open("w", encoding="utf-8") as handle:
                handle.write(",".join(fieldnames) + "\n")
                for row in effects:
                    handle.write(
                        ",".join(str(row[name]) for name in fieldnames) + "\n"
                    )
            render_figure2(
                {
                    "subject_level": None,
                    "paired_contrasts": None,
                    "dataset_effects": effects_path,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                },
                out,
            )
            panel_c = read_csv_rows(out / "source_data" / "figure2_panel_c_d180_sensitivity.csv")
        self.assertEqual({int(float(r["duration_s"])) for r in panel_c}, {180})
        by_band = {r["band"]: float(r["effect"]) for r in panel_c}
        self.assertAlmostEqual(by_band["theta"], 0.11)
        self.assertAlmostEqual(by_band["alpha"], 0.22)
        self.assertNotIn(0.99, by_band.values())

    def test_nested_durations_match_intended_sample_counts(self) -> None:
        times = tuple(float(i) for i in range(300))
        block = ContiguousBlock(start_s=0.0, end_s=299.0, time_s=times)
        center, segments = build_duration_segments(block)
        self.assertIsNotNone(center)
        self.assertEqual(segments[240].n_samples, 240)
        self.assertEqual(segments[180].n_samples, 180)
        self.assertEqual(segments[120].n_samples, 120)
        self.assertEqual(segments[60].n_samples, 60)
        self.assertTrue(segments[180].eligible)
        self.assertAlmostEqual(segments[240].center_s, segments[180].center_s)
        self.assertGreaterEqual(segments[180].start_s, segments[240].start_s)
        self.assertLessEqual(segments[180].end_s, segments[240].end_s)
        self.assertEqual(CENTER_SELECTION, "midpoint_of_longest_common_support")

    def test_changing_duration_changes_extracted_length(self) -> None:
        times = tuple(float(i) for i in range(300))
        block = ContiguousBlock(start_s=0.0, end_s=299.0, time_s=times)
        seg240 = place_centered_segment(block, 240, center_s=None)
        seg180 = place_centered_segment(block, 180, center_s=seg240.center_s)
        self.assertEqual(seg240.n_samples, 240)
        self.assertEqual(seg180.n_samples, 180)
        self.assertNotEqual(seg240.n_samples, seg180.n_samples)
        self.assertAlmostEqual(seg240.duration_s, 240.0)
        self.assertAlmostEqual(seg180.duration_s, 180.0)

    def test_paired_keys_one_to_one_for_common_sample_merge(self) -> None:
        """Unique merge keys for duration-matched contrast rows."""
        rows = [
            {
                "dataset_id": "hiit",
                "participant_id": "01",
                "contrast_id": "ph_pre_rest__tetris",
                "band": "theta",
                "duration_s": duration,
                "endpoint_name": ENDPOINT_ZLPI,
                "power_representation": PRIMARY_REPRESENTATION,
                "delta_endpoint_index": 0.1 if duration == 240 else 0.12,
            }
            for duration in (240, 180)
        ]
        keys_240 = []
        keys_180 = []
        for row in rows:
            key = (
                row["dataset_id"],
                row["participant_id"],
                row["contrast_id"],
                str(row["band"]).casefold(),
                row["endpoint_name"],
                row["power_representation"],
            )
            if int(row["duration_s"]) == 240:
                keys_240.append(key)
            else:
                keys_180.append(key)
        self.assertEqual(len(keys_240), len(set(keys_240)))
        self.assertEqual(len(keys_180), len(set(keys_180)))
        self.assertEqual(set(keys_240), set(keys_180))


class DurationContractOverlapTests(unittest.TestCase):
    def test_constant_overlap_matches_contract(self) -> None:
        c240 = __import__(
            "ppg_eeg.confirmatory.duration_contracts", fromlist=["contract_for_duration"]
        ).contract_for_duration(240)
        c180 = __import__(
            "ppg_eeg.confirmatory.duration_contracts", fromlist=["contract_for_duration"]
        ).contract_for_duration(180)
        self.assertEqual(c240.expected_constant_overlap_if_fully_finite, 120)
        self.assertEqual(c180.expected_constant_overlap_if_fully_finite, 60)
        self.assertEqual(c240.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(c180.endpoint_name, ENDPOINT_ZLPI)
        self.assertEqual(c240.lag_max_s, c180.lag_max_s)


if __name__ == "__main__":
    unittest.main()
