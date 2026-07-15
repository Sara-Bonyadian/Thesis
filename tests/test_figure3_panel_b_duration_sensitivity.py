"""Figure 3 Panel B duration-sensitivity guards."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.artifact_controls import (
    duration_sensitivity_effects,
    participant_duration_sensitivity_effects,
)
from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
)
from ppg_eeg.confirmatory.figures import (
    FIGURE3_PANEL_B_ENCODING_NOTE,
    render_figure3,
)
from ppg_eeg.confirmatory.harmonize import CENTER_SELECTION, build_duration_segments
from ppg_eeg.confirmatory.harmonize import ContiguousBlock


def _paired_row(
    *,
    participant_id: str,
    contrast_id: str,
    band: str,
    duration_s: int,
    endpoint_name: str,
    delta: float,
    dataset_id: str = "hiit",
    eligible: bool = True,
    representation: str = "absolute_log10",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "contrast_id": contrast_id,
        "band": band,
        "duration_s": duration_s,
        "endpoint_name": endpoint_name,
        "power_representation": representation,
        "delta_endpoint_index": delta,
        "contrast_eligible": eligible,
    }


class TestNestedDurationWindows(unittest.TestCase):
    def test_midpoint_nested_windows_are_subsets(self) -> None:
        self.assertEqual(CENTER_SELECTION, "midpoint_of_longest_common_support")
        times = tuple(float(t) for t in range(0, 300))
        block = ContiguousBlock(start_s=0.0, end_s=299.0, time_s=times)
        _center, segments = build_duration_segments(block)
        sets = {d: set(segments[d].time_s) for d in (240, 180, 120, 60)}
        self.assertTrue(sets[60] <= sets[120] <= sets[180] <= sets[240])
        for duration in (240, 180, 120, 60):
            self.assertTrue(segments[duration].eligible)
            self.assertEqual(segments[duration].n_samples, duration)


class TestParticipantDurationEstimand(unittest.TestCase):
    def test_equal_participant_weight_and_no_endpoint_mix(self) -> None:
        rows = []
        # Participant A: 3 contrasts at 240; B: 1 contrast — equal weight.
        for i in range(3):
            rows.append(
                _paired_row(
                    participant_id="A",
                    contrast_id=f"c{i}",
                    band="theta",
                    duration_s=240,
                    endpoint_name=ENDPOINT_ZLPI,
                    delta=1.0,
                )
            )
        rows.append(
            _paired_row(
                participant_id="B",
                contrast_id="c0",
                band="theta",
                duration_s=240,
                endpoint_name=ENDPOINT_ZLPI,
                delta=0.0,
            )
        )
        # Wrong representation / band / endpoint / ineligible must not enter.
        rows.append(
            _paired_row(
                participant_id="A",
                contrast_id="c9",
                band="theta",
                duration_s=240,
                endpoint_name=ENDPOINT_ZLPI,
                delta=9.0,
                representation="relative",
            )
        )
        rows.append(
            _paired_row(
                participant_id="A",
                contrast_id="c8",
                band="alpha",
                duration_s=240,
                endpoint_name=ENDPOINT_ZLPI,
                delta=9.0,
            )
        )
        rows.append(
            _paired_row(
                participant_id="A",
                contrast_id="c7",
                band="theta",
                duration_s=120,
                endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                delta=9.0,
            )
        )
        rows.append(
            _paired_row(
                participant_id="C",
                contrast_id="c0",
                band="theta",
                duration_s=240,
                endpoint_name=ENDPOINT_ZLPI,
                delta=0.5,
                eligible=False,
            )
        )
        summary, participants = participant_duration_sensitivity_effects(rows)
        theta240 = [
            r
            for r in summary
            if r["duration_s"] == 240 and r["band"] == "theta" and r["endpoint_name"] == ENDPOINT_ZLPI
        ]
        self.assertEqual(len(theta240), 1)
        self.assertEqual(theta240[0]["n"], 2)
        self.assertAlmostEqual(float(theta240[0]["effect_estimate"]), 0.5, places=12)
        part_theta = [
            r
            for r in participants
            if r["duration_s"] == 240 and r["band"] == "theta"
        ]
        self.assertEqual({r["participant_id"] for r in part_theta}, {"A", "B"})
        self.assertEqual(
            next(r for r in part_theta if r["participant_id"] == "A")["n_contrasts"], 3
        )

    def test_endpoint_contracts_by_duration(self) -> None:
        rows = []
        for duration, endpoint in (
            (240, ENDPOINT_ZLPI),
            (180, ENDPOINT_ZLPI),
            (120, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX),
            (60, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX),
        ):
            for participant in ("01", "02", "03"):
                rows.append(
                    _paired_row(
                        participant_id=participant,
                        contrast_id="c0",
                        band="theta",
                        duration_s=duration,
                        endpoint_name=endpoint,
                        delta=0.1,
                    )
                )
        summary, _ = participant_duration_sensitivity_effects(rows)
        by_d = {int(r["duration_s"]): r["endpoint_name"] for r in summary if r["band"] == "theta"}
        self.assertEqual(by_d[240], ENDPOINT_ZLPI)
        self.assertEqual(by_d[180], ENDPOINT_ZLPI)
        self.assertEqual(by_d[120], ENDPOINT_MID_WINDOW_PROXIMAL_INDEX)
        self.assertEqual(by_d[60], ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX)

    def test_contrast_cell_table_does_not_inflate_participant_weight_in_panel_summary(
        self,
    ) -> None:
        rows = []
        for participant in ("01", "02"):
            for contrast in ("c1", "c2", "c3", "c4"):
                rows.append(
                    _paired_row(
                        participant_id=participant,
                        contrast_id=contrast,
                        band="theta",
                        duration_s=240,
                        endpoint_name=ENDPOINT_ZLPI,
                        delta=0.2 if participant == "01" else -0.1,
                    )
                )
        cell = duration_sensitivity_effects(rows)
        summary, _ = participant_duration_sensitivity_effects(rows)
        # Contrast cells: one row per contrast×band (n=2 participants inside each).
        self.assertEqual(
            len([r for r in cell if r["band"] == "theta" and r["duration_s"] == 240]), 4
        )
        self.assertTrue(
            all(
                int(r["n"]) == 2
                for r in cell
                if r["band"] == "theta" and r["duration_s"] == 240
            )
        )
        # Panel estimand: n = 2 participants.
        theta = [r for r in summary if r["band"] == "theta" and r["duration_s"] == 240][0]
        self.assertEqual(theta["n"], 2)
        self.assertAlmostEqual(float(theta["effect_estimate"]), 0.05, places=12)


class TestFigure3PanelBRender(unittest.TestCase):
    def test_panel_b_exports_participant_level_and_matches_plot_table(self) -> None:
        rows = []
        for duration, endpoint in (
            (240, ENDPOINT_ZLPI),
            (180, ENDPOINT_ZLPI),
            (120, ENDPOINT_MID_WINDOW_PROXIMAL_INDEX),
            (60, ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX),
        ):
            for participant in ("01", "02", "03"):
                rows.append(
                    _paired_row(
                        participant_id=participant,
                        contrast_id="rest__task",
                        band="theta",
                        duration_s=duration,
                        endpoint_name=endpoint,
                        delta=0.05 * int(participant),
                    )
                )
                rows.append(
                    _paired_row(
                        participant_id=participant,
                        contrast_id="rest__task",
                        band="alpha",
                        duration_s=duration,
                        endpoint_name=endpoint,
                        delta=-0.02,
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
            out = root / "figures"
            artifacts = render_figure3({"paired_contrasts": paired_path}, out)
            panel_csv = out / "source_data" / "figure3_panel_b_duration.csv"
            part_csv = out / "source_data" / "figure3_panel_b_participant_estimates.csv"
            self.assertTrue(panel_csv.is_file())
            self.assertTrue(part_csv.is_file())
            with panel_csv.open(encoding="utf-8") as handle:
                plotted = list(csv.DictReader(handle))
            # 4 durations × 2 bands.
            self.assertEqual(len(plotted), 8)
            self.assertTrue(all(r.get("estimand") == "mean_of_participant_means" for r in plotted))
            self.assertTrue(all(int(float(r["n"])) == 3 for r in plotted))
            self.assertEqual(
                sorted(int(float(r["duration_s"])) for r in plotted if r["band"] == "theta"),
                [60, 120, 180, 240],
            )
            expected, _ = participant_duration_sensitivity_effects(rows)
            by_key = {
                (int(r["duration_s"]), r["band"], r["endpoint_name"]): r for r in expected
            }
            for row in plotted:
                key = (int(float(row["duration_s"])), row["band"], row["endpoint_name"])
                ref = by_key[key]
                self.assertAlmostEqual(
                    float(row["effect_estimate"]), float(ref["effect_estimate"]), places=12
                )
                self.assertAlmostEqual(float(row["ci_low"]), float(ref["ci_low"]), places=12)
            self.assertIn("duration_sensitivity", {p.panel_id for p in artifacts.panels})
            notes = next(p.notes for p in artifacts.panels if p.panel_id == "duration_sensitivity")
            self.assertIn("Color = EEG frequency band", FIGURE3_PANEL_B_ENCODING_NOTE)
            self.assertIn("Marker shape", FIGURE3_PANEL_B_ENCODING_NOTE)
            self.assertIn("Color = EEG frequency band", notes)
            self.assertIn("not imply equivalence", notes)


if __name__ == "__main__":
    unittest.main()
