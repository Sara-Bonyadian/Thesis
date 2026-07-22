"""Figure 1 Panel D: combined HIIT sensitivity heatmap row."""

from __future__ import annotations

import unittest

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figure1_panels import (
    subject_level_mean_zlpi_cells,
    surrogate_significance_marks,
)
from ppg_eeg.confirmatory.figures import PRIMARY_REPRESENTATION
from ppg_eeg.confirmatory.null_delta_inference import PRIMARY_NULL_TYPE


def _subject(
    *,
    participant_id: str,
    condition: str,
    band: str,
    zlpi: float,
    dataset_id: str = "hiit",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": condition.split("_", 1)[0],
        "condition": condition,
        "band": band,
        "endpoint_name": ENDPOINT_ZLPI,
        "duration_s": 240,
        "power_representation": PRIMARY_REPRESENTATION,
        "endpoint_eligible": True,
        "endpoint_index": zlpi,
    }


class Figure1PanelDHiitCombinedHeatmapTests(unittest.TestCase):
    def test_hiit_session_subject_units_then_group_mean(self) -> None:
        rows = [
            # 01_ph: mean(0.10, 0.30) = 0.20; 01_ps: mean(0.40, 0.20) = 0.30
            _subject(participant_id="01", condition="ph_pre_rest", band="alpha", zlpi=0.10),
            _subject(participant_id="01", condition="ph_post_rest", band="alpha", zlpi=0.30),
            _subject(participant_id="01", condition="ps_pre_rest", band="alpha", zlpi=0.40),
            _subject(participant_id="01", condition="ps_post_rest", band="alpha", zlpi=0.20),
            # 02_ph: mean(-0.10, 0.10) = 0.00; 02_ps: mean(0.00, 0.20) = 0.10
            _subject(participant_id="02", condition="ph_pre_rest", band="alpha", zlpi=-0.10),
            _subject(participant_id="02", condition="ph_post_rest", band="alpha", zlpi=0.10),
            _subject(participant_id="02", condition="ps_pre_rest", band="alpha", zlpi=0.00),
            _subject(participant_id="02", condition="ps_post_rest", band="alpha", zlpi=0.20),
            # Effort conditions must not enter Panel D absolute ZLPI cells.
            _subject(participant_id="01", condition="ph_pre_tetris", band="alpha", zlpi=9.0),
            # Incomplete session uses available observations only.
            _subject(participant_id="03", condition="ph_pre_rest", band="alpha", zlpi=1.0),
            # Primary dataset unchanged path.
            _subject(
                participant_id="S1",
                condition="rest",
                band="alpha",
                zlpi=-0.25,
                dataset_id="ds003838",
            ),
            _subject(
                participant_id="S2",
                condition="rest",
                band="alpha",
                zlpi=-0.15,
                dataset_id="ds003838",
            ),
        ]
        cells = subject_level_mean_zlpi_cells(rows)
        by_key = {(c["dataset_id"], c["band"]): c for c in cells}
        self.assertIn(("hiit", "alpha"), by_key)
        self.assertNotIn(("hiit_ph", "alpha"), by_key)
        self.assertNotIn(("hiit_ps", "alpha"), by_key)
        # 5 session units: 0.20, 0.30, 0.00, 0.10, 1.0
        self.assertAlmostEqual(
            float(by_key[("hiit", "alpha")]["mean_zlpi"]),
            (0.20 + 0.30 + 0.00 + 0.10 + 1.0) / 5.0,
            places=12,
        )
        self.assertEqual(int(by_key[("hiit", "alpha")]["n_participants"]), 5)
        self.assertEqual(int(by_key[("hiit", "alpha")]["n_participant_sessions"]), 5)
        self.assertEqual(
            by_key[("hiit", "alpha")]["aggregation"],
            "session_subject_mean_of_available_low_demand_zlpi",
        )
        self.assertEqual(by_key[("hiit", "alpha")]["display_label"], "HIIT")
        self.assertEqual(by_key[("hiit", "alpha")]["dataset_role"], "sensitivity")
        self.assertAlmostEqual(
            float(by_key[("ds003838", "alpha")]["mean_zlpi"]), -0.20, places=12
        )
        self.assertEqual(int(by_key[("ds003838", "alpha")]["n_participants"]), 2)

    def test_hiit_surrogate_marks_are_combined(self) -> None:
        null_rows = [
            {
                "dataset_id": "hiit",
                "condition": "ph_pre_rest",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "theta",
                "power_representation": PRIMARY_REPRESENTATION,
                "null_type": PRIMARY_NULL_TYPE,
                "median_empirical_p": 0.01,
            },
            {
                "dataset_id": "hiit",
                "condition": "ph_post_rest",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "theta",
                "power_representation": PRIMARY_REPRESENTATION,
                "null_type": PRIMARY_NULL_TYPE,
                "median_empirical_p": 0.02,
            },
            {
                "dataset_id": "hiit",
                "condition": "ps_pre_rest",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "theta",
                "power_representation": PRIMARY_REPRESENTATION,
                "null_type": PRIMARY_NULL_TYPE,
                "median_empirical_p": 0.40,
            },
            {
                "dataset_id": "hiit",
                "condition": "ps_post_rest",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "theta",
                "power_representation": PRIMARY_REPRESENTATION,
                "null_type": PRIMARY_NULL_TYPE,
                "median_empirical_p": 0.50,
            },
        ]
        marks = surrogate_significance_marks(null_rows)
        # Median of [0.01, 0.02, 0.40, 0.50] = 0.21 → not significant
        self.assertFalse(marks[("hiit", "theta")])
        self.assertNotIn(("hiit_ph", "theta"), marks)
        self.assertNotIn(("hiit_ps", "theta"), marks)


if __name__ == "__main__":
    unittest.main()
