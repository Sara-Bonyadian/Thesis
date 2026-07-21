"""Display-only combined HIIT sensitivity row for alpha forest panels."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import (
    PRIMARY_REPRESENTATION,
    read_csv_rows,
    render_figure1,
    render_figure2,
)
from ppg_eeg.confirmatory.forest_display import (
    HIIT_FOREST_DISPLAY_LABEL,
    ROW_TYPE_POOLED,
    ROW_TYPE_PRIMARY,
    ROW_TYPE_SENSITIVITY_DISPLAY,
    hiit_session_sensitivity_forest_rows,
    student_t_effect_summary,
)
from ppg_eeg.confirmatory.inference import (
    META_EXCLUDED_DATASETS,
    PRIMARY_META_CONTRASTS,
    _enters_primary_meta,
    run_meta_analysis,
)


def _paired(
    *,
    participant_id: str,
    contrast_id: str,
    delta: float,
    band: str = "alpha",
) -> dict[str, object]:
    return {
        "dataset_id": "hiit",
        "contrast_id": contrast_id,
        "participant_id": participant_id,
        "session_id": contrast_id.split("_", 1)[0],
        "endpoint_name": ENDPOINT_ZLPI,
        "duration_s": 240,
        "band": band,
        "power_representation": PRIMARY_REPRESENTATION,
        "contrast_eligible": True,
        "delta_endpoint_index": delta,
    }


class HiitSensitivityForestDisplayTests(unittest.TestCase):
    def test_participant_mean_of_available_then_group_t_ci(self) -> None:
        # P01: mean(0.10, 0.30, 0.40, 0.20) = 0.25
        # P02: mean(-0.10, 0.10, 0.00, 0.20) = 0.05
        rows = [
            _paired(participant_id="01", contrast_id="ph_pre_rest__tetris", delta=0.10),
            _paired(participant_id="01", contrast_id="ph_post_rest__tetris", delta=0.30),
            _paired(participant_id="01", contrast_id="ps_pre_rest__tetris", delta=0.40),
            _paired(participant_id="01", contrast_id="ps_post_rest__tetris", delta=0.20),
            _paired(participant_id="02", contrast_id="ph_pre_rest__tetris", delta=-0.10),
            _paired(participant_id="02", contrast_id="ph_post_rest__tetris", delta=0.10),
            _paired(participant_id="02", contrast_id="ps_pre_rest__tetris", delta=0.00),
            _paired(participant_id="02", contrast_id="ps_post_rest__tetris", delta=0.20),
        ]
        out = hiit_session_sensitivity_forest_rows(rows)
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["dataset_id"], "hiit")
        expected = student_t_effect_summary([0.25, 0.05])
        self.assertEqual(row["n_pairs"], 2)
        self.assertEqual(row["n_participants"], 2)
        self.assertEqual(row["n_participant_sessions"], 4)  # 2 participants × PH+PS
        self.assertAlmostEqual(
            float(row["effect_mean"]), float(expected["effect_mean"]), places=12
        )
        self.assertAlmostEqual(float(row["ci_low"]), float(expected["ci_low"]), places=12)
        self.assertEqual(row["row_type"], ROW_TYPE_SENSITIVITY_DISPLAY)
        self.assertFalse(row["enters_meta"])
        self.assertIn(HIIT_FOREST_DISPLAY_LABEL, str(row["display_label"]))
        self.assertEqual(row["display_label"], HIIT_FOREST_DISPLAY_LABEL)
        self.assertNotIn("unique participants", str(row["display_label"]))

        # Incomplete contrast still contributes at participant level.
        incomplete = [
            _paired(participant_id="03", contrast_id="ph_pre_rest__tetris", delta=1.0),
        ]
        incomplete_out = hiit_session_sensitivity_forest_rows(incomplete)
        self.assertEqual(len(incomplete_out), 1)
        self.assertEqual(incomplete_out[0]["n_participants"], 1)
        self.assertEqual(incomplete_out[0]["n_participant_sessions"], 1)
        self.assertAlmostEqual(float(incomplete_out[0]["effect_mean"]), 1.0, places=12)

    def test_hiit_never_enters_primary_meta_gate(self) -> None:
        self.assertIn("hiit", META_EXCLUDED_DATASETS)
        self.assertFalse(
            any(ds == "hiit" for ds, _contrast in PRIMARY_META_CONTRASTS)
        )
        self.assertFalse(
            _enters_primary_meta(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                is_primary_analysis=True,
                n_pairs=20,
                effect_se=0.05,
            )
        )

    def test_run_meta_analysis_ignores_sensitivity_display_rows(self) -> None:
        primary_effects = [
            {
                "dataset_id": "ds003838",
                "contrast_id": "rest__memory",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "effect_mean": -0.2,
                "effect_var": 0.01,
                "enters_meta": True,
            },
            {
                "dataset_id": "ds006848",
                "contrast_id": "rest__verbalwm",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "effect_mean": -0.1,
                "effect_var": 0.01,
                "enters_meta": True,
            },
        ]
        sensitivity_like = [
            {
                "dataset_id": "hiit",
                "contrast_id": "hiit_combined_ph_ps_pre_post_mean",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "effect_mean": 0.99,
                "effect_var": 0.01,
                "enters_meta": False,
            }
        ]
        meta_before = run_meta_analysis(primary_effects)
        meta_after = run_meta_analysis(primary_effects + sensitivity_like)
        self.assertEqual(len(meta_before), 1)
        self.assertEqual(len(meta_after), 1)
        self.assertAlmostEqual(
            float(meta_before[0]["pooled_effect"]),
            float(meta_after[0]["pooled_effect"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(meta_before[0]["ci_low"]), float(meta_after[0]["ci_low"]), places=12
        )
        self.assertAlmostEqual(
            float(meta_before[0]["ci_high"]), float(meta_after[0]["ci_high"]), places=12
        )
        self.assertAlmostEqual(
            float(meta_before[0]["prediction_low"]),
            float(meta_after[0]["prediction_low"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(meta_before[0]["prediction_high"]),
            float(meta_after[0]["prediction_high"]),
            places=12,
        )
        self.assertEqual(meta_before[0]["n_datasets"], 2)
        self.assertEqual(meta_after[0]["n_datasets"], 2)

    def test_figure_exports_include_sensitivity_and_primary_row_types(self) -> None:
        effects = [
            {
                "dataset_id": "ds003838",
                "contrast_id": "rest__memory",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "is_primary_analysis": True,
                "n_pairs": 4,
                "effect_mean": -0.20,
                "ci_low": -0.30,
                "ci_high": -0.10,
                "enters_meta": True,
            },
            {
                "dataset_id": "ds006848",
                "contrast_id": "rest__verbalwm",
                "duration_s": 240,
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "is_primary_analysis": True,
                "n_pairs": 4,
                "effect_mean": -0.10,
                "ci_low": -0.20,
                "ci_high": 0.00,
                "enters_meta": True,
            },
        ]
        meta = [
            {
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": 240,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "is_primary_analysis": True,
                "pooled_effect": -0.15,
                "ci_low": -0.25,
                "ci_high": -0.05,
                "prediction_low": -0.40,
                "prediction_high": 0.10,
                "n_datasets": 2,
            }
        ]
        paired = [
            _paired(participant_id="01", contrast_id="ph_pre_rest__tetris", delta=0.10),
            _paired(participant_id="01", contrast_id="ph_post_rest__tetris", delta=0.30),
            _paired(participant_id="01", contrast_id="ps_pre_rest__tetris", delta=0.40),
            _paired(participant_id="01", contrast_id="ps_post_rest__tetris", delta=0.20),
            _paired(participant_id="02", contrast_id="ph_pre_rest__tetris", delta=-0.10),
            _paired(participant_id="02", contrast_id="ph_post_rest__tetris", delta=0.10),
            _paired(participant_id="02", contrast_id="ps_pre_rest__tetris", delta=0.00),
            _paired(participant_id="02", contrast_id="ps_post_rest__tetris", delta=0.20),
        ]

        def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
            import csv

            if not rows:
                path.write_text("dataset_id\n", encoding="utf-8")
                return
            fields = list(rows[0].keys())
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "dataset_effects.csv", effects)
            _write_csv(root / "meta_analysis_results.csv", meta)
            _write_csv(root / "paired_contrasts.csv", paired)
            _write_csv(root / "protocol_audit.csv", [])
            inputs = {
                "dataset_effects": root / "dataset_effects.csv",
                "meta_analysis": root / "meta_analysis_results.csv",
                "paired_contrasts": root / "paired_contrasts.csv",
                "protocol_audit": root / "protocol_audit.csv",
                "curves_d240": None,
                "endpoints_d240": None,
                "subject_level": None,
                "null_summary": None,
                "peak_params": None,
                "peak_equivalence": None,
                "mixed_model": None,
            }
            out1 = root / "fig1"
            out2 = root / "fig2"
            render_figure1(inputs, out1)
            render_figure2(inputs, out2)
            for stem in (
                out1 / "source_data" / "figure1_panel_e_alpha_replication_forest.csv",
                out2 / "source_data" / "figure2_panel_c_alpha_meta_forest.csv",
            ):
                rows = read_csv_rows(stem)
                types = {r.get("row_type") for r in rows}
                self.assertEqual(
                    types,
                    {
                        ROW_TYPE_PRIMARY,
                        ROW_TYPE_SENSITIVITY_DISPLAY,
                        ROW_TYPE_POOLED,
                    },
                )
                sens = [r for r in rows if r.get("row_type") == ROW_TYPE_SENSITIVITY_DISPLAY]
                self.assertEqual(len(sens), 1)
                self.assertEqual(sens[0].get("dataset_id"), "hiit")
                self.assertTrue(
                    all(
                        r.get("enters_meta", "").lower() in {"false", "0", ""}
                        for r in sens
                    )
                )
                pooled = next(r for r in rows if r.get("row_type") == ROW_TYPE_POOLED)
                self.assertAlmostEqual(float(pooled["effect_mean"]), -0.15, places=12)
            caption1 = (out1 / "figure1_caption.txt").read_text(encoding="utf-8")
            self.assertIn("excluded from the pooled random-effects", caption1)
            self.assertIn("combined within participant", caption1)
            caption2 = (out2 / "figure2_caption.txt").read_text(encoding="utf-8")
            self.assertIn("excluded from the pooled random-effects", caption2)


if __name__ == "__main__":
    unittest.main()
