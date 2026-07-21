"""Regression tests for Figure 2 Panel C (PRIMARY_META alpha absolute ΔZLPI)."""

from __future__ import annotations

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
)


def _effect_row(
    *,
    duration_s: int,
    band: str,
    effect_mean: float,
    power_representation: str = PRIMARY_REPRESENTATION,
    dataset_id: str = "ds003838",
    contrast_id: str = "rest__memory",
    enters_meta: bool | None = None,
) -> dict[str, object]:
    if enters_meta is None:
        enters_meta = (
            duration_s == 240 and power_representation == PRIMARY_REPRESENTATION
        )
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
        "enters_meta": enters_meta,
    }


class Figure2PanelCAlphaMetaTests(unittest.TestCase):
    def test_panel_c_uses_alpha_enters_meta_absolute_delta_only(self) -> None:
        effects = [
            _effect_row(duration_s=240, band="alpha", effect_mean=-0.20),
            _effect_row(duration_s=240, band="theta", effect_mean=-0.50),
            _effect_row(
                duration_s=240,
                band="alpha",
                effect_mean=-0.99,
                power_representation="relative",
                enters_meta=False,
            ),
            _effect_row(duration_s=180, band="alpha", effect_mean=-0.11, enters_meta=False),
        ]
        meta = [
            {
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": 240,
                "band": "alpha",
                "power_representation": PRIMARY_REPRESENTATION,
                "is_primary_analysis": True,
                "pooled_effect": -0.18,
                "ci_low": -0.25,
                "ci_high": -0.11,
                "prediction_low": -0.35,
                "prediction_high": -0.02,
                "n_datasets": 1,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            effects_path = out / "dataset_effects.csv"
            meta_path = out / "meta.csv"
            fieldnames = list(effects[0].keys())
            with effects_path.open("w", encoding="utf-8") as handle:
                handle.write(",".join(fieldnames) + "\n")
                for row in effects:
                    handle.write(
                        ",".join(str(row[name]) for name in fieldnames) + "\n"
                    )
            mfields = list(meta[0].keys())
            with meta_path.open("w", encoding="utf-8") as handle:
                handle.write(",".join(mfields) + "\n")
                handle.write(",".join(str(meta[0][n]) for n in mfields) + "\n")
            render_figure2(
                {
                    "subject_level": None,
                    "paired_contrasts": None,
                    "curves_d240": None,
                    "dataset_effects": effects_path,
                    "meta_analysis": meta_path,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                },
                out,
            )
            panel_c = read_csv_rows(
                out / "source_data" / "figure2_panel_c_alpha_meta_forest.csv"
            )
        bands = {r["band"] for r in panel_c}
        self.assertEqual(bands, {"alpha"})
        study = next(r for r in panel_c if r["dataset_id"] == "ds003838")
        self.assertAlmostEqual(float(study["effect_mean"]), -0.20)
        self.assertNotIn("percent", ",".join(panel_c[0].keys()).casefold())
        self.assertFalse(
            (out / "source_data" / "figure2_panel_c_d180_sensitivity.csv").exists()
        )

    def test_empty_primary_meta_panel_annotated_expected_not_applicable(self) -> None:
        from ppg_eeg.confirmatory.figures import (
            PANEL_STATUS_EXPECTED_NOT_APPLICABLE,
            primary_meta_expected_na_message,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            artifacts = render_figure2(
                {
                    "subject_level": None,
                    "paired_contrasts": None,
                    "curves_d240": None,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                },
                out,
            )
            panel_c = next(
                p for p in artifacts.panels if p.panel_id == "alpha_primary_meta_forest"
            )
            self.assertIn(
                f"panel_status={PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
                panel_c.analysis_keys,
            )
            self.assertIn("Expected not applicable", panel_c.notes)
            self.assertIn(
                "Expected not applicable",
                primary_meta_expected_na_message("No PRIMARY_META alpha study effects."),
            )

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


if __name__ == "__main__":
    unittest.main()
