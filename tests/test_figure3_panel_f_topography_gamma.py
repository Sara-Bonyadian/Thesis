"""Integrity tests for Figure 3 Panel F topography / gamma specificity."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ppg_eeg.confirmatory.panel_f_topography_gamma import (
    ECG_PRONE_SET_ID,
    MAP_GAMMA_AFTER,
    MAP_GAMMA_BEFORE,
    MAP_ORDER,
    MAP_REST_ALPHA,
    MAP_TASK_ATTENUATION,
    PANEL_F_DURATION_S,
    PANEL_F_ENDPOINT,
    PANEL_F_FIGURE_TITLE,
    PANEL_F_REPRESENTATION,
    PANEL_F_STEM,
    PanelFResult,
    load_ecg_prone_channels,
    panel_f_caption,
    render_panel_f_figure,
    verify_panel_f_integrity,
)


def _summary_row(map_id: str, channel: str, estimate: float, *, scale: str, L: float) -> dict:
    return {
        "panel_map": map_id,
        "channel": channel,
        "estimate": estimate,
        "lower_ci": float("nan"),
        "upper_ci": float("nan"),
        "n_observations": 2,
        "n_participants": 2,
        "n_datasets": 1,
        "color_scale_group": scale,
        "color_limit_min": -L,
        "color_limit_max": L,
        "aggregation_method": "test",
        "channel_x": 0.0,
        "channel_y": 0.0,
        "channel_z": 0.0,
        "band": "alpha" if "alpha" in map_id else "low_gamma",
    }


class TestFigure3PanelFContracts(unittest.TestCase):
    def test_ecg_prone_yaml_loads_locked_set(self) -> None:
        set_id, channels = load_ecg_prone_channels()
        self.assertEqual(set_id, ECG_PRONE_SET_ID)
        self.assertIn("T7", channels)
        self.assertIn("Fp1", channels)

    def test_verify_rejects_gamma_scale_mismatch_and_nc_zero(self) -> None:
        retained = ["Cz", "Pz"]
        full = ["Cz", "Pz", "T7"]
        summary = []
        for map_id in MAP_ORDER:
            scale = "alpha_shared" if "alpha" in map_id else "gamma_shared"
            L = 0.1 if scale == "alpha_shared" else 0.2
            channels = full if map_id == MAP_GAMMA_BEFORE else retained
            for ch in channels:
                est = 0.05 if ch == "Cz" else (-0.04 if ch == "Pz" else 0.08)
                row = _summary_row(map_id, ch, est, scale=scale, L=L)
                if map_id == MAP_GAMMA_AFTER:
                    row["color_limit_max"] = 0.25  # mismatch
                summary.append(row)
        result = PanelFResult(
            observation_rows=(
                {
                    "panel_map": MAP_GAMMA_AFTER,
                    "observation_id": "o1",
                    "channel": "Cz",
                    "computable": False,
                    "zpli_value": 0.0,
                    "not_computable_reason": "ecg_prone_default_v1_removed",
                },
            ),
            summary_rows=tuple(summary),
            montage_rows=tuple(
                {
                    "channel": ch,
                    "in_alpha_common_montage": ch in retained,
                    "in_gamma_restricted_montage": ch in retained,
                    "in_gamma_full_montage": ch in full,
                    "ecg_prone_default_v1": ch == "T7",
                    "channel_x": 0.0,
                    "channel_y": 0.0,
                    "channel_z": 0.0,
                }
                for ch in full
            ),
            gamma_comparison_rows=(
                {
                    "metric": "retained_channel_value_identity",
                    "value": 0.0,
                    "max_abs_after_minus_before": 0.0,
                    "is_deterministic_identity": True,
                    "bootstrap_used": False,
                },
            ),
            diagnostic_rows=(),
            metadata={
                "analysis_framing": "gamma_sensitivity_to_ecg_prone_channel_exclusion",
                "locked_estimand": {
                    "duration_s": PANEL_F_DURATION_S,
                    "endpoint_name": PANEL_F_ENDPOINT,
                    "power_representation": PANEL_F_REPRESENTATION,
                    "task_attenuation_definition": "ZLPI_task - ZLPI_rest (negative = attenuation)",
                },
            },
        )
        failures = verify_panel_f_integrity(result)
        self.assertTrue(any("color limits" in f for f in failures))
        self.assertTrue(any("numerical zero" in f for f in failures))

    def test_verify_rejects_bootstrapped_identity(self) -> None:
        retained = ["Cz"]
        full = ["Cz", "T7"]
        summary = []
        for map_id in MAP_ORDER:
            scale = "alpha_shared" if "alpha" in map_id else "gamma_shared"
            L = 0.1
            channels = full if map_id == MAP_GAMMA_BEFORE else retained
            for ch in channels:
                summary.append(_summary_row(map_id, ch, 0.01, scale=scale, L=L))
        result = PanelFResult(
            observation_rows=(),
            summary_rows=tuple(summary),
            montage_rows=tuple(
                {
                    "channel": ch,
                    "in_alpha_common_montage": ch in retained,
                    "in_gamma_restricted_montage": ch in retained,
                    "in_gamma_full_montage": ch in full,
                    "ecg_prone_default_v1": ch == "T7",
                }
                for ch in full
            ),
            gamma_comparison_rows=(
                {
                    "metric": "retained_channel_value_identity",
                    "value": 0.0,
                    "is_deterministic_identity": True,
                    "bootstrap_used": True,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                },
            ),
            diagnostic_rows=(),
            metadata={
                "analysis_framing": "gamma_sensitivity_to_ecg_prone_channel_exclusion",
                "locked_estimand": {
                    "task_attenuation_definition": "ZLPI_task - ZLPI_rest (negative = attenuation)"
                },
            },
        )
        failures = verify_panel_f_integrity(result)
        self.assertTrue(any("bootstrapped" in f for f in failures))
        self.assertTrue(any("confidence interval" in f for f in failures))

    def test_caption_names_exclusion_sensitivity_and_unavailable(self) -> None:
        result = PanelFResult(
            observation_rows=(),
            summary_rows=(),
            montage_rows=(),
            gamma_comparison_rows=(),
            diagnostic_rows=(),
            metadata={
                "n_alpha_common_channels": 53,
                "n_gamma_full_channels": 63,
                "n_retained_channels": 53,
                "n_gamma_paired_observations": 77,
                "locked_estimand": {
                    "task_attenuation_definition": "ZLPI_task - ZLPI_rest (negative = attenuation)"
                },
            },
        )
        caption = panel_f_caption(result)
        self.assertTrue(caption.startswith(PANEL_F_FIGURE_TITLE))
        self.assertIn(ECG_PRONE_SET_ID, caption)
        self.assertIn("restricted-montage sensitivity", caption)
        self.assertIn("deterministic implementation identity", caption)
        self.assertIn("artifact-indeterminate", caption)
        self.assertIn("ICA", caption)
        self.assertNotIn("available channel-level artifact controls", caption)
        self.assertNotIn("before and after available", caption)


@unittest.skipUnless(
    Path(
        "derivatives/confirmatory_temporal_coupling/sensitivity/hiit/C7/publish/"
        "paired_contrasts.csv"
    ).is_file()
    and Path(
        "derivatives/confirmatory_temporal_coupling/sensitivity/hiit/C7/figures/source_data/"
        f"{PANEL_F_STEM}_channel_zlpi_cache.csv"
    ).is_file(),
    "HIIT C7 Panel F cache not present",
)
class TestFigure3PanelFHIITCached(unittest.TestCase):
    def test_hiit_montage_sensitivity_contracts(self) -> None:
        from ppg_eeg.confirmatory.figures import resolve_reporting_inputs, read_csv_rows
        from ppg_eeg.confirmatory.panel_f_topography_gamma import compute_panel_f_topography

        root = Path("derivatives/confirmatory_temporal_coupling/sensitivity/hiit/C7")
        result = compute_panel_f_topography(
            confirmatory_root=root,
            paired_rows=read_csv_rows(resolve_reporting_inputs(root).get("paired_contrasts")),
            force_recompute_channel_zlpi=False,
        )
        failures = verify_panel_f_integrity(result)
        self.assertEqual(failures, [], failures)

        retained = {
            r["channel"]
            for r in result.montage_rows
            if bool(r.get("in_gamma_restricted_montage"))
        }
        gamma_full = {
            r["channel"] for r in result.montage_rows if bool(r.get("in_gamma_full_montage"))
        }
        self.assertGreaterEqual(len(retained), 20)
        self.assertGreater(len(gamma_full), len(retained))

        by_map = {}
        for row in result.summary_rows:
            by_map.setdefault(row["panel_map"], set()).add(row["channel"])
        self.assertEqual(by_map[MAP_REST_ALPHA], retained)
        self.assertEqual(by_map[MAP_TASK_ATTENUATION], retained)
        self.assertEqual(by_map[MAP_GAMMA_AFTER], retained)
        self.assertEqual(by_map[MAP_GAMMA_BEFORE], gamma_full)
        self.assertNotEqual(by_map[MAP_GAMMA_BEFORE], by_map[MAP_GAMMA_AFTER])

        g_b = [r for r in result.summary_rows if r["panel_map"] == MAP_GAMMA_BEFORE]
        g_a = [r for r in result.summary_rows if r["panel_map"] == MAP_GAMMA_AFTER]
        self.assertEqual(
            {(r["color_limit_min"], r["color_limit_max"]) for r in g_b},
            {(r["color_limit_min"], r["color_limit_max"]) for r in g_a},
        )

        self.assertIn(
            "ZLPI_task - ZLPI_rest",
            str(result.metadata["locked_estimand"]["task_attenuation_definition"]),
        )
        self.assertIn("ecg_prone", str(result.metadata["analysis_framing"]))

        identity = next(
            r
            for r in result.gamma_comparison_rows
            if r["metric"] == "retained_channel_value_identity"
        )
        self.assertTrue(identity["is_deterministic_identity"])
        self.assertFalse(identity["bootstrap_used"])
        self.assertAlmostEqual(float(identity["value"]), 0.0, places=12)
        self.assertNotIn("ci_lower", identity)

        montage_change = next(
            r
            for r in result.gamma_comparison_rows
            if r["metric"] == "paired_restricted_minus_inclusive_mean_signed_zlpi"
        )
        self.assertIn("ci_lower", montage_change)
        self.assertTrue(math.isfinite(float(montage_change["value"])))

        # Display regeneration does not alter estimates
        before = [(r["panel_map"], r["channel"], r["estimate"]) for r in result.summary_rows]
        with TemporaryDirectory() as tmp:
            paths = render_panel_f_figure(result, Path(tmp), include_internal_qc=False)
            caption = Path(paths["caption"]).read_text(encoding="utf-8")
        after = [(r["panel_map"], r["channel"], r["estimate"]) for r in result.summary_rows]
        self.assertEqual(before, after)
        self.assertIn("artifact-indeterminate", caption)
        self.assertNotIn("[0, 0]", caption)

        for row in result.observation_rows:
            if row.get("duration_s") not in ("", None):
                self.assertEqual(int(float(row["duration_s"])), PANEL_F_DURATION_S)
            if row.get("endpoint_name"):
                self.assertEqual(row["endpoint_name"], PANEL_F_ENDPOINT)
            if row.get("power_representation"):
                self.assertEqual(row["power_representation"], PANEL_F_REPRESENTATION)


if __name__ == "__main__":
    unittest.main()
