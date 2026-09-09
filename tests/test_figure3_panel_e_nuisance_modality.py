"""Integrity and visual-layout tests for Figure 3 Panel E."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.panel_e_nuisance_modality import (
    ESTIMABLE_SPEC_ORDER,
    PANEL_E_BAND,
    PANEL_E_DURATION_S,
    PANEL_E_ENDPOINT,
    PANEL_E_FIGURE_TITLE,
    PANEL_E_REPRESENTATION,
    PANEL_E_STEM,
    SENSITIVITY_SPEC_ORDER,
    compute_panel_e_nuisance_modality,
    fit_paired_intercept_cluster_boot,
    render_panel_e_figure,
    verify_panel_e_integrity,
)
from ppg_eeg.confirmatory.panel_e_nuisance_upstream import (
    load_panel_e_result_from_upstream,
    run_confirmatory_panel_e_upstream,
)


# Locked corrected-model exports (HIIT C7) — regeneration must not alter these.
LOCKED_HIIT_ESTIMATES = {
    "baseline": {
        "estimate": 0.010500922058655202,
        "ci_lower": -0.037188899970962815,
        "ci_upper": 0.059875460899979235,
    },
    "delta_mean_hr": {
        "estimate": 0.01017883222836329,
        "ci_lower": -0.0371722775954695,
        "ci_upper": 0.05679752593869082,
        "change_from_baseline": -0.000322089830291911,
    },
    "delta_broadband_power_ex_alpha": {
        "estimate": -0.007089052000421837,
        "ci_lower": -0.06242927631633548,
        "ci_upper": 0.05443894190815517,
        "change_from_baseline": -0.01758997405907704,
    },
    "delta_hr_broadband_ex_alpha": {
        "estimate": -0.006796466150664936,
        "ci_lower": -0.0660921003856284,
        "ci_upper": 0.06109926349841765,
        "change_from_baseline": -0.017297388209320138,
    },
    "delta_broadband_power": {
        "estimate": 0.019183749323494297,
        "ci_lower": -0.027049548449224418,
        "ci_upper": 0.06954083518076754,
    },
    "delta_hr_broadband": {
        "estimate": 0.018704735267138495,
        "ci_lower": -0.028684959933308756,
        "ci_upper": 0.06703084733996235,
    },
}


def _paired_row(
    pid: str, *, delta: float, hr_rest: float, hr_tet: float, bb_rest: float, bb_tet: float
) -> tuple[dict, list[dict]]:
    low = f"obs-{pid}-rest"
    effort = f"obs-{pid}-tetris"
    paired = {
        "dataset_id": "ds_test",
        "contrast_id": "rest__tetris",
        "participant_id": pid,
        "session_id": "single",
        "low_demand_condition": "rest",
        "cognitive_effort_condition": "tetris",
        "duration_s": PANEL_E_DURATION_S,
        "endpoint_name": ENDPOINT_ZLPI,
        "band": PANEL_E_BAND,
        "power_representation": PANEL_E_REPRESENTATION,
        "low_observation_ids": low,
        "effort_observation_ids": effort,
        "delta_endpoint_index": delta,
        "contrast_eligible": True,
    }
    aligned = []
    for obs, hr, bb in ((low, hr_rest, bb_rest), (effort, hr_tet, bb_tet)):
        for t in range(240):
            aligned.append(
                {
                    "observation_id": obs,
                    "time_s": float(t),
                    "hr_bpm": hr,
                    "theta_absolute_log10_power": bb,
                    "alpha_absolute_log10_power": bb + 0.2,
                    "beta_absolute_log10_power": bb - 0.1,
                    "low_gamma_absolute_log10_power": bb,
                }
            )
    return paired, aligned


def _endpoint_rows_from_pairs(pairs: list[dict]) -> list[dict]:
    rows = []
    for i, p in enumerate(pairs):
        low = p["low_observation_ids"]
        effort = p["effort_observation_ids"]
        d = float(p["delta_endpoint_index"])
        rest_y = -0.1 * (i + 1)
        tet_y = rest_y + d
        for obs, y, cond in ((low, rest_y, "rest"), (effort, tet_y, "tetris")):
            rows.append(
                {
                    "dataset_id": "ds_test",
                    "observation_id": obs,
                    "condition": cond,
                    "duration_s": PANEL_E_DURATION_S,
                    "endpoint_name": ENDPOINT_ZLPI,
                    "band": PANEL_E_BAND,
                    "power_representation": PANEL_E_REPRESENTATION,
                    "endpoint_index": y,
                    "eligible": True,
                }
            )
    return rows


class TestFigure3PanelEDeltaNuisance(unittest.TestCase):
    def _toy_result(self):
        paired = []
        aligned = []
        data_audit = []
        dhr_vals = [4.0, 12.0, 6.0, 14.0, 8.0, 10.0]
        dbb_vals = [0.20, -0.10, 0.05, 0.15, -0.05, 0.10]
        for i, pid in enumerate(["p0", "p1", "p2", "p3", "p4", "p5"]):
            dhr = dhr_vals[i]
            dbb = dbb_vals[i]
            delta = -0.05 + 0.015 * dhr + 0.4 * dbb
            p, a = _paired_row(
                pid,
                delta=delta,
                hr_rest=60.0,
                hr_tet=60.0 + dhr,
                bb_rest=1.0,
                bb_tet=1.0 + dbb,
            )
            paired.append(p)
            aligned.extend(a)
            for role in ("rest", "tetris"):
                data_audit.append(
                    {
                        "dataset_id": "ds_test",
                        "observation_id": f"obs-{pid}-{role}",
                        "cardiac_signal_type": "ppg",
                    }
                )
        return compute_panel_e_nuisance_modality(
            paired_rows=paired,
            aligned_rows=aligned,
            data_audit_rows=data_audit,
            endpoint_rows=_endpoint_rows_from_pairs(paired),
        )

    def test_primary_uses_ex_alpha_and_uncentered_deltas(self) -> None:
        result = self._toy_result()
        self.assertEqual(failures := verify_panel_e_integrity(result), [], failures)
        self.assertEqual(
            [r["specification_id"] for r in result.common_sample_rows if r.get("plotted")],
            list(ESTIMABLE_SPEC_ORDER),
        )
        self.assertIn("delta_broadband_power_ex_alpha", ESTIMABLE_SPEC_ORDER)
        self.assertIn("delta_broadband_power", SENSITIVITY_SPEC_ORDER)

        for row in result.observation_rows:
            self.assertAlmostEqual(
                float(row["delta_mean_hr"]),
                float(row["tetris_mean_hr"]) - float(row["rest_mean_hr"]),
                places=10,
            )
            self.assertAlmostEqual(
                float(row["delta_broadband_power"]),
                float(row["tetris_broadband_power"]) - float(row["rest_broadband_power"]),
                places=10,
            )
            self.assertAlmostEqual(
                float(row["delta_broadband_power_ex_alpha"]),
                float(row["tetris_broadband_power_ex_alpha"])
                - float(row["rest_broadband_power_ex_alpha"]),
                places=10,
            )

        for row in result.common_sample_rows:
            self.assertFalse(bool(row.get("predictors_centered")))
            self.assertEqual(
                row.get("zero_nuisance_interpretation"),
                "no low-demand–high-demand nuisance change",
            )
            if row["specification_id"] in ESTIMABLE_SPEC_ORDER:
                self.assertEqual(
                    int(row.get("n_specification_usable")),
                    int(row.get("n_observations")),
                )
                self.assertFalse(bool(row.get("composition_differs_from_baseline")))

        base = next(r for r in result.common_sample_rows if r["specification_id"] == "baseline")
        y = [float(r["delta_endpoint_index"]) for r in result.observation_rows]
        self.assertAlmostEqual(float(base["estimate"]), float(np.mean(y)), places=10)

        hr = next(r for r in result.common_sample_rows if r["specification_id"] == "delta_mean_hr")
        self.assertGreater(abs(float(hr["estimate"]) - float(base["estimate"])), 1e-8)
        self.assertEqual(
            hr.get("change_ci_method"),
            "paired_cluster_bootstrap_coefficient_difference",
        )

        for row in result.specification_rows:
            if row["specification_id"] in SENSITIVITY_SPEC_ORDER:
                self.assertFalse(bool(row.get("plotted")))
                self.assertIn("sensitivity only", str(row.get("circularity_warning", "")).casefold())
                self.assertIn("not an independent", str(row.get("circularity_warning", "")).casefold())

    def test_uncentered_zero_means_no_change(self) -> None:
        rows = []
        for i, pid in enumerate(["a", "b", "c", "d", "e", "f"]):
            dhr = float(i - 2)
            rows.append(
                {
                    "participant_id": pid,
                    "delta_endpoint_index": 0.1 + 0.02 * dhr,
                    "delta_mean_hr": dhr,
                }
            )
        fit = fit_paired_intercept_cluster_boot(rows, ("delta_mean_hr",), seed=1, center=False)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(float(fit["estimate"]), 0.1, places=8)

    def test_visual_layout_and_display_only_regeneration(self) -> None:
        result = self._toy_result()
        before = [
            (r["specification_id"], r["estimate"], r["ci_lower"], r["ci_upper"], r.get("change_from_baseline"))
            for r in result.specification_rows
        ]

        captured: dict[str, object] = {}

        def _capture_save(fig, output_dir, stem, **kwargs):
            captured["fig"] = fig
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            pdf = out / f"{stem}.pdf"
            svg = out / f"{stem}.svg"
            png = out / f"{stem}.png"
            fig.savefig(pdf)
            fig.savefig(svg)
            fig.savefig(png, dpi=120)
            return pdf, svg, png

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with mock.patch(
                "ppg_eeg.confirmatory.figures.save_figure_trio",
                side_effect=_capture_save,
            ):
                paths = render_panel_e_figure(result, out, include_internal_qc=True)

            after = [
                (
                    r["specification_id"],
                    r["estimate"],
                    r["ci_lower"],
                    r["ci_upper"],
                    r.get("change_from_baseline"),
                )
                for r in result.specification_rows
            ]
            self.assertEqual(before, after)

            caption = Path(paths["caption"]).read_text(encoding="utf-8")
            self.assertEqual(caption.splitlines()[0], PANEL_E_FIGURE_TITLE)
            self.assertEqual(caption.count("Figure 3E"), 1)
            self.assertEqual(caption.casefold().count("common-sample"), 1)
            self.assertIn("excluding alpha", caption.casefold())
            self.assertIn("sensitivity-only", caption.casefold())
            self.assertIn("not proxied", caption.casefold())

            fig = captured["fig"]
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            if not isinstance(fig.canvas, FigureCanvasAgg):
                FigureCanvasAgg(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legends = [c for c in fig.legends] if hasattr(fig, "legends") else []
            if not legends and fig.legend_ is not None:
                legends = [fig.legend_]
            self.assertTrue(legends)
            legend_bbox = legends[0].get_window_extent(renderer=renderer)

            axes = fig.axes
            self.assertGreaterEqual(len(axes), 3)
            ax_coef, ax_delta, ax_table = axes[0], axes[1], axes[2]
            coef_title_bbox = ax_coef.title.get_window_extent(renderer=renderer)
            delta_title_bbox = ax_delta.title.get_window_extent(renderer=renderer)
            self.assertGreaterEqual(legend_bbox.y0, coef_title_bbox.y1 - 4.0)
            self.assertGreaterEqual(legend_bbox.y0, delta_title_bbox.y1 - 4.0)

            # Sample note once among figure texts
            sample_notes = [
                t for t in fig.texts if "Common sample" in (t.get_text() or "")
            ]
            self.assertEqual(len(sample_notes), 1)

            # No repeated per-row n / Participants annotations on coefficient panels
            for artist in list(ax_coef.texts) + list(ax_delta.texts):
                txt = artist.get_text() or ""
                self.assertNotIn("Participants=", txt)
                self.assertNotRegex(txt, r"^n=\d+")

            xlabel_bbox = ax_coef.xaxis.label.get_window_extent(renderer=renderer)
            section_b = ax_table.title.get_window_extent(renderer=renderer)
            self.assertGreaterEqual(xlabel_bbox.y0, section_b.y1 - 4.0)

            tables = ax_table.tables
            self.assertTrue(tables)
            table = tables[0]
            for (_row, _col), cell in table.get_celld().items():
                tb = cell.get_text().get_window_extent(renderer=renderer)
                self.assertLessEqual(tb.x1, fig.bbox.x1 + 12.0)
                self.assertGreaterEqual(tb.x0, fig.bbox.x0 - 12.0)
                self.assertLessEqual(tb.y1, fig.bbox.y1 + 12.0)
                self.assertGreaterEqual(tb.y0, fig.bbox.y0 - 12.0)

            # Reason column contains full primary phrases
            reason_blob = " ".join(
                cell.get_text().get_text().replace("\n", " ")
                for (_r, c), cell in table.get_celld().items()
                if c == 2
            )
            for phrase in (
                "respiration measure",
                "accelerometer summary",
                "rejected-window",
                "ocular-artifact",
                "low-demand versus high-demand",
                "muscle-artifact",
                "Cardiac modality is constant within this dataset (PPG)",
            ):
                self.assertIn(phrase, reason_blob)

            labels = [r["display_label"] for r in result.common_sample_rows if r.get("plotted")]
            self.assertTrue(any("excluding alpha" in lab for lab in labels))
            self.assertFalse(any("including alpha" in lab for lab in labels))

            for ext in ("png", "pdf", "svg"):
                self.assertTrue(Path(paths[ext]).is_file())
                self.assertGreater(Path(paths[ext]).stat().st_size, 1000)

    def test_c6_upstream_roundtrip_preserves_estimates(self) -> None:
        result = self._toy_result()
        with TemporaryDirectory() as tmp:
            c6 = Path(tmp) / "C6"
            from ppg_eeg.confirmatory.panel_e_nuisance_modality import (
                write_panel_e_upstream_exports,
            )

            write_panel_e_upstream_exports(result, c6)
            loaded = load_panel_e_result_from_upstream(c6)
        before = {
            r["specification_id"]: (
                r["estimate"],
                r["ci_lower"],
                r["ci_upper"],
                r.get("change_from_baseline"),
            )
            for r in result.specification_rows
        }
        after = {
            r["specification_id"]: (
                float(r["estimate"]),
                float(r["ci_lower"]),
                float(r["ci_upper"]),
                float(r.get("change_from_baseline") or 0),
            )
            for r in loaded.specification_rows
        }
        self.assertEqual(before, after)

    def test_c7_render_does_not_refit(self) -> None:
        result = self._toy_result()
        with TemporaryDirectory() as tmp:
            c6 = Path(tmp) / "C6"
            out = Path(tmp) / "figures"
            from ppg_eeg.confirmatory.panel_e_nuisance_modality import (
                write_panel_e_upstream_exports,
            )

            write_panel_e_upstream_exports(result, c6)
            inputs = {
                "panel_e_specifications": c6 / f"{PANEL_E_STEM}_specifications.csv",
                "panel_e_metadata": c6 / f"{PANEL_E_STEM}_metadata.json",
                "panel_e_common_sample": c6 / f"{PANEL_E_STEM}_common_sample.csv",
                "panel_e_availability": c6 / f"{PANEL_E_STEM}_availability.csv",
            }
            with mock.patch(
                "ppg_eeg.confirmatory.panel_e_nuisance_modality.compute_panel_e_nuisance_modality",
                side_effect=AssertionError("C7 must not refit Panel E models"),
            ):
                from ppg_eeg.confirmatory.figures import render_figure3_panel_e

                paths = render_figure3_panel_e(inputs, out, include_internal_qc=False)

            svg_text = Path(paths["svg"]).read_text(encoding="utf-8")
            self.assertIn("Respiration", svg_text)
            self.assertIn("Cardiac modality is constant within this dataset (PPG)", svg_text)


@unittest.skipUnless(
    Path("derivatives/confirmatory_temporal_coupling/sensitivity/hiit/C5/paired_contrasts.csv").is_file(),
    "HIIT C5 confirmatory tables not present",
)
class TestFigure3PanelELockedHIITValues(unittest.TestCase):
    def test_locked_estimates_unchanged(self) -> None:
        from ppg_eeg.confirmatory.figures import read_csv_rows, resolve_reporting_inputs

        root = Path("derivatives/confirmatory_temporal_coupling/sensitivity/hiit")
        c5 = root / "C5"
        if not (c5 / "paired_contrasts.csv").is_file():
            self.skipTest("HIIT C5 paired contrasts not present")
        upstream = run_confirmatory_panel_e_upstream(
            c0_dir=root / "C0",
            c1b_dir=root / "C1b",
            c1c_dir=root / "C1c",
            c3_dir=root / "C3",
            c5_dir=c5,
            output_dir=root / "C6",
        )
        result = upstream.result
        by_id = {r["specification_id"]: r for r in result.specification_rows}
        for sid, locked in LOCKED_HIIT_ESTIMATES.items():
            row = by_id[sid]
            for key, expected in locked.items():
                self.assertAlmostEqual(
                    float(row[key]),
                    float(expected),
                    places=12,
                    msg=f"{sid}.{key}",
                )
        self.assertEqual(
            [r["specification_id"] for r in result.common_sample_rows if r.get("plotted")],
            list(ESTIMABLE_SPEC_ORDER),
        )
        self.assertEqual(int(result.metadata["n_contrasts_common_sample"]), 77)
        self.assertEqual(int(result.metadata["n_participants_common_sample"]), 20)
        self.assertEqual(int(result.metadata["n_baseline_eligible"]), 77)

        loaded = load_panel_e_result_from_upstream(root / "C6")
        self.assertEqual(
            len(loaded.specification_rows),
            len(result.specification_rows),
        )

        # Regenerating visualization must not alter locked numeric exports.
        with TemporaryDirectory() as tmp:
            before = [
                (r["specification_id"], r["estimate"], r["ci_lower"], r["ci_upper"])
                for r in result.specification_rows
            ]
            inputs = resolve_reporting_inputs(root / "C7")
            from ppg_eeg.confirmatory.figures import render_figure3_panel_e

            render_figure3_panel_e(inputs, Path(tmp), include_internal_qc=False)
            after = [
                (r["specification_id"], r["estimate"], r["ci_lower"], r["ci_upper"])
                for r in loaded.specification_rows
            ]
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
