"""Figure 2 Panel A HIIT sensitivity: observation-level matched pairs."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.endpoints import fisher_z
from ppg_eeg.confirmatory.figure2_panels import (
    HIIT_CLUSTER_BOOTSTRAP_CI_METHOD,
    HIIT_MATCHED_PAIR_AGGREGATION,
    build_curve_lag_index,
    build_hiit_sensitivity_panel_a_series,
    filter_hiit_sensitivity_paired_rows,
    filter_primary_meta_paired_rows,
    hiit_matched_pair_cluster_bootstrap_ci,
    render_figure2,
)
from ppg_eeg.confirmatory.figures import (
    PANEL_STATUS_EXPECTED_NOT_APPLICABLE,
    PANEL_STATUS_SENSITIVITY_DISPLAY,
    PRIMARY_REPRESENTATION,
    read_csv_rows,
)
from ppg_eeg.confirmatory.forest_display import HIIT_ALL_CONTRASTS
from ppg_eeg.confirmatory.inference import META_EXCLUDED_DATASETS, PRIMARY_META_CONTRASTS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_row(
    *,
    dataset_id: str,
    contrast_id: str,
    participant_id: str,
    session_id: str,
    band: str,
    low_id: str,
    effort_id: str,
    low_has_identifiable_peak: bool = True,
    effort_has_identifiable_peak: bool = True,
    low_peak_center_mu_s: float = 0.5,
    effort_peak_center_mu_s: float = -0.5,
    low_fwhm_s: float = 4.0,
    effort_fwhm_s: float = 5.0,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "contrast_id": contrast_id,
        "band": band,
        "duration_s": 240,
        "endpoint_name": ENDPOINT_ZLPI,
        "power_representation": PRIMARY_REPRESENTATION,
        "contrast_eligible": True,
        "low_observation_ids": low_id,
        "effort_observation_ids": effort_id,
        "delta_endpoint_index": -0.1,
        "low_has_identifiable_peak": low_has_identifiable_peak,
        "effort_has_identifiable_peak": effort_has_identifiable_peak,
        "low_peak_center_mu_s": low_peak_center_mu_s,
        "effort_peak_center_mu_s": effort_peak_center_mu_s,
        "low_fwhm_s": low_fwhm_s,
        "effort_fwhm_s": effort_fwhm_s,
    }


def _curve_rows(
    observation_id: str,
    *,
    band: str = "theta",
    r_by_lag: dict[int, float] | None = None,
) -> list[dict[str, object]]:
    r_by_lag = r_by_lag or {0: 0.3, 1: 0.2, -1: 0.25}
    return [
        {
            "observation_id": observation_id,
            "band": band,
            "lag_s": lag,
            "r": r,
            "duration_s": 240,
            "power_representation": PRIMARY_REPRESENTATION,
            "subject_id": "",
            "condition": "",
            "task": "",
            "dataset_id": "hiit",
        }
        for lag, r in r_by_lag.items()
    ]


class HiitSensitivityFilterTests(unittest.TestCase):
    def test_hiit_not_in_primary_meta(self) -> None:
        self.assertIn("hiit", META_EXCLUDED_DATASETS)
        self.assertFalse(any(ds == "hiit" for ds, _ in PRIMARY_META_CONTRASTS))

    def test_filter_primary_meta_ignores_hiit(self) -> None:
        rows = [
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="hiit-01-ph-pre-rest",
                effort_id="hiit-01-ph-pre-tetris",
            ),
            _paired_row(
                dataset_id="ds003838",
                contrast_id="rest__memory",
                participant_id="s1",
                session_id="single",
                band="theta",
                low_id="ds003838-s1-rest",
                effort_id="ds003838-s1-memory",
            ),
        ]
        meta = filter_primary_meta_paired_rows(rows)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]["dataset_id"], "ds003838")
        hiit = filter_hiit_sensitivity_paired_rows(rows)
        self.assertEqual(len(hiit), 1)


class MatchingIntegrityTests(unittest.TestCase):
    def test_pre_never_cross_matched_to_post(self) -> None:
        paired = [
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="hiit-01-ph-pre-rest",
                effort_id="hiit-01-ph-pre-tetris",
            ),
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_post_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="hiit-01-ph-post-rest",
                effort_id="hiit-01-ph-post-tetris",
            ),
        ]
        curves: list[dict[str, object]] = []
        for oid, r in (
            ("hiit-01-ph-pre-rest", 0.4),
            ("hiit-01-ph-pre-tetris", 0.1),
            ("hiit-01-ph-post-rest", 0.5),
            ("hiit-01-ph-post-tetris", 0.2),
        ):
            curves.extend(_curve_rows(oid, r_by_lag={0: r}))
        series, gaps = build_hiit_sensitivity_panel_a_series(
            paired, build_curve_lag_index(curves)
        )
        self.assertEqual(gaps, [])
        lag0 = [r for r in series if int(r["lag_s"]) == 0]
        self.assertEqual(len(lag0), 2)
        contrasts = {r["contrast_id"] for r in lag0}
        self.assertEqual(
            contrasts, {"ph_pre_rest__tetris", "ph_post_rest__tetris"}
        )
        for r in lag0:
            self.assertEqual(r["cluster_id"], "01_ph")
            self.assertNotEqual(r["contrast_id"], "hiit_combined_ph_ps_pre_post_mean")
            self.assertEqual(r["aggregation"], HIIT_MATCHED_PAIR_AGGREGATION)

    def test_no_pre_post_collapse_both_pairs_in_mean(self) -> None:
        paired = [
            _paired_row(
                dataset_id="hiit",
                contrast_id=c,
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id=f"low-{c}",
                effort_id=f"eff-{c}",
            )
            for c in ("ph_pre_rest__tetris", "ph_post_rest__tetris")
        ]
        curves: list[dict[str, object]] = []
        # Distinct low z via r=0.5 and r=0.0 → different fisher_z; mean of pairs
        curves.extend(_curve_rows("low-ph_pre_rest__tetris", r_by_lag={0: 0.5}))
        curves.extend(_curve_rows("eff-ph_pre_rest__tetris", r_by_lag={0: 0.0}))
        curves.extend(_curve_rows("low-ph_post_rest__tetris", r_by_lag={0: 0.0}))
        curves.extend(_curve_rows("eff-ph_post_rest__tetris", r_by_lag={0: 0.0}))
        series, _ = build_hiit_sensitivity_panel_a_series(
            paired, build_curve_lag_index(curves)
        )
        ci = hiit_matched_pair_cluster_bootstrap_ci(
            series, band="theta", value_field="z_low", n_bootstrap=50
        )
        self.assertEqual(int(ci[0]["n_matched_pairs"]), 2)
        self.assertEqual(int(ci[0]["n_session_clusters"]), 1)
        # Mean of fisher_z(0.5) and fisher_z(0.0) — not a single collapsed value
        expected = 0.5 * (fisher_z(0.5) + fisher_z(0.0))
        self.assertAlmostEqual(float(ci[0]["mean"]), expected, places=10)

    def test_ph_and_ps_remain_separate_clusters(self) -> None:
        paired = [
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="alpha",
                low_id="hiit-01-ph-pre-rest",
                effort_id="hiit-01-ph-pre-tetris",
            ),
            _paired_row(
                dataset_id="hiit",
                contrast_id="ps_pre_rest__tetris",
                participant_id="01",
                session_id="ps",
                band="alpha",
                low_id="hiit-01-ps-pre-rest",
                effort_id="hiit-01-ps-pre-tetris",
            ),
        ]
        curves: list[dict[str, object]] = []
        for oid in (
            "hiit-01-ph-pre-rest",
            "hiit-01-ph-pre-tetris",
            "hiit-01-ps-pre-rest",
            "hiit-01-ps-pre-tetris",
        ):
            curves.extend(_curve_rows(oid, band="alpha", r_by_lag={0: 0.2}))
        series, _ = build_hiit_sensitivity_panel_a_series(
            paired, build_curve_lag_index(curves)
        )
        clusters = {r["cluster_id"] for r in series}
        self.assertEqual(clusters, {"01_ph", "01_ps"})
        ci = hiit_matched_pair_cluster_bootstrap_ci(
            series, band="alpha", value_field="z_low", n_bootstrap=20
        )
        self.assertEqual(int(ci[0]["n_matched_pairs"]), 2)
        self.assertEqual(int(ci[0]["n_session_clusters"]), 2)


class ClusterBootstrapTests(unittest.TestCase):
    def test_resamples_clusters_not_pairs_as_iid(self) -> None:
        """Two pairs in one cluster: n_session_clusters=1, n_matched_pairs=2."""
        series = []
        for contrast, z in (("ph_pre_rest__tetris", 0.1), ("ph_post_rest__tetris", 0.9)):
            series.append(
                {
                    "dataset_id": "hiit",
                    "participant_id": "01",
                    "session_id": "ph",
                    "contrast_id": contrast,
                    "cluster_id": "01_ph",
                    "pair_id": f"01_ph::{contrast}",
                    "band": "theta",
                    "lag_s": 0,
                    "z_low": z,
                    "z_effort": 0.0,
                }
            )
        rows = hiit_matched_pair_cluster_bootstrap_ci(
            series, band="theta", value_field="z_low", n_bootstrap=100, seed=1
        )
        self.assertEqual(int(rows[0]["n_matched_pairs"]), 2)
        self.assertEqual(int(rows[0]["n_session_clusters"]), 1)
        self.assertEqual(rows[0]["ci_method"], HIIT_CLUSTER_BOOTSTRAP_CI_METHOD)
        # With one cluster, every bootstrap draw is the same mean of both pairs
        self.assertAlmostEqual(float(rows[0]["mean"]), 0.5, places=12)
        # Single cluster → every bootstrap replicate is identical
        self.assertAlmostEqual(float(rows[0]["ci_low"]), 0.5, places=12)
        self.assertAlmostEqual(float(rows[0]["ci_high"]), 0.5, places=12)

    def test_two_clusters_vary_ci(self) -> None:
        series = [
            {
                "dataset_id": "hiit",
                "cluster_id": "01_ph",
                "pair_id": "01_ph::ph_pre_rest__tetris",
                "contrast_id": "ph_pre_rest__tetris",
                "band": "theta",
                "lag_s": 0,
                "z_low": 0.0,
                "z_effort": 0.0,
            },
            {
                "dataset_id": "hiit",
                "cluster_id": "01_ps",
                "pair_id": "01_ps::ps_pre_rest__tetris",
                "contrast_id": "ps_pre_rest__tetris",
                "band": "theta",
                "lag_s": 0,
                "z_low": 1.0,
                "z_effort": 0.0,
            },
        ]
        rows = hiit_matched_pair_cluster_bootstrap_ci(
            series, band="theta", value_field="z_low", n_bootstrap=500, seed=7
        )
        self.assertEqual(int(rows[0]["n_session_clusters"]), 2)
        self.assertEqual(int(rows[0]["n_matched_pairs"]), 2)
        self.assertAlmostEqual(float(rows[0]["mean"]), 0.5, places=12)
        # Cluster resampling of {0,1} yields variable means → CI width > 0
        self.assertLess(float(rows[0]["ci_low"]), float(rows[0]["ci_high"]))


class Figure2PanelAHiitFallbackRenderTests(unittest.TestCase):
    def test_primary_meta_path_unchanged_when_meta_pairs_present(self) -> None:
        paired = [
            _paired_row(
                dataset_id="ds003838",
                contrast_id="rest__memory",
                participant_id="s1",
                session_id="single",
                band="theta",
                low_id="obs-low",
                effort_id="obs-effort",
            ),
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="hiit-low",
                effort_id="hiit-effort",
            ),
        ]
        curves = _curve_rows("obs-low", r_by_lag={0: 0.5}) + _curve_rows(
            "obs-effort", r_by_lag={0: 0.1}
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path = root / "paired.csv"
            curves_path = root / "curves.csv"
            _write_csv(paired_path, paired)
            _write_csv(curves_path, curves)
            out = root / "figures"
            artifacts = render_figure2(
                {
                    "paired_contrasts": paired_path,
                    "curves_d240": curves_path,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                    "subject_level": None,
                },
                out,
            )
            panel_a = next(
                p for p in artifacts.panels if p.panel_id == "matched_low_effort_curves"
            )
            self.assertIn("cohort=PRIMARY_META_C5_pairs", panel_a.analysis_keys)
            self.assertNotIn(
                f"panel_status={PANEL_STATUS_SENSITIVITY_DISPLAY}",
                panel_a.analysis_keys,
            )
            curves_out = read_csv_rows(
                out / "source_data" / "figure2_panel_a_matched_lag_curves.csv"
            )
            self.assertTrue(curves_out)
            self.assertIn("n_participants", curves_out[0])
            self.assertNotIn("n_matched_pairs", curves_out[0])

    def test_hiit_only_observation_level_export(self) -> None:
        paired = []
        curves = []
        for contrast, session, low, effort, z_r in (
            ("ph_pre_rest__tetris", "ph", "hiit-01-ph-pre-rest", "hiit-01-ph-pre-tetris", 0.4),
            ("ph_post_rest__tetris", "ph", "hiit-01-ph-post-rest", "hiit-01-ph-post-tetris", 0.5),
            ("ps_pre_rest__tetris", "ps", "hiit-01-ps-pre-rest", "hiit-01-ps-pre-tetris", 0.35),
        ):
            for band in ("theta", "alpha", "beta", "low_gamma"):
                paired.append(
                    _paired_row(
                        dataset_id="hiit",
                        contrast_id=contrast,
                        participant_id="01",
                        session_id=session,
                        band=band,
                        low_id=low,
                        effort_id=effort,
                    )
                )
            for band in ("theta", "alpha", "beta", "low_gamma"):
                curves.extend(_curve_rows(low, band=band, r_by_lag={0: z_r}))
                curves.extend(_curve_rows(effort, band=band, r_by_lag={0: z_r / 2}))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path = root / "paired.csv"
            curves_path = root / "curves.csv"
            _write_csv(paired_path, paired)
            _write_csv(curves_path, curves)
            out = root / "figures"
            artifacts = render_figure2(
                {
                    "paired_contrasts": paired_path,
                    "curves_d240": curves_path,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                    "subject_level": None,
                },
                out,
            )
            panel_a = next(
                p for p in artifacts.panels if p.panel_id == "matched_low_effort_curves"
            )
            self.assertIn("HIIT Sensitivity", panel_a.title)
            self.assertIn(
                "cohort=HIIT_sensitivity_matched_observation_pairs",
                panel_a.analysis_keys,
            )
            self.assertIn(
                f"aggregation={HIIT_MATCHED_PAIR_AGGREGATION}",
                panel_a.analysis_keys,
            )
            self.assertIn(
                f"ci={HIIT_CLUSTER_BOOTSTRAP_CI_METHOD}",
                panel_a.analysis_keys,
            )
            curves_out = read_csv_rows(
                out / "source_data" / "figure2_panel_a_matched_lag_curves.csv"
            )
            self.assertTrue(curves_out)
            self.assertEqual({int(float(r["n_matched_pairs"])) for r in curves_out}, {3})
            self.assertEqual(
                {int(float(r["n_session_clusters"])) for r in curves_out}, {2}
            )
            self.assertNotIn("n_participants", curves_out[0])
            caption = (out / "figure2_caption.txt").read_text(encoding="utf-8")
            self.assertIn("n_matched_pairs", caption)
            self.assertIn("n_session_clusters", caption)
            self.assertIn("cluster bootstrap", caption.casefold())

            panel_b = next(
                p
                for p in artifacts.panels
                if p.panel_id == "matched_lag_difference_curves"
            )
            self.assertIn("HIIT Sensitivity", panel_b.title)
            self.assertIn(
                "cohort=HIIT_sensitivity_matched_observation_pairs",
                panel_b.analysis_keys,
            )
            self.assertIn(
                f"ci={HIIT_CLUSTER_BOOTSTRAP_CI_METHOD}",
                panel_b.analysis_keys,
            )
            self.assertIn(
                f"panel_status={PANEL_STATUS_SENSITIVITY_DISPLAY}",
                panel_b.analysis_keys,
            )
            self.assertNotIn(
                f"panel_status={PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
                panel_b.analysis_keys,
            )
            delta_out = read_csv_rows(
                out / "source_data" / "figure2_panel_b_lag_difference_curves.csv"
            )
            self.assertTrue(delta_out)
            self.assertEqual({int(float(r["n_matched_pairs"])) for r in delta_out}, {3})
            self.assertEqual(
                {int(float(r["n_session_clusters"])) for r in delta_out}, {2}
            )
            self.assertNotIn("n_participants", delta_out[0])
            self.assertEqual({r["value_field"] for r in delta_out}, {"delta_z"})
            self.assertIn("lag-difference", caption.casefold())
            self.assertIn("Δz", caption)
            self.assertIn("Gaussian kernel", caption)
            self.assertIn("unsmoothed data", caption.casefold())
            self.assertIn("display_smooth=gaussian_sigma_2s", panel_a.analysis_keys)
            self.assertIn("source_data=unsmoothed", panel_a.analysis_keys)
            self.assertIn("display_smooth=gaussian_sigma_2s", panel_b.analysis_keys)

            panel_c = next(
                p for p in artifacts.panels if p.panel_id == "alpha_primary_meta_forest"
            )
            self.assertIn("HIIT Sensitivity", panel_c.title)
            self.assertIn(
                f"panel_status={PANEL_STATUS_SENSITIVITY_DISPLAY}",
                panel_c.analysis_keys,
            )

            panel_e = next(
                p for p in artifacts.panels if p.panel_id == "paired_peaks_mu_fwhm"
            )
            self.assertIn("HIIT Sensitivity", panel_e.title)
            self.assertIn(
                f"panel_status={PANEL_STATUS_SENSITIVITY_DISPLAY}",
                panel_e.analysis_keys,
            )
            peaks = read_csv_rows(
                out / "source_data" / "figure2_panel_e_paired_peaks.csv"
            )
            self.assertTrue(peaks)
            self.assertEqual({r["row_type"] for r in peaks}, {"sensitivity_display"})
            self.assertEqual(len(peaks), 12)  # 3 contrasts × 4 bands

            panel_f = next(p for p in artifacts.panels if p.panel_id == "graded_ds003690")
            self.assertIn(
                f"panel_status={PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
                panel_f.analysis_keys,
            )

    def test_unmatched_effort_excluded(self) -> None:
        paired = [
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="hiit-01-ph-pre-rest",
                effort_id="hiit-01-ph-pre-tetris",
            )
        ]
        # Only rest curve present — effort missing → wiring gap, empty panel A
        curves = _curve_rows("hiit-01-ph-pre-rest", r_by_lag={0: 0.3})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path = root / "paired.csv"
            curves_path = root / "curves.csv"
            _write_csv(paired_path, paired)
            _write_csv(curves_path, curves)
            out = root / "figures"
            render_figure2(
                {
                    "paired_contrasts": paired_path,
                    "curves_d240": curves_path,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                    "subject_level": None,
                },
                out,
            )
            self.assertEqual(
                read_csv_rows(
                    out / "source_data" / "figure2_panel_a_matched_lag_curves.csv"
                ),
                [],
            )
            gaps = read_csv_rows(
                out / "source_data" / "figure2_panel_a_wiring_gaps.csv"
            )
            self.assertTrue(gaps)
            self.assertEqual(gaps[0]["reason"], "observation_ids_missing_from_curves")

    def test_low_demand_mean_matches_observation_pool_fisher_z(self) -> None:
        """Same Rest obs as Fig1 Panel B style: mean of pair z_low == mean fisher_z(r)."""
        paired = [
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_pre_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="rest-a",
                effort_id="tet-a",
            ),
            _paired_row(
                dataset_id="hiit",
                contrast_id="ph_post_rest__tetris",
                participant_id="01",
                session_id="ph",
                band="theta",
                low_id="rest-b",
                effort_id="tet-b",
            ),
        ]
        r_rest = (0.2, -0.1)
        curves: list[dict[str, object]] = []
        curves.extend(_curve_rows("rest-a", r_by_lag={0: r_rest[0]}))
        curves.extend(_curve_rows("tet-a", r_by_lag={0: 0.0}))
        curves.extend(_curve_rows("rest-b", r_by_lag={0: r_rest[1]}))
        curves.extend(_curve_rows("tet-b", r_by_lag={0: 0.0}))
        series, _ = build_hiit_sensitivity_panel_a_series(
            paired, build_curve_lag_index(curves)
        )
        ci = hiit_matched_pair_cluster_bootstrap_ci(
            series, band="theta", value_field="z_low", n_bootstrap=30
        )
        expected = float(np.mean([fisher_z(r) for r in r_rest]))
        self.assertAlmostEqual(float(ci[0]["mean"]), expected, places=12)

    def test_empty_inputs_remain_expected_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            artifacts = render_figure2(
                {
                    "paired_contrasts": None,
                    "curves_d240": None,
                    "dataset_effects": None,
                    "meta_analysis": None,
                    "peak_equivalence": None,
                    "mixed_model": None,
                    "protocol_audit": None,
                    "subject_level": None,
                },
                out,
            )
            panel_a = next(
                p for p in artifacts.panels if p.panel_id == "matched_low_effort_curves"
            )
            self.assertIn(
                f"panel_status={PANEL_STATUS_EXPECTED_NOT_APPLICABLE}",
                panel_a.analysis_keys,
            )


if __name__ == "__main__":
    unittest.main()
