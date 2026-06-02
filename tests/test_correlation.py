from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_eeg.config import CorrelationConfig, PathsConfig, PipelineConfig
from ppg_eeg.correlation import (
    CORRELATIONS_FDR_PRIMARY_COLUMNS,
    apply_fdr,
    build_correlation_heatmap_tables,
    compute_pairwise_correlations,
    compute_trend_agreement,
    plot_correlation_heatmap,
    significance_symbol,
    write_correlation_heatmap,
)


def _test_cfg() -> PipelineConfig:
    return PipelineConfig(
        dataset_ids=["hiit", "ds006848"],
        dataset_id="hiit",
        paths=PathsConfig(raw_root=Path("."), out_root=Path("./derivatives")),
        subjects=[],
        tasks=[],
        conditions=[],
        sessions=[],
    )


def _test_cfg_auto() -> PipelineConfig:
    cfg = _test_cfg()
    return PipelineConfig(
        dataset_ids=cfg.dataset_ids,
        dataset_id=cfg.dataset_id,
        paths=cfg.paths,
        subjects=cfg.subjects,
        tasks=cfg.tasks,
        conditions=cfg.conditions,
        sessions=cfg.sessions,
        eeg=cfg.eeg,
        ppg=cfg.ppg,
        features=cfg.features,
        correlation=CorrelationConfig(
            alpha=cfg.correlation.alpha,
            min_n=cfg.correlation.min_n,
            methods=["auto"],
            primary_method="auto",
            fdr_method=cfg.correlation.fdr_method,
        ),
        output=cfg.output,
    )


class TestCorrelation(unittest.TestCase):
    def test_pairwise_fdr_and_trend(self) -> None:
        cfg = _test_cfg()
        rows = []
        for dataset_id in ["hiit", "ds006848"]:
            for i in range(8):
                x = float(i)
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "observation_id": f"{dataset_id}-{i}",
                        "subject_id": str(i),
                        "task_label": "rest",
                        "condition_label": "rest",
                        "session_label": "",
                        "modality": "",
                        "timepoint": "",
                        "state": "",
                        "eeg_fm_theta": x,
                        "ppg_mean_hr_bpm": x * 1.5 + 0.2,
                    }
                )
        merged = pd.DataFrame(rows)

        raw = compute_pairwise_correlations(
            merged,
            eeg_features=["eeg_fm_theta"],
            ppg_features=["ppg_mean_hr_bpm"],
            cfg=cfg,
        )
        self.assertGreater(len(raw), 0)
        self.assertIn("p_value", raw.columns)

        fdr = apply_fdr(raw, cfg=cfg)
        self.assertIn("q_value", fdr.columns)
        self.assertIn("reject_fdr", fdr.columns)
        self.assertIn("sig_p_005", fdr.columns)
        self.assertIn("sig_q_005", fdr.columns)
        self.assertIn("sig_q_010", fdr.columns)
        self.assertIn("sig_q_015", fdr.columns)
        self.assertEqual(list(fdr.columns[: len(CORRELATIONS_FDR_PRIMARY_COLUMNS)]), list(CORRELATIONS_FDR_PRIMARY_COLUMNS))
        self.assertTrue(bool(fdr["sig_p_005"].iloc[0]))
        self.assertTrue(bool(fdr["sig_q_005"].iloc[0]))

        trend, summary = compute_trend_agreement(fdr, cfg=cfg)
        self.assertEqual(len(trend), 1)
        self.assertEqual(summary["n_pairs"], 1)

    def test_auto_method_selection_uses_normality_logic(self) -> None:
        cfg = _test_cfg_auto()

        normal_x = np.array(
            [-1.6, -1.2, -1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.6]
        )
        normal_y = 0.8 * normal_x + 0.2

        skew_x = np.exp(np.linspace(0.0, 2.5, 25))
        skew_y = 0.7 * skew_x + 0.1

        rows = []
        for i, (x, y) in enumerate(zip(normal_x, normal_y, strict=True)):
            rows.append(
                {
                    "dataset_id": "hiit",
                    "observation_id": f"normal-{i}",
                    "subject_id": str(i),
                    "task_label": "rest",
                    "condition_label": "rest",
                    "session_label": "",
                    "modality": "",
                    "timepoint": "",
                    "state": "",
                    "eeg_feature_x": float(x),
                    "ppg_feature_y": float(y),
                }
            )

        for i, (x, y) in enumerate(zip(skew_x, skew_y, strict=True)):
            rows.append(
                {
                    "dataset_id": "ds006848",
                    "observation_id": f"skew-{i}",
                    "subject_id": str(i),
                    "task_label": "rest",
                    "condition_label": "rest",
                    "session_label": "",
                    "modality": "",
                    "timepoint": "",
                    "state": "",
                    "eeg_feature_x": float(x),
                    "ppg_feature_y": float(y),
                }
            )

        merged = pd.DataFrame(rows)
        raw = compute_pairwise_correlations(
            merged,
            eeg_features=["eeg_feature_x"],
            ppg_features=["ppg_feature_y"],
            cfg=cfg,
        )

        normal_row = raw[raw["dataset_id"] == "hiit"].iloc[0]
        skew_row = raw[raw["dataset_id"] == "ds006848"].iloc[0]

        self.assertEqual(normal_row["method"], "auto")
        self.assertEqual(normal_row["selected_method"], "pearson")
        self.assertGreater(float(normal_row["x_shapiro_p"]), 0.05)
        self.assertGreater(float(normal_row["y_shapiro_p"]), 0.05)

        self.assertEqual(skew_row["method"], "auto")
        self.assertEqual(skew_row["selected_method"], "spearman")
        self.assertLess(float(skew_row["x_shapiro_p"]), 0.05)
        self.assertLess(float(skew_row["y_shapiro_p"]), 0.05)

    def test_significance_symbol_thresholds(self) -> None:
        self.assertEqual(significance_symbol(q_value=0.049, p_value=0.8), "***")
        self.assertEqual(significance_symbol(q_value=0.09, p_value=0.8), "**")
        self.assertEqual(significance_symbol(q_value=0.12, p_value=0.8), "*")
        self.assertEqual(significance_symbol(q_value=0.25, p_value=0.04), "+")
        self.assertEqual(significance_symbol(q_value=float("nan"), p_value=0.03), "+")
        self.assertEqual(significance_symbol(q_value=0.50, p_value=0.5), "")

    def test_build_correlation_heatmap_tables(self) -> None:
        corr_fdr = pd.DataFrame(
            [
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_a",
                    "ppg_feature": "ppg_1",
                    "correlation": 0.4,
                    "p_value": 0.001,
                    "q_value": 0.01,
                },
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_a",
                    "ppg_feature": "ppg_2",
                    "correlation": -0.3,
                    "p_value": 0.04,
                    "q_value": 0.11,
                },
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_b",
                    "ppg_feature": "ppg_1",
                    "correlation": 0.1,
                    "p_value": 0.03,
                    "q_value": 0.21,
                },
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_b",
                    "ppg_feature": "ppg_2",
                    "correlation": -0.2,
                    "p_value": 0.8,
                    "q_value": 0.8,
                },
            ]
        )

        corr_matrix, symbol_matrix = build_correlation_heatmap_tables(
            corr_fdr,
            dataset_id="hiit",
            method="spearman",
        )

        self.assertEqual(corr_matrix.shape, (2, 2))
        self.assertEqual(symbol_matrix.loc["eeg_a", "ppg_1"], "***")
        self.assertEqual(symbol_matrix.loc["eeg_a", "ppg_2"], "*")
        self.assertEqual(symbol_matrix.loc["eeg_b", "ppg_1"], "+")
        self.assertEqual(symbol_matrix.loc["eeg_b", "ppg_2"], "")

    def test_plot_correlation_heatmap(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        corr_fdr = pd.DataFrame(
            [
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_a",
                    "ppg_feature": "ppg_1",
                    "correlation": 0.4,
                    "p_value": 0.001,
                    "q_value": 0.01,
                },
                {
                    "dataset_id": "hiit",
                    "method": "spearman",
                    "eeg_feature": "eeg_b",
                    "ppg_feature": "ppg_1",
                    "correlation": -0.1,
                    "p_value": 0.03,
                    "q_value": 0.2,
                },
            ]
        )

        fig, ax = plot_correlation_heatmap(
            corr_fdr,
            dataset_id="hiit",
            method="spearman",
            show_values=True,
        )
        self.assertIn("Correlation heatmap", ax.get_title())
        self.assertEqual(ax.get_xlabel(), "PPG feature")
        plt.close(fig)


    def test_write_correlation_heatmap(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "correlations_heatmap.png"
            corr_fdr = pd.DataFrame(
                [
                    {
                        "dataset_id": "hiit",
                        "method": "spearman",
                        "eeg_feature": "eeg_a",
                        "ppg_feature": "ppg_1",
                        "correlation": 0.4,
                        "p_value": 0.001,
                        "q_value": 0.01,
                    },
                    {
                        "dataset_id": "hiit",
                        "method": "spearman",
                        "eeg_feature": "eeg_b",
                        "ppg_feature": "ppg_1",
                        "correlation": -0.1,
                        "p_value": 0.03,
                        "q_value": 0.2,
                    },
                ]
            )

            fig = write_correlation_heatmap(
                corr_fdr,
                out_path,
                dataset_id="hiit",
                method="spearman",
            )
            self.assertIsNotNone(fig)
            self.assertTrue(out_path.exists())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
