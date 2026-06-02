from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ppg_eeg.config import CorrelationConfig, FeaturesConfig, OutputConfig, PathsConfig, PipelineConfig
from ppg_eeg.plot_review import (
    PAIRS_SUMMARY_FILE,
    SCATTER_GRID_FILE,
    load_merged_features,
    run_cross_dataset_review,
    run_dataset_review,
    select_review_pairs,
    write_dataset_review_plots,
)
from ppg_eeg.pipeline import CORRELATIONS_FDR_FILE, MERGED_FEATURES_FILE, OBSERVATIONS_INDEX_FILE


def _cfg(out_root: Path, dataset_id: str) -> PipelineConfig:
    return PipelineConfig(
        dataset_ids=[dataset_id],
        dataset_id=dataset_id,
        paths=PathsConfig(raw_root=Path("."), out_root=out_root),
        subjects=[],
        tasks=[],
        conditions=[],
        sessions=[],
        features=FeaturesConfig(eeg=["eeg_a"], ppg=["ppg_1"], include_robust_z=False),
        correlation=CorrelationConfig(primary_method="auto"),
        output=OutputConfig(),
    )


def _write_subject_outputs(dataset_dir: Path, subject_id: str, eeg_val: float, ppg_val: float) -> None:
    subject_dir = dataset_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / OBSERVATIONS_INDEX_FILE).write_text(
        "dataset_id,observation_id,subject_id\n"
        f"demo,demo-{subject_id},{subject_id}\n",
        encoding="utf-8",
    )
    merged = pd.DataFrame(
        [
            {
                "dataset_id": "demo",
                "observation_id": f"demo-{subject_id}",
                "subject_id": subject_id,
                "eeg_a": eeg_val,
                "ppg_1": ppg_val,
            }
        ]
    )
    merged.to_csv(subject_dir / MERGED_FEATURES_FILE, index=False)


def _write_corr_fdr(dataset_dir: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(dataset_dir / CORRELATIONS_FDR_FILE, index=False)


class TestPlotReview(unittest.TestCase):
    def test_select_review_pairs_prioritizes_significance(self) -> None:
        corr_fdr = pd.DataFrame(
            [
                {
                    "method": "auto",
                    "eeg_feature": "eeg_a",
                    "ppg_feature": "ppg_1",
                    "correlation": 0.2,
                    "p_value": 0.9,
                    "q_value": 0.9,
                    "sig_q_005": False,
                    "sig_q_010": False,
                    "sig_q_015": False,
                    "sig_p_005": False,
                    "n": 5,
                },
                {
                    "method": "auto",
                    "eeg_feature": "eeg_b",
                    "ppg_feature": "ppg_1",
                    "correlation": 0.95,
                    "p_value": 0.01,
                    "q_value": 0.04,
                    "sig_q_005": True,
                    "sig_q_010": True,
                    "sig_q_015": True,
                    "sig_p_005": True,
                    "n": 5,
                },
            ]
        )
        selected = select_review_pairs(corr_fdr, primary_method="auto", max_pairs=1)
        self.assertEqual(selected.iloc[0]["eeg_feature"], "eeg_b")

    def test_write_dataset_review_plots(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            dataset_dir = out_root / "demo"
            dataset_dir.mkdir(parents=True)
            for subject_id, eeg_val, ppg_val in [("s1", 1.0, 2.0), ("s2", 2.0, 4.0), ("s3", 3.0, 6.0)]:
                _write_subject_outputs(dataset_dir, subject_id, eeg_val, ppg_val)
            _write_corr_fdr(
                dataset_dir,
                [
                    {
                        "dataset_id": "demo",
                        "method": "auto",
                        "selected_method": "pearson",
                        "eeg_feature": "eeg_a",
                        "ppg_feature": "ppg_1",
                        "correlation": 0.99,
                        "p_value": 0.01,
                        "q_value": 0.02,
                        "sig_q_005": True,
                        "sig_q_010": True,
                        "sig_q_015": True,
                        "sig_p_005": True,
                        "n": 3,
                    }
                ],
            )
            merged = load_merged_features(dataset_dir)
            corr_fdr = pd.read_csv(dataset_dir / CORRELATIONS_FDR_FILE)
            written = write_dataset_review_plots(
                dataset_dir=dataset_dir,
                corr_fdr=corr_fdr,
                merged=merged,
                dataset_id="demo",
                primary_method="auto",
                max_pairs=3,
            )
            review_dir = dataset_dir / "review"
            self.assertTrue((review_dir / SCATTER_GRID_FILE).exists())
            self.assertTrue((review_dir / PAIRS_SUMMARY_FILE).exists())
            self.assertTrue(any(path.name.startswith("scatter_") for path in written))

    def test_run_cross_dataset_review(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            for dataset_id, corr in [("a", 0.8), ("b", -0.4)]:
                dataset_dir = out_root / dataset_id
                dataset_dir.mkdir(parents=True)
                _write_subject_outputs(dataset_dir, "s1", 1.0, 1.0)
                _write_subject_outputs(dataset_dir, "s2", 2.0, 2.0)
                _write_corr_fdr(
                    dataset_dir,
                    [
                        {
                            "dataset_id": dataset_id,
                            "method": "auto",
                            "selected_method": "pearson",
                            "eeg_feature": "eeg_a",
                            "ppg_feature": "ppg_1",
                            "correlation": corr,
                            "p_value": 0.05,
                            "q_value": 0.10,
                            "sig_q_005": False,
                            "sig_q_010": True,
                            "sig_q_015": True,
                            "sig_p_005": True,
                            "n": 2,
                        }
                    ],
                )

            cfgs = [_cfg(out_root, "a"), _cfg(out_root, "b")]
            compare_out = out_root / "cross"
            written = run_cross_dataset_review(cfgs, compare_out=compare_out, max_pairs=3)
            self.assertTrue(any(path.suffix == ".png" for path in written))
            self.assertTrue((compare_out / "correlations_cross_dataset.csv").exists())

    def test_run_dataset_review(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)

        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            dataset_dir = out_root / "demo"
            dataset_dir.mkdir(parents=True)
            _write_subject_outputs(dataset_dir, "s1", 1.0, 1.0)
            _write_corr_fdr(
                dataset_dir,
                [
                    {
                        "dataset_id": "demo",
                        "method": "auto",
                        "selected_method": "pearson",
                        "eeg_feature": "eeg_a",
                        "ppg_feature": "ppg_1",
                        "correlation": 0.5,
                        "p_value": 0.2,
                        "q_value": 0.2,
                        "sig_q_005": False,
                        "sig_q_010": False,
                        "sig_q_015": False,
                        "sig_p_005": False,
                        "n": 1,
                    }
                ],
            )
            written = run_dataset_review(_cfg(out_root, "demo"), max_pairs=2)
            self.assertTrue(written)


if __name__ == "__main__":
    unittest.main()
