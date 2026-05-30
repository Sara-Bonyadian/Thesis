from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from ppg_eeg.config import PathsConfig, PipelineConfig
from ppg_eeg.correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement


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

        trend, summary = compute_trend_agreement(fdr, cfg=cfg)
        self.assertEqual(len(trend), 1)
        self.assertEqual(summary["n_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
