from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.hiit_combine import (
    combine_aligned_dataframes,
    combined_observation_id,
)
from ppg_eeg.temporal_coupling.paths import hiit_condition_from_observation_id as partition_key


class TestHiitCombine(unittest.TestCase):
    def test_combined_observation_id(self) -> None:
        self.assertEqual(combined_observation_id(subject_id="01", modality="ps", timepoint="pre"), "hiit-01-ps-pre")

    def test_partition_key_combined(self) -> None:
        self.assertEqual(partition_key("hiit-01-ps-pre"), "ps_pre")
        self.assertEqual(partition_key("hiit-01-ph-post"), "ph_post")

    def test_partition_key_per_session(self) -> None:
        self.assertEqual(partition_key("hiit-01-ps-pre-rest"), "ps_pre_rest")

    def test_combine_aligned_concatenates_rest_then_tetris(self) -> None:
        def _seg(offset: float, n: int = 5) -> pd.DataFrame:
            time_s = np.arange(n, dtype=float) + offset
            return pd.DataFrame(
                {
                    "dataset_id": "hiit",
                    "subject_id": "01",
                    "task": "rest",
                    "observation_id": "hiit-01-ps-pre-rest",
                    "time_s": time_s,
                    "hr": np.linspace(1.0, 2.0, n),
                    "rmssd": np.linspace(2.0, 3.0, n),
                    "sdnn": np.linspace(3.0, 4.0, n),
                    "mean_rr": np.linspace(4.0, 5.0, n),
                    "theta_env": np.linspace(5.0, 6.0, n),
                    "alpha_env": np.linspace(6.0, 7.0, n),
                    "beta_env": np.linspace(7.0, 8.0, n),
                }
            )

        rest = _seg(30.0)
        tetris = _seg(100.0)
        combined = combine_aligned_dataframes(
            rest,
            tetris,
            combined_observation_id="hiit-01-ps-pre",
            combined_condition="ps_pre",
            timepoint="pre",
            fs_hz=1.0,
            z_score=True,
        )
        self.assertEqual(len(combined), 10)
        self.assertEqual(combined.iloc[0]["observation_id"], "hiit-01-ps-pre")
        self.assertEqual(combined.iloc[0]["condition"], "ps_pre")
        self.assertAlmostEqual(float(combined.iloc[-1]["hr"]), 2.0)
        self.assertAlmostEqual(float(combined.iloc[5]["hr"]), 1.0)
        self.assertAlmostEqual(float(combined["hr_z"].std(ddof=0)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
