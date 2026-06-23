from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ppg_eeg.datasets import build_observations
from ppg_eeg.datasets.hiit import hiit_task_condition_label
from ppg_eeg.temporal_coupling.paths import (
    HIIT_PARTITION_MODE_TASK_ONLY,
    hiit_task_partition_from_observation_id,
    partition_key_from_row,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


class TestHiit40Partitions(unittest.TestCase):
    def test_task_condition_label(self) -> None:
        self.assertEqual(hiit_task_condition_label("pre", "rest"), "pre_rest")
        self.assertEqual(hiit_task_condition_label("POST", "tetris"), "post_tetris")

    def test_task_partition_from_observation_id(self) -> None:
        self.assertEqual(
            hiit_task_partition_from_observation_id("hiit-01-ph-pre-rest"),
            "pre_rest",
        )
        self.assertEqual(
            hiit_task_partition_from_observation_id("hiit-20-ps-post-tetris"),
            "post_tetris",
        )

    def test_partition_key_task_only_pools_protocols(self) -> None:
        for obs_id in ("hiit-01-ps-pre-rest", "hiit-01-ph-pre-rest"):
            self.assertEqual(
                partition_key_from_row(
                    dataset_id="hiit",
                    task="rest",
                    condition="pre_rest",
                    observation_id=obs_id,
                    hiit_partition_mode=HIIT_PARTITION_MODE_TASK_ONLY,
                ),
                "pre_rest",
            )

    def test_build_observations_task_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HIIT"
            for folder_name, stem in (
                ("HIIT_01_PS", "HIIT_01_PS_PRE"),
                ("HIIT_01_PH", "HIIT_01_PH_PRE"),
            ):
                folder = root / folder_name
                _touch(folder / f"{stem}.vhdr")
                _touch(folder / f"{stem}.eeg")
                _touch(folder / f"{stem}.vmrk")

            rows = build_observations(
                "hiit",
                Path(tmp),
                conditions=["pre_rest"],
                hiit_partition_mode="task_only",
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual({row.condition_label for row in rows}, {"pre_rest"})
            self.assertEqual({row.subject_id for row in rows}, {"01_ps", "01_ph"})


if __name__ == "__main__":
    unittest.main()
