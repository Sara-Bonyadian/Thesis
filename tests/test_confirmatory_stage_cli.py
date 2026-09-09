from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ppg_eeg.confirmatory.run import (
    STAGE_ORDER,
    STAGE_REQUIRES,
    StageContext,
    StageError,
    _assert_no_stale_short_duration_peak_inference,
    expand_stages,
    main,
    resolve_master_and_dataset,
)
from ppg_eeg.confirmatory.production import master_config_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestConfirmatoryStageCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        cls.config_dir = cls.repo / "zero-lag-reanalysis-repo"
        cls.hiit = cls.config_dir / "smoke" / "hiit" / "confirmatory.yaml"

    def test_expand_stages_all_and_aliases(self) -> None:
        self.assertEqual(expand_stages("all"), list(STAGE_ORDER))
        self.assertEqual(expand_stages("c0"), ["C0"])
        self.assertEqual(expand_stages("C1a"), ["C1a"])
        with self.assertRaises(ValueError):
            expand_stages("C9")

    def test_stage_graph_covers_dependencies(self) -> None:
        self.assertEqual(STAGE_REQUIRES["C1c"], ("C1a", "C1b"))
        self.assertEqual(STAGE_REQUIRES["C2"], ("C1c",))
        self.assertIn("C5", STAGE_REQUIRES["C6"])

    def test_resolve_hiit_smoke_config(self) -> None:
        master, dataset, _, _ = resolve_master_and_dataset(
            self.hiit, config_dir=self.config_dir
        )
        self.assertEqual(dataset.dataset_id, "hiit")
        self.assertTrue(str(dataset.output_root).endswith("smoke/hiit_m13b"))
        self.assertEqual(master_config_path(self.config_dir).name, "master.yaml")
        self.assertEqual(master.durations.primary_s, 240)

    def test_main_rejects_mode_and_stage_together(self) -> None:
        code = main(["--mode", "preflight", "--stage", "C0", "--config", str(self.hiit)])
        self.assertEqual(code, 2)

    def test_require_stages_blocks_c2_without_c1c(self) -> None:
        master, dataset, master_path, dataset_path = resolve_master_and_dataset(
            self.hiit, config_dir=self.config_dir
        )
        ctx = StageContext(
            master=master,
            dataset=dataset,
            master_path=master_path,
            dataset_path=dataset_path,
        )
        with TemporaryDirectory() as tmp:
            empty = Path(tmp)

            def _empty_stage_dir(stage: str) -> Path:
                return empty / stage

            with patch.object(ctx, "stage_dir", side_effect=_empty_stage_dir):
                from ppg_eeg.confirmatory.run import _require_stages

                with self.assertRaises(StageError):
                    _require_stages(ctx, "C2")

    def test_integrity_check_rejects_short_duration_peak_parameters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "peak_fit_params.csv",
                [
                    {
                        "duration_s": 60,
                        "endpoint_name": "short_window_proximal_index",
                        "peak_height_A": "0.4",
                        "peak_center_mu_s": "",
                        "sigma_s": "",
                        "fwhm_s": "",
                    }
                ],
            )
            with self.assertRaises(StageError):
                _assert_no_stale_short_duration_peak_inference(
                    root=root, stage_name="C7/publish"
                )

    def test_integrity_check_rejects_short_duration_equivalence_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "peak_center_equivalence.csv",
                [
                    {
                        "duration_s": 120,
                        "endpoint_name": "mid_window_proximal_index",
                    }
                ],
            )
            with self.assertRaises(StageError):
                _assert_no_stale_short_duration_peak_inference(
                    root=root, stage_name="C6"
                )

    def test_integrity_check_accepts_duration_endpoint_contracts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(
                root / "duration_sensitivity.csv",
                [
                    {"duration_s": 60, "endpoint_name": "short_window_proximal_index"},
                    {"duration_s": 120, "endpoint_name": "mid_window_proximal_index"},
                    {"duration_s": 180, "endpoint_name": "zlpi"},
                    {"duration_s": 240, "endpoint_name": "zlpi"},
                ],
            )
            _write_csv(
                root / "peak_center_equivalence.csv",
                [{"duration_s": 180, "endpoint_name": "zlpi"}],
            )
            _assert_no_stale_short_duration_peak_inference(root=root, stage_name="C6")


if __name__ == "__main__":
    unittest.main()
