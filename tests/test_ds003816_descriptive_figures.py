from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg", force=True)

from ppg_eeg.confirmatory.ds003816_descriptive_figures import (
    DATASET_ID,
    REQUIRED_STEMS,
    biological_id_from_row,
    render_ds003816_descriptive_qc,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tiny_tree(root: Path) -> None:
    _write_csv(
        root / "C0" / "data_audit.csv",
        [
            {
                "dataset_id": DATASET_ID,
                "observation_id": "ds003816-p1-ses-01-task-preresting",
                "participant_id": "p1",
                "eeg_duration_s": 400,
                "usable": True,
            },
            {
                "dataset_id": DATASET_ID,
                "observation_id": "ds003816-p2-ses-01-task-preresting",
                "participant_id": "p2",
                "eeg_duration_s": 40,
                "usable": False,
            },
        ],
    )
    _write_csv(
        root / "C0" / "eligibility_by_duration.csv",
        [
            {
                "dataset_id": DATASET_ID,
                "observation_id": "ds003816-p1-ses-01-task-preresting",
                "duration_s": 60,
                "status": "eligible",
                "endpoint_alias": "SWPI",
            },
            {
                "dataset_id": DATASET_ID,
                "observation_id": "ds003816-p2-ses-01-task-preresting",
                "duration_s": 60,
                "status": "ineligible",
                "endpoint_alias": "SWPI",
            },
        ],
    )
    _write_csv(
        root / "C1b" / "cardiac_peak_qc.csv",
        [
            {
                "dataset_id": DATASET_ID,
                "observation_id": "ds003816-p1-ses-01-task-preresting",
                "usable": True,
                "reason_code": "",
            }
        ],
    )
    curve_rows = []
    endpoint_rows = []
    subject_rows = []
    for pid, obs, r0 in (("p1", "ds003816-p1-ses-01-task-preresting", 0.2), ("p2", "ds003816-p2-ses-01-task-lkmself", 0.1)):
        for band in ("theta", "alpha", "beta", "low_gamma"):
            for lag in range(-2, 3):
                curve_rows.append(
                    {
                        "dataset_id": DATASET_ID,
                        "observation_id": obs,
                        "participant_id": pid,
                        "duration_s": 60,
                        "endpoint_alias": "SWPI",
                        "band": band,
                        "power_representation": "absolute_log10",
                        "lag_s": lag,
                        "r": r0 - 0.02 * abs(lag),
                    }
                )
            endpoint_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "observation_id": obs,
                    "participant_id": pid,
                    "condition": "preresting" if pid == "p1" else "lkmself",
                    "duration_s": 60,
                    "endpoint_alias": "SWPI",
                    "band": band,
                    "power_representation": "absolute_log10",
                    "eligible": True,
                    "endpoint_index": r0,
                    "n_common_support": 20,
                }
            )
            subject_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "observation_ids": obs,
                    "participant_id": pid,
                    "condition": "preresting" if pid == "p1" else "lkmself",
                    "duration_s": 60,
                    "endpoint_alias": "SWPI",
                    "band": band,
                    "power_representation": "absolute_log10",
                    "endpoint_eligible": True,
                    "endpoint_index": r0,
                    "n_common_support": 20,
                }
            )
    _write_csv(root / "C2" / "confirmatory_cross_correlation_curves_D60.csv", curve_rows)
    _write_csv(root / "C3" / "confirmatory_endpoint_metrics_D60.csv", endpoint_rows)
    _write_csv(root / "C5" / "subject_level_metrics.csv", subject_rows)
    _write_csv(
        root / "C5" / "pairing_qc.csv",
        [
            {
                "dataset_id": DATASET_ID,
                "pairing_status": "no_prespecified_contrast",
                "reason_code": "no_prespecified_contrast",
            }
        ],
    )


class TestDs003816DescriptiveFigures(unittest.TestCase):
    def test_biological_id_from_observation(self) -> None:
        self.assertEqual(
            biological_id_from_row({"observation_id": "ds003816-01lt-ses-01-task-preresting"}),
            "01lt",
        )
        self.assertEqual(
            biological_id_from_row({"observation_ids": "ds003816-01lt-ses-01-task-preresting"}),
            "01lt",
        )

    def test_renderer_writes_required_package_without_forbidden_panels(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds003816"
            _tiny_tree(root)
            paths = render_ds003816_descriptive_qc(root)
            figure_dir = root / "C7" / "figures"
            for stem in REQUIRED_STEMS:
                self.assertTrue((figure_dir / f"{stem}.png").is_file(), stem)
                self.assertTrue((figure_dir / f"{stem}.pdf").is_file(), stem)
                source = figure_dir / "source_data" / f"{stem}.csv"
                self.assertTrue(source.is_file(), stem)
                with source.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertGreater(len(rows), 0, stem)
                caption = (figure_dir / f"{stem}_caption.txt").read_text(encoding="utf-8")
                self.assertIn("Sensitivity / descriptive only", caption)
                self.assertIn("Not pooled", caption)
                self.assertIn("No prespecified paired contrast", caption)
            created = {path.name.lower() for path in figure_dir.glob("ds003816_*")}
            for token in ("forest", "mu_tost", "fwhm", "equivalence", "temporal_null"):
                self.assertFalse(any(token in name for name in created), token)
            inventory = (figure_dir / "ds003816_descriptive_qc_inventory.json").read_text(encoding="utf-8")
            self.assertIn('"generated_paired_forest": false', inventory)
            self.assertIn('"generated_mu_fwhm_timing": false', inventory)
            self.assertTrue(paths["lag_png"].is_file())
            # Finite bio-level alpha mean in source
            with (figure_dir / "source_data" / "ds003816_alpha_d60_descriptive_summary.csv").open() as handle:
                alpha_rows = list(csv.DictReader(handle))
            bio = next(row for row in alpha_rows if row["estimand"] == "participant_mean_swpi" and row["n"])
            self.assertTrue(math.isfinite(float(bio["mean_swpi"])))
            with (figure_dir / "source_data" / "ds003816_duration_common_support_qc.csv").open() as handle:
                qc_rows = {row["item"]: row for row in csv.DictReader(handle)}
            self.assertEqual(int(qc_rows["c3_d60_swpi_observations"]["n"]), 2)
            self.assertEqual(int(qc_rows["c3_d60_swpi_participants"]["n"]), 2)


if __name__ == "__main__":
    unittest.main()
