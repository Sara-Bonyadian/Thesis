"""Non-HIIT confirmatory path: real dataset when present, else fixture."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from ppg_eeg.confirmatory.capability_resolution import resolve_effective_capabilities
from ppg_eeg.confirmatory.config import load_dataset_config, load_master_config
from ppg_eeg.confirmatory.protocol_audit import (
    EligibilityMetadata,
    evaluate_all_durations,
    observations_from_dataset_config,
)
from ppg_eeg.confirmatory.reason_codes import ENDPOINT_CONTRACT_NON_ZLPI
from ppg_eeg.confirmatory.validation import validate_dataset_configuration

REPO = Path(__file__).resolve().parents[1]
MASTER = REPO / "zero-lag-reanalysis-repo" / "master.yaml"
DATASETS = REPO / "zero-lag-reanalysis-repo" / "datasets"


def _first_available_non_hiit() -> Path | None:
    master = load_master_config(MASTER)
    order = list(master.dataset_roles.primary) + list(master.dataset_roles.sensitivity)
    for dataset_id in order:
        if dataset_id.casefold() == "hiit":
            continue
        yaml_path = DATASETS / f"{dataset_id}.yaml"
        if not yaml_path.is_file():
            continue
        cfg = load_dataset_config(yaml_path, master=master)
        if Path(cfg.paths.raw_root).is_dir():
            return yaml_path
    for yaml_path in sorted(DATASETS.glob("*.yaml")):
        if yaml_path.stem.casefold() == "hiit":
            continue
        cfg = load_dataset_config(yaml_path, master=master)
        if Path(cfg.paths.raw_root).is_dir():
            return yaml_path
    return None


class TestNonHiitVerification(unittest.TestCase):
    def test_real_or_fixture_non_hiit_path(self) -> None:
        master = load_master_config(MASTER)
        yaml_path = _first_available_non_hiit()
        if yaml_path is not None:
            cfg = load_dataset_config(yaml_path, master=master)
            observations = observations_from_dataset_config(cfg)
            issues = validate_dataset_configuration(cfg, observations)
            # Soft errors are allowed; hard empty discovery is not.
            self.assertGreater(len(observations), 0)
            effective, rows = resolve_effective_capabilities(
                cfg, observations
            )
            self.assertTrue(rows)
            self.assertTrue(any(r.capability == "has_eeg" for r in rows))
            condition = next(iter(cfg.normalization.condition_semantics), "rest")
            meta = EligibilityMetadata(
                dataset_id=cfg.dataset_id,
                observation_id="fixture-obs",
                participant_id="p1",
                condition=condition,
                source_data_supplied=True,
                eeg_exists=True,
                cardiac_exists=True,
                raw_overlap_s=400.0,
                requires_paired_state=False,
                paired_state_available=True,
                pairing_resolved=True,
                protocol_match=True,
            )
            decisions = {d.duration_s: d for d in evaluate_all_durations([meta])}
            self.assertFalse(decisions[60].standard_zlpi_computable)
            self.assertFalse(decisions[120].standard_zlpi_computable)
            self.assertEqual(
                decisions[60].standard_zlpi_reason_code, ENDPOINT_CONTRACT_NON_ZLPI
            )
            with tempfile.TemporaryDirectory() as tmp:
                note = Path(tmp) / "non_hiit_verification.csv"
                with note.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "dataset_id",
                            "n_observations",
                            "n_validation_issues",
                            "effective_has_eeg",
                            "effective_has_hr",
                            "effective_supports_d240",
                            "mode",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "dataset_id": cfg.dataset_id,
                            "n_observations": len(observations),
                            "n_validation_issues": len(issues),
                            "effective_has_eeg": effective.has_eeg,
                            "effective_has_hr": effective.has_hr,
                            "effective_supports_d240": effective.supports_d240,
                            "mode": "real_dataset",
                        }
                    )
                self.assertTrue(note.is_file())
            return

        # Fixture verification (clearly labeled): use ds003838 YAML structure
        # without requiring a raw root.
        fixture_yaml = DATASETS / "ds003838.yaml"
        self.assertTrue(fixture_yaml.is_file())
        cfg = load_dataset_config(fixture_yaml, master=master)
        self.assertEqual(cfg.dataset_id, "ds003838")
        effective, rows = resolve_effective_capabilities(cfg, [])
        self.assertFalse(effective.has_eeg)
        self.assertTrue(any(r.effective_value == "false" for r in rows))
        meta = EligibilityMetadata(
            dataset_id="ds003838",
            observation_id="fixture-obs",
            participant_id="p1",
            condition="rest",
            source_data_supplied=True,
            eeg_exists=True,
            cardiac_exists=True,
            raw_overlap_s=400.0,
            requires_paired_state=True,
            paired_state_available=True,
            pairing_resolved=True,
            protocol_match=True,
        )
        d60 = next(d for d in evaluate_all_durations([meta]) if d.duration_s == 60)
        self.assertFalse(d60.standard_zlpi_computable)
        self.assertEqual(d60.standard_zlpi_reason_code, ENDPOINT_CONTRACT_NON_ZLPI)


if __name__ == "__main__":
    unittest.main()
