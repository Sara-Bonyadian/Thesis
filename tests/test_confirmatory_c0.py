from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ppg_eeg.confirmatory.data_audit import audit_observation
from ppg_eeg.confirmatory.run import StageContext, run_c0
from ppg_eeg.datasets.base import CanonicalObservation
from ppg_eeg.temporal_coupling.data_audit import SignalFileInfo


def _fake_info(duration_s: float = 300.0, *, cardiac: bool = False) -> SignalFileInfo:
    if cardiac:
        return SignalFileInfo(
            duration_s=duration_s,
            sfreq=100.0,
            source="test",
            ch_names=("photosensor",),
            ch_types=("misc",),
        )
    return SignalFileInfo(
        duration_s=duration_s,
        sfreq=250.0,
        source="test",
        ch_names=("Fz", "Cz", "Pz"),
        ch_types=("eeg", "eeg", "eeg"),
    )


class TestConfirmatoryC0DataAudit(unittest.TestCase):
    def test_audit_record_exposes_required_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            eeg = root / "eeg.eeg"
            cardiac = root / "card.set"
            eeg.write_bytes(b"x")
            cardiac.write_bytes(b"x")
            obs = CanonicalObservation(
                dataset_id="hiit",
                observation_id="hiit-01-ph-pre-rest",
                subject_id="01_ph",
                task_label="rest",
                condition_label="ph_pre_rest",
                eeg_path=eeg,
                eeg_format="brainvision",
                ppg_source="separate_file",
                ppg_path=cardiac,
                ppg_format="eeglab",
                session_label="ph",
            )

            def _fake_read(path, *, data_format=None, cache=None):
                _ = data_format, cache
                if Path(path) == eeg:
                    return _fake_info(280.0)
                return _fake_info(260.0, cardiac=True)

            with patch(
                "ppg_eeg.confirmatory.data_audit.read_signal_file_info",
                side_effect=_fake_read,
            ):
                record = audit_observation(obs, lag_max_s=60.0)

        self.assertTrue(record.usable)
        self.assertEqual(record.raw_overlap_s, 260.0)
        self.assertEqual(record.available_cardiac_channel, "photosensor")
        self.assertEqual(record.cardiac_signal_type, "ppg")
        self.assertEqual(record.session_id, "ph")
        self.assertEqual(record.participant_id, "01")
        self.assertGreater(record.recommended_max_lag_s, 0.0)
        row = record.to_row()
        for key in (
            "observation_id",
            "participant_id",
            "session_id",
            "condition",
            "task",
            "eeg_duration_s",
            "cardiac_duration_s",
            "raw_overlap_s",
            "available_cardiac_channel",
            "cardiac_signal_type",
            "recommended_max_lag_s",
            "usable",
            "exclusion_reason",
        ):
            self.assertIn(key, row)

    def test_run_c0_writes_full_artifact_set(self) -> None:
        from ppg_eeg.confirmatory.run import resolve_master_and_dataset

        repo = Path(__file__).resolve().parents[1]
        config_dir = repo / "zero-lag-reanalysis-repo"
        hiit = config_dir / "smoke" / "hiit" / "confirmatory.yaml"
        master, dataset, master_path, dataset_path = resolve_master_and_dataset(
            hiit, config_dir=config_dir
        )
        with TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "conf_out"
            observations: list[CanonicalObservation] = []
            for task, condition, obs_id in (
                ("rest", "ph_pre_rest", "hiit-01-ph-pre-rest"),
                ("tetris", "ph_pre_tetris", "hiit-01-ph-pre-tetris"),
            ):
                eeg = Path(tmp) / f"{obs_id}_eeg.eeg"
                cardiac = Path(tmp) / f"{obs_id}_card.set"
                eeg.write_bytes(b"x")
                cardiac.write_bytes(b"x")
                observations.append(
                    CanonicalObservation(
                        dataset_id="hiit",
                        observation_id=obs_id,
                        subject_id="01_ph",
                        task_label=task,
                        condition_label=condition,
                        eeg_path=eeg,
                        eeg_format="brainvision",
                        ppg_source="separate_file",
                        ppg_path=cardiac,
                        ppg_format="eeglab",
                        session_label="ph",
                    )
                )
            ctx = StageContext(
                master=master,
                dataset=dataset,
                master_path=master_path,
                dataset_path=dataset_path,
            )

            def _fake_read(path, *, data_format=None, cache=None):
                _ = path, data_format, cache
                name = Path(path).name
                if name.endswith("_eeg.eeg"):
                    return _fake_info(300.0)
                return _fake_info(300.0, cardiac=True)

            with (
                patch.object(
                    StageContext,
                    "root",
                    property(lambda self: out_root),
                ),
                patch(
                    "ppg_eeg.confirmatory.run._load_observations",
                    return_value=observations,
                ),
                patch(
                    "ppg_eeg.confirmatory.protocol_audit.observations_from_dataset_config",
                    return_value=observations,
                ),
                patch(
                    "ppg_eeg.confirmatory.data_audit.read_signal_file_info",
                    side_effect=_fake_read,
                ),
            ):
                mapping = run_c0(ctx)

            expected = {
                "protocol_audit.csv",
                "paired_subject_sets.json",
                "data_audit.csv",
                "eligibility_by_duration.csv",
                "eligibility_qc_summary.csv",
                "capability_resolution.csv",
                "stage_status.json",
            }
            c0 = out_root / "C0"
            self.assertEqual(expected, {path.name for path in c0.iterdir()})
            self.assertTrue(mapping["data_audit"].is_file())
            status = json.loads((c0 / "stage_status.json").read_text(encoding="utf-8"))
            self.assertEqual(
                status["audits_completed"],
                [
                    "raw_data_audit",
                    "pairing_audit",
                    "duration_eligibility_audit",
                ],
            )
            with (c0 / "data_audit.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["raw_overlap_s"]), 300.0)
            with (c0 / "eligibility_by_duration.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                elig = list(csv.DictReader(handle))
            by_duration = {
                int(row["duration_s"]): row
                for row in elig
                if row["observation_id"] == observations[0].observation_id
            }
            for duration in (240, 180, 120, 60):
                self.assertEqual(by_duration[duration]["status"], "eligible")
                self.assertEqual(by_duration[duration]["clean_beat_span_s"], "")


if __name__ == "__main__":
    unittest.main()
