"""Canonical identity invariants across BIDS and non-BIDS conventions."""

from __future__ import annotations

import unittest
from pathlib import Path

from ppg_eeg.datasets import (
    CanonicalObservation,
    canonical_observation_identity,
    validate_observation_identities,
)


def _obs(**overrides: object) -> CanonicalObservation:
    values: dict[str, object] = {
        "dataset_id": "example",
        "observation_id": "example-sub-01-ses-a-run-02-task-rest",
        "subject_id": "sub-01",
        "task_label": "rest",
        "condition_label": "rest",
        "eeg_path": Path("."),
        "eeg_format": "n/a",
        "ppg_source": "embedded_eeg",
    }
    values.update(overrides)
    return CanonicalObservation(**values)  # type: ignore[arg-type]


class TestCanonicalObservationIdentity(unittest.TestCase):
    def test_bids_entities_produce_stable_pairing_identity(self) -> None:
        identity = canonical_observation_identity(
            _obs(participant_id="01", session_id="a")
        )
        self.assertEqual(identity.dataset_id, "example")
        self.assertEqual(identity.participant_id, "01")
        self.assertEqual(identity.session_id, "a")
        self.assertEqual(identity.run_id, "02")
        self.assertEqual(identity.condition_id, "rest")
        self.assertEqual(identity.pairing_id, "example::01::a::02")

    def test_non_bids_normalized_fields_win(self) -> None:
        identity = canonical_observation_identity(
            _obs(
                observation_id="hiit-01-ph-pre-rest",
                subject_id="01_ph",
                participant_id="01",
                session_id="ph",
                run_id="single",
                condition_id="ph_pre_rest",
            )
        )
        self.assertEqual(identity.participant_id, "01")
        self.assertEqual(identity.session_id, "ph")
        self.assertEqual(identity.condition_id, "ph_pre_rest")
        self.assertEqual(identity.pairing_id, "example::01::ph::single")

    def test_collision_check_rejects_conflicting_same_observation(self) -> None:
        first = _obs(participant_id="01", session_id="a")
        conflicting = _obs(participant_id="02", session_id="a")
        with self.assertRaisesRegex(ValueError, "identity collision"):
            validate_observation_identities([first, conflicting])


if __name__ == "__main__":
    unittest.main()
