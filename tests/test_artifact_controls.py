from __future__ import annotations

import csv
import math
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from ppg_eeg.confirmatory.artifact_controls import (
    ARTIFACT_CONTROL_RESULTS_FILENAME,
    CONTROL_BROADBAND,
    CONTROL_CARDIAC_FIELD,
    CONTROL_D120,
    CONTROL_D180,
    CONTROL_D60,
    CONTROL_MOTION,
    CONTROL_RESPIRATION,
    CORE_SENSITIVITY_CONTROL_IDS,
    DURATION_SENSITIVITY_FILENAME,
    OPTIONAL_ARTIFACT_CONTROL_IDS,
    PRIMARY_CONTROL_ID,
    SENSITIVITY_QC_FILENAME,
    SENSITIVITY_RESULTS_FILENAME,
    SPECIFICATION_MATRIX_FILENAME,
    beat_density,
    broadband_residualize_log_power,
    control_availability,
    is_primary_cell,
    qrs_interpolate_eeg,
    residualize_series,
    run_artifact_controls,
    write_artifact_control_outputs,
)
from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
)


def _paired(
    *,
    dataset_id: str = "ds003838",
    contrast_id: str = "rest__memory",
    participant_id: str = "p01",
    band: str = "theta",
    delta: float,
    duration_s: int = 240,
    endpoint_name: str = ENDPOINT_ZLPI,
    power_representation: str = "absolute_log10",
    modality: str = "ecg",
    low: str = "rest",
    effort: str = "memory",
    nuisance_control: str = "",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "contrast_id": contrast_id,
        "participant_id": participant_id,
        "session_id": "single",
        "low_demand_condition": low,
        "cognitive_effort_condition": effort,
        "duration_s": duration_s,
        "endpoint_name": endpoint_name,
        "band": band,
        "power_representation": power_representation,
        "modality": modality,
        "delta_endpoint_index": delta,
        "nuisance_control": nuisance_control,
    }


def _subject(
    *,
    dataset_id: str = "ds003838",
    participant_id: str = "p01",
    condition: str,
    band: str = "theta",
    endpoint_index: float,
    modality: str = "ecg",
    mean_hr: float = 70.0,
    beat_count: float = 240.0,
    eye_state: str = "closed",
    duration_s: int = 240,
    endpoint_name: str = ENDPOINT_ZLPI,
    power_representation: str = "absolute_log10",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": "single",
        "condition": condition,
        "band": band,
        "duration_s": duration_s,
        "endpoint_name": endpoint_name,
        "power_representation": power_representation,
        "modality": modality,
        "endpoint_index": endpoint_index,
        "endpoint_eligible": True,
        "mean_hr": mean_hr,
        "beat_count": beat_count,
        "eye_state": eye_state,
        "observation_id": f"{dataset_id}-{participant_id}-{condition}-{modality}",
    }


class TestArtifactInjection(unittest.TestCase):
    def test_qrs_interpolation_reduces_beat_locked_spike(self) -> None:
        n = 240
        times = np.arange(n, dtype=float)
        hr = np.sin(2 * np.pi * times / 40.0)
        eeg = hr.copy()
        beats = np.arange(10, n, 20, dtype=float)
        # Inject large cardiac-field spikes at beats.
        contaminated = eeg.copy()
        for beat in beats:
            idx = int(beat)
            contaminated[max(0, idx - 1) : idx + 2] += 5.0
        cleaned = qrs_interpolate_eeg(contaminated, times, beats, half_width_s=1.0)
        # Spike neighborhoods should move toward surrounding EEG.
        for beat in beats:
            idx = int(beat)
            self.assertLess(abs(cleaned[idx] - eeg[idx]), abs(contaminated[idx] - eeg[idx]))

    def test_cardiac_field_control_does_not_alter_hr(self) -> None:
        n = 240
        times = np.arange(n, dtype=float)
        rng = np.random.default_rng(0)
        hr = rng.normal(size=n)
        eeg = hr + 0.1 * rng.normal(size=n)
        beats = np.arange(15, n, 25, dtype=float)
        for beat in beats:
            eeg[int(beat)] += 3.0
        hr_before = hr.copy()
        result = run_artifact_controls(
            [],
            [],
            enable_optional_artifact_controls=True,
            series_controls=[
                {
                    "observation_id": "obs-1",
                    "dataset_id": "ds_test",
                    "band": "theta",
                    "duration_s": 240,
                    "hr_z": hr,
                    "eeg_z": eeg,
                    "times_s": times,
                    "beat_times_s": beats,
                    "artifact_injected": True,
                }
            ],
        )
        np.testing.assert_array_equal(hr, hr_before)
        self.assertEqual(len(result.artifact_rows), 1)
        self.assertTrue(result.artifact_rows[0]["control_applied"])
        self.assertEqual(result.artifact_rows[0]["status"], "ok")


class TestMissingNuisanceSignals(unittest.TestCase):
    def test_unavailable_nuisance_not_treated_as_zero(self) -> None:
        with self.assertRaises(ValueError):
            residualize_series(np.ones(10), np.asarray([], dtype=float))
        available, note = control_availability(
            {"motion": False, "respiration": "unavailable"},
            CONTROL_MOTION,
        )
        self.assertFalse(available)
        self.assertIn("unavailable", note.casefold())

        paired = [
            _paired(participant_id=f"p{i}", delta=-0.2 - 0.01 * i) for i in range(6)
        ]
        subjects = []
        for i in range(6):
            subjects.append(_subject(participant_id=f"p{i}", condition="rest", endpoint_index=0.4))
            subjects.append(
                _subject(participant_id=f"p{i}", condition="memory", endpoint_index=0.2)
            )
        result = run_artifact_controls(
            subjects,
            paired,
            enable_optional_artifact_controls=True,
            control_inventory={"obs-a": {"motion": False, "respiration": False}},
        )
        motion = [
            r for r in result.sensitivity_rows if r["control_id"] == CONTROL_MOTION
        ]
        self.assertTrue(motion)
        self.assertEqual(motion[0]["status"], "control_unavailable")
        self.assertTrue(math.isnan(float(motion[0]["effect_estimate"])))
        self.assertEqual(motion[0]["n"], 0)

        respiration = [
            r
            for r in result.sensitivity_rows
            if r["control_id"] == CONTROL_RESPIRATION
        ]
        self.assertEqual(respiration[0]["status"], "control_unavailable")


class TestOptionalArtifactControlsDefault(unittest.TestCase):
    def test_default_omits_optional_controls_from_sensitivity_and_qc(self) -> None:
        paired = [_paired(participant_id=f"p{i}", delta=-0.2) for i in range(6)]
        result = run_artifact_controls([], paired)
        sensitivity_ids = {r["control_id"] for r in result.sensitivity_rows}
        qc_ids = {r["control_id"] for r in result.qc_rows}
        for control_id in OPTIONAL_ARTIFACT_CONTROL_IDS:
            self.assertNotIn(control_id, sensitivity_ids)
            self.assertNotIn(control_id, qc_ids)
        for control_id in CORE_SENSITIVITY_CONTROL_IDS:
            self.assertIn(control_id, sensitivity_ids)
        self.assertIn(PRIMARY_CONTROL_ID, sensitivity_ids)
        self.assertIn(CONTROL_BROADBAND, sensitivity_ids)
        self.assertIn(CONTROL_D180, sensitivity_ids)
        spec_ids = {r["control_id"] for r in result.specification_rows}
        self.assertEqual(
            spec_ids,
            {PRIMARY_CONTROL_ID, *CORE_SENSITIVITY_CONTROL_IDS},
        )
        self.assertFalse(result.artifact_rows)

    def test_optional_enabled_without_inventory_marks_unavailable(self) -> None:
        paired = [_paired(participant_id=f"p{i}", delta=-0.2) for i in range(6)]
        result = run_artifact_controls(
            [],
            paired,
            enable_optional_artifact_controls=True,
        )
        motion = [
            r for r in result.sensitivity_rows if r["control_id"] == CONTROL_MOTION
        ]
        self.assertTrue(motion)
        self.assertEqual(motion[0]["status"], "control_unavailable")
        cfa = [
            r
            for r in result.sensitivity_rows
            if r["control_id"] == CONTROL_CARDIAC_FIELD
        ]
        self.assertTrue(cfa)
        self.assertEqual(cfa[0]["status"], "control_unavailable")


class TestDurationSeparation(unittest.TestCase):
    def test_duration_endpoints_stay_separate_and_nonprimary(self) -> None:
        paired = []
        for i in range(8):
            paired.append(_paired(participant_id=f"p{i}", delta=-0.25, duration_s=240))
            paired.append(
                _paired(
                    participant_id=f"p{i}",
                    delta=-0.10,
                    duration_s=180,
                    endpoint_name=ENDPOINT_ZLPI,
                )
            )
            paired.append(
                _paired(
                    participant_id=f"p{i}",
                    delta=-0.80,
                    duration_s=120,
                    endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                )
            )
            paired.append(
                _paired(
                    participant_id=f"p{i}",
                    delta=-0.90,
                    duration_s=60,
                    endpoint_name=ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
                )
            )
        result = run_artifact_controls([], paired)
        d180 = [r for r in result.duration_rows if int(r["duration_s"]) == 180]
        d120 = [r for r in result.duration_rows if int(r["duration_s"]) == 120]
        d60 = [r for r in result.duration_rows if int(r["duration_s"]) == 60]
        self.assertTrue(d180 and d120 and d60)
        self.assertTrue(all(r["endpoint_name"] == ENDPOINT_ZLPI for r in d180))
        self.assertTrue(
            all(r["endpoint_name"] == ENDPOINT_MID_WINDOW_PROXIMAL_INDEX for r in d120)
        )
        self.assertTrue(
            all(r["endpoint_name"] == ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX for r in d60)
        )
        self.assertTrue(all(not r["is_primary_analysis"] for r in d180 + d120 + d60))
        self.assertTrue(all(not r["can_rescue_primary"] for r in d180 + d120 + d60))
        # Strong MWPI must not appear as primary.
        self.assertTrue(
            all(
                not r["is_primary_analysis"]
                for r in result.sensitivity_rows
                if r["control_id"] in {CONTROL_D180, CONTROL_D120, CONTROL_D60}
            )
        )


class TestBroadbandResidualization(unittest.TestCase):
    def test_broadband_residualization_removes_shared_component(self) -> None:
        rng = np.random.default_rng(1)
        broadband = rng.normal(size=200)
        band = 0.8 * broadband + 0.2 * rng.normal(size=200)
        resid = broadband_residualize_log_power(band, broadband)
        finite = np.isfinite(resid) & np.isfinite(broadband)
        corr = np.corrcoef(resid[finite], broadband[finite])[0, 1]
        self.assertLess(abs(corr), 0.05)

    def test_broadband_paired_rows_are_sensitivity_only(self) -> None:
        paired = [
            _paired(participant_id=f"p{i}", delta=-0.2) for i in range(6)
        ] + [
            _paired(
                participant_id=f"p{i}",
                delta=-0.15,
                power_representation="broadband_residualized",
            )
            for i in range(6)
        ]
        result = run_artifact_controls([], paired)
        broadband = [
            r for r in result.sensitivity_rows if r["control_id"] == CONTROL_BROADBAND
        ]
        self.assertTrue(broadband)
        self.assertTrue(all(not r["is_primary_analysis"] for r in broadband))
        self.assertTrue(all(not r["can_rescue_primary"] for r in broadband))
        self.assertTrue(
            all(r["power_representation"] == "broadband_residualized" for r in broadband)
        )


class TestPrimaryProtection(unittest.TestCase):
    def test_primary_reference_unchanged_and_guardrail_ok(self) -> None:
        paired = [
            _paired(participant_id=f"p{i}", delta=-0.3 + 0.01 * i) for i in range(10)
        ]
        # Dramatic sensitivity that must not replace primary.
        paired += [
            _paired(
                participant_id=f"p{i}",
                delta=-0.99,
                duration_s=120,
                endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
            )
            for i in range(10)
        ]
        result = run_artifact_controls([], paired)
        primary = [
            r for r in result.sensitivity_rows if r["control_id"] == PRIMARY_CONTROL_ID
        ]
        self.assertTrue(primary)
        self.assertTrue(all(r["is_primary_analysis"] for r in primary))
        self.assertTrue(all(not r["can_rescue_primary"] for r in result.sensitivity_rows))
        self.assertTrue(
            is_primary_cell(
                endpoint_name=ENDPOINT_ZLPI,
                duration_s=240,
                power_representation="absolute_log10",
            )
        )
        self.assertFalse(
            is_primary_cell(
                endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                duration_s=120,
                power_representation="absolute_log10",
            )
        )
        guard = [
            r for r in result.qc_rows if r["control_id"] == "guardrail"
        ]
        self.assertEqual(guard[0]["status"], "ok")
        # Default specification matrix is primary + core only.
        control_ids = {r["control_id"] for r in result.specification_rows}
        self.assertIn(PRIMARY_CONTROL_ID, control_ids)
        self.assertIn(CONTROL_D120, control_ids)
        self.assertIn(CONTROL_BROADBAND, control_ids)
        self.assertNotIn(CONTROL_CARDIAC_FIELD, control_ids)
        primary = next(
            r for r in result.specification_rows if r["control_id"] == PRIMARY_CONTROL_ID
        )
        self.assertEqual(primary["analysis_priority"], 1)
        self.assertEqual(primary["execution_order"], 1)
        priorities = [int(r["analysis_priority"]) for r in result.specification_rows]
        self.assertEqual(priorities, sorted(priorities))


class TestHelpersAndOutputs(unittest.TestCase):
    def test_beat_density_and_write_outputs(self) -> None:
        self.assertAlmostEqual(beat_density(120, 60), 2.0, places=12)
        self.assertTrue(math.isnan(beat_density(10, 0)))
        paired = [_paired(participant_id=f"p{i}", delta=-0.2) for i in range(5)]
        result = run_artifact_controls([], paired)
        with TemporaryDirectory() as tmp:
            paths = write_artifact_control_outputs(result, tmp)
            expected = {
                SENSITIVITY_RESULTS_FILENAME,
                ARTIFACT_CONTROL_RESULTS_FILENAME,
                DURATION_SENSITIVITY_FILENAME,
                SPECIFICATION_MATRIX_FILENAME,
                SENSITIVITY_QC_FILENAME,
            }
            self.assertEqual({p.name for p in paths.values()}, expected)
            for path in paths.values():
                self.assertTrue(path.is_file())
            self.assertNotIn("modality_comparison", paths)
            with paths["artifact_control_results"].open(encoding="utf-8", newline="") as handle:
                artifact = list(csv.DictReader(handle))
            self.assertEqual(len(artifact), 1)
            self.assertEqual(artifact[0]["table_status"], "skipped_not_requested")


if __name__ == "__main__":
    unittest.main()
