from __future__ import annotations

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
    CONTROL_ECG_VS_PPG,
    CONTROL_MOTION,
    CONTROL_RESPIRATION,
    DURATION_SENSITIVITY_FILENAME,
    MODALITY_COMPARISON_FILENAME,
    PRIMARY_CONTROL_ID,
    SENSITIVITY_QC_FILENAME,
    SENSITIVITY_RESULTS_FILENAME,
    SPECIFICATION_MATRIX_FILENAME,
    beat_density,
    broadband_residualize_log_power,
    compare_matched_modalities,
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


class TestMatchedModalityComparison(unittest.TestCase):
    def test_compares_only_matched_ecg_ppg_pairs(self) -> None:
        subjects = [
            _subject(
                participant_id="p01",
                condition="rest",
                modality="ecg",
                endpoint_index=0.40,
            ),
            _subject(
                participant_id="p01",
                condition="rest",
                modality="ppg",
                endpoint_index=0.30,
            ),
            _subject(
                participant_id="p02",
                condition="rest",
                modality="ecg",
                endpoint_index=0.50,
            ),
            # p02 PPG missing → unmatched
        ]
        rows = compare_matched_modalities(subjects)
        matched = [r for r in rows if r["matched"]]
        unmatched = [r for r in rows if not r["matched"]]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(float(matched[0]["delta_ecg_minus_ppg"]), 0.10, places=12)
        self.assertTrue(unmatched)
        self.assertEqual(unmatched[0]["status"], "control_unavailable")

        result = run_artifact_controls(subjects, [])
        modality_effects = [
            r for r in result.sensitivity_rows if r["control_id"] == CONTROL_ECG_VS_PPG
        ]
        self.assertTrue(modality_effects)
        self.assertFalse(modality_effects[0]["is_primary_analysis"])
        self.assertFalse(modality_effects[0]["can_rescue_primary"])


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
        # Specification matrix includes every control id.
        control_ids = {r["control_id"] for r in result.specification_rows}
        self.assertIn(PRIMARY_CONTROL_ID, control_ids)
        self.assertIn(CONTROL_D120, control_ids)
        self.assertIn(CONTROL_CARDIAC_FIELD, control_ids)


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
                MODALITY_COMPARISON_FILENAME,
                DURATION_SENSITIVITY_FILENAME,
                SPECIFICATION_MATRIX_FILENAME,
                SENSITIVITY_QC_FILENAME,
            }
            self.assertEqual({p.name for p in paths.values()}, expected)
            for path in paths.values():
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
