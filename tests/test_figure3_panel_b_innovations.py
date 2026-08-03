from __future__ import annotations

import math
import unittest

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI
from ppg_eeg.confirmatory.figures import _panel_b_cross_subject_and_innovation
from ppg_eeg.confirmatory.nulls import (
    SeriesUnit,
    ar1_innovations,
    circular_shift_series,
    compute_endpoint_index_from_series,
    deterministic_seed,
    valid_circular_shifts,
)


def _make_series(seed: int, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # AR(1)-like coupled processes with finite variance and non-trivial coupling.
    hr = np.zeros(n, dtype=float)
    eeg = np.zeros(n, dtype=float)
    hr_noise = rng.normal(0.0, 0.7, size=n)
    eeg_noise = rng.normal(0.0, 0.7, size=n)
    for t in range(1, n):
        hr[t] = 0.6 * hr[t - 1] + hr_noise[t]
        eeg[t] = 0.5 * eeg[t - 1] + 0.25 * hr[t - 1] + eeg_noise[t]
    # z-score each series.
    hr = (hr - float(np.mean(hr))) / float(np.std(hr, ddof=1))
    eeg = (eeg - float(np.mean(eeg))) / float(np.std(eeg, ddof=1))
    return hr, eeg


def _make_units() -> list[SeriesUnit]:
    units: list[SeriesUnit] = []
    participants = ("01", "02", "03")
    conditions = ("ph_post_rest", "ph_post_tetris")
    seed = 10
    for condition in conditions:
        for participant in participants:
            hr, eeg = _make_series(seed)
            seed += 1
            units.append(
                SeriesUnit(
                    dataset_id="hiit",
                    subject_id=participant,
                    task=condition,
                    condition=condition,
                    observation_id=f"hiit-{participant}-ph-{condition.split('_')[1]}-{condition.split('_')[2]}",
                    modality="ecg",
                    duration_s=240,
                    duration_role="primary",
                    band="theta",
                    power_representation="absolute_log10",
                    is_primary_representation=True,
                    pair="hr_x_theta_absolute_log10",
                    hr_z=hr,
                    eeg_z=eeg,
                )
            )
    return units


class TestFigure3PanelBInnovations(unittest.TestCase):
    def test_controls_share_estimand_scale_and_units(self) -> None:
        units = _make_units()
        obs_rows, _draw_rows, _diag_rows = _panel_b_cross_subject_and_innovation(
            units,
            n_null_draws=60,
        )
        self.assertEqual(len(obs_rows), len(units))
        eligible = [row for row in obs_rows if bool(row.get("eligibility_flag"))]
        self.assertTrue(eligible)
        for row in eligible:
            self.assertEqual(str(row.get("endpoint_name")), ENDPOINT_ZLPI)
            self.assertEqual(int(row.get("duration_s")), 240)
            self.assertEqual(str(row.get("representation")), "absolute_log10")
            for metric in (
                "correct_null_normalized_effect",
                "cross_subject_null_normalized_effect",
                "paired_specificity_contrast_delta_z",
                "innovation_null_normalized_effect",
            ):
                value = float(row.get(metric))
                self.assertTrue(math.isfinite(value), msg=f"{metric} non-finite")
            # Guard against pathological inflation in innovations scaling.
            self.assertLess(abs(float(row["innovation_null_normalized_effect"])), 50.0)
            self.assertEqual(int(row.get("n_cross_subject_draws")), 60)
            self.assertEqual(int(row.get("n_innovation_null_draws")), 60)

    def test_trimmed_innovations_avoid_structural_nan_shift_collapse(self) -> None:
        unit = _make_units()[0]
        identity = {
            "dataset_id": unit.dataset_id,
            "subject_id": unit.subject_id,
            "task": unit.task,
            "condition": unit.condition,
            "observation_id": unit.observation_id,
        }
        draws = 80
        seed = deterministic_seed(
            unit.observation_id,
            f"{unit.duration_s}|{unit.band}|{unit.power_representation}|innovation",
            "circular_shift",
        )
        rng = np.random.default_rng(seed)

        hr_inn_full = ar1_innovations(unit.hr_z)
        eeg_inn_full = ar1_innovations(unit.eeg_z)
        shifts_full = valid_circular_shifts(len(eeg_inn_full))
        finite_full = 0
        for _ in range(draws):
            shift = int(rng.choice(shifts_full))
            eeg_shift = circular_shift_series(eeg_inn_full, shift)
            metrics = compute_endpoint_index_from_series(
                hr_inn_full,
                eeg_shift,
                duration_s=unit.duration_s,
                identity=identity,
                band=unit.band,
                power_representation=unit.power_representation,
                duration_role=unit.duration_role,
                is_primary_representation=unit.is_primary_representation,
                pair=unit.pair,
            )
            if math.isfinite(float(metrics.get("endpoint_index"))):
                finite_full += 1

        rng = np.random.default_rng(seed)
        hr_inn = hr_inn_full[1:]
        eeg_inn = eeg_inn_full[1:]
        shifts_trim = valid_circular_shifts(len(eeg_inn))
        finite_trim = 0
        for _ in range(draws):
            shift = int(rng.choice(shifts_trim))
            eeg_shift = circular_shift_series(eeg_inn, shift)
            metrics = compute_endpoint_index_from_series(
                hr_inn,
                eeg_shift,
                duration_s=unit.duration_s,
                identity=identity,
                band=unit.band,
                power_representation=unit.power_representation,
                duration_role=unit.duration_role,
                is_primary_representation=unit.is_primary_representation,
                pair=unit.pair,
            )
            if math.isfinite(float(metrics.get("endpoint_index"))):
                finite_trim += 1

        self.assertLess(finite_full, draws)
        self.assertEqual(finite_trim, draws)


if __name__ == "__main__":
    unittest.main()
