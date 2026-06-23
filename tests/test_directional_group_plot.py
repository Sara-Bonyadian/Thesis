from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ppg_eeg.temporal_coupling.directional_group_plot import (
    bootstrap_median_ci,
    plot_directional_median_bootstrap_grid,
)


class TestDirectionalGroupPlot(unittest.TestCase):
    def test_bootstrap_median_ci(self) -> None:
        values = np.array(
            [
                [-0.1, 0.0, 0.1],
                [0.0, 0.2, 0.3],
                [0.1, 0.1, 0.2],
            ],
            dtype=float,
        )
        median, lower, upper = bootstrap_median_ci(values, n_boot=500, ci_percent=95, seed=0)
        self.assertAlmostEqual(float(median[1]), 0.1)
        self.assertTrue(float(lower[1]) <= float(median[1]) <= float(upper[1]))

    def test_plot_directional_grid_writes_file(self) -> None:
        lags = np.arange(-30, 35, 5, dtype=float)
        curve_rows: list[dict[str, object]] = []
        peak_rows: list[dict[str, object]] = []
        for obs_id, direction, offset in (
            ("obs-a", "cardiac_leads", 0.2),
            ("obs-b", "cardiac_leads", 0.15),
            ("obs-c", "eeg_leads", -0.2),
            ("obs-d", "eeg_leads", -0.15),
        ):
            peak_lag = 10.0 if direction == "cardiac_leads" else -10.0
            peak_rows.append(
                {
                    "observation_id": obs_id,
                    "pair": "hr__theta",
                    "task": "rest",
                    "condition": "pre_rest",
                    "peak_direction": direction,
                    "peak_lag_s": peak_lag,
                }
            )
            for lag in lags:
                curve_rows.append(
                    {
                        "observation_id": obs_id,
                        "pair": "hr__theta",
                        "task": "rest",
                        "condition": "pre_rest",
                        "lag_s": lag,
                        "r": offset + 0.01 * lag,
                    }
                )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "directional.png"
            plot_directional_median_bootstrap_grid(
                pd.DataFrame(curve_rows),
                pd.DataFrame(peak_rows),
                out_path,
                lag_min=-30,
                lag_max=30,
                n_boot=200,
                seed=1,
                partition="pre_rest",
            )
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
