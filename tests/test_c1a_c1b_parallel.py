"""Tests for C1a/C1b parallelization, DPSS cache, debug_plot, and resume."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from ppg_eeg.confirmatory.multitaper_power import (
    COMPLETE_MARKER_FILENAME as C1A_COMPLETE,
    FEATURES_FILENAME as MULTITAPER_FEATURES,
    QC_FILENAME as MULTITAPER_QC,
    clear_dpss_cache,
    compute_multitaper_power,
    run_confirmatory_multitaper,
    write_multitaper_outputs,
)
from ppg_eeg.confirmatory.parallel_util import resolve_c1a_n_jobs, resolve_n_jobs
from ppg_eeg.confirmatory.peak_detection import (
    COMPLETE_MARKER_FILENAME as C1B_COMPLETE,
    PEAKS_FILENAME,
    detect_peaks_for_observation,
    run_confirmatory_c1b,
    tc_config_for_peak_detection,
)
from ppg_eeg.datasets import CanonicalObservation
from ppg_eeg.temporal_coupling.cardiac_detectors import CHANNEL_PREVIEW_FILENAME
from ppg_eeg.confirmatory.instant_hr import FEATURES_FILENAME as INSTANT_HR_FEATURES


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sine_stack(*, sfreq: float = 100.0, duration_s: float = 6.0) -> np.ndarray:
    time = np.arange(int(round(sfreq * duration_s)), dtype=float) / sfreq
    return np.vstack(
        [
            np.sin(2 * np.pi * 5.0 * time),
            np.sin(2 * np.pi * 10.0 * time),
            np.sin(2 * np.pi * 20.0 * time),
        ]
    )


class TestResolveJobs(unittest.TestCase):
    def test_resolve_n_jobs(self) -> None:
        self.assertGreaterEqual(resolve_n_jobs(-1), 1)
        self.assertEqual(resolve_n_jobs(1), 1)
        self.assertEqual(resolve_n_jobs(3), 3)

    def test_c1a_ram_cap_at_least_one(self) -> None:
        self.assertGreaterEqual(resolve_c1a_n_jobs(1), 1)
        self.assertGreaterEqual(resolve_c1a_n_jobs(-1), 1)

    def test_c1a_large_payload_forces_serial(self) -> None:
        # ~2.5 GB on disk × 8 ≈ 20 GB peak estimate → must be serial on 16 GB host
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            huge = tmp / "huge.eeg"
            # Sparse file: seek+write makes apparent size large without filling disk
            with huge.open("wb") as handle:
                handle.seek(2_500_000_000 - 1)
                handle.write(b"\0")
            header = tmp / "huge.vhdr"
            header.write_text("placeholder", encoding="utf-8")
            workers = resolve_c1a_n_jobs(-1, eeg_paths=[header])
            self.assertEqual(workers, 1)


class TestDpssCacheIdentity(unittest.TestCase):
    def test_cache_matches_no_cache(self) -> None:
        data = _sine_stack()
        clear_dpss_cache()
        cached = compute_multitaper_power(
            data,
            sfreq=100.0,
            ch_names=["a", "b", "c"],
            line_frequency_hz=None,
            apply_line_notch=False,
            cache_dpss=True,
        )
        clear_dpss_cache()
        uncached = compute_multitaper_power(
            data,
            sfreq=100.0,
            ch_names=["a", "b", "c"],
            line_frequency_hz=None,
            apply_line_notch=False,
            cache_dpss=False,
        )
        self.assertEqual(len(cached.features), len(uncached.features))
        for left, right in zip(cached.features, uncached.features, strict=True):
            self.assertEqual(left.absolute_power, right.absolute_power)
            self.assertEqual(left.absolute_log10_power, right.absolute_log10_power)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_multitaper_outputs(cached, root / "cached")
            write_multitaper_outputs(uncached, root / "uncached")
            self.assertEqual(
                _file_sha(root / "cached" / MULTITAPER_FEATURES),
                _file_sha(root / "uncached" / MULTITAPER_FEATURES),
            )
            self.assertEqual(
                _file_sha(root / "cached" / MULTITAPER_QC),
                _file_sha(root / "uncached" / MULTITAPER_QC),
            )


class TestC1aOrchestration(unittest.TestCase):
    def _synthetic_obs(self, tmp: Path, n: int = 3) -> list[CanonicalObservation]:
        tmp.mkdir(parents=True, exist_ok=True)
        obs: list[CanonicalObservation] = []
        for i in range(n):
            eeg = tmp / f"eeg_{i}.vhdr"
            eeg.write_text("placeholder", encoding="utf-8")
            oid = f"synth-{i:02d}"
            obs.append(
                CanonicalObservation(
                    dataset_id="synth",
                    observation_id=oid,
                    subject_id=f"s{i}",
                    task_label="rest",
                    condition_label="rest",
                    eeg_path=eeg,
                    eeg_format="brainvision",
                    ppg_source="embedded_eeg",
                )
            )
        return obs

    def test_resume_and_failure_no_complete_marker(self) -> None:
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            observations = self._synthetic_obs(tmp / "raw", n=3)

            def fake_extract(eeg_path, eeg_format, output_dir, **kwargs):
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                identity = kwargs.get("identity") or {}
                oid = str(identity.get("observation_id", "unknown"))
                (out / MULTITAPER_FEATURES).write_text(f"features,{oid}\n", encoding="utf-8")
                (out / MULTITAPER_QC).write_text(f"qc,{oid}\n", encoding="utf-8")
                return out / MULTITAPER_FEATURES, out / MULTITAPER_QC

            with mock.patch(
                "ppg_eeg.confirmatory.multitaper_power.extract_multitaper_file",
                side_effect=fake_extract,
            ):
                serial = tmp / "serial"
                r1 = run_confirmatory_multitaper(
                    observations, serial, n_jobs=1, progress=False
                )
                self.assertTrue(r1["complete"])
                self.assertTrue((serial / C1A_COMPLETE).is_file())
                hashes = {
                    obs.observation_id: _file_sha(
                        serial / obs.observation_id / MULTITAPER_FEATURES
                    )
                    for obs in observations
                }

                (serial / C1A_COMPLETE).unlink()
                r2 = run_confirmatory_multitaper(
                    observations, serial, n_jobs=1, progress=False
                )
                self.assertEqual(r2["n_ok"], 3)
                self.assertTrue((serial / C1A_COMPLETE).is_file())
                for obs in observations:
                    self.assertEqual(
                        hashes[obs.observation_id],
                        _file_sha(serial / obs.observation_id / MULTITAPER_FEATURES),
                    )

            fail_root = tmp / "fail"
            call_count = {"n": 0}

            def flaky_extract(eeg_path, eeg_format, output_dir, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("boom")
                return fake_extract(eeg_path, eeg_format, output_dir, **kwargs)

            with mock.patch(
                "ppg_eeg.confirmatory.multitaper_power.extract_multitaper_file",
                side_effect=flaky_extract,
            ):
                result = run_confirmatory_multitaper(
                    observations, fail_root, n_jobs=1, progress=False
                )
                self.assertGreater(result["n_error"], 0)
                self.assertFalse((fail_root / C1A_COMPLETE).is_file())
                self.assertFalse(bool(result["complete"]))


class TestSmokeIdentity(unittest.TestCase):
    """Live smoke-data tests (skipped when raw/derivatives unavailable)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[1]
        from ppg_eeg.confirmatory.config import load_dataset_config, load_master_config
        from ppg_eeg.confirmatory.production import master_config_path
        from ppg_eeg.confirmatory.protocol_audit import observations_from_dataset_config

        cls.master = load_master_config(
            master_config_path(cls.repo / "zero-lag-reanalysis-repo")
        )
        cls.dataset = load_dataset_config(
            cls.repo
            / "zero-lag-reanalysis-repo"
            / "smoke"
            / "hiit"
            / "confirmatory.yaml",
            master=cls.master,
        )
        try:
            cls.observations = observations_from_dataset_config(cls.dataset)
        except FileNotFoundError:
            cls.observations = []

    def test_c1a_serial_equals_parallel(self) -> None:
        if len(self.observations) < 2:
            self.skipTest("HIIT smoke observations unavailable")
        subset = sorted(self.observations, key=lambda o: o.observation_id)[:2]
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            serial = tmp / "serial"
            parallel = tmp / "parallel"
            run_confirmatory_multitaper(subset, serial, n_jobs=1, progress=False)
            run_confirmatory_multitaper(subset, parallel, n_jobs=2, progress=False)
            for obs in subset:
                for name in (MULTITAPER_FEATURES, MULTITAPER_QC):
                    self.assertEqual(
                        _file_sha(serial / obs.observation_id / name),
                        _file_sha(parallel / obs.observation_id / name),
                        msg=f"{obs.observation_id}/{name}",
                    )

    def test_c1b_serial_equals_parallel_and_debug_plot(self) -> None:
        if len(self.observations) < 2:
            self.skipTest("HIIT smoke observations unavailable")
        self.assertFalse(self.dataset.cardiac.debug_plot)
        subset = sorted(self.observations, key=lambda o: o.observation_id)[:2]
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            serial = tmp / "serial"
            parallel = tmp / "parallel"
            r1 = run_confirmatory_c1b(
                subset, self.dataset, self.master, serial, n_jobs=1, progress=False
            )
            r2 = run_confirmatory_c1b(
                subset, self.dataset, self.master, parallel, n_jobs=2, progress=False
            )
            self.assertTrue(r1["complete"])
            self.assertTrue(r2["complete"])
            self.assertTrue((serial / C1B_COMPLETE).is_file())
            for obs in subset:
                for name in (PEAKS_FILENAME, INSTANT_HR_FEATURES):
                    self.assertEqual(
                        _file_sha(serial / obs.observation_id / name),
                        _file_sha(parallel / obs.observation_id / name),
                        msg=f"{obs.observation_id}/{name}",
                    )
                self.assertFalse(
                    (serial / obs.observation_id / CHANNEL_PREVIEW_FILENAME).is_file()
                )

            # debug_plot true adds PNG only; peaks stay identical
            dataset_on = replace(
                self.dataset,
                cardiac=replace(self.dataset.cardiac, debug_plot=True),
            )
            plotted = tmp / "plotted"
            obs = subset[0]
            cfg = tc_config_for_peak_detection(dataset_on, self.master, out_root=plotted)
            detect_peaks_for_observation(obs, cfg, plotted / obs.observation_id)
            self.assertTrue(
                (plotted / obs.observation_id / CHANNEL_PREVIEW_FILENAME).is_file()
            )
            self.assertEqual(
                _file_sha(serial / obs.observation_id / PEAKS_FILENAME),
                _file_sha(plotted / obs.observation_id / PEAKS_FILENAME),
            )

    def test_c1b_failure_no_complete_marker(self) -> None:
        if len(self.observations) < 2:
            self.skipTest("HIIT smoke observations unavailable")
        subset = sorted(self.observations, key=lambda o: o.observation_id)[:2]
        with TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            with mock.patch(
                "ppg_eeg.confirmatory.peak_detection.process_one_c1b_observation",
                side_effect=RuntimeError("boom"),
            ):
                result = run_confirmatory_c1b(
                    subset, self.dataset, self.master, tmp, n_jobs=1, progress=False
                )
            self.assertGreater(result["n_error"], 0)
            self.assertFalse((tmp / C1B_COMPLETE).is_file())
            self.assertFalse(bool(result["complete"]))


if __name__ == "__main__":
    unittest.main()
