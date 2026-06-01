from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np

from ppg_eeg.config import PathsConfig, PipelineConfig, PpgConfig
from ppg_eeg.datasets.base import CanonicalObservation
from ppg_eeg.features_core import CORE_EEG_FEATURES, CORE_PPG_FEATURES, _read_raw, extract_core_feature_tables


def _test_cfg() -> PipelineConfig:
    return PipelineConfig(
        dataset_ids=["hiit"],
        dataset_id="hiit",
        paths=PathsConfig(raw_root=Path("."), out_root=Path("./derivatives")),
        subjects=[],
        tasks=[],
        conditions=[],
        sessions=[],
        ppg=PpgConfig(start_time_s=0.0, end_time_s=9.0),
    )


def _synthetic_raw() -> mne.io.BaseRaw:
    sfreq = 256.0
    duration_s = 10.0
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples, dtype=float) / sfreq

    eeg_ch = ["F3", "F4", "F7", "F8", "Fz", "FCz", "Cz", "P3", "P4", "O1", "O2", "CP1", "CP2", "POz", "Oz"]
    ch_names = [*eeg_ch, "photosensor"]
    ch_types = ["eeg"] * len(eeg_ch) + ["misc"]

    data = []
    for idx, _ch in enumerate(eeg_ch):
        # 10 Hz alpha + 20 Hz beta component in Volts
        sig = (8e-6 * np.sin(2 * np.pi * 10 * t + idx * 0.1)) + (3e-6 * np.sin(2 * np.pi * 20 * t))
        data.append(sig)

    # Synthetic PPG-like waveform
    ppg = 0.7 * np.sin(2 * np.pi * 1.2 * t) + 0.25 * np.sin(2 * np.pi * 2.4 * t)
    data.append(ppg)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(np.array(data), info, verbose=False)


class TestFeaturesCore(unittest.TestCase):
    def test_read_raw_eeglab_v73_fallback(self) -> None:
        fake_eeg = {
            "data": np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=float),
            "srate": 1000.0,
            "chanlocs": {"labels": ["PPG", "ECG"]},
        }
        with (
            patch(
                "ppg_eeg.features_core.mne.io.read_raw_eeglab",
                side_effect=NotImplementedError("Please use HDF reader for matlab v7.3 files, e.g. h5py"),
            ),
            patch("pymatreader.read_mat", return_value={"EEG": fake_eeg}),
        ):
            raw = _read_raw(Path("/tmp/fake.set"), "eeglab")

        self.assertEqual(raw.info["sfreq"], 1000.0)
        self.assertEqual(raw.ch_names, ["PPG", "ECG"])
        self.assertEqual(raw.get_channel_types(), ["misc", "ecg"])
        np.testing.assert_allclose(raw.get_data()[0], np.array([1.0, 2.0, 3.0]) * 1e-6)

    def test_extract_core_feature_tables_shape(self) -> None:
        cfg = _test_cfg()
        obs = CanonicalObservation(
            dataset_id="hiit",
            observation_id="hiit-01-ph-pre-rest",
            subject_id="01",
            task_label="rest",
            condition_label="ph_pre_rest",
            eeg_path=Path("/tmp/fake.vhdr"),
            eeg_format="brainvision",
            ppg_source="embedded_eeg",
            session_label="ph",
            modality="ph",
            timepoint="pre",
            state="rest",
        )

        raw = _synthetic_raw()
        with patch("ppg_eeg.features_core._read_raw", side_effect=lambda *_args, **_kwargs: raw.copy()):
            result = extract_core_feature_tables([obs], cfg)

        self.assertEqual(len(result.eeg_features), 1)
        self.assertEqual(len(result.ppg_features), 1)
        self.assertEqual(len(result.merged_features), 1)
        self.assertEqual(result.eeg_features.loc[0, "eeg_error"], "ok")
        self.assertEqual(result.ppg_features.loc[0, "ppg_error"], "ok")
        for feature in CORE_EEG_FEATURES:
            self.assertIn(feature, result.eeg_features.columns)
        for feature in CORE_PPG_FEATURES:
            self.assertIn(feature, result.ppg_features.columns)
            self.assertIn(feature, result.merged_features.columns)


if __name__ == "__main__":
    unittest.main()
