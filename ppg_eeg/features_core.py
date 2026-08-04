"""Compatibility re-export. Prefer ``ppg_eeg.core_eeg_ppg.features_core``."""

from ppg_eeg.core_eeg_ppg.features_core import *  # noqa: F403
from ppg_eeg.core_eeg_ppg.features_core import (  # noqa: F401
    CORE_EEG_FEATURES,
    CORE_PPG_FEATURES,
    _read_raw,
    extract_core_feature_tables,
)