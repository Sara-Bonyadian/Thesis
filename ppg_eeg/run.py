"""Compatibility CLI: ``python -m ppg_eeg.run``. Prefer ``python -m ppg_eeg.core_eeg_ppg``."""

from ppg_eeg.core_eeg_ppg.run import *  # noqa: F403
from ppg_eeg.core_eeg_ppg.run import main

if __name__ == "__main__":
    main()
