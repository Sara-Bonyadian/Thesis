"""Compatibility CLI: ``python -m ppg_eeg.run_batched``."""

from ppg_eeg.core_eeg_ppg.run_batched import *  # noqa: F403
from ppg_eeg.core_eeg_ppg.run_batched import main

if __name__ == "__main__":
    raise SystemExit(main())
