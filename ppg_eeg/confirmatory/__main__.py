"""CLI entrypoint: ``python -m ppg_eeg.confirmatory``.

Examples::

    python -m ppg_eeg.confirmatory --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C0
    python -m ppg_eeg.confirmatory --config … --stage C1b
    python -m ppg_eeg.confirmatory --config … --stage all
    python -m ppg_eeg.confirmatory --mode preflight
"""

from __future__ import annotations

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
