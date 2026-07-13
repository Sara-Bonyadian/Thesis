"""CLI entrypoint: ``python -m ppg_eeg.confirmatory``."""

from __future__ import annotations

from .production import main

if __name__ == "__main__":
    raise SystemExit(main())
