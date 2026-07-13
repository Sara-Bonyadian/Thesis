"""Compatibility shim. Prefer ``ppg_eeg.confirmatory``.

Kept so ``python -m ppg_eeg.temporal_coupling.confirmatory`` and old
``from ppg_eeg.temporal_coupling.confirmatory import …`` continue to work.
Submodule imports (e.g. ``…confirmatory.instant_hr``) should use
``ppg_eeg.confirmatory`` directly.
"""

from ppg_eeg.confirmatory import *  # noqa: F403
from ppg_eeg.confirmatory import __all__  # noqa: F401
