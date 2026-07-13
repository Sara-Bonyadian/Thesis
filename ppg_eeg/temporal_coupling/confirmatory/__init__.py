"""Confirmatory zero-lag EEG–heart-rate reanalysis."""

from .config import (
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
    load_dataset_config,
    load_master_config,
)
from .protocol_audit import (
    run_protocol_audit,
    write_duration_eligibility,
    write_protocol_audit,
)

__all__ = [
    "ConfirmatoryDatasetConfig",
    "ConfirmatoryMasterConfig",
    "load_dataset_config",
    "load_master_config",
    "run_protocol_audit",
    "write_duration_eligibility",
    "write_protocol_audit",
]
