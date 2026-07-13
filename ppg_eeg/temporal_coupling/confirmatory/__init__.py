"""Confirmatory zero-lag EEG–heart-rate reanalysis."""

from .config import (
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
    load_dataset_config,
    load_master_config,
)
from .instant_hr import reconstruct_instant_hr, reconstruct_instant_hr_file
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
    "reconstruct_instant_hr",
    "reconstruct_instant_hr_file",
    "run_protocol_audit",
    "write_duration_eligibility",
    "write_protocol_audit",
]
