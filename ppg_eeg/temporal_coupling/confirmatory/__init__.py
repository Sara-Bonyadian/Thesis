"""Confirmatory zero-lag EEG–heart-rate reanalysis."""

from .config import (
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
    load_dataset_config,
    load_master_config,
)
from .correlation import (
    compute_signed_lag_curves,
    run_confirmatory_correlations,
    write_correlation_outputs,
)
from .duration_contracts import (
    DURATION_ANALYSIS_CONTRACTS,
    contract_for_duration,
    standard_zlpi_pool_durations,
)
from .harmonize import (
    harmonize_observation,
    write_harmonize_outputs,
)
from .instant_hr import reconstruct_instant_hr, reconstruct_instant_hr_file
from .multitaper_power import (
    compute_multitaper_power,
    extract_multitaper_file,
    extract_multitaper_from_raw,
)
from .protocol_audit import (
    run_protocol_audit,
    write_duration_eligibility,
    write_protocol_audit,
)

__all__ = [
    "ConfirmatoryDatasetConfig",
    "ConfirmatoryMasterConfig",
    "DURATION_ANALYSIS_CONTRACTS",
    "load_dataset_config",
    "load_master_config",
    "compute_multitaper_power",
    "compute_signed_lag_curves",
    "contract_for_duration",
    "extract_multitaper_file",
    "extract_multitaper_from_raw",
    "harmonize_observation",
    "reconstruct_instant_hr",
    "reconstruct_instant_hr_file",
    "run_confirmatory_correlations",
    "run_protocol_audit",
    "standard_zlpi_pool_durations",
    "write_correlation_outputs",
    "write_duration_eligibility",
    "write_harmonize_outputs",
    "write_protocol_audit",
]
