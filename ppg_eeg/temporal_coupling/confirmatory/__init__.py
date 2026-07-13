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
from .endpoints import (
    compute_endpoints_from_curves,
    run_confirmatory_endpoints,
    write_endpoint_outputs,
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
from .artifact_controls import (
    run_artifact_controls,
    run_confirmatory_artifact_controls,
    write_artifact_control_outputs,
)
from .figures import generate_confirmatory_figures
from .group_tables import (
    build_group_tables,
    run_confirmatory_group_tables,
    write_group_table_outputs,
)
from .inference import (
    run_confirmatory_inference,
    run_confirmatory_inference_from_dir,
    write_inference_outputs,
)
from .manifest import build_figure_source_manifest, build_run_manifest
from .nulls import (
    run_confirmatory_nulls,
    run_null_battery,
    write_null_outputs,
)
from .peak_model import (
    compute_peak_fits_from_curves,
    run_confirmatory_peak_fits,
    write_peak_fit_outputs,
)
from .production import (
    build_synthetic_aligned_tables,
    primary_blockers_summary,
    run_production,
    run_production_preflight,
)
from .protocol_audit import (
    run_protocol_audit,
    write_duration_eligibility,
    write_protocol_audit,
)
from .report import generate_confirmatory_report, run_confirmatory_reporting

__all__ = [
    "ConfirmatoryDatasetConfig",
    "ConfirmatoryMasterConfig",
    "DURATION_ANALYSIS_CONTRACTS",
    "load_dataset_config",
    "load_master_config",
    "build_figure_source_manifest",
    "build_group_tables",
    "build_run_manifest",
    "compute_endpoints_from_curves",
    "compute_multitaper_power",
    "compute_peak_fits_from_curves",
    "compute_signed_lag_curves",
    "contract_for_duration",
    "extract_multitaper_file",
    "extract_multitaper_from_raw",
    "generate_confirmatory_figures",
    "generate_confirmatory_report",
    "harmonize_observation",
    "build_synthetic_aligned_tables",
    "primary_blockers_summary",
    "reconstruct_instant_hr",
    "reconstruct_instant_hr_file",
    "run_artifact_controls",
    "run_confirmatory_artifact_controls",
    "run_confirmatory_correlations",
    "run_confirmatory_endpoints",
    "run_confirmatory_group_tables",
    "run_confirmatory_inference",
    "run_confirmatory_inference_from_dir",
    "run_confirmatory_nulls",
    "run_confirmatory_peak_fits",
    "run_confirmatory_reporting",
    "run_null_battery",
    "run_production",
    "run_production_preflight",
    "run_protocol_audit",
    "standard_zlpi_pool_durations",
    "write_artifact_control_outputs",
    "write_correlation_outputs",
    "write_duration_eligibility",
    "write_endpoint_outputs",
    "write_group_table_outputs",
    "write_harmonize_outputs",
    "write_inference_outputs",
    "write_null_outputs",
    "write_peak_fit_outputs",
    "write_protocol_audit",
]
