"""EEG–PPG analysis package.

Layout (by analysis):

```
ppg_eeg/
  datasets/          # shared BIDS/dataset adapters
  core_eeg_ppg/      # core EEG–PPG correlations  → configs: core-eeg-ppg/
  temporal_coupling/ # exploratory Stages 0–4     → configs: exploratory-temporal-coupling/
  confirmatory/      # confirmatory zero-lag      → configs: zero-lag-reanalysis-repo/
```
"""

from .core_eeg_ppg import (
    PipelineConfig,
    load_config,
    plot_correlation_heatmap,
    run_pipeline,
    run_stage1,
    run_stage2_from_base_csvs,
    run_two_stage_pipeline,
    write_artifacts,
)

__all__ = [
    "PipelineConfig",
    "load_config",
    "plot_correlation_heatmap",
    "run_pipeline",
    "run_stage1",
    "run_stage2_from_base_csvs",
    "run_two_stage_pipeline",
    "write_artifacts",
]
