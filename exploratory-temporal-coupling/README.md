# Exploratory temporal coupling (Stages 0–4)

Configs for the **exploratory** EEG–cardiac lag / peak-search pipeline.  
Confirmatory zero-lag analysis lives in `zero-lag-reanalysis-repo/`.

## Layout

```text
exploratory-temporal-coupling/
  datasets/     # full-cohort runs (one YAML per dataset / HIIT variant)
  smoke/        # small subject subsets
  README.md
```

| You want to… | Open / pass |
|--------------|-------------|
| Full ds003838 run | `datasets/ds003838.yaml` |
| HIIT 8-condition run | `datasets/hiit.yaml` |
| HIIT 40 PPG run | `datasets/hiit_40_ppg.yaml` |
| Smoke (e.g. ds003838) | `smoke/ds003838.yaml` |

## Separation from other analyses

| | This folder | Confirmatory | Core EEG–PPG |
|--|-------------|--------------|--------------|
| Code | `ppg_eeg/temporal_coupling/` Stages 0–4 | `ppg_eeg/confirmatory/` | `ppg_eeg/core_eeg_ppg/` |
| Configs | **here** | `zero-lag-reanalysis-repo/` | `core-eeg-ppg/` |
| Results | `derivatives/run_*`, `derivatives/smoke_*` | `derivatives/confirmatory_temporal_coupling/` | `derivatives/` (base) |

Raw data stays shared: `./data` (repo root). Run commands from the **repo root**.

## Example

```bash
.venv/bin/python -m ppg_eeg.temporal_coupling \
  --config exploratory-temporal-coupling/smoke/ds003838.yaml --stage all
```
