# Zero-lag confirmatory analysis (inside ppg-eeg)

This directory holds confirmatory Zero-Lag EEG–Heart Rate configs and
notes. It is **not** a separate git repository — the main repo is `ppg-eeg`.

## Separation from exploratory work

| | Exploratory (existing) | Confirmatory (this analysis) |
|--|------------------------|------------------------------|
| Code | `ppg_eeg/temporal_coupling/` Stages 0–4 | `ppg_eeg/temporal_coupling/confirmatory/` |
| Configs | `config.run.*.yaml`, `config.smoke.*.yaml` | `config.confirmatory.*.yaml` (this folder) |
| Results | `derivatives/run_*`, `derivatives/smoke_*` | `derivatives/confirmatory_temporal_coupling/` |

Raw data stays shared: `./data` (repo root).

## Configs in this folder

- `config.confirmatory.master.yaml` — frozen protocol
- `config.confirmatory.<dataset>.yaml` — full cohort
- `config.confirmatory.smoke.<dataset>.yaml` — smoke subsets

## Status

M1 config schema lives under `ppg_eeg/temporal_coupling/confirmatory/`.
Pipeline stages are not implemented yet.
