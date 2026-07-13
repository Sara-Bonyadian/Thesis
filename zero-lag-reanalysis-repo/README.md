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

## Duration–lag–endpoint contracts

Canonical source: `ppg_eeg/temporal_coupling/confirmatory/duration_contracts.py`.

| Duration | Lag grid | Endpoint | Distant flanks | Shoulders | Standard ZLPI? | Pool with ZLPI? |
|----------|----------|----------|----------------|-----------|----------------|-----------------|
| D240 | −60…+60 (121) | `zlpi` | 20–60 s | 5–15 s | yes | yes |
| D180 | −60…+60 (121) | `zlpi` | 20–60 s | 5–15 s | yes | yes |
| D120 | −30…+30 (61) | `mid_window_proximal_index` (MWPI) | 20–30 s | 5–15 s | **no** | **no** |
| D60 | −20…+20 (41) | `short_window_proximal_index` (SWPI) | 10–20 s | 5–15 s | **no** | **no** |

Lag correlations use common-support cropping so `n_overlap` is constant across lags.
Fully finite segments yield overlap 120 / 60 / 60 / 20 for D240 / D180 / D120 / D60.

## Status

Confirmatory package modules through M5 live under
`ppg_eeg/temporal_coupling/confirmatory/`. Fisher-z / endpoint numerics are M6+.
