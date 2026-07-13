# Zero-lag confirmatory analysis (inside ppg-eeg)

This folder holds **only confirmatory** Zero-Lag configs. It is not a separate git repo.

## Layout (use this map)

```text
zero-lag-reanalysis-repo/
  master.yaml              # frozen protocol (bands, lags, ZLPI contracts, dataset roles)
  datasets/                # full-cohort runs (one YAML per dataset)
    ds003838.yaml
    hiit.yaml
    ...
  smoke/                   # small subsets for smoke / M13
    ds003838.yaml
    ...
    hiit/                  # HIIT needs an extra beats file (see below)
      confirmatory.yaml    # who/what to analyze (pairing × PH/PS)
      beats.yaml           # Stage 0/1b only: photosensor PPG peaks
      verification.json    # stop rules / PPG lock (not a run config)
  README.md                # this file
```

| You want to… | Open / pass |
|--------------|-------------|
| Frozen protocol | `master.yaml` |
| Full HIIT cohort | `datasets/hiit.yaml` |
| HIIT smoke (confirmatory) | `smoke/hiit/confirmatory.yaml` |
| HIIT smoke (get `detected_peaks.csv`) | `smoke/hiit/beats.yaml` |
| Other dataset smoke | `smoke/<dataset>.yaml` |

## Separation from exploratory work

| | Exploratory | Confirmatory |
|--|-------------|--------------|
| Code | `ppg_eeg/temporal_coupling/` Stages 0–4 | `ppg_eeg/confirmatory/` |
| Configs | `exploratory-temporal-coupling/` | **this folder** |
| Results | `derivatives/run_*`, `derivatives/smoke_*` | `derivatives/confirmatory_temporal_coupling/` |

Raw data stays shared: `./data` (repo root).

## HIIT smoke (C-stage CLI)

Everything for HIIT smoke lives under `smoke/hiit/`.

```bash
# Protocol audit
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C0

# Beat peaks still come from exploratory Stage 1b (until peaks are under confirmatory control)
.venv/bin/python -m ppg_eeg.temporal_coupling \
  --config zero-lag-reanalysis-repo/smoke/hiit/beats.yaml --stage 0
.venv/bin/python -m ppg_eeg.temporal_coupling \
  --config zero-lag-reanalysis-repo/smoke/hiit/beats.yaml --stage 1b

# Instant HR from those peaks
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C1b \
  --peaks-root derivatives/smoke_hiit_m13b_temporal_coupling

# Full remaining pipeline
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage all \
  --peaks-root derivatives/smoke_hiit_m13b_temporal_coupling
```

Stages: `C0` `C1a` `C1b` `C1c` `C2` `C3` `C4` `C5` `C6` `C7` (or `all`).
Production ops remain: `python -m ppg_eeg.confirmatory --mode preflight|smoke|…`.

## Duration–lag–endpoint contracts

Canonical source: `ppg_eeg/confirmatory/duration_contracts.py`.

| Duration | Lag grid | Endpoint | Pool with ZLPI? |
|----------|----------|----------|-----------------|
| D240 | −60…+60 | `zlpi` | yes |
| D180 | −60…+60 | `zlpi` | yes |
| D120 | −30…+30 | MWPI | **no** |
| D60 | −20…+20 | SWPI | **no** |

## Status

M1–M12 modules + **C0–C7 stage CLI** (`ppg_eeg/confirmatory/run.py`) are implemented.
Production ops: `python -m ppg_eeg.confirmatory --mode preflight|smoke|…`.
