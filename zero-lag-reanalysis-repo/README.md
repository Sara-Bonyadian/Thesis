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
| Code | `ppg_eeg/temporal_coupling/` Stages 0–4 | `ppg_eeg/temporal_coupling/confirmatory/` |
| Configs | repo-root `config.run.*.yaml`, `config.smoke.*.yaml` | **this folder** |
| Results | `derivatives/run_*`, `derivatives/smoke_*` | `derivatives/confirmatory_temporal_coupling/` |

Raw data stays shared: `./data` (repo root).

## HIIT smoke (one folder)

Everything for M13b lives under `smoke/hiit/`. Do not hunt for `m13b*.json` or root `config.smoke.hiit.m13b.*`.

```bash
# 1) confirmatory ops / preflight
.venv/bin/python -m ppg_eeg.temporal_coupling.confirmatory.hiit_smoke_m13b --stage preflight

# 2) beat peaks (exploratory Stage 0 then 1b)
.venv/bin/python -m ppg_eeg.temporal_coupling \
  --config zero-lag-reanalysis-repo/smoke/hiit/beats.yaml --stage 0
.venv/bin/python -m ppg_eeg.temporal_coupling \
  --config zero-lag-reanalysis-repo/smoke/hiit/beats.yaml --stage 1b
```

## Duration–lag–endpoint contracts

Canonical source: `ppg_eeg/temporal_coupling/confirmatory/duration_contracts.py`.

| Duration | Lag grid | Endpoint | Pool with ZLPI? |
|----------|----------|----------|-----------------|
| D240 | −60…+60 | `zlpi` | yes |
| D180 | −60…+60 | `zlpi` | yes |
| D120 | −30…+30 | MWPI | **no** |
| D60 | −20…+20 | SWPI | **no** |

## Status

M1–M12 confirmatory modules are implemented. Production preflight/smoke: `production.py`. Unified C0–C7 CLI still planned (see project plan §3).
