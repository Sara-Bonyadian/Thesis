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
    hiit/                  # HIIT package
      confirmatory.yaml    # full workflow config (selection + cardiac peaks)
      verification.json    # stop rules / PPG lock (not a run config)
  README.md                # this file
```

| You want to… | Open / pass |
|--------------|-------------|
| Frozen protocol | `master.yaml` |
| Full HIIT cohort | `datasets/hiit.yaml` |
| HIIT smoke | `smoke/hiit/confirmatory.yaml` |
| Other dataset smoke | `smoke/<dataset>.yaml` |

## Separation from exploratory work

| | Exploratory | Confirmatory |
|--|-------------|--------------|
| Code | `ppg_eeg/temporal_coupling/` Stages 0–4 | `ppg_eeg/confirmatory/` |
| Configs | `exploratory-temporal-coupling/` | **this folder** |
| Results | `derivatives/run_*`, `derivatives/smoke_*` | `derivatives/confirmatory_temporal_coupling/` |

Raw data stays shared: `./data` (repo root). Exploratory Stage 0/1b is **not** required for confirmatory runs.

## HIIT smoke (C-stage CLI)

Everything for HIIT smoke lives under `smoke/hiit/`.

```bash
# C0: raw-data audit + pairing + duration eligibility
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C0

# C1b: peak detection + cardiac QC + instantaneous HR (under confirmatory C1b/)
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C1b

# Remaining stages (or full pipeline after C0)
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage all
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

## Robustness design (finalized)

### Default confirmatory robustness (always on in C4/C6)

| Analysis | Stage | Notes |
|----------|-------|-------|
| Temporal surrogate nulls | C4 | Circular shift, phase randomization, block shuffle, cross-subject mismatch, AR(1) innovations |
| Broadband EEG residualization | C6 | Core representation sensitivity; cannot rescue primary |
| Duration sensitivity | C6 | D180 ZLPI; D120 MWPI; D60 SWPI; cannot rescue primary |

### Optional dataset-conditional artifact controls (off by default)

Enable only with an explicit CLI flag **and** when required signals exist:

```bash
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml \
  --stage C6 --optional-artifact-controls
```

Optional controls: cardiac-field/QRS, motion/EOG/EMG, respiration, mean HR, beat count/density, eye state.

### Cardiac modality

Each dataset config / `PROTOCOL_SPECS` chooses **one** primary cardiac source (ECG or PPG) for instantaneous HR derivation. Dual-modality comparison tables are not part of the confirmatory design.

### Figure 3 (manuscript layout — finalized)

Main figure is **four panels (A–D)**, not a six-panel redesign:

| Panel | Content |
|-------|---------|
| A | Circular-shift null Δ forest (θ spotlight); other C4 nulls in supplements |
| B | Nested duration sensitivity (ZLPI / MWPI / SWPI; non-rescuing) |
| C | Broadband residualization |
| D | Specification matrix (+ optional controls only if enabled) |

Manuscript-facing text: `derivatives/confirmatory_temporal_coupling/review/figure3_manuscript_alignment/FIGURE3_MANUSCRIPT_SPEC.md`.

## Status

See `.cursor/plans/zero_lag_redesign_e761cae5.plan.md`.
