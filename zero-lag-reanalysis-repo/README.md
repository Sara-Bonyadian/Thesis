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

## HIIT full cohort (C-stage CLI)

Config: `datasets/hiit.yaml` (all subjects; production nulls = 500 surrogates).

Outputs go to:
`derivatives/confirmatory_temporal_coupling/sensitivity/hiit/`

From the repo root:

```bash
# Recommended: run C0 first and inspect eligibility / pairing
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml \
  --stage C0

# Then either continue stage-by-stage…
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml \
  --stage C1a
# …C1b, C1c, C2, C3, C4, C5, C6, C7

# …or run the full pipeline (C0–C7) in one go
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml \
  --stage all

# Same pipeline, force serial C1a/C1b/C4
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml \
  --stage all \
  --n-jobs 1

# C4 only, all CPUs (default --n-jobs -1)
.venv/bin/python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml \
  --stage C4 \
  --n-jobs -1
```

Notes:
- C4 nulls use **500** surrogates by default (`DEFAULT_N_SURROGATES`; production contract).
  This is an explicit Monte Carlo change from the prior **1000**-surrogate default:
  results are **not** identical. Add-one empirical p resolves to `1/(n+1)` → finest p ≈
  `1/501 ≈ 0.0020` (was `1/1001 ≈ 0.0010`); Monte Carlo SE scales up by ≈√2.
  Seed generation is unchanged; only the draw count changes. Override only if
  intentional: `--n-surrogates 20` (smoke-like; not for manuscript production).
- `--n-jobs` parallelizes **C1a**, **C1b**, and **C4** (default `-1` = all CPUs;
  `--n-jobs 1` = serial). C1a also applies a **file-size-aware RAM cap** (large
  OpenNeuro EEG can force serial on a 16 GB laptop). Prefer `--n-jobs 1` for C1a
  on laptops when recordings are multi-GB. C1a caches DPSS tapers per window
  length; C1b honors `cardiac.debug_plot` (no preview PNGs when false).
  Per-observation checkpoints live under `C1a/_obs_checkpoints/` /
  `C1b/_obs_checkpoints/` / `C4/_unit_checkpoints/`, with `C1a_COMPLETE.json` /
  `C1b_COMPLETE.json` / `C4_COMPLETE.json` written only after a fully successful
  stage.
- Optional artifact controls (CFA/QRS, motion/EOG/EMG, …) stay off unless you add
  `--optional-artifact-controls` (typically on C6 / `all`).
- Full HIIT is a **sensitivity** dataset in `master.yaml` (not a primary meta cohort).

## Peak model (Figure 1 Panel F)

Production subject-level fits (`ppg_eeg/confirmatory/peak_model.py`, stage **C3**) estimate the
**near-zero central peak**, not the strongest peak on ±60 s:

1. Linear baseline `B(τ)=b0+b1τ` from distant flanks `20 ≤ |τ| ≤ 60` s.
2. Nonnegative Gaussian on baseline-adjusted central data `|τ| ≤ 20` s.
3. μ initialized from the central baseline-adjusted argmax; constrained to ±20 s.
4. Identifiable peaks only (`A ≥ 1.8×RMSE`, SE(A), weak-edge) enter group inference.

**Group hierarchy (C6):** equal-weight mean of per-participant means (HIIT PH/PS nested
within participant), then μ TOST (±2 s) on that mean. FWHM uses the same nesting on
log-FWHM (back-transformed). See `peak_hierarchical_summaries.csv` and
`peak_center_equivalence.csv`.

Manuscript-ready wording and QC live under:
`derivatives/.../C7/figures/internal_qc/panel_f_option_c_update/` and
`.../panel_f_hierarchical_inference/`.

## HIIT smoke (C-stage CLI)

Everything for HIIT smoke lives under `smoke/hiit/` (subjects 01–03; `n_surrogates: 20`).

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
Production ops: `python -m ppg_eeg.confirmatory --mode preflight|smoke|primary|sensitivity|clean_root`.
Post-run QC (not a stage): `--mode qc-report` (below).

## Post-run QC report (read-only)

After C0–C7 (or any **partial** stage tree), generate a **non-binding** review report
under `{dataset_output_root}/QC/`.

| Property | Rule |
|----------|------|
| Invocation | `--mode qc-report` + `--config …` (optional `--qc-root`) |
| In `STAGE_ORDER` / `--stage all`? | **No** |
| Writes | **Only** `{output_root}/QC/` (may overwrite prior QC files) |
| Reads | Existing stage QC / eligibility CSVs only — no recomputation |
| Effect on C0–C7 | **None** (hashes of stage trees must stay unchanged) |

```bash
# HIIT smoke → derivatives/.../smoke/hiit_m13b/QC/
.venv/bin/python -m ppg_eeg.confirmatory \
  --mode qc-report \
  --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml

# Full HIIT → derivatives/.../sensitivity/hiit/QC/ (does not re-run analysis)
.venv/bin/python -m ppg_eeg.confirmatory \
  --mode qc-report \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml
```

### Outputs under `QC/`

| File | Grain |
|------|-------|
| `observation_qc.csv` | observation (cardiac, polarity gap, IHR, EEG, review flags) |
| `duration_qc.csv` | observation × duration |
| `coupling_unit_qc.csv` | observation × duration × band × representation |
| `contrast_qc.csv` | C5 contrast keys (empty if C5 absent → domain `not_assessed`) |
| `null_unit_qc.csv` | C4 null keys |
| `model_qc.csv` | C6 inference components |
| `dataset_qc_summary.csv` | dataset flag tallies |
| `visual_review_list.csv` | prioritized human review queue |
| `QC_REPORT.md` | narrative + disclaimer |
| `qc_params.json` | resolved heuristic thresholds + disclaimer |
| `qc_manifest.json` | schema version, git commit, input SHA-256, missing inputs |
| `extensions/` | reserved for future modules |

### Labels (do not overstate)

Use only: `pipeline_fact`, `derived_metric`, `heuristic_review_suggestion`,
`analytical_exclusion`, `not_assessed`, `no_automated_concern_detected`,
`flagged_for_review`. Absence of flags is **not** evidence of scientific correctness.
Cardiac review domain is `review_cardiac` (ECG or PPG via `signal_type` / `channel_used`).

Design authority: `.cursor/plans/post-run_qc_report_f40bad51.plan.md`.
Implementation: `ppg_eeg/confirmatory/qc_report.py`.

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

#### Figure 3 export categories

| Category | Stem / location | Include in manuscript export? | Include in supplementary export? |
|----------|-----------------|-------------------------------|----------------------------------|
| **Manuscript** | `figure3_temporal_artifact_specificity` | Yes (only this stem) | No |
| **Supplementary** | `figure3_supplement_null_diagnostics` | No | Yes |
| **Internal QC** | `figures/internal_qc/figure3_qc_participant_null_forests_*` | No | No (unless explicitly requested) |

Dataset-specific participant null forests are **internal QC artifacts**, not manuscript or supplementary figures. See `figures/figure_export_categories.csv` and the C7 visual review checklist.

Manuscript-facing text: `derivatives/confirmatory_temporal_coupling/review/figure3_manuscript_alignment/FIGURE3_MANUSCRIPT_SPEC.md`.

## Status

Pipeline status: `.cursor/plans/zero_lag_redesign_e761cae5.plan.md`.
Post-run QC design: `.cursor/plans/post-run_qc_report_f40bad51.plan.md`.
