# Implementation Plan: Temporal EEG–Cardiac Coupling

Technical plan for within-subject temporal coupling. Stages 1a/1b read **raw EEG + raw PPG/ECG**; Stages 1c–4 read prior CSV outputs (no raw reload).

**Last updated:** Stages 0–4 validated on ds003838 rest (65 subjects). Multi-dataset configs and adapter-based audit added for ds003838 (all tasks), ds006848, and HIIT. Cross-dataset replication summaries in progress.

---

## Goal

Move beyond between-subject summary correlations to examine **within-subject temporal coupling**:

- When HR or HRV changes within an individual, do EEG oscillations change before, during, or after?
- Do EEG changes predict later HR/HRV changes?
- **Cross-dataset:** Do coupling patterns replicate across ds003838, ds006848, and HIIT tasks/conditions?

Timing is central — one value per time point (default 1 Hz after resampling), not one summary per observation.

**Interpretation:** Descriptive timing only. Group permutation support (`sig_group_perm_fdr`) is the stricter evidence layer; selected peak-r FDR alone is exploratory. No causal claims.

---

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Separate module | Phase 1 pipeline unchanged; opt-in analysis |
| Config-driven | YAML ROIs, windows, lag limits, detector choice, QC plots |
| Per-observation first | Core math within observation; group stats in Stages 3–4 |
| One task/condition per group summary | Never mix tasks or HIIT states in one aggregate |
| Stage 1 = 3 CSVs | Reusable cache; Stages 2–4 avoid raw I/O |
| Dataset adapters | Stage 0 discovers observations via `build_observations` |
| Observation output dirs | `{observation_id}/` prevents HIIT multi-condition overwrites |
| Visual QC | Stage 1a/1b debug plots before alignment |
| Permutation null | Circular-shift control for peak strength (Stage 2 + Stage 3 group perm) |
| Subject-balanced events | Stage 4 averages within subject first, then across subjects |

---

## Code Layout

```
ppg_eeg/temporal_coupling/
  __main__.py              # CLI: python -m ppg_eeg.temporal_coupling
  config.py                # YAML → dataclasses
  run.py                   # stage dispatcher
  paths.py                 # observation_output_dir, group_output_dir, partition keys ✅
  data_audit.py            # Stage 0 ✅ (adapters + BIDS + BrainVision)
  eeg_envelope.py          # Stage 1a ✅
  cardiac_common.py        # shared types, IBI cleaning, ROI helpers ✅
  cardiac_detectors.py     # channel inventory, detector comparison, ECG R-peak ✅
  cardiac_timeseries.py    # Stage 1b ✅
  resample.py              # Stage 1c ✅
  cross_correlation.py     # Stage 2 ✅
  group_summary.py         # Stage 3 ✅
  events.py                # Stage 4 ✅
  cross_dataset_summary.py # cross-dataset tables + plots ⏳ planned
```

**Tests:** `tests/test_eeg_envelope.py`, `tests/test_cardiac_timeseries.py`, `tests/test_resample.py`, `tests/test_cross_correlation.py`, `tests/test_group_summary.py`, `tests/test_events.py`

**Configs:**

| Config | Dataset | Scope | Output root |
|--------|---------|-------|-------------|
| `config.smoke.ds003838.temporal_coupling.yaml` | ds003838 | rest, 3 subjects | `derivatives/smoke_ds003838_temporal_coupling/` |
| `config.validation.ds003838.temporal_coupling.yaml` | ds003838 | **rest only** (reference) | `derivatives/validation_ds003838_temporal_coupling/` |
| `config.run.ds003838.temporal_coupling.yaml` | ds003838 | **rest + memory** | `derivatives/run_ds003838_temporal_coupling/` |
| `config.run.ds006848.temporal_coupling.yaml` | ds006848 | rest, verbalwm | `derivatives/run_ds006848_temporal_coupling/` |
| `config.run.hiit.temporal_coupling.yaml` | HIIT | 8 conditions | `derivatives/run_hiit_temporal_coupling/` |

All **run** configs share the same `temporal_coupling` block as validation (ROIs, `fs_hz: 1.0`, `lag_step_s: 5`, `n_permutations: 100`, events on, `n_group_permutations: 100`).

---

## Stage 0: Data Feasibility Audit ✅

**Module:** `data_audit.py`  
**CLI:** `--stage 0`  
**Output:** `group/data_audit.csv`

| Column | Description |
|--------|-------------|
| `dataset_id`, `subject_id`, `task`, `condition`, `observation_id` | Keys |
| `eeg_file`, `eeg_format`, `cardiac_file`, `cardiac_format`, `ppg_source` | Paths and modality |
| `eeg_duration_s`, `cardiac_duration_s`, `overlap_duration_s` | Durations |
| `eeg_sfreq`, `cardiac_sfreq` | Sampling rates |
| `usable`, `skip_reason` | Gate for Stage 1 |
| `recommended_max_lag_s` | `min(lag_max_s, floor(overlap_s / 3))` |

**Discovery:** `list_configured_observations()` → `build_observations()` per dataset adapter.

| Dataset | EEG format | Cardiac source |
|---------|------------|----------------|
| ds003838 | EEGLAB `.set` (BIDS) | Separate ECG `.set` |
| ds006848 | BrainVision `.vhdr` | Embedded PPG in EEG |
| HIIT | BrainVision `.vhdr` | Embedded PPG in EEG |

**Minimum thresholds (configurable):** `min_overlap_s: 120`, `min_clean_beats: 30`

Metadata: BIDS JSON sidecar when present; else MNE header (`read_raw_eeglab` or `read_raw_brainvision`).

---

## Stage 1: Temporal Feature Extraction ✅

Three outputs per observation. Run individually: `--stage 1a`, `--stage 1b`, `--stage 1c`, or together: `--stage 1`.

**Output root:** `{out_root}/{dataset_id}/{observation_id}/`

### Stage 1a: `features_temporal_eeg_envelope.csv` ✅

**Module:** `eeg_envelope.py`

**Columns:** `dataset_id | subject_id | task | observation_id | time_s | theta_env | alpha_env | beta_env`

**Pipeline:**
1. Load raw EEG (`eeg_format` from audit)
2. Preprocess (1–60 Hz, bad-channel rejection, average reference)
3. Per band: bandpass → Hilbert amplitude → Gaussian smooth (2 s)
4. Average envelope across configurable ROI channels
5. Downsample to `envelope_output_fs_hz` (10 Hz) before CSV write

**ROI defaults (all run configs):**

| Band | Requested ROI |
|------|---------------|
| Theta | Fz, F1, F2, FCz, FC1, FC2, Cz |
| Alpha | Pz, P3, P4, POz, PO3, PO4, Oz, O1, O2 |
| Beta | Fz, F1, F2, FCz, FC1, FC2, Cz, C3, C4 |

Missing channels: skip, do not crash; report in `eeg_envelope_qc.csv`.

**Group:** `group/eeg_envelope_qc.csv` — includes `condition`, `*_roi_channels_used`, `usable_eeg_envelope`, warnings.

---

### Stage 1b: `features_temporal_cardiac.csv` ✅

**Modules:** `cardiac_timeseries.py`, `cardiac_detectors.py`, `cardiac_common.py`

**Columns:** `dataset_id | subject_id | task | observation_id | time_s | hr | rmssd | sdnn | mean_rr | n_beats_hr_window | n_beats_hrv_window | quality_flag | usable_for_hr | usable_for_hrv`

**Pipeline:**
1. Load cardiac file (`cardiac_format` from audit; embedded PPG uses same file as EEG)
2. Channel inventory + auto selection
3. Detector comparison: `simple`, `ecg_rpeak`, `ppg_peak`
4. Peak detection → IBI cleaning
5. Sliding-window HR / mean_rr / RMSSD / SDNN

| Variable | Window | Notes |
|----------|--------|-------|
| HR, mean_rr | 15 s | `min_beats_hr: 5` |
| RMSSD, SDNN | 60 s, step 2 s | `min_beats_hrv: 20` |

**Group:** `group/cardiac_qc.csv` — includes `condition`, detector metadata, `usable_for_hr`, `usable_for_hrv`.

---

### Stage 1c: `features_temporal_aligned.csv` ✅

**Module:** `resample.py`

**Columns:** raw + z-scored cardiac and envelope columns (see README).

**Steps:**
1. Gate on upstream QC flags per `observation_id`
2. Intersect valid time range
3. Resample to `fs_hz: 1.0`
4. Interpolate short gaps only (`interpolate_max_gap_s: 5`)
5. Z-score within observation

**Group:** `group/alignment_qc.csv` — `recommended_xcorr_lag_s`, `usable_for_xcorr`.

---

## Dataset-Specific Notes

### ds003838

- Separate EEG and ECG files; assume **t = 0** alignment unless trigger metadata exists (report in methods).
- **rest** (~3–4 min): validated reference, 65 subjects, Stages 0–4 complete under validation config.
- **memory**: analyze separately from rest; task events may confound coupling.

### ds006848

- BrainVision EEG with embedded PPG.
- Tasks: `rest` (22 obs), `verbalwm` (30 obs) — 52 total; Stage 0: 52/52 usable.
- Longer verbalwm recordings → audit may still cap lag at config `lag_max_s: 60`.

### HIIT

- BrainVision EEG with embedded PPG (photosensor `ph` or pressure `ps`).
- **8 conditions:** `{ph,ps}_{pre,post}_{rest,tetris}` — ~20 subjects each, 160 observations.
- Never combine PS and PH in one group summary unless explicitly labeled exploratory.

---

## Stage 2: Lagged Cross-Correlation ✅

**Module:** `cross_correlation.py`  
**Input:** `features_temporal_aligned.csv` (`*_z`)  
**CLI:** `--stage 2`

**9 pairs:** HR/RMSSD/SDNN × theta/alpha/beta

**Lag:** per-observation `recommended_xcorr_lag_s` from alignment QC, capped by `lag_max_s` and `floor(overlap_s / 3)`.

#### Peak selection rule (primary)

**Raw peak** = lag where |r| is maximum on the lag grid.

Primary columns: `peak_lag_s` = `raw_peak_lag_s`, etc. Interior/preferred peaks are QC-only.

**Per-observation outputs:** `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`, grid/overlay plots.

**Group outputs:** aggregated peaks/curves, `cross_correlation_qc_summary.csv`, `cross_correlation_peak_validation.csv`, distribution plots.

**Permutation null:** `n_permutations: 100` in run configs → `p_perm` per observation.

**Peak row columns include:** `dataset_id`, `subject_id`, `task`, `condition`, `observation_id`, `pair`, `raw_peak_*`, `p_perm`, …

---

## Stage 3: Group Analysis ✅

**Module:** `group_summary.py`  
**CLI:** `--stage 3`

**Outputs:**
- `peak_correlation_summary.csv`
- `group_cross_correlation_summary.csv`
- `mean_cross_correlation_curves.csv`
- `stage3_interpretation_notes.txt`
- Plots: mean ± SEM grid, peak heatmaps, lag/r distributions, edge peak summary

### Three inference layers

| Layer | Columns | Interpretation |
|-------|---------|----------------|
| 1. Selected peak signed-r | `sig_peak_signed_r_fdr`, `q_value_peak_signed_r` | Exploratory if not perm-supported |
| 2. Group permutation peak strength | `sig_group_perm_fdr`, `group_perm_q`, `group_perm_stat` | **Stricter evidence** |
| 3. Peak-lag direction | `sig_peak_lag_fdr`, `q_value_peak_lag` | Direction stability across subjects |

**ds003838 rest reference (n=65):**

| Pair | `sig_peak_signed_r_fdr` | `sig_group_perm_fdr` | Notes |
|------|-------------------------|----------------------|-------|
| All 9 pairs | true | — | Selected peak-r layer |
| rmssd × beta | true | **true** (q≈0.045) | Permutation-supported |
| sdnn × beta | true | **true** (q≈0.045) | Permutation-supported; edge peaks ~15% |
| Others | true | false | Exploratory at group-perm layer |

**Planned:** `subject_consistency_summary.csv` per dataset/task/condition/pair with labels `high_consistency`, `moderate_consistency`, `low_consistency`, `inconclusive_low_n`.

**Planned:** Partitioned group outputs when one config lists multiple tasks/conditions (`group/{partition}/`).

---

## Stage 4: Event-Triggered Analysis ✅

**Module:** `events.py`  
**CLI:** `--stage 4`  
**Input:** `features_temporal_aligned.csv`

**Epoch window:** `epoch_pre_s` / `epoch_post_s` (default ±60 s).

### Event types

| Direction | Event types |
|-----------|-------------|
| HR → EEG | HR increase, HR decrease |
| EEG → cardiac | Theta burst, alpha suppression, beta burst |

### Usability

| Rule | Default |
|------|---------|
| `min_total_events` | 10 |
| `min_contributing_subjects` | 4 |

**Outputs:** `event_counts.csv`, `event_list.csv`, `event_qc.csv`, triggered CSVs/PNGs, `stage4_interpretation_notes.txt`.

**`event_qc.csv` columns:** `event_type`, `total_events`, `n_subjects`, `usable_for_group_plot`, `subject_balanced_average`, `warning`.

**Planned:** `cross_dataset_event_qc_summary.csv` (dataset × task × condition × event_type).

---

## Cross-Dataset Replication (in progress)

**Goal:** Compare ds003838, ds006848, and HIIT without causal claims.

**Planned outputs** (under `derivatives/cross_dataset_temporal_coupling/`):

| File | Rows | Key columns |
|------|------|-------------|
| `cross_dataset_temporal_coupling_summary.csv` | dataset × task × condition × pair | peak stats, perm q, consistency, `replication_label` |
| `subject_consistency_summary.csv` | dataset × task × condition × pair | IQR, % positive r, % edge peaks, `consistency_label` |
| `cross_dataset_event_qc_summary.csv` | dataset × task × condition × event_type | usability flags |

**Replication labels:** `replicates_strongly`, `replicates_partially`, `does_not_replicate`, `inconclusive_low_n_or_qc`.

**Planned plots:** peak strength, group perm, lag, edge peak, consistency heatmaps (rows = dataset/task/condition, columns = 9 pairs).

**Reference anchor:** ds003838 rest from `validation_ds003838_temporal_coupling` (65 subjects).

---

## Config Reference (run configs)

Shared `temporal_coupling` block across `config.run.*.temporal_coupling.yaml` and `config.validation.ds003838.temporal_coupling.yaml`:

```yaml
temporal_coupling:
  audit:
    min_overlap_s: 120
    min_clean_beats: 30
  eeg:
    bands: { theta: [4,8], alpha: [8,13], beta: [13,30] }
    envelope_smooth_s: 2.0
    envelope_output_fs_hz: 10
    rois:
      theta: [Fz, F1, F2, FCz, FC1, FC2, Cz]
      alpha: [Pz, P3, P4, POz, PO3, PO4, Oz, O1, O2]
      beta: [Fz, F1, F2, FCz, FC1, FC2, Cz, C3, C4]
  cardiac:
    channel: auto
    detector: auto
    hr_window_s: 15.0
    hrv_window_s: 60.0
    hrv_step_s: 2.0
  resample:
    fs_hz: 1.0
    z_score: true
    interpolate_max_gap_s: 5
  cross_correlation:
    lag_max_s: 60
    lag_step_s: 5
    n_permutations: 100
    edge_margin_s: 0
    min_peak_distance_s: 10
  events:
    enabled: true
    epoch_pre_s: 60
    epoch_post_s: 60
    hr_percentile: 10
    min_total_events: 10
    min_contributing_subjects: 4
  group:
    plot_common_lag_only: true
    n_group_permutations: 100
```

Dataset-specific top-level filters:

```yaml
# ds003838 all tasks
tasks: [rest, memory]
conditions: [rest, memory]

# ds006848
tasks: [rest, verbalwm]
conditions: [rest, verbalwm]

# HIIT
tasks: [rest, tetris]
conditions: [ph_pre_rest, ph_pre_tetris, ph_post_rest, ph_post_tetris,
            ps_pre_rest, ps_pre_tetris, ps_post_rest, ps_post_tetris]
sessions: [ph, ps]
paths:
  raw_root: ./data/raw/HIIT
```

---

## Build Order

| # | Task | Status |
|---|------|--------|
| 1 | Config + Stage 0 audit | ✅ |
| 2 | Smoke 3 subjects (ds003838 rest) | ✅ |
| 3 | Stages 1a–1c + QC | ✅ |
| 4 | Stage 2 xcorr + permutation null | ✅ |
| 5 | Stage 3 group summary + group perm | ✅ |
| 6 | Stage 4 events | ✅ |
| 7 | ds003838 rest full cohort (validation config) | ✅ **65 subjects** |
| 8 | Multi-dataset audit (adapters, BrainVision, embedded PPG) | ✅ |
| 9 | Observation-level output dirs (`paths.py`) | ✅ |
| 10 | Run configs: ds003838 all tasks, ds006848, HIIT | ✅ |
| 11 | ds006848 Stages 1–4 | ⏳ Stage 0 done (52/52) |
| 12 | HIIT Stages 0–4 (8 conditions) | ⏳ |
| 13 | ds003838 memory + full run config | ⏳ |
| 14 | Partitioned group outputs (per task/condition) | ⏳ |
| 15 | `subject_consistency_summary.csv` | ⏳ |
| 16 | `cross_dataset_temporal_coupling_summary.csv` + plots | ⏳ |
| 17 | Cross-dataset replication report (conservative) | ⏳ |

---

## Deliverables Checklist

### Per observation
- [x] `features_temporal_eeg_envelope.csv`
- [x] `features_temporal_cardiac.csv`
- [x] `detected_peaks.csv`
- [x] `features_temporal_aligned.csv`
- [x] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`

### Per dataset run (`group/`)
- [x] `data_audit.csv` (with `condition`, `eeg_format`, `ppg_source`)
- [x] `eeg_envelope_qc.csv`, `cardiac_qc.csv`, `alignment_qc.csv`
- [x] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`, `cross_correlation_qc_summary.csv`
- [x] `cross_correlation_peak_validation.csv`
- [x] `peak_correlation_summary.csv`, `group_cross_correlation_summary.csv`
- [x] `mean_cross_correlation_curves.csv`, `stage3_interpretation_notes.txt`
- [x] Stage 3 plots (heatmaps, lag/r distributions, mean ± SEM grid)
- [x] `event_counts.csv`, `event_list.csv`, `event_qc.csv`
- [x] `event_triggered_hr_to_eeg.csv`, `event_triggered_eeg_to_hr.csv`
- [x] `event_triggered_hr_to_eeg.png`, `event_triggered_eeg_to_hr.png`
- [x] `stage4_interpretation_notes.txt`

### Per dataset × task × condition (planned)
- [ ] Partitioned `group/{partition}/` copies of Stage 2–4 outputs
- [ ] `subject_consistency_summary.csv`

### Cross-dataset (planned)
- [ ] `cross_dataset_temporal_coupling_summary.csv`
- [ ] `cross_dataset_event_qc_summary.csv`
- [ ] Cross-dataset heatmaps
- [ ] Conservative replication report for Russell

---

## Running (staged workflow)

```bash
PY=".venv/bin/python"
CFG=config.run.ds006848.temporal_coupling.yaml   # or ds003838 / hiit

$PY -m ppg_eeg.temporal_coupling --config $CFG --stage 0
# inspect group/data_audit.csv

$PY -m ppg_eeg.temporal_coupling --config $CFG --stage 1
# inspect eeg_envelope_qc.csv, cardiac_qc.csv, alignment_qc.csv

$PY -m ppg_eeg.temporal_coupling --config $CFG --stage 2
# inspect cross_correlation_peak_validation.csv

$PY -m ppg_eeg.temporal_coupling --config $CFG --stage 3
$PY -m ppg_eeg.temporal_coupling --config $CFG --stage 4
```

Do not use `--stage all` on full cohorts until QC gates are verified per stage.
