# Implementation Plan: Temporal EEG–Cardiac Coupling

Technical plan for within-subject temporal coupling. Stages 1a/1b read **raw EEG + raw PPG/ECG**; Stages 1c–4 read prior CSV outputs (no raw reload).

**Last updated:** Stages 0–4 implemented and validated on ds003838 rest (smoke: 3 subjects; validation: 8 subjects).

---

## Goal

Move beyond between-subject summary correlations to examine **within-subject temporal coupling**:

- When HR or HRV changes within an individual, do EEG oscillations change before, during, or after?
- Do EEG changes predict later HR/HRV changes?

Timing is central — one value per time point (e.g. every 0.5–1 s), not one summary per observation.

---

## Design Principles

| Principle | Rationale |
|-----------|-----------|
| Separate module | Phase 1 pipeline unchanged; opt-in analysis |
| Config-driven | YAML ROIs, windows, lag limits, detector choice, QC plots |
| Per-subject first | Core math within subject; group stats in Stages 3–4 |
| Stage 1 = 3 CSVs | Reusable cache; Stages 2–4 avoid raw I/O |
| Visual QC | Stage 1a/1b debug plots before alignment |
| Permutation null | Peak-r across lags needs circular-shift control (Stage 2) |
| Subject-balanced events | Stage 4 averages within subject first, then across subjects |

---

## Code Layout

```
ppg_eeg/temporal_coupling/
  __main__.py              # CLI: python -m ppg_eeg.temporal_coupling
  config.py                # YAML → dataclasses
  run.py                   # stage dispatcher
  data_audit.py            # Stage 0 ✅
  eeg_envelope.py          # Stage 1a ✅
  cardiac_common.py        # shared types, IBI cleaning, ROI helpers ✅
  cardiac_detectors.py     # channel inventory, detector comparison, ECG R-peak ✅
  cardiac_timeseries.py    # Stage 1b ✅
  resample.py              # Stage 1c ✅
  cross_correlation.py     # Stage 2 ✅
  group_summary.py         # Stage 3 ✅
  events.py                # Stage 4 ✅
```

**Tests:** `tests/test_eeg_envelope.py`, `tests/test_cardiac_timeseries.py`, `tests/test_resample.py`, `tests/test_cross_correlation.py`, `tests/test_group_summary.py`, `tests/test_events.py`

**Configs:**

| Config | Purpose |
|--------|---------|
| `config.smoke.ds003838.temporal_coupling.yaml` | 3-subject smoke |
| `config.validation.ds003838.temporal_coupling.yaml` | 8-subject validation (+ permutations) |
| `config.run.ds003838.temporal_coupling.yaml` | Full-dataset run |

---

## Stage 0: Data Feasibility Audit ✅

**Module:** `data_audit.py`  
**CLI:** `--stage 0`  
**Output:** `group/data_audit.csv`

| Column | Description |
|--------|-------------|
| `dataset_id`, `subject_id`, `task`, `observation_id` | Keys |
| `eeg_file`, `cardiac_file` | BIDS paths |
| `eeg_duration_s`, `cardiac_duration_s`, `overlap_duration_s` | Durations |
| `eeg_sfreq`, `cardiac_sfreq` | Sampling rates |
| `usable`, `skip_reason` | Gate for Stage 1 |
| `recommended_max_lag_s` | `min(lag_max_s, floor(overlap_s / 3))` |

**Minimum thresholds (configurable):** `min_overlap_s: 120`, `min_clean_beats: 30`

Reads BIDS JSON sidecars first; falls back to MNE `preload=False` for metadata.

---

## Stage 1: Temporal Feature Extraction ✅

Three outputs per subject×observation. Run individually: `--stage 1a`, `--stage 1b`, `--stage 1c`, or together: `--stage 1`.

### Stage 1a: `features_temporal_eeg_envelope.csv` ✅

**Module:** `eeg_envelope.py`

**Columns:** `dataset_id | subject_id | task | observation_id | time_s | theta_env | alpha_env | beta_env`

**Pipeline:**
1. Load raw EEG; preprocess (1–60 Hz, bad-channel rejection, average reference)
2. Per band: bandpass (theta 4–8, alpha 8–13, beta 13–30 Hz) → Hilbert amplitude → Gaussian smooth (default 2 s)
3. Average envelope across configurable ROI channels
4. Downsample to `envelope_output_fs_hz` before CSV write (smoke: **10 Hz**)

**ROI defaults (ds003838):**

| Band | Requested ROI |
|------|---------------|
| Theta | Fz, F1, F2, FCz, FC1, FC2, Cz |
| Alpha | Pz, P3, P4, POz, PO3, PO4, Oz, O1, O2 |
| Beta | Fz, F1, F2, FCz, FC1, FC2, Cz, C3, C4 |

Missing requested channels are skipped; envelopes average over **available** requested channels only. `group/eeg_envelope_qc.csv` records `*_roi_channels_used` and warnings such as `missing_roi_channel`.

Z-scoring deferred to Stage 1c aligned CSV.

#### Stage 1a QC outputs ✅

**Per subject:** `eeg_envelope_debug_*.png`, `eeg_envelope_timeseries.png`, optional `eeg_psd_debug.png`

**Group:** `group/eeg_envelope_qc.csv` — ROI channels used, NaN %, min/max/median, `usable_eeg_envelope`, warnings.

---

### Stage 1b: `features_temporal_cardiac.csv` ✅

**Modules:** `cardiac_timeseries.py`, `cardiac_detectors.py`, `cardiac_common.py`

**Columns:** `dataset_id | subject_id | task | observation_id | time_s | hr | rmssd | sdnn | mean_rr | n_beats_hr_window | n_beats_hrv_window | quality_flag | usable_for_hr | usable_for_hrv`

**Pipeline:**
1. Channel inventory + auto selection (or configured channel)
2. Detector comparison: `simple`, `ecg_rpeak` (normal/inverted), `ppg_peak`
3. Peak detection → IBI cleaning (`ibi_min_ms`–`ibi_max_ms`, outlier jump rejection)
4. Sliding-window HR / mean_rr / RMSSD / SDNN

| Variable | Window (smoke) | Notes |
|----------|----------------|-------|
| HR, mean_rr | 15 s | `min_beats_hr: 5` |
| RMSSD, SDNN | 60 s, step 2 s | `min_beats_hrv: 20` |

#### Cardiac peak detection (implemented)

| Detector | Preprocessing | Peak finder |
|----------|---------------|-------------|
| `ecg_rpeak` | Detrend → 5–30 Hz bandpass → z-score | `scipy.find_peaks` prominence + 0.45 s refractory |
| `ppg_peak` | Detrend → z-score | Prominence-based peaks |
| `simple` | Detrend → z-score | Legacy height threshold |

**Auto mode:** scores each channel × detector × polarity; picks highest `quality_score`.

**IBI limits:** `ibi_min_ms: 400`, `ibi_max_ms: 1500`

#### Stage 1b QC outputs ✅

**Per subject:** `detected_peaks.csv`, `cardiac_channel_inventory.csv`, `peak_detector_comparison.csv`, debug plots.

**Group:** `group/cardiac_qc.csv`

**ds003838 smoke results:** all three subjects use **ECG + ecg_rpeak**; median HR 71–79 bpm; 100% valid HR/HRV.

---

### Stage 1c: `features_temporal_aligned.csv` ✅

**Module:** `resample.py`  
**CLI:** `--stage 1c`

**Columns:** `dataset_id | subject_id | task | observation_id | time_s | hr | rmssd | sdnn | mean_rr | theta_env | alpha_env | beta_env | hr_z | rmssd_z | sdnn_z | mean_rr_z | theta_env_z | alpha_env_z | beta_env_z`

**Steps:**
1. Gate on `cardiac_qc.usable_for_hr` and `eeg_envelope_qc.usable_eeg_envelope`
2. Intersect valid time range (EEG + cardiac overlap)
3. Resample to `temporal_coupling.resample.fs_hz` (default 1 Hz)
4. Interpolate short gaps only (`interpolate_max_gap_s`) — no long forward-fill
5. Z-score within subject×observation → `*_z` columns

**Group:** `group/alignment_qc.csv` — aligned duration, missing % per variable, `recommended_xcorr_lag_s`, `usable_for_xcorr`.

---

## ds003838 Synchronization Warning

EEG and ECG/PPG are **separate files**. Without shared trigger metadata, assume **t = 0** alignment. **Report this assumption** in audit + methods — a 5–10 s offset could invalidate lag conclusions.

Build **rest task first**; memory task separately (task events confound coupling).

---

## Stage 2: Lagged Cross-Correlation ✅

**Module:** `cross_correlation.py`  
**Input:** `features_temporal_aligned.csv` (`*_z`)  
**CLI:** `--stage 2`

**9 pairs:** HR/RMSSD/SDNN × theta/alpha/beta

**Lag:** per-subject `recommended_xcorr_lag_s` from alignment QC (capped by config `lag_max_s`; smoke/validation: ±60 s)

#### Peak selection rule (primary)

**Raw peak** = lag where **|r| is maximum** on the lag grid (after optional local-peak detection via `scipy.find_peaks` on signed r).

Primary output columns mirror the raw peak:

| Primary column | Equals |
|----------------|--------|
| `peak_lag_s` | `raw_peak_lag_s` |
| `peak_signed_r` | `raw_peak_signed_r` |
| `peak_abs_r` | `raw_peak_abs_r` |
| `peak_at_lag_edge` | `raw_peak_at_edge` |

**Interior** and **preferred** peak columns (`interior_peak_*`, `preferred_peak_*`) are **QC diagnostics only** — e.g. edge-flagged raw peaks with same-sign interior alternatives. They do **not** replace the primary peak in Stage 3 group summaries or main plots.

**Per-subject outputs:**
- `cross_correlation_peaks.csv` — peak lag, signed r, edge flags, optional `p_perm`
- `cross_correlation_curves.csv` — `pair, lag_s, r, n_overlap` (if `save_curves: true`)
- `cross_correlation_grid_subject.png`, overlay plots, peak distribution plots

**Group outputs:**
- `group/cross_correlation_peaks.csv`, `group/cross_correlation_curves.csv`
- `group/cross_correlation_qc_summary.csv`
- `group/cross_correlation_peak_validation.csv`
- QC plots: peak lag/strength distributions, edge peak summary

**Permutation null:** circular time-shift one signal `n_permutations` times; `p_perm` for observed peak (validation config: 100 permutations).

---

## Stage 3: Group Analysis ✅

**Module:** `group_summary.py`  
**CLI:** `--stage 3`  
**Input:** group Stage 2 peaks/curves

**Outputs:**
- `peak_correlation_summary.csv` — per-pair median/mean peak lag and r, FDR q-values
- `group_cross_correlation_summary.csv` — interpretation labels (direction, coupling strength)
- `mean_cross_correlation_curves.csv` — group mean ± SEM with common lag range
- `stage3_interpretation_notes.txt`

**Plots:** mean ± SEM grid, peak summary heatmap, edge peak rate heatmap, peak lag/r distribution plots.

**Tests per pair:** signed peak r ≠ 0; median lag ≠ 0; peak r vs permutation null; BH-FDR across 9 pairs (when permutations run).

---

## Stage 4: Event-Triggered Analysis ✅

**Module:** `events.py`  
**CLI:** `--stage 4`  
**Input:** `features_temporal_aligned.csv` only (no raw reload)

**Epoch window:** configurable `epoch_pre_s` / `epoch_post_s` (default ±60 s for ds003838).

### Averaging method

`subject_mean_then_group_mean` (`subject_balanced_average: true`):

1. Average **accepted events within each subject** → one trajectory per subject.
2. Average **subject-level trajectories** across subjects (equal weight per subject).

Subjects with many events do **not** dominate the group curve.

### Heart-led (HR → EEG)

- Compute ΔHR z over `hr_delta_window_s` (default 15 s).
- Top/bottom tail events via `hr_percentile` / `event_percentile` (default top/bottom 10%).
- Extract `theta_env_z`, `alpha_env_z`, `beta_env_z` epochs around each accepted event.

### Brain-led (EEG → cardiac)

Configurable per band (`fixed_z` or `percentile`):

| Event | Default detection |
|-------|-------------------|
| Theta burst | top 10% `theta_env_z`, ≥ `min_event_duration_s`, `min_event_distance_s` between onsets |
| Alpha suppression | bottom 10% `alpha_env_z` (or `fixed_z` &lt; −threshold) |
| Beta burst | top 10% `beta_env_z` |

Extract `hr_z`, `rmssd_z`, `sdnn_z` around each accepted event onset.

### Usability thresholds

| Rule | Default |
|------|---------|
| `min_total_events` | 10 |
| `min_contributing_subjects` | 4 |

Below threshold → `usable_for_group_plot: false`, marked exploratory in QC and plot titles.

### Validation cohort status (8 subjects)

| Event type | Events | Subjects | Status |
|------------|--------|----------|--------|
| HR increase | 40 | 8 | usable |
| HR decrease | 40 | 8 | usable |
| Beta burst | 10 | 7 | usable (cautious) |
| Theta burst | 8 | 5 | exploratory |
| Alpha suppression | 5 | 5 | exploratory |

### Outputs

| File | Description |
|------|-------------|
| `event_counts.csv` | Per-subject accepted event counts |
| `event_list.csv` | Every detected event: time, value, epoch window, `accepted`, `rejection_reason` |
| `event_qc.csv` | Group usability + averaging metadata |
| `event_triggered_hr_to_eeg.csv` | Group HR→EEG curves (mean_z, sem_z, n_subjects, n_events) |
| `event_triggered_eeg_to_hr.csv` | Group EEG→cardiac curves |
| `event_triggered_hr_to_eeg.png` | HR-led plots with exploratory labels |
| `event_triggered_eeg_to_hr.png` | Brain-led plots |
| `stage4_interpretation_notes.txt` | Usability summary + interpretation guidance |

**`event_list.csv` columns:** `dataset_id | subject_id | task | observation_id | event_type | event_time_s | event_value | epoch_start_s | epoch_end_s | accepted | rejection_reason`

**`event_qc.csv` columns:** `event_type | total_events | n_subjects | mean/min/max_events_per_subject | usable_for_group_plot | averaging_method | subject_balanced_average | min_total_events_required | min_subjects_required | warning`

**Interpretation (validation run):**
- HR-triggered EEG plots are the most reliable Stage 4 result.
- EEG-triggered cardiac plots are exploratory, especially theta and alpha.
- Beta burst meets minimum counts but interpret cautiously.
- No biological conclusions until the full cohort is run.

### Events config reference

`events.enabled` defaults to **`false`** in smoke configs (Stage 4 is opt-in during development). For validation and full-dataset runs that include event-triggered analysis, set **`enabled: true`**.

```yaml
temporal_coupling:
  events:
    enabled: false   # set true for validation/full Stage 4 runs
    hr_delta_window_s: 15
    epoch_pre_s: 60
    epoch_post_s: 60
    hr_percentile: 10
    min_event_duration_s: 2
    min_event_distance_s: 10
    min_total_events: 10
    min_contributing_subjects: 4
    theta_burst_method: percentile
    theta_burst_percentile: 90
    alpha_suppression_method: percentile
    alpha_suppression_percentile: 10
    beta_burst_method: percentile
    beta_burst_percentile: 90
    eeg_event_threshold_z: 2.0   # used when method: fixed_z
```

---

## Dataset Order

1. **ds003838 rest** — ~3–4 min, ±60 s lag, separate EEG/ECG files — **pipeline validated (smoke + 8-subject validation)**
2. **ds003838 rest full cohort** — `config.run.ds003838.temporal_coupling.yaml` — **next scale-up**
3. **ds006848** — longer rest; cross-dataset validation after ds003838 full
4. **HIIT** — ~5 min epochs only; separate PRE/POST × rest/tetris × ph/ps

---

## Config Reference (smoke)

```yaml
# config.smoke.ds003838.temporal_coupling.yaml

dataset_id: ds003838
paths:
  raw_root: ./data/raw/ds003838
  out_root: ./derivatives/smoke_ds003838_temporal_coupling

subjects: [sub-033, sub-036, sub-038]
tasks: [rest]

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
    ecg: { bandpass_hz: [5,30], min_peak_distance_s: 0.45 }
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
    n_permutations: 0
  events:
    epoch_pre_s: 60
    epoch_post_s: 60
    min_event_distance_s: 10
    min_total_events: 10
    min_contributing_subjects: 4
    theta_burst_method: percentile
    alpha_suppression_method: percentile
    alpha_suppression_percentile: 10
```

---

## Build Order

| # | Task | Status |
|---|------|--------|
| 1 | Config + Stage 0 audit | ✅ |
| 2 | Audit ds003838 rest (smoke) | ✅ |
| 3 | `eeg_envelope.py` + QC plots | ✅ |
| 4 | `cardiac_timeseries.py` + detectors + QC plots | ✅ |
| 5 | `resample.py` (Stage 1c) | ✅ |
| 6 | Synthetic xcorr test | ✅ |
| 7 | Smoke 3 subjects end-to-end | ✅ |
| 8 | Permutation null in `cross_correlation.py` | ✅ |
| 9 | `group_summary.py` (Stage 3) | ✅ |
| 10 | `events.py` (Stage 4) | ✅ |
| 11 | Validation 8 subjects (Stages 2–4) | ✅ |
| 12 | ds003838 rest full cohort (`config.run.ds003838.temporal_coupling.yaml`) | ⏳ **next** |
| 13 | ds006848 rest | ⏳ |
| 14 | HIIT full cohort | ⏳ |
| 15 | Full-cohort biological interpretation | ⏳ |

---

## Deliverables Checklist

### Per subject
- [x] `features_temporal_eeg_envelope.csv`
- [x] `features_temporal_cardiac.csv`
- [x] `detected_peaks.csv`
- [x] EEG envelope QC plots + `eeg_envelope_qc.csv` (group)
- [x] Cardiac QC plots + `cardiac_qc.csv` (group)
- [x] `features_temporal_aligned.csv`
- [x] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`

### Per dataset (`group/`)
- [x] `data_audit.csv`
- [x] `eeg_envelope_qc.csv`
- [x] `cardiac_qc.csv`
- [x] `alignment_qc.csv`
- [x] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`, `cross_correlation_qc_summary.csv`
- [x] `peak_correlation_summary.csv`, `group_cross_correlation_summary.csv`
- [x] `mean_cross_correlation_curves.csv`, `stage3_interpretation_notes.txt`
- [x] Mean xcorr curves, peak lag/r distributions, summary heatmaps
- [x] `event_counts.csv`, `event_list.csv`, `event_qc.csv`
- [x] `event_triggered_hr_to_eeg.csv`, `event_triggered_eeg_to_hr.csv`
- [x] `event_triggered_hr_to_eeg.png`, `event_triggered_eeg_to_hr.png`
- [x] `stage4_interpretation_notes.txt`
- [ ] Full-cohort interpretation report
