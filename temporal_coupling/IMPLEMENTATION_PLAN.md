# Implementation Plan: Temporal EEG–Cardiac Coupling

Technical plan for within-subject temporal coupling. Reads **raw EEG + raw PPG/ECG only**.

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
| Config-driven | YAML ROIs, windows, lag limits, permutation count |
| Per-subject first | Core math within subject; group stats in Stage 3 |
| Stage 1 = 3 CSVs | Reusable cache; Stages 2–4 avoid raw I/O |
| Permutation null | Peak-r across lags needs circular-shift control |

---

## Code Layout

```
ppg_eeg/temporal_coupling/
  data_audit.py
  eeg_envelope.py
  cardiac_timeseries.py
  resample.py
  cross_correlation.py   # includes circular-shift permutation
  group_summary.py
  events.py
  run.py
```

---

## Stage 0: Data Feasibility Audit

**Output:** `group/data_audit.csv`

| Column | Description |
|--------|-------------|
| `dataset_id`, `subject_id`, `observation_id` | Keys |
| `eeg_duration_s`, `cardiac_duration_s`, `overlap_duration_s` | Durations |
| `sampling_rate` | EEG sfreq |
| `n_detected_beats`, `n_clean_beats` | Beat counts |
| `valid_duration_s` | Non-missing overlap estimate |
| `recommended_max_lag_s` | See formula below |
| `usable`, `skip_reason` | Gate for Stage 1 |

**Minimum thresholds (configurable):** `min_overlap_s: 120`, `min_clean_beats: 30`

**Adaptive lag:**

```text
valid_duration_s = duration where both signals have valid data
recommended_max_lag_s = min(requested_lag_max_s, floor(valid_duration_s / 3))
```

For 240–305 s recordings: typically **±60 s**, up to **±90 s** if enough valid data.

---

## Stage 1: Temporal Feature Extraction

Three outputs per subject×observation. **CLI:** `--stage 1`

### Output 1: `features_temporal_eeg_envelope.csv`

`dataset_id | subject_id | observation_id | time_s | theta_env | alpha_env | beta_env`

**Steps:**
1. Load raw EEG; preprocess (filter, bad channels, CAR)
2. Bandpass: theta 4–8, alpha 8–13, beta 13–30 Hz
3. `envelope(t) = abs(Hilbert(signal))`
4. Optional `envelope_mode: amplitude | power` → `envelope**2` for power-like envelope
5. Smooth 1–5 s (default 2 s)
6. Average within **configurable ROIs**:

| Band | Default ROI |
|------|-------------|
| Theta | Fz, FCz, Cz |
| Alpha | Pz, POz, Oz, or whole-scalp |
| Beta | Frontal/central/global (config) |

Z-scoring happens in aligned CSV; interpretation = relative fluctuations over time.

### Output 2: `features_temporal_cardiac.csv`

`dataset_id | subject_id | observation_id | time_s | hr | rmssd | sdnn | mean_rr`

**Steps:** peak detect → IBI clean → sliding windows

| Variable | Window | Notes |
|----------|--------|-------|
| HR, mean_rr | 10–20 s | Local estimate |
| RMSSD, SDNN | 60 s | **Local short-window HRV**, not clinical HRV |

**Skip HRV windows with < 20 clean beats** (configurable `min_beats_hrv`). Do not use 3-beat minimum for RMSSD/SDNN.

### Output 3: `features_temporal_aligned.csv`

`dataset_id | subject_id | observation_id | time_s | hr_z | rmssd_z | sdnn_z | mean_rr_z | theta_env_z | alpha_env_z | beta_env_z`

**Steps:**
1. Intersect valid time range
2. Resample to 1–2 Hz (default 2 Hz)
3. Interpolate short gaps only — no long forward-fill
4. Z-score within subject×observation → `*_z` columns

---

## ds003838 Synchronization Warning

EEG and ECG/PPG are **separate files**. Without shared trigger metadata, assume **t = 0** alignment. **Report this assumption** in audit + methods — a 5–10 s offset could invalidate lag conclusions.

Build **rest task first**; memory task separately (task events confound coupling).

---

## Stage 2: Lagged Cross-Correlation

**Input:** `features_temporal_aligned.csv` (`*_z`). **CLI:** `--stage 2`

**9 pairs:** HR/RMSSD/SDNN × theta/alpha/beta

**Lag:** `±recommended_max_lag_s`

**Save curves:** `pair, lag_s, r` (long format)

**Peak features:** `peak_signed_r`, `peak_abs_r`, `peak_lag_s`, `peak_direction`

**Permutation null (required):** circular time-shift one signal `n_permutations` times; rebuild max |r| null; `p_perm` for observed peak.

**Outputs:** `cross_correlation_curves.csv`, `cross_correlation_peaks.csv`

---

## Stage 3: Group Analysis

**CLI:** `--stage 3`

**Outputs:** `peak_correlation_summary.csv`, `group_cross_correlation_summary.csv`

**Tests per pair:** signed peak r ≠ 0 (Fisher z or Wilcoxon); median lag ≠ 0; peak r vs permutation null; BH-FDR across 9 pairs.

**Plots:** mean ± SEM curves; peak lag/r distributions; summary heatmap.

---

## Stage 4: Event-Triggered Analysis

**CLI:** `--stage 4`. Epoch: typically **±60 s** (not ±300 s).

### Heart-led
- **ΔHR or HR slope over 10–20 s** (not top 10% absolute HR)
- Top/bottom 10% increases/decreases → average EEG envelopes

### Brain-led (on `*_z`)
- Theta burst: `theta_env_z > +2` for ≥ 2 s
- Alpha suppression: `alpha_env_z < −2` for ≥ 2 s
- Beta burst: `beta_env_z > +2` for ≥ 2 s
- Average HR trajectories

**Outputs:** `event_triggered_hr_to_eeg.png`, `event_triggered_eeg_to_hr.png`, `event_counts.csv`, `event_triggered_summary.csv`

---

## Dataset Order

1. **ds003838 rest** — ~4 min, ±60 s lag, separate EEG/ECG files
2. **ds006848** — longer rest; validate pipeline
3. **HIIT** — ~5 min epochs only; not 30-min recovery; separate PRE/POST × rest/tetris × ph/ps

---

## Config Sketch

```yaml
temporal_coupling:
  audit:
    min_overlap_s: 120
    min_clean_beats: 30
    requested_lag_max_s: 300
  eeg:
    bands: { theta: [4,8], alpha: [8,13], beta: [13,30] }
    envelope_smooth_s: 2.0
    envelope_mode: amplitude   # or power
    rois:
      theta: [Fz, FCz, Cz]
      alpha: posterior_mean      # Pz, POz, Oz or global
      beta: global_mean
  cardiac:
    hr_window_s: 15.0
    hrv_window_s: 60.0
    min_beats_hrv: 20
  resample:
    fs_hz: 2.0
  cross_correlation:
    lag_step_s: 1
    n_permutations: 500
  events:
    hr_delta_window_s: 15.0
    hr_percentile: 10
    eeg_threshold_sd: 2.0
    epoch_pre_s: 60
    epoch_post_s: 60
```

---

## Build Order

1. Config + Stage 0 audit
2. Audit ds003838 rest
3. `eeg_envelope.py`
4. `cardiac_timeseries.py`
5. `resample.py`
6. Synthetic xcorr test
7. Smoke 2–5 subjects
8. Permutation null in `cross_correlation.py`
9. `group_summary.py`
10. `events.py` (ΔHR events)
11. ds006848 → HIIT

---

## Deliverables Checklist

### Per subject
- [ ] `features_temporal_eeg_envelope.csv`
- [ ] `features_temporal_cardiac.csv`
- [ ] `features_temporal_aligned.csv`
- [ ] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`

### Per dataset (`group/`)
- [ ] `data_audit.csv`
- [ ] `group_cross_correlation_summary.csv`
- [ ] Mean xcorr curves, peak lag/r distributions, summary heatmap
- [ ] `event_triggered_*.png`, `event_triggered_summary.csv`
