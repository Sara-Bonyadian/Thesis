# Implementation Plan: Temporal EEG–Cardiac Coupling

Technical plan for within-subject temporal coupling. Reads **raw EEG + raw PPG/ECG only**.

**Last updated:** Stages 0, 1a, and 1b implemented and smoke-tested on ds003838 rest (sub-033, sub-036, sub-038). Stage 1c+ pending.

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
| Per-subject first | Core math within subject; group stats in Stage 3 |
| Stage 1 = 3 CSVs | Reusable cache; Stages 2–4 avoid raw I/O |
| Visual QC | Stage 1a/1b debug plots before alignment |
| Permutation null | Peak-r across lags needs circular-shift control (Stage 2) |

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
  resample.py              # Stage 1c ⏳
  cross_correlation.py     # Stage 2 ⏳
  group_summary.py         # Stage 3 ⏳
  events.py                # Stage 4 ⏳
```

**Tests:** `tests/test_eeg_envelope.py`, `tests/test_cardiac_timeseries.py`

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

## Stage 1: Temporal Feature Extraction

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

| Band | Requested ROI | Notes |
|------|---------------|-------|
| Theta | Fz, FCz, Cz | FCz often missing → uses Fz, Cz |
| Alpha | Pz, POz, Oz | |
| Beta | Fz, FCz, Cz | FCz often missing → uses Fz, Cz |

Z-scoring deferred to Stage 1c aligned CSV.

#### Stage 1a QC outputs ✅

**Per subject:**

| File | Description |
|------|-------------|
| `eeg_envelope_debug_user_window.png` | 9-panel plot: per band (theta/alpha/beta) × raw / bandpass / envelope |
| `eeg_envelope_debug_auto_window.png` | Same layout, auto-selected center window |
| `eeg_envelope_timeseries.png` | Full-recording theta/alpha/beta envelopes |
| `eeg_psd_debug.png` | Optional PSD with band regions shaded |

**Group:** `group/eeg_envelope_qc.csv`

| Column | Description |
|--------|-------------|
| `theta/alpha/beta_roi_channels_used` | Channels actually averaged |
| `*_nan_percent`, `*_min`, `*_max`, `*_median` | Per-band stats on downsampled CSV |
| `usable_eeg_envelope` | Pass if no NaNs, non-flat, enough samples |
| `warning` | e.g. `missing_roi_channel`, `envelope_flat` |

**Debug plot config:**

```yaml
temporal_coupling:
  eeg:
    envelope_output_fs_hz: 10
    debug_plot:
      enabled: true
      start_time_s: 40
      end_time_s: 60
      auto_window_s: 20
      also_auto_window: true
      save_psd: true
      channels:
        theta: Fz
        alpha: Pz
        beta: Fz
```

If a debug `channels.*` name is missing, falls back to first available ROI channel and warns.

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
| `simple` | Detrend → z-score | Legacy height threshold (`ppg.detect_heartbeats`) |

**Auto mode:** scores each channel × detector × polarity; picks highest `quality_score` (coverage, clean IBIs, plausible HR 50–110 bpm).

**IBI limits:** `ibi_min_ms: 400`, `ibi_max_ms: 1500`

#### Stage 1b QC outputs ✅

**Per subject:**

| File | Description |
|------|-------------|
| `features_temporal_cardiac.csv` | Sliding-window cardiac features |
| `detected_peaks.csv` | All peaks with `is_accepted_peak`, `peak_y`, `ibi_ms` |
| `cardiac_channel_inventory.csv` | All channels: type, stats |
| `cardiac_channel_preview.png` | 10 s preview of all cardiac channels |
| `peak_detector_comparison.csv` | Detector × channel × polarity scores |
| `cardiac_peak_detection_debug_10s_user_window.png` | Raw + processed signal, accepted/rejected peaks |
| `cardiac_peak_detection_debug_10s_auto_window.png` | Auto window |
| `cardiac_peak_detection_debug_40s_overview.png` | 40 s overview |

**Group:** `group/cardiac_qc.csv`

| Column | Description |
|--------|-------------|
| `channel_used`, `detector_used`, `detector_polarity`, `selection_reason` | Auto-selection audit trail |
| `n_raw_peaks`, `n_clean_ibis`, `median_hr`, `min_hr`, `max_hr` | Peak-level stats |
| `percent_valid_hr`, `percent_valid_hrv` | Sliding-window coverage |
| `usable_for_hr`, `usable_for_hrv` | Gated by HR plausibility, IBI coverage, clean-IBI % |
| `warning` | e.g. `median_hr_high`, `poor_peak_coverage` |

**Cardiac debug plot config:**

```yaml
temporal_coupling:
  cardiac:
    channel: auto
    signal_type: auto
    detector: auto
    ecg:
      bandpass_hz: [5, 30]
      min_peak_distance_s: 0.45
      prominence: auto
      height: auto
      test_inverted: true
    debug_plot:
      enabled: true
      windows: [[35, 45], [50, 60], [65, 75]]
      auto_window_s: 10
      overview_window_s: 40
```

**ds003838 smoke results:** all three subjects use **ECG + ecg_rpeak**; median HR 71–79 bpm; 100% valid HR/HRV; `usable_for_hr/hrv = True`.

---

### Stage 1c: `features_temporal_aligned.csv` ⏳

**Module:** `resample.py` (not yet implemented)  
**CLI:** `--stage 1c`

**Columns:** `dataset_id | subject_id | observation_id | time_s | hr_z | rmssd_z | sdnn_z | mean_rr_z | theta_env_z | alpha_env_z | beta_env_z`

**Steps:**
1. Gate on `cardiac_qc.usable_for_hr` and `eeg_envelope_qc.usable_eeg_envelope`
2. Intersect valid time range (EEG + cardiac overlap)
3. Resample to `temporal_coupling.resample.fs_hz` (default 1–2 Hz)
4. Interpolate short gaps only — no long forward-fill
5. Z-score within subject×observation → `*_z` columns

---

## ds003838 Synchronization Warning

EEG and ECG/PPG are **separate files**. Without shared trigger metadata, assume **t = 0** alignment. **Report this assumption** in audit + methods — a 5–10 s offset could invalidate lag conclusions.

Build **rest task first**; memory task separately (task events confound coupling).

---

## Stage 2: Lagged Cross-Correlation ⏳

**Input:** `features_temporal_aligned.csv` (`*_z`). **CLI:** `--stage 2`

**9 pairs:** HR/RMSSD/SDNN × theta/alpha/beta

**Lag:** `±recommended_max_lag_s` from audit (smoke: ±60 s)

**Save curves:** `pair, lag_s, r` (long format)

**Peak features:** `peak_signed_r`, `peak_abs_r`, `peak_lag_s`, `peak_direction`

**Permutation null:** circular time-shift one signal `n_permutations` times; `p_perm` for observed peak.

**Outputs:** `cross_correlation_curves.csv`, `cross_correlation_peaks.csv`

---

## Stage 3: Group Analysis ⏳

**CLI:** `--stage 3`

**Outputs:** `peak_correlation_summary.csv`, `group_cross_correlation_summary.csv`

**Tests per pair:** signed peak r ≠ 0; median lag ≠ 0; peak r vs permutation null; BH-FDR across 9 pairs.

**Plots:** mean ± SEM curves; peak lag/r distributions; summary heatmap.

---

## Stage 4: Event-Triggered Analysis ⏳

**CLI:** `--stage 4`. Epoch: typically **±60 s**.

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

1. **ds003838 rest** — ~3–4 min, ±60 s lag, separate EEG/ECG files — **smoke in progress**
2. **ds006848** — longer rest; validate pipeline
3. **HIIT** — ~5 min epochs only; separate PRE/POST × rest/tetris × ph/ps

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

eeg:
  l_freq: 1.0
  h_freq: 60.0
  reference: average

ppg:
  ibi_min_ms: 400.0
  ibi_max_ms: 1500.0

temporal_coupling:
  audit:
    min_overlap_s: 120
    min_clean_beats: 30
  eeg:
    bands: { theta: [4,8], alpha: [8,13], beta: [13,30] }
    envelope_smooth_s: 2.0
    envelope_output_fs_hz: 10
    rois:
      theta: [Fz, FCz, Cz]
      alpha: [Pz, POz, Oz]
      beta: [Fz, FCz, Cz]
    debug_plot: { enabled: true, start_time_s: 40, end_time_s: 60, save_psd: true }
  cardiac:
    channel: auto
    detector: auto
    ecg: { bandpass_hz: [5,30], min_peak_distance_s: 0.45 }
    hr_window_s: 15.0
    hrv_window_s: 60.0
    hrv_step_s: 2.0
    debug_plot: { enabled: true }
  resample:
    fs_hz: 1.0
    z_score: true
  cross_correlation:
    lag_max_s: 60
    lag_step_s: 5
    n_permutations: 0
```

---

## Build Order

| # | Task | Status |
|---|------|--------|
| 1 | Config + Stage 0 audit | ✅ |
| 2 | Audit ds003838 rest (smoke) | ✅ |
| 3 | `eeg_envelope.py` + QC plots | ✅ |
| 4 | `cardiac_timeseries.py` + detectors + QC plots | ✅ |
| 5 | `resample.py` (Stage 1c) | ⏳ **next** |
| 6 | Synthetic xcorr test | ⏳ |
| 7 | Smoke 2–5 subjects end-to-end | ⏳ (1a/1b done) |
| 8 | Permutation null in `cross_correlation.py` | ⏳ |
| 9 | `group_summary.py` | ⏳ |
| 10 | `events.py` (ΔHR events) | ⏳ |
| 11 | ds006848 → HIIT | ⏳ |

---

## Deliverables Checklist

### Per subject
- [x] `features_temporal_eeg_envelope.csv`
- [x] `features_temporal_cardiac.csv`
- [x] `detected_peaks.csv`
- [x] EEG envelope QC plots + `eeg_envelope_qc.csv` (group)
- [x] Cardiac QC plots + `cardiac_qc.csv` (group)
- [ ] `features_temporal_aligned.csv`
- [ ] `cross_correlation_peaks.csv`, `cross_correlation_curves.csv`

### Per dataset (`group/`)
- [x] `data_audit.csv`
- [x] `eeg_envelope_qc.csv`
- [x] `cardiac_qc.csv`
- [ ] `group_cross_correlation_summary.csv`
- [ ] Mean xcorr curves, peak lag/r distributions, summary heatmap
- [ ] `event_triggered_*.png`, `event_triggered_summary.csv`
