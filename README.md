# EEG–PPG correlation pipeline

This repository contains a two-stage **raw-to-base-to-features** pipeline that runs across `HIIT`, `ds003838`, and `ds006848`, then computes inter-subject EEG↔PPG correlations and cross-dataset trend agreement.

## Scope

- **EEG preprocessing:**
  - 1–60 Hz bandpass
  - bad-channel removal using variance z-score threshold
  - common average reference
- **Core EEG features (v1):**
  - base EEG power: theta, alpha, beta
  - FM-theta
  - frontal beta
  - frontal alpha asymmetry
  - global alpha power (dB)
  - global beta power (dB)
- **Core PPG features (v1):**
  - mean HR
  - RMSSD
  - SDNN
  - mean RR
  - peak HR
  - reusable cleaned IBI table for deriving these features without raw PPG
- **Analysis:**
  - pairwise EEG×PPG correlations per dataset (Spearman/Pearson/Auto)
  - Benjamini-Hochberg FDR correction per dataset
  - cross-dataset trend agreement summary

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Run both stages in one command:

```bash
python -m ppg_eeg.run --config config.example.yaml
```

Or split Stage 1 and Stage 2 across separate runs (same config, same `paths.out_root`):

```bash
# Stage 1: raw → base CSVs only
python -m ppg_eeg.run --config config.example.yaml --stage 1

# Stage 2: base CSVs → derived features + correlations
python -m ppg_eeg.run --config config.example.yaml --stage 2
```

Stage 2 reads `observations_index.csv`, `features_base_eeg_power.csv`, and
`features_base_ppg_ibi.csv` from `paths.out_root/<dataset_id>/`. You can change
`features.eeg`, `features.ppg`, or correlation settings between runs without
re-extracting raw data.

### Two-Stage Workflow

1. **Stage 1: base extraction**  
   Reads raw EEG/PPG observations and writes only the reusable base inputs:
   `observations_index.csv`, `features_base_eeg_power.csv`, and
   `features_base_ppg_ibi.csv`.

2. **Stage 2: derived analysis outputs**  
   Reloads those Stage 1 CSVs from `paths.out_root/<dataset_id>/`, derives
   `features_eeg.csv`, `features_ppg.csv`, and `features_merged.csv`, then
   computes per-dataset correlations and cross-dataset trend agreement.

## Output artifacts

Per dataset, under `derivatives/<dataset_id>/`:

Stage 1:

- `observations_index.csv`
- `features_base_eeg_power.csv`
- `features_base_ppg_ibi.csv`

Stage 2:

- `features_eeg.csv`
- `features_ppg.csv`
- `features_merged.csv`
- `correlations_raw.csv`
- `correlations_fdr.csv`

Base EEG CSV schema (`features_base_eeg_power.csv`):

- `dataset_id`
- `observation_id`
- `subject_id`
- `task_label`
- `condition_label`
- `session_label`
- `modality`
- `timepoint`
- `state`
- `eeg_format`
- `eeg_path`
- `n_bad_channels`
- `eeg_error`
- `channel`
- `theta_power_uv2`
- `alpha_power_uv2`
- `beta_power_uv2`
- `l_freq`
- `h_freq`
- `reference`
- `psd_fmin`
- `psd_fmax`
- `n_fft`
- `processing_version`

Base PPG IBI CSV schema (`features_base_ppg_ibi.csv`):

- `dataset_id`
- `observation_id`
- `subject_id`
- `task_label`
- `condition_label`
- `session_label`
- `modality`
- `timepoint`
- `state`
- `eeg_path`
- `eeg_format`
- `ppg_source`
- `ppg_path`
- `ppg_format`
- `ppg_error`
- `ppg_channel`
- `ppg_segment_start_s`
- `ppg_segment_end_s`
- `sfreq`
- `peak_min_distance_s`
- `peak_height`
- `ibi_min_ms`
- `ibi_max_ms`
- `n_peaks`
- `n_ibi_raw`
- `n_ibi_clean`
- `ibi_index`
- `peak_time_relative_s`
- `peak_time_absolute_s`
- `peak_index`
- `ibi_ms_clean`
- `n_ibi_raw_invalid`
- `ibi_ms_raw_min`
- `ibi_ms_raw_max`
- `processing_version`

Cross-dataset, under `derivatives/cross_dataset/`:

- `trend_agreement.csv`
- `trend_agreement_summary.json`

## Notes

- This repo intentionally does **not** use ICA. If artifact removal beyond filtering + bad-channel handling is required, it must be added explicitly and justified.

