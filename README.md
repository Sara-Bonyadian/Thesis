# EEG–PPG correlation pipeline

This repository contains a **raw-to-features** pipeline that runs across `HIIT`, `ds003838`, and `ds006848`, then computes inter-subject EEG↔PPG correlations and cross-dataset trend agreement.

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

```bash
python -m ppg_eeg.run --config config.example.yaml
```

### Feature Iteration Workflow

1. **Base extraction run**  
   Use `output.eeg_base_csv_mode: base_only` to write `features_base_eeg_power.csv`.

2. **Later runs with new EEG features**  
   Add new feature names under `features.eeg`, then choose:
   - `output.eeg_base_csv_mode: append_columns` to add/update feature columns in the same base CSV.
   - `output.eeg_base_csv_mode: new_file` to keep base CSV unchanged and write an expanded file using `output.eeg_base_csv_new_file_name`.

To reuse previously extracted feature CSVs (and avoid recomputing from raw files), set:

- `features.reuse_eeg_features_csv: true` to load `features_core_eeg.csv`
- `features.reuse_ppg_features_csv: true` to load `features_core_ppg.csv`

Both are loaded from `paths.out_root/<dataset_id>/`.

## Output artifacts

Per dataset, under `derivatives/<dataset_id>/`:

- `observations_index.csv`
- `features_base_eeg_power.csv`
- `features_base_eeg_power_extended.csv` (only when `output.eeg_base_csv_mode: new_file`)
- `features_core_eeg.csv`
- `features_core_ppg.csv`
- `features_core_merged.csv`
- `correlations_raw.csv`
- `correlations_fdr.csv`

Base EEG CSV schema (`features_base_eeg_power.csv`):

- `dataset_id`
- `observation_id`
- `subject_id`
- `task_label`
- `modality`
- `state`
- `eeg_format`
- `eeg_path`
- `n_bad_channels`
- `eeg_error`
- `channel`
- `power_theta`
- `power_alpha`
- `power_beta`

Cross-dataset, under `derivatives/cross_dataset/`:

- `trend_agreement.csv`
- `trend_agreement_summary.json`

## Notes

- This repo intentionally does **not** use ICA. If artifact removal beyond filtering + bad-channel handling is required, it must be added explicitly and justified.

