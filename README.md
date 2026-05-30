# EEG–PPG correlation pipeline

This repository contains a **raw-to-features** pipeline that runs across `HIIT`, `ds003838`, and `ds006848`, then computes inter-subject EEG↔PPG correlations and cross-dataset trend agreement.

## Scope

- **EEG preprocessing:**
  - 1–60 Hz bandpass
  - bad-channel removal using variance z-score threshold
  - common average reference
- **Core EEG features (v1):**
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
  - pairwise EEG×PPG correlations per dataset (Spearman/Pearson)
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

## Output artifacts

Per dataset, under `derivatives/<dataset_id>/`:

- `observations_index.csv`
- `features_core_eeg.csv`
- `features_core_ppg.csv`
- `features_core_merged.csv`
- `correlations_raw.csv`
- `correlations_fdr.csv`

Cross-dataset, under `derivatives/cross_dataset/`:

- `trend_agreement.csv`
- `trend_agreement_summary.json`

## Notes

- This repo intentionally does **not** use ICA. If artifact removal beyond filtering + bad-channel handling is required, it must be added explicitly and justified.

