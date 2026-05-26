# EEG–PPG correlation pipeline (no ICA)

This repository contains a **raw-to-features** pipeline to compute inter-subject correlations between EEG spectral/features and PPG/HRV features.

## Scope

- **EEG preprocessing :**
- 1–60 Hz bandpass,
- bad-channel removal using a z-score threshold on channel variance,
- common average reference.

- **EEG spectral/features:** 
- Welch PSD (log power in dB), 
- band powers (delta/theta/alpha/beta) and higher-level EEG features (e.g., PAF, FMθ, FAA) as configured.
- **PPG features (when present in the recording):** 
- heartbeat peak detection → IBI → rHR, rMSSD, SDNN, mean RR, peak HR, pulse amplitude change, autonomic reactivity indices.
- **Analysis:** inter-subject correlation matrix (Pearson/Spearman with normality check + FDR).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (planned entrypoint)

```bash
python -m ppg_eeg.run --config config.yaml
```

## Notes

- This repo intentionally does **not** use ICA. If artifact removal beyond filtering + bad-channel handling is required, it must be added explicitly and justified.

