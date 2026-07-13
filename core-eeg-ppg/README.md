# Core EEG–PPG correlation pipeline

Configs for the original **raw → base → features → inter-subject correlation** pipeline (`python -m ppg_eeg.core_eeg_ppg`).

Code lives in `ppg_eeg/core_eeg_ppg/`.

This is separate from temporal coupling (exploratory Stages 0–4 and confirmatory zero-lag).

## Layout

```text
core-eeg-ppg/
  example.yaml   # template
  datasets/      # full runs
  smoke/         # small subsets
  fast/          # faster / reduced-PSD presets
  README.md
```

## Example

```bash
.venv/bin/python -m ppg_eeg.core_eeg_ppg --config core-eeg-ppg/example.yaml
.venv/bin/python -m ppg_eeg.core_eeg_ppg --config core-eeg-ppg/smoke/ds003838.yaml --stage 1
```

Run from the **repo root** so `./data` and `./derivatives` paths resolve correctly.
