---
name: Zero Lag Redesign
overview: Incrementally add a confirmatory Zero-Lag EEG–Heart Rate analysis path beside the existing exploratory Stages 0–4, reusing dataset discovery, beat detections, alignment primitives, cross-correlation math, QC conventions, and plotting utilities. The redesign regenerates EEG spectral features from raw EEG, reconstructs instantaneous HR from saved beats, introduces harmonized duration/lag rules and ZLPI-centered inference, and preserves all legacy outputs for regression and provenance.
todos:
  - id: m1-audit-config
    content: Define confirmatory schemas, protocol audit, pairing keys, and eligibility outputs
    status: completed
  - id: m2-instant-hr
    content: Build and validate instantaneous HR from saved beat detections
    status: completed
  - id: m3-multitaper
    content: Build and validate four-band multitaper EEG power extraction
    status: completed
  - id: m4-harmonize
    content: Implement common support and deterministic nested duration segments
    status: completed
  - id: m5-correlations
    content: Adapt signed HR-only lag curves while preserving legacy Stage 2
    status: completed
  - id: m6-endpoints
    content: Implement Fisher-z, ZLPI, local prominence, and endpoint QC
    status: completed
  - id: m7-peak-model
    content: Implement weighted Gaussian fits and hierarchical peak inference
    status: completed
  - id: m8-group-tables
    content: Build normalized participant tables and paired state contrasts
    status: completed
  - id: m9-nulls
    content: Implement deterministic temporal surrogate and mismatch analyses
    status: completed
  - id: m10-inference
    content: Implement mixed models, equivalence, meta-analysis, LOO, and FDR
    status: completed
  - id: m11-sensitivities
    content: Implement artifact, nuisance, duration, and modality sensitivities
    status: completed
  - id: m12-reporting
    content: Generate publication figures, result tables, reports, and manifests
    status: completed
  - id: m13a-production-preflight
    content: "M13a: production preflight + synthetic smoke orchestration (modes, schemas, manifests)"
    status: completed
  - id: m13b-hiit-smoke
    content: "M13b: HIIT real-smoke config (PPG/photosensor, PRE/POST×PH/PS); operator runs stages"
    status: in_progress
  - id: c-stage-cli
    content: "Architecture: confirmatory/__main__.py + run.py with stages C0,C1a,C1b,C1c,C2,C3,C4,C5,C6,C7,all"
    status: completed
  - id: m13c-primary-run
    content: "M13c: freeze configs; run primary cohorts once raw data complete; no hypothesis changes mid-run"
    status: pending
  - id: m13d-sensitivity-cleanroot
    content: "M13d: sensitivity cohorts + clean-root reproducibility; lock hashes; manuscript tables"
    status: pending
isProject: false
---

# Zero-Lag EEG–Heart Rate Pipeline Implementation Plan

## Status (2026-07-13)

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1–M12 | **Done** | Package under `ppg_eeg/temporal_coupling/confirmatory/`; configs in `zero-lag-reanalysis-repo/` |
| M13a | **Done** | `production.py` modes: preflight / smoke / primary / sensitivity / clean_root; synthetic E2E smoke; artifacts under `derivatives/confirmatory_temporal_coupling/production/` |
| M13b | **In progress** | HIIT smoke configs + stage CLI ready; **operator must run** stages (no full cohort yet) |
| C-stage CLI | **Done** | `python -m ppg_eeg.confirmatory --config … --stage C0…C7|all` (+ `--mode` for M13a ops) |
| M13c–d | **Blocked** | Primary raw missing for `ds003838`, `ds006848`, `ds003690` (and sensitivity `ds004582`) |

**Duration contracts (frozen):** D240/D180 → ZLPI (±60); D120 → MWPI; D60 → SWPI; never pool MWPI/SWPI with ZLPI. Primary = D240 absolute-power ZLPI.

## 1. Executive summary

The current pipeline asks where the largest signed or absolute EEG–cardiac correlation occurs across lags, using Hilbert EEG envelopes and windowed HR/HRV. The confirmatory pipeline instead asks whether **instantaneous HR and frequency-resolved EEG power have a specific zero-lag excess** after accounting for distant-lag background correlation.

Build a parallel confirmatory path under [`ppg_eeg/temporal_coupling/confirmatory/`](ppg_eeg/temporal_coupling/confirmatory/) and keep the current CLI and Stages 0–4 operational. Reuse observation adapters, Stage 0 metadata, saved beat detections, preprocessing primitives, output-layout helpers, lag-correlation functions, BH-FDR logic, circular-shift primitives, and plotting conventions. Replace only the confirmatory feature and inference layers:

- Hilbert envelopes → 2 s / 1 s-step DPSS multitaper power in theta 4–7, alpha 8–12, beta 13–29, gamma 30–45 Hz.
- 15 s windowed HR + HRV → PCHIP instantaneous HR reconstructed from clean accepted beats.
- Exploratory peak search → signed HR-only curves on a fixed ±60 s, 1 s lag grid.
- Peak-r inference → Fisher-z, ZLPI, local prominence, Gaussian peak parameters, paired contrasts, mixed models, random-effects meta-analysis, LOO, and temporal nulls.
- Existing Stage 4 event analysis → retained as legacy/supplementary only; it is not a dependency of the confirmatory path.

```mermaid
flowchart TD
  RawMeta["Raw metadata and adapters"] --> C0["C0 protocol audit"]
  SavedBeats["Saved beat detections"] --> C1b["C1b instantaneous HR"]
  RawEEG["Raw EEG"] --> C1a["C1a multitaper power"]
  C0 --> C1a
  C0 --> C1b
  C1a --> C1c["C1c common support and nested segments"]
  C1b --> C1c
  C1c --> C2["C2 signed lag curves"]
  C2 --> C3["C3 ZLPI and peak models"]
  C1c --> C4["C4 temporal nulls"]
  C3 --> C5["C5 paired and group tables"]
  C4 --> C5
  C5 --> C6["C6 mixed models and meta-analysis"]
  C6 --> C7["C7 figures, report, manifest"]
```

## 2. Gap analysis and disposition of existing stages

### Stage 0 — data audit: Modify

- Reuse [`data_audit.py`](ppg_eeg/temporal_coupling/data_audit.py), all adapters in [`ppg_eeg/datasets/`](ppg_eeg/datasets/), and `data_audit.csv` schema.
- Add a confirmatory audit that joins raw overlap, channel inventory, beat span, clean-beat count, nuisance channels, line frequency, paired-state availability, and eligibility at D240/D180/D120/D60.
- Add explicit exclusion codes: raw-short, beat-span-short, missing-state, sparse-beat, missing-EEG-feature, protocol-window, and modality-QC failure.
- Do not make Stage 0 depend on existing `usable_for_xcorr`, because that flag reflects windowed HR/HRV and exploratory envelope requirements.

### Stage 1A — EEG envelopes: Replace for confirmatory, preserve for legacy

- Keep [`eeg_envelope.py`](ppg_eeg/temporal_coupling/eeg_envelope.py) unchanged for exploratory reproducibility.
- Reuse `_read_raw`, `eeg_segment_bounds`, `preprocess_eeg`, ROI/config patterns, observation iteration, QC/error handling, and per-observation output layout.
- Add `confirmatory/multitaper_power.py`; do not add gamma to `eeg_envelope.py` or overload Hilbert output schemas.

### Stage 1B — cardiac HR/HRV: Modify and split

- Reuse detector/channel-selection paths in [`cardiac_detectors.py`](ppg_eeg/temporal_coupling/cardiac_detectors.py), IBI cleaning in [`cardiac_common.py`](ppg_eeg/temporal_coupling/cardiac_common.py), and `detected_peaks.csv` written by [`cardiac_timeseries.py`](ppg_eeg/temporal_coupling/cardiac_timeseries.py).
- Add `confirmatory/instant_hr.py`, which reads beat derivatives and never calls `compute_cardiac_timeseries()` for confirmatory HR.
- Keep RMSSD/SDNN files and code for legacy analyses, but remove HRV from confirmatory variable families and reports.

### Stage 1C — alignment: Modify

- Reuse `build_time_grid`, interpolation helpers, overlap calculations, z-score helpers, and QC conventions in [`resample.py`](ppg_eeg/temporal_coupling/resample.py).
- Add `confirmatory/harmonize.py` for feature intersection, clean contiguous blocks, deterministic nested segments, power-representation construction, and confirmatory QC.
- Do not force confirmatory columns into the legacy `features_temporal_aligned.csv` schema.

### Stage 1D — HIIT combine: Modify only for sensitivity

- Reuse participant/session parsing and source provenance from [`hiit_combine.py`](ppg_eeg/temporal_coupling/hiit_combine.py).
- Do not concatenate rest and Tetris into one signal before confirmatory estimation. Compute each state separately, then pair metrics within participant-session.

### Stage 2 — lagged cross-correlation: Modify

- Reuse `build_lag_grid`, `correlate_at_lag`, `compute_correlation_curve`, sign convention, finite-value masking, and overlap counts in [`cross_correlation.py`](ppg_eeg/temporal_coupling/cross_correlation.py).
- Add a configurable pair list and confirmatory writer rather than changing legacy `CARDIAC_VARS`/`EEG_VARS` globally.
- Disable exploratory positive-only/absolute peak selection for confirmatory inference. Save signed HR × band-power curves and per-lag `n_overlap`.

### Stage 3 — group summary: Split

- Reuse mean/SEM curve aggregation, common-support plotting concepts, bootstrap utilities, and `_apply_bh_fdr` from [`group_summary.py`](ppg_eeg/temporal_coupling/group_summary.py).
- Add `confirmatory/endpoints.py`, `peak_model.py`, `group_tables.py`, and `inference.py`; do not append confirmatory columns to exploratory peak-summary schemas.
- Refactor circular-shift code behind a statistic callback so the null statistic can be ZLPI instead of maximum absolute r.

### Stage 4 — event-triggered analyses: Remove from confirmatory DAG

- Keep [`events.py`](ppg_eeg/temporal_coupling/events.py) unchanged and label its outputs exploratory/supplementary.
- Do not spend confirmatory compute or multiplicity budget on Stage 4 unless it is later requested as an optional supplement.

### Configuration: Modify

- Preserve exploratory configs under `exploratory-temporal-coupling/` as legacy provenance.
- Add confirmatory configs under [`zero-lag-reanalysis-repo/`](zero-lag-reanalysis-repo/): `master.yaml`, `datasets/<id>.yaml`, `smoke/<id>.yaml` (HIIT smoke package in `smoke/hiit/`).
- Add `confirmatory/config.py`; parse shared dataset/path/filter fields through the current loader and strictly validate a new `confirmatory:` block.

### QC: Modify and add

- Reuse channel inventories, detector comparisons, beat plots, preprocessing warnings, and group QC file patterns.
- Add QC for beat coverage, HR interpolation gaps, multitaper spectral leakage, band/ROI channel coverage, common support, selected segment, ZLPI flank completeness, and Gaussian-fit convergence.

### Visualization and manuscript outputs: Replace for confirmatory

- Reuse plotting style and bootstrap helpers from [`directional_group_plot.py`](ppg_eeg/temporal_coupling/directional_group_plot.py) and [`group_summary.py`](ppg_eeg/temporal_coupling/group_summary.py).
- Keep [`temporal_coupling/generate_cross_dataset_report_html.py`](temporal_coupling/generate_cross_dataset_report_html.py) and current Russell report files as legacy artifacts.
- Add confirmatory figure/report generation driven only by tidy confirmatory outputs.

## 3. Architecture redesign

Add these modules (status as of 2026-07-13):

| Module | Role | Status |
|--------|------|--------|
| `confirmatory/__main__.py` | CLI with stages `C0`, `C1a`, `C1b`, `C1c`, `C2`, `C3`, `C4`, `C5`, `C6`, `C7`, `all` | **Done** — `python -m ppg_eeg.confirmatory --config … --stage C0|…|all`; production `--mode` still available |
| `confirmatory/run.py` | Stage dependency checks and resumable dispatch; no implicit deletion or overwrite | **Done** |
| `confirmatory/config.py` | Immutable dataclasses, master/dataset config loading, cross-field validation | Done |
| `confirmatory/protocol_audit.py` | Eligibility, nuisance inventory, participant-key normalization, paired sets | Done (`C0` / M1; raw overlap from in-C0 `data_audit`) |
| `confirmatory/data_audit.py` | Observation-level raw EEG–cardiac audit written under `C0/` | Done — no separate temporal_coupling Stage 0 required |
| `confirmatory/instant_hr.py` | Derivative-based beat-to-HR reconstruction and QC | Done (`C1b` / M2) |
| `confirmatory/multitaper_power.py` | Raw EEG preprocessing and DPSS features | Done (`C1a` / M3) |
| `confirmatory/harmonize.py` | 1 Hz alignment, clean blocks, nested duration windows, representations, z-scoring | Done (`C1c` / M4) |
| `confirmatory/correlation.py` | Thin adapter around reusable lag functions and HR-only pair schemas | Done (`C2` / M5) |
| `confirmatory/endpoints.py` | Fisher-z, ZLPI, local prominence, endpoint QC | Done (`C3` endpoints / M6) |
| `confirmatory/peak_model.py` | Weighted Gaussian fitting and hierarchical parameter tables | Done (`C3` peaks / M7) |
| `confirmatory/nulls.py` | Circular shift, phase randomization, block shuffle, cross-subject mismatch, AR(1) innovations | Done (`C4` / M9) |
| `confirmatory/group_tables.py` | Subject/state/band tidy tables and paired contrasts | Done (`C5` / M8) |
| `confirmatory/inference.py` | Mixed models, equivalence, dataset effects, random-effects meta-analysis, LOO, multiplicity | Done (`C6` / M10) |
| `confirmatory/artifact_controls.py` | QRS/ICA/nuisance/modality sensitivity orchestration | Done (M11; expose under `C6` sensitivities or dedicated flag) |
| `confirmatory/figures.py` | Figures 1–3 and supplementary duration/QC panels | Done (`C7` / M12) |
| `confirmatory/manifest.py` | Provenance, hashes, seeds, versions, inclusion counts | Done (`C7` / M12) |
| `confirmatory/report.py` | Methods/results-ready tables and machine-readable result bundle | Done (`C7` / M12) |
| `confirmatory/production.py` | Production preflight / synthetic smoke / run-plan orchestration (M13a) | Done (ops layer; not a substitute for C-stage CLI) |
| `confirmatory/hiit_smoke_m13b.py` | Temporary HIIT M13b operator driver | Interim until C-stage `__main__.py` covers real-data smoke end-to-end |

### Intended C-stage CLI (still the architecture target)

```text
python -m ppg_eeg.confirmatory \
  --config zero-lag-reanalysis-repo/datasets/<dataset>.yaml \
  --stage C0|C1a|C1b|C1c|C2|C3|C4|C5|C6|C7|all
```

| Stage | Name | Implements |
|-------|------|------------|
| `C0` | Protocol + raw-data + duration eligibility audit | `protocol_audit.py`, `data_audit.py` |
| `C1a` | Multitaper EEG power | `multitaper_power.py` |
| `C1b` | Instantaneous HR | `instant_hr.py` (from saved beats) |
| `C1c` | Harmonize / nested durations | `harmonize.py` |
| `C2` | Signed lag curves | `correlation.py` |
| `C3` | Endpoints + peaks | `endpoints.py`, `peak_model.py` |
| `C4` | Temporal nulls | `nulls.py` |
| `C5` | Group / paired tables | `group_tables.py` |
| `C6` | Inference (+ sensitivities) | `inference.py`, `artifact_controls.py` |
| `C7` | Figures, report, manifest | `figures.py`, `report.py`, `manifest.py` |
| `all` | Dependency-ordered full run | `run.py` dispatch |

Production modes (`--mode preflight|smoke|…`) stay as an **ops** entrypoint for cohort gating; they should call the same C-stage runner rather than reimplement stages.

Preserve the current observation layout helper in [`paths.py`](ppg_eeg/temporal_coupling/paths.py), but add confirmatory path constants so legacy and confirmatory outputs cannot collide.

## 4. Dataset audit plan

For every adapter, C0 must emit a completed checklist with: source files, task/state labels, normalized participant/session/run key, low-demand state, effort state, raw overlap, clean-beat span, projected instant-HR overlap, D240/D180/D120/D60 eligibility, ECG/PPG modality, line frequency, EOG/EMG/respiration/motion channels, pair membership, and exclusion code.

### ds003838

- Low demand: `rest`; effort: `memory`; pair on `subject_id`.
- Primary external ECG; EEG and ECG are split EEGLAB files.
- Confirm D240 paired set from actual reconstructed HR, not prior projected estimates; retain D180 as mandatory nested sensitivity.
- Expected planning baseline: about 41 pairs at conservative D240 and all 65 at D180; actual C0/C1b output is authoritative.

### ds006848

- Low demand: `rest`; effort: `verbalwm`; pair on `subject_id`.
- Inventory both embedded ECG and PPG; ECG is confirmatory primary and PPG is modality sensitivity.
- Verify low-demand sample size and missing-state reasons from adapter output rather than documentation labels.

### ds003690

- Low demand: `passive`; effort contrasts: `simplert` and `gonogo`, tested separately.
- Normalize run-level keys to the underlying participant while preserving run in provenance.
- Embedded EKG is primary; inventory VEO/HEO as nuisance channels.
- Ensure one observation per participant/state or prespecify run aggregation before pairing; use within-participant mean endpoint if multiple eligible runs remain.

### HIIT

- Low demand: PRE/POST rest; effort: matched PRE/POST Tetris.
- Pair within participant × protocol session (PH/PS) × timepoint; include session as repeated-measure level.
- PPG remains the primary sensitivity modality because existing validated run uses PPG; process ECG as a prespecified sensor-modality sensitivity when channel QC passes; inventory respiration.
- Keep HIIT outside the four-dataset primary confirmatory family.

### ds004582

- Single `ff` state; no paired effort contrast.
- ECGBIT/ECG primary; inventory respiration.
- Use as external low-demand/single-state replication only, not for task-attenuation claims.

### ds004587

- Low demand: 8-minute eyes-closed `rest`; effort: `ig`.
- Preserve the configured first-480-s EEG crop for rest.
- Normalize IG run suffix for session-level pairing while preserving original observation IDs.
- ECGBIT is primary; verify 100 paired sessions and duration eligibility from C0.

### ds003816

- Embedded ECG; resting and short task states exist, but clean beat spans are frequently short.
- Resolve an inconsistency in the earlier specification: D60 cannot support ±60-s lags or the original 20–60-s ZLPI flank. Therefore exclude ds003816 from confirmatory ZLPI, mixed models, and meta-analysis.
- Retain only a clearly labeled supplementary feasibility analysis with D60, lags ±20 s, and a separate short-window index using 10–20-s flanks. Never pool that index with ZLPI.

### Mindfulness

- Steps 1–3 are graded sensitivity states, not assumed paired until C0 strips task suffixes and verifies a stable underlying participant/session key.
- Existing PPG run is primary sensitivity; inventory co-recorded ECG and process it as modality sensitivity when complete.
- Keep outside the primary state-attenuation family unless pairing audit confirms identical participants and interpretable ordered protocol states.

## 5. Signal processing plan

### Instantaneous HR

- Select rows with accepted peaks and clean IBIs from `detected_peaks.csv`; define beat HR as `60_000 / ibi_ms` at the later beat time.
- Reject nonfinite values and retain the configured physiological IBI bounds; do not silently repair rejected beats.
- Interpolate clean beat HR to the 1 Hz grid with `scipy.interpolate.PchipInterpolator`, without extrapolation.
- Use a 1-s bilateral physiological edge after first/before last valid beat and mask intervals spanning a beat gap greater than 5 s.
- Save source beat indices and interpolation masks so every HR sample is traceable.
- Validate against constant-IBI and known-ramp synthetic beats and compare summary HR to legacy outputs as a sanity check only.

### EEG power

- Reuse `preprocess_eeg`: 1–60 Hz filter, bad-channel variance z=3, average reference, and task-specific cropping.
- Audit and apply line-frequency notch from metadata before gamma estimation; record actual notch frequencies.
- Use 2-s windows, 1-s steps, DPSS time-bandwidth product 3, five tapers, and ROI-mean power.
- Bands: theta 4–7, alpha 8–12, beta 13–29, gamma 30–45 Hz. Gamma ROI defaults to the beta frontal-central ROI and is explicit in config.
- Produce log10 absolute power as the primary representation; relative power and broadband-residualized log power are secondary. Fit broadband residualization within each selected segment to prevent cross-state leakage.
- Primary analysis uses the current preprocessing without ICA. ICA is a separate sensitivity path so component decisions cannot alter the primary cohort.

### Harmonization and duration

- Intersect finite multitaper and instantaneous-HR support at 1 Hz; split at gaps >5 s; select the longest clean contiguous block, with earliest-start tie-break.
- Use D240 as primary. Build D180, D120, and D60 as centered windows nested on the same temporal center; this makes duration contrasts paired and avoids quality-driven window selection.
- Do not select windows by maximum beat density, because that would condition on cardiac physiology.
- Z-score each variable after segment selection and separately for each duration.
- D240/D180/D120 use ±60-s lag support; require at least 60 finite overlapping points at every retained lag. D60 uses only the ds003816 supplementary ±20-s analysis.

## 6. Statistical redesign

### Endpoints

- Fisher transform: clip r to ±0.999999 then apply `atanh`.
- Primary ZLPI: `z(r0) - mean(z(rτ))` over integer lags `-60..-20` and `20..60`; require complete predefined flank support for primary analysis rather than silently changing the denominator.
- Local prominence: `z(r0) - max(mean(z at +5..+15), mean(z at -15..-5))`.
- Save r0, flank mean, left/right shoulders, all overlap counts, and eligibility flags.

### Peak model and hierarchy

- Fit `z(τ) = C + A exp(-(τ-μ)^2/(2σ²))` by weighted nonlinear least squares, weighting by per-lag overlap.
- Constrain μ to ±20 s for the zero-lag confirmatory fit, A ≥ 0, and σ to 1–60 s; save convergence, boundary, covariance, and residual diagnostics. FWHM is `2.355σ`.
- Implement hierarchical estimation as an explicit two-stage frequentist model: subject-level weighted fits followed by mixed-effects models on A, μ, and log-FWHM with state/band fixed effects and subject/dataset random intercepts. Do not introduce an unplanned Bayesian dependency.
- Test μ equivalence to zero using two one-sided tests with bounds -2 and +2 s; report CI and both one-sided p-values.

### State models

- Build one row per dataset × normalized participant × observation/state × duration × band × power representation × modality.
- Primary mixed model: absolute-power ZLPI with state, band, and their interaction; participant random intercept, plus dataset random intercept or dataset fixed effects if MixedLM convergence requires. Include beat density and usable clean span as prespecified nuisance covariates; do not include post-selection exclusion codes as numeric predictors.
- Run paired dataset contrasts from intersection tables before pooled models.

### Meta-analysis

- Compute each dataset/state contrast and sampling variance from paired subject endpoints.
- Use random-effects meta-analysis with REML when supported by `statsmodels`; otherwise use Paule–Mandel already available in `statsmodels.stats.meta_analysis`, recording the estimator in outputs.
- Produce heterogeneity Q, τ², I², prediction interval, and leave-one-dataset-out results.
- ds003816 and unpaired single-state datasets do not enter task-attenuation meta-analysis.

### Temporal nulls

- Circular shift: integer shifts uniformly sampled from valid offsets at least 60 s from zero, with wraparound; statistic is ZLPI.
- Phase randomization: preserve each EEG series amplitude spectrum and conjugate symmetry while randomizing phases.
- Block shuffle: permute nonoverlapping 30-s EEG blocks and reject identity order; trim only incomplete terminal block consistently.
- Cross-subject mismatch: seeded derangements within dataset × state × modality × duration.
- Innovations: fit AR(1) separately to HR and EEG and recompute curves from innovations.
- Use 1,000 surrogates for final runs and 20 for smoke tests; derive deterministic seeds from a stable SHA-256 hash of observation ID, analysis key, and null type—not Python's process-randomized `hash()`.

### Multiplicity

- Use BH-FDR at 0.05 in prespecified, nonoverlapping families:
  - Primary low-demand ZLPI: 4 bands × 4 primary paired datasets.
  - Primary state attenuation: 4 bands × 5 dataset contrasts (ds003690 contributes two contrasts).
  - Meta-analytic band effects: 4 tests.
  - μ equivalence and local-prominence families reported separately.
  - Null-type robustness is a separate family by band and dataset.
- Absolute power/D240 is the only primary representation-duration combination. Relative/residualized power and D180/D120 are sensitivity families and cannot rescue a failed primary result.

## 7. Figure redesign

### Figure 1 — lag-resolved zero-lag structure

- Inputs: D240 absolute-power signed Fisher-z curves, common-support counts, dataset/state labels.
- Panels: pooled low-demand curves/surfaces by band plus dataset-specific anchor panels and overlap-count inset.
- Statistics: mean/CI curve, r0, flank region shading, Gaussian A/μ/FWHM overlays.
- Reuse: curve aggregation and plotting conventions from `group_summary.py` and `directional_group_plot.py`.
- New: fixed zero/flank annotations, Fisher-z scale, band facets, common-support QC.

### Figure 2 — state attenuation and replication

- Inputs: `subject_level_metrics.csv`, `paired_contrasts.csv`, dataset effects, meta-analysis.
- Panels: paired ZLPI distributions for ds003838, ds006848, ds003690 contrasts, ds004587, plus forest/meta and D180 sensitivity.
- New: participant-linked paired plots, forest plot, dataset weights/heterogeneity annotations.

### Figure 3 — temporal and artifact specificity

- Inputs: observed/null ZLPI, modality, ICA/QRS/nuisance/broadband controls.
- Panels: null distributions, observed-vs-null effect ratios, duration robustness, ECG-vs-PPG, broadband/ICA controls, LOO summary.
- Reuse: bootstrap/CI utilities only; statistical content is new.

Generate vector PDF/SVG plus 300-dpi PNG and a `figure_source_manifest.json` listing exact input hashes and analysis keys.

## 8. Configuration redesign

Create a master config containing global immutable defaults and a list of dataset-config paths. Dataset configs may override only source/channel/protocol fields, not primary endpoints.

Required validated fields:

- Bands and ROI channel lists, including explicit gamma ROI.
- Multitaper window/step/time-bandwidth/taper count.
- Instant-HR IBI limits, gap limit, edge, interpolation method.
- Durations `[240,180,120,60]`, primary duration 240, nested-window rule.
- Lag limits/step, ZLPI flank 20–60, shoulder 5–15, μ equivalence ±2.
- Primary representation/modalities/datasets and sensitivity declarations.
- Null types, permutation counts, stable root seed.
- Pairing-key normalizers and state contrast definitions.
- Multiplicity-family IDs.

Validation must reject: D ≤ max lag for confirmatory ZLPI, flank outside lag grid, noninteger 1-Hz endpoint windows, missing gamma ROI, sensitivity dataset accidentally marked primary, output root colliding with a legacy root, or any unknown YAML key. The last rule closes the verified legacy gap where `events.subject_balanced_average` appears in YAML but is silently ignored by the current parser.

## 9. Repository refactoring plan

### Remove

- Remove **no files** in the initial redesign. Existing code and report artifacts are required for regression and provenance.
- Remove legacy HRV/event/abs-peak stages only from the confirmatory dependency graph and manuscript claims, not from the repository.

### Modify

- [`cross_correlation.py`](ppg_eeg/temporal_coupling/cross_correlation.py): expose reusable pair-agnostic curve and surrogate primitives without changing legacy defaults.
- [`resample.py`](ppg_eeg/temporal_coupling/resample.py): expose generic alignment helpers if needed; keep legacy writers stable.
- [`paths.py`](ppg_eeg/temporal_coupling/paths.py): add collision-safe confirmatory output helpers.
- [`requirements.txt`](requirements.txt): no new statistical package is required initially; only pin/document versions in the reproducibility manifest. Add dependencies only if implementation proves existing APIs insufficient.

### Split

- Do not expand [`group_summary.py`](ppg_eeg/temporal_coupling/group_summary.py); extract or wrap generic FDR/bootstrap utilities into `confirmatory` helpers while preserving imports/tests.
- Do not expand [`cardiac_timeseries.py`](ppg_eeg/temporal_coupling/cardiac_timeseries.py); instantaneous HR belongs in its own module.

### Rename

- Rename no existing files or outputs. New outputs use `confirmatory_` prefixes or the separate confirmatory root.

### Add tests

- `tests/test_confirmatory_config.py`
- `tests/test_protocol_audit.py`
- `tests/test_duration_contracts.py`
- `tests/test_instant_hr.py`
- `tests/test_multitaper_power.py`
- `tests/test_confirmatory_harmonize.py`
- `tests/test_confirmatory_correlation.py`
- `tests/test_zlpi.py`
- `tests/test_peak_model.py`
- `tests/test_group_tables.py`
- `tests/test_temporal_nulls.py`
- `tests/test_confirmatory_inference.py`
- `tests/test_artifact_controls.py`
- `tests/test_confirmatory_outputs.py`
- `tests/test_production_preflight.py` (M13a synthetic E2E)
- `tests/test_m13b_hiit_smoke_config.py` (M13b config/verification lock)

## 10. Milestone roadmap

### M1 — Freeze schemas and protocol audit (Medium)

- Files: confirmatory config, paths, protocol audit, master/dataset configs, config/audit tests.
- Dependencies: existing adapters and audits.
- Outputs: `protocol_audit.csv`, `paired_subject_sets.json`, `eligibility_by_duration.csv`, nuisance inventory.
- Validation: adapter fixture tests, key normalization tests, expected-count snapshot with differences explained rather than blindly accepted.

### M2 — Instantaneous HR derivative path (Medium)

- Files: `instant_hr.py`, tests, C1b dispatch.
- Dependencies: M1 and existing `detected_peaks.csv` schema.
- Outputs: per-observation `features_instant_hr.csv`, `instant_hr_qc.csv`.
- Validation: constant/ramped IBI numerical tests, no-extrapolation/gap tests, ds003838 duration audit proving removal of the legacy 30-s edge artifact.

### M3 — Multitaper EEG feature path (Large)

- Files: `multitaper_power.py`, spectral QC, tests.
- Dependencies: M1; raw EEG access.
- Outputs: `features_multitaper_power.csv`, `multitaper_qc.csv`, diagnostic PSD/band plots.
- Validation: synthetic sine mixtures recover correct bands; gamma leakage/line-noise checks; channel/ROI failure tests; smoke subjects across EEGLAB and BrainVision.

### M4 — Harmonization and nested duration segments (Medium)

- Files: `harmonize.py`, generic resample exposure, tests.
- Dependencies: M2/M3.
- Outputs: `features_confirmatory_aligned_D*.csv`, `segment_manifest.json`, `alignment_qc_confirmatory.csv`.
- Validation: exact 1-Hz grids, nested centers, deterministic tie-break, gaps, full support and exclusion-code tests.

### M5 — Signed HR-only lag curves (Medium)

- Files: confirmatory correlation adapter and minimal generic changes to `cross_correlation.py`.
- Dependencies: M4.
- Outputs: `confirmatory_cross_correlation_curves_D*.csv`.
- Validation: synthetic known zero/positive/negative shifts, exact sign convention, 121 lags for D240/D180/D120, overlap counts, legacy Stage 2 regression suite unchanged.

### M6 — ZLPI and local prominence (Small)

- Files: `endpoints.py`, tests.
- Dependencies: M5.
- Outputs: per-observation endpoint metrics and QC.
- Validation: hand-calculated curves, clipping, missing flank rejection, asymmetrical shoulders, null/flat curves.

### M7 — Gaussian and hierarchical peak models (Large)

- Files: `peak_model.py`, fit diagnostics, tests.
- Dependencies: M5/M6.
- Outputs: `peak_fit_params.csv`, hierarchical parameter tables, μ equivalence results.
- Validation: synthetic Gaussian recovery across noise levels, boundary/failure behavior, bootstrap coverage simulation, TOST unit tests.

### M8 — Participant tables and paired contrasts (Medium)

- Files: `group_tables.py`, pairing tests.
- Dependencies: M1/M6/M7.
- Outputs: `subject_level_metrics.csv`, `paired_contrasts.csv`, inclusion flow tables.
- Validation: exact joins for all dataset key conventions, duplicate-run aggregation, missing-state exclusions, no cross-session pairing.

### M9 — Temporal null battery (Large)

- Files: `nulls.py`, statistic callback exposure, tests.
- Dependencies: M4–M6.
- Outputs: `null_subject_results.csv`, `null_summary.csv`, empirical p-values.
- Validation: seeded reproducibility; phase-spectrum preservation; block composition preservation; derangement proof; AR(1) autocorrelation reduction; uniform empirical p-values under synthetic null.

### M10 — Mixed models, meta-analysis, multiplicity (Large)

- Files: `inference.py`, tests.
- Dependencies: M8/M9.
- Outputs: model coefficients/diagnostics, dataset effects, meta/heterogeneity/LOO tables, FDR family registry.
- Validation: synthetic known effects, comparison against direct formulas/reference statsmodels calls, convergence/fallback tests, no sensitivity-to-primary promotion.

### M11 — Artifact and modality sensitivities (Large)

- Files: `artifact_controls.py`, dataset configs, QC tests.
- Dependencies: M2–M5 and M8.
- Outputs: ICA/QRS interpolation, nuisance-adjusted, broadband, ECG-vs-PPG sensitivity tables.
- Validation: artifact injection tests, signal-retention plots, modality agreement diagnostics. Implement raw-EEG QRS interpolation as the cardiac-field sensitivity; do not mask 1-Hz HR samples.

### M12 — Figures, reports, and manifest (Medium)

- Files: `figures.py`, `report.py`, `manifest.py`, output contract tests.
- Dependencies: M10/M11.
- Outputs: Figures 1–3 in PDF/SVG/PNG, source data CSVs, machine-readable result bundle, methods/result tables, manifest.
- Validation: schema tests, no hard-coded results, deterministic figure-source hashes, visual review checklist.

### M13 — Full reproducible run and manuscript lock (Very Large compute/operations)

Split into operational sub-milestones so engineering (preflight/orchestration) can finish before raw cohorts are available.

#### M13a — Production preflight + synthetic smoke (done)

- Module: [`production.py`](ppg_eeg/temporal_coupling/confirmatory/production.py); CLI: `python -m ppg_eeg.temporal_coupling.confirmatory --mode {preflight,smoke,primary,sensitivity,clean_root}`.
- Modes verify configs, pairing inventory, channel/protocol declarations, raw/derivative availability; missing data = **blockers**, never participant exclusions.
- Synthetic smoke runs M1 + fixture M4 → M5–M12 with schema checks; records commit, config hashes, runtime.
- Artifacts: `production_preflight.csv`, `production_run_plan.json`, `stage_execution_log.csv`, `smoke_run_manifest.json` under `derivatives/confirmatory_temporal_coupling/production/`.
- Confirmatory unit suite last green at M13a: **141** tests (includes production E2E smoke).

#### M13b — HIIT real-smoke subset (in progress; operator-run)

- Configs (all under one folder):
  - [`smoke/hiit/confirmatory.yaml`](zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml) — participants `01–03`, sessions `ph`/`ps`, PRE/POST rest↔Tetris.
  - [`smoke/hiit/beats.yaml`](zero-lag-reanalysis-repo/smoke/hiit/beats.yaml) — Stage 0/1b photosensor PPG.
  - [`smoke/hiit/verification.json`](zero-lag-reanalysis-repo/smoke/hiit/verification.json) — PPG lock + stop rules.
- Stage driver: [`hiit_smoke_m13b.py`](ppg_eeg/temporal_coupling/confirmatory/hiit_smoke_m13b.py) — one `--stage` at a time; hard-stops on pairing/modality failures.
- Pairing: PRE-rest↔PRE-Tetris and POST-rest↔POST-Tetris within each PH/PS session; never collapse to task-only.
- Remaining: operator executes Stage 0 → 1b → M2–M4 assembly → M5–M12; confirm photosensor on PS sessions; stop if assumptions fail.

#### M13c — Primary production run (pending; blocked on raw)

- Freeze configs/manifest before inspecting primary inferential results.
- Require raw for all primary datasets: `ds003838`, `ds006848`, `ds003690`, `ds004587` (latter present; first three missing as of 2026-07-13).
- Refuse full cohort when raw incomplete; do not silently drop participants as “exclusions.”

#### M13d — Sensitivities, clean-root reproducibility, manuscript lock (pending)

- Run sensitivity datasets (`ds004582`, `ds003816`, HIIT, mindfulness) after primary QC review.
- Re-run from clean output root; compare hashes and inclusion counts; archive logs.
- Populate manuscript only from generated tables/figure-source files.

## 11. Validation strategy

- Unit tests: every pure transformation, config invariant, pair key, output schema, and deterministic seed.
- Numerical validation: synthetic HR/EEG with known lags, known Gaussian peaks, analytically computed ZLPI, null signals, and known mixed/meta effects.
- Regression: run all existing tests, especially [`test_cross_correlation.py`](tests/test_cross_correlation.py), [`test_group_summary.py`](tests/test_group_summary.py), [`test_resample.py`](tests/test_resample.py), [`test_cardiac_timeseries.py`](tests/test_cardiac_timeseries.py), and adapter tests; legacy output schemas must not change.
- Integration: one smoke subject per file format/modality, then one complete paired dataset smoke run.
- Visual QC: raw/processed ECG peaks, instant-HR interpolation, multitaper bands, selected segment, lag curve with endpoint windows, Gaussian residuals, null distributions.
- Statistical validation: simulation-based type-I error and power for ZLPI/nulls; TOST coverage; meta-analysis against hand/reference calculations; sensitivity analyses cannot alter primary decisions.
- Reproducibility: clean-root rerun, stable hashes, deterministic seeds, environment/version capture, config and source commit hashes.

## 12. Ranked risks and mitigation

### Critical

- **Gamma contamination/cardio-field artifact:** mitigate with line-frequency audit, scalp/topographic QC, raw-EEG QRS interpolation and ICA sensitivities, and broadband-residualized analysis.
- **Duration-driven selection in ds003838:** retain D240 primary, mandate nested D180, publish exclusion flow, compare retained/excluded metadata and effect direction; never replace primary using sensitivity significance.
- **ds003816 endpoint incompatibility:** isolate the short-window analysis and prohibit pooling with ZLPI.
- **Incorrect participant pairing:** centralize key normalization, assert uniqueness, emit paired-set files, and test every adapter convention.

### High

- **Autocorrelation invalidates parametric r uncertainty:** base primary significance on temporal surrogates and participant-level inference; retain innovations sensitivity.
- **Peak-fit nonidentifiability:** constrained weighted fits, fit diagnostics, prespecified failure criteria, and ZLPI as the primary endpoint so peak failure cannot invalidate all subjects.
- **Cardiac modality heterogeneity:** ECG primary where available, modality-stratified sensitivity, metadata covariate, no unlabelled pooling.
- **Compute cost of raw EEG and 1,000 nulls:** cache C1a/C1b/C1c, vectorize curves, smoke with 20 nulls, parallelize by observation, checkpoint null batches.

### Medium

- **Config drift:** immutable master defaults, strict override whitelist, config hashes.
- **Stale derivatives and documentation:** current derivative folders contain heatmaps/distribution plots removed from active Stage 3 code, and some stored group-permutation columns are NaN. Never discover confirmatory inputs by filename presence alone; regenerate every confirmatory artifact under the isolated output root and make reports consume manifest-listed files only.
- **Output volume:** long-form compressed parquet may be added only after retaining CSV source tables required by manuscript/review; manifest every format.
- **Mixed-model convergence:** center covariates, simplify random effects via a documented deterministic fallback, report diagnostics.
- **Preprocessing inconsistency across formats:** shared preprocessing function and cross-format synthetic/smoke tests.

### Low

- **Legacy breakage:** separate namespace/output root and full regression suite.
- **Figure/report divergence:** generate both from the same frozen tidy tables and hash those inputs.

## 13. Estimated effort summary

- Small: endpoint formulas and isolated schema utilities.
- Medium: protocol audit/config, instant HR, harmonization, lag adapter, group tables, figures/report/manifest.
- Large: multitaper processing, peak hierarchy, null battery, mixed/meta inference, artifact controls.
- Very Large: complete eight-dataset production computation, manual QC adjudication, reproducibility rerun, and manuscript lock.

Expected engineering effort is roughly 25–35 person-days for the mandatory core (M1–M10/M12), plus 8–15 days for the full artifact/modality suite and operational time for production runs. Raw-EEG multitaper and null generation dominate compute; derivative-only milestones should run in hours.

## 14. Final recommended implementation order

1. ~~Commit output contracts, confirmatory config schemas, protocol audit, and pairing rules.~~ **Done (M1)**
2. ~~Commit instant-HR reconstruction and duration validation.~~ **Done (M2)**
3. ~~Commit multitaper theta/alpha/beta/gamma features and spectral QC.~~ **Done (M3)**
4. ~~Commit common-support alignment and deterministic nested D240/D180/D120 segments.~~ **Done (M4)**
5. ~~Commit signed HR-only 1-s lag curves while proving legacy Stage 2 unchanged.~~ **Done (M5)**
6. ~~Commit Fisher-z, ZLPI, local prominence, and endpoint QC.~~ **Done (M6)**
7. ~~Commit Gaussian peak fitting and two-stage hierarchical parameter inference.~~ **Done (M7)**
8. ~~Commit participant tables, paired contrasts, and inclusion-flow outputs.~~ **Done (M8)**
9. ~~Commit temporal null battery with deterministic parallel execution.~~ **Done (M9)**
10. ~~Commit mixed-effects, equivalence, meta-analysis, LOO, and multiplicity registry.~~ **Done (M10)**
11. ~~Commit artifact, nuisance, duration, and modality sensitivities.~~ **Done (M11)**
12. ~~Commit Figures 1–3, source-data exports, report generator, and manifests.~~ **Done (M12)**
13. ~~Production preflight + synthetic smoke orchestration.~~ **Done (M13a)**
14. **Next:** Operator-run HIIT real smoke (M13b) — Stage 0/1b PPG photosensor → M2–M12; stop on pairing/modality/D240–D180 failures.
15. Freeze primary configs; run primary cohorts when raw complete (M13c).
16. Sensitivities, clean-root reproducibility, lock hashes, manuscript tables (M13d).

This order keeps each commit independently testable, delays expensive raw-data computation until schemas are fixed, and prevents exploratory outputs or sensitivity results from contaminating the primary confirmatory analysis.