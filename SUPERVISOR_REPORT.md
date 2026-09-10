# Supervisor report: Confirmatory EEG–cardiac temporal coupling

**Project:** Zero-lag confirmatory reanalysis of EEG–heart-rate coupling  
**Code:** `ppg_eeg/confirmatory/`  
**Protocol:** `zero-lag-reanalysis-repo/master.yaml`  
**Results:** `derivatives/confirmatory_temporal_coupling/`  
**Date:** 2026-09-10  

## Abstract

This report covers the locked confirmatory EEG–cardiac coupling analysis: a full C0–C7 pipeline testing whether instantaneous heart rate and EEG band power show near-zero-lag coupling (ZLPI), attenuation under cognitive demand, and peak timing near lag 0, across three primary paired datasets (n = 138). The primary claims were **not confirmed**—pooled low-demand alpha ZLPI was +0.0195 (95% CI [−0.009, 0.048], p = 0.180), state attenuation was null across bands (alpha Δ = −0.0079, p = 0.671; 0/56 FDR rejects), and peak-center equivalence (±2 s) was not established. The main positive lead is secondary: alpha coupling exceeded several temporal nulls after FDR. Manuscript Figures 1–3 and exact numbers are below.

---

## 1. Executive summary (for a quick read)

We built and ran a **prespecified confirmatory pipeline** asking whether instantaneous heart rate and EEG band power show:

1. a **near-zero-lag** coupling peak (ZLPI), especially in alpha at low cognitive demand;  
2. **attenuation** of that coupling under high cognitive demand;  
3. peak timing consistent with **μ ≈ 0 ± 2 s**;  
4. robustness to **temporal nulls** and artifact/nuisance controls.

**Primary sample (paired meta):** n = **70 + 47 + 21 = 138** participants across `ds003690`, `ds003838`, `ds006848`.

### Bottom line

| Prespecified claim | Result |
|--------------------|--------|
| Low-demand alpha ZLPI ≠ 0 (pooled) | **Not established** — pooled +0.0195, 95% CI [−0.009, 0.048], p = 0.180 |
| State attenuation of ZLPI (high − low) | **Not established** — alpha pooled Δ = −0.0079, p = 0.671; all bands NS |
| Peak center μ within ±2 s (TOST) | **Not established** (`equivalent = False` on primary low-demand alpha) |
| Theta ZLPI > circular-shift null | **Not established** — Δ = +0.0069, p = 0.272 |
| Primary BH-FDR family | **0 / 56** rejects |
| Secondary: alpha > several temporal nulls (FDR) | **Supported (secondary only)** |

**One-sentence takeaway for supervision:** the confirmatory analysis is complete and transparent; the locked primary physiological claims were **not confirmed**. The main positive lead is **secondary** evidence that **alpha** coupling exceeds several temporal shuffle nulls.

---

## 2. What has been done

### 2.1 Scientific goal

Test whether EEG band-power envelopes and instantaneous heart rate show a reproducible **lag-resolved near-zero coupling** signature that:

- is detectable at rest / low demand,  
- **weakens under cognitive effort**,  
- is centered near lag 0,  
- and is unlikely under autocorrelation-preserving nulls.

### 2.2 Engineering / analysis deliverables completed

| Deliverable | Status |
|-------------|--------|
| Locked protocol (bands, lags, ZLPI/MWPI/SWPI contracts, seeds, dataset roles) | Done (`zero-lag-reanalysis-repo/`) |
| Full C0–C7 pipeline (eligibility → features → lags → endpoints → nulls → inference → figures) | Done (`ppg_eeg/confirmatory/`) |
| Primary cohorts run end-to-end | `ds003690`, `ds003838`, `ds006848` |
| Sensitivity cohorts run end-to-end | `hiit`, `mindfulness`, `ds004582`, `ds004587`, `ds003816` (D60 only by design) |
| Merged manuscript tree + Figures 1–3 | `derivatives/confirmatory_temporal_coupling/primary/merged/` |
| Source-data CSVs beside figures | `…/C7/figures/source_data/` |
| Tests / architecture audits for roles and contracts | Present under `tests/` |

### 2.3 Pipeline stages (C0–C7)

| Stage | Purpose |
|-------|---------|
| C0 | Data/protocol audit, pairing, duration eligibility |
| C1a | Multitaper EEG band power |
| C1b | Cardiac peaks → instantaneous HR |
| C1c | Align HR + EEG on nested windows (60/120/180/240 s) |
| C2 | Lagged correlations (primary grid ±60 s @ 1 s) |
| C3 | Endpoints (ZLPI) + near-zero Gaussian peak fits |
| C4 | Surrogate nulls (500 production surrogates) |
| C5 | Subject-level metrics + paired high−low contrasts |
| C6 | Mixed models, RE meta, TOST, BH-FDR, robustness panels |
| C7 | Publish tables + manuscript figures |

### 2.4 Locked primary design

- **Primary duration / endpoint:** 240 s **ZLPI** on `absolute_log10` power  
- **ZLPI definition:** \(z(0) − \overline{z}(|\tau|\in[20,60]\text{ s})\) (Fisher-z of Pearson *r*)  
- **Primary meta contrasts (exactly one per dataset):**
  - `ds003838`: `rest__memory`
  - `ds006848`: `rest__verbalwm`
  - `ds003690`: `passive__gonogo`
- **Sensitivity (not pooled into primary meta):** HIIT, mindfulness, `ds004582`, `ds004587`, `ds003816`

---

## 3. Primary results (with figures)

All figures below are the manuscript renders from:

`derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/`

### 3.1 Figure 1 — Structure, alpha replication, peak timing

![Figure 1. Lag-resolved zero-lag structure and alpha replication](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure1_lag_resolved_zero_lag.png)

**Caption (abbrev.):** low-demand lag curves; lag-category means; dataset×band ZLPI heatmap; alpha replication forest; peak μ / FWHM. Low-demand alpha point estimates are positive; pooled CI includes zero. μ equivalence is not established.

#### Low-demand alpha ZLPI (Fig 1E)

| Dataset | n | Mean ZLPI | 95% CI | p |
|---------|---|-----------|--------|---|
| ds003690 | 70 | +0.0208 | [−0.0198, 0.0614] | 0.310 |
| ds003838 | 47 | +0.0165 | [−0.0309, 0.0639] | 0.488 |
| ds006848 | 21 | +0.0245 | [−0.0692, 0.1181] | 0.592 |
| **Pooled RE (Paule–Mandel)** | 3 datasets | **+0.0195** | **[−0.0090, 0.0481]** | **0.180** |

**Interpretation:** consistently positive direction; **not statistically confirmed** after pooling.

#### Peak-center TOST (±2 s), low-demand alpha (Fig 1F)

| Dataset | Identifiable n | Mean μ (s) | TOST p | Equivalent? |
|---------|----------------|------------|--------|-------------|
| ds003690 | 23 | −0.21 | 0.186 | No |
| ds003838 | 23 | −3.70 | 0.814 | No |
| ds006848 | 11 | −0.95 | 0.386 | No |

**Interpretation:** near-zero peak centering **not established**.

---

### 3.2 Figure 2 — State attenuation

![Figure 2. State-dependent attenuation](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure2_state_attenuation_replication.png)

**Caption (abbrev.):** matched low vs high lag curves; lag-difference curves; alpha attenuation forest; model marginals; state peaks; graded ds003690 display.

#### Paired ΔZLPI (high − low), entering primary meta — alpha

| Dataset | Contrast | n pairs | ΔZLPI | 95% CI | p |
|---------|----------|---------|-------|--------|---|
| ds003690 | passive__gonogo | 70 | −0.0130 | [−0.0627, 0.0367] | 0.603 |
| ds003838 | rest__memory | 47 | −0.0037 | [−0.0695, 0.0620] | 0.909 |
| ds006848 | rest__verbalwm | 21 | +0.0039 | [−0.1036, 0.1114] | 0.940 |

#### Pooled RE meta — all bands (D240, absolute_log10)

| Band | Pooled Δ | 95% CI | p | I² |
|------|----------|--------|---|-----|
| alpha | −0.0079 | [−0.0441, 0.0284] | 0.671 | 0% |
| beta | +0.0034 | [−0.0334, 0.0401] | 0.858 | 0% |
| low_gamma | −0.0129 | [−0.0473, 0.0215] | 0.463 | 0% |
| theta | −0.0025 | [−0.0312, 0.0261] | 0.861 | 0% |

**Primary multiplicity:** **0 / 56** BH-FDR rejections.

**Interpretation:** confirmatory **state attenuation is not established**.

---

### 3.3 Figure 3 — Temporal nulls and robustness

![Figure 3. Temporal specificity and robustness](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure3_temporal_artifact_specificity.png)

**Caption (abbrev.):** theta vs nulls; pairing controls; duration sensitivity; cardiac controls. Official caption is cautious: null exceedance not established for the primary theta claim; pairing inconclusive; many cardiac controls not computable; gamma artifact-indeterminate.

#### Prespecified theta vs circular-shift null

| Quantity | Value |
|----------|-------|
| Mean Δ (observed − null) | +0.0069 |
| 95% CI | [−0.0055, 0.0193] |
| p (two-sided) | 0.272 |
| Analysis units / biological participants / condition estimates | 245 / 170 / 533 |
| Surrogates | 500 |
| Recorded interpretation | directionally positive but inconclusive |

#### Secondary FDR family — alpha null exceedances that reject

| Band × null | Mean Δ | p | q | Reject FDR? |
|-------------|--------|---|---|-------------|
| alpha × circular_shift | 0.0178 | 0.0127 | 0.045 | **Yes** |
| alpha × phase_randomization | 0.0193 | 0.0063 | 0.031 | **Yes** |
| alpha × cross_subject_mismatch | 0.0191 | 0.0057 | 0.031 | **Yes** |
| alpha × ar1_innovations | 0.0132 | 0.0067 | 0.031 | **Yes** |

**Interpretation:**  

- Primary theta-null claim: **not established**.  
- Secondary alpha-null exceedances: **positive lead**, but not a substitute for failed primary ZLPI / attenuation / μ claims.

#### Additional Figure 3 sheets (optional viewing)

![Figure 3E. Nuisance / modality](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure3_panel_e_nuisance_modality.png)

![Figure 3F. Topography / low-γ](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure3_panel_f_topography_gamma.png)

![Figure 3 supplement. Null diagnostics](derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure3_supplement_null_diagnostics.png)

---

## 4. Eligibility snapshot (primary D240)

| Dataset | Eligible obs / participants @ 240 s | Notes |
|---------|--------------------------------------|-------|
| ds003690 | 375 / 75 | All durations eligible |
| ds003838 | 120 / 65 | +10 ineligible (`insufficient_raw_duration`) |
| ds006848 | 44 / 22 | +8 ineligible (`missing_paired_state`) |

Paired analysis n for meta (after pairing/QC): **70 / 47 / 21**.

---

## 5. Sensitivity cohorts (outside primary meta)

Full C0–C7 trees exist under `derivatives/confirmatory_temporal_coupling/sensitivity/`.

| Dataset | Role | Brief result note |
|---------|------|-------------------|
| hiit | Exercise-state moderation (PPG) | Primary-flagged Rest–Tetris ΔZLPI contrasts not significant |
| mindfulness | Internal-attention steps (PPG) | `step1__step3` alpha Δ = −0.0467, **nominal p = 0.027**, n = 29 (not meta) |
| ds004582 / ds004587 | External single-state ECG generalization | Completed; empty primary meta by design |
| ds003816 | Duration sensitivity | D60 SWPI descriptive only; longer windows excluded by manuscript design |

These cohorts support generalization / robustness discussion; they **do not** enter the primary pooled diamond.

---

## 6. Interpretation for supervision

### What we can say defensibly

1. The confirmatory pipeline is **implemented, locked, and executed** on primary + sensitivity datasets with manuscript figures and machine-readable source data.  
2. Across three primary paired datasets, **prespecified near-zero-lag and state-attenuation claims were not confirmed**.  
3. There is a **secondary** signal that **alpha** coupling is temporally structured relative to several nulls.  
4. Peak centering near zero was **not** established by TOST.

### What we should not claim

- A confirmed cross-study zero-lag physiological coupling signature.  
- Confirmed cognitive-load attenuation of ZLPI.  
- That secondary alpha-null results rescue the primary hypotheses.

### Practical implication

This is a successful **negative / inconclusive confirmatory** package: the question was asked cleanly and answered. The most promising follow-up is not re-tweaking the same primary meta, but (a) understanding estimand mismatch vs exploratory peak-\|r\| attenuation, and (b) carefully characterizing the alpha-null signal under cardiac/modality controls (see `FUTURE_DIRECTIONS.md`).

---

## 7. Where everything lives (for review)

| Item | Path |
|------|------|
| Figure 1 PNG/PDF | `derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/figure1_lag_resolved_zero_lag.*` |
| Figure 2 PNG/PDF | `…/figure2_state_attenuation_replication.*` |
| Figure 3 PNG/PDF | `…/figure3_temporal_artifact_specificity.*` |
| Inference tables | `…/primary/merged/C6/` (`meta_analysis_results.csv`, `low_demand_alpha_replication_*.csv`, `multiplicity_results.csv`, …) |
| Figure source data | `…/C7/figures/source_data/` |
| Per-dataset primary runs | `…/primary/{ds003690,ds003838,ds006848}/` |
| Sensitivity runs | `…/sensitivity/` |
| Protocol | `zero-lag-reanalysis-repo/master.yaml` |
| Plain-language story | `RESEARCH_STORY.md` |
| Technical findings + code checklist | `RESEARCH_FINDINGS.md` |
| Suggested next analyses | `FUTURE_DIRECTIONS.md` |

### How to view this report with figures

Open `SUPERVISOR_REPORT.md` in the repo (Cursor / VS Code / GitHub) so the relative image links resolve. PDFs of the same figures are alongside the PNGs if a print-friendly attachment is needed.

---

## 8. Suggested talking points for the meeting

1. **Pipeline is done** — C0–C7 + merged Figs 1–3.  
2. **Primary confirmatory claims failed/inconclusive** — numbers in §3.  
3. **Secondary alpha-null lead** exists and is quantified.  
4. **Next step options:** alpha-null deep-dive; peak-\|r\| vs ZLPI estimand comparison; ECG/PPG + cardiac-control feasibility (not re-fishing the same primary meta).

---

*Numbers in this report were taken from `primary/merged/C6` and `C7/figures/source_data` (production confirmatory outputs).*
