# Suggestions to explore next

Ideas worth chasing **now that the data, exploratory pipeline, and confirmatory C0–C7 stack already exist**. These are suggestions, not locked protocol. Anything that would become a new confirmatory claim should be **preregistered separately** (the current primary ZLPI tests already used their one shot).

Related docs: [`RESEARCH_STORY.md`](RESEARCH_STORY.md), [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).

---

## How to read this list

| Priority tag | Meaning |
|--------------|---------|
| **P0** | High value, mostly reusing existing tables / small new analysis |
| **P1** | Interesting science; needs some new code or careful design |
| **P2** | Bigger / riskier / longer-horizon |

Start with P0 before building new machinery.

---

## 1. Turn the “almost” findings into a focused follow-up thesis thread

### 1.1 Alpha beat several nulls — but primary ZLPI did not (P0)

**Why interesting:** Your clearest confirmatory-positive signal was **secondary**: alpha ZLPI exceeded circular-shift / phase / cross-subject / AR(1) nulls after FDR, while pooled low-demand alpha ZLPI still included zero and state attenuation failed.

**What to check:**

- Is alpha’s null exceedance driven by a few subjects, one dataset, or one condition role (low-demand only vs all states)?
- Does alpha local prominence (shoulders) tell a cleaner story than flank-based ZLPI?
- Re-plot Fig3-style Δ(observed−null) stratified by dataset × state × band using existing C4 tables.

**Why it matters for the thesis:** Frames a precise next claim: *“alpha coupling is temporally structured (beats nulls) even if mean ZLPI is small and state attenuation is weak.”*

### 1.2 Why exploratory rest-vs-task attenuation did not survive confirmatory ZLPI (P0)

**Why interesting:** Exploratory peak \|r\| often looked weaker under active tasks; confirmatory paired ΔZLPI did not.

**What to check (mostly re-analysis):**

- Same subjects, same windows: compare exploratory peak \|r\| Δ vs confirmatory ΔZLPI.
- Ask whether attenuation lives in **overall correlation magnitude** but not in **lag-0 prominence vs flanks**.
- Check if attenuation appears only for HRV (RMSSD/SDNN) pairs in exploratory data, while confirmatory uses instantaneous HR only.

**Thesis angle:** Method mismatch can be a finding: “task weakens raw coupling strength more than zero-lag prominence.”

---

## 2. Physiology-shaped hypotheses your pipeline is almost ready for

### 2.1 Timescale match: baroreflex / RSA windows vs ±60 s ZLPI (P1)

**Why interesting:** Cardiac–brain effects often live on ~2–15 s (vascular / autonomic) or respiratory (~0.1 Hz) scales, not necessarily a ±60 s flank contrast.

**What to check:**

- Restrict lag interest to a physiology window (e.g. \|τ\| ≤ 10–15 s) and redefine a *proximal* index without pretending it is ZLPI.
- Compare HR vs RMSSD/SDNN coupling to alpha (exploratory already had HRV pairs; confirmatory is HR-centric).
- Use existing D120/D60 MWPI/SWPI outputs as “short-timescale” companions, but keep names separate from ZLPI.

**Caution:** New endpoint = new estimand. Do not retrofit primary claims.

### 2.2 Alpha topography: occipital/parietal vs frontal (P1)

**Why interesting:** Alpha is spatially structured; a robust-median whole-scalp channel may dilute a focal effect. Fig3 topography/gamma panels already flag artifact ambiguity for low-γ.

**What to check:**

- Where channel-level multitaper exists, recompute alpha ZLPI for posterior vs frontal ROIs.
- Ask whether posterior alpha–HR coupling is larger and more null-resistant than global median.
- Keep ECG-prone channels explicit (you already have cardiac-field channel lists).

### 2.3 Sensor modality as a first-class factor: ECG vs PPG (P0/P1)

**Why interesting:** Primary meta is ECG-heavy; HIIT/mindfulness are PPG. Exploratory report already noted sensor type matters.

**What to check:**

- Meta-regression / stratified forest: ECG-only vs PPG-only coupling magnitude and null Δ.
- Within ds003838 (ECG + PPG available in places): same-state ECG vs PPG ZLPI agreement.
- Ask whether PPG noise explains weak HIIT confirmatory contrasts more than “exercise biology.”

---

## 3. Design / sample questions that may explain null primary results

### 3.1 Power and heterogeneity audit (P0)

**Why interesting:** Pooled effects were tiny (alpha Δ ≈ −0.008; low-demand alpha ≈ +0.020) with n_meta = 138. That may be true null, underpowered small effect, or estimand mismatch.

**What to check:**

- Report smallest detectable effect given observed SEs (you already have SEs in C6 CSVs).
- Leave-one-dataset-out and prediction intervals (already partly in outputs) as a “what would replicate?” section.
- Simulate: how large would ΔZLPI need to be for 80% power under your actual variances?

**Thesis value:** A clean “we were powered for X, observed Y” paragraph strengthens a null/inconclusive paper.

### 3.2 Duration eligibility as a scientific variable, not only QC (P0)

**Why interesting:** D240 drops people (`insufficient_raw_duration`, missing pairs). Survivors may be a biased subset.

**What to check:**

- Compare D180 ZLPI (more people) vs D240 on the same estimand family.
- Does attenuation or alpha null exceedance appear at D180 but not D240 (or the reverse)?
- For ds003816, treat D60 SWPI as a dedicated short-rest meditation/compassion analysis rather than a failed long-window study.

### 3.3 Graded demand in ds003690 (P0)

**Why interesting:** You already have `passive__simplert` and `passive__gonogo`. Fig2 Panel F is display-oriented.

**What to check:**

- Is there a monotonic passive → simplert → gonogo pattern in ZLPI or peak \|r\|?
- Even if gonogo meta fails, a graded slope within one well-powered dataset can be a thesis chapter.

---

## 4. Individual differences and “who shows coupling?”

### 4.1 Trait / state cardiac predictors (P1)

**Why interesting:** Mean ZLPI near zero can hide a subset of strong couplers.

**What to check (using existing subject-level tables):**

- Correlate subject ZLPI with mean HR, RMSSD, SDNN, usable duration, motion/QC flags.
- Mixture / clustering: “high coupler” vs “non-coupler” and whether high couplers drive null exceedances.
- HIIT: does post-exercise HR elevation moderate Rest–Tetris ΔZLPI more than PRE?

### 4.2 Within-person reliability (P1)

**Why interesting:** If ZLPI is unreliable within person, meta attenuation tests are doomed.

**What to check:**

- Split-half or odd/even window reliability of ZLPI where recordings are long (ds003838 memory, ds006848 verbalwm, ds004582).
- HIIT PH vs PS as nested reliability (you already nest PH/PS carefully in places).

---

## 5. Causality-adjacent analyses (keep language careful)

### 5.1 Directionality without overclaiming (P1)

**Why interesting:** Exploratory Stage 4 said HR→EEG curves were more reliable than EEG→HR.

**What to check:**

- Formalize HR-event → alpha/theta response curves on primary low-demand states only.
- Compare lag-signed summaries to ZLPI (orthogonal: magnitude-at-zero vs event-locked direction).
- Use cross-subject mismatch nulls you already compute as a pairing control for event averages.

### 5.2 Finish the cardiac-artifact story (P1)

**Why interesting:** Fig3 cardiac controls were often **not computable**; low-γ remained artifact-indeterminate.

**What to check:**

- Inventory exactly which controls are NC per dataset and whether raw ingredients exist to make them computable (template subtraction, ECG-prone channel exclusion).
- Prioritize **one** high-feasibility control (e.g. beat-count / mean-HR residualization, or ECG-channel exclusion for alpha) and run it cleanly end-to-end.
- Ask: does alpha’s null exceedance survive the feasible cardiac controls?

If alpha dies under a simple cardiac control, that redirects the thesis toward artifact characterization — still publishable and important.

---

## 6. Productive “negative results” papers / chapters

These are valuable precisely because primary confirmatory claims failed.

| Chapter idea | Core message |
|--------------|--------------|
| **A. Cross-dataset descriptive atlas** | Coupling exists widely; timing does not unify (exploratory report → polished atlas). |
| **B. Locked ZLPI confirmatory null/inconclusive** | Prespecified near-zero + attenuation not established; transparent pipeline. |
| **C. Alpha temporal structure** | Secondary null exceedances as the positive companion finding. |
| **D. Estimand sensitivity** | Peak \|r\| vs ZLPI vs MWPI/SWPI: when “attenuation” appears or disappears. |
| **E. Modality & QC limits** | ECG vs PPG; what cardiac controls cannot yet claim. |

---

## 7. Concrete next experiments (smallest first)

Do these in order if you want maximum thesis progress per week:

1. **P0 — Alpha null deep-dive** from existing C4/`figure3_secondary_nulls_fdr.csv`: stratify by dataset and state; write one page on what survives.
2. **P0 — Estimand mismatch**: exploratory peak \|r\| Δ vs confirmatory ΔZLPI on overlapping primary pairs.
3. **P0 — Power statement**: detectable effect sizes from C6 SEs; one table for the thesis methods/results.
4. **P0 — D180 vs D240** same ZLPI estimand on primary cohorts (duration sensitivity already partly exported).
5. **P1 — Posterior alpha ROI** pilot on one primary ECG dataset with good channel coverage.
6. **P1 — One feasible cardiac control** that is currently NC, made computable, applied to alpha.
7. **P1 — Preregister a new confirmatory family** only after (1)–(3) clarify the best estimand (do not silently reuse the old primary claims).

---

## 8. What I would *not* prioritize right now

- Re-running the **same** primary ZLPI attenuation meta with tiny analysis tweaks hoping for significance.
- Adding many new bands/endpoints into one FDR soup without a new preregistration.
- Heavy deep-learning coupling models before the simple estimand mismatch (peak \|r\| vs ZLPI) is understood.
- Treating mindfulness’s nominal p ≈ 0.027 as confirmatory without a dedicated locked plan.

---

## Bottom line

You already have the hard parts: multi-dataset data access, exploratory atlas, and a locked confirmatory machine.

The highest-yield next science is probably:

1. explain **why descriptive attenuation ≠ ZLPI attenuation**,  
2. characterize **alpha’s null exceedance** carefully, and  
3. close the **cardiac-artifact / modality** gaps that currently block strong physiological claims.

That path turns an “inconclusive confirmatory” project into a sharp thesis: *what heart–brain coupling is (and is not), under which estimands, sensors, and controls.*
