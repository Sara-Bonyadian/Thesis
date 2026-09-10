# What this research did (simple → detailed)

This is a plain-language companion to [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md).

- **Part 1** — super simple: what we asked, the main steps, what we found.
- **Part 2** — same steps, with more detail on goals, methods, and results.

---

# Part 1 — High level (super simple)

## The big question

We recorded **brain activity (EEG)** and **heart signals (ECG or PPG)** from the same people, in several different studies.

We asked:

> When someone’s heart rate changes, does their brain rhythm change at the **same time** (near zero lag)?  
> And does that heart–brain link get **weaker** when the mind is under hard cognitive load?

In everyday terms: is there a real, repeatable “heart and brain moving together” signature — and does it shrink when people are working hard?

## What we were looking for

1. **Coupling exists** — brain band power (especially alpha) and heart rate move together more than by chance.
2. **Near zero lag** — the strongest link is roughly simultaneous (not many seconds apart).
3. **State attenuation** — the link is stronger at rest / low demand than during hard tasks.
4. **Not a fluke** — the pattern survives careful checks (null shuffles, multiple datasets, locked rules).

## The main steps (roadmap)

```text
1. Collect & organize many EEG + heart datasets
2. Exploratory look: do heart and brain timelines line up at all?
3. Lock a confirmatory recipe (rules we refuse to change later)
4. Run that recipe on primary studies (rest vs hard task)
5. Stress-test with nulls and sensitivity datasets
6. Decide what we can honestly claim
```

## What we achieved (short answer)

| Goal | Simple outcome |
|------|----------------|
| See heart–brain co-fluctuation | **Yes (descriptive).** Across many datasets/tasks, some coupling showed up. |
| Find one shared timing signature across studies | **No.** Timing was messy; lags did not clearly replicate. |
| Confirm near-zero-lag coupling (locked test) | **Not confirmed.** Alpha looked slightly positive at rest, but not strong enough after pooling. |
| Confirm “harder task → weaker coupling” | **Not confirmed** in the locked confirmatory test. |
| Show coupling beats random temporal nulls | **Partly.** Alpha beat several shuffle tests (secondary evidence). Theta’s main null test did not. |
| Build a reusable, auditable pipeline | **Yes.** Full exploratory + confirmatory code, configs, figures, and eligibility audits. |

**One-sentence takeaway:**  
Heart and brain signals often **wiggle together** in a descriptive sense, but we did **not** lock in a clean, confirmatory “always near zero lag, and always weaker under cognitive load” signature across the primary studies.

---

# Part 2 — Same steps, in more detail

Each section below expands one step from Part 1.

---

## Step 1 — Collect & organize many EEG + heart datasets

### Goal

Have enough independent studies to ask whether a pattern is **study-specific** or **cross-study**.

### What was done

- Brought in cohorts with simultaneous EEG and cardiac signals (ECG or PPG/photosensor).
- Built **dataset adapters** so each study can be read the same way (`ppg_eeg/datasets/`).
- Split scientific roles later into:
  - **Primary** (enter the main pooled confirmatory tests): `ds003690`, `ds003838`, `ds006848`
  - **Sensitivity / generalization**: HIIT, mindfulness, `ds004582`, `ds004587`, `ds003816`

### Why it matters

Without multiple datasets and clear roles, a “finding” in one lab could be an accident of task length, sensor type, or preprocessing.

### Result of this step

A shared data map and config system (`core-eeg-ppg/`, `exploratory-temporal-coupling/`, `zero-lag-reanalysis-repo/`) so the same scientific question can be asked on many cohorts without rewriting the pipeline each time.

---

## Step 2 — Exploratory look: do heart and brain timelines line up?

### Goal

Before locking strict tests, **describe** what coupling looks like: how strong, at what lag, under rest vs task.

### What was done

Ran the **exploratory temporal coupling** track (Stages 0–4 in `ppg_eeg/temporal_coupling/`):

- Extract heart-rate / HRV and EEG band envelopes over time for each person.
- Slide the timelines past each other (**lagged cross-correlation**).
- Summarize peak coupling strength and lag direction across subjects.
- Optionally look at **event-triggered** curves (e.g. around HR jumps).

This produced a cross-dataset inventory (report under `temporal_coupling/`, ~8 datasets / 22 dataset–task partitions).

### What we were looking for here

- Are peak correlations mostly near zero, or all over the place?
- Is coupling stronger at rest than during hard tasks?
- Does the same lag direction appear in every study?

### Results (exploratory)

- **Coupling presence:** non-zero peak coupling showed up in all completed Stage 3 partitions (rough magnitude span ~0.04–0.48).
- **Task pattern:** within a cohort, active tasks often looked **weaker** than rest (descriptive).
- **Timing:** lag direction was **mixed** and did **not** cleanly replicate (only a couple of partitions had FDR-significant lag-direction findings).
- **Caveats:** short windows (e.g. parts of `ds003816`) and ECG vs PPG differences still matter.

### Honest conclusion from Step 2

“Something is there descriptively, especially magnitude and rest-vs-task trends — but **timing is not a clear shared signature**.”  
That inconclusive timing story is why Step 3 locked a sharper confirmatory question.

---

## Step 3 — Lock a confirmatory recipe

### Goal

Stop moving the goalposts. Decide **in advance** what counts as success.

### What was locked (`zero-lag-reanalysis-repo/master.yaml` + confirmatory code)

| Locked piece | Choice |
|--------------|--------|
| Main window | **240 seconds** of usable overlap |
| Main endpoint | **ZLPI** (Zero-Lag Prominence Index): how much correlation at lag 0 sticks out above distant flanks |
| Main power form | `absolute_log10` EEG band power |
| Confirmatory band emphasis | especially **alpha** for the replication forest |
| Main contrasts | rest/low-demand vs hard task: `rest__memory`, `rest__verbalwm`, `passive__gonogo` |
| Timing check | peak center μ within **±2 seconds** of zero (TOST) |
| Null battery | 500 surrogates; circular shift, phase scramble, block shuffle, cross-subject mismatch, AR(1) innovations |
| Role rule | sensitivity datasets can be shown, but **do not** enter the primary pooled meta |

Shorter windows (180 / 120 / 60 s) exist as **sensitivity only** and cannot “rescue” a failed primary ZLPI claim.

### Why it matters

Exploratory analysis is allowed to wander. Confirmatory analysis is allowed to be boring — and must say “not established” when the locked test fails.

### Result of this step

A frozen protocol and a stage pipeline **C0→C7** (`ppg_eeg/confirmatory/`) that audits eligibility, extracts features, correlates, scores ZLPI, runs nulls, builds group tables, runs inference, and publishes figures.

---

## Step 4 — Run the recipe on primary studies

### Goal

Test the locked claims on the three primary paired datasets (n after pairing for meta ≈ **70 + 47 + 21 = 138** people).

### Pipeline in plain language

| Stage | In human words |
|-------|----------------|
| C0 | Can we use this recording for 60/120/180/240 s? Who has both rest and task? |
| C1a | Measure EEG band power over time (multitaper). |
| C1b | Find heartbeats → instantaneous heart rate. |
| C1c | Align brain and heart onto the same clocks/windows. |
| C2 | Compute correlation at many lags (−60…+60 s for ZLPI). |
| C3 | Turn each curve into ZLPI (and fit a near-zero peak shape). |
| C5 | Build per-person scores and paired rest−task differences. |
| C6 | Pool studies (meta), models, FDR, equivalence tests. |
| C7 | Make manuscript figures + captions + source-data CSVs. |

(Null stage **C4** runs from aligned series and feeds the robustness story in Step 5.)

### Locked questions asked in Step 4

1. Is low-demand **alpha ZLPI** clearly above zero when primary studies are pooled?
2. Is high-demand ZLPI clearly **lower** than low-demand ZLPI (state attenuation)?
3. Is the coupling peak centered near **0 ± 2 s**?

### Results (primary confirmatory)

**Low-demand alpha ZLPI (replication forest)**  
- Each primary dataset’s mean was slightly **positive**.  
- Pooled estimate ≈ **+0.020**, 95% CI roughly **[−0.009, +0.048]**, p ≈ **0.18**.  
- **Verdict:** direction looks right; **not confirmed** (CI includes zero).

**State attenuation (hard − easy)**  
- Pooled alpha Δ ≈ **−0.008**, p ≈ **0.67** (other bands also non-significant).  
- Primary FDR family: **0 / 56** rejects.  
- **Verdict:** confirmatory attenuation **not established** (even though exploratory rest-vs-task trends had looked suggestive).

**Peak center near zero**  
- TOST equivalence within ±2 s: **not established** for the checked primary cells.

### What this step achieved

A complete, auditable primary analysis with merged manuscript outputs under:

`derivatives/confirmatory_temporal_coupling/primary/merged/`

— and a clear scientific answer on the locked estimands: **main confirmatory claims did not pass**.

---

## Step 5 — Stress-test with nulls and sensitivity datasets

### Goal

Ask: even if primary means are small, does observed coupling still beat **time-scrambled** versions of the data? And do other cohorts behave similarly?

### What was done

- **C4 nulls** on primary data (500 surrogates per null type).
- Figure 3–style summaries: theta’s **prespecified** circular-shift test; secondary family across bands/nulls with FDR.
- Sensitivity runs for HIIT, mindfulness, external ECG cohorts, and short-window `ds003816` (D60 descriptive only).

### Results

**Prespecified theta vs circular-shift null**  
- Mean advantage ≈ **+0.007**, p ≈ **0.27** → **inconclusive / not established**.

**Secondary alpha vs several nulls (FDR)**  
- Alpha exceeded circular-shift, phase-randomization, cross-subject mismatch, and AR(1) innovations after FDR.  
- **Verdict:** useful **secondary** support that alpha coupling is not pure temporal noise — **not** a replacement for the failed primary attenuation / μ / theta-null claims.

**Sensitivity sketches**  
- HIIT rest–task ZLPI contrasts: not significant at confirmatory primary flags.  
- Mindfulness `step1__step3` alpha: nominal p ≈ 0.027 (not in primary meta).  
- `ds003816`: kept as short-window descriptive sensitivity by design.

### What this step achieved

A robustness layer that prevents overclaiming from exploratory plots, and documents where evidence is **secondary** vs **prespecified primary**.

---

## Step 6 — Decide what we can honestly claim

### Goal

Separate “interesting descriptive pattern” from “confirmed scientific claim.”

### Final scoreboard (aligned with Part 1)

| Claim | Status |
|-------|--------|
| Descriptive heart–brain co-fluctuation across many datasets | **Supported** |
| One shared lag-timing signature across studies | **Not established** |
| Confirmed near-zero-lag alpha ZLPI (pooled primary) | **Not established** |
| Confirmed state attenuation of ZLPI | **Not established** |
| Confirmed peak μ ≈ 0 (±2 s) | **Not established** |
| Prespecified theta > temporal null | **Not established** |
| Secondary: alpha > several temporal nulls (FDR) | **Supported (secondary)** |
| Engineering: locked, reproducible C0–C7 + figures | **Achieved** |

### Why this is still a successful research project

“Not confirmed” is a result. The work:

1. Mapped the exploratory landscape honestly (coupling present; timing messy).
2. Turned that into a locked confirmatory protocol.
3. Ran it end-to-end on primary + sensitivity cohorts.
4. Reported what passed, what failed, and what is only secondary.

That is stronger science than finding a pretty plot and stopping.

---

## Where to go next in the repo

| If you want… | Open |
|--------------|------|
| Technical reference, numbers, code map, directory guide | [`RESEARCH_FINDINGS.md`](RESEARCH_FINDINGS.md) |
| How to install / run pipelines | [`README.md`](README.md) |
| Locked confirmatory protocol | `zero-lag-reanalysis-repo/README.md` |
| Manuscript figures | `derivatives/confirmatory_temporal_coupling/primary/merged/C7/figures/` |
| Exploratory cross-dataset report | `temporal_coupling/cross_dataset_temporal_coupling_report.html` |

---

## Tiny glossary

| Term | Plain meaning |
|------|----------------|
| **EEG band power** | How strong a brain rhythm (theta/alpha/beta/…) is over time |
| **Instantaneous HR** | Heart rate as a continuous timeline from beat times |
| **Lag** | Time shift between heart and brain timelines (0 = same moment) |
| **ZLPI** | Score for “is lag-0 coupling special compared with far-away lags?” |
| **Primary meta** | Pooling only the three prechosen paired studies/contrasts |
| **Null / surrogate** | Fake scrambled timelines used to ask “would chance look this good?” |
| **FDR** | Correction for testing many things at once |
| **TOST** | Test that a value is close enough to a target (here, lag ≈ 0) |
