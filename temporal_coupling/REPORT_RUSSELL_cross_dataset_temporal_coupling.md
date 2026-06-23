# Cross-Dataset Temporal EEG–Cardiac Coupling Report

**Prepared for:** Russell  
**Date:** 21 June 2026  
**Analyst note:** Conservative replication summary based on completed pipeline runs (Stages 0–4).

---

## Executive summary

We ran the within-subject temporal coupling pipeline on three derivative folders:

| Run | Path | Cardiac source | Partitions analyzed |
|-----|------|----------------|---------------------|
| **ds003838** | `derivatives/run_ds003838_temporal_coupling` | Separate ECG (BIDS) | rest (n=65), memory (n=64) |
| **ds006848** | `derivatives/run_ds006848_temporal_coupling` | Embedded PPG (BrainVision) | rest (n=21), verbalwm (n=28) |
| **HIIT-40 PPG** | `derivatives/run_hiit_40_ppg_temporal_coupling` | Embedded photosensor PPG | pre_rest, pre_tetris, post_rest, post_tetris (n=40 each) |

**Bottom line (conservative read):**

1. **Coupling strength is detectable descriptively** — median peak |r| is consistently positive across datasets and most partitions (typically 0.05–0.25), and selected peak-r tests (FDR across 9 pairs) are significant in nearly all partitions. These tests do **not** control for searching over lags.

2. **Permutation-controlled group evidence is absent** — circular-shift null tests (`sig_group_perm_fdr`) did **not** survive FDR in **any** dataset × partition × pair (0/72 partition-pairs checked). Under the stricter evidence layer, we cannot claim group-level coupling beyond what the null would produce after lag search.

3. **Lag direction does not replicate cleanly** — every partition carries a `mixed_peak_lag_direction` warning. Peak-lag vs zero is rarely FDR-significant. Comparing to the **reference anchor** (ds003838 rest), several partitions flip direction (especially HRV–alpha pairs in HIIT post-rest and ds006848 rest).

4. **Task and modality matter** — ds003838 **memory** shows the same broad pattern as rest but at **~3× weaker** peak |r|. HIIT **tetris** blocks are weaker than rest blocks. ds006848 (embedded PPG) is weaker and noisier than ds003838 rest (gold-standard ECG).

5. **Stage 4 event-triggered results are usable for HR→EEG** (large event counts), but EEG→HR curves remain exploratory.

**Overall replication label:** **partial / inconclusive** — positive descriptive coupling replicates weakly across datasets; timing direction and permutation-supported effects do not replicate strongly enough for causal or directional claims.

---

## Methods snapshot

- **Signals:** z-scored 1 Hz time series of EEG band envelopes (theta, alpha, beta) and cardiac features (HR, RMSSD, SDNN).
- **Stage 3:** Lagged cross-correlation (−60 to +60 s for most runs; ds003838 rest mean curves use −40 to +40 s). Group summaries use median raw peak lag and signed r per subject, then aggregate.
- **Inference layers (in order of conservatism):**
  1. Selected peak signed-r vs zero — exploratory (no lag-search correction).
  2. Permutation-controlled group peak strength (circular-shift EEG null) — **preferred** for group claims.
  3. Peak lag vs zero — exploratory; unstable in these cohorts.
- **Stage 4:** Subject-mean-then-group-mean event-triggered averages (HR increase/decrease; theta burst; alpha suppression; beta burst).
- **Convention:** Negative lag → EEG leads cardiac; positive lag → cardiac leads EEG.

Full pipeline documentation: `temporal_coupling/README.md`, `temporal_coupling/IMPLEMENTATION_PLAN.md`.

---

## Data coverage and QC

### Cohort sizes

| Dataset | Partition | Subjects in Stage 3 | Observations in audit |
|---------|-----------|--------------------|-----------------------|
| ds003838 | rest | 65 | 65 |
| ds003838 | memory | 64 | 65 |
| ds006848 | rest | 21 | 22 |
| ds006848 | verbalwm | 28 | 30 |
| HIIT-40 PPG | each of 4 partitions | 40 | 160 total (20 unique IDs × 8 sessions) |

### Cardiac modality

- **ds003838:** Dedicated ECG — highest signal quality; used as **reference anchor**.
- **ds006848 & HIIT-40 PPG:** Embedded PPG/photosensor — more artifact-prone; likely contributes to weaker peaks and higher edge-peak rates.

### Edge-peak QC flag

Peaks landing at the search-window boundary suggest unreliable lag estimates. Pooled edge-peak rates (all pairs):

| Dataset | Edge-peak rate (pooled) | Worst pairs |
|---------|-------------------------|-------------|
| ds003838 | 3–17% | sdnn__beta (17%), sdnn__alpha (16%) |
| ds006848 | 2–12% | sdnn__theta (12%) |
| HIIT-40 PPG | 4–18% | sdnn__alpha (18%), rmssd__alpha (16%) |

---

## Stage 3 results by dataset

### Reference anchor: ds003838 rest (n=65, ECG)

Strongest and cleanest results in the project so far.

| Pair | Median lag (s) | Median peak r | Strength | Direction |
|------|---------------|---------------|----------|-----------|
| hr__alpha | −5 | 0.236 | moderate | EEG leads |
| hr__beta | +10 | 0.235 | moderate | cardiac leads |
| hr__theta | 0 | 0.214 | moderate | near-zero |
| rmssd__alpha | −10 | 0.226 | moderate | EEG leads |
| rmssd__beta | −15 | 0.255 | moderate | EEG leads |
| sdnn__alpha | −7.5 | 0.236 | moderate | EEG leads |
| sdnn__beta | −10 | 0.245 | moderate | EEG leads |

HRV–EEG pairs show a consistent **EEG-leads-by ~5–20 s** pattern. HR–EEG pairs are moderate with mixed direction. Selected peak-r FDR: all 9/9 significant. Permutation FDR: 0/9. Peak-lag FDR: 0/9.

**Plots:** `derivatives/run_ds003838_temporal_coupling/ds003838/group/rest/group_cross_correlation_mean_sem_grid.png`

---

### ds003838 memory (n=64, ECG)

Same **directional sign** as rest for most pairs, but **much weaker** amplitudes (median r ≈ 0.07–0.11 vs 0.19–0.25 at rest). All pairs classified weak–moderate. Only `sdnn__beta` reaches moderate strength.

Interpretation: active memory task **attenuates** coupling magnitude without reversing the broad EEG-leads-HRV pattern. Partial replication of rest **in direction**, not in strength.

---

### ds006848 (embedded PPG)

#### Rest (n=21)

| Pair | Median lag | Median r | vs ds003838 rest |
|------|-----------|----------|------------------|
| hr__theta | −5 s | 0.133 | Same EEG-leads direction; ~62% of reference r |
| hr__alpha | −15 s | 0.268 | Same direction; similar r (but n=21) |
| rmssd__alpha | **+25 s** | 0.082 | **Direction flip** (ref: −10 s, EEG leads) |
| sdnn__alpha | **+20 s** | 0.131 | **Direction flip** (ref: −7.5 s, EEG leads) |

Small sample (21 rest subjects with peaks). Several SDNN pairs have high edge-peak rates (19–24%).

#### Verbal working memory (n=28)

Uniformly **weak** coupling (median r ≈ 0.04–0.08). EEG-leads direction for most pairs but peak-lag tests non-significant. Descriptive peak-r FDR significant for all pairs; permutation FDR none.

---

### HIIT-40 PPG (photosensor, n=40 per partition)

20 participants × 4 conditions (pre/post × rest/tetris); PS and PH groups pooled per partition.

#### Rest blocks (pre_rest, post_rest)

- **Moderate** HR–EEG coupling (median r ≈ 0.17–0.22) — comparable to, but slightly below, ds003838 rest.
- **pre_rest:** hr__alpha EEG-leads (−5 s, r=0.220) — **matches reference direction**.
- **post_rest:** hr__alpha **cardiac-leads** (+7.5 s, r=0.197) — **opposite to reference**; rmssd/sdnn alpha also flip vs reference in post_rest.
- HRV–theta pairs: EEG-leads in pre_rest; mixed in post_rest.

#### Tetris blocks (pre_tetris, post_tetris)

- Weaker coupling (median r ≈ 0.09–0.15 for HR pairs; many HRV pairs weak).
- **post_tetris hr__alpha:** only partition across all datasets with **FDR-significant peak lag** (median +17.5 s, cardiac leads, q=0.011) — but still no permutation support.
- Active gameplay reduces coupling magnitude vs rest segments within the same subjects.

**Plots (examples):**

- `derivatives/run_hiit_40_ppg_temporal_coupling/hiit/group/pre_tetris/group_cross_correlation_mean_sem_grid.png`
- `derivatives/run_hiit_40_ppg_temporal_coupling/hiit/group/post_rest/group_cross_correlation_mean_sem_grid.png`

---

## Cross-dataset replication matrix

Reference: **ds003838 rest**. Comparison focuses on median peak lag sign and coupling strength.

| Comparison | Peak strength | Lag direction | Replication label |
|------------|--------------|---------------|---------------------|
| ds003838 rest → ds003838 memory | Weaker (~3×) | Mostly same (EEG-leads HRV) | **Partial** |
| ds003838 rest → ds006848 rest | Weaker | Mixed (HRV-alpha flips) | **Inconclusive** (low n, PPG) |
| ds003838 rest → ds006848 verbalwm | Much weaker | Broadly same sign | **Partial** (weak) |
| ds003838 rest → HIIT pre_rest | Similar HR pairs | hr__alpha matches | **Partial** |
| ds003838 rest → HIIT post_rest | Similar HR pairs | HRV-alpha **flips** | **Does not replicate** (direction) |
| ds003838 rest → HIIT tetris | Attenuated | Mixed | **Inconclusive** |

### Pairs with most cross-dataset consistency

- **hr__theta:** EEG-leads (~−5 s) in ds003838 memory, ds006848 rest/verbalwm, and all HIIT partitions. Best candidate for a reproducible directional pattern (still without permutation support).
- **HRV–beta (rmssd__beta, sdnn__beta):** EEG-leads in ds003838 rest and HIIT pre_tetris; inconsistent elsewhere.

### Pairs with poorest replication

- **rmssd__alpha, sdnn__alpha:** Direction flips in HIIT post_rest and ds006848 rest vs reference. High edge-peak rates. Treat lag estimates as unreliable.

---

## Stage 4: Event-triggered averages

All partitions meet usability thresholds for group plots (≥10 total events, ≥4 subjects). HR-triggered EEG events are abundant; EEG-triggered cardiac events are sparser in short rest recordings.

| Dataset | Partition | HR events/subject | Theta burst/subject |
|---------|-----------|-------------------|---------------------|
| ds003838 | rest | ~7.5 | ~1.6 |
| ds003838 | memory | ~710 | ~135 |
| ds006848 | rest | ~58 | ~10 |
| ds006848 | verbalwm | ~479 | ~86 |
| HIIT PPG | rest blocks | ~14 | ~3 |
| HIIT PPG | tetris blocks | ~44 | ~8 |

Pipeline guidance (consistent across all `stage4_interpretation_notes.txt` files):

- **HR → EEG plots:** most reliable Stage 4 output.
- **EEG → HR plots:** exploratory only.
- Event curves are z-scored descriptive averages — not causal evidence.

Example plots per partition: `event_triggered_hr_to_eeg.png`, `event_triggered_eeg_to_hr.png` in each `group/{partition}/` folder.

---

## Statistical evidence summary

| Test | ds003838 rest | ds003838 memory | ds006848 rest | ds006848 verbalwm | HIIT (4 partitions) |
|------|--------------|-----------------|---------------|-------------------|---------------------|
| Selected peak-r FDR | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 each |
| Peak-lag FDR | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 (except post_tetris hr__alpha 1/9) |
| Permutation peak FDR | **0/9** | **0/9** | **0/9** | **0/9** | **0/9 each** |

**Conservative conclusion:** The only test that controls lag-search bias at the group level does not support coupling in any partition. Selected peak-r significance alone is insufficient for publication-grade claims.

---

## Limitations

1. **Different cardiac sensors** — ECG (ds003838) vs embedded PPG (ds006848, HIIT). Direct amplitude comparison is unfair; direction comparisons are also sensor-dependent.
2. **Small ds006848 rest n=21** — limits power for group lag tests and permutation inference.
3. **HIIT post vs pre confound** — exercise intervention may shift autonomic dynamics and coupling direction (seen in post_rest HRV-alpha flip).
4. **Edge peaks** — 10–24% of peaks at window boundary for some HRV pairs; lag estimates for those pairs should not be interpreted literally.
5. **Multiple comparisons** — 9 pairs × 8 partitions × 3 datasets; focus on patterns, not individual p-values.
6. **No causal claims** — all results are descriptive timing within subjects.

---

## Recommended next steps

1. **Lead with ds003838 rest ECG results** as the internal benchmark; treat PPG datasets as secondary replication attempts.
2. **Report permutation null results prominently** — do not rely on selected peak-r FDR alone.
3. **Split HIIT by PS vs PH** (or pre vs post) before drawing intervention conclusions — pooled n=40 may hide condition × direction interactions already visible between pre_rest and post_rest.
4. **Run ECG-based HIIT partition** (`derivatives/run_hiit_40_ecg_temporal_coupling`) when available — separates exercise effects from PPG quality effects.
5. **Automate** `cross_dataset_temporal_coupling_summary.csv` (planned in IMPLEMENTATION_PLAN) to formalize replication labels per pair.
6. **Stage 4:** prioritize HR-triggered EEG figures for Russell; defer EEG-triggered cardiac figures to supplementary material.

---

## Key file index

| Content | Location |
|---------|----------|
| ds003838 group summaries | `derivatives/run_ds003838_temporal_coupling/ds003838/group/{rest,memory}/` |
| ds006848 group summaries | `derivatives/run_ds006848_temporal_coupling/ds006848/group/{rest,verbalwm}/` |
| HIIT-40 PPG group summaries | `derivatives/run_hiit_40_ppg_temporal_coupling/hiit/group/{pre_rest,pre_tetris,post_rest,post_tetris}/` |
| Interpretation notes | `stage3_interpretation_notes.txt`, `stage4_interpretation_notes.txt` in each partition folder |
| Peak statistics | `peak_correlation_summary.csv`, `group_cross_correlation_summary.csv` |
| Mean ± SEM curves | `group_cross_correlation_mean_sem_grid.png` |
| Data audit | `derivatives/run_ds003838_temporal_coupling/ds003838/group/data_audit.csv` (130 rows) |
| | `derivatives/run_ds006848_temporal_coupling/ds006848/group/data_audit.csv` (52 rows) |

---

*Report generated from pipeline outputs in the three derivative runs listed above. Interpretation follows the conservative evidence hierarchy defined in the temporal coupling module.*
