---
name: Post-run QC report
overview: Design-reviewed read-only QC reporting layer. Architecture frozen after review—separate grains, configurable heuristics, no analysis coupling. Implementation deferred until explicit approval.
todos:
  - id: qc-module
    content: "Implement qc_report.py: read C0–C7 QC tables, join, write QC/ outputs"
    status: completed
  - id: qc-flags
    content: Implement review flags, dispositions, visual_review_list, QC_REPORT.md
    status: completed
  - id: qc-cli
    content: Wire explicit --mode qc-report (not a STAGE_ORDER stage) + README
    status: completed
  - id: qc-tests
    content: Add immutability + flag unit tests in tests/test_qc_report.py
    status: completed
isProject: false
---

# Read-only confirmatory QC reporting layer

**Status:** Design review complete. Implementation not started. Architecture below incorporates review fixes.

---

## Design review (2026-07-15)

### 1. Separation of responsibilities

**Verdict:** Intent is sound; two risks to eliminate before coding.

| Risk in original proposal | Why it could leak into analysis | Safer alternative |
|---------------------------|----------------------------------|-------------------|
| Wire as `--stage QC` inside confirmatory stage machinery | Future `all` / stage graphs / status files might treat QC as a pipeline stage; operators may think QC gates runs | Use **`--mode qc-report`** only (like preflight). Do **not** add `QC` to `STAGE_ORDER`. Do not write `stage_status.json` for QC. |
| `generate_qc_report(ctx)` sharing StageContext write helpers | Accidental reuse of stage writers | Public API: `generate_qc_report(dataset_output_root: Path, *, qc_params: QcReportParams \| None) -> Path` reading only; mkdir `QC/` exclusively |
| Scanning `C7/figures/source_data` to set “entered figures” | Temptation to regenerate figures or rewrite manifests | Read-only membership check; column named `listed_in_figure_source_tables` (not “validated for manuscript”) |
| Hash/immutability tests only | Production still needs hard isolation | QC never opens C0–C7 files for write; open with read-only intent; write only under `…/QC/` |

**Confirmed requirements (frozen):**

- Never modify C0–C7 outputs.
- Never recompute peaks, HR, correlations, endpoints, nulls, inference, or figures.
- Never change eligibility, pairing, preprocessing, statistics, or figure generation.
- Write **only** under `{dataset_output_root}/QC/`.

---

### 2. Scientific facts vs heuristic review rules

| Flag / rule | Classification | Source | Notes |
|-------------|----------------|--------|-------|
| C0 `usable` / `skip_reason` | **Pipeline fact** | `data_audit.csv` | Echo as-is |
| C1b `usable`, warnings, n_clean_ibis, coverage, polarity | **Pipeline fact** | `cardiac_peak_qc.csv` | Echo as-is |
| Detector `quality_score` selected / runner-up / gap | **Derived exact** | `peak_detector_comparison.csv` | Gap = selected − max(other); not a new analysis |
| `pct_clean_ibi` | **Pipeline fact or exact derive** | comparison `percent_clean_ibi` or `n_clean_ibis/(n_raw_peaks-1)` | Prefer stored `percent_clean_ibi`; else label `derived` |
| `n_rejected_ibi_est` | **Weak estimate** | — | **Remove** from v1 (ambiguous vs rejected peaks) |
| IHR status, gap counts, max_beat_gap | **Pipeline fact** | `instant_hr_qc.csv` | Echo |
| HR min/median/max/percentiles from series | **Derived exact** | `features_instant_hr.csv` where `is_valid_hr` | Label columns `derived_from_ihr_series` in schema notes |
| `n_hr_outside_40_180` | **Heuristic count** using pipeline’s existing [40,180] usable band | derived + threshold | Threshold = **same as C1b usable HR band** → document as `aligned_to_c1b_usable_hr_band`; still configurable |
| `n_hr_abs_diff_gt_20` | **New heuristic** | derived | Make **configurable** (`hr_abs_diff_bpm`); default 20 |
| `pct_gap_masked > 5` | **New heuristic** | derived | Configurable (`gap_masked_pct`) |
| Polarity gap `< 50` or `< 5%` | **New heuristic** | derived | Configurable (`polarity_score_gap`, `polarity_score_rel_gap`) |
| EEG `pct_channels_rejected > 25` | **New heuristic** | derived from multitaper QC | Configurable |
| Multitaper `status`, `spectral_qc_passed` | **Pipeline fact** | multitaper_qc | Echo |
| C1c/C0/C2/C3 eligibility & exclusions | **Pipeline fact** | eligibility / alignment / corr / endpoint QC | Echo |
| “Primary D240 absolute_log10” rollups | **Convention**, not HIIT-specific | master contracts | Use `EXPECTED_PRIMARY_DURATION_S` + primary representation from contracts; band-agnostic worst-case across expected bands |
| `review_ppg` “when PPG expected” | **HIIT-leaning** | — | Generalize: flag cardiac channel/type issues from `signal_type` / C0 cardiac_exists; rename domain `review_cardiac` (alias `review_ppg` deprecated) |

**Disposition vocabulary (frozen wording):**

| Label | Meaning |
|-------|---------|
| `pipeline_fact` | Copied from stage QC |
| `derived_metric` | Exact transform of existing files |
| `heuristic_review_suggestion` | Thresholded review aid (configurable) |
| `analytical_exclusion` | Copied from pipeline eligibility (`eligible=False`, etc.) |
| `not_assessed` | Inputs absent for this domain |

Never use: validated, artifact-free, proven correct, biologically valid.

---

### 3. Observation vs contrast vs model grain (revised table set)

**Problem in original design:** stuffing C5 contrast eligibility, C6 model convergence, and “entered inference” into `observation_qc_master` forces ambiguous many-to-many joins (one obs → many bands/reps/contrasts; C6 is dataset/model level).

**Recommended tables (clean separation):**

| File | Grain | Contents |
|------|-------|----------|
| `observation_qc.csv` | observation | Cardiac, peaks, polarity gap, IHR, EEG multitaper, review flags for those domains, stage coverage |
| `duration_qc.csv` | observation × duration | C0 + C1c (+ optional C2/C3 primary-representation summary columns per duration, not all band rows) |
| `coupling_unit_qc.csv` | observation × duration × band × power_representation | C2 exclusion, overlap, C3 eligible, peak-fit — **unit of coupling analysis** |
| `contrast_qc.csv` | dataset × contrast × participant × session × duration × band × representation | C5 `contrast_eligible`, deltas presence; link `low_observation_ids` / `effort_observation_ids` |
| `null_unit_qc.csv` | same keys as C4 null_qc | status, n_surrogates — do not collapse into observation |
| `model_qc.csv` | dataset / component (from `inference_qc.csv`) | MixedLM/meta status — **not** observation-level |
| `dataset_qc_summary.csv` | dataset | Counts, flag tallies (replaces separate cardiac/eeg summary or keep thin domain summaries as views) |
| `visual_review_list.csv` | mixed keys + `grain` column | Prioritized human review queue |
| `QC_REPORT.md` | dataset | Narrative |
| `qc_params.json` | run | Frozen thresholds + code version for reproducibility |

Drop wide “C6 on every observation row.” Instead: observation report links “see `model_qc.csv`” when C6 present.

Keep a thin `observation_qc_master.csv` **only as a convenience join** of observation + worst-case duration D240 analytical exclusion + any review flags — or omit master and document joins in README. **Recommendation:** omit bloated master; ship the grain tables above + `dataset_qc_summary.csv`.

---

### 4. Wording — do not overstate

| Avoid | Prefer |
|-------|--------|
| QC passed | `no_automated_concern_detected` |
| PPG validated | `cardiac_pipeline_metrics_present; human review of waveform not performed` |
| Peak detection correct | `peak_qc_metrics_recorded` / `flagged_for_review` |
| Polarity correct | `polarity_selected_by_quality_score; confidence_gap=…` |
| EEG artifact-free | `multitaper_qc_status=…; channel_rejection_rate=…` |
| Coupling valid | `endpoint_eligible=True (pipeline gate)` vs `flagged_for_review` |

`QC_REPORT.md` must open with: this layer does not validate scientific truth; absence of flags is not evidence of correctness.

---

### 5. Metric audit — keep / derive / remove

| Proposed column | Verdict |
|-----------------|---------|
| Detector, polarity, usable, warnings, coverage, n_clean_ibis, median HR (peaks) | **Keep** — direct |
| quality_score selected / runner-up / gap | **Keep** — exact derive |
| pct_clean_ibi | **Keep** if from comparison file; else derived + label |
| n_rejected_ibi_est | **Remove** |
| IHR status, valid/gap counts, max_beat_gap | **Keep** — direct |
| HR percentiles / outside-band / abs-diff counts | **Keep** as **derived**; document formula |
| C0/C1c duration fields | **Keep** — direct |
| EEG channel counts, rejected list, status | **Keep** — direct |
| “finite-sample coverage” as vague % | Prefer existing `n_nonfinite_power_rows` / `n_windows` — **don’t invent** coverage % without definition |
| C2/C3 fields at coupling-unit grain | **Keep** — direct |
| C4/C5/C6 on observation master | **Move** to null/contrast/model tables |
| entered_final_inference on observation | **Replace** with contrast-level `contrast_eligible` + model_qc presence |
| listed_in_figure_source_tables | **Keep** optional, observation or contrast id string match — label as **inventory**, not endorsement |

---

### 6. Generality across datasets

| Fragile assumption | Fix |
|--------------------|-----|
| PPG-specific `review_ppg` | `review_cardiac` (works for ECG/PPG/`signal_type`) |
| Rest/Tetris naming | Never required; keys are `observation_id` / conditions from tables |
| Eyes-open / seated | Out of scope for QC report (protocol metadata separate) |
| Multi-session | Duration/contrast tables work with 1+ sessions |
| “Primary theta” only in duration matrix | Use contract primary duration + primary representation; for band use **worst-case across EXPECTED_BANDS** or emit coupling_unit rows for all bands |
| HIIT photosensor | Read `channel_used` / `signal_type` from QC, no hard-code |
| ds004582 no contrasts / ds003816 no contrasts | `contrast_qc.csv` empty; `not_assessed` for contrast domain; model_qc may still exist |
| Sensitivity datasets excluded from primary meta | model_qc will show status; don’t flag as “failed coupling” solely for meta exclusion |

---

### 7. Future extensibility

**Add before implementation:**

```text
QC/
  qc_manifest.json          # schema_version, inputs hashed (read-only), params
  observation_qc.csv
  duration_qc.csv
  coupling_unit_qc.csv
  contrast_qc.csv
  null_unit_qc.csv
  model_qc.csv
  dataset_qc_summary.csv
  visual_review_list.csv
  QC_REPORT.md
  extensions/               # reserved; empty in v1
```

- `schema_version: confirmatory_qc_report_v1`
- Each row may include `metric_source` where non-obvious (`pipeline` | `derived` | `heuristic`)
- Future modules (morphology, respiration, manual annotations) write `extensions/<module>.csv` and register in manifest without altering v1 columns
- `QcReportParams` dataclass / `qc_params.json` for all heuristic thresholds

---

### 8. Performance

| Dataset scale | Approx cost |
|---------------|-------------|
| HIIT full ~160 obs | Peak QC + multitaper group reads: seconds. **Bottleneck:** per-obs `features_instant_hr.csv` (~300 rows) × 160 for percentiles ≈ fine (&lt;1–2 s). Detector comparison × 160 ≈ fine. C2/C3 QC ~2k–8k rows: fine. |
| Multi-dataset future | Same per dataset root |

**Recommendations:**

- Prefer **group** CSVs (`C1b/cardiac_peak_qc.csv`, `C2/*_qc_D*.csv`) over walking every obs dir when group file exists.
- Lazy-load IHR series **only** for observations needed for HR derived metrics / visual review extremes (or all obs once — still small).
- No caching layer required for v1.
- Avoid loading full correlation **curve** tables; QC CSVs only.

---

### 9. Final recommendation

**Strengths**

- Clear write isolation under `QC/`.
- Read-only aggregation matches the scientific need after C0.
- Immutability tests are the right safety net.
- Review flags as non-binding is correct given `usable` does not gate C1c.

**Weaknesses of the pre-review proposal**

- Over-wide observation master mixing grains.
- Several new heuristic thresholds presented as if pipeline facts.
- `--stage QC` risks pipeline entanglement.
- Ambiguous / estimated IBI rejection column.
- HIIT-centric cardiac naming.
- Risk of overstating “entered figures/inference.”

**Suggested improvements (adopted for freeze)**

1. `--mode qc-report`, not a stage in `STAGE_ORDER`.
2. Grain-separated tables (section 3).
3. Configurable `QcReportParams` for all heuristics; echo pipeline thresholds when aligned (e.g. HR 40–180).
4. Remove `n_rejected_ibi_est`.
5. Rename cardiac review domain; band-agnostic primary-slice rules.
6. Explicit metric_source / disposition vocabulary; disclaimer in report.
7. Extensibility via `qc_manifest.json` + `extensions/`.

**Unnecessary complexity to drop**

- Convenience “master” with C6/C5 collapsed fields.
- Dual polarity gap rules (keep one absolute gap + one relative gap, both configurable).
- Scanning figure PNGs (CSV source_data membership only, optional).

**Ready to implement?** **Yes, after freezing this revised architecture** — not the original wide-master draft.

---

## Frozen architecture (post-review)

### CLI

```bash
.venv/bin/python -m ppg_eeg.confirmatory \
  --mode qc-report \
  --config zero-lag-reanalysis-repo/datasets/hiit.yaml
# resolves dataset output_root and writes output_root/QC/
```

Optional: `--qc-root` override. Not invoked by `--stage all`.

### Default `QcReportParams` (configurable; written to `QC/qc_params.json`)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `hr_plausible_min_bpm` / `max` | 40 / 180 | Align with C1b `usable` HR gate |
| `hr_abs_diff_bpm` | 20 | Abrupt change review aid |
| `hr_outside_band_frac` | 0.01 | Fraction of valid samples outside band |
| `gap_masked_pct` | 5.0 | Review if heavy gap masking |
| `polarity_score_gap` | 50 | Absolute quality_score separation |
| `polarity_score_rel_gap` | 0.05 | Relative gap vs selected score |
| `eeg_channel_reject_pct` | 25.0 | High channel loss review |
| `visual_review_n_extremes` | 10 | List size |
| `visual_review_n_random` | 10 | Seed fixed in params |

### Module layout

- [`ppg_eeg/confirmatory/qc_report.py`](ppg_eeg/confirmatory/qc_report.py) — orchestration
- Optional split later: `qc_report/load.py`, `flags.py`, `write.py` (single file OK for v1 if &lt;~800 LOC)

### Tests

1. SHA-256 immutability of all `C0`–`C7` files.
2. Only `QC/` paths created/modified.
3. Grain tests: coupling_unit rows match C3 keys; contrast rows match C5; model_qc row count matches inference_qc.
4. Missing C4–C6 → `not_assessed`, no crash.
5. Flag unit tests with synthetic rows for each heuristic.

### Implementation order (after explicit “implement” approval)

1. Params + loaders + grain writers  
2. Flags + visual list + markdown + manifest  
3. `--mode qc-report` wiring  
4. Tests + README  

**Do not implement until the user explicitly approves this post-review freeze.**
