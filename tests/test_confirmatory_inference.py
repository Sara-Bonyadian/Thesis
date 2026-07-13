from __future__ import annotations

import math
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ppg_eeg.confirmatory.duration_contracts import (
    ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
    ENDPOINT_ZLPI,
)
from ppg_eeg.confirmatory.inference import (
    DATASET_EFFECTS_FILENAME,
    FAMILY_META_BAND,
    FAMILY_PRIMARY_STATE_ATTENUATION,
    FDR_ALPHA,
    INFERENCE_QC_FILENAME,
    LEAVE_ONE_DATASET_OUT_FILENAME,
    META_ANALYSIS_RESULTS_FILENAME,
    META_EXCLUDED_DATASETS,
    MIXED_MODEL_RESULTS_FILENAME,
    MULTIPLICITY_RESULTS_FILENAME,
    PEAK_CENTER_EQUIVALENCE_FILENAME,
    bh_fdr,
    estimate_dataset_effects,
    fit_mixed_model,
    leave_one_dataset_out,
    prediction_interval,
    random_effects_meta,
    run_confirmatory_inference,
    run_meta_analysis,
    tost_peak_center_equivalence,
    write_inference_outputs,
)


def _subject_row(
    *,
    dataset_id: str,
    participant_id: str,
    condition: str,
    state: str,
    band: str,
    endpoint_index: float,
    mean_hr: float = 70.0,
    modality: str = "ecg",
    duration_s: int = 240,
    endpoint_name: str = ENDPOINT_ZLPI,
    local_prominence: float = 0.1,
    peak_center_mu_s: float = 0.0,
    has_identifiable_peak: bool = True,
    power_representation: str = "absolute_log10",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "participant_id": participant_id,
        "session_id": "single",
        "condition": condition,
        "state": state,
        "modality": modality,
        "mean_hr": mean_hr,
        "band": band,
        "duration_s": duration_s,
        "endpoint_name": endpoint_name,
        "power_representation": power_representation,
        "endpoint_eligible": True,
        "endpoint_index": endpoint_index,
        "local_prominence": local_prominence,
        "peak_center_mu_s": peak_center_mu_s,
        "has_identifiable_peak": has_identifiable_peak,
        "is_standard_zlpi": endpoint_name == ENDPOINT_ZLPI and duration_s in {240, 180},
    }


def _paired_row(
    *,
    dataset_id: str,
    contrast_id: str,
    participant_id: str,
    band: str,
    delta: float,
    duration_s: int = 240,
    endpoint_name: str = ENDPOINT_ZLPI,
    power_representation: str = "absolute_log10",
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "contrast_id": contrast_id,
        "participant_id": participant_id,
        "session_id": "single",
        "duration_s": duration_s,
        "endpoint_name": endpoint_name,
        "is_standard_zlpi": endpoint_name == ENDPOINT_ZLPI and duration_s in {240, 180},
        "band": band,
        "power_representation": power_representation,
        "delta_endpoint_index": delta,
    }


def _synthetic_state_effect_subjects(
    *,
    n_per_cell: int = 8,
    effect: float = -0.35,
    noise: float = 0.05,
    seed: int = 0,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    bands = ("delta", "theta", "alpha", "beta")
    datasets = (
        ("ds003838", "rest", "memory", "rest__memory"),
        ("ds006848", "rest", "verbalwm", "rest__verbalwm"),
        ("ds003690", "passive", "gonogo", "passive__gonogo"),
        ("ds004587", "rest", "ig", "rest__ig"),
    )
    rows: list[dict[str, object]] = []
    for dataset_id, low, effort, _contrast in datasets:
        for i in range(n_per_cell):
            pid = f"p{i:02d}"
            for band in bands:
                base = 0.40 + 0.02 * bands.index(band)
                rows.append(
                    _subject_row(
                        dataset_id=dataset_id,
                        participant_id=pid,
                        condition=low,
                        state="low_demand",
                        band=band,
                        endpoint_index=base + float(rng.normal(0.0, noise)),
                        mean_hr=65 + float(rng.normal(0, 2)),
                        peak_center_mu_s=float(rng.normal(0.1, 0.3)),
                    )
                )
                rows.append(
                    _subject_row(
                        dataset_id=dataset_id,
                        participant_id=pid,
                        condition=effort,
                        state="cognitive_effort",
                        band=band,
                        endpoint_index=base
                        + effect
                        + float(rng.normal(0.0, noise)),
                        mean_hr=72 + float(rng.normal(0, 2)),
                        peak_center_mu_s=float(rng.normal(0.2, 0.3)),
                    )
                )
    return rows


def _synthetic_paired_from_subjects(
    subject_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    contrast_map = {
        ("ds003838", "memory"): ("rest", "rest__memory"),
        ("ds006848", "verbalwm"): ("rest", "rest__verbalwm"),
        ("ds003690", "gonogo"): ("passive", "passive__gonogo"),
        ("ds003690", "simplert"): ("passive", "passive__simplert"),
        ("ds004587", "ig"): ("rest", "rest__ig"),
    }
    by_key: dict[tuple[str, ...], float] = {}
    for row in subject_rows:
        key = (
            row["dataset_id"],
            row["participant_id"],
            row["band"],
            row["condition"],
            row["duration_s"],
            row["endpoint_name"],
            row["power_representation"],
        )
        by_key[key] = float(row["endpoint_index"])

    paired: list[dict[str, object]] = []
    for (dataset_id, effort), (low, contrast_id) in contrast_map.items():
        participants = sorted(
            {
                r["participant_id"]
                for r in subject_rows
                if r["dataset_id"] == dataset_id
            }
        )
        bands = sorted(
            {r["band"] for r in subject_rows if r["dataset_id"] == dataset_id}
        )
        for pid in participants:
            for band in bands:
                low_key = (
                    dataset_id,
                    pid,
                    band,
                    low,
                    240,
                    ENDPOINT_ZLPI,
                    "absolute_log10",
                )
                effort_key = (
                    dataset_id,
                    pid,
                    band,
                    effort,
                    240,
                    ENDPOINT_ZLPI,
                    "absolute_log10",
                )
                if low_key not in by_key or effort_key not in by_key:
                    continue
                paired.append(
                    _paired_row(
                        dataset_id=dataset_id,
                        contrast_id=contrast_id,
                        participant_id=pid,
                        band=band,
                        delta=by_key[effort_key] - by_key[low_key],
                    )
                )
    return paired


class TestFDRAndPredictionInterval(unittest.TestCase):
    def test_bh_fdr_orders_and_controls(self) -> None:
        p_values = [0.001, 0.01, 0.04, 0.20]
        q_values = bh_fdr(p_values, alpha=FDR_ALPHA)
        self.assertEqual(len(q_values), 4)
        self.assertLessEqual(q_values[0], q_values[1])
        self.assertTrue(q_values[0] <= FDR_ALPHA)
        self.assertTrue(q_values[-1] > FDR_ALPHA)

    def test_prediction_interval_wider_than_ci_when_tau2_positive(self) -> None:
        pooled = 0.2
        var = 0.01
        ci_half = 1.959963984540054 * math.sqrt(var)
        pred_low, pred_high = prediction_interval(pooled, var, tau2=0.05)
        self.assertLess(pred_low, pooled - ci_half)
        self.assertGreater(pred_high, pooled + ci_half)


class TestKnownAndNullEffects(unittest.TestCase):
    def test_meta_recovers_known_negative_attenuation(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.35, noise=0.04)
        paired = _synthetic_paired_from_subjects(subjects)
        effects = estimate_dataset_effects(paired)
        primary = [e for e in effects if e["is_primary_analysis"] and e["band"] == "theta"]
        self.assertGreaterEqual(len(primary), 4)
        for row in primary:
            self.assertLess(float(row["effect_mean"]), -0.2)
        metas = run_meta_analysis(effects)
        theta = next(m for m in metas if m["band"] == "theta")
        self.assertTrue(theta["is_primary_analysis"])
        self.assertLess(float(theta["pooled_effect"]), -0.2)
        self.assertLess(float(theta["ci_high"]), 0.0)
        self.assertTrue(math.isfinite(float(theta["prediction_low"])))
        self.assertTrue(math.isfinite(float(theta["q"])))
        self.assertTrue(math.isfinite(float(theta["tau2"])))
        self.assertGreaterEqual(float(theta["i2"]), 0.0)

    def test_null_effects_center_near_zero(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=0.0, noise=0.03, seed=3)
        paired = _synthetic_paired_from_subjects(subjects)
        effects = estimate_dataset_effects(paired)
        metas = run_meta_analysis(effects)
        for meta in metas:
            self.assertLess(abs(float(meta["pooled_effect"])), 0.08)


class TestHeterogeneity(unittest.TestCase):
    def test_heterogeneous_effects_increase_tau2_and_i2(self) -> None:
        homogeneous = random_effects_meta(
            [0.20, 0.21, 0.19, 0.205],
            [0.002, 0.002, 0.002, 0.002],
            labels=["a", "b", "c", "d"],
        )
        heterogeneous = random_effects_meta(
            [-0.4, -0.05, 0.25, 0.55],
            [0.01, 0.01, 0.01, 0.01],
            labels=["a", "b", "c", "d"],
        )
        self.assertGreater(float(heterogeneous["tau2"]), float(homogeneous["tau2"]))
        self.assertGreater(float(heterogeneous["i2"]), float(homogeneous["i2"]))
        self.assertGreater(float(heterogeneous["q"]), float(homogeneous["q"]))


class TestEquivalence(unittest.TestCase):
    def test_tost_detects_equivalence_near_zero(self) -> None:
        rows = []
        rng = np.random.default_rng(1)
        for i in range(40):
            rows.append(
                _subject_row(
                    dataset_id="ds003838",
                    participant_id=f"p{i}",
                    condition="rest",
                    state="low_demand",
                    band="theta",
                    endpoint_index=0.3,
                    peak_center_mu_s=float(rng.normal(0.0, 0.25)),
                )
            )
        eq = tost_peak_center_equivalence(rows)
        theta = next(r for r in eq if r["band"] == "theta")
        self.assertTrue(theta["equivalent"])
        self.assertLess(float(theta["tost_p"]), FDR_ALPHA)

    def test_tost_rejects_far_from_zero(self) -> None:
        rows = [
            _subject_row(
                dataset_id="ds003838",
                participant_id=f"p{i}",
                condition="rest",
                state="low_demand",
                band="theta",
                endpoint_index=0.3,
                peak_center_mu_s=4.0 + 0.01 * i,
            )
            for i in range(20)
        ]
        eq = tost_peak_center_equivalence(rows)
        theta = next(r for r in eq if r["band"] == "theta")
        self.assertFalse(theta["equivalent"])


class TestLeaveOneOut(unittest.TestCase):
    def test_loo_stable_under_homogeneous_effects(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.30, noise=0.02, seed=2)
        paired = _synthetic_paired_from_subjects(subjects)
        effects = estimate_dataset_effects(paired)
        metas = run_meta_analysis(effects)
        loo = leave_one_dataset_out(effects, metas)
        theta_loo = [r for r in loo if r["band"] == "theta"]
        self.assertGreaterEqual(len(theta_loo), 3)
        full = next(m for m in metas if m["band"] == "theta")
        for row in theta_loo:
            self.assertLess(abs(float(row["delta_vs_full"])), 0.08)
            self.assertLess(
                abs(float(row["pooled_effect"]) - float(full["pooled_effect"])),
                0.08,
            )


class TestMultiplicity(unittest.TestCase):
    def test_fdr_applied_within_prespecified_families(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.40, noise=0.03)
        # Add second ds003690 contrast for attenuation family coverage.
        extra = []
        for row in subjects:
            if row["dataset_id"] == "ds003690" and row["condition"] == "gonogo":
                clone = dict(row)
                clone["condition"] = "simplert"
                clone["endpoint_index"] = float(row["endpoint_index"]) + 0.02
                extra.append(clone)
        subjects = subjects + extra
        paired = _synthetic_paired_from_subjects(subjects)
        result = run_confirmatory_inference(
            subjects,
            paired,
            fit_endpoints=(ENDPOINT_ZLPI,),
        )
        families = {r["family_id"] for r in result.multiplicity_rows}
        self.assertIn(FAMILY_PRIMARY_STATE_ATTENUATION, families)
        self.assertIn(FAMILY_META_BAND, families)
        atten = [
            r
            for r in result.multiplicity_rows
            if r["family_id"] == FAMILY_PRIMARY_STATE_ATTENUATION
        ]
        self.assertTrue(any(r["reject_fdr"] for r in atten))
        # q-values are monotone within family after sorting by test_id, but must be finite.
        self.assertTrue(any(math.isfinite(float(r["q_value"])) for r in atten))


class TestMixedModelFallback(unittest.TestCase):
    def test_mixed_model_recovers_state_effect(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.35, noise=0.05, seed=5)
        coef_rows, qc = fit_mixed_model(subjects, endpoint_name=ENDPOINT_ZLPI)
        self.assertTrue(coef_rows)
        self.assertIn(qc["status"], {"ok", "fallback_ols"})
        state_terms = [
            r for r in coef_rows if "state" in str(r["term"]).casefold()
        ]
        self.assertTrue(state_terms)

    def test_convergence_failure_falls_back_deterministically(self) -> None:
        subjects = _synthetic_state_effect_subjects(n_per_cell=4, effect=-0.2, seed=9)

        class _Boom:
            def fit(self, *args, **kwargs):  # noqa: ANN002, ANN003
                raise RuntimeError("forced_convergence_failure")

        with patch(
            "ppg_eeg.confirmatory.inference.smf.mixedlm",
            side_effect=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("forced_mixedlm_failure")
            ),
        ):
            coef_rows, qc = fit_mixed_model(subjects, endpoint_name=ENDPOINT_ZLPI)
        self.assertEqual(qc["model_backend"], "ols_cluster_participant")
        self.assertEqual(qc["status"], "fallback_ols")
        self.assertTrue(coef_rows)
        # Second call identical → deterministic fallback path.
        with patch(
            "ppg_eeg.confirmatory.inference.smf.mixedlm",
            side_effect=RuntimeError("forced_mixedlm_failure"),
        ):
            coef_rows2, qc2 = fit_mixed_model(subjects, endpoint_name=ENDPOINT_ZLPI)
        self.assertEqual(qc2["model_backend"], qc["model_backend"])
        self.assertEqual(
            [r["term"] for r in coef_rows2],
            [r["term"] for r in coef_rows],
        )


class TestEndpointSeparation(unittest.TestCase):
    def test_mwpi_swpi_never_enter_primary_meta_or_rescue_zlpi(self) -> None:
        subjects = _synthetic_state_effect_subjects(effect=-0.3, seed=4)
        paired = _synthetic_paired_from_subjects(subjects)
        # Add strong MWPI attenuation that must not enter primary meta.
        for i in range(8):
            paired.append(
                _paired_row(
                    dataset_id="ds003838",
                    contrast_id="rest__memory",
                    participant_id=f"p{i:02d}",
                    band="theta",
                    delta=-0.9,
                    duration_s=120,
                    endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                )
            )
            paired.append(
                _paired_row(
                    dataset_id="ds003838",
                    contrast_id="rest__memory",
                    participant_id=f"p{i:02d}",
                    band="theta",
                    delta=-0.9,
                    duration_s=60,
                    endpoint_name=ENDPOINT_SHORT_WINDOW_PROXIMAL_INDEX,
                )
            )
            subjects.append(
                _subject_row(
                    dataset_id="ds003838",
                    participant_id=f"p{i:02d}",
                    condition="rest",
                    state="low_demand",
                    band="theta",
                    endpoint_index=0.5,
                    duration_s=120,
                    endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                )
            )
            subjects.append(
                _subject_row(
                    dataset_id="ds003838",
                    participant_id=f"p{i:02d}",
                    condition="memory",
                    state="cognitive_effort",
                    band="theta",
                    endpoint_index=-0.4,
                    duration_s=120,
                    endpoint_name=ENDPOINT_MID_WINDOW_PROXIMAL_INDEX,
                )
            )

        result = run_confirmatory_inference(subjects, paired)
        self.assertTrue(all(m["endpoint_name"] == ENDPOINT_ZLPI for m in result.meta_rows))
        self.assertTrue(
            all(m["is_primary_analysis"] for m in result.meta_rows)
        )
        mwpi_effects = [
            e
            for e in result.dataset_effect_rows
            if e["endpoint_name"] == ENDPOINT_MID_WINDOW_PROXIMAL_INDEX
        ]
        self.assertTrue(mwpi_effects)
        self.assertTrue(all(not e["enters_meta"] for e in mwpi_effects))
        self.assertTrue(
            all(not e["enters_meta"] for e in result.dataset_effect_rows if e["duration_s"] in {120, 60})
        )
        # Excluded unpaired datasets stay out.
        self.assertTrue(
            all(
                e["dataset_id"] not in META_EXCLUDED_DATASETS or not e["enters_meta"]
                for e in result.dataset_effect_rows
            )
        )


class TestWriteOutputs(unittest.TestCase):
    def test_writes_all_required_inference_files(self) -> None:
        subjects = _synthetic_state_effect_subjects(n_per_cell=5, effect=-0.25, seed=6)
        paired = _synthetic_paired_from_subjects(subjects)
        result = run_confirmatory_inference(
            subjects, paired, fit_endpoints=(ENDPOINT_ZLPI,)
        )
        with TemporaryDirectory() as tmp:
            paths = write_inference_outputs(result, tmp)
            expected = {
                MIXED_MODEL_RESULTS_FILENAME,
                DATASET_EFFECTS_FILENAME,
                META_ANALYSIS_RESULTS_FILENAME,
                LEAVE_ONE_DATASET_OUT_FILENAME,
                PEAK_CENTER_EQUIVALENCE_FILENAME,
                MULTIPLICITY_RESULTS_FILENAME,
                INFERENCE_QC_FILENAME,
            }
            self.assertEqual({p.name for p in paths.values()}, expected)
            for path in paths.values():
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
