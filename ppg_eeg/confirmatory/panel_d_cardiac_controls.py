"""Figure 3 Panel D cardiac-field control summaries.

Panel D must preserve ECG-vs-PPG semantics:
- ECG datasets: electrical R-peak interpretation.
- PPG datasets: pulse-event interpretation only (not electrical leakage tests).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S, ZLPI_FLANKS_S
from .paired_delta_inference import cluster_bootstrap_mean_ci

CONTROL_BASELINE = "baseline"
CONTROL_ICA_TEMPLATE = "cardiac_template_subtraction"
CONTROL_RPEAK_MASK = "cardiac_event_masking"
CONTROL_BEAT_COUNT = "beat_count_adjusted"
CONTROL_ECG_CHANNELS = "ecg_prone_channels_removed"

# Upstream observation-level control IDs (preferred when precomputed).
CONTROL_PPG_TEMPLATE = "ppg_pulse_locked_template_subtraction"
CONTROL_PPG_MASK = "ppg_systolic_peak_mask"
CONTROL_ECG_TEMPLATE = "ecg_cardiac_template_subtraction"
CONTROL_ECG_MASK = "ecg_r_peak_mask"

CONTROL_ORDER: tuple[str, ...] = (
    CONTROL_BASELINE,
    CONTROL_ICA_TEMPLATE,
    CONTROL_RPEAK_MASK,
    CONTROL_BEAT_COUNT,
    CONTROL_ECG_CHANNELS,
)

# Manuscript Panel D row order (baseline is a shared reference, not a row).
PANEL_D_PLOT_CONTROL_ORDER: tuple[str, ...] = (
    CONTROL_BEAT_COUNT,
    CONTROL_PPG_TEMPLATE,
    CONTROL_PPG_MASK,
    CONTROL_ECG_TEMPLATE,
    CONTROL_ECG_CHANNELS,
)

PANEL_D_SHORT_DISPLAY_LABELS: dict[str, str] = {
    CONTROL_BEAT_COUNT: "Beat-count adjusted",
    CONTROL_PPG_TEMPLATE: "PPG template subtraction",
    CONTROL_PPG_MASK: "PPG event mask",
    CONTROL_ECG_TEMPLATE: "ECG template subtraction",
    CONTROL_ECG_CHANNELS: "ECG-prone channels removed",
    CONTROL_BASELINE: "Baseline",
    CONTROL_ICA_TEMPLATE: "Cardiac template subtraction",
    CONTROL_RPEAK_MASK: "Cardiac event mask",
    CONTROL_ECG_MASK: "ECG R-peak mask",
}

PANEL_D_CONTROL_FAMILY: dict[str, str] = {
    CONTROL_BASELINE: "primary_reference",
    CONTROL_BEAT_COUNT: "statistical_sensitivity",
    CONTROL_PPG_TEMPLATE: "signal_level",
    CONTROL_PPG_MASK: "signal_level",
    CONTROL_ECG_TEMPLATE: "signal_level",
    CONTROL_ECG_CHANNELS: "signal_level",
    CONTROL_ICA_TEMPLATE: "signal_level",
    CONTROL_RPEAK_MASK: "signal_level",
    CONTROL_ECG_MASK: "signal_level",
}

# Map alternate/fallback control IDs onto the locked display rows.
PANEL_D_CONTROL_ALIASES: dict[str, str] = {
    CONTROL_ICA_TEMPLATE: CONTROL_PPG_TEMPLATE,
    CONTROL_RPEAK_MASK: CONTROL_PPG_MASK,
    CONTROL_ECG_MASK: CONTROL_PPG_MASK,
}

OBSERVATION_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "participant_id",
    "biological_participant_id",
    "session_id",
    "protocol",
    "period",
    "state",
    "band",
    "duration_s",
    "control",
    "endpoint_name",
    "endpoint_value",
    "baseline_endpoint_value",
    "paired_change",
    "beat_count",
    "mean_hr",
    "n_masked_samples",
    "masked_sample_fraction",
    "n_channels_original",
    "n_channels_removed",
    "n_channels_retained",
    "removed_channels",
    "cardiac_signal_type",
    "source_channel",
    "cardiac_event_type",
    "detector_name",
    "sampling_rate",
    "n_events",
    "alignment_status",
    "computable",
    "not_computable_reason",
    "eligibility",
    "exclusion_reason",
    "analysis_role",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "control",
    "display_label",
    "display_label_short",
    "control_family",
    "endpoint_name",
    "band",
    "state_or_contrast",
    "baseline_estimate",
    "controlled_estimate",
    "estimate",
    "ci_lower",
    "ci_upper",
    "change_from_baseline",
    "change_ci_lower",
    "change_ci_upper",
    "paired_correlation",
    "sign_retention_fraction",
    "n_sign_changes",
    "n_observations",
    "n_baseline_eligible",
    "n_biological_participants",
    "denominator_label",
    "analysis_role",
    "computability_status",
    "computability_reason",
    "short_reason_code",
)

DATASET_QC_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "participant_id",
    "observation_id",
    "cardiac_signal_type",
    "source_channel",
    "cardiac_event_type",
    "detector_name",
    "detector_parameters",
    "sampling_rate",
    "n_events",
    "expected_n_events",
    "event_rate_per_min",
    "median_inter_event_interval",
    "invalid_interval_count",
    "alignment_status",
    "computable",
    "not_computable_reason",
)


@dataclass(frozen=True)
class PanelDCardiacControlsResult:
    observations: tuple[dict[str, object], ...]
    summaries: tuple[dict[str, object], ...]
    dataset_qc: tuple[dict[str, object], ...]
    metadata: dict[str, object]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    text = _as_str(value)
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: object) -> bool:
    return _as_str(value).casefold() in {"1", "true", "yes"}


def _sign(value: float) -> int:
    if not math.isfinite(value):
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _paired_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        _as_str(row.get("dataset_id")).casefold(),
        _as_str(row.get("participant_id")).casefold(),
        _as_str(row.get("session_id"), "single").casefold(),
        _as_str(row.get("contrast_id")).casefold(),
        _as_str(row.get("band")).casefold(),
        str(_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S)),
        _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold(),
        _as_str(row.get("power_representation"), "absolute_log10").casefold(),
    )


def _subject_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return (
        _as_str(row.get("dataset_id")).casefold(),
        _as_str(row.get("participant_id") or row.get("subject_id")).casefold(),
        _as_str(row.get("session_id"), "single").casefold(),
        _as_str(row.get("condition")).casefold(),
        str(_as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S)),
        _as_str(row.get("band")).casefold(),
        _as_str(row.get("power_representation"), "absolute_log10").casefold(),
    )


def _protocol_modalities(
    protocol_rows: Sequence[Mapping[str, object]],
) -> dict[str, set[str]]:
    by_dataset: dict[str, set[str]] = {}
    for row in protocol_rows:
        dataset_id = _as_str(row.get("dataset_id")).casefold()
        if not dataset_id:
            continue
        modality = _as_str(row.get("cardiac_modality")).casefold()
        tokens: set[str] = set()
        if "ecg" in modality:
            tokens.add("ECG")
        if "ppg" in modality or "photo" in modality or "pleth" in modality:
            tokens.add("PPG")
        if not tokens:
            tokens.add("UNKNOWN")
        by_dataset[dataset_id] = tokens
    return by_dataset


def _primary_baseline_rows(
    paired_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for row in paired_rows:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI).casefold() != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), EXPECTED_PRIMARY_DURATION_S) != EXPECTED_PRIMARY_DURATION_S:
            continue
        if _as_str(row.get("power_representation"), "absolute_log10").casefold() != "absolute_log10":
            continue
        if str(row.get("contrast_eligible", "")).casefold() in {"false", "0", "no"}:
            continue
        delta = _as_float(row.get("delta_endpoint_index"))
        if not math.isfinite(delta):
            continue
        out.append(row)
    return out


def _build_dataset_qc(
    baseline_obs: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[Mapping[str, object]],
    cardiac_qc_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    modalities = _protocol_modalities(protocol_rows)
    dataset_ids = sorted({_as_str(row.get("dataset_id")).casefold() for row in baseline_obs if _as_str(row.get("dataset_id"))})
    out: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        rows = [r for r in cardiac_qc_rows if _as_str(r.get("dataset_id")).casefold() == dataset_id]
        expected_modalities = modalities.get(dataset_id, {"UNKNOWN"})
        if not rows:
            if expected_modalities == {"ECG"}:
                signal_type = "ECG"
                event_type = "r_peak"
            elif expected_modalities == {"PPG"}:
                signal_type = "PPG"
                event_type = "ppg_systolic_peak"
            else:
                signal_type = "UNKNOWN"
                event_type = "unknown"
            out.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": "",
                    "observation_id": "",
                    "cardiac_signal_type": signal_type,
                    "source_channel": "",
                    "cardiac_event_type": event_type,
                    "detector_name": "",
                    "detector_parameters": "",
                    "sampling_rate": float("nan"),
                    "n_events": float("nan"),
                    "expected_n_events": float("nan"),
                    "event_rate_per_min": float("nan"),
                    "median_inter_event_interval": float("nan"),
                    "invalid_interval_count": float("nan"),
                    "alignment_status": "not_verifiable_from_frozen_outputs",
                    "computable": False,
                    "not_computable_reason": (
                        "cardiac_peak_qc.csv unavailable in C7 publish; cannot audit detector/channel/event counts"
                    ),
                }
            )
            continue

        signal_types = {_as_str(r.get("signal_type")).casefold() for r in rows}
        if "ecg" in signal_types:
            selected_rows = [r for r in rows if _as_str(r.get("signal_type")).casefold() == "ecg"]
            signal_type = "ECG"
            event_type = "r_peak"
        elif "ppg" in signal_types:
            selected_rows = [r for r in rows if _as_str(r.get("signal_type")).casefold() == "ppg"]
            signal_type = "PPG"
            event_type = "ppg_systolic_peak"
        else:
            selected_rows = rows
            signal_type = "UNKNOWN"
            event_type = "unknown"

        channels = sorted({_as_str(r.get("channel_used")) for r in selected_rows if _as_str(r.get("channel_used"))})
        detectors = sorted({_as_str(r.get("detector_used")) for r in selected_rows if _as_str(r.get("detector_used"))})
        sfreq = np.asarray([_as_float(r.get("sfreq")) for r in selected_rows], dtype=float)
        sfreq = sfreq[np.isfinite(sfreq)]

        n_events = sum(max(0, _as_int(r.get("n_accepted_peaks"))) for r in selected_rows)
        invalid_intervals = sum(
            max(0, _as_int(r.get("n_raw_peaks")) - _as_int(r.get("n_clean_ibis")))
            for r in selected_rows
        )
        expected = 0.0
        coverage_total = 0.0
        ibi_medians: list[float] = []
        for row in selected_rows:
            coverage = _as_float(row.get("clean_ibi_coverage_s"))
            hr = _as_float(row.get("median_hr_bpm"))
            if math.isfinite(coverage) and coverage > 0:
                coverage_total += coverage
            if math.isfinite(hr) and hr > 0 and math.isfinite(coverage) and coverage > 0:
                expected += (coverage / 60.0) * hr
                ibi_medians.append(60000.0 / hr)
        event_rate = float("nan")
        if coverage_total > 0:
            event_rate = 60.0 * float(n_events) / coverage_total
        median_ibi = float(np.median(ibi_medians)) if ibi_medians else float("nan")
        not_computable_reason = ""
        computable = True
        if "ECG" in expected_modalities and signal_type != "ECG":
            computable = False
            not_computable_reason = (
                "ECG-specific control requested but only non-ECG cardiac events were detected; "
                "PPG events are not valid substitutes for ECG R-peaks"
            )
        out.append(
            {
                "dataset_id": dataset_id,
                "participant_id": "",
                "observation_id": "",
                "cardiac_signal_type": signal_type,
                "source_channel": ";".join(channels),
                "cardiac_event_type": event_type,
                "detector_name": ";".join(detectors),
                "detector_parameters": json.dumps(
                    {
                        "detector_polarities": sorted(
                            {
                                _as_str(r.get("detector_polarity"))
                                for r in selected_rows
                                if _as_str(r.get("detector_polarity"))
                            }
                        ),
                        "selection_reason_present": any(_as_str(r.get("selection_reason")) for r in selected_rows),
                    },
                    sort_keys=True,
                ),
                "sampling_rate": float(np.median(sfreq)) if sfreq.size else float("nan"),
                "n_events": float(n_events),
                "expected_n_events": float(expected) if expected > 0 else float("nan"),
                "event_rate_per_min": event_rate,
                "median_inter_event_interval": median_ibi,
                "invalid_interval_count": float(invalid_intervals),
                "alignment_status": "not_verifiable_from_frozen_outputs",
                "computable": computable,
                "not_computable_reason": not_computable_reason,
            }
        )
    return out


def _dataset_qc_lookup(dataset_qc_rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {
        _as_str(row.get("dataset_id")).casefold(): row
        for row in dataset_qc_rows
        if _as_str(row.get("dataset_id"))
    }


def _decorate_with_dataset_qc(
    row: dict[str, object],
    dataset_qc_lookup: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    dataset_id = _as_str(row.get("dataset_id")).casefold()
    qc = dataset_qc_lookup.get(dataset_id, {})
    row["cardiac_signal_type"] = _as_str(qc.get("cardiac_signal_type"), "UNKNOWN")
    row["source_channel"] = _as_str(qc.get("source_channel"))
    row["cardiac_event_type"] = _as_str(qc.get("cardiac_event_type"), "unknown")
    row["detector_name"] = _as_str(qc.get("detector_name"))
    row["sampling_rate"] = _as_float(qc.get("sampling_rate"))
    row["n_events"] = _as_float(qc.get("n_events"))
    row["alignment_status"] = _as_str(
        qc.get("alignment_status"),
        "not_verifiable_from_frozen_outputs",
    )
    row["computable"] = bool(qc.get("computable", False))
    row["not_computable_reason"] = _as_str(qc.get("not_computable_reason"))
    return row


def _build_baseline_observations(
    paired_rows: Sequence[Mapping[str, object]],
    dataset_qc_lookup: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for row in _primary_baseline_rows(paired_rows):
        delta = _as_float(row.get("delta_endpoint_index"))
        contrast_id = _as_str(row.get("contrast_id"))
        item = {
            "dataset_id": _as_str(row.get("dataset_id")),
            "participant_id": _as_str(row.get("participant_id")),
            "biological_participant_id": _as_str(row.get("participant_id")),
            "session_id": _as_str(row.get("session_id"), "single"),
            "protocol": contrast_id,
            "period": contrast_id.split("__", 1)[0] if "__" in contrast_id else "",
            "state": contrast_id,
            "band": _as_str(row.get("band")),
            "duration_s": EXPECTED_PRIMARY_DURATION_S,
            "control": CONTROL_BASELINE,
            "endpoint_name": ENDPOINT_ZLPI,
            "endpoint_value": delta,
            "baseline_endpoint_value": delta,
            "paired_change": 0.0,
            "beat_count": float("nan"),
            "mean_hr": float("nan"),
            "n_masked_samples": float("nan"),
            "masked_sample_fraction": float("nan"),
            "n_channels_original": float("nan"),
            "n_channels_removed": float("nan"),
            "n_channels_retained": float("nan"),
            "removed_channels": "",
            "eligibility": "computed",
            "exclusion_reason": "",
            "analysis_role": "primary_reference",
        }
        observations.append(_decorate_with_dataset_qc(item, dataset_qc_lookup))
    return observations


def _control_reason(control: str, signal_type: str) -> str:
    if control == CONTROL_ICA_TEMPLATE:
        if signal_type == "PPG":
            return (
                "PPG pulse-locked template subtraction is a pulse-synchronous sensitivity test "
                "and does not directly test ECG electrical-field leakage; required high-rate "
                "time-locked EEG/cardio traces are not retained in frozen tables"
            )
        return (
            "ECG R-locked template subtraction requires high-rate multichannel EEG and preserved "
            "R-peak-aligned decomposition artifacts that are not retained in frozen tables"
        )
    if control == CONTROL_RPEAK_MASK:
        if signal_type == "PPG":
            return (
                "PPG pulse-event masking requires high-rate EEG samples with pulse-event timestamps; "
                "1 Hz envelope tables cannot support event-centered masking"
            )
        return (
            "ECG R-peak masking requires high-rate EEG samples with R-peak timestamps; "
            "1 Hz envelope tables cannot support event-centered masking"
        )
    if control == CONTROL_ECG_CHANNELS:
        return (
            "channel-level pre-aggregation EEG and prespecified ECG-prone channel applicability "
            "map are not retained in frozen tables"
        )
    return "not_computable_from_retained_tables"


def _emit_not_computable(
    baseline_obs: Sequence[Mapping[str, object]],
    *,
    control: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in baseline_obs:
        item = dict(row)
        signal_type = _as_str(item.get("cardiac_signal_type"), "UNKNOWN")
        item["control"] = control
        item["endpoint_value"] = float("nan")
        item["paired_change"] = float("nan")
        item["eligibility"] = "not_computable"
        item["exclusion_reason"] = _control_reason(control, signal_type)
        item["not_computable_reason"] = item["exclusion_reason"]
        item["analysis_role"] = "cardiac_control"
        rows.append(item)
    return rows


def _beat_count_adjusted_observations(
    baseline_obs: Sequence[Mapping[str, object]],
    subject_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], str]:
    cov_lookup: dict[tuple[str, ...], dict[str, float]] = {}
    for row in subject_rows:
        key = _subject_key(row)
        beat_count = _as_float(row.get("beat_count") or row.get("n_beats"))
        mean_hr = _as_float(row.get("mean_hr") or row.get("mean_hr_bpm"))
        if math.isfinite(beat_count) and math.isfinite(mean_hr):
            cov_lookup[key] = {"beat_count": beat_count, "mean_hr": mean_hr}

    design_rows: list[tuple[int, float, float, float]] = []
    for i, row in enumerate(baseline_obs):
        contrast_id = _as_str(row.get("state"))
        if "__" not in contrast_id:
            continue
        low_condition, effort_condition = contrast_id.split("__", 1)
        common = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("participant_id")).casefold(),
            _as_str(row.get("session_id"), "single").casefold(),
            str(EXPECTED_PRIMARY_DURATION_S),
            _as_str(row.get("band")).casefold(),
            "absolute_log10",
        )
        low_key = (common[0], common[1], common[2], low_condition.casefold(), common[3], common[4], common[5])
        effort_key = (common[0], common[1], common[2], effort_condition.casefold(), common[3], common[4], common[5])
        if low_key not in cov_lookup or effort_key not in cov_lookup:
            continue
        beat = 0.5 * (cov_lookup[low_key]["beat_count"] + cov_lookup[effort_key]["beat_count"])
        mean_hr = 0.5 * (cov_lookup[low_key]["mean_hr"] + cov_lookup[effort_key]["mean_hr"])
        y = _as_float(row.get("endpoint_value"))
        if math.isfinite(y):
            design_rows.append((i, y, beat, mean_hr))

    if len(design_rows) < 4:
        return [], "beat_count and mean_hr unavailable for matched D240 pairs."

    y = np.asarray([x[1] for x in design_rows], dtype=float)
    beat = np.asarray([x[2] for x in design_rows], dtype=float)
    mean_hr = np.asarray([x[3] for x in design_rows], dtype=float)
    beat_c = beat - float(np.mean(beat))
    mean_hr_c = mean_hr - float(np.mean(mean_hr))
    design = np.column_stack((np.ones(y.size), beat_c, mean_hr_c))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    adjusted = coef[0] + (y - fitted)

    adjusted_rows: list[dict[str, object]] = []
    for idx, _y, beat_i, hr_i in design_rows:
        base = dict(baseline_obs[idx])
        adj = float(adjusted[len(adjusted_rows)])
        base["control"] = CONTROL_BEAT_COUNT
        base["endpoint_value"] = adj
        base["paired_change"] = adj - _as_float(base.get("baseline_endpoint_value"))
        base["beat_count"] = float(beat_i)
        base["mean_hr"] = float(hr_i)
        base["eligibility"] = "computed"
        base["exclusion_reason"] = ""
        base["analysis_role"] = "cardiac_control"
        adjusted_rows.append(base)
    return adjusted_rows, ""


def _cluster_ci(changes: Sequence[dict[str, object]]) -> tuple[float, float, float]:
    boot_rows = []
    for row in changes:
        delta = _as_float(row.get("paired_change"))
        if not math.isfinite(delta):
            continue
        boot_rows.append(
            {
                "delta_endpoint_index": delta,
                "participant_id": _as_str(row.get("biological_participant_id")),
                "dataset_id": _as_str(row.get("dataset_id")),
            }
        )
    if len(boot_rows) < 2:
        return float("nan"), float("nan"), float("nan")
    return cluster_bootstrap_mean_ci(
        boot_rows,
        unit_field="participant_id",
        n_draws=2000,
        seed=17,
    )


def canonical_panel_d_control(control: str) -> str:
    """Map alternate control IDs onto the locked Panel D display IDs."""
    key = _as_str(control)
    return PANEL_D_CONTROL_ALIASES.get(key, key)


def panel_d_short_display_label(control: str) -> str:
    key = canonical_panel_d_control(control)
    return PANEL_D_SHORT_DISPLAY_LABELS.get(key, PANEL_D_SHORT_DISPLAY_LABELS.get(control, control))


def panel_d_control_family(control: str) -> str:
    key = canonical_panel_d_control(control)
    return PANEL_D_CONTROL_FAMILY.get(key, PANEL_D_CONTROL_FAMILY.get(control, "signal_level"))


def short_not_computable_reason_code(reason: str) -> str:
    """Compact figure annotation; full reason remains in tables/metadata."""
    text = _as_str(reason).casefold()
    if not text:
        return ""
    # Structural 1 Hz event-mask / strict-common-support incompatibility (not a near-miss).
    if (
        "1hz_envelope_cannot_support_event_centered_masking" in text
        or "strict_global_common_support_after_masking" in text
        or ("1 hz" in text and "mask" in text)
        or ("1hz" in text and "mask" in text)
        or "cannot support event-centered masking" in text
        or "cannot support event_centered_masking" in text
    ):
        return "1 Hz mask incompatible"
    if "insufficient_common_support" in text or text == "insufficient_overlap_after_masking":
        return "insufficient support"
    if "support" in text or "masking" in text or "common_support" in text:
        return "insufficient support"
    if (
        "ppg_only" in text
        or "no_ecg" in text
        or ("ecg" in text and ("unavailable" in text or "not available" in text))
    ):
        return "ECG unavailable"
    if "channel" in text or "reaggreg" in text:
        return "channel reaggregation unavailable"
    if "covariate" in text or "beat_count" in text:
        return "covariates unavailable"
    return "not computable"


def verify_panel_d_summary_integrity(
    summaries: Sequence[Mapping[str, object]],
    *,
    n_baseline_eligible: int | None = None,
) -> list[str]:
    """Return human-readable integrity failures for visualization gating."""
    failures: list[str] = []
    by_control = {_as_str(row.get("control")): row for row in summaries}
    baseline = by_control.get(CONTROL_BASELINE, {})
    baseline_n = _as_int(baseline.get("n_observations"))
    if n_baseline_eligible is None:
        n_baseline_eligible = baseline_n
    for control in PANEL_D_PLOT_CONTROL_ORDER:
        row = by_control.get(control)
        if row is None:
            continue
        status = _as_str(row.get("computability_status")).casefold()
        estimate = _as_float(row.get("estimate"))
        lo = _as_float(row.get("ci_lower"))
        hi = _as_float(row.get("ci_upper"))
        change = _as_float(row.get("change_from_baseline"))
        clo = _as_float(row.get("change_ci_lower"))
        chi = _as_float(row.get("change_ci_upper"))
        if status == "computed":
            if not math.isfinite(estimate):
                failures.append(f"{control}: computed row missing finite estimate")
            if not math.isfinite(change):
                failures.append(f"{control}: computed row missing finite change_from_baseline")
            n_obs = _as_int(row.get("n_observations"))
            if n_obs >= 2:
                for name, value in (
                    ("ci_lower", lo),
                    ("ci_upper", hi),
                    ("change_ci_lower", clo),
                    ("change_ci_upper", chi),
                ):
                    if not math.isfinite(value):
                        failures.append(f"{control}: computed row missing finite {name}")
            if math.isfinite(lo) and math.isfinite(hi) and lo > hi:
                failures.append(f"{control}: coefficient CI inverted")
            if math.isfinite(clo) and math.isfinite(chi) and clo > chi:
                failures.append(f"{control}: change CI inverted")
            # Δ sign convention: change_from_baseline must equal controlled − baseline
            # when both point estimates are present.
            base_est = _as_float(row.get("baseline_estimate"))
            ctl_est = _as_float(row.get("controlled_estimate"))
            if math.isfinite(base_est) and math.isfinite(ctl_est) and math.isfinite(change):
                expected = ctl_est - base_est
                if abs(expected - change) > 1e-9:
                    failures.append(
                        f"{control}: Δ sign/value mismatch "
                        f"(controlled-baseline={expected}, change_from_baseline={change})"
                    )
        else:
            for name, value in (
                ("estimate", estimate),
                ("ci_lower", lo),
                ("ci_upper", hi),
                ("change_from_baseline", change),
                ("change_ci_lower", clo),
                ("change_ci_upper", chi),
            ):
                if math.isfinite(value):
                    failures.append(f"{control}: NC row has finite {name}={value}")
            if not _as_str(row.get("computability_reason")):
                failures.append(f"{control}: NC row missing not_computable_reason")
        denom = _as_int(row.get("n_baseline_eligible"), baseline_n)
        if n_baseline_eligible > 0 and denom != n_baseline_eligible:
            failures.append(
                f"{control}: n_baseline_eligible={denom} != baseline eligible {n_baseline_eligible}"
            )
    return failures


def _summary_display_labels(dataset_qc_rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    present = {_as_str(r.get("cardiac_signal_type")) for r in dataset_qc_rows if _as_str(r.get("cardiac_signal_type"))}
    ecg_only = present == {"ECG"}
    ppg_only = present == {"PPG"}
    return {
        CONTROL_BASELINE: "Baseline",
        CONTROL_ICA_TEMPLATE: (
            "ECG cardiac-template subtraction"
            if ecg_only
            else ("PPG pulse-locked template subtraction" if ppg_only else "Cardiac-template subtraction")
        ),
        CONTROL_RPEAK_MASK: (
            "ECG R-peak mask"
            if ecg_only
            else ("PPG pulse-event mask" if ppg_only else "Cardiac-event mask")
        ),
        CONTROL_BEAT_COUNT: "Beat-count adjusted",
        CONTROL_ECG_CHANNELS: "ECG-prone channels removed",
    }


def _summarize_control(
    control: str,
    baseline_obs: Sequence[Mapping[str, object]],
    control_obs: Sequence[Mapping[str, object]],
    *,
    computability_status: str,
    computability_reason: str,
    display_label: str,
    n_baseline_eligible: int | None = None,
) -> dict[str, object]:
    by_key = {_paired_key(row): row for row in baseline_obs}
    pairs: list[tuple[float, float, str]] = []
    for row in control_obs:
        key = _paired_key(row)
        if key not in by_key:
            continue
        base = _as_float(by_key[key].get("endpoint_value"))
        ctl = _as_float(row.get("endpoint_value"))
        pid = _as_str(row.get("biological_participant_id"))
        if math.isfinite(base) and math.isfinite(ctl):
            pairs.append((base, ctl, pid))
    if control == CONTROL_BASELINE:
        pairs = [
            (
                _as_float(row.get("endpoint_value")),
                _as_float(row.get("endpoint_value")),
                _as_str(row.get("biological_participant_id")),
            )
            for row in baseline_obs
            if math.isfinite(_as_float(row.get("endpoint_value")))
        ]

    baseline_eligible = (
        int(n_baseline_eligible)
        if n_baseline_eligible is not None
        else sum(1 for row in baseline_obs if math.isfinite(_as_float(row.get("endpoint_value"))))
    )
    family = panel_d_control_family(control)
    short_label = panel_d_short_display_label(control)
    reason_code = (
        ""
        if computability_status == "computed"
        else short_not_computable_reason_code(computability_reason)
    )

    if not pairs:
        return {
            "control": control,
            "display_label": display_label,
            "display_label_short": short_label,
            "control_family": family,
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "pooled",
            "state_or_contrast": "pooled",
            "baseline_estimate": float("nan"),
            "controlled_estimate": float("nan"),
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "change_from_baseline": float("nan"),
            "change_ci_lower": float("nan"),
            "change_ci_upper": float("nan"),
            "paired_correlation": float("nan"),
            "sign_retention_fraction": float("nan"),
            "n_sign_changes": 0,
            "n_observations": 0,
            "n_baseline_eligible": baseline_eligible,
            "n_biological_participants": 0,
            "denominator_label": f"0/{baseline_eligible}",
            "analysis_role": "cardiac_control",
            "computability_status": computability_status,
            "computability_reason": computability_reason,
            "short_reason_code": reason_code,
        }

    baseline_vals = np.asarray([p[0] for p in pairs], dtype=float)
    controlled_vals = np.asarray([p[1] for p in pairs], dtype=float)
    changes = controlled_vals - baseline_vals
    # Coefficient CI: cluster-bootstrap the controlled endpoint mean.
    _coef_mean, coef_lo, coef_hi = _cluster_ci(
        [
            {
                "paired_change": float(controlled_vals[i]),
                "biological_participant_id": pairs[i][2],
                "dataset_id": "",
            }
            for i in range(len(pairs))
        ]
    )
    # Paired-change CI: cluster-bootstrap control − baseline.
    _chg_mean, chg_lo, chg_hi = _cluster_ci(
        [
            {
                "paired_change": float(changes[i]),
                "biological_participant_id": pairs[i][2],
                "dataset_id": "",
            }
            for i in range(len(pairs))
        ]
    )
    corr = float("nan")
    if baseline_vals.size >= 2:
        corr = float(np.corrcoef(baseline_vals, controlled_vals)[0, 1])
    n_sign_changes = 0
    n_sign_same = 0
    for b, c in zip(baseline_vals, controlled_vals, strict=False):
        sb = _sign(float(b))
        sc = _sign(float(c))
        if sb == sc:
            n_sign_same += 1
        else:
            n_sign_changes += 1
    sign_retention = float(n_sign_same / len(pairs)) if pairs else float("nan")
    unique_participants = len({p[2] for p in pairs if p[2]})
    n_obs = len(pairs)
    return {
        "control": control,
        "display_label": display_label,
        "display_label_short": short_label,
        "control_family": family,
        "endpoint_name": ENDPOINT_ZLPI,
        "band": "pooled",
        "state_or_contrast": "pooled",
        "baseline_estimate": float(np.mean(baseline_vals)),
        "controlled_estimate": float(np.mean(controlled_vals)),
        "estimate": float(np.mean(controlled_vals)),
        "ci_lower": coef_lo if math.isfinite(coef_lo) else float("nan"),
        "ci_upper": coef_hi if math.isfinite(coef_hi) else float("nan"),
        "change_from_baseline": float(np.mean(changes)),
        "change_ci_lower": chg_lo if math.isfinite(chg_lo) else float("nan"),
        "change_ci_upper": chg_hi if math.isfinite(chg_hi) else float("nan"),
        "paired_correlation": corr,
        "sign_retention_fraction": sign_retention,
        "n_sign_changes": n_sign_changes,
        "n_observations": n_obs,
        "n_baseline_eligible": baseline_eligible,
        "n_biological_participants": unique_participants,
        "denominator_label": f"{n_obs}/{baseline_eligible}",
        "analysis_role": ("primary_reference" if control == CONTROL_BASELINE else "cardiac_control"),
        "computability_status": computability_status,
        "computability_reason": computability_reason,
        "short_reason_code": reason_code,
    }


def compute_panel_d_cardiac_controls(
    paired_rows: Sequence[Mapping[str, object]],
    subject_rows: Sequence[Mapping[str, object]],
    *,
    protocol_rows: Sequence[Mapping[str, object]] = (),
    cardiac_qc_rows: Sequence[Mapping[str, object]] = (),
) -> PanelDCardiacControlsResult:
    baseline_seed = _primary_baseline_rows(paired_rows)
    baseline_stub = [{"dataset_id": _as_str(row.get("dataset_id"))} for row in baseline_seed]
    dataset_qc = _build_dataset_qc(baseline_stub, protocol_rows, cardiac_qc_rows)
    qc_lookup = _dataset_qc_lookup(dataset_qc)
    baseline_obs = _build_baseline_observations(paired_rows, qc_lookup)

    control_rows: dict[str, list[dict[str, object]]] = {
        CONTROL_BASELINE: list(baseline_obs),
    }
    control_rows[CONTROL_ICA_TEMPLATE] = _emit_not_computable(
        baseline_obs,
        control=CONTROL_ICA_TEMPLATE,
    )
    if not control_rows[CONTROL_ICA_TEMPLATE]:
        control_rows[CONTROL_ICA_TEMPLATE] = [
            {
                "control": CONTROL_ICA_TEMPLATE,
                "eligibility": "not_computable",
                "exclusion_reason": _control_reason(CONTROL_ICA_TEMPLATE, "UNKNOWN"),
                "not_computable_reason": _control_reason(CONTROL_ICA_TEMPLATE, "UNKNOWN"),
            }
        ]
    control_rows[CONTROL_RPEAK_MASK] = _emit_not_computable(
        baseline_obs,
        control=CONTROL_RPEAK_MASK,
    )
    if not control_rows[CONTROL_RPEAK_MASK]:
        control_rows[CONTROL_RPEAK_MASK] = [
            {
                "control": CONTROL_RPEAK_MASK,
                "eligibility": "not_computable",
                "exclusion_reason": _control_reason(CONTROL_RPEAK_MASK, "UNKNOWN"),
                "not_computable_reason": _control_reason(CONTROL_RPEAK_MASK, "UNKNOWN"),
            }
        ]
    beat_rows, beat_reason = _beat_count_adjusted_observations(baseline_obs, subject_rows)
    if beat_rows:
        control_rows[CONTROL_BEAT_COUNT] = beat_rows
    else:
        control_rows[CONTROL_BEAT_COUNT] = _emit_not_computable(
            baseline_obs,
            control=CONTROL_BEAT_COUNT,
        )
        if not control_rows[CONTROL_BEAT_COUNT]:
            # No baseline rows: still emit an explicit NC reason for summaries.
            control_rows[CONTROL_BEAT_COUNT] = [
                {
                    "control": CONTROL_BEAT_COUNT,
                    "eligibility": "not_computable",
                    "exclusion_reason": beat_reason
                    or _control_reason(CONTROL_BEAT_COUNT, "UNKNOWN"),
                    "not_computable_reason": beat_reason
                    or _control_reason(CONTROL_BEAT_COUNT, "UNKNOWN"),
                }
            ]
        for row in control_rows[CONTROL_BEAT_COUNT]:
            row["exclusion_reason"] = beat_reason or _as_str(
                row.get("exclusion_reason")
            ) or _control_reason(CONTROL_BEAT_COUNT, "UNKNOWN")
            row["not_computable_reason"] = row["exclusion_reason"]
    control_rows[CONTROL_ECG_CHANNELS] = _emit_not_computable(
        baseline_obs,
        control=CONTROL_ECG_CHANNELS,
    )
    if not control_rows[CONTROL_ECG_CHANNELS]:
        control_rows[CONTROL_ECG_CHANNELS] = [
            {
                "control": CONTROL_ECG_CHANNELS,
                "eligibility": "not_computable",
                "exclusion_reason": _control_reason(CONTROL_ECG_CHANNELS, "UNKNOWN"),
                "not_computable_reason": _control_reason(CONTROL_ECG_CHANNELS, "UNKNOWN"),
            }
        ]

    observations: list[dict[str, object]] = []
    for control in CONTROL_ORDER:
        observations.extend(control_rows.get(control, []))

    display_labels = _summary_display_labels(dataset_qc)
    n_baseline_eligible = sum(
        1 for row in baseline_obs if math.isfinite(_as_float(row.get("endpoint_value")))
    )
    summaries: list[dict[str, object]] = []
    for control in CONTROL_ORDER:
        rows = control_rows.get(control, [])
        if control == CONTROL_BASELINE:
            summaries.append(
                _summarize_control(
                    control,
                    baseline_obs,
                    rows,
                    computability_status="computed" if baseline_obs else "not_computable",
                    computability_reason=(
                        ""
                        if baseline_obs
                        else "no_baseline_eligible_observations"
                    ),
                    display_label=display_labels[control],
                    n_baseline_eligible=n_baseline_eligible,
                )
            )
            continue
        computed = any(_as_str(r.get("eligibility")) == "computed" for r in rows)
        reason = ""
        if not computed:
            for row in rows:
                reason = _as_str(row.get("exclusion_reason") or row.get("not_computable_reason"))
                if reason:
                    break
            if not reason:
                reason = _control_reason(control, "UNKNOWN")
        summaries.append(
            _summarize_control(
                control,
                baseline_obs,
                rows,
                computability_status=("computed" if computed else "not_computable"),
                computability_reason=reason,
                display_label=display_labels[control],
                n_baseline_eligible=n_baseline_eligible,
            )
        )

    metadata = {
        "endpoint_contract": "D240 absolute-log10 ZLPI",
        "duration_s": EXPECTED_PRIMARY_DURATION_S,
        "lag_range_s": [-60, 60],
        "flank_range_s": [int(ZLPI_FLANKS_S[0]), int(ZLPI_FLANKS_S[1])],
        "fisher_transformation": "fisher_z",
        "cardiac_event_semantics": {
            "ECG": "r_peak",
            "PPG": "ppg_systolic_peak",
        },
        "cardiac_event_interpretation": (
            "PPG pulse-event mask tests pulse-synchronous contamination and does not "
            "directly test ECG electrical-field leakage."
        ),
        "control_display_labels": display_labels,
        "panel_d_plot_control_order": list(PANEL_D_PLOT_CONTROL_ORDER),
        "panel_d_short_display_labels": dict(PANEL_D_SHORT_DISPLAY_LABELS),
        "unit_definitions": {
            "n_observations": (
                "observation × band rows with finite paired baseline and controlled values"
            ),
            "n_baseline_eligible": (
                "baseline-eligible observation × band rows for the locked D240 absolute_log10 ZLPI"
            ),
            "n_biological_participants": "unique biological participants in the paired summary",
            "denominator_label": "n_observations / n_baseline_eligible",
        },
        "r_peak_mask_width_s": {"pre_r": 0.05, "post_r": 0.05},
        "mask_handling_method": "not_computable",
        "cardiac_template_method": "not_computable",
        "ecg_prone_channel_list": [],
        "beat_count_model_formula": (
            "delta_endpoint_index ~ 1 + centered_beat_count + centered_mean_hr "
            "(participant-level clustered bootstrap for CI)"
        ),
        "covariates": ["beat_count", "mean_hr"],
        "bootstrap_clustering_unit": "biological_participant_id",
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 17,
        "software_version": "confirmatory_panel_d_v3_viz",
        "git_commit": "unknown",
        "not_computable_rules": {
            CONTROL_ICA_TEMPLATE: "requires high-rate multichannel EEG + event-locked decomposition artifacts",
            CONTROL_RPEAK_MASK: "requires high-rate EEG + retained event timestamps on raw time base",
            CONTROL_ECG_CHANNELS: "requires channel-level pre-aggregation EEG and prespecified channel applicability",
        },
        "visualization_note": (
            "Panel D is a visualization-only paired-estimation display. Coefficient CIs "
            "(ci_lower/ci_upper) are participant-clustered bootstraps of the controlled "
            "coefficient; change CIs are bootstraps of control−baseline. NC values are "
            "missing, not zero."
        ),
    }
    return PanelDCardiacControlsResult(
        observations=tuple(observations),
        summaries=tuple(summaries),
        dataset_qc=tuple(dataset_qc),
        metadata=metadata,
    )


def write_panel_d_cardiac_control_exports(
    result: PanelDCardiacControlsResult,
    source_dir: Path,
) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    observations_path = source_dir / "figure3_panel_d_cardiac_controls_observations.csv"
    summaries_path = source_dir / "figure3_panel_d_cardiac_controls_summaries.csv"
    dataset_qc_path = source_dir / "figure3_panel_d_cardiac_controls_dataset_qc.csv"
    metadata_path = source_dir / "figure3_panel_d_cardiac_controls_metadata.json"

    def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
        import csv

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                payload: dict[str, object] = {}
                for field in fields:
                    value = row.get(field, "")
                    if isinstance(value, float) and not math.isfinite(value):
                        payload[field] = ""
                    else:
                        payload[field] = value
                writer.writerow(payload)

    _write_csv(observations_path, result.observations, OBSERVATION_COLUMNS)
    _write_csv(summaries_path, result.summaries, SUMMARY_COLUMNS)
    _write_csv(dataset_qc_path, result.dataset_qc, DATASET_QC_COLUMNS)
    metadata_path.write_text(json.dumps(result.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "observations": observations_path,
        "summaries": summaries_path,
        "dataset_qc": dataset_qc_path,
        "metadata": metadata_path,
    }


def compute_panel_d_from_observation_controls(
    observation_rows: Sequence[Mapping[str, object]],
    *,
    dataset_qc_rows: Sequence[Mapping[str, object]] = (),
    metadata: Mapping[str, object] | None = None,
) -> PanelDCardiacControlsResult:
    """Summarize precomputed upstream observation-level controls for C7 plotting."""
    obs = []
    for row in observation_rows:
        item = dict(row)
        if "control" not in item:
            item["control"] = _as_str(item.get("control_type"))
        if "endpoint_name" not in item:
            item["endpoint_name"] = ENDPOINT_ZLPI
        if "endpoint_value" not in item:
            item["endpoint_value"] = item.get("controlled_zlpi", "")
        if "baseline_endpoint_value" not in item:
            item["baseline_endpoint_value"] = item.get("baseline_zlpi", "")
        if "paired_change" not in item:
            item["paired_change"] = item.get("delta_vs_baseline", "")
        if "eligibility" not in item:
            item["eligibility"] = "computed" if _as_bool(item.get("computable")) else "not_computable"
        if "exclusion_reason" not in item:
            item["exclusion_reason"] = _as_str(item.get("not_computable_reason"))
        obs.append(item)

    present_controls = {
        _as_str(row.get("control_type")) for row in obs if _as_str(row.get("control_type"))
    }
    # Locked manuscript order: baseline reference + Panel D plot rows, then any extras.
    ordered_controls: list[str] = [CONTROL_BASELINE]
    for control in PANEL_D_PLOT_CONTROL_ORDER:
        if control not in ordered_controls:
            ordered_controls.append(control)
    for control in sorted(present_controls):
        if control not in ordered_controls:
            ordered_controls.append(control)

    baseline_by_key: dict[tuple[str, ...], dict[str, object]] = {}
    for row in obs:
        if _as_str(row.get("control_type")) != CONTROL_BASELINE:
            continue
        key = (
            _as_str(row.get("dataset_id")).casefold(),
            _as_str(row.get("participant_id")).casefold(),
            _as_str(row.get("session_id"), "single").casefold(),
            _as_str(row.get("observation_id")).casefold(),
            _as_str(row.get("band")).casefold(),
        )
        baseline_by_key[key] = row

    n_baseline_eligible = 0
    for row in baseline_by_key.values():
        base = _as_float(row.get("controlled_zlpi") or row.get("baseline_zlpi"))
        if math.isfinite(base) and _as_bool(row.get("computable", True)):
            n_baseline_eligible += 1
    if n_baseline_eligible == 0:
        n_baseline_eligible = len(baseline_by_key)

    label_map = {
        CONTROL_BASELINE: "Baseline",
        CONTROL_ECG_MASK: "ECG R-peak mask",
        CONTROL_PPG_MASK: "PPG pulse-event mask",
        CONTROL_ECG_TEMPLATE: "ECG cardiac-template subtraction",
        CONTROL_PPG_TEMPLATE: "PPG pulse-locked template subtraction",
        CONTROL_BEAT_COUNT: "Beat-count adjusted",
        CONTROL_ECG_CHANNELS: "ECG-prone channels removed",
        CONTROL_ICA_TEMPLATE: "Cardiac-template subtraction",
        CONTROL_RPEAK_MASK: "Cardiac-event mask",
    }

    def _empty_summary(control: str, reason: str) -> dict[str, object]:
        return {
            "control": control,
            "display_label": label_map.get(control, control),
            "display_label_short": panel_d_short_display_label(control),
            "control_family": panel_d_control_family(control),
            "endpoint_name": ENDPOINT_ZLPI,
            "band": "pooled",
            "state_or_contrast": "pooled",
            "baseline_estimate": float("nan"),
            "controlled_estimate": float("nan"),
            "estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "change_from_baseline": float("nan"),
            "change_ci_lower": float("nan"),
            "change_ci_upper": float("nan"),
            "paired_correlation": float("nan"),
            "sign_retention_fraction": float("nan"),
            "n_sign_changes": 0,
            "n_observations": 0,
            "n_baseline_eligible": n_baseline_eligible,
            "n_biological_participants": 0,
            "denominator_label": f"0/{n_baseline_eligible}",
            "analysis_role": ("primary_reference" if control == CONTROL_BASELINE else "cardiac_control"),
            "computability_status": "not_computable",
            "computability_reason": reason,
            "short_reason_code": short_not_computable_reason_code(reason),
        }

    summaries: list[dict[str, object]] = []
    for control in ordered_controls:
        members = [row for row in obs if _as_str(row.get("control_type")) == control]
        if not members and control != CONTROL_BASELINE:
            summaries.append(
                _empty_summary(
                    control,
                    "control_absent_from_upstream_observation_table",
                )
            )
            continue
        pairs: list[tuple[float, float, str]] = []
        computability_reason = ""
        for row in members:
            key = (
                _as_str(row.get("dataset_id")).casefold(),
                _as_str(row.get("participant_id")).casefold(),
                _as_str(row.get("session_id"), "single").casefold(),
                _as_str(row.get("observation_id")).casefold(),
                _as_str(row.get("band")).casefold(),
            )
            base_row = baseline_by_key.get(key)
            if base_row is None:
                continue
            base = _as_float(base_row.get("controlled_zlpi") or base_row.get("baseline_zlpi"))
            ctl = _as_float(row.get("controlled_zlpi"))
            pid = _as_str(row.get("participant_id"))
            if math.isfinite(base) and math.isfinite(ctl) and _as_bool(row.get("computable", True)):
                pairs.append((base, ctl, pid))
            if not _as_bool(row.get("computable")) and not computability_reason:
                computability_reason = _as_str(row.get("not_computable_reason"))
        if not pairs:
            summaries.append(
                _empty_summary(
                    control,
                    computability_reason or "no_matched_computable_observations",
                )
            )
            continue
        baseline_vals = np.asarray([p[0] for p in pairs], dtype=float)
        controlled_vals = np.asarray([p[1] for p in pairs], dtype=float)
        changes = controlled_vals - baseline_vals
        _coef_mean, coef_lo, coef_hi = _cluster_ci(
            [
                {
                    "paired_change": float(controlled_vals[i]),
                    "biological_participant_id": pairs[i][2],
                    "dataset_id": "",
                }
                for i in range(len(pairs))
            ]
        )
        _chg_mean, chg_lo, chg_hi = _cluster_ci(
            [
                {
                    "paired_change": float(changes[i]),
                    "biological_participant_id": pairs[i][2],
                    "dataset_id": "",
                }
                for i in range(len(pairs))
            ]
        )
        corr = float("nan")
        if baseline_vals.size >= 2:
            corr = float(np.corrcoef(baseline_vals, controlled_vals)[0, 1])
        n_sign_changes = 0
        n_sign_same = 0
        for b, c in zip(baseline_vals, controlled_vals, strict=False):
            if _sign(float(b)) == _sign(float(c)):
                n_sign_same += 1
            else:
                n_sign_changes += 1
        n_obs = len(pairs)
        summaries.append(
            {
                "control": control,
                "display_label": label_map.get(control, control),
                "display_label_short": panel_d_short_display_label(control),
                "control_family": panel_d_control_family(control),
                "endpoint_name": ENDPOINT_ZLPI,
                "band": "pooled",
                "state_or_contrast": "pooled",
                "baseline_estimate": float(np.mean(baseline_vals)),
                "controlled_estimate": float(np.mean(controlled_vals)),
                "estimate": float(np.mean(controlled_vals)),
                "ci_lower": coef_lo if math.isfinite(coef_lo) else float("nan"),
                "ci_upper": coef_hi if math.isfinite(coef_hi) else float("nan"),
                "change_from_baseline": float(np.mean(changes)),
                "change_ci_lower": chg_lo if math.isfinite(chg_lo) else float("nan"),
                "change_ci_upper": chg_hi if math.isfinite(chg_hi) else float("nan"),
                "paired_correlation": corr,
                "sign_retention_fraction": float(n_sign_same / len(pairs)),
                "n_sign_changes": n_sign_changes,
                "n_observations": n_obs,
                "n_baseline_eligible": n_baseline_eligible,
                "n_biological_participants": len({p[2] for p in pairs if p[2]}),
                "denominator_label": f"{n_obs}/{n_baseline_eligible}",
                "analysis_role": ("primary_reference" if control == CONTROL_BASELINE else "cardiac_control"),
                "computability_status": "computed",
                "computability_reason": "",
                "short_reason_code": "",
            }
        )

    payload = dict(metadata or {})
    payload.setdefault("panel_d_plot_control_order", list(PANEL_D_PLOT_CONTROL_ORDER))
    payload.setdefault("panel_d_short_display_labels", dict(PANEL_D_SHORT_DISPLAY_LABELS))
    payload.setdefault(
        "unit_definitions",
        {
            "n_observations": (
                "observation × band rows with finite paired baseline and controlled values"
            ),
            "n_baseline_eligible": (
                "baseline-eligible observation × band rows for the locked D240 absolute_log10 ZLPI"
            ),
            "n_biological_participants": "unique biological participants in the paired summary",
            "denominator_label": "n_observations / n_baseline_eligible",
        },
    )
    payload.setdefault(
        "visualization_note",
        (
            "Panel D is a visualization-only paired-estimation display. Coefficient CIs "
            "(ci_lower/ci_upper) are participant-clustered bootstraps of the controlled "
            "coefficient; change CIs are bootstraps of control−baseline. NC values are "
            "missing, not zero."
        ),
    )
    payload["n_baseline_eligible_observation_x_band"] = n_baseline_eligible
    payload["n_unique_baseline_observations"] = len(
        {
            (
                _as_str(row.get("dataset_id")).casefold(),
                _as_str(row.get("observation_id")).casefold(),
            )
            for row in baseline_by_key.values()
        }
    )
    payload["n_unique_baseline_participants"] = len(
        {
            (
                _as_str(row.get("dataset_id")).casefold(),
                _as_str(row.get("participant_id")).casefold(),
            )
            for row in baseline_by_key.values()
        }
    )
    return PanelDCardiacControlsResult(
        observations=tuple(obs),
        summaries=tuple(summaries),
        dataset_qc=tuple(dict(row) for row in dataset_qc_rows),
        metadata=payload,
    )


__all__ = [
    "CONTROL_BASELINE",
    "CONTROL_BEAT_COUNT",
    "CONTROL_ECG_CHANNELS",
    "CONTROL_ECG_MASK",
    "CONTROL_ECG_TEMPLATE",
    "CONTROL_ICA_TEMPLATE",
    "CONTROL_ORDER",
    "CONTROL_PPG_MASK",
    "CONTROL_PPG_TEMPLATE",
    "CONTROL_RPEAK_MASK",
    "DATASET_QC_COLUMNS",
    "OBSERVATION_COLUMNS",
    "PANEL_D_CONTROL_FAMILY",
    "PANEL_D_PLOT_CONTROL_ORDER",
    "PANEL_D_SHORT_DISPLAY_LABELS",
    "PanelDCardiacControlsResult",
    "SUMMARY_COLUMNS",
    "canonical_panel_d_control",
    "compute_panel_d_cardiac_controls",
    "compute_panel_d_from_observation_controls",
    "panel_d_control_family",
    "panel_d_short_display_label",
    "short_not_computable_reason_code",
    "verify_panel_d_summary_integrity",
    "write_panel_d_cardiac_control_exports",
]
