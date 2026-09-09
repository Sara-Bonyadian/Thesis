#!/usr/bin/env python3
"""Post-fix validation for Figure 3D cardiac controls (read-only on raw data)."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ppg_eeg.confirmatory.cardiac_controls_upstream import (
    CONTROL_BASELINE,
    CONTROL_BEAT_ADJUST,
    CONTROL_ECG_CHANNELS,
    CONTROL_ECG_MASK,
    CONTROL_ECG_TEMPLATE,
    CONTROL_PPG_MASK,
    CONTROL_PPG_TEMPLATE,
    run_confirmatory_cardiac_controls_upstream,
)
from ppg_eeg.confirmatory.duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S
from ppg_eeg.confirmatory.panel_d_cardiac_controls import compute_panel_d_from_observation_controls
from ppg_eeg.confirmatory.reason_codes import INSUFFICIENT_COMMON_SUPPORT, UNSUPPORTED_CONTROL_FOR_MODALITY

ROOT = Path(__file__).resolve().parents[1]
DERIV_ROOT = ROOT / "derivatives" / "confirmatory_temporal_coupling"

DATASETS: list[tuple[str, str]] = [
    # Smallest / fastest first for incremental validation output.
    ("primary", "ds006848"),
    ("primary", "ds003838"),
    ("primary", "ds004587"),
    ("sensitivity", "ds004582"),
    ("sensitivity", "mindfulness"),
    ("sensitivity", "hiit"),
    ("primary", "ds003690"),
]

REPORT_CONTROLS: tuple[str, ...] = (
    CONTROL_BASELINE,
    CONTROL_BEAT_ADJUST,
    CONTROL_ECG_TEMPLATE,
    CONTROL_PPG_TEMPLATE,
    CONTROL_ECG_MASK,
    CONTROL_PPG_MASK,
    CONTROL_ECG_CHANNELS,
)

CONTROL_LABELS: dict[str, str] = {
    CONTROL_BASELINE: "baseline",
    CONTROL_BEAT_ADJUST: "beat_count_adjusted",
    CONTROL_ECG_TEMPLATE: "ecg_template_subtraction",
    CONTROL_PPG_TEMPLATE: "ppg_template_subtraction",
    CONTROL_ECG_MASK: "ecg_event_mask",
    CONTROL_PPG_MASK: "ppg_event_mask",
    CONTROL_ECG_CHANNELS: "ecg_prone_channels_removed",
}


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _as_float(value: object) -> float:
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _read_protocol_modality(dataset_dir: Path) -> str:
    path = dataset_dir / "C0" / "protocol_audit.csv"
    if not path.is_file():
        return "unknown"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mods = sorted(
        {
            str(r.get("cardiac_modality", "")).strip().casefold()
            for r in rows
            if str(r.get("cardiac_modality", "")).strip()
        }
    )
    return ";".join(mods) if mods else "unknown"


def _stage_ready(dataset_dir: Path) -> bool:
    needed = ("C0", "C1a", "C1b", "C1c", "C3")
    return all((dataset_dir / stage).is_dir() for stage in needed)


def _observation_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("dataset_id", "")).casefold(),
        str(row.get("participant_id", "")).casefold(),
        str(row.get("session_id", "single")).casefold(),
        str(row.get("observation_id", "")).casefold(),
        str(row.get("band", "")).casefold(),
    )


def _aggregate_observations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[str(row.get("control_type", ""))].append(row)

    baseline_rows = by_control.get(CONTROL_BASELINE, [])
    baseline_eligible_keys = {
        _observation_key(r)
        for r in baseline_rows
        if _as_bool(r.get("computable"))
        and math.isfinite(_as_float(r.get("baseline_zlpi") or r.get("controlled_zlpi")))
    }
    n_baseline_eligible = len(baseline_eligible_keys)
    participants = {
        str(r.get("participant_id", "")).casefold()
        for r in baseline_rows
        if str(r.get("participant_id", "")).strip()
    }

    out: dict[str, dict[str, Any]] = {}
    for control in REPORT_CONTROLS:
        members = by_control.get(control, [])
        n_total = len(members)
        computable_rows = [r for r in members if _as_bool(r.get("computable"))]
        n_computable = len(computable_rows)
        nc_rows = [r for r in members if not _as_bool(r.get("computable"))]

        paired_keys: set[tuple[str, str, str, str, str]] = set()
        for row in computable_rows:
            key = _observation_key(row)
            if key in baseline_eligible_keys and math.isfinite(_as_float(row.get("controlled_zlpi"))):
                paired_keys.add(key)

        reason_codes = Counter(
            str(r.get("reason_code") or "").strip()
            for r in nc_rows
            if str(r.get("reason_code") or "").strip()
        )

        retained: list[int] = []
        if control == CONTROL_ECG_CHANNELS:
            for row in members:
                val = _as_float(row.get("n_channels_retained"))
                if math.isfinite(val) and val > 0:
                    retained.append(int(val))

        out[control] = {
            "n_baseline_eligible": n_baseline_eligible,
            "n_control_computable": n_computable,
            "n_paired_complete_case": len(paired_keys),
            "composition_differs_from_baseline": bool(len(paired_keys) != n_baseline_eligible),
            "n_total_rows": n_total,
            "n_nc": len(nc_rows),
            "pct_computable": (100.0 * n_computable / n_total) if n_total else 0.0,
            "reason_code_frequencies": dict(sorted(reason_codes.items(), key=lambda kv: (-kv[1], kv[0]))),
            "n_biological_participants": len(participants),
            "retained_channels": {
                "min": min(retained) if retained else None,
                "median": int(statistics.median(retained)) if retained else None,
                "max": max(retained) if retained else None,
                "n_rows_with_retained_count": len(retained),
            },
            "nc_zero_violations": sum(
                1
                for r in nc_rows
                if math.isfinite(_as_float(r.get("controlled_zlpi")))
                or (
                    str(r.get("controlled_zlpi", "")).strip() == "0"
                    or str(r.get("controlled_zlpi", "")).strip() == "0.0"
                )
            ),
        }
    return out


def _grade_control(
    *,
    role: str,
    dataset_id: str,
    declared_modality: str,
    control: str,
    metrics: dict[str, Any],
    checks: dict[str, Any],
) -> str:
    mod = declared_modality.casefold()
    ecg_only = "ecg" in mod and "ppg" not in mod
    ppg_only = "ppg" in mod and "ecg" not in mod
    n_comp = int(metrics["n_control_computable"])
    n_total = int(metrics["n_total_rows"])

    if control == CONTROL_BASELINE:
        return "PASS" if n_comp > 0 else "NC"

    if n_total == 0:
        return "NC"

    if ecg_only and control in {CONTROL_PPG_MASK, CONTROL_PPG_TEMPLATE}:
        if n_comp == 0 and UNSUPPORTED_CONTROL_FOR_MODALITY in metrics["reason_code_frequencies"]:
            return "PASS"
        if n_comp > 0:
            return "NC"
        return "PARTIAL"

    if ppg_only and control in {CONTROL_ECG_TEMPLATE, CONTROL_ECG_CHANNELS, CONTROL_ECG_MASK}:
        if n_comp == 0:
            return "PASS" if metrics["reason_code_frequencies"] else "NC"
        return "NC"

    if control == CONTROL_PPG_MASK and "ppg" in mod:
        if n_comp == 0 and INSUFFICIENT_COMMON_SUPPORT in metrics["reason_code_frequencies"]:
            return "PASS"
        if n_comp > 0:
            return "NC"
        return "PARTIAL"

    if n_comp == 0:
        return "NC"

    if metrics["composition_differs_from_baseline"]:
        return "PARTIAL"

    if control == CONTROL_ECG_CHANNELS and metrics["retained_channels"]["n_rows_with_retained_count"] == 0:
        return "PARTIAL"

    return "PASS"


def _run_checks(
    *,
    dataset_id: str,
    declared_modality: str,
    obs_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    metrics_by_control: dict[str, dict[str, Any]],
    figure3_uses_upstream: bool,
) -> dict[str, Any]:
    mod = declared_modality.casefold()
    ecg_only = "ecg" in mod and "ppg" not in mod

    ecg_channel = metrics_by_control[CONTROL_ECG_CHANNELS]
    ecg_template = metrics_by_control[CONTROL_ECG_TEMPLATE]

    ppg_controls = [CONTROL_PPG_MASK, CONTROL_PPG_TEMPLATE]
    silent_ppg = []
    if ecg_only:
        for control in ppg_controls:
            if metrics_by_control[control]["n_control_computable"] > 0:
                silent_ppg.append(control)

    ppg_mask_nc = [
        r
        for r in obs_rows
        if r.get("control_type") == CONTROL_PPG_MASK and not _as_bool(r.get("computable"))
    ]
    structural_ppg_mask = all(
        str(r.get("reason_code", "")) == INSUFFICIENT_COMMON_SUPPORT for r in ppg_mask_nc
    ) if ppg_mask_nc else True

    nc_zero_violations = sum(
        int(metrics_by_control[c]["nc_zero_violations"]) for c in REPORT_CONTROLS
    )

    composition_flags = {
        CONTROL_LABELS[c]: bool(metrics_by_control[c]["composition_differs_from_baseline"])
        for c in REPORT_CONTROLS
        if metrics_by_control[c]["n_control_computable"] > 0
    }

    zlpi_contract_ok = (
        int(metadata.get("duration_s", 0)) == EXPECTED_PRIMARY_DURATION_S
        and str(metadata.get("endpoint_name", "")).casefold() == ENDPOINT_ZLPI
        and str(metadata.get("power_representation", "")) == "absolute_log10"
        and metadata.get("lag_range_s") == [-60, 60]
    )

    return {
        "ecg_prone_genuinely_computed": ecg_channel["n_control_computable"] > 0,
        "ecg_template_genuinely_computed": ecg_template["n_control_computable"] > 0,
        "no_silent_ppg_on_ecg_declared": len(silent_ppg) == 0,
        "silent_ppg_controls_if_any": silent_ppg,
        "structural_ppg_mask_nc_preserved": structural_ppg_mask,
        "zlpi_contract_ok": zlpi_contract_ok,
        "composition_flags": composition_flags,
        "nc_zero_violations": nc_zero_violations,
        "figure3_reads_upstream_when_present": figure3_uses_upstream,
    }


def validate_dataset(role: str, dataset_id: str) -> dict[str, Any]:
    dataset_dir = DERIV_ROOT / role / dataset_id
    declared_modality = _read_protocol_modality(dataset_dir)
    figure3_upstream_exists = (dataset_dir / "C6" / "cardiac_controls_observation_level.csv").is_file()

    if not _stage_ready(dataset_dir):
        return {
            "role": role,
            "dataset_id": dataset_id,
            "declared_modality": declared_modality,
            "status": "not_ready",
            "blocker": "missing required C0/C1a/C1b/C1c/C3 stages",
            "grades": {CONTROL_LABELS[c]: "NC" for c in REPORT_CONTROLS},
        }

    aligned = dataset_dir / "C1c" / "features_confirmatory_aligned_D240.csv"
    if not aligned.is_file() or aligned.stat().st_size == 0:
        return {
            "role": role,
            "dataset_id": dataset_id,
            "declared_modality": declared_modality,
            "status": "not_ready",
            "blocker": "empty or missing D240 aligned table",
            "grades": {CONTROL_LABELS[c]: "NC" for c in REPORT_CONTROLS},
        }

    with tempfile.TemporaryDirectory(prefix=f"fig3d_val_{dataset_id}_") as tmp:
        result = run_confirmatory_cardiac_controls_upstream(
            c0_dir=dataset_dir / "C0",
            c1a_dir=dataset_dir / "C1a",
            c1b_dir=dataset_dir / "C1b",
            c1c_dir=dataset_dir / "C1c",
            c3_dir=dataset_dir / "C3",
            output_dir=Path(tmp),
            code_version="post_fix_validation",
        )

    obs_rows = [dict(r) for r in result.observation_rows]
    panel = compute_panel_d_from_observation_controls(
        obs_rows,
        dataset_qc_rows=[dict(r) for r in result.dataset_qc_rows],
        metadata=dict(result.metadata),
    )

    metrics_by_control = _aggregate_observations(obs_rows)
    checks = _run_checks(
        dataset_id=dataset_id,
        declared_modality=declared_modality,
        obs_rows=obs_rows,
        metadata=dict(result.metadata),
        metrics_by_control=metrics_by_control,
        figure3_uses_upstream=True,
    )

    grades: dict[str, str] = {}
    for control in REPORT_CONTROLS:
        grades[CONTROL_LABELS[control]] = _grade_control(
            role=role,
            dataset_id=dataset_id,
            declared_modality=declared_modality,
            control=control,
            metrics=metrics_by_control[control],
            checks=checks,
        )

    summary_lookup = {str(s.get("control")): s for s in panel.summaries}
    for control in REPORT_CONTROLS:
        summary = summary_lookup.get(control, {})
        metrics_by_control[control]["summary_n_baseline_eligible"] = summary.get("n_baseline_eligible")
        metrics_by_control[control]["summary_n_control_computable"] = summary.get("n_control_computable")
        metrics_by_control[control]["summary_n_paired_complete_case"] = summary.get("n_paired_complete_case")
        metrics_by_control[control]["summary_composition_differs"] = summary.get(
            "composition_differs_from_baseline"
        )

    return {
        "role": role,
        "dataset_id": dataset_id,
        "declared_modality": declared_modality,
        "status": "validated",
        "metadata_contract": {
            k: result.metadata.get(k)
            for k in ("duration_s", "endpoint_name", "power_representation", "lag_range_s", "flanks_s")
        },
        "checks": checks,
        "grades": grades,
        "controls": {
            CONTROL_LABELS[c]: metrics_by_control[c] for c in REPORT_CONTROLS
        },
        "existing_c6_upstream_file_present": figure3_upstream_exists,
    }


def main() -> int:
    out_path = ROOT / "derivatives" / "confirmatory_temporal_coupling" / "figure3d_cardiac_controls_validation.json"
    payload: list[dict[str, Any]] = []
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
    done = {(item.get("role"), item.get("dataset_id")) for item in payload}

    for role, dataset_id in DATASETS:
        key = (role, dataset_id)
        if key in done:
            print(f"Skipping {role}/{dataset_id} (already validated)", flush=True)
            continue
        print(f"Validating {role}/{dataset_id}...", flush=True)
        payload = [item for item in payload if (item.get("role"), item.get("dataset_id")) != key]
        payload.append(validate_dataset(role, dataset_id))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Updated {out_path} ({len(payload)} datasets)", flush=True)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
