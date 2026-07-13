"""Methods/results tables and machine-readable results bundle (M12)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .duration_contracts import (
    ENDPOINT_ZLPI,
    EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
    EXPECTED_PRIMARY_DURATION_S,
    ZLPI_FLANKS_S,
)
from .figures import resolve_reporting_inputs
from .manifest import (
    count_inclusions,
    sha256_file,
)

METHODS_SUMMARY_FILENAME = "methods_summary.csv"
RESULTS_SUMMARY_FILENAME = "results_summary.csv"
RESULTS_BUNDLE_FILENAME = "results_bundle.json"
VISUAL_REVIEW_CHECKLIST_FILENAME = "visual_review_checklist.csv"


@dataclass(frozen=True)
class ReportResult:
    methods_summary: Path
    results_summary: Path
    results_bundle: Path
    visual_review_checklist: Path


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
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                else:
                    payload[field] = value
            writer.writerow(payload)
    return path


def build_methods_summary(inputs: Mapping[str, Path | None]) -> list[dict[str, object]]:
    """Prespecified methods rows derived from contracts + available frozen tables."""
    rows = [
        {
            "section": "endpoint",
            "item": "primary_endpoint",
            "value": ENDPOINT_ZLPI,
            "source": "duration_contracts",
            "notes": "D120/D60 use MWPI/SWPI and remain separate.",
        },
        {
            "section": "endpoint",
            "item": "primary_duration_s",
            "value": EXPECTED_PRIMARY_DURATION_S,
            "source": "duration_contracts",
            "notes": "",
        },
        {
            "section": "endpoint",
            "item": "zlpi_flanks_s",
            "value": f"{ZLPI_FLANKS_S[0]}-{ZLPI_FLANKS_S[1]}",
            "source": "duration_contracts",
            "notes": "",
        },
        {
            "section": "endpoint",
            "item": "mu_equivalence_bounds_s",
            "value": f"±{EXPECTED_PEAK_CENTER_EQUIVALENCE_S}",
            "source": "duration_contracts",
            "notes": "TOST bounds for low-demand peak centers.",
        },
        {
            "section": "multiplicity",
            "item": "fdr_method",
            "value": "bh_fdr_0.05",
            "source": "inference",
            "notes": "Applied within prespecified families only.",
        },
        {
            "section": "sensitivity",
            "item": "can_rescue_primary",
            "value": False,
            "source": "artifact_controls",
            "notes": "Sensitivity/D180/D120/D60 cannot replace primary D240 ZLPI.",
        },
    ]
    for key, path in sorted(inputs.items()):
        rows.append(
            {
                "section": "frozen_inputs",
                "item": key,
                "value": str(path) if path is not None else "",
                "source": "confirmatory_root",
                "notes": "available" if path is not None else "missing",
            }
        )
    return rows


def build_results_summary(inputs: Mapping[str, Path | None]) -> list[dict[str, object]]:
    """Results table populated only from frozen confirmatory outputs."""
    rows: list[dict[str, object]] = []
    meta = read_csv_rows(inputs.get("meta_analysis"))
    for row in meta:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        if _as_int(row.get("duration_s"), 240) != 240:
            continue
        rows.append(
            {
                "family": "meta_analysis",
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": 240,
                "band": _as_str(row.get("band")),
                "estimate": _as_float(row.get("pooled_effect")),
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n_datasets")),
                "p_value": _as_float(row.get("p_value")),
                "source_table": str(inputs.get("meta_analysis") or ""),
                "notes": _as_str(row.get("notes")),
            }
        )

    effects = read_csv_rows(inputs.get("dataset_effects"))
    for row in effects:
        if str(row.get("is_primary_analysis", "")).lower() not in {"true", "1", "yes"}:
            continue
        rows.append(
            {
                "family": "dataset_effect",
                "endpoint_name": _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
                "duration_s": _as_int(row.get("duration_s"), 240),
                "band": _as_str(row.get("band")),
                "estimate": _as_float(row.get("effect_mean")),
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n_pairs")),
                "p_value": _as_float(row.get("p_value")),
                "source_table": str(inputs.get("dataset_effects") or ""),
                "notes": f"dataset={_as_str(row.get('dataset_id'))}; contrast={_as_str(row.get('contrast_id'))}",
            }
        )

    equivalence = read_csv_rows(inputs.get("peak_equivalence"))
    for row in equivalence:
        rows.append(
            {
                "family": "mu_equivalence",
                "endpoint_name": _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI),
                "duration_s": _as_int(row.get("duration_s"), 240),
                "band": _as_str(row.get("band")),
                "estimate": _as_float(row.get("mean_mu")),
                "ci_low": _as_float(row.get("ci_low")),
                "ci_high": _as_float(row.get("ci_high")),
                "n": _as_int(row.get("n")),
                "p_value": _as_float(row.get("tost_p")),
                "source_table": str(inputs.get("peak_equivalence") or ""),
                "notes": f"equivalent={_as_str(row.get('equivalent'))}",
            }
        )

    nulls = read_csv_rows(inputs.get("null_summary"))
    for row in nulls:
        if _as_str(row.get("endpoint_name"), ENDPOINT_ZLPI) != ENDPOINT_ZLPI:
            continue
        rows.append(
            {
                "family": "null_summary",
                "endpoint_name": ENDPOINT_ZLPI,
                "duration_s": _as_int(row.get("duration_s"), 240),
                "band": _as_str(row.get("band")),
                "estimate": _as_float(row.get("median_observed_endpoint_index")),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": _as_int(row.get("n_units")),
                "p_value": _as_float(row.get("median_empirical_p")),
                "source_table": str(inputs.get("null_summary") or ""),
                "notes": f"null_type={_as_str(row.get('null_type'))}",
            }
        )
    return rows


def build_results_bundle(
    inputs: Mapping[str, Path | None],
    *,
    methods_rows: Sequence[Mapping[str, object]],
    results_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Machine-readable bundle with hashed provenance — no hard-coded claims."""
    input_hashes = {}
    for key, path in inputs.items():
        if path is not None and path.is_file():
            input_hashes[key] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    return {
        "schema_version": "confirmatory_results_bundle_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": ENDPOINT_ZLPI,
        "primary_duration_s": EXPECTED_PRIMARY_DURATION_S,
        "mu_equivalence_bounds_s": EXPECTED_PEAK_CENTER_EQUIVALENCE_S,
        "input_tables": input_hashes,
        "methods_summary": list(methods_rows),
        "results_summary": list(results_rows),
        "notes": [
            "All numeric results are copied from frozen confirmatory outputs.",
            "ZLPI, MWPI, and SWPI are never pooled.",
            "Sensitivity analyses cannot rescue primary D240 absolute-power ZLPI.",
        ],
    }


def build_visual_review_checklist() -> list[dict[str, object]]:
    return [
        {
            "item_id": "fig1_lag_sign",
            "figure_id": "figure1",
            "check": "Lag axis uses corr(HR(t), EEG(t+τ)); +τ = EEG follows HR",
            "status": "pending_review",
        },
        {
            "item_id": "fig1_flanks",
            "figure_id": "figure1",
            "check": "ZLPI flank shading matches 20–60 s windows",
            "status": "pending_review",
        },
        {
            "item_id": "fig1_ci",
            "figure_id": "figure1",
            "check": "Mean Fisher-z curves display 95% CI ribbons and participant n",
            "status": "pending_review",
        },
        {
            "item_id": "fig2_paired",
            "figure_id": "figure2",
            "check": "Paired attenuation panels report dataset participant counts",
            "status": "pending_review",
        },
        {
            "item_id": "fig2_equivalence",
            "figure_id": "figure2",
            "check": "±2 s peak-center equivalence region is visible",
            "status": "pending_review",
        },
        {
            "item_id": "fig2_d180",
            "figure_id": "figure2",
            "check": "D180 panel labeled sensitivity-only / cannot rescue primary",
            "status": "pending_review",
        },
        {
            "item_id": "fig3_endpoint_sep",
            "figure_id": "figure3",
            "check": "ZLPI/MWPI/SWPI markers remain visually distinct",
            "status": "pending_review",
        },
        {
            "item_id": "fig3_nulls",
            "figure_id": "figure3",
            "check": "Surrogate/null markers shown against observed endpoints",
            "status": "pending_review",
        },
        {
            "item_id": "traceability",
            "figure_id": "all",
            "check": "Each panel listed in figure_source_manifest.json with input hashes",
            "status": "pending_review",
        },
    ]


def generate_confirmatory_report(
    confirmatory_root: str | Path,
    output_dir: str | Path,
) -> ReportResult:
    """Write methods/results summaries, results bundle, and review checklist."""
    root = Path(confirmatory_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = resolve_reporting_inputs(root)

    methods_rows = build_methods_summary(inputs)
    results_rows = build_results_summary(inputs)
    bundle = build_results_bundle(
        inputs, methods_rows=methods_rows, results_rows=results_rows
    )
    # Attach inclusion counts when eligibility exists.
    bundle["inclusion_counts"] = {
        "eligibility_by_status": count_inclusions(
            inputs.get("eligibility"), key_column="status"
        )
    }

    methods_path = _write_csv(
        out / METHODS_SUMMARY_FILENAME,
        methods_rows,
        ("section", "item", "value", "source", "notes"),
    )
    results_path = _write_csv(
        out / RESULTS_SUMMARY_FILENAME,
        results_rows,
        (
            "family",
            "endpoint_name",
            "duration_s",
            "band",
            "estimate",
            "ci_low",
            "ci_high",
            "n",
            "p_value",
            "source_table",
            "notes",
        ),
    )
    checklist_path = _write_csv(
        out / VISUAL_REVIEW_CHECKLIST_FILENAME,
        build_visual_review_checklist(),
        ("item_id", "figure_id", "check", "status"),
    )
    bundle_path = out / RESULTS_BUNDLE_FILENAME
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReportResult(
        methods_summary=methods_path,
        results_summary=results_path,
        results_bundle=bundle_path,
        visual_review_checklist=checklist_path,
    )


def run_confirmatory_reporting(
    confirmatory_root: str | Path,
    output_dir: str | Path,
    *,
    config_paths: Sequence[str | Path] = (),
    repo_root: str | Path | None = None,
    seeds: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Generate figures, reports, and manifests under ``output_dir``."""
    from .figures import generate_confirmatory_figures
    from .manifest import build_run_manifest, hash_directory_files

    root = Path(confirmatory_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    figures_dir = out / "figures"
    reports_dir = out / "reports"
    manifests_dir = out / "manifests"

    figures = generate_confirmatory_figures(root, figures_dir)
    report = generate_confirmatory_report(root, reports_dir)
    inputs = resolve_reporting_inputs(root)
    inclusion = {
        "eligibility_by_status": count_inclusions(
            inputs.get("eligibility"), key_column="status"
        ),
        "subject_level_rows": len(read_csv_rows(inputs.get("subject_level"))),
        "paired_contrast_rows": len(read_csv_rows(inputs.get("paired_contrasts"))),
    }
    run_manifest = build_run_manifest(
        output_dir=manifests_dir,
        confirmatory_root=root,
        config_paths=config_paths,
        repo_root=repo_root,
        seeds=seeds,
        inclusion_counts=inclusion,
        input_hashes=hash_directory_files(root),
        extra={
            "figure_source_manifest": str(figures.figure_source_manifest),
            "results_bundle": str(report.results_bundle),
        },
    )
    return {
        "figure1_pdf": figures.figure1.pdf,
        "figure1_svg": figures.figure1.svg,
        "figure1_png": figures.figure1.png,
        "figure2_pdf": figures.figure2.pdf,
        "figure2_svg": figures.figure2.svg,
        "figure2_png": figures.figure2.png,
        "figure3_pdf": figures.figure3.pdf,
        "figure3_svg": figures.figure3.svg,
        "figure3_png": figures.figure3.png,
        "figure_source_manifest": figures.figure_source_manifest,
        "methods_summary": report.methods_summary,
        "results_summary": report.results_summary,
        "results_bundle": report.results_bundle,
        "visual_review_checklist": report.visual_review_checklist,
        "run_manifest": Path(run_manifest["manifest_path"]),
    }


__all__ = [
    "METHODS_SUMMARY_FILENAME",
    "RESULTS_BUNDLE_FILENAME",
    "RESULTS_SUMMARY_FILENAME",
    "VISUAL_REVIEW_CHECKLIST_FILENAME",
    "ReportResult",
    "build_methods_summary",
    "build_results_bundle",
    "build_results_summary",
    "generate_confirmatory_report",
    "run_confirmatory_reporting",
]
