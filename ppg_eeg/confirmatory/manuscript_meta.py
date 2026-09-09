"""Build merged manuscript primary Figures 1/2/3 artifacts.

This module creates a canonical merged tree:

    primary/merged/C5 -> primary/merged/C6 -> primary/merged/C7

from already-computed per-dataset primary trees, without rerunning raw stages.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np

from .correlation import CURVES_TEMPLATE
from .endpoints import METRICS_TEMPLATE, QC_TEMPLATE
from .figure1_panels import render_figure1
from .figure2_panels import render_figure2
from .figures import (
    render_figure3,
    render_figure3_panel_e,
    render_figure3_panel_f,
    resolve_reporting_inputs,
    save_figure_trio,
    write_source_csv,
)
from .panel_f_topography_gamma import PANEL_F_STEM
from .panel_f_topography_upstream import PANEL_F_C6_ARTIFACTS, run_confirmatory_panel_f_upstream
from .group_tables import (
    PAIRED_CONTRASTS_FILENAME,
    PAIRING_QC_FILENAME,
    SUBJECT_LEVEL_FILENAME,
)
from .inference import (
    DATASET_EFFECTS_FILENAME,
    LEAVE_ONE_DATASET_OUT_FILENAME,
    LOO_FIELDS,
    LOW_DEMAND_ALPHA_EFFECTS_FILENAME,
    LOW_DEMAND_ALPHA_EFFECT_FIELDS,
    LOW_DEMAND_ALPHA_LOO_FILENAME,
    LOW_DEMAND_ALPHA_META_FIELDS,
    LOW_DEMAND_ALPHA_META_FILENAME,
    META_ANALYSIS_RESULTS_FILENAME,
    MIXED_MODEL_RESULTS_FILENAME,
    PEAK_CENTER_EQUIVALENCE_FILENAME,
    low_demand_alpha_replication,
    read_csv_rows,
    run_confirmatory_inference_from_dir,
)
from .peak_model import PARAMS_FILENAME
from .protocol_audit import (
    ELIGIBILITY_BY_DURATION_FILENAME,
    ELIGIBILITY_QC_SUMMARY_FILENAME,
    PROTOCOL_AUDIT_FILENAME,
)
from .dataset_roles import ROLE_PRIMARY, is_runtime_blocked, resolve_dataset_role

PRIMARY_MANUSCRIPT_DATASETS: tuple[str, ...] = ("ds003690", "ds003838", "ds006848")
PRIMARY_MANUSCRIPT_DATASET_SET = frozenset(ds.casefold() for ds in PRIMARY_MANUSCRIPT_DATASETS)
QUARANTINED_PRIMARY_DATASETS = frozenset({"ds004587"})
PRIMARY_META_EXPECTED: frozenset[tuple[str, str]] = frozenset(
    {
        ("ds003690", "passive__gonogo"),
        ("ds003838", "rest__memory"),
        ("ds006848", "rest__verbalwm"),
    }
)
ALPHA_BAND = "alpha"
PRIMARY_REPRESENTATION = "absolute_log10"
PRIMARY_ENDPOINT = "zlpi"
PRIMARY_DURATION_S = 240
PANEL_E_STEM = "figure3_panel_e_nuisance_modality"
PANEL_F_COMPOSITE_STEM = PANEL_F_STEM
FIGURE3_MAIN_STEM = "figure3_temporal_artifact_specificity"


@dataclass(frozen=True)
class MergedManuscriptResult:
    merged_root: Path
    c5_dir: Path
    c6_dir: Path
    c7_dir: Path
    c7_publish_dir: Path
    figure2_png: Path
    figure2_source_csv: Path
    figure1_png: Path | None
    figure1_source_csv: Path | None
    summary_json: Path


@dataclass(frozen=True)
class MergedFigure3ManuscriptResult:
    merged_root: Path
    c4_dir: Path
    c5_dir: Path
    c6_dir: Path
    c7_dir: Path
    c7_publish_dir: Path
    figure3_main_png: Path
    figure3_panel_e_png: Path
    figure3_panel_f_png: Path
    summary_json: Path


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fields, rows


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload: dict[str, object] = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and not math.isfinite(value):
                    payload[field] = ""
                elif isinstance(value, bool):
                    payload[field] = str(value)
                else:
                    payload[field] = value
            writer.writerow(payload)


def _merge_csv(
    sources: Sequence[Path],
    destination: Path,
    *,
    required: bool = True,
) -> int:
    merged_fields: list[str] = []
    merged_rows: list[dict[str, str]] = []
    for src in sources:
        if not src.is_file():
            if required:
                raise FileNotFoundError(f"Missing required merged source CSV: {src}")
            continue
        fields, rows = _read_csv_rows(src)
        if not merged_fields:
            merged_fields = list(fields)
        else:
            for field in fields:
                if field not in merged_fields:
                    merged_fields.append(field)
        merged_rows.extend(rows)
    if not merged_fields:
        raise FileNotFoundError(
            f"No CSV sources available to merge for {destination.name}: {sources}"
        )
    _write_csv_rows(destination, merged_fields, merged_rows)
    return len(merged_rows)


def _merge_csv_streaming(
    sources: Sequence[Path],
    destination: Path,
    *,
    required: bool = True,
) -> int:
    fields: list[str] = []
    existing_sources: list[Path] = []
    for src in sources:
        if not src.is_file():
            if required:
                raise FileNotFoundError(f"Missing required merged source CSV: {src}")
            continue
        existing_sources.append(src)
        with src.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in list(reader.fieldnames or []):
                if field not in fields:
                    fields.append(field)
    if not existing_sources or not fields:
        raise FileNotFoundError(
            f"No CSV sources available to merge for {destination.name}: {sources}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for src in existing_sources:
            with src.open(encoding="utf-8", newline="") as src_handle:
                reader = csv.DictReader(src_handle)
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in fields})
                    row_count += 1
    return row_count


def _dataset_stage_dir(confirmatory_root: Path, dataset_id: str, stage: str) -> Path:
    key = str(dataset_id).strip().casefold()
    if key in QUARANTINED_PRIMARY_DATASETS:
        marker = confirmatory_root / "primary" / key / "STALE_INVALIDATED.txt"
        if marker.is_file():
            raise ValueError(
                f"Quarantined stale primary tree cannot be resolved for manuscript assembly: {key}"
            )
    return confirmatory_root / "primary" / dataset_id / stage


def _dataset_stage_file(
    confirmatory_root: Path,
    dataset_id: str,
    stage: str,
    filename: str,
) -> Path | None:
    key = str(dataset_id).strip().casefold()
    if key not in QUARANTINED_PRIMARY_DATASETS:
        primary = confirmatory_root / "primary" / dataset_id / stage / filename
        if primary.is_file():
            return primary
    sensitivity = confirmatory_root / "sensitivity" / dataset_id / stage / filename
    if sensitivity.is_file():
        return sensitivity
    return None


def _filter_csv_to_primary_dataset_rows(path: Path) -> int:
    fields, rows = _read_csv_rows(path)
    if not fields:
        return 0
    if "dataset_id" not in fields:
        return len(rows)
    filtered: list[dict[str, str]] = []
    for row in rows:
        ds = str(row.get("dataset_id", "")).strip().casefold()
        if ds not in PRIMARY_MANUSCRIPT_DATASET_SET:
            continue
        if "dataset_role" in fields:
            row["dataset_role"] = ROLE_PRIMARY
        filtered.append(row)
    _write_csv_rows(path, fields, filtered)
    return len(filtered)


def _refresh_global_protocol_audit_roles(confirmatory_root: Path) -> dict[str, object]:
    audit_path = confirmatory_root / "audit" / PROTOCOL_AUDIT_FILENAME
    if not audit_path.is_file():
        return {"path": str(audit_path), "updated": False, "reason": "missing"}
    fields, rows = _read_csv_rows(audit_path)
    if "dataset_id" not in fields or "dataset_role" not in fields:
        return {
            "path": str(audit_path),
            "updated": False,
            "reason": "missing_dataset_role_columns",
        }
    updated = 0
    for row in rows:
        ds = str(row.get("dataset_id", "")).strip().casefold()
        if not ds:
            continue
        canonical = resolve_dataset_role(ds)
        if str(row.get("dataset_role", "")).strip().casefold() != canonical:
            row["dataset_role"] = canonical
            updated += 1
    _write_csv_rows(audit_path, fields, rows)
    return {
        "path": str(audit_path),
        "updated": bool(updated),
        "updated_rows": updated,
        "rows": len(rows),
    }


def _copy_json_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _merge_dataset_metadata_json(
    sources_by_dataset: Mapping[str, Path],
    destination: Path,
) -> int:
    payload: dict[str, object] = {
        "schema_version": "merged_cardiac_controls_metadata_v1",
        "source_stage": "C6",
        "primary_datasets": list(PRIMARY_MANUSCRIPT_DATASETS),
        "datasets": {},
    }
    datasets_payload = payload["datasets"]
    assert isinstance(datasets_payload, dict)
    count = 0
    for dataset_id in PRIMARY_MANUSCRIPT_DATASETS:
        source = sources_by_dataset[dataset_id]
        if not source.is_file():
            raise FileNotFoundError(f"Missing required merged source JSON: {source}")
        try:
            meta = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {source}: {exc}") from exc
        datasets_payload[dataset_id] = meta
        count += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return count


def _merge_primary_c5(confirmatory_root: Path, merged_c5_dir: Path) -> dict[str, int]:
    merged_c5_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for filename in (SUBJECT_LEVEL_FILENAME, PAIRED_CONTRASTS_FILENAME, PAIRING_QC_FILENAME):
        sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C5") / filename
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        row_counts[filename] = _merge_csv(sources, merged_c5_dir / filename, required=True)
    return row_counts


def _merge_primary_upstream_tables(confirmatory_root: Path, merged_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    c0 = merged_root / "C0"
    c2 = merged_root / "C2"
    c3 = merged_root / "C3"
    c4 = merged_root / "C4"
    for folder in (c0, c2, c3, c4):
        folder.mkdir(parents=True, exist_ok=True)

    # C0 tables used by figure annotations and reporting manifests.
    for filename in (
        PROTOCOL_AUDIT_FILENAME,
        ELIGIBILITY_BY_DURATION_FILENAME,
        ELIGIBILITY_QC_SUMMARY_FILENAME,
    ):
        sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C0") / filename
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        counts[f"C0/{filename}"] = _merge_csv(sources, c0 / filename, required=True)
        counts[f"C0/{filename}"] = _filter_csv_to_primary_dataset_rows(c0 / filename)

    # C2 lag curves used by Figure 2 A/B reconstruction.
    for duration in (240, 180, 120, 60):
        filename = CURVES_TEMPLATE.format(duration_s=duration)
        sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C2") / filename
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        counts[f"C2/{filename}"] = _merge_csv(sources, c2 / filename, required=True)

    # C3 peak params are consumed by Figure 2E.
    counts[f"C3/{PARAMS_FILENAME}"] = _merge_csv(
        [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C3") / PARAMS_FILENAME
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ],
        c3 / PARAMS_FILENAME,
        required=True,
    )

    # Also merge endpoint tables for completeness.
    for duration in (240, 180, 120, 60):
        metric_name = METRICS_TEMPLATE.format(duration_s=duration)
        qc_name = QC_TEMPLATE.format(duration_s=duration)
        metric_sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C3") / metric_name
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        qc_sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C3") / qc_name
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        counts[f"C3/{metric_name}"] = _merge_csv(metric_sources, c3 / metric_name, required=True)
        counts[f"C3/{qc_name}"] = _merge_csv(qc_sources, c3 / qc_name, required=True)

    # Optional C4 tables can be carried through when available.
    for filename in ("null_subject_results.csv", "null_summary.csv", "null_qc.csv"):
        sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C4") / filename
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        try:
            counts[f"C4/{filename}"] = _merge_csv(sources, c4 / filename, required=False)
        except FileNotFoundError:
            continue
    surrogate_sources = [
        _dataset_stage_dir(confirmatory_root, dataset_id, "C4") / "null_surrogate_values.csv"
        for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
    ]
    try:
        counts["C4/null_surrogate_values.csv"] = _merge_csv_streaming(
            surrogate_sources,
            c4 / "null_surrogate_values.csv",
            required=False,
        )
    except FileNotFoundError:
        pass
    return counts


def _ensure_primary_panel_f_upstream(confirmatory_root: Path) -> dict[str, bool]:
    """Regenerate missing C6 Panel F upstream tables from existing derivatives only."""
    repaired: dict[str, bool] = {}
    required_files = tuple(PANEL_F_C6_ARTIFACTS.values())
    for dataset_id in PRIMARY_MANUSCRIPT_DATASETS:
        c6_dir = _dataset_stage_dir(confirmatory_root, dataset_id, "C6")
        missing = [name for name in required_files if not (c6_dir / name).is_file()]
        if not missing:
            repaired[dataset_id] = False
            continue
        run_confirmatory_panel_f_upstream(
            c0_dir=_dataset_stage_dir(confirmatory_root, dataset_id, "C0"),
            c5_dir=_dataset_stage_dir(confirmatory_root, dataset_id, "C5"),
            output_dir=c6_dir,
            confirmatory_root=_dataset_stage_dir(confirmatory_root, dataset_id, "C6").parent,
            force_recompute_channel_zlpi=False,
        )
        repaired[dataset_id] = True
    return repaired


def _merge_primary_figure3_c6_inputs(confirmatory_root: Path, merged_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    c6 = merged_root / "C6"
    c6.mkdir(parents=True, exist_ok=True)
    for filename in (
        "duration_sensitivity.csv",
        "cardiac_controls_observation_level.csv",
        "cardiac_controls_dataset_qc.csv",
    ):
        sources = [
            _dataset_stage_dir(confirmatory_root, dataset_id, "C6") / filename
            for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
        ]
        if filename == "cardiac_controls_observation_level.csv":
            counts[f"C6/{filename}"] = _merge_csv_streaming(sources, c6 / filename, required=True)
        else:
            counts[f"C6/{filename}"] = _merge_csv(sources, c6 / filename, required=True)
    metadata_sources = {
        dataset_id: _dataset_stage_dir(confirmatory_root, dataset_id, "C6")
        / "cardiac_controls_metadata.json"
        for dataset_id in PRIMARY_MANUSCRIPT_DATASETS
    }
    counts["C6/cardiac_controls_metadata.json"] = _merge_dataset_metadata_json(
        metadata_sources,
        c6 / "cardiac_controls_metadata.json",
    )
    # Panel C sensitivity overlay: include ds003816 D60-only rows when present.
    ds003816_path = _dataset_stage_file(
        confirmatory_root,
        "ds003816",
        "C6",
        "duration_sensitivity.csv",
    )
    if ds003816_path is not None:
        fields, rows = _read_csv_rows(c6 / "duration_sensitivity.csv")
        _fields_sens, sens_rows = _read_csv_rows(ds003816_path)
        rows.extend(sens_rows)
        _write_csv_rows(c6 / "duration_sensitivity.csv", fields, rows)
        counts["C6/duration_sensitivity.csv"] = len(rows)

    # Keep dataset-specific Panel E/F upstream artifacts in a canonical merged C6 tree.
    panel_e_root = c6 / "panel_e_by_dataset"
    panel_f_root = c6 / "panel_f_by_dataset"
    for dataset_id in PRIMARY_MANUSCRIPT_DATASETS:
        src_dir = _dataset_stage_dir(confirmatory_root, dataset_id, "C6")
        dst_e = panel_e_root / dataset_id
        dst_f = panel_f_root / dataset_id
        dst_e.mkdir(parents=True, exist_ok=True)
        dst_f.mkdir(parents=True, exist_ok=True)
        for path in src_dir.glob("figure3_panel_e_nuisance_modality_*"):
            if path.is_file():
                shutil.copy2(path, dst_e / path.name)
        for path in src_dir.glob(f"{PANEL_F_STEM}_*"):
            if path.is_file():
                shutil.copy2(path, dst_f / path.name)
    return counts


def _write_low_demand_alpha_replication(c5_dir: Path, c6_dir: Path) -> dict[str, int]:
    """Compute and persist merged Figure 1 low-demand alpha replication tables."""
    subject_rows = read_csv_rows(c5_dir / SUBJECT_LEVEL_FILENAME)
    effects, meta_row, loo_rows = low_demand_alpha_replication(
        subject_rows,
        primary_dataset_ids=PRIMARY_MANUSCRIPT_DATASETS,
    )
    _write_csv_rows(c6_dir / LOW_DEMAND_ALPHA_EFFECTS_FILENAME, LOW_DEMAND_ALPHA_EFFECT_FIELDS, effects)
    _write_csv_rows(c6_dir / LOW_DEMAND_ALPHA_META_FILENAME, LOW_DEMAND_ALPHA_META_FIELDS, [meta_row])
    _write_csv_rows(c6_dir / LOW_DEMAND_ALPHA_LOO_FILENAME, LOO_FIELDS, loo_rows)
    return {
        LOW_DEMAND_ALPHA_EFFECTS_FILENAME: len(effects),
        LOW_DEMAND_ALPHA_META_FILENAME: 1,
        LOW_DEMAND_ALPHA_LOO_FILENAME: len(loo_rows),
    }


def _write_merged_publish_tree(merged_root: Path) -> None:
    publish = merged_root / "C7" / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    for stage, filenames in (
        (
            "C0",
            (
                PROTOCOL_AUDIT_FILENAME,
                ELIGIBILITY_BY_DURATION_FILENAME,
                ELIGIBILITY_QC_SUMMARY_FILENAME,
            ),
        ),
        (
            "C2",
            tuple(CURVES_TEMPLATE.format(duration_s=d) for d in (240, 180, 120, 60)),
        ),
        (
            "C3",
            (PARAMS_FILENAME, *[METRICS_TEMPLATE.format(duration_s=d) for d in (240, 180, 120, 60)]),
        ),
        (
            "C4",
            (
                "null_subject_results.csv",
                "null_summary.csv",
                "null_qc.csv",
                "null_surrogate_values.csv",
            ),
        ),
        (
            "C5",
            (SUBJECT_LEVEL_FILENAME, PAIRED_CONTRASTS_FILENAME, PAIRING_QC_FILENAME),
        ),
        (
            "C6",
            (
                DATASET_EFFECTS_FILENAME,
                META_ANALYSIS_RESULTS_FILENAME,
                LEAVE_ONE_DATASET_OUT_FILENAME,
                MIXED_MODEL_RESULTS_FILENAME,
                PEAK_CENTER_EQUIVALENCE_FILENAME,
                "mixed_model_marginal_estimates.csv",
                "mixed_model_contrasts.csv",
                "peak_hierarchical_summaries.csv",
                "multiplicity_results.csv",
                "inference_qc.csv",
                "duration_sensitivity.csv",
                "cardiac_controls_observation_level.csv",
                "cardiac_controls_dataset_qc.csv",
                "cardiac_controls_metadata.json",
                LOW_DEMAND_ALPHA_EFFECTS_FILENAME,
                LOW_DEMAND_ALPHA_META_FILENAME,
                LOW_DEMAND_ALPHA_LOO_FILENAME,
            ),
        ),
    ):
        stage_dir = merged_root / stage
        for filename in filenames:
            src = stage_dir / filename
            if src.is_file():
                shutil.copy2(src, publish / filename)
    _copy_json_if_present(merged_root / "C6" / "panel_f_topography_gamma_metadata.json", publish / "panel_f_topography_gamma_metadata.json")
    for filename in (
        PROTOCOL_AUDIT_FILENAME,
        ELIGIBILITY_BY_DURATION_FILENAME,
        ELIGIBILITY_QC_SUMMARY_FILENAME,
    ):
        publish_file = publish / filename
        if publish_file.is_file():
            _filter_csv_to_primary_dataset_rows(publish_file)


def _assert_dataset_ids_primary_only(
    path: Path,
    *,
    allow_special_dataset_ids: frozenset[str] = frozenset(),
) -> dict[str, int]:
    if not path.is_file():
        return {"rows": 0, "non_primary_rows": 0}
    fields, rows = _read_csv_rows(path)
    if "dataset_id" not in fields:
        return {"rows": len(rows), "non_primary_rows": 0}
    bad = [
        row
        for row in rows
        if str(row.get("dataset_id", "")).strip()
        and str(row.get("dataset_id", "")).strip().casefold()
        not in PRIMARY_MANUSCRIPT_DATASET_SET.union(allow_special_dataset_ids)
    ]
    if bad:
        bad_ids = sorted({str(r.get("dataset_id", "")).strip().casefold() for r in bad})
        raise ValueError(f"{path.name} contains non-primary dataset rows: {bad_ids}")
    return {"rows": len(rows), "non_primary_rows": 0}


def _assert_manuscript_role_integrity(
    confirmatory_root: Path,
    merged_root: Path,
    *,
    include_figure3_checks: bool,
) -> dict[str, object]:
    c0_dir = merged_root / "C0"
    publish_dir = merged_root / "C7" / "publish"
    source_dir = merged_root / "C7" / "figures" / "source_data"
    checked: dict[str, dict[str, int]] = {}

    for folder in (c0_dir, publish_dir):
        for filename in (
            PROTOCOL_AUDIT_FILENAME,
            ELIGIBILITY_BY_DURATION_FILENAME,
            ELIGIBILITY_QC_SUMMARY_FILENAME,
        ):
            path = folder / filename
            checked[str(path.relative_to(merged_root))] = _assert_dataset_ids_primary_only(path)
            fields, rows = _read_csv_rows(path)
            if filename == PROTOCOL_AUDIT_FILENAME and "dataset_role" in fields:
                for row in rows:
                    ds = str(row.get("dataset_id", "")).strip().casefold()
                    if not ds:
                        continue
                    canonical = resolve_dataset_role(ds)
                    if canonical != ROLE_PRIMARY:
                        raise ValueError(
                            f"{path.name} includes non-primary dataset role row: {ds}={canonical}"
                        )
                    role = str(row.get("dataset_role", "")).strip().casefold()
                    if role != canonical:
                        raise ValueError(
                            f"{path.name} role mismatch for {ds}: row={role!r} canonical={canonical!r}"
                        )

    figure_checks = [
        source_dir / "figure1_panel_e_alpha_replication_forest.csv",
        source_dir / "figure2_panel_c_alpha_meta_forest.csv",
    ]
    if include_figure3_checks:
        figure_checks.extend(
            (
                source_dir / "figure3_panel_a_independent_unit_verdict.csv",
                source_dir / "figure3_panel_c_duration_sensitivity.csv",
            )
        )
    for figure_file in figure_checks:
        allow = frozenset({"pooled"}) if "forest" in figure_file.name else frozenset()
        checked[str(figure_file.relative_to(merged_root))] = _assert_dataset_ids_primary_only(
            figure_file,
            allow_special_dataset_ids=allow,
        )

    fields, effect_rows = _read_csv_rows(merged_root / "C6" / DATASET_EFFECTS_FILENAME)
    if "enters_meta" in fields and "dataset_id" in fields:
        for row in effect_rows:
            enters = str(row.get("enters_meta", "")).strip().casefold() in {"true", "1", "yes"}
            if not enters:
                continue
            ds = str(row.get("dataset_id", "")).strip().casefold()
            if ds not in PRIMARY_MANUSCRIPT_DATASET_SET:
                raise ValueError(f"enters_meta leak outside primary set: {ds}")

    blocked_hits: set[str] = set()
    for path in (
        source_dir / "figure1_panel_d_zlpi_heatmap.csv",
        source_dir / "figure1_panel_e_alpha_replication_forest.csv",
        source_dir / "figure2_panel_c_alpha_meta_forest.csv",
        source_dir / "figure3_panel_c_duration_sensitivity.csv",
    ):
        if not path.is_file():
            continue
        fields, rows = _read_csv_rows(path)
        if "dataset_id" not in fields:
            continue
        for row in rows:
            ds = str(row.get("dataset_id", "")).strip().casefold()
            if ds and is_runtime_blocked(ds):
                blocked_hits.add(ds)
    if blocked_hits:
        raise ValueError(f"Blocked datasets appeared in manuscript-facing claim rows: {sorted(blocked_hits)}")

    stale_marker = confirmatory_root / "primary" / "ds004587" / "STALE_INVALIDATED.txt"
    if stale_marker.is_file():
        resolved = _dataset_stage_file(
            confirmatory_root,
            "ds004587",
            "C0",
            PROTOCOL_AUDIT_FILENAME,
        )
        if resolved is not None and "primary/ds004587" in str(resolved):
            raise ValueError("Quarantined primary/ds004587 was resolved by stage file lookup.")

    return {
        "checked_files": checked,
        "primary_datasets": list(PRIMARY_MANUSCRIPT_DATASETS),
    }


def _assert_figure2_meta_integrity(merged_root: Path) -> dict[str, object]:
    c6_dir = merged_root / "C6"
    c7_source = merged_root / "C7" / "figures" / "source_data" / "figure2_panel_c_alpha_meta_forest.csv"
    loo_path = c6_dir / LEAVE_ONE_DATASET_OUT_FILENAME
    effect_path = c6_dir / DATASET_EFFECTS_FILENAME

    _fields, forest_rows = _read_csv_rows(c7_source)
    primary_rows = [row for row in forest_rows if str(row.get("row_type", "")).casefold() == "primary"]
    pooled_rows = [row for row in forest_rows if str(row.get("row_type", "")).casefold() == "pooled"]
    hiit_rows = [row for row in forest_rows if str(row.get("dataset_id", "")).casefold() == "hiit"]

    found_pairs = {
        (
            str(row.get("dataset_id", "")).casefold(),
            str(row.get("contrast_id", "")).casefold(),
        )
        for row in primary_rows
    }
    if found_pairs != PRIMARY_META_EXPECTED:
        raise ValueError(
            f"Figure 2C primary rows mismatch: expected {sorted(PRIMARY_META_EXPECTED)} got {sorted(found_pairs)}"
        )
    if len(primary_rows) != 3:
        raise ValueError(f"Figure 2C requires exactly 3 primary rows; got {len(primary_rows)}")
    if len(pooled_rows) != 1:
        raise ValueError(f"Figure 2C requires exactly 1 pooled row; got {len(pooled_rows)}")
    pooled = pooled_rows[0]
    if int(float(str(pooled.get("n_datasets") or "0"))) != 3:
        raise ValueError(
            f"Figure 2C pooled row must have n_datasets=3; got {pooled.get('n_datasets')!r}"
        )

    if hiit_rows:
        for row in hiit_rows:
            if str(row.get("row_type", "")).casefold() != "sensitivity_display":
                raise ValueError("HIIT row appeared outside sensitivity_display.")
            if str(row.get("enters_meta", "")).strip().casefold() in {"true", "1", "yes"}:
                raise ValueError("HIIT row incorrectly marked enters_meta.")

    _fields, loo_rows = _read_csv_rows(loo_path)
    loo_alpha = [
        row
        for row in loo_rows
        if str(row.get("endpoint_name", "")).casefold() == PRIMARY_ENDPOINT
        and str(row.get("band", "")).casefold() == ALPHA_BAND
        and int(float(str(row.get("duration_s") or "0"))) == PRIMARY_DURATION_S
        and str(row.get("power_representation", "")).casefold() == PRIMARY_REPRESENTATION
        and str(row.get("analysis_status", "")).casefold() == "completed"
    ]
    if len(loo_alpha) != 3:
        raise ValueError(f"Expected 3 alpha LOO omission rows; got {len(loo_alpha)}")

    _fields, effect_rows = _read_csv_rows(effect_path)
    alpha_meta_rows = [
        row
        for row in effect_rows
        if str(row.get("band", "")).casefold() == ALPHA_BAND
        and str(row.get("endpoint_name", "")).casefold() == PRIMARY_ENDPOINT
        and int(float(str(row.get("duration_s") or "0"))) == PRIMARY_DURATION_S
        and str(row.get("power_representation", "")).casefold() == PRIMARY_REPRESENTATION
        and str(row.get("enters_meta", "")).strip().casefold() in {"true", "1", "yes"}
    ]
    if len(alpha_meta_rows) != 3:
        raise ValueError(f"Expected 3 enters_meta alpha dataset effects; got {len(alpha_meta_rows)}")

    return {
        "primary_rows": len(primary_rows),
        "pooled_rows": len(pooled_rows),
        "loo_alpha_rows": len(loo_alpha),
        "hiit_rows": len(hiit_rows),
    }


def _assert_figure1_replication_integrity(merged_root: Path) -> dict[str, object]:
    c6_dir = merged_root / "C6"
    c7_source = (
        merged_root
        / "C7"
        / "figures"
        / "source_data"
        / "figure1_panel_e_alpha_replication_forest.csv"
    )
    _fields, forest_rows = _read_csv_rows(c7_source)
    primary_rows = [
        row for row in forest_rows if str(row.get("row_type", "")).casefold() == "primary"
    ]
    pooled_rows = [
        row for row in forest_rows if str(row.get("row_type", "")).casefold() == "pooled"
    ]
    expected_datasets = set(PRIMARY_MANUSCRIPT_DATASETS)
    found_primary = {
        str(row.get("dataset_id", "")).casefold()
        for row in primary_rows
        if str(row.get("dataset_id", "")).strip()
    }
    if found_primary != expected_datasets:
        raise ValueError(
            "Figure 1E primary datasets mismatch: "
            f"expected {sorted(expected_datasets)} got {sorted(found_primary)}"
        )
    if len(primary_rows) != 3:
        raise ValueError(f"Figure 1E requires exactly 3 primary rows; got {len(primary_rows)}")
    if len(pooled_rows) != 1:
        raise ValueError(f"Figure 1E requires exactly 1 pooled row; got {len(pooled_rows)}")

    pooled = pooled_rows[0]
    if int(float(str(pooled.get("n_datasets") or "0"))) != 3:
        raise ValueError("Figure 1E pooled row must have n_datasets=3.")

    effects_path = c6_dir / LOW_DEMAND_ALPHA_EFFECTS_FILENAME
    _fields, effects_rows = _read_csv_rows(effects_path)
    effect_datasets = {
        str(row.get("dataset_id", "")).casefold()
        for row in effects_rows
        if str(row.get("enters_meta", "")).strip().casefold() in {"true", "1", "yes"}
    }
    if effect_datasets != expected_datasets:
        raise ValueError(
            "Low-demand alpha effects missing primary entries: "
            f"expected {sorted(expected_datasets)} got {sorted(effect_datasets)}"
        )

    loo_path = c6_dir / LOW_DEMAND_ALPHA_LOO_FILENAME
    _fields, loo_rows = _read_csv_rows(loo_path)
    completed = [
        row
        for row in loo_rows
        if str(row.get("analysis_status", "")).casefold() == "completed"
    ]
    if len(completed) != 3:
        raise ValueError(f"Figure 1E low-demand LOO requires 3 rows; got {len(completed)}")
    return {
        "primary_rows": len(primary_rows),
        "pooled_rows": len(pooled_rows),
        "loo_rows": len(completed),
    }


def _build_panel_composite_figure(
    *,
    component_pngs: Sequence[Path],
    dataset_ids: Sequence[str],
    title: str,
    output_dir: Path,
    stem: str,
    footer: str = "",
) -> tuple[Path, Path, Path]:
    fig, axes = plt.subplots(1, len(component_pngs), figsize=(6.5 * len(component_pngs), 6.5))
    if len(component_pngs) == 1:
        axes = [axes]

    def _trim_whitespace(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            gray = img
        else:
            gray = img[..., :3].mean(axis=2)
        content = gray < 0.99
        if not np.any(content):
            return img
        ys, xs = np.where(content)
        y0, y1 = int(np.min(ys)), int(np.max(ys))
        x0, x1 = int(np.min(xs)), int(np.max(xs))
        pad_y = max(2, int(0.02 * img.shape[0]))
        pad_x = max(2, int(0.02 * img.shape[1]))
        y0 = max(0, y0 - pad_y)
        y1 = min(img.shape[0] - 1, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(img.shape[1] - 1, x1 + pad_x)
        return img[y0 : y1 + 1, x0 : x1 + 1]

    for ax, png, dataset_id in zip(axes, component_pngs, dataset_ids, strict=True):
        image = _trim_whitespace(plt.imread(png))
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(str(dataset_id).casefold(), fontsize=14, fontweight="bold", pad=10)
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.985)
    if footer:
        fig.text(
            0.5,
            0.020,
            footer,
            ha="center",
            va="bottom",
            fontsize=11,
            color="#555555",
        )
    fig.tight_layout(rect=(0.0, 0.02 if footer else 0.0, 1.0, 0.95))
    return save_figure_trio(fig, output_dir, stem, bbox_inches=None, pad_inches=0.10)


def _render_figure3_dataset_composites(
    confirmatory_root: Path,
    merged_root: Path,
) -> dict[str, Path]:
    figures_dir = merged_root / "C7" / "figures"
    source_dir = figures_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = figures_dir / "_merged_panel_components"
    panel_e_pngs: list[Path] = []
    panel_f_pngs: list[Path] = []
    panel_e_rows: list[dict[str, object]] = []
    panel_f_rows: list[dict[str, object]] = []

    for dataset_id in PRIMARY_MANUSCRIPT_DATASETS:
        dataset_publish = _dataset_stage_dir(confirmatory_root, dataset_id, "C7") / "publish"
        dataset_inputs = resolve_reporting_inputs(dataset_publish)
        out_dir = tmp_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_e = render_figure3_panel_e(dataset_inputs, out_dir, include_internal_qc=False)
        panel_f = render_figure3_panel_f(
            dataset_inputs,
            out_dir,
            confirmatory_root=_dataset_stage_dir(confirmatory_root, dataset_id, "C6").parent,
            include_internal_qc=False,
            force_recompute_channel_zlpi=False,
            component_mode=True,
        )
        e_png = out_dir / f"{PANEL_E_STEM}.png"
        f_png = out_dir / f"{PANEL_F_STEM}.png"
        panel_e_pngs.append(e_png)
        panel_f_pngs.append(f_png)
        panel_e_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": "primary",
                "component_png": str(e_png),
                "upstream_specifications_csv": str(dataset_inputs.get("panel_e_specifications") or ""),
                "upstream_availability_csv": str(dataset_inputs.get("panel_e_availability") or ""),
                "analysis_role": "dataset_specific_panel_e_component",
            }
        )
        panel_f_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_role": "primary",
                "component_png": str(f_png),
                "upstream_summary_csv": str(dataset_inputs.get("panel_f_summary") or ""),
                "upstream_montage_csv": str(dataset_inputs.get("panel_f_montage") or ""),
                "analysis_role": "dataset_specific_panel_f_component",
            }
        )

    panel_e_pdf, panel_e_svg, panel_e_png = _build_panel_composite_figure(
        component_pngs=panel_e_pngs,
        dataset_ids=PRIMARY_MANUSCRIPT_DATASETS,
        title="Figure 3E | Nuisance robustness by dataset",
        output_dir=figures_dir,
        stem=PANEL_E_STEM,
    )
    panel_f_pdf, panel_f_svg, panel_f_png = _build_panel_composite_figure(
        component_pngs=panel_f_pngs,
        dataset_ids=PRIMARY_MANUSCRIPT_DATASETS,
        title="Figure 3F | Topography and low-γ sensitivity by dataset",
        output_dir=figures_dir,
        stem=PANEL_F_COMPOSITE_STEM,
        footer=(
            "Low-γ remains artifact-indeterminate; ECG-prone markers are channel-status markers, "
            "not significance markers."
        ),
    )
    panel_e_manifest = source_dir / "figure3_panel_e_dataset_composite_manifest.csv"
    panel_f_manifest = source_dir / "figure3_panel_f_dataset_composite_manifest.csv"
    write_source_csv(
        panel_e_manifest,
        panel_e_rows,
        (
            "dataset_id",
            "dataset_role",
            "component_png",
            "upstream_specifications_csv",
            "upstream_availability_csv",
            "analysis_role",
        ),
    )
    write_source_csv(
        panel_f_manifest,
        panel_f_rows,
        (
            "dataset_id",
            "dataset_role",
            "component_png",
            "upstream_summary_csv",
            "upstream_montage_csv",
            "analysis_role",
        ),
    )
    return {
        "panel_e_pdf": panel_e_pdf,
        "panel_e_svg": panel_e_svg,
        "panel_e_png": panel_e_png,
        "panel_f_pdf": panel_f_pdf,
        "panel_f_svg": panel_f_svg,
        "panel_f_png": panel_f_png,
        "panel_e_manifest": panel_e_manifest,
        "panel_f_manifest": panel_f_manifest,
    }


def _assert_figure3_merged_integrity(merged_root: Path) -> dict[str, object]:
    source = merged_root / "C7" / "figures" / "source_data"
    c6_dir = merged_root / "C6"
    panel_a_dist = source / "figure3_panel_a_null_distributions.csv"
    panel_b_obs = source / "figure3_panel_b_observation_estimates.csv"
    panel_b_participants = source / "figure3_panel_b_participant_estimates.csv"
    panel_b_group = source / "figure3_panel_b_group_summaries.csv"
    panel_c = source / "figure3_panel_c_duration_sensitivity.csv"
    panel_d = source / "figure3_panel_d_cardiac_controls_summaries.csv"
    panel_a_flag = source / "figure3_panel_a_incomplete_surrogate_export.flag"
    if panel_a_flag.is_file():
        raise ValueError("Panel A incomplete flag present; full surrogate distributions required.")

    _fields_a, panel_a_rows = _read_csv_rows(panel_a_dist)
    required_nulls = {"circular_shift", "phase_randomization", "block_shuffle"}
    primary_datasets = {row.get("dataset_id", "").casefold() for row in panel_a_rows if row.get("dataset_role", "").casefold() == "primary"}
    if primary_datasets != set(PRIMARY_MANUSCRIPT_DATASETS):
        raise ValueError(
            f"Panel A primary dataset set mismatch: expected {PRIMARY_MANUSCRIPT_DATASETS}, got {sorted(primary_datasets)}"
        )
    nulls = {row.get("null_type", "").casefold() for row in panel_a_rows}
    if nulls != required_nulls:
        raise ValueError(f"Panel A null set mismatch: expected {sorted(required_nulls)} got {sorted(nulls)}")
    pooled_tags = {
        row.get("dataset_id", "").casefold()
        for row in panel_a_rows
        if row.get("dataset_id", "").casefold() in {"pooled", "all"}
    }
    if pooled_tags:
        raise ValueError("Panel A contains cross-dataset pooled rows.")

    _fields_b_obs, panel_b_obs_rows = _read_csv_rows(panel_b_obs)
    _fields_b_participants, panel_b_participant_rows = _read_csv_rows(panel_b_participants)
    _fields_b, panel_b_rows = _read_csv_rows(panel_b_group)
    if not panel_b_rows:
        raise ValueError("Panel B group summaries are empty.")
    if not panel_b_participant_rows:
        raise ValueError("Panel B participant estimates are empty.")
    if any(not str(row.get("dataset_id", "")).strip() for row in panel_b_rows):
        raise ValueError("Panel B summaries must be dataset-stratified.")
    if any("pooled" == str(row.get("dataset_id", "")).casefold() for row in panel_b_rows):
        raise ValueError("Panel B must not include pooled cross-dataset summary rows.")
    eligible_obs = [
        row
        for row in panel_b_obs_rows
        if str(row.get("eligibility_flag", "")).casefold() in {"true", "1"}
    ]
    if not eligible_obs:
        raise ValueError("Panel B observation estimates contain no eligible rows.")
    required_estimands = {
        "correct_null_normalized_effect",
        "cross_subject_null_normalized_effect",
        "innovation_null_normalized_effect",
    }
    group_dataset_ids = {
        str(row.get("dataset_id", "")).casefold()
        for row in panel_b_rows
        if str(row.get("dataset_id", "")).strip()
    }
    expected_datasets = set(PRIMARY_MANUSCRIPT_DATASETS)
    if group_dataset_ids != expected_datasets:
        raise ValueError(
            "Panel B group summaries dataset set mismatch: "
            f"expected {sorted(expected_datasets)}, got {sorted(group_dataset_ids)}"
        )
    participant_dataset_ids = {
        str(row.get("dataset_id", "")).casefold()
        for row in panel_b_participant_rows
        if str(row.get("dataset_id", "")).strip()
    }
    if participant_dataset_ids != expected_datasets:
        raise ValueError(
            "Panel B participant estimates dataset set mismatch: "
            f"expected {sorted(expected_datasets)}, got {sorted(participant_dataset_ids)}"
        )
    estimands_by_dataset: dict[str, set[str]] = {}
    for row in panel_b_rows:
        dataset_id = str(row.get("dataset_id", "")).casefold()
        estimand = str(row.get("estimand", "")).casefold()
        estimands_by_dataset.setdefault(dataset_id, set()).add(estimand)
    for dataset_id in sorted(expected_datasets):
        missing_estimands = required_estimands - estimands_by_dataset.get(dataset_id, set())
        if missing_estimands:
            raise ValueError(
                f"Panel B missing estimands for {dataset_id}: {sorted(missing_estimands)}"
            )

    _fields_c, panel_c_rows = _read_csv_rows(panel_c)
    for row in panel_c_rows:
        d = int(float(str(row.get("duration_s") or "0")))
        endpoint_alias = str(row.get("endpoint_alias", "")).casefold()
        ds = str(row.get("dataset_id", "")).casefold()
        if d == 60 and endpoint_alias != "swpi":
            raise ValueError("Panel C D60 must map to SWPI.")
        if d == 120 and endpoint_alias != "mwpi":
            raise ValueError("Panel C D120 must map to MWPI.")
        if d in {180, 240} and endpoint_alias != "zlpi":
            raise ValueError("Panel C D180/D240 must map to ZLPI.")
        if ds == "ds003816" and d != 60:
            raise ValueError("ds003816 must remain D60-only sensitivity in Panel C.")
        if ds == "ds003816" and str(row.get("dataset_role", "")).casefold() == "primary":
            raise ValueError("ds003816 must not appear as primary in Panel C.")

    _fields_d, panel_d_rows = _read_csv_rows(panel_d)
    key_set = {
        (
            str(row.get("dataset_id", "")).casefold(),
            str(row.get("control", "")).casefold(),
        )
        for row in panel_d_rows
    }
    if any(not k[0] for k in key_set):
        raise ValueError("Panel D summaries must include dataset_id.")
    if len(key_set) != len(panel_d_rows):
        raise ValueError("Panel D summaries contain duplicate dataset/control keys.")
    for required_field in (
        "composition_differs_from_baseline",
        "n_baseline_eligible",
        "n_control_computable",
        "n_paired_complete_case",
    ):
        if any(str(row.get(required_field, "")).strip() == "" for row in panel_d_rows):
            raise ValueError(f"Panel D missing required field: {required_field}")

    for dataset_id in PRIMARY_MANUSCRIPT_DATASETS:
        panel_e_spec = c6_dir / "panel_e_by_dataset" / dataset_id / f"{PANEL_E_STEM}_specifications.csv"
        panel_f_summary = c6_dir / "panel_f_by_dataset" / dataset_id / f"{PANEL_F_STEM}_summary.csv"
        if not panel_e_spec.is_file():
            raise FileNotFoundError(f"Missing Panel E upstream for {dataset_id}: {panel_e_spec}")
        if not panel_f_summary.is_file():
            raise FileNotFoundError(f"Missing Panel F upstream for {dataset_id}: {panel_f_summary}")

    return {
        "panel_a_primary_dataset_count": len(primary_datasets),
        "panel_a_null_type_count": len(nulls),
        "panel_b_eligible_observation_rows": len(eligible_obs),
        "panel_b_participant_rows": len(panel_b_participant_rows),
        "panel_b_rows": len(panel_b_rows),
        "panel_c_rows": len(panel_c_rows),
        "panel_d_rows": len(panel_d_rows),
    }


def build_manuscript_primary_merged_meta_tree(confirmatory_root: str | Path) -> MergedManuscriptResult:
    """Build canonical merged C5->C6->C7 manuscript artifacts (Figures 1 and 2)."""
    root = Path(confirmatory_root).expanduser().resolve()
    merged_root = root / "primary" / "merged"
    c5_dir = merged_root / "C5"
    c6_dir = merged_root / "C6"
    c7_dir = merged_root / "C7"
    c7_publish = c7_dir / "publish"
    c7_figures = c7_dir / "figures"
    global_protocol_audit_refresh = _refresh_global_protocol_audit_roles(root)

    merged_counts = _merge_primary_upstream_tables(root, merged_root)
    merged_counts.update(_merge_primary_c5(root, c5_dir))
    run_confirmatory_inference_from_dir(c5_dir, c6_dir)
    merged_counts.update(_write_low_demand_alpha_replication(c5_dir, c6_dir))
    _write_merged_publish_tree(merged_root)

    inputs = resolve_reporting_inputs(c7_publish)
    figure1_artifacts = render_figure1(inputs, c7_figures)
    artifacts = render_figure2(inputs, c7_figures)
    integrity = _assert_figure2_meta_integrity(merged_root)
    figure1_integrity = _assert_figure1_replication_integrity(merged_root)
    role_integrity = _assert_manuscript_role_integrity(
        root,
        merged_root,
        include_figure3_checks=False,
    )

    summary = {
        "primary_datasets": list(PRIMARY_MANUSCRIPT_DATASETS),
        "merged_root": str(merged_root),
        "c5_dir": str(c5_dir),
        "c6_dir": str(c6_dir),
        "c7_dir": str(c7_dir),
        "c7_publish_dir": str(c7_publish),
        "figure1_png": str(figure1_artifacts.png),
        "figure1_source_csv": str(c7_figures / "source_data" / "figure1_panel_e_alpha_replication_forest.csv"),
        "figure2_png": str(artifacts.png),
        "figure2_source_csv": str(c7_figures / "source_data" / "figure2_panel_c_alpha_meta_forest.csv"),
        "merged_row_counts": merged_counts,
        "figure1_integrity": figure1_integrity,
        "integrity": integrity,
        "role_integrity": role_integrity,
        "global_protocol_audit_refresh": global_protocol_audit_refresh,
    }
    summary_path = merged_root / "manuscript_meta_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MergedManuscriptResult(
        merged_root=merged_root,
        c5_dir=c5_dir,
        c6_dir=c6_dir,
        c7_dir=c7_dir,
        c7_publish_dir=c7_publish,
        figure1_png=figure1_artifacts.png,
        figure1_source_csv=c7_figures / "source_data" / "figure1_panel_e_alpha_replication_forest.csv",
        figure2_png=artifacts.png,
        figure2_source_csv=c7_figures / "source_data" / "figure2_panel_c_alpha_meta_forest.csv",
        summary_json=summary_path,
    )


def build_manuscript_primary_merged_figure3_tree(
    confirmatory_root: str | Path,
) -> MergedFigure3ManuscriptResult:
    """Build canonical merged C4->C5->C6->C7 manuscript Figure 3 artifacts."""
    root = Path(confirmatory_root).expanduser().resolve()
    merged_root = root / "primary" / "merged"
    c4_dir = merged_root / "C4"
    c5_dir = merged_root / "C5"
    c6_dir = merged_root / "C6"
    c7_dir = merged_root / "C7"
    c7_publish = c7_dir / "publish"
    c7_figures = c7_dir / "figures"
    global_protocol_audit_refresh = _refresh_global_protocol_audit_roles(root)

    merged_counts = _merge_primary_upstream_tables(root, merged_root)
    merged_counts.update(_merge_primary_c5(root, c5_dir))
    run_confirmatory_inference_from_dir(c5_dir, c6_dir)
    merged_counts.update(_write_low_demand_alpha_replication(c5_dir, c6_dir))
    panel_f_repair = _ensure_primary_panel_f_upstream(root)
    merged_counts.update(_merge_primary_figure3_c6_inputs(root, merged_root))
    _write_merged_publish_tree(merged_root)

    inputs = resolve_reporting_inputs(c7_publish)
    figure3 = render_figure3(
        inputs,
        c7_figures,
        include_internal_qc=True,
        require_full_surrogate_distributions=True,
    )
    composite_paths = _render_figure3_dataset_composites(root, merged_root)
    integrity = _assert_figure3_merged_integrity(merged_root)
    role_integrity = _assert_manuscript_role_integrity(
        root,
        merged_root,
        include_figure3_checks=True,
    )

    summary = {
        "primary_datasets": list(PRIMARY_MANUSCRIPT_DATASETS),
        "merged_root": str(merged_root),
        "c4_dir": str(c4_dir),
        "c5_dir": str(c5_dir),
        "c6_dir": str(c6_dir),
        "c7_dir": str(c7_dir),
        "c7_publish_dir": str(c7_publish),
        "figure3_main_png": str(figure3.manuscript.png),
        "figure3_panel_e_png": str(composite_paths["panel_e_png"]),
        "figure3_panel_f_png": str(composite_paths["panel_f_png"]),
        "figure3_panel_e_manifest": str(composite_paths["panel_e_manifest"]),
        "figure3_panel_f_manifest": str(composite_paths["panel_f_manifest"]),
        "panel_f_upstream_regenerated": panel_f_repair,
        "merged_row_counts": merged_counts,
        "integrity_figure3": integrity,
        "role_integrity": role_integrity,
        "global_protocol_audit_refresh": global_protocol_audit_refresh,
    }
    summary_path = merged_root / "manuscript_figure3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MergedFigure3ManuscriptResult(
        merged_root=merged_root,
        c4_dir=c4_dir,
        c5_dir=c5_dir,
        c6_dir=c6_dir,
        c7_dir=c7_dir,
        c7_publish_dir=c7_publish,
        figure3_main_png=figure3.manuscript.png,
        figure3_panel_e_png=composite_paths["panel_e_png"],
        figure3_panel_f_png=composite_paths["panel_f_png"],
        summary_json=summary_path,
    )


__all__ = [
    "MergedFigure3ManuscriptResult",
    "MergedManuscriptResult",
    "PRIMARY_MANUSCRIPT_DATASETS",
    "build_manuscript_primary_merged_figure3_tree",
    "build_manuscript_primary_merged_meta_tree",
]
