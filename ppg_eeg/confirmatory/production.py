"""Production preflight and batch orchestration for confirmatory M1–M12 (M13a).

Supports ``preflight``, ``smoke``, ``primary``, ``sensitivity``, and
``clean_root`` modes. Missing raw or derivative inputs are recorded as blockers
and never interpreted as participant exclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .artifact_controls import (
    ARTIFACT_CONTROL_RESULTS_FILENAME,
    ARTIFACT_FIELDS,
    DURATION_FIELDS,
    DURATION_SENSITIVITY_FILENAME,
    SENSITIVITY_FIELDS,
    SENSITIVITY_QC_FILENAME,
    SENSITIVITY_RESULTS_FILENAME,
    SPEC_MATRIX_FIELDS,
    SPECIFICATION_MATRIX_FILENAME,
    run_confirmatory_artifact_controls,
)
from .config import (
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
    load_dataset_config,
    load_master_config,
)
from .correlation import (
    BAND_ORDER,
    CURVE_FIELDS,
    CURVES_TEMPLATE,
    QC_FIELDS as CORRELATION_QC_FIELDS,
    QC_TEMPLATE as CORRELATION_QC_TEMPLATE,
    run_confirmatory_correlations,
)
from .duration_contracts import EXPECTED_DURATIONS_S
from .endpoints import (
    METRICS_FIELDS,
    METRICS_TEMPLATE,
    QC_FIELDS as ENDPOINT_QC_FIELDS,
    QC_TEMPLATE as ENDPOINT_QC_TEMPLATE,
    run_confirmatory_endpoints,
)
from .group_tables import (
    PAIRED_CONTRAST_FIELDS,
    PAIRED_CONTRASTS_FILENAME,
    PAIRING_QC_FIELDS,
    PAIRING_QC_FILENAME,
    SUBJECT_LEVEL_FIELDS,
    SUBJECT_LEVEL_FILENAME,
    run_confirmatory_group_tables,
)
from .harmonize import ALIGNED_FEATURES_TEMPLATE
from .inference import (
    DATASET_EFFECT_FIELDS,
    DATASET_EFFECTS_FILENAME,
    EQUIVALENCE_FIELDS,
    INFERENCE_QC_FILENAME,
    LOO_FIELDS,
    LEAVE_ONE_DATASET_OUT_FILENAME,
    META_ANALYSIS_RESULTS_FILENAME,
    META_FIELDS,
    MIXED_MODEL_FIELDS,
    MIXED_MODEL_RESULTS_FILENAME,
    MULTIPLICITY_FIELDS,
    MULTIPLICITY_RESULTS_FILENAME,
    PEAK_CENTER_EQUIVALENCE_FILENAME,
    QC_FIELDS as INFERENCE_QC_FIELDS,
    run_confirmatory_inference_from_dir,
)
from .manifest import config_hash, git_commit_hash, sha256_file
from .nulls import (
    NULL_QC_FILENAME,
    NULL_SUBJECT_RESULTS_FILENAME,
    NULL_SUMMARY_FILENAME,
    QC_FIELDS as NULL_QC_FIELDS,
    SMOKE_N_SURROGATES,
    SUBJECT_RESULT_FIELDS as NULL_SUBJECT_FIELDS,
    SUMMARY_FIELDS as NULL_SUMMARY_FIELDS,
    run_confirmatory_nulls,
)
from .peak_model import (
    PARAMS_FIELDS,
    PARAMS_FILENAME,
    QC_FIELDS as PEAK_QC_FIELDS,
    QC_FILENAME as PEAK_QC_FILENAME,
    run_confirmatory_peak_fits,
)
from .protocol_audit import (
    ELIGIBILITY_BY_DURATION_FILENAME,
    ELIGIBILITY_QC_SUMMARY_FILENAME,
    PAIRED_SUBJECT_SETS_FILENAME,
    PROTOCOL_AUDIT_FILENAME,
    PROTOCOL_SPECS,
    EligibilityMetadata,
    observations_from_dataset_config,
    protocol_spec,
    write_duration_eligibility,
    write_protocol_audit,
)
from .report import run_confirmatory_reporting

PRODUCTION_PREFLIGHT_FILENAME = "production_preflight.csv"
PRODUCTION_RUN_PLAN_FILENAME = "production_run_plan.json"
STAGE_EXECUTION_LOG_FILENAME = "stage_execution_log.csv"
SMOKE_RUN_MANIFEST_FILENAME = "smoke_run_manifest.json"

VALID_MODES = (
    "preflight",
    "smoke",
    "primary",
    "sensitivity",
    "clean_root",
)

STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("M1", "protocol_audit"),
    ("M2", "instant_hr"),
    ("M3", "multitaper_power"),
    ("M4", "harmonize"),
    ("M5", "correlation"),
    ("M6", "endpoints"),
    ("M7", "peak_model"),
    ("M8", "group_tables"),
    ("M9", "nulls"),
    ("M10", "inference"),
    ("M11", "artifact_controls"),
    ("M12", "reporting"),
)

# Raw-signal stages cannot run from fixtures alone.
RAW_DEPENDENT_STAGES = frozenset({"M2", "M3", "M4"})

PREFLIGHT_FIELDS = (
    "check_id",
    "scope",
    "dataset_id",
    "role",
    "config_path",
    "status",
    "severity_class",
    "detail",
    "raw_root",
    "raw_available",
    "output_root",
)

STAGE_LOG_FIELDS = (
    "stage_id",
    "stage_name",
    "mode",
    "status",
    "started_at_utc",
    "finished_at_utc",
    "runtime_s",
    "command",
    "outputs",
    "schema_ok",
    "detail",
)


@dataclass(frozen=True)
class PreflightRow:
    check_id: str
    scope: str
    dataset_id: str
    role: str
    config_path: str
    status: str
    severity_class: str
    detail: str
    raw_root: str = ""
    raw_available: str = ""
    output_root: str = ""

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StageLogRow:
    stage_id: str
    stage_name: str
    mode: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    runtime_s: float
    command: str
    outputs: str
    schema_ok: str
    detail: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ProductionResult:
    mode: str
    production_root: Path
    preflight_path: Path
    run_plan_path: Path
    stage_log_path: Path
    smoke_manifest_path: Path | None
    preflight_rows: list[PreflightRow] = field(default_factory=list)
    stage_logs: list[StageLogRow] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready_for_primary: bool = False


def discover_config_dir(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "zero-lag-reanalysis-repo",
        Path.cwd() / "zero-lag-reanalysis-repo",
    ]
    for candidate in candidates:
        if master_config_path(candidate).is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate zero-lag-reanalysis-repo with master.yaml "
        "(or legacy config.confirmatory.master.yaml)."
    )


def master_config_path(config_dir: str | Path) -> Path:
    """Prefer ``master.yaml``; fall back to legacy flat filename."""
    root = Path(config_dir)
    preferred = root / "master.yaml"
    if preferred.is_file():
        return preferred
    legacy = root / "config.confirmatory.master.yaml"
    return legacy


def iter_dataset_config_paths(config_dir: Path) -> list[Path]:
    """Full-cohort dataset YAMLs under ``datasets/`` (new) or legacy flat names."""
    root = Path(config_dir)
    nested = sorted((root / "datasets").glob("*.yaml")) if (root / "datasets").is_dir() else []
    if nested:
        return nested
    return [
        path
        for path in sorted(root.glob("config.confirmatory.*.yaml"))
        if path.name != "config.confirmatory.master.yaml" and ".smoke." not in path.name
    ]


def iter_smoke_config_paths(config_dir: Path) -> list[Path]:
    """Smoke YAMLs: ``smoke/*.yaml`` plus ``smoke/*/confirmatory.yaml`` (e.g. HIIT)."""
    root = Path(config_dir)
    smoke_root = root / "smoke"
    paths: list[Path] = []
    if smoke_root.is_dir():
        paths.extend(sorted(smoke_root.glob("*.yaml")))
        paths.extend(sorted(smoke_root.glob("*/confirmatory.yaml")))
    if paths:
        return paths
    return sorted(root.glob("config.confirmatory.smoke.*.yaml"))


def default_production_root(master: ConfirmatoryMasterConfig) -> Path:
    return (master.output_root / "production").resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _read_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return list(next(reader))
        except StopIteration:
            return []


def validate_csv_schema(
    path: Path,
    expected_fields: Sequence[str],
    *,
    allow_extra: bool = True,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing file: {path.name}"
    header = _read_csv_header(path)
    if not header:
        return False, f"empty header: {path.name}"
    missing = [field for field in expected_fields if field not in header]
    if missing:
        return False, f"{path.name} missing columns: {missing}"
    if not allow_extra:
        unexpected = [field for field in header if field not in expected_fields]
        if unexpected:
            return False, f"{path.name} unexpected columns: {unexpected}"
    return True, "ok"


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=float)
    if int(np.sum(finite)) < 2:
        return result
    subset = values[finite]
    std = float(np.std(subset, ddof=0))
    mean = float(np.mean(subset))
    if std <= 0 or not math.isfinite(std):
        result[finite] = 0.0
        return result
    result[finite] = (subset - mean) / std
    return result


def _aligned_row(
    *,
    dataset_id: str,
    subject_id: str,
    task: str,
    condition: str,
    observation_id: str,
    duration_s: int,
    duration_role: str,
    time_s: float,
    hr: float,
    hr_z: float,
    band_raw: Mapping[str, float],
    band_z: Mapping[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "task": task,
        "condition": condition,
        "observation_id": observation_id,
        "duration_s": duration_s,
        "duration_role": duration_role,
        "time_s": time_s,
        "hr_bpm": hr,
        "hr_z": hr_z,
    }
    for band in BAND_ORDER:
        raw = float(band_raw[band])
        z_value = float(band_z[band])
        row[f"{band}_absolute_power"] = max(raw, 1e-12)
        row[f"{band}_absolute_log10_power"] = math.log10(max(raw, 1e-12))
        row[f"{band}_relative_power"] = 0.25
        row[f"{band}_broadband_residualized_log10"] = z_value
        row[f"{band}_absolute_log10_power_z"] = z_value
        row[f"{band}_relative_power_z"] = z_value
        row[f"{band}_broadband_residualized_log10_z"] = z_value
    return row


def build_synthetic_aligned_tables(
    output_dir: str | Path,
    *,
    seed: int = 20260713,
    n_participants: int = 4,
) -> dict[str, Path]:
    """Write synthetic M4-aligned duration tables for smoke E2E (no raw EEG)."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fixtures = (
        {
            "dataset_id": "ds003838",
            "low": "rest",
            "effort": "memory",
            "subjects": [f"sub-{index:03d}" for index in range(32, 32 + n_participants)],
        },
        {
            "dataset_id": "ds006848",
            "low": "rest",
            "effort": "verbalwm",
            "subjects": [f"sub-{index:03d}" for index in range(1, 1 + n_participants)],
        },
    )

    by_duration: dict[int, list[dict[str, object]]] = {
        duration: [] for duration in EXPECTED_DURATIONS_S
    }

    for fixture in fixtures:
        dataset_id = str(fixture["dataset_id"])
        for subject_id in fixture["subjects"]:
            assert isinstance(subject_id, str)
            for condition, lag_shift, amp in (
                (str(fixture["low"]), 0, 0.55),
                (str(fixture["effort"]), 2, 0.25),
            ):
                observation_id = (
                    f"{dataset_id}-{subject_id}-ses-single-task-{condition}"
                )
                latent = rng.normal(size=max(EXPECTED_DURATIONS_S))
                noise_hr = 0.35 * rng.normal(size=latent.size)
                hr_full = 70.0 + amp * 8.0 * latent + noise_hr
                band_full: dict[str, np.ndarray] = {}
                for band_i, band in enumerate(BAND_ORDER):
                    shared = np.roll(latent, lag_shift + band_i)
                    band_full[band] = np.exp(
                        -1.0
                        + amp * 0.8 * shared
                        + 0.25 * rng.normal(size=latent.size)
                    )

                for duration in EXPECTED_DURATIONS_S:
                    duration_role = "primary" if duration == 240 else "sensitivity"
                    start = (latent.size - duration) // 2
                    stop = start + duration
                    hr = hr_full[start:stop]
                    hr_z = _zscore(hr)
                    band_seg = {
                        band: values[start:stop] for band, values in band_full.items()
                    }
                    band_z = {band: _zscore(values) for band, values in band_seg.items()}
                    times = np.arange(duration, dtype=float)
                    for index, time_s in enumerate(times):
                        by_duration[duration].append(
                            _aligned_row(
                                dataset_id=dataset_id,
                                subject_id=subject_id,
                                task=condition,
                                condition=condition,
                                observation_id=observation_id,
                                duration_s=duration,
                                duration_role=duration_role,
                                time_s=float(time_s),
                                hr=float(hr[index]),
                                hr_z=float(hr_z[index]),
                                band_raw={
                                    band: float(band_seg[band][index])
                                    for band in BAND_ORDER
                                },
                                band_z={
                                    band: float(band_z[band][index])
                                    for band in BAND_ORDER
                                },
                            )
                        )

    written: dict[str, Path] = {}
    for duration, rows in by_duration.items():
        path = out / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
        fieldnames = list(rows[0].keys()) if rows else [
            "dataset_id",
            "subject_id",
            "task",
            "condition",
            "observation_id",
            "duration_s",
            "duration_role",
            "time_s",
            "hr_bpm",
            "hr_z",
        ]
        _write_csv(path, rows, fieldnames)
        written[f"aligned_D{duration}"] = path
    return written


def _primary_dataset_config_paths(config_dir: Path) -> list[Path]:
    master = load_master_config(master_config_path(config_dir))
    primary_ids = set(master.dataset_roles.primary)
    paths: list[Path] = []
    for path in iter_dataset_config_paths(config_dir):
        try:
            cfg = load_dataset_config(path, master=master)
        except Exception:
            continue
        if cfg.dataset_id in primary_ids and cfg.role == "primary":
            paths.append(path)
    return paths


def _sensitivity_dataset_config_paths(config_dir: Path) -> list[Path]:
    master = load_master_config(master_config_path(config_dir))
    sens_ids = set(master.dataset_roles.sensitivity)
    paths: list[Path] = []
    for path in iter_dataset_config_paths(config_dir):
        try:
            cfg = load_dataset_config(path, master=master)
        except Exception:
            continue
        if cfg.dataset_id in sens_ids and cfg.role == "sensitivity":
            paths.append(path)
    return paths


def resolve_dataset_configs_for_mode(
    config_dir: Path,
    mode: str,
    *,
    master: ConfirmatoryMasterConfig,
) -> list[ConfirmatoryDatasetConfig]:
    if mode == "smoke":
        paths = iter_smoke_config_paths(config_dir)
    elif mode in {"primary", "clean_root"}:
        paths = _primary_dataset_config_paths(config_dir)
    elif mode == "sensitivity":
        paths = _sensitivity_dataset_config_paths(config_dir)
    elif mode == "preflight":
        paths = [
            *iter_dataset_config_paths(config_dir),
            *iter_smoke_config_paths(config_dir),
        ]
    else:
        raise ValueError(f"Unknown mode {mode!r}.")
    return [load_dataset_config(path, master=master) for path in paths]


def _safe_n_observations(config: ConfirmatoryDatasetConfig) -> tuple[int | None, str]:
    if not config.paths.raw_root.exists():
        return None, "raw_root_missing"
    try:
        observations = observations_from_dataset_config(config)
    except FileNotFoundError as exc:
        return None, f"raw_discovery_failed:{exc}"
    except Exception as exc:  # noqa: BLE001 — preflight must not crash
        return None, f"raw_discovery_error:{type(exc).__name__}:{exc}"
    return len(observations), "ok"


def run_production_preflight(
    master: ConfirmatoryMasterConfig,
    dataset_configs: Sequence[ConfirmatoryDatasetConfig],
    *,
    config_paths: Sequence[Path],
) -> list[PreflightRow]:
    """Verify configs, pairing rules, channel declarations, and data availability."""
    rows: list[PreflightRow] = []

    rows.append(
        PreflightRow(
            check_id="master_output_root",
            scope="master",
            dataset_id="",
            role="",
            config_path=str(config_paths[0]) if config_paths else "",
            status="pass",
            severity_class="info",
            detail=(
                f"Dedicated confirmatory output_root={master.output_root} "
                f"(exists={master.output_root.exists()})."
            ),
            output_root=str(master.output_root),
        )
    )
    rows.append(
        PreflightRow(
            check_id="master_duration_contracts",
            scope="master",
            dataset_id="",
            role="",
            config_path="",
            status="pass",
            severity_class="info",
            detail=(
                f"durations={list(master.durations.all_s)}; "
                f"primary={master.durations.primary_s}; "
                f"hr_mode={master.cardiac.hr_mode}."
            ),
        )
    )

    known_specs = set(PROTOCOL_SPECS)
    master_primary = set(master.dataset_roles.primary)
    master_sensitivity = set(master.dataset_roles.sensitivity)
    if known_specs != (master_primary | master_sensitivity):
        rows.append(
            PreflightRow(
                check_id="dataset_role_coverage",
                scope="master",
                dataset_id="",
                role="",
                config_path="",
                status="fail",
                severity_class="blocker",
                detail=(
                    "Master dataset_roles must cover exactly PROTOCOL_SPECS. "
                    f"roles={sorted(master_primary | master_sensitivity)}; "
                    f"specs={sorted(known_specs)}."
                ),
            )
        )
    else:
        rows.append(
            PreflightRow(
                check_id="dataset_role_coverage",
                scope="master",
                dataset_id="",
                role="",
                config_path="",
                status="pass",
                severity_class="info",
                detail="Master dataset_roles match PROTOCOL_SPECS.",
            )
        )

    for config in dataset_configs:
        raw_exists = config.paths.raw_root.exists()
        n_obs, discovery_status = _safe_n_observations(config)
        spec = protocol_spec(config.dataset_id)

        rows.append(
            PreflightRow(
                check_id="dataset_config_load",
                scope="dataset",
                dataset_id=config.dataset_id,
                role=config.role,
                config_path=str(config.source_path),
                status="pass",
                severity_class="info",
                detail="Dataset YAML loaded and validated against master.",
                raw_root=str(config.paths.raw_root),
                raw_available=str(raw_exists),
                output_root=str(config.output_root),
            )
        )

        if not raw_exists or discovery_status != "ok":
            rows.append(
                PreflightRow(
                    check_id="raw_data_available",
                    scope="dataset",
                    dataset_id=config.dataset_id,
                    role=config.role,
                    config_path=str(config.source_path),
                    status="fail",
                    severity_class="blocker",
                    detail=(
                        "Required raw data unavailable "
                        f"({discovery_status}). This is a production blocker, "
                        "not a participant exclusion."
                    ),
                    raw_root=str(config.paths.raw_root),
                    raw_available="False",
                    output_root=str(config.output_root),
                )
            )
        else:
            rows.append(
                PreflightRow(
                    check_id="raw_data_available",
                    scope="dataset",
                    dataset_id=config.dataset_id,
                    role=config.role,
                    config_path=str(config.source_path),
                    status="pass",
                    severity_class="info",
                    detail=f"Raw root present; discovered_observations={n_obs}.",
                    raw_root=str(config.paths.raw_root),
                    raw_available="True",
                    output_root=str(config.output_root),
                )
            )

        channel_detail = (
            f"cardiac_modality={spec.cardiac_modality}; "
            f"cardiac_source={spec.cardiac_source}; "
            f"nuisance_signals={list(spec.nuisance_signals)}"
        )
        rows.append(
            PreflightRow(
                check_id="channel_protocol_inventory",
                scope="dataset",
                dataset_id=config.dataset_id,
                role=config.role,
                config_path=str(config.source_path),
                status="pass" if spec.cardiac_modality else "fail",
                severity_class="info" if spec.cardiac_modality else "blocker",
                detail=channel_detail,
                raw_root=str(config.paths.raw_root),
                raw_available=str(raw_exists),
                output_root=str(config.output_root),
            )
        )

        pairing_ok = True
        pairing_detail = (
            f"contrasts={[c.contrast_id for c in spec.contrasts]}; "
            f"run_pairing_policy={spec.run_pairing_policy}"
        )
        if config.role == "primary" and not spec.contrasts:
            # ds004582 is sensitivity/single-state; primary must have contrasts.
            if config.dataset_id not in {"ds004582"}:
                pairing_ok = False
                pairing_detail = (
                    "Primary dataset missing prespecified contrast pairing rules."
                )
        rows.append(
            PreflightRow(
                check_id="pairing_rules",
                scope="dataset",
                dataset_id=config.dataset_id,
                role=config.role,
                config_path=str(config.source_path),
                status="pass" if pairing_ok else "fail",
                severity_class="info" if pairing_ok else "blocker",
                detail=pairing_detail,
                raw_root=str(config.paths.raw_root),
                raw_available=str(raw_exists),
                output_root=str(config.output_root),
            )
        )

        if master.output_root not in config.output_root.parents:
            rows.append(
                PreflightRow(
                    check_id="output_root_isolation",
                    scope="dataset",
                    dataset_id=config.dataset_id,
                    role=config.role,
                    config_path=str(config.source_path),
                    status="fail",
                    severity_class="blocker",
                    detail="Dataset output_root is not under confirmatory master root.",
                    output_root=str(config.output_root),
                )
            )
        else:
            rows.append(
                PreflightRow(
                    check_id="output_root_isolation",
                    scope="dataset",
                    dataset_id=config.dataset_id,
                    role=config.role,
                    config_path=str(config.source_path),
                    status="pass",
                    severity_class="info",
                    detail="Dataset output_root is under confirmatory master root.",
                    output_root=str(config.output_root),
                )
            )

        if spec.unresolved_assumptions:
            rows.append(
                PreflightRow(
                    check_id="unresolved_protocol_assumptions",
                    scope="dataset",
                    dataset_id=config.dataset_id,
                    role=config.role,
                    config_path=str(config.source_path),
                    status="warn",
                    severity_class="warning",
                    detail="; ".join(spec.unresolved_assumptions),
                    raw_root=str(config.paths.raw_root),
                    raw_available=str(raw_exists),
                    output_root=str(config.output_root),
                )
            )

        # Derivative inputs for stages after M4 are optional at preflight; note presence.
        aligned_present = []
        for duration in EXPECTED_DURATIONS_S:
            aligned = (
                config.output_root
                / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
            )
            if aligned.is_file():
                aligned_present.append(duration)
        rows.append(
            PreflightRow(
                check_id="derivative_aligned_inputs",
                scope="dataset",
                dataset_id=config.dataset_id,
                role=config.role,
                config_path=str(config.source_path),
                status="pass" if aligned_present else "warn",
                severity_class="info" if aligned_present else "warning",
                detail=(
                    f"Existing aligned M4 tables for durations {aligned_present}."
                    if aligned_present
                    else "No M4 aligned derivative tables found under dataset output_root."
                ),
                output_root=str(config.output_root),
            )
        )

    return rows


def build_run_plan(
    *,
    mode: str,
    master: ConfirmatoryMasterConfig,
    dataset_configs: Sequence[ConfirmatoryDatasetConfig],
    preflight_rows: Sequence[PreflightRow],
    config_paths: Sequence[Path],
    production_root: Path,
    repo_root: Path,
    execute_stages: bool,
) -> dict[str, object]:
    blockers = [
        f"{row.dataset_id}:{row.check_id}:{row.detail}"
        for row in preflight_rows
        if row.severity_class == "blocker" and row.status == "fail"
    ]
    raw_blockers = [
        row
        for row in preflight_rows
        if row.check_id == "raw_data_available" and row.status == "fail"
    ]
    datasets_payload = []
    for config in dataset_configs:
        raw_ok = config.paths.raw_root.exists()
        datasets_payload.append(
            {
                "dataset_id": config.dataset_id,
                "role": config.role,
                "config_path": str(config.source_path),
                "raw_root": str(config.paths.raw_root),
                "raw_available": raw_ok,
                "output_root": str(config.output_root),
                "execution": (
                    "synthetic_smoke_fixture"
                    if mode == "smoke"
                    else ("planned" if raw_ok and execute_stages else "blocked_missing_raw")
                ),
            }
        )

    refuse_real_cohort = mode in {"primary", "sensitivity", "clean_root"} and bool(
        raw_blockers
    )
    return {
        "schema_version": "confirmatory_production_run_plan_v1",
        "generated_at_utc": _utc_now(),
        "mode": mode,
        "execute_stages": bool(execute_stages) and not refuse_real_cohort,
        "refuse_full_cohort_reason": (
            "Required raw data are unavailable; refusing full real-cohort execution. "
            "Missing data are production blockers, not participant exclusions."
            if refuse_real_cohort
            else ""
        ),
        "code_commit": git_commit_hash(repo_root),
        "config_hash": config_hash(config_paths),
        "root_seed": master.root_seed,
        "master_output_root": str(master.output_root),
        "production_root": str(production_root),
        "stage_order": [
            {"stage_id": stage_id, "stage_name": name}
            for stage_id, name in STAGE_ORDER
        ],
        "datasets": datasets_payload,
        "blockers": blockers,
        "ready_for_primary": mode == "primary"
        and not blockers
        and not refuse_real_cohort,
        "notes": [
            "Never interpret missing raw/derivative inputs as participant exclusion.",
            "Smoke mode runs synthetic fixtures end-to-end without requiring raw EEG.",
            "Primary/sensitivity/clean_root require raw availability for all planned datasets.",
        ],
    }


def _record_stage(
    logs: list[StageLogRow],
    *,
    stage_id: str,
    stage_name: str,
    mode: str,
    status: str,
    started: float,
    command: str,
    outputs: Sequence[Path] | Mapping[str, Path] | None = None,
    schema_ok: bool | None = None,
    detail: str = "",
) -> StageLogRow:
    finished = time.perf_counter()
    finished_at = _utc_now()
    if outputs is None:
        output_text = ""
    elif isinstance(outputs, Mapping):
        output_text = ";".join(f"{key}={path}" for key, path in outputs.items())
    else:
        output_text = ";".join(str(path) for path in outputs)
    row = StageLogRow(
        stage_id=stage_id,
        stage_name=stage_name,
        mode=mode,
        status=status,
        started_at_utc=finished_at,
        finished_at_utc=finished_at,
        runtime_s=round(finished - started, 6),
        command=command,
        outputs=output_text,
        schema_ok="" if schema_ok is None else str(bool(schema_ok)),
        detail=detail,
    )
    logs.append(row)
    return row


def _assert_schemas(
    checks: Sequence[tuple[Path, Sequence[str]]],
) -> tuple[bool, str]:
    messages: list[str] = []
    ok_all = True
    for path, fields in checks:
        ok, message = validate_csv_schema(path, fields)
        if not ok:
            ok_all = False
        messages.append(message)
    return ok_all, "; ".join(messages)


def run_safe_protocol_audit(
    master: ConfirmatoryMasterConfig,
    dataset_configs: Sequence[ConfirmatoryDatasetConfig],
    output_dir: Path,
) -> dict[str, Path]:
    """M1 audit that treats missing raw roots as not_supplied, never excluded."""
    observations_by_dataset: dict[str, Sequence[object]] = {}
    roles: dict[str, str] = {}
    metadata: list[EligibilityMetadata] = []
    for config in dataset_configs:
        roles[config.dataset_id] = config.role
        if not config.paths.raw_root.exists():
            observations_by_dataset[config.dataset_id] = ()
            metadata.append(
                EligibilityMetadata(
                    dataset_id=config.dataset_id,
                    observation_id="",
                    participant_id="",
                    condition="",
                    source_data_supplied=False,
                    eeg_exists=None,
                    cardiac_exists=None,
                    pairing_resolved=None,
                    protocol_match=None,
                    notes=(
                        "Raw root missing at preflight/smoke; recorded as "
                        "data_not_supplied, not participant exclusion."
                    ),
                )
            )
            continue
        try:
            observations = observations_from_dataset_config(config)
        except Exception as exc:  # noqa: BLE001
            observations_by_dataset[config.dataset_id] = ()
            metadata.append(
                EligibilityMetadata(
                    dataset_id=config.dataset_id,
                    observation_id="",
                    participant_id="",
                    condition="",
                    source_data_supplied=False,
                    eeg_exists=None,
                    cardiac_exists=None,
                    pairing_resolved=None,
                    protocol_match=None,
                    notes=(
                        f"Observation discovery failed ({type(exc).__name__}: {exc}); "
                        "data_not_supplied, not exclusion."
                    ),
                )
            )
            continue
        observations_by_dataset[config.dataset_id] = observations

    # Prefer derived metadata when discoveries succeeded.
    from .protocol_audit import eligibility_metadata_from_observations

    derived = eligibility_metadata_from_observations(observations_by_dataset)
    # Replace empty-dataset not_supplied notes with our explicit missing-data notes.
    by_dataset = {row.dataset_id: row for row in metadata}
    merged: list[EligibilityMetadata] = []
    for row in derived:
        if row.dataset_id in by_dataset and not row.source_data_supplied:
            merged.append(by_dataset[row.dataset_id])
        else:
            merged.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_csv, paired_json = write_protocol_audit(
        observations_by_dataset,
        output_dir,
        dataset_roles=roles,
    )
    eligibility_csv, summary_csv = write_duration_eligibility(merged, output_dir)
    return {
        "protocol_audit": protocol_csv,
        "paired_subject_sets": paired_json,
        "eligibility_by_duration": eligibility_csv,
        "eligibility_qc_summary": summary_csv,
    }


def production_publish_source_dirs(work_root: Path) -> tuple[Path, ...]:
    """Stage product directories flattened into ``publish/`` for M12 reporting.

    Must include ``m6_endpoints`` so ``confirmatory_endpoint_metrics_D*.csv``
    reach Figure 1 Panel C without manual copying.
    """
    root = Path(work_root)
    return (
        root / "m5_curves",
        root / "m6_endpoints",
        root / "m7_peaks",
        root / "m8_group",
        root / "m9_nulls",
        root / "m1_audit",
        root / "m10_inference",
        root / "m11_artifacts",
    )


def flatten_production_publish(
    work_root: str | Path,
    publish_root: str | Path | None = None,
) -> Path:
    """Copy stage CSVs/JSON into a flat ``publish/`` directory for figures."""
    work = Path(work_root).expanduser().resolve()
    out = (
        Path(publish_root).expanduser().resolve()
        if publish_root is not None
        else work / "publish"
    )
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for source in production_publish_source_dirs(work):
        if not source.is_dir():
            continue
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, out / path.name)
    return out


def execute_smoke_pipeline(
    *,
    master: ConfirmatoryMasterConfig,
    dataset_configs: Sequence[ConfirmatoryDatasetConfig],
    work_root: Path,
    config_paths: Sequence[Path],
    repo_root: Path,
    mode: str = "smoke",
) -> tuple[list[StageLogRow], dict[str, object]]:
    """Run M1–M12 smoke path using synthetic M4 fixtures for raw-dependent stages."""
    logs: list[StageLogRow] = []
    work_root.mkdir(parents=True, exist_ok=True)
    aligned_dir = work_root / "m4_aligned"
    curves_dir = work_root / "m5_curves"
    endpoints_dir = work_root / "m6_endpoints"
    peaks_dir = work_root / "m7_peaks"
    group_dir = work_root / "m8_group"
    nulls_dir = work_root / "m9_nulls"
    inference_dir = work_root / "m10_inference"
    artifacts_dir = work_root / "m11_artifacts"
    report_dir = work_root / "m12_report"
    audit_dir = work_root / "m1_audit"

    stage_outputs: dict[str, object] = {}

    # M1
    started = time.perf_counter()
    command = "run_safe_protocol_audit(...)"
    try:
        m1_paths = run_safe_protocol_audit(master, dataset_configs, audit_dir)
        schema_ok, detail = True, "protocol audit written"
        # Soft schema: audit CSVs must exist and be non-empty headers.
        for key, path in m1_paths.items():
            if not Path(path).is_file():
                schema_ok = False
                detail = f"missing {key}"
                break
        status = "ok" if schema_ok else "schema_fail"
        _record_stage(
            logs,
            stage_id="M1",
            stage_name="protocol_audit",
            mode=mode,
            status=status,
            started=started,
            command=command,
            outputs=m1_paths,
            schema_ok=schema_ok,
            detail=detail,
        )
        stage_outputs["M1"] = {k: str(v) for k, v in m1_paths.items()}
    except Exception as exc:  # noqa: BLE001
        _record_stage(
            logs,
            stage_id="M1",
            stage_name="protocol_audit",
            mode=mode,
            status="error",
            started=started,
            command=command,
            schema_ok=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
        raise

    # M2–M4: inject synthetic fixtures instead of raw processing.
    started = time.perf_counter()
    written = build_synthetic_aligned_tables(
        aligned_dir, seed=master.root_seed, n_participants=3
    )
    _record_stage(
        logs,
        stage_id="M2",
        stage_name="instant_hr",
        mode=mode,
        status="skipped_synthetic_fixture",
        started=started,
        command="build_synthetic_aligned_tables (covers M2–M4 inputs)",
        schema_ok=True,
        detail=(
            "Raw instant-HR skipped in smoke mode; synthetic aligned M4 tables "
            "provide the required downstream schema."
        ),
    )
    _record_stage(
        logs,
        stage_id="M3",
        stage_name="multitaper_power",
        mode=mode,
        status="skipped_synthetic_fixture",
        started=started,
        command="build_synthetic_aligned_tables (covers M2–M4 inputs)",
        schema_ok=True,
        detail="Raw multitaper skipped in smoke mode; synthetic fixtures used.",
    )
    _record_stage(
        logs,
        stage_id="M4",
        stage_name="harmonize",
        mode=mode,
        status="ok_synthetic",
        started=started,
        command="build_synthetic_aligned_tables(...)",
        outputs=written,
        schema_ok=True,
        detail=f"Wrote synthetic aligned tables: {sorted(written)}",
    )
    stage_outputs["M4"] = {k: str(v) for k, v in written.items()}

    # M5
    started = time.perf_counter()
    command = f"run_confirmatory_correlations({aligned_dir}, {curves_dir})"
    corr = run_confirmatory_correlations(aligned_dir, curves_dir)
    checks = []
    for duration in corr:
        checks.append(
            (
                curves_dir / CURVES_TEMPLATE.format(duration_s=duration),
                CURVE_FIELDS,
            )
        )
        checks.append(
            (
                curves_dir / CORRELATION_QC_TEMPLATE.format(duration_s=duration),
                CORRELATION_QC_FIELDS,
            )
        )
    schema_ok, detail = _assert_schemas(checks)
    _record_stage(
        logs,
        stage_id="M5",
        stage_name="correlation",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={str(d): curves_dir / CURVES_TEMPLATE.format(duration_s=d) for d in corr},
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M5"] = {
        str(d): str(curves_dir / CURVES_TEMPLATE.format(duration_s=d)) for d in corr
    }

    # M6
    started = time.perf_counter()
    command = f"run_confirmatory_endpoints({curves_dir}, {endpoints_dir})"
    endpoints = run_confirmatory_endpoints(curves_dir, endpoints_dir)
    checks = []
    for duration in endpoints:
        checks.append(
            (
                endpoints_dir / METRICS_TEMPLATE.format(duration_s=duration),
                METRICS_FIELDS,
            )
        )
        checks.append(
            (
                endpoints_dir / ENDPOINT_QC_TEMPLATE.format(duration_s=duration),
                ENDPOINT_QC_FIELDS,
            )
        )
    schema_ok, detail = _assert_schemas(checks)
    _record_stage(
        logs,
        stage_id="M6",
        stage_name="endpoints",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            str(d): endpoints_dir / METRICS_TEMPLATE.format(duration_s=d)
            for d in endpoints
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M6"] = {
        str(d): str(endpoints_dir / METRICS_TEMPLATE.format(duration_s=d))
        for d in endpoints
    }

    # M7
    started = time.perf_counter()
    command = f"run_confirmatory_peak_fits({curves_dir}, {peaks_dir})"
    run_confirmatory_peak_fits(curves_dir, peaks_dir)
    schema_ok, detail = _assert_schemas(
        [
            (peaks_dir / PARAMS_FILENAME, PARAMS_FIELDS),
            (peaks_dir / PEAK_QC_FILENAME, PEAK_QC_FIELDS),
        ]
    )
    _record_stage(
        logs,
        stage_id="M7",
        stage_name="peak_model",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            "params": peaks_dir / PARAMS_FILENAME,
            "qc": peaks_dir / PEAK_QC_FILENAME,
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M7"] = {
        "params": str(peaks_dir / PARAMS_FILENAME),
        "qc": str(peaks_dir / PEAK_QC_FILENAME),
    }

    # M8
    started = time.perf_counter()
    command = (
        f"run_confirmatory_group_tables({endpoints_dir}, {group_dir}, "
        f"peaks_dir={peaks_dir})"
    )
    run_confirmatory_group_tables(endpoints_dir, group_dir, peaks_dir=peaks_dir)
    schema_ok, detail = _assert_schemas(
        [
            (group_dir / SUBJECT_LEVEL_FILENAME, SUBJECT_LEVEL_FIELDS),
            (group_dir / PAIRED_CONTRASTS_FILENAME, PAIRED_CONTRAST_FIELDS),
            (group_dir / PAIRING_QC_FILENAME, PAIRING_QC_FIELDS),
        ]
    )
    _record_stage(
        logs,
        stage_id="M8",
        stage_name="group_tables",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            "subject_level": group_dir / SUBJECT_LEVEL_FILENAME,
            "paired_contrasts": group_dir / PAIRED_CONTRASTS_FILENAME,
            "pairing_qc": group_dir / PAIRING_QC_FILENAME,
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M8"] = {
        "subject_level": str(group_dir / SUBJECT_LEVEL_FILENAME),
        "paired_contrasts": str(group_dir / PAIRED_CONTRASTS_FILENAME),
        "pairing_qc": str(group_dir / PAIRING_QC_FILENAME),
    }

    # M9
    started = time.perf_counter()
    command = (
        f"run_confirmatory_nulls({aligned_dir}, {nulls_dir}, "
        f"n_surrogates={SMOKE_N_SURROGATES}, durations=(240,))"
    )
    run_confirmatory_nulls(
        aligned_dir,
        nulls_dir,
        n_surrogates=SMOKE_N_SURROGATES,
        durations=(240,),
    )
    schema_ok, detail = _assert_schemas(
        [
            (nulls_dir / NULL_SUBJECT_RESULTS_FILENAME, NULL_SUBJECT_FIELDS),
            (nulls_dir / NULL_SUMMARY_FILENAME, NULL_SUMMARY_FIELDS),
            (nulls_dir / NULL_QC_FILENAME, NULL_QC_FIELDS),
        ]
    )
    _record_stage(
        logs,
        stage_id="M9",
        stage_name="nulls",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            "null_subject_results": nulls_dir / NULL_SUBJECT_RESULTS_FILENAME,
            "null_summary": nulls_dir / NULL_SUMMARY_FILENAME,
            "null_qc": nulls_dir / NULL_QC_FILENAME,
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M9"] = {
        "null_subject_results": str(nulls_dir / NULL_SUBJECT_RESULTS_FILENAME),
        "null_summary": str(nulls_dir / NULL_SUMMARY_FILENAME),
        "null_qc": str(nulls_dir / NULL_QC_FILENAME),
    }

    # M10
    started = time.perf_counter()
    command = f"run_confirmatory_inference_from_dir({group_dir}, {inference_dir})"
    run_confirmatory_inference_from_dir(group_dir, inference_dir)
    schema_ok, detail = _assert_schemas(
        [
            (inference_dir / MIXED_MODEL_RESULTS_FILENAME, MIXED_MODEL_FIELDS),
            (inference_dir / DATASET_EFFECTS_FILENAME, DATASET_EFFECT_FIELDS),
            (inference_dir / META_ANALYSIS_RESULTS_FILENAME, META_FIELDS),
            (inference_dir / LEAVE_ONE_DATASET_OUT_FILENAME, LOO_FIELDS),
            (inference_dir / PEAK_CENTER_EQUIVALENCE_FILENAME, EQUIVALENCE_FIELDS),
            (inference_dir / MULTIPLICITY_RESULTS_FILENAME, MULTIPLICITY_FIELDS),
            (inference_dir / INFERENCE_QC_FILENAME, INFERENCE_QC_FIELDS),
        ]
    )
    _record_stage(
        logs,
        stage_id="M10",
        stage_name="inference",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            "mixed_model_results": inference_dir / MIXED_MODEL_RESULTS_FILENAME,
            "meta_analysis_results": inference_dir / META_ANALYSIS_RESULTS_FILENAME,
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M10"] = {
        "mixed_model_results": str(inference_dir / MIXED_MODEL_RESULTS_FILENAME),
        "meta_analysis_results": str(inference_dir / META_ANALYSIS_RESULTS_FILENAME),
    }

    # M11
    started = time.perf_counter()
    command = f"run_confirmatory_artifact_controls({group_dir}, {artifacts_dir}, enable_optional_artifact_controls=False)"
    run_confirmatory_artifact_controls(
        group_dir,
        artifacts_dir,
        enable_optional_artifact_controls=False,
    )
    schema_ok, detail = _assert_schemas(
        [
            (artifacts_dir / SENSITIVITY_RESULTS_FILENAME, SENSITIVITY_FIELDS),
            (artifacts_dir / ARTIFACT_CONTROL_RESULTS_FILENAME, ARTIFACT_FIELDS),
            (artifacts_dir / DURATION_SENSITIVITY_FILENAME, DURATION_FIELDS),
            (artifacts_dir / SPECIFICATION_MATRIX_FILENAME, SPEC_MATRIX_FIELDS),
        ]
    )
    # sensitivity_qc may use sparse fields; require file exists.
    qc_path = artifacts_dir / SENSITIVITY_QC_FILENAME
    if not qc_path.is_file():
        schema_ok = False
        detail = f"{detail}; missing {SENSITIVITY_QC_FILENAME}"
    _record_stage(
        logs,
        stage_id="M11",
        stage_name="artifact_controls",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs={
            "sensitivity_results": artifacts_dir / SENSITIVITY_RESULTS_FILENAME,
            "specification_matrix": artifacts_dir / SPECIFICATION_MATRIX_FILENAME,
        },
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M11"] = {
        "sensitivity_results": str(artifacts_dir / SENSITIVITY_RESULTS_FILENAME),
        "specification_matrix": str(artifacts_dir / SPECIFICATION_MATRIX_FILENAME),
    }

    # M12 — publish reports from group/inference trees assembled under work_root.
    # Figures expect curves, endpoint metrics, and group tables under one root.
    publish_root = flatten_production_publish(work_root)

    started = time.perf_counter()
    command = f"run_confirmatory_reporting({publish_root}, {report_dir})"
    report_paths = run_confirmatory_reporting(
        publish_root,
        report_dir,
        config_paths=config_paths,
        repo_root=repo_root,
        seeds={"root_seed": master.root_seed, "n_surrogates": SMOKE_N_SURROGATES},
    )
    required = ("figure1_pdf", "results_bundle", "run_manifest")
    schema_ok = all(key in report_paths and Path(report_paths[key]).is_file() for key in required)
    detail = "reporting artifacts written" if schema_ok else "missing report artifacts"
    _record_stage(
        logs,
        stage_id="M12",
        stage_name="reporting",
        mode=mode,
        status="ok" if schema_ok else "schema_fail",
        started=started,
        command=command,
        outputs=report_paths,
        schema_ok=schema_ok,
        detail=detail,
    )
    stage_outputs["M12"] = {k: str(v) for k, v in report_paths.items()}

    manifest = {
        "schema_version": "confirmatory_smoke_run_manifest_v1",
        "generated_at_utc": _utc_now(),
        "mode": mode,
        "code_commit": git_commit_hash(repo_root),
        "config_hash": config_hash(config_paths),
        "root_seed": master.root_seed,
        "n_surrogates": SMOKE_N_SURROGATES,
        "work_root": str(work_root),
        "stage_outputs": stage_outputs,
        "stage_statuses": {
            row.stage_id: {"status": row.status, "schema_ok": row.schema_ok, "runtime_s": row.runtime_s}
            for row in logs
        },
        "input_availability": {
            "raw_dependent_stages": sorted(RAW_DEPENDENT_STAGES),
            "synthetic_fixture": True,
            "note": (
                "Smoke used synthetic M4-aligned fixtures; M2/M3 raw extraction "
                "was not executed against unavailable cohort files."
            ),
        },
    }
    return logs, manifest


def run_production(
    *,
    mode: str,
    config_dir: str | Path | None = None,
    production_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    clean_smoke_work: bool = True,
) -> ProductionResult:
    """Orchestrate confirmatory production preflight and optional smoke execution."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}.")

    config_root = discover_config_dir(config_dir)
    master_path = master_config_path(config_root)
    master = load_master_config(master_path)
    repo = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else config_root.parent.resolve()
    )
    out_root = (
        Path(production_root).expanduser().resolve()
        if production_root is not None
        else default_production_root(master)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_configs = resolve_dataset_configs_for_mode(
        config_root, mode, master=master
    )
    config_paths = [master_path, *[cfg.source_path for cfg in dataset_configs]]

    preflight_rows = run_production_preflight(
        master,
        dataset_configs,
        config_paths=config_paths,
    )
    execute_stages = mode == "smoke"
    run_plan = build_run_plan(
        mode=mode,
        master=master,
        dataset_configs=dataset_configs,
        preflight_rows=preflight_rows,
        config_paths=config_paths,
        production_root=out_root,
        repo_root=repo,
        execute_stages=execute_stages,
    )

    # clean_root mode: require empty/absent target output roots before planning execution.
    if mode == "clean_root":
        dirty = [
            str(cfg.output_root)
            for cfg in dataset_configs
            if cfg.output_root.exists() and any(cfg.output_root.iterdir())
        ]
        if dirty:
            run_plan["execute_stages"] = False
            run_plan["refuse_full_cohort_reason"] = (
                "Clean-root reproducibility requires empty dataset output roots; "
                f"non-empty: {dirty}. Refusing execution."
            )
            for path in dirty:
                preflight_rows.append(
                    PreflightRow(
                        check_id="clean_root_empty",
                        scope="dataset",
                        dataset_id=Path(path).name,
                        role="primary",
                        config_path="",
                        status="fail",
                        severity_class="blocker",
                        detail=f"Output root not empty for clean-root mode: {path}",
                        output_root=path,
                    )
                )
            run_plan["blockers"] = [
                f"{row.dataset_id}:{row.check_id}:{row.detail}"
                for row in preflight_rows
                if row.severity_class == "blocker" and row.status == "fail"
            ]

    preflight_path = _write_csv(
        out_root / PRODUCTION_PREFLIGHT_FILENAME,
        [row.to_row() for row in preflight_rows],
        PREFLIGHT_FIELDS,
    )
    run_plan_path = out_root / PRODUCTION_RUN_PLAN_FILENAME
    run_plan_path.write_text(
        json.dumps(run_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stage_logs: list[StageLogRow] = []
    smoke_manifest_path: Path | None = None

    if mode == "smoke":
        work_root = out_root / "smoke_e2e"
        if clean_smoke_work and work_root.exists():
            shutil.rmtree(work_root)
        stage_logs, smoke_manifest = execute_smoke_pipeline(
            master=master,
            dataset_configs=dataset_configs,
            work_root=work_root,
            config_paths=config_paths,
            repo_root=repo,
            mode=mode,
        )
        smoke_manifest["production_preflight"] = str(preflight_path)
        smoke_manifest["production_run_plan"] = str(run_plan_path)
        smoke_manifest["preflight_sha256"] = sha256_file(preflight_path)
        smoke_manifest["run_plan_sha256"] = sha256_file(run_plan_path)
        smoke_manifest_path = out_root / SMOKE_RUN_MANIFEST_FILENAME
        smoke_manifest_path.write_text(
            json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif mode in {"primary", "sensitivity", "clean_root"}:
        # Explicit no-op execution when blocked; still log refusal for every stage.
        reason = str(run_plan.get("refuse_full_cohort_reason") or "")
        for stage_id, stage_name in STAGE_ORDER:
            started = time.perf_counter()
            _record_stage(
                stage_logs,
                stage_id=stage_id,
                stage_name=stage_name,
                mode=mode,
                status="blocked" if reason else "planned_only",
                started=started,
                command="(not executed in M13a)",
                schema_ok=None,
                detail=reason
                or (
                    "M13a writes the run plan only for this mode; "
                    "full cohort execution is deferred until raw data are available."
                ),
            )
    else:
        # preflight only
        started = time.perf_counter()
        _record_stage(
            stage_logs,
            stage_id="M0",
            stage_name="preflight_only",
            mode=mode,
            status="ok",
            started=started,
            command="run_production_preflight(...)",
            outputs={"production_preflight": preflight_path},
            schema_ok=True,
            detail="Preflight-only mode; no M1–M12 execution.",
        )

    stage_log_path = _write_csv(
        out_root / STAGE_EXECUTION_LOG_FILENAME,
        [row.to_row() for row in stage_logs],
        STAGE_LOG_FIELDS,
    )

    blockers = [
        str(item) for item in run_plan.get("blockers", []) if str(item).strip()
    ]
    ready = bool(run_plan.get("ready_for_primary"))
    return ProductionResult(
        mode=mode,
        production_root=out_root,
        preflight_path=preflight_path,
        run_plan_path=run_plan_path,
        stage_log_path=stage_log_path,
        smoke_manifest_path=smoke_manifest_path,
        preflight_rows=preflight_rows,
        stage_logs=stage_logs,
        blockers=blockers,
        ready_for_primary=ready,
    )


def primary_blockers_summary(result: ProductionResult) -> list[str]:
    """Human-readable blockers that must be resolved before primary production."""
    lines: list[str] = []
    for row in result.preflight_rows:
        if row.severity_class != "blocker" or row.status != "fail":
            continue
        scope = row.dataset_id or row.scope
        lines.append(f"[{row.check_id}] {scope}: {row.detail}")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Confirmatory production preflight and smoke orchestration (M13a). "
            "Configs: zero-lag-reanalysis-repo/."
        )
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        required=True,
        help="preflight | smoke | primary | sensitivity | clean_root",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help=(
            "Confirmatory config root (default: zero-lag-reanalysis-repo/). "
            "Expects master.yaml, datasets/, smoke/."
        ),
    )
    parser.add_argument(
        "--production-root",
        type=str,
        default=None,
        help="Override production output directory",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Git repo root for commit capture",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_production(
            mode=args.mode,
            config_dir=args.config_dir,
            production_root=args.production_root,
            repo_root=args.repo_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[confirmatory.production] error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"mode={result.mode}")
    print(f"production_root={result.production_root}")
    print(f"production_preflight={result.preflight_path}")
    print(f"production_run_plan={result.run_plan_path}")
    print(f"stage_execution_log={result.stage_log_path}")
    if result.smoke_manifest_path is not None:
        print(f"smoke_run_manifest={result.smoke_manifest_path}")
    print(f"ready_for_primary={result.ready_for_primary}")
    blockers = primary_blockers_summary(result)
    if blockers:
        print("blockers_before_primary:")
        for line in blockers:
            print(f"  - {line}")
    if result.mode == "smoke":
        failed = [
            row
            for row in result.stage_logs
            if row.status in {"error", "schema_fail"}
        ]
        if failed:
            print(
                f"[confirmatory.production] smoke failed stages: "
                f"{[row.stage_id for row in failed]}",
                file=sys.stderr,
            )
            return 2
    return 0


__all__ = [
    "PRODUCTION_PREFLIGHT_FILENAME",
    "PRODUCTION_RUN_PLAN_FILENAME",
    "RAW_DEPENDENT_STAGES",
    "SMOKE_RUN_MANIFEST_FILENAME",
    "STAGE_EXECUTION_LOG_FILENAME",
    "STAGE_ORDER",
    "VALID_MODES",
    "PreflightRow",
    "ProductionResult",
    "StageLogRow",
    "build_run_plan",
    "build_synthetic_aligned_tables",
    "default_production_root",
    "discover_config_dir",
    "execute_smoke_pipeline",
    "flatten_production_publish",
    "iter_dataset_config_paths",
    "iter_smoke_config_paths",
    "main",
    "master_config_path",
    "primary_blockers_summary",
    "production_publish_source_dirs",
    "resolve_dataset_configs_for_mode",
    "run_production",
    "run_production_preflight",
    "run_safe_protocol_audit",
    "validate_csv_schema",
]


if __name__ == "__main__":
    raise SystemExit(main())
