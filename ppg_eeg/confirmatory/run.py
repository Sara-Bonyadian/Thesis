"""Confirmatory stage runner: C0–C7 dependency-checked dispatch.

Usage (via ``python -m ppg_eeg.confirmatory``)::

    --config zero-lag-reanalysis-repo/smoke/hiit/confirmatory.yaml --stage C0
    --config … --stage all

Never deletes existing stage outputs; resumes by writing beside them.
Missing prerequisites raise clear errors (blockers), not silent exclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ..core_eeg_ppg.output_layout import safe_subject_dir_name
from ..datasets import CanonicalObservation
from .artifact_controls import run_confirmatory_artifact_controls
from .config import (
    ConfirmatoryDatasetConfig,
    ConfirmatoryMasterConfig,
    load_dataset_config,
    load_master_config,
)
from .correlation import run_confirmatory_correlations
from .endpoints import run_confirmatory_endpoints
from .group_tables import run_confirmatory_group_tables
from .harmonize import (
    ALIGNED_FEATURES_TEMPLATE,
    BAND_ORDER,
    harmonize_observation,
    write_harmonize_outputs,
)
from .inference import run_confirmatory_inference_from_dir
from .instant_hr import FEATURES_FILENAME as INSTANT_HR_FEATURES
from .instant_hr import reconstruct_instant_hr_file
from .peak_detection import PEAKS_FILENAME, run_confirmatory_peak_detection
from .multitaper_power import (
    FEATURES_FILENAME as MULTITAPER_FEATURES,
    ROBUST_MEDIAN_CHANNEL,
    extract_multitaper_file,
)
from .nulls import DEFAULT_N_SURROGATES, SMOKE_N_SURROGATES, run_confirmatory_nulls
from .peak_model import run_confirmatory_peak_fits
from .production import discover_config_dir, master_config_path
from .data_audit import run_confirmatory_data_audit
from .protocol_audit import (
    eligibility_metadata_from_observations,
    enrich_eligibility_metadata_from_csv,
    observations_from_dataset_config,
    run_protocol_audit,
)
from .report import run_confirmatory_reporting

VALID_STAGES = (
    "C0",
    "C1a",
    "C1b",
    "C1c",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "all",
)

STAGE_ORDER = (
    "C0",
    "C1a",
    "C1b",
    "C1c",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
)

# Soft prerequisites: must exist before the stage runs (unless --force).
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "C0": (),
    "C1a": (),
    "C1b": (),
    "C1c": ("C1a", "C1b"),
    "C2": ("C1c",),
    "C3": ("C2",),
    "C4": ("C1c",),
    "C5": ("C3",),
    "C6": ("C5",),
    "C7": ("C2", "C5"),
}

STAGE_STATUS_FILENAME = "stage_status.json"


class StageError(RuntimeError):
    """Hard stop for missing inputs or failed stage contracts."""


@dataclass
class StageContext:
    master: ConfirmatoryMasterConfig
    dataset: ConfirmatoryDatasetConfig
    master_path: Path
    dataset_path: Path
    n_surrogates: int = DEFAULT_N_SURROGATES
    n_jobs: int = -1
    force: bool = False
    enable_optional_artifact_controls: bool = False
    repo_root: Path | None = None
    logs: list[dict[str, object]] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.dataset.output_root

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _obs_dir(stage_root: Path, observation_id: str) -> Path:
    return stage_root / safe_subject_dir_name(observation_id)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_stage_status(ctx: StageContext, stage: str, status: str, detail: str) -> None:
    path = ctx.root / STAGE_STATUS_FILENAME
    payload: dict[str, object] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.setdefault("stages", {})
    assert isinstance(stages, dict)
    stages[stage] = {
        "status": status,
        "detail": detail,
        "finished_at_utc": _utc_now(),
    }
    payload["dataset_id"] = ctx.dataset.dataset_id
    payload["output_root"] = str(ctx.root)
    payload["updated_at_utc"] = _utc_now()
    ctx.root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_stages(ctx: StageContext, stage: str) -> None:
    if ctx.force:
        return
    missing = [
        req
        for req in STAGE_REQUIRES.get(stage, ())
        if not ctx.stage_dir(req).exists()
    ]
    if missing:
        raise StageError(
            f"Stage {stage} requires prior stage outputs {missing} under "
            f"{ctx.root}. Run those stages first (or pass --force)."
        )


def _load_observations(ctx: StageContext) -> list[CanonicalObservation]:
    if not ctx.dataset.paths.raw_root.exists():
        raise StageError(
            f"raw_root missing: {ctx.dataset.paths.raw_root}. "
            "This is a production blocker, not a participant exclusion."
        )
    try:
        return observations_from_dataset_config(ctx.dataset)
    except FileNotFoundError as exc:
        raise StageError(
            f"Observation discovery failed: {exc}. "
            "Missing data are blockers, not exclusions."
        ) from exc


def _write_c0_stage_status(
    out: Path,
    *,
    dataset_id: str,
    artifacts: Mapping[str, Path],
) -> Path:
    """Record that raw-data, pairing, and duration-eligibility audits ran in C0."""
    path = out / STAGE_STATUS_FILENAME
    payload = {
        "dataset_id": dataset_id,
        "stage": "C0",
        "status": "ok",
        "finished_at_utc": _utc_now(),
        "audits_completed": [
            "raw_data_audit",
            "pairing_audit",
            "duration_eligibility_audit",
        ],
        "artifacts": {key: str(value) for key, value in artifacts.items()},
        "notes": (
            "C0 performs the observation-level raw-data audit internally; "
            "exploratory temporal_coupling --stage 0 is not required. "
            "clean_beat_span_s is not computed here (beat detection deferred)."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_c0(ctx: StageContext) -> dict[str, Path]:
    out = ctx.stage_dir("C0")
    out.mkdir(parents=True, exist_ok=True)
    observations = _load_observations(ctx)
    lag_max_s = float(ctx.master.lag.max_s)
    data_audit_path, _records = run_confirmatory_data_audit(
        observations,
        out,
        lag_max_s=lag_max_s,
    )
    metadata = eligibility_metadata_from_observations(
        {ctx.dataset.dataset_id: observations}
    )
    metadata = enrich_eligibility_metadata_from_csv(
        metadata,
        raw_audit_paths=[data_audit_path],
    )
    paths = run_protocol_audit(
        ctx.master,
        [ctx.dataset],
        output_dir=out,
        eligibility_metadata=metadata,
    )
    mapping = {
        "protocol_audit": paths[0],
        "paired_subject_sets": paths[1],
        "data_audit": data_audit_path,
        "eligibility_by_duration": paths[2],
        "eligibility_qc_summary": paths[3],
    }
    mapping["stage_status"] = _write_c0_stage_status(
        out,
        dataset_id=ctx.dataset.dataset_id,
        artifacts=mapping,
    )
    return mapping


def run_c1a(ctx: StageContext) -> dict[str, object]:
    observations = _load_observations(ctx)
    out_root = ctx.stage_dir("C1a")
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    errors: list[str] = []
    for obs in observations:
        obs_out = _obs_dir(out_root, obs.observation_id)
        try:
            extract_multitaper_file(
                obs.eeg_path,
                obs.eeg_format,
                obs_out,
                identity={
                    "dataset_id": obs.dataset_id,
                    "subject_id": obs.subject_id,
                    "task": obs.task_label,
                    "condition": obs.condition_label,
                    "observation_id": obs.observation_id,
                },
                line_frequency_hz=ctx.dataset.eeg.line_frequency_hz,
            )
            written.append(obs.observation_id)
        except Exception as exc:  # noqa: BLE001 — continue other observations
            errors.append(f"{obs.observation_id}: {type(exc).__name__}: {exc}")
    if not written:
        raise StageError(f"C1a wrote no observations. Errors: {errors[:5]}")
    return {"n_ok": len(written), "n_error": len(errors), "errors": errors[:20]}


def run_c1b(ctx: StageContext) -> dict[str, object]:
    """Detect peaks, write QC, and reconstruct instantaneous HR under C1b/."""
    observations = _load_observations(ctx)
    out_root = ctx.stage_dir("C1b")
    out_root.mkdir(parents=True, exist_ok=True)

    peaks_paths, peak_qc, detect_errors = run_confirmatory_peak_detection(
        observations,
        ctx.dataset,
        ctx.master,
        out_root,
    )
    if not peaks_paths:
        raise StageError(
            "C1b peak detection wrote no detected_peaks.csv. "
            f"Errors: {detect_errors[:5]}"
        )

    written: list[str] = []
    errors: list[str] = list(detect_errors)
    by_obs = {
        path.parent.name: path for path in peaks_paths
    }
    for obs in observations:
        safe = safe_subject_dir_name(obs.observation_id)
        peaks = by_obs.get(safe) or (out_root / safe / PEAKS_FILENAME)
        if not Path(peaks).is_file():
            errors.append(f"{obs.observation_id}: detected_peaks.csv missing after detection")
            continue
        obs_out = _obs_dir(out_root, obs.observation_id)
        try:
            reconstruct_instant_hr_file(peaks, obs_out)
            written.append(obs.observation_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{obs.observation_id}: {type(exc).__name__}: {exc}")

    if not written:
        raise StageError(
            "C1b reconstructed no instantaneous-HR features. "
            f"Errors: {errors[:5]}"
        )
    return {
        "n_ok": len(written),
        "n_peaks": len(peaks_paths),
        "n_peak_qc": len(peak_qc),
        "n_error": len(errors),
        "errors": errors[:20],
    }


def _load_instant_hr_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    if not rows:
        raise StageError(f"Empty instant-HR table: {path}")
    time_s = np.asarray([float(r["time_s"]) for r in rows], dtype=float)
    hr = np.asarray(
        [
            float(r["instant_hr_bpm"])
            if str(r.get("instant_hr_bpm", "")).strip()
            else float("nan")
            for r in rows
        ],
        dtype=float,
    )
    valid = np.asarray(
        [
            str(r.get("is_valid_hr", "")).strip().casefold() in {"1", "true", "yes"}
            for r in rows
        ],
        dtype=bool,
    )
    return time_s, hr, valid


def _load_multitaper_maps(
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    rows = _read_csv(path)
    median_rows = [
        r
        for r in rows
        if str(r.get("channel", "")).strip() == ROBUST_MEDIAN_CHANNEL
        or str(r.get("aggregation", "")).strip().casefold() == "robust_median"
    ]
    if not median_rows:
        # Fall back: average channel rows per window_center if median absent
        by_time: dict[float, list[dict[str, str]]] = {}
        for row in rows:
            try:
                t = float(row.get("window_center_s") or row.get("time_s") or "nan")
            except ValueError:
                continue
            if not math.isfinite(t):
                continue
            by_time.setdefault(t, []).append(row)
        times = np.asarray(sorted(by_time), dtype=float)
        absolute: dict[str, list[float]] = {band: [] for band in BAND_ORDER}
        abs_log: dict[str, list[float]] = {band: [] for band in BAND_ORDER}
        for t in times:
            band_vals = {band: [] for band in BAND_ORDER}
            band_logs = {band: [] for band in BAND_ORDER}
            for row in by_time[float(t)]:
                band = str(row.get("band", "")).strip().casefold()
                if band not in BAND_ORDER:
                    continue
                try:
                    band_vals[band].append(float(row["absolute_power"]))
                    band_logs[band].append(float(row["absolute_log10_power"]))
                except (KeyError, ValueError):
                    continue
            for band in BAND_ORDER:
                absolute[band].append(
                    float(np.nanmedian(band_vals[band])) if band_vals[band] else float("nan")
                )
                abs_log[band].append(
                    float(np.nanmedian(band_logs[band])) if band_logs[band] else float("nan")
                )
        abs_arr = {b: np.asarray(v, dtype=float) for b, v in absolute.items()}
        log_arr = {b: np.asarray(v, dtype=float) for b, v in abs_log.items()}
        valid = np.all(
            np.vstack([np.isfinite(log_arr[b]) for b in BAND_ORDER]), axis=0
        )
        return times, abs_arr, log_arr, valid

    # Robust-median long-form: one row per band × time
    by_time: dict[float, dict[str, dict[str, float]]] = {}
    for row in median_rows:
        try:
            t = float(row.get("window_center_s") or row.get("time_s") or "nan")
        except ValueError:
            continue
        band = str(row.get("band", "")).strip().casefold()
        if band not in BAND_ORDER or not math.isfinite(t):
            continue
        by_time.setdefault(t, {})[band] = {
            "absolute_power": float(row["absolute_power"]),
            "absolute_log10_power": float(row["absolute_log10_power"]),
        }
    times = np.asarray(sorted(by_time), dtype=float)
    absolute = {band: np.full(times.shape, np.nan, dtype=float) for band in BAND_ORDER}
    abs_log = {band: np.full(times.shape, np.nan, dtype=float) for band in BAND_ORDER}
    for index, t in enumerate(times):
        payload = by_time[float(t)]
        for band in BAND_ORDER:
            if band not in payload:
                continue
            absolute[band][index] = payload[band]["absolute_power"]
            abs_log[band][index] = payload[band]["absolute_log10_power"]
    valid = np.all(np.vstack([np.isfinite(abs_log[b]) for b in BAND_ORDER]), axis=0)
    return times, absolute, abs_log, valid


def run_c1c(ctx: StageContext) -> dict[str, object]:
    c1a = ctx.stage_dir("C1a")
    c1b = ctx.stage_dir("C1b")
    out = ctx.stage_dir("C1c")
    out.mkdir(parents=True, exist_ok=True)

    obs_by_safe = {
        safe_subject_dir_name(obs.observation_id): obs
        for obs in _load_observations(ctx)
    }

    obs_ids = sorted(
        {
            p.parent.name
            for p in c1b.rglob(INSTANT_HR_FEATURES)
            if p.is_file()
        }
    )
    if not obs_ids:
        raise StageError(f"C1c found no {INSTANT_HR_FEATURES} under {c1b}")

    combined_by_duration: dict[int, list[dict[str, object]]] = {
        240: [],
        180: [],
        120: [],
        60: [],
    }
    n_ok = 0
    errors: list[str] = []
    for obs_id in obs_ids:
        hr_path = c1b / obs_id / INSTANT_HR_FEATURES
        eeg_path = c1a / obs_id / MULTITAPER_FEATURES
        if not hr_path.is_file() or not eeg_path.is_file():
            errors.append(f"{obs_id}: missing HR or multitaper features")
            continue
        try:
            hr_time, hr_bpm, hr_valid = _load_instant_hr_series(hr_path)
            eeg_time, absolute, abs_log, eeg_valid = _load_multitaper_maps(eeg_path)
            hr_rows = _read_csv(hr_path)
            eeg_rows = _read_csv(eeg_path)
            discovered = obs_by_safe.get(obs_id)
            identity = {
                "dataset_id": (
                    (discovered.dataset_id if discovered is not None else "")
                    or hr_rows[0].get("dataset_id", "")
                    or eeg_rows[0].get("dataset_id", ctx.dataset.dataset_id)
                ),
                "subject_id": (
                    (discovered.subject_id if discovered is not None else "")
                    or hr_rows[0].get("subject_id", "")
                    or eeg_rows[0].get("subject_id", "")
                ),
                "task": (
                    (discovered.task_label if discovered is not None else "")
                    or hr_rows[0].get("task", "")
                    or eeg_rows[0].get("task", "")
                ),
                # Prefer protocol condition labels (e.g. ph_pre_rest) over task-only
                # HR rows, which currently omit condition.
                "condition": (
                    (discovered.condition_label if discovered is not None else "")
                    or eeg_rows[0].get("condition", "")
                    or hr_rows[0].get("condition", "")
                ),
                "observation_id": (
                    (discovered.observation_id if discovered is not None else "")
                    or hr_rows[0].get("observation_id", "")
                    or eeg_rows[0].get("observation_id", obs_id)
                ),
            }
            result = harmonize_observation(
                hr_time_s=hr_time,
                hr_bpm=hr_bpm,
                hr_valid=hr_valid,
                eeg_time_s=eeg_time,
                eeg_absolute_power=absolute,
                eeg_absolute_log10_power=abs_log,
                eeg_valid=eeg_valid,
                identity=identity,
            )
            write_harmonize_outputs(result, out / obs_id)
            for duration, rows in result.features_by_duration.items():
                combined_by_duration[int(duration)].extend(rows)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{obs_id}: {type(exc).__name__}: {exc}")

    # Aggregate duration tables for C2+
    for duration, rows in combined_by_duration.items():
        path = out / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    if n_ok == 0:
        raise StageError(f"C1c produced no harmonized observations. Errors: {errors[:8]}")
    return {"n_ok": n_ok, "n_error": len(errors), "errors": errors[:20]}


def run_c2(ctx: StageContext) -> dict[str, object]:
    aligned = ctx.stage_dir("C1c")
    out = ctx.stage_dir("C2")
    condition_by_observation: dict[str, str] = {}
    for obs in _load_observations(ctx):
        condition = str(obs.condition_label).strip()
        if not condition:
            continue
        condition_by_observation[obs.observation_id] = condition
        condition_by_observation[obs.observation_id.casefold()] = condition
        condition_by_observation[safe_subject_dir_name(obs.observation_id)] = condition
    results = run_confirmatory_correlations(
        aligned,
        out,
        condition_by_observation=condition_by_observation,
    )
    return {"durations": sorted(results)}


def run_c3(ctx: StageContext) -> dict[str, object]:
    curves = ctx.stage_dir("C2")
    out = ctx.stage_dir("C3")
    endpoints = run_confirmatory_endpoints(curves, out)
    run_confirmatory_peak_fits(curves, out)
    return {"endpoint_durations": sorted(endpoints)}


def run_c4(ctx: StageContext) -> dict[str, object]:
    aligned = ctx.stage_dir("C1c")
    out = ctx.stage_dir("C4")
    run_confirmatory_nulls(
        aligned,
        out,
        n_surrogates=ctx.n_surrogates,
        durations=(240,),
        n_jobs=ctx.n_jobs,
        progress=True,
    )
    return {"n_surrogates": ctx.n_surrogates, "n_jobs": ctx.n_jobs}


def run_c5(ctx: StageContext) -> dict[str, object]:
    endpoints = ctx.stage_dir("C3")
    out = ctx.stage_dir("C5")
    run_confirmatory_group_tables(
        endpoints,
        out,
        peaks_dir=endpoints,
        dataset_ids=(ctx.dataset.dataset_id,),
    )
    return {"group_tables": str(out)}


def run_c6(ctx: StageContext) -> dict[str, object]:
    group = ctx.stage_dir("C5")
    out = ctx.stage_dir("C6")
    run_confirmatory_inference_from_dir(group, out)
    run_confirmatory_artifact_controls(
        group,
        out,
        enable_optional_artifact_controls=bool(ctx.enable_optional_artifact_controls),
    )
    return {"inference_and_sensitivities": str(out)}


def run_c7(ctx: StageContext) -> dict[str, object]:
    publish = ctx.stage_dir("C7") / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    for stage in ("C0", "C2", "C3", "C4", "C5", "C6"):
        src = ctx.stage_dir(stage)
        if not src.is_dir():
            continue
        for path in src.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                # Flatten top-level stage products into publish/
                if path.parent == src or path.parent.name in {
                    "C0",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                }:
                    shutil.copy2(path, publish / path.name)
                elif stage == "C1c" and path.name.startswith("features_confirmatory"):
                    shutil.copy2(path, publish / path.name)
    # Ensure C1c aligned tables are present for figures
    for path in ctx.stage_dir("C1c").glob("features_confirmatory_aligned_D*.csv"):
        shutil.copy2(path, publish / path.name)
    for path in ctx.stage_dir("C2").glob("*.csv"):
        shutil.copy2(path, publish / path.name)
    # Figure 1 Panel C requires observation-level endpoint metrics (C3).
    for path in ctx.stage_dir("C3").glob("confirmatory_endpoint_metrics_D*.csv"):
        shutil.copy2(path, publish / path.name)
    for path in ctx.stage_dir("C3").glob("confirmatory_endpoint_qc_D*.csv"):
        shutil.copy2(path, publish / path.name)

    report_dir = ctx.stage_dir("C7")
    paths = run_confirmatory_reporting(
        publish,
        report_dir,
        config_paths=[ctx.master_path, ctx.dataset_path],
        repo_root=ctx.repo_root,
        seeds={
            "root_seed": ctx.master.root_seed,
            "n_surrogates": ctx.n_surrogates,
        },
    )
    return {key: str(value) for key, value in paths.items() if value is not None}


STAGE_RUNNERS: dict[str, Callable[[StageContext], Mapping[str, object]]] = {
    "C0": run_c0,
    "C1a": run_c1a,
    "C1b": run_c1b,
    "C1c": run_c1c,
    "C2": run_c2,
    "C3": run_c3,
    "C4": run_c4,
    "C5": run_c5,
    "C6": run_c6,
    "C7": run_c7,
}


def expand_stages(stage: str) -> list[str]:
    key = stage.strip()
    if key.casefold() == "all":
        return list(STAGE_ORDER)
    mapping = {name.casefold(): name for name in STAGE_ORDER}
    normalized = mapping.get(key.casefold())
    if normalized is None:
        raise ValueError(f"Unknown stage {stage!r}. Allowed: {VALID_STAGES}")
    return [normalized]


def run_stages(ctx: StageContext, stages: Sequence[str]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for stage in stages:
        _require_stages(ctx, stage)
        started = _utc_now()
        print(f"[confirmatory] stage={stage} start={started}")
        try:
            payload = dict(STAGE_RUNNERS[stage](ctx))
            _write_stage_status(ctx, stage, "ok", json.dumps(payload, default=str)[:500])
            print(f"[confirmatory] stage={stage} status=ok")
            results.append({"stage": stage, "status": "ok", "result": payload})
        except Exception as exc:
            _write_stage_status(ctx, stage, "error", f"{type(exc).__name__}: {exc}")
            print(f"[confirmatory] stage={stage} status=error: {exc}", file=sys.stderr)
            raise
    return results


def resolve_master_and_dataset(
    dataset_config: str | Path,
    *,
    master_config: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> tuple[ConfirmatoryMasterConfig, ConfirmatoryDatasetConfig, Path, Path]:
    dataset_path = Path(dataset_config).expanduser().resolve()
    if master_config is not None:
        master_path = Path(master_config).expanduser().resolve()
    else:
        root = discover_config_dir(config_dir)
        master_path = master_config_path(root)
    master = load_master_config(master_path)
    dataset = load_dataset_config(dataset_path, master=master)
    return master, dataset, master_path, dataset_path


def build_stage_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confirmatory Zero-Lag stages C0–C7 (and production modes)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset confirmatory YAML (required for --stage).",
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional master.yaml override (default: zero-lag-reanalysis-repo/master.yaml).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Confirmatory config root (for master discovery / production modes).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="C0,C1a,C1b,C1c,C2,C3,C4,C5,C6,C7, or all",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=(
            "preflight",
            "smoke",
            "primary",
            "sensitivity",
            "clean_root",
            "qc-report",
        ),
        help=(
            "Production ops mode (M13a), or qc-report for read-only QC under "
            "output_root/QC/. Mutually exclusive with --stage. "
            "qc-report is not part of STAGE_ORDER / --stage all."
        ),
    )
    parser.add_argument(
        "--qc-root",
        type=str,
        default=None,
        help="Optional override output directory for --mode qc-report (default: output_root/QC).",
    )
    parser.add_argument(
        "--n-surrogates",
        type=int,
        default=None,
        help=(
            "Null surrogates for C4. Defaults to dataset YAML n_surrogates when "
            f"set, otherwise {DEFAULT_N_SURROGATES} (production). Smoke configs "
            f"set n_surrogates: {SMOKE_N_SURROGATES}."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Process-pool workers for C4 null battery. "
            "-1 uses all CPUs; 1 forces serial execution."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip prerequisite stage-directory checks.",
    )
    parser.add_argument(
        "--optional-artifact-controls",
        action="store_true",
        help=(
            "Enable optional dataset-conditional C6 artifact controls "
            "(cardiac-field/QRS; motion/EOG/EMG; respiration; mean HR; "
            "beat count/density; eye state). Disabled by default and executed "
            "only when this flag is set and required signals are available. "
            "Default confirmatory robustness remains C4 temporal surrogate "
            "nulls plus C6 broadband residualization and duration sensitivity."
        ),
    )
    parser.add_argument(
        "--production-root",
        type=str,
        default=None,
        help="Override production output root for --mode.",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Git repo root for manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_stage_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode and args.stage:
        print("Pass only one of --mode or --stage.", file=sys.stderr)
        return 2

    if args.mode == "qc-report":
        if not args.config:
            parser.error("--config is required with --mode qc-report.")
        from .qc_report import run_qc_report_from_config

        try:
            run_qc_report_from_config(
                args.config,
                master_config=args.master,
                config_dir=args.config_dir,
                qc_root=args.qc_root,
                repo_root=args.repo_root,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[confirmatory] STOP: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"[confirmatory] error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.mode:
        from .production import main as production_main

        prod_argv = ["--mode", args.mode]
        if args.config_dir:
            prod_argv.extend(["--config-dir", args.config_dir])
        if args.production_root:
            prod_argv.extend(["--production-root", args.production_root])
        if args.repo_root:
            prod_argv.extend(["--repo-root", args.repo_root])
        return production_main(prod_argv)

    if not args.stage:
        parser.error(
            "Provide --stage C0|…|all (or --mode for production ops / qc-report)."
        )
    if not args.config:
        parser.error("--config is required with --stage.")

    try:
        stages = expand_stages(args.stage)
        master, dataset, master_path, dataset_path = resolve_master_and_dataset(
            args.config,
            master_config=args.master,
            config_dir=args.config_dir,
        )
        repo_root = (
            Path(args.repo_root).expanduser().resolve()
            if args.repo_root
            else discover_config_dir(args.config_dir).parent
        )
        ctx = StageContext(
            master=master,
            dataset=dataset,
            master_path=master_path,
            dataset_path=dataset_path,
            n_surrogates=(
                int(args.n_surrogates)
                if args.n_surrogates is not None
                else (
                    int(dataset.n_surrogates)
                    if dataset.n_surrogates is not None
                    else DEFAULT_N_SURROGATES
                )
            ),
            n_jobs=int(args.n_jobs),
            force=bool(args.force),
            enable_optional_artifact_controls=bool(args.optional_artifact_controls),
            repo_root=repo_root,
        )
        print(f"[confirmatory] dataset={dataset.dataset_id}")
        print(f"[confirmatory] output_root={dataset.output_root}")
        print(f"[confirmatory] stages={stages}")
        run_stages(ctx, stages)
    except (StageError, ValueError, FileNotFoundError) as exc:
        print(f"[confirmatory] STOP: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[confirmatory] error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "STAGE_ORDER",
    "STAGE_REQUIRES",
    "VALID_STAGES",
    "StageContext",
    "StageError",
    "expand_stages",
    "main",
    "resolve_master_and_dataset",
    "run_stages",
]
