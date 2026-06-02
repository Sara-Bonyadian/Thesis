from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep CPU-heavy numeric libraries single-threaded by default to reduce
# thermal/memory pressure on laptops during long batch runs.
THREAD_LIMIT_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _env_name in THREAD_LIMIT_ENV_VARS:
    os.environ.setdefault(_env_name, "1")

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from .config import PipelineConfig, load_config
from .correlation import apply_fdr, compute_pairwise_correlations, compute_trend_agreement
from .datasets import build_observations
from .features_core import (
    OBSERVATION_KEY_COLUMNS,
    add_dataset_local_robust_zscores,
    base_eeg_power_columns,
    base_ppg_ibi_columns,
)


@dataclass(frozen=True)
class BatchPlan:
    index: int
    subjects: list[str]
    output_dir: Path
    config_path: Path

    @property
    def done_marker(self) -> Path:
        return self.output_dir / "_done.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_from_repo(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _safe_relative_or_absolute(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _drop_duplicates_if_possible(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    usable = [col for col in columns if col in df.columns]
    if not usable:
        return df
    return df.drop_duplicates(subset=usable, keep="last").reset_index(drop=True)


def _drop_robust_z_columns(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    to_drop: list[str] = []
    for feature_name in feature_names:
        rz_col = f"{feature_name}_rz"
        if rz_col in df.columns:
            to_drop.append(rz_col)
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def _discover_subjects(cfg: PipelineConfig, dataset_id: str) -> list[str]:
    observations = build_observations(
        dataset_id,
        cfg.paths.raw_root,
        subjects=cfg.subjects,
        tasks=cfg.tasks,
        conditions=cfg.conditions,
        sessions=cfg.sessions,
    )
    usable_subjects = sorted({obs.subject_id for obs in observations if bool(obs.is_usable)})
    if not usable_subjects:
        raise RuntimeError(f"No usable observations found for dataset_id={dataset_id!r}.")
    return usable_subjects


def _build_batch_config_data(
    *,
    source_config_data: dict[str, Any],
    subjects: list[str],
    batch_output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    batch_data = copy.deepcopy(source_config_data)
    batch_data["subjects"] = subjects
    batch_data.setdefault("features", {})
    # Batch runs should be deterministic and independent.
    batch_data["features"]["reuse_eeg_features_csv"] = False
    batch_data["features"]["reuse_ppg_features_csv"] = False
    batch_data.setdefault("paths", {})
    batch_data["paths"]["out_root"] = _safe_relative_or_absolute(batch_output_dir, repo_root)
    return batch_data


def _required_dataset_files(cfg: PipelineConfig) -> list[str]:
    required = [
        "features_core_eeg.csv",
        "features_core_ppg.csv",
        "features_core_merged.csv",
        "features_base_eeg_power.csv",
        "features_base_ppg_ibi.csv",
        "correlations_raw.csv",
        "correlations_fdr.csv",
    ]
    if cfg.output.save_observation_index:
        required.append("observations_index.csv")
    return required


def _run_single_batch(
    *,
    batch: BatchPlan,
    dataset_id: str,
    cfg: PipelineConfig,
    repo_root: Path,
) -> None:
    cache_root = repo_root / ".cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env.setdefault("MPLCONFIGDIR", str((cache_root / "matplotlib").resolve()))
    child_env.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
    for env_name in THREAD_LIMIT_ENV_VARS:
        child_env.setdefault(env_name, "1")

    cmd = [sys.executable, "-m", "ppg_eeg.run", "--config", str(batch.config_path.resolve())]
    subprocess.run(cmd, cwd=str(repo_root), env=child_env, check=True)

    dataset_dir = batch.output_dir / dataset_id
    missing_files = [name for name in _required_dataset_files(cfg) if not (dataset_dir / name).exists()]
    if missing_files:
        missing_list = ", ".join(sorted(missing_files))
        raise RuntimeError(
            f"Batch {batch.index:03d} finished but expected outputs are missing: {missing_list}"
        )

    marker_payload = {
        "batch_index": batch.index,
        "subject_count": len(batch.subjects),
        "subjects": batch.subjects,
        "dataset_id": dataset_id,
        "created_at": _utc_now_iso(),
    }
    batch.done_marker.parent.mkdir(parents=True, exist_ok=True)
    batch.done_marker.write_text(json.dumps(marker_payload, indent=2))


def _completed_batch_indices(batches: list[BatchPlan]) -> list[int]:
    return [batch.index for batch in batches if batch.done_marker.exists()]


def _write_state(
    *,
    path: Path,
    dataset_id: str,
    config_path: Path,
    batch_size: int,
    sleep_seconds: float,
    total_subjects: int,
    total_batches: int,
    completed_batches: list[int],
    status: str,
) -> None:
    payload = {
        "dataset_id": dataset_id,
        "config_path": str(config_path.resolve()),
        "batch_size": batch_size,
        "sleep_seconds": sleep_seconds,
        "total_subjects": total_subjects,
        "total_batches": total_batches,
        "completed_batches": completed_batches,
        "status": status,
        "updated_at": _utc_now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _merge_completed_batches(
    *,
    batches: list[BatchPlan],
    dataset_id: str,
    cfg: PipelineConfig,
    final_out_root: Path,
) -> dict[str, int]:
    obs_frames: list[pd.DataFrame] = []
    eeg_frames: list[pd.DataFrame] = []
    ppg_frames: list[pd.DataFrame] = []
    base_frames: list[pd.DataFrame] = []
    ppg_ibi_frames: list[pd.DataFrame] = []

    for batch in batches:
        dataset_dir = batch.output_dir / dataset_id
        if cfg.output.save_observation_index:
            obs_df = _safe_read_csv(dataset_dir / "observations_index.csv")
            if not obs_df.empty:
                obs_frames.append(obs_df)

        eeg_df = _safe_read_csv(dataset_dir / "features_core_eeg.csv")
        ppg_df = _safe_read_csv(dataset_dir / "features_core_ppg.csv")
        base_df = _safe_read_csv(dataset_dir / "features_base_eeg_power.csv")
        ppg_ibi_df = _safe_read_csv(dataset_dir / "features_base_ppg_ibi.csv")

        if not eeg_df.empty:
            eeg_frames.append(eeg_df)
        if not ppg_df.empty:
            ppg_frames.append(ppg_df)
        if not base_df.empty:
            base_frames.append(base_df)
        if not ppg_ibi_df.empty:
            ppg_ibi_frames.append(ppg_ibi_df)

    obs_all = pd.concat(obs_frames, ignore_index=True) if obs_frames else pd.DataFrame()
    eeg_all = pd.concat(eeg_frames, ignore_index=True) if eeg_frames else pd.DataFrame()
    ppg_all = pd.concat(ppg_frames, ignore_index=True) if ppg_frames else pd.DataFrame()
    base_all = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()
    ppg_ibi_all = pd.concat(ppg_ibi_frames, ignore_index=True) if ppg_ibi_frames else pd.DataFrame()

    obs_all = _drop_duplicates_if_possible(obs_all, ["observation_id"])
    eeg_all = _drop_duplicates_if_possible(eeg_all, ["observation_id"])
    ppg_all = _drop_duplicates_if_possible(ppg_all, ["observation_id"])
    base_all = _drop_duplicates_if_possible(base_all, ["observation_id", "channel"])
    ppg_ibi_all = _drop_duplicates_if_possible(ppg_ibi_all, ["observation_id", "ibi_index"])

    selected_eeg = list(cfg.features.eeg)
    selected_ppg = list(cfg.features.ppg)

    eeg_all = _drop_robust_z_columns(eeg_all, selected_eeg)
    ppg_all = _drop_robust_z_columns(ppg_all, selected_ppg)

    if cfg.features.include_robust_z:
        if not eeg_all.empty:
            eeg_all = add_dataset_local_robust_zscores(eeg_all, selected_eeg)
        if not ppg_all.empty:
            ppg_all = add_dataset_local_robust_zscores(ppg_all, selected_ppg)

    merge_key_missing = any(col not in eeg_all.columns or col not in ppg_all.columns for col in OBSERVATION_KEY_COLUMNS)
    if eeg_all.empty or ppg_all.empty or merge_key_missing:
        merged_all = pd.DataFrame()
    else:
        ppg_cols: list[str] = [*OBSERVATION_KEY_COLUMNS, *selected_ppg]
        ppg_cols.extend(f"{name}_rz" for name in selected_ppg if f"{name}_rz" in ppg_all.columns)
        for extra_col in (
            "ppg_error",
            "ppg_channel",
            "ppg_segment_start_s",
            "ppg_segment_end_s",
            "n_ibi_clean",
        ):
            if extra_col in ppg_all.columns:
                ppg_cols.append(extra_col)
        ppg_cols = [col for col in dict.fromkeys(ppg_cols) if col in ppg_all.columns]
        merged_all = eeg_all.merge(ppg_all[ppg_cols], on=OBSERVATION_KEY_COLUMNS, how="inner")

    corr_raw = compute_pairwise_correlations(
        merged_all,
        eeg_features=selected_eeg,
        ppg_features=selected_ppg,
        cfg=cfg,
    )
    corr_fdr = apply_fdr(corr_raw, cfg=cfg)
    trend_df, trend_summary = compute_trend_agreement(corr_fdr, cfg=cfg)

    dataset_dir = final_out_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if cfg.output.save_observation_index:
        _write_csv(obs_all, dataset_dir / "observations_index.csv")
    _write_csv(eeg_all, dataset_dir / "features_core_eeg.csv")

    if base_all.empty:
        base_all = pd.DataFrame(columns=base_eeg_power_columns("export"))
    else:
        ordered_base_cols = base_eeg_power_columns("export")
        for col in ordered_base_cols:
            if col not in base_all.columns:
                base_all[col] = pd.NA
        base_all = base_all[ordered_base_cols]
    _write_csv(base_all, dataset_dir / "features_base_eeg_power.csv")

    _write_csv(ppg_all, dataset_dir / "features_core_ppg.csv")
    if ppg_ibi_all.empty:
        ppg_ibi_all = pd.DataFrame(columns=base_ppg_ibi_columns())
    else:
        ordered_ppg_ibi_cols = base_ppg_ibi_columns()
        for col in ordered_ppg_ibi_cols:
            if col not in ppg_ibi_all.columns:
                ppg_ibi_all[col] = pd.NA
        ppg_ibi_all = ppg_ibi_all[ordered_ppg_ibi_cols]
    _write_csv(ppg_ibi_all, dataset_dir / "features_base_ppg_ibi.csv")

    _write_csv(merged_all, dataset_dir / "features_core_merged.csv")
    _write_csv(corr_raw, dataset_dir / "correlations_raw.csv")
    _write_csv(corr_fdr, dataset_dir / "correlations_fdr.csv")

    cross_dir = final_out_root / "cross_dataset"
    cross_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(trend_df, cross_dir / "trend_agreement.csv")
    if cfg.output.save_summary_json:
        (cross_dir / "trend_agreement_summary.json").write_text(json.dumps(trend_summary, indent=2))

    return {
        "observations": int(len(obs_all)),
        "merged_rows": int(len(merged_all)),
        "corr_tests": int(len(corr_raw)),
    }


def _build_batches(
    *,
    subjects: list[str],
    batch_size: int,
    batch_root: Path,
    batch_config_dir: Path,
    config_data: dict[str, Any],
    repo_root: Path,
) -> list[BatchPlan]:
    total_batches = math.ceil(len(subjects) / batch_size)
    batches: list[BatchPlan] = []
    for index in range(1, total_batches + 1):
        start = (index - 1) * batch_size
        end = min(index * batch_size, len(subjects))
        batch_subjects = subjects[start:end]
        output_dir = batch_root / f"batch_{index:03d}"
        config_path = batch_config_dir / f"batch_{index:03d}.yaml"
        config_payload = _build_batch_config_data(
            source_config_data=config_data,
            subjects=batch_subjects,
            batch_output_dir=output_dir,
            repo_root=repo_root,
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
        batches.append(
            BatchPlan(
                index=index,
                subjects=batch_subjects,
                output_dir=output_dir,
                config_path=config_path,
            )
        )
    return batches


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run one-dataset EEG/PPG pipeline in resumable subject batches."
    )
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--batch-size", type=int, default=5, help="Subjects per batch (default: 5).")
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=20.0,
        help="Cooldown between batches in seconds (default: 20).",
    )
    ap.add_argument(
        "--stop-after-batch",
        type=int,
        default=0,
        help="Run only first N batches, then exit without merge.",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing batch done markers and rerun all requested batches.",
    )
    ap.add_argument(
        "--skip-merge",
        action="store_true",
        help="Run extraction batches only and skip final merged outputs.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be >= 0.")
    if args.stop_after_batch < 0:
        raise ValueError("--stop-after-batch must be >= 0.")

    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    cfg = load_config(config_path)
    config_data = yaml.safe_load(config_path.read_text())
    if not isinstance(config_data, dict):
        raise ValueError("Config file must parse to a mapping.")

    dataset_ids = list(dict.fromkeys(cfg.dataset_ids))
    if len(dataset_ids) != 1:
        joined = ", ".join(dataset_ids)
        raise ValueError(
            "Batched runner supports exactly one dataset_id per run. "
            f"Current config dataset_ids={joined!r}"
        )
    dataset_id = dataset_ids[0]

    final_out_root = _resolve_from_repo(Path(cfg.paths.out_root), repo_root)
    batch_root = final_out_root.parent / f"{final_out_root.name}_batches"
    batch_config_dir = batch_root / "_configs"
    state_path = batch_root / "_batch_state.json"
    batch_root.mkdir(parents=True, exist_ok=True)

    subjects = _discover_subjects(cfg, dataset_id)
    batches = _build_batches(
        subjects=subjects,
        batch_size=args.batch_size,
        batch_root=batch_root,
        batch_config_dir=batch_config_dir,
        config_data=config_data,
        repo_root=repo_root,
    )

    total_batches = len(batches)
    print(
        f"Batch plan: dataset={dataset_id} subjects={len(subjects)} "
        f"batch_size={args.batch_size} total_batches={total_batches}"
    )
    print(f"Batch outputs: {batch_root}")
    print(f"Final merged out_root: {final_out_root}")

    _write_state(
        path=state_path,
        dataset_id=dataset_id,
        config_path=config_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        total_subjects=len(subjects),
        total_batches=total_batches,
        completed_batches=_completed_batch_indices(batches),
        status="running",
    )

    resume_enabled = not args.no_resume

    for batch in batches:
        if args.stop_after_batch and batch.index > args.stop_after_batch:
            break

        if resume_enabled and batch.done_marker.exists():
            print(f"Skip batch {batch.index:03d}/{total_batches} (already complete).")
            continue

        print(
            f"Run batch {batch.index:03d}/{total_batches}: "
            f"{batch.subjects[0]}..{batch.subjects[-1]} ({len(batch.subjects)} subjects)"
        )
        _run_single_batch(
            batch=batch,
            dataset_id=dataset_id,
            cfg=cfg,
            repo_root=repo_root,
        )
        print(f"Done batch {batch.index:03d}/{total_batches}.")

        _write_state(
            path=state_path,
            dataset_id=dataset_id,
            config_path=config_path,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            total_subjects=len(subjects),
            total_batches=total_batches,
            completed_batches=_completed_batch_indices(batches),
            status="running",
        )

        if (
            args.sleep_seconds > 0
            and batch.index < total_batches
            and (not args.stop_after_batch or batch.index < args.stop_after_batch)
        ):
            print(f"Cooldown: sleeping {args.sleep_seconds:.1f}s before next batch...")
            time.sleep(args.sleep_seconds)

    if args.stop_after_batch and args.stop_after_batch < total_batches:
        _write_state(
            path=state_path,
            dataset_id=dataset_id,
            config_path=config_path,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            total_subjects=len(subjects),
            total_batches=total_batches,
            completed_batches=_completed_batch_indices(batches),
            status="paused_stop_after_batch",
        )
        print(
            f"Stopped after batch {args.stop_after_batch}. "
            "Re-run without --stop-after-batch to resume."
        )
        return

    if args.skip_merge:
        _write_state(
            path=state_path,
            dataset_id=dataset_id,
            config_path=config_path,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            total_subjects=len(subjects),
            total_batches=total_batches,
            completed_batches=_completed_batch_indices(batches),
            status="complete_batches_only",
        )
        print("Batch extraction completed (merge skipped by --skip-merge).")
        return

    missing_for_merge = [batch.index for batch in batches if not batch.done_marker.exists()]
    if missing_for_merge:
        missing_text = ", ".join(str(i) for i in missing_for_merge)
        raise RuntimeError(
            "Cannot merge because some batches are incomplete. "
            f"Missing batch indices: {missing_text}"
        )

    print("Merging completed batch outputs...")
    summary = _merge_completed_batches(
        batches=batches,
        dataset_id=dataset_id,
        cfg=cfg,
        final_out_root=final_out_root,
    )

    _write_state(
        path=state_path,
        dataset_id=dataset_id,
        config_path=config_path,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        total_subjects=len(subjects),
        total_batches=total_batches,
        completed_batches=_completed_batch_indices(batches),
        status="complete_merged",
    )
    print(
        "Merged outputs complete: "
        f"observations={summary['observations']} "
        f"merged_rows={summary['merged_rows']} "
        f"corr_tests={summary['corr_tests']}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user. Re-run the same command to resume.")
        raise SystemExit(130)
