from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from ..datasets import CanonicalObservation, build_observations
from .config import TemporalCouplingConfig
from .cardiac_timeseries import run_stage1b
from .cross_correlation import run_stage2
from .data_audit import run_stage0_audit
from .events import run_stage4
from .group_summary import run_stage3
from .eeg_envelope import run_stage1a
from .resample import run_stage1c


@dataclass(frozen=True)
class ResolvedObservations:
    observations: tuple[CanonicalObservation, ...]
    skipped_subjects: tuple[str, ...]
    warnings: tuple[str, ...]


def _stages_to_run(stage: str) -> list[str]:
    if stage == "all":
        return ["0", "1a", "1b", "1c", "2", "3", "4"]
    if stage == "1":
        return ["1a", "1b", "1c"]
    return [stage]


def _missing_pair_reason(
    raw_root: Path,
    dataset_id: str,
    subject_id: str,
    tasks: list[str],
) -> str:
    dataset_root = raw_root
    if dataset_root.name != dataset_id:
        candidate = raw_root / dataset_id
        if candidate.exists():
            dataset_root = candidate

    if not (dataset_root / subject_id).is_dir():
        return f"subject directory not found under {dataset_root}"

    task_labels = tasks or ["rest"]
    missing_parts: list[str] = []
    for task in task_labels:
        eeg_glob = list((dataset_root / subject_id / "eeg").glob(f"*_task-{task}_eeg.set"))
        ecg_glob = list((dataset_root / subject_id / "ecg").glob(f"*_task-{task}_ecg.set"))
        if not eeg_glob:
            missing_parts.append(f"EEG rest file missing for task={task!r}")
        if not ecg_glob:
            missing_parts.append(f"ECG/PPG rest file missing for task={task!r}")
    if missing_parts:
        return "; ".join(missing_parts)
    return "no matching EEG+ECG observation for configured filters"


def resolve_observations(cfg: TemporalCouplingConfig) -> ResolvedObservations:
    configured_subjects = list(cfg.subjects)
    warnings_out: list[str] = []

    try:
        observations = build_observations(
            cfg.dataset_id,
            cfg.paths.raw_root,
            subjects=cfg.subjects or None,
            tasks=cfg.tasks or None,
            conditions=cfg.conditions or None,
            sessions=cfg.sessions or None,
        )
    except FileNotFoundError as exc:
        message = f"[temporal_coupling] dataset root not found: {exc}"
        warnings.warn(message, stacklevel=2)
        return ResolvedObservations(observations=(), skipped_subjects=tuple(configured_subjects), warnings=(message,))

    found_subjects = {obs.subject_id for obs in observations}
    skipped: list[str] = []

    if configured_subjects:
        for subject_id in configured_subjects:
            if subject_id in found_subjects:
                continue
            skipped.append(subject_id)
            reason = _missing_pair_reason(cfg.paths.raw_root, cfg.dataset_id, subject_id, cfg.tasks)
            message = (
                f"[temporal_coupling] skipping {subject_id}: {reason}. "
                "Configure subjects with paired EEG and ECG/PPG files."
            )
            warnings.warn(message, stacklevel=2)
            warnings_out.append(message)
    elif not observations:
        message = (
            f"[temporal_coupling] no observations matched dataset={cfg.dataset_id!r} "
            f"under raw_root={cfg.paths.raw_root}."
        )
        warnings.warn(message, stacklevel=2)
        warnings_out.append(message)

    return ResolvedObservations(
        observations=tuple(observations),
        skipped_subjects=tuple(skipped),
        warnings=tuple(warnings_out),
    )


def _run_stage(
    stage: str,
    cfg: TemporalCouplingConfig,
    resolved: ResolvedObservations,
) -> None:
    if stage == "0":
        run_stage0_audit(cfg)
        return
    if stage == "1a":
        run_stage1a(cfg)
        return
    if stage == "1b":
        run_stage1b(cfg)
        return
    if stage == "1c":
        run_stage1c(cfg)
        return
    if stage == "2":
        run_stage2(cfg)
        return
    if stage == "3":
        run_stage3(cfg)
        return
    if stage == "4":
        run_stage4(cfg)
        return

    n_obs = len(resolved.observations)
    print(f"[temporal_coupling] stage={stage} (skeleton — no computation yet)")
    print(
        f"[temporal_coupling] dataset={cfg.dataset_id} observations={n_obs} "
        f"skipped_subjects={len(resolved.skipped_subjects)}"
    )
    if n_obs == 0:
        print("[temporal_coupling] no usable observations; stage skipped safely.")


def run_temporal_coupling(cfg: TemporalCouplingConfig, *, stage: str) -> None:
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[temporal_coupling] requested stage={stage!r}")
    print(f"[temporal_coupling] loaded config: dataset_id={cfg.dataset_id!r}")
    print(f"[temporal_coupling] raw_root={cfg.paths.raw_root}")
    print(f"[temporal_coupling] out_root={cfg.paths.out_root}")

    resolved = resolve_observations(cfg)
    if resolved.observations:
        subject_ids = sorted({obs.subject_id for obs in resolved.observations})
        print(f"[temporal_coupling] usable subjects ({len(subject_ids)}): {', '.join(subject_ids)}")
    else:
        print("[temporal_coupling] usable subjects (0): none")

    for step in _stages_to_run(stage):
        _run_stage(step, cfg, resolved)

    if stage == "all":
        print("[temporal_coupling] completed all requested stages.")
    elif stage in {"0", "1a", "1b", "1c", "1", "2", "3", "4"}:
        print(f"[temporal_coupling] completed stage={stage!r}.")
    else:
        print(f"[temporal_coupling] completed stage={stage!r} (skeleton).")
