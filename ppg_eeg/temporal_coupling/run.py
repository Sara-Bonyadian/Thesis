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
from .hiit_combine import run_stage1d
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
    if stage == "prepost":
        return ["1d", "2", "3", "4"]
    return [stage]


def _normalize_subject_key(subject_id: str) -> str:
    key = subject_id.strip().casefold()
    if key.startswith("sub-"):
        return key[4:]
    return key


def _observation_matches_configured_subject(
    configured_subject: str,
    observation: CanonicalObservation,
) -> bool:
    configured_key = _normalize_subject_key(configured_subject)
    if not configured_key:
        return False

    obs_subject_key = _normalize_subject_key(observation.subject_id.split("_", 1)[0])
    if obs_subject_key == configured_key:
        return True

    bids_folder = f"sub-{configured_key}"
    return any(part.casefold() == bids_folder for part in observation.eeg_path.parts)


def _dataset_root(raw_root: Path, dataset_id: str) -> Path:
    dataset_root = raw_root
    if dataset_root.name != dataset_id:
        candidate = raw_root / dataset_id
        if candidate.exists():
            dataset_root = candidate
    return dataset_root


def _subject_dir(dataset_root: Path, subject_id: str) -> Path | None:
    key = subject_id.strip()
    candidates = [dataset_root / key]
    normalized = _normalize_subject_key(key)
    candidates.append(dataset_root / f"sub-{normalized}")
    seen: set[str] = set()
    for path in candidates:
        marker = str(path.casefold())
        if marker in seen:
            continue
        seen.add(marker)
        if path.is_dir():
            return path
    return None


def _missing_pair_reason(
    raw_root: Path,
    dataset_id: str,
    subject_id: str,
    tasks: list[str],
) -> str:
    dataset_root = _dataset_root(raw_root, dataset_id)
    subject_dir = _subject_dir(dataset_root, subject_id)
    if subject_dir is None:
        return f"subject directory not found under {dataset_root}"

    task_labels = tasks or ["rest"]
    missing_parts: list[str] = []
    for task in task_labels:
        task_key = task.casefold()
        eeg_files = [
            path
            for path in subject_dir.glob("**/eeg/*_eeg.*")
            if task_key in path.stem.casefold()
        ]
        ecg_files = [
            path
            for path in subject_dir.glob("**/ecg/*_ecg.*")
            if task_key in path.stem.casefold()
        ]
        if not eeg_files:
            missing_parts.append(f"EEG file missing for task={task!r}")
        elif not ecg_files:
            embedded_candidates = [
                path
                for path in eeg_files
                if path.suffix.casefold() in {".edf", ".vhdr", ".set"}
            ]
            if not embedded_candidates:
                missing_parts.append(f"ECG/PPG file missing for task={task!r}")
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
            hiit_partition_mode=cfg.hiit_partition_mode,
        )
    except FileNotFoundError as exc:
        message = f"[temporal_coupling] dataset root not found: {exc}"
        warnings.warn(message, stacklevel=2)
        return ResolvedObservations(observations=(), skipped_subjects=tuple(configured_subjects), warnings=(message,))

    skipped: list[str] = []

    if configured_subjects:
        for subject_id in configured_subjects:
            if any(
                _observation_matches_configured_subject(subject_id, obs)
                for obs in observations
            ):
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
    if stage == "1d":
        run_stage1d(cfg)
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
        n_obs = len(resolved.observations)
        subject_ids = sorted({obs.subject_id for obs in resolved.observations})
        task_labels = sorted({obs.task_label for obs in resolved.observations})
        print(
            f"[temporal_coupling] resolved observations: {n_obs} "
            f"({len(subject_ids)} subject-session-run keys, tasks={', '.join(task_labels)})"
        )
    else:
        print("[temporal_coupling] resolved observations: 0")

    for step in _stages_to_run(stage):
        _run_stage(step, cfg, resolved)

    if stage == "all":
        print("[temporal_coupling] completed all requested stages.")
    elif stage in {"0", "1a", "1b", "1c", "1d", "1", "2", "3", "4", "prepost"}:
        print(f"[temporal_coupling] completed stage={stage!r}.")
    else:
        print(f"[temporal_coupling] completed stage={stage!r} (skeleton).")
