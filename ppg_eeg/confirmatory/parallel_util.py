"""Shared helpers for confirmatory stage parallelization (C1a/C1b/C4-style)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve worker count: ``None``/``-1`` → all CPUs; ``>=1`` → that many."""
    if n_jobs is None or int(n_jobs) == -1:
        return max(1, int(os.cpu_count() or 1))
    resolved = int(n_jobs)
    if resolved < 1:
        raise ValueError("n_jobs must be >= 1 or -1 (all CPUs).")
    return resolved


def total_ram_bytes() -> int | None:
    """Best-effort physical RAM size in bytes."""
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def eeg_payload_bytes(eeg_path: str | Path) -> int:
    """Largest on-disk payload beside an EEG header (``.eeg`` / ``.fif`` / path)."""
    path = Path(eeg_path).expanduser()
    candidates = [
        path.with_suffix(".eeg"),
        path.with_suffix(".fif"),
        path.with_suffix(".edf"),
        path.with_suffix(".bdf"),
        path,
    ]
    sizes = [c.stat().st_size for c in candidates if c.is_file()]
    return max(sizes) if sizes else 0


def estimate_c1a_mem_per_worker_gb(
    eeg_paths: Sequence[str | Path] | None = None,
    *,
    default_gb: float = 2.5,
    max_gb: float = 16.0,
) -> float:
    """Estimate peak RAM per C1a worker from on-disk EEG size.

    BrainVision/EEGLAB payloads are often int16 on disk but loaded as float64
    (≈4×), and ``preprocess_eeg`` copies the Raw (another ≈2× transient peak).
    Use an 8× multiplier of the largest payload, floored at ``default_gb``.
    """
    max_bytes = 0
    for raw_path in eeg_paths or ():
        max_bytes = max(max_bytes, eeg_payload_bytes(raw_path))
    if max_bytes <= 0:
        return float(default_gb)
    file_gb = float(max_bytes) / float(1024**3)
    return float(min(max_gb, max(default_gb, 8.0 * file_gb)))


def resolve_c1a_n_jobs(
    n_jobs: int | None,
    *,
    mem_per_worker_gb: float | None = None,
    ram_fraction: float = 0.35,
    eeg_paths: Sequence[str | Path] | None = None,
    hard_cap: int = 4,
) -> int:
    """CPU workers capped by a RAM budget for C1a EEG preload/preprocess.

    Defaults are intentionally conservative: large OpenNeuro recordings can be
    multi-GB on disk and much larger once preloaded as float64 with a Raw copy.
    """
    workers = resolve_n_jobs(n_jobs)
    # Explicit ``n_jobs >= 1`` still gets a safety hard-cap so a typo cannot
    # launch 8× multi-GB workers; use serial ``1`` when RAM is tight.
    estimated = (
        float(mem_per_worker_gb)
        if mem_per_worker_gb is not None
        else estimate_c1a_mem_per_worker_gb(eeg_paths)
    )
    total = total_ram_bytes()
    if total is not None and estimated > 0:
        usable = max(0.0, float(ram_fraction) * float(total))
        budget = int(usable // (estimated * (1024**3)))
        workers = min(workers, max(1, budget))
    workers = min(workers, max(1, int(hard_cap)))
    # If one worker's estimate exceeds usable RAM, force serial.
    if total is not None and estimated * (1024**3) > float(ram_fraction) * float(total):
        workers = 1
    return max(1, workers)


def configure_blas_threads(n_threads: int = 1) -> None:
    """Limit BLAS/OpenMP threads in this process (call in workers)."""
    value = str(max(1, int(n_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = value


def format_hms(seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def report_progress(
    *,
    label: str,
    done: int,
    total: int,
    start_time: float,
    last_report: float,
    force: bool = False,
) -> float:
    now = time.perf_counter()
    if not force and done not in (1, total) and (now - last_report) < 1.0:
        return last_report
    elapsed = now - start_time
    rate = elapsed / done if done else 0.0
    remaining = rate * (total - done) if done else 0.0
    print(
        f"[confirmatory] {label} progress: {done}/{total} observations | "
        f"elapsed={format_hms(elapsed)} | ETA={format_hms(remaining)}",
        flush=True,
    )
    return now


def file_identity(path: Path) -> dict[str, object]:
    resolved = Path(path).expanduser().absolute()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_csv_rows(
    path: Path,
    rows: list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
    fieldnames: list[str],
) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def prepare_obs_checkpoint_dir(
    checkpoint_dir: Path,
    *,
    manifest: Mapping[str, object],
) -> Path:
    """Create/reset observation checkpoint dir when the run manifest changes."""
    import shutil

    path = Path(checkpoint_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    for tmp in path.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass
    manifest_path = path / "run_manifest.json"
    expected = dict(manifest)
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing != expected:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, expected)
    return path


__all__ = [
    "atomic_write_bytes",
    "atomic_write_csv_rows",
    "atomic_write_json",
    "atomic_write_text",
    "configure_blas_threads",
    "eeg_payload_bytes",
    "estimate_c1a_mem_per_worker_gb",
    "file_identity",
    "format_hms",
    "prepare_obs_checkpoint_dir",
    "report_progress",
    "resolve_c1a_n_jobs",
    "resolve_n_jobs",
    "total_ram_bytes",
]
