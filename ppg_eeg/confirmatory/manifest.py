"""Run and figure-source manifests for confirmatory reporting (M12)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

FIGURE_SOURCE_MANIFEST_FILENAME = "figure_source_manifest.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"

PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "matplotlib",
)


@dataclass(frozen=True)
class FileHashRecord:
    relative_path: str
    sha256: str
    nbytes: int


@dataclass
class FigurePanelSource:
    figure_id: str
    panel_id: str
    title: str
    endpoint_name: str
    duration_s: int
    input_tables: list[str]
    source_data_csv: str
    analysis_keys: list[str] = field(default_factory=list)
    notes: str = ""
    # manuscript | supplementary | internal_qc
    export_category: str = "manuscript"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_directory_files(
    root: str | Path,
    *,
    patterns: Sequence[str] = ("*.csv", "*.json", "*.yaml", "*.yml"),
) -> list[FileHashRecord]:
    """Hash files under ``root`` matching ``patterns`` (deterministic order)."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(base.rglob(pattern))
    records: list[FileHashRecord] = []
    for path in sorted(paths, key=lambda p: str(p.relative_to(base)).casefold()):
        if not path.is_file():
            continue
        rel = str(path.relative_to(base)).replace("\\", "/")
        records.append(
            FileHashRecord(
                relative_path=rel,
                sha256=sha256_file(path),
                nbytes=int(path.stat().st_size),
            )
        )
    return records


def software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def git_commit_hash(repo_root: str | Path | None = None) -> str:
    """Return current HEAD commit hash, or ``unknown`` if unavailable."""
    cwd = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    text = completed.stdout.strip()
    return text or "unknown"


def config_hash(config_paths: Sequence[str | Path]) -> str:
    """Stable hash over concatenated sorted config file bytes."""
    blobs: list[bytes] = []
    for path in sorted((Path(p).expanduser().resolve() for p in config_paths), key=str):
        if path.is_file():
            blobs.append(path.read_bytes())
    return sha256_bytes(b"\n".join(blobs)) if blobs else sha256_text("")


def count_inclusions(table_path: str | Path | None, *, key_column: str = "status") -> dict[str, int]:
    """Count rows by ``key_column`` when a CSV exists; empty dict otherwise."""
    if table_path is None:
        return {}
    path = Path(table_path)
    if not path.is_file():
        return {}
    import csv

    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = str(row.get(key_column, "")).strip() or "unspecified"
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_figure_source_manifest(
    panels: Sequence[FigurePanelSource],
    *,
    input_hashes: Sequence[FileHashRecord] | Mapping[str, str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Machine-readable mapping of figure panels → inputs and source CSVs."""
    if isinstance(input_hashes, Mapping):
        hash_payload = dict(input_hashes)
    else:
        hash_payload = {record.relative_path: record.sha256 for record in input_hashes}
    payload = {
        "schema_version": "confirmatory_figure_source_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file_sha256": hash_payload,
        "panels": [asdict(panel) for panel in panels],
    }
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / FIGURE_SOURCE_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = sha256_file(path)
    return payload


def build_run_manifest(
    *,
    output_dir: str | Path,
    confirmatory_root: str | Path,
    config_paths: Sequence[str | Path] = (),
    repo_root: str | Path | None = None,
    seeds: Mapping[str, object] | None = None,
    inclusion_counts: Mapping[str, object] | None = None,
    input_hashes: Sequence[FileHashRecord] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Write the confirmatory run manifest (config/code/versions/seeds/hashes)."""
    root = Path(confirmatory_root).expanduser().resolve()
    hashes = list(input_hashes) if input_hashes is not None else hash_directory_files(root)
    payload: dict[str, Any] = {
        "schema_version": "confirmatory_run_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_root": str(root),
        "config_sha256": config_hash(config_paths),
        "config_paths": [str(Path(p)) for p in config_paths],
        "code_commit": git_commit_hash(repo_root),
        "software_versions": software_versions(),
        "seeds": dict(seeds or {}),
        "inclusion_counts": dict(inclusion_counts or {}),
        "input_files": [asdict(record) for record in hashes],
    }
    if extra:
        payload["extra"] = dict(extra)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / RUN_MANIFEST_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = sha256_file(path)
    return payload


__all__ = [
    "FIGURE_SOURCE_MANIFEST_FILENAME",
    "RUN_MANIFEST_FILENAME",
    "FileHashRecord",
    "FigurePanelSource",
    "build_figure_source_manifest",
    "build_run_manifest",
    "config_hash",
    "count_inclusions",
    "git_commit_hash",
    "hash_directory_files",
    "sha256_file",
    "sha256_text",
    "software_versions",
]
