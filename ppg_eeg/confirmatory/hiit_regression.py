"""Freeze and compare HIIT confirmatory regression snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SNAPSHOT_VERSION = "hiit_regression_v1"
DEFAULT_SNAPSHOT_DIR = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "hiit_regression"
    / SNAPSHOT_VERSION
)

# Relative paths under sensitivity/hiit that form the frozen baseline.
SNAPSHOT_RELATIVE_PATHS: tuple[str, ...] = (
    "run_metadata.json",
    "C0/eligibility_by_duration.csv",
    "C0/eligibility_qc_summary.csv",
    "C0/capability_resolution.csv",
    "C0/protocol_audit.csv",
    "C0/data_audit.csv",
    "C7/figures/source_data/figure3_panel_c_duration_sensitivity.csv",
    "C7/figures/source_data/figure3_panel_d_cardiac_controls_summaries.csv",
    "C7/figures/source_data/figure3_panel_e_nuisance_modality_specifications.csv",
    "C7/figures/source_data/figure3_panel_e_nuisance_modality_common_sample.csv",
    "C7/figures/source_data/figure3_panel_f_topography_gamma_summary.csv",
    "C7/figures/source_data/figure3_panel_f_topography_gamma_common_montage.csv",
    "C7/figures/source_data/figure3_panel_b_group_summaries.csv",
    "C7/figures/source_data/figure3_panel_a_inference.csv",
)

# Exact-match categorical / identifier columns across CSVs.
EXACT_COLUMNS = frozenset(
    {
        "dataset_id",
        "observation_id",
        "participant_id",
        "condition",
        "contrast_id",
        "duration_s",
        "status",
        "exclusion_code",
        "endpoint_name",
        "endpoint_alias",
        "endpoint_status",
        "endpoint_reason_code",
        "standard_zlpi_computable",
        "standard_zlpi_reason_code",
        "is_standard_zlpi",
        "endpoint_computable",
        "has_required_duration",
        "has_required_lag_support",
        "capability",
        "declared_value",
        "effective_value",
        "reason_code",
        "affected_stage",
        "control",
        "computability_status",
        "computability_reason",
        "specification_id",
        "channel",
        "band",
        "eligibility_status",
    }
)

NUMERIC_ABS_TOL = 1e-9
NUMERIC_REL_TOL = 1e-9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref:"):
        ref_path = repo / ".git" / ref.split(" ", 1)[1].strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()[:40]
        return "unknown"
    return ref[:40]


def freeze_hiit_regression_snapshot(
    hiit_root: str | Path,
    snapshot_dir: str | Path | None = None,
    *,
    specification_version: str = "phase1_closure",
) -> Path:
    """Copy accepted HIIT outputs into a versioned regression fixture."""
    src = Path(hiit_root).expanduser().resolve()
    dest = Path(snapshot_dir or DEFAULT_SNAPSHOT_DIR).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(
            f"Snapshot already exists at {dest}. Do not overwrite without "
            "explaining every difference against the frozen baseline."
        )
    dest.mkdir(parents=True, exist_ok=False)
    copied: list[dict[str, str]] = []
    for rel in SNAPSHOT_RELATIVE_PATHS:
        source = src / rel
        if not source.is_file():
            continue
        target = dest / "data" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(
            {
                "relative_path": rel,
                "sha256": _sha256_file(target),
                "bytes": str(target.stat().st_size),
            }
        )

    repo = Path(__file__).resolve().parents[2]
    run_meta_path = src / "run_metadata.json"
    run_meta = {}
    if run_meta_path.is_file():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "specification_version": specification_version,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_hiit_root": str(src),
        "code_commit": _git_commit(repo),
        "configuration_hash": run_meta.get("configuration_hash_sha256")
        or run_meta.get("configuration_hash")
        or run_meta.get("config_hash")
        or "",
        "random_seeds": {
            "root_seed": run_meta.get("root_seed"),
            **(run_meta.get("seeds") or {}),
            **(run_meta.get("random_seeds") or {}),
        },
        "files": copied,
        "tolerances": {
            "categorical_exact": True,
            "numeric_abs_tol": NUMERIC_ABS_TOL,
            "numeric_rel_tol": NUMERIC_REL_TOL,
        },
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _values_equal(a: str, b: str, *, column: str) -> bool:
    if column in EXACT_COLUMNS or a == b:
        return a == b
    try:
        fa = float(a) if a not in {"", "None", "nan", "NaN"} else float("nan")
        fb = float(b) if b not in {"", "None", "nan", "NaN"} else float("nan")
    except ValueError:
        return a == b
    if math.isnan(fa) and math.isnan(fb):
        return True
    if not math.isfinite(fa) or not math.isfinite(fb):
        return fa == fb
    return math.isclose(fa, fb, rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)


def compare_csv_to_snapshot(
    current: Path,
    baseline: Path,
    *,
    key_columns: Sequence[str] = (),
) -> list[str]:
    """Return human-readable differences between current and frozen CSV."""
    if not baseline.is_file():
        return [f"missing baseline: {baseline}"]
    if not current.is_file():
        return [f"missing current: {current}"]
    cur_rows = _read_csv(current)
    base_rows = _read_csv(baseline)
    failures: list[str] = []
    if len(cur_rows) != len(base_rows):
        failures.append(
            f"row count {current.name}: current={len(cur_rows)} baseline={len(base_rows)}"
        )
    n = min(len(cur_rows), len(base_rows))
    for idx in range(n):
        left = cur_rows[idx]
        right = base_rows[idx]
        cols = sorted(set(left) | set(right))
        for col in cols:
            if not _values_equal(left.get(col, ""), right.get(col, ""), column=col):
                key = ""
                if key_columns:
                    key = "|" + "|".join(left.get(k, "") for k in key_columns)
                failures.append(
                    f"{current.name}[{idx}{key}].{col}: "
                    f"current={left.get(col)!r} baseline={right.get(col)!r}"
                )
                if len(failures) >= 50:
                    failures.append("... truncated after 50 differences")
                    return failures
    return failures


def compare_hiit_root_to_snapshot(
    hiit_root: str | Path,
    snapshot_dir: str | Path | None = None,
) -> list[str]:
    """Compare live HIIT outputs against the frozen snapshot."""
    src = Path(hiit_root).expanduser().resolve()
    snap = Path(snapshot_dir or DEFAULT_SNAPSHOT_DIR).expanduser().resolve()
    failures: list[str] = []
    for rel in SNAPSHOT_RELATIVE_PATHS:
        baseline = snap / "data" / rel
        if not baseline.is_file():
            continue
        failures.extend(compare_csv_to_snapshot(src / rel, baseline) if rel.endswith(".csv") else [])
        if rel.endswith(".json"):
            if not (src / rel).is_file():
                failures.append(f"missing current json: {rel}")
                continue
            # Hash identity for metadata JSON except volatile timestamps handled
            # by field-level compare of selected keys.
            cur = json.loads((src / rel).read_text(encoding="utf-8"))
            base = json.loads(baseline.read_text(encoding="utf-8"))
            for key in ("configuration_hash", "config_hash", "seeds", "random_seeds"):
                if key in base and cur.get(key) != base.get(key):
                    failures.append(
                        f"{rel}.{key}: current={cur.get(key)!r} baseline={base.get(key)!r}"
                    )
    return failures


__all__ = [
    "DEFAULT_SNAPSHOT_DIR",
    "SNAPSHOT_VERSION",
    "compare_hiit_root_to_snapshot",
    "compare_csv_to_snapshot",
    "freeze_hiit_regression_snapshot",
]
