"""C6 upstream exports for Figure 3 Panel F topography / gamma sensitivity.

Panel F channel-level computation runs in C6. C7 must load these canonical
tables and render only.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .panel_f_topography_gamma import (
    PANEL_F_STEM,
    PanelFResult,
    compute_panel_f_topography,
    write_panel_f_exports,
)

OBSERVATION_FILENAME = f"{PANEL_F_STEM}_observation_level.csv"
SUMMARY_FILENAME = f"{PANEL_F_STEM}_summary.csv"
MONTAGE_FILENAME = f"{PANEL_F_STEM}_montage_membership.csv"
COMMON_MONTAGE_FILENAME = f"{PANEL_F_STEM}_common_montage.csv"
GAMMA_SENSITIVITY_FILENAME = f"{PANEL_F_STEM}_gamma_montage_sensitivity.csv"
GAMMA_CONTROL_COMPARISON_FILENAME = f"{PANEL_F_STEM}_gamma_control_comparison.csv"
DIAGNOSTICS_FILENAME = f"{PANEL_F_STEM}_diagnostics.csv"
CHANNEL_LISTS_FILENAME = f"{PANEL_F_STEM}_channel_lists.csv"
PARTICIPANT_SCALP_FILENAME = f"{PANEL_F_STEM}_participant_scalp_summaries.csv"
METADATA_FILENAME = f"{PANEL_F_STEM}_metadata.json"
CHANNEL_CACHE_FILENAME = f"{PANEL_F_STEM}_channel_zlpi_cache.csv"

PANEL_F_C6_ARTIFACTS: dict[str, str] = {
    "observations": OBSERVATION_FILENAME,
    "summary": SUMMARY_FILENAME,
    "montage": MONTAGE_FILENAME,
    "common_montage": COMMON_MONTAGE_FILENAME,
    "gamma_montage_sensitivity": GAMMA_SENSITIVITY_FILENAME,
    "gamma_control_comparison": GAMMA_CONTROL_COMPARISON_FILENAME,
    "diagnostics": DIAGNOSTICS_FILENAME,
    "channel_lists": CHANNEL_LISTS_FILENAME,
    "participant_scalp": PARTICIPANT_SCALP_FILENAME,
    "metadata": METADATA_FILENAME,
    "channel_cache": CHANNEL_CACHE_FILENAME,
}


@dataclass(frozen=True)
class PanelFUpstreamResult:
    result: PanelFResult
    paths: dict[str, Path]


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _compact_row(row: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, str) and not value.strip():
            continue
        out[str(key)] = value
    return out


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _as_str(value).casefold() in {"1", "true", "yes", "y", "t"}


def _load_capabilities(c0_dir: Path) -> dict[str, object]:
    rows = _read_csv(c0_dir / "capability_resolution.csv")
    by_name = {_as_str(row.get("capability")): row for row in rows}

    def _effective(name: str, default: bool) -> bool:
        row = by_name.get(name)
        if not row:
            return default
        return _as_bool(row.get("effective_value"))

    def _reason_code(name: str) -> str:
        row = by_name.get(name)
        if not row:
            return ""
        return _as_str(row.get("reason_code"))

    def _evidence(name: str) -> str:
        row = by_name.get(name)
        if not row:
            return ""
        return _as_str(row.get("observed_evidence"))

    return {
        "supports_topography": _effective("supports_topography", True),
        "supports_topography_reason_code": _reason_code("supports_topography"),
        "supports_topography_evidence": _evidence("supports_topography"),
        "supports_gamma": _effective("supports_gamma", True),
        "supports_gamma_reason_code": _reason_code("supports_gamma"),
        "supports_gamma_evidence": _evidence("supports_gamma"),
    }


def load_panel_f_result_from_upstream(
    upstream_dir: Path,
    *,
    require_complete: bool = True,
) -> PanelFResult:
    """Rebuild ``PanelFResult`` from canonical C6 Panel F exports."""
    root = Path(upstream_dir)
    required = (
        OBSERVATION_FILENAME,
        SUMMARY_FILENAME,
        MONTAGE_FILENAME,
        GAMMA_SENSITIVITY_FILENAME,
        DIAGNOSTICS_FILENAME,
        METADATA_FILENAME,
    )
    if require_complete:
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Panel F upstream exports missing in "
                f"{root}: {', '.join(missing)}"
            )
    metadata: dict[str, object] = {}
    if (root / METADATA_FILENAME).is_file():
        metadata = json.loads((root / METADATA_FILENAME).read_text(encoding="utf-8"))
    return PanelFResult(
        observation_rows=tuple(_compact_row(r) for r in _read_csv(root / OBSERVATION_FILENAME)),
        summary_rows=tuple(_compact_row(r) for r in _read_csv(root / SUMMARY_FILENAME)),
        montage_rows=tuple(_compact_row(r) for r in _read_csv(root / MONTAGE_FILENAME)),
        gamma_comparison_rows=tuple(_compact_row(r) for r in _read_csv(root / GAMMA_SENSITIVITY_FILENAME)),
        diagnostic_rows=tuple(_compact_row(r) for r in _read_csv(root / DIAGNOSTICS_FILENAME)),
        metadata=metadata,
    )


def run_confirmatory_panel_f_upstream(
    *,
    c0_dir: Path,
    c5_dir: Path,
    output_dir: Path,
    confirmatory_root: Path | None = None,
    force_recompute_channel_zlpi: bool = False,
) -> PanelFUpstreamResult:
    """Compute Panel F in C6 and write canonical upstream exports."""
    paired_rows = _read_csv(c5_dir / "paired_contrasts.csv")
    capabilities = _load_capabilities(c0_dir)
    root = Path(confirmatory_root).resolve() if confirmatory_root is not None else Path(output_dir).resolve().parent
    cache_path = Path(output_dir).resolve() / CHANNEL_CACHE_FILENAME
    result = compute_panel_f_topography(
        confirmatory_root=root,
        paired_rows=paired_rows,
        force_recompute_channel_zlpi=force_recompute_channel_zlpi,
        capabilities=capabilities,
        channel_cache_path=cache_path,
    )
    paths = write_panel_f_exports(result, Path(output_dir), stage="C6")
    return PanelFUpstreamResult(result=result, paths=paths)


__all__ = [
    "CHANNEL_CACHE_FILENAME",
    "CHANNEL_LISTS_FILENAME",
    "COMMON_MONTAGE_FILENAME",
    "DIAGNOSTICS_FILENAME",
    "GAMMA_CONTROL_COMPARISON_FILENAME",
    "GAMMA_SENSITIVITY_FILENAME",
    "METADATA_FILENAME",
    "MONTAGE_FILENAME",
    "OBSERVATION_FILENAME",
    "PANEL_F_C6_ARTIFACTS",
    "PARTICIPANT_SCALP_FILENAME",
    "SUMMARY_FILENAME",
    "PanelFUpstreamResult",
    "load_panel_f_result_from_upstream",
    "run_confirmatory_panel_f_upstream",
]
