"""C6 upstream exports for Figure 3 Panel E nuisance/modality robustness.

All paired Δ-nuisance OLS models are fit here. C7 must load these tables and
plot only — it must not refit nuisance specifications.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .panel_e_nuisance_modality import (
    PANEL_E_STEM,
    PanelEResult,
    compute_panel_e_nuisance_modality,
    write_panel_e_upstream_exports,
)

OBSERVATION_FILENAME = f"{PANEL_E_STEM}_observation_level.csv"
SPECIFICATIONS_FILENAME = f"{PANEL_E_STEM}_specifications.csv"
COMMON_SAMPLE_FILENAME = f"{PANEL_E_STEM}_common_sample.csv"
OBSLEVEL_MODELS_FILENAME = f"{PANEL_E_STEM}_observation_level_models.csv"
DIAGNOSTICS_FILENAME = f"{PANEL_E_STEM}_diagnostics.csv"
AVAILABILITY_FILENAME = f"{PANEL_E_STEM}_availability.csv"
NUISANCE_STATE_FILENAME = f"{PANEL_E_STEM}_nuisance_state_summary.csv"
METADATA_FILENAME = f"{PANEL_E_STEM}_metadata.json"

PANEL_E_C6_ARTIFACTS: dict[str, str] = {
    "observations": OBSERVATION_FILENAME,
    "specifications": SPECIFICATIONS_FILENAME,
    "common_sample": COMMON_SAMPLE_FILENAME,
    "observation_level_models": OBSLEVEL_MODELS_FILENAME,
    "diagnostics": DIAGNOSTICS_FILENAME,
    "availability": AVAILABILITY_FILENAME,
    "nuisance_state_summary": NUISANCE_STATE_FILENAME,
    "metadata": METADATA_FILENAME,
}


@dataclass(frozen=True)
class PanelEUpstreamResult:
    result: PanelEResult
    paths: dict[str, Path]


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    import math

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    if isinstance(value, str) and not value.strip():
        return float("nan")
    return out


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y", "t"}


def load_panel_e_result_from_upstream(
    upstream_dir: Path,
    *,
    require_complete: bool = True,
) -> PanelEResult:
    """Rebuild ``PanelEResult`` from canonical C6 Panel E exports."""
    root = Path(upstream_dir)
    required = (
        SPECIFICATIONS_FILENAME,
        COMMON_SAMPLE_FILENAME,
        OBSERVATION_FILENAME,
        AVAILABILITY_FILENAME,
        METADATA_FILENAME,
    )
    if require_complete:
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Panel E upstream exports missing in "
                f"{root}: {', '.join(missing)}"
            )

    metadata_path = root / METADATA_FILENAME
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    specification_rows = tuple(_read_csv(root / SPECIFICATIONS_FILENAME))
    common_sample_rows = tuple(_read_csv(root / COMMON_SAMPLE_FILENAME))
    if not common_sample_rows and specification_rows:
        common_sample_rows = tuple(
            row
            for row in specification_rows
            if _as_str(row.get("sample_scheme")) == "common_sample"
            or _as_bool(row.get("plotted"))
        ) or specification_rows

    def _normalize_spec(row: dict[str, object]) -> dict[str, object]:
        item = dict(row)
        item["predictors_centered"] = _as_bool(item.get("predictors_centered"))
        item["plotted"] = _as_bool(item.get("plotted"))
        item["rank_deficient"] = _as_bool(item.get("rank_deficient"))
        item["composition_differs_from_baseline"] = _as_bool(
            item.get("composition_differs_from_baseline")
        )
        for key in (
            "estimate",
            "ci_lower",
            "ci_upper",
            "standard_error",
            "p_value",
            "change_from_baseline",
            "change_ci_lower",
            "change_ci_upper",
            "change_standard_error",
        ):
            if key in item:
                item[key] = _as_float(item.get(key))
        for key in ("n_observations", "n_specification_usable", "n_baseline_eligible", "n_participants"):
            if key in item and _as_str(item.get(key)):
                item[key] = int(float(item[key]))  # type: ignore[arg-type]
        if "n_specification_usable" not in item and "n_observations" in item:
            item["n_specification_usable"] = item["n_observations"]
        return item

    specification_rows = tuple(_normalize_spec(dict(r)) for r in specification_rows)
    common_sample_rows = tuple(_normalize_spec(dict(r)) for r in common_sample_rows)

    return PanelEResult(
        observation_rows=tuple(_read_csv(root / OBSERVATION_FILENAME)),
        specification_rows=specification_rows,
        common_sample_rows=common_sample_rows,
        observation_level_rows=tuple(_read_csv(root / OBSLEVEL_MODELS_FILENAME)),
        diagnostic_rows=tuple(_read_csv(root / DIAGNOSTICS_FILENAME)),
        missingness_rows=tuple(_read_csv(root / AVAILABILITY_FILENAME)),
        availability_rows=tuple(_read_csv(root / AVAILABILITY_FILENAME)),
        nuisance_state_summary_rows=tuple(_read_csv(root / NUISANCE_STATE_FILENAME)),
        metadata=metadata,
    )


def run_confirmatory_panel_e_upstream(
    *,
    c0_dir: Path,
    c1b_dir: Path,
    c1c_dir: Path,
    c3_dir: Path,
    c5_dir: Path,
    output_dir: Path,
    code_version: str = "panel_e_nuisance_upstream_v1",
) -> PanelEUpstreamResult:
    """Fit Panel E specifications and write canonical C6 exports."""
    del code_version  # reserved for metadata stamping in compute
    paired_rows = _read_csv(c5_dir / "paired_contrasts.csv")
    aligned_rows = _read_csv(c1c_dir / "features_confirmatory_aligned_D240.csv")
    data_audit_rows = _read_csv(c0_dir / "data_audit.csv")
    peak_qc_rows = _read_csv(c1b_dir / "cardiac_peak_qc.csv")
    protocol_rows = _read_csv(c0_dir / "protocol_audit.csv")
    endpoint_rows = _read_csv(c3_dir / "confirmatory_endpoint_metrics_D240.csv")
    subject_rows = _read_csv(c5_dir / "subject_level_metrics.csv")

    result = compute_panel_e_nuisance_modality(
        paired_rows=paired_rows,
        aligned_rows=aligned_rows,
        data_audit_rows=data_audit_rows,
        peak_qc_rows=peak_qc_rows,
        protocol_rows=protocol_rows,
        endpoint_rows=endpoint_rows,
        subject_rows=subject_rows,
    )
    paths = write_panel_e_upstream_exports(result, output_dir)
    return PanelEUpstreamResult(result=result, paths=paths)


__all__ = [
    "AVAILABILITY_FILENAME",
    "COMMON_SAMPLE_FILENAME",
    "DIAGNOSTICS_FILENAME",
    "METADATA_FILENAME",
    "NUISANCE_STATE_FILENAME",
    "OBSERVATION_FILENAME",
    "OBSLEVEL_MODELS_FILENAME",
    "PANEL_E_C6_ARTIFACTS",
    "SPECIFICATIONS_FILENAME",
    "PanelEUpstreamResult",
    "load_panel_e_result_from_upstream",
    "run_confirmatory_panel_e_upstream",
]
