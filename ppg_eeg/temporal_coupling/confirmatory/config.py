"""Strict configuration schema for the confirmatory zero-lag reanalysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA_VERSION = 1

EXPECTED_BANDS_HZ: dict[str, tuple[float, float]] = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 29.0),
    "low_gamma": (30.0, 45.0),
}
EXPECTED_DURATIONS_S = (240, 180, 120, 60)
EXPECTED_PRIMARY_DURATION_S = 240
EXPECTED_LAG_RANGE_S = (-60, 60)
EXPECTED_LAG_STEP_S = 1
EXPECTED_ZLPI_FLANKS_S = (20, 60)
EXPECTED_SHOULDERS_S = (5, 15)
EXPECTED_PEAK_CENTER_EQUIVALENCE_S = 2
EXPECTED_HR_MODE = "instantaneous"


@dataclass(frozen=True)
class FrequencyBandConfig:
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class EegBandsConfig:
    theta: FrequencyBandConfig
    alpha: FrequencyBandConfig
    beta: FrequencyBandConfig
    low_gamma: FrequencyBandConfig


@dataclass(frozen=True)
class DurationConfig:
    all_s: tuple[int, ...]
    primary_s: int


@dataclass(frozen=True)
class LagConfig:
    min_s: int
    max_s: int
    step_s: int


@dataclass(frozen=True)
class EndpointConfig:
    zlpi_flanks_s: tuple[int, int]
    shoulders_s: tuple[int, int]
    peak_center_equivalence_s: int


@dataclass(frozen=True)
class CardiacConfig:
    hr_mode: str


@dataclass(frozen=True)
class DatasetRolesConfig:
    primary: tuple[str, ...]
    sensitivity: tuple[str, ...]

    def role_for(self, dataset_id: str) -> str:
        if dataset_id in self.primary:
            return "primary"
        if dataset_id in self.sensitivity:
            return "sensitivity"
        raise ValueError(
            f"Dataset {dataset_id!r} is not assigned a primary or sensitivity role."
        )


@dataclass(frozen=True)
class ConfirmatoryMasterConfig:
    schema_version: int
    source_path: Path
    output_root: Path
    root_seed: int
    bands: EegBandsConfig
    durations: DurationConfig
    lag: LagConfig
    endpoints: EndpointConfig
    cardiac: CardiacConfig
    dataset_roles: DatasetRolesConfig


@dataclass(frozen=True)
class DatasetPathsConfig:
    raw_root: Path
    output_subdir: Path


@dataclass(frozen=True)
class DatasetSelectionConfig:
    subjects: tuple[str, ...]
    tasks: tuple[str, ...]
    conditions: tuple[str, ...]
    sessions: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmatoryDatasetConfig:
    schema_version: int
    source_path: Path
    dataset_id: str
    role: str
    paths: DatasetPathsConfig
    selection: DatasetSelectionConfig
    output_root: Path


def _as_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a YAML mapping.")
    return dict(value)


def _reject_unknown_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        rendered = ", ".join(repr(key) for key in unknown)
        raise ValueError(f"Unknown key(s) at {path}: {rendered}.")


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        rendered = ", ".join(repr(key) for key in missing)
        raise ValueError(f"Missing required key(s) at {path}: {rendered}.")


def _as_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return value


def _as_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric.")
    return float(value)


def _as_int_pair(value: Any, *, path: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be a two-item list.")
    return (
        _as_int(value[0], path=f"{path}[0]"),
        _as_int(value[1], path=f"{path}[1]"),
    )


def _as_string_tuple(value: Any, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{path}[{index}] must be a non-empty string.")
        result.append(item.strip().casefold())
    if len(result) != len(set(result)):
        raise ValueError(f"{path} must not contain duplicates.")
    return tuple(result)


def _read_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return config_path, _as_mapping(payload, path="<root>")


def _resolve_path(config_path: Path, value: Any, *, path: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _parse_band(value: Any, *, name: str) -> FrequencyBandConfig:
    path = f"confirmatory.bands.{name}"
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be [low_hz, high_hz].")
    limits = (
        _as_number(value[0], path=f"{path}[0]"),
        _as_number(value[1], path=f"{path}[1]"),
    )
    expected = EXPECTED_BANDS_HZ[name]
    if limits != expected:
        raise ValueError(f"{path} must be exactly {list(expected)}, got {list(limits)}.")
    return FrequencyBandConfig(low_hz=limits[0], high_hz=limits[1])


def load_master_config(path: str | Path) -> ConfirmatoryMasterConfig:
    """Load and strictly validate the immutable confirmatory analysis settings."""
    config_path, root = _read_yaml(path)
    _reject_unknown_keys(
        root, allowed={"schema_version", "confirmatory"}, path="<root>"
    )
    _require_keys(
        root, required={"schema_version", "confirmatory"}, path="<root>"
    )

    schema_version = _as_int(root["schema_version"], path="schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}, got {schema_version}."
        )

    raw = _as_mapping(root["confirmatory"], path="confirmatory")
    allowed = {
        "output_root",
        "root_seed",
        "bands",
        "durations",
        "lag",
        "endpoints",
        "cardiac",
        "dataset_roles",
    }
    _reject_unknown_keys(raw, allowed=allowed, path="confirmatory")
    _require_keys(raw, required=allowed, path="confirmatory")

    output_root = _resolve_path(
        config_path, raw["output_root"], path="confirmatory.output_root"
    )
    if "confirmatory" not in str(output_root).casefold():
        raise ValueError(
            "confirmatory.output_root must be a dedicated path containing "
            "'confirmatory'."
        )

    root_seed = _as_int(raw["root_seed"], path="confirmatory.root_seed")
    if not 0 <= root_seed <= 2**32 - 1:
        raise ValueError("confirmatory.root_seed must be in [0, 2**32 - 1].")

    bands_raw = _as_mapping(raw["bands"], path="confirmatory.bands")
    _reject_unknown_keys(
        bands_raw, allowed=set(EXPECTED_BANDS_HZ), path="confirmatory.bands"
    )
    _require_keys(
        bands_raw, required=set(EXPECTED_BANDS_HZ), path="confirmatory.bands"
    )
    bands = EegBandsConfig(
        theta=_parse_band(bands_raw["theta"], name="theta"),
        alpha=_parse_band(bands_raw["alpha"], name="alpha"),
        beta=_parse_band(bands_raw["beta"], name="beta"),
        low_gamma=_parse_band(bands_raw["low_gamma"], name="low_gamma"),
    )

    durations_raw = _as_mapping(raw["durations"], path="confirmatory.durations")
    _reject_unknown_keys(
        durations_raw,
        allowed={"all_s", "primary_s"},
        path="confirmatory.durations",
    )
    _require_keys(
        durations_raw,
        required={"all_s", "primary_s"},
        path="confirmatory.durations",
    )
    all_s_raw = durations_raw["all_s"]
    if not isinstance(all_s_raw, (list, tuple)):
        raise ValueError("confirmatory.durations.all_s must be a list.")
    all_s = tuple(
        _as_int(value, path=f"confirmatory.durations.all_s[{index}]")
        for index, value in enumerate(all_s_raw)
    )
    if all_s != EXPECTED_DURATIONS_S:
        raise ValueError(
            "confirmatory.durations.all_s must be exactly "
            f"{list(EXPECTED_DURATIONS_S)}, got {list(all_s)}."
        )
    primary_s = _as_int(
        durations_raw["primary_s"], path="confirmatory.durations.primary_s"
    )
    if primary_s != EXPECTED_PRIMARY_DURATION_S:
        raise ValueError(
            "confirmatory.durations.primary_s must be "
            f"{EXPECTED_PRIMARY_DURATION_S}."
        )
    durations = DurationConfig(all_s=all_s, primary_s=primary_s)

    lag_raw = _as_mapping(raw["lag"], path="confirmatory.lag")
    _reject_unknown_keys(
        lag_raw, allowed={"min_s", "max_s", "step_s"}, path="confirmatory.lag"
    )
    _require_keys(
        lag_raw, required={"min_s", "max_s", "step_s"}, path="confirmatory.lag"
    )
    lag = LagConfig(
        min_s=_as_int(lag_raw["min_s"], path="confirmatory.lag.min_s"),
        max_s=_as_int(lag_raw["max_s"], path="confirmatory.lag.max_s"),
        step_s=_as_int(lag_raw["step_s"], path="confirmatory.lag.step_s"),
    )
    if (lag.min_s, lag.max_s) != EXPECTED_LAG_RANGE_S:
        raise ValueError(
            "confirmatory.lag range must be exactly "
            f"{EXPECTED_LAG_RANGE_S[0]} to {EXPECTED_LAG_RANGE_S[1]} s."
        )
    if lag.step_s != EXPECTED_LAG_STEP_S:
        raise ValueError(
            f"confirmatory.lag.step_s must be {EXPECTED_LAG_STEP_S}."
        )

    endpoints_raw = _as_mapping(raw["endpoints"], path="confirmatory.endpoints")
    endpoint_keys = {
        "zlpi_flanks_s",
        "shoulders_s",
        "peak_center_equivalence_s",
    }
    _reject_unknown_keys(
        endpoints_raw, allowed=endpoint_keys, path="confirmatory.endpoints"
    )
    _require_keys(
        endpoints_raw, required=endpoint_keys, path="confirmatory.endpoints"
    )
    endpoints = EndpointConfig(
        zlpi_flanks_s=_as_int_pair(
            endpoints_raw["zlpi_flanks_s"],
            path="confirmatory.endpoints.zlpi_flanks_s",
        ),
        shoulders_s=_as_int_pair(
            endpoints_raw["shoulders_s"],
            path="confirmatory.endpoints.shoulders_s",
        ),
        peak_center_equivalence_s=_as_int(
            endpoints_raw["peak_center_equivalence_s"],
            path="confirmatory.endpoints.peak_center_equivalence_s",
        ),
    )
    if endpoints.zlpi_flanks_s != EXPECTED_ZLPI_FLANKS_S:
        raise ValueError(
            "confirmatory.endpoints.zlpi_flanks_s must be exactly "
            f"{list(EXPECTED_ZLPI_FLANKS_S)}."
        )
    if endpoints.shoulders_s != EXPECTED_SHOULDERS_S:
        raise ValueError(
            "confirmatory.endpoints.shoulders_s must be exactly "
            f"{list(EXPECTED_SHOULDERS_S)}."
        )
    if (
        endpoints.peak_center_equivalence_s
        != EXPECTED_PEAK_CENTER_EQUIVALENCE_S
    ):
        raise ValueError(
            "confirmatory.endpoints.peak_center_equivalence_s must be "
            f"{EXPECTED_PEAK_CENTER_EQUIVALENCE_S}."
        )

    cardiac_raw = _as_mapping(raw["cardiac"], path="confirmatory.cardiac")
    _reject_unknown_keys(
        cardiac_raw, allowed={"hr_mode"}, path="confirmatory.cardiac"
    )
    _require_keys(
        cardiac_raw, required={"hr_mode"}, path="confirmatory.cardiac"
    )
    hr_mode = str(cardiac_raw["hr_mode"]).strip().casefold()
    if hr_mode != EXPECTED_HR_MODE:
        raise ValueError(
            "confirmatory.cardiac.hr_mode must be exactly "
            f"{EXPECTED_HR_MODE!r}."
        )
    cardiac = CardiacConfig(hr_mode=hr_mode)

    roles_raw = _as_mapping(
        raw["dataset_roles"], path="confirmatory.dataset_roles"
    )
    _reject_unknown_keys(
        roles_raw,
        allowed={"primary", "sensitivity"},
        path="confirmatory.dataset_roles",
    )
    _require_keys(
        roles_raw,
        required={"primary", "sensitivity"},
        path="confirmatory.dataset_roles",
    )
    roles = DatasetRolesConfig(
        primary=_as_string_tuple(
            roles_raw["primary"], path="confirmatory.dataset_roles.primary"
        ),
        sensitivity=_as_string_tuple(
            roles_raw["sensitivity"],
            path="confirmatory.dataset_roles.sensitivity",
        ),
    )
    if not roles.primary or not roles.sensitivity:
        raise ValueError(
            "confirmatory.dataset_roles primary and sensitivity must both be non-empty."
        )
    overlap = sorted(set(roles.primary).intersection(roles.sensitivity))
    if overlap:
        raise ValueError(
            "Datasets cannot have both primary and sensitivity roles: "
            + ", ".join(overlap)
        )

    return ConfirmatoryMasterConfig(
        schema_version=schema_version,
        source_path=config_path,
        output_root=output_root,
        root_seed=root_seed,
        bands=bands,
        durations=durations,
        lag=lag,
        endpoints=endpoints,
        cardiac=cardiac,
        dataset_roles=roles,
    )


def load_dataset_config(
    path: str | Path,
    *,
    master: ConfirmatoryMasterConfig,
) -> ConfirmatoryDatasetConfig:
    """Load a dataset selector and validate it against the master protocol."""
    config_path, root = _read_yaml(path)
    top_keys = {"schema_version", "dataset_id", "role", "paths", "selection"}
    _reject_unknown_keys(root, allowed=top_keys, path="<root>")
    _require_keys(root, required=top_keys, path="<root>")

    schema_version = _as_int(root["schema_version"], path="schema_version")
    if schema_version != master.schema_version:
        raise ValueError(
            "Dataset config schema_version must match the master config "
            f"({master.schema_version})."
        )

    dataset_id_raw = root["dataset_id"]
    if not isinstance(dataset_id_raw, str) or not dataset_id_raw.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    dataset_id = dataset_id_raw.strip().casefold()

    role_raw = root["role"]
    if not isinstance(role_raw, str):
        raise ValueError("role must be 'primary' or 'sensitivity'.")
    role = role_raw.strip().casefold()
    if role not in {"primary", "sensitivity"}:
        raise ValueError("role must be 'primary' or 'sensitivity'.")
    expected_role = master.dataset_roles.role_for(dataset_id)
    if role != expected_role:
        raise ValueError(
            f"Dataset {dataset_id!r} is assigned role {expected_role!r} "
            f"in the master config, not {role!r}."
        )

    paths_raw = _as_mapping(root["paths"], path="paths")
    _reject_unknown_keys(
        paths_raw, allowed={"raw_root", "output_subdir"}, path="paths"
    )
    _require_keys(
        paths_raw, required={"raw_root", "output_subdir"}, path="paths"
    )
    raw_root = _resolve_path(config_path, paths_raw["raw_root"], path="paths.raw_root")
    output_subdir_raw = paths_raw["output_subdir"]
    if not isinstance(output_subdir_raw, str) or not output_subdir_raw.strip():
        raise ValueError("paths.output_subdir must be a non-empty relative path.")
    output_subdir = Path(output_subdir_raw.strip())
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError(
            "paths.output_subdir must be relative and must not contain '..'."
        )
    output_root = (master.output_root / output_subdir).resolve()
    if output_root == raw_root or master.output_root not in output_root.parents:
        raise ValueError(
            "Dataset output_root must be a child of the dedicated confirmatory "
            "output root and separate from raw_root."
        )

    selection_raw = _as_mapping(root["selection"], path="selection")
    selection_keys = {"subjects", "tasks", "conditions", "sessions"}
    _reject_unknown_keys(
        selection_raw, allowed=selection_keys, path="selection"
    )
    _require_keys(
        selection_raw, required=selection_keys, path="selection"
    )
    selection = DatasetSelectionConfig(
        subjects=_as_string_tuple(
            selection_raw["subjects"], path="selection.subjects"
        ),
        tasks=_as_string_tuple(selection_raw["tasks"], path="selection.tasks"),
        conditions=_as_string_tuple(
            selection_raw["conditions"], path="selection.conditions"
        ),
        sessions=_as_string_tuple(
            selection_raw["sessions"], path="selection.sessions"
        ),
    )

    return ConfirmatoryDatasetConfig(
        schema_version=schema_version,
        source_path=config_path,
        dataset_id=dataset_id,
        role=role,
        paths=DatasetPathsConfig(
            raw_root=raw_root,
            output_subdir=output_subdir,
        ),
        selection=selection,
        output_root=output_root,
    )
