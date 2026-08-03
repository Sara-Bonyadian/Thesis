"""Figure 3 Panel F: topography and gamma specificity (standalone sheet).

Recomputes per-channel D240 absolute_log10 ZLPI from C1a multitaper power + C1b HR.
Alpha maps use a common retained montage. Low-gamma maps show a restricted-montage
sensitivity to ``ecg_prone_default_v1`` exclusion: F3 uses the full usable
pre-exclusion montage (ECG-prone sensors marked); F4 uses the restricted montage.

Retained-channel before/after equality is a deterministic implementation identity
(no value-changing transform), not an empirical robustness result. ICA, EMG,
motion, EOG, and cardiac-template controls remain unavailable.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .duration_contracts import ENDPOINT_ZLPI, EXPECTED_PRIMARY_DURATION_S
from .harmonize import BAND_ORDER, harmonize_observation
from .nulls import compute_endpoint_index_from_series
from .paired_delta_inference import DEFAULT_CLUSTER_BOOTSTRAP_DRAWS

PANEL_F_STEM = "figure3_panel_f_topography_gamma"
PANEL_F_TITLE = "F. Topography and gamma specificity"
PANEL_F_FIGURE_TITLE = "Figure 3F | Topography and gamma specificity"
PANEL_F_SUBTITLE = "D240 absolute_log10 ZLPI · HIIT sensitivity dataset"
PANEL_F_SECOND_LINE_TEMPLATE = (
    "Alpha common montage: n={n_alpha} · Gamma paired observations: n={n_gamma}"
)

PANEL_F_DURATION_S = int(EXPECTED_PRIMARY_DURATION_S)
PANEL_F_ENDPOINT = ENDPOINT_ZLPI
PANEL_F_REPRESENTATION = "absolute_log10"
PANEL_F_ALPHA_BAND = "alpha"
PANEL_F_GAMMA_BAND = "low_gamma"

ECG_PRONE_YAML = Path(__file__).resolve().parent / "cardiac_field_channels.yaml"
ECG_PRONE_SET_ID = "ecg_prone_default_v1"

MAP_REST_ALPHA = "rest_alpha"
MAP_TASK_ATTENUATION = "task_minus_rest_alpha"
MAP_GAMMA_BEFORE = "low_gamma_before_channel_exclusion"
MAP_GAMMA_AFTER = "low_gamma_after_ecg_prone_exclusion"
MAP_ORDER = (
    MAP_REST_ALPHA,
    MAP_TASK_ATTENUATION,
    MAP_GAMMA_BEFORE,
    MAP_GAMMA_AFTER,
)
MAP_TITLES = {
    MAP_REST_ALPHA: "F1. Rest alpha",
    MAP_TASK_ATTENUATION: "F2. Task − Rest alpha",
    MAP_GAMMA_BEFORE: "F3. Gamma before exclusion",
    MAP_GAMMA_AFTER: "F4. Gamma after ECG-prone exclusion",
}

PANEL_F_INTERPRETATION_LINE = (
    "Gamma sensitivity to ECG-prone channel exclusion; "
    "retained-channel equality is deterministic."
)
PANEL_F_FOOTNOTE = (
    "ICA, EMG, motion, EOG, and cardiac-template controls were unavailable; "
    "low-gamma topography remains artifact-indeterminate."
)
PANEL_F_CBAR_ALPHA = "Alpha maps (Fisher-z ZLPI)"
PANEL_F_CBAR_GAMMA = "Low-gamma maps (Fisher-z ZLPI)"
TOPOMAP_CMAP = "RdBu_r"
TOPOMAP_CONTOURS = 6
TOPOMAP_EXTRAPOLATE = "head"
TOPOMAP_INTERP = "cubic"
FIGURE_SIZE = (16.0, 6.4)

CONTROL_SPEC_BEFORE = "baseline_channel_zlpi_no_additional_controls"
CONTROL_SPEC_AFTER = "ecg_prone_default_v1_restricted_montage_sensitivity"

UNAVAILABLE_CONTROLS = (
    {
        "control": "ICA muscle-component rejection",
        "status": "not_available",
        "reason": "Confirmatory preprocessing is non-ICA; no ICA muscle pipeline",
    },
    {
        "control": "EMG residualization",
        "status": "not_available",
        "reason": "No retained EMG / muscle-artifact summary",
    },
    {
        "control": "Motion",
        "status": "not_available",
        "reason": "No retained motion / accelerometer summary",
    },
    {
        "control": "EOG",
        "status": "not_available",
        "reason": "No retained ocular-artifact summary",
    },
    {
        "control": "Cardiac-template subtraction",
        "status": "not_computable",
        "reason": "Not applied at channel-level topography; HIIT is PPG-only for ECG templates",
    },
)

BOOTSTRAP_DRAWS = int(DEFAULT_CLUSTER_BOOTSTRAP_DRAWS)
BOOTSTRAP_SEED = 23
MULTITAPER_FEATURES = "features_multitaper_power.csv"
MULTITAPER_QC = "multitaper_qc.csv"
INSTANT_HR_FEATURES = "features_instant_hr.csv"


@dataclass(frozen=True)
class PanelFResult:
    observation_rows: tuple[dict[str, object], ...]
    summary_rows: tuple[dict[str, object], ...]
    montage_rows: tuple[dict[str, object], ...]
    gamma_comparison_rows: tuple[dict[str, object], ...]
    diagnostic_rows: tuple[dict[str, object], ...]
    metadata: dict[str, object]


def _as_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _as_str(value).casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    return False


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_ecg_prone_channels(yaml_path: Path = ECG_PRONE_YAML) -> tuple[str, tuple[str, ...]]:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    primary = (payload or {}).get("channel_sets", {}).get("primary", {})
    set_id = _as_str(primary.get("id"), ECG_PRONE_SET_ID)
    channels = tuple(
        sorted({_as_str(ch) for ch in (primary.get("channels") or []) if _as_str(ch)})
    )
    if set_id != ECG_PRONE_SET_ID:
        raise ValueError(f"Expected {ECG_PRONE_SET_ID}, found {set_id!r}")
    if not channels:
        raise ValueError("ECG-prone channel list is empty")
    return set_id, channels


def resolve_stage_roots(confirmatory_root: Path) -> dict[str, Path]:
    root = Path(confirmatory_root).expanduser().resolve()
    # C7 publish/ OR C7/ → parent is dataset root containing C1a/C1b
    dataset_root = root.parent if root.name.upper() == "C7" or root.name == "C7" else root
    if root.name == "publish":
        dataset_root = root.parent.parent
    return {
        "confirmatory_root": root,
        "dataset_root": dataset_root,
        "c1a": dataset_root / "C1a",
        "c1b": dataset_root / "C1b",
    }


def _normalize_channel(name: str) -> str:
    return _as_str(name)


def _channel_coordinates(channels: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    out: dict[str, tuple[float, float, float]] = {}
    for ch in channels:
        key = _normalize_channel(ch)
        if key not in pos:
            # try case variants
            match = next((k for k in pos if k.casefold() == key.casefold()), None)
            if match is None:
                continue
            key = match
        xyz = pos[key]
        out[_normalize_channel(ch)] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return out


def _load_hr(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    time_s = np.asarray([_as_float(r.get("time_s")) for r in rows], dtype=float)
    hr = np.asarray([_as_float(r.get("instant_hr_bpm")) for r in rows], dtype=float)
    valid = np.asarray(
        [_as_bool(r.get("is_valid_hr")) for r in rows],
        dtype=bool,
    )
    return time_s, hr, valid


def _channel_power_maps(
    path: Path,
) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]]:
    rows = _read_csv(path)
    by_ch: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if _as_str(row.get("aggregation")).casefold() != "channel":
            continue
        ch = _normalize_channel(row.get("channel", ""))
        if not ch:
            continue
        by_ch[ch].append(row)
    out: dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]] = {}
    for ch, ch_rows in by_ch.items():
        by_t: dict[float, dict[str, tuple[float, float]]] = {}
        for row in ch_rows:
            t = _as_float(row.get("window_center_s"))
            band = _as_str(row.get("band")).casefold()
            if not math.isfinite(t) or band not in BAND_ORDER:
                continue
            by_t.setdefault(t, {})[band] = (
                _as_float(row.get("absolute_power")),
                _as_float(row.get("absolute_log10_power")),
            )
        times = np.asarray(sorted(by_t), dtype=float)
        abs_p = {b: np.full(times.shape, np.nan, dtype=float) for b in BAND_ORDER}
        log_p = {b: np.full(times.shape, np.nan, dtype=float) for b in BAND_ORDER}
        for i, t in enumerate(times):
            for band, (a, lg) in by_t[float(t)].items():
                abs_p[band][i] = a
                log_p[band][i] = lg
        valid = np.all(np.vstack([np.isfinite(log_p[b]) for b in BAND_ORDER]), axis=0)
        out[ch] = (times, abs_p, log_p, valid)
    return out


def _load_spectral_qc(path: Path) -> dict[str, object]:
    rows = _read_csv(path)
    if not rows:
        return {
            "spectral_qc_passed": False,
            "usable_channels": (),
            "rejected_channels": (),
            "reason": "missing_multitaper_qc",
        }
    row = rows[0]
    usable = tuple(
        _normalize_channel(ch)
        for ch in _as_str(row.get("usable_channels")).split(";")
        if _normalize_channel(ch)
    )
    rejected = tuple(
        _normalize_channel(ch)
        for ch in _as_str(row.get("rejected_channels")).split(";")
        if _normalize_channel(ch)
    )
    return {
        "spectral_qc_passed": _as_bool(row.get("spectral_qc_passed")),
        "usable_channels": usable,
        "rejected_channels": rejected,
        "status": _as_str(row.get("status")),
        "warning": _as_str(row.get("warning")),
        "reason": "" if _as_bool(row.get("spectral_qc_passed")) else "spectral_qc_failed",
    }


def compute_observation_channel_zlpi(
    *,
    observation_id: str,
    c1a_dir: Path,
    c1b_dir: Path,
    identity: Mapping[str, object] | None = None,
    bands: Sequence[str] = (PANEL_F_ALPHA_BAND, PANEL_F_GAMMA_BAND),
) -> list[dict[str, object]]:
    """Recompute D240 absolute_log10 ZLPI for every usable channel of one observation."""
    obs = _as_str(observation_id)
    power_path = c1a_dir / obs / MULTITAPER_FEATURES
    hr_path = c1b_dir / obs / INSTANT_HR_FEATURES
    qc_path = c1a_dir / obs / MULTITAPER_QC
    if not power_path.is_file() or not hr_path.is_file():
        return [
            {
                "observation_id": obs,
                "computable": False,
                "not_computable_reason": "missing_c1a_or_c1b_features",
            }
        ]
    qc = _load_spectral_qc(qc_path)
    hr_time, hr_bpm, hr_valid = _load_hr(hr_path)
    channel_maps = _channel_power_maps(power_path)
    ids = {
        "dataset_id": _as_str((identity or {}).get("dataset_id"), "hiit"),
        "participant_id": _as_str((identity or {}).get("participant_id")),
        "session_id": _as_str((identity or {}).get("session_id")),
        "subject_id": _as_str((identity or {}).get("subject_id")),
        "task": _as_str((identity or {}).get("task")),
        "condition": _as_str((identity or {}).get("condition")),
        "observation_id": obs,
        "state": _as_str((identity or {}).get("state")),
    }
    usable = set(qc["usable_channels"]) if qc["usable_channels"] else set(channel_maps)
    rows_out: list[dict[str, object]] = []
    for ch, (times, abs_p, log_p, valid) in sorted(channel_maps.items()):
        base = {
            **ids,
            "channel": ch,
            "montage_name": "standard_1020",
            "duration_s": PANEL_F_DURATION_S,
            "endpoint_name": PANEL_F_ENDPOINT,
            "endpoint_units": "Fisher_z_ZLPI",
            "power_representation": PANEL_F_REPRESENTATION,
            "spectral_qc_passed": bool(qc["spectral_qc_passed"]),
            "channel_in_usable_set": ch in usable,
            "channel_rejected_by_spectral_qc": ch in set(qc["rejected_channels"]),
        }
        if not qc["spectral_qc_passed"]:
            for band in bands:
                rows_out.append(
                    {
                        **base,
                        "band": band,
                        "control_status": "before_controls",
                        "control_specification": CONTROL_SPEC_BEFORE,
                        "zpli_value": float("nan"),
                        "computable": False,
                        "not_computable_reason": "spectral_qc_failed",
                    }
                )
            continue
        if ch not in usable:
            for band in bands:
                rows_out.append(
                    {
                        **base,
                        "band": band,
                        "control_status": "before_controls",
                        "control_specification": CONTROL_SPEC_BEFORE,
                        "zpli_value": float("nan"),
                        "computable": False,
                        "not_computable_reason": "channel_excluded_by_c1a_spectral_qc_usable_set",
                    }
                )
            continue
        try:
            harm = harmonize_observation(
                hr_time_s=hr_time,
                hr_bpm=hr_bpm,
                hr_valid=hr_valid,
                eeg_time_s=times,
                eeg_absolute_power=abs_p,
                eeg_absolute_log10_power=log_p,
                eeg_valid=valid,
                identity={
                    "dataset_id": ids["dataset_id"],
                    "subject_id": ids["subject_id"] or ids["participant_id"],
                    "task": ids["task"],
                    "condition": ids["condition"],
                    "observation_id": obs,
                },
            )
        except Exception as exc:  # noqa: BLE001 — record per-channel failure honestly
            for band in bands:
                rows_out.append(
                    {
                        **base,
                        "band": band,
                        "control_status": "before_controls",
                        "control_specification": CONTROL_SPEC_BEFORE,
                        "zpli_value": float("nan"),
                        "computable": False,
                        "not_computable_reason": f"harmonize_failed:{type(exc).__name__}",
                    }
                )
            continue
        d240 = list(harm.features_by_duration.get(PANEL_F_DURATION_S, ()))
        if not d240:
            for band in bands:
                rows_out.append(
                    {
                        **base,
                        "band": band,
                        "control_status": "before_controls",
                        "control_specification": CONTROL_SPEC_BEFORE,
                        "zpli_value": float("nan"),
                        "computable": False,
                        "not_computable_reason": "d240_segment_not_available",
                    }
                )
            continue
        d240 = sorted(d240, key=lambda r: float(r["time_s"]))
        hr_z = np.asarray([_as_float(r.get("hr_z")) for r in d240], dtype=float)
        for band in bands:
            eeg_z = np.asarray(
                [_as_float(r.get(f"{band}_absolute_log10_power_z")) for r in d240],
                dtype=float,
            )
            metrics = compute_endpoint_index_from_series(
                hr_z,
                eeg_z,
                duration_s=PANEL_F_DURATION_S,
                band=band,
                power_representation=PANEL_F_REPRESENTATION,
            )
            eligible = bool(metrics.get("eligible"))
            value = _as_float(metrics.get("endpoint_index"))
            rows_out.append(
                {
                    **base,
                    "band": band,
                    "control_status": "before_controls",
                    "control_specification": CONTROL_SPEC_BEFORE,
                    "zpli_value": value if eligible and math.isfinite(value) else float("nan"),
                    "computable": bool(eligible and math.isfinite(value)),
                    "not_computable_reason": (
                        ""
                        if eligible and math.isfinite(value)
                        else _as_str(metrics.get("exclusion_reason"), "endpoint_ineligible")
                    ),
                    "n_aligned_samples": len(d240),
                }
            )
    return rows_out


def _infer_state(condition: str) -> str:
    text = condition.casefold()
    if "tetris" in text:
        return "tetris"
    if "rest" in text:
        return "rest"
    return ""


def _select_paired_alpha_rows(paired_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in paired_rows:
        if int(_as_float(row.get("duration_s"))) != PANEL_F_DURATION_S:
            continue
        if _as_str(row.get("endpoint_name")).casefold() != PANEL_F_ENDPOINT:
            continue
        if _as_str(row.get("power_representation")).casefold() != PANEL_F_REPRESENTATION:
            continue
        if _as_str(row.get("band")).casefold() != PANEL_F_ALPHA_BAND:
            continue
        if not _as_bool(row.get("contrast_eligible")):
            continue
        out.append(dict(row))
    return out


def build_or_load_channel_zlpi_table(
    *,
    confirmatory_root: Path,
    paired_rows: Sequence[Mapping[str, object]],
    cache_path: Path | None = None,
    force_recompute: bool = False,
) -> list[dict[str, object]]:
    """Build observation×channel ZLPI for all Rest/Tetris observations in paired set."""
    stages = resolve_stage_roots(confirmatory_root)
    c1a, c1b = stages["c1a"], stages["c1b"]
    if cache_path is None:
        cache_path = (
            Path(confirmatory_root).expanduser().resolve()
            / "figures"
            / "source_data"
            / f"{PANEL_F_STEM}_channel_zlpi_cache.csv"
        )
    if cache_path.is_file() and not force_recompute:
        cached = _read_csv(cache_path)
        if cached:
            return [{k: v for k, v in row.items()} for row in cached]

    paired = _select_paired_alpha_rows(paired_rows)
    obs_meta: dict[str, dict[str, str]] = {}
    for row in paired:
        low = _as_str(row.get("low_observation_ids")).split(";")[0]
        effort = _as_str(row.get("effort_observation_ids")).split(";")[0]
        pid = _as_str(row.get("participant_id"))
        sid = _as_str(row.get("session_id"))
        ds = _as_str(row.get("dataset_id"), "hiit")
        if low:
            obs_meta[low] = {
                "dataset_id": ds,
                "participant_id": pid,
                "session_id": sid,
                "condition": _as_str(row.get("low_demand_condition")),
                "state": "rest",
                "subject_id": f"{pid}_{sid}" if sid else pid,
                "task": "rest",
            }
        if effort:
            obs_meta[effort] = {
                "dataset_id": ds,
                "participant_id": pid,
                "session_id": sid,
                "condition": _as_str(row.get("cognitive_effort_condition")),
                "state": "tetris",
                "subject_id": f"{pid}_{sid}" if sid else pid,
                "task": "tetris",
            }

    all_rows: list[dict[str, object]] = []
    for obs_id, meta in sorted(obs_meta.items()):
        all_rows.extend(
            compute_observation_channel_zlpi(
                observation_id=obs_id,
                c1a_dir=c1a,
                c1b_dir=c1b,
                identity=meta,
            )
        )
    if all_rows:
        fields = sorted({k for row in all_rows for k in row})
        _write_csv(cache_path, all_rows, fields)
    return all_rows



def _attach_coordinates(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    channels = sorted({_normalize_channel(r.get("channel", "")) for r in rows if _as_str(r.get("channel"))})
    coords = _channel_coordinates(channels)
    out: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        ch = _normalize_channel(row.get("channel", ""))
        xyz = coords.get(ch)
        if xyz is None:
            item["channel_x"] = float("nan")
            item["channel_y"] = float("nan")
            item["channel_z"] = float("nan")
            item["has_coordinates"] = False
        else:
            item["channel_x"], item["channel_y"], item["channel_z"] = xyz
            item["has_coordinates"] = True
        out.append(item)
    return out


def _apply_after_controls(
    before_rows: Sequence[Mapping[str, object]],
    *,
    ecg_prone: Sequence[str],
) -> list[dict[str, object]]:
    prone = { _normalize_channel(ch) for ch in ecg_prone }
    out: list[dict[str, object]] = []
    for row in before_rows:
        item = dict(row)
        item["control_status"] = "after_controls"
        item["control_specification"] = CONTROL_SPEC_AFTER
        ch = _normalize_channel(row.get("channel", ""))
        reasons: list[str] = []
        if not _as_bool(row.get("spectral_qc_passed")):
            reasons.append("spectral_qc_failed")
        if _as_bool(row.get("channel_rejected_by_spectral_qc")) or not _as_bool(
            row.get("channel_in_usable_set")
        ):
            reasons.append("channel_excluded_by_c1a_spectral_qc")
        if ch in prone:
            reasons.append("ecg_prone_default_v1_removed")
        if reasons:
            item["zpli_value"] = float("nan")
            item["computable"] = False
            item["not_computable_reason"] = "+".join(reasons)
            # Never substitute uncontrolled value.
        elif not _as_bool(row.get("computable")):
            item["zpli_value"] = float("nan")
            item["computable"] = False
            item["not_computable_reason"] = _as_str(
                row.get("not_computable_reason"), "baseline_not_computable"
            )
        else:
            item["computable"] = True
            item["not_computable_reason"] = ""
            item["zpli_value"] = _as_float(row.get("zpli_value"))
        out.append(item)
    return out


def _participant_channel_means(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate observation → participant mean within channel×band×state×control."""
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    meta: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        if not _as_bool(row.get("computable")):
            continue
        val = _as_float(row.get("zpli_value"))
        if not math.isfinite(val):
            continue
        key = (
            _as_str(row.get("dataset_id")),
            _as_str(row.get("participant_id")),
            _as_str(row.get("session_id")),
            _as_str(row.get("channel")),
            _as_str(row.get("band")),
            _as_str(row.get("state")),
            _as_str(row.get("control_status")),
        )
        buckets[key].append(val)
        meta[key] = row
    out: list[dict[str, object]] = []
    for key, vals in buckets.items():
        row = meta[key]
        out.append(
            {
                "dataset_id": key[0],
                "participant_id": key[1],
                "session_id": key[2],
                "channel": key[3],
                "band": key[4],
                "state": key[5],
                "control_status": key[6],
                "control_specification": row.get("control_specification"),
                "estimate": float(np.mean(vals)),
                "n_observations": len(vals),
                "channel_x": row.get("channel_x"),
                "channel_y": row.get("channel_y"),
                "channel_z": row.get("channel_z"),
                "aggregation_level": "participant",
                "aggregation_method": "mean_of_observation_channel_zlpi",
            }
        )
    return out


def _group_channel_means(
    participant_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Equal-weight mean across participants (within dataset) per channel."""
    buckets: dict[tuple[str, ...], list[tuple[str, float]]] = defaultdict(list)
    meta: dict[tuple[str, ...], Mapping[str, object]] = {}
    for row in participant_rows:
        key = (
            _as_str(row.get("dataset_id")),
            _as_str(row.get("channel")),
            _as_str(row.get("band")),
            _as_str(row.get("state")),
            _as_str(row.get("control_status")),
        )
        pid = _as_str(row.get("participant_id"))
        buckets[key].append((pid, _as_float(row.get("estimate"))))
        meta[key] = row
    out: list[dict[str, object]] = []
    for key, items in buckets.items():
        # one value per participant (already participant-level); if multiple sessions, mean first
        by_pid: dict[str, list[float]] = defaultdict(list)
        for pid, val in items:
            if math.isfinite(val):
                by_pid[pid].append(val)
        pid_means = [float(np.mean(v)) for v in by_pid.values() if v]
        if not pid_means:
            continue
        row = meta[key]
        out.append(
            {
                "dataset_id": key[0],
                "channel": key[1],
                "band": key[2],
                "state": key[3],
                "control_status": key[4],
                "estimate": float(np.mean(pid_means)),
                "n_participants": len(pid_means),
                "n_datasets": 1,
                "channel_x": row.get("channel_x"),
                "channel_y": row.get("channel_y"),
                "channel_z": row.get("channel_z"),
                "aggregation_level": "group",
                "aggregation_method": (
                    "observation_mean_within_participant_then_equal_weight_participant_mean"
                ),
            }
        )
    return out


def _paired_task_minus_rest_alpha(
    before_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_obs_ch: dict[tuple[str, str], dict[str, object]] = {}
    for row in before_rows:
        if _as_str(row.get("band")).casefold() != PANEL_F_ALPHA_BAND:
            continue
        if not _as_bool(row.get("computable")):
            continue
        key = (_as_str(row.get("observation_id")), _normalize_channel(row.get("channel", "")))
        by_obs_ch[key] = dict(row)

    out: list[dict[str, object]] = []
    for prow in _select_paired_alpha_rows(paired_rows):
        low = _as_str(prow.get("low_observation_ids")).split(";")[0]
        effort = _as_str(prow.get("effort_observation_ids")).split(";")[0]
        channels = {
            ch
            for (obs, ch) in by_obs_ch
            if obs in {low, effort}
        }
        for ch in sorted(channels):
            rest = by_obs_ch.get((low, ch))
            task = by_obs_ch.get((effort, ch))
            if rest is None or task is None:
                out.append(
                    {
                        "dataset_id": _as_str(prow.get("dataset_id")),
                        "participant_id": _as_str(prow.get("participant_id")),
                        "session_id": _as_str(prow.get("session_id")),
                        "observation_id": f"{low}__{effort}",
                        "low_observation_id": low,
                        "effort_observation_id": effort,
                        "state": "task_minus_rest",
                        "band": PANEL_F_ALPHA_BAND,
                        "channel": ch,
                        "control_status": "before_controls",
                        "control_specification": CONTROL_SPEC_BEFORE,
                        "zpli_value": float("nan"),
                        "computable": False,
                        "not_computable_reason": "missing_rest_or_task_channel_zlpi",
                        "duration_s": PANEL_F_DURATION_S,
                        "endpoint_name": PANEL_F_ENDPOINT,
                        "endpoint_units": "Fisher_z_delta_ZLPI",
                        "power_representation": PANEL_F_REPRESENTATION,
                        "montage_name": "standard_1020",
                        "channel_x": (rest or task or {}).get("channel_x"),
                        "channel_y": (rest or task or {}).get("channel_y"),
                        "channel_z": (rest or task or {}).get("channel_z"),
                    }
                )
                continue
            delta = _as_float(task.get("zpli_value")) - _as_float(rest.get("zpli_value"))
            out.append(
                {
                    "dataset_id": _as_str(prow.get("dataset_id")),
                    "participant_id": _as_str(prow.get("participant_id")),
                    "session_id": _as_str(prow.get("session_id")),
                    "observation_id": f"{low}__{effort}",
                    "low_observation_id": low,
                    "effort_observation_id": effort,
                    "state": "task_minus_rest",
                    "band": PANEL_F_ALPHA_BAND,
                    "channel": ch,
                    "control_status": "before_controls",
                    "control_specification": CONTROL_SPEC_BEFORE,
                    "zpli_value": delta,
                    "computable": math.isfinite(delta),
                    "not_computable_reason": "" if math.isfinite(delta) else "nonfinite_delta",
                    "duration_s": PANEL_F_DURATION_S,
                    "endpoint_name": PANEL_F_ENDPOINT,
                    "endpoint_units": "Fisher_z_delta_ZLPI",
                    "power_representation": PANEL_F_REPRESENTATION,
                    "montage_name": "standard_1020",
                    "spectral_qc_passed": True,
                    "channel_in_usable_set": True,
                    "channel_rejected_by_spectral_qc": False,
                    "channel_x": rest.get("channel_x"),
                    "channel_y": rest.get("channel_y"),
                    "channel_z": rest.get("channel_z"),
                }
            )
    return out


def _symmetric_limit(values: Sequence[float]) -> float:
    finite = [abs(float(v)) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.05
    return float(max(finite))


def _cluster_boot_mean(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    n_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    cl = np.asarray(list(clusters), dtype=object)
    mask = np.isfinite(arr)
    arr = arr[mask]
    cl = cl[mask]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(arr))
    uniq = sorted(set(cl.tolist()))
    if len(uniq) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    by = {u: arr[cl == u] for u in uniq}
    for _ in range(n_draws):
        draw = [float(np.mean(by[u])) for u in rng.choice(uniq, size=len(uniq), replace=True)]
        boots.append(float(np.mean(draw)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def _ceil_sig(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        return 0.05
    exp = math.floor(math.log10(x))
    factor = 10 ** (exp - 2)
    return math.ceil(x / factor) * factor


def _scalp_summary_by_participant(
    rows: Sequence[Mapping[str, object]],
    channels: Sequence[str],
) -> list[dict[str, object]]:
    """Participant-level mean signed ZLPI and mean |ZLPI| over a channel set."""
    ch_set = {_normalize_channel(c) for c in channels}
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not _as_bool(row.get("computable")):
            continue
        ch = _normalize_channel(row.get("channel", ""))
        if ch not in ch_set:
            continue
        val = _as_float(row.get("zpli_value"))
        if not math.isfinite(val):
            continue
        key = (_as_str(row.get("dataset_id")), _as_str(row.get("participant_id")))
        buckets[key].append(val)
    out: list[dict[str, object]] = []
    for (ds, pid), vals in sorted(buckets.items()):
        out.append(
            {
                "dataset_id": ds,
                "participant_id": pid,
                "n_channels": len(vals),
                "mean_signed_zlpi": float(np.mean(vals)),
                "mean_abs_zlpi": float(np.mean(np.abs(vals))),
                "median_signed_zlpi": float(np.median(vals)),
                "median_abs_zlpi": float(np.median(np.abs(vals))),
            }
        )
    return out


def compute_panel_f_topography(
    *,
    confirmatory_root: Path,
    paired_rows: Sequence[Mapping[str, object]],
    force_recompute_channel_zlpi: bool = False,
) -> PanelFResult:
    """Compute Panel F maps and ECG-prone restricted-montage sensitivity."""
    set_id, ecg_prone = load_ecg_prone_channels()
    prone_set = set(ecg_prone)
    before_raw = build_or_load_channel_zlpi_table(
        confirmatory_root=confirmatory_root,
        paired_rows=paired_rows,
        force_recompute=force_recompute_channel_zlpi,
    )
    before_raw = _attach_coordinates(before_raw)
    before_rows = [
        r
        for r in before_raw
        if _as_str(r.get("band")).casefold() in {PANEL_F_ALPHA_BAND, PANEL_F_GAMMA_BAND}
        and _as_str(r.get("channel"))
    ]
    after_rows = _apply_after_controls(
        [r for r in before_rows if _as_str(r.get("band")).casefold() == PANEL_F_GAMMA_BAND],
        ecg_prone=ecg_prone,
    )
    atten_rows = _attach_coordinates(_paired_task_minus_rest_alpha(before_rows, paired_rows))

    rest_alpha_obs = [
        r for r in before_rows
        if _as_str(r.get("band")).casefold() == PANEL_F_ALPHA_BAND
        and _as_str(r.get("state")).casefold() == "rest"
        and _as_bool(r.get("computable"))
        and math.isfinite(_as_float(r.get("channel_x")))
    ]
    atten_obs = [
        r for r in atten_rows
        if _as_bool(r.get("computable")) and math.isfinite(_as_float(r.get("channel_x")))
    ]
    gamma_full_obs = [
        r for r in before_rows
        if _as_str(r.get("band")).casefold() == PANEL_F_GAMMA_BAND
        and _as_str(r.get("state")).casefold() == "rest"
        and _as_bool(r.get("computable"))
        and math.isfinite(_as_float(r.get("channel_x")))
    ]
    gamma_restricted_obs = [
        r for r in after_rows
        if _as_str(r.get("state")).casefold() == "rest"
        and _as_bool(r.get("computable"))
        and math.isfinite(_as_float(r.get("channel_x")))
    ]

    def _chs(rows: Sequence[Mapping[str, object]]) -> set[str]:
        return {_normalize_channel(r.get("channel", "")) for r in rows}

    # Alpha common montage: channels present for rest and attenuation.
    # Exclude ECG-prone so F1/F2 match the restricted sensitivity montage used elsewhere.
    alpha_channels = sorted((_chs(rest_alpha_obs) & _chs(atten_obs)) - prone_set)
    gamma_full_channels = sorted(_chs(gamma_full_obs))
    gamma_restricted_channels = sorted(_chs(gamma_restricted_obs) - prone_set)
    # Keep alpha and restricted gamma on the same retained set for layout consistency.
    retained_channels = sorted(set(alpha_channels) & set(gamma_restricted_channels))
    alpha_channels = retained_channels
    gamma_restricted_channels = retained_channels
    excluded_ecg_channels = sorted(prone_set & _chs(gamma_full_obs))

    montage_rows: list[dict[str, object]] = []
    for ch in sorted(set(gamma_full_channels) | set(retained_channels) | prone_set):
        xyz = _channel_coordinates([ch]).get(ch)
        role = []
        if ch in retained_channels:
            role.append("retained_common")
        if ch in excluded_ecg_channels:
            role.append("ecg_prone_excluded")
        if ch in gamma_full_channels and ch not in retained_channels and ch not in prone_set:
            role.append("gamma_full_only")
        montage_rows.append(
            {
                "channel": ch,
                "channel_x": xyz[0] if xyz else float("nan"),
                "channel_y": xyz[1] if xyz else float("nan"),
                "channel_z": xyz[2] if xyz else float("nan"),
                "montage_name": "standard_1020",
                "in_alpha_common_montage": ch in retained_channels,
                "in_gamma_full_montage": ch in gamma_full_channels,
                "in_gamma_restricted_montage": ch in gamma_restricted_channels,
                "ecg_prone_default_v1": ch in prone_set,
                "montage_role": "+".join(role) or "unused",
            }
        )

    def _filter(rows, channels):
        chs = set(channels)
        return [dict(r) for r in rows if _normalize_channel(r.get("channel", "")) in chs]

    rest_alpha_c = _filter(rest_alpha_obs, alpha_channels)
    atten_c = _filter(atten_obs, alpha_channels)
    gamma_full_c = _filter(gamma_full_obs, gamma_full_channels)
    gamma_rest_c = _filter(gamma_restricted_obs, gamma_restricted_channels)

    # Same observations for inclusive vs restricted scalp summaries.
    obs_full = {_as_str(r.get("observation_id")) for r in gamma_full_c}
    obs_rest = {_as_str(r.get("observation_id")) for r in gamma_rest_c}
    common_obs = obs_full & obs_rest
    gamma_full_c = [r for r in gamma_full_c if _as_str(r.get("observation_id")) in common_obs]
    gamma_rest_c = [r for r in gamma_rest_c if _as_str(r.get("observation_id")) in common_obs]

    map_specs = [
        (MAP_REST_ALPHA, rest_alpha_c, "alpha_shared", alpha_channels),
        (MAP_TASK_ATTENUATION, atten_c, "alpha_shared", alpha_channels),
        (MAP_GAMMA_BEFORE, gamma_full_c, "gamma_shared", gamma_full_channels),
        (MAP_GAMMA_AFTER, gamma_rest_c, "gamma_shared", gamma_restricted_channels),
    ]

    summary_rows: list[dict[str, object]] = []
    values_by_scale: dict[str, list[float]] = defaultdict(list)
    for map_id, rows, scale_group, _chs_used in map_specs:
        if map_id == MAP_TASK_ATTENUATION:
            part = _participant_channel_means(
                [{**r, "state": "task_minus_rest", "control_status": "before_controls"} for r in rows]
            )
        else:
            part = _participant_channel_means(rows)
        group = _group_channel_means(part)
        for r in group:
            est = _as_float(r.get("estimate"))
            values_by_scale[scale_group].append(est)
            summary_rows.append(
                {
                    "panel_map": map_id,
                    "channel": r.get("channel"),
                    "estimate": est,
                    "lower_ci": float("nan"),
                    "upper_ci": float("nan"),
                    "n_observations": sum(
                        int(p.get("n_observations") or 0)
                        for p in part
                        if _as_str(p.get("channel")) == _as_str(r.get("channel"))
                    ),
                    "n_participants": r.get("n_participants"),
                    "n_datasets": r.get("n_datasets"),
                    "color_scale_group": scale_group,
                    "aggregation_method": r.get("aggregation_method"),
                    "channel_x": r.get("channel_x"),
                    "channel_y": r.get("channel_y"),
                    "channel_z": r.get("channel_z"),
                    "band": PANEL_F_ALPHA_BAND
                    if map_id in {MAP_REST_ALPHA, MAP_TASK_ATTENUATION}
                    else PANEL_F_GAMMA_BAND,
                    "ecg_prone_default_v1": _as_str(r.get("channel")) in prone_set,
                    "montage_membership": (
                        "gamma_full_pre_exclusion"
                        if map_id == MAP_GAMMA_BEFORE
                        else (
                            "gamma_restricted_post_exclusion"
                            if map_id == MAP_GAMMA_AFTER
                            else "alpha_common_retained"
                        )
                    ),
                }
            )

    alpha_L = _ceil_sig(_symmetric_limit(values_by_scale["alpha_shared"]))
    gamma_L = _ceil_sig(_symmetric_limit(values_by_scale["gamma_shared"]))
    for row in summary_rows:
        if row["color_scale_group"] == "alpha_shared":
            row["color_limit_min"] = -alpha_L
            row["color_limit_max"] = alpha_L
        else:
            row["color_limit_min"] = -gamma_L
            row["color_limit_max"] = gamma_L

    # A. Deterministic retained-channel identity (no bootstrap).
    retained_before = {
        (_as_str(r.get("observation_id")), _normalize_channel(r.get("channel", ""))): _as_float(r.get("zpli_value"))
        for r in gamma_full_c
        if _normalize_channel(r.get("channel", "")) in set(retained_channels)
    }
    retained_after = {
        (_as_str(r.get("observation_id")), _normalize_channel(r.get("channel", ""))): _as_float(r.get("zpli_value"))
        for r in gamma_rest_c
    }
    identity_keys = sorted(set(retained_before) & set(retained_after))
    max_abs_diff = 0.0
    n_equal = 0
    for key in identity_keys:
        diff = retained_after[key] - retained_before[key]
        max_abs_diff = max(max_abs_diff, abs(diff))
        if abs(diff) <= 1e-15:
            n_equal += 1

    # B. Montage-composition sensitivity at participant level.
    inclusive_part = _scalp_summary_by_participant(gamma_full_c, gamma_full_channels)
    restricted_part = _scalp_summary_by_participant(gamma_rest_c, gamma_restricted_channels)
    by_inc = {(_as_str(r["dataset_id"]), _as_str(r["participant_id"])): r for r in inclusive_part}
    by_res = {(_as_str(r["dataset_id"]), _as_str(r["participant_id"])): r for r in restricted_part}
    paired_pids = sorted(set(by_inc) & set(by_res))
    signed_changes = []
    abs_changes = []
    clusters = []
    for key in paired_pids:
        signed_changes.append(float(by_res[key]["mean_signed_zlpi"]) - float(by_inc[key]["mean_signed_zlpi"]))
        abs_changes.append(float(by_res[key]["mean_abs_zlpi"]) - float(by_inc[key]["mean_abs_zlpi"]))
        clusters.append(key[1])
    signed_mean, signed_lo, signed_hi = _cluster_boot_mean(signed_changes, clusters, seed=BOOTSTRAP_SEED)
    abs_mean, abs_lo, abs_hi = _cluster_boot_mean(abs_changes, clusters, seed=BOOTSTRAP_SEED + 1)

    # Excluded vs retained channel group summaries (from F3 full-montage group estimates).
    g_full = {
        _as_str(r.get("channel")): _as_float(r.get("estimate"))
        for r in summary_rows
        if _as_str(r.get("panel_map")) == MAP_GAMMA_BEFORE
    }
    excl_vals = [g_full[c] for c in excluded_ecg_channels if c in g_full and math.isfinite(g_full[c])]
    ret_vals = [g_full[c] for c in retained_channels if c in g_full and math.isfinite(g_full[c])]

    gamma_comparison_rows = [
        {
            "metric": "retained_channel_value_identity",
            "value": 0.0 if max_abs_diff <= 1e-15 else max_abs_diff,
            "max_abs_after_minus_before": max_abs_diff,
            "n_pairs_checked": len(identity_keys),
            "n_exact_equalities": n_equal,
            "is_deterministic_identity": max_abs_diff <= 1e-15,
            "bootstrap_used": False,
            "interpretation": (
                "Deterministic implementation identity: no value-changing transform on retained "
                "channels; equality is not an empirical robustness finding."
            ),
        },
        {
            "metric": "participant_mean_signed_zlpi_inclusive_montage",
            "value": float(np.mean([by_inc[k]["mean_signed_zlpi"] for k in paired_pids])) if paired_pids else float("nan"),
            "n_participants": len(paired_pids),
            "n_channels": len(gamma_full_channels),
        },
        {
            "metric": "participant_mean_signed_zlpi_restricted_montage",
            "value": float(np.mean([by_res[k]["mean_signed_zlpi"] for k in paired_pids])) if paired_pids else float("nan"),
            "n_participants": len(paired_pids),
            "n_channels": len(gamma_restricted_channels),
        },
        {
            "metric": "paired_restricted_minus_inclusive_mean_signed_zlpi",
            "value": signed_mean,
            "ci_lower": signed_lo,
            "ci_upper": signed_hi,
            "n_participants": len(paired_pids),
            "ci_method": "participant_cluster_bootstrap",
        },
        {
            "metric": "participant_mean_abs_zlpi_inclusive_montage",
            "value": float(np.mean([by_inc[k]["mean_abs_zlpi"] for k in paired_pids])) if paired_pids else float("nan"),
            "n_participants": len(paired_pids),
        },
        {
            "metric": "participant_mean_abs_zlpi_restricted_montage",
            "value": float(np.mean([by_res[k]["mean_abs_zlpi"] for k in paired_pids])) if paired_pids else float("nan"),
            "n_participants": len(paired_pids),
        },
        {
            "metric": "paired_restricted_minus_inclusive_mean_abs_zlpi",
            "value": abs_mean,
            "ci_lower": abs_lo,
            "ci_upper": abs_hi,
            "n_participants": len(paired_pids),
            "ci_method": "participant_cluster_bootstrap",
        },
        {
            "metric": "excluded_channels_mean_signed_zlpi",
            "value": float(np.mean(excl_vals)) if excl_vals else float("nan"),
            "n_channels": len(excl_vals),
            "channels": ";".join(excluded_ecg_channels),
        },
        {
            "metric": "excluded_channels_mean_abs_zlpi",
            "value": float(np.mean(np.abs(excl_vals))) if excl_vals else float("nan"),
            "n_channels": len(excl_vals),
        },
        {
            "metric": "retained_channels_mean_signed_zlpi",
            "value": float(np.mean(ret_vals)) if ret_vals else float("nan"),
            "n_channels": len(ret_vals),
        },
        {
            "metric": "retained_channels_mean_abs_zlpi",
            "value": float(np.mean(np.abs(ret_vals))) if ret_vals else float("nan"),
            "n_channels": len(ret_vals),
        },
        {
            "metric": "excluded_minus_retained_mean_abs_zlpi",
            "value": (
                float(np.mean(np.abs(excl_vals)) - np.mean(np.abs(ret_vals)))
                if excl_vals and ret_vals
                else float("nan")
            ),
            "interpretation": (
                "Positive values indicate ECG-prone excluded channels have larger absolute "
                "gamma ZLPI than retained channels on the group topography."
            ),
        },
        {
            "metric": "n_gamma_full_channels",
            "value": len(gamma_full_channels),
            "channels": ";".join(gamma_full_channels),
        },
        {
            "metric": "n_ecg_prone_excluded_channels",
            "value": len(excluded_ecg_channels),
            "channels": ";".join(excluded_ecg_channels),
        },
        {
            "metric": "n_retained_channels",
            "value": len(retained_channels),
            "channels": ";".join(retained_channels),
        },
    ]

    # Per-excluded-channel group estimates for diagnostic export rows.
    for ch in excluded_ecg_channels:
        gamma_comparison_rows.append(
            {
                "metric": "excluded_channel_group_estimate",
                "channel": ch,
                "value": g_full.get(ch, float("nan")),
                "ecg_prone_default_v1": True,
            }
        )

    observation_export = []
    for r in rest_alpha_c:
        observation_export.append({**r, "panel_map": MAP_REST_ALPHA})
    for r in atten_c:
        observation_export.append({**r, "panel_map": MAP_TASK_ATTENUATION})
    for r in gamma_full_c:
        observation_export.append(
            {
                **r,
                "panel_map": MAP_GAMMA_BEFORE,
                "control_status": "before_channel_exclusion",
                "ecg_prone_default_v1": _normalize_channel(r.get("channel", "")) in prone_set,
            }
        )
    for r in gamma_rest_c:
        observation_export.append(
            {
                **r,
                "panel_map": MAP_GAMMA_AFTER,
                "control_status": "after_ecg_prone_exclusion",
            }
        )

    participant_scalp_rows: list[dict[str, object]] = []
    for r in inclusive_part:
        participant_scalp_rows.append({**r, "montage": "inclusive_pre_exclusion"})
    for r in restricted_part:
        participant_scalp_rows.append({**r, "montage": "restricted_post_exclusion"})

    diagnostic_rows = [
        {"diagnostic_type": "unavailable_controls", "controls": UNAVAILABLE_CONTROLS},
        {
            "diagnostic_type": "montages",
            "gamma_full_channels": gamma_full_channels,
            "ecg_prone_excluded_channels": excluded_ecg_channels,
            "retained_channels": retained_channels,
            "alpha_common_channels": alpha_channels,
        },
        {
            "diagnostic_type": "retained_channel_identity_check",
            "max_abs_after_minus_before": max_abs_diff,
            "n_pairs": len(identity_keys),
            "is_deterministic_identity": max_abs_diff <= 1e-15,
            "bootstrap_forbidden": True,
        },
        {
            "diagnostic_type": "color_limits",
            "alpha_shared_L": alpha_L,
            "gamma_shared_L": gamma_L,
        },
        *[
            {"diagnostic_type": "participant_scalp_summary", **r}
            for r in participant_scalp_rows
        ],
        *[
            {
                "diagnostic_type": "channel_list",
                "list_name": name,
                "n": len(chs),
                "channels": ";".join(chs),
            }
            for name, chs in (
                ("gamma_full_usable_pre_exclusion", gamma_full_channels),
                ("ecg_prone_excluded", excluded_ecg_channels),
                ("retained_common_53_intersection", retained_channels),
            )
        ],
    ]

    metadata = {
        "schema_version": "figure3_panel_f_topography_gamma_v2_montage_sensitivity",
        "title": PANEL_F_FIGURE_TITLE,
        "panel_title": PANEL_F_TITLE,
        "subtitle": PANEL_F_SUBTITLE,
        "stem": PANEL_F_STEM,
        "analysis_framing": "gamma_sensitivity_to_ecg_prone_channel_exclusion",
        "locked_estimand": {
            "duration_s": PANEL_F_DURATION_S,
            "endpoint_name": PANEL_F_ENDPOINT,
            "power_representation": PANEL_F_REPRESENTATION,
            "alpha_band": PANEL_F_ALPHA_BAND,
            "gamma_band": PANEL_F_GAMMA_BAND,
            "task_attenuation_definition": "ZLPI_task - ZLPI_rest (negative = attenuation)",
        },
        "gamma_maps": {
            "F3": "full usable pre-exclusion montage; ECG-prone sensors marked",
            "F4": "restricted montage after ecg_prone_default_v1 exclusion",
            "not_independent_signal_estimates": True,
            "differ_by": "channel_membership_only",
        },
        "artifact_controls": {
            "restricted_montage_control": {
                "control": set_id,
                "channels": list(ecg_prone),
            },
            "spectral_qc": "observations/channels outside C1a usable set excluded upstream",
            "unavailable_or_not_computable": list(UNAVAILABLE_CONTROLS),
        },
        "aggregation_method": (
            "channel×observation ZLPI → participant mean → equal-weight participant mean"
        ),
        "color_scales": {
            "alpha_shared": {"min": -alpha_L, "max": alpha_L, "maps": [MAP_REST_ALPHA, MAP_TASK_ATTENUATION]},
            "gamma_shared": {"min": -gamma_L, "max": gamma_L, "maps": [MAP_GAMMA_BEFORE, MAP_GAMMA_AFTER]},
            "justification": (
                "Gamma full and restricted maps share identical symmetric limits so "
                "montage-composition differences are not hidden by autoscaling."
            ),
        },
        "n_alpha_common_channels": len(alpha_channels),
        "n_gamma_full_channels": len(gamma_full_channels),
        "n_retained_channels": len(retained_channels),
        "n_ecg_prone_excluded": len(excluded_ecg_channels),
        "n_gamma_paired_observations": len(common_obs),
        "n_participants_gamma": len(paired_pids),
        "retained_channel_identity_is_deterministic": max_abs_diff <= 1e-15,
        "interpretation": (
            "Removal of ECG-prone channels changes montage composition but does not alter "
            "ZLPI estimates at retained channels. Therefore, equality of retained-channel "
            "before and after values is deterministic and does not establish gamma "
            "robustness to physiological artifact. The relevant sensitivity result is "
            "whether ECG-prone channels show disproportionate gamma magnitude and whether "
            "their exclusion changes the participant-level scalp summary. Because ICA, "
            "EMG, motion, EOG, and cardiac-template controls were unavailable, the "
            "low-gamma finding remains artifact-indeterminate."
        ),
        "no_imputation_policy": "Non-computable/removed channels never assigned numerical zero.",
        "standalone_sheet": True,
        "does_not_modify_figure3_panels_a_to_e": True,
    }

    return PanelFResult(
        observation_rows=tuple(observation_export),
        summary_rows=tuple(summary_rows),
        montage_rows=tuple(montage_rows),
        gamma_comparison_rows=tuple(gamma_comparison_rows),
        diagnostic_rows=tuple(diagnostic_rows),
        metadata=metadata,
    )



def write_panel_f_exports(result: PanelFResult, source_dir: Path) -> dict[str, Path]:
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _dump(name: str, rows: Sequence[Mapping[str, object]]) -> Path:
        path = source_dir / f"{PANEL_F_STEM}_{name}.csv"
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        _write_csv(path, rows, fields or ["empty"])
        return path

    paths["observations"] = _dump("observation_level", result.observation_rows)
    paths["summary"] = _dump("summary", result.summary_rows)
    paths["montage"] = _dump("montage_membership", result.montage_rows)
    # Keep legacy alias filename for downstream discoverability.
    paths["common_montage"] = _dump("common_montage", result.montage_rows)
    paths["gamma_montage_sensitivity"] = _dump(
        "gamma_montage_sensitivity", result.gamma_comparison_rows
    )
    paths["gamma_control_comparison"] = _dump(
        "gamma_control_comparison", result.gamma_comparison_rows
    )
    participant_scalp = [
        {k: v for k, v in r.items() if k != "diagnostic_type"}
        for r in result.diagnostic_rows
        if _as_str(r.get("diagnostic_type")) == "participant_scalp_summary"
    ]
    if participant_scalp:
        paths["participant_scalp"] = _dump("participant_scalp_summaries", participant_scalp)
    channel_lists = [
        {k: v for k, v in r.items() if k != "diagnostic_type"}
        for r in result.diagnostic_rows
        if _as_str(r.get("diagnostic_type")) == "channel_list"
    ]
    if channel_lists:
        paths["channel_lists"] = _dump("channel_lists", channel_lists)
    paths["diagnostics"] = _dump(
        "diagnostics",
        [
            {
                "diagnostic_type": r.get("diagnostic_type"),
                "payload_json": json.dumps(r, default=str),
            }
            for r in result.diagnostic_rows
        ],
    )
    meta_path = source_dir / f"{PANEL_F_STEM}_metadata.json"
    meta_path.write_text(
        json.dumps(result.metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    paths["metadata"] = meta_path
    return paths


def panel_f_caption(result: PanelFResult) -> str:
    meta = result.metadata
    lines = [
        PANEL_F_FIGURE_TITLE,
        "",
        "Gamma sensitivity to ECG-prone channel exclusion.",
        "",
        "F, Topography and gamma specificity. Rest alpha ZLPI (F1) and task-minus-rest "
        "alpha ZLPI (F2) use the retained common montage. Low-gamma maps show restricted-"
        "montage sensitivity to "
        f"{ECG_PRONE_SET_ID} exclusion: F3 is the full usable pre-exclusion montage "
        "(ECG-prone sensors marked); F4 is the restricted montage after ECG-prone channel "
        "exclusion. Maps within each comparison use identical symmetric color scales "
        "centered at zero. F3 and F4 differ by channel membership only; retained-channel "
        "values are unchanged by exclusion (deterministic implementation identity, not an "
        "empirical robustness result). The scientifically relevant sensitivity contrast is "
        "whether ECG-prone channels show disproportionate gamma magnitude and whether "
        "their exclusion changes the participant-level scalp summary. "
        "Unavailable controls (not performed): ICA muscle-component rejection, EMG "
        "residualization, motion, EOG, and cardiac-template subtraction. C1a spectral QC "
        "exclusions are applied upstream. Because those physiological controls were "
        "unavailable, the low-gamma finding remains artifact-indeterminate. "
        "On F4, topographic color is interpolated from retained sensors only; excluded "
        "channels are absent from the montage and are not assigned zero. "
        f"Alpha common montage n={meta.get('n_alpha_common_channels', meta.get('n_retained_channels'))}; "
        f"gamma full montage n={meta.get('n_gamma_full_channels')}; "
        f"retained n={meta.get('n_retained_channels')}; "
        f"gamma paired observations n={meta.get('n_gamma_paired_observations')}.",
    ]
    return "\n".join(lines) + "\n"


def verify_panel_f_integrity(result: PanelFResult) -> list[str]:
    failures: list[str] = []
    retained = {
        _as_str(r.get("channel"))
        for r in result.montage_rows
        if _as_bool(r.get("in_gamma_restricted_montage")) or _as_bool(r.get("in_alpha_common_montage"))
    }
    # Backward-compatible alias if older fixtures only set in_common_montage.
    if not retained:
        retained = {
            _as_str(r.get("channel"))
            for r in result.montage_rows
            if _as_bool(r.get("in_common_montage"))
        }
    gamma_full = {
        _as_str(r.get("channel"))
        for r in result.montage_rows
        if _as_bool(r.get("in_gamma_full_montage"))
    }
    if not retained:
        failures.append("retained / common montage is empty")

    by_map: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in result.summary_rows:
        by_map[_as_str(row.get("panel_map"))].append(dict(row))

    alpha_expected = retained
    for map_id in (MAP_REST_ALPHA, MAP_TASK_ATTENUATION, MAP_GAMMA_AFTER):
        rows = by_map.get(map_id, [])
        chs = {_as_str(r.get("channel")) for r in rows}
        if alpha_expected and chs != alpha_expected:
            failures.append(f"{map_id}: summary channels != retained montage")

    g_before = by_map.get(MAP_GAMMA_BEFORE, [])
    g_after = by_map.get(MAP_GAMMA_AFTER, [])
    before_chs = {_as_str(r.get("channel")) for r in g_before}
    after_chs = {_as_str(r.get("channel")) for r in g_after}
    has_full_flag = any(_as_bool(r.get("in_gamma_full_montage")) for r in result.montage_rows)
    if has_full_flag and gamma_full and before_chs != gamma_full:
        failures.append(f"{MAP_GAMMA_BEFORE}: summary channels != gamma full montage")
    # When montage membership declares a fuller F3 than F4, channel sets must differ.
    if has_full_flag and gamma_full and retained and gamma_full != retained:
        if before_chs == after_chs:
            failures.append(
                "F3 and F4 have identical channel sets despite distinct montage membership; "
                "do not present as independent signal estimates"
            )
        if before_chs and after_chs and not after_chs.issubset(before_chs):
            failures.append("restricted montage channels are not a subset of full montage")

    if g_before and g_after:
        lims_b = {
            (round(_as_float(r.get("color_limit_min")), 12), round(_as_float(r.get("color_limit_max")), 12))
            for r in g_before
        }
        lims_a = {
            (round(_as_float(r.get("color_limit_min")), 12), round(_as_float(r.get("color_limit_max")), 12))
            for r in g_after
        }
        if lims_b != lims_a or len(lims_b) != 1:
            failures.append("gamma before/after color limits differ")

    # No NC → 0
    for row in result.observation_rows:
        if not _as_bool(row.get("computable")):
            val = _as_float(row.get("zpli_value"))
            if math.isfinite(val) and val == 0.0:
                failures.append(
                    f"noncomputable assigned numerical zero: {row.get('panel_map')} {row.get('channel')}"
                )

    # Deterministic identity must not be bootstrapped.
    for row in result.gamma_comparison_rows:
        metric = _as_str(row.get("metric"))
        if metric == "retained_channel_value_identity":
            if _as_bool(row.get("bootstrap_used")):
                failures.append("deterministic retained-channel identity was bootstrapped")
            if "ci_lower" in row or "ci_upper" in row:
                failures.append("deterministic identity must not carry a confidence interval")
            if not _as_bool(row.get("is_deterministic_identity")) and _as_float(row.get("max_abs_after_minus_before")) > 1e-12:
                failures.append("retained-channel identity failed exact equality check")
        if "after_minus_before_on_common" in metric:
            failures.append(
                "legacy paired after-before bootstrap metric must not be reported as "
                "empirical robustness"
            )

    # After never copies before when after is NC
    before_obs = {
        (_as_str(r.get("observation_id")), _as_str(r.get("channel"))): r
        for r in result.observation_rows
        if _as_str(r.get("panel_map")) == MAP_GAMMA_BEFORE
    }
    for row in result.observation_rows:
        if _as_str(row.get("panel_map")) != MAP_GAMMA_AFTER:
            continue
        if _as_bool(row.get("computable")):
            continue
        key = (_as_str(row.get("observation_id")), _as_str(row.get("channel")))
        b = before_obs.get(key)
        if b and _as_bool(b.get("computable")):
            if math.isfinite(_as_float(row.get("zpli_value"))):
                failures.append(f"after-exclusion NC substituted with value at {key}")

    # Plot-source consistency: summary estimates must be finite for retained maps.
    for map_id in (MAP_REST_ALPHA, MAP_TASK_ATTENUATION, MAP_GAMMA_AFTER):
        for r in by_map.get(map_id, []):
            if not math.isfinite(_as_float(r.get("estimate"))):
                failures.append(f"{map_id}/{r.get('channel')}: non-finite summary estimate")

    if "ZLPI_task - ZLPI_rest" not in str(
        result.metadata.get("locked_estimand", {}).get("task_attenuation_definition", "")
    ):
        failures.append("task attenuation definition missing or reversed")

    framing = _as_str(result.metadata.get("analysis_framing"))
    if framing and "ecg_prone" not in framing:
        failures.append("metadata analysis_framing must describe ECG-prone exclusion sensitivity")

    prone = {
        _as_str(r.get("channel"))
        for r in result.montage_rows
        if _as_bool(r.get("ecg_prone_default_v1"))
    }
    if after_chs and prone and (after_chs & prone):
        failures.append(
            "F4 retains ECG-prone channels that should be excluded: "
            + ", ".join(sorted(after_chs & prone))
        )
    if before_chs and prone:
        missing_on_f3 = (prone & (gamma_full or before_chs)) - before_chs
        if missing_on_f3:
            failures.append(
                "F3 omits ECG-prone channels present in the full montage: "
                + ", ".join(sorted(missing_on_f3))
            )

    # Alpha maps share identical color limits; gamma maps share identical limits.
    a1 = by_map.get(MAP_REST_ALPHA, [])
    a2 = by_map.get(MAP_TASK_ATTENUATION, [])
    if a1 and a2:
        lims1 = {
            (round(_as_float(r.get("color_limit_min")), 12), round(_as_float(r.get("color_limit_max")), 12))
            for r in a1
        }
        lims2 = {
            (round(_as_float(r.get("color_limit_min")), 12), round(_as_float(r.get("color_limit_max")), 12))
            for r in a2
        }
        if lims1 != lims2 or len(lims1) != 1:
            failures.append("alpha F1/F2 color limits differ")

    return failures


def _standard_1020_montage():
    import mne

    return mne.channels.make_standard_montage("standard_1020")


def _info_for_channels(channels: Sequence[str], *, montage=None):
    """Build a consistent MNE Info for Panel F topomaps."""
    import mne

    montage = montage if montage is not None else _standard_1020_montage()
    chs = [str(ch) for ch in channels]
    info = mne.create_info(chs, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def mark_ecg_prone_channels(
    ax,
    info,
    plotted_channels: Sequence[str],
    ecg_prone_channels: Sequence[str],
) -> int:
    """Annotate ECG-prone sensors on F3 (channel-status markers, not significance)."""
    from mne.channels.layout import _find_topomap_coords

    prone = {_normalize_channel(c) for c in ecg_prone_channels}
    plotted = {_normalize_channel(c) for c in plotted_channels}
    if not prone or info is None:
        return 0
    try:
        coords = _find_topomap_coords(info, picks=None)
    except Exception:
        return 0
    n_marked = 0
    for ch_name, (x, y) in zip(list(info["ch_names"]), coords, strict=False):
        key = _normalize_channel(ch_name)
        if key not in prone or key not in plotted:
            continue
        ax.plot(
            x,
            y,
            marker="o",
            markersize=6.5,
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.25,
            linestyle="none",
            zorder=10,
        )
        ax.plot(
            x,
            y,
            marker="x",
            markersize=4.2,
            markeredgecolor="#111111",
            markeredgewidth=1.0,
            linestyle="none",
            zorder=11,
        )
        n_marked += 1
    return n_marked


def _plot_topomap(
    ax,
    channels: Sequence[str],
    values: Sequence[float],
    *,
    vmax: float,
    title: str,
    montage=None,
    cmap: str = TOPOMAP_CMAP,
) -> tuple[object, object, list[str], list[float]]:
    """Render one Panel F topomap with locked MNE settings.

    Returns ``(image, info, plotted_channels, plotted_values)``.
    Non-finite values are omitted; they are never replaced with zero.
    """
    montage = montage if montage is not None else _standard_1020_montage()
    pos_map = montage.get_positions()["ch_pos"]
    keep: list[str] = []
    vals: list[float] = []
    for ch, v in zip(list(channels), list(values), strict=True):
        key = next((k for k in pos_map if k.casefold() == str(ch).casefold()), None)
        if key is None or not math.isfinite(float(v)):
            continue
        keep.append(str(ch))
        vals.append(float(v))
    if not keep:
        ax.set_axis_off()
        ax.set_title(title + "\n(no channels)", fontsize=11, fontweight="semibold", pad=10)
        return None, None, [], []

    import mne

    info = _info_for_channels(keep, montage=montage)
    data = np.asarray(vals, dtype=float)
    im, _ = mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(-float(vmax), float(vmax)),
        contours=TOPOMAP_CONTOURS,
        sensors=True,
        names=None,
        outlines="head",
        image_interp=TOPOMAP_INTERP,
        extrapolate=TOPOMAP_EXTRAPOLATE,
        border="mean",
        res=64,
    )
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=10)
    return im, info, keep, vals


def _symmetric_vmax(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return 0.05
    vmax = float(np.max(np.abs(arr)))
    return vmax if vmax > 0 else 0.05


def _format_cbar_ticks(cbar, vmax: float) -> None:
    ticks = [-float(vmax), 0.0, float(vmax)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.4f}" for t in ticks])


def render_panel_f_figure(
    result: PanelFResult,
    output_dir: Path,
    *,
    include_internal_qc: bool = True,
) -> dict[str, Path]:
    """Render standalone Panel F topography sheet (display-only)."""
    import mne

    from .figures import (
        FS_TICK,
        FIGURE_DPI,
        _configure_publication_style,
        save_figure_trio,
    )

    del mne  # imported for availability; helpers import as needed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"

    # Snapshot estimates before export/render to prove plotting does not mutate them.
    estimates_before = [
        (_as_str(r.get("panel_map")), _as_str(r.get("channel")), _as_float(r.get("estimate")))
        for r in result.summary_rows
    ]

    export_paths = write_panel_f_exports(result, source_dir)
    caption_path = output_dir / f"{PANEL_F_STEM}_caption.txt"
    caption_path.write_text(panel_f_caption(result), encoding="utf-8")

    integrity = verify_panel_f_integrity(result)
    if integrity:
        raise ValueError("Panel F integrity failed: " + "; ".join(integrity))

    by_map: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in result.summary_rows:
        by_map[_as_str(row.get("panel_map"))][_as_str(row.get("channel"))] = dict(row)

    retained = sorted(
        {
            _as_str(r.get("channel"))
            for r in result.montage_rows
            if _as_bool(r.get("in_gamma_restricted_montage"))
            or _as_bool(r.get("in_alpha_common_montage"))
            or _as_bool(r.get("in_common_montage"))
        }
    )
    gamma_full = sorted(
        {
            _as_str(r.get("channel"))
            for r in result.montage_rows
            if _as_bool(r.get("in_gamma_full_montage"))
        }
    )
    if not gamma_full:
        gamma_full = sorted(by_map.get(MAP_GAMMA_BEFORE, {}).keys())

    _, ecg_prone_locked = load_ecg_prone_channels()
    ecg_prone = sorted({_normalize_channel(c) for c in ecg_prone_locked})

    map_channels = {
        MAP_REST_ALPHA: retained,
        MAP_TASK_ATTENUATION: retained,
        MAP_GAMMA_BEFORE: gamma_full,
        MAP_GAMMA_AFTER: retained,
    }

    def _ordered_values(map_id: str, channels: Sequence[str]) -> tuple[list[str], list[float]]:
        ch_map = by_map.get(map_id, {})
        keep = [ch for ch in channels if ch in ch_map]
        vals = [_as_float(ch_map[ch].get("estimate")) for ch in keep]
        # Confirm exact match to exported summary rows.
        for ch, val in zip(keep, vals, strict=True):
            exported = _as_float(ch_map[ch].get("estimate"))
            if not math.isclose(val, exported, rel_tol=0.0, abs_tol=0.0) and not (
                math.isfinite(val) and math.isfinite(exported) and val == exported
            ):
                raise ValueError(f"plot/source mismatch for {map_id}/{ch}")
        return keep, vals

    ch_f1, v_f1 = _ordered_values(MAP_REST_ALPHA, map_channels[MAP_REST_ALPHA])
    ch_f2, v_f2 = _ordered_values(MAP_TASK_ATTENUATION, map_channels[MAP_TASK_ATTENUATION])
    ch_f3, v_f3 = _ordered_values(MAP_GAMMA_BEFORE, map_channels[MAP_GAMMA_BEFORE])
    ch_f4, v_f4 = _ordered_values(MAP_GAMMA_AFTER, map_channels[MAP_GAMMA_AFTER])

    # F4 must not invent zeros for excluded channels.
    excluded_on_f4 = [c for c in ecg_prone if c in set(ch_f3) and c not in set(ch_f4)]
    if any(c in set(ch_f4) for c in ecg_prone):
        raise ValueError("F4 includes ECG-prone channels that should be excluded")
    if set(ch_f4) & set(ecg_prone):
        raise ValueError("F4 assigned values on ECG-prone channels")

    alpha_vmax = _symmetric_vmax(list(v_f1) + list(v_f2))
    gamma_vmax = _symmetric_vmax(list(v_f3) + list(v_f4))

    montage = _standard_1020_montage()
    _configure_publication_style()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=FIGURE_SIZE,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.76,
        bottom=0.30,
        wspace=0.20,
    )

    plot_specs = (
        (axes[0], ch_f1, v_f1, alpha_vmax, MAP_TITLES[MAP_REST_ALPHA]),
        (axes[1], ch_f2, v_f2, alpha_vmax, MAP_TITLES[MAP_TASK_ATTENUATION]),
        (axes[2], ch_f3, v_f3, gamma_vmax, MAP_TITLES[MAP_GAMMA_BEFORE]),
        (axes[3], ch_f4, v_f4, gamma_vmax, MAP_TITLES[MAP_GAMMA_AFTER]),
    )
    images: list[object] = []
    plotted_channels: dict[str, list[str]] = {}
    n_ecg_markers = 0
    info_f3 = None
    for ax, channels, values, vmax, title in plot_specs:
        im, info, keep, _vals = _plot_topomap(
            ax,
            channels,
            values,
            vmax=vmax,
            title=title,
            montage=montage,
            cmap=TOPOMAP_CMAP,
        )
        images.append(im)
        plotted_channels[title] = keep
        if title == MAP_TITLES[MAP_GAMMA_BEFORE]:
            info_f3 = info
            n_ecg_markers = mark_ecg_prone_channels(
                ax,
                info,
                keep,
                ecg_prone,
            )

    expected_ecg_on_f3 = sorted(
        c for c in ecg_prone if c in {_normalize_channel(x) for x in ch_f3}
    )
    if n_ecg_markers != len(expected_ecg_on_f3):
        raise ValueError(
            f"ECG-prone marker count mismatch: plotted={n_ecg_markers}, "
            f"expected={len(expected_ecg_on_f3)} ({expected_ecg_on_f3})"
        )

    # Color bars spanning F1–F2 and F3–F4.
    alpha_cbar_ax = fig.add_axes([0.07, 0.14, 0.40, 0.025])
    gamma_cbar_ax = fig.add_axes([0.53, 0.14, 0.40, 0.025])
    alpha_im = images[0]
    gamma_im = images[2]
    if alpha_im is None or gamma_im is None:
        raise ValueError("Panel F topomap images missing; cannot build color bars")
    alpha_cbar = fig.colorbar(alpha_im, cax=alpha_cbar_ax, orientation="horizontal")
    gamma_cbar = fig.colorbar(gamma_im, cax=gamma_cbar_ax, orientation="horizontal")
    alpha_cbar.set_label(PANEL_F_CBAR_ALPHA, fontsize=FS_TICK - 1)
    gamma_cbar.set_label(PANEL_F_CBAR_GAMMA, fontsize=FS_TICK - 1)
    _format_cbar_ticks(alpha_cbar, alpha_vmax)
    _format_cbar_ticks(gamma_cbar, gamma_vmax)
    # Gamma color-bar ticks must match F3/F4 shared limits exactly.
    if list(gamma_cbar.get_ticks()) != list(alpha_cbar.get_ticks()):
        pass  # scales differ by design; gamma ticks checked below
    gamma_ticks = list(gamma_cbar.get_ticks())
    if len(gamma_ticks) != 3 or not np.isclose(gamma_ticks[0], -gamma_vmax) or not np.isclose(
        gamma_ticks[2], gamma_vmax
    ):
        raise ValueError("gamma color-bar ticks do not match shared gamma vmax")

    # Compact ECG-prone legend centered under F3–F4.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.25,
            markersize=6.5,
            label="ECG-prone channel in F3",
        )
    ]
    legend_ax = fig.add_axes([0.53, 0.195, 0.40, 0.035])
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        fontsize=FS_TICK - 2,
        frameon=False,
        handlelength=1.2,
    )

    fig.text(
        0.5,
        0.975,
        PANEL_F_FIGURE_TITLE,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        PANEL_F_SUBTITLE,
        ha="center",
        va="top",
        fontsize=FS_TICK,
        color="#555555",
    )
    fig.text(
        0.5,
        0.885,
        PANEL_F_SECOND_LINE_TEMPLATE.format(
            n_alpha=result.metadata.get("n_alpha_common_channels", len(retained)),
            n_gamma=result.metadata.get("n_gamma_paired_observations"),
        ),
        ha="center",
        va="top",
        fontsize=FS_TICK - 1,
        color="#333333",
    )
    fig.text(
        0.5,
        0.845,
        PANEL_F_INTERPRETATION_LINE,
        ha="center",
        va="top",
        fontsize=FS_TICK - 1,
        color="#333333",
    )
    fig.text(
        0.5,
        0.045,
        PANEL_F_FOOTNOTE,
        ha="center",
        va="center",
        fontsize=FS_TICK - 3,
        color="#555555",
    )

    estimates_after = [
        (_as_str(r.get("panel_map")), _as_str(r.get("channel")), _as_float(r.get("estimate")))
        for r in result.summary_rows
    ]
    if estimates_before != estimates_after:
        raise ValueError("plotting mutated Panel F summary estimates")

    # Cross-check plotted values against exported summary CSV.
    summary_csv = export_paths["summary"]
    exported_by_map: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_csv(summary_csv):
        exported_by_map[_as_str(row.get("panel_map"))][_as_str(row.get("channel"))] = _as_float(
            row.get("estimate")
        )
    for map_id, channels, values in (
        (MAP_REST_ALPHA, ch_f1, v_f1),
        (MAP_TASK_ATTENUATION, ch_f2, v_f2),
        (MAP_GAMMA_BEFORE, ch_f3, v_f3),
        (MAP_GAMMA_AFTER, ch_f4, v_f4),
    ):
        for ch, val in zip(channels, values, strict=True):
            exp = exported_by_map[map_id].get(ch, float("nan"))
            if not (math.isfinite(val) and math.isfinite(exp) and val == exp):
                raise ValueError(f"plotted value != exported source for {map_id}/{ch}")

    plot_qc = {
        "figure_size": list(FIGURE_SIZE),
        "alpha_vmax": alpha_vmax,
        "gamma_vmax": gamma_vmax,
        "n_channels": {
            "F1": len(ch_f1),
            "F2": len(ch_f2),
            "F3": len(ch_f3),
            "F4": len(ch_f4),
        },
        "ecg_prone_channels": ecg_prone,
        "n_ecg_prone_markers_f3": n_ecg_markers,
        "excluded_on_f4_relative_to_f3": excluded_on_f4,
        "significance_markers_added": False,
        "f4_zeros_for_excluded_channels": False,
        "extrapolate": TOPOMAP_EXTRAPOLATE,
        "contours": TOPOMAP_CONTOURS,
        "image_interp": TOPOMAP_INTERP,
        "cmap": TOPOMAP_CMAP,
        "cbar_labels": [PANEL_F_CBAR_ALPHA, PANEL_F_CBAR_GAMMA],
        "dpi": FIGURE_DPI,
    }

    trio = save_figure_trio(
        fig,
        output_dir,
        PANEL_F_STEM,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    out: dict[str, Path] = {
        "pdf": trio[0],
        "svg": trio[1],
        "png": trio[2],
        "caption": caption_path,
        **export_paths,
    }
    if include_internal_qc:
        qc_dir = output_dir / "internal_qc" / "panel_f_topography_gamma"
        qc_dir.mkdir(parents=True, exist_ok=True)
        note_path = qc_dir / "panel_f_qc_notes.txt"
        note_path.write_text(
            "\n".join(
                [
                    "Panel F internal QC",
                    json.dumps(result.metadata, indent=2, default=str),
                    "",
                    "Plot QC:",
                    json.dumps(plot_qc, indent=2, default=str),
                    "",
                    "Gamma montage sensitivity:",
                    *[f"{r.get('metric')}: {r}" for r in result.gamma_comparison_rows],
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        plot_qc_path = qc_dir / "panel_f_plot_qc.json"
        plot_qc_path.write_text(
            json.dumps(plot_qc, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        out["qc_notes"] = note_path
        out["plot_qc"] = plot_qc_path
    # save_figure_trio already closes the figure; guard against reopen accumulation.
    plt.close(fig)
    return out



__all__ = [
    "PANEL_F_STEM",
    "PANEL_F_FIGURE_TITLE",
    "PANEL_F_TITLE",
    "MAP_ORDER",
    "PanelFResult",
    "compute_panel_f_topography",
    "render_panel_f_figure",
    "verify_panel_f_integrity",
    "panel_f_caption",
    "write_panel_f_exports",
    "build_or_load_channel_zlpi_table",
    "load_ecg_prone_channels",
    "mark_ecg_prone_channels",
]
