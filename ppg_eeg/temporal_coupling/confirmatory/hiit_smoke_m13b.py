"""HIIT M13b real-smoke stage driver (operator-run only).

Use one ``--stage`` at a time. This module will refuse to continue when pairing,
PPG/photosensor modality, or D240/D180 eligibility cannot be verified. Missing
data are blockers, never participant exclusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from ...datasets import build_observations
from .config import load_dataset_config, load_master_config
from .correlation import run_confirmatory_correlations
from .endpoints import run_confirmatory_endpoints
from .group_tables import run_confirmatory_group_tables
from .harmonize import ALIGNED_FEATURES_TEMPLATE
from .inference import run_confirmatory_inference_from_dir
from .nulls import SMOKE_N_SURROGATES, run_confirmatory_nulls
from .artifact_controls import run_confirmatory_artifact_controls
from .peak_model import run_confirmatory_peak_fits
from .protocol_audit import (
    PAIRED_SUBJECT_SETS_FILENAME,
    ELIGIBILITY_BY_DURATION_FILENAME,
    run_protocol_audit,
)
from .report import run_confirmatory_reporting
from .production import master_config_path

REPO_DEFAULT = Path(__file__).resolve().parents[3]
CONFIG_DIR_DEFAULT = REPO_DEFAULT / "zero-lag-reanalysis-repo"
HIIT_SMOKE_DIR = CONFIG_DIR_DEFAULT / "smoke" / "hiit"
CONFIRMATORY_SMOKE = HIIT_SMOKE_DIR / "confirmatory.yaml"
VERIFICATION_JSON = HIIT_SMOKE_DIR / "verification.json"
EXPLORATORY_SMOKE = HIIT_SMOKE_DIR / "beats.yaml"
MASTER_CONFIG = master_config_path(CONFIG_DIR_DEFAULT)

REQUIRED_CONTRASTS = (
    "ph_pre_rest__tetris",
    "ph_post_rest__tetris",
    "ps_pre_rest__tetris",
    "ps_post_rest__tetris",
)

STAGES = (
    "preflight",
    "m1",
    "verify_pairing",
    "stage0_hint",
    "stage1b_hint",
    "m2_hint",
    "m3_hint",
    "m4_hint",
    "m5",
    "m6",
    "m7",
    "m8",
    "m9",
    "m10",
    "m11",
    "m12",
)


class SmokeStop(RuntimeError):
    """Hard stop: assumption or required smoke input failed verification."""


def _load_verification() -> dict:
    return json.loads(VERIFICATION_JSON.read_text(encoding="utf-8"))


def _paths() -> tuple[object, object]:
    master = load_master_config(MASTER_CONFIG)
    dataset = load_dataset_config(CONFIRMATORY_SMOKE, master=master)
    return master, dataset


def cmd_preflight() -> int:
    master, dataset = _paths()
    verification = _load_verification()
    print(f"master_ok={master.source_path}")
    print(f"dataset_ok={dataset.source_path}")
    print(f"raw_root={dataset.paths.raw_root}")
    print(f"raw_exists={dataset.paths.raw_root.exists()}")
    print(f"output_root={dataset.output_root}")
    print(f"primary_cardiac_channel={verification['primary_cardiac_channel']}")
    print(f"primary_cardiac_modality={verification['primary_cardiac_modality']}")
    if not dataset.paths.raw_root.exists():
        raise SmokeStop(
            "raw_root missing — production blocker, not participant exclusion."
        )
    if verification["primary_cardiac_channel"].casefold() != "photosensor":
        raise SmokeStop("M13b requires photosensor as primary PPG channel.")
    if verification["primary_cardiac_modality"].casefold() != "ppg":
        raise SmokeStop("M13b requires PPG as primary cardiac modality.")
    print("preflight_pass=True")
    print(
        "NEXT: run exploratory Stage 0, then Stage 1b with "
        f"{EXPLORATORY_SMOKE.name} before confirmatory M2+."
    )
    return 0


def cmd_m1() -> int:
    master, dataset = _paths()
    out = dataset.output_root / "m1_audit"
    paths = run_protocol_audit(master, [dataset], output_dir=out)
    print("wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


def cmd_verify_pairing() -> int:
    _, dataset = _paths()
    audit_dir = dataset.output_root / "m1_audit"
    paired_path = audit_dir / PAIRED_SUBJECT_SETS_FILENAME
    eligibility_path = audit_dir / ELIGIBILITY_BY_DURATION_FILENAME
    if not paired_path.is_file():
        raise SmokeStop(f"Missing {paired_path}; run --stage m1 first.")
    payload = json.loads(paired_path.read_text(encoding="utf-8"))
    hiit = payload["datasets"]["hiit"]
    contrasts = hiit["contrasts"]
    for contrast_id in REQUIRED_CONTRASTS:
        block = contrasts.get(contrast_id)
        if block is None:
            raise SmokeStop(f"Required contrast missing: {contrast_id}")
        n_paired = int(block["n_paired"])
        print(f"{contrast_id}: n_paired={n_paired}")
        if n_paired <= 0:
            raise SmokeStop(
                f"No paired keys for {contrast_id}. Stop — pairing cannot be "
                "verified (do not treat as exclusion)."
            )

    if not eligibility_path.is_file():
        raise SmokeStop(f"Missing {eligibility_path}; run --stage m1 first.")
    with eligibility_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for duration in (240, 180):
        subset = [r for r in rows if str(r.get("duration_s")) == str(duration)]
        statuses = {r.get("status", "") for r in subset}
        print(f"D{duration} eligibility statuses={sorted(statuses)}")
        if not subset:
            raise SmokeStop(f"No eligibility rows for D{duration}.")
        # Unknown timing while raw present → not_computable is expected at M1c
        # structural audit; operator must re-check after Stage 1b durations.
        if statuses == {"ineligible"}:
            raise SmokeStop(
                f"All D{duration} rows ineligible — stop before features."
            )

    # Observation inventory: expected up to 8 per complete participant.
    observations = build_observations(
        dataset.dataset_id,
        dataset.paths.raw_root,
        subjects=dataset.selection.subjects or None,
        tasks=dataset.selection.tasks or None,
        conditions=dataset.selection.conditions or None,
        sessions=dataset.selection.sessions or None,
        hiit_partition_mode="protocol_task",
    )
    print(f"discovered_observations={len(observations)}")
    by_subject: dict[str, set[str]] = {}
    for obs in observations:
        by_subject.setdefault(obs.subject_id, set()).add(obs.condition_label)
    for subject_id, conditions in sorted(by_subject.items()):
        print(f"  {subject_id}: n_conditions={len(conditions)} {sorted(conditions)}")
    if not observations:
        raise SmokeStop("Zero observations discovered for smoke selection.")
    print("verify_pairing_pass=True")
    print(
        "NOTE: After Stage 1b, confirm cardiac_channel_inventory / cardiac_qc "
        "channel_used == photosensor and signal_type == ppg for every observation."
    )
    return 0


def _require_aligned(root: Path) -> Path:
    for duration in (240, 180):
        path = root / ALIGNED_FEATURES_TEMPLATE.format(duration_s=duration)
        if not path.is_file():
            raise SmokeStop(
                f"Missing aligned table {path.name}. Complete M2–M4 (HR + "
                "multitaper + harmonize) before this stage."
            )
    return root


def cmd_m5(work: Path) -> int:
    aligned = _require_aligned(work / "m4_aligned")
    out = work / "m5_curves"
    run_confirmatory_correlations(aligned, out, durations=(240, 180, 120, 60))
    print(f"wrote curves under {out}")
    return 0


def cmd_m6(work: Path) -> int:
    curves = work / "m5_curves"
    out = work / "m6_endpoints"
    if not curves.is_dir():
        raise SmokeStop("Missing m5_curves; run --stage m5 first.")
    run_confirmatory_endpoints(curves, out)
    print(f"wrote endpoints under {out}")
    return 0


def cmd_m7(work: Path) -> int:
    curves = work / "m5_curves"
    out = work / "m7_peaks"
    run_confirmatory_peak_fits(curves, out)
    print(f"wrote peaks under {out}")
    return 0


def cmd_m8(work: Path) -> int:
    out = work / "m8_group"
    run_confirmatory_group_tables(
        work / "m6_endpoints", out, peaks_dir=work / "m7_peaks"
    )
    print(f"wrote group tables under {out}")
    return 0


def cmd_m9(work: Path) -> int:
    aligned = _require_aligned(work / "m4_aligned")
    out = work / "m9_nulls"
    run_confirmatory_nulls(
        aligned, out, n_surrogates=SMOKE_N_SURROGATES, durations=(240,)
    )
    print(f"wrote nulls under {out}")
    return 0


def cmd_m10(work: Path) -> int:
    out = work / "m10_inference"
    run_confirmatory_inference_from_dir(work / "m8_group", out)
    print(f"wrote inference under {out}")
    return 0


def cmd_m11(work: Path) -> int:
    out = work / "m11_artifacts"
    run_confirmatory_artifact_controls(work / "m8_group", out)
    print(f"wrote artifact controls under {out}")
    return 0


def cmd_m12(work: Path) -> int:
    import shutil

    publish = work / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    for name in (
        "m1_audit",
        "m5_curves",
        "m6_endpoints",
        "m7_peaks",
        "m8_group",
        "m9_nulls",
        "m10_inference",
        "m11_artifacts",
    ):
        src = work / name
        if not src.is_dir():
            continue
        for path in src.iterdir():
            if path.is_file():
                shutil.copy2(path, publish / path.name)
    out = work / "m12_report"
    paths = run_confirmatory_reporting(
        publish,
        out,
        config_paths=[MASTER_CONFIG, CONFIRMATORY_SMOKE],
        repo_root=REPO_DEFAULT,
        seeds={"n_surrogates": SMOKE_N_SURROGATES},
    )
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HIIT M13b smoke stages (run one at a time)."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=STAGES,
        help="Single stage to execute.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    _, dataset = _paths()
    work = dataset.output_root
    try:
        if args.stage == "preflight":
            return cmd_preflight()
        if args.stage == "m1":
            return cmd_m1()
        if args.stage == "verify_pairing":
            return cmd_verify_pairing()
        if args.stage == "stage0_hint":
            print(
                "Run exploratory Stage 0 (file/overlap audit) with:\n"
                f"  .venv/bin/python -m ppg_eeg.temporal_coupling "
                f"--config {EXPLORATORY_SMOKE} --stage 0\n"
                "STOP if Stage 0 reports missing files, no_cardiac_channels, "
                "or insufficient overlap for paired PRE/POST states."
            )
            return 0
        if args.stage == "stage1b_hint":
            print(
                "Run exploratory Stage 1b (PPG photosensor peaks) with:\n"
                f"  .venv/bin/python -m ppg_eeg.temporal_coupling "
                f"--config {EXPLORATORY_SMOKE} --stage 1b\n"
                "STOP unless every observation QC row has "
                "channel_used=photosensor and signal_type=ppg."
            )
            return 0
        if args.stage == "m2_hint":
            print(
                "M2 (instantaneous HR): for each observation directory under\n"
                f"  derivatives/smoke_hiit_m13b_temporal_coupling/hiit/\n"
                "call reconstruct_instant_hr_file(detected_peaks.csv, <obs_out>).\n"
                "Expected: features_instant_hr.csv, instant_hr_qc.csv\n"
                "STOP if status!=ok or clean_beat_span_s < 180 for D180 / < 240 for D240 targets."
            )
            return 0
        if args.stage == "m3_hint":
            print(
                "M3 (multitaper): for each BrainVision .vhdr observation, call\n"
                "  extract_multitaper_file(eeg_path, 'brainvision', <obs_out>)\n"
                "Expected: features_multitaper_power.csv, multitaper_qc.csv\n"
                "STOP if no clean EEG channels remain after rejection."
            )
            return 0
        if args.stage == "m4_hint":
            print(
                "M4 (harmonize): align instant HR + multitaper per observation, then\n"
                f"aggregate to {work}/m4_aligned/features_confirmatory_aligned_D{{240,180,120,60}}.csv\n"
                "STOP if alignment_qc eligible=False for required PRE/POST pairs at D240/D180."
            )
            return 0
        if args.stage == "m5":
            return cmd_m5(work)
        if args.stage == "m6":
            return cmd_m6(work)
        if args.stage == "m7":
            return cmd_m7(work)
        if args.stage == "m8":
            return cmd_m8(work)
        if args.stage == "m9":
            return cmd_m9(work)
        if args.stage == "m10":
            return cmd_m10(work)
        if args.stage == "m11":
            return cmd_m11(work)
        if args.stage == "m12":
            return cmd_m12(work)
    except SmokeStop as exc:
        print(f"[hiit_smoke_m13b] STOP: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[hiit_smoke_m13b] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
