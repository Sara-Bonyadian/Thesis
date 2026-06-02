from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import (
    run_stage1,
    run_stage2_from_base_csvs,
    run_two_stage_pipeline,
    write_stage1_artifacts,
    write_stage2_artifacts,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="EEG–PPG correlation pipeline (no ICA).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "Run only one stage: 1 = base extraction to CSV, "
            "2 = derive features/correlations from existing Stage 1 CSVs. "
            "Omit to run both stages in one command."
        ),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded config for dataset_ids={cfg.dataset_ids!r}")
    print(f"raw_root={cfg.paths.raw_root}")
    print(f"out_root={cfg.paths.out_root}")

    if args.stage == 1:
        stage1 = run_stage1(cfg)
        write_stage1_artifacts(cfg, stage1)
        for dataset_id, data in stage1.per_dataset.items():
            print(
                f"[{dataset_id}] stage=1 observations={len(data.observations)} "
                f"eeg_base_rows={len(data.eeg_base_features)} "
                f"ppg_ibi_rows={len(data.ppg_ibi_features)}"
            )
        return

    if args.stage == 2:
        artifacts = run_stage2_from_base_csvs(cfg)
        write_stage2_artifacts(cfg, artifacts)
        for dataset_id, data in artifacts.per_dataset.items():
            print(
                f"[{dataset_id}] stage=2 merged_rows={len(data.merged_features)} "
                f"corr_tests={len(data.correlations_raw)}"
            )
        print(
            f"cross_dataset_pairs={len(artifacts.trend_agreement)} "
            f"summary={out_root / 'cross_dataset' / 'trend_agreement_summary.json'}"
        )
        return

    artifacts = run_two_stage_pipeline(cfg)
    for dataset_id, data in artifacts.per_dataset.items():
        print(
            f"[{dataset_id}] observations={len(data.observations)} "
            f"merged_rows={len(data.merged_features)} "
            f"corr_tests={len(data.correlations_raw)}"
        )
    print(
        f"cross_dataset_pairs={len(artifacts.trend_agreement)} "
        f"summary={out_root / 'cross_dataset' / 'trend_agreement_summary.json'}"
    )


if __name__ == "__main__":
    main()

