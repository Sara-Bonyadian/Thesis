from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_two_stage_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="EEG–PPG correlation pipeline (no ICA).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    artifacts = run_two_stage_pipeline(cfg)

    print(f"Loaded config for dataset_ids={cfg.dataset_ids!r}")
    print(f"raw_root={cfg.paths.raw_root}")
    print(f"out_root={cfg.paths.out_root}")
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

