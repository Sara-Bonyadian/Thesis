from __future__ import annotations

import argparse
import sys

from .config import VALID_STAGES, load_config, validate_stage
from .recommend import print_recommendations
from .run import run_temporal_coupling


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Exploratory temporal EEG–cardiac coupling (Stages 0–4). "
            "Configs: exploratory-temporal-coupling/datasets|smoke/*.yaml. "
            "For confirmatory zero-lag use: python -m ppg_eeg.confirmatory"
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to YAML under exploratory-temporal-coupling/ "
            "(e.g. exploratory-temporal-coupling/smoke/ds003838.yaml)."
        ),
    )
    ap.add_argument(
        "--stage",
        type=str,
        required=False,
        help=f"Pipeline stage to run. Allowed: {', '.join(sorted(VALID_STAGES, key=lambda s: (len(s), s)))}",
    )
    ap.add_argument(
        "--recommend-config-values",
        action="store_true",
        help="Print data-driven recommendations for min_overlap_s and cardiac windows.",
    )
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config)
        if args.recommend_config_values:
            print_recommendations(cfg)
            return 0
        if not args.stage:
            ap.error("--stage is required unless --recommend-config-values is set.")
        stage = validate_stage(args.stage)
        run_temporal_coupling(cfg, stage=stage)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[temporal_coupling] error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
