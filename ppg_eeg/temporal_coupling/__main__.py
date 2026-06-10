from __future__ import annotations

import argparse
import sys

from .config import VALID_STAGES, load_config, validate_stage
from .run import run_temporal_coupling


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Within-subject temporal EEG–cardiac coupling pipeline.",
    )
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    ap.add_argument(
        "--stage",
        type=str,
        required=True,
        help=f"Pipeline stage to run. Allowed: {', '.join(sorted(VALID_STAGES, key=lambda s: (len(s), s)))}",
    )
    args = ap.parse_args(argv)

    try:
        stage = validate_stage(args.stage)
        cfg = load_config(args.config)
        run_temporal_coupling(cfg, stage=stage)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[temporal_coupling] error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
