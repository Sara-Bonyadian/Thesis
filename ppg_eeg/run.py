from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="EEG–PPG correlation pipeline (no ICA).")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Intentionally a scaffold: dataset-specific loaders will be added next.
    # This repo does not import or depend on any ICA code.
    print(f"Loaded config for dataset_id={cfg.dataset_id!r}")
    print(f"raw_root={cfg.paths.raw_root}")
    print(f"out_root={cfg.paths.out_root}")
    print("Next: implement dataset loader + feature extraction loops.")


if __name__ == "__main__":
    main()

