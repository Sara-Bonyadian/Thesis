#!/usr/bin/env python3
"""Build canonical merged primary Figure 1 manuscript replication artifacts."""

from __future__ import annotations

import argparse
import json

from ppg_eeg.confirmatory.manuscript_meta import (
    build_manuscript_primary_merged_meta_tree,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge primary datasets (ds003690/ds003838/ds006848), generate merged "
            "low-demand alpha replication C6 tables, and render merged Figure 1."
        )
    )
    parser.add_argument(
        "--confirmatory-root",
        type=str,
        default="derivatives/confirmatory_temporal_coupling",
        help="Confirmatory derivatives root containing primary/<dataset>/C5..C7 trees.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_manuscript_primary_merged_meta_tree(args.confirmatory_root)
    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
