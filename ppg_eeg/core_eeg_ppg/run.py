from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, load_config
from .correlation import display_correlation_heatmaps
from .preflight import print_preflight_report, run_preflight
from .pipeline import (
    CORRELATIONS_HEATMAP_FILE,
    PipelineArtifacts,
    run_stage1,
    run_stage2_from_base_csvs,
    write_stage1_artifacts,
    write_stage2_artifacts,
)


def _should_show_heatmap(cfg: PipelineConfig, *, no_show_heatmap: bool) -> bool:
    if no_show_heatmap:
        return False
    return cfg.output.show_heatmap


def _report_stage2_outputs(
    cfg: PipelineConfig,
    artifacts: PipelineArtifacts,
    *,
    out_root: Path,
    stage_label: str | None = None,
) -> None:
    prefix = f"stage={stage_label} " if stage_label else ""
    for dataset_id, data in artifacts.per_dataset.items():
        dataset_dir = out_root / dataset_id
        print(
            f"[{dataset_id}] {prefix}merged_rows={len(data.merged_features)} "
            f"corr_tests={len(data.correlations_raw)}"
        )
        print(f"[{dataset_id}] correlations_raw={dataset_dir / 'correlations_raw.csv'}")
        print(f"[{dataset_id}] correlations_fdr={dataset_dir / 'correlations_fdr.csv'}")
        if cfg.output.save_heatmap:
            print(f"[{dataset_id}] correlations_heatmap={dataset_dir / CORRELATIONS_HEATMAP_FILE}")
    print(
        f"cross_dataset_pairs={len(artifacts.trend_agreement)} "
        f"summary={out_root / 'cross_dataset' / 'trend_agreement_summary.json'}"
    )


def _write_and_maybe_show_stage2(
    cfg: PipelineConfig,
    artifacts: PipelineArtifacts,
    *,
    show_heatmap: bool,
) -> None:
    heatmap_figures = write_stage2_artifacts(cfg, artifacts)
    if show_heatmap:
        display_correlation_heatmaps(heatmap_figures)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "EEG–PPG correlation pipeline (no ICA). "
            "Configs: core-eeg-ppg/ (e.g. core-eeg-ppg/example.yaml)."
        ),
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config under core-eeg-ppg/ (run from repo root).",
    )
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
    ap.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate raw inputs and output folders for each configured subject, "
            "write per-subject preflight manifests, then exit."
        ),
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks before Stage 1 or full pipeline runs.",
    )
    ap.add_argument(
        "--no-show-heatmap",
        action="store_true",
        help="Save the correlation heatmap PNG but do not open an interactive plot window.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg.paths.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    show_heatmap = _should_show_heatmap(cfg, no_show_heatmap=args.no_show_heatmap)

    print(f"Loaded config for dataset_ids={cfg.dataset_ids!r}")
    print(f"raw_root={cfg.paths.raw_root}")
    print(f"out_root={cfg.paths.out_root}")

    should_preflight = args.preflight or (not args.skip_preflight and args.stage != 2)
    if should_preflight:
        report = run_preflight(cfg)
        print_preflight_report(report)
        if args.preflight:
            raise SystemExit(0 if report.ready else 1)
        if not report.ready:
            raise SystemExit(
                "Preflight failed: fix missing raw files or output paths, "
                "or re-run with --skip-preflight if you intentionally want to proceed."
            )

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
        _write_and_maybe_show_stage2(cfg, artifacts, show_heatmap=show_heatmap)
        _report_stage2_outputs(cfg, artifacts, out_root=out_root, stage_label="2")
        return

    stage1 = run_stage1(cfg)
    write_stage1_artifacts(cfg, stage1)
    artifacts = run_stage2_from_base_csvs(cfg)
    _write_and_maybe_show_stage2(cfg, artifacts, show_heatmap=show_heatmap)
    for dataset_id, data in stage1.per_dataset.items():
        print(
            f"[{dataset_id}] stage=1 observations={len(data.observations)} "
            f"eeg_base_rows={len(data.eeg_base_features)} "
            f"ppg_ibi_rows={len(data.ppg_ibi_features)}"
        )
    _report_stage2_outputs(cfg, artifacts, out_root=out_root)


if __name__ == "__main__":
    main()
