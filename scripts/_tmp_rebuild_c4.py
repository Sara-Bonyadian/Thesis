#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from ppg_eeg.confirmatory.nulls import run_confirmatory_nulls


def main() -> None:
    root = Path("derivatives/confirmatory_temporal_coupling/primary")
    for dataset_id in ("ds003690", "ds003838", "ds006848"):
        out_root = root / dataset_id
        result = run_confirmatory_nulls(
            aligned_dir=out_root / "C1c",
            output_dir=out_root / "C4",
            n_surrogates=500,
            durations=(240,),
            n_jobs=-1,
            progress=True,
        )
        print(
            dataset_id,
            "subject_rows",
            len(result.subject_rows),
            "surrogate_rows",
            len(result.surrogate_rows),
            flush=True,
        )


if __name__ == "__main__":
    main()
