from __future__ import annotations

import re
from pathlib import Path

from .config import PipelineConfig


def safe_subject_dir_name(subject_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", subject_id.strip())
    return cleaned or "unknown_subject"


def dataset_output_dir(cfg: PipelineConfig, dataset_id: str) -> Path:
    return Path(cfg.paths.out_root) / dataset_id


def subject_output_dir(cfg: PipelineConfig, dataset_id: str, subject_id: str) -> Path:
    return dataset_output_dir(cfg, dataset_id) / safe_subject_dir_name(subject_id)
