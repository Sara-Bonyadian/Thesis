"""Load BIDS continuous physiological recordings (beh/*_physio.tsv.gz)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def physio_json_path(physio_path: Path) -> Path:
    if not physio_path.name.endswith("_physio.tsv.gz"):
        raise ValueError(f"Expected *_physio.tsv.gz, got {physio_path.name!r}")
    return physio_path.parent / physio_path.name.replace("_physio.tsv.gz", "_physio.json")


def read_physio_sidecar(physio_path: Path) -> dict[str, object]:
    sidecar = physio_json_path(physio_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid physio sidecar payload in {sidecar}")
    return payload


def physio_channel_type(channel_name: str) -> str:
    low = channel_name.casefold()
    if "ecg" in low:
        return "ecg"
    if any(token in low for token in ("oxi", "ppg", "pleth", "pulse", "photo")):
        return "bio"
    if "resp" in low:
        return "resp"
    return "misc"


def count_physio_samples(physio_path: Path) -> int:
    with gzip.open(physio_path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def read_physio_channel_info(
    physio_path: Path,
) -> tuple[list[str], list[str], float, float] | None:
    if not physio_path.is_file():
        return None
    sidecar_path = physio_json_path(physio_path)
    if not sidecar_path.is_file():
        return None
    try:
        payload = read_physio_sidecar(physio_path)
        columns = payload.get("Columns")
        sfreq = payload.get("SamplingFrequency")
        if not isinstance(columns, list) or not columns or sfreq is None:
            return None
        sfreq_hz = float(sfreq)
        if sfreq_hz <= 0:
            return None
        ch_names = [str(name) for name in columns]
        ch_types = [physio_channel_type(name) for name in ch_names]
        n_samples = count_physio_samples(physio_path)
        if n_samples <= 0:
            return None
        duration_s = float(n_samples) / sfreq_hz
        return ch_names, ch_types, sfreq_hz, duration_s
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def read_physio_raw(physio_path: Path) -> mne.io.BaseRaw:
    payload = read_physio_sidecar(physio_path)
    columns = payload.get("Columns")
    sfreq = payload.get("SamplingFrequency")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Missing Columns in {physio_json_path(physio_path)}")
    if sfreq is None:
        raise ValueError(f"Missing SamplingFrequency in {physio_json_path(physio_path)}")

    sfreq_hz = float(sfreq)
    if sfreq_hz <= 0:
        raise ValueError(f"Invalid SamplingFrequency in {physio_json_path(physio_path)}: {sfreq!r}")

    with gzip.open(physio_path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle, sep="\t", header=None, names=[str(c) for c in columns])

    data = frame.to_numpy(dtype=float).T
    ch_types = [physio_channel_type(name) for name in columns]
    info = mne.create_info(ch_names=[str(c) for c in columns], sfreq=sfreq_hz, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)
