from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import PipelineConfig


@dataclass(frozen=True)
class CorrelationResult:
    raw: pd.DataFrame
    fdr: pd.DataFrame
    trend_agreement: pd.DataFrame
    trend_summary: dict[str, Any]


def _corr_pair(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if method == "spearman":
        rho, p_value = stats.spearmanr(x, y, nan_policy="omit")
        return float(rho), float(p_value)
    if method == "pearson":
        rho, p_value = stats.pearsonr(x, y)
        return float(rho), float(p_value)
    raise ValueError(f"Unsupported correlation method: {method!r}")


def compute_pairwise_correlations(
    merged_features: pd.DataFrame,
    *,
    eeg_features: Sequence[str],
    ppg_features: Sequence[str],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [m.casefold() for m in cfg.correlation.methods]

    if merged_features.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "method",
                "eeg_feature",
                "ppg_feature",
                "n",
                "correlation",
                "p_value",
            ]
        )

    for dataset_id, group in merged_features.groupby("dataset_id", dropna=False):
        for method in methods:
            for eeg_feature in eeg_features:
                for ppg_feature in ppg_features:
                    if eeg_feature not in group.columns or ppg_feature not in group.columns:
                        continue

                    x = pd.to_numeric(group[eeg_feature], errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(group[ppg_feature], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(x) & np.isfinite(y)
                    n = int(np.sum(valid))

                    corr = float("nan")
                    p_value = float("nan")
                    if n >= cfg.correlation.min_n:
                        try:
                            corr, p_value = _corr_pair(x[valid], y[valid], method)
                        except Exception:
                            corr = float("nan")
                            p_value = float("nan")

                    rows.append(
                        {
                            "dataset_id": str(dataset_id),
                            "method": method,
                            "eeg_feature": eeg_feature,
                            "ppg_feature": ppg_feature,
                            "n": n,
                            "correlation": corr,
                            "p_value": p_value,
                        }
                    )

    return pd.DataFrame(rows)


def apply_fdr(raw_correlations: pd.DataFrame, *, cfg: PipelineConfig) -> pd.DataFrame:
    if raw_correlations.empty:
        out = raw_correlations.copy()
        out["q_value"] = pd.Series(dtype=float)
        out["reject_fdr"] = pd.Series(dtype=bool)
        return out

    out = raw_correlations.copy()
    out["q_value"] = np.nan
    out["reject_fdr"] = False

    for (dataset_id, method), idx in out.groupby(["dataset_id", "method"]).groups.items():
        group = out.loc[idx]
        valid_mask = np.isfinite(group["p_value"].to_numpy(dtype=float))
        if not np.any(valid_mask):
            continue

        pvals = group.loc[valid_mask, "p_value"].to_numpy(dtype=float)
        reject, qvals, _, _ = multipletests(
            pvals,
            alpha=cfg.correlation.alpha,
            method=cfg.correlation.fdr_method,
        )
        valid_indices = group.index[valid_mask]
        out.loc[valid_indices, "q_value"] = qvals
        out.loc[valid_indices, "reject_fdr"] = reject

    return out


def compute_trend_agreement(
    corr_fdr: pd.DataFrame,
    *,
    cfg: PipelineConfig,
    primary_method: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method = (primary_method or cfg.correlation.primary_method).casefold()
    focus = corr_fdr[corr_fdr["method"].str.casefold() == method].copy()
    if focus.empty:
        empty = pd.DataFrame(
            columns=[
                "eeg_feature",
                "ppg_feature",
                "n_datasets_present",
                "n_datasets_significant",
                "sign_consistent",
                "trend_direction",
                "mean_correlation",
                "mean_abs_correlation",
                "median_q_value",
                "effect_rank",
            ]
        )
        summary = {
            "primary_method": method,
            "n_pairs": 0,
            "n_replicated_pairs": 0,
            "replication_rule": "sign-consistent and significant in >=2 datasets",
            "top_pairs": [],
        }
        return empty, summary

    records: list[dict[str, Any]] = []
    for (eeg_feature, ppg_feature), grp in focus.groupby(["eeg_feature", "ppg_feature"]):
        rho = pd.to_numeric(grp["correlation"], errors="coerce")
        q_values = pd.to_numeric(grp["q_value"], errors="coerce")
        finite_rho = rho[np.isfinite(rho)]
        finite_q = q_values[np.isfinite(q_values)]

        n_present = int(finite_rho.shape[0])
        n_sig = int(np.sum(finite_q < cfg.correlation.alpha))

        signs = np.sign(finite_rho.to_numpy(dtype=float))
        nonzero_signs = signs[signs != 0]
        sign_consistent = bool(nonzero_signs.size > 0 and np.all(nonzero_signs == nonzero_signs[0]))

        if nonzero_signs.size == 0:
            trend_direction = "undetermined"
        elif np.all(nonzero_signs > 0):
            trend_direction = "positive"
        elif np.all(nonzero_signs < 0):
            trend_direction = "negative"
        else:
            trend_direction = "mixed"

        records.append(
            {
                "eeg_feature": eeg_feature,
                "ppg_feature": ppg_feature,
                "n_datasets_present": n_present,
                "n_datasets_significant": n_sig,
                "sign_consistent": sign_consistent,
                "trend_direction": trend_direction,
                "mean_correlation": float(np.nanmean(finite_rho)) if n_present > 0 else np.nan,
                "mean_abs_correlation": float(np.nanmean(np.abs(finite_rho))) if n_present > 0 else np.nan,
                "median_q_value": float(np.nanmedian(finite_q)) if finite_q.shape[0] > 0 else np.nan,
            }
        )

    trend_df = pd.DataFrame(records)
    trend_df["effect_rank"] = trend_df["mean_abs_correlation"].rank(method="dense", ascending=False)
    trend_df = trend_df.sort_values(
        by=["n_datasets_significant", "sign_consistent", "mean_abs_correlation"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    replicated_mask = (trend_df["n_datasets_significant"] >= 2) & trend_df["sign_consistent"]
    summary = {
        "primary_method": method,
        "n_pairs": int(trend_df.shape[0]),
        "n_replicated_pairs": int(np.sum(replicated_mask)),
        "replication_rule": "sign-consistent and significant in >=2 datasets",
        "top_pairs": trend_df.head(10).to_dict(orient="records"),
    }
    return trend_df, summary


def run_correlation_analysis(
    merged_features: pd.DataFrame,
    *,
    eeg_features: Sequence[str],
    ppg_features: Sequence[str],
    cfg: PipelineConfig,
) -> CorrelationResult:
    raw = compute_pairwise_correlations(
        merged_features,
        eeg_features=eeg_features,
        ppg_features=ppg_features,
        cfg=cfg,
    )
    fdr = apply_fdr(raw, cfg=cfg)
    trend_df, trend_summary = compute_trend_agreement(fdr, cfg=cfg)
    return CorrelationResult(
        raw=raw,
        fdr=fdr,
        trend_agreement=trend_df,
        trend_summary=trend_summary,
    )
