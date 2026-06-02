from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import PipelineConfig

AUTO_NORMALITY_ALPHA = 0.05

CORRELATIONS_FDR_PRIMARY_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "x_shapiro_p",
    "y_shapiro_p",
    "selected_method",
    "eeg_feature",
    "ppg_feature",
    "correlation",
    "p_value",
    "sig_p_005",
    "q_value",
    "sig_q_005",
    "sig_q_010",
    "sig_q_015",
)
CORRELATIONS_FDR_METADATA_COLUMNS: tuple[str, ...] = (
    "method",
    "n",
    "reject_fdr",
)


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


def _shapiro_pvalue(values: np.ndarray) -> float:
    try:
        return float(stats.shapiro(values).pvalue)
    except Exception:
        return float("nan")


def _corr_pair_with_auto_method(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> tuple[float, float, str, float, float]:
    if method != "auto":
        corr, p_value = _corr_pair(x, y, method)
        return corr, p_value, method, float("nan"), float("nan")

    x_shapiro_p = _shapiro_pvalue(x)
    y_shapiro_p = _shapiro_pvalue(y)

    if (x_shapiro_p > AUTO_NORMALITY_ALPHA) and (y_shapiro_p > AUTO_NORMALITY_ALPHA):
        resolved_method = "pearson"
    else:
        resolved_method = "spearman"

    corr, p_value = _corr_pair(x, y, resolved_method)
    return corr, p_value, resolved_method, x_shapiro_p, y_shapiro_p


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
                "selected_method",
                "eeg_feature",
                "ppg_feature",
                "n",
                "correlation",
                "p_value",
                "x_shapiro_p",
                "y_shapiro_p",
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
                    selected_method = method
                    x_shapiro_p = float("nan")
                    y_shapiro_p = float("nan")
                    if n >= cfg.correlation.min_n:
                        try:
                            corr, p_value, selected_method, x_shapiro_p, y_shapiro_p = _corr_pair_with_auto_method(
                                x[valid],
                                y[valid],
                                method,
                            )
                        except Exception:
                            corr = float("nan")
                            p_value = float("nan")
                            selected_method = method
                            x_shapiro_p = float("nan")
                            y_shapiro_p = float("nan")
                    elif method == "auto":
                        selected_method = "too_few_data"

                    rows.append(
                        {
                            "dataset_id": str(dataset_id),
                            "method": method,
                            "selected_method": selected_method,
                            "eeg_feature": eeg_feature,
                            "ppg_feature": ppg_feature,
                            "n": n,
                            "correlation": corr,
                            "p_value": p_value,
                            "x_shapiro_p": x_shapiro_p,
                            "y_shapiro_p": y_shapiro_p,
                        }
                    )

    return pd.DataFrame(rows)


def _add_significance_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p_values = pd.to_numeric(out["p_value"], errors="coerce")
    q_values = pd.to_numeric(out["q_value"], errors="coerce")
    out["sig_p_005"] = p_values < 0.05
    out["sig_q_005"] = q_values < 0.05
    out["sig_q_010"] = q_values < 0.10
    out["sig_q_015"] = q_values < 0.15
    return out


def format_correlations_fdr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return FDR correlation table with significance flags and stable column order."""
    out = _add_significance_flags(df)
    ordered = [
        *CORRELATIONS_FDR_PRIMARY_COLUMNS,
        *[col for col in CORRELATIONS_FDR_METADATA_COLUMNS if col in out.columns],
    ]
    extras = [col for col in out.columns if col not in ordered]
    return out[[*ordered, *extras]]


def apply_fdr(raw_correlations: pd.DataFrame, *, cfg: PipelineConfig) -> pd.DataFrame:
    if raw_correlations.empty:
        out = raw_correlations.copy()
        out["q_value"] = pd.Series(dtype=float)
        out["reject_fdr"] = pd.Series(dtype=bool)
        out["sig_p_005"] = pd.Series(dtype=bool)
        out["sig_q_005"] = pd.Series(dtype=bool)
        out["sig_q_010"] = pd.Series(dtype=bool)
        out["sig_q_015"] = pd.Series(dtype=bool)
        return format_correlations_fdr_table(out)

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

    return format_correlations_fdr_table(out)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def significance_symbol(*, q_value: Any, p_value: Any) -> str:
    """Map q/p values to significance symbols for correlation heatmaps."""
    q = _coerce_float(q_value)
    p = _coerce_float(p_value)

    if np.isfinite(q):
        if q < 0.05:
            return "***"
        if q < 0.10:
            return "**"
        if q < 0.15:
            return "*"
    if np.isfinite(p) and p < 0.05:
        return "+"
    return ""


def build_correlation_heatmap_tables(
    correlations_fdr: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    method: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return correlation and significance tables indexed as EEG x PPG."""
    required = {"eeg_feature", "ppg_feature", "correlation", "p_value"}
    missing = required.difference(correlations_fdr.columns)
    if missing:
        needed = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns for heatmap: {needed}")

    focus = correlations_fdr.copy()

    if dataset_id is not None:
        if "dataset_id" not in focus.columns:
            raise ValueError("dataset_id filter was provided but no 'dataset_id' column exists.")
        focus = focus[focus["dataset_id"].astype(str) == str(dataset_id)]

    if method is not None:
        if "method" not in focus.columns:
            raise ValueError("method filter was provided but no 'method' column exists.")
        focus = focus[focus["method"].astype(str).str.casefold() == method.casefold()]

    if focus.empty:
        raise ValueError("No rows available to build a heatmap after filtering.")

    corr_matrix = focus.pivot_table(
        index="eeg_feature",
        columns="ppg_feature",
        values="correlation",
        aggfunc="first",
    )
    p_matrix = focus.pivot_table(
        index="eeg_feature",
        columns="ppg_feature",
        values="p_value",
        aggfunc="first",
    )

    if "q_value" in focus.columns:
        q_matrix = focus.pivot_table(
            index="eeg_feature",
            columns="ppg_feature",
            values="q_value",
            aggfunc="first",
        )
    else:
        q_matrix = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)

    p_matrix = p_matrix.reindex(index=corr_matrix.index, columns=corr_matrix.columns)
    q_matrix = q_matrix.reindex(index=corr_matrix.index, columns=corr_matrix.columns)

    symbol_matrix = pd.DataFrame("", index=corr_matrix.index, columns=corr_matrix.columns)
    for row_idx, eeg_feature in enumerate(corr_matrix.index):
        for col_idx, ppg_feature in enumerate(corr_matrix.columns):
            symbol_matrix.iat[row_idx, col_idx] = significance_symbol(
                q_value=q_matrix.at[eeg_feature, ppg_feature],
                p_value=p_matrix.at[eeg_feature, ppg_feature],
            )

    return corr_matrix, symbol_matrix


def plot_correlation_heatmap(
    correlations_fdr: pd.DataFrame,
    *,
    dataset_id: str | None = None,
    method: str | None = None,
    show_values: bool = True,
    decimals: int = 2,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (9.0, 6.0),
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """
    Plot EEG x PPG correlation heatmap with significance markers.

    Marker legend:
    - ***: FDR q < 0.05
    - ** : FDR q < 0.10
    - *  : FDR q < 0.15
    - +  : p-value < 0.05 (only when none of the FDR thresholds is met)
    """
    corr_matrix, symbol_matrix = build_correlation_heatmap_tables(
        correlations_fdr,
        dataset_id=dataset_id,
        method=method,
    )

    import matplotlib.pyplot as plt

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True
    else:
        fig = ax.figure

    image = ax.imshow(corr_matrix.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")

    ax.set_xticks(np.arange(corr_matrix.shape[1]))
    ax.set_yticks(np.arange(corr_matrix.shape[0]))
    ax.set_xticklabels(corr_matrix.columns.tolist(), rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index.tolist())
    ax.set_xlabel("PPG feature", labelpad=8)
    ax.set_ylabel("EEG feature")
    ax.tick_params(axis="x", pad=2)

    if title is None:
        parts = ["Correlation heatmap"]
        if dataset_id is not None:
            parts.append(f"dataset={dataset_id}")
        if method is not None:
            parts.append(f"method={method}")
        title = " | ".join(parts)
    ax.set_title(title)

    values = corr_matrix.to_numpy(dtype=float)
    for row_idx in range(corr_matrix.shape[0]):
        for col_idx in range(corr_matrix.shape[1]):
            corr_value = values[row_idx, col_idx]
            marker = str(symbol_matrix.iat[row_idx, col_idx])
            if show_values and np.isfinite(corr_value):
                label = f"{corr_value:.{decimals}f}{marker}"
            else:
                label = marker
            if not label:
                continue
            text_color = "white" if np.isfinite(corr_value) and abs(corr_value) >= 0.5 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(np.arange(-0.5, corr_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, corr_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    legend_text = "*** q<0.05   ** q<0.10   * q<0.15   + p<0.05"
    if created_figure:
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
        fig.text(
            0.5,
            0.04,
            legend_text,
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        fig.text(
            0.5,
            0.02,
            legend_text,
            ha="center",
            va="bottom",
            fontsize=9,
            transform=fig.transFigure,
        )

    return fig, ax


def write_correlation_heatmap(
    correlations_fdr: pd.DataFrame,
    path: Path,
    *,
    dataset_id: str | None = None,
    method: str | None = None,
    dpi: int = 150,
    **plot_kwargs: Any,
) -> Any | None:
    """Render and save an EEG x PPG correlation heatmap PNG."""
    if correlations_fdr.empty:
        return None

    try:
        fig, _ax = plot_correlation_heatmap(
            correlations_fdr,
            dataset_id=dataset_id,
            method=method,
            **plot_kwargs,
        )
    except ValueError:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return fig


def display_correlation_heatmaps(figures: dict[str, Any]) -> None:
    """Show saved correlation heatmap figures (interactive backends only)."""
    if not figures:
        return

    import matplotlib.pyplot as plt

    plt.show()


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
