#!/usr/bin/env python3
"""Generate a print-ready HTML cross-dataset temporal coupling report from JSON."""

from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = Path(__file__).resolve().parent / "russell_cross_dataset_report_data.json"
OUT_HTML = Path(__file__).resolve().parent / "cross_dataset_temporal_coupling_report.html"

DETAIL_HEADERS = [
    "Cardiac Duration (s)",
    "Median HR",
    "Minimum HR",
    "Maximum HR",
    "Median Mean RR",
    "Median RMSSD",
    "Median SDNN",
    "EEG Duration (s)",
    "Theta Min",
    "Theta Max",
    "Theta Median",
    "Alpha Min",
    "Alpha Max",
    "Alpha Median",
    "Beta Min",
    "Beta Max",
    "Beta Median",
    "Peak Coupling Strengths",
    "Lag Distributions",
    "Significant Lag-Direction Findings",
    "Group Cross-Correlation Mean +/- SEM Grid",
]

SUMMARY_HEADERS = [
    "Dataset",
    "Task/Condition",
    "Usable Subjects",
    "Cardiac channel/source",
    "Peak Coupling Strengths",
    "Lag Distributions",
    "Significant Lag-Direction Findings",
    "Stage 3 Interpretation",
    "Stage 4 Interpretation",
    "Mean +/- SEM Grid",
]

INTERPRETATION = {
    "bottom_line": (
        "Partial / inconclusive. Descriptive EEG-cardiac coupling is detectable in all "
        "{n} completed Stage 3 runs, but a single reproducible timing signature has not "
        "yet emerged across studies."
    ),
    "common_patterns": [
        (
            "All {n} dataset/task partitions with Stage 3 outputs show non-zero median peak |r| "
            "across HR/HRV x theta/alpha/beta pairs. Coupling magnitude spans roughly 0.04-0.48. "
            "The most reproducible pattern is task-related attenuation: active tasks (memory, "
            "verbal WM, tetris) are consistently weaker than rest within the same cohort."
        ),
        (
            "ds003838 rest (dedicated ECG, n=65) remains the best internal benchmark. "
            "ds004582 (ff, n=70) and ds004587 (ig, n=97) contribute dedicated ECG cohorts "
            "with moderate coupling (|r| about 0.11-0.26)."
        ),
    ],
    "key_differences": (
        "ds003816 shows the largest peaks (|r| 0.37-0.48) but on short usable cardiac windows "
        "(~60-85 s). ds006848 verbalwm and ds003838 memory remain among the weakest. Sensor type "
        "still matters: dedicated ECG (ds003838, ds004582, ds004587, ds006848) vs embedded "
        "PPG/photosensor (HIIT, mindfulness)."
    ),
    "lag_direction": (
        "Lag direction does not replicate robustly. Most partitions show mixed subject-level "
        "peak-lag signs. 2 of {n} entries have FDR-significant peak-lag direction (including "
        "ds004587 hr__beta and HIIT post_tetris hr__alpha). HRV-alpha pairs still flip between "
        "datasets."
    ),
    "stage4": (
        "Stage 4 is usable in all completed partitions including ds004582 and ds004587. "
        "HR-to-EEG triggered curves are the most reliable output; EEG-to-HR remains exploratory."
    ),
    "evidence_table": [
        ("Coupling presence", "Supported (descriptive)", "Non-zero peaks in all {n} Stage 3 runs"),
        (
            "Coupling magnitude",
            "Partially supported",
            "Moderate in ECG rest cohorts; weak under active tasks",
        ),
        (
            "Lag direction",
            "Not supported",
            "Mixed directions; 2/{n} FDR-significant lag findings",
        ),
        (
            "Cross-study signature",
            "Not yet established",
            "Amplitude partially replicates; timing does not",
        ),
        (
            "Causal/robust coupling claim",
            "Not supported",
            "Permutation-controlled group evidence required before strong claims",
        ),
    ],
    "conclusion": (
        "Conservative conclusion: a unified physiological coupling signature has not emerged. "
        "What is defensible is descriptive co-fluctuation between EEG band envelopes and cardiac "
        "measures, systematic weakening during active tasks, and heterogeneous lag timing. All "
        "eight requested datasets now have Stage 3/4 group outputs (ds004582/ff uses a flat "
        "group/ folder; ds004587/ig uses group/ig/)."
    ),
}

GRID_PATH_FIXES = {
    (
        "ds004587",
        "ig",
    ): REPO_ROOT
    / "derivatives/run_ds004587_temporal_coupling/ds004587/group/ig/group_cross_correlation_mean_sem_grid.png",
}


def _fmt_duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_scalar(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_envelope(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def _resolve_grid_path(entry: dict) -> Path | None:
    key = (entry["dataset"], entry["task"])
    if key in GRID_PATH_FIXES:
        candidate = GRID_PATH_FIXES[key]
        if candidate.is_file():
            return candidate
    raw = entry.get("mean_sem_grid_png")
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    return None


def _rel_href(path: Path) -> str:
    return Path(os.path.relpath(path, OUT_HTML.parent)).as_posix()


def _detail_row(entry: dict, grid_path: Path | None) -> list[str]:
    grid_cell = "present" if grid_path and grid_path.is_file() else "missing"
    return [
        _fmt_duration(entry.get("cardiac_duration_s")),
        _fmt_scalar(entry.get("median_hr")),
        _fmt_scalar(entry.get("min_hr")),
        _fmt_scalar(entry.get("max_hr")),
        _fmt_scalar(entry.get("median_mean_rr")),
        _fmt_scalar(entry.get("median_rmssd")),
        _fmt_scalar(entry.get("median_sdnn")),
        _fmt_duration(entry.get("eeg_duration_s")),
        _fmt_envelope(entry.get("theta_min")),
        _fmt_envelope(entry.get("theta_max")),
        _fmt_envelope(entry.get("theta_median")),
        _fmt_envelope(entry.get("alpha_min")),
        _fmt_envelope(entry.get("alpha_max")),
        _fmt_envelope(entry.get("alpha_median")),
        _fmt_envelope(entry.get("beta_min")),
        _fmt_envelope(entry.get("beta_max")),
        _fmt_envelope(entry.get("beta_median")),
        str(entry.get("peak_coupling_strengths") or "n/a"),
        str(entry.get("lag_distributions") or "n/a"),
        str(entry.get("significant_lag_direction_findings") or "none"),
        grid_cell,
    ]


def _summary_row(entry: dict, grid_path: Path | None) -> list[str]:
    grid_cell = "present" if grid_path and grid_path.is_file() else "missing"
    return [
        entry["dataset"],
        entry["task"],
        str(entry.get("n_usable_subjects", "n/a")),
        str(entry.get("cardiac_channel_source") or "n/a"),
        str(entry.get("peak_coupling_strengths") or "n/a"),
        str(entry.get("lag_distributions") or "n/a"),
        str(entry.get("significant_lag_direction_findings") or "none"),
        str(entry.get("stage3_interpretation") or "n/a"),
        str(entry.get("stage4_interpretation") or "n/a"),
        grid_cell,
    ]


def _detail_kv_rows(entry: dict, grid_path: Path | None) -> list[tuple[str, str]]:
    values = _detail_row(entry, grid_path)
    return list(zip(DETAIL_HEADERS, values, strict=True))


def _summary_kv_rows(entry: dict, grid_path: Path | None) -> list[tuple[str, str]]:
    values = _summary_row(entry, grid_path)
    return list(zip(SUMMARY_HEADERS, values, strict=True))


def _kv_table(rows: list[tuple[str, str]], *, table_class: str = "kv") -> str:
    body_rows = []
    for label, value in rows:
        body_rows.append(
            f'<tr><th scope="row">{html.escape(label)}</th><td>{html.escape(value)}</td></tr>'
        )
    return (
        f'<div class="table-wrap"><table class="{html.escape(table_class)} kv">'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )

def _table(headers: list[str], rows: list[list[str]], *, table_class: str = "") -> str:
    cls = f' class="{html.escape(table_class)}"' if table_class else ""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table{cls}><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def _paragraphs(texts: list[str]) -> str:
    return "".join(f"<p>{html.escape(text)}</p>" for text in texts)


def build_html(entries: list[dict]) -> str:
    n_entries = len(entries)
    datasets: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        datasets[entry["dataset"]].append(entry)

    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    dataset_count = len(datasets)

    detail_sections: list[str] = []
    summary_sections: list[str] = []

    for dataset in sorted(datasets):
        detail_sections.append(f"<h3>{html.escape(dataset)}</h3>")
        for entry in datasets[dataset]:
            grid_path = _resolve_grid_path(entry)
            title = f"{entry['dataset']} / {entry['task']}"
            n = entry.get("n_usable_subjects", "n/a")
            cardiac = entry.get("cardiac_channel_source", "n/a")
            block_parts = [
                f"<div class=\"task-block\">",
                f"<h4>{html.escape(title)} &middot; n={html.escape(str(n))} &middot; "
                f"{html.escape(str(cardiac))}</h4>",
                _kv_table(_detail_kv_rows(entry, grid_path), table_class="detail"),
            ]
            if grid_path and grid_path.is_file():
                href = _rel_href(grid_path)
                block_parts.append(
                    f'<figure class="grid-figure">'
                    f'<img src="{html.escape(href)}" alt="{html.escape(title)} mean +/- SEM grid">'
                    f"<figcaption>{html.escape(title)} &mdash; group cross-correlation mean +/- SEM</figcaption>"
                    f"</figure>"
                )
            block_parts.append("</div>")
            detail_sections.append("".join(block_parts))
            detail_sections.append('<hr class="section-rule">')

            summary_sections.append(f"<h4>{html.escape(title)}</h4>")
            summary_sections.append(_kv_table(_summary_kv_rows(entry, grid_path), table_class="summary"))
            summary_sections.append('<hr class="section-rule">')

    interp = INTERPRETATION
    evidence_rows = [
        [dim, verdict, summary.format(n=n_entries)] for dim, verdict, summary in interp["evidence_table"]
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cross-Dataset Temporal Coupling Report</title>
  <style>
    :root {{
      --text: #1a1a1a;
      --muted: #555;
      --border: #d9d9d9;
      --accent: #8b3a2a;
      --callout-bg: #fff8e8;
      --callout-border: #e6c77a;
    }}
    @page {{
      size: landscape;
      margin: 0.35in 0.3in;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      color: var(--text);
      line-height: 1.45;
      margin: 0 auto;
      padding: 1.25rem 1.5rem 2rem;
      max-width: none;
      width: 100%;
    }}
    h1, h2, h3, h4 {{ line-height: 1.2; margin: 1.25rem 0 0.65rem; }}
    h1 {{ font-size: 1.75rem; margin-top: 0; }}
    h2 {{ font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }}
    h3 {{ font-size: 1.1rem; color: var(--accent); }}
    h4 {{ font-size: 0.95rem; margin-top: 0.75rem; margin-bottom: 0.35rem; }}
    p, .meta {{ color: var(--text); }}
    .meta {{ font-size: 0.92rem; color: var(--muted); }}
    .callout {{
      background: var(--callout-bg);
      border: 1px solid var(--callout-border);
      border-left: 4px solid var(--accent);
      padding: 0.9rem 1rem;
      margin: 1rem 0 1.25rem;
    }}
    .callout strong {{ display: block; margin-bottom: 0.35rem; }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 0.35rem 0 0.5rem;
      -webkit-overflow-scrolling: touch;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin: 0;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.45rem 0.55rem;
      vertical-align: top;
      text-align: left;
      line-height: 1.35;
    }}
    th {{
      background: #f5f5f5;
      font-weight: 600;
      font-size: 0.88rem;
    }}
    table.kv {{
      max-width: 100%;
    }}
    table.kv th[scope="row"],
    table.kv td {{
      padding: 0.18rem 0.4rem;
      line-height: 1.2;
      vertical-align: middle;
    }}
    table.kv th[scope="row"] {{
      width: 26%;
      min-width: 9rem;
      background: #f5f5f5;
      font-weight: 600;
      white-space: normal;
    }}
    table.kv td {{
      white-space: normal;
      word-break: break-word;
    }}
    table.detail.kv,
    table.summary.kv {{
      font-size: 0.82rem;
    }}
    table.striped tbody tr:nth-child(even) {{ background: #fafafa; }}
    .task-block {{
      margin: 0.55rem 0 0.5rem;
    }}
    .grid-figure {{
      margin: 0.35rem 0 0;
      page-break-inside: avoid;
    }}
    .grid-figure img {{
      width: 100%;
      max-width: 100%;
      max-height: 5in;
      height: auto;
      object-fit: contain;
      object-position: left top;
      border: 1px solid var(--border);
      display: block;
    }}
    figcaption {{
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 0.2rem;
      line-height: 1.2;
    }}
    .section-rule {{
      border: 0;
      border-top: 1px solid var(--border);
      margin: 0.65rem 0;
    }}
    @media print {{
      body {{
        padding: 0;
        width: 100%;
      }}
      h2, h3, h4 {{ page-break-after: avoid; }}
      .task-block {{
        page-break-inside: avoid;
        break-inside: avoid;
      }}
      .grid-figure img {{
        max-height: 4.25in;
      }}
      .table-wrap {{
        overflow: visible;
      }}
      table.detail.kv,
      table.summary.kv {{
        font-size: 8pt;
      }}
      table.kv th[scope="row"],
      table.kv td {{
        padding: 0.04in 0.08in;
        line-height: 1.15;
      }}
      table.kv th[scope="row"] {{
        width: 28%;
      }}
      th, td {{
        padding: 0.12in 0.1in;
      }}
      table, .grid-figure {{ page-break-inside: avoid; }}
      a {{ color: inherit; text-decoration: none; }}
    }}
  </style>
</head>
<body>
  <h1>Cross-Dataset Temporal Coupling Report</h1>
  <p class="meta">Source: local derivatives runs in <code>ppg-eeg/derivatives</code> &middot; Cardiac and EEG metrics are medians across usable subjects per task.</p>
  <p class="meta">Last updated: {html.escape(updated)} &middot; {n_entries} dataset/task entries across {dataset_count} datasets ({n_entries} with Stage 3 outputs).</p>

  <h2>Detailed Tables by Dataset and Task</h2>
  <p>Each metric is shown on its own row (metric name in the first column, value in the second).</p>
  {''.join(detail_sections)}

  <h2>Cross-Dataset Summary: Stage 3 and Stage 4 Interpretations</h2>
  <p>Interpretation fields per dataset/task, one row per field.</p>
  {''.join(summary_sections)}

  <h2>Overall Interpretation: Is a Consistent Physiological Coupling Signature Emerging?</h2>
  <div class="callout">
    <strong>Bottom line</strong>
    {html.escape(interp["bottom_line"].format(n=n_entries))}
  </div>

  <h3>Common patterns</h3>
  {_paragraphs([text.format(n=n_entries) for text in interp["common_patterns"]])}

  <h3>Key differences across studies</h3>
  <p>{html.escape(interp["key_differences"])}</p>

  <h3>Reproducibility of lag direction</h3>
  <p>{html.escape(interp["lag_direction"].format(n=n_entries))}</p>

  <h3>Stage 4 event-triggered evidence</h3>
  <p>{html.escape(interp["stage4"])}</p>

  <h3>Does the evidence support a robust physiological relationship?</h3>
  {_table(["Evidence dimension", "Verdict", "Summary"], evidence_rows, table_class="striped")}

  <p>{html.escape(interp["conclusion"])}</p>
</body>
</html>
"""


def main() -> None:
    entries = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"Expected JSON array in {DATA_JSON}")

    html_text = build_html(entries)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
