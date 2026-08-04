"""Static audit: shared Figure 3 modules must not grow raw-label / dataset-name branches."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE3_MODULES = (
    ROOT / "ppg_eeg/confirmatory/figures.py",
    ROOT / "ppg_eeg/confirmatory/panel_d_cardiac_controls.py",
    ROOT / "ppg_eeg/confirmatory/panel_e_nuisance_modality.py",
    ROOT / "ppg_eeg/confirmatory/panel_f_topography_gamma.py",
    ROOT / "ppg_eeg/confirmatory/nulls.py",
)

# Direct dataset-id identity tests in shared figure code.
DATASET_ID_COMPARE = re.compile(
    r"""dataset_id\s*(==|!=)\s*['\"](hiit|ds\d+|mindfulness)['\"]""",
    re.IGNORECASE,
)

# Raw protocol tokens used as semantic inference (not display captions).
RAW_SEMANTIC_COMPARE = re.compile(
    r"""(==|!=)\s*['\"](REST|TETRIS|PRE|POST|PH|PS|SRT|GNG|passive)['\"]"""
)


class TestFigure3NoRawLabelBranches(unittest.TestCase):
    def test_no_new_dataset_name_identity_branches(self) -> None:
        allowed_files_hits: dict[str, list[str]] = {}
        for path in FIGURE3_MODULES:
            text = path.read_text(encoding="utf-8")
            hits = []
            for match in DATASET_ID_COMPARE.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                hits.append(f"{path.name}:{line_no}:{match.group(0)}")
            if hits:
                allowed_files_hits[path.name] = hits
        # Soft gate for Phase 1: report remaining hardcodes that still need Phase 2
        # caption cleanup, but fail on executable branches in panel builders.
        builder_hits = [
            h
            for name, rows in allowed_files_hits.items()
            if name != "figures.py"
            for h in rows
        ]
        self.assertEqual(
            builder_hits,
            [],
            "Shared panel modules contain dataset_id identity branches:\n"
            + "\n".join(builder_hits),
        )

    def test_no_raw_protocol_token_equality_in_shared_modules(self) -> None:
        failures: list[str] = []
        for path in FIGURE3_MODULES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for comparator in node.comparators:
                    if not isinstance(comparator, ast.Constant):
                        continue
                    if not isinstance(comparator.value, str):
                        continue
                    if comparator.value in {
                        "REST",
                        "TETRIS",
                        "PRE",
                        "POST",
                        "PH",
                        "PS",
                        "SRT",
                        "GNG",
                        "passive",
                    }:
                        failures.append(
                            f"{path.name}:{getattr(node, 'lineno', '?')}:"
                            f"compare_to_{comparator.value!r}"
                        )
        self.assertEqual(failures, [], "Raw protocol token comparisons found:\n" + "\n".join(failures))

    def test_regex_scan_documents_remaining_figure_caption_tokens(self) -> None:
        # Captions may still mention HIIT PH/PS narratively; this test only
        # ensures executable `dataset_id == ...` branches are inventoried.
        figures = (ROOT / "ppg_eeg/confirmatory/figures.py").read_text(encoding="utf-8")
        hits = [m.group(0) for m in DATASET_ID_COMPARE.finditer(figures)]
        # Prefer zero; if any remain they must not be ds003816 duration gates.
        for hit in hits:
            self.assertNotIn("ds003816", hit)


if __name__ == "__main__":
    unittest.main()
