"""Phase 2 acceptance gates for metadata-driven display and StructuredReason."""

from __future__ import annotations

import ast
import csv
import io
import json
import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from ppg_eeg.confirmatory.figures import (
    FIGURE2_PANEL_A_SENSITIVITY_NOTE,
    FIGURE2_PANEL_C_NOTE,
    FIGURE2_PANEL_E_NOTE,
    FIGURE2_TITLE,
    FIGURE3_TITLE,
    format_role_display,
    sensitivity_display_title,
)
from ppg_eeg.confirmatory.nulls import StaleC4CacheError, validate_c4_cache_for_run
from ppg_eeg.confirmatory.reason_codes import STRUCTURED_NC_FIELDS

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = ROOT / "ppg_eeg" / "confirmatory"
DS004587 = Path("derivatives/confirmatory_temporal_coupling/primary/ds004587")

HIIT_DISPLAY_TOKENS = ("Rest", "Tetris", "PH/PS", "PRE/POST")
SHARED_SCIENTIFIC = (
    CONFIRMATORY / "group_tables.py",
    CONFIRMATORY / "null_delta_inference.py",
    CONFIRMATORY / "artifact_controls.py",
    CONFIRMATORY / "capability_resolution.py",
    CONFIRMATORY / "protocol_audit.py",
    CONFIRMATORY / "endpoints.py",
    CONFIRMATORY / "correlation.py",
    CONFIRMATORY / "harmonize.py",
    CONFIRMATORY / "nulls.py",
)
SHARED_DISPLAY = (
    CONFIRMATORY / "figures.py",
    CONFIRMATORY / "figure1_panels.py",
    CONFIRMATORY / "figure2_panels.py",
    CONFIRMATORY / "panel_d_cardiac_controls.py",
    CONFIRMATORY / "panel_e_nuisance_modality.py",
    CONFIRMATORY / "panel_f_topography_gamma.py",
)
DATASET_ID_COMPARE = re.compile(
    r"""dataset_id\s*(==|!=)\s*['\"](hiit|ds\d+|mindfulness)['\"]""",
    re.IGNORECASE,
)


class TestPhase2CaptionMetadata(unittest.TestCase):
    def test_figure2_notes_are_metadata_general(self) -> None:
        for text in (
            FIGURE2_TITLE,
            FIGURE2_PANEL_A_SENSITIVITY_NOTE,
            FIGURE2_PANEL_C_NOTE,
            FIGURE2_PANEL_E_NOTE,
        ):
            for token in HIIT_DISPLAY_TOKENS:
                self.assertNotIn(token, text, msg=f"{token!r} in {text[:80]!r}")
        self.assertIn("low-demand", FIGURE2_PANEL_A_SENSITIVITY_NOTE.casefold())
        self.assertIn("high-demand", FIGURE2_PANEL_C_NOTE.casefold())

    def test_format_role_display_uses_yaml_raw_label(self) -> None:
        self.assertEqual(
            format_role_display("state_low", raw_label="rest"),
            "Low-demand state (rest)",
        )
        self.assertEqual(
            format_role_display("state_high", raw_label="inhibitory-control task"),
            "High-demand state (inhibitory-control task)",
        )
        title = sensitivity_display_title(
            "matched lag curves", dataset_id="ds004587"
        )
        self.assertTrue(title.startswith("Sensitivity display"))
        self.assertNotIn("Rest", title)
        self.assertNotIn("Tetris", title)

    def test_non_hiit_caption_constants_omit_hiit_protocol_tokens(self) -> None:
        # Shared constants must not force Rest/Tetris/PH/PS unless YAML supplies them.
        blob = "\n".join(
            [
                FIGURE2_TITLE,
                FIGURE3_TITLE,
                FIGURE2_PANEL_A_SENSITIVITY_NOTE,
                FIGURE2_PANEL_C_NOTE,
                FIGURE2_PANEL_E_NOTE,
            ]
        )
        for token in ("Rest–Tetris", "Rest-Tetris", "PH/PS", "PRE/POST", "HIIT PH"):
            self.assertNotIn(token, blob)


class TestPhase2NoDatasetBranches(unittest.TestCase):
    def test_no_dataset_name_branches_in_shared_scientific_code(self) -> None:
        hits: list[str] = []
        for path in SHARED_SCIENTIFIC:
            text = path.read_text(encoding="utf-8")
            for match in DATASET_ID_COMPARE.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                hits.append(f"{path.name}:{line_no}:{match.group(0)}")
        self.assertEqual(hits, [], msg="\n".join(hits))

    def test_no_unexplained_hiit_display_helpers_in_shared_captions(self) -> None:
        """Caption/title constants and shared helpers must not hard-code Rest/Tetris."""
        caption_files = (
            CONFIRMATORY / "figures.py",
            CONFIRMATORY / "panel_f_topography_gamma.py",
        )
        banned = re.compile(r"\b(Rest–Tetris|Rest-Tetris|PH/PS|PRE/POST)\b")
        failures: list[str] = []
        for path in caption_files:
            text = path.read_text(encoding="utf-8")
            for match in banned.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1].strip()
                if "deprecated" in line.casefold() or "alias" in line.casefold():
                    continue
                if line.lstrip().startswith("#"):
                    continue
                failures.append(f"{path.name}:{line_no}:{match.group(0)}")
        self.assertEqual(failures, [], msg="\n".join(failures))


class TestPhase2StructuredReasonCoverage(unittest.TestCase):
    def test_c1_to_c6_modules_import_structured_helpers(self) -> None:
        required = {
            "C1a": CONFIRMATORY / "multitaper_power.py",
            "C1b": CONFIRMATORY / "peak_detection.py",
            "C1c": CONFIRMATORY / "harmonize.py",
            "C2": CONFIRMATORY / "correlation.py",
            "C3": CONFIRMATORY / "endpoints.py",
            "C4": CONFIRMATORY / "nulls.py",
            "C5": CONFIRMATORY / "group_tables.py",
            "C6": CONFIRMATORY / "artifact_controls.py",
        }
        for stage, path in required.items():
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                "attach_structured_reason",
                text,
                msg=f"{stage} ({path.name}) missing attach_structured_reason",
            )
            self.assertIn("STRUCTURED_NC_FIELDS", text, msg=stage)

    def test_nc_row_envelope_has_required_fields(self) -> None:
        from ppg_eeg.confirmatory.reason_codes import attach_structured_reason

        row = attach_structured_reason(
            {
                "dataset_id": "ds004587",
                "participant_id": "ffe001",
                "session_id": "ses-01",
                "observation_id": "ds004587-ffe001-ses-01-rest",
                "eligible": False,
                "exclusion_reason": "insufficient_lag_support",
            },
            stage="C3",
            specification_id="D180_absolute_log10",
        )
        for field in STRUCTURED_NC_FIELDS:
            self.assertIn(field, row)
            if field in {
                "dataset_id",
                "stage",
                "participant_id",
                "session_id",
                "observation_id",
                "reason_code",
                "status",
            }:
                self.assertTrue(str(row[field]).strip(), msg=field)


class TestPhase2StaleC4Rejection(unittest.TestCase):
    def test_downstream_rejects_20_surrogate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            c4 = Path(tmp)
            (c4 / "C4_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "n_surrogates": 20,
                        "n_units": 3,
                        "null_types": ["circular_shift"],
                        "configuration_hash_sha256": "hash",
                        "root_seed": 20260713,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(StaleC4CacheError):
                validate_c4_cache_for_run(
                    c4,
                    expected_n_surrogates=500,
                    configuration_hash_sha256="hash",
                    root_seed=20260713,
                )


class TestPhase2FontWarnings(unittest.TestCase):
    def test_panel_f_titles_use_supported_fontweight(self) -> None:
        text = (CONFIRMATORY / "panel_f_topography_gamma.py").read_text(encoding="utf-8")
        self.assertNotIn('fontweight="semibold"', text)
        self.assertNotIn("fontweight='semibold'", text)
        self.assertIn('fontweight="bold"', text)

    def test_standard_figure_constants_avoid_semibold(self) -> None:
        for path in SHARED_DISPLAY:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("semibold", text.casefold(), msg=path.name)


@unittest.skipUnless(
    (DS004587 / "C4" / "C4_COMPLETE.json").is_file(),
    "production ds004587 C4_COMPLETE.json absent",
)
class TestPhase2Ds004587ProductionC4(unittest.TestCase):
    def test_c4_has_500_surrogates(self) -> None:
        complete = json.loads(
            (DS004587 / "C4" / "C4_COMPLETE.json").read_text(encoding="utf-8")
        )
        self.assertEqual(int(complete.get("n_surrogates", -1)), 500)
        meta = json.loads((DS004587 / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(int(meta.get("n_surrogates", -1)), 500)

    def test_surrogate_table_has_expected_draws(self) -> None:
        path = DS004587 / "C4" / "null_surrogate_values.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertTrue(rows)
        by_key: dict[tuple[str, str, str], set[str]] = {}
        for row in rows:
            key = (
                str(row.get("null_type") or ""),
                str(row.get("specification_id") or row.get("spec_id") or ""),
                str(row.get("observation_id") or ""),
            )
            sid = str(row.get("surrogate_id") or row.get("surrogate_index") or "")
            by_key.setdefault(key, set()).add(sid)
        # Spot-check: every populated unit has exactly 500 unique surrogate ids.
        for key, ids in list(by_key.items())[:20]:
            self.assertEqual(len(ids), 500, msg=f"{key} -> {len(ids)}")


@unittest.skipUnless(
    (DS004587 / "C5" / "aggregation_manifest.csv").is_file(),
    "ds004587 C5 aggregation_manifest absent",
)
class TestPhase2AggregationFamilies(unittest.TestCase):
    def test_manifest_keeps_endpoint_families(self) -> None:
        with (DS004587 / "C5" / "aggregation_manifest.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return
        families = {r.get("endpoint_family") for r in rows}
        self.assertTrue({"standard_zlpi", "mwpi", "swpi"} & families)


if __name__ == "__main__":
    unittest.main()
