"""Stage-14.8 unit and pre-final integration checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from stage14_8_final_gate import (
    CONFIGURATION_SOURCES,
    FROZEN_REFERENCE_PATH,
    PROFILE_PATH,
    RAW_SOURCES,
    audit_raw_repetitions,
    compare_canonical_row,
    load_tagged_raw_rows,
    profile_regression,
)


ROOT = Path(__file__).resolve().parent


class Stage148FinalGateTests(unittest.TestCase):
    def test_all_declared_sources_exist(self) -> None:
        for _stage, path in RAW_SOURCES:
            self.assertTrue(path.is_file(), path)
        for path in CONFIGURATION_SOURCES:
            self.assertTrue(path.is_file(), path)

    def test_raw_repetition_audit_passes(self) -> None:
        audit = audit_raw_repetitions(load_tagged_raw_rows())
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(all(audit["checks"].values()))
        self.assertGreater(audit["completed_count"], 0)
        self.assertGreater(audit["noncompleted_count"], 0)

    def test_profiled_and_original_identity_match(self) -> None:
        frozen = json.loads(FROZEN_REFERENCE_PATH.read_text(encoding="utf-8"))
        profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        audit = profile_regression(frozen, profile)
        self.assertTrue(audit["passed"], audit)

    def test_existing_canonical_harness_rows_match_frozen_reference(self) -> None:
        frozen = json.loads(FROZEN_REFERENCE_PATH.read_text(encoding="utf-8"))
        rows = json.loads(RAW_SOURCES[0][1].read_text(encoding="utf-8"))
        completed = [row for row in rows if row["status"] == "completed"]
        self.assertEqual(len(completed), 3)
        self.assertTrue(all(compare_canonical_row(row, frozen)["passed"] for row in completed))

    def test_notebook_has_final_gate_before_stop_marker(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertLess(ids.index("stage14-8-heading"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("full_scaling_sweeps_rerun", source)
        self.assertIn("Beyond Stage 14 — not started", source)


if __name__ == "__main__":
    unittest.main()
