"""Regression checks for the Phase-6 paper-ready closeout contract."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

import project_paths
from phase6_paper_closeout import (
    CLOSEOUT_SCHEMA,
    OUTPUT_DIR,
    RESOLUTIONS_M,
    _canonical_json,
    _row,
    _source_path,
)


class Phase6PaperCloseoutTests(unittest.TestCase):
    def setUp(self) -> None:
        missing = [path for dx in RESOLUTIONS_M if not (path := _source_path(dx)).is_file()]
        if missing:
            self.skipTest("Phase-6 raw pilot results are not available in this checkout")

    def test_first_and_repeated_timing_mapping_is_explicit(self) -> None:
        for dx in RESOLUTIONS_M:
            source = json.loads(_source_path(dx).read_text(encoding="utf-8"))
            row = _row(dx, source)
            dqn = source["frozen_dqn_local_sse"]
            self.assertEqual(
                row["dqn_first_solve_sec"],
                float(dqn["warmup_wall_sec_excluded"]),
            )
            self.assertEqual(
                row["dqn_repeated_query_sec"],
                float(dqn["wall_runtime_sec"]),
            )

    def test_empirical_quality_pattern_is_frozen(self) -> None:
        rows = {
            dx: _row(
                dx, json.loads(_source_path(dx).read_text(encoding="utf-8"))
            )
            for dx in RESOLUTIONS_M
        }
        self.assertTrue(all(
            row["same_sensor_J_A_10pct_diagnostic_pass"]
            for row in rows.values()
        ))
        passing_certificates = [
            dx for dx, row in rows.items()
            if row["dqn_selected_sensor_exact_local_sse_verified"]
        ]
        self.assertEqual(passing_certificates, [100.0])
        self.assertTrue(all(
            not row["phase6_pilot_quality_admissible"]
            or row["phase6_pilot_label"] == "BELLMAN_FIRST_SOLVE_FASTER"
            for row in rows.values()
        ))

    def test_closeout_manifest_and_artifacts_are_self_consistent(self) -> None:
        manifest_path = OUTPUT_DIR / "phase6_paper_closeout.json"
        if not manifest_path.is_file():
            self.skipTest("paper closeout has not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], CLOSEOUT_SCHEMA)
        self.assertEqual(manifest["status"], "complete")
        recorded_hash = manifest.pop("manifest_sha256")
        self.assertEqual(
            recorded_hash,
            sha256(_canonical_json(manifest).encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(manifest["rows"]), len(RESOLUTIONS_M))
        for name, relative_path in manifest["outputs"].items():
            path = project_paths.WORKSPACE_ROOT / Path(relative_path)
            self.assertTrue(path.is_file(), msg=f"missing artifact: {name}")
            digest = sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["output_sha256"][name])


if __name__ == "__main__":
    unittest.main()
