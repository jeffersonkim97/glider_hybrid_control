"""Fast Stage-14.3 spatial-sweep contract tests."""

from __future__ import annotations

import unittest

from stage14_3_contract import (
    SPATIAL_RESOLUTIONS_M,
    configuration_audit,
    new_stage14_3_cases,
    reused_stage14_2c_cases,
    stage14_3_cases,
)
from stage14_3_worker import run_worker


class Stage143ContractTests(unittest.TestCase):
    def test_twenty_resolutions_have_matched_local_and_oracle_cases(self) -> None:
        cases = stage14_3_cases()
        self.assertEqual(len(cases), 40)
        for variant in ("local_sse", "global_oracle"):
            values = sorted(
                case.parameter_dict["spatial_resolution_m"]
                for case in cases if case.worker_mode == variant
            )
            self.assertEqual(values, sorted(SPATIAL_RESOLUTIONS_M))

    def test_only_spatial_resolution_changes(self) -> None:
        audit = configuration_audit(stage14_3_cases())
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertTrue(all(
            row["stage14_3_sweep_variable"] == "spatial_resolution_m"
            for row in audit["cases"]
        ))

    def test_reuse_and_new_case_counts_are_explicit(self) -> None:
        self.assertEqual(len(reused_stage14_2c_cases()), 6)
        self.assertEqual(len(new_stage14_3_cases()), 34)
        self.assertEqual(
            sorted({case.parameter_dict["spatial_resolution_m"] for case in new_stage14_3_cases()}),
            sorted(set(SPATIAL_RESOLUTIONS_M) - {25.0, 50.0, 100.0}),
        )

    def test_worker_source_preserves_exact_stage14_2c_solver_path(self) -> None:
        self.assertEqual(run_worker.__module__, "stage14_3_worker")


if __name__ == "__main__":
    unittest.main()
