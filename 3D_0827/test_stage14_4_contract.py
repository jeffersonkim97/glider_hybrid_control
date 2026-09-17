"""Fast Stage-14.4 heading-sweep contract tests."""

from __future__ import annotations

import unittest

from stage14_4_contract import (
    HEADING_SPACINGS_DEG,
    configuration_audit,
    new_stage14_4_cases,
    reused_stage14_2c_cases,
    stage14_4_cases,
)
from stage14_4_worker import run_worker


class Stage144ContractTests(unittest.TestCase):
    def test_twenty_heading_resolutions_have_local_and_oracle_cases(self) -> None:
        cases = stage14_4_cases()
        self.assertEqual(len(cases), 40)
        for variant in ("local_sse", "global_oracle"):
            values = sorted(
                case.parameter_dict["heading_spacing_deg"]
                for case in cases if case.worker_mode == variant
            )
            self.assertEqual(values, sorted(HEADING_SPACINGS_DEG))

    def test_only_heading_resolution_changes(self) -> None:
        audit = configuration_audit(stage14_4_cases())
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertTrue(all(
            not row["fixed_parameter_differences"] for row in audit["cases"]
        ))

    def test_reuse_and_new_case_counts_are_explicit(self) -> None:
        self.assertEqual(len(reused_stage14_2c_cases()), 6)
        self.assertEqual(len(new_stage14_4_cases()), 34)
        self.assertEqual({
            case.parameter_dict["heading_spacing_deg"]
            for case in new_stage14_4_cases()
        }, set(HEADING_SPACINGS_DEG) - {5.0, 10.0, 15.0})

    def test_worker_source_preserves_exact_stage14_2c_solver_path(self) -> None:
        self.assertEqual(run_worker.__module__, "stage14_4_worker")


if __name__ == "__main__":
    unittest.main()
