"""Fast Stage-14.5 switching-density contract tests."""

from __future__ import annotations

import unittest

from stage14_5_contract import (
    CONTOUR_SAMPLE_COUNTS,
    RADIAL_SAMPLE_COUNT,
    configuration_audit,
    stage14_5_cases,
)
from stage14_5_worker import FUNNEL_KEYS, run_worker


class Stage145ContractTests(unittest.TestCase):
    def test_three_monotonic_densities_have_local_and_oracle_cases(self) -> None:
        cases = stage14_5_cases()
        self.assertEqual(len(cases), 6)
        for variant in ("local_sse", "global_oracle"):
            values = sorted(
                case.parameter_dict["switching_contour_sample_count"]
                for case in cases if case.worker_mode == variant
            )
            self.assertEqual(values, list(CONTOUR_SAMPLE_COUNTS))

    def test_only_contour_candidate_density_changes(self) -> None:
        audit = configuration_audit(stage14_5_cases())
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertTrue(all(
            not row["fixed_parameter_differences"] for row in audit["cases"]
        ))
        self.assertEqual(audit["radial_sample_count"], RADIAL_SAMPLE_COUNT)
        self.assertEqual(audit["expected_raw_candidate_counts"], [48, 72, 96])

    def test_filter_funnel_contract_is_explicit(self) -> None:
        self.assertEqual(FUNNEL_KEYS, (
            "N_C_raw", "N_C_unique", "N_C_powered_feasible",
            "N_C_energy_feasible", "N_C_virtual_feasible",
            "N_C_goal_reachable", "N_C_feasible",
        ))
        self.assertEqual(run_worker.__module__, "stage14_5_worker")


if __name__ == "__main__":
    unittest.main()
