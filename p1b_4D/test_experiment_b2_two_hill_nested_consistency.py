"""Structural tests for the B2 experiment driver (no production solves)."""
from __future__ import annotations

import unittest

from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    COMMON_EVALUATOR_SAMPLE_COUNTS,
    _select_global_evaluator,
    build_b2_case_matrix,
)


class B2ExperimentDriverTests(unittest.TestCase):
    def test_case_matrix_is_the_frozen_nine_case_design(self) -> None:
        cases = build_b2_case_matrix()
        self.assertEqual(len(cases), 9)
        self.assertEqual(len({case["case_id"] for case in cases}), 9)
        self.assertEqual(
            sum(case["role"] == "main" for case in cases), 3
        )
        self.assertEqual(
            sum(case["role"] == "ablation" for case in cases), 3
        )
        self.assertEqual(
            sum(case["role"] == "speed_sensitivity" for case in cases), 2
        )
        self.assertEqual(
            sum(case["role"] == "quadrature_sensitivity" for case in cases),
            1,
        )

    def test_global_evaluator_selects_first_pair_passing_every_policy(self) -> None:
        pairs = tuple(zip(
            COMMON_EVALUATOR_SAMPLE_COUNTS[:-1],
            COMMON_EVALUATOR_SAMPLE_COUNTS[1:],
        ))
        cases = []
        for case_index in range(2):
            qualifications = {}
            for lower, upper in pairs:
                qualifications[f"{lower}_vs_{upper}"] = {
                    "passed": lower >= 1025
                }
            cases.append({
                "sensor_name": f"sensor_{case_index}",
                "case_id": "policy",
                "status": "feasible",
                "high_fidelity": {"qualifications": qualifications},
            })
        sample_count, gate = _select_global_evaluator(cases)
        self.assertEqual(sample_count, 1025)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["qualification_pair"], (1025, 2049))


if __name__ == "__main__":
    unittest.main()
