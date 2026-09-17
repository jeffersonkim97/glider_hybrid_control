"""Fast Stage-14.6 radius-sweep contract and accounting tests."""

from __future__ import annotations

import unittest

from stage14_6_contract import (
    DEFENDER_ACTION_COUNT,
    DEFENDER_X_MAP,
    FULL_RADIUS,
    R_NEIGHBOR_VALUES,
    configuration_audit,
    stage14_6_cases,
)
from stage14_6_worker import _expected_neighbors, _local_audit


class Stage146ContractTests(unittest.TestCase):
    def test_fixed_grid_and_radius_cases(self) -> None:
        cases = stage14_6_cases()
        local = [case for case in cases if case.worker_mode == "local_sse"]
        oracle = [case for case in cases if case.worker_mode == "global_oracle"]
        self.assertEqual(len(cases), 5)
        self.assertEqual(
            tuple(case.parameter_dict["r_neighbor"] for case in local),
            R_NEIGHBOR_VALUES,
        )
        self.assertEqual(len(oracle), 1)
        self.assertEqual(oracle[0].parameter_dict, local[-1].parameter_dict)
        self.assertEqual(len(DEFENDER_X_MAP), DEFENDER_ACTION_COUNT)
        self.assertEqual(FULL_RADIUS, DEFENDER_ACTION_COUNT - 1)

    def test_only_radius_varies_and_contract_passes(self) -> None:
        audit = configuration_audit(stage14_6_cases())
        self.assertTrue(audit["passed"])
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["r_neighbor_values"], [1, 2, 3, 5])
        self.assertEqual(DEFENDER_X_MAP, (5.0, 6.0, 7.0, 8.0, 9.0, 10.0))
        self.assertEqual(audit["neighborhood_rule"], "0 < abs(j - i) <= r_neighbor")

    def test_boundary_interior_and_full_neighbors(self) -> None:
        self.assertEqual(_expected_neighbors(0, 1), [1])
        self.assertEqual(_expected_neighbors(3, 2), [1, 2, 4, 5])
        self.assertEqual(
            _expected_neighbors(0, FULL_RADIUS),
            list(range(1, DEFENDER_ACTION_COUNT)),
        )
        center = DEFENDER_ACTION_COUNT // 2
        self.assertEqual(
            _expected_neighbors(center, FULL_RADIUS),
            [index for index in range(DEFENDER_ACTION_COUNT) if index != center],
        )

    def test_raw_unique_and_cache_accounting(self) -> None:
        records = [
            {"action_id": action_id, "attacker_br_runtime_s": 1.0}
            for action_id in range(4)
        ]
        payload = {
            "local_search": {
                "initial_defender_action_id": 0,
                "unique_defender_evaluations": 4,
                "cached_evaluation_reuses": 4,
                "evaluated_defender_actions": [0, 1, 2, 3],
                "visited_defender_actions": [0, 1, 2],
                "local_search_iterations": 3,
                "exact_evaluation_records": records,
                "iterations": [
                    {"iteration": 0, "current_action_id": 0, "neighbor_action_ids": [1], "chosen_next_action_id": 1, "payoff_improvement": 1.0},
                    {"iteration": 1, "current_action_id": 1, "neighbor_action_ids": [0, 2], "chosen_next_action_id": 2, "payoff_improvement": 1.0},
                    {"iteration": 2, "current_action_id": 2, "neighbor_action_ids": [1, 3], "chosen_next_action_id": None, "payoff_improvement": 0.0},
                ],
            },
            "timing": {"totals": {"T_SSE_s": 10.0, "T_graph_s": 1.0}},
        }
        audit = _local_audit(payload, 1)
        self.assertEqual(audit["raw_evaluation_requests"], 8)
        self.assertEqual(audit["cache_hits"], 4)
        self.assertEqual(audit["unique_exact_attacker_br_evaluations"], 4)
        self.assertEqual(audit["local_search_overhead_s"], 5.0)
        self.assertTrue(audit["neighborhood_rule_passed"])
        self.assertTrue(audit["request_accounting_passed"])
        self.assertTrue(audit["reported_cache_accounting_passed"])
        self.assertTrue(audit["unique_record_accounting_passed"])


if __name__ == "__main__":
    unittest.main()
