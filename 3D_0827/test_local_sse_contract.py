"""Stage 14.2A unit tests for the local-SSE mathematical contract."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from local_sse_contract import (
    DefenderGridTopology,
    DefenderNeighborhoodConfig,
    FollowerPayoffRecord,
    LocalDefenderEvaluation,
    audit_strong_follower_selection,
    global_sse_action_ids,
    verify_local_sse,
)


def _feasible(action_id: int, value: float) -> LocalDefenderEvaluation:
    return LocalDefenderEvaluation(
        action_id=action_id,
        status="feasible",
        defender_value=value,
        attacker_objective=10.0 - value,
        selected_attacker_response_id=100 + action_id,
        exact_attacker_best_response_verified=True,
        strong_tie_break_verified=True,
    )


class LocalSSEContractTests(unittest.TestCase):
    def test_lsse_a1_handwritten_line_distinguishes_local_and_global(self) -> None:
        values = (1.0, 4.0, 3.0, 7.0, 6.0)
        evaluations = {index: _feasible(index, value) for index, value in enumerate(values)}
        topology = DefenderGridTopology.ordered_line(tuple(evaluations))
        configuration = DefenderNeighborhoodConfig(r_neighbor=1)
        local_ids = tuple(
            action_id for action_id in evaluations
            if verify_local_sse(action_id, evaluations, topology, configuration).local_sse_verified
        )
        self.assertEqual(local_ids, (1, 3))
        self.assertEqual(global_sse_action_ids(evaluations), (3,))

    def test_lsse_a2_boundary_and_configurable_radius(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1, 2, 3, 4))
        self.assertEqual(
            topology.neighbors(0, DefenderNeighborhoodConfig(r_neighbor=1)), (1,)
        )
        self.assertEqual(
            topology.neighbors(0, DefenderNeighborhoodConfig(r_neighbor=2)), (1, 2)
        )
        self.assertEqual(
            topology.neighbors(2, DefenderNeighborhoodConfig(r_neighbor=2)),
            (0, 1, 3, 4),
        )
        full = DefenderNeighborhoodConfig(r_neighbor=4)
        self.assertTrue(topology.radius_covers_all_actions(full))
        evaluations = {index: _feasible(index, 5.0 - index) for index in range(5)}
        boundary = verify_local_sse(0, evaluations, topology)
        self.assertTrue(boundary.local_sse_verified)
        self.assertEqual(boundary.neighbor_action_ids, (1,))

    def test_two_dimensional_chebyshev_square_radius(self) -> None:
        topology = DefenderGridTopology(tuple(
            (row * 5 + column, (row, column))
            for row in range(5) for column in range(5)
        ))
        center = 2 * 5 + 2
        radius_one = topology.neighbors(center, DefenderNeighborhoodConfig(1))
        radius_two = topology.neighbors(center, DefenderNeighborhoodConfig(2))
        self.assertEqual(len(radius_one), 8)
        self.assertEqual(len(radius_two), 24)

    def test_lsse_a3_infeasible_neighbor_is_diagnostic_not_fabricated(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1, 2))
        evaluations = {
            0: LocalDefenderEvaluation(
                action_id=0, status="model_infeasible", defender_value=None,
                attacker_objective=None, selected_attacker_response_id=None,
                exact_attacker_best_response_verified=True,
                strong_tie_break_verified=False,
                diagnostic="no feasible Attacker response",
            ),
            1: _feasible(1, 4.0),
            2: _feasible(2, 3.0),
        }
        result = verify_local_sse(1, evaluations, topology)
        self.assertTrue(result.local_sse_verified)
        self.assertEqual(
            result.infeasible_neighbor_diagnostics,
            ((0, "no feasible Attacker response"),),
        )
        self.assertEqual(result.feasible_neighbor_ids, (2,))

    def test_unknown_or_failed_neighbor_prevents_certification(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1, 2))
        evaluations = {
            0: _feasible(0, 1.0),
            1: _feasible(1, 4.0),
            2: LocalDefenderEvaluation(
                action_id=2, status="timeout", defender_value=None,
                attacker_objective=None, selected_attacker_response_id=None,
                exact_attacker_best_response_verified=False,
                strong_tie_break_verified=False, diagnostic="test timeout",
            ),
        }
        result = verify_local_sse(1, evaluations, topology)
        self.assertFalse(result.local_sse_verified)
        self.assertEqual(result.unknown_neighbor_diagnostics, ((2, "timeout"),))

    def test_lsse_a4_strong_follower_tie_is_audited(self) -> None:
        records = (
            FollowerPayoffRecord(10, attacker_objective=1.0, defender_payoff=0.2),
            FollowerPayoffRecord(11, attacker_objective=1.0, defender_payoff=0.7),
            FollowerPayoffRecord(12, attacker_objective=2.0, defender_payoff=0.9),
        )
        passed = audit_strong_follower_selection(records, 11)
        failed = audit_strong_follower_selection(records, 10)
        self.assertTrue(passed.passed)
        self.assertEqual(passed.attacker_best_response_ids, (10, 11))
        self.assertEqual(passed.expected_selected_response_id, 11)
        self.assertFalse(failed.passed)

    def test_lsse_a5_global_implies_local_and_full_radius_is_equivalent(self) -> None:
        values = (1.0, 4.0, 3.0, 7.0, 6.0)
        evaluations = {index: _feasible(index, value) for index, value in enumerate(values)}
        topology = DefenderGridTopology.ordered_line(tuple(evaluations))
        global_id = global_sse_action_ids(evaluations)[0]
        local = verify_local_sse(global_id, evaluations, topology)
        full = verify_local_sse(
            global_id, evaluations, topology,
            DefenderNeighborhoodConfig(r_neighbor=len(values) - 1),
        )
        self.assertTrue(local.local_sse_verified)
        self.assertTrue(full.local_sse_verified)
        self.assertTrue(full.global_condition_equivalent)
        full_local_ids = tuple(
            action_id for action_id in evaluations
            if verify_local_sse(
                action_id, evaluations, topology,
                DefenderNeighborhoodConfig(r_neighbor=len(values) - 1),
            ).local_sse_verified
        )
        self.assertEqual(full_local_ids, global_sse_action_ids(evaluations))

    def test_contract_objects_are_immutable_and_radius_is_validated(self) -> None:
        configuration = DefenderNeighborhoodConfig()
        self.assertEqual(configuration.r_neighbor, 1)
        with self.assertRaises(FrozenInstanceError):
            configuration.r_neighbor = 2  # type: ignore[misc]
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                DefenderNeighborhoodConfig(invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
