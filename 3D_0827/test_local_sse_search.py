"""Stage 14.2B local-search unit tests on hand-written finite games."""

from __future__ import annotations

import unittest

from local_sse_contract import (
    DefenderGridTopology,
    DefenderNeighborhoodConfig,
    LocalDefenderEvaluation,
    global_sse_action_ids,
)
from local_sse_search import run_local_sse_search


def _feasible(action_id: int, value: float) -> LocalDefenderEvaluation:
    return LocalDefenderEvaluation(
        action_id=action_id, status="feasible", defender_value=value,
        attacker_objective=10.0 - value,
        selected_attacker_response_id=100 + action_id,
        exact_attacker_best_response_verified=True,
        strong_tie_break_verified=True,
    )


def _game(values: tuple[float, ...]):
    evaluations = {index: _feasible(index, value) for index, value in enumerate(values)}
    calls: list[int] = []

    def evaluator(action_id: int) -> LocalDefenderEvaluation:
        calls.append(action_id)
        return evaluations[action_id]

    return evaluations, evaluator, calls, DefenderGridTopology.ordered_line(tuple(evaluations))


class LocalSSESearchTests(unittest.TestCase):
    def test_lsse_b1_monotone_line_reaches_endpoint(self) -> None:
        _, evaluator, calls, topology = _game((1.0, 2.0, 3.0, 4.0))
        result = run_local_sse_search(0, topology, evaluator)
        self.assertTrue(result.local_sse_verified)
        self.assertEqual(result.final_local_sse_action_id, 3)
        self.assertEqual(result.visited_defender_actions, (0, 1, 2, 3))
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(result.unique_defender_evaluations, 4)
        self.assertGreater(result.cached_evaluation_reuses, 0)

    def test_lsse_b2_two_basins_depend_on_initialization(self) -> None:
        values = (1.0, 4.0, 3.0, 2.0, 5.0, 1.0)
        _, evaluator_left, _, topology = _game(values)
        _, evaluator_right, _, _ = _game(values)
        left = run_local_sse_search(0, topology, evaluator_left)
        right = run_local_sse_search(5, topology, evaluator_right)
        self.assertEqual(left.final_local_sse_action_id, 1)
        self.assertEqual(right.final_local_sse_action_id, 4)
        self.assertTrue(left.local_sse_verified and right.local_sse_verified)

    def test_lsse_b3_local_but_not_global_is_not_mislabeled(self) -> None:
        evaluations, evaluator, _, topology = _game((1.0, 4.0, 3.0, 7.0, 6.0))
        result = run_local_sse_search(0, topology, evaluator)
        self.assertEqual(result.final_local_sse_action_id, 1)
        self.assertNotIn(1, global_sse_action_ids(evaluations))
        self.assertTrue(result.local_sse_verified)
        self.assertFalse(result.global_optimality_evaluated)
        self.assertIsNone(result.global_optimal)
        self.assertIn("not a global", result.solution_scope)

    def test_lsse_b4_boundary_solution_uses_one_sided_neighbor(self) -> None:
        _, evaluator, _, topology = _game((1.0, 2.0, 3.0))
        result = run_local_sse_search(2, topology, evaluator)
        self.assertEqual(result.final_local_sse_action_id, 2)
        self.assertEqual(result.iterations[0].neighbor_action_ids, (1,))
        self.assertEqual(result.visited_defender_actions, (2,))

    def test_lsse_b5_equal_plateau_terminates_without_cycle(self) -> None:
        _, evaluator, _, topology = _game((1.0, 4.0, 4.0, 3.0))
        result = run_local_sse_search(1, topology, evaluator)
        self.assertEqual(result.final_local_sse_action_id, 1)
        self.assertEqual(result.visited_defender_actions, (1,))
        self.assertEqual(
            result.iterations[0].local_verification.equal_payoff_neighbor_ids, (2,)
        )
        self.assertEqual(result.termination_status, "certified_local_sse")

    def test_infeasible_initial_action_recovers_within_radius(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1, 2))

        def evaluator(action_id: int) -> LocalDefenderEvaluation:
            if action_id == 0:
                return LocalDefenderEvaluation(
                    action_id=0, status="model_infeasible", defender_value=None,
                    attacker_objective=None, selected_attacker_response_id=None,
                    exact_attacker_best_response_verified=True,
                    strong_tie_break_verified=False, diagnostic="no exact response",
                )
            return _feasible(action_id, float(action_id))

        result = run_local_sse_search(0, topology, evaluator)
        self.assertEqual(result.termination_status, "certified_local_sse")
        self.assertEqual(result.visited_defender_actions, (0, 1, 2))
        self.assertEqual(result.evaluated_defender_actions, (0, 1, 2))
        self.assertTrue(result.initialization_recovery_attempted)
        self.assertEqual(result.initialization_neighbor_action_ids, (1,))
        self.assertEqual(result.initialization_feasible_neighbor_ids, (1,))
        self.assertEqual(result.initialization_recovery_action_id, 1)
        self.assertTrue(result.local_sse_verified)

    def test_recovery_radius_is_measured_in_grid_index_steps(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1, 2, 3))

        def evaluator(action_id: int) -> LocalDefenderEvaluation:
            if action_id < 2:
                return LocalDefenderEvaluation(
                    action_id=action_id, status="model_infeasible", defender_value=None,
                    attacker_objective=None, selected_attacker_response_id=None,
                    exact_attacker_best_response_verified=True,
                    strong_tie_break_verified=False, diagnostic="no exact response",
                )
            return _feasible(action_id, float(action_id))

        radius_one = run_local_sse_search(
            0, topology, evaluator, DefenderNeighborhoodConfig(r_neighbor=1)
        )
        self.assertEqual(radius_one.termination_status, "initial_neighborhood_infeasible")
        self.assertEqual(radius_one.evaluated_defender_actions, (0, 1))
        self.assertEqual(radius_one.initialization_neighbor_action_ids, (1,))

        radius_two = run_local_sse_search(
            0, topology, evaluator, DefenderNeighborhoodConfig(r_neighbor=2)
        )
        self.assertTrue(radius_two.local_sse_verified)
        self.assertEqual(radius_two.initialization_neighbor_action_ids, (1, 2))
        self.assertEqual(radius_two.initialization_recovery_action_id, 2)

    def test_required_neighbor_failure_prevents_certificate(self) -> None:
        topology = DefenderGridTopology.ordered_line((0, 1))

        def evaluator(action_id: int) -> LocalDefenderEvaluation:
            if action_id == 0:
                return _feasible(0, 2.0)
            raise TimeoutError("controlled")

        result = run_local_sse_search(0, topology, evaluator)
        self.assertEqual(result.termination_status, "required_neighbor_comparison_unknown")
        self.assertFalse(result.local_sse_verified)
        self.assertEqual(result.unknown_neighbor_diagnostics[0]["status"], "timeout")

    def test_configurable_radius_changes_search_neighborhood(self) -> None:
        _, evaluator, _, topology = _game((1.0, 2.0, 9.0, 3.0, 4.0))
        result = run_local_sse_search(
            0, topology, evaluator, DefenderNeighborhoodConfig(r_neighbor=2)
        )
        self.assertEqual(result.visited_defender_actions[0:2], (0, 2))
        self.assertEqual(result.iterations[0].neighbor_action_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
