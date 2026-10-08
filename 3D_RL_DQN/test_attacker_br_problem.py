"""Parity tests for the Phase 1 solver-independent attacker BR boundary."""

from __future__ import annotations

import unittest

import numpy as np

import project_paths  # noqa: F401
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import QLearningConfig, switching_candidates
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem, solve_attacker_br


class AttackerBRProblemParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = build_scene(ComputationCondition(spatial_resolution_m=100.0))
        cls.sensor = tuple(
            float(value)
            for value in cls.scene.defender_grid.position(cls.scene.seed_action_id)
        )
        cls.problem = AttackerBRProblem(cls.scene, cls.sensor)
        cls.oracle = exact_best_response(cls.scene, cls.sensor, keep_solution=True)
        if not cls.oracle.feasible:
            raise unittest.SkipTest("canonical cube exact response is infeasible")

    def test_initial_condition_and_sensor_are_the_existing_scene_values(self) -> None:
        np.testing.assert_array_equal(
            self.problem.scene.config.start.as_array(),
            self.scene.config.start.as_array(),
        )
        self.assertEqual(self.problem.sensor_map, self.sensor)

    def test_state_round_trip_uses_bellman_state_directly(self) -> None:
        for state_id in self.oracle.glide_state_ids:
            self.assertEqual(
                self.problem.state_id(self.problem.state(state_id)), state_id,
            )

    def test_local_sse_radius_is_not_an_attacker_problem_attribute(self) -> None:
        self.assertFalse(hasattr(self.problem, "r_neighbor"))

    def test_switching_candidates_match_the_existing_generator(self) -> None:
        expected = switching_candidates(self.scene, self.sensor).state_ids
        self.assertEqual(self.problem.switching_state_ids(), expected)

    def test_terminal_classification_matches_the_shared_graph(self) -> None:
        terminal = int(self.oracle.glide_state_ids[-1])
        nonterminal = int(self.oracle.glide_state_ids[0])
        self.assertTrue(self.problem.is_terminal(terminal))
        self.assertFalse(self.problem.is_terminal(nonterminal))
        self.assertEqual(
            self.problem.is_terminal(terminal),
            bool(self.scene.graph.terminal_mask[terminal]),
        )

    def test_successors_are_the_existing_goal_reachable_adjacency(self) -> None:
        for state_id in self.oracle.glide_state_ids[:-1]:
            common = self.problem.successors(state_id)
            legacy = tuple(self.scene.graph.adjacency[state_id])
            self.assertEqual(
                tuple(edge.target_id for edge in common),
                tuple(edge.target_id for edge in legacy),
            )
            self.assertEqual(common, legacy)

    def test_raw_feasibility_filters_only_by_goal_reachability(self) -> None:
        state_id = int(self.oracle.glide_state_ids[0])
        raw, _, _ = self.problem.raw_transition_candidates(state_id)
        expected = tuple(
            edge for edge in raw if self.scene.graph.node_mask[edge.target_id]
        )
        self.assertEqual(self.problem.successors(state_id), expected)

    def test_immediate_cost_matches_bellman_cost_rows(self) -> None:
        solution = self.oracle.solution
        checked = 0
        for state_id in self.oracle.glide_state_ids[:-1]:
            legacy_row = solution.edge_cost_by_source[state_id]
            common = self.problem.successors(state_id)
            self.assertEqual(len(legacy_row), len(common))
            for action_index, expected in enumerate(legacy_row):
                actual = self.problem.transition(state_id, action_index)
                self.assertAlmostEqual(actual.stage_cost, expected, delta=1.0e-12)
                checked += 1
        self.assertGreater(checked, 0)

    def test_full_trajectory_objective_matches_exact_response(self) -> None:
        actual = self.problem.attacker_objective(
            int(self.oracle.switching_state_id),
            tuple(self.oracle.glide_state_ids),
        )
        self.assertAlmostEqual(actual, self.oracle.attacker_objective, delta=1.0e-12)

    def test_bellman_adapter_preserves_result_and_local_sse_contract(self) -> None:
        result = solve_attacker_br(self.problem, "bellman")
        self.assertTrue(result.success)
        self.assertTrue(result.reached_goal)
        self.assertEqual(result.trajectory, self.oracle.glide_state_ids)
        self.assertAlmostEqual(
            result.attacker_objective, self.oracle.attacker_objective, delta=1.0e-12,
        )
        local = result.as_local_defender_evaluation(self.scene.seed_action_id)
        self.assertTrue(local.feasible)
        self.assertTrue(local.exact_attacker_best_response_verified)
        self.assertTrue(local.strong_tie_break_verified)

    def test_tabular_adapter_uses_supplied_existing_configuration(self) -> None:
        config = QLearningConfig(episodes=200, evaluation_interval=100)
        result = solve_attacker_br(self.problem, "tabular_q", config)
        self.assertEqual(result.method, "tabular_q")
        self.assertEqual(result.diagnostics["config"], config.as_dict())
        if result.success:
            self.assertTrue(result.reached_goal)
            local = result.as_local_defender_evaluation(self.scene.seed_action_id)
            self.assertFalse(local.exact_attacker_best_response_verified)
            self.assertFalse(local.strong_tie_break_verified)


if __name__ == "__main__":
    unittest.main()
