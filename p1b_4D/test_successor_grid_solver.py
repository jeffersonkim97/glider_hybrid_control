"""Independent validation of the physical successor-grid follower."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import networkx as nx
import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stackelberg_solver import solve_attacker_best_response
from p1b_4D.stage_cost import construct_state_grids
from p1b_4D.successor_grid_solver import (
    _select_exact_minimum_candidate,
    build_successor_grid_graph,
    solve_successor_grid_bellman,
)
from p1b_4D.test_continuous_replay_evaluation import _build_toy_bundles


class SuccessorGridSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.configuration, _, _ = _build_toy_bundles(
            Path(self.temporary_directory.name)
        )
        self.configuration["primary_result"]["attacker_solver_config"][
            "transition_model"
        ] = "successor_grid_physical_edge"
        self.addCleanup(
            close_phase_logger,
            self.configuration["primary_result"]["logging_utilities"]["logger"],
        )

    def test_virtual_switch_response_is_continuously_executable(self) -> None:
        result = solve_attacker_best_response(
            600.0, self.configuration, "physical-successor-test"
        )
        best = result["primary_result"]["best_found_attacker_response"]
        self.assertEqual(
            result["metadata"]["transition_model"],
            "successor_grid_physical_edge",
        )
        self.assertTrue(best["metadata"]["virtual_switching_state"])
        self.assertFalse(best["metadata"]["endpoint_snapping"])
        self.assertTrue(best["metadata"]["all_segment_geometry_certificate"])
        self.assertTrue(best["validation"]["checks"]["terrain_clearance"])
        self.assertTrue(best["validation"]["checks"]["los_feasibility"])
        self.assertTrue(best["validation"]["checks"]["airspace_feasibility"])
        self.assertTrue(best["continuous_replay_validation"]["feasible"])
        self.assertTrue(best["validation"]["checks"]["physical_edge_endpoint_alignment"])
        self.assertLessEqual(
            best["constraint_residuals"]["maximum_edge_endpoint_residual"],
            self.configuration["primary_result"]["validation_config"]["solver_tolerance"],
        )
        self.assertEqual(
            len(best["duration_profile"]), len(best["trajectory"]) - 1
        )
        self.assertTrue(
            result["validation"]["checks"]["selection_is_exact_minimum_cost"]
        )
        candidates = result["primary_result"]["bellman_candidate_bundle"][
            "primary_result"
        ]["candidates"]
        graph_metadata = result["primary_result"]["bellman_candidate_bundle"][
            "metadata"
        ]["graph_metadata"]
        self.assertTrue(graph_metadata["all_segment_geometry_certificate"])
        self.assertEqual(
            float(best["mission_cost"]),
            min(float(candidate["mission_cost"]) for candidate in candidates),
        )

    def test_near_tie_never_displaces_absolute_minimum(self) -> None:
        tolerance = float(
            self.configuration["primary_result"]["validation_config"][
                "objective_tolerance"
            ]
        )
        delta = max(0.5 * tolerance, float(np.spacing(1.0)))
        self.assertLessEqual(delta, tolerance)
        exact_minimum = {
            "mission_cost": 1.0,
            "switching_point": np.array([20.0, 0.0]),
            "metadata": {"seed_index": 1},
        }
        earlier_near_tie = {
            "mission_cost": 1.0 + delta,
            "switching_point": np.array([10.0, 0.0]),
            "metadata": {"seed_index": 0},
        }

        best, ordered, tied = _select_exact_minimum_candidate(
            [earlier_near_tie, exact_minimum]
        )

        self.assertIs(best, exact_minimum)
        self.assertIs(ordered[0], exact_minimum)
        self.assertEqual(tied, [exact_minimum])

    def test_exact_cost_tie_uses_smallest_switching_z(self) -> None:
        later = {
            "mission_cost": 1.0,
            "switching_point": np.array([20.0, 0.0]),
            "metadata": {"seed_index": 1},
        }
        earlier = {
            "mission_cost": 1.0,
            "switching_point": np.array([10.0, 0.0]),
            "metadata": {"seed_index": 2},
        }

        best, ordered, tied = _select_exact_minimum_candidate([later, earlier])

        self.assertIs(best, earlier)
        self.assertEqual(ordered, [earlier, later])
        self.assertEqual(tied, [earlier, later])

    def test_bellman_values_match_independent_networkx_shortest_paths(self) -> None:
        geometry = build_geometry_bundle(self.configuration)
        grids = construct_state_grids(
            self.configuration["primary_result"]["environment_config"],
            self.configuration["primary_result"]["vehicle_config"],
        )
        graph = build_successor_grid_graph(self.configuration, geometry, grids)
        policy = solve_successor_grid_bellman(
            graph, grids, geometry["primary_result"]["goal_position"]
        )
        sink = ("goal",)
        independent = nx.DiGraph()
        nz, nh = policy["value"].shape
        for zi in range(nz):
            for hi in range(nh):
                node = (zi, hi)
                if policy["goal_mask"][zi, hi]:
                    independent.add_edge(node, sink, weight=0.0)
                for ai, action in enumerate(graph["actions"]):
                    if not graph["valid"][zi, hi, ai]:
                        continue
                    successor = sink if graph["terminal"][zi, hi, ai] else (
                        zi + action["forward_cells"],
                        hi - action["descent_cells"],
                    )
                    weight = float(graph["cost"][zi, hi, ai])
                    previous = independent.get_edge_data(node, successor)
                    if previous is None or weight < previous["weight"]:
                        independent.add_edge(node, successor, weight=weight)
        reversed_graph = independent.reverse(copy=False)
        distances = nx.single_source_dijkstra_path_length(
            reversed_graph, sink, weight="weight"
        )
        for node, distance in distances.items():
            if node == sink:
                continue
            self.assertAlmostEqual(policy["value"][node], distance, places=11)

    def test_two_hill_geometry_runs_without_hill_specific_logic(self) -> None:
        configuration = deepcopy(self.configuration)
        configuration["primary_result"]["environment_config"]["terrain"]["hills"] = (
            {"z_ridge": 400.0, "h_ridge": 50.0, "width": 60.0},
            {"z_ridge": 180.0, "h_ridge": 0.5, "width": 35.0},
        )
        result = solve_attacker_best_response(
            600.0, configuration, "physical-successor-two-hill"
        )
        best = result["primary_result"]["best_found_attacker_response"]
        self.assertTrue(result["status"]["success"])
        self.assertTrue(best["continuous_replay_validation"]["feasible"])
        self.assertEqual(
            result["primary_result"]["bellman_candidate_bundle"]["metadata"][
                "graph_metadata"
            ]["multi_hill_geometry_api"],
            True,
        )

    def test_legacy_solver_remains_selectable(self) -> None:
        legacy = deepcopy(self.configuration)
        legacy["primary_result"]["attacker_solver_config"][
            "transition_model"
        ] = "snapped_fixed_time_step"
        result = solve_attacker_best_response(600.0, legacy, "legacy-switch-test")
        self.assertEqual(result["metadata"]["transition_model"], "snapped_fixed_time_step")
        self.assertEqual(result["metadata"]["solution_method"], "bellman_dynamic_programming")


if __name__ == "__main__":
    unittest.main()
