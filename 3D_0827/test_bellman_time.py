"""Stage-6 physical time-objective and exact Bellman tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import unittest

import numpy as np

from bellman_geometry import GlideEdge
from bellman_graph import BellmanGraph, GraphStatistics, build_bellman_graph
from bellman_objectives import TimeObjective
from bellman_solver import (
    independent_dijkstra_time,
    replay_time_optimal_path,
    run_time_optimal_problem,
    solve_time_optimal,
)
from bellman_state import BellmanState, BellmanStateGrid
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import MapBounds
from scenario import Point3D
from solver_metrics import SolverMetrics, SolverTiming


@dataclass(frozen=True)
class FlatTerrainFixture:
    bounds: MapBounds
    ground_z: float = 0.0

    @property
    def maximum_height(self) -> float:
        return self.ground_z

    def contains_solid(self, point: np.ndarray, *, tolerance: float = 0.0) -> bool:
        return False

    def segment_intersects_solid(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        tolerance: float = 1.0e-9,
    ) -> bool:
        return False


def _manual_edge(
    grid: BellmanStateGrid,
    source: BellmanState,
    target: BellmanState,
    duration_s: float,
) -> GlideEdge:
    speed = DEFAULT_GLIDER.best_glide_speed_mps
    return GlideEdge(
        source_id=grid.encode(source),
        target_id=grid.encode(target),
        source_state=source,
        target_state=target,
        horizontal_distance_m=duration_s * speed,
        altitude_loss_m=float(
            (source.altitude_index - target.altitude_index)
            * grid.altitude_spacing_map
            * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
        ),
        duration_s=duration_s,
        heading_change_rad=0.0,
    )


def _hand_written_graph(*, tied: bool = False) -> tuple[BellmanGraph, int, int]:
    """S-A-G costs 2+2; S-B-G costs 1+5, or 2+2 when tied."""
    bounds = MapBounds(0.0, 3.0, 0.0, 1.0)
    grid = BellmanStateGrid(
        bounds=bounds,
        minimum_altitude_map=0.0,
        maximum_altitude_map=3.0,
        altitude_spacing_map=1.0,
    )
    terrain = FlatTerrainFixture(bounds)
    start = BellmanState(0, 0, 3, 0)
    first = BellmanState(1, 0, 2, 3)
    second = BellmanState(2, 0, 1, 1)
    goal_state = BellmanState(3, 0, 0, 0)
    start_id = grid.encode(start)
    first_id = grid.encode(first)
    second_id = grid.encode(second)
    goal_id = grid.encode(goal_state)

    first_branch = (
        _manual_edge(grid, start, first, 2.0),
        _manual_edge(grid, first, goal_state, 2.0),
    )
    second_branch = (
        _manual_edge(grid, start, second, 2.0 if tied else 1.0),
        _manual_edge(grid, second, goal_state, 2.0 if tied else 5.0),
    )
    adjacency: list[tuple[GlideEdge, ...]] = [
        () for _ in range(grid.state_count)
    ]
    # Put action 3 first intentionally; deterministic tie-breaking must still
    # choose action 1 when both branches have the same total duration.
    adjacency[start_id] = (first_branch[0], second_branch[0])
    adjacency[first_id] = (first_branch[1],)
    adjacency[second_id] = (second_branch[1],)
    predecessors: list[list[int]] = [[] for _ in range(grid.state_count)]
    for source_id, edges in enumerate(adjacency):
        for edge in edges:
            predecessors[edge.target_id].append(source_id)

    node_mask = np.zeros(grid.state_count, dtype=bool)
    node_mask[[start_id, first_id, second_id, goal_id]] = True
    terminal_mask = np.zeros(grid.state_count, dtype=bool)
    terminal_mask[goal_id] = True
    graph = BellmanGraph(
        grid=grid,
        terrain=terrain,
        goal=Point3D(3.0, 0.0, 0.0),
        node_mask=node_mask,
        terminal_mask=terminal_mask,
        adjacency=tuple(adjacency),
        predecessors=tuple(tuple(values) for values in predecessors),
        statistics=GraphStatistics(
            cartesian_state_count=grid.state_count,
            state_count=4,
            terrain_excluded_state_count=grid.state_count - 4,
            valid_edge_count=4,
            rejected_by_bounds=0,
            rejected_by_terrain=0,
            rejected_by_turn=0,
            rejected_by_altitude=0,
            terminal_state_count=1,
        ),
    )
    return graph, start_id, second_id


class BellmanTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bounds = MapBounds(0.0, 3.0, -1.0, 1.0)
        cls.grid = BellmanStateGrid(
            bounds=cls.bounds,
            maximum_altitude_map=0.3,
        )
        cls.terrain = FlatTerrainFixture(cls.bounds)
        cls.goal = Point3D(3.0, 0.0, 0.0)
        cls.graph = build_bellman_graph(cls.grid, cls.terrain, cls.goal)
        cls.solution = solve_time_optimal(cls.graph)
        cls.start = BellmanState(0, 1, 3, 0)
        cls.start_id = cls.grid.encode(cls.start)

    def test_time_edge_cost_uses_documented_horizontal_convention(self) -> None:
        objective = TimeObjective()
        self.assertEqual(objective.path_length_convention, "horizontal")
        cardinal = next(
            edge
            for edge in self.graph.adjacency[self.start_id]
            if edge.target_state.heading_bin == 0
        )
        self.assertAlmostEqual(cardinal.horizontal_distance_m, 100.0)
        self.assertAlmostEqual(
            objective.edge_cost_s(cardinal),
            100.0 / DEFAULT_GLIDER.best_glide_speed_mps,
            places=12,
        )

    def test_b6_1_obstacle_free_direct_path_matches_analytical_bound(self) -> None:
        path = replay_time_optimal_path(self.solution, self.start_id)
        expected = 300.0 / DEFAULT_GLIDER.best_glide_speed_mps
        self.assertAlmostEqual(path.bellman_value_s, expected, places=12)
        self.assertEqual(len(path.edges), 3)
        positions = np.vstack([
            self.grid.position_map(self.grid.decode(state_id))
            for state_id in path.state_ids
        ])
        np.testing.assert_allclose(positions[:, 1], 0.0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            positions[:, 0],
            np.array([0.0, 1.0, 2.0, 3.0]),
            rtol=0.0,
            atol=0.0,
        )

    def test_b6_2_hand_written_two_path_dag_returns_four_seconds(self) -> None:
        graph, start_id, _ = _hand_written_graph()
        solution = solve_time_optimal(graph)
        self.assertAlmostEqual(solution.value_s[start_id], 4.0, places=12)
        self.assertEqual(solution.policy_action_index[start_id], 3)

    def test_b6_3_bellman_matches_independent_dijkstra(self) -> None:
        reference = independent_dijkstra_time(self.graph)
        finite = np.isfinite(self.solution.value_s)
        np.testing.assert_array_equal(finite, np.isfinite(reference))
        np.testing.assert_allclose(
            self.solution.value_s[finite],
            reference[finite],
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_b6_4_bellman_edge_sum_and_geometric_replay_agree(self) -> None:
        path = replay_time_optimal_path(self.solution, self.start_id)
        self.assertAlmostEqual(
            path.bellman_value_s,
            path.summed_edge_duration_s,
            places=12,
        )
        self.assertAlmostEqual(
            path.summed_edge_duration_s,
            path.geometric_replay_duration_s,
            places=12,
        )

    def test_b6_5_unreachable_state_stays_infinite_without_policy(self) -> None:
        unreachable_id = self.grid.encode(BellmanState(0, 1, 2, 0))
        self.assertTrue(np.isposinf(self.solution.value_s[unreachable_id]))
        self.assertEqual(self.solution.policy_successor[unreachable_id], -1)
        self.assertEqual(self.solution.policy_action_index[unreachable_id], -1)
        self.assertEqual(self.solution.backtrack(unreachable_id), ())

    def test_b6_6_equal_time_tie_chooses_lowest_action_index(self) -> None:
        graph, start_id, expected_successor = _hand_written_graph(tied=True)
        solution = solve_time_optimal(graph)
        self.assertAlmostEqual(solution.value_s[start_id], 4.0, places=12)
        self.assertEqual(solution.policy_action_index[start_id], 1)
        self.assertEqual(solution.policy_successor[start_id], expected_successor)

    def test_runtime_metrics_are_returned_programmatically(self) -> None:
        run = run_time_optimal_problem(
            self.grid,
            self.terrain,
            self.goal,
            self.start,
        )
        self.assertIsInstance(run.metrics, SolverMetrics)
        self.assertIsInstance(run.metrics.timing, SolverTiming)
        self.assertEqual(run.metrics.state_count, self.graph.statistics.state_count)
        self.assertEqual(run.metrics.edge_count, self.graph.statistics.valid_edge_count)
        self.assertIsNotNone(run.path)
        for duration in (
            run.metrics.timing.graph_build_s,
            run.metrics.timing.solve_s,
            run.metrics.timing.backtrack_s,
        ):
            self.assertGreaterEqual(duration, 0.0)

    def test_stage6_layer_has_no_later_objective_or_game_dependency(self) -> None:
        directory = Path(__file__).resolve().parent
        imported_modules: set[str] = set()
        for filename in (
            "bellman_objectives.py",
            "bellman_solver.py",
            "solver_metrics.py",
        ):
            tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "hazard",
            "detection",
            "candidate_energy",
            "switching_candidates",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
        ):
            self.assertFalse(any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for module in imported_modules
            ))


if __name__ == "__main__":
    unittest.main()
