"""Stage-5 glide-graph geometry, DAG, and unit-reachability tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import unittest

import numpy as np

from bellman_geometry import GlideTransitionModel, wrapped_angle_difference
from bellman_graph import (
    build_bellman_graph,
    independent_reverse_reachability,
    solve_unit_cost_reachability,
)
from bellman_state import BellmanState, BellmanStateGrid, is_goal_terminal
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import MapBounds
from scenario import Point3D
from terrain_catalog import build_terrain


@dataclass(frozen=True)
class FlatTerrainFixture:
    """Minimal obstacle-free terrain query object for graph tests."""

    bounds: MapBounds
    ground_z: float = 0.0

    @property
    def maximum_height(self) -> float:
        return self.ground_z

    def contains_solid(self, point: np.ndarray, *, tolerance: float = 0.0) -> bool:
        values = np.asarray(point, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError("point must contain three finite coordinates")
        return False

    def segment_intersects_solid(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        tolerance: float = 1.0e-9,
    ) -> bool:
        return False


class BellmanGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.small_bounds = MapBounds(0.0, 3.0, -1.0, 1.0)
        cls.small_grid = BellmanStateGrid(
            bounds=cls.small_bounds,
            horizontal_spacing_map=1.0,
            minimum_altitude_map=0.0,
            maximum_altitude_map=0.3,
            altitude_spacing_map=0.1,
            heading_bin_count=8,
        )
        cls.flat_terrain = FlatTerrainFixture(cls.small_bounds)
        cls.goal = Point3D(3.0, 0.0, 0.0)
        cls.small_graph = build_bellman_graph(
            cls.small_grid,
            cls.flat_terrain,
            cls.goal,
        )
        cls.small_solution = solve_unit_cost_reachability(cls.small_graph)

    def test_state_encoding_and_decoding_are_deterministic(self) -> None:
        ids = []
        for state in self.small_grid.iter_states():
            state_id = self.small_grid.encode(state)
            ids.append(state_id)
            self.assertEqual(self.small_grid.decode(state_id), state)
        self.assertEqual(ids, list(range(self.small_grid.state_count)))
        self.assertEqual(self.small_grid.state_count, 4 * 3 * 4 * 8)

    def test_b5_1_empty_terrain_straight_goal_is_reachable(self) -> None:
        start = BellmanState(
            x_index=0,
            y_index=1,
            altitude_index=3,
            heading_bin=0,
        )
        start_id = self.small_grid.encode(start)
        self.assertEqual(self.small_solution.value[start_id], 3.0)
        path = self.small_solution.backtrack(start_id)
        self.assertEqual(len(path), 4)
        self.assertTrue(self.small_graph.terminal_mask[path[-1]])
        positions = [
            self.small_grid.position_map(self.small_grid.decode(state_id))
            for state_id in path
        ]
        np.testing.assert_allclose(
            positions,
            np.array([
                [0.0, 0.0, 0.3],
                [1.0, 0.0, 0.2],
                [2.0, 0.0, 0.1],
                [3.0, 0.0, 0.0],
            ]),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_b5_2_insufficient_altitude_is_unreachable(self) -> None:
        start = BellmanState(0, 1, 2, 0)
        start_id = self.small_grid.encode(start)
        self.assertTrue(np.isposinf(self.small_solution.value[start_id]))
        self.assertEqual(self.small_solution.policy_successor[start_id], -1)
        self.assertEqual(self.small_solution.backtrack(start_id), ())

    def test_b5_3_terrain_crossing_edge_is_absent(self) -> None:
        grid = BellmanStateGrid(
            bounds=MapBounds(-3.0, 3.0, -2.0, 2.0),
            maximum_altitude_map=2.0,
        )
        terrain = build_terrain("centered_cube")
        model = GlideTransitionModel(grid, terrain)
        source = BellmanState(1, 2, 20, 0)  # (-2, 0, 2), on cube side face
        edges, statistics, rejected = model.successors(
            source,
            include_rejected=True,
        )
        self.assertNotIn(0, [edge.target_state.heading_bin for edge in edges])
        self.assertGreater(statistics.rejected_by_terrain, 0)
        self.assertTrue(any(
            record.target_heading_bin == 0 and record.reason == "terrain collision"
            for record in rejected
        ))

    def test_b5_4_turn_constraint_accepts_inside_and_rejects_outside(self) -> None:
        grid = BellmanStateGrid(
            bounds=MapBounds(-2.0, 2.0, -2.0, 2.0),
            maximum_altitude_map=1.0,
        )
        model = GlideTransitionModel(grid, FlatTerrainFixture(grid.bounds))
        source = BellmanState(2, 2, 10, 0)
        edges, statistics, rejected = model.successors(
            source,
            include_rejected=True,
        )
        valid_headings = {edge.target_state.heading_bin for edge in edges}
        self.assertIn(1, valid_headings)  # east -> northeast, 45 degrees
        self.assertNotIn(2, valid_headings)  # east -> north, 90 degrees
        self.assertGreater(statistics.rejected_by_turn, 0)
        self.assertTrue(any(
            record.target_heading_bin == 2 and record.reason == "turn constraint"
            for record in rejected
        ))

    def test_b5_5_goal_tolerance_is_inclusive_and_three_dimensional(self) -> None:
        goal = Point3D(0.0, 0.0, 0.0)
        for distance_m, expected in ((24.9, True), (25.0, True), (25.1, False)):
            with self.subTest(distance_m=distance_m):
                point = np.array([
                    distance_m / DEFAULT_PHYSICAL_SCALE.meters_per_map_unit,
                    0.0,
                    0.0,
                ])
                self.assertEqual(is_goal_terminal(point, goal), expected)
        self.assertTrue(is_goal_terminal(np.array([0.0, 0.0, 0.249]), goal))
        self.assertFalse(is_goal_terminal(np.array([0.0, 0.0, 0.251]), goal))

    def test_b5_6_heading_wrap_uses_smallest_angle(self) -> None:
        self.assertAlmostEqual(
            wrapped_angle_difference(np.pi - 1.0e-6, -np.pi + 1.0e-6),
            2.0e-6,
            places=12,
        )
        self.assertAlmostEqual(wrapped_angle_difference(0.0, 2.0 * np.pi), 0.0)
        self.assertAlmostEqual(
            wrapped_angle_difference(-np.pi / 4.0, 7.0 * np.pi / 4.0),
            0.0,
        )

    def test_l_over_d_and_turn_invariants_hold_for_every_small_edge(self) -> None:
        maximum_turn_rate = DEFAULT_GLIDER.maximum_turn_rate_rad_s
        for source_id in self.small_graph.node_ids:
            for edge in self.small_graph.adjacency[int(source_id)]:
                self.assertLess(
                    edge.target_state.altitude_index,
                    edge.source_state.altitude_index,
                )
                self.assertLessEqual(
                    edge.horizontal_distance_m / edge.altitude_loss_m,
                    DEFAULT_GLIDER.best_glide_ratio + 1.0e-12,
                )
                self.assertLessEqual(
                    edge.heading_change_rad,
                    maximum_turn_rate * edge.duration_s + 1.0e-12,
                )
                self.assertEqual(edge.unit_cost, 1.0)

    def test_b5_7_graph_is_a_dag_and_statistics_are_conservative(self) -> None:
        ordering = self.small_graph.topological_sort()
        self.assertEqual(len(ordering), self.small_graph.statistics.state_count)
        self.assertEqual(set(ordering), set(self.small_graph.node_ids.tolist()))
        stats = self.small_graph.statistics
        self.assertEqual(stats.cartesian_state_count, 384)
        self.assertEqual(stats.state_count, 384)
        self.assertEqual(stats.terrain_excluded_state_count, 0)
        self.assertEqual(stats.terminal_state_count, 24)
        self.assertEqual(
            stats.valid_edge_count
            + stats.rejected_by_bounds
            + stats.rejected_by_terrain
            + stats.rejected_by_turn
            + stats.rejected_by_altitude,
            (stats.state_count - stats.terminal_state_count) * 8,
        )

    def test_b5_8_bellman_reachability_matches_independent_graph_search(self) -> None:
        independent = independent_reverse_reachability(self.small_graph)
        np.testing.assert_array_equal(
            self.small_solution.goal_reachable,
            independent,
        )
        for state_id in self.small_graph.node_ids:
            integer_id = int(state_id)
            if self.small_solution.goal_reachable[integer_id]:
                path = self.small_solution.backtrack(integer_id)
                self.assertTrue(path)
                self.assertTrue(self.small_graph.terminal_mask[path[-1]])
                self.assertLessEqual(len(path), self.small_grid.altitude_count + 1)
                altitude_indices = [
                    self.small_grid.decode(path_id).altitude_index
                    for path_id in path
                ]
                self.assertTrue(all(
                    second < first
                    for first, second in zip(
                        altitude_indices,
                        altitude_indices[1:],
                    )
                ))
            else:
                self.assertTrue(np.isposinf(self.small_solution.value[integer_id]))
                self.assertEqual(self.small_solution.policy_successor[integer_id], -1)

    def test_unit_cost_layer_contains_no_later_stage_dependency(self) -> None:
        directory = Path(__file__).resolve().parent
        imported_modules: set[str] = set()
        combined_source = ""
        for filename in (
            "bellman_state.py",
            "bellman_geometry.py",
            "bellman_graph.py",
        ):
            source = (directory / filename).read_text(encoding="utf-8")
            combined_source += source
            tree = ast.parse(source)
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "candidate_energy",
            "switching_candidates",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "hazard",
            "detection",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(
                    forbidden in module.lower() for module in imported_modules
                ))
        self.assertNotIn("detection_probability", combined_source)


if __name__ == "__main__":
    unittest.main()
