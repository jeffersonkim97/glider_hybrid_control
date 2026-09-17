"""Dense-oracle equivalence tests for goal-backward sparse reachability."""

from __future__ import annotations

import unittest

import numpy as np

from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from map_geometry import MapBounds
from scenario import Point3D
from additive_bellman import descendant_mask
from attacker_best_response import _shared_solution
from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    GlideDetectionHazardModel,
)
from energy_model import DEFAULT_PHYSICAL_SCALE
from sparse_additive import solve_sparse_additive_bellman
from sparse_reachability import (
    build_forward_descendant_mask,
    build_goal_backward_reachable_graph,
)
from terrain_catalog import build_terrain


class SparseReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.terrain = build_terrain("centered_cube")
        cls.goal = Point3D(4.0, 0.0, 0.0)
        cls.grid = BellmanStateGrid(
            MapBounds(-4.0, 4.0, -3.0, 3.0),
            maximum_altitude_map=1.0,
        )
        cls.dense_graph = build_bellman_graph(
            cls.grid, cls.terrain, cls.goal,
        )
        cls.dense = solve_unit_cost_reachability(cls.dense_graph)
        cls.sparse = build_goal_backward_reachable_graph(
            cls.grid, cls.terrain, cls.goal,
        )

    def test_reachable_mask_and_unit_cost_match_dense_oracle(self) -> None:
        np.testing.assert_array_equal(
            self.sparse.unit_reachability.goal_reachable,
            self.dense.goal_reachable,
        )
        reachable = self.dense.goal_reachable
        np.testing.assert_allclose(
            self.sparse.unit_reachability.value[reachable],
            self.dense.value[reachable],
            rtol=0.0,
            atol=0.0,
        )

    def test_every_lazy_reachable_row_matches_dense_filtered_row(self) -> None:
        mask = self.dense.goal_reachable
        for state_id in np.flatnonzero(mask):
            expected = tuple(
                edge.target_id
                for edge in self.dense_graph.adjacency[int(state_id)]
                if mask[edge.target_id]
            )
            actual = tuple(
                edge.target_id
                for edge in self.sparse.graph.adjacency[int(state_id)]
            )
            self.assertEqual(actual, expected, int(state_id))

    def test_sparse_graph_materializes_only_goal_reachable_states(self) -> None:
        metrics = self.sparse.metrics
        self.assertEqual(
            metrics.reachable_state_count,
            self.dense.goal_reachable_state_count,
        )
        self.assertLess(metrics.reachable_state_count, metrics.cartesian_state_count)
        self.assertEqual(
            len(self.sparse.graph.topological_sort()),
            metrics.reachable_state_count,
        )

    def test_vectorized_forward_corridor_matches_graph_search(self) -> None:
        reachable_ids = np.flatnonzero(self.sparse.graph.node_mask)
        starts = tuple(map(int, reachable_ids[-5:]))
        expected = descendant_mask(self.sparse.graph, starts)
        actual = build_forward_descendant_mask(self.sparse.graph, starts)
        np.testing.assert_array_equal(actual, expected)

    def test_vectorized_additive_values_match_dense_solver(self) -> None:
        reachable_ids = np.flatnonzero(self.sparse.graph.node_mask)
        maximum_altitude = max(
            self.grid.decode(int(state_id)).altitude_index
            for state_id in reachable_ids
        )
        starts = tuple(
            int(state_id) for state_id in reachable_ids
            if self.grid.decode(int(state_id)).altitude_index == maximum_altitude
        )[:3]
        hazard_field = GlideDetectionHazardModel(
            self.terrain, Point3D(5.0, 0.0, 0.0),
        )
        dense_solution, _, _, _ = _shared_solution(
            self.dense_graph,
            {0: type("Connections", (), {"feasible_proposals": tuple(
                type("Proposal", (), {
                    "target_state_id": state_id,
                    "bellman_reachable": True,
                })()
                for state_id in starts
            )})()},
            hazard_field,
            objective_parameters=DEFAULT_ATTACKER_HAZARD_TIME,
            physical_scale=DEFAULT_PHYSICAL_SCALE,
            quadrature_resolution=4,
        )
        sparse_run = solve_sparse_additive_bellman(
            self.sparse.graph,
            starts,
            hazard_field,
            objective_parameters=DEFAULT_ATTACKER_HAZARD_TIME,
            quadrature_resolution=4,
        )
        mask = sparse_run.solution.goal_reachable
        np.testing.assert_array_equal(
            mask, dense_solution.goal_reachable,
        )
        np.testing.assert_allclose(
            sparse_run.solution.value[mask],
            dense_solution.value[mask],
            rtol=1.0e-13,
            atol=1.0e-14,
        )
        np.testing.assert_array_equal(
            sparse_run.solution.policy_successor[mask],
            dense_solution.policy_successor[mask],
        )

    def test_optional_batch_terrain_queries_match_scalar_contract(self) -> None:
        rng = np.random.default_rng(13)
        points = rng.uniform(
            low=(-5.0, -4.0, 0.0), high=(5.0, 4.0, 5.0), size=(128, 3),
        )
        starts = rng.uniform(
            low=(-5.0, -4.0, 0.0), high=(5.0, 4.0, 5.0), size=(128, 3),
        )
        ends = rng.uniform(
            low=(-5.0, -4.0, 0.0), high=(5.0, 4.0, 5.0), size=(128, 3),
        )
        np.testing.assert_array_equal(
            self.terrain.contains_solid_many(points),
            [self.terrain.contains_solid(point) for point in points],
        )
        np.testing.assert_array_equal(
            self.terrain.segments_intersect_solid_many(starts, ends),
            [
                self.terrain.segment_intersects_solid(start, end)
                for start, end in zip(starts, ends)
            ],
        )


if __name__ == "__main__":
    unittest.main()
