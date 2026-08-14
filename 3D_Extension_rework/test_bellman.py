"""Regression tests for exact physical Bellman cost-to-go and projection."""

from __future__ import annotations

import unittest

import numpy as np

from .bellman import build_cost_to_go_bundle, _signed_heading_change
from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d


class BellmanCostToGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configuration = build_configuration()
        cls.geometry = build_geometry(cls.configuration)
        cls.detection = build_symbolic_detection_bundle(
            cls.configuration, cls.geometry,
        )
        cls.stage = construct_stage_cost_6d(
            cls.configuration, cls.geometry, cls.detection,
        )
        cls.bundle = build_cost_to_go_bundle(
            cls.configuration, cls.geometry, cls.detection, cls.stage,
        )

    def test_complete_bundle_passes(self) -> None:
        self.assertTrue(self.bundle["status"]["success"])
        self.assertTrue(self.bundle["graph"]["validation"]["passed"])
        self.assertTrue(self.bundle["policy"]["validation"]["passed"])
        self.assertTrue(self.bundle["projection"]["validation"]["passed"])

    def test_physical_edges_land_exactly_on_grid_nodes(self) -> None:
        graph = self.bundle["graph"]
        grids = graph["grids"]
        spacing = np.array([
            grids["x"][1] - grids["x"][0],
            grids["y"][1] - grids["y"][0],
            grids["h"][1] - grids["h"][0],
        ])
        for action in graph["actions"]:
            expected = spacing * np.array([
                action["forward_cells"], action["lateral_cells"],
                -action["descent_cells"],
            ])
            np.testing.assert_allclose(action["edge_m"], expected)
        self.assertFalse(graph["metadata"]["endpoint_snapping"])

    def test_selected_policy_satisfies_bellman_equation(self) -> None:
        graph = self.bundle["graph"]
        policy = self.bundle["policy"]
        value = policy["value_heading_state"]
        finite = np.argwhere(
            np.isfinite(value) & ~policy["goal_mask"][..., None]
        )
        sample_indices = np.linspace(0, finite.shape[0] - 1, 64).astype(int)
        maximum_residual = 0.0
        for x_index, y_index, h_index, heading_index in finite[sample_indices]:
            action_index = int(policy["policy_action_index"][
                x_index, y_index, h_index, heading_index
            ])
            action = graph["actions"][action_index]
            self.assertTrue(graph["valid"][x_index, y_index, h_index, action_index])
            allowed_change = np.deg2rad(
                self.configuration["vehicle"]["max_turn_rate_deg_s"]
            ) * action["duration_s"]
            actual_change = abs(float(_signed_heading_change(
                action["heading_rad"],
                policy["heading_states"][heading_index],
            )))
            self.assertLessEqual(actual_change, allowed_change + 1.0e-12)
            downstream = 0.0
            if not graph["terminal"][x_index, y_index, h_index, action_index]:
                downstream = value[
                    x_index + action["forward_cells"],
                    y_index + action["lateral_cells"],
                    h_index - action["descent_cells"],
                    action["heading_state_index"],
                ]
            reconstructed = (
                graph["cost"][x_index, y_index, h_index, action_index]
                + downstream
            )
            maximum_residual = max(
                maximum_residual,
                abs(float(value[x_index, y_index, h_index, heading_index])
                    - float(reconstructed)),
            )
        self.assertLessEqual(maximum_residual, 1.0e-12)

    def test_projection_is_minimum_value_not_local_j6d(self) -> None:
        policy = self.bundle["policy"]
        projection = self.bundle["projection"]
        direct = np.min(policy["value_heading_state"], axis=3)
        np.testing.assert_array_equal(projection["projected_cost_to_go"], direct)
        self.assertTrue(projection["metadata"]["is_cost_to_go"])
        self.assertFalse(projection["metadata"]["local_stage_cost_projection"])

    def test_switching_and_trajectory_are_not_prematurely_computed(self) -> None:
        self.assertFalse(self.bundle["metadata"]["switching_candidates_evaluated"])
        self.assertFalse(self.bundle["metadata"]["trajectory_extracted"])


if __name__ == "__main__":
    unittest.main()
