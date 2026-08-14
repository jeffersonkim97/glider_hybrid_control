"""Regression tests for process-faithful 3D switching-point selection."""

from __future__ import annotations

import math
import unittest

import numpy as np

from .bellman import build_cost_to_go_bundle, _signed_heading_change
from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d
from .switching import (
    _glide_rate,
    _powered_rate,
    generate_switching_surface_seeds,
    select_switching_point,
)
from .trajectory import extract_optimal_trajectory
from .continuous_replay import (
    integrate_action_sequence_3d,
    replay_glide_continuous_3d,
)


class SwitchingPointSelectionTests(unittest.TestCase):
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
        cls.cost_to_go = build_cost_to_go_bundle(
            cls.configuration, cls.geometry, cls.detection, cls.stage,
        )
        cls.result = select_switching_point(
            cls.configuration, cls.geometry, cls.cost_to_go,
        )
        cls.trajectory = extract_optimal_trajectory(
            cls.configuration, cls.geometry, cls.cost_to_go, cls.result,
        )
        cls.replay = replay_glide_continuous_3d(
            cls.configuration, cls.geometry, cls.detection, cls.trajectory,
        )

    def test_switching_seeds_are_on_los_boundary_not_altitude_snapped(self) -> None:
        points, _ = generate_switching_surface_seeds(
            self.configuration, self.geometry, self.cost_to_go["graph"],
        )
        function = self.detection["functions"]["los"]
        boundary = np.array([
            float(function(*point)[0]) for point in points
        ])
        np.testing.assert_allclose(points[:, 2], boundary, atol=1.0e-10, rtol=0.0)
        h_grid = self.cost_to_go["graph"]["grids"]["h"]
        self.assertTrue(np.any(np.min(np.abs(
            points[:, 2, None] - h_grid[None, :]
        ), axis=1) > 1.0e-6))

    def test_numeric_detection_replay_matches_authoritative_functions(self) -> None:
        point = np.array([[900.0, 300.0, 333.3507442]])
        sensor = self.geometry["sensor_position"]
        powered_speed = self.configuration["vehicle"]["powered_speed_mps"]
        powered_expected = float(self.detection["functions"][
            "powered_detection_components"
        ](*point[0], powered_speed, *sensor)[-1])
        self.assertAlmostEqual(
            float(_powered_rate(
                point, powered_speed, self.configuration, self.geometry,
            )[0]), powered_expected, places=13,
        )
        speed, gamma, heading = 16.3, math.radians(-10.0), math.radians(5.0)
        glide_expected = float(self.detection["functions"][
            "glide_detection_components"
        ](*point[0], speed, gamma, heading, *sensor)[-1])
        self.assertAlmostEqual(
            float(_glide_rate(
                point, speed, gamma, heading,
                self.configuration, self.geometry,
            )[0]), glide_expected, places=13,
        )

    def test_selected_candidate_is_exact_feasible_minimum(self) -> None:
        self.assertTrue(self.result["status"]["success"])
        best = self.result["best"]
        self.assertEqual(
            best["mission_cost"],
            min(candidate["mission_cost"] for candidate in self.result["candidates"]),
        )
        self.assertTrue(best["powered"]["certificate"]["los_clear"])
        self.assertTrue(best["connection"]["edge"]["certificate"]["los_clear"])
        self.assertTrue(
            self.result["validation"]["checks"][
                "projected_v3d_not_used_for_selection"
            ]
        )

    def test_exact_incoming_heading_respects_turn_rate(self) -> None:
        best = self.result["best"]
        edge = best["connection"]["edge"]
        heading_change = abs(float(_signed_heading_change(
            edge["heading_rad"], best["powered"]["heading_rad"],
        )))
        allowed = math.radians(
            self.configuration["vehicle"]["max_turn_rate_deg_s"]
        ) * edge["duration_s"]
        self.assertLessEqual(heading_change, allowed + 1.0e-12)

    def test_full_policy_trajectory_reaches_goal_without_nlp(self) -> None:
        self.assertTrue(self.trajectory["status"]["success"])
        checks = self.trajectory["validation"]["checks"]
        self.assertTrue(checks["goal_reached"])
        self.assertTrue(checks["stored_policy_followed"])
        self.assertTrue(checks["endpoint_snapping_absent"])
        self.assertTrue(checks["continuous_nlp_absent"])

    def test_extracted_objective_matches_switching_selection(self) -> None:
        self.assertAlmostEqual(
            self.trajectory["mission"]["cost"],
            self.result["best"]["mission_cost"], places=10,
        )
        self.assertLessEqual(
            self.trajectory["validation"]["metrics"]["downstream_cost_residual"],
            1.0e-10,
        )

    def test_3d_action_integrator_matches_analytic_increment(self) -> None:
        result = integrate_action_sequence_3d(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0]), np.array([0.0]), np.array([np.pi / 2.0]),
            np.array([2.0]), max_steps=1,
        )
        np.testing.assert_allclose(
            result["trajectory"][-1], np.array([1.0, 22.0, 3.0]), atol=1.0e-12,
        )

    def test_continuous_replay_is_unsnapped_and_feasible(self) -> None:
        self.assertTrue(self.replay["status"]["success"], self.replay["status"])
        self.assertTrue(self.replay["feasible"])
        self.assertTrue(self.replay["reached_goal"])
        metrics = self.replay["validation"]["metrics"]
        self.assertLessEqual(metrics["maximum_endpoint_drift_m"], 1.0e-9)
        self.assertLessEqual(metrics["maximum_hazard_edge_residual"], 1.0e-7)
        self.assertLessEqual(abs(metrics["mission_objective_residual"]), 1.0e-7)
        self.assertFalse(self.replay["metadata"]["state_reset"])
        self.assertFalse(self.replay["metadata"]["continuous_nlp_applied"])


if __name__ == "__main__":
    unittest.main()
