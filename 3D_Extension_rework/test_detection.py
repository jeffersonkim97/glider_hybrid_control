"""Regression tests for the 3D symbolic detection extension."""

from __future__ import annotations

import unittest

import casadi as ca
import numpy as np

from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry


def outputs(function: ca.Function, *arguments: float) -> list[float]:
    values = function(*arguments)
    result = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in result]


class DetectionStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configuration = build_configuration()
        cls.geometry = build_geometry(cls.configuration)
        cls.bundle = build_symbolic_detection_bundle(
            cls.configuration, cls.geometry,
        )

    def test_bundle_validation_and_state_contract(self) -> None:
        self.assertTrue(self.bundle["status"]["success"])
        self.assertTrue(self.bundle["validation"]["passed"])
        self.assertEqual(
            self.bundle["metadata"]["state_axis_order"],
            ("x", "y", "h", "v", "gamma", "psi"),
        )
        self.assertEqual(
            self.bundle["metadata"]["projection_role"],
            "visualization_only_not_bellman_input",
        )

    def test_3d_range_and_floor(self) -> None:
        function = self.bundle["functions"]["range"]
        sensor = self.geometry["sensor_position"]
        point = sensor + np.array([-30.0, 40.0, 12.0])
        result = outputs(function, *point, *sensor)
        expected_delta = sensor - point
        self.assertTrue(np.allclose(result[:3], expected_delta))
        self.assertAlmostEqual(result[3], 50.0)
        self.assertAlmostEqual(result[4], np.sqrt(50.0**2 + 12.0**2))
        self.assertEqual(
            outputs(function, *sensor, *sensor)[-1],
            self.configuration["detection"]["range_floor_m"],
        )

    def test_heading_changes_radial_velocity_by_3d_dot_product(self) -> None:
        function = self.bundle["functions"]["glide_detection_components"]
        sensor = self.geometry["sensor_position"]
        point = np.array([2000.0, 300.0, 150.0])
        speed = 18.0
        gamma = np.deg2rad(-12.0)
        for heading in (np.deg2rad(-80.0), 0.0, np.deg2rad(65.0)):
            result = outputs(
                function, *point, speed, gamma, heading, *sensor,
            )
            velocity = speed * np.array([
                np.cos(gamma) * np.cos(heading),
                np.cos(gamma) * np.sin(heading),
                np.sin(gamma),
            ])
            line_of_sight = sensor - point
            expected = np.dot(velocity, line_of_sight) / np.linalg.norm(line_of_sight)
            self.assertAlmostEqual(result[6], expected)

    def test_center_plane_reduces_to_completed_2d_formula(self) -> None:
        function = self.bundle["functions"]["glide_detection_components"]
        sensor = self.geometry["sensor_position"]
        detection = self.configuration["detection"]
        point = np.array([1800.0, sensor[1], 250.0])
        speed = 18.0
        gamma = np.deg2rad(-10.0)
        result = outputs(function, *point, speed, gamma, 0.0, *sensor)
        horizontal_range = sensor[0] - point[0]
        vertical_range = sensor[2] - point[2]
        sensor_range = max(
            np.hypot(horizontal_range, vertical_range),
            detection["range_floor_m"],
        )
        los_angle = np.arctan2(vertical_range, horizontal_range)
        expected_cosine_aspect = np.cos(gamma - los_angle)
        expected_rcs = detection["rcs_min"] + (
            detection["rcs_max"] - detection["rcs_min"]
        ) * expected_cosine_aspect**2
        expected_radial = speed * (
            np.cos(gamma) * horizontal_range
            + np.sin(gamma) * vertical_range
        ) / sensor_range
        self.assertAlmostEqual(result[3], expected_cosine_aspect)
        self.assertAlmostEqual(result[4], expected_rcs)
        self.assertAlmostEqual(result[6], expected_radial)

    def test_los_interpolant_reproduces_geometry_boundary_on_grid(self) -> None:
        function = self.bundle["functions"]["los"]
        x_grid = self.geometry["x_grid"]
        y_grid = self.geometry["y_grid"]
        for x_index in range(0, x_grid.size, 13):
            for y_index in range(0, y_grid.size, 9):
                boundary = outputs(
                    function, x_grid[x_index], y_grid[y_index], 0.0,
                )[0]
                self.assertAlmostEqual(
                    boundary,
                    self.geometry["los_boundary_height"][x_index, y_index],
                    places=10,
                )

    def test_los_surface_gates_glide_but_not_powered_detection(self) -> None:
        los_function = self.bundle["functions"]["los"]
        glide_function = self.bundle["functions"]["glide_detection_components"]
        powered_function = self.bundle["functions"]["powered_detection_components"]
        sensor = self.geometry["sensor_position"]
        index = np.argwhere(self.geometry["non_visible_airspace_mask"])[0]
        point = np.array([
            self.geometry["x_grid"][index[0]],
            self.geometry["y_grid"][index[1]],
            self.geometry["h_grid"][index[2]],
        ])
        self.assertEqual(outputs(los_function, *point)[3], 1.0)
        glide = outputs(
            glide_function, *point, 18.0, np.deg2rad(-10.0), 0.0, *sensor,
        )
        powered = outputs(powered_function, *point, 21.0, *sensor)
        self.assertEqual(glide[-1], 0.0)
        self.assertGreater(powered[-1], 0.0)

    def test_mission_hazard_and_pod_are_unchanged(self) -> None:
        mission = outputs(
            self.bundle["functions"]["mission_detection"], 0.2, 0.3,
        )
        self.assertAlmostEqual(mission[0], 0.5)
        self.assertAlmostEqual(mission[-1], 1.0 - np.exp(-0.5))


if __name__ == "__main__":
    unittest.main()
