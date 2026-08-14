"""Regression test for LOS-dependent powered acoustic attenuation."""

from __future__ import annotations

import unittest

import numpy as np

from p1b_3DExtension.bellman import generate_switching_point_seeds
from p1b_3DExtension.detection import build_symbolic_detection_bundle
from p1b_3DExtension.experiment_extreme_ridge_fine import (
    build_fine_configuration,
)
from p1b_3DExtension.geometry import build_geometry_bundle, terrain_height
from p1b_3DExtension.phase_logging import close_phase_logger


class AcousticOcclusionTests(unittest.TestCase):
    def test_los_boundary_surface_matches_ray_marched_visibility(self) -> None:
        configuration = build_fine_configuration()
        logger = configuration["primary_result"]["logging_utilities"]["logger"]
        try:
            geometry = build_geometry_bundle(configuration)["primary_result"]
            environment = configuration["primary_result"]["environment_config"]
            h_grid = np.linspace(
                environment["grid"]["h_min"],
                environment["grid"]["h_max"],
                environment["grid"]["h_count"],
            )
            boundary = geometry["los_masks"]["los_boundary_height"]
            predicted_visible = h_grid[None, None, :] >= boundary[:, :, None]
            airspace = ~geometry["los_masks"]["terrain_mask"]
            self.assertTrue(np.array_equal(
                predicted_visible[airspace],
                geometry["los_masks"]["los_mask"][airspace],
            ))

            x_grid = geometry["terrain_arrays"]["x"]
            y_grid = geometry["terrain_arrays"]["y"]
            seeds = generate_switching_point_seeds(
                {"primary_result": geometry}, x_grid, y_grid, h_grid,
                boundary_only=True,
            )
            spacing = float(h_grid[1] - h_grid[0])
            for x, y, h in seeds:
                xi = int(np.argmin(np.abs(x_grid - x)))
                yi = int(np.argmin(np.abs(y_grid - y)))
                self.assertLessEqual(
                    abs(h - boundary[xi, yi]), 0.5 * spacing + 1.0e-12,
                )
        finally:
            close_phase_logger(logger)

    def test_occluded_rate_is_scaled_and_visible_rate_is_full(self) -> None:
        configuration = build_fine_configuration()
        logger = configuration["primary_result"]["logging_utilities"]["logger"]
        try:
            geometry = build_geometry_bundle(configuration)
            detection = build_symbolic_detection_bundle(configuration, geometry)
            functions = detection["primary_result"]["functions"]
            primary = configuration["primary_result"]
            specification = primary["sensor_config"]["detection"]
            speed = float(primary["vehicle_config"]["powered_speed"])
            sensor = geometry["primary_result"]["sensor_position"]

            launch_h = float(terrain_height(
                geometry["primary_result"]["terrain_model"], 0.0, 0.0,
            ))
            points = (
                (0.0, 0.0, launch_h, specification["acoustic_occluded_rate_scale"]),
                (float(sensor[0]), float(sensor[1]), 200.0, 1.0),
            )
            for x, y, h, expected_factor in points:
                outputs = functions["powered_detection_components"](
                    x, y, h, speed, *sensor,
                )
                attenuated_rate = float(outputs[0])
                distance = max(
                    float(np.linalg.norm(np.array([x, y, h]) - sensor)),
                    specification["range_floor"],
                )
                raw_rate = (
                    specification["acoustic_coefficient"]
                    * speed ** specification["acoustic_speed_exponent"]
                    / distance**2
                )
                self.assertAlmostEqual(
                    attenuated_rate / raw_rate, expected_factor, places=10,
                )
        finally:
            close_phase_logger(logger)

    def test_visible_powered_total_rate_includes_radar_and_doppler(self) -> None:
        configuration = build_fine_configuration()
        logger = configuration["primary_result"]["logging_utilities"]["logger"]
        try:
            geometry = build_geometry_bundle(configuration)
            detection = build_symbolic_detection_bundle(configuration, geometry)
            functions = detection["primary_result"]["functions"]
            sensor = geometry["primary_result"]["sensor_position"]
            speed = float(
                configuration["primary_result"]["vehicle_config"]["powered_speed"]
            )
            x, y, h = float(sensor[0]), float(sensor[1]), 200.0
            acoustic = functions["powered_detection_components"](
                x, y, h, speed, *sensor,
            )
            total = functions["powered_total_detection_components"](
                x, y, h, speed, np.deg2rad(8.0), 0.0, *sensor,
            )
            self.assertGreater(float(total[-1]), float(acoustic[-1]))
            self.assertAlmostEqual(
                float(total[-1]),
                float(total[0]) + float(total[1]) + float(total[2]),
                places=12,
            )
        finally:
            close_phase_logger(logger)


if __name__ == "__main__":
    unittest.main()
