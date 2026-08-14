"""Tests for backward-compatible elliptical Gaussian terrain."""

from __future__ import annotations

import unittest
from copy import deepcopy

import numpy as np

from p1b_3DExtension.experiment_extreme_ridge_fine import (
    build_fine_configuration,
)
from p1b_3DExtension.geometry import _sum_of_hills, build_geometry_bundle


class AnisotropicTerrainTests(unittest.TestCase):
    def test_isotropic_width_remains_backward_compatible(self) -> None:
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 2.0])
        hill = ({
            "x_ridge": 0.0,
            "y_ridge": 0.0,
            "h_ridge": 10.0,
            "width": 2.0,
        },)
        terrain = _sum_of_hills(x, y, hill)
        self.assertAlmostEqual(terrain[1, 0], 10.0 * np.exp(-0.5))
        self.assertAlmostEqual(terrain[0, 1], 10.0 * np.exp(-0.5))

    def test_width_x_and_width_y_create_elliptical_ridge(self) -> None:
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 2.0])
        hill = ({
            "x_ridge": 0.0,
            "y_ridge": 0.0,
            "h_ridge": 10.0,
            "width_x": 1.0,
            "width_y": 2.0,
        },)
        terrain = _sum_of_hills(x, y, hill)
        self.assertAlmostEqual(terrain[1, 0], 10.0 * np.exp(-2.0))
        self.assertAlmostEqual(terrain[0, 1], 10.0 * np.exp(-0.5))
        self.assertGreater(terrain[0, 1], terrain[1, 0])

    def test_continuous_off_grid_sensor_uses_exact_position_validation(self) -> None:
        configuration = deepcopy(build_fine_configuration())
        sensor = configuration["primary_result"]["sensor_config"]
        sensor["default_x_sensor"] = 1200.0
        sensor["default_y_sensor"] = 0.0

        geometry = build_geometry_bundle(configuration)

        self.assertTrue(geometry["status"]["success"])
        self.assertFalse(
            geometry["validation"]["metrics"]["sensor_is_grid_aligned"]
        )
        self.assertTrue(
            geometry["validation"]["checks"]["sensor_own_cell_visible"]
        )

    def test_grid_aligned_sensor_retains_own_cell_validation(self) -> None:
        geometry = build_geometry_bundle(build_fine_configuration())

        self.assertTrue(geometry["status"]["success"])
        self.assertTrue(
            geometry["validation"]["metrics"]["sensor_is_grid_aligned"]
        )
        self.assertTrue(
            geometry["validation"]["checks"]["sensor_own_cell_visible"]
        )


if __name__ == "__main__":
    unittest.main()
