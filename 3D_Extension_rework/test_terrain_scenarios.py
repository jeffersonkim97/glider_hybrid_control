"""Terrain-catalog and multi-Gaussian geometry regression tests."""

from __future__ import annotations

import unittest

import numpy as np

from .geometry import build_terrain_model, terrain_height
from .terrain_scenarios import build_scenario_configuration


class TerrainScenarioTests(unittest.TestCase):
    def test_two_hill_is_sum_of_two_isotropic_gaussians(self) -> None:
        configuration = build_scenario_configuration("two_hill")
        model = build_terrain_model(configuration)
        expected = 100.0 + 50.0 * np.exp(-0.5 * 10.0**2)
        self.assertAlmostEqual(float(terrain_height(model, 1000.0, 500.0)), expected)
        self.assertEqual(len(model.hills), 2)

    def test_goal_and_state_grid_align_for_multiterrain_cases(self) -> None:
        for scenario_id in ("two_hill", "goal_in_valley"):
            configuration = build_scenario_configuration(scenario_id)
            environment = configuration["environment"]
            x_grid = np.linspace(
                *environment["x_bounds_m"], configuration["state_grid"]["x_count"],
            )
            self.assertLessEqual(
                float(np.min(np.abs(x_grid - environment["goal_xy_m"][0]))), 1.0e-10,
            )

    def test_symmetric_cases_enable_only_valid_reflection(self) -> None:
        self.assertTrue(build_scenario_configuration("two_hill")["defender_search"]["use_y_reflection_symmetry"])
        self.assertFalse(build_scenario_configuration("asymmetric_single_hill")["defender_search"]["use_y_reflection_symmetry"])


if __name__ == "__main__":
    unittest.main()
