"""Fast contract tests for the modular 3D Defender outer search."""

from __future__ import annotations

import unittest

import numpy as np

from .configuration import build_configuration
from .geometry import build_geometry, terrain_height
from .stackelberg import configuration_for_sensor, hierarchical_candidate_search


class StackelbergOuterSearchContractTests(unittest.TestCase):
    def test_configured_sensor_region_and_terrain_height_rule(self) -> None:
        base = build_configuration()
        self.assertEqual(base["defender_search"]["x_bounds_m"], (1750.0, 2500.0))
        self.assertEqual(base["defender_search"]["y_bounds_m"], (0.0, 1000.0))
        nested = configuration_for_sensor(base, (2125.0, 250.0))
        geometry = build_geometry(nested)
        sensor = geometry["sensor_position"]
        self.assertEqual(tuple(sensor[:2]), (2125.0, 250.0))
        self.assertAlmostEqual(
            sensor[2], float(terrain_height(
                geometry["terrain_model"], sensor[0], sensor[1],
            )), places=12,
        )

    def test_hierarchical_search_is_two_dimensional_and_modular(self) -> None:
        configuration = build_configuration()
        options = dict(configuration["defender_search"])
        options.update({
            "coarse_x_count": 3, "coarse_y_count": 3,
            "refinement_levels": 1,
            "use_y_reflection_symmetry": True,
            "verify_y_reflection_symmetry": True,
        })

        def evaluate(x: float, y: float):
            objective = 1.0 - ((x - 2125.0) / 750.0) ** 2 - ((y - 500.0) / 1000.0) ** 2
            return {
                "sensor_position_m": [x, y, 0.0],
                "defender_objective": objective,
                "switching_point_m": [900.0, y, 300.0],
                "feasible": True,
            }

        result = hierarchical_candidate_search(evaluate, options)
        np.testing.assert_allclose(
            result["best_summary"]["sensor_position_m"][:2], [2125.0, 500.0],
        )
        self.assertTrue(result["metadata"]["symmetry_verified"])
        self.assertEqual(len(result["levels"]), 2)

    def test_out_of_region_sensor_is_rejected(self) -> None:
        configuration = build_configuration()
        with self.assertRaises(ValueError):
            configuration_for_sensor(configuration, (1700.0, 500.0))


if __name__ == "__main__":
    unittest.main()
