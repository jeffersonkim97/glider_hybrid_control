"""Regression tests for the authoritative 6D local stage cost."""

from __future__ import annotations

import unittest

import numpy as np

from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import AXIS_ORDER, construct_stage_cost_6d


def outputs(function, *arguments: float) -> list[float]:
    values = function(*arguments)
    result = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in result]


class StageCost6DTests(unittest.TestCase):
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

    def test_stage_bundle_passes_and_has_six_axes(self) -> None:
        self.assertTrue(self.stage["status"]["success"])
        self.assertTrue(self.stage["validation"]["passed"])
        self.assertEqual(self.stage["grid_metadata"]["axis_order"], AXIS_ORDER)
        expected_shape = (31, 11, 21, 3, 6, 36)
        self.assertEqual(self.stage["j6d"].shape, expected_shape)
        self.assertEqual(
            self.stage["grid_metadata"]["state_count"],
            int(np.prod(expected_shape)),
        )

    def test_phase_feasibility_matches_completed_2d_process(self) -> None:
        masks = self.stage["validity_masks"]
        indices = masks["nested_geometry_indices"]
        nested = np.ix_(indices["x"], indices["y"], indices["h"])
        terrain = self.geometry["terrain_mask"][nested]
        los = self.geometry["los_mask"][nested]
        occluded_air = self.geometry["non_visible_airspace_mask"][nested]
        np.testing.assert_array_equal(
            masks["spatial_glide_valid"], ~terrain & los,
        )
        np.testing.assert_array_equal(
            masks["spatial_powered_valid"], ~terrain & occluded_air,
        )

    def test_finite_and_infinite_costs_follow_masks(self) -> None:
        feasible = self.stage["feasible_mask"]
        powered = self.stage["validity_masks"]["powered_feasible_mask"]
        self.assertTrue(np.all(np.isfinite(self.stage["j6d"][feasible])))
        self.assertTrue(np.all(np.isposinf(self.stage["j6d"][~feasible])))
        self.assertTrue(np.all(np.isfinite(
            self.stage["powered_stage_cost_6d"][powered]
        )))
        self.assertTrue(np.all(np.isposinf(
            self.stage["powered_stage_cost_6d"][~powered]
        )))

    def test_center_plane_state_uses_detection_without_reconstruction(self) -> None:
        grids = self.stage["grids"]
        x_index = int(np.argmin(np.abs(grids["x"] - 1800.0)))
        y_index = int(np.argmin(np.abs(grids["y"] - 500.0)))
        h_index = int(np.argmin(np.abs(grids["h"] - 200.0)))
        control_indices = np.argwhere(
            self.stage["feasible_mask"][x_index, y_index, h_index]
        )
        self.assertGreater(control_indices.shape[0], 0)
        v_index, gamma_index, psi_index = control_indices[0]
        state = (
            grids["x"][x_index], grids["y"][y_index], grids["h"][h_index],
            grids["v"][v_index], grids["gamma"][gamma_index],
            grids["psi"][psi_index],
        )
        glide = outputs(
            self.detection["functions"]["glide_detection_components"],
            *state, *self.geometry["sensor_position"],
        )
        objective = outputs(
            self.detection["functions"]["attacker_objective"],
            0.0,
            glide[-1] * self.configuration["vehicle"]["time_step_s"],
            0.0,
            self.configuration["vehicle"]["time_step_s"],
        )
        index = (
            x_index, y_index, h_index,
            v_index, gamma_index, psi_index,
        )
        self.assertAlmostEqual(
            self.stage["component_maps"]["glide_detection_rate"][index],
            glide[-1],
        )
        self.assertAlmostEqual(self.stage["j6d"][index], objective[-1])

    def test_stage_cost_is_not_mislabeled_as_cost_to_go(self) -> None:
        self.assertFalse(self.stage["metadata"]["is_value_function"])
        self.assertFalse(self.stage["metadata"]["is_cost_to_go"])


if __name__ == "__main__":
    unittest.main()
