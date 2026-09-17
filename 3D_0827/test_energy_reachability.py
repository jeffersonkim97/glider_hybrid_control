"""Regression tests for the modular Stage-6 energy prototype."""

from __future__ import annotations

import unittest

import numpy as np

from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    SwitchingState,
)
from glide_reachability import BoundedTurnGlideModel
from los_explorer_gui import compute_energy_case


def _switching_state(*, heading_rad: float, altitude_m: float = 100.0) -> SwitchingState:
    speed = DEFAULT_GLIDER.powered_speed_mps
    position_m = np.array([0.0, 0.0, altitude_m], dtype=float)
    velocity = speed * np.array(
        [np.cos(heading_rad), np.sin(heading_rad), 0.0], dtype=float
    )
    energy = DEFAULT_GLIDER.mass_kg * (
        DEFAULT_GLIDER.gravity_mps2 * altitude_m + 0.5 * speed**2
    )
    return SwitchingState(
        position_map=position_m / DEFAULT_PHYSICAL_SCALE.meters_per_map_unit,
        position_m=position_m,
        velocity_mps=velocity,
        powered_path_length_m=100.0,
        flight_path_angle_rad=0.0,
        heading_rad=heading_rad,
        total_mechanical_energy_j=energy,
        powered_feasible=True,
    )


class EnergyReachabilityTests(unittest.TestCase):
    def test_goal_tolerance_is_physical_25_meters(self) -> None:
        model = BoundedTurnGlideModel(DEFAULT_GLIDER)
        result = model.evaluate(
            _switching_state(heading_rad=0.0, altitude_m=0.0),
            np.array([24.9, 0.0, 0.0], dtype=float),
        )
        self.assertTrue(result.reachable)
        self.assertEqual(result.minimum_glide_path_m, 0.0)

    def test_switch_heading_changes_required_glide_path(self) -> None:
        model = BoundedTurnGlideModel(DEFAULT_GLIDER)
        goal = np.array([500.0, 0.0, 0.0], dtype=float)
        toward = model.evaluate(_switching_state(heading_rad=0.0), goal)
        away = model.evaluate(_switching_state(heading_rad=np.pi), goal)
        self.assertLess(toward.minimum_glide_path_m, away.minimum_glide_path_m)

    def test_centered_case_contains_both_energy_classes(self) -> None:
        result = compute_energy_case(
            "centered_cube",
            5.0,
            0.0,
            probe_grid_size=25,
            radial_section_count=7,
        )
        surface = result.energy_surface
        contour = result.los_result.tangent_contour
        los_surface = result.los_result.los_surface
        self.assertGreater(surface.reachable_face_count, 0)
        self.assertGreater(surface.unreachable_face_count, 0)
        self.assertEqual(len(result.los_result.visualization_rays.rays), 10)
        self.assertFalse(contour.closed)
        self.assertGreater(contour.discarded_ground_candidate_count, 0)
        self.assertGreater(
            float(np.min(contour.tangent_points[:, 2])),
            result.los_result.terrain_map.ground_z + 1.0e-6,
        )
        self.assertEqual(los_surface.panel_count, len(contour.rays) - 1)
        expected_open_mesh_triangles = (2 * 7 - 3) * (surface.contour_count - 1)
        self.assertEqual(len(surface.triangles), expected_open_mesh_triangles)
        self.assertGreaterEqual(
            float(np.min(surface.reachable_region.height_margin_m)),
            -1.0e-8,
        )
        finite_red_margin = surface.unreachable_region.height_margin_m[
            np.isfinite(surface.unreachable_region.height_margin_m)
        ]
        self.assertLessEqual(float(np.max(finite_red_margin)), 1.0e-8)

    def test_degraded_glide_and_turn_penalty_parameters(self) -> None:
        self.assertEqual(DEFAULT_GLIDER.best_glide_ratio, 10.0)
        self.assertAlmostEqual(DEFAULT_GLIDER.turn_glide_ratio, 7.5)
        self.assertEqual(DEFAULT_GLIDER.switch_energy_loss_height_m, 10.0)


if __name__ == "__main__":
    unittest.main()
