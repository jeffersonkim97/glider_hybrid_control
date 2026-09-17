"""Stage-4 Part-A powered-flight and manual energy checks."""

from __future__ import annotations

from math import asin, atan2
from pathlib import Path
import unittest

import numpy as np

from candidate_energy import evaluate_switching_candidate
from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    StraightPoweredPhaseModel,
)
from los_explorer_gui import compute_los_case
from switching_candidates import generate_switching_candidates


SELECTED_CANDIDATE_ID = 54
BLOCKED_CANDIDATE_ID = 6


class PoweredPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.los_result = compute_los_case("centered_cube", 5.0, 0.0)
        cls.candidates = generate_switching_candidates(
            cls.los_result.tangent_contour,
        )
        cls.powered_model = StraightPoweredPhaseModel()

    def test_p1_selected_candidate_has_clear_powered_segment(self) -> None:
        candidate = self.candidates[SELECTED_CANDIDATE_ID]
        evaluation = evaluate_switching_candidate(
            candidate,
            self.los_result.tangent_contour,
            self.los_result.terrain_map,
            self.los_result.mission_points,
        )
        self.assertEqual(candidate.candidate_id, SELECTED_CANDIDATE_ID)
        self.assertTrue(evaluation.powered_feasible)
        self.assertIsNone(evaluation.switching_state.infeasibility_reason)

    def test_p2_blocked_candidate_reports_terrain_collision(self) -> None:
        candidate = self.candidates[BLOCKED_CANDIDATE_ID]
        evaluation = evaluate_switching_candidate(
            candidate,
            self.los_result.tangent_contour,
            self.los_result.terrain_map,
            self.los_result.mission_points,
        )
        self.assertFalse(evaluation.powered_feasible)
        self.assertFalse(evaluation.reachable)
        self.assertEqual(
            evaluation.infeasibility_reason,
            "powered segment intersects terrain",
        )

    def test_p3_switching_point_equal_to_start_is_infeasible(self) -> None:
        mission = self.los_result.mission_points
        state = self.powered_model.state_at(
            mission.start.as_array(),
            mission,
            self.los_result.terrain_map,
        )
        self.assertFalse(state.powered_feasible)
        self.assertEqual(
            state.infeasibility_reason,
            "switching point coincides with start",
        )

    def test_p4_vertical_arrival_has_no_horizontal_heading(self) -> None:
        mission = self.los_result.mission_points
        vertical_switch = mission.start.as_array() + np.array([0.0, 0.0, 1.0])
        state = self.powered_model.state_at(
            vertical_switch,
            mission,
            self.los_result.terrain_map,
        )
        self.assertFalse(state.powered_feasible)
        self.assertEqual(
            state.infeasibility_reason,
            "powered arrival has no horizontal heading",
        )
        self.assertAlmostEqual(state.flight_path_angle_rad, np.pi / 2.0)

    def test_p5_selected_candidate_matches_manual_kinematics_and_energy(self) -> None:
        candidate = self.candidates[SELECTED_CANDIDATE_ID]
        mission = self.los_result.mission_points
        state = self.powered_model.state_at(
            candidate.position_map,
            mission,
            self.los_result.terrain_map,
        )
        displacement_map = candidate.position_map - mission.start.as_array()
        distance_map = float(np.linalg.norm(displacement_map))
        direction = displacement_map / distance_map
        expected_velocity = DEFAULT_GLIDER.powered_speed_mps * direction
        expected_heading = atan2(float(direction[1]), float(direction[0]))
        expected_flight_path_angle = asin(float(direction[2]))
        altitude_m = candidate.position_map[2] * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
        expected_energy = DEFAULT_GLIDER.mass_kg * (
            DEFAULT_GLIDER.gravity_mps2 * altitude_m
            + 0.5 * DEFAULT_GLIDER.powered_speed_mps**2
        )

        self.assertAlmostEqual(
            state.powered_path_length_m,
            distance_map * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit,
            places=10,
        )
        np.testing.assert_allclose(
            state.velocity_mps, expected_velocity, rtol=0.0, atol=1.0e-12,
        )
        self.assertAlmostEqual(state.heading_rad, expected_heading, places=14)
        self.assertAlmostEqual(
            state.flight_path_angle_rad,
            expected_flight_path_angle,
            places=14,
        )
        self.assertAlmostEqual(
            state.total_mechanical_energy_j,
            expected_energy,
            places=8,
        )

    def test_p6_physical_scale_is_exactly_100_metres_per_map_unit(self) -> None:
        self.assertEqual(DEFAULT_PHYSICAL_SCALE.meters_per_map_unit, 100.0)
        self.assertEqual(DEFAULT_PHYSICAL_SCALE.distance_m(1.0), 100.0)
        np.testing.assert_array_equal(
            DEFAULT_PHYSICAL_SCALE.position_m(np.array([1.0, -2.0, 3.0])),
            np.array([100.0, -200.0, 300.0]),
        )

    def test_required_vehicle_constants_are_unchanged(self) -> None:
        self.assertAlmostEqual(DEFAULT_GLIDER.powered_speed_mps, 80.0 / 3.6)
        self.assertAlmostEqual(DEFAULT_GLIDER.best_glide_speed_mps, 80.0 / 3.6)
        self.assertEqual(DEFAULT_GLIDER.best_glide_ratio, 10.0)
        self.assertEqual(DEFAULT_GLIDER.maximum_bank_deg, 30.0)
        self.assertEqual(DEFAULT_GLIDER.goal_tolerance_m, 25.0)
        self.assertEqual(DEFAULT_GLIDER.switch_energy_loss_height_m, 10.0)

    def test_p7_powered_phase_uses_generic_terrain_collision_query(self) -> None:
        source = (
            Path(__file__).resolve().parent / "energy_model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("terrain_map.segment_intersects_solid", source)
        self.assertNotIn("obstacle_boxes", source)
        self.assertNotIn("BoxObstacle", source)


if __name__ == "__main__":
    unittest.main()
