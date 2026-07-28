"""Self-validation for continuous_replay_evaluation.py before it is
trusted to gate the plan-item-4 baseline re-run. See
p1b_roadmap_0727.md's "continuous replay" section.
"""
from __future__ import annotations

import math
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.continuous_replay_evaluation import (
    integrate_action_sequence,
    replay_glide_continuous,
)
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stackelberg_solver import evaluate_defender_position

# Same toy hill used by test_discrete_optimality_crosscheck.py (ridge/width
# = 6.67, clears the launch terrain_tolerance) -- reused here so both
# verification tools share one known-good scenario.
_TOY_DOMAIN = {"z_min": 0.0, "z_max": 800.0, "h_min": 0.0, "h_max": 100.0}
_TOY_HILL = {"z_ridge": 400.0, "h_ridge": 50.0, "width": 60.0}


def _build_toy_bundles(project_root: Path):
    cb = deepcopy(build_configuration_bundle(project_root))
    env = cb["primary_result"]["environment_config"]
    env["z_start"] = 0.0
    env["z_goal"] = 700.0
    env["terrain"] = {"z_min": _TOY_DOMAIN["z_min"], "z_max": _TOY_DOMAIN["z_max"], "hills": (_TOY_HILL,)}
    z_min, z_max = _TOY_DOMAIN["z_min"], _TOY_DOMAIN["z_max"]
    h_min, h_max = _TOY_DOMAIN["h_min"], _TOY_DOMAIN["h_max"]
    env["grid"] = {
        "z_min": z_min, "z_max": z_max, "z_count": 33,
        "z_spacing": (z_max - z_min) / 32,
        "h_min": h_min, "h_max": h_max, "h_count": 21,
        "h_spacing": (h_max - h_min) / 20,
        "v_count": 3, "gamma_count": 5,
        "axis_order_4d": ("z", "h", "v", "gamma"),
    }
    env["airspace"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max}
    env["simulation"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max, "max_path_steps": 200}
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = 3
    vehicle["gamma_count"] = 5
    cb["primary_result"]["sensor_config"]["default_z_sensor"] = 600.0
    cb["primary_result"]["defender_config"]["continuous_search_bounds"] = {
        "z_sensor_min": 500.0, "z_sensor_max": 700.0,
    }
    geometry = build_geometry_bundle(cb)
    detection = build_symbolic_detection_bundle(cb, geometry)
    return cb, geometry, detection


class ContinuousReplaySelfValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.cb, self.geometry, self.detection = _build_toy_bundles(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger, self.cb["primary_result"]["logging_utilities"]["logger"],
        )
        self.geom = self.geometry["primary_result"]
        self.functions = self.detection["primary_result"]["functions"]
        self.validation_config = self.cb["primary_result"]["validation_config"]
        self.vehicle = self.cb["primary_result"]["vehicle_config"]

    def _replay(self, switching_point, speeds, gammas, **overrides):
        kwargs = dict(
            switching_point=np.asarray(switching_point),
            speed_profile=np.asarray(speeds),
            gamma_profile=np.asarray(gammas),
            time_step=self.vehicle["time_step"],
            goal_position=np.asarray(self.geom["goal_position"]),
            goal_radius=self.validation_config["goal_radius"],
            terrain_model=self.geom["terrain_model"],
            los_geometry=self.geom["los_geometry"],
            sensor_position=np.asarray(self.geom["sensor_position"]),
            glide_detection_rate_function=self.functions["glide_detection_components"],
            terrain_tolerance=self.validation_config["terrain_tolerance"],
            segment_check_count=self.cb["primary_result"]["bellman_config"]["search_options"]["segment_check_count"],
            max_steps=1000,
        )
        kwargs.update(overrides)
        return replay_glide_continuous(**kwargs)

    def test_analytic_straight_line_kinematics_no_early_termination(self) -> None:
        # Start at/past the sensor's own along-track position, where
        # visibility holds unconditionally (`sample_z >= sensor_z`)
        # regardless of terrain -- isolates pure kinematics from the LOS
        # check (covered separately in test_terrain_violation_detected_
        # mid_segment). High altitude, shallow descent, far from goal.
        v, gamma = 15.0, math.radians(-2.0)
        start = (650.0, 95.0)
        steps = 5
        result = integrate_action_sequence(
            np.asarray(start),
            np.full(steps, v),
            np.full(steps, gamma),
            time_step=self.vehicle["time_step"],
            max_steps=1000,
        )
        expected_time = steps * self.vehicle["time_step"]
        self.assertAlmostEqual(result["elapsed_time"], expected_time, places=9)
        expected_z = start[0] + steps * v * math.cos(gamma) * self.vehicle["time_step"]
        expected_h = start[1] + steps * v * math.sin(gamma) * self.vehicle["time_step"]
        final_z, final_h = result["trajectory"][-1]
        self.assertAlmostEqual(final_z, expected_z, places=6)
        self.assertAlmostEqual(final_h, expected_h, places=6)

    def test_hazard_matches_raw_detection_rate_formula(self) -> None:
        v, gamma = 12.0, math.radians(-5.0)
        start = (50.0, 90.0)
        result = self._replay(start, [v], [gamma])
        sensor = self.geom["sensor_position"]
        outputs = self.functions["glide_detection_components"](
            start[0], start[1], v, gamma, float(sensor[0]), float(sensor[1]),
        )
        expected_rate = float(outputs[-1])
        expected_hazard = expected_rate * self.vehicle["time_step"]
        expected_pod = 1.0 - math.exp(-expected_hazard)
        self.assertAlmostEqual(result["continuous_glide_hazard"], expected_hazard, places=9)
        self.assertAlmostEqual(result["continuous_mission_pod_glide_only"], expected_pod, places=9)

    def test_goal_radius_threshold(self) -> None:
        goal = np.asarray(self.geom["goal_position"])
        radius = self.validation_config["goal_radius"]
        # Zero-step replay starting exactly on the boundary: reached.
        on_boundary = goal + np.array([radius - 1.0e-6, 0.0])
        result_in = self._replay(on_boundary, [], [])
        self.assertTrue(result_in["feasible"])
        self.assertTrue(result_in["reached_goal"])
        # Just outside, with no steps to reach it: not reached, infeasible.
        outside = goal + np.array([radius + 5.0, 0.0])
        result_out = self._replay(outside, [], [])
        self.assertFalse(result_out["feasible"])
        self.assertEqual(result_out["violation"], "no_steps_and_not_at_goal")

    def test_terrain_violation_detected_mid_segment(self) -> None:
        # A single large step whose straight-line interpolation dips under
        # the (curved, Gaussian) hill profile partway through, even though
        # both endpoints individually clear the terrain.
        ridge_z = _TOY_HILL["z_ridge"]
        half_width = _TOY_HILL["width"]
        start = (ridge_z - half_width, _TOY_HILL["h_ridge"] * 0.55)
        # One big step straight across the ridge at ~constant altitude
        # (small gamma, high speed) -- the peak (h_ridge=50) exceeds the
        # nearly-flat flight path's altitude at the ridge crest.
        v, gamma = self.vehicle["glide_speed_max"], math.radians(-1.0)
        result = self._replay(start, [v], [gamma], max_steps=1)
        self.assertFalse(result["feasible"])
        self.assertIsNotNone(result["violation"])
        self.assertTrue(str(result["violation"]).startswith("terrain_violation"))

    def test_grid_snapping_drift_is_explicitly_diagnosed(self) -> None:
        # This deliberately coarse toy grid exposes the distinction between
        # Bellman's snapped lattice path and execution of the same actions
        # without a state reset at every node.  The evaluator must report the
        # physical infeasibility and the accumulated state drift; silently
        # treating the lattice path as a continuous trajectory is forbidden.
        evaluation = evaluate_defender_position(600.0, self.cb, "continuous-replay-sanity")
        best = evaluation["primary_result"]["best_found_attacker_response"]
        result = self._replay(
            best["switching_point"], best["speed_profile"], best["gamma_profile"],
            reference_trajectory=best["trajectory"],
        )
        self.assertFalse(result["feasible"])
        self.assertTrue(
            str(result["violation"]).startswith(("los_violation", "terrain_violation"))
        )
        failing_step = result["step_diagnostics"][-1]
        self.assertIn("reference_grid_start", failing_step)
        self.assertGreater(failing_step["start_drift_norm"], 0.0)
        self.assertIsNotNone(failing_step["first_invalid_sample"])

    def test_production_replay_accepts_a_goal_reaching_sequence(self) -> None:
        goal = np.asarray(self.geom["goal_position"])
        start = goal + np.array([-20.0, 1.0])
        speed = 15.0
        gamma = math.atan2(-1.0, 15.0)
        result = self._replay(start, [speed], [gamma])
        self.assertTrue(result["feasible"], result["violation"])
        self.assertTrue(result["reached_goal"])
        self.assertLessEqual(result["goal_miss"], self.validation_config["goal_radius"])

    def test_production_replay_rejects_exhausted_sequence_before_goal(self) -> None:
        # This action is kinematically valid and remains in clear airspace,
        # but it is intentionally too short to reach the goal.  Production
        # replay must preserve strict mission semantics after the low-level
        # kinematics integrator was separated from feasibility evaluation.
        start = (650.0, 95.0)
        result = self._replay(start, [15.0], [math.radians(-2.0)])
        self.assertFalse(result["feasible"])
        self.assertFalse(result["reached_goal"])
        self.assertEqual(
            result["violation"],
            "action_sequence_exhausted_without_reaching_goal",
        )


if __name__ == "__main__":
    unittest.main()
