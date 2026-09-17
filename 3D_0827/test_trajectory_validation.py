"""Stage-10 independent replay and injected-failure tests R1-R7."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from attacker_best_response import run_attacker_best_response
from detection_hazard import GlideDetectionHazardModel
from energy_model import DEFAULT_GLIDER
from game_types import AttackerInitialCondition, DefenderAction
from scenario import Point3D
from trajectory_validation import (
    DEFAULT_REPLAY_TOLERANCES,
    energy_margin_is_valid,
    goal_is_valid,
    snapshot_selected_trajectory,
    validate_trajectory_replay,
)


class TrajectoryValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attacker_run = run_attacker_best_response(
            DefenderAction(np.array([5.0, 0.0, 0.0])),
            AttackerInitialCondition(
                Point3D(-8.0, 0.0, 0.0), Point3D(8.0, 0.0, 0.0),
            ),
        )
        cls.snapshot = snapshot_selected_trajectory(cls.attacker_run)
        cls.hazard = GlideDetectionHazardModel(
            cls.attacker_run.terrain, cls.attacker_run.mission_points.sensor,
        )
        cls.valid_audit = cls.validate(cls.snapshot)

    @classmethod
    def validate(cls, snapshot, *, parameters=DEFAULT_GLIDER):
        return validate_trajectory_replay(
            snapshot,
            cls.attacker_run.terrain,
            cls.attacker_run.mission_points,
            cls.attacker_run.tangent_contour,
            cls.hazard,
            parameters=parameters,
        )

    def test_r1_known_valid_trajectory_passes_every_check(self) -> None:
        report = self.valid_audit.report
        self.assertTrue(report.passed)
        self.assertTrue(report.terrain_clear)
        self.assertTrue(report.los_phase_valid)
        self.assertTrue(report.turn_constraints_valid)
        self.assertTrue(report.energy_valid)
        self.assertTrue(report.time_consistent)
        self.assertTrue(report.hazard_consistent)
        self.assertTrue(report.goal_valid)
        self.assertEqual(report.max_turn_violation, 0.0)
        self.assertIsNone(report.min_terrain_clearance)
        self.assertLessEqual(report.time_error_s, DEFAULT_REPLAY_TOLERANCES.time_s)
        self.assertLessEqual(report.hazard_error, DEFAULT_REPLAY_TOLERANCES.hazard)

    def test_r2_injected_terrain_collision_is_detected(self) -> None:
        positions = self.snapshot.discrete_positions_map.copy()
        positions[0] = np.array([0.0, 0.0, 2.0])
        injected = replace(self.snapshot, discrete_positions_map=positions)
        audit = self.validate(injected)
        self.assertFalse(audit.report.passed)
        self.assertFalse(audit.report.terrain_clear)
        self.assertIsNotNone(audit.first_failure_segment_index)
        self.assertIsNotNone(audit.first_failure_position_map)

    def test_r3_injected_turn_violation_is_detected(self) -> None:
        headings = list(self.snapshot.discrete_headings_rad)
        headings[0] = np.pi
        injected = replace(self.snapshot, discrete_headings_rad=tuple(headings))
        audit = self.validate(injected)
        self.assertFalse(audit.report.passed)
        self.assertFalse(audit.report.turn_constraints_valid)
        self.assertGreater(audit.report.max_turn_violation, 0.0)

    def test_r4_injected_time_inconsistency_is_detected(self) -> None:
        injected = replace(
            self.snapshot,
            stored_mission_time_s=self.snapshot.stored_mission_time_s + 1.0e-4,
        )
        report = self.validate(injected).report
        self.assertFalse(report.passed)
        self.assertFalse(report.time_consistent)
        self.assertGreater(report.time_error_s, DEFAULT_REPLAY_TOLERANCES.time_s)

    def test_r5_injected_hazard_inconsistency_is_detected(self) -> None:
        injected = replace(
            self.snapshot,
            stored_cumulative_hazard=self.snapshot.stored_cumulative_hazard + 1.0e-5,
        )
        report = self.validate(injected).report
        self.assertFalse(report.passed)
        self.assertFalse(report.hazard_consistent)
        self.assertGreater(report.hazard_error, DEFAULT_REPLAY_TOLERANCES.hazard)

    def test_r6_goal_just_inside_boundary_and_outside(self) -> None:
        tolerance = DEFAULT_GLIDER.goal_tolerance_m
        epsilon = DEFAULT_REPLAY_TOLERANCES.goal_boundary_m
        self.assertTrue(goal_is_valid(tolerance - epsilon))
        self.assertTrue(goal_is_valid(tolerance))
        self.assertTrue(goal_is_valid(tolerance + epsilon))
        self.assertFalse(goal_is_valid(tolerance + 2.0 * epsilon))

    def test_r7_energy_margin_boundary_uses_named_tolerance(self) -> None:
        margin = self.valid_audit.energy_margin_m
        epsilon = DEFAULT_REPLAY_TOLERANCES.energy_height_m
        within_parameters = replace(
            DEFAULT_GLIDER,
            switch_energy_loss_height_m=(
                DEFAULT_GLIDER.switch_energy_loss_height_m + margin + 0.5 * epsilon
            ),
        )
        outside_parameters = replace(
            DEFAULT_GLIDER,
            switch_energy_loss_height_m=(
                DEFAULT_GLIDER.switch_energy_loss_height_m + margin + 2.0 * epsilon
            ),
        )
        within = self.validate(self.snapshot, parameters=within_parameters)
        outside = self.validate(self.snapshot, parameters=outside_parameters)
        self.assertLess(within.energy_margin_m, 0.0)
        self.assertTrue(energy_margin_is_valid(within.energy_margin_m))
        self.assertTrue(within.report.energy_valid)
        self.assertFalse(energy_margin_is_valid(outside.energy_margin_m))
        self.assertFalse(outside.report.energy_valid)

    def test_replay_module_does_not_import_optimizer_recursion_at_runtime(self) -> None:
        source = Path(__file__).with_name("trajectory_validation.py").read_text(
            encoding="utf-8",
        )
        self.assertNotIn("from mission_response import", source)
        self.assertNotIn("from edge_hazard import", source)
        self.assertNotIn("from additive_bellman import", source)


if __name__ == "__main__":
    unittest.main()
