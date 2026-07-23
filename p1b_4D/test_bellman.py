"""Phase 6 unfiltered multi-start coarse Bellman tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from p1b_4D.bellman import generate_bellman_candidates
from p1b_4D.bellman_io import export_bellman_candidate_bundle, import_bellman_candidate_bundle
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.projection import construct_projected_cost_map
from p1b_4D.stage_cost import construct_stage_cost_4d


class BellmanTests(unittest.TestCase):
    """Verify J4D policy use, physical candidates, and portable persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.configuration = build_configuration_bundle(Path(cls.temporary_directory.name))
        cls.geometry = build_geometry_bundle(cls.configuration)
        cls.detection = build_symbolic_detection_bundle(cls.configuration, cls.geometry)
        cls.stage = construct_stage_cost_4d(cls.configuration, cls.geometry, cls.detection)
        cls.projection = construct_projected_cost_map(
            cls.configuration, cls.geometry, cls.detection, cls.stage
        )
        cls.bellman = generate_bellman_candidates(
            cls.configuration, cls.geometry, cls.detection, cls.stage, cls.projection
        )

    @classmethod
    def tearDownClass(cls) -> None:
        logger = cls.configuration["primary_result"]["logging_utilities"]["logger"]
        close_phase_logger(logger)
        cls.temporary_directory.cleanup()

    def test_complete_unfiltered_candidate_set_passes(self) -> None:
        result = self.bellman["primary_result"]
        self.assertTrue(self.bellman["status"]["success"])
        self.assertEqual(result["attempted_start_count"], 20)
        self.assertGreater(result["candidate_count"], 1)
        self.assertFalse(result["filtering_applied"])
        self.assertFalse(result["ranking_applied"])

    def test_candidates_reach_goal_and_preserve_profiles(self) -> None:
        goal = np.array([2500.0, 0.0])
        goal_radius = self.configuration["primary_result"]["validation_config"][
            "goal_radius"
        ]
        for candidate in self.bellman["primary_result"]["candidates"]:
            self.assertLessEqual(
                np.linalg.norm(candidate["trajectory"][-1] - goal),
                goal_radius + 1.0e-6,
            )
            self.assertTrue(np.all(np.diff(candidate["trajectory"][:, 0]) > 0.0))
            self.assertLessEqual(
                np.max(candidate["trajectory"][:, 0]),
                goal[0] + goal_radius + 1.0e-6,
            )
            self.assertEqual(
                candidate["speed_profile"].size,
                candidate["trajectory"].shape[0] - 1,
            )
            self.assertEqual(candidate["gamma_profile"].size, candidate["speed_profile"].size)
            self.assertTrue(candidate["validation"]["passed"])

    def test_switching_points_are_on_los_tangent(self) -> None:
        tangent = self.geometry["primary_result"]["los_geometry"]
        for candidate in self.bellman["primary_result"]["candidates"]:
            z_switch, h_switch = candidate["switching_point"]
            self.assertAlmostEqual(
                h_switch,
                tangent["tangent_slope"] * z_switch + tangent["tangent_intercept"],
                places=10,
            )

    def test_upper_defender_bound_has_fractional_goal_entry_without_overshoot(self) -> None:
        boundary_configuration = deepcopy(self.configuration)
        boundary_configuration["primary_result"]["sensor_config"][
            "default_z_sensor"
        ] = boundary_configuration["primary_result"]["defender_config"][
            "continuous_search_bounds"
        ]["z_sensor_max"]
        geometry = build_geometry_bundle(boundary_configuration)
        detection = build_symbolic_detection_bundle(
            boundary_configuration, geometry
        )
        stage = construct_stage_cost_4d(
            boundary_configuration, geometry, detection
        )
        bellman = generate_bellman_candidates(
            boundary_configuration, geometry, detection, stage, None
        )
        environment = boundary_configuration["primary_result"][
            "environment_config"
        ]
        goal = np.array([environment["z_goal"], environment["h_goal"]])
        goal_radius = boundary_configuration["primary_result"][
            "validation_config"
        ]["goal_radius"]
        for candidate in bellman["primary_result"]["candidates"]:
            trajectory = candidate["trajectory"]
            self.assertTrue(np.all(np.diff(trajectory[:, 0]) > 0.0))
            self.assertLessEqual(
                np.linalg.norm(trajectory[-1] - goal), goal_radius + 1.0e-6
            )
            self.assertLessEqual(
                np.max(trajectory[:, 0]), environment["z_goal"] + goal_radius
            )

    def test_candidate_ranking_cost_uses_complete_mission_objective(self) -> None:
        for candidate in self.bellman["primary_result"]["candidates"]:
            breakdown = candidate["objective_breakdown"]
            self.assertAlmostEqual(
                candidate["mission_cost"], breakdown["total_cost"], places=14
            )
            self.assertAlmostEqual(
                candidate["mission_cost"],
                0.5 * breakdown["pod_normalized"]
                + 0.5 * breakdown["time_normalized"],
                places=14,
            )

    def test_projected_cost_values_cannot_change_policy(self) -> None:
        poisoned = deepcopy(self.projection)
        shape = poisoned["primary_result"]["projected_cost"].shape
        poisoned["primary_result"]["projected_cost"] = np.full(shape, -1.0e30)
        poisoned_result = generate_bellman_candidates(
            self.configuration, self.geometry, self.detection, self.stage, poisoned
        )
        original = self.bellman["primary_result"]["candidates"]
        changed = poisoned_result["primary_result"]["candidates"]
        self.assertEqual(len(original), len(changed))
        np.testing.assert_array_equal(
            [candidate["mission_cost"] for candidate in original],
            [candidate["mission_cost"] for candidate in changed],
        )
        for original_candidate, changed_candidate in zip(original, changed, strict=True):
            np.testing.assert_array_equal(original_candidate["trajectory"], changed_candidate["trajectory"])

    def test_json_npz_round_trip(self) -> None:
        exported = export_bellman_candidate_bundle(self.bellman, self.configuration)
        imported = import_bellman_candidate_bundle(exported["primary_result"]["json_path"])
        self.assertTrue(imported["status"]["success"])
        arrays = imported["primary_result"]["arrays"]
        np.testing.assert_array_equal(
            arrays["mission_costs"],
            [candidate["mission_cost"] for candidate in self.bellman["primary_result"]["candidates"]],
        )
        self.assertEqual(arrays["trajectory_offsets"].size, len(arrays["mission_costs"]) + 1)
        self.assertFalse(arrays["trajectory_points"].flags.writeable)


if __name__ == "__main__":
    unittest.main()
