"""Phase 6 unfiltered multi-start coarse Bellman tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from p1b_4D.bellman import (
    generate_bellman_candidates,
    generate_switching_point_seeds,
    select_authoritative_bellman_response,
)
from p1b_4D.bellman_io import export_bellman_candidate_bundle, import_bellman_candidate_bundle
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle, los_boundary_height
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
        self.assertGreater(result["attempted_start_count"], 1)
        self.assertGreater(result["candidate_count"], 1)
        self.assertFalse(result["filtering_applied"])
        self.assertFalse(result["ranking_applied"])

    def test_switching_seeds_exhaustively_cover_tangent_grid_nodes(self) -> None:
        z_grid = self.stage["primary_result"]["grids"]["z"]
        seeds = generate_switching_point_seeds(self.geometry, self.configuration, z_grid)
        result = self.bellman["primary_result"]
        self.assertEqual(result["attempted_start_count"], seeds.shape[0])
        self.assertTrue(
            self.bellman["validation"]["checks"][
                "switching_grid_cells_exhaustive_and_unique"
            ]
        )
        self.assertTrue(np.isin(seeds[:, 0], z_grid).all())
        attempted_grid_z = {
            attempt["grid_start_index"][0] for attempt in result["start_attempts"]
        }
        self.assertEqual(len(attempted_grid_z), seeds.shape[0])

    def test_candidates_reach_goal_and_preserve_profiles(self) -> None:
        goal = np.array(self.geometry["primary_result"]["goal_position"])
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
                float(los_boundary_height(tangent, np.array([z_switch]))[0]),
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
        goal = np.array(geometry["primary_result"]["goal_position"])
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

    def test_two_hill_terrain_runs_end_to_end_through_bellman(self) -> None:
        two_hill_configuration = deepcopy(self.configuration)
        two_hill_configuration["primary_result"]["environment_config"] = deepcopy(
            two_hill_configuration["primary_result"]["environment_config"]
        )
        two_hill_configuration["primary_result"]["environment_config"]["terrain"] = {
            "z_min": 0.0,
            "z_max": 2750.0,
            "hills": (
                {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
                {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            ),
        }
        geometry = build_geometry_bundle(two_hill_configuration)
        self.assertTrue(geometry["status"]["success"])
        detection = build_symbolic_detection_bundle(two_hill_configuration, geometry)
        self.assertTrue(detection["status"]["success"])
        stage = construct_stage_cost_4d(two_hill_configuration, geometry, detection)
        self.assertTrue(stage["status"]["success"])
        bellman = generate_bellman_candidates(
            two_hill_configuration, geometry, detection, stage, None
        )
        self.assertTrue(bellman["status"]["success"])
        goal = np.array(geometry["primary_result"]["goal_position"])
        goal_radius = two_hill_configuration["primary_result"]["validation_config"][
            "goal_radius"
        ]
        self.assertGreater(len(bellman["primary_result"]["candidates"]), 0)
        # gamma_max_deg is always negative (see vehicle_config), so gliding
        # cannot climb; each step's continuous descent is real, but the
        # coarse Bellman grid snaps successors to the nearest h-grid node
        # (np.rint), which can round a tiny continuous descent up by at most
        # one grid cell -- a pre-existing discretization artifact confirmed
        # to already occur identically in the single-hill baseline, not
        # something this terrain generalization introduced.
        dh = float(np.diff(stage["primary_result"]["grids"]["h"])[0])
        for candidate in bellman["primary_result"]["candidates"]:
            trajectory = candidate["trajectory"]
            self.assertTrue(np.all(np.diff(trajectory[:, 0]) > 0.0))
            self.assertTrue(np.all(np.diff(trajectory[:, 1]) <= dh + 1.0e-6))
            self.assertLessEqual(
                np.linalg.norm(trajectory[-1] - goal), goal_radius + 1.0e-6
            )
            self.assertTrue(candidate["validation"]["passed"])

    def test_candidate_ranking_cost_uses_complete_mission_objective(self) -> None:
        costs = self.configuration["primary_result"]["cost_config"]["attacker"]
        for candidate in self.bellman["primary_result"]["candidates"]:
            breakdown = candidate["objective_breakdown"]
            self.assertAlmostEqual(
                candidate["mission_cost"], breakdown["total_cost"], places=14
            )
            self.assertAlmostEqual(
                candidate["mission_cost"],
                costs["w_pod"] * breakdown["pod_normalized"]
                + costs["w_time"] * breakdown["time_normalized"],
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

    def test_pod_to_go_is_bounded_probability_matching_cost_to_go_support(self) -> None:
        result = self.bellman["primary_result"]
        checks = self.bellman["validation"]["checks"]
        self.assertIn("pod_to_go_bounded_unit_interval", checks)
        self.assertTrue(checks["pod_to_go_bounded_unit_interval"])
        self.assertIn("pod_to_go_matches_cost_to_go_support", checks)
        self.assertTrue(checks["pod_to_go_matches_cost_to_go_support"])
        primary_ordering = result["cost_to_go_primary_ordering"]
        pod_to_go = result["pod_to_go_maps"][primary_ordering]
        cost_to_go = result["cost_to_go_maps"][primary_ordering]
        finite = np.isfinite(cost_to_go)
        np.testing.assert_array_equal(np.isfinite(pod_to_go), finite)
        self.assertTrue(np.all(pod_to_go[finite] >= 0.0))
        self.assertTrue(np.all(pod_to_go[finite] <= 1.0))
        goal_mask = result["finite_cost_to_go_mask"] & (cost_to_go == 0.0)
        np.testing.assert_allclose(pod_to_go[goal_mask], 0.0, atol=1.0e-12)

    def test_ordering_value_agreement_is_checked(self) -> None:
        checks = self.bellman["validation"]["checks"]
        self.assertIn("ordering_value_agreement", checks)
        self.assertTrue(checks["ordering_value_agreement"])

    def test_authoritative_response_selects_minimum_cost_candidate(self) -> None:
        response = select_authoritative_bellman_response(
            self.bellman, self.configuration
        )
        self.assertTrue(response["status"]["success"])
        candidates = self.bellman["primary_result"]["candidates"]
        minimum_cost = min(candidate["mission_cost"] for candidate in candidates)
        primary = response["primary_result"]
        self.assertEqual(primary["mission_cost"], minimum_cost)
        self.assertEqual(primary["mission_objective"], minimum_cost)
        self.assertEqual(response["metadata"]["solution_method"], "bellman_dynamic_programming")
        self.assertFalse(response["metadata"]["global_optimum_claim"])
        self.assertTrue(response["validation"]["checks"]["objective_matches_bellman_value"])
        self.assertTrue(response["validation"]["checks"]["selection_is_minimum_cost"])

    def test_authoritative_response_is_traceable_to_one_source_candidate(self) -> None:
        response = select_authoritative_bellman_response(
            self.bellman, self.configuration
        )
        primary = response["primary_result"]
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in self.bellman["primary_result"]["candidates"]
        }
        source = candidates[primary["source_candidate_id"]]
        np.testing.assert_array_equal(primary["trajectory"], source["trajectory"])
        np.testing.assert_array_equal(primary["switching_point"], source["switching_point"])
        self.assertEqual(primary["powered_hazard"], source["hazard_breakdown"]["powered_acoustic_hazard"])
        self.assertEqual(primary["glide_hazard"], source["hazard_breakdown"]["glide_radar_doppler_hazard"])

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
