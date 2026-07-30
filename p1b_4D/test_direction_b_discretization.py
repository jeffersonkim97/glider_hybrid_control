"""B1 regression gates for the frozen nested-discretization protocol."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from p1b_4D.bellman import generate_switching_point_seeds
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import (
    DIRECTION_B_GRID_COUNTS,
    DIRECTION_B_SHALLOW_BACKBONE_FORWARD_CELLS_L0,
    DIRECTION_B_SPEEDS,
    build_direction_b_configuration,
    construct_direction_b_grids,
    direction_b_physical_envelope,
)
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.high_fidelity_policy_evaluation import qualify_common_evaluator
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.successor_grid_solver import (
    build_successor_grid_graph,
    regular_action_offsets,
    solve_successor_grid_attacker,
    virtual_switch_target_indices,
)
from p1b_4D.test_continuous_replay_evaluation import _build_toy_bundles


class DirectionBNestedDiscretizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger,
            self.base["primary_result"]["logging_utilities"]["logger"],
        )

    def test_spatial_nodes_are_exact_subsets_for_every_terrain(self) -> None:
        for terrain_name in DIRECTION_B_GRID_COUNTS:
            base = _configuration_for_terrain(self.base, terrain_name)
            grids = [
                construct_direction_b_grids(
                    build_direction_b_configuration(base, terrain_name, level)
                )
                for level in range(3)
            ]
            for coarse, fine in zip(grids[:-1], grids[1:]):
                np.testing.assert_array_equal(coarse["z"], fine["z"][::2])
                np.testing.assert_array_equal(coarse["h"], fine["h"][::2])

    def test_physical_envelope_is_level_invariant(self) -> None:
        expected = {
            "single_hill": (137.5, 8.0),
            "two_hill": (137.5, 8.0),
            "goal_in_valley": (16000.0 / 116.0, 8.0),
        }
        for terrain_name, target in expected.items():
            base = _configuration_for_terrain(self.base, terrain_name)
            envelopes = [
                direction_b_physical_envelope(
                    build_direction_b_configuration(base, terrain_name, level)
                )
                for level in range(3)
            ]
            for envelope in envelopes:
                np.testing.assert_allclose(envelope, target, rtol=0.0, atol=1e-12)

    def test_regular_action_families_are_physically_nested(self) -> None:
        base = _configuration_for_terrain(self.base, "two_hill")
        for family in ("enriched", "transported"):
            physical_sets = []
            offset_counts = []
            for level in range(3):
                bundle = build_direction_b_configuration(
                    base, "two_hill", level, action_family=family
                )
                configs = bundle["primary_result"]
                grid = configs["environment_config"]["grid"]
                options = configs["attacker_solver_config"]["successor_grid"]
                offsets = regular_action_offsets(options)
                offset_counts.append(len(offsets))
                physical_sets.append({
                    (p * grid["z_spacing"], q * grid["h_spacing"])
                    for p, q in offsets
                })
            if family == "enriched":
                self.assertEqual(offset_counts, [3, 9, 33])
                self.assertTrue(physical_sets[0] <= physical_sets[1])
                self.assertTrue(physical_sets[1] <= physical_sets[2])
                backbone = (
                    DIRECTION_B_SHALLOW_BACKBONE_FORWARD_CELLS_L0
                    * 34.375,
                    4.0,
                )
                self.assertIn(backbone, physical_sets[0])
                self.assertIn(backbone, physical_sets[1])
                self.assertIn(backbone, physical_sets[2])
            else:
                self.assertEqual(offset_counts, [2, 2, 2])
                self.assertEqual(physical_sets[0], physical_sets[1])
                self.assertEqual(physical_sets[1], physical_sets[2])

    def test_v5_is_exact_subset_of_v9(self) -> None:
        np.testing.assert_array_equal(
            DIRECTION_B_SPEEDS["V5"], DIRECTION_B_SPEEDS["V9"][::2]
        )

    def test_fixed_geometry_switch_seeds_and_virtual_targets_are_nested(self) -> None:
        terrain_name = "two_hill"
        base = _configuration_for_terrain(self.base, terrain_name)
        bundles = [
            build_direction_b_configuration(base, terrain_name, level)
            for level in range(3)
        ]
        grids = [construct_direction_b_grids(bundle) for bundle in bundles]
        geometry = [build_geometry_bundle(bundle) for bundle in bundles]
        for item in geometry[1:]:
            np.testing.assert_array_equal(
                geometry[0]["primary_result"]["terrain_model"].z_grid,
                item["primary_result"]["terrain_model"].z_grid,
            )
            np.testing.assert_array_equal(
                geometry[0]["primary_result"]["los_geometry"]["los_boundary"],
                item["primary_result"]["los_geometry"]["los_boundary"],
            )
        seeds = [
            generate_switching_point_seeds(g, b, grid["z"])
            for g, b, grid in zip(geometry, bundles, grids)
        ]
        for coarse, fine in zip(seeds[:-1], seeds[1:]):
            fine_by_z = {float(seed[0]): seed for seed in fine}
            for seed in coarse:
                np.testing.assert_array_equal(seed, fine_by_z[float(seed[0])])

        switching_point = next(seed for seed in seeds[0] if seed[1] > 10.0)
        for family in ("enriched", "transported"):
            target_sets = []
            for level in range(3):
                bundle = build_direction_b_configuration(
                    base, terrain_name, level, action_family=family
                )
                grid = construct_direction_b_grids(bundle)
                options = bundle["primary_result"]["attacker_solver_config"][
                    "successor_grid"
                ]
                target_sets.append({
                    (float(grid["z"][zi]), float(grid["h"][hi]))
                    for zi, hi in virtual_switch_target_indices(
                        switching_point, grid, options
                    )
                })
            if family == "enriched":
                self.assertTrue(target_sets[0] <= target_sets[1])
                self.assertTrue(target_sets[1] <= target_sets[2])
            else:
                self.assertEqual(target_sets[0], target_sets[1])
                self.assertEqual(target_sets[1], target_sets[2])

    def test_all_frozen_edges_reconstruct_endpoints_to_machine_precision(self) -> None:
        for terrain_name in DIRECTION_B_GRID_COUNTS:
            base = _configuration_for_terrain(self.base, terrain_name)
            for family in ("enriched", "transported"):
                for level in range(3):
                    bundle = build_direction_b_configuration(
                        base, terrain_name, level, action_family=family
                    )
                    configs = bundle["primary_result"]
                    grid = configs["environment_config"]["grid"]
                    options = configs["attacker_solver_config"]["successor_grid"]
                    for forward, descent in regular_action_offsets(options):
                        displacement = np.array([
                            forward * grid["z_spacing"],
                            -descent * grid["h_spacing"],
                        ])
                        gamma = math.atan2(displacement[1], displacement[0])
                        length = float(np.linalg.norm(displacement))
                        for speed in DIRECTION_B_SPEEDS["V5"]:
                            duration = length / float(speed)
                            reconstructed = duration * speed * np.array(
                                [math.cos(gamma), math.sin(gamma)]
                            )
                            self.assertLessEqual(
                                np.linalg.norm(reconstructed - displacement),
                                1e-12,
                            )

    def test_common_evaluator_passes_129_vs_257_toy_qualification(self) -> None:
        toy, geometry, detection = _build_toy_bundles(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger,
            toy["primary_result"]["logging_utilities"]["logger"],
        )
        toy["primary_result"]["attacker_solver_config"][
            "transition_model"
        ] = "successor_grid_physical_edge"
        _, response = solve_successor_grid_attacker(toy, geometry, detection)
        qualification = qualify_common_evaluator(
            response["primary_result"], toy, geometry
        )
        self.assertTrue(qualification["passed"], qualification)
        self.assertLessEqual(
            qualification["objective_absolute_difference"], 1e-6
        )
        self.assertLessEqual(
            qualification["mission_pod_absolute_difference"], 1e-6
        )

    def test_direction_b_l0_graph_uses_physical_virtual_box(self) -> None:
        base = _configuration_for_terrain(self.base, "two_hill")
        bundle = build_direction_b_configuration(base, "two_hill", 0)
        geometry = build_geometry_bundle(bundle)
        graph = build_successor_grid_graph(bundle, geometry)
        metadata = graph["metadata"]
        self.assertEqual(metadata["action_family"], "enriched")
        self.assertEqual(
            metadata["virtual_switch_target_family"],
            "physical_box_enriched",
        )
        self.assertFalse(metadata["endpoint_snapping"])

    def test_revised_enriched_l0_l1_are_feasible_for_b2_sensors(self) -> None:
        for sensor_z in (1966.4609053497943, 1982.9218106995881):
            for level in (0, 1):
                base = _configuration_for_terrain(self.base, "two_hill")
                base["primary_result"]["sensor_config"][
                    "default_z_sensor"
                ] = sensor_z
                bundle = build_direction_b_configuration(
                    base, "two_hill", level
                )
                geometry = build_geometry_bundle(bundle)
                detection = build_symbolic_detection_bundle(bundle, geometry)
                _, response = solve_successor_grid_attacker(
                    bundle, geometry, detection
                )
                self.assertTrue(response["status"]["success"])


def _configuration_for_terrain(configuration, terrain_name):
    result = deepcopy(configuration)
    configs = result["primary_result"]
    env = configs["environment_config"]
    if terrain_name == "single_hill":
        z_max, h_max, z_goal, sensor_z = 5500.0, 400.0, 5000.0, 4000.0
        hills = ({"z_ridge": 2500.0, "h_ridge": 200.0, "width": 400.0},)
    elif terrain_name == "two_hill":
        z_max, h_max, z_goal, sensor_z = 2750.0, 200.0, 2500.0, 1982.9218106995881
        hills = (
            {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 2000.0, "h_ridge": 50.0, "width": 100.0},
        )
    else:
        z_max, h_max, z_goal, sensor_z = 4000.0, 200.0, 2500.0, 1950.0
        hills = (
            {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 3500.0, "h_ridge": 100.0, "width": 100.0},
        )
    env["z_start"] = 0.0
    env["h_start"] = 0.0
    env["z_goal"] = z_goal
    env["terrain"] = {"z_min": 0.0, "z_max": z_max, "hills": hills}
    env["grid"].update({
        "z_min": 0.0, "z_max": z_max,
        "h_min": 0.0, "h_max": h_max,
    })
    env["airspace"] = {
        "z_min": 0.0, "z_max": z_max, "h_min": 0.0, "h_max": h_max,
    }
    env["simulation"] = {
        "z_min": 0.0, "z_max": z_max, "h_min": 0.0, "h_max": h_max,
        "max_path_steps": 1000,
    }
    configs["sensor_config"]["default_z_sensor"] = sensor_z
    configs["defender_config"]["continuous_search_bounds"] = {
        "z_sensor_min": max(1.0, 0.6 * z_max),
        "z_sensor_max": z_max,
    }
    return result


if __name__ == "__main__":
    unittest.main()
