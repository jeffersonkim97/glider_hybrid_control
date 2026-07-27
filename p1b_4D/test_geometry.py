"""Phase 2 terrain, LOS geometry, and Geometry Bundle tests."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.geometry import (
    build_geometry_bundle,
    sensor_position_from_z,
    terrain_curvature,
    terrain_gradient,
    terrain_height,
)
from p1b_4D.geometry_io import export_geometry_bundle, import_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger


def _sum_of_gaussian_hills(z: np.ndarray, hills: tuple[dict, ...]) -> np.ndarray:
    """Independent (test-local) reimplementation used to check geometry.py."""
    height = np.zeros_like(z, dtype=float)
    for hill in hills:
        height = height + hill["h_ridge"] * np.exp(
            -0.5 * ((z - hill["z_ridge"]) / hill["width"]) ** 2
        )
    return height


class GeometryTests(unittest.TestCase):
    """Verify authoritative terrain, geometry, masks, coverage, and persistence."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.configuration_bundle = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        logger = self.configuration_bundle["primary_result"]["logging_utilities"][
            "logger"
        ]
        self.addCleanup(close_phase_logger, logger)
        self.geometry_bundle = build_geometry_bundle(self.configuration_bundle)

    def test_geometry_bundle_passes_validation(self) -> None:
        self.assertTrue(self.geometry_bundle["status"]["success"])
        self.assertTrue(self.geometry_bundle["validation"]["passed"])
        self.assertEqual(
            set(self.geometry_bundle),
            {"primary_result", "validation", "metadata", "status"},
        )

    def test_existing_terrain_samples_are_reproduced(self) -> None:
        result = self.geometry_bundle["primary_result"]
        arrays = result["terrain_arrays"]
        terrain_config = self.configuration_bundle["primary_result"][
            "environment_config"
        ]["terrain"]
        expected = _sum_of_gaussian_hills(arrays["z"], terrain_config["hills"])
        # atol, not exact equality: far enough from every hill, the Gaussian
        # value underflows into the subnormal range (~1e-150), where two
        # mathematically-equivalent summation orders can legitimately differ
        # in their last bits -- physically meaningless noise, not a real
        # terrain-height mismatch.
        np.testing.assert_allclose(arrays["height"], expected, rtol=0.0, atol=1.0e-6)
        model = result["terrain_model"]
        self.assertTrue(np.all(np.isfinite(terrain_gradient(model, arrays["z"]))))
        self.assertTrue(np.all(np.isfinite(terrain_curvature(model, arrays["z"]))))

    def test_two_hill_terrain_is_one_combined_profile(self) -> None:
        two_hill_configuration = deepcopy(self.configuration_bundle)
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
        geometry_bundle = build_geometry_bundle(two_hill_configuration)
        self.assertTrue(geometry_bundle["status"]["success"])
        arrays = geometry_bundle["primary_result"]["terrain_arrays"]
        terrain_config = two_hill_configuration["primary_result"][
            "environment_config"
        ]["terrain"]
        expected = _sum_of_gaussian_hills(arrays["z"], terrain_config["hills"])
        # atol, not exact equality: far enough from every hill, the Gaussian
        # value underflows into the subnormal range (~1e-150), where two
        # mathematically-equivalent summation orders can legitimately differ
        # in their last bits -- physically meaningless noise, not a real
        # terrain-height mismatch.
        np.testing.assert_allclose(arrays["height"], expected, rtol=0.0, atol=1.0e-6)
        # A point exactly between the two equal hills should sit measurably
        # above either hill alone (it is fed by both), not equal to a
        # single-hill profile evaluated at the same point.
        model = geometry_bundle["primary_result"]["terrain_model"]
        midpoint_height = float(terrain_height(model, 1250.0))
        single_hill_height = 100.0 * np.exp(-0.5 * ((1250.0 - 1000.0) / 100.0) ** 2)
        self.assertGreater(midpoint_height, single_hill_height)

    def test_sensor_always_follows_terrain(self) -> None:
        result = self.geometry_bundle["primary_result"]
        sensor_position = result["sensor_position"]
        model = result["terrain_model"]
        mount_height = self.configuration_bundle["primary_result"][
            "sensor_config"
        ]["mount_height"]
        self.assertAlmostEqual(
            sensor_position[1],
            float(terrain_height(model, sensor_position[0])) + mount_height,
        )

    def test_goal_follows_terrain(self) -> None:
        result = self.geometry_bundle["primary_result"]
        environment = self.configuration_bundle["primary_result"][
            "environment_config"
        ]
        model = result["terrain_model"]
        goal_position = result["goal_position"]
        self.assertEqual(goal_position[0], environment["z_goal"])
        self.assertAlmostEqual(
            goal_position[1],
            float(terrain_height(model, environment["z_goal"])),
        )

    def test_masks_partition_grid_and_include_terrain_in_occlusion(self) -> None:
        result = self.geometry_bundle["primary_result"]
        masks = result["los_masks"]
        expected_shape = (
            self.configuration_bundle["primary_result"]["environment_config"][
                "grid"
            ]["z_count"],
            self.configuration_bundle["primary_result"]["environment_config"][
                "grid"
            ]["h_count"],
        )
        self.assertEqual(masks["los_mask"].shape, expected_shape)
        np.testing.assert_array_equal(
            masks["los_mask"], ~masks["occlusion_mask"]
        )
        self.assertTrue(
            np.all(masks["occlusion_mask"][masks["terrain_mask"]])
        )

    def test_tangent_and_coverage_are_consistent(self) -> None:
        result = self.geometry_bundle["primary_result"]
        geometry = result["los_geometry"]
        coverage = result["coverage"]
        self.assertLessEqual(
            abs(geometry["tangent_residual"]),
            self.configuration_bundle["primary_result"]["validation_config"][
                "los_tolerance"
            ],
        )
        self.assertGreaterEqual(coverage["coverage_area"], 0.0)
        self.assertLessEqual(
            coverage["coverage_area"], coverage["admissible_airspace_area"]
        )
        self.assertGreaterEqual(coverage["normalized_coverage_area"], 0.0)
        self.assertLessEqual(coverage["normalized_coverage_area"], 1.0)

    def test_out_of_domain_sensor_is_rejected(self) -> None:
        result = self.geometry_bundle["primary_result"]
        model = result["terrain_model"]
        sensor_config = self.configuration_bundle["primary_result"][
            "sensor_config"
        ]
        with self.assertRaises(ValueError):
            sensor_position_from_z(model, -1.0, sensor_config)

    def test_json_npz_round_trip(self) -> None:
        exported = export_geometry_bundle(
            self.geometry_bundle, self.configuration_bundle
        )
        self.assertTrue(exported["status"]["success"])
        json_path = exported["primary_result"]["json_path"]
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundle_type"], "GeometryBundle")
        self.assertNotIn("los_mask", manifest["geometry"])
        imported = import_geometry_bundle(json_path)
        self.assertTrue(imported["status"]["success"])
        original = self.geometry_bundle["primary_result"]["los_masks"]["los_mask"]
        np.testing.assert_array_equal(
            imported["primary_result"]["arrays"]["los_mask"],
            original,
        )


if __name__ == "__main__":
    unittest.main()
