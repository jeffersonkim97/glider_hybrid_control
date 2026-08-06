"""Regression tests for all-segment terrain/LOS geometry certificates."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.interpolate import CubicSpline

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.geometry import (
    TerrainModel,
    build_geometry_bundle,
    los_boundary_height,
    terrain_height,
)
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.segment_feasibility import (
    certify_straight_segment_geometry,
    minimum_los_margin_on_segment,
    minimum_terrain_margin_on_segment,
)


def _terrain_model(z: np.ndarray, h: np.ndarray) -> TerrainModel:
    return TerrainModel(
        z_grid=z,
        sampled_height=h,
        interpolant=CubicSpline(z, h, bc_type="natural"),
    )


def _los_boundary(z: np.ndarray, h: np.ndarray) -> dict:
    return {"los_boundary": np.column_stack((z, h))}


class SegmentFeasibilityTests(unittest.TestCase):
    def test_cubic_interior_peak_is_not_missed_by_endpoint_checks(self) -> None:
        model = _terrain_model(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])
        )
        result = minimum_terrain_margin_on_segment(
            np.array([0.0, 0.75]), np.array([2.0, 0.75]), model
        )
        self.assertAlmostEqual(result["argmin_z"], 1.0, places=12)
        self.assertAlmostEqual(result["minimum_margin"], -0.25, places=12)
        self.assertGreater(result["candidate_count"], 2)

    def test_piecewise_linear_los_breakpoint_is_checked(self) -> None:
        los = _los_boundary(
            np.array([0.0, 0.51, 1.0]), np.array([0.0, 1.0, 0.0])
        )
        visible = minimum_los_margin_on_segment(
            np.array([0.0, 0.75]), np.array([1.0, 0.75]),
            los, sensor_z=1.0, requirement="visible",
        )
        self.assertAlmostEqual(visible["argmin_z"], 0.51, places=12)
        self.assertAlmostEqual(visible["minimum_margin"], -0.25, places=12)

        occluded = minimum_los_margin_on_segment(
            np.array([0.0, 0.75]), np.array([1.0, 0.75]),
            los, sensor_z=1.0, requirement="occluded", los_tolerance=0.1,
        )
        self.assertAlmostEqual(occluded["minimum_margin"], -0.65, places=12)

    def test_visible_constraint_stops_at_sensor(self) -> None:
        los = _los_boundary(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
        )
        result = minimum_los_margin_on_segment(
            np.array([1.25, -100.0]), np.array([2.0, -100.0]),
            los, sensor_z=1.0, requirement="visible",
        )
        self.assertEqual(result["minimum_margin"], np.inf)
        self.assertIsNone(result["argmin_z"])

    def test_combined_certificate_reports_each_constraint(self) -> None:
        model = _terrain_model(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])
        )
        los = _los_boundary(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
        )
        result = certify_straight_segment_geometry(
            np.array([0.0, 0.75]), np.array([2.0, 0.75]), model, los,
            sensor_z=2.0,
            airspace={"z_min": 0.0, "z_max": 2.0, "h_min": 0.0, "h_max": 2.0},
            terrain_tolerance=0.0,
            los_requirement="visible",
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["terrain_clear"])
        self.assertTrue(result["los_clear"])
        self.assertTrue(result["domain_clear"])

    def test_zero_length_powered_segment_is_a_point_certificate(self) -> None:
        model = _terrain_model(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
        )
        los = _los_boundary(
            np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
        )
        result = certify_straight_segment_geometry(
            np.array([0.0, 0.0]), np.array([0.0, 0.0]), model, los,
            sensor_z=2.0,
            airspace={"z_min": 0.0, "z_max": 2.0, "h_min": 0.0, "h_max": 2.0},
            terrain_tolerance=0.0,
            los_requirement="occluded",
            los_tolerance=0.0,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["certificate"], "stationary_point_geometry_evaluation")

    def test_vertical_powered_segment_uses_endpoint_extrema(self) -> None:
        model = _terrain_model(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0])
        )
        los = _los_boundary(
            np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
        )
        result = certify_straight_segment_geometry(
            np.array([0.0, 0.0]), np.array([0.0, 0.75]), model, los,
            sensor_z=2.0,
            airspace={"z_min": 0.0, "z_max": 2.0, "h_min": 0.0, "h_max": 2.0},
            terrain_tolerance=0.0,
            los_requirement="occluded",
            los_tolerance=0.0,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["minimum_terrain_margin"], 0.0)
        self.assertAlmostEqual(result["minimum_los_margin"], 0.25)
        self.assertEqual(result["certificate"], "vertical_segment_endpoint_extrema")

    def test_exact_margins_match_dense_reference_on_multi_hill_geometry(self) -> None:
        temporary_directory = TemporaryDirectory()
        logger = None
        try:
            directory = temporary_directory.name
            configuration = build_configuration_bundle(Path(directory))
            logger = configuration["primary_result"]["logging_utilities"]["logger"]
            modified = deepcopy(configuration)
            environment = modified["primary_result"]["environment_config"]
            environment["terrain"] = {
                "z_min": 0.0,
                "z_max": 2750.0,
                "hills": (
                    {"z_ridge": 900.0, "h_ridge": 110.0, "width": 130.0},
                    {"z_ridge": 1550.0, "h_ridge": 75.0, "width": 90.0},
                    {"z_ridge": 2050.0, "h_ridge": 95.0, "width": 120.0},
                ),
            }
            environment["grid"] = {
                **environment["grid"],
                "z_min": 0.0,
                "z_max": 2750.0,
                "z_count": 321,
                "z_spacing": 2750.0 / 320.0,
                "h_min": 0.0,
                "h_max": 300.0,
                "h_count": 201,
                "h_spacing": 1.5,
            }
            environment["airspace"] = {
                "z_min": 0.0, "z_max": 2750.0, "h_min": 0.0, "h_max": 300.0
            }
            environment["simulation"] = {
                **environment["simulation"], "z_min": 0.0, "z_max": 2750.0,
                "h_min": 0.0, "h_max": 300.0,
            }
            environment["z_goal"] = 2500.0
            modified["primary_result"]["sensor_config"]["default_z_sensor"] = 2300.0
            geometry = build_geometry_bundle(modified)["primary_result"]
            terrain = geometry["terrain_model"]
            los = geometry["los_geometry"]
            sensor_z = float(geometry["sensor_position"][0])
            rng = np.random.default_rng(1729)
            for _ in range(20):
                z_start = float(rng.uniform(100.0, 2000.0))
                z_end = float(rng.uniform(z_start + 20.0, min(z_start + 350.0, 2250.0)))
                h_start = float(rng.uniform(80.0, 260.0))
                h_end = float(rng.uniform(20.0, h_start - 1.0))
                start = np.array([z_start, h_start])
                end = np.array([z_end, h_end])
                terrain_exact = minimum_terrain_margin_on_segment(start, end, terrain)
                los_exact = minimum_los_margin_on_segment(
                    start, end, los, sensor_z, "visible"
                )
                fractions = np.linspace(0.0, 1.0, 20001)
                path = start[None, :] + fractions[:, None] * (end - start)[None, :]
                terrain_dense = float(np.min(
                    path[:, 1] - terrain_height(terrain, path[:, 0])
                ))
                boundary = los_boundary_height(los, path[:, 0])
                active = path[:, 0] <= sensor_z
                los_dense = (
                    float(np.min(path[active, 1] - boundary[active]))
                    if np.any(active) else np.inf
                )
                self.assertLessEqual(terrain_exact["minimum_margin"], terrain_dense + 1e-10)
                self.assertLess(terrain_dense - terrain_exact["minimum_margin"], 1e-4)
                self.assertLessEqual(los_exact["minimum_margin"], los_dense + 1e-10)
                self.assertLess(los_dense - los_exact["minimum_margin"], 1e-9)
        finally:
            if logger is not None:
                close_phase_logger(logger)
            temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
