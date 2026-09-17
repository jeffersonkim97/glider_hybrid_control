"""Stage-7 detection hazard, quadrature, and objective tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import unittest

import numpy as np

from bellman_geometry import GlideEdge
from bellman_state import BellmanState, BellmanStateGrid
from detection_hazard import (
    AttackerHazardTimeParameters,
    ConstantHazardField,
    GlideDetectionHazardModel,
    HazardRateEvaluation,
    PiecewiseConstantHazardField,
    evaluate_attacker_hazard_time_objective,
    hazard_to_detection_probability,
)
from edge_hazard import (
    accumulate_path_hazard,
    integrate_edge_hazard,
    precompute_edge_hazards,
)
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import MapBounds
from scenario import Point3D
from solver_metrics import HazardTiming
from terrain_catalog import build_terrain


def _state_at(
    grid: BellmanStateGrid,
    x: float,
    y: float,
    altitude: float,
    heading_bin: int,
) -> BellmanState:
    return BellmanState(
        int(round((x - grid.bounds.x_min) / grid.horizontal_spacing_map)),
        int(round((y - grid.bounds.y_min) / grid.horizontal_spacing_map)),
        int(round(
            (altitude - grid.minimum_altitude_map)
            / grid.altitude_spacing_map
        )),
        heading_bin,
    )


def _edge_between(
    grid: BellmanStateGrid,
    source_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    heading_bin: int,
) -> GlideEdge:
    source = _state_at(grid, *source_xyz, heading_bin)
    target = _state_at(grid, *target_xyz, heading_bin)
    source_position = grid.position_map(source)
    target_position = grid.position_map(target)
    horizontal_distance_m = DEFAULT_PHYSICAL_SCALE.distance_m(float(
        np.linalg.norm(target_position[:2] - source_position[:2])
    ))
    duration_s = horizontal_distance_m / DEFAULT_GLIDER.best_glide_speed_mps
    return GlideEdge(
        source_id=grid.encode(source),
        target_id=grid.encode(target),
        source_state=source,
        target_state=target,
        horizontal_distance_m=horizontal_distance_m,
        altitude_loss_m=DEFAULT_PHYSICAL_SCALE.distance_m(
            float(source_position[2] - target_position[2])
        ),
        duration_s=duration_s,
        heading_change_rad=0.0,
    )


@dataclass(frozen=True)
class SmoothQuadraticHazardField:
    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation:
        rate = float(position_map[0] ** 2)
        return HazardRateEvaluation(
            visible=True,
            sensor_range_m=0.0,
            radial_velocity_mps=0.0,
            cosine_aspect=0.0,
            radar_cross_section=0.0,
            radar_rate_per_s=rate,
            doppler_rate_per_s=0.0,
            total_rate_per_s=rate,
        )


class DetectionHazardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simple_grid = BellmanStateGrid(
            bounds=MapBounds(0.0, 4.0, 0.0, 1.0),
            minimum_altitude_map=0.0,
            maximum_altitude_map=0.4,
            altitude_spacing_map=0.1,
        )
        cls.simple_edge = _edge_between(
            cls.simple_grid,
            (0.0, 0.0, 0.4),
            (4.0, 0.0, 0.0),
            0,
        )

        cls.los_bounds = MapBounds(-8.0, 8.0, -8.0, 8.0)
        cls.los_grid = BellmanStateGrid(
            bounds=cls.los_bounds,
            minimum_altitude_map=0.0,
            maximum_altitude_map=2.0,
            altitude_spacing_map=0.1,
        )
        cls.terrain = build_terrain("centered_cube", bounds=cls.los_bounds)
        cls.sensor = Point3D(5.0, 0.0, 0.0)
        cls.detection_model = GlideDetectionHazardModel(
            cls.terrain,
            cls.sensor,
        )
        cls.occluded_edge = _edge_between(
            cls.los_grid,
            (-3.0, 3.0, 1.0),
            (-3.0, 4.0, 0.9),
            2,
        )
        cls.crossing_edge = _edge_between(
            cls.los_grid,
            (-3.0, 4.0, 1.0),
            (-3.0, 6.0, 0.8),
            2,
        )
        cls.visible_edge = _edge_between(
            cls.los_grid,
            (3.0, 0.0, 1.0),
            (4.0, 0.0, 0.9),
            0,
        )

    def test_h1_constant_field_matches_c_times_duration(self) -> None:
        rate = 0.25
        result = integrate_edge_hazard(
            self.simple_edge,
            self.simple_grid,
            ConstantHazardField(rate),
            quadrature_resolution=4,
        )
        self.assertAlmostEqual(result.hazard, rate * self.simple_edge.duration_s, places=12)
        self.assertEqual(result.visible_sample_count, 4)

    def test_h2_zero_hazard_has_zero_pod(self) -> None:
        result = integrate_edge_hazard(
            self.simple_edge,
            self.simple_grid,
            ConstantHazardField(0.0),
            quadrature_resolution=8,
        )
        self.assertEqual(result.hazard, 0.0)
        self.assertEqual(hazard_to_detection_probability(result.hazard), 0.0)

    def test_h3_piecewise_constant_matches_manual_fraction(self) -> None:
        field = PiecewiseConstantHazardField(
            axis_index=0,
            boundary_map=2.0,
            below_rate_per_s=0.2,
            above_rate_per_s=0.6,
        )
        result = integrate_edge_hazard(
            self.simple_edge,
            self.simple_grid,
            field,
            quadrature_resolution=5,
        )
        expected = 0.2 * self.simple_edge.duration_s / 2.0 + 0.6 * self.simple_edge.duration_s / 2.0
        self.assertAlmostEqual(result.hazard, expected, places=12)

    def test_h4_los_gating_orders_occluded_crossing_and_visible_edges(self) -> None:
        occluded = integrate_edge_hazard(
            self.occluded_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=33,
        )
        crossing = integrate_edge_hazard(
            self.crossing_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=33,
        )
        visible = integrate_edge_hazard(
            self.visible_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=33,
        )
        self.assertEqual(occluded.visible_sample_count, 0)
        self.assertEqual(occluded.hazard, 0.0)
        self.assertGreater(crossing.visible_sample_count, 0)
        self.assertGreater(crossing.occluded_sample_count, 0)
        self.assertEqual(visible.occluded_sample_count, 0)
        self.assertLess(occluded.hazard, crossing.hazard)
        self.assertLess(crossing.hazard, visible.hazard)

    def test_h5_path_hazard_is_sum_of_edge_hazards(self) -> None:
        first = integrate_edge_hazard(
            self.occluded_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=16,
        )
        second = integrate_edge_hazard(
            self.crossing_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=16,
        )
        third = integrate_edge_hazard(
            self.visible_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=16,
        )
        self.assertAlmostEqual(
            accumulate_path_hazard((first, second, third)),
            first.hazard + second.hazard + third.hazard,
            places=15,
        )

    def test_h6_hazard_to_pod_explicit_values(self) -> None:
        self.assertEqual(hazard_to_detection_probability(0.0), 0.0)
        self.assertAlmostEqual(hazard_to_detection_probability(np.log(2.0)), 0.5, places=15)
        self.assertGreater(hazard_to_detection_probability(20.0), 0.999999)
        self.assertEqual(hazard_to_detection_probability(np.inf), 1.0)

    def test_h7_trapezoidal_quadrature_converges(self) -> None:
        edge = _edge_between(
            self.simple_grid,
            (0.0, 0.0, 0.1),
            (1.0, 0.0, 0.0),
            0,
        )
        analytical = edge.duration_s / 3.0
        resolutions = (4, 8, 16, 32)
        values = [
            integrate_edge_hazard(
                edge,
                self.simple_grid,
                SmoothQuadraticHazardField(),
                quadrature_resolution=resolution,
            ).hazard
            for resolution in resolutions
        ]
        errors = [abs(value - analytical) for value in values]
        self.assertTrue(all(
            later < earlier
            for earlier, later in zip(errors, errors[1:])
        ))
        self.assertLess(errors[-1], errors[0] / 50.0)

    def test_h8_actual_attacker_objective_decomposes_exactly(self) -> None:
        parameters = AttackerHazardTimeParameters()
        result = evaluate_attacker_hazard_time_objective(0.4, 100.0)
        expected_hazard_term = 0.5 * 0.4 / 1.0
        expected_time_term = 0.5 * 100.0 / (5000.0 / 22.6)
        self.assertAlmostEqual(result.weighted_hazard_term, expected_hazard_term, places=15)
        self.assertAlmostEqual(result.weighted_time_term, expected_time_term, places=15)
        self.assertAlmostEqual(
            result.objective_value,
            expected_hazard_term + expected_time_term,
            places=15,
        )
        self.assertAlmostEqual(result.mission_pod, 1.0 - np.exp(-0.4), places=15)
        self.assertEqual(parameters.objective_id, "attacker_hazard_time_v2")

    def test_every_edge_exposes_duration_visibility_resolution_and_samples(self) -> None:
        result = integrate_edge_hazard(
            self.crossing_edge,
            self.los_grid,
            self.detection_model,
            quadrature_resolution=16,
        )
        self.assertEqual(result.duration_s, self.crossing_edge.duration_s)
        self.assertEqual(result.quadrature_resolution, 16)
        self.assertEqual(len(result.samples), 16)
        self.assertEqual(
            result.visible_sample_count,
            sum(sample.visible for sample in result.samples),
        )
        self.assertAlmostEqual(
            result.hazard,
            sum(sample.hazard_contribution for sample in result.samples),
            places=15,
        )
        self.assertTrue(all(not sample.position_map.flags.writeable for sample in result.samples))

    def test_hazard_runtime_is_returned_separately(self) -> None:
        precomputed = precompute_edge_hazards(
            ((self.occluded_edge, self.crossing_edge, self.visible_edge),),
            self.los_grid,
            self.detection_model,
            quadrature_resolution=8,
        )
        self.assertIsInstance(precomputed.timing, HazardTiming)
        self.assertEqual(precomputed.timing.edge_count, 3)
        self.assertEqual(precomputed.timing.quadrature_resolution, 8)
        self.assertGreaterEqual(precomputed.timing.precompute_s, 0.0)
        self.assertGreaterEqual(precomputed.timing.per_edge_s, 0.0)
        self.assertEqual(len(precomputed.hazard_by_source[0]), 3)

    def test_hazard_modules_do_not_import_bellman_recursion(self) -> None:
        directory = Path(__file__).resolve().parent
        imported_modules: set[str] = set()
        for filename in ("detection_hazard.py", "edge_hazard.py"):
            tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "bellman_solver",
            "bellman_graph",
            "switching_candidates",
            "candidate_energy",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "defender",
        ):
            self.assertFalse(any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for module in imported_modules
            ))


if __name__ == "__main__":
    unittest.main()
