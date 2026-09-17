"""Stage-11 centralized resolution and configurable-heading tests."""

from __future__ import annotations

from dataclasses import replace
from math import atan2
import unittest

import numpy as np

from bellman_geometry import GlideTransitionModel
from bellman_state import BellmanState, BellmanStateGrid
from discretization_config import (
    DiscretizationConfig,
    discretization_from_physical_steps,
    minimum_motion_primitive_radius,
)
from energy_model import DEFAULT_GLIDER
from map_geometry import MapBounds


class _FlatTerrain:
    ground_z = 0.0

    def __init__(self, bounds: MapBounds) -> None:
        self.bounds = bounds

    @property
    def maximum_height(self) -> float:
        return 0.0

    def contains_solid(self, point: np.ndarray, *, tolerance: float = 0.0) -> bool:
        return False

    def segment_intersects_solid(
        self, start: np.ndarray, end: np.ndarray, *, tolerance: float = 1.0e-9,
    ) -> bool:
        return False


class DiscretizationConfigTests(unittest.TestCase):
    def test_default_stencil_exactly_preserves_original_eight_directions(self) -> None:
        grid = DiscretizationConfig().build_bellman_grid(
            MapBounds(-2.0, 2.0, -2.0, 2.0),
        )
        self.assertEqual(grid.motion_offsets, (
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1),
        ))
        np.testing.assert_allclose(
            [grid.heading_rad(index) for index in range(8)],
            [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4,
             np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4],
            rtol=0.0, atol=1.0e-15,
        )

    def test_sixteen_and_thirty_two_heading_stencils_are_configurable(self) -> None:
        bounds = MapBounds(-4.0, 4.0, -4.0, 4.0)
        for count, radius in ((16, 2), (32, 3)):
            with self.subTest(count=count):
                grid = BellmanStateGrid(
                    bounds,
                    maximum_altitude_map=0.2,
                    heading_bin_count=count,
                    motion_primitive_radius=radius,
                )
                self.assertEqual(len(grid.motion_offsets), count)
                self.assertEqual(len(set(grid.motion_offsets)), count)
                self.assertLessEqual(
                    max(max(abs(x), abs(y)) for x, y in grid.motion_offsets),
                    radius,
                )
                for index, (x_offset, y_offset) in enumerate(grid.motion_offsets):
                    self.assertAlmostEqual(
                        grid.heading_rad(index), atan2(y_offset, x_offset), places=15,
                    )

    def test_transition_count_comes_from_configured_stencil(self) -> None:
        bounds = MapBounds(-4.0, 4.0, -4.0, 4.0)
        grid = BellmanStateGrid(
            bounds,
            maximum_altitude_map=1.0,
            heading_bin_count=16,
            motion_primitive_radius=2,
        )
        model = GlideTransitionModel(
            grid,
            _FlatTerrain(bounds),
            parameters=replace(DEFAULT_GLIDER, maximum_bank_deg=89.0),
        )
        edges, statistics, _ = model.successors(BellmanState(4, 4, 10, 0))
        self.assertEqual(statistics.considered, 16)
        self.assertEqual(len(edges), 16)
        self.assertEqual(
            {edge.target_state.heading_bin for edge in edges}, set(range(16)),
        )

    def test_all_named_resolutions_round_trip_to_report(self) -> None:
        config = DiscretizationConfig(
            horizontal_spacing_map=0.5,
            altitude_spacing_map=0.05,
            heading_bin_count=16,
            motion_primitive_radius=2,
            motion_primitive_step_cells=2,
            switching_contour_sample_count=24,
            switching_radial_sample_count=12,
            hazard_quadrature_resolution=16,
        )
        report = config.as_dict()
        self.assertEqual(report["horizontal_spacing_map"], 0.5)
        self.assertEqual(report["heading_bin_count"], 16)
        self.assertEqual(len(report["switching_radial_scales"]), 12)
        grid = config.build_bellman_grid(MapBounds(-2.0, 2.0, -2.0, 2.0))
        self.assertEqual(grid.heading_bin_count, 16)
        self.assertEqual(grid.motion_primitive_radius, 2)
        self.assertEqual(grid.motion_primitive_step_cells, 2)

    def test_step_cells_preserve_physical_horizon_under_spatial_refinement(self) -> None:
        bounds = MapBounds(-2.0, 2.0, -2.0, 2.0)
        baseline = DiscretizationConfig().build_bellman_grid(bounds)
        fine = DiscretizationConfig(
            horizontal_spacing_map=0.5,
            altitude_spacing_map=0.05,
            motion_primitive_step_cells=2,
        ).build_bellman_grid(bounds)
        self.assertEqual(
            fine.motion_offsets,
            tuple((2 * x, 2 * y) for x, y in baseline.motion_offsets),
        )
        baseline_lengths = sorted(
            baseline.horizontal_spacing_map * np.hypot(x, y)
            for x, y in baseline.motion_offsets
        )
        fine_lengths = sorted(
            fine.horizontal_spacing_map * np.hypot(x, y)
            for x, y in fine.motion_offsets
        )
        np.testing.assert_allclose(fine_lengths, baseline_lengths)

    def test_radius_must_supply_enough_unique_directions(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer unique lattice directions"):
            BellmanStateGrid(
                MapBounds(-2.0, 2.0, -2.0, 2.0),
                heading_bin_count=16,
                motion_primitive_radius=1,
            )

    def test_gui_physical_steps_build_complete_five_degree_grid(self) -> None:
        config = discretization_from_physical_steps(25.0, 25.0, 5.0)
        self.assertEqual(config.horizontal_spacing_map, 0.25)
        self.assertEqual(config.altitude_spacing_map, 0.25)
        self.assertEqual(config.heading_bin_count, 72)
        self.assertEqual(
            config.motion_primitive_radius,
            minimum_motion_primitive_radius(72),
        )
        grid = config.build_bellman_grid(MapBounds(-8.0, 8.0, -4.0, 4.0))
        self.assertEqual(grid.state_count, 3_243_240)

    def test_gui_heading_step_must_divide_circle(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide 360"):
            discretization_from_physical_steps(25.0, 25.0, 7.0)


if __name__ == "__main__":
    unittest.main()
