"""Stage-13 scaling configuration and instrumentation smoke tests."""

from __future__ import annotations

from dataclasses import replace
import time
import unittest

import numpy as np

from scaling_benchmark import (
    ProcessPeakMemorySampler,
    ScalingRunConfig,
    differing_fields,
)


class ScalingSmokeTests(unittest.TestCase):
    def test_scaling_configuration_serializes_every_required_dimension(self) -> None:
        config = ScalingRunConfig(
            run_id="smoke",
            sweep_variable="baseline",
            defender_x_map=(5.0, 7.0),
            repeat_count=2,
        )
        values = config.as_dict()
        self.assertEqual(values["defender_action_count"], 2)
        self.assertEqual(values["repeat_count"], 2)
        for name in (
            "horizontal_spacing_map", "altitude_spacing_map",
            "heading_bin_count", "motion_primitive_radius",
            "motion_primitive_step_cells",
            "switching_contour_sample_count", "hazard_quadrature_resolution",
        ):
            self.assertIn(name, values["discretization"])

    def test_heading_sweep_changes_only_heading_and_motion_stencil(self) -> None:
        baseline = ScalingRunConfig("baseline", "baseline")
        heading = ScalingRunConfig(
            "heading12",
            "heading",
            discretization=replace(
                baseline.discretization,
                heading_bin_count=12,
                motion_primitive_radius=2,
            ),
        )
        self.assertEqual(differing_fields(baseline, heading), (
            "discretization.heading_bin_count",
            "discretization.motion_primitive_radius",
        ))

    def test_spatial_sweep_changes_only_coupled_xy_altitude_resolution(self) -> None:
        baseline = ScalingRunConfig("baseline", "baseline")
        coarse = ScalingRunConfig(
            "coarse", "spatial",
            discretization=replace(
                baseline.discretization,
                horizontal_spacing_map=2.0,
                altitude_spacing_map=0.2,
                motion_primitive_step_cells=2,
            ),
        )
        self.assertEqual(differing_fields(baseline, coarse), (
            "discretization.horizontal_spacing_map",
            "discretization.altitude_spacing_map",
            "discretization.motion_primitive_step_cells",
        ))

    def test_candidate_defender_and_terrain_sweeps_are_isolated(self) -> None:
        baseline = ScalingRunConfig("baseline", "baseline")
        candidate = ScalingRunConfig(
            "candidate", "candidate",
            discretization=replace(
                baseline.discretization,
                switching_contour_sample_count=16,
            ),
        )
        defender = ScalingRunConfig(
            "defender", "defender", defender_x_map=(5.0, 7.0),
        )
        terrain = ScalingRunConfig(
            "terrain", "terrain", terrain_category="stepped_pyramid",
        )
        self.assertEqual(
            differing_fields(baseline, candidate),
            ("discretization.switching_contour_sample_count",),
        )
        self.assertEqual(differing_fields(baseline, defender), ("defender_x_map",))
        self.assertEqual(differing_fields(baseline, terrain), ("terrain_category",))

    def test_process_peak_memory_sampler_reports_actual_rss(self) -> None:
        with ProcessPeakMemorySampler(interval_s=0.001) as sampler:
            allocation = np.ones(1_000_000, dtype=np.float64)
            time.sleep(0.01)
            self.assertEqual(allocation.size, 1_000_000)
        self.assertGreater(sampler.start_rss, 0)
        self.assertGreaterEqual(sampler.peak_rss, sampler.start_rss)
        self.assertEqual(
            sampler.delta_peak_bytes,
            sampler.peak_rss - sampler.start_rss,
        )

    def test_invalid_benchmark_configurations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ScalingRunConfig("", "baseline")
        with self.assertRaises(ValueError):
            ScalingRunConfig("x", "baseline", defender_x_map=())
        with self.assertRaises(ValueError):
            ScalingRunConfig("x", "baseline", repeat_count=0)


if __name__ == "__main__":
    unittest.main()
