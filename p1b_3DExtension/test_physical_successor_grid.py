"""Regression tests for exact physical 3D successor edges."""

from __future__ import annotations

import unittest

import numpy as np

from p1b_3DExtension.successor_grid_solver import (
    physical_edge_endpoint_residual,
    physical_action_offsets,
)


class PhysicalSuccessorGridTests(unittest.TestCase):
    def test_physical_envelope_is_preserved_across_grids(self) -> None:
        search = {
            "physical_action_envelope": {
                "forward_min_m": 50.0,
                "forward_max_m": 170.0,
                "lateral_max_m": 225.0,
                "descent_min_m": 4.0,
                "descent_max_m": 32.0,
            }
        }
        grids = (
            (np.linspace(0.0, 2750.0, 31), np.linspace(-1500.0, 1500.0, 21), np.linspace(0.0, 200.0, 31)),
            (np.linspace(0.0, 2750.0, 41), np.linspace(-1500.0, 1500.0, 31), np.linspace(0.0, 200.0, 41)),
            (np.linspace(0.0, 2750.0, 51), np.linspace(-1500.0, 1500.0, 41), np.linspace(0.0, 200.0, 51)),
        )
        counts = []
        for x_grid, y_grid, h_grid in grids:
            offsets = physical_action_offsets(x_grid, y_grid, h_grid, search)
            counts.append(len(offsets))
            dx, dy, dh = x_grid[1], y_grid[1] - y_grid[0], h_grid[1]
            for forward, lateral, descent in offsets:
                self.assertGreaterEqual(forward * dx, 50.0 - 1.0e-10)
                self.assertLessEqual(forward * dx, 170.0 + 1.0e-10)
                self.assertLessEqual(abs(lateral * dy), 225.0 + 1.0e-10)
                self.assertGreaterEqual(descent * dh, 4.0 - 1.0e-10)
                self.assertLessEqual(descent * dh, 32.0 + 1.0e-10)
        self.assertEqual(counts, [12, 60, 168])

    def test_fine_envelope_reproduces_original_cell_offsets(self) -> None:
        search = {
            "physical_action_envelope": {
                "forward_min_m": 50.0,
                "forward_max_m": 170.0,
                "lateral_max_m": 225.0,
                "descent_min_m": 4.0,
                "descent_max_m": 32.0,
            }
        }
        offsets = physical_action_offsets(
            np.linspace(0.0, 2750.0, 51),
            np.linspace(-1500.0, 1500.0, 41),
            np.linspace(0.0, 200.0, 51),
            search,
        )
        expected = {
            (forward, lateral, descent)
            for forward in range(1, 4)
            for lateral in range(-3, 4)
            for descent in range(1, 9)
        }
        self.assertEqual(set(offsets), expected)

    def test_exact_edges_reconstruct_their_nodes(self) -> None:
        trajectory = np.array([
            [0.0, 0.0, 100.0],
            [30.0, 40.0, 90.0],
            [80.0, 40.0, 80.0],
        ])
        deltas = np.diff(trajectory, axis=0)
        lengths = np.linalg.norm(deltas, axis=1)
        horizontal = np.linalg.norm(deltas[:, :2], axis=1)
        speeds = np.array([10.0, 20.0])
        durations = lengths / speeds
        gammas = np.arctan2(deltas[:, 2], horizontal)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        residual = physical_edge_endpoint_residual(
            trajectory, speeds, gammas, headings, durations,
        )
        self.assertLessEqual(residual, 1.0e-12)

    def test_snapped_endpoint_is_detected(self) -> None:
        trajectory = np.array([
            [0.0, 0.0, 100.0],
            [50.0, 0.0, 90.0],
        ])
        residual = physical_edge_endpoint_residual(
            trajectory,
            np.array([10.0]),
            np.array([0.0]),
            np.array([0.0]),
            np.array([4.0]),
        )
        self.assertAlmostEqual(residual, np.sqrt(200.0))


if __name__ == "__main__":
    unittest.main()
