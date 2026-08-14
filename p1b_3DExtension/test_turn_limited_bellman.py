"""Focused regression tests for the 3D heading-state extension."""

from __future__ import annotations

import unittest

import numpy as np

from p1b_3DExtension.bellman import solve_coarse_bellman
from p1b_3DExtension.turn_dynamics import (
    heading_change_metrics,
    heading_transition_mask,
    nearest_heading_index,
)


class TurnDynamicsTests(unittest.TestCase):
    def test_periodic_transition_wraps_across_pi(self) -> None:
        grid = np.deg2rad(np.arange(-180.0, 180.0, 10.0))
        mask = heading_transition_mask(
            grid, np.deg2rad(5.0), transition_duration=3.0,
        )
        minus_180 = nearest_heading_index(np.deg2rad(-180.0), grid)
        plus_170 = nearest_heading_index(np.deg2rad(170.0), grid)
        plus_160 = nearest_heading_index(np.deg2rad(160.0), grid)
        self.assertTrue(mask[minus_180, plus_170])
        self.assertFalse(mask[minus_180, plus_160])

    def test_metrics_use_shortest_periodic_change(self) -> None:
        metrics = heading_change_metrics(
            np.deg2rad(170.0),
            np.deg2rad(np.array([-180.0, -170.0])),
            transition_duration=2.0,
        )
        self.assertAlmostEqual(metrics["maximum_heading_change_deg"], 10.0)
        self.assertAlmostEqual(metrics["maximum_turn_rate_deg_s"], 5.0)

    def test_bellman_value_depends_on_incoming_heading(self) -> None:
        headings = np.deg2rad(np.array([-90.0, 0.0, 90.0, 180.0]))
        grids = {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0]),
            "h": np.array([0.0, 1.0, 2.0]),
            "v": np.array([1.0]),
            "gamma": np.array([-0.1]),
            "heading": headings,
        }
        shape = (3, 1, 3, 1, 1, 4)
        transitions = {
            "next_x_index": np.full(shape, -1, dtype=np.int32),
            "next_y_index": np.full(shape, -1, dtype=np.int32),
            "next_h_index": np.full(shape, -1, dtype=np.int32),
            "transition_valid": np.zeros(shape, dtype=bool),
            "terminal_transition": np.zeros(shape, dtype=bool),
            "terminal_fraction": np.ones(shape),
            "coarse_step_count": 1,
        }
        course_zero = 1
        first = (0, 0, 2, 0, 0, course_zero)
        second = (1, 0, 1, 0, 0, course_zero)
        transitions["transition_valid"][first] = True
        transitions["next_x_index"][first] = 1
        transitions["next_y_index"][first] = 0
        transitions["next_h_index"][first] = 1
        transitions["transition_valid"][second] = True
        transitions["terminal_transition"][second] = True
        j6d = np.ones(shape)
        common = dict(
            j6d=j6d,
            transitions=transitions,
            grids=grids,
            goal_position=np.array([2.0, 0.0, 0.0]),
            validation_config={"goal_radius": 0.1},
            exploration_ordering="low_gamma_first",
            bellman_config={"maximum_iterations": 1},
        )
        strict = solve_coarse_bellman(
            **common,
            vehicle_config={
                "time_step": 1.0,
                "turn_dynamics": {"max_turn_rate_deg_s": 10.0},
            },
        )
        self.assertAlmostEqual(strict["value"][0, 0, 2, course_zero], 2.0)
        self.assertTrue(np.isinf(strict["value"][0, 0, 2, 2]))

        permissive = solve_coarse_bellman(
            **common,
            vehicle_config={
                "time_step": 1.0,
                "turn_dynamics": {"max_turn_rate_deg_s": 100.0},
            },
        )
        self.assertAlmostEqual(permissive["value"][0, 0, 2, 2], 2.0)


if __name__ == "__main__":
    unittest.main()
