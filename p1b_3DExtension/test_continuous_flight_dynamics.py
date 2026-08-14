"""Unit tests for continuous 3-DOF refinement dynamics."""

from __future__ import annotations

import unittest

import numpy as np

from p1b_3DExtension.configuration import vehicle_config
from p1b_3DExtension.continuous_flight_dynamics import (
    aerodynamic_forces,
    point_mass_rhs_numpy,
    powered_switch_state,
    rk4_step_numpy,
)


class ContinuousFlightDynamicsTests(unittest.TestCase):
    def test_powered_switch_state_preserves_velocity_direction(self) -> None:
        state = powered_switch_state(
            np.array([0.0, 0.0, 5.0]),
            powered_speed=20.0,
            powered_time=10.0,
            flight_path_angle=np.deg2rad(10.0),
            heading=np.deg2rad(-20.0),
        )
        self.assertAlmostEqual(state[3], 20.0)
        self.assertAlmostEqual(state[4], np.deg2rad(10.0))
        self.assertAlmostEqual(state[5], np.deg2rad(-20.0))
        self.assertAlmostEqual(state[6], 0.0)
        self.assertGreater(state[2], 5.0)

    def test_zero_bank_has_zero_heading_rate(self) -> None:
        state = np.array([0.0, 0.0, 100.0, 18.0, -0.05, 0.2, 0.0])
        derivative = point_mass_rhs_numpy(
            state, np.array([0.15, 0.0]), vehicle_config,
        )
        self.assertAlmostEqual(derivative[5], 0.0, places=14)

    def test_positive_bank_turns_toward_positive_heading(self) -> None:
        state = np.array([
            0.0, 0.0, 100.0, 18.0, -0.05, 0.0, np.deg2rad(20.0),
        ])
        derivative = point_mass_rhs_numpy(
            state, np.array([0.15, 0.0]), vehicle_config,
        )
        self.assertGreater(derivative[5], 0.0)

    def test_drag_is_positive_and_rk4_is_finite(self) -> None:
        _, drag, cd = aerodynamic_forces(18.0, 0.15, vehicle_config)
        self.assertGreater(float(drag), 0.0)
        self.assertGreater(float(cd), 0.0)
        state = np.array([0.0, 0.0, 100.0, 18.0, -0.05, 0.0, 0.0])
        advanced = rk4_step_numpy(
            state, np.array([0.15, 0.0]), 0.1, vehicle_config,
        )
        self.assertTrue(np.all(np.isfinite(advanced)))
        self.assertGreater(advanced[0], state[0])


if __name__ == "__main__":
    unittest.main()
