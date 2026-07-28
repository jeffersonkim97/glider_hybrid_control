"""Regression coverage for successor-grid time-history reconstruction."""
from __future__ import annotations

import unittest

import numpy as np

from p1b_4D.visualization import _reconstruct_mission_hazard_history


class SuccessorGridVisualizationTimingTests(unittest.TestCase):
    def test_duration_profile_controls_glide_time_axis(self) -> None:
        detection = {
            "range_floor": 1.0,
            "acoustic_coefficient": 0.0,
            "acoustic_speed_exponent": 1.0,
            "acoustic_rate_scale": 1.0,
            "rcs_min": 1.0,
            "rcs_max": 1.0,
            "radar_coefficient": 1.0,
            "radar_rate_scale": 1.0,
            "doppler_coefficient": 0.0,
            "radial_velocity_rate_scale": 1.0,
        }
        arrays = {
            "optimal_powered_path": np.array([[0.0, 0.0], [10.0, 0.0]]),
            "optimal_trajectory": np.array(
                [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
            ),
            "optimal_velocity_profile": np.array([10.0, 10.0]),
            "optimal_gamma_profile": np.array([0.0, 0.0]),
            "optimal_duration_profile": np.array([2.0, 3.0]),
            "final_sensor_position": np.array([100.0, 0.0]),
            "final_terrain_z": np.array([0.0, 100.0]),
            "final_terrain_h_grid": np.array([0.0, 10.0]),
            "final_los_mask": np.ones((2, 2), dtype=bool),
        }
        manifest = {
            "configuration": {
                "sensor_config": {"detection": detection},
                "vehicle_config": {
                    "time_step": 99.0,
                    "powered_speed": 10.0,
                },
                "attacker_solver_config": {
                    "transition_model": "successor_grid_physical_edge",
                    "successor_grid": {"edge_quadrature_count": 3},
                },
            }
        }
        history = _reconstruct_mission_hazard_history({
            "stackelberg": {"arrays": arrays, "manifest": manifest}
        })
        np.testing.assert_allclose(
            history["state_times"], np.array([0.0, 1.0, 3.0, 6.0])
        )
        self.assertAlmostEqual(history["switching_time"], 1.0)
        self.assertAlmostEqual(history["times"][-1], 6.0)
        self.assertEqual(history["action_times"].shape, history["v"].shape)
        self.assertEqual(history["action_times"].shape, history["gamma"].shape)


if __name__ == "__main__":
    unittest.main()
