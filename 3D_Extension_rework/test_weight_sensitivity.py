"""Tests for cached Defender-weight sensitivity analysis."""

from __future__ import annotations

import unittest

from .weight_sensitivity import defender_score, select_cached_candidate


class WeightSensitivityTests(unittest.TestCase):
    def test_score_uses_complementary_weights(self) -> None:
        result = defender_score(0.02, 0.60, 0.10)
        self.assertAlmostEqual(result["w_pod"], 0.90)
        self.assertAlmostEqual(result["defender_objective"], 0.078)

    def test_candidate_selection_changes_with_weight(self) -> None:
        center = {
            "feasible": True, "sensor_position_m": [2500.0, 500.0, 8.8],
            "defender_pod_normalized": 0.0244,
            "coverage_volume_normalized": 0.5928, "mission_pod": 0.0247,
            "attacker_objective": 0.337,
            "switching_point_m": [900.0, 300.0, 333.0],
        }
        edge = {
            "feasible": True, "sensor_position_m": [2500.0, 0.0, 4.0],
            "defender_pod_normalized": 0.0075,
            "coverage_volume_normalized": 0.6539, "mission_pod": 0.0075,
            "attacker_objective": 0.317,
            "switching_point_m": [800.0, 500.0, 304.0],
        }
        self.assertEqual(
            select_cached_candidate([center, edge], 0.10)["sensor_position_m"],
            center["sensor_position_m"],
        )
        self.assertEqual(
            select_cached_candidate([center, edge], 0.50)["sensor_position_m"],
            edge["sensor_position_m"],
        )


if __name__ == "__main__":
    unittest.main()
