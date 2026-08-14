"""Contract tests for the 3D strategic-baseline comparison."""

from __future__ import annotations

import unittest

from .strategic_baselines import (
    evaluate_selected_sensors,
    reconcile_stackelberg_candidate,
    select_fixed_sensor,
)
from .terrain_scenarios import build_scenario_configuration


class StrategicBaselineTests(unittest.TestCase):
    def test_fixed_sensor_is_preconfigured_and_attacker_independent(self) -> None:
        configuration = build_scenario_configuration("centered_single_hill")
        self.assertEqual(select_fixed_sensor(configuration), (2125.0, 500.0))

    def test_all_selections_share_one_evaluator(self) -> None:
        calls: list[tuple[float, float]] = []

        def evaluator(x: float, y: float) -> dict:
            calls.append((x, y))
            return {"feasible": True, "defender_objective": x + y}

        results = evaluate_selected_sensors(
            {"fixed": (1.0, 2.0), "coverage_only": (3.0, 4.0)}, evaluator,
        )
        self.assertEqual(calls, [(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(results["coverage_only"]["defender_objective"], 7.0)

    def test_reconciliation_promotes_better_evaluated_baseline(self) -> None:
        stack = {
            "sensor_position_m": [1.0, 2.0, 0.0],
            "defender_objective": 0.2,
            "feasible": True,
        }
        baselines = {
            "fixed": {
                "sensor_position_m": [3.0, 4.0, 0.0],
                "defender_objective": 0.3,
                "feasible": True,
            },
            "failed": {"defender_objective": float("-inf"), "feasible": False},
        }
        reconciled = reconcile_stackelberg_candidate(stack, baselines)
        self.assertTrue(reconciled["promoted"])
        self.assertEqual(reconciled["source"], "fixed")


if __name__ == "__main__":
    unittest.main()
