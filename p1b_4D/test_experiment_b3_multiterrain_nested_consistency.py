"""Structural tests for the B3 experiment driver (no production solves)."""
from __future__ import annotations

from pathlib import Path
import unittest

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import build_direction_b_configuration
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    COMMON_EVALUATOR_SAMPLE_COUNTS,
    build_b2_case_matrix,
)
from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    B2_COMMON_EVALUATOR_SAMPLE_COUNT,
    B3_TERRAIN_SPECIFICATIONS,
    _select_b3_global_evaluator,
    build_b3_physical_configuration,
)
from p1b_4D.phase_logging import close_phase_logger


class B3ExperimentDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = build_configuration_bundle(Path.cwd())

    @classmethod
    def tearDownClass(cls) -> None:
        close_phase_logger(
            cls.base["primary_result"]["logging_utilities"]["logger"]
        )

    def test_scope_has_two_terrains_and_two_fixed_candidates_each(self) -> None:
        self.assertEqual(
            set(B3_TERRAIN_SPECIFICATIONS),
            {"single_hill", "goal_in_valley"},
        )
        for specification in B3_TERRAIN_SPECIFICATIONS.values():
            self.assertEqual(
                set(specification["sensor_candidates"]),
                {"coverage", "stackelberg"},
            )
        self.assertEqual(len(build_b2_case_matrix()), 9)

    def test_every_b3_level_uses_frozen_nested_grid_counts(self) -> None:
        expected = {
            "single_hill": ((161, 101), (321, 201), (641, 401)),
            "goal_in_valley": ((117, 51), (233, 101), (465, 201)),
        }
        for terrain_name, terrain_specification in (
            B3_TERRAIN_SPECIFICATIONS.items()
        ):
            sensor_z = terrain_specification["sensor_candidates"]["coverage"]
            physical = build_b3_physical_configuration(
                self.base, terrain_name, sensor_z
            )
            for level, expected_counts in enumerate(expected[terrain_name]):
                configured = build_direction_b_configuration(
                    physical, terrain_name, level
                )
                grid = configured["primary_result"]["environment_config"][
                    "grid"
                ]
                self.assertEqual(
                    (grid["z_count"], grid["h_count"]), expected_counts
                )

    def test_global_evaluator_requires_every_terrain_policy_to_pass(self) -> None:
        pairs = tuple(zip(
            COMMON_EVALUATOR_SAMPLE_COUNTS[:-1],
            COMMON_EVALUATOR_SAMPLE_COUNTS[1:],
        ))
        cases = []
        for terrain_index, terrain_name in enumerate(
            B3_TERRAIN_SPECIFICATIONS
        ):
            qualifications = {
                f"{lower}_vs_{upper}": {
                    "passed": lower >= 513 + 512 * terrain_index
                }
                for lower, upper in pairs
            }
            cases.append({
                "terrain_name": terrain_name,
                "sensor_name": "coverage",
                "case_id": "policy",
                "status": "feasible",
                "high_fidelity": {"qualifications": qualifications},
            })
        sample_count, gate = _select_b3_global_evaluator(cases)
        self.assertEqual(sample_count, 1025)
        self.assertEqual(sample_count, B2_COMMON_EVALUATOR_SAMPLE_COUNT)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["feasible_policy_count"], 2)


if __name__ == "__main__":
    unittest.main()
