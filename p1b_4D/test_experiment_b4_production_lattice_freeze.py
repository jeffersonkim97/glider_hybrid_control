"""Regression gates for the B4 production-lattice freeze."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import (
    DIRECTION_B_GRID_COUNTS,
    DIRECTION_B_PRODUCTION_CONFIGURATION_ID,
    build_direction_b_production_configuration,
    construct_direction_b_grids,
)
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    B2_SENSOR_CANDIDATES,
    build_two_hill_configuration,
)
from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    B3_TERRAIN_SPECIFICATIONS,
    build_b3_physical_configuration,
)
from p1b_4D.experiment_b4_production_lattice_freeze import (
    _configuration_summary,
    _select_global_settings,
)
from p1b_4D.phase_logging import close_phase_logger


class B4ProductionLatticeFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        self.addCleanup(
            close_phase_logger,
            self.base["primary_result"]["logging_utilities"]["logger"],
        )

    def test_production_factory_freezes_exact_settings_on_every_terrain(self):
        for terrain_name in DIRECTION_B_GRID_COUNTS:
            if terrain_name == "two_hill":
                physical = build_two_hill_configuration(
                    self.base, B2_SENSOR_CANDIDATES["coverage"]
                )
            else:
                specification = B3_TERRAIN_SPECIFICATIONS[terrain_name]
                physical = build_b3_physical_configuration(
                    self.base,
                    terrain_name,
                    specification["sensor_candidates"]["coverage"],
                )
            bundle = build_direction_b_production_configuration(
                physical, terrain_name
            )
            protocol = bundle["primary_result"]["direction_b_protocol"]
            grid = bundle["primary_result"]["environment_config"]["grid"]
            self.assertEqual(
                (grid["z_count"], grid["h_count"]),
                DIRECTION_B_GRID_COUNTS[terrain_name][2],
            )
            self.assertEqual(protocol["level"], 2)
            self.assertEqual(protocol["action_family"], "enriched")
            self.assertEqual(protocol["speed_family"], "V9")
            self.assertEqual(protocol["planning_quadrature_count"], 9)
            self.assertEqual(protocol["common_evaluator_sample_count"], 1025)
            self.assertTrue(protocol["production_frozen"])
            self.assertEqual(
                protocol["production_configuration_id"],
                DIRECTION_B_PRODUCTION_CONFIGURATION_ID,
            )
            self.assertFalse(protocol["continuous_optimum_claimed"])

    def test_production_summary_has_33_directions_and_297_actions(self):
        terrain_name = "single_hill"
        specification = B3_TERRAIN_SPECIFICATIONS[terrain_name]
        physical = build_b3_physical_configuration(
            self.base,
            terrain_name,
            specification["sensor_candidates"]["coverage"],
        )
        bundle = build_direction_b_production_configuration(
            physical, terrain_name
        )
        summary = _configuration_summary(bundle)
        self.assertEqual(
            summary["movement"]["displacement_direction_count"], 33
        )
        self.assertEqual(summary["movement"]["speed_count"], 9)
        self.assertEqual(summary["movement"]["regular_action_count"], 297)
        self.assertEqual(construct_direction_b_grids(bundle)["v"].size, 9)

    def test_global_selection_uses_most_conservative_terrain_requirement(self):
        b2 = _synthetic_b2()
        b3 = _synthetic_b3()
        result = _select_global_settings(b2, b3)
        selected = result["selected"]
        self.assertEqual(selected["level"], 2)
        self.assertEqual(selected["action_family"], "enriched")
        self.assertEqual(selected["speed_family"], "V9")
        self.assertEqual(selected["planning_quadrature_count"], 9)
        self.assertEqual(selected["evaluator_sample_count"], 1025)


def _analysis(speed_choice, quadrature_choice, speed_value, tau):
    return {
        "speed_sensitivity": {
            "coverage": {"l1": speed_value, "l2": speed_value},
            "stackelberg": {"l1": speed_value, "l2": speed_value},
        },
        "quadrature_sensitivity": {"coverage": 0.0, "stackelberg": 0.0},
        "tau_b": tau,
        "production_speed_family": speed_choice,
        "production_planning_quadrature_count": quadrature_choice,
        "production_speed_family_if_terrain_only": speed_choice,
        "production_planning_quadrature_count_if_terrain_only": (
            quadrature_choice
        ),
        "ranking_diagnostically_resolved": False,
    }


def _synthetic_b2():
    return {
        "global_common_evaluator_sample_count": 1025,
        "analysis": _analysis("V9", 9, 0.002, 0.001),
    }


def _synthetic_b3():
    return {
        "global_common_evaluator_sample_count": 1025,
        "analysis": {
            "single_hill": _analysis("V5", 9, 0.0, 0.001),
            "goal_in_valley": _analysis("V5", 9, 0.0, 0.001),
        },
    }


if __name__ == "__main__":
    unittest.main()
