"""Stage-10 independent replay integration/regression exit gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
FIGURE_DIRECTORY = ROOT / "figure" / "stage_10_independent_replay"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage10ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (FIGURE_DIRECTORY / "trajectory_validation_report.json").read_text(
                encoding="utf-8",
            )
        )
        cls.injections = json.loads(
            (FIGURE_DIRECTORY / "injected_failure_reports.json").read_text(
                encoding="utf-8",
            )
        )

    def test_selected_trajectory_passes_every_required_check(self) -> None:
        replay = self.summary["independent_replay"]
        report = replay["report"]
        self.assertTrue(report["passed"])
        for field in (
            "terrain_clear", "los_phase_valid", "turn_constraints_valid",
            "energy_valid", "time_consistent", "hazard_consistent", "goal_valid",
        ):
            self.assertTrue(report[field], field)
        self.assertEqual(report["max_turn_violation"], 0.0)
        self.assertIsNone(report["min_terrain_clearance"])
        self.assertEqual(report["goal_error_m"], 0.0)

    def test_optimizer_and_independent_values_match_tolerances(self) -> None:
        optimizer = self.summary["optimizer"]
        replay = self.summary["independent_replay"]
        report = replay["report"]
        tolerances = json.loads(
            (FIGURE_DIRECTORY / "replay_tolerances.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertLessEqual(report["time_error_s"], tolerances["time_s"])
        self.assertLessEqual(report["hazard_error"], tolerances["hazard"])
        self.assertAlmostEqual(
            optimizer["mission_time_s"], replay["recomputed_mission_time_s"], places=12,
        )
        self.assertAlmostEqual(
            optimizer["cumulative_hazard"],
            replay["recomputed_cumulative_hazard"],
            places=14,
        )
        self.assertEqual(tolerances["configured_goal_tolerance_m"], 25.0)

    def test_all_injected_failures_are_detected(self) -> None:
        self.assertFalse(
            self.injections["R2_injected_terrain_collision"]["report"]["terrain_clear"]
        )
        self.assertFalse(
            self.injections["R3_injected_turn_violation"]["report"]
            ["turn_constraints_valid"]
        )
        self.assertFalse(
            self.injections["R4_injected_time_inconsistency"]["report"]
            ["time_consistent"]
        )
        self.assertFalse(
            self.injections["R5_injected_hazard_inconsistency"]["report"]
            ["hazard_consistent"]
        )
        self.assertTrue(self.injections["R6_goal_boundary"]["inside"])
        self.assertTrue(self.injections["R6_goal_boundary"]["boundary"])
        self.assertFalse(self.injections["R6_goal_boundary"]["outside"])
        self.assertTrue(
            self.injections["R7_energy_boundary"]["within_tolerance_valid"]
        )
        self.assertFalse(
            self.injections["R7_energy_boundary"]["outside_tolerance_valid"]
        )

    def test_segment_audit_is_complete_and_continuous(self) -> None:
        with (FIGURE_DIRECTORY / "independent_segment_replay.csv").open(
            newline="", encoding="utf-8",
        ) as handle:
            rows = tuple(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["phase"], "powered")
        self.assertEqual(rows[1]["phase"], "virtual")
        self.assertTrue(all(row["phase"] == "glide" for row in rows[2:]))
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual(previous["end_x_map"], current["start_x_map"])
            self.assertEqual(previous["end_y_map"], current["start_y_map"])
            self.assertEqual(previous["end_z_map"], current["start_z_map"])
        self.assertTrue(all(row["terrain_clear"] == "True" for row in rows))

    def test_validation_figures_show_pass_failure_and_goal_sphere(self) -> None:
        valid = (FIGURE_DIRECTORY / "valid_selected_trajectory.html").read_text(
            encoding="utf-8",
        )
        for label in (
            "Independent Continuous Trajectory Replay", "powered 0 (PASS)",
            "virtual 1 (PASS)", "glide 2 (PASS)", "25 m goal tolerance sphere",
        ):
            self.assertIn(label, valid)
        failure = (FIGURE_DIRECTORY / "injected_collision_failure.html").read_text(
            encoding="utf-8",
        )
        self.assertIn("First validation failure", failure)
        self.assertIn("FAIL", failure)

    def test_validation_path_is_independent_and_terrain_generic(self) -> None:
        independence = self.summary["independence"]
        self.assertFalse(independence["bellman_value_or_policy_used_during_validation"])
        self.assertFalse(independence["mission_response_energy_certificate_reused"])
        self.assertFalse(independence["edge_hazard_integrator_reused"])
        source = (ROOT / "trajectory_validation.py").read_text(encoding="utf-8")
        for forbidden in (
            "CubeObstacle", "TerrainMap3D", "from mission_response import",
            "from edge_hazard import", "from additive_bellman import",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("segment_intersects_solid", source)

    def test_original_gui_notebook_remains_frozen(self) -> None:
        digest = hashlib.sha256(
            (ROOT / "3D_bellman_0827.ipynb").read_bytes()
        ).hexdigest()
        self.assertEqual(digest, ORIGINAL_NOTEBOOK_SHA256)


if __name__ == "__main__":
    unittest.main()
