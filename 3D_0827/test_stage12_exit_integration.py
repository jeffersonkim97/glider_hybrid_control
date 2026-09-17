"""Stage-12 finite Stackelberg numerical/visual exit gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
FIGURE_DIRECTORY = ROOT / "figure" / "stage_12_finite_stackelberg"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage12ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (FIGURE_DIRECTORY / "stackelberg_summary.json").read_text(
                encoding="utf-8",
            )
        )
        with (FIGURE_DIRECTORY / "defender_results.csv").open(
            newline="", encoding="utf-8",
        ) as handle:
            cls.rows = tuple(csv.DictReader(handle))

    def test_geometry_backed_three_action_game_is_fully_defined(self) -> None:
        self.assertEqual(len(self.rows), 3)
        self.assertEqual(
            [float(row["sensor_x_map"]) for row in self.rows],
            [5.0, 7.0, 7.5],
        )
        self.assertTrue(all(row["feasible"] == "True" for row in self.rows))
        self.assertTrue(all(row["attacker_candidate_id"] for row in self.rows))

    def test_literal_leader_maximum_selects_d2(self) -> None:
        selected = self.summary["selected"]
        payoffs = [float(row["detection_probability"]) for row in self.rows]
        self.assertEqual(selected["defender_action_id"], 2)
        self.assertEqual(selected["sensor_position_map"], [7.5, 0.0, 0.0])
        self.assertAlmostEqual(selected["defender_payoff_pod"], max(payoffs), places=14)
        self.assertTrue(
            self.summary["literal_exhaustive_check"]
            ["all_defender_actions_evaluated"]
        )
        self.assertTrue(
            self.summary["literal_exhaustive_check"]
            ["selected_payoff_equals_maximum"]
        )

    def test_sse_follower_tie_is_applied_to_geometry(self) -> None:
        first = self.summary["defender_actions"][0]
        self.assertEqual(first["attacker_cooptimal_candidate_ids"], [25, 34])
        self.assertEqual(first["attacker_candidate_id"], 34)
        self.assertIn("maximize Defender PoD", self.summary["payoff_convention"]["follower_tie"])

    def test_selected_equilibrium_values_are_regressed(self) -> None:
        selected = self.summary["selected"]
        self.assertEqual(selected["attacker_candidate_id"], 25)
        self.assertAlmostEqual(selected["attacker_objective"], 1.0912385246088796, places=12)
        self.assertAlmostEqual(selected["defender_payoff_pod"], 0.8325883732933018, places=13)
        self.assertAlmostEqual(selected["mission_time_s"], 87.42862400860827, places=11)
        self.assertAlmostEqual(selected["cumulative_hazard"], 1.78729966869885, places=13)

    def test_selected_continuous_trajectory_passes_independent_replay(self) -> None:
        validation = self.summary["validation"]
        report = validation["report"]
        for field in (
            "passed", "terrain_clear", "los_phase_valid",
            "turn_constraints_valid", "energy_valid", "time_consistent",
            "hazard_consistent", "goal_valid",
        ):
            self.assertTrue(report[field], field)
        self.assertLessEqual(report["time_error_s"], 1.0e-9)
        self.assertLessEqual(report["hazard_error"], 1.0e-10)
        self.assertEqual(validation["goal_distance_m"], 20.0)

    def test_repeated_geometry_run_is_deterministic(self) -> None:
        determinism = self.summary["determinism"]
        self.assertTrue(determinism["repeated"])
        self.assertTrue(determinism["passed"])
        self.assertGreater(determinism["second_run_total_s"], 0.0)

    def test_runtime_is_stored_per_defender_and_in_total(self) -> None:
        timing = self.summary["timing"]
        self.assertEqual(len(timing["per_defender_action_s"]), 3)
        self.assertTrue(all(value > 0.0 for value in timing["per_defender_action_s"]))
        self.assertGreater(timing["total_stackelberg_s"], 0.0)
        self.assertGreaterEqual(
            timing["total_stackelberg_s"], sum(timing["per_defender_action_s"]),
        )

    def test_visual_diagnostics_contain_equilibrium_layers(self) -> None:
        solution = (FIGURE_DIRECTORY / "finite_stackelberg_solution.html").read_text(
            encoding="utf-8",
        )
        for label in (
            "Exact Finite Stackelberg Solution", "Finite Defender actions",
            "Selected Defender D2", "STACKELBERG DEFENDER",
            "Selected powered", "Selected virtual", "Selected Bellman glide",
        ):
            self.assertIn(label, solution)
        comparison = (
            FIGURE_DIRECTORY / "defender_payoff_comparison.html"
        ).read_text(encoding="utf-8")
        self.assertIn("Defender payoff (PoD)", comparison)
        self.assertIn("Attacker objective", comparison)

    def test_bellman_layer_has_no_defender_enumeration_dependency(self) -> None:
        for filename in (
            "bellman_state.py", "bellman_geometry.py", "bellman_graph.py",
            "additive_bellman.py", "bellman_solver.py",
        ):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertNotIn("stackelberg_solver", source, filename)
            self.assertNotIn("DefenderCandidate", source, filename)

    def test_original_gui_notebook_remains_unchanged(self) -> None:
        original = hashlib.sha256((ROOT / "3D_bellman_0827.ipynb").read_bytes()).hexdigest()
        self.assertEqual(original, ORIGINAL_NOTEBOOK_SHA256)


if __name__ == "__main__":
    unittest.main()
