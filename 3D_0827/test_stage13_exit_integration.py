"""Stage-13 measured scaling-preparation exit gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_13_scaling_preparation"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage13ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (OUTPUT / "stage13_summary.json").read_text(encoding="utf-8")
        )
        cls.records = json.loads(
            (OUTPUT / "benchmark_results.json").read_text(encoding="utf-8")
        )
        cls.failures = json.loads(
            (OUTPUT / "benchmark_failures.json").read_text(encoding="utf-8")
        )
        cls.configurations = json.loads(
            (OUTPUT / "benchmark_configurations.json").read_text(encoding="utf-8")
        )

    def test_all_requested_profiles_are_accounted_for(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertEqual(self.summary["profile_count"], 7)
        self.assertEqual(self.summary["valid_result_count"], 6)
        self.assertEqual(self.summary["exact_infeasible_count"], 1)
        accounted = {item["run_id"] for item in self.records} | {
            item["run_id"] for item in self.failures
        }
        self.assertEqual(accounted, set(self.summary["run_ids"]))

    def test_required_machine_readable_columns_and_exactness_pass(self) -> None:
        required = {
            "run_id", "terrain", "grid_dx", "grid_dy",
            "altitude_resolution", "heading_bins", "state_count",
            "edge_count", "switch_candidate_count", "defender_action_count",
            "t_graph_build_s", "t_los_s", "t_hazard_s",
            "t_attacker_br_s", "t_sse_s", "peak_memory_bytes",
            "attacker_objective", "defender_payoff", "validation_passed",
        }
        for record in self.records:
            self.assertTrue(required.issubset(record), record["run_id"])
            self.assertTrue(record["validation_passed"], record["run_id"])
            self.assertTrue(record["exhaustive_attacker_verified"])
            self.assertTrue(record["exhaustive_defender_verified"])
            self.assertGreater(record["peak_memory_delta_bytes"], 0)
        with (OUTPUT / "benchmark_results.csv").open(
            newline="", encoding="utf-8",
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), len(self.records))
        self.assertTrue(required.issubset(csv_rows[0]))

    def test_profiles_are_isolated_and_memory_runs_start_clean(self) -> None:
        first_repeat_starts = [
            record["repeats"][0]["process_rss_start_bytes"]
            for record in self.records
        ]
        self.assertTrue(all(value < 80 * 2**20 for value in first_repeat_starts))
        by_id = {item["run_id"]: item for item in self.configurations}
        fine = by_id["spatial_fine_50m_5m"]["discretization"]
        self.assertEqual(fine["horizontal_spacing_map"], 0.5)
        self.assertEqual(fine["altitude_spacing_map"], 0.05)
        self.assertEqual(fine["motion_primitive_step_cells"], 2)
        heading = by_id["heading_12"]["discretization"]
        self.assertEqual(heading["heading_bin_count"], 12)
        self.assertEqual(heading["motion_primitive_radius"], 2)

    def test_baseline_repeat_and_frozen_result_are_deterministic(self) -> None:
        baseline = next(
            item for item in self.records if item["run_id"] == "baseline_repeat"
        )
        self.assertEqual(baseline["repeat_count"], 2)
        self.assertTrue(baseline["deterministic"])
        self.assertEqual(baseline["selected_attacker_candidate_id"], 34)
        self.assertAlmostEqual(
            baseline["attacker_objective"], 0.2135615223650064, places=14,
        )
        self.assertAlmostEqual(
            baseline["defender_payoff"], 0.017098090252410693, places=14,
        )

    def test_spatial_scale_and_all_terrain_categories_are_exercised(self) -> None:
        fine = next(
            item for item in self.records
            if item["run_id"] == "spatial_fine_50m_5m"
        )
        self.assertGreater(fine["state_count"], 400_000)
        self.assertGreater(fine["edge_count"], 1_000_000)
        self.assertTrue(fine["validation_passed"])
        self.assertEqual(
            set(self.summary["terrain_categories_attempted"]),
            {"centered_cube", "offset_cube_left", "stepped_pyramid"},
        )

    def test_stepped_pyramid_infeasibility_is_exhaustively_explained(self) -> None:
        self.assertEqual(len(self.failures), 1)
        failure = self.failures[0]
        self.assertEqual(failure["terrain"], "stepped_pyramid")
        self.assertEqual(failure["status"], "exact_finite_infeasible")
        self.assertTrue(failure["validation_passed"])
        self.assertFalse(failure["trajectory_replay_applicable"])
        action = failure["actions"][0]
        self.assertEqual(action["candidate_count"], 96)
        self.assertEqual(action["feasible_candidate_count"], 0)
        self.assertTrue(action["candidate_ids_unique"])
        self.assertEqual(sum(action["status_counts"].values()), 96)
        self.assertIsNone(failure["attacker_objective"])
        self.assertIsNone(failure["defender_payoff"])

    def test_measured_figures_exist_and_original_notebook_is_frozen(self) -> None:
        for filename in (
            "runtime_vs_state_count.html",
            "defender_count_vs_runtime.html",
            "memory_vs_state_count.html",
            "resolution_vs_objective.html",
            "runtime_decomposition.html",
            "heading_vs_runtime.html",
        ):
            path = OUTPUT / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 1_000, filename)
        self.assertEqual(
            hashlib.sha256((ROOT / "3D_bellman_0827.ipynb").read_bytes()).hexdigest(),
            ORIGINAL_NOTEBOOK_SHA256,
        )
        self.assertFalse(self.summary["rl_implemented"])


if __name__ == "__main__":
    unittest.main()
