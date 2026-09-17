"""Stage 14.2C raw-data, scaling, oracle, plot, and notebook exit gates."""

from __future__ import annotations

import json
import statistics
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2c_local_scaling"


class Stage142CExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (OUTPUT / "stage14_2c_summary.json").read_text(encoding="utf-8")
        )
        cls.validation = json.loads(
            (OUTPUT / "regression_validation_report.json").read_text(encoding="utf-8")
        )
        cls.configurations = json.loads(
            (OUTPUT / "benchmark_configurations.json").read_text(encoding="utf-8")
        )
        cls.rows = json.loads(
            (OUTPUT / "raw_repetitions.json").read_text(encoding="utf-8")
        )
        cls.local_summaries = json.loads(
            (OUTPUT / "local_scaling_summaries.json").read_text(encoding="utf-8")
        )
        cls.all_summaries = json.loads(
            (OUTPUT / "configuration_summaries.json").read_text(encoding="utf-8")
        )
        cls.comparisons = json.loads(
            (OUTPUT / "local_vs_global_oracle.json").read_text(encoding="utf-8")
        )

    def test_reproducible_runner_records_sixty_unique_cold_process_rows(self) -> None:
        self.assertEqual(len(self.rows), 60)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 60)
        self.assertEqual(Counter(row["status"] for row in self.rows), {"completed": 60})
        self.assertTrue(all(row["process_semantics"] == "fresh cold process" for row in self.rows))
        case_counts = Counter(row["case_id"] for row in self.rows)
        self.assertTrue(all(count == 3 for count in case_counts.values()))
        self.assertEqual(len(case_counts), 20)

    def test_four_sweeps_are_one_factor_at_a_time_on_centered_cube(self) -> None:
        self.assertTrue(self.configurations["passed"])
        self.assertTrue(all(self.configurations["checks"].values()))
        local_groups = {
            row["sweep_group"] for row in self.configurations["cases"]
            if row["algorithm_variant"] == "local_sse"
        }
        self.assertEqual(local_groups, {"baseline", "spatial", "heading", "switching", "defender"})
        for row in self.configurations["cases"]:
            parameters = row["parameters"]
            self.assertEqual(parameters["terrain_category"], "centered_cube")
            self.assertFalse(parameters["reinforcement_learning"])
            self.assertFalse(parameters["approximate_planner"])
            self.assertFalse(parameters["multi_start"])

    def test_every_local_repetition_is_certified_exact_and_replayed(self) -> None:
        local = [row for row in self.rows if row["algorithm_variant"] == "local_sse"]
        self.assertEqual(len(local), 30)
        for row in local:
            self.assertTrue(row["local_search"]["local_sse_verified"])
            self.assertTrue(row["exactness"]["all_required_final_neighbors_evaluated"])
            self.assertTrue(row["exactness"]["all_evaluated_attacker_responses_exact"])
            self.assertTrue(row["exactness"]["strong_follower_tie_break_verified"])
            self.assertTrue(row["independent_replay_passed"])
            margin = row["local_search"]["local_optimality_margin"]
            if margin is None:
                self.assertTrue(row["local_search"]["isolated_feasible_local_solution"])
                self.assertEqual(
                    row["local_search"]["termination_status"],
                    "isolated_feasible_local_sse",
                )
            else:
                self.assertGreaterEqual(margin, -1.0e-12)

    def test_local_summary_medians_reproduce_raw_rows(self) -> None:
        self.assertEqual(len(self.local_summaries), 10)
        self.assertTrue(all(row["algorithm_variant"] == "local_sse" for row in self.local_summaries))
        for summary in self.local_summaries:
            raw = [
                row for row in self.rows
                if row["case_id"] == summary["case_id"] and row["status"] == "completed"
            ]
            runtimes = [row["timing"]["totals"]["T_SSE_s"] for row in raw]
            self.assertEqual(len(runtimes), 3)
            self.assertAlmostEqual(
                summary["runtime_statistics_s"]["median"], statistics.median(runtimes), places=12
            )

    def test_all_required_counts_and_local_metrics_are_recorded(self) -> None:
        count_fields = {
            "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
            "N_S_goal_reachable", "N_S_active", "N_E", "N_C", "N_D", "N_eval",
            "B", "Q",
        }
        for summary in self.local_summaries:
            self.assertTrue(count_fields <= set(summary["state_and_game_size"]))
            metrics = summary["local_search_statistics"]
            self.assertIn("unique_defender_evaluations", metrics)
            self.assertIn("local_search_iterations", metrics)
            self.assertIn("mean_neighborhood_size", metrics)
            self.assertIn("cached_evaluation_reuses", metrics)
            self.assertIn("local_optimality_margin", metrics)
            self.assertGreater(summary["peak_rss_statistics_bytes"]["median"], 0)

    def test_global_solver_is_labeled_and_used_only_as_oracle(self) -> None:
        oracle = [row for row in self.rows if row["algorithm_variant"] == "global_oracle"]
        self.assertEqual(len(oracle), 30)
        for row in oracle:
            self.assertEqual(
                row["oracle_metadata"]["role"],
                "tractable finite global oracle/reference only",
            )
            self.assertTrue(row["exactness"]["global_defender_enumeration_complete"])
            self.assertTrue(row["independent_replay_passed"])

    def test_gap_exists_only_for_completed_oracles_and_is_nonnegative(self) -> None:
        self.assertEqual(len(self.comparisons), 10)
        for comparison in self.comparisons:
            if comparison["status"] == "completed":
                self.assertIsNotNone(comparison["delta_J_D"])
                self.assertGreaterEqual(comparison["delta_J_D"], -1.0e-12)
                self.assertIsNone(comparison["unavailable_reason"])
            else:
                self.assertEqual(comparison["status"], "global_oracle_unavailable")
                self.assertIsNone(comparison["delta_J_D"])
                self.assertIsNotNone(comparison["unavailable_reason"])

    def test_geometry_cases_demonstrate_local_is_not_necessarily_global(self) -> None:
        by_profile = {row["profile"]: row for row in self.comparisons}
        for profile in ("defender_count_3", "defender_count_5", "switch_contour_6"):
            self.assertGreater(by_profile[profile]["delta_J_D"], 0.7)
            self.assertNotEqual(
                by_profile[profile]["local_defender_action"],
                by_profile[profile]["global_defender_action"],
            )
        self.assertEqual(by_profile["canonical_anchor"]["delta_J_D"], 0.0)
        self.assertEqual(
            set(self.summary["isolated_local_sse_profiles"]),
            {"defender_count_3", "defender_count_5", "switch_contour_6"},
        )
        self.assertEqual(self.summary["isolated_local_sse_repetition_count"], 9)

    def test_required_ten_figures_and_machine_readable_artifacts_exist(self) -> None:
        self.assertEqual(len(self.summary["figure_files"]), 10)
        self.assertTrue(all(name.endswith(".png") for name in self.summary["figure_files"]))
        required = tuple(self.summary["figure_files"]) + (
            "raw_repetitions.jsonl", "raw_repetitions.json", "raw_repetitions.csv",
            "configuration_summaries.json", "configuration_summaries.csv",
            "local_scaling_summaries.json", "local_scaling_summaries.csv",
            "local_vs_global_oracle.json", "local_vs_global_oracle.csv",
            "regression_validation_report.json", "stage14_2c_summary.json",
        )
        for name in required:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)
            if name.endswith(".png"):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_claim_boundaries_and_all_exit_checks_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertFalse(self.validation["power_law_fit_performed"])
        claims = self.validation["claim_separation"]
        self.assertTrue(claims["local_sse_certification"])
        self.assertTrue(claims["local_sse_computational_scaling"])
        self.assertTrue(claims["global_oracle_gap_only_on_completed_cases"])
        self.assertFalse(claims["bounded_global_approximation_claimed"])
        self.assertFalse(self.summary["interpretation"]["multi_start_performed"])

    def test_notebook_orders_executes_and_labels_stage14_2c(self) -> None:
        notebook = json.loads(
            (ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8")
        )
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-2c-heading"), ids.index("stage14-2b-figure"))
        self.assertLess(ids.index("stage14-2c-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("run_stage14_2c_diagnostics", source)
        self.assertIn("RECOMPUTE_STAGE14_2 = False", source)
        self.assertIn("empirical values", source)
        self.assertNotIn(".show()", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".html", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")
        self.assertTrue(notebook["cells"][ids.index("stage14-2c-summary")]["outputs"])
        self.assertTrue(notebook["cells"][ids.index("stage14-2c-figures")]["outputs"])


if __name__ == "__main__":
    unittest.main()
