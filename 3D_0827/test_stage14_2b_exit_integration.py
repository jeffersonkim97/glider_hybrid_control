"""Stage 14.2B canonical local-search and notebook exit gates."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2b_local_sse_search"


class Stage142BExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (OUTPUT / "stage14_2b_summary.json").read_text(encoding="utf-8")
        )
        cls.result = json.loads(
            (OUTPUT / "canonical_local_sse_result.json").read_text(encoding="utf-8")
        )
        cls.history = json.loads(
            (OUTPUT / "local_search_history.json").read_text(encoding="utf-8")
        )
        cls.worker = json.loads(
            (OUTPUT / "canonical_worker_result.json").read_text(encoding="utf-8")
        )

    def test_canonical_run_is_a_certified_local_sse(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertEqual(self.result["termination_status"], "certified_local_sse")
        self.assertTrue(self.result["local_sse_verified"])
        self.assertEqual(self.result["initial_defender_action_id"], 0)
        self.assertEqual(self.result["final_local_sse_action_id"], 2)
        self.assertEqual(self.result["final_attacker_response_id"], 26)
        self.assertGreaterEqual(self.result["local_optimality_margin"], -1.0e-12)

    def test_search_moves_by_strict_improvement_and_terminates(self) -> None:
        self.assertEqual(self.result["visited_defender_actions"], [0, 1, 2])
        self.assertEqual(self.result["evaluated_defender_actions"], [0, 1, 2])
        iterations = self.result["iterations"]
        self.assertEqual([row["chosen_next_action_id"] for row in iterations], [1, 2, None])
        self.assertTrue(all(row["payoff_improvement"] > 0.0 for row in iterations[:-1]))
        self.assertEqual(iterations[-1]["payoff_improvement"], 0.0)

    def test_every_evaluated_action_has_exact_exhaustive_follower_audit(self) -> None:
        records = self.history["exact_evaluation_records"]
        self.assertEqual(
            {row["action_id"] for row in records},
            set(self.result["evaluated_defender_actions"]),
        )
        for record in records:
            self.assertTrue(record["exact_exhaustive_minimum_verified"])
            self.assertTrue(record["strong_follower_tie_audit_passed"])
            self.assertTrue(record["local_evaluation"][
                "exact_attacker_best_response_verified"
            ])
            self.assertGreater(record["candidate_count"], 0)

    def test_final_certificate_compares_every_required_neighbor(self) -> None:
        final = self.result["iterations"][-1]
        evaluated = set(self.result["evaluated_defender_actions"])
        self.assertLessEqual(set(final["neighbor_action_ids"]), evaluated)
        self.assertFalse(final["unknown_neighbor_ids"])
        self.assertFalse(final["local_verification"]["unknown_neighbor_diagnostics"])
        self.assertTrue(final["local_verification"]["local_sse_verified"])

    def test_independent_replay_and_frozen_regression_pass(self) -> None:
        self.assertTrue(self.result["independent_replay_status"])
        checks = self.summary["gate_checks"]
        self.assertTrue(checks["frozen_canonical_action_matches_as_regression_only"])
        self.assertTrue(checks["frozen_canonical_payoffs_within_tolerance"])
        self.assertTrue(checks["frozen_trajectory_identity_matches"])
        self.assertTrue(checks["global_solver_source_unchanged"])

    def test_result_is_not_mislabeled_as_global(self) -> None:
        self.assertFalse(self.result["global_optimality_evaluated"])
        self.assertIsNone(self.result["global_optimal"])
        self.assertIn("not a global", self.result["solution_scope"])
        self.assertFalse(self.summary["global_solver_modified"])

    def test_terrain_stays_behind_the_existing_abstract_interface(self) -> None:
        source = (ROOT / "local_stackelberg_solver.py").read_text(encoding="utf-8")
        self.assertIn("terrain: TerrainModel", source)
        self.assertNotIn("centered_cube", source)
        self.assertNotIn("build_terrain", source)
        self.assertNotIn("run_finite_stackelberg", source)

    def test_runtime_and_peak_memory_are_recorded(self) -> None:
        runtime = self.result["runtime_decomposition_s"]
        self.assertGreater(runtime["T_local_s"], 0.0)
        self.assertAlmostEqual(runtime["T_reconciliation_error_s"], 0.0, places=9)
        self.assertGreater(self.result["peak_memory"]["peak_rss_bytes"], 0)
        self.assertGreaterEqual(
            self.result["peak_memory"]["peak_rss_bytes"],
            self.result["peak_memory"]["delta_peak_rss_bytes"],
        )

    def test_required_artifacts_exist_and_are_nonempty(self) -> None:
        for name in (
            "canonical_worker_result.json",
            "canonical_local_sse_result.json",
            "local_search_history.json",
            "runtime_memory_summary.json",
            "local_search_diagnostic.png",
            "stage14_2b_summary.json",
        ):
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_integrated_notebook_orders_and_executes_14_2b(self) -> None:
        notebook = json.loads(
            (ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8")
        )
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(
            cell_ids.index("stage14-2b-heading"), cell_ids.index("stage14-2a-figure")
        )
        self.assertLess(
            cell_ids.index("stage14-2b-figure"), cell_ids.index("future-substages")
        )
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("run_stage14_2b_diagnostics", source)
        self.assertIn("RECOMPUTE_STAGE14_2 = False", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")
        stage_cells = {
            cell["id"]: cell for cell in notebook["cells"]
            if cell["id"].startswith("stage14-2b-")
        }
        self.assertTrue(stage_cells["stage14-2b-summary"].get("outputs"))
        self.assertTrue(stage_cells["stage14-2b-figure"].get("outputs"))


if __name__ == "__main__":
    unittest.main()
