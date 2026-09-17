"""Stage 14.2A artifact, architecture, and notebook exit gates."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2a_local_sse_contract"


class Stage142AExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (OUTPUT / "stage14_2a_summary.json").read_text(encoding="utf-8")
        )
        cls.contract = json.loads(
            (OUTPUT / "local_sse_mathematical_contract.json").read_text(encoding="utf-8")
        )
        cls.neighborhood = json.loads(
            (OUTPUT / "neighborhood_metadata.json").read_text(encoding="utf-8")
        )
        cls.tiny = json.loads(
            (OUTPUT / "tiny_game_verification.json").read_text(encoding="utf-8")
        )
        cls.existence = json.loads(
            (OUTPUT / "current_discretized_existence_report.json").read_text(
                encoding="utf-8"
            )
        )

    def test_contract_explicitly_distinguishes_local_and_global(self) -> None:
        self.assertIn("not a global", self.contract["local_scope"])
        self.assertIn("every feasible d in D", self.contract["global_scope"])
        self.assertIn("exact finite Attacker best response", self.contract[
            "attacker_response_requirement"
        ])
        self.assertIn("maximize Defender payoff", self.contract[
            "follower_tie_requirement"
        ])

    def test_default_radius_and_chebyshev_examples_are_exported(self) -> None:
        default = self.neighborhood["default_configuration"]
        self.assertEqual(default["r_neighbor"], 1)
        self.assertEqual(default["metric"], "chebyshev_grid_index")
        self.assertEqual(
            self.neighborhood["one_dimensional_examples"]["r1_neighbors"], [1, 3]
        )
        self.assertEqual(
            self.neighborhood["one_dimensional_examples"]["r2_neighbors"],
            [0, 1, 3, 4],
        )
        self.assertEqual(
            self.neighborhood["two_dimensional_examples"]["r1_neighbor_count"], 8
        )
        self.assertEqual(
            self.neighborhood["two_dimensional_examples"]["r2_neighbor_count"], 24
        )

    def test_tiny_game_separates_local_global_and_full_radius(self) -> None:
        self.assertEqual(self.tiny["local_sse_action_ids"], [1, 3])
        self.assertEqual(self.tiny["global_sse_action_ids"], [3])
        self.assertEqual(self.tiny["local_but_not_global_action_ids"], [1])
        self.assertTrue(self.tiny["full_radius_equals_global"])
        self.assertTrue(self.tiny["strong_follower_tie_audit"]["passed"])

    def test_current_discretized_existence_and_global_implies_local(self) -> None:
        self.assertEqual(self.existence["terrain_category"], "centered_cube")
        self.assertTrue(self.existence["existence_condition_satisfied"])
        self.assertTrue(
            self.existence["global_action_local_verification"]["local_sse_verified"]
        )
        self.assertTrue(all(self.existence["existence_assumptions"].values()))

    def test_existing_global_solver_source_is_frozen(self) -> None:
        self.assertTrue(self.summary["solver_source_unchanged"])
        self.assertFalse(self.summary["global_solver_modified"])
        self.assertFalse(self.summary["local_search_implemented"])
        local_source = (ROOT / "local_sse_contract.py").read_text(encoding="utf-8")
        self.assertNotIn("run_finite_stackelberg", local_source)
        self.assertNotIn("run_attacker_best_response", local_source)

    def test_all_stage_gate_checks_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(all(self.summary["gate_checks"].values()))

    def test_required_artifacts_and_diagnostic_exist(self) -> None:
        required = (
            "local_sse_mathematical_contract.json",
            "neighborhood_metadata.json",
            "tiny_game_verification.json",
            "current_discretized_existence_report.json",
            "local_vs_global_payoff_diagnostic.png",
            "stage14_2a_summary.json",
        )
        for name in required:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_stage14_notebook_orders_14_2a_after_14_2(self) -> None:
        notebook = json.loads(
            (ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8")
        )
        cell_ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(
            cell_ids.index("stage14-2a-heading"), cell_ids.index("stage14-2-figure")
        )
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("run_stage14_2a_diagnostics", source)
        self.assertIn("RECOMPUTE_STAGE14_2 = False", source)
        self.assertGreater(
            cell_ids.index("stage14-2b-heading"), cell_ids.index("stage14-2a-figure")
        )


if __name__ == "__main__":
    unittest.main()
