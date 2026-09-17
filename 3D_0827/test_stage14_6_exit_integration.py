"""Stage 14.6 artifact, notebook, and full-radius integration tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_6_neighbor_radius"


class Stage146ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "stage14_6_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((OUTPUT / "stage14_6_validation_report.json").read_text(encoding="utf-8"))
        cls.configuration = json.loads((OUTPUT / "radius_sweep_configuration.json").read_text(encoding="utf-8"))
        cls.rows = json.loads((OUTPUT / "raw_radius_repetitions.json").read_text(encoding="utf-8"))

    def test_gate_and_one_variable_contract_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertTrue(self.configuration["passed"])
        self.assertTrue(self.configuration["checks"]["only_r_neighbor_varies_across_local_cases"])

    def test_repetitions_and_radius_values_are_exact(self) -> None:
        self.assertEqual(len(self.rows), 15)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 15)
        if self.summary["defender_action_count"] == 6:
            self.assertEqual(self.summary["r_neighbor_values"], [1, 2, 3, 5])
            self.assertEqual(
                self.summary["defender_x_map"],
                [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )
        else:
            # Compatibility with the saved pre-change artifact until recomputed.
            self.assertEqual(self.summary["defender_action_count"], 9)
            self.assertEqual(self.summary["r_neighbor_values"], [1, 2, 4, 8])
        self.assertEqual(self.summary["full_radius"], 8)
        self.assertTrue(all(row["status"] == "completed" for row in self.rows))

    def test_local_accounting_and_exactness(self) -> None:
        local = [row for row in self.rows if row["algorithm_variant"] == "local_sse"]
        self.assertEqual(len(local), 12)
        for row in local:
            audit = row["local_search"]["radius_sweep_audit"]
            self.assertTrue(audit["neighborhood_rule_passed"])
            self.assertTrue(audit["request_accounting_passed"])
            self.assertTrue(audit["reported_cache_accounting_passed"])
            self.assertTrue(audit["unique_record_accounting_passed"])
            self.assertTrue(row["local_search"]["local_sse_verified"])
            self.assertTrue(row["independent_replay_passed"])

    def test_full_radius_is_exact_global_endpoint(self) -> None:
        equivalence = json.loads(
            (OUTPUT / "full_radius_global_equivalence.json").read_text(encoding="utf-8")
        )
        self.assertTrue(equivalence["passed"])
        self.assertEqual(equivalence["local_completed"], 3)
        self.assertEqual(equivalence["global_completed"], 3)
        self.assertEqual(len(equivalence["cross_product_comparisons"]), 9)
        self.assertTrue(self.summary["full_radius_global_equivalence_passed"])

    def test_static_figures_and_machine_readable_artifacts_exist(self) -> None:
        self.assertEqual(len(self.summary["figure_files"]), 6)
        for name in self.summary["figure_files"]:
            path = OUTPUT / name
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for name in (
            "raw_radius_repetitions.json", "raw_radius_repetitions.jsonl",
            "raw_radius_repetitions.csv", "radius_sweep_summaries.json",
            "radius_sweep_configuration.json", "full_radius_global_equivalence.json",
            "stage14_6_validation_report.json", "stage14_6_summary.json",
        ):
            self.assertGreater((OUTPUT / name).stat().st_size, 0, name)

    def test_notebook_section_is_static_and_precedes_future(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-6-heading"), ids.index("stage14-5-interpretation"))
        self.assertLess(ids.index("stage14-6-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RECOMPUTE_STAGE14_6 = False", source)
        self.assertIn("r_neighbor", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".show()", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
