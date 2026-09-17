"""Stage 14.5 artifact, notebook, exactness, and regression tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_5_switching_candidates"


class Stage145ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "stage14_5_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((OUTPUT / "stage14_5_validation_report.json").read_text(encoding="utf-8"))
        cls.configuration = json.loads((OUTPUT / "candidate_sweep_configuration.json").read_text(encoding="utf-8"))
        cls.rows = json.loads((OUTPUT / "raw_candidate_repetitions.json").read_text(encoding="utf-8"))
        cls.summaries = json.loads((OUTPUT / "candidate_sweep_summaries.json").read_text(encoding="utf-8"))

    def test_gate_and_one_variable_contract_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertTrue(self.configuration["passed"])
        self.assertTrue(self.configuration["checks"]["only_switching_candidate_density_varies"])

    def test_repetition_and_candidate_counts_are_explicit(self) -> None:
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 18)
        self.assertEqual(self.summary["raw_candidate_counts"], [48, 72, 96])
        expected_keys = self.validation["candidate_funnel_keys"]
        for row in self.rows:
            counts = row["state_and_game_size"]
            self.assertTrue(all(key in counts for key in expected_keys))
            self.assertEqual(
                counts["N_C_raw"],
                row["configuration"]["parameters"]["switching_contour_sample_count"] * 8,
            )

    def test_completed_results_are_exact_replayed_and_deterministic(self) -> None:
        self.assertTrue(all(row["status"] == "completed" for row in self.rows))
        for row in self.rows:
            self.assertTrue(row["independent_replay_passed"])
            self.assertTrue(all(row["exactness"].values()))
        self.assertTrue(all(item["deterministic_solution_identity"] for item in self.summaries))
        self.assertTrue(all(item["candidate_funnel_stable_across_repetitions"] for item in self.summaries))

    def test_five_static_figures_exist(self) -> None:
        self.assertEqual(self.summary["figure_files"], [
            "01_total_sse_runtime.png",
            "02_runtime_decomposition.png",
            "03_peak_memory.png",
            "04_objectives.png",
            "05_equilibrium_selection.png",
            "06_trajectory_identity.png",
        ])
        for name in self.summary["figure_files"]:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_machine_readable_artifacts_exist(self) -> None:
        for name in (
            "raw_candidate_repetitions.json", "raw_candidate_repetitions.jsonl",
            "raw_candidate_repetitions.csv", "candidate_sweep_summaries.json",
            "candidate_sweep_summaries.csv", "runner_resume_report.json",
            "candidate_sweep_configuration.json", "stage14_5_validation_report.json",
            "stage14_5_summary.json",
        ):
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_notebook_stage14_5_section_is_static_and_precedes_future(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-5-heading"), ids.index("stage14-4-interpretation"))
        self.assertLess(ids.index("stage14-5-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RECOMPUTE_STAGE14_5 = False", source)
        self.assertIn("same six-figure schema", source)
        self.assertIn("not a primary performance figure", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".show()", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
