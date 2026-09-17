"""Stage 14.4 data, visualization, notebook, and regression gate tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_4_heading_resolution"


class Stage144ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "stage14_4_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((OUTPUT / "stage14_4_validation_report.json").read_text(encoding="utf-8"))
        cls.configuration = json.loads((OUTPUT / "heading_sweep_configuration.json").read_text(encoding="utf-8"))
        cls.rows = json.loads((OUTPUT / "raw_heading_repetitions.json").read_text(encoding="utf-8"))
        cls.summaries = json.loads((OUTPUT / "heading_sweep_summaries.json").read_text(encoding="utf-8"))
        cls.source_manifest = json.loads(
            (OUTPUT / "source_reuse_manifest.json").read_text(encoding="utf-8")
        )

    def test_gate_and_one_variable_contract_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertTrue(self.configuration["passed"])
        self.assertTrue(self.configuration["checks"]["only_heading_resolution_varies"])

    def test_reused_and_new_repetitions_are_complete(self) -> None:
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 120)
        if self.source_manifest.get("mode") == "full_fresh":
            self.assertEqual(self.summary["reused_repetition_count"], 0)
            self.assertEqual(self.summary["new_repetition_count"], 120)
        else:
            self.assertEqual(self.summary["reused_repetition_count"], 18)
            self.assertEqual(self.summary["new_repetition_count"], 102)

    def test_requested_and_realized_heading_grids_are_explicit(self) -> None:
        for row in self.rows:
            heading = row["realized_heading_grid"]
            requested = heading["requested_heading_spacing_deg"]
            expected_bins = int(round(360.0 / requested))
            realized = 360.0 / expected_bins
            self.assertEqual(
                (heading["heading_bin_count"], heading["realized_heading_spacing_deg"],
                 heading["maximum_heading_quantization_error_deg"]),
                (expected_bins, realized, realized / 2.0),
            )

    def test_all_completed_results_are_exact_replayed_and_deterministic(self) -> None:
        completed = [row for row in self.rows if row["status"] == "completed"]
        self.assertTrue(completed)
        for row in completed:
            self.assertTrue(row["independent_replay_passed"])
            self.assertTrue(all(row["exactness"].values()))
        self.assertTrue(all(
            row["deterministic_solution_identity"]
            for row in self.summaries if row["successful_repetitions"]
        ))
        self.assertTrue(all(
            row["failure_category"]
            for row in self.rows if row["status"] != "completed"
        ))

    def test_six_static_figures_use_shared_resolution_contract(self) -> None:
        self.assertEqual(self.summary["figure_files"], [
            "01_total_sse_runtime.png",
            "02_runtime_decomposition.png",
            "03_peak_memory.png",
            "04_objectives.png",
            "05_equilibrium_selection.png",
            "06_trajectory_identity.png",
        ])
        methodology = self.summary["figure_methodology"]
        self.assertIn("Delta psi", methodology["controlled_x_axis"])
        self.assertIn("metadata only", methodology["realized_heading_bins_location"])
        self.assertTrue(methodology["shared_resolution_figure_schema"])
        self.assertEqual(methodology["primary_algorithm"], "local_sse")
        for name in self.summary["figure_files"]:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_machine_readable_artifacts_exist(self) -> None:
        for name in (
            "raw_heading_repetitions.json", "raw_heading_repetitions.jsonl",
            "raw_heading_repetitions.csv", "heading_sweep_summaries.json",
            "heading_sweep_summaries.csv", "source_reuse_manifest.json",
            "heading_sweep_configuration.json", "stage14_4_validation_report.json",
            "stage14_4_summary.json",
        ):
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)

    def test_notebook_is_static_and_stops_before_stage14_5(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-4-heading"), ids.index("stage14-3-interpretation"))
        self.assertLess(ids.index("stage14-4-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RECOMPUTE_STAGE14_4 = False", source)
        self.assertIn("requested heading resolution", source)
        self.assertIn("Individual isolated-process runs (n=3)", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".show()", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
