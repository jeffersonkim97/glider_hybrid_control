"""Stage 14.3 raw-data, visualization, notebook, and regression gate tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_3_spatial_resolution"


class Stage143ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "stage14_3_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((OUTPUT / "stage14_3_validation_report.json").read_text(encoding="utf-8"))
        cls.configuration = json.loads((OUTPUT / "spatial_sweep_configuration.json").read_text(encoding="utf-8"))
        cls.rows = json.loads((OUTPUT / "raw_spatial_repetitions.json").read_text(encoding="utf-8"))
        cls.summaries = json.loads((OUTPUT / "spatial_sweep_summaries.json").read_text(encoding="utf-8"))
        cls.source_manifest = json.loads(
            (OUTPUT / "source_reuse_manifest.json").read_text(encoding="utf-8")
        )

    def test_gate_and_one_variable_contract_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertTrue(self.configuration["passed"])
        self.assertTrue(self.configuration["checks"]["only_spatial_resolution_varies"])
        self.assertFalse(self.summary["scientific_interpretation"]["power_law_fit_performed"])
        self.assertFalse(self.summary["scientific_interpretation"]["objective_convergence_established"])

    def test_reused_and_new_repetitions_are_unique_and_complete_as_records(self) -> None:
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(len({row["repetition_id"] for row in self.rows}), 120)
        if self.source_manifest.get("mode") == "full_fresh":
            self.assertEqual(self.summary["reused_repetition_count"], 0)
            self.assertEqual(self.summary["new_repetition_count"], 120)
        else:
            self.assertEqual(self.summary["reused_repetition_count"], 18)
            self.assertEqual(self.summary["new_repetition_count"], 102)
        self.assertEqual({row["stage14_3_source"] for row in self.rows}, {
            "reused_stage14_2c_cold_process_rows", "new_stage14_3_cold_process_rows",
        })

    def test_coarse_baseline_and_completed_finer_oracle_exist(self) -> None:
        completed = self.validation["completed_global_oracle_resolutions_m"]
        self.assertTrue({20.0, 25.0, 50.0, 100.0}.issubset(completed))
        self.assertIn(20.0, self.validation["finer_resolution_outcome"]["completed_resolutions_m"])

    def test_every_completed_result_is_exact_and_replayed(self) -> None:
        completed = [row for row in self.rows if row["status"] == "completed"]
        self.assertTrue(completed)
        for row in completed:
            self.assertTrue(row["independent_replay_passed"])
            self.assertTrue(all(row["exactness"].values()))

    def test_finer_failures_are_explicit_not_silently_skipped(self) -> None:
        finer = [row for row in self.rows if row["realized_grid"]["requested_spatial_resolution_m"] < 25.0]
        if self.source_manifest.get("mode") == "full_fresh":
            self.assertFalse(any(
                row["status"] == "initial_defender_infeasible" for row in finer
            ))
        self.assertTrue(any(
            row["status"] in {
                "completed", "initial_neighborhood_infeasible",
                "initial_neighborhood_comparison_unknown",
            }
            for row in finer
        ))
        self.assertTrue(any(row["status"] == "worker_failure" for row in finer))
        self.assertTrue(all(row["failure_category"] for row in finer if row["status"] != "completed"))

    def test_realized_axes_and_counts_are_exported(self) -> None:
        for row in self.rows:
            grid = row["realized_grid"]
            for axis in ("x", "y", "altitude"):
                self.assertEqual(grid[axis]["count"], len(grid[axis]["coordinates_map"]))
        for summary in self.summaries:
            if summary["state_and_game_size"]:
                counts = summary["state_and_game_size"]
                grid = summary["realized_grid"]
                self.assertEqual(counts["N_x"], grid["x"]["count"])
                self.assertEqual(counts["N_y"], grid["y"]["count"])
                self.assertEqual(counts["N_h"], grid["altitude"]["count"])

    def test_required_png_and_machine_readable_artifacts_exist(self) -> None:
        self.assertEqual(self.summary["figure_files"], [
            "01_total_sse_runtime.png",
            "02_runtime_decomposition.png",
            "03_peak_memory.png",
            "04_objectives.png",
            "05_equilibrium_selection.png",
            "06_trajectory_identity.png",
        ])
        methodology = self.summary["figure_methodology"]
        self.assertFalse(methodology["combined_state_edge_metric_used"])
        self.assertIn("isolated-process", methodology["individual_measurement_label"])
        self.assertTrue(methodology["shared_resolution_figure_schema"])
        self.assertEqual(methodology["primary_algorithm"], "local_sse")
        required = tuple(self.summary["figure_files"]) + (
            "raw_spatial_repetitions.json", "raw_spatial_repetitions.jsonl",
            "raw_spatial_repetitions.csv", "spatial_sweep_summaries.json",
            "spatial_sweep_summaries.csv", "source_reuse_manifest.json",
            "spatial_sweep_configuration.json", "stage14_3_validation_report.json",
            "stage14_3_summary.json",
        )
        for name in required:
            path = OUTPUT / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 0, name)
            if name.endswith(".png"):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_notebook_stage14_3_section_is_static_and_precedes_stage14_4(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-3-heading"), ids.index("stage14-2c-interpretation"))
        self.assertLess(ids.index("stage14-3-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RECOMPUTE_STAGE14_3 = False", source)
        self.assertIn("Stage 14.3 shared resolution-sweep figure guide", source)
        self.assertIn("Individual isolated-process runs (n=3)", source)
        self.assertIn("State/edge growth is retained only", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".show()", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
