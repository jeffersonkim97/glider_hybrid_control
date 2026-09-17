"""Stage 14.7 artifact, figure, notebook, and claim-boundary tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_7_scaling_analysis"


class Stage147ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((OUTPUT / "stage14_7_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads((OUTPUT / "stage14_7_validation_report.json").read_text(encoding="utf-8"))
        cls.fits = json.loads((OUTPUT / "fit_parameters.json").read_text(encoding="utf-8"))
        cls.limits = json.loads((OUTPUT / "computational_limit_cases.json").read_text(encoding="utf-8"))

    def test_exit_gate_and_claim_boundary(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        self.assertTrue(self.validation["gate_passed"])
        self.assertTrue(all(self.validation["checks"].values()))
        self.assertEqual(self.summary["solver_or_benchmark_workers_started"], 0)
        self.assertFalse(self.summary["next_stage_started"])
        self.assertIn("never formal Big-O", self.validation["claim_boundary"])

    def test_fit_metadata_is_complete_and_bounded(self) -> None:
        self.assertEqual(len(self.fits), 33)
        self.assertEqual(self.summary["completed_fit_count"], 33)
        for fit in self.fits:
            self.assertEqual(fit["status"], "completed")
            self.assertGreaterEqual(fit["sample_count"], 3)
            self.assertTrue(fit["empirical_fit_only"])
            self.assertFalse(fit["formal_big_o_claim"])
            self.assertFalse(fit["extrapolation_performed"])
            self.assertNotIn("N_S_active+N_E", fit["independent_variable"])

    def test_figure_x_axes_follow_discretization_controls(self) -> None:
        self.assertEqual(self.summary["x_axis_policy"], {
            "spatial": "spatial_resolution_m",
            "heading": "heading_resolution_deg",
            "switching": "switching_contour_spacing=1/switching_contour_sample_count",
            "neighbor_search": "r_neighbor in Defender grid lengths",
            "state_and_edge_counts": "secondary y-axis diagnostics only; never an x-axis",
        })
        self.assertTrue(
            self.validation["checks"]["all_figure_x_axes_use_declared_discretization_controls"]
        )

    def test_noncompleted_cases_are_retained_and_classified(self) -> None:
        self.assertEqual(self.limits["noncompleted_repetition_count"], 9)
        self.assertEqual(self.limits["model_infeasible_repetition_count"], 6)
        self.assertEqual(
            self.limits["computational_or_instrumentation_failure_repetition_count"], 3
        )
        self.assertEqual(self.limits["timeout_repetition_count"], 0)
        self.assertEqual(self.limits["memory_limit_repetition_count"], 0)

    def test_nine_static_figures_and_deliverables_exist(self) -> None:
        self.assertEqual(len(self.summary["figure_files"]), 9)
        for name in self.summary["figure_files"]:
            self.assertEqual((OUTPUT / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for name in (
            "fit_parameters.csv", "fit_parameters.json", "normalized_scaling_measurements.json",
            "raw_summary_crosscheck.json", "computational_limit_cases.json", "plot_data.json",
            "theoretical_vs_empirical_interpretation.md", "stage14_7_validation_report.json",
            "stage14_7_summary.json",
        ):
            self.assertGreater((OUTPUT / name).stat().st_size, 0, name)

    def test_notebook_stage14_7_is_static_and_ordered(self) -> None:
        notebook = json.loads((ROOT / "3D_Stage14_Computation_Load.ipynb").read_text(encoding="utf-8"))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertGreater(ids.index("stage14-7-heading"), ids.index("stage14-6-interpretation"))
        self.assertLess(ids.index("stage14-7-interpretation"), ids.index("future-substages"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("RECOMPUTE_STAGE14_7 = False", source)
        self.assertIn("solver_or_benchmark_workers_started", source)
        self.assertIn("spatial_resolution_m", source)
        self.assertIn("heading_resolution_deg", source)
        self.assertIn("switching_contour_spacing", source)
        self.assertNotIn("IFrame", source)
        self.assertNotIn(".show()", source)
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
