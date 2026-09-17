"""Stage-11 integrated-notebook execution and regression gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import nbformat


ROOT = Path(__file__).resolve().parent
FIGURE_DIRECTORY = ROOT / "figure" / "stage_11_integrated_notebook"
STAGE12_FIGURE_DIRECTORY = ROOT / "figure" / "stage_12_integrated_notebook"
NOTEBOOK_PATH = ROOT / "3D_Attacker_Bellman_Validated.ipynb"
ORIGINAL_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


class Stage11ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            (FIGURE_DIRECTORY / "stage11_summary.json").read_text(encoding="utf-8")
        )
        cls.notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
        cls.stage12_summary = json.loads(
            (STAGE12_FIGURE_DIRECTORY / "stage12_integrated_summary.json").read_text(
                encoding="utf-8",
            )
        )

    def test_new_notebook_executed_top_to_bottom_without_errors(self) -> None:
        self.assertEqual(len(self.notebook.cells), 14)
        self.assertTrue(all(cell.cell_type == "code" for cell in self.notebook.cells))
        self.assertEqual(
            [cell.execution_count for cell in self.notebook.cells], list(range(1, 15)),
        )
        errors = [
            output
            for cell in self.notebook.cells
            for output in cell.get("outputs", ())
            if output.output_type == "error"
        ]
        self.assertEqual(errors, [])

    def test_configuration_is_centralized_and_records_every_discretization(self) -> None:
        source = "\n".join(cell.source for cell in self.notebook.cells)
        self.assertIn("CONFIG = Stage11Config()", self.notebook.cells[0].source)
        self.assertNotIn("create_los_explorer", source)
        resolution = self.summary["configuration"]["discretization"]
        for name in (
            "horizontal_spacing_map", "altitude_spacing_map",
            "heading_bin_count", "motion_primitive_radius",
            "los_probe_grid_size", "los_boundary_refinement_steps",
            "switching_contour_sample_count", "switching_radial_sample_count",
            "hazard_quadrature_resolution",
        ):
            self.assertIn(name, resolution)

    def test_canonical_exact_response_matches_frozen_stage10_result(self) -> None:
        result = self.summary["exact_attacker_best_response"]
        self.assertEqual(result["selected_candidate_id"], 25)
        self.assertEqual(result["cooptimal_candidate_ids"], [25, 34])
        self.assertEqual(result["total_candidates"], 96)
        self.assertEqual(result["feasible_candidates"], 10)
        self.assertAlmostEqual(result["objective"], 0.21356152236500636, places=13)
        self.assertAlmostEqual(result["mission_time_s"], 90.68077307322775, places=11)
        self.assertAlmostEqual(result["cumulative_hazard"], 0.0172459504390233, places=14)

    def test_independent_replay_and_exact_minimum_gate_pass(self) -> None:
        self.assertTrue(self.summary["gate_passed"])
        report = self.summary["validation_report"]
        for field in (
            "passed", "terrain_clear", "los_phase_valid",
            "turn_constraints_valid", "energy_valid", "time_consistent",
            "hazard_consistent", "goal_valid",
        ):
            self.assertTrue(report[field], field)
        self.assertLessEqual(report["time_error_s"], 1.0e-9)
        self.assertLessEqual(report["hazard_error"], 1.0e-10)

    def test_timing_report_has_required_nonnegative_components(self) -> None:
        timing = self.summary["timing_s"]
        for field in (
            "terrain_LOS_s", "candidate_generation_s", "energy_filtering_s",
            "graph_build_s", "hazard_precompute_s", "Bellman_solve_s",
            "all_candidate_BR_s", "validation_s", "notebook_total_s",
        ):
            self.assertIn(field, timing)
            self.assertGreaterEqual(timing[field], 0.0)

    def test_all_required_interactive_figures_exist(self) -> None:
        expected = (
            "cell_02_terrain.html", "cell_03_los_surface.html",
            "cell_04_switching_candidates.html", "cell_05_energy_filtering.html",
            "cell_06_bellman_reachability.html", "cell_07_single_candidate.html",
            "cell_08_exact_attacker_best_response.html",
            "cell_08_candidate_objectives.html",
            "cell_09_independent_validation.html",
        )
        for filename in expected:
            path = FIGURE_DIRECTORY / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 1000, filename)

    def test_original_gui_notebook_is_untouched(self) -> None:
        digest = hashlib.sha256((ROOT / "3D_bellman_0827.ipynb").read_bytes()).hexdigest()
        self.assertEqual(digest, ORIGINAL_NOTEBOOK_SHA256)

    def test_fine_stage12_and_standalone_gui_are_integrated(self) -> None:
        result = self.stage12_summary
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["horizontal_step_m"], 25.0)
        self.assertEqual(result["altitude_step_m"], 25.0)
        self.assertEqual(result["requested_heading_step_deg"], 5.0)
        self.assertEqual(result["heading_bins"], 72)
        self.assertLessEqual(result["maximum_realized_heading_error_deg"], 2.5)
        self.assertLess(
            result["active_corridor_state_count"],
            result["goal_reachable_state_count"],
        )
        self.assertLess(
            result["goal_reachable_state_count"],
            result["cartesian_state_count"],
        )
        for filename in (
            "cell_11_sparse_stackelberg_reachable_set.html",
            "cell_12_defender_payoff_comparison.html",
        ):
            path = STAGE12_FIGURE_DIRECTORY / filename
            self.assertTrue(path.is_file(), filename)
            self.assertGreater(path.stat().st_size, 1_000, filename)
        gui_source = self.notebook.cells[12].source
        self.assertIn("launch_stackelberg_gui", gui_source)
        self.assertIn("open_browser", gui_source)

    def test_post_gui_trajectory_time_histories_are_integrated(self) -> None:
        last_source = self.notebook.cells[-1].source
        self.assertIn("build_trajectory_time_history", last_source)
        self.assertIn("plot_trajectory_time_history", last_source)
        figure = STAGE12_FIGURE_DIRECTORY / "cell_14_trajectory_time_histories.html"
        summary_path = (
            STAGE12_FIGURE_DIRECTORY / "cell_14_trajectory_time_history_summary.json"
        )
        self.assertTrue(figure.is_file())
        self.assertGreater(figure.stat().st_size, 1_000)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertLessEqual(summary["mission_time_error_s"], 1.0e-10)
        self.assertLessEqual(summary["cumulative_hazard_error"], 1.0e-10)
        self.assertLessEqual(summary["detection_probability_error"], 1.0e-10)
        self.assertEqual(summary["cumulative_acoustic_hazard"], 0.0)
        self.assertAlmostEqual(
            summary["cumulative_total_hazard"],
            summary["cumulative_rcs_radar_hazard"]
            + summary["cumulative_radial_velocity_doppler_hazard"],
            places=13,
        )


if __name__ == "__main__":
    unittest.main()
