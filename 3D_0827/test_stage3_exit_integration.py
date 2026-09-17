"""Progressive Stage-3 exit gate for the existing LOS/energy/GUI pipeline."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from los_explorer_gui import create_energy_explorer, create_los_explorer
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from switching_candidates import generate_switching_candidates
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL
from visualization import plot_switching_candidates


EXIT_CASE_IDS = ("case_a", "case_e")


class Stage3ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        selected = {
            case.case_id: case
            for case in CANONICAL_CASES
            if case.case_id in EXIT_CASE_IDS
        }
        cls.results = {
            case_id: compute_baseline_case(selected[case_id])
            for case_id in EXIT_CASE_IDS
        }
        cls.summaries = {
            case_id: summarize_baseline_case(selected[case_id], cls.results[case_id])
            for case_id in EXIT_CASE_IDS
        }

    def test_candidates_attach_to_existing_los_results_without_mutation(self) -> None:
        for case_id, result in self.results.items():
            with self.subTest(case=case_id):
                contour = result.los_result.tangent_contour
                candidates = generate_switching_candidates(contour)
                self.assertEqual(len(candidates), 96)
                self.assertTrue(all(
                    candidate.surface_residual <= 1.0e-8
                    for candidate in candidates
                ))
                self.assertEqual(
                    result.los_result.los_surface.tangent_contour,
                    contour,
                )

    def test_geometry_los_and_energy_remain_at_frozen_baseline(self) -> None:
        exact_fields = (
            "terrain",
            "obstacle_box_count",
            "obstacle_triangle_count",
            "tangent_ray_count",
            "discarded_ground_candidate_count",
            "contour_closed",
            "los_surface_panel_count",
            "reachable_face_count",
            "unreachable_face_count",
            "reachable_vertex_count",
            "unreachable_vertex_count",
            "powered_infeasible_vertex_count",
        )
        geometry_fields = (
            "sensor",
            "bounds",
            "maximum_height",
            "minimum_tangent_altitude",
            "representative_reachable_point",
            "representative_unreachable_point",
        )
        energy_fields = (
            "representative_reachable_margin_j",
            "representative_unreachable_margin_j",
        )
        for case_id in EXIT_CASE_IDS:
            actual = self.summaries[case_id]
            expected = EXPECTED[case_id]
            with self.subTest(case=case_id):
                for field in exact_fields:
                    self.assertEqual(actual[field], expected[field], field)
                for field in geometry_fields:
                    np.testing.assert_allclose(
                        actual[field], expected[field], rtol=0.0, atol=GEOMETRY_ATOL,
                        err_msg=f"{case_id}: {field}",
                    )
                for field in energy_fields:
                    np.testing.assert_allclose(
                        actual[field], expected[field], rtol=ENERGY_RTOL, atol=ENERGY_ATOL_J,
                        err_msg=f"{case_id}: {field}",
                    )

    def test_candidate_figure_contains_no_energy_classification(self) -> None:
        los_result = self.results["case_a"].los_result
        candidates = generate_switching_candidates(los_result.tangent_contour)
        figure = plot_switching_candidates(
            los_result.terrain_map,
            los_result.mission_points,
            los_result.los_surface,
            los_result.visualization_rays,
            candidates,
        )
        trace_names = [str(trace.name).lower() for trace in figure.data]
        self.assertTrue(any("switching candidates (96)" in name for name in trace_names))
        self.assertFalse(any("goal-reachable" in name for name in trace_names))
        self.assertFalse(any("goal-unreachable" in name for name in trace_names))

    def test_existing_gui_controllers_still_run(self) -> None:
        with redirect_stdout(io.StringIO()):
            los_gui = create_los_explorer(probe_grid_size=25)
            energy_gui = create_energy_explorer(
                probe_grid_size=25,
                radial_section_count=7,
                contour_section_count=24,
            )
            los_gui.terrain_toggle.value = "stepped_pyramid"
            energy_gui.terrain_toggle.value = "stepped_pyramid"
        for name, controller in (("los", los_gui), ("energy", energy_gui)):
            with self.subTest(controller=name):
                self.assertIsNone(controller.latest_error)
                self.assertIsNotNone(controller.latest_result)
                self.assertIn("Ready", controller.status.value)

    def test_candidate_module_dependency_is_geometry_only(self) -> None:
        directory = Path(__file__).resolve().parent
        source = (directory / "switching_candidates.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertIn("los_geometry", imports)
        for forbidden in (
            "energy_model",
            "reachability_surface",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "bellman",
            "hazard",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(forbidden in name.lower() for name in imports))


if __name__ == "__main__":
    unittest.main()
