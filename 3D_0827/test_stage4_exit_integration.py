"""Progressive Stage-4 exit gate across candidate and legacy pipelines."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from candidate_energy import evaluate_switching_candidates
from los_explorer_gui import create_energy_explorer, create_los_explorer
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from switching_candidates import generate_switching_candidates
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL
from visualization import (
    plot_candidate_energy_classification,
    plot_single_candidate_energy,
)


EXIT_CASE_IDS = ("case_a", "case_e")


class Stage4ExitIntegrationTests(unittest.TestCase):
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
        cls.candidate_results = {}
        for case_id, result in cls.results.items():
            los_result = result.los_result
            candidates = generate_switching_candidates(los_result.tangent_contour)
            evaluations = evaluate_switching_candidates(
                candidates,
                los_result.tangent_contour,
                los_result.terrain_map,
                los_result.mission_points,
            )
            cls.candidate_results[case_id] = (candidates, evaluations)

    def test_candidate_certificates_integrate_with_existing_los_results(self) -> None:
        for case_id, (candidates, evaluations) in self.candidate_results.items():
            with self.subTest(case=case_id):
                self.assertEqual(len(candidates), 96)
                self.assertEqual(len(evaluations), 96)
                self.assertTrue(all(
                    result.acoustically_neutralized for result in evaluations
                ))
                self.assertTrue(any(result.reachable for result in evaluations))
                self.assertTrue(any(
                    result.powered_feasible and not result.reachable
                    for result in evaluations
                ))
                self.assertTrue(any(
                    not result.powered_feasible for result in evaluations
                ))

    def test_single_and_all_candidate_figures_have_required_layers(self) -> None:
        los_result = self.results["case_a"].los_result
        _, evaluations = self.candidate_results["case_a"]
        single = plot_single_candidate_energy(
            los_result.terrain_map,
            los_result.mission_points,
            los_result.los_surface,
            los_result.visualization_rays,
            evaluations[54],
        )
        single_names = [str(trace.name) for trace in single.data]
        self.assertIn("Straight powered segment", single_names)
        self.assertIn("Selected switch ID 54", single_names)

        all_candidates = plot_candidate_energy_classification(
            los_result.terrain_map,
            los_result.mission_points,
            los_result.los_surface,
            los_result.visualization_rays,
            evaluations,
        )
        category_names = [str(trace.name) for trace in all_candidates.data]
        self.assertTrue(any(name.startswith("Powered infeasible (24)") for name in category_names))
        self.assertTrue(any(name.startswith("Powered feasible / glide unreachable (12)") for name in category_names))
        self.assertTrue(any(name.startswith("Powered feasible / glide reachable (60)") for name in category_names))

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

    def test_candidate_energy_layer_has_no_game_solver_dependency(self) -> None:
        directory = Path(__file__).resolve().parent
        source = (directory / "candidate_energy.py").read_text(encoding="utf-8")
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
        for forbidden in (
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "bellman",
            "hazard",
            "defender",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(forbidden in name.lower() for name in imports))


if __name__ == "__main__":
    unittest.main()
