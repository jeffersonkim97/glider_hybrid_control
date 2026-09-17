"""Progressive Stage-5 exit gate across graph and prior-stage pipelines."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from bellman_geometry import GlideTransitionModel
from bellman_graph import (
    build_bellman_graph,
    independent_reverse_reachability,
    solve_unit_cost_reachability,
)
from bellman_state import BellmanState, BellmanStateGrid
from candidate_energy import evaluate_switching_candidates
from los_explorer_gui import create_energy_explorer, create_los_explorer
from map_geometry import MapBounds
from scenario import Point3D
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from switching_candidates import generate_switching_candidates
from terrain_catalog import build_terrain
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL
from visualization import plot_bellman_successor_debug


EXPECTED_GRAPH_STATISTICS = {
    "cartesian_state_count": 62424,
    "state_count": 59616,
    "terrain_excluded_state_count": 2808,
    "valid_edge_count": 190852,
    "rejected_by_bounds": 61944,
    "rejected_by_terrain": 7508,
    "rejected_by_turn": 209841,
    "rejected_by_altitude": 6591,
    "terminal_state_count": 24,
}


class Stage5ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bounds = MapBounds(-8.0, 8.0, -4.0, 4.0)
        cls.grid = BellmanStateGrid(bounds=cls.bounds)
        cls.terrain = build_terrain("centered_cube", bounds=cls.bounds)
        cls.graph = build_bellman_graph(
            cls.grid,
            cls.terrain,
            Point3D(8.0, 0.0, 0.0),
        )
        cls.solution = solve_unit_cost_reachability(cls.graph)

        selected = {
            case.case_id: case
            for case in CANONICAL_CASES
            if case.case_id in ("case_a", "case_e")
        }
        cls.previous_results = {
            case_id: compute_baseline_case(selected[case_id])
            for case_id in ("case_a", "case_e")
        }
        cls.previous_summaries = {
            case_id: summarize_baseline_case(selected[case_id], result)
            for case_id, result in cls.previous_results.items()
        }

    def test_canonical_graph_statistics_are_deterministic(self) -> None:
        actual = vars(self.graph.statistics)
        self.assertEqual(actual, EXPECTED_GRAPH_STATISTICS)
        self.assertEqual(self.solution.goal_reachable_state_count, 43136)
        self.assertEqual(self.solution.maximum_finite_value, 25.0)

    def test_canonical_graph_dag_and_reference_reachability_gate(self) -> None:
        ordering = self.graph.topological_sort()
        self.assertEqual(len(ordering), self.graph.statistics.state_count)
        independent = independent_reverse_reachability(self.graph)
        np.testing.assert_array_equal(self.solution.goal_reachable, independent)
        for state_id in self.graph.node_ids:
            for edge in self.graph.adjacency[int(state_id)]:
                self.assertLess(
                    edge.target_state.altitude_index,
                    edge.source_state.altitude_index,
                )

    def test_successor_diagnostic_contains_valid_and_terrain_rejected_edges(self) -> None:
        selected_state = BellmanState(6, 4, 20, 3)
        model = GlideTransitionModel(self.grid, self.terrain)
        edges, statistics, rejected = model.successors(
            selected_state,
            include_rejected=True,
        )
        self.assertEqual(len(edges), 4)
        self.assertEqual(statistics.rejected_by_terrain, 1)
        self.assertTrue(any(record.reason == "terrain collision" for record in rejected))
        figure = plot_bellman_successor_debug(
            self.graph,
            model,
            selected_state,
        )
        names = [str(trace.name) for trace in figure.data]
        self.assertIn("Valid successors (4)", names)
        self.assertIn("Rejected terrain edges (1)", names)

    def test_stage4_candidate_certification_still_runs(self) -> None:
        for case_id, result in self.previous_results.items():
            los_result = result.los_result
            candidates = generate_switching_candidates(los_result.tangent_contour)
            evaluations = evaluate_switching_candidates(
                candidates,
                los_result.tangent_contour,
                los_result.terrain_map,
                los_result.mission_points,
            )
            with self.subTest(case=case_id):
                self.assertEqual(len(evaluations), 96)
                self.assertTrue(any(item.reachable for item in evaluations))
                self.assertTrue(any(not item.powered_feasible for item in evaluations))

    def test_prior_geometry_los_and_energy_match_frozen_baseline(self) -> None:
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
        for case_id in ("case_a", "case_e"):
            actual = self.previous_summaries[case_id]
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

    def test_graph_layer_has_no_later_stage_or_switching_dependency(self) -> None:
        directory = Path(__file__).resolve().parent
        imported_modules: set[str] = set()
        for filename in (
            "bellman_state.py",
            "bellman_geometry.py",
            "bellman_graph.py",
        ):
            tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "switching_candidates",
            "candidate_energy",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "hazard",
            "detection",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(
                    forbidden in module.lower() for module in imported_modules
                ))

        downstream_modules = (
            "map_geometry.py",
            "los_geometry.py",
            "energy_model.py",
            "reachability_surface.py",
            "switching_candidates.py",
            "candidate_energy.py",
        )
        for filename in downstream_modules:
            source = (directory / filename).read_text(encoding="utf-8")
            with self.subTest(downstream=filename):
                self.assertNotIn("bellman_graph", source)
                self.assertNotIn("bellman_geometry", source)
                self.assertNotIn("bellman_state", source)


if __name__ == "__main__":
    unittest.main()
