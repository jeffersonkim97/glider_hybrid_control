"""Progressive Stage-6 exit gate across time and all prior pipelines."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import unittest

import numpy as np

from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_solver import (
    TimeOptimalRun,
    replay_time_optimal_path,
    solve_time_optimal,
)
from bellman_state import BellmanState, BellmanStateGrid
from los_explorer_gui import create_energy_explorer, create_los_explorer
from map_geometry import MapBounds
from scenario import Point3D
from solver_metrics import SolverMetrics, SolverTiming
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from terrain_catalog import build_terrain
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL
from test_stage5_exit_integration import EXPECTED_GRAPH_STATISTICS
from visualization import plot_time_optimal_trajectory


EXPECTED_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)
EXPECTED_CANONICAL_TIME_S = 79.45584412271572


class Stage6ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(__file__).resolve().parent
        cls.bounds = MapBounds(-8.0, 8.0, -4.0, 4.0)
        cls.grid = BellmanStateGrid(bounds=cls.bounds)
        cls.terrain = build_terrain("centered_cube", bounds=cls.bounds)
        cls.graph = build_bellman_graph(
            cls.grid,
            cls.terrain,
            Point3D(8.0, 0.0, 0.0),
        )
        cls.unit_solution = solve_unit_cost_reachability(cls.graph)
        cls.time_solution = solve_time_optimal(cls.graph)
        cls.start_state = BellmanState(0, 4, 20, 0)
        cls.start_id = cls.grid.encode(cls.start_state)
        cls.path = replay_time_optimal_path(cls.time_solution, cls.start_id)

        selected = {
            case.case_id: case
            for case in CANONICAL_CASES
            if case.case_id in ("case_a", "case_e")
        }
        cls.previous_summaries = {}
        for case_id in ("case_a", "case_e"):
            result = compute_baseline_case(selected[case_id])
            cls.previous_summaries[case_id] = summarize_baseline_case(
                selected[case_id],
                result,
            )

    def test_canonical_time_solution_and_replay_certificate(self) -> None:
        self.assertEqual(vars(self.graph.statistics), EXPECTED_GRAPH_STATISTICS)
        self.assertAlmostEqual(
            self.time_solution.value_s[self.start_id],
            EXPECTED_CANONICAL_TIME_S,
            places=12,
        )
        self.assertAlmostEqual(
            self.path.bellman_value_s,
            self.path.summed_edge_duration_s,
            places=12,
        )
        self.assertAlmostEqual(
            self.path.summed_edge_duration_s,
            self.path.geometric_replay_duration_s,
            places=12,
        )
        self.assertEqual(len(self.path.edges), 16)
        self.assertTrue(self.graph.terminal_mask[self.path.state_ids[-1]])
        self.assertTrue(all(
            not self.terrain.segment_intersects_solid(
                self.grid.position_map(edge.source_state),
                self.grid.position_map(edge.target_state),
            )
            for edge in self.path.edges
        ))

    def test_stage5_unit_cost_solution_is_unchanged(self) -> None:
        self.assertEqual(self.unit_solution.goal_reachable_state_count, 43136)
        self.assertEqual(self.unit_solution.maximum_finite_value, 25.0)
        np.testing.assert_array_equal(
            self.time_solution.goal_reachable,
            self.unit_solution.goal_reachable,
        )

    def test_time_visualization_consumes_stored_result(self) -> None:
        run = TimeOptimalRun(
            graph=self.graph,
            solution=self.time_solution,
            path=self.path,
            start_state_id=self.start_id,
            metrics=SolverMetrics(
                timing=SolverTiming(0.0, 0.0, 0.0),
                state_count=self.graph.statistics.state_count,
                edge_count=self.graph.statistics.valid_edge_count,
            ),
        )
        figure = plot_time_optimal_trajectory(run)
        names = [str(trace.name) for trace in figure.data]
        self.assertTrue(any(name.startswith("Goal-reachable positions") for name in names))
        self.assertIn("Goal", names)
        self.assertIn("Goal tolerance (25 m)", names)
        self.assertIn("Time-optimal trajectory (16 edges)", names)
        self.assertIn("Selected start", names)
        self.assertIn("Trajectory direction", names)
        self.assertIn("Bellman=79.455844 s", figure.layout.title.text)
        self.assertIn("replay=79.455844 s", figure.layout.title.text)

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
                        actual[field],
                        expected[field],
                        rtol=0.0,
                        atol=GEOMETRY_ATOL,
                        err_msg=f"{case_id}: {field}",
                    )
                for field in energy_fields:
                    np.testing.assert_allclose(
                        actual[field],
                        expected[field],
                        rtol=ENERGY_RTOL,
                        atol=ENERGY_ATOL_J,
                        err_msg=f"{case_id}: {field}",
                    )

    def test_existing_gui_controllers_and_reference_notebook_survive(self) -> None:
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

        notebook_hash = hashlib.sha256(
            (self.directory / "3D_bellman_0827.ipynb").read_bytes()
        ).hexdigest()
        self.assertEqual(notebook_hash, EXPECTED_NOTEBOOK_SHA256)

    def test_time_layer_is_one_way_and_has_no_future_dependency(self) -> None:
        imported_modules: set[str] = set()
        for filename in (
            "bellman_objectives.py",
            "bellman_solver.py",
            "solver_metrics.py",
        ):
            source = (self.directory / filename).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "hazard",
            "detection",
            "candidate_energy",
            "switching_candidates",
            "game_types",
            "attacker_best_response",
            "stackelberg_interface",
            "defender",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for module in imported_modules
                ))

        for filename in (
            "map_geometry.py",
            "los_geometry.py",
            "energy_model.py",
            "reachability_surface.py",
            "switching_candidates.py",
            "candidate_energy.py",
            "bellman_state.py",
            "bellman_geometry.py",
            "bellman_graph.py",
        ):
            source = (self.directory / filename).read_text(encoding="utf-8")
            with self.subTest(upstream=filename):
                self.assertNotIn("bellman_solver", source)
                self.assertNotIn("bellman_objectives", source)


if __name__ == "__main__":
    unittest.main()
