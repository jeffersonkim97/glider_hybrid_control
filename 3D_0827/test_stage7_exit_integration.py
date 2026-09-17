"""Progressive Stage-7 exit gate across hazard and all prior pipelines."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import unittest

import numpy as np

from bellman_geometry import GlideEdge
from bellman_graph import build_bellman_graph
from bellman_solver import replay_time_optimal_path, solve_time_optimal
from bellman_state import BellmanState, BellmanStateGrid
from detection_hazard import (
    DEFAULT_ATTACKER_HAZARD_TIME,
    DEFAULT_DETECTION_HAZARD,
    GlideDetectionHazardModel,
    evaluate_attacker_hazard_time_objective,
)
from edge_hazard import integrate_edge_hazard
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from los_explorer_gui import create_energy_explorer, create_los_explorer
from map_geometry import MapBounds
from scenario import Point3D
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from terrain_catalog import build_terrain
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL
from test_stage5_exit_integration import EXPECTED_GRAPH_STATISTICS
from visualization import plot_edge_hazard_audit


EXPECTED_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)
EXPECTED_CANONICAL_TIME_S = 79.45584412271572
EXPECTED_CROSSING_HAZARD = 3.2469110755758525e-05


def _state_at(
    grid: BellmanStateGrid,
    x: float,
    y: float,
    altitude: float,
    heading_bin: int,
) -> BellmanState:
    return BellmanState(
        int(round((x - grid.bounds.x_min) / grid.horizontal_spacing_map)),
        int(round((y - grid.bounds.y_min) / grid.horizontal_spacing_map)),
        int(round(altitude / grid.altitude_spacing_map)),
        heading_bin,
    )


def _edge_between(
    grid: BellmanStateGrid,
    source_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    heading_bin: int,
) -> GlideEdge:
    source = _state_at(grid, *source_xyz, heading_bin)
    target = _state_at(grid, *target_xyz, heading_bin)
    source_position = grid.position_map(source)
    target_position = grid.position_map(target)
    horizontal_distance_m = DEFAULT_PHYSICAL_SCALE.distance_m(float(
        np.linalg.norm(target_position[:2] - source_position[:2])
    ))
    return GlideEdge(
        source_id=grid.encode(source),
        target_id=grid.encode(target),
        source_state=source,
        target_state=target,
        horizontal_distance_m=horizontal_distance_m,
        altitude_loss_m=DEFAULT_PHYSICAL_SCALE.distance_m(
            float(source_position[2] - target_position[2])
        ),
        duration_s=horizontal_distance_m / DEFAULT_GLIDER.best_glide_speed_mps,
        heading_change_rad=0.0,
    )


class Stage7ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = Path(__file__).resolve().parent
        cls.hazard_bounds = MapBounds(-8.0, 8.0, -8.0, 8.0)
        cls.hazard_grid = BellmanStateGrid(
            bounds=cls.hazard_bounds,
            maximum_altitude_map=2.0,
        )
        cls.terrain = build_terrain("centered_cube", bounds=cls.hazard_bounds)
        cls.sensor = Point3D(5.0, 0.0, 0.0)
        cls.hazard_model = GlideDetectionHazardModel(cls.terrain, cls.sensor)
        cls.crossing_edge = _edge_between(
            cls.hazard_grid,
            (-3.0, 4.0, 1.0),
            (-3.0, 6.0, 0.8),
            2,
        )
        cls.crossing_result = integrate_edge_hazard(
            cls.crossing_edge,
            cls.hazard_grid,
            cls.hazard_model,
            quadrature_resolution=33,
        )

        graph_bounds = MapBounds(-8.0, 8.0, -4.0, 4.0)
        cls.graph_grid = BellmanStateGrid(bounds=graph_bounds)
        cls.graph = build_bellman_graph(
            cls.graph_grid,
            build_terrain("centered_cube", bounds=graph_bounds),
            Point3D(8.0, 0.0, 0.0),
        )
        cls.time_solution = solve_time_optimal(cls.graph)
        cls.time_start_id = cls.graph_grid.encode(BellmanState(0, 4, 20, 0))
        cls.time_path = replay_time_optimal_path(
            cls.time_solution,
            cls.time_start_id,
        )

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

    def test_crossing_edge_is_sample_auditable_and_deterministic(self) -> None:
        result = self.crossing_result
        self.assertAlmostEqual(result.hazard, EXPECTED_CROSSING_HAZARD, places=15)
        self.assertEqual(result.quadrature_resolution, 33)
        self.assertEqual(result.visible_sample_count, 11)
        self.assertEqual(result.occluded_sample_count, 22)
        self.assertAlmostEqual(
            result.hazard,
            sum(sample.hazard_contribution for sample in result.samples),
            places=15,
        )
        self.assertTrue(all(
            sample.total_rate_per_s == 0.0
            for sample in result.samples
            if not sample.visible
        ))
        self.assertTrue(all(
            sample.total_rate_per_s > 0.0
            for sample in result.samples
            if sample.visible
        ))

    def test_hazard_constants_and_objective_match_intended_v2(self) -> None:
        self.assertEqual(DEFAULT_DETECTION_HAZARD.range_floor_m, 10.0)
        self.assertEqual(DEFAULT_DETECTION_HAZARD.radar_coefficient, 1.3e7)
        self.assertEqual(DEFAULT_DETECTION_HAZARD.doppler_coefficient, 3.325e4)
        self.assertEqual(DEFAULT_ATTACKER_HAZARD_TIME.hazard_weight, 0.5)
        self.assertEqual(DEFAULT_ATTACKER_HAZARD_TIME.time_weight, 0.5)
        self.assertEqual(DEFAULT_ATTACKER_HAZARD_TIME.hazard_reference, 1.0)
        self.assertEqual(
            DEFAULT_ATTACKER_HAZARD_TIME.time_reference_s,
            5000.0 / 22.6,
        )
        breakdown = evaluate_attacker_hazard_time_objective(0.4, 100.0)
        self.assertAlmostEqual(
            breakdown.objective_value,
            0.5 * 0.4 + 0.5 * 100.0 / (5000.0 / 22.6),
            places=15,
        )

    def test_hazard_visualization_contains_required_audit_layers(self) -> None:
        figure = plot_edge_hazard_audit(
            self.terrain,
            self.sensor,
            self.crossing_result,
        )
        names = [str(trace.name) for trace in figure.data]
        self.assertIn("Sensor", names)
        self.assertIn("Selected edge", names)
        self.assertIn("Visible quadrature samples (11)", names)
        self.assertIn("Occluded quadrature samples (22)", names)
        self.assertIn("Visible sensor rays", names)
        self.assertIn("Occluded sensor rays", names)
        self.assertIn("Edge direction", names)
        self.assertIn("visible=11/33", figure.layout.title.text)

    def test_nontrivial_second_terrain_uses_the_same_hazard_interface(self) -> None:
        pyramid = build_terrain("stepped_pyramid", bounds=self.hazard_bounds)
        model = GlideDetectionHazardModel(pyramid, self.sensor)
        result = integrate_edge_hazard(
            self.crossing_edge,
            self.hazard_grid,
            model,
            quadrature_resolution=16,
        )
        self.assertTrue(np.isfinite(result.hazard))
        self.assertGreaterEqual(result.hazard, 0.0)
        self.assertEqual(result.quadrature_resolution, 16)

    def test_stage6_time_solution_remains_at_frozen_geometry(self) -> None:
        self.assertEqual(vars(self.graph.statistics), EXPECTED_GRAPH_STATISTICS)
        self.assertEqual(self.time_solution.goal_reachable_state_count, 43136)
        self.assertAlmostEqual(
            self.time_path.bellman_value_s,
            EXPECTED_CANONICAL_TIME_S,
            places=12,
        )
        self.assertAlmostEqual(
            self.time_path.bellman_value_s,
            self.time_path.geometric_replay_duration_s,
            places=12,
        )

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
                    )
                for field in energy_fields:
                    np.testing.assert_allclose(
                        actual[field], expected[field], rtol=ENERGY_RTOL, atol=ENERGY_ATOL_J,
                    )

    def test_existing_gui_and_reference_notebook_survive(self) -> None:
        with redirect_stdout(io.StringIO()):
            los_gui = create_los_explorer(probe_grid_size=25)
            energy_gui = create_energy_explorer(
                probe_grid_size=25,
                radial_section_count=7,
                contour_section_count=24,
            )
            los_gui.terrain_toggle.value = "stepped_pyramid"
            energy_gui.terrain_toggle.value = "stepped_pyramid"
        for controller in (los_gui, energy_gui):
            self.assertIsNone(controller.latest_error)
            self.assertIsNotNone(controller.latest_result)
            self.assertIn("Ready", controller.status.value)
        notebook_hash = hashlib.sha256(
            (self.directory / "3D_bellman_0827.ipynb").read_bytes()
        ).hexdigest()
        self.assertEqual(notebook_hash, EXPECTED_NOTEBOOK_SHA256)

    def test_dependency_direction_stops_before_stage8(self) -> None:
        imported_modules: set[str] = set()
        for filename in ("detection_hazard.py", "edge_hazard.py"):
            tree = ast.parse((self.directory / filename).read_text(encoding="utf-8"))
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
        for forbidden in (
            "bellman_solver",
            "bellman_graph",
            "switching_candidates",
            "candidate_energy",
            "attacker_best_response",
            "stackelberg_interface",
            "defender",
        ):
            self.assertFalse(any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for module in imported_modules
            ))
        hazard_source = (self.directory / "detection_hazard.py").read_text(
            encoding="utf-8",
        )
        self.assertNotIn("BoxObstacle", hazard_source)
        self.assertNotIn("centered_cube", hazard_source)
        for filename in (
            "bellman_graph.py",
            "bellman_solver.py",
            "bellman_objectives.py",
        ):
            source = (self.directory / filename).read_text(encoding="utf-8")
            self.assertNotIn("detection_hazard", source)
            self.assertNotIn("edge_hazard", source)


if __name__ == "__main__":
    unittest.main()
