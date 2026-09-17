"""Stage-1 exit gate against the existing unrefactored downstream pipeline."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from los_explorer_gui import (
    create_energy_explorer,
    create_los_explorer,
)
from map_geometry import TerrainModel
from stage0_baseline import (
    CANONICAL_CASES,
    compute_baseline_case,
    summarize_baseline_case,
)
from test_baseline_regression import (
    ENERGY_ATOL_J,
    ENERGY_RTOL,
    EXPECTED,
    GEOMETRY_ATOL,
)


EXIT_CASE_IDS = ("case_a", "case_e")


class Stage1ExitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cases = {
            case.case_id: case for case in CANONICAL_CASES
            if case.case_id in EXIT_CASE_IDS
        }
        cls.results = {
            case_id: compute_baseline_case(cases[case_id])
            for case_id in EXIT_CASE_IDS
        }
        cls.summaries = {
            case_id: summarize_baseline_case(cases[case_id], cls.results[case_id])
            for case_id in EXIT_CASE_IDS
        }

    def test_existing_los_pipeline_accepts_terrain_model(self) -> None:
        for case_id in EXIT_CASE_IDS:
            with self.subTest(case=case_id):
                result = self.results[case_id]
                terrain = result.los_result.terrain_map
                contour = result.los_result.tangent_contour
                self.assertIsInstance(terrain, TerrainModel)
                self.assertGreater(len(contour.rays), 0)
                self.assertFalse(contour.closed)
                self.assertGreater(contour.discarded_ground_candidate_count, 0)
                self.assertEqual(
                    result.los_result.los_surface.panel_count,
                    len(contour.rays) - 1,
                )

    def test_geometry_los_and_energy_match_frozen_baseline(self) -> None:
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
            with self.subTest(case=case_id):
                actual = self.summaries[case_id]
                expected = EXPECTED[case_id]
                for field in exact_fields:
                    self.assertEqual(actual[field], expected[field], field)
                for field in geometry_fields:
                    np.testing.assert_allclose(
                        actual[field], expected[field],
                        rtol=0.0, atol=GEOMETRY_ATOL,
                        err_msg=f"{case_id}: {field}",
                    )
                for field in energy_fields:
                    np.testing.assert_allclose(
                        actual[field], expected[field],
                        rtol=ENERGY_RTOL, atol=ENERGY_ATOL_J,
                        err_msg=f"{case_id}: {field}",
                    )

    def test_existing_gui_controllers_compute_without_error(self) -> None:
        # Low display resolution keeps this a smoke test; the default-resolution
        # centered/pyramid computations are validated independently above.
        with redirect_stdout(io.StringIO()):
            los_gui = create_los_explorer(probe_grid_size=25)
            energy_gui = create_energy_explorer(
                probe_grid_size=25,
                radial_section_count=7,
                contour_section_count=24,
            )
            los_gui.terrain_toggle.value = "stepped_pyramid"
            energy_gui.terrain_toggle.value = "stepped_pyramid"
        for name, controller in (
            ("los", los_gui), ("energy", energy_gui),
        ):
            with self.subTest(controller=name):
                self.assertIsNone(controller.latest_error)
                self.assertIsNotNone(controller.latest_result)
                self.assertIn("Ready", controller.status.value)

    def test_downstream_solver_modules_do_not_know_box_implementation(self) -> None:
        directory = Path(__file__).resolve().parent
        solver_modules = (
            "scenario.py",
            "energy_model.py",
            "glide_reachability.py",
            "reachability_surface.py",
            "los_geometry.py",
        )
        for filename in solver_modules:
            source = (directory / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn("BoxObstacle", source)
                self.assertNotIn("obstacle_boxes", source)


if __name__ == "__main__":
    unittest.main()
