"""Progressive Stage-2 exit gate across LOS, energy, and GUI callers."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import numpy as np

from los_explorer_gui import create_energy_explorer, create_los_explorer
from los_geometry import LOSModel
from stage0_baseline import CANONICAL_CASES, compute_baseline_case, summarize_baseline_case
from test_baseline_regression import ENERGY_ATOL_J, ENERGY_RTOL, EXPECTED, GEOMETRY_ATOL


EXIT_CASE_IDS = ("case_a", "case_e")


class Stage2ExitIntegrationTests(unittest.TestCase):
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

    def test_centered_cube_and_pyramid_flow_through_los_model(self) -> None:
        for case_id, result in self.results.items():
            with self.subTest(case=case_id):
                los_result = result.los_result
                self.assertIsInstance(los_result.los_model, LOSModel)
                self.assertIs(los_result.los_model.terrain, los_result.terrain_map)
                self.assertGreater(len(los_result.tangent_contour.rays), 0)
                self.assertGreater(result.energy_surface.reachable_face_count, 0)
                self.assertGreater(result.energy_surface.unreachable_face_count, 0)

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

    def test_high_level_callers_do_not_gain_concrete_terrain_knowledge(self) -> None:
        directory = Path(__file__).resolve().parent
        forbidden = ("BoxObstacle", "CompositeTerrainMap", "obstacle_boxes")
        for filename in (
            "los_explorer_gui.py",
            "visualization.py",
            "scenario.py",
            "energy_model.py",
            "glide_reachability.py",
            "reachability_surface.py",
        ):
            source = (directory / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                for concrete_detail in forbidden:
                    self.assertNotIn(concrete_detail, source)

        gui_source = (directory / "los_explorer_gui.py").read_text(encoding="utf-8")
        self.assertNotIn("target_mesh_index", gui_source)
        self.assertNotIn("surface_meshes", gui_source)


if __name__ == "__main__":
    unittest.main()
