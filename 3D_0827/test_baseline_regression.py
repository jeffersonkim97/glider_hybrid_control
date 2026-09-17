"""Stage-0 regression freeze for terrain, LOS, and energy behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

from stage0_baseline import (
    CANONICAL_CASES,
    compute_baseline_case,
    summarize_baseline_case,
)


GEOMETRY_ATOL = 1.0e-6
ENERGY_RTOL = 1.0e-6
ENERGY_ATOL_J = 1.0e-6
REFERENCE_NOTEBOOK_SHA256 = (
    "3b5938317df97814facf1c25066777d4efcdeca28fc8116fe99cd8e63b3a09a2"
)


EXPECTED = {
    "case_a": {
        "terrain": "centered_cube",
        "sensor": [5.0, 0.0, 0.0],
        "bounds": [-10.0, 10.0, -8.0, 8.0],
        "maximum_height": 4.0,
        "obstacle_box_count": 1,
        "obstacle_triangle_count": 12,
        "tangent_ray_count": 213,
        "discarded_ground_candidate_count": 71,
        "contour_closed": False,
        "minimum_tangent_altitude": 0.026178010471204157,
        "los_surface_panel_count": 212,
        "reachable_face_count": 2938,
        "unreachable_face_count": 1997,
        "reachable_vertex_count": 1396,
        "unreachable_vertex_count": 909,
        "powered_infeasible_vertex_count": 768,
        "representative_reachable_point": [
            -7.0, 0.25068228335537784, 15.999999985376993,
        ],
        "representative_unreachable_point": [
            -7.0, 7.99999999325937, 0.10471204188481663,
        ],
        "representative_reachable_margin_j": 4299444.516554748,
        "representative_unreachable_margin_j": -542906.1737779398,
    },
    "case_b": {
        "terrain": "centered_cube",
        "sensor": [8.0, 0.0, 0.0],
        "bounds": [-10.0, 10.0, -8.0, 8.0],
        "maximum_height": 4.0,
        "obstacle_box_count": 1,
        "obstacle_triangle_count": 12,
        "tangent_ray_count": 213,
        "discarded_ground_candidate_count": 71,
        "contour_closed": False,
        "minimum_tangent_altitude": 0.03433476394849782,
        "los_surface_panel_count": 212,
        "reachable_face_count": 2983,
        "unreachable_face_count": 2083,
        "reachable_vertex_count": 1361,
        "unreachable_vertex_count": 944,
        "powered_infeasible_vertex_count": 750,
        "representative_reachable_point": [
            -15.999999999999996, 3.2516316561480583, 15.9999999888738,
        ],
        "representative_unreachable_point": [
            -16.0, 7.999999995906978, 0.13733905579399128,
        ],
        "representative_reachable_margin_j": 3930579.532856119,
        "representative_unreachable_margin_j": -819341.7807836838,
    },
    "case_c": {
        "terrain": "offset_cube_left",
        "sensor": [5.0, 0.0, 0.0],
        "bounds": [-10.0, 10.0, -8.0, 8.0],
        "maximum_height": 4.0,
        "obstacle_box_count": 1,
        "obstacle_triangle_count": 12,
        "tangent_ray_count": 213,
        "discarded_ground_candidate_count": 71,
        "contour_closed": False,
        "minimum_tangent_altitude": 0.046296296276994824,
        "los_surface_panel_count": 212,
        "reachable_face_count": 3725,
        "unreachable_face_count": 1442,
        "reachable_vertex_count": 1730,
        "unreachable_vertex_count": 575,
        "powered_infeasible_vertex_count": 314,
        "representative_reachable_point": [
            -7.0, -4.475195835459102, 15.999999998335426,
        ],
        "representative_unreachable_point": [
            -22.99999991622162, -4.0, 0.23456790058812965,
        ],
        "representative_reachable_margin_j": 4249935.955819707,
        "representative_unreachable_margin_j": -987539.5079464513,
    },
    "case_d": {
        "terrain": "offset_cube_right",
        "sensor": [5.0, 0.0, 0.0],
        "bounds": [-10.0, 10.0, -8.0, 8.0],
        "maximum_height": 4.0,
        "obstacle_box_count": 1,
        "obstacle_triangle_count": 12,
        "tangent_ray_count": 213,
        "discarded_ground_candidate_count": 71,
        "contour_closed": False,
        "minimum_tangent_altitude": 0.046296296276994824,
        "los_surface_panel_count": 212,
        "reachable_face_count": 3705,
        "unreachable_face_count": 1422,
        "reachable_vertex_count": 1730,
        "unreachable_vertex_count": 575,
        "powered_infeasible_vertex_count": 314,
        "representative_reachable_point": [
            -7.000000000000002, 4.475195835459109, 15.999999998335428,
        ],
        "representative_unreachable_point": [
            -22.99999991622164, 4.0, 0.23456790058821214,
        ],
        "representative_reachable_margin_j": 4249935.955819707,
        "representative_unreachable_margin_j": -987539.5079464274,
    },
    "case_e": {
        "terrain": "stepped_pyramid",
        "sensor": [5.0, 0.0, 0.0],
        "bounds": [-10.0, 10.0, -8.0, 8.0],
        "maximum_height": 4.0,
        "obstacle_box_count": 3,
        "obstacle_triangle_count": 36,
        "tangent_ray_count": 213,
        "discarded_ground_candidate_count": 71,
        "contour_closed": False,
        "minimum_tangent_altitude": 0.015942606616181708,
        "los_surface_panel_count": 212,
        "reachable_face_count": 2850,
        "unreachable_face_count": 2097,
        "reachable_vertex_count": 1350,
        "unreachable_vertex_count": 955,
        "powered_infeasible_vertex_count": 876,
        "representative_reachable_point": [
            -9.0, 2.562698152424864, 15.999999986285253,
        ],
        "representative_unreachable_point": [
            -3.0, 11.99999999980995, 0.06377042646472683,
        ],
        "representative_reachable_margin_j": 4184042.244129398,
        "representative_unreachable_margin_j": -536961.743358558,
    },
}


class BaselineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {
            case.case_id: compute_baseline_case(case)
            for case in CANONICAL_CASES
        }
        cls.summaries = {
            case.case_id: summarize_baseline_case(
                case, cls.results[case.case_id],
            )
            for case in CANONICAL_CASES
        }

    def test_reference_notebook_is_unchanged(self) -> None:
        notebook = Path(__file__).with_name("3D_bellman_0827.ipynb")
        digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
        self.assertEqual(digest, REFERENCE_NOTEBOOK_SHA256)

    def test_canonical_numerical_regression(self) -> None:
        integer_fields = (
            "obstacle_box_count",
            "obstacle_triangle_count",
            "tangent_ray_count",
            "discarded_ground_candidate_count",
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
        for case_id, expected in EXPECTED.items():
            with self.subTest(case=case_id):
                actual = self.summaries[case_id]
                self.assertEqual(actual["terrain"], expected["terrain"])
                self.assertEqual(
                    actual["contour_closed"], expected["contour_closed"],
                )
                for field in integer_fields:
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

    def test_ground_candidates_are_excluded(self) -> None:
        for case in CANONICAL_CASES:
            with self.subTest(case=case.case_id):
                result = self.results[case.case_id]
                contour = result.los_result.tangent_contour
                ground = result.los_result.terrain_map.ground_z
                self.assertFalse(contour.closed)
                self.assertGreater(contour.discarded_ground_candidate_count, 0)
                self.assertGreater(
                    float(np.min(contour.tangent_points[:, 2])),
                    ground + 1.0e-6,
                )

    def test_all_terrain_categories_render(self) -> None:
        rendered_categories = {
            case.terrain_category for case in CANONICAL_CASES
            if len(self.results[case.case_id].figure.data) > 0
        }
        self.assertEqual(
            rendered_categories,
            {
                "centered_cube",
                "offset_cube_left",
                "offset_cube_right",
                "stepped_pyramid",
            },
        )

    def test_centered_cube_has_both_energy_classes(self) -> None:
        surface = self.results["case_a"].energy_surface
        self.assertGreater(surface.reachable_face_count, 0)
        self.assertGreater(surface.unreachable_face_count, 0)
        self.assertGreater(np.count_nonzero(surface.vertex_reachable), 0)
        self.assertGreater(np.count_nonzero(~surface.vertex_reachable), 0)


if __name__ == "__main__":
    unittest.main()
