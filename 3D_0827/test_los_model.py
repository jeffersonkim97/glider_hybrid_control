"""Stage-2 unit and numerical regression tests for ``LOSModel``."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from los_geometry import LOSModel, trace_terrain_tangent_contour
from scenario import Point3D
from terrain_catalog import build_terrain
from visualization import _crease_edges


GEOMETRY_ATOL = 1.0e-6
GROUND_CLEARANCE = 1.0e-6


def _set_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(max(np.max(np.min(distances, axis=1)), np.max(np.min(distances, axis=0))))


class LOSModelTests(unittest.TestCase):
    def test_l1_centered_cube_matches_legacy_api_exactly(self) -> None:
        terrain = build_terrain("centered_cube")
        sensor = Point3D(5.0, 0.0, terrain.ground_z)
        legacy = trace_terrain_tangent_contour(
            terrain.surface_meshes(),
            sensor,
            target_mesh_index=1,
            ground_height=terrain.ground_z,
            probe_grid_size=101,
            boundary_refinement_steps=24,
        )
        los_model = LOSModel(terrain)
        current = los_model.trace_tangent_contour(sensor)
        surface = los_model.build_tangent_surface(current)

        self.assertEqual(len(current.rays), len(legacy.rays))
        self.assertEqual(current.closed, legacy.closed)
        self.assertEqual(
            current.discarded_ground_candidate_count,
            legacy.discarded_ground_candidate_count,
        )
        np.testing.assert_allclose(
            current.tangent_points,
            legacy.tangent_points,
            rtol=0.0,
            atol=GEOMETRY_ATOL,
        )
        self.assertAlmostEqual(
            float(np.min(current.tangent_points[:, 2])),
            float(np.min(legacy.tangent_points[:, 2])),
            delta=GEOMETRY_ATOL,
        )
        self.assertEqual(surface.panel_count, len(current.rays) - 1)

    def test_l2_sensor_translation_is_finite_and_mirrored(self) -> None:
        terrain = build_terrain("centered_cube")
        contours = {}
        for sensor_x, sensor_y in ((5.0, -3.0), (5.0, 0.0), (5.0, 3.0), (8.0, 0.0)):
            with self.subTest(sensor=(sensor_x, sensor_y)):
                contour = LOSModel(terrain).trace_tangent_contour(
                    Point3D(sensor_x, sensor_y, terrain.ground_z),
                    probe_grid_size=51,
                )
                contours[(sensor_x, sensor_y)] = contour
                directions = np.vstack([ray.unit_direction for ray in contour.rays])
                self.assertTrue(np.all(np.isfinite(directions)))
                self.assertTrue(np.all(np.linalg.norm(directions, axis=1) > 0.0))
                self.assertGreater(
                    float(np.min(contour.tangent_points[:, 2])),
                    terrain.ground_z + GROUND_CLEARANCE - GEOMETRY_ATOL,
                )
                self.assertGreater(contour.discarded_ground_candidate_count, 0)

        reflected = contours[(5.0, -3.0)].tangent_points.copy()
        reflected[:, 1] *= -1.0
        self.assertLessEqual(
            _set_distance(reflected, contours[(5.0, 3.0)].tangent_points),
            GEOMETRY_ATOL,
        )

    def test_l3_offset_terrains_have_mirrored_tangent_geometry(self) -> None:
        sensor = Point3D(5.0, 0.0, 0.0)
        left = LOSModel(build_terrain("offset_cube_left")).trace_tangent_contour(
            sensor, probe_grid_size=51,
        )
        right = LOSModel(build_terrain("offset_cube_right")).trace_tangent_contour(
            sensor, probe_grid_size=51,
        )
        reflected = left.tangent_points.copy()
        reflected[:, 1] *= -1.0
        self.assertLessEqual(
            _set_distance(reflected, right.tangent_points),
            GEOMETRY_ATOL,
        )

    def test_l4_stepped_pyramid_uses_same_public_api(self) -> None:
        terrain = build_terrain("stepped_pyramid")
        model = LOSModel(terrain)
        contour = model.trace_tangent_contour(Point3D(5.0, 0.0, 0.0), probe_grid_size=51)
        self.assertGreater(len(contour.rays), 0)
        self.assertFalse(contour.closed)
        self.assertEqual(model.build_tangent_surface(contour).panel_count, len(contour.rays) - 1)

    def test_l5_retained_rays_first_hit_obstacle_and_not_ground(self) -> None:
        terrain = build_terrain("centered_cube")
        contour = LOSModel(terrain).trace_tangent_contour(
            Point3D(5.0, 0.0, 0.0), probe_grid_size=51,
        )
        self.assertGreater(contour.discarded_ground_candidate_count, 0)
        self.assertGreater(
            float(np.min(contour.tangent_points[:, 2])),
            terrain.ground_z + GROUND_CLEARANCE - GEOMETRY_ATOL,
        )
        for ray in contour.rays:
            hit = terrain.first_ray_hit(
                ray.origin.as_array(), ray.unit_direction, include_ground=True,
            )
            self.assertIsNotNone(hit)
            assert hit is not None
            self.assertEqual(hit.mesh_index, 1)
            np.testing.assert_allclose(
                hit.point, ray.tangent_point.as_array(), rtol=0.0, atol=GEOMETRY_ATOL,
            )

    def test_l6_direct_los_boundary_semantics(self) -> None:
        model = LOSModel(build_terrain("centered_cube"))
        self.assertTrue(model.has_line_of_sight(
            np.array([5.0, 5.0, 2.0]), np.array([-5.0, 5.0, 2.0]),
        ))
        self.assertFalse(model.has_line_of_sight(
            np.array([5.0, 0.0, 2.0]), np.array([-5.0, 0.0, 2.0]),
        ))
        self.assertTrue(model.has_line_of_sight(
            np.array([5.0, 2.0, 4.0]), np.array([-5.0, 2.0, 4.0]),
        ))
        self.assertTrue(model.has_line_of_sight(
            np.array([5.0, 5.0, 0.0]), np.array([-5.0, 5.0, 0.0]),
        ))
        self.assertTrue(model.has_line_of_sight(
            np.array([5.0, 0.0, 2.0]), np.array([5.0, 0.0, 2.0]),
        ))
        self.assertFalse(model.has_line_of_sight(
            np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 2.0]),
        ))
        with self.assertRaises(ValueError):
            model.has_line_of_sight(
                np.array([5.0, 0.0, -0.1]), np.array([5.0, 0.0, 2.0]),
            )
        with self.assertRaises(ValueError):
            model.has_line_of_sight(
                np.array([np.nan, 0.0, 0.0]), np.array([5.0, 0.0, 2.0]),
            )

    def test_renderer_uses_generic_mesh_crease_edges(self) -> None:
        self.assertEqual(len(_crease_edges(build_terrain("centered_cube").obstacle_mesh())), 12)
        self.assertEqual(len(_crease_edges(build_terrain("stepped_pyramid").obstacle_mesh())), 36)
        source = (Path(__file__).resolve().parent / "visualization.py").read_text(
            encoding="utf-8",
        )
        self.assertNotIn("obstacle_boxes", source)

    def test_production_caller_hides_low_level_mesh_contract(self) -> None:
        source = (Path(__file__).resolve().parent / "los_explorer_gui.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("LOSModel(terrain_map)", source)
        self.assertNotIn("target_mesh_index", source)
        self.assertNotIn("surface_meshes", source)
        self.assertNotIn("trace_terrain_tangent_contour", source)


if __name__ == "__main__":
    unittest.main()
