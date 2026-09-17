"""Stage-1 tests for the solver-facing terrain abstraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import numpy as np

from map_geometry import BoxObstacle, TerrainModel
from terrain_catalog import TERRAIN_LABELS, build_terrain


MESH_SHA256 = {
    "centered_cube": "21057f87470e5e7afa44e1f0a221f983362cdd071182a4d46513d9fdedec17c9",
    "offset_cube_left": "1042c1730005dd42ec79ff382373b0447c3333308a28cb17e7a20e69e6fdf818",
    "offset_cube_right": "1b0740d8b2a6cefd169e1977d29369b2241fa2fb6d98cacead10fc0761d9b133",
    "stepped_pyramid": "5548065eacb9ffdb19aa273f0b02acdf2acdd6ae141d696dd778e4bb3bd313e6",
}


def _mesh_digest(terrain: TerrainModel) -> str:
    ground = terrain.ground_mesh()
    obstacle = terrain.obstacle_mesh()
    digest = hashlib.sha256()
    for values in (
        ground.vertices,
        ground.triangles,
        obstacle.vertices,
        obstacle.triangles,
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _legacy_segment_intersects_box_interior(
    start: np.ndarray,
    end: np.ndarray,
    box: BoxObstacle,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    """Frozen copy of the pre-refactor powered-phase box test."""
    lower = np.array(
        [box.x_limits[0], box.y_limits[0], box.base_z], dtype=float,
    ) + tolerance
    upper = np.array(
        [box.x_limits[1], box.y_limits[1], box.top_z], dtype=float,
    ) - tolerance
    if np.any(lower >= upper):
        return False
    displacement = end - start
    entry, exit_ = 0.0, 1.0
    for axis in range(3):
        if abs(displacement[axis]) <= tolerance:
            if not lower[axis] < start[axis] < upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / displacement[axis]
        second = (upper[axis] - start[axis]) / displacement[axis]
        axis_entry, axis_exit = sorted((float(first), float(second)))
        entry = max(entry, axis_entry)
        exit_ = min(exit_, axis_exit)
        if entry >= exit_:
            return False
    return exit_ > max(entry, 0.0) and entry < 1.0


class TerrainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centered = build_terrain("centered_cube")

    def test_point_inside_uses_strict_interior_convention(self) -> None:
        probes = {
            "inside": (np.array([0.0, 0.0, 2.0]), True),
            "outside": (np.array([5.0, 0.0, 2.0]), False),
            "side_face": (np.array([2.0, 0.0, 2.0]), False),
            "top_face": (np.array([0.0, 0.0, 4.0]), False),
            "ground": (np.array([0.0, 0.0, 0.0]), False),
        }
        for name, (point, expected) in probes.items():
            with self.subTest(probe=name):
                self.assertEqual(self.centered.contains_solid(point), expected)

    def test_segment_collision_boundary_convention(self) -> None:
        probes = {
            "outside": (
                np.array([-8.0, 6.0, 2.0]),
                np.array([8.0, 6.0, 2.0]),
                False,
            ),
            "through_center": (
                np.array([-8.0, 0.0, 2.0]),
                np.array([8.0, 0.0, 2.0]),
                True,
            ),
            "ends_before_cube": (
                np.array([-8.0, 0.0, 2.0]),
                np.array([-3.0, 0.0, 2.0]),
                False,
            ),
            "starts_inside": (
                np.array([0.0, 0.0, 2.0]),
                np.array([5.0, 0.0, 2.0]),
                True,
            ),
            "side_face_tangent": (
                np.array([-8.0, 2.0, 2.0]),
                np.array([8.0, 2.0, 2.0]),
                False,
            ),
            "top_edge_tangent": (
                np.array([-8.0, 2.0, 4.0]),
                np.array([8.0, 2.0, 4.0]),
                False,
            ),
            "zero_length_outside": (
                np.array([5.0, 0.0, 2.0]),
                np.array([5.0, 0.0, 2.0]),
                False,
            ),
            "zero_length_inside": (
                np.array([0.0, 0.0, 2.0]),
                np.array([0.0, 0.0, 2.0]),
                True,
            ),
        }
        for name, (start, end, expected) in probes.items():
            with self.subTest(probe=name):
                self.assertEqual(
                    self.centered.segment_intersects_solid(start, end),
                    expected,
                )

    def test_ray_intersection_and_ground_selection(self) -> None:
        direct = self.centered.first_ray_hit(
            np.array([5.0, 0.0, 2.0]),
            np.array([-1.0, 0.0, 0.0]),
            include_ground=True,
        )
        self.assertIsNotNone(direct)
        self.assertAlmostEqual(direct.distance, 3.0)
        np.testing.assert_allclose(direct.point, [2.0, 0.0, 2.0], atol=1.0e-10)
        self.assertEqual((direct.mesh_index, direct.triangle_index), (1, 6))

        away = self.centered.first_ray_hit(
            np.array([5.0, 0.0, 2.0]), np.array([1.0, 0.0, 0.0]),
            include_ground=False,
        )
        self.assertIsNone(away)

        downward_origin = np.array([5.0, 0.0, 2.0])
        downward = np.array([0.0, 0.0, -1.0])
        ground_hit = self.centered.first_ray_hit(
            downward_origin, downward, include_ground=True,
        )
        self.assertIsNotNone(ground_hit)
        self.assertEqual(ground_hit.mesh_index, 0)
        self.assertAlmostEqual(ground_hit.distance, 2.0)
        np.testing.assert_allclose(ground_hit.point, [5.0, 0.0, 0.0])
        self.assertIsNone(self.centered.first_ray_hit(
            downward_origin, downward, include_ground=False,
        ))

        self.assertIsNone(self.centered.first_ray_hit(
            downward_origin, np.array([0.0, 0.0, 1.0]),
            include_ground=True,
        ))

        grazing_arguments = (
            np.array([5.0, 2.0, 4.0]), np.array([-1.0, 0.0, 0.0]),
        )
        first_grazing = self.centered.first_ray_hit(
            *grazing_arguments, include_ground=False,
        )
        second_grazing = self.centered.first_ray_hit(
            *grazing_arguments, include_ground=False,
        )
        self.assertIsNotNone(first_grazing)
        self.assertIsNotNone(second_grazing)
        np.testing.assert_allclose(first_grazing.point, [2.0, 2.0, 4.0])
        self.assertEqual(
            (first_grazing.distance, first_grazing.triangle_index),
            (second_grazing.distance, second_grazing.triangle_index),
        )

    def test_stepped_pyramid_uses_same_generic_contract(self) -> None:
        pyramid = build_terrain("stepped_pyramid")
        self.assertIsInstance(self.centered, TerrainModel)
        self.assertIsInstance(pyramid, TerrainModel)
        hit = pyramid.first_ray_hit(
            np.array([5.0, 0.0, 2.0]),
            np.array([-1.0, 0.0, 0.0]),
            include_ground=False,
        )
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.distance, 2.75)
        np.testing.assert_allclose(hit.point, [2.25, 0.0, 2.0])
        self.assertTrue(pyramid.segment_intersects_solid(
            np.array([5.0, 0.0, 2.0]), np.array([0.0, 0.0, 2.0]),
        ))

    def test_meshes_are_bitwise_identical_to_pre_refactor_baseline(self) -> None:
        for terrain_name, expected_digest in MESH_SHA256.items():
            with self.subTest(terrain=terrain_name):
                self.assertEqual(
                    _mesh_digest(build_terrain(terrain_name)), expected_digest,
                )

    def test_seeded_segments_match_legacy_box_collision(self) -> None:
        rng = np.random.default_rng(8272026)
        for terrain_name in TERRAIN_LABELS:
            terrain = build_terrain(terrain_name)
            for segment_index in range(100):
                start = np.array([
                    rng.uniform(terrain.bounds.x_min, terrain.bounds.x_max),
                    rng.uniform(terrain.bounds.y_min, terrain.bounds.y_max),
                    rng.uniform(terrain.ground_z, terrain.maximum_height + 2.0),
                ])
                end = np.array([
                    rng.uniform(terrain.bounds.x_min, terrain.bounds.x_max),
                    rng.uniform(terrain.bounds.y_min, terrain.bounds.y_max),
                    rng.uniform(terrain.ground_z, terrain.maximum_height + 2.0),
                ])
                legacy = any(
                    _legacy_segment_intersects_box_interior(start, end, box)
                    for box in terrain.obstacle_boxes()
                )
                generic = terrain.segment_intersects_solid(start, end)
                with self.subTest(
                    terrain=terrain_name, segment=segment_index,
                ):
                    self.assertEqual(generic, legacy)

    def test_solver_callers_contain_no_box_specific_loop(self) -> None:
        directory = Path(__file__).resolve().parent
        for filename in ("scenario.py", "energy_model.py"):
            source = (directory / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertNotIn("obstacle_boxes", source)
                self.assertNotIn("BoxObstacle", source)

    def test_invalid_query_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.centered.contains_solid(np.array([0.0, 0.0]))
        with self.assertRaises(ValueError):
            self.centered.contains_solid(
                np.array([0.0, 0.0, 2.0]), tolerance=-1.0,
            )
        with self.assertRaises(ValueError):
            self.centered.segment_intersects_solid(
                np.zeros(3), np.array([0.0, np.nan, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
