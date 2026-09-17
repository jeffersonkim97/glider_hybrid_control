"""Stage-3 switching-candidate geometry and determinism tests."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import numpy as np

from los_geometry import LOSModel
from scenario import Point3D
from switching_candidates import (
    INITIAL_CONTOUR_SAMPLE_COUNT,
    INITIAL_RADIAL_SCALES,
    SwitchingCandidate,
    build_switching_candidate,
    generate_switching_candidates,
)
from terrain_catalog import build_terrain


SURFACE_RESIDUAL_TOLERANCE = 1.0e-8
TERRAIN_CASES = (
    "centered_cube",
    "offset_cube_left",
    "offset_cube_right",
    "stepped_pyramid",
)


class SwitchingCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contours = {}
        for terrain_name in TERRAIN_CASES:
            terrain = build_terrain(terrain_name)
            cls.contours[terrain_name] = LOSModel(terrain).trace_tangent_contour(
                Point3D(5.0, 0.0, terrain.ground_z),
                probe_grid_size=51,
            )

    def test_s3_1_open_contour_endpoints_do_not_wrap(self) -> None:
        contour = self.contours["centered_cube"]
        origin = contour.origin.as_array()
        scale = 2.0
        at_zero = build_switching_candidate(
            contour, candidate_id=0, contour_fraction=0.0, radial_scale=scale,
        )
        at_half = build_switching_candidate(
            contour, candidate_id=1, contour_fraction=0.5, radial_scale=scale,
        )
        near_one = build_switching_candidate(
            contour, candidate_id=2, contour_fraction=1.0 - 1.0e-10, radial_scale=scale,
        )
        at_one = build_switching_candidate(
            contour, candidate_id=3, contour_fraction=1.0, radial_scale=scale,
        )
        np.testing.assert_allclose(
            at_zero.position_map,
            origin + scale * contour.rays[0].vector,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            at_half.position_map,
            origin + scale * contour.tangent_vector_at(0.5),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            at_one.position_map,
            origin + scale * contour.rays[-1].vector,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertLess(
            np.linalg.norm(near_one.position_map - at_one.position_map),
            np.linalg.norm(near_one.position_map - at_zero.position_map),
        )

    def test_s3_2_surface_residual_is_below_tolerance(self) -> None:
        candidates = generate_switching_candidates(
            self.contours["centered_cube"],
            contour_sample_count=48,
            radial_scales=np.linspace(0.2, 4.0, 20),
        )
        self.assertEqual(len(candidates), 48 * 20)
        self.assertLessEqual(
            max(candidate.surface_residual for candidate in candidates),
            SURFACE_RESIDUAL_TOLERANCE,
        )

    def test_s3_3_all_candidates_have_positive_radial_direction(self) -> None:
        contour = self.contours["centered_cube"]
        origin = contour.origin.as_array()
        candidates = generate_switching_candidates(contour)
        for candidate in candidates:
            vector = contour.tangent_vector_at(candidate.contour_fraction)
            self.assertGreater(float(np.dot(candidate.position_map - origin, vector)), 0.0)

    def test_s3_4_generation_is_independent_of_display_geometry(self) -> None:
        contour = self.contours["centered_cube"]
        los_model = LOSModel(build_terrain("centered_cube"))
        short_display = los_model.build_tangent_surface(
            contour, display_extension_factor=2.0,
        )
        long_display = los_model.build_tangent_surface(
            contour, display_extension_factor=8.0,
        )
        self.assertFalse(np.array_equal(
            short_display.display_mesh().vertices,
            long_display.display_mesh().vertices,
        ))
        first = generate_switching_candidates(short_display.tangent_contour)
        second = generate_switching_candidates(long_display.tangent_contour)
        np.testing.assert_allclose(
            np.vstack([candidate.position_map for candidate in first]),
            np.vstack([candidate.position_map for candidate in second]),
            rtol=0.0,
            atol=0.0,
        )
        source = (
            Path(__file__).resolve().parent / "switching_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("display_mesh", source)
        self.assertNotIn("display_extension_factor", source)

    def test_s3_5_all_required_terrain_cases_use_identical_api(self) -> None:
        for terrain_name, contour in self.contours.items():
            with self.subTest(terrain=terrain_name):
                candidates = generate_switching_candidates(contour)
                positions = np.vstack([candidate.position_map for candidate in candidates])
                self.assertEqual(
                    len(candidates),
                    INITIAL_CONTOUR_SAMPLE_COUNT * len(INITIAL_RADIAL_SCALES),
                )
                self.assertTrue(np.all(np.isfinite(positions)))
                self.assertLessEqual(
                    max(candidate.surface_residual for candidate in candidates),
                    SURFACE_RESIDUAL_TOLERANCE,
                )

    def test_s3_6_candidate_identity_and_radial_major_order_are_stable(self) -> None:
        contour = self.contours["centered_cube"]
        first = generate_switching_candidates(contour)
        second = generate_switching_candidates(contour)
        self.assertEqual(
            [candidate.candidate_id for candidate in first],
            list(range(len(first))),
        )
        self.assertEqual(
            [(candidate.contour_fraction, candidate.radial_scale) for candidate in first],
            [(candidate.contour_fraction, candidate.radial_scale) for candidate in second],
        )
        np.testing.assert_array_equal(
            np.vstack([candidate.position_map for candidate in first]),
            np.vstack([candidate.position_map for candidate in second]),
        )
        self.assertTrue(all(
            candidate.radial_scale == INITIAL_RADIAL_SCALES[0]
            for candidate in first[:INITIAL_CONTOUR_SAMPLE_COUNT]
        ))
        self.assertTrue(all(
            candidate.radial_scale == INITIAL_RADIAL_SCALES[1]
            for candidate in first[
                INITIAL_CONTOUR_SAMPLE_COUNT:2 * INITIAL_CONTOUR_SAMPLE_COUNT
            ]
        ))

    def test_candidate_contract_is_immutable_and_validated(self) -> None:
        position = np.array([1.0, 2.0, 3.0])
        candidate = SwitchingCandidate(0, position, 0.5, 1.0, 0.0)
        position[:] = -1.0
        np.testing.assert_array_equal(candidate.position_map, [1.0, 2.0, 3.0])
        self.assertFalse(candidate.position_map.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            candidate.radial_scale = 2.0
        with self.assertRaises(ValueError):
            SwitchingCandidate(-1, np.zeros(3), 0.5, 1.0, 0.0)
        with self.assertRaises(ValueError):
            SwitchingCandidate(0, np.zeros(2), 0.5, 1.0, 0.0)
        with self.assertRaises(ValueError):
            SwitchingCandidate(0, np.zeros(3), 1.1, 1.0, 0.0)
        with self.assertRaises(ValueError):
            SwitchingCandidate(0, np.zeros(3), 0.5, 0.0, 0.0)
        with self.assertRaises(ValueError):
            generate_switching_candidates(
                self.contours["centered_cube"], radial_scales=(1.0, 0.5),
            )

    def test_candidate_layer_has_no_mission_cost_dependency(self) -> None:
        source = (
            Path(__file__).resolve().parent / "switching_candidates.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        imported_modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        for forbidden in (
            "energy_model",
            "reachability",
            "bellman",
            "hazard",
            "defender",
            "game_types",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(any(
                    forbidden in module.lower() for module in imported_modules
                ))


if __name__ == "__main__":
    unittest.main()
