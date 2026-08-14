"""Regression tests for the clean Stage 1/2 geometry."""

from __future__ import annotations

import unittest

import numpy as np

from .configuration import build_configuration
from .geometry import build_geometry


class GeometryStageTests(unittest.TestCase):
    def test_geometry_only_mode_does_not_require_tangent_contacts(self) -> None:
        configuration = build_configuration()
        configuration["environment"]["sensor_xy_m"] = (2500.0, 0.0)
        geometry = build_geometry(configuration, require_tangent_manifold=False)
        self.assertTrue(geometry["validation"]["passed"])
        self.assertFalse(geometry["validation"]["tangent_manifold_required"])
        self.assertEqual(geometry["tangent_manifold"]["contact_points"].shape, (0, 3))

    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = build_geometry(build_configuration())

    def test_geometry_validation_passes(self) -> None:
        self.assertTrue(self.geometry["validation"]["passed"])

    def test_masks_partition_airspace(self) -> None:
        self.assertTrue(np.array_equal(
            self.geometry["los_mask"], ~self.geometry["occlusion_mask"],
        ))

    def test_tangent_contacts_satisfy_tangency(self) -> None:
        residual = self.geometry["tangent_manifold"]["tangent_residuals"]
        self.assertLessEqual(float(np.max(np.abs(residual))), 2.0e-6)
        self.assertGreaterEqual(
            self.geometry["validation"]["minimum_tangent_ray_clearance_m"],
            -1.0e-8,
        )

    def test_requested_positions_and_peak(self) -> None:
        self.assertTrue(np.allclose(
            self.geometry["sensor_position"][:2], (2500.0, 500.0),
        ))
        self.assertTrue(np.allclose(
            self.geometry["goal_position"][:2], (3000.0, 500.0),
        ))
        self.assertAlmostEqual(
            float(np.max(self.geometry["terrain_height"])), 200.0, places=10,
        )


if __name__ == "__main__":
    unittest.main()
