"""Phase 5 visualization-only 2D projected-cost tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.projection import construct_projected_cost_map
from p1b_4D.projection_io import (
    export_projected_cost_bundle,
    import_projected_cost_bundle,
)
from p1b_4D.stage_cost import construct_stage_cost_4d


class ProjectionTests(unittest.TestCase):
    """Verify minimum projection, local diagnostics, markers, and persistence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = TemporaryDirectory()
        cls.configuration_bundle = build_configuration_bundle(
            Path(cls.temporary_directory.name)
        )
        primary = cls.configuration_bundle["primary_result"]
        environment = primary["environment_config"]
        vehicle = primary["vehicle_config"]
        environment["grid"].update(
            {
                "z_count": 41,
                "z_spacing": 68.75,
                "h_count": 21,
                "h_spacing": 10.0,
                "v_count": 3,
                "gamma_count": 7,
            }
        )
        vehicle.update({"glide_speed_count": 3, "gamma_count": 7})
        cls.geometry_bundle = build_geometry_bundle(cls.configuration_bundle)
        cls.detection_bundle = build_symbolic_detection_bundle(
            cls.configuration_bundle, cls.geometry_bundle
        )
        cls.stage_cost_bundle = construct_stage_cost_4d(
            cls.configuration_bundle,
            cls.geometry_bundle,
            cls.detection_bundle,
        )
        cls.projection_bundle = construct_projected_cost_map(
            cls.configuration_bundle,
            cls.geometry_bundle,
            cls.detection_bundle,
            cls.stage_cost_bundle,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        logger = cls.configuration_bundle["primary_result"]["logging_utilities"][
            "logger"
        ]
        close_phase_logger(logger)
        cls.temporary_directory.cleanup()

    def test_projection_bundle_passes(self) -> None:
        self.assertTrue(self.projection_bundle["status"]["success"])
        self.assertTrue(self.projection_bundle["validation"]["passed"])
        self.assertEqual(
            self.projection_bundle["primary_result"]["projected_cost"].shape,
            (41, 21),
        )

    def test_projected_cost_is_direct_minimum(self) -> None:
        projection = self.projection_bundle["primary_result"]
        j4d = self.stage_cost_bundle["primary_result"]["j4d"]
        np.testing.assert_array_equal(
            projection["projected_cost"],
            np.min(j4d, axis=(2, 3)),
        )

    def test_local_control_indices_reconstruct_minimum(self) -> None:
        projection = self.projection_bundle["primary_result"]
        stage = self.stage_cost_bundle["primary_result"]
        mask = projection["projection_mask"]
        spatial = np.nonzero(mask)
        selected = stage["j4d"][
            spatial[0],
            spatial[1],
            projection["optimal_velocity_index"][mask],
            projection["optimal_gamma_index"][mask],
        ]
        np.testing.assert_array_equal(selected, projection["projected_cost"][mask])
        np.testing.assert_array_equal(
            projection["optimal_velocity"][mask],
            stage["grids"]["v"][projection["optimal_velocity_index"][mask]],
        )
        np.testing.assert_array_equal(
            projection["optimal_gamma"][mask],
            stage["grids"]["gamma"][projection["optimal_gamma_index"][mask]],
        )

    def test_invalid_cells_have_explicit_diagnostics(self) -> None:
        projection = self.projection_bundle["primary_result"]
        invalid = ~projection["projection_mask"]
        self.assertTrue(np.any(invalid))
        self.assertTrue(np.all(np.isposinf(projection["projected_cost"][invalid])))
        self.assertTrue(np.all(np.isnan(projection["optimal_velocity"][invalid])))
        self.assertTrue(np.all(np.isnan(projection["optimal_gamma"][invalid])))
        self.assertTrue(
            np.all(projection["optimal_velocity_index"][invalid] == -1)
        )
        self.assertTrue(np.all(projection["optimal_gamma_index"][invalid] == -1))

    def test_projection_is_marked_visualization_only(self) -> None:
        metadata = self.projection_bundle["metadata"]
        projection_metadata = self.projection_bundle["primary_result"][
            "projection_metadata"
        ]
        self.assertTrue(metadata["visualization_only"])
        self.assertFalse(projection_metadata["bellman_policy_input"])
        self.assertFalse(projection_metadata["is_value_function"])
        self.assertFalse(projection_metadata["is_cost_to_go"])
        self.assertIn("bellman_policy", metadata["prohibited_uses"])

    def test_json_npz_round_trip(self) -> None:
        exported = export_projected_cost_bundle(
            self.projection_bundle,
            self.configuration_bundle,
        )
        self.assertTrue(exported["status"]["success"])
        json_path = exported["primary_result"]["json_path"]
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["visualization_only"])
        self.assertFalse(manifest["bellman_policy_input"])
        imported = import_projected_cost_bundle(json_path)
        self.assertTrue(imported["status"]["success"])
        np.testing.assert_array_equal(
            imported["primary_result"]["arrays"]["projected_cost"],
            self.projection_bundle["primary_result"]["projected_cost"],
        )
        np.testing.assert_array_equal(
            imported["primary_result"]["arrays"]["projection_mask"],
            self.projection_bundle["primary_result"]["projection_mask"],
        )


if __name__ == "__main__":
    unittest.main()
