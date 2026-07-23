"""Phase 4 authoritative 4D stage-cost tests."""

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
from p1b_4D.stage_cost import construct_stage_cost_4d
from p1b_4D.stage_cost_io import (
    export_stage_cost_4d_bundle,
    import_stage_cost_4d_bundle,
)


class StageCostTests(unittest.TestCase):
    """Validate grids, masks, components, objective, and persistence."""

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
            cls.configuration_bundle,
            cls.geometry_bundle,
        )
        cls.stage_cost_bundle = construct_stage_cost_4d(
            cls.configuration_bundle,
            cls.geometry_bundle,
            cls.detection_bundle,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        logger = cls.configuration_bundle["primary_result"]["logging_utilities"][
            "logger"
        ]
        close_phase_logger(logger)
        cls.temporary_directory.cleanup()

    def test_stage_cost_bundle_passes(self) -> None:
        self.assertTrue(self.stage_cost_bundle["status"]["success"])
        self.assertTrue(self.stage_cost_bundle["validation"]["passed"])
        self.assertEqual(
            set(self.stage_cost_bundle),
            {"primary_result", "validation", "metadata", "status"},
        )

    def test_standard_grid_shape_and_axis_order(self) -> None:
        result = self.stage_cost_bundle["primary_result"]
        grids = result["grids"]
        expected_shape = tuple(
            grids[name].size for name in ("z", "h", "v", "gamma")
        )
        self.assertEqual(result["j4d"].shape, expected_shape)
        self.assertEqual(
            result["grid_metadata"]["axis_order"],
            ("z", "h", "v", "gamma"),
        )

    def test_invalid_states_have_infinite_cost(self) -> None:
        result = self.stage_cost_bundle["primary_result"]
        feasible = result["feasible_mask"]
        self.assertTrue(np.all(np.isfinite(result["j4d"][feasible])))
        self.assertTrue(np.all(np.isposinf(result["j4d"][~feasible])))
        terrain = result["validity_masks"]["terrain_penetration_mask"]
        self.assertTrue(np.all(~feasible[terrain]))

    def test_component_and_objective_consistency(self) -> None:
        result = self.stage_cost_bundle["primary_result"]
        components = result["component_maps"]
        feasible = result["feasible_mask"]
        costs = self.configuration_bundle["primary_result"]["cost_config"][
            "attacker"
        ]
        reconstructed = (
            costs["w_pod"] * components["pod_normalized"]
            + costs["w_time"] * components["time_normalized"]
        )
        np.testing.assert_allclose(
            result["j4d"][feasible],
            reconstructed[feasible],
            rtol=0.0,
            atol=self.configuration_bundle["primary_result"][
                "validation_config"
            ]["objective_tolerance"],
        )
        np.testing.assert_allclose(
            components["stage_pod"][feasible],
            -np.expm1(
                -components["pod_normalized"][feasible]
                * costs["normalization"]["pod"]["hazard_reference"]
            ),
        )

    def test_powered_and_glide_components_are_separate(self) -> None:
        result = self.stage_cost_bundle["primary_result"]
        powered_feasible = result["validity_masks"]["powered_feasible_mask"]
        self.assertTrue(np.any(powered_feasible))
        self.assertTrue(
            np.all(
                np.isfinite(
                    result["powered_stage_cost_4d"][powered_feasible]
                )
            )
        )
        self.assertEqual(self.stage_cost_bundle["metadata"]["j4d_mode"], "glide")

    def test_all_required_component_maps_exist(self) -> None:
        components = self.stage_cost_bundle["primary_result"]["component_maps"]
        required = {
            "stage_pod",
            "stage_time",
            "acoustic_rate",
            "radar_rate",
            "radial_velocity",
            "radial_velocity_detection_rate",
            "aspect_angle",
            "rcs",
            "pod_normalized",
            "time_normalized",
            "stage_objective",
        }
        self.assertTrue(required.issubset(components))

    def test_json_npz_round_trip(self) -> None:
        exported = export_stage_cost_4d_bundle(
            self.stage_cost_bundle,
            self.configuration_bundle,
        )
        self.assertTrue(exported["status"]["success"])
        json_path = exported["primary_result"]["json_path"]
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundle_type"], "StageCost4DBundle")
        self.assertIn("j4d", manifest["payloads"]["npz"]["arrays"])
        imported = import_stage_cost_4d_bundle(
            json_path,
            ("j4d", "feasible_mask", "component__stage_objective"),
        )
        self.assertTrue(imported["status"]["success"])
        np.testing.assert_array_equal(
            imported["primary_result"]["arrays"]["feasible_mask"],
            self.stage_cost_bundle["primary_result"]["feasible_mask"],
        )
        np.testing.assert_array_equal(
            imported["primary_result"]["arrays"]["j4d"],
            self.stage_cost_bundle["primary_result"]["j4d"],
        )


if __name__ == "__main__":
    unittest.main()
