"""Phase 3 CasADi symbolic detection-model tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import casadi as ca
import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.detection_io import export_detection_bundle, import_detection_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger


class DetectionTests(unittest.TestCase):
    """Verify graph construction, formulas, dimensions, and persistence."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.configuration_bundle = build_configuration_bundle(
            Path(self.temporary_directory.name)
        )
        logger = self.configuration_bundle["primary_result"]["logging_utilities"][
            "logger"
        ]
        self.addCleanup(close_phase_logger, logger)
        self.geometry_bundle = build_geometry_bundle(self.configuration_bundle)
        self.detection_bundle = build_symbolic_detection_bundle(
            self.configuration_bundle,
            self.geometry_bundle,
        )

    def test_bundle_and_graph_validation_pass(self) -> None:
        self.assertTrue(self.detection_bundle["status"]["success"])
        self.assertTrue(self.detection_bundle["validation"]["passed"])
        self.assertEqual(
            set(self.detection_bundle),
            {"primary_result", "validation", "metadata", "status"},
        )
        functions = self.detection_bundle["primary_result"]["functions"]
        self.assertTrue(all(isinstance(value, ca.Function) for value in functions.values()))

    def test_standard_symbol_contract(self) -> None:
        symbols = self.detection_bundle["primary_result"]["symbolic_variables"]
        self.assertEqual(
            tuple(symbols),
            ("z", "h", "v", "gamma", "z_sensor", "h_sensor"),
        )
        self.assertTrue(all(isinstance(value, ca.SX) for value in symbols.values()))

    def test_range_model_reuses_floor(self) -> None:
        function = self.detection_bundle["primary_result"]["functions"]["range"]
        sensor_position = self.geometry_bundle["primary_result"]["sensor_position"]
        outputs = _outputs(
            function,
            sensor_position[0],
            sensor_position[1],
            sensor_position[0],
            sensor_position[1],
        )
        range_floor = self.configuration_bundle["primary_result"]["sensor_config"][
            "detection"
        ]["range_floor"]
        self.assertEqual(outputs[0], 0.0)
        self.assertEqual(outputs[1], 0.0)
        self.assertEqual(outputs[2], 0.0)
        self.assertEqual(outputs[3], range_floor)

    def test_powered_acoustic_matches_existing_formula(self) -> None:
        function = self.detection_bundle["primary_result"]["functions"][
            "powered_detection_components"
        ]
        sensor_position = self.geometry_bundle["primary_result"]["sensor_position"]
        vehicle = self.configuration_bundle["primary_result"]["vehicle_config"]
        detection = self.configuration_bundle["primary_result"]["sensor_config"][
            "detection"
        ]
        z = sensor_position[0] - 100.0
        h = sensor_position[1]
        speed = vehicle["powered_speed"]
        acoustic_rate, powered_rate = _outputs(
            function, z, h, speed, sensor_position[0], sensor_position[1]
        )
        expected = (
            detection["acoustic_coefficient"]
            * speed ** detection["acoustic_speed_exponent"]
            / 100.0**2
        )
        self.assertAlmostEqual(acoustic_rate, expected)
        self.assertAlmostEqual(
            powered_rate,
            detection["acoustic_rate_scale"] * expected,
        )

    def test_glide_components_match_existing_visible_formula(self) -> None:
        function = self.detection_bundle["primary_result"]["functions"][
            "glide_detection_components"
        ]
        geometry = self.geometry_bundle["primary_result"]
        sensor_position = geometry["sensor_position"]
        detection = self.configuration_bundle["primary_result"]["sensor_config"][
            "detection"
        ]
        z = 1750.0
        h = 100.0
        speed = 15.0
        gamma = -0.2
        outputs = _outputs(
            function,
            z,
            h,
            speed,
            gamma,
            sensor_position[0],
            sensor_position[1],
        )
        delta_z = sensor_position[0] - z
        delta_h = sensor_position[1] - h
        distance = max(np.hypot(delta_z, delta_h), detection["range_floor"])
        los_angle = np.arctan2(delta_h, delta_z)
        aspect = np.arctan2(
            np.sin(gamma - los_angle),
            np.cos(gamma - los_angle),
        )
        rcs = detection["rcs_min"] + (
            detection["rcs_max"] - detection["rcs_min"]
        ) * np.cos(aspect) ** 2
        radar = detection["radar_coefficient"] * rcs / distance**4
        radial = speed * (
            np.cos(gamma) * delta_z + np.sin(gamma) * delta_h
        ) / distance
        doppler = detection["doppler_coefficient"] * radial**2 / distance**4
        # The real formula gates radar/doppler on LOS visibility; this fixed
        # test point's visibility depends on where the configured hill sits,
        # so read the actual visibility rather than assuming it's always 1
        # (assuming so made this test fragile to terrain/domain changes).
        los_visible = _outputs(
            self.detection_bundle["primary_result"]["functions"]["los"],
            z, h, sensor_position[0],
        )[2]
        self.assertAlmostEqual(outputs[1], aspect)
        self.assertAlmostEqual(outputs[2], rcs)
        self.assertAlmostEqual(outputs[3], radar)
        self.assertAlmostEqual(outputs[4], radial)
        self.assertAlmostEqual(outputs[5], doppler)
        self.assertAlmostEqual(outputs[8], los_visible * (radar + doppler))

    def test_mission_fusion_time_and_attacker_objective(self) -> None:
        functions = self.detection_bundle["primary_result"]["functions"]
        powered_hazard = 0.2
        glide_hazard = 0.3
        powered_time = 10.0
        glide_time = 20.0
        mission = _outputs(
            functions["mission_detection"], powered_hazard, glide_hazard
        )
        objective = _outputs(
            functions["attacker_objective"],
            powered_hazard,
            glide_hazard,
            powered_time,
            glide_time,
        )
        costs = self.configuration_bundle["primary_result"]["cost_config"][
            "attacker"
        ]
        expected_pod = 1.0 - np.exp(-(powered_hazard + glide_hazard))
        expected_time = powered_time + glide_time
        expected_time_normalized = (
            expected_time / costs["normalization"]["time"]["reference_seconds"]
        )
        expected_detection_normalized = (
            powered_hazard + glide_hazard
        ) / costs["normalization"]["pod"]["hazard_reference"]
        expected_objective = (
            costs["w_pod"] * expected_detection_normalized
            + costs["w_time"] * expected_time_normalized
        )
        self.assertAlmostEqual(mission[0], powered_hazard + glide_hazard)
        self.assertAlmostEqual(mission[-1], expected_pod)
        self.assertAlmostEqual(objective[1], expected_time)
        self.assertAlmostEqual(objective[2], expected_detection_normalized)
        self.assertAlmostEqual(objective[3], expected_time_normalized)
        self.assertAlmostEqual(objective[4], expected_objective)

    def test_detection_bundle_json_npz_round_trip(self) -> None:
        exported = export_detection_bundle(
            self.detection_bundle,
            self.configuration_bundle,
            self.geometry_bundle,
        )
        self.assertTrue(exported["status"]["success"])
        json_path = exported["primary_result"]["json_path"]
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["bundle_type"], "DetectionBundle")
        self.assertEqual(
            len(manifest["symbolic_metadata"]["function_metadata"]),
            9,
        )
        imported = import_detection_bundle(json_path)
        self.assertTrue(imported["status"]["success"])
        self.assertIn(
            "validation_sample_outputs",
            imported["primary_result"]["arrays"],
        )


def _outputs(function: ca.Function, *arguments: float) -> list[float]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in outputs]


if __name__ == "__main__":
    unittest.main()
