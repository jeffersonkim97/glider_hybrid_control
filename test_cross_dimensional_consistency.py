"""Cross-package consistency tests between p1b_4D (2D) and p1b_3DExtension (3D).

These tests do not require the two packages to share a scenario (domain
size, hill placement, sensor default, and cost weights all differ between
them independently). Instead they check a narrower, load-bearing claim:
at a degenerate 3D state with y = y_sensor = 0 and heading = 0 (motion and
range confined to the x-h plane, exactly p1b_4D's (z, h) plane), the two
packages' detection-rate formulas must agree, because p1b_3DExtension's
detection.py is documented as "the direct generalization" of p1b_4D's
formulas from (z, h) to (x, y, h). If a future edit breaks that
generalization in either package, this test is the tripwire.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import p1b_3DExtension.configuration as configuration_3d
import p1b_3DExtension.detection as detection_3d
import p1b_3DExtension.geometry as geometry_3d
import p1b_4D.configuration as configuration_2d
import p1b_4D.detection as detection_2d
import p1b_4D.geometry as geometry_2d
from p1b_3DExtension.phase_logging import close_phase_logger as close_phase_logger_3d
from p1b_4D.phase_logging import close_phase_logger as close_phase_logger_2d


def _named_outputs(function, *arguments: float) -> dict[str, float]:
    values = function(*arguments)
    values = values if isinstance(values, tuple) else (values,)
    return {
        function.name_out(index): float(value)
        for index, value in enumerate(values)
    }


class DetectionFormulaConsistencyTests(unittest.TestCase):
    """At y = 0, heading = 0, 3D detection rates must equal 2D's."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory_2d = TemporaryDirectory()
        cls._temporary_directory_3d = TemporaryDirectory()
        configuration_bundle_2d = configuration_2d.build_configuration_bundle(
            Path(cls._temporary_directory_2d.name)
        )
        configuration_bundle_3d = configuration_3d.build_configuration_bundle(
            Path(cls._temporary_directory_3d.name)
        )
        cls._logger_2d = configuration_bundle_2d["primary_result"][
            "logging_utilities"
        ]["logger"]
        cls._logger_3d = configuration_bundle_3d["primary_result"][
            "logging_utilities"
        ]["logger"]
        geometry_bundle_2d = geometry_2d.build_geometry_bundle(configuration_bundle_2d)
        geometry_bundle_3d = geometry_3d.build_geometry_bundle(configuration_bundle_3d)
        detection_bundle_2d = detection_2d.build_symbolic_detection_bundle(
            configuration_bundle_2d, geometry_bundle_2d,
        )
        detection_bundle_3d = detection_3d.build_symbolic_detection_bundle(
            configuration_bundle_3d, geometry_bundle_3d,
        )
        cls.functions_2d = detection_bundle_2d["primary_result"]["functions"]
        cls.functions_3d = detection_bundle_3d["primary_result"]["functions"]
        # Sanity precondition for the whole file: the two packages must
        # still share the same sensor coefficients (radar/doppler/acoustic,
        # rcs bounds) for a formula-level match to mean anything -- if this
        # ever fails, the packages have drifted apart on physical constants,
        # not just on scenario/domain choices, which is a different bug.
        detection_2d_config = configuration_bundle_2d["primary_result"][
            "sensor_config"
        ]["detection"]
        detection_3d_config = configuration_bundle_3d["primary_result"][
            "sensor_config"
        ]["detection"]
        for key in (
            "radar_coefficient", "doppler_coefficient", "acoustic_coefficient",
            "acoustic_speed_exponent", "rcs_min", "rcs_max", "range_floor",
        ):
            cls_msg = f"sensor_config.detection[{key!r}] has drifted between packages"
            assert detection_2d_config[key] == detection_3d_config[key], cls_msg

    @classmethod
    def tearDownClass(cls) -> None:
        close_phase_logger_2d(cls._logger_2d)
        close_phase_logger_3d(cls._logger_3d)
        cls._temporary_directory_2d.cleanup()
        cls._temporary_directory_3d.cleanup()

    def setUp(self) -> None:
        # A synthetic degenerate state, independent of either package's own
        # domain/hill scenario: only the shared sensor coefficients and the
        # formulas themselves are under test here, not a real solve. Chosen
        # comfortably outside the range_floor clamp (~430 m >> 10 m) so the
        # 2D atan2-based aspect angle and the 3D dot-product-based cos_aspect
        # are on equal footing (see module docstring and design notes below),
        # and also chosen (empirically) to read LOS-visible under both
        # packages' own default terrain, so the gated-rate test below
        # exercises real assertions instead of always skipping.
        self.along = 3000.0
        self.h = 150.0
        self.v = 18.0
        self.gamma = np.deg2rad(-20.0)
        self.heading = 0.0
        self.sensor_along = 3400.0
        self.y = 0.0
        self.y_sensor = 0.0
        self.h_sensor = 60.0

    def test_range_components_match(self) -> None:
        outputs_2d = _named_outputs(
            self.functions_2d["range"],
            self.along, self.h, self.sensor_along, self.h_sensor,
        )
        outputs_3d = _named_outputs(
            self.functions_3d["range"],
            self.along, self.y, self.h,
            self.sensor_along, self.y_sensor, self.h_sensor,
        )
        self.assertAlmostEqual(
            outputs_2d["horizontal_range"], outputs_3d["horizontal_range_x"], places=9,
        )
        self.assertEqual(outputs_3d["horizontal_range_y"], 0.0)
        self.assertAlmostEqual(
            outputs_2d["vertical_range"], outputs_3d["vertical_range"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["slant_range"], outputs_3d["slant_range"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["sensor_range"], outputs_3d["sensor_range"], places=9,
        )

    def test_powered_detection_rate_matches(self) -> None:
        # Powered detection has no LOS gate in either package (acoustic
        # emission is omnidirectional and unoccluded by design), so this
        # comparison needs no y=0/heading=0 argument beyond matching range.
        outputs_2d = _named_outputs(
            self.functions_2d["powered_detection_components"],
            self.along, self.h, self.v, self.sensor_along, self.h_sensor,
        )
        outputs_3d = _named_outputs(
            self.functions_3d["powered_detection_components"],
            self.along, self.y, self.h, self.v,
            self.sensor_along, self.y_sensor, self.h_sensor,
        )
        self.assertAlmostEqual(
            outputs_2d["acoustic_rate"], outputs_3d["acoustic_rate"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["powered_detection_rate"],
            outputs_3d["powered_detection_rate"],
            places=9,
        )

    def test_glide_detection_pre_los_rates_match(self) -> None:
        # radar_rate/radial_velocity_rate/glide_detection_rate are each
        # gated by los_visible, and the two packages compute visibility
        # from genuinely different terrain/geometry (different hills, and a
        # 1D swept-boundary interpolant vs. a 3D trilinear viewshed) -- that
        # design difference is intentional (see p1b_3DExtension.geometry's
        # docstring) and is exactly what this test must NOT be sensitive to.
        # So it compares only the pre-gate rate/aspect formulas, which are
        # pure functions of (range, v, gamma) and never touch LOS.
        outputs_2d = _named_outputs(
            self.functions_2d["glide_detection_components"],
            self.along, self.h, self.v, self.gamma, self.sensor_along, self.h_sensor,
        )
        outputs_3d = _named_outputs(
            self.functions_3d["glide_detection_components"],
            self.along, self.y, self.h, self.v, self.gamma, self.heading,
            self.sensor_along, self.y_sensor, self.h_sensor,
        )
        # 2D exposes cos(aspect_angle) only implicitly (via aspect_angle);
        # 3D exposes cos_aspect directly. Recover it from 2D's aspect_angle
        # for a like-for-like comparison.
        cos_aspect_2d = float(np.cos(outputs_2d["aspect_angle"]))
        self.assertAlmostEqual(cos_aspect_2d, outputs_3d["cos_aspect"], places=9)
        self.assertAlmostEqual(outputs_2d["rcs"], outputs_3d["rcs"], places=9)
        self.assertAlmostEqual(
            outputs_2d["radar_rate_raw"], outputs_3d["radar_rate_raw"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["radial_velocity"], outputs_3d["radial_velocity"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["radial_velocity_rate_raw"],
            outputs_3d["radial_velocity_rate_raw"],
            places=9,
        )

    def test_glide_detection_rate_matches_when_both_sides_are_visible(self) -> None:
        # With the pre-gate formulas already shown identical above, the
        # gated rates match too whenever both packages happen to classify
        # the synthetic state as visible -- confirming the los_visible
        # multiplication itself (0/1 gate, scale factors) is applied the
        # same way, without asserting the two LOS *geometries* agree.
        los_2d = _named_outputs(
            self.functions_2d["los"], self.along, self.h, self.sensor_along,
        )
        los_3d = _named_outputs(
            self.functions_3d["los"],
            self.along, self.y, self.h, self.sensor_along, self.y_sensor, self.h_sensor,
        )
        if los_2d["visible"] < 0.5 or los_3d["visible"] < 0.5:
            self.skipTest(
                "synthetic degenerate state is occluded under one package's "
                "own terrain -- pre-gate formula agreement is already "
                "covered by test_glide_detection_pre_los_rates_match"
            )
        outputs_2d = _named_outputs(
            self.functions_2d["glide_detection_components"],
            self.along, self.h, self.v, self.gamma, self.sensor_along, self.h_sensor,
        )
        outputs_3d = _named_outputs(
            self.functions_3d["glide_detection_components"],
            self.along, self.y, self.h, self.v, self.gamma, self.heading,
            self.sensor_along, self.y_sensor, self.h_sensor,
        )
        self.assertAlmostEqual(
            outputs_2d["radar_rate"], outputs_3d["radar_rate"], places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["radial_velocity_rate"],
            outputs_3d["radial_velocity_rate"],
            places=9,
        )
        self.assertAlmostEqual(
            outputs_2d["glide_detection_rate"],
            outputs_3d["glide_detection_rate"],
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
