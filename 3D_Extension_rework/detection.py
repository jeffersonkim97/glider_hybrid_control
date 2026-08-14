"""Symbolic 3D detection model extended from ``p1b_4D.detection``.

The completed 2D coefficients, hazard fusion, and objective normalization are
unchanged.  Spatial range, aspect, and radial velocity are written as 3D
vector operations, and the LOS gate consumes the Stage 1 boundary surface.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


def build_symbolic_detection_bundle(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Build reusable CasADi functions for the 6D state ``(x,y,h,v,gamma,psi)``."""
    if not geometry.get("validation", {}).get("passed", False):
        raise ValueError("geometry must pass validation before detection is built")

    detection = configuration["detection"]
    vehicle = configuration["vehicle"]
    cost = configuration["cost"]

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    h = ca.SX.sym("h")
    v = ca.SX.sym("v")
    gamma = ca.SX.sym("gamma")
    psi = ca.SX.sym("psi")
    x_sensor = ca.SX.sym("x_sensor")
    y_sensor = ca.SX.sym("y_sensor")
    h_sensor = ca.SX.sym("h_sensor")
    powered_hazard = ca.SX.sym("powered_hazard")
    glide_hazard = ca.SX.sym("glide_hazard")
    powered_time = ca.SX.sym("powered_time")
    glide_time = ca.SX.sym("glide_time")
    coverage_input = ca.SX.sym("coverage_volume_normalized")

    delta_x = x_sensor - x
    delta_y = y_sensor - y
    delta_h = h_sensor - h
    horizontal_range = ca.sqrt(delta_x**2 + delta_y**2)
    slant_range = ca.sqrt(horizontal_range**2 + delta_h**2)
    sensor_range = ca.fmax(slant_range, detection["range_floor_m"])

    boundary_values = np.asarray(
        geometry["los_boundary_height"], dtype=float,
    )
    los_boundary_interpolant = ca.interpolant(
        "los_boundary_height_3d",
        "linear",
        [
            np.ascontiguousarray(geometry["x_grid"], dtype=float),
            np.ascontiguousarray(geometry["y_grid"], dtype=float),
        ],
        np.ascontiguousarray(boundary_values.ravel(order="F")),
    )
    los_boundary_height = los_boundary_interpolant(ca.vertcat(x, y))
    los_margin = h - los_boundary_height
    los_visible = ca.if_else(h < los_boundary_height, 0.0, 1.0)
    occlusion_indicator = 1.0 - los_visible

    velocity_x = v * ca.cos(gamma) * ca.cos(psi)
    velocity_y = v * ca.cos(gamma) * ca.sin(psi)
    velocity_h = v * ca.sin(gamma)
    inverse_sensor_range = 1.0 / sensor_range
    los_unit_x = delta_x * inverse_sensor_range
    los_unit_y = delta_y * inverse_sensor_range
    los_unit_h = delta_h * inverse_sensor_range
    radial_velocity = (
        velocity_x * los_unit_x
        + velocity_y * los_unit_y
        + velocity_h * los_unit_h
    )
    cosine_aspect = ca.fmin(ca.fmax(radial_velocity / ca.fmax(v, 1.0e-9), -1.0), 1.0)
    aspect_angle = ca.acos(cosine_aspect)
    los_azimuth = ca.atan2(delta_y, delta_x)
    los_elevation = ca.atan2(delta_h, horizontal_range)

    acoustic_rate = (
        detection["acoustic_coefficient"]
        * v ** detection["acoustic_speed_exponent"]
        / sensor_range**2
    )
    powered_detection_rate = detection["acoustic_rate_scale"] * acoustic_rate
    rcs = detection["rcs_min"] + (
        detection["rcs_max"] - detection["rcs_min"]
    ) * cosine_aspect**2
    radar_rate_raw = detection["radar_coefficient"] * rcs / sensor_range**4
    radial_velocity_rate_raw = (
        detection["doppler_coefficient"] * radial_velocity**2 / sensor_range**4
    )
    radar_rate = los_visible * detection["radar_rate_scale"] * radar_rate_raw
    radial_velocity_rate = (
        los_visible
        * detection["radial_velocity_rate_scale"]
        * radial_velocity_rate_raw
    )
    glide_detection_rate = radar_rate + radial_velocity_rate

    mission_hazard = powered_hazard + glide_hazard
    powered_pod = 1.0 - ca.exp(-powered_hazard)
    glide_pod = 1.0 - ca.exp(-glide_hazard)
    mission_pod = 1.0 - ca.exp(-mission_hazard)
    mission_time = powered_time + glide_time
    attacker = cost["attacker"]
    pod_normalized = mission_hazard / attacker["hazard_reference"]
    time_normalized = mission_time / attacker["time_reference_s"]
    attacker_objective = (
        attacker["w_pod"] * pod_normalized
        + attacker["w_time"] * time_normalized
    )
    defender = cost["defender"]
    defender_pod_normalized = mission_hazard / (
        mission_hazard + defender["hazard_reference"]
    )
    coverage_normalized = ca.fmin(ca.fmax(coverage_input, 0.0), 1.0)
    defender_objective = (
        defender["w_pod"] * defender_pod_normalized
        + defender["w_coverage"] * coverage_normalized
    )

    expressions = {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_h": delta_h,
        "horizontal_range": horizontal_range,
        "slant_range": slant_range,
        "sensor_range": sensor_range,
        "los_boundary_height": los_boundary_height,
        "los_margin": los_margin,
        "los_visible": los_visible,
        "occlusion_indicator": occlusion_indicator,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "velocity_h": velocity_h,
        "los_azimuth": los_azimuth,
        "los_elevation": los_elevation,
        "cosine_aspect": cosine_aspect,
        "aspect_angle": aspect_angle,
        "rcs": rcs,
        "radial_velocity": radial_velocity,
        "acoustic_rate": acoustic_rate,
        "powered_detection_rate": powered_detection_rate,
        "radar_rate_raw": radar_rate_raw,
        "radial_velocity_rate_raw": radial_velocity_rate_raw,
        "radar_rate": radar_rate,
        "radial_velocity_rate": radial_velocity_rate,
        "glide_detection_rate": glide_detection_rate,
        "mission_hazard": mission_hazard,
        "powered_pod": powered_pod,
        "glide_pod": glide_pod,
        "mission_pod": mission_pod,
        "mission_time": mission_time,
        "pod_normalized": pod_normalized,
        "time_normalized": time_normalized,
        "attacker_objective": attacker_objective,
        "defender_pod_normalized": defender_pod_normalized,
        "coverage_normalized": coverage_normalized,
        "defender_objective": defender_objective,
    }
    symbols = {
        "x": x,
        "y": y,
        "h": h,
        "v": v,
        "gamma": gamma,
        "psi": psi,
        "x_sensor": x_sensor,
        "y_sensor": y_sensor,
        "h_sensor": h_sensor,
    }
    auxiliary = {
        "powered_hazard": powered_hazard,
        "glide_hazard": glide_hazard,
        "powered_time": powered_time,
        "glide_time": glide_time,
        "coverage_volume_normalized": coverage_input,
    }
    functions = _build_functions(symbols, auxiliary, expressions)
    validation = validate_symbolic_detection(
        functions, configuration, geometry,
    )
    return {
        "symbols": symbols,
        "auxiliary_symbols": auxiliary,
        "expressions": expressions,
        "functions": functions,
        "function_metadata": {
            name: {
                "input_names": tuple(function.name_in(i) for i in range(function.n_in())),
                "output_names": tuple(function.name_out(i) for i in range(function.n_out())),
            }
            for name, function in functions.items()
        },
        "detection_components": {
            "powered": ("acoustic",),
            "glide": ("radar", "radial_velocity", "rcs"),
            "mission_fusion": "additive_hazard",
        },
        "metadata": {
            "state_axis_order": ("x", "y", "h", "v", "gamma", "psi"),
            "spatial_projection_axes": ("x", "y", "h"),
            "projection_role": "visualization_only_not_bellman_input",
            "source_model": "p1b_4D.detection",
            "coefficient_policy": "unchanged_from_p1b_4D",
            "casadi_version": ca.__version__,
            "attacker_objective_id": attacker["objective_id"],
            "defender_objective_id": defender["objective_id"],
            "powered_speed_mps": vehicle["powered_speed_mps"],
        },
        "validation": validation,
        "status": {
            "success": validation["passed"],
            "message": validation["summary"],
        },
    }


def _build_functions(
    symbols: dict[str, ca.SX],
    auxiliary: dict[str, ca.SX],
    expression: dict[str, ca.SX],
) -> dict[str, ca.Function]:
    x, y, h, v, gamma, psi, xs, ys, hs = (
        symbols[name]
        for name in (
            "x", "y", "h", "v", "gamma", "psi",
            "x_sensor", "y_sensor", "h_sensor",
        )
    )
    powered_hazard, glide_hazard, powered_time, glide_time, coverage = (
        auxiliary[name]
        for name in (
            "powered_hazard", "glide_hazard", "powered_time", "glide_time",
            "coverage_volume_normalized",
        )
    )
    return {
        "range": ca.Function(
            "SensorRange3DFunction", [x, y, h, xs, ys, hs],
            [expression[name] for name in (
                "delta_x", "delta_y", "delta_h", "horizontal_range",
                "slant_range", "sensor_range",
            )],
            ["x", "y", "h", "x_sensor", "y_sensor", "h_sensor"],
            ["delta_x", "delta_y", "delta_h", "horizontal_range", "slant_range", "sensor_range"],
        ),
        "los": ca.Function(
            "SymbolicLos3DFunction", [x, y, h],
            [expression[name] for name in (
                "los_boundary_height", "los_margin", "los_visible",
                "occlusion_indicator",
            )],
            ["x", "y", "h"],
            ["boundary_height", "los_margin", "visible", "occluded"],
        ),
        "powered_detection_components": ca.Function(
            "PoweredDetectionComponents3DFunction", [x, y, h, v, xs, ys, hs],
            [expression["acoustic_rate"], expression["powered_detection_rate"]],
            ["x", "y", "h", "v", "x_sensor", "y_sensor", "h_sensor"],
            ["acoustic_rate", "powered_detection_rate"],
        ),
        "glide_detection_components": ca.Function(
            "GlideDetectionComponents3DFunction",
            [x, y, h, v, gamma, psi, xs, ys, hs],
            [expression[name] for name in (
                "los_azimuth", "los_elevation", "aspect_angle",
                "cosine_aspect", "rcs", "radar_rate_raw",
                "radial_velocity", "radial_velocity_rate_raw", "radar_rate",
                "radial_velocity_rate", "glide_detection_rate",
            )],
            [
                "x", "y", "h", "v", "gamma", "psi",
                "x_sensor", "y_sensor", "h_sensor",
            ],
            [
                "los_azimuth", "los_elevation", "aspect_angle",
                "cosine_aspect", "rcs", "radar_rate_raw",
                "radial_velocity", "radial_velocity_rate_raw", "radar_rate",
                "radial_velocity_rate", "glide_detection_rate",
            ],
        ),
        "mission_detection": ca.Function(
            "MissionDetection3DFunction", [powered_hazard, glide_hazard],
            [expression[name] for name in (
                "mission_hazard", "powered_pod", "glide_pod", "mission_pod",
            )],
            ["powered_hazard", "glide_hazard"],
            ["mission_hazard", "powered_pod", "glide_pod", "mission_pod"],
        ),
        "mission_time": ca.Function(
            "MissionTime3DFunction", [powered_time, glide_time],
            [expression["mission_time"]], ["powered_time", "glide_time"],
            ["mission_time"],
        ),
        "attacker_objective": ca.Function(
            "AttackerObjective3DFunction",
            [powered_hazard, glide_hazard, powered_time, glide_time],
            [expression[name] for name in (
                "mission_pod", "mission_time", "pod_normalized",
                "time_normalized", "attacker_objective",
            )],
            ["powered_hazard", "glide_hazard", "powered_time", "glide_time"],
            ["mission_pod", "mission_time", "pod_normalized", "time_normalized", "attacker_objective"],
        ),
        "defender_components": ca.Function(
            "DefenderObjectiveComponents3DFunction",
            [powered_hazard, glide_hazard, coverage],
            [expression["defender_pod_normalized"], expression["coverage_normalized"]],
            ["powered_hazard", "glide_hazard", "coverage_volume_fraction"],
            ["pod_normalized", "coverage_volume_normalized"],
        ),
        "defender_objective": ca.Function(
            "DefenderObjective3DFunction",
            [powered_hazard, glide_hazard, coverage],
            [
                expression["defender_pod_normalized"],
                expression["coverage_normalized"], expression["defender_objective"],
            ],
            ["powered_hazard", "glide_hazard", "coverage_volume_fraction"],
            ["pod_normalized", "coverage_volume_normalized", "defender_objective"],
        ),
    }


def validate_symbolic_detection(
    functions: dict[str, ca.Function],
    configuration: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Check graph dimensions, 3D formulas, LOS gating, and objective reuse."""
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    detection = configuration["detection"]
    vehicle = configuration["vehicle"]
    tolerance = configuration["validation"]
    sample = np.array([1800.0, 500.0, 250.0])
    speed = 0.5 * (
        vehicle["glide_speed_min_mps"] + vehicle["glide_speed_max_mps"]
    )
    gamma = np.deg2rad(-15.0)
    psi = 0.0
    range_values = _numeric_outputs(functions["range"], *sample, *sensor)
    expected_delta = sensor - sample
    expected_slant = float(np.linalg.norm(expected_delta))
    powered_values = _numeric_outputs(
        functions["powered_detection_components"],
        *sample, vehicle["powered_speed_mps"], *sensor,
    )
    glide_values = _numeric_outputs(
        functions["glide_detection_components"],
        *sample, speed, gamma, psi, *sensor,
    )
    velocity = speed * np.array([
        np.cos(gamma) * np.cos(psi),
        np.cos(gamma) * np.sin(psi),
        np.sin(gamma),
    ])
    expected_radial_velocity = float(
        np.dot(velocity, expected_delta / expected_slant)
    )

    visible_index = np.argwhere(geometry["los_mask"])[0]
    occluded_index = np.argwhere(geometry["non_visible_airspace_mask"])[0]
    visible_point = (
        geometry["x_grid"][visible_index[0]],
        geometry["y_grid"][visible_index[1]],
        geometry["h_grid"][visible_index[2]],
    )
    occluded_point = (
        geometry["x_grid"][occluded_index[0]],
        geometry["y_grid"][occluded_index[1]],
        geometry["h_grid"][occluded_index[2]],
    )
    visible_los = _numeric_outputs(functions["los"], *visible_point)
    occluded_los = _numeric_outputs(functions["los"], *occluded_point)

    grid_x, grid_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    mapped_los = functions["los"].map(grid_x.size)(
        grid_x.reshape(1, -1), grid_y.reshape(1, -1),
        np.zeros((1, grid_x.size)),
    )
    mapped_los_values = mapped_los if isinstance(mapped_los, tuple) else (mapped_los,)
    evaluated_boundary = np.asarray(
        mapped_los_values[0], dtype=float,
    ).reshape(grid_x.shape)
    maximum_boundary_lookup_error = float(np.max(np.abs(
        evaluated_boundary - geometry["los_boundary_height"]
    )))

    powered_hazard_value = powered_values[-1] * vehicle["time_step_s"]
    glide_hazard_value = glide_values[-1] * vehicle["time_step_s"]
    mission = _numeric_outputs(
        functions["mission_detection"], powered_hazard_value, glide_hazard_value,
    )
    objective = _numeric_outputs(
        functions["attacker_objective"], powered_hazard_value,
        glide_hazard_value, vehicle["time_step_s"], vehicle["time_step_s"],
    )
    attacker = configuration["cost"]["attacker"]
    expected_objective = (
        attacker["w_pod"]
        * (powered_hazard_value + glide_hazard_value)
        / attacker["hazard_reference"]
        + attacker["w_time"]
        * (2.0 * vehicle["time_step_s"])
        / attacker["time_reference_s"]
    )
    checks = {
        "nine_symbolic_functions": bool(
            len(functions) == 9
            and all(isinstance(function, ca.Function) for function in functions.values())
        ),
        "six_dimensional_state_contract": (
            functions["glide_detection_components"].n_in() == 9
        ),
        "range_vector_matches_3d_geometry": bool(
            np.allclose(range_values[:3], expected_delta, atol=1.0e-12)
            and abs(range_values[4] - expected_slant) <= 1.0e-12
        ),
        "range_floor_reused": _numeric_outputs(
            functions["range"], *sensor, *sensor,
        )[-1] == detection["range_floor_m"],
        "radial_velocity_is_3d_dot_product": bool(
            abs(glide_values[6] - expected_radial_velocity) <= 1.0e-12
        ),
        "los_gate_matches_visible_voxel": visible_los[2] == 1.0,
        "los_gate_matches_occluded_voxel": occluded_los[3] == 1.0,
        "los_surface_lookup_matches_geometry_grid": bool(
            maximum_boundary_lookup_error <= 1.0e-10
        ),
        "mission_probability_bounds": bool(
            -tolerance["detection_probability_tolerance"]
            <= mission[-1]
            <= 1.0 + tolerance["detection_probability_tolerance"]
        ),
        "mission_additive_hazard": bool(
            abs(mission[0] - powered_hazard_value - glide_hazard_value)
            <= tolerance["objective_tolerance"]
        ),
        "attacker_objective_unchanged": bool(
            abs(objective[-1] - expected_objective)
            <= tolerance["objective_tolerance"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "sample_slant_range_m": expected_slant,
            "sample_radial_velocity_mps": glide_values[6],
            "sample_powered_rate_per_s": powered_values[-1],
            "sample_glide_rate_per_s": glide_values[-1],
            "sample_mission_pod": mission[-1],
            "sample_attacker_objective": objective[-1],
            "maximum_los_boundary_lookup_error_m": maximum_boundary_lookup_error,
        },
        "summary": (
            "3D symbolic detection validation passed"
            if not failed
            else f"3D symbolic detection failed checks: {failed}"
        ),
    }


def _numeric_outputs(function: ca.Function, *arguments: float) -> list[float]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in outputs]
