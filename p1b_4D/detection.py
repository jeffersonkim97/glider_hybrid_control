"""Authoritative CasADi symbolic detection and mission-objective model."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


def build_symbolic_detection_bundle(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Build all reusable CasADi detection and mission functions.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.
    geometry_bundle:
        Successful Phase 2 GeometryBundle supplying LOS parameters and
        normalized coverage.

    Outputs
    -------
    dict
        Universal result envelope containing standard symbols, expressions,
        CasADi Functions, component metadata, and validation.

    Assumptions
    -----------
    Detection coefficients and normalization methods come only from Phase 1.
    LOS tangent parameters come only from Phase 2 and are explicit function
    arguments wherever sensor-dependent visibility is evaluated.

    Notes
    -----
    This function builds symbolic graphs only. It does not optimize, construct
    cost maps, reconstruct terrain, write files, or plot.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    sensor = configs["sensor_config"]
    vehicle = configs["vehicle_config"]
    costs = configs["cost_config"]
    validation_config = configs["validation_config"]
    detection = sensor["detection"]

    z = ca.SX.sym("z")
    h = ca.SX.sym("h")
    v = ca.SX.sym("v")
    gamma = ca.SX.sym("gamma")
    z_sensor = ca.SX.sym("z_sensor")
    h_sensor = ca.SX.sym("h_sensor")
    tangent_z = ca.SX.sym("tangent_z")
    tangent_slope = ca.SX.sym("tangent_slope")
    tangent_intercept = ca.SX.sym("tangent_intercept")
    powered_hazard = ca.SX.sym("powered_hazard")
    glide_hazard = ca.SX.sym("glide_hazard")
    powered_time = ca.SX.sym("powered_time")
    glide_time = ca.SX.sym("glide_time")
    coverage_area_normalized_input = ca.SX.sym("coverage_area_normalized")

    horizontal_range = z_sensor - z
    vertical_range = h_sensor - h
    slant_range = ca.sqrt(horizontal_range**2 + vertical_range**2)
    sensor_range = ca.fmax(slant_range, detection["range_floor"])

    los_boundary_height = tangent_slope * z + tangent_intercept
    los_margin = h - los_boundary_height
    is_occluded = ca.logic_and(z < tangent_z, h < los_boundary_height)
    los_visible = ca.if_else(is_occluded, 0.0, 1.0)
    occlusion_indicator = 1.0 - los_visible

    acoustic_rate = (
        detection["acoustic_coefficient"]
        * v ** detection["acoustic_speed_exponent"]
        / sensor_range**2
    )
    los_angle = ca.atan2(vertical_range, horizontal_range)
    aspect_angle = ca.atan2(
        ca.sin(gamma - los_angle),
        ca.cos(gamma - los_angle),
    )
    rcs = detection["rcs_min"] + (
        detection["rcs_max"] - detection["rcs_min"]
    ) * ca.cos(aspect_angle) ** 2
    radar_rate_raw = detection["radar_coefficient"] * rcs / sensor_range**4
    radial_velocity = v * (
        ca.cos(gamma) * horizontal_range
        + ca.sin(gamma) * vertical_range
    ) / sensor_range
    radial_velocity_rate_raw = (
        detection["doppler_coefficient"]
        * radial_velocity**2
        / sensor_range**4
    )
    radar_rate = los_visible * detection["radar_rate_scale"] * radar_rate_raw
    radial_velocity_rate = (
        los_visible
        * detection["radial_velocity_rate_scale"]
        * radial_velocity_rate_raw
    )
    glide_detection_rate = radar_rate + radial_velocity_rate
    powered_detection_rate = detection["acoustic_rate_scale"] * acoustic_rate

    mission_hazard = powered_hazard + glide_hazard
    powered_pod = 1.0 - ca.exp(-powered_hazard)
    glide_pod = 1.0 - ca.exp(-glide_hazard)
    mission_pod = 1.0 - ca.exp(-mission_hazard)
    mission_time = powered_time + glide_time

    attacker_normalization = costs["attacker"]["normalization"]
    attacker_pod_specification = attacker_normalization["pod"]
    if attacker_pod_specification["method"] == "cumulative_hazard_reference":
        pod_normalized = (
            mission_hazard / attacker_pod_specification["hazard_reference"]
        )
    elif attacker_pod_specification["method"] == "probability":
        pod_normalized = mission_pod
    else:
        raise ValueError(
            "Unsupported Attacker detection normalization method: "
            f"{attacker_pod_specification['method']}"
        )
    time_normalized = (
        mission_time
        / attacker_normalization["time"]["reference_seconds"]
    )
    attacker_objective = (
        costs["attacker"]["w_pod"] * pod_normalized
        + costs["attacker"]["w_time"] * time_normalized
    )

    defender_pod_specification = costs["defender"]["normalization"]["pod"]
    if defender_pod_specification["method"] == "hazard_reference":
        hazard_reference = defender_pod_specification["hazard_reference"]
        defender_pod_normalized = mission_hazard / (
            mission_hazard + hazard_reference
        )
    elif defender_pod_specification["method"] == "probability":
        defender_pod_normalized = mission_pod
    else:
        raise ValueError(
            "Unsupported Defender PoD normalization method: "
            f"{defender_pod_specification['method']}"
        )
    coverage_area_normalized = ca.fmin(
        ca.fmax(coverage_area_normalized_input, 0.0),
        1.0,
    )
    defender_objective = (
        costs["defender"]["w_pod"] * defender_pod_normalized
        + costs["defender"]["w_cover"] * coverage_area_normalized
    )

    standard_symbols = {
        "z": z,
        "h": h,
        "v": v,
        "gamma": gamma,
        "z_sensor": z_sensor,
        "h_sensor": h_sensor,
    }
    auxiliary_symbols = {
        "tangent_z": tangent_z,
        "tangent_slope": tangent_slope,
        "tangent_intercept": tangent_intercept,
        "powered_hazard": powered_hazard,
        "glide_hazard": glide_hazard,
        "powered_time": powered_time,
        "glide_time": glide_time,
        "coverage_area_normalized": coverage_area_normalized_input,
    }
    functions = _build_functions(
        standard_symbols,
        auxiliary_symbols,
        {
            "horizontal_range": horizontal_range,
            "vertical_range": vertical_range,
            "slant_range": slant_range,
            "sensor_range": sensor_range,
            "los_boundary_height": los_boundary_height,
            "los_margin": los_margin,
            "los_visible": los_visible,
            "occlusion_indicator": occlusion_indicator,
            "acoustic_rate": acoustic_rate,
            "powered_detection_rate": powered_detection_rate,
            "los_angle": los_angle,
            "aspect_angle": aspect_angle,
            "rcs": rcs,
            "radar_rate_raw": radar_rate_raw,
            "radial_velocity": radial_velocity,
            "radial_velocity_rate_raw": radial_velocity_rate_raw,
            "radar_rate": radar_rate,
            "radial_velocity_rate": radial_velocity_rate,
            "glide_detection_rate": glide_detection_rate,
            "powered_pod": powered_pod,
            "glide_pod": glide_pod,
            "mission_hazard": mission_hazard,
            "mission_pod": mission_pod,
            "mission_time": mission_time,
            "pod_normalized": pod_normalized,
            "time_normalized": time_normalized,
            "attacker_objective": attacker_objective,
            "defender_pod_normalized": defender_pod_normalized,
            "coverage_area_normalized": coverage_area_normalized,
            "defender_objective": defender_objective,
        },
    )
    expressions = {
        "range": {
            "horizontal": horizontal_range,
            "vertical": vertical_range,
            "slant": slant_range,
            "sensor": sensor_range,
        },
        "los": {
            "boundary_height": los_boundary_height,
            "margin": los_margin,
            "visible": los_visible,
            "occlusion": occlusion_indicator,
        },
        "powered_detection": {
            "acoustic_rate": acoustic_rate,
            "detection_rate": powered_detection_rate,
        },
        "glide_detection": {
            "los_angle": los_angle,
            "aspect_angle": aspect_angle,
            "rcs": rcs,
            "radar_rate": radar_rate,
            "radial_velocity": radial_velocity,
            "radial_velocity_rate": radial_velocity_rate,
            "detection_rate": glide_detection_rate,
        },
        "mission": {
            "hazard": mission_hazard,
            "detection": mission_pod,
            "time": mission_time,
            "pod_normalized": pod_normalized,
            "time_normalized": time_normalized,
            "attacker_objective": attacker_objective,
            "defender_pod_normalized": defender_pod_normalized,
            "coverage_area_normalized": coverage_area_normalized,
        },
    }
    validation = validate_symbolic_detection(
        functions,
        configuration_bundle,
        geometry_bundle,
        validation_config,
    )
    function_metadata = {
        name: {
            "name": function.name(),
            "input_names": [function.name_in(index) for index in range(function.n_in())],
            "output_names": [
                function.name_out(index) for index in range(function.n_out())
            ],
            "input_count": function.n_in(),
            "output_count": function.n_out(),
        }
        for name, function in functions.items()
    }
    return {
        "primary_result": {
            "symbolic_variables": standard_symbols,
            "auxiliary_symbols": auxiliary_symbols,
            "expressions": expressions,
            "functions": functions,
            "function_metadata": function_metadata,
            "detection_components": {
                "powered": ("acoustic",),
                "glide": ("radar", "radial_velocity", "rcs"),
                "mission_fusion": "additive_hazard",
            },
        },
        "validation": validation,
        "metadata": {
            "schema_name": "DetectionBundle",
            "schema_version": "1.0.0",
            "producer_phase": 3,
            "producer_module": "p1b_4D.detection",
            "casadi_version": ca.__version__,
            "standard_symbol_order": (
                "z",
                "h",
                "v",
                "gamma",
                "z_sensor",
                "h_sensor",
            ),
            "attacker_objective_id": costs["attacker"]["objective_id"],
            "geometry_schema_version": geometry_bundle["metadata"]["schema_version"],
            "configuration_schema_version": configuration_bundle["metadata"][
                "schema_version"
            ],
            "units": {
                "range": "m",
                "speed": "m/s",
                "angle": "rad",
                "time": "s",
                "hazard": "dimensionless",
                "probability": "dimensionless",
            },
            "goal_position": (
                environment["z_goal"],
                environment["h_goal"],
            ),
            "powered_speed": vehicle["powered_speed"],
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "DETECTION_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def validate_symbolic_detection(
    functions: dict[str, ca.Function],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate graph, dimensions, finiteness, bounds, and configuration reuse."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    costs = configs["cost_config"]
    geometry = geometry_bundle["primary_result"]
    sensor_position = geometry["sensor_position"]
    tangent = geometry["los_geometry"]
    coverage = geometry["coverage"]["normalized_coverage_area"]
    sample_z = 0.5 * (environment["z_start"] + environment["z_goal"])
    sample_h = 0.5 * environment["grid"]["h_max"]
    sample_v = 0.5 * (
        vehicle["glide_speed_min"] + vehicle["glide_speed_max"]
    )
    sample_gamma = np.deg2rad(
        0.5 * (vehicle["gamma_min_deg"] + vehicle["gamma_max_deg"])
    )
    state = (
        sample_z,
        sample_h,
        sample_v,
        sample_gamma,
        float(sensor_position[0]),
        float(sensor_position[1]),
    )
    los_parameters = (
        float(tangent["tangent_point"][0]),
        float(tangent["tangent_slope"]),
        float(tangent["tangent_intercept"]),
    )
    range_outputs = _numeric_outputs(functions["range"], *state[:2], *state[4:])
    powered_outputs = _numeric_outputs(
        functions["powered_detection_components"],
        state[0],
        state[1],
        vehicle["powered_speed"],
        state[4],
        state[5],
    )
    glide_outputs = _numeric_outputs(
        functions["glide_detection_components"],
        *state,
        *los_parameters,
    )
    powered_hazard = powered_outputs[-1] * vehicle["time_step"]
    glide_hazard = glide_outputs[-1] * vehicle["time_step"]
    mission_outputs = _numeric_outputs(
        functions["mission_detection"],
        powered_hazard,
        glide_hazard,
    )
    time_outputs = _numeric_outputs(
        functions["mission_time"],
        vehicle["time_step"],
        vehicle["time_step"],
    )
    objective_outputs = _numeric_outputs(
        functions["attacker_objective"],
        powered_hazard,
        glide_hazard,
        vehicle["time_step"],
        vehicle["time_step"],
    )
    defender_outputs = _numeric_outputs(
        functions["defender_objective"],
        powered_hazard,
        glide_hazard,
        coverage,
    )
    all_outputs = np.asarray(
        range_outputs
        + powered_outputs
        + glide_outputs
        + mission_outputs
        + time_outputs
        + objective_outputs
        + defender_outputs,
        dtype=float,
    )
    probability_tolerance = validation_config[
        "detection_probability_tolerance"
    ]
    expected_attacker_objective = (
        costs["attacker"]["w_pod"] * objective_outputs[2]
        + costs["attacker"]["w_time"] * objective_outputs[3]
    )
    expected_defender_objective = (
        costs["defender"]["w_pod"] * defender_outputs[0]
        + costs["defender"]["w_cover"] * defender_outputs[1]
    )
    checks = {
        "symbolic_graph_construction": all(
            isinstance(function, ca.Function) for function in functions.values()
        ),
        "standard_range_dimensions": (
            functions["range"].n_in() == 4
            and functions["range"].n_out() == 4
        ),
        "mission_detection_dimensions": (
            functions["mission_detection"].n_in() == 2
            and functions["mission_detection"].n_out() == 4
        ),
        "attacker_objective_dimensions": (
            functions["attacker_objective"].n_in() == 4
            and functions["attacker_objective"].n_out() == 5
        ),
        "defender_objective_dimensions": (
            functions["defender_objective"].n_in() == 3
            and functions["defender_objective"].n_out() == 3
        ),
        "finite_numerical_outputs": bool(np.all(np.isfinite(all_outputs))),
        "mission_probability_bounds": (
            -probability_tolerance
            <= mission_outputs[-1]
            <= 1.0 + probability_tolerance
        ),
        "mission_time_addition": abs(
            time_outputs[0] - 2.0 * vehicle["time_step"]
        )
        <= validation_config["objective_tolerance"],
        "attacker_objective_consistency": abs(
            objective_outputs[-1] - expected_attacker_objective
        )
        <= validation_config["objective_tolerance"],
        "attacker_objective_parameter_consistency": (
            costs["attacker"]["objective_id"]
            == configs["bellman_config"]["attacker_objective_id"]
            == configs["nlp_config"]["attacker_objective_id"]
        ),
        "coverage_component_consistency": abs(
            defender_outputs[1] - coverage
        )
        <= validation_config["objective_tolerance"],
        "defender_objective_consistency": abs(
            defender_outputs[2] - expected_defender_objective
        ) <= validation_config["objective_tolerance"],
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "function_count": len(functions),
            "sample_mission_hazard": mission_outputs[0],
            "sample_mission_pod": mission_outputs[-1],
            "sample_mission_time": time_outputs[0],
            "sample_attacker_objective": objective_outputs[-1],
            "sample_defender_pod_normalized": defender_outputs[0],
            "sample_coverage_area_normalized": defender_outputs[1],
            "sample_defender_objective": defender_outputs[2],
            "sample_outputs": all_outputs,
        },
        "tolerances": {
            "probability": probability_tolerance,
            "objective": validation_config["objective_tolerance"],
        },
        "warnings": [],
        "failed_checks": failed_checks,
        "summary": (
            "Phase 3 symbolic detection validation passed"
            if not failed_checks
            else f"Phase 3 symbolic detection failed checks: {failed_checks}"
        ),
    }


def _build_functions(
    symbols: dict[str, ca.SX],
    auxiliary: dict[str, ca.SX],
    expression: dict[str, ca.SX],
) -> dict[str, ca.Function]:
    z, h, v, gamma, z_sensor, h_sensor = (
        symbols[name]
        for name in ("z", "h", "v", "gamma", "z_sensor", "h_sensor")
    )
    tangent_z, tangent_slope, tangent_intercept = (
        auxiliary[name]
        for name in ("tangent_z", "tangent_slope", "tangent_intercept")
    )
    powered_hazard, glide_hazard, powered_time, glide_time = (
        auxiliary[name]
        for name in (
            "powered_hazard",
            "glide_hazard",
            "powered_time",
            "glide_time",
        )
    )
    coverage = auxiliary["coverage_area_normalized"]
    return {
        "range": ca.Function(
            "SensorRangeFunction",
            [z, h, z_sensor, h_sensor],
            [
                expression["horizontal_range"],
                expression["vertical_range"],
                expression["slant_range"],
                expression["sensor_range"],
            ],
            ["z", "h", "z_sensor", "h_sensor"],
            ["horizontal_range", "vertical_range", "slant_range", "sensor_range"],
        ),
        "los": ca.Function(
            "SymbolicLosFunction",
            [z, h, tangent_z, tangent_slope, tangent_intercept],
            [
                expression["los_boundary_height"],
                expression["los_margin"],
                expression["los_visible"],
                expression["occlusion_indicator"],
            ],
            ["z", "h", "tangent_z", "tangent_slope", "tangent_intercept"],
            ["boundary_height", "los_margin", "visible", "occluded"],
        ),
        "powered_detection_components": ca.Function(
            "PoweredDetectionComponentsFunction",
            [z, h, v, z_sensor, h_sensor],
            [expression["acoustic_rate"], expression["powered_detection_rate"]],
            ["z", "h", "v", "z_sensor", "h_sensor"],
            ["acoustic_rate", "powered_detection_rate"],
        ),
        "glide_detection_components": ca.Function(
            "GlideDetectionComponentsFunction",
            [
                z,
                h,
                v,
                gamma,
                z_sensor,
                h_sensor,
                tangent_z,
                tangent_slope,
                tangent_intercept,
            ],
            [
                expression["los_angle"],
                expression["aspect_angle"],
                expression["rcs"],
                expression["radar_rate_raw"],
                expression["radial_velocity"],
                expression["radial_velocity_rate_raw"],
                expression["radar_rate"],
                expression["radial_velocity_rate"],
                expression["glide_detection_rate"],
            ],
            [
                "z",
                "h",
                "v",
                "gamma",
                "z_sensor",
                "h_sensor",
                "tangent_z",
                "tangent_slope",
                "tangent_intercept",
            ],
            [
                "los_angle",
                "aspect_angle",
                "rcs",
                "radar_rate_raw",
                "radial_velocity",
                "radial_velocity_rate_raw",
                "radar_rate",
                "radial_velocity_rate",
                "glide_detection_rate",
            ],
        ),
        "mission_detection": ca.Function(
            "MissionDetectionFunction",
            [powered_hazard, glide_hazard],
            [
                expression["mission_hazard"],
                expression["powered_pod"],
                expression["glide_pod"],
                expression["mission_pod"],
            ],
            ["powered_hazard", "glide_hazard"],
            ["mission_hazard", "powered_pod", "glide_pod", "mission_pod"],
        ),
        "mission_time": ca.Function(
            "MissionTimeFunction",
            [powered_time, glide_time],
            [expression["mission_time"]],
            ["powered_time", "glide_time"],
            ["mission_time"],
        ),
        "attacker_objective": ca.Function(
            "AttackerObjectiveFunction",
            [powered_hazard, glide_hazard, powered_time, glide_time],
            [
                expression["mission_pod"],
                expression["mission_time"],
                expression["pod_normalized"],
                expression["time_normalized"],
                expression["attacker_objective"],
            ],
            ["powered_hazard", "glide_hazard", "powered_time", "glide_time"],
            [
                "mission_pod",
                "mission_time",
                "pod_normalized",
                "time_normalized",
                "attacker_objective",
            ],
        ),
        "defender_components": ca.Function(
            "DefenderObjectiveComponentsFunction",
            [powered_hazard, glide_hazard, coverage],
            [
                expression["defender_pod_normalized"],
                expression["coverage_area_normalized"],
            ],
            ["powered_hazard", "glide_hazard", "coverage_area_fraction"],
            ["pod_normalized", "coverage_area_normalized"],
        ),
        "defender_objective": ca.Function(
            "DefenderObjectiveFunction",
            [powered_hazard, glide_hazard, coverage],
            [
                expression["defender_pod_normalized"],
                expression["coverage_area_normalized"],
                expression["defender_objective"],
            ],
            ["powered_hazard", "glide_hazard", "coverage_area_fraction"],
            ["pod_normalized", "coverage_area_normalized", "defender_objective"],
        ),
    }


def _numeric_outputs(function: ca.Function, *arguments: float) -> list[float]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in outputs]


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict):
        raise TypeError(f"{name} must be a dictionary")
    if not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must have successful status")
