"""Authoritative CasADi symbolic detection and mission-objective model (3D).

Mirrors p1b_4D.detection's role and formulas, extended from (z, h) to
(x, y, h) and from (v, gamma) to (v, gamma, heading). The radar/Doppler/
acoustic rate formulas are already expressed as Euclidean-range and
velocity-dot-product relationships, so they extend to 3D with no change
in functional form -- only the LOS visibility gate changes shape: instead
of a 1D tangent-line CasADi interpolant, this bakes the geometry bundle's
3D boolean viewshed mask (los_mask) into a trilinear CasADi interpolant,
exactly mirroring how p1b_4D bakes its 1D swept LOS boundary.
"""

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
        Successful Phase 2 GeometryBundle supplying the 3D viewshed mask
        and normalized coverage.

    Outputs
    -------
    dict
        Universal result envelope containing standard symbols, expressions,
        CasADi Functions, component metadata, and validation.

    Assumptions
    -----------
    Detection coefficients and normalization methods come only from Phase 1.
    The LOS visibility volume comes only from Phase 2 and is baked into
    this bundle's functions as a fixed trilinear lookup table (this bundle
    is always rebuilt alongside its geometry_bundle for one fixed sensor
    position, so the mask is never stale relative to the functions using it).

    Notes
    -----
    This function builds symbolic graphs only. It does not optimize,
    construct cost maps, reconstruct terrain, write files, or plot.
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
    geometry = geometry_bundle["primary_result"]

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    h = ca.SX.sym("h")
    v = ca.SX.sym("v")
    gamma = ca.SX.sym("gamma")
    heading = ca.SX.sym("heading")
    x_sensor = ca.SX.sym("x_sensor")
    y_sensor = ca.SX.sym("y_sensor")
    h_sensor = ca.SX.sym("h_sensor")
    powered_hazard = ca.SX.sym("powered_hazard")
    glide_hazard = ca.SX.sym("glide_hazard")
    powered_time = ca.SX.sym("powered_time")
    glide_time = ca.SX.sym("glide_time")
    coverage_volume_normalized_input = ca.SX.sym("coverage_volume_normalized")

    horizontal_range_x = x_sensor - x
    horizontal_range_y = y_sensor - y
    vertical_range = h_sensor - h
    slant_range = ca.sqrt(
        horizontal_range_x**2 + horizontal_range_y**2 + vertical_range**2
    )
    sensor_range = ca.fmax(slant_range, detection["range_floor"])

    # 3D viewshed baked as a trilinear interpolant over the boolean
    # los_mask -- the direct generalization of p1b_4D's 1D swept-boundary
    # interpolant to a genuine volume (no single "tangent line" exists
    # once terrain is a 2D surface, see geometry.py's docstring).
    x_grid = np.asarray(geometry["terrain_arrays"]["x"], dtype=float)
    y_grid = np.asarray(geometry["terrain_arrays"]["y"], dtype=float)
    grid = environment["grid"]
    h_grid = np.linspace(grid["h_min"], grid["h_max"], grid["h_count"])
    los_mask = np.asarray(geometry["los_masks"]["los_mask"], dtype=float)
    los_visibility_interpolant = ca.interpolant(
        "los_visibility_3d", "linear",
        [x_grid, y_grid, h_grid],
        los_mask.ravel(order="F"),
    )
    los_visible = los_visibility_interpolant(ca.vertcat(x, y, h))

    velocity_x = v * ca.cos(gamma) * ca.cos(heading)
    velocity_y = v * ca.cos(gamma) * ca.sin(heading)
    velocity_h = v * ca.sin(gamma)
    radial_velocity = (
        velocity_x * horizontal_range_x
        + velocity_y * horizontal_range_y
        + velocity_h * vertical_range
    ) / sensor_range
    cos_aspect = radial_velocity / v
    rcs = detection["rcs_min"] + (
        detection["rcs_max"] - detection["rcs_min"]
    ) * cos_aspect**2

    acoustic_rate = (
        detection["acoustic_coefficient"]
        * v ** detection["acoustic_speed_exponent"]
        / sensor_range**2
    )
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
        mission_time / attacker_normalization["time"]["reference_seconds"]
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
    coverage_volume_normalized = ca.fmin(
        ca.fmax(coverage_volume_normalized_input, 0.0), 1.0,
    )
    defender_objective = (
        costs["defender"]["w_pod"] * defender_pod_normalized
        + costs["defender"]["w_cover"] * coverage_volume_normalized
    )

    standard_symbols = {
        "x": x, "y": y, "h": h, "v": v, "gamma": gamma, "heading": heading,
        "x_sensor": x_sensor, "y_sensor": y_sensor, "h_sensor": h_sensor,
    }
    auxiliary_symbols = {
        "powered_hazard": powered_hazard,
        "glide_hazard": glide_hazard,
        "powered_time": powered_time,
        "glide_time": glide_time,
        "coverage_volume_normalized": coverage_volume_normalized_input,
    }
    functions = _build_functions(
        standard_symbols, auxiliary_symbols,
        {
            "horizontal_range_x": horizontal_range_x,
            "horizontal_range_y": horizontal_range_y,
            "vertical_range": vertical_range,
            "slant_range": slant_range,
            "sensor_range": sensor_range,
            "los_visible": los_visible,
            "occlusion_indicator": 1.0 - los_visible,
            "acoustic_rate": acoustic_rate,
            "powered_detection_rate": powered_detection_rate,
            "cos_aspect": cos_aspect,
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
            "coverage_volume_normalized": coverage_volume_normalized,
            "defender_objective": defender_objective,
        },
    )
    expressions = {
        "range": {
            "horizontal_x": horizontal_range_x,
            "horizontal_y": horizontal_range_y,
            "vertical": vertical_range,
            "slant": slant_range,
            "sensor": sensor_range,
        },
        "los": {"visible": los_visible, "occlusion": 1.0 - los_visible},
        "powered_detection": {
            "acoustic_rate": acoustic_rate,
            "detection_rate": powered_detection_rate,
        },
        "glide_detection": {
            "cos_aspect": cos_aspect,
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
            "coverage_volume_normalized": coverage_volume_normalized,
        },
    }
    validation = validate_symbolic_detection(
        functions, configuration_bundle, geometry_bundle, validation_config,
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
            "schema_name": "DetectionBundle3D",
            "schema_version": "1.0.0",
            "producer_phase": 3,
            "producer_module": "p1b_3DExtension.detection",
            "casadi_version": ca.__version__,
            "standard_symbol_order": (
                "x", "y", "h", "v", "gamma", "heading", "x_sensor", "y_sensor", "h_sensor",
            ),
            "attacker_objective_id": costs["attacker"]["objective_id"],
            "geometry_schema_version": geometry_bundle["metadata"]["schema_version"],
            "configuration_schema_version": configuration_bundle["metadata"][
                "schema_version"
            ],
            "units": {
                "range": "m", "speed": "m/s", "angle": "rad", "time": "s",
                "hazard": "dimensionless", "probability": "dimensionless",
            },
            "goal_position": tuple(
                float(v) for v in geometry["goal_position"]
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
    coverage = geometry["coverage"]["normalized_coverage_volume"]
    sample_x = 0.5 * (environment["x_start"] + environment["x_goal"])
    sample_y = 0.5 * (environment["y_start"] + environment["y_goal"])
    sample_h = 0.5 * environment["grid"]["h_max"]
    sample_v = 0.5 * (vehicle["glide_speed_min"] + vehicle["glide_speed_max"])
    sample_gamma = np.deg2rad(
        0.5 * (vehicle["gamma_min_deg"] + vehicle["gamma_max_deg"])
    )
    sample_heading = 0.0
    state = (
        sample_x, sample_y, sample_h, sample_v, sample_gamma, sample_heading,
        float(sensor_position[0]), float(sensor_position[1]), float(sensor_position[2]),
    )
    range_outputs = _numeric_outputs(
        functions["range"], state[0], state[1], state[2], state[6], state[7], state[8],
    )
    powered_outputs = _numeric_outputs(
        functions["powered_detection_components"],
        state[0], state[1], state[2], state[3], state[6], state[7], state[8],
    )
    glide_outputs = _numeric_outputs(functions["glide_detection_components"], *state)
    powered_hazard = powered_outputs[-1] * vehicle["time_step"]
    glide_hazard = glide_outputs[-1] * vehicle["time_step"]
    mission_outputs = _numeric_outputs(
        functions["mission_detection"], powered_hazard, glide_hazard,
    )
    time_outputs = _numeric_outputs(
        functions["mission_time"], vehicle["time_step"], vehicle["time_step"],
    )
    objective_outputs = _numeric_outputs(
        functions["attacker_objective"],
        powered_hazard, glide_hazard, vehicle["time_step"], vehicle["time_step"],
    )
    defender_outputs = _numeric_outputs(
        functions["defender_objective"], powered_hazard, glide_hazard, coverage,
    )
    all_outputs = np.asarray(
        range_outputs + powered_outputs + glide_outputs + mission_outputs
        + time_outputs + objective_outputs + defender_outputs,
        dtype=float,
    )
    probability_tolerance = validation_config["detection_probability_tolerance"]
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
            functions["range"].n_in() == 6 and functions["range"].n_out() == 5
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
            -probability_tolerance <= mission_outputs[-1] <= 1.0 + probability_tolerance
        ),
        "mission_time_addition": abs(
            time_outputs[0] - 2.0 * vehicle["time_step"]
        ) <= validation_config["objective_tolerance"],
        "attacker_objective_consistency": abs(
            objective_outputs[-1] - expected_attacker_objective
        ) <= validation_config["objective_tolerance"],
        "attacker_objective_parameter_consistency": (
            costs["attacker"]["objective_id"]
            == configs["bellman_config"]["attacker_objective_id"]
        ),
        "coverage_component_consistency": abs(
            defender_outputs[1] - coverage
        ) <= validation_config["objective_tolerance"],
        "defender_objective_consistency": abs(
            defender_outputs[2] - expected_defender_objective
        ) <= validation_config["objective_tolerance"],
        # Not the sensor's own exact continuous position: that point sits
        # inside whichever grid cell the ray-marched viewshed happens to
        # straddle (the sensor can genuinely be on a slope with real
        # near-field self-occlusion in some directions, confirmed
        # separately), so trilinear interpolation there is not guaranteed
        # to read > 0.5. Straight up from the sensor at the airspace
        # ceiling is unambiguous: nothing can occlude a point directly
        # above the observer.
        "los_visible_directly_above_sensor": bool(
            _numeric_outputs(
                functions["los"],
                float(sensor_position[0]), float(sensor_position[1]),
                float(environment["grid"]["h_max"]),
                float(sensor_position[0]), float(sensor_position[1]), float(sensor_position[2]),
            )[0] > 0.5
        ),
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
            "sample_coverage_volume_normalized": defender_outputs[1],
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
    x, y, h, v, gamma, heading, x_sensor, y_sensor, h_sensor = (
        symbols[name]
        for name in ("x", "y", "h", "v", "gamma", "heading", "x_sensor", "y_sensor", "h_sensor")
    )
    powered_hazard, glide_hazard, powered_time, glide_time = (
        auxiliary[name]
        for name in ("powered_hazard", "glide_hazard", "powered_time", "glide_time")
    )
    coverage = auxiliary["coverage_volume_normalized"]
    return {
        "range": ca.Function(
            "SensorRangeFunction3D",
            [x, y, h, x_sensor, y_sensor, h_sensor],
            [
                expression["horizontal_range_x"], expression["horizontal_range_y"],
                expression["vertical_range"], expression["slant_range"], expression["sensor_range"],
            ],
            ["x", "y", "h", "x_sensor", "y_sensor", "h_sensor"],
            ["horizontal_range_x", "horizontal_range_y", "vertical_range", "slant_range", "sensor_range"],
        ),
        "los": ca.Function(
            "SymbolicLosFunction3D",
            [x, y, h, x_sensor, y_sensor, h_sensor],
            [expression["los_visible"], expression["occlusion_indicator"]],
            ["x", "y", "h", "x_sensor", "y_sensor", "h_sensor"],
            ["visible", "occluded"],
        ),
        "powered_detection_components": ca.Function(
            "PoweredDetectionComponentsFunction3D",
            [x, y, h, v, x_sensor, y_sensor, h_sensor],
            [expression["acoustic_rate"], expression["powered_detection_rate"]],
            ["x", "y", "h", "v", "x_sensor", "y_sensor", "h_sensor"],
            ["acoustic_rate", "powered_detection_rate"],
        ),
        "glide_detection_components": ca.Function(
            "GlideDetectionComponentsFunction3D",
            [x, y, h, v, gamma, heading, x_sensor, y_sensor, h_sensor],
            [
                expression["cos_aspect"], expression["rcs"], expression["radar_rate_raw"],
                expression["radial_velocity"], expression["radial_velocity_rate_raw"],
                expression["radar_rate"], expression["radial_velocity_rate"],
                expression["glide_detection_rate"],
            ],
            ["x", "y", "h", "v", "gamma", "heading", "x_sensor", "y_sensor", "h_sensor"],
            [
                "cos_aspect", "rcs", "radar_rate_raw", "radial_velocity",
                "radial_velocity_rate_raw", "radar_rate", "radial_velocity_rate",
                "glide_detection_rate",
            ],
        ),
        "mission_detection": ca.Function(
            "MissionDetectionFunction",
            [powered_hazard, glide_hazard],
            [
                expression["mission_hazard"], expression["powered_pod"],
                expression["glide_pod"], expression["mission_pod"],
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
                expression["mission_pod"], expression["mission_time"],
                expression["pod_normalized"], expression["time_normalized"],
                expression["attacker_objective"],
            ],
            ["powered_hazard", "glide_hazard", "powered_time", "glide_time"],
            ["mission_pod", "mission_time", "pod_normalized", "time_normalized", "attacker_objective"],
        ),
        "defender_components": ca.Function(
            "DefenderObjectiveComponentsFunction",
            [powered_hazard, glide_hazard, coverage],
            [expression["defender_pod_normalized"], expression["coverage_volume_normalized"]],
            ["powered_hazard", "glide_hazard", "coverage_volume_fraction"],
            ["pod_normalized", "coverage_volume_normalized"],
        ),
        "defender_objective": ca.Function(
            "DefenderObjectiveFunction",
            [powered_hazard, glide_hazard, coverage],
            [
                expression["defender_pod_normalized"], expression["coverage_volume_normalized"],
                expression["defender_objective"],
            ],
            ["powered_hazard", "glide_hazard", "coverage_volume_fraction"],
            ["pod_normalized", "coverage_volume_normalized", "defender_objective"],
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
