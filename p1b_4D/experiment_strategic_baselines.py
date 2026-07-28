"""Plan item 4 (p1b_roadmap_0727.md): multi-terrain x strategic-baseline
sweep -- sensor-selection library, imported by experiment_multiterrain_
baselines.py.

Four independent sensor-selection rules, each producing ONE candidate
z_sensor per terrain WITHOUT peeking at the true (Stackelberg) attacker
best response:
  - fixed:            sensor_config's own default_z_sensor, untouched.
  - coverage_only:     maximizes geometry-only LOS coverage area. Never
                        builds detection/stage_cost/bellman -- cheapest.
  - nominal_path:      maximizes hazard against a FIXED, sensor-independent
                        "nominal" attacker path -- computed once via the
                        real Bellman solver with (w_pod=0, w_time=1) at an
                        arbitrary reference sensor position (bounds
                        midpoint), so the nominal path ignores detection
                        entirely and only reflects the time-optimal route.
  - stackelberg:       DIRECT search using the REAL attacker best response
                        (same as the existing production Defender solve).

Every candidate is then re-evaluated with the SAME authoritative
`evaluate_defender_position` (real Bellman best-response attacker) -- the
selection processes above never share results with each other or with the
final evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.optimize import direct

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.stackelberg_solver import (
    evaluate_defender_position,
    solve_attacker_best_response,
)
from p1b_4D.phase_logging import close_phase_logger


def _configuration_at_sensor(configuration_bundle: dict, z_sensor: float) -> dict:
    cb = deepcopy(configuration_bundle)
    cb["primary_result"]["sensor_config"]["default_z_sensor"] = float(z_sensor)
    return cb


def select_fixed_sensor(configuration_bundle: dict) -> float:
    return float(configuration_bundle["primary_result"]["sensor_config"]["default_z_sensor"])


def select_coverage_only_sensor(configuration_bundle: dict, bounds: tuple[float, float]) -> float:
    """Maximize LOS coverage area alone -- no detection/attacker information."""

    def negative_coverage(x: np.ndarray) -> float:
        z_sensor = float(x[0])
        cb = _configuration_at_sensor(configuration_bundle, z_sensor)
        geometry = build_geometry_bundle(cb)
        return -geometry["primary_result"]["coverage"]["normalized_coverage_area"]

    result = direct(negative_coverage, bounds=[bounds], maxfun=60, locally_biased=False)
    return float(result.x[0])


def compute_nominal_attacker_path(configuration_bundle: dict, reference_z_sensor: float) -> dict:
    """Time-only-optimal attacker response (w_pod=0, w_time=1) at an
    arbitrary reference sensor position -- used only to fix the switching-
    point/feasible-seed structure; the resulting path's own cost never
    depended on detection, so it does not "know" about any candidate
    sensor position evaluated against it later.
    """
    cb = deepcopy(configuration_bundle)
    attacker_costs = cb["primary_result"]["cost_config"]["attacker"]
    attacker_costs["w_pod"] = 0.0
    attacker_costs["w_time"] = 1.0
    result = solve_attacker_best_response(reference_z_sensor, cb, "nominal-path-reference")
    return result["primary_result"]["best_found_attacker_response"]


def hazard_against_fixed_path(
    configuration_bundle: dict,
    z_sensor: float,
    nominal_response: dict,
) -> float:
    """Total mission hazard a sensor at z_sensor would accumulate against
    the FIXED nominal trajectory (recomputing detection rates only, not
    re-solving for a new path).
    """
    cb = _configuration_at_sensor(configuration_bundle, z_sensor)
    geometry = build_geometry_bundle(cb)
    detection = build_symbolic_detection_bundle(cb, geometry)
    functions = detection["primary_result"]["functions"]
    vehicle = cb["primary_result"]["vehicle_config"]
    time_step = vehicle["time_step"]

    # Powered segment: straight line, fixed shape, re-evaluate acoustic
    # rate at the new sensor position along the SAME stored path.
    powered_path = np.asarray(nominal_response["powered_path"])
    sensor_position = geometry["primary_result"]["sensor_position"]
    powered_speed = vehicle["powered_speed"]
    powered_hazard = 0.0
    if powered_path.shape[0] >= 2:
        rates = []
        for point in powered_path:
            outputs = functions["powered_detection_components"](
                float(point[0]), float(point[1]), powered_speed,
                float(sensor_position[0]), float(sensor_position[1]),
            )
            rates.append(float(outputs[-1]))
        distance = float(np.hypot(*(powered_path[-1] - powered_path[0])))
        powered_time = distance / powered_speed if powered_speed > 0 else 0.0
        sample_times = np.linspace(0.0, powered_time, len(rates))
        powered_hazard = float(np.trapezoid(rates, sample_times)) if powered_time > 0.0 else 0.0

    # Glide segment: use the trajectory/speed/gamma profiles already
    # stored on the nominal response (one action per step already
    # resolved), re-evaluate glide_detection_rate at the new sensor.
    trajectory = np.asarray(nominal_response["trajectory"])
    speed_profile = np.asarray(nominal_response["speed_profile"])
    gamma_profile = np.asarray(nominal_response["gamma_profile"])
    glide_hazard = 0.0
    step_count = min(trajectory.shape[0] - 1, speed_profile.shape[0], gamma_profile.shape[0])
    for index in range(step_count):
        z_point, h_point = trajectory[index]
        v_point = float(speed_profile[index])
        gamma_point = float(gamma_profile[index])
        outputs = functions["glide_detection_components"](
            float(z_point), float(h_point), v_point, gamma_point,
            float(sensor_position[0]), float(sensor_position[1]),
        )
        glide_rate = float(outputs[-1])
        glide_hazard += glide_rate * time_step

    return powered_hazard + glide_hazard


def select_nominal_path_optimal_sensor(
    configuration_bundle: dict,
    bounds: tuple[float, float],
    nominal_response: dict,
) -> float:
    def negative_hazard(x: np.ndarray) -> float:
        return -hazard_against_fixed_path(configuration_bundle, float(x[0]), nominal_response)

    result = direct(negative_hazard, bounds=[bounds], maxfun=60, locally_biased=False)
    return float(result.x[0])


def select_stackelberg_optimal_sensor(configuration_bundle: dict, bounds: tuple[float, float]) -> float:
    def negative_defender_objective(x: np.ndarray) -> float:
        z_sensor = float(x[0])
        evaluation = evaluate_defender_position(z_sensor, configuration_bundle, "baseline-selection")
        return -evaluation["primary_result"]["defender_objective"]

    result = direct(negative_defender_objective, bounds=[bounds], maxfun=60, locally_biased=False)
    return float(result.x[0])
