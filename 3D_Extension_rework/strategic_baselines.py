"""Strategic Defender baselines for the canonical 3D single-hill case.

The selection rules mirror ``p1b_4D.experiment_strategic_baselines``.  A
baseline may use only the information named by its rule; every selected
sensor is subsequently evaluated against the same authoritative adaptive
Bellman follower.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np
from scipy.optimize import direct

from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stackelberg import configuration_for_sensor, evaluate_defender_position


def select_fixed_sensor(configuration: dict[str, Any]) -> tuple[float, float]:
    """Return the preconfigured, attacker-independent sensor position."""
    x, y = configuration["environment"]["sensor_xy_m"]
    return float(x), float(y)


def select_coverage_only_sensor(
    configuration: dict[str, Any], *, maxfun: int = 80,
) -> dict[str, Any]:
    """Maximize geometry-only normalized LOS volume in the 2D sensor region."""
    bounds = configuration["defender_search"]
    evaluations: list[dict[str, float]] = []

    def negative_coverage(position: np.ndarray) -> float:
        sensor_xy = float(position[0]), float(position[1])
        local = configuration_for_sensor(configuration, sensor_xy)
        geometry = build_geometry(local, require_tangent_manifold=False)
        coverage = float(geometry["coverage"]["normalized_los_volume"])
        evaluations.append({"x_m": sensor_xy[0], "y_m": sensor_xy[1], "coverage": coverage})
        return -coverage

    result = direct(
        negative_coverage,
        bounds=[bounds["x_bounds_m"], bounds["y_bounds_m"]],
        maxfun=int(maxfun), locally_biased=False,
    )
    return {
        "sensor_xy_m": (float(result.x[0]), float(result.x[1])),
        "selection_score": -float(result.fun),
        "evaluation_count": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": tuple(evaluations),
    }


def compute_nominal_attacker_response(
    configuration: dict[str, Any], reference_sensor_xy_m: tuple[float, float],
) -> dict[str, Any]:
    """Solve one detection-independent, time-only physical Bellman path."""
    nominal_configuration = deepcopy(configuration)
    attacker = nominal_configuration["cost"]["attacker"]
    attacker["w_pod"] = 0.0
    attacker["w_time"] = 1.0
    result = evaluate_defender_position(
        reference_sensor_xy_m, nominal_configuration, retain_full_pipeline=True,
    )
    if not result["status"]["success"]:
        raise RuntimeError(result["status"]["message"])
    return result


def _mapped_last_output(function, arguments: list[np.ndarray]) -> np.ndarray:
    count = int(arguments[0].size)
    outputs = function.map(count)(*(value.reshape(1, count) for value in arguments))
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.asarray(values[-1], dtype=float).reshape(count)


def hazard_against_fixed_nominal_path(
    configuration: dict[str, Any],
    sensor_xy_m: tuple[float, float],
    nominal_response: dict[str, Any],
) -> float:
    """Re-evaluate detection along one fixed physical path at a new sensor."""
    local = configuration_for_sensor(configuration, sensor_xy_m)
    geometry = build_geometry(local, require_tangent_manifold=False)
    detection = build_symbolic_detection_bundle(local, geometry)
    functions = detection["functions"]
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    trajectory = nominal_response["pipeline"]["trajectory"]

    powered_path = np.asarray(trajectory["powered_path"], dtype=float)
    powered_speed = float(local["vehicle"]["powered_speed_mps"])
    powered_rates = _mapped_last_output(
        functions["powered_detection_components"],
        [
            powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
            np.full(powered_path.shape[0], powered_speed),
            np.full(powered_path.shape[0], sensor[0]),
            np.full(powered_path.shape[0], sensor[1]),
            np.full(powered_path.shape[0], sensor[2]),
        ],
    )
    powered_distance = np.concatenate((
        np.asarray([0.0]),
        np.cumsum(np.linalg.norm(np.diff(powered_path, axis=0), axis=1)),
    ))
    powered_times = powered_distance / powered_speed
    powered_hazard = float(np.trapezoid(powered_rates, powered_times))

    path = np.asarray(trajectory["glide_trajectory"], dtype=float)
    speeds = np.asarray(trajectory["speed_profile_mps"], dtype=float)
    gammas = np.asarray(trajectory["gamma_profile_rad"], dtype=float)
    headings = np.asarray(trajectory["heading_profile_rad"], dtype=float)
    durations = np.asarray(trajectory["duration_profile_s"], dtype=float)
    glide_hazard = 0.0
    for edge_index in range(speeds.size):
        sample_count = int(
            local["bellman"]["virtual_edge_quadrature_count"]
            if edge_index == 0 else local["bellman"]["edge_quadrature_count"]
        )
        fractions = np.linspace(0.0, 1.0, sample_count)
        samples = (
            path[edge_index][None, :]
            + fractions[:, None] * (path[edge_index + 1] - path[edge_index])[None, :]
        )
        rates = _mapped_last_output(
            functions["glide_detection_components"],
            [
                samples[:, 0], samples[:, 1], samples[:, 2],
                np.full(sample_count, speeds[edge_index]),
                np.full(sample_count, gammas[edge_index]),
                np.full(sample_count, headings[edge_index]),
                np.full(sample_count, sensor[0]),
                np.full(sample_count, sensor[1]),
                np.full(sample_count, sensor[2]),
            ],
        )
        glide_hazard += float(np.trapezoid(rates, fractions) * durations[edge_index])
    return powered_hazard + glide_hazard


def select_nominal_path_sensor(
    configuration: dict[str, Any],
    nominal_response: dict[str, Any],
    *,
    maxfun: int = 80,
) -> dict[str, Any]:
    """Maximize detection hazard against the fixed time-only path."""
    bounds = configuration["defender_search"]
    evaluations: list[dict[str, float]] = []

    def negative_hazard(position: np.ndarray) -> float:
        sensor_xy = float(position[0]), float(position[1])
        hazard = hazard_against_fixed_nominal_path(
            configuration, sensor_xy, nominal_response,
        )
        evaluations.append({"x_m": sensor_xy[0], "y_m": sensor_xy[1], "hazard": hazard})
        return -hazard

    result = direct(
        negative_hazard,
        bounds=[bounds["x_bounds_m"], bounds["y_bounds_m"]],
        maxfun=int(maxfun), locally_biased=False,
    )
    return {
        "sensor_xy_m": (float(result.x[0]), float(result.x[1])),
        "selection_score": -float(result.fun),
        "evaluation_count": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "evaluations": tuple(evaluations),
    }


def reconcile_stackelberg_candidate(
    stackelberg_record: dict[str, Any],
    evaluated_baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Promote any evaluated baseline that beats the budgeted search result."""
    candidates = [("hierarchical_search", stackelberg_record)] + [
        (name, record) for name, record in evaluated_baselines.items()
        if record.get("feasible", False) and np.isfinite(record["defender_objective"])
    ]
    source, selected = max(
        candidates,
        key=lambda item: (
            item[1]["defender_objective"],
            -item[1]["sensor_position_m"][0],
            -item[1]["sensor_position_m"][1],
        ),
    )
    return {
        "source": source,
        "promoted": source != "hierarchical_search",
        "record": selected,
    }


def evaluate_selected_sensors(
    selections: dict[str, tuple[float, float]],
    evaluator: Callable[[float, float], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply the same authoritative cached follower evaluator to all rules."""
    return {
        name: evaluator(float(position[0]), float(position[1]))
        for name, position in selections.items()
    }
