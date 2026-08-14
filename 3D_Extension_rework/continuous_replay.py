"""Unsnapped continuous replay of the authoritative 3D Bellman response.

This is the 3D counterpart of ``p1b_4D.continuous_replay_evaluation``.  It
does not optimize or smooth the path.  It executes the selected
``(v, gamma, psi, duration)`` sequence continuously from the exact switching
point and independently rechecks kinematics, terrain, LOS, goal, detection
hazard, time, and objective against the planning-time result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .geometry import terrain_height


def integrate_action_sequence_3d(
    initial_position: np.ndarray,
    speed_profile: np.ndarray,
    gamma_profile: np.ndarray,
    heading_profile: np.ndarray,
    duration_profile: np.ndarray,
    *,
    max_steps: int,
) -> dict[str, Any]:
    """Integrate one 3D action sequence without any grid-state reset."""
    initial = np.asarray(initial_position, dtype=float)
    speeds = np.asarray(speed_profile, dtype=float)
    gammas = np.asarray(gamma_profile, dtype=float)
    headings = np.asarray(heading_profile, dtype=float)
    durations = np.asarray(duration_profile, dtype=float)
    profiles = (speeds, gammas, headings, durations)
    if initial.shape != (3,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_position must contain three finite coordinates")
    if any(profile.ndim != 1 for profile in profiles):
        raise ValueError("action profiles must be one-dimensional")
    if len({profile.size for profile in profiles}) != 1:
        raise ValueError("action profiles must have equal length")
    if any(not np.all(np.isfinite(profile)) for profile in profiles):
        raise ValueError("action profiles must contain finite values")
    if np.any(speeds <= 0.0) or np.any(durations <= 0.0):
        raise ValueError("speeds and durations must be positive")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if speeds.size > max_steps:
        raise ValueError("action sequence exceeds max_steps")
    velocity = np.column_stack((
        speeds * np.cos(gammas) * np.cos(headings),
        speeds * np.cos(gammas) * np.sin(headings),
        speeds * np.sin(gammas),
    ))
    increments = velocity * durations[:, None]
    path = np.vstack((initial, initial + np.cumsum(increments, axis=0)))
    return {
        "trajectory": path,
        "increments": increments,
        "elapsed_time_s": float(np.sum(durations)),
        "step_count": int(speeds.size),
    }


def _mapped_last_output(function, arguments: list[np.ndarray]) -> np.ndarray:
    count = int(arguments[0].size)
    outputs = function.map(count)(*(value.reshape(1, count) for value in arguments))
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.asarray(output_tuple[-1], dtype=float).reshape(count)


def replay_glide_continuous_3d(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    detection_bundle: dict[str, Any],
    trajectory_result: dict[str, Any],
) -> dict[str, Any]:
    """Replay the extracted glide controls continuously without snapping."""
    if not trajectory_result.get("status", {}).get("success", False):
        raise ValueError("trajectory result must pass validation")
    reference = np.asarray(trajectory_result["glide_trajectory"], dtype=float)
    speeds = np.asarray(trajectory_result["speed_profile_mps"], dtype=float)
    gammas = np.asarray(trajectory_result["gamma_profile_rad"], dtype=float)
    headings = np.asarray(trajectory_result["heading_profile_rad"], dtype=float)
    durations = np.asarray(trajectory_result["duration_profile_s"], dtype=float)
    planned_hazards = np.asarray(trajectory_result["hazard_profile"], dtype=float)
    if reference.shape != (speeds.size + 1, 3):
        raise ValueError("reference trajectory and action profiles are inconsistent")
    integrated = integrate_action_sequence_3d(
        reference[0], speeds, gammas, headings, durations,
        max_steps=int(configuration["bellman"]["maximum_path_steps"]),
    )
    replay_path = np.asarray(integrated["trajectory"], dtype=float)
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    goal = np.asarray(geometry["goal_position"], dtype=float)
    environment = configuration["environment"]
    tolerance = float(configuration["validation"]["terrain_tolerance_m"])
    los_boundary = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]),
        geometry["los_boundary_height"], method="linear",
        bounds_error=False, fill_value=np.inf,
    )
    glide_function = detection_bundle["functions"]["glide_detection_components"]
    replay_hazards: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    feasible = True
    violation: str | None = None

    for edge_index in range(speeds.size):
        start = replay_path[edge_index]
        end = replay_path[edge_index + 1]
        sample_count = int(
            configuration["bellman"]["virtual_edge_quadrature_count"]
            if edge_index == 0
            else configuration["bellman"]["edge_quadrature_count"]
        )
        fractions = np.linspace(0.0, 1.0, sample_count)
        samples = start[None, :] + fractions[:, None] * (end - start)[None, :]
        terrain_margin = samples[:, 2] - terrain_height(
            geometry["terrain_model"], samples[:, 0], samples[:, 1],
        ) - float(configuration["bellman"]["terrain_clearance_m"])
        boundary = los_boundary(samples[:, :2])
        los_margin = samples[:, 2] - boundary
        domain = (
            (samples[:, 0] >= environment["x_bounds_m"][0])
            & (samples[:, 0] <= environment["x_bounds_m"][1])
            & (samples[:, 1] >= environment["y_bounds_m"][0])
            & (samples[:, 1] <= environment["y_bounds_m"][1])
            & (samples[:, 2] >= environment["h_bounds_m"][0])
            & (samples[:, 2] <= environment["h_bounds_m"][1])
        )
        if np.any(terrain_margin < -tolerance):
            feasible = False
            violation = f"terrain_violation_edge_{edge_index}"
        elif np.any(los_margin < -tolerance):
            feasible = False
            violation = f"los_violation_edge_{edge_index}"
        elif not np.all(domain):
            feasible = False
            violation = f"domain_violation_edge_{edge_index}"

        rates = _mapped_last_output(glide_function, [
            samples[:, 0], samples[:, 1], samples[:, 2],
            np.full(sample_count, speeds[edge_index]),
            np.full(sample_count, gammas[edge_index]),
            np.full(sample_count, headings[edge_index]),
            np.full(sample_count, sensor[0]),
            np.full(sample_count, sensor[1]),
            np.full(sample_count, sensor[2]),
        ])
        replay_hazard = float(
            np.trapezoid(rates, fractions) * durations[edge_index]
        )
        replay_hazards.append(replay_hazard)
        start_drift = start - reference[edge_index]
        endpoint_drift = end - reference[edge_index + 1]
        diagnostics.append({
            "edge_index": edge_index,
            "reference_start": tuple(reference[edge_index]),
            "continuous_start": tuple(start),
            "reference_endpoint": tuple(reference[edge_index + 1]),
            "continuous_endpoint": tuple(end),
            "start_drift": tuple(start_drift),
            "start_drift_norm_m": float(np.linalg.norm(start_drift)),
            "endpoint_drift": tuple(endpoint_drift),
            "endpoint_drift_norm_m": float(np.linalg.norm(endpoint_drift)),
            "minimum_terrain_margin_m": float(np.min(terrain_margin)),
            "minimum_los_margin_m": float(np.min(los_margin)),
            "domain_clear": bool(np.all(domain)),
            "planned_hazard": float(planned_hazards[edge_index]),
            "continuous_hazard": replay_hazard,
            "hazard_residual": replay_hazard - float(planned_hazards[edge_index]),
            "quadrature_count": sample_count,
        })
        if not feasible:
            break

    replay_hazard_array = np.asarray(replay_hazards, dtype=float)
    used_count = replay_hazard_array.size
    continuous_glide_hazard = float(np.sum(replay_hazard_array))
    continuous_glide_time = float(np.sum(durations[:used_count]))
    powered_hazard = float(
        trajectory_result["mission"]["hazard"]
        - np.sum(trajectory_result["hazard_profile"])
    )
    powered_time = float(trajectory_result["mission"]["powered_time_s"])
    objective_outputs = detection_bundle["functions"]["attacker_objective"](
        powered_hazard, continuous_glide_hazard,
        powered_time, continuous_glide_time,
    )
    objective_tuple = objective_outputs if isinstance(objective_outputs, tuple) else (objective_outputs,)
    continuous_mission_cost = float(objective_tuple[-1])
    continuous_mission_hazard = powered_hazard + continuous_glide_hazard
    goal_distance = float(np.linalg.norm(replay_path[min(used_count, replay_path.shape[0] - 1)] - goal))
    reached_goal = bool(
        used_count == speeds.size
        and goal_distance <= float(configuration["bellman"]["goal_radius_m"]) + 1.0e-9
    )
    if feasible and not reached_goal:
        feasible = False
        violation = "action_sequence_exhausted_without_reaching_goal"

    endpoint_drifts = np.asarray([
        diagnostic["endpoint_drift_norm_m"] for diagnostic in diagnostics
    ])
    hazard_residuals = np.asarray([
        diagnostic["hazard_residual"] for diagnostic in diagnostics
    ])
    maximum_endpoint_drift = float(np.max(endpoint_drifts)) if endpoint_drifts.size else 0.0
    maximum_hazard_residual = float(np.max(np.abs(hazard_residuals))) if hazard_residuals.size else 0.0
    planned_cost = float(trajectory_result["mission"]["cost"])
    objective_residual = continuous_mission_cost - planned_cost
    checks = {
        "continuous_replay_feasible": feasible,
        "goal_reached": reached_goal,
        "reference_endpoint_drift_small": maximum_endpoint_drift <= 1.0e-9,
        "hazard_replay_matches_planning": maximum_hazard_residual <= 1.0e-7,
        "objective_replay_matches_planning": abs(objective_residual) <= 1.0e-7,
        "all_actions_replayed": used_count == speeds.size,
        "continuous_nlp_absent": True,
        "state_reset_absent": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "feasible": feasible,
        "violation": violation,
        "reached_goal": reached_goal,
        "trajectory": replay_path[: used_count + 1],
        "reference_trajectory": reference,
        "continuous_glide_time_s": continuous_glide_time,
        "continuous_glide_hazard": continuous_glide_hazard,
        "continuous_glide_pod": float(1.0 - math.exp(-continuous_glide_hazard)),
        "continuous_mission_time_s": powered_time + continuous_glide_time,
        "continuous_mission_hazard": continuous_mission_hazard,
        "continuous_mission_pod": float(1.0 - math.exp(-continuous_mission_hazard)),
        "continuous_mission_cost": continuous_mission_cost,
        "replay_hazard_profile": replay_hazard_array,
        "step_diagnostics": tuple(diagnostics),
        "metadata": {
            "source_process": "p1b_4D continuous replay validation",
            "state_reset": False,
            "continuous_nlp_applied": False,
            "action_sequence_changed": False,
            "hazard_integration": "same per-edge trapezoidal quadrature as 3D planner",
        },
        "validation": {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "metrics": {
                "step_count_used": used_count,
                "goal_distance_m": goal_distance,
                "maximum_endpoint_drift_m": maximum_endpoint_drift,
                "maximum_hazard_edge_residual": maximum_hazard_residual,
                "total_glide_hazard_residual": continuous_glide_hazard - float(np.sum(planned_hazards)),
                "mission_objective_residual": objective_residual,
                "minimum_terrain_margin_m": float(min(
                    diagnostic["minimum_terrain_margin_m"] for diagnostic in diagnostics
                )),
                "minimum_los_margin_m": float(min(
                    diagnostic["minimum_los_margin_m"] for diagnostic in diagnostics
                )),
            },
        },
        "status": {
            "success": not failed,
            "message": "3D continuous replay validation passed" if not failed else f"failed: {failed}",
        },
    }
