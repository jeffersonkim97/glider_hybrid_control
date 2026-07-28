"""Off-grid continuous replay of a Bellman attacker response.

This does **not** prove continuous-dynamics optimality -- it answers a
narrower, still-important question: if the discrete-optimal action
sequence `(v_k, gamma_k)` is executed exactly, tracking position
continuously instead of snapping to the next grid cell after every step,
how much do mission_time/hazard/PoD/goal-miss differ from what the
planning-time (grid-based) computation reported, and does the reported
sensor-position ranking survive?

Naming follows that scope deliberately: `replay_glide_continuous`, not
`continuous_optimum`. See discrete_optimality_proposition.md's "scope"
section -- this module is the empirical check backing that section's
claims about how far the discretized-exact answer can drift from the
continuous quantity it approximates, not a new optimality proof.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import los_boundary_height, terrain_height


def integrate_action_sequence(
    initial_position: np.ndarray,
    speed_profile: np.ndarray,
    gamma_profile: np.ndarray,
    *,
    time_step: float,
    max_steps: int,
) -> dict[str, Any]:
    """Integrate an action sequence without applying mission-feasibility rules.

    This low-level routine deliberately knows nothing about terrain, LOS, or
    the goal region.  It exists so that kinematic validation is not coupled to
    the production replay rule that a mission must reach its goal.
    """
    initial = np.asarray(initial_position, dtype=float)
    speeds = np.asarray(speed_profile, dtype=float)
    gammas = np.asarray(gamma_profile, dtype=float)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_position must contain two finite coordinates")
    if speeds.ndim != 1 or gammas.ndim != 1 or speeds.size != gammas.size:
        raise ValueError("speed_profile and gamma_profile must be equal-length 1D arrays")
    if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(gammas)):
        raise ValueError("action profiles must contain only finite values")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be one positive finite scalar")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 0:
        raise ValueError("max_steps must be one nonnegative integer")
    if speeds.size > max_steps:
        raise ValueError("action sequence exceeds max_steps")

    increments = np.column_stack((
        speeds * np.cos(gammas) * time_step,
        speeds * np.sin(gammas) * time_step,
    ))
    trajectory = np.vstack((initial, initial + np.cumsum(increments, axis=0)))
    return {
        "trajectory": trajectory,
        "increments": increments,
        "elapsed_time": float(speeds.size) * float(time_step),
        "step_count": int(speeds.size),
    }


def replay_glide_continuous(
    switching_point: np.ndarray,
    speed_profile: np.ndarray,
    gamma_profile: np.ndarray,
    *,
    time_step: float,
    goal_position: np.ndarray,
    goal_radius: float,
    terrain_model: Any,
    los_geometry: dict[str, Any],
    sensor_position: np.ndarray,
    glide_detection_rate_function,
    terrain_tolerance: float,
    segment_check_count: int,
    max_steps: int,
    reference_trajectory: np.ndarray | None = None,
    duration_profile: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay one glide policy continuously from its exact switching point.

    `glide_detection_rate_function` must be the detection bundle's
    `functions["glide_detection_components"]` CasADi Function (z, h, v,
    gamma, z_sensor, h_sensor) -> outputs whose LAST entry is
    `glide_detection_rate` -- matching the same rate this replay
    accumulates hazard from, and the same one the discrete pipeline uses
    (see bellman.extract_coarse_candidate), so this is a like-for-like
    comparison of accumulation method (continuous position vs. grid
    lookup), not a different physical model.
    """
    z, h = float(switching_point[0]), float(switching_point[1])
    goal_z, goal_h = float(goal_position[0]), float(goal_position[1])
    sensor_z, sensor_h = float(sensor_position[0]), float(sensor_position[1])
    trajectory = [(z, h)]
    total_time = 0.0
    total_hazard = 0.0
    feasible = True
    violation = None
    reached_goal = False
    step_diagnostics: list[dict[str, Any]] = []

    reference = None
    if reference_trajectory is not None:
        reference = np.asarray(reference_trajectory, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 2:
            raise ValueError("reference_trajectory must have shape (N, 2)")

    step_count = min(len(speed_profile), len(gamma_profile))
    durations = None
    if duration_profile is not None:
        durations = np.asarray(duration_profile, dtype=float)
        if durations.ndim != 1 or durations.size != step_count:
            raise ValueError("duration_profile must match the action profiles")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("duration_profile must contain positive finite values")
    if step_count == 0:
        reached_goal = math.hypot(z - goal_z, h - goal_h) <= goal_radius
        if not reached_goal:
            feasible, violation = False, "no_steps_and_not_at_goal"

    for step_index in range(step_count):
        if step_index >= max_steps:
            feasible, violation = False, "max_steps_exceeded"
            break

        v = float(speed_profile[step_index])
        gamma = float(gamma_profile[step_index])
        step_time = float(durations[step_index]) if durations is not None else time_step
        delta_z = v * math.cos(gamma) * step_time
        delta_h = v * math.sin(gamma) * step_time
        step_start = np.array([z, h], dtype=float)

        # Exact same goal-circle-segment-intersection formula as
        # bellman.construct_coarse_transitions, evaluated at this one
        # continuous state instead of the whole grid at once.
        relative_z = z - goal_z
        relative_h = h - goal_h
        quadratic_a = delta_z**2 + delta_h**2
        quadratic_b = relative_z * delta_z + relative_h * delta_h
        quadratic_c = relative_z**2 + relative_h**2 - goal_radius**2
        discriminant = quadratic_b**2 - quadratic_a * quadratic_c
        terminal_fraction = 1.0
        step_reaches_goal = False
        if quadratic_a > 0.0 and quadratic_c > 0.0 and discriminant >= 0.0:
            first_intersection = (-quadratic_b - math.sqrt(discriminant)) / quadratic_a
            if 0.0 < first_intersection <= 1.0:
                terminal_fraction = first_intersection
                step_reaches_goal = True
        elif quadratic_c <= 0.0:
            # Already inside the goal circle before this step starts.
            reached_goal = True
            break

        # Sub-sample within this step (same segment_check_count convention
        # as the discrete transition validity check) for terrain/LOS
        # violations that a single endpoint check could miss.
        fractions = np.linspace(0.0, 1.0, segment_check_count)[1:]
        minimum_terrain_margin = math.inf
        minimum_los_margin = math.inf
        first_invalid_sample = None
        for fraction in fractions:
            effective_fraction = fraction * terminal_fraction
            sample_z = z + effective_fraction * delta_z
            sample_h = h + effective_fraction * delta_h
            ground = float(terrain_height(terrain_model, np.asarray([sample_z]))[0])
            terrain_margin = sample_h - ground
            boundary = float(los_boundary_height(los_geometry, np.asarray([sample_z]))[0])
            los_margin = math.inf if sample_z >= sensor_z else sample_h - boundary
            minimum_terrain_margin = min(minimum_terrain_margin, terrain_margin)
            minimum_los_margin = min(minimum_los_margin, los_margin)
            if terrain_margin < -terrain_tolerance:
                feasible, violation = False, f"terrain_violation_step_{step_index}"
                first_invalid_sample = {
                    "fraction": float(effective_fraction),
                    "position": (float(sample_z), float(sample_h)),
                    "terrain_height": ground,
                    "terrain_margin": terrain_margin,
                    "los_boundary_height": boundary,
                    "los_margin": los_margin,
                }
                break
            visible = sample_z >= sensor_z or los_margin >= 0.0
            if not visible:
                feasible, violation = False, f"los_violation_step_{step_index}"
                first_invalid_sample = {
                    "fraction": float(effective_fraction),
                    "position": (float(sample_z), float(sample_h)),
                    "terrain_height": ground,
                    "terrain_margin": terrain_margin,
                    "los_boundary_height": boundary,
                    "los_margin": los_margin,
                }
                break
        nominal_endpoint = step_start + np.array([delta_z, delta_h])
        diagnostic: dict[str, Any] = {
            "step_index": step_index,
            "continuous_start": tuple(step_start),
            "continuous_nominal_endpoint": tuple(nominal_endpoint),
            "terminal_fraction": float(terminal_fraction),
            "minimum_terrain_margin": float(minimum_terrain_margin),
            "minimum_los_margin": float(minimum_los_margin),
            "first_invalid_sample": first_invalid_sample,
        }
        if reference is not None and step_index < reference.shape[0]:
            grid_start = reference[step_index]
            diagnostic["reference_grid_start"] = tuple(grid_start)
            diagnostic["start_drift"] = tuple(step_start - grid_start)
            diagnostic["start_drift_norm"] = float(np.linalg.norm(step_start - grid_start))
        if reference is not None and step_index + 1 < reference.shape[0]:
            grid_endpoint = reference[step_index + 1]
            diagnostic["reference_grid_endpoint"] = tuple(grid_endpoint)
            diagnostic["nominal_endpoint_to_grid"] = tuple(nominal_endpoint - grid_endpoint)
        step_diagnostics.append(diagnostic)
        if not feasible:
            break

        # Hazard accumulated using the rate at the START of this step
        # (left-Riemann convention), matching extract_coarse_candidate's
        # own accumulation exactly -- the only difference here is that
        # (z, h) is the true continuous position, not a grid-snapped one.
        rate_outputs = glide_detection_rate_function(z, h, v, gamma, sensor_z, sensor_h)
        rate = float(rate_outputs[-1] if isinstance(rate_outputs, (tuple, list)) else rate_outputs)
        step_duration = terminal_fraction * step_time
        total_hazard += rate * step_duration
        total_time += step_duration

        z = z + terminal_fraction * delta_z
        h = h + terminal_fraction * delta_h
        trajectory.append((z, h))

        if step_reaches_goal:
            reached_goal = True
            break
    else:
        if step_count > 0 and not reached_goal:
            final_goal_distance = math.hypot(z - goal_z, h - goal_h)
            if final_goal_distance <= goal_radius + 1.0e-9:
                reached_goal = True
            else:
                feasible, violation = False, "action_sequence_exhausted_without_reaching_goal"

    goal_miss = math.hypot(z - goal_z, h - goal_h)
    if feasible and not reached_goal and goal_miss > goal_radius:
        feasible, violation = False, "goal_not_reached_within_radius"

    mission_pod = 1.0 - math.exp(-total_hazard)

    return {
        "feasible": feasible,
        "violation": violation,
        "reached_goal": reached_goal,
        "continuous_mission_time": total_time,
        "continuous_glide_hazard": total_hazard,
        "continuous_mission_pod_glide_only": mission_pod,
        "goal_miss": goal_miss,
        "trajectory": trajectory,
        "step_count_used": len(trajectory) - 1,
        "step_diagnostics": step_diagnostics,
    }
