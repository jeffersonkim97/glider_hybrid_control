"""Independent common evaluator for Direction-B selected policies.

Planning arrays and planning-time edge hazards are deliberately ignored.
Every powered and glide edge is reconstructed from physical endpoints and
reevaluated with one common trapezoidal sampling rule.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geometry import los_boundary_height, terrain_height


def evaluate_policy_high_fidelity(
    policy: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    *,
    sample_count: int = 129,
) -> dict[str, Any]:
    """Reevaluate one powered-plus-glide physical policy without snapping."""
    if isinstance(sample_count, bool) or not isinstance(
        sample_count, (int, np.integer)
    ) or sample_count < 2:
        raise ValueError("sample_count must be an integer at least 2")
    if "primary_result" in policy and "trajectory" not in policy:
        policy = policy["primary_result"]

    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    validation = configs["validation_config"]
    geometry = geometry_bundle["primary_result"]
    detection = configs["sensor_config"]["detection"]
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    goal = np.asarray(geometry["goal_position"], dtype=float)
    launch = np.array(
        [environment["z_start"], environment["h_start"]], dtype=float
    )

    trajectory = np.asarray(policy["trajectory"], dtype=float)
    speeds = np.asarray(policy["speed_profile"], dtype=float)
    gammas = np.asarray(policy["gamma_profile"], dtype=float)
    durations = np.asarray(policy["duration_profile"], dtype=float)
    _validate_policy_arrays(trajectory, speeds, gammas, durations)
    if not np.allclose(trajectory[0], policy["switching_point"], rtol=0.0, atol=0.0):
        raise ValueError("trajectory must begin at the exact switching_point")

    minimum_terrain_margin = math.inf
    minimum_glide_los_margin = math.inf
    minimum_powered_occlusion_margin = math.inf
    maximum_endpoint_residual = 0.0
    first_invalid_sample: dict[str, Any] | None = None
    edge_diagnostics: list[dict[str, Any]] = []

    powered_end = trajectory[0]
    powered_delta = powered_end - launch
    powered_distance = float(np.linalg.norm(powered_delta))
    powered_time = powered_distance / float(vehicle["powered_speed"])
    powered_path, powered_times = _sample_edge(
        launch, powered_end, powered_time, sample_count
    )
    powered_terrain = powered_path[:, 1] - terrain_height(
        geometry["terrain_model"], powered_path[:, 0]
    )
    powered_boundary = los_boundary_height(
        geometry["los_geometry"], powered_path[:, 0]
    )
    powered_occlusion = (
        powered_boundary + float(validation["los_tolerance"])
        - powered_path[:, 1]
    )
    powered_domain = _inside_domain(powered_path, environment["airspace"])
    powered_valid = (
        powered_domain
        & (powered_terrain >= -float(validation["terrain_tolerance"]))
        & (powered_occlusion >= 0.0)
    )
    minimum_terrain_margin = min(
        minimum_terrain_margin, float(np.min(powered_terrain))
    )
    minimum_powered_occlusion_margin = float(np.min(powered_occlusion))
    powered_rates = _powered_detection_rate(
        powered_path[:, 0], powered_path[:, 1],
        float(vehicle["powered_speed"]), sensor, detection,
    )
    powered_hazard = float(np.trapezoid(powered_rates, powered_times))
    if not np.all(powered_valid):
        invalid_index = int(np.flatnonzero(~powered_valid)[0])
        first_invalid_sample = _invalid_sample_record(
            "powered", 0, invalid_index, powered_path[invalid_index],
            powered_terrain[invalid_index], powered_occlusion[invalid_index],
            powered_domain[invalid_index],
        )

    glide_hazard = 0.0
    glide_time = 0.0
    for edge_index, (start, end, speed, gamma, duration) in enumerate(
        zip(trajectory[:-1], trajectory[1:], speeds, gammas, durations)
    ):
        displacement = end - start
        reconstructed = start + duration * speed * np.array(
            [math.cos(float(gamma)), math.sin(float(gamma))]
        )
        endpoint_residual = float(np.linalg.norm(reconstructed - end))
        maximum_endpoint_residual = max(
            maximum_endpoint_residual, endpoint_residual
        )
        path, times = _sample_edge(start, end, float(duration), sample_count)
        terrain_margin = path[:, 1] - terrain_height(
            geometry["terrain_model"], path[:, 0]
        )
        boundary = los_boundary_height(geometry["los_geometry"], path[:, 0])
        los_margin = path[:, 1] - boundary
        before_sensor = path[:, 0] < sensor[0]
        visible = (~before_sensor) | (los_margin >= 0.0)
        in_domain = _inside_domain(path, environment["airspace"])
        edge_valid_samples = (
            in_domain
            & (terrain_margin >= -float(validation["terrain_tolerance"]))
            & visible
        )
        if np.any(before_sensor):
            minimum_glide_los_margin = min(
                minimum_glide_los_margin,
                float(np.min(los_margin[before_sensor])),
            )
        minimum_terrain_margin = min(
            minimum_terrain_margin, float(np.min(terrain_margin))
        )
        rates = _glide_detection_rate(
            path[:, 0], path[:, 1], float(speed), float(gamma), sensor,
            boundary, detection,
        )
        edge_hazard = float(np.trapezoid(rates, times))
        glide_hazard += edge_hazard
        glide_time += float(duration)
        edge_diagnostics.append({
            "edge_index": edge_index,
            "sample_count": int(sample_count),
            "duration": float(duration),
            "hazard": edge_hazard,
            "minimum_terrain_margin": float(np.min(terrain_margin)),
            "minimum_los_margin": (
                float(np.min(los_margin[before_sensor]))
                if np.any(before_sensor) else math.inf
            ),
            "maximum_endpoint_residual": endpoint_residual,
            "valid": bool(np.all(edge_valid_samples)),
            "physical_displacement": tuple(float(value) for value in displacement),
        })
        if first_invalid_sample is None and not np.all(edge_valid_samples):
            invalid_index = int(np.flatnonzero(~edge_valid_samples)[0])
            first_invalid_sample = _invalid_sample_record(
                "glide", edge_index, invalid_index, path[invalid_index],
                terrain_margin[invalid_index], los_margin[invalid_index],
                in_domain[invalid_index],
            )

    goal_distance = float(np.linalg.norm(trajectory[-1] - goal))
    arithmetic_tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(goal)))
    )
    reached_goal = goal_distance <= (
        float(validation["goal_radius"]) + arithmetic_tolerance
    )
    endpoint_aligned = maximum_endpoint_residual <= arithmetic_tolerance
    physical_feasible = first_invalid_sample is None and endpoint_aligned
    feasible = physical_feasible and reached_goal
    violation = None
    if first_invalid_sample is not None:
        violation = first_invalid_sample["violation"]
    elif not endpoint_aligned:
        violation = "physical_edge_endpoint_misalignment"
    elif not reached_goal:
        violation = "goal_not_reached_within_radius"

    mission_hazard = powered_hazard + glide_hazard
    mission_time = powered_time + glide_time
    mission_pod = 1.0 - math.exp(-mission_hazard)
    mission_objective, pod_normalized, time_normalized = _attacker_objective(
        mission_hazard, mission_pod, mission_time,
        configs["cost_config"]["attacker"],
    )
    return {
        "feasible": feasible,
        "physical_feasible": physical_feasible,
        "reached_goal": reached_goal,
        "violation": violation,
        "sample_count_per_edge": int(sample_count),
        "powered_hazard": powered_hazard,
        "glide_hazard": glide_hazard,
        "mission_hazard": mission_hazard,
        "powered_time": powered_time,
        "glide_time": glide_time,
        "mission_time": mission_time,
        "mission_pod": mission_pod,
        "attacker_objective": mission_objective,
        "pod_normalized": pod_normalized,
        "time_normalized": time_normalized,
        "goal_distance": goal_distance,
        "minimum_terrain_margin": minimum_terrain_margin,
        "minimum_glide_los_margin": minimum_glide_los_margin,
        "minimum_powered_occlusion_margin": minimum_powered_occlusion_margin,
        "maximum_edge_endpoint_residual": maximum_endpoint_residual,
        "first_invalid_sample": first_invalid_sample,
        "edge_diagnostics": tuple(edge_diagnostics),
        "metadata": {
            "evaluation_role": "direction_b_common_high_fidelity_replay",
            "integration_rule": "equally_spaced_endpoint_inclusive_trapezoid",
            "endpoint_snapping": False,
            "analytic_continuous_time_certificate": False,
        },
    }


def qualify_common_evaluator(
    policy: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    *,
    candidate_sample_count: int = 129,
    reference_sample_count: int = 257,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Apply the frozen 129-vs-257 (or fallback) qualification gate."""
    candidate = evaluate_policy_high_fidelity(
        policy, configuration_bundle, geometry_bundle,
        sample_count=candidate_sample_count,
    )
    reference = evaluate_policy_high_fidelity(
        policy, configuration_bundle, geometry_bundle,
        sample_count=reference_sample_count,
    )
    objective_difference = abs(
        candidate["attacker_objective"] - reference["attacker_objective"]
    )
    pod_difference = abs(candidate["mission_pod"] - reference["mission_pod"])
    checks = {
        "feasibility_classification_matches": (
            candidate["feasible"] == reference["feasible"]
        ),
        "goal_classification_matches": (
            candidate["reached_goal"] == reference["reached_goal"]
        ),
        "attacker_objective_within_tolerance": (
            objective_difference <= absolute_tolerance
        ),
        "mission_pod_within_tolerance": pod_difference <= absolute_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "objective_absolute_difference": objective_difference,
        "mission_pod_absolute_difference": pod_difference,
        "absolute_tolerance": float(absolute_tolerance),
        "candidate_evaluation": candidate,
        "reference_evaluation": reference,
        "fallback_required": not all(checks.values()),
    }


def _validate_policy_arrays(trajectory, speeds, gammas, durations) -> None:
    if (
        trajectory.ndim != 2 or trajectory.shape[1] != 2
        or trajectory.shape[0] < 1 or not np.all(np.isfinite(trajectory))
    ):
        raise ValueError("trajectory must have finite shape (N, 2)")
    edge_count = trajectory.shape[0] - 1
    for name, values in (
        ("speed_profile", speeds),
        ("gamma_profile", gammas),
        ("duration_profile", durations),
    ):
        if values.ndim != 1 or values.size != edge_count:
            raise ValueError(f"{name} must have one entry per trajectory edge")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values")
    if np.any(speeds <= 0.0) or np.any(durations <= 0.0):
        raise ValueError("speed and duration profiles must be positive")


def _sample_edge(start, end, duration, sample_count):
    fractions = np.linspace(0.0, 1.0, sample_count)
    path = start[None, :] + fractions[:, None] * (end - start)[None, :]
    return path, fractions * duration


def _inside_domain(path: np.ndarray, airspace: dict[str, Any]) -> np.ndarray:
    return (
        (path[:, 0] >= float(airspace["z_min"]))
        & (path[:, 0] <= float(airspace["z_max"]))
        & (path[:, 1] >= float(airspace["h_min"]))
        & (path[:, 1] <= float(airspace["h_max"]))
    )


def _powered_detection_rate(z, h, speed, sensor, detection):
    sensor_range = np.maximum(
        np.hypot(sensor[0] - z, sensor[1] - h), detection["range_floor"]
    )
    acoustic = (
        detection["acoustic_coefficient"]
        * speed ** detection["acoustic_speed_exponent"]
        / sensor_range**2
    )
    return detection["acoustic_rate_scale"] * acoustic


def _glide_detection_rate(z, h, speed, gamma, sensor, boundary, detection):
    horizontal = sensor[0] - z
    vertical = sensor[1] - h
    sensor_range = np.maximum(
        np.hypot(horizontal, vertical), detection["range_floor"]
    )
    los_visible = ~((z < sensor[0]) & (h < boundary))
    los_angle = np.arctan2(vertical, horizontal)
    aspect = np.arctan2(
        np.sin(gamma - los_angle), np.cos(gamma - los_angle)
    )
    rcs = detection["rcs_min"] + (
        detection["rcs_max"] - detection["rcs_min"]
    ) * np.cos(aspect) ** 2
    radar = (
        detection["radar_rate_scale"] * detection["radar_coefficient"]
        * rcs / sensor_range**4
    )
    radial_velocity = speed * (
        math.cos(gamma) * horizontal + math.sin(gamma) * vertical
    ) / sensor_range
    doppler = (
        detection["radial_velocity_rate_scale"]
        * detection["doppler_coefficient"] * radial_velocity**2
        / sensor_range**4
    )
    return los_visible.astype(float) * (radar + doppler)


def _attacker_objective(mission_hazard, mission_pod, mission_time, attacker):
    pod_spec = attacker["normalization"]["pod"]
    if pod_spec["method"] == "cumulative_hazard_reference":
        pod_normalized = mission_hazard / float(pod_spec["hazard_reference"])
    elif pod_spec["method"] == "probability":
        pod_normalized = mission_pod
    else:
        raise ValueError(
            f"Unsupported attacker PoD normalization: {pod_spec['method']}"
        )
    time_normalized = mission_time / float(
        attacker["normalization"]["time"]["reference_seconds"]
    )
    objective = (
        float(attacker["w_pod"]) * pod_normalized
        + float(attacker["w_time"]) * time_normalized
    )
    return float(objective), float(pod_normalized), float(time_normalized)


def _invalid_sample_record(
    phase, edge_index, sample_index, position, terrain_margin, los_margin,
    in_domain,
):
    if not bool(in_domain):
        violation = f"{phase}_domain_violation_edge_{edge_index}"
    elif terrain_margin < 0.0:
        violation = f"{phase}_terrain_violation_edge_{edge_index}"
    else:
        violation = f"{phase}_los_violation_edge_{edge_index}"
    return {
        "violation": violation,
        "phase": phase,
        "edge_index": int(edge_index),
        "sample_index": int(sample_index),
        "position": tuple(float(value) for value in position),
        "terrain_margin": float(terrain_margin),
        "los_margin": float(los_margin),
        "inside_domain": bool(in_domain),
    }
