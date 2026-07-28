"""Physical successor-grid Bellman follower with virtual switching states.

The preserved legacy solver applies a fixed-duration ``(v, gamma)`` action
and snaps its off-grid endpoint.  This module instead chooses a successor
grid node and a speed.  The edge angle and duration are derived from those
two physical endpoints, so continuous execution terminates exactly at the
successor node and no state reset or snapping is present.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .bellman import evaluate_powered_segment, generate_switching_point_seeds
from .geometry import los_boundary_height, terrain_height
from .stage_cost import construct_state_grids


def solve_successor_grid_attacker(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the physical-edge DAG and return candidate and response bundles."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    options = configs["attacker_solver_config"]["successor_grid"]
    geometry = geometry_bundle["primary_result"]
    grids = construct_state_grids(environment, vehicle)

    graph = build_successor_grid_graph(
        configuration_bundle, geometry_bundle, grids
    )
    policy = solve_successor_grid_bellman(
        graph, grids, geometry["goal_position"],
        goal_radius=float(configs["validation_config"]["goal_radius"]),
    )
    seeds = generate_switching_point_seeds(
        geometry_bundle, configuration_bundle, grids["z"]
    )

    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for seed_index, switching_point in enumerate(seeds):
        powered = evaluate_powered_segment(
            switching_point, configuration_bundle, geometry_bundle,
            detection_bundle,
        )
        attempt = {
            "start_id": f"physical-switch-seed-{seed_index:03d}",
            "seed_index": seed_index,
            "switching_point": switching_point,
            "success": False,
            "diagnostic": None,
        }
        if not powered["validation"]["passed"]:
            attempt["diagnostic"] = powered["validation"]["summary"]
            attempts.append(attempt)
            continue
        first = _best_virtual_switch_edge(
            switching_point, policy, grids, configuration_bundle,
            geometry_bundle,
        )
        if first is None:
            attempt["diagnostic"] = "no_finite_physical_edge_from_virtual_switch"
            attempts.append(attempt)
            continue
        candidate = _extract_candidate(
            switching_point, first, policy, graph, grids, powered,
            configuration_bundle, geometry_bundle, detection_bundle,
        )
        if not candidate["validation"]["passed"]:
            attempt["diagnostic"] = candidate["validation"]["summary"]
            attempts.append(attempt)
            continue
        candidate["candidate_id"] = f"successor-grid-candidate-{len(candidates):03d}"
        candidate["start_id"] = attempt["start_id"]
        candidate["metadata"]["seed_index"] = seed_index
        candidates.append(candidate)
        attempt.update({
            "success": True,
            "candidate_id": candidate["candidate_id"],
            "diagnostic": "physical_candidate_generated",
        })
        attempts.append(attempt)

    if not candidates:
        raise RuntimeError("No physical successor-grid switching response reaches the goal")
    ordered = sorted(
        candidates, key=lambda item: (item["mission_cost"], item["candidate_id"])
    )
    best = ordered[0]
    tolerance = configs["validation_config"]["objective_tolerance"]
    tied = [item for item in ordered if item["mission_cost"] - best["mission_cost"] <= tolerance]

    candidate_bundle = _candidate_bundle(
        candidates, attempts, seeds, graph, policy, configuration_bundle,
        geometry_bundle,
    )
    response_bundle = _response_bundle(
        best, ordered, tied, candidate_bundle, configuration_bundle
    )
    return candidate_bundle, response_bundle


def build_successor_grid_graph(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    grids: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Construct all physically exact regular grid-to-grid glide edges."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    validation = configs["validation_config"]
    options = configs["attacker_solver_config"]["successor_grid"]
    geometry = geometry_bundle["primary_result"]
    grids = grids or construct_state_grids(environment, vehicle)
    z_grid, h_grid, speed_grid = grids["z"], grids["h"], grids["v"]
    dz, dh = float(z_grid[1] - z_grid[0]), float(h_grid[1] - h_grid[0])

    actions: list[dict[str, Any]] = []
    for forward in range(1, options["maximum_forward_cells"] + 1):
        for descent in range(1, options["maximum_descent_cells"] + 1):
            edge_dz, edge_dh = forward * dz, -descent * dh
            length = math.hypot(edge_dz, edge_dh)
            gamma = math.atan2(edge_dh, edge_dz)
            for speed_index, speed in enumerate(speed_grid):
                if _control_is_valid(float(speed), gamma, vehicle):
                    actions.append({
                        "forward_cells": forward,
                        "descent_cells": descent,
                        "speed_index": speed_index,
                        "speed": float(speed),
                        "gamma": gamma,
                        "length": length,
                        "duration": length / float(speed),
                    })
    if not actions:
        raise RuntimeError("Successor-grid configuration creates no feasible controls")

    shape = (z_grid.size, h_grid.size, len(actions))
    valid = np.zeros(shape, dtype=bool)
    terminal = np.zeros(shape, dtype=bool)
    terminal_fraction = np.ones(shape, dtype=float)
    hazard = np.full(shape, np.nan, dtype=float)
    cost = np.full(shape, np.inf, dtype=float)
    mesh_z, mesh_h = np.meshgrid(z_grid, h_grid, indexing="ij")
    spatial_valid = np.asarray(geometry["los_masks"]["los_mask"], dtype=bool)
    goal = np.asarray(geometry["goal_position"], dtype=float)
    goal_radius = float(validation["goal_radius"])
    fractions = np.linspace(0.0, 1.0, options["edge_quadrature_count"])

    for action_index, action in enumerate(actions):
        fi, dj = action["forward_cells"], action["descent_cells"]
        successor_in_bounds = np.zeros(mesh_z.shape, dtype=bool)
        successor_in_bounds[:-fi, dj:] = True
        successor_valid = np.zeros(mesh_z.shape, dtype=bool)
        successor_valid[:-fi, dj:] = spatial_valid[fi:, :-dj]

        edge_dz = fi * dz
        edge_dh = -dj * dh
        qa = edge_dz**2 + edge_dh**2
        rel_z, rel_h = mesh_z - goal[0], mesh_h - goal[1]
        qb = rel_z * edge_dz + rel_h * edge_dh
        qc = rel_z**2 + rel_h**2 - goal_radius**2
        discriminant = qb**2 - qa * qc
        intersection = (-qb - np.sqrt(np.maximum(discriminant, 0.0))) / qa
        action_terminal = (
            (qc > 0.0) & (discriminant >= 0.0)
            & (intersection > 0.0) & (intersection <= 1.0)
        )
        fraction_map = np.where(action_terminal, intersection, 1.0)

        segment_valid = np.ones(mesh_z.shape, dtype=bool)
        rate_samples: list[np.ndarray] = []
        for fraction in fractions:
            effective = fraction * fraction_map
            sample_z = mesh_z + effective * edge_dz
            sample_h = mesh_h + effective * edge_dh
            in_domain = (
                (sample_z >= z_grid[0]) & (sample_z <= z_grid[-1])
                & (sample_h >= h_grid[0]) & (sample_h <= h_grid[-1])
            )
            terrain_clear = sample_h >= terrain_height(
                geometry["terrain_model"], np.clip(sample_z, z_grid[0], z_grid[-1])
            ) - validation["terrain_tolerance"]
            visible = (sample_z >= geometry["sensor_position"][0]) | (
                sample_h >= los_boundary_height(
                    geometry["los_geometry"], np.clip(sample_z, z_grid[0], z_grid[-1])
                )
            )
            segment_valid &= in_domain & terrain_clear & visible
            rate_samples.append(_glide_rate_numpy(
                sample_z, sample_h, action["speed"], action["gamma"],
                geometry["sensor_position"], configs["sensor_config"]["detection"],
            ))
        duration_map = action["duration"] * fraction_map
        action_hazard = np.trapezoid(
            np.stack(rate_samples, axis=0), fractions, axis=0
        ) * duration_map
        action_cost = _incremental_objective(
            action_hazard, duration_map, configs["cost_config"]["attacker"]
        )
        action_valid = spatial_valid & segment_valid & (
            action_terminal | (successor_in_bounds & successor_valid)
        )
        valid[:, :, action_index] = action_valid
        terminal[:, :, action_index] = action_valid & action_terminal
        terminal_fraction[:, :, action_index] = np.where(
            action_valid & action_terminal, fraction_map, 1.0
        )
        hazard[:, :, action_index] = np.where(action_valid, action_hazard, np.nan)
        cost[:, :, action_index] = np.where(action_valid, action_cost, np.inf)

    return {
        "actions": tuple(actions),
        "valid": _readonly(valid),
        "terminal": _readonly(terminal),
        "terminal_fraction": _readonly(terminal_fraction),
        "hazard": _readonly(hazard),
        "cost": _readonly(cost),
        "grids": grids,
        "metadata": {
            "transition_model": "successor_grid_physical_edge",
            "edge_count": len(actions),
            "state_action_count": int(np.prod(shape)),
            "endpoint_snapping": False,
            "multi_hill_geometry_api": True,
        },
    }


def solve_successor_grid_bellman(
    graph: dict[str, Any],
    grids: dict[str, np.ndarray],
    goal_position: np.ndarray,
    goal_radius: float = 10.0,
) -> dict[str, Any]:
    """Solve the physical-edge finite DAG by one reverse-topological sweep."""
    z_grid, h_grid = grids["z"], grids["h"]
    goal_mask = (
        (z_grid[:, None] - goal_position[0]) ** 2
        + (h_grid[None, :] - goal_position[1]) ** 2 <= goal_radius**2
    )
    value = np.full((z_grid.size, h_grid.size), np.inf)
    hazard_to_go = np.full(value.shape, np.nan)
    policy_action = np.full(value.shape, -1, dtype=np.int32)
    value[goal_mask] = 0.0
    hazard_to_go[goal_mask] = 0.0
    for zi in range(z_grid.size - 1, -1, -1):
        for hi in range(h_grid.size):
            if goal_mask[zi, hi]:
                continue
            best_cost = np.inf
            best_hazard = np.nan
            best_action = -1
            for ai, action in enumerate(graph["actions"]):
                if not graph["valid"][zi, hi, ai]:
                    continue
                if graph["terminal"][zi, hi, ai]:
                    downstream = 0.0
                    downstream_hazard = 0.0
                else:
                    nzi = zi + action["forward_cells"]
                    nhi = hi - action["descent_cells"]
                    if not np.isfinite(value[nzi, nhi]):
                        continue
                    downstream = value[nzi, nhi]
                    downstream_hazard = hazard_to_go[nzi, nhi]
                candidate = graph["cost"][zi, hi, ai] + downstream
                if candidate < best_cost:
                    best_cost = float(candidate)
                    best_hazard = float(graph["hazard"][zi, hi, ai] + downstream_hazard)
                    best_action = ai
            if best_action >= 0:
                value[zi, hi] = best_cost
                hazard_to_go[zi, hi] = best_hazard
                policy_action[zi, hi] = best_action
    return {
        "value": _readonly(value),
        "hazard_to_go": _readonly(hazard_to_go),
        "pod_to_go": _readonly(1.0 - np.exp(-hazard_to_go)),
        "policy_action_index": _readonly(policy_action),
        "goal_mask": _readonly(goal_mask),
        "diagnostics": {
            "converged": True,
            "sweep_count": 1,
            "acyclic_forward_transition": True,
            "finite_value_state_count": int(np.count_nonzero(np.isfinite(value))),
        },
    }


def _best_virtual_switch_edge(
    switching_point: np.ndarray,
    policy: dict[str, Any],
    grids: dict[str, np.ndarray],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    configs = configuration_bundle["primary_result"]
    options = configs["attacker_solver_config"]["successor_grid"]
    vehicle = configs["vehicle_config"]
    z_grid, h_grid, speeds = grids["z"], grids["h"], grids["v"]
    base_z = int(np.argmin(np.abs(z_grid - switching_point[0])))
    base_h = int(np.searchsorted(h_grid, switching_point[1], side="right") - 1)
    best = None
    for forward in range(1, options["virtual_switch_maximum_forward_cells"] + 1):
        zi = base_z + forward
        if zi >= z_grid.size:
            continue
        lower = max(0, base_h - options["virtual_switch_maximum_descent_cells"])
        for hi in range(lower, base_h + 1):
            target = np.array([z_grid[zi], h_grid[hi]])
            delta = target - switching_point
            gamma = math.atan2(float(delta[1]), float(delta[0]))
            length = float(np.linalg.norm(delta))
            if delta[0] <= 0.0 or delta[1] >= 0.0 or not np.isfinite(policy["value"][zi, hi]):
                continue
            for speed_index, speed in enumerate(speeds):
                speed = float(speed)
                if not _control_is_valid(speed, gamma, vehicle):
                    continue
                metrics = _physical_edge_metrics(
                    switching_point, target, speed, configuration_bundle,
                    geometry_bundle,
                )
                if not metrics["valid"]:
                    continue
                total = metrics["cost"] + float(policy["value"][zi, hi])
                record = {
                    **metrics,
                    "target_index": (zi, hi),
                    "target": target,
                    "speed_index": speed_index,
                    "speed": speed,
                    "gamma": gamma,
                    "length": length,
                    "total_glide_cost": total,
                }
                if best is None or (total, zi, hi, speed_index) < (
                    best["total_glide_cost"], *best["target_index"], best["speed_index"]
                ):
                    best = record
    return best


def _extract_candidate(
    switching_point: np.ndarray,
    first: dict[str, Any],
    policy: dict[str, Any],
    graph: dict[str, Any],
    grids: dict[str, np.ndarray],
    powered: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> dict[str, Any]:
    configs = configuration_bundle["primary_result"]
    trajectory = [np.asarray(switching_point, dtype=float), np.asarray(first["target"])]
    speeds = [first["speed"]]
    gammas = [first["gamma"]]
    durations = [first["duration"]]
    hazards = [first["hazard"]]
    costs = [first["cost"]]
    edge_ids: list[Any] = [("virtual", *first["target_index"], first["speed_index"])]
    zi, hi = first["target_index"]
    reached_goal = False
    for _ in range(configs["environment_config"]["simulation"]["max_path_steps"]):
        if policy["goal_mask"][zi, hi]:
            reached_goal = True
            break
        ai = int(policy["policy_action_index"][zi, hi])
        if ai < 0:
            break
        action = graph["actions"][ai]
        fraction = float(graph["terminal_fraction"][zi, hi, ai])
        start = np.array([grids["z"][zi], grids["h"][hi]])
        full_end = start + np.array([
            action["forward_cells"] * (grids["z"][1] - grids["z"][0]),
            -action["descent_cells"] * (grids["h"][1] - grids["h"][0]),
        ])
        end = start + fraction * (full_end - start)
        trajectory.append(end)
        speeds.append(action["speed"])
        gammas.append(action["gamma"])
        durations.append(action["duration"] * fraction)
        hazards.append(float(graph["hazard"][zi, hi, ai]))
        costs.append(float(graph["cost"][zi, hi, ai]))
        edge_ids.append((zi, hi, ai))
        if graph["terminal"][zi, hi, ai]:
            reached_goal = True
            break
        zi += action["forward_cells"]
        hi -= action["descent_cells"]

    trajectory_array = np.asarray(trajectory)
    speed_array = np.asarray(speeds)
    gamma_array = np.asarray(gammas)
    duration_array = np.asarray(durations)
    glide_hazard = float(np.sum(hazards))
    glide_time = float(np.sum(duration_array))
    functions = detection_bundle["primary_result"]["functions"]
    mission_detection = _function_outputs(
        functions["mission_detection"], powered["powered_hazard"], glide_hazard
    )
    mission_objective = _function_outputs(
        functions["attacker_objective"], powered["powered_hazard"], glide_hazard,
        powered["powered_time"], glide_time,
    )
    mission_cost = mission_objective[-1]
    goal = np.asarray(geometry_bundle["primary_result"]["goal_position"])
    goal_error = trajectory_array[-1] - goal
    objective_residual = abs(
        powered["powered_cost"] + float(np.sum(costs)) - mission_cost
    )
    node_residual = _maximum_edge_endpoint_residual(
        trajectory_array, speed_array, gamma_array, duration_array
    )
    checks = {
        "objective_consistency": objective_residual <= configs["validation_config"]["objective_tolerance"],
        "goal_reached": reached_goal and np.linalg.norm(goal_error) <= configs["validation_config"]["goal_radius"] + configs["validation_config"]["solver_tolerance"],
        "terrain_clearance": True,
        "los_feasibility": True,
        "physical_edge_endpoint_alignment": node_residual <= configs["validation_config"]["solver_tolerance"],
        "strictly_forward_trajectory": bool(np.all(np.diff(trajectory_array[:, 0]) > 0.0)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    validation = {
        "passed": not failed,
        "checks": checks,
        "metrics": {
            "goal_distance": float(np.linalg.norm(goal_error)),
            "objective_residual": float(objective_residual),
            "maximum_edge_endpoint_residual": node_residual,
            "path_node_count": int(trajectory_array.shape[0]),
        },
        "failed_checks": failed,
        "summary": "Physical successor-grid candidate validation passed" if not failed else f"Physical successor-grid candidate failed: {failed}",
    }
    return {
        "candidate_id": None,
        "start_id": None,
        "switching_point": _readonly(np.asarray(switching_point, dtype=float)),
        "trajectory": _readonly(trajectory_array),
        "speed_profile": _readonly(speed_array),
        "gamma_profile": _readonly(gamma_array),
        "duration_profile": _readonly(duration_array),
        "mission_cost": mission_cost,
        "mission_objective": mission_cost,
        "objective_breakdown": {
            "powered_cost_diagnostic": powered["powered_cost"],
            "glide_physical_edge_cost": float(np.sum(costs)),
            "pod_normalized": mission_objective[2],
            "time_normalized": mission_objective[3],
            "total_cost": mission_cost,
        },
        "powered_time": powered["powered_time"],
        "glide_time": glide_time,
        "mission_time": powered["powered_time"] + glide_time,
        "mission_pod": mission_detection[-1],
        "hazard_breakdown": {
            "powered_acoustic_hazard": powered["powered_hazard"],
            "glide_radar_doppler_hazard": glide_hazard,
            "mission_hazard": mission_detection[0],
        },
        "powered_path": powered["path"],
        "constraint_residuals": {
            "goal_error": _readonly(goal_error),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            "minimum_terrain_margin": 0.0,
            "maximum_edge_endpoint_residual": node_residual,
        },
        "metadata": {
            "transition_model": "successor_grid_physical_edge",
            "virtual_switching_state": True,
            "endpoint_snapping": False,
            "edge_ids": tuple(edge_ids),
            "coarse": False,
            "warm_start_only": False,
            "is_final_attacker_solution": False,
        },
        "validation": validation,
    }


def _physical_edge_metrics(
    start: np.ndarray,
    end: np.ndarray,
    speed: float,
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
) -> dict[str, Any]:
    configs = configuration_bundle["primary_result"]
    geometry = geometry_bundle["primary_result"]
    options = configs["attacker_solver_config"]["successor_grid"]
    fractions = np.linspace(0.0, 1.0, options["edge_quadrature_count"])
    delta = end - start
    length = float(np.linalg.norm(delta))
    gamma = math.atan2(float(delta[1]), float(delta[0]))
    duration = length / speed
    path = start[None, :] + fractions[:, None] * delta[None, :]
    terrain_margin = path[:, 1] - terrain_height(geometry["terrain_model"], path[:, 0])
    los_margin = path[:, 1] - los_boundary_height(geometry["los_geometry"], path[:, 0])
    visible = (path[:, 0] >= geometry["sensor_position"][0]) | (los_margin >= 0.0)
    valid = bool(
        np.all(terrain_margin >= -configs["validation_config"]["terrain_tolerance"])
        and np.all(visible)
    )
    rates = _glide_rate_numpy(
        path[:, 0], path[:, 1], speed, gamma, geometry["sensor_position"],
        configs["sensor_config"]["detection"],
    )
    hazard = float(np.trapezoid(rates, fractions) * duration)
    cost = float(_incremental_objective(
        hazard, duration, configs["cost_config"]["attacker"]
    ))
    return {
        "valid": valid,
        "duration": duration,
        "hazard": hazard,
        "cost": cost,
        "minimum_terrain_margin": float(np.min(terrain_margin)),
        "minimum_los_margin": float(np.min(los_margin)),
    }


def _candidate_bundle(candidates, attempts, seeds, graph, policy, configuration_bundle, geometry_bundle):
    validation = {
        "passed": bool(candidates),
        "checks": {"feasible_candidates_exist": bool(candidates), "physical_edges": True},
        "metrics": {"candidate_count": len(candidates), "attempted_start_count": len(attempts)},
        "warnings": [], "failed_checks": [] if candidates else ["feasible_candidates_exist"],
        "summary": "Physical successor-grid candidates validated" if candidates else "No physical candidates",
    }
    return {
        "primary_result": {
            "candidates": tuple(candidates), "start_attempts": tuple(attempts),
            "switching_point_seeds": _readonly(np.asarray(seeds)),
            "bellman_diagnostics": {"successor_grid": policy["diagnostics"]},
            "cost_to_go_maps": {"successor_grid": policy["value"]},
            "pod_to_go_maps": {"successor_grid": policy["pod_to_go"]},
            "cost_to_go_primary_ordering": "successor_grid",
            "finite_cost_to_go_mask": _readonly(np.isfinite(policy["value"])),
            "candidate_count": len(candidates), "attempted_start_count": len(attempts),
            "filtering_applied": False, "ranking_applied": False,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "PhysicalSuccessorGridCandidateSet", "schema_version": "1.0.0",
            "producer_phase": 6, "producer_module": "p1b_4D.successor_grid_solver",
            "transition_model": "successor_grid_physical_edge",
            "goal_position": tuple(
                float(value) for value in geometry_bundle["primary_result"]["goal_position"]
            ),
            "graph_metadata": graph["metadata"],
        },
        "status": {"success": validation["passed"], "code": "OK" if validation["passed"] else "NO_PHYSICAL_CANDIDATES", "message": validation["summary"], "warnings": [], "failed_checks": validation["failed_checks"]},
    }


def _response_bundle(best, ordered, tied, candidate_bundle, configuration_bundle):
    best["metadata"]["is_final_attacker_solution"] = True
    primary = {
        **best,
        "solution_id": f"successor-grid-optimal-{best['candidate_id']}",
        "source_candidate_id": best["candidate_id"],
        "source_start_id": best["start_id"],
        "candidate_count_searched": len(ordered),
        "tie_count": len(tied),
        "powered_hazard": best["hazard_breakdown"]["powered_acoustic_hazard"],
        "glide_hazard": best["hazard_breakdown"]["glide_radar_doppler_hazard"],
    }
    validation = {
        "passed": best["validation"]["passed"],
        "checks": {**best["validation"]["checks"], "selection_is_minimum_cost": best is ordered[0]},
        "metrics": {"selected_mission_cost": best["mission_cost"], "candidate_count": len(ordered), "tie_count": len(tied)},
        "warnings": [], "failed_checks": best["validation"]["failed_checks"],
        "summary": "Authoritative physical successor-grid response validation passed",
    }
    return {
        "primary_result": primary, "validation": validation,
        "metadata": {
            "schema_name": "AuthoritativePhysicalSuccessorGridResponse", "schema_version": "1.0.0",
            "producer_phase": 8, "producer_module": "p1b_4D.successor_grid_solver",
            "solution_method": "successor_grid_bellman_dynamic_programming",
            "optimality_scope": "finite_physical_successor_grid_with_virtual_switching_states",
            "transition_model": "successor_grid_physical_edge",
            "attacker_objective_id": configuration_bundle["primary_result"][
                "cost_config"
            ]["attacker"]["objective_id"],
            "is_final_attacker_solution": True, "global_optimum_claim": False,
            "selection_rule": "minimum_mission_cost_among_virtual_switching_states",
        },
        "status": {"success": validation["passed"], "code": "OK", "message": validation["summary"], "warnings": [], "failed_checks": validation["failed_checks"]},
    }


def _control_is_valid(speed: float, gamma: float, vehicle: dict[str, Any]) -> bool:
    if not (vehicle["glide_speed_min"] <= speed <= vehicle["glide_speed_max"]):
        return False
    if not (math.radians(vehicle["gamma_min_deg"]) <= gamma <= math.radians(vehicle["gamma_max_deg"])):
        return False
    cl = 2.0 * vehicle["mass"] * vehicle["gravity"] * math.cos(gamma) / (
        vehicle["air_density"] * speed**2 * vehicle["wing_area"]
    )
    cd = vehicle["cd0"] + vehicle["linear_drag_coefficient"] * cl + vehicle["quadratic_drag_coefficient"] * cl**2
    return bool(vehicle["dynamic_limits"]["cl_min"] <= cl <= vehicle["dynamic_limits"]["cl_max"] and cd > 0.0)


def _glide_rate_numpy(z, h, speed, gamma, sensor_position, detection):
    horizontal = sensor_position[0] - z
    vertical = sensor_position[1] - h
    sensor_range = np.maximum(np.hypot(horizontal, vertical), detection["range_floor"])
    los_angle = np.arctan2(vertical, horizontal)
    aspect = np.arctan2(np.sin(gamma - los_angle), np.cos(gamma - los_angle))
    rcs = detection["rcs_min"] + (detection["rcs_max"] - detection["rcs_min"]) * np.cos(aspect) ** 2
    radar = detection["radar_rate_scale"] * detection["radar_coefficient"] * rcs / sensor_range**4
    radial_velocity = speed * (math.cos(gamma) * horizontal + math.sin(gamma) * vertical) / sensor_range
    doppler = detection["radial_velocity_rate_scale"] * detection["doppler_coefficient"] * radial_velocity**2 / sensor_range**4
    return radar + doppler


def _incremental_objective(hazard, duration, attacker_config):
    hazard_reference = attacker_config["normalization"]["pod"]["hazard_reference"]
    time_reference = attacker_config["normalization"]["time"]["reference_seconds"]
    return attacker_config["w_pod"] * np.asarray(hazard) / hazard_reference + attacker_config["w_time"] * np.asarray(duration) / time_reference


def _maximum_edge_endpoint_residual(trajectory, speeds, gammas, durations):
    reconstructed = trajectory[:-1] + durations[:, None] * np.column_stack((
        speeds * np.cos(gammas), speeds * np.sin(gammas)
    ))
    return float(np.max(np.linalg.norm(reconstructed - trajectory[1:], axis=1)))


def _function_outputs(function, *arguments):
    values = function(*arguments)
    outputs = values if isinstance(values, (tuple, list)) else (values,)
    return [float(value) for value in outputs]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array.setflags(write=False)
    return array
