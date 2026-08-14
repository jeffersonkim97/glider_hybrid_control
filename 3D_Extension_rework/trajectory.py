"""Extract the authoritative physical Bellman trajectory in 3D.

The implementation mirrors ``p1b_4D.successor_grid_solver._extract_candidate``:
the continuous switching state and its virtual edge are preserved, then the
stored Bellman policy is followed until the physical terminal edge intersects
the goal ball.  No NLP, smoothing, interpolation of controls, or endpoint
snapping occurs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .bellman import _incremental_objective, _signed_heading_change
from .switching import _certify_segment, _readonly


def extract_optimal_trajectory(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    cost_to_go_bundle: dict[str, Any],
    switching_result: dict[str, Any],
) -> dict[str, Any]:
    """Follow the selected virtual edge and heading-state Bellman policy."""
    if not cost_to_go_bundle.get("status", {}).get("success", False):
        raise ValueError("cost-to-go bundle must pass validation")
    if not switching_result.get("status", {}).get("success", False):
        raise ValueError("switching result must pass validation")
    graph = cost_to_go_bundle["graph"]
    policy = cost_to_go_bundle["policy"]
    best = switching_result["best"]
    connection = best["connection"]
    switching_point = np.asarray(best["switching_point"], dtype=float)
    entry_point = np.asarray(connection["target"], dtype=float)
    entry_index = tuple(int(value) for value in connection["target_index"])

    trajectory = [switching_point, entry_point]
    speeds = [float(connection["speed_mps"])]
    gammas = [float(connection["edge"]["gamma_rad"])]
    headings = [float(connection["edge"]["heading_rad"])]
    durations = [float(connection["edge"]["duration_s"])]
    hazards = [float(connection["edge"]["hazard"])]
    costs = [float(connection["edge"]["cost"])]
    fractions = [1.0]
    action_indices = [-1]
    edge_kinds = ["virtual_switch_edge"]
    node_indices = [entry_index]

    current_index = entry_index
    action_index = int(connection["downstream"]["action_index"])
    incoming_heading_index: int | None = None
    reached_goal = bool(policy["goal_mask"][current_index])
    policy_consistent = True
    maximum_steps = int(configuration["bellman"]["maximum_path_steps"])

    for step in range(maximum_steps):
        if reached_goal:
            break
        if action_index < 0:
            raise RuntimeError("Bellman trajectory encountered a missing action")
        if step > 0:
            expected = int(policy["policy_action_index"][
                current_index[0], current_index[1], current_index[2],
                incoming_heading_index,
            ])
            policy_consistent &= action_index == expected
        action = graph["actions"][action_index]
        start = np.array([
            graph["grids"]["x"][current_index[0]],
            graph["grids"]["y"][current_index[1]],
            graph["grids"]["h"][current_index[2]],
        ], dtype=float)
        fraction = float(graph["terminal_fraction"][
            current_index[0], current_index[1], current_index[2], action_index,
        ])
        end = start + fraction * np.asarray(action["edge_m"], dtype=float)
        trajectory.append(end)
        speeds.append(float(action["speed_mps"]))
        gammas.append(float(action["gamma_rad"]))
        headings.append(float(action["heading_rad"]))
        durations.append(float(action["duration_s"]) * fraction)
        hazards.append(float(graph["hazard"][
            current_index[0], current_index[1], current_index[2], action_index,
        ]))
        costs.append(float(graph["cost"][
            current_index[0], current_index[1], current_index[2], action_index,
        ]))
        fractions.append(fraction)
        action_indices.append(action_index)
        edge_kinds.append("bellman_terminal_edge" if bool(graph["terminal"][
            current_index[0], current_index[1], current_index[2], action_index,
        ]) else "bellman_grid_edge")
        if bool(graph["terminal"][
            current_index[0], current_index[1], current_index[2], action_index,
        ]):
            reached_goal = True
            break

        current_index = (
            current_index[0] + int(action["forward_cells"]),
            current_index[1] + int(action["lateral_cells"]),
            current_index[2] - int(action["descent_cells"]),
        )
        node_indices.append(current_index)
        incoming_heading_index = int(action["heading_state_index"])
        action_index = int(policy["policy_action_index"][
            current_index[0], current_index[1], current_index[2],
            incoming_heading_index,
        ])
    else:
        raise RuntimeError("Bellman trajectory exceeded maximum_path_steps")

    trajectory_array = np.asarray(trajectory, dtype=float)
    speed_array = np.asarray(speeds, dtype=float)
    gamma_array = np.asarray(gammas, dtype=float)
    heading_array = np.asarray(headings, dtype=float)
    duration_array = np.asarray(durations, dtype=float)
    hazard_array = np.asarray(hazards, dtype=float)
    cost_array = np.asarray(costs, dtype=float)
    fraction_array = np.asarray(fractions, dtype=float)
    goal = np.asarray(geometry["goal_position"], dtype=float)
    goal_distance = float(np.linalg.norm(trajectory_array[-1] - goal))

    certificates = tuple(
        _certify_segment(
            trajectory_array[index], trajectory_array[index + 1],
            configuration, geometry, los_requirement="visible",
            sample_count=int(configuration["bellman"]["virtual_edge_quadrature_count"]),
        )
        for index in range(trajectory_array.shape[0] - 1)
    )
    maximum_turn_rate = math.radians(
        float(configuration["vehicle"]["max_turn_rate_deg_s"])
    )
    previous_headings = np.concatenate((
        np.asarray([best["powered"]["heading_rad"]]), heading_array[:-1],
    ))
    turn_residual = np.abs(_signed_heading_change(
        heading_array, previous_headings,
    )) - maximum_turn_rate * duration_array

    powered = best["powered"]
    glide_hazard = float(np.sum(hazard_array))
    glide_time = float(np.sum(duration_array))
    mission_hazard = float(powered["hazard"] + glide_hazard)
    mission_time = float(powered["duration_s"] + glide_time)
    mission_cost = float(_incremental_objective(
        np.asarray(mission_hazard), np.asarray(mission_time),
        configuration["cost"]["attacker"],
    ))
    additive_cost = float(powered["cost"] + np.sum(cost_array))
    objective_residual = abs(mission_cost - best["mission_cost"])
    downstream_cost_residual = abs(
        float(np.sum(cost_array[1:])) - float(connection["downstream"]["cost"])
    )
    downstream_hazard_residual = abs(
        float(np.sum(hazard_array[1:])) - float(connection["downstream"]["hazard"])
    )
    goal_radius = float(configuration["bellman"]["goal_radius_m"])
    checks = {
        "goal_reached": bool(reached_goal and goal_distance <= goal_radius + 1.0e-9),
        "strictly_forward_x": bool(np.all(np.diff(trajectory_array[:, 0]) > 0.0)),
        "all_glide_segments_terrain_clear": bool(all(
            certificate["terrain_clear"] for certificate in certificates
        )),
        "all_glide_segments_los_visible": bool(all(
            certificate["los_clear"] for certificate in certificates
        )),
        "all_glide_segments_in_domain": bool(all(
            certificate["domain_clear"] for certificate in certificates
        )),
        "turn_rate_feasible": bool(np.max(turn_residual) <= 1.0e-12),
        "stored_policy_followed": bool(policy_consistent),
        "mission_objective_reconstructed": objective_residual <= 1.0e-10,
        "additive_edge_cost_consistent": abs(additive_cost - mission_cost) <= 1.0e-10,
        "downstream_bellman_cost_reconstructed": downstream_cost_residual <= 1.0e-10,
        "downstream_hazard_reconstructed": downstream_hazard_residual <= 1.0e-7,
        "endpoint_snapping_absent": True,
        "continuous_nlp_absent": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    full_path = np.vstack((np.asarray(powered["path"]), trajectory_array[1:]))
    return {
        "switching_point": _readonly(switching_point),
        "bellman_entry_point": _readonly(entry_point),
        "powered_path": powered["path"],
        "glide_trajectory": _readonly(trajectory_array),
        "full_path": _readonly(full_path),
        "speed_profile_mps": _readonly(speed_array),
        "gamma_profile_rad": _readonly(gamma_array),
        "heading_profile_rad": _readonly(heading_array),
        "duration_profile_s": _readonly(duration_array),
        "hazard_profile": _readonly(hazard_array),
        "cost_profile": _readonly(cost_array),
        "terminal_fraction_profile": _readonly(fraction_array),
        "action_index_profile": _readonly(np.asarray(action_indices, dtype=np.int32)),
        "edge_kind_profile": tuple(edge_kinds),
        "bellman_node_indices": tuple(node_indices),
        "mission": {
            "cost": mission_cost,
            "hazard": mission_hazard,
            "pod": float(1.0 - math.exp(-mission_hazard)),
            "time_s": mission_time,
            "powered_time_s": float(powered["duration_s"]),
            "glide_time_s": glide_time,
        },
        "metadata": {
            "source_process": "p1b_4D successor_grid _extract_candidate",
            "trajectory_type": "physical virtual edge plus heading-state Bellman policy",
            "endpoint_snapping": False,
            "continuous_nlp_applied": False,
            "continuous_replay_applied": False,
        },
        "validation": {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "metrics": {
                "glide_edge_count": int(duration_array.size),
                "bellman_edge_count": int(duration_array.size - 1),
                "glide_node_count": int(trajectory_array.shape[0]),
                "goal_distance_m": goal_distance,
                "minimum_terrain_margin_m": float(min(
                    certificate["minimum_terrain_margin_m"] for certificate in certificates
                )),
                "minimum_los_margin_m": float(min(
                    certificate["minimum_los_margin_m"] for certificate in certificates
                )),
                "maximum_turn_rate_residual_rad": float(np.max(turn_residual)),
                "objective_residual": objective_residual,
                "downstream_cost_residual": downstream_cost_residual,
            },
        },
        "status": {
            "success": not failed,
            "message": "physical Bellman trajectory extraction passed" if not failed else f"failed: {failed}",
        },
    }
