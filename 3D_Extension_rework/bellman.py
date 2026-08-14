"""Exact physical-successor Bellman cost-to-go for the 3D rework.

The dense J6D bundle establishes the local detection/objective/feasibility
contract.  As in the completed 2D physical-successor solver, Bellman edges
connect grid nodes exactly and integrate that same local objective along the
physical segment; endpoints are never rounded or reset.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .geometry import terrain_height


def _signed_heading_change(target: Any, source: Any) -> np.ndarray:
    return np.arctan2(np.sin(np.asarray(target) - np.asarray(source)),
                      np.cos(np.asarray(target) - np.asarray(source)))


def _control_is_valid(speed: float, gamma: float, vehicle: dict[str, Any]) -> bool:
    lift = (
        2.0 * vehicle["mass_kg"] * vehicle["gravity_mps2"] * math.cos(gamma)
        / (vehicle["air_density_kgpm3"] * speed**2 * vehicle["wing_area_m2"])
    )
    drag = (
        vehicle["cd0"] + vehicle["linear_drag_coefficient"] * lift
        + vehicle["quadratic_drag_coefficient"] * lift**2
    )
    return bool(
        vehicle["glide_speed_min_mps"] <= speed <= vehicle["glide_speed_max_mps"]
        and vehicle["cl_min"] <= lift <= vehicle["cl_max"]
        and drag > 0.0 and np.isfinite(lift) and np.isfinite(drag)
    )


def _incremental_objective(
    hazard: np.ndarray, duration: np.ndarray, attacker: dict[str, Any],
) -> np.ndarray:
    return (
        attacker["w_pod"] * hazard / attacker["hazard_reference"]
        + attacker["w_time"] * duration / attacker["time_reference_s"]
    )


def build_physical_successor_graph(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Build exact node-to-node glide edges and integrate their local cost."""
    if not stage_cost_bundle.get("status", {}).get("success", False):
        raise ValueError("stage cost must pass validation")
    if stage_cost_bundle["metadata"].get("is_cost_to_go", True):
        raise ValueError("Bellman requires local J6D, not a projected value map")
    grids = stage_cost_bundle["grids"]
    x_grid, y_grid, h_grid, speed_grid = (
        grids[name] for name in ("x", "y", "h", "v")
    )
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dh = float(h_grid[1] - h_grid[0])
    options = configuration["bellman"]
    vehicle = configuration["vehicle"]
    actions: list[dict[str, Any]] = []
    for forward in range(1, int(options["maximum_forward_cells"]) + 1):
        for lateral in range(
            -int(options["maximum_lateral_cells"]),
            int(options["maximum_lateral_cells"]) + 1,
        ):
            for descent in range(1, int(options["maximum_descent_cells"]) + 1):
                edge = np.array([
                    forward * dx, lateral * dy, -descent * dh,
                ], dtype=float)
                horizontal = float(np.hypot(edge[0], edge[1]))
                gamma = math.atan2(float(edge[2]), horizontal)
                heading = math.atan2(float(edge[1]), float(edge[0]))
                length = float(np.linalg.norm(edge))
                for speed_index, speed_value in enumerate(speed_grid):
                    speed = float(speed_value)
                    if _control_is_valid(speed, gamma, vehicle):
                        actions.append({
                            "forward_cells": forward,
                            "lateral_cells": lateral,
                            "descent_cells": descent,
                            "speed_index": speed_index,
                            "speed_mps": speed,
                            "gamma_rad": gamma,
                            "heading_rad": heading,
                            "edge_m": edge,
                            "length_m": length,
                            "duration_s": length / speed,
                        })
    if not actions:
        raise RuntimeError("physical successor configuration creates no actions")

    heading_states = np.asarray(sorted({
        round(float(action["heading_rad"]), 14) for action in actions
    }), dtype=float)
    for action in actions:
        action["heading_state_index"] = int(np.argmin(np.abs(
            _signed_heading_change(heading_states, action["heading_rad"])
        )))

    spatial_shape = (x_grid.size, y_grid.size, h_grid.size)
    graph_shape = (*spatial_shape, len(actions))
    valid = np.zeros(graph_shape, dtype=bool)
    terminal = np.zeros(graph_shape, dtype=bool)
    terminal_fraction = np.ones(graph_shape, dtype=float)
    hazard = np.full(graph_shape, np.nan, dtype=np.float32)
    cost = np.full(graph_shape, np.inf, dtype=float)
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        x_grid, y_grid, h_grid, indexing="ij",
    )
    current_valid = np.asarray(
        stage_cost_bundle["validity_masks"]["spatial_glide_valid"], dtype=bool,
    )
    goal = np.asarray(geometry["goal_position"], dtype=float)
    goal_radius = float(options["goal_radius_m"])
    quadrature = np.linspace(0.0, 1.0, int(options["edge_quadrature_count"]))
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    spatial_size = int(np.prod(spatial_shape))
    glide_map = detection_bundle["functions"][
        "glide_detection_components"
    ].map(spatial_size * quadrature.size)
    los_interpolator = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]),
        geometry["los_boundary_height"], method="linear",
        bounds_error=False, fill_value=np.inf,
    )
    terrain_tolerance = float(configuration["validation"]["terrain_tolerance_m"])
    clearance = float(options["terrain_clearance_m"])

    for action_index, action in enumerate(actions):
        edge = action["edge_m"]
        forward = action["forward_cells"]
        lateral = action["lateral_cells"]
        descent = action["descent_cells"]
        successor_in_bounds = np.zeros(spatial_shape, dtype=bool)
        source_x = slice(0, x_grid.size - forward)
        target_x = slice(forward, x_grid.size)
        if lateral >= 0:
            source_y = slice(0, y_grid.size - lateral) if lateral else slice(None)
            target_y = slice(lateral, y_grid.size) if lateral else slice(None)
        else:
            source_y = slice(-lateral, y_grid.size)
            target_y = slice(0, y_grid.size + lateral)
        source_h = slice(descent, h_grid.size)
        target_h = slice(0, h_grid.size - descent)
        successor_in_bounds[source_x, source_y, source_h] = True
        successor_valid = np.zeros(spatial_shape, dtype=bool)
        successor_valid[source_x, source_y, source_h] = current_valid[
            target_x, target_y, target_h
        ]

        relative = np.stack(
            (mesh_x - goal[0], mesh_y - goal[1], mesh_h - goal[2]), axis=-1,
        )
        quadratic_a = float(np.dot(edge, edge))
        quadratic_b = np.sum(relative * edge, axis=-1)
        quadratic_c = np.sum(relative**2, axis=-1) - goal_radius**2
        discriminant = quadratic_b**2 - quadratic_a * quadratic_c
        intersection = (
            -quadratic_b - np.sqrt(np.maximum(discriminant, 0.0))
        ) / quadratic_a
        action_terminal = (
            (quadratic_c > 0.0) & (discriminant >= 0.0)
            & (intersection > 0.0) & (intersection <= 1.0)
        )
        fraction_map = np.where(
            action_terminal, np.clip(intersection, 0.0, 1.0), 1.0,
        )

        segment_valid = np.ones(spatial_shape, dtype=bool)
        sample_x_values: list[np.ndarray] = []
        sample_y_values: list[np.ndarray] = []
        sample_h_values: list[np.ndarray] = []
        for fraction in quadrature:
            effective = fraction * fraction_map
            sample_x = mesh_x + effective * edge[0]
            sample_y = mesh_y + effective * edge[1]
            sample_h = mesh_h + effective * edge[2]
            in_domain = (
                (sample_x >= x_grid[0]) & (sample_x <= x_grid[-1])
                & (sample_y >= y_grid[0]) & (sample_y <= y_grid[-1])
                & (sample_h >= h_grid[0]) & (sample_h <= h_grid[-1])
            )
            clipped_x = np.clip(sample_x, x_grid[0], x_grid[-1])
            clipped_y = np.clip(sample_y, y_grid[0], y_grid[-1])
            terrain_clear = (
                sample_h >= terrain_height(
                    geometry["terrain_model"], clipped_x, clipped_y,
                ) + clearance - terrain_tolerance
            )
            boundary = los_interpolator(np.column_stack((
                clipped_x.ravel(), clipped_y.ravel(),
            ))).reshape(spatial_shape)
            los_visible = sample_h >= boundary - terrain_tolerance
            segment_valid &= in_domain & terrain_clear & los_visible
            sample_x_values.append(sample_x)
            sample_y_values.append(sample_y)
            sample_h_values.append(sample_h)

        stacked_x = np.stack(sample_x_values).reshape(1, -1)
        stacked_y = np.stack(sample_y_values).reshape(1, -1)
        stacked_h = np.stack(sample_h_values).reshape(1, -1)
        sample_count = stacked_x.size
        outputs = glide_map(
            stacked_x, stacked_y, stacked_h,
            np.full((1, sample_count), action["speed_mps"]),
            np.full((1, sample_count), action["gamma_rad"]),
            np.full((1, sample_count), action["heading_rad"]),
            np.full((1, sample_count), sensor[0]),
            np.full((1, sample_count), sensor[1]),
            np.full((1, sample_count), sensor[2]),
        )
        output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
        rate = np.asarray(output_tuple[-1], dtype=float).reshape(
            quadrature.size, *spatial_shape,
        )
        duration = action["duration_s"] * fraction_map
        action_hazard = np.trapezoid(rate, quadrature, axis=0) * duration
        action_cost = _incremental_objective(
            action_hazard, duration, configuration["cost"]["attacker"],
        )
        action_valid = current_valid & segment_valid & (
            action_terminal | (successor_in_bounds & successor_valid)
        )
        valid[..., action_index] = action_valid
        terminal[..., action_index] = action_valid & action_terminal
        terminal_fraction[..., action_index] = np.where(
            action_valid & action_terminal, fraction_map, 1.0,
        )
        hazard[..., action_index] = np.where(
            action_valid, action_hazard, np.nan,
        ).astype(np.float32)
        cost[..., action_index] = np.where(action_valid, action_cost, np.inf)

    validation = _validate_graph(
        valid, terminal, terminal_fraction, cost, actions, current_valid,
    )
    return {
        "actions": tuple(actions),
        "heading_states": _readonly(heading_states),
        "valid": _readonly(valid),
        "terminal": _readonly(terminal),
        "terminal_fraction": _readonly(terminal_fraction),
        "hazard": _readonly(hazard),
        "cost": _readonly(cost),
        "grids": grids,
        "metadata": {
            "transition_model": "exact_physical_successor_grid",
            "endpoint_snapping": False,
            "edge_quadrature_count": int(quadrature.size),
            "action_count": len(actions),
            "heading_state_count": int(heading_states.size),
            "state_action_count": int(np.prod(graph_shape)),
            "local_objective_contract": "StageCost6DResult",
            "segment_feasibility": "terrain_clear_and_los_visible",
        },
        "validation": validation,
        "status": {
            "success": validation["passed"],
            "message": validation["summary"],
        },
    }


def _validate_graph(
    valid: np.ndarray,
    terminal: np.ndarray,
    terminal_fraction: np.ndarray,
    cost: np.ndarray,
    actions: list[dict[str, Any]],
    spatial_valid: np.ndarray,
) -> dict[str, Any]:
    source_valid = np.broadcast_to(spatial_valid[..., None], valid.shape)
    checks = {
        "actions_exist": bool(actions),
        "strict_altitude_descent": bool(all(
            action["descent_cells"] > 0 for action in actions
        )),
        "exact_node_offsets": bool(all(
            action["forward_cells"] >= 1 for action in actions
        )),
        "valid_edges_start_in_glide_domain": bool(np.all(~valid | source_valid)),
        "finite_valid_cost": bool(np.all(np.isfinite(cost[valid]))),
        "infinite_invalid_cost": bool(np.all(np.isposinf(cost[~valid]))),
        "terminal_edges_exist": bool(np.any(terminal)),
        "terminal_fractions_are_physical": bool(
            np.all((terminal_fraction[terminal] > 0.0)
                   & (terminal_fraction[terminal] <= 1.0))
            and np.all(terminal_fraction[~terminal] == 1.0)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "valid_edge_count": int(np.count_nonzero(valid)),
            "terminal_edge_count": int(np.count_nonzero(terminal)),
            "action_count": len(actions),
        },
        "summary": (
            "physical successor graph validation passed"
            if not failed else f"physical successor graph failed: {failed}"
        ),
    }


def solve_cost_to_go(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Solve the heading-state DAG by one altitude-ascending Bellman sweep."""
    if not graph.get("status", {}).get("success", False):
        raise ValueError("successor graph must pass validation")
    grids = graph["grids"]
    x_grid, y_grid, h_grid = grids["x"], grids["y"], grids["h"]
    heading_states = graph["heading_states"]
    state_shape = (x_grid.size, y_grid.size, h_grid.size, heading_states.size)
    value = np.full(state_shape, np.inf)
    hazard_to_go = np.full(state_shape, np.nan)
    policy_action = np.full(state_shape, -1, dtype=np.int32)
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        x_grid, y_grid, h_grid, indexing="ij",
    )
    goal = np.asarray(geometry["goal_position"], dtype=float)
    goal_mask = (
        (mesh_x - goal[0])**2 + (mesh_y - goal[1])**2
        + (mesh_h - goal[2])**2
        <= float(configuration["bellman"]["goal_radius_m"])**2
    )
    value[goal_mask, :] = 0.0
    hazard_to_go[goal_mask, :] = 0.0
    maximum_turn_rate = np.deg2rad(
        configuration["vehicle"]["max_turn_rate_deg_s"]
    )
    updated_state_count = 0

    for h_index in range(h_grid.size):
        for x_index in range(x_grid.size - 1, -1, -1):
            for y_index in range(y_grid.size):
                if goal_mask[x_index, y_index, h_index]:
                    continue
                for action_index, action in enumerate(graph["actions"]):
                    if not graph["valid"][x_index, y_index, h_index, action_index]:
                        continue
                    terminal = bool(graph["terminal"][
                        x_index, y_index, h_index, action_index
                    ])
                    next_heading_index = int(action["heading_state_index"])
                    if terminal:
                        downstream_cost = 0.0
                        downstream_hazard = 0.0
                    else:
                        next_x = x_index + int(action["forward_cells"])
                        next_y = y_index + int(action["lateral_cells"])
                        next_h = h_index - int(action["descent_cells"])
                        downstream_cost = value[
                            next_x, next_y, next_h, next_heading_index
                        ]
                        downstream_hazard = hazard_to_go[
                            next_x, next_y, next_h, next_heading_index
                        ]
                        if not np.isfinite(downstream_cost):
                            continue
                    heading_change = np.abs(_signed_heading_change(
                        action["heading_rad"], heading_states,
                    ))
                    compatible = heading_change <= (
                        maximum_turn_rate * action["duration_s"] + 1.0e-12
                    )
                    local_cost = float(graph["cost"][
                        x_index, y_index, h_index, action_index
                    ])
                    local_hazard = float(graph["hazard"][
                        x_index, y_index, h_index, action_index
                    ])
                    candidate = local_cost + downstream_cost
                    improve = compatible & (candidate < value[
                        x_index, y_index, h_index, :
                    ])
                    if np.any(improve):
                        value[x_index, y_index, h_index, improve] = candidate
                        hazard_to_go[x_index, y_index, h_index, improve] = (
                            local_hazard + downstream_hazard
                        )
                        policy_action[x_index, y_index, h_index, improve] = action_index
                        updated_state_count += int(np.count_nonzero(improve))

    validation = _validate_policy(
        value, hazard_to_go, policy_action, goal_mask, graph,
    )
    return {
        "value_heading_state": _readonly(value),
        "hazard_to_go_heading_state": _readonly(hazard_to_go),
        "pod_to_go_heading_state": _readonly(1.0 - np.exp(-hazard_to_go)),
        "policy_action_index": _readonly(policy_action),
        "goal_mask": _readonly(goal_mask),
        "heading_states": heading_states,
        "metadata": {
            "state_axis_order": ("x", "y", "h", "psi_in"),
            "source_local_cost_axes": ("x", "y", "h", "v", "gamma", "psi_action"),
            "sweep_axis": "h_ascending",
            "acyclic": True,
            "updated_state_count": updated_state_count,
            "is_cost_to_go": True,
        },
        "validation": validation,
        "status": {
            "success": validation["passed"],
            "message": validation["summary"],
        },
    }


def _validate_policy(
    value: np.ndarray,
    hazard: np.ndarray,
    policy_action: np.ndarray,
    goal_mask: np.ndarray,
    graph: dict[str, Any],
) -> dict[str, Any]:
    finite = np.isfinite(value)
    finite_non_goal = finite & ~goal_mask[..., None]
    checks = {
        "goal_value_zero": bool(np.all(value[goal_mask, :] == 0.0)),
        "finite_states_exist": bool(np.any(finite_non_goal)),
        "finite_states_have_policy": bool(np.all(
            policy_action[finite_non_goal] >= 0
        )),
        "unreachable_states_have_no_policy": bool(np.all(
            policy_action[~finite] == -1
        )),
        "finite_hazard_for_finite_value": bool(np.all(np.isfinite(
            hazard[finite]
        ))),
        "acyclic_graph": bool(all(
            action["descent_cells"] > 0 for action in graph["actions"]
        )),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "finite_heading_state_count": int(np.count_nonzero(finite)),
            "finite_non_goal_state_count": int(np.count_nonzero(finite_non_goal)),
            "total_heading_state_count": int(value.size),
        },
        "summary": (
            "Bellman cost-to-go validation passed"
            if not failed else f"Bellman cost-to-go failed: {failed}"
        ),
    }


def project_cost_to_go_3d(policy: dict[str, Any]) -> dict[str, Any]:
    """Minimize the Bellman value over incoming heading only."""
    if not policy.get("status", {}).get("success", False):
        raise ValueError("Bellman policy must pass validation")
    value = np.asarray(policy["value_heading_state"])
    finite = np.isfinite(value)
    projection_mask = np.any(finite, axis=3)
    safe_value = np.where(finite, value, np.inf)
    optimal_heading_index = np.argmin(safe_value, axis=3).astype(np.int32)
    projected_cost = np.min(safe_value, axis=3)
    projected_pod = np.full(projected_cost.shape, np.nan)
    projected_heading = np.full(projected_cost.shape, np.nan)
    spatial_indices = np.nonzero(projection_mask)
    selected_heading = optimal_heading_index[projection_mask]
    projected_pod[projection_mask] = policy["pod_to_go_heading_state"][
        spatial_indices[0], spatial_indices[1], spatial_indices[2], selected_heading
    ]
    projected_heading[projection_mask] = policy["heading_states"][selected_heading]
    optimal_heading_index[~projection_mask] = -1
    validation = {
        "passed": bool(
            np.all(np.isfinite(projected_cost[projection_mask]))
            and np.all(np.isposinf(projected_cost[~projection_mask]))
            and np.allclose(
                projected_cost[projection_mask],
                value[
                    spatial_indices[0], spatial_indices[1], spatial_indices[2],
                    selected_heading,
                ],
                rtol=0.0, atol=0.0,
            )
        ),
        "minimum_selection_residual": 0.0,
    }
    return {
        "projected_cost_to_go": _readonly(projected_cost),
        "projected_pod_to_go": _readonly(projected_pod),
        "optimal_incoming_heading": _readonly(projected_heading),
        "optimal_incoming_heading_index": _readonly(optimal_heading_index),
        "projection_mask": _readonly(projection_mask),
        "metadata": {
            "source_axis_order": ("x", "y", "h", "psi_in"),
            "projected_axis_order": ("x", "y", "h"),
            "projection_rule": "minimum_bellman_cost_over_incoming_heading",
            "pod_projection_rule": "PoD_along_cost_minimizing_heading_policy",
            "is_cost_to_go": True,
            "local_stage_cost_projection": False,
        },
        "validation": validation,
        "status": {"success": validation["passed"]},
    }


def build_cost_to_go_bundle(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_bundle: dict[str, Any],
) -> dict[str, Any]:
    graph = build_physical_successor_graph(
        configuration, geometry, detection_bundle, stage_cost_bundle,
    )
    policy = solve_cost_to_go(configuration, geometry, graph)
    projection = project_cost_to_go_3d(policy)
    passed = bool(
        graph["status"]["success"]
        and policy["status"]["success"]
        and projection["status"]["success"]
    )
    return {
        "graph": graph,
        "policy": policy,
        "projection": projection,
        "metadata": {
            "process_chain": (
                "J6D local objective contract",
                "exact physical successor Bellman V4D(x,y,h,psi_in)",
                "projected cost-to-go V3D(x,y,h)",
            ),
            "switching_candidates_evaluated": False,
            "trajectory_extracted": False,
        },
        "status": {
            "success": passed,
            "message": "3D projected Bellman cost-to-go completed" if passed else "failed",
        },
    }


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
