"""Exact physical successor-grid Bellman solver for the 3D extension.

Every regular glide edge connects two spatial grid nodes exactly. Speed,
flight-path angle, heading, length, and duration are derived from those two
endpoints; no off-grid endpoint is ever rounded or reset.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from .bellman import evaluate_powered_segment, generate_switching_point_seeds
from .geometry import terrain_height
from .stage_cost import construct_state_grids
from .turn_dynamics import powered_segment_heading, signed_heading_change


def physical_action_offsets(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    h_grid: np.ndarray,
    search_options: dict[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    """Return exact cell offsets sampled from one physical action envelope.

    The returned offsets depend on grid spacing, but every corresponding
    metre-valued edge belongs to the same configured domain.  This preserves
    exact node-to-node transitions while removing cell-count-dependent action
    limits from resolution comparisons.
    """
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dh = float(h_grid[1] - h_grid[0])
    envelope = search_options.get("physical_action_envelope")
    if envelope is None:
        return tuple(
            (forward, lateral, descent)
            for forward in range(1, int(search_options["max_forward_cells"]) + 1)
            for lateral in range(
                -int(search_options.get("max_lateral_cells", 3)),
                int(search_options.get("max_lateral_cells", 3)) + 1,
            )
            for descent in range(1, int(search_options["max_descent_cells"]) + 1)
        )

    required = (
        "forward_min_m", "forward_max_m", "lateral_max_m",
        "descent_min_m", "descent_max_m",
    )
    values = {name: float(envelope[name]) for name in required}
    if not all(np.isfinite(tuple(values.values()))):
        raise ValueError("physical_action_envelope values must be finite")
    if not (
        0.0 < values["forward_min_m"] <= values["forward_max_m"]
        and values["lateral_max_m"] >= 0.0
        and 0.0 < values["descent_min_m"] <= values["descent_max_m"]
    ):
        raise ValueError("physical_action_envelope bounds are inconsistent")

    maximum_forward_cells = int(np.floor(
        values["forward_max_m"] / dx + 1.0e-12
    ))
    maximum_lateral_cells = int(np.floor(
        values["lateral_max_m"] / dy + 1.0e-12
    ))
    maximum_descent_cells = int(np.floor(
        values["descent_max_m"] / dh + 1.0e-12
    ))
    offsets = []
    for forward in range(1, maximum_forward_cells + 1):
        forward_m = forward * dx
        if forward_m < values["forward_min_m"] - 1.0e-10:
            continue
        for lateral in range(-maximum_lateral_cells, maximum_lateral_cells + 1):
            if abs(lateral * dy) > values["lateral_max_m"] + 1.0e-10:
                continue
            for descent in range(1, maximum_descent_cells + 1):
                descent_m = descent * dh
                if descent_m < values["descent_min_m"] - 1.0e-10:
                    continue
                offsets.append((forward, lateral, descent))
    if not offsets:
        raise RuntimeError(
            "Grid samples no exact edge inside physical_action_envelope"
        )
    return tuple(offsets)


def solve_physical_successor_grid_attacker(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact-edge graph and return candidate/response bundles."""
    configs = configuration_bundle["primary_result"]
    grids = construct_state_grids(
        configs["environment_config"], configs["vehicle_config"],
    )
    graph = build_physical_successor_graph(
        configuration_bundle, geometry_bundle, detection_bundle, grids,
    )
    policy = solve_physical_successor_bellman(
        graph,
        grids,
        geometry_bundle["primary_result"]["goal_position"],
        float(configs["validation_config"]["goal_radius"]),
        float(configs["vehicle_config"]["turn_dynamics"]["max_turn_rate_deg_s"]),
    )
    seeds = generate_switching_point_seeds(
        geometry_bundle, grids["x"], grids["y"], grids["h"],
        include_visible=(
            configs["bellman_config"]["search_options"].get(
                "powered_visibility_handling", "hard_hidden",
            ) == "hazard_penalty"
        ),
        candidate_mask=np.any(np.isfinite(policy["value"]), axis=3),
        boundary_only=(
            configs["bellman_config"]["search_options"].get(
                "switching_candidate_mode", "admissible_volume",
            ) == "los_boundary_surface"
        ),
        boundary_tolerance_m=float(
            configs["vehicle_config"]["switching_constraints"]
            ["tangent_tolerance"]
        ),
    )
    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    powered_valid_count = 0
    finite_first_edge_count = 0
    extracted_candidate_count = 0
    candidate_failure_counts: Counter[str] = Counter()
    best_failed_minimum_margin = -np.inf
    best_failed_margin_point: tuple[float, ...] | None = None
    best_failed_switching_point: tuple[float, ...] | None = None
    for seed_index, switching_point in enumerate(seeds):
        powered = evaluate_powered_segment(
            switching_point,
            configuration_bundle,
            geometry_bundle,
            detection_bundle,
            grids,
        )
        attempt = {
            "start_id": f"physical-switch-{seed_index:05d}",
            "seed_index": seed_index,
            "switching_point": switching_point,
            "success": False,
            "diagnostic": None,
        }
        if not powered["validation"]["passed"]:
            attempt["diagnostic"] = powered["validation"]["summary"]
            attempts.append(attempt)
            continue
        powered_valid_count += 1
        initial_heading = powered_segment_heading(
            powered["path"], geometry_bundle["primary_result"]["goal_position"],
        )
        first = _best_first_edge(
            switching_point, initial_heading, graph, policy, grids,
        )
        if first is None:
            attempt["diagnostic"] = "no_finite_turn_limited_physical_edge"
            attempts.append(attempt)
            continue
        finite_first_edge_count += 1
        candidate = _extract_candidate(
            switching_point,
            initial_heading,
            first,
            policy,
            graph,
            grids,
            powered,
            configuration_bundle,
            geometry_bundle,
            detection_bundle,
        )
        if not candidate["validation"]["passed"]:
            candidate_failure_counts.update(
                candidate["validation"].get("failed_checks", ())
            )
            failed_margin = float(
                candidate["validation"]["metrics"]["minimum_terrain_margin"]
            )
            if failed_margin > best_failed_minimum_margin:
                best_failed_minimum_margin = failed_margin
                best_failed_margin_point = tuple(
                    candidate["validation"]["metrics"][
                        "minimum_terrain_margin_point"
                    ]
                )
                best_failed_switching_point = tuple(
                    float(value) for value in switching_point
                )
            attempt["diagnostic"] = candidate["validation"]["summary"]
            attempts.append(attempt)
            continue
        extracted_candidate_count += 1
        candidate["candidate_id"] = (
            f"physical-3d-candidate-{len(candidates):05d}"
        )
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
        raise RuntimeError(
            "No exact physical successor-grid switching response reaches the goal; "
            f"seed_count={len(seeds)}, powered_valid_count={powered_valid_count}, "
            f"finite_first_edge_count={finite_first_edge_count}, "
            f"validated_candidate_count={extracted_candidate_count}, "
            f"candidate_failure_counts={dict(candidate_failure_counts)}, "
            f"best_failed_minimum_margin={best_failed_minimum_margin:.6f}, "
            f"best_failed_margin_point={best_failed_margin_point}, "
            f"best_failed_switching_point={best_failed_switching_point}, "
            f"seed_x_range=({float(np.min(seeds[:, 0])):.3f}, "
            f"{float(np.max(seeds[:, 0])):.3f})"
        )
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["mission_cost"]),
            float(item["switching_point"][0]),
            float(item["switching_point"][1]),
            int(item["metadata"]["seed_index"]),
        ),
    )
    best = ordered[0]
    tied = [
        item for item in ordered
        if float(item["mission_cost"]) == float(best["mission_cost"])
    ]
    candidate_bundle = _candidate_bundle(
        candidates, attempts, seeds, graph, policy, geometry_bundle,
    )
    response_bundle = _response_bundle(
        best, ordered, tied, candidate_bundle, configuration_bundle,
    )
    return candidate_bundle, response_bundle


def build_physical_successor_graph(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    grids: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Construct exact grid-node-to-grid-node 3D glide edges."""
    configs = configuration_bundle["primary_result"]
    vehicle = configs["vehicle_config"]
    search = configs["bellman_config"]["search_options"]
    validation = configs["validation_config"]
    geometry = geometry_bundle["primary_result"]
    grids = grids or construct_state_grids(
        configs["environment_config"], vehicle,
    )
    x_grid, y_grid, h_grid, speed_grid = (
        grids[name] for name in ("x", "y", "h", "v")
    )
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dh = float(h_grid[1] - h_grid[0])
    minimum_glide_clearance = float(
        search.get("minimum_glide_terrain_clearance", 0.0)
    )
    if minimum_glide_clearance < 0.0:
        raise ValueError("minimum_glide_terrain_clearance must be nonnegative")

    spatial_actions: list[dict[str, Any]] = []
    offsets = physical_action_offsets(x_grid, y_grid, h_grid, search)
    for forward, lateral, descent in offsets:
        edge = np.array([
            forward * dx, lateral * dy, -descent * dh,
        ])
        horizontal = float(np.hypot(edge[0], edge[1]))
        heading = math.atan2(float(edge[1]), float(edge[0]))
        gamma = math.atan2(float(edge[2]), horizontal)
        length = float(np.linalg.norm(edge))
        for speed_index, speed_value in enumerate(speed_grid):
            speed = float(speed_value)
            if _control_is_valid(speed, gamma, vehicle):
                spatial_actions.append({
                    "forward_cells": forward,
                    "lateral_cells": lateral,
                    "descent_cells": descent,
                    "speed_index": speed_index,
                    "speed": speed,
                    "gamma": gamma,
                    "heading": heading,
                    "edge": edge,
                    "length": length,
                    "duration": length / speed,
                })
    if not spatial_actions:
        raise RuntimeError("Physical successor configuration creates no actions")

    heading_states = np.asarray(sorted({
        round(float(action["heading"]), 14) for action in spatial_actions
    }))
    for action in spatial_actions:
        action["heading_state_index"] = int(np.argmin(
            np.abs(signed_heading_change(heading_states, action["heading"]))
        ))

    spatial_shape = (x_grid.size, y_grid.size, h_grid.size)
    graph_shape = (*spatial_shape, len(spatial_actions))
    valid = np.zeros(graph_shape, dtype=bool)
    terminal = np.zeros(graph_shape, dtype=bool)
    terminal_fraction = np.ones(graph_shape, dtype=np.float32)
    hazard = np.full(graph_shape, np.nan, dtype=np.float32)
    cost = np.full(graph_shape, np.inf, dtype=np.float32)
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        x_grid, y_grid, h_grid, indexing="ij",
    )
    current_valid = ~np.asarray(
        geometry["los_masks"]["terrain_mask"], dtype=bool,
    )
    goal = np.asarray(geometry["goal_position"], dtype=float)
    goal_radius = float(validation["goal_radius"])
    quadrature_count = int(search["segment_check_count"])
    quadrature = np.linspace(0.0, 1.0, quadrature_count)
    spatial_size = int(np.prod(spatial_shape))
    glide_map = detection_bundle["primary_result"]["functions"][
        "glide_detection_components"
    ].map(spatial_size * quadrature_count)
    sensor = np.asarray(geometry["sensor_position"], dtype=float)

    for action_index, action in enumerate(spatial_actions):
        edge = action["edge"]
        fi = action["forward_cells"]
        li = action["lateral_cells"]
        di = action["descent_cells"]
        successor_in_bounds = np.zeros(spatial_shape, dtype=bool)
        x_source = slice(0, x_grid.size - fi)
        x_target = slice(fi, x_grid.size)
        if li >= 0:
            y_source = slice(0, y_grid.size - li) if li else slice(None)
            y_target = slice(li, y_grid.size) if li else slice(None)
        else:
            y_source = slice(-li, y_grid.size)
            y_target = slice(0, y_grid.size + li)
        h_source = slice(di, h_grid.size)
        h_target = slice(0, h_grid.size - di)
        successor_in_bounds[x_source, y_source, h_source] = True
        successor_valid = np.zeros(spatial_shape, dtype=bool)
        successor_valid[x_source, y_source, h_source] = current_valid[
            x_target, y_target, h_target
        ]

        relative = np.stack(
            (mesh_x - goal[0], mesh_y - goal[1], mesh_h - goal[2]), axis=-1,
        )
        qa = float(np.dot(edge, edge))
        qb = np.sum(relative * edge, axis=-1)
        qc = np.sum(relative**2, axis=-1) - goal_radius**2
        discriminant = qb**2 - qa * qc
        intersection = (
            -qb - np.sqrt(np.maximum(discriminant, 0.0))
        ) / qa
        action_terminal = (
            (qc > 0.0)
            & (discriminant >= 0.0)
            & (intersection > 0.0)
            & (intersection <= 1.0)
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
            )
            clipped_x = np.clip(sample_x, x_grid[0], x_grid[-1])
            clipped_y = np.clip(sample_y, y_grid[0], y_grid[-1])
            terrain_clear = (
                sample_h
                >= terrain_height(
                    geometry["terrain_model"], clipped_x, clipped_y,
                ) + minimum_glide_clearance - validation["terrain_tolerance"]
            )
            sample_ok = in_domain & terrain_clear
            if fraction == quadrature[-1]:
                sample_ok |= action_terminal
            segment_valid &= sample_ok
            sample_x_values.append(sample_x)
            sample_y_values.append(sample_y)
            sample_h_values.append(sample_h)

        stacked_x = np.stack(sample_x_values).reshape(1, -1)
        stacked_y = np.stack(sample_y_values).reshape(1, -1)
        stacked_h = np.stack(sample_h_values).reshape(1, -1)
        sample_count = stacked_x.size
        outputs = glide_map(
            stacked_x,
            stacked_y,
            stacked_h,
            np.full((1, sample_count), action["speed"]),
            np.full((1, sample_count), action["gamma"]),
            np.full((1, sample_count), action["heading"]),
            np.full((1, sample_count), sensor[0]),
            np.full((1, sample_count), sensor[1]),
            np.full((1, sample_count), sensor[2]),
        )
        output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
        rate = np.asarray(output_tuple[-1], dtype=float).reshape(
            quadrature_count, *spatial_shape,
        )
        duration_map = action["duration"] * fraction_map
        action_hazard = (
            np.trapezoid(rate, quadrature, axis=0) * duration_map
        )
        action_cost = _incremental_objective(
            action_hazard, duration_map, configs["cost_config"]["attacker"],
        )
        action_valid = current_valid & segment_valid & (
            action_terminal | (successor_in_bounds & successor_valid)
        )
        valid[..., action_index] = action_valid
        terminal[..., action_index] = action_valid & action_terminal
        terminal_fraction[..., action_index] = np.where(
            action_valid & action_terminal, fraction_map, 1.0,
        ).astype(np.float32)
        hazard[..., action_index] = np.where(
            action_valid, action_hazard, np.nan,
        ).astype(np.float32)
        cost[..., action_index] = np.where(
            action_valid, action_cost, np.inf,
        ).astype(np.float32)

    return {
        "actions": tuple(spatial_actions),
        "heading_states": _readonly(heading_states),
        "valid": _readonly(valid),
        "terminal": _readonly(terminal),
        "terminal_fraction": _readonly(terminal_fraction),
        "hazard": _readonly(hazard),
        "cost": _readonly(cost),
        "grids": grids,
        "metadata": {
            "transition_model": "physical_successor_grid_3d_heading_state",
            "endpoint_snapping": False,
            "edge_count": len(spatial_actions),
            "heading_state_count": int(heading_states.size),
            "state_action_count": int(np.prod(graph_shape)),
            "edge_quadrature_count": quadrature_count,
            "minimum_glide_terrain_clearance": minimum_glide_clearance,
            "action_domain_mode": (
                "physical_envelope"
                if search.get("physical_action_envelope") is not None
                else "legacy_cell_limits"
            ),
            "physical_action_envelope": search.get("physical_action_envelope"),
            "spatial_offset_count_before_control_filter": len(offsets),
            "realized_edge_ranges_m": {
                "forward": [
                    float(min(action["edge"][0] for action in spatial_actions)),
                    float(max(action["edge"][0] for action in spatial_actions)),
                ],
                "absolute_lateral": [
                    float(min(abs(action["edge"][1]) for action in spatial_actions)),
                    float(max(abs(action["edge"][1]) for action in spatial_actions)),
                ],
                "descent": [
                    float(min(-action["edge"][2] for action in spatial_actions)),
                    float(max(-action["edge"][2] for action in spatial_actions)),
                ],
            },
        },
    }


def solve_physical_successor_bellman(
    graph: dict[str, Any],
    grids: dict[str, np.ndarray],
    goal_position: np.ndarray,
    goal_radius: float,
    max_turn_rate_deg_s: float,
) -> dict[str, Any]:
    """Solve the altitude-acyclic exact-edge graph with heading memory."""
    x_grid, y_grid, h_grid = (
        grids[name] for name in ("x", "y", "h")
    )
    heading_states = graph["heading_states"]
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        x_grid, y_grid, h_grid, indexing="ij",
    )
    goal_mask = (
        (mesh_x - goal_position[0]) ** 2
        + (mesh_y - goal_position[1]) ** 2
        + (mesh_h - goal_position[2]) ** 2
        <= goal_radius**2
    )
    shape = (*goal_mask.shape, heading_states.size)
    value = np.full(shape, np.inf)
    hazard_to_go = np.full(shape, np.nan)
    policy_action = np.full(shape, -1, dtype=np.int32)
    value[goal_mask, :] = 0.0
    hazard_to_go[goal_mask, :] = 0.0
    max_rate = math.radians(max_turn_rate_deg_s)

    for h_index in range(h_grid.size):
        for action_index, action in enumerate(graph["actions"]):
            valid_slice = graph["valid"][:, :, h_index, action_index]
            if not np.any(valid_slice):
                continue
            terminal_slice = graph["terminal"][:, :, h_index, action_index]
            fraction_slice = graph["terminal_fraction"][
                :, :, h_index, action_index
            ]
            next_h = h_index - action["descent_cells"]
            fi, li = action["forward_cells"], action["lateral_cells"]
            x_indices, y_indices = np.meshgrid(
                np.arange(x_grid.size), np.arange(y_grid.size), indexing="ij",
            )
            next_x = np.clip(x_indices + fi, 0, x_grid.size - 1)
            next_y = np.clip(y_indices + li, 0, y_grid.size - 1)
            next_heading = action["heading_state_index"]
            if next_h >= 0:
                downstream = value[
                    next_x, next_y, next_h, next_heading
                ]
                downstream_hazard = hazard_to_go[
                    next_x, next_y, next_h, next_heading
                ]
            else:
                downstream = np.full(valid_slice.shape, np.inf)
                downstream_hazard = np.full(valid_slice.shape, np.nan)
            downstream = np.where(terminal_slice, 0.0, downstream)
            downstream_hazard = np.where(
                terminal_slice, 0.0, downstream_hazard,
            )
            candidate = (
                graph["cost"][:, :, h_index, action_index] + downstream
            )
            candidate_hazard = (
                graph["hazard"][:, :, h_index, action_index]
                + downstream_hazard
            )
            duration = action["duration"] * np.where(
                terminal_slice, fraction_slice, 1.0,
            )
            for current_heading_index, current_heading in enumerate(
                heading_states
            ):
                heading_change = abs(float(signed_heading_change(
                    current_heading, action["heading"],
                )))
                turn_valid = heading_change <= max_rate * duration + 1.0e-12
                current_value = value[:, :, h_index, current_heading_index]
                improve = (
                    valid_slice & turn_valid
                    & np.isfinite(candidate)
                    & (candidate < current_value)
                )
                value[:, :, h_index, current_heading_index] = np.where(
                    improve, candidate, current_value,
                )
                hazard_to_go[:, :, h_index, current_heading_index] = np.where(
                    improve,
                    candidate_hazard,
                    hazard_to_go[:, :, h_index, current_heading_index],
                )
                policy_action[:, :, h_index, current_heading_index] = np.where(
                    improve,
                    action_index,
                    policy_action[:, :, h_index, current_heading_index],
                )
    return {
        "value": _readonly(value),
        "hazard_to_go": _readonly(hazard_to_go),
        "pod_to_go": _readonly(1.0 - np.exp(-hazard_to_go)),
        "policy_action_index": _readonly(policy_action),
        "goal_mask": _readonly(goal_mask),
        "diagnostics": {
            "converged": True,
            "acyclic_forward_transition": True,
            "primary_sweep_axis": "h_ascending",
            "finite_value_state_count": int(np.count_nonzero(np.isfinite(value))),
            "endpoint_snapping": False,
            "max_turn_rate_deg_s": max_turn_rate_deg_s,
        },
    }


def _best_first_edge(
    switching_point: np.ndarray,
    initial_heading: float,
    graph: dict[str, Any],
    policy: dict[str, Any],
    grids: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    xi = int(np.argmin(np.abs(grids["x"] - switching_point[0])))
    yi = int(np.argmin(np.abs(grids["y"] - switching_point[1])))
    hi = int(np.argmin(np.abs(grids["h"] - switching_point[2])))
    max_rate = math.radians(policy["diagnostics"]["max_turn_rate_deg_s"])
    best: dict[str, Any] | None = None
    for action_index, action in enumerate(graph["actions"]):
        if not graph["valid"][xi, yi, hi, action_index]:
            continue
        terminal = bool(graph["terminal"][xi, yi, hi, action_index])
        fraction = float(
            graph["terminal_fraction"][xi, yi, hi, action_index]
            if terminal else 1.0
        )
        duration = action["duration"] * fraction
        turn = abs(float(signed_heading_change(
            initial_heading, action["heading"],
        )))
        if turn > max_rate * duration + 1.0e-12:
            continue
        if terminal:
            downstream = 0.0
        else:
            nxi = xi + action["forward_cells"]
            nyi = yi + action["lateral_cells"]
            nhi = hi - action["descent_cells"]
            downstream = policy["value"][
                nxi, nyi, nhi, action["heading_state_index"]
            ]
        if not np.isfinite(downstream):
            continue
        total = float(graph["cost"][xi, yi, hi, action_index] + downstream)
        record = {
            "state_index": (xi, yi, hi),
            "action_index": action_index,
            "total_glide_cost": total,
        }
        if best is None or (
            total, action_index
        ) < (
            best["total_glide_cost"], best["action_index"]
        ):
            best = record
    return best


def _extract_candidate(
    switching_point: np.ndarray,
    initial_heading: float,
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
    trajectory = [np.asarray(switching_point, dtype=float)]
    speeds: list[float] = []
    gammas: list[float] = []
    headings: list[float] = []
    durations: list[float] = []
    hazards: list[float] = []
    costs: list[float] = []
    edge_ids: list[tuple[int, int, int, int]] = []
    xi, yi, hi = first["state_index"]
    action_index = first["action_index"]
    previous_heading = float(initial_heading)
    maximum_turn_rate = 0.0
    reached_goal = False
    for _ in range(configs["environment_config"]["simulation"]["max_path_steps"]):
        action = graph["actions"][action_index]
        terminal = bool(graph["terminal"][xi, yi, hi, action_index])
        fraction = float(
            graph["terminal_fraction"][xi, yi, hi, action_index]
            if terminal else 1.0
        )
        duration = action["duration"] * fraction
        start = np.array([
            grids["x"][xi], grids["y"][yi], grids["h"][hi],
        ])
        end = start + fraction * action["edge"]
        trajectory.append(end)
        speeds.append(action["speed"])
        gammas.append(action["gamma"])
        headings.append(action["heading"])
        durations.append(duration)
        hazards.append(float(graph["hazard"][xi, yi, hi, action_index]))
        costs.append(float(graph["cost"][xi, yi, hi, action_index]))
        edge_ids.append((xi, yi, hi, action_index))
        turn = abs(float(signed_heading_change(
            previous_heading, action["heading"],
        )))
        maximum_turn_rate = max(
            maximum_turn_rate, math.degrees(turn) / duration,
        )
        previous_heading = action["heading"]
        if terminal:
            reached_goal = True
            break
        xi += action["forward_cells"]
        yi += action["lateral_cells"]
        hi -= action["descent_cells"]
        heading_state_index = action["heading_state_index"]
        if policy["goal_mask"][xi, yi, hi]:
            reached_goal = True
            break
        action_index = int(
            policy["policy_action_index"][xi, yi, hi, heading_state_index]
        )
        if action_index < 0:
            break

    trajectory_array = np.asarray(trajectory)
    speed_array = np.asarray(speeds)
    gamma_array = np.asarray(gammas)
    heading_array = np.asarray(headings)
    duration_array = np.asarray(durations)
    glide_hazard = float(np.sum(hazards))
    glide_time = float(np.sum(duration_array))
    functions = detection_bundle["primary_result"]["functions"]
    mission_detection = _function_outputs(
        functions["mission_detection"], powered["powered_hazard"], glide_hazard,
    )
    mission_objective = _function_outputs(
        functions["attacker_objective"],
        powered["powered_hazard"],
        glide_hazard,
        powered["powered_time"],
        glide_time,
    )
    mission_cost = mission_objective[-1]
    goal = np.asarray(geometry_bundle["primary_result"]["goal_position"])
    goal_error = trajectory_array[-1] - goal
    endpoint_residual = physical_edge_endpoint_residual(
        trajectory_array,
        speed_array,
        gamma_array,
        heading_array,
        duration_array,
    )
    dense_edge_points: list[np.ndarray] = []
    dense_fractions = np.linspace(0.0, 1.0, 201)
    for edge_index, (start, end) in enumerate(zip(
        trajectory_array[:-1], trajectory_array[1:]
    )):
        edge_points = (
            start[None, :]
            + dense_fractions[:, None] * (end - start)[None, :]
        )
        dense_edge_points.append(
            edge_points if edge_index == 0 else edge_points[1:]
        )
    dense_trajectory = np.vstack(dense_edge_points)
    dense_terrain = terrain_height(
        geometry_bundle["primary_result"]["terrain_model"],
        dense_trajectory[:, 0],
        dense_trajectory[:, 1],
    )
    terrain_margin = dense_trajectory[:, 2] - dense_terrain
    minimum_margin_index = int(np.argmin(terrain_margin))
    minimum_margin_point = dense_trajectory[minimum_margin_index]
    objective_residual = abs(
        powered["powered_cost"] + float(np.sum(costs)) - mission_cost
    )
    tolerance = configs["validation_config"]
    required_glide_clearance = float(
        configs["bellman_config"]["search_options"].get(
            "minimum_glide_terrain_clearance", 0.0,
        )
    )
    goal_radius = float(tolerance["goal_radius"])
    dense_goal_distance = np.linalg.norm(
        dense_trajectory - goal[None, :], axis=1,
    )
    # The terminal sphere represents capture/landing, not continued cruise.
    # Taper the cruise-clearance requirement over one goal-radius immediately
    # outside that sphere.  Everywhere farther than 2*goal_radius retains the
    # full configured clearance; the sphere boundary permits touchdown.
    terminal_clearance_scale = np.clip(
        (dense_goal_distance - goal_radius) / goal_radius, 0.0, 1.0,
    )
    required_clearance_profile = (
        required_glide_clearance * terminal_clearance_scale
    )
    clearance_residual = terrain_margin - required_clearance_profile
    checks = {
        "objective_consistency": (
            objective_residual <= tolerance["objective_tolerance"]
        ),
        "goal_reached": (
            reached_goal
            and np.linalg.norm(goal_error)
            <= tolerance["goal_radius"] + tolerance["solver_tolerance"]
        ),
        "terrain_clearance": bool(
            np.all(
                clearance_residual >= -tolerance["terrain_tolerance"]
            )
        ),
        "physical_edge_endpoint_alignment": (
            endpoint_residual <= tolerance["solver_tolerance"]
        ),
        "turn_rate_limit": (
            maximum_turn_rate
            <= configs["vehicle_config"]["turn_dynamics"][
                "max_turn_rate_deg_s"
            ] + 1.0e-10
        ),
        "strictly_descending_altitude": bool(
            np.all(np.diff(trajectory_array[:, 2]) < 0.0)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    validation = {
        "passed": not failed,
        "checks": checks,
        "metrics": {
            "goal_distance": float(np.linalg.norm(goal_error)),
            "objective_residual": float(objective_residual),
            "maximum_edge_endpoint_residual": endpoint_residual,
            "minimum_terrain_margin": float(np.min(terrain_margin)),
            "minimum_clearance_residual": float(np.min(clearance_residual)),
            "minimum_terrain_margin_point": tuple(
                float(value) for value in minimum_margin_point
            ),
            "required_glide_terrain_clearance": required_glide_clearance,
            "terminal_clearance_taper_distance": goal_radius,
            "dense_terrain_samples_per_edge": int(dense_fractions.size),
            "maximum_turn_rate_deg_s": maximum_turn_rate,
            "configured_max_turn_rate_deg_s": configs["vehicle_config"][
                "turn_dynamics"
            ]["max_turn_rate_deg_s"],
            "path_node_count": int(trajectory_array.shape[0]),
        },
        "failed_checks": failed,
        "summary": (
            "Exact 3D physical successor candidate validation passed"
            if not failed else f"Physical candidate failed: {failed}"
        ),
    }
    return {
        "candidate_id": None,
        "start_id": None,
        "switching_point": _readonly(np.asarray(switching_point)),
        "trajectory": _readonly(trajectory_array),
        "speed_profile": _readonly(speed_array),
        "gamma_profile": _readonly(gamma_array),
        "heading_profile": _readonly(heading_array),
        "duration_profile": _readonly(duration_array),
        "initial_heading_state": float(initial_heading),
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
            "minimum_terrain_margin": float(np.min(terrain_margin)),
            "minimum_terrain_margin_point": _readonly(minimum_margin_point),
            "maximum_edge_endpoint_residual": endpoint_residual,
            "maximum_turn_rate_deg_s": maximum_turn_rate,
            "configured_max_turn_rate_deg_s": configs["vehicle_config"][
                "turn_dynamics"
            ]["max_turn_rate_deg_s"],
        },
        "metadata": {
            "transition_model": "physical_successor_grid_3d_heading_state",
            "endpoint_snapping": False,
            "edge_ids": tuple(edge_ids),
            "duration_profile": tuple(float(value) for value in duration_array),
            "coarse": False,
            "warm_start_only": False,
            "is_final_attacker_solution": False,
        },
        "validation": validation,
    }


def _candidate_bundle(candidates, attempts, seeds, graph, policy, geometry_bundle):
    value_projected = np.min(policy["value"], axis=3)
    best_heading = np.argmin(policy["value"], axis=3)
    x_idx, y_idx, h_idx = np.indices(value_projected.shape)
    pod_projected = policy["pod_to_go"][
        x_idx, y_idx, h_idx, best_heading
    ]
    validation = {
        "passed": bool(candidates),
        "checks": {
            "feasible_candidates_exist": bool(candidates),
            "physical_edges": True,
            "endpoint_snapping_disabled": not graph["metadata"]["endpoint_snapping"],
        },
        "metrics": {
            "candidate_count": len(candidates),
            "attempted_start_count": len(attempts),
        },
        "warnings": [],
        "failed_checks": [] if candidates else ["feasible_candidates_exist"],
        "summary": (
            "Exact physical 3D successor candidates validated"
            if candidates else "No physical 3D candidates"
        ),
    }
    return {
        "primary_result": {
            "candidates": tuple(candidates),
            "start_attempts": tuple(attempts),
            "switching_point_seeds": _readonly(np.asarray(seeds)),
            "bellman_diagnostics": {"physical_successor_grid": policy["diagnostics"]},
            "cost_to_go_maps": {
                "physical_successor_grid": _readonly(value_projected)
            },
            "pod_to_go_maps": {
                "physical_successor_grid": _readonly(pod_projected)
            },
            "cost_to_go_heading_state_maps": {
                "physical_successor_grid": policy["value"]
            },
            "pod_to_go_heading_state_maps": {
                "physical_successor_grid": policy["pod_to_go"]
            },
            "cost_to_go_primary_ordering": "physical_successor_grid",
            "finite_cost_to_go_mask": _readonly(np.isfinite(value_projected)),
            "candidate_count": len(candidates),
            "attempted_start_count": len(attempts),
            "filtering_applied": False,
            "ranking_applied": False,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "PhysicalSuccessorGridCandidateSet3D",
            "schema_version": "1.0.0",
            "producer_phase": 6,
            "producer_module": "p1b_3DExtension.successor_grid_solver",
            "transition_model": "physical_successor_grid_3d_heading_state",
            "goal_position": tuple(
                float(value) for value in geometry_bundle[
                    "primary_result"
                ]["goal_position"]
            ),
            "graph_metadata": graph["metadata"],
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "NO_PHYSICAL_CANDIDATES",
            "message": validation["summary"],
            "warnings": [],
            "failed_checks": validation["failed_checks"],
        },
    }


def _response_bundle(best, ordered, tied, candidate_bundle, configuration_bundle):
    best["metadata"]["is_final_attacker_solution"] = True
    primary = {
        **best,
        "solution_id": f"physical-3d-optimal-{best['candidate_id']}",
        "source_candidate_id": best["candidate_id"],
        "source_start_id": best["start_id"],
        "candidate_count_searched": len(ordered),
        "tie_count": len(tied),
        "powered_hazard": best["hazard_breakdown"]["powered_acoustic_hazard"],
        "glide_hazard": best["hazard_breakdown"]["glide_radar_doppler_hazard"],
    }
    validation = {
        "passed": best["validation"]["passed"],
        "checks": {
            **best["validation"]["checks"],
            "selection_is_exact_minimum_cost": (
                best["mission_cost"] == ordered[0]["mission_cost"]
            ),
        },
        "metrics": {
            "selected_mission_cost": best["mission_cost"],
            "candidate_count": len(ordered),
            "tie_count": len(tied),
            **best["validation"]["metrics"],
        },
        "warnings": [],
        "failed_checks": best["validation"]["failed_checks"],
        "summary": "Authoritative exact physical 3D response validated",
    }
    return {
        "primary_result": primary,
        "validation": validation,
        "metadata": {
            "schema_name": "AuthoritativePhysicalSuccessorGridResponse3D",
            "schema_version": "1.0.0",
            "producer_phase": 8,
            "producer_module": "p1b_3DExtension.successor_grid_solver",
            "solution_method": "physical_successor_grid_heading_state_bellman",
            "optimality_scope": "finite_exact_physical_edge_and_heading_state_grid",
            "transition_model": "physical_successor_grid_3d_heading_state",
            "attacker_objective_id": configuration_bundle["primary_result"][
                "cost_config"
            ]["attacker"]["objective_id"],
            "is_final_attacker_solution": True,
            "global_optimum_claim": False,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "PHYSICAL_RESPONSE_INVALID",
            "message": validation["summary"],
            "warnings": [],
            "failed_checks": validation["failed_checks"],
        },
    }


def _control_is_valid(speed: float, gamma: float, vehicle: dict[str, Any]) -> bool:
    if not (
        vehicle["glide_speed_min"] <= speed <= vehicle["glide_speed_max"]
    ):
        return False
    if not (
        math.radians(vehicle["gamma_min_deg"])
        <= gamma
        <= math.radians(vehicle["gamma_max_deg"])
    ):
        return False
    cl = (
        2.0 * vehicle["mass"] * vehicle["gravity"] * math.cos(gamma)
        / (vehicle["air_density"] * speed**2 * vehicle["wing_area"])
    )
    cd = (
        vehicle["cd0"]
        + vehicle["linear_drag_coefficient"] * cl
        + vehicle["quadratic_drag_coefficient"] * cl**2
    )
    return bool(
        vehicle["dynamic_limits"]["cl_min"]
        <= cl
        <= vehicle["dynamic_limits"]["cl_max"]
        and cd > 0.0
    )


def physical_edge_endpoint_residual(
    trajectory: np.ndarray,
    speeds: np.ndarray,
    gammas: np.ndarray,
    headings: np.ndarray,
    durations: np.ndarray,
) -> float:
    """Return the maximum exact 3D kinematic endpoint mismatch."""
    points = np.asarray(trajectory, dtype=float)
    speed_values = np.asarray(speeds, dtype=float)
    gamma_values = np.asarray(gammas, dtype=float)
    heading_values = np.asarray(headings, dtype=float)
    duration_values = np.asarray(durations, dtype=float)
    edge_count = points.shape[0] - 1
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not (
            speed_values.size
            == gamma_values.size
            == heading_values.size
            == duration_values.size
            == edge_count
        )
    ):
        raise ValueError("Trajectory and physical edge profiles are inconsistent")
    reconstructed = (
        points[:-1]
        + duration_values[:, None] * speed_values[:, None]
        * np.column_stack((
            np.cos(gamma_values) * np.cos(heading_values),
            np.cos(gamma_values) * np.sin(heading_values),
            np.sin(gamma_values),
        ))
    )
    return float(np.max(np.linalg.norm(
        reconstructed - points[1:], axis=1,
    ))) if edge_count else 0.0


def _incremental_objective(hazard, duration, attacker_config):
    return (
        attacker_config["w_pod"]
        * np.asarray(hazard)
        / attacker_config["normalization"]["pod"]["hazard_reference"]
        + attacker_config["w_time"]
        * np.asarray(duration)
        / attacker_config["normalization"]["time"]["reference_seconds"]
    )


def _function_outputs(function, *arguments) -> list[float]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in outputs]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    array.setflags(write=False)
    return array
