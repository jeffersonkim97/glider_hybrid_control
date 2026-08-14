"""Turn-limited multi-start coarse Bellman generator using authoritative J6D.

Mirrors p1b_4D.bellman's role exactly, with the primary sweep axis
switched from z to h (see stage_cost.py's module docstring and the
notebook's design-rationale cell): h is the only state coordinate that is
guaranteed monotonic along every glide trajectory regardless of heading
(dh/dt = v*sin(gamma) has no heading dependence and gamma never reaches
zero), so backward induction sweeps h ascending from the goal's altitude
up to h_max, with (x, y) free inside every h-slice -- the direct 3D
analog of p1b_4D's "sweep z descending, h free inside each z-slice".

Heading is now a periodic Bellman state, so the dynamic-programming state
is (x, y, h, psi). J6D remains the local spatial/action cost map over
(x, y, h, v, gamma, selected_course); a transition may select only a
course within the configured turn-rate envelope of psi, and the selected
course becomes the successor heading state.

The per-h-slice update is vectorized over the full (x, y, psi) slice
instead of using nested Python loops over every spatial index (as
p1b_4D's much smaller z*h grid could afford): no cell in an h-slice can
ever be a valid non-terminal successor of another cell in the *same*
slice (transitions strictly decrease h by construction), so every
spatial cell in a slice can be updated in one vectorized pass per
candidate action without changing the DP's result.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .geometry import terrain_height
from .turn_dynamics import (
    heading_change_metrics,
    heading_transition_mask,
    nearest_heading_index,
    powered_segment_heading,
)


def generate_bellman_candidates(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_6d_bundle: dict[str, Any],
    projected_cost_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate all coarse Attacker candidates without filtering or ranking.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.
    geometry_bundle:
        Successful Phase 2 GeometryBundle.
    detection_bundle:
        Successful Phase 3 DetectionBundle.
    stage_cost_6d_bundle:
        Successful authoritative J6D local-stage result.
    projected_cost_bundle:
        Successful visualization-only projection. It is checked solely to
        enforce its prohibited-use marker and is never read by Bellman.

    Outputs
    -------
    dict
        Universal BellmanCandidateSet with every successful coarse solution,
        every attempted seed, Bellman diagnostics, metadata, and validation.

    Assumptions
    -----------
    Coarse transitions use constant speed, gamma, and selected course over
    each coarse interval. Heading is carried between intervals and the
    course increment is turn-rate limited.

    Notes
    -----
    No candidate ranking, duplicate removal, continuous refinement, Defender
    optimization, file writing, or plotting occurs here.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    _require_successful_bundle(detection_bundle, "detection_bundle")
    _require_successful_bundle(stage_cost_6d_bundle, "stage_cost_6d_bundle")
    if projected_cost_bundle is not None:
        _require_successful_bundle(projected_cost_bundle, "projected_cost_bundle")
        if not projected_cost_bundle["metadata"].get("visualization_only", False):
            raise ValueError("ProjectedCost input must be marked visualization_only")
        if projected_cost_bundle["primary_result"]["projection_metadata"].get(
            "bellman_policy_input", True
        ):
            raise ValueError("ProjectedCost must explicitly prohibit Bellman use")

    configs = configuration_bundle["primary_result"]
    vehicle = configs["vehicle_config"]
    bellman_config = configs["bellman_config"]
    validation_config = configs["validation_config"]
    goal_position = geometry_bundle["primary_result"]["goal_position"]
    stage = stage_cost_6d_bundle["primary_result"]
    grids = stage["grids"]
    j6d = stage["j6d"]
    transitions = construct_coarse_transitions(
        geometry_bundle,
        stage_cost_6d_bundle,
        configuration_bundle,
    )
    orderings = bellman_config["search_options"]["exploration_orderings"]
    policies = {
        ordering: solve_coarse_bellman(
            j6d,
            transitions,
            grids,
            goal_position,
            validation_config,
            ordering,
            bellman_config,
            vehicle,
        )
        for ordering in orderings
    }
    glide_detection_rate = stage["component_maps"]["glide_detection_rate"]
    time_step = vehicle["time_step"]
    coarse_step_count = transitions["coarse_step_count"]
    pod_to_go_maps = {
        ordering: _compute_glide_pod_to_go(
            policy, transitions, glide_detection_rate, time_step, coarse_step_count,
        )
        for ordering, policy in policies.items()
    }
    seeds = generate_switching_point_seeds(
        geometry_bundle, grids["x"], grids["y"], grids["h"],
        include_visible=(
            bellman_config["search_options"].get(
                "powered_visibility_handling", "hard_hidden",
            ) == "hazard_penalty"
        ),
        candidate_mask=np.any(np.stack([
            np.any(np.isfinite(policy["value"]), axis=3)
            for policy in policies.values()
        ]), axis=0),
        boundary_only=(
            bellman_config["search_options"].get(
                "switching_candidate_mode", "admissible_volume",
            ) == "los_boundary_surface"
        ),
        boundary_tolerance_m=float(
            vehicle["switching_constraints"]["tangent_tolerance"]
        ),
    )
    candidates: list[dict[str, Any]] = []
    start_attempts: list[dict[str, Any]] = []
    for seed_index, switching_point in enumerate(seeds):
        ordering = orderings[seed_index % len(orderings)]
        spatial_start_index = _switching_grid_index(switching_point, grids)
        attempt = {
            "start_id": f"switch-seed-{seed_index:04d}",
            "seed_index": seed_index,
            "switching_point": switching_point,
            "grid_start_index": spatial_start_index,
            "exploration_ordering": ordering,
            "success": False,
            "diagnostic": None,
        }
        powered = evaluate_powered_segment(
            switching_point,
            configuration_bundle,
            geometry_bundle,
            detection_bundle,
            grids,
        )
        if not powered["validation"]["passed"]:
            attempt["diagnostic"] = powered["validation"]["summary"]
            start_attempts.append(attempt)
            continue
        initial_heading = powered_segment_heading(powered["path"], goal_position)
        initial_heading_index = nearest_heading_index(
            initial_heading, grids["heading"],
        )
        start_index = (*spatial_start_index, initial_heading_index)
        attempt["initial_heading"] = initial_heading
        attempt["initial_heading_index"] = initial_heading_index
        extracted = extract_coarse_candidate(
            switching_point,
            start_index,
            policies[ordering],
            transitions,
            stage_cost_6d_bundle,
            configuration_bundle,
            geometry_bundle,
            detection_bundle,
            powered,
        )
        if not extracted["success"]:
            attempt["diagnostic"] = extracted["diagnostic"]
            start_attempts.append(attempt)
            continue
        candidate = extracted["candidate"]
        candidate["candidate_id"] = f"bellman-candidate-{len(candidates):04d}"
        candidate["start_id"] = attempt["start_id"]
        candidate["metadata"].update(
            {
                "seed_index": seed_index,
                "exploration_ordering": ordering,
                "grid_start_index": spatial_start_index,
                "heading_state_start_index": initial_heading_index,
            }
        )
        attempt["success"] = True
        attempt["candidate_id"] = candidate["candidate_id"]
        attempt["diagnostic"] = "candidate_generated"
        candidates.append(candidate)
        start_attempts.append(attempt)

    validation = validate_bellman_candidate_set(
        candidates,
        start_attempts,
        policies,
        configuration_bundle,
        geometry_bundle,
        stage_cost_6d_bundle,
        pod_to_go_maps,
    )

    primary_ordering = orderings[0]
    primary_cost_to_go_state = policies[primary_ordering]["value"]
    primary_cost_to_go = np.min(primary_cost_to_go_state, axis=3)
    finite_cost_to_go_mask = np.isfinite(primary_cost_to_go)

    return {
        "primary_result": {
            "candidates": tuple(candidates),
            "start_attempts": tuple(start_attempts),
            "switching_point_seeds": _readonly(seeds),
            "bellman_diagnostics": {
                ordering: policy["diagnostics"]
                for ordering, policy in policies.items()
            },
            "cost_to_go_maps": {
                ordering: _readonly(np.min(policy["value"], axis=3))
                for ordering, policy in policies.items()
            },
            "cost_to_go_heading_state_maps": {
                ordering: policy["value"] for ordering, policy in policies.items()
            },
            "pod_to_go_maps": {
                ordering: _readonly(_finite_minimum(values, axis=3))
                for ordering, values in pod_to_go_maps.items()
            },
            "pod_to_go_heading_state_maps": {
                ordering: _readonly(values)
                for ordering, values in pod_to_go_maps.items()
            },
            "cost_to_go_primary_ordering": primary_ordering,
            "finite_cost_to_go_mask": _readonly(finite_cost_to_go_mask),
            "candidate_count": len(candidates),
            "attempted_start_count": len(start_attempts),
            "filtering_applied": False,
            "ranking_applied": False,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "BellmanCandidateSet3D",
            "schema_version": "1.1.0",
            "producer_phase": 6,
            "producer_module": "p1b_3DExtension.bellman",
            "candidate_role": "coarse_topology_and_nlp_warm_start",
            "is_final_attacker_solution": False,
            "global_optimum_claim": False,
            "local_cost_source": "StageCost6DResult.j6d",
            "cost_to_go_role": "Bellman value map for exported visualization",
            "local_cost_source_schema": stage_cost_6d_bundle["metadata"][
                "schema_version"
            ],
            "projected_cost_used_for_policy": False,
            "projected_cost_dependency_role": (
                "prohibited_use_contract_check_only"
                if projected_cost_bundle is not None
                else "not_supplied"
            ),
            "attacker_objective_id": configs["cost_config"]["attacker"][
                "objective_id"
            ],
            "objective_weights": {
                "w_pod": configs["cost_config"]["attacker"]["w_pod"],
                "w_time": configs["cost_config"]["attacker"]["w_time"],
            },
            "random_seed": bellman_config["random_seed"],
            "filtering_applied": False,
            "primary_sweep_axis": "h_ascending",
            "state_axis_order": ("x", "y", "h", "heading"),
            "turn_dynamics_model": vehicle["turn_dynamics"]["model"],
            "max_turn_rate_deg_s": vehicle["turn_dynamics"]["max_turn_rate_deg_s"],
            "goal_position": (
                float(goal_position[0]), float(goal_position[1]), float(goal_position[2]),
            ),
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "BELLMAN_CANDIDATES_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def generate_switching_point_seeds(
    geometry_bundle: dict[str, Any],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    h_grid: np.ndarray,
    include_visible: bool = False,
    candidate_mask: np.ndarray | None = None,
    boundary_only: bool = False,
    boundary_tolerance_m: float | None = None,
) -> np.ndarray:
    """Enumerate reachable airspace cells as powered switch candidates.

    Direct 3D analog of p1b_4D's exhaustive z-grid-node enumeration along
    its single LOS boundary curve: there is no single boundary curve here
    (terrain is a 2D surface), so there is also no single "top of shadow"
    height per (x, y) column that is safe to assume reachable -- unlike
    p1b_4D's 1D tangent line (which is, by construction, always the
    silhouette of the *one* obstacle between launch and the sensor), a
    column's shadow ceiling here says nothing about whether a straight
    powered line from launch can actually reach it without first clipping
    the hill (an empirically confirmed failure mode: picking only the
    ceiling made every one of a first attempt's ~1000 seeds fail terrain
    clearance). In hard-hidden mode every occluded (x, y, h) cell is its
    own candidate.  In hazard-penalty mode visible powered flight is legal,
    so every non-terrain cell allowed by `candidate_mask` is eligible;
    otherwise the seed generator would silently reintroduce the obsolete
    hard-LOS constraint.  `candidate_mask` normally retains only cells with
    a finite glide policy to the goal, avoiding pointless powered checks.
    """
    masks = geometry_bundle["primary_result"]["los_masks"]
    non_visible = masks["non_visible_airspace_mask"]
    expected_shape = (x_grid.size, y_grid.size, h_grid.size)
    if non_visible.shape != expected_shape:
        raise ValueError("non_visible_airspace_mask does not match the Bellman grid")
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != expected_shape:
            raise ValueError("candidate_mask does not match the Bellman grid")
    if boundary_only:
        if boundary_tolerance_m is not None and boundary_tolerance_m < 0.0:
            raise ValueError("boundary_tolerance_m must be nonnegative")
        boundary = np.asarray(masks["los_boundary_height"], dtype=float)
        if boundary.shape != expected_shape[:2]:
            raise ValueError("los_boundary_height does not match the horizontal grid")
        sensor_x = float(
            geometry_bundle["primary_result"]["sensor_position"][0]
        )
        seeds: list[tuple[float, float, float]] = []
        for x_index, x_value in enumerate(x_grid):
            if x_value >= sensor_x:
                continue
            for y_index, y_value in enumerate(y_grid):
                height = float(boundary[x_index, y_index])
                # As in p1b_4D's final switching-consistency check, a true
                # boundary lying above the ceiling is not silently accepted
                # as an on-boundary switch after clipping.
                if height < h_grid[0] or height > h_grid[-1]:
                    continue
                h_index = int(np.argmin(np.abs(h_grid - height)))
                if (
                    boundary_tolerance_m is not None
                    and abs(float(h_grid[h_index]) - height)
                    > boundary_tolerance_m
                ):
                    continue
                if masks["terrain_mask"][x_index, y_index, h_index]:
                    continue
                if candidate_mask is not None and not candidate_mask[
                    x_index, y_index, h_index
                ]:
                    continue
                # The physical successor graph is node-to-node exact.  Use
                # its nearest altitude node only for the discrete initializer;
                # the continuous refinement below enforces the unsnapped
                # equality h_switch = H_LOS(x_switch, y_switch).
                seeds.append((
                    float(x_value), float(y_value), float(h_grid[h_index]),
                ))
        if not seeds:
            raise ValueError("No LOS-boundary switching seed is glide-feasible")
        return np.asarray(seeds, dtype=float)
    if include_visible:
        admissible = ~np.asarray(masks["terrain_mask"], dtype=bool)
    else:
        admissible = np.asarray(non_visible, dtype=bool).copy()
    if candidate_mask is not None:
        admissible &= candidate_mask
    x_indices, y_indices, h_indices = np.nonzero(admissible)
    if x_indices.size == 0:
        raise ValueError("No admissible airspace cell exists for a powered switch")
    # Lower switching altitude first within each column: a straight line
    # from ground-level launch to a lower target has a gentler climb and
    # is geometrically more likely to clear the hill, so ordering seeds
    # this way finds a feasible candidate earlier rather than only after
    # exhausting every high-altitude cell first.
    order = np.lexsort((h_indices, y_indices, x_indices))
    return np.column_stack(
        (x_grid[x_indices], y_grid[y_indices], h_grid[h_indices])
    )[order]


def construct_coarse_transitions(
    geometry_bundle: dict[str, Any],
    stage_cost_6d_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct coarse kinematic successors without using ProjectedCost."""
    geometry = geometry_bundle["primary_result"]
    stage = stage_cost_6d_bundle["primary_result"]
    grids = stage["grids"]
    vehicle = configuration_bundle["primary_result"]["vehicle_config"]
    bellman = configuration_bundle["primary_result"]["bellman_config"]
    validation = configuration_bundle["primary_result"]["validation_config"]
    x_grid, y_grid, h_grid, v_grid, gamma_grid, heading_grid = (
        grids[name] for name in ("x", "y", "h", "v", "gamma", "heading")
    )
    shape = stage["j6d"].shape
    next_x_index = np.full(shape, -1, dtype=np.int32)
    next_y_index = np.full(shape, -1, dtype=np.int32)
    next_h_index = np.full(shape, -1, dtype=np.int32)
    transition_valid = np.zeros(shape, dtype=bool)
    terminal_transition = np.zeros(shape, dtype=bool)
    terminal_fraction = np.ones(shape, dtype=float)
    mesh_x, mesh_y, mesh_h = np.meshgrid(x_grid, y_grid, h_grid, indexing="ij")
    dx_grid = float(x_grid[1] - x_grid[0])
    dy_grid = float(y_grid[1] - y_grid[0])
    dh_grid = float(h_grid[1] - h_grid[0])
    terrain_model = geometry["terrain_model"]
    terrain_mask = geometry["los_masks"]["terrain_mask"]
    goal_position = geometry["goal_position"]
    segment_count = bellman["search_options"]["segment_check_count"]
    fractions = np.linspace(0.0, 1.0, segment_count)
    goal_radius = float(validation["goal_radius"])
    terrain_tolerance = validation["terrain_tolerance"]
    current_h_indices = np.arange(h_grid.size)[None, None, :]

    # x, y are free (unlike p1b_4D's single free h axis, which shared its
    # grid with the swept z axis and so had no analogous mismatch): the
    # fastest, shallowest single-time-step glide action moves at most
    # ~glide_speed_max*cos(gamma_max_deg) horizontally, which on this
    # grid's spacing is *less* than half a cell -- every action's nearest-
    # grid-cell successor then rounds right back to its own (x, y) cell,
    # trapping every glide chain in its starting column with no way to
    # ever reach the goal's column (confirmed empirically: with a literal
    # one-time-step transition, 0 of tens of thousands of candidate
    # switching points found a path). A coarse Bellman transition here
    # therefore holds (v, gamma, heading) constant for `coarse_step_count`
    # physical time steps instead of one -- enough that the fastest action
    # reliably crosses into an adjacent cell -- while every other part of
    # the pipeline (the J6D spatial grid, its per-time-step rate/cost
    # formulas) is unchanged.
    max_horizontal_speed = float(vehicle["glide_speed_max"]) * float(
        np.cos(np.deg2rad(vehicle["gamma_max_deg"]))
    )
    half_min_spacing = 0.5 * min(dx_grid, dy_grid)
    coarse_step_count = max(
        1,
        int(np.ceil(half_min_spacing / (max_horizontal_speed * vehicle["time_step"]))) + 1,
    )
    effective_dt = coarse_step_count * vehicle["time_step"]

    for velocity_index, velocity in enumerate(v_grid):
        for gamma_index, gamma in enumerate(gamma_grid):
            for heading_index, heading in enumerate(heading_grid):
                delta_x = velocity * effective_dt * np.cos(gamma) * np.cos(heading)
                delta_y = velocity * effective_dt * np.cos(gamma) * np.sin(heading)
                delta_h = velocity * effective_dt * np.sin(gamma)
                next_x = mesh_x + delta_x
                next_y = mesh_y + delta_y
                next_h = mesh_h + delta_h
                relative_x = mesh_x - goal_position[0]
                relative_y = mesh_y - goal_position[1]
                relative_h = mesh_h - goal_position[2]
                quadratic_a = delta_x**2 + delta_y**2 + delta_h**2
                quadratic_b = (
                    relative_x * delta_x + relative_y * delta_y + relative_h * delta_h
                )
                quadratic_c = (
                    relative_x**2 + relative_y**2 + relative_h**2 - goal_radius**2
                )
                discriminant = quadratic_b**2 - quadratic_a * quadratic_c
                first_intersection = (
                    -quadratic_b - np.sqrt(np.maximum(discriminant, 0.0))
                ) / quadratic_a
                terminal = (
                    (quadratic_c > 0.0)
                    & (discriminant >= 0.0)
                    & (first_intersection > 0.0)
                    & (first_intersection <= 1.0)
                )
                action_terminal_fraction = np.where(
                    terminal, np.clip(first_intersection, 0.0, 1.0), 1.0,
                )
                # x, y are the free spatial axes (like p1b_4D's h): plain
                # nearest-grid rounding. h is the monotone sweep axis (like
                # p1b_4D's z): floor, biased toward the direction of travel
                # (h only ever decreases), so rounding never manufactures a
                # spurious advance; `advances` below still gates explicitly.
                mapped_x = np.rint((next_x - x_grid[0]) / dx_grid).astype(np.int64)
                mapped_y = np.rint((next_y - y_grid[0]) / dy_grid).astype(np.int64)
                mapped_h = np.floor((next_h - h_grid[0]) / dh_grid).astype(np.int64)
                inside = (
                    (mapped_x >= 0) & (mapped_x < x_grid.size)
                    & (mapped_y >= 0) & (mapped_y < y_grid.size)
                    & (mapped_h >= 0) & (mapped_h < h_grid.size)
                )
                advances = mapped_h < current_h_indices
                mapped_x_safe = np.clip(mapped_x, 0, x_grid.size - 1)
                mapped_y_safe = np.clip(mapped_y, 0, y_grid.size - 1)
                mapped_h_safe = np.clip(mapped_h, 0, h_grid.size - 1)
                # Glide feasibility only requires clearing terrain, not
                # staying visible/hidden -- see stage_cost.py's module
                # docstring: the hazard-rate formula already gates
                # detection on visibility, so re-hiding mid-glide is a
                # legal maneuver here, unlike p1b_4D.
                successor_spatial_valid = ~terrain_mask[
                    mapped_x_safe, mapped_y_safe, mapped_h_safe
                ]
                successor_goal = (
                    (x_grid[mapped_x_safe] - goal_position[0]) ** 2
                    + (y_grid[mapped_y_safe] - goal_position[1]) ** 2
                    + (h_grid[mapped_h_safe] - goal_position[2]) ** 2
                    <= goal_radius**2
                )
                segment_valid = np.ones(mesh_x.shape, dtype=bool)
                for fraction_index, fraction in enumerate(fractions[1:], start=1):
                    effective_fraction = fraction * action_terminal_fraction
                    sample_x = mesh_x + effective_fraction * delta_x
                    sample_y = mesh_y + effective_fraction * delta_y
                    sample_h = mesh_h + effective_fraction * delta_h
                    sample_in_domain = (
                        (sample_x >= x_grid[0]) & (sample_x <= x_grid[-1])
                        & (sample_y >= y_grid[0]) & (sample_y <= y_grid[-1])
                    )
                    clipped_x = np.clip(sample_x, x_grid[0], x_grid[-1])
                    clipped_y = np.clip(sample_y, y_grid[0], y_grid[-1])
                    terrain_clear = (
                        sample_h >= terrain_height(terrain_model, clipped_x, clipped_y)
                        - terrain_tolerance
                    )
                    sample_valid = sample_in_domain & terrain_clear
                    if fraction_index == fractions.size - 1:
                        sample_valid = sample_valid | terminal
                    segment_valid &= sample_valid
                local_finite = np.isfinite(
                    stage["j6d"][:, :, :, velocity_index, gamma_index, heading_index]
                )
                valid = (
                    local_finite
                    & segment_valid
                    & (
                        terminal
                        | (inside & advances & successor_spatial_valid & ~successor_goal)
                    )
                )
                index = (slice(None), slice(None), slice(None), velocity_index, gamma_index, heading_index)
                next_x_index[index] = np.where(valid & ~terminal, mapped_x, -1)
                next_y_index[index] = np.where(valid & ~terminal, mapped_y, -1)
                next_h_index[index] = np.where(valid & ~terminal, mapped_h, -1)
                transition_valid[index] = valid
                terminal_transition[index] = valid & terminal
                terminal_fraction[index] = np.where(
                    valid & terminal, action_terminal_fraction, 1.0,
                )
    return {
        "next_x_index": _readonly(next_x_index),
        "next_y_index": _readonly(next_y_index),
        "next_h_index": _readonly(next_h_index),
        "transition_valid": _readonly(transition_valid),
        "terminal_transition": _readonly(terminal_transition),
        "terminal_fraction": _readonly(terminal_fraction),
        # Physical time steps held per coarse transition (see the
        # derivation above `effective_dt`): local cost/hazard/time
        # incurred by one non-terminal transition scale by this factor
        # relative to stage_cost.py's per-time-step rates.
        "coarse_step_count": coarse_step_count,
    }


def solve_coarse_bellman(
    j6d: np.ndarray,
    transitions: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    goal_position: np.ndarray,
    validation_config: dict[str, Any],
    exploration_ordering: str,
    bellman_config: dict[str, Any],
    vehicle_config: dict[str, Any],
) -> dict[str, Any]:
    """Solve the heading-state, forward-acyclic Bellman recursion.

    Sweeps h ascending (successors of an h-slice always live at strictly
    smaller h), vectorizing the update over (x, y) for each candidate
    action and the compatible incoming heading states.
    """
    x_grid, y_grid, h_grid, v_grid, gamma_grid, heading_grid = (
        grids[name] for name in ("x", "y", "h", "v", "gamma", "heading")
    )
    coarse_step_count = transitions["coarse_step_count"]
    transition_duration = coarse_step_count * vehicle_config["time_step"]
    max_turn_rate = np.deg2rad(
        vehicle_config["turn_dynamics"]["max_turn_rate_deg_s"]
    )
    turn_mask = heading_transition_mask(
        heading_grid, max_turn_rate, transition_duration,
    )
    shape = (x_grid.size, y_grid.size, h_grid.size, heading_grid.size)
    value = np.full(shape, np.inf)
    policy_velocity = np.full(shape, -1, dtype=np.int32)
    policy_gamma = np.full(shape, -1, dtype=np.int32)
    policy_heading = np.full(shape, -1, dtype=np.int32)
    policy_next_x = np.full(shape, -1, dtype=np.int32)
    policy_next_y = np.full(shape, -1, dtype=np.int32)
    policy_next_h = np.full(shape, -1, dtype=np.int32)
    policy_terminal = np.zeros(shape, dtype=bool)
    mesh_x, mesh_y, mesh_h = np.meshgrid(x_grid, y_grid, h_grid, indexing="ij")
    goal_mask = (
        (mesh_x - goal_position[0]) ** 2
        + (mesh_y - goal_position[1]) ** 2
        + (mesh_h - goal_position[2]) ** 2
        <= validation_config["goal_radius"] ** 2
    )
    value[goal_mask, :] = 0.0
    actions = _ordered_actions(v_grid, gamma_grid, heading_grid, exploration_ordering)
    updated_state_count = 0
    slice_shape = (x_grid.size, y_grid.size, heading_grid.size)
    for h_index in range(h_grid.size):
        active = np.broadcast_to(
            (~goal_mask[:, :, h_index])[:, :, None], slice_shape,
        )
        if not np.any(active):
            continue
        best_cost = np.full(slice_shape, np.inf)
        best_velocity = np.full(slice_shape, -1, dtype=np.int32)
        best_gamma = np.full(slice_shape, -1, dtype=np.int32)
        best_heading = np.full(slice_shape, -1, dtype=np.int32)
        best_next_x = np.full(slice_shape, -1, dtype=np.int32)
        best_next_y = np.full(slice_shape, -1, dtype=np.int32)
        best_next_h = np.full(slice_shape, -1, dtype=np.int32)
        best_terminal = np.zeros(slice_shape, dtype=bool)
        for velocity_index, gamma_index, heading_index in actions:
            compatible_heading_indices = np.flatnonzero(turn_mask[:, heading_index])
            if compatible_heading_indices.size == 0:
                continue
            action_index = (
                slice(None), slice(None), h_index, velocity_index, gamma_index, heading_index,
            )
            valid_slice = transitions["transition_valid"][action_index]
            if not np.any(valid_slice):
                continue
            terminal_slice = transitions["terminal_transition"][action_index]
            next_x_slice = transitions["next_x_index"][action_index]
            next_y_slice = transitions["next_y_index"][action_index]
            next_h_slice = transitions["next_h_index"][action_index]
            fraction_slice = transitions["terminal_fraction"][action_index]
            safe_next_x = np.clip(next_x_slice, 0, x_grid.size - 1)
            safe_next_y = np.clip(next_y_slice, 0, y_grid.size - 1)
            safe_next_h = np.clip(next_h_slice, 0, h_grid.size - 1)
            downstream = np.where(
                terminal_slice,
                0.0,
                value[safe_next_x, safe_next_y, safe_next_h, heading_index],
            )
            local_fraction = np.where(terminal_slice, fraction_slice, 1.0)
            # A non-terminal transition covers coarse_step_count physical
            # time steps at this constant action, so its local cost scales
            # by the same factor relative to stage_cost.py's per-step rate
            # (a terminal transition's own fraction already accounts for
            # its partial-segment duration).
            local_cost = coarse_step_count * j6d[
                :, :, h_index, velocity_index, gamma_index, heading_index
            ]
            candidate_cost = local_fraction * local_cost + downstream
            active_subset = active[:, :, compatible_heading_indices]
            best_subset = best_cost[:, :, compatible_heading_indices]
            improve = (
                active_subset
                & valid_slice[:, :, None]
                & (candidate_cost[:, :, None] < best_subset)
            )
            best_cost[:, :, compatible_heading_indices] = np.where(
                improve, candidate_cost[:, :, None], best_subset,
            )
            for policy_array, selected_value in (
                (best_velocity, velocity_index),
                (best_gamma, gamma_index),
                (best_heading, heading_index),
            ):
                subset = policy_array[:, :, compatible_heading_indices]
                policy_array[:, :, compatible_heading_indices] = np.where(
                    improve, selected_value, subset,
                )
            for policy_array, selected_value in (
                (best_next_x, next_x_slice),
                (best_next_y, next_y_slice),
                (best_next_h, next_h_slice),
                (best_terminal, terminal_slice),
            ):
                subset = policy_array[:, :, compatible_heading_indices]
                policy_array[:, :, compatible_heading_indices] = np.where(
                    improve, selected_value[:, :, None], subset,
                )
        finalize = active & np.isfinite(best_cost)
        value[:, :, h_index, :] = np.where(
            finalize, best_cost, value[:, :, h_index, :],
        )
        policy_velocity[:, :, h_index, :] = np.where(
            finalize, best_velocity, policy_velocity[:, :, h_index, :],
        )
        policy_gamma[:, :, h_index, :] = np.where(
            finalize, best_gamma, policy_gamma[:, :, h_index, :],
        )
        policy_heading[:, :, h_index, :] = np.where(
            finalize, best_heading, policy_heading[:, :, h_index, :],
        )
        policy_next_x[:, :, h_index, :] = np.where(
            finalize, best_next_x, policy_next_x[:, :, h_index, :],
        )
        policy_next_y[:, :, h_index, :] = np.where(
            finalize, best_next_y, policy_next_y[:, :, h_index, :],
        )
        policy_next_h[:, :, h_index, :] = np.where(
            finalize, best_next_h, policy_next_h[:, :, h_index, :],
        )
        policy_terminal[:, :, h_index, :] = np.where(
            finalize, best_terminal, policy_terminal[:, :, h_index, :],
        )
        updated_state_count += int(np.count_nonzero(finalize))
    return {
        "value": _readonly(value),
        "policy_velocity_index": _readonly(policy_velocity),
        "policy_gamma_index": _readonly(policy_gamma),
        "policy_heading_index": _readonly(policy_heading),
        "policy_next_x_index": _readonly(policy_next_x),
        "policy_next_y_index": _readonly(policy_next_y),
        "policy_next_h_index": _readonly(policy_next_h),
        "policy_terminal": _readonly(policy_terminal),
        "goal_mask": _readonly(goal_mask),
        "diagnostics": {
            "exploration_ordering": exploration_ordering,
            "converged": True,
            "sweep_count": 1,
            "maximum_sweeps": bellman_config["maximum_iterations"],
            "finite_value_state_count": int(np.count_nonzero(np.isfinite(value))),
            "updated_state_count": updated_state_count,
            "acyclic_forward_transition": True,
            "primary_sweep_axis": "h_ascending",
            "local_cost_source": "j6d",
            "coarse_step_count": coarse_step_count,
            "state_axis_order": ("x", "y", "h", "heading"),
            "max_turn_rate_deg_s": vehicle_config["turn_dynamics"][
                "max_turn_rate_deg_s"
            ],
            "maximum_heading_change_per_transition_deg": float(
                np.rad2deg(max_turn_rate * transition_duration)
            ),
            "allowed_heading_transition_count": int(np.count_nonzero(turn_mask)),
        },
    }


def _compute_glide_pod_to_go(
    policy: dict[str, Any],
    transitions: dict[str, np.ndarray],
    glide_detection_rate: np.ndarray,
    time_step: float,
    coarse_step_count: int,
) -> np.ndarray:
    """Replay the fixed Bellman-optimal policy accumulating hazard, not cost.

    Vectorized per h-slice ascending, mirroring `solve_coarse_bellman`: this
    performs no additional optimization, it follows the same policy
    pointers already computed there.
    """
    value = policy["value"]
    x_size, y_size, h_size, heading_state_size = value.shape
    hazard_to_go = np.full(value.shape, np.nan, dtype=float)
    x_idx, y_idx, _ = np.meshgrid(
        np.arange(x_size),
        np.arange(y_size),
        np.arange(heading_state_size),
        indexing="ij",
    )
    v_size, g_size, hd_size = glide_detection_rate.shape[3:]
    for h_index in range(h_size):
        goal_slice = policy["goal_mask"][:, :, h_index]
        goal_state_slice = goal_slice[:, :, None]
        hazard_to_go[:, :, h_index, :] = np.where(
            goal_state_slice, 0.0, hazard_to_go[:, :, h_index, :],
        )
        finite_slice = (
            np.isfinite(value[:, :, h_index, :]) & ~goal_state_slice
        )
        if not np.any(finite_slice):
            continue
        velocity_index = np.clip(
            policy["policy_velocity_index"][:, :, h_index, :], 0, v_size - 1,
        )
        gamma_index = np.clip(
            policy["policy_gamma_index"][:, :, h_index, :], 0, g_size - 1,
        )
        heading_index = np.clip(
            policy["policy_heading_index"][:, :, h_index, :], 0, hd_size - 1,
        )
        terminal_slice = policy["policy_terminal"][:, :, h_index, :]
        rate_slice = glide_detection_rate[
            x_idx, y_idx, h_index, velocity_index, gamma_index, heading_index
        ]
        fraction_slice = np.where(
            terminal_slice,
            transitions["terminal_fraction"][
                x_idx, y_idx, h_index, velocity_index, gamma_index, heading_index
            ],
            1.0,
        )
        local_hazard = rate_slice * coarse_step_count * time_step * fraction_slice
        next_x = np.clip(
            policy["policy_next_x_index"][:, :, h_index, :], 0, x_size - 1,
        )
        next_y = np.clip(
            policy["policy_next_y_index"][:, :, h_index, :], 0, y_size - 1,
        )
        next_h = np.clip(
            policy["policy_next_h_index"][:, :, h_index, :], 0, h_size - 1,
        )
        downstream = np.where(
            terminal_slice,
            0.0,
            hazard_to_go[next_x, next_y, next_h, heading_index],
        )
        computed = local_hazard + downstream
        hazard_to_go[:, :, h_index, :] = np.where(
            finite_slice, computed, hazard_to_go[:, :, h_index, :]
        )
    return 1.0 - np.exp(-hazard_to_go)


def evaluate_powered_segment(
    switching_point: np.ndarray,
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    grids: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Evaluate straight powered flight from launch to one switching seed."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    bellman = configs["bellman_config"]
    validation_config = configs["validation_config"]
    geometry = geometry_bundle["primary_result"]
    functions = detection_bundle["primary_result"]["functions"]
    terrain_model_for_launch = geometry["terrain_model"]
    # h_start is configured as a literal 0.0, unlike h_goal/h_sensor which
    # are always terrain-following -- with p1b_4D's narrow hill (width=200)
    # that gap was invisible (terrain near launch was ~0 either way), but
    # this scenario's wider hill (width=500, per the notebook's config
    # correction) has a long enough Gaussian tail that terrain at (x_start,
    # y_start) is genuinely ~1.1 m, not 0. Using the literal config value
    # buries launch under terrain by that amount, failing every powered
    # segment's very first sample point regardless of switching point
    # (confirmed: 0 of tens of thousands of candidates passed). Launch
    # altitude is derived here the same way goal/sensor altitude already
    # are, rather than changed in configuration.py, since this is the only
    # consumer that treats it as a physical position.
    launch_h = float(
        terrain_height(
            terrain_model_for_launch, environment["x_start"], environment["y_start"]
        )
    )
    launch = np.array([environment["x_start"], environment["y_start"], launch_h])
    delta = np.asarray(switching_point, dtype=float) - launch
    distance = float(np.linalg.norm(delta))
    powered_time = distance / vehicle["powered_speed"]
    sample_count = bellman["search_options"]["segment_check_count"]
    fractions = np.linspace(0.0, 1.0, sample_count)
    path = launch[None, :] + fractions[:, None] * delta[None, :]
    terrain_model = geometry["terrain_model"]
    terrain_margin = path[:, 2] - terrain_height(terrain_model, path[:, 0], path[:, 1])
    # LOS remains a diagnostic, but the default model no longer hard-rejects
    # visible powered flight.  Its visible radar/Doppler exposure is included
    # in powered hazard below, so shadowed and non-shadowed switch candidates
    # can compete on the same physical objective without a radar-free loophole.
    hidden = _grid_lookup(
        geometry["los_masks"]["non_visible_airspace_mask"],
        path[:, 0], path[:, 1], path[:, 2], grids,
    )
    embedded = _grid_lookup(
        geometry["los_masks"]["terrain_mask"],
        path[:, 0], path[:, 1], path[:, 2], grids,
    )
    sensor_position = geometry["sensor_position"]
    horizontal = float(np.hypot(delta[0], delta[1]))
    powered_gamma = float(np.arctan2(delta[2], horizontal))
    powered_heading = float(np.arctan2(delta[1], delta[0]))
    visibility_handling = bellman["search_options"].get(
        "powered_visibility_handling", "hard_hidden",
    )
    if visibility_handling == "hazard_penalty":
        powered_function = functions[
            "powered_total_detection_components"
        ].map(sample_count)
        outputs = _mapped_outputs(
            powered_function,
            path[:, 0].reshape(1, sample_count),
            path[:, 1].reshape(1, sample_count),
            path[:, 2].reshape(1, sample_count),
            np.full((1, sample_count), vehicle["powered_speed"]),
            np.full((1, sample_count), powered_gamma),
            np.full((1, sample_count), powered_heading),
            np.full((1, sample_count), sensor_position[0]),
            np.full((1, sample_count), sensor_position[1]),
            np.full((1, sample_count), sensor_position[2]),
        )
    elif visibility_handling == "hard_hidden":
        powered_function = functions["powered_detection_components"].map(sample_count)
        outputs = _mapped_outputs(
            powered_function,
            path[:, 0].reshape(1, sample_count),
            path[:, 1].reshape(1, sample_count),
            path[:, 2].reshape(1, sample_count),
            np.full((1, sample_count), vehicle["powered_speed"]),
            np.full((1, sample_count), sensor_position[0]),
            np.full((1, sample_count), sensor_position[1]),
            np.full((1, sample_count), sensor_position[2]),
        )
    else:
        raise ValueError(
            "powered_visibility_handling must be 'hazard_penalty' or 'hard_hidden'"
        )
    powered_detection_rate = outputs[1].reshape(sample_count)
    sample_times = fractions * powered_time
    powered_hazard = (
        float(np.trapezoid(powered_detection_rate, sample_times))
        if powered_time > 0.0
        else 0.0
    )
    objective = _function_outputs(
        functions["attacker_objective"], powered_hazard, 0.0, powered_time, 0.0,
    )
    terrain_clear = bool(np.all(terrain_margin >= -validation_config["terrain_tolerance"]))
    hidden_valid = bool(np.all(hidden[1:] & ~embedded[1:]))
    passed = terrain_clear and (
        hidden_valid if visibility_handling == "hard_hidden" else True
    )
    return {
        "path": _readonly(path),
        "powered_time": powered_time,
        "powered_hazard": powered_hazard,
        "powered_pod": objective[0],
        "powered_cost": objective[-1],
        "validation": {
            "passed": passed,
            "terrain_clear": terrain_clear,
            "hidden_valid": hidden_valid,
            "visibility_handling": visibility_handling,
            "powered_gamma": powered_gamma,
            "powered_heading": powered_heading,
            "minimum_terrain_margin": float(np.min(terrain_margin)),
            "summary": (
                "Powered segment feasible"
                if passed
                else "Powered segment violates terrain or occlusion constraints"
            ),
        },
    }


def extract_coarse_candidate(
    switching_point: np.ndarray,
    start_index: tuple[int, int, int, int],
    policy: dict[str, Any],
    transitions: dict[str, np.ndarray],
    stage_cost_6d_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    powered: dict[str, Any],
) -> dict[str, Any]:
    """Extract one coarse path and its physical/objective diagnostics."""
    stage = stage_cost_6d_bundle["primary_result"]
    grids = stage["grids"]
    j6d = stage["j6d"]
    components = stage["component_maps"]
    time_step = configuration_bundle["primary_result"]["vehicle_config"]["time_step"]
    coarse_step_count = transitions["coarse_step_count"]
    effective_dt = coarse_step_count * time_step
    simulation = configuration_bundle["primary_result"]["environment_config"]["simulation"]
    functions = detection_bundle["primary_result"]["functions"]
    x_index, y_index, h_index, heading_state_index = start_index
    initial_heading_state = float(grids["heading"][heading_state_index])
    if not np.isfinite(
        policy["value"][x_index, y_index, h_index, heading_state_index]
    ):
        return {"success": False, "diagnostic": "no_finite_bellman_value"}

    trajectory = [np.asarray(switching_point, dtype=float)]
    speeds: list[float] = []
    gammas: list[float] = []
    headings: list[float] = []
    action_indices: list[tuple[int, int, int, int, int, int]] = []
    glide_stage_costs: list[float] = []
    segment_fractions: list[float] = []
    glide_hazard = 0.0
    visited: set[tuple[int, int, int, int]] = set()
    reached_goal = False
    for _ in range(simulation["max_path_steps"]):
        state_index = (x_index, y_index, h_index, heading_state_index)
        if state_index in visited:
            return {"success": False, "diagnostic": "policy_cycle_detected"}
        visited.add(state_index)
        velocity_index = int(policy["policy_velocity_index"][state_index])
        gamma_index = int(policy["policy_gamma_index"][state_index])
        heading_index = int(policy["policy_heading_index"][state_index])
        if velocity_index < 0 or gamma_index < 0 or heading_index < 0:
            return {"success": False, "diagnostic": "missing_policy_action"}
        speed = float(grids["v"][velocity_index])
        gamma = float(grids["gamma"][gamma_index])
        heading = float(grids["heading"][heading_index])
        next_x = int(policy["policy_next_x_index"][state_index])
        next_y = int(policy["policy_next_y_index"][state_index])
        next_h = int(policy["policy_next_h_index"][state_index])
        terminal = bool(policy["policy_terminal"][state_index])
        action_key = (x_index, y_index, h_index, velocity_index, gamma_index, heading_index)
        segment_fraction = (
            float(transitions["terminal_fraction"][action_key])
            if terminal
            else 1.0
        )
        local_cost = coarse_step_count * float(j6d[action_key]) * segment_fraction
        glide_hazard += float(
            components["glide_detection_rate"][action_key]
            * coarse_step_count * time_step * segment_fraction
        )
        speeds.append(speed)
        gammas.append(gamma)
        headings.append(heading)
        action_indices.append(action_key)
        glide_stage_costs.append(local_cost)
        segment_fractions.append(segment_fraction)
        if terminal:
            current = np.array(
                [grids["x"][x_index], grids["y"][y_index], grids["h"][h_index]]
            )
            velocity_vector = np.array(
                [
                    speed * np.cos(gamma) * np.cos(heading),
                    speed * np.cos(gamma) * np.sin(heading),
                    speed * np.sin(gamma),
                ]
            )
            terminal_point = current + segment_fraction * effective_dt * velocity_vector
            trajectory.append(terminal_point)
            reached_goal = True
            break
        trajectory.append(
            np.array([grids["x"][next_x], grids["y"][next_y], grids["h"][next_h]])
        )
        x_index, y_index, h_index = next_x, next_y, next_h
        heading_state_index = heading_index
    if not reached_goal:
        return {"success": False, "diagnostic": "goal_not_reached"}

    glide_time = float(np.sum(segment_fractions)) * effective_dt
    mission_detection = _function_outputs(
        functions["mission_detection"], powered["powered_hazard"], glide_hazard,
    )
    glide_topology_cost = float(np.sum(glide_stage_costs))
    mission_objective = _function_outputs(
        functions["attacker_objective"],
        powered["powered_hazard"], glide_hazard, powered["powered_time"], glide_time,
    )
    mission_cost = mission_objective[-1]
    trajectory_array = np.asarray(trajectory)
    speed_array = np.asarray(speeds)
    gamma_array = np.asarray(gammas)
    heading_array = np.asarray(headings)
    validation = validate_bellman_candidate(
        switching_point,
        trajectory_array,
        speed_array,
        gamma_array,
        heading_array,
        initial_heading_state,
        effective_dt,
        mission_cost,
        powered,
        glide_topology_cost,
        policy["value"][start_index],
        mission_objective[-1],
        configuration_bundle,
        geometry_bundle,
        reached_goal,
    )
    return {
        "success": validation["passed"],
        "diagnostic": validation["summary"],
        "candidate": {
            "candidate_id": None,
            "start_id": None,
            "switching_point": _readonly(np.asarray(switching_point, dtype=float)),
            "trajectory": _readonly(trajectory_array),
            "speed_profile": _readonly(speed_array),
            "gamma_profile": _readonly(gamma_array),
            "heading_profile": _readonly(heading_array),
            "initial_heading_state": initial_heading_state,
            "mission_cost": mission_cost,
            "objective_breakdown": {
                "powered_cost_diagnostic": powered["powered_cost"],
                "glide_bellman_topology_cost": glide_topology_cost,
                "bellman_local_stage_sum": (
                    powered["powered_cost"] + glide_topology_cost
                ),
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
            "metadata": {
                "coarse": True,
                "warm_start_only": True,
                "is_final_attacker_solution": False,
                "local_cost_source": "j6d",
                "action_indices": tuple(action_indices),
                "segment_fractions": tuple(segment_fractions),
                "goal_region_radius": configuration_bundle["primary_result"][
                    "validation_config"
                ]["goal_radius"],
                "heading_state_axis_added": True,
                "turn_dynamics_model": configuration_bundle["primary_result"][
                    "vehicle_config"
                ]["turn_dynamics"]["model"],
            },
            "validation": validation,
        },
    }


def validate_bellman_candidate(
    switching_point: np.ndarray,
    trajectory: np.ndarray,
    speed_profile: np.ndarray,
    gamma_profile: np.ndarray,
    heading_profile: np.ndarray,
    initial_heading_state: float,
    transition_duration: float,
    mission_cost: float,
    powered: dict[str, Any],
    glide_topology_cost: float,
    bellman_value: float,
    recomputed_mission_objective: float,
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    reached_goal: bool,
) -> dict[str, Any]:
    """Validate one coarse candidate without filtering or ranking it."""
    validation = configuration_bundle["primary_result"]["validation_config"]
    goal_position = geometry_bundle["primary_result"]["goal_position"]
    terrain_model = geometry_bundle["primary_result"]["terrain_model"]
    terrain_margin = trajectory[:, 2] - terrain_height(
        terrain_model, trajectory[:, 0], trajectory[:, 1]
    )
    objective_residual = abs(mission_cost - recomputed_mission_objective)
    combined_local_residual = abs(
        powered["powered_cost"] + glide_topology_cost - mission_cost
    )
    goal_distance = float(np.linalg.norm(trajectory[-1] - goal_position))
    delta_h = np.diff(trajectory[:, 2])
    vehicle = configuration_bundle["primary_result"]["vehicle_config"]
    max_turn_rate = vehicle["turn_dynamics"]["max_turn_rate_deg_s"]
    turn_metrics = heading_change_metrics(
        initial_heading_state, heading_profile, transition_duration,
    )
    checks = {
        "goal_reached": (
            reached_goal
            and goal_distance
            <= validation["goal_radius"] + validation["solver_tolerance"]
        ),
        "strictly_descending_altitude": bool(np.all(delta_h < 0.0)),
        "terrain_clearance": bool(
            np.all(terrain_margin[:-1] >= -validation["terrain_tolerance"])
        ),
        "powered_feasibility": powered["validation"]["passed"],
        "switching_consistency": bool(
            np.allclose(trajectory[0], switching_point, rtol=0.0, atol=0.0)
        ),
        "profile_dimensions": (
            speed_profile.size == gamma_profile.size == heading_profile.size
            == trajectory.shape[0] - 1
        ),
        "turn_rate_limit": (
            turn_metrics["maximum_turn_rate_deg_s"]
            <= max_turn_rate + 1.0e-10
        ),
        "objective_consistency": (
            objective_residual <= validation["objective_tolerance"]
            and combined_local_residual <= validation["objective_tolerance"]
            and abs(glide_topology_cost - bellman_value)
            <= validation["objective_tolerance"]
        ),
        "bellman_convergence": np.isfinite(bellman_value),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "minimum_terrain_margin": float(np.min(terrain_margin[:-1])),
            "goal_distance": goal_distance,
            "minimum_delta_h": float(np.min(delta_h)) if delta_h.size else float("nan"),
            "switching_altitude": float(switching_point[2]),
            "objective_residual": float(objective_residual),
            "combined_local_objective_residual": float(combined_local_residual),
            "glide_value_residual": float(abs(glide_topology_cost - bellman_value)),
            "path_node_count": int(trajectory.shape[0]),
            **turn_metrics,
            "configured_max_turn_rate_deg_s": max_turn_rate,
            "transition_duration_s": transition_duration,
        },
        "failed_checks": failed_checks,
        "summary": (
            "Coarse Bellman candidate validation passed"
            if not failed_checks
            else f"Candidate failed checks: {failed_checks}"
        ),
    }


def validate_bellman_candidate_set(
    candidates: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    stage_cost_6d_bundle: dict[str, Any],
    pod_to_go_maps: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validate the complete unfiltered multi-start candidate set."""
    objective_id = configuration_bundle["primary_result"]["cost_config"][
        "attacker"
    ]["objective_id"]
    attempted_grid_cells = [attempt["grid_start_index"] for attempt in attempts]
    checks = {
        "all_starts_attempted": len(attempts) > 0,
        "switching_grid_cells_exhaustive_and_unique": (
            len(set(attempted_grid_cells)) == len(attempted_grid_cells)
        ),
        "feasible_candidates_generated": len(candidates) > 1,
        "all_candidates_valid": all(
            candidate["validation"]["passed"] for candidate in candidates
        ),
        "candidate_ids_unique": len(
            {candidate["candidate_id"] for candidate in candidates}
        )
        == len(candidates),
        "no_filtering": True,
        "no_ranking": True,
        "bellman_converged": all(
            policy["diagnostics"]["converged"] for policy in policies.values()
        ),
        "j6d_source": (
            stage_cost_6d_bundle["metadata"]["attacker_objective_id"] == objective_id
        ),
        "objective_weights_unchanged": (
            configuration_bundle["primary_result"]["bellman_config"][
                "attacker_objective_id"
            ]
            == objective_id
        ),
    }
    ordering_agreement = _ordering_value_agreement(
        policies,
        configuration_bundle["primary_result"]["validation_config"][
            "objective_tolerance"
        ],
    )
    checks["ordering_value_agreement"] = ordering_agreement["agree"]
    primary_pod_to_go = pod_to_go_maps[next(iter(policies))]
    primary_finite = np.isfinite(policies[next(iter(policies))]["value"])
    finite_pod = primary_pod_to_go[primary_finite]
    checks["pod_to_go_bounded_unit_interval"] = bool(
        finite_pod.size == 0
        or np.all((finite_pod >= 0.0) & (finite_pod <= 1.0))
    )
    checks["pod_to_go_matches_cost_to_go_support"] = bool(
        np.array_equal(np.isfinite(primary_pod_to_go), primary_finite)
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    failed_attempt_count = sum(not attempt["success"] for attempt in attempts)
    warnings = (
        [f"{failed_attempt_count} switching starts did not produce candidates"]
        if failed_attempt_count
        else []
    )
    mission_costs = np.asarray(
        [candidate["mission_cost"] for candidate in candidates], dtype=float
    )
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "attempted_start_count": len(attempts),
            "candidate_count": len(candidates),
            "failed_start_count": failed_attempt_count,
            "minimum_candidate_cost": (
                float(np.min(mission_costs)) if mission_costs.size else np.inf
            ),
            "maximum_candidate_cost": (
                float(np.max(mission_costs)) if mission_costs.size else np.inf
            ),
            "exploration_ordering_count": len(policies),
            "ordering_maximum_value_disagreement": ordering_agreement["max_diff"],
        },
        "tolerances": {
            "terrain": configuration_bundle["primary_result"][
                "validation_config"
            ]["terrain_tolerance"],
            "goal_radius": configuration_bundle["primary_result"][
                "validation_config"
            ]["goal_radius"],
        },
        "warnings": warnings,
        "failed_checks": failed_checks,
        "summary": (
            "Phase 6 multi-start Bellman candidate validation passed"
            if not failed_checks
            else f"Bellman candidate set failed checks: {failed_checks}"
        ),
    }


def _ordering_value_agreement(
    policies: dict[str, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Compare converged cost-to-go maps across exploration orderings."""
    values = list(policies.values())
    max_diff = 0.0
    for first_index in range(len(values)):
        for second_index in range(first_index + 1, len(values)):
            first_value = values[first_index]["value"]
            second_value = values[second_index]["value"]
            both_finite = np.isfinite(first_value) & np.isfinite(second_value)
            if np.any(both_finite):
                max_diff = max(
                    max_diff,
                    float(np.max(np.abs(
                        first_value[both_finite] - second_value[both_finite]
                    ))),
                )
    return {"agree": max_diff <= tolerance, "max_diff": max_diff}


def select_authoritative_bellman_response(
    bellman_candidate_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Select the Bellman-optimal Attacker response from the candidate set.

    This is the sole authoritative Attacker best response. It performs no
    optimization of its own: it deterministically selects the minimum-cost
    member of the already-converged, already-validated `generate_bellman_
    candidates` output and re-exposes its fields under a stable schema.

    Optimality is scoped to the discretized switching-point seed grid,
    heading-state grid, and (velocity, gamma, selected-course) action grid
    used by Bellman; no continuous global optimum is claimed.
    """
    _require_successful_bundle(bellman_candidate_bundle, "bellman_candidate_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    result = bellman_candidate_bundle["primary_result"]
    candidates = list(result["candidates"])
    if not candidates:
        raise ValueError("BellmanCandidateSet contains no feasible candidates")

    tolerance = configuration_bundle["primary_result"]["validation_config"][
        "objective_tolerance"
    ]
    ordered = sorted(
        candidates, key=lambda candidate: (candidate["mission_cost"], candidate["candidate_id"])
    )
    best = ordered[0]
    tied = [
        candidate
        for candidate in ordered
        if abs(candidate["mission_cost"] - best["mission_cost"]) <= tolerance
    ]
    hazard_breakdown = best["hazard_breakdown"]
    goal_error = best["trajectory"][-1] - np.array(
        bellman_candidate_bundle["metadata"]["goal_position"]
    )

    primary_result = {
        "solution_id": f"bellman-optimal-{best['candidate_id']}",
        "source_candidate_id": best["candidate_id"],
        "source_start_id": best["start_id"],
        "candidate_count_searched": len(candidates),
        "tie_count": len(tied),
        "switching_point": best["switching_point"],
        "trajectory": best["trajectory"],
        "speed_profile": best["speed_profile"],
        "gamma_profile": best["gamma_profile"],
        "heading_profile": best["heading_profile"],
        "initial_heading_state": best["initial_heading_state"],
        "mission_cost": best["mission_cost"],
        "mission_objective": best["mission_cost"],
        "objective_breakdown": best["objective_breakdown"],
        "powered_time": best["powered_time"],
        "glide_time": best["glide_time"],
        "mission_time": best["mission_time"],
        "mission_pod": best["mission_pod"],
        "hazard_breakdown": hazard_breakdown,
        "powered_hazard": hazard_breakdown["powered_acoustic_hazard"],
        "glide_hazard": hazard_breakdown["glide_radar_doppler_hazard"],
        "powered_path": best["powered_path"],
        "constraint_residuals": {
            "goal_error": _readonly(goal_error),
            "goal_error_norm": float(np.linalg.norm(goal_error)),
            "minimum_terrain_margin": best["validation"]["metrics"][
                "minimum_terrain_margin"
            ],
            "maximum_turn_rate_deg_s": best["validation"]["metrics"][
                "maximum_turn_rate_deg_s"
            ],
            "configured_max_turn_rate_deg_s": best["validation"]["metrics"][
                "configured_max_turn_rate_deg_s"
            ],
        },
        "metadata": {
            **best["metadata"],
            "coarse": False,
            "warm_start_only": False,
            "is_final_attacker_solution": True,
        },
        "validation": best["validation"],
    }
    validation = validate_authoritative_bellman_response(
        best, ordered, tied, bellman_candidate_bundle, configuration_bundle
    )
    return {
        "primary_result": primary_result,
        "validation": validation,
        "metadata": {
            "schema_name": "AuthoritativeBellmanAttackerResponse3D",
            "schema_version": "1.1.0",
            "producer_phase": 8,
            "producer_module": "p1b_3DExtension.bellman",
            "solution_method": "bellman_dynamic_programming",
            "optimality_scope": (
                "discretized_switching_point_heading_state_and_action_grid"
            ),
            "attacker_objective_id": configuration_bundle["primary_result"][
                "cost_config"
            ]["attacker"]["objective_id"],
            "is_final_attacker_solution": True,
            "global_optimum_claim": False,
            "selection_rule": "minimum_mission_cost_among_bellman_candidates",
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "BELLMAN_RESPONSE_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def validate_authoritative_bellman_response(
    best: dict[str, Any],
    ordered_candidates: list[dict[str, Any]],
    tied_candidates: list[dict[str, Any]],
    bellman_candidate_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Validate the selected response against the full candidate set and grid."""
    tie_break_ok = tied_candidates[0]["candidate_id"] == best["candidate_id"]
    checks = {
        "bellman_candidate_set_valid": bellman_candidate_bundle["status"]["success"],
        "selected_candidate_valid": best["validation"]["passed"],
        "selection_is_minimum_cost": best["mission_cost"]
        == ordered_candidates[0]["mission_cost"],
        "deterministic_tie_break_applied": tie_break_ok,
        "objective_matches_bellman_value": best["validation"]["checks"][
            "objective_consistency"
        ],
        "goal_reached": best["validation"]["checks"]["goal_reached"],
        "terrain_clearance": best["validation"]["checks"]["terrain_clearance"],
        "turn_rate_limit": best["validation"]["checks"]["turn_rate_limit"],
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    warnings = (
        [f"{len(tied_candidates)} candidates tied within objective_tolerance"]
        if len(tied_candidates) > 1
        else []
    )
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "selected_mission_cost": best["mission_cost"],
            "candidate_count": len(ordered_candidates),
            "tie_count": len(tied_candidates),
            "minimum_candidate_cost": ordered_candidates[0]["mission_cost"],
            "maximum_candidate_cost": ordered_candidates[-1]["mission_cost"],
        },
        "warnings": warnings,
        "failed_checks": failed_checks,
        "summary": (
            "Authoritative Bellman Attacker response validation passed"
            if not failed_checks
            else f"Authoritative Bellman response failed checks: {failed_checks}"
        ),
    }


def _switching_grid_index(
    switching_point: np.ndarray,
    grids: dict[str, np.ndarray],
) -> tuple[int, int, int]:
    x_index = int(np.argmin(np.abs(grids["x"] - switching_point[0])))
    y_index = int(np.argmin(np.abs(grids["y"] - switching_point[1])))
    h_index = int(np.argmin(np.abs(grids["h"] - switching_point[2])))
    return x_index, y_index, h_index


def _ordered_actions(
    v_grid: np.ndarray,
    gamma_grid: np.ndarray,
    heading_grid: np.ndarray,
    ordering: str,
) -> tuple[tuple[int, int, int], ...]:
    actions = [
        (velocity_index, gamma_index, heading_index)
        for velocity_index in range(v_grid.size)
        for gamma_index in range(gamma_grid.size)
        for heading_index in range(heading_grid.size)
    ]
    if ordering == "low_gamma_first":
        key = lambda item: (gamma_grid[item[1]], v_grid[item[0]], heading_grid[item[2]])
    elif ordering == "high_gamma_first":
        key = lambda item: (-gamma_grid[item[1]], v_grid[item[0]], heading_grid[item[2]])
    elif ordering == "low_speed_first":
        key = lambda item: (v_grid[item[0]], gamma_grid[item[1]], heading_grid[item[2]])
    elif ordering == "high_speed_first":
        key = lambda item: (-v_grid[item[0]], gamma_grid[item[1]], heading_grid[item[2]])
    else:
        raise ValueError(f"Unknown exploration ordering: {ordering}")
    return tuple(sorted(actions, key=key))


def _grid_lookup(
    mask: np.ndarray,
    x: np.ndarray, y: np.ndarray, h: np.ndarray,
    grids: dict[str, np.ndarray],
) -> np.ndarray:
    """Nearest-grid-cell boolean lookup for continuous (x, y, h) points."""
    x_grid, y_grid, h_grid = grids["x"], grids["y"], grids["h"]
    x_index = np.clip(
        np.rint((x - x_grid[0]) / (x_grid[1] - x_grid[0])).astype(np.int64),
        0, x_grid.size - 1,
    )
    y_index = np.clip(
        np.rint((y - y_grid[0]) / (y_grid[1] - y_grid[0])).astype(np.int64),
        0, y_grid.size - 1,
    )
    h_index = np.clip(
        np.rint((h - h_grid[0]) / (h_grid[1] - h_grid[0])).astype(np.int64),
        0, h_grid.size - 1,
    )
    return mask[x_index, y_index, h_index]


def _mapped_outputs(function: ca.Function, *arguments: np.ndarray) -> list[np.ndarray]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [np.asarray(value, dtype=float) for value in outputs]


def _function_outputs(function: ca.Function, *arguments: float) -> list[float]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [float(value) for value in outputs]


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict):
        raise TypeError(f"{name} must be a dictionary")
    if not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must have successful status")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _finite_minimum(values: np.ndarray, axis: int) -> np.ndarray:
    """Minimize finite entries without all-NaN slice warnings."""
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    result = np.min(np.where(finite, array, np.inf), axis=axis)
    return np.where(np.any(finite, axis=axis), result, np.nan)
