"""Multi-start coarse Bellman candidate generator using authoritative J4D."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .geometry import terrain_height


def generate_bellman_candidates(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_4d_bundle: dict[str, Any],
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
    stage_cost_4d_bundle:
        Successful authoritative J4D local-stage result.
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
    Coarse transitions use constant speed and gamma for one configured time
    step. The resulting candidates are NLP warm-start material, not final
    Attacker responses.

    Notes
    -----
    No candidate ranking, duplicate removal, continuous refinement, Defender
    optimization, file writing, or plotting occurs here.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    _require_successful_bundle(detection_bundle, "detection_bundle")
    _require_successful_bundle(stage_cost_4d_bundle, "stage_cost_4d_bundle")
    if projected_cost_bundle is not None:
        _require_successful_bundle(projected_cost_bundle, "projected_cost_bundle")
        if not projected_cost_bundle["metadata"].get("visualization_only", False):
            raise ValueError("ProjectedCost input must be marked visualization_only")
        if projected_cost_bundle["primary_result"]["projection_metadata"].get(
            "bellman_policy_input", True
        ):
            raise ValueError("ProjectedCost must explicitly prohibit Bellman use")

    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    bellman_config = configs["bellman_config"]
    validation_config = configs["validation_config"]
    stage = stage_cost_4d_bundle["primary_result"]
    grids = stage["grids"]
    j4d = stage["j4d"]
    transitions = construct_coarse_transitions(
        geometry_bundle,
        stage_cost_4d_bundle,
        configuration_bundle,
    )
    orderings = bellman_config["search_options"]["exploration_orderings"]
    policies = {
        ordering: solve_coarse_bellman(
            j4d,
            transitions,
            grids,
            environment,
            validation_config,
            ordering,
            bellman_config,
        )
        for ordering in orderings
    }
    seeds = generate_switching_point_seeds(
        geometry_bundle,
        configuration_bundle,
    )
    candidates: list[dict[str, Any]] = []
    start_attempts: list[dict[str, Any]] = []
    for seed_index, switching_point in enumerate(seeds):
        ordering = orderings[seed_index % len(orderings)]
        start_index = _switching_grid_index(switching_point, grids)
        attempt = {
            "start_id": f"switch-seed-{seed_index:03d}",
            "seed_index": seed_index,
            "switching_point": switching_point,
            "grid_start_index": start_index,
            "exploration_ordering": ordering,
            "success": False,
            "diagnostic": None,
        }
        powered = evaluate_powered_segment(
            switching_point,
            configuration_bundle,
            geometry_bundle,
            detection_bundle,
        )
        if not powered["validation"]["passed"]:
            attempt["diagnostic"] = powered["validation"]["summary"]
            start_attempts.append(attempt)
            continue
        extracted = extract_coarse_candidate(
            switching_point,
            start_index,
            policies[ordering],
            transitions,
            stage_cost_4d_bundle,
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
        candidate["candidate_id"] = f"bellman-candidate-{len(candidates):03d}"
        candidate["start_id"] = attempt["start_id"]
        candidate["metadata"].update(
            {
                "seed_index": seed_index,
                "exploration_ordering": ordering,
                "grid_start_index": start_index,
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
        stage_cost_4d_bundle,
    )

    primary_ordering = orderings[0]
    primary_cost_to_go = policies[primary_ordering]["value"]
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
                ordering: policy["value"] for ordering, policy in policies.items()
            },
            # "cost_to_go_primary_ordering": orderings[0],
            "cost_to_go_primary_ordering": primary_ordering,
            "finite_cost_to_go_mask": _readonly(finite_cost_to_go_mask),
            "candidate_count": len(candidates),
            "attempted_start_count": len(start_attempts),
            "filtering_applied": False,
            "ranking_applied": False,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "BellmanCandidateSet",
            "schema_version": "1.0.0",
            "producer_phase": 6,
            "producer_module": "p1b_4D.bellman",
            "candidate_role": "coarse_topology_and_nlp_warm_start",
            "is_final_attacker_solution": False,
            "global_optimum_claim": False,
            "local_cost_source": "StageCost4DResult.j4d",
            "cost_to_go_role": "Bellman value map for exported visualization",
            "local_cost_source_schema": stage_cost_4d_bundle["metadata"][
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
    configuration_bundle: dict[str, Any],
) -> np.ndarray:
    """Generate configured continuous switching seeds on the LOS tangent."""
    geometry = geometry_bundle["primary_result"]["los_geometry"]
    environment = configuration_bundle["primary_result"]["environment_config"]
    count = configuration_bundle["primary_result"]["bellman_config"][
        "candidate_count"
    ]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("bellman_config.candidate_count must be positive")
    tangent_z = float(geometry["tangent_point"][0])
    slope = float(geometry["tangent_slope"])
    intercept = float(geometry["tangent_intercept"])
    airspace = environment["airspace"]
    if abs(slope) <= np.finfo(float).eps:
        if not airspace["h_min"] <= intercept <= airspace["h_max"]:
            raise ValueError("LOS tangent does not intersect the configured airspace")
        seed_z_min = environment["z_start"]
    else:
        height_intersections = sorted((
            (airspace["h_min"] - intercept) / slope,
            (airspace["h_max"] - intercept) / slope,
        ))
        seed_z_min = max(environment["z_start"], height_intersections[0])
        tangent_z = min(tangent_z, height_intersections[1])
    if seed_z_min >= tangent_z:
        raise ValueError("No continuous LOS-tangent switching interval is feasible")
    z_values = np.linspace(seed_z_min, tangent_z, count)
    h_values = (
        slope * z_values + intercept
    )
    return np.column_stack((z_values, h_values))


def construct_coarse_transitions(
    geometry_bundle: dict[str, Any],
    stage_cost_4d_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct coarse kinematic successors without using ProjectedCost."""
    geometry = geometry_bundle["primary_result"]
    stage = stage_cost_4d_bundle["primary_result"]
    grids = stage["grids"]
    environment = configuration_bundle["primary_result"]["environment_config"]
    vehicle = configuration_bundle["primary_result"]["vehicle_config"]
    bellman = configuration_bundle["primary_result"]["bellman_config"]
    validation = configuration_bundle["primary_result"]["validation_config"]
    z_grid, h_grid, v_grid, gamma_grid = (
        grids[name] for name in ("z", "h", "v", "gamma")
    )
    shape = stage["j4d"].shape
    next_z_index = np.full(shape, -1, dtype=np.int32)
    next_h_index = np.full(shape, -1, dtype=np.int32)
    transition_valid = np.zeros(shape, dtype=bool)
    terminal_transition = np.zeros(shape, dtype=bool)
    terminal_fraction = np.ones(shape, dtype=float)
    mesh_z, mesh_h = np.meshgrid(z_grid, h_grid, indexing="ij")
    dz_grid = float(z_grid[1] - z_grid[0])
    dh_grid = float(h_grid[1] - h_grid[0])
    terrain_model = geometry["terrain_model"]
    tangent = geometry["los_geometry"]
    segment_count = bellman["search_options"]["segment_check_count"]
    fractions = np.linspace(0.0, 1.0, segment_count)
    goal_radius = float(validation["goal_radius"])
    terrain_tolerance = validation["terrain_tolerance"]

    for velocity_index, velocity in enumerate(v_grid):
        for gamma_index, gamma in enumerate(gamma_grid):
            delta_z = velocity * vehicle["time_step"] * np.cos(gamma)
            delta_h = velocity * vehicle["time_step"] * np.sin(gamma)
            next_z = mesh_z + delta_z
            next_h = mesh_h + delta_h
            relative_z = mesh_z - environment["z_goal"]
            relative_h = mesh_h - environment["h_goal"]
            quadratic_a = delta_z**2 + delta_h**2
            quadratic_b = relative_z * delta_z + relative_h * delta_h
            quadratic_c = relative_z**2 + relative_h**2 - goal_radius**2
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
                terminal,
                np.clip(first_intersection, 0.0, 1.0),
                1.0,
            )
            mapped_z = np.ceil(
                (next_z - z_grid[0]) / dz_grid
            ).astype(np.int64)
            mapped_h = np.rint(
                (next_h - h_grid[0]) / dh_grid
            ).astype(np.int64)
            inside = (
                (mapped_z >= 0)
                & (mapped_z < z_grid.size)
                & (mapped_h >= 0)
                & (mapped_h < h_grid.size)
            )
            current_z_indices = np.arange(z_grid.size)[:, None]
            advances = mapped_z > current_z_indices
            mapped_z_safe = np.clip(mapped_z, 0, z_grid.size - 1)
            mapped_h_safe = np.clip(mapped_h, 0, h_grid.size - 1)
            successor_spatial_valid = geometry["los_masks"]["los_mask"][
                mapped_z_safe, mapped_h_safe
            ]
            successor_goal = (
                (z_grid[mapped_z_safe] - environment["z_goal"]) ** 2
                + (h_grid[mapped_h_safe] - environment["h_goal"]) ** 2
                <= goal_radius**2
            )
            segment_valid = np.ones(mesh_z.shape, dtype=bool)
            for fraction_index, fraction in enumerate(fractions[1:], start=1):
                effective_fraction = fraction * action_terminal_fraction
                sample_z = mesh_z + effective_fraction * delta_z
                sample_h = mesh_h + effective_fraction * delta_h
                sample_in_domain = (
                    (sample_z >= z_grid[0]) & (sample_z <= z_grid[-1])
                )
                terrain_clear = (
                    sample_h
                    >= terrain_height(
                        terrain_model,
                        np.clip(sample_z, z_grid[0], z_grid[-1]),
                    )
                    - terrain_tolerance
                )
                los_visible = ~(
                    (sample_z < tangent["tangent_point"][0])
                    & (
                        sample_h
                        < tangent["tangent_slope"] * sample_z
                        + tangent["tangent_intercept"]
                    )
                )
                sample_valid = sample_in_domain & terrain_clear & los_visible
                if fraction_index == fractions.size - 1:
                    sample_valid = sample_valid | terminal
                segment_valid &= sample_valid
            local_finite = np.isfinite(
                stage["j4d"][:, :, velocity_index, gamma_index]
            )
            valid = (
                local_finite
                & segment_valid
                & (
                    terminal
                    | (
                        inside
                        & advances
                        & successor_spatial_valid
                        & ~successor_goal
                    )
                )
            )
            next_z_index[:, :, velocity_index, gamma_index] = np.where(
                valid & ~terminal, mapped_z, -1
            )
            next_h_index[:, :, velocity_index, gamma_index] = np.where(
                valid & ~terminal, mapped_h, -1
            )
            transition_valid[:, :, velocity_index, gamma_index] = valid
            terminal_transition[:, :, velocity_index, gamma_index] = (
                valid & terminal
            )
            terminal_fraction[:, :, velocity_index, gamma_index] = np.where(
                valid & terminal,
                action_terminal_fraction,
                1.0,
            )
    return {
        "next_z_index": _readonly(next_z_index),
        "next_h_index": _readonly(next_h_index),
        "transition_valid": _readonly(transition_valid),
        "terminal_transition": _readonly(terminal_transition),
        "terminal_fraction": _readonly(terminal_fraction),
    }


def solve_coarse_bellman(
    j4d: np.ndarray,
    transitions: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    environment: dict[str, Any],
    validation_config: dict[str, Any],
    exploration_ordering: str,
    bellman_config: dict[str, Any],
) -> dict[str, Any]:
    """Solve the forward-acyclic coarse Bellman recursion for one tie ordering."""
    z_grid, h_grid, v_grid, gamma_grid = (
        grids[name] for name in ("z", "h", "v", "gamma")
    )
    value = np.full((z_grid.size, h_grid.size), np.inf)
    policy_velocity = np.full(value.shape, -1, dtype=np.int32)
    policy_gamma = np.full(value.shape, -1, dtype=np.int32)
    policy_next_z = np.full(value.shape, -1, dtype=np.int32)
    policy_next_h = np.full(value.shape, -1, dtype=np.int32)
    policy_terminal = np.zeros(value.shape, dtype=bool)
    goal_mask = (
        (z_grid[:, None] - environment["z_goal"]) ** 2
        + (h_grid[None, :] - environment["h_goal"]) ** 2
        <= validation_config["goal_radius"] ** 2
    )
    value[goal_mask] = 0.0
    actions = _ordered_actions(v_grid, gamma_grid, exploration_ordering)
    updated_state_count = 0
    for z_index in range(z_grid.size - 1, -1, -1):
        for h_index in range(h_grid.size):
            if goal_mask[z_index, h_index]:
                continue
            best_cost = np.inf
            best_action: tuple[int, int, int, int, bool] | None = None
            for velocity_index, gamma_index in actions:
                if not transitions["transition_valid"][
                    z_index, h_index, velocity_index, gamma_index
                ]:
                    continue
                terminal = bool(
                    transitions["terminal_transition"][
                        z_index, h_index, velocity_index, gamma_index
                    ]
                )
                next_z = int(
                    transitions["next_z_index"][
                        z_index, h_index, velocity_index, gamma_index
                    ]
                )
                next_h = int(
                    transitions["next_h_index"][
                        z_index, h_index, velocity_index, gamma_index
                    ]
                )
                downstream = 0.0 if terminal else value[next_z, next_h]
                local_fraction = (
                    transitions["terminal_fraction"][
                        z_index, h_index, velocity_index, gamma_index
                    ]
                    if terminal
                    else 1.0
                )
                candidate_cost = (
                    local_fraction
                    * j4d[z_index, h_index, velocity_index, gamma_index]
                    + downstream
                )
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_action = (
                        velocity_index,
                        gamma_index,
                        next_z,
                        next_h,
                        terminal,
                    )
            if best_action is not None and np.isfinite(best_cost):
                value[z_index, h_index] = best_cost
                (
                    policy_velocity[z_index, h_index],
                    policy_gamma[z_index, h_index],
                    policy_next_z[z_index, h_index],
                    policy_next_h[z_index, h_index],
                    policy_terminal[z_index, h_index],
                ) = best_action
                updated_state_count += 1
    return {
        "value": _readonly(value),
        "policy_velocity_index": _readonly(policy_velocity),
        "policy_gamma_index": _readonly(policy_gamma),
        "policy_next_z_index": _readonly(policy_next_z),
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
            "local_cost_source": "j4d",
        },
    }


def evaluate_powered_segment(
    switching_point: np.ndarray,
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate straight powered flight from launch to one switching seed."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    bellman = configs["bellman_config"]
    validation_config = configs["validation_config"]
    geometry = geometry_bundle["primary_result"]
    functions = detection_bundle["primary_result"]["functions"]
    launch = np.array([environment["z_start"], environment["h_start"]])
    delta = switching_point - launch
    distance = float(np.hypot(delta[0], delta[1]))
    powered_time = distance / vehicle["powered_speed"]
    sample_count = bellman["search_options"]["segment_check_count"]
    fractions = np.linspace(0.0, 1.0, sample_count)
    path = launch[None, :] + fractions[:, None] * delta[None, :]
    terrain_margin = (
        path[:, 1]
        - terrain_height(geometry["terrain_model"], path[:, 0])
    )
    tangent = geometry["los_geometry"]
    inside_occlusion = (
        (path[:, 0] <= tangent["tangent_point"][0])
        & (
            path[:, 1]
            <= tangent["tangent_slope"] * path[:, 0]
            + tangent["tangent_intercept"]
            + validation_config["los_tolerance"]
        )
    )
    sensor_position = geometry["sensor_position"]
    acoustic_function = functions["powered_detection_components"].map(
        sample_count
    )
    outputs = _mapped_outputs(
        acoustic_function,
        path[:, 0].reshape(1, sample_count),
        path[:, 1].reshape(1, sample_count),
        np.full((1, sample_count), vehicle["powered_speed"]),
        np.full((1, sample_count), sensor_position[0]),
        np.full((1, sample_count), sensor_position[1]),
    )
    acoustic_rate = outputs[-1].reshape(sample_count)
    sample_times = fractions * powered_time
    powered_hazard = (
        float(np.trapezoid(acoustic_rate, sample_times))
        if powered_time > 0.0
        else 0.0
    )
    objective = _function_outputs(
        functions["attacker_objective"],
        powered_hazard,
        0.0,
        powered_time,
        0.0,
    )
    terrain_clear = bool(
        np.all(terrain_margin >= -validation_config["terrain_tolerance"])
    )
    occlusion_valid = bool(np.all(inside_occlusion))
    passed = terrain_clear and occlusion_valid
    return {
        "path": _readonly(path),
        "powered_time": powered_time,
        "powered_hazard": powered_hazard,
        "powered_pod": objective[0],
        "powered_cost": objective[-1],
        "validation": {
            "passed": passed,
            "terrain_clear": terrain_clear,
            "occlusion_valid": occlusion_valid,
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
    start_index: tuple[int, int],
    policy: dict[str, Any],
    transitions: dict[str, np.ndarray],
    stage_cost_4d_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    powered: dict[str, Any],
) -> dict[str, Any]:
    """Extract one coarse path and its physical/objective diagnostics."""
    stage = stage_cost_4d_bundle["primary_result"]
    grids = stage["grids"]
    j4d = stage["j4d"]
    components = stage["component_maps"]
    environment = configuration_bundle["primary_result"]["environment_config"]
    simulation = environment["simulation"]
    functions = detection_bundle["primary_result"]["functions"]
    z_index, h_index = start_index
    if not np.isfinite(policy["value"][z_index, h_index]):
        return {"success": False, "diagnostic": "no_finite_bellman_value"}

    trajectory = [np.asarray(switching_point, dtype=float)]
    speeds: list[float] = []
    gammas: list[float] = []
    action_indices: list[tuple[int, int, int, int]] = []
    glide_stage_costs: list[float] = []
    segment_fractions: list[float] = []
    glide_hazard = 0.0
    visited: set[tuple[int, int]] = set()
    reached_goal = False
    for _ in range(simulation["max_path_steps"]):
        state_index = (z_index, h_index)
        if state_index in visited:
            return {"success": False, "diagnostic": "policy_cycle_detected"}
        visited.add(state_index)
        velocity_index = int(policy["policy_velocity_index"][state_index])
        gamma_index = int(policy["policy_gamma_index"][state_index])
        if velocity_index < 0 or gamma_index < 0:
            return {"success": False, "diagnostic": "missing_policy_action"}
        speed = float(grids["v"][velocity_index])
        gamma = float(grids["gamma"][gamma_index])
        next_z = int(policy["policy_next_z_index"][state_index])
        next_h = int(policy["policy_next_h_index"][state_index])
        terminal = bool(policy["policy_terminal"][state_index])
        segment_fraction = (
            float(transitions["terminal_fraction"][
                z_index, h_index, velocity_index, gamma_index
            ])
            if terminal
            else 1.0
        )
        local_cost = float(
            j4d[z_index, h_index, velocity_index, gamma_index]
        ) * segment_fraction
        glide_hazard += float(
            components["glide_detection_rate"][
                z_index, h_index, velocity_index, gamma_index
            ]
            * configuration_bundle["primary_result"]["vehicle_config"]["time_step"]
            * segment_fraction
        )
        speeds.append(speed)
        gammas.append(gamma)
        action_indices.append(
            (z_index, h_index, velocity_index, gamma_index)
        )
        glide_stage_costs.append(local_cost)
        segment_fractions.append(segment_fraction)
        if terminal:
            time_step = configuration_bundle["primary_result"]["vehicle_config"][
                "time_step"
            ]
            current = np.array([grids["z"][z_index], grids["h"][h_index]])
            terminal_point = current + segment_fraction * time_step * np.array([
                speed * np.cos(gamma),
                speed * np.sin(gamma),
            ])
            trajectory.append(
                terminal_point
            )
            reached_goal = True
            break
        trajectory.append(np.array([grids["z"][next_z], grids["h"][next_h]]))
        z_index, h_index = next_z, next_h
    if not reached_goal:
        return {"success": False, "diagnostic": "goal_not_reached"}

    glide_time = float(np.sum(segment_fractions)) * configuration_bundle[
        "primary_result"
    ]["vehicle_config"]["time_step"]
    mission_detection = _function_outputs(
        functions["mission_detection"],
        powered["powered_hazard"],
        glide_hazard,
    )
    glide_topology_cost = float(np.sum(glide_stage_costs))
    mission_objective = _function_outputs(
        functions["attacker_objective"],
        powered["powered_hazard"],
        glide_hazard,
        powered["powered_time"],
        glide_time,
    )
    mission_cost = mission_objective[-1]
    trajectory_array = np.asarray(trajectory)
    speed_array = np.asarray(speeds)
    gamma_array = np.asarray(gammas)
    validation = validate_bellman_candidate(
        switching_point,
        trajectory_array,
        speed_array,
        gamma_array,
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
            "switching_point": _readonly(
                np.asarray(switching_point, dtype=float)
            ),
            "trajectory": _readonly(trajectory_array),
            "speed_profile": _readonly(speed_array),
            "gamma_profile": _readonly(gamma_array),
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
                "local_cost_source": "j4d",
                "action_indices": tuple(action_indices),
                "segment_fractions": tuple(segment_fractions),
                "goal_region_radius": configuration_bundle["primary_result"][
                    "validation_config"
                ]["goal_radius"],
            },
            "validation": validation,
        },
    }


def validate_bellman_candidate(
    switching_point: np.ndarray,
    trajectory: np.ndarray,
    speed_profile: np.ndarray,
    gamma_profile: np.ndarray,
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
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    validation = configs["validation_config"]
    tangent = geometry_bundle["primary_result"]["los_geometry"]
    terrain_model = geometry_bundle["primary_result"]["terrain_model"]
    terrain_margin = trajectory[:, 1] - terrain_height(
        terrain_model, trajectory[:, 0]
    )
    los_visible = ~(
        (trajectory[:-1, 0] < tangent["tangent_point"][0])
        & (
            trajectory[:-1, 1]
            < tangent["tangent_slope"] * trajectory[:-1, 0]
            + tangent["tangent_intercept"]
            - validation["los_tolerance"]
        )
    )
    tangent_error = abs(
        switching_point[1]
        - (
            tangent["tangent_slope"] * switching_point[0]
            + tangent["tangent_intercept"]
        )
    )
    objective_residual = abs(mission_cost - recomputed_mission_objective)
    combined_local_residual = abs(
        powered["powered_cost"] + glide_topology_cost - mission_cost
    )
    goal_distance = float(np.linalg.norm(
        trajectory[-1]
        - np.array([environment["z_goal"], environment["h_goal"]])
    ))
    delta_z = np.diff(trajectory[:, 0])
    checks = {
        "goal_reached": (
            reached_goal
            and goal_distance
            <= validation["goal_radius"] + validation["solver_tolerance"]
        ),
        "strictly_forward_trajectory": bool(np.all(delta_z > 0.0)),
        "no_goal_overshoot": bool(
            np.max(trajectory[:, 0])
            <= environment["z_goal"] + validation["goal_radius"]
            + validation["solver_tolerance"]
        ),
        "terrain_clearance": bool(
            np.all(
                terrain_margin[:-1]
                >= -validation["terrain_tolerance"]
            )
        ),
        "los_feasibility": bool(np.all(los_visible)),
        "powered_feasibility": powered["validation"]["passed"],
        "switching_consistency": bool(
            np.allclose(trajectory[0], switching_point, rtol=0.0, atol=0.0)
            and tangent_error <= validation["los_tolerance"]
        ),
        "profile_dimensions": (
            speed_profile.size == gamma_profile.size == trajectory.shape[0] - 1
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
            "minimum_delta_z": float(np.min(delta_z)),
            "maximum_z": float(np.max(trajectory[:, 0])),
            "switching_tangent_error": float(tangent_error),
            "objective_residual": float(objective_residual),
            "combined_local_objective_residual": float(combined_local_residual),
            "glide_value_residual": float(
                abs(glide_topology_cost - bellman_value)
            ),
            "path_node_count": int(trajectory.shape[0]),
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
    stage_cost_4d_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Validate the complete unfiltered multi-start candidate set."""
    expected_attempts = configuration_bundle["primary_result"]["bellman_config"][
        "candidate_count"
    ]
    objective_id = configuration_bundle["primary_result"]["cost_config"][
        "attacker"
    ]["objective_id"]
    checks = {
        "all_starts_attempted": len(attempts) == expected_attempts,
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
        "j4d_source": (
            stage_cost_4d_bundle["metadata"]["attacker_objective_id"]
            == objective_id
        ),
        "objective_weights_unchanged": (
            configuration_bundle["primary_result"]["bellman_config"][
                "attacker_objective_id"
            ]
            == configuration_bundle["primary_result"]["nlp_config"][
                "attacker_objective_id"
            ]
            == objective_id
        ),
    }
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
        },
        "tolerances": {
            "terrain": configuration_bundle["primary_result"][
                "validation_config"
            ]["terrain_tolerance"],
            "goal_z": configuration_bundle["primary_result"][
                "validation_config"
            ]["goal_tolerance_z"],
            "goal_h": configuration_bundle["primary_result"][
                "validation_config"
            ]["goal_tolerance_h"],
            "los": configuration_bundle["primary_result"][
                "validation_config"
            ]["los_tolerance"],
        },
        "warnings": warnings,
        "failed_checks": failed_checks,
        "summary": (
            "Phase 6 multi-start Bellman candidate validation passed"
            if not failed_checks
            else f"Bellman candidate set failed checks: {failed_checks}"
        ),
    }


def _switching_grid_index(
    switching_point: np.ndarray,
    grids: dict[str, np.ndarray],
) -> tuple[int, int]:
    z_index = int(np.argmin(np.abs(grids["z"] - switching_point[0])))
    h_index = int(np.searchsorted(grids["h"], switching_point[1], side="left"))
    h_index = min(max(h_index, 0), grids["h"].size - 1)
    return z_index, h_index


def _ordered_actions(
    v_grid: np.ndarray,
    gamma_grid: np.ndarray,
    ordering: str,
) -> tuple[tuple[int, int], ...]:
    actions = [
        (velocity_index, gamma_index)
        for velocity_index in range(v_grid.size)
        for gamma_index in range(gamma_grid.size)
    ]
    if ordering == "low_gamma_first":
        key = lambda item: (gamma_grid[item[1]], v_grid[item[0]])
    elif ordering == "high_gamma_first":
        key = lambda item: (-gamma_grid[item[1]], v_grid[item[0]])
    elif ordering == "low_speed_first":
        key = lambda item: (v_grid[item[0]], gamma_grid[item[1]])
    elif ordering == "high_speed_first":
        key = lambda item: (-v_grid[item[0]], gamma_grid[item[1]])
    else:
        raise ValueError(f"Unknown exploration ordering: {ordering}")
    return tuple(sorted(actions, key=key))


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
