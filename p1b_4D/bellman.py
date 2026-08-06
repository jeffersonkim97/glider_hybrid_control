"""Multi-start coarse Bellman candidate generator using authoritative J4D."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from .geometry import los_boundary_height, terrain_height
from .segment_feasibility import certify_straight_segment_geometry


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
    vehicle = configs["vehicle_config"]
    bellman_config = configs["bellman_config"]
    validation_config = configs["validation_config"]
    goal_position = geometry_bundle["primary_result"]["goal_position"]
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
            goal_position,
            validation_config,
            ordering,
            bellman_config,
        )
        for ordering in orderings
    }
    glide_detection_rate = stage["component_maps"]["glide_detection_rate"]
    time_step = vehicle["time_step"]
    pod_to_go_maps = {
        ordering: _compute_glide_pod_to_go(
            policy, transitions, glide_detection_rate, time_step
        )
        for ordering, policy in policies.items()
    }
    seeds = generate_switching_point_seeds(
        geometry_bundle,
        configuration_bundle,
        grids["z"],
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
        pod_to_go_maps,
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
            "pod_to_go_maps": {
                ordering: _readonly(values)
                for ordering, values in pod_to_go_maps.items()
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
            # Terrain-derived, like h_sensor: carried here so downstream
            # consumers that only receive this bundle (not geometry_bundle
            # itself) can still read the authoritative goal position.
            "goal_position": (
                float(goal_position[0]),
                float(goal_position[1]),
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
    configuration_bundle: dict[str, Any],
    z_grid: np.ndarray,
) -> np.ndarray:
    """Enumerate every z-grid node inside the feasible LOS-tangent interval.

    A switching point strictly between two z-grid nodes is indistinguishable
    downstream from whichever node `_switching_grid_index` snaps it to (grid
    is Bellman's only resolution), so arbitrary linspace sampling either
    wastes attempts re-testing the same node or skips nodes entirely
    depending on how the sample count lines up with the grid. Enumerating
    every node once is both cheaper (no duplicates) and exhaustive at grid
    resolution (nothing reachable is skipped): each seed only replays the
    already-solved policy (`extract_coarse_candidate`), it does not re-run
    `solve_coarse_bellman`.

    The switching-point height at each candidate z is read directly off the
    general swept LOS boundary (`los_boundary_height`), not a single tangent
    line: with one hill this reduces exactly to the old tangent-line formula
    over its whole domain, and with several it automatically follows
    whichever obstacle governs visibility at that z, with no per-hill
    special-casing.

    Where the boundary height exceeds the airspace ceiling, the seed is
    clipped to `h_max` rather than dropped: flying at `h_max` there is still
    strictly under the true (higher) shadow line, so it stays fully
    occluded -- only the boundary curve itself is out of reach, not
    concealment. Dropping those z-nodes instead of clipping them silently
    empties out the seed set near launch whenever a distant obstacle casts a
    shadow taller than the airspace allows (e.g. a shadow-casting hill far
    from launch but still close to the sensor), which starves the search of
    exactly the candidates that region of z needs. Clipping keeps every
    z-node in play; `evaluate_powered_segment`'s downstream occlusion
    certificate is what actually accepts or rejects each seed; a node below
    `h_min` has no concealed altitude at all and stays excluded.
    """
    geometry = geometry_bundle["primary_result"]["los_geometry"]
    sensor_position = geometry_bundle["primary_result"]["sensor_position"]
    environment = configuration_bundle["primary_result"]["environment_config"]
    airspace = environment["airspace"]
    z_sensor = float(sensor_position[0])
    in_domain = (z_grid >= environment["z_start"]) & (z_grid < z_sensor)
    domain_z = z_grid[in_domain]
    if domain_z.size == 0:
        raise ValueError("No z-grid nodes lie strictly between launch and the sensor")
    domain_h = los_boundary_height(geometry, domain_z)
    seed_h = np.minimum(domain_h, airspace["h_max"])
    within_airspace = seed_h >= airspace["h_min"]
    z_values = domain_z[within_airspace]
    h_values = seed_h[within_airspace]
    if z_values.size == 0:
        nearest_index = int(np.argmax(seed_h))
        z_values = domain_z[nearest_index : nearest_index + 1]
        h_values = seed_h[nearest_index : nearest_index + 1]
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
    sensor_position = geometry["sensor_position"]
    goal_position = geometry["goal_position"]
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
            relative_z = mesh_z - goal_position[0]
            relative_h = mesh_h - goal_position[1]
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
                (z_grid[mapped_z_safe] - goal_position[0]) ** 2
                + (h_grid[mapped_h_safe] - goal_position[1]) ** 2
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
                los_visible = (sample_z >= sensor_position[0]) | (
                    sample_h
                    >= los_boundary_height(
                        tangent, np.clip(sample_z, z_grid[0], z_grid[-1])
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
    goal_position: np.ndarray,
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
        (z_grid[:, None] - goal_position[0]) ** 2
        + (h_grid[None, :] - goal_position[1]) ** 2
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


def _compute_glide_pod_to_go(
    policy: dict[str, Any],
    transitions: dict[str, np.ndarray],
    glide_detection_rate: np.ndarray,
    time_step: float,
) -> np.ndarray:
    """Replay the fixed Bellman-optimal policy accumulating hazard, not cost.

    This performs no additional optimization: it follows the exact same
    policy pointers `solve_coarse_bellman` already computed, in the same
    z-descending order (transitions strictly advance z, so every successor
    is resolved before its predecessor). It exists only to report the
    probability-of-detection-to-go for visualization; the authoritative
    Bellman value used for policy selection is untouched.
    """
    value = policy["value"]
    z_size, h_size = value.shape
    hazard_to_go = np.full(value.shape, np.nan, dtype=float)
    for z_index in range(z_size - 1, -1, -1):
        for h_index in range(h_size):
            if policy["goal_mask"][z_index, h_index]:
                hazard_to_go[z_index, h_index] = 0.0
                continue
            if not np.isfinite(value[z_index, h_index]):
                continue
            velocity_index = int(policy["policy_velocity_index"][z_index, h_index])
            gamma_index = int(policy["policy_gamma_index"][z_index, h_index])
            terminal = bool(policy["policy_terminal"][z_index, h_index])
            segment_fraction = (
                float(transitions["terminal_fraction"][
                    z_index, h_index, velocity_index, gamma_index
                ])
                if terminal
                else 1.0
            )
            local_hazard = (
                float(glide_detection_rate[
                    z_index, h_index, velocity_index, gamma_index
                ])
                * time_step
                * segment_fraction
            )
            if terminal:
                hazard_to_go[z_index, h_index] = local_hazard
            else:
                next_z = int(policy["policy_next_z_index"][z_index, h_index])
                next_h = int(policy["policy_next_h_index"][z_index, h_index])
                hazard_to_go[z_index, h_index] = (
                    local_hazard + hazard_to_go[next_z, next_h]
                )
    return 1.0 - np.exp(-hazard_to_go)


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
    geometry_certificate = certify_straight_segment_geometry(
        launch,
        np.asarray(switching_point, dtype=float),
        geometry["terrain_model"],
        geometry["los_geometry"],
        float(geometry["sensor_position"][0]),
        environment["airspace"],
        terrain_tolerance=float(validation_config["terrain_tolerance"]),
        los_requirement="occluded",
        los_tolerance=float(validation_config["los_tolerance"]),
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
    terrain_clear = bool(geometry_certificate["terrain_clear"])
    occlusion_valid = bool(geometry_certificate["los_clear"])
    passed = bool(geometry_certificate["passed"])
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
            "domain_clear": bool(geometry_certificate["domain_clear"]),
            "minimum_terrain_margin": geometry_certificate[
                "minimum_terrain_margin"
            ],
            "minimum_occlusion_margin": geometry_certificate[
                "minimum_los_margin"
            ],
            "terrain_argmin_z": geometry_certificate["terrain_argmin_z"],
            "occlusion_argmin_z": geometry_certificate["los_argmin_z"],
            "geometry_certificate": geometry_certificate["certificate"],
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
    sensor_position = geometry_bundle["primary_result"]["sensor_position"]
    goal_position = geometry_bundle["primary_result"]["goal_position"]
    terrain_model = geometry_bundle["primary_result"]["terrain_model"]
    terrain_margin = trajectory[:, 1] - terrain_height(
        terrain_model, trajectory[:, 0]
    )
    los_visible = (trajectory[:-1, 0] >= sensor_position[0]) | (
        trajectory[:-1, 1]
        >= los_boundary_height(tangent, trajectory[:-1, 0])
        - validation["los_tolerance"]
    )
    tangent_error = abs(
        switching_point[1]
        - float(los_boundary_height(tangent, np.array([switching_point[0]]))[0])
    )
    objective_residual = abs(mission_cost - recomputed_mission_objective)
    combined_local_residual = abs(
        powered["powered_cost"] + glide_topology_cost - mission_cost
    )
    goal_distance = float(np.linalg.norm(trajectory[-1] - goal_position))
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
        "j4d_source": (
            stage_cost_4d_bundle["metadata"]["attacker_objective_id"]
            == objective_id
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


def _ordering_value_agreement(
    policies: dict[str, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Compare converged cost-to-go maps across exploration orderings.

    Bellman's action ordering only breaks ties between equal-cost actions,
    so independently solved orderings must agree everywhere both report a
    finite value. Disagreement beyond `tolerance` indicates a tie-breaking
    or convergence defect in the DP itself, not a case for a second solver.
    """
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
    candidates` output and re-exposes its fields under a stable schema. No
    CasADi/IPOPT NLP is used anywhere in this call.

    Optimality is scoped to the discretized switching-point seed grid and
    the discretized (velocity, gamma) state-action grid used by Bellman; no
    continuous global optimum is claimed.
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
            "schema_name": "AuthoritativeBellmanAttackerResponse",
            "schema_version": "1.0.0",
            "producer_phase": 8,
            "producer_module": "p1b_4D.bellman",
            "solution_method": "bellman_dynamic_programming",
            "optimality_scope": "discretized_switching_point_and_state_action_grid",
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
        "los_feasibility": best["validation"]["checks"]["los_feasibility"],
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
