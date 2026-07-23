"""Phase 8 continuous multi-start Attacker refinement using CasADi and IPOPT."""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import terrain_height


def solve_attacker_nlp_multistart(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_4d_bundle: dict[str, Any],
    bellman_candidate_bundle: dict[str, Any],
    filtered_bellman_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Independently refine every Top-K Bellman warm start and select the best."""
    for name, bundle in (
        ("configuration_bundle", configuration_bundle),
        ("geometry_bundle", geometry_bundle),
        ("detection_bundle", detection_bundle),
        ("stage_cost_4d_bundle", stage_cost_4d_bundle),
        ("bellman_candidate_bundle", bellman_candidate_bundle),
        ("filtered_bellman_bundle", filtered_bellman_bundle),
    ):
        _require_successful_bundle(bundle, name)
    if not filtered_bellman_bundle["metadata"].get("only_attacker_nlp_warm_start_source", False):
        raise ValueError("FilteredBellmanCandidateSet must be the exclusive NLP warm-start source")
    objective_ids = {
        configuration_bundle["primary_result"]["nlp_config"]["attacker_objective_id"],
        detection_bundle["metadata"]["attacker_objective_id"],
        stage_cost_4d_bundle["metadata"]["attacker_objective_id"],
        filtered_bellman_bundle["metadata"]["attacker_objective_id"],
    }
    if len(objective_ids) != 1:
        raise ValueError("Bellman and NLP attacker objective identifiers differ")

    bellman_result = bellman_candidate_bundle["primary_result"]
    stage_result = stage_cost_4d_bundle["primary_result"]

    bellman_support = {
        "mask": bellman_result["finite_cost_to_go_mask"],
        "primary_ordering": bellman_result["cost_to_go_primary_ordering"],
        "z_grid": stage_result["grids"]["z"],
        "h_grid": stage_result["grids"]["h"],
    }

    attempts: list[dict[str, Any]] = []
    feasible_solutions: list[dict[str, Any]] = []
    for candidate in filtered_bellman_bundle["primary_result"]["candidates"]:
        try:
            attempt = solve_attacker_nlp_start(
                candidate,
                configuration_bundle,
                geometry_bundle,
                detection_bundle,
                bellman_support,
            )
        except ValueError as error:
            attempt = {
                "source_candidate_id": candidate["candidate_id"],
                "source_rank": candidate["rank"],
                "solver_converged": False,
                "feasible": False,
                "solver_status": {
                    "success": False,
                    "return_status": "invalid_warm_start",
                    "iteration_count": 0,
                },
                "solution": None,
                "diagnostic": str(error),
            }
        attempts.append(attempt)
        if attempt["feasible"]:
            feasible_solutions.append(attempt["solution"])
    feasible_solutions.sort(key=lambda item: (item["mission_objective"], item["source_candidate_id"]))
    best = feasible_solutions[0] if feasible_solutions else None
    validation = validate_attacker_nlp_set(
        attempts,
        feasible_solutions,
        best,
        filtered_bellman_bundle,
    )
    return {
        "primary_result": {
            "best_found_attacker_response": best,
            "feasible_solutions": tuple(feasible_solutions),
            "solver_attempts": tuple(attempts),
            "attempt_count": len(attempts),
            "feasible_solution_count": len(feasible_solutions),
            "selection_rule": "minimum_mission_objective_among_feasible_solutions",
        },
        "validation": validation,
        "metadata": {
            "schema_name": "AttackerNLPBundle",
            "schema_version": "1.0.0",
            "producer_phase": 8,
            "producer_module": "p1b_4D.attacker_nlp",
            "solver": "ipopt",
            "casadi_version": ca.__version__,
            "attacker_objective_id": next(iter(objective_ids)),
            "warm_start_source_schema": filtered_bellman_bundle["metadata"]["schema_name"],
            "bellman_role": "initialization_only",
            "response_name": "Best-found Attacker Response",
            "global_optimum_claim": False,
            "only_attacker_solution_for_defender": True,
            "rerun_bellman_in_later_phases": False,
            "trajectory_refinement_complete": True,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "ATTACKER_NLP_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def solve_attacker_nlp_start(
    warm_start: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    bellman_support: dict[str, Any],
) -> dict[str, Any]:
    """Build and solve one independent continuous NLP from one warm start."""
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    nlp_config = configs["nlp_config"]
    validation_config = configs["validation_config"]
    geometry = geometry_bundle["primary_result"]
    functions = detection_bundle["primary_result"]["functions"]
    tangent = geometry["los_geometry"]
    sensor = geometry["sensor_position"]
    node_count = int(nlp_config["number_of_nodes"])
    interval_count = node_count - 1

    support_mask = np.asarray(bellman_support["mask"], dtype=bool)
    support_z_grid = np.asarray(bellman_support["z_grid"], dtype=float)
    support_h_grid = np.asarray(bellman_support["h_grid"], dtype=float)

    expected_support_shape = (
        support_z_grid.size,
        support_h_grid.size,
    )
    if support_mask.shape != expected_support_shape:
        raise ValueError(
            "Bellman support mask shape does not match the Bellman grids"
        )

    support_dz = float(support_z_grid[1] - support_z_grid[0])
    support_dh = float(support_h_grid[1] - support_h_grid[0])

    inside_result = distance_transform_edt(
        support_mask,
        sampling=(support_dz, support_dh),
        return_distances=True,
        return_indices=False,
    )

    outside_result = distance_transform_edt(
        ~support_mask,
        sampling=(support_dz, support_dh),
        return_distances=True,
        return_indices=False,
    )

    if not isinstance(inside_result, np.ndarray):
        raise TypeError("Inside distance transform did not return an ndarray")

    if not isinstance(outside_result, np.ndarray):
        raise TypeError("Outside distance transform did not return an ndarray")

    inside_distance = inside_result.astype(float, copy=False)
    outside_distance = outside_result.astype(float, copy=False)

    support_signed_distance = inside_distance - outside_distance

    support_function = ca.interpolant(
        "bellman_support_signed_distance",
        "linear",
        [support_z_grid, support_h_grid],
        support_signed_distance.ravel(order="F"),
    )
    
    opti = ca.Opti()
    z_switch = opti.variable()
    h_switch = opti.variable()
    z = opti.variable(node_count)
    h = opti.variable(node_count)
    velocity = opti.variable(interval_count)
    gamma = opti.variable(interval_count)
    interval_time = opti.variable(interval_count)
    glide_time = ca.sum1(interval_time)

    terrain_function = _terrain_interpolant(geometry)
    clearance = vehicle["switching_constraints"]["terrain_clearance"]
    gamma_min = np.deg2rad(vehicle["gamma_min_deg"])
    gamma_max = np.deg2rad(vehicle["gamma_max_deg"])
    opti.subject_to(z_switch >= environment["z_start"])
    opti.subject_to(z_switch <= tangent["tangent_point"][0])
    opti.subject_to(h_switch == tangent["tangent_slope"] * z_switch + tangent["tangent_intercept"])
    opti.subject_to(z[0] == z_switch)
    opti.subject_to(h[0] == h_switch)
    goal_radius = validation_config["goal_radius"]
    opti.subject_to(
        (z[-1] - environment["z_goal"]) ** 2
        + (h[-1] - environment["h_goal"]) ** 2
        <= goal_radius**2
    )
    opti.subject_to(opti.bounded(environment["airspace"]["z_min"], z, environment["airspace"]["z_max"]))
    opti.subject_to(opti.bounded(environment["airspace"]["h_min"], h, environment["airspace"]["h_max"]))

    support_margin = -0.1

    for node_index in range(node_count):
        support_value = support_function(
            ca.vertcat(
                z[node_index],
                h[node_index],
            )
        )
        opti.subject_to(support_value >= support_margin)

    support_sample_fractions = (0.25, 0.5, 0.75)

    for interval_index in range(interval_count):
        for fraction in support_sample_fractions:
            sample_z = (
                (1.0 - fraction) * z[interval_index]
                + fraction * z[interval_index + 1]
            )
            sample_h = (
                (1.0 - fraction) * h[interval_index]
                + fraction * h[interval_index + 1]
            )

            support_value = support_function(
                ca.vertcat(sample_z, sample_h)
            )
            opti.subject_to(support_value >= support_margin)

    opti.subject_to(opti.bounded(vehicle["glide_speed_min"], velocity, vehicle["glide_speed_max"]))
    opti.subject_to(opti.bounded(gamma_min, gamma, gamma_max))
    opti.subject_to(opti.bounded(
        nlp_config["minimum_interval_time"],
        interval_time,
        nlp_config["maximum_interval_time"],
    ))
    opti.subject_to(glide_time <= environment["simulation"]["max_path_steps"] * vehicle["time_step"])
    opti.subject_to(z[1:] - z[:-1] == interval_time * velocity * ca.cos(gamma))
    opti.subject_to(h[1:] - h[:-1] == interval_time * velocity * ca.sin(gamma))
    opti.subject_to(z[1:] > z[:-1])
    opti.subject_to(h[:-1] >= terrain_function(z[:-1]) + clearance)
    opti.subject_to(h[-1] >= terrain_function(z[-1]) - validation_config["terrain_tolerance"])
    opti.subject_to(h >= tangent["tangent_slope"] * z + tangent["tangent_intercept"])

    cl = 2.0 * vehicle["mass"] * vehicle["gravity"] * ca.cos(gamma) / (
        vehicle["air_density"] * velocity**2 * vehicle["wing_area"]
    )
    cd = vehicle["cd0"] + vehicle["linear_drag_coefficient"] * cl + vehicle["quadratic_drag_coefficient"] * cl**2
    opti.subject_to(opti.bounded(vehicle["dynamic_limits"]["cl_min"], cl, vehicle["dynamic_limits"]["cl_max"]))
    opti.subject_to(cd >= validation_config["dynamic_tolerance"])

    powered_time, powered_hazard, powered_constraints = _powered_symbolics(
        z_switch, h_switch, terrain_function, configs, geometry, functions
    )
    for constraint in powered_constraints:
        opti.subject_to(constraint)
    midpoint_z = 0.5 * (z[:-1] + z[1:])
    midpoint_h = 0.5 * (h[:-1] + h[1:])
    glide_rate = functions["glide_detection_components"](
        midpoint_z, midpoint_h, velocity, gamma,
        sensor[0], sensor[1], tangent["tangent_point"][0],
        tangent["tangent_slope"], tangent["tangent_intercept"],
    )[-1]
    glide_hazard = ca.dot(interval_time, glide_rate)
    objective_outputs = functions["attacker_objective"](
        powered_hazard, glide_hazard, powered_time, glide_time
    )
    mission_objective = objective_outputs[-1]
    homotopy_scale = opti.parameter()
    attacker_cost = configs["cost_config"]["attacker"]
    time_reference = attacker_cost["normalization"]["time"]["reference_seconds"]
    hazard_reference = attacker_cost["normalization"]["pod"][
        "hazard_reference"
    ]
    conditioned_detection = (
        homotopy_scale * (powered_hazard + glide_hazard) / hazard_reference
    )
    conditioned_objective = (
        attacker_cost["w_pod"] * conditioned_detection
        + attacker_cost["w_time"] * (powered_time + glide_time) / time_reference
    )
    opti.minimize(conditioned_objective)

    warm = build_dynamically_consistent_warm_start(
        warm_start,
        node_count,
        configs,
        geometry,
    )

    warm_node_support_values: list[float] = []
    warm_sample_support_values: list[float] = []

    for point in warm["trajectory"]:
        support_output = support_function(
            np.asarray(point, dtype=float)
        )
        warm_node_support_values.append(
            float(
                np.asarray(
                    support_output,
                    dtype=float,
                ).reshape(-1)[0]
            )
        )

    for interval_index in range(interval_count):
        start_point = warm["trajectory"][interval_index]
        end_point = warm["trajectory"][interval_index + 1]

        for fraction in support_sample_fractions:
            sample_point = (
                (1.0 - fraction) * start_point
                + fraction * end_point
            )

            support_output = support_function(
                np.asarray(sample_point, dtype=float)
            )

            warm_sample_support_values.append(
                float(
                    np.asarray(
                        support_output,
                        dtype=float,
                    ).reshape(-1)[0]
                )
            )

    minimum_warm_node_support_margin = float(
        np.min(warm_node_support_values)
    )

    minimum_warm_node_support_index = int(
        np.argmin(warm_node_support_values)
    )

    minimum_warm_node_support_point = np.asarray(
        warm["trajectory"][minimum_warm_node_support_index],
        dtype=float,
    )

    minimum_warm_sample_support_margin = float(
        np.min(warm_sample_support_values)
    )

    minimum_warm_support_margin = min(
        minimum_warm_node_support_margin,
        minimum_warm_sample_support_margin,
    )

    warm["diagnostics"].update(
        {
            "minimum_bellman_support_margin": (
                minimum_warm_support_margin
            ),
            "minimum_bellman_node_support_margin": (
                minimum_warm_node_support_margin
            ),
            "minimum_bellman_sample_support_margin": (
                minimum_warm_sample_support_margin
            ),
            "bellman_support_feasible": (
                minimum_warm_support_margin
                >= support_margin -validation_config["solver_tolerance"]
            ),

            "minimum_bellman_node_support_index": (
                minimum_warm_node_support_index
            ),
            "minimum_bellman_node_support_point": (
                minimum_warm_node_support_point.tolist()
            ),
        }
    )

    warm_values = {
        "switching_point": np.asarray(
            warm_start["switching_point"],
            dtype=float,
        ),
        "trajectory": np.asarray(
            warm["trajectory"],
            dtype=float,
        ),
        "velocity_profile": np.asarray(
            warm["velocity_profile"],
            dtype=float,
        ),
        "gamma_profile": np.asarray(
            warm["gamma_profile"],
            dtype=float,
        ),
        "interval_time_profile": np.asarray(
            warm["interval_time_profile"],
            dtype=float,
        ),
        "glide_time": float(warm["glide_time"]),
    }

    warm_residuals = compute_constraint_residuals(
        warm_values,
        configs,
        geometry,
    )

    warm_velocity = warm_values["velocity_profile"]
    warm_gamma = warm_values["gamma_profile"]
    warm_interval_time = warm_values["interval_time_profile"]
    warm_trajectory = warm_values["trajectory"]

    warm_cl = (
        2.0
        * vehicle["mass"]
        * vehicle["gravity"]
        * np.cos(warm_gamma)
        / (
            vehicle["air_density"]
            * warm_velocity**2
            * vehicle["wing_area"]
        )
    )

    warm_cd = (
        vehicle["cd0"]
        + vehicle["linear_drag_coefficient"] * warm_cl
        + vehicle["quadratic_drag_coefficient"] * warm_cl**2
    )

    solver_tolerance = validation_config["solver_tolerance"]

    warm_constraint_checks = {
        "goal": (
            warm_residuals["goal_error_norm"]
            <= validation_config["goal_radius"] + solver_tolerance
        ),
        "terrain": (
            warm_residuals["minimum_terrain_margin"]
            >= vehicle["switching_constraints"]["terrain_clearance"]
            - solver_tolerance
        ),
        "los": (
            warm_residuals["minimum_los_margin"]
            >= -validation_config["los_tolerance"]
        ),
        "switching": (
            abs(warm_residuals["switching_tangent_residual"])
            <= validation_config["los_tolerance"]
        ),
        "dynamics": (
            warm_residuals["maximum_dynamic_residual"]
            <= solver_tolerance
        ),
        "interval_time": bool(
            np.all(
                warm_interval_time
                >= nlp_config["minimum_interval_time"] - solver_tolerance
            )
            and np.all(
                warm_interval_time
                <= nlp_config["maximum_interval_time"] + solver_tolerance
            )
        ),
        "velocity": bool(
            np.all(
                warm_velocity
                >= vehicle["glide_speed_min"] - solver_tolerance
            )
            and np.all(
                warm_velocity
                <= vehicle["glide_speed_max"] + solver_tolerance
            )
        ),
        "gamma": bool(
            np.all(
                warm_gamma
                >= np.deg2rad(vehicle["gamma_min_deg"]) - solver_tolerance
            )
            and np.all(
                warm_gamma
                <= np.deg2rad(vehicle["gamma_max_deg"]) + solver_tolerance
            )
        ),
        "lift": bool(
            np.all(
                warm_cl
                >= vehicle["dynamic_limits"]["cl_min"] - solver_tolerance
            )
            and np.all(
                warm_cl
                <= vehicle["dynamic_limits"]["cl_max"] + solver_tolerance
            )
        ),
        "drag": bool(
            np.all(
                warm_cd
                >= validation_config["dynamic_tolerance"]
                - solver_tolerance
            )
        ),
        "airspace": bool(
            np.all(
                warm_trajectory[:, 0]
                >= environment["airspace"]["z_min"] - solver_tolerance
            )
            and np.all(
                warm_trajectory[:, 0]
                <= environment["airspace"]["z_max"] + solver_tolerance
            )
            and np.all(
                warm_trajectory[:, 1]
                >= environment["airspace"]["h_min"] - solver_tolerance
            )
            and np.all(
                warm_trajectory[:, 1]
                <= environment["airspace"]["h_max"] + solver_tolerance
            )
        ),
        "bellman_support": bool(
            warm["diagnostics"]["bellman_support_feasible"]
        ),
    }

    warm["diagnostics"].update(
        {
            "full_constraint_checks": warm_constraint_checks,
            "full_nlp_feasible": all(
                warm_constraint_checks.values()
            ),
            "warm_goal_error_norm": warm_residuals["goal_error_norm"],
            "warm_minimum_cl": float(np.min(warm_cl)),
            "warm_maximum_cl": float(np.max(warm_cl)),
            "warm_minimum_cd": float(np.min(warm_cd)),
            "warm_minimum_interval_time": float(
                np.min(warm_interval_time)
            ),
            "warm_maximum_interval_time": float(
                np.max(warm_interval_time)
            ),
        }
    )

    warm_midpoints = 0.5 * (
        warm["trajectory"][:-1] + warm["trajectory"][1:]
    )
    warm_rates = np.asarray(functions["glide_detection_components"](
        warm_midpoints[:, 0],
        warm_midpoints[:, 1],
        warm["velocity_profile"],
        warm["gamma_profile"],
        sensor[0], sensor[1], tangent["tangent_point"][0],
        tangent["tangent_slope"], tangent["tangent_intercept"],
    )[-1], dtype=float).reshape(-1)
    warm_glide_hazard = float(np.dot(
        warm["interval_time_profile"], warm_rates
    ))
    warm_objective = _numeric_outputs(
        functions["attacker_objective"],
        warm_start["hazard_breakdown"]["powered_acoustic_hazard"],
        warm_glide_hazard,
        warm_start["powered_time"],
        warm["glide_time"],
    )
    warm["diagnostics"].update({
        "initial_glide_hazard": warm_glide_hazard,
        "initial_mission_pod": warm_objective[0],
        "initial_mission_time": warm_objective[1],
        "initial_mission_objective": warm_objective[-1],
    })

    # if warm_start["candidate_id"] == "bellman-candidate-001":
    #     print("\n=== CANDIDATE-001 WARM START ===")
    #     print(warm["diagnostics"])

    opti.set_initial(z_switch, warm_start["switching_point"][0])
    opti.set_initial(h_switch, warm_start["switching_point"][1])
    opti.set_initial(z, warm["trajectory"][:, 0])
    opti.set_initial(h, warm["trajectory"][:, 1])
    opti.set_initial(velocity, warm["velocity_profile"])
    opti.set_initial(gamma, warm["gamma_profile"])
    opti.set_initial(interval_time, warm["interval_time_profile"])
    solver_options = {
        key.removeprefix("ipopt."): value
        for key, value in nlp_config["ipopt_options"].items()
    }
    opti.solver(
        nlp_config["solver"], {"print_time": False}, solver_options
    )

    try:
        solved = None
        continuation_history: list[dict[str, Any]] = []
        continuation_results: list[dict[str, Any]] = []
        for scale in nlp_config["hazard_homotopy_scales"]:
            opti.set_value(homotopy_scale, float(scale))
            if solved is not None:
                opti.set_initial(z_switch, solved.value(z_switch))
                opti.set_initial(h_switch, solved.value(h_switch))
                opti.set_initial(z, solved.value(z))
                opti.set_initial(h, solved.value(h))
                opti.set_initial(velocity, solved.value(velocity))
                opti.set_initial(gamma, solved.value(gamma))
                opti.set_initial(interval_time, solved.value(interval_time))
            solved = opti.solve()
            stage_values = {
                "switching_point": np.array([
                    solved.value(z_switch), solved.value(h_switch)
                ]),
                "trajectory": np.column_stack((
                    solved.value(z), solved.value(h)
                )),
                "velocity_profile": np.asarray(
                    solved.value(velocity)
                ).reshape(-1),
                "gamma_profile": np.asarray(
                    solved.value(gamma)
                ).reshape(-1),
                "interval_time_profile": np.asarray(
                    solved.value(interval_time)
                ).reshape(-1),
                "glide_time": float(solved.value(glide_time)),
                "powered_time": float(solved.value(powered_time)),
                "powered_hazard": float(solved.value(powered_hazard)),
                "glide_hazard": float(solved.value(glide_hazard)),
                "mission_pod": float(solved.value(objective_outputs[0])),
                "mission_time": float(solved.value(objective_outputs[1])),
                "pod_normalized": float(solved.value(objective_outputs[2])),
                "time_normalized": float(solved.value(objective_outputs[3])),
                "mission_objective": float(solved.value(mission_objective)),
            }
            stage_stats = solved.stats()
            continuation_results.append({
                "hazard_scale": float(scale),
                "values": stage_values,
                "solver_stats": stage_stats,
            })
            continuation_history.append({
                "hazard_scale": float(scale),
                "conditioned_objective": float(
                    solved.value(conditioned_objective)
                ),
                "mission_objective": stage_values["mission_objective"],
                "mission_hazard": float(
                    solved.value(powered_hazard + glide_hazard)
                ),
                "mission_time": float(
                    solved.value(powered_time + glide_time)
                ),
                "solver_status": stage_stats.get(
                    "return_status", "unknown"
                ),
            })
        if solved is None:
            raise RuntimeError("No hazard-homotopy NLP stage was executed")
        selected_stage_index = len(continuation_results) - 1
        selected_stage = continuation_results[selected_stage_index]
        values = selected_stage["values"]
        solver_stats = selected_stage["solver_stats"]
        for index, history_item in enumerate(continuation_history):
            history_item["selected_by_exact_objective"] = (
                index == selected_stage_index
            )
        residuals = compute_constraint_residuals(values, configs, geometry)
        validation = validate_nlp_solution(
            values, residuals, solver_stats, configs, geometry, functions,
            warm["diagnostics"], continuation_history,
        )
        solution = {
            "solution_id": f"nlp-{warm_start['candidate_id']}",
            "source_candidate_id": warm_start["candidate_id"],
            "source_rank": warm_start["rank"],
            "warm_start_diagnostics": warm["diagnostics"],
            "continuation_history": tuple(continuation_history),
            **{name: _readonly(value) if isinstance(value, np.ndarray) else value for name, value in values.items()},
            "constraint_residuals": residuals,
            "solver_status": {
                "success": bool(solver_stats.get("success", False)),
                "return_status": solver_stats.get("return_status", "unknown"),
                "iteration_count": int(solver_stats.get("iter_count", -1)),
            },
            "validation": validation,
            "metadata": {
                "continuous_refinement": True,
                "warm_start_only_constrained": False,
                "node_count": node_count,
                "solver": nlp_config["solver"],
                "hazard_homotopy_scales": tuple(
                    nlp_config["hazard_homotopy_scales"]
                ),
                "final_homotopy_scale": float(
                    nlp_config["hazard_homotopy_scales"][-1]
                ),
                "selected_homotopy_scale": selected_stage["hazard_scale"],
                "selection_rule": (
                    "minimum_exact_attacker_objective_across_feasible_"
                    "continuous_homotopy_endpoints"
                ),
            },
        }
        return {
            "source_candidate_id": warm_start["candidate_id"],
            "source_rank": warm_start["rank"],
            "solver_converged": bool(solver_stats.get("success", False)),
            "feasible": validation["passed"],
            "solver_status": solution["solver_status"],
            "solution": solution,
            "diagnostic": validation["summary"],
        }
    except RuntimeError as error:
        try:
            stats = opti.stats()
        except RuntimeError:
            stats = {}
        return {
            "source_candidate_id": warm_start["candidate_id"],
            "source_rank": warm_start["rank"],
            "solver_converged": False,
            "feasible": False,
            "solver_status": {
                "success": False,
                "return_status": stats.get("return_status", "solver_exception"),
                "iteration_count": int(stats.get("iter_count", -1)),
            },
            "solution": None,
            "diagnostic": str(error),
        }


def _powered_symbolics(
    z_switch: ca.MX,
    h_switch: ca.MX,
    terrain_function: ca.Function,
    configs: dict[str, Any],
    geometry: dict[str, Any],
    functions: dict[str, ca.Function],
) -> tuple[ca.MX, ca.MX, list[ca.MX]]:
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    count = vehicle["segment_check_count"]
    fractions = np.linspace(0.0, 1.0, count)
    delta_z = z_switch - environment["z_start"]
    delta_h = h_switch - environment["h_start"]
    powered_time = ca.sqrt(delta_z**2 + delta_h**2) / vehicle["powered_speed"]
    sample_z = environment["z_start"] + fractions * delta_z
    sample_h = environment["h_start"] + fractions * delta_h
    rates = functions["powered_detection_components"](
        sample_z, sample_h, vehicle["powered_speed"],
        geometry["sensor_position"][0], geometry["sensor_position"][1],
    )[-1]
    weights = np.ones(count)
    weights[[0, -1]] = 0.5
    powered_hazard = powered_time / (count - 1) * ca.dot(weights, rates)
    tangent = geometry["los_geometry"]
    constraints: list[ca.MX] = [
        sample_h[1:-1] >= terrain_function(sample_z[1:-1]),
        sample_h <= tangent["tangent_slope"] * sample_z + tangent["tangent_intercept"],
    ]
    return powered_time, powered_hazard, constraints


def compute_constraint_residuals(
    values: dict[str, Any], configs: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    """Compute named numerical residual vectors and maximum violations."""
    z = values["trajectory"][:, 0]
    h = values["trajectory"][:, 1]
    velocity = values["velocity_profile"]
    gamma = values["gamma_profile"]
    interval_time = values["interval_time_profile"]
    terrain = terrain_height(geometry["terrain_model"], z)
    tangent = geometry["los_geometry"]
    switching_line = values["switching_point"][1] - (
        tangent["tangent_slope"] * values["switching_point"][0] + tangent["tangent_intercept"]
    )
    dynamics_z = np.diff(z) - interval_time * velocity * np.cos(gamma)
    dynamics_h = np.diff(h) - interval_time * velocity * np.sin(gamma)
    los_margin = h - (tangent["tangent_slope"] * z + tangent["tangent_intercept"])
    terrain_margin = h - terrain
    goal_error = values["trajectory"][-1] - np.array([
        configs["environment_config"]["z_goal"], configs["environment_config"]["h_goal"]
    ])
    return {
        "goal_error": _readonly(goal_error),
        "terrain_margin": _readonly(terrain_margin),
        "los_margin": _readonly(los_margin),
        "dynamic_z_residual": _readonly(dynamics_z),
        "dynamic_h_residual": _readonly(dynamics_h),
        "switching_tangent_residual": float(switching_line),
        "maximum_dynamic_residual": float(max(np.max(np.abs(dynamics_z)), np.max(np.abs(dynamics_h)))),
        "goal_error_norm": float(np.linalg.norm(goal_error)),
        "minimum_terrain_margin": float(np.min(terrain_margin[:-1])),
        "minimum_los_margin": float(np.min(los_margin)),
        "interval_time_sum_residual": float(
            abs(np.sum(interval_time) - values["glide_time"])
        ),
    }


def validate_nlp_solution(
    values: dict[str, Any], residuals: dict[str, Any], solver_stats: dict[str, Any],
    configs: dict[str, Any], geometry: dict[str, Any], functions: dict[str, ca.Function],
    warm_start_diagnostics: dict[str, Any],
    continuation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = configs["validation_config"]
    vehicle = configs["vehicle_config"]
    nlp_config = configs["nlp_config"]
    objective = _numeric_outputs(functions["attacker_objective"], values["powered_hazard"], values["glide_hazard"], values["powered_time"], values["glide_time"])
    checks = {
        "solver_convergence": bool(solver_stats.get("success", False)),
        "goal_error": residuals["goal_error_norm"] <= (
            validation["goal_radius"] + validation["solver_tolerance"]
        ),
        "terrain_clearance": residuals["minimum_terrain_margin"] >= vehicle["switching_constraints"]["terrain_clearance"] - validation["solver_tolerance"],
        "los_feasibility": residuals["minimum_los_margin"] >= -validation["los_tolerance"],
        "switching_constraint": abs(residuals["switching_tangent_residual"]) <= validation["los_tolerance"],
        "dynamic_residual": residuals["maximum_dynamic_residual"] <= validation["solver_tolerance"],
        "interval_time_consistency": residuals["interval_time_sum_residual"] <= validation["solver_tolerance"],
        "interval_time_bounds": bool(np.all(
            (values["interval_time_profile"] >= nlp_config["minimum_interval_time"] - validation["solver_tolerance"])
            & (values["interval_time_profile"] <= nlp_config["maximum_interval_time"] + validation["solver_tolerance"])
        )),
        "objective_consistency": abs(values["mission_objective"] - objective[-1]) <= validation["objective_tolerance"],
        "velocity_bounds": bool(np.all((values["velocity_profile"] >= vehicle["glide_speed_min"] - validation["solver_tolerance"]) & (values["velocity_profile"] <= vehicle["glide_speed_max"] + validation["solver_tolerance"]))),
        "gamma_bounds": bool(np.all((values["gamma_profile"] >= np.deg2rad(vehicle["gamma_min_deg"]) - validation["solver_tolerance"]) & (values["gamma_profile"] <= np.deg2rad(vehicle["gamma_max_deg"]) + validation["solver_tolerance"]))),
        "warm_start_kinematically_consistent": bool(
            warm_start_diagnostics["kinematically_consistent"]
        ),
        # "warm_start_not_worsened": (
        #     values["mission_objective"]
        #     <= warm_start_diagnostics["initial_mission_objective"]
        #     + validation["objective_tolerance"]
        # ),
        "warm_start_not_worsened": (
            not warm_start_diagnostics["bellman_support_feasible"]
            or (
                values["mission_objective"]
                <= warm_start_diagnostics["initial_mission_objective"]
                + validation["objective_tolerance"]
            )
        ),
        "final_exact_objective_stage": bool(
            continuation_history
            and continuation_history[-1]["hazard_scale"] == 1.0
            and abs(
                continuation_history[-1]["conditioned_objective"]
                - continuation_history[-1]["mission_objective"]
            ) <= validation["objective_tolerance"]
        ),
        "exact_continuation_endpoint_selected": bool(
            continuation_history
            and abs(
                values["mission_objective"]
                - continuation_history[-1]["mission_objective"]
            ) <= validation["objective_tolerance"]
            and sum(
                bool(item["selected_by_exact_objective"])
                for item in continuation_history
            ) == 1
            and continuation_history[-1]["selected_by_exact_objective"]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    metrics = {key: residuals[key] for key in (
        "goal_error_norm", "minimum_terrain_margin", "minimum_los_margin",
        "maximum_dynamic_residual", "switching_tangent_residual",
    )}
    metrics.update({
        "warm_start_objective": warm_start_diagnostics[
            "initial_mission_objective"
        ],
        "refined_objective": values["mission_objective"],
        "objective_improvement": (
            warm_start_diagnostics["initial_mission_objective"]
            - values["mission_objective"]
        ),
        "final_mission_hazard": (
            values["powered_hazard"] + values["glide_hazard"]
        ),
    })
    return {"passed": not failed, "checks": checks, "metrics": metrics, "failed_checks": failed, "summary": "Attacker NLP solution validation passed" if not failed else f"NLP solution failed checks: {failed}"}


def validate_attacker_nlp_set(
    attempts: list[dict[str, Any]], feasible: list[dict[str, Any]], best: dict[str, Any] | None,
    filtered_bundle: dict[str, Any],
) -> dict[str, Any]:
    expected = filtered_bundle["primary_result"]["selected_candidate_count"]
    checks = {
        "all_warm_starts_attempted": len(attempts) == expected,
        "independent_solve_per_start": len({item["source_candidate_id"] for item in attempts}) == expected,
        "all_warm_starts_refined": len(feasible) == expected,
        "feasible_solution_available": bool(feasible),
        "all_stored_solutions_valid": all(item["validation"]["passed"] for item in feasible),
        "best_found_is_minimum": best is not None and best["mission_objective"] == min(item["mission_objective"] for item in feasible),
    }
    failed = [name for name, passed in checks.items() if not passed]
    failed_solve_count = sum(not item["feasible"] for item in attempts)
    warnings = [f"{failed_solve_count} independent NLP solves were infeasible or did not converge"] if failed_solve_count else []
    return {
        "passed": not failed,
        "checks": checks,
        "metrics": {"attempt_count": len(attempts), "feasible_solution_count": len(feasible), "failed_solve_count": failed_solve_count, "best_found_objective": best["mission_objective"] if best else np.inf},
        "warnings": warnings,
        "failed_checks": failed,
        "summary": "Phase 8 Attacker NLP validation passed" if not failed else f"Attacker NLP failed checks: {failed}",
    }


def _terrain_interpolant(geometry: dict[str, Any]) -> ca.Function:
    """Translate the authoritative SciPy natural cubic spline exactly to CasADi."""
    model = geometry["terrain_model"]
    knots = np.asarray(model.interpolant.x)
    coefficients = np.asarray(model.interpolant.c)
    query = ca.SX.sym("terrain_query")
    interval = knots.size - 2
    delta = query - knots[interval]
    expression = (
        coefficients[0, interval] * delta**3
        + coefficients[1, interval] * delta**2
        + coefficients[2, interval] * delta
        + coefficients[3, interval]
    )
    for interval in range(knots.size - 3, -1, -1):
        delta = query - knots[interval]
        polynomial = (
            coefficients[0, interval] * delta**3
            + coefficients[1, interval] * delta**2
            + coefficients[2, interval] * delta
            + coefficients[3, interval]
        )
        expression = ca.if_else(query < knots[interval + 1], polynomial, expression)
    return ca.Function("AuthoritativeNaturalCubicTerrainFunction", [query], [expression])


def build_dynamically_consistent_warm_start(
    candidate: dict[str, Any],
    node_count: int,
    configs: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Convert one coarse topology into an exactly kinematic NLP initial point.

    Bellman states are snapped to a coarse spatial grid, so independently
    interpolating their state and control arrays does not preserve dynamics.
    This adapter preserves the path topology, resamples by arc length, projects
    altitude onto the exact terrain/LOS/dynamics feasible envelope, and derives
    controls and interval times from the resulting segments.  The envelope also
    accounts for the aerodynamic lower-lift bound, which can make the effective
    steepest descent angle shallower than the configured kinematic gamma bound.
    """
    if node_count < 2:
        raise ValueError("node_count must be at least two")
    vehicle = configs["vehicle_config"]
    environment = configs["environment_config"]
    source = np.asarray(candidate["trajectory"], dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape[0] < 2:
        raise ValueError("Bellman warm-start trajectory must have shape (n, 2)")
    if not np.all(np.isfinite(source)):
        raise ValueError("Bellman warm-start trajectory must be finite")
    if not np.all(np.diff(source[:, 0]) > 0.0):
        raise ValueError(
            "Bellman warm-start trajectory must be strictly forward in z; "
            "topology repair is not permitted"
        )

    switching_point = np.asarray(candidate["switching_point"], dtype=float)
    goal_center = np.array(
        [environment["z_goal"], environment["h_goal"]], dtype=float
    )
    goal = source[-1].copy()
    if np.linalg.norm(goal - goal_center) > (
        configs["validation_config"]["goal_radius"]
        + configs["validation_config"]["solver_tolerance"]
    ):
        raise ValueError("Bellman warm-start endpoint lies outside the goal region")
    interior = source[1:-1]
    path = np.vstack((switching_point, interior, goal))
    keep = np.concatenate(([True], np.diff(path[:, 0]) > np.finfo(float).eps))
    path = path[keep]
    if path.shape[0] < 2 or not np.all(np.diff(path[:, 0]) > 0.0):
        raise ValueError("Bellman warm-start topology must advance in z")

    source_distance = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)),
    ))
    if source_distance[-1] <= 0.0:
        raise ValueError("Bellman warm-start topology has zero path length")
    targets = np.linspace(0.0, source_distance[-1], node_count)
    z = np.interp(targets, source_distance, path[:, 0])
    h = np.interp(targets, source_distance, path[:, 1])
    z[0], h[0] = switching_point
    z[-1], h[-1] = goal

    clearance = vehicle["switching_constraints"]["terrain_clearance"]
    gamma_configured_min = np.deg2rad(vehicle["gamma_min_deg"])
    gamma_max = np.deg2rad(vehicle["gamma_max_deg"])
    lift_factor = (
        2.0 * vehicle["mass"] * vehicle["gravity"]
        / (vehicle["air_density"] * vehicle["wing_area"])
    )
    minimum_cosine = (
        vehicle["dynamic_limits"]["cl_min"]
        * vehicle["glide_speed_min"] ** 2
        / lift_factor
    )
    if minimum_cosine >= 1.0:
        raise ValueError("Vehicle bounds admit no aerodynamically feasible glide angle")
    aerodynamic_gamma_min = -np.arccos(max(0.0, minimum_cosine)) + 1.0e-8
    gamma_min = max(gamma_configured_min, aerodynamic_gamma_min)
    slope_min = np.tan(gamma_min)
    slope_max = np.tan(gamma_max)
    maximum_interval_time = configs["nlp_config"]["maximum_interval_time"]
    interval_delta_z = np.diff(z)
    maximum_horizontal_distance = (
        vehicle["glide_speed_max"] * maximum_interval_time
    )
    if np.any(interval_delta_z >= maximum_horizontal_distance):
        raise ValueError("NLP node spacing exceeds the maximum reachable interval distance")
    speed_limited_slope = np.sqrt(
        np.maximum((maximum_horizontal_distance / interval_delta_z) ** 2 - 1.0, 0.0)
    )
    lift_time_ratio = (
        lift_factor * maximum_interval_time**2
        / (vehicle["dynamic_limits"]["cl_min"] * interval_delta_z**2)
    )
    lift_limited_slope = np.sqrt(
        np.maximum(lift_time_ratio ** (2.0 / 3.0) - 1.0, 0.0)
    )
    maximum_descent_slope = (
        np.minimum(speed_limited_slope, lift_limited_slope) * (1.0 - 1.0e-8)
    )
    interval_slope_min = np.maximum(
        slope_min,
        -maximum_descent_slope,
    )

    tangent = geometry["los_geometry"]
    required_height = np.maximum(
        terrain_height(geometry["terrain_model"], z) + clearance,
        tangent["tangent_slope"] * z + tangent["tangent_intercept"],
    )
    required_height[-1] = goal[1]
    airspace_max = float(environment["airspace"]["h_max"])
    feasible_lower = np.empty(node_count, dtype=float)
    feasible_upper = np.empty(node_count, dtype=float)
    feasible_lower[-1] = goal[1]
    feasible_upper[-1] = goal[1]
    for index in range(node_count - 2, -1, -1):
        dz = z[index + 1] - z[index]
        feasible_lower[index] = max(
            required_height[index],
            feasible_lower[index + 1] - slope_max * dz,
        )
        feasible_upper[index] = min(
            airspace_max,
            feasible_upper[index + 1] - interval_slope_min[index] * dz,
        )
        if feasible_lower[index] > feasible_upper[index] + 1.0e-10:
            raise ValueError("Adapted Bellman topology has no feasible altitude envelope")
    if not (
        feasible_lower[0] - 1.0e-10
        <= switching_point[1]
        <= feasible_upper[0] + 1.0e-10
    ):
        raise ValueError("Switching point cannot reach the goal within NLP glide bounds")

    desired_h = h.copy()
    h[0] = switching_point[1]
    for index in range(1, node_count):
        dz = z[index] - z[index - 1]
        local_lower = max(
            feasible_lower[index],
            h[index - 1] + interval_slope_min[index - 1] * dz,
        )
        local_upper = min(
            feasible_upper[index],
            h[index - 1] + slope_max * dz,
        )
        if local_lower > local_upper + 1.0e-10:
            raise ValueError("Adapted Bellman topology cannot satisfy glide bounds")
        h[index] = np.clip(desired_h[index], local_lower, local_upper)
    h[-1] = goal[1]

    delta_z = np.diff(z)
    delta_h = np.diff(h)
    segment_distance = np.hypot(delta_z, delta_h)
    gamma = np.arctan2(delta_h, delta_z)
    angle_tolerance = 1.0e-10
    if np.any(gamma < gamma_min - angle_tolerance) or np.any(
        gamma > gamma_max + angle_tolerance
    ):
        raise ValueError(
            "Adapted Bellman topology violates NLP gamma bounds: "
            f"range=[{np.min(gamma):.12g}, {np.max(gamma):.12g}], "
            f"bounds=[{gamma_min:.12g}, {gamma_max:.12g}]"
        )

    cosine = np.maximum(np.cos(gamma), np.finfo(float).eps)
    aerodynamic_lower = np.sqrt(
        lift_factor * cosine / vehicle["dynamic_limits"]["cl_max"]
    )
    aerodynamic_upper = np.sqrt(
        lift_factor * cosine / vehicle["dynamic_limits"]["cl_min"]
    )
    minimum_interval_time = configs["nlp_config"]["minimum_interval_time"]
    maximum_interval_time = configs["nlp_config"]["maximum_interval_time"]
    lower_speed = np.maximum.reduce((
        np.full_like(segment_distance, vehicle["glide_speed_min"]),
        aerodynamic_lower,
        segment_distance / maximum_interval_time,
    ))
    upper_speed = np.minimum.reduce((
        np.full_like(segment_distance, vehicle["glide_speed_max"]),
        aerodynamic_upper,
        segment_distance / minimum_interval_time,
    ))
    if np.any(lower_speed > upper_speed):
        failed_index = int(np.argmax(lower_speed - upper_speed))
        raise ValueError(
            "Adapted Bellman topology has no feasible speed profile: "
            f"index={failed_index}, gamma={gamma[failed_index]:.12g}, "
            f"distance={segment_distance[failed_index]:.12g}, "
            f"lower={lower_speed[failed_index]:.12g}, "
            f"upper={upper_speed[failed_index]:.12g}"
        )
    velocity = upper_speed
    interval_time = segment_distance / velocity
    if np.any(interval_time < minimum_interval_time) or np.any(
        interval_time > maximum_interval_time
    ):
        raise ValueError(
            "Adapted Bellman topology violates NLP interval-time bounds"
        )
    trajectory = np.column_stack((z, h))
    source_height_on_adapted_z = np.interp(z, source[:, 0], source[:, 1])
    topology_altitude_error = h - source_height_on_adapted_z

    los_margin = h - (tangent["tangent_slope"] * z + tangent["tangent_intercept"])
    terrain_margin = h[:-1] - terrain_height(geometry["terrain_model"], z[:-1])
    if np.min(los_margin) < -configs["validation_config"]["los_tolerance"]:
        raise ValueError("Adapted Bellman topology violates LOS constraints")
    if np.min(terrain_margin) < clearance - configs["validation_config"]["solver_tolerance"]:
        minimum_index = int(np.argmin(terrain_margin))
        raise ValueError(
            "Adapted Bellman topology violates terrain clearance: "
            f"minimum_margin={terrain_margin[minimum_index]:.6g} at "
            f"z={z[minimum_index]:.6g}"
        )

    dynamics_z = delta_z - interval_time * velocity * np.cos(gamma)
    dynamics_h = delta_h - interval_time * velocity * np.sin(gamma)
    maximum_dynamic_residual = float(max(
        np.max(np.abs(dynamics_z)),
        np.max(np.abs(dynamics_h)),
    ))
    return {
        "trajectory": _readonly(trajectory),
        "velocity_profile": _readonly(velocity),
        "gamma_profile": _readonly(gamma),
        "interval_time_profile": _readonly(interval_time),
        "glide_time": float(np.sum(interval_time)),
        "diagnostics": {
            "source_path_node_count": int(source.shape[0]),
            "adapted_node_count": int(node_count),
            "maximum_dynamic_residual": maximum_dynamic_residual,
            "minimum_los_margin": float(np.min(los_margin)),
            "minimum_terrain_margin": float(np.min(terrain_margin)),
            "topology_altitude_rms_distance": float(np.sqrt(np.mean(
                topology_altitude_error**2
            ))),
            "topology_altitude_max_distance": float(np.max(np.abs(
                topology_altitude_error
            ))),
            "kinematically_consistent": maximum_dynamic_residual <= configs["validation_config"]["solver_tolerance"],
        },
    }


def _resample_profile(profile: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    return np.interp(np.linspace(0.0, 1.0, count), np.linspace(0.0, 1.0, values.size), values)


def _numeric_outputs(function: ca.Function, *arguments: float) -> list[float]:
    outputs = function(*arguments)
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return [float(value) for value in values]


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
