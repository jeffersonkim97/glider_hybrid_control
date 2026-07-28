"""Algorithm-independent continuous Defender and nested Stackelberg solver."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Protocol

import numpy as np
from scipy.optimize import direct, minimize_scalar

from .bellman import generate_bellman_candidates, select_authoritative_bellman_response
from .continuous_replay_evaluation import replay_glide_continuous
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry_bundle, terrain_height
from .stage_cost import construct_stage_cost_4d
from .successor_grid_solver import solve_successor_grid_attacker

# An asymptotically-global search (DIRECT) samples across the *whole* bounded
# interval, including geometrically degenerate sensor positions where no
# Attacker path can reach the goal at all (e.g. a sensor sitting exactly on
# a ridge crest with terrain that otherwise blocks every switching point).
# Such a position is a real, valid outcome of the search space, not a bug,
# so it must be scored as "far worse than any real objective" rather than
# crashing the whole outer search. Real defender_objective values are
# normalized combinations of probabilities/areas in [0, 1], so this finite
# (not inf/nan, to stay JSON-portable) sentinel is unambiguously dominated.
_INFEASIBLE_DEFENDER_OBJECTIVE = -1.0e6


class DefenderOptimizer(Protocol):
    """Implementation-independent outer optimizer callable contract."""

    def __call__(
        self,
        evaluate: Callable[[float], dict[str, Any]],
        bounds: tuple[float, float],
        options: dict[str, Any],
    ) -> dict[str, Any]: ...


def build_defender_optimizer_interface(
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Describe the required continuous maximizer without choosing an algorithm."""
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    defender = configuration_bundle["primary_result"]["defender_config"]
    bounds = defender["continuous_search_bounds"]
    return {
        "primary_result": {
            "call_signature": "optimizer(evaluate, bounds, options) -> result",
            "evaluation_signature": "evaluate(z_sensor: float) -> DefenderEvaluation",
            "required_result_keys": ("z_sensor", "converged", "metadata"),
            "objective_direction": "maximize",
            "bounds": (bounds["z_sensor_min"], bounds["z_sensor_max"]),
            "continuous_decision_variable": "z_sensor",
            "dependent_sensor_height": "terrain(z_sensor) + mount_height",
            "configured_optimizer": defender["optimizer"],
        },
        "validation": {
            "passed": defender["optimizer"] == "hierarchical_coarse_to_fine_brent",
            "checks": {"default_optimizer_configured": defender["optimizer"] == "hierarchical_coarse_to_fine_brent"},
            "metrics": {"decision_dimension": 1},
            "summary": "Algorithm-independent Defender optimizer interface ready",
        },
        "metadata": {
            "schema_name": "DefenderOptimizerInterface",
            "schema_version": "1.0.0",
            "producer_phase": 9,
        },
        "status": {
            "success": defender["optimizer"] == "hierarchical_coarse_to_fine_brent",
            "code": "OK" if defender["optimizer"] == "hierarchical_coarse_to_fine_brent" else "OPTIMIZER_NOT_CONFIGURED",
            "message": "Defender optimizer interface ready",
            "warnings": [],
            "failed_checks": [] if defender["optimizer"] == "hierarchical_coarse_to_fine_brent" else ["default_optimizer_configured"],
        },
    }


def solve_attacker_best_response(
    z_sensor: float,
    configuration_bundle: dict[str, Any],
    evaluation_id: str = "attacker-best-response",
) -> dict[str, Any]:
    """Execute the one authoritative Bellman Attacker computation.

    The Attacker best response is the Bellman-optimal path on the
    discretized switching-point and state-action grid. No CasADi/IPOPT NLP
    is used here; see `attacker_nlp.py` for the deprecated, disconnected
    continuous-refinement experiment.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if isinstance(z_sensor, bool) or not np.isscalar(z_sensor) or not np.isfinite(z_sensor):
        raise TypeError("z_sensor must be one finite continuous scalar")
    nested_configuration = _configuration_for_sensor(configuration_bundle, float(z_sensor))
    defender = nested_configuration["primary_result"]["defender_config"]
    bounds = defender["continuous_search_bounds"]
    if not bounds["z_sensor_min"] <= float(z_sensor) <= bounds["z_sensor_max"]:
        raise ValueError("z_sensor lies outside continuous Defender bounds")

    geometry = build_geometry_bundle(nested_configuration)
    detection = build_symbolic_detection_bundle(nested_configuration, geometry)
    stage = construct_stage_cost_4d(nested_configuration, geometry, detection)
    transition_model = nested_configuration["primary_result"][
        "attacker_solver_config"
    ]["transition_model"]
    if transition_model == "snapped_fixed_time_step":
        bellman = generate_bellman_candidates(
            nested_configuration, geometry, detection, stage, None
        )
        attacker = select_authoritative_bellman_response(
            bellman, nested_configuration
        )
    elif transition_model == "successor_grid_physical_edge":
        bellman, attacker = solve_successor_grid_attacker(
            nested_configuration, geometry, detection
        )
    else:
        raise ValueError(f"Unsupported attacker transition_model: {transition_model}")
    if not attacker["status"]["success"]:
        raise RuntimeError(attacker["status"]["message"])
    best = attacker["primary_result"]
    configs = nested_configuration["primary_result"]
    geometry_result = geometry["primary_result"]
    continuous_replay = replay_glide_continuous(
        best["switching_point"],
        best["speed_profile"],
        best["gamma_profile"],
        time_step=configs["vehicle_config"]["time_step"],
        goal_position=geometry_result["goal_position"],
        goal_radius=configs["validation_config"]["goal_radius"],
        terrain_model=geometry_result["terrain_model"],
        los_geometry=geometry_result["los_geometry"],
        sensor_position=geometry_result["sensor_position"],
        glide_detection_rate_function=detection["primary_result"]["functions"][
            "glide_detection_components"
        ],
        terrain_tolerance=configs["validation_config"]["terrain_tolerance"],
        segment_check_count=configs["bellman_config"]["search_options"][
            "segment_check_count"
        ],
        max_steps=configs["environment_config"]["simulation"]["max_path_steps"],
        reference_trajectory=best["trajectory"],
        duration_profile=best.get("duration_profile"),
    )
    best["continuous_replay_validation"] = continuous_replay
    best["metadata"]["continuous_feasibility_checked"] = True
    best["metadata"]["continuous_feasibility_passed"] = continuous_replay[
        "feasible"
    ]
    best["metadata"]["continuous_feasibility_guaranteed_by_bellman"] = False
    attacker["metadata"]["continuous_feasibility_checked"] = True
    attacker["metadata"]["continuous_feasibility_passed"] = continuous_replay[
        "feasible"
    ]
    if not continuous_replay["feasible"]:
        attacker["status"]["warnings"] = [
            *attacker["status"].get("warnings", []),
            "Discrete-optimal response failed unsnapped continuous replay: "
            f"{continuous_replay['violation']}",
        ]
    return {
        "primary_result": {
            "evaluation_id": evaluation_id,
            "z_sensor": float(z_sensor),
            "configuration_bundle": nested_configuration,
            "geometry_bundle": geometry,
            "detection_bundle": detection,
            "stage_cost_4d_bundle": stage,
            "bellman_candidate_bundle": bellman,
            "bellman_response_bundle": attacker,
            "best_found_attacker_response": attacker["primary_result"],
        },
        "validation": attacker["validation"],
        "metadata": {
            "schema_name": "AuthoritativeAttackerBestResponseBundle",
            "schema_version": "1.0.0",
            "producer_phase": 8,
            "same_computation_for_fixed_and_outer_evaluations": True,
            "projected_cost_used_for_policy": False,
            "attacker_objective_id": attacker["metadata"][
                "attacker_objective_id"
            ],
            "solution_method": attacker["metadata"]["solution_method"],
            "optimality_scope": attacker["metadata"]["optimality_scope"],
            "transition_model": transition_model,
        },
        "status": dict(attacker["status"]),
    }


def evaluate_defender_position(
    z_sensor: float,
    configuration_bundle: dict[str, Any],
    evaluation_id: str = "defender-evaluation",
) -> dict[str, Any]:
    """Evaluate one Defender position using the authoritative Attacker solve."""
    attacker_pipeline = solve_attacker_best_response(
        z_sensor,
        configuration_bundle,
        evaluation_id=f"{evaluation_id}-attacker",
    )
    pipeline = attacker_pipeline["primary_result"]
    nested_configuration = pipeline["configuration_bundle"]
    geometry = pipeline["geometry_bundle"]
    detection = pipeline["detection_bundle"]
    stage = pipeline["stage_cost_4d_bundle"]
    bellman = pipeline["bellman_candidate_bundle"]
    attacker = pipeline["bellman_response_bundle"]
    best = attacker["primary_result"]
    coverage = geometry["primary_result"]["coverage"]
    defender_components = evaluate_defender_objective(
        best,
        coverage,
        detection,
    )
    pod_normalized = defender_components["pod_normalized"]
    coverage_normalized = defender_components["coverage_area_normalized"]
    defender_objective = defender_components["defender_objective"]
    weights = nested_configuration["primary_result"]["cost_config"]["defender"]
    sensor_position = geometry["primary_result"]["sensor_position"]
    validation = validate_defender_evaluation(
        float(z_sensor),
        sensor_position,
        best,
        coverage,
        defender_objective,
        nested_configuration,
        geometry,
        pod_normalized,
    )
    return {
        "primary_result": {
            "evaluation_id": evaluation_id,
            "z_sensor": float(z_sensor),
            "h_sensor": float(sensor_position[1]),
            "sensor_position": sensor_position,
            "best_found_attacker_response": best,
            "attacker_best_response_bundle": attacker_pipeline,
            "attacker_response_summary": attacker["validation"]["metrics"],
            "coverage": coverage,
            "coverage_maps": {
                "los_mask": geometry["primary_result"]["los_masks"]["los_mask"],
                "occlusion_mask": geometry["primary_result"]["los_masks"]["occlusion_mask"],
                "terrain_mask": geometry["primary_result"]["los_masks"]["terrain_mask"],
            },
            "visualization_payload": _build_defender_visualization_payload(
                geometry, bellman, evaluation_id
            ),
            "defender_objective": defender_objective,
            "objective_breakdown": {
                "defender_pod_normalized": pod_normalized,
                "mission_pod_probability": best["mission_pod"],
                "coverage_area": coverage["coverage_area"],
                "coverage_area_normalized": coverage_normalized,
                "weighted_pod": weights["w_pod"] * pod_normalized,
                "weighted_coverage": weights["w_cover"] * coverage_normalized,
                "total": defender_objective,
            },
            "nested_pipeline_execution": (
                "geometry", "detection", "stage_cost_4d", "bellman",
                "bellman_optimal_response",
            ),
        },
        "validation": validation,
        "metadata": {
            "schema_name": "DefenderEvaluation",
            "schema_version": "1.0.0",
            "producer_phase": 9,
            "fresh_nested_attacker_solve": True,
            "authoritative_attacker_solver": "solve_attacker_best_response",
            "attacker_solution_source": "Bellman-optimal Attacker Response",
            "stage_cost_schema": stage["metadata"]["schema_name"],
            "bellman_schema": bellman["metadata"]["schema_name"],
            "bellman_response_schema": attacker["metadata"]["schema_name"],
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "DEFENDER_EVALUATION_INVALID",
            "message": validation["summary"],
            "warnings": [],
            "failed_checks": validation["failed_checks"],
        },
    }


def evaluate_defender_objective(
    best_attacker_response: dict[str, Any],
    coverage: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> dict[str, float]:
    """Evaluate the one authoritative Defender objective function."""
    function = detection_bundle["primary_result"]["functions"][
        "defender_objective"
    ]
    raw_outputs = function(
        best_attacker_response["powered_hazard"],
        best_attacker_response["glide_hazard"],
        coverage["normalized_coverage_area"],
    )
    outputs = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)
    pod_normalized, coverage_normalized, objective = (
        float(output) for output in outputs
    )
    return {
        "pod_normalized": pod_normalized,
        "coverage_area_normalized": coverage_normalized,
        "defender_objective": objective,
    }


def _build_defender_visualization_payload(
    geometry_bundle: dict[str, Any],
    bellman_bundle: dict[str, Any],
    evaluation_id: str,
) -> dict[str, Any]:
    """Retain the exact final-evaluation arrays required by Figure 5."""
    geometry = geometry_bundle["primary_result"]
    bellman = bellman_bundle["primary_result"]
    ordering = bellman["cost_to_go_primary_ordering"]
    terrain = geometry["terrain_arrays"]
    los = geometry["los_geometry"]
    masks = geometry["los_masks"]
    return {
        "evaluation_id": evaluation_id,
        "z_sensor": float(geometry["sensor_position"][0]),
        "terrain_z": terrain["z"],
        "terrain_height": terrain["height"],
        "terrain_h_grid": terrain["h_grid"],
        "terrain_mask": masks["terrain_mask"],
        "los_mask": masks["los_mask"],
        "occlusion_mask": masks["occlusion_mask"],
        "sensor_position": geometry["sensor_position"],
        "goal_position": geometry["goal_position"],
        "tangent_point": los["tangent_point"],
        "tangent_slope": float(los["tangent_slope"]),
        "tangent_intercept": float(los["tangent_intercept"]),
        "tangent_line_height": los["tangent_line_height"],
        "cost_to_go": bellman["cost_to_go_maps"][ordering],
        "pod_to_go": bellman["pod_to_go_maps"][ordering],
        "cost_to_go_ordering": ordering,
        "geometry_schema": geometry_bundle["metadata"]["schema_name"],
        "bellman_schema": bellman_bundle["metadata"]["schema_name"],
    }


def solve_stackelberg_game(
    configuration_bundle: dict[str, Any],
    optimizer: DefenderOptimizer | None = None,
) -> dict[str, Any]:
    """Run an injected continuous maximizer around fresh nested evaluations."""
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    defender = configuration_bundle["primary_result"]["defender_config"]
    if optimizer is None:
        if defender["optimizer"] != "hierarchical_coarse_to_fine_brent":
            raise ValueError("Unsupported default Defender optimizer")
        optimizer = hierarchical_coarse_to_fine_optimizer
    if not callable(optimizer):
        raise TypeError("optimizer must implement the DefenderOptimizer interface")
    bounds_spec = defender["continuous_search_bounds"]
    bounds = (bounds_spec["z_sensor_min"], bounds_spec["z_sensor_max"])
    evaluation_summaries: list[dict[str, Any]] = []
    evaluation_counter = 0

    def fresh_evaluation(z_sensor: float) -> dict[str, Any]:
        nonlocal evaluation_counter
        evaluation_counter += 1
        evaluation_id = f"outer-evaluation-{evaluation_counter:04d}"
        try:
            evaluation = evaluate_defender_position(
                float(z_sensor), configuration_bundle, evaluation_id=evaluation_id,
            )
        except (ValueError, RuntimeError) as error:
            # No Attacker path reaches the goal from this z_sensor (e.g. the
            # LOS boundary never re-enters the airspace altitude band): a
            # real, scoreable search-space outcome, not a crash.
            evaluation_summaries.append({
                "evaluation_id": evaluation_id,
                "z_sensor": float(z_sensor),
                # h_sensor/attacker_objective/mission_pod/defender_pod_normalized/
                # coverage_area_normalized are NaN, not None: exports build
                # np.asarray([item[key] for item in summaries]) over these
                # fields, and a bare None would silently degrade that to an
                # object-dtype array. defender_objective alone stays a real
                # finite sentinel because optimizers compare/max() over it.
                "h_sensor": float("nan"),
                "defender_objective": _INFEASIBLE_DEFENDER_OBJECTIVE,
                "attacker_objective": float("nan"),
                "mission_pod": float("nan"),
                "defender_pod_normalized": float("nan"),
                "coverage_area_normalized": float("nan"),
                "fresh_nested_attacker_solve": True,
                "infeasible": True,
                "infeasibility_reason": str(error),
            })
            return {
                "primary_result": {
                    "evaluation_id": evaluation_id,
                    "z_sensor": float(z_sensor),
                    "infeasible": True,
                    "infeasibility_reason": str(error),
                },
                "validation": {"passed": False, "checks": {}, "metrics": {}, "failed_checks": ["attacker_best_response_exists"], "summary": str(error)},
                "metadata": {"schema_name": "DefenderEvaluation", "producer_phase": 9, "infeasible": True},
                "status": {
                    "success": False,
                    "code": "DEFENDER_EVALUATION_INFEASIBLE",
                    "message": str(error),
                    "warnings": [],
                    "failed_checks": ["attacker_best_response_exists"],
                },
            }
        primary = evaluation["primary_result"]
        evaluation_summaries.append({
            "evaluation_id": primary["evaluation_id"],
            "z_sensor": primary["z_sensor"],
            "h_sensor": primary["h_sensor"],
            "defender_objective": primary["defender_objective"],
            "attacker_objective": primary["best_found_attacker_response"]["mission_objective"],
            "mission_pod": primary["best_found_attacker_response"]["mission_pod"],
            "defender_pod_normalized": primary["objective_breakdown"][
                "defender_pod_normalized"
            ],
            "coverage_area_normalized": primary["coverage"]["normalized_coverage_area"],
            "fresh_nested_attacker_solve": True,
            "infeasible": False,
        })
        return evaluation

    optimizer_result = optimizer(
        fresh_evaluation,
        bounds,
        {
            "termination_tolerance": defender["termination_tolerance"],
            "xtol": defender["xtol"],
            "maximum_iterations": defender["maximum_iterations"],
            "coarse_sample_count": defender["coarse_sample_count"],
            "basin_prominence_threshold": defender["basin_prominence_threshold"],
            "objective_direction": "maximize",
        },
    )
    _validate_optimizer_result(optimizer_result, bounds)
    final_evaluation = fresh_evaluation(float(optimizer_result["z_sensor"]))
    if not final_evaluation["status"]["success"]:
        # The optimizer must never settle on an infeasible position: every
        # feasible point strictly dominates the -1e6 sentinel, so this can
        # only happen if no feasible z_sensor exists anywhere in bounds.
        # That is a real, reportable failure, not something to paper over.
        raise RuntimeError(
            "Outer optimizer selected an infeasible z_sensor as its final "
            f"answer: {final_evaluation['status']['message']}"
        )
    final_primary = final_evaluation["primary_result"]
    best = final_primary["best_found_attacker_response"]
    final_solution = {
        "optimal_z_sensor": final_primary["z_sensor"],
        "optimal_h_sensor": final_primary["h_sensor"],
        "optimal_sensor_position": final_primary["sensor_position"],
        "optimal_attacker_strategy": best,
        "optimal_switching_point": best["switching_point"],
        "optimal_glide_trajectory": best["trajectory"],
        "mission_pod": best["mission_pod"],
        "coverage_area": final_primary["coverage"]["coverage_area"],
        "coverage_area_normalized": final_primary["coverage"]["normalized_coverage_area"],
        "defender_pod_normalized": final_primary["objective_breakdown"]["defender_pod_normalized"],
        "attacker_objective": best["mission_objective"],
        "defender_objective": final_primary["defender_objective"],
        "objective_breakdown": final_primary["objective_breakdown"],
        "coverage_maps": final_primary["coverage_maps"],
        "visualization_payload": final_primary["visualization_payload"],
    }
    validation = validate_stackelberg_solution(
        optimizer_result,
        final_solution,
        final_evaluation,
        evaluation_summaries,
        configuration_bundle,
    )
    return {
        "primary_result": {
            "final_stackelberg_solution": final_solution,
            "best_found_attacker_response": best,
            "final_defender_evaluation": final_evaluation,
            "outer_evaluation_summaries": tuple(evaluation_summaries),
            "outer_optimizer_result": optimizer_result,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "StackelbergSolutionBundle",
            "schema_version": "1.0.0",
            "producer_phase": 9,
            "producer_module": "p1b_4D.stackelberg_solver",
            "outer_optimizer_algorithm": optimizer_result["metadata"].get("algorithm"),
            "outer_optimizer_injected": optimizer is not hierarchical_coarse_to_fine_optimizer,
            "defender_decision_continuous": True,
            "sensor_height_independent_variable": False,
            "fresh_nested_solve_per_evaluation": True,
            "global_attacker_optimum_claim": False,
            "figure5_evaluation_id": final_primary[
                "visualization_payload"
            ]["evaluation_id"],
            "figure5_cost_to_go_ordering": final_primary[
                "visualization_payload"
            ]["cost_to_go_ordering"],
            "figure5_uses_final_defender_evaluation_only": True,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "STACKELBERG_SOLUTION_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def hierarchical_coarse_to_fine_optimizer(
    evaluate: Callable[[float], dict[str, Any]],
    bounds: tuple[float, float],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Run configurable coarse sweep, basin detection, and bounded Brent search."""
    coarse_z = np.linspace(bounds[0], bounds[1], int(options["coarse_sample_count"]))
    coarse_results: list[dict[str, Any]] = []
    for z_sensor in coarse_z:
        evaluation = evaluate(float(z_sensor))["primary_result"]
        coarse_results.append(_outer_result_summary(evaluation))
    objectives = np.asarray([item["defender_objective"] for item in coarse_results])
    basins = detect_candidate_basins(
        coarse_z,
        objectives,
        float(options["basin_prominence_threshold"]),
    )
    brent_history: list[dict[str, Any]] = []
    basin_solutions: list[dict[str, Any]] = []
    for basin_index, basin in enumerate(basins):
        def negative_objective(z_sensor: float) -> float:
            evaluation = evaluate(float(z_sensor))["primary_result"]
            summary = _outer_result_summary(evaluation)
            summary["basin_index"] = basin_index
            summary["brent_evaluation_index"] = len(brent_history)
            brent_history.append(summary)
            return -summary["defender_objective"]

        optimized = minimize_scalar(
            negative_objective,
            bounds=(basin["left_boundary"], basin["right_boundary"]),
            method="bounded",
            options={
                "xatol": float(options["xtol"]),
                "maxiter": int(options["maximum_iterations"]),
            },
        )
        basin_solutions.append({
            "basin_index": basin_index,
            "z_sensor": float(optimized.x),
            "defender_objective": float(-optimized.fun),
            "converged": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "function_evaluations": int(optimized.nfev),
        })
    evaluated_candidates = [
        {
            "source": "coarse_sweep",
            "evaluation_id": item["evaluation_id"],
            "z_sensor": item["z_sensor"],
            "defender_objective": item["defender_objective"],
            "converged": True,
            "basin_index": None,
        }
        for item in coarse_results
    ] + [
        {
            "source": "brent_evaluation",
            "evaluation_id": item["evaluation_id"],
            "z_sensor": item["z_sensor"],
            "defender_objective": item["defender_objective"],
            "converged": True,
            "basin_index": item["basin_index"],
        }
        for item in brent_history
    ]
    selected = max(
        evaluated_candidates,
        key=lambda item: (item["defender_objective"], -item["z_sensor"]),
    )
    return {
        "z_sensor": selected["z_sensor"],
        "converged": bool(selected["converged"]),
        "metadata": {
            "algorithm": "hierarchical_coarse_to_fine_brent",
            "scipy_method": "bounded",
            "objective_direction": "maximize_via_negative_minimization",
            "coarse_sample_count": len(coarse_results),
            "detected_basin_count": len(basins),
            "brent_evaluation_count": len(brent_history),
            "selected_basin_index": selected["basin_index"],
            "selected_candidate_source": selected["source"],
            "xtol": options["xtol"],
        },
        "coarse_sweep_results": tuple(coarse_results),
        "detected_basins": tuple(basins),
        "brent_optimization_history": tuple(brent_history),
        "basin_solutions": tuple(basin_solutions),
        "selected_basin_solution": selected,
        "evaluated_candidate_solutions": tuple(evaluated_candidates),
    }


def detect_candidate_basins(
    z_values: np.ndarray,
    objectives: np.ndarray,
    prominence_threshold: float,
) -> tuple[dict[str, float], ...]:
    """Detect endpoint and interior local maxima and form bounded local basins."""
    if z_values.ndim != 1 or objectives.shape != z_values.shape or z_values.size < 3:
        raise ValueError("Coarse basin detection requires aligned one-dimensional arrays")
    peak_indices = []
    for index in range(z_values.size):
        left = objectives[index - 1] if index > 0 else -np.inf
        right = objectives[index + 1] if index < z_values.size - 1 else -np.inf
        if objectives[index] >= left and objectives[index] >= right:
            peak_indices.append(index)
    basins: list[dict[str, float]] = []
    for index in peak_indices:
        left_index = max(0, index - 1)
        right_index = min(z_values.size - 1, index + 1)
        boundary_reference = max(
            objectives[left_index] if left_index != index else -np.inf,
            objectives[right_index] if right_index != index else -np.inf,
        )
        prominence = (
            float(objectives[index] - boundary_reference)
            if np.isfinite(boundary_reference)
            else float("inf")
        )
        if prominence >= prominence_threshold:
            basins.append({
                "left_boundary": float(z_values[left_index]),
                "right_boundary": float(z_values[right_index]),
                "peak_location": float(z_values[index]),
                "peak_objective": float(objectives[index]),
                "prominence": prominence,
                "coarse_peak_index": int(index),
            })
    if not basins:
        index = int(np.argmax(objectives))
        basins.append({
            "left_boundary": float(z_values[max(0, index - 1)]),
            "right_boundary": float(z_values[min(z_values.size - 1, index + 1)]),
            "peak_location": float(z_values[index]),
            "peak_objective": float(objectives[index]),
            "prominence": 0.0,
            "coarse_peak_index": index,
        })
    return tuple(basins)


def direct_global_optimizer(
    evaluate: Callable[[float], dict[str, Any]],
    bounds: tuple[float, float],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Asymptotically-global search over z_sensor using SciPy's DIRECT.

    Unlike `hierarchical_coarse_to_fine_optimizer` (a fixed coarse sample
    followed by local Brent refinement within detected basins, with no
    guarantee against missing a basin the coarse sample skipped over),
    DIRECT adaptively partitions the bounded interval and is provably
    convergent to the global optimum as its evaluation budget grows, for
    objectives that are at least approximately Lipschitz continuous. It is
    used here with `locally_biased=False` (the original, unbiased DIRECT
    variant) since the goal is global coverage, not fast local convergence.

    That convergence guarantee is asymptotic (budget -> infinity), not a
    finite-sample certificate: this function's *return value* is always a
    best-found result within the configured evaluation budget, not a
    certified global optimum, regardless of whether DIRECT's own len_tol
    stopping criterion was reached. See `metadata["terminated_via"]` for
    which happened on this run.

    This is a drop-in `DefenderOptimizer`: it treats `evaluate(z_sensor)`
    as a black box and does not know or care how the Attacker best response
    inside it was computed.
    """
    evaluation_history: list[dict[str, Any]] = []

    def negative_objective(x: np.ndarray) -> float:
        z_sensor = float(x[0])
        evaluation = evaluate(z_sensor)["primary_result"]
        summary = _outer_result_summary(evaluation)
        summary["evaluation_index"] = len(evaluation_history)
        evaluation_history.append(summary)
        return -summary["defender_objective"]

    result = direct(
        negative_objective,
        bounds=[bounds],
        maxfun=int(options.get("direct_maxfun", 80)),
        maxiter=int(options.get("direct_maxiter", 200)),
        len_tol=float(options.get("direct_len_tol", 1.0e-4)),
        locally_biased=False,
    )
    best = max(evaluation_history, key=lambda item: item["defender_objective"])
    return {
        "z_sensor": float(result.x[0]),
        # `converged` reflects DIRECT's own len_tol/vol_tol stopping
        # criterion (result.success), not evaluation-budget exhaustion.
        # For an expensive real-world objective (a full Bellman solve per
        # evaluation) it is normal and expected to terminate on `maxfun`
        # instead -- that is still a valid, meaningful best-found result,
        # not a failure, but it is honestly a *different* outcome than
        # reaching the algorithm's own convergence tolerance, and callers
        # (e.g. validate_stackelberg_solution) that warn on non-
        # convergence should see the real signal instead of a constant.
        "converged": bool(result.success),
        "metadata": {
            "algorithm": "scipy_direct_global",
            "objective_direction": "maximize_via_negative_minimization",
            # DIRECT's global-convergence guarantee is asymptotic (budget
            # -> infinity); no finite run -- regardless of `converged`
            # above -- constitutes a certificate of global optimality, so
            # this field describes the algorithm's theoretical class, not
            # a claim about this particular result. Downstream code and
            # the paper should say "best-found, asymptotically-global
            # algorithm", never "certified global".
            "algorithm_class": "direct_asymptotically_global",
            "terminated_via": "length_tolerance" if result.success else "evaluation_budget",
            "locally_biased": False,
            "function_evaluations": int(result.nfev),
            "iterations": int(result.nit),
            "termination_message": str(result.message),
            "direct_reported_success": bool(result.success),
        },
        "evaluation_history": tuple(evaluation_history),
        "evaluated_candidate_solutions": tuple(evaluation_history),
        "selected_evaluation": best,
    }


def _outer_result_summary(primary: dict[str, Any]) -> dict[str, Any]:
    if primary.get("infeasible"):
        return {
            "evaluation_id": primary["evaluation_id"],
            "z_sensor": primary["z_sensor"],
            "h_sensor": float("nan"),
            "defender_objective": _INFEASIBLE_DEFENDER_OBJECTIVE,
            "coverage_area": float("nan"),
            "coverage_area_normalized": float("nan"),
            "defender_pod_normalized": float("nan"),
            "mission_pod": float("nan"),
            "attacker_objective": float("nan"),
            "best_found_attacker_response": None,
            "infeasible": True,
        }
    best = primary["best_found_attacker_response"]
    return {
        "evaluation_id": primary["evaluation_id"],
        "z_sensor": primary["z_sensor"],
        "h_sensor": primary["h_sensor"],
        "defender_objective": primary["defender_objective"],
        "coverage_area": primary["coverage"]["coverage_area"],
        "coverage_area_normalized": primary["coverage"]["normalized_coverage_area"],
        "defender_pod_normalized": primary["objective_breakdown"][
            "defender_pod_normalized"
        ],
        "mission_pod": best["mission_pod"],
        "attacker_objective": best["mission_objective"],
        "best_found_attacker_response": {
            "solution_id": best["solution_id"],
            "switching_point": best["switching_point"],
            "mission_objective": best["mission_objective"],
            "mission_pod": best["mission_pod"],
            "mission_time": best["mission_time"],
        },
        "infeasible": False,
    }


def validate_defender_evaluation(
    z_sensor: float, sensor_position: np.ndarray, best: dict[str, Any],
    coverage: dict[str, Any], objective: float,
    configuration_bundle: dict[str, Any], geometry_bundle: dict[str, Any],
    defender_pod_normalized: float,
) -> dict[str, Any]:
    configs = configuration_bundle["primary_result"]
    weights = configs["cost_config"]["defender"]
    expected_height = float(terrain_height(
        geometry_bundle["primary_result"]["terrain_model"], z_sensor
    )) + configs["sensor_config"]["mount_height"]
    mission_hazard = best["powered_hazard"] + best["glide_hazard"]
    pod_specification = weights["normalization"]["pod"]
    if pod_specification["method"] == "hazard_reference":
        reference = pod_specification["hazard_reference"]
        expected_defender_pod = mission_hazard / (mission_hazard + reference)
    elif pod_specification["method"] == "probability":
        expected_defender_pod = 1.0 - np.exp(-mission_hazard)
    else:
        raise ValueError("Unsupported Defender PoD normalization")
    reconstructed = weights["w_pod"] * defender_pod_normalized + weights["w_cover"] * coverage["normalized_coverage_area"]
    expected_mission_pod = 1.0 - np.exp(-mission_hazard)
    goal_error_norm = best["constraint_residuals"]["goal_error_norm"]
    checks = {
        "attacker_convergence": best["validation"]["passed"],
        "attacker_objective_consistency": best["validation"]["checks"][
            "objective_consistency"
        ],
        "attacker_goal_region_reached": goal_error_norm <= (
            configs["validation_config"]["goal_radius"]
            + configs["validation_config"]["solver_tolerance"]
        ),
        "geometry_consistency": abs(sensor_position[0] - z_sensor) <= configs["validation_config"]["solver_tolerance"] and abs(sensor_position[1] - expected_height) <= configs["validation_config"]["terrain_tolerance"],
        "coverage_consistency": 0.0 <= coverage["normalized_coverage_area"] <= 1.0 and coverage["coverage_area"] >= 0.0,
        "mission_pod_consistency": abs(best["mission_pod"] - expected_mission_pod) <= configs["validation_config"]["detection_probability_tolerance"],
        "defender_pod_normalization_consistency": abs(defender_pod_normalized - expected_defender_pod) <= configs["validation_config"]["objective_tolerance"],
        "objective_consistency": abs(objective - reconstructed) <= configs["validation_config"]["objective_tolerance"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "metrics": {"expected_sensor_height": expected_height, "objective_residual": abs(objective - reconstructed), "mission_hazard": mission_hazard, "expected_defender_pod_normalized": expected_defender_pod, "goal_error_norm": goal_error_norm}, "failed_checks": failed, "summary": "Defender evaluation validation passed" if not failed else f"Defender evaluation failed checks: {failed}"}


def validate_stackelberg_solution(
    optimizer_result: dict[str, Any], solution: dict[str, Any],
    final_evaluation: dict[str, Any], summaries: list[dict[str, Any]],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    tolerance = configuration_bundle["primary_result"]["validation_config"]["objective_tolerance"]
    final_primary = final_evaluation["primary_result"]
    payload = solution["visualization_payload"]
    attacker_pipeline = final_primary["attacker_best_response_bundle"]
    attacker_bellman_response = attacker_pipeline["primary_result"][
        "bellman_response_bundle"
    ]
    switch = solution["optimal_switching_point"]
    tangent_residual = abs(
        switch[1]
        - float(np.interp(switch[0], payload["terrain_z"], payload["tangent_line_height"]))
    )
    pre_final_summaries = summaries[:-1] or summaries
    evaluated_maximum = max(
        item["defender_objective"] for item in pre_final_summaries
    )
    checks = {
        # Not a hard pass/fail gate: DIRECT terminating via evaluation-
        # budget exhaustion rather than its own length-tolerance criterion
        # is normal and expected for an expensive per-evaluation objective
        # (see direct_global_optimizer's docstring/comments) -- it is
        # surfaced as a warning below, not treated as a validation failure.
        "objective_consistency": abs(solution["defender_objective"] - final_primary["defender_objective"]) <= tolerance,
        "attacker_convergence": solution["optimal_attacker_strategy"]["validation"]["passed"],
        "authoritative_attacker_solver_used": attacker_pipeline["metadata"][
            "same_computation_for_fixed_and_outer_evaluations"
        ],
        "bellman_response_valid": attacker_bellman_response["validation"][
            "passed"
        ],
        "attacker_goal_region_reached": final_evaluation["validation"][
            "checks"
        ]["attacker_goal_region_reached"],
        "geometry_consistency": final_evaluation["validation"]["checks"]["geometry_consistency"],
        "coverage_consistency": final_evaluation["validation"]["checks"]["coverage_consistency"],
        "mission_pod_consistency": final_evaluation["validation"]["checks"]["mission_pod_consistency"],
        "fresh_nested_solve_every_evaluation": all(item["fresh_nested_attacker_solve"] for item in summaries),
        "selected_candidate_dominates_all_outer_evaluations": solution["defender_objective"] >= evaluated_maximum - tolerance,
        "switching_point_on_final_tangent": tangent_residual <= configuration_bundle["primary_result"]["validation_config"]["los_tolerance"],
        "figure5_sensor_matches_final_defender": np.allclose(payload["sensor_position"], solution["optimal_sensor_position"], rtol=0.0, atol=configuration_bundle["primary_result"]["validation_config"]["solver_tolerance"]),
        "figure5_evaluation_id_matches_final": payload["evaluation_id"] == final_primary["evaluation_id"],
        "figure5_arrays_consistent": payload["cost_to_go"].shape == payload["pod_to_go"].shape == payload["los_mask"].shape == payload["occlusion_mask"].shape == payload["terrain_mask"].shape,
    }
    failed = [name for name, passed in checks.items() if not passed]
    warnings = [] if optimizer_result["converged"] else [
        "Outer optimizer terminated via evaluation-budget exhaustion, not "
        "its own length-tolerance criterion -- a valid best-found result, "
        "not a certified global optimum for this run."
    ]
    return {"passed": not failed, "checks": checks, "metrics": {"outer_evaluation_count": len(summaries), "optimal_z_sensor": solution["optimal_z_sensor"], "defender_objective": solution["defender_objective"], "maximum_pre_final_evaluation_objective": evaluated_maximum, "switching_tangent_residual": tangent_residual, "figure5_evaluation_id": payload["evaluation_id"], "outer_optimizer_converged": bool(optimizer_result["converged"])}, "warnings": warnings, "failed_checks": failed, "summary": "Phase 9 Stackelberg solution validation passed" if not failed else f"Stackelberg solution failed checks: {failed}"}


def _configuration_for_sensor(bundle: dict[str, Any], z_sensor: float) -> dict[str, Any]:
    primary = bundle["primary_result"]
    cloned_primary = dict(primary)
    for key in (
        "environment_config", "vehicle_config", "sensor_config", "cost_config",
        "bellman_config", "nlp_config", "defender_config", "plot_config",
        "io_config", "validation_config",
    ):
        cloned_primary[key] = deepcopy(primary[key])
    cloned_primary["sensor_config"]["default_z_sensor"] = z_sensor
    return {**bundle, "primary_result": cloned_primary}


def _validate_optimizer_result(result: Any, bounds: tuple[float, float]) -> None:
    if not isinstance(result, dict):
        raise TypeError("optimizer result must be a dictionary")
    missing = {"z_sensor", "converged", "metadata"} - set(result)
    if missing:
        raise ValueError(f"optimizer result missing keys: {sorted(missing)}")
    if not np.isfinite(result["z_sensor"]) or not bounds[0] <= result["z_sensor"] <= bounds[1]:
        raise ValueError("optimizer result z_sensor is outside bounds")
    if not isinstance(result["converged"], bool) or not isinstance(result["metadata"], dict):
        raise TypeError("optimizer converged/metadata fields have invalid types")


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")
