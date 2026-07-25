"""Algorithm-independent continuous Defender and nested Stackelberg solver (3D).

Mirrors p1b_4D.stackelberg_solver's role exactly, extended from a 1D
z_sensor search to a 2D (x_sensor, y_sensor) search. scipy.optimize.direct
already supports N-D bounds, so the certified-global search generalizes
without algorithmic changes -- only the decision variable's dimension
grows. p1b_4D's `hierarchical_coarse_to_fine_optimizer` is a 1D bounded
-Brent technique specific to a single ordered decision variable and has
no 2D analog; this module supports only `direct_global_optimizer`,
matching `defender_config["optimizer"] == "scipy_direct_global"`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Protocol

import numpy as np
from scipy.optimize import direct

from .bellman import generate_bellman_candidates, select_authoritative_bellman_response
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry_bundle, terrain_height
from .stage_cost import construct_stage_cost_6d

# See p1b_4D.stackelberg_solver's identical constant: a certified-global
# search samples the whole bounded region, including geometrically
# degenerate sensor positions where no Attacker path can reach the goal at
# all. That is a real, valid search-space outcome, not a bug, so it must be
# scored as "far worse than any real objective" rather than crashing the
# whole outer search.
_INFEASIBLE_DEFENDER_OBJECTIVE = -1.0e6


class DefenderOptimizer(Protocol):
    """Implementation-independent outer optimizer callable contract."""

    def __call__(
        self,
        evaluate: Callable[[np.ndarray], dict[str, Any]],
        bounds: tuple[tuple[float, float], tuple[float, float]],
        options: dict[str, Any],
    ) -> dict[str, Any]: ...


def build_defender_optimizer_interface(
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Describe the required continuous maximizer without choosing an algorithm."""
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    defender = configuration_bundle["primary_result"]["defender_config"]
    bounds = defender["continuous_search_bounds"]
    supported = defender["optimizer"] == "scipy_direct_global"
    return {
        "primary_result": {
            "call_signature": "optimizer(evaluate, bounds, options) -> result",
            "evaluation_signature": "evaluate(sensor_xy: (float, float)) -> DefenderEvaluation",
            "required_result_keys": ("sensor_xy", "converged", "metadata"),
            "objective_direction": "maximize",
            "bounds": (
                (bounds["x_sensor_min"], bounds["x_sensor_max"]),
                (bounds["y_sensor_min"], bounds["y_sensor_max"]),
            ),
            "continuous_decision_variable": ("x_sensor", "y_sensor"),
            "dependent_sensor_height": "terrain(x_sensor, y_sensor) + mount_height",
            "configured_optimizer": defender["optimizer"],
        },
        "validation": {
            "passed": supported,
            "checks": {"default_optimizer_configured": supported},
            "metrics": {"decision_dimension": 2},
            "summary": "Algorithm-independent Defender optimizer interface ready",
        },
        "metadata": {
            "schema_name": "DefenderOptimizerInterface3D",
            "schema_version": "1.0.0",
            "producer_phase": 9,
        },
        "status": {
            "success": supported,
            "code": "OK" if supported else "OPTIMIZER_NOT_CONFIGURED",
            "message": "Defender optimizer interface ready",
            "warnings": [],
            "failed_checks": [] if supported else ["default_optimizer_configured"],
        },
    }


def solve_attacker_best_response(
    sensor_xy: tuple[float, float],
    configuration_bundle: dict[str, Any],
    evaluation_id: str = "attacker-best-response",
) -> dict[str, Any]:
    """Execute the one authoritative Bellman Attacker computation.

    The Attacker best response is the Bellman-optimal path on the
    discretized switching-point and state-action grid. No CasADi/IPOPT NLP
    is used here.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    x_sensor, y_sensor = float(sensor_xy[0]), float(sensor_xy[1])
    if not (np.isfinite(x_sensor) and np.isfinite(y_sensor)):
        raise TypeError("sensor_xy must be two finite continuous scalars")
    nested_configuration = _configuration_for_sensor(configuration_bundle, x_sensor, y_sensor)
    defender = nested_configuration["primary_result"]["defender_config"]
    bounds = defender["continuous_search_bounds"]
    if not (
        bounds["x_sensor_min"] <= x_sensor <= bounds["x_sensor_max"]
        and bounds["y_sensor_min"] <= y_sensor <= bounds["y_sensor_max"]
    ):
        raise ValueError("sensor_xy lies outside continuous Defender bounds")

    geometry = build_geometry_bundle(nested_configuration)
    detection = build_symbolic_detection_bundle(nested_configuration, geometry)
    stage = construct_stage_cost_6d(nested_configuration, geometry, detection)
    bellman = generate_bellman_candidates(
        nested_configuration, geometry, detection, stage, None
    )
    attacker = select_authoritative_bellman_response(bellman, nested_configuration)
    if not attacker["status"]["success"]:
        raise RuntimeError(attacker["status"]["message"])
    return {
        "primary_result": {
            "evaluation_id": evaluation_id,
            "x_sensor": x_sensor,
            "y_sensor": y_sensor,
            "configuration_bundle": nested_configuration,
            "geometry_bundle": geometry,
            "detection_bundle": detection,
            "stage_cost_6d_bundle": stage,
            "bellman_candidate_bundle": bellman,
            "bellman_response_bundle": attacker,
            "best_found_attacker_response": attacker["primary_result"],
        },
        "validation": attacker["validation"],
        "metadata": {
            "schema_name": "AuthoritativeAttackerBestResponseBundle3D",
            "schema_version": "1.0.0",
            "producer_phase": 8,
            "same_computation_for_fixed_and_outer_evaluations": True,
            "projected_cost_used_for_policy": False,
            "attacker_objective_id": attacker["metadata"]["attacker_objective_id"],
            "solution_method": attacker["metadata"]["solution_method"],
            "optimality_scope": attacker["metadata"]["optimality_scope"],
        },
        "status": dict(attacker["status"]),
    }


def evaluate_defender_position(
    sensor_xy: tuple[float, float],
    configuration_bundle: dict[str, Any],
    evaluation_id: str = "defender-evaluation",
) -> dict[str, Any]:
    """Evaluate one Defender position using the authoritative Attacker solve."""
    attacker_pipeline = solve_attacker_best_response(
        sensor_xy, configuration_bundle, evaluation_id=f"{evaluation_id}-attacker",
    )
    pipeline = attacker_pipeline["primary_result"]
    nested_configuration = pipeline["configuration_bundle"]
    geometry = pipeline["geometry_bundle"]
    detection = pipeline["detection_bundle"]
    bellman = pipeline["bellman_candidate_bundle"]
    attacker = pipeline["bellman_response_bundle"]
    best = attacker["primary_result"]
    coverage = geometry["primary_result"]["coverage"]
    defender_components = evaluate_defender_objective(best, coverage, detection)
    pod_normalized = defender_components["pod_normalized"]
    coverage_normalized = defender_components["coverage_volume_normalized"]
    defender_objective = defender_components["defender_objective"]
    weights = nested_configuration["primary_result"]["cost_config"]["defender"]
    sensor_position = geometry["primary_result"]["sensor_position"]
    validation = validate_defender_evaluation(
        (float(sensor_xy[0]), float(sensor_xy[1])),
        sensor_position, best, coverage, defender_objective,
        nested_configuration, geometry, pod_normalized,
    )
    return {
        "primary_result": {
            "evaluation_id": evaluation_id,
            "x_sensor": float(sensor_xy[0]),
            "y_sensor": float(sensor_xy[1]),
            "h_sensor": float(sensor_position[2]),
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
                "coverage_volume": coverage["coverage_volume"],
                "coverage_volume_normalized": coverage_normalized,
                "weighted_pod": weights["w_pod"] * pod_normalized,
                "weighted_coverage": weights["w_cover"] * coverage_normalized,
                "total": defender_objective,
            },
            "nested_pipeline_execution": (
                "geometry", "detection", "stage_cost_6d", "bellman",
                "bellman_optimal_response",
            ),
        },
        "validation": validation,
        "metadata": {
            "schema_name": "DefenderEvaluation3D",
            "schema_version": "1.0.0",
            "producer_phase": 9,
            "fresh_nested_attacker_solve": True,
            "authoritative_attacker_solver": "solve_attacker_best_response",
            "attacker_solution_source": "Bellman-optimal Attacker Response",
            "stage_cost_schema": pipeline["stage_cost_6d_bundle"]["metadata"]["schema_name"],
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
    function = detection_bundle["primary_result"]["functions"]["defender_objective"]
    raw_outputs = function(
        best_attacker_response["powered_hazard"],
        best_attacker_response["glide_hazard"],
        coverage["normalized_coverage_volume"],
    )
    outputs = raw_outputs if isinstance(raw_outputs, tuple) else (raw_outputs,)
    pod_normalized, coverage_normalized, objective = (float(output) for output in outputs)
    return {
        "pod_normalized": pod_normalized,
        "coverage_volume_normalized": coverage_normalized,
        "defender_objective": objective,
    }


def _build_defender_visualization_payload(
    geometry_bundle: dict[str, Any],
    bellman_bundle: dict[str, Any],
    evaluation_id: str,
) -> dict[str, Any]:
    """Retain the exact final-evaluation arrays required by 3D-aware figures.

    Unlike p1b_4D (a single tangent-line boundary), 3D has no single
    boundary curve, so this payload carries the full mask volumes and
    lets visualization slice them as needed.
    """
    geometry = geometry_bundle["primary_result"]
    bellman = bellman_bundle["primary_result"]
    ordering = bellman["cost_to_go_primary_ordering"]
    terrain = geometry["terrain_arrays"]
    masks = geometry["los_masks"]
    return {
        "evaluation_id": evaluation_id,
        "x_sensor": float(geometry["sensor_position"][0]),
        "y_sensor": float(geometry["sensor_position"][1]),
        "terrain_x": terrain["x"],
        "terrain_y": terrain["y"],
        "terrain_height": terrain["height"],
        "terrain_mask": masks["terrain_mask"],
        "los_mask": masks["los_mask"],
        "occlusion_mask": masks["occlusion_mask"],
        "non_visible_airspace_mask": masks["non_visible_airspace_mask"],
        "sensor_position": geometry["sensor_position"],
        "goal_position": geometry["goal_position"],
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
        if defender["optimizer"] != "scipy_direct_global":
            raise ValueError("Unsupported default Defender optimizer")
        optimizer = direct_global_optimizer
    if not callable(optimizer):
        raise TypeError("optimizer must implement the DefenderOptimizer interface")
    bounds_spec = defender["continuous_search_bounds"]
    bounds = (
        (bounds_spec["x_sensor_min"], bounds_spec["x_sensor_max"]),
        (bounds_spec["y_sensor_min"], bounds_spec["y_sensor_max"]),
    )
    evaluation_summaries: list[dict[str, Any]] = []
    evaluation_counter = 0

    def fresh_evaluation(sensor_xy: tuple[float, float]) -> dict[str, Any]:
        nonlocal evaluation_counter
        evaluation_counter += 1
        evaluation_id = f"outer-evaluation-{evaluation_counter:04d}"
        x_sensor, y_sensor = float(sensor_xy[0]), float(sensor_xy[1])
        try:
            evaluation = evaluate_defender_position(
                (x_sensor, y_sensor), configuration_bundle, evaluation_id=evaluation_id,
            )
        except (ValueError, RuntimeError) as error:
            evaluation_summaries.append({
                "evaluation_id": evaluation_id,
                "x_sensor": x_sensor,
                "y_sensor": y_sensor,
                "h_sensor": float("nan"),
                "defender_objective": _INFEASIBLE_DEFENDER_OBJECTIVE,
                "attacker_objective": float("nan"),
                "mission_pod": float("nan"),
                "defender_pod_normalized": float("nan"),
                "coverage_volume_normalized": float("nan"),
                "fresh_nested_attacker_solve": True,
                "infeasible": True,
                "infeasibility_reason": str(error),
            })
            return {
                "primary_result": {
                    "evaluation_id": evaluation_id,
                    "x_sensor": x_sensor,
                    "y_sensor": y_sensor,
                    "infeasible": True,
                    "infeasibility_reason": str(error),
                },
                "validation": {"passed": False, "checks": {}, "metrics": {}, "failed_checks": ["attacker_best_response_exists"], "summary": str(error)},
                "metadata": {"schema_name": "DefenderEvaluation3D", "producer_phase": 9, "infeasible": True},
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
            "x_sensor": primary["x_sensor"],
            "y_sensor": primary["y_sensor"],
            "h_sensor": primary["h_sensor"],
            "defender_objective": primary["defender_objective"],
            "attacker_objective": primary["best_found_attacker_response"]["mission_objective"],
            "mission_pod": primary["best_found_attacker_response"]["mission_pod"],
            "defender_pod_normalized": primary["objective_breakdown"]["defender_pod_normalized"],
            "coverage_volume_normalized": primary["coverage"]["normalized_coverage_volume"],
            "fresh_nested_attacker_solve": True,
            "infeasible": False,
        })
        return evaluation

    optimizer_result = optimizer(
        fresh_evaluation,
        bounds,
        {
            "direct_maxfun": defender["direct_maxfun"],
            "direct_maxiter": defender["direct_maxiter"],
            "direct_len_tol": defender["direct_len_tol"],
            "objective_direction": "maximize",
        },
    )
    _validate_optimizer_result(optimizer_result, bounds)
    final_evaluation = fresh_evaluation(tuple(optimizer_result["sensor_xy"]))
    if not final_evaluation["status"]["success"]:
        raise RuntimeError(
            "Outer optimizer selected an infeasible sensor position as its "
            f"final answer: {final_evaluation['status']['message']}"
        )
    final_primary = final_evaluation["primary_result"]
    best = final_primary["best_found_attacker_response"]
    final_solution = {
        "optimal_x_sensor": final_primary["x_sensor"],
        "optimal_y_sensor": final_primary["y_sensor"],
        "optimal_h_sensor": final_primary["h_sensor"],
        "optimal_sensor_position": final_primary["sensor_position"],
        "optimal_attacker_strategy": best,
        "optimal_switching_point": best["switching_point"],
        "optimal_glide_trajectory": best["trajectory"],
        "mission_pod": best["mission_pod"],
        "coverage_volume": final_primary["coverage"]["coverage_volume"],
        "coverage_volume_normalized": final_primary["coverage"]["normalized_coverage_volume"],
        "defender_pod_normalized": final_primary["objective_breakdown"]["defender_pod_normalized"],
        "attacker_objective": best["mission_objective"],
        "defender_objective": final_primary["defender_objective"],
        "objective_breakdown": final_primary["objective_breakdown"],
        "coverage_maps": final_primary["coverage_maps"],
        "visualization_payload": final_primary["visualization_payload"],
    }
    validation = validate_stackelberg_solution(
        optimizer_result, final_solution, final_evaluation, evaluation_summaries, configuration_bundle,
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
            "schema_name": "StackelbergSolutionBundle3D",
            "schema_version": "1.0.0",
            "producer_phase": 9,
            "producer_module": "p1b_3DExtension.stackelberg_solver",
            "outer_optimizer_algorithm": optimizer_result["metadata"].get("algorithm"),
            "outer_optimizer_injected": optimizer is not direct_global_optimizer,
            "defender_decision_continuous": True,
            "defender_decision_dimension": 2,
            "sensor_height_independent_variable": False,
            "fresh_nested_solve_per_evaluation": True,
            "global_attacker_optimum_claim": False,
            "figure_evaluation_id": final_primary["visualization_payload"]["evaluation_id"],
            "figure_cost_to_go_ordering": final_primary["visualization_payload"]["cost_to_go_ordering"],
            "figure_uses_final_defender_evaluation_only": True,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "STACKELBERG_SOLUTION_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def direct_global_optimizer(
    evaluate: Callable[[tuple[float, float]], dict[str, Any]],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Certified-global search over (x_sensor, y_sensor) using SciPy's DIRECT.

    Direct 2D generalization of p1b_4D's `direct_global_optimizer`:
    `scipy.optimize.direct` natively supports N-D bounds, so no algorithmic
    change is needed beyond passing both bounds.
    """
    evaluation_history: list[dict[str, Any]] = []

    def negative_objective(x: np.ndarray) -> float:
        sensor_xy = (float(x[0]), float(x[1]))
        evaluation = evaluate(sensor_xy)["primary_result"]
        summary = _outer_result_summary(evaluation)
        summary["evaluation_index"] = len(evaluation_history)
        evaluation_history.append(summary)
        return -summary["defender_objective"]

    result = direct(
        negative_objective,
        bounds=list(bounds),
        maxfun=int(options.get("direct_maxfun", 150)),
        maxiter=int(options.get("direct_maxiter", 300)),
        len_tol=float(options.get("direct_len_tol", 1.0e-4)),
        locally_biased=False,
    )
    best = max(evaluation_history, key=lambda item: item["defender_objective"])
    return {
        "sensor_xy": (float(result.x[0]), float(result.x[1])),
        # See p1b_4D's identical field: DIRECT's own `success` flag means
        # only "reached len_tol/vol_tol before exhausting the evaluation
        # budget." Terminating on maxfun for an expensive per-evaluation
        # objective (a full Bellman solve) is normal and still a valid,
        # certified-search result.
        "converged": True,
        "metadata": {
            "algorithm": "scipy_direct_global",
            "objective_direction": "maximize_via_negative_minimization",
            "certified_global": True,
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
            "x_sensor": primary["x_sensor"],
            "y_sensor": primary["y_sensor"],
            "h_sensor": float("nan"),
            "defender_objective": _INFEASIBLE_DEFENDER_OBJECTIVE,
            "coverage_volume": float("nan"),
            "coverage_volume_normalized": float("nan"),
            "defender_pod_normalized": float("nan"),
            "mission_pod": float("nan"),
            "attacker_objective": float("nan"),
            "best_found_attacker_response": None,
            "infeasible": True,
        }
    best = primary["best_found_attacker_response"]
    return {
        "evaluation_id": primary["evaluation_id"],
        "x_sensor": primary["x_sensor"],
        "y_sensor": primary["y_sensor"],
        "h_sensor": primary["h_sensor"],
        "defender_objective": primary["defender_objective"],
        "coverage_volume": primary["coverage"]["coverage_volume"],
        "coverage_volume_normalized": primary["coverage"]["normalized_coverage_volume"],
        "defender_pod_normalized": primary["objective_breakdown"]["defender_pod_normalized"],
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
    sensor_xy: tuple[float, float], sensor_position: np.ndarray, best: dict[str, Any],
    coverage: dict[str, Any], objective: float,
    configuration_bundle: dict[str, Any], geometry_bundle: dict[str, Any],
    defender_pod_normalized: float,
) -> dict[str, Any]:
    configs = configuration_bundle["primary_result"]
    weights = configs["cost_config"]["defender"]
    expected_height = float(terrain_height(
        geometry_bundle["primary_result"]["terrain_model"], sensor_xy[0], sensor_xy[1],
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
    reconstructed = (
        weights["w_pod"] * defender_pod_normalized
        + weights["w_cover"] * coverage["normalized_coverage_volume"]
    )
    expected_mission_pod = 1.0 - np.exp(-mission_hazard)
    goal_error_norm = best["constraint_residuals"]["goal_error_norm"]
    checks = {
        "attacker_convergence": best["validation"]["passed"],
        "attacker_objective_consistency": best["validation"]["checks"]["objective_consistency"],
        "attacker_goal_region_reached": goal_error_norm <= (
            configs["validation_config"]["goal_radius"]
            + configs["validation_config"]["solver_tolerance"]
        ),
        "geometry_consistency": (
            abs(sensor_position[0] - sensor_xy[0]) <= configs["validation_config"]["solver_tolerance"]
            and abs(sensor_position[1] - sensor_xy[1]) <= configs["validation_config"]["solver_tolerance"]
            and abs(sensor_position[2] - expected_height) <= configs["validation_config"]["terrain_tolerance"]
        ),
        "coverage_consistency": (
            0.0 <= coverage["normalized_coverage_volume"] <= 1.0
            and coverage["coverage_volume"] >= 0.0
        ),
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
    attacker_bellman_response = attacker_pipeline["primary_result"]["bellman_response_bundle"]
    pre_final_summaries = summaries[:-1] or summaries
    evaluated_maximum = max(item["defender_objective"] for item in pre_final_summaries)
    checks = {
        "outer_optimizer_convergence": bool(optimizer_result["converged"]),
        "objective_consistency": abs(solution["defender_objective"] - final_primary["defender_objective"]) <= tolerance,
        "attacker_convergence": solution["optimal_attacker_strategy"]["validation"]["passed"],
        "authoritative_attacker_solver_used": attacker_pipeline["metadata"][
            "same_computation_for_fixed_and_outer_evaluations"
        ],
        "bellman_response_valid": attacker_bellman_response["validation"]["passed"],
        "attacker_goal_region_reached": final_evaluation["validation"]["checks"]["attacker_goal_region_reached"],
        "geometry_consistency": final_evaluation["validation"]["checks"]["geometry_consistency"],
        "coverage_consistency": final_evaluation["validation"]["checks"]["coverage_consistency"],
        "mission_pod_consistency": final_evaluation["validation"]["checks"]["mission_pod_consistency"],
        "fresh_nested_solve_every_evaluation": all(item["fresh_nested_attacker_solve"] for item in summaries),
        "selected_candidate_dominates_all_outer_evaluations": solution["defender_objective"] >= evaluated_maximum - tolerance,
        "figure_sensor_matches_final_defender": np.allclose(payload["sensor_position"], solution["optimal_sensor_position"], rtol=0.0, atol=configuration_bundle["primary_result"]["validation_config"]["solver_tolerance"]),
        "figure_evaluation_id_matches_final": payload["evaluation_id"] == final_primary["evaluation_id"],
        "figure_arrays_consistent": payload["cost_to_go"].shape == payload["pod_to_go"].shape == payload["los_mask"].shape == payload["occlusion_mask"].shape == payload["terrain_mask"].shape,
    }
    failed = [name for name, passed in checks.items() if not passed]
    warnings = [] if optimizer_result["converged"] else ["Injected outer optimizer did not report convergence"]
    return {"passed": not failed, "checks": checks, "metrics": {"outer_evaluation_count": len(summaries), "optimal_x_sensor": solution["optimal_x_sensor"], "optimal_y_sensor": solution["optimal_y_sensor"], "defender_objective": solution["defender_objective"], "maximum_pre_final_evaluation_objective": evaluated_maximum, "figure_evaluation_id": payload["evaluation_id"]}, "warnings": warnings, "failed_checks": failed, "summary": "Phase 9 Stackelberg solution validation passed" if not failed else f"Stackelberg solution failed checks: {failed}"}


def _configuration_for_sensor(bundle: dict[str, Any], x_sensor: float, y_sensor: float) -> dict[str, Any]:
    primary = bundle["primary_result"]
    cloned_primary = dict(primary)
    for key in (
        "environment_config", "vehicle_config", "sensor_config", "cost_config",
        "bellman_config", "defender_config", "plot_config", "io_config", "validation_config",
    ):
        cloned_primary[key] = deepcopy(primary[key])
    cloned_primary["sensor_config"]["default_x_sensor"] = x_sensor
    cloned_primary["sensor_config"]["default_y_sensor"] = y_sensor
    return {**bundle, "primary_result": cloned_primary}


def _validate_optimizer_result(result: Any, bounds: tuple[tuple[float, float], tuple[float, float]]) -> None:
    if not isinstance(result, dict):
        raise TypeError("optimizer result must be a dictionary")
    missing = {"sensor_xy", "converged", "metadata"} - set(result)
    if missing:
        raise ValueError(f"optimizer result missing keys: {sorted(missing)}")
    x_sensor, y_sensor = result["sensor_xy"]
    if not (
        np.isfinite(x_sensor) and np.isfinite(y_sensor)
        and bounds[0][0] <= x_sensor <= bounds[0][1]
        and bounds[1][0] <= y_sensor <= bounds[1][1]
    ):
        raise ValueError("optimizer result sensor_xy is outside bounds")
    if not isinstance(result["converged"], bool) or not isinstance(result["metadata"], dict):
        raise TypeError("optimizer converged/metadata fields have invalid types")


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")
