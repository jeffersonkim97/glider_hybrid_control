"""Direction-B B2 two-hill nested-discretization experiment.

The two defender candidates are fixed.  This module never reruns the outer
defender optimization; it solves and independently reevaluates only the
frozen follower matrix.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.detection import build_symbolic_detection_bundle
from p1b_4D.direction_b_discretization import build_direction_b_configuration
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.high_fidelity_policy_evaluation import qualify_common_evaluator
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.successor_grid_solver import solve_successor_grid_attacker


B2_SENSOR_CANDIDATES = {
    "coverage": 1966.4609053497943,
    "stackelberg": 1982.9218106995881,
}

COMMON_EVALUATOR_SAMPLE_COUNTS = (129, 257, 513, 1025, 2049, 4097)


def build_b2_case_matrix() -> tuple[dict[str, Any], ...]:
    """Return the frozen nine-case matrix for one sensor candidate."""
    cases: list[dict[str, Any]] = []
    for level in range(3):
        cases.append(_case("enriched", "V5", 9, level, "main"))
    for level in range(3):
        cases.append(_case("transported", "V5", 9, level, "ablation"))
    for level in (1, 2):
        cases.append(_case("enriched", "V9", 9, level, "speed_sensitivity"))
    cases.append(_case("enriched", "V5", 17, 2, "quadrature_sensitivity"))
    return tuple(cases)


def run_b2_two_hill(
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Execute the complete fixed-sensor B2 matrix and write one JSON result."""
    base = build_configuration_bundle(project_root)
    started = perf_counter()
    cases: list[dict[str, Any]] = []
    common_coverage: dict[str, float] = {}
    try:
        for sensor_name, sensor_z in B2_SENSOR_CANDIDATES.items():
            physical = build_two_hill_configuration(base, sensor_z)
            coverage_config = build_direction_b_configuration(
                physical, "two_hill", 2
            )
            coverage_geometry = build_geometry_bundle(coverage_config)
            common_coverage[sensor_name] = float(
                coverage_geometry["primary_result"]["coverage"][
                    "normalized_coverage_area"
                ]
            )
            del coverage_geometry, coverage_config
            gc.collect()
            for specification in build_b2_case_matrix():
                print(
                    "B2",
                    sensor_name,
                    specification["case_id"],
                    "started",
                    flush=True,
                )
                case = _run_case(
                    physical,
                    sensor_name,
                    sensor_z,
                    specification,
                    common_coverage[sensor_name],
                )
                cases.append(case)
                print(
                    "B2",
                    sensor_name,
                    specification["case_id"],
                    case["status"],
                    f"seconds={case['elapsed_seconds']:.3f}",
                    flush=True,
                )
                gc.collect()
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )

    final_sample_count, evaluator_gate = _select_global_evaluator(cases)
    for case in cases:
        if case["status"] != "feasible":
            continue
        evaluation = case["high_fidelity"]["evaluations"][
            str(final_sample_count)
        ]
        case["selected_high_fidelity"] = evaluation
        case["selected_defender_objective"] = _defender_objective(
            evaluation["mission_hazard"],
            common_coverage[case["sensor_name"]],
            base["primary_result"]["cost_config"]["defender"],
        )

    analysis = _analyze_b2_cases(cases, final_sample_count)
    result = {
        "schema_name": "DirectionBTwoHillNestedConsistency",
        "schema_version": "1.0.0",
        "status": (
            "complete" if evaluator_gate["passed"] else "evaluator_gate_failed"
        ),
        "fixed_sensor_candidates": B2_SENSOR_CANDIDATES,
        "common_coverage_l2_reference": common_coverage,
        "global_common_evaluator_sample_count": final_sample_count,
        "common_evaluator_gate": evaluator_gate,
        "case_count": len(cases),
        "feasible_case_count": sum(
            case["status"] == "feasible" for case in cases
        ),
        "elapsed_seconds": perf_counter() - started,
        "cases": cases,
        "analysis": analysis,
        "scope": {
            "outer_defender_optimization_repeated": False,
            "continuous_optimum_claimed": False,
            "physical_geometry_fixed_across_levels": True,
            "shallow_backbone_vector_two_hill_m": (137.5, -4.0),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def build_two_hill_configuration(
    base: dict[str, Any], sensor_z: float
) -> dict[str, Any]:
    """Apply the fixed P2 two-hill physical problem to a Phase-1 bundle."""
    bundle = deepcopy(base)
    configs = bundle["primary_result"]
    environment = configs["environment_config"]
    environment["z_start"] = 0.0
    environment["h_start"] = 0.0
    environment["z_goal"] = 2500.0
    environment["terrain"] = {
        "z_min": 0.0,
        "z_max": 2750.0,
        "hills": (
            {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 2000.0, "h_ridge": 50.0, "width": 100.0},
        ),
    }
    environment["grid"].update({
        "z_min": 0.0,
        "z_max": 2750.0,
        "h_min": 0.0,
        "h_max": 200.0,
    })
    environment["airspace"] = {
        "z_min": 0.0, "z_max": 2750.0,
        "h_min": 0.0, "h_max": 200.0,
    }
    environment["simulation"] = {
        **environment["airspace"], "max_path_steps": 1000,
    }
    configs["sensor_config"]["default_z_sensor"] = float(sensor_z)
    configs["defender_config"]["continuous_search_bounds"] = {
        "z_sensor_min": 1300.0,
        "z_sensor_max": 2200.0,
    }
    return bundle


def _case(action_family, speed_family, quadrature, level, role):
    return {
        "case_id": (
            f"{action_family}_{speed_family.lower()}_q{quadrature}_l{level}"
        ),
        "action_family": action_family,
        "speed_family": speed_family,
        "planning_quadrature_count": quadrature,
        "level": level,
        "role": role,
    }


def _run_case(
    physical,
    sensor_name,
    sensor_z,
    specification,
    coverage,
    *,
    terrain_name="two_hill",
):
    started = perf_counter()
    bundle = build_direction_b_configuration(
        physical,
        terrain_name,
        specification["level"],
        action_family=specification["action_family"],
        speed_family=specification["speed_family"],
        edge_quadrature_count=specification["planning_quadrature_count"],
    )
    geometry = build_geometry_bundle(bundle)
    detection = build_symbolic_detection_bundle(bundle, geometry)
    result = {
        **specification,
        "sensor_name": sensor_name,
        "sensor_z": float(sensor_z),
        "status": "infeasible",
        "diagnostic": None,
        "elapsed_seconds": 0.0,
    }
    try:
        candidates, response = solve_successor_grid_attacker(
            bundle, geometry, detection
        )
    except RuntimeError as error:
        result["diagnostic"] = str(error)
        result["elapsed_seconds"] = perf_counter() - started
        return result

    policy = response["primary_result"]
    evaluations: dict[str, Any] = {}
    qualifications: dict[str, Any] = {}
    for lower_count, upper_count in zip(
        COMMON_EVALUATOR_SAMPLE_COUNTS[:-1],
        COMMON_EVALUATOR_SAMPLE_COUNTS[1:],
    ):
        qualification = qualify_common_evaluator(
            policy, bundle, geometry,
            candidate_sample_count=lower_count,
            reference_sample_count=upper_count,
        )
        evaluations[str(lower_count)] = qualification[
            "candidate_evaluation"
        ]
        evaluations[str(upper_count)] = qualification[
            "reference_evaluation"
        ]
        qualifications[f"{lower_count}_vs_{upper_count}"] = (
            _qualification_summary(qualification)
        )
    result.update({
        "status": "feasible",
        "diagnostic": "selected_policy_generated",
        "elapsed_seconds": perf_counter() - started,
        "planning": {
            "mission_cost": float(policy["mission_cost"]),
            "mission_pod": float(policy["mission_pod"]),
            "mission_time": float(policy["mission_time"]),
            "mission_hazard": float(
                policy["hazard_breakdown"]["mission_hazard"]
            ),
            "switching_point": np.asarray(policy["switching_point"]),
            "trajectory": np.asarray(policy["trajectory"]),
            "speed_profile": np.asarray(policy["speed_profile"]),
            "gamma_profile": np.asarray(policy["gamma_profile"]),
            "duration_profile": np.asarray(policy["duration_profile"]),
            "path_node_count": int(len(policy["trajectory"])),
            "candidate_count": int(
                response["primary_result"]["candidate_count_searched"]
            ),
            "tie_count": int(response["primary_result"]["tie_count"]),
            "maximum_endpoint_residual": float(
                policy["constraint_residuals"][
                    "maximum_edge_endpoint_residual"
                ]
            ),
        },
        "common_coverage_l2_reference": float(coverage),
        "high_fidelity": {
            "qualifications": qualifications,
            "evaluations": evaluations,
        },
        "graph_metadata": candidates["metadata"]["graph_metadata"],
    })
    del candidates, response, detection, geometry, bundle
    return result


def _qualification_summary(result):
    return {
        "passed": bool(result["passed"]),
        "checks": result["checks"],
        "objective_absolute_difference": float(
            result["objective_absolute_difference"]
        ),
        "mission_pod_absolute_difference": float(
            result["mission_pod_absolute_difference"]
        ),
    }


def _select_global_evaluator(cases):
    feasible = [case for case in cases if case["status"] == "feasible"]
    for lower_count, upper_count in zip(
        COMMON_EVALUATOR_SAMPLE_COUNTS[:-1],
        COMMON_EVALUATOR_SAMPLE_COUNTS[1:],
    ):
        key = f"{lower_count}_vs_{upper_count}"
        if all(
            case["high_fidelity"]["qualifications"][key]["passed"]
            for case in feasible
        ):
            return lower_count, {
                "passed": True,
                "rule": (
                    f"all feasible policies passed {lower_count}_vs_"
                    f"{upper_count}"
                ),
                "feasible_policy_count": len(feasible),
                "qualification_pair": (lower_count, upper_count),
            }
    final_lower = COMMON_EVALUATOR_SAMPLE_COUNTS[-2]
    final_upper = COMMON_EVALUATOR_SAMPLE_COUNTS[-1]
    failed_cases = [
        f"{case['sensor_name']}:{case['case_id']}"
        for case in feasible
        if not case["high_fidelity"]["qualifications"][
            f"{final_lower}_vs_{final_upper}"
        ]["passed"]
    ]
    return final_lower, {
        "passed": False,
        "rule": (
            f"at least one policy failed {final_lower}_vs_{final_upper}"
        ),
        "feasible_policy_count": len(feasible),
        "qualification_pair": (final_lower, final_upper),
        "failed_cases": failed_cases,
    }


def _analyze_b2_cases(cases, sample_count):
    lookup = {
        (case["sensor_name"], case["case_id"]): case for case in cases
    }
    resolution = {}
    path_comparisons = {}
    for sensor_name in B2_SENSOR_CANDIDATES:
        sensor_errors = {}
        sensor_paths = {}
        for lower, upper in ((0, 1), (1, 2)):
            left = lookup[(sensor_name, f"enriched_v5_q9_l{lower}")]
            right = lookup[(sensor_name, f"enriched_v5_q9_l{upper}")]
            key = f"l{lower}_to_l{upper}"
            if left["status"] != "feasible" or right["status"] != "feasible":
                sensor_errors[key] = None
                sensor_paths[key] = None
                continue
            sensor_errors[key] = abs(
                left["selected_high_fidelity"]["attacker_objective"]
                - right["selected_high_fidelity"]["attacker_objective"]
            )
            sensor_paths[key] = _path_metrics(left, right)
        resolution[sensor_name] = sensor_errors
        path_comparisons[sensor_name] = sensor_paths

    margins = {}
    for level in range(3):
        coverage = lookup[("coverage", f"enriched_v5_q9_l{level}")]
        stack = lookup[("stackelberg", f"enriched_v5_q9_l{level}")]
        margins[f"l{level}"] = (
            stack["selected_defender_objective"]
            - coverage["selected_defender_objective"]
        )
    resolution_shift = max(
        abs(
            lookup[(sensor, "enriched_v5_q9_l2")][
                "selected_defender_objective"
            ]
            - lookup[(sensor, "enriched_v5_q9_l1")][
                "selected_defender_objective"
            ]
        )
        for sensor in B2_SENSOR_CANDIDATES
    )
    nonzero_signs = [math.copysign(1.0, value) for value in margins.values()]
    sign_stable = len(set(nonzero_signs)) == 1
    ranking_resolved = (
        sign_stable and abs(margins["l2"]) > 2.0 * resolution_shift
    )

    speed_sensitivity = {}
    quadrature_sensitivity = {}
    for sensor in B2_SENSOR_CANDIDATES:
        speed_sensitivity[sensor] = {}
        for level in (1, 2):
            v5 = lookup[(sensor, f"enriched_v5_q9_l{level}")]
            v9 = lookup[(sensor, f"enriched_v9_q9_l{level}")]
            speed_sensitivity[sensor][f"l{level}"] = abs(
                v9["selected_high_fidelity"]["attacker_objective"]
                - v5["selected_high_fidelity"]["attacker_objective"]
            )
        q9 = lookup[(sensor, "enriched_v5_q9_l2")]
        q17 = lookup[(sensor, "enriched_v5_q17_l2")]
        quadrature_sensitivity[sensor] = abs(
            q17["selected_high_fidelity"]["attacker_objective"]
            - q9["selected_high_fidelity"]["attacker_objective"]
        )
    tau_b = max(1e-6, 0.1 * resolution_shift)
    maximum_speed_sensitivity = max(
        value
        for by_level in speed_sensitivity.values()
        for value in by_level.values()
    )
    maximum_quadrature_sensitivity = max(quadrature_sensitivity.values())
    return {
        "common_evaluator_sample_count": sample_count,
        "attacker_objective_resolution_shift": resolution,
        "path_comparisons": path_comparisons,
        "defender_margin_by_level": margins,
        "defender_resolution_shift_l1_to_l2": resolution_shift,
        "ranking_sign_stable": sign_stable,
        "ranking_diagnostically_resolved": ranking_resolved,
        "speed_sensitivity": speed_sensitivity,
        "quadrature_sensitivity": quadrature_sensitivity,
        "tau_b": tau_b,
        "production_speed_family": (
            "V5" if maximum_speed_sensitivity <= tau_b else "V9"
        ),
        "production_planning_quadrature_count": (
            9 if maximum_quadrature_sensitivity <= tau_b else 17
        ),
    }


def _path_metrics(left, right):
    left_path = np.asarray(left["planning"]["trajectory"], dtype=float)
    right_path = np.asarray(right["planning"]["trajectory"], dtype=float)
    common_z = np.linspace(
        max(left_path[0, 0], right_path[0, 0]),
        min(left_path[-1, 0], right_path[-1, 0]),
        512,
    )
    left_h = np.interp(common_z, left_path[:, 0], left_path[:, 1])
    right_h = np.interp(common_z, right_path[:, 0], right_path[:, 1])
    difference = right_h - left_h
    left_resampled = np.column_stack((common_z, left_h))
    right_resampled = np.column_stack((common_z, right_h))
    hausdorff = max(
        directed_hausdorff(left_resampled, right_resampled)[0],
        directed_hausdorff(right_resampled, left_resampled)[0],
    )
    left_switch = np.asarray(left["planning"]["switching_point"])
    right_switch = np.asarray(right["planning"]["switching_point"])
    left_region = int(np.searchsorted((1000.0, 2000.0), left_switch[0]))
    right_region = int(np.searchsorted((1000.0, 2000.0), right_switch[0]))
    return {
        "switching_displacement": float(
            np.linalg.norm(right_switch - left_switch)
        ),
        "common_z_altitude_rmse": float(np.sqrt(np.mean(difference**2))),
        "common_z_maximum_altitude_difference": float(
            np.max(np.abs(difference))
        ),
        "symmetric_hausdorff_distance": float(hausdorff),
        "left_path_node_count": int(left_path.shape[0]),
        "right_path_node_count": int(right_path.shape[0]),
        "topology_signature_left": f"switch_region_{left_region}",
        "topology_signature_right": f"switch_region_{right_region}",
        "topology_change": left_region != right_region,
    }


def _defender_objective(mission_hazard, coverage, defender):
    pod_spec = defender["normalization"]["pod"]
    if pod_spec["method"] == "hazard_reference":
        pod_value = mission_hazard / (
            mission_hazard + float(pod_spec["hazard_reference"])
        )
    elif pod_spec["method"] == "probability":
        pod_value = 1.0 - math.exp(-mission_hazard)
    else:
        raise ValueError(f"Unsupported defender PoD normalization: {pod_spec}")
    return float(
        defender["w_pod"] * pod_value + defender["w_cover"] * coverage
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/direction_b/b2_two_hill_nested_consistency.json"),
    )
    arguments = parser.parse_args()
    result = run_b2_two_hill(Path.cwd(), arguments.output)
    print(json.dumps(_jsonable({
        "status": result["status"],
        "case_count": result["case_count"],
        "feasible_case_count": result["feasible_case_count"],
        "common_evaluator_sample_count": result[
            "global_common_evaluator_sample_count"
        ],
        "analysis": result["analysis"],
        "output": str(arguments.output.resolve()),
    }), indent=2))


if __name__ == "__main__":
    main()
