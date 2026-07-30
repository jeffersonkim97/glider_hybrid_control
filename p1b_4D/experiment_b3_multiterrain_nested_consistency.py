"""Direction-B B3 nested-consistency extension to two additional terrains.

The final P2 coverage-only and Stackelberg sensor candidates are held fixed.
For each candidate this driver repeats the frozen B2 nine-case follower
matrix and reevaluates every feasible selected policy with one common
high-fidelity evaluator.  It does not repeat the outer defender search.
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
from p1b_4D.direction_b_discretization import build_direction_b_configuration
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    COMMON_EVALUATOR_SAMPLE_COUNTS,
    _defender_objective,
    _jsonable,
    _run_case,
    build_b2_case_matrix,
)
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger


B3_TERRAIN_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "single_hill": {
        "z_max": 5500.0,
        "h_max": 400.0,
        "z_goal": 5000.0,
        "hills": (
            {"z_ridge": 2500.0, "h_ridge": 200.0, "width": 400.0},
        ),
        "sensor_bounds": (2750.0, 4500.0),
        "sensor_candidates": {
            "coverage": 4499.8666361835085,
            "stackelberg": 4499.955545394503,
        },
    },
    "goal_in_valley": {
        "z_max": 4000.0,
        "h_max": 200.0,
        "z_goal": 2500.0,
        "hills": (
            {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 3500.0, "h_ridge": 100.0, "width": 100.0},
        ),
        "sensor_bounds": (1700.0, 2200.0),
        "sensor_candidates": {
            "coverage": 2199.6570644718795,
            "stackelberg": 2197.828074988569,
        },
    },
}
B2_COMMON_EVALUATOR_SAMPLE_COUNT = 1025


def build_b3_physical_configuration(
    base: dict[str, Any], terrain_name: str, sensor_z: float
) -> dict[str, Any]:
    """Apply one frozen B3 terrain and sensor candidate to a Phase-1 bundle."""
    if terrain_name not in B3_TERRAIN_SPECIFICATIONS:
        raise ValueError(f"Unsupported B3 terrain: {terrain_name}")
    specification = B3_TERRAIN_SPECIFICATIONS[terrain_name]
    bundle = deepcopy(base)
    configs = bundle["primary_result"]
    environment = configs["environment_config"]
    environment["z_start"] = 0.0
    environment["h_start"] = 0.0
    environment["z_goal"] = float(specification["z_goal"])
    environment["terrain"] = {
        "z_min": 0.0,
        "z_max": float(specification["z_max"]),
        "hills": tuple(specification["hills"]),
    }
    environment["grid"].update({
        "z_min": 0.0,
        "z_max": float(specification["z_max"]),
        "h_min": 0.0,
        "h_max": float(specification["h_max"]),
    })
    environment["airspace"] = {
        "z_min": 0.0,
        "z_max": float(specification["z_max"]),
        "h_min": 0.0,
        "h_max": float(specification["h_max"]),
    }
    environment["simulation"] = {
        **environment["airspace"],
        "max_path_steps": 2000,
    }
    configs["sensor_config"]["default_z_sensor"] = float(sensor_z)
    lower, upper = specification["sensor_bounds"]
    configs["defender_config"]["continuous_search_bounds"] = {
        "z_sensor_min": float(lower),
        "z_sensor_max": float(upper),
    }
    return bundle


def run_b3_multiterrain(
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Execute the complete B3 matrix and write one machine-readable result."""
    base = build_configuration_bundle(project_root)
    started = perf_counter()
    cases: list[dict[str, Any]] = []
    common_coverage: dict[str, dict[str, float]] = {}
    try:
        for terrain_name, terrain_specification in (
            B3_TERRAIN_SPECIFICATIONS.items()
        ):
            common_coverage[terrain_name] = {}
            for sensor_name, sensor_z in terrain_specification[
                "sensor_candidates"
            ].items():
                physical = build_b3_physical_configuration(
                    base, terrain_name, sensor_z
                )
                coverage_config = build_direction_b_configuration(
                    physical, terrain_name, 2
                )
                coverage_geometry = build_geometry_bundle(coverage_config)
                coverage = float(
                    coverage_geometry["primary_result"]["coverage"][
                        "normalized_coverage_area"
                    ]
                )
                common_coverage[terrain_name][sensor_name] = coverage
                del coverage_geometry, coverage_config
                gc.collect()

                for case_specification in build_b2_case_matrix():
                    print(
                        "B3",
                        terrain_name,
                        sensor_name,
                        case_specification["case_id"],
                        "started",
                        flush=True,
                    )
                    case = _run_case(
                        physical,
                        sensor_name,
                        sensor_z,
                        case_specification,
                        coverage,
                        terrain_name=terrain_name,
                    )
                    case["terrain_name"] = terrain_name
                    cases.append(case)
                    print(
                        "B3",
                        terrain_name,
                        sensor_name,
                        case_specification["case_id"],
                        case["status"],
                        f"seconds={case['elapsed_seconds']:.3f}",
                        flush=True,
                    )
                    gc.collect()
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )

    final_sample_count, evaluator_gate = _select_b3_global_evaluator(cases)
    defender_config = base["primary_result"]["cost_config"]["defender"]
    for case in cases:
        if case["status"] != "feasible":
            continue
        evaluation = case["high_fidelity"]["evaluations"][
            str(final_sample_count)
        ]
        case["selected_high_fidelity"] = evaluation
        case["selected_defender_objective"] = _defender_objective(
            evaluation["mission_hazard"],
            common_coverage[case["terrain_name"]][case["sensor_name"]],
            defender_config,
        )

    analyses = {
        terrain_name: _analyze_terrain(
            terrain_name, cases, final_sample_count
        )
        for terrain_name in B3_TERRAIN_SPECIFICATIONS
    }
    result = {
        "schema_name": "DirectionBMultiTerrainNestedConsistency",
        "schema_version": "1.0.0",
        "status": (
            "complete" if evaluator_gate["passed"]
            else "evaluator_gate_failed"
        ),
        "terrains": tuple(B3_TERRAIN_SPECIFICATIONS),
        "fixed_sensor_candidates": {
            terrain_name: specification["sensor_candidates"]
            for terrain_name, specification in (
                B3_TERRAIN_SPECIFICATIONS.items()
            )
        },
        "common_coverage_l2_reference": common_coverage,
        "global_common_evaluator_sample_count": final_sample_count,
        "common_evaluator_gate": evaluator_gate,
        "case_count": len(cases),
        "feasible_case_count": sum(
            case["status"] == "feasible" for case in cases
        ),
        "elapsed_seconds": perf_counter() - started,
        "cases": cases,
        "analysis": analyses,
        "scope": {
            "outer_defender_optimization_repeated": False,
            "continuous_optimum_claimed": False,
            "physical_geometry_fixed_across_levels": True,
            "b2_nine_case_matrix_reused_without_changes": True,
            "minimum_common_evaluator_inherited_from_b2": (
                B2_COMMON_EVALUATOR_SAMPLE_COUNT
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _select_b3_global_evaluator(cases):
    feasible = [case for case in cases if case["status"] == "feasible"]
    if not feasible:
        return COMMON_EVALUATOR_SAMPLE_COUNTS[-2], {
            "passed": False,
            "rule": "no feasible B3 policy was available for qualification",
            "feasible_policy_count": 0,
        }
    for lower_count, upper_count in zip(
        COMMON_EVALUATOR_SAMPLE_COUNTS[:-1],
        COMMON_EVALUATOR_SAMPLE_COUNTS[1:],
    ):
        if lower_count < B2_COMMON_EVALUATOR_SAMPLE_COUNT:
            continue
        key = f"{lower_count}_vs_{upper_count}"
        if all(
            case["high_fidelity"]["qualifications"][key]["passed"]
            for case in feasible
        ):
            return lower_count, {
                "passed": True,
                "rule": (
                    f"all feasible B3 policies passed {lower_count}_vs_"
                    f"{upper_count}"
                ),
                "feasible_policy_count": len(feasible),
                "qualification_pair": (lower_count, upper_count),
            }
    final_lower = COMMON_EVALUATOR_SAMPLE_COUNTS[-2]
    final_upper = COMMON_EVALUATOR_SAMPLE_COUNTS[-1]
    key = f"{final_lower}_vs_{final_upper}"
    failed_cases = [
        f"{case['terrain_name']}:{case['sensor_name']}:{case['case_id']}"
        for case in feasible
        if not case["high_fidelity"]["qualifications"][key]["passed"]
    ]
    return final_lower, {
        "passed": False,
        "rule": f"at least one B3 policy failed {key}",
        "feasible_policy_count": len(feasible),
        "qualification_pair": (final_lower, final_upper),
        "failed_cases": failed_cases,
    }


def reanalyze_existing_b3_result(
    project_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute selected-evaluator fields without repeating follower solves."""
    result = json.loads(output_path.read_text(encoding="utf-8"))
    cases = result["cases"]
    final_sample_count, evaluator_gate = _select_b3_global_evaluator(cases)
    base = build_configuration_bundle(project_root)
    try:
        defender_config = base["primary_result"]["cost_config"]["defender"]
        coverage_reference = result["common_coverage_l2_reference"]
        for case in cases:
            if case["status"] != "feasible":
                continue
            evaluation = case["high_fidelity"]["evaluations"][
                str(final_sample_count)
            ]
            case["selected_high_fidelity"] = evaluation
            case["selected_defender_objective"] = _defender_objective(
                evaluation["mission_hazard"],
                coverage_reference[case["terrain_name"]][
                    case["sensor_name"]
                ],
                defender_config,
            )
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )
    result["global_common_evaluator_sample_count"] = final_sample_count
    result["common_evaluator_gate"] = evaluator_gate
    result["status"] = (
        "complete" if evaluator_gate["passed"] else "evaluator_gate_failed"
    )
    result["analysis"] = {
        terrain_name: _analyze_terrain(
            terrain_name, cases, final_sample_count
        )
        for terrain_name in B3_TERRAIN_SPECIFICATIONS
    }
    result["scope"]["minimum_common_evaluator_inherited_from_b2"] = (
        B2_COMMON_EVALUATOR_SAMPLE_COUNT
    )
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _analyze_terrain(terrain_name, all_cases, sample_count):
    terrain_cases = [
        case for case in all_cases if case["terrain_name"] == terrain_name
    ]
    lookup = {
        (case["sensor_name"], case["case_id"]): case
        for case in terrain_cases
    }
    sensor_names = tuple(
        B3_TERRAIN_SPECIFICATIONS[terrain_name]["sensor_candidates"]
    )
    resolution = {}
    path_comparisons = {}
    for sensor_name in sensor_names:
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
            sensor_paths[key] = _terrain_path_metrics(
                left, right, terrain_name
            )
        resolution[sensor_name] = sensor_errors
        path_comparisons[sensor_name] = sensor_paths

    margins = {}
    for level in range(3):
        coverage = lookup[("coverage", f"enriched_v5_q9_l{level}")]
        stack = lookup[("stackelberg", f"enriched_v5_q9_l{level}")]
        if coverage["status"] != "feasible" or stack["status"] != "feasible":
            margins[f"l{level}"] = None
        else:
            margins[f"l{level}"] = (
                stack["selected_defender_objective"]
                - coverage["selected_defender_objective"]
            )
    if all(value is not None for value in margins.values()):
        resolution_shift = max(
            abs(
                lookup[(sensor, "enriched_v5_q9_l2")][
                    "selected_defender_objective"
                ]
                - lookup[(sensor, "enriched_v5_q9_l1")][
                    "selected_defender_objective"
                ]
            )
            for sensor in sensor_names
        )
        nonzero_signs = {
            math.copysign(1.0, value)
            for value in margins.values()
            if value != 0.0
        }
        sign_stable = (
            len(nonzero_signs) == 1
            and all(value != 0.0 for value in margins.values())
        )
        ranking_resolved = (
            sign_stable
            and abs(margins["l2"]) > 2.0 * resolution_shift
        )
    else:
        resolution_shift = None
        sign_stable = False
        ranking_resolved = False

    speed_sensitivity = {}
    quadrature_sensitivity = {}
    for sensor_name in sensor_names:
        speed_sensitivity[sensor_name] = {}
        for level in (1, 2):
            v5 = lookup[(sensor_name, f"enriched_v5_q9_l{level}")]
            v9 = lookup[(sensor_name, f"enriched_v9_q9_l{level}")]
            speed_sensitivity[sensor_name][f"l{level}"] = (
                _objective_difference(v5, v9)
            )
        q9 = lookup[(sensor_name, "enriched_v5_q9_l2")]
        q17 = lookup[(sensor_name, "enriched_v5_q17_l2")]
        quadrature_sensitivity[sensor_name] = _objective_difference(q9, q17)

    finite_speed = [
        value
        for values in speed_sensitivity.values()
        for value in values.values()
        if value is not None
    ]
    finite_quadrature = [
        value for value in quadrature_sensitivity.values()
        if value is not None
    ]
    tau_b = (
        max(1e-6, 0.1 * resolution_shift)
        if resolution_shift is not None else None
    )
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
        "production_speed_family_if_terrain_only": (
            "V5" if finite_speed and max(finite_speed) <= tau_b else "V9"
        ) if tau_b is not None else None,
        "production_planning_quadrature_count_if_terrain_only": (
            9 if finite_quadrature and max(finite_quadrature) <= tau_b else 17
        ) if tau_b is not None else None,
        "feasible_case_count": sum(
            case["status"] == "feasible" for case in terrain_cases
        ),
        "case_count": len(terrain_cases),
    }


def _objective_difference(left, right):
    if left["status"] != "feasible" or right["status"] != "feasible":
        return None
    return abs(
        left["selected_high_fidelity"]["attacker_objective"]
        - right["selected_high_fidelity"]["attacker_objective"]
    )


def _terrain_path_metrics(left, right, terrain_name):
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
    ridges = tuple(
        hill["z_ridge"]
        for hill in B3_TERRAIN_SPECIFICATIONS[terrain_name]["hills"]
    )
    left_region = int(np.searchsorted(ridges, left_switch[0]))
    right_region = int(np.searchsorted(ridges, right_switch[0]))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/direction_b/b3_multiterrain_nested_consistency.json"
        ),
    )
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="reuse stored policies/evaluations and recompute summary fields",
    )
    arguments = parser.parse_args()
    if arguments.reanalyze_existing:
        result = reanalyze_existing_b3_result(
            Path.cwd(), arguments.output
        )
    else:
        result = run_b3_multiterrain(Path.cwd(), arguments.output)
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
