"""Generate and validate the exact 2D Bellman reference artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from p1b_4D.stackelberg_solver import solve_attacker_best_response
from p1b_4D.stage_cost import construct_state_grids
from p1b_4D.successor_grid_solver import (
    build_successor_grid_graph,
    solve_successor_grid_bellman,
)

from .reference_configuration import ReferenceConfiguration


REFERENCE_SCHEMA_VERSION = "1.0.0"
VALUE_TOLERANCE = 1.0e-12


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _action_arrays(actions: tuple[dict[str, Any], ...]) -> dict[str, np.ndarray]:
    fields = {
        "action_forward_cells": ("forward_cells", np.int16),
        "action_descent_cells": ("descent_cells", np.int16),
        "action_speed_index": ("speed_index", np.int16),
        "action_speed": ("speed", np.float64),
        "action_gamma": ("gamma", np.float64),
        "action_length": ("length", np.float64),
        "action_duration": ("duration", np.float64),
    }
    return {
        output_name: np.asarray(
            [action[source_name] for action in actions], dtype=dtype
        )
        for output_name, (source_name, dtype) in fields.items()
    }


def validate_exact_policy(
    graph: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Check feasibility and the Bellman identity for every greedy state."""

    value = np.asarray(policy["value"])
    selected = np.asarray(policy["policy_action_index"])
    goal_mask = np.asarray(policy["goal_mask"], dtype=bool)
    state_z, state_h = np.nonzero(selected >= 0)
    action_index = selected[state_z, state_h]

    selected_valid = np.asarray(graph["valid"])[state_z, state_h, action_index]
    selected_terminal = np.asarray(graph["terminal"])[
        state_z, state_h, action_index
    ]
    edge_cost = np.asarray(graph["cost"])[state_z, state_h, action_index]

    forward = np.asarray(
        [action["forward_cells"] for action in graph["actions"]], dtype=int
    )
    descent = np.asarray(
        [action["descent_cells"] for action in graph["actions"]], dtype=int
    )
    next_z = state_z + forward[action_index]
    next_h = state_h - descent[action_index]
    downstream = np.zeros_like(edge_cost)
    nonterminal = ~selected_terminal
    downstream[nonterminal] = value[next_z[nonterminal], next_h[nonterminal]]
    residuals = np.abs(value[state_z, state_h] - (edge_cost + downstream))
    maximum_residual = float(np.max(residuals)) if residuals.size else 0.0

    finite_non_goal = np.isfinite(value) & ~goal_mask
    checks = {
        "goal_values_are_zero": bool(np.all(value[goal_mask] == 0.0)),
        "goal_policy_is_terminal_marker": bool(np.all(selected[goal_mask] == -1)),
        "every_finite_non_goal_state_has_action": bool(
            np.all(selected[finite_non_goal] >= 0)
        ),
        "every_selected_action_is_feasible": bool(np.all(selected_valid)),
        "selected_nonterminal_successors_are_finite": bool(
            np.all(np.isfinite(downstream[nonterminal]))
        ),
        "bellman_identity": maximum_residual <= VALUE_TOLERANCE,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "metrics": {
            "finite_value_state_count": int(np.count_nonzero(np.isfinite(value))),
            "greedy_policy_state_count": int(state_z.size),
            "maximum_bellman_residual": maximum_residual,
        },
        "failed_checks": failed,
    }


def _reference_arrays(
    pipeline: dict[str, Any],
    grids: dict[str, np.ndarray],
    graph: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, np.ndarray]:
    primary = pipeline["primary_result"]
    geometry = primary["geometry_bundle"]["primary_result"]
    bellman = primary["bellman_candidate_bundle"]["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]

    arrays: dict[str, np.ndarray] = {
        "z_grid": np.asarray(grids["z"]),
        "h_grid": np.asarray(grids["h"]),
        "speed_grid": np.asarray(grids["v"]),
        "gamma_grid_stage_cost_only": np.asarray(grids["gamma"]),
        "terrain_z": np.asarray(geometry["terrain_arrays"]["z"]),
        "terrain_height": np.asarray(geometry["terrain_arrays"]["height"]),
        "sensor_position": np.asarray(geometry["sensor_position"]),
        "goal_position": np.asarray(geometry["goal_position"]),
        "los_tangent_point": np.asarray(
            geometry["los_geometry"]["tangent_point"]
        ),
        "los_tangent_line_height": np.asarray(
            geometry["los_geometry"]["tangent_line_height"]
        ),
        "los_boundary": np.asarray(geometry["los_geometry"]["los_boundary"]),
        "spatial_los_valid_mask": np.asarray(
            geometry["los_masks"]["los_mask"], dtype=bool
        ),
        "spatial_occlusion_mask": np.asarray(
            geometry["los_masks"]["occlusion_mask"], dtype=bool
        ),
        "spatial_terrain_mask": np.asarray(
            geometry["los_masks"]["terrain_mask"], dtype=bool
        ),
        "transition_valid": np.asarray(graph["valid"]),
        "transition_terminal": np.asarray(graph["terminal"]),
        "transition_terminal_fraction": np.asarray(graph["terminal_fraction"]),
        "edge_hazard": np.asarray(graph["hazard"]),
        "edge_cost": np.asarray(graph["cost"]),
        "exact_value": np.asarray(policy["value"]),
        "exact_hazard_to_go": np.asarray(policy["hazard_to_go"]),
        "exact_pod_to_go": np.asarray(policy["pod_to_go"]),
        "exact_policy_action_index": np.asarray(policy["policy_action_index"]),
        "goal_mask": np.asarray(policy["goal_mask"]),
        "switching_point_seeds": np.asarray(bellman["switching_point_seeds"]),
        "selected_switching_point": np.asarray(best["switching_point"]),
        "optimal_glide_trajectory": np.asarray(best["trajectory"]),
        "optimal_speed_profile": np.asarray(best["speed_profile"]),
        "optimal_gamma_profile": np.asarray(best["gamma_profile"]),
        "optimal_duration_profile": np.asarray(best["duration_profile"]),
        "powered_path": np.asarray(best["powered_path"]),
        "continuous_replay_trajectory": np.asarray(replay["trajectory"]),
        **_action_arrays(graph["actions"]),
    }
    return arrays


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(_json_native(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def generate_bellman_reference(
    reference: ReferenceConfiguration, output_directory: Path
) -> dict[str, Any]:
    """Run the authoritative case, independently export its exact MDP, and validate it."""

    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pipeline = solve_attacker_best_response(
        reference.sensor_z,
        reference.configuration_bundle,
        evaluation_id=f"2d-rl-reference-{reference.scenario_id}",
    )
    if not pipeline["status"]["success"]:
        raise RuntimeError(pipeline["status"]["message"])

    primary = pipeline["primary_result"]
    nested_configuration = primary["configuration_bundle"]
    geometry_bundle = primary["geometry_bundle"]
    configs = nested_configuration["primary_result"]
    geometry = geometry_bundle["primary_result"]
    grids = construct_state_grids(
        configs["environment_config"], configs["vehicle_config"]
    )
    graph = build_successor_grid_graph(
        nested_configuration, geometry_bundle, grids
    )
    policy = solve_successor_grid_bellman(
        graph,
        grids,
        geometry["goal_position"],
        goal_radius=float(configs["validation_config"]["goal_radius"]),
    )

    embedded_value = np.asarray(
        primary["bellman_candidate_bundle"]["primary_result"][
            "cost_to_go_maps"
        ]["successor_grid"]
    )
    exact_value = np.asarray(policy["value"])
    finite_match = np.array_equal(np.isfinite(embedded_value), np.isfinite(exact_value))
    common_finite = np.isfinite(embedded_value) & np.isfinite(exact_value)
    value_difference = (
        float(np.max(np.abs(embedded_value[common_finite] - exact_value[common_finite])))
        if np.any(common_finite)
        else 0.0
    )
    policy_validation = validate_exact_policy(graph, policy)
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    checks = {
        "authoritative_pipeline_success": bool(pipeline["status"]["success"]),
        "authoritative_response_validation": bool(pipeline["validation"]["passed"]),
        "production_and_export_finite_masks_match": bool(finite_match),
        "production_and_export_values_match": value_difference <= VALUE_TOLERANCE,
        "exact_policy_validation": bool(policy_validation["passed"]),
        "continuous_replay_feasible": bool(replay["feasible"]),
        "continuous_replay_reaches_goal": bool(replay["reached_goal"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Bellman reference validation failed: {failed}")

    arrays = _reference_arrays(pipeline, grids, graph, policy)
    artifact_path = output_directory / "bellman_reference.npz"
    _write_npz(artifact_path, arrays)
    elapsed = time.perf_counter() - started

    summary = {
        "schema_name": "Exact2DBellmanRLReference",
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "scenario_id": reference.scenario_id,
        "sensor_z": reference.sensor_z,
        "configuration_sha256": reference.configuration_hash,
        "configuration_snapshot": reference.configuration_snapshot,
        "transition_contract": {
            "state": ["z", "h"],
            "action": (
                "discrete physical-successor index encoding forward_cells, "
                "descent_cells, speed, derived gamma, and duration"
            ),
            "reward": "negative edge_cost",
            "terminal": "goal-intersecting physical edge",
            "persistent_velocity_or_gamma_state": False,
        },
        "grid": {
            "z_count": int(np.asarray(grids["z"]).size),
            "h_count": int(np.asarray(grids["h"]).size),
            "stage_cost_speed_count": int(np.asarray(grids["v"]).size),
            "stage_cost_gamma_count": int(np.asarray(grids["gamma"]).size),
            "bellman_state_shape": list(exact_value.shape),
            "action_count": len(graph["actions"]),
        },
        "mission": {
            "selected_switching_point": best["switching_point"],
            "mission_objective": best["mission_objective"],
            "mission_pod": best["mission_pod"],
            "mission_time": best["mission_time"],
            "powered_time": best["powered_time"],
            "glide_time": best["glide_time"],
            "candidate_count_searched": best["candidate_count_searched"],
            "trajectory_node_count": int(np.asarray(best["trajectory"]).shape[0]),
        },
        "continuous_replay": {
            "feasible": replay["feasible"],
            "violation": replay["violation"],
            "reached_goal": replay["reached_goal"],
            "goal_miss": replay["goal_miss"],
            "continuous_mission_time": replay["continuous_mission_time"],
            "continuous_glide_hazard": replay["continuous_glide_hazard"],
            "step_count_used": replay["step_count_used"],
        },
        "validation": {
            "passed": True,
            "checks": checks,
            "metrics": {
                "production_export_value_maximum_difference": value_difference,
                **policy_validation["metrics"],
            },
            "failed_checks": [],
        },
        "artifact": {
            "file": artifact_path.name,
            "sha256": _sha256_file(artifact_path),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        },
        "runtime_seconds": elapsed,
    }
    summary_path = output_directory / "bellman_reference_summary.json"
    _write_json(summary_path, summary)
    return {
        "summary": summary,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
    }
