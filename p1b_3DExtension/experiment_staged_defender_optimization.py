"""Budgeted coarse-to-fine 2-D Defender sensor search for the extreme ridge.

Stage A screens a bounded sensor grid against a south/north continuous
trajectory library.  Stage B recomputes three spatially diverse finalists
with the exact physical-successor medium Bellman solver and continuous 3-DOF
attacker refinement.  Stage C runs one full fine Bellman confirmation at the
best continuously re-ranked sensor position.

This is a transparent budgeted search, not a global-optimality proof.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_trajectory_refinement import (
    REPO_ROOT,
    _dense_validate,
    solve_continuous_refinement,
)
from .detection import build_symbolic_detection_bundle
from .experiment_extreme_ridge_fine import build_fine_configuration
from .geometry import build_geometry_bundle
from .phase_logging import close_phase_logger
from .successor_grid_solver import solve_physical_successor_grid_attacker


OUTPUT_DIR = REPO_ROOT / "results" / "staged_defender_optimization"
REFERENCE_CONTINUOUS_DIR = REPO_ROOT / "results" / "extreme_ridge_275_continuous"
SCREEN_RESOLUTION = {
    "x_count": 31, "y_count": 21, "h_count": 31,
    "v_count": 3, "gamma_count": 6, "heading_count": 36,
}
MEDIUM_RESOLUTION = {
    "x_count": 41, "y_count": 31, "h_count": 41,
    "v_count": 3, "gamma_count": 6, "heading_count": 36,
}
FINE_RESOLUTION = {
    "x_count": 51, "y_count": 41, "h_count": 51,
    "v_count": 3, "gamma_count": 6, "heading_count": 36,
}
SCREEN_X = tuple(float(value) for value in np.linspace(1000.0, 2600.0, 9))
SCREEN_Y = tuple(float(value) for value in np.linspace(-600.0, 600.0, 7))
SPECIAL_SCREEN_POINTS = ((1980.0, 225.0), (1500.0, 0.0))
MEDIUM_FINALIST_COUNT = 3
MINIMUM_FINALIST_SEPARATION_M = 300.0


def _configuration(
    resolution: dict[str, int], sensor_xy: tuple[float, float],
) -> dict[str, Any]:
    bundle = deepcopy(build_fine_configuration())
    primary = bundle["primary_result"]
    primary["environment_config"]["grid"].update(resolution)
    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = resolution["v_count"]
    vehicle["gamma_count"] = resolution["gamma_count"]
    vehicle["heading_count"] = resolution["heading_count"]
    sensor = primary["sensor_config"]
    sensor["default_x_sensor"] = float(sensor_xy[0])
    sensor["default_y_sensor"] = float(sensor_xy[1])
    primary["bellman_config"]["search_options"]["exploration_orderings"] = (
        "low_gamma_first",
    )
    return bundle


def _mapped_last(function, arguments: tuple[np.ndarray, ...]) -> np.ndarray:
    count = arguments[0].size
    outputs = function.map(count)(
        *(np.asarray(argument, dtype=float).reshape(1, -1) for argument in arguments)
    )
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.maximum(0.0, np.asarray(values[-1], dtype=float).reshape(-1))


def _trajectory_library() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    with (REFERENCE_CONTINUOUS_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with np.load(REFERENCE_CONTINUOUS_DIR / "trajectory_data.npz") as handle:
        powered = np.asarray(handle["powered_path"], dtype=float)
        dense_time = np.asarray(handle["dense_time"], dtype=float)
        dense_states = np.asarray(handle["dense_states"], dtype=float)
    library = {}
    for topology, sign in (("south", 1.0), ("north", -1.0)):
        powered_path = powered.copy()
        states = dense_states.copy()
        if sign < 0.0:
            powered_path[:, 1] *= -1.0
            states[1] *= -1.0
            states[5] *= -1.0
            states[6] *= -1.0
        library[topology] = {
            "powered_path": powered_path,
            "dense_time": dense_time,
            "dense_states": states,
        }
    constants = {
        "powered_time_s": float(summary["powered_time_s"]),
        "mission_time_s": float(summary["mission_time_s"]),
        "powered_speed_m_s": 21.0,
        "powered_gamma_rad": float(np.deg2rad(summary["powered_gamma_deg"])),
        "powered_heading_abs_rad": abs(float(np.deg2rad(summary["powered_heading_deg"]))),
    }
    return library, constants


def _screen_one(
    sensor_xy: tuple[float, float],
    library: dict[str, dict[str, np.ndarray]],
    constants: dict[str, float],
) -> dict[str, Any]:
    configuration = _configuration(SCREEN_RESOLUTION, sensor_xy)
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    started = time.perf_counter()
    try:
        geometry_bundle = build_geometry_bundle(configuration)
        if not geometry_bundle["status"]["success"]:
            raise RuntimeError(geometry_bundle["status"]["message"])
        detection_bundle = build_symbolic_detection_bundle(
            configuration, geometry_bundle,
        )
        if not detection_bundle["status"]["success"]:
            raise RuntimeError(detection_bundle["status"]["message"])
        geometry = geometry_bundle["primary_result"]
        functions = detection_bundle["primary_result"]["functions"]
        sensor = np.asarray(geometry["sensor_position"], dtype=float)
        coverage = float(geometry["coverage"]["normalized_coverage_volume"])
        attacker_options = []
        for topology, path in library.items():
            powered = path["powered_path"]
            states = path["dense_states"]
            powered_count = powered.shape[0]
            heading_sign = -1.0 if topology == "south" else 1.0
            powered_rates = _mapped_last(
                functions["powered_total_detection_components"],
                (
                    powered[:, 0], powered[:, 1], powered[:, 2],
                    np.full(powered_count, constants["powered_speed_m_s"]),
                    np.full(powered_count, constants["powered_gamma_rad"]),
                    np.full(
                        powered_count,
                        heading_sign * constants["powered_heading_abs_rad"],
                    ),
                    np.full(powered_count, sensor[0]),
                    np.full(powered_count, sensor[1]),
                    np.full(powered_count, sensor[2]),
                ),
            )
            powered_hazard = float(np.trapezoid(
                powered_rates,
                np.linspace(0.0, constants["powered_time_s"], powered_count),
            ))
            glide_count = states.shape[1]
            glide_rates = _mapped_last(
                functions["glide_detection_components"],
                (
                    states[0], states[1], states[2], states[3], states[4], states[5],
                    np.full(glide_count, sensor[0]),
                    np.full(glide_count, sensor[1]),
                    np.full(glide_count, sensor[2]),
                ),
            )
            glide_hazard = float(np.trapezoid(glide_rates, path["dense_time"]))
            attacker_outputs = functions["attacker_objective"](
                powered_hazard, glide_hazard,
                constants["powered_time_s"],
                constants["mission_time_s"] - constants["powered_time_s"],
            )
            output_tuple = (
                attacker_outputs if isinstance(attacker_outputs, tuple)
                else (attacker_outputs,)
            )
            attacker_options.append({
                "topology": topology,
                "powered_hazard": powered_hazard,
                "glide_hazard": glide_hazard,
                "mission_hazard": powered_hazard + glide_hazard,
                "mission_pod": float(1.0 - np.exp(-(powered_hazard + glide_hazard))),
                "attacker_objective": float(output_tuple[-1]),
            })
        attacker = min(attacker_options, key=lambda item: item["attacker_objective"])
        defender_outputs = functions["defender_objective"](
            attacker["powered_hazard"], attacker["glide_hazard"], coverage,
        )
        defender_tuple = (
            defender_outputs if isinstance(defender_outputs, tuple)
            else (defender_outputs,)
        )
        return {
            "status_success": True,
            "x_sensor": float(sensor[0]),
            "y_sensor": float(sensor[1]),
            "h_sensor": float(sensor[2]),
            "coverage_volume_normalized": coverage,
            "selected_attacker_topology": attacker["topology"],
            "attacker_objective": attacker["attacker_objective"],
            "mission_hazard": attacker["mission_hazard"],
            "mission_pod": attacker["mission_pod"],
            "defender_pod_normalized": float(defender_tuple[0]),
            "defender_objective": float(defender_tuple[-1]),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        close_phase_logger(logger)


def _select_diverse_finalists(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (record for record in records if record.get("status_success")),
        key=lambda record: record["defender_objective"],
        reverse=True,
    )
    selected = []
    for record in ordered:
        point = np.array([record["x_sensor"], record["y_sensor"]])
        if all(
            np.linalg.norm(point - np.array([item["x_sensor"], item["y_sensor"]]))
            >= MINIMUM_FINALIST_SEPARATION_M
            for item in selected
        ):
            selected.append(record)
        if len(selected) == MEDIUM_FINALIST_COUNT:
            break
    return selected


def _save_discrete(
    output_dir: Path,
    resolution_name: str,
    resolution: dict[str, int],
    sensor_xy: tuple[float, float],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(resolution, sensor_xy)
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    started = time.perf_counter()
    try:
        geometry_bundle = build_geometry_bundle(configuration)
        if not geometry_bundle["status"]["success"]:
            raise RuntimeError(geometry_bundle["status"]["message"])
        detection_bundle = build_symbolic_detection_bundle(
            configuration, geometry_bundle,
        )
        bellman_bundle, attacker_bundle = solve_physical_successor_grid_attacker(
            configuration, geometry_bundle, detection_bundle,
        )
        geometry = geometry_bundle["primary_result"]
        best = attacker_bundle["primary_result"]
        powered_hazard = float(best["hazard_breakdown"]["powered_acoustic_hazard"])
        glide_hazard = float(best["hazard_breakdown"]["glide_radar_doppler_hazard"])
        coverage = float(geometry["coverage"]["normalized_coverage_volume"])
        defender_outputs = detection_bundle["primary_result"]["functions"][
            "defender_objective"
        ](powered_hazard, glide_hazard, coverage)
        defender_tuple = (
            defender_outputs if isinstance(defender_outputs, tuple)
            else (defender_outputs,)
        )
        trajectory = np.asarray(best["trajectory"])
        summary = {
            "status_success": True,
            "stage": resolution_name,
            "resolution": resolution,
            "elapsed_seconds": time.perf_counter() - started,
            "sensor_position": np.asarray(geometry["sensor_position"]).tolist(),
            "goal_position": np.asarray(geometry["goal_position"]).tolist(),
            "switching_point": np.asarray(best["switching_point"]).tolist(),
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "mission_time": float(best["mission_time"]),
            "powered_time": float(best["powered_time"]),
            "glide_time": float(best["glide_time"]),
            "mission_hazard": powered_hazard + glide_hazard,
            "coverage_volume_normalized": coverage,
            "defender_pod_normalized": float(defender_tuple[0]),
            "defender_objective": float(defender_tuple[-1]),
            "trajectory_node_count": int(trajectory.shape[0]),
            "minimum_glide_terrain_clearance_m": float(
                best["constraint_residuals"]["minimum_terrain_margin"]
            ),
            "maximum_turn_rate_deg_s": float(
                best["constraint_residuals"]["maximum_turn_rate_deg_s"]
            ),
            "goal_error_m": float(best["constraint_residuals"]["goal_error_norm"]),
            "projection_6d_to_3d_modified": False,
            "projection_used": False,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        terrain = geometry["terrain_arrays"]
        np.savez_compressed(
            output_dir / "trajectory_data.npz",
            terrain_x=np.asarray(terrain["x"]),
            terrain_y=np.asarray(terrain["y"]),
            terrain_height=np.asarray(terrain["height"]),
            sensor_position=np.asarray(geometry["sensor_position"]),
            goal_position=np.asarray(geometry["goal_position"]),
            switching_point=np.asarray(best["switching_point"]),
            powered_path=np.asarray(best["powered_path"]),
            trajectory=trajectory,
            speed_profile=np.asarray(best["speed_profile"]),
            gamma_profile=np.asarray(best["gamma_profile"]),
            heading_profile=np.asarray(best["heading_profile"]),
            initial_heading_state=np.asarray(best["initial_heading_state"]),
            duration_profile=np.asarray(best["duration_profile"]),
        )
        return summary
    finally:
        close_phase_logger(logger)
        gc.collect()


def _continuous_refine(
    output_dir: Path,
    sensor_xy: tuple[float, float],
    discrete_result_dir: Path,
    maximum_cpu_time_s: float = 120.0,
    maximum_iterations: int = 3000,
    topologies: tuple[str, ...] = ("south", "north"),
) -> dict[str, Any]:
    with (discrete_result_dir / "summary.json").open(encoding="utf-8") as handle:
        discrete_summary = json.load(handle)
    with np.load(discrete_result_dir / "trajectory_data.npz") as handle:
        launch = np.asarray(handle["powered_path"][0], dtype=float)
    switching_point = np.asarray(discrete_summary["switching_point"], dtype=float)
    horizontal_distance = float(np.linalg.norm(switching_point[:2] - launch[:2]))
    # The NLP reserves 2 m below the ceiling for post-switch upward inertia.
    # Preserve the discrete horizontal switching location while nudging an
    # on-ceiling discrete seed into that admissible continuous buffer.
    seed_switch_height = min(float(switching_point[2]), 197.5)
    vertical_distance = seed_switch_height - launch[2]
    initial_gamma_deg = float(np.rad2deg(np.arctan2(
        vertical_distance, horizontal_distance,
    )))
    initial_powered_time_s = float(
        np.hypot(horizontal_distance, vertical_distance) / 21.0
    )
    attempts: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    attempt_records: list[dict[str, Any]] = []
    for topology in topologies:
        try:
            result = solve_continuous_refinement(
                interval_count=50,
                initial_topology=topology,
                initial_gamma_deg=initial_gamma_deg,
                initial_powered_time_s=initial_powered_time_s,
                maximum_cpu_time_s=maximum_cpu_time_s,
                discrete_result_dir=discrete_result_dir,
                initialization_source="continuous_solution",
                continuous_warm_start_dir=REFERENCE_CONTINUOUS_DIR,
                sensor_xy=sensor_xy,
                accept_limited_solution=True,
                maximum_iterations=maximum_iterations,
            )
            validation = _dense_validate(result)
            hazard = validation["mission_hazard"]
            coverage = float(result["coverage"]["normalized_coverage_volume"])
            weights = result["configuration"]["primary_result"]["cost_config"][
                "defender"
            ]
            hazard_reference = float(
                weights["normalization"]["pod"]["hazard_reference"]
            )
            pod_normalized = hazard / (hazard + hazard_reference)
            defender_objective = (
                float(weights["w_pod"]) * pod_normalized
                + float(weights["w_cover"]) * coverage
            )
            record = {
                "status_success": bool(validation["passed"]),
                "initial_topology": topology,
                "initialization_source": "validated_continuous_solution",
                "dense_validation_checks": validation["checks"],
                "solver_return_status": result["solver_stats"].get(
                    "return_status", ""
                ),
                "initial_powered_time_s": initial_powered_time_s,
                "initial_powered_gamma_deg": initial_gamma_deg,
                "sensor_position": np.asarray(result["sensor"]).tolist(),
                "switching_point": np.asarray(result["switch_state"][:3]).tolist(),
                "attacker_objective": result["physical_objective"],
                "mission_hazard": hazard,
                "mission_pod": validation["mission_pod"],
                "mission_time": result["powered_time"] + result["glide_time"],
                "coverage_volume_normalized": coverage,
                "defender_pod_normalized": pod_normalized,
                "defender_objective": defender_objective,
                "minimum_terrain_clearance_m": validation[
                    "minimum_terrain_clearance_m"
                ],
                "maximum_altitude_m": validation["maximum_altitude_m"],
                "maximum_bank_deg": validation["maximum_bank_deg"],
                "maximum_roll_rate_deg_s": validation["maximum_roll_rate_deg_s"],
                "minimum_speed_m_s": validation["minimum_speed_m_s"],
                "maximum_speed_m_s": validation["maximum_speed_m_s"],
                "goal_error_m": validation["goal_error_m"],
                "switch_continuity_residual": validation[
                    "switch_continuity_residual"
                ],
                "maximum_dense_propagation_residual": validation[
                    "maximum_dense_propagation_residual"
                ],
                "powered_hazard_reintegration_residual": validation[
                    "powered_hazard_reintegration_residual"
                ],
                "objective_reintegration_residual": validation[
                    "objective_reintegration_residual"
                ],
                "projection_6d_to_3d_modified": False,
                "projection_used": False,
            }
            attempts.append((record, result, validation))
            attempt_records.append(record)
        except Exception as exc:
            attempt_records.append({
                "status_success": False,
                "initial_topology": topology,
                "initialization_source": "validated_continuous_solution",
                "error": str(exc),
            })
    with (output_dir / "continuous_attempts.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(attempt_records, handle, indent=2)
    valid_attempts = [item for item in attempts if item[0]["status_success"]]
    if not valid_attempts:
        raise RuntimeError(
            "Neither south nor north continuous warm start produced a "
            "densely validated attacker response"
        )
    record, result, validation = min(
        valid_attempts, key=lambda item: item[0]["attacker_objective"]
    )
    record["attacker_best_response_selected_from"] = len(valid_attempts)
    with (output_dir / "continuous_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    np.savez_compressed(
        output_dir / "continuous_trajectory.npz",
        dense_time=validation["dense_time"],
        dense_states=validation["dense_states"],
        powered_path=(
            result["launch"][None, :]
            + np.linspace(0.0, 1.0, 301)[:, None]
            * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
        ),
        controls=result["controls"],
    )
    return record


def _write_progress(payload: dict[str, Any]) -> None:
    with (OUTPUT_DIR / "optimization_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    library, constants = _trajectory_library()
    points = [
        (x_sensor, y_sensor) for x_sensor in SCREEN_X for y_sensor in SCREEN_Y
    ]
    points.extend(point for point in SPECIAL_SCREEN_POINTS if point not in points)
    screening = []
    print(f"START screening {len(points)} sensor positions", flush=True)
    for index, point in enumerate(points, start=1):
        try:
            record = _screen_one(point, library, constants)
        except Exception as exc:
            record = {
                "status_success": False,
                "x_sensor": point[0], "y_sensor": point[1],
                "error": str(exc),
            }
        screening.append(record)
        if index % 10 == 0 or index == len(points):
            successful = [item for item in screening if item.get("status_success")]
            best_value = max(
                (item["defender_objective"] for item in successful),
                default=float("nan"),
            )
            print(
                f"SCREEN {index}/{len(points)} valid={len(successful)} "
                f"best={best_value:.6f}", flush=True,
            )
    finalists = _select_diverse_finalists(screening)
    payload: dict[str, Any] = {
        "status_success": False,
        "search_scope": "budgeted staged 2-D sensor search",
        "global_optimality_claimed": False,
        "screen_resolution": SCREEN_RESOLUTION,
        "screening_records": screening,
        "selected_medium_finalists": finalists,
        "medium_records": [],
        "continuous_medium_records": [],
        "fine_record": None,
        "continuous_fine_record": None,
        "failures": [],
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    _write_progress(payload)

    for index, finalist in enumerate(finalists):
        sensor_xy = (finalist["x_sensor"], finalist["y_sensor"])
        run_dir = OUTPUT_DIR / f"medium_{index + 1}_x{sensor_xy[0]:g}_y{sensor_xy[1]:g}"
        print(f"START medium {index + 1}/{len(finalists)} sensor={sensor_xy}", flush=True)
        try:
            discrete = _save_discrete(
                run_dir, "medium", MEDIUM_RESOLUTION, sensor_xy,
            )
            payload["medium_records"].append(discrete)
            print(
                f"DONE medium sensor={sensor_xy} D={discrete['defender_objective']:.6f} "
                f"PoD={100.0 * discrete['mission_pod']:.4f}%",
                flush=True,
            )
            continuous = _continuous_refine(run_dir, sensor_xy, run_dir)
            continuous["source_directory"] = str(run_dir)
            payload["continuous_medium_records"].append(continuous)
            print(
                f"DONE continuous sensor={sensor_xy} D={continuous['defender_objective']:.6f} "
                f"valid={continuous['status_success']}",
                flush=True,
            )
        except Exception as exc:
            payload["failures"].append({
                "stage": "medium", "sensor_xy": sensor_xy,
                "error": str(exc), "traceback": traceback.format_exc(),
            })
            print(f"FAILED medium sensor={sensor_xy}: {exc}", flush=True)
        _write_progress(payload)

    valid_continuous = [
        record for record in payload["continuous_medium_records"]
        if record["status_success"]
    ]
    if not valid_continuous:
        raise RuntimeError("No medium finalist produced a validated continuous response")
    selected = max(valid_continuous, key=lambda record: record["defender_objective"])
    selected_sensor = tuple(float(value) for value in selected["sensor_position"][:2])
    fine_dir = OUTPUT_DIR / f"fine_selected_x{selected_sensor[0]:g}_y{selected_sensor[1]:g}"
    print(f"START full fine confirmation sensor={selected_sensor}", flush=True)
    try:
        fine = _save_discrete(
            fine_dir, "fine", FINE_RESOLUTION, selected_sensor,
        )
        payload["fine_record"] = fine
        print(
            f"DONE fine D={fine['defender_objective']:.6f} "
            f"PoD={100.0 * fine['mission_pod']:.4f}% elapsed={fine['elapsed_seconds']:.1f}s",
            flush=True,
        )
        continuous_fine = _continuous_refine(
            fine_dir, selected_sensor, fine_dir,
        )
        payload["continuous_fine_record"] = continuous_fine
        payload["status_success"] = bool(continuous_fine["status_success"])
    except Exception as exc:
        payload["failures"].append({
            "stage": "fine", "sensor_xy": selected_sensor,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        print(f"FAILED fine sensor={selected_sensor}: {exc}", flush=True)
    payload["selected_sensor_xy"] = list(selected_sensor)
    payload["selected_medium_continuous_record"] = selected
    _write_progress(payload)
    print(json.dumps({
        "status_success": payload["status_success"],
        "selected_sensor_xy": payload["selected_sensor_xy"],
        "medium_evaluation_count": len(payload["medium_records"]),
        "failure_count": len(payload["failures"]),
        "fine_record": payload["fine_record"],
        "continuous_fine_record": payload["continuous_fine_record"],
        "global_optimality_claimed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
