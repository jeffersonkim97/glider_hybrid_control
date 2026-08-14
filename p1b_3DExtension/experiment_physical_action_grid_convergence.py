"""Grid convergence with one transported physical action envelope.

Coarse and medium grids are solved from scratch.  The existing fine result is
reused only after asserting that the new metre-valued envelope generates the
exact same fine-grid cell offsets as the result's original 3 x 7 x 8 domain.
The legacy 6D-to-3D projection is not imported or used.
"""

from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_trajectory_refinement import (
    _dense_validate,
    solve_continuous_refinement,
)
from .detection import build_symbolic_detection_bundle
from .experiment_extreme_ridge_fine import (
    OUTPUT_DIR as FINE_RESULT_DIR,
    RESOLUTION as FINE_RESOLUTION,
    build_fine_configuration,
)
from .geometry import build_geometry_bundle
from .phase_logging import close_phase_logger
from .successor_grid_solver import (
    physical_action_offsets,
    solve_physical_successor_grid_attacker,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "physical_action_grid_convergence"
RESOLUTIONS = {
    "coarse": {
        "x_count": 31, "y_count": 21, "h_count": 31,
        "v_count": 3, "gamma_count": 6, "heading_count": 36,
    },
    "medium": {
        "x_count": 41, "y_count": 31, "h_count": 41,
        "v_count": 3, "gamma_count": 6, "heading_count": 36,
    },
    "fine": dict(FINE_RESOLUTION),
}
THRESHOLDS = {
    "objective_relative": 0.01,
    "pod_absolute": 0.0002,
    "time_relative": 0.02,
    "switch_distance_m": 50.0,
    "trajectory_rms_m": 25.0,
}
RESAMPLE_COUNT = 401


def _configuration(resolution_name: str) -> dict[str, Any]:
    bundle = deepcopy(build_fine_configuration())
    resolution = RESOLUTIONS[resolution_name]
    primary = bundle["primary_result"]
    primary["environment_config"]["grid"].update(resolution)
    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = resolution["v_count"]
    vehicle["gamma_count"] = resolution["gamma_count"]
    vehicle["heading_count"] = resolution["heading_count"]
    return bundle


def _action_domain_record(configuration: dict[str, Any]) -> dict[str, Any]:
    primary = configuration["primary_result"]
    grid = primary["environment_config"]["grid"]
    x = np.linspace(grid["x_min"], grid["x_max"], grid["x_count"])
    y = np.linspace(grid["y_min"], grid["y_max"], grid["y_count"])
    h = np.linspace(grid["h_min"], grid["h_max"], grid["h_count"])
    search = primary["bellman_config"]["search_options"]
    offsets = physical_action_offsets(x, y, h, search)
    edges = np.asarray([
        (forward * (x[1] - x[0]), lateral * (y[1] - y[0]), descent * (h[1] - h[0]))
        for forward, lateral, descent in offsets
    ])
    return {
        "mode": "physical_envelope",
        "configured_envelope_m": deepcopy(search["physical_action_envelope"]),
        "spatial_offset_count_before_control_filter": len(offsets),
        "grid_spacing_m": {
            "dx": float(x[1] - x[0]),
            "dy": float(y[1] - y[0]),
            "dh": float(h[1] - h[0]),
        },
        "realized_ranges_m": {
            "forward": [float(np.min(edges[:, 0])), float(np.max(edges[:, 0]))],
            "absolute_lateral": [
                float(np.min(np.abs(edges[:, 1]))),
                float(np.max(np.abs(edges[:, 1]))),
            ],
            "descent": [float(np.min(edges[:, 2])), float(np.max(edges[:, 2]))],
        },
        "offsets": [list(offset) for offset in offsets],
    }


def _save_discrete_result(
    output_dir: Path,
    resolution_name: str,
    elapsed_seconds: float,
    geometry_bundle: dict[str, Any],
    bellman_bundle: dict[str, Any],
    attacker_bundle: dict[str, Any],
    action_domain: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = geometry_bundle["primary_result"]
    best = attacker_bundle["primary_result"]
    trajectory = np.asarray(best["trajectory"])
    summary = {
        "status_success": True,
        "resolution_name": resolution_name,
        "resolution": RESOLUTIONS[resolution_name],
        "elapsed_seconds": elapsed_seconds,
        "action_domain": action_domain,
        "graph_metadata": bellman_bundle["metadata"]["graph_metadata"],
        "sensor_position": np.asarray(geometry["sensor_position"]).tolist(),
        "goal_position": np.asarray(geometry["goal_position"]).tolist(),
        "switching_point": np.asarray(best["switching_point"]).tolist(),
        "mission_cost": float(best["mission_cost"]),
        "mission_pod": float(best["mission_pod"]),
        "mission_time": float(best["mission_time"]),
        "powered_time": float(best["powered_time"]),
        "glide_time": float(best["glide_time"]),
        "trajectory_node_count": int(trajectory.shape[0]),
        "maximum_absolute_lateral_excursion_m": float(np.max(np.abs(trajectory[:, 1]))),
        "minimum_glide_terrain_clearance_m": float(
            best["constraint_residuals"]["minimum_terrain_margin"]
        ),
        "maximum_turn_rate_deg_s": float(
            best["constraint_residuals"]["maximum_turn_rate_deg_s"]
        ),
        "goal_error_m": float(best["constraint_residuals"]["goal_error_norm"]),
        "maximum_edge_endpoint_residual_m": float(
            best["constraint_residuals"]["maximum_edge_endpoint_residual"]
        ),
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


def _solve_discrete(resolution_name: str) -> tuple[dict[str, Any], Path]:
    run_dir = OUTPUT_DIR / resolution_name
    configuration = _configuration(resolution_name)
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    action_domain = _action_domain_record(configuration)
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
        summary = _save_discrete_result(
            run_dir, resolution_name, time.perf_counter() - started,
            geometry_bundle, bellman_bundle, attacker_bundle, action_domain,
        )
        return summary, run_dir
    finally:
        close_phase_logger(logger)


def _fine_reference() -> tuple[dict[str, Any], Path]:
    configuration = _configuration("fine")
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    try:
        action_domain = _action_domain_record(configuration)
    finally:
        close_phase_logger(logger)
    generated = {tuple(offset) for offset in action_domain["offsets"]}
    legacy = {
        (forward, lateral, descent)
        for forward in range(1, 4)
        for lateral in range(-3, 4)
        for descent in range(1, 9)
    }
    if generated != legacy:
        raise RuntimeError("Fine physical envelope does not reproduce legacy action set")
    with (FINE_RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        source = json.load(handle)
    summary = {
        **source,
        "resolution_name": "fine",
        "resolution": RESOLUTIONS["fine"],
        "action_domain": action_domain,
        "fine_reference_reused": True,
        "exact_action_set_identity_verified": True,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    return summary, FINE_RESULT_DIR


def _discrete_curve(result_dir: Path, summary: dict[str, Any]) -> np.ndarray:
    with np.load(result_dir / "trajectory_data.npz") as handle:
        powered = np.asarray(handle["powered_path"], dtype=float)
        glide = np.asarray(handle["trajectory"], dtype=float)
        durations = np.asarray(handle["duration_profile"], dtype=float)
    powered_time = float(summary["powered_time"])
    powered_times = np.linspace(0.0, powered_time, powered.shape[0])
    glide_times = powered_time + np.concatenate(([0.0], np.cumsum(durations)))
    time_values = np.concatenate((powered_times, glide_times[1:]))
    positions = np.vstack((powered, glide[1:]))
    query = np.linspace(0.0, 1.0, RESAMPLE_COUNT)
    normalized_time = time_values / time_values[-1]
    return np.column_stack([
        np.interp(query, normalized_time, positions[:, dimension])
        for dimension in range(3)
    ])


def _continuous_record(
    resolution_name: str, result_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    result = solve_continuous_refinement(
        interval_count=50,
        initial_gamma_deg=8.0,
        initial_topology="south",
        maximum_cpu_time_s=60.0,
        discrete_result_dir=result_dir,
    )
    validation = _dense_validate(result)
    if not validation["passed"]:
        raise RuntimeError(f"Continuous validation failed: {validation['checks']}")
    record = {
        "resolution_name": resolution_name,
        "validation_passed": True,
        "physical_objective": result["physical_objective"],
        "mission_pod": validation["mission_pod"],
        "mission_time_s": result["powered_time"] + result["glide_time"],
        "switch_position_m": np.asarray(result["switch_state"][:3]).tolist(),
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_bank_deg": validation["maximum_bank_deg"],
        "maximum_roll_rate_deg_s": validation["maximum_roll_rate_deg_s"],
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    powered_fraction = np.linspace(0.0, 1.0, 301)
    powered_time = powered_fraction * result["powered_time"]
    powered_position = (
        result["launch"][None, :]
        + powered_fraction[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    glide_time = np.asarray(validation["dense_time"])
    glide_position = np.asarray(validation["dense_states"][:3]).T
    times = np.concatenate((powered_time, glide_time[1:]))
    positions = np.vstack((powered_position, glide_position[1:]))
    normalized_time = times / times[-1]
    query = np.linspace(0.0, 1.0, RESAMPLE_COUNT)
    curve = np.column_stack([
        np.interp(query, normalized_time, positions[:, dimension])
        for dimension in range(3)
    ])
    return record, curve


def _comparison(
    records: dict[str, dict[str, Any]],
    curves: dict[str, np.ndarray],
    *,
    objective_key: str,
    pod_key: str,
    time_key: str,
    switch_key: str,
) -> dict[str, Any]:
    reference = records["fine"]
    output = {}
    for name, record in records.items():
        differences = {
            "objective_relative": abs(record[objective_key] - reference[objective_key])
            / max(abs(reference[objective_key]), 1.0e-12),
            "pod_absolute": abs(record[pod_key] - reference[pod_key]),
            "time_relative": abs(record[time_key] - reference[time_key])
            / max(abs(reference[time_key]), 1.0e-12),
            "switch_distance_m": float(np.linalg.norm(
                np.asarray(record[switch_key]) - np.asarray(reference[switch_key])
            )),
            "trajectory_rms_m": float(np.sqrt(np.mean(np.sum(
                (curves[name] - curves["fine"]) ** 2, axis=1
            )))),
        }
        output[name] = {
            "differences_from_fine": differences,
            "within_thresholds": {
                key: value <= THRESHOLDS[key]
                for key, value in differences.items()
            },
        }
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    discrete_records: dict[str, dict[str, Any]] = {}
    result_dirs: dict[str, Path] = {}
    failures = []
    for name in ("coarse", "medium"):
        print(f"START discrete {name}", flush=True)
        try:
            record, result_dir = _solve_discrete(name)
            discrete_records[name] = record
            result_dirs[name] = result_dir
            print(
                f"DONE discrete {name}: objective={record['mission_cost']:.9f} "
                f"PoD={100.0 * record['mission_pod']:.5f}% "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failures.append({
                "stage": "discrete", "resolution": name,
                "error": str(exc), "traceback": traceback.format_exc(),
            })
            print(f"FAILED discrete {name}: {exc}", flush=True)
    fine_record, fine_dir = _fine_reference()
    discrete_records["fine"] = fine_record
    result_dirs["fine"] = fine_dir
    print("VERIFIED fine action-set identity; reused validated reference", flush=True)

    discrete_curves = {
        name: _discrete_curve(result_dirs[name], record)
        for name, record in discrete_records.items()
    }
    continuous_records: dict[str, dict[str, Any]] = {}
    continuous_curves: dict[str, np.ndarray] = {}
    for name in ("coarse", "medium"):
        if name not in result_dirs:
            continue
        print(f"START continuous refinement from {name}", flush=True)
        try:
            record, curve = _continuous_record(name, result_dirs[name])
            continuous_records[name] = record
            continuous_curves[name] = curve
            print(
                f"DONE continuous {name}: objective={record['physical_objective']:.9f} "
                f"PoD={100.0 * record['mission_pod']:.5f}%",
                flush=True,
            )
        except Exception as exc:
            failures.append({
                "stage": "continuous", "resolution": name,
                "error": str(exc), "traceback": traceback.format_exc(),
            })
            print(f"FAILED continuous {name}: {exc}", flush=True)
    # Re-solve the inexpensive continuous fine refinement so every reported
    # continuous row is produced by the current code and validator.
    fine_continuous, fine_curve = _continuous_record("fine", fine_dir)
    continuous_records["fine"] = fine_continuous
    continuous_curves["fine"] = fine_curve

    discrete_comparison = _comparison(
        discrete_records, discrete_curves,
        objective_key="mission_cost", pod_key="mission_pod",
        time_key="mission_time", switch_key="switching_point",
    )
    continuous_comparison = _comparison(
        continuous_records, continuous_curves,
        objective_key="physical_objective", pod_key="mission_pod",
        time_key="mission_time_s", switch_key="switch_position_m",
    )
    medium_discrete_passed = all(
        discrete_comparison.get("medium", {}).get("within_thresholds", {}).values()
    )
    medium_continuous_passed = all(
        continuous_comparison.get("medium", {}).get("within_thresholds", {}).values()
    )
    payload = {
        "status_success": not failures,
        "physical_action_domain_preserved": True,
        "fine_reference_reused": True,
        "fine_exact_action_set_identity_verified": True,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
        "thresholds": THRESHOLDS,
        "discrete_medium_vs_fine_passed": medium_discrete_passed,
        "continuous_medium_vs_fine_passed": medium_continuous_passed,
        "overall_convergence_passed": (
            not failures and medium_discrete_passed and medium_continuous_passed
        ),
        "discrete_records": discrete_records,
        "discrete_comparison": discrete_comparison,
        "continuous_records": continuous_records,
        "continuous_comparison": continuous_comparison,
        "failures": failures,
    }
    with (OUTPUT_DIR / "convergence_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    np.savez_compressed(
        OUTPUT_DIR / "normalized_trajectory_curves.npz",
        **{f"discrete_{name}": curve for name, curve in discrete_curves.items()},
        **{f"continuous_{name}": curve for name, curve in continuous_curves.items()},
    )
    print(json.dumps({
        key: payload[key] for key in (
            "status_success", "discrete_medium_vs_fine_passed",
            "continuous_medium_vs_fine_passed", "overall_convergence_passed",
            "discrete_comparison", "continuous_comparison", "failures",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
