"""Continuation solve from the canonical sensor to the (2600, 0) candidate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_trajectory_refinement import (
    OUTPUT_DIR as CANONICAL_CONTINUOUS_DIR,
    REPO_ROOT,
    _dense_validate,
    solve_continuous_refinement,
)


OUTPUT_DIR = REPO_ROOT / "results" / "defender_sensor_continuation_x2600_y0"
WAYPOINTS = (
    (2100.0, 180.0),
    (2225.0, 135.0),
    (2350.0, 90.0),
    (2412.5, 67.5),
    (2600.0, 0.0),
)


def _defender_metrics(result: dict[str, Any], validation: dict[str, Any]) -> dict:
    hazard = float(validation["mission_hazard"])
    coverage = float(result["coverage"]["normalized_coverage_volume"])
    weights = result["configuration"]["primary_result"]["cost_config"]["defender"]
    hazard_reference = float(weights["normalization"]["pod"]["hazard_reference"])
    normalized_pod = hazard / (hazard + hazard_reference)
    return {
        "defender_pod_normalized": normalized_pod,
        "coverage_volume_normalized": coverage,
        "defender_objective": (
            float(weights["w_pod"]) * normalized_pod
            + float(weights["w_cover"]) * coverage
        ),
    }


def _record(
    result: dict[str, Any], validation: dict[str, Any],
    sensor_xy: tuple[float, float], stage_index: int,
) -> dict[str, Any]:
    metrics = _defender_metrics(result, validation)
    return {
        "status_success": bool(validation["passed"]),
        "stage_index": stage_index,
        "sensor_xy": list(sensor_xy),
        "sensor_position": np.asarray(result["sensor"]).tolist(),
        "solver_return_status": result["solver_stats"].get("return_status", ""),
        "switching_point": np.asarray(result["switch_state"][:3]).tolist(),
        "powered_time_s": float(result["powered_time"]),
        "powered_gamma_deg": float(np.rad2deg(result["powered_gamma"])),
        "powered_heading_deg": float(np.rad2deg(result["powered_heading"])),
        "glide_time_s": float(result["glide_time"]),
        "mission_time_s": float(result["powered_time"] + result["glide_time"]),
        "attacker_objective": float(result["physical_objective"]),
        "mission_hazard": float(validation["mission_hazard"]),
        "mission_pod": float(validation["mission_pod"]),
        **metrics,
        "dense_validation_checks": validation["checks"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "minimum_speed_m_s": validation["minimum_speed_m_s"],
        "maximum_speed_m_s": validation["maximum_speed_m_s"],
        "goal_error_m": validation["goal_error_m"],
        "switch_continuity_residual": validation["switch_continuity_residual"],
        "maximum_dense_propagation_residual": validation[
            "maximum_dense_propagation_residual"
        ],
        "powered_hazard_reintegration_residual": validation[
            "powered_hazard_reintegration_residual"
        ],
        "objective_reintegration_residual": validation[
            "objective_reintegration_residual"
        ],
        "nlp_speed_buffer_m_s": float(result["nlp_speed_buffer_m_s"]),
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }


def _save_stage(
    stage_dir: Path, record: dict[str, Any], result: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    np.savez_compressed(
        stage_dir / "trajectory_data.npz",
        shooting_states=np.asarray(result["states"]),
        controls=np.asarray(result["controls"]),
        dense_time=np.asarray(validation["dense_time"]),
        dense_states=np.asarray(validation["dense_states"]),
        switch_state=np.asarray(result["switch_state"]),
        sensor_position=np.asarray(result["sensor"]),
        goal_position=np.asarray(result["goal"]),
    )


def _load_stage(stage_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with (stage_dir / "summary.json").open(encoding="utf-8") as handle:
        record = json.load(handle)
    with np.load(stage_dir / "trajectory_data.npz") as handle:
        result = {
            "states": np.asarray(handle["shooting_states"]),
            "controls": np.asarray(handle["controls"]),
            "powered_time": float(record["powered_time_s"]),
            "powered_gamma": float(np.deg2rad(record["powered_gamma_deg"])),
            "powered_heading": float(np.deg2rad(record["powered_heading_deg"])),
            "glide_time": float(record["glide_time_s"]),
        }
    return record, result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    previous_result: dict[str, Any] | None = None
    for index, sensor_xy in enumerate(WAYPOINTS, start=1):
        stage_dir = OUTPUT_DIR / f"stage_{index}_x{sensor_xy[0]:g}_y{sensor_xy[1]:g}"
        if (stage_dir / "summary.json").exists():
            existing_record, existing_result = _load_stage(stage_dir)
            if existing_record["status_success"]:
                print(f"SKIP validated stage sensor={sensor_xy}", flush=True)
                records.append(existing_record)
                previous_result = existing_result
                continue
            # Reproject the nearly feasible prior attempt at the same sensor.
            previous_result = existing_result

        successful = False
        for attempt in range(1, 4):
            print(
                f"START continuation {index}/{len(WAYPOINTS)} "
                f"sensor={sensor_xy} attempt={attempt}", flush=True,
            )
            initialization_source = (
                "continuous_solution" if previous_result is None else "result_mapping"
            )
            result = solve_continuous_refinement(
                interval_count=50,
                initial_topology="south",
                initialization_source=initialization_source,
                continuous_warm_start_dir=CANONICAL_CONTINUOUS_DIR,
                initial_result=previous_result,
                sensor_xy=sensor_xy,
                maximum_cpu_time_s=90.0,
                maximum_iterations=5000,
                accept_limited_solution=True,
                nlp_speed_buffer_m_s=0.02,
            )
            validation = _dense_validate(result)
            record = _record(result, validation, sensor_xy, index)
            attempt_dir = OUTPUT_DIR / (
                f"stage_{index}_x{sensor_xy[0]:g}_y{sensor_xy[1]:g}_attempt{attempt}"
            )
            _save_stage(attempt_dir, record, result, validation)
            print(
                f"DONE sensor={sensor_xy} valid={record['status_success']} "
                f"D={record['defender_objective']:.6f} "
                f"PoD={100.0 * record['mission_pod']:.4f}% "
                f"solver={record['solver_return_status']}",
                flush=True,
            )
            if validation["passed"]:
                _save_stage(stage_dir, record, result, validation)
                records.append(record)
                previous_result = result
                successful = True
                break
            previous_result = result
        if not successful:
            raise RuntimeError(
                f"Continuation failed strict validation at {sensor_xy} "
                "after three projected restarts"
            )
        with (OUTPUT_DIR / "continuation_summary.json").open(
            "w", encoding="utf-8",
        ) as handle:
            json.dump({
                "status_success": False,
                "target_sensor_xy": list(WAYPOINTS[-1]),
                "records": records,
                "global_optimality_claimed": False,
                "projection_6d_to_3d_modified": False,
                "projection_used": False,
            }, handle, indent=2)

    payload = {
        "status_success": True,
        "target_sensor_xy": list(WAYPOINTS[-1]),
        "records": records,
        "selected_record": records[-1],
        "global_optimality_claimed": False,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    with (OUTPUT_DIR / "continuation_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["selected_record"], indent=2), flush=True)


if __name__ == "__main__":
    main()
