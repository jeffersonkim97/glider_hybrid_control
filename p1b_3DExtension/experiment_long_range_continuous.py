"""Continuous 3-DOF refinement of the 5 km single-hill scenario."""

from __future__ import annotations

import json

import numpy as np

from .continuous_trajectory_refinement import (
    _dense_validate,
    solve_continuous_refinement,
)
from .experiment_long_range_single_hill import (
    OUTPUT_DIR,
    SENSOR_XY,
    build_long_range_configuration,
)
INTERVAL_COUNT = 80


def main() -> None:
    with (OUTPUT_DIR / "summary.json").open(encoding="utf-8") as handle:
        discrete_summary = json.load(handle)
    with np.load(OUTPUT_DIR / "trajectory_data.npz") as handle:
        launch = np.asarray(handle["powered_path"][0], dtype=float)
    configuration = build_long_range_configuration()
    switch = np.asarray(discrete_summary["switching_point"], dtype=float)
    delta = switch - launch
    horizontal = float(np.linalg.norm(delta[:2]))
    initial_gamma_deg = float(np.rad2deg(np.arctan2(delta[2], horizontal)))
    initial_powered_time_s = float(np.linalg.norm(delta) / 21.0)
    prior_result = None
    initialization_source = "discrete"
    prior_summary_path = OUTPUT_DIR / "continuous_summary.json"
    prior_trajectory_path = OUTPUT_DIR / "continuous_trajectory.npz"
    if prior_summary_path.exists() and prior_trajectory_path.exists():
        with prior_summary_path.open(encoding="utf-8") as handle:
            prior_summary = json.load(handle)
        if (
            prior_summary.get("status_success", False)
            and prior_summary.get("switching_candidate_mode")
            == "los_boundary_surface"
        ):
            with np.load(prior_trajectory_path) as handle:
                prior_result = {
                    "states": np.asarray(handle["shooting_states"]),
                    "controls": np.asarray(handle["controls"]),
                    "powered_time": float(prior_summary["powered_time_s"]),
                    "powered_gamma": float(np.deg2rad(
                        prior_summary["powered_gamma_deg"]
                    )),
                    "powered_heading": float(np.deg2rad(
                        prior_summary["powered_heading_deg"]
                    )),
                    "glide_time": float(prior_summary["glide_time_s"]),
                }
            initialization_source = "result_mapping"

    result = solve_continuous_refinement(
        interval_count=INTERVAL_COUNT,
        initial_topology="south",
        initial_gamma_deg=initial_gamma_deg,
        initial_powered_time_s=initial_powered_time_s,
        maximum_cpu_time_s=300.0,
        maximum_iterations=9000,
        discrete_result_dir=OUTPUT_DIR,
        initialization_source=initialization_source,
        initial_result=prior_result,
        sensor_xy=SENSOR_XY,
        accept_limited_solution=True,
        nlp_speed_buffer_m_s=0.06,
        detection_hazard_scale=1.0,
        use_limited_memory_hessian=True,
        configuration_bundle=configuration,
        powered_time_bounds_s=(5.0, 180.0),
        glide_time_bounds_s=(120.0, 320.0),
        powered_clearance_factor=1.35,
        integration_substeps_per_interval=8,
        constrain_switch_to_los_boundary=True,
    )
    validation = _dense_validate(result)
    hazard = float(validation["mission_hazard"])
    coverage = float(result["coverage"]["normalized_coverage_volume"])
    weights = result["configuration"]["primary_result"]["cost_config"]["defender"]
    hazard_reference = float(weights["normalization"]["pod"]["hazard_reference"])
    normalized_pod = hazard / (hazard + hazard_reference)
    summary = {
        "status_success": bool(validation["passed"]),
        "scenario": "5 km single elliptical hill continuous 3-DOF",
        "solver_return_status": result["solver_stats"].get("return_status", ""),
        "interval_count": INTERVAL_COUNT,
        "sensor_position": np.asarray(result["sensor"]).tolist(),
        "goal_position": np.asarray(result["goal"]).tolist(),
        "switching_point": np.asarray(result["switch_state"][:3]).tolist(),
        "powered_time_s": float(result["powered_time"]),
        "powered_gamma_deg": float(np.rad2deg(result["powered_gamma"])),
        "powered_heading_deg": float(np.rad2deg(result["powered_heading"])),
        "glide_time_s": float(result["glide_time"]),
        "mission_time_s": float(result["powered_time"] + result["glide_time"]),
        "attacker_objective": float(result["physical_objective"]),
        "mission_hazard": hazard,
        "mission_pod": float(validation["mission_pod"]),
        "coverage_volume_normalized": coverage,
        "defender_pod_normalized": normalized_pod,
        "defender_objective": (
            float(weights["w_pod"]) * normalized_pod
            + float(weights["w_cover"]) * coverage
        ),
        "dense_validation_checks": validation["checks"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "minimum_speed_m_s": validation["minimum_speed_m_s"],
        "maximum_speed_m_s": validation["maximum_speed_m_s"],
        "maximum_bank_deg": validation["maximum_bank_deg"],
        "maximum_roll_rate_deg_s": validation["maximum_roll_rate_deg_s"],
        "goal_error_m": validation["goal_error_m"],
        "switch_continuity_residual": validation["switch_continuity_residual"],
        "switching_candidate_mode": "los_boundary_surface",
        "switch_los_boundary_height_m": validation[
            "switch_los_boundary_height_m"
        ],
        "switch_los_boundary_residual_m": validation[
            "switch_los_boundary_residual_m"
        ],
        "switch_los_boundary_tolerance_m": validation[
            "switch_los_boundary_tolerance_m"
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
        "detection_hazard_scale": 1.0,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }
    with (OUTPUT_DIR / "continuous_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    powered_path = (
        result["launch"][None, :]
        + np.linspace(0.0, 1.0, 401)[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    np.savez_compressed(
        OUTPUT_DIR / "continuous_trajectory.npz",
        shooting_states=np.asarray(result["states"]),
        controls=np.asarray(result["controls"]),
        dense_time=np.asarray(validation["dense_time"]),
        dense_states=np.asarray(validation["dense_states"]),
        powered_path=powered_path,
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not validation["passed"]:
        raise RuntimeError(
            f"Long-range continuous validation failed: {validation['checks']}"
        )


if __name__ == "__main__":
    main()
