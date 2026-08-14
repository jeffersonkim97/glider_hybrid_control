"""Fine-grid 5 km single-hill scenario with a 400 m flight ceiling."""

from __future__ import annotations

import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from .detection import build_symbolic_detection_bundle
from .experiment_extreme_ridge_fine import build_fine_configuration
from .geometry import build_geometry_bundle
from .phase_logging import close_phase_logger
from .successor_grid_solver import solve_physical_successor_grid_attacker


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "long_range_single_hill_5km_fine"
GOAL_X_M = 5000.0
DOMAIN_X_MAX_M = 5500.0
AIRSPACE_CEILING_M = 400.0
SENSOR_XY = (5100.0, 0.0)
RIDGE = {
    "x_ridge": 1500.0,
    "y_ridge": 0.0,
    "h_ridge": 300.0,
    "width_x": 350.0,
    "width_y": 600.0,
}
RESOLUTION = {
    "x_count": 51,
    "y_count": 41,
    "h_count": 51,
    "v_count": 3,
    "gamma_count": 6,
    "heading_count": 36,
}


def build_long_range_configuration() -> dict:
    bundle = deepcopy(build_fine_configuration())
    primary = bundle["primary_result"]
    environment = primary["environment_config"]
    environment["x_goal"] = GOAL_X_M
    environment["terrain"]["x_max"] = DOMAIN_X_MAX_M
    environment["terrain"]["hills"] = (dict(RIDGE),)
    for section in ("grid", "airspace", "simulation"):
        environment[section]["x_max"] = DOMAIN_X_MAX_M
        environment[section]["h_max"] = AIRSPACE_CEILING_M
    environment["grid"].update(RESOLUTION)

    sensor = primary["sensor_config"]
    sensor["default_x_sensor"], sensor["default_y_sensor"] = SENSOR_XY
    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = RESOLUTION["v_count"]
    vehicle["gamma_count"] = RESOLUTION["gamma_count"]
    vehicle["heading_count"] = RESOLUTION["heading_count"]
    primary["cost_config"]["attacker"]["normalization"]["time"][
        "reference_seconds"
    ] = GOAL_X_M / float(vehicle["glide_speed_max"])
    defender_bounds = primary["defender_config"]["continuous_search_bounds"]
    defender_bounds["x_sensor_max"] = 5200.0
    search = primary["bellman_config"]["search_options"]
    search["switching_candidate_mode"] = "los_boundary_surface"
    search["exploration_orderings"] = ("low_gamma_first",)
    search["minimum_glide_terrain_clearance"] = float(
        vehicle["switching_constraints"]["terrain_clearance"]
    )
    return bundle


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = build_long_range_configuration()
    primary = configuration["primary_result"]
    logger = primary["logging_utilities"]["logger"]
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
        if not attacker_bundle["status"]["success"]:
            raise RuntimeError(attacker_bundle["status"]["message"])

        geometry = geometry_bundle["primary_result"]
        best = attacker_bundle["primary_result"]
        trajectory = np.asarray(best["trajectory"])
        terrain = geometry["terrain_arrays"]
        switch = np.asarray(best["switching_point"], dtype=float)
        switch_x_index = int(np.argmin(np.abs(terrain["x"] - switch[0])))
        switch_y_index = int(np.argmin(np.abs(terrain["y"] - switch[1])))
        switch_boundary_height = float(
            geometry["los_masks"]["los_boundary_height"][
                switch_x_index, switch_y_index
            ]
        )
        clearance = float(
            primary["bellman_config"]["search_options"][
                "minimum_glide_terrain_clearance"
            ]
        )
        summary = {
            "status_success": True,
            "scenario": "5 km single elliptical hill, 400 m ceiling",
            "elapsed_seconds": time.perf_counter() - started,
            "goal_x_m": GOAL_X_M,
            "domain_x_max_m": DOMAIN_X_MAX_M,
            "ridge": RIDGE,
            "airspace_ceiling_m": AIRSPACE_CEILING_M,
            "sensor_xy_rule": "100 m beyond goal on centerline",
            "resolution": RESOLUTION,
            "grid_spacing_m": {
                "dx": DOMAIN_X_MAX_M / (RESOLUTION["x_count"] - 1),
                "dy": 3000.0 / (RESOLUTION["y_count"] - 1),
                "dh": AIRSPACE_CEILING_M / (RESOLUTION["h_count"] - 1),
            },
            "sampled_terrain_peak_m": float(np.max(terrain["height"])),
            "required_terrain_clearance_m": clearance,
            "direct_peak_overflight_feasible": bool(
                RIDGE["h_ridge"] + clearance <= AIRSPACE_CEILING_M
            ),
            "sensor_position": np.asarray(geometry["sensor_position"]).tolist(),
            "goal_position": np.asarray(geometry["goal_position"]).tolist(),
            "switching_point": switch.tolist(),
            "switching_candidate_mode": "los_boundary_surface",
            "switch_los_boundary_height_m": switch_boundary_height,
            "switch_los_boundary_residual_m": float(
                abs(switch[2] - switch_boundary_height)
            ),
            "switch_los_boundary_tolerance_m": float(
                primary["vehicle_config"]["switching_constraints"]
                ["tangent_tolerance"]
            ),
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "mission_hazard": float(
                best["hazard_breakdown"]["mission_hazard"]
            ),
            "mission_time": float(best["mission_time"]),
            "powered_time": float(best["powered_time"]),
            "glide_time": float(best["glide_time"]),
            "maximum_path_altitude_m": float(np.max(trajectory[:, 2])),
            "maximum_absolute_lateral_excursion_m": float(
                np.max(np.abs(trajectory[:, 1]))
            ),
            "trajectory_node_count": int(trajectory.shape[0]),
            "minimum_glide_terrain_clearance_m": float(
                best["constraint_residuals"]["minimum_terrain_margin"]
            ),
            "maximum_turn_rate_deg_s": float(
                best["constraint_residuals"]["maximum_turn_rate_deg_s"]
            ),
            "goal_error_m": float(np.linalg.norm(
                trajectory[-1] - np.asarray(geometry["goal_position"])
            )),
            "projection_6d_to_3d_modified": False,
            "projection_used": False,
        }
        with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        np.savez_compressed(
            OUTPUT_DIR / "trajectory_data.npz",
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
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        close_phase_logger(logger)
        gc.collect()


if __name__ == "__main__":
    main()
