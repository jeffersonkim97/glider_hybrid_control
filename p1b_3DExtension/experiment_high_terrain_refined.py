"""Refined-grid 3D detour test with an intentionally impassable hilltop.

The Gaussian hill peak and the airspace ceiling are both 200 m. With the
configured 1 m terrain clearance, a route cannot pass directly over the
summit; a feasible response must use the lateral dimension instead.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from .configuration import build_configuration_bundle
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry_bundle
from .phase_logging import close_phase_logger
from .successor_grid_solver import solve_physical_successor_grid_attacker


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "high_terrain_200_refined"
SENSOR_XY = (2000.0, 0.0)
RESOLUTION = {
    "x_count": 41,
    "y_count": 31,
    "h_count": 41,
    "v_count": 3,
    "gamma_count": 6,
    "heading_count": 36,
}
PEAK_HEIGHT_M = 200.0
AIRSPACE_CEILING_M = 200.0


def build_refined_configuration() -> dict:
    """Return the isolated high-terrain scenario configuration."""
    bundle = deepcopy(build_configuration_bundle(OUTPUT_DIR))
    primary = bundle["primary_result"]
    environment = primary["environment_config"]
    grid = environment["grid"]
    grid.update(RESOLUTION)
    grid["h_max"] = AIRSPACE_CEILING_M
    environment["airspace"]["h_max"] = AIRSPACE_CEILING_M
    environment["simulation"]["h_max"] = AIRSPACE_CEILING_M

    hills = list(environment["terrain"]["hills"])
    hills[0] = {**hills[0], "h_ridge": PEAK_HEIGHT_M}
    environment["terrain"]["hills"] = tuple(hills)

    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = RESOLUTION["v_count"]
    vehicle["gamma_count"] = RESOLUTION["gamma_count"]
    vehicle["heading_count"] = RESOLUTION["heading_count"]
    primary["bellman_config"]["search_options"]["exploration_orderings"] = (
        "low_gamma_first",
    )
    primary["bellman_config"]["search_options"][
        "minimum_glide_terrain_clearance"
    ] = float(vehicle["switching_constraints"]["terrain_clearance"])
    return bundle


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = build_refined_configuration()
    primary_config = configuration["primary_result"]
    logger = primary_config["logging_utilities"]["logger"]
    started = time.perf_counter()
    try:
        geometry_bundle = build_geometry_bundle(configuration)
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
        terrain = geometry["terrain_arrays"]
        diagnostics = bellman_bundle["primary_result"]["bellman_diagnostics"][
            "physical_successor_grid"
        ]
        trajectory = np.asarray(best["trajectory"])
        clearance = float(
            primary_config["vehicle_config"]["switching_constraints"][
                "terrain_clearance"
            ]
        )
        elapsed = time.perf_counter() - started
        summary = {
            "status_success": True,
            "elapsed_seconds": elapsed,
            "scenario": "200 m hill peak equals 200 m airspace ceiling",
            "resolution": RESOLUTION,
            "grid_spacing_m": {
                "dx": 2750.0 / (RESOLUTION["x_count"] - 1),
                "dy": 3000.0 / (RESOLUTION["y_count"] - 1),
                "dh": AIRSPACE_CEILING_M / (RESOLUTION["h_count"] - 1),
            },
            "terrain_peak_m": float(np.max(terrain["height"])),
            "airspace_ceiling_m": AIRSPACE_CEILING_M,
            "required_terrain_clearance_m": clearance,
            "direct_summit_overflight_feasible": bool(
                PEAK_HEIGHT_M + clearance <= AIRSPACE_CEILING_M
            ),
            "sensor_position": np.asarray(geometry["sensor_position"]).tolist(),
            "goal_position": np.asarray(geometry["goal_position"]).tolist(),
            "switching_point": np.asarray(best["switching_point"]).tolist(),
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "mission_time": float(best["mission_time"]),
            "powered_time": float(best["powered_time"]),
            "glide_time": float(best["glide_time"]),
            "maximum_path_altitude_m": float(np.max(trajectory[:, 2])),
            "maximum_absolute_lateral_excursion_m": float(
                np.max(np.abs(trajectory[:, 1]))
            ),
            "initial_heading_state_deg": float(
                np.rad2deg(best["initial_heading_state"])
            ),
            "maximum_turn_rate_deg_s": float(
                best["constraint_residuals"]["maximum_turn_rate_deg_s"]
            ),
            "configured_max_turn_rate_deg_s": float(
                best["constraint_residuals"]["configured_max_turn_rate_deg_s"]
            ),
            "trajectory_node_count": int(trajectory.shape[0]),
            "goal_error_m": float(
                best["constraint_residuals"]["goal_error_norm"]
            ),
            "maximum_edge_endpoint_residual_m": float(
                best["constraint_residuals"]["maximum_edge_endpoint_residual"]
            ),
            "minimum_glide_terrain_clearance_m": float(
                best["constraint_residuals"]["minimum_terrain_margin"]
            ),
            "minimum_clearance_point": np.asarray(
                best["constraint_residuals"]["minimum_terrain_margin_point"]
            ).tolist(),
            "model": "physical_successor_grid_3d_heading_state",
            "endpoint_snapping": False,
            "experiment_scope": (
                "refined high-terrain detour test; cell-offset action limits "
                "held fixed, so this is not a transported-action convergence run"
            ),
            "bellman_diagnostics": diagnostics,
        }
        with (OUTPUT_DIR / "summary.json").open(
            "w", encoding="utf-8",
        ) as handle:
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


if __name__ == "__main__":
    main()
