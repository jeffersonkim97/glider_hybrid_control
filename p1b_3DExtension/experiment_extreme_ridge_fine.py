"""Fine-grid 3D test with a ridge protruding above the flight ceiling."""

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
OUTPUT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_fine"
# Nearest fine-grid node to the intended (2000, 200) asymmetric placement.
# Grid alignment makes the geometry bundle's own-cell visibility check exact.
SENSOR_XY = (1980.0, 225.0)
RESOLUTION = {
    "x_count": 51,
    "y_count": 41,
    "h_count": 51,
    "v_count": 3,
    "gamma_count": 6,
    "heading_count": 36,
}
RIDGE = {
    "x_ridge": 1500.0,
    "y_ridge": 0.0,
    "h_ridge": 275.0,
    "width_x": 350.0,
    "width_y": 600.0,
}
AIRSPACE_CEILING_M = 200.0


def build_fine_configuration() -> dict:
    """Return the isolated extreme-ridge fine-grid configuration."""
    bundle = deepcopy(build_configuration_bundle(OUTPUT_DIR))
    primary = bundle["primary_result"]
    environment = primary["environment_config"]
    grid = environment["grid"]
    grid.update(RESOLUTION)
    grid["h_max"] = AIRSPACE_CEILING_M
    environment["airspace"]["h_max"] = AIRSPACE_CEILING_M
    environment["simulation"]["h_max"] = AIRSPACE_CEILING_M
    environment["terrain"]["hills"] = (dict(RIDGE),)

    sensor = primary["sensor_config"]
    sensor["default_x_sensor"], sensor["default_y_sensor"] = SENSOR_XY
    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = RESOLUTION["v_count"]
    vehicle["gamma_count"] = RESOLUTION["gamma_count"]
    vehicle["heading_count"] = RESOLUTION["heading_count"]
    search = primary["bellman_config"]["search_options"]
    search["exploration_orderings"] = ("low_gamma_first",)
    search["minimum_glide_terrain_clearance"] = float(
        vehicle["switching_constraints"]["terrain_clearance"]
    )
    return bundle


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = build_fine_configuration()
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
        trajectory = np.asarray(best["trajectory"])
        diagnostics = bellman_bundle["primary_result"]["bellman_diagnostics"][
            "physical_successor_grid"
        ]
        required_clearance = float(
            primary_config["bellman_config"]["search_options"][
                "minimum_glide_terrain_clearance"
            ]
        )
        elapsed = time.perf_counter() - started
        summary = {
            "status_success": True,
            "elapsed_seconds": elapsed,
            "scenario": "275 m elliptical ridge above 200 m ceiling",
            "ridge": RIDGE,
            "resolution": RESOLUTION,
            "grid_spacing_m": {
                "dx": 2750.0 / (RESOLUTION["x_count"] - 1),
                "dy": 3000.0 / (RESOLUTION["y_count"] - 1),
                "dh": AIRSPACE_CEILING_M / (RESOLUTION["h_count"] - 1),
            },
            "sampled_terrain_peak_m": float(np.max(terrain["height"])),
            "airspace_ceiling_m": AIRSPACE_CEILING_M,
            "required_terrain_clearance_m": required_clearance,
            "acoustic_occluded_rate_scale": float(
                primary_config["sensor_config"]["detection"][
                    "acoustic_occluded_rate_scale"
                ]
            ),
            "direct_ridge_overflight_feasible": bool(
                RIDGE["h_ridge"] + required_clearance
                <= AIRSPACE_CEILING_M
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
                "fine extreme-ridge demonstration using the shared "
                "grid-independent physical action envelope"
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
