"""Run one reproducible fixed-sensor mission with turn-limited heading state."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from .configuration import build_configuration_bundle
from .phase_logging import close_phase_logger
from .stackelberg_solver import evaluate_defender_position


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "turn_limited_3d_coarse"
SENSOR_XY = (2000.0, 0.0)
RESOLUTION = {
    "x_count": 31,
    "y_count": 21,
    "h_count": 31,
    "v_count": 3,
    "gamma_count": 6,
    "heading_count": 24,
}


def build_coarse_configuration() -> dict:
    bundle = deepcopy(build_configuration_bundle(OUTPUT_DIR))
    primary = bundle["primary_result"]
    grid = primary["environment_config"]["grid"]
    grid.update(RESOLUTION)
    vehicle = primary["vehicle_config"]
    vehicle["glide_speed_count"] = RESOLUTION["v_count"]
    vehicle["gamma_count"] = RESOLUTION["gamma_count"]
    vehicle["heading_count"] = RESOLUTION["heading_count"]
    # Exploration orderings differ only in deterministic tie breaking and
    # would duplicate the expensive 4D heading-state value solve.
    primary["bellman_config"]["search_options"]["exploration_orderings"] = (
        "low_gamma_first",
    )
    return bundle


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configuration = build_coarse_configuration()
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    started = time.perf_counter()
    try:
        result = evaluate_defender_position(
            SENSOR_XY,
            configuration,
            evaluation_id="turn-limited-coarse-fixed-sensor",
        )
        primary = result["primary_result"]
        best = primary["best_found_attacker_response"]
        geometry = primary["attacker_best_response_bundle"]["primary_result"][
            "geometry_bundle"
        ]["primary_result"]
        bellman_diagnostics = primary["attacker_best_response_bundle"][
            "primary_result"
        ]["bellman_candidate_bundle"]["primary_result"]["bellman_diagnostics"][
            "physical_successor_grid"
        ]
        terrain = geometry["terrain_arrays"]
        elapsed = time.perf_counter() - started
        summary = {
            "status_success": result["status"]["success"],
            "elapsed_seconds": elapsed,
            "resolution": RESOLUTION,
            "sensor_position": np.asarray(primary["sensor_position"]).tolist(),
            "switching_point": np.asarray(best["switching_point"]).tolist(),
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "mission_time": float(best["mission_time"]),
            "powered_time": float(best["powered_time"]),
            "glide_time": float(best["glide_time"]),
            "initial_heading_state_deg": float(
                np.rad2deg(best["initial_heading_state"])
            ),
            "maximum_turn_rate_deg_s": float(
                best["constraint_residuals"]["maximum_turn_rate_deg_s"]
            ),
            "configured_max_turn_rate_deg_s": float(
                best["constraint_residuals"]["configured_max_turn_rate_deg_s"]
            ),
            "trajectory_node_count": int(np.asarray(best["trajectory"]).shape[0]),
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
            "presentation_ready": True,
            "model": "physical_successor_grid_3d_heading_state",
            "authoritative_scope": (
                "coarse exact-edge fixed-sensor Bellman result; finite grid "
                "optimality only"
            ),
            "endpoint_snapping": False,
            "bellman_diagnostics": bellman_diagnostics,
        }
        with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        np.savez_compressed(
            OUTPUT_DIR / "trajectory_data.npz",
            terrain_x=np.asarray(terrain["x"]),
            terrain_y=np.asarray(terrain["y"]),
            terrain_height=np.asarray(terrain["height"]),
            sensor_position=np.asarray(primary["sensor_position"]),
            goal_position=np.asarray(geometry["goal_position"]),
            switching_point=np.asarray(best["switching_point"]),
            powered_path=np.asarray(best["powered_path"]),
            trajectory=np.asarray(best["trajectory"]),
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
