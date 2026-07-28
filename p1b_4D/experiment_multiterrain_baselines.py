"""Plan item 4 runner (p1b_roadmap_0727.md): 3 terrains x 4 sensor-
selection baselines, each baseline re-evaluated by the SAME authoritative
evaluate_defender_position.

Run from the repo root: python -m p1b_4D.experiment_multiterrain_baselines
"""
from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import p1b_4D.experiment_strategic_baselines as sb
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.stackelberg_solver import evaluate_defender_position
from p1b_4D.phase_logging import close_phase_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "multiterrain_strategic_baselines"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

# Terrain params confirmed from each scenario's existing geometry_bundle.json
# export (results_scaled_singlehill_wcov05 / results_two_hill /
# results_goal_in_valley). Resolution for two-hill/valley reuses each
# terrain's own already-validated native grid. single_hill's grid was
# corrected 2026-07-28 to the actual production resolution ("fine" =
# current p1b_4D.configuration default) after the grid-snapping bias
# investigation (see experiment_grid_snapping_bias.py and the roadmap's
# item 1 correction) showed "medium" carries a directional undercounting
# bias -- results generated with the earlier medium-resolution single_hill
# config should be treated as superseded, not authoritative.
TERRAINS = {
    "single_hill": {
        "z_min": 0.0, "z_max": 5500.0,
        "hills": ({"z_ridge": 2500.0, "h_ridge": 200.0, "width": 400.0},),
        "z_goal": 5000.0,
        "grid": {"z_count": 321, "h_count": 201, "v_count": 5, "gamma_count": 20},  # production resolution
        "h_max": 400.0,
        "sensor_bounds": (2750.0, 4500.0),
        "default_z_sensor": 4000.0,
    },
    "two_hill": {
        "z_min": 0.0, "z_max": 2750.0,
        "hills": (
            {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 2000.0, "h_ridge": 50.0, "width": 100.0},
        ),
        "z_goal": 2500.0,
        "grid": {"z_count": 161, "h_count": 101, "v_count": 5, "gamma_count": 20},  # terrain's own native resolution
        "h_max": 200.0,
        "sensor_bounds": (1300.0, 2200.0),
        "default_z_sensor": 1750.0,
    },
    "goal_in_valley": {
        "z_min": 0.0, "z_max": 4000.0,
        "hills": (
            {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 3500.0, "h_ridge": 100.0, "width": 100.0},
        ),
        "z_goal": 2500.0,  # valley floor, between the two hills
        "grid": {"z_count": 234, "h_count": 101, "v_count": 5, "gamma_count": 20},  # terrain's own native resolution
        "h_max": 200.0,
        "sensor_bounds": (1700.0, 2200.0),
        "default_z_sensor": 1950.0,
    },
}


def build_terrain_config(terrain_name: str, project_root: Path) -> dict:
    params = TERRAINS[terrain_name]
    cb = build_configuration_bundle(project_root)
    cb = deepcopy(cb)
    cb["primary_result"]["attacker_solver_config"][
        "transition_model"
    ] = PAPER_TRANSITION_MODEL
    env = cb["primary_result"]["environment_config"]
    env["z_start"] = 0.0
    env["z_goal"] = params["z_goal"]
    env["terrain"] = {"z_min": params["z_min"], "z_max": params["z_max"], "hills": params["hills"]}
    grid_params = params["grid"]
    z_min, z_max = params["z_min"], params["z_max"]
    h_min, h_max = 0.0, params["h_max"]
    env["grid"] = {
        "z_min": z_min, "z_max": z_max, "z_count": grid_params["z_count"],
        "z_spacing": (z_max - z_min) / (grid_params["z_count"] - 1),
        "h_min": h_min, "h_max": h_max, "h_count": grid_params["h_count"],
        "h_spacing": (h_max - h_min) / (grid_params["h_count"] - 1),
        "v_count": grid_params["v_count"], "gamma_count": grid_params["gamma_count"],
        "axis_order_4d": ("z", "h", "v", "gamma"),
    }
    env["airspace"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max}
    env["simulation"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max, "max_path_steps": 1000}
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = grid_params["v_count"]
    vehicle["gamma_count"] = grid_params["gamma_count"]
    sensor = cb["primary_result"]["sensor_config"]
    sensor["default_z_sensor"] = params["default_z_sensor"]
    costs = cb["primary_result"]["cost_config"]["attacker"]
    costs["w_pod"] = 0.5
    costs["w_time"] = 0.5
    costs["normalization"]["time"]["reference_seconds"] = params["z_goal"] / vehicle["glide_speed_max"]
    defender = cb["primary_result"]["defender_config"]
    z_sensor_min, z_sensor_max = params["sensor_bounds"]
    defender["continuous_search_bounds"] = {"z_sensor_min": z_sensor_min, "z_sensor_max": z_sensor_max}
    return cb


def evaluate_baseline(configuration_bundle: dict, z_sensor: float, run_id: str) -> dict:
    result = evaluate_defender_position(z_sensor, configuration_bundle, run_id)
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    return {
        "z_sensor_selected": z_sensor,
        "status_success": result["status"]["success"],
        "mission_cost": best["mission_cost"],
        "mission_pod": best["mission_pod"],
        "mission_time": best["mission_time"],
        "defender_objective": primary["defender_objective"],
        "switching_point": list(np.asarray(best["switching_point"]).tolist()),
        "transition_model": result["metadata"]["transition_model"],
        "continuous_feasible": replay["feasible"],
        "continuous_violation": replay["violation"],
        "continuous_goal_miss": replay["goal_miss"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, dict]] = {}

    for terrain_name in TERRAINS:
        params = TERRAINS[terrain_name]
        bounds = params["sensor_bounds"]
        all_results[terrain_name] = {}
        print(f"\n=== TERRAIN: {terrain_name} ===", flush=True)

        cb = build_terrain_config(terrain_name, OUTPUT_DIR / terrain_name)
        logger = cb["primary_result"]["logging_utilities"]["logger"]
        try:
            selected_positions: dict[str, float] = {}

            t0 = time.perf_counter()
            selected_positions["fixed"] = sb.select_fixed_sensor(cb)
            print(f"  [{terrain_name}] fixed selected: {selected_positions['fixed']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)

            t0 = time.perf_counter()
            selected_positions["coverage_only"] = sb.select_coverage_only_sensor(cb, bounds)
            print(f"  [{terrain_name}] coverage_only selected: {selected_positions['coverage_only']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)

            t0 = time.perf_counter()
            reference_z = 0.5 * (bounds[0] + bounds[1])
            nominal_response = sb.compute_nominal_attacker_path(cb, reference_z)
            selected_positions["nominal_path"] = sb.select_nominal_path_optimal_sensor(cb, bounds, nominal_response)
            print(f"  [{terrain_name}] nominal_path selected: {selected_positions['nominal_path']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)

            t0 = time.perf_counter()
            selected_positions["stackelberg"] = sb.select_stackelberg_optimal_sensor(cb, bounds)
            print(f"  [{terrain_name}] stackelberg selected: {selected_positions['stackelberg']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)

            for baseline_name, z_sensor in selected_positions.items():
                run_id = f"{terrain_name}-{baseline_name}"
                try:
                    evaluation = evaluate_baseline(cb, z_sensor, run_id)
                    all_results[terrain_name][baseline_name] = evaluation
                    print(
                        f"  [{run_id}] EVAL z={z_sensor:.2f} "
                        f"mission_pod={evaluation['mission_pod']:.4f} "
                        f"defender_objective={evaluation['defender_objective']:.4f}",
                        flush=True,
                    )
                except Exception:
                    print(f"  [{run_id}] EVAL FAILED:", flush=True)
                    traceback.print_exc()
                    all_results[terrain_name][baseline_name] = {"status_success": False, "error": traceback.format_exc()}
        except Exception:
            print(f"  [{terrain_name}] TERRAIN-LEVEL FAILURE:", flush=True)
            traceback.print_exc()
        finally:
            close_phase_logger(logger)

        with open(OUTPUT_DIR / "multiterrain_baseline_results.json", "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)

    print(f"\nAll terrains complete. Results saved to {OUTPUT_DIR / 'multiterrain_baseline_results.json'}", flush=True)


if __name__ == "__main__":
    main()
