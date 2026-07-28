"""Fills a gap flagged by external review of p1b_roadmap_0727.md: the
grid-snapping bias investigation (experiment_grid_snapping_bias.py) was
only run against single_hill. two_hill/valley never had their own
resolution-convergence check, so it is currently unknown whether the
Stackelberg-vs-coverage-only margins reported for them
(experiment_multiterrain_baselines.py: ~0.0027 and ~0.0033) exceed or fall
within those terrains' own discretization noise.

Note: two_hill's and valley's existing "native" grids already use the
same physical spacing as single_hill's production ("fine") tier
(dz=dh=17.1875/2.0 m for two_hill; ~17.17/2.0 m for valley) -- they were
independently chosen that way earlier this session, before this
convergence investigation existed. This script treats that native
resolution as the "fine" comparison point and adds one deliberately
coarser tier per terrain to see whether the same ceil-snapping bias
pattern shows up there too.

Run from the repo root: python -m p1b_4D.experiment_resolution_convergence_multiterrain
"""
from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.stackelberg_solver import evaluate_defender_position
from p1b_4D.phase_logging import close_phase_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "resolution_convergence_multiterrain"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

TERRAINS = {
    "two_hill": {
        "z_min": 0.0, "z_max": 2750.0,
        "hills": (
            {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 2000.0, "h_ridge": 50.0, "width": 100.0},
        ),
        "z_goal": 2500.0,
        "h_max": 200.0,
        "sensor_bounds": (1300.0, 2200.0),
        "sensor_positions": [1400.0, 1750.0, 2100.0],
        "resolutions": {
            "coarse": {"z_count": 81, "h_count": 51, "v_count": 5, "gamma_count": 20},
            "fine": {"z_count": 161, "h_count": 101, "v_count": 5, "gamma_count": 20},  # native, == production spacing
        },
    },
    "goal_in_valley": {
        "z_min": 0.0, "z_max": 4000.0,
        "hills": (
            {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 3500.0, "h_ridge": 100.0, "width": 100.0},
        ),
        "z_goal": 2500.0,
        "h_max": 200.0,
        "sensor_bounds": (1700.0, 2200.0),
        "sensor_positions": [1750.0, 1950.0, 2150.0],
        "resolutions": {
            "coarse": {"z_count": 117, "h_count": 51, "v_count": 5, "gamma_count": 20},
            "fine": {"z_count": 234, "h_count": 101, "v_count": 5, "gamma_count": 20},  # native, ~production spacing
        },
    },
}


def build_config(terrain_name: str, resolution_name: str, project_root: Path) -> dict:
    params = TERRAINS[terrain_name]
    res = params["resolutions"][resolution_name]
    cb = build_configuration_bundle(project_root)
    cb = deepcopy(cb)
    cb["primary_result"]["attacker_solver_config"][
        "transition_model"
    ] = PAPER_TRANSITION_MODEL
    env = cb["primary_result"]["environment_config"]
    env["z_start"] = 0.0
    env["z_goal"] = params["z_goal"]
    env["terrain"] = {"z_min": params["z_min"], "z_max": params["z_max"], "hills": params["hills"]}
    z_min, z_max = params["z_min"], params["z_max"]
    h_min, h_max = 0.0, params["h_max"]
    env["grid"] = {
        "z_min": z_min, "z_max": z_max, "z_count": res["z_count"],
        "z_spacing": (z_max - z_min) / (res["z_count"] - 1),
        "h_min": h_min, "h_max": h_max, "h_count": res["h_count"],
        "h_spacing": (h_max - h_min) / (res["h_count"] - 1),
        "v_count": res["v_count"], "gamma_count": res["gamma_count"],
        "axis_order_4d": ("z", "h", "v", "gamma"),
    }
    env["airspace"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max}
    env["simulation"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max, "max_path_steps": 1000}
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = res["v_count"]
    vehicle["gamma_count"] = res["gamma_count"]
    costs = cb["primary_result"]["cost_config"]["attacker"]
    costs["w_pod"] = 0.5
    costs["w_time"] = 0.5
    costs["normalization"]["time"]["reference_seconds"] = params["z_goal"] / vehicle["glide_speed_max"]
    defender = cb["primary_result"]["defender_config"]
    z_sensor_min, z_sensor_max = params["sensor_bounds"]
    defender["continuous_search_bounds"] = {"z_sensor_min": z_sensor_min, "z_sensor_max": z_sensor_max}
    return cb


def summarize(result: dict) -> dict:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    return {
        "status_success": result["status"]["success"],
        "mission_cost": best["mission_cost"],
        "mission_pod": best["mission_pod"],
        "mission_time": best["mission_time"],
        "switching_point": list(np.asarray(best["switching_point"]).tolist()),
        "defender_objective": primary["defender_objective"],
        "transition_model": result["metadata"]["transition_model"],
        "continuous_feasible": replay["feasible"],
        "continuous_violation": replay["violation"],
        "continuous_goal_miss": replay["goal_miss"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    for terrain_name, params in TERRAINS.items():
        for resolution_name in params["resolutions"]:
            for z_sensor in params["sensor_positions"]:
                run_id = f"{terrain_name}-{resolution_name}-z{int(z_sensor)}"
                started = datetime.now(timezone.utc)
                print(f"[{started.isoformat()}] START {run_id}", flush=True)
                cb = build_config(terrain_name, resolution_name, OUTPUT_DIR / terrain_name / resolution_name)
                logger = cb["primary_result"]["logging_utilities"]["logger"]
                start = time.perf_counter()
                try:
                    result = evaluate_defender_position(z_sensor, cb, run_id)
                    elapsed = time.perf_counter() - start
                    summary = summarize(result)
                    summary.update({
                        "terrain": terrain_name, "resolution": resolution_name,
                        "z_sensor": z_sensor, "elapsed_seconds": elapsed,
                    })
                    all_results.append(summary)
                    print(
                        f"[{run_id}] elapsed={elapsed:.1f}s "
                        f"mission_cost={summary['mission_cost']:.4f} "
                        f"mission_pod={summary['mission_pod']:.6f} "
                        f"defender_objective={summary['defender_objective']:.4f}",
                        flush=True,
                    )
                except Exception:
                    print(f"[{run_id}] FAILED:", flush=True)
                    traceback.print_exc()
                    all_results.append({
                        "terrain": terrain_name, "resolution": resolution_name,
                        "z_sensor": z_sensor, "status_success": False,
                        "error": traceback.format_exc(),
                    })
                finally:
                    close_phase_logger(logger)

                with open(OUTPUT_DIR / "convergence_results_multiterrain.json", "w", encoding="utf-8") as handle:
                    json.dump(all_results, handle, indent=2)

    print(f"\nAll runs complete. Results saved to "
          f"{OUTPUT_DIR / 'convergence_results_multiterrain.json'}", flush=True)


if __name__ == "__main__":
    main()
