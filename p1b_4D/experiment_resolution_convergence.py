"""2D follower- and defender-level resolution convergence sweep (plan
items 1-2 in p1b_roadmap_0727.md).

For 3 representative sensor positions x 3 grid resolutions (coarse/medium/
fine, "fine" == the current production default), run the full attacker
best-response pipeline and record everything needed to check both:
  - follower-level convergence: for each FIXED sensor position, do J_A,
    PoD, mission_time, switching_point, path topology stabilize as
    resolution increases?
  - defender-level convergence: for each FIXED resolution, does the J_D
    landscape shape / winning basin / sensor ranking across positions
    stay the same as resolution increases?

Both analyses are sliced from the same 3x3=9 evaluations, so one sweep
serves both plan items.

Run from the repo root: python -m p1b_4D.experiment_resolution_convergence
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
from p1b_4D.result_provenance import (
    build_result_provenance,
    provenance_from_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "resolution_convergence_2d"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

SENSOR_POSITIONS = [2900.0, 3625.0, 4400.0]  # near-lower, mid, near-upper within [2750, 4500]

RESOLUTIONS = {
    # z_count=81 (and below) produces zero feasible Bellman candidates at
    # this domain scale (0-5500) regardless of h/v/gamma -- empirically
    # verified before launching; z_count=121 is the coarsest that works.
    "coarse": {"z_count": 121, "h_count": 81, "v_count": 3, "gamma_count": 8},
    "medium": {"z_count": 201, "h_count": 131, "v_count": 4, "gamma_count": 14},
    "fine": {"z_count": 321, "h_count": 201, "v_count": 5, "gamma_count": 20},  # == production default
}


def build_config_at_resolution(resolution_name: str, project_root: Path):
    cb = build_configuration_bundle(project_root)
    cb = deepcopy(cb)
    cb["primary_result"]["attacker_solver_config"][
        "transition_model"
    ] = PAPER_TRANSITION_MODEL
    res = RESOLUTIONS[resolution_name]
    env = cb["primary_result"]["environment_config"]
    grid = env["grid"]
    z_min, z_max = grid["z_min"], grid["z_max"]
    h_min, h_max = grid["h_min"], grid["h_max"]
    grid["z_count"] = res["z_count"]
    grid["z_spacing"] = (z_max - z_min) / (res["z_count"] - 1)
    grid["h_count"] = res["h_count"]
    grid["h_spacing"] = (h_max - h_min) / (res["h_count"] - 1)
    grid["v_count"] = res["v_count"]
    grid["gamma_count"] = res["gamma_count"]
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = res["v_count"]
    vehicle["gamma_count"] = res["gamma_count"]
    return cb


def summarize(result: dict, configuration_bundle: dict) -> dict:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    return {
        "z_sensor": primary["z_sensor"],
        "h_sensor": primary["h_sensor"],
        "status_success": result["status"]["success"],
        "mission_cost": best["mission_cost"],
        "mission_pod": best["mission_pod"],
        "mission_time": best["mission_time"],
        "switching_point": list(np.asarray(best["switching_point"]).tolist()),
        "trajectory_point_count": int(np.asarray(best["trajectory"]).shape[0]),
        "trajectory_final": list(np.asarray(best["trajectory"])[-1].tolist()),
        "defender_objective": primary["defender_objective"],
        "transition_model": result["metadata"]["transition_model"],
        "continuous_feasible": replay["feasible"],
        "continuous_violation": replay["violation"],
        "continuous_goal_miss": replay["goal_miss"],
        "provenance": provenance_from_evaluation(
            configuration_bundle,
            result,
            script_identifier="p1b_4D/experiment_resolution_convergence.py",
        ),
        "objective_breakdown": {
            k: (float(v) if np.isscalar(v) else v)
            for k, v in primary["objective_breakdown"].items()
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for resolution_name in RESOLUTIONS:
        for z_sensor in SENSOR_POSITIONS:
            run_id = f"{resolution_name}-z{int(z_sensor)}"
            started = datetime.now(timezone.utc)
            print(f"[{started.isoformat()}] START {run_id}", flush=True)
            cb = build_config_at_resolution(resolution_name, OUTPUT_DIR)
            logger = cb["primary_result"]["logging_utilities"]["logger"]
            start = time.perf_counter()
            try:
                result = evaluate_defender_position(z_sensor, cb, run_id)
                elapsed = time.perf_counter() - start
                summary = summarize(result, cb)
                summary["resolution"] = resolution_name
                summary["resolution_params"] = RESOLUTIONS[resolution_name]
                summary["elapsed_seconds"] = elapsed
                all_results.append(summary)
                print(
                    f"[{run_id}] elapsed={elapsed:.1f}s "
                    f"mission_cost={summary['mission_cost']:.4f} "
                    f"mission_pod={summary['mission_pod']:.6f} "
                    f"switching_point={summary['switching_point']} "
                    f"defender_objective={summary['defender_objective']:.4f}",
                    flush=True,
                )
            except Exception:
                print(f"[{run_id}] FAILED:", flush=True)
                traceback.print_exc()
                all_results.append({
                    "resolution": resolution_name,
                    "z_sensor": z_sensor,
                    "status_success": False,
                    "error": traceback.format_exc(),
                    "provenance": build_result_provenance(
                        cb,
                        script_identifier=(
                            "p1b_4D/experiment_resolution_convergence.py"
                        ),
                    ),
                })
            finally:
                close_phase_logger(logger)

            with open(OUTPUT_DIR / "convergence_results.json", "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=2)

    print(f"\nAll runs complete. Results saved to {OUTPUT_DIR / 'convergence_results.json'}", flush=True)


if __name__ == "__main__":
    main()
