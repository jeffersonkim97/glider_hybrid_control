"""3D independent resolution convergence check.

Checks whether 3D's OWN solution (mission_cost/PoD/switching point/sensor
ranking) stabilizes with resolution, independent of any 2D comparison --
per the correction that a growing 2D-vs-3D gap at finer resolution is not
by itself evidence of non-convergence (3D can express lateral escape paths
2D structurally cannot, so the two dimensions must be checked separately).

Given the 2D grid-snapping bias finding (p1b_4D.experiment_grid_snapping_
bias -- spatial grid dominates, action grid barely matters), the same
mechanism likely applies here too; this has not yet been isolated with a
2D-style factorial for 3D specifically -- known gap, not yet closed.

Run from the repo root: python -m p1b_3DExtension.experiment_resolution_convergence
"""
from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p1b_3DExtension.configuration import build_configuration_bundle
from p1b_3DExtension.stackelberg_solver import evaluate_defender_position
from p1b_3DExtension.phase_logging import close_phase_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "resolution_convergence_3d"

SENSOR_POSITIONS = [(1300.0, 0.0), (2300.0, 0.0)]  # within x:[1000,2600], y:[-600,600]

RESOLUTIONS = {
    # x_count=21/h_count=21 (and coarser) produced zero feasible Bellman
    # candidates -- empirically verified before launching; 31/31 is the
    # coarsest that worked in a quick smoke test.
    "coarse": {"x_count": 31, "y_count": 21, "h_count": 31, "v_count": 3, "gamma_count": 6, "heading_count": 24},
    "medium": {"x_count": 41, "y_count": 31, "h_count": 41, "v_count": 3, "gamma_count": 6, "heading_count": 36},  # == current default
    "fine": {"x_count": 61, "y_count": 41, "h_count": 61, "v_count": 4, "gamma_count": 8, "heading_count": 36},
}


def build_config_at_resolution(resolution_name: str, project_root: Path):
    cb = build_configuration_bundle(project_root)
    cb = deepcopy(cb)
    res = RESOLUTIONS[resolution_name]
    env = cb["primary_result"]["environment_config"]
    grid = env["grid"]
    grid["x_count"] = res["x_count"]
    grid["y_count"] = res["y_count"]
    grid["h_count"] = res["h_count"]
    grid["v_count"] = res["v_count"]
    grid["gamma_count"] = res["gamma_count"]
    grid["heading_count"] = res["heading_count"]
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = res["v_count"]
    vehicle["gamma_count"] = res["gamma_count"]
    vehicle["heading_count"] = res["heading_count"]
    return cb


def summarize(result: dict) -> dict:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    return {
        "x_sensor": primary["x_sensor"],
        "y_sensor": primary["y_sensor"],
        "status_success": result["status"]["success"],
        "mission_cost": best["mission_cost"],
        "mission_pod": best["mission_pod"],
        "mission_time": best["mission_time"],
        "switching_point": list(np.asarray(best["switching_point"]).tolist()),
        "trajectory_point_count": int(np.asarray(best["trajectory"]).shape[0]),
        "trajectory_final": list(np.asarray(best["trajectory"])[-1].tolist()),
        "defender_objective": primary["defender_objective"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for resolution_name in RESOLUTIONS:
        for x_sensor, y_sensor in SENSOR_POSITIONS:
            run_id = f"{resolution_name}-x{int(x_sensor)}"
            started = datetime.now(timezone.utc)
            print(f"[{started.isoformat()}] START {run_id}", flush=True)
            cb = build_config_at_resolution(resolution_name, OUTPUT_DIR)
            logger = cb["primary_result"]["logging_utilities"]["logger"]
            start = time.perf_counter()
            try:
                result = evaluate_defender_position((x_sensor, y_sensor), cb, run_id)
                elapsed = time.perf_counter() - start
                summary = summarize(result)
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
                    "x_sensor": x_sensor,
                    "y_sensor": y_sensor,
                    "status_success": False,
                    "error": traceback.format_exc(),
                })
            finally:
                close_phase_logger(logger)

            with open(OUTPUT_DIR / "convergence_results_3d.json", "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=2)

    print(f"\nAll 3D convergence runs complete. Results saved to "
          f"{OUTPUT_DIR / 'convergence_results_3d.json'}", flush=True)


if __name__ == "__main__":
    main()
