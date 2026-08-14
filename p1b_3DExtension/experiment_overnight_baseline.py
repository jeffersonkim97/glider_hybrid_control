"""Overnight reference-baseline run: the full exact p1b_3DExtension
Stackelberg solve (DIRECT outer loop over sensor (x, y), exact Bellman
attacker response nested inside), at the current committed default config.

Expected wall time: several hours (measured ~5 min/evaluation; one run
completed in 5.42 hours / 190 evaluations on 2026-07-28). This exists to
give the eventual RL/approximate solution something exact to validate
against for at least one scenario.

Note: the full result bundle cannot be pickled (CasADi SX objects inside
it raise `Cannot pickle SX objects without a casadi context`) -- this
script instead extracts and saves the JSON-serializable numeric fields
(trajectory, switching point, objective values) directly, so a re-run
does not silently lose everything but the printed summary again.

Run from the repo root: python -m p1b_3DExtension.experiment_overnight_baseline
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p1b_3DExtension.configuration import build_configuration_bundle
from p1b_3DExtension.stackelberg_solver import solve_stackelberg_game
from p1b_3DExtension.phase_logging import close_phase_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "overnight_3d_baseline"


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    print(f"[{started_at.isoformat()}] starting overnight 3D exact Stackelberg baseline", flush=True)

    configuration_bundle = build_configuration_bundle(OUTPUT_DIR)
    logger = configuration_bundle["primary_result"]["logging_utilities"]["logger"]

    start = time.perf_counter()
    try:
        result = solve_stackelberg_game(configuration_bundle)
        elapsed = time.perf_counter() - start
        print(f"elapsed_seconds: {elapsed:.1f} ({elapsed / 3600.0:.2f} hours)", flush=True)
        print(f"status: {result['status']['success']} {result['status']['message']}", flush=True)

        solution = result["primary_result"]["final_stackelberg_solution"]
        best = solution["optimal_attacker_strategy"]
        print(f"optimal_x_sensor: {solution['optimal_x_sensor']}", flush=True)
        print(f"optimal_y_sensor: {solution['optimal_y_sensor']}", flush=True)
        print(f"optimal_sensor_position: {solution['optimal_sensor_position']}", flush=True)
        print(f"defender_objective: {solution['defender_objective']}", flush=True)
        print(f"attacker_objective: {solution['attacker_objective']}", flush=True)
        print(f"mission_pod: {solution['mission_pod']}", flush=True)
        print(f"optimal_switching_point: {solution['optimal_switching_point']}", flush=True)
        print(
            f"outer_evaluation_count: "
            f"{len(result['primary_result']['outer_evaluation_summaries'])}",
            flush=True,
        )

        numeric_summary = {
            "elapsed_seconds": elapsed,
            "status_success": result["status"]["success"],
            "optimal_x_sensor": _to_jsonable(solution["optimal_x_sensor"]),
            "optimal_y_sensor": _to_jsonable(solution["optimal_y_sensor"]),
            "optimal_sensor_position": _to_jsonable(solution["optimal_sensor_position"]),
            "defender_objective": _to_jsonable(solution["defender_objective"]),
            "attacker_objective": _to_jsonable(solution["attacker_objective"]),
            "mission_pod": _to_jsonable(solution["mission_pod"]),
            "optimal_switching_point": _to_jsonable(solution["optimal_switching_point"]),
            "optimal_glide_trajectory": _to_jsonable(np.asarray(solution["optimal_glide_trajectory"])),
            "mission_time": _to_jsonable(best.get("mission_time")),
            "powered_path": _to_jsonable(np.asarray(best["powered_path"])) if "powered_path" in best else None,
            "outer_evaluation_count": len(result["primary_result"]["outer_evaluation_summaries"]),
            "outer_evaluation_summaries": [
                {k: _to_jsonable(v) for k, v in item.items()}
                for item in result["primary_result"]["outer_evaluation_summaries"]
            ],
        }
        with open(OUTPUT_DIR / "overnight_3d_baseline_summary.json", "w", encoding="utf-8") as handle:
            json.dump(numeric_summary, handle, indent=2)
        print(f"saved numeric summary to {OUTPUT_DIR / 'overnight_3d_baseline_summary.json'}", flush=True)
    except Exception:
        print("FAILED with exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        close_phase_logger(logger)
        finished_at = datetime.now(timezone.utc)
        print(f"[{finished_at.isoformat()}] done", flush=True)


if __name__ == "__main__":
    main()
