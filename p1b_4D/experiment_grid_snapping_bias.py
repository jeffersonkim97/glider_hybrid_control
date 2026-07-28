"""2x2 factorial isolating spatial-grid vs. action-grid resolution effects.

The filename is retained for compatibility with the earlier legacy-snapping
investigation, but current runs explicitly use the physical successor-grid
solver.  No conclusion from the legacy factorial is carried into these new
results; the experiment re-estimates spatial, action, and interaction effects
for the authoritative transition model.

Run from the repo root: python -m p1b_4D.experiment_grid_snapping_bias
"""
from __future__ import annotations

import json
from pathlib import Path
import traceback

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.stackelberg_solver import evaluate_defender_position
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.result_provenance import (
    build_result_provenance,
    provenance_from_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "grid_snapping_bias_factorial"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

Z_SENSOR = 4400.0  # the resolution-convergence sweep's worst-case position
SPATIAL = {"coarse": (121, 81), "fine": (321, 201)}  # (z_count, h_count)
ACTION = {"coarse": (3, 8), "fine": (5, 20)}  # (v_count, gamma_count)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
    for spatial_name, (z_count, h_count) in SPATIAL.items():
        for action_name, (v_count, gamma_count) in ACTION.items():
            cb = build_configuration_bundle(OUTPUT_DIR / f"{spatial_name}_{action_name}")
            cb["primary_result"]["attacker_solver_config"][
                "transition_model"
            ] = PAPER_TRANSITION_MODEL
            grid = cb["primary_result"]["environment_config"]["grid"]
            grid["z_count"] = z_count
            grid["z_spacing"] = (grid["z_max"] - grid["z_min"]) / (z_count - 1)
            grid["h_count"] = h_count
            grid["h_spacing"] = (grid["h_max"] - grid["h_min"]) / (h_count - 1)
            grid["v_count"] = v_count
            grid["gamma_count"] = gamma_count
            vehicle = cb["primary_result"]["vehicle_config"]
            vehicle["glide_speed_count"] = v_count
            vehicle["gamma_count"] = gamma_count
            logger = cb["primary_result"]["logging_utilities"]["logger"]
            try:
                result = evaluate_defender_position(
                    Z_SENSOR, cb, f"{spatial_name}-{action_name}",
                )
                best = result["primary_result"]["best_found_attacker_response"]
                replay = best["continuous_replay_validation"]
                all_results.append({
                    "spatial_resolution": spatial_name,
                    "action_resolution": action_name,
                    "z_sensor": Z_SENSOR,
                    "mission_cost": best["mission_cost"],
                    "mission_pod": best["mission_pod"],
                    "mission_time": best["mission_time"],
                    "switching_point": [
                        float(value) for value in best["switching_point"]
                    ],
                    "transition_model": result["metadata"]["transition_model"],
                    "continuous_feasible": replay["feasible"],
                    "continuous_violation": replay["violation"],
                    "continuous_goal_miss": replay["goal_miss"],
                    "provenance": provenance_from_evaluation(
                        cb,
                        result,
                        script_identifier="p1b_4D/experiment_grid_snapping_bias.py",
                    ),
                })
                print(
                    f"spatial={spatial_name}({z_count},{h_count}) "
                    f"action={action_name}({v_count},{gamma_count}): "
                    f"mission_cost={best['mission_cost']:.4f} "
                    f"pod={best['mission_pod']:.6f} "
                    f"time={best['mission_time']:.2f}",
                    flush=True,
                )
            except Exception:
                all_results.append({
                    "spatial_resolution": spatial_name,
                    "action_resolution": action_name,
                    "z_sensor": Z_SENSOR,
                    "status_success": False,
                    "error": traceback.format_exc(),
                    "provenance": build_result_provenance(
                        cb,
                        script_identifier=(
                            "p1b_4D/experiment_grid_snapping_bias.py"
                        ),
                    ),
                })
            finally:
                (OUTPUT_DIR / "factorial_results.json").write_text(
                    json.dumps(all_results, indent=2), encoding="utf-8"
                )
                close_phase_logger(logger)


if __name__ == "__main__":
    main()
