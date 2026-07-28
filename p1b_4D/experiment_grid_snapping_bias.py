"""2x2 factorial isolating spatial-grid vs. action-grid resolution effects
on mission_cost/PoD (follow-up to experiment_resolution_convergence.py,
which conflated the two axes in each "tier"). Root-causes the follower-
level convergence instability documented in p1b_roadmap_0727.md item 1:
the spatial (z,h) grid is the entire driver -- construct_coarse_
transitions' ceil-based grid-snapping systematically overstates distance
covered per discrete step at coarse spacing, directionally undercounting
mission_time/hazard (not just adding noise). See the roadmap's item 1
correction (2026-07-28) for the full writeup.

Run from the repo root: python -m p1b_4D.experiment_grid_snapping_bias
"""
from __future__ import annotations

from pathlib import Path

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.stackelberg_solver import evaluate_defender_position
from p1b_4D.phase_logging import close_phase_logger

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "grid_snapping_bias_factorial"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

Z_SENSOR = 4400.0  # the resolution-convergence sweep's worst-case position
SPATIAL = {"coarse": (121, 81), "fine": (321, 201)}  # (z_count, h_count)
ACTION = {"coarse": (3, 8), "fine": (5, 20)}  # (v_count, gamma_count)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
                print(
                    f"spatial={spatial_name}({z_count},{h_count}) "
                    f"action={action_name}({v_count},{gamma_count}): "
                    f"mission_cost={best['mission_cost']:.4f} "
                    f"pod={best['mission_pod']:.6f} "
                    f"time={best['mission_time']:.2f}",
                    flush=True,
                )
            finally:
                close_phase_logger(logger)


if __name__ == "__main__":
    main()
