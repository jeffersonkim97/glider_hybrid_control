"""Run one 3D multi-terrain Stackelberg validation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .run_stackelberg import (
    _create_figure,
    _json_default,
    _progress,
    _serializable_result,
)
from .stackelberg import solve_stackelberg_game
from .terrain_scenarios import build_scenario_configuration


ROOT = Path(__file__).resolve().parent
SUPPORTED_SCENARIOS = ("two_hill", "goal_in_valley")


def output_paths(scenario_id: str) -> dict[str, Path]:
    result_dir = ROOT / "results" / "stage_10_multiterrain" / scenario_id
    figure_dir = ROOT / "figures" / "stage_10_multiterrain"
    return {
        "result_dir": result_dir,
        "cache_dir": result_dir / "evaluation_cache",
        "summary": result_dir / "stackelberg_solution.json",
        "npz": result_dir / "stackelberg_solution.npz",
        "png": figure_dir / f"{scenario_id}_stackelberg_solution.png",
        "pdf": figure_dir / f"{scenario_id}_stackelberg_solution.pdf",
    }


def save_result(result: dict, scenario_id: str) -> dict[str, Path]:
    paths = output_paths(scenario_id)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    paths["png"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(
        json.dumps(_serializable_result(result), indent=2, default=_json_default),
        encoding="utf-8",
    )
    pipeline = result["final_evaluation"]["pipeline"]
    geometry = pipeline["geometry"]
    trajectory = pipeline["trajectory"]
    replay = pipeline["continuous_replay"]
    np.savez_compressed(
        paths["npz"],
        optimal_sensor_position=np.asarray(result["optimal_sensor_position_m"]),
        terrain_x=geometry["x_grid"], terrain_y=geometry["y_grid"],
        terrain_height=geometry["terrain_height"],
        los_boundary_height=geometry["los_boundary_height"],
        powered_path=trajectory["powered_path"],
        glide_trajectory=trajectory["glide_trajectory"],
        continuous_replay_trajectory=replay["trajectory"],
        duration_profile_s=trajectory["duration_profile_s"],
        speed_profile_mps=trajectory["speed_profile_mps"],
        gamma_profile_rad=trajectory["gamma_profile_rad"],
        heading_profile_rad=trajectory["heading_profile_rad"],
        hazard_profile=trajectory["hazard_profile"],
        powered_time_s=np.asarray(trajectory["mission"]["powered_time_s"]),
        switching_point=trajectory["switching_point"],
        goal_position=geometry["goal_position"],
    )
    figure = _create_figure(
        result, stage_label=f"Multi-terrain: {scenario_id.replace('_', ' ')}",
    )
    figure.savefig(paths["png"], dpi=230, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SUPPORTED_SCENARIOS, required=True)
    args = parser.parse_args()
    configuration = build_scenario_configuration(args.scenario)
    paths = output_paths(args.scenario)
    result = solve_stackelberg_game(
        configuration, cache_dir=paths["cache_dir"], progress_callback=_progress,
    )
    if not result["status"]["success"]:
        raise RuntimeError(result["status"]["message"])
    save_result(result, args.scenario)
    for key in ("summary", "npz", "png", "pdf"):
        print(paths[key], flush=True)


if __name__ == "__main__":
    main()
