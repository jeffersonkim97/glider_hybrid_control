"""Regenerate standard plots for the P2 multi-terrain successor-grid cases.

This runner re-evaluates the already-selected P2 Stackelberg sensor candidate
with the authoritative ``successor_grid_physical_edge`` follower, exports the
fresh computational bundles, and regenerates figures 1--7 from those exports.

Run from the repository root, for example::

    python -m p1b_4D.regenerate_successor_scenario_results two_hill
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    B3_TERRAIN_SPECIFICATIONS,
)
from p1b_4D.experiment_multiterrain_baselines import build_terrain_config
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.projection import construct_projected_cost_map
from p1b_4D.result_export import export_all_results
from p1b_4D.result_import import import_result_collection
from p1b_4D.stackelberg_solver import solve_stackelberg_game
from p1b_4D.visualization import generate_project_visualizations


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "two_hill": {
        "output_root": REPO_ROOT / "results_two_hill",
        "z_sensor": 1500.0,
        "selection_role": "representative_fixed_sensor",
        "selection_source": "user_selected_two_hill_visual_example",
    },
    "goal_in_valley": {
        "output_root": REPO_ROOT / "results_goal_in_valley",
        "z_sensor": B3_TERRAIN_SPECIFICATIONS["goal_in_valley"][
            "sensor_candidates"
        ]["stackelberg"],
        "selection_role": "fixed_p2_stackelberg_candidate",
        "selection_source": (
            "results/multiterrain_strategic_baselines_refined/"
            "multiterrain_baseline_results.json"
        ),
    },
}


def regenerate_scenario(scenario_name: str) -> dict[str, Any]:
    """Recompute one successor-grid response and replace its standard outputs."""
    specification = SCENARIOS[scenario_name]
    output_root = Path(specification["output_root"])
    selected_z = float(specification["z_sensor"])
    configuration = build_terrain_config(scenario_name, output_root)
    configs = configuration["primary_result"]
    configs["sensor_config"]["default_z_sensor"] = selected_z
    configs["attacker_solver_config"][
        "transition_model"
    ] = "successor_grid_physical_edge"
    logger = configs["logging_utilities"]["logger"]

    def replay_selected_candidate(_evaluate, bounds, _options):
        if not bounds[0] <= selected_z <= bounds[1]:
            raise ValueError(
                f"Selected P2 sensor z={selected_z} lies outside {bounds}"
            )
        return {
            "z_sensor": selected_z,
            "converged": True,
            "metadata": {
                "algorithm": "fixed_sensor_candidate_replay",
                "selection_role": specification["selection_role"],
                "selection_source": specification["selection_source"],
                "outer_search_repeated": False,
                "fresh_successor_follower_solve": True,
            },
        }

    try:
        stackelberg = solve_stackelberg_game(
            configuration, optimizer=replay_selected_candidate
        )
        final_evaluation = stackelberg["primary_result"][
            "final_defender_evaluation"
        ]["primary_result"]
        attacker_pipeline = final_evaluation[
            "attacker_best_response_bundle"
        ]["primary_result"]
        final_configuration = attacker_pipeline["configuration_bundle"]
        geometry = attacker_pipeline["geometry_bundle"]
        detection = attacker_pipeline["detection_bundle"]
        stage_cost = attacker_pipeline["stage_cost_4d_bundle"]
        bellman = attacker_pipeline["bellman_candidate_bundle"]
        response = attacker_pipeline["bellman_response_bundle"]
        projected = construct_projected_cost_map(
            final_configuration, geometry, detection, stage_cost
        )
        exported = export_all_results(
            final_configuration,
            geometry,
            detection,
            stage_cost,
            projected,
            bellman,
            response,
            stackelberg,
        )
        manifest_path = exported["primary_result"]["master_manifest_path"]
        imported = import_result_collection(manifest_path)
        visualized = generate_project_visualizations(
            imported,
            output_root / "results" / "figures",
        )
        best = response["primary_result"]
        report = {
            "schema_name": "SuccessorScenarioRegeneration",
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario_name,
            "transition_model": final_configuration["primary_result"][
                "attacker_solver_config"
            ]["transition_model"],
            "selected_sensor_z": selected_z,
            "sensor_selection_role": specification["selection_role"],
            "outer_search_repeated": False,
            "fresh_successor_follower_solve": True,
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "mission_time": float(best["mission_time"]),
            "switching_point": [
                float(value) for value in best["switching_point"]
            ],
            "continuous_replay": best["continuous_replay_validation"],
            "export_status": exported["status"],
            "visualization_status": visualized["status"],
            "figure_count": len(
                visualized["primary_result"]["generated_figures"]
            ),
        }
        report_path = (
            output_root
            / "results"
            / "metadata"
            / "successor_scenario_regeneration.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report
    finally:
        close_phase_logger(logger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=tuple(SCENARIOS))
    args = parser.parse_args()
    report = regenerate_scenario(args.scenario)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
