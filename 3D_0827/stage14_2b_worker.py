"""Fresh-process canonical worker for Stage 14.2B local-SSE certification."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_sse_contract import DefenderGridTopology, DefenderNeighborhoodConfig
from local_stackelberg_solver import run_exact_local_stackelberg
from stage11_notebook_support import write_json
from stage14_benchmark_contract import (
    CANONICAL_DEFENDER_X_MAP,
    canonical_attacker_kwargs,
    canonical_stage14_config,
)
from stackelberg_solver import generate_defender_line_actions
from terrain_catalog import build_terrain


def run_canonical_local_sse() -> dict[str, object]:
    config = canonical_stage14_config()
    candidates = generate_defender_line_actions(CANONICAL_DEFENDER_X_MAP)
    topology = DefenderGridTopology.ordered_line(
        tuple(candidate.action_id for candidate in candidates)
    )
    neighborhood = DefenderNeighborhoodConfig(r_neighbor=1)
    run = run_exact_local_stackelberg(
        candidates,
        initial_defender_action_id=0,
        topology=topology,
        configuration=neighborhood,
        scenario=config.attacker_initial_condition,
        terrain=build_terrain(config.terrain_category),
        stage_config=config,
        attacker_kwargs=canonical_attacker_kwargs(config),
        reuse_reachability_graph=True,
    )
    payload = run.compact_dict()
    payload.update({
        "stage": "14.2B",
        "status": run.search_result.termination_status,
        "configuration": {
            "terrain_category": config.terrain_category,
            "initial_defender_action_id": 0,
            "defender_x_map": list(CANONICAL_DEFENDER_X_MAP),
            "neighborhood": neighborhood.as_metadata(),
            "attacker_response": "existing exact finite Attacker best response",
            "local_search": "single-start strict-improvement neighborhood ascent",
        },
        "state_and_search_size": {
            "N_D": len(candidates),
            "N_eval_local": run.search_result.unique_defender_evaluations,
            "K_local": run.search_result.local_search_iterations,
            "per_evaluated_action": [
                {
                    "action_id": evaluation.candidate.action_id,
                    "N_C": evaluation.attacker_run.metrics.number_of_candidates,
                    "N_S_active": evaluation.attacker_run.metrics.active_state_count,
                    "N_E": evaluation.attacker_run.metrics.hazard_edge_count,
                }
                for evaluation in run.detailed_evaluations
            ],
        },
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = run_canonical_local_sse()
    write_json(arguments.output, payload)
    result = payload["search_result"]
    print(
        f"Stage 14.2B: {result['termination_status']}; "
        f"D{result['initial_defender_action_id']} -> "
        f"D{result['final_local_sse_action_id']}"
    )


if __name__ == "__main__":
    main()
