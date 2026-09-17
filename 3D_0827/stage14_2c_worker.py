"""Fresh-process exact local-SSE/global-oracle worker for Stage 14.2C."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from discretization_config import discretization_from_physical_steps
from defender_grid_config import DEFENDER_ACTION_COUNT
from local_sse_contract import DefenderGridTopology, DefenderNeighborhoodConfig
from local_stackelberg_solver import run_exact_local_stackelberg
from stage11_notebook_support import write_json
from stage14_2c_contract import STAGE14_2C_SCHEMA_VERSION
from stage14_benchmark_contract import (
    CANONICAL_DEFENDER_X_MAP,
    canonical_attacker_kwargs,
    canonical_stage14_config,
    environment_manifest,
)
from stage14_profiling import decompose_exact_sse_timing
from stackelberg_solver import (
    DefenderEvaluation,
    generate_defender_line_actions,
    run_finite_stackelberg,
)
from stackelberg_validation import validate_selected_stackelberg_trajectory
from terrain_catalog import build_terrain


ROOT = Path(__file__).resolve().parent


def _build_case(configuration: dict[str, Any]):
    parameters = configuration["parameters"]
    if parameters["terrain_category"] != "centered_cube":
        raise ValueError("Stage 14.2C terrain must remain centered_cube")
    if parameters["approximate_planner"] or parameters["reinforcement_learning"]:
        raise ValueError("Stage 14.2C requires the exact non-RL solver")
    if parameters["multi_start"]:
        raise ValueError("Stage 14.2C defers multi-start")
    base = canonical_stage14_config()
    spatial = float(parameters["spatial_resolution_m"])
    heading = float(parameters["heading_spacing_deg"])
    discretization = discretization_from_physical_steps(
        spatial,
        spatial,
        heading,
        meters_per_map_unit=base.physical_scale.meters_per_map_unit,
        template=base.discretization,
    )
    discretization = replace(
        discretization,
        switching_contour_sample_count=int(
            parameters["switching_contour_sample_count"]
        ),
        switching_radial_sample_count=int(parameters["switching_radial_sample_count"]),
    )
    stage_config = replace(base, discretization=discretization)
    policy = str(parameters["defender_grid_policy"])
    count = int(parameters["defender_action_count"])
    if policy == "canonical_six":
        if count != DEFENDER_ACTION_COUNT:
            raise ValueError(
                f"canonical_six requires exactly {DEFENDER_ACTION_COUNT} actions"
            )
        defender_x = CANONICAL_DEFENDER_X_MAP
    elif policy == "uniform_fixed_domain":
        if count < 2:
            raise ValueError("uniform_fixed_domain requires at least two actions")
        defender_x = tuple(map(float, np.linspace(
            float(parameters["defender_domain_x_min"]),
            float(parameters["defender_domain_x_max"]),
            count,
        )))
    else:
        raise ValueError(f"unsupported Defender grid policy: {policy!r}")
    candidates = generate_defender_line_actions(defender_x)
    initial_id = int(parameters["initial_defender_action_id"])
    if initial_id not in {candidate.action_id for candidate in candidates}:
        raise ValueError("initial Defender action is outside the configured action set")
    return stage_config, candidates, initial_id


def _admissible_counts(grid, terrain) -> tuple[int, int]:
    x_values, y_values, z_values = np.meshgrid(
        grid.x_coordinates,
        grid.y_coordinates,
        grid.altitude_coordinates,
        indexing="xy",
    )
    positions = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    vectorized = getattr(terrain, "contains_solid_many", None)
    excluded = (
        np.asarray(vectorized(positions), dtype=bool)
        if callable(vectorized)
        else np.asarray([terrain.contains_solid(point) for point in positions], dtype=bool)
    )
    position_count = int(np.count_nonzero(~excluded))
    return position_count, position_count * grid.heading_bin_count


def _trajectory_identity(evaluation: DefenderEvaluation) -> dict[str, Any]:
    selected = evaluation.sse_follower_result
    if selected is None or selected.selected_option is None:
        raise ValueError("selected exact outcome has no trajectory")
    option = selected.selected_option
    state_ids = [option.connection.target_state_id]
    state_ids.extend(edge.target_id for edge in option.glide_edges)
    payload = {
        "switching_position_map": selected.position_map.tolist(),
        "virtual_target_state_id": option.connection.target_state_id,
        "bellman_state_ids": state_ids,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _size_record(
    evaluations: tuple[DefenderEvaluation, ...],
    *,
    total_defender_count: int,
    quadrature_resolution: int,
    terrain,
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("at least one exact Defender evaluation is required")
    final = evaluations[-1]
    representative = final.attacker_run
    grid = representative.graph.grid
    metrics = representative.metrics
    admissible_positions, admissible_states = _admissible_counts(grid, terrain)

    def candidate_funnel(evaluation: DefenderEvaluation) -> dict[str, Any]:
        results = evaluation.attacker_run.candidate_results
        positions = np.asarray(
            [result.position_map for result in results], dtype=float,
        )
        unique_positions = (
            0 if not len(positions)
            else int(len(np.unique(np.round(positions, decimals=12), axis=0)))
        )
        return {
            "N_C_raw": len(results),
            "N_C_unique": unique_positions,
            "N_C_powered_feasible": sum(
                result.powered_feasible for result in results
            ),
            "N_C_energy_feasible": sum(
                result.powered_feasible and result.energy_feasible
                for result in results
            ),
            "N_C_virtual_feasible": sum(
                result.powered_feasible
                and result.energy_feasible
                and result.virtual_feasible
                for result in results
            ),
            "N_C_goal_reachable": sum(
                result.powered_feasible
                and result.energy_feasible
                and result.virtual_feasible
                and result.goal_reachable
                for result in results
            ),
            "N_C_feasible": sum(result.feasible for result in results),
        }

    representative_funnel = candidate_funnel(final)
    return {
        "N_x": grid.x_count,
        "N_y": grid.y_count,
        "N_h": grid.altitude_count,
        "N_psi": grid.heading_bin_count,
        "N_S_cart": grid.state_count,
        "N_position_cart": grid.x_count * grid.y_count * grid.altitude_count,
        "N_position_admissible": admissible_positions,
        "N_S_admissible": admissible_states,
        "N_S_goal_reachable": metrics.goal_reachable_state_count,
        "N_S_active": metrics.active_state_count,
        "N_E": metrics.hazard_edge_count,
        "N_E_goal_reachable": representative.graph.statistics.valid_edge_count,
        "N_C": metrics.number_of_candidates,
        **representative_funnel,
        "N_C_unique_definition": (
            "unique physical switching positions after rounding map coordinates "
            "to 12 decimal places; diagnostic count only, no solver deduplication"
        ),
        "N_D": total_defender_count,
        "N_eval": len(evaluations),
        "B": len(grid.motion_offsets),
        "B_effective": (
            metrics.hazard_edge_count / metrics.active_state_count
            if metrics.active_state_count else 0.0
        ),
        "Q": quadrature_resolution,
        "per_evaluated_action": [
            {
                "action_id": evaluation.candidate.action_id,
                "N_C": evaluation.attacker_run.metrics.number_of_candidates,
                **candidate_funnel(evaluation),
                "N_S_active": evaluation.attacker_run.metrics.active_state_count,
                "N_E": evaluation.attacker_run.metrics.hazard_edge_count,
            }
            for evaluation in evaluations
        ],
    }


def _solution_identity(
    evaluation: DefenderEvaluation,
    *,
    attacker_objective: float,
    defender_payoff: float,
    equilibrium_scope: str,
    leader_cooptimal_action_ids: list[int] | None,
) -> dict[str, Any]:
    selected = evaluation.sse_follower_result
    if selected is None:
        raise ValueError("selected exact outcome has no follower")
    return {
        "feasible": True,
        "selected_defender_action_id": evaluation.candidate.action_id,
        "selected_sensor_position_map": evaluation.candidate.action.sensor_position_map.tolist(),
        "selected_attacker_candidate_id": selected.candidate_id,
        "sse_selected_follower_candidate_id": selected.candidate_id,
        "attacker_objective_cooptimal_candidate_ids": list(
            evaluation.attacker_run.cooptimal_candidate_ids
        ),
        "leader_cooptimal_action_ids": leader_cooptimal_action_ids,
        "attacker_objective": attacker_objective,
        "defender_objective_pod": defender_payoff,
        "mission_time_s": selected.mission_time_s,
        "cumulative_hazard": selected.cumulative_hazard,
        "detection_probability": selected.detection_probability,
        "trajectory_identity": _trajectory_identity(evaluation),
        "equilibrium_scope": equilibrium_scope,
        "tie_break_convention": {
            "follower": (
                "maximize Defender PoD among Attacker-objective co-optima, "
                "then lowest candidate ID"
            ),
            "leader": (
                "stable configured action order" if leader_cooptimal_action_ids is not None
                else "strict-improvement local search with deterministic topology order"
            ),
        },
    }


def _local_payload(configuration: dict[str, Any]) -> dict[str, Any]:
    stage_config, candidates, initial_id = _build_case(configuration)
    terrain = build_terrain("centered_cube")
    topology = DefenderGridTopology.ordered_line(tuple(
        candidate.action_id for candidate in candidates
    ))
    neighborhood = DefenderNeighborhoodConfig(
        r_neighbor=int(configuration["parameters"]["r_neighbor"])
    )
    run = run_exact_local_stackelberg(
        candidates,
        initial_defender_action_id=initial_id,
        topology=topology,
        configuration=neighborhood,
        scenario=stage_config.attacker_initial_condition,
        terrain=terrain,
        stage_config=stage_config,
        attacker_kwargs=canonical_attacker_kwargs(stage_config),
        reuse_reachability_graph=True,
    )
    search = run.search_result
    if not search.local_sse_verified or search.final_local_sse_action_id is None:
        return {
            "schema_version": STAGE14_2C_SCHEMA_VERSION,
            "status": search.termination_status,
            "failure_type": search.termination_status,
            "failure_message": "local SSE was not certified",
            "algorithm_variant": "local_sse",
            "local_search": search.as_dict(),
        }
    evaluation_by_id = {
        evaluation.candidate.action_id: evaluation for evaluation in run.detailed_evaluations
    }
    final = evaluation_by_id[search.final_local_sse_action_id]
    totals = dict(search.runtime_decomposition_s)
    totals["T_SSE_s"] = totals["T_local_s"]
    totals["T_attacker_BR_s"] = sum(
        evaluation.attacker_br_runtime_s for evaluation in run.detailed_evaluations
    )
    size_evaluations = tuple(
        evaluation for evaluation in run.detailed_evaluations
        if evaluation.candidate.action_id != search.final_local_sse_action_id
    ) + (final,)
    sizes = _size_record(
        size_evaluations,
        total_defender_count=len(candidates),
        quadrature_resolution=stage_config.discretization.hazard_quadrature_resolution,
        terrain=terrain,
    )
    neighborhood_counts = search.neighbor_count_per_iteration
    feasible_counts = search.feasible_neighbor_count_per_iteration
    local_record = search.as_dict()
    local_record.update({
        "mean_neighborhood_size": float(np.mean(neighborhood_counts)),
        "max_neighborhood_size": max(neighborhood_counts),
        "feasible_neighbor_comparisons": int(sum(feasible_counts)),
        "unique_feasible_neighbor_evaluations": max(
            0, search.unique_defender_evaluations - 1
        ),
        "search_path_length": len(search.visited_defender_actions),
        "exact_evaluation_records": [
            record.as_dict() for record in run.exact_evaluation_records
        ],
    })
    return {
        "schema_version": STAGE14_2C_SCHEMA_VERSION,
        "status": "completed",
        "algorithm_variant": "local_sse",
        "terrain_category": "centered_cube",
        "timing": {"schema_version": STAGE14_2C_SCHEMA_VERSION, "totals": totals},
        "state_and_game_size": sizes,
        "solution_identity": _solution_identity(
            final,
            attacker_objective=float(search.final_J_A),
            defender_payoff=float(search.final_J_D),
            equilibrium_scope=search.solution_scope,
            leader_cooptimal_action_ids=None,
        ),
        "independent_replay": run.independent_replay_report,
        "local_search": local_record,
        "exactness": {
            "all_evaluated_attacker_responses_exact": search.attacker_exactness_verified,
            "strong_follower_tie_break_verified": search.strong_tie_break_verified,
            "all_required_final_neighbors_evaluated": set(
                search.iterations[-1].neighbor_action_ids
            ) <= set(search.evaluated_defender_actions),
            "local_sse_verified": search.local_sse_verified,
        },
    }


def _global_payload(configuration: dict[str, Any]) -> dict[str, Any]:
    stage_config, candidates, _ = _build_case(configuration)
    terrain = build_terrain("centered_cube")
    try:
        run = run_finite_stackelberg(
            candidates,
            stage_config.attacker_initial_condition,
            terrain=terrain,
            attacker_kwargs=canonical_attacker_kwargs(stage_config),
            reuse_reachability_graph=True,
        )
    except RuntimeError as error:
        if str(error) != "no Defender action has a feasible Attacker response":
            raise
        return {
            "schema_version": STAGE14_2C_SCHEMA_VERSION,
            "status": "model_infeasible",
            "failure_type": "model_infeasible",
            "failure_message": str(error),
            "algorithm_variant": "global_oracle",
            "terrain_category": "centered_cube",
            "oracle_metadata": {
                "role": "tractable finite global oracle/reference only",
                "all_defender_actions_model_infeasible": True,
            },
        }
    validation_started = perf_counter()
    audit = validate_selected_stackelberg_trajectory(run, stage_config)
    validation_s = perf_counter() - validation_started
    payoffs = [
        float(evaluation.outcome.defender_payoff)
        for evaluation in run.evaluations if evaluation.outcome is not None
    ]
    maximum = max(payoffs)
    leader_cooptimal = [
        evaluation.candidate.action_id for evaluation in run.evaluations
        if evaluation.outcome is not None
        and float(evaluation.outcome.defender_payoff) >= maximum - run.leader_payoff_tolerance
    ]
    attacker_exact = all(
        evaluation.outcome is None
        or np.isclose(
            float(evaluation.outcome.attacker_payoff),
            min(
                float(candidate.objective)
                for candidate in evaluation.attacker_run.candidate_results
                if candidate.feasible
            ),
            rtol=0.0,
            atol=run.follower_objective_tolerance,
        )
        for evaluation in run.evaluations
    )
    timing = decompose_exact_sse_timing(run, validation_time_s=validation_s)
    timing["totals"]["T_attacker_BR_s"] = sum(
        evaluation.attacker_br_runtime_s for evaluation in run.evaluations
    )
    size_evaluations = tuple(
        evaluation for evaluation in run.evaluations
        if evaluation.candidate.action_id != run.selected_evaluation.candidate.action_id
    ) + (run.selected_evaluation,)
    sizes = _size_record(
        size_evaluations,
        total_defender_count=len(candidates),
        quadrature_resolution=stage_config.discretization.hazard_quadrature_resolution,
        terrain=terrain,
    )
    per_action_records = [
        {
            "action_id": evaluation.candidate.action_id,
            "sensor_position_map": evaluation.candidate.action.sensor_position_map.tolist(),
            "attacker_br_runtime_s": evaluation.attacker_br_runtime_s,
            "status": (
                "feasible" if evaluation.outcome is not None else "model_infeasible"
            ),
            "attacker_objective": (
                None if evaluation.outcome is None
                else float(evaluation.outcome.attacker_payoff)
            ),
            "defender_value": (
                None if evaluation.outcome is None
                else float(evaluation.outcome.defender_payoff)
            ),
            "selected_attacker_response_id": (
                None if evaluation.sse_follower_result is None
                else evaluation.sse_follower_result.candidate_id
            ),
            "attacker_cooptimal_candidate_ids": list(
                evaluation.attacker_run.cooptimal_candidate_ids
            ),
        }
        for evaluation in run.evaluations
    ]
    return {
        "schema_version": STAGE14_2C_SCHEMA_VERSION,
        "status": "completed",
        "algorithm_variant": "global_oracle",
        "terrain_category": "centered_cube",
        "timing": timing,
        "state_and_game_size": sizes,
        "solution_identity": _solution_identity(
            run.selected_evaluation,
            attacker_objective=float(run.outcome.attacker_payoff),
            defender_payoff=float(run.outcome.defender_payoff),
            equilibrium_scope=run.equilibrium_scope,
            leader_cooptimal_action_ids=leader_cooptimal,
        ),
        "independent_replay": {"passed": bool(audit.report.passed)},
        "local_search": None,
        "exactness": {
            "all_evaluated_attacker_responses_exact": attacker_exact,
            "strong_follower_tie_break_verified": True,
            "global_defender_enumeration_complete": len(run.evaluations) == len(candidates),
            "independent_replay_passed": bool(audit.report.passed),
        },
        "oracle_metadata": {
            "role": "tractable finite global oracle/reference only",
            "global_optimality_scope": run.equilibrium_scope,
            "per_action_evaluation_records": per_action_records,
        },
    }


def run_worker_case(configuration: dict[str, Any]) -> dict[str, Any]:
    mode = str(configuration["worker_mode"])
    if mode == "local_sse":
        payload = _local_payload(configuration)
    elif mode == "global_oracle":
        payload = _global_payload(configuration)
    else:
        raise ValueError(f"unsupported Stage-14.2C worker mode: {mode!r}")
    environment = environment_manifest(ROOT.parent)
    payload["solver_source_fingerprint"] = environment["software_revision"][
        "solver_source"
    ]["aggregate_sha256"]
    payload["stage14_2c_case_id"] = configuration["case_id"]
    payload["stage14_2c_repetition_id"] = configuration["repetition_id"]
    payload["complete_configuration"] = configuration
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text(encoding="utf-8"))
    payload = run_worker_case(configuration)
    write_json(arguments.output, payload)
    print(f"{configuration['repetition_id']}: {payload['status']}")


if __name__ == "__main__":
    main()
