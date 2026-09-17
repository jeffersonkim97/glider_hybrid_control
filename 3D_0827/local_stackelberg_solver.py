"""Exact-follower production wrapper for the Stage 14.2B local-SSE search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Mapping, Sequence

from attacker_best_response import (
    AttackerBestResponseRun,
    CandidateMissionResult,
    attacker_response_from_result,
    run_attacker_best_response,
)
from bellman_state import BellmanStateGrid
from local_sse_contract import (
    DefenderGridTopology,
    DefenderNeighborhoodConfig,
    FollowerPayoffRecord,
    LocalDefenderEvaluation,
    audit_strong_follower_selection,
)
from local_sse_search import LocalSSESearchResult, run_local_sse_search
from map_geometry import TerrainModel
from sparse_reachability import build_goal_backward_reachable_graph
from stackelberg_solver import (
    DefenderCandidate,
    DefenderEvaluation,
    FiniteStackelbergRun,
    StackelbergTiming,
    select_geometry_sse_follower,
)
from stackelberg_validation import validate_selected_stackelberg_trajectory
from stage11_config import Stage11Config


@dataclass(frozen=True)
class ExactLocalEvaluationRecord:
    action_id: int
    sensor_position_map: tuple[float, float, float]
    local_evaluation: LocalDefenderEvaluation
    attacker_cooptimal_candidate_ids: tuple[int, ...]
    selected_candidate_id: int | None
    attacker_br_runtime_s: float
    exact_exhaustive_minimum_verified: bool
    strong_follower_tie_audit_passed: bool
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactLocalStackelbergRun:
    search_result: LocalSSESearchResult
    exact_evaluation_records: tuple[ExactLocalEvaluationRecord, ...]
    detailed_evaluations: tuple[DefenderEvaluation, ...]
    final_trajectory_identity: dict[str, Any] | None
    independent_replay_report: dict[str, Any] | None
    shared_graph_build_s: float

    def compact_dict(self) -> dict[str, Any]:
        return {
            "search_result": self.search_result.as_dict(),
            "exact_evaluation_records": [
                record.as_dict() for record in self.exact_evaluation_records
            ],
            "final_trajectory_identity": self.final_trajectory_identity,
            "independent_replay_report": self.independent_replay_report,
            "shared_graph_build_s": self.shared_graph_build_s,
        }


def _trajectory_identity(evaluation: DefenderEvaluation) -> dict[str, Any]:
    selected = evaluation.sse_follower_result
    if selected is None or selected.selected_option is None:
        raise ValueError("final local SSE has no selected trajectory")
    option = selected.selected_option
    state_ids = [option.connection.target_state_id]
    state_ids.extend(edge.target_id for edge in option.glide_edges)
    payload: dict[str, Any] = {
        "switching_position_map": selected.position_map.tolist(),
        "virtual_target_state_id": option.connection.target_state_id,
        "bellman_state_ids": state_ids,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _audit_exact_follower(
    attacker_run: AttackerBestResponseRun,
    selected: CandidateMissionResult,
    *,
    objective_tolerance: float,
) -> tuple[bool, bool]:
    feasible = tuple(item for item in attacker_run.candidate_results if item.feasible)
    minimum = min(float(item.objective) for item in feasible)
    cooptimal = tuple(sorted(
        item.candidate_id for item in feasible
        if float(item.objective) <= minimum + objective_tolerance
    ))
    exact_verified = bool(
        cooptimal == tuple(attacker_run.cooptimal_candidate_ids)
        and selected.candidate_id in cooptimal
    )
    tie_audit = audit_strong_follower_selection(
        tuple(
            FollowerPayoffRecord(
                response_id=item.candidate_id,
                attacker_objective=float(item.objective),
                defender_payoff=float(item.detection_probability),
            )
            for item in feasible
        ),
        selected_response_id=selected.candidate_id,
        objective_tolerance=objective_tolerance,
    )
    return exact_verified, tie_audit.passed


def run_exact_local_stackelberg(
    defender_candidates: Sequence[DefenderCandidate],
    initial_defender_action_id: int,
    topology: DefenderGridTopology,
    configuration: DefenderNeighborhoodConfig,
    scenario,
    *,
    terrain: TerrainModel,
    stage_config: Stage11Config,
    attacker_kwargs: Mapping[str, object],
    follower_objective_tolerance: float = 1.0e-12,
    leader_payoff_tolerance: float = 1.0e-12,
    reuse_reachability_graph: bool = True,
) -> ExactLocalStackelbergRun:
    """Evaluate only actions requested by local search, with exact finite BRs."""
    candidates = tuple(defender_candidates)
    candidate_by_id = {candidate.action_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Defender action IDs must be unique")
    if set(candidate_by_id) != set(topology.action_ids):
        raise ValueError("topology must contain exactly the configured Defender actions")
    kwargs = dict(attacker_kwargs)
    wrapper_started = perf_counter()
    graph_build_s = 0.0
    if reuse_reachability_graph:
        grid = kwargs.get("bellman_grid")
        if not isinstance(grid, BellmanStateGrid):
            raise TypeError("exact local graph reuse requires a BellmanStateGrid")
        graph_started = perf_counter()
        kwargs["prebuilt_reachability"] = build_goal_backward_reachable_graph(
            grid,
            terrain,
            scenario.goal,
            parameters=kwargs["parameters"],
            physical_scale=kwargs["physical_scale"],
        )
        graph_build_s = perf_counter() - graph_started
    detailed: dict[int, DefenderEvaluation] = {}
    records: dict[int, ExactLocalEvaluationRecord] = {}

    def evaluator(action_id: int) -> LocalDefenderEvaluation:
        candidate = candidate_by_id[action_id]
        started = perf_counter()
        attacker_run = run_attacker_best_response(
            candidate.action,
            scenario,
            terrain=terrain,
            objective_tolerance=follower_objective_tolerance,
            **kwargs,
        )
        runtime = perf_counter() - started
        try:
            selected, outcome = select_geometry_sse_follower(
                attacker_run,
                objective_tolerance=follower_objective_tolerance,
            )
        except ValueError as error:
            evaluation = DefenderEvaluation(
                candidate=candidate,
                attacker_run=attacker_run,
                sse_follower_result=None,
                outcome=None,
                attacker_br_runtime_s=runtime,
                infeasibility_reason=str(error),
            )
            local = LocalDefenderEvaluation(
                action_id=action_id,
                status="model_infeasible",
                defender_value=None,
                attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=True,
                strong_tie_break_verified=False,
                diagnostic=str(error),
            )
            exact_verified = True
            strong_verified = False
            selected_id = None
        else:
            exact_verified, strong_verified = _audit_exact_follower(
                attacker_run,
                selected,
                objective_tolerance=follower_objective_tolerance,
            )
            evaluation = DefenderEvaluation(
                candidate=candidate,
                attacker_run=attacker_run,
                sse_follower_result=selected,
                outcome=outcome,
                attacker_br_runtime_s=runtime,
            )
            local = LocalDefenderEvaluation(
                action_id=action_id,
                status="feasible",
                defender_value=float(outcome.defender_payoff),
                attacker_objective=float(outcome.attacker_payoff),
                selected_attacker_response_id=selected.candidate_id,
                exact_attacker_best_response_verified=exact_verified,
                strong_tie_break_verified=strong_verified,
            )
            selected_id = selected.candidate_id
        detailed[action_id] = evaluation
        records[action_id] = ExactLocalEvaluationRecord(
            action_id=action_id,
            sensor_position_map=tuple(float(value) for value in candidate.action.sensor_position_map),
            local_evaluation=local,
            attacker_cooptimal_candidate_ids=tuple(attacker_run.cooptimal_candidate_ids),
            selected_candidate_id=selected_id,
            attacker_br_runtime_s=runtime,
            exact_exhaustive_minimum_verified=exact_verified,
            strong_follower_tie_audit_passed=strong_verified,
            candidate_count=len(attacker_run.candidate_results),
        )
        return local

    search = run_local_sse_search(
        initial_defender_action_id,
        topology,
        evaluator,
        configuration,
        leader_payoff_tolerance=leader_payoff_tolerance,
    )
    replay_report: dict[str, Any] | None = None
    trajectory_identity: dict[str, Any] | None = None
    validation_s = 0.0
    if search.local_sse_verified and search.final_local_sse_action_id is not None:
        final_id = search.final_local_sse_action_id
        selected_evaluation = detailed[final_id]
        evaluated_order = tuple(search.evaluated_defender_actions)
        validation_run = FiniteStackelbergRun(
            defender_candidates=tuple(candidate_by_id[action_id] for action_id in evaluated_order),
            evaluations=tuple(detailed[action_id] for action_id in evaluated_order),
            selected_evaluation=selected_evaluation,
            timing=StackelbergTiming(
                per_action_s=tuple(
                    detailed[action_id].attacker_br_runtime_s for action_id in evaluated_order
                ),
                total_s=perf_counter() - wrapper_started,
                shared_graph_build_s=graph_build_s,
            ),
            follower_objective_tolerance=follower_objective_tolerance,
            leader_payoff_tolerance=leader_payoff_tolerance,
            equilibrium_scope=search.solution_scope,
        )
        validation_started = perf_counter()
        audit = validate_selected_stackelberg_trajectory(validation_run, stage_config)
        validation_s = perf_counter() - validation_started
        replay_report = asdict(audit.report)
        replay_report["passed"] = bool(audit.report.passed)
        trajectory_identity = _trajectory_identity(selected_evaluation)
    total_wrapper_s = perf_counter() - wrapper_started
    evaluated_details = tuple(
        detailed[action_id] for action_id in search.evaluated_defender_actions
    )
    los_s = sum(item.attacker_run.metrics.timing.los_s for item in evaluated_details)
    switch_s = sum(
        item.attacker_run.metrics.timing.candidate_generation_s
        + item.attacker_run.metrics.timing.energy_filter_s
        + item.attacker_run.metrics.timing.virtual_connection_s
        for item in evaluated_details
    )
    hazard_s = sum(
        item.attacker_run.metrics.timing.hazard_precompute_s for item in evaluated_details
    )
    bellman_s = sum(
        max(
            0.0,
            item.attacker_run.metrics.timing.bellman_solve_s
            - item.attacker_run.metrics.timing.hazard_precompute_s,
        )
        if item.attacker_run.metrics.bellman_backend == "goal_backward_sparse"
        else item.attacker_run.metrics.timing.bellman_solve_s
        for item in evaluated_details
    )
    evaluation_wall_s = sum(item.attacker_br_runtime_s for item in evaluated_details)
    named_inside_evaluation = los_s + switch_s + hazard_s + bellman_s
    attacker_residual_s = max(0.0, evaluation_wall_s - named_inside_evaluation)
    search_control_s = max(
        0.0,
        search.runtime_decomposition_s["total_local_search_s"]
        - search.runtime_decomposition_s["exact_defender_evaluation_s"],
    )
    base_accounted = (
        graph_build_s + los_s + switch_s + hazard_s + bellman_s
        + attacker_residual_s + search_control_s + validation_s
    )
    wrapper_residual_s = max(0.0, total_wrapper_s - base_accounted)
    accounted = base_accounted + wrapper_residual_s
    runtime = {
        "sparse_bellman_timer_semantics": "hazard-exclusive-v1",
        "T_local_s": total_wrapper_s,
        "T_graph_s": graph_build_s,
        "T_LOS_s": los_s,
        "T_switch_s": switch_s,
        "T_hazard_s": hazard_s,
        "T_Bellman_s": bellman_s,
        "T_attacker_residual_s": attacker_residual_s,
        "T_local_search_control_s": search_control_s,
        "T_validation_s": validation_s,
        "T_wrapper_residual_s": wrapper_residual_s,
        "T_accounted_s": accounted,
        "T_reconciliation_error_s": total_wrapper_s - accounted,
    }
    search = replace(
        search,
        independent_replay_status=(
            None if replay_report is None else bool(replay_report["passed"])
        ),
        runtime_decomposition_s=runtime,
    )
    return ExactLocalStackelbergRun(
        search_result=search,
        exact_evaluation_records=tuple(
            records[action_id] for action_id in search.evaluated_defender_actions
        ),
        detailed_evaluations=evaluated_details,
        final_trajectory_identity=trajectory_identity,
        independent_replay_report=replay_report,
        shared_graph_build_s=graph_build_s,
    )


__all__ = [
    "ExactLocalEvaluationRecord", "ExactLocalStackelbergRun",
    "run_exact_local_stackelberg",
]
