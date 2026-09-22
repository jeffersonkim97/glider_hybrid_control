"""Single-start local-SSE search over an explicit Defender grid topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np

from local_sse_contract import (
    LOCAL_SSE_SCOPE,
    DefenderGridTopology,
    DefenderNeighborhoodConfig,
    LocalDefenderEvaluation,
    LocalSSEVerification,
    verify_local_sse,
)


LOCAL_SEARCH_SCHEMA_VERSION = "stage14.2B-v2"
DefenderEvaluator = Callable[[int], LocalDefenderEvaluation]


@dataclass(frozen=True)
class LocalSearchIteration:
    iteration: int
    current_action_id: int
    current_defender_value: float
    neighbor_action_ids: tuple[int, ...]
    feasible_neighbor_ids: tuple[int, ...]
    infeasible_neighbor_ids: tuple[int, ...]
    unknown_neighbor_ids: tuple[int, ...]
    chosen_next_action_id: int | None
    payoff_improvement: float
    local_verification: LocalSSEVerification

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalSSESearchResult:
    initial_defender_action_id: int
    final_local_sse_action_id: int | None
    final_attacker_response_id: int | None
    final_J_A: float | None
    final_J_D: float | None
    visited_defender_actions: tuple[int, ...]
    evaluated_defender_actions: tuple[int, ...]
    unique_defender_evaluations: int
    cached_evaluation_reuses: int
    initialization_recovery_attempted: bool
    initialization_neighbor_action_ids: tuple[int, ...]
    initialization_feasible_neighbor_ids: tuple[int, ...]
    initialization_infeasible_neighbor_ids: tuple[int, ...]
    initialization_unknown_neighbor_ids: tuple[int, ...]
    initialization_recovery_action_id: int | None
    local_search_iterations: int
    neighbor_count_per_iteration: tuple[int, ...]
    feasible_neighbor_count_per_iteration: tuple[int, ...]
    infeasible_neighbor_diagnostics: tuple[dict[str, Any], ...]
    unknown_neighbor_diagnostics: tuple[dict[str, Any], ...]
    payoff_improvement_per_iteration: tuple[float, ...]
    local_optimality_margin: float | None
    termination_status: str
    local_sse_verified: bool
    attacker_exactness_verified: bool
    strong_tie_break_verified: bool
    independent_replay_status: bool | None
    isolated_feasible_local_solution: bool
    initialization_dependent_solution_concept: bool
    global_optimality_evaluated: bool
    global_optimal: bool | None
    solution_scope: str
    neighborhood_metadata: dict[str, Any]
    iterations: tuple[LocalSearchIteration, ...]
    runtime_decomposition_s: dict[str, float]
    peak_memory: dict[str, Any] | None = None
    exact_attacker_verification_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failure_evaluation(action_id: int, error: BaseException) -> LocalDefenderEvaluation:
    if isinstance(error, TimeoutError):
        status = "timeout"
    elif isinstance(error, MemoryError):
        status = "memory_limit"
    else:
        status = "numerical_failure"
    return LocalDefenderEvaluation(
        action_id=action_id,
        status=status,
        defender_value=None,
        attacker_objective=None,
        selected_attacker_response_id=None,
        exact_attacker_best_response_verified=False,
        strong_tie_break_verified=False,
        diagnostic=f"{type(error).__name__}: {error}",
    )


def run_local_sse_search(
    initial_defender_action_id: int,
    topology: DefenderGridTopology,
    evaluator: DefenderEvaluator,
    configuration: DefenderNeighborhoodConfig = DefenderNeighborhoodConfig(),
    *,
    leader_payoff_tolerance: float = 1.0e-12,
    require_exact_attacker_verification: bool = True,
) -> LocalSSESearchResult:
    """Climb by strict improvement until all required neighbors certify local SSE.

    With ``require_exact_attacker_verification`` left at ``True`` the search only
    terminates on a certified local SSE, which is what the exact solver needs.
    An approximate Attacker solver sets it to ``False``: it still reports its own
    responses honestly as inexact, and the result still carries
    ``local_sse_verified=False``, but the climb is no longer blocked by a standard
    no approximation can meet.  Genuine unknowns - a crashed, timed-out or
    out-of-memory evaluation - keep blocking in both modes.
    """
    tolerance = float(leader_payoff_tolerance)
    require_exact = bool(require_exact_attacker_verification)

    def infeasibility_known(evaluation: LocalDefenderEvaluation) -> bool:
        return evaluation.status == "model_infeasible" and (
            evaluation.exact_attacker_best_response_verified or not require_exact
        )

    def comparison_ready(evaluation: LocalDefenderEvaluation) -> bool:
        return bool(evaluation.feasible) and (
            not require_exact
            or (
                evaluation.exact_attacker_best_response_verified
                and evaluation.strong_tie_break_verified
            )
        )
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("leader_payoff_tolerance must be finite and nonnegative")
    if initial_defender_action_id not in topology.action_ids:
        raise KeyError("initial Defender action is absent from the topology")
    started = perf_counter()
    evaluation_s = 0.0
    neighborhood_s = 0.0
    verification_s = 0.0
    cache: dict[int, LocalDefenderEvaluation] = {}
    evaluation_order: list[int] = []
    cache_reuses = 0

    def get_evaluation(action_id: int) -> LocalDefenderEvaluation:
        nonlocal evaluation_s, cache_reuses
        if action_id in cache:
            cache_reuses += 1
            return cache[action_id]
        action_started = perf_counter()
        try:
            evaluation = evaluator(action_id)
            if not isinstance(evaluation, LocalDefenderEvaluation):
                raise TypeError("Defender evaluator must return LocalDefenderEvaluation")
            if evaluation.action_id != action_id:
                raise ValueError("Defender evaluator returned the wrong action ID")
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            evaluation = _failure_evaluation(action_id, error)
        evaluation_s += perf_counter() - action_started
        cache[action_id] = evaluation
        evaluation_order.append(action_id)
        return evaluation

    visited = [int(initial_defender_action_id)]
    iteration_records: list[LocalSearchIteration] = []
    current_id = int(initial_defender_action_id)
    current = get_evaluation(current_id)
    recovery_attempted = False
    recovery_neighbor_ids: tuple[int, ...] = ()
    recovery_feasible_ids: tuple[int, ...] = ()
    recovery_infeasible_ids: tuple[int, ...] = ()
    recovery_unknown_ids: tuple[int, ...] = ()
    recovery_action_id: int | None = None

    # A model-infeasible leader action has no V_D value, so ordinary payoff
    # ascent cannot start there.  Treat the configured neighborhood as a
    # feasibility-restoration phase instead of incorrectly declaring the
    # local problem infeasible after evaluating only the initial action.
    if infeasibility_known(current):
        recovery_attempted = True
        neighborhood_started = perf_counter()
        recovery_neighbor_ids = topology.neighbors(current_id, configuration)
        neighborhood_s += perf_counter() - neighborhood_started
        recovery_evaluations = tuple(
            get_evaluation(action_id) for action_id in recovery_neighbor_ids
        )
        recovery_feasible = tuple(
            evaluation for evaluation in recovery_evaluations
            if comparison_ready(evaluation)
        )
        recovery_feasible_ids = tuple(
            evaluation.action_id for evaluation in recovery_feasible
        )
        recovery_infeasible_ids = tuple(
            evaluation.action_id for evaluation in recovery_evaluations
            if infeasibility_known(evaluation)
        )
        classified_ids = set(recovery_feasible_ids) | set(recovery_infeasible_ids)
        recovery_unknown_ids = tuple(
            action_id for action_id in recovery_neighbor_ids
            if action_id not in classified_ids
        )
        if recovery_feasible:
            maximum = max(float(item.defender_value) for item in recovery_feasible)
            tied_ids = {
                item.action_id for item in recovery_feasible
                if float(item.defender_value) >= maximum - tolerance
            }
            recovery_action_id = next(
                action_id for action_id in topology.action_ids
                if action_id in tied_ids
            )
            current_id = recovery_action_id
            visited.append(current_id)
            current = cache[current_id]

    if not current.feasible:
        if recovery_attempted:
            status = (
                "initial_neighborhood_comparison_unknown"
                if recovery_unknown_ids
                else "initial_neighborhood_infeasible"
            )
        else:
            status = "initial_defender_evaluation_failure"
        total = perf_counter() - started
        infeasible_diagnostics = tuple(
            {
                "action_id": action_id,
                "status": evaluation.status,
                "diagnostic": evaluation.diagnostic,
            }
            for action_id, evaluation in cache.items()
            if infeasibility_known(evaluation)
        )
        unknown_diagnostics = tuple(
            {
                "action_id": action_id,
                "status": evaluation.status,
                "diagnostic": evaluation.diagnostic,
            }
            for action_id, evaluation in cache.items()
            if not infeasibility_known(evaluation)
        )
        return LocalSSESearchResult(
            initial_defender_action_id=current_id,
            final_local_sse_action_id=None,
            final_attacker_response_id=None,
            final_J_A=None,
            final_J_D=None,
            visited_defender_actions=tuple(visited),
            evaluated_defender_actions=tuple(evaluation_order),
            unique_defender_evaluations=len(cache),
            cached_evaluation_reuses=cache_reuses,
            initialization_recovery_attempted=recovery_attempted,
            initialization_neighbor_action_ids=recovery_neighbor_ids,
            initialization_feasible_neighbor_ids=recovery_feasible_ids,
            initialization_infeasible_neighbor_ids=recovery_infeasible_ids,
            initialization_unknown_neighbor_ids=recovery_unknown_ids,
            initialization_recovery_action_id=recovery_action_id,
            local_search_iterations=0,
            neighbor_count_per_iteration=(),
            feasible_neighbor_count_per_iteration=(),
            infeasible_neighbor_diagnostics=infeasible_diagnostics,
            unknown_neighbor_diagnostics=unknown_diagnostics,
            payoff_improvement_per_iteration=(),
            local_optimality_margin=None,
            termination_status=status,
            local_sse_verified=False,
            attacker_exactness_verified=False,
            strong_tie_break_verified=False,
            independent_replay_status=None,
            isolated_feasible_local_solution=False,
            initialization_dependent_solution_concept=True,
            global_optimality_evaluated=False,
            global_optimal=None,
            solution_scope=LOCAL_SSE_SCOPE,
            neighborhood_metadata={
                "configuration": configuration.as_metadata(),
                "topology": topology.as_metadata(),
            },
            iterations=(),
            runtime_decomposition_s={
                "total_local_search_s": total,
                "exact_defender_evaluation_s": evaluation_s,
                "neighborhood_enumeration_s": neighborhood_s,
                "local_verification_s": 0.0,
                "search_control_residual_s": max(
                    0.0, total - evaluation_s - neighborhood_s
                ),
            },
            exact_attacker_verification_required=require_exact,
        )

    termination_status = "uncertified"
    final_verification: LocalSSEVerification | None = None
    while True:
        neighborhood_started = perf_counter()
        neighbor_ids = topology.neighbors(current_id, configuration)
        neighborhood_s += perf_counter() - neighborhood_started
        for neighbor_id in neighbor_ids:
            get_evaluation(neighbor_id)
        verification_started = perf_counter()
        verification = verify_local_sse(
            current_id,
            cache,
            topology,
            configuration,
            leader_payoff_tolerance=tolerance,
            require_exact_attacker_verification=require_exact,
        )
        verification_s += perf_counter() - verification_started
        current = cache[current_id]
        blocked = bool(verification.unknown_neighbor_diagnostics) or (
            require_exact
            and (
                not verification.exact_attacker_responses_verified
                or not verification.strong_tie_breaking_verified
            )
        )
        if blocked:
            chosen = None
            improvement = 0.0
            termination_status = "required_neighbor_comparison_unknown"
        else:
            improving = [
                cache[action_id] for action_id in verification.improving_neighbor_ids
            ]
            if improving:
                maximum = max(float(item.defender_value) for item in improving)
                tied_ids = {
                    item.action_id for item in improving
                    if float(item.defender_value) >= maximum - tolerance
                }
                chosen = next(action_id for action_id in topology.action_ids if action_id in tied_ids)
                improvement = float(cache[chosen].defender_value) - float(current.defender_value)
                termination_status = "improving"
            else:
                chosen = None
                improvement = 0.0
                termination_status = (
                    "isolated_feasible_local_sse"
                    if verification.isolated_feasible_local_solution
                    else (
                        "certified_local_sse" if require_exact
                        else "approximate_local_sse"
                    )
                )
        iteration_records.append(LocalSearchIteration(
            iteration=len(iteration_records),
            current_action_id=current_id,
            current_defender_value=float(current.defender_value),
            neighbor_action_ids=neighbor_ids,
            feasible_neighbor_ids=verification.feasible_neighbor_ids,
            infeasible_neighbor_ids=tuple(
                action_id for action_id, _ in verification.infeasible_neighbor_diagnostics
            ),
            unknown_neighbor_ids=tuple(
                action_id for action_id, _ in verification.unknown_neighbor_diagnostics
            ),
            chosen_next_action_id=chosen,
            payoff_improvement=improvement,
            local_verification=verification,
        ))
        if chosen is None:
            final_verification = verification
            break
        current_id = chosen
        visited.append(current_id)
        current = get_evaluation(current_id)

    final = cache[current_id]
    strictly_certified = bool(
        final_verification is not None and final_verification.local_sse_verified
    )
    # What the search is willing to return.  ``local_sse_verified`` below still
    # reports the strict answer, so an approximate run never claims certification.
    certified = strictly_certified if require_exact else bool(
        final_verification is not None
        and final_verification.local_optimum_under_supplied_evaluations
    )
    total = perf_counter() - started
    accounted = evaluation_s + neighborhood_s + verification_s
    infeasible_diagnostics = tuple(
        {
            "action_id": action_id,
            "status": evaluation.status,
            "diagnostic": evaluation.diagnostic,
        }
        for action_id, evaluation in cache.items()
        if evaluation.status == "model_infeasible"
    )
    unknown_diagnostics = tuple(
        {
            "action_id": action_id,
            "status": evaluation.status,
            "diagnostic": evaluation.diagnostic,
        }
        for action_id, evaluation in cache.items()
        if evaluation.status not in {"feasible", "model_infeasible"}
    )
    return LocalSSESearchResult(
        initial_defender_action_id=int(initial_defender_action_id),
        final_local_sse_action_id=current_id if certified else None,
        final_attacker_response_id=(
            final.selected_attacker_response_id if certified else None
        ),
        final_J_A=float(final.attacker_objective) if certified else None,
        final_J_D=float(final.defender_value) if certified else None,
        visited_defender_actions=tuple(visited),
        evaluated_defender_actions=tuple(evaluation_order),
        unique_defender_evaluations=len(cache),
        cached_evaluation_reuses=cache_reuses,
        initialization_recovery_attempted=recovery_attempted,
        initialization_neighbor_action_ids=recovery_neighbor_ids,
        initialization_feasible_neighbor_ids=recovery_feasible_ids,
        initialization_infeasible_neighbor_ids=recovery_infeasible_ids,
        initialization_unknown_neighbor_ids=recovery_unknown_ids,
        initialization_recovery_action_id=recovery_action_id,
        local_search_iterations=len(iteration_records),
        neighbor_count_per_iteration=tuple(
            len(item.neighbor_action_ids) for item in iteration_records
        ),
        feasible_neighbor_count_per_iteration=tuple(
            len(item.feasible_neighbor_ids) for item in iteration_records
        ),
        infeasible_neighbor_diagnostics=infeasible_diagnostics,
        unknown_neighbor_diagnostics=unknown_diagnostics,
        payoff_improvement_per_iteration=tuple(
            item.payoff_improvement for item in iteration_records
        ),
        local_optimality_margin=(
            final_verification.local_optimality_margin
            if certified and final_verification is not None else None
        ),
        termination_status=termination_status,
        local_sse_verified=strictly_certified,
        attacker_exactness_verified=bool(
            strictly_certified and all(
                evaluation.exact_attacker_best_response_verified
                for evaluation in cache.values() if evaluation.feasible
            )
        ),
        strong_tie_break_verified=bool(
            strictly_certified and all(
                evaluation.strong_tie_break_verified
                for evaluation in cache.values() if evaluation.feasible
            )
        ),
        independent_replay_status=None,
        isolated_feasible_local_solution=bool(
            final_verification is not None
            and final_verification.isolated_feasible_local_solution
        ),
        initialization_dependent_solution_concept=True,
        global_optimality_evaluated=False,
        global_optimal=None,
        solution_scope=LOCAL_SSE_SCOPE,
        neighborhood_metadata={
            "configuration": configuration.as_metadata(),
            "topology": topology.as_metadata(),
        },
        iterations=tuple(iteration_records),
        runtime_decomposition_s={
            "total_local_search_s": total,
            "exact_defender_evaluation_s": evaluation_s,
            "neighborhood_enumeration_s": neighborhood_s,
            "local_verification_s": verification_s,
            "search_control_residual_s": max(0.0, total - accounted),
        },
        exact_attacker_verification_required=require_exact,
    )


__all__ = [
    "DefenderEvaluator", "LOCAL_SEARCH_SCHEMA_VERSION", "LocalSSESearchResult",
    "LocalSearchIteration", "run_local_sse_search",
]
