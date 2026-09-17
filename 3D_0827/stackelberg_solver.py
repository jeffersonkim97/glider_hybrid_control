"""Stage-12 exact enumeration for a finite Defender action set."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from attacker_best_response import (
    DEFAULT_GRAPH_BOUNDS,
    AttackerBestResponseRun,
    CandidateMissionResult,
    attacker_response_from_result,
    run_attacker_best_response,
)
from bellman_state import BellmanStateGrid
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from game_types import AttackerInitialCondition, DefenderAction, GameOutcome
from map_geometry import TerrainModel
from stackelberg_interface import select_sse_follower_outcome
from sparse_reachability import build_goal_backward_reachable_graph


@dataclass(frozen=True)
class DefenderCandidate:
    """One deterministically identified finite leader action."""

    action_id: int
    action: DefenderAction

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_id, Integral)
            or isinstance(self.action_id, bool)
            or int(self.action_id) < 0
        ):
            raise ValueError("action_id must be a nonnegative integer")
        if not isinstance(self.action, DefenderAction):
            raise TypeError("action must be a DefenderAction")
        object.__setattr__(self, "action_id", int(self.action_id))


@dataclass(frozen=True)
class DefenderEvaluation:
    """Exact follower response and leader payoff for one action."""

    candidate: DefenderCandidate
    attacker_run: AttackerBestResponseRun
    sse_follower_result: CandidateMissionResult | None
    outcome: GameOutcome | None
    attacker_br_runtime_s: float
    infeasibility_reason: str | None = None

    @property
    def feasible(self) -> bool:
        return self.outcome is not None

    @property
    def defender_payoff(self) -> float | None:
        return None if self.outcome is None else self.outcome.defender_payoff


@dataclass(frozen=True)
class StackelbergTiming:
    per_action_s: tuple[float, ...]
    total_s: float
    shared_graph_build_s: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.per_action_s)
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("per-action runtimes must be finite and nonnegative")
        total = float(self.total_s)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("total runtime must be finite and nonnegative")
        shared = float(self.shared_graph_build_s)
        if not np.isfinite(shared) or shared < 0.0:
            raise ValueError("shared graph runtime must be finite and nonnegative")
        object.__setattr__(self, "per_action_s", values)
        object.__setattr__(self, "total_s", total)
        object.__setattr__(self, "shared_graph_build_s", shared)


@dataclass(frozen=True)
class FiniteStackelbergRun:
    """Complete exact result over one finite Defender action sequence."""

    defender_candidates: tuple[DefenderCandidate, ...]
    evaluations: tuple[DefenderEvaluation, ...]
    selected_evaluation: DefenderEvaluation
    timing: StackelbergTiming
    follower_objective_tolerance: float
    leader_payoff_tolerance: float
    equilibrium_scope: str = (
        "exact exhaustive Stackelberg solution over the configured finite "
        "Defender actions, switching candidates, and Bellman lattice"
    )

    @property
    def outcome(self) -> GameOutcome:
        outcome = self.selected_evaluation.outcome
        if outcome is None:
            raise RuntimeError("selected Defender evaluation is infeasible")
        return outcome


def generate_defender_line_actions(
    x_coordinates: Iterable[float],
    *,
    y_map: float = 0.0,
    z_map: float = 0.0,
) -> tuple[DefenderCandidate, ...]:
    """Generate a deterministic finite sensor line for the initial game."""
    positions = tuple(float(value) for value in x_coordinates)
    if not positions or any(not np.isfinite(value) for value in positions):
        raise ValueError("x_coordinates must contain finite values")
    y_value = float(y_map)
    z_value = float(z_map)
    if not np.isfinite(y_value) or not np.isfinite(z_value):
        raise ValueError("sensor y/z coordinates must be finite")
    return tuple(
        DefenderCandidate(
            action_id=index,
            action=DefenderAction(np.array([x_value, y_value, z_value])),
        )
        for index, x_value in enumerate(positions)
    )


def select_stackelberg_outcome(
    outcomes: Sequence[GameOutcome],
    *,
    payoff_tolerance: float = 0.0,
) -> GameOutcome:
    """Select maximum Defender payoff; retain stable order for remaining ties."""
    sequence = tuple(outcomes)
    if not sequence:
        raise ValueError("at least one feasible GameOutcome is required")
    if any(not isinstance(outcome, GameOutcome) for outcome in sequence):
        raise TypeError("outcomes must contain only GameOutcome values")
    tolerance = float(payoff_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("payoff_tolerance must be finite and nonnegative")
    maximum = max(outcome.defender_payoff for outcome in sequence)
    return next(
        outcome
        for outcome in sequence
        if outcome.defender_payoff >= maximum - tolerance
    )


def select_geometry_sse_follower(
    attacker_run: AttackerBestResponseRun,
    *,
    objective_tolerance: float = 1.0e-12,
) -> tuple[CandidateMissionResult, GameOutcome]:
    """Apply leader-favouring SSE selection to stored candidate responses."""
    tolerance = float(objective_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("objective_tolerance must be finite and nonnegative")
    feasible = tuple(
        result for result in attacker_run.candidate_results if result.feasible
    )
    if not feasible:
        raise ValueError("Defender action has no feasible Attacker response")
    minimum = min(float(result.objective) for result in feasible)
    cooptimal = tuple(sorted(
        (
            result for result in feasible
            if float(result.objective) <= minimum + tolerance
        ),
        key=lambda result: result.candidate_id,
    ))
    pairs = tuple(
        (
            result,
            GameOutcome(
                defender_action=attacker_run.defender_action,
                attacker_response=attacker_response_from_result(result),
                defender_payoff=float(result.detection_probability),
                attacker_payoff=float(result.objective),
            ),
        )
        for result in cooptimal
    )
    selected_outcome = select_sse_follower_outcome(
        tuple(outcome for _, outcome in pairs),
        objective_tolerance=tolerance,
    )
    selected_result = next(
        result for result, outcome in pairs if outcome is selected_outcome
    )
    return selected_result, selected_outcome


def run_finite_stackelberg(
    defender_candidates: Sequence[DefenderCandidate],
    scenario: AttackerInitialCondition,
    *,
    terrain: TerrainModel,
    follower_objective_tolerance: float = 1.0e-12,
    leader_payoff_tolerance: float = 1.0e-12,
    attacker_kwargs: Mapping[str, object] | None = None,
    reuse_reachability_graph: bool = False,
) -> FiniteStackelbergRun:
    """Literally enumerate every leader action and its exact follower BR."""
    candidates = tuple(defender_candidates)
    if not candidates:
        raise ValueError("at least one Defender candidate is required")
    if any(not isinstance(candidate, DefenderCandidate) for candidate in candidates):
        raise TypeError("defender_candidates must contain DefenderCandidate values")
    action_ids = tuple(candidate.action_id for candidate in candidates)
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("Defender action IDs must be unique")
    if not isinstance(scenario, AttackerInitialCondition):
        raise TypeError("scenario must be an AttackerInitialCondition")
    follower_tolerance = float(follower_objective_tolerance)
    leader_tolerance = float(leader_payoff_tolerance)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (follower_tolerance, leader_tolerance)
    ):
        raise ValueError("Stackelberg tolerances must be finite and nonnegative")
    kwargs = dict(attacker_kwargs or {})
    if "terrain" in kwargs:
        raise ValueError("terrain must be supplied only through the named argument")

    total_start = perf_counter()
    shared_graph_build_s = 0.0
    if reuse_reachability_graph:
        backend = kwargs.get("bellman_backend", "dense")
        if backend != "goal_backward_sparse":
            raise ValueError(
                "reachability reuse requires goal_backward_sparse backend"
            )
        if "prebuilt_reachability" in kwargs:
            raise ValueError(
                "prebuilt_reachability and reuse_reachability_graph are exclusive"
            )
        active_grid = kwargs.get("bellman_grid")
        if active_grid is None:
            bounds = kwargs.get("graph_bounds", DEFAULT_GRAPH_BOUNDS)
            active_grid = BellmanStateGrid(bounds)  # type: ignore[arg-type]
        if not isinstance(active_grid, BellmanStateGrid):
            raise TypeError("bellman_grid must be a BellmanStateGrid")
        graph_started = perf_counter()
        kwargs["prebuilt_reachability"] = build_goal_backward_reachable_graph(
            active_grid,
            terrain,
            scenario.goal,
            parameters=kwargs.get("parameters", DEFAULT_GLIDER),  # type: ignore[arg-type]
            physical_scale=kwargs.get(  # type: ignore[arg-type]
                "physical_scale", DEFAULT_PHYSICAL_SCALE,
            ),
        )
        shared_graph_build_s = perf_counter() - graph_started
    evaluations: list[DefenderEvaluation] = []
    for candidate in candidates:
        action_start = perf_counter()
        attacker_run = run_attacker_best_response(
            candidate.action,
            scenario,
            terrain=terrain,
            objective_tolerance=follower_tolerance,
            **kwargs,
        )
        runtime = perf_counter() - action_start
        try:
            follower_result, outcome = select_geometry_sse_follower(
                attacker_run,
                objective_tolerance=follower_tolerance,
            )
            evaluations.append(DefenderEvaluation(
                candidate=candidate,
                attacker_run=attacker_run,
                sse_follower_result=follower_result,
                outcome=outcome,
                attacker_br_runtime_s=runtime,
            ))
        except ValueError as error:
            evaluations.append(DefenderEvaluation(
                candidate=candidate,
                attacker_run=attacker_run,
                sse_follower_result=None,
                outcome=None,
                attacker_br_runtime_s=runtime,
                infeasibility_reason=str(error),
            ))

    feasible = tuple(evaluation for evaluation in evaluations if evaluation.feasible)
    if not feasible:
        raise RuntimeError("no Defender action has a feasible Attacker response")
    selected_outcome = select_stackelberg_outcome(
        tuple(evaluation.outcome for evaluation in feasible if evaluation.outcome is not None),
        payoff_tolerance=leader_tolerance,
    )
    selected = next(
        evaluation for evaluation in feasible
        if evaluation.outcome is selected_outcome
    )
    total_s = perf_counter() - total_start
    result_tuple = tuple(evaluations)
    return FiniteStackelbergRun(
        defender_candidates=candidates,
        evaluations=result_tuple,
        selected_evaluation=selected,
        timing=StackelbergTiming(
            per_action_s=tuple(item.attacker_br_runtime_s for item in result_tuple),
            total_s=total_s,
            shared_graph_build_s=shared_graph_build_s,
        ),
        follower_objective_tolerance=follower_tolerance,
        leader_payoff_tolerance=leader_tolerance,
    )


__all__ = [
    "DefenderCandidate", "DefenderEvaluation", "FiniteStackelbergRun",
    "StackelbergTiming", "generate_defender_line_actions",
    "run_finite_stackelberg", "select_geometry_sse_follower",
    "select_stackelberg_outcome",
]
