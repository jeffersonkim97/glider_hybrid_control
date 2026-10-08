"""Solver-independent access to the existing P1b attacker best response.

The implementation deliberately delegates every model decision to ``3D_0827``:
the shared reachable graph owns feasible transitions and terminals,
``AttackerMDP`` owns the already-tested lazy hazard/stage-cost rows, and the two
solver adapters call the existing Bellman and Tabular entry points.  This module
adds a stable boundary and result shape; it does not define new physics, costs,
rewards, observations, or Local-SSE neighbourhood behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np

import project_paths  # noqa: F401 - configures imports from the sibling core
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import (
    AttackerMDP,
    QLearningConfig,
    QTable,
    TrajectoryEvaluation,
    evaluate_trajectory,
    learned_best_response,
    switching_candidates,
)
from P1b_condition import Scene
from bellman_geometry import GlideEdge, GlideTransitionModel, RejectedTransition
from bellman_state import BellmanState
from local_sse_contract import LocalDefenderEvaluation


SolverMethod = Literal["bellman", "tabular_q"]


@dataclass(frozen=True)
class AttackerTransition:
    """One authoritative feasible transition with its existing cost metadata."""

    source_state_id: int
    action_index: int
    edge: GlideEdge
    stage_cost: float
    cumulative_hazard: float
    terminal: bool

    @property
    def target_state_id(self) -> int:
        return int(self.edge.target_id)


class AttackerBRProblem:
    """Common problem view shared by solver adapters and later DQN wrappers.

    ``r_neighbor`` is intentionally absent.  It belongs to the Defender's Local
    SSE search and has no role in the Attacker transition relation.
    """

    def __init__(self, scene: Scene, sensor_map: tuple[float, float, float]) -> None:
        self.scene = scene
        self.sensor_map = tuple(float(value) for value in sensor_map)
        if len(self.sensor_map) != 3 or not np.all(np.isfinite(self.sensor_map)):
            raise ValueError("sensor_map must contain three finite coordinates")
        self.mdp = AttackerMDP(scene, self.sensor_map)
        self._switching = None
        self.transition_model = GlideTransitionModel(
            scene.grid,
            scene.terrain,
            parameters=scene.config.glider,
            physical_scale=scene.config.physical_scale,
        )

    @property
    def grid(self) -> Any:
        return self.scene.grid

    @property
    def graph(self) -> Any:
        return self.scene.graph

    @property
    def objective(self) -> Any:
        return self.scene.config.attacker_objective

    def state(self, state_id: int) -> BellmanState:
        """Decode the canonical integer state identifier."""

        return self.grid.decode(int(state_id))

    def state_id(self, state: BellmanState) -> int:
        """Encode the existing ``BellmanState`` without another state scheme."""

        return int(self.grid.encode(state))

    def position_map(self, state_id: int) -> np.ndarray:
        return np.asarray(self.grid.position_map(self.state(state_id)), dtype=float)

    def is_terminal(self, state_id: int) -> bool:
        return self.mdp.is_terminal(int(state_id))

    def successors(self, state_id: int) -> tuple[GlideEdge, ...]:
        """Return the exact goal-reachable adjacency used by both legacy solvers."""

        return tuple(self.mdp.actions(int(state_id)))

    def raw_transition_candidates(
        self, state_id: int,
    ) -> tuple[tuple[GlideEdge, ...], tuple[RejectedTransition, ...], dict[str, int]]:
        """Expose the existing raw dynamics and rejection records for diagnostics.

        Accepted raw edges can include targets outside the goal-backward reachable
        set.  ``successors`` is the attacker-BR action set and filters those edges
        exactly as the existing implicit graph does.
        """

        edges, statistics, rejected = self.transition_model.successors(
            self.state(state_id), include_rejected=True,
        )
        return tuple(edges), tuple(rejected), {
            "considered": statistics.considered,
            "valid": statistics.valid,
            "rejected_by_bounds": statistics.rejected_by_bounds,
            "rejected_by_terrain": statistics.rejected_by_terrain,
            "rejected_by_turn": statistics.rejected_by_turn,
            "rejected_by_altitude": statistics.rejected_by_altitude,
        }

    def transition(self, state_id: int, action_index: int) -> AttackerTransition:
        source = int(state_id)
        index = int(action_index)
        edges = self.successors(source)
        if not 0 <= index < len(edges):
            raise IndexError(
                f"action_index {index} is outside the {len(edges)} feasible actions"
            )
        edge = edges[index]
        return AttackerTransition(
            source_state_id=source,
            action_index=index,
            edge=edge,
            stage_cost=float(self.mdp.cost(source, index)),
            cumulative_hazard=float(self.mdp.hazard_row(source)[index]),
            terminal=self.is_terminal(int(edge.target_id)),
        )

    def evaluate_trajectory(
        self, state_ids: tuple[int, ...], *, reached_goal: bool | None = None,
    ) -> TrajectoryEvaluation:
        return evaluate_trajectory(
            self.mdp, state_ids, reached_goal=reached_goal,
        )

    def switching_state_ids(self) -> tuple[int, ...]:
        return self._switching_candidates().state_ids

    def _switching_candidates(self) -> Any:
        if self._switching is None:
            self._switching = switching_candidates(self.scene, self.sensor_map)
        return self._switching

    def powered_cost(self, switching_state_id: int) -> float:
        candidates = self._switching_candidates()
        try:
            return float(candidates.powered_cost_of[int(switching_state_id)])
        except KeyError as error:
            raise ValueError(
                f"state {switching_state_id} is not an admissible switching state"
            ) from error

    def clear_runtime_caches(self) -> None:
        """Drop deterministic lazy MDP rows while preserving the model and starts."""

        self.mdp._actions.clear()
        self.mdp._cost.clear()
        self.mdp._hazard.clear()

    def attacker_objective(
        self, switching_state_id: int, trajectory: tuple[int, ...],
    ) -> float:
        """Evaluate full powered-plus-glide ``J_A`` using existing components."""

        if not trajectory or int(trajectory[0]) != int(switching_state_id):
            raise ValueError("trajectory must begin at switching_state_id")
        scored = self.evaluate_trajectory(trajectory)
        if not scored.reached_goal:
            raise ValueError("a best-response trajectory must reach the goal")
        return self.powered_cost(switching_state_id) + scored.cost


@dataclass(frozen=True)
class AttackerBRResult:
    """Backend-neutral attacker best-response result."""

    method: SolverMethod
    status: str
    success: bool
    sensor_map: tuple[float, float, float]
    switching_state_id: int | None
    trajectory: tuple[int, ...]
    reached_goal: bool
    attacker_objective: float | None
    detection_probability: float | None
    runtime_sec: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    backend_result: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "success": self.success,
            "sensor_map": list(self.sensor_map),
            "switching_state_id": self.switching_state_id,
            "trajectory": list(self.trajectory),
            "reached_goal": self.reached_goal,
            "attacker_objective": self.attacker_objective,
            "detection_probability": self.detection_probability,
            "runtime_sec": self.runtime_sec,
            "diagnostics": self.diagnostics,
        }

    def as_local_defender_evaluation(self, action_id: int) -> LocalDefenderEvaluation:
        """Adapt the unchanged backend payoff to the existing Local-SSE contract."""

        exact = self.method == "bellman"
        if not self.success:
            return LocalDefenderEvaluation(
                action_id=action_id,
                status="model_infeasible",
                defender_value=None,
                attacker_objective=None,
                selected_attacker_response_id=None,
                exact_attacker_best_response_verified=exact,
                strong_tie_break_verified=exact,
                diagnostic=self.status,
            )
        if self.detection_probability is None:
            raise ValueError("successful backend result omitted existing J_D payoff")
        return LocalDefenderEvaluation(
            action_id=action_id,
            status="feasible",
            defender_value=float(self.detection_probability),
            attacker_objective=float(self.attacker_objective),
            selected_attacker_response_id=int(self.switching_state_id),
            exact_attacker_best_response_verified=exact,
            strong_tie_break_verified=exact,
            diagnostic=None,
        )


class BellmanAttackerBRAdapter:
    method: SolverMethod = "bellman"

    def solve(self, problem: AttackerBRProblem) -> AttackerBRResult:
        started = perf_counter()
        response = exact_best_response(problem.scene, problem.sensor_map)
        runtime = perf_counter() - started
        trajectory = tuple(int(value) for value in response.glide_state_ids)
        reached = bool(trajectory and problem.is_terminal(trajectory[-1]))
        return AttackerBRResult(
            method=self.method,
            status="success" if response.feasible else "model_infeasible",
            success=bool(response.feasible),
            sensor_map=problem.sensor_map,
            switching_state_id=response.switching_state_id,
            trajectory=trajectory,
            reached_goal=reached,
            attacker_objective=response.attacker_objective,
            detection_probability=response.detection_probability,
            runtime_sec=runtime,
            diagnostics={
                "cooptimal_state_ids": list(response.cooptimal_state_ids),
                "admissible_switch_states": response.admissible_switch_states,
                "timing": dict(response.timing),
                "sizes": dict(response.sizes),
            },
            backend_result=response,
        )


class TabularQAttackerBRAdapter:
    method: SolverMethod = "tabular_q"

    def __init__(self, config: QLearningConfig | None = None) -> None:
        self.config = config or QLearningConfig()

    def solve(self, problem: AttackerBRProblem) -> AttackerBRResult:
        table = QTable(
            action_count=len(problem.grid.motion_offsets),
            initial_q=self.config.initial_q,
        )
        started = perf_counter()
        response = learned_best_response(
            problem.scene, problem.sensor_map, table, self.config,
        )
        runtime = perf_counter() - started
        trajectory = tuple(int(value) for value in response.trajectory)
        reached = bool(trajectory and problem.is_terminal(trajectory[-1]))
        return AttackerBRResult(
            method=self.method,
            status="success" if response.feasible else "model_infeasible",
            success=bool(response.feasible),
            sensor_map=problem.sensor_map,
            switching_state_id=response.switching_state_id,
            trajectory=trajectory,
            reached_goal=reached,
            attacker_objective=response.attacker_objective,
            detection_probability=response.detection_probability,
            runtime_sec=runtime,
            diagnostics={
                "training_seconds": response.training_seconds,
                "episodes": response.episodes,
                "visited_states": table.visited_states,
                "tried_state_action_pairs": table.tried_pairs,
                "config": self.config.as_dict(),
            },
            backend_result=response,
        )


def solve_attacker_br(
    problem: AttackerBRProblem,
    method: SolverMethod,
    solver_config: QLearningConfig | None = None,
) -> AttackerBRResult:
    """Solve through the common boundary without changing either backend."""

    if method == "bellman":
        if solver_config is not None:
            raise ValueError("Bellman does not accept Q-learning configuration")
        return BellmanAttackerBRAdapter().solve(problem)
    if method == "tabular_q":
        return TabularQAttackerBRAdapter(solver_config).solve(problem)
    raise ValueError(f"unsupported attacker BR solver method: {method!r}")


__all__ = [
    "AttackerBRProblem",
    "AttackerBRResult",
    "AttackerTransition",
    "BellmanAttackerBRAdapter",
    "SolverMethod",
    "TabularQAttackerBRAdapter",
    "solve_attacker_br",
]
