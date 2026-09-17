"""Exact Stage-6 time-only Bellman solver on the validated glide DAG."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from bellman_geometry import GlideEdge
from bellman_graph import BellmanGraph, build_bellman_graph
from bellman_objectives import TimeObjective
from bellman_state import BellmanState, BellmanStateGrid
from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    GliderParameters,
    PhysicalScale,
)
from map_geometry import TerrainModel
from scenario import Point3D
from solver_metrics import SolverMetrics, SolverTiming


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TimeOptimalPath:
    """Backtracked policy and its independent time-replay certificate."""

    state_ids: tuple[int, ...]
    edges: tuple[GlideEdge, ...]
    bellman_value_s: float
    summed_edge_duration_s: float
    geometric_replay_duration_s: float

    def __post_init__(self) -> None:
        if len(self.state_ids) != len(self.edges) + 1:
            raise ValueError("a path must contain one more state than edge")
        for name in (
            "bellman_value_s",
            "summed_edge_duration_s",
            "geometric_replay_duration_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TimeOptimalSolution:
    """Exact time-to-go values and deterministic successor policy."""

    graph: BellmanGraph
    objective: TimeObjective
    value_s: FloatArray
    policy_successor: IntArray
    policy_action_index: IntArray
    goal_reachable: BoolArray
    tie_tolerance_s: float

    def __post_init__(self) -> None:
        expected_shape = (self.graph.grid.state_count,)
        arrays = (
            ("value_s", self.value_s),
            ("policy_successor", self.policy_successor),
            ("policy_action_index", self.policy_action_index),
            ("goal_reachable", self.goal_reachable),
        )
        for name, values in arrays:
            if np.asarray(values).shape != expected_shape:
                raise ValueError(f"{name} must have one entry per grid state")
        if not np.isfinite(self.tie_tolerance_s) or self.tie_tolerance_s < 0.0:
            raise ValueError("tie_tolerance_s must be finite and nonnegative")

    @property
    def goal_reachable_state_count(self) -> int:
        return int(np.count_nonzero(self.goal_reachable))

    def backtrack(self, start_state_id: int) -> tuple[int, ...]:
        """Return the stored optimal state sequence, or empty if unreachable."""
        self.graph._validate_node_id(start_state_id)
        current = int(start_state_id)
        if not self.goal_reachable[current]:
            return ()
        path = [current]
        for _ in range(self.graph.statistics.state_count):
            if self.graph.terminal_mask[current]:
                return tuple(path)
            successor = int(self.policy_successor[current])
            if successor < 0:
                raise RuntimeError("finite time state has no stored successor")
            if (
                self.graph.grid.decode(successor).altitude_index
                >= self.graph.grid.decode(current).altitude_index
            ):
                raise RuntimeError("time policy does not strictly decrease altitude")
            path.append(successor)
            current = successor
        raise RuntimeError("time-policy replay exceeded the finite graph size")


@dataclass(frozen=True)
class TimeOptimalRun:
    """Complete build/solve/backtrack result with structured metrics."""

    graph: BellmanGraph
    solution: TimeOptimalSolution
    path: TimeOptimalPath | None
    start_state_id: int
    metrics: SolverMetrics


def solve_time_optimal(
    graph: BellmanGraph,
    *,
    objective: TimeObjective | None = None,
    tie_tolerance_s: float = 1.0e-12,
) -> TimeOptimalSolution:
    """Solve the positive-duration time objective in reverse topological order."""
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    time_objective = objective or TimeObjective()
    tolerance = float(tie_tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance_s must be finite and nonnegative")

    value = np.full(graph.grid.state_count, np.inf, dtype=float)
    policy_successor = np.full(graph.grid.state_count, -1, dtype=np.int64)
    policy_action = np.full(graph.grid.state_count, -1, dtype=np.int64)
    value[graph.terminal_ids] = 0.0

    topological_order = graph.topological_sort()
    for state_id in reversed(topological_order):
        if graph.terminal_mask[state_id]:
            continue
        best_value = np.inf
        best_action = -1
        best_successor = -1
        for edge in graph.adjacency[state_id]:
            downstream_value = value[edge.target_id]
            if not np.isfinite(downstream_value):
                continue
            candidate = time_objective.edge_cost_s(edge) + downstream_value
            action_index = edge.target_state.heading_bin
            improves = candidate < best_value - tolerance
            ties_with_lower_action = (
                abs(candidate - best_value) <= tolerance
                and (best_action < 0 or action_index < best_action)
            )
            if improves or ties_with_lower_action:
                best_value = candidate
                best_action = action_index
                best_successor = edge.target_id
        if best_successor >= 0:
            value[state_id] = best_value
            policy_successor[state_id] = best_successor
            policy_action[state_id] = best_action

    reachable = graph.node_mask & np.isfinite(value)
    return TimeOptimalSolution(
        graph=graph,
        objective=time_objective,
        value_s=value,
        policy_successor=policy_successor,
        policy_action_index=policy_action,
        goal_reachable=reachable,
        tie_tolerance_s=tolerance,
    )


def _policy_edges(
    solution: TimeOptimalSolution,
    state_ids: tuple[int, ...],
) -> tuple[GlideEdge, ...]:
    edges: list[GlideEdge] = []
    for source_id, target_id in zip(state_ids, state_ids[1:]):
        matches = tuple(
            edge
            for edge in solution.graph.adjacency[source_id]
            if edge.target_id == target_id
        )
        if len(matches) != 1:
            raise RuntimeError("stored time policy does not identify exactly one edge")
        edges.append(matches[0])
    return tuple(edges)


def replay_time_optimal_path(
    solution: TimeOptimalSolution,
    start_state_id: int,
    *,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    consistency_tolerance_s: float = 1.0e-10,
) -> TimeOptimalPath:
    """Independently replay a reachable policy using stored and geometric time."""
    state_ids = solution.backtrack(start_state_id)
    if not state_ids:
        raise ValueError("cannot replay an unreachable start state")
    edges = _policy_edges(solution, state_ids)
    summed = float(sum(solution.objective.edge_cost_s(edge) for edge in edges))
    geometric = 0.0
    for source_id, target_id in zip(state_ids, state_ids[1:]):
        source = solution.graph.grid.position_map(
            solution.graph.grid.decode(source_id),
        )
        target = solution.graph.grid.position_map(
            solution.graph.grid.decode(target_id),
        )
        geometric += solution.objective.geometric_segment_duration_s(
            source,
            target,
            physical_scale=physical_scale,
        )
    bellman_value = float(solution.value_s[int(start_state_id)])
    if not (
        np.isclose(bellman_value, summed, rtol=0.0, atol=consistency_tolerance_s)
        and np.isclose(summed, geometric, rtol=0.0, atol=consistency_tolerance_s)
    ):
        raise RuntimeError(
            "Bellman value, summed edge duration, and geometric replay disagree"
        )
    return TimeOptimalPath(
        state_ids=state_ids,
        edges=edges,
        bellman_value_s=bellman_value,
        summed_edge_duration_s=summed,
        geometric_replay_duration_s=geometric,
    )


def independent_dijkstra_time(
    graph: BellmanGraph,
    *,
    objective: TimeObjective | None = None,
) -> FloatArray:
    """Reference reverse Dijkstra solve independent of DAG recursion order."""
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    time_objective = objective or TimeObjective()
    incoming: list[list[tuple[int, GlideEdge]]] = [
        [] for _ in range(graph.grid.state_count)
    ]
    for source_id in graph.node_ids:
        for edge in graph.adjacency[int(source_id)]:
            incoming[edge.target_id].append((int(source_id), edge))

    value = np.full(graph.grid.state_count, np.inf, dtype=float)
    queue: list[tuple[float, int]] = []
    for terminal_id in graph.terminal_ids:
        integer_id = int(terminal_id)
        value[integer_id] = 0.0
        heapq.heappush(queue, (0.0, integer_id))
    while queue:
        target_value, target_id = heapq.heappop(queue)
        if target_value > value[target_id]:
            continue
        for predecessor_id, edge in incoming[target_id]:
            candidate = target_value + time_objective.edge_cost_s(edge)
            if candidate < value[predecessor_id]:
                value[predecessor_id] = candidate
                heapq.heappush(queue, (candidate, predecessor_id))
    return value


def run_time_optimal_problem(
    grid: BellmanStateGrid,
    terrain: TerrainModel,
    goal: Point3D,
    start_state: BellmanState,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    tie_tolerance_s: float = 1.0e-12,
) -> TimeOptimalRun:
    """Build, solve, and replay one time-only problem with runtime metrics."""
    graph_start = perf_counter()
    graph = build_bellman_graph(
        grid,
        terrain,
        goal,
        parameters=parameters,
        physical_scale=physical_scale,
    )
    graph_build_s = perf_counter() - graph_start

    objective = TimeObjective(parameters.best_glide_speed_mps)
    solve_start = perf_counter()
    solution = solve_time_optimal(
        graph,
        objective=objective,
        tie_tolerance_s=tie_tolerance_s,
    )
    solve_s = perf_counter() - solve_start

    start_state_id = grid.encode(start_state)
    graph._validate_node_id(start_state_id)
    backtrack_start = perf_counter()
    state_ids = solution.backtrack(start_state_id)
    path = (
        replay_time_optimal_path(
            solution,
            start_state_id,
            physical_scale=physical_scale,
        )
        if state_ids
        else None
    )
    backtrack_s = perf_counter() - backtrack_start
    return TimeOptimalRun(
        graph=graph,
        solution=solution,
        path=path,
        start_state_id=start_state_id,
        metrics=SolverMetrics(
            timing=SolverTiming(
                graph_build_s=graph_build_s,
                solve_s=solve_s,
                backtrack_s=backtrack_s,
            ),
            state_count=graph.statistics.state_count,
            edge_count=graph.statistics.valid_edge_count,
        ),
    )
