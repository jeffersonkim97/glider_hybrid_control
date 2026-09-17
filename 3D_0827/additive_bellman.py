"""Generic additive Bellman recursion on the validated Stage-5 glide DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bellman_geometry import GlideEdge
from bellman_graph import BellmanGraph


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AdditiveBellmanSolution:
    """Exact additive cost-to-go and deterministic successor policy."""

    graph: BellmanGraph
    edge_cost_by_source: tuple[tuple[float, ...], ...]
    active_mask: BoolArray
    value: FloatArray
    policy_successor: IntArray
    policy_action_index: IntArray
    goal_reachable: BoolArray
    tie_tolerance: float

    def backtrack(self, start_state_id: int) -> tuple[int, ...]:
        self.graph._validate_node_id(start_state_id)
        current = int(start_state_id)
        if not self.goal_reachable[current]:
            return ()
        state_ids = [current]
        for _ in range(self.graph.statistics.state_count):
            if self.graph.terminal_mask[current]:
                return tuple(state_ids)
            successor = int(self.policy_successor[current])
            if successor < 0:
                raise RuntimeError("finite additive state has no stored successor")
            if (
                self.graph.grid.decode(successor).altitude_index
                >= self.graph.grid.decode(current).altitude_index
            ):
                raise RuntimeError("additive policy does not decrease altitude")
            state_ids.append(successor)
            current = successor
        raise RuntimeError("additive-policy replay exceeded graph size")

    def path_edges(self, start_state_id: int) -> tuple[GlideEdge, ...]:
        state_ids = self.backtrack(start_state_id)
        edges: list[GlideEdge] = []
        for source_id, target_id in zip(state_ids, state_ids[1:]):
            matches = tuple(
                edge for edge in self.graph.adjacency[source_id]
                if edge.target_id == target_id
            )
            if len(matches) != 1:
                raise RuntimeError("stored policy does not identify exactly one edge")
            edges.append(matches[0])
        return tuple(edges)


def descendant_mask(
    graph: BellmanGraph,
    start_state_ids: Sequence[int],
) -> BoolArray:
    """Return every state reachable downstream from the supplied starts."""
    active = np.zeros(graph.grid.state_count, dtype=bool)
    stack = sorted({int(value) for value in start_state_ids}, reverse=True)
    for state_id in stack:
        graph._validate_node_id(state_id)
    while stack:
        state_id = stack.pop()
        if active[state_id]:
            continue
        active[state_id] = True
        for edge in graph.adjacency[state_id]:
            if not active[edge.target_id]:
                stack.append(edge.target_id)
    return active


def restricted_adjacency(
    graph: BellmanGraph,
    active_mask: BoolArray,
) -> tuple[tuple[GlideEdge, ...], ...]:
    """Preserve edge order while blanking sources outside an active subgraph."""
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != (graph.grid.state_count,):
        raise ValueError("active_mask must contain one value per graph state")
    return tuple(
        graph.adjacency[state_id] if active[state_id] else ()
        for state_id in range(graph.grid.state_count)
    )


def solve_additive_bellman(
    graph: BellmanGraph,
    edge_cost_by_source: Sequence[Sequence[float]],
    *,
    active_mask: BoolArray | None = None,
    tie_tolerance: float = 1.0e-12,
) -> AdditiveBellmanSolution:
    """Solve externally precomputed nonnegative edge costs on the glide DAG."""
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    costs = tuple(tuple(float(value) for value in row) for row in edge_cost_by_source)
    if len(costs) != graph.grid.state_count:
        raise ValueError("edge costs must contain one row per graph state")
    active = (
        graph.node_mask.copy()
        if active_mask is None
        else np.asarray(active_mask, dtype=bool).copy()
    )
    if active.shape != (graph.grid.state_count,):
        raise ValueError("active_mask must contain one value per graph state")
    if np.any(active & ~graph.node_mask):
        raise ValueError("active_mask includes terrain-excluded states")
    tolerance = float(tie_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and nonnegative")
    for state_id in np.flatnonzero(active):
        if len(costs[int(state_id)]) != len(graph.adjacency[int(state_id)]):
            raise ValueError("active edge-cost row is not aligned with graph adjacency")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in costs[int(state_id)]
        ):
            raise ValueError("active edge costs must be finite and nonnegative")

    value = np.full(graph.grid.state_count, np.inf, dtype=float)
    successor = np.full(graph.grid.state_count, -1, dtype=np.int64)
    action = np.full(graph.grid.state_count, -1, dtype=np.int64)
    active_terminals = graph.terminal_mask & active
    value[active_terminals] = 0.0

    # IDs are altitude-major and every glide edge descends strictly, so lower
    # layers have already been solved when their predecessors are visited.
    for state_id_value in np.flatnonzero(active):
        state_id = int(state_id_value)
        if graph.terminal_mask[state_id]:
            continue
        best_value = np.inf
        best_action = -1
        best_successor = -1
        for edge_index, edge in enumerate(graph.adjacency[state_id]):
            if not active[edge.target_id] or not np.isfinite(value[edge.target_id]):
                continue
            candidate = costs[state_id][edge_index] + value[edge.target_id]
            action_index = edge.target_state.heading_bin
            improves = candidate < best_value - tolerance
            lower_tie = (
                abs(candidate - best_value) <= tolerance
                and (
                    best_action < 0
                    or action_index < best_action
                    or (
                        action_index == best_action
                        and edge.target_id < best_successor
                    )
                )
            )
            if improves or lower_tie:
                best_value = candidate
                best_action = action_index
                best_successor = edge.target_id
        if best_successor >= 0:
            value[state_id] = best_value
            successor[state_id] = best_successor
            action[state_id] = best_action

    reachable = active & np.isfinite(value)
    return AdditiveBellmanSolution(
        graph=graph,
        edge_cost_by_source=costs,
        active_mask=active,
        value=value,
        policy_successor=successor,
        policy_action_index=action,
        goal_reachable=reachable,
        tie_tolerance=tolerance,
    )
