"""Stage-5 discrete glide DAG and exact unit-cost goal reachability."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq

import numpy as np
from numpy.typing import NDArray

from bellman_geometry import GlideEdge, GlideTransitionModel, TransitionStatistics
from bellman_state import BellmanState, BellmanStateGrid, is_goal_terminal
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from map_geometry import TerrainModel
from scenario import Point3D


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GraphStatistics:
    cartesian_state_count: int
    state_count: int
    terrain_excluded_state_count: int
    valid_edge_count: int
    rejected_by_bounds: int
    rejected_by_terrain: int
    rejected_by_turn: int
    rejected_by_altitude: int
    terminal_state_count: int


@dataclass(frozen=True)
class BellmanGraph:
    """Immutable adjacency/predecessor representation of the glide DAG."""

    grid: BellmanStateGrid
    terrain: TerrainModel
    goal: Point3D
    node_mask: BoolArray
    terminal_mask: BoolArray
    adjacency: tuple[tuple[GlideEdge, ...], ...]
    predecessors: tuple[tuple[int, ...], ...]
    statistics: GraphStatistics

    @property
    def node_ids(self) -> IntArray:
        return np.flatnonzero(self.node_mask)

    @property
    def terminal_ids(self) -> IntArray:
        return np.flatnonzero(self.terminal_mask)

    def successors(self, state_id: int) -> tuple[GlideEdge, ...]:
        self._validate_node_id(state_id)
        return self.adjacency[int(state_id)]

    def _validate_node_id(self, state_id: int) -> None:
        if not isinstance(state_id, (int, np.integer)) or isinstance(state_id, bool):
            raise TypeError("state_id must be an integer")
        if not 0 <= int(state_id) < self.grid.state_count:
            raise ValueError("state_id lies outside the graph grid")
        if not self.node_mask[int(state_id)]:
            raise ValueError("state_id is excluded by terrain occupancy")

    def topological_sort(self) -> tuple[int, ...]:
        """Run deterministic Kahn sorting as an explicit cycle check."""
        indegree = np.zeros(self.grid.state_count, dtype=np.int64)
        for source_id in self.node_ids:
            for edge in self.adjacency[int(source_id)]:
                indegree[edge.target_id] += 1
        ready = [
            int(state_id)
            for state_id in self.node_ids
            if indegree[int(state_id)] == 0
        ]
        heapq.heapify(ready)
        ordering: list[int] = []
        while ready:
            source_id = heapq.heappop(ready)
            ordering.append(source_id)
            for edge in self.adjacency[source_id]:
                indegree[edge.target_id] -= 1
                if indegree[edge.target_id] == 0:
                    heapq.heappush(ready, edge.target_id)
        if len(ordering) != self.statistics.state_count:
            raise RuntimeError("glide graph contains a directed cycle")
        return tuple(ordering)


@dataclass(frozen=True)
class UnitCostReachability:
    """Exact Stage-5 unit-edge Bellman values and successor policy."""

    graph: BellmanGraph
    value: FloatArray
    policy_successor: IntArray
    goal_reachable: BoolArray

    @property
    def goal_reachable_state_count(self) -> int:
        return int(np.count_nonzero(self.goal_reachable))

    @property
    def maximum_finite_value(self) -> float:
        finite = self.value[np.isfinite(self.value)]
        return float(np.max(finite)) if len(finite) else float("nan")

    def backtrack(self, start_state_id: int) -> tuple[int, ...]:
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
                raise RuntimeError("finite Bellman state has no stored successor")
            current_altitude = self.graph.grid.decode(current).altitude_index
            successor_altitude = self.graph.grid.decode(successor).altitude_index
            if successor_altitude >= current_altitude:
                raise RuntimeError("stored policy does not strictly decrease altitude")
            path.append(successor)
            current = successor
        raise RuntimeError("policy replay exceeded the finite graph size")


def build_bellman_graph(
    grid: BellmanStateGrid,
    terrain: TerrainModel,
    goal: Point3D,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> BellmanGraph:
    """Build all admissible states and valid edges for the Stage-5 DAG."""
    if not isinstance(grid, BellmanStateGrid):
        raise TypeError("grid must be a BellmanStateGrid")
    if not isinstance(goal, Point3D):
        raise TypeError("goal must be a Point3D")

    node_mask = np.zeros(grid.state_count, dtype=bool)
    terminal_mask = np.zeros(grid.state_count, dtype=bool)
    terrain_excluded = 0
    for altitude_index in range(grid.altitude_count):
        for y_index in range(grid.y_count):
            for x_index in range(grid.x_count):
                # Heading does not affect point occupancy.
                representative = BellmanState(
                    x_index, y_index, altitude_index, 0,
                )
                position = grid.position_map(representative)
                occupied = terrain.contains_solid(position)
                terminal = (
                    not occupied
                    and is_goal_terminal(
                        position,
                        goal,
                        physical_scale=physical_scale,
                        goal_tolerance_m=parameters.goal_tolerance_m,
                    )
                )
                for heading_bin in range(grid.heading_bin_count):
                    state = BellmanState(
                        x_index,
                        y_index,
                        altitude_index,
                        heading_bin,
                    )
                    state_id = grid.encode(state)
                    if occupied:
                        terrain_excluded += 1
                    else:
                        node_mask[state_id] = True
                        terminal_mask[state_id] = terminal

    transition_model = GlideTransitionModel(
        grid,
        terrain,
        parameters,
        physical_scale,
    )
    adjacency_lists: list[tuple[GlideEdge, ...]] = [
        () for _ in range(grid.state_count)
    ]
    predecessor_lists: list[list[int]] = [
        [] for _ in range(grid.state_count)
    ]
    transition_statistics = TransitionStatistics()
    for state_id in np.flatnonzero(node_mask):
        integer_id = int(state_id)
        if terminal_mask[integer_id]:
            continue
        state = grid.decode(integer_id)
        edges, local_statistics, _ = transition_model.successors(state)
        for edge in edges:
            if not node_mask[edge.target_id]:
                raise RuntimeError("valid edge targets a terrain-excluded state")
            predecessor_lists[edge.target_id].append(integer_id)
        adjacency_lists[integer_id] = edges
        transition_statistics.add(local_statistics)

    statistics = GraphStatistics(
        cartesian_state_count=grid.state_count,
        state_count=int(np.count_nonzero(node_mask)),
        terrain_excluded_state_count=terrain_excluded,
        valid_edge_count=transition_statistics.valid,
        rejected_by_bounds=transition_statistics.rejected_by_bounds,
        rejected_by_terrain=transition_statistics.rejected_by_terrain,
        rejected_by_turn=transition_statistics.rejected_by_turn,
        rejected_by_altitude=transition_statistics.rejected_by_altitude,
        terminal_state_count=int(np.count_nonzero(terminal_mask)),
    )
    return BellmanGraph(
        grid=grid,
        terrain=terrain,
        goal=goal,
        node_mask=node_mask,
        terminal_mask=terminal_mask,
        adjacency=tuple(adjacency_lists),
        predecessors=tuple(tuple(values) for values in predecessor_lists),
        statistics=statistics,
    )


def solve_unit_cost_reachability(graph: BellmanGraph) -> UnitCostReachability:
    """Solve exact unit-cost reachability in reverse altitude order."""
    value = np.full(graph.grid.state_count, np.inf, dtype=float)
    policy = np.full(graph.grid.state_count, -1, dtype=np.int64)
    terminal_ids = graph.terminal_ids
    value[terminal_ids] = 0.0

    # Encoded IDs are altitude-major, so increasing ID processes every lower
    # altitude layer before states whose successors point into that layer.
    for state_id in graph.node_ids:
        integer_id = int(state_id)
        if graph.terminal_mask[integer_id]:
            continue
        finite_successors = [
            edge
            for edge in graph.adjacency[integer_id]
            if np.isfinite(value[edge.target_id])
        ]
        if not finite_successors:
            continue
        selected = min(
            finite_successors,
            key=lambda edge: (value[edge.target_id], edge.target_id),
        )
        value[integer_id] = selected.unit_cost + value[selected.target_id]
        policy[integer_id] = selected.target_id

    reachable = graph.node_mask & np.isfinite(value)
    return UnitCostReachability(
        graph=graph,
        value=value,
        policy_successor=policy,
        goal_reachable=reachable,
    )


def independent_reverse_reachability(graph: BellmanGraph) -> BoolArray:
    """Reference graph search independent of Bellman value recursion."""
    reachable = np.zeros(graph.grid.state_count, dtype=bool)
    queue: deque[int] = deque(int(state_id) for state_id in graph.terminal_ids)
    reachable[graph.terminal_ids] = True
    while queue:
        target_id = queue.popleft()
        for predecessor_id in graph.predecessors[target_id]:
            if not reachable[predecessor_id]:
                reachable[predecessor_id] = True
                queue.append(predecessor_id)
    return reachable
