"""Exact goal-backward reachability without materializing the full edge graph.

The transition relation is identical to :mod:`bellman_geometry`.  States are
discovered from goal terminals through analytically generated predecessors;
successor ``GlideEdge`` objects are generated lazily only when a reachable
source row is requested.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from bellman_geometry import GlideEdge, GlideTransitionModel
from bellman_graph import BellmanGraph, GraphStatistics, UnitCostReachability
from bellman_state import BellmanState, BellmanStateGrid, is_goal_terminal
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from map_geometry import TerrainModel
from scenario import Point3D


class ImplicitReachableAdjacency(Sequence[tuple[GlideEdge, ...]]):
    """Sequence facade that generates only rows in the proven reachable set."""

    def __init__(
        self,
        grid: BellmanStateGrid,
        terrain: TerrainModel,
        reachable_mask: np.ndarray,
        terminal_mask: np.ndarray,
        *,
        parameters: GliderParameters,
        physical_scale: PhysicalScale,
    ) -> None:
        self.grid = grid
        self.terrain = terrain
        self.reachable_mask = np.asarray(reachable_mask, dtype=bool)
        self.terminal_mask = np.asarray(terminal_mask, dtype=bool)
        self.transition_model = GlideTransitionModel(
            grid, terrain, parameters, physical_scale,
        )

    def __len__(self) -> int:
        return self.grid.state_count

    def __getitem__(self, index: int | slice) -> tuple[GlideEdge, ...] | tuple[tuple[GlideEdge, ...], ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        state_id = int(index)
        if not 0 <= state_id < len(self):
            raise IndexError(state_id)
        if not self.reachable_mask[state_id] or self.terminal_mask[state_id]:
            return ()
        state = self.grid.decode(state_id)
        edges, _, _ = self.transition_model.successors(state)
        return tuple(
            edge for edge in edges if self.reachable_mask[edge.target_id]
        )

    def __iter__(self) -> Iterator[tuple[GlideEdge, ...]]:
        for state_id in range(len(self)):
            yield self[state_id]


@dataclass(frozen=True)
class SparseReachabilityMetrics:
    cartesian_state_count: int
    terminal_state_count: int
    reachable_state_count: int
    reachable_edge_count: int
    predecessor_candidates: int
    rejected_by_bounds: int
    rejected_by_terminal_source: int
    rejected_by_terrain: int
    build_s: float
    dense_state_fraction: float


@dataclass(frozen=True)
class SparseReachabilityRun:
    graph: BellmanGraph
    unit_reachability: UnitCostReachability
    metrics: SparseReachabilityMetrics


def build_forward_descendant_mask(
    graph: BellmanGraph,
    start_state_ids: Sequence[int],
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> np.ndarray:
    """Propagate a finite start set through the exact DAG in altitude layers.

    This is the forward half of the two-sided corridor reduction.  The graph's
    ``node_mask`` is already the goal-backward set, so the result contains only
    states that are both descendants of a supplied start and ancestors of the
    goal.  Terrain implementations without vectorized helpers retain the same
    scalar contract through the fallback wrappers above.
    """
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    starts = tuple(sorted({int(value) for value in start_state_ids}))
    active = np.zeros(graph.grid.state_count, dtype=bool)
    for state_id in starts:
        graph._validate_node_id(state_id)
        active[state_id] = True
    if not starts:
        return active

    grid = graph.grid
    active_4d = active.reshape((
        grid.altitude_count,
        grid.y_count,
        grid.x_count,
        grid.heading_bin_count,
    ))
    node_4d = graph.node_mask.reshape(active_4d.shape)
    terminal_positions = np.any(
        graph.terminal_mask.reshape(active_4d.shape), axis=3,
    )
    transition_model = GlideTransitionModel(
        grid, graph.terrain, parameters, physical_scale,
    )
    source_headings: list[tuple[int, ...]] = []
    altitude_losses: list[int] = []
    for target_heading_bin, (x_offset, y_offset) in enumerate(grid.motion_offsets):
        distance_map = grid.horizontal_spacing_map * float(
            np.hypot(x_offset, y_offset)
        )
        distance_m = physical_scale.distance_m(distance_map)
        target_heading = grid.heading_rad(target_heading_bin)
        source_headings.append(tuple(
            source_heading_bin
            for source_heading_bin in range(grid.heading_bin_count)
            if transition_model.turn_is_feasible(
                grid.heading_rad(source_heading_bin), target_heading, distance_m,
            )
        ))
        altitude_losses.append(
            transition_model.altitude_loss_bins(distance_map)
        )

    for source_altitude in range(grid.altitude_count - 1, -1, -1):
        for target_heading_bin, allowed_headings in enumerate(source_headings):
            altitude_loss = altitude_losses[target_heading_bin]
            target_altitude = source_altitude - altitude_loss
            if target_altitude < 0:
                continue
            source_mask = np.any(
                active_4d[source_altitude, :, :, allowed_headings], axis=0,
            )
            source_mask &= ~terminal_positions[source_altitude]
            if not np.any(source_mask):
                continue
            x_offset, y_offset = grid.motion_offset(target_heading_bin)
            source_x_start = max(0, -x_offset)
            source_x_stop = min(grid.x_count, grid.x_count - x_offset)
            source_y_start = max(0, -y_offset)
            source_y_stop = min(grid.y_count, grid.y_count - y_offset)
            if source_x_start >= source_x_stop or source_y_start >= source_y_stop:
                continue
            bounded_source = source_mask[
                source_y_start:source_y_stop,
                source_x_start:source_x_stop,
            ]
            target_x_start = source_x_start + x_offset
            target_x_stop = source_x_stop + x_offset
            target_y_start = source_y_start + y_offset
            target_y_stop = source_y_stop + y_offset
            bounded_source &= node_4d[
                target_altitude,
                target_y_start:target_y_stop,
                target_x_start:target_x_stop,
                target_heading_bin,
            ]
            source_indices_local = np.argwhere(bounded_source)
            if not len(source_indices_local):
                continue
            source_y_values = source_indices_local[:, 0] + source_y_start
            source_x_values = source_indices_local[:, 1] + source_x_start
            source_positions = np.column_stack((
                grid.bounds.x_min
                + source_x_values * grid.horizontal_spacing_map,
                grid.bounds.y_min
                + source_y_values * grid.horizontal_spacing_map,
                np.full(
                    len(source_indices_local),
                    grid.minimum_altitude_map
                    + source_altitude * grid.altitude_spacing_map,
                ),
            ))
            target_positions = source_positions + np.array([
                x_offset * grid.horizontal_spacing_map,
                y_offset * grid.horizontal_spacing_map,
                -altitude_loss * grid.altitude_spacing_map,
            ])
            geometry_valid = ~(
                _contains_solid_many(graph.terrain, source_positions)
                | _segments_intersect_solid_many(
                    graph.terrain, source_positions, target_positions,
                )
            )
            if not np.any(geometry_valid):
                continue
            target_y_values = source_y_values[geometry_valid] + y_offset
            target_x_values = source_x_values[geometry_valid] + x_offset
            active_4d[
                target_altitude,
                target_y_values,
                target_x_values,
                target_heading_bin,
            ] = True
    return active


def _goal_terminal_ids(
    grid: BellmanStateGrid,
    terrain: TerrainModel,
    goal: Point3D,
    *,
    parameters: GliderParameters,
    physical_scale: PhysicalScale,
) -> tuple[int, ...]:
    tolerance_map = parameters.goal_tolerance_m / physical_scale.meters_per_map_unit
    x_values = grid.x_coordinates
    y_values = grid.y_coordinates
    z_values = grid.altitude_coordinates
    x_indices = np.flatnonzero(np.abs(x_values - goal.x) <= tolerance_map + 1.0e-12)
    y_indices = np.flatnonzero(np.abs(y_values - goal.y) <= tolerance_map + 1.0e-12)
    z_indices = np.flatnonzero(np.abs(z_values - goal.z) <= tolerance_map + 1.0e-12)
    result: list[int] = []
    for altitude_index in z_indices:
        for y_index in y_indices:
            for x_index in x_indices:
                representative = BellmanState(
                    int(x_index), int(y_index), int(altitude_index), 0,
                )
                position = grid.position_map(representative)
                if terrain.contains_solid(position) or not is_goal_terminal(
                    position,
                    goal,
                    physical_scale=physical_scale,
                    goal_tolerance_m=parameters.goal_tolerance_m,
                ):
                    continue
                result.extend(
                    grid.encode(BellmanState(
                        int(x_index), int(y_index), int(altitude_index), heading_bin,
                    ))
                    for heading_bin in range(grid.heading_bin_count)
                )
    return tuple(sorted(result))


def _contains_solid_many(terrain: TerrainModel, points: np.ndarray) -> np.ndarray:
    accelerated = getattr(terrain, "contains_solid_many", None)
    if callable(accelerated):
        return np.asarray(accelerated(points), dtype=bool)
    return np.asarray([terrain.contains_solid(point) for point in points], dtype=bool)


def _segments_intersect_solid_many(
    terrain: TerrainModel,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    accelerated = getattr(terrain, "segments_intersect_solid_many", None)
    if callable(accelerated):
        return np.asarray(accelerated(starts, ends), dtype=bool)
    return np.asarray([
        terrain.segment_intersects_solid(start, end)
        for start, end in zip(starts, ends)
    ], dtype=bool)


def build_goal_backward_reachable_graph(
    grid: BellmanStateGrid,
    terrain: TerrainModel,
    goal: Point3D,
    *,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
) -> SparseReachabilityRun:
    """Build the exact finite goal-reachable set using inverse transitions."""
    if not isinstance(grid, BellmanStateGrid):
        raise TypeError("grid must be a BellmanStateGrid")
    if not isinstance(goal, Point3D):
        raise TypeError("goal must be a Point3D")
    started = perf_counter()
    transition_model = GlideTransitionModel(
        grid, terrain, parameters, physical_scale,
    )
    shape = (
        grid.altitude_count,
        grid.y_count,
        grid.x_count,
        grid.heading_bin_count,
    )
    reachable_4d = np.zeros(shape, dtype=bool)
    terminal_4d = np.zeros(shape, dtype=bool)
    value_4d = np.full(shape, np.inf, dtype=float)
    policy_4d = np.full(shape, -1, dtype=np.int64)
    terminal_ids = _goal_terminal_ids(
        grid, terrain, goal,
        parameters=parameters, physical_scale=physical_scale,
    )
    if not terminal_ids:
        raise RuntimeError("goal tolerance contains no admissible lattice state")
    terminal = terminal_4d.reshape(-1)
    reachable = reachable_4d.reshape(-1)
    terminal[np.asarray(terminal_ids, dtype=np.int64)] = True
    reachable[np.asarray(terminal_ids, dtype=np.int64)] = True
    value_4d.reshape(-1)[np.asarray(terminal_ids, dtype=np.int64)] = 0.0

    predecessor_headings: list[tuple[int, ...]] = []
    altitude_losses: list[int] = []
    for target_heading_bin, (x_offset, y_offset) in enumerate(grid.motion_offsets):
        distance_map = grid.horizontal_spacing_map * float(np.hypot(x_offset, y_offset))
        distance_m = physical_scale.distance_m(distance_map)
        target_heading = grid.heading_rad(target_heading_bin)
        predecessor_headings.append(tuple(
            source_heading_bin
            for source_heading_bin in range(grid.heading_bin_count)
            if transition_model.turn_is_feasible(
                grid.heading_rad(source_heading_bin), target_heading, distance_m,
            )
        ))
        altitude_losses.append(transition_model.altitude_loss_bins(distance_map))

    predecessor_candidates = 0
    rejected_by_bounds = 0
    rejected_by_terminal_source = 0
    rejected_by_terrain = 0
    reachable_edge_count = 0
    terminal_positions = np.any(terminal_4d, axis=3)
    for target_altitude in range(grid.altitude_count):
        for target_heading_bin, source_heading_bins in enumerate(predecessor_headings):
            target_mask = reachable_4d[
                target_altitude, :, :, target_heading_bin,
            ]
            target_count = int(np.count_nonzero(target_mask))
            if not target_count:
                continue
            heading_count = len(source_heading_bins)
            predecessor_candidates += target_count * heading_count
            source_altitude = target_altitude + altitude_losses[target_heading_bin]
            if source_altitude >= grid.altitude_count:
                rejected_by_bounds += target_count * heading_count
                continue
            x_offset, y_offset = grid.motion_offset(target_heading_bin)
            source_mask = np.zeros((grid.y_count, grid.x_count), dtype=bool)
            source_x_start = max(0, -x_offset)
            source_x_stop = min(grid.x_count, grid.x_count - x_offset)
            source_y_start = max(0, -y_offset)
            source_y_stop = min(grid.y_count, grid.y_count - y_offset)
            if source_x_start >= source_x_stop or source_y_start >= source_y_stop:
                rejected_by_bounds += target_count * heading_count
                continue
            target_x_start = source_x_start + x_offset
            target_x_stop = source_x_stop + x_offset
            target_y_start = source_y_start + y_offset
            target_y_stop = source_y_stop + y_offset
            source_mask[
                source_y_start:source_y_stop,
                source_x_start:source_x_stop,
            ] = target_mask[
                target_y_start:target_y_stop,
                target_x_start:target_x_stop,
            ]
            in_bounds_count = int(np.count_nonzero(source_mask))
            rejected_by_bounds += (target_count - in_bounds_count) * heading_count
            if not in_bounds_count:
                continue

            terminal_sources = source_mask & terminal_positions[source_altitude]
            terminal_count = int(np.count_nonzero(terminal_sources))
            rejected_by_terminal_source += terminal_count * heading_count
            source_mask &= ~terminal_positions[source_altitude]
            source_indices = np.argwhere(source_mask)
            if not len(source_indices):
                continue
            source_positions = np.column_stack((
                grid.bounds.x_min
                + source_indices[:, 1] * grid.horizontal_spacing_map,
                grid.bounds.y_min
                + source_indices[:, 0] * grid.horizontal_spacing_map,
                np.full(
                    len(source_indices),
                    grid.minimum_altitude_map
                    + source_altitude * grid.altitude_spacing_map,
                ),
            ))
            target_positions = source_positions + np.array([
                x_offset * grid.horizontal_spacing_map,
                y_offset * grid.horizontal_spacing_map,
                -altitude_losses[target_heading_bin]
                * grid.altitude_spacing_map,
            ])
            geometry_valid = ~(
                _contains_solid_many(terrain, source_positions)
                | _segments_intersect_solid_many(
                    terrain, source_positions, target_positions,
                )
            )
            invalid_geometry_count = int(np.count_nonzero(~geometry_valid))
            rejected_by_terrain += invalid_geometry_count * heading_count
            valid_indices = source_indices[geometry_valid]
            if not len(valid_indices):
                continue
            reachable_edge_count += len(valid_indices) * heading_count
            source_y_values = valid_indices[:, 0]
            source_x_values = valid_indices[:, 1]
            target_y_values = source_y_values + y_offset
            target_x_values = source_x_values + x_offset
            target_values = value_4d[
                target_altitude,
                target_y_values,
                target_x_values,
                target_heading_bin,
            ]
            target_ids = (
                (
                    (
                        target_altitude * grid.y_count + target_y_values
                    ) * grid.x_count
                    + target_x_values
                ) * grid.heading_bin_count
                + target_heading_bin
            ).astype(np.int64, copy=False)
            candidate_values = target_values + 1.0
            for source_heading_bin in source_heading_bins:
                current_values = value_4d[
                    source_altitude,
                    source_y_values,
                    source_x_values,
                    source_heading_bin,
                ]
                current_policies = policy_4d[
                    source_altitude,
                    source_y_values,
                    source_x_values,
                    source_heading_bin,
                ]
                improves = (candidate_values < current_values) | (
                    (candidate_values == current_values)
                    & ((current_policies < 0) | (target_ids < current_policies))
                )
                if not np.any(improves):
                    continue
                improved_y = source_y_values[improves]
                improved_x = source_x_values[improves]
                reachable_4d[
                    source_altitude,
                    improved_y,
                    improved_x,
                    source_heading_bin,
                ] = True
                value_4d[
                    source_altitude,
                    improved_y,
                    improved_x,
                    source_heading_bin,
                ] = candidate_values[improves]
                policy_4d[
                    source_altitude,
                    improved_y,
                    improved_x,
                    source_heading_bin,
                ] = target_ids[improves]

    adjacency = ImplicitReachableAdjacency(
        grid,
        terrain,
        reachable,
        terminal,
        parameters=parameters,
        physical_scale=physical_scale,
    )
    empty_predecessors: tuple[tuple[int, ...], ...] = ()
    statistics = GraphStatistics(
        cartesian_state_count=grid.state_count,
        state_count=int(np.count_nonzero(reachable)),
        terrain_excluded_state_count=0,
        valid_edge_count=reachable_edge_count,
        rejected_by_bounds=rejected_by_bounds,
        rejected_by_terrain=rejected_by_terrain,
        rejected_by_turn=0,
        rejected_by_altitude=0,
        terminal_state_count=len(terminal_ids),
    )
    graph = BellmanGraph(
        grid=grid,
        terrain=terrain,
        goal=goal,
        node_mask=reachable,
        terminal_mask=terminal,
        adjacency=adjacency,  # type: ignore[arg-type]
        predecessors=empty_predecessors,
        statistics=statistics,
    )
    unit = UnitCostReachability(
        graph=graph,
        value=value_4d.reshape(-1),
        policy_successor=policy_4d.reshape(-1),
        goal_reachable=reachable.copy(),
    )
    elapsed = perf_counter() - started
    reachable_count = int(np.count_nonzero(reachable))
    metrics = SparseReachabilityMetrics(
        cartesian_state_count=grid.state_count,
        terminal_state_count=len(terminal_ids),
        reachable_state_count=reachable_count,
        reachable_edge_count=reachable_edge_count,
        predecessor_candidates=predecessor_candidates,
        rejected_by_bounds=rejected_by_bounds,
        rejected_by_terminal_source=rejected_by_terminal_source,
        rejected_by_terrain=rejected_by_terrain,
        build_s=elapsed,
        dense_state_fraction=reachable_count / grid.state_count,
    )
    return SparseReachabilityRun(graph=graph, unit_reachability=unit, metrics=metrics)


__all__ = [
    "ImplicitReachableAdjacency",
    "SparseReachabilityMetrics",
    "SparseReachabilityRun",
    "build_forward_descendant_mask",
    "build_goal_backward_reachable_graph",
]
