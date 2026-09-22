"""Vectorized additive Bellman solve on a two-sided reachable corridor."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from additive_bellman import AdditiveBellmanSolution
from bellman_geometry import GlideTransitionModel
from bellman_graph import BellmanGraph
from detection_hazard import AttackerHazardTimeParameters, GlideDetectionHazardModel
from edge_hazard import HazardPrecomputation, integrate_edge_hazard
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, GliderParameters, PhysicalScale
from solver_metrics import HazardTiming
from sparse_reachability import build_forward_descendant_mask


class _LazyHazardRows(Sequence[tuple[float, ...]]):
    """Recompute only requested active rows; no global Python-edge table."""

    def __init__(
        self,
        graph: BellmanGraph,
        active_mask: np.ndarray,
        hazard_field: GlideDetectionHazardModel,
        quadrature_resolution: int,
        physical_scale: PhysicalScale,
    ) -> None:
        self.graph = graph
        self.active_mask = active_mask
        self.hazard_field = hazard_field
        self.quadrature_resolution = quadrature_resolution
        self.physical_scale = physical_scale
        self._cache: dict[int, tuple[float, ...]] = {}

    def __len__(self) -> int:
        return self.graph.grid.state_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        state_id = int(index)
        if not 0 <= state_id < len(self):
            raise IndexError(state_id)
        if not self.active_mask[state_id]:
            return ()
        cached = self._cache.get(state_id)
        if cached is not None:
            return cached
        result = tuple(
            integrate_edge_hazard(
                edge,
                self.graph.grid,
                self.hazard_field,
                quadrature_resolution=self.quadrature_resolution,
                physical_scale=self.physical_scale,
            ).hazard
            for edge in self.graph.adjacency[state_id]
        )
        self._cache[state_id] = result
        return result

    def __iter__(self) -> Iterator[tuple[float, ...]]:
        for state_id in range(len(self)):
            yield self[state_id]


class _LazyCostRows(Sequence[tuple[float, ...]]):
    def __init__(
        self,
        graph: BellmanGraph,
        hazards: _LazyHazardRows,
        parameters: AttackerHazardTimeParameters,
    ) -> None:
        self.graph = graph
        self.hazards = hazards
        self.parameters = parameters

    def __len__(self) -> int:
        return self.graph.grid.state_count

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        state_id = int(index)
        edges = self.graph.adjacency[state_id]
        hazards = self.hazards[state_id]
        return tuple(
            self.parameters.hazard_weight * hazard
            / self.parameters.hazard_reference
            + self.parameters.time_weight * edge.duration_s
            / self.parameters.time_reference_s
            for edge, hazard in zip(edges, hazards)
        )

    def __iter__(self) -> Iterator[tuple[float, ...]]:
        for state_id in range(len(self)):
            yield self[state_id]


@dataclass(frozen=True)
class SparseAdditiveMetrics:
    active_state_count: int
    active_edge_count: int
    visibility_sample_count: int
    corridor_s: float
    solve_s: float


@dataclass(frozen=True)
class SparseAdditiveRun:
    solution: AdditiveBellmanSolution
    hazards: HazardPrecomputation
    metrics: SparseAdditiveMetrics


def _batch_edge_hazard(
    starts: np.ndarray,
    ends: np.ndarray,
    duration_s: float | np.ndarray,
    hazard_field: GlideDetectionHazardModel,
    quadrature_resolution: int,
    physical_scale: PhysicalScale,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of the Stage-7 trapezoidal edge integral.

    ``duration_s`` may be one value for the whole batch, as the sweep uses it -
    every edge there shares a motion offset - or one value per edge, which lets a
    caller batch edges that leave the same state along different offsets.
    """
    if not len(starts):
        return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)
    fractions = np.linspace(0.0, 1.0, quadrature_resolution)
    durations = np.broadcast_to(
        np.asarray(duration_s, dtype=float).reshape(-1), (len(starts),),
    )
    spacing_s = durations / (quadrature_resolution - 1)
    weights = np.repeat(spacing_s[:, None], quadrature_resolution, axis=1)
    weights[:, 0] *= 0.5
    weights[:, -1] *= 0.5
    displacement = ends - starts
    positions = (
        starts[:, None, :]
        + fractions[None, :, None] * displacement[:, None, :]
    )
    velocity = (
        displacement * physical_scale.meters_per_map_unit / durations[:, None]
    )
    sensor = hazard_field.sensor.as_array()
    sensor_delta_m = (
        sensor[None, None, :] - positions
    ) * physical_scale.meters_per_map_unit
    slant_range = np.linalg.norm(sensor_delta_m, axis=2)
    sensor_range = np.maximum(
        slant_range, hazard_field.parameters.range_floor_m,
    )
    los_unit = sensor_delta_m / sensor_range[:, :, None]
    radial_velocity = np.sum(velocity[:, None, :] * los_unit, axis=2)
    speed = np.linalg.norm(velocity, axis=1)
    cosine_aspect = np.clip(
        radial_velocity / np.maximum(speed[:, None], 1.0e-9), -1.0, 1.0,
    )
    radar_cross_section = (
        hazard_field.parameters.rcs_min
        + (
            hazard_field.parameters.rcs_max
            - hazard_field.parameters.rcs_min
        ) * cosine_aspect**2
    )
    flattened = positions.reshape(-1, 3)
    sensor_ends = np.repeat(sensor[None, :], len(flattened), axis=0)
    batch_visibility = getattr(
        hazard_field.terrain, "segments_intersect_solid_many", None,
    )
    if callable(batch_visibility):
        visible = ~np.asarray(
            batch_visibility(flattened, sensor_ends), dtype=bool,
        ).reshape(len(starts), quadrature_resolution)
    else:
        visible = np.asarray([
            not hazard_field.terrain.segment_intersects_solid(point, sensor)
            for point in flattened
        ], dtype=bool).reshape(len(starts), quadrature_resolution)
    inverse_range_fourth = 1.0 / sensor_range**4
    radar_rate = (
        visible
        * hazard_field.parameters.radar_rate_scale
        * hazard_field.parameters.radar_coefficient
        * radar_cross_section
        * inverse_range_fourth
    )
    doppler_rate = (
        visible
        * hazard_field.parameters.radial_velocity_rate_scale
        * hazard_field.parameters.doppler_coefficient
        * radial_velocity**2
        * inverse_range_fourth
    )
    hazard = np.sum((radar_rate + doppler_rate) * weights, axis=1)
    return hazard, np.count_nonzero(visible, axis=1).astype(np.int64)


def solve_sparse_additive_bellman(
    graph: BellmanGraph,
    start_state_ids: Sequence[int],
    hazard_field: GlideDetectionHazardModel,
    *,
    objective_parameters: AttackerHazardTimeParameters,
    parameters: GliderParameters = DEFAULT_GLIDER,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    quadrature_resolution: int = 8,
    tie_tolerance: float = 1.0e-12,
) -> SparseAdditiveRun:
    """Solve exact hazard/time cost only on start-to-goal corridor states."""
    if not isinstance(graph, BellmanGraph):
        raise TypeError("graph must be a BellmanGraph")
    if not isinstance(hazard_field, GlideDetectionHazardModel):
        raise TypeError("sparse solve requires GlideDetectionHazardModel")
    if not isinstance(quadrature_resolution, int) or quadrature_resolution < 2:
        raise ValueError("quadrature_resolution must be an integer >= 2")
    corridor_start = perf_counter()
    active = build_forward_descendant_mask(
        graph,
        start_state_ids,
        parameters=parameters,
        physical_scale=physical_scale,
    )
    corridor_s = perf_counter() - corridor_start
    grid = graph.grid
    shape = (
        grid.altitude_count,
        grid.y_count,
        grid.x_count,
        grid.heading_bin_count,
    )
    active_4d = active.reshape(shape)
    terminal_4d = graph.terminal_mask.reshape(shape)
    value_4d = np.full(shape, np.inf, dtype=float)
    policy_4d = np.full(shape, -1, dtype=np.int64)
    action_4d = np.full(shape, -1, dtype=np.int64)
    value_4d[terminal_4d & active_4d] = 0.0
    transition_model = GlideTransitionModel(
        grid, graph.terrain, parameters, physical_scale,
    )
    allowed_sources: list[tuple[int, ...]] = []
    altitude_losses: list[int] = []
    durations: list[float] = []
    for target_heading, (x_offset, y_offset) in enumerate(grid.motion_offsets):
        distance_map = grid.horizontal_spacing_map * float(
            np.hypot(x_offset, y_offset)
        )
        distance_m = physical_scale.distance_m(distance_map)
        target_angle = grid.heading_rad(target_heading)
        allowed_sources.append(tuple(
            source_heading
            for source_heading in range(grid.heading_bin_count)
            if transition_model.turn_is_feasible(
                grid.heading_rad(source_heading), target_angle, distance_m,
            )
        ))
        altitude_losses.append(transition_model.altitude_loss_bins(distance_map))
        durations.append(distance_m / parameters.best_glide_speed_mps)

    edge_count = 0
    visibility_count = 0
    hazard_s = 0.0
    solve_start = perf_counter()
    for source_altitude in range(grid.altitude_count):
        for target_heading, source_headings in enumerate(allowed_sources):
            target_altitude = source_altitude - altitude_losses[target_heading]
            if target_altitude < 0:
                continue
            x_offset, y_offset = grid.motion_offset(target_heading)
            source_x_start = max(0, -x_offset)
            source_x_stop = min(grid.x_count, grid.x_count - x_offset)
            source_y_start = max(0, -y_offset)
            source_y_stop = min(grid.y_count, grid.y_count - y_offset)
            if source_x_start >= source_x_stop or source_y_start >= source_y_stop:
                continue
            target_x_start = source_x_start + x_offset
            target_x_stop = source_x_stop + x_offset
            target_y_start = source_y_start + y_offset
            target_y_stop = source_y_stop + y_offset
            target_values = value_4d[
                target_altitude,
                target_y_start:target_y_stop,
                target_x_start:target_x_stop,
                target_heading,
            ]
            source_union = np.any(
                active_4d[
                    source_altitude,
                    source_y_start:source_y_stop,
                    source_x_start:source_x_stop,
                    source_headings,
                ],
                axis=0,
            )
            geometry_mask = source_union & np.isfinite(target_values)
            local_indices = np.argwhere(geometry_mask)
            if not len(local_indices):
                continue
            source_y = local_indices[:, 0] + source_y_start
            source_x = local_indices[:, 1] + source_x_start
            target_y = source_y + y_offset
            target_x = source_x + x_offset
            starts = np.column_stack((
                grid.bounds.x_min + source_x * grid.horizontal_spacing_map,
                grid.bounds.y_min + source_y * grid.horizontal_spacing_map,
                np.full(
                    len(local_indices),
                    grid.minimum_altitude_map
                    + source_altitude * grid.altitude_spacing_map,
                ),
            ))
            ends = np.column_stack((
                grid.bounds.x_min + target_x * grid.horizontal_spacing_map,
                grid.bounds.y_min + target_y * grid.horizontal_spacing_map,
                np.full(
                    len(local_indices),
                    grid.minimum_altitude_map
                    + target_altitude * grid.altitude_spacing_map,
                ),
            ))
            hazard_start = perf_counter()
            hazards, visible_samples = _batch_edge_hazard(
                starts,
                ends,
                durations[target_heading],
                hazard_field,
                quadrature_resolution,
                physical_scale,
            )
            hazard_s += perf_counter() - hazard_start
            target_ids = (
                ((target_altitude * grid.y_count + target_y) * grid.x_count + target_x)
                * grid.heading_bin_count
                + target_heading
            ).astype(np.int64, copy=False)
            downstream_values = value_4d[
                target_altitude, target_y, target_x, target_heading,
            ]
            edge_costs = (
                objective_parameters.hazard_weight * hazards
                / objective_parameters.hazard_reference
                + objective_parameters.time_weight * durations[target_heading]
                / objective_parameters.time_reference_s
            )
            candidates = edge_costs + downstream_values
            for source_heading in source_headings:
                eligible = active_4d[
                    source_altitude, source_y, source_x, source_heading,
                ]
                if not np.any(eligible):
                    continue
                y_values = source_y[eligible]
                x_values = source_x[eligible]
                candidate_values = candidates[eligible]
                candidate_ids = target_ids[eligible]
                current_values = value_4d[
                    source_altitude, y_values, x_values, source_heading,
                ]
                current_actions = action_4d[
                    source_altitude, y_values, x_values, source_heading,
                ]
                current_policies = policy_4d[
                    source_altitude, y_values, x_values, source_heading,
                ]
                improves = candidate_values < current_values - tie_tolerance
                ties = np.abs(candidate_values - current_values) <= tie_tolerance
                lower_tie = ties & (
                    (current_actions < 0)
                    | (target_heading < current_actions)
                    | (
                        (target_heading == current_actions)
                        & (candidate_ids < current_policies)
                    )
                )
                update = improves | lower_tie
                if np.any(update):
                    update_y = y_values[update]
                    update_x = x_values[update]
                    value_4d[
                        source_altitude, update_y, update_x, source_heading,
                    ] = candidate_values[update]
                    policy_4d[
                        source_altitude, update_y, update_x, source_heading,
                    ] = candidate_ids[update]
                    action_4d[
                        source_altitude, update_y, update_x, source_heading,
                    ] = target_heading
                edge_count += int(np.count_nonzero(eligible))
                visibility_count += int(np.sum(visible_samples[eligible]))
    solve_s = perf_counter() - solve_start
    value = value_4d.reshape(-1)
    reachable = active & np.isfinite(value)
    lazy_hazards = _LazyHazardRows(
        graph,
        active,
        hazard_field,
        quadrature_resolution,
        physical_scale,
    )
    lazy_costs = _LazyCostRows(graph, lazy_hazards, objective_parameters)
    solution = AdditiveBellmanSolution(
        graph=graph,
        edge_cost_by_source=lazy_costs,  # type: ignore[arg-type]
        active_mask=active,
        value=value,
        policy_successor=policy_4d.reshape(-1),
        policy_action_index=action_4d.reshape(-1),
        goal_reachable=reachable,
        tie_tolerance=tie_tolerance,
    )
    hazards_result = HazardPrecomputation(
        hazard_by_source=lazy_hazards,  # type: ignore[arg-type]
        timing=HazardTiming(
            precompute_s=hazard_s,
            per_edge_s=hazard_s / edge_count if edge_count else 0.0,
            edge_count=edge_count,
            quadrature_resolution=quadrature_resolution,
        ),
        visibility_sample_count=visibility_count,
    )
    return SparseAdditiveRun(
        solution=solution,
        hazards=hazards_result,
        metrics=SparseAdditiveMetrics(
            active_state_count=int(np.count_nonzero(active)),
            active_edge_count=edge_count,
            visibility_sample_count=visibility_count,
            corridor_s=corridor_s,
            solve_s=solve_s,
        ),
    )


__all__ = [
    "SparseAdditiveMetrics",
    "SparseAdditiveRun",
    "solve_sparse_additive_bellman",
]
