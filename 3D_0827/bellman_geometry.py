"""Stage-5 glide-edge geometry, turn limits, and terrain rejection."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot

import numpy as np
from numpy.typing import NDArray

from bellman_state import BellmanState, BellmanStateGrid
from energy_model import (
    DEFAULT_GLIDER,
    DEFAULT_PHYSICAL_SCALE,
    GliderParameters,
    PhysicalScale,
)
from map_geometry import TerrainModel


FloatArray = NDArray[np.float64]
def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def wrapped_angle_difference(first_rad: float, second_rad: float) -> float:
    """Return the smallest absolute angular separation."""
    return abs(wrap_angle(float(second_rad) - float(first_rad)))


@dataclass(frozen=True)
class GlideEdge:
    """One valid unit-cost edge in the discrete glide DAG."""

    source_id: int
    target_id: int
    source_state: BellmanState
    target_state: BellmanState
    horizontal_distance_m: float
    altitude_loss_m: float
    duration_s: float
    heading_change_rad: float
    unit_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.target_state.altitude_index >= self.source_state.altitude_index:
            raise ValueError("every glide edge must strictly decrease altitude")
        positive = (
            self.horizontal_distance_m,
            self.altitude_loss_m,
            self.duration_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("edge distance, altitude loss, and duration must be positive")
        if not np.isfinite(self.heading_change_rad) or self.heading_change_rad < 0.0:
            raise ValueError("heading change must be finite and nonnegative")
        if self.unit_cost != 1.0:
            raise ValueError("Stage-5 edge cost must remain exactly one")


@dataclass(frozen=True)
class RejectedTransition:
    """Optional debug record for one rejected successor attempt."""

    source_position_map: FloatArray
    target_position_map: FloatArray
    target_heading_bin: int
    reason: str


@dataclass
class TransitionStatistics:
    considered: int = 0
    valid: int = 0
    rejected_by_bounds: int = 0
    rejected_by_terrain: int = 0
    rejected_by_turn: int = 0
    rejected_by_altitude: int = 0

    def add(self, other: "TransitionStatistics") -> None:
        self.considered += other.considered
        self.valid += other.valid
        self.rejected_by_bounds += other.rejected_by_bounds
        self.rejected_by_terrain += other.rejected_by_terrain
        self.rejected_by_turn += other.rejected_by_turn
        self.rejected_by_altitude += other.rejected_by_altitude


class GlideTransitionModel:
    """Generate terrain-safe, turn-feasible, altitude-decreasing successors."""

    def __init__(
        self,
        grid: BellmanStateGrid,
        terrain: TerrainModel,
        parameters: GliderParameters = DEFAULT_GLIDER,
        physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    ) -> None:
        if not isinstance(grid, BellmanStateGrid):
            raise TypeError("grid must be a BellmanStateGrid")
        self.grid = grid
        self.terrain = terrain
        self.parameters = parameters
        self.physical_scale = physical_scale

    def altitude_loss_bins(self, horizontal_distance_map: float) -> int:
        distance = float(horizontal_distance_map)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("horizontal distance must be finite and positive")
        required_loss_map = distance / self.parameters.best_glide_ratio
        return max(1, int(ceil(
            required_loss_map / self.grid.altitude_spacing_map - 1.0e-12,
        )))

    def turn_is_feasible(
        self,
        current_heading_rad: float,
        target_heading_rad: float,
        horizontal_distance_m: float,
    ) -> bool:
        distance = float(horizontal_distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("horizontal distance must be finite and positive")
        duration = distance / self.parameters.best_glide_speed_mps
        allowed_change = self.parameters.maximum_turn_rate_rad_s * duration
        actual_change = wrapped_angle_difference(
            current_heading_rad,
            target_heading_rad,
        )
        return bool(actual_change <= allowed_change + 1.0e-12)

    def successors(
        self,
        state: BellmanState,
        *,
        include_rejected: bool = False,
    ) -> tuple[tuple[GlideEdge, ...], TransitionStatistics, tuple[RejectedTransition, ...]]:
        if not self.grid.contains_state(state):
            raise ValueError("state lies outside the transition grid")
        source_position = self.grid.position_map(state)
        if self.terrain.contains_solid(source_position):
            raise ValueError("cannot generate successors from inside terrain")

        source_id = self.grid.encode(state)
        current_heading = self.grid.heading_rad(state.heading_bin)
        edges: list[GlideEdge] = []
        rejected: list[RejectedTransition] = []
        statistics = TransitionStatistics()

        for target_heading_bin, (x_offset, y_offset) in enumerate(
            self.grid.motion_offsets,
        ):
            statistics.considered += 1
            target_x_index = state.x_index + x_offset
            target_y_index = state.y_index + y_offset
            horizontal_distance_map = self.grid.horizontal_spacing_map * hypot(
                x_offset,
                y_offset,
            )
            loss_bins = self.altitude_loss_bins(horizontal_distance_map)
            target_altitude_index = state.altitude_index - loss_bins
            target_position = source_position + np.array([
                x_offset * self.grid.horizontal_spacing_map,
                y_offset * self.grid.horizontal_spacing_map,
                -loss_bins * self.grid.altitude_spacing_map,
            ])

            if not (
                0 <= target_x_index < self.grid.x_count
                and 0 <= target_y_index < self.grid.y_count
            ):
                statistics.rejected_by_bounds += 1
                self._record_rejection(
                    rejected,
                    include_rejected,
                    source_position,
                    target_position,
                    target_heading_bin,
                    "map bounds",
                )
                continue

            horizontal_distance_m = self.physical_scale.distance_m(
                horizontal_distance_map,
            )
            target_heading = self.grid.heading_rad(target_heading_bin)
            heading_change = wrapped_angle_difference(
                current_heading,
                target_heading,
            )
            if not self.turn_is_feasible(
                current_heading,
                target_heading,
                horizontal_distance_m,
            ):
                statistics.rejected_by_turn += 1
                self._record_rejection(
                    rejected,
                    include_rejected,
                    source_position,
                    target_position,
                    target_heading_bin,
                    "turn constraint",
                )
                continue

            if target_altitude_index < 0:
                statistics.rejected_by_altitude += 1
                self._record_rejection(
                    rejected,
                    include_rejected,
                    source_position,
                    target_position,
                    target_heading_bin,
                    "insufficient altitude",
                )
                continue

            target_state = BellmanState(
                target_x_index,
                target_y_index,
                target_altitude_index,
                target_heading_bin,
            )
            target_position = self.grid.position_map(target_state)
            if self.terrain.segment_intersects_solid(
                source_position,
                target_position,
            ):
                statistics.rejected_by_terrain += 1
                self._record_rejection(
                    rejected,
                    include_rejected,
                    source_position,
                    target_position,
                    target_heading_bin,
                    "terrain collision",
                )
                continue

            duration_s = (
                horizontal_distance_m / self.parameters.best_glide_speed_mps
            )
            edge = GlideEdge(
                source_id=source_id,
                target_id=self.grid.encode(target_state),
                source_state=state,
                target_state=target_state,
                horizontal_distance_m=horizontal_distance_m,
                altitude_loss_m=self.physical_scale.distance_m(
                    loss_bins * self.grid.altitude_spacing_map,
                ),
                duration_s=duration_s,
                heading_change_rad=heading_change,
            )
            edges.append(edge)
            statistics.valid += 1

        return tuple(edges), statistics, tuple(rejected)

    @staticmethod
    def _record_rejection(
        records: list[RejectedTransition],
        enabled: bool,
        source_position: FloatArray,
        target_position: FloatArray,
        target_heading_bin: int,
        reason: str,
    ) -> None:
        if enabled:
            records.append(RejectedTransition(
                source_position_map=source_position.copy(),
                target_position_map=target_position.copy(),
                target_heading_bin=target_heading_bin,
                reason=reason,
            ))
