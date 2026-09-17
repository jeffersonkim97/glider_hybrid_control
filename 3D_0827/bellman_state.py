"""Discrete Stage-5 glide state grid and physical goal convention."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, gcd, pi
from numbers import Integral
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE, PhysicalScale
from map_geometry import MapBounds
from scenario import Point3D


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, order=True)
class BellmanState:
    """Integer index of one ``(x, y, h, heading)`` grid state."""

    x_index: int
    y_index: int
    altitude_index: int
    heading_bin: int

    def __post_init__(self) -> None:
        values = (
            self.x_index,
            self.y_index,
            self.altitude_index,
            self.heading_bin,
        )
        if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
            raise TypeError("Bellman state indices must be integers")
        if any(int(value) < 0 for value in values):
            raise ValueError("Bellman state indices must be nonnegative")
        object.__setattr__(self, "x_index", int(self.x_index))
        object.__setattr__(self, "y_index", int(self.y_index))
        object.__setattr__(self, "altitude_index", int(self.altitude_index))
        object.__setattr__(self, "heading_bin", int(self.heading_bin))


@dataclass(frozen=True)
class BellmanStateGrid:
    """Uniform map/altitude grid with configurable lattice headings.

    Heading bins are backed by integer motion primitives.  This keeps every
    successor exactly on the Cartesian lattice while allowing the resolution
    study to use wider direction stencils (the canonical radius/count pairs
    are 1/8, 2/16, and 3/32).  ``motion_primitive_step_cells`` independently
    scales their physical horizon when spatial spacing changes.  The default
    1/8/1 stencil is bit-for-bit compatible with the original graph.
    """

    bounds: MapBounds
    horizontal_spacing_map: float = 1.0
    minimum_altitude_map: float = 0.0
    maximum_altitude_map: float = 5.0
    altitude_spacing_map: float = 0.1
    heading_bin_count: int = 8
    motion_primitive_radius: int = 1
    motion_primitive_step_cells: int = 1
    _x_count: int = field(init=False, repr=False)
    _y_count: int = field(init=False, repr=False)
    _altitude_count: int = field(init=False, repr=False)
    _motion_offsets: tuple[tuple[int, int], ...] = field(init=False, repr=False)
    _heading_angles_rad: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, MapBounds):
            raise TypeError("bounds must be MapBounds")
        positive = (
            self.horizontal_spacing_map,
            self.altitude_spacing_map,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("grid spacings must be finite and positive")
        if (
            not np.isfinite(self.minimum_altitude_map)
            or not np.isfinite(self.maximum_altitude_map)
            or self.minimum_altitude_map < 0.0
            or self.minimum_altitude_map >= self.maximum_altitude_map
        ):
            raise ValueError("altitude bounds must be finite, nonnegative, and increasing")
        if (
            not isinstance(self.heading_bin_count, Integral)
            or isinstance(self.heading_bin_count, bool)
            or int(self.heading_bin_count) < 4
        ):
            raise ValueError("heading_bin_count must be an integer of at least four")
        if (
            not isinstance(self.motion_primitive_radius, Integral)
            or isinstance(self.motion_primitive_radius, bool)
            or int(self.motion_primitive_radius) < 1
        ):
            raise ValueError("motion_primitive_radius must be a positive integer")
        if (
            not isinstance(self.motion_primitive_step_cells, Integral)
            or isinstance(self.motion_primitive_step_cells, bool)
            or int(self.motion_primitive_step_cells) < 1
        ):
            raise ValueError(
                "motion_primitive_step_cells must be a positive integer"
            )
        offsets = self._build_motion_offsets(
            int(self.heading_bin_count), int(self.motion_primitive_radius),
            int(self.motion_primitive_step_cells),
        )
        x_count = self._aligned_count(
            self.bounds.x_min,
            self.bounds.x_max,
            self.horizontal_spacing_map,
            "x bounds",
        )
        y_count = self._aligned_count(
            self.bounds.y_min,
            self.bounds.y_max,
            self.horizontal_spacing_map,
            "y bounds",
        )
        altitude_count = self._aligned_count(
            self.minimum_altitude_map,
            self.maximum_altitude_map,
            self.altitude_spacing_map,
            "altitude bounds",
        )
        object.__setattr__(self, "_x_count", x_count)
        object.__setattr__(self, "_y_count", y_count)
        object.__setattr__(self, "_altitude_count", altitude_count)
        object.__setattr__(self, "heading_bin_count", int(self.heading_bin_count))
        object.__setattr__(self, "motion_primitive_radius", int(self.motion_primitive_radius))
        object.__setattr__(
            self,
            "motion_primitive_step_cells",
            int(self.motion_primitive_step_cells),
        )
        object.__setattr__(self, "_motion_offsets", offsets)
        object.__setattr__(self, "_heading_angles_rad", tuple(
            float(atan2(y_offset, x_offset)) for x_offset, y_offset in offsets
        ))

    @staticmethod
    def _build_motion_offsets(
        heading_bin_count: int,
        radius: int,
        step_cells: int,
    ) -> tuple[tuple[int, int], ...]:
        """Select deterministic primitive integer directions near uniform bins."""
        candidates = [
            (x_offset, y_offset)
            for y_offset in range(-radius, radius + 1)
            for x_offset in range(-radius, radius + 1)
            if (x_offset or y_offset)
            and gcd(abs(x_offset), abs(y_offset)) == 1
        ]
        if len(candidates) < heading_bin_count:
            raise ValueError(
                "motion_primitive_radius provides fewer unique lattice "
                "directions than heading_bin_count"
            )
        available = set(candidates)
        selected: list[tuple[int, int]] = []
        for heading_index in range(heading_bin_count):
            target = 2.0 * pi * heading_index / heading_bin_count

            def ranking(offset: tuple[int, int]) -> tuple[float, int, int, int]:
                angle = atan2(offset[1], offset[0]) % (2.0 * pi)
                difference = abs((angle - target + pi) % (2.0 * pi) - pi)
                return (
                    difference,
                    offset[0] * offset[0] + offset[1] * offset[1],
                    -offset[0],
                    -offset[1],
                )

            chosen = min(available, key=ranking)
            selected.append(chosen)
            available.remove(chosen)
        return tuple(
            (step_cells * x_offset, step_cells * y_offset)
            for x_offset, y_offset in selected
        )

    @staticmethod
    def _aligned_count(
        minimum: float,
        maximum: float,
        spacing: float,
        name: str,
    ) -> int:
        interval_count = (maximum - minimum) / spacing
        rounded = int(round(interval_count))
        if not np.isclose(interval_count, rounded, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{name} must align exactly with its grid spacing")
        return rounded + 1

    @property
    def x_count(self) -> int:
        return self._x_count

    @property
    def y_count(self) -> int:
        return self._y_count

    @property
    def altitude_count(self) -> int:
        return self._altitude_count

    @property
    def state_count(self) -> int:
        return self.x_count * self.y_count * self.altitude_count * self.heading_bin_count

    @property
    def x_coordinates(self) -> FloatArray:
        return np.linspace(self.bounds.x_min, self.bounds.x_max, self.x_count)

    @property
    def y_coordinates(self) -> FloatArray:
        return np.linspace(self.bounds.y_min, self.bounds.y_max, self.y_count)

    @property
    def altitude_coordinates(self) -> FloatArray:
        return np.linspace(
            self.minimum_altitude_map,
            self.maximum_altitude_map,
            self.altitude_count,
        )

    def contains_state(self, state: BellmanState) -> bool:
        return bool(
            0 <= state.x_index < self.x_count
            and 0 <= state.y_index < self.y_count
            and 0 <= state.altitude_index < self.altitude_count
            and 0 <= state.heading_bin < self.heading_bin_count
        )

    def encode(self, state: BellmanState) -> int:
        """Encode using altitude-major, then y/x/heading deterministic order."""
        if not isinstance(state, BellmanState):
            raise TypeError("state must be a BellmanState")
        if not self.contains_state(state):
            raise ValueError("state indices lie outside this grid")
        return int((
            (
                state.altitude_index * self.y_count
                + state.y_index
            )
            * self.x_count
            + state.x_index
        ) * self.heading_bin_count + state.heading_bin)

    def decode(self, state_id: int) -> BellmanState:
        if not isinstance(state_id, Integral) or isinstance(state_id, bool):
            raise TypeError("state_id must be an integer")
        remaining = int(state_id)
        if not 0 <= remaining < self.state_count:
            raise ValueError("state_id lies outside this grid")
        heading_bin = remaining % self.heading_bin_count
        remaining //= self.heading_bin_count
        x_index = remaining % self.x_count
        remaining //= self.x_count
        y_index = remaining % self.y_count
        altitude_index = remaining // self.y_count
        return BellmanState(x_index, y_index, altitude_index, heading_bin)

    def position_map(self, state: BellmanState) -> FloatArray:
        if not self.contains_state(state):
            raise ValueError("state indices lie outside this grid")
        return np.array([
            self.bounds.x_min + state.x_index * self.horizontal_spacing_map,
            self.bounds.y_min + state.y_index * self.horizontal_spacing_map,
            self.minimum_altitude_map
            + state.altitude_index * self.altitude_spacing_map,
        ])

    def heading_rad(self, heading_bin: int) -> float:
        if not isinstance(heading_bin, Integral) or isinstance(heading_bin, bool):
            raise TypeError("heading_bin must be an integer")
        if not 0 <= int(heading_bin) < self.heading_bin_count:
            raise ValueError("heading_bin lies outside this grid")
        return self._heading_angles_rad[int(heading_bin)]

    @property
    def motion_offsets(self) -> tuple[tuple[int, int], ...]:
        """Integer lattice displacement associated with every heading bin."""
        return self._motion_offsets

    def motion_offset(self, heading_bin: int) -> tuple[int, int]:
        if not isinstance(heading_bin, Integral) or isinstance(heading_bin, bool):
            raise TypeError("heading_bin must be an integer")
        if not 0 <= int(heading_bin) < self.heading_bin_count:
            raise ValueError("heading_bin lies outside this grid")
        return self._motion_offsets[int(heading_bin)]

    def iter_states(self) -> Iterator[BellmanState]:
        for altitude_index in range(self.altitude_count):
            for y_index in range(self.y_count):
                for x_index in range(self.x_count):
                    for heading_bin in range(self.heading_bin_count):
                        yield BellmanState(
                            x_index,
                            y_index,
                            altitude_index,
                            heading_bin,
                        )


def is_goal_terminal(
    position_map: FloatArray,
    goal: Point3D | FloatArray,
    *,
    physical_scale: PhysicalScale = DEFAULT_PHYSICAL_SCALE,
    goal_tolerance_m: float = DEFAULT_GLIDER.goal_tolerance_m,
) -> bool:
    """Use an inclusive physical three-dimensional goal ball."""
    position = np.asarray(position_map, dtype=float)
    goal_position = goal.as_array() if isinstance(goal, Point3D) else np.asarray(goal, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_map must contain three finite coordinates")
    if goal_position.shape != (3,) or not np.all(np.isfinite(goal_position)):
        raise ValueError("goal must contain three finite coordinates")
    tolerance = float(goal_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("goal_tolerance_m must be finite and nonnegative")
    distance_m = physical_scale.distance_m(
        float(np.linalg.norm(position - goal_position)),
    )
    return bool(distance_m <= tolerance + 1.0e-12)
