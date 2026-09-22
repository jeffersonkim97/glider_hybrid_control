"""Stage-15 two-dimensional Defender placement grid.

Stage 14 fixed the Defender to six points on a line (``defender_grid_config``:
x in {5,...,10}, y = 0, z = 0).  Stage 15 replaces that with every lattice point
of the Defender-feasible region, so that the number of Defender evaluations
becomes a controlled experimental variable rather than a constant.

Region (docs/exact_vs_rl_decision_framework.md 2):

    x   from the terrain's maximum x to the goal's x
    y   the full graph-bounds span
    z   0 (ground); a terrain-following z is deferred to the 3D stage

The Defender lattice is the state x-y lattice - there is no separate spacing
parameter.  Refining the state grid therefore refines Defender placement too.
This couples ``N_D`` to resolution for the global oracle, but NOT for local SSE:
``r_neighbor`` is measured in grid indices, so a radius-r neighbourhood holds
``(2r+1)^2`` actions at every resolution.  That is what keeps the resolution
axis and the Defender-search axis separable.

``defender_grid_config`` is deliberately left untouched: Stage-14 contracts
import ``DEFENDER_ACTION_COUNT`` from it and their frozen gates must keep
passing.  Nothing here modifies or shadows those symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from local_sse_contract import DefenderGridTopology
from map_geometry import MapBounds


SCHEMA_VERSION = "stage15-defender-grid-2d-v1"
DEFENDER_Z_MAP = 0.0
SPAN_INTEGRALITY_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class DefenderRegion:
    """Axis-aligned Defender-feasible rectangle in map units, at z = 0."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_min", "y_max"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Defender region must have positive extent on both axes")

    @property
    def x_span(self) -> float:
        return self.x_max - self.x_min

    @property
    def y_span(self) -> float:
        return self.y_max - self.y_min

    def as_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min, "x_max": self.x_max,
            "y_min": self.y_min, "y_max": self.y_max, "z": DEFENDER_Z_MAP,
        }


def inside_obstacle_footprint(terrain: Any, x_map: float, y_map: float) -> bool:
    """Whether a ground point stands on the plan footprint of a terrain solid.

    ``contains_solid`` is not enough on its own here.  The Defender sits at z = 0,
    which is the base plane of every box, and a point exactly on a face counts as
    outside - so a sensor placed at the very foot of an obstacle passes that test
    while being, physically, right against it.  The LOS tracer then has part of the
    box behind the sensor and cannot trace a silhouette at all; measured at 50 m,
    nine such positions per terrain raised rather than returning a contour, and a
    Defender search that stepped onto one recorded a numerical failure and could
    certify nothing.  Those are not sensor placements the model means to offer, so
    they are excluded from the lattice rather than left to fail later.
    """
    for box in terrain.obstacle_boxes():
        x_low, x_high = box.x_limits
        y_low, y_high = box.y_limits
        if x_low <= float(x_map) <= x_high and y_low <= float(y_map) <= y_high:
            return True
    return False


def terrain_maximum_x_map(terrain: Any) -> float:
    """Largest x reached by any terrain solid, in map units."""
    boxes = tuple(terrain.obstacle_boxes())
    if not boxes:
        raise ValueError("terrain exposes no obstacle boxes")
    return max(float(box.center_x) + float(box.width_x) / 2.0 for box in boxes)


def defender_region_from_scene(
    terrain: Any,
    goal_x_map: float,
    bounds: MapBounds,
    *,
    goal_margin_map: float = 0.0,
) -> DefenderRegion:
    """Region between the terrain's far edge and the goal, spanning all y.

    ``goal_margin_map`` pulls the far edge back from the goal.  With the sensor
    allowed to sit on the goal the game is degenerate - detection is certain, and
    the accumulated hazard (127 at 100 m) swamps the time term so completely that
    the objective stops distinguishing trajectories at all.  One map unit of
    margin is enough: measured at 100 m, hazard drops 127 -> 0.28 and its share
    of the objective goes 99.7% -> 42.8%, which is the balance the weights were
    chosen for.
    """
    x_min = terrain_maximum_x_map(terrain)
    x_max = float(goal_x_map) - float(goal_margin_map)
    if x_max <= x_min:
        raise ValueError(
            "goal x must lie beyond the terrain maximum x for a nonempty Defender region"
        )
    return DefenderRegion(
        x_min=x_min, x_max=x_max,
        y_min=float(bounds.y_min), y_max=float(bounds.y_max),
    )


def _axis_count(span: float, spacing: float, axis: str) -> int:
    intervals = span / spacing
    rounded = int(round(intervals))
    if abs(intervals - rounded) > SPAN_INTEGRALITY_TOLERANCE * max(1.0, intervals):
        raise ValueError(
            f"{axis} span {span!r} is not an integer multiple of spacing {spacing!r}; "
            "the Defender lattice must align with the state lattice"
        )
    return rounded + 1


@dataclass(frozen=True)
class DefenderGrid2D:
    """Ordered Defender action set on the x-y lattice, with its topology."""

    region: DefenderRegion
    spacing_map: float
    x_values_map: tuple[float, ...]
    y_values_map: tuple[float, ...]
    grid_indices: tuple[tuple[int, int], ...]
    positions_map: tuple[tuple[float, float, float], ...]
    excluded_solid_count: int

    @property
    def action_count(self) -> int:
        return len(self.positions_map)

    @property
    def x_count(self) -> int:
        return len(self.x_values_map)

    @property
    def y_count(self) -> int:
        return len(self.y_values_map)

    @property
    def action_ids(self) -> tuple[int, ...]:
        return tuple(range(self.action_count))

    def topology(self) -> DefenderGridTopology:
        """Integer-index topology consumed by the existing local-SSE search.

        ``DefenderGridTopology.neighbors`` already applies the Chebyshev rule
        ``0 < max(|p-i|, |q-j|) <= r`` in any dimension, so no new neighbourhood
        logic is introduced here.
        """
        return DefenderGridTopology(
            tuple(
                (action_id, indices)
                for action_id, indices in zip(self.action_ids, self.grid_indices)
            )
        )

    def position(self, action_id: int) -> tuple[float, float, float]:
        return self.positions_map[int(action_id)]

    def nearest_action_id(self, x_map: float, y_map: float) -> int:
        """Action closest to a map position; used to seed the local search."""
        target = np.array([float(x_map), float(y_map)], dtype=float)
        positions = np.asarray([(x, y) for x, y, _ in self.positions_map], dtype=float)
        return int(np.argmin(np.linalg.norm(positions - target, axis=1)))

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "region_map": self.region.as_dict(),
            "spacing_map": self.spacing_map,
            "x_count": self.x_count,
            "y_count": self.y_count,
            "action_count": self.action_count,
            "excluded_solid_count": self.excluded_solid_count,
            "ordering": "row-major over (x index, y index), action_id = position in that order",
            "z_policy": "fixed ground plane z = 0; terrain-following z deferred",
            "lattice_policy": "Defender lattice is the state x-y lattice; no separate spacing",
        }


def build_defender_grid_2d(
    region: DefenderRegion,
    spacing_map: float,
    *,
    terrain: Any | None = None,
) -> DefenderGrid2D:
    """Enumerate every lattice point of the region, dropping terrain interiors."""
    spacing = float(spacing_map)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_map must be finite and positive")
    x_count = _axis_count(region.x_span, spacing, "x")
    y_count = _axis_count(region.y_span, spacing, "y")
    x_values = tuple(
        float(value) for value in np.linspace(region.x_min, region.x_max, x_count)
    )
    y_values = tuple(
        float(value) for value in np.linspace(region.y_min, region.y_max, y_count)
    )

    grid_indices: list[tuple[int, int]] = []
    positions: list[tuple[float, float, float]] = []
    excluded = 0
    for x_index, x_value in enumerate(x_values):
        for y_index, y_value in enumerate(y_values):
            point = (x_value, y_value, DEFENDER_Z_MAP)
            if terrain is not None and (
                terrain.contains_solid(np.array(point, dtype=float))
                or inside_obstacle_footprint(terrain, x_value, y_value)
            ):
                excluded += 1
                continue
            grid_indices.append((x_index, y_index))
            positions.append(point)
    if not positions:
        raise ValueError("the Defender region contains no admissible lattice points")
    return DefenderGrid2D(
        region=region,
        spacing_map=spacing,
        x_values_map=x_values,
        y_values_map=y_values,
        grid_indices=tuple(grid_indices),
        positions_map=tuple(positions),
        excluded_solid_count=excluded,
    )


def defender_grid_for_resolution(
    terrain: Any,
    goal_x_map: float,
    bounds: MapBounds,
    spatial_resolution_m: float,
    *,
    meters_per_map_unit: float = 100.0,
    goal_margin_map: float = 0.0,
) -> DefenderGrid2D:
    """Convenience constructor keyed by the sweep's physical resolution."""
    resolution = float(spatial_resolution_m)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("spatial_resolution_m must be finite and positive")
    region = defender_region_from_scene(
        terrain, goal_x_map, bounds, goal_margin_map=goal_margin_map,
    )
    return build_defender_grid_2d(
        region, resolution / float(meters_per_map_unit), terrain=terrain,
    )


def neighborhood_size(radius: int) -> int:
    """Chebyshev neighbourhood cardinality in 2D, excluding the centre."""
    if radius < 1:
        raise ValueError("radius must be at least one")
    return (2 * int(radius) + 1) ** 2 - 1


def saturating_radius(grid: DefenderGrid2D) -> int:
    """Radius at which EVERY action's neighbourhood spans the whole grid.

    This is the guaranteed bound, matching
    ``DefenderGridTopology.radius_covers_all_actions`` which quantifies over all
    action IDs.  Use it when the starting action is unknown.
    """
    return int(max(grid.x_count, grid.y_count)) - 1


def centre_saturating_radius(grid: DefenderGrid2D) -> int:
    """Radius at which a search STARTED AT THE CENTRE already sees everything.

    Half the guaranteed bound, because the centre is equidistant from both
    edges.  This is the practically relevant one: the Stage-15 local search is
    seeded at a configured action, and once r reaches this value the first
    iteration already enumerates every action, making "local" SSE a global
    oracle in all but name.  Such points must be recorded as saturated rather
    than plotted as local-SSE measurements.
    """
    return -(-saturating_radius(grid) // 2)


def saturates_from(
    grid: DefenderGrid2D, action_id: int, radius: int,
) -> bool:
    """Exact test for one seed action, using the existing topology rule."""
    from local_sse_contract import DefenderNeighborhoodConfig

    topology = grid.topology()
    neighbours = topology.neighbors(
        int(action_id), DefenderNeighborhoodConfig(r_neighbor=int(radius)),
    )
    return len(neighbours) == grid.action_count - 1


def summarize_resolutions(
    terrain: Any,
    goal_x_map: float,
    bounds: MapBounds,
    resolutions_m: Iterable[float],
    *,
    meters_per_map_unit: float = 100.0,
) -> list[dict[str, Any]]:
    """Per-resolution Defender-grid sizes, for the Phase-1 verification figure."""
    rows: list[dict[str, Any]] = []
    for resolution in resolutions_m:
        grid = defender_grid_for_resolution(
            terrain, goal_x_map, bounds, resolution,
            meters_per_map_unit=meters_per_map_unit,
        )
        rows.append({
            "spatial_resolution_m": float(resolution),
            "spacing_map": grid.spacing_map,
            "x_count": grid.x_count,
            "y_count": grid.y_count,
            "N_D": grid.action_count,
            "excluded_solid_count": grid.excluded_solid_count,
            "saturating_radius": saturating_radius(grid),
            "centre_saturating_radius": centre_saturating_radius(grid),
        })
    return rows


__all__ = [
    "DEFENDER_Z_MAP", "SCHEMA_VERSION", "DefenderGrid2D", "DefenderRegion",
    "build_defender_grid_2d", "centre_saturating_radius",
    "defender_grid_for_resolution", "defender_region_from_scene",
    "inside_obstacle_footprint", "neighborhood_size", "saturates_from",
    "saturating_radius",
    "summarize_resolutions", "terrain_maximum_x_map",
]
