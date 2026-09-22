"""Shared computation condition for the exact-versus-RL complexity comparison.

``P1b_complexity.ipynb`` sets one condition here and hands the *same* object to
``P1b_Exact_Local_SSE`` and ``P1b_RL_approximation``.  Neither module defines a
condition of its own, so "same machine, same condition" is enforced by
construction rather than by convention.

Three knobs, on a single terrain:

    spatial_resolution_m   dx = dy = dh
    heading_spacing_deg    heading-angle discretisation
    r_neighbor             Defender neighbourhood radius, Chebyshev distance in
                           discretised Defender-grid indices

The goal-backward reachable graph depends on the lattice, the terrain and the
goal - never on the sensor position - so it is built once here and shared.  Both
methods would otherwise rebuild it for every Defender position, which is real
work neither of them needs to repeat and which would inflate whichever side is
measured with it included.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

from defender_grid_2d import DefenderGrid2D, defender_grid_for_resolution
from discretization_config import discretization_from_physical_steps
from scenario import MissionPoints, Point3D
from sparse_reachability import build_goal_backward_reachable_graph
from stage11_config import Stage11Config
from terrain_catalog import build_terrain


@dataclass(frozen=True)
class ComputationCondition:
    """One point in the (spatial, heading, radius) condition space."""

    spatial_resolution_m: float = 25.0
    heading_spacing_deg: float = 5.0
    r_neighbor: int = 1
    terrain_category: str = "centered_cube"
    # Attacker objective weights, which must sum to one.  Left as None they keep
    # whatever Stage11Config declares; set, they replace it for this condition.
    # They belong here rather than in either solver because both must weigh the
    # objective identically or the comparison measures the weighting, not the
    # solvers.
    hazard_weight: float | None = None
    time_weight: float | None = None
    # Keep the sensor off the goal itself.  On the goal, detection is certain and
    # the hazard term drowns out everything else; one map unit back restores the
    # intended time/hazard balance.  See defender_region_from_scene.
    defender_goal_margin_map: float = 1.0

    def __post_init__(self) -> None:
        if not self.spatial_resolution_m > 0.0:
            raise ValueError("spatial_resolution_m must be positive")
        if not self.heading_spacing_deg > 0.0:
            raise ValueError("heading_spacing_deg must be positive")
        if int(self.r_neighbor) < 1:
            raise ValueError("r_neighbor must be at least one")
        object.__setattr__(self, "r_neighbor", int(self.r_neighbor))
        if (self.hazard_weight is None) != (self.time_weight is None):
            raise ValueError(
                "give both objective weights or neither; one alone would leave the "
                "pair not summing to one"
            )

    @property
    def weight_tag(self) -> str | None:
        """``w25-75`` and the like, or None when the base weights are kept."""
        if self.hazard_weight is None and self.time_weight is None:
            return None
        return (
            f"w{round(100 * float(self.hazard_weight)):02d}"
            f"-{round(100 * float(self.time_weight)):02d}"
        )

    @property
    def label(self) -> str:
        weights = (
            "" if self.weight_tag is None
            else f" wH={self.hazard_weight:g} wT={self.time_weight:g}"
        )
        return (
            f"dx={self.spatial_resolution_m:g}m "
            f"dpsi={self.heading_spacing_deg:g}deg "
            f"r={self.r_neighbor}{weights}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "spatial_resolution_m": self.spatial_resolution_m,
            "heading_spacing_deg": self.heading_spacing_deg,
            "r_neighbor": self.r_neighbor,
            "terrain_category": self.terrain_category,
            "hazard_weight": self.hazard_weight,
            "time_weight": self.time_weight,
            "defender_goal_margin_map": self.defender_goal_margin_map,
        }


@dataclass
class Scene:
    """Everything both methods need, built once for one condition."""

    condition: ComputationCondition
    config: Stage11Config
    terrain: Any
    grid: Any
    defender_grid: DefenderGrid2D
    reachability: Any
    seed_action_id: int
    build_timing: dict[str, float] = field(default_factory=dict)

    @property
    def graph(self) -> Any:
        return self.reachability.graph

    @property
    def goal_reachable(self) -> Any:
        return self.reachability.unit_reachability.goal_reachable

    def mission_for(self, sensor_map: tuple[float, float, float]) -> MissionPoints:
        return MissionPoints(
            sensor=Point3D(*(float(value) for value in sensor_map)),
            start=self.config.start,
            goal=self.config.goal,
        )

    def sizes(self) -> dict[str, Any]:
        return {
            "N_x": self.grid.x_count,
            "N_y": self.grid.y_count,
            "N_h": self.grid.altitude_count,
            "N_psi": self.grid.heading_bin_count,
            "N_S_cart": self.grid.state_count,
            "N_S_goal_reachable": self.reachability.unit_reachability.goal_reachable_state_count,
            "A": len(self.grid.motion_offsets),
            "N_D": self.defender_grid.action_count,
            "neighbourhood_size": (2 * self.condition.r_neighbor + 1) ** 2,
        }


def build_scene(condition: ComputationCondition) -> Scene:
    """Build the condition's lattice, Defender grid and shared glide graph."""
    timing: dict[str, float] = {}

    started = perf_counter()
    base = Stage11Config()
    discretization = discretization_from_physical_steps(
        condition.spatial_resolution_m,
        condition.spatial_resolution_m,
        condition.heading_spacing_deg,
        meters_per_map_unit=base.physical_scale.meters_per_map_unit,
        template=base.discretization,
    )
    objective = base.attacker_objective
    if condition.hazard_weight is not None:
        objective = replace(
            objective,
            hazard_weight=float(condition.hazard_weight),
            time_weight=float(condition.time_weight),
        )
    config = replace(
        base,
        terrain_category=condition.terrain_category,
        discretization=discretization,
        attacker_objective=objective,
    )
    terrain = build_terrain(condition.terrain_category)
    grid = discretization.build_bellman_grid(config.graph_bounds)
    timing["T_setup_s"] = perf_counter() - started

    started = perf_counter()
    defender_grid = defender_grid_for_resolution(
        terrain, config.goal.x, config.graph_bounds,
        condition.spatial_resolution_m,
        meters_per_map_unit=config.physical_scale.meters_per_map_unit,
        goal_margin_map=condition.defender_goal_margin_map,
    )
    timing["T_defender_grid_s"] = perf_counter() - started

    # Shared, because it is independent of the sensor position.
    started = perf_counter()
    reachability = build_goal_backward_reachable_graph(
        grid, terrain, config.goal,
        parameters=config.glider, physical_scale=config.physical_scale,
    )
    timing["T_shared_graph_s"] = perf_counter() - started

    return Scene(
        condition=condition,
        config=config,
        terrain=terrain,
        grid=grid,
        defender_grid=defender_grid,
        reachability=reachability,
        seed_action_id=defender_grid.nearest_action_id(
            config.sensor.x, config.sensor.y,
        ),
        build_timing=timing,
    )


__all__ = ["ComputationCondition", "Scene", "build_scene"]
