"""Custom-scene construction for the extended random-box Phase 16.4 pilot."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

import project_paths  # noqa: F401
from P1b_condition import ComputationCondition, Scene
from defender_grid_2d import DefenderRegion, build_defender_grid_2d
from discretization_config import discretization_from_physical_steps
from phase4_extended_dataset import (
    ExtendedPilotConfig,
    ExtendedScenarioSpec,
    RandomBoxTerrainSpec,
    build_random_box_terrain,
)
from scenario import Point3D
from sparse_reachability import build_goal_backward_reachable_graph
from stage11_config import Stage11Config


def build_extended_scene(
    scenario: ExtendedScenarioSpec,
    terrain_spec: RandomBoxTerrainSpec,
    *,
    spatial_resolution_m: float = 100.0,
    heading_spacing_deg: float = 5.0,
    r_neighbor: int = 1,
    pilot_config: ExtendedPilotConfig = ExtendedPilotConfig(),
) -> Scene:
    """Build one authoritative scene from manifest-carried geometry and mission."""

    if terrain_spec.terrain_instance_id != scenario.terrain_category:
        raise ValueError("scenario and terrain instance IDs differ")
    if terrain_spec.terrain_hash != scenario.terrain_hash:
        raise ValueError("scenario and terrain geometry hashes differ")
    timing: dict[str, float] = {}
    started = perf_counter()
    base = Stage11Config()
    discretization = discretization_from_physical_steps(
        float(spatial_resolution_m),
        float(spatial_resolution_m),
        float(heading_spacing_deg),
        meters_per_map_unit=base.physical_scale.meters_per_map_unit,
        template=base.discretization,
    )
    glider = replace(
        base.glider,
        goal_tolerance_m=float(scenario.goal_tolerance_m),
    )
    sensor = Point3D(*scenario.sensor_map)
    start = Point3D(*scenario.start_map)
    goal = Point3D(*scenario.goal_map)
    config = replace(
        base,
        terrain_category=scenario.terrain_category,
        sensor=sensor,
        start=start,
        goal=goal,
        glider=glider,
        discretization=discretization,
    )
    terrain = build_random_box_terrain(terrain_spec, pilot_config)
    config.mission_points.validate_against(terrain)
    grid = discretization.build_bellman_grid(config.graph_bounds)
    timing["T_setup_s"] = perf_counter() - started

    started = perf_counter()
    # The legacy Defender region begins at the maximum obstacle x-coordinate.
    # Random continuous box widths make that boundary non-lattice-aligned.  This
    # pilot is an attacker-training dataset, so retain a deterministic aligned
    # placeholder over the graph bounds; the later Local-SSE integration phase
    # will freeze its own approved potential-sensor region.
    defender_grid = build_defender_grid_2d(
        DefenderRegion(
            config.graph_bounds.x_min,
            config.graph_bounds.x_max,
            config.graph_bounds.y_min,
            config.graph_bounds.y_max,
        ),
        float(spatial_resolution_m)
        / config.physical_scale.meters_per_map_unit,
        terrain=terrain,
    )
    timing["T_defender_grid_s"] = perf_counter() - started

    started = perf_counter()
    reachability = build_goal_backward_reachable_graph(
        grid,
        terrain,
        config.goal,
        parameters=config.glider,
        physical_scale=config.physical_scale,
    )
    timing["T_shared_graph_s"] = perf_counter() - started

    condition = ComputationCondition(
        spatial_resolution_m=float(spatial_resolution_m),
        heading_spacing_deg=float(heading_spacing_deg),
        r_neighbor=int(r_neighbor),
        terrain_category=scenario.terrain_category,
    )
    scene = Scene(
        condition=condition,
        config=config,
        terrain=terrain,
        grid=grid,
        defender_grid=defender_grid,
        reachability=reachability,
        seed_action_id=defender_grid.nearest_action_id(sensor.x, sensor.y),
        build_timing=timing,
    )
    if not np.any(scene.graph.terminal_mask):
        raise RuntimeError("extended goal contains no terminal lattice state")
    return scene


__all__ = ["build_extended_scene"]
