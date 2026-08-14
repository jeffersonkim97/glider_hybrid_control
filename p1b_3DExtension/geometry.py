"""Authoritative 3D terrain and sensor-dependent LOS geometry module.

Mirrors p1b_4D.geometry's role and bundle shape, generalized from a 1D
terrain profile h(z) to a 2D terrain surface h(x, y), and from the 1D
"single outward sweep" LOS boundary trick to a genuine 3D viewshed and
its corresponding two-dimensional LOS boundary surface:
for every (x, y, h) grid point, ray-march the straight line from the
sensor to that point and check it never dips below the terrain surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclass(frozen=True)
class TerrainModel:
    """Reusable terrain surface backed by a 2D bicubic spline."""

    x_grid: np.ndarray
    y_grid: np.ndarray
    sampled_height: np.ndarray  # shape (x_grid.size, y_grid.size)
    interpolant: RectBivariateSpline


def build_terrain_model(environment: dict[str, Any]) -> TerrainModel:
    """Construct the sampled Gaussian-hill terrain surface and its spline.

    Inputs
    ------
    environment:
        Validated environment_config from the Phase 1 ConfigurationBundle.

    Outputs
    -------
    TerrainModel
        Reusable sampled terrain and 2D spline interpolant.

    Assumptions
    -----------
    Grid coordinates use metres and are strictly increasing.

    Notes
    -----
    Terrain height is the sum of every configured hill's Gaussian, sampled
    on a regular (x, y) grid then fit with one bicubic spline -- terrain
    generation (the Gaussian-hill formula) and terrain querying (the
    spline) are deliberately decoupled, exactly like p1b_4D's terrain
    model, so a future real heightmap could replace _sum_of_hills without
    touching any caller of terrain_height.
    """
    _require_mapping(environment, "environment")
    grid = environment["grid"]
    terrain = environment["terrain"]
    x_grid = np.linspace(grid["x_min"], grid["x_max"], grid["x_count"])
    y_grid = np.linspace(grid["y_min"], grid["y_max"], grid["y_count"])
    sampled_height = _sum_of_hills(x_grid, y_grid, terrain["hills"])
    interpolant = RectBivariateSpline(x_grid, y_grid, sampled_height)
    return TerrainModel(
        x_grid=_readonly(x_grid),
        y_grid=_readonly(y_grid),
        sampled_height=_readonly(sampled_height),
        interpolant=interpolant,
    )


def terrain_height(terrain_model: TerrainModel, x: Any, y: Any) -> np.ndarray:
    """Evaluate terrain height at paired (x, y) coordinates (any shape).

    Inputs
    ------
    terrain_model:
        TerrainModel returned by build_terrain_model.
    x, y:
        Broadcastable horizontal coordinate arrays in metres.

    Outputs
    -------
    numpy.ndarray
        Terrain elevation with the broadcast shape of x and y.

    Assumptions
    -----------
    Coordinates lie inside the terrain model domain.
    """
    x_values, y_values = _validated_coordinates(terrain_model, x, y)
    flat_x = x_values.ravel()
    flat_y = y_values.ravel()
    flat_height = terrain_model.interpolant.ev(flat_x, flat_y)
    return np.asarray(flat_height, dtype=float).reshape(x_values.shape)


def terrain_gradient(terrain_model: TerrainModel, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate (dh/dx, dh/dy) at paired (x, y) coordinates."""
    x_values, y_values = _validated_coordinates(terrain_model, x, y)
    flat_x = x_values.ravel()
    flat_y = y_values.ravel()
    dx = terrain_model.interpolant.ev(flat_x, flat_y, dx=1, dy=0).reshape(x_values.shape)
    dy = terrain_model.interpolant.ev(flat_x, flat_y, dx=0, dy=1).reshape(x_values.shape)
    return np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)


def sensor_position_from_xy(
    terrain_model: TerrainModel,
    x_sensor: float,
    y_sensor: float,
    sensor: dict[str, Any],
) -> np.ndarray:
    """Return terrain-following sensor position [x_sensor, y_sensor, h_sensor].

    Mirrors p1b_4D's sensor_position_from_z exactly, extended by one
    horizontal coordinate: h_sensor = terrain_height(x_sensor, y_sensor) +
    mount_height.
    """
    _require_mapping(sensor, "sensor")
    x_coordinate = float(x_sensor)
    y_coordinate = float(y_sensor)
    if not (np.isfinite(x_coordinate) and np.isfinite(y_coordinate)):
        raise ValueError("x_sensor and y_sensor must be finite")
    mount_height = float(sensor["mount_height"])
    if not np.isfinite(mount_height) or mount_height < 0.0:
        raise ValueError("mount_height must be finite and nonnegative")
    height = float(terrain_height(terrain_model, x_coordinate, y_coordinate)) + mount_height
    return _readonly(np.array([x_coordinate, y_coordinate, height], dtype=float))


def goal_position_from_environment(
    environment: dict[str, Any], terrain_model: TerrainModel
) -> np.ndarray:
    """Return the terrain-following goal position [x_goal, y_goal, h_goal].

    Mirrors p1b_4D's goal_position_from_environment: h_goal is not a free
    configuration value, it is the terrain elevation at (x_goal, y_goal).
    """
    _require_mapping(environment, "environment")
    x_goal = float(environment["x_goal"])
    y_goal = float(environment["y_goal"])
    if not (np.isfinite(x_goal) and np.isfinite(y_goal)):
        raise ValueError("x_goal and y_goal must be finite")
    h_goal = float(terrain_height(terrain_model, x_goal, y_goal))
    goal = np.array([x_goal, y_goal, h_goal], dtype=float)
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        raise ValueError("goal position must contain three finite coordinates")
    return _readonly(goal)


def compute_los_geometry(
    terrain_model: TerrainModel,
    sensor_position: Any,
    x_grid: Any,
    y_grid: Any,
    h_grid: Any,
    sensor: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Compute the full 3D viewshed, terrain/LOS/occlusion masks, and coverage.

    Inputs
    ------
    terrain_model:
        Authoritative terrain model.
    sensor_position:
        Shape-(3,) terrain-following sensor coordinate.
    x_grid, y_grid, h_grid:
        Strictly increasing one-dimensional airspace grids in metres.
    sensor:
        Validated sensor_config (unused fields kept for parity with p1b_4D;
        LOS-specific tolerances come from validation).
    validation:
        Validated validation_config with geometry tolerances.

    Outputs
    -------
    dict
        3D terrain/LOS/occlusion masks and absolute/normalized LOS coverage
        volume.

    Assumptions
    -----------
    A point is visible iff the straight 3D line segment from the sensor to
    that point never dips below the terrain surface, sampled at a finite
    number of points along the segment (ray marching).

    Notes
    -----
    This replaces p1b_4D's single outward 1D sweep (a closed-form trick
    that only exists in 1D) with a genuine viewshed computation: every grid
    point independently ray-marches its own line back to the sensor. There
    is no single "tangent line" in 3D, so this bundle reports a boolean
    los_mask/occlusion_mask volume directly instead of a boundary line.
    """
    _require_mapping(sensor, "sensor")
    _require_mapping(validation, "validation")
    position = np.asarray(sensor_position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("sensor_position must have shape (3,) and be finite")
    x_values = _validated_grid(terrain_model, x_grid, "x_grid", axis=0)
    y_values = _validated_grid(terrain_model, y_grid, "y_grid", axis=1)
    h_values = np.asarray(h_grid, dtype=float)
    if h_values.ndim != 1 or h_values.size < 2:
        raise ValueError("h_grid must be a one-dimensional array with two points")
    if not np.all(np.isfinite(h_values)) or not np.all(np.diff(h_values) > 0.0):
        raise ValueError("h_grid must be finite and strictly increasing")

    x_sensor, y_sensor, h_sensor = position
    mesh_x, mesh_y, mesh_h = np.meshgrid(x_values, y_values, h_values, indexing="ij")

    terrain_tolerance = float(validation["terrain_tolerance"])
    terrain_heights = terrain_height(terrain_model, mesh_x[:, :, 0], mesh_y[:, :, 0])
    terrain_mask = mesh_h <= terrain_heights[:, :, None] + terrain_tolerance

    ray_sample_count = int(validation.get("los_ray_sample_count", 24))
    visible = _ray_march_visibility(
        terrain_model, position, mesh_x, mesh_y, mesh_h,
        terrain_tolerance, ray_sample_count,
    )
    los_boundary_height = _ray_march_boundary_height(
        terrain_model, position, x_values, y_values,
        terrain_tolerance, ray_sample_count,
    )

    non_visible_airspace = ~visible & ~terrain_mask
    occlusion_mask = terrain_mask | non_visible_airspace
    los_mask = ~occlusion_mask

    cell_volume = float(
        np.diff(x_values)[0] * np.diff(y_values)[0] * np.diff(h_values)[0]
    )
    admissible_mask = ~terrain_mask
    admissible_volume = float(np.count_nonzero(admissible_mask) * cell_volume)
    coverage_volume = float(np.count_nonzero(los_mask) * cell_volume)
    if admissible_volume <= 0.0:
        raise ValueError("Admissible airspace volume must be positive")
    normalized_coverage = coverage_volume / admissible_volume

    return {
        "mesh_x": _readonly(mesh_x),
        "mesh_y": _readonly(mesh_y),
        "mesh_h": _readonly(mesh_h),
        "terrain_mask": _readonly(terrain_mask),
        "los_mask": _readonly(los_mask),
        "occlusion_mask": _readonly(occlusion_mask),
        "non_visible_airspace_mask": _readonly(non_visible_airspace),
        "los_boundary_height": _readonly(los_boundary_height),
        "cell_volume": cell_volume,
        "admissible_airspace_volume": admissible_volume,
        "coverage_volume": coverage_volume,
        "normalized_coverage_volume": float(normalized_coverage),
        "ray_sample_count": ray_sample_count,
    }


def _ray_march_visibility(
    terrain_model: TerrainModel,
    sensor_position: np.ndarray,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    mesh_h: np.ndarray,
    terrain_tolerance: float,
    sample_count: int,
) -> np.ndarray:
    """Viewshed core: ray-march every grid point's line back to the sensor.

    A point is visible iff, at every interior sample along the straight
    line from the sensor to that point, the line's altitude is at or above
    the terrain height directly beneath that sample. Endpoints are
    excluded: t=0 is the sensor itself (always "clear" of its own ray),
    t=1 is the target point, whose own terrain relationship is already
    covered by terrain_mask.
    """
    x_sensor, y_sensor, h_sensor = sensor_position
    visible = np.ones(mesh_x.shape, dtype=bool)
    fractions = np.linspace(0.0, 1.0, sample_count)[1:-1]
    for fraction in fractions:
        sample_x = x_sensor + fraction * (mesh_x - x_sensor)
        sample_y = y_sensor + fraction * (mesh_y - y_sensor)
        sample_h_line = h_sensor + fraction * (mesh_h - h_sensor)
        sample_terrain = terrain_height(terrain_model, sample_x, sample_y)
        visible &= sample_h_line >= sample_terrain - terrain_tolerance
    return visible


def _ray_march_boundary_height(
    terrain_model: TerrainModel,
    sensor_position: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    terrain_tolerance: float,
    sample_count: int,
) -> np.ndarray:
    """Return the 3D analogue of p1b_4D's LOS tangent boundary.

    For every horizontal target coordinate ``(x, y)``, the returned value
    is the minimum target altitude whose sensor-to-target ray clears every
    interior terrain sample used by :func:`_ray_march_visibility`.  Hence
    ``h == boundary[x, y]`` is the two-dimensional switching surface that
    replaces p1b_4D's one-dimensional tangent/boundary line without changing
    the underlying ray-march LOS definition.
    """
    x_sensor, y_sensor, h_sensor = np.asarray(sensor_position, dtype=float)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    boundary = np.full(mesh_x.shape, -np.inf, dtype=float)
    fractions = np.linspace(0.0, 1.0, sample_count)[1:-1]
    for fraction in fractions:
        sample_x = x_sensor + fraction * (mesh_x - x_sensor)
        sample_y = y_sensor + fraction * (mesh_y - y_sensor)
        sample_terrain = terrain_height(terrain_model, sample_x, sample_y)
        required_target_height = h_sensor + (
            sample_terrain - terrain_tolerance - h_sensor
        ) / fraction
        boundary = np.maximum(boundary, required_target_height)
    if not np.all(np.isfinite(boundary)):
        raise ValueError("LOS boundary surface contains non-finite heights")
    return boundary


def build_geometry_bundle(configuration_bundle: dict[str, Any]) -> dict[str, Any]:
    """Build and validate the complete Phase 2 3D Geometry Bundle.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.

    Outputs
    -------
    dict
        Universal result envelope containing terrain model/arrays, sensor
        and goal positions, LOS masks, coverage, metadata, and validation.

    Assumptions
    -----------
    Configuration is not mutated and no optimization decision is made.

    Notes
    -----
    This function performs geometry only and produces no figures or files.
    """
    if not isinstance(configuration_bundle, dict):
        raise TypeError("configuration_bundle must be a dictionary")
    if not configuration_bundle.get("status", {}).get("success", False):
        raise ValueError("configuration_bundle must have successful status")
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    sensor = configs["sensor_config"]
    validation_config = configs["validation_config"]

    model = build_terrain_model(environment)
    grid = environment["grid"]
    x_grid = np.linspace(grid["x_min"], grid["x_max"], grid["x_count"])
    y_grid = np.linspace(grid["y_min"], grid["y_max"], grid["y_count"])
    h_grid = np.linspace(grid["h_min"], grid["h_max"], grid["h_count"])
    sensor_position = sensor_position_from_xy(
        model, sensor["default_x_sensor"], sensor["default_y_sensor"], sensor
    )
    goal_position = goal_position_from_environment(environment, model)
    los = compute_los_geometry(
        model, sensor_position, x_grid, y_grid, h_grid, sensor, validation_config,
    )
    validation = validate_geometry(
        model, sensor_position, goal_position, los, environment, sensor, validation_config,
    )
    primary_result = {
        "terrain_model": model,
        "terrain_arrays": {
            "x": _readonly(x_grid),
            "y": _readonly(y_grid),
            "height": model.sampled_height,
        },
        "sensor_position": sensor_position,
        "goal_position": goal_position,
        "los_masks": {
            key: los[key]
            for key in (
                "los_mask", "occlusion_mask", "terrain_mask", "non_visible_airspace_mask",
                "los_boundary_height",
            )
        },
        "coverage": {
            key: los[key]
            for key in (
                "cell_volume", "admissible_airspace_volume",
                "coverage_volume", "normalized_coverage_volume",
            )
        },
    }
    return {
        "primary_result": primary_result,
        "validation": validation,
        "metadata": {
            "schema_name": "GeometryBundle3D",
            "schema_version": "1.0.0",
            "producer_phase": 2,
            "producer_module": "p1b_3DExtension.geometry",
            "config_schema_version": configuration_bundle["metadata"]["schema_version"],
            "coordinate_convention": ("x", "y", "h"),
            "units": {"x": "m", "y": "m", "h": "m", "volume": "m^3"},
            "dimensions": {
                "x_count": int(x_grid.size),
                "y_count": int(y_grid.size),
                "h_count": int(h_grid.size),
                "mask_shape": tuple(los["los_mask"].shape),
            },
            "axis_order": ("x", "y", "h"),
            "sensor_position": [float(v) for v in sensor_position],
            "goal_position": [float(v) for v in goal_position],
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "GEOMETRY_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def validate_geometry(
    terrain_model: TerrainModel,
    sensor_position: np.ndarray,
    goal_position: np.ndarray,
    los: dict[str, Any],
    environment: dict[str, Any],
    sensor: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate dimensions, positions, and coverage consistency."""
    expected_shape = (
        environment["grid"]["x_count"],
        environment["grid"]["y_count"],
        environment["grid"]["h_count"],
    )
    expected_sensor_h = float(
        terrain_height(terrain_model, sensor_position[0], sensor_position[1])
        + sensor["mount_height"]
    )
    expected_goal_h = float(
        terrain_height(terrain_model, environment["x_goal"], environment["y_goal"])
    )
    terrain_tolerance = float(validation["terrain_tolerance"])
    nearest_x_index = int(np.argmin(np.abs(terrain_model.x_grid - sensor_position[0])))
    nearest_y_index = int(np.argmin(np.abs(terrain_model.y_grid - sensor_position[1])))
    sensor_is_grid_aligned = bool(
        abs(terrain_model.x_grid[nearest_x_index] - sensor_position[0])
        <= terrain_tolerance
        and abs(terrain_model.y_grid[nearest_y_index] - sensor_position[1])
        <= terrain_tolerance
    )
    checks = {
        "los_mask_dimensions": los["los_mask"].shape == expected_shape,
        "occlusion_mask_dimensions": los["occlusion_mask"].shape == expected_shape,
        "los_boundary_surface_dimensions": (
            los["los_boundary_height"].shape == expected_shape[:2]
        ),
        "mask_partition": np.array_equal(los["los_mask"], ~los["occlusion_mask"]),
        "sensor_position": (
            sensor_position.shape == (3,)
            and abs(sensor_position[2] - expected_sensor_h) <= terrain_tolerance
        ),
        "goal_position": (
            goal_position.shape == (3,)
            and goal_position[0] == environment["x_goal"]
            and goal_position[1] == environment["y_goal"]
            and abs(goal_position[2] - expected_goal_h) <= terrain_tolerance
        ),
        "coverage_volume": (
            0.0 <= los["coverage_volume"] <= los["admissible_airspace_volume"]
            and 0.0 <= los["normalized_coverage_volume"] <= 1.0
        ),
        "sensor_own_cell_visible": bool(
            # A continuous Defender position generally has no grid-owned
            # cell.  The old nearest-column test could mark a valid sensor
            # invisible merely because the neighboring terrain sample was
            # higher.  Apply the discrete own-cell invariant only when the
            # sensor is actually grid aligned; its exact continuous height
            # is already checked by `sensor_position` above.
            not sensor_is_grid_aligned
            or los["los_mask"][
                nearest_x_index,
                nearest_y_index,
                min(
                    np.searchsorted(
                        np.linspace(
                            environment["grid"]["h_min"],
                            environment["grid"]["h_max"],
                            environment["grid"]["h_count"],
                        ),
                        sensor_position[2],
                    ),
                    environment["grid"]["h_count"] - 1,
                ),
            ]
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "coverage_volume": los["coverage_volume"],
            "normalized_coverage_volume": los["normalized_coverage_volume"],
            "visible_cell_count": int(np.count_nonzero(los["los_mask"])),
            "occluded_cell_count": int(np.count_nonzero(los["occlusion_mask"])),
            "sensor_is_grid_aligned": sensor_is_grid_aligned,
        },
        "tolerances": {"terrain": terrain_tolerance},
        "warnings": [],
        "failed_checks": failed_checks,
        "summary": (
            "Phase 2 geometry validation passed"
            if not failed_checks
            else f"Phase 2 geometry failed checks: {failed_checks}"
        ),
    }


def _sum_of_hills(x_grid: np.ndarray, y_grid: np.ndarray, hills: Any) -> np.ndarray:
    """Sum every configured Gaussian hill into one combined 2D terrain surface.

    A single-hill tuple is the v1 toy case; multiple hills are not separate
    terrain objects, this returns one combined surface, mirroring p1b_4D's
    "terrain is one surface" invariant exactly.
    """
    if not hills:
        raise ValueError("environment_config.terrain.hills must be non-empty")
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    sampled_height = np.zeros_like(mesh_x, dtype=float)
    for hill in hills:
        isotropic_width = hill.get("width")
        width_x = hill.get("width_x", isotropic_width)
        width_y = hill.get("width_y", isotropic_width)
        if width_x is None or width_y is None:
            raise ValueError(
                "Each hill requires width or both width_x and width_y"
            )
        width_x = float(width_x)
        width_y = float(width_y)
        if not (
            np.isfinite(width_x) and np.isfinite(width_y)
            and width_x > 0.0 and width_y > 0.0
        ):
            raise ValueError("Hill widths must be finite and positive")
        sampled_height = sampled_height + hill["h_ridge"] * np.exp(
            -0.5 * (
                ((mesh_x - hill["x_ridge"]) / width_x) ** 2
                + ((mesh_y - hill["y_ridge"]) / width_y) ** 2
            )
        )
    return sampled_height


def _validated_coordinates(model: TerrainModel, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    if not (np.all(np.isfinite(x_values)) and np.all(np.isfinite(y_values))):
        raise ValueError("terrain coordinates must be finite")
    if np.any(x_values < model.x_grid[0]) or np.any(x_values > model.x_grid[-1]):
        raise ValueError("x coordinates lie outside the model domain")
    if np.any(y_values < model.y_grid[0]) or np.any(y_values > model.y_grid[-1]):
        raise ValueError("y coordinates lie outside the model domain")
    return x_values, y_values


def _validated_grid(model: TerrainModel, values: Any, name: str, axis: int) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0.0):
        raise ValueError(f"{name} must be one-dimensional and increasing")
    reference = model.x_grid if axis == 0 else model.y_grid
    if grid[0] < reference[0] or grid[-1] > reference[-1]:
        raise ValueError(f"{name} lies outside the terrain model domain")
    return grid


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
