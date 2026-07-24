"""Authoritative terrain and sensor-dependent LOS geometry module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class TerrainModel:
    """Reusable terrain model backed by the existing natural cubic spline."""

    z_grid: np.ndarray
    sampled_height: np.ndarray
    interpolant: CubicSpline


def build_terrain_model(environment: dict[str, Any]) -> TerrainModel:
    """Construct the existing Gaussian ridge and natural cubic spline.

    Inputs
    ------
    environment:
        Validated environment_config from the Phase 1 ConfigurationBundle.

    Outputs
    -------
    TerrainModel
        Reusable sampled terrain and spline interpolant.

    Assumptions
    -----------
    Grid coordinates use metres and are strictly increasing.

    Notes
    -----
    This reproduces the existing notebook terrain without tuning parameters.
    """
    _require_mapping(environment, "environment")
    grid = environment["grid"]
    terrain = environment["terrain"]
    z_grid = np.linspace(grid["z_min"], grid["z_max"], grid["z_count"])
    sampled_height = _sum_of_hills(z_grid, terrain["hills"])
    interpolant = CubicSpline(z_grid, sampled_height, bc_type="natural")
    return TerrainModel(
        z_grid=_readonly(z_grid),
        sampled_height=_readonly(sampled_height),
        interpolant=interpolant,
    )


def terrain_height(terrain_model: TerrainModel, z: Any) -> np.ndarray:
    """Evaluate terrain height at scalar or array horizontal coordinates.

    Inputs
    ------
    terrain_model:
        TerrainModel returned by build_terrain_model.
    z:
        Horizontal coordinate or array in metres.

    Outputs
    -------
    numpy.ndarray
        Terrain elevation with the same broadcast shape as z.

    Assumptions
    -----------
    Coordinates lie inside the terrain model domain.

    Notes
    -----
    Extrapolation is rejected to prevent silent out-of-domain geometry.
    """
    values = _validated_coordinates(terrain_model, z)
    return np.asarray(terrain_model.interpolant(values), dtype=float)


def terrain_gradient(terrain_model: TerrainModel, z: Any) -> np.ndarray:
    """Evaluate the first terrain derivative with respect to z."""
    values = _validated_coordinates(terrain_model, z)
    return np.asarray(terrain_model.interpolant(values, 1), dtype=float)


def terrain_curvature(terrain_model: TerrainModel, z: Any) -> np.ndarray:
    """Evaluate the second terrain derivative with respect to z."""
    values = _validated_coordinates(terrain_model, z)
    return np.asarray(terrain_model.interpolant(values, 2), dtype=float)


def terrain_profile(
    terrain_model: TerrainModel, z_grid: Any
) -> dict[str, np.ndarray]:
    """Return terrain coordinate, height, gradient, and curvature arrays."""
    coordinates = _validated_coordinates(terrain_model, z_grid)
    if coordinates.ndim != 1:
        raise ValueError("z_grid must be one-dimensional")
    if coordinates.size < 2 or not np.all(np.diff(coordinates) > 0.0):
        raise ValueError("z_grid must contain strictly increasing coordinates")
    return {
        "z": _readonly(coordinates.copy()),
        "height": _readonly(terrain_height(terrain_model, coordinates)),
        "gradient": _readonly(terrain_gradient(terrain_model, coordinates)),
        "curvature": _readonly(terrain_curvature(terrain_model, coordinates)),
    }


def sensor_position_from_z(
    terrain_model: TerrainModel,
    z_sensor: float,
    sensor: dict[str, Any],
) -> np.ndarray:
    """Return terrain-following sensor position [z_sensor, h_sensor].

    Inputs
    ------
    terrain_model:
        Authoritative terrain model.
    z_sensor:
        Continuous sensor horizontal coordinate in metres.
    sensor:
        Validated sensor_config containing mount_height.

    Outputs
    -------
    numpy.ndarray
        Shape-(2,) sensor position in metres.

    Assumptions
    -----------
    Sensor height equals terrain_height(z_sensor) plus mount_height.
    """
    _require_mapping(sensor, "sensor")
    coordinate = float(z_sensor)
    if not np.isfinite(coordinate):
        raise ValueError("z_sensor must be finite")
    mount_height = float(sensor["mount_height"])
    if not np.isfinite(mount_height) or mount_height < 0.0:
        raise ValueError("mount_height must be finite and nonnegative")
    height = float(terrain_height(terrain_model, coordinate)) + mount_height
    return _readonly(np.array([coordinate, height], dtype=float))


def goal_position_from_environment(environment: dict[str, Any]) -> np.ndarray:
    """Return the fixed configured goal position [z_goal, h_goal]."""
    _require_mapping(environment, "environment")
    goal = np.array([environment["z_goal"], environment["h_goal"]], dtype=float)
    if goal.shape != (2,) or not np.all(np.isfinite(goal)):
        raise ValueError("goal position must contain two finite coordinates")
    return _readonly(goal)


def compute_los_geometry(
    terrain_model: TerrainModel,
    sensor_position: Any,
    z_grid: Any,
    h_grid: Any,
    sensor: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Compute tangent, boundary, masks, and LOS coverage for one sensor.

    Inputs
    ------
    terrain_model:
        Authoritative terrain model.
    sensor_position:
        Shape-(2,) terrain-following sensor coordinate.
    z_grid, h_grid:
        Strictly increasing one-dimensional airspace grids in metres.
    sensor:
        Validated sensor_config with LOS settings.
    validation:
        Validated validation_config with geometry tolerances.

    Outputs
    -------
    dict
        Tangent point/line, LOS boundary, terrain/LOS/occlusion masks, and
        absolute and normalized LOS coverage area.

    Assumptions
    -----------
    The attacker always approaches from z below the sensor. Airspace cells
    at or below terrain are occluded.

    Notes
    -----
    The LOS boundary is computed by a single outward sweep from the sensor,
    not by finding one ridge's tangent point: for every terrain sample left
    of the sensor, the minimum-required-height ray slope to clear every
    intervening terrain sample is the running minimum of the point-to-sensor
    ray slope, accumulated from the sensor outward. This is the general
    multi-obstacle visibility boundary -- with one hill it reduces exactly
    to that hill's tangent line; with several, shadows merge or stay
    separate correctly without treating any hill specially. `tangent_point`/
    `tangent_slope`/`tangent_intercept` below report only the single most
    restrictive sample of that sweep, kept for diagnostics/backward
    compatibility -- the actual boundary used for the masks is the full
    swept array.
    """
    _require_mapping(sensor, "sensor")
    _require_mapping(validation, "validation")
    position = np.asarray(sensor_position, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("sensor_position must have shape (2,) and be finite")
    z_values = _validated_grid(terrain_model, z_grid, "z_grid")
    h_values = np.asarray(h_grid, dtype=float)
    if h_values.ndim != 1 or h_values.size < 2:
        raise ValueError("h_grid must be a one-dimensional array with two points")
    if not np.all(np.isfinite(h_values)) or not np.all(np.diff(h_values) > 0.0):
        raise ValueError("h_grid must be finite and strictly increasing")

    z_sensor, h_sensor = position
    candidates = z_values[z_values < z_sensor]
    if candidates.size < 2:
        raise ValueError("At least two terrain samples must lie left of the sensor")

    slope_to_sensor = (
        terrain_height(terrain_model, candidates) - h_sensor
    ) / (candidates - z_sensor)
    if not np.all(np.isfinite(slope_to_sensor)):
        raise ValueError("No sensor-visible LOS tangent sign change was found")
    # Sweep from the sensor outward (candidates are ascending, i.e. sensor
    # end last): the boundary slope at each point is the tightest (most
    # occluding) ray slope anywhere between it and the sensor.
    boundary_slope = np.minimum.accumulate(slope_to_sensor[::-1])[::-1]
    boundary_height_candidates = h_sensor + boundary_slope * (candidates - z_sensor)
    boundary_height = np.full(z_values.shape, boundary_height_candidates[-1])
    boundary_height[: candidates.size] = boundary_height_candidates
    los_boundary = np.column_stack((z_values, boundary_height))

    tangent_index = int(np.argmin(slope_to_sensor))
    tangent_z = float(candidates[tangent_index])
    tangent_h = float(terrain_height(terrain_model, tangent_z))
    tangent_slope = float(slope_to_sensor[tangent_index])
    tangent_intercept = float(h_sensor - tangent_slope * z_sensor)

    mesh_z, mesh_h = np.meshgrid(z_values, h_values, indexing="ij")
    terrain_tolerance = float(validation["terrain_tolerance"])
    terrain_heights = terrain_height(terrain_model, z_values)
    terrain_mask = mesh_h <= terrain_heights[:, None] + terrain_tolerance
    non_visible_airspace = (
        (mesh_z < z_sensor)
        & (mesh_h < boundary_height[:, None])
        & ~terrain_mask
    )
    occlusion_mask = terrain_mask | non_visible_airspace
    los_mask = ~occlusion_mask

    cell_area = float(np.diff(z_values)[0] * np.diff(h_values)[0])
    admissible_mask = ~terrain_mask
    admissible_area = float(np.count_nonzero(admissible_mask) * cell_area)
    coverage_area = float(np.count_nonzero(los_mask) * cell_area)
    if admissible_area <= 0.0:
        raise ValueError("Admissible airspace area must be positive")
    normalized_coverage = coverage_area / admissible_area

    tangent_gradient = float(terrain_gradient(terrain_model, tangent_z))
    tangent_residual = tangent_slope - tangent_gradient
    return {
        "tangent_point": _readonly(
            np.array([tangent_z, tangent_h], dtype=float)
        ),
        "tangent_slope": float(tangent_slope),
        "tangent_intercept": float(tangent_intercept),
        "tangent_line_height": _readonly(boundary_height),
        "los_boundary": _readonly(los_boundary),
        "mesh_z": _readonly(mesh_z),
        "mesh_h": _readonly(mesh_h),
        "terrain_mask": _readonly(terrain_mask),
        "los_mask": _readonly(los_mask),
        "occlusion_mask": _readonly(occlusion_mask),
        "non_visible_airspace_mask": _readonly(non_visible_airspace),
        "cell_area": cell_area,
        "admissible_airspace_area": admissible_area,
        "coverage_area": coverage_area,
        "normalized_coverage_area": float(normalized_coverage),
        "tangent_residual": float(tangent_residual),
    }


def los_boundary_height(los_geometry: dict[str, Any], z: Any) -> np.ndarray:
    """Interpolate the general swept LOS visibility boundary height at z.

    Inputs
    ------
    los_geometry:
        The `los_geometry` sub-dict of a GeometryBundle's primary_result
        (must contain `los_boundary`, an (N, 2) array of [z, boundary_height]
        pairs from `compute_los_geometry`).
    z:
        Scalar or array horizontal coordinate(s) in metres.

    Outputs
    -------
    numpy.ndarray
        Boundary height with the same broadcast shape as z.

    Notes
    -----
    Piecewise-linear interpolation over the swept boundary array is exact
    (not approximate) everywhere the governing obstacle does not change
    strictly between two adjacent z-grid nodes, because the boundary height
    is itself piecewise-linear in z between such changes (see
    `compute_los_geometry`'s docstring); with one hill this reproduces the
    single tangent line exactly over its whole domain. This is the one
    general replacement for the old `tangent_slope * z + tangent_intercept`
    formula used by every occlusion check.
    """
    boundary = np.asarray(los_geometry["los_boundary"], dtype=float)
    return np.interp(np.asarray(z, dtype=float), boundary[:, 0], boundary[:, 1])


def build_geometry_bundle(configuration_bundle: dict[str, Any]) -> dict[str, Any]:
    """Build and validate the complete Phase 2 Geometry Bundle.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.

    Outputs
    -------
    dict
        Universal result envelope containing terrain model/arrays, sensor and
        goal positions, LOS geometry/masks, coverage, metadata, and validation.

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
    z_grid = np.linspace(
        environment["grid"]["z_min"],
        environment["grid"]["z_max"],
        environment["grid"]["z_count"],
    )
    h_grid = np.linspace(
        environment["grid"]["h_min"],
        environment["grid"]["h_max"],
        environment["grid"]["h_count"],
    )
    profile = terrain_profile(model, z_grid)
    sensor_position = sensor_position_from_z(
        model, sensor["default_z_sensor"], sensor
    )
    goal_position = goal_position_from_environment(environment)
    los = compute_los_geometry(
        model,
        sensor_position,
        z_grid,
        h_grid,
        sensor,
        validation_config,
    )
    validation = validate_geometry(
        model,
        profile,
        sensor_position,
        goal_position,
        los,
        environment,
        sensor,
        validation_config,
    )
    primary_result = {
        "terrain_model": model,
        "terrain_arrays": {
            **profile,
            "h_grid": _readonly(h_grid),
            "terrain_mask": los["terrain_mask"],
        },
        "sensor_position": sensor_position,
        "goal_position": goal_position,
        "los_geometry": {
            key: los[key]
            for key in (
                "tangent_point",
                "tangent_slope",
                "tangent_intercept",
                "tangent_line_height",
                "los_boundary",
                "tangent_residual",
            )
        },
        "los_masks": {
            key: los[key]
            for key in (
                "los_mask",
                "occlusion_mask",
                "terrain_mask",
                "non_visible_airspace_mask",
            )
        },
        "coverage": {
            key: los[key]
            for key in (
                "cell_area",
                "admissible_airspace_area",
                "coverage_area",
                "normalized_coverage_area",
            )
        },
    }
    return {
        "primary_result": primary_result,
        "validation": validation,
        "metadata": {
            "schema_name": "GeometryBundle",
            "schema_version": "1.0.0",
            "producer_phase": 2,
            "producer_module": "p1b_4D.geometry",
            "config_schema_version": configuration_bundle["metadata"][
                "schema_version"
            ],
            "coordinate_convention": ("z", "h"),
            "units": {"z": "m", "h": "m", "area": "m^2"},
            "dimensions": {
                "z_count": int(z_grid.size),
                "h_count": int(h_grid.size),
                "mask_shape": tuple(los["los_mask"].shape),
            },
            "axis_order": ("z", "h"),
            "z_sensor": float(sensor_position[0]),
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
    profile: dict[str, np.ndarray],
    sensor_position: np.ndarray,
    goal_position: np.ndarray,
    los: dict[str, Any],
    environment: dict[str, Any],
    sensor: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate dimensions, positions, coverage, and tangent consistency."""
    expected_shape = (
        environment["grid"]["z_count"],
        environment["grid"]["h_count"],
    )
    expected_sensor_h = float(
        terrain_height(terrain_model, sensor_position[0])
        + sensor["mount_height"]
    )
    tangent_tolerance = float(validation["los_tolerance"])
    checks = {
        "terrain_dimensions": (
            profile["z"].shape
            == profile["height"].shape
            == profile["gradient"].shape
            == profile["curvature"].shape
        ),
        "los_mask_dimensions": los["los_mask"].shape == expected_shape,
        "occlusion_mask_dimensions": (
            los["occlusion_mask"].shape == expected_shape
        ),
        "mask_partition": np.array_equal(
            los["los_mask"], ~los["occlusion_mask"]
        ),
        "sensor_position": (
            sensor_position.shape == (2,)
            and abs(sensor_position[1] - expected_sensor_h)
            <= float(validation["terrain_tolerance"])
        ),
        "goal_position": (
            goal_position.shape == (2,)
            and np.allclose(
                goal_position,
                [environment["z_goal"], environment["h_goal"]],
                rtol=0.0,
                atol=0.0,
            )
        ),
        "coverage_area": (
            0.0 <= los["coverage_area"] <= los["admissible_airspace_area"]
            and 0.0 <= los["normalized_coverage_area"] <= 1.0
        ),
        "tangent_consistency": (
            abs(los["tangent_residual"]) <= tangent_tolerance
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "tangent_residual": los["tangent_residual"],
            "coverage_area": los["coverage_area"],
            "normalized_coverage_area": los["normalized_coverage_area"],
            "terrain_point_count": int(profile["z"].size),
            "visible_cell_count": int(np.count_nonzero(los["los_mask"])),
            "occluded_cell_count": int(
                np.count_nonzero(los["occlusion_mask"])
            ),
        },
        "tolerances": {
            "terrain": float(validation["terrain_tolerance"]),
            "los": tangent_tolerance,
        },
        "warnings": [],
        "failed_checks": failed_checks,
        "summary": (
            "Phase 2 geometry validation passed"
            if not failed_checks
            else f"Phase 2 geometry failed checks: {failed_checks}"
        ),
    }


def _sum_of_hills(z_grid: np.ndarray, hills: Any) -> np.ndarray:
    """Sum every configured Gaussian hill into one continuous height profile.

    A single-hill tuple reproduces the original terrain exactly. Multiple
    hills are not separate terrain objects: this returns one combined
    profile, which is the only thing `build_terrain_model` ever turns into
    a spline, and the only thing any later LOS/visibility computation ever
    sees.
    """
    if not hills:
        raise ValueError("environment_config.terrain.hills must be non-empty")
    sampled_height = np.zeros_like(z_grid, dtype=float)
    for hill in hills:
        sampled_height = sampled_height + hill["h_ridge"] * np.exp(
            -0.5 * ((z_grid - hill["z_ridge"]) / hill["width"]) ** 2
        )
    return sampled_height


def _validated_coordinates(model: TerrainModel, values: Any) -> np.ndarray:
    coordinates = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("terrain coordinates must be finite")
    if np.any(coordinates < model.z_grid[0]) or np.any(
        coordinates > model.z_grid[-1]
    ):
        raise ValueError("terrain coordinates lie outside the model domain")
    return coordinates


def _validated_grid(model: TerrainModel, values: Any, name: str) -> np.ndarray:
    grid = _validated_coordinates(model, values)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0.0):
        raise ValueError(f"{name} must be one-dimensional and increasing")
    return grid


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
