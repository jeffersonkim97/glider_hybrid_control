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
    sampled_height = terrain["h_ridge"] * np.exp(
        -0.5 * ((z_grid - terrain["z_ridge"]) / terrain["width"]) ** 2
    )
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
    The existing single-ridge geometry has a sensor-visible tangent left of
    the sensor. Airspace cells at or below terrain are occluded.

    Notes
    -----
    Every residual sign change is refined. Selecting the minimum ray slope
    preserves the existing ridge tangent when a non-occluding stationary root
    appears over the flat terrain tail for a terrain-following sensor.
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
    chord_denominator = candidates - z_sensor
    residual = (
        terrain_height(terrain_model, candidates) - h_sensor
    ) / chord_denominator - terrain_gradient(terrain_model, candidates)
    changes = np.flatnonzero(
        np.signbit(residual[:-1]) != np.signbit(residual[1:])
    )
    if changes.size == 0:
        raise ValueError("No sensor-visible LOS tangent sign change was found")

    iterations = int(sensor["los"]["tangent_bisection_iterations"])
    if iterations <= 0:
        raise ValueError("tangent_bisection_iterations must be positive")
    roots: list[tuple[float, float, float]] = []
    for change in changes:
        left = float(candidates[change])
        right = float(candidates[change + 1])
        for _ in range(iterations):
            middle = 0.5 * (left + right)
            left_residual = _tangent_residual(
                terrain_model, left, z_sensor, h_sensor
            )
            middle_residual = _tangent_residual(
                terrain_model, middle, z_sensor, h_sensor
            )
            if np.signbit(left_residual) == np.signbit(middle_residual):
                left = middle
            else:
                right = middle
        root_z = 0.5 * (left + right)
        root_h = float(terrain_height(terrain_model, root_z))
        root_slope = (root_h - h_sensor) / (root_z - z_sensor)
        roots.append((float(root_slope), float(root_z), root_h))
    tangent_slope, tangent_z, tangent_h = min(roots, key=lambda item: item[0])
    tangent_intercept = h_sensor - tangent_slope * z_sensor
    boundary_height = tangent_slope * z_values + tangent_intercept
    los_boundary = np.column_stack((z_values, boundary_height))

    mesh_z, mesh_h = np.meshgrid(z_values, h_values, indexing="ij")
    terrain_tolerance = float(validation["terrain_tolerance"])
    terrain_heights = terrain_height(terrain_model, z_values)
    terrain_mask = mesh_h <= terrain_heights[:, None] + terrain_tolerance
    non_visible_airspace = (
        (mesh_z < tangent_z)
        & (mesh_h < tangent_slope * mesh_z + tangent_intercept)
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


def _tangent_residual(
    model: TerrainModel, z: float, z_sensor: float, h_sensor: float
) -> float:
    return float(
        (terrain_height(model, z) - h_sensor) / (z - z_sensor)
        - terrain_gradient(model, z)
    )


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
