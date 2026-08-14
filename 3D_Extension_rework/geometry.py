"""Stage 2 terrain, LOS, occlusion, and tangent-manifold geometry.

The implementation follows p1b_4D's outward LOS sweep process.  For every
horizontal target coordinate, the sensor-to-target ray is swept over the
terrain.  The most restrictive terrain sample defines the minimum visible
target altitude.  In 3D this produces a boundary surface H_LOS(x, y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class GaussianHill:
    center_x: float
    center_y: float
    peak_height: float
    width_x: float
    width_y: float


@dataclass(frozen=True)
class TerrainModel:
    scenario_id: str
    hills: tuple[GaussianHill, ...]

    @property
    def dominant_hill(self) -> GaussianHill:
        return max(self.hills, key=lambda hill: hill.peak_height)

    # Compatibility accessors for existing single-hill presentation code.
    @property
    def center_x(self) -> float:
        return self.dominant_hill.center_x

    @property
    def center_y(self) -> float:
        return self.dominant_hill.center_y

    @property
    def peak_height(self) -> float:
        return self.dominant_hill.peak_height


def build_terrain_model(configuration: dict[str, Any]) -> TerrainModel:
    terrain = configuration["environment"]["terrain"]
    return TerrainModel(
        scenario_id=str(terrain["scenario_id"]),
        hills=tuple(
            GaussianHill(
                center_x=float(hill["center_xy_m"][0]),
                center_y=float(hill["center_xy_m"][1]),
                peak_height=float(hill["height_m"]),
                width_x=float(hill["width_x_m"]),
                width_y=float(hill["width_y_m"]),
            )
            for hill in terrain["hills"]
        ),
    )


def terrain_height(model: TerrainModel, x: Any, y: Any) -> np.ndarray:
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
    )
    result = np.zeros(x_values.shape, dtype=float)
    for hill in model.hills:
        exponent = -0.5 * (
            ((x_values - hill.center_x) / hill.width_x) ** 2
            + ((y_values - hill.center_y) / hill.width_y) ** 2
        )
        result += hill.peak_height * np.exp(exponent)
    return result


def terrain_gradient(
    model: TerrainModel, x: Any, y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float),
    )
    gradient_x = np.zeros(x_values.shape, dtype=float)
    gradient_y = np.zeros(y_values.shape, dtype=float)
    for hill in model.hills:
        exponent = -0.5 * (
            ((x_values - hill.center_x) / hill.width_x) ** 2
            + ((y_values - hill.center_y) / hill.width_y) ** 2
        )
        height = hill.peak_height * np.exp(exponent)
        gradient_x -= (x_values - hill.center_x) * height / hill.width_x**2
        gradient_y -= (y_values - hill.center_y) * height / hill.width_y**2
    return gradient_x, gradient_y


def _distance_to_rectangle_boundary(
    origin: np.ndarray,
    direction: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> float:
    distances: list[float] = []
    if direction[0] > 0.0:
        distances.append((x_bounds[1] - origin[0]) / direction[0])
    elif direction[0] < 0.0:
        distances.append((x_bounds[0] - origin[0]) / direction[0])
    if direction[1] > 0.0:
        distances.append((y_bounds[1] - origin[1]) / direction[1])
    elif direction[1] < 0.0:
        distances.append((y_bounds[0] - origin[1]) / direction[1])
    positive = [value for value in distances if value > 0.0]
    return float(min(positive)) if positive else 0.0


def compute_tangent_manifold(
    model: TerrainModel,
    sensor_position: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    azimuth_count: int,
    range_sample_count: int,
) -> dict[str, np.ndarray]:
    """Find terrain contact points of sensor rays tangent to the hill.

    Each horizontal azimuth defines a vertical plane through the sensor.
    Along that ray, maximizing elevation slope (terrain-sensor height)/range
    is exactly the 3D counterpart of p1b_4D's minimum-slope tangent search.
    The maximizer is the ray/terrain tangent contact for that azimuth.
    """
    sensor_xy = np.asarray(sensor_position[:2], dtype=float)
    sensor_h = float(sensor_position[2])
    contacts: list[tuple[float, float, float]] = []
    rays: list[tuple[float, float, float]] = []
    azimuths: list[float] = []
    residuals: list[float] = []
    for azimuth in np.linspace(0.0, 2.0 * np.pi, azimuth_count, endpoint=False):
        direction = np.array([np.cos(azimuth), np.sin(azimuth)])
        maximum_range = _distance_to_rectangle_boundary(
            sensor_xy, direction, x_bounds, y_bounds,
        )
        if maximum_range <= 2.0:
            continue

        def negative_elevation_slope(distance: float) -> float:
            point = sensor_xy + distance * direction
            height = float(terrain_height(model, point[0], point[1]))
            return -(height - sensor_h) / distance

        # A multi-hill ray can have several local slope maxima.  Locate the
        # global sampled maximum first, then continuously refine only its
        # neighboring bracket.
        sampled_distance = np.linspace(1.0e-3, maximum_range, range_sample_count)
        sampled_xy = sensor_xy[None, :] + sampled_distance[:, None] * direction[None, :]
        sampled_slope = (
            terrain_height(model, sampled_xy[:, 0], sampled_xy[:, 1]) - sensor_h
        ) / sampled_distance
        maximum_index = int(np.argmax(sampled_slope))
        if maximum_index == 0 or maximum_index == range_sample_count - 1:
            continue
        result = minimize_scalar(
            negative_elevation_slope,
            bounds=(sampled_distance[maximum_index - 1], sampled_distance[maximum_index + 1]),
            method="bounded",
            options={"xatol": 1.0e-8, "maxiter": 200},
        )
        distance = float(result.x)
        elevation_slope = -float(result.fun)
        if (
            not result.success
            or elevation_slope <= 1.0e-7
            or distance <= 1.0
            or distance >= maximum_range - 1.0
        ):
            continue
        point_xy = sensor_xy + distance * direction
        point_h = float(terrain_height(model, point_xy[0], point_xy[1]))
        gradient_x, gradient_y = terrain_gradient(
            model, point_xy[0], point_xy[1],
        )
        directional_gradient = (
            float(gradient_x) * direction[0]
            + float(gradient_y) * direction[1]
        )
        residual = directional_gradient - elevation_slope
        contacts.append((float(point_xy[0]), float(point_xy[1]), point_h))
        rays.append((float(direction[0]), float(direction[1]), elevation_slope))
        azimuths.append(float(azimuth))
        residuals.append(float(residual))
    if not contacts:
        raise RuntimeError("No sensor-ray/terrain tangent contacts were found")
    return {
        "contact_points": np.asarray(contacts, dtype=float),
        "ray_directions_and_slopes": np.asarray(rays, dtype=float),
        "azimuths": np.asarray(azimuths, dtype=float),
        "tangent_residuals": np.asarray(residuals, dtype=float),
    }


def compute_los_boundary_surface(
    model: TerrainModel,
    sensor_position: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return H_LOS(x,y) and the governing ray fraction at each column."""
    sensor_x, sensor_y, sensor_h = np.asarray(sensor_position, dtype=float)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    boundary = np.full(mesh_x.shape, -np.inf, dtype=float)
    governing_fraction = np.zeros(mesh_x.shape, dtype=float)
    for fraction in np.linspace(0.0, 1.0, sample_count)[1:]:
        sample_x = sensor_x + fraction * (mesh_x - sensor_x)
        sample_y = sensor_y + fraction * (mesh_y - sensor_y)
        sample_terrain = terrain_height(model, sample_x, sample_y)
        required_height = sensor_h + (sample_terrain - sensor_h) / fraction
        update = required_height > boundary
        boundary[update] = required_height[update]
        governing_fraction[update] = fraction
    return boundary, governing_fraction


def build_geometry(
    configuration: dict[str, Any], *, require_tangent_manifold: bool = True,
) -> dict[str, Any]:
    environment = configuration["environment"]
    grid = configuration["grid"]
    validation = configuration["validation"]
    x_bounds = tuple(float(value) for value in environment["x_bounds_m"])
    y_bounds = tuple(float(value) for value in environment["y_bounds_m"])
    h_bounds = tuple(float(value) for value in environment["h_bounds_m"])
    x_grid = np.linspace(*x_bounds, int(grid["x_count"]))
    y_grid = np.linspace(*y_bounds, int(grid["y_count"]))
    h_grid = np.linspace(*h_bounds, int(grid["h_count"]))
    model = build_terrain_model(configuration)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    sampled_terrain = terrain_height(model, mesh_x, mesh_y)

    sensor_xy = np.asarray(environment["sensor_xy_m"], dtype=float)
    sensor_h = float(terrain_height(model, *sensor_xy)) + float(
        environment["sensor_mount_height_m"]
    )
    sensor_position = np.array([*sensor_xy, sensor_h], dtype=float)
    launch_xy = np.asarray(environment["launch_xy_m"], dtype=float)
    launch_position = np.array([
        *launch_xy, float(terrain_height(model, *launch_xy)),
    ])
    goal_xy = np.asarray(environment["goal_xy_m"], dtype=float)
    goal_position = np.array([
        *goal_xy, float(terrain_height(model, *goal_xy)),
    ])

    los_boundary, governing_fraction = compute_los_boundary_surface(
        model, sensor_position, x_grid, y_grid,
        int(grid["los_ray_sample_count"]),
    )
    terrain_mask = h_grid[None, None, :] <= (
        sampled_terrain[:, :, None]
        + float(validation["terrain_tolerance_m"])
    )
    non_visible_airspace = (
        (h_grid[None, None, :] < los_boundary[:, :, None])
        & ~terrain_mask
    )
    occlusion_mask = terrain_mask | non_visible_airspace
    los_mask = ~occlusion_mask
    if require_tangent_manifold:
        tangent = compute_tangent_manifold(
            model, sensor_position, x_bounds, y_bounds,
            int(grid["tangent_azimuth_count"]),
            int(grid["tangent_range_sample_count"]),
        )

        # A genuine tangent ray may touch the terrain, but it must not pass through
        # it before reaching the contact point.
        contact_points = tangent["contact_points"]
        ray_fractions = np.linspace(0.0, 1.0, 129)[None, :]
        ray_x = sensor_position[0] + (
            contact_points[:, 0] - sensor_position[0]
        )[:, None] * ray_fractions
        ray_y = sensor_position[1] + (
            contact_points[:, 1] - sensor_position[1]
        )[:, None] * ray_fractions
        ray_h = sensor_position[2] + (
            contact_points[:, 2] - sensor_position[2]
        )[:, None] * ray_fractions
        tangent_ray_clearance = ray_h - terrain_height(model, ray_x, ray_y)
        minimum_tangent_ray_clearance = float(np.min(tangent_ray_clearance))
    else:
        tangent = {
            "contact_points": np.empty((0, 3), dtype=float),
            "ray_directions_and_slopes": np.empty((0, 3), dtype=float),
            "azimuths": np.empty(0, dtype=float),
            "tangent_residuals": np.empty(0, dtype=float),
        }
        minimum_tangent_ray_clearance = float("nan")

    cell_volume = float(
        (x_grid[1] - x_grid[0])
        * (y_grid[1] - y_grid[0])
        * (h_grid[1] - h_grid[0])
    )
    admissible_mask = ~terrain_mask
    admissible_volume = float(np.count_nonzero(admissible_mask) * cell_volume)
    los_volume = float(np.count_nonzero(los_mask) * cell_volume)
    checks = {
        "terrain_peak": bool(
            float(np.max(sampled_terrain)) >= 0.95 * model.peak_height
            and float(np.max(sampled_terrain))
            <= sum(hill.peak_height for hill in model.hills) + 1.0e-10
        ),
        "mask_partition": np.array_equal(los_mask, ~occlusion_mask),
        "airspace_partition": np.array_equal(
            los_mask | non_visible_airspace, admissible_mask,
        ),
        "boundary_above_terrain": bool(np.all(
            los_boundary >= sampled_terrain - 1.0e-8
        )),
        "tangent_manifold_nonempty": bool(
            not require_tangent_manifold or tangent["contact_points"].shape[0] > 2
        ),
        "tangent_residual": bool(
            not require_tangent_manifold
            or np.max(np.abs(tangent["tangent_residuals"]))
            <= float(validation["tangent_residual_tolerance"])
        ),
        "tangent_rays_clear_terrain": bool(
            not require_tangent_manifold
            or minimum_tangent_ray_clearance >= -1.0e-8
        ),
        "sensor_on_terrain": bool(abs(
            sensor_position[2] - float(terrain_height(model, *sensor_xy))
        ) < 1.0e-10),
        "goal_on_terrain": bool(abs(
            goal_position[2] - float(terrain_height(model, *goal_xy))
        ) < 1.0e-10),
    }
    return {
        "configuration": configuration,
        "terrain_model": model,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "h_grid": h_grid,
        "terrain_height": sampled_terrain,
        "sensor_position": sensor_position,
        "launch_position": launch_position,
        "goal_position": goal_position,
        "los_boundary_height": los_boundary,
        "governing_ray_fraction": governing_fraction,
        "terrain_mask": terrain_mask,
        "los_mask": los_mask,
        "occlusion_mask": occlusion_mask,
        "non_visible_airspace_mask": non_visible_airspace,
        "tangent_manifold": tangent,
        "coverage": {
            "cell_volume_m3": cell_volume,
            "admissible_volume_m3": admissible_volume,
            "los_volume_m3": los_volume,
            "normalized_los_volume": los_volume / admissible_volume,
        },
        "validation": {
            "passed": all(checks.values()),
            "checks": checks,
            "tangent_manifold_required": bool(require_tangent_manifold),
            "maximum_tangent_residual": float(
                np.max(np.abs(tangent["tangent_residuals"]))
                if require_tangent_manifold else np.nan
            ),
            "minimum_tangent_ray_clearance_m": minimum_tangent_ray_clearance,
            "tangent_contact_count": int(tangent["contact_points"].shape[0]),
        },
    }
