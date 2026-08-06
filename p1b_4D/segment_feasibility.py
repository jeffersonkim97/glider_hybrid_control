"""All-segment geometry certificates for straight physical edges.

The implemented terrain is a SciPy ``CubicSpline`` and the swept LOS boundary
is piecewise linear.  A straight edge is linear in horizontal position, so its
minimum clearance occurs at a finite set that can be enumerated exactly for
these representations: spline-interval endpoints and stationary points for
terrain, and LOS breakpoints for visibility/occlusion.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from .geometry import TerrainModel, los_boundary_height, terrain_height


LosRequirement = Literal["visible", "occluded", "none"]


def minimum_terrain_margin_on_segment(
    start: np.ndarray,
    end: np.ndarray,
    terrain_model: TerrainModel,
) -> dict[str, float]:
    """Return the global minimum of straight-edge altitude minus terrain.

    The minimization is exact up to floating-point polynomial root evaluation
    for the piecewise-cubic terrain representation stored in ``TerrainModel``.
    """
    start, end = _validated_forward_segment(start, end)
    z_start, h_start = (float(value) for value in start)
    z_end, h_end = (float(value) for value in end)
    _require_interval_in_domain(
        z_start, z_end, float(terrain_model.z_grid[0]),
        float(terrain_model.z_grid[-1]), "terrain"
    )
    slope = (h_end - h_start) / (z_end - z_start)
    spline = terrain_model.interpolant
    knots = np.asarray(spline.x, dtype=float)
    coefficients = np.asarray(spline.c, dtype=float)
    candidates: list[float] = [z_start, z_end]

    first_interval = max(0, int(np.searchsorted(knots, z_start, side="right") - 1))
    last_interval = min(
        knots.size - 2,
        int(np.searchsorted(knots, z_end, side="left")),
    )
    for interval in range(first_interval, last_interval + 1):
        left = max(z_start, float(knots[interval]))
        right = min(z_end, float(knots[interval + 1]))
        if right < left:
            continue
        candidates.extend((left, right))
        c3, c2, c1, _ = coefficients[:, interval]
        # terrain'(z) = edge slope is the stationarity condition for
        # edge_height(z) - terrain(z).  The local coordinate is
        # t = z - knots[interval].
        roots = _real_quadratic_roots(3.0 * c3, 2.0 * c2, c1 - slope)
        local_left = left - float(knots[interval])
        local_right = right - float(knots[interval])
        scale = max(1.0, abs(local_left), abs(local_right))
        tolerance = 128.0 * np.finfo(float).eps * scale
        for root in roots:
            if local_left - tolerance <= root <= local_right + tolerance:
                candidates.append(
                    float(knots[interval])
                    + min(max(root, local_left), local_right)
                )

    z_candidates = _unique_sorted(candidates, z_start, z_end)
    edge_height = h_start + slope * (z_candidates - z_start)
    margins = edge_height - terrain_height(terrain_model, z_candidates)
    index = int(np.argmin(margins))
    return {
        "minimum_margin": float(margins[index]),
        "argmin_z": float(z_candidates[index]),
        "candidate_count": int(z_candidates.size),
    }


def minimum_los_margin_on_segment(
    start: np.ndarray,
    end: np.ndarray,
    los_geometry: dict[str, Any],
    sensor_z: float,
    requirement: Literal["visible", "occluded"],
    los_tolerance: float = 0.0,
) -> dict[str, float | int | None]:
    """Return the global LOS clearance for one straight edge.

    ``visible`` uses ``edge_height - boundary`` and applies the constraint only
    to the portion at or left of the sensor. ``occluded`` uses
    ``boundary + los_tolerance - edge_height`` and requires the entire segment
    to stay on the pre-sensor side.
    """
    start, end = _validated_forward_segment(start, end)
    if requirement not in {"visible", "occluded"}:
        raise ValueError(f"Unsupported LOS requirement: {requirement}")
    if not math.isfinite(los_tolerance) or los_tolerance < 0.0:
        raise ValueError("los_tolerance must be finite and nonnegative")
    boundary = np.asarray(los_geometry["los_boundary"], dtype=float)
    if (
        boundary.ndim != 2
        or boundary.shape[1] != 2
        or boundary.shape[0] < 2
        or not np.all(np.isfinite(boundary))
        or not np.all(np.diff(boundary[:, 0]) > 0.0)
    ):
        raise ValueError("los_boundary must be a finite increasing Nx2 array")
    z_start, h_start = (float(value) for value in start)
    z_end, h_end = (float(value) for value in end)
    _require_interval_in_domain(
        z_start, z_end, float(boundary[0, 0]), float(boundary[-1, 0]), "LOS"
    )
    if not math.isfinite(sensor_z):
        raise ValueError("sensor_z must be finite")
    coordinate_scale = max(1.0, abs(z_start), abs(z_end), abs(sensor_z))
    coordinate_tolerance = 128.0 * np.finfo(float).eps * coordinate_scale
    if requirement == "occluded" and z_end > sensor_z + coordinate_tolerance:
        return {
            "minimum_margin": -math.inf,
            "argmin_z": float(sensor_z),
            "candidate_count": 0,
        }

    constrained_end = min(z_end, float(sensor_z))
    if z_start > constrained_end + coordinate_tolerance:
        # A glide segment entirely at/after the sensor is visible by the
        # finite model's rule and has no active LOS-boundary constraint.
        return {
            "minimum_margin": math.inf,
            "argmin_z": None,
            "candidate_count": 0,
        }
    constrained_end = max(z_start, constrained_end)
    internal = boundary[
        (boundary[:, 0] > z_start) & (boundary[:, 0] < constrained_end), 0
    ]
    z_candidates = _unique_sorted(
        [z_start, constrained_end, *internal.tolist()], z_start, constrained_end
    )
    slope = (h_end - h_start) / (z_end - z_start)
    edge_height = h_start + slope * (z_candidates - z_start)
    boundary_height = los_boundary_height(los_geometry, z_candidates)
    if requirement == "visible":
        margins = edge_height - boundary_height
    else:
        margins = boundary_height + los_tolerance - edge_height
    index = int(np.argmin(margins))
    return {
        "minimum_margin": float(margins[index]),
        "argmin_z": float(z_candidates[index]),
        "candidate_count": int(z_candidates.size),
    }


def certify_straight_segment_geometry(
    start: np.ndarray,
    end: np.ndarray,
    terrain_model: TerrainModel,
    los_geometry: dict[str, Any],
    sensor_z: float,
    airspace: dict[str, float],
    *,
    terrain_tolerance: float,
    los_requirement: LosRequirement,
    los_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Certify one complete straight segment under the implemented geometry."""
    start, end = _validated_segment_coordinates(start, end)
    if los_requirement not in {"visible", "occluded", "none"}:
        raise ValueError(f"Unsupported LOS requirement: {los_requirement}")
    if not math.isfinite(terrain_tolerance) or terrain_tolerance < 0.0:
        raise ValueError("terrain_tolerance must be finite and nonnegative")
    coordinate_scale = max(1.0, *(abs(float(value)) for value in (*start, *end)))
    coordinate_tolerance = 128.0 * np.finfo(float).eps * coordinate_scale
    if abs(float(end[0] - start[0])) <= coordinate_tolerance:
        return _certify_vertical_segment(
            start,
            end,
            terrain_model,
            los_geometry,
            sensor_z,
            airspace,
            terrain_tolerance=terrain_tolerance,
            los_requirement=los_requirement,
            los_tolerance=los_tolerance,
        )
    if float(end[0]) < float(start[0]):
        raise ValueError("straight-segment certificate requires increasing z")
    terrain = minimum_terrain_margin_on_segment(start, end, terrain_model)
    domain = _airspace_certificate(start, end, airspace)
    if los_requirement == "none":
        los = {
            "minimum_margin": math.inf,
            "argmin_z": None,
            "candidate_count": 0,
        }
    else:
        los = minimum_los_margin_on_segment(
            start, end, los_geometry, sensor_z, los_requirement, los_tolerance
        )
    terrain_clear = terrain["minimum_margin"] >= -terrain_tolerance
    los_clear = los["minimum_margin"] >= 0.0
    passed = bool(domain["inside"] and terrain_clear and los_clear)
    return {
        "passed": passed,
        "terrain_clear": bool(terrain_clear),
        "los_clear": bool(los_clear),
        "domain_clear": bool(domain["inside"]),
        "minimum_terrain_margin": float(terrain["minimum_margin"]),
        "terrain_argmin_z": float(terrain["argmin_z"]),
        "minimum_los_margin": float(los["minimum_margin"]),
        "los_argmin_z": los["argmin_z"],
        "minimum_airspace_margin": float(domain["minimum_margin"]),
        "terrain_candidate_count": int(terrain["candidate_count"]),
        "los_candidate_count": int(los["candidate_count"]),
        "los_requirement": los_requirement,
        "certificate": (
            "piecewise_cubic_terrain_and_piecewise_linear_los_global_minimum"
        ),
    }


def _certify_vertical_segment(
    start: np.ndarray,
    end: np.ndarray,
    terrain_model: TerrainModel,
    los_geometry: dict[str, Any],
    sensor_z: float,
    airspace: dict[str, float],
    *,
    terrain_tolerance: float,
    los_requirement: LosRequirement,
    los_tolerance: float,
) -> dict[str, Any]:
    z = float(start[0])
    minimum_h = min(float(start[1]), float(end[1]))
    maximum_h = max(float(start[1]), float(end[1]))
    terrain_margin = minimum_h - float(terrain_height(terrain_model, z))
    if los_requirement == "none" or (
        los_requirement == "visible" and z >= float(sensor_z)
    ):
        los_margin = math.inf
        los_argmin_z = None
        los_candidate_count = 0
    else:
        boundary = float(los_boundary_height(los_geometry, z))
        if los_requirement == "visible":
            los_margin = minimum_h - boundary
        else:
            los_margin = (
                boundary + los_tolerance - maximum_h
                if z <= float(sensor_z) else -math.inf
            )
        los_argmin_z = z
        los_candidate_count = 1
    domain = _airspace_certificate(start, end, airspace)
    terrain_clear = terrain_margin >= -terrain_tolerance
    los_clear = los_margin >= 0.0
    return {
        "passed": bool(domain["inside"] and terrain_clear and los_clear),
        "terrain_clear": bool(terrain_clear),
        "los_clear": bool(los_clear),
        "domain_clear": bool(domain["inside"]),
        "minimum_terrain_margin": float(terrain_margin),
        "terrain_argmin_z": z,
        "minimum_los_margin": float(los_margin),
        "los_argmin_z": los_argmin_z,
        "minimum_airspace_margin": float(domain["minimum_margin"]),
        "terrain_candidate_count": 1,
        "los_candidate_count": los_candidate_count,
        "los_requirement": los_requirement,
        "certificate": (
            "stationary_point_geometry_evaluation"
            if np.array_equal(start, end)
            else "vertical_segment_endpoint_extrema"
        ),
    }


def _airspace_certificate(
    start: np.ndarray, end: np.ndarray, airspace: dict[str, float]
) -> dict[str, float | bool]:
    required = {"z_min", "z_max", "h_min", "h_max"}
    if not isinstance(airspace, dict) or not required <= set(airspace):
        raise ValueError(f"airspace must contain {sorted(required)}")
    bounds = {name: float(airspace[name]) for name in required}
    if not all(math.isfinite(value) for value in bounds.values()):
        raise ValueError("airspace bounds must be finite")
    margins = (
        float(start[0]) - bounds["z_min"],
        bounds["z_max"] - float(end[0]),
        min(float(start[1]), float(end[1])) - bounds["h_min"],
        bounds["h_max"] - max(float(start[1]), float(end[1])),
    )
    minimum = min(margins)
    scale = max(1.0, *(abs(value) for value in (*start, *end, *bounds.values())))
    tolerance = 128.0 * np.finfo(float).eps * scale
    return {"inside": bool(minimum >= -tolerance), "minimum_margin": minimum}


def _validated_forward_segment(
    start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    start_array, end_array = _validated_segment_coordinates(start, end)
    if not float(end_array[0]) > float(start_array[0]):
        raise ValueError("straight-segment certificate requires increasing z")
    return start_array, end_array


def _validated_segment_coordinates(
    start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    if (
        start_array.shape != (2,)
        or end_array.shape != (2,)
        or not np.all(np.isfinite(start_array))
        or not np.all(np.isfinite(end_array))
    ):
        raise ValueError("start and end must contain two finite coordinates")
    return start_array, end_array


def _require_interval_in_domain(
    start: float, end: float, lower: float, upper: float, name: str
) -> None:
    scale = max(1.0, abs(start), abs(end), abs(lower), abs(upper))
    tolerance = 128.0 * np.finfo(float).eps * scale
    if start < lower - tolerance or end > upper + tolerance:
        raise ValueError(f"segment lies outside the {name} model domain")


def _real_quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    scale = max(1.0, abs(a), abs(b), abs(c))
    tolerance = 128.0 * np.finfo(float).eps * scale
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            return ()
        return (-c / b,)
    discriminant = b * b - 4.0 * a * c
    discriminant_tolerance = 256.0 * np.finfo(float).eps * max(
        1.0, b * b, abs(4.0 * a * c)
    )
    if discriminant < -discriminant_tolerance:
        return ()
    root_discriminant = math.sqrt(max(discriminant, 0.0))
    # The standard quadratic expression is adequate here because every
    # returned candidate is reevaluated in the original spline.  Include both
    # roots and deduplicate the double-root case below.
    roots = ((-b - root_discriminant) / (2.0 * a),
             (-b + root_discriminant) / (2.0 * a))
    if abs(roots[1] - roots[0]) <= tolerance:
        return (roots[0],)
    return roots


def _unique_sorted(
    values: list[float], lower: float, upper: float
) -> np.ndarray:
    ordered = np.asarray(sorted(float(value) for value in values), dtype=float)
    scale = max(1.0, abs(lower), abs(upper))
    tolerance = 128.0 * np.finfo(float).eps * scale
    unique: list[float] = []
    for value in ordered:
        clipped = min(max(value, lower), upper)
        if not unique or abs(clipped - unique[-1]) > tolerance:
            unique.append(clipped)
    return np.asarray(unique, dtype=float)
