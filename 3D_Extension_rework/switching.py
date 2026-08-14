"""Process-faithful LOS-tangent-manifold switching-point selection.

This is the 3D extension of ``p1b_4D.successor_grid_solver``'s continuous
switching state.  The terrain contact curve generates the tangent-ray
boundary surface H_LOS(x,y); just as the 2D solver samples points along the
tangent line rather than switching at the terrain contact itself, this solver
samples switching points on that boundary surface.  A straight powered
segment reaches that exact point, then a physical virtual glide edge connects
it to the already-solved Bellman lattice.
The projected V3D heatmap is never used for planning: heading-state V4D and
the exact incoming powered heading determine the feasible continuation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bellman import _control_is_valid, _incremental_objective, _signed_heading_change
from .geometry import terrain_height


def _readonly(values: Any) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result


def _segment_samples(start: np.ndarray, end: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.linspace(0.0, 1.0, int(count))
    return fractions, start[None, :] + fractions[:, None] * (end - start)[None, :]


def _los_interpolator(geometry: dict[str, Any]) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]),
        geometry["los_boundary_height"], method="linear",
        bounds_error=False, fill_value=np.inf,
    )


def _certify_segment(
    start: np.ndarray,
    end: np.ndarray,
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    *,
    los_requirement: str,
    sample_count: int,
    los_boundary: RegularGridInterpolator | None = None,
) -> dict[str, Any]:
    """Apply the same full-segment sampled geometry contract as Bellman."""
    _, path = _segment_samples(start, end, sample_count)
    environment = configuration["environment"]
    x_bounds = environment["x_bounds_m"]
    y_bounds = environment["y_bounds_m"]
    h_bounds = environment["h_bounds_m"]
    tolerance = float(configuration["validation"]["terrain_tolerance_m"])
    domain = (
        (path[:, 0] >= x_bounds[0]) & (path[:, 0] <= x_bounds[1])
        & (path[:, 1] >= y_bounds[0]) & (path[:, 1] <= y_bounds[1])
        & (path[:, 2] >= h_bounds[0]) & (path[:, 2] <= h_bounds[1])
    )
    terrain_margin = path[:, 2] - terrain_height(
        geometry["terrain_model"], path[:, 0], path[:, 1],
    ) - float(configuration["bellman"]["terrain_clearance_m"])
    boundary = (los_boundary or _los_interpolator(geometry))(path[:, :2])
    if los_requirement == "occluded":
        los_margin = boundary - path[:, 2]
    elif los_requirement == "visible":
        los_margin = path[:, 2] - boundary
    else:
        raise ValueError("los_requirement must be 'occluded' or 'visible'")
    terrain_clear = bool(np.all(terrain_margin >= -tolerance))
    los_clear = bool(np.all(los_margin >= -tolerance))
    domain_clear = bool(np.all(domain))
    return {
        "passed": terrain_clear and los_clear and domain_clear,
        "terrain_clear": terrain_clear,
        "los_clear": los_clear,
        "domain_clear": domain_clear,
        "minimum_terrain_margin_m": float(np.min(terrain_margin)),
        "minimum_los_margin_m": float(np.min(los_margin)),
        "terrain_argmin_fraction": float(np.argmin(terrain_margin) / (path.shape[0] - 1)),
        "los_argmin_fraction": float(np.argmin(los_margin) / (path.shape[0] - 1)),
        "sample_count": int(path.shape[0]),
    }


def _powered_rate(points: np.ndarray, speed: float, configuration: dict[str, Any], geometry: dict[str, Any]) -> np.ndarray:
    detection = configuration["detection"]
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    sensor_range = np.maximum(
        np.linalg.norm(sensor[None, :] - points, axis=1),
        float(detection["range_floor_m"]),
    )
    return (
        float(detection["acoustic_rate_scale"])
        * float(detection["acoustic_coefficient"])
        * speed ** int(detection["acoustic_speed_exponent"])
        / sensor_range**2
    )


def _glide_rate(
    points: np.ndarray,
    speed: float,
    gamma: float,
    heading: float,
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    los_boundary: RegularGridInterpolator | None = None,
) -> np.ndarray:
    detection = configuration["detection"]
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    delta = sensor[None, :] - points
    sensor_range = np.maximum(
        np.linalg.norm(delta, axis=1), float(detection["range_floor_m"]),
    )
    velocity = speed * np.array([
        math.cos(gamma) * math.cos(heading),
        math.cos(gamma) * math.sin(heading),
        math.sin(gamma),
    ])
    radial = np.sum(delta * velocity[None, :], axis=1) / sensor_range
    cosine_aspect = np.clip(radial / max(speed, 1.0e-9), -1.0, 1.0)
    rcs = float(detection["rcs_min"]) + (
        float(detection["rcs_max"]) - float(detection["rcs_min"])
    ) * cosine_aspect**2
    visible = points[:, 2] >= (
        los_boundary or _los_interpolator(geometry)
    )(points[:, :2])
    return visible.astype(float) * (
        float(detection["radar_rate_scale"])
        * float(detection["radar_coefficient"]) * rcs / sensor_range**4
        + float(detection["radial_velocity_rate_scale"])
        * float(detection["doppler_coefficient"]) * radial**2 / sensor_range**4
    )


def evaluate_powered_segment(
    switching_point: np.ndarray,
    configuration: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate straight powered flight from launch to one exact contact."""
    launch = np.asarray(geometry["launch_position"], dtype=float)
    switching_point = np.asarray(switching_point, dtype=float)
    count = int(configuration["bellman"]["powered_segment_quadrature_count"])
    fractions, path = _segment_samples(launch, switching_point, count)
    delta = switching_point - launch
    length = float(np.linalg.norm(delta))
    horizontal = float(np.hypot(delta[0], delta[1]))
    speed = float(configuration["vehicle"]["powered_speed_mps"])
    duration = length / speed
    heading = math.atan2(float(delta[1]), float(delta[0]))
    gamma = math.atan2(float(delta[2]), horizontal)
    rate = _powered_rate(path, speed, configuration, geometry)
    hazard = float(np.trapezoid(rate, fractions) * duration)
    cost = float(_incremental_objective(
        np.asarray(hazard), np.asarray(duration), configuration["cost"]["attacker"],
    ))
    certificate = _certify_segment(
        launch, switching_point, configuration, geometry,
        los_requirement="occluded", sample_count=count,
    )
    return {
        "path": _readonly(path),
        "duration_s": duration,
        "hazard": hazard,
        "cost": cost,
        "heading_rad": heading,
        "gamma_rad": gamma,
        "certificate": certificate,
        "validation": {
            "passed": bool(certificate["passed"]),
            "summary": (
                "powered segment feasible" if certificate["passed"]
                else "powered segment violates terrain, occlusion, or domain"
            ),
        },
    }


def _best_downstream_from_node(
    node_index: tuple[int, int, int],
    incoming_heading: float,
    configuration: dict[str, Any],
    graph: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    xi, yi, hi = node_index
    if bool(policy["goal_mask"][xi, yi, hi]):
        return {"cost": 0.0, "hazard": 0.0, "action_index": -1}
    maximum_turn_rate = math.radians(
        float(configuration["vehicle"]["max_turn_rate_deg_s"])
    )
    best: dict[str, Any] | None = None
    for action_index, action in enumerate(graph["actions"]):
        if not bool(graph["valid"][xi, yi, hi, action_index]):
            continue
        heading_change = abs(float(_signed_heading_change(
            action["heading_rad"], incoming_heading,
        )))
        if heading_change > maximum_turn_rate * action["duration_s"] + 1.0e-12:
            continue
        if bool(graph["terminal"][xi, yi, hi, action_index]):
            downstream_cost = 0.0
            downstream_hazard = 0.0
        else:
            next_index = (
                xi + int(action["forward_cells"]),
                yi + int(action["lateral_cells"]),
                hi - int(action["descent_cells"]),
                int(action["heading_state_index"]),
            )
            downstream_cost = float(policy["value_heading_state"][next_index])
            downstream_hazard = float(policy["hazard_to_go_heading_state"][next_index])
            if not np.isfinite(downstream_cost):
                continue
        candidate = {
            "cost": float(graph["cost"][xi, yi, hi, action_index]) + downstream_cost,
            "hazard": float(graph["hazard"][xi, yi, hi, action_index]) + downstream_hazard,
            "action_index": int(action_index),
        }
        if best is None or (candidate["cost"], action_index) < (best["cost"], best["action_index"]):
            best = candidate
    return best


def _virtual_target_indices(
    switching_point: np.ndarray, graph: dict[str, Any], configuration: dict[str, Any],
) -> tuple[tuple[int, int, int], ...]:
    grids = graph["grids"]
    x_grid, y_grid, h_grid = grids["x"], grids["y"], grids["h"]
    options = configuration["bellman"]
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    dh = float(h_grid[1] - h_grid[0])
    tolerance = 1.0e-10
    x_indices = np.flatnonzero(
        (x_grid - switching_point[0] > tolerance)
        & (x_grid - switching_point[0] <= options["maximum_forward_cells"] * dx + tolerance)
    )
    y_indices = np.flatnonzero(
        np.abs(y_grid - switching_point[1])
        <= options["maximum_lateral_cells"] * dy + tolerance
    )
    h_indices = np.flatnonzero(
        (switching_point[2] - h_grid > tolerance)
        & (switching_point[2] - h_grid <= options["maximum_descent_cells"] * dh + tolerance)
    )
    return tuple(
        (int(xi), int(yi), int(hi))
        for xi in x_indices for yi in y_indices for hi in h_indices
    )


def _evaluate_virtual_edge(
    start: np.ndarray,
    end: np.ndarray,
    speed: float,
    powered_heading: float,
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    los_boundary: RegularGridInterpolator,
) -> dict[str, Any] | None:
    delta = end - start
    horizontal = float(np.hypot(delta[0], delta[1]))
    if delta[0] <= 0.0 or delta[2] >= 0.0 or horizontal <= 0.0:
        return None
    length = float(np.linalg.norm(delta))
    gamma = math.atan2(float(delta[2]), horizontal)
    heading = math.atan2(float(delta[1]), float(delta[0]))
    vehicle = configuration["vehicle"]
    if not (
        math.radians(vehicle["gamma_min_deg"]) <= gamma
        <= math.radians(vehicle["gamma_max_deg"])
        and _control_is_valid(speed, gamma, vehicle)
    ):
        return None
    duration = length / speed
    maximum_turn = math.radians(vehicle["max_turn_rate_deg_s"]) * duration
    if abs(float(_signed_heading_change(heading, powered_heading))) > maximum_turn + 1.0e-12:
        return None
    count = int(configuration["bellman"]["virtual_edge_quadrature_count"])
    fractions, path = _segment_samples(start, end, count)
    certificate = _certify_segment(
        start, end, configuration, geometry,
        los_requirement="visible", sample_count=count,
        los_boundary=los_boundary,
    )
    if not certificate["passed"]:
        return None
    rate = _glide_rate(
        path, speed, gamma, heading, configuration, geometry, los_boundary,
    )
    hazard = float(np.trapezoid(rate, fractions) * duration)
    cost = float(_incremental_objective(
        np.asarray(hazard), np.asarray(duration), configuration["cost"]["attacker"],
    ))
    return {
        "path": _readonly(path), "duration_s": duration,
        "hazard": hazard, "cost": cost, "heading_rad": heading,
        "gamma_rad": gamma, "certificate": certificate,
    }


def generate_switching_surface_seeds(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    graph: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate the Bellman horizontal lattice on the LOS tangent surface.

    This is the direct 3D analogue of p1b_4D's exhaustive z-grid enumeration
    on its tangent boundary line.  Heights are evaluated continuously from
    H_LOS and are not snapped to the altitude grid.
    """
    x_grid, y_grid = graph["grids"]["x"], graph["grids"]["y"]
    launch = np.asarray(geometry["launch_position"], dtype=float)
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    h_min, h_max = configuration["environment"]["h_bounds_m"]
    stride = int(configuration["bellman"]["switching_manifold_stride"])
    horizontal_indices: list[tuple[int, int]] = []
    points: list[tuple[float, float, float]] = []
    boundary = _los_interpolator(geometry)
    for xi in range(0, x_grid.size, stride):
        x = float(x_grid[xi])
        if x < launch[0] or x >= sensor[0]:
            continue
        for yi in range(0, y_grid.size, stride):
            y = float(y_grid[yi])
            height = float(boundary(np.array([[x, y]]))[0])
            if not np.isfinite(height) or height < h_min or height > h_max:
                continue
            horizontal_indices.append((xi, yi))
            points.append((x, y, height))
    if not points:
        raise RuntimeError("No Bellman horizontal nodes lie on the in-airspace LOS boundary")
    return np.asarray(points, dtype=float), np.asarray(horizontal_indices, dtype=np.int32)


def select_switching_point(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    cost_to_go_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Exhaustively select the minimum sampled tangent-manifold response."""
    if not cost_to_go_bundle.get("status", {}).get("success", False):
        raise ValueError("cost-to-go bundle must pass validation")
    graph = cost_to_go_bundle["graph"]
    policy = cost_to_go_bundle["policy"]
    switching_points, horizontal_indices = generate_switching_surface_seeds(
        configuration, geometry, graph,
    )
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    candidates: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    speeds = np.asarray(graph["grids"]["v"], dtype=float)
    los_boundary = _los_interpolator(geometry)

    for manifold_index, switching_point in enumerate(switching_points):
        tangent_azimuth = math.atan2(
            float(switching_point[1] - sensor[1]),
            float(switching_point[0] - sensor[0]),
        )
        powered = evaluate_powered_segment(switching_point, configuration, geometry)
        attempt = {
            "manifold_index": int(manifold_index),
            "horizontal_grid_index": tuple(int(value) for value in horizontal_indices[manifold_index]),
            "azimuth_rad": tangent_azimuth,
            "switching_point": _readonly(switching_point),
            "powered_feasible": bool(powered["validation"]["passed"]),
            "virtual_target_count": 0,
            "feasible_virtual_edge_count": 0,
            "success": False,
        }
        if not powered["validation"]["passed"]:
            attempt["diagnostic"] = powered["validation"]["summary"]
            attempts.append(attempt)
            continue

        best_connection: dict[str, Any] | None = None
        target_indices = _virtual_target_indices(switching_point, graph, configuration)
        attempt["virtual_target_count"] = len(target_indices)
        for target_index in target_indices:
            target = np.array([
                graph["grids"]["x"][target_index[0]],
                graph["grids"]["y"][target_index[1]],
                graph["grids"]["h"][target_index[2]],
            ])
            edge_heading = math.atan2(
                float(target[1] - switching_point[1]),
                float(target[0] - switching_point[0]),
            )
            downstream = _best_downstream_from_node(
                target_index, edge_heading, configuration, graph, policy,
            )
            if downstream is None:
                continue
            for speed_index, speed in enumerate(speeds):
                edge = _evaluate_virtual_edge(
                    switching_point, target, float(speed), powered["heading_rad"],
                    configuration, geometry, los_boundary,
                )
                if edge is None:
                    continue
                attempt["feasible_virtual_edge_count"] += 1
                total_glide_cost = edge["cost"] + downstream["cost"]
                record = {
                    "target_index": target_index,
                    "target": _readonly(target),
                    "speed_index": int(speed_index),
                    "speed_mps": float(speed),
                    "edge": edge,
                    "downstream": downstream,
                    "total_glide_cost": float(total_glide_cost),
                    "total_glide_hazard": float(edge["hazard"] + downstream["hazard"]),
                }
                key = (total_glide_cost, *target_index, speed_index)
                if best_connection is None or key < best_connection["selection_key"]:
                    record["selection_key"] = key
                    best_connection = record
        if best_connection is None:
            attempt["diagnostic"] = "no_finite_physical_virtual_edge_to_bellman_policy"
            attempts.append(attempt)
            continue

        mission_cost = powered["cost"] + best_connection["total_glide_cost"]
        mission_hazard = powered["hazard"] + best_connection["total_glide_hazard"]
        candidate = {
            "candidate_index": len(candidates),
            "manifold_index": int(manifold_index),
            "horizontal_grid_index": tuple(int(value) for value in horizontal_indices[manifold_index]),
            "azimuth_rad": tangent_azimuth,
            "switching_point": _readonly(switching_point),
            "powered": powered,
            "connection": best_connection,
            "mission_cost": float(mission_cost),
            "mission_hazard": float(mission_hazard),
            "mission_pod": float(1.0 - math.exp(-mission_hazard)),
        }
        candidates.append(candidate)
        attempt.update({"success": True, "diagnostic": "candidate_generated"})
        attempts.append(attempt)

    if not candidates:
        raise RuntimeError("No tangent-manifold switching candidate reaches the Bellman policy")
    ordered = sorted(
        candidates,
        key=lambda item: (item["mission_cost"], item["manifold_index"]),
    )
    best = ordered[0]
    exact_ties = tuple(
        candidate for candidate in ordered
        if candidate["mission_cost"] == best["mission_cost"]
    )
    manifold_cost = np.full(switching_points.shape[0], np.inf)
    manifold_pod = np.full(switching_points.shape[0], np.nan)
    for candidate in candidates:
        manifold_cost[candidate["manifold_index"]] = candidate["mission_cost"]
        manifold_pod[candidate["manifold_index"]] = candidate["mission_pod"]
    point_residual = float(np.linalg.norm(
        best["switching_point"] - switching_points[best["manifold_index"]]
    ))
    boundary_residual = abs(float(
        best["switching_point"][2]
        - _los_interpolator(geometry)(best["switching_point"][None, :2])[0]
    ))
    checks = {
        "feasible_candidates_exist": bool(candidates),
        "selected_point_is_exact_manifold_sample": point_residual == 0.0,
        "selected_point_is_on_los_boundary_surface": boundary_residual <= 1.0e-10,
        "selected_powered_segment_is_occluded": bool(
            best["powered"]["certificate"]["los_clear"]
        ),
        "selected_virtual_edge_is_visible": bool(
            best["connection"]["edge"]["certificate"]["los_clear"]
        ),
        "selected_cost_is_exact_sampled_minimum": bool(
            best["mission_cost"] == min(item["mission_cost"] for item in candidates)
        ),
        "projected_v3d_not_used_for_selection": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "best": best,
        "candidates": tuple(candidates),
        "attempts": tuple(attempts),
        "switching_surface_points": _readonly(switching_points),
        "switching_surface_horizontal_indices": _readonly(horizontal_indices),
        "manifold_cost": _readonly(manifold_cost),
        "manifold_pod": _readonly(manifold_pod),
        "metadata": {
            "source_process": "p1b_4D physical virtual switching state",
            "candidate_geometry": "Bellman horizontal lattice sampled on H_LOS tangent boundary surface",
            "terrain_contact_role": "generates tangent rays; not itself the switching locus",
            "candidate_count_evaluated": int(switching_points.shape[0]),
            "feasible_candidate_count": len(candidates),
            "exact_minimum_tie_count": len(exact_ties),
            "selection_rule": "minimum mission cost, then smallest manifold index for exact ties",
            "selection_value_source": "heading-state Bellman V4D through physical virtual edge",
            "projected_v3d_role": "visualization_only",
            "endpoint_snapping": False,
            "continuous_refinement_applied": False,
        },
        "validation": {
            "passed": not failed,
            "checks": checks,
            "failed_checks": failed,
            "metrics": {
                "selected_manifold_index": int(best["manifold_index"]),
                "selected_mission_cost": float(best["mission_cost"]),
                "selected_mission_pod": float(best["mission_pod"]),
                "exact_minimum_tie_count": len(exact_ties),
                "manifold_point_residual_m": point_residual,
                "los_boundary_residual_m": boundary_residual,
            },
        },
        "status": {
            "success": not failed,
            "message": "tangent-manifold switching selection passed" if not failed else f"failed: {failed}",
        },
    }
