"""Authoritative 6D local stage-cost map construction (3D extension).

Mirrors p1b_4D.stage_cost's role exactly, extended from a 4D (z, h, v,
gamma) J-map to a 6D (x, y, h, v, gamma, heading) J-map.

One deliberate departure from p1b_4D, flagged here and in the notebook:
p1b_4D's spatial_glide_valid requires `los_mask` (visible) for the ENTIRE
glide phase -- a redundant hard gate on top of the hazard-rate formula,
which already zeroes detection rate when occluded (see detection.py's
`los_visible` multiplier). That redundant gate is what makes 2D's
"hide again mid-glide" maneuver structurally impossible regardless of
course freedom. Since this 3D extension's whole point is to let lateral
motion exploit terrain masking (including re-hiding, e.g. a
helical descent inside the hill's shadow), spatial_glide_valid here only
requires being above terrain -- not also visible. The powered phase keeps
p1b_4D's non_visible requirement unchanged: it has no radar hazard term
at all (acoustic only), so it must stay in shadow to avoid a "radar-free
while visible" modeling loophole -- that reason is unrelated to the glide
question and still applies.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


def construct_stage_cost_6d(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Construct the complete local 6D glide cost and component maps.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.
    geometry_bundle:
        Successful Phase 2 GeometryBundle.
    detection_bundle:
        Successful Phase 3 symbolic DetectionBundle.

    Outputs
    -------
    dict
        Universal StageCost6DResult envelope containing J6D, component
        maps, feasible masks, standard grids, metadata, and validation.

    Assumptions
    -----------
    J6D is the local glide-mode Attacker objective. Powered acoustic local
    cost is preserved separately for the powered segment before switching.

    Notes
    -----
    This function performs local-stage evaluation only. It performs no
    Bellman propagation, cost-to-go computation, path extraction, or plotting.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    _require_successful_bundle(detection_bundle, "detection_bundle")
    configs = configuration_bundle["primary_result"]
    environment = configs["environment_config"]
    vehicle = configs["vehicle_config"]
    validation_config = configs["validation_config"]
    functions = detection_bundle["primary_result"]["functions"]
    geometry = geometry_bundle["primary_result"]

    grids = construct_state_grids(environment, vehicle)
    x_grid, y_grid, h_grid, v_grid, gamma_grid, heading_grid = (
        grids[name] for name in ("x", "y", "h", "v", "gamma", "heading")
    )
    shape = (x_grid.size, y_grid.size, h_grid.size, v_grid.size, gamma_grid.size, heading_grid.size)
    spatial_shape = (x_grid.size, y_grid.size, h_grid.size)
    spatial_size = int(np.prod(spatial_shape))

    mesh_x, mesh_y, mesh_h = np.meshgrid(x_grid, y_grid, h_grid, indexing="ij")
    flat_x = mesh_x.reshape(1, spatial_size)
    flat_y = mesh_y.reshape(1, spatial_size)
    flat_h = mesh_h.reshape(1, spatial_size)
    sensor_position = geometry["sensor_position"]
    sensor_arguments = (
        np.full((1, spatial_size), sensor_position[0]),
        np.full((1, spatial_size), sensor_position[1]),
        np.full((1, spatial_size), sensor_position[2]),
    )
    time_step = float(vehicle["time_step"])
    zero = np.zeros((1, spatial_size))
    stage_time = np.full((1, spatial_size), time_step)

    glide_function = functions["glide_detection_components"].map(spatial_size)
    powered_function = functions["powered_detection_components"].map(spatial_size)
    objective_function = functions["attacker_objective"].map(spatial_size)

    component_maps = {
        "stage_pod": np.empty(shape),
        "stage_time": np.full(shape, time_step),
        "acoustic_rate": np.empty(shape),
        "powered_stage_pod": np.empty(shape),
        "powered_stage_objective": np.empty(shape),
        "glide_detection_rate": np.empty(shape),
        "pod_normalized": np.empty(shape),
        "time_normalized": np.empty(shape),
        "stage_objective": np.empty(shape),
    }

    powered_outputs = _mapped_outputs(
        powered_function,
        flat_x, flat_y, flat_h,
        np.full((1, spatial_size), vehicle["powered_speed"]),
        *sensor_arguments,
    )
    acoustic_rate_3d = powered_outputs[0].reshape(spatial_shape)
    powered_rate_3d = powered_outputs[1].reshape(spatial_shape)
    powered_hazard = powered_rate_3d.reshape(1, spatial_size) * time_step
    powered_objective_outputs = _mapped_outputs(
        objective_function, powered_hazard, zero, stage_time, zero,
    )
    powered_pod_3d = powered_objective_outputs[0].reshape(spatial_shape)
    powered_objective_3d = powered_objective_outputs[-1].reshape(spatial_shape)

    for velocity_index, velocity in enumerate(v_grid):
        velocity_values = np.full((1, spatial_size), velocity)
        for gamma_index, gamma in enumerate(gamma_grid):
            gamma_values = np.full((1, spatial_size), gamma)
            for heading_index, heading in enumerate(heading_grid):
                heading_values = np.full((1, spatial_size), heading)
                outputs = _mapped_outputs(
                    glide_function,
                    flat_x, flat_y, flat_h,
                    velocity_values, gamma_values, heading_values,
                    *sensor_arguments,
                )
                glide_rate = outputs[7].reshape(spatial_shape)
                glide_hazard = glide_rate.reshape(1, spatial_size) * time_step
                objective_outputs = _mapped_outputs(
                    objective_function, zero, glide_hazard, zero, stage_time,
                )
                index = (slice(None), slice(None), slice(None), velocity_index, gamma_index, heading_index)
                component_maps["stage_pod"][index] = objective_outputs[0].reshape(spatial_shape)
                component_maps["glide_detection_rate"][index] = glide_rate
                component_maps["pod_normalized"][index] = objective_outputs[2].reshape(spatial_shape)
                component_maps["time_normalized"][index] = objective_outputs[3].reshape(spatial_shape)
                component_maps["stage_objective"][index] = objective_outputs[4].reshape(spatial_shape)
                component_maps["acoustic_rate"][index] = acoustic_rate_3d
                component_maps["powered_stage_pod"][index] = powered_pod_3d
                component_maps["powered_stage_objective"][index] = powered_objective_3d

    masks = construct_state_validity_masks(geometry_bundle, grids, vehicle)
    j6d = np.where(masks["feasible_mask"], component_maps["stage_objective"], np.inf)
    powered_stage_cost = np.where(
        masks["powered_feasible_mask"], component_maps["powered_stage_objective"], np.inf,
    )
    validation = validate_stage_cost_6d(
        j6d, powered_stage_cost, component_maps, masks, grids,
        configs["cost_config"], validation_config,
    )
    readonly_components = {name: _readonly(values) for name, values in component_maps.items()}
    readonly_masks = {name: _readonly(values) for name, values in masks.items()}
    return {
        "primary_result": {
            "j6d": _readonly(j6d),
            "powered_stage_cost_6d": _readonly(powered_stage_cost),
            "component_maps": readonly_components,
            "feasible_mask": readonly_masks["feasible_mask"],
            "validity_masks": readonly_masks,
            "grids": grids,
            "grid_metadata": {
                "shape": shape,
                "axis_order": ("x", "y", "h", "v", "gamma", "heading"),
                "state_count": int(np.prod(shape)),
                "units": {
                    "x": "m", "y": "m", "h": "m", "v": "m/s",
                    "gamma": "rad", "heading": "rad", "time": "s",
                },
            },
        },
        "validation": validation,
        "metadata": {
            "schema_name": "StageCost6DResult",
            "schema_version": "1.0.0",
            "producer_phase": 4,
            "producer_module": "p1b_3DExtension.stage_cost",
            "configuration_schema_version": configuration_bundle["metadata"]["schema_version"],
            "geometry_schema_version": geometry_bundle["metadata"]["schema_version"],
            "detection_schema_version": detection_bundle["metadata"]["schema_version"],
            "attacker_objective_id": configs["cost_config"]["attacker"]["objective_id"],
            "j6d_mode": "glide",
            "invalid_cost": "positive_infinity",
            "powered_cost_role": "pre_switch_local_component",
            "glide_feasibility_requires_los_visible": False,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "STAGE_COST_6D_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def construct_state_grids(
    environment: dict[str, Any],
    vehicle: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct the standard X, Y, H, velocity, gamma, heading discretization."""
    grid = environment["grid"]
    arrays = {
        "x": np.linspace(grid["x_min"], grid["x_max"], grid["x_count"]),
        "y": np.linspace(grid["y_min"], grid["y_max"], grid["y_count"]),
        "h": np.linspace(grid["h_min"], grid["h_max"], grid["h_count"]),
        "v": np.linspace(
            vehicle["glide_speed_min"], vehicle["glide_speed_max"], vehicle["glide_speed_count"],
        ),
        "gamma": np.deg2rad(
            np.linspace(vehicle["gamma_min_deg"], vehicle["gamma_max_deg"], vehicle["gamma_count"])
        ),
        # heading is periodic (-180 deg == +180 deg): endpoint=False avoids
        # duplicating one course/state heading at both ends of the grid, unlike
        # the other, non-periodic axes above.
        "heading": np.deg2rad(
            np.linspace(
                vehicle["heading_min_deg"], vehicle["heading_max_deg"],
                vehicle["heading_count"], endpoint=False,
            )
        ),
    }
    return {name: _readonly(values) for name, values in arrays.items()}


def construct_state_validity_masks(
    geometry_bundle: dict[str, Any],
    grids: dict[str, np.ndarray],
    vehicle: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct terrain, control, and phase-feasibility masks."""
    geometry = geometry_bundle["primary_result"]
    terrain_mask_3d = geometry["los_masks"]["terrain_mask"]
    non_visible_3d = geometry["los_masks"]["non_visible_airspace_mask"]
    x_grid, y_grid, h_grid, v_grid, gamma_grid, heading_grid = (
        grids[name] for name in ("x", "y", "h", "v", "gamma", "heading")
    )
    expected_spatial_shape = (x_grid.size, y_grid.size, h_grid.size)
    if terrain_mask_3d.shape != expected_spatial_shape:
        raise ValueError("Geometry terrain mask does not match the stage-cost grid")

    velocity_mesh, gamma_mesh = np.meshgrid(v_grid, gamma_grid, indexing="ij")
    velocity_valid = (
        (velocity_mesh >= vehicle["glide_speed_min"]) & (velocity_mesh <= vehicle["glide_speed_max"])
    )
    gamma_min = np.deg2rad(vehicle["gamma_min_deg"])
    gamma_max = np.deg2rad(vehicle["gamma_max_deg"])
    gamma_valid = (gamma_mesh >= gamma_min) & (gamma_mesh <= gamma_max)
    cl = (
        2.0 * vehicle["mass"] * vehicle["gravity"] * np.cos(gamma_mesh)
        / (vehicle["air_density"] * velocity_mesh**2 * vehicle["wing_area"])
    )
    cd = (
        vehicle["cd0"] + vehicle["linear_drag_coefficient"] * cl
        + vehicle["quadratic_drag_coefficient"] * cl**2
    )
    dynamic_valid = (
        (cl >= vehicle["dynamic_limits"]["cl_min"]) & (cl <= vehicle["dynamic_limits"]["cl_max"])
        & (cd > 0.0) & np.isfinite(cl) & np.isfinite(cd)
    )
    control_valid_2d = velocity_valid & gamma_valid & dynamic_valid  # (v, gamma)
    control_valid = np.broadcast_to(
        control_valid_2d[:, :, None], (v_grid.size, gamma_grid.size, heading_grid.size),
    )  # Local validity is heading-independent. Bellman separately enforces
       # the inter-segment turn-rate constraint using heading as a state.

    # See module docstring: glide feasibility requires only "above terrain",
    # not also "visible" -- the hazard-rate formula's own los_visible gate
    # already handles detection correctly, so this is not a redundant
    # weakening, it removes a separate, stricter, non-hazard-based gate.
    spatial_glide_valid = ~terrain_mask_3d
    spatial_powered_valid = ~terrain_mask_3d & non_visible_3d
    feasible_mask = (
        spatial_glide_valid[:, :, :, None, None, None]
        & control_valid[None, None, None, :, :, :]
    )
    powered_feasible_mask = np.broadcast_to(
        spatial_powered_valid[:, :, :, None, None, None], feasible_mask.shape,
    ).copy()
    return {
        "feasible_mask": feasible_mask,
        "powered_feasible_mask": powered_feasible_mask,
        "terrain_penetration_mask": np.broadcast_to(
            terrain_mask_3d[:, :, :, None, None, None], feasible_mask.shape,
        ).copy(),
        "control_valid_mask": np.broadcast_to(
            control_valid[None, None, None, :, :, :], feasible_mask.shape,
        ).copy(),
        "powered_spatial_feasibility_mask": powered_feasible_mask.copy(),
    }


def validate_stage_cost_6d(
    j6d: np.ndarray,
    powered_stage_cost: np.ndarray,
    components: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    grids: dict[str, np.ndarray],
    cost_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate grid, finite values, masks, normalization, and objectives."""
    expected_shape = tuple(grids[name].size for name in ("x", "y", "h", "v", "gamma", "heading"))
    feasible = masks["feasible_mask"]
    powered_feasible = masks["powered_feasible_mask"]
    objective_tolerance = validation_config["objective_tolerance"]
    reconstructed = (
        cost_config["attacker"]["w_pod"] * components["pod_normalized"]
        + cost_config["attacker"]["w_time"] * components["time_normalized"]
    )
    objective_residual = float(
        np.max(np.abs(reconstructed[feasible] - components["stage_objective"][feasible]))
    ) if np.any(feasible) else np.inf
    normalized_pod = components["pod_normalized"][feasible]
    normalized_time = components["time_normalized"][feasible]
    pod_specification = cost_config["attacker"]["normalization"]["pod"]
    if pod_specification["method"] == "cumulative_hazard_reference":
        expected_stage_pod = -np.expm1(-normalized_pod * pod_specification["hazard_reference"])
        pod_normalization_valid = bool(
            np.all(np.isfinite(normalized_pod)) and np.all(normalized_pod >= 0.0)
        )
    else:
        expected_stage_pod = normalized_pod
        pod_normalization_valid = bool(np.all((normalized_pod >= 0.0) & (normalized_pod <= 1.0)))
    checks = {
        "grid_dimensions": (
            j6d.shape == expected_shape and all(values.ndim == 1 for values in grids.values())
        ),
        "component_dimensions": all(values.shape == expected_shape for values in components.values()),
        "feasible_mask_dimensions": feasible.shape == expected_shape,
        "feasible_states_exist": bool(np.any(feasible)),
        "powered_feasible_states_exist": bool(np.any(powered_feasible)),
        "finite_feasible_evaluations": bool(np.all(np.isfinite(j6d[feasible]))),
        "infinite_invalid_cost": bool(np.all(np.isposinf(j6d[~feasible]))),
        "finite_powered_evaluations": bool(np.all(np.isfinite(powered_stage_cost[powered_feasible]))),
        "infinite_invalid_powered_cost": bool(
            np.all(np.isposinf(powered_stage_cost[~powered_feasible]))
        ),
        "pod_normalization": pod_normalization_valid,
        "time_normalization": bool(np.all(normalized_time >= 0.0)),
        "component_consistency": bool(
            np.allclose(
                components["stage_pod"][feasible], expected_stage_pod,
                rtol=0.0, atol=objective_tolerance,
            )
        ),
        "objective_consistency": objective_residual <= objective_tolerance,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    feasible_values = j6d[feasible]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "minimum_cost": float(np.min(feasible_values)) if feasible_values.size else float("nan"),
            "maximum_cost": float(np.max(feasible_values)) if feasible_values.size else float("nan"),
            "invalid_state_count": int(np.count_nonzero(~feasible)),
            "feasible_state_count": int(np.count_nonzero(feasible)),
            "powered_feasible_state_count": int(np.count_nonzero(powered_feasible)),
            "total_state_count": int(j6d.size),
            "objective_maximum_residual": objective_residual,
        },
        "tolerances": {"objective": objective_tolerance},
        "warnings": [],
        "failed_checks": failed_checks,
        "summary": (
            "Phase 4 6D stage-cost validation passed"
            if not failed_checks
            else f"Phase 4 6D stage-cost failed checks: {failed_checks}"
        ),
    }


def _mapped_outputs(function: ca.Function, *arguments: np.ndarray) -> list[np.ndarray]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [np.asarray(value, dtype=float) for value in outputs]


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict):
        raise TypeError(f"{name} must be a dictionary")
    if not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must have successful status")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
