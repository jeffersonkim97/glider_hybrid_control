"""Authoritative dense 6D local stage cost for the clean 3D rework.

This module is a direct process extension of ``p1b_4D.stage_cost``:

* ``J4D(z,h,v,gamma)`` becomes ``J6D(x,y,h,v,gamma,psi)``;
* glide feasibility remains terrain-free AND LOS-visible;
* powered feasibility remains terrain-free AND occluded;
* the completed detection, hazard, time, and vehicle-polar definitions are
  reused without retuning.

It does not compute a Bellman value function or cost-to-go projection.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np


AXIS_ORDER = ("x", "y", "h", "v", "gamma", "psi")


def construct_state_grids(configuration: dict[str, Any]) -> dict[str, np.ndarray]:
    """Construct the nested 6D state/control grid."""
    environment = configuration["environment"]
    counts = configuration["state_grid"]
    vehicle = configuration["vehicle"]
    grids = {
        "x": np.linspace(*environment["x_bounds_m"], int(counts["x_count"])),
        "y": np.linspace(*environment["y_bounds_m"], int(counts["y_count"])),
        "h": np.linspace(*environment["h_bounds_m"], int(counts["h_count"])),
        "v": np.linspace(
            vehicle["glide_speed_min_mps"], vehicle["glide_speed_max_mps"],
            int(counts["v_count"]),
        ),
        "gamma": np.deg2rad(np.linspace(
            vehicle["gamma_min_deg"], vehicle["gamma_max_deg"],
            int(counts["gamma_count"]),
        )),
        # psi is periodic, so +180 degrees is not duplicated.
        "psi": np.deg2rad(np.linspace(
            vehicle["heading_min_deg"], vehicle["heading_max_deg"],
            int(counts["psi_count"]), endpoint=False,
        )),
    }
    return {name: _readonly(values) for name, values in grids.items()}


def _nested_indices(fine_grid: np.ndarray, coarse_grid: np.ndarray) -> np.ndarray:
    indices = np.asarray([
        int(np.argmin(np.abs(fine_grid - value))) for value in coarse_grid
    ])
    if not np.allclose(fine_grid[indices], coarse_grid, rtol=0.0, atol=1.0e-10):
        raise ValueError("state grid must be an exact nested subset of geometry grid")
    if np.unique(indices).size != indices.size:
        raise ValueError("nested state-grid indices must be unique")
    return indices


def construct_state_validity_masks(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    grids: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct the exact 3D extension of the completed 2D masks."""
    x_indices = _nested_indices(geometry["x_grid"], grids["x"])
    y_indices = _nested_indices(geometry["y_grid"], grids["y"])
    h_indices = _nested_indices(geometry["h_grid"], grids["h"])
    nested = np.ix_(x_indices, y_indices, h_indices)
    terrain = np.asarray(geometry["terrain_mask"][nested], dtype=bool)
    los = np.asarray(geometry["los_mask"][nested], dtype=bool)
    non_visible = np.asarray(
        geometry["non_visible_airspace_mask"][nested], dtype=bool,
    )

    vehicle = configuration["vehicle"]
    velocity_mesh, gamma_mesh = np.meshgrid(
        grids["v"], grids["gamma"], indexing="ij",
    )
    velocity_valid = (
        (velocity_mesh >= vehicle["glide_speed_min_mps"])
        & (velocity_mesh <= vehicle["glide_speed_max_mps"])
    )
    gamma_valid = (
        (gamma_mesh >= np.deg2rad(vehicle["gamma_min_deg"]))
        & (gamma_mesh <= np.deg2rad(vehicle["gamma_max_deg"]))
    )
    lift_coefficient = (
        2.0
        * vehicle["mass_kg"]
        * vehicle["gravity_mps2"]
        * np.cos(gamma_mesh)
        / (
            vehicle["air_density_kgpm3"]
            * velocity_mesh**2
            * vehicle["wing_area_m2"]
        )
    )
    drag_coefficient = (
        vehicle["cd0"]
        + vehicle["linear_drag_coefficient"] * lift_coefficient
        + vehicle["quadratic_drag_coefficient"] * lift_coefficient**2
    )
    dynamic_valid = (
        (lift_coefficient >= vehicle["cl_min"])
        & (lift_coefficient <= vehicle["cl_max"])
        & (drag_coefficient > 0.0)
        & np.isfinite(lift_coefficient)
        & np.isfinite(drag_coefficient)
    )
    control_valid_2d = velocity_valid & gamma_valid & dynamic_valid
    control_valid = np.broadcast_to(
        control_valid_2d[:, :, None],
        (grids["v"].size, grids["gamma"].size, grids["psi"].size),
    )

    # These two lines intentionally match p1b_4D.stage_cost exactly.
    spatial_glide_valid = ~terrain & los
    spatial_powered_valid = ~terrain & non_visible
    full_shape = tuple(grids[name].size for name in AXIS_ORDER)
    feasible = (
        spatial_glide_valid[:, :, :, None, None, None]
        & control_valid[None, None, None, :, :, :]
    )
    powered_feasible = np.broadcast_to(
        spatial_powered_valid[:, :, :, None, None, None], full_shape,
    ).copy()
    return {
        "feasible_mask": feasible,
        "powered_feasible_mask": powered_feasible,
        "terrain_penetration_mask": np.broadcast_to(
            terrain[:, :, :, None, None, None], full_shape,
        ).copy(),
        "airspace_boundary_mask": np.ones(full_shape, dtype=bool),
        "los_validity_mask": np.broadcast_to(
            los[:, :, :, None, None, None], full_shape,
        ).copy(),
        "velocity_limit_mask": np.broadcast_to(
            velocity_valid[None, None, None, :, :, None], full_shape,
        ).copy(),
        "gamma_limit_mask": np.broadcast_to(
            gamma_valid[None, None, None, :, :, None], full_shape,
        ).copy(),
        "glide_dynamic_feasibility_mask": np.broadcast_to(
            dynamic_valid[None, None, None, :, :, None], full_shape,
        ).copy(),
        "control_valid_mask": np.broadcast_to(
            control_valid[None, None, None, :, :, :], full_shape,
        ).copy(),
        "nested_geometry_indices": {
            "x": x_indices, "y": y_indices, "h": h_indices,
        },
        "spatial_glide_valid": spatial_glide_valid,
        "spatial_powered_valid": spatial_powered_valid,
        "lift_coefficient": lift_coefficient,
        "drag_coefficient": drag_coefficient,
    }


def construct_stage_cost_6d(
    configuration: dict[str, Any],
    geometry: dict[str, Any],
    detection_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the authoritative local J6D and powered pre-switch cost."""
    if not geometry.get("validation", {}).get("passed", False):
        raise ValueError("geometry must pass validation")
    if not detection_bundle.get("status", {}).get("success", False):
        raise ValueError("detection bundle must pass validation")
    if detection_bundle["metadata"]["state_axis_order"] != AXIS_ORDER:
        raise ValueError("detection state axis order does not match J6D")

    grids = construct_state_grids(configuration)
    shape = tuple(grids[name].size for name in AXIS_ORDER)
    spatial_shape = tuple(grids[name].size for name in ("x", "y", "h"))
    spatial_size = int(np.prod(spatial_shape))
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        grids["x"], grids["y"], grids["h"], indexing="ij",
    )
    flat_x = mesh_x.reshape(1, spatial_size)
    flat_y = mesh_y.reshape(1, spatial_size)
    flat_h = mesh_h.reshape(1, spatial_size)
    sensor = np.asarray(geometry["sensor_position"], dtype=float)
    sensor_arguments = tuple(
        np.full((1, spatial_size), value) for value in sensor
    )
    time_step = float(configuration["vehicle"]["time_step_s"])
    zero = np.zeros((1, spatial_size))
    stage_time = np.full((1, spatial_size), time_step)
    functions = detection_bundle["functions"]
    glide_function = functions["glide_detection_components"].map(spatial_size)
    powered_function = functions["powered_detection_components"].map(spatial_size)
    objective_function = functions["attacker_objective"].map(spatial_size)

    components = {
        "stage_pod": np.empty(shape),
        "stage_time": np.full(shape, time_step),
        "acoustic_rate": np.empty(shape),
        "powered_stage_pod": np.empty(shape),
        "powered_stage_objective": np.empty(shape),
        "radar_rate": np.empty(shape),
        "radial_velocity": np.empty(shape),
        "radial_velocity_detection_rate": np.empty(shape),
        "aspect_angle": np.empty(shape),
        "rcs": np.empty(shape),
        "glide_detection_rate": np.empty(shape),
        "pod_normalized": np.empty(shape),
        "time_normalized": np.empty(shape),
        "stage_objective": np.empty(shape),
    }

    powered_outputs = _mapped_outputs(
        powered_function, flat_x, flat_y, flat_h,
        np.full((1, spatial_size), configuration["vehicle"]["powered_speed_mps"]),
        *sensor_arguments,
    )
    acoustic_rate = powered_outputs[0].reshape(spatial_shape)
    powered_rate = powered_outputs[1].reshape(spatial_shape)
    powered_hazard = powered_rate.reshape(1, spatial_size) * time_step
    powered_objective = _mapped_outputs(
        objective_function, powered_hazard, zero, stage_time, zero,
    )
    powered_pod = powered_objective[0].reshape(spatial_shape)
    powered_stage_objective = powered_objective[-1].reshape(spatial_shape)

    for velocity_index, velocity in enumerate(grids["v"]):
        velocity_values = np.full((1, spatial_size), velocity)
        for gamma_index, gamma in enumerate(grids["gamma"]):
            gamma_values = np.full((1, spatial_size), gamma)
            for psi_index, psi in enumerate(grids["psi"]):
                psi_values = np.full((1, spatial_size), psi)
                outputs = _mapped_outputs(
                    glide_function, flat_x, flat_y, flat_h,
                    velocity_values, gamma_values, psi_values,
                    *sensor_arguments,
                )
                glide_rate = outputs[10].reshape(spatial_shape)
                glide_hazard = glide_rate.reshape(1, spatial_size) * time_step
                objective = _mapped_outputs(
                    objective_function, zero, glide_hazard, zero, stage_time,
                )
                index = (
                    slice(None), slice(None), slice(None),
                    velocity_index, gamma_index, psi_index,
                )
                components["stage_pod"][index] = objective[0].reshape(spatial_shape)
                components["radar_rate"][index] = outputs[8].reshape(spatial_shape)
                components["radial_velocity"][index] = outputs[6].reshape(spatial_shape)
                components["radial_velocity_detection_rate"][index] = outputs[9].reshape(spatial_shape)
                components["aspect_angle"][index] = outputs[2].reshape(spatial_shape)
                components["rcs"][index] = outputs[4].reshape(spatial_shape)
                components["glide_detection_rate"][index] = glide_rate
                components["pod_normalized"][index] = objective[2].reshape(spatial_shape)
                components["time_normalized"][index] = objective[3].reshape(spatial_shape)
                components["stage_objective"][index] = objective[4].reshape(spatial_shape)
                components["acoustic_rate"][index] = acoustic_rate
                components["powered_stage_pod"][index] = powered_pod
                components["powered_stage_objective"][index] = powered_stage_objective

    masks = construct_state_validity_masks(configuration, geometry, grids)
    j6d = np.where(masks["feasible_mask"], components["stage_objective"], np.inf)
    powered_stage_cost = np.where(
        masks["powered_feasible_mask"],
        components["powered_stage_objective"], np.inf,
    )
    validation = validate_stage_cost_6d(
        j6d, powered_stage_cost, components, masks, grids, configuration,
    )
    readonly_masks = {
        name: (_readonly(value) if isinstance(value, np.ndarray) else value)
        for name, value in masks.items()
    }
    return {
        "j6d": _readonly(j6d),
        "powered_stage_cost_6d": _readonly(powered_stage_cost),
        "component_maps": {
            name: _readonly(value) for name, value in components.items()
        },
        "feasible_mask": readonly_masks["feasible_mask"],
        "validity_masks": readonly_masks,
        "grids": grids,
        "grid_metadata": {
            "shape": shape,
            "axis_order": AXIS_ORDER,
            "state_count": int(np.prod(shape)),
            "spatial_shape": spatial_shape,
            "geometry_grid_is_separate_and_finer": True,
            "units": {
                "x": "m", "y": "m", "h": "m", "v": "m/s",
                "gamma": "rad", "psi": "rad", "time": "s",
            },
        },
        "metadata": {
            "schema_name": "StageCost6DResult",
            "source_process": "p1b_4D.stage_cost direct 3D extension",
            "j6d_mode": "glide",
            "invalid_cost": "positive_infinity",
            "powered_cost_role": "pre_switch_local_component",
            "glide_feasibility_rule": "airspace_and_not_terrain_and_los",
            "powered_feasibility_rule": "airspace_and_not_terrain_and_occluded",
            "is_value_function": False,
            "is_cost_to_go": False,
        },
        "validation": validation,
        "status": {
            "success": validation["passed"],
            "message": validation["summary"],
        },
    }


def validate_stage_cost_6d(
    j6d: np.ndarray,
    powered_stage_cost: np.ndarray,
    components: dict[str, np.ndarray],
    masks: dict[str, Any],
    grids: dict[str, np.ndarray],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    expected_shape = tuple(grids[name].size for name in AXIS_ORDER)
    feasible = masks["feasible_mask"]
    powered_feasible = masks["powered_feasible_mask"]
    attacker = configuration["cost"]["attacker"]
    tolerance = configuration["validation"]["objective_tolerance"]
    reconstructed = (
        attacker["w_pod"] * components["pod_normalized"]
        + attacker["w_time"] * components["time_normalized"]
    )
    objective_residual = (
        float(np.max(np.abs(
            reconstructed[feasible] - components["stage_objective"][feasible]
        ))) if np.any(feasible) else np.inf
    )
    expected_stage_pod = -np.expm1(
        -components["pod_normalized"][feasible] * attacker["hazard_reference"]
    )
    spatial_partition = (
        masks["spatial_glide_valid"] | masks["spatial_powered_valid"]
    )
    spatial_air = ~np.any(
        masks["terrain_penetration_mask"], axis=(3, 4, 5),
    )
    checks = {
        "grid_dimensions": bool(j6d.shape == expected_shape),
        "axis_order": tuple(configuration["state_grid"]["axis_order_6d"]) == AXIS_ORDER,
        "component_dimensions": bool(all(
            value.shape == expected_shape for value in components.values()
        )),
        "feasible_states_exist": bool(np.any(feasible)),
        "powered_feasible_states_exist": bool(np.any(powered_feasible)),
        "finite_feasible_cost": bool(np.all(np.isfinite(j6d[feasible]))),
        "infinite_invalid_cost": bool(np.all(np.isposinf(j6d[~feasible]))),
        "finite_powered_feasible_cost": bool(np.all(np.isfinite(
            powered_stage_cost[powered_feasible]
        ))),
        "infinite_invalid_powered_cost": bool(np.all(np.isposinf(
            powered_stage_cost[~powered_feasible]
        ))),
        "stage_pod_consistency": bool(np.allclose(
            components["stage_pod"][feasible], expected_stage_pod,
            rtol=0.0, atol=tolerance,
        )),
        "objective_consistency": bool(objective_residual <= tolerance),
        "spatial_phase_partition": bool(np.array_equal(
            spatial_partition, spatial_air,
        )),
        "psi_periodic_endpoint_not_duplicated": bool(
            grids["psi"].size > 1
            and not np.isclose(grids["psi"][0], grids["psi"][-1])
        ),
        "nested_geometry_grid": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    feasible_values = j6d[feasible]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "minimum_cost": float(np.min(feasible_values)),
            "maximum_cost": float(np.max(feasible_values)),
            "feasible_state_count": int(np.count_nonzero(feasible)),
            "powered_feasible_state_count": int(np.count_nonzero(powered_feasible)),
            "total_state_count": int(j6d.size),
            "objective_maximum_residual": objective_residual,
        },
        "summary": (
            "6D local stage-cost validation passed"
            if not failed
            else f"6D local stage-cost failed checks: {failed}"
        ),
    }


def _mapped_outputs(function: ca.Function, *arguments: np.ndarray) -> list[np.ndarray]:
    values = function(*arguments)
    outputs = values if isinstance(values, tuple) else (values,)
    return [np.asarray(value, dtype=float) for value in outputs]


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
