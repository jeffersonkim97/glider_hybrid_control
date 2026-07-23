"""Visualization-only 2D projection of the authoritative 4D stage cost."""

from __future__ import annotations

from typing import Any

import numpy as np


def construct_projected_cost_map(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_4d_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Project J4D onto (z, h) by feasible local control minimization.

    Inputs
    ------
    configuration_bundle:
        Successful Phase 1 ConfigurationBundle.
    geometry_bundle:
        Successful Phase 2 GeometryBundle.
    detection_bundle:
        Successful Phase 3 DetectionBundle.
    stage_cost_4d_bundle:
        Successful authoritative 4D local stage-cost result.

    Outputs
    -------
    dict
        Universal Projection2DResult containing ProjectedCost, diagnostic
        local velocity/gamma values and indices, mask, metadata, and validation.

    Assumptions
    -----------
    J4D axis order is (z, h, v, gamma), and invalid states have positive
    infinite cost.

    Notes
    -----
    This result is visualization-only. It is not a Bellman value function,
    cost-to-go map, trajectory policy, warm start, or trajectory extractor.
    """
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    _require_successful_bundle(detection_bundle, "detection_bundle")
    _require_successful_bundle(stage_cost_4d_bundle, "stage_cost_4d_bundle")

    stage_result = stage_cost_4d_bundle["primary_result"]
    j4d = np.asarray(stage_result["j4d"])
    grids = stage_result["grids"]
    expected_shape = tuple(
        grids[name].size for name in ("z", "h", "v", "gamma")
    )
    if j4d.shape != expected_shape:
        raise ValueError(
            f"J4D shape {j4d.shape} does not match grid shape {expected_shape}"
        )
    if stage_result["grid_metadata"]["axis_order"] != ("z", "h", "v", "gamma"):
        raise ValueError("J4D axis order must be (z, h, v, gamma)")

    projection_mask = np.any(np.isfinite(j4d), axis=(2, 3))
    projected_cost = np.min(j4d, axis=(2, 3))
    flattened_controls = j4d.reshape(j4d.shape[0], j4d.shape[1], -1)
    optimal_flat_index = np.argmin(flattened_controls, axis=2)
    optimal_velocity_index = optimal_flat_index // grids["gamma"].size
    optimal_gamma_index = optimal_flat_index % grids["gamma"].size

    optimal_velocity_index = optimal_velocity_index.astype(np.int64)
    optimal_gamma_index = optimal_gamma_index.astype(np.int64)
    optimal_velocity_index[~projection_mask] = -1
    optimal_gamma_index[~projection_mask] = -1
    optimal_velocity = np.full(projected_cost.shape, np.nan)
    optimal_gamma = np.full(projected_cost.shape, np.nan)
    optimal_velocity[projection_mask] = grids["v"][
        optimal_velocity_index[projection_mask]
    ]
    optimal_gamma[projection_mask] = grids["gamma"][
        optimal_gamma_index[projection_mask]
    ]

    validation = validate_projected_cost_map(
        projected_cost,
        optimal_velocity,
        optimal_gamma,
        optimal_velocity_index,
        optimal_gamma_index,
        projection_mask,
        j4d,
        grids,
        configuration_bundle["primary_result"]["validation_config"],
    )
    finite_projected_cost = projected_cost[projection_mask]
    projection_metadata = {
        "projection_rule": "minimum_feasible_stage_cost_over_v_gamma",
        "source_axis_order": ("z", "h", "v", "gamma"),
        "projected_axes": ("z", "h"),
        "control_flattening_order": ("v", "gamma"),
        "minimum_cost": float(np.min(finite_projected_cost)),
        "maximum_cost": float(np.max(finite_projected_cost)),
        "valid_spatial_cell_count": int(np.count_nonzero(projection_mask)),
        "invalid_spatial_cell_count": int(np.count_nonzero(~projection_mask)),
        "total_spatial_cell_count": int(projection_mask.size),
        "projection_status": "complete" if validation["passed"] else "invalid",
        "visualization_only": True,
        "bellman_policy_input": False,
        "is_value_function": False,
        "is_cost_to_go": False,
    }
    return {
        "primary_result": {
            "projected_cost": _readonly(projected_cost),
            "optimal_velocity": _readonly(optimal_velocity),
            "optimal_gamma": _readonly(optimal_gamma),
            "optimal_velocity_index": _readonly(optimal_velocity_index),
            "optimal_gamma_index": _readonly(optimal_gamma_index),
            "projection_mask": _readonly(projection_mask),
            "grids": {
                "z": grids["z"],
                "h": grids["h"],
            },
            "projection_metadata": projection_metadata,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "Projection2DResult",
            "schema_version": "1.0.0",
            "producer_phase": 5,
            "producer_module": "p1b_4D.projection",
            "source_stage_cost_schema_version": stage_cost_4d_bundle["metadata"][
                "schema_version"
            ],
            "source_attacker_objective_id": stage_cost_4d_bundle["metadata"][
                "attacker_objective_id"
            ],
            "geometry_schema_version": geometry_bundle["metadata"][
                "schema_version"
            ],
            "detection_schema_version": detection_bundle["metadata"][
                "schema_version"
            ],
            "shape": tuple(projected_cost.shape),
            "axis_order": ("z", "h"),
            "units": {"z": "m", "h": "m", "velocity": "m/s", "gamma": "rad"},
            "visualization_only": True,
            "prohibited_uses": (
                "bellman_policy",
                "cost_to_go",
                "trajectory_extraction",
                "nlp_initialization",
            ),
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "PROJECTION_2D_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def validate_projected_cost_map(
    projected_cost: np.ndarray,
    optimal_velocity: np.ndarray,
    optimal_gamma: np.ndarray,
    optimal_velocity_index: np.ndarray,
    optimal_gamma_index: np.ndarray,
    projection_mask: np.ndarray,
    j4d: np.ndarray,
    grids: dict[str, np.ndarray],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate projection dimensions, minima, invalid cells, and completeness."""
    expected_shape = (grids["z"].size, grids["h"].size)
    tolerance = validation_config["objective_tolerance"]
    spatial_indices = np.nonzero(projection_mask)
    selected_cost = j4d[
        spatial_indices[0],
        spatial_indices[1],
        optimal_velocity_index[projection_mask],
        optimal_gamma_index[projection_mask],
    ]
    direct_minimum = np.min(j4d, axis=(2, 3))
    minimum_residual = (
        float(
            np.max(
                np.abs(
                    selected_cost
                    - projected_cost[projection_mask]
                )
            )
        )
        if selected_cost.size
        else np.inf
    )
    direct_residual = (
        float(
            np.max(
                np.abs(
                    direct_minimum[projection_mask]
                    - projected_cost[projection_mask]
                )
            )
        )
        if selected_cost.size
        else np.inf
    )
    checks = {
        "projected_cost_dimensions": projected_cost.shape == expected_shape,
        "control_dimensions": (
            optimal_velocity.shape == expected_shape
            and optimal_gamma.shape == expected_shape
        ),
        "index_dimensions": (
            optimal_velocity_index.shape == expected_shape
            and optimal_gamma_index.shape == expected_shape
        ),
        "projection_mask_dimensions": projection_mask.shape == expected_shape,
        "finite_valid_values": bool(
            np.all(np.isfinite(projected_cost[projection_mask]))
        ),
        "infinite_invalid_values": bool(
            np.all(np.isposinf(projected_cost[~projection_mask]))
        ),
        "invalid_control_values": bool(
            np.all(np.isnan(optimal_velocity[~projection_mask]))
            and np.all(np.isnan(optimal_gamma[~projection_mask]))
            and np.all(optimal_velocity_index[~projection_mask] == -1)
            and np.all(optimal_gamma_index[~projection_mask] == -1)
        ),
        "valid_control_indices": bool(
            np.all(
                (optimal_velocity_index[projection_mask] >= 0)
                & (optimal_velocity_index[projection_mask] < grids["v"].size)
            )
            and np.all(
                (optimal_gamma_index[projection_mask] >= 0)
                & (optimal_gamma_index[projection_mask] < grids["gamma"].size)
            )
        ),
        "minimum_cost_consistency": minimum_residual <= tolerance,
        "projection_consistency": direct_residual <= tolerance,
        "projection_completeness": bool(
            np.array_equal(
                projection_mask,
                np.any(np.isfinite(j4d), axis=(2, 3)),
            )
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "minimum_cost": float(np.min(projected_cost[projection_mask])),
            "maximum_cost": float(np.max(projected_cost[projection_mask])),
            "valid_spatial_cell_count": int(np.count_nonzero(projection_mask)),
            "invalid_spatial_cell_count": int(np.count_nonzero(~projection_mask)),
            "minimum_selection_residual": minimum_residual,
            "direct_projection_residual": direct_residual,
        },
        "tolerances": {"objective": tolerance},
        "warnings": [],
        "failed_checks": failed_checks,
        "summary": (
            "Phase 5 2D projected-cost validation passed"
            if not failed_checks
            else f"Phase 5 2D projection failed checks: {failed_checks}"
        ),
    }


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict):
        raise TypeError(f"{name} must be a dictionary")
    if not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must have successful status")


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result
