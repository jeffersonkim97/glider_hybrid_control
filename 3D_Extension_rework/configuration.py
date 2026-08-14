"""Stage 1 configuration for the clean single-hill 3D rework."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIGURATION: dict[str, Any] = {
    "environment": {
        "x_bounds_m": (0.0, 3000.0),
        "y_bounds_m": (0.0, 1000.0),
        "h_bounds_m": (0.0, 400.0),
        "terrain": {
            "scenario_id": "asymmetric_single_hill",
            "hills": (
                {
                    "center_xy_m": (1500.0, 700.0),
                    "height_m": 200.0,
                    "width_x_m": 400.0,
                    "width_y_m": 400.0,
                },
            ),
        },
        "sensor_xy_m": (2500.0, 500.0),
        "goal_xy_m": (3000.0, 500.0),
        "launch_xy_m": (0.0, 500.0),
        "sensor_mount_height_m": 0.0,
    },
    "grid": {
        "x_count": 151,
        "y_count": 81,
        "h_count": 81,
        "los_ray_sample_count": 161,
        "tangent_azimuth_count": 721,
        "tangent_range_sample_count": 513,
    },
    # Nested subset of the fine geometry grid used for the dense 6D stage
    # cost and later Bellman solve.  Keeping these grids separate mirrors the
    # completed 2D workflow while keeping the 6D array tractable.
    "state_grid": {
        "x_count": 31,
        "y_count": 11,
        "h_count": 21,
        "v_count": 3,
        "gamma_count": 6,
        # Preserve the previous 3D extension's 10-degree periodic heading
        # resolution so later paths are not forced into 30-degree courses.
        "psi_count": 36,
        "axis_order_6d": ("x", "y", "h", "v", "gamma", "psi"),
    },
    # Physical detection and Attacker-objective parameters retain the completed
    # p1b_4D values.  Defender weights are the Stage-9 sensitivity selection.
    "vehicle": {
        "powered_speed_mps": 21.0,
        "glide_speed_min_mps": 10.0,
        "glide_speed_max_mps": 22.6,
        "gamma_min_deg": -90.0,
        "gamma_max_deg": -1.0,
        "heading_min_deg": -180.0,
        "heading_max_deg": 180.0,
        "time_step_s": 1.0,
        "mass_kg": 9.34 / 9.81,
        "gravity_mps2": 9.81,
        "air_density_kgpm3": 1.225,
        "wing_area_m2": 0.321,
        "cd0": 0.0213,
        "linear_drag_coefficient": -0.056,
        "quadratic_drag_coefficient": 0.22,
        "cl_min": 0.05,
        "cl_max": 0.50,
        "max_turn_rate_deg_s": 5.0,
    },
    "detection": {
        "range_floor_m": 10.0,
        "acoustic_coefficient": 2.4e-3,
        "acoustic_speed_exponent": 4,
        "radar_coefficient": 1.3e7,
        "doppler_coefficient": 3.325e4,
        "rcs_min": 0.1,
        "rcs_max": 1.0,
        "radar_rate_scale": 1.0,
        "radial_velocity_rate_scale": 1.0,
        "acoustic_rate_scale": 1.0,
    },
    "cost": {
        "attacker": {
            "w_pod": 0.5,
            "w_time": 0.5,
            "hazard_reference": 1.0,
            "time_reference_s": 5000.0 / 22.6,
            "objective_id": "attacker_hazard_time_v2",
        },
        "defender": {
            "w_pod": 0.9,
            "w_coverage": 0.1,
            "hazard_reference": 1.0,
            "objective_id": "defender_pod_los_coverage_v1",
        },
    },
    # All leader-search settings live here so the admissible sensor region
    # and search density can be changed without touching solver code.
    "defender_search": {
        "x_bounds_m": (1750.0, 2500.0),
        "y_bounds_m": (0.0, 1000.0),
        "sensor_height_rule": "terrain_following_plus_mount_height",
        "coarse_x_count": 3,
        "coarse_y_count": 5,
        "refinement_levels": 1,
        "refinement_factor": 2.0,
        "local_stencil_count": 3,
        "cache_round_decimals": 6,
        # The hill is intentionally shifted to y=700, so the former y=500
        # reflection shortcut is invalid and every y candidate is solved.
        "use_y_reflection_symmetry": False,
        "verify_y_reflection_symmetry": False,
        "symmetry_center_y_m": 500.0,
        "symmetry_tolerance": 1.0e-9,
    },
    # Post-process cached follower responses under complementary Defender
    # weights.  All sensitivity controls live here so the sweep is repeatable.
    "weight_sensitivity": {
        "reference_sensor_xy_m": (2500.0, 500.0),
        "coverage_weights": (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50),
        "coverage_weight_plot_bounds": (0.0, 0.50),
        "dense_plot_count": 501,
        "weight_sum": 1.0,
    },
    "bellman": {
        "transition_model": "exact_physical_successor_grid",
        "maximum_forward_cells": 3,
        "maximum_lateral_cells": 3,
        "maximum_descent_cells": 8,
        "edge_quadrature_count": 9,
        "goal_radius_m": 15.0,
        "terrain_clearance_m": 0.0,
        # The continuous switch is connected to the solved Bellman lattice
        # by the same physical first-edge construction used in p1b_4D.
        "switching_manifold_stride": 1,
        "powered_segment_quadrature_count": 65,
        "virtual_edge_quadrature_count": 17,
        "maximum_path_steps": 1000,
    },
    "validation": {
        "terrain_tolerance_m": 1.0e-8,
        "tangent_residual_tolerance": 2.0e-6,
        "detection_probability_tolerance": 1.0e-12,
        "objective_tolerance": 1.0e-8,
    },
}


def validate_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """Validate one complete configuration after all scenario overrides."""
    environment = configuration["environment"]
    grid = configuration["grid"]
    for name in ("x_bounds_m", "y_bounds_m", "h_bounds_m"):
        lower, upper = environment[name]
        if not lower < upper:
            raise ValueError(f"{name} must be strictly ordered")
    for name in (
        "x_count", "y_count", "h_count", "los_ray_sample_count",
        "tangent_range_sample_count",
    ):
        if int(grid[name]) < 3:
            raise ValueError(f"{name} must contain at least three samples")
    if int(grid["tangent_azimuth_count"]) < 16:
        raise ValueError("tangent_azimuth_count is too small")
    state_grid = configuration["state_grid"]
    for name in ("x_count", "y_count", "h_count", "v_count", "gamma_count"):
        if int(state_grid[name]) < 2:
            raise ValueError(f"state_grid.{name} must contain at least two samples")
    if int(state_grid["psi_count"]) < 4:
        raise ValueError("state_grid.psi_count must contain at least four samples")
    if tuple(state_grid["axis_order_6d"]) != (
        "x", "y", "h", "v", "gamma", "psi",
    ):
        raise ValueError("state_grid.axis_order_6d is fixed by the 6D contract")
    terrain = environment["terrain"]
    hills = terrain["hills"]
    if not hills:
        raise ValueError("terrain.hills must be nonempty")
    for index, hill in enumerate(hills):
        if hill["height_m"] <= 0.0:
            raise ValueError(f"terrain hill {index} height must be positive")
        if hill["width_x_m"] <= 0.0 or hill["width_y_m"] <= 0.0:
            raise ValueError(f"terrain hill {index} widths must be positive")
    vehicle = configuration["vehicle"]
    if not 0.0 < vehicle["glide_speed_min_mps"] <= vehicle["glide_speed_max_mps"]:
        raise ValueError("glide speed bounds must be positive and ordered")
    detection = configuration["detection"]
    if detection["range_floor_m"] <= 0.0:
        raise ValueError("range_floor_m must be positive")
    if not 0.0 <= detection["rcs_min"] <= detection["rcs_max"]:
        raise ValueError("RCS bounds must be nonnegative and ordered")
    defender_cost = configuration["cost"]["defender"]
    if defender_cost["w_pod"] < 0.0 or defender_cost["w_coverage"] < 0.0:
        raise ValueError("Defender weights must be nonnegative")
    if abs(defender_cost["w_pod"] + defender_cost["w_coverage"] - 1.0) > 1.0e-12:
        raise ValueError("Defender weights must sum to one")
    bellman = configuration["bellman"]
    for name in (
        "maximum_forward_cells", "maximum_lateral_cells",
        "maximum_descent_cells",
    ):
        if int(bellman[name]) < 1:
            raise ValueError(f"bellman.{name} must be positive")
    if int(bellman["edge_quadrature_count"]) < 3:
        raise ValueError("bellman.edge_quadrature_count must be at least three")
    if bellman["goal_radius_m"] <= 0.0:
        raise ValueError("bellman.goal_radius_m must be positive")
    for name in (
        "switching_manifold_stride", "powered_segment_quadrature_count",
        "virtual_edge_quadrature_count",
        "maximum_path_steps",
    ):
        if int(bellman[name]) < 1:
            raise ValueError(f"bellman.{name} must be positive")
    defender = configuration["defender_search"]
    for name in ("x_bounds_m", "y_bounds_m"):
        lower, upper = defender[name]
        if not lower < upper:
            raise ValueError(f"defender_search.{name} must be strictly ordered")
    x_bounds, y_bounds = environment["x_bounds_m"], environment["y_bounds_m"]
    if not (
        x_bounds[0] <= defender["x_bounds_m"][0] < defender["x_bounds_m"][1] <= x_bounds[1]
        and y_bounds[0] <= defender["y_bounds_m"][0] < defender["y_bounds_m"][1] <= y_bounds[1]
    ):
        raise ValueError("defender search bounds must lie inside the map")
    for name in ("coarse_x_count", "coarse_y_count", "local_stencil_count"):
        if int(defender[name]) < 3:
            raise ValueError(f"defender_search.{name} must be at least three")
    if int(defender["refinement_levels"]) < 0:
        raise ValueError("defender_search.refinement_levels must be nonnegative")
    if float(defender["refinement_factor"]) <= 1.0:
        raise ValueError("defender_search.refinement_factor must exceed one")
    sensitivity = configuration["weight_sensitivity"]
    lower_weight, upper_weight = sensitivity["coverage_weight_plot_bounds"]
    if not 0.0 <= lower_weight < upper_weight <= sensitivity["weight_sum"]:
        raise ValueError("weight-sensitivity plot bounds must lie inside the weight sum")
    for coverage_weight in sensitivity["coverage_weights"]:
        if not lower_weight <= coverage_weight <= upper_weight:
            raise ValueError("reported coverage weights must lie inside the plot bounds")
    if int(sensitivity["dense_plot_count"]) < 3:
        raise ValueError("weight_sensitivity.dense_plot_count must be at least three")
    return configuration


def build_configuration() -> dict[str, Any]:
    """Return a validated independent copy of the default configuration."""
    return validate_configuration(deepcopy(CONFIGURATION))
