"""Central Phase 1 configuration for the 3D Stackelberg security project.

Mirrors p1b_4D.configuration's structure and validation philosophy exactly,
extended from (z, h) to (x, y, h): h keeps its 2D role (the Bellman DP's
monotonic sweep axis, since gamma stays negative-only regardless of
heading), x and y are the two free position dimensions (replacing z's
single free-along-track role), and heading is a third, unconstrained
action dimension alongside v and gamma. See p1b_3DExtension.ipynb's first
cell for the full design rationale.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

from .phase_logging import close_phase_logger, create_phase_logger, log_phase
from .project_paths import ProjectPaths, create_project_paths

GLOBAL_RANDOM_SEED = 20260722
CONFIG_SCHEMA_VERSION = "1.0.0"

environment_config: dict[str, Any] = {
    "x_start": 0.0,
    "y_start": 0.0,
    "h_start": 0.0,
    "x_goal": 2500.0,
    "y_goal": 0.0,
    # h_goal is not configured here: it is the terrain elevation at
    # (x_goal, y_goal), computed once terrain exists (mirrors p1b_4D's
    # h_goal = terrain_height(z_goal) fix -- h_goal is always derived, never
    # a free config value, exactly like h_sensor).
    "terrain": {
        "x_min": 0.0,
        "x_max": 2750.0,
        "y_min": -1500.0,
        "y_max": 1500.0,
        # One radially symmetric Gaussian hill for the v1 toy (single
        # obstacle first, mirroring how p1b_4D validated one hill before
        # generalizing to N). Terrain is the sum of every hill's Gaussian
        # -- a single-hill tuple today, extensible to multiple without
        # changing the surface representation, exactly like p1b_4D's
        # "terrain is one combined surface" invariant.
        "hills": (
            {"x_ridge": 1500.0, "y_ridge": 0.0, "h_ridge": 100.0, "width": 500.0},
        ),
    },
    "grid": {
        "x_min": 0.0,
        "x_max": 2750.0,
        "x_count": 41,
        "y_min": -1500.0,
        "y_max": 1500.0,
        "y_count": 61,
        "h_min": 0.0,
        "h_max": 200.0,
        "h_count": 41,
        "v_count": 3,
        "gamma_count": 6,
        # 10 deg spacing over the full circle. heading is periodic
        # (-180 == +180), so the grid built from this count must use an
        # endpoint-excluding convention (e.g. np.linspace(-180, 180, 36,
        # endpoint=False) or equivalently arange(-180, 180, 10)) to avoid
        # duplicating one heading at both ends -- unlike x/y/h/v/gamma,
        # which are non-periodic and use inclusive linspace.
        "heading_count": 36,
        "axis_order_6d": ("x", "y", "h", "v", "gamma", "heading"),
    },
    "airspace": {
        "x_min": 0.0,
        "x_max": 2750.0,
        "y_min": -1500.0,
        "y_max": 1500.0,
        "h_min": 0.0,
        "h_max": 200.0,
    },
    "simulation": {
        "x_min": 0.0,
        "x_max": 2750.0,
        "y_min": -1500.0,
        "y_max": 1500.0,
        "h_min": 0.0,
        "h_max": 200.0,
        "max_path_steps": 1000,
    },
    "units": {"distance": "m", "time": "s", "speed": "m/s", "angle": "rad"},
}

vehicle_config: dict[str, Any] = {
    "active_model": "byu_unicorn_flight_test",
    "powered_speed": 21.0,
    "glide_speed_min": 10.0,
    "glide_speed_max": 22.6,
    "glide_speed_count": 3,
    "gamma_min_deg": -90.0,
    "gamma_max_deg": -1.0,
    "gamma_count": 6,
    # Heading is a free, unconstrained third action dimension (see the
    # notebook's design-rationale cell): no turn-rate limit, so helical
    # paths are legal and heading never needs to become part of the state.
    "heading_min_deg": -180.0,
    "heading_max_deg": 180.0,
    "heading_count": 36,
    "switching_constraints": {
        "terrain_clearance": 1.0,
        "tangent_tolerance": 3.0,
    },
    "dynamic_limits": {
        "cl_min": 0.05,
        "cl_max": 0.50,
        "gamma_feasibility_tolerance_deg": 2.0,
    },
    "time_step": 1.0,
    "segment_length": None,
    "segment_check_count": 9,
    "mass": 9.34 / 9.81,
    "gravity": 9.81,
    "air_density": 1.225,
    "wing_area": 0.321,
    "drag_polar_type": "quadratic_cl",
    "cd0": 0.0213,
    "linear_drag_coefficient": -0.056,
    "quadratic_drag_coefficient": 0.22,
    "gamma_penalty_scale_deg": 10.0,
    "source": (
        "Ostler et al. 2009 Unicorn flight-test polar; powered speed retained "
        "from the existing p1b_4D configuration"
    ),
}

sensor_config: dict[str, Any] = {
    "default_x_sensor": 2000.0,
    "default_y_sensor": 0.0,
    "mount_height": 0.0,
    "height_rule": "terrain(x_sensor, y_sensor) + mount_height",
    "los": {
        "terrain_numerical_tolerance": 1.0e-6,
        "edge_gamma_tolerance_deg": 7.0,
    },
    # Detection-rate formulas are unchanged from p1b_4D (already expressed
    # as Euclidean-range/RCS/radial-velocity relationships, so they extend
    # to 3D range with no change in form -- see detection.py).
    "detection": {
        "range_floor": 10.0,
        # radar/doppler/acoustic coefficients x10 vs. p1b_4D's defaults, per
        # the notebook's sensor-sensitivity comparison.
        "acoustic_coefficient": 2.4e-2,
        "acoustic_speed_exponent": 4,
        "radar_coefficient": 1.3e8,
        "doppler_coefficient": 3.325e5,
        "rcs_min": 0.1,
        "rcs_max": 1.0,
        "radar_rate_scale": 1.0,
        "radial_velocity_rate_scale": 1.0,
        "acoustic_rate_scale": 1.0,
    },
}

cost_config: dict[str, Any] = {
    "attacker": {
        "w_pod": 0.5,
        "w_time": 0.5,
        "normalization": {
            "pod": {
                "method": "cumulative_hazard_reference",
                "hazard_reference": 1.0,
                "mission_pod_reporting_only": True,
            },
            "time": {
                "method": "reference_time",
                "reference_seconds": 2500.0 / 22.6,
            },
        },
        "objective_id": "attacker_hazard_time_v2",
    },
    "defender": {
        "w_pod": 0.8,
        "w_cover": 0.2,
        "normalization": {
            "pod": {
                "method": "hazard_reference",
                "hazard_reference": 1.0,
                "diagnostic_only": False,
            },
            "coverage": {"method": "los_fraction"},
        },
        "objective_id": "defender_pod_los_coverage_v1",
    },
    "local_stage_cost": {
        "detection_weight": 1.0,
        "time_weight": 0.01,
        "steady_glide_deviation_weight": 0.02,
    },
}

bellman_config: dict[str, Any] = {
    "maximum_iterations": 4,
    "convergence_tolerance": 1.0e-8,
    "top_k": 3,
    "duplicate_threshold": {
        "switching_distance": 50.0,
        "trajectory_rms_distance": 20.0,
        "mission_cost_difference": 0.05,
        "path_length_relative_difference": 0.08,
        "resample_point_count": 64,
    },
    "warm_start": {
        "enabled": True,
        "preserve_switching_point": True,
        "preserve_trajectory_profiles": True,
    },
    "search_options": {
        "mode": "multi_start_coarse_bellman",
        "max_forward_cells": 3,
        "max_descent_cells": 8,
        "segment_check_count": 9,
        # Tie-break orderings for the (v, gamma, heading) action loop --
        # extends p1b_4D's (v, gamma) orderings with a heading axis.
        "exploration_orderings": (
            "low_gamma_first",
            "high_gamma_first",
            "low_speed_first",
            "high_speed_first",
        ),
        "initialization": "terminal_goal_region",
    },
    "attacker_objective_id": "attacker_hazard_time_v2",
    "random_seed": GLOBAL_RANDOM_SEED,
}

defender_config: dict[str, Any] = {
    # 2D continuous search bounds (z_sensor) become a 2D (x_sensor,
    # y_sensor) bounded region -- scipy.optimize.direct already supports
    # N-D bounds, so the certified-global search generalizes directly.
    "continuous_search_bounds": {
        "x_sensor_min": 1000.0,
        "x_sensor_max": 2600.0,
        "y_sensor_min": -600.0,
        "y_sensor_max": 600.0,
    },
    "optimizer": "scipy_direct_global",
    "direct_maxfun": 150,
    "direct_maxiter": 300,
    "direct_len_tol": 1.0e-4,
}

plot_config: dict[str, Any] = {
    "default_figure_size": (10.0, 6.0),
    "dpi": 300,
    "export_formats": ("png", "pdf", "svg"),
    "font_family": "DejaVu Sans",
    "font_size": 11,
    "line_width": 1.8,
    "colormaps": {
        "projected_cost": "viridis",
        "projected_cost_to_go": "viridis",
    },
    "line_styles": {
        "terrain": "-",
        "los_tangent": "--",
        "bellman": "--",
        "stackelberg": "-",
    },
    "marker_styles": {
        "sensor": "^",
        "goal": "*",
        "switching_point": "o",
        "defender": "^",
    },
    "colors": {
        "terrain_fill": "white",
        "terrain_outline": "black",
        "los": "green",
        "occlusion": "red",
        "bellman": "tab:blue",
        "stackelberg": "tab:purple",
        "sensor": "black",
        "goal": "gold",
    },
    "terrain_zorder": 1000,
    "heatmap_alpha": 1.0,
}

io_config: dict[str, Any] = {
    "results_directory": "results",
    "json_directory": "results/json",
    "npz_directory": "results/npz",
    "figure_directory": "results/figures",
    "log_directory": "results/logs",
    "json_extension": ".json",
    "npz_extension": ".npz",
    "figure_extensions": (".png", ".pdf"),
}

validation_config: dict[str, Any] = {
    "terrain_tolerance": 1.0e-6,
    "goal_radius": 15.0,
    "los_tolerance": 3.0,
    "dynamic_tolerance": 1.0e-6,
    "solver_tolerance": 1.0e-6,
    "objective_tolerance": 1.0e-8,
    "detection_probability_tolerance": 1.0e-12,
    # Interior samples per ray-marched viewshed line-of-sight check
    # (geometry.compute_los_geometry); not needed by p1b_4D's 1D sweep.
    "los_ray_sample_count": 24,
}

_REQUIRED_KEYS: dict[str, set[str]] = {
    "environment_config": {
        "x_start", "y_start", "h_start", "x_goal", "y_goal", "terrain",
        "grid", "airspace", "simulation", "units",
    },
    "vehicle_config": {
        "powered_speed", "glide_speed_min", "glide_speed_max", "gamma_min_deg",
        "gamma_max_deg", "heading_min_deg", "heading_max_deg", "time_step",
        "mass", "wing_area", "dynamic_limits",
    },
    "sensor_config": {"default_x_sensor", "default_y_sensor", "mount_height", "los", "detection"},
    "cost_config": {"attacker", "defender", "local_stage_cost"},
    "bellman_config": {
        "maximum_iterations", "top_k",
        "duplicate_threshold", "warm_start", "search_options",
    },
    "defender_config": {"continuous_search_bounds", "optimizer"},
    "plot_config": {
        "default_figure_size", "colormaps", "line_styles", "marker_styles",
        "colors", "terrain_zorder", "heatmap_alpha",
    },
    "io_config": {
        "results_directory", "json_directory", "npz_directory",
        "figure_directory", "log_directory",
    },
    "validation_config": {
        "terrain_tolerance", "goal_radius", "los_tolerance",
        "dynamic_tolerance", "solver_tolerance",
    },
}


def _configuration_dicts() -> dict[str, dict[str, Any]]:
    """Return independent copies of all public configuration dictionaries."""
    return {
        "environment_config": deepcopy(environment_config),
        "vehicle_config": deepcopy(vehicle_config),
        "sensor_config": deepcopy(sensor_config),
        "cost_config": deepcopy(cost_config),
        "bellman_config": deepcopy(bellman_config),
        "defender_config": deepcopy(defender_config),
        "plot_config": deepcopy(plot_config),
        "io_config": deepcopy(io_config),
        "validation_config": deepcopy(validation_config),
    }


def validate_configuration(
    configs: dict[str, dict[str, Any]], project_paths: ProjectPaths
) -> dict[str, Any]:
    """Validate dictionaries, cross-config invariants, and output directories.

    Mirrors p1b_4D.configuration.validate_configuration's checks, adapted
    from (z, h) to (x, y, h) and from a 1D to a 2D defender search region.
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for name, required_keys in _REQUIRED_KEYS.items():
        exists = name in configs and isinstance(configs[name], dict)
        checks[f"{name}.exists"] = exists
        if not exists:
            errors.append(f"Missing configuration dictionary: {name}")
            continue
        missing = required_keys - set(configs[name])
        checks[f"{name}.mandatory_keys"] = not missing
        if missing:
            errors.append(f"{name} missing keys: {sorted(missing)}")

    env = configs.get("environment_config", {})
    vehicle = configs.get("vehicle_config", {})
    costs = configs.get("cost_config", {})
    bellman = configs.get("bellman_config", {})
    defender = configs.get("defender_config", {})
    grid = env.get("grid", {})
    airspace = env.get("airspace", {})
    checks["grid.x_spacing_positive"] = grid.get("x_count", 0) >= 2 and grid.get("x_max", 0.0) > grid.get("x_min", 0.0)
    checks["grid.y_spacing_positive"] = grid.get("y_count", 0) >= 2 and grid.get("y_max", 0.0) > grid.get("y_min", 0.0)
    checks["grid.h_spacing_positive"] = grid.get("h_count", 0) >= 2 and grid.get("h_max", 0.0) > grid.get("h_min", 0.0)
    # h_goal is terrain-derived (see environment_config's h_goal comment),
    # so it cannot be bounds-checked before Phase 2 builds terrain -- exactly
    # like h_sensor, which is likewise not bounds-checked here either.
    checks["goal.inside_environment"] = (
        airspace.get("x_min", 1.0) <= env.get("x_goal", 0.0) <= airspace.get("x_max", -1.0)
        and airspace.get("y_min", 1.0) <= env.get("y_goal", 0.0) <= airspace.get("y_max", -1.0)
    )
    checks["vehicle.speed_bounds"] = (
        0.0 < vehicle.get("glide_speed_min", 0.0)
        < vehicle.get("glide_speed_max", 0.0)
    )
    checks["vehicle.gamma_always_negative"] = (
        vehicle.get("gamma_min_deg", 0.0) < vehicle.get("gamma_max_deg", 0.0) < 0.0
    )
    checks["vehicle.heading_full_range"] = (
        vehicle.get("heading_min_deg", 1.0) < vehicle.get("heading_max_deg", -1.0)
        and vehicle.get("heading_max_deg", 0.0) - vehicle.get("heading_min_deg", 0.0) <= 360.0
    )
    attacker = costs.get("attacker", {})
    defender_cost = costs.get("defender", {})
    checks["cost.attacker_weights_sum"] = abs(
        attacker.get("w_pod", 0.0) + attacker.get("w_time", 0.0) - 1.0
    ) <= 1.0e-12
    attacker_pod = attacker.get("normalization", {}).get("pod", {})
    checks["cost.attacker_hazard_normalization"] = (
        attacker_pod.get("method") == "cumulative_hazard_reference"
        and isfinite(attacker_pod.get("hazard_reference", 0.0))
        and attacker_pod.get("hazard_reference", 0.0) > 0.0
        and attacker_pod.get("mission_pod_reporting_only") is True
    )
    checks["cost.defender_weights_sum"] = abs(
        defender_cost.get("w_pod", 0.0)
        + defender_cost.get("w_cover", 0.0)
        - 1.0
    ) <= 1.0e-12
    checks["cost.shared_attacker_objective"] = (
        bellman.get("attacker_objective_id") == attacker.get("objective_id")
    )
    validation = configs.get("validation_config", {})
    checks["goal.region_radius"] = (
        isfinite(validation.get("goal_radius", 0.0))
        and validation.get("goal_radius", 0.0) > 0.0
    )
    bounds = defender.get("continuous_search_bounds", {})
    checks["defender.continuous_bounds"] = (
        airspace.get("x_min", 1.0) <= bounds.get("x_sensor_min", 0.0)
        < bounds.get("x_sensor_max", 0.0) <= airspace.get("x_max", -1.0)
        and airspace.get("y_min", 1.0) <= bounds.get("y_sensor_min", 0.0)
        < bounds.get("y_sensor_max", 0.0) <= airspace.get("y_max", -1.0)
    )
    checks["defender.optimizer_supported"] = defender.get("optimizer") == "scipy_direct_global"
    checks["random_seed.shared"] = (
        bellman.get("random_seed") == GLOBAL_RANDOM_SEED
    )
    for key, path in project_paths.__dict__.items():
        if key != "project_root":
            checks[f"paths.{key}_exists"] = Path(path).is_dir()

    duplicate_threshold = bellman.get("duplicate_threshold")
    required_duplicate_thresholds = {
        "switching_distance",
        "trajectory_rms_distance",
        "mission_cost_difference",
        "path_length_relative_difference",
        "resample_point_count",
    }
    checks["bellman.duplicate_threshold_complete"] = (
        isinstance(duplicate_threshold, dict)
        and required_duplicate_thresholds <= set(duplicate_threshold)
        and duplicate_threshold["switching_distance"] >= 0.0
        and duplicate_threshold["trajectory_rms_distance"] >= 0.0
        and duplicate_threshold["mission_cost_difference"] >= 0.0
        and duplicate_threshold["path_length_relative_difference"] >= 0.0
        and duplicate_threshold["resample_point_count"] >= 2
    )
    if vehicle.get("segment_length") is None:
        warnings.append(
            "Vehicle segment_length is deferred to the future transcription "
            "contract instead of fabricating a Phase 1 value"
        )

    failed_checks = [name for name, passed in checks.items() if not passed]
    errors.extend(f"Failed configuration check: {name}" for name in failed_checks)
    passed = not errors
    return {
        "passed": passed,
        "checks": checks,
        "metrics": {
            "configuration_dictionary_count": len(configs),
            "required_dictionary_count": len(_REQUIRED_KEYS),
            "directory_count": 5,
        },
        "tolerances": {"weight_sum": 1.0e-12},
        "summary": (
            "Phase 1 configuration is valid"
            if passed
            else f"Phase 1 configuration has {len(errors)} error(s)"
        ),
        "status": {
            "success": passed,
            "code": "OK" if passed else "CONFIGURATION_INVALID",
            "message": "Configuration validation passed" if passed else "; ".join(errors),
            "warnings": warnings,
            "failed_checks": failed_checks,
        },
    }


def build_configuration_bundle(project_root: Path | None = None) -> dict[str, Any]:
    """Build the complete validated Phase 1 ConfigurationBundle.

    Inputs
    ------
    project_root:
        Optional repository root for the path manager.

    Outputs
    -------
    dict
        Universal result envelope. The primary result contains all nine
        dictionaries, project paths, global seed, and logging utilities.

    Assumptions
    -----------
    Values are carried over from p1b_4D where the physics is unchanged
    (vehicle aero, detection coefficients, cost weights); only the
    position/action dimensionality and defender search space differ.

    Notes
    -----
    Directory creation and logger setup are the only side effects. No
    terrain, LOS, symbolic, cost, optimization, or plot computation occurs.
    """
    project_paths = create_project_paths(project_root)
    configs = _configuration_dicts()
    logger = create_phase_logger(project_paths.log_dir)
    with log_phase(logger, "Phase 1: Configuration") as phase_record:
        validation = validate_configuration(configs, project_paths)
        phase_record["warnings"].extend(validation["status"]["warnings"])
        if not validation["passed"]:
            raise ValueError(validation["status"]["message"])

    primary_result: dict[str, Any] = {
        **configs,
        "project_paths": project_paths,
        "global_random_seed": GLOBAL_RANDOM_SEED,
        "logging_utilities": {
            "logger": logger,
            "phase_context": log_phase,
            "close_logger": close_phase_logger,
        },
    }
    return {
        "primary_result": primary_result,
        "validation": {
            key: value for key, value in validation.items() if key != "status"
        },
        "metadata": {
            "schema_name": "ConfigurationBundle3D",
            "schema_version": CONFIG_SCHEMA_VERSION,
            "producer_phase": 1,
            "producer_module": "p1b_3DExtension.configuration",
            "random_seed": GLOBAL_RANDOM_SEED,
            "units": deepcopy(environment_config["units"]),
            "project_paths": project_paths.as_dict(),
            "source": "p1b_3DExtension/p1b_3DExtension.ipynb v1 toy configuration",
        },
        "status": validation["status"],
    }
