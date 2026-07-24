"""Central Phase 1 configuration for the Stackelberg security project."""

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
    "z_start": 0.0,
    "h_start": 0.0,
    "z_goal": 2500.0,
    # h_goal is not configured here: it is the terrain elevation at z_goal,
    # computed once terrain exists (see geometry.goal_position_from_environment),
    # exactly like h_sensor is derived from z_sensor rather than configured.
    "terrain": {
        "z_min": 0.0,
        "z_max": 2750.0,
        # Terrain height is the sum of every hill's Gaussian. A single-hill
        # tuple reproduces the original one-ridge terrain exactly; multiple
        # hills combine into one continuous profile (one CubicSpline), not
        # separate terrain objects with separately-computed LOS geometry.
        "hills": (
            {"z_ridge": 1250.0, "h_ridge": 100.0, "width": 200.0},
        ),
    },
    "grid": {
        "z_min": 0.0,
        "z_max": 2750.0,
        "z_count": 161,
        "z_spacing": 17.1875,
        "h_min": 0.0,
        "h_max": 200.0,
        "h_count": 101,
        "h_spacing": 2.0,
        "v_count": 5,
        "gamma_count": 20,
        "axis_order_4d": ("z", "h", "v", "gamma"),
    },
    "airspace": {
        "z_min": 0.0,
        "z_max": 2750.0,
        "h_min": 0.0,
        "h_max": 200.0,
    },
    "simulation": {
        "z_min": 0.0,
        "z_max": 2750.0,
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
    "glide_speed_count": 5,
    "gamma_min_deg": -90.0,
    "gamma_max_deg": -1.0,
    "gamma_count": 20,
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
        "from the existing 4D notebook"
    ),
}

sensor_config: dict[str, Any] = {
    "default_z_sensor": 2000.0,
    "mount_height": 0.0,
    "height_rule": "terrain(z_sensor) + mount_height",
    "los": {
        "terrain_numerical_tolerance": 1.0e-6,
        "edge_gamma_tolerance_deg": 7.0,
        "tangent_bisection_iterations": 64,
    },
    "detection": {
        "range_floor": 10.0,
        "acoustic_coefficient": 2.4e-3,
        "acoustic_speed_exponent": 4,
        # radar/doppler scaled 1/40 from prior values: those saturated PoD to
        # ~1 within ~300m (mission_pod > 0.999 almost everywhere reachable),
        # collapsing the Defender placement game to a single kill radius.
        "radar_coefficient": 1.3e7,
        "doppler_coefficient": 3.325e4,
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

# Experimental only: consumed solely by the deprecated, disconnected
# attacker_nlp.py continuous-refinement comparison. Not required by
# validate_configuration and not read by the authoritative Bellman solver.
nlp_config: dict[str, Any] = {
    "number_of_nodes": 80,
    # "hazard_homotopy_scales": (0.01, 0.1, 1.0),
    "hazard_homotopy_scales": (1.0,),
    "minimum_interval_time": 0.25,
    "maximum_interval_time": 3.0,
    "solver": "ipopt",
    "solver_tolerance": 1.0e-6,
    "maximum_iterations": 1000,
    "ipopt_options": {
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 2000,
        "ipopt.tol": 1.0e-6,
        "ipopt.acceptable_tol": 1.0e-4,
    },
    "multi_start": {
        "enabled": True,
        "start_count": 3,
        "random_seed": GLOBAL_RANDOM_SEED,
    },
    "constraint_tolerance": 1.0e-6,
    "initialization": {
        "source": "bellman_warm_start",
        "fallback": None,
    },
    "attacker_objective_id": "attacker_hazard_time_v2",
}

defender_config: dict[str, Any] = {
    "continuous_search_bounds": {
        "z_sensor_min": 1500.0,
        "z_sensor_max": 2600.0,
    },
    "termination_tolerance": 5.0,
    "xtol": 5.0,
    "maximum_iterations": 41,
    "optimizer": "hierarchical_coarse_to_fine_brent",
    "optimizer_contract": "continuous_derivative_free_callback",
    "coarse_sample_count": 5,
    "basin_prominence_threshold": 1.0e-4,
    "diagnostic_search": {
        "coarse_sensor_count": 41,
        "fine_neighborhood_half_width": 50.0,
        "fine_spacing": 5.0,
        "max_fine_centers": 3,
        "switch_jump_threshold": 100.0,
    },
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
    "goal_radius": 10.0,
    "goal_tolerance_z": 10.0,
    "goal_tolerance_h": 10.0,
    "los_tolerance": 3.0,
    "dynamic_tolerance": 1.0e-6,
    "solver_tolerance": 1.0e-6,
    "objective_tolerance": 1.0e-8,
    "detection_probability_tolerance": 1.0e-12,
}

_REQUIRED_KEYS: dict[str, set[str]] = {
    "environment_config": {
        "z_start", "h_start", "z_goal", "terrain", "grid",
        "airspace", "simulation", "units",
    },
    "vehicle_config": {
        "powered_speed", "glide_speed_min", "glide_speed_max", "gamma_min_deg",
        "gamma_max_deg", "time_step", "mass", "wing_area", "dynamic_limits",
    },
    "sensor_config": {"default_z_sensor", "mount_height", "los", "detection"},
    "cost_config": {"attacker", "defender", "local_stage_cost"},
    "bellman_config": {
        "maximum_iterations", "top_k",
        "duplicate_threshold", "warm_start", "search_options",
    },
    "defender_config": {
        "continuous_search_bounds", "termination_tolerance",
        "xtol", "maximum_iterations", "optimizer", "coarse_sample_count",
        "basin_prominence_threshold",
    },
    "plot_config": {
        "default_figure_size", "colormaps", "line_styles", "marker_styles",
        "colors", "terrain_zorder", "heatmap_alpha",
    },
    "io_config": {
        "results_directory", "json_directory", "npz_directory",
        "figure_directory", "log_directory",
    },
    "validation_config": {
        "terrain_tolerance", "goal_radius", "goal_tolerance_z", "goal_tolerance_h",
        "los_tolerance", "dynamic_tolerance", "solver_tolerance",
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
        "nlp_config": deepcopy(nlp_config),
        "defender_config": deepcopy(defender_config),
        "plot_config": deepcopy(plot_config),
        "io_config": deepcopy(io_config),
        "validation_config": deepcopy(validation_config),
    }


def validate_configuration(
    configs: dict[str, dict[str, Any]], project_paths: ProjectPaths
) -> dict[str, Any]:
    """Validate dictionaries, cross-config invariants, and output directories.

    Inputs
    ------
    configs:
        Mapping containing all ten configuration dictionaries.
    project_paths:
        Resolved paths returned by the centralized path manager.

    Outputs
    -------
    dict
        Validation metrics, checks, status, warnings, and diagnostics.

    Assumptions
    -----------
    No terrain, LOS, symbolic, cost-map, or optimization object is present.

    Notes
    -----
    This function is deterministic and has no side effects.
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
    checks["grid.z_spacing_consistent"] = (
        grid.get("z_spacing")
        == (grid.get("z_max", 0.0) - grid.get("z_min", 0.0))
        / (grid.get("z_count", 2) - 1)
    )
    checks["grid.h_spacing_consistent"] = (
        grid.get("h_spacing")
        == (grid.get("h_max", 0.0) - grid.get("h_min", 0.0))
        / (grid.get("h_count", 2) - 1)
    )
    # h_goal is terrain-derived (see environment_config's h_goal comment),
    # so it cannot be bounds-checked before Phase 2 builds terrain -- exactly
    # like h_sensor, which is likewise not bounds-checked here either.
    checks["goal.inside_environment"] = (
        env.get("z_start", 1.0) <= env.get("z_goal", 0.0)
        <= env.get("airspace", {}).get("z_max", -1.0)
    )
    checks["vehicle.speed_bounds"] = (
        0.0 < vehicle.get("glide_speed_min", 0.0)
        < vehicle.get("glide_speed_max", 0.0)
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
        env.get("airspace", {}).get("z_min", 1.0)
        <= bounds.get("z_sensor_min", 0.0)
        < bounds.get("z_sensor_max", 0.0)
        <= env.get("airspace", {}).get("z_max", -1.0)
    )
    checks["defender.optimizer_supported"] = (
        defender.get("optimizer") == "hierarchical_coarse_to_fine_brent"
        and defender.get("coarse_sample_count", 0) >= 3
        and defender.get("xtol", 0.0) > 0.0
        and defender.get("basin_prominence_threshold", -1.0) >= 0.0
    )
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
        Universal result envelope. The primary result contains all ten
        dictionaries, project paths, global seed, and logging utilities.

    Assumptions
    -----------
    Existing notebook values are authoritative and are not tuned here.

    Notes
    -----
    Directory creation and logger setup are the only side effects. No terrain,
    LOS, symbolic, cost, optimization, or plot computation occurs.
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
            "schema_name": "ConfigurationBundle",
            "schema_version": CONFIG_SCHEMA_VERSION,
            "producer_phase": 1,
            "producer_module": "p1b_4D.configuration",
            "random_seed": GLOBAL_RANDOM_SEED,
            "units": deepcopy(environment_config["units"]),
            "project_paths": project_paths.as_dict(),
            "source": "p1b_4D/p1b_casadi.ipynb active configuration",
        },
        "status": validation["status"],
    }
