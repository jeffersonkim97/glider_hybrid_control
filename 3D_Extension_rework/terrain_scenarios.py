"""Modular 3D terrain catalog derived from the authoritative 2D cases."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .configuration import build_configuration, validate_configuration


TERRAIN_SCENARIOS: dict[str, dict[str, Any]] = {
    "centered_single_hill": {
        "x_bounds_m": (0.0, 3000.0), "h_bounds_m": (0.0, 400.0),
        "goal_xy_m": (3000.0, 500.0), "sensor_x_bounds_m": (1750.0, 2500.0),
        "geometry_x_count": 151, "state_x_count": 31,
        "hills": ({"center_xy_m": (1500.0, 500.0), "height_m": 200.0,
                   "width_x_m": 400.0, "width_y_m": 400.0},),
    },
    "asymmetric_single_hill": {
        "x_bounds_m": (0.0, 3000.0), "h_bounds_m": (0.0, 400.0),
        "goal_xy_m": (3000.0, 500.0), "sensor_x_bounds_m": (1750.0, 2500.0),
        "geometry_x_count": 151, "state_x_count": 31,
        "hills": ({"center_xy_m": (1500.0, 700.0), "height_m": 200.0,
                   "width_x_m": 400.0, "width_y_m": 400.0},),
    },
    "two_hill": {
        "x_bounds_m": (0.0, 2750.0), "h_bounds_m": (0.0, 200.0),
        "goal_xy_m": (2500.0, 500.0), "sensor_x_bounds_m": (1300.0, 2200.0),
        "geometry_x_count": 166, "state_x_count": 34,
        "hills": (
            {"center_xy_m": (1000.0, 500.0), "height_m": 100.0,
             "width_x_m": 100.0, "width_y_m": 100.0},
            {"center_xy_m": (2000.0, 500.0), "height_m": 50.0,
             "width_x_m": 100.0, "width_y_m": 100.0},
        ),
    },
    "goal_in_valley": {
        "x_bounds_m": (0.0, 4000.0), "h_bounds_m": (0.0, 200.0),
        "goal_xy_m": (2500.0, 500.0), "sensor_x_bounds_m": (1700.0, 2200.0),
        "geometry_x_count": 161, "state_x_count": 33,
        "hills": (
            {"center_xy_m": (1500.0, 500.0), "height_m": 100.0,
             "width_x_m": 100.0, "width_y_m": 100.0},
            {"center_xy_m": (3500.0, 500.0), "height_m": 100.0,
             "width_x_m": 100.0, "width_y_m": 100.0},
        ),
    },
}


def build_scenario_configuration(scenario_id: str) -> dict[str, Any]:
    """Return one independent scenario while preserving solver contracts."""
    if scenario_id not in TERRAIN_SCENARIOS:
        raise ValueError(f"Unknown 3D terrain scenario: {scenario_id}")
    configuration = build_configuration()
    specification = deepcopy(TERRAIN_SCENARIOS[scenario_id])
    environment = configuration["environment"]
    environment["x_bounds_m"] = specification["x_bounds_m"]
    environment["h_bounds_m"] = specification["h_bounds_m"]
    environment["goal_xy_m"] = specification["goal_xy_m"]
    environment["terrain"] = {
        "scenario_id": scenario_id, "hills": specification["hills"],
    }
    x_sensor_bounds = specification["sensor_x_bounds_m"]
    environment["sensor_xy_m"] = (
        0.5 * (x_sensor_bounds[0] + x_sensor_bounds[1]), 500.0,
    )
    configuration["grid"]["x_count"] = specification["geometry_x_count"]
    configuration["grid"]["h_count"] = 81
    configuration["state_grid"]["x_count"] = specification["state_x_count"]
    configuration["state_grid"]["h_count"] = 21
    defender = configuration["defender_search"]
    defender["x_bounds_m"] = x_sensor_bounds
    defender["y_bounds_m"] = (0.0, 1000.0)
    centered = all(abs(hill["center_xy_m"][1] - 500.0) <= 1.0e-12
                   for hill in specification["hills"])
    defender["use_y_reflection_symmetry"] = centered
    defender["verify_y_reflection_symmetry"] = centered
    configuration["cost"]["attacker"]["time_reference_s"] = (
        specification["goal_xy_m"][0]
        / configuration["vehicle"]["glide_speed_max_mps"]
    )
    configuration["weight_sensitivity"]["reference_sensor_xy_m"] = (
        x_sensor_bounds[1], 500.0,
    )
    return validate_configuration(configuration)
