"""Frozen nested finite-problem family for ACC Direction B.

This module translates ``b0_nested_discretization_protocol.md`` into one
machine-checkable configuration factory.  It does not run the follower or
claim anything about the continuous optimum.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .configuration import validate_configuration
from .stage_cost import construct_state_grids


DIRECTION_B_GRID_COUNTS: dict[str, tuple[tuple[int, int], ...]] = {
    "single_hill": ((161, 101), (321, 201), (641, 401)),
    "two_hill": ((81, 51), (161, 101), (321, 201)),
    "goal_in_valley": ((117, 51), (233, 101), (465, 201)),
}

DIRECTION_B_DOMAIN_SPANS: dict[str, tuple[float, float]] = {
    "single_hill": (5500.0, 400.0),
    "two_hill": (2750.0, 200.0),
    "goal_in_valley": (4000.0, 200.0),
}

DIRECTION_B_SPEEDS: dict[str, np.ndarray] = {
    "V5": np.linspace(10.0, 22.6, 5),
    "V9": np.linspace(10.0, 22.6, 9),
}
DIRECTION_B_SHALLOW_BACKBONE_FORWARD_CELLS_L0 = 4
DIRECTION_B_PRODUCTION_LEVEL = 2
DIRECTION_B_PRODUCTION_ACTION_FAMILY = "enriched"
DIRECTION_B_PRODUCTION_SPEED_FAMILY = "V9"
DIRECTION_B_PRODUCTION_PLANNING_QUADRATURE_COUNT = 9
DIRECTION_B_PRODUCTION_EVALUATOR_SAMPLE_COUNT = 1025
DIRECTION_B_PRODUCTION_CONFIGURATION_ID = (
    "direction_b_l2_enriched_v9_q9_e1025"
)
for _speeds in DIRECTION_B_SPEEDS.values():
    _speeds.setflags(write=False)


def build_direction_b_configuration(
    configuration_bundle: dict[str, Any],
    terrain_name: str,
    level: int,
    *,
    action_family: str = "enriched",
    speed_family: str = "V5",
    edge_quadrature_count: int = 9,
    shallow_backbone_forward_cells_l0: int | None = (
        DIRECTION_B_SHALLOW_BACKBONE_FORWARD_CELLS_L0
    ),
) -> dict[str, Any]:
    """Return an independent bundle configured for one frozen B problem.

    The caller supplies the physical terrain/goal data.  This function checks
    that its domain span matches the named B terrain before changing only the
    grid, action lattice, speed lattice, and planning quadrature.
    """
    if terrain_name not in DIRECTION_B_GRID_COUNTS:
        raise ValueError(f"Unsupported Direction-B terrain: {terrain_name}")
    if isinstance(level, bool) or level not in (0, 1, 2):
        raise ValueError("Direction-B level must be 0, 1, or 2")
    if action_family not in {"enriched", "transported"}:
        raise ValueError("action_family must be 'enriched' or 'transported'")
    if speed_family not in DIRECTION_B_SPEEDS:
        raise ValueError("speed_family must be 'V5' or 'V9'")
    if isinstance(edge_quadrature_count, bool) or edge_quadrature_count < 2:
        raise ValueError("edge_quadrature_count must be an integer at least 2")
    if shallow_backbone_forward_cells_l0 is not None and (
        isinstance(shallow_backbone_forward_cells_l0, bool)
        or not isinstance(shallow_backbone_forward_cells_l0, (int, np.integer))
        or shallow_backbone_forward_cells_l0 < 2
    ):
        raise ValueError(
            "shallow_backbone_forward_cells_l0 must be an integer at least 2"
        )
    if action_family == "transported":
        shallow_backbone_forward_cells_l0 = None

    bundle = deepcopy(configuration_bundle)
    configs = bundle["primary_result"]
    environment = configs["environment_config"]
    grid = environment["grid"]
    z_span = float(grid["z_max"] - grid["z_min"])
    h_span = float(grid["h_max"] - grid["h_min"])
    expected_span = DIRECTION_B_DOMAIN_SPANS[terrain_name]
    if not np.allclose((z_span, h_span), expected_span, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{terrain_name} requires domain spans {expected_span}, "
            f"received {(z_span, h_span)}"
        )

    z_count, h_count = DIRECTION_B_GRID_COUNTS[terrain_name][level]
    environment["terrain"]["reference_z_count"] = (
        DIRECTION_B_GRID_COUNTS[terrain_name][2][0]
    )
    grid.update({
        "z_count": z_count,
        "z_spacing": z_span / (z_count - 1),
        "h_count": h_count,
        "h_spacing": h_span / (h_count - 1),
        "v_count": DIRECTION_B_SPEEDS[speed_family].size,
    })
    vehicle = configs["vehicle_config"]
    vehicle["glide_speed_min"] = float(DIRECTION_B_SPEEDS[speed_family][0])
    vehicle["glide_speed_max"] = float(DIRECTION_B_SPEEDS[speed_family][-1])
    vehicle["glide_speed_count"] = int(DIRECTION_B_SPEEDS[speed_family].size)

    maximum_forward_cells = 2**level
    maximum_descent_cells = 2 ** (level + 1)
    supplemental_offsets: tuple[tuple[int, int], ...] = ()
    if shallow_backbone_forward_cells_l0 is not None:
        stride = 2**level
        supplemental_offsets = ((
            int(shallow_backbone_forward_cells_l0) * stride,
            stride,
        ),)
    regular_maximum_forward_cells = max(
        (maximum_forward_cells,)
        + tuple(offset[0] for offset in supplemental_offsets)
    )
    options = configs["attacker_solver_config"]["successor_grid"]
    options.update({
        "maximum_forward_cells": maximum_forward_cells,
        "maximum_descent_cells": maximum_descent_cells,
        "edge_quadrature_count": int(edge_quadrature_count),
        "virtual_switch_maximum_forward_cells": maximum_forward_cells,
        "virtual_switch_maximum_descent_cells": maximum_descent_cells,
        "virtual_switch_maximum_forward_distance": (
            DIRECTION_B_DOMAIN_SPANS[terrain_name][0]
            / (DIRECTION_B_GRID_COUNTS[terrain_name][0][0] - 1)
        ),
        "virtual_switch_maximum_descent_distance": (
            DIRECTION_B_DOMAIN_SPANS[terrain_name][1]
            * 2.0 / (DIRECTION_B_GRID_COUNTS[terrain_name][0][1] - 1)
        ),
        "action_family": action_family,
        "virtual_switch_target_family": f"physical_box_{action_family}",
        "nested_level": level,
        "supplemental_offsets": supplemental_offsets,
    })
    if supplemental_offsets:
        options["virtual_switch_maximum_forward_distance"] = (
            regular_maximum_forward_cells * grid["z_spacing"]
        )
    configs["attacker_solver_config"][
        "transition_model"
    ] = "successor_grid_physical_edge"
    configs["bellman_config"]["search_options"][
        "segment_check_count"
    ] = int(edge_quadrature_count)
    configs["direction_b_protocol"] = {
        "terrain_name": terrain_name,
        "level": level,
        "action_family": action_family,
        "speed_family": speed_family,
        "planning_quadrature_count": int(edge_quadrature_count),
        "common_evaluator_sample_count": 129,
        "physical_geometry_reference_z_count": (
            DIRECTION_B_GRID_COUNTS[terrain_name][2][0]
        ),
        "shallow_backbone_forward_cells_l0": (
            int(shallow_backbone_forward_cells_l0)
            if shallow_backbone_forward_cells_l0 is not None else None
        ),
        "endpoint_snapping": False,
    }

    validation = validate_configuration(configs, configs["project_paths"])
    if not validation["passed"]:
        raise ValueError(validation["status"]["message"])
    bundle["validation"] = {
        key: value for key, value in validation.items() if key != "status"
    }
    bundle["status"] = validation["status"]
    bundle["metadata"] = {
        **bundle.get("metadata", {}),
        "direction_b_protocol": deepcopy(configs["direction_b_protocol"]),
    }
    return bundle


def build_direction_b_production_configuration(
    configuration_bundle: dict[str, Any],
    terrain_name: str,
) -> dict[str, Any]:
    """Return the B4-frozen configuration used by the finite C-lite game."""
    bundle = build_direction_b_configuration(
        configuration_bundle,
        terrain_name,
        DIRECTION_B_PRODUCTION_LEVEL,
        action_family=DIRECTION_B_PRODUCTION_ACTION_FAMILY,
        speed_family=DIRECTION_B_PRODUCTION_SPEED_FAMILY,
        edge_quadrature_count=(
            DIRECTION_B_PRODUCTION_PLANNING_QUADRATURE_COUNT
        ),
    )
    protocol = bundle["primary_result"]["direction_b_protocol"]
    protocol.update({
        "common_evaluator_sample_count": (
            DIRECTION_B_PRODUCTION_EVALUATOR_SAMPLE_COUNT
        ),
        "production_frozen": True,
        "production_configuration_id": (
            DIRECTION_B_PRODUCTION_CONFIGURATION_ID
        ),
        "intended_use": "finite_c_lite_leader_follower_game",
        "continuous_optimum_claimed": False,
    })
    bundle["metadata"]["direction_b_protocol"] = deepcopy(protocol)
    return bundle


def construct_direction_b_grids(
    configuration_bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct grids and verify that the configured speed family is exact."""
    configs = configuration_bundle["primary_result"]
    protocol = configs.get("direction_b_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("configuration_bundle is not a Direction-B bundle")
    grids = construct_state_grids(
        configs["environment_config"], configs["vehicle_config"]
    )
    expected_speed = DIRECTION_B_SPEEDS[protocol["speed_family"]]
    if not np.array_equal(grids["v"], expected_speed):
        raise RuntimeError("Constructed speed grid does not match frozen B lattice")
    return grids


def direction_b_physical_envelope(
    configuration_bundle: dict[str, Any],
) -> tuple[float, float]:
    """Return the maximum regular-edge displacement in physical metres."""
    configs = configuration_bundle["primary_result"]
    options = configs["attacker_solver_config"]["successor_grid"]
    grid = configs["environment_config"]["grid"]
    offsets = [
        (forward, descent)
        for forward in range(1, int(options["maximum_forward_cells"]) + 1)
        for descent in range(1, int(options["maximum_descent_cells"]) + 1)
    ]
    offsets.extend(options.get("supplemental_offsets", ()))
    return (
        float(max(forward for forward, _ in offsets) * grid["z_spacing"]),
        float(max(descent for _, descent in offsets) * grid["h_spacing"]),
    )
