from __future__ import annotations

import numpy as np

from p1b_4D.experiment_multiterrain_baselines import (
    TERRAINS,
    _reconcile_stackelberg_dominance,
)
from p1b_4D.experiment_strategic_baselines import _integrate_fixed_glide_edges


def test_p2_uses_preflight_selected_refined_spatial_grids() -> None:
    assert TERRAINS["single_hill"]["grid"]["z_count"] == 641
    assert TERRAINS["single_hill"]["grid"]["h_count"] == 401
    assert TERRAINS["two_hill"]["grid"]["z_count"] == 321
    assert TERRAINS["two_hill"]["grid"]["h_count"] == 201
    assert TERRAINS["goal_in_valley"]["grid"]["z_count"] == 467
    assert TERRAINS["goal_in_valley"]["grid"]["h_count"] == 201


def test_fixed_glide_hazard_uses_each_physical_edge_duration() -> None:
    trajectory = np.array([[0.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    speeds = np.array([10.0, 12.0])
    gammas = np.array([-0.1, -0.2])
    durations = np.array([2.0, 0.5])

    def constant_rate(*_arguments: float) -> tuple[float, float]:
        return (0.0, 3.0)

    hazard = _integrate_fixed_glide_edges(
        trajectory,
        speeds,
        gammas,
        durations,
        constant_rate,
        np.array([5.0, 0.0]),
        quadrature_count=9,
    )
    assert hazard == 3.0 * (2.0 + 0.5)


def test_fixed_glide_hazard_uses_trapezoidal_samples_along_edge() -> None:
    trajectory = np.array([[0.0, 1.0], [4.0, 0.0]])

    def rate_linear_in_z(
        z: float, *_arguments: float
    ) -> tuple[float, float]:
        return (0.0, 1.0 + z)

    hazard = _integrate_fixed_glide_edges(
        trajectory,
        np.array([10.0]),
        np.array([-0.1]),
        np.array([2.0]),
        rate_linear_in_z,
        np.array([5.0, 0.0]),
        quadrature_count=9,
    )
    # Linear rate from 1 to 5 has time-average 3 over a two-second edge.
    assert hazard == 6.0


def test_stackelberg_reconciliation_promotes_better_evaluated_baseline() -> None:
    terrain_state = {
        "selected_positions": {
            "fixed": 1.0,
            "coverage_only": 2.0,
            "nominal_path": 3.0,
            "stackelberg": 4.0,
        },
        "evaluations": {
            "fixed": {"status_success": True, "z_sensor_selected": 1.0, "defender_objective": 0.1},
            "coverage_only": {"status_success": True, "z_sensor_selected": 2.0, "defender_objective": 0.2},
            "nominal_path": {"status_success": True, "z_sensor_selected": 3.0, "defender_objective": 0.30001},
            "stackelberg": {"status_success": True, "z_sensor_selected": 4.0, "defender_objective": 0.3},
        },
    }
    _reconcile_stackelberg_dominance(terrain_state)
    assert terrain_state["selected_positions"]["stackelberg"] == 3.0
    assert terrain_state["evaluations"]["stackelberg"]["defender_objective"] == 0.30001
    assert terrain_state["selection_reconciliation"]["promoted_from"] == "nominal_path"


def test_stackelberg_reconciliation_does_not_repeat_for_equal_result() -> None:
    terrain_state = {
        "selected_positions": {
            "fixed": 1.0,
            "coverage_only": 2.0,
            "nominal_path": 3.0,
            "stackelberg": 3.0,
        },
        "evaluations": {
            "fixed": {"status_success": True, "z_sensor_selected": 1.0, "defender_objective": 0.1},
            "coverage_only": {"status_success": True, "z_sensor_selected": 2.0, "defender_objective": 0.2},
            "nominal_path": {"status_success": True, "z_sensor_selected": 3.0, "defender_objective": 0.3},
            "stackelberg": {"status_success": True, "z_sensor_selected": 3.0, "defender_objective": 0.3},
        },
    }
    _reconcile_stackelberg_dominance(terrain_state)
    assert "selection_reconciliation" not in terrain_state
