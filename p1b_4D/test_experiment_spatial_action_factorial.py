from __future__ import annotations

import numpy as np

from p1b_4D.experiment_spatial_action_factorial import build_factorial_config
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.stage_cost import construct_state_grids


def test_p4_speed_grid_is_nested_and_stencil_is_applied(tmp_path) -> None:
    standard = build_factorial_config(
        "native", "v5", "stencil_3x8", tmp_path / "standard"
    )
    enriched = build_factorial_config(
        "native", "v9", "stencil_6x16", tmp_path / "enriched"
    )
    try:
        standard_primary = standard["primary_result"]
        enriched_primary = enriched["primary_result"]
        standard_grids = construct_state_grids(
            standard_primary["environment_config"],
            standard_primary["vehicle_config"],
        )
        enriched_grids = construct_state_grids(
            enriched_primary["environment_config"],
            enriched_primary["vehicle_config"],
        )
        np.testing.assert_allclose(enriched_grids["v"][::2], standard_grids["v"])
        options = enriched_primary["attacker_solver_config"]["successor_grid"]
        assert options["maximum_forward_cells"] == 6
        assert options["maximum_descent_cells"] == 16
        assert options["virtual_switch_maximum_forward_cells"] == 6
        assert options["virtual_switch_maximum_descent_cells"] == 16
    finally:
        close_phase_logger(standard["primary_result"]["logging_utilities"]["logger"])
        close_phase_logger(enriched["primary_result"]["logging_utilities"]["logger"])
