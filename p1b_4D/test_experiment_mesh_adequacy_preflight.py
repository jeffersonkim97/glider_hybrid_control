from __future__ import annotations

import numpy as np

from p1b_4D.experiment_mesh_adequacy_preflight import (
    build_preflight_config,
    refined_count,
    successor_direction_signature,
)
from p1b_4D.phase_logging import close_phase_logger


def test_refined_count_halves_original_intervals() -> None:
    assert refined_count(321) == 641
    assert refined_count(201) == 401
    assert refined_count(234) == 467


def test_refinement_halves_both_spacings_and_preserves_directions(tmp_path) -> None:
    native = build_preflight_config("two_hill", "native", tmp_path / "native")
    refined = build_preflight_config(
        "two_hill", "refined_2x", tmp_path / "refined"
    )
    try:
        native_grid = native["primary_result"]["environment_config"]["grid"]
        refined_grid = refined["primary_result"]["environment_config"]["grid"]

        assert refined_grid["z_count"] == 321
        assert refined_grid["h_count"] == 201
        assert refined_grid["z_spacing"] == native_grid["z_spacing"] / 2.0
        assert refined_grid["h_spacing"] == native_grid["h_spacing"] / 2.0
        np.testing.assert_allclose(
            successor_direction_signature(native),
            successor_direction_signature(refined),
            rtol=0.0,
            atol=1e-15,
        )
    finally:
        close_phase_logger(native["primary_result"]["logging_utilities"]["logger"])
        close_phase_logger(refined["primary_result"]["logging_utilities"]["logger"])
