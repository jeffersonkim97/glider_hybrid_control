from __future__ import annotations

from p1b_4D.experiment_p2_selected_ranking_stability import (
    RESOLUTION_COUNTS,
    build_resolution_config,
)
from p1b_4D.phase_logging import close_phase_logger


def test_p3_resolution_counts_match_protocol() -> None:
    assert RESOLUTION_COUNTS["single_hill"] == {
        "coarse": (161, 101),
        "native": (321, 201),
        "refined": (641, 401),
    }
    assert RESOLUTION_COUNTS["two_hill"] == {
        "coarse": (81, 51),
        "native": (161, 101),
        "refined": (321, 201),
    }
    assert RESOLUTION_COUNTS["goal_in_valley"] == {
        "coarse": (117, 51),
        "native": (234, 101),
        "refined": (467, 201),
    }


def test_p3_native_and_refined_preserve_successor_aspect_ratio(tmp_path) -> None:
    native = build_resolution_config("two_hill", "native", tmp_path / "native")
    refined = build_resolution_config("two_hill", "refined", tmp_path / "refined")
    try:
        native_grid = native["primary_result"]["environment_config"]["grid"]
        refined_grid = refined["primary_result"]["environment_config"]["grid"]
        assert native_grid["z_spacing"] / native_grid["h_spacing"] == (
            refined_grid["z_spacing"] / refined_grid["h_spacing"]
        )
    finally:
        close_phase_logger(native["primary_result"]["logging_utilities"]["logger"])
        close_phase_logger(refined["primary_result"]["logging_utilities"]["logger"])
