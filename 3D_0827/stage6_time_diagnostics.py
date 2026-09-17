"""Generate Stage-6 exact time-only Bellman diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from bellman_solver import run_time_optimal_problem
from bellman_state import BellmanState, BellmanStateGrid
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import MapBounds
from scenario import Point3D
from stage5_bellman_diagnostics import ObstacleFreeTerrain
from terrain_catalog import build_terrain
from visualization import plot_time_optimal_trajectory


DEFAULT_STAGE6_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_6_bellman_time_objective"
)
DEFAULT_GRAPH_BOUNDS = MapBounds(-8.0, 8.0, -4.0, 4.0)


def _path_summary(run: Any) -> dict[str, Any]:
    if run.path is None:
        return {
            "reachable": False,
            "bellman_value_s": float("inf"),
            "path_state_ids": [],
            "path_positions_map": [],
        }
    path = run.path
    return {
        "reachable": True,
        "bellman_value_s": path.bellman_value_s,
        "summed_edge_duration_s": path.summed_edge_duration_s,
        "geometric_replay_duration_s": path.geometric_replay_duration_s,
        "path_edge_count": len(path.edges),
        "path_state_ids": list(path.state_ids),
        "path_positions_map": [
            run.graph.grid.position_map(
                run.graph.grid.decode(state_id),
            ).tolist()
            for state_id in path.state_ids
        ],
    }


def generate_stage6_artifacts(
    output_directory: Path = DEFAULT_STAGE6_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Run obstacle-free and centered-cube time solves and write diagnostics."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    empty_bounds = MapBounds(0.0, 3.0, -1.0, 1.0)
    empty_grid = BellmanStateGrid(
        bounds=empty_bounds,
        maximum_altitude_map=0.3,
    )
    empty_start = BellmanState(0, 1, 3, 0)
    empty_run = run_time_optimal_problem(
        empty_grid,
        ObstacleFreeTerrain(empty_bounds),
        Point3D(3.0, 0.0, 0.0),
        empty_start,
    )
    if empty_run.path is None:
        raise RuntimeError("canonical obstacle-free start unexpectedly unreachable")
    direct_lower_bound_s = (
        3.0
        * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
        / DEFAULT_GLIDER.best_glide_speed_mps
    )
    if abs(empty_run.path.bellman_value_s - direct_lower_bound_s) > 1.0e-10:
        raise RuntimeError("obstacle-free Bellman time misses analytical bound")

    canonical_grid = BellmanStateGrid(bounds=DEFAULT_GRAPH_BOUNDS)
    canonical_start = BellmanState(0, 4, 20, 0)  # (-8, 0, 2), heading east
    canonical_run = run_time_optimal_problem(
        canonical_grid,
        build_terrain("centered_cube", bounds=DEFAULT_GRAPH_BOUNDS),
        Point3D(8.0, 0.0, 0.0),
        canonical_start,
    )
    if canonical_run.path is None:
        raise RuntimeError("canonical centered-cube start unexpectedly unreachable")

    for filename, run in (
        ("obstacle_free_time_optimal_trajectory.html", empty_run),
        ("centered_cube_time_optimal_trajectory.html", canonical_run),
    ):
        figure = plot_time_optimal_trajectory(run)
        figure.write_html(
            output_directory / filename,
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )

    summary = {
        "objective": {
            "name": "elapsed_time",
            "path_length_convention": "horizontal",
            "glide_speed_mps": DEFAULT_GLIDER.best_glide_speed_mps,
            "best_glide_ratio": DEFAULT_GLIDER.best_glide_ratio,
            "meters_per_map_unit": (
                DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
            ),
            "goal_tolerance_m": DEFAULT_GLIDER.goal_tolerance_m,
            "tie_break": "lowest action index",
        },
        "obstacle_free": {
            "start_state_id": empty_run.start_state_id,
            "analytical_direct_lower_bound_s": direct_lower_bound_s,
            "graph_statistics": asdict(empty_run.graph.statistics),
            "metrics": asdict(empty_run.metrics),
            "path": _path_summary(empty_run),
        },
        "centered_cube": {
            "start_state_id": canonical_run.start_state_id,
            "start_position_map": canonical_grid.position_map(
                canonical_start,
            ).tolist(),
            "graph_statistics": asdict(canonical_run.graph.statistics),
            "goal_reachable_state_count": (
                canonical_run.solution.goal_reachable_state_count
            ),
            "metrics": asdict(canonical_run.metrics),
            "path": _path_summary(canonical_run),
        },
    }
    (output_directory / "bellman_time_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "Stage 6 time objective: "
        f"direct={empty_run.path.bellman_value_s:.6f} s, "
        f"centered_cube={canonical_run.path.bellman_value_s:.6f} s, "
        f"states={canonical_run.metrics.state_count}, "
        f"edges={canonical_run.metrics.edge_count}, "
        f"solve={canonical_run.metrics.timing.solve_s:.4f} s"
    )
    return summary


if __name__ == "__main__":
    generate_stage6_artifacts()
    print(
        "Generated Stage-6 diagnostics in "
        f"{DEFAULT_STAGE6_FIGURE_DIRECTORY}"
    )
