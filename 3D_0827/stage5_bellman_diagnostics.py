"""Generate Stage-5 unit-cost glide-graph geometry diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from bellman_geometry import GlideTransitionModel
from bellman_graph import (
    build_bellman_graph,
    independent_reverse_reachability,
    solve_unit_cost_reachability,
)
from bellman_state import BellmanState, BellmanStateGrid
from map_geometry import MapBounds, TriangleMesh
from ray_tracing import RayHit, TriangleRayTracer
from scenario import Point3D
from terrain_catalog import build_terrain
from visualization import (
    plot_bellman_policy_path,
    plot_bellman_reachability,
    plot_bellman_successor_debug,
)


DEFAULT_STAGE5_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_5_bellman_graph_geometry"
)
DEFAULT_GRAPH_BOUNDS = MapBounds(-8.0, 8.0, -4.0, 4.0)


@dataclass(frozen=True)
class ObstacleFreeTerrain:
    """Complete obstacle-free TerrainModel used only for Stage-5 diagnostics."""

    bounds: MapBounds
    ground_z: float = 0.0

    @property
    def maximum_height(self) -> float:
        return self.ground_z

    def ground_mesh(self) -> TriangleMesh:
        z = self.ground_z
        return TriangleMesh(
            vertices=np.array([
                [self.bounds.x_min, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_min, z],
                [self.bounds.x_max, self.bounds.y_max, z],
                [self.bounds.x_min, self.bounds.y_max, z],
            ]),
            triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        )

    def obstacle_mesh(self) -> TriangleMesh:
        return TriangleMesh(
            vertices=np.empty((0, 3), dtype=float),
            triangles=np.empty((0, 3), dtype=np.int64),
        )

    def surface_meshes(self) -> tuple[TriangleMesh, ...]:
        return self.ground_mesh(), self.obstacle_mesh()

    def contains_solid(
        self,
        point: np.ndarray,
        *,
        tolerance: float = 0.0,
    ) -> bool:
        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (3,) or not np.all(np.isfinite(point_array)):
            raise ValueError("point must contain three finite coordinates")
        return False

    def segment_intersects_solid(
        self,
        start: np.ndarray,
        end: np.ndarray,
        *,
        tolerance: float = 1.0e-9,
    ) -> bool:
        return False

    def first_ray_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        *,
        include_ground: bool = True,
    ) -> RayHit | None:
        if not include_ground:
            return None
        return TriangleRayTracer((self.ground_mesh(),)).first_hit(origin, direction)


def generate_stage5_artifacts(
    output_directory: Path = DEFAULT_STAGE5_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Build the canonical coarse graph and write four interactive views."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    grid = BellmanStateGrid(bounds=DEFAULT_GRAPH_BOUNDS)
    terrain = build_terrain("centered_cube", bounds=DEFAULT_GRAPH_BOUNDS)
    goal = Point3D(8.0, 0.0, 0.0)
    graph = build_bellman_graph(grid, terrain, goal)
    solution = solve_unit_cost_reachability(graph)
    independent = independent_reverse_reachability(graph)
    np.testing.assert_array_equal(solution.goal_reachable, independent)
    topological_order = graph.topological_sort()

    strict_altitude = all(
        edge.target_state.altitude_index < edge.source_state.altitude_index
        for state_id in graph.node_ids
        for edge in graph.adjacency[int(state_id)]
    )
    selected_state = BellmanState(
        x_index=6,  # x=-2, cube side face
        y_index=4,  # y=0
        altitude_index=20,
        heading_bin=3,  # northwest: valid escape edges plus one terrain rejection
    )
    transition_model = GlideTransitionModel(grid, terrain)
    selected_edges, selected_stats, selected_rejections = (
        transition_model.successors(selected_state, include_rejected=True)
    )

    reachable_figure = plot_bellman_reachability(
        graph,
        solution,
        show_reachable=True,
        show_unreachable=False,
    )
    unreachable_figure = plot_bellman_reachability(
        graph,
        solution,
        show_reachable=False,
        show_unreachable=True,
        maximum_heading_arrows=0,
    )
    successor_figure = plot_bellman_successor_debug(
        graph,
        transition_model,
        selected_state,
    )
    for name, figure in (
        ("centered_cube_goal_reachable_positions.html", reachable_figure),
        ("centered_cube_unreachable_positions.html", unreachable_figure),
        ("centered_cube_selected_state_successors.html", successor_figure),
    ):
        figure.write_html(
            output_directory / name,
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )

    empty_bounds = MapBounds(0.0, 3.0, -1.0, 1.0)
    empty_grid = BellmanStateGrid(
        bounds=empty_bounds,
        maximum_altitude_map=0.3,
    )
    empty_goal = Point3D(3.0, 0.0, 0.0)
    empty_graph = build_bellman_graph(
        empty_grid,
        ObstacleFreeTerrain(empty_bounds),
        empty_goal,
    )
    empty_solution = solve_unit_cost_reachability(empty_graph)
    empty_start_id = empty_grid.encode(BellmanState(0, 1, 3, 0))
    empty_path = empty_solution.backtrack(empty_start_id)
    policy_figure = plot_bellman_policy_path(
        empty_graph,
        empty_solution,
        empty_start_id,
    )
    policy_figure.write_html(
        output_directory / "obstacle_free_straight_goal_policy.html",
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
    )

    summary = {
        "configuration": {
            "bounds": [
                grid.bounds.x_min,
                grid.bounds.x_max,
                grid.bounds.y_min,
                grid.bounds.y_max,
            ],
            "horizontal_spacing_map": grid.horizontal_spacing_map,
            "altitude_range_map": [
                grid.minimum_altitude_map,
                grid.maximum_altitude_map,
            ],
            "altitude_spacing_map": grid.altitude_spacing_map,
            "heading_bin_count": grid.heading_bin_count,
            "goal_map": goal.as_array().tolist(),
            "goal_tolerance_m": 25.0,
            "unit_edge_cost": 1.0,
        },
        "graph_statistics": asdict(graph.statistics),
        "goal_reachable_state_count": solution.goal_reachable_state_count,
        "maximum_finite_unit_cost": solution.maximum_finite_value,
        "topological_state_count": len(topological_order),
        "strict_altitude_decrease": strict_altitude,
        "independent_reachability_equal": bool(np.array_equal(
            solution.goal_reachable,
            independent,
        )),
        "selected_state": {
            "state_id": grid.encode(selected_state),
            "position_map": grid.position_map(selected_state).tolist(),
            "heading_rad": grid.heading_rad(selected_state.heading_bin),
            "valid_successor_count": len(selected_edges),
            "statistics": asdict(selected_stats),
            "rejection_reasons": sorted({
                record.reason for record in selected_rejections
            }),
        },
        "obstacle_free_policy": {
            "start_state_id": empty_start_id,
            "unit_cost_value": empty_solution.value[empty_start_id],
            "path_state_ids": list(empty_path),
            "path_positions_map": [
                empty_grid.position_map(empty_grid.decode(state_id)).tolist()
                for state_id in empty_path
            ],
        },
    }
    (output_directory / "bellman_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "Stage 5 graph: "
        f"states={graph.statistics.state_count}, "
        f"edges={graph.statistics.valid_edge_count}, "
        f"terminal={graph.statistics.terminal_state_count}, "
        f"goal_reachable={solution.goal_reachable_state_count}, "
        f"max_unit_cost={solution.maximum_finite_value:g}"
    )
    return summary


if __name__ == "__main__":
    generated = generate_stage5_artifacts()
    print(
        "Generated Stage-5 diagnostics in "
        f"{DEFAULT_STAGE5_FIGURE_DIRECTORY}"
    )
