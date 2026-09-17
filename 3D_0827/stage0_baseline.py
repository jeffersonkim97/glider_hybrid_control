"""Deterministic Stage-0 baseline capture for the 3D prototype.

This module records existing behavior only.  It deliberately does not alter
terrain, LOS, energy, Bellman, detection, or Defender logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from los_explorer_gui import EnergyExplorerResult, compute_energy_case
from visualization import plot_terrain_map


@dataclass(frozen=True)
class BaselineCase:
    """One canonical terrain/sensor configuration from the implementation plan."""

    case_id: str
    terrain_category: str
    sensor_x: float
    sensor_y: float


CANONICAL_CASES = (
    BaselineCase("case_a", "centered_cube", 5.0, 0.0),
    BaselineCase("case_b", "centered_cube", 8.0, 0.0),
    BaselineCase("case_c", "offset_cube_left", 5.0, 0.0),
    BaselineCase("case_d", "offset_cube_right", 5.0, 0.0),
    BaselineCase("case_e", "stepped_pyramid", 5.0, 0.0),
)

DEFAULT_STAGE0_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_0_baseline"
)


def compute_baseline_case(case: BaselineCase) -> EnergyExplorerResult:
    """Run one canonical case at the current default regression resolution."""
    return compute_energy_case(
        case.terrain_category,
        case.sensor_x,
        case.sensor_y,
    )


def summarize_baseline_case(
    case: BaselineCase,
    result: EnergyExplorerResult,
) -> dict[str, Any]:
    """Return the deterministic numerical quantities required by Stage 0."""
    terrain = result.los_result.terrain_map
    contour = result.los_result.tangent_contour
    surface = result.energy_surface
    reachable = np.asarray(surface.vertex_reachable, dtype=bool)
    powered_feasible = np.asarray(surface.vertex_powered_feasible, dtype=bool)
    energy_margin = np.asarray(surface.vertex_energy_margin_j, dtype=float)

    reachable_indices = np.flatnonzero(reachable & powered_feasible)
    unreachable_indices = np.flatnonzero(~reachable & powered_feasible)
    if reachable_indices.size == 0 or unreachable_indices.size == 0:
        raise RuntimeError(
            f"{case.case_id} must contain representative reachable and "
            "unreachable powered-feasible vertices"
        )
    reachable_index = int(
        reachable_indices[np.argmax(energy_margin[reachable_indices])]
    )
    unreachable_index = int(
        unreachable_indices[np.argmin(energy_margin[unreachable_indices])]
    )

    return {
        "terrain": case.terrain_category,
        "sensor": [case.sensor_x, case.sensor_y, terrain.ground_z],
        "bounds": [
            terrain.bounds.x_min,
            terrain.bounds.x_max,
            terrain.bounds.y_min,
            terrain.bounds.y_max,
        ],
        "maximum_height": terrain.maximum_height,
        "obstacle_box_count": len(terrain.obstacle_boxes()),
        "obstacle_triangle_count": len(terrain.obstacle_mesh().triangles),
        "tangent_ray_count": len(contour.rays),
        "discarded_ground_candidate_count": (
            contour.discarded_ground_candidate_count
        ),
        "contour_closed": contour.closed,
        "minimum_tangent_altitude": float(np.min(contour.tangent_points[:, 2])),
        "los_surface_panel_count": result.los_result.los_surface.panel_count,
        "reachable_face_count": surface.reachable_face_count,
        "unreachable_face_count": surface.unreachable_face_count,
        "reachable_vertex_count": int(np.count_nonzero(reachable)),
        "unreachable_vertex_count": int(np.count_nonzero(~reachable)),
        "powered_infeasible_vertex_count": (
            surface.powered_infeasible_vertex_count
        ),
        "representative_reachable_point": (
            surface.vertices[reachable_index].tolist()
        ),
        "representative_unreachable_point": (
            surface.vertices[unreachable_index].tolist()
        ),
        "representative_reachable_margin_j": float(
            energy_margin[reachable_index]
        ),
        "representative_unreachable_margin_j": float(
            energy_margin[unreachable_index]
        ),
    }


def generate_stage0_artifacts(
    output_directory: Path = DEFAULT_STAGE0_FIGURE_DIRECTORY,
) -> dict[str, dict[str, Any]]:
    """Save separate interactive terrain, LOS, and energy figures per case."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}

    for case in CANONICAL_CASES:
        result = compute_baseline_case(case)
        summaries[case.case_id] = summarize_baseline_case(case, result)

        terrain_figure = plot_terrain_map(result.los_result.terrain_map)
        terrain_figure.update_layout(
            title=(
                f"Stage 0 terrain baseline: {case.case_id} / "
                f"{case.terrain_category}"
            )
        )
        figures = {
            "terrain": terrain_figure,
            "los_tangent_surface": result.los_result.figure,
            "energy_reachability": result.figure,
        }
        for figure_name, figure in figures.items():
            filename = (
                f"{case.case_id}_{case.terrain_category}_{figure_name}.html"
            )
            figure.write_html(
                output_directory / filename,
                include_plotlyjs="directory",
                full_html=True,
                auto_open=False,
            )

    summary_path = output_directory / "baseline_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    generated = generate_stage0_artifacts()
    print(
        f"Generated {3 * len(generated)} interactive figures in "
        f"{DEFAULT_STAGE0_FIGURE_DIRECTORY}"
    )
