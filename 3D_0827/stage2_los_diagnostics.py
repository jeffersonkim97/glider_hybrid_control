"""Generate Stage-2 LOS abstraction diagnostics for canonical cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from los_explorer_gui import compute_los_case
from los_geometry import trace_terrain_tangent_contour
from stage0_baseline import CANONICAL_CASES


DEFAULT_STAGE2_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_2_los_abstraction"
)


def generate_stage2_artifacts(
    output_directory: Path = DEFAULT_STAGE2_FIGURE_DIRECTORY,
) -> dict[str, dict[str, Any]]:
    """Write one interactive LOS figure and regression record per case."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}

    for case in CANONICAL_CASES:
        result = compute_los_case(
            case.terrain_category,
            case.sensor_x,
            case.sensor_y,
        )
        terrain = result.terrain_map
        contour = result.tangent_contour
        legacy = trace_terrain_tangent_contour(
            terrain.surface_meshes(),
            result.mission_points.sensor,
            target_mesh_index=1,
            ground_height=terrain.ground_z,
            probe_grid_size=101,
            boundary_refinement_steps=24,
        )
        point_errors = np.linalg.norm(
            contour.tangent_points - legacy.tangent_points,
            axis=1,
        )
        first_hit_errors: list[float] = []
        first_hit_mesh_indices: list[int] = []
        for ray in contour.rays:
            hit = terrain.first_ray_hit(
                ray.origin.as_array(),
                ray.unit_direction,
                include_ground=True,
            )
            if hit is None:
                raise RuntimeError(f"{case.case_id}: retained tangent ray has no first hit")
            first_hit_mesh_indices.append(hit.mesh_index)
            first_hit_errors.append(float(np.linalg.norm(
                hit.point - ray.tangent_point.as_array(),
            )))

        summaries[case.case_id] = {
            "terrain": case.terrain_category,
            "sensor": [case.sensor_x, case.sensor_y, terrain.ground_z],
            "tangent_ray_count": len(contour.rays),
            "discarded_ground_candidate_count": (
                contour.discarded_ground_candidate_count
            ),
            "minimum_tangent_altitude": float(np.min(contour.tangent_points[:, 2])),
            "contour_closed": contour.closed,
            "los_surface_panel_count": result.los_surface.panel_count,
            "legacy_new_max_point_error": float(np.max(point_errors)),
            "retained_first_hit_max_point_error": float(np.max(first_hit_errors)),
            "retained_first_hit_mesh_indices": sorted(set(first_hit_mesh_indices)),
        }
        result.figure.update_layout(
            title={
                "text": (
                    f"Stage 2 LOSModel: {case.case_id} / {case.terrain_category}"
                    "<br><sup>terrain, sensor, 10 sampled rays, all tangent points, "
                    "and ruled surface</sup>"
                ),
                "x": 0.5,
            }
        )
        result.figure.write_html(
            output_directory / f"{case.case_id}_{case.terrain_category}_los_model.html",
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )

    (output_directory / "los_abstraction_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    generated = generate_stage2_artifacts()
    print(
        f"Generated {len(generated)} interactive LOS diagnostics in "
        f"{DEFAULT_STAGE2_FIGURE_DIRECTORY}"
    )
