"""Generate Stage-1 Plotly diagnostics through the generic terrain API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from map_geometry import TerrainModel
from terrain_catalog import TERRAIN_LABELS, build_terrain
from visualization import plot_terrain_map


DEFAULT_STAGE1_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_1_terrain_abstraction"
)


def _diagnostic_segments(terrain: TerrainModel) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    obstacle = terrain.obstacle_mesh()
    center = np.mean(obstacle.vertices, axis=0)
    center[2] = terrain.ground_z + 0.5 * (
        terrain.maximum_height - terrain.ground_z
    )
    x0 = terrain.bounds.x_min + 1.0
    x1 = terrain.bounds.x_max - 1.0
    return (
        (
            "through terrain center",
            np.array([x0, center[1], center[2]]),
            np.array([x1, center[1], center[2]]),
        ),
        (
            "begins inside terrain",
            center.copy(),
            np.array([x1, center[1], center[2]]),
        ),
        (
            "outside terrain",
            np.array([x0, terrain.bounds.y_max - 0.5, center[2]]),
            np.array([x1, terrain.bounds.y_max - 0.5, center[2]]),
        ),
        (
            "top-surface tangent",
            np.array([x0, center[1], terrain.maximum_height]),
            np.array([x1, center[1], terrain.maximum_height]),
        ),
    )


def _diagnostic_rays(terrain: TerrainModel) -> tuple[tuple[str, np.ndarray, np.ndarray, bool], ...]:
    obstacle = terrain.obstacle_mesh()
    center = np.mean(obstacle.vertices, axis=0)
    center[2] = terrain.ground_z + 0.5 * (
        terrain.maximum_height - terrain.ground_z
    )
    obstacle_origin = np.array([
        terrain.bounds.x_max - 1.0, center[1], center[2],
    ])
    ground_origin = np.array([
        terrain.bounds.x_max - 2.0,
        terrain.bounds.y_min + 2.0,
        terrain.maximum_height + 1.0,
    ])
    maximum_z = float(np.max(obstacle.vertices[:, 2]))
    top_vertices = obstacle.vertices[
        np.isclose(obstacle.vertices[:, 2], maximum_z)
    ]
    edge_target = top_vertices[
        np.argmax(top_vertices[:, 0] + top_vertices[:, 1])
    ]
    edge_origin = edge_target + np.array([3.0, 0.0, 0.0])
    return (
        ("obstacle first hit", obstacle_origin, center - obstacle_origin, True),
        ("ground first hit", ground_origin, np.array([0.0, 0.0, -1.0]), True),
        ("terrain-edge grazing hit", edge_origin, edge_target - edge_origin, False),
    )


def build_terrain_diagnostic(
    terrain_name: str,
) -> tuple[go.Figure, dict[str, Any]]:
    """Build one terrain figure and its machine-readable query results."""
    terrain = build_terrain(terrain_name)
    figure = plot_terrain_map(terrain)
    segment_records: list[dict[str, Any]] = []
    for name, start, end in _diagnostic_segments(terrain):
        collides = terrain.segment_intersects_solid(start, end)
        color = "#d62728" if collides else "#2ca02c"
        figure.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+markers",
            line={"color": color, "width": 7},
            marker={"color": color, "size": 3},
            name=f"{'collision' if collides else 'clear'}: {name}",
        ))
        segment_records.append({
            "name": name,
            "start": start.tolist(),
            "end": end.tolist(),
            "intersects_solid": collides,
        })

    ray_records: list[dict[str, Any]] = []
    for name, origin, direction, include_ground in _diagnostic_rays(terrain):
        hit = terrain.first_ray_hit(
            origin, direction, include_ground=include_ground,
        )
        if hit is None:
            ray_end = origin + direction
            hit_record = None
        else:
            ray_end = hit.point
            hit_record = {
                "distance": hit.distance,
                "point": hit.point.tolist(),
                "mesh_index": hit.mesh_index,
                "triangle_index": hit.triangle_index,
            }
            hit_color = "#9467bd" if include_ground and hit.mesh_index == 0 else "#111111"
            figure.add_trace(go.Scatter3d(
                x=[hit.point[0]], y=[hit.point[1]], z=[hit.point[2]],
                mode="markers",
                marker={"color": hit_color, "size": 7, "symbol": "diamond"},
                name=f"hit: {name}",
            ))
        figure.add_trace(go.Scatter3d(
            x=[origin[0], ray_end[0]],
            y=[origin[1], ray_end[1]],
            z=[origin[2], ray_end[2]],
            mode="lines",
            line={"color": "#111111", "width": 4, "dash": "dash"},
            name=f"ray: {name}",
        ))
        ray_records.append({
            "name": name,
            "origin": origin.tolist(),
            "direction": direction.tolist(),
            "include_ground": include_ground,
            "hit": hit_record,
        })

    figure.update_layout(
        title={
            "text": (
                f"Stage 1 TerrainModel diagnostics: {TERRAIN_LABELS[terrain_name]}"
                "<br><sup>red=solid penetration; green=clear/tangent; diamonds=first ray hits</sup>"
            ),
            "x": 0.5,
        }
    )
    return figure, {
        "terrain": terrain_name,
        "segments": segment_records,
        "rays": ray_records,
    }


def generate_stage1_artifacts(
    output_directory: Path = DEFAULT_STAGE1_FIGURE_DIRECTORY,
) -> dict[str, dict[str, Any]]:
    """Write one independent interactive diagnostic per current terrain."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for terrain_name in TERRAIN_LABELS:
        figure, summary = build_terrain_diagnostic(terrain_name)
        summaries[terrain_name] = summary
        figure.write_html(
            output_directory / f"{terrain_name}_terrain_model_diagnostic.html",
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )
    (output_directory / "terrain_query_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8",
    )
    return summaries


if __name__ == "__main__":
    generated = generate_stage1_artifacts()
    print(
        f"Generated {len(generated)} interactive terrain diagnostics in "
        f"{DEFAULT_STAGE1_FIGURE_DIRECTORY}"
    )
