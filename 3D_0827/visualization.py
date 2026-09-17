"""Interactive Plotly visualization for the simplified 3D LOS study."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from map_geometry import TerrainModel, TriangleMesh
from los_geometry import LOSTangentSurface, TangentContour, VisualizationRaySet
from reachability_surface import EnergyReachabilitySurface
from scenario import MissionPoints, Point3D

if TYPE_CHECKING:
    from bellman_geometry import GlideTransitionModel
    from bellman_graph import BellmanGraph, UnitCostReachability
    from bellman_solver import TimeOptimalRun
    from bellman_state import BellmanState
    from candidate_energy import CandidateEnergyEvaluation
    from edge_hazard import EdgeHazardResult
    from switching_candidates import SwitchingCandidate
    from mission_response import SingleCandidateMissionResponse
    from attacker_best_response import AttackerBestResponseRun
    from trajectory_validation import TrajectoryReplayAudit, TrajectoryReplaySnapshot
    from stackelberg_solver import FiniteStackelbergRun


def _mesh_trace(
    mesh: TriangleMesh,
    *,
    name: str,
    color: str,
    opacity: float,
) -> go.Mesh3d:
    vertices = mesh.vertices
    triangles = mesh.triangles
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
        hovertemplate=(
            f"{name}<br>x=%{{x:.2f}} map unit<br>y=%{{y:.2f}} map unit"
            "<br>z=%{z:.2f} map unit<extra></extra>"
        ),
    )


def _crease_edges(
    mesh: TriangleMesh,
    *,
    coplanar_tolerance: float = 1.0e-10,
) -> tuple[tuple[int, int], ...]:
    """Return boundary/crease edges while suppressing triangulation diagonals."""
    triangle_vertices = mesh.vertices[mesh.triangles]
    normals = np.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]

    adjacency: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(mesh.triangles):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            adjacency.setdefault(edge, []).append(triangle_index)

    crease_edges: list[tuple[int, int]] = []
    for edge, triangle_indices in adjacency.items():
        if len(triangle_indices) != 2:
            crease_edges.append(edge)
            continue
        first_normal = normals[triangle_indices[0]]
        second_normal = normals[triangle_indices[1]]
        if abs(float(np.dot(first_normal, second_normal))) < 1.0 - coplanar_tolerance:
            crease_edges.append(edge)
    return tuple(sorted(crease_edges))


def _obstacle_edge_trace(terrain_map: TerrainModel) -> go.Scatter3d:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []

    mesh = terrain_map.obstacle_mesh()
    for start_index, end_index in _crease_edges(mesh):
        start = mesh.vertices[start_index]
        end = mesh.vertices[end_index]
        x_values.extend((start[0], end[0], None))
        y_values.extend((start[1], end[1], None))
        z_values.extend((start[2], end[2], None))

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="lines",
        name="Terrain edges",
        line={"color": "#17202a", "width": 5},
        hoverinfo="skip",
        showlegend=False,
    )


def plot_terrain_map(terrain_map: TerrainModel) -> go.Figure:
    """Build an interactive ground-and-obstacle terrain figure."""
    ground_mesh, obstacle_mesh = terrain_map.surface_meshes()
    figure = go.Figure(
        data=[
            _mesh_trace(
                ground_mesh,
                name="Ground plane",
                color="#d9e2ec",
                opacity=0.72,
            ),
            _mesh_trace(
                obstacle_mesh,
                name="Obstacle terrain",
                color="#4c78a8",
                opacity=0.90,
            ),
            _obstacle_edge_trace(terrain_map),
        ]
    )

    z_span = max((terrain_map.maximum_height - terrain_map.ground_z) * 1.5, 1.0)
    figure.update_layout(
        title={"text": "Stage 1: Empty Plane with One Cube", "x": 0.5},
        template="plotly_white",
        width=950,
        height=700,
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        legend={"x": 0.02, "y": 0.98},
        scene={
            "xaxis": {
                "title": "x [map unit]",
                "range": [terrain_map.bounds.x_min, terrain_map.bounds.x_max],
                "showspikes": False,
            },
            "yaxis": {
                "title": "y [map unit]",
                "range": [terrain_map.bounds.y_min, terrain_map.bounds.y_max],
                "showspikes": False,
            },
            "zaxis": {
                "title": "z [map unit]",
                "range": [terrain_map.ground_z, terrain_map.ground_z + z_span],
                "showspikes": False,
            },
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.55, "y": -1.65, "z": 1.15},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
        },
    )
    return figure


def _point_trace(
    point: Point3D,
    *,
    name: str,
    color: str,
    symbol: str,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=[point.x],
        y=[point.y],
        z=[point.z],
        mode="markers+text",
        name=name,
        text=[name],
        textposition="top center",
        marker={
            "size": 8,
            "color": color,
            "symbol": symbol,
            "line": {"color": "#17202a", "width": 1.5},
        },
        hovertemplate=(
            f"{name}<br>x=%{{x:.2f}} map unit<br>y=%{{y:.2f}} map unit"
            "<br>z=%{z:.2f} map unit<extra></extra>"
        ),
    )


def plot_mission_scenario(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
) -> go.Figure:
    """Plot the map together with the Stage-2 sensor, start, and goal points."""
    mission_points.validate_against(terrain_map)
    figure = plot_terrain_map(terrain_map)

    sensor = mission_points.sensor
    if sensor.z > terrain_map.ground_z:
        figure.add_trace(
            go.Scatter3d(
                x=[sensor.x, sensor.x],
                y=[sensor.y, sensor.y],
                z=[terrain_map.ground_z, sensor.z],
                mode="lines",
                name="Sensor mount",
                line={"color": "#7a5195", "width": 5, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    figure.add_traces(
        [
            _point_trace(
                mission_points.sensor,
                name="Sensor",
                color="#7a5195",
                symbol="diamond",
            ),
            _point_trace(
                mission_points.start,
                name="Start",
                color="#1b9e77",
                symbol="circle",
            ),
            _point_trace(
                mission_points.goal,
                name="Goal",
                color="#e6a700",
                symbol="square",
            ),
        ]
    )
    figure.update_layout(title={"text": "Stage 2: Sensor, Start, and Goal", "x": 0.5})
    return figure


def plot_tangent_los_rays(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    tangent_contour: TangentContour,
    visualization_rays: VisualizationRaySet,
) -> go.Figure:
    """Plot the dense tangent contour with only the selected LOS overlays."""
    mission_points.validate_against(terrain_map)
    if tangent_contour.origin != mission_points.sensor:
        raise ValueError("dense tangent-contour origin must equal the scenario sensor")
    if visualization_rays.origin != mission_points.sensor:
        raise ValueError("visualization-ray origin must equal the scenario sensor")

    figure = plot_mission_scenario(terrain_map, mission_points)
    ray_x: list[float | None] = []
    ray_y: list[float | None] = []
    ray_z: list[float | None] = []
    for ray in visualization_rays.rays:
        ray_x.extend((ray.origin.x, ray.tangent_point.x, None))
        ray_y.extend((ray.origin.y, ray.tangent_point.y, None))
        ray_z.extend((ray.origin.z, ray.tangent_point.z, None))

    figure.add_trace(
        go.Scatter3d(
            x=ray_x,
            y=ray_y,
            z=ray_z,
            mode="lines",
            name=f"Displayed tangent LOS samples ({len(visualization_rays.rays)})",
            line={"color": "#00a6d6", "width": 5},
            hoverinfo="skip",
        )
    )
    tangent_points = visualization_rays.tangent_points
    figure.add_trace(
        go.Scatter3d(
            x=tangent_points[:, 0],
            y=tangent_points[:, 1],
            z=tangent_points[:, 2],
            mode="markers",
            name="Tangent points",
            marker={
                "size": 4,
                "color": "#d81b60",
                "symbol": "circle",
                "line": {"color": "#17202a", "width": 1.0},
            },
            hovertemplate=(
                "Tangent point<br>x=%{x:.3f} map unit<br>y=%{y:.3f} map unit"
                "<br>z=%{z:.3f} map unit<extra></extra>"
            ),
        )
    )
    dense_contour_points = tangent_contour.tangent_points
    contour_points = (
        np.vstack((dense_contour_points, dense_contour_points[0]))
        if tangent_contour.closed
        else dense_contour_points
    )
    figure.add_trace(
        go.Scatter3d(
            x=contour_points[:, 0],
            y=contour_points[:, 1],
            z=contour_points[:, 2],
            mode="lines+markers",
            name="Ray-traced upper tangent horizon",
            line={"color": "#d81b60", "width": 4},
            marker={"color": "#d81b60", "size": 2},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title={"text": "Stage 3: Ground-Trimmed Terrain-Tangent LOS Rays", "x": 0.5}
    )
    return figure


def plot_los_tangent_surface(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
) -> go.Figure:
    """Plot the Stage-4 infinite-ray extensions and clipped tangent surface."""
    tangent_contour = los_surface.tangent_contour
    figure = plot_tangent_los_rays(
        terrain_map,
        mission_points,
        tangent_contour,
        visualization_rays,
    )
    tangent_points = visualization_rays.tangent_points
    sample_far_points = np.vstack(
        [
            ray.point_at(los_surface.display_extension_factor)
            for ray in visualization_rays.rays
        ]
    )
    dense_far_points = los_surface.section_points(los_surface.display_extension_factor)

    extension_x: list[float | None] = []
    extension_y: list[float | None] = []
    extension_z: list[float | None] = []
    for tangent_point, far_point in zip(tangent_points, sample_far_points, strict=True):
        extension_x.extend((tangent_point[0], far_point[0], None))
        extension_y.extend((tangent_point[1], far_point[1], None))
        extension_z.extend((tangent_point[2], far_point[2], None))

    figure.add_trace(
        _mesh_trace(
            los_surface.display_mesh(),
            name=f"LOS tangent surface ({los_surface.panel_count} panels)",
            color="#58c4dd",
            opacity=0.28,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=extension_x,
            y=extension_y,
            z=extension_z,
            mode="lines",
            name="Extended LOS rays",
            line={"color": "#0077b6", "width": 4, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    display_clip = (
        np.vstack((dense_far_points, dense_far_points[0]))
        if tangent_contour.closed
        else dense_far_points
    )
    figure.add_trace(
        go.Scatter3d(
            x=display_clip[:, 0],
            y=display_clip[:, 1],
            z=display_clip[:, 2],
            mode="lines+markers",
            name=f"Display clip (scale={los_surface.display_extension_factor:g})",
            line={"color": "#005f73", "width": 5},
            marker={"size": 3, "color": "#005f73"},
            hovertemplate=(
                "Display clip<br>x=%{x:.3f} map unit<br>y=%{y:.3f} map unit"
                "<br>z=%{z:.3f} map unit<extra></extra>"
            ),
        )
    )

    all_x = np.concatenate(
        ([terrain_map.bounds.x_min, terrain_map.bounds.x_max], dense_far_points[:, 0])
    )
    all_y = np.concatenate(
        ([terrain_map.bounds.y_min, terrain_map.bounds.y_max], dense_far_points[:, 1])
    )
    all_z = np.concatenate(
        ([terrain_map.ground_z, terrain_map.maximum_height], dense_far_points[:, 2])
    )
    x_padding = max(0.05 * float(np.ptp(all_x)), 0.25)
    y_padding = max(0.05 * float(np.ptp(all_y)), 0.25)
    z_padding = max(0.05 * float(np.ptp(all_z)), 0.25)
    figure.update_layout(
        title={"text": "Stage 4: Continuous Unbounded LOS Tangent Surface", "x": 0.5},
        scene={
            "xaxis": {
                "range": [float(np.min(all_x) - x_padding), float(np.max(all_x) + x_padding)]
            },
            "yaxis": {
                "range": [float(np.min(all_y) - y_padding), float(np.max(all_y) + y_padding)]
            },
            "zaxis": {
                "range": [float(np.min(all_z)), float(np.max(all_z) + z_padding)]
            },
        },
    )
    return figure


def plot_single_candidate_mission(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
    response: "SingleCandidateMissionResponse",
) -> go.Figure:
    """Overlay the three Stage-8 trajectory phases on the LOS geometry."""
    if response.selected_option is None or response.replay is None:
        raise ValueError("a feasible selected mission response is required")
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    option = response.selected_option
    start = mission_points.start.as_array()
    switching = response.connections.switching_state.position_map
    first_lattice = option.connection.target_position_map
    glide_positions = [first_lattice]
    glide_positions.extend(
        response.bellman_solution.graph.grid.position_map(edge.target_state)
        for edge in option.glide_edges
    )
    glide = np.asarray(glide_positions)

    figure.add_trace(go.Scatter3d(
        x=[start[0], switching[0]],
        y=[start[1], switching[1]],
        z=[start[2], switching[2]],
        mode="lines",
        name="Powered phase",
        line={"color": "#f59e0b", "width": 9},
        hovertemplate="Powered phase<extra></extra>",
    ))
    figure.add_trace(go.Scatter3d(
        x=[switching[0], first_lattice[0]],
        y=[switching[1], first_lattice[1]],
        z=[switching[2], first_lattice[2]],
        mode="lines",
        name="Virtual connection",
        line={"color": "#d946ef", "width": 9, "dash": "dot"},
        hovertemplate="Virtual connection<extra></extra>",
    ))
    figure.add_trace(go.Scatter3d(
        x=glide[:, 0],
        y=glide[:, 1],
        z=glide[:, 2],
        mode="lines+markers",
        name="Bellman glide path",
        line={"color": "#16a34a", "width": 8},
        marker={"color": "#166534", "size": 4},
        hovertemplate=(
            "Glide lattice<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[switching[0]],
        y=[switching[1]],
        z=[switching[2]],
        mode="markers",
        name=f"Fixed switch candidate {response.candidate_id}",
        marker={"color": "#111827", "size": 9, "symbol": "diamond"},
        hovertemplate=(
            f"Candidate {response.candidate_id}<br>"
            "x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[first_lattice[0]],
        y=[first_lattice[1]],
        z=[first_lattice[2]],
        mode="markers",
        name="First Bellman state",
        marker={"color": "#d946ef", "size": 8, "symbol": "square"},
        hovertemplate=(
            "First Bellman state<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<extra></extra>"
        ),
    ))
    objective = option.objective
    figure.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#374151",
        text=(
            f"<b>Fixed candidate {response.candidate_id}</b><br>"
            f"powered feasible: {option.connection.powered_feasible}<br>"
            f"virtual feasible: {option.connection.feasible}<br>"
            f"Bellman feasible: "
            f"{response.bellman_solution.goal_reachable[option.connection.target_state_id]}<br>"
            f"projection error: {option.connection.projection_error_m:.3f} m<br>"
            f"total time: {objective.mission_time_s:.3f} s<br>"
            f"total hazard: {objective.mission_hazard:.6f}<br>"
            f"PoD: {objective.mission_pod:.6f}<br>"
            f"objective: {objective.objective_value:.6f}"
        ),
    )
    figure.update_layout(
        title={
            "text": (
                "Stage 8: One Fixed Switching Candidate → Virtual Edge → "
                "Bellman Response"
            ),
            "x": 0.5,
        },
        legend={"groupclick": "toggleitem"},
    )
    return figure


def plot_attacker_best_response(
    run: "AttackerBestResponseRun",
) -> go.Figure:
    """Show all candidate outcomes and the exact selected Stage-9 trajectory."""
    figure = plot_los_tangent_surface(
        run.terrain,
        run.mission_points,
        run.los_surface,
        run.visualization_rays,
    )
    category_style = {
        "powered_infeasible": ("Powered infeasible", "#6b7280", "circle"),
        "energy_infeasible": ("Energy infeasible", "#dc2626", "circle"),
        "virtual_infeasible": ("Virtual infeasible", "#f97316", "circle"),
        "goal_unreachable": ("Goal unreachable", "#7c3aed", "circle"),
        "path_energy_infeasible": ("Path-energy infeasible", "#db2777", "circle"),
        "feasible": ("Feasible", "#16a34a", "circle"),
    }
    for category, (label, color, symbol) in category_style.items():
        results = tuple(
            result for result in run.candidate_results
            if result.status_category == category
        )
        if not results:
            continue
        positions = np.vstack([result.position_map for result in results])
        customdata = np.asarray([
            [
                result.candidate_id,
                "" if result.objective is None else f"{result.objective:.9f}",
                result.infeasibility_reason or "",
            ]
            for result in results
        ], dtype=object)
        figure.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers",
            name=f"{label} ({len(results)})",
            marker={"color": color, "size": 5, "symbol": symbol, "opacity": 0.86},
            customdata=customdata,
            hovertemplate=(
                f"{label}<br>candidate=%{{customdata[0]}}"
                "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                "<br>objective=%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
            ),
        ))

    selected = run.selected_result
    if selected is not None and selected.selected_option is not None:
        option = selected.selected_option
        start = run.mission_points.start.as_array()
        switch = selected.position_map
        first = option.connection.target_position_map
        glide = np.vstack((
            first,
            *(
                run.graph.grid.position_map(edge.target_state)
                for edge in option.glide_edges
            ),
        ))
        figure.add_trace(go.Scatter3d(
            x=[start[0], switch[0]], y=[start[1], switch[1]],
            z=[start[2], switch[2]], mode="lines", name="Selected powered",
            line={"color": "#facc15", "width": 10}, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter3d(
            x=[switch[0], first[0]], y=[switch[1], first[1]],
            z=[switch[2], first[2]], mode="lines", name="Selected virtual",
            line={"color": "#d946ef", "width": 10, "dash": "dot"},
            hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter3d(
            x=glide[:, 0], y=glide[:, 1], z=glide[:, 2],
            mode="lines+markers", name="Selected Bellman glide",
            line={"color": "#052e16", "width": 9},
            marker={"color": "#052e16", "size": 4}, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter3d(
            x=[switch[0]], y=[switch[1]], z=[switch[2]],
            mode="markers", name=f"Selected candidate {selected.candidate_id}",
            marker={
                "color": "#00ffff", "size": 12, "symbol": "diamond",
                "line": {"color": "#111827", "width": 3},
            },
            hovertemplate=(
                f"<b>Selected candidate {selected.candidate_id}</b><br>"
                f"objective={selected.objective:.9f}<br>"
                f"time={selected.mission_time_s:.3f} s<br>"
                f"hazard={selected.cumulative_hazard:.9f}<br>"
                f"PoD={selected.detection_probability:.9f}<extra></extra>"
            ),
        ))
    figure.update_layout(title={
        "text": (
            "Stage 9: Exhaustive Fixed-Defender Attacker Best Response"
            f"<br><sup>{run.metrics.number_of_candidates} candidates; "
            f"selected={None if selected is None else selected.candidate_id}</sup>"
        ),
        "x": 0.5,
    })
    return figure


def plot_candidate_objective_diagnostics(
    run: "AttackerBestResponseRun",
) -> go.Figure:
    """Plot exhaustive feasible-candidate objective, time, and hazard values."""
    feasible = tuple(result for result in run.candidate_results if result.feasible)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Attacker objective", "Mission time", "Cumulative hazard"),
    )
    candidate_ids = [result.candidate_id for result in feasible]
    selected_id = None if run.selected_result is None else run.selected_result.candidate_id
    colors = ["#00bcd4" if value == selected_id else "#16a34a" for value in candidate_ids]
    for row, values, name in (
        (1, [result.objective for result in feasible], "objective"),
        (2, [result.mission_time_s for result in feasible], "time [s]"),
        (3, [result.cumulative_hazard for result in feasible], "hazard"),
    ):
        figure.add_trace(go.Scatter(
            x=candidate_ids,
            y=values,
            mode="markers+lines",
            name=name,
            marker={"color": colors, "size": 9},
            line={"color": "#94a3b8", "width": 1},
            hovertemplate=(
                "candidate=%{x}<br>value=%{y:.9f}<extra></extra>"
            ),
        ), row=row, col=1)
    figure.update_xaxes(title_text="candidate ID", row=3, col=1)
    figure.update_layout(
        title={"text": "Stage 9 Exhaustive Candidate Diagnostics", "x": 0.5},
        height=850,
        showlegend=False,
    )
    return figure


def plot_trajectory_validation(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    snapshot: "TrajectoryReplaySnapshot",
    audit: "TrajectoryReplayAudit",
    *,
    goal_tolerance_map: float,
) -> go.Figure:
    """Show segment-level independent replay status and first failure location."""
    figure = plot_mission_scenario(terrain_map, mission_points)
    phase_style = {
        "powered": ("#f59e0b", "solid", 9),
        "virtual": ("#d946ef", "dot", 9),
        "glide": ("#16a34a", "solid", 7),
    }
    for segment in audit.segments:
        locally_valid = bool(
            segment.terrain_clear and segment.turn_valid and segment.altitude_valid
        )
        base_color, dash, width = phase_style[segment.phase]
        color = base_color if locally_valid else "#dc2626"
        start = segment.start_position_map
        end = segment.end_position_map
        figure.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines+markers",
            name=(
                f"{segment.phase} {segment.segment_index} "
                f"({'PASS' if locally_valid else 'FAIL'})"
            ),
            line={"color": color, "width": width, "dash": dash},
            marker={"color": color, "size": 3},
            customdata=[[segment.duration_s, segment.hazard]] * 2,
            hovertemplate=(
                f"{segment.phase} segment {segment.segment_index}<br>"
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                "<br>duration=%{customdata[0]:.6f} s"
                "<br>hazard=%{customdata[1]:.9f}<extra></extra>"
            ),
        ))
    switch = snapshot.switching_position_map
    figure.add_trace(go.Scatter3d(
        x=[switch[0]], y=[switch[1]], z=[switch[2]],
        mode="markers", name=f"Switch candidate {snapshot.candidate_id}",
        marker={"size": 10, "color": "#111827", "symbol": "diamond"},
    ))

    goal = mission_points.goal.as_array()
    azimuth = np.linspace(0.0, 2.0 * np.pi, 36)
    polar = np.linspace(0.0, np.pi, 19)
    sphere_x = goal[0] + goal_tolerance_map * np.outer(
        np.cos(azimuth), np.sin(polar),
    )
    sphere_y = goal[1] + goal_tolerance_map * np.outer(
        np.sin(azimuth), np.sin(polar),
    )
    sphere_z = goal[2] + goal_tolerance_map * np.outer(
        np.ones_like(azimuth), np.cos(polar),
    )
    figure.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        name="25 m goal tolerance sphere",
        showscale=False, opacity=0.20,
        colorscale=[[0.0, "#22c55e"], [1.0, "#22c55e"]],
        hoverinfo="skip",
    ))
    if audit.first_failure_position_map is not None:
        failure = audit.first_failure_position_map
        figure.add_trace(go.Scatter3d(
            x=[failure[0]], y=[failure[1]], z=[failure[2]],
            mode="markers+text", name="First validation failure",
            text=["FIRST FAILURE"], textposition="top center",
            marker={
                "size": 12, "color": "#ef4444", "symbol": "x",
                "line": {"color": "#7f1d1d", "width": 2},
            },
        ))
    report = audit.report
    figure.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        xanchor="left", yanchor="top", align="left", showarrow=False,
        bgcolor="rgba(255,255,255,0.90)", bordercolor="#374151",
        text=(
            f"<b>Independent replay: {'PASS' if report.passed else 'FAIL'}</b><br>"
            f"terrain: {report.terrain_clear}; LOS phase: {report.los_phase_valid}<br>"
            f"turn: {report.turn_constraints_valid}; energy: {report.energy_valid}<br>"
            f"time error: {report.time_error_s:.3e} s<br>"
            f"hazard error: {report.hazard_error:.3e}<br>"
            f"goal distance: {audit.goal_distance_m:.6f} m"
        ),
    )
    figure.update_layout(title={
        "text": (
            "Stage 10: Independent Continuous Trajectory Replay — "
            f"{'PASS' if report.passed else 'FAIL'}"
        ),
        "x": 0.5,
    })
    return figure


def plot_switching_candidates(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
    candidates: tuple["SwitchingCandidate", ...],
) -> go.Figure:
    """Overlay pure geometric switching candidates on the LOS surface."""
    if not candidates:
        raise ValueError("at least one switching candidate is required")
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    positions = np.vstack([candidate.position_map for candidate in candidates])
    customdata = np.column_stack((
        np.fromiter((candidate.candidate_id for candidate in candidates), dtype=int),
        np.fromiter((candidate.contour_fraction for candidate in candidates), dtype=float),
        np.fromiter((candidate.radial_scale for candidate in candidates), dtype=float),
        np.fromiter((candidate.surface_residual for candidate in candidates), dtype=float),
    ))
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        name=f"Switching candidates ({len(candidates)})",
        marker={
            "size": 5,
            "color": customdata[:, 2],
            "colorscale": "Viridis",
            "opacity": 0.90,
            "line": {"color": "#17202a", "width": 0.5},
            "colorbar": {"title": "radial scale"},
        },
        customdata=customdata,
        hovertemplate=(
            "Candidate %{customdata[0]:.0f}"
            "<br>x=%{x:.3f} map unit<br>y=%{y:.3f} map unit"
            "<br>z=%{z:.3f} map unit"
            "<br>contour fraction=%{customdata[1]:.6f}"
            "<br>radial scale=%{customdata[2]:.3f}"
            "<br>surface residual=%{customdata[3]:.3e}<extra></extra>"
        ),
    ))
    figure.update_layout(title={
        "text": (
            "Stage 3: Pure Geometric Switching Candidates"
            "<br><sup>Marker color denotes radial scale; no energy classification</sup>"
        ),
        "x": 0.5,
    })
    return figure


def plot_single_candidate_energy(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
    evaluation: "CandidateEnergyEvaluation",
) -> go.Figure:
    """Show one independently certified start-to-switch powered segment."""
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    state = evaluation.switching_state
    start = mission_points.start.as_array()
    switch = evaluation.candidate.position_map
    segment_color = "#1b9e77" if evaluation.powered_feasible else "#e67e22"
    figure.add_trace(go.Scatter3d(
        x=[start[0], switch[0]],
        y=[start[1], switch[1]],
        z=[start[2], switch[2]],
        mode="lines",
        name="Straight powered segment",
        line={"color": segment_color, "width": 9},
        hoverinfo="skip",
    ))
    speed = float(np.linalg.norm(state.velocity_mps))
    customdata = [[
        evaluation.candidate_id,
        np.degrees(state.heading_rad),
        np.degrees(state.flight_path_angle_rad),
        speed,
        state.powered_path_length_m,
        state.total_mechanical_energy_j / 1.0e6,
    ]]
    feasibility_text = (
        "powered feasible"
        if evaluation.powered_feasible
        else str(evaluation.infeasibility_reason)
    )
    figure.add_trace(go.Scatter3d(
        x=[switch[0]],
        y=[switch[1]],
        z=[switch[2]],
        mode="markers+text",
        name=f"Selected switch ID {evaluation.candidate_id}",
        text=[f"Switch {evaluation.candidate_id}"],
        textposition="top center",
        marker={
            "size": 10,
            "color": segment_color,
            "symbol": "diamond",
            "line": {"color": "#111111", "width": 2},
        },
        customdata=customdata,
        hovertext=[feasibility_text],
        hovertemplate=(
            "Candidate %{customdata[0]:.0f}"
            "<br>x=%{x:.3f} map unit<br>y=%{y:.3f} map unit"
            "<br>z=%{z:.3f} map unit"
            "<br>heading=%{customdata[1]:.3f}°"
            "<br>flight-path angle=%{customdata[2]:.3f}°"
            "<br>speed=%{customdata[3]:.3f} m/s"
            "<br>powered length=%{customdata[4]:.3f} m"
            "<br>total energy=%{customdata[5]:.6f} MJ"
            "<br>%{hovertext}<extra></extra>"
        ),
    ))
    figure.update_layout(title={
        "text": (
            f"Stage 4A: Single Switching Candidate {evaluation.candidate_id}"
            f"<br><sup>heading={np.degrees(state.heading_rad):.2f}°, "
            f"speed={speed:.2f} m/s, "
            f"energy={state.total_mechanical_energy_j / 1.0e6:.3f} MJ, "
            f"powered feasible={evaluation.powered_feasible}</sup>"
        ),
        "x": 0.5,
    })
    return figure


def plot_candidate_energy_classification(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
    evaluations: tuple["CandidateEnergyEvaluation", ...],
) -> go.Figure:
    """Show categorical powered/glide outcomes for switching candidates."""
    if not evaluations:
        raise ValueError("at least one candidate evaluation is required")
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    categories = (
        (
            "Acoustic condition failed",
            "#7b2cbf",
            "cross",
            lambda result: not result.acoustically_neutralized,
        ),
        (
            "Powered infeasible",
            "#f28e2b",
            "diamond",
            lambda result: (
                result.acoustically_neutralized and not result.powered_feasible
            ),
        ),
        (
            "Powered feasible / glide unreachable",
            "#d62728",
            "x",
            lambda result: result.powered_feasible and not result.reachable,
        ),
        (
            "Powered feasible / glide reachable",
            "#2ca02c",
            "circle",
            lambda result: result.reachable,
        ),
    )
    for category_name, color, symbol, predicate in categories:
        selected = tuple(result for result in evaluations if predicate(result))
        if not selected:
            continue
        positions = np.vstack([
            result.candidate.position_map for result in selected
        ])
        customdata = np.column_stack((
            np.fromiter((result.candidate_id for result in selected), dtype=int),
            np.fromiter((result.energy_margin_j / 1.0e6 for result in selected), dtype=float),
            np.fromiter((result.equivalent_height_margin_m for result in selected), dtype=float),
        ))
        reasons = [
            result.infeasibility_reason or "certified reachable"
            for result in selected
        ]
        figure.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            name=f"{category_name} ({len(selected)})",
            marker={
                "size": 6,
                "color": color,
                "symbol": symbol,
                "opacity": 0.95,
                "line": {"color": "#111111", "width": 0.7},
            },
            customdata=customdata,
            text=reasons,
            hovertemplate=(
                "Candidate %{customdata[0]:.0f}"
                "<br>x=%{x:.3f} map unit<br>y=%{y:.3f} map unit"
                "<br>z=%{z:.3f} map unit"
                "<br>energy margin=%{customdata[1]:.6f} MJ"
                "<br>height margin=%{customdata[2]:.3f} m"
                "<br>%{text}<extra></extra>"
            ),
        ))
    figure.update_layout(title={
        "text": (
            "Stage 4B: Switching-Candidate Powered/Energy Certification"
            "<br><sup>Categorical outcomes; no Bellman path search</sup>"
        ),
        "x": 0.5,
    })
    return figure


def _bellman_position_classes(
    graph: "BellmanGraph",
    solution: "UnitCostReachability",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse heading states into reachable/unreachable position records."""
    from bellman_state import BellmanState

    reachable_positions: list[np.ndarray] = []
    reachable_heading_counts: list[int] = []
    unreachable_positions: list[np.ndarray] = []
    unreachable_heading_counts: list[int] = []
    grid = graph.grid
    for altitude_index in range(grid.altitude_count):
        for y_index in range(grid.y_count):
            for x_index in range(grid.x_count):
                state_ids = np.array([
                    grid.encode(BellmanState(
                        x_index,
                        y_index,
                        altitude_index,
                        heading_bin,
                    ))
                    for heading_bin in range(grid.heading_bin_count)
                ])
                admissible = graph.node_mask[state_ids]
                if not np.any(admissible):
                    continue
                reachable_count = int(np.count_nonzero(
                    solution.goal_reachable[state_ids] & admissible,
                ))
                position = grid.position_map(BellmanState(
                    x_index,
                    y_index,
                    altitude_index,
                    0,
                ))
                if reachable_count:
                    reachable_positions.append(position)
                    reachable_heading_counts.append(reachable_count)
                else:
                    unreachable_positions.append(position)
                    unreachable_heading_counts.append(int(np.count_nonzero(admissible)))

    def stack(values: list[np.ndarray]) -> np.ndarray:
        return np.vstack(values) if values else np.empty((0, 3), dtype=float)

    return (
        stack(reachable_positions),
        np.asarray(reachable_heading_counts, dtype=int),
        stack(unreachable_positions),
        np.asarray(unreachable_heading_counts, dtype=int),
    )


def plot_bellman_reachability(
    graph: "BellmanGraph",
    solution: "UnitCostReachability",
    *,
    show_reachable: bool = True,
    show_unreachable: bool = True,
    maximum_heading_arrows: int = 80,
) -> go.Figure:
    """Plot heading-collapsed goal-reachable and unreachable grid positions."""
    if not show_reachable and not show_unreachable:
        raise ValueError("at least one Bellman position class must be shown")
    figure = plot_terrain_map(graph.terrain)
    figure.add_trace(_point_trace(
        graph.goal,
        name="Goal",
        color="#e6a700",
        symbol="square",
    ))
    reachable, reachable_counts, unreachable, unreachable_counts = (
        _bellman_position_classes(graph, solution)
    )
    if show_unreachable and len(unreachable):
        figure.add_trace(go.Scatter3d(
            x=unreachable[:, 0],
            y=unreachable[:, 1],
            z=unreachable[:, 2],
            mode="markers",
            name=f"Unreachable positions ({len(unreachable)})",
            marker={"size": 2.8, "color": "#9aa0a6", "opacity": 0.35},
            customdata=unreachable_counts[:, None],
            hovertemplate=(
                "Unreachable position<br>x=%{x:.2f}<br>y=%{y:.2f}<br>h=%{z:.2f}"
                "<br>admissible headings=%{customdata[0]:.0f}<extra></extra>"
            ),
        ))
    if show_reachable and len(reachable):
        figure.add_trace(go.Scatter3d(
            x=reachable[:, 0],
            y=reachable[:, 1],
            z=reachable[:, 2],
            mode="markers",
            name=f"Goal-reachable positions ({len(reachable)})",
            marker={
                "size": 3.2,
                "color": reachable_counts,
                "colorscale": "Viridis",
                "cmin": 1,
                "cmax": graph.grid.heading_bin_count,
                "opacity": 0.72,
                "colorbar": {"title": "reachable headings"},
            },
            customdata=reachable_counts[:, None],
            hovertemplate=(
                "Reachable position<br>x=%{x:.2f}<br>y=%{y:.2f}<br>h=%{z:.2f}"
                "<br>reachable headings=%{customdata[0]:.0f}<extra></extra>"
            ),
        ))

    policy_ids = np.flatnonzero(
        solution.goal_reachable & (solution.policy_successor >= 0),
    )
    if show_reachable and len(policy_ids) and maximum_heading_arrows > 0:
        step = max(1, len(policy_ids) // maximum_heading_arrows)
        selected_ids = policy_ids[::step][:maximum_heading_arrows]
        arrow_positions = np.vstack([
            graph.grid.position_map(graph.grid.decode(int(state_id)))
            for state_id in selected_ids
        ])
        arrow_headings = np.array([
            graph.grid.heading_rad(graph.grid.decode(int(state_id)).heading_bin)
            for state_id in selected_ids
        ])
        figure.add_trace(go.Cone(
            x=arrow_positions[:, 0],
            y=arrow_positions[:, 1],
            z=arrow_positions[:, 2],
            u=np.cos(arrow_headings),
            v=np.sin(arrow_headings),
            w=np.zeros(len(arrow_headings)),
            sizemode="absolute",
            sizeref=0.28,
            anchor="tail",
            colorscale=[[0.0, "#1f4e79"], [1.0, "#1f4e79"]],
            showscale=False,
            name="Sparse heading arrows",
            hoverinfo="skip",
        ))
    figure.update_layout(
        title={"text": "Stage 5: Unit-Cost Glide-Graph Reachability", "x": 0.5},
        scene={
            "zaxis": {
                "range": [
                    graph.grid.minimum_altitude_map,
                    graph.grid.maximum_altitude_map + 0.25,
                ]
            }
        },
    )
    return figure


def plot_bellman_successor_debug(
    graph: "BellmanGraph",
    transition_model: "GlideTransitionModel",
    selected_state: "BellmanState",
) -> go.Figure:
    """Show valid successors and rejected terrain-crossing attempts."""
    figure = plot_terrain_map(graph.terrain)
    source = graph.grid.position_map(selected_state)
    edges, _, rejected = transition_model.successors(
        selected_state,
        include_rejected=True,
    )
    figure.add_trace(go.Scatter3d(
        x=[source[0]], y=[source[1]], z=[source[2]],
        mode="markers+text",
        text=[f"State {graph.grid.encode(selected_state)}"],
        textposition="top center",
        name="Selected state",
        marker={"size": 9, "color": "#1f4e79", "symbol": "diamond"},
    ))

    def segment_trace(targets: list[np.ndarray], name: str, color: str, dash: str) -> None:
        if not targets:
            return
        x_values: list[float | None] = []
        y_values: list[float | None] = []
        z_values: list[float | None] = []
        for target in targets:
            x_values.extend((source[0], target[0], None))
            y_values.extend((source[1], target[1], None))
            z_values.extend((source[2], target[2], None))
        figure.add_trace(go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            name=name,
            line={"color": color, "width": 7, "dash": dash},
            hoverinfo="skip",
        ))

    segment_trace(
        [graph.grid.position_map(edge.target_state) for edge in edges],
        f"Valid successors ({len(edges)})",
        "#2ca02c",
        "solid",
    )
    terrain_rejections = [
        record.target_position_map
        for record in rejected
        if record.reason == "terrain collision"
    ]
    segment_trace(
        terrain_rejections,
        f"Rejected terrain edges ({len(terrain_rejections)})",
        "#d62728",
        "dash",
    )
    heading = graph.grid.heading_rad(selected_state.heading_bin)
    figure.add_trace(go.Cone(
        x=[source[0]], y=[source[1]], z=[source[2]],
        u=[np.cos(heading)], v=[np.sin(heading)], w=[0.0],
        sizemode="absolute", sizeref=0.5, anchor="tail",
        colorscale=[[0.0, "#1f4e79"], [1.0, "#1f4e79"]],
        showscale=False, name="Current heading", hoverinfo="skip",
    ))
    figure.update_layout(title={
        "text": "Stage 5: Selected-State Successor Geometry",
        "x": 0.5,
    })
    return figure


def plot_bellman_policy_path(
    graph: "BellmanGraph",
    solution: "UnitCostReachability",
    start_state_id: int,
) -> go.Figure:
    """Plot one exact unit-cost policy replay to the goal region."""
    path = solution.backtrack(start_state_id)
    if not path:
        raise ValueError("selected start state is not goal-reachable")
    positions = np.vstack([
        graph.grid.position_map(graph.grid.decode(state_id))
        for state_id in path
    ])
    figure = plot_terrain_map(graph.terrain)
    figure.add_trace(_point_trace(
        graph.goal,
        name="Goal",
        color="#e6a700",
        symbol="square",
    ))
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines+markers",
        name=f"Unit-cost policy ({len(path) - 1} edges)",
        line={"color": "#1f77b4", "width": 8},
        marker={"color": "#1f77b4", "size": 5},
        customdata=np.asarray(path)[:, None],
        hovertemplate=(
            "State %{customdata[0]:.0f}<br>x=%{x:.2f}<br>y=%{y:.2f}"
            "<br>h=%{z:.2f}<extra></extra>"
        ),
    ))
    figure.update_layout(title={
        "text": "Stage 5: Obstacle-Free Unit-Cost Policy Replay",
        "x": 0.5,
    })
    return figure


def plot_time_optimal_trajectory(
    run: "TimeOptimalRun",
    *,
    goal_tolerance_m: float = 25.0,
    meters_per_map_unit: float = 100.0,
) -> go.Figure:
    """Plot a stored Stage-6 time-only result without recomputing the solve."""
    if run.path is None:
        raise ValueError("selected Stage-6 start state is not goal-reachable")
    if not np.isfinite(goal_tolerance_m) or goal_tolerance_m <= 0.0:
        raise ValueError("goal_tolerance_m must be finite and positive")
    if not np.isfinite(meters_per_map_unit) or meters_per_map_unit <= 0.0:
        raise ValueError("meters_per_map_unit must be finite and positive")

    graph = run.graph
    solution = run.solution
    path = run.path
    positions = np.vstack([
        graph.grid.position_map(graph.grid.decode(state_id))
        for state_id in path.state_ids
    ])
    cumulative_time = np.concatenate((
        np.array([0.0]),
        np.cumsum([edge.duration_s for edge in path.edges]),
    ))
    reachable, reachable_counts, _, _ = _bellman_position_classes(
        graph,
        solution,
    )

    figure = plot_terrain_map(graph.terrain)
    if len(reachable):
        figure.add_trace(go.Scatter3d(
            x=reachable[:, 0],
            y=reachable[:, 1],
            z=reachable[:, 2],
            mode="markers",
            name=f"Goal-reachable positions ({len(reachable)})",
            marker={
                "size": 2.5,
                "color": reachable_counts,
                "colorscale": "Viridis",
                "cmin": 1,
                "cmax": graph.grid.heading_bin_count,
                "opacity": 0.3,
                "colorbar": {"title": "reachable headings"},
            },
            customdata=reachable_counts[:, None],
            hovertemplate=(
                "Reachable position<br>x=%{x:.2f}<br>y=%{y:.2f}<br>h=%{z:.2f}"
                "<br>reachable headings=%{customdata[0]:.0f}<extra></extra>"
            ),
        ))

    figure.add_trace(_point_trace(
        graph.goal,
        name="Goal",
        color="#e6a700",
        symbol="square",
    ))
    tolerance_map = goal_tolerance_m / meters_per_map_unit
    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    elevation = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 17)
    sphere_x = (
        graph.goal.x
        + tolerance_map * np.outer(np.cos(elevation), np.cos(azimuth))
    )
    sphere_y = (
        graph.goal.y
        + tolerance_map * np.outer(np.cos(elevation), np.sin(azimuth))
    )
    sphere_z = (
        graph.goal.z
        + tolerance_map * np.outer(np.sin(elevation), np.ones_like(azimuth))
    )
    figure.add_trace(go.Surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        name=f"Goal tolerance ({goal_tolerance_m:g} m)",
        colorscale=[[0.0, "#f2c94c"], [1.0, "#f2c94c"]],
        opacity=0.22,
        showscale=False,
        hoverinfo="skip",
    ))

    figure.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines+markers",
        name=f"Time-optimal trajectory ({len(path.edges)} edges)",
        line={"color": "#d62728", "width": 9},
        marker={"color": "#d62728", "size": 5},
        customdata=np.column_stack((np.asarray(path.state_ids), cumulative_time)),
        hovertemplate=(
            "State %{customdata[0]:.0f}<br>x=%{x:.2f}<br>y=%{y:.2f}"
            "<br>h=%{z:.2f}<br>elapsed=%{customdata[1]:.3f} s"
            "<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[0, 1]],
        z=[positions[0, 2]],
        mode="markers+text",
        text=[f"Start {run.start_state_id}"],
        textposition="top center",
        name="Selected start",
        marker={"size": 9, "color": "#1f4e79", "symbol": "diamond"},
    ))

    if len(path.edges):
        displacement = positions[1:] - positions[:-1]
        norms = np.linalg.norm(displacement, axis=1)
        direction = displacement / norms[:, None]
        anchors = 0.5 * (positions[:-1] + positions[1:])
        figure.add_trace(go.Cone(
            x=anchors[:, 0],
            y=anchors[:, 1],
            z=anchors[:, 2],
            u=direction[:, 0],
            v=direction[:, 1],
            w=direction[:, 2],
            sizemode="absolute",
            sizeref=0.35,
            anchor="center",
            colorscale=[[0.0, "#7f0000"], [1.0, "#7f0000"]],
            showscale=False,
            name="Trajectory direction",
            hoverinfo="skip",
        ))

    timing = run.metrics.timing
    figure.update_layout(
        title={
            "text": (
                "Stage 6: Exact Time-Only Bellman Trajectory"
                f"<br><sup>Bellman={path.bellman_value_s:.6f} s; "
                f"replay={path.geometric_replay_duration_s:.6f} s; "
                f"states={run.metrics.state_count}; edges={run.metrics.edge_count}; "
                f"solve={timing.solve_s:.4f} s</sup>"
            ),
            "x": 0.5,
        },
        scene={
            "zaxis": {
                "range": [
                    min(-tolerance_map, graph.grid.minimum_altitude_map),
                    graph.grid.maximum_altitude_map + 0.25,
                ]
            }
        },
    )
    return figure


def plot_edge_hazard_audit(
    terrain: TerrainModel,
    sensor: Point3D,
    result: "EdgeHazardResult",
) -> go.Figure:
    """Plot one stored Stage-7 quadrature table and its LOS classifications."""
    if not result.samples:
        raise ValueError("edge hazard result must contain quadrature samples")
    positions = np.vstack([sample.position_map for sample in result.samples])
    visible_mask = np.asarray([sample.visible for sample in result.samples], dtype=bool)
    sensor_position = sensor.as_array()
    figure = plot_terrain_map(terrain)
    figure.add_trace(_point_trace(
        sensor,
        name="Sensor",
        color="#8e44ad",
        symbol="diamond",
    ))
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines",
        name="Selected edge",
        line={"color": "#1f4e79", "width": 9},
        hoverinfo="skip",
    ))

    for mask, expected_visibility, name, color in (
        (visible_mask, True, "Visible quadrature samples", "#2ca02c"),
        (~visible_mask, False, "Occluded quadrature samples", "#d62728"),
    ):
        selected = positions[mask]
        selected_samples = [
            sample
            for sample in result.samples
            if sample.visible == expected_visibility
        ]
        if not len(selected):
            continue
        customdata = np.array([
            [
                sample.sample_index,
                sample.time_s,
                sample.total_rate_per_s,
                sample.hazard_contribution,
            ]
            for sample in selected_samples
        ])
        figure.add_trace(go.Scatter3d(
            x=selected[:, 0],
            y=selected[:, 1],
            z=selected[:, 2],
            mode="markers",
            name=f"{name} ({len(selected)})",
            marker={"size": 6, "color": color, "opacity": 0.9},
            customdata=customdata,
            hovertemplate=(
                "sample=%{customdata[0]:.0f}<br>x=%{x:.3f}<br>y=%{y:.3f}"
                "<br>h=%{z:.3f}<br>t=%{customdata[1]:.3f} s"
                "<br>lambda=%{customdata[2]:.6e} 1/s"
                "<br>dH=%{customdata[3]:.6e}<extra></extra>"
            ),
        ))

        ray_x: list[float | None] = []
        ray_y: list[float | None] = []
        ray_z: list[float | None] = []
        for sample_position in selected:
            ray_x.extend((sensor_position[0], sample_position[0], None))
            ray_y.extend((sensor_position[1], sample_position[1], None))
            ray_z.extend((sensor_position[2], sample_position[2], None))
        figure.add_trace(go.Scatter3d(
            x=ray_x,
            y=ray_y,
            z=ray_z,
            mode="lines",
            name=(
                "Visible sensor rays" if expected_visibility
                else "Occluded sensor rays"
            ),
            line={"color": color, "width": 2, "dash": "dot"},
            opacity=0.28,
            hoverinfo="skip",
        ))

    displacement = positions[-1] - positions[0]
    direction = displacement / np.linalg.norm(displacement)
    midpoint = 0.5 * (positions[0] + positions[-1])
    figure.add_trace(go.Cone(
        x=[midpoint[0]],
        y=[midpoint[1]],
        z=[midpoint[2]],
        u=[direction[0]],
        v=[direction[1]],
        w=[direction[2]],
        sizemode="absolute",
        sizeref=0.55,
        anchor="center",
        colorscale=[[0.0, "#1f4e79"], [1.0, "#1f4e79"]],
        showscale=False,
        name="Edge direction",
        hoverinfo="skip",
    ))
    figure.update_layout(title={
        "text": (
            "Stage 7: LOS-Gated Edge-Hazard Quadrature"
            f"<br><sup>duration={result.duration_s:.6f} s; "
            f"hazard={result.hazard:.8e}; "
            f"visible={result.visible_sample_count}/{result.quadrature_resolution}; "
            f"occluded={result.occluded_sample_count}</sup>"
        ),
        "x": 0.5,
    })
    return figure


def plot_energy_reachability_surface(
    terrain_map: TerrainModel,
    mission_points: MissionPoints,
    los_surface: LOSTangentSurface,
    visualization_rays: VisualizationRaySet,
    energy_surface: EnergyReachabilitySurface,
    *,
    aircraft_name: str,
    meters_per_map_unit: float,
    goal_tolerance_m: float,
) -> go.Figure:
    """Color the displayed LOS surface by state-aware goal reachability."""
    figure = plot_los_tangent_surface(
        terrain_map,
        mission_points,
        los_surface,
        visualization_rays,
    )
    for trace in figure.data:
        if str(trace.name).startswith("LOS tangent surface"):
            trace.visible = False
            trace.showlegend = False

    for region, name, color in (
        (
            energy_surface.unreachable_region,
            "Goal-unreachable switching region",
            "#d62728",
        ),
        (
            energy_surface.reachable_region,
            "Goal-reachable switching region",
            "#2ca02c",
        ),
    ):
        mesh = region.triangle_mesh()
        if not len(mesh.triangles):
            continue
        customdata = np.column_stack(
            (
                region.available_energy_j / 1.0e6,
                region.required_energy_j / 1.0e6,
                region.energy_margin_j / 1.0e6,
                region.height_margin_m,
            )
        )
        figure.add_trace(
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.triangles[:, 0],
                j=mesh.triangles[:, 1],
                k=mesh.triangles[:, 2],
                name=name,
                color=color,
                opacity=0.25,
                flatshading=True,
                customdata=customdata,
                hovertemplate=(
                    f"{name}<br>x=%{{x:.3f}} map unit"
                    "<br>y=%{y:.3f} map unit<br>z=%{z:.3f} map unit"
                    "<br>available energy=%{customdata[0]:.3f} MJ"
                    "<br>required energy=%{customdata[1]:.3f} MJ"
                    "<br>energy margin=%{customdata[2]:.3f} MJ"
                    "<br>energy-height margin=%{customdata[3]:.1f} m<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title={
            "text": (
                "Stage 6: Total-Energy Goal Reachability on LOS Tangent Surface"
                f"<br><sup>{aircraft_name}; 1 map unit = {meters_per_map_unit:g} m; "
                f"goal tolerance = {goal_tolerance_m:g} m</sup>"
            ),
            "x": 0.5,
        }
    )
    return figure


def plot_finite_stackelberg_solution(
    run: "FiniteStackelbergRun",
) -> go.Figure:
    """Show every finite sensor action and the selected SSE trajectory."""
    selected_evaluation = run.selected_evaluation
    follower = selected_evaluation.sse_follower_result
    if follower is None:
        raise ValueError("selected Stackelberg evaluation needs a follower response")
    selected_attacker_run = replace(
        selected_evaluation.attacker_run,
        selected_result=follower,
    )
    figure = plot_attacker_best_response(selected_attacker_run)
    positions = np.vstack([
        evaluation.candidate.action.sensor_position_map
        for evaluation in run.evaluations
    ])
    payoffs = np.asarray([
        np.nan if evaluation.defender_payoff is None else evaluation.defender_payoff
        for evaluation in run.evaluations
    ])
    customdata = np.asarray([
        [
            evaluation.candidate.action_id,
            "" if evaluation.defender_payoff is None else f"{evaluation.defender_payoff:.9f}",
            "" if evaluation.outcome is None else f"{evaluation.outcome.attacker_payoff:.9f}",
            evaluation.infeasibility_reason or "feasible",
        ]
        for evaluation in run.evaluations
    ], dtype=object)
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers+text",
        text=[f"D{evaluation.candidate.action_id}" for evaluation in run.evaluations],
        textposition="bottom center",
        name="Finite Defender actions",
        marker={
            "size": 9,
            "color": np.nan_to_num(payoffs, nan=0.0),
            "colorscale": "Plasma",
            "colorbar": {"title": "Defender PoD"},
            "line": {"color": "#111827", "width": 1.0},
        },
        customdata=customdata,
        hovertemplate=(
            "Defender D%{customdata[0]}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<br>PoD=%{customdata[1]}<br>attacker J=%{customdata[2]}"
            "<br>%{customdata[3]}<extra></extra>"
        ),
    ))
    selected_position = selected_evaluation.candidate.action.sensor_position_map
    figure.add_trace(go.Scatter3d(
        x=[selected_position[0]], y=[selected_position[1]], z=[selected_position[2]],
        mode="markers+text",
        text=["STACKELBERG DEFENDER"],
        textposition="top center",
        name=f"Selected Defender D{selected_evaluation.candidate.action_id}",
        marker={
            "size": 15, "color": "#fde047", "symbol": "diamond",
            "line": {"color": "#7c2d12", "width": 3},
        },
    ))
    figure.update_layout(title={
        "text": (
            "Stage 12: Exact Finite Stackelberg Solution"
            f"<br><sup>Selected D{selected_evaluation.candidate.action_id}; "
            f"sensor=({selected_position[0]:.2f}, {selected_position[1]:.2f}, "
            f"{selected_position[2]:.2f}); Defender PoD={run.outcome.defender_payoff:.6f}</sup>"
        ),
        "x": 0.5,
    })
    return figure


def plot_defender_payoff_comparison(
    run: "FiniteStackelbergRun",
) -> go.Figure:
    """Compare Defender PoD and corresponding follower cost per action."""
    action_ids = [evaluation.candidate.action_id for evaluation in run.evaluations]
    payoffs = [
        np.nan if evaluation.outcome is None else evaluation.outcome.defender_payoff
        for evaluation in run.evaluations
    ]
    attacker_costs = [
        np.nan if evaluation.outcome is None else evaluation.outcome.attacker_payoff
        for evaluation in run.evaluations
    ]
    colors = [
        "#f59e0b" if action_id == run.selected_evaluation.candidate.action_id else "#64748b"
        for action_id in action_ids
    ]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Bar(
        x=action_ids, y=payoffs, name="Defender payoff (PoD)",
        marker={"color": colors},
        hovertemplate="D%{x}<br>PoD=%{y:.9f}<extra></extra>",
    ), secondary_y=False)
    figure.add_trace(go.Scatter(
        x=action_ids, y=attacker_costs, name="Attacker objective",
        mode="lines+markers", line={"color": "#2563eb", "width": 3},
        marker={"size": 10},
        hovertemplate="D%{x}<br>attacker J=%{y:.9f}<extra></extra>",
    ), secondary_y=True)
    figure.update_xaxes(title_text="Defender action ID", tickmode="array", tickvals=action_ids)
    figure.update_yaxes(title_text="Defender PoD (maximize)", secondary_y=False)
    figure.update_yaxes(title_text="Attacker objective (minimize)", secondary_y=True)
    figure.update_layout(
        title={"text": "Stage 12: Exhaustive Defender-Action Comparison", "x": 0.5},
        template="plotly_white", width=950, height=600,
    )
    return figure
