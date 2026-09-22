"""Visual verification of the exact local SSE, one figure per algorithm stage.

    1    LOS tangent surface at given d
    2    energy & reachable region at given d
    3    bellman cost-to-go at given d
    4    exact local SSE solution              (3-D)
    4-1  exact local SSE solution, side view   (x-z)
    4-2  exact local SSE solution, top view    (x-y)

Every figure is written twice: an interactive ``.html`` to inspect by rotating,
and a ``.png`` of the same figure so the result is reviewable without a browser.

    python P1b_Exact_figures.py [spatial_resolution_m]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from P1b_Exact_Local_SSE import (
    _phase_cost,
    _sample_tangent_surface,
    exact_best_response,
    exact_local_sse,
)
from P1b_condition import ComputationCondition, Scene, build_scene
from energy_model import StraightPoweredPhaseModel
from los_geometry import LOSModel


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "P1b_exact_verification"

INK = "#17201D"
INK_SOFT = "#46544F"
INK_FAINT = "#9AA6A1"
TERRAIN = "#8A8F8C"
EXACT = "#1C5770"
SENSOR = "#8A3838"
GOAL = "#D4A017"
SWITCH = "#9C6511"

WIDTH = 1280
HEIGHT = 780


def _terrain_mesh(terrain: Any) -> list[go.Mesh3d]:
    meshes = []
    for box in terrain.obstacle_boxes():
        x0, x1 = box.center_x - box.width_x / 2.0, box.center_x + box.width_x / 2.0
        y0, y1 = box.center_y - box.width_y / 2.0, box.center_y + box.width_y / 2.0
        z0 = float(getattr(box, "base_z", 0.0))
        z1 = z0 + float(box.height)
        meshes.append(go.Mesh3d(
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z0, z0, z0, z1, z1, z1, z1],
            i=[0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3],
            j=[1, 2, 4, 3, 5, 6, 2, 5, 3, 6, 0, 7],
            k=[2, 3, 5, 7, 6, 7, 6, 6, 7, 7, 4, 4],
            color=TERRAIN, opacity=0.45, flatshading=True,
            name="terrain", showlegend=True, hoverinfo="name",
        ))
    return meshes


def _terrain_rectangles(terrain: Any, plane: str) -> list[dict[str, Any]]:
    """Terrain footprint as a 2-D shape for the side (x-z) or top (x-y) view."""
    shapes = []
    for box in terrain.obstacle_boxes():
        x0, x1 = box.center_x - box.width_x / 2.0, box.center_x + box.width_x / 2.0
        if plane == "side":
            y0 = float(getattr(box, "base_z", 0.0))
            y1 = y0 + float(box.height)
        else:
            y0, y1 = box.center_y - box.width_y / 2.0, box.center_y + box.width_y / 2.0
        shapes.append({
            "type": "rect", "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "fillcolor": TERRAIN, "opacity": 0.45,
            "line": {"color": INK_SOFT, "width": 1},
            "layer": "below",
        })
    return shapes


def _scene_markers(scene: Scene, sensor_map) -> list[go.Scatter3d]:
    config = scene.config
    coincident = (
        abs(sensor_map[0] - config.goal.x) < 1e-9
        and abs(sensor_map[1] - config.goal.y) < 1e-9
    )
    return [
        go.Scatter3d(
            x=[config.start.x], y=[config.start.y], z=[config.start.z],
            mode="markers", name="start",
            marker={"size": 7, "color": "white",
                    "line": {"color": INK, "width": 2}},
        ),
        go.Scatter3d(
            x=[config.goal.x], y=[config.goal.y], z=[config.goal.z],
            mode="markers", name="goal",
            marker={"size": 11, "color": GOAL, "symbol": "diamond",
                    "line": {"color": INK, "width": 1}},
        ),
        go.Scatter3d(
            x=[sensor_map[0]], y=[sensor_map[1]], z=[sensor_map[2]],
            mode="markers",
            name="sensor d (= goal)" if coincident else "sensor d",
            marker={"size": 9, "color": SENSOR, "symbol": "cross"},
        ),
    ]


def _layout_3d(scene: Scene) -> dict[str, Any]:
    bounds = scene.config.graph_bounds
    return {
        "scene": {
            "xaxis": {"title": "x [map unit]", "range": [bounds.x_min, bounds.x_max]},
            "yaxis": {"title": "y [map unit]", "range": [bounds.y_min, bounds.y_max]},
            "zaxis": {"title": "altitude [map unit]",
                      "range": [0.0, scene.grid.maximum_altitude_map]},
            "aspectmode": "manual",
            "aspectratio": {"x": 2.0, "y": 1.0, "z": 0.7},
            "camera": {"eye": {"x": -1.6, "y": -1.9, "z": 1.0}},
        },
        "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
        "legend": {"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        "width": WIDTH, "height": HEIGHT,
        "template": "plotly_white",
    }


def _save(figure: go.Figure, output: Path, stem: str) -> str:
    figure.write_html(output / f"{stem}.html", include_plotlyjs="cdn")
    try:
        figure.write_image(output / f"{stem}.png", width=WIDTH, height=HEIGHT, scale=1)
    except Exception as error:  # noqa: BLE001 - static export is a convenience
        print(f"    (png export skipped for {stem}: {type(error).__name__})")
    return stem


def _domain_mask(points: np.ndarray, scene: Scene) -> np.ndarray:
    bounds = scene.config.graph_bounds
    return (
        (points[..., 0] >= bounds.x_min) & (points[..., 0] <= bounds.x_max)
        & (points[..., 1] >= bounds.y_min) & (points[..., 1] <= bounds.y_max)
        & (points[..., 2] >= 0.0)
        & (points[..., 2] <= scene.grid.maximum_altitude_map)
    )


def _contour_for(scene: Scene, sensor_map):
    discretization = scene.config.discretization
    return LOSModel(scene.terrain).trace_tangent_contour(
        scene.mission_for(sensor_map).sensor,
        probe_grid_size=discretization.los_probe_grid_size,
        boundary_refinement_steps=discretization.los_boundary_refinement_steps,
    )


def figure_1_los_surface(scene: Scene, sensor_map, output: Path) -> str:
    discretization = scene.config.discretization
    contour = _contour_for(scene, sensor_map)
    origin = contour.origin.as_array()
    fractions = np.linspace(0.0, 1.0, 180, endpoint=False)
    scales = np.linspace(
        discretization.switching_radial_min, discretization.switching_radial_max, 48,
    )
    directions = np.asarray(
        [contour.tangent_vector_at(float(v)) for v in fractions], dtype=float,
    )
    surface = origin[None, None, :] + scales[None, :, None] * directions[:, None, :]
    # Rays run far past the lattice; outside it they cannot become switching
    # points, so they are masked rather than drawn.
    masked = surface.copy()
    masked[~_domain_mask(surface, scene)] = np.nan

    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)
    figure.add_trace(go.Surface(
        x=masked[:, :, 0], y=masked[:, :, 1], z=masked[:, :, 2],
        colorscale=[[0, EXACT], [1, EXACT]], showscale=False, opacity=0.45,
        name="LOS tangent surface", showlegend=True, hoverinfo="name",
    ))
    for trace in _scene_markers(scene, sensor_map):
        figure.add_trace(trace)
    figure.update_layout(**_layout_3d(scene))
    return _save(figure, output, "1_LOS_tangent_surface")


def figure_2_reachable_region(scene: Scene, sensor_map, response, output: Path) -> str:
    discretization = scene.config.discretization
    contour = _contour_for(scene, sensor_map)
    points = _sample_tangent_surface(
        contour, discretization.switching_radial_min,
        discretization.switching_radial_max,
    )
    visible = points[_domain_mask(points, scene)]
    step = max(1, len(visible) // 6000)
    admissible = np.asarray(
        [scene.grid.position_map(scene.grid.decode(i))
         for i in response.admissible_state_ids], dtype=float,
    ) if response.admissible_state_ids else np.empty((0, 3))

    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)
    figure.add_trace(go.Scatter3d(
        x=visible[::step, 0], y=visible[::step, 1], z=visible[::step, 2],
        mode="markers", name=f"surface samples ({len(visible):,} in domain)",
        marker={"size": 1.6, "color": INK_FAINT, "opacity": 0.45},
    ))
    if len(admissible):
        figure.add_trace(go.Scatter3d(
            x=admissible[:, 0], y=admissible[:, 1], z=admissible[:, 2],
            mode="markers",
            name=f"admissible switching states ({len(admissible):,})",
            marker={"size": 4.5, "color": SWITCH,
                    "line": {"color": INK, "width": 0.5}},
        ))
    for trace in _scene_markers(scene, sensor_map):
        figure.add_trace(trace)
    figure.update_layout(**_layout_3d(scene))
    return _save(figure, output, "2_energy_reachable_region")


def figure_3_cost_to_go(scene: Scene, sensor_map, response, output: Path) -> str:
    solution = response.solution
    value = np.asarray(solution.value, dtype=float)
    finite = np.flatnonzero(np.isfinite(value) & solution.goal_reachable)
    # Collapse heading: show the best cost-to-go available at each position,
    # which is what the switch selection competes over.
    best: dict[tuple[int, int, int], float] = {}
    for state_id in finite:
        state = scene.grid.decode(int(state_id))
        key = (state.x_index, state.y_index, state.altitude_index)
        if key not in best or value[state_id] < best[key]:
            best[key] = float(value[state_id])
    keys = np.asarray(list(best), dtype=int)
    values = np.asarray([best[tuple(k)] for k in keys], dtype=float)
    positions = np.column_stack((
        scene.grid.bounds.x_min + keys[:, 0] * scene.grid.horizontal_spacing_map,
        scene.grid.bounds.y_min + keys[:, 1] * scene.grid.horizontal_spacing_map,
        scene.grid.minimum_altitude_map + keys[:, 2] * scene.grid.altitude_spacing_map,
    ))

    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers",
        name=f"cost-to-go ({len(values):,} positions)",
        marker={
            "size": 3.4, "color": values, "colorscale": "Viridis_r",
            "opacity": 0.85,
            "colorbar": {"title": "V(s)", "len": 0.6, "x": 1.02},
        },
        hovertemplate="V=%{marker.color:.3f}<extra></extra>",
    ))
    for trace in _scene_markers(scene, sensor_map):
        figure.add_trace(trace)
    figure.update_layout(**_layout_3d(scene))
    return _save(figure, output, "3_bellman_cost_to_go")


def _solution_paths(scene: Scene, sse, response):
    sensor_map = tuple(
        sse.selected_sensor_map or scene.defender_grid.position(scene.seed_action_id)
    )
    switch = np.asarray(response.switching_position_map, dtype=float)
    start = scene.config.start.as_array()
    glide = np.asarray(
        [scene.grid.position_map(scene.grid.decode(i))
         for i in response.glide_state_ids], dtype=float,
    )
    visited = np.asarray(
        [scene.defender_grid.position(a) for a in sse.visited_action_ids], dtype=float,
    )
    return sensor_map, start, switch, glide, visited


def figure_4_solution(scene: Scene, sse, response, output: Path) -> str:
    sensor_map, start, switch, glide, visited = _solution_paths(scene, sse, response)
    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)
    figure.add_trace(go.Scatter3d(
        x=[start[0], switch[0]], y=[start[1], switch[1]], z=[start[2], switch[2]],
        mode="lines", name="powered leg (straight)",
        line={"color": SENSOR, "width": 6, "dash": "dash"},
    ))
    if len(glide) > 1:
        figure.add_trace(go.Scatter3d(
            x=glide[:, 0], y=glide[:, 1], z=glide[:, 2],
            mode="lines+markers", name=f"glide ({len(glide) - 1} edges)",
            line={"color": EXACT, "width": 7},
            marker={"size": 4, "color": EXACT},
        ))
    figure.add_trace(go.Scatter3d(
        x=[switch[0]], y=[switch[1]], z=[switch[2]],
        mode="markers", name="switching point",
        marker={"size": 8, "color": SWITCH, "line": {"color": INK, "width": 1}},
    ))
    if len(visited) > 1:
        figure.add_trace(go.Scatter3d(
            x=visited[:, 0], y=visited[:, 1], z=visited[:, 2],
            mode="lines+markers",
            name=f"defender search path ({len(visited)} positions)",
            line={"color": SENSOR, "width": 3}, opacity=0.55,
            marker={"size": 4, "symbol": "square", "color": SENSOR},
        ))
    for trace in _scene_markers(scene, sensor_map):
        figure.add_trace(trace)
    figure.update_layout(**_layout_3d(scene))
    return _save(figure, output, "4_exact_local_SSE_solution")


def _figure_4_projection(
    scene: Scene, sse, response, output: Path, *, plane: str,
) -> str:
    """Side (x-z) or top (x-y) projection of the same solution."""
    sensor_map, start, switch, glide, visited = _solution_paths(scene, sse, response)
    axis = 2 if plane == "side" else 1
    bounds = scene.config.graph_bounds
    config = scene.config

    figure = go.Figure()
    # Shapes carry no legend entry, and in the side view the rectangle is only a
    # projection: a path drawn across it may well pass beside the terrain in y.
    # Say so, otherwise the view reads as a trajectory flying through solid rock.
    figure.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        name="terrain (x-z projection)" if plane == "side" else "terrain footprint",
        marker={"size": 12, "color": TERRAIN, "symbol": "square",
                "line": {"color": INK_SOFT, "width": 1}},
    ))
    figure.add_trace(go.Scatter(
        x=[start[0], switch[0]], y=[start[axis], switch[axis]],
        mode="lines", name="powered leg (straight)",
        line={"color": SENSOR, "width": 3, "dash": "dash"},
    ))
    if len(glide) > 1:
        figure.add_trace(go.Scatter(
            x=glide[:, 0], y=glide[:, axis], mode="lines+markers",
            name=f"glide ({len(glide) - 1} edges)",
            line={"color": EXACT, "width": 4}, marker={"size": 7},
        ))
    figure.add_trace(go.Scatter(
        x=[switch[0]], y=[switch[axis]], mode="markers", name="switching point",
        marker={"size": 12, "color": SWITCH, "line": {"color": INK, "width": 1}},
    ))
    if plane == "top" and len(visited) > 1:
        figure.add_trace(go.Scatter(
            x=visited[:, 0], y=visited[:, 1], mode="lines+markers",
            name=f"defender search path ({len(visited)})",
            line={"color": SENSOR, "width": 2}, opacity=0.6,
            marker={"size": 8, "symbol": "square"},
        ))
    goal_axis = config.goal.z if plane == "side" else config.goal.y
    start_axis = config.start.z if plane == "side" else config.start.y
    sensor_axis = sensor_map[2] if plane == "side" else sensor_map[1]
    figure.add_trace(go.Scatter(
        x=[config.start.x], y=[start_axis], mode="markers", name="start",
        marker={"size": 11, "color": "white", "line": {"color": INK, "width": 2}},
    ))
    figure.add_trace(go.Scatter(
        x=[config.goal.x], y=[goal_axis], mode="markers", name="goal",
        marker={"size": 15, "color": GOAL, "symbol": "star",
                "line": {"color": INK, "width": 1}},
    ))
    figure.add_trace(go.Scatter(
        x=[sensor_map[0]], y=[sensor_axis], mode="markers", name="sensor d",
        marker={"size": 13, "color": SENSOR, "symbol": "triangle-down"},
    ))

    y_title = "altitude [map unit]" if plane == "side" else "y [map unit]"
    y_range = (
        [0.0, scene.grid.maximum_altitude_map] if plane == "side"
        else [bounds.y_min, bounds.y_max]
    )
    figure.update_layout(
        shapes=_terrain_rectangles(scene.terrain, plane),
        xaxis={"title": "x [map unit]", "range": [bounds.x_min, bounds.x_max]},
        yaxis={"title": y_title, "range": y_range,
               "scaleanchor": "x" if plane == "top" else None},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        width=WIDTH, height=560, template="plotly_white",
        margin={"l": 60, "r": 20, "t": 20, "b": 50},
    )
    stem = (
        "4-1_exact_local_SSE_solution_side_view" if plane == "side"
        else "4-2_exact_local_SSE_solution_top_view"
    )
    return _save(figure, output, stem)


def figure_5_switching_map(
    scene: Scene, sensor_map, response, sse, output: Path,
    *, stem: str = "5_switching_point_objective_map",
) -> str:
    """The Attacker objective as a function of where it switches to glide.

    Each cell is ``min over altitude and heading of [ powered(s) + V(s) ]`` at that
    (x, y): the total objective the Attacker would pay by switching there and then
    flying optimally.  Collapsing by minimum is what makes the map comparable to
    the solution - the Attacker is free to pick the altitude and heading it arrives
    with, so the best one at a column is what that column is worth.

    Only admissible switching states appear.  A cell is blank either because the
    tangent surface never passes over it, or because the powered leg cannot reach
    it, or because no glide from it reaches the goal; those are different reasons
    and none of them is a large objective, so drawing them as one extreme colour
    would invent a landscape that is not there.

    The minimum of this map is the exact local SSE objective in the title, which is
    the check worth making when reading it.
    """
    config = scene.config
    grid = scene.grid
    bounds = grid.bounds
    value = np.asarray(response.solution.value, dtype=float)
    powered_model = StraightPoweredPhaseModel(
        parameters=config.glider, physical_scale=config.physical_scale,
    )
    mission = scene.mission_for(sensor_map)

    best: dict[tuple[int, int], float] = {}
    for state_id in response.admissible_state_ids:
        state_id = int(state_id)
        glide = float(value[state_id])
        if not np.isfinite(glide):
            continue
        state = grid.decode(state_id)
        position = grid.position_map(state)
        switching_state = powered_model.state_at(position, mission, scene.terrain)
        if not switching_state.powered_feasible:
            continue
        objective = glide + _phase_cost(
            0.0,
            switching_state.powered_path_length_m / config.glider.powered_speed_mps,
            config.attacker_objective,
        )
        key = (state.x_index, state.y_index)
        if key not in best or objective < best[key]:
            best[key] = objective

    if not best:
        raise RuntimeError("no admissible switching state has a finite objective")

    field = np.full((grid.y_count, grid.x_count), np.nan, dtype=float)
    for (x_index, y_index), objective in best.items():
        field[y_index, x_index] = objective
    x_values = bounds.x_min + np.arange(grid.x_count) * grid.horizontal_spacing_map
    y_values = bounds.y_min + np.arange(grid.y_count) * grid.horizontal_spacing_map

    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        x=x_values, y=y_values, z=field, colorscale="Viridis_r",
        # The interesting structure sits near the minimum; a few unreachable-looking
        # corners an order of magnitude above it would otherwise flatten everything.
        zmin=float(np.nanmin(field)),
        zmax=float(np.nanpercentile(field, 95)),
        colorbar={"title": "J_A", "len": 0.8},
        hovertemplate="x %{x}  y %{y}<br>J_A %{z:.6f}<extra></extra>",
        hoverongaps=False,
    ))
    for shape in _terrain_rectangles(scene.terrain, "top"):
        figure.add_shape(shape)

    minimum_key = min(best, key=lambda key: best[key])
    figure.add_trace(go.Scatter(
        x=[bounds.x_min + minimum_key[0] * grid.horizontal_spacing_map],
        y=[bounds.y_min + minimum_key[1] * grid.horizontal_spacing_map],
        mode="markers", name=f"best switching column  J_A = {best[minimum_key]:.6f}",
        marker={"size": 15, "color": SWITCH, "symbol": "diamond",
                "line": {"color": "white", "width": 2}},
    ))
    figure.add_trace(go.Scatter(
        x=[config.start.x], y=[config.start.y], mode="markers", name="start",
        marker={"size": 11, "color": "white", "line": {"color": INK, "width": 2}},
    ))
    figure.add_trace(go.Scatter(
        x=[config.goal.x], y=[config.goal.y], mode="markers", name="goal",
        marker={"size": 15, "color": GOAL, "symbol": "star",
                "line": {"color": INK, "width": 1}},
    ))
    figure.add_trace(go.Scatter(
        x=[sensor_map[0]], y=[sensor_map[1]], mode="markers", name="sensor d",
        marker={"size": 13, "color": SENSOR, "symbol": "triangle-down"},
    ))
    figure.update_layout(
        title={
            "text": f"exact local SSE   J_A = {sse.attacker_objective:.6f}"
                    f"   at d = {[float(v) for v in sensor_map]}",
            "x": 0.5, "xanchor": "center", "font": {"size": 15, "color": INK},
        },
        xaxis={"title": "x [map unit]", "range": [bounds.x_min, bounds.x_max],
               "constrain": "domain"},
        yaxis={"title": "y [map unit]", "range": [bounds.y_min, bounds.y_max],
               "scaleanchor": "x"},
        # Outside the axes: the admissible bands reach the top of the map, and a
        # legend floated over them hides cells the figure exists to show.
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0.0, "bgcolor": "rgba(0,0,0,0)"},
        width=WIDTH, height=660, template="plotly_white",
        margin={"l": 60, "r": 20, "t": 92, "b": 50},
    )
    return _save(figure, output, stem)


def _switching_objective_by_state(scene: Scene, sensor_map, response):
    """J_A for every admissible switching state: powered leg plus exact cost-to-go."""
    config = scene.config
    value = np.asarray(response.solution.value, dtype=float)
    powered_model = StraightPoweredPhaseModel(
        parameters=config.glider, physical_scale=config.physical_scale,
    )
    mission = scene.mission_for(sensor_map)
    objective_of: dict[int, float] = {}
    for state_id in response.admissible_state_ids:
        state_id = int(state_id)
        glide = float(value[state_id])
        if not np.isfinite(glide):
            continue
        position = scene.grid.position_map(scene.grid.decode(state_id))
        switching_state = powered_model.state_at(position, mission, scene.terrain)
        if not switching_state.powered_feasible:
            continue
        objective_of[state_id] = glide + _phase_cost(
            0.0,
            switching_state.powered_path_length_m / config.glider.powered_speed_mps,
            config.attacker_objective,
        )
    return objective_of


def figure_5_switching_map_3d(
    scene: Scene, sensor_map, response, sse, output: Path,
    *, stem: str = "5_switching_point_objective_map_3d",
) -> str:
    """The same objective map, but on the tangent surface itself rather than in plan.

    The plan view collapses altitude away, and the admissible set is a band only
    because a surface projects to a curve.  Here the switching states sit at their
    own (x, y, z), so the shape they actually form is visible, and the surface they
    lie on is drawn faintly behind them - which is what makes the empty stretches
    legible as parts of the surface no switching point can use rather than as
    nothing at all.

    Several heading bins share one lattice position; heading cannot be drawn here,
    so a position takes the best of them, exactly as the plan view does.
    """
    discretization = scene.config.discretization
    contour = _contour_for(scene, sensor_map)
    objective_of = _switching_objective_by_state(scene, sensor_map, response)
    if not objective_of:
        raise RuntimeError("no admissible switching state has a finite objective")

    best: dict[tuple[float, float, float], float] = {}
    for state_id, objective in objective_of.items():
        key = tuple(
            float(v) for v in scene.grid.position_map(scene.grid.decode(state_id))
        )
        if key not in best or objective < best[key]:
            best[key] = objective
    positions = np.asarray(list(best), dtype=float)
    objectives = np.asarray([best[tuple(p)] for p in positions], dtype=float)

    figure = go.Figure()
    for mesh in _terrain_mesh(scene.terrain):
        figure.add_trace(mesh)

    origin = contour.origin.as_array()
    fractions = np.linspace(0.0, 1.0, 180, endpoint=False)
    scales = np.linspace(
        discretization.switching_radial_min, discretization.switching_radial_max, 48,
    )
    directions = np.asarray(
        [contour.tangent_vector_at(float(v)) for v in fractions], dtype=float,
    )
    surface = origin[None, None, :] + scales[None, :, None] * directions[:, None, :]
    masked = surface.copy()
    masked[~_domain_mask(surface, scene)] = np.nan
    figure.add_trace(go.Surface(
        x=masked[:, :, 0], y=masked[:, :, 1], z=masked[:, :, 2],
        colorscale=[[0, INK_FAINT], [1, INK_FAINT]], showscale=False, opacity=0.18,
        name="LOS tangent surface", showlegend=True, hoverinfo="name",
    ))

    figure.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], mode="markers",
        name=f"admissible switching points ({len(positions):,})",
        marker={
            "size": 5.5, "color": objectives, "colorscale": "Viridis_r",
            # The tail is long - at 50 m the best switching point scores 0.308, the
            # median 0.325, and the worst 7.57 - so a scale that spanned it would
            # paint nine points out of ten the same colour.  Clipping at the ninth
            # decile keeps the structure readable, and the colourbar says so rather
            # than letting a saturated top look like a plateau.
            "cmin": float(objectives.min()),
            "cmax": float(np.percentile(objectives, 90)),
            "opacity": 0.95, "line": {"width": 0},
            "colorbar": {
                "title": "J_A<br><sub>clipped at p90</sub>", "len": 0.7, "x": 1.02,
            },
        },
        hovertemplate="x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<br>"
                      "J_A %{marker.color:.6f}<extra></extra>",
    ))

    index = int(np.argmin(objectives))
    figure.add_trace(go.Scatter3d(
        x=[positions[index, 0]], y=[positions[index, 1]], z=[positions[index, 2]],
        mode="markers", name=f"best switching point  J_A = {objectives[index]:.6f}",
        marker={"size": 10, "color": SWITCH, "symbol": "diamond",
                "line": {"color": "white", "width": 2}},
    ))
    for trace in _scene_markers(scene, sensor_map):
        figure.add_trace(trace)

    layout = _layout_3d(scene)
    layout["title"] = {
        "text": f"exact local SSE   J_A = {sse.attacker_objective:.6f}"
                f"   at d = {[float(v) for v in sensor_map]}",
        "x": 0.5, "xanchor": "center", "font": {"size": 15, "color": INK},
    }
    margin = dict(layout.get("margin") or {})
    margin["t"] = max(int(margin.get("t", 0)), 56)
    layout["margin"] = margin
    figure.update_layout(**layout)
    return _save(figure, output, stem)


def run_switching_map(
    resolution_m: float,
    *,
    stem: str | None = None,
    r_neighbor: int = 1,
    terrain_category: str = "centered_cube",
    output_directory: Path = OUTPUT,
) -> list[str]:
    """Just the switching-point objective map, at one resolution, in plan and in 3-D.

    Separate from ``run_exact_figures`` because the map is worth having at several
    lattices - the admissible switching set is only a couple of cells deep at 100 m
    and the shape of the landscape only emerges as the lattice refines - while the
    other six figures say the same thing at every resolution and are not worth the
    minutes a fine lattice costs.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    condition = ComputationCondition(
        spatial_resolution_m=resolution_m, r_neighbor=r_neighbor,
        terrain_category=terrain_category,
    )
    scene = build_scene(condition)
    print(f"condition: {condition.label}  terrain {terrain_category}", flush=True)
    sse = exact_local_sse(scene)
    sensor_map = tuple(
        sse.selected_sensor_map or scene.defender_grid.position(scene.seed_action_id)
    )
    response = exact_best_response(scene, sensor_map, keep_solution=True)
    print(f"  local SSE J_A={sse.attacker_objective} at d={list(sensor_map)}"
          f"  ({response.admissible_switch_states} admissible switching states)",
          flush=True)
    switch = scene.grid.decode(int(response.switching_state_id))
    position = scene.grid.position_map(switch)
    bounds = scene.grid.bounds
    at_edge = [
        name for name, touching in (
            ("ceiling", abs(position[2] - scene.grid.maximum_altitude_map) < 1e-9),
            ("y_max", abs(position[1] - bounds.y_max) < 1e-9),
            ("y_min", abs(position[1] - bounds.y_min) < 1e-9),
            ("x_min", abs(position[0] - bounds.x_min) < 1e-9),
        ) if touching
    ]
    print(f"  chosen switching point {[float(v) for v in position]}"
          f"  -> on the domain boundary at: {', '.join(at_edge) or 'nowhere'}",
          flush=True)
    base = stem or f"5_switching_point_objective_map_{resolution_m:g}m"
    written = [
        figure_5_switching_map(
            scene, sensor_map, response, sse, output_directory, stem=base,
        ),
        figure_5_switching_map_3d(
            scene, sensor_map, response, sse, output_directory, stem=f"{base}_3d",
        ),
    ]
    print("  wrote: " + ", ".join(written), flush=True)
    return written


def run_exact_figures(
    resolution_m: float = 100.0, r_neighbor: int = 1,
    output_directory: Path = OUTPUT,
) -> list[str]:
    output_directory.mkdir(parents=True, exist_ok=True)
    condition = ComputationCondition(
        spatial_resolution_m=resolution_m, r_neighbor=r_neighbor,
    )
    scene = build_scene(condition)
    print(f"condition: {condition.label}")

    sse = exact_local_sse(scene)
    sensor_map = tuple(
        sse.selected_sensor_map or scene.defender_grid.position(scene.seed_action_id)
    )
    response = exact_best_response(scene, sensor_map, keep_solution=True)
    print(f"  local SSE feasible={sse.feasible} action={sse.selected_action_id} "
          f"J_A={sse.attacker_objective} J_D={sse.detection_probability}")
    print(f"  switching point {response.switching_position_map}, "
          f"glide states {len(response.glide_state_ids)}")

    written = [
        figure_1_los_surface(scene, sensor_map, output_directory),
        figure_2_reachable_region(scene, sensor_map, response, output_directory),
        figure_3_cost_to_go(scene, sensor_map, response, output_directory),
        figure_4_solution(scene, sse, response, output_directory),
        _figure_4_projection(scene, sse, response, output_directory, plane="side"),
        _figure_4_projection(scene, sse, response, output_directory, plane="top"),
        figure_5_switching_map(scene, sensor_map, response, sse, output_directory),
    ]
    print("  wrote: " + ", ".join(written))
    return written


if __name__ == "__main__":
    resolution = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    run_exact_figures(resolution)
