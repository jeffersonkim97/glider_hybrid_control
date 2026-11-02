"""Render Phase-6 final Bellman/DQN Local-SSE solutions as interactive 3D HTML."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go

import project_paths
from los_geometry import LOSModel
from phase4_extended_dataset import build_random_box_terrain
from phase4_extended_scene import build_extended_scene
from phase4_v2_run_a import RunAGoalReachableProblem
from phase5_g2_final_id_test import (
    _add_terrain,
    configure_final_terrain,
    freeze_final_id_000,
)


TERRAIN_INDEX = 1
RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
SOURCE_ROOT = (
    project_paths.DQN_DIR / "runs" / "phase6_g2_local_sse"
    / f"final_id_{TERRAIN_INDEX:03d}"
)
OUTPUT_DIR = SOURCE_ROOT / "paper_closeout_v1"


def _result(dx_m: float) -> dict[str, Any]:
    path = (
        SOURCE_ROOT / f"dx{dx_m:g}m_dpsi5deg_r1"
        / "phase6_g2_local_sse_pilot.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _dqn_final_record(result: dict[str, Any]) -> dict[str, Any]:
    final_id = int(
        result["frozen_dqn_local_sse"]["search"]
        ["final_local_sse_action_id"]
    )
    matches = [
        item for item in result["frozen_dqn_local_sse"]["evaluated"]
        if int(item["action_id"]) == final_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing DQN final response for action {final_id}")
    return matches[0]


def _positions(
    problem: RunAGoalReachableProblem, state_ids: Iterable[int], scale: float,
) -> np.ndarray:
    values = [problem.position_map(int(state_id)) for state_id in state_ids]
    return np.asarray(values, dtype=float) * scale


def _add_los_surface(
    figure: go.Figure, scene: Any, sensor_map: tuple[float, float, float],
    *, name: str, color: str, scale: float,
) -> None:
    mission = scene.mission_for(sensor_map)
    model = LOSModel(scene.terrain)
    contour = model.trace_tangent_contour(
        mission.sensor,
        probe_grid_size=scene.config.discretization.los_probe_grid_size,
        boundary_refinement_steps=(
            scene.config.discretization.los_boundary_refinement_steps
        ),
    )
    mesh = model.build_tangent_surface(
        contour, display_extension_factor=2.2,
    ).display_mesh()
    vertices = np.asarray(mesh.vertices, dtype=float) * scale
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=color, opacity=0.12, flatshading=True,
        name=f"{name} LOS tangent surface",
        hovertemplate=f"{name} LOS tangent surface<extra></extra>",
    ))


def _add_solution(
    figure: go.Figure, *, method: str, color: str, symbol: str,
    sensor_map: tuple[float, float, float], trajectory: Iterable[int],
    problem: RunAGoalReachableProblem, start_m: np.ndarray, scale: float,
) -> None:
    sensor_m = np.asarray(sensor_map, dtype=float) * scale
    positions = _positions(problem, trajectory, scale)
    if not len(positions):
        raise RuntimeError(f"{method} final trajectory is empty")
    switching = positions[0]
    figure.add_trace(go.Scatter3d(
        x=[sensor_m[0]], y=[sensor_m[1]], z=[sensor_m[2]],
        mode="markers+text", text=[f"{method} SENSOR"],
        textposition="top center",
        marker={"size": 8, "color": color, "symbol": symbol},
        name=f"{method} final sensor",
        hovertemplate=(
            f"{method} final sensor<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m"
            "<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[start_m[0], switching[0]], y=[start_m[1], switching[1]],
        z=[start_m[2], switching[2]], mode="lines",
        line={"color": color, "width": 7, "dash": "dot"},
        name=f"{method} powered segment",
    ))
    figure.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="lines+markers", line={"color": color, "width": 7},
        marker={"size": 3, "color": color},
        name=f"{method} glide trajectory",
        hovertemplate=(
            f"{method}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m"
            "<br>z=%{z:.1f} m<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=[switching[0]], y=[switching[1]], z=[switching[2]],
        mode="markers+text", text=[f"{method} SWITCH"],
        textposition="top center",
        marker={
            "size": 10, "color": color, "symbol": symbol,
            "line": {"color": "white", "width": 1},
        },
        name=f"{method} switching point",
    ))
    figure.add_trace(go.Scatter3d(
        x=[sensor_m[0], switching[0]], y=[sensor_m[1], switching[1]],
        z=[sensor_m[2], switching[2]], mode="lines",
        line={"color": color, "width": 2, "dash": "dash"},
        name=f"{method} tangent ray", hoverinfo="skip",
    ))


def render(dx_m: float) -> Path:
    configure_final_terrain(TERRAIN_INDEX)
    _, scenarios, terrain_spec, config = freeze_final_id_000()
    mixed = [item for item in scenarios if item.visibility_class == "mixed"]
    scenario = mixed[1]
    result = _result(dx_m)
    dqn_record = _dqn_final_record(result)
    exact_record = result["exact_bellman_local_sse"]["final_response"]
    exact_sensor = tuple(float(item) for item in exact_record["sensor_map"])
    dqn_sensor = tuple(float(item) for item in dqn_record["sensor_map"])
    scene = build_extended_scene(
        scenario,
        terrain_spec,
        spatial_resolution_m=dx_m,
        heading_spacing_deg=5.0,
        r_neighbor=1,
        pilot_config=config.pilot,
    )
    exact_problem = RunAGoalReachableProblem(scene, exact_sensor)
    dqn_problem = RunAGoalReachableProblem(scene, dqn_sensor)
    scale = float(config.pilot.meters_per_map_unit)
    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    figure = go.Figure()
    _add_terrain(figure, terrain, meters_per_unit=scale)
    _add_los_surface(
        figure, scene, exact_sensor, name="Bellman", color="#D55E00", scale=scale,
    )
    if not np.allclose(exact_sensor, dqn_sensor):
        _add_los_surface(
            figure, scene, dqn_sensor, name="DQN", color="#0072B2", scale=scale,
        )
    start_m = np.asarray(scenario.start_map, dtype=float) * scale
    goal_m = np.asarray(scenario.goal_map, dtype=float) * scale
    figure.add_trace(go.Scatter3d(
        x=[start_m[0]], y=[start_m[1]], z=[start_m[2]],
        mode="markers+text", text=["START"], textposition="top center",
        marker={"size": 8, "color": "black", "symbol": "diamond"}, name="start",
    ))
    figure.add_trace(go.Scatter3d(
        x=[goal_m[0]], y=[goal_m[1]], z=[goal_m[2]],
        mode="markers+text", text=["GOAL"], textposition="top center",
        marker={"size": 9, "color": "#9467bd", "symbol": "diamond"}, name="goal",
    ))
    _add_solution(
        figure, method="Bellman", color="#D55E00", symbol="circle",
        sensor_map=exact_sensor, trajectory=exact_record["trajectory"],
        problem=exact_problem, start_m=start_m, scale=scale,
    )
    _add_solution(
        figure, method="DQN", color="#0072B2", symbol="square",
        sensor_map=dqn_sensor, trajectory=dqn_record["trajectory"],
        problem=dqn_problem, start_m=start_m, scale=scale,
    )
    certificate = result["comparison"]["dqn_selected_sensor_exact_neighbor_audit"][
        "exact_local_sse_verified"
    ]
    x0, x1, y0, y1 = config.pilot.map_bounds
    figure.update_layout(
        title=(
            f"Phase 6 final Local-SSE solutions — dx={dx_m:g} m"
            f"<br><sup>Bellman sensor={exact_sensor[:2]}; DQN sensor={dqn_sensor[:2]}; "
            f"DQN exact-local certificate={'PASS' if certificate else 'FAIL'}</sup>"
        ),
        width=1300, height=900, margin={"l": 0, "r": 0, "t": 95, "b": 0},
        legend={"orientation": "h", "y": 0.01, "x": 0.01},
        scene={
            "xaxis": {"range": [x0 * scale, x1 * scale], "title": "x [m]"},
            "yaxis": {"range": [y0 * scale, y1 * scale], "title": "y [m]"},
            "zaxis": {"range": [0.0, 520.0], "title": "z [m]"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.25, "y": 1.0, "z": 0.45},
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.0}},
        },
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"phase6_final_solutions_dx{dx_m:g}m_3d.html"
    figure.write_html(
        output, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    exact_problem.clear_runtime_caches()
    dqn_problem.clear_runtime_caches()
    del exact_problem, dqn_problem, scene, figure
    gc.collect()
    return output


def main() -> None:
    for dx in RESOLUTIONS_M:
        print(render(dx), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["render"]
