"""Create full-SSE stepped-pyramid 3D HTML and cross-resolution reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import plotly.graph_objects as go

import project_paths
from P1b_Exact_Local_SSE import exact_best_response
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from los_geometry import LOSModel


DEFAULT_RESOLUTIONS = (10.0, 25.0, 50.0, 100.0)
TERRAIN = "stepped_pyramid"


def _tag(resolution_m: float) -> str:
    clean = f"{float(resolution_m):g}".replace(".", "p")
    return f"dx{clean}m_dpsi5deg"


def _relative_percent(approximate: float, exact: float) -> float | None:
    if abs(exact) <= 1.0e-15:
        return None
    return 100.0 * abs(approximate - exact) / abs(exact)


def _add_mesh(
    figure: go.Figure, mesh: Any, *, name: str, color: str, opacity: float,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        name=name, color=color, opacity=opacity, flatshading=True,
        hovertemplate=f"{name}<extra></extra>",
    ))


def _los_mesh(scene: Any, sensor: tuple[float, float, float]) -> Any:
    model = LOSModel(scene.terrain)
    sensor_point = scene.mission_for(sensor).sensor
    contour = model.trace_tangent_contour(
        sensor_point,
        probe_grid_size=scene.config.discretization.los_probe_grid_size,
        boundary_refinement_steps=(
            scene.config.discretization.los_boundary_refinement_steps
        ),
    )
    return model.build_tangent_surface(
        contour, display_extension_factor=2.2,
    ).display_mesh()


def _write_html(
    *, output: Path, resolution_m: float, scene: Any,
    exact_sensor: tuple[float, float, float], dqn_sensor: tuple[float, float, float],
    exact_positions: np.ndarray, dqn_positions: np.ndarray,
    exact_J_A: float, exact_J_D: float, dqn_J_A: float, dqn_J_D: float,
    exact_time: float, dqn_time: float,
) -> None:
    figure = go.Figure()
    _add_mesh(
        figure, scene.terrain.obstacle_mesh(), name="stepped-pyramid terrain",
        color="#777777", opacity=0.48,
    )
    _add_mesh(
        figure, _los_mesh(scene, exact_sensor), name="Bellman LOS tangent surface",
        color="#56B4E9", opacity=0.11,
    )
    _add_mesh(
        figure, _los_mesh(scene, dqn_sensor), name="DQN LOS tangent surface",
        color="#F0E442", opacity=0.09,
    )

    for method, positions, sensor, color, dash, symbol, J_A, J_D in (
        ("Bellman", exact_positions, exact_sensor, "#0072B2", "solid", "circle", exact_J_A, exact_J_D),
        ("DQN", dqn_positions, dqn_sensor, "#E69F00", "dash", "square", dqn_J_A, dqn_J_D),
    ):
        figure.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines+markers", name=f"{method} attacker trajectory",
            line={"color": color, "width": 7, "dash": dash},
            marker={"color": color, "size": 4, "symbol": symbol},
            hovertemplate=(
                f"{method}<br>J_A={J_A:.8f}<br>J_D={J_D:.8f}"
                "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
            ),
        ))
        figure.add_trace(go.Scatter3d(
            x=[sensor[0]], y=[sensor[1]], z=[sensor[2]],
            mode="markers+text", text=[f"{method} sensor"],
            textposition="top center", name=f"{method} final defender sensor",
            marker={"color": color, "size": 9, "symbol": "diamond",
                    "line": {"color": "black", "width": 1}},
            hovertemplate=(
                f"{method} final defender sensor"
                "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
            ),
        ))
        figure.add_trace(go.Scatter3d(
            x=[sensor[0], positions[0, 0]],
            y=[sensor[1], positions[0, 1]],
            z=[sensor[2], positions[0, 2]], mode="lines",
            line={"color": color, "width": 3, "dash": "dot"},
            name=f"{method} tangent ray", hoverinfo="skip", showlegend=False,
        ))

    figure.add_trace(go.Scatter3d(
        x=[exact_positions[-1, 0]], y=[exact_positions[-1, 1]],
        z=[exact_positions[-1, 2]], mode="markers+text", text=["goal"],
        textposition="top center", name="goal",
        marker={"color": "#009E73", "size": 9, "symbol": "diamond"},
        hovertemplate="goal<extra></extra>",
    ))

    J_A_percent = _relative_percent(dqn_J_A, exact_J_A)
    J_D_percent = _relative_percent(dqn_J_D, exact_J_D)
    speedup = exact_time / dqn_time
    figure.update_layout(
        title={
            "text": (
                f"{resolution_m:g} m stepped-pyramid full Local-SSE: Bellman vs DQN"
                f"<br><sup>Bellman: J_A={exact_J_A:.8f}, J_D={exact_J_D:.8f}, {exact_time:.3f} s; "
                f"DQN: J_A={dqn_J_A:.8f}, J_D={dqn_J_D:.8f}, {dqn_time:.3f} s"
                f"<br>|delta J_A|/|Bellman|={J_A_percent:.2f}%; "
                f"|delta J_D|/|Bellman|={J_D_percent:.2f}%; speedup={speedup:.3f}x</sup>"
            ),
            "x": 0.5,
        },
        scene={
            "xaxis": {"title": "x [map unit]"},
            "yaxis": {"title": "y [map unit]"},
            "zaxis": {"title": "z [map unit]"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 1.05}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 110},
        template="plotly_white",
        uirevision=f"full-sse-{resolution_m:g}m",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output, include_plotlyjs=True, full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        auto_open=False,
    )


def process_resolution(resolution_m: float, run_label: str) -> dict[str, Any]:
    condition_tag = _tag(resolution_m)
    full_sse_dir = (
        project_paths.CORE_DIR / "figure" / "phase_4_multi_terrain_dqn"
        / condition_tag / run_label / "full_sse"
    )
    comparison_path = full_sse_dir / "full_sse_stepped_pyramid_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    exact = comparison["exact_bellman"]
    dqn = comparison["trained_dqn"]

    condition = ComputationCondition(
        spatial_resolution_m=resolution_m,
        heading_spacing_deg=5.0,
        r_neighbor=1,
        terrain_category=TERRAIN,
    )
    scene = build_scene(condition)
    exact_sensor = tuple(float(value) for value in exact["selected_sensor_map"])
    dqn_sensor = tuple(float(value) for value in dqn["selected_sensor_map"])

    replay_started = perf_counter()
    exact_response = exact_best_response(scene, exact_sensor)
    exact_replay_sec = perf_counter() - replay_started
    if not exact_response.feasible:
        raise RuntimeError(f"exact replay is infeasible at {resolution_m:g} m")
    if not np.isclose(exact_response.attacker_objective, exact["J_A"], atol=1e-10):
        raise RuntimeError(f"exact J_A replay mismatch at {resolution_m:g} m")
    if not np.isclose(exact_response.detection_probability, exact["J_D"], atol=1e-10):
        raise RuntimeError(f"exact J_D replay mismatch at {resolution_m:g} m")

    dqn_record = next(
        item for item in dqn["evaluated"]
        if int(item["action_id"]) == int(dqn["selected_action_id"])
    )
    exact_problem = AttackerBRProblem(scene, exact_sensor)
    dqn_problem = AttackerBRProblem(scene, dqn_sensor)
    exact_positions = np.asarray([
        exact_problem.position_map(int(state_id))
        for state_id in exact_response.glide_state_ids
    ])
    dqn_positions = np.asarray([
        dqn_problem.position_map(int(state_id))
        for state_id in dqn_record["trajectory"]
    ])

    output_html = full_sse_dir / "full_sse_stepped_pyramid_3d.html"
    _write_html(
        output=output_html, resolution_m=resolution_m, scene=scene,
        exact_sensor=exact_sensor, dqn_sensor=dqn_sensor,
        exact_positions=exact_positions, dqn_positions=dqn_positions,
        exact_J_A=float(exact["J_A"]), exact_J_D=float(exact["J_D"]),
        dqn_J_A=float(dqn["predicted_J_A"]), dqn_J_D=float(dqn["predicted_J_D"]),
        exact_time=float(exact["wall_runtime_sec"]),
        dqn_time=float(dqn["wall_runtime_sec"]),
    )

    exact_J_A = float(exact["J_A"])
    exact_J_D = float(exact["J_D"])
    dqn_J_A = float(dqn["predicted_J_A"])
    dqn_J_D = float(dqn["predicted_J_D"])
    exact_time = float(exact["wall_runtime_sec"])
    dqn_time = float(dqn["wall_runtime_sec"])
    return {
        "spatial_resolution_m": resolution_m,
        "heading_spacing_deg": 5.0,
        "r_neighbor": 1,
        "terrain": TERRAIN,
        "bellman_sensor": list(exact_sensor),
        "dqn_sensor": list(dqn_sensor),
        "bellman_J_A": exact_J_A,
        "dqn_J_A": dqn_J_A,
        "J_A_absolute_difference": abs(dqn_J_A - exact_J_A),
        "J_A_relative_difference_percent": _relative_percent(dqn_J_A, exact_J_A),
        "bellman_J_D": exact_J_D,
        "dqn_J_D": dqn_J_D,
        "J_D_absolute_difference": abs(dqn_J_D - exact_J_D),
        "J_D_relative_difference_percent": _relative_percent(dqn_J_D, exact_J_D),
        "bellman_full_sse_sec": exact_time,
        "dqn_full_sse_sec": dqn_time,
        "bellman_over_dqn_speedup": exact_time / dqn_time,
        "bellman_defender_evaluations": int(exact["unique_defender_evaluations"]),
        "dqn_defender_evaluations": int(dqn["unique_defender_evaluations"]),
        "selected_defender_action_match": bool(
            comparison["comparison"]["selected_defender_action_match"]
        ),
        "exact_replay_for_plot_sec_excluded_from_benchmark": exact_replay_sec,
        "source": str(comparison_path).replace("\\", "/"),
        "interactive_3d_html": str(output_html).replace("\\", "/"),
    }


def write_aggregate(rows: list[dict[str, Any]]) -> Path:
    output_dir = (
        project_paths.CORE_DIR / "figure" / "phase_4_multi_terrain_dqn"
        / "stepped_pyramid_resolution_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "full_sse_resolution_comparison.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "full_sse_resolution_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "full_sse_resolution_comparison.md"
    lines = [
        "# Stepped-pyramid full Local-SSE resolution comparison",
        "",
        "All rows use heading spacing 5 deg and Chebyshev r=1. Full-SSE times "
        "exclude shared scene construction, checkpoint loading, and the exact "
        "trajectory replay used only to build the 3D HTML.",
        "",
        "| dx | Bellman J_A | DQN J_A | J_A diff | Bellman J_D | DQN J_D | J_D diff | Bellman (s) | DQN (s) | Speedup | 3D HTML |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| {row['spatial_resolution_m']:g} m | {row['bellman_J_A']:.8f} | "
        f"{row['dqn_J_A']:.8f} | {row['J_A_relative_difference_percent']:.2f}% | "
        f"{row['bellman_J_D']:.8f} | {row['dqn_J_D']:.8f} | "
        f"{row['J_D_relative_difference_percent']:.2f}% | "
        f"{row['bellman_full_sse_sec']:.3f} | {row['dqn_full_sse_sec']:.3f} | "
        f"{row['bellman_over_dqn_speedup']:.3f}x | "
        f"[open]({row['interactive_3d_html']}) |"
        for row in sorted(rows, key=lambda item: item["spatial_resolution_m"])
    )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS,
    )
    parser.add_argument("--run-label", default="official_batched_v1")
    args = parser.parse_args()
    rows = [
        process_resolution(float(resolution), args.run_label)
        for resolution in args.resolutions
    ]
    output = write_aggregate(rows)
    print(output)


if __name__ == "__main__":
    main()
