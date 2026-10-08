"""Create per-terrain 3D audits for the Phase 16.4 representative paths.

This is a post-processing command.  It reads the completed official run and
does not train or modify a DQN.  Each plot includes the actual three-dimensional
LOS tangent surface so that the lattice-snapped Bellman and DQN switching states
can be checked without relying on the top-down Phase 16.4 summary figure.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import project_paths
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from detection_hazard import hazard_to_detection_probability
from los_geometry import LOSModel


DEFAULT_RUN_LABEL = "official_batched_v1"
DEFAULT_SENSOR = (5.0, 0.0, 0.0)
STEPPED_PYRAMID = "stepped_pyramid"


def _condition_tag(spatial_resolution_m: float, heading_spacing_deg: float) -> str:
    def clean(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")

    return f"dx{clean(spatial_resolution_m)}m_dpsi{clean(heading_spacing_deg)}deg"


def _sensor_tag(sensor: tuple[float, float, float]) -> str:
    def clean(value: float) -> str:
        sign = "p" if value >= 0.0 else "m"
        return sign + f"{abs(float(value)):g}".replace(".", "p")

    return "_".join(f"{axis}{clean(value)}" for axis, value in zip("xyz", sensor))


def _bellman_display_positions(
    terrain_category: str,
    bellman_positions: np.ndarray,
    dqn_positions: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Put a symmetric stepped-pyramid Bellman BR on the DQN side for display."""

    displayed = np.array(bellman_positions, dtype=float, copy=True)
    if (
        terrain_category == STEPPED_PYRAMID
        and displayed[0, 1] * float(dqn_positions[0, 1]) < 0.0
    ):
        displayed[:, 1] *= -1.0
        return displayed, "reflect_y_about_0"
    return displayed, "identity"


def _load_official_records(
    output_dir: Path, sensor: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_manifest = json.loads(
        (output_dir / "phase4_generalized_model_manifest.json").read_text(encoding="utf-8")
    )
    evaluation_history = json.loads(
        (output_dir / "phase4_evaluation_history.json").read_text(encoding="utf-8")
    )
    selected_seed = int(model_manifest["selected_seed"])
    seed_record = next(
        item for item in evaluation_history["seeds"]
        if int(item["seed"]) == selected_seed
    )
    records = [
        item for item in seed_record["official_best_records"]
        if np.allclose(item["sensor_map"], sensor, rtol=0.0, atol=1.0e-12)
    ]
    if not records:
        raise RuntimeError(f"no official representative records for sensor {sensor}")
    return records, model_manifest


def _distance_to_tangent_surface(
    point: np.ndarray, contour: Any, *, samples: int = 8192,
) -> float:
    """Approximate distance to the continuous ruled surface in map units.

    For every densely sampled contour direction, the nearest point on its
    nonnegative ray is analytic.  The contour interpolation is continuous, so
    8192 angular samples make the remaining numerical error negligible relative
    to a 0.25-map-unit (25 m) lattice cell.
    """

    fractions = np.linspace(0.0, 1.0, samples, endpoint=not contour.closed)
    directions = np.asarray([
        contour.tangent_vector_at(float(fraction)) for fraction in fractions
    ])
    origin = contour.origin.as_array()
    offset = np.asarray(point, dtype=float) - origin
    scale = np.maximum(
        0.0,
        np.einsum("ij,j->i", directions, offset)
        / np.einsum("ij,ij->i", directions, directions),
    )
    nearest = origin + scale[:, None] * directions
    return float(np.min(np.linalg.norm(nearest - point, axis=1)))


def _add_mesh(
    ax: Any, mesh: Any, *, facecolor: str, alpha: float,
    edgecolor: str, linewidth: float, label: str | None = None,
) -> None:
    faces = np.asarray(mesh.vertices)[np.asarray(mesh.triangles, dtype=np.int64)]
    collection = Poly3DCollection(
        faces, facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha,
    )
    if label is not None:
        collection.set_label(label)
    ax.add_collection3d(collection)


def _draw_case(ax: Any, case: dict[str, Any], *, compact: bool = False) -> None:
    _add_mesh(
        ax, case["terrain_mesh"], facecolor="#8f8f8f", alpha=0.48,
        edgecolor="#4d4d4d", linewidth=0.35, label="terrain",
    )
    _add_mesh(
        ax, case["los_mesh"], facecolor="#56B4E9", alpha=0.12,
        edgecolor="none", linewidth=0.0, label="LOS tangent surface",
    )

    bellman = case["bellman_positions"]
    dqn = case["dqn_positions"]
    sensor = case["sensor"]
    bellman_reflected = case["bellman_display_transform"] == "reflect_y_about_0"
    bellman_glide_label = (
        "Bellman glide (y-reflected)" if bellman_reflected else "Bellman glide"
    )
    bellman_switch_label = (
        "Bellman switch (y-reflected)" if bellman_reflected else "Bellman switch"
    )
    ax.plot(
        bellman[:, 0], bellman[:, 1], bellman[:, 2], "o-",
        color="#0072B2", linewidth=2.0, markersize=3.5,
        label=bellman_glide_label,
    )
    ax.plot(
        dqn[:, 0], dqn[:, 1], dqn[:, 2], "s--",
        color="#E69F00", linewidth=2.0, markersize=3.5, label="DQN glide",
    )
    ax.plot(
        [sensor[0], bellman[0, 0]], [sensor[1], bellman[0, 1]],
        [sensor[2], bellman[0, 2]], color="#0072B2", linestyle=":",
        linewidth=1.2, alpha=0.9,
    )
    ax.plot(
        [sensor[0], dqn[0, 0]], [sensor[1], dqn[0, 1]],
        [sensor[2], dqn[0, 2]], color="#E69F00", linestyle=":",
        linewidth=1.2, alpha=0.9,
    )
    ax.scatter(
        *sensor, marker="^", color="#CC79A7", edgecolors="black",
        linewidths=0.5, s=65, label="sensor", depthshade=False,
    )
    ax.scatter(
        *bellman[0], marker="o", color="#0072B2", edgecolors="white",
        linewidths=0.8, s=75, label=bellman_switch_label, depthshade=False,
    )
    ax.scatter(
        *dqn[0], marker="s", color="#E69F00", edgecolors="white",
        linewidths=0.8, s=75, label="DQN switch", depthshade=False,
    )
    ax.text(
        bellman[0, 0], bellman[0, 1], bellman[0, 2] + 0.15, "B",
        color="#00568A", fontsize=8, fontweight="bold",
    )
    ax.text(
        dqn[0, 0], dqn[0, 1], dqn[0, 2] + 0.15, "D",
        color="#A66F00", fontsize=8, fontweight="bold",
    )
    ax.scatter(
        *bellman[-1], marker="*", color="#009E73", edgecolors="black",
        linewidths=0.35, s=90, label="goal", depthshade=False,
    )

    ax.set_xlim(-3.5, 8.0)
    ax.set_ylim(-5.5, 5.5)
    ax.set_zlim(0.0, 5.25)
    ax.set_xlabel("x [map unit]")
    ax.set_ylabel("y [map unit]")
    ax.set_zlabel("z [map unit]")
    ax.set_box_aspect((16.0, 12.0, 6.5))
    ax.view_init(elev=25.0, azim=-58.0)
    ax.grid(alpha=0.25)
    diagnostics = case["diagnostics"]
    title = (
        f"{case['terrain_category']} | sensor (5, 0, 0)\n"
        f"J_A error={100.0 * case['relative_error']:.2f}% | "
        f"surface gap: Bellman={diagnostics['bellman']['los_surface_distance_m']:.2f} m, "
        f"DQN={diagnostics['dqn']['los_surface_distance_m']:.2f} m"
    )
    if bellman_reflected:
        title += "\nBellman shown with y -> -y symmetry"
    ax.set_title(title, fontsize=9 if compact else 11)
    ax.legend(loc="upper left", fontsize=6.5 if compact else 8)


def _write_interactive_case(case: dict[str, Any], output: Path) -> None:
    """Write one self-contained Plotly scene with orbit, zoom, and hover."""

    import plotly.graph_objects as go

    terrain = case["terrain_mesh"]
    terrain_vertices = np.asarray(terrain.vertices)
    terrain_triangles = np.asarray(terrain.triangles, dtype=np.int64)
    los = case["los_mesh"]
    los_vertices = np.asarray(los.vertices)
    los_triangles = np.asarray(los.triangles, dtype=np.int64)
    bellman = case["bellman_positions"]
    dqn = case["dqn_positions"]
    sensor = case["sensor"]
    diagnostics = case["diagnostics"]
    bellman_reflected = case["bellman_display_transform"] == "reflect_y_about_0"
    bellman_glide_label = (
        "Bellman glide (y-reflected)" if bellman_reflected else "Bellman glide"
    )

    figure = go.Figure()
    figure.add_trace(go.Mesh3d(
        x=terrain_vertices[:, 0], y=terrain_vertices[:, 1], z=terrain_vertices[:, 2],
        i=terrain_triangles[:, 0], j=terrain_triangles[:, 1], k=terrain_triangles[:, 2],
        color="#777777", opacity=0.48, flatshading=True, name="terrain",
        hovertemplate="terrain<extra></extra>",
    ))
    figure.add_trace(go.Mesh3d(
        x=los_vertices[:, 0], y=los_vertices[:, 1], z=los_vertices[:, 2],
        i=los_triangles[:, 0], j=los_triangles[:, 1], k=los_triangles[:, 2],
        color="#56B4E9", opacity=0.17, flatshading=True,
        name="LOS tangent surface",
        hovertemplate="LOS tangent surface<extra></extra>",
    ))
    figure.add_trace(go.Scatter3d(
        x=bellman[:, 0], y=bellman[:, 1], z=bellman[:, 2], mode="lines+markers",
        line={"color": "#0072B2", "width": 6},
        marker={"color": "#0072B2", "size": 4, "symbol": "circle"},
        name=bellman_glide_label,
        hovertemplate=(
            "Bellman" + (" (y-reflected)" if bellman_reflected else "")
            + f"<br>J_A={case['bellman_J_A']:.8f}"
            + f"<br>J_D={case['bellman_J_D']:.8f}"
            + "<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter3d(
        x=dqn[:, 0], y=dqn[:, 1], z=dqn[:, 2], mode="lines+markers",
        line={"color": "#E69F00", "width": 6, "dash": "dash"},
        marker={"color": "#E69F00", "size": 4, "symbol": "square"},
        name="DQN glide",
        hovertemplate=(
            f"DQN<br>J_A={case['dqn_J_A']:.8f}"
            f"<br>J_D={case['dqn_J_D']:.8f}"
            "<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"
        ),
    ))
    for name, point, color in (
        ("Bellman tangent ray", bellman[0], "#0072B2"),
        ("DQN tangent ray", dqn[0], "#E69F00"),
    ):
        figure.add_trace(go.Scatter3d(
            x=[sensor[0], point[0]], y=[sensor[1], point[1]],
            z=[sensor[2], point[2]], mode="lines",
            line={"color": color, "width": 3, "dash": "dot"},
            name=name, showlegend=False,
            hoverinfo="skip",
        ))
    figure.add_trace(go.Scatter3d(
        x=[sensor[0]], y=[sensor[1]], z=[sensor[2]], mode="markers+text",
        marker={"color": "#CC79A7", "size": 8, "symbol": "diamond",
                "line": {"color": "black", "width": 1}},
        text=["sensor"], textposition="top center", name="sensor",
        hovertemplate="sensor<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
    ))
    for method, point, color, symbol, short in (
        ("bellman", bellman[0], "#0072B2", "circle", "B"),
        ("dqn", dqn[0], "#E69F00", "square", "D"),
    ):
        detail = diagnostics[method]
        figure.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers+text",
            marker={"color": color, "size": 9, "symbol": symbol,
                    "line": {"color": "white", "width": 1}},
            text=[short], textposition="top center",
            name=f"{method.capitalize()} switch",
            customdata=[[detail["switching_state_id"], detail["los_surface_distance_m"]]],
            hovertemplate=(
                f"{method.capitalize()} switch<br>state=%{{customdata[0]}}"
                "<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}"
                "<br>LOS surface gap=%{customdata[1]:.3f} m<extra></extra>"
            ),
        ))
    figure.add_trace(go.Scatter3d(
        x=[bellman[-1, 0]], y=[bellman[-1, 1]], z=[bellman[-1, 2]],
        mode="markers+text", marker={"color": "#009E73", "size": 8, "symbol": "diamond",
        }, text=["goal"], textposition="top center", name="goal",
        hovertemplate="goal<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
    ))

    figure.update_layout(
        title={
            "text": (
                f"{case['terrain_category']} - Bellman vs trained DQN"
                f"<br><sup>sensor (5,0,0); "
                f"Bellman J_A={case['bellman_J_A']:.8f}, J_D={case['bellman_J_D']:.8f}; "
                f"DQN J_A={case['dqn_J_A']:.8f}, J_D={case['dqn_J_D']:.8f}"
                f"<br>J_A error={100.0 * case['relative_error']:.2f}%; "
                f"|delta J_D|={abs(case['dqn_J_D'] - case['bellman_J_D']):.8f}; "
                "surface gap: "
                f"Bellman={diagnostics['bellman']['los_surface_distance_m']:.2f} m, "
                f"DQN={diagnostics['dqn']['los_surface_distance_m']:.2f} m"
                + ("; Bellman shown with y -> -y symmetry" if bellman_reflected else "")
                + "</sup>"
            ),
            "x": 0.5,
        },
        scene={
            "xaxis": {"title": "x [map unit]", "range": [-3.5, 8.0]},
            "yaxis": {"title": "y [map unit]", "range": [-5.5, 5.5]},
            "zaxis": {"title": "z [map unit]", "range": [0.0, 5.25]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.6, "y": 1.2, "z": 0.65},
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 1.05}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 85},
        template="plotly_white",
        uirevision=case["terrain_category"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output, include_plotlyjs=True, full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        auto_open=False,
    )


def _write_runtime_tables(
    cases: list[dict[str, Any]], output_dir: Path, *, selected_seed: int,
    selected_episode: int,
) -> tuple[Path, Path]:
    rows = []
    for case in cases:
        bellman = float(case["bellman_runtime_sec"])
        dqn = float(case["dqn_inference_runtime_sec"])
        rows.append({
            "terrain": case["terrain_category"],
            "split": case["split"],
            "bellman_sec": bellman,
            "trained_dqn_sec": dqn,
            "bellman_over_dqn_speedup": bellman / dqn,
            "switching_candidates": int(case["dqn_candidate_count"]),
            "relative_J_A_error_percent": 100.0 * float(case["relative_error"]),
        })
    csv_path = output_dir / "phase4_representative_trajectory_runtime.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "phase4_representative_trajectory_runtime.md"
    lines = [
        "# Phase 16.4 representative attacker-trajectory runtime",
        "",
        f"Official generalized model: seed {selected_seed}, episode "
        f"{selected_episode:,}. Times cover one full "
        "attacker best-response query at sensor `(5, 0, 0)`: Bellman solves the "
        "exact dynamic program; trained DQN rolls out and scores every admissible "
        "switching candidate. Plot rendering and DQN training are excluded.",
        "",
        "| Terrain | Split | Bellman (s) | Trained DQN (s) | Speedup | Candidates | J_A error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row['terrain']}` | `{row['split']}` | {row['bellman_sec']:.3f} | "
        f"{row['trained_dqn_sec']:.3f} | {row['bellman_over_dqn_speedup']:.2f}x | "
        f"{row['switching_candidates']} | {row['relative_J_A_error_percent']:.2f}% |"
        for row in rows
    )
    lines.extend([
        "",
        f"Arithmetic mean | Bellman `{np.mean([row['bellman_sec'] for row in rows]):.3f} s`; "
        f"trained DQN `{np.mean([row['trained_dqn_sec'] for row in rows]):.3f} s`.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


def _write_payoff_tables(
    cases: list[dict[str, Any]], output_dir: Path, *, selected_seed: int,
    selected_episode: int,
) -> tuple[Path, Path]:
    rows = [
        {
            "terrain": case["terrain_category"],
            "split": case["split"],
            "sensor": "(5, 0, 0)",
            "bellman_J_A": float(case["bellman_J_A"]),
            "dqn_J_A": float(case["dqn_J_A"]),
            "relative_J_A_error_percent": 100.0 * float(case["relative_error"]),
            "bellman_J_D": float(case["bellman_J_D"]),
            "dqn_J_D": float(case["dqn_J_D"]),
            "absolute_J_D_difference": abs(
                float(case["dqn_J_D"]) - float(case["bellman_J_D"])
            ),
        }
        for case in cases
    ]
    csv_path = output_dir / "phase4_representative_J_A_J_D.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "phase4_representative_J_A_J_D.md"
    lines = [
        "# Phase 16.4 representative J_A and J_D",
        "",
        f"Official generalized model: seed {selected_seed}, episode "
        f"{selected_episode:,}. Sensor: `(5, 0, 0)`. `J_D` is the authoritative "
        "whole glide-path detection probability `1 - exp(-cumulative hazard)`.",
        "",
        "| Terrain | Split | Bellman J_A | DQN J_A | J_A error | Bellman J_D | DQN J_D | |delta J_D| |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row['terrain']}` | `{row['split']}` | {row['bellman_J_A']:.8f} | "
        f"{row['dqn_J_A']:.8f} | {row['relative_J_A_error_percent']:.2f}% | "
        f"{row['bellman_J_D']:.8f} | {row['dqn_J_D']:.8f} | "
        f"{row['absolute_J_D_difference']:.8f} |"
        for row in rows
    )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


def create_plots(
    *, spatial_resolution_m: float = 25.0, heading_spacing_deg: float = 5.0,
    run_label: str = DEFAULT_RUN_LABEL, sensor: tuple[float, float, float] = DEFAULT_SENSOR,
    write_png: bool = True, write_html: bool = False,
) -> dict[str, Any]:
    condition_tag = _condition_tag(spatial_resolution_m, heading_spacing_deg)
    output_dir = (
        project_paths.CORE_DIR / "figure" / "phase_4_multi_terrain_dqn"
        / condition_tag / run_label
    )
    records, model_manifest = _load_official_records(output_dir, sensor)
    condition_data = model_manifest["condition"]
    meters_per_map_unit: float | None = None
    cases: list[dict[str, Any]] = []

    for record in records:
        terrain_category = str(record["terrain_category"])
        condition = ComputationCondition(
            spatial_resolution_m=float(condition_data["spatial_resolution_m"]),
            heading_spacing_deg=float(condition_data["heading_spacing_deg"]),
            r_neighbor=int(condition_data["r_neighbor"]),
            terrain_category=terrain_category,
            hazard_weight=condition_data["hazard_weight"],
            time_weight=condition_data["time_weight"],
            defender_goal_margin_map=float(condition_data["defender_goal_margin_map"]),
        )
        scene = build_scene(condition)
        problem = AttackerBRProblem(scene, sensor)
        candidates = set(problem.switching_state_ids())
        mission = scene.mission_for(sensor)
        los_model = LOSModel(scene.terrain)
        contour = los_model.trace_tangent_contour(
            mission.sensor,
            probe_grid_size=scene.config.discretization.los_probe_grid_size,
            boundary_refinement_steps=(
                scene.config.discretization.los_boundary_refinement_steps
            ),
        )
        los_surface = los_model.build_tangent_surface(
            contour, display_extension_factor=2.2,
        )
        bellman_ids = tuple(int(value) for value in record["bellman_trajectory"])
        dqn_ids = tuple(int(value) for value in record["dqn"]["trajectory"])
        bellman_scored = problem.evaluate_trajectory(bellman_ids)
        dqn_scored = problem.evaluate_trajectory(dqn_ids)
        bellman_J_D = hazard_to_detection_probability(
            bellman_scored.cumulative_hazard
        )
        dqn_J_D = hazard_to_detection_probability(dqn_scored.cumulative_hazard)
        bellman_positions_original = np.asarray([
            problem.position_map(state_id) for state_id in bellman_ids
        ])
        dqn_positions = np.asarray([
            problem.position_map(state_id) for state_id in dqn_ids
        ])
        if not len(bellman_positions_original) or not len(dqn_positions):
            raise RuntimeError(f"empty representative trajectory for {terrain_category}")
        bellman_positions, bellman_display_transform = _bellman_display_positions(
            terrain_category, bellman_positions_original, dqn_positions,
        )
        scale = float(scene.config.physical_scale.meters_per_map_unit)
        meters_per_map_unit = scale

        diagnostics: dict[str, Any] = {}
        for method, state_id, position in (
            ("bellman", bellman_ids[0], bellman_positions_original[0]),
            ("dqn", dqn_ids[0], dqn_positions[0]),
        ):
            gap_map = _distance_to_tangent_surface(position, contour)
            diagnostics[method] = {
                "switching_state_id": state_id,
                "position_map": position.tolist(),
                "is_admissible_switching_candidate": state_id in candidates,
                "los_surface_distance_map_unit": gap_map,
                "los_surface_distance_m": gap_map * scale,
            }
            if state_id not in candidates:
                raise RuntimeError(
                    f"{terrain_category} {method} start is not an admissible switch"
                )

        diagnostics["bellman"]["display_position_map"] = bellman_positions[0].tolist()
        diagnostics["bellman"]["display_transform"] = bellman_display_transform

        case = {
            "terrain_category": terrain_category,
            "split": str(record["split"]),
            "sensor": np.asarray(sensor, dtype=float),
            "relative_error": float(record["relative_error"]),
            "bellman_J_A": float(record["bellman_J_A"]),
            "dqn_J_A": float(record["dqn"]["attacker_objective"]),
            "bellman_J_D": bellman_J_D,
            "dqn_J_D": dqn_J_D,
            "bellman_runtime_sec": float(record["bellman_runtime_sec"]),
            "dqn_inference_runtime_sec": float(record["dqn"]["inference_runtime_sec"]),
            "dqn_candidate_count": int(record["dqn"]["candidate_count"]),
            "bellman_positions": bellman_positions,
            "bellman_display_transform": bellman_display_transform,
            "dqn_positions": dqn_positions,
            "terrain_mesh": scene.terrain.obstacle_mesh(),
            "los_mesh": los_surface.display_mesh(),
            "diagnostics": diagnostics,
        }
        cases.append(case)

        if write_png:
            figure = plt.figure(figsize=(11.5, 8.0))
            axis = figure.add_subplot(111, projection="3d")
            _draw_case(axis, case)
            figure.tight_layout()
            figure.savefig(
                output_dir
                / f"phase4_3d_{terrain_category}_sensor_{_sensor_tag(sensor)}.png",
                dpi=190, bbox_inches="tight",
            )
            plt.close(figure)
        if write_html:
            _write_interactive_case(
                case,
                output_dir
                / f"phase4_3d_{terrain_category}_sensor_{_sensor_tag(sensor)}.html",
            )
        del problem, scene
        gc.collect()

    gallery_path = output_dir / "phase4_representative_trajectories_3d.png"
    if write_png:
        gallery = plt.figure(figsize=(18.0, 11.0))
        for index, case in enumerate(cases, start=1):
            axis = gallery.add_subplot(2, 3, index, projection="3d")
            _draw_case(axis, case, compact=True)
        gallery.suptitle(
            "Phase 16.4 representative trajectories on the 3D LOS tangent surface",
            fontsize=15,
        )
        gallery.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        gallery.savefig(gallery_path, dpi=170, bbox_inches="tight")
        plt.close(gallery)

    runtime_csv, runtime_markdown = _write_runtime_tables(
        cases,
        output_dir,
        selected_seed=int(model_manifest["selected_seed"]),
        selected_episode=int(model_manifest["selected_episode"]),
    )
    payoff_csv, payoff_markdown = _write_payoff_tables(
        cases,
        output_dir,
        selected_seed=int(model_manifest["selected_seed"]),
        selected_episode=int(model_manifest["selected_episode"]),
    )

    report = {
        "purpose": "3D audit of representative switching states and LOS tangent surface",
        "source_run": run_label,
        "selected_seed": int(model_manifest["selected_seed"]),
        "selected_episode": int(model_manifest["selected_episode"]),
        "condition": condition_data,
        "sensor_map": list(sensor),
        "meters_per_map_unit": meters_per_map_unit,
        "interpretation": (
            "Switching candidates originate on the continuous LOS tangent surface and "
            "are then snapped to the Bellman lattice; a nonzero plotted gap up to the "
            "spatial discretization scale is therefore expected. For stepped_pyramid, "
            "the plotted Bellman trajectory may be reflected across y=0 to place its "
            "equivalent symmetric best response on the same side as DQN; stored solver "
            "coordinates, objective values, and state IDs remain unchanged."
        ),
        "cases": [
            {
                "terrain_category": case["terrain_category"],
                "relative_J_A_error": case["relative_error"],
                "bellman_J_A": case["bellman_J_A"],
                "dqn_J_A": case["dqn_J_A"],
                "bellman_J_D": case["bellman_J_D"],
                "dqn_J_D": case["dqn_J_D"],
                "absolute_J_D_difference": abs(
                    case["dqn_J_D"] - case["bellman_J_D"]
                ),
                "bellman_runtime_sec": case["bellman_runtime_sec"],
                "trained_dqn_inference_runtime_sec": case["dqn_inference_runtime_sec"],
                "bellman_over_dqn_speedup": (
                    case["bellman_runtime_sec"] / case["dqn_inference_runtime_sec"]
                ),
                "bellman_display_transform": case["bellman_display_transform"],
                **case["diagnostics"],
                "plot": str(
                    output_dir
                    / f"phase4_3d_{case['terrain_category']}_sensor_{_sensor_tag(sensor)}.png"
                ).replace("\\", "/"),
                "interactive_html": str(
                    output_dir
                    / f"phase4_3d_{case['terrain_category']}_sensor_{_sensor_tag(sensor)}.html"
                ).replace("\\", "/") if write_html else None,
            }
            for case in cases
        ],
        "gallery": str(gallery_path).replace("\\", "/"),
        "runtime_table_csv": str(runtime_csv).replace("\\", "/"),
        "runtime_table_markdown": str(runtime_markdown).replace("\\", "/"),
        "payoff_table_csv": str(payoff_csv).replace("\\", "/"),
        "payoff_table_markdown": str(payoff_markdown).replace("\\", "/"),
    }
    report_path = output_dir / "phase4_representative_trajectories_3d_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-resolution-m", type=float, default=25.0)
    parser.add_argument("--heading-spacing-deg", type=float, default=5.0)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument(
        "--html-only", action="store_true",
        help="write interactive standalone HTML files without rewriting existing PNGs",
    )
    args = parser.parse_args()
    report = create_plots(
        spatial_resolution_m=args.spatial_resolution_m,
        heading_spacing_deg=args.heading_spacing_deg,
        run_label=args.run_label,
        write_png=not args.html_only,
        write_html=args.html_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
