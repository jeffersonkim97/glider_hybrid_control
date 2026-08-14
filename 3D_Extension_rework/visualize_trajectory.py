"""Generate the Stage 6 authoritative physical Bellman trajectory figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .bellman import build_cost_to_go_bundle
from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry, terrain_height
from .stage_cost import construct_stage_cost_6d
from .switching import select_switching_point
from .trajectory import extract_optimal_trajectory


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_6_trajectory"
FIGURE_DIR = ROOT / "figures"
NPZ_PATH = RESULT_DIR / "optimal_physical_trajectory.npz"
PNG_PATH = FIGURE_DIR / "stage_6_optimal_physical_trajectory.png"
PDF_PATH = FIGURE_DIR / "stage_6_optimal_physical_trajectory.pdf"


def _densify_polyline(points: np.ndarray, samples_per_edge: int = 25) -> np.ndarray:
    pieces = []
    fractions = np.linspace(0.0, 1.0, samples_per_edge)
    for index in range(points.shape[0] - 1):
        piece = points[index][None, :] + fractions[:, None] * (
            points[index + 1] - points[index]
        )[None, :]
        pieces.append(piece if index == 0 else piece[1:])
    return np.vstack(pieces)


def _cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.concatenate((
        np.zeros(1), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
    ))


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    cost_to_go = build_cost_to_go_bundle(
        configuration, geometry, detection, stage,
    )
    switching = select_switching_point(configuration, geometry, cost_to_go)
    trajectory = extract_optimal_trajectory(
        configuration, geometry, cost_to_go, switching,
    )
    if not trajectory["status"]["success"]:
        raise RuntimeError(trajectory["status"]["message"])

    powered = np.asarray(trajectory["powered_path"])
    glide_nodes = np.asarray(trajectory["glide_trajectory"])
    glide_dense = _densify_polyline(glide_nodes)
    full_dense = np.vstack((powered, glide_dense[1:]))
    switch = trajectory["switching_point"]
    entry = trajectory["bellman_entry_point"]
    contacts = geometry["tangent_manifold"]["contact_points"]

    figure = plt.figure(figsize=(16.0, 9.2), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(1.28, 1.0), height_ratios=(1.0, 0.78))
    axis_3d = figure.add_subplot(layout[:, 0], projection="3d", computed_zorder=False)
    axis_plan = figure.add_subplot(layout[0, 1])
    axis_profile = figure.add_subplot(layout[1, 1])
    terrain_x, terrain_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    axis_3d.plot_surface(
        terrain_x, terrain_y, geometry["terrain_height"],
        color="#aa9068", alpha=0.72, linewidth=0.12,
        edgecolor=(0.2, 0.15, 0.08, 0.12), zorder=2,
    )
    axis_3d.plot(contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.5,
                 color="#d000d0", linewidth=2.6, zorder=6)
    axis_3d.plot(powered[:, 0], powered[:, 1], powered[:, 2],
                 color="#f28e2b", linewidth=4.5, zorder=9)
    axis_3d.plot(glide_dense[:, 0], glide_dense[:, 1], glide_dense[:, 2],
                 color="#0072b2", linewidth=4.0, zorder=9)
    axis_3d.scatter(
        glide_nodes[1:-1, 0], glide_nodes[1:-1, 1], glide_nodes[1:-1, 2],
        s=34, color="white", edgecolor="#0072b2", linewidth=1.3, zorder=10,
    )
    axis_3d.scatter(*geometry["launch_position"], marker="s", s=100,
                    color="black", edgecolor="white", zorder=11)
    axis_3d.scatter(*geometry["sensor_position"], marker="^", s=125,
                    color="#d7191c", edgecolor="black", zorder=11)
    axis_3d.scatter(*geometry["goal_position"], marker="X", s=130,
                    color="#1a9850", edgecolor="black", zorder=11)
    axis_3d.scatter(*switch, marker="*", s=270, color="#ffd92f",
                    edgecolor="black", linewidth=1.1, zorder=12)
    axis_3d.set(
        xlim=(0, 3000), ylim=(0, 1000), zlim=(0, 400),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  Extracted physical Bellman trajectory",
    )
    axis_3d.set_box_aspect((3.0, 1.0, 0.78))
    axis_3d.view_init(elev=28.0, azim=-62.0)

    contour = axis_plan.contour(
        terrain_x, terrain_y, geometry["terrain_height"],
        levels=np.arange(20, 201, 20), colors="#8c6d31",
        linewidths=0.7, alpha=0.7,
    )
    axis_plan.clabel(contour, inline=True, fontsize=7, fmt="%.0f m")
    axis_plan.plot(contacts[:, 0], contacts[:, 1], color="#d000d0", linewidth=2.3)
    axis_plan.plot(powered[:, 0], powered[:, 1], color="#f28e2b", linewidth=3.3)
    axis_plan.plot(glide_dense[:, 0], glide_dense[:, 1], color="#0072b2", linewidth=3.0)
    axis_plan.scatter(glide_nodes[1:-1, 0], glide_nodes[1:-1, 1], s=32,
                      color="white", edgecolor="#0072b2", linewidth=1.2, zorder=8)
    axis_plan.scatter(switch[0], switch[1], marker="*", s=220,
                      color="#ffd92f", edgecolor="black", zorder=10)
    axis_plan.scatter(entry[0], entry[1], marker="o", s=70,
                      facecolor="white", edgecolor="#0072b2", linewidth=2, zorder=10)
    axis_plan.scatter(*geometry["launch_position"][:2], marker="s", s=75,
                      color="black", zorder=10)
    axis_plan.scatter(*geometry["sensor_position"][:2], marker="^", s=90,
                      color="#d7191c", edgecolor="black", zorder=10)
    axis_plan.scatter(*geometry["goal_position"][:2], marker="X", s=90,
                      color="#1a9850", edgecolor="black", zorder=10)
    axis_plan.set(
        xlim=(0, 3000), ylim=(0, 1000), xlabel="x [m]", ylabel="y [m]",
        title="B  Plan view",
    )
    axis_plan.set_aspect("equal", adjustable="box")
    axis_plan.grid(alpha=0.18)

    distance = _cumulative_distance(full_dense)
    terrain_under_path = terrain_height(
        geometry["terrain_model"], full_dense[:, 0], full_dense[:, 1],
    )
    boundary_interpolator = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]),
        geometry["los_boundary_height"], bounds_error=False, fill_value=np.nan,
    )
    los_boundary = boundary_interpolator(full_dense[:, :2])
    switching_distance = float(_cumulative_distance(powered)[-1])
    axis_profile.fill_between(distance, 0.0, terrain_under_path,
                              color="#aa9068", alpha=0.55, label="Terrain")
    axis_profile.plot(distance, los_boundary, color="#d000d0", linewidth=2.0,
                      linestyle="--", label=r"$H_{LOS}$")
    axis_profile.plot(distance, full_dense[:, 2], color="#333333", linewidth=3.0,
                      label="Trajectory altitude")
    axis_profile.axvspan(0.0, switching_distance, color="#f28e2b", alpha=0.11,
                         label="Powered / occluded")
    axis_profile.axvspan(switching_distance, distance[-1], color="#0072b2", alpha=0.09,
                         label="Glide / LOS-visible")
    axis_profile.axvline(switching_distance, color="#555555", linestyle=":", linewidth=1.4)
    axis_profile.scatter(switching_distance, switch[2], marker="*", s=150,
                         color="#ffd92f", edgecolor="black", zorder=8)
    axis_profile.set(
        xlim=(0.0, distance[-1]), ylim=(0.0, 400.0),
        xlabel="Cumulative path distance [m]", ylabel="h [m]",
        title="C  Terrain and LOS feasibility along the extracted path",
    )
    axis_profile.grid(alpha=0.2)
    axis_profile.legend(loc="upper right", fontsize=8.2, ncol=2)

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markersize=8, label="Launch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d7191c", markeredgecolor="black", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#1a9850", markeredgecolor="black", markersize=9, label="Goal"),
        Line2D([0], [0], color="#d000d0", linewidth=3, label="Tangent-contact curve"),
        Line2D([0], [0], color="#f28e2b", linewidth=3, label="Powered path"),
        Line2D([0], [0], color="#0072b2", linewidth=3, label="Bellman glide path"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd92f", markeredgecolor="black", markersize=13, label="Switch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#0072b2", markersize=8, label="Physical grid nodes"),
    ]
    axis_3d.legend(handles=legend, loc="upper left", fontsize=8.5, framealpha=0.95)
    figure.suptitle(
        "Stage 6 - Authoritative 3D Physical Bellman Path\n"
        "No NLP, no smoothing, no endpoint snapping",
        fontsize=15, fontweight="bold",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        powered_path=powered,
        glide_trajectory=glide_nodes,
        full_path=trajectory["full_path"],
        speed_profile_mps=trajectory["speed_profile_mps"],
        gamma_profile_rad=trajectory["gamma_profile_rad"],
        heading_profile_rad=trajectory["heading_profile_rad"],
        duration_profile_s=trajectory["duration_profile_s"],
        hazard_profile=trajectory["hazard_profile"],
        cost_profile=trajectory["cost_profile"],
        terminal_fraction_profile=trajectory["terminal_fraction_profile"],
        switching_point=trajectory["switching_point"],
        bellman_entry_point=trajectory["bellman_entry_point"],
    )
    summary = {
        "metadata": trajectory["metadata"],
        "mission": trajectory["mission"],
        "validation": trajectory["validation"],
        "switching_point_m": trajectory["switching_point"].tolist(),
        "bellman_entry_point_m": trajectory["bellman_entry_point"].tolist(),
        "glide_trajectory_nodes_m": glide_nodes.tolist(),
        "edge_kinds": list(trajectory["edge_kind_profile"]),
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    figure.savefig(PNG_PATH, dpi=230, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)
    print(NPZ_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
