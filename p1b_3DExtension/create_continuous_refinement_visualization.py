"""Visualize the validated continuous 3-DOF post-Bellman refinement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_continuous"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def create_figure() -> plt.Figure:
    with np.load(RESULT_DIR / "trajectory_data.npz") as handle:
        arrays = {name: np.asarray(handle[name]) for name in handle.files}
    with (RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    x_grid = arrays["terrain_x"]
    y_grid = arrays["terrain_y"]
    terrain = arrays["terrain_height"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    powered = arrays["powered_path"]
    dense = arrays["dense_states"]
    time = arrays["dense_time"]
    glide = dense[:3].T
    discrete = arrays["discrete_trajectory"]
    sensor = arrays["sensor_position"]
    goal = arrays["goal_position"]
    switch = glide[0]
    terrain_lookup = RegularGridInterpolator(
        (x_grid, y_grid), terrain, bounds_error=False, fill_value=np.nan,
    )
    powered_ground = terrain_lookup(powered[:, :2])
    glide_ground = terrain_lookup(glide[:, :2])
    powered_time = np.linspace(0.0, summary["powered_time_s"], powered.shape[0])

    figure = plt.figure(figsize=(15.5, 8.4), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    axis_3d = figure.add_subplot(
        layout[:, 0], projection="3d", computed_zorder=False,
    )
    axis_top = figure.add_subplot(layout[0, 1])
    axis_states = figure.add_subplot(layout[1, 1])

    axis_3d.plot_surface(
        mesh_x, mesh_y, terrain, cmap="terrain", alpha=0.98,
        linewidth=0.0, antialiased=True, zorder=0,
    )
    axis_3d.contour(
        mesh_x, mesh_y, terrain, levels=[200.0], zdir="z", offset=200.0,
        colors="#7F0000", linewidths=2.2, zorder=5,
    )
    axis_3d.plot(
        arrays["discrete_powered_path"][:, 0],
        arrays["discrete_powered_path"][:, 1],
        arrays["discrete_powered_path"][:, 2],
        color="0.45", linestyle="--", linewidth=1.4, alpha=0.65,
        label="Discrete baseline", zorder=7,
    )
    axis_3d.plot(
        discrete[:, 0], discrete[:, 1], discrete[:, 2],
        color="0.45", linestyle="--", linewidth=1.4, alpha=0.65,
        zorder=7,
    )
    axis_3d.plot(
        powered[:, 0], powered[:, 1], powered[:, 2],
        color="#2B8CBE", linewidth=3.4, label="Continuous powered", zorder=9,
    )
    axis_3d.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        color="black", linewidth=6.0, alpha=0.90, zorder=9,
    )
    axis_3d.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        color="#D01C8B", linewidth=3.5, label="Continuous 3-DOF glide",
        zorder=10,
    )
    axis_3d.scatter(
        *switch, s=75, color="#E31A1C", edgecolor="black",
        label="Continuous switch", zorder=12,
    )
    axis_3d.scatter(*sensor, marker="^", s=100, color="black", zorder=12)
    axis_3d.scatter(
        *goal, marker="*", s=200, color="#FFD92F", edgecolor="black", zorder=12,
    )
    axis_3d.set(
        xlim=(-50.0, 2600.0), ylim=(-850.0, 650.0), zlim=(0.0, 290.0),
        xlabel="x (m)", ylabel="y (m)", zlabel="h (m)",
        title="Continuous 3-DOF trajectory",
    )
    axis_3d.view_init(elev=29, azim=-63)
    axis_3d.set_box_aspect((1.58, 0.90, 0.52))
    axis_3d.legend(loc="upper left", fontsize=8.5)

    contours = axis_top.contourf(
        mesh_x, mesh_y, terrain, levels=18, cmap="terrain", alpha=0.85,
    )
    axis_top.contour(
        mesh_x, mesh_y, terrain, levels=12, colors="0.35",
        linewidths=0.35, alpha=0.55,
    )
    axis_top.contourf(
        mesh_x, mesh_y, terrain, levels=[200.0, float(np.max(terrain)) + 1.0],
        colors=["#7F0000"], alpha=0.18, hatches=["////"],
    )
    ceiling_contour = axis_top.contour(
        mesh_x, mesh_y, terrain, levels=[200.0],
        colors="#7F0000", linewidths=2.0,
    )
    axis_top.clabel(
        ceiling_contour, fmt={200.0: "200 m ceiling"}, fontsize=7.5,
    )
    axis_top.plot(
        discrete[:, 0], discrete[:, 1], color="0.45", linestyle="--",
        linewidth=1.4, label="Discrete baseline",
    )
    axis_top.plot(
        powered[:, 0], powered[:, 1], color="#2B8CBE", linewidth=3.0,
        label="Powered",
    )
    axis_top.plot(
        glide[:, 0], glide[:, 1], color="#D01C8B", linewidth=3.2,
        label="Continuous glide",
    )
    axis_top.scatter(switch[0], switch[1], s=55, color="#E31A1C", edgecolor="black")
    axis_top.scatter(sensor[0], sensor[1], marker="^", s=75, color="black")
    axis_top.scatter(
        goal[0], goal[1], marker="*", s=145,
        color="#FFD92F", edgecolor="black",
    )
    handles, labels = axis_top.get_legend_handles_labels()
    handles.append(Patch(
        facecolor="#7F0000", edgecolor="#7F0000", hatch="////", alpha=0.25,
        label="Terrain above ceiling",
    ))
    labels.append("Terrain above ceiling")
    axis_top.legend(handles, labels, loc="lower right", fontsize=8.0)
    axis_top.set(
        xlim=(-50.0, 2600.0), ylim=(-850.0, 650.0),
        xlabel="x (m)", ylabel="y (m)",
        title="Top view: dynamics-generated curved detour",
    )
    axis_top.set_aspect("equal", adjustable="box")
    figure.colorbar(contours, ax=axis_top, label="terrain height (m)", shrink=0.82)

    full_time = np.concatenate((powered_time, time[1:]))
    full_gamma = np.concatenate((
        np.full(powered_time.shape, summary["powered_gamma_deg"]),
        np.rad2deg(dense[4, 1:]),
    ))
    full_bank = np.concatenate((
        np.zeros(powered_time.shape), np.rad2deg(dense[6, 1:]),
    ))
    full_speed = np.concatenate((
        np.full(powered_time.shape, 21.0), dense[3, 1:],
    ))
    axis_states.plot(full_time, full_gamma, color="#2C7BB6", linewidth=2.2, label="γ")
    axis_states.plot(full_time, full_bank, color="#D7191C", linewidth=2.0, label="bank φ")
    axis_states.axvline(
        summary["powered_time_s"], color="0.2", linestyle="--", linewidth=1.1,
        label="switch",
    )
    speed_axis = axis_states.twinx()
    speed_axis.plot(full_time, full_speed, color="#31A354", linewidth=1.8, label="speed")
    axis_states.set(
        xlabel="mission time (s)", ylabel="angle (deg)",
        title="Continuous state history across the switch",
    )
    speed_axis.set_ylabel("speed (m/s)", color="#31A354")
    axis_states.grid(alpha=0.25)
    handles_a, labels_a = axis_states.get_legend_handles_labels()
    handles_b, labels_b = speed_axis.get_legend_handles_labels()
    axis_states.legend(handles_a + handles_b, labels_a + labels_b, loc="lower left", fontsize=8.2)
    axis_states.text(
        0.98, 0.96,
        (
            f"switch h: {summary['switch_state'][2]:.2f} m\n"
            f"post-switch climb: +{summary['post_switch_altitude_gain_m']:.2f} m\n"
            f"apex delay: {summary['time_to_post_switch_apex_s']:.2f} s\n"
            f"switch residual: {summary['switch_continuity_residual']:.1e}\n"
            f"PoD: {100.0 * summary['mission_pod']:.3f}%"
        ),
        transform=axis_states.transAxes, ha="right", va="top", fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.94},
    )

    figure.suptitle(
        "Post-Bellman Continuous 3-DOF Refinement — Projection Unchanged",
        fontsize=15.5, fontweight="bold",
    )
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    png = OUTPUT_DIR / "continuous_3dof_refinement.png"
    pdf = OUTPUT_DIR / "continuous_3dof_refinement.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
