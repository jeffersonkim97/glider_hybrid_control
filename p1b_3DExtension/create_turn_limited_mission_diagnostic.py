"""Render the turn-limited coarse mission and its continuous-replay mismatch."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "turn_limited_3d_coarse"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def create_figure() -> plt.Figure:
    with np.load(RESULT_DIR / "trajectory_data.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    with (RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    x_grid = arrays["terrain_x"]
    y_grid = arrays["terrain_y"]
    terrain = arrays["terrain_height"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    powered = arrays["powered_path"]
    snapped = arrays["trajectory"]
    continuous = arrays["continuous_control_reconstruction"]
    sensor = arrays["sensor_position"]
    goal = arrays["goal_position"]
    heading_deg = np.rad2deg(arrays["heading_profile"])
    initial_heading_deg = float(np.rad2deg(arrays["initial_heading_state"]))
    dt = float(arrays["coarse_step_count"] * arrays["time_step"])
    heading_time = dt * np.arange(heading_deg.size + 1)
    heading_state = np.concatenate(([initial_heading_deg], heading_deg))

    figure = plt.figure(figsize=(15.0, 8.2), constrained_layout=True)
    layout = figure.add_gridspec(
        2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.15, 0.85),
    )
    axis_3d = figure.add_subplot(layout[:, 0], projection="3d")
    axis_top = figure.add_subplot(layout[0, 1])
    axis_heading = figure.add_subplot(layout[1, 1])

    axis_3d.plot_surface(
        mesh_x, mesh_y, terrain,
        cmap="terrain", alpha=0.76, linewidth=0.0, antialiased=True,
    )
    axis_3d.plot(
        powered[:, 0], powered[:, 1], powered[:, 2],
        color="#2B8CBE", linewidth=3.0, label="Powered path",
    )
    axis_3d.plot(
        snapped[:, 0], snapped[:, 1], snapped[:, 2],
        "-o", color="#6A3D9A", linewidth=3.0, markersize=3.8,
        label="Bellman grid nodes (snapped)",
    )
    axis_3d.plot(
        continuous[:, 0], continuous[:, 1], continuous[:, 2],
        "--", color="#E66101", linewidth=3.0,
        label="Recorded controls replayed open-loop (no grid reset)",
    )
    axis_3d.scatter(
        *sensor, marker="^", s=95, color="black", label="Sensor", depthshade=False,
    )
    axis_3d.scatter(
        *goal, marker="*", s=190, color="#FFD92F", edgecolor="black",
        label="Goal", depthshade=False,
    )
    axis_3d.set(
        xlabel="x (m)", ylabel="y (m)", zlabel="h (m)",
        title="Actual coarse mission diagnostic",
    )
    axis_3d.view_init(elev=28, azim=-63)
    axis_3d.set_box_aspect((1.55, 1.15, 0.42))
    axis_3d.legend(loc="upper left", fontsize=8.5)

    contours = axis_top.contourf(
        mesh_x, mesh_y, terrain, levels=16, cmap="terrain", alpha=0.80,
    )
    axis_top.contour(
        mesh_x, mesh_y, terrain, levels=10,
        colors="0.35", linewidths=0.35, alpha=0.55,
    )
    axis_top.plot(
        snapped[:, 0], snapped[:, 1], "-o",
        color="#6A3D9A", linewidth=2.5, markersize=3.5,
        label="Snapped Bellman path",
    )
    axis_top.plot(
        continuous[:, 0], continuous[:, 1], "--",
        color="#E66101", linewidth=2.7,
        label="Open-loop replay (no grid reset)",
    )
    axis_top.scatter(
        sensor[0], sensor[1], marker="^", s=75, color="black", zorder=5,
    )
    axis_top.scatter(
        goal[0], goal[1], marker="*", s=145,
        color="#FFD92F", edgecolor="black", zorder=5,
    )
    axis_top.set(
        xlabel="x (m)", ylabel="y (m)",
        title="Top view: grid snapping suppresses lateral motion",
    )
    axis_top.set_aspect("equal", adjustable="box")
    axis_top.legend(loc="lower left", fontsize=8.5)
    figure.colorbar(contours, ax=axis_top, label="terrain height (m)", shrink=0.82)

    axis_heading.step(
        heading_time, heading_state, where="post",
        color="#00798C", linewidth=2.6, label="heading state",
    )
    axis_heading.axhline(
        0.0, color="0.45", linewidth=0.8, linestyle=":",
    )
    axis_heading.set(
        xlabel="glide time (s)", ylabel="heading ψ (deg)",
        title="Heading changes are bounded, but spatial endpoints are not physical",
    )
    axis_heading.grid(alpha=0.25)
    axis_heading.text(
        0.02, 0.96,
        (
            f"realized max turn rate: {summary['maximum_turn_rate_deg_s']:.2f}°/s\n"
            f"configured limit: {summary['configured_max_turn_rate_deg_s']:.2f}°/s\n"
            "continuous replay goal error: "
            f"{summary['continuous_control_reconstruction_goal_error_m']:.0f} m"
        ),
        transform=axis_heading.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.94},
    )

    figure.suptitle(
        "Turn-Limited 3D Rerun — Diagnostic Only, Not Presentation-Ready",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        (
            "Heading memory is fixed. The remaining discrepancy is the legacy "
            "off-grid-endpoint reset, which requires a physical successor-grid solver."
        ),
        ha="center",
        fontsize=10,
        color="0.28",
    )
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    png_path = OUTPUT_DIR / "turn_limited_3d_mission_diagnostic.png"
    pdf_path = OUTPUT_DIR / "turn_limited_3d_mission_diagnostic.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
