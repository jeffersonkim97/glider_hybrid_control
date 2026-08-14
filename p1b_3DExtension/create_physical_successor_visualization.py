"""Render the validated exact-edge 3D fixed-sensor mission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "turn_limited_3d_coarse"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def create_figure(
    result_dir: Path = RESULT_DIR,
    title: str = "3D Fixed-Sensor Example — Exact Physical Successor Grid",
) -> plt.Figure:
    with np.load(result_dir / "trajectory_data.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    x_grid = arrays["terrain_x"]
    y_grid = arrays["terrain_y"]
    terrain = arrays["terrain_height"]
    airspace_ceiling = summary.get("airspace_ceiling_m")
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    powered = arrays["powered_path"]
    glide = arrays["trajectory"]
    sensor = arrays["sensor_position"]
    goal = arrays["goal_position"]
    switch = arrays["switching_point"]
    headings = np.rad2deg(arrays["heading_profile"])
    initial_heading = float(np.rad2deg(arrays["initial_heading_state"]))
    durations = arrays["duration_profile"]
    glide_time = np.concatenate(([0.0], np.cumsum(durations)))
    heading_state = np.concatenate(([initial_heading], headings))
    terrain_interpolator = RegularGridInterpolator(
        (x_grid, y_grid), terrain, bounds_error=False, fill_value=np.nan,
    )
    glide_ground = terrain_interpolator(glide[:, :2])

    figure = plt.figure(figsize=(15.2, 8.0), constrained_layout=True)
    layout = figure.add_gridspec(
        2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.15, 0.85),
    )
    axis_3d = figure.add_subplot(
        layout[:, 0], projection="3d", computed_zorder=False,
    )
    axis_top = figure.add_subplot(layout[0, 1])
    axis_heading = figure.add_subplot(layout[1, 1])

    axis_3d.plot_surface(
        mesh_x, mesh_y, terrain,
        cmap="terrain", alpha=0.98, linewidth=0.0, antialiased=True,
        zorder=0,
    )
    if (
        airspace_ceiling is not None
        and float(np.max(terrain)) > float(airspace_ceiling)
    ):
        axis_3d.contour(
            mesh_x, mesh_y, terrain,
            levels=[float(airspace_ceiling)], zdir="z",
            offset=float(airspace_ceiling), colors="#7F0000",
            linewidths=2.2, zorder=6,
        )
    axis_3d.plot(
        powered[:, 0], powered[:, 1], powered[:, 2],
        color="#2B8CBE", linewidth=3.0, label="Powered phase", zorder=8,
    )
    axis_3d.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        "-", color="black", linewidth=6.0, alpha=0.92, zorder=9,
    )
    axis_3d.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        "-o", color="#D01C8B", linewidth=3.5, markersize=4.8,
        markeredgecolor="black", markeredgewidth=0.55,
        label="Exact physical glide edges", zorder=10,
    )
    for point, ground_height in zip(glide, glide_ground):
        axis_3d.plot(
            [point[0], point[0]],
            [point[1], point[1]],
            [ground_height, point[2]],
            color="#D01C8B", linestyle=":", linewidth=1.0, alpha=0.72,
            zorder=8,
        )
    axis_3d.scatter(
        *switch, marker="o", s=70, color="#E31A1C",
        edgecolor="black", label="Switch", zorder=12,
    )
    axis_3d.scatter(
        *sensor, marker="^", s=95, color="black", label="Sensor", zorder=12,
    )
    axis_3d.scatter(
        *goal, marker="*", s=190, color="#FFD92F",
        edgecolor="black", label="Goal", zorder=12,
    )
    axis_3d.set(
        xlabel="x (m)", ylabel="y (m)", zlabel="h (m)",
        title="Validated 3D physical-edge trajectory",
    )
    axis_3d.view_init(elev=29, azim=-63)
    axis_3d.set_box_aspect((1.55, 1.12, 0.43))
    axis_3d.legend(loc="upper left", fontsize=8.5)

    contours = axis_top.contourf(
        mesh_x, mesh_y, terrain, levels=16, cmap="terrain", alpha=0.82,
    )
    axis_top.contour(
        mesh_x, mesh_y, terrain, levels=10,
        colors="0.35", linewidths=0.35, alpha=0.5,
    )
    ceiling_patch = None
    if (
        airspace_ceiling is not None
        and float(np.max(terrain)) > float(airspace_ceiling)
    ):
        ceiling = float(airspace_ceiling)
        axis_top.contourf(
            mesh_x, mesh_y, terrain,
            levels=[ceiling, float(np.max(terrain)) + 1.0],
            colors=["#7F0000"], alpha=0.18, hatches=["////"],
        )
        ceiling_contour = axis_top.contour(
            mesh_x, mesh_y, terrain, levels=[ceiling],
            colors="#7F0000", linewidths=2.0,
        )
        axis_top.clabel(
            ceiling_contour, fmt={ceiling: f"{ceiling:.0f} m ceiling"},
            fontsize=8.0,
        )
        ceiling_patch = Patch(
            facecolor="#7F0000", edgecolor="#7F0000", alpha=0.25,
            hatch="////", label="Terrain above flight ceiling",
        )
    axis_top.plot(
        powered[:, 0], powered[:, 1],
        color="#2B8CBE", linewidth=2.6, label="Powered",
    )
    axis_top.plot(
        glide[:, 0], glide[:, 1], "-o",
        color="#D01C8B", linewidth=3.0, markersize=4.0,
        label="Physical glide",
    )
    axis_top.scatter(
        sensor[0], sensor[1], marker="^", s=75, color="black", zorder=6,
    )
    axis_top.scatter(
        goal[0], goal[1], marker="*", s=145,
        color="#FFD92F", edgecolor="black", zorder=6,
    )
    axis_top.scatter(
        switch[0], switch[1], marker="o", s=55,
        color="#E31A1C", edgecolor="black", zorder=6,
    )
    for index in range(glide.shape[0] - 1):
        delta = glide[index + 1, :2] - glide[index, :2]
        midpoint = 0.5 * (glide[index + 1, :2] + glide[index, :2])
        axis_top.quiver(
            midpoint[0], midpoint[1], delta[0], delta[1],
            angles="xy", scale_units="xy", scale=2.7,
            width=0.006, color="#9E015D", alpha=0.75,
        )
    axis_top.set(
        xlabel="x (m)", ylabel="y (m)",
        title="Top view: exact node-to-node lateral route",
    )
    axis_top.set_aspect("equal", adjustable="box")
    top_handles, top_labels = axis_top.get_legend_handles_labels()
    if ceiling_patch is not None:
        top_handles.append(ceiling_patch)
        top_labels.append(ceiling_patch.get_label())
    axis_top.legend(
        top_handles, top_labels, loc="lower right", fontsize=8.2,
    )
    figure.colorbar(
        contours, ax=axis_top, label="terrain height (m)", shrink=0.82,
    )

    axis_heading.step(
        glide_time, heading_state, where="post",
        color="#00798C", linewidth=2.7,
    )
    axis_heading.scatter(
        glide_time, heading_state, s=24, color="#00798C", zorder=4,
    )
    axis_heading.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axis_heading.set(
        xlabel="glide time (s)", ylabel="heading ψ (deg)",
        title="Heading state along the physical edges",
    )
    axis_heading.grid(alpha=0.25)
    axis_heading.text(
        0.02, 0.96,
        (
            f"endpoint residual: "
            f"{summary['maximum_edge_endpoint_residual_m']:.2e} m\n"
            f"goal error: {summary['goal_error_m']:.2f} m "
            "(15 m goal radius)\n"
            f"max turn rate: {summary['maximum_turn_rate_deg_s']:.2f}°/s "
            f"< {summary['configured_max_turn_rate_deg_s']:.2f}°/s\n"
            f"min terrain clearance: "
            f"{summary['minimum_glide_terrain_clearance_m']:.2f} m\n"
            f"PoD: {100.0 * summary['mission_pod']:.2f}%   "
            f"time: {summary['mission_time']:.2f} s"
        ),
        transform=axis_heading.transAxes,
        va="top",
        fontsize=9.6,
        bbox={"facecolor": "white", "edgecolor": "0.70", "alpha": 0.95},
    )

    figure.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.012,
        (
            "Every glide edge satisfies endpoint = start + velocity × duration; "
            "no off-grid endpoint reset or trajectory smoothing is used."
        ),
        ha="center",
        fontsize=9.8,
        color="0.28",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--output-stem", default="physical_successor_3d_validated",
    )
    parser.add_argument(
        "--title",
        default="3D Fixed-Sensor Example — Exact Physical Successor Grid",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = create_figure(args.result_dir, args.title)
    png_path = args.output_dir / f"{args.output_stem}.png"
    pdf_path = args.output_dir / f"{args.output_stem}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
