"""Agent-following chase-camera animation of the selected 3D mission."""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator

from .create_selected_defender_3d_animation import (
    DISPLAY_SECONDS,
    FPS,
    RESULT_DIR,
    _uniform_mission_samples,
)
from .create_staged_defender_visualization import _cumulative_pod
from .experiment_staged_defender_optimization import REPO_ROOT


OUTPUT_PATH = (
    REPO_ROOT
    / "result_3D_visualization"
    / "selected_defender_x2600_agent_perspective.gif"
)


def _smoothed_heading(positions: np.ndarray) -> np.ndarray:
    velocity = np.gradient(positions[:, :2], axis=0)
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    window = 13
    padding = window // 2
    padded = np.pad(heading, padding, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _window_limits(
    point: np.ndarray,
    heading: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    # Bias the map window forward in the direction of travel while keeping it
    # inside the sampled terrain domain.
    forward = np.array([np.cos(heading), np.sin(heading)])
    center = point[:2] + 210.0 * forward
    x_width = 1350.0
    y_width = 1120.0
    x_low = float(np.clip(
        center[0] - 0.5 * x_width,
        x_bounds[0], x_bounds[1] - x_width,
    ))
    y_low = float(np.clip(
        center[1] - 0.5 * y_width,
        y_bounds[0], y_bounds[1] - y_width,
    ))
    return (x_low, x_low + x_width), (y_low, y_low + y_width)


def create_animation() -> FuncAnimation:
    with (RESULT_DIR / "continuous_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with np.load(RESULT_DIR / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain_height = np.asarray(handle["terrain_height"])
        goal = np.asarray(handle["goal_position"])
    with np.load(RESULT_DIR / "continuous_trajectory.npz") as handle:
        powered_path = np.asarray(handle["powered_path"])
        dense_time = np.asarray(handle["dense_time"])
        dense_states = np.asarray(handle["dense_states"])

    sensor = np.asarray(summary["sensor_position"])
    switch = np.asarray(summary["switching_point"])
    frame_count = int(round(FPS * DISPLAY_SECONDS))
    frame_time, frame_positions = _uniform_mission_samples(
        powered_path, dense_time, dense_states, frame_count,
    )
    headings = _smoothed_heading(frame_positions)
    pod_time, pod_percent, powered_time = _cumulative_pod(
        (2600.0, 0.0), powered_path, dense_time, dense_states,
    )
    frame_pod = np.interp(frame_time, pod_time, pod_percent)
    terrain_interpolator = RegularGridInterpolator(
        (terrain_x, terrain_y), terrain_height,
        bounds_error=False, fill_value=np.nan,
    )
    mesh_x, mesh_y = np.meshgrid(terrain_x, terrain_y, indexing="ij")
    full_path = np.vstack((powered_path, dense_states[:3, 1:].T))

    figure = plt.figure(figsize=(13.8, 8.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.plot_surface(
        mesh_x, mesh_y, terrain_height,
        cmap="terrain", alpha=0.97, linewidth=0.22,
        edgecolor=(0.18, 0.18, 0.18, 0.22), antialiased=True, zorder=0,
    )
    axis.contour(
        mesh_x, mesh_y, terrain_height,
        levels=[200.0], zdir="z", offset=200.0,
        colors="#8b0000", linewidths=2.3, zorder=4,
    )
    axis.plot(
        full_path[:, 0], full_path[:, 1], full_path[:, 2],
        color="white", linewidth=5.0, alpha=0.48, zorder=7,
    )
    axis.plot(
        full_path[:, 0], full_path[:, 1], full_path[:, 2],
        color="#666666", linestyle="--", linewidth=1.4, alpha=0.48,
        label="Planned route", zorder=8,
    )

    trace_halo, = axis.plot([], [], [], color="black", linewidth=6.5, zorder=10)
    trace, = axis.plot(
        [], [], [], color="#1769aa", linewidth=4.1,
        label="Flown trace", zorder=11,
    )
    vehicle, = axis.plot(
        [], [], [], marker="D", linestyle="None", markersize=11.0,
        markerfacecolor="#ffd43b", markeredgecolor="black",
        markeredgewidth=1.4, label="Agent", zorder=15,
    )
    axis.scatter(
        *switch, marker="o", s=76, color="#e31a1c", edgecolor="black",
        label="Switch", zorder=13,
    )
    axis.scatter(
        *sensor, marker="^", s=145, color="#c51b7d", edgecolor="black",
        label="Sensor", zorder=14,
    )
    axis.scatter(
        *goal, marker="X", s=130, color="#39a96b", edgecolor="black",
        label="Goal", zorder=14,
    )

    hud = axis.text2D(
        0.015, 0.97, "", transform=axis.transAxes,
        ha="left", va="top", fontsize=11.3, family="monospace",
        bbox={"facecolor": "#101820", "edgecolor": "#7f8c8d", "alpha": 0.88},
        color="white",
    )
    axis.text2D(
        0.69, 0.018,
        "Agent-following chase camera\n"
        "Camera yaw follows vehicle heading",
        transform=axis.transAxes, fontsize=9.4, color="#102a43",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    axis.set(
        zlim=(0.0, 290.0),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title=(
            "Agent Perspective — Selected Defender Mission\n"
            "Local terrain window follows position and heading"
        ),
    )
    axis.set_box_aspect((1.55, 1.20, 0.72))
    axis.legend(loc="upper right", fontsize=8.7, ncols=2, framealpha=0.94)

    x_bounds = (float(terrain_x[0]), float(terrain_x[-1]))
    y_bounds = (float(terrain_y[0]), float(terrain_y[-1]))

    def update(frame_index: int):
        point = frame_positions[frame_index]
        heading = headings[frame_index]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        flown = frame_positions[:frame_index + 1]
        trace_halo.set_data_3d(flown[:, 0], flown[:, 1], flown[:, 2])
        phase_color = "#d95f02" if frame_time[frame_index] <= powered_time else "#1769aa"
        trace.set_color(phase_color)
        trace.set_data_3d(flown[:, 0], flown[:, 1], flown[:, 2])

        x_limits, y_limits = _window_limits(
            point, heading, x_bounds, y_bounds,
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        # Camera position is behind the horizontal velocity vector, looking
        # forward over the agent with a modest downward chase angle.
        axis.view_init(elev=19.5, azim=np.rad2deg(heading) + 180.0)

        ground = float(terrain_interpolator(point[None, :2])[0])
        clearance = point[2] - ground
        phase = "POWERED" if frame_time[frame_index] <= powered_time else "GLIDE"
        hud.set_text(
            "AGENT CHASE VIEW\n"
            f"phase       {phase:>7s}\n"
            f"time        {frame_time[frame_index]:6.1f} / {frame_time[-1]:.1f} s\n"
            f"heading     {np.rad2deg(heading):6.1f} deg\n"
            f"altitude    {point[2]:6.1f} m\n"
            f"clearance   {clearance:6.1f} m\n"
            f"cum. PoD    {frame_pod[frame_index]:6.3f}% / {frame_pod[-1]:.3f}%"
        )
        return trace_halo, trace, vehicle, hud

    return FuncAnimation(
        figure, update, frames=frame_count,
        interval=1000.0 / FPS, blit=False, repeat=True,
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    animation = create_animation()
    animation.save(
        OUTPUT_PATH,
        writer=PillowWriter(fps=FPS),
        dpi=105,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(animation._fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
