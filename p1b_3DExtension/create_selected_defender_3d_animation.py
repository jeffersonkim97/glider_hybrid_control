"""Full-frame zoomed 3D animation of the selected Defender solution."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator

from .create_staged_defender_visualization import _cumulative_pod
from .experiment_staged_defender_optimization import OUTPUT_DIR, REPO_ROOT


RESULT_DIR = OUTPUT_DIR / "fine_selected_x2600_y0"
OUTPUT_PATH = (
    REPO_ROOT
    / "result_3D_visualization"
    / "selected_defender_x2600_zoomed_3d.gif"
)
FPS = 18
DISPLAY_SECONDS = 15.0


def _uniform_mission_samples(
    powered_path: np.ndarray,
    dense_time: np.ndarray,
    dense_states: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    powered_time = float(dense_time[0])
    powered_clock = np.linspace(0.0, powered_time, powered_path.shape[0])
    mission_clock = np.concatenate((powered_clock, dense_time[1:]))
    mission_positions = np.vstack((powered_path, dense_states[:3, 1:].T))
    frame_time = np.linspace(0.0, float(dense_time[-1]), frame_count)
    frame_positions = np.column_stack([
        np.interp(frame_time, mission_clock, mission_positions[:, coordinate])
        for coordinate in range(3)
    ])
    return frame_time, frame_positions


def create_animation() -> FuncAnimation:
    with (RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        discrete_summary = json.load(handle)
    with (RESULT_DIR / "continuous_summary.json").open(encoding="utf-8") as handle:
        continuous_summary = json.load(handle)
    with np.load(RESULT_DIR / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain_height = np.asarray(handle["terrain_height"])
        goal = np.asarray(handle["goal_position"])
    with np.load(RESULT_DIR / "continuous_trajectory.npz") as handle:
        powered_path = np.asarray(handle["powered_path"])
        dense_time = np.asarray(handle["dense_time"])
        dense_states = np.asarray(handle["dense_states"])

    sensor = np.asarray(continuous_summary["sensor_position"])
    switch = np.asarray(continuous_summary["switching_point"])
    frame_count = int(round(FPS * DISPLAY_SECONDS))
    frame_time, frame_positions = _uniform_mission_samples(
        powered_path, dense_time, dense_states, frame_count,
    )
    pod_time, pod_percent, powered_time = _cumulative_pod(
        (2600.0, 0.0), powered_path, dense_time, dense_states,
    )
    frame_pod = np.interp(frame_time, pod_time, pod_percent)
    terrain_interpolator = RegularGridInterpolator(
        (terrain_x, terrain_y), terrain_height,
        bounds_error=False, fill_value=np.nan,
    )
    mesh_x, mesh_y = np.meshgrid(terrain_x, terrain_y, indexing="ij")

    figure = plt.figure(figsize=(13.8, 8.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.plot_surface(
        mesh_x, mesh_y, terrain_height,
        cmap="terrain", alpha=0.94, linewidth=0.20,
        edgecolor=(0.20, 0.20, 0.20, 0.20), antialiased=True, zorder=0,
    )
    axis.contour(
        mesh_x, mesh_y, terrain_height,
        levels=[200.0], zdir="z", offset=200.0,
        colors="#8b0000", linewidths=2.4, zorder=4,
    )

    full_path = np.vstack((powered_path, dense_states[:3, 1:].T))
    axis.plot(
        full_path[:, 0], full_path[:, 1], full_path[:, 2],
        color="white", linewidth=5.2, alpha=0.58, zorder=7,
    )
    axis.plot(
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        color="#d95f02", linestyle="--", linewidth=2.0, alpha=0.55,
        label="Planned powered path", zorder=8,
    )
    axis.plot(
        dense_states[0], dense_states[1], dense_states[2],
        color="#1769aa", linestyle="--", linewidth=2.0, alpha=0.55,
        label="Planned glide path", zorder=8,
    )

    powered_halo, = axis.plot([], [], [], color="black", linewidth=6.3, zorder=10)
    powered_trace, = axis.plot(
        [], [], [], color="#d95f02", linewidth=4.1,
        label="Powered trace", zorder=11,
    )
    glide_halo, = axis.plot([], [], [], color="black", linewidth=6.3, zorder=10)
    glide_trace, = axis.plot(
        [], [], [], color="#1769aa", linewidth=4.1,
        label="Glide trace", zorder=11,
    )
    vehicle, = axis.plot(
        [], [], [], marker="D", linestyle="None", markersize=10.5,
        markerfacecolor="#ffd43b", markeredgecolor="black",
        markeredgewidth=1.3, label="Vehicle", zorder=15,
    )
    axis.scatter(
        *switch, marker="o", s=78, color="#e31a1c", edgecolor="black",
        label="Switch", zorder=13,
    )
    axis.scatter(
        *sensor, marker="^", s=150, color="#c51b7d", edgecolor="black",
        label="Selected sensor", zorder=14,
    )
    axis.scatter(
        *goal, marker="X", s=135, color="#39a96b", edgecolor="black",
        label="Goal", zorder=14,
    )
    status_text = axis.text2D(
        0.015, 0.965, "", transform=axis.transAxes,
        ha="left", va="top", fontsize=11.3,
        bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.94},
    )
    axis.text2D(
        0.69, 0.018,
        "Final physical model: detection scale = 1.0\n"
        "Red contour: terrain intersects 200 m ceiling",
        transform=axis.transAxes, fontsize=9.2, color="#6f0000",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
    )
    axis.set(
        xlim=(-50.0, 2725.0),
        ylim=(-900.0, 430.0),
        zlim=(0.0, 290.0),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title=(
            "Selected Defender Sensor (2600, 0) — Zoomed 3D Mission\n"
            f"Final PoD {100.0 * continuous_summary['mission_pod']:.3f}%  |  "
            f"Defender objective {continuous_summary['defender_objective']:.4f}"
        ),
    )
    axis.view_init(elev=30.0, azim=-62.0)
    axis.set_box_aspect((2.05, 1.18, 0.65))
    axis.legend(loc="upper right", fontsize=8.5, ncols=2, framealpha=0.94)

    first_glide_frame = int(np.searchsorted(frame_time, powered_time))

    def update(frame_index: int):
        point = frame_positions[frame_index]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        if frame_index <= first_glide_frame:
            powered_segment = frame_positions[:frame_index + 1]
            powered_halo.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            powered_trace.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            glide_halo.set_data_3d([], [], [])
            glide_trace.set_data_3d([], [], [])
            phase = "POWERED CLIMB"
            phase_color = "#d95f02"
        else:
            powered_segment = frame_positions[:first_glide_frame + 1]
            powered_halo.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            powered_trace.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            glide_segment = frame_positions[first_glide_frame:frame_index + 1]
            glide_halo.set_data_3d(
                glide_segment[:, 0], glide_segment[:, 1], glide_segment[:, 2]
            )
            glide_trace.set_data_3d(
                glide_segment[:, 0], glide_segment[:, 1], glide_segment[:, 2]
            )
            phase = "GLIDE / TERRAIN DETOUR"
            phase_color = "#1769aa"

        ground = float(terrain_interpolator(point[None, :2])[0])
        clearance = point[2] - ground
        status_text.set_text(
            f"{phase}\n"
            f"mission time: {frame_time[frame_index]:6.1f} / {frame_time[-1]:.1f} s\n"
            f"position: ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.1f}) m\n"
            f"terrain clearance: {clearance:.1f} m\n"
            f"cumulative PoD: {frame_pod[frame_index]:.3f}% / "
            f"{frame_pod[-1]:.3f}%"
        )
        status_text.set_color(phase_color)
        # A subtle camera drift makes the lateral curvature legible without
        # turning the animation into a spinning presentation.
        progress = frame_index / max(frame_count - 1, 1)
        axis.view_init(elev=30.0, azim=-65.0 + 7.0 * progress)
        return (
            powered_halo, powered_trace, glide_halo, glide_trace,
            vehicle, status_text,
        )

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
