"""Animate the validated fine-grid elliptical-ridge mission in 3D."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator

from .detection import build_symbolic_detection_bundle
from .experiment_extreme_ridge_fine import build_fine_configuration
from .geometry import build_geometry_bundle
from .phase_logging import close_phase_logger


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_fine"
OUTPUT_PATH = (
    REPO_ROOT / "result_3D_visualization" / "extreme_ridge_275_fine_3d.gif"
)
FPS = 18
DISPLAY_SECONDS = 14.0


def _mission_samples(
    powered_path: np.ndarray,
    glide_path: np.ndarray,
    powered_time: float,
    glide_durations: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the saved trajectory at uniformly spaced mission times."""
    powered_lengths = np.linalg.norm(np.diff(powered_path, axis=0), axis=1)
    powered_nodes = np.concatenate(([0.0], np.cumsum(powered_lengths)))
    powered_nodes *= powered_time / powered_nodes[-1]
    glide_nodes = powered_time + np.concatenate(
        ([0.0], np.cumsum(glide_durations))
    )
    mission_time = float(glide_nodes[-1])
    frame_times = np.linspace(0.0, mission_time, frame_count)
    positions = np.empty((frame_count, 3), dtype=float)
    for coordinate in range(3):
        powered_mask = frame_times <= powered_time
        positions[powered_mask, coordinate] = np.interp(
            frame_times[powered_mask], powered_nodes, powered_path[:, coordinate]
        )
        positions[~powered_mask, coordinate] = np.interp(
            frame_times[~powered_mask], glide_nodes, glide_path[:, coordinate]
        )
    return frame_times, positions, frame_times <= powered_time


def _cumulative_trapezoid(rate: np.ndarray, time: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(rate, dtype=float)
    cumulative[1:] = np.cumsum(
        0.5 * (rate[:-1] + rate[1:]) * np.diff(time)
    )
    return cumulative


def _function_last_output(function, *arguments: np.ndarray) -> np.ndarray:
    outputs = function(*arguments)
    output_tuple = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.asarray(output_tuple[-1], dtype=float).reshape(-1)


def _cumulative_pod_at_frames(
    arrays: dict[str, np.ndarray],
    summary: dict,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Reintegrate phase-specific detection rates along the saved path."""
    configuration = build_fine_configuration()
    logger = configuration["primary_result"]["logging_utilities"]["logger"]
    try:
        geometry = build_geometry_bundle(configuration)
        detection = build_symbolic_detection_bundle(configuration, geometry)
        functions = detection["primary_result"]["functions"]
        sensor = arrays["sensor_position"]
        powered_speed = float(
            configuration["primary_result"]["vehicle_config"]["powered_speed"]
        )

        powered_path = arrays["powered_path"]
        powered_time = float(summary["powered_time"])
        powered_fraction = np.linspace(0.0, 1.0, 801)
        powered_points = (
            powered_path[0][None, :]
            + powered_fraction[:, None]
            * (powered_path[-1] - powered_path[0])[None, :]
        )
        powered_sample_time = powered_fraction * powered_time
        powered_count = powered_fraction.size
        powered_rate = _function_last_output(
            functions["powered_detection_components"].map(powered_count),
            powered_points[:, 0].reshape(1, -1),
            powered_points[:, 1].reshape(1, -1),
            powered_points[:, 2].reshape(1, -1),
            np.full((1, powered_count), powered_speed),
            np.full((1, powered_count), sensor[0]),
            np.full((1, powered_count), sensor[1]),
            np.full((1, powered_count), sensor[2]),
        )
        powered_hazard = _cumulative_trapezoid(
            powered_rate, powered_sample_time,
        )
        hazard_times = list(powered_sample_time)
        cumulative_hazard = list(powered_hazard)

        glide_path = arrays["trajectory"]
        durations = arrays["duration_profile"]
        speeds = arrays["speed_profile"]
        gammas = arrays["gamma_profile"]
        headings = arrays["heading_profile"]
        time_offset = powered_time
        hazard_offset = float(powered_hazard[-1])
        edge_fraction = np.linspace(0.0, 1.0, 101)
        edge_count = edge_fraction.size
        for edge_index, duration in enumerate(durations):
            start = glide_path[edge_index]
            end = glide_path[edge_index + 1]
            points = (
                start[None, :]
                + edge_fraction[:, None] * (end - start)[None, :]
            )
            local_time = edge_fraction * float(duration)
            rate = _function_last_output(
                functions["glide_detection_components"].map(edge_count),
                points[:, 0].reshape(1, -1),
                points[:, 1].reshape(1, -1),
                points[:, 2].reshape(1, -1),
                np.full((1, edge_count), speeds[edge_index]),
                np.full((1, edge_count), gammas[edge_index]),
                np.full((1, edge_count), headings[edge_index]),
                np.full((1, edge_count), sensor[0]),
                np.full((1, edge_count), sensor[1]),
                np.full((1, edge_count), sensor[2]),
            )
            local_hazard = hazard_offset + _cumulative_trapezoid(
                rate, local_time,
            )
            hazard_times.extend((time_offset + local_time[1:]).tolist())
            cumulative_hazard.extend(local_hazard[1:].tolist())
            time_offset += float(duration)
            hazard_offset = float(local_hazard[-1])

        hazard_times_array = np.asarray(hazard_times)
        cumulative_hazard_array = np.asarray(cumulative_hazard)
        target_hazard = -np.log1p(-float(summary["mission_pod"]))
        if cumulative_hazard_array[-1] <= 0.0:
            raise RuntimeError("Reintegrated mission hazard is nonpositive")
        cumulative_hazard_array *= target_hazard / cumulative_hazard_array[-1]
        frame_hazard = np.interp(
            frame_times, hazard_times_array, cumulative_hazard_array,
        )
        return 1.0 - np.exp(-frame_hazard)
    finally:
        close_phase_logger(logger)


def create_animation() -> FuncAnimation:
    with np.load(RESULT_DIR / "trajectory_data.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    with (RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    x_grid = arrays["terrain_x"]
    y_grid = arrays["terrain_y"]
    terrain = arrays["terrain_height"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    powered = arrays["powered_path"]
    glide = arrays["trajectory"]
    sensor = arrays["sensor_position"]
    goal = arrays["goal_position"]
    switch = arrays["switching_point"]
    frame_count = int(round(FPS * DISPLAY_SECONDS))
    frame_times, positions, powered_mask = _mission_samples(
        powered,
        glide,
        float(summary["powered_time"]),
        arrays["duration_profile"],
        frame_count,
    )
    cumulative_pod = _cumulative_pod_at_frames(arrays, summary, frame_times)
    terrain_interpolator = RegularGridInterpolator(
        (x_grid, y_grid), terrain, bounds_error=False, fill_value=np.nan,
    )

    figure = plt.figure(figsize=(12.0, 7.4), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.plot_surface(
        mesh_x, mesh_y, terrain,
        cmap="terrain", alpha=0.98, linewidth=0.0, antialiased=True,
        zorder=0,
    )
    ceiling = float(summary["airspace_ceiling_m"])
    axis.contour(
        mesh_x, mesh_y, terrain, levels=[ceiling], zdir="z", offset=ceiling,
        colors="#7F0000", linewidths=2.4, zorder=5,
    )
    axis.plot(
        powered[:, 0], powered[:, 1], powered[:, 2],
        linestyle="--", color="#2B8CBE", linewidth=1.5, alpha=0.30,
        label="Planned powered path", zorder=7,
    )
    axis.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        linestyle="--", color="#D01C8B", linewidth=1.5, alpha=0.30,
        label="Planned glide path", zorder=7,
    )
    powered_trace, = axis.plot(
        [], [], [], color="#2B8CBE", linewidth=4.0,
        label="Powered trace", zorder=9,
    )
    glide_halo, = axis.plot(
        [], [], [], color="black", linewidth=6.0, alpha=0.88, zorder=9,
    )
    glide_trace, = axis.plot(
        [], [], [], color="#D01C8B", linewidth=3.5,
        label="Glide trace", zorder=10,
    )
    vehicle, = axis.plot(
        [], [], [], marker="D", linestyle="None", markersize=9.5,
        markerfacecolor="#FFD92F", markeredgecolor="black",
        markeredgewidth=1.2, label="Vehicle", zorder=14,
    )
    axis.scatter(
        *switch, marker="o", s=75, color="#E31A1C", edgecolor="black",
        label="Switch", zorder=12,
    )
    axis.scatter(
        *sensor, marker="^", s=110, color="black", label="Sensor", zorder=12,
    )
    axis.scatter(
        *goal, marker="*", s=220, color="#FFD92F", edgecolor="black",
        label="Goal", zorder=12,
    )
    status_text = axis.text2D(
        0.02, 0.96, "", transform=axis.transAxes, va="top", fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.94},
    )
    axis.text2D(
        0.70, 0.02,
        "Red contour: terrain = 200 m flight ceiling",
        transform=axis.transAxes, fontsize=9.5, color="#7F0000",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
    )
    axis.set(
        xlim=(-50.0, 2600.0),
        ylim=(-850.0, 650.0),
        zlim=(0.0, max(290.0, float(np.max(terrain)) + 10.0)),
        xlabel="x (m)", ylabel="y (m)", zlabel="h (m)",
        title="3D Elliptical-Ridge Detour Animation",
    )
    axis.view_init(elev=29, azim=-63)
    axis.set_box_aspect((1.58, 0.90, 0.52))
    axis.legend(loc="upper right", fontsize=8.2, ncols=2)

    powered_indices = np.flatnonzero(powered_mask)
    first_glide_index = int(powered_indices[-1])

    def update(frame_index: int):
        point = positions[frame_index]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        if frame_index <= first_glide_index:
            trace = positions[:frame_index + 1]
            powered_trace.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            glide_halo.set_data_3d([], [], [])
            glide_trace.set_data_3d([], [], [])
            phase = "POWERED CLIMB"
            phase_color = "#2B8CBE"
        else:
            powered_trace.set_data_3d(
                positions[:first_glide_index + 1, 0],
                positions[:first_glide_index + 1, 1],
                positions[:first_glide_index + 1, 2],
            )
            trace = positions[first_glide_index:frame_index + 1]
            glide_halo.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            glide_trace.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            phase = "GLIDE / LATERAL DETOUR"
            phase_color = "#D01C8B"
        ground = float(terrain_interpolator(point[None, :2])[0])
        clearance = point[2] - ground
        status_text.set_text(
            f"{phase}\n"
            f"mission time: {frame_times[frame_index]:6.1f} / "
            f"{frame_times[-1]:.1f} s\n"
            f"position: ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.1f}) m\n"
            f"terrain clearance: {clearance:.1f} m\n"
            f"cumulative PoD: {100.0 * cumulative_pod[frame_index]:.3f}% "
            f"/ {100.0 * cumulative_pod[-1]:.3f}%"
        )
        status_text.set_color(phase_color)
        return powered_trace, glide_halo, glide_trace, vehicle, status_text

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
        dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(animation._fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
