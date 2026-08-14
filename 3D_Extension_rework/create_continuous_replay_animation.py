"""Create a zoomed 3D animation of the validated continuous replay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry


ROOT = Path(__file__).resolve().parent
TRAJECTORY_DIR = ROOT / "results" / "stage_6_trajectory"
REPLAY_DIR = ROOT / "results" / "stage_7_continuous_replay"
OUTPUT_PATH = ROOT / "figures" / "stage_7_continuous_replay_3d_animation.gif"
FPS = 14
DISPLAY_SECONDS = 12.0


def _last_output(function, arguments: list[np.ndarray]) -> np.ndarray:
    count = int(arguments[0].size)
    outputs = function.map(count)(*(value.reshape(1, count) for value in arguments))
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.asarray(values[-1], dtype=float).reshape(count)


def _cumulative_trapezoid(rate: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros(rate.size, dtype=float)
    result[1:] = np.cumsum(0.5 * (rate[:-1] + rate[1:]) * np.diff(time))
    return result


def _mission_frames(
    powered_path: np.ndarray,
    glide_path: np.ndarray,
    powered_time: float,
    glide_durations: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    powered_lengths = np.linalg.norm(np.diff(powered_path, axis=0), axis=1)
    powered_clock = np.concatenate(([0.0], np.cumsum(powered_lengths)))
    powered_clock *= powered_time / powered_clock[-1]
    glide_clock = powered_time + np.concatenate(([0.0], np.cumsum(glide_durations)))
    mission_clock = np.concatenate((powered_clock, glide_clock[1:]))
    mission_path = np.vstack((powered_path, glide_path[1:]))
    frame_time = np.linspace(0.0, float(mission_clock[-1]), frame_count)
    frame_position = np.column_stack([
        np.interp(frame_time, mission_clock, mission_path[:, coordinate])
        for coordinate in range(3)
    ])
    return frame_time, frame_position, frame_time <= powered_time


def _cumulative_hazard_history(
    configuration: dict,
    geometry: dict,
    detection: dict,
    powered_path: np.ndarray,
    glide_path: np.ndarray,
    powered_time: float,
    durations: np.ndarray,
    speeds: np.ndarray,
    gammas: np.ndarray,
    headings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sensor = np.asarray(geometry["sensor_position"])
    functions = detection["functions"]
    powered_fraction = np.linspace(0.0, 1.0, 301)
    powered_points = powered_path[0][None, :] + powered_fraction[:, None] * (
        powered_path[-1] - powered_path[0]
    )[None, :]
    powered_sample_time = powered_fraction * powered_time
    powered_rate = _last_output(functions["powered_detection_components"], [
        powered_points[:, 0], powered_points[:, 1], powered_points[:, 2],
        np.full(powered_fraction.size, configuration["vehicle"]["powered_speed_mps"]),
        np.full(powered_fraction.size, sensor[0]),
        np.full(powered_fraction.size, sensor[1]),
        np.full(powered_fraction.size, sensor[2]),
    ])
    powered_hazard = _cumulative_trapezoid(powered_rate, powered_sample_time)
    history_time = list(powered_sample_time)
    history_hazard = list(powered_hazard)
    time_offset = powered_time
    hazard_offset = float(powered_hazard[-1])
    edge_fraction = np.linspace(0.0, 1.0, 101)
    for edge_index, duration in enumerate(durations):
        points = glide_path[edge_index][None, :] + edge_fraction[:, None] * (
            glide_path[edge_index + 1] - glide_path[edge_index]
        )[None, :]
        local_time = edge_fraction * float(duration)
        rate = _last_output(functions["glide_detection_components"], [
            points[:, 0], points[:, 1], points[:, 2],
            np.full(edge_fraction.size, speeds[edge_index]),
            np.full(edge_fraction.size, gammas[edge_index]),
            np.full(edge_fraction.size, headings[edge_index]),
            np.full(edge_fraction.size, sensor[0]),
            np.full(edge_fraction.size, sensor[1]),
            np.full(edge_fraction.size, sensor[2]),
        ])
        local_hazard = hazard_offset + _cumulative_trapezoid(rate, local_time)
        history_time.extend((time_offset + local_time[1:]).tolist())
        history_hazard.extend(local_hazard[1:].tolist())
        time_offset += float(duration)
        hazard_offset = float(local_hazard[-1])
    return np.asarray(history_time), np.asarray(history_hazard)


def create_animation() -> FuncAnimation:
    with np.load(TRAJECTORY_DIR / "optimal_physical_trajectory.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    with (TRAJECTORY_DIR / "summary.json").open(encoding="utf-8") as handle:
        trajectory_summary = json.load(handle)
    with (REPLAY_DIR / "summary.json").open(encoding="utf-8") as handle:
        replay_summary = json.load(handle)

    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    powered = arrays["powered_path"]
    glide = arrays["glide_trajectory"]
    durations = arrays["duration_profile_s"]
    speeds = arrays["speed_profile_mps"]
    gammas = arrays["gamma_profile_rad"]
    headings = arrays["heading_profile_rad"]
    powered_time = float(trajectory_summary["mission"]["powered_time_s"])
    frame_count = int(round(FPS * DISPLAY_SECONDS))
    frame_time, frame_position, powered_mask = _mission_frames(
        powered, glide, powered_time, durations, frame_count,
    )
    hazard_time, cumulative_hazard = _cumulative_hazard_history(
        configuration, geometry, detection, powered, glide, powered_time,
        durations, speeds, gammas, headings,
    )
    frame_hazard = np.interp(frame_time, hazard_time, cumulative_hazard)
    frame_pod = 1.0 - np.exp(-frame_hazard)
    terrain_interpolator = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]), geometry["terrain_height"],
        bounds_error=False, fill_value=np.nan,
    )
    los_interpolator = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]), geometry["los_boundary_height"],
        bounds_error=False, fill_value=np.nan,
    )
    mesh_x, mesh_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    full_path = np.vstack((powered, glide[1:]))
    switch = arrays["switching_point"]
    sensor = geometry["sensor_position"]
    goal = geometry["goal_position"]
    contacts = geometry["tangent_manifold"]["contact_points"]

    figure = plt.figure(figsize=(13.6, 7.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    axis.plot_surface(
        mesh_x, mesh_y, geometry["terrain_height"],
        color="#aa9068", alpha=0.94, linewidth=0.15,
        edgecolor=(0.2, 0.15, 0.08, 0.14), antialiased=True, zorder=1,
    )
    axis.plot(contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.5,
              color="#d000d0", linewidth=2.8, alpha=0.88,
              label="Tangent-contact curve", zorder=6)
    axis.plot(full_path[:, 0], full_path[:, 1], full_path[:, 2],
              color="white", linewidth=6.0, alpha=0.72, zorder=7)
    axis.plot(powered[:, 0], powered[:, 1], powered[:, 2],
              color="#f28e2b", linewidth=1.8, linestyle="--", alpha=0.42,
              label="Planned powered path", zorder=8)
    axis.plot(glide[:, 0], glide[:, 1], glide[:, 2],
              color="#0072b2", linewidth=1.8, linestyle="--", alpha=0.42,
              label="Planned glide path", zorder=8)
    powered_halo, = axis.plot([], [], [], color="black", linewidth=7.0, alpha=0.88, zorder=10)
    powered_trace, = axis.plot([], [], [], color="#f28e2b", linewidth=4.5,
                               label="Powered replay", zorder=11)
    glide_halo, = axis.plot([], [], [], color="black", linewidth=7.0, alpha=0.88, zorder=10)
    glide_trace, = axis.plot([], [], [], color="#0072b2", linewidth=4.5,
                             label="Glide replay", zorder=11)
    vehicle, = axis.plot([], [], [], marker="D", linestyle="None", markersize=10.5,
                         markerfacecolor="#ffd92f", markeredgecolor="black",
                         markeredgewidth=1.3, label="Vehicle", zorder=15)
    axis.scatter(*geometry["launch_position"], marker="s", s=105,
                 color="black", edgecolor="white", label="Launch", zorder=13)
    axis.scatter(*switch, marker="*", s=245, color="#ffd92f",
                 edgecolor="black", label="Switch", zorder=14)
    axis.scatter(*sensor, marker="^", s=145, color="#d7191c",
                 edgecolor="black", label="Sensor", zorder=14)
    axis.scatter(*goal, marker="X", s=140, color="#1a9850",
                 edgecolor="black", label="Goal", zorder=14)
    status_text = axis.text2D(
        0.015, 0.965, "", transform=axis.transAxes, ha="left", va="top",
        fontsize=11.0,
        bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.95},
    )
    axis.text2D(
        0.66, 0.018,
        "Validated unsnapped replay\n"
        f"max endpoint drift = {replay_summary['validation']['metrics']['maximum_endpoint_drift_m']:.2e} m",
        transform=axis.transAxes, fontsize=9.2, color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    axis.set(
        xlim=(-40.0, 3040.0), ylim=(-40.0, 1040.0), zlim=(0.0, 410.0),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title=(
            "3D Continuous Replay of the Bellman-Optimal Mission\n"
            "same actions, continuous propagation, no grid reset"
        ),
    )
    axis.set_box_aspect((3.0, 1.05, 0.76))
    # A higher, slightly more cross-map viewpoint makes the isotropic
    # Gaussian hill's circular horizontal footprint legible while retaining
    # enough elevation perspective to read the climb and glide.
    axis.view_init(elev=42.0, azim=-68.0)
    axis.legend(loc="upper right", fontsize=8.0, ncols=2, framealpha=0.94)
    first_glide_frame = int(np.flatnonzero(powered_mask)[-1])

    def update(frame_index: int):
        point = frame_position[frame_index]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        if frame_index <= first_glide_frame:
            trace = frame_position[: frame_index + 1]
            powered_halo.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            powered_trace.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            glide_halo.set_data_3d([], [], [])
            glide_trace.set_data_3d([], [], [])
            phase = "POWERED / OCCLUDED"
            phase_color = "#c35a00"
            boundary_name = "occlusion margin"
            boundary_margin = float(los_interpolator(point[None, :2])[0] - point[2])
        else:
            powered_segment = frame_position[: first_glide_frame + 1]
            powered_halo.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            powered_trace.set_data_3d(
                powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2]
            )
            trace = frame_position[first_glide_frame : frame_index + 1]
            glide_halo.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            glide_trace.set_data_3d(trace[:, 0], trace[:, 1], trace[:, 2])
            phase = "GLIDE / LOS-VISIBLE"
            phase_color = "#005a91"
            boundary_name = "LOS margin"
            boundary_margin = float(point[2] - los_interpolator(point[None, :2])[0])
        terrain = float(terrain_interpolator(point[None, :2])[0])
        clearance = point[2] - terrain
        status_text.set_text(
            f"{phase}\n"
            f"mission time: {frame_time[frame_index]:6.1f} / {frame_time[-1]:.1f} s\n"
            f"position: ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.1f}) m\n"
            f"terrain clearance: {clearance:.1f} m\n"
            f"{boundary_name}: {boundary_margin:.1f} m\n"
            f"cumulative PoD: {100.0 * frame_pod[frame_index]:.3f}%"
        )
        status_text.set_color(phase_color)
        progress = frame_index / max(frame_count - 1, 1)
        axis.view_init(elev=42.0, azim=-70.0 + 4.0 * progress)
        return powered_halo, powered_trace, glide_halo, glide_trace, vehicle, status_text

    return FuncAnimation(
        figure, update, frames=frame_count,
        interval=1000.0 / FPS, blit=False, repeat=True,
    )


def main() -> None:
    if not (TRAJECTORY_DIR / "optimal_physical_trajectory.npz").exists():
        raise FileNotFoundError("Run visualize_trajectory first")
    if not (REPLAY_DIR / "summary.json").exists():
        raise FileNotFoundError("Run visualize_continuous_replay first")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    animation = create_animation()
    animation.save(
        OUTPUT_PATH, writer=PillowWriter(fps=FPS), dpi=100,
        savefig_kwargs={"facecolor": "white"},
    )
    plt.close(animation._fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
