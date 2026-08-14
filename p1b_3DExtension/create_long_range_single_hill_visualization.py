"""Visualize the validated 5 km single-hill continuous mission."""

from __future__ import annotations

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .detection import build_symbolic_detection_bundle
from .experiment_long_range_single_hill import (
    AIRSPACE_CEILING_M,
    OUTPUT_DIR,
    REPO_ROOT,
    RIDGE,
    SENSOR_XY,
    build_long_range_configuration,
)
from .geometry import build_geometry_bundle, terrain_height


FIGURE_DIR = REPO_ROOT / "result_3D_visualization"
PNG_PATH = FIGURE_DIR / "long_range_single_hill_5km.png"
PDF_PATH = FIGURE_DIR / "long_range_single_hill_5km.pdf"


def _last_mapped(function, arguments: tuple[np.ndarray, ...]) -> np.ndarray:
    count = arguments[0].size
    outputs = function.map(count)(
        *(np.asarray(argument).reshape(1, -1) for argument in arguments)
    )
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.maximum(0.0, np.asarray(values[-1]).reshape(-1))


def _mission_arrays(
    powered_path: np.ndarray,
    dense_time: np.ndarray,
    dense_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    configuration = build_long_range_configuration()
    geometry_bundle = build_geometry_bundle(configuration)
    detection_bundle = build_symbolic_detection_bundle(
        configuration, geometry_bundle,
    )
    geometry = geometry_bundle["primary_result"]
    sensor = np.asarray(geometry["sensor_position"])
    model = geometry["terrain_model"]
    powered_time = float(dense_time[0])
    powered_clock = np.linspace(0.0, powered_time, powered_path.shape[0])
    displacement = powered_path[-1] - powered_path[0]
    horizontal = float(np.linalg.norm(displacement[:2]))
    gamma = float(np.arctan2(displacement[2], horizontal))
    heading = float(np.arctan2(displacement[1], displacement[0]))
    speed = float(np.linalg.norm(displacement) / powered_time)
    count = powered_path.shape[0]
    function = detection_bundle["primary_result"]["functions"][
        "powered_total_detection_components"
    ]
    rates = _last_mapped(function, (
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        np.full(count, speed), np.full(count, gamma), np.full(count, heading),
        np.full(count, sensor[0]), np.full(count, sensor[1]),
        np.full(count, sensor[2]),
    ))
    increments = 0.5 * (rates[:-1] + rates[1:]) * np.diff(powered_clock)
    powered_hazard = np.concatenate(([0.0], np.cumsum(increments)))
    glide_hazard = dense_states[7] - dense_states[7, 0] + powered_hazard[-1]
    clock = np.concatenate((powered_clock, dense_time[1:]))
    positions = np.vstack((powered_path, dense_states[:3, 1:].T))
    hazard = np.concatenate((powered_hazard, glide_hazard[1:]))
    ground = np.asarray(terrain_height(
        model, positions[:, 0], positions[:, 1],
    ))
    return clock, positions, ground, 100.0 * (1.0 - np.exp(-hazard))


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "continuous_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("status_success", False):
        raise RuntimeError(
            "Refusing to visualize an unvalidated continuous trajectory"
        )
    if summary.get("switching_candidate_mode") != "los_boundary_surface":
        raise RuntimeError(
            "Refusing to visualize a trajectory that did not switch on the "
            "configured LOS boundary surface"
        )
    with np.load(OUTPUT_DIR / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain_grid = np.asarray(handle["terrain_height"])
        goal = np.asarray(handle["goal_position"])
    with np.load(OUTPUT_DIR / "continuous_trajectory.npz") as handle:
        powered_path = np.asarray(handle["powered_path"])
        dense_time = np.asarray(handle["dense_time"])
        dense_states = np.asarray(handle["dense_states"])
    clock, positions, ground, cumulative_pod = _mission_arrays(
        powered_path, dense_time, dense_states,
    )
    sensor = np.asarray(summary["sensor_position"])
    switch = np.asarray(summary["switching_point"])
    powered_time = float(summary["powered_time_s"])
    mesh_x, mesh_y = np.meshgrid(terrain_x, terrain_y, indexing="ij")

    figure = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.65, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)
    axis_altitude = figure.add_subplot(grid[0, 1])
    axis_pod = figure.add_subplot(grid[1, 1])

    axis_3d.plot_surface(
        mesh_x, mesh_y, terrain_grid, cmap="terrain", alpha=0.92,
        linewidth=0.18, edgecolor=(0.2, 0.2, 0.2, 0.18), zorder=0,
    )
    axis_3d.plot(
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        color="black", linewidth=6.0, alpha=0.82, zorder=9,
    )
    axis_3d.plot(
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        color="#d95f02", linewidth=3.8, label="Powered", zorder=10,
    )
    axis_3d.plot(
        dense_states[0], dense_states[1], dense_states[2],
        color="black", linewidth=6.0, alpha=0.82, zorder=9,
    )
    axis_3d.plot(
        dense_states[0], dense_states[1], dense_states[2],
        color="#1769aa", linewidth=3.8, label="Glide (continuous 3-DOF)", zorder=10,
    )
    axis_3d.scatter(*switch, s=80, color="#e31a1c", edgecolor="black",
                    label="Switch", zorder=13)
    axis_3d.scatter(*sensor, s=130, marker="^", color="#c51b7d",
                    edgecolor="black", label="Sensor", zorder=13)
    axis_3d.scatter(*goal, s=125, marker="X", color="#39a96b",
                    edgecolor="black", label="Goal", zorder=13)
    axis_3d.set(
        xlim=(-100.0, 5250.0), ylim=(-1050.0, 650.0),
        zlim=(0.0, 425.0), xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  Validated 5 km single-hill mission",
    )
    axis_3d.view_init(elev=27.0, azim=-63.0)
    axis_3d.set_box_aspect((2.65, 1.15, 0.72))
    axis_3d.legend(loc="upper left", framealpha=0.94)

    axis_altitude.fill_between(
        clock, 0.0, ground, color="#8c6d31", alpha=0.32,
        label="Terrain beneath route",
    )
    axis_altitude.plot(clock, ground, color="#6b4f1d", linewidth=1.5)
    axis_altitude.plot(clock, positions[:, 2], color="#2b6cb0", linewidth=2.6,
                       label="Vehicle altitude")
    axis_altitude.axhline(
        AIRSPACE_CEILING_M, color="#8b0000", linestyle="--", linewidth=1.5,
        label="400 m ceiling",
    )
    axis_altitude.axvline(powered_time, color="black", linestyle=":", linewidth=1.2)
    axis_altitude.set(
        xlabel="Mission time [s]", ylabel="Height [m]",
        title="B  Altitude versus terrain under the trajectory",
        ylim=(0.0, 425.0),
    )
    axis_altitude.grid(alpha=0.25)
    axis_altitude.legend(loc="upper right", fontsize=8.5)

    axis_pod.axvspan(0.0, powered_time, color="#d95f02", alpha=0.10,
                     label="Powered phase")
    axis_pod.axvspan(powered_time, clock[-1], color="#1769aa", alpha=0.08,
                     label="Glide phase")
    axis_pod.plot(clock, cumulative_pod, color="#6a3d9a", linewidth=2.7,
                  label=f"Final PoD = {100.0 * summary['mission_pod']:.3f}%")
    axis_pod.axvline(powered_time, color="black", linestyle=":", linewidth=1.2)
    axis_pod.set(
        xlabel="Mission time [s]", ylabel="Cumulative PoD [%]",
        title="C  Reintegrated cumulative detection probability",
    )
    axis_pod.grid(alpha=0.25)
    axis_pod.legend(loc="upper left", fontsize=8.5)

    figure.suptitle(
        "Single Hill: center 1500 m, peak 300 m, goal 5000 m, ceiling 400 m\n"
        f"Switch ({switch[0]:.0f}, {switch[1]:.0f}, {switch[2]:.1f}) m  |  "
        f"maximum altitude {summary['maximum_altitude_m']:.2f} m  |  "
        f"mission time {summary['mission_time_s']:.1f} s",
        fontsize=14.5, fontweight="bold",
    )
    figure.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
