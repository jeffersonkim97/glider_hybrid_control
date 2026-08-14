"""Create standalone 3D plot and animation for the selected Stackelberg result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource

from .configuration import build_configuration
from .create_continuous_replay_animation import (
    _cumulative_hazard_history,
    _mission_frames,
)
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry, terrain_height
from .stackelberg import configuration_for_sensor
from .terrain_scenarios import build_scenario_configuration


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_8_stackelberg"
FIGURE_DIR = ROOT / "figures"
SUMMARY_PATH = RESULT_DIR / "stackelberg_solution.json"
NPZ_PATH = RESULT_DIR / "stackelberg_solution.npz"
FPS = 14
DISPLAY_SECONDS = 11.0


def _scenario_tag(configuration: dict) -> str:
    scenario_id = configuration["environment"]["terrain"]["scenario_id"]
    defender = configuration["cost"]["defender"]
    number = lambda value: f"{float(value):.1f}".replace(".", "p")
    return (
        f"{scenario_id}_wpod{number(defender['w_pod'])}"
        f"_wcov{number(defender['w_coverage'])}"
    )


def _load_result(
    summary_path: Path = SUMMARY_PATH,
    npz_path: Path = NPZ_PATH,
    base_configuration: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray], dict, dict, dict]:
    if not summary_path.exists() or not npz_path.exists():
        raise FileNotFoundError("Run the 0.9/0.1 Stage-8 Stackelberg solve first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    required = {
        "powered_path", "continuous_replay_trajectory", "duration_profile_s",
        "speed_profile_mps", "gamma_profile_rad", "heading_profile_rad",
        "powered_time_s", "switching_point",
    }
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"Stage-8 NPZ is missing animation profiles: {missing}")
    base = build_configuration() if base_configuration is None else base_configuration
    weights = base["cost"]["defender"]
    if abs(weights["w_pod"] - 0.9) > 1.0e-12 or abs(weights["w_coverage"] - 0.1) > 1.0e-12:
        raise ValueError("This output is reserved for Defender weights 0.9/0.1")
    sensor_xy = tuple(summary["optimal_sensor_position_m"][:2])
    configuration = configuration_for_sensor(base, sensor_xy)
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    expected_height = float(terrain_height(
        geometry["terrain_model"], sensor_xy[0], sensor_xy[1],
    ))
    if abs(expected_height - summary["optimal_sensor_position_m"][2]) > 1.0e-9:
        raise ValueError("Summary sensor height does not match current terrain")
    return summary, arrays, configuration, geometry, detection


def _terrain_facecolors(terrain: np.ndarray) -> np.ndarray:
    light = LightSource(azdeg=315.0, altdeg=38.0)
    return light.shade(
        terrain, cmap=plt.get_cmap("terrain"), vert_exag=0.75,
        vmin=0.0, vmax=max(float(np.max(terrain)), 1.0), blend_mode="soft",
    )


def _draw_scene(
    axis, geometry: dict, arrays: dict[str, np.ndarray], summary: dict,
    *, animation_mode: bool = False,
) -> None:
    mesh_x, mesh_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    terrain = geometry["terrain_height"]
    axis.plot_surface(
        mesh_x, mesh_y, terrain, facecolors=_terrain_facecolors(terrain),
        linewidth=0.0, antialiased=True, shade=False, alpha=0.93, zorder=1,
    )
    axis.plot_wireframe(
        mesh_x, mesh_y, terrain + 0.8, rstride=10, cstride=8,
        color=(0.18, 0.13, 0.07, 0.16), linewidth=0.35, zorder=2,
    )
    contacts = geometry["tangent_manifold"]["contact_points"]
    axis.plot(
        contacts[:, 0], contacts[:, 1], contacts[:, 2] + 2.0,
        color="#c51b8a", linewidth=2.8, alpha=0.9,
        label="Terrain tangent-contact manifold", zorder=6,
    )
    powered = arrays["powered_path"]
    replay = arrays["continuous_replay_trajectory"]
    if animation_mode:
        axis.plot(powered[:, 0], powered[:, 1], powered[:, 2],
                  color="#f28e2b", linewidth=2.2, linestyle="--", alpha=0.32,
                  label="Planned powered path", zorder=7)
        axis.plot(replay[:, 0], replay[:, 1], replay[:, 2],
                  color="#0072b2", linewidth=2.2, linestyle="--", alpha=0.32,
                  label="Planned glide path", zorder=7)
    else:
        axis.plot(powered[:, 0], powered[:, 1], powered[:, 2],
                  color="black", linewidth=7.5, alpha=0.75, zorder=8)
        axis.plot(powered[:, 0], powered[:, 1], powered[:, 2],
                  color="#f28e2b", linewidth=4.8, label="Powered phase", zorder=9)
        axis.plot(replay[:, 0], replay[:, 1], replay[:, 2],
                  color="black", linewidth=7.5, alpha=0.75, zorder=8)
        axis.plot(replay[:, 0], replay[:, 1], replay[:, 2],
                  color="#0072b2", linewidth=4.8,
                  label="Continuous glide replay", zorder=9)
    axis.scatter(replay[1:-1, 0], replay[1:-1, 1], replay[1:-1, 2],
                 s=35, color="white", edgecolor="#0072b2", linewidth=1.1, zorder=10)
    sensor = np.asarray(summary["optimal_sensor_position_m"], dtype=float)
    switch = arrays["switching_point"]
    launch = geometry["launch_position"]
    goal = geometry["goal_position"]
    axis.scatter(*launch, marker="s", s=115, color="black", edgecolor="white",
                 linewidth=1.1, label="Launch", zorder=13)
    axis.scatter(*switch, marker="*", s=310, color="#ffd92f", edgecolor="black",
                 linewidth=1.2, label="Switch", zorder=14)
    axis.scatter(*sensor, marker="^", s=190, color="#d7191c", edgecolor="black",
                 linewidth=1.1, label="Optimal sensor", zorder=14)
    axis.scatter(*goal, marker="X", s=170, color="#1a9850", edgecolor="black",
                 linewidth=1.1, label="Goal", zorder=14)
    terrain_interp = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]), terrain,
        bounds_error=False, fill_value=np.nan,
    )
    for point in replay[:-1]:
        ground = float(terrain_interp(point[None, :2])[0])
        axis.plot([point[0], point[0]], [point[1], point[1]], [ground, point[2]],
                  color="#4d4d4d", linestyle=":", linewidth=0.8, alpha=0.38, zorder=4)
    environment = geometry["configuration"]["environment"]
    x_bounds = environment["x_bounds_m"]
    y_bounds = environment["y_bounds_m"]
    h_bounds = environment["h_bounds_m"]
    x_pad = 0.02 * (x_bounds[1] - x_bounds[0])
    y_pad = 0.02 * (y_bounds[1] - y_bounds[0])
    h_pad = 0.025 * (h_bounds[1] - h_bounds[0])
    axis.set(
        xlim=(x_bounds[0] - x_pad, x_bounds[1] + x_pad),
        ylim=(y_bounds[0] - y_pad, y_bounds[1] + y_pad),
        zlim=(h_bounds[0], h_bounds[1] + h_pad),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
    )
    spans = np.asarray([
        x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0],
        2.4 * (h_bounds[1] - h_bounds[0]),
    ])
    axis.set_box_aspect(tuple(spans / spans[1]), zoom=1.24 if animation_mode else 1.06)
    axis.view_init(elev=38.0, azim=-66.0)


def create_static_plot(
    summary: dict, arrays: dict[str, np.ndarray], geometry: dict,
) -> plt.Figure:
    figure = plt.figure(figsize=(15.2, 8.8))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    _draw_scene(axis, geometry, arrays, summary)
    attacker = summary["final_attacker_summary"]
    scenario = geometry["configuration"]["environment"]["terrain"]["scenario_id"]
    axis.set_title(
        f"3D Stackelberg Equilibrium - {scenario.replace('_', ' ').title()}\n"
        f"sensor = ({attacker['sensor_position_m'][0]:.0f}, "
        f"{attacker['sensor_position_m'][1]:.0f}, "
        f"{attacker['sensor_position_m'][2]:.2f}) m; "
        f"mission PoD = {100.0 * attacker['mission_pod']:.3f}%",
        fontsize=15, fontweight="bold", pad=18,
    )
    axis.legend(loc="upper left", ncols=2, fontsize=9.2, framealpha=0.96)
    axis.text2D(
        0.72, 0.035,
        f"Switch: ({arrays['switching_point'][0]:.0f}, "
        f"{arrays['switching_point'][1]:.0f}, {arrays['switching_point'][2]:.1f}) m\n"
        f"Mission time: {attacker['mission_time_s']:.1f} s\n"
        "Continuous replay validated",
        transform=axis.transAxes, fontsize=10.0,
        bbox={"facecolor": "white", "edgecolor": "0.45", "alpha": 0.92},
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
    return figure


def create_animation(
    summary: dict, arrays: dict[str, np.ndarray], configuration: dict,
    geometry: dict, detection: dict,
) -> FuncAnimation:
    powered = arrays["powered_path"]
    replay = arrays["continuous_replay_trajectory"]
    powered_time = float(arrays["powered_time_s"])
    durations = arrays["duration_profile_s"]
    speeds = arrays["speed_profile_mps"]
    gammas = arrays["gamma_profile_rad"]
    headings = arrays["heading_profile_rad"]
    frame_count = int(round(FPS * DISPLAY_SECONDS))
    frame_time, frame_position, powered_mask = _mission_frames(
        powered, replay, powered_time, durations, frame_count,
    )
    hazard_time, cumulative_hazard = _cumulative_hazard_history(
        configuration, geometry, detection, powered, replay, powered_time,
        durations, speeds, gammas, headings,
    )
    frame_pod = 1.0 - np.exp(-np.interp(frame_time, hazard_time, cumulative_hazard))
    terrain_interp = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]), geometry["terrain_height"],
        bounds_error=False, fill_value=np.nan,
    )
    los_interp = RegularGridInterpolator(
        (geometry["x_grid"], geometry["y_grid"]), geometry["los_boundary_height"],
        bounds_error=False, fill_value=np.nan,
    )

    figure = plt.figure(figsize=(14.0, 8.3))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    _draw_scene(axis, geometry, arrays, summary, animation_mode=True)
    powered_halo, = axis.plot([], [], [], color="black", linewidth=8.0, zorder=16)
    powered_trace, = axis.plot([], [], [], color="#f28e2b", linewidth=5.0, zorder=17)
    glide_halo, = axis.plot([], [], [], color="black", linewidth=8.0, zorder=16)
    glide_trace, = axis.plot([], [], [], color="#0072b2", linewidth=5.0, zorder=17)
    vehicle, = axis.plot(
        [], [], [], marker="D", linestyle="None", markersize=11.5,
        markerfacecolor="#fff200", markeredgecolor="black", markeredgewidth=1.4,
        label="Agent", zorder=20,
    )
    status = axis.text2D(
        0.015, 0.965, "", transform=axis.transAxes, ha="left", va="top",
        fontsize=10.8,
        bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.95},
    )
    axis.set_title(
        "3D Stackelberg Mission Replay - Defender Weights 0.9/0.1\n"
        f"scenario = {geometry['configuration']['environment']['terrain']['scenario_id'].replace('_', ' ')}",
        fontsize=15, fontweight="bold", pad=16,
    )
    axis.legend(loc="upper right", ncols=2, fontsize=8.3, framealpha=0.95)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.92)
    first_glide_frame = int(np.flatnonzero(powered_mask)[-1])

    def update(frame_index: int):
        point = frame_position[frame_index]
        vehicle.set_data_3d([point[0]], [point[1]], [point[2]])
        powered_segment = frame_position[: min(frame_index, first_glide_frame) + 1]
        powered_halo.set_data_3d(
            powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2],
        )
        powered_trace.set_data_3d(
            powered_segment[:, 0], powered_segment[:, 1], powered_segment[:, 2],
        )
        if frame_index > first_glide_frame:
            glide_segment = frame_position[first_glide_frame: frame_index + 1]
            glide_halo.set_data_3d(
                glide_segment[:, 0], glide_segment[:, 1], glide_segment[:, 2],
            )
            glide_trace.set_data_3d(
                glide_segment[:, 0], glide_segment[:, 1], glide_segment[:, 2],
            )
            phase = "GLIDE"
            color = "#005a91"
        else:
            glide_halo.set_data_3d([], [], [])
            glide_trace.set_data_3d([], [], [])
            phase = "POWERED"
            color = "#bd5700"
        ground = float(terrain_interp(point[None, :2])[0])
        los_height = float(los_interp(point[None, :2])[0])
        status.set_text(
            f"{phase}\n"
            f"time: {frame_time[frame_index]:6.1f} / {frame_time[-1]:.1f} s\n"
            f"position: ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.1f}) m\n"
            f"terrain clearance: {point[2] - ground:.1f} m\n"
            f"LOS margin (h-H_LOS): {point[2] - los_height:.1f} m\n"
            f"cumulative PoD: {100.0 * frame_pod[frame_index]:.3f}%"
        )
        status.set_color(color)
        progress = frame_index / max(frame_count - 1, 1)
        axis.view_init(elev=38.0, azim=-69.0 + 6.0 * progress)
        return powered_halo, powered_trace, glide_halo, glide_trace, vehicle, status

    return FuncAnimation(
        figure, update, frames=frame_count,
        interval=1000.0 / FPS, blit=False, repeat=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("default", "two_hill", "goal_in_valley"),
        default="default",
    )
    parser.add_argument("--animation", action="store_true")
    args = parser.parse_args()
    if args.scenario == "default":
        summary_path, npz_path = SUMMARY_PATH, NPZ_PATH
        base = build_configuration()
        stage_prefix = "stage_8"
    else:
        result_dir = ROOT / "results" / "stage_10_multiterrain" / args.scenario
        summary_path = result_dir / "stackelberg_solution.json"
        npz_path = result_dir / "stackelberg_solution.npz"
        base = build_scenario_configuration(args.scenario)
        stage_prefix = "stage_10"
    summary, arrays, configuration, geometry, detection = _load_result(
        summary_path, npz_path, base,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tag = _scenario_tag(configuration)
    plot_path = FIGURE_DIR / f"{stage_prefix}_stackelberg_3d_plot_{tag}.png"
    plot_pdf_path = FIGURE_DIR / f"{stage_prefix}_stackelberg_3d_plot_{tag}.pdf"
    animation_path = FIGURE_DIR / f"{stage_prefix}_stackelberg_3d_animation_{tag}.gif"
    static = create_static_plot(summary, arrays, geometry)
    static.savefig(plot_path, dpi=240, bbox_inches="tight", facecolor="white")
    static.savefig(plot_pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(static)
    if args.animation:
        animation = create_animation(summary, arrays, configuration, geometry, detection)
        animation.save(
            animation_path, writer=PillowWriter(fps=FPS), dpi=105,
            savefig_kwargs={"facecolor": "white"},
        )
        plt.close(animation._fig)
    print(plot_path)
    print(plot_pdf_path)
    if args.animation:
        print(animation_path)


if __name__ == "__main__":
    main()
