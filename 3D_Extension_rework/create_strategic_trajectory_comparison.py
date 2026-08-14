"""Create one high-resolution 3D trajectory comparison for strategic baselines."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from .geometry import build_geometry, terrain_height
from .stackelberg import evaluate_defender_position
from .strategic_baselines import compute_nominal_attacker_response
from .terrain_scenarios import build_scenario_configuration


ROOT = Path(__file__).resolve().parent
RESULT_PATH = (
    ROOT / "results" / "stage_11_strategic_baselines"
    / "centered_single_hill" / "strategic_baseline_results.json"
)
ARRAY_PATH = RESULT_PATH.with_name("strategic_trajectory_comparison.npz")
FIGURE_PATH = (
    ROOT / "figures" / "stage_11_strategic_baselines"
    / "centered_single_hill_trajectory_comparison_3d_500dpi.png"
)


def _combined_path(evaluation: dict) -> np.ndarray:
    pipeline = evaluation["pipeline"]
    powered = np.asarray(pipeline["trajectory"]["powered_path"], dtype=float)
    replay = np.asarray(pipeline["continuous_replay"]["trajectory"], dtype=float)
    return np.vstack((powered, replay[1:]))


def _generate_arrays(configuration: dict, summary: dict) -> dict[str, np.ndarray]:
    fixed_xy = tuple(summary["selected_sensor_xy_m"]["fixed"])
    coverage_xy = tuple(summary["selected_sensor_xy_m"]["coverage_only"])
    nominal_xy = tuple(summary["selected_sensor_xy_m"]["nominal_path"])

    print(f"fixed/Stackelberg full response: {fixed_xy}", flush=True)
    fixed = evaluate_defender_position(fixed_xy, configuration, retain_full_pipeline=True)
    print(f"coverage-only full response: {coverage_xy}", flush=True)
    coverage = evaluate_defender_position(
        coverage_xy, configuration, retain_full_pipeline=True,
    )
    print("nominal time-only assumed response", flush=True)
    nominal = compute_nominal_attacker_response(configuration, fixed_xy)
    geometry = fixed["pipeline"]["geometry"]
    nominal_sensor_height = float(terrain_height(
        geometry["terrain_model"], nominal_xy[0], nominal_xy[1],
    ))
    arrays = {
        "terrain_x": np.asarray(geometry["x_grid"]),
        "terrain_y": np.asarray(geometry["y_grid"]),
        "terrain_height": np.asarray(geometry["terrain_height"]),
        "fixed_stack_path": _combined_path(fixed),
        "coverage_path": _combined_path(coverage),
        "nominal_assumed_path": _combined_path(nominal),
        "fixed_stack_sensor": np.asarray(fixed["summary"]["sensor_position_m"]),
        "coverage_sensor": np.asarray(coverage["summary"]["sensor_position_m"]),
        "nominal_sensor": np.asarray([*nominal_xy, nominal_sensor_height]),
        "fixed_stack_switch": np.asarray(fixed["summary"]["switching_point_m"]),
        "coverage_switch": np.asarray(coverage["summary"]["switching_point_m"]),
        "nominal_assumed_switch": np.asarray(nominal["summary"]["switching_point_m"]),
        "launch": np.asarray(geometry["launch_position"]),
        "goal": np.asarray(geometry["goal_position"]),
    }
    ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAY_PATH, **arrays)
    return arrays


def _load_or_generate(configuration: dict, summary: dict) -> dict[str, np.ndarray]:
    if ARRAY_PATH.exists():
        with np.load(ARRAY_PATH) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    return _generate_arrays(configuration, summary)


def _terrain_colors(terrain: np.ndarray) -> np.ndarray:
    return LightSource(azdeg=315.0, altdeg=40.0).shade(
        terrain, cmap=plt.get_cmap("terrain"), vert_exag=0.8,
        vmin=0.0, vmax=float(np.max(terrain)), blend_mode="soft",
    )


def _plot_path(axis, path: np.ndarray, color: str, label: str, *, dashed=False) -> None:
    linestyle = "--" if dashed else "-"
    axis.plot(
        path[:, 0], path[:, 1], path[:, 2], color="black",
        linewidth=7.2, linestyle=linestyle, alpha=0.65, zorder=9,
    )
    axis.plot(
        path[:, 0], path[:, 1], path[:, 2], color=color,
        linewidth=4.5, linestyle=linestyle, label=label, zorder=10,
    )


def create_figure(configuration: dict, arrays: dict[str, np.ndarray]) -> plt.Figure:
    figure = plt.figure(figsize=(15.2, 9.0))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    mesh_x, mesh_y = np.meshgrid(
        arrays["terrain_x"], arrays["terrain_y"], indexing="ij",
    )
    terrain = arrays["terrain_height"]
    axis.plot_surface(
        mesh_x, mesh_y, terrain, facecolors=_terrain_colors(terrain),
        linewidth=0.0, antialiased=True, shade=False, alpha=0.94, zorder=1,
    )
    axis.plot_wireframe(
        mesh_x, mesh_y, terrain + 0.7, rstride=10, cstride=8,
        color=(0.15, 0.12, 0.08, 0.14), linewidth=0.3, zorder=2,
    )

    _plot_path(
        axis, arrays["fixed_stack_path"], "#0072b2",
        "Fixed = Stackelberg: adaptive Bellman path",
    )
    _plot_path(
        axis, arrays["coverage_path"], "#8e63b6",
        "Coverage-only: adaptive Bellman path",
    )
    _plot_path(
        axis, arrays["nominal_assumed_path"], "#f28e2b",
        "Nominal-path: time-only assumed path", dashed=True,
    )

    sensor_specs = (
        ("fixed_stack_sensor", "#0072b2", "Fixed = Stackelberg sensor"),
        ("coverage_sensor", "#8e63b6", "Coverage-only sensor"),
        ("nominal_sensor", "#f28e2b", "Nominal-path sensor (adaptive infeasible)"),
    )
    for key, color, label in sensor_specs:
        point = arrays[key]
        axis.scatter(
            *point, marker="^", s=175, color=color, edgecolor="black",
            linewidth=1.2, label=label, zorder=15,
        )
    switch_specs = (
        ("fixed_stack_switch", "#0072b2"),
        ("coverage_switch", "#8e63b6"),
        ("nominal_assumed_switch", "#f28e2b"),
    )
    for key, color in switch_specs:
        point = arrays[key]
        axis.scatter(
            *point, marker="*", s=220, color=color, edgecolor="black",
            linewidth=1.0, zorder=16,
        )
    axis.scatter(
        *arrays["launch"], marker="s", s=105, color="black", edgecolor="white",
        linewidth=1.0, label="Launch", zorder=16,
    )
    axis.scatter(
        *arrays["goal"], marker="X", s=155, color="#1a9850", edgecolor="black",
        linewidth=1.1, label="Goal", zorder=16,
    )

    environment = configuration["environment"]
    x_bounds = environment["x_bounds_m"]
    y_bounds = environment["y_bounds_m"]
    h_bounds = environment["h_bounds_m"]
    x_padding = 0.018 * (x_bounds[1] - x_bounds[0])
    y_padding = 0.025 * (y_bounds[1] - y_bounds[0])
    axis.set(
        xlim=(x_bounds[0] - x_padding, x_bounds[1] + x_padding),
        ylim=(y_bounds[0] - y_padding, y_bounds[1] + y_padding),
        zlim=(h_bounds[0], h_bounds[1] + 0.02 * (h_bounds[1] - h_bounds[0])),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
    )
    axis.set_box_aspect((3.0, 1.0, 0.88), zoom=1.13)
    axis.view_init(elev=35.0, azim=-67.0)
    axis.set_title(
        "Centered Single-Hill: Strategic Trajectory Comparison\n"
        "solid = realized adaptive response; dashed = nominal time-only assumption",
        fontsize=15, fontweight="bold", pad=17,
    )
    axis.legend(
        loc="upper left", ncols=2, fontsize=8.6, framealpha=0.96,
    )
    axis.text2D(
        0.68, 0.035,
        "Nominal-path selected sensor has no feasible adaptive response\n"
        "under the current LOS-tangent switching contract.",
        transform=axis.transAxes, fontsize=9.5, color="#a50f15",
        bbox={"facecolor": "white", "edgecolor": "#a50f15", "alpha": 0.94},
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.90)
    return figure


def main() -> None:
    if not RESULT_PATH.exists():
        raise FileNotFoundError("Run the centered single-hill strategic test first")
    summary = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    configuration = build_scenario_configuration("centered_single_hill")
    arrays = _load_or_generate(configuration, summary)
    figure = create_figure(configuration, arrays)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(ARRAY_PATH, flush=True)
    print(FIGURE_PATH, flush=True)


if __name__ == "__main__":
    main()
