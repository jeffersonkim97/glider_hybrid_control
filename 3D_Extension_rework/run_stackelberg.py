"""Run, checkpoint, export, and visualize the 3D Stackelberg search."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import build_configuration
from .stackelberg import solve_stackelberg_game


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_8_stackelberg"
FIGURE_DIR = ROOT / "figures"
SUMMARY_PATH = RESULT_DIR / "stackelberg_solution.json"
NPZ_PATH = RESULT_DIR / "stackelberg_solution.npz"
PNG_PATH = FIGURE_DIR / "stage_8_stackelberg_solution.png"
PDF_PATH = FIGURE_DIR / "stage_8_stackelberg_solution.pdf"


def _progress(event: dict) -> None:
    print(json.dumps(event), flush=True)


def _json_default(value):
    """Convert NumPy scalar/array values produced by validation to JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serializable_result(result: dict) -> dict:
    final = result["final_evaluation"]
    return {
        "optimal_sensor_position_m": result["optimal_sensor_position_m"],
        "defender_objective": result["defender_objective"],
        "final_attacker_summary": final["summary"],
        "final_objective_breakdown": final["objective_breakdown"],
        "search": {
            "best_summary": result["search"]["best_summary"],
            "evaluations": list(result["search"]["evaluations"]),
            "levels": list(result["search"]["levels"]),
            "metadata": result["search"]["metadata"],
        },
        "cache_stats": result["cache_stats"],
        "metadata": result["metadata"],
        "validation": result["validation"],
        "status": result["status"],
    }


def _create_figure(result: dict, *, stage_label: str = "Stage 8") -> plt.Figure:
    final = result["final_evaluation"]
    pipeline = final["pipeline"]
    geometry = pipeline["geometry"]
    trajectory = pipeline["trajectory"]
    evaluations = [item for item in result["search"]["evaluations"] if item["feasible"]]
    sensor_xy = np.asarray([item["sensor_position_m"][:2] for item in evaluations])
    objective = np.asarray([item["defender_objective"] for item in evaluations])
    coverage = np.asarray([item["coverage_volume_normalized"] for item in evaluations])
    pod_component = np.asarray([item["defender_pod_normalized"] for item in evaluations])
    best_sensor = np.asarray(result["optimal_sensor_position_m"])
    powered = np.asarray(trajectory["powered_path"])
    glide = np.asarray(trajectory["glide_trajectory"])

    figure = plt.figure(figsize=(16.0, 9.3), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(1.28, 1.0))
    axis_map = figure.add_subplot(layout[0, 0])
    axis_3d = figure.add_subplot(layout[1, 0], projection="3d", computed_zorder=False)
    axis_tradeoff = figure.add_subplot(layout[:, 1])

    mesh_x, mesh_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    terrain_max = float(np.max(geometry["terrain_height"]))
    contour_levels = np.linspace(
        terrain_max / 10.0, terrain_max, 10,
    ) if terrain_max > 0.0 else np.asarray([0.0])
    contour = axis_map.contour(
        mesh_x, mesh_y, geometry["terrain_height"],
        levels=contour_levels, colors="#8c6d31",
        linewidths=0.7, alpha=0.65,
    )
    axis_map.clabel(contour, inline=True, fontsize=7, fmt="%.0f m")
    scatter = axis_map.scatter(
        sensor_xy[:, 0], sensor_xy[:, 1], c=objective,
        cmap="viridis", s=88, edgecolor="black", linewidth=0.45,
    )
    axis_map.scatter(best_sensor[0], best_sensor[1], marker="*", s=310,
                     color="#ffd92f", edgecolor="black", linewidth=1.2,
                     label="Selected Defender sensor", zorder=10)
    bounds = pipeline["configuration"]["defender_search"]
    x_lower, x_upper = bounds["x_bounds_m"]
    y_lower, y_upper = bounds["y_bounds_m"]
    x_padding = 0.025 * (x_upper - x_lower)
    y_padding = 0.025 * (y_upper - y_lower)
    axis_map.set(
        xlim=(x_lower - x_padding, x_upper + x_padding),
        ylim=(y_lower - y_padding, y_upper + y_padding),
        xlabel=r"$x_{sensor}$ [m]", ylabel=r"$y_{sensor}$ [m]",
        title="A  Evaluated 2D Defender candidate region",
    )
    axis_map.grid(alpha=0.2)
    axis_map.legend(loc="upper left")
    figure.colorbar(scatter, ax=axis_map, label="Defender objective")

    axis_3d.plot_surface(
        mesh_x, mesh_y, geometry["terrain_height"], color="#aa9068",
        alpha=0.75, linewidth=0.12,
        edgecolor=(0.2, 0.15, 0.08, 0.12), zorder=1,
    )
    axis_3d.plot(powered[:, 0], powered[:, 1], powered[:, 2],
                 color="#f28e2b", linewidth=4.0, label="Powered path", zorder=8)
    axis_3d.plot(glide[:, 0], glide[:, 1], glide[:, 2],
                 color="#0072b2", linewidth=4.0, label="Glide path", zorder=8)
    axis_3d.scatter(*best_sensor, marker="^", s=150, color="#d7191c",
                    edgecolor="black", label="Optimal sensor", zorder=12)
    axis_3d.scatter(*geometry["goal_position"], marker="X", s=125,
                    color="#1a9850", edgecolor="black", label="Goal", zorder=12)
    axis_3d.scatter(*trajectory["switching_point"], marker="*", s=240,
                    color="#ffd92f", edgecolor="black", label="Switch", zorder=12)
    environment = pipeline["configuration"]["environment"]
    x_map_bounds = environment["x_bounds_m"]
    y_map_bounds = environment["y_bounds_m"]
    h_map_bounds = environment["h_bounds_m"]
    axis_3d.set(
        xlim=x_map_bounds, ylim=y_map_bounds, zlim=h_map_bounds,
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="B  Final Stackelberg sensor and Attacker response",
    )
    spans = np.asarray([
        x_map_bounds[1] - x_map_bounds[0],
        y_map_bounds[1] - y_map_bounds[0],
        2.0 * (h_map_bounds[1] - h_map_bounds[0]),
    ])
    axis_3d.set_box_aspect(tuple(spans / spans[1]))
    axis_3d.view_init(elev=37.0, azim=-66.0)
    axis_3d.legend(loc="upper left", fontsize=8.2, ncols=2)

    tradeoff = axis_tradeoff.scatter(
        coverage, pod_component, c=objective, cmap="viridis",
        s=105, edgecolor="black", linewidth=0.45,
    )
    selected = final["objective_breakdown"]
    axis_tradeoff.scatter(
        selected["coverage_volume_normalized"], selected["pod_normalized"],
        marker="*", s=340, color="#ffd92f", edgecolor="black", linewidth=1.2,
        label="Selected equilibrium evaluation", zorder=10,
    )
    for point, position in zip(
        np.column_stack((coverage, pod_component)), sensor_xy,
    ):
        axis_tradeoff.annotate(
            f"({position[0]:.0f},{position[1]:.0f})", point,
            xytext=(4, 4), textcoords="offset points", fontsize=6.6, alpha=0.72,
        )
    axis_tradeoff.set(
        xlabel="Normalized LOS coverage volume",
        ylabel="Defender PoD component",
        title="C  Defender objective components",
    )
    axis_tradeoff.grid(alpha=0.2)
    axis_tradeoff.legend(loc="best")
    figure.colorbar(tradeoff, ax=axis_tradeoff, label="Defender objective")
    figure.suptitle(
        f"3D Stackelberg - {stage_label}\n"
        r"leader: $(x_s,y_s)$; follower: physical Bellman Attacker response"
        f"; $w_{{PoD}}={pipeline['configuration']['cost']['defender']['w_pod']:.1f}$, "
        f"$w_{{cov}}={pipeline['configuration']['cost']['defender']['w_coverage']:.1f}$",
        fontsize=15, fontweight="bold",
    )
    return figure


def main() -> None:
    configuration = build_configuration()
    result = solve_stackelberg_game(configuration, progress_callback=_progress)
    if not result["status"]["success"]:
        raise RuntimeError(result["status"]["message"])
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    serializable = _serializable_result(result)
    SUMMARY_PATH.write_text(
        json.dumps(serializable, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pipeline = result["final_evaluation"]["pipeline"]
    geometry = pipeline["geometry"]
    trajectory = pipeline["trajectory"]
    replay = pipeline["continuous_replay"]
    np.savez_compressed(
        NPZ_PATH,
        optimal_sensor_position=np.asarray(result["optimal_sensor_position_m"]),
        terrain_x=geometry["x_grid"], terrain_y=geometry["y_grid"],
        terrain_height=geometry["terrain_height"],
        los_boundary_height=geometry["los_boundary_height"],
        powered_path=trajectory["powered_path"],
        glide_trajectory=trajectory["glide_trajectory"],
        continuous_replay_trajectory=replay["trajectory"],
        duration_profile_s=trajectory["duration_profile_s"],
        speed_profile_mps=trajectory["speed_profile_mps"],
        gamma_profile_rad=trajectory["gamma_profile_rad"],
        heading_profile_rad=trajectory["heading_profile_rad"],
        hazard_profile=trajectory["hazard_profile"],
        powered_time_s=np.asarray(trajectory["mission"]["powered_time_s"]),
        switching_point=trajectory["switching_point"],
        goal_position=geometry["goal_position"],
    )
    figure = _create_figure(result)
    figure.savefig(PNG_PATH, dpi=230, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(SUMMARY_PATH, flush=True)
    print(NPZ_PATH, flush=True)
    print(PNG_PATH, flush=True)


if __name__ == "__main__":
    main()
