"""Visualize LOS-independent switching-surface multistart convergence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "switching_surface_multistart"
FINE_DIR = REPO_ROOT / "results" / "extreme_ridge_275_fine"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def main() -> None:
    with (RESULT_DIR / "multistart_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with np.load(RESULT_DIR / "normalized_trajectories.npz") as handle:
        curves = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(FINE_DIR / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain = np.asarray(handle["terrain_height"])
        sensor = np.asarray(handle["sensor_position"])
        goal = np.asarray(handle["goal_position"])

    records = summary["records"]
    feasible = [record for record in records if record["dense_validation_passed"]]
    topology_color = {"south": "#2563eb", "center": "#10b981", "north": "#ef4444"}
    source_marker = {
        "legacy_discrete_shadow": "*",
        "physical_switching_surface": "o",
        "continuous_mirrored_topology": "X",
    }

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_switch = fig.add_subplot(grid[0, 1])
    ax_path = fig.add_subplot(grid[1, 0], projection="3d")
    ax_metrics = fig.add_subplot(grid[1, 1])

    contour = ax_top.contourf(
        terrain_x, terrain_y, terrain.T, levels=18, cmap="terrain", alpha=0.82,
    )
    fig.colorbar(contour, ax=ax_top, label="terrain height (m)")
    for record in records:
        initial = np.asarray(record["initial_switch_position_m"])
        color = topology_color[record["topology"]]
        marker = source_marker[record["source"]]
        ax_top.scatter(
            initial[0], initial[1], color=color, marker=marker, s=75,
            edgecolor="black", linewidth=0.5,
        )
        if record["dense_validation_passed"]:
            optimized = np.asarray(record["optimized_switch_position_m"])
            ax_top.annotate(
                "", xy=optimized[:2], xytext=initial[:2],
                arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.45, "lw": 1.2},
            )
    best = summary["best_solution"]
    best_switch = np.asarray(best["optimized_switch_position_m"])
    ax_top.scatter(
        best_switch[0], best_switch[1], marker="P", s=150,
        color="#facc15", edgecolor="black", label="common optimized switch",
    )
    ax_top.scatter(sensor[0], sensor[1], marker="^", s=90, color="black", label="sensor")
    ax_top.scatter(goal[0], goal[1], marker="*", s=150, color="gold", edgecolor="black", label="goal")
    ax_top.set(
        title="A. LOS-independent initial switches → common optimum",
        xlabel="x (m)", ylabel="y (m)", xlim=(0, 2600), ylim=(-850, 850),
    )
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.legend(fontsize=9, loc="lower right")

    for record in records:
        initial = np.asarray(record["initial_switch_position_m"])
        color = topology_color[record["topology"]]
        marker = source_marker[record["source"]]
        ax_switch.scatter(
            initial[0], initial[2], color=color, marker=marker, s=75,
            edgecolor="black", linewidth=0.5,
        )
        if record["dense_validation_passed"]:
            optimized = np.asarray(record["optimized_switch_position_m"])
            ax_switch.plot(
                [initial[0], optimized[0]], [initial[2], optimized[2]],
                color=color, alpha=0.45, lw=1.2,
            )
    ax_switch.scatter(
        best_switch[0], best_switch[2], marker="P", s=150,
        color="#facc15", edgecolor="black",
    )
    ax_switch.axhline(200.0, color="#991b1b", ls="--", lw=1.3, label="airspace ceiling")
    ax_switch.set(
        title="B. Initial switching surface in physical coordinates",
        xlabel="switch x (m)", ylabel="switch altitude h (m)",
    )
    ax_switch.grid(alpha=0.25)
    ax_switch.legend(fontsize=9)

    mesh_x, mesh_y = np.meshgrid(terrain_x, terrain_y, indexing="ij")
    stride_x = max(1, terrain_x.size // 26)
    stride_y = max(1, terrain_y.size // 22)
    ax_path.plot_surface(
        mesh_x[::stride_x, ::stride_y], mesh_y[::stride_x, ::stride_y],
        terrain[::stride_x, ::stride_y], cmap="terrain", alpha=0.42,
        linewidth=0, antialiased=True,
    )
    for record in feasible:
        curve = curves[record["run_id"]]
        ax_path.plot(
            curve[:, 0], curve[:, 1], curve[:, 2],
            color=topology_color[record["topology"]], alpha=0.42, lw=1.5,
        )
    best_curve = curves[best["run_id"]]
    ax_path.plot(
        best_curve[:, 0], best_curve[:, 1], best_curve[:, 2],
        color="#d946ef", lw=3.2, label="selected trajectory",
    )
    ax_path.scatter(*best_switch, color="#facc15", edgecolor="black", s=75, marker="P")
    ax_path.set(
        title="C. Eight validated trajectories overlap",
        xlabel="x (m)", ylabel="y (m)", zlabel="h (m)",
        xlim=(0, 2600), ylim=(-850, 650), zlim=(0, 285),
    )
    ax_path.view_init(elev=24, azim=-63)
    ax_path.legend(fontsize=9)

    labels = [record["run_id"].replace("_", "\n") for record in records]
    objectives = [record.get("physical_objective", np.nan) for record in records]
    pods = [100.0 * record.get("mission_pod", np.nan) for record in records]
    positions = np.arange(len(records))
    colors = [topology_color[record["topology"]] for record in records]
    bars = ax_metrics.bar(positions, objectives, color=colors, alpha=0.78)
    for bar, record in zip(bars, records):
        if not record["dense_validation_passed"]:
            bar.set_hatch("///")
            bar.set_alpha(0.35)
    ax_metrics.set_xticks(positions, labels, rotation=0, fontsize=7)
    ax_metrics.set_ylabel("physical objective")
    ax_metrics.set_ylim(0.585, max(value for value in objectives if np.isfinite(value)) + 0.005)
    ax_metrics.grid(axis="y", alpha=0.25)
    ax_pod = ax_metrics.twinx()
    ax_pod.plot(positions, pods, "ko--", ms=4, lw=1, label="mission PoD")
    ax_pod.set_ylabel("mission PoD (%)")
    ax_metrics.set_title("D. Basin ranking; hatched north solve is invalid")
    ax_metrics.text(
        0.02, 0.97,
        "8/9 dense-valid\n"
        "validated objective spread: 4.1e-12\n"
        "final switch spread: 7.2e-8 m\n"
        "global optimality not claimed",
        transform=ax_metrics.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    fig.suptitle(
        "LOS-Independent Switching-Surface Multistart — Projection Unchanged",
        fontsize=17, fontweight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "switching_surface_multistart.png"
    pdf = OUTPUT_DIR / "switching_surface_multistart.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
