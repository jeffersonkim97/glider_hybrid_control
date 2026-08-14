"""Generate and export Stage 5 tangent-surface switching diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .bellman import build_cost_to_go_bundle
from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d
from .switching import select_switching_point


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_5_switching"
FIGURE_DIR = ROOT / "figures"
NPZ_PATH = RESULT_DIR / "switching_selection.npz"
PNG_PATH = FIGURE_DIR / "stage_5_switching_selection.png"
PDF_PATH = FIGURE_DIR / "stage_5_switching_selection.pdf"


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    cost_to_go = build_cost_to_go_bundle(
        configuration, geometry, detection, stage,
    )
    result = select_switching_point(configuration, geometry, cost_to_go)
    if not result["status"]["success"]:
        raise RuntimeError(result["status"]["message"])

    points = result["switching_surface_points"]
    costs = result["manifold_cost"]
    feasible = np.isfinite(costs) & (costs > 0.0)
    norm = Normalize(vmin=float(np.min(costs[feasible])), vmax=float(np.max(costs[feasible])))
    best = result["best"]
    switch = best["switching_point"]
    target = best["connection"]["target"]
    powered_path = best["powered"]["path"]
    virtual_path = best["connection"]["edge"]["path"]
    contacts = geometry["tangent_manifold"]["contact_points"]

    figure = plt.figure(figsize=(16.0, 8.7), constrained_layout=True)
    layout = figure.add_gridspec(1, 2, width_ratios=(1.10, 1.0))
    axis_3d = figure.add_subplot(layout[0, 0], projection="3d", computed_zorder=False)
    axis_plan = figure.add_subplot(layout[0, 1])
    terrain_x, terrain_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    axis_3d.plot_surface(
        terrain_x, terrain_y, geometry["terrain_height"],
        color="#aa9068", alpha=0.68, linewidth=0.12,
        edgecolor=(0.2, 0.15, 0.08, 0.12), zorder=2,
    )
    axis_3d.scatter(
        points[~feasible, 0], points[~feasible, 1], points[~feasible, 2],
        color="#bdbdbd", s=12, alpha=0.28, linewidths=0, zorder=3,
    )
    cost_scatter = axis_3d.scatter(
        points[feasible, 0], points[feasible, 1], points[feasible, 2],
        c=costs[feasible], cmap="viridis", norm=norm,
        s=31, alpha=0.88, linewidths=0, depthshade=False, zorder=5,
    )
    axis_3d.plot(
        contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.5,
        color="#d000d0", linewidth=3.0, zorder=7,
    )
    axis_3d.plot(
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        color="#f28e2b", linewidth=4.0, zorder=9,
    )
    axis_3d.plot(
        virtual_path[:, 0], virtual_path[:, 1], virtual_path[:, 2],
        color="#0072b2", linewidth=3.5, linestyle="--", zorder=9,
    )
    axis_3d.scatter(*geometry["launch_position"], marker="s", s=95,
                    color="black", edgecolor="white", linewidth=0.8, zorder=11)
    axis_3d.scatter(*geometry["sensor_position"], marker="^", s=120,
                    color="#d7191c", edgecolor="black", zorder=11)
    axis_3d.scatter(*geometry["goal_position"], marker="X", s=120,
                    color="#1a9850", edgecolor="black", zorder=11)
    axis_3d.scatter(*switch, marker="*", s=260, color="#ffd92f",
                    edgecolor="black", linewidth=1.1, zorder=12)
    axis_3d.scatter(*target, marker="o", s=85, facecolor="white",
                    edgecolor="#0072b2", linewidth=2.0, zorder=12)
    axis_3d.set(
        xlim=(0, 3000), ylim=(0, 1000), zlim=(0, 400),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  3D tangent-surface candidates and selected switch",
    )
    axis_3d.set_box_aspect((3.0, 1.0, 0.78))
    axis_3d.view_init(elev=28.0, azim=-62.0)

    terrain_contours = axis_plan.contour(
        terrain_x, terrain_y, geometry["terrain_height"],
        levels=np.arange(20.0, 201.0, 20.0), colors="#8c6d31",
        linewidths=0.65, alpha=0.65,
    )
    axis_plan.clabel(terrain_contours, inline=True, fontsize=7, fmt="%.0f m")
    axis_plan.scatter(
        points[~feasible, 0], points[~feasible, 1],
        color="#c7c7c7", s=20, alpha=0.40,
    )
    axis_plan.scatter(
        points[feasible, 0], points[feasible, 1],
        c=costs[feasible], cmap="viridis", norm=norm,
        s=48, edgecolor=(0, 0, 0, 0.18), linewidth=0.35,
    )
    axis_plan.plot(contacts[:, 0], contacts[:, 1], color="#d000d0", linewidth=2.4)
    axis_plan.plot(powered_path[:, 0], powered_path[:, 1], color="#f28e2b", linewidth=3.0)
    axis_plan.plot(virtual_path[:, 0], virtual_path[:, 1], color="#0072b2",
                   linewidth=2.6, linestyle="--")
    axis_plan.scatter(switch[0], switch[1], marker="*", s=240,
                      color="#ffd92f", edgecolor="black", zorder=10)
    axis_plan.scatter(target[0], target[1], marker="o", s=80,
                      facecolor="white", edgecolor="#0072b2", linewidth=2, zorder=10)
    axis_plan.scatter(*geometry["launch_position"][:2], marker="s", s=75,
                      color="black", zorder=10)
    axis_plan.scatter(*geometry["sensor_position"][:2], marker="^", s=90,
                      color="#d7191c", edgecolor="black", zorder=10)
    axis_plan.scatter(*geometry["goal_position"][:2], marker="X", s=90,
                      color="#1a9850", edgecolor="black", zorder=10)
    axis_plan.set(
        xlim=(0, 3000), ylim=(0, 1000), xlabel="x [m]", ylabel="y [m]",
        title="B  Plan view: exhaustive sampled-manifold objective",
    )
    axis_plan.set_aspect("equal", adjustable="box")
    axis_plan.grid(alpha=0.18)

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="black", markersize=8, label="Launch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d7191c", markeredgecolor="black", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#1a9850", markeredgecolor="black", markersize=9, label="Goal"),
        Line2D([0], [0], color="#d000d0", linewidth=3, label="Terrain tangent-contact curve"),
        Line2D([0], [0], color="#f28e2b", linewidth=3, label="Powered path (occluded)"),
        Line2D([0], [0], color="#0072b2", linewidth=3, linestyle="--", label="Physical virtual glide edge"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd92f", markeredgecolor="black", markersize=13, label="Selected switch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#0072b2", markersize=8, label="Bellman entry node"),
    ]
    axis_plan.legend(
        handles=legend, loc="lower left", bbox_to_anchor=(0.0, 1.16),
        ncol=2, fontsize=8.4, framealpha=0.95, borderaxespad=0.0,
    )
    colorbar = figure.colorbar(cost_scatter, ax=(axis_3d, axis_plan),
                              shrink=0.76, pad=0.02, aspect=28)
    colorbar.set_label("Total mission objective")
    figure.suptitle(
        "Stage 5 - LOS Tangent-Surface Switching Selection\n"
        "powered cost + physical virtual edge + heading-state Bellman cost-to-go",
        fontsize=15, fontweight="bold",
    )
    figure.text(
        0.5, 0.018,
        "Gray: infeasible/unreachable candidate; two mirror-image minima tie exactly; "
        "full glide policy extraction is the next stage",
        ha="center", fontsize=10, color="#444444",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        switching_surface_points=points,
        mission_cost=costs,
        mission_pod=result["manifold_pod"],
        feasible_mask=feasible,
        selected_switching_point=switch,
        bellman_entry_node=target,
        powered_path=powered_path,
        virtual_glide_path=virtual_path,
    )
    summary = {
        "metadata": result["metadata"],
        "validation": result["validation"],
        "selected": {
            "switching_point_m": switch.tolist(),
            "bellman_entry_node_m": target.tolist(),
            "mission_cost": float(best["mission_cost"]),
            "mission_pod": float(best["mission_pod"]),
            "powered_duration_s": float(best["powered"]["duration_s"]),
            "powered_minimum_occlusion_margin_m": float(
                best["powered"]["certificate"]["minimum_los_margin_m"]
            ),
            "virtual_minimum_los_margin_m": float(
                best["connection"]["edge"]["certificate"]["minimum_los_margin_m"]
            ),
        },
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    figure.savefig(PNG_PATH, dpi=230, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)
    print(NPZ_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
