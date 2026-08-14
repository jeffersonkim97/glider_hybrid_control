"""Create the presentation-only standalone 3D Bellman cost-to-go heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
GEOMETRY_PATH = ROOT / "results" / "stage_1_geometry" / "geometry_data.npz"
COST_PATH = ROOT / "results" / "stage_4_cost_to_go" / "projected_cost_to_go_3d.npz"
PNG_PATH = ROOT / "figures" / "stage_4_bellman_cost_to_go_heatmap_3d.png"
PDF_PATH = ROOT / "figures" / "stage_4_bellman_cost_to_go_heatmap_3d.pdf"


def main() -> None:
    if not GEOMETRY_PATH.exists() or not COST_PATH.exists():
        raise FileNotFoundError(
            "Run visualize_geometry and visualize_cost_to_go before this script"
        )
    geometry = np.load(GEOMETRY_PATH)
    result = np.load(COST_PATH)
    x_grid = result["x_grid"]
    y_grid = result["y_grid"]
    h_grid = result["h_grid"]
    cost = result["projected_cost_to_go"]
    support = result["projection_mask"].astype(bool)
    mesh_x, mesh_y, mesh_h = np.meshgrid(
        x_grid, y_grid, h_grid, indexing="ij",
    )
    positive = support & np.isfinite(cost) & (cost > 0.0)
    finite_cost = cost[positive]
    norm = LogNorm(vmin=float(np.min(finite_cost)), vmax=float(np.max(finite_cost)))
    cmap = plt.get_cmap("viridis")

    figure = plt.figure(figsize=(15.8, 8.8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)

    terrain_x, terrain_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    axis.plot_surface(
        terrain_x, terrain_y, geometry["terrain_height"],
        color="#a98b5b", alpha=0.66, linewidth=0.15,
        edgecolor=(0.18, 0.14, 0.08, 0.13), zorder=2,
    )

    # Draw high-altitude points first so low/near points remain legible.
    points = np.column_stack((
        mesh_x[positive], mesh_y[positive], mesh_h[positive], cost[positive],
    ))
    order = np.lexsort((-points[:, 0], -points[:, 2]))
    points = points[order]
    axis.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 3], cmap=cmap, norm=norm,
        s=18.0, alpha=0.78, linewidths=0.0,
        depthshade=False, zorder=4,
    )

    contacts = geometry["tangent_contact_points"]
    axis.plot(
        contacts[:, 0], contacts[:, 1], contacts[:, 2] + 2.0,
        color="#d000d0", linewidth=4.0, zorder=8,
    )
    sensor = geometry["sensor_position"]
    goal = geometry["goal_position"]
    axis.scatter(
        *sensor, marker="^", s=155, color="#d7191c",
        edgecolor="black", linewidth=1.1, zorder=10,
    )
    axis.scatter(
        *goal, marker="X", s=150, color="#1a9850",
        edgecolor="black", linewidth=1.1, zorder=10,
    )

    axis.set(
        xlim=(0.0, 3000.0), ylim=(0.0, 1000.0), zlim=(0.0, 400.0),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
    )
    axis.set_box_aspect((3.0, 1.0, 0.78))
    axis.view_init(elev=27.0, azim=-61.0)
    axis.grid(alpha=0.23)
    axis.set_title(
        r"Projected Bellman Cost-to-Go $V^*(x,y,h)$",
        fontsize=18.0, fontweight="bold", pad=18,
    )
    figure.text(
        0.50, 0.915,
        "Exact physical-successor Bellman solution; logarithmic color scale",
        ha="center", va="center", fontsize=12.2,
    )
    figure.text(
        0.50, 0.035,
        "Empty volume: terrain/LOS infeasible or no feasible Bellman path to the goal",
        ha="center", va="center", fontsize=10.5, color="#444444",
    )

    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=axis,
        shrink=0.74, pad=0.05, aspect=28,
    )
    colorbar.set_label("Bellman cost-to-go (log color)", fontsize=11.5)
    legend = [
        Patch(facecolor="#a98b5b", alpha=0.66, label="Terrain"),
        Line2D([0], [0], color="#d000d0", linewidth=4,
               label="LOS tangent manifold"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d7191c",
               markeredgecolor="black", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#1a9850",
               markeredgecolor="black", markersize=9, label="Goal"),
    ]
    figure.legend(
        handles=legend, loc="upper left", bbox_to_anchor=(0.105, 0.865),
        framealpha=0.96, fontsize=10.5,
    )

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=240, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
