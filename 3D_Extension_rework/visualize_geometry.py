"""Generate the Stage 1/2 geometry visualization and exported arrays."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .configuration import build_configuration
from .geometry import build_geometry


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_1_geometry"
FIGURE_DIR = ROOT / "figures"
PNG_PATH = FIGURE_DIR / "stage_1_los_geometry.png"
PDF_PATH = FIGURE_DIR / "stage_1_los_geometry.pdf"


def _zone_points(geometry: dict, stride: tuple[int, int, int]):
    xs = geometry["x_grid"][::stride[0]]
    ys = geometry["y_grid"][::stride[1]]
    hs = geometry["h_grid"][::stride[2]]
    mesh_x, mesh_y, mesh_h = np.meshgrid(xs, ys, hs, indexing="ij")
    los = geometry["los_mask"][::stride[0], ::stride[1], ::stride[2]]
    occluded_air = geometry["non_visible_airspace_mask"][
        ::stride[0], ::stride[1], ::stride[2]
    ]
    return (
        np.column_stack((mesh_x[los], mesh_y[los], mesh_h[los])),
        np.column_stack((
            mesh_x[occluded_air], mesh_y[occluded_air], mesh_h[occluded_air],
        )),
    )


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    if not geometry["validation"]["passed"]:
        raise RuntimeError(f"Geometry validation failed: {geometry['validation']}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    tangent = geometry["tangent_manifold"]
    np.savez_compressed(
        RESULT_DIR / "geometry_data.npz",
        x_grid=geometry["x_grid"],
        y_grid=geometry["y_grid"],
        h_grid=geometry["h_grid"],
        terrain_height=geometry["terrain_height"],
        sensor_position=geometry["sensor_position"],
        launch_position=geometry["launch_position"],
        goal_position=geometry["goal_position"],
        los_boundary_height=geometry["los_boundary_height"],
        terrain_mask=geometry["terrain_mask"],
        los_mask=geometry["los_mask"],
        occlusion_mask=geometry["occlusion_mask"],
        non_visible_airspace_mask=geometry["non_visible_airspace_mask"],
        tangent_contact_points=tangent["contact_points"],
        tangent_residuals=tangent["tangent_residuals"],
    )
    summary = {
        "scenario": "single Gaussian hill Stage 1 geometry",
        "environment": configuration["environment"],
        "grid": configuration["grid"],
        "sensor_position_xyz_m": geometry["sensor_position"].tolist(),
        "goal_position_xyz_m": geometry["goal_position"].tolist(),
        "sampled_terrain_peak_m": float(np.max(geometry["terrain_height"])),
        "tangent_contact_count": geometry["validation"]["tangent_contact_count"],
        "maximum_tangent_residual": geometry["validation"][
            "maximum_tangent_residual"
        ],
        "normalized_los_volume": geometry["coverage"]["normalized_los_volume"],
        "validation": geometry["validation"],
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    x_grid = geometry["x_grid"]
    y_grid = geometry["y_grid"]
    terrain = geometry["terrain_height"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    sensor = geometry["sensor_position"]
    goal = geometry["goal_position"]
    contacts = tangent["contact_points"]
    los_points, occlusion_points = _zone_points(geometry, (5, 4, 5))

    figure = plt.figure(figsize=(16.5, 9.2), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=(1.75, 1.0))
    axis_3d = figure.add_subplot(layout[:, 0], projection="3d", computed_zorder=False)
    axis_top = figure.add_subplot(layout[0, 1])
    axis_cross = figure.add_subplot(layout[1, 1])

    axis_3d.scatter(
        los_points[:, 0], los_points[:, 1], los_points[:, 2],
        s=5.0, color="#4aa8d8", alpha=0.035, depthshade=False, zorder=1,
    )
    axis_3d.scatter(
        occlusion_points[:, 0], occlusion_points[:, 1], occlusion_points[:, 2],
        s=7.0, color="#ef8a62", alpha=0.075, depthshade=False, zorder=2,
    )
    axis_3d.plot_surface(
        mesh_x, mesh_y, terrain, cmap="terrain", alpha=0.96,
        linewidth=0.15, edgecolor=(0.15, 0.15, 0.15, 0.16), zorder=4,
    )
    clipped_boundary = np.clip(
        geometry["los_boundary_height"], 0.0, 400.0,
    )
    boundary_visible = np.where(
        clipped_boundary > terrain + 0.5, clipped_boundary, np.nan,
    )
    axis_3d.plot_surface(
        mesh_x, mesh_y, boundary_visible, color="#56b4e9", alpha=0.16,
        linewidth=0.0, antialiased=True, zorder=3,
    )
    axis_3d.plot(
        contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.0,
        color="#cc00cc", linewidth=4.5, label="LOS tangent manifold", zorder=10,
    )
    ray_indices = np.linspace(0, contacts.shape[0] - 1, 11).astype(int)
    for index in ray_indices:
        contact = contacts[index]
        axis_3d.plot(
            [sensor[0], contact[0]], [sensor[1], contact[1]],
            [sensor[2], contact[2]], color="#8e44ad", linestyle="--",
            linewidth=1.0, alpha=0.72, zorder=8,
        )
    axis_3d.scatter(*sensor, s=150, marker="^", color="#d7191c",
                    edgecolor="black", zorder=12)
    axis_3d.scatter(*goal, s=140, marker="X", color="#1a9850",
                    edgecolor="black", zorder=12)
    axis_3d.set(
        xlim=(0, 3000), ylim=(0, 1000), zlim=(0, 400),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  3D LOS and terrain-occlusion geometry",
    )
    axis_3d.view_init(elev=28.0, azim=-64.0)
    axis_3d.set_box_aspect((3.0, 1.0, 0.75))
    legend_handles = [
        Patch(facecolor="#4aa8d8", alpha=0.22, label="LOS zone"),
        Patch(facecolor="#ef8a62", alpha=0.28, label="Occlusion zone"),
        Patch(facecolor="#8c9a63", alpha=0.9, label="Terrain"),
        Line2D([0], [0], color="#cc00cc", linewidth=4,
               label="LOS tangent manifold"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d7191c",
               markeredgecolor="black", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#1a9850",
               markeredgecolor="black", markersize=9, label="Goal"),
    ]
    axis_3d.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    top_levels = np.linspace(0.0, 200.0, 17)
    contour = axis_top.contourf(
        mesh_x, mesh_y, terrain, levels=top_levels, cmap="terrain",
    )
    axis_top.plot(contacts[:, 0], contacts[:, 1], color="#cc00cc", linewidth=3)
    for index in ray_indices:
        axis_top.plot(
            [sensor[0], contacts[index, 0]],
            [sensor[1], contacts[index, 1]],
            color="#8e44ad", linestyle="--", linewidth=0.8, alpha=0.7,
        )
    axis_top.scatter(sensor[0], sensor[1], s=90, marker="^", color="#d7191c",
                     edgecolor="black", zorder=5)
    axis_top.scatter(goal[0], goal[1], s=90, marker="X", color="#1a9850",
                     edgecolor="black", zorder=5)
    axis_top.set(
        xlim=(0, 3000), ylim=(0, 1000), xlabel="x [m]", ylabel="y [m]",
        title="B  Plan view of tangent contacts and rays",
    )
    axis_top.set_aspect("equal", adjustable="box")
    figure.colorbar(contour, ax=axis_top, label="Terrain height [m]", shrink=0.86)

    center_index = int(np.argmin(np.abs(y_grid - 500.0)))
    center_terrain = terrain[:, center_index]
    center_boundary = geometry["los_boundary_height"][:, center_index]
    occlusion_top = np.minimum(center_boundary, 400.0)
    axis_cross.fill_between(
        x_grid, center_terrain, occlusion_top,
        where=occlusion_top > center_terrain,
        color="#ef8a62", alpha=0.28, label="Occlusion zone",
    )
    axis_cross.fill_between(
        x_grid, np.maximum(center_boundary, center_terrain), 400.0,
        where=np.maximum(center_boundary, center_terrain) < 400.0,
        color="#4aa8d8", alpha=0.20, label="LOS zone",
    )
    axis_cross.fill_between(
        x_grid, 0.0, center_terrain, color="#8c6d31", alpha=0.48,
        label="Terrain",
    )
    axis_cross.plot(x_grid, center_terrain, color="#5b461f", linewidth=1.8)
    axis_cross.plot(
        x_grid, center_boundary, color="#2475a7", linewidth=2.0,
        label=r"$H_{LOS}(x,y=500)$",
    )
    center_contacts = contacts[np.argsort(np.abs(contacts[:, 1] - 500.0))[:2]]
    axis_cross.scatter(
        center_contacts[:, 0], center_contacts[:, 2], s=70,
        color="#cc00cc", edgecolor="black", zorder=6,
        label="Tangent contact",
    )
    axis_cross.scatter(sensor[0], sensor[2], s=80, marker="^", color="#d7191c",
                       edgecolor="black", zorder=6)
    axis_cross.scatter(goal[0], goal[2], s=80, marker="X", color="#1a9850",
                       edgecolor="black", zorder=6)
    axis_cross.set(
        xlim=(0, 3000), ylim=(0, 400), xlabel="x [m]", ylabel="h [m]",
        title="C  Centerline section at y = 500 m",
    )
    axis_cross.grid(alpha=0.25)
    axis_cross.legend(loc="upper left", fontsize=8.5, ncol=2)

    figure.suptitle(
        "Clean 3D Extension - Single-Hill Terrain, LOS, Occlusion, and Tangent Manifold\n"
        "hill=(1500, 500, 200) m, sensor=(2500, 500) m, "
        "goal=(3000, 500) m, ceiling=400 m",
        fontsize=14.5, fontweight="bold",
    )
    figure.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
