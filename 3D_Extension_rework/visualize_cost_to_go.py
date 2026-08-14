"""Render and export the projected Bellman cost-to-go heatmap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .bellman import build_cost_to_go_bundle
from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_4_cost_to_go"
FIGURE_DIR = ROOT / "figures"
PNG_PATH = FIGURE_DIR / "stage_4_projected_cost_to_go.png"
PDF_PATH = FIGURE_DIR / "stage_4_projected_cost_to_go.pdf"
NPZ_PATH = RESULT_DIR / "projected_cost_to_go_3d.npz"


def _positive_log_norm(values: np.ndarray) -> LogNorm:
    positive = np.asarray(values)[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("map has no positive finite values")
    return LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    bundle = build_cost_to_go_bundle(
        configuration, geometry, detection, stage,
    )
    if not bundle["status"]["success"]:
        raise RuntimeError(bundle["status"]["message"])

    graph = bundle["graph"]
    policy = bundle["policy"]
    projection = bundle["projection"]
    grids = graph["grids"]
    cost = projection["projected_cost_to_go"]
    pod = projection["projected_pod_to_go"]
    support = projection["projection_mask"]
    cost_norm = _positive_log_norm(cost)
    pod_norm = _positive_log_norm(pod)
    center_y_index = int(np.argmin(np.abs(grids["y"] - 500.0)))
    finite_per_height = np.count_nonzero(support, axis=(0, 1))
    horizontal_h_index = int(np.argmax(finite_per_height))

    mesh_x, mesh_y, mesh_h = np.meshgrid(
        grids["x"], grids["y"], grids["h"], indexing="ij",
    )
    scatter_mask = support & np.isfinite(cost) & (cost > 0.0)
    center_cost = np.ma.masked_where(
        ~support[:, center_y_index, :] | (cost[:, center_y_index, :] <= 0.0),
        cost[:, center_y_index, :],
    )
    horizontal_cost = np.ma.masked_where(
        ~support[:, :, horizontal_h_index] | (cost[:, :, horizontal_h_index] <= 0.0),
        cost[:, :, horizontal_h_index],
    )
    center_pod = np.ma.masked_where(
        ~support[:, center_y_index, :] | (pod[:, center_y_index, :] <= 0.0),
        pod[:, center_y_index, :],
    )

    figure = plt.figure(figsize=(16.2, 9.5), constrained_layout=True)
    layout = figure.add_gridspec(2, 2)
    axis_3d = figure.add_subplot(layout[0, 0], projection="3d")
    axis_center = figure.add_subplot(layout[0, 1])
    axis_horizontal = figure.add_subplot(layout[1, 0])
    axis_pod = figure.add_subplot(layout[1, 1])

    scatter = axis_3d.scatter(
        mesh_x[scatter_mask], mesh_y[scatter_mask], mesh_h[scatter_mask],
        c=cost[scatter_mask], cmap="viridis", norm=cost_norm,
        s=11.0, alpha=0.72, depthshade=False,
    )
    terrain_x, terrain_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    axis_3d.plot_surface(
        terrain_x, terrain_y, geometry["terrain_height"],
        cmap="terrain", alpha=0.48, linewidth=0.0,
    )
    contacts = geometry["tangent_manifold"]["contact_points"]
    axis_3d.plot(
        contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.0,
        color="#cc00cc", linewidth=3.0,
    )
    axis_3d.scatter(
        *geometry["sensor_position"], marker="^", s=85,
        color="#d7191c", edgecolor="black",
    )
    axis_3d.scatter(
        *geometry["goal_position"], marker="X", s=85,
        color="#1a9850", edgecolor="black",
    )
    axis_3d.set(
        xlim=(0.0, 3000.0), ylim=(0.0, 1000.0), zlim=(0.0, 400.0),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  Finite projected Bellman cost-to-go support",
    )
    axis_3d.set_box_aspect((3.0, 1.0, 0.75))
    axis_3d.view_init(elev=27.0, azim=-63.0)
    figure.colorbar(scatter, ax=axis_3d, label="Cost-to-go (log color)", shrink=0.76)

    center_image = axis_center.pcolormesh(
        grids["x"], grids["h"], center_cost.T,
        shading="nearest", cmap="viridis", norm=cost_norm,
    )
    center_terrain = np.interp(
        grids["x"], geometry["x_grid"],
        geometry["terrain_height"][:, int(np.argmin(np.abs(geometry["y_grid"] - 500.0)))],
    )
    center_boundary = np.interp(
        grids["x"], geometry["x_grid"],
        geometry["los_boundary_height"][:, int(np.argmin(np.abs(geometry["y_grid"] - 500.0)))],
    )
    axis_center.fill_between(
        grids["x"], 0.0, center_terrain, color="#8c6d31", alpha=0.72,
        label="Terrain",
    )
    axis_center.plot(
        grids["x"], np.minimum(center_boundary, 400.0),
        color="#ff4d4d", linewidth=1.8, label=r"$H_{LOS}$",
    )
    axis_center.set_title("B  Projected cost-to-go at y = 500 m")
    axis_center.legend(loc="upper left")
    figure.colorbar(center_image, ax=axis_center, label="Cost-to-go (log color)")

    horizontal_image = axis_horizontal.pcolormesh(
        grids["x"], grids["y"], horizontal_cost.T,
        shading="nearest", cmap="viridis", norm=cost_norm,
    )
    axis_horizontal.plot(
        contacts[:, 0], contacts[:, 1], color="#cc00cc", linewidth=2.2,
        label="LOS tangent manifold",
    )
    axis_horizontal.scatter(
        geometry["sensor_position"][0], geometry["sensor_position"][1],
        marker="^", s=65, color="#d7191c", edgecolor="black", label="Sensor",
    )
    axis_horizontal.scatter(
        geometry["goal_position"][0], geometry["goal_position"][1],
        marker="X", s=65, color="#1a9850", edgecolor="black", label="Goal",
    )
    axis_horizontal.set_title(
        f"C  Horizontal cost-to-go slice at h = {grids['h'][horizontal_h_index]:.0f} m"
    )
    axis_horizontal.legend(loc="upper left")
    figure.colorbar(horizontal_image, ax=axis_horizontal,
                    label="Cost-to-go (log color)")

    pod_image = axis_pod.pcolormesh(
        grids["x"], grids["h"], center_pod.T,
        shading="nearest", cmap="magma", norm=pod_norm,
    )
    axis_pod.fill_between(
        grids["x"], 0.0, center_terrain, color="#8c6d31", alpha=0.72,
    )
    axis_pod.plot(
        grids["x"], np.minimum(center_boundary, 400.0),
        color="#56b4e9", linewidth=1.8,
    )
    axis_pod.set_title("D  PoD-to-go along the cost-minimizing Bellman policy")
    figure.colorbar(pod_image, ax=axis_pod, label="PoD-to-go (log color)")

    for axis in (axis_center, axis_pod):
        axis.set(xlim=(0.0, 3000.0), ylim=(0.0, 400.0),
                 xlabel="x [m]", ylabel="h [m]")
    axis_horizontal.set(xlim=(0.0, 3000.0), ylim=(0.0, 1000.0),
                        xlabel="x [m]", ylabel="y [m]")
    axis_horizontal.set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Projected 3D Bellman Cost-to-Go - Exact Physical Successor Grid\n"
        "J6D objective contract -> V4D(x,y,h,psi_in) -> min over psi_in",
        fontsize=14.0, fontweight="bold",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    action_table = np.asarray([
        [
            action["forward_cells"], action["lateral_cells"],
            action["descent_cells"], action["speed_mps"],
            action["gamma_rad"], action["heading_rad"], action["duration_s"],
        ]
        for action in graph["actions"]
    ], dtype=float)
    np.savez_compressed(
        NPZ_PATH,
        projected_cost_to_go=cost,
        projected_pod_to_go=pod,
        optimal_incoming_heading=projection["optimal_incoming_heading"],
        projection_mask=support,
        value_heading_state=policy["value_heading_state"],
        pod_heading_state=policy["pod_to_go_heading_state"],
        policy_action_index=policy["policy_action_index"],
        heading_states=policy["heading_states"],
        action_table=action_table,
        x_grid=grids["x"], y_grid=grids["y"], h_grid=grids["h"],
    )
    finite_positive_cost = cost[np.isfinite(cost) & (cost > 0.0)]
    finite_pod = pod[np.isfinite(pod)]
    summary = {
        "process_chain": list(bundle["metadata"]["process_chain"]),
        "graph_metadata": graph["metadata"],
        "graph_validation": graph["validation"],
        "policy_metadata": policy["metadata"],
        "policy_validation": policy["validation"],
        "projection_metadata": projection["metadata"],
        "projection_validation": projection["validation"],
        "finite_projected_cell_count": int(np.count_nonzero(support)),
        "positive_cost_to_go_range": [
            float(np.min(finite_positive_cost)), float(np.max(finite_positive_cost)),
        ],
        "pod_to_go_range": [float(np.min(finite_pod)), float(np.max(finite_pod))],
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    figure.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)
    print(NPZ_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
