"""Render diagnostic slices of the Stage 2 symbolic detection model."""

from __future__ import annotations

import json
from pathlib import Path

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_2_detection"
FIGURE_DIR = ROOT / "figures"
PNG_PATH = FIGURE_DIR / "stage_2_detection_diagnostics.png"
PDF_PATH = FIGURE_DIR / "stage_2_detection_diagnostics.pdf"


def _mapped_outputs(function: ca.Function, *inputs: np.ndarray) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in inputs]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays):
        raise ValueError("all mapped inputs must share one shape")
    mapped = function.map(arrays[0].size)
    values = mapped(*[value.reshape(1, -1) for value in arrays])
    result = values if isinstance(values, tuple) else (values,)
    return [np.asarray(value, dtype=float).reshape(shape) for value in result]


def _log_norm(values: np.ma.MaskedArray) -> LogNorm:
    finite = np.asarray(values.compressed(), dtype=float)
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise ValueError("diagnostic map contains no positive detection rate")
    return LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    bundle = build_symbolic_detection_bundle(configuration, geometry)
    if not bundle["validation"]["passed"]:
        raise RuntimeError(bundle["validation"]["summary"])

    sensor = geometry["sensor_position"]
    functions = bundle["functions"]
    speed = 18.0
    gamma = np.deg2rad(-10.0)
    heading = 0.0
    slice_height = 150.0
    x_grid = geometry["x_grid"]
    y_grid = geometry["y_grid"]
    h_grid = geometry["h_grid"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    horizontal_shape = mesh_x.shape
    horizontal_h = np.full(horizontal_shape, slice_height)
    fixed_speed = np.full(horizontal_shape, speed)
    fixed_gamma = np.full(horizontal_shape, gamma)
    fixed_heading = np.full(horizontal_shape, heading)
    sensor_fields = [np.full(horizontal_shape, value) for value in sensor]

    powered = _mapped_outputs(
        functions["powered_detection_components"],
        mesh_x, mesh_y, horizontal_h, fixed_speed, *sensor_fields,
    )[-1]
    glide = _mapped_outputs(
        functions["glide_detection_components"],
        mesh_x, mesh_y, horizontal_h, fixed_speed, fixed_gamma,
        fixed_heading, *sensor_fields,
    )[-1]
    terrain_mask = slice_height <= geometry["terrain_height"]
    powered_plot = np.ma.masked_where(terrain_mask | (powered <= 0.0), powered)
    glide_plot = np.ma.masked_where(terrain_mask | (glide <= 0.0), glide)

    center_y_index = int(np.argmin(np.abs(y_grid - 500.0)))
    cross_x, cross_h = np.meshgrid(x_grid, h_grid, indexing="ij")
    cross_shape = cross_x.shape
    cross_y = np.full(cross_shape, y_grid[center_y_index])
    cross_glide = _mapped_outputs(
        functions["glide_detection_components"],
        cross_x, cross_y, cross_h,
        np.full(cross_shape, speed), np.full(cross_shape, gamma),
        np.full(cross_shape, heading),
        *[np.full(cross_shape, value) for value in sensor],
    )[-1]
    cross_terrain = geometry["terrain_height"][:, center_y_index]
    cross_mask = cross_h <= cross_terrain[:, None]
    cross_plot = np.ma.masked_where(cross_mask | (cross_glide <= 0.0), cross_glide)

    heading_grid = np.linspace(-np.pi, np.pi, 361)
    sample_point = np.array([2000.0, 500.0, 150.0])
    heading_shape = heading_grid.shape
    heading_outputs = _mapped_outputs(
        functions["glide_detection_components"],
        *[np.full(heading_shape, value) for value in sample_point],
        np.full(heading_shape, speed), np.full(heading_shape, gamma), heading_grid,
        *[np.full(heading_shape, value) for value in sensor],
    )

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), constrained_layout=True)
    powered_image = axes[0, 0].pcolormesh(
        mesh_x, mesh_y, powered_plot, shading="auto", cmap="magma",
        norm=_log_norm(powered_plot),
    )
    axes[0, 0].contour(
        mesh_x, mesh_y, geometry["terrain_height"], levels=[slice_height],
        colors="white", linewidths=1.5,
    )
    axes[0, 0].scatter(sensor[0], sensor[1], marker="^", s=75,
                       color="#00ffff", edgecolor="black")
    axes[0, 0].set_title("A  Powered acoustic rate at h = 150 m")
    figure.colorbar(powered_image, ax=axes[0, 0], label="Detection rate [1/s]")

    glide_image = axes[0, 1].pcolormesh(
        mesh_x, mesh_y, glide_plot, shading="auto", cmap="viridis",
        norm=_log_norm(glide_plot),
    )
    axes[0, 1].contour(
        mesh_x, mesh_y,
        geometry["los_boundary_height"] <= slice_height,
        levels=[0.5], colors="#ff4d4d", linewidths=2.0,
    )
    axes[0, 1].scatter(sensor[0], sensor[1], marker="^", s=75,
                       color="#00ffff", edgecolor="black")
    axes[0, 1].set_title("B  Glide radar + Doppler rate; red = LOS boundary")
    figure.colorbar(glide_image, ax=axes[0, 1], label="Detection rate [1/s]")

    cross_image = axes[1, 0].pcolormesh(
        cross_x, cross_h, cross_plot, shading="auto", cmap="viridis",
        norm=_log_norm(cross_plot),
    )
    axes[1, 0].fill_between(
        x_grid, 0.0, cross_terrain, color="#8c6d31", alpha=0.72,
        label="Terrain",
    )
    axes[1, 0].plot(
        x_grid, np.minimum(geometry["los_boundary_height"][:, center_y_index], 400.0),
        color="#ff4d4d", linewidth=2.0, label=r"$H_{LOS}(x,500)$",
    )
    axes[1, 0].scatter(sensor[0], sensor[2], marker="^", s=75,
                       color="#00ffff", edgecolor="black")
    axes[1, 0].set_title("C  Glide detection center section at y = 500 m")
    axes[1, 0].legend(loc="upper left")
    figure.colorbar(cross_image, ax=axes[1, 0], label="Detection rate [1/s]")

    axes[1, 1].semilogy(
        np.rad2deg(heading_grid), heading_outputs[8],
        label="Radar rate", linewidth=2.0,
    )
    axes[1, 1].semilogy(
        np.rad2deg(heading_grid), heading_outputs[9],
        label="Doppler rate", linewidth=2.0,
    )
    axes[1, 1].semilogy(
        np.rad2deg(heading_grid), heading_outputs[10],
        label="Total glide rate", color="black", linewidth=2.3,
    )
    axes[1, 1].set_title("D  Heading dependence at (2000, 500, 150) m")
    axes[1, 1].set_xlabel(r"Heading $\psi$ [deg]")
    axes[1, 1].set_ylabel("Detection rate [1/s]")
    axes[1, 1].set_xlim(-180.0, 180.0)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    for axis in axes[0, :]:
        axis.set(xlim=(0.0, 3000.0), ylim=(0.0, 1000.0),
                 xlabel="x [m]", ylabel="y [m]")
        axis.set_aspect("equal", adjustable="box")
    axes[1, 0].set(xlim=(0.0, 3000.0), ylim=(0.0, 400.0),
                   xlabel="x [m]", ylabel="h [m]")
    figure.suptitle(
        r"3D Detection Diagnostics: fixed $v=18$ m/s, $\gamma=-10^\circ$, "
        r"$\psi=0^\circ$ (not an optimized policy)",
        fontsize=14.0, fontweight="bold",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "state_axis_order": list(bundle["metadata"]["state_axis_order"]),
        "fixed_state": {
            "speed_mps": speed,
            "gamma_deg": float(np.rad2deg(gamma)),
            "heading_deg": float(np.rad2deg(heading)),
            "horizontal_slice_height_m": slice_height,
        },
        "maximum_powered_detection_rate_per_s": float(np.max(powered_plot)),
        "maximum_glide_detection_rate_per_s": float(np.max(glide_plot)),
        "validation": bundle["validation"],
    }
    with (RESULT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    figure.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)
    print(PNG_PATH)
    print(PDF_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
