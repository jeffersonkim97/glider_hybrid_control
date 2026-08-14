"""Export and visualize fixed-control diagnostics for the local J6D map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch

from .configuration import build_configuration
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_3_stage_cost"
FIGURE_DIR = ROOT / "figures"
PNG_PATH = FIGURE_DIR / "stage_3_j6d_diagnostics.png"
PDF_PATH = FIGURE_DIR / "stage_3_j6d_diagnostics.pdf"
NPZ_PATH = RESULT_DIR / "stage_cost_6d.npz"


def _masked_log_norm(values: np.ma.MaskedArray) -> LogNorm:
    positive = np.asarray(values.compressed(), dtype=float)
    positive = positive[positive > 0.0]
    return LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    if not stage["validation"]["passed"]:
        raise RuntimeError(stage["validation"]["summary"])

    grids = stage["grids"]
    masks = stage["validity_masks"]
    control_valid = masks["control_valid_mask"][0, 0, 0]
    candidates = np.argwhere(control_valid)
    score = (
        np.abs(grids["v"][candidates[:, 0]] - 18.0) / 12.6
        + np.abs(np.rad2deg(grids["gamma"][candidates[:, 1]]) + 10.0) / 89.0
        + np.abs(np.rad2deg(grids["psi"][candidates[:, 2]])) / 180.0
    )
    velocity_index, gamma_index, psi_index = candidates[int(np.argmin(score))]
    velocity = float(grids["v"][velocity_index])
    gamma = float(grids["gamma"][gamma_index])
    psi = float(grids["psi"][psi_index])

    y_index = int(np.argmin(np.abs(grids["y"] - 500.0)))
    h_index = int(np.argmin(np.abs(grids["h"] - 200.0)))
    fixed_index = (velocity_index, gamma_index, psi_index)
    center_cost = stage["j6d"][:, y_index, :, *fixed_index]
    horizontal_cost = stage["j6d"][:, :, h_index, *fixed_index]
    center_plot = np.ma.masked_invalid(np.where(
        np.isfinite(center_cost), center_cost, np.nan,
    ))
    horizontal_plot = np.ma.masked_invalid(np.where(
        np.isfinite(horizontal_cost), horizontal_cost, np.nan,
    ))

    phase = np.zeros(masks["spatial_glide_valid"].shape, dtype=np.int8)
    phase[masks["spatial_powered_valid"]] = 1
    phase[masks["spatial_glide_valid"]] = 2
    center_phase = phase[:, y_index, :]
    phase_cmap = ListedColormap(["#8c6d31", "#ef8a62", "#56b4e9"])
    phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], phase_cmap.N)

    sample_x_index = int(np.argmin(np.abs(grids["x"] - 2000.0)))
    sample_h_index = int(np.argmin(np.abs(grids["h"] - 200.0)))
    heading_cost = stage["j6d"][
        sample_x_index, y_index, sample_h_index,
        velocity_index, gamma_index, :,
    ]

    figure, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), constrained_layout=True)
    axes[0, 0].pcolormesh(
        grids["x"], grids["h"], center_phase.T,
        shading="nearest", cmap=phase_cmap, norm=phase_norm,
    )
    axes[0, 0].set_title("A  Phase-feasible spatial states at y = 500 m")
    axes[0, 0].legend(handles=[
        Patch(color="#8c6d31", label="Terrain / invalid"),
        Patch(color="#ef8a62", label="Powered feasible (occluded)"),
        Patch(color="#56b4e9", label="Glide feasible (LOS)"),
    ], loc="upper left")

    center_image = axes[0, 1].pcolormesh(
        grids["x"], grids["h"], center_plot.T,
        shading="nearest", cmap="viridis", norm=_masked_log_norm(center_plot),
    )
    axes[0, 1].set_title("B  Fixed-control local J6D center section")
    figure.colorbar(center_image, ax=axes[0, 1], label="One-step local cost")

    horizontal_image = axes[1, 0].pcolormesh(
        grids["x"], grids["y"], horizontal_plot.T,
        shading="nearest", cmap="viridis", norm=_masked_log_norm(horizontal_plot),
    )
    contacts = geometry["tangent_manifold"]["contact_points"]
    axes[1, 0].plot(
        contacts[:, 0], contacts[:, 1], color="#cc00cc", linewidth=2.3,
        label="LOS tangent manifold",
    )
    axes[1, 0].scatter(
        geometry["sensor_position"][0], geometry["sensor_position"][1],
        marker="^", s=75, color="#d7191c", edgecolor="black", label="Sensor",
    )
    axes[1, 0].set_title("C  Fixed-control local J6D at h = 200 m")
    axes[1, 0].legend(loc="upper left")
    figure.colorbar(horizontal_image, ax=axes[1, 0], label="One-step local cost")

    axes[1, 1].plot(
        np.rad2deg(grids["psi"]), heading_cost,
        marker="o", linewidth=2.0, color="#3b528b",
    )
    axes[1, 1].set_title("D  Local J6D heading dependence at (2000,500,200) m")
    axes[1, 1].set(
        xlabel=r"Heading $\psi$ [deg]", ylabel="One-step local cost",
        xlim=(-180.0, 180.0),
    )
    axes[1, 1].grid(alpha=0.25)

    for axis in axes[0, :]:
        axis.set(xlim=(0.0, 3000.0), ylim=(0.0, 400.0),
                 xlabel="x [m]", ylabel="h [m]")
    axes[1, 0].set(xlim=(0.0, 3000.0), ylim=(0.0, 1000.0),
                   xlabel="x [m]", ylabel="y [m]")
    axes[1, 0].set_aspect("equal", adjustable="box")
    figure.suptitle(
        "Authoritative Local J6D Diagnostics - not cost-to-go\n"
        f"fixed v={velocity:.1f} m/s, gamma={np.rad2deg(gamma):.1f} deg, "
        f"psi={np.rad2deg(psi):.1f} deg",
        fontsize=14.0, fontweight="bold",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        j6d=stage["j6d"],
        powered_stage_cost_6d=stage["powered_stage_cost_6d"],
        feasible_mask=stage["feasible_mask"],
        x_grid=grids["x"], y_grid=grids["y"], h_grid=grids["h"],
        v_grid=grids["v"], gamma_grid=grids["gamma"], psi_grid=grids["psi"],
    )
    summary = {
        "axis_order": list(stage["grid_metadata"]["axis_order"]),
        "shape": list(stage["grid_metadata"]["shape"]),
        "state_count": stage["grid_metadata"]["state_count"],
        "fixed_diagnostic_control": {
            "v_mps": velocity,
            "gamma_deg": float(np.rad2deg(gamma)),
            "psi_deg": float(np.rad2deg(psi)),
        },
        "metadata": stage["metadata"],
        "validation": stage["validation"],
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
