"""Visual summary of physical-action-preserving grid convergence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "physical_action_grid_convergence"
FINE_DIR = REPO_ROOT / "results" / "extreme_ridge_275_fine"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def main() -> None:
    with (RESULT_DIR / "convergence_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with np.load(RESULT_DIR / "normalized_trajectory_curves.npz") as handle:
        curves = {name: np.asarray(handle[name]) for name in handle.files}
    with np.load(FINE_DIR / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain = np.asarray(handle["terrain_height"])
        sensor = np.asarray(handle["sensor_position"])
        goal = np.asarray(handle["goal_position"])

    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_actions = fig.add_subplot(grid[0, 0])
    ax_paths = fig.add_subplot(grid[0, 1])
    ax_altitude = fig.add_subplot(grid[1, 0])
    ax_metrics = fig.add_subplot(grid[1, 1])

    colors = {"coarse": "#6b7280", "medium": "#f59e0b", "fine": "#2563eb"}
    for name in ("coarse", "medium", "fine"):
        record = summary["discrete_records"].get(name)
        if record is None:
            # Coarse failed before a trajectory existed; reconstruct its
            # sampled offsets from the known shared envelope/grid spacing.
            from .experiment_physical_action_grid_convergence import (
                _action_domain_record,
                _configuration,
            )
            from .phase_logging import close_phase_logger
            configuration = _configuration(name)
            try:
                domain = _action_domain_record(configuration)
            finally:
                close_phase_logger(
                    configuration["primary_result"]["logging_utilities"]["logger"]
                )
        else:
            domain = record["action_domain"]
        spacing = domain["grid_spacing_m"]
        offsets = np.asarray(domain["offsets"])
        forward = offsets[:, 0] * spacing["dx"]
        lateral = offsets[:, 1] * spacing["dy"]
        unique_xy = np.unique(np.column_stack((forward, lateral)), axis=0)
        ax_actions.scatter(
            unique_xy[:, 0], unique_xy[:, 1], s=34, alpha=0.75,
            color=colors[name], label=f"{name}: {len(offsets)} 3-D offsets",
        )
    envelope = summary["discrete_records"]["fine"]["action_domain"][
        "configured_envelope_m"
    ]
    ax_actions.axvspan(
        envelope["forward_min_m"], envelope["forward_max_m"],
        ymin=0.0, ymax=1.0, color="#10b981", alpha=0.07,
        label="shared physical envelope",
    )
    ax_actions.axhline(envelope["lateral_max_m"], color="#047857", ls="--", lw=1)
    ax_actions.axhline(-envelope["lateral_max_m"], color="#047857", ls="--", lw=1)
    ax_actions.set(
        title="A. Same physical action domain, denser exact-edge samples",
        xlabel="forward displacement per edge (m)",
        ylabel="lateral displacement per edge (m)",
    )
    ax_actions.grid(alpha=0.25)
    ax_actions.legend(fontsize=9, loc="lower left")

    contour = ax_paths.contourf(
        terrain_x, terrain_y, terrain.T, levels=18, cmap="terrain", alpha=0.82,
    )
    fig.colorbar(contour, ax=ax_paths, label="terrain height (m)")
    for name in ("medium", "fine"):
        discrete = curves[f"discrete_{name}"]
        continuous = curves[f"continuous_{name}"]
        ax_paths.plot(
            discrete[:, 0], discrete[:, 1], ls="--", lw=2.2,
            color=colors[name], label=f"{name} discrete",
        )
        ax_paths.plot(
            continuous[:, 0], continuous[:, 1], lw=2.6,
            color=colors[name], alpha=0.9, label=f"{name} → continuous",
        )
    ax_paths.scatter(sensor[0], sensor[1], marker="^", s=90, color="black", label="sensor")
    ax_paths.scatter(goal[0], goal[1], marker="*", s=150, color="gold", edgecolor="black", label="goal")
    ax_paths.set(
        title="B. Discrete paths differ; continuous refinements coincide",
        xlabel="x (m)", ylabel="y (m)", xlim=(0, 2600), ylim=(-850, 650),
    )
    ax_paths.set_aspect("equal", adjustable="box")
    ax_paths.legend(fontsize=8, loc="lower right", ncol=2)

    normalized_time = np.linspace(0.0, 1.0, curves["discrete_fine"].shape[0])
    for name in ("medium", "fine"):
        ax_altitude.plot(
            normalized_time, curves[f"discrete_{name}"][:, 2], ls="--", lw=2.2,
            color=colors[name], label=f"{name} discrete",
        )
        ax_altitude.plot(
            normalized_time, curves[f"continuous_{name}"][:, 2], lw=2.6,
            color=colors[name], label=f"{name} → continuous",
        )
    ax_altitude.axhline(200.0, color="#991b1b", lw=1.2, ls=":", label="ceiling")
    ax_altitude.set(
        title="C. Altitude history",
        xlabel="normalized mission time", ylabel="altitude h (m)",
    )
    ax_altitude.grid(alpha=0.25)
    ax_altitude.legend(fontsize=9, ncol=2)

    labels = ["objective", "PoD", "time", "switch", "trajectory RMS"]
    keys = [
        "objective_relative", "pod_absolute", "time_relative",
        "switch_distance_m", "trajectory_rms_m",
    ]
    discrete_diff = summary["discrete_comparison"]["medium"]["differences_from_fine"]
    continuous_diff = summary["continuous_comparison"]["medium"]["differences_from_fine"]
    thresholds = summary["thresholds"]
    discrete_ratio = [max(discrete_diff[key] / thresholds[key], 1.0e-9) for key in keys]
    continuous_ratio = [max(continuous_diff[key] / thresholds[key], 1.0e-9) for key in keys]
    index = np.arange(len(labels))
    width = 0.36
    ax_metrics.bar(index - width / 2, discrete_ratio, width, color="#f59e0b", label="discrete medium/fine")
    ax_metrics.bar(index + width / 2, continuous_ratio, width, color="#2563eb", label="continuous medium/fine")
    ax_metrics.axhline(1.0, color="#991b1b", ls="--", lw=1.5, label="acceptance limit")
    ax_metrics.set_yscale("log")
    ax_metrics.set_xticks(index, labels, rotation=18, ha="right")
    ax_metrics.set_ylabel("difference / acceptance threshold")
    ax_metrics.set_title("D. Convergence verdict (below 1 passes)")
    ax_metrics.grid(axis="y", which="both", alpha=0.25)
    ax_metrics.legend(fontsize=9)
    ax_metrics.text(
        0.02, 0.97,
        "Coarse discrete: no feasible response\n"
        "Medium discrete: FAIL\n"
        "Continuous refinement: PASS",
        transform=ax_metrics.transAxes, va="top", ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.7"},
    )

    fig.suptitle(
        "Physical-Action-Preserving Grid Convergence — Projection Unchanged",
        fontsize=17, fontweight="bold",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "physical_action_grid_convergence.png"
    pdf = OUTPUT_DIR / "physical_action_grid_convergence.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
