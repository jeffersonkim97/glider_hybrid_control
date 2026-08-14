"""Generate Stage 7 unsnapped continuous-replay diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .bellman import build_cost_to_go_bundle
from .configuration import build_configuration
from .continuous_replay import replay_glide_continuous_3d
from .detection import build_symbolic_detection_bundle
from .geometry import build_geometry
from .stage_cost import construct_stage_cost_6d
from .switching import select_switching_point
from .trajectory import extract_optimal_trajectory


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "stage_7_continuous_replay"
FIGURE_DIR = ROOT / "figures"
NPZ_PATH = RESULT_DIR / "continuous_replay_validation.npz"
PNG_PATH = FIGURE_DIR / "stage_7_continuous_replay_validation.png"
PDF_PATH = FIGURE_DIR / "stage_7_continuous_replay_validation.pdf"


def main() -> None:
    configuration = build_configuration()
    geometry = build_geometry(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    stage = construct_stage_cost_6d(configuration, geometry, detection)
    cost_to_go = build_cost_to_go_bundle(
        configuration, geometry, detection, stage,
    )
    switching = select_switching_point(configuration, geometry, cost_to_go)
    trajectory = extract_optimal_trajectory(
        configuration, geometry, cost_to_go, switching,
    )
    replay = replay_glide_continuous_3d(
        configuration, geometry, detection, trajectory,
    )
    if not replay["status"]["success"]:
        raise RuntimeError(replay["status"]["message"])

    reference = np.asarray(replay["reference_trajectory"])
    continuous = np.asarray(replay["trajectory"])
    diagnostics = replay["step_diagnostics"]
    edge_index = np.arange(len(diagnostics))
    drift = np.asarray([
        item["endpoint_drift_norm_m"] for item in diagnostics
    ])
    terrain_margin = np.asarray([
        item["minimum_terrain_margin_m"] for item in diagnostics
    ])
    los_margin = np.asarray([
        item["minimum_los_margin_m"] for item in diagnostics
    ])
    planned_hazard = np.asarray(trajectory["hazard_profile"])
    replay_hazard = np.asarray(replay["replay_hazard_profile"])
    cumulative_planned = np.concatenate(([0.0], np.cumsum(planned_hazard)))
    cumulative_replay = np.concatenate(([0.0], np.cumsum(replay_hazard)))

    figure = plt.figure(figsize=(16.0, 10.0), constrained_layout=True)
    layout = figure.add_gridspec(3, 2, width_ratios=(1.32, 1.0))
    axis_3d = figure.add_subplot(layout[:, 0], projection="3d", computed_zorder=False)
    axis_drift = figure.add_subplot(layout[0, 1])
    axis_margin = figure.add_subplot(layout[1, 1])
    axis_hazard = figure.add_subplot(layout[2, 1])
    terrain_x, terrain_y = np.meshgrid(
        geometry["x_grid"], geometry["y_grid"], indexing="ij",
    )
    axis_3d.plot_surface(
        terrain_x, terrain_y, geometry["terrain_height"],
        color="#aa9068", alpha=0.72, linewidth=0.12,
        edgecolor=(0.2, 0.15, 0.08, 0.12), zorder=2,
    )
    contacts = geometry["tangent_manifold"]["contact_points"]
    axis_3d.plot(contacts[:, 0], contacts[:, 1], contacts[:, 2] + 1.5,
                 color="#d000d0", linewidth=2.5, zorder=5)
    axis_3d.plot(reference[:, 0], reference[:, 1], reference[:, 2],
                 color="#0072b2", linewidth=6.0, label="Planning reference", zorder=8)
    axis_3d.plot(continuous[:, 0], continuous[:, 1], continuous[:, 2],
                 color="#f28e2b", linewidth=2.8, linestyle="--",
                 label="Continuous replay", zorder=9)
    axis_3d.scatter(reference[1:-1, 0], reference[1:-1, 1], reference[1:-1, 2],
                    s=38, color="white", edgecolor="#0072b2", linewidth=1.2, zorder=10)
    axis_3d.scatter(*geometry["sensor_position"], marker="^", s=120,
                    color="#d7191c", edgecolor="black", zorder=11)
    axis_3d.scatter(*geometry["goal_position"], marker="X", s=125,
                    color="#1a9850", edgecolor="black", zorder=11)
    axis_3d.scatter(*trajectory["switching_point"], marker="*", s=250,
                    color="#ffd92f", edgecolor="black", zorder=12)
    axis_3d.set(
        xlim=(0, 3000), ylim=(0, 1000), zlim=(0, 400),
        xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
        title="A  Planning path vs. unsnapped continuous replay",
    )
    axis_3d.set_box_aspect((3.0, 1.0, 0.78))
    axis_3d.view_init(elev=28.0, azim=-62.0)
    legend = [
        Line2D([0], [0], color="#0072b2", linewidth=5, label="Planning reference"),
        Line2D([0], [0], color="#f28e2b", linewidth=3, linestyle="--", label="Continuous replay"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#0072b2", markersize=8, label="Physical edge nodes"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd92f",
               markeredgecolor="black", markersize=12, label="Switch"),
    ]
    axis_3d.legend(handles=legend, loc="upper left", framealpha=0.95)

    axis_drift.semilogy(edge_index, np.maximum(drift, 1.0e-16),
                        marker="o", color="#7b3294", linewidth=1.8)
    axis_drift.axhline(1.0e-9, color="#555555", linestyle="--", linewidth=1.2,
                       label="Validation tolerance")
    axis_drift.set(
        xlabel="Glide edge index", ylabel="Endpoint drift [m]",
        title="B  State drift without grid reset",
        xticks=edge_index,
    )
    axis_drift.grid(alpha=0.22, which="both")
    axis_drift.legend(loc="upper right", fontsize=8.5)
    axis_drift.yaxis.tick_right()
    axis_drift.yaxis.set_label_position("right")

    axis_margin.plot(edge_index, terrain_margin, marker="o", linewidth=2.0,
                     color="#8c6d31", label="Terrain margin")
    axis_margin.plot(edge_index, los_margin, marker="s", linewidth=2.0,
                     color="#d000d0", label="LOS margin")
    axis_margin.axhline(0.0, color="black", linewidth=1.0)
    axis_margin.set(
        xlabel="Glide edge index", ylabel="Minimum margin [m]",
        title="C  Full-edge terrain and LOS checks", xticks=edge_index,
    )
    axis_margin.grid(alpha=0.22)
    axis_margin.legend(loc="upper right", fontsize=8.5)
    axis_margin.yaxis.tick_right()
    axis_margin.yaxis.set_label_position("right")

    node_index = np.arange(cumulative_planned.size)
    axis_hazard.plot(node_index, cumulative_planned, marker="o", linewidth=3.5,
                     color="#0072b2", label="Planning accumulation")
    axis_hazard.plot(node_index, cumulative_replay, marker="x", linewidth=1.8,
                     linestyle="--", color="#f28e2b", label="Continuous replay")
    axis_hazard.set(
        xlabel="Glide node index", ylabel="Cumulative glide hazard",
        title="D  Detection accumulation replay", xticks=node_index,
    )
    axis_hazard.grid(alpha=0.22)
    axis_hazard.legend(loc="upper left", fontsize=8.5)
    axis_hazard.yaxis.tick_right()
    axis_hazard.yaxis.set_label_position("right")
    figure.suptitle(
        "Stage 7 - 3D Continuous Replay Validation\n"
        "same actions, continuous state propagation, no reset and no optimization",
        fontsize=15, fontweight="bold",
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        reference_trajectory=reference,
        continuous_trajectory=continuous,
        endpoint_drift_m=drift,
        minimum_terrain_margin_m=terrain_margin,
        minimum_los_margin_m=los_margin,
        planned_hazard_profile=planned_hazard,
        replay_hazard_profile=replay_hazard,
    )
    summary = {
        "metadata": replay["metadata"],
        "status": replay["status"],
        "feasible": replay["feasible"],
        "violation": replay["violation"],
        "reached_goal": replay["reached_goal"],
        "continuous_glide_time_s": replay["continuous_glide_time_s"],
        "continuous_glide_hazard": replay["continuous_glide_hazard"],
        "continuous_mission_time_s": replay["continuous_mission_time_s"],
        "continuous_mission_hazard": replay["continuous_mission_hazard"],
        "continuous_mission_pod": replay["continuous_mission_pod"],
        "continuous_mission_cost": replay["continuous_mission_cost"],
        "validation": replay["validation"],
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
