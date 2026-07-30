"""Publication-oriented figures for the Direction-B B3 terrain extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import build_direction_b_configuration
from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    B3_TERRAIN_SPECIFICATIONS,
    build_b3_physical_configuration,
)
from p1b_4D.geometry import (
    build_geometry_bundle,
    los_boundary_height,
    terrain_height,
)
from p1b_4D.phase_logging import close_phase_logger


LEVEL_COLORS = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}
SENSOR_COLORS = {"coverage": "#4C78A8", "stackelberg": "#E45756"}
TERRAIN_LABELS = {
    "single_hill": "Single hill",
    "goal_in_valley": "Goal in valley",
}


def create_b3_figures(
    result_path: Path,
    output_directory: Path,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Create trajectory, consistency, and evaluator-qualification figures."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError("B3 result must have complete status")
    output_directory.mkdir(parents=True, exist_ok=True)
    base = build_configuration_bundle(project_root)
    try:
        geometries: dict[tuple[str, str], dict[str, Any]] = {}
        for terrain_name, specification in B3_TERRAIN_SPECIFICATIONS.items():
            for sensor_name, sensor_z in specification[
                "sensor_candidates"
            ].items():
                physical = build_b3_physical_configuration(
                    base, terrain_name, sensor_z
                )
                configured = build_direction_b_configuration(
                    physical, terrain_name, 2
                )
                geometries[(terrain_name, sensor_name)] = (
                    build_geometry_bundle(configured)
                )
        trajectory_path = (
            output_directory / "b3_multiterrain_trajectories.png"
        )
        consistency_path = (
            output_directory / "b3_multiterrain_consistency.png"
        )
        evaluator_path = (
            output_directory / "b3_common_evaluator_qualification.png"
        )
        _plot_trajectories(result, geometries, trajectory_path)
        _plot_consistency(result, consistency_path)
        _plot_evaluator_qualification(result, evaluator_path)
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )
    return trajectory_path, consistency_path, evaluator_path


def _plot_trajectories(result, geometries, output_path):
    figure, axes = plt.subplots(
        2, 2, figsize=(16.0, 10.2), constrained_layout=True
    )
    for row, terrain_name in enumerate(B3_TERRAIN_SPECIFICATIONS):
        for column, sensor_name in enumerate(("coverage", "stackelberg")):
            _plot_trajectory_panel(
                axes[row, column],
                result,
                geometries[(terrain_name, sensor_name)],
                terrain_name,
                sensor_name,
            )
    figure.suptitle(
        "B3 Multi-Terrain Nested-Discretization Trajectories",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_trajectory_panel(
    axis, result, geometry_bundle, terrain_name, sensor_name
):
    geometry = geometry_bundle["primary_result"]
    specification = B3_TERRAIN_SPECIFICATIONS[terrain_name]
    z = np.linspace(0.0, specification["z_max"], 2400)
    ground = terrain_height(geometry["terrain_model"], z)
    boundary = los_boundary_height(geometry["los_geometry"], z)
    sensor = np.asarray(geometry["sensor_position"])
    goal = np.asarray(geometry["goal_position"])
    before_sensor = z <= sensor[0]

    axis.fill_between(
        z, 0.0, ground, color="#5A5A5A", alpha=0.72, label="Terrain"
    )
    axis.fill_between(
        z[before_sensor],
        ground[before_sensor],
        np.maximum(ground[before_sensor], boundary[before_sensor]),
        color="#B8B8B8",
        alpha=0.22,
        label="LOS-occluded airspace",
    )
    axis.plot(
        z[before_sensor], boundary[before_sensor],
        color="black", linestyle="--", linewidth=1.3,
        label="LOS boundary",
    )
    for level in range(3):
        case = _case(
            result, terrain_name, sensor_name,
            f"enriched_v5_q9_l{level}",
        )
        path = np.asarray(case["planning"]["trajectory"], dtype=float)
        objective = case["selected_high_fidelity"]["attacker_objective"]
        axis.plot(
            path[:, 0], path[:, 1],
            color=LEVEL_COLORS[level], linewidth=2.05,
            marker="o", markersize=2.1 if level == 2 else 3.0,
            markevery=max(1, path.shape[0] // 18),
            label=f"L{level}: $J_A^{{HF}}$={objective:.4f}",
        )
        switch = np.asarray(case["planning"]["switching_point"])
        axis.scatter(
            switch[0], switch[1], s=48, facecolors="white",
            edgecolors=LEVEL_COLORS[level], linewidths=1.5, zorder=7,
        )
    axis.scatter(
        sensor[0], sensor[1], marker="^", s=115,
        color=SENSOR_COLORS[sensor_name], edgecolors="black",
        linewidths=0.8, zorder=8, label="Sensor",
    )
    axis.scatter(
        goal[0], goal[1], marker="*", s=150,
        color="#F2CF5B", edgecolors="black", linewidths=0.8,
        zorder=8, label="Goal",
    )
    transported = [
        case for case in result["cases"]
        if case["terrain_name"] == terrain_name
        and case["sensor_name"] == sensor_name
        and case["action_family"] == "transported"
    ]
    feasible_transport = sum(
        case["status"] == "feasible" for case in transported
    )
    axis.text(
        0.015, 0.025,
        f"Transported feasible: {feasible_transport}/{len(transported)}",
        transform=axis.transAxes, fontsize=9.3,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "alpha": 0.85},
    )
    axis.set_title(
        f"{TERRAIN_LABELS[terrain_name]} — {sensor_name.title()} sensor\n"
        f"$z_s$={sensor[0]:.3f} m",
        fontsize=12.2,
    )
    axis.set_xlim(0.0, specification["z_max"])
    axis.set_ylim(0.0, specification["h_max"])
    axis.set_xlabel("Downrange $z$ (m)")
    axis.set_ylabel("Altitude $h$ (m)")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper right", fontsize=8.3, ncol=2)


def _plot_consistency(result, output_path):
    figure, axes = plt.subplots(
        2, 2, figsize=(14.5, 9.2), constrained_layout=True
    )
    levels = np.arange(3)
    for column, terrain_name in enumerate(B3_TERRAIN_SPECIFICATIONS):
        objective_axis = axes[0, column]
        for sensor_name in ("coverage", "stackelberg"):
            values = [
                _case(
                    result, terrain_name, sensor_name,
                    f"enriched_v5_q9_l{level}",
                )["selected_high_fidelity"]["attacker_objective"]
                for level in levels
            ]
            objective_axis.plot(
                levels, values, marker="o", linewidth=2.2,
                color=SENSOR_COLORS[sensor_name],
                linestyle="-" if sensor_name == "coverage" else "--",
                label=sensor_name.title(),
            )
            for level, value in zip(levels, values):
                y_offset = 9 if sensor_name == "coverage" else -15
                objective_axis.annotate(
                    f"{value:.4f}", (level, value),
                    xytext=(0, y_offset), textcoords="offset points",
                    ha="center",
                    va="bottom" if sensor_name == "coverage" else "top",
                    fontsize=8.5,
                )
        objective_axis.set_title(
            f"{TERRAIN_LABELS[terrain_name]}: common-HF objective",
            fontsize=12.2,
        )
        objective_axis.set_xticks(levels, ("L0", "L1", "L2"))
        objective_axis.set_ylabel(r"Attacker objective $J_A^{HF}$")
        objective_axis.margins(y=0.10)
        objective_axis.grid(alpha=0.22)
        objective_axis.legend()

        margin_axis = axes[1, column]
        analysis = result["analysis"][terrain_name]
        margins = np.asarray([
            analysis["defender_margin_by_level"][f"l{level}"]
            for level in levels
        ])
        resolution_shift = analysis[
            "defender_resolution_shift_l1_to_l2"
        ]
        colors = ["#54A24B" if value >= 0.0 else "#E45756" for value in margins]
        bars = margin_axis.bar(levels, margins, color=colors, width=0.58)
        margin_axis.axhline(0.0, color="black", linewidth=1.0)
        margin_axis.axhline(
            2.0 * resolution_shift, color="#7A5195",
            linestyle="--", linewidth=1.4, label=r"$+2R_{1\to2}$",
        )
        margin_axis.axhline(
            -2.0 * resolution_shift, color="#7A5195",
            linestyle="--", linewidth=1.4, label=r"$-2R_{1\to2}$",
        )
        for bar, value in zip(bars, margins):
            offset = 4 if value >= 0.0 else -13
            margin_axis.annotate(
                f"{value:+.3e}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va="bottom" if value >= 0.0 else "top",
                fontsize=8.8,
            )
        resolved = analysis["ranking_diagnostically_resolved"]
        margin_axis.set_title(
            f"Defender margin: {'resolved' if resolved else 'UNRESOLVED'}",
            fontsize=12.2,
            color="#2F6B3C" if resolved else "#9C3D10",
        )
        margin_axis.set_xticks(levels, ("L0", "L1", "L2"))
        margin_axis.set_ylabel(
            r"$J_D(s_{stack})-J_D(s_{coverage})$"
        )
        limit = max(
            np.max(np.abs(margins)) * 1.5,
            2.0 * resolution_shift * 1.25,
            1e-5,
        )
        margin_axis.set_ylim(-limit, limit)
        margin_axis.grid(axis="y", alpha=0.22)
        margin_axis.legend(loc="upper right", fontsize=8.5)
    figure.suptitle(
        "B3 Objective and Fixed-Candidate Ranking Diagnostics",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_evaluator_qualification(result, output_path):
    feasible = [case for case in result["cases"] if case["status"] == "feasible"]
    pair_keys = list(feasible[0]["high_fidelity"]["qualifications"])
    lower_counts = np.asarray([int(key.split("_vs_")[0]) for key in pair_keys])
    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for terrain_name in B3_TERRAIN_SPECIFICATIONS:
        terrain_cases = [
            case for case in feasible if case["terrain_name"] == terrain_name
        ]
        maximum_errors = [
            max(
                case["high_fidelity"]["qualifications"][key][
                    "objective_absolute_difference"
                ]
                for case in terrain_cases
            )
            for key in pair_keys
        ]
        if any(error > 0.0 for error in maximum_errors):
            axis.loglog(
                lower_counts, maximum_errors, marker="o", linewidth=2.1,
                label=TERRAIN_LABELS[terrain_name], zorder=3,
            )
        else:
            axis.plot(
                [], [], marker="o", linewidth=2.1,
                label=f"{TERRAIN_LABELS[terrain_name]} (zero)",
            )
            axis.text(
                0.985, 0.04,
                "Goal-in-valley: $|\\Delta J_A|=0$ at every sampled pair",
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=9.2, color="#F58518",
            )
    all_errors = [
        max(
            case["high_fidelity"]["qualifications"][key][
                "objective_absolute_difference"
            ]
            for case in feasible
        )
        for key in pair_keys
    ]
    axis.loglog(
        lower_counts, all_errors, color="black", marker="s",
        markerfacecolor="none", linewidth=1.7, linestyle=":",
        label="Maximum over B3", zorder=2,
    )
    axis.axhline(1e-6, color="#E45756", linestyle=":", linewidth=2.0,
                 label=r"Objective gate $10^{-6}$")
    selected = result["global_common_evaluator_sample_count"]
    axis.axvline(
        selected, color="#54A24B", linestyle="-.", linewidth=2.0,
        label=f"Direction-B common evaluator: {selected}",
    )
    axis.set_title(
        "B3 Common High-Fidelity Evaluator Qualification",
        fontsize=15.5,
        fontweight="bold",
    )
    axis.set_xlabel("Candidate samples per physical edge")
    axis.set_ylabel(r"Maximum adjacent $|\Delta J_A|$")
    axis.grid(which="both", alpha=0.23)
    axis.legend(fontsize=9.5)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _case(result, terrain_name, sensor_name, case_id):
    matches = [
        case for case in result["cases"]
        if case["terrain_name"] == terrain_name
        and case["sensor_name"] == sensor_name
        and case["case_id"] == case_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one B3 case for {terrain_name}/{sensor_name}/{case_id}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "results/direction_b/b3_multiterrain_nested_consistency.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/direction_b/figures"),
    )
    arguments = parser.parse_args()
    paths = create_b3_figures(
        arguments.result, arguments.output_directory, Path.cwd()
    )
    print(json.dumps({
        "trajectory_figure": str(paths[0].resolve()),
        "consistency_figure": str(paths[1].resolve()),
        "evaluator_figure": str(paths[2].resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
