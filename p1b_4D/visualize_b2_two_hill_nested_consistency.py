"""Publication-oriented figures for the Direction-B B2 two-hill result."""
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
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    B2_SENSOR_CANDIDATES,
    build_two_hill_configuration,
)
from p1b_4D.geometry import (
    build_geometry_bundle,
    los_boundary_height,
    terrain_height,
)
from p1b_4D.phase_logging import close_phase_logger


LEVEL_COLORS = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}
SENSOR_COLORS = {"coverage": "#4C78A8", "stackelberg": "#E45756"}


def create_b2_figures(
    result_path: Path,
    output_directory: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Create the trajectory/consistency dashboard and evaluator figure."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError("B2 result must have complete status")
    output_directory.mkdir(parents=True, exist_ok=True)
    base = build_configuration_bundle(project_root)
    try:
        geometries = {}
        for sensor_name, sensor_z in B2_SENSOR_CANDIDATES.items():
            physical = build_two_hill_configuration(base, sensor_z)
            bundle = build_direction_b_configuration(
                physical, "two_hill", 2
            )
            geometries[sensor_name] = build_geometry_bundle(bundle)
        summary_path = output_directory / "b2_two_hill_nested_consistency.png"
        evaluator_path = output_directory / "b2_common_evaluator_qualification.png"
        _plot_summary(result, geometries, summary_path)
        _plot_evaluator_qualification(result, evaluator_path)
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )
    return summary_path, evaluator_path


def _plot_summary(result, geometries, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), constrained_layout=True)
    for column, sensor_name in enumerate(("coverage", "stackelberg")):
        _plot_trajectory_panel(
            axes[0, column], result, geometries[sensor_name], sensor_name
        )
    _plot_objective_panel(axes[1, 0], result)
    _plot_ranking_panel(axes[1, 1], result)
    figure.suptitle(
        "B2 Two-Hill Nested-Discretization Consistency",
        fontsize=17,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_trajectory_panel(axis, result, geometry_bundle, sensor_name):
    geometry = geometry_bundle["primary_result"]
    z = np.linspace(0.0, 2750.0, 1800)
    ground = terrain_height(geometry["terrain_model"], z)
    boundary = los_boundary_height(geometry["los_geometry"], z)
    sensor = np.asarray(geometry["sensor_position"])
    goal = np.asarray(geometry["goal_position"])
    before_sensor = z <= sensor[0]

    axis.fill_between(z, 0.0, ground, color="#5A5A5A", alpha=0.72, label="Terrain")
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
        color="black", linestyle="--", linewidth=1.35, label="LOS boundary",
    )
    for level in range(3):
        case = _case(result, sensor_name, f"enriched_v5_q9_l{level}")
        path = np.asarray(case["planning"]["trajectory"], dtype=float)
        objective = case["selected_high_fidelity"]["attacker_objective"]
        axis.plot(
            path[:, 0], path[:, 1],
            color=LEVEL_COLORS[level], linewidth=2.15,
            marker="o", markersize=2.2 if level == 2 else 3.0,
            markevery=max(1, path.shape[0] // 18),
            label=f"L{level}: $J_A^{{HF}}$={objective:.4f}",
        )
        switch = path[0]
        axis.scatter(
            switch[0], switch[1], s=48, facecolors="white",
            edgecolors=LEVEL_COLORS[level], linewidths=1.6, zorder=7,
        )
    axis.scatter(
        sensor[0], sensor[1], marker="^", s=105, color="#D62728",
        edgecolor="white", linewidth=0.7, zorder=8, label="Sensor",
    )
    axis.scatter(
        goal[0], goal[1], marker="*", s=175, color="#F2C80F",
        edgecolor="black", linewidth=0.6, zorder=8, label="Goal",
    )
    axis.set_xlim(0.0, 2750.0)
    axis.set_ylim(0.0, 205.0)
    axis.set_xlabel("Along-track position, $z$ (m)")
    axis.set_ylabel("Altitude, $h$ (m)")
    title = "Coverage-only sensor" if sensor_name == "coverage" else "Stackelberg sensor"
    axis.set_title(f"{title}: $z_s$={sensor[0]:.2f} m", fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(loc="upper right", fontsize=8.2, framealpha=0.94, ncol=2)


def _plot_objective_panel(axis, result):
    levels = np.arange(3)
    for sensor_name in ("coverage", "stackelberg"):
        hf = []
        planning = []
        for level in levels:
            case = _case(result, sensor_name, f"enriched_v5_q9_l{level}")
            hf.append(case["selected_high_fidelity"]["attacker_objective"])
            planning.append(case["planning"]["mission_cost"])
        label = "Coverage" if sensor_name == "coverage" else "Stackelberg"
        color = SENSOR_COLORS[sensor_name]
        axis.plot(
            levels, hf, color=color, marker="o", linewidth=2.4,
            label=f"{label}: common HF",
        )
        axis.plot(
            levels, planning, color=color, marker="x", linewidth=1.3,
            linestyle=":", alpha=0.75, label=f"{label}: planning Q9",
        )
        for level, value in zip(levels, hf):
            axis.annotate(
                f"{value:.3f}", (level, value), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8,
                color=color,
            )
    axis.set_xticks(levels, ("L0", "L1", "L2"))
    axis.set_xlabel("Nested spatial level")
    axis.set_ylabel("Attacker objective")
    axis.set_title("Resolution response of the selected follower", fontweight="bold")
    axis.grid(alpha=0.23)
    axis.legend(fontsize=8.5, ncol=2)


def _plot_ranking_panel(axis, result):
    analysis = result["analysis"]
    margins = np.array([
        analysis["defender_margin_by_level"][f"l{level}"]
        for level in range(3)
    ])
    resolution_shift = analysis["defender_resolution_shift_l1_to_l2"]
    x = np.arange(3)
    bars = axis.bar(
        x, margins, color=[LEVEL_COLORS[level] for level in range(3)],
        width=0.62, alpha=0.9, label=r"$M_\ell$",
    )
    axis.axhline(
        2.0 * resolution_shift, color="#B279A2", linestyle="--",
        linewidth=2.0, label="$2R_{1\to2}$ diagnostic threshold",
    )
    for bar, value in zip(bars, margins):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.00045,
            f"{value:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    axis.set_xticks(x, ("L0", "L1", "L2"))
    axis.set_xlabel("Nested spatial level")
    axis.set_ylabel("Defender margin: Stackelberg − Coverage")
    axis.set_ylim(0.0, 0.032)
    axis.set_title("Fixed-candidate ranking diagnostic", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8.7, loc="upper right")
    decision = (
        f"Sign stable: {analysis['ranking_sign_stable']}\n"
        f"Diagnostic resolved: {analysis['ranking_diagnostically_resolved']}\n"
        f"Production: {analysis['production_speed_family']} / "
        f"Q{analysis['production_planning_quadrature_count']}\n"
        f"Common replay: {analysis['common_evaluator_sample_count']} samples/edge"
    )
    axis.text(
        0.03, 0.97, decision, transform=axis.transAxes,
        ha="left", va="top", fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92},
    )


def _plot_evaluator_qualification(result, output_path):
    feasible = [case for case in result["cases"] if case["status"] == "feasible"]
    pair_keys = tuple(next(iter(feasible))["high_fidelity"]["qualifications"])
    lower_counts = np.array([int(key.split("_vs_")[0]) for key in pair_keys])
    objective_differences = np.array([
        [
            case["high_fidelity"]["qualifications"][key][
                "objective_absolute_difference"
            ]
            for key in pair_keys
        ]
        for case in feasible
    ])
    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for values in objective_differences:
        axis.loglog(
            lower_counts, values, color="#9C9C9C", alpha=0.34,
            linewidth=1.0,
        )
    maximum = np.max(objective_differences, axis=0)
    axis.loglog(
        lower_counts, maximum, color="#E45756", marker="o",
        markersize=6.5, linewidth=2.5, label="Maximum over 12 feasible policies",
    )
    axis.axhline(1e-6, color="black", linestyle="--", linewidth=1.6, label="Acceptance tolerance")
    selected = result["global_common_evaluator_sample_count"]
    axis.axvline(
        selected, color="#54A24B", linestyle="-.", linewidth=1.8,
        label=f"Selected common evaluator: {selected}",
    )
    for x_value, y_value in zip(lower_counts, maximum):
        axis.annotate(
            f"{y_value:.1e}", (x_value, y_value), xytext=(5, 6),
            textcoords="offset points", fontsize=8.5,
        )
    axis.set_xlabel("Candidate samples per physical edge")
    axis.set_ylabel(r"Adjacent-resolution $|\Delta J_A|$")
    axis.set_title(
        "B2 Common-Evaluator Qualification\n"
        "Feasibility and goal classifications matched at every count",
        fontweight="bold",
    )
    axis.grid(which="both", alpha=0.22)
    axis.legend(fontsize=9.2)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _case(result: dict[str, Any], sensor_name: str, case_id: str) -> dict[str, Any]:
    return next(
        case for case in result["cases"]
        if case["sensor_name"] == sensor_name and case["case_id"] == case_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/direction_b/b2_two_hill_nested_consistency.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/direction_b/figures"),
    )
    arguments = parser.parse_args()
    outputs = create_b2_figures(
        arguments.result, arguments.output_directory, Path.cwd()
    )
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
