"""Create a deliberately simple two-panel explanation of the 3D update."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "results" / "turn_limited_3d_coarse"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def _bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    count: int = 80,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)[:, None]
    return (
        (1.0 - t) ** 3 * p0
        + 3.0 * (1.0 - t) ** 2 * t * p1
        + 3.0 * (1.0 - t) * t**2 * p2
        + t**3 * p3
    )


def create_figure() -> plt.Figure:
    with np.load(RESULT_DIR / "trajectory_data.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    with (RESULT_DIR / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    figure, (axis_rule, axis_actual) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), constrained_layout=True,
    )

    # Panel 1: one common start and target. Only the turning rule differs.
    start = np.array([0.0, 0.0])
    corner = np.array([250.0, 0.0])
    target = np.array([430.0, 180.0])
    legacy = np.vstack((start, corner, target))
    straight_entry = np.column_stack(
        (np.linspace(0.0, 180.0, 25), np.zeros(25))
    )
    bend = _bezier(
        np.array([180.0, 0.0]),
        np.array([270.0, 0.0]),
        np.array([345.0, 95.0]),
        target,
    )
    gradual = np.vstack((straight_entry, bend[1:]))
    axis_rule.plot(
        legacy[:, 0], legacy[:, 1], "--o",
        color="#D1495B", linewidth=3.0, markersize=6,
        label="Old: heading can jump",
    )
    axis_rule.plot(
        gradual[:, 0], gradual[:, 1],
        color="#00798C", linewidth=3.4,
        label="New rule: heading changes gradually",
    )
    axis_rule.scatter(
        *start, marker="*", s=170, color="#EDAE49",
        edgecolor="black", zorder=5,
    )
    axis_rule.scatter(
        *target, marker="*", s=170, color="#59A14F",
        edgecolor="black", zorder=5,
    )
    axis_rule.annotate(
        "unlimited corner",
        xy=corner,
        xytext=(175.0, 85.0),
        color="#9E2736",
        arrowprops={"arrowstyle": "->", "color": "#9E2736"},
        fontsize=11,
    )
    axis_rule.text(
        0.04, 0.95,
        "FIXED\nheading is now a state\n|turn rate| ≤ 5°/s",
        transform=axis_rule.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "#E6F4F6", "edgecolor": "#00798C", "alpha": 0.95},
    )
    axis_rule.set(
        title="1. Turning rule",
        xlabel="x (schematic)",
        ylabel="y (schematic)",
    )
    axis_rule.set_aspect("equal", adjustable="box")
    axis_rule.grid(alpha=0.20)
    axis_rule.legend(loc="lower right", fontsize=9)

    # Panel 2: only the actual top-view diagnostic.
    x_grid = arrays["terrain_x"]
    y_grid = arrays["terrain_y"]
    terrain = arrays["terrain_height"]
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid, indexing="ij")
    snapped = arrays["trajectory"]
    replay = arrays["continuous_control_reconstruction"]
    goal = arrays["goal_position"]
    contours = axis_actual.contourf(
        mesh_x, mesh_y, terrain, levels=15, cmap="terrain", alpha=0.78,
    )
    axis_actual.plot(
        snapped[:, 0], snapped[:, 1], "-o",
        color="#6A3D9A", linewidth=3.0, markersize=3.5,
        label="Grid-snapped result",
    )
    axis_actual.plot(
        replay[:, 0], replay[:, 1], "--",
        color="#E66101", linewidth=3.2,
        label="Open-loop replay of recorded controls (no grid reset)",
    )
    axis_actual.scatter(
        goal[0], goal[1], marker="*", s=180,
        color="#59A14F", edgecolor="black", zorder=6, label="Goal",
    )
    axis_actual.scatter(
        replay[-1, 0], replay[-1, 1], marker="X", s=105,
        color="#E66101", edgecolor="black", zorder=6,
    )
    axis_actual.annotate(
        "continuous replay\nends here",
        xy=replay[-1, :2],
        xytext=(1080.0, -900.0),
        arrowprops={"arrowstyle": "->", "color": "#A94700"},
        color="#A94700",
        fontsize=10.5,
    )
    axis_actual.annotate(
        "snapping makes the path\nartificially reach the goal",
        xy=snapped[-3, :2],
        xytext=(1650.0, 520.0),
        arrowprops={"arrowstyle": "->", "color": "#4E287A"},
        color="#4E287A",
        fontsize=10.5,
    )
    axis_actual.text(
        0.03, 0.97,
        (
            "NOT FIXED YET\n"
            "off-grid endpoint → nearest node reset\n"
            f"continuous goal error ≈ "
            f"{summary['continuous_control_reconstruction_goal_error_m']:.0f} m"
        ),
        transform=axis_actual.transAxes,
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "#FFF0E6", "edgecolor": "#E66101", "alpha": 0.95},
    )
    axis_actual.set(
        title="2. Remaining spatial-transition problem",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axis_actual.set_aspect("equal", adjustable="box")
    figure.colorbar(
        contours, ax=axis_actual, label="terrain height (m)", shrink=0.80,
    )

    figure.suptitle(
        "3D Update: One Issue Fixed, One Issue Still Remaining",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.015,
        (
            "Left is a model schematic. Right is the actual coarse-run "
            "diagnostic. The right-hand path must not be used as a final result."
        ),
        ha="center",
        fontsize=9.5,
        color="0.30",
    )
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    png_path = OUTPUT_DIR / "3d_update_fixed_vs_remaining.png"
    pdf_path = OUTPUT_DIR / "3d_update_fixed_vs_remaining.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
