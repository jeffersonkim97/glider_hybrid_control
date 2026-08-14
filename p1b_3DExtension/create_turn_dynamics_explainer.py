"""Create an advisor-facing schematic of the 3D heading-state update."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import vehicle_config
from .turn_dynamics import signed_heading_change


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


def _path_from_headings(headings_deg: np.ndarray, segment_length: float) -> np.ndarray:
    headings = np.deg2rad(headings_deg)
    increments = segment_length * np.column_stack(
        (np.cos(headings), np.sin(headings))
    )
    return np.vstack((np.zeros(2), np.cumsum(increments, axis=0)))


def create_figure() -> plt.Figure:
    max_rate = float(
        vehicle_config["turn_dynamics"]["max_turn_rate_deg_s"]
    )
    coarse_dt = 3.0
    heading_spacing = 10.0
    speed = 20.0
    segment_length = speed * coarse_dt

    legacy_heading = np.array([0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 50.0, 50.0])
    limited_heading = np.array([0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 50.0])
    legacy_path = _path_from_headings(legacy_heading, segment_length)
    limited_path = _path_from_headings(limited_heading, segment_length)
    node_time = coarse_dt * np.arange(legacy_heading.size)

    figure = plt.figure(figsize=(13.0, 5.0), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.05))
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    axes[0].plot(
        legacy_path[:, 0], legacy_path[:, 1], "-o",
        color="#D1495B", linewidth=2.8, markersize=4.5,
    )
    corner = legacy_path[4]
    axes[0].annotate(
        "50° instantaneous change\n(no heading memory)",
        xy=corner,
        xytext=(corner[0] - 165.0, corner[1] + 125.0),
        arrowprops={"arrowstyle": "->", "color": "#8B1E2D"},
        color="#8B1E2D",
        fontsize=10,
    )
    axes[0].set_title("Legacy: heading is an action")

    axes[1].plot(
        limited_path[:, 0], limited_path[:, 1], "-o",
        color="#00798C", linewidth=2.8, markersize=4.5,
    )
    for index in range(1, 7):
        heading = np.deg2rad(limited_heading[index - 1])
        axes[1].quiver(
            limited_path[index - 1, 0],
            limited_path[index - 1, 1],
            np.cos(heading),
            np.sin(heading),
            angles="xy",
            scale_units="xy",
            scale=0.018,
            width=0.008,
            color="#28536B",
            alpha=0.8,
        )
    axes[1].annotate(
        "10° grid increments\nwithin 15° per-step bound",
        xy=limited_path[4],
        xytext=(limited_path[4, 0] - 145.0, limited_path[4, 1] + 105.0),
        arrowprops={"arrowstyle": "->", "color": "#005B66"},
        color="#005B66",
        fontsize=10,
    )
    axes[1].set_title("Updated: heading is a state")

    for axis in axes[:2]:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.22)
        axis.scatter([0.0], [0.0], marker="*", s=150, color="#EDAE49", zorder=5)

    legacy_change = np.rad2deg(
        signed_heading_change(
            np.deg2rad(legacy_heading[:-1]), np.deg2rad(legacy_heading[1:])
        )
    )
    limited_change = np.rad2deg(
        signed_heading_change(
            np.deg2rad(limited_heading[:-1]), np.deg2rad(limited_heading[1:])
        )
    )
    axes[2].step(
        node_time, legacy_heading, where="post",
        linewidth=2.5, color="#D1495B", label="Legacy heading",
    )
    axes[2].step(
        node_time, limited_heading, where="post",
        linewidth=2.5, color="#00798C", label="Turn-limited heading",
    )
    axes[2].fill_between(
        node_time,
        limited_heading - max_rate * coarse_dt,
        limited_heading + max_rate * coarse_dt,
        step="post",
        color="#00798C",
        alpha=0.10,
        label="± max-rate envelope",
    )
    axes[2].set_title("Heading carried between stages")
    axes[2].set_xlabel("glide time (s)")
    axes[2].set_ylabel("heading ψ (deg)")
    axes[2].grid(alpha=0.22)
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].text(
        0.03, 0.96,
        (
            f"Configured |ψ̇| ≤ {max_rate:.0f}°/s\n"
            f"coarse Δt ≈ {coarse_dt:.0f} s; heading grid = {heading_spacing:.0f}°\n"
            f"legacy max jump = {np.max(np.abs(legacy_change)):.0f}°\n"
            f"updated max jump = {np.max(np.abs(limited_change)):.0f}°"
        ),
        transform=axes[2].transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.95},
    )

    figure.suptitle(
        "Why a Free Heading Did Not Produce a Smooth 3D Turn",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5, -0.035,
        (
            "Model schematic (not an optimized mission result): straight "
            "corridors remain expected, but unlimited corners are removed."
        ),
        ha="center",
        fontsize=9.5,
        color="0.30",
    )
    return figure


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure = create_figure()
    png_path = OUTPUT_DIR / "heading_state_turn_rate_explainer.png"
    pdf_path = OUTPUT_DIR / "heading_state_turn_rate_explainer.pdf"
    figure.savefig(png_path, dpi=260, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
