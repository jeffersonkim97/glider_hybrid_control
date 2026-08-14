"""Create a presentation-ready static figure from the exploratory 3D result.

The exploratory result is stored as self-contained Plotly HTML.  This script
recovers the numerical arrays from those files and renders a deterministic
Matplotlib PNG/PDF suitable for slides, without rerunning the multi-hour
Stackelberg solve.

Run from the repository root::

    python -m p1b_3DExtension.create_meeting_visualization
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_DIR = REPO_ROOT / "result_3D_visualization"


def _plotly_arguments(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract data and layout JSON from the final Plotly.newPlot call."""
    text = path.read_text(encoding="utf-8")
    marker = "Plotly.newPlot("
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(f"No Plotly.newPlot call found in {path}")
    source = text[start + len(marker) :]
    arguments: list[str] = []
    argument_start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "[{(":
            depth += 1
        elif character in "]})":
            if character == ")" and depth == 0:
                arguments.append(source[argument_start:index].strip())
                break
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(source[argument_start:index].strip())
            argument_start = index + 1
    if len(arguments) < 3:
        raise ValueError(f"Could not parse Plotly arguments from {path}")
    return json.loads(arguments[1]), json.loads(arguments[2])


def _array(value: Any) -> np.ndarray:
    """Decode Plotly 6 typed-array JSON or a conventional JSON list."""
    if not isinstance(value, dict) or "bdata" not in value:
        return np.asarray(value)
    result = np.frombuffer(
        base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"])
    )
    if "shape" in value:
        result = result.reshape(tuple(int(item) for item in value["shape"].split(", ")))
    return result


def _coordinates(trace: dict[str, Any], dimensions: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([_array(trace[name]) for name in dimensions])


def create_meeting_figure(result_dir: Path = DEFAULT_RESULT_DIR) -> tuple[Path, Path]:
    trajectory_data, _ = _plotly_arguments(result_dir / "trajectory_3d.html")
    occlusion_data, _ = _plotly_arguments(result_dir / "occlusion_topdown.html")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))

    terrain_trace = trajectory_data[0]
    terrain_x = _array(terrain_trace["x"])
    terrain_y = _array(terrain_trace["y"])
    terrain_h = _array(terrain_trace["z"])
    terrain_x_mesh, terrain_y_mesh = np.meshgrid(terrain_x, terrain_y)

    powered = _coordinates(trajectory_data[1], ("x", "y", "z"))
    glide = _coordinates(trajectory_data[2], ("x", "y", "z"))
    sensor = _coordinates(trajectory_data[3], ("x", "y", "z"))[0]
    goal = _coordinates(trajectory_data[4], ("x", "y", "z"))[0]
    switch = glide[0]

    occlusion_trace = occlusion_data[0]
    occlusion_x = _array(occlusion_trace["x"])
    occlusion_y = _array(occlusion_trace["y"])
    occlusion = _array(occlusion_trace["z"])

    powered_color = "#d97706"
    glide_color = "#5b21b6"
    sensor_color = "#c62828"
    goal_color = "#f4c20d"
    switch_color = "#8e5bb7"

    figure = plt.figure(figsize=(16.0, 8.4), constrained_layout=False)
    figure.subplots_adjust(
        left=0.035, right=0.985, top=0.865, bottom=0.155, wspace=0.12
    )
    grid = figure.add_gridspec(1, 2, width_ratios=(1.2, 1.0))
    axis_3d = figure.add_subplot(grid[0, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])

    axis_3d.plot_surface(
        terrain_x_mesh,
        terrain_y_mesh,
        terrain_h,
        cmap="terrain",
        vmin=0.0,
        vmax=max(100.0, float(np.max(terrain_h))),
        alpha=0.68,
        linewidth=0.15,
        edgecolor=(0.15, 0.15, 0.15, 0.18),
        antialiased=True,
        shade=True,
    )
    axis_3d.plot(
        powered[:, 0], powered[:, 1], powered[:, 2],
        color=powered_color, linestyle="--", linewidth=4.0,
    )
    axis_3d.plot(
        glide[:, 0], glide[:, 1], glide[:, 2],
        color=glide_color, linewidth=5.0,
    )
    axis_3d.scatter(*switch, color=switch_color, edgecolor="white", s=90, zorder=10)
    axis_3d.scatter(*sensor, marker="^", color=sensor_color, edgecolor="black", s=105)
    axis_3d.scatter(*goal, marker="*", color=goal_color, edgecolor="black", s=165)
    axis_3d.set_xlabel("Downrange x [m]", labelpad=8)
    axis_3d.set_ylabel("Cross-range y [m]", labelpad=10)
    axis_3d.set_zlabel("Altitude h [m]", labelpad=3)
    axis_3d.set_xlim(0.0, 2750.0)
    axis_3d.set_ylim(-700.0, 500.0)
    axis_3d.set_zlim(0.0, 210.0)
    axis_3d.set_box_aspect((2.45, 1.45, 0.62))
    axis_3d.view_init(elev=26.0, azim=-70.0)
    axis_3d.set_title("3D terrain and powered-to-glide response", pad=10)
    axis_3d.grid(True, alpha=0.25)

    occlusion_cmap = ListedColormap(["#edf7ed", "#4b4b4b"])
    axis_top.pcolormesh(
        occlusion_x,
        occlusion_y,
        occlusion,
        cmap=occlusion_cmap,
        vmin=0,
        vmax=1,
        shading="nearest",
        alpha=0.92,
    )
    terrain_levels = (10.0, 30.0, 50.0, 70.0, 90.0)
    contours = axis_top.contour(
        terrain_x_mesh,
        terrain_y_mesh,
        terrain_h,
        levels=terrain_levels,
        colors="#2f6b3c",
        linewidths=0.8,
        alpha=0.75,
    )
    axis_top.clabel(contours, inline=True, fontsize=8, fmt="%d m")
    axis_top.plot(powered[:, 0], powered[:, 1], color=powered_color, linestyle="--", linewidth=2.7)
    axis_top.plot(glide[:, 0], glide[:, 1], color=glide_color, linewidth=3.2)
    axis_top.scatter(switch[0], switch[1], color=switch_color, edgecolor="white", s=75, zorder=5)
    axis_top.scatter(sensor[0], sensor[1], marker="^", color=sensor_color, edgecolor="black", s=90, zorder=5)
    axis_top.scatter(goal[0], goal[1], marker="*", color=goal_color, edgecolor="black", s=145, zorder=5)
    axis_top.annotate(
        "lateral offset",
        xy=(1450.0, -150.0),
        xytext=(1150.0, -520.0),
        arrowprops=dict(arrowstyle="->", color=glide_color, linewidth=1.4),
        color=glide_color,
        fontsize=10,
    )
    axis_top.set_xlim(0.0, 2750.0)
    axis_top.set_ylim(-900.0, 650.0)
    axis_top.set_aspect("equal", adjustable="box")
    axis_top.set_xlabel("Downrange x [m]")
    axis_top.set_ylabel("Cross-range y [m]")
    axis_top.set_title("Top-down visibility at h = 100 m")
    axis_top.grid(True, linewidth=0.4, alpha=0.28)

    metric_text = (
        f"Sensor = ({sensor[0]:.0f}, {sensor[1]:.0f}, {sensor[2]:.1f}) m\n"
        f"Switch = ({switch[0]:.0f}, {switch[1]:.0f}, {switch[2]:.1f}) m\n"
        f"Mission PoD = {summary['mission_pod']:.3f}   |   "
        f"Mission time = {summary['mission_time']:.1f} s"
    )
    axis_top.text(
        0.02,
        0.02,
        metric_text,
        transform=axis_top.transAxes,
        fontsize=9.5,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#777777", alpha=0.92),
    )

    legend_handles = [
        Line2D([0], [0], color=powered_color, linestyle="--", linewidth=3, label="Powered phase (occluded)"),
        Line2D([0], [0], color=glide_color, linewidth=3.5, label="Glide phase"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=switch_color, markeredgecolor="white", markersize=9, label="Switch"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=sensor_color, markeredgecolor="black", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=goal_color, markeredgecolor="black", markersize=12, label="Goal"),
        Line2D([0], [0], color="#4b4b4b", linewidth=8, label="Terrain-occluded @ 100 m"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=6,
        frameon=True,
        fontsize=9.5,
    )
    figure.suptitle(
        "Qualitative 3D Extension: Terrain-Aware Lateral Evasion",
        fontsize=18,
        y=0.975,
    )
    figure.text(
        0.5,
        0.93,
        "Coarse-grid prototype — spatial state (x, y, h), controls (v, γ, heading)",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )

    png_path = result_dir / "meeting_3d_extension_example.png"
    pdf_path = result_dir / "meeting_3d_extension_example.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()
    png_path, pdf_path = create_meeting_figure(args.result_dir.resolve())
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
