"""Render the saved 3D PoD/time-weight comparison for presentation.

The expensive 3D solves are already embedded as Plotly outputs in
``p1b_3DExtension.ipynb``.  This script extracts those numerical outputs and
creates a deterministic, slide-ready comparison without rerunning Bellman.

Run from the repository root::

    python -m p1b_3DExtension.create_weight_tradeoff_visualization
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "p1b_3DExtension" / "p1b_3DExtension.ipynb"
OUTPUT_DIR = REPO_ROOT / "result_3D_visualization"


SCENARIO_STYLE = {
    "w_pod=0.75, w_time=0.25": {
        "short": "Detection priority",
        "color": "#c0392b",
        "w_pod": 0.75,
        "w_time": 0.25,
    },
    "w_pod=0.5, w_time=0.5": {
        "short": "Balanced",
        "color": "#2f6fb2",
        "w_pod": 0.50,
        "w_time": 0.50,
    },
    "w_pod=0.25, w_time=0.75": {
        "short": "Time priority",
        "color": "#2e9e5b",
        "w_pod": 0.25,
        "w_time": 0.75,
    },
}


def _array(value: Any) -> np.ndarray:
    if not isinstance(value, dict) or "bdata" not in value:
        return np.asarray(value, dtype=float)
    result = np.frombuffer(
        base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"])
    )
    if "shape" in value:
        result = result.reshape(tuple(int(item) for item in value["shape"].split(", ")))
    return np.asarray(result, dtype=float)


def _coordinates(trace: dict[str, Any], dimensions: str) -> np.ndarray:
    return np.column_stack([_array(trace[name]) for name in dimensions])


def _saved_plotly_figures() -> tuple[dict[str, Any], dict[str, Any]]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    figures: list[dict[str, Any]] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            figure = output.get("data", {}).get("application/vnd.plotly.v1+json")
            if figure is not None:
                figures.append(figure)
    trajectory = next(
        figure for figure in figures
        if any(trace.get("type") == "surface" for trace in figure["data"])
        and sum(trace.get("type") == "scatter3d" for trace in figure["data"]) >= 6
    )
    pod_history = next(
        figure for figure in figures
        if len(figure["data"]) == 3
        and all(trace.get("type") == "scatter" for trace in figure["data"])
    )
    return trajectory, pod_history


def _hill_height(x: float, y: float) -> float:
    return float(100.0 * np.exp(-((x - 1500.0) ** 2 + y**2) / (2.0 * 500.0**2)))


def create_figure() -> tuple[Path, Path]:
    trajectory_figure, pod_figure = _saved_plotly_figures()
    traces = trajectory_figure["data"]
    terrain_trace = traces[0]
    terrain_x = _array(terrain_trace["x"])
    terrain_y = _array(terrain_trace["y"])
    terrain_h = _array(terrain_trace["z"])
    terrain_x_mesh, terrain_y_mesh = np.meshgrid(terrain_x, terrain_y)

    scenarios: dict[str, dict[str, Any]] = {}
    for index in range(1, len(traces), 2):
        powered_trace = traces[index]
        glide_trace = traces[index + 1]
        label = glide_trace["name"]
        scenarios[label] = {
            **SCENARIO_STYLE[label],
            "powered": _coordinates(powered_trace, "xyz"),
            "glide": _coordinates(glide_trace, "xyz"),
        }
    for trace in pod_figure["data"]:
        scenarios[trace["name"]]["times"] = _array(trace["x"])
        scenarios[trace["name"]]["pod"] = _array(trace["y"])

    ordered_labels = (
        "w_pod=0.75, w_time=0.25",
        "w_pod=0.5, w_time=0.5",
        "w_pod=0.25, w_time=0.75",
    )
    sensor = np.array([2000.0, 0.0, _hill_height(2000.0, 0.0)])
    goal = np.array([2500.0, 0.0, _hill_height(2500.0, 0.0)])

    figure = plt.figure(figsize=(16.0, 9.0), constrained_layout=False)
    figure.subplots_adjust(
        left=0.035, right=0.985, top=0.86, bottom=0.14,
        wspace=0.18, hspace=0.34,
    )
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_top = figure.add_subplot(grid[0, 1])
    axis_pod = figure.add_subplot(grid[1, 1])

    axis_3d.plot_surface(
        terrain_x_mesh,
        terrain_y_mesh,
        terrain_h,
        cmap="terrain",
        vmin=0.0,
        vmax=100.0,
        alpha=0.58,
        linewidth=0.12,
        edgecolor=(0.1, 0.1, 0.1, 0.14),
        antialiased=True,
    )
    for label in ordered_labels[::-1]:
        scenario = scenarios[label]
        powered = scenario["powered"]
        glide = scenario["glide"]
        color = scenario["color"]
        # A pale underlay keeps all three paths legible over the terrain.
        axis_3d.plot(
            powered[:, 0], powered[:, 1], powered[:, 2],
            color="white", linestyle="--", linewidth=5.8, alpha=0.85,
        )
        axis_3d.plot(
            powered[:, 0], powered[:, 1], powered[:, 2],
            color=color, linestyle="--", linewidth=3.2,
        )
        axis_3d.plot(
            glide[:, 0], glide[:, 1], glide[:, 2],
            color="white", linewidth=6.8, alpha=0.85,
        )
        axis_3d.plot(
            glide[:, 0], glide[:, 1], glide[:, 2],
            color=color, linewidth=4.0,
        )
        axis_3d.scatter(
            *glide[0], color=color, edgecolor="white", s=70, zorder=10
        )
    axis_3d.scatter(*sensor, marker="^", color="#111111", edgecolor="white", s=110)
    axis_3d.scatter(*goal, marker="*", color="#f4c20d", edgecolor="black", s=175)
    axis_3d.set_xlim(0.0, 2750.0)
    axis_3d.set_ylim(-1050.0, 350.0)
    axis_3d.set_zlim(0.0, 205.0)
    axis_3d.set_box_aspect((2.35, 1.35, 0.62))
    axis_3d.view_init(elev=28.0, azim=-62.0)
    axis_3d.set_xlabel("Downrange x [m]", labelpad=8)
    axis_3d.set_ylabel("Cross-range y [m]", labelpad=9)
    axis_3d.set_zlabel("Altitude h [m]", labelpad=3)
    axis_3d.set_title("3D route comparison", pad=12)
    axis_3d.grid(True, alpha=0.22)

    terrain_fill = axis_top.contourf(
        terrain_x_mesh,
        terrain_y_mesh,
        terrain_h,
        levels=np.linspace(0.0, 100.0, 16),
        cmap="terrain",
        alpha=0.80,
    )
    terrain_contours = axis_top.contour(
        terrain_x_mesh,
        terrain_y_mesh,
        terrain_h,
        levels=(20.0, 40.0, 60.0, 80.0),
        colors="#314c36",
        linewidths=0.65,
        alpha=0.70,
    )
    axis_top.clabel(terrain_contours, fontsize=7.5, fmt="%d m")
    for label in ordered_labels[::-1]:
        scenario = scenarios[label]
        powered = scenario["powered"]
        glide = scenario["glide"]
        color = scenario["color"]
        axis_top.plot(
            powered[:, 0], powered[:, 1], color=color,
            linestyle="--", linewidth=2.1, alpha=0.9,
        )
        axis_top.plot(glide[:, 0], glide[:, 1], color=color, linewidth=3.1)
        axis_top.scatter(
            glide[0, 0], glide[0, 1], color=color,
            edgecolor="white", s=55, zorder=6,
        )
    axis_top.scatter(sensor[0], sensor[1], marker="^", color="#111111", edgecolor="white", s=75, zorder=7)
    axis_top.scatter(goal[0], goal[1], marker="*", color="#f4c20d", edgecolor="black", s=120, zorder=7)
    axis_top.set_xlim(0.0, 2750.0)
    axis_top.set_ylim(-1050.0, 350.0)
    axis_top.set_aspect("equal", adjustable="box")
    axis_top.set_xlabel("Downrange x [m]")
    axis_top.set_ylabel("Cross-range y [m]")
    axis_top.set_title("Top-down lateral route choice")
    axis_top.grid(True, linewidth=0.35, alpha=0.25)

    for label in ordered_labels:
        scenario = scenarios[label]
        times = scenario["times"]
        pod = scenario["pod"]
        color = scenario["color"]
        powered_sample_count = scenario["powered"].shape[0]
        switch_index = min(powered_sample_count - 1, times.size - 1)
        axis_pod.plot(
            times, 100.0 * pod, color=color, linewidth=3.0,
            label=scenario["short"],
        )
        axis_pod.scatter(
            times[switch_index], 100.0 * pod[switch_index],
            color=color, edgecolor="white", s=45, zorder=6,
        )
        axis_pod.annotate(
            f"{100.0 * pod[-1]:.2f}%\n{times[-1]:.1f} s",
            xy=(times[-1], 100.0 * pod[-1]),
            xytext=(-5, 7),
            textcoords="offset points",
            color=color,
            fontsize=8.5,
            ha="right",
        )
    axis_pod.set_xlabel("Mission time [s]")
    axis_pod.set_ylabel("Cumulative PoD [%]")
    axis_pod.set_ylim(0.0, 6.4)
    axis_pod.set_title("Detection–time trade-off")
    axis_pod.grid(True, linewidth=0.45, alpha=0.30)

    legend_handles = []
    for label in ordered_labels:
        scenario = scenarios[label]
        y_min = float(np.min(scenario["glide"][:, 1]))
        final_pod = 100.0 * float(scenario["pod"][-1])
        final_time = float(scenario["times"][-1])
        legend_handles.append(Line2D(
            [0], [0], color=scenario["color"], linewidth=4,
            label=(
                f"{scenario['short']}  "
                f"(wPoD={scenario['w_pod']:.2f}, y_min={y_min:.0f} m, "
                f"PoD={final_pod:.2f}%, T={final_time:.1f} s)"
            ),
        ))
    legend_handles.extend((
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=3, label="Powered phase"),
        Line2D([0], [0], color="#555555", linewidth=3, label="Glide phase"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#111111", markeredgecolor="white", markersize=9, label="Sensor"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#f4c20d", markeredgecolor="black", markersize=12, label="Goal"),
    ))
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=True,
        fontsize=8.7,
    )
    figure.suptitle(
        "3D Extension: Detection–Time Weighting Changes the Lateral Route",
        fontsize=18,
        y=0.975,
    )
    figure.text(
        0.5,
        0.932,
        "Same terrain and sensor; only the attacker objective weights change — coarse-grid qualitative prototype",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "meeting_3d_weight_tradeoff.png"
    pdf_path = OUTPUT_DIR / "meeting_3d_weight_tradeoff.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = create_figure()
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
