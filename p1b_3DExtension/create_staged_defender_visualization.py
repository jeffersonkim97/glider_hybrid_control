"""Create the audited staged Defender-search meeting figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .detection import build_symbolic_detection_bundle
from .experiment_staged_defender_optimization import (
    FINE_RESOLUTION,
    OUTPUT_DIR,
    REPO_ROOT,
    SCREEN_X,
    SCREEN_Y,
    _configuration,
)
from .geometry import build_geometry_bundle


FIGURE_DIR = REPO_ROOT / "result_3D_visualization"


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _last_mapped(function, arguments: tuple[np.ndarray, ...]) -> np.ndarray:
    count = arguments[0].size
    outputs = function.map(count)(
        *(np.asarray(value).reshape(1, -1) for value in arguments)
    )
    values = outputs if isinstance(outputs, tuple) else (outputs,)
    return np.maximum(0.0, np.asarray(values[-1]).reshape(-1))


def _cumulative_pod(
    sensor_xy: tuple[float, float], powered_path: np.ndarray,
    dense_time: np.ndarray, dense_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    configuration = _configuration(FINE_RESOLUTION, sensor_xy)
    geometry = build_geometry_bundle(configuration)
    detection = build_symbolic_detection_bundle(configuration, geometry)
    function = detection["primary_result"]["functions"][
        "powered_total_detection_components"
    ]
    powered_time = float(dense_time[0])
    powered_clock = np.linspace(0.0, powered_time, powered_path.shape[0])
    displacement = powered_path[-1] - powered_path[0]
    horizontal = float(np.linalg.norm(displacement[:2]))
    gamma = float(np.arctan2(displacement[2], horizontal))
    heading = float(np.arctan2(displacement[1], displacement[0]))
    speed = float(np.linalg.norm(displacement) / powered_time)
    sensor = np.asarray(geometry["primary_result"]["sensor_position"])
    count = powered_path.shape[0]
    rates = _last_mapped(function, (
        powered_path[:, 0], powered_path[:, 1], powered_path[:, 2],
        np.full(count, speed), np.full(count, gamma), np.full(count, heading),
        np.full(count, sensor[0]), np.full(count, sensor[1]),
        np.full(count, sensor[2]),
    ))
    increments = 0.5 * (rates[:-1] + rates[1:]) * np.diff(powered_clock)
    powered_hazard = np.concatenate(([0.0], np.cumsum(increments)))
    # Use the independently propagated dense hazard after switching, shifted
    # by the recomputed powered quadrature to make the curve continuous.
    glide_hazard = dense_states[7] - dense_states[7, 0] + powered_hazard[-1]
    clock = np.concatenate((powered_clock, dense_time[1:]))
    hazard = np.concatenate((powered_hazard, glide_hazard[1:]))
    return clock, 100.0 * (1.0 - np.exp(-hazard)), powered_time


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_json(OUTPUT_DIR / "optimization_summary.json")
    final = summary["continuous_fine_record"]
    fine_dir = Path(summary["selected_fine_directory"])
    with np.load(fine_dir / "trajectory_data.npz") as handle:
        terrain_x = np.asarray(handle["terrain_x"])
        terrain_y = np.asarray(handle["terrain_y"])
        terrain_height = np.asarray(handle["terrain_height"])
        goal = np.asarray(handle["goal_position"])
    with np.load(fine_dir / "continuous_trajectory.npz") as handle:
        powered_path = np.asarray(handle["powered_path"])
        dense_time = np.asarray(handle["dense_time"])
        dense_states = np.asarray(handle["dense_states"])

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
    })
    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.18))

    # A: fixed-trajectory screening map.  It is explicitly labelled as a
    # surrogate so it cannot be mistaken for the optimized Stackelberg value.
    ax_map = fig.add_subplot(grid[0, 0])
    screen = np.full((len(SCREEN_Y), len(SCREEN_X)), np.nan)
    records = summary["screening_records"]
    by_point = {
        (record["x_sensor"], record["y_sensor"]): record
        for record in records if record.get("status_success")
    }
    for ix, x_value in enumerate(SCREEN_X):
        for iy, y_value in enumerate(SCREEN_Y):
            record = by_point.get((x_value, y_value))
            if record is not None:
                screen[iy, ix] = record["defender_objective"]
    image = ax_map.imshow(
        screen, origin="lower", aspect="auto",
        extent=(min(SCREEN_X) - 100, max(SCREEN_X) + 100,
                min(SCREEN_Y) - 100, max(SCREEN_Y) + 100),
        cmap="viridis", norm=Normalize(vmin=0.0, vmax=np.nanmax(screen)),
    )
    fig.colorbar(image, ax=ax_map, pad=0.02, label="Surrogate Defender objective")
    ax_map.scatter(2600, 0, s=130, marker="*", color="#f5c542",
                   edgecolor="black", linewidth=1.0, label="Strictly validated")
    ax_map.scatter(1000, -400, s=75, marker="o", facecolor="none",
                   edgecolor="white", linewidth=1.8,
                   label="Prior validated candidate")
    ax_map.scatter(2400, 0, s=70, marker="x", color="white", linewidth=2.0,
                   label="Medium graph infeasible")
    ax_map.set(title="A  Screening map (fixed-path surrogate)",
               xlabel="Sensor x [m]", ylabel="Sensor y [m]")
    ax_map.legend(loc="lower right", framealpha=0.92)

    # B: terrain and the selected continuous attacker best response.
    ax_3d = fig.add_subplot(grid[0, 1], projection="3d")
    mesh_x, mesh_y = np.meshgrid(terrain_x, terrain_y, indexing="ij")
    ax_3d.plot_surface(
        mesh_x, mesh_y, terrain_height, cmap="terrain", alpha=0.72,
        linewidth=0.15, edgecolor=(0.25, 0.25, 0.25, 0.2), antialiased=True,
    )
    ax_3d.plot(*powered_path.T, color="#d95f02", linewidth=3.2,
               label="Powered")
    ax_3d.plot(*dense_states[:3], color="#1769aa", linewidth=3.2,
               label="Glide (continuous 3-DOF)")
    sensor = np.asarray(final["sensor_position"])
    ax_3d.scatter(*sensor, s=85, marker="^", color="#c51b7d",
                  edgecolor="black", label="Selected sensor")
    ax_3d.scatter(*goal, s=80, marker="X", color="#39a96b",
                  edgecolor="black", label="Goal")
    ax_3d.set(xlim=(500, 2700), ylim=(-1050, 1050), zlim=(0, 220),
              xlabel="x [m]", ylabel="y [m]", zlabel="h [m]",
              title="B  Fine + continuous validated best response")
    ax_3d.view_init(elev=27, azim=-61)
    ax_3d.set_box_aspect((2.2, 1.8, 0.65))
    ax_3d.legend(loc="upper left", framealpha=0.92)

    # C: candidate audit. Hatched continuous bars are informative but failed
    # the predeclared strict dense-validation thresholds.
    ax_bar = fig.add_subplot(grid[1, 0])
    labels = ("(1200, 0)", "(1000, -400)", "(2600, 0)")
    points = ((1200.0, 0.0), (1000.0, -400.0), (2600.0, 0.0))
    screening_values = [by_point[p]["defender_objective"] for p in points]
    medium_by_point = {
        tuple(record["sensor_position"][:2]): record
        for record in summary["medium_records"]
    }
    medium_values = [medium_by_point[p]["defender_objective"] for p in points]
    audits = summary["continuous_candidate_audits"]
    continuous_attempts = (
        audits["x1200_y0"][0],
        summary["continuous_medium_records"][0],
        summary["continuous_medium_records"][1],
    )
    continuous_values = [item["defender_objective"] for item in continuous_attempts]
    positions = np.arange(len(labels))
    width = 0.24
    ax_bar.bar(positions - width, screening_values, width, color="#9e9e9e",
               label="Screen surrogate")
    ax_bar.bar(positions, medium_values, width, color="#4c78a8",
               label="Exact medium")
    for index, (value, attempt) in enumerate(zip(
        continuous_values, continuous_attempts, strict=True,
    )):
        valid = attempt.get("status_success", False)
        ax_bar.bar(
            positions[index] + width, value, width,
            color="#59a14f" if valid else "#f28e2b",
            hatch=None if valid else "///", edgecolor="black", linewidth=0.6,
        )
    ax_bar.scatter(positions[2] + width, final["defender_objective"],
                   marker="*", s=110, color="#f5c542", edgecolor="black",
                   zorder=5)
    ax_bar.set_xticks(positions, labels)
    ax_bar.set(ylabel="Defender objective", xlabel="Sensor (x, y) [m]",
               title="C  Re-optimization audit (hatched = strict validation failed)")
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(handles=(
        Patch(facecolor="#9e9e9e", label="Screen surrogate"),
        Patch(facecolor="#4c78a8", label="Exact medium"),
        Patch(facecolor="#59a14f", edgecolor="black", label="Continuous valid"),
        Patch(facecolor="#f28e2b", hatch="///", edgecolor="black",
              label="Continuous unverified"),
    ), loc="upper left")

    # D: detection accumulates during both powered and glide phases.
    ax_pod = fig.add_subplot(grid[1, 1])
    clock, cumulative_pod, powered_time = _cumulative_pod(
        (2600.0, 0.0), powered_path, dense_time, dense_states,
    )
    ax_pod.axvspan(0.0, powered_time, color="#d95f02", alpha=0.10)
    ax_pod.axvspan(powered_time, clock[-1], color="#1769aa", alpha=0.08)
    ax_pod.plot(clock, cumulative_pod, color="#6a3d9a", linewidth=2.7)
    ax_pod.axvline(powered_time, color="black", linestyle="--", linewidth=1.1)
    ax_pod.text(powered_time, 0.97 * max(cumulative_pod), " switch",
                va="top", ha="left")
    ax_pod.set(xlabel="Mission time [s]", ylabel="Cumulative PoD [%]",
               title="D  Reintegrated cumulative detection probability")
    ax_pod.grid(alpha=0.25)
    ax_pod.legend(handles=(
        Patch(facecolor="#d95f02", alpha=0.18, label="Powered phase"),
        Patch(facecolor="#1769aa", alpha=0.14, label="Glide phase"),
        Line2D([0], [0], color="#6a3d9a", linewidth=2.7,
               label=f"Final PoD = {100.0 * final['mission_pod']:.3f}%"),
    ), loc="upper left")

    fig.suptitle(
        "Budgeted 2-D Defender Sensor Search — Strictly Validated Selection\n"
        "Selected (2600, 0) m after fine-grid confirmation and physical "
        "scale-1 detection homotopy",
        fontsize=15, fontweight="bold",
    )
    png_path = FIGURE_DIR / "staged_defender_optimization.png"
    pdf_path = FIGURE_DIR / "staged_defender_optimization.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
