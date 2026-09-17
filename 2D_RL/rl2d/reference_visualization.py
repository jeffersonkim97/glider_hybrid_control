"""Visual verification report for the exact 2D Bellman reference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def _pixel_extent(z: np.ndarray, h: np.ndarray) -> tuple[float, ...]:
    dz = float(z[1] - z[0]) / 2.0
    dh = float(h[1] - h[0]) / 2.0
    return (float(z[0] - dz), float(z[-1] + dz), float(h[0] - dh), float(h[-1] + dh))


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_path(axis: plt.Axes, data: Any, heatmap: bool = False) -> None:
    powered = np.asarray(data["powered_path"])
    glide = np.asarray(data["optimal_glide_trajectory"])
    switch = np.asarray(data["selected_switching_point"])
    if not heatmap:
        axis.plot(
            powered[:, 0], powered[:, 1], color="#e67e22", linewidth=2.4,
            label="Powered phase", zorder=20,
        )
    axis.plot(
        glide[:, 0], glide[:, 1], color="#1565c0", linewidth=2.4,
        label="Bellman-optimal glide", zorder=21,
    )
    axis.scatter(
        switch[0], switch[1], s=55, marker="o", facecolor="white",
        edgecolor="black", linewidth=1.2, label="Selected switch", zorder=25,
    )


def _draw_geometry(axis: plt.Axes, data: Any) -> None:
    z = np.asarray(data["z_grid"])
    h = np.asarray(data["h_grid"])
    extent = _pixel_extent(z, h)
    terrain_z = np.asarray(data["terrain_z"])
    terrain_height = np.asarray(data["terrain_height"])
    los = np.asarray(data["spatial_los_valid_mask"], dtype=bool)
    terrain = np.asarray(data["spatial_terrain_mask"], dtype=bool)
    non_visible = np.asarray(data["spatial_occlusion_mask"], dtype=bool) & ~terrain

    zone_code = np.zeros(los.shape, dtype=np.uint8)
    zone_code[los] = 1
    zone_code[non_visible] = 2
    axis.imshow(
        zone_code.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=ListedColormap(["white", "#e8f5e9", "#e3f2fd"]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
        alpha=0.9,
        zorder=1,
    )
    axis.fill_between(
        terrain_z, 0.0, terrain_height, color="#595959", alpha=0.95,
        label="Terrain", zorder=10,
    )
    axis.plot(
        z,
        np.asarray(data["los_tangent_line_height"]),
        color="#616161",
        linestyle="--",
        linewidth=1.4,
        label="LOS tangent line",
        zorder=12,
    )
    _draw_path(axis, data)
    sensor = np.asarray(data["sensor_position"])
    goal = np.asarray(data["goal_position"])
    axis.scatter(
        sensor[0], sensor[1], marker="^", s=90, color="#c62828",
        edgecolor="black", linewidth=0.7, label="Sensor", zorder=30,
    )
    axis.scatter(
        goal[0], goal[1], marker="*", s=140, color="#fdd835",
        edgecolor="black", linewidth=0.7, label="Goal", zorder=30,
    )
    axis.plot([], [], color="#81c784", linewidth=7, alpha=0.45, label="LOS zone")
    axis.plot([], [], color="#90caf9", linewidth=7, alpha=0.45, label="Occlusion zone")
    axis.set_title("A. Geometry and selected mission path")
    axis.set_ylabel("Altitude h [m]")
    axis.legend(loc="upper right", ncol=2, fontsize=7.4, framealpha=0.95)


def _draw_policy_map(
    axis: plt.Axes,
    data: Any,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    z = np.asarray(data["z_grid"])
    h = np.asarray(data["h_grid"])
    masked = np.ma.masked_invalid(values)
    image = axis.imshow(
        masked.T,
        origin="lower",
        aspect="auto",
        extent=_pixel_extent(z, h),
        cmap=cmap,
        interpolation="nearest",
    )
    _draw_path(axis, data, heatmap=True)
    axis.set_title(title)
    bar = axis.figure.colorbar(image, ax=axis, pad=0.012, fraction=0.045)
    bar.set_label(colorbar_label)


def create_reference_verification_figure(
    artifact_path: Path,
    summary_path: Path,
    output_path: Path,
    dpi: int = 350,
) -> Path:
    """Create one self-contained visual and numerical verification sheet."""

    artifact_path = Path(artifact_path).resolve()
    summary_path = Path(summary_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(summary_path)

    with np.load(artifact_path) as data:
        value = np.asarray(data["exact_value"])
        policy = np.asarray(data["exact_policy_action_index"])
        gamma_actions = np.rad2deg(np.asarray(data["action_gamma"]))
        speed_actions = np.asarray(data["action_speed"])
        gamma_policy = np.full(policy.shape, np.nan)
        speed_policy = np.full(policy.shape, np.nan)
        selected = policy >= 0
        gamma_policy[selected] = gamma_actions[policy[selected]]
        speed_policy[selected] = speed_actions[policy[selected]]

        figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.3), constrained_layout=True)
        _draw_geometry(axes[0, 0], data)
        _draw_policy_map(
            axes[0, 1], data,
            np.where(np.isfinite(value), value, np.nan),
            "B. Exact Bellman cost-to-go V*(z,h)",
            "Normalized mission cost-to-go",
            "viridis",
        )
        _draw_policy_map(
            axes[1, 0], data, gamma_policy,
            "C. Greedy physical-edge flight-path angle",
            "γ [deg]",
            "coolwarm",
        )
        _draw_policy_map(
            axes[1, 1], data, speed_policy,
            "D. Greedy physical-edge speed",
            "Speed [m/s]",
            "plasma",
        )

    for axis in axes.flat:
        axis.set_xlim(float(summary["configuration_snapshot"]["environment_config"]["grid"]["z_min"]),
                      float(summary["configuration_snapshot"]["environment_config"]["grid"]["z_max"]))
        axis.set_ylim(float(summary["configuration_snapshot"]["environment_config"]["grid"]["h_min"]),
                      float(summary["configuration_snapshot"]["environment_config"]["grid"]["h_max"]))
        axis.set_xlabel("Downrange z [m]")
        axis.grid(color="white", alpha=0.18, linewidth=0.5)

    mission = summary["mission"]
    validation = summary["validation"]["metrics"]
    replay = summary["continuous_replay"]
    grid = summary["grid"]
    switch = mission["selected_switching_point"]
    metrics = (
        f"Grid {grid['z_count']}×{grid['h_count']}  |  actions {grid['action_count']}  |  "
        f"finite states {validation['finite_value_state_count']:,}\n"
        f"Switch ({switch[0]:.2f}, {switch[1]:.2f}) m  |  "
        f"J={mission['mission_objective']:.6f}  |  PoD={mission['mission_pod']:.6f}  |  "
        f"time={mission['mission_time']:.3f} s\n"
        f"Production/export ΔVmax={validation['production_export_value_maximum_difference']:.1e}  |  "
        f"Bellman residual={validation['maximum_bellman_residual']:.1e}  |  "
        f"continuous replay={'PASS' if replay['feasible'] and replay['reached_goal'] else 'FAIL'}  |  "
        f"goal miss={replay['goal_miss']:.6f} m"
    )
    figure.suptitle(
        "Exact 2D Bellman Reference — Numerical and Visual Verification\n" + metrics,
        fontsize=12.0,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path
