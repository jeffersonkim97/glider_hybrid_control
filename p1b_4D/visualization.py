"""Phase 11 publication visualizations from standardized exported data only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_BUNDLES = (
    "geometry", "detection", "projected_cost", "bellman",
    "bellman_response", "stackelberg",
)


def generate_project_visualizations(
    imported_collection: dict[str, Any],
    figure_directory: Path,
) -> dict[str, Any]:
    """Render all possible required figures without calling computational code."""
    if not imported_collection.get("status", {}).get("success", False):
        raise ValueError("imported_collection must pass standardized import validation")
    bundles = imported_collection["primary_result"]["bundles"]
    missing = [name for name in REQUIRED_BUNDLES if name not in bundles]
    required_available = [name for name in REQUIRED_BUNDLES if name != "stackelberg"]
    unavailable_required = [name for name in required_available if name not in bundles]
    if unavailable_required:
        raise ValueError(f"Required visualization bundles missing: {unavailable_required}")
    output_directory = Path(figure_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_config = bundles["geometry"]["manifest"]["configuration"]["plot_config"]
    _apply_style(plot_config)
    generated: list[dict[str, Any]] = []
    generated.append(_export_figure(
        _figure_geometry(bundles, plot_config), "figure_1_geometry_overview",
        output_directory, plot_config,
    ))
    generated.append(_export_figure(
        _figure_projected_cost(bundles, plot_config), "figure_2_projected_cost",
        output_directory, plot_config,
    ))
    generated.append(_export_figure(
        _figure_cost_to_go(bundles, plot_config, False), "figure_3_cost_to_go",
        output_directory, plot_config,
    ))
    generated.append(_export_figure(
        _figure_cost_to_go(bundles, plot_config, True), "figure_4_all_paths",
        output_directory, plot_config,
    ))
    mission_history = None
    if "stackelberg" in bundles:
        generated.append(_export_figure(
            _figure_stackelberg(bundles, plot_config), "figure_5_stackelberg_solution",
            output_directory, plot_config,
        ))
        mission_history = _reconstruct_mission_hazard_history(bundles)
        generated.append(_export_figure(
            _figure_attacker_state_history(mission_history, plot_config),
            "figure_6_attacker_state_history", output_directory, plot_config,
        ))
        generated.append(_export_figure(
            _figure_defender_pod_accumulation(bundles, mission_history, plot_config),
            "figure_7_defender_pod_accumulation", output_directory, plot_config,
        ))
    generated_names = {item["figure_id"] for item in generated}
    expected_names = {
        "figure_1_geometry_overview", "figure_2_projected_cost",
        "figure_3_cost_to_go", "figure_4_all_paths", "figure_5_stackelberg_solution",
        "figure_6_attacker_state_history", "figure_7_defender_pod_accumulation",
    }
    absent_figures = sorted(expected_names - generated_names)
    validation = _validate_visualizations(
        bundles, generated, absent_figures, plot_config, mission_history
    )
    return {
        "primary_result": {
            "generated_figures": tuple(item["figure_id"] for item in generated),
            "exported_figure_paths": tuple(path for item in generated for path in item["paths"]),
            "missing_figures": tuple(absent_figures),
            "missing_input_bundles": tuple(missing),
        },
        "validation": validation,
        "metadata": {
            "schema_name": "VisualizationResult",
            "schema_version": "1.0.0",
            "producer_phase": 11,
            "producer_module": "p1b_4D.visualization",
            "input_policy": "standardized_exports_only",
            "computation_modules_called": False,
            "figure_formats": tuple(plot_config["export_formats"]),
            "dpi": plot_config["dpi"],
            "terrain_drawn_last": True,
        },
        "status": {
            "success": validation["passed"] and not absent_figures,
            "code": "OK" if validation["passed"] and not absent_figures else "VISUALIZATION_INPUT_INCOMPLETE",
            "message": "All required figures generated" if not absent_figures else f"Available figures generated; missing: {absent_figures}",
            "warnings": [f"Figure unavailable: {name}" for name in absent_figures],
            "failed_checks": validation["failed_checks"],
        },
    }


def _figure_geometry(bundles: dict[str, Any], config: dict[str, Any]):
    figure, axis = plt.subplots(figsize=config["default_figure_size"], constrained_layout=True)
    _draw_zones(axis, bundles, config)
    _draw_geometry_markers(axis, bundles, config)
    _draw_terrain_last(axis, bundles, config)
    axis.set_title("Geometry Overview")
    _finish_axis(axis, bundles)
    return figure


def _figure_projected_cost(bundles: dict[str, Any], config: dict[str, Any]):
    figure, axis = plt.subplots(figsize=config["default_figure_size"], constrained_layout=True)
    image = _draw_heatmap(axis, bundles["projected_cost"]["arrays"]["projected_cost"], bundles, config["colormaps"]["projected_cost"], config)
    _draw_zones(axis, bundles, config)
    _draw_geometry_markers(axis, bundles, config)
    _draw_terrain_last(axis, bundles, config)
    figure.colorbar(image, ax=axis, label="Projected local stage cost")
    axis.set_title("Projected Cost Heatmap")
    _finish_axis(axis, bundles)
    return figure


def _figure_cost_to_go(bundles: dict[str, Any], config: dict[str, Any], paths: bool):
    figure, axis = plt.subplots(figsize=config["default_figure_size"], constrained_layout=True)
    image = _draw_heatmap(axis, bundles["bellman"]["arrays"]["pod_to_go"], bundles, config["colormaps"]["projected_cost_to_go"], config, vmin=0.0, vmax=1.0)
    _draw_zones(axis, bundles, config)
    if paths:
        _draw_all_paths(axis, bundles, config)
    _draw_geometry_markers(axis, bundles, config)
    _draw_terrain_last(axis, bundles, config)
    figure.colorbar(image, ax=axis, label="Probability of detection (PoD) cost-to-go")
    axis.set_title(
        "PoD Cost-to-Go with Bellman Candidates and Bellman-optimal Path"
        if paths else "PoD Cost-to-Go Heatmap"
    )
    _finish_axis(axis, bundles)
    return figure


def _figure_stackelberg(bundles: dict[str, Any], config: dict[str, Any]):
    figure, axis = plt.subplots(figsize=config["default_figure_size"], constrained_layout=True)
    stack = bundles["stackelberg"]["arrays"]
    z = stack["final_terrain_z"]
    h_grid = stack["final_terrain_h_grid"]
    extent = _pixel_extent(z, h_grid)
    pod_to_go = np.ma.masked_invalid(np.ma.masked_where(
        ~np.isfinite(stack["final_pod_to_go"]),
        stack["final_pod_to_go"],
    ))
    image = axis.imshow(
        pod_to_go.T, origin="lower", aspect="auto", extent=extent,
        cmap=config["colormaps"]["projected_cost_to_go"],
        alpha=config["heatmap_alpha"], vmin=0.0, vmax=1.0, zorder=1,
    )
    axis.imshow(
        stack["final_los_mask"].T, origin="lower", aspect="auto",
        extent=extent, cmap="Greens", alpha=0.13,
        interpolation="nearest", zorder=4,
    )
    axis.imshow(
        stack["final_occlusion_mask"].T, origin="lower", aspect="auto",
        extent=extent, cmap="Reds", alpha=0.13,
        interpolation="nearest", zorder=5,
    )
    axis.plot([], [], color=config["colors"]["los"], linewidth=6, alpha=0.35, label="LOS zone")
    axis.plot([], [], color=config["colors"]["occlusion"], linewidth=6, alpha=0.35, label="Occlusion zone")
    trajectory = stack["optimal_trajectory"]
    sensor = stack["final_sensor_position"]
    switching = stack["optimal_switching_point"]
    full_path = np.vstack((stack["optimal_powered_path"], trajectory))
    axis.plot(full_path[:, 0], full_path[:, 1], color=config["colors"]["stackelberg"], linewidth=2.8, label="Bellman-optimal path", zorder=40)
    axis.scatter(sensor[0], sensor[1], marker=config["marker_styles"]["defender"], color=config["colors"]["stackelberg"], edgecolor="black", s=90, label="Optimal Defender", zorder=50)
    axis.scatter(switching[0], switching[1], marker=config["marker_styles"]["switching_point"], color=config["colors"]["stackelberg"], s=55, label="Optimal switch", zorder=50)
    axis.plot(z, stack["final_tangent_line_height"], linestyle=config["line_styles"]["los_tangent"], color="dimgray", linewidth=config["line_width"], label="LOS tangent", zorder=30)
    goal = stack["final_goal_position"]
    axis.scatter(goal[0], goal[1], marker=config["marker_styles"]["goal"], color=config["colors"]["goal"], edgecolor="black", s=110, label="Goal", zorder=50)
    terrain = stack["final_terrain_height"]
    terrain_zorder = config["terrain_zorder"]
    axis.fill_between(z, 0.0, terrain, color=config["colors"]["terrain_fill"], zorder=terrain_zorder, label="Terrain")
    axis.plot(z, terrain, color=config["colors"]["terrain_outline"], linestyle=config["line_styles"]["terrain"], linewidth=config["line_width"], zorder=terrain_zorder + 1)
    figure.colorbar(image, ax=axis, label="Probability of detection (PoD) cost-to-go")
    axis.set_title("Final Stackelberg Solution")
    axis.set_xlabel("Horizontal position z [m]")
    axis.set_ylabel("Altitude h [m]")
    axis.set_xlim(z[0], z[-1])
    axis.set_ylim(h_grid[0], h_grid[-1])
    axis.grid(True, linewidth=0.4, alpha=0.25)
    _place_legend_outside(axis)
    return figure


def _reconstruct_mission_hazard_history(bundles: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rebuild z/h/v/gamma and cumulative hazard/PoD vs time for the final path.

    Built entirely from already-exported arrays and the exported configuration
    snapshot -- no computation module (detection.py, bellman.py, ...) is
    called, matching this module's "exported data only" contract. The rate
    formulas below are copied verbatim from detection.py's symbolic graph
    (powered = acoustic only; glide = radar + Doppler, gated by LOS
    visibility) so the reconstructed cumulative hazard matches the
    authoritative mission_pod this bundle already reports.
    """
    stack = bundles["stackelberg"]["arrays"]
    manifest = bundles["stackelberg"]["manifest"]
    detection_cfg = manifest["configuration"]["sensor_config"]["detection"]
    vehicle_cfg = manifest["configuration"]["vehicle_config"]
    attacker_solver_cfg = manifest["configuration"]["attacker_solver_config"]
    transition_model = attacker_solver_cfg["transition_model"]
    time_step = float(vehicle_cfg["time_step"])
    powered_speed = float(vehicle_cfg["powered_speed"])

    powered_path = np.asarray(stack["optimal_powered_path"], dtype=float)
    trajectory = np.asarray(stack["optimal_trajectory"], dtype=float)
    velocity_profile = np.asarray(stack["optimal_velocity_profile"], dtype=float)
    gamma_profile = np.asarray(stack["optimal_gamma_profile"], dtype=float)
    sensor = np.asarray(stack["final_sensor_position"], dtype=float)
    z_grid = np.asarray(stack["final_terrain_z"], dtype=float)
    h_grid = np.asarray(stack["final_terrain_h_grid"], dtype=float)
    los_mask = np.asarray(stack["final_los_mask"], dtype=bool)

    def rate_at(z: float, h: float, v: float, gamma: float, powered: bool) -> float:
        horizontal_range = sensor[0] - z
        vertical_range = sensor[1] - h
        slant_range = float(np.hypot(horizontal_range, vertical_range))
        sensor_range = max(slant_range, float(detection_cfg["range_floor"]))
        if powered:
            acoustic_rate = (
                float(detection_cfg["acoustic_coefficient"])
                * v ** float(detection_cfg["acoustic_speed_exponent"])
                / sensor_range**2
            )
            return float(detection_cfg["acoustic_rate_scale"]) * acoustic_rate
        z_index = int(np.clip(np.searchsorted(z_grid, z), 0, z_grid.size - 1))
        h_index = int(np.clip(np.searchsorted(h_grid, h), 0, h_grid.size - 1))
        if (
            transition_model != "successor_grid_physical_edge"
            and not bool(los_mask[z_index, h_index])
        ):
            return 0.0
        los_angle = np.arctan2(vertical_range, horizontal_range)
        aspect_angle = np.arctan2(np.sin(gamma - los_angle), np.cos(gamma - los_angle))
        rcs = float(detection_cfg["rcs_min"]) + (
            float(detection_cfg["rcs_max"]) - float(detection_cfg["rcs_min"])
        ) * np.cos(aspect_angle) ** 2
        radar_rate = float(detection_cfg["radar_coefficient"]) * rcs / sensor_range**4
        radial_velocity = v * (
            np.cos(gamma) * horizontal_range + np.sin(gamma) * vertical_range
        ) / sensor_range
        doppler_rate = (
            float(detection_cfg["doppler_coefficient"]) * radial_velocity**2 / sensor_range**4
        )
        return (
            float(detection_cfg["radar_rate_scale"]) * radar_rate
            + float(detection_cfg["radial_velocity_rate_scale"]) * doppler_rate
        )

    # Powered phase: same straight-line samples evaluate_powered_segment used,
    # integrated the same way (trapezoid over the sample times).
    powered_delta = powered_path[-1] - powered_path[0]
    powered_distance = float(np.hypot(powered_delta[0], powered_delta[1]))
    powered_time = powered_distance / powered_speed if powered_speed > 0.0 else 0.0
    powered_gamma = float(np.arctan2(powered_delta[1], powered_delta[0]))
    powered_count = powered_path.shape[0]
    t_powered = np.linspace(0.0, powered_time, powered_count)
    rate_powered = np.array([
        rate_at(z, h, powered_speed, powered_gamma, True) for z, h in powered_path
    ])
    hazard_powered = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (rate_powered[:-1] + rate_powered[1:]) * np.diff(t_powered)),
    )) if powered_count > 1 else np.zeros(1)

    # Glide phase: successor-grid edges have nonuniform physical durations.
    # Use the exported duration profile when present; fixed-time legacy
    # exports retain the time_step fallback.
    glide_count = trajectory.shape[0]
    v_glide = np.array([
        velocity_profile[min(i, velocity_profile.size - 1)] for i in range(glide_count)
    ]) if velocity_profile.size else np.full(glide_count, powered_speed)
    gamma_glide = np.array([
        gamma_profile[min(i, gamma_profile.size - 1)] for i in range(glide_count)
    ]) if gamma_profile.size else np.zeros(glide_count)
    edge_count = max(glide_count - 1, 0)
    if "optimal_duration_profile" in stack:
        durations = np.asarray(stack["optimal_duration_profile"], dtype=float)
        if durations.shape != (edge_count,):
            raise ValueError(
                "optimal_duration_profile must contain one duration per glide edge"
            )
        if np.any(~np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("optimal_duration_profile must be positive and finite")
    else:
        durations = np.full(edge_count, time_step, dtype=float)
    t_glide = powered_time + np.concatenate(([0.0], np.cumsum(durations)))
    hazard_glide = np.zeros(glide_count)
    edge_quadrature_count = int(
        attacker_solver_cfg.get("successor_grid", {}).get(
            "edge_quadrature_count", 2
        )
    )
    for index in range(1, glide_count):
        if transition_model == "successor_grid_physical_edge":
            fractions = np.linspace(0.0, 1.0, edge_quadrature_count)
            points = (
                trajectory[index - 1][None, :]
                + fractions[:, None]
                * (trajectory[index] - trajectory[index - 1])[None, :]
            )
            rates = np.asarray([
                rate_at(
                    point[0], point[1], v_glide[index - 1],
                    gamma_glide[index - 1], False,
                )
                for point in points
            ])
            edge_hazard = float(
                np.trapezoid(rates, fractions) * durations[index - 1]
            )
        else:
            rate = rate_at(
                trajectory[index - 1, 0], trajectory[index - 1, 1],
                v_glide[index - 1], gamma_glide[index - 1], False,
            )
            edge_hazard = rate * durations[index - 1]
        hazard_glide[index] = hazard_glide[index - 1] + edge_hazard

    state_times = np.concatenate((t_powered, t_glide[1:]))
    z_history = np.concatenate((powered_path[:, 0], trajectory[1:, 0]))
    h_history = np.concatenate((powered_path[:, 1], trajectory[1:, 1]))
    action_times = np.concatenate(([0.0, powered_time], t_glide))
    v_history = np.concatenate((
        [powered_speed, powered_speed], v_glide,
    ))
    gamma_history = np.concatenate((
        [powered_gamma, powered_gamma], gamma_glide,
    ))
    hazard = np.concatenate((hazard_powered, hazard_powered[-1] + hazard_glide[1:]))

    return {
        "times": state_times,
        "state_times": state_times,
        "action_times": action_times,
        "z": z_history,
        "h": h_history,
        "v": v_history,
        "gamma": gamma_history,
        "cumulative_hazard": hazard,
        "cumulative_pod": 1.0 - np.exp(-hazard),
        "switching_time": powered_time,
        "reconstructed_mission_pod": float(1.0 - np.exp(-hazard[-1])),
    }


def _figure_attacker_state_history(history: dict[str, np.ndarray], config: dict[str, Any]):
    figure, axes = plt.subplots(
        4, 1, figsize=(config["default_figure_size"][0], config["default_figure_size"][1] * 1.4),
        sharex=True, constrained_layout=True,
    )
    panels = (
        (axes[0], history["state_times"], history["z"], "z [m]", None),
        (axes[1], history["state_times"], history["h"], "h [m]", None),
        (axes[2], history["action_times"], history["v"], "v [m/s]", "steps-post"),
        (
            axes[3], history["action_times"], np.degrees(history["gamma"]),
            "gamma [deg]", "steps-post",
        ),
    )
    for axis, times, values, label, drawstyle in panels:
        axis.plot(
            times, values, color=config["colors"]["stackelberg"],
            drawstyle=drawstyle or "default",
            linewidth=config["line_width"] + 0.4,
            label="Bellman-optimal attacker state",
        )
        axis.axvline(
            history["switching_time"], color="dimgray", linestyle="--",
            linewidth=1.2, label="Powered to glide switch",
        )
        axis.set_ylabel(label)
        axis.grid(True, linewidth=0.4, alpha=0.25)
    axes[-1].set_xlabel("Mission time [s]")
    axes[0].set_title("Attacker Optimal Strategy: State History")
    _place_legend_outside(axes[-1], ncol=2, y_offset=-0.32)
    return figure


def _figure_defender_pod_accumulation(
    bundles: dict[str, Any], history: dict[str, np.ndarray], config: dict[str, Any],
):
    stack = bundles["stackelberg"]["arrays"]
    z_grid = np.asarray(stack["final_terrain_z"], dtype=float)
    h_grid = np.asarray(stack["final_terrain_h_grid"], dtype=float)
    cell_area = float(np.diff(z_grid)[0] * np.diff(h_grid)[0])
    admissible_mask = ~np.asarray(stack["coverage_terrain_mask"], dtype=bool)
    coverage_mask = np.asarray(stack["coverage_los_mask"], dtype=bool)
    admissible_area = float(np.count_nonzero(admissible_mask)) * cell_area
    coverage_area = float(np.count_nonzero(coverage_mask)) * cell_area
    coverage_normalized = coverage_area / admissible_area if admissible_area > 0.0 else 0.0

    figure, (axis_pod, axis_coverage) = plt.subplots(
        2, 1, figsize=config["default_figure_size"],
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True,
    )
    axis_pod.plot(
        history["times"], history["cumulative_pod"],
        color=config["colors"]["stackelberg"], linewidth=config["line_width"] + 0.4,
        label="Cumulative probability of detection",
    )
    axis_pod.axvline(
        history["switching_time"], color="dimgray", linestyle="--",
        linewidth=1.2, label="Powered to glide switch",
    )
    axis_pod.set_ylabel("Cumulative PoD")
    axis_pod.set_ylim(0.0, 1.0)
    axis_pod.grid(True, linewidth=0.4, alpha=0.25)
    axis_pod.set_title("Defender Optimal Strategy: PoD Accumulation and Coverage")

    axis_coverage.barh(
        [0], [coverage_normalized], color=config["colors"]["stackelberg"],
        label="Normalized coverage area",
    )
    axis_coverage.set_xlim(0.0, 1.0)
    axis_coverage.set_yticks([])
    axis_coverage.set_xlabel("Normalized coverage area (fraction of admissible airspace)")
    axis_coverage.text(
        min(coverage_normalized + 0.02, 0.7), 0, f"{coverage_area:,.0f} m^2",
        va="center", fontsize=config["font_size"] - 1,
    )
    axis_coverage.grid(True, axis="x", linewidth=0.4, alpha=0.25)

    axis_pod.set_xlabel("Mission time [s]")
    _place_legend_outside(
        axis_coverage, ncol=2, y_offset=-0.55, source_axes=(axis_pod, axis_coverage),
    )
    return figure


def _pixel_extent(z, h):
    """Outer image bounds for imshow given cell-center grid coordinates.

    `z`/`h` are linspace-sampled cell centers, but `imshow`'s `extent` is the
    outer edge of the pixel grid. Passing cell centers directly shifts every
    pixel center half a cell away from its true coordinate, misaligning the
    heatmap against anything else plotted at exact grid coordinates (e.g. the
    Bellman-optimal trajectory).
    """
    dz = (z[1] - z[0]) / 2.0 if z.size > 1 else 0.0
    dh = (h[1] - h[0]) / 2.0 if h.size > 1 else 0.0
    return (z[0] - dz, z[-1] + dz, h[0] - dh, h[-1] + dh)


def _draw_heatmap(axis, values, bundles, cmap, config, vmin=None, vmax=None):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    h = geometry["terrain_h_grid"]
    masked = np.ma.masked_invalid(np.ma.masked_where(~np.isfinite(values), values))
    return axis.imshow(masked.T, origin="lower", aspect="auto", extent=_pixel_extent(z, h), cmap=cmap, alpha=config["heatmap_alpha"], vmin=vmin, vmax=vmax, zorder=1)


def _draw_zones(axis, bundles, config):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    h = geometry["terrain_h_grid"]
    extent = _pixel_extent(z, h)
    axis.imshow(geometry["los_los_mask"].T, origin="lower", aspect="auto", extent=extent, cmap="Greens", alpha=0.13, interpolation="nearest", zorder=4)
    axis.imshow(geometry["los_occlusion_mask"].T, origin="lower", aspect="auto", extent=extent, cmap="Reds", alpha=0.13, interpolation="nearest", zorder=5)
    axis.plot([], [], color=config["colors"]["los"], linewidth=6, alpha=0.35, label="LOS zone")
    axis.plot([], [], color=config["colors"]["occlusion"], linewidth=6, alpha=0.35, label="Occlusion zone")


def _draw_geometry_markers(axis, bundles, config, sensor=True):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    tangent = geometry["tangent_line_height"]
    axis.plot(z, tangent, linestyle=config["line_styles"]["los_tangent"], color="dimgray", linewidth=config["line_width"], label="LOS tangent", zorder=30)
    if sensor:
        point = geometry["sensor_position"]
        axis.scatter(point[0], point[1], marker=config["marker_styles"]["sensor"], color=config["colors"]["sensor"], s=70, label="Sensor", zorder=50)
    goal = geometry["goal_position"]
    axis.scatter(goal[0], goal[1], marker=config["marker_styles"]["goal"], color=config["colors"]["goal"], edgecolor="black", s=110, label="Goal", zorder=50)


def _draw_all_paths(axis, bundles, config):
    bellman = bundles["bellman"]["arrays"]
    glide_trajectories = _unpack(bellman["trajectory_points"], bellman["trajectory_offsets"])
    powered_paths = _unpack(bellman["powered_path_points"], bellman["powered_path_offsets"])
    for index, (powered_path, trajectory) in enumerate(zip(powered_paths, glide_trajectories)):
        full_path = np.vstack((powered_path, trajectory))
        axis.plot(full_path[:, 0], full_path[:, 1], color=config["colors"]["bellman"], linestyle=config["line_styles"]["bellman"], linewidth=1.0, alpha=0.55, label="Bellman candidates" if index == 0 else None, zorder=20)
    response = bundles["bellman_response"]["arrays"]
    response_full_path = np.vstack((response["powered_path"], response["trajectory"]))
    axis.plot(response_full_path[:, 0], response_full_path[:, 1], color=config["colors"]["stackelberg"], linestyle="-", linewidth=config["line_width"] + 0.6, alpha=0.95, label="Bellman-optimal response", zorder=25)
    switches = np.vstack((bellman["switching_points"], response["switching_point"].reshape(1, 2)))
    axis.scatter(switches[:, 0], switches[:, 1], marker=config["marker_styles"]["switching_point"], facecolors="none", edgecolors="black", s=30, label="Switching points", zorder=35)


def _draw_terrain_last(axis, bundles, config):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    terrain = geometry["terrain_height"]
    zorder = config["terrain_zorder"]
    axis.fill_between(z, 0.0, terrain, color=config["colors"]["terrain_fill"], zorder=zorder, label="Terrain")
    axis.plot(z, terrain, color=config["colors"]["terrain_outline"], linestyle=config["line_styles"]["terrain"], linewidth=config["line_width"], zorder=zorder + 1)


def _finish_axis(axis, bundles):
    axis.set_xlabel("Horizontal position z [m]")
    axis.set_ylabel("Altitude h [m]")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, bundles["geometry"]["arrays"]["terrain_h_grid"][-1])
    axis.grid(True, linewidth=0.4, alpha=0.25)
    _place_legend_outside(axis)


def _place_legend_outside(axis, ncol=3, y_offset=-0.14, source_axes=None):
    combined_handles: list = []
    combined_labels: list = []
    for source in (source_axes or (axis,)):
        handles, labels = source.get_legend_handles_labels()
        combined_handles.extend(handles)
        combined_labels.extend(labels)
    unique = dict(zip(combined_labels, combined_handles))
    axis.legend(
        unique.values(), unique.keys(),
        loc="upper left", bbox_to_anchor=(0.0, y_offset),
        ncol=ncol, frameon=True,
    )


def _export_figure(figure, figure_id, directory, config):
    paths = []
    for extension in config["export_formats"]:
        path = directory / f"{figure_id}.{extension}"
        figure.savefig(path, dpi=config["dpi"], bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return {"figure_id": figure_id, "paths": tuple(paths)}


def _unpack(values, offsets):
    return tuple(values[offsets[index]:offsets[index + 1]] for index in range(offsets.size - 1))


def _apply_style(config):
    plt.rcParams.update({"font.family": config["font_family"], "font.size": config["font_size"], "axes.linewidth": 0.8, "lines.linewidth": config["line_width"], "savefig.dpi": config["dpi"]})


def _validate_visualizations(
    bundles, generated, missing_figures, config, mission_history=None,
):
    geometry = bundles["geometry"]["arrays"]
    shape = geometry["los_los_mask"].shape
    bellman_pod_to_go = bundles["bellman"]["arrays"]["pod_to_go"]
    dimensions_consistent = (
        geometry["los_occlusion_mask"].shape == shape
        and bundles["projected_cost"]["arrays"]["projected_cost"].shape == shape
        and bundles["bellman"]["arrays"]["cost_to_go"].shape == shape
        and bellman_pod_to_go.shape == shape
    )
    finite_pod = bellman_pod_to_go[np.isfinite(bellman_pod_to_go)]
    pod_to_go_bounded = bool(
        finite_pod.size == 0 or np.all((finite_pod >= 0.0) & (finite_pod <= 1.0))
    )
    paths = [path for item in generated for path in item["paths"]]
    stack = bundles.get("stackelberg", {}).get("arrays")
    stack_required = (
        "final_cost_to_go", "final_pod_to_go", "final_terrain_z", "final_terrain_height",
        "final_terrain_h_grid", "final_terrain_mask", "final_los_mask",
        "final_occlusion_mask", "final_sensor_position",
        "final_goal_position", "final_tangent_line_height",
    )
    stack_arrays_present = stack is None or all(
        key in stack for key in stack_required
    )
    stack_dimensions_consistent = stack is None or (
        stack_arrays_present
        and stack["final_cost_to_go"].shape
        == stack["final_pod_to_go"].shape
        == stack["final_los_mask"].shape
        == stack["final_occlusion_mask"].shape
        == stack["final_terrain_mask"].shape
    )
    stack_sensor_consistent = stack is None or np.allclose(
        stack["final_sensor_position"], stack["optimal_sensor_position"],
        rtol=0.0, atol=1.0e-9,
    )
    reconstructed_pod_consistent = True
    if stack is not None and mission_history is not None:
        authoritative_pod = float(
            bundles["stackelberg"]["manifest"]["objective_values"][
                "mission_pod"
            ]
        )
        reconstructed_pod_consistent = bool(np.isclose(
            mission_history["reconstructed_mission_pod"],
            authoritative_pod,
            rtol=0.0,
            atol=1.0e-9,
        ))
    checks = {
        "available_bundles_loaded": all(name in bundles for name in REQUIRED_BUNDLES if name != "stackelberg"),
        "required_arrays_present": all(key in geometry for key in ("terrain_z", "terrain_height", "terrain_h_grid", "los_los_mask", "los_occlusion_mask", "sensor_position", "goal_position", "tangent_line_height")),
        "dimensions_consistent": dimensions_consistent,
        "generated_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in paths),
        "format_count_consistent": len(paths) == len(generated) * len(config["export_formats"]),
        "figure5_final_arrays_present": stack_arrays_present,
        "figure5_final_dimensions_consistent": stack_dimensions_consistent,
        "figure5_sensor_consistent": stack_sensor_consistent,
        "time_history_pod_matches_authoritative": reconstructed_pod_consistent,
        "pod_to_go_bounded_unit_interval": pod_to_go_bounded,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "metrics": {"generated_figure_count": len(generated), "exported_file_count": len(paths), "missing_figure_count": len(missing_figures)}, "warnings": [f"Not generated: {name}" for name in missing_figures], "failed_checks": failed, "summary": "Available visualization validation passed" if not failed else f"Visualization validation failed: {failed}"}
