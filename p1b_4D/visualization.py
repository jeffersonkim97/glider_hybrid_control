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
    "filtered_bellman", "attacker_nlp", "stackelberg",
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
    if "stackelberg" in bundles:
        generated.append(_export_figure(
            _figure_stackelberg(bundles, plot_config), "figure_5_stackelberg_solution",
            output_directory, plot_config,
        ))
    generated_names = {item["figure_id"] for item in generated}
    expected_names = {
        "figure_1_geometry_overview", "figure_2_projected_cost",
        "figure_3_cost_to_go", "figure_4_all_paths", "figure_5_stackelberg_solution",
    }
    absent_figures = sorted(expected_names - generated_names)
    validation = _validate_visualizations(bundles, generated, absent_figures, plot_config)
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
    image = _draw_heatmap(axis, bundles["bellman"]["arrays"]["cost_to_go"], bundles, config["colormaps"]["projected_cost_to_go"], config)
    _draw_zones(axis, bundles, config)
    if paths:
        _draw_all_paths(axis, bundles, config)
    _draw_geometry_markers(axis, bundles, config)
    _draw_terrain_last(axis, bundles, config)
    figure.colorbar(image, ax=axis, label="Bellman cost-to-go")
    axis.set_title("Cost-to-Go with Bellman and NLP Paths" if paths else "Projected Cost-to-Go Heatmap")
    _finish_axis(axis, bundles)
    return figure


def _figure_stackelberg(bundles: dict[str, Any], config: dict[str, Any]):
    figure, axis = plt.subplots(figsize=config["default_figure_size"], constrained_layout=True)
    stack = bundles["stackelberg"]["arrays"]
    z = stack["final_terrain_z"]
    h_grid = stack["final_terrain_h_grid"]
    extent = (z[0], z[-1], h_grid[0], h_grid[-1])
    cost_to_go = np.ma.masked_invalid(np.ma.masked_where(
        ~np.isfinite(stack["final_cost_to_go"]),
        stack["final_cost_to_go"],
    ))
    image = axis.imshow(
        cost_to_go.T, origin="lower", aspect="auto", extent=extent,
        cmap=config["colormaps"]["projected_cost_to_go"],
        alpha=config["heatmap_alpha"], zorder=1,
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
    axis.plot(trajectory[:, 0], trajectory[:, 1], color=config["colors"]["stackelberg"], linewidth=2.8, label="Optimal NLP path", zorder=40)
    axis.scatter(sensor[0], sensor[1], marker=config["marker_styles"]["defender"], color=config["colors"]["stackelberg"], edgecolor="black", s=90, label="Optimal Defender", zorder=50)
    axis.scatter(switching[0], switching[1], marker=config["marker_styles"]["switching_point"], color=config["colors"]["stackelberg"], s=55, label="Optimal switch", zorder=50)
    axis.plot(z, stack["final_tangent_line_height"], linestyle=config["line_styles"]["los_tangent"], color="dimgray", linewidth=config["line_width"], label="LOS tangent", zorder=30)
    goal = stack["final_goal_position"]
    axis.scatter(goal[0], goal[1], marker=config["marker_styles"]["goal"], color=config["colors"]["goal"], edgecolor="black", s=110, label="Goal", zorder=50)
    terrain = stack["final_terrain_height"]
    terrain_zorder = config["terrain_zorder"]
    axis.fill_between(z, 0.0, terrain, color=config["colors"]["terrain_fill"], zorder=terrain_zorder, label="Terrain")
    axis.plot(z, terrain, color=config["colors"]["terrain_outline"], linestyle=config["line_styles"]["terrain"], linewidth=config["line_width"], zorder=terrain_zorder + 1)
    figure.colorbar(image, ax=axis, label="Bellman cost-to-go")
    axis.set_title("Final Stackelberg Solution")
    axis.set_xlabel("Horizontal position z [m]")
    axis.set_ylabel("Altitude h [m]")
    axis.set_xlim(z[0], z[-1])
    axis.set_ylim(h_grid[0], h_grid[-1])
    axis.grid(True, linewidth=0.4, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", frameon=True)
    return figure


def _draw_heatmap(axis, values, bundles, cmap, config):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    h = geometry["terrain_h_grid"]
    masked = np.ma.masked_invalid(np.ma.masked_where(~np.isfinite(values), values))
    return axis.imshow(masked.T, origin="lower", aspect="auto", extent=(z[0], z[-1], h[0], h[-1]), cmap=cmap, alpha=config["heatmap_alpha"], zorder=1)


def _draw_zones(axis, bundles, config):
    geometry = bundles["geometry"]["arrays"]
    z = geometry["terrain_z"]
    h = geometry["terrain_h_grid"]
    extent = (z[0], z[-1], h[0], h[-1])
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
    for index, trajectory in enumerate(_unpack(bellman["trajectory_points"], bellman["trajectory_offsets"])):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=config["colors"]["bellman"], linestyle=config["line_styles"]["bellman"], linewidth=1.0, alpha=0.55, label="Bellman candidates" if index == 0 else None, zorder=20)
    nlp = bundles["attacker_nlp"]["arrays"]
    for index, trajectory in enumerate(_unpack(nlp["trajectory_points"], nlp["trajectory_offsets"])):
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=config["colors"]["nlp"], linestyle=config["line_styles"]["nlp"], linewidth=config["line_width"], alpha=0.85, label="NLP refinements" if index == 0 else None, zorder=25)
    switches = np.vstack((bellman["switching_points"], nlp["switching_points"]))
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
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="best", frameon=True)


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


def _validate_visualizations(bundles, generated, missing_figures, config):
    geometry = bundles["geometry"]["arrays"]
    shape = geometry["los_los_mask"].shape
    dimensions_consistent = (
        geometry["los_occlusion_mask"].shape == shape
        and bundles["projected_cost"]["arrays"]["projected_cost"].shape == shape
        and bundles["bellman"]["arrays"]["cost_to_go"].shape == shape
    )
    paths = [path for item in generated for path in item["paths"]]
    stack = bundles.get("stackelberg", {}).get("arrays")
    stack_required = (
        "final_cost_to_go", "final_terrain_z", "final_terrain_height",
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
        == stack["final_los_mask"].shape
        == stack["final_occlusion_mask"].shape
        == stack["final_terrain_mask"].shape
    )
    stack_sensor_consistent = stack is None or np.allclose(
        stack["final_sensor_position"], stack["optimal_sensor_position"],
        rtol=0.0, atol=1.0e-9,
    )
    checks = {
        "available_bundles_loaded": all(name in bundles for name in REQUIRED_BUNDLES if name != "stackelberg"),
        "required_arrays_present": all(key in geometry for key in ("terrain_z", "terrain_height", "terrain_h_grid", "los_los_mask", "los_occlusion_mask", "sensor_position", "goal_position", "tangent_line_height")),
        "dimensions_consistent": dimensions_consistent,
        "generated_files_exist": all(path.is_file() and path.stat().st_size > 0 for path in paths),
        "format_count_consistent": len(paths) == len(generated) * len(config["export_formats"]),
        "figure5_final_arrays_present": stack_arrays_present,
        "figure5_final_dimensions_consistent": stack_dimensions_consistent,
        "figure5_sensor_consistent": stack_sensor_consistent,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "metrics": {"generated_figure_count": len(generated), "exported_file_count": len(paths), "missing_figure_count": len(missing_figures)}, "warnings": [f"Not generated: {name}" for name in missing_figures], "failed_checks": failed, "summary": "Available visualization validation passed" if not failed else f"Visualization validation failed: {failed}"}
