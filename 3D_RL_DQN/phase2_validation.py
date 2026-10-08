"""Phase 16.2 observation validation, manifest, and development figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths
from P1b_Exact_Local_SSE import exact_best_response
from P1b_RL_approximation import default_sensor
from P1b_condition import ComputationCondition, build_scene
from attacker_br_problem import AttackerBRProblem
from terrain_observation import (
    EGO_FEATURE_NAMES,
    GOAL_FEATURE_NAMES,
    HAZARD_CHANNEL_NAMES,
    ObservationConfig,
    TERRAIN_CHANNEL_NAMES,
    TerrainObservationBuilder,
    VALIDITY_CHANNEL_NAMES,
)


PHASE0_DIR = project_paths.CORE_DIR / "figure" / "phase_0_baseline"
PHASE1_DIR = project_paths.CORE_DIR / "figure" / "phase_1_attacker_br_interface"
OUTPUT_DIR = project_paths.CORE_DIR / "figure" / "phase_2_terrain_observation"
MANIFEST_NAME = "phase2_terrain_observation_manifest.json"
TOLERANCE = 1.0e-12


def _relative(path: Path) -> str:
    return path.resolve().relative_to(project_paths.WORKSPACE_ROOT.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_from_phase0(manifest: dict[str, Any]) -> ComputationCondition:
    stored = manifest["canonical_cube_case"]["configuration"]["condition"]
    return ComputationCondition(
        spatial_resolution_m=float(stored["spatial_resolution_m"]),
        heading_spacing_deg=float(stored["heading_spacing_deg"]),
        r_neighbor=int(stored["r_neighbor"]),
        terrain_category=str(stored["terrain_category"]),
        hazard_weight=stored.get("hazard_weight"),
        time_weight=stored.get("time_weight"),
        defender_goal_margin_map=float(stored["defender_goal_margin_map"]),
    )


def _save(figure: Any, output: Path) -> str:
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return _relative(output)


def _local_extent(config: ObservationConfig) -> tuple[float, float, float, float]:
    half = 0.5 * config.local_window_extent_m
    return (-half, half, -half, half)


def _imshow_local(ax: Any, values: np.ndarray, config: ObservationConfig, **kwargs: Any) -> Any:
    image = ax.imshow(
        values, origin="upper", extent=_local_extent(config), aspect="equal", **kwargs,
    )
    ax.set_xlabel("right offset [m]")
    ax.set_ylabel("forward offset [m]")
    return image


def _add_terrain_top(ax: Any, problem: AttackerBRProblem) -> None:
    for box in problem.scene.terrain.obstacle_boxes():
        ax.add_patch(plt.Rectangle(
            (box.x_limits[0], box.y_limits[0]), box.width_x, box.width_y,
            facecolor="#bdbdbd", edgecolor="#333333", alpha=0.65,
        ))
    bounds = problem.grid.bounds
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("global x [map unit]")
    ax.set_ylabel("global y [map unit]")


def figure_global_to_local(
    problem: AttackerBRProblem,
    builder: TerrainObservationBuilder,
    state_id: int,
    output: Path,
) -> str:
    position = problem.position_map(state_id)
    state = problem.state(state_id)
    heading = problem.grid.heading_rad(state.heading_bin)
    sample_map = builder.sample_positions_map(state_id)
    corners = sample_map[[0, 0, -1, -1, 0], [0, -1, -1, 0, 0], :2]
    goal = problem.scene.config.goal.as_array()
    scale = problem.scene.config.physical_scale.meters_per_map_unit

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    _add_terrain_top(ax, problem)
    bounds = problem.grid.bounds
    ax.add_patch(plt.Rectangle(
        (bounds.x_min, bounds.y_min), bounds.x_max - bounds.x_min,
        bounds.y_max - bounds.y_min, fill=False, edgecolor="#555555",
        linestyle="--", linewidth=1.2, label="attacker domain",
    ))
    ax.plot(corners[:, 0], corners[:, 1], color="#0072B2", linewidth=2.0, label="2 km local window")
    ax.scatter(position[0], position[1], color="#D55E00", s=55, zorder=4, label="attacker")
    ax.scatter(goal[0], goal[1], marker="*", color="#009E73", s=130, zorder=4, label="goal")
    ax.arrow(
        position[0], position[1], 2.0 * np.cos(heading), 2.0 * np.sin(heading),
        width=0.04, head_width=0.32, color="#D55E00", length_includes_head=True,
    )
    x_values = np.concatenate((corners[:, 0], [bounds.x_min, bounds.x_max]))
    y_values = np.concatenate((corners[:, 1], [bounds.y_min, bounds.y_max]))
    margin = 0.5
    ax.set_xlim(float(x_values.min() - margin), float(x_values.max() + margin))
    ax.set_ylim(float(y_values.min() - margin), float(y_values.max() + margin))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Global footprint: state {state_id}")

    forward_m, right_m = builder.local_offsets_m()
    axes[1].scatter(right_m.ravel(), forward_m.ravel(), s=7, color="#0072B2", alpha=0.55)
    axes[1].scatter(0.0, 0.0, color="#D55E00", s=55, label="attacker")
    delta_goal_m = problem.scene.config.physical_scale.position_m(goal - position)
    forward = np.asarray([np.cos(heading), np.sin(heading)])
    right = np.asarray([np.sin(heading), -np.cos(heading)])
    axes[1].scatter(
        np.dot(delta_goal_m[:2], right), np.dot(delta_goal_m[:2], forward),
        marker="*", color="#009E73", s=130, label="goal (may lie outside window)",
    )
    half = 0.5 * builder.config.local_window_extent_m
    axes[1].set_xlim(-half, half)
    axes[1].set_ylim(-half, half)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("right offset [m]")
    axes[1].set_ylabel("forward offset [m]")
    axes[1].set_title(f"Heading-aligned local samples ({builder.config.sample_spacing_m[0]:.0f} m spacing)")
    axes[1].legend(fontsize=8)
    figure.suptitle(f"Global-to-local observation geometry (1 map unit = {scale:.0f} m)")
    return _save(figure, output)


def figure_terrain_channel(
    observation: Any, builder: TerrainObservationBuilder, output: Path,
) -> str:
    values = observation.terrain_channels[0]
    valid_values = values[np.isfinite(values)]
    figure, ax = plt.subplots(figsize=(7.2, 5.8))
    image = _imshow_local(ax, values, builder.config, cmap="terrain")
    figure.colorbar(image, ax=ax, label="attacker-to-surface clearance [m]")
    ax.set_title(
        f"{TERRAIN_CHANNEL_NAMES[0]}\nvalid min={valid_values.min():.1f} m, max={valid_values.max():.1f} m"
    )
    return _save(figure, output)


def figure_hazard_channels(
    observation: Any, builder: TerrainObservationBuilder, output: Path,
) -> str:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    notes = (
        "LOS gate [0/1]", "sensor LOS projected on ego forward [-1,1]",
        "sensor LOS projected on ego right [-1,1]", "sensor LOS vertical component [-1,1]",
        "(range floor / sensor range)^4 [0,1]",
    )
    for index, (name, note) in enumerate(zip(HAZARD_CHANNEL_NAMES, notes)):
        ax = axes.flat[index]
        values = observation.hazard_channels[index]
        finite = values[np.isfinite(values)]
        image = _imshow_local(ax, values, builder.config, cmap="viridis")
        figure.colorbar(image, ax=ax, fraction=0.046)
        ax.set_title(f"{name}\n{note}\nmin={finite.min():.3g}, max={finite.max():.3g}", fontsize=9)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05, 0.82,
        "These five channels preserve the current\nposition/LOS/range terms. Candidate action\nvelocity supplies speed and radial velocity\nto reconstruct instantaneous hazard exactly.",
        va="top", fontsize=11,
    )
    figure.suptitle("Local authoritative hazard sufficient statistics")
    return _save(figure, output)


def figure_validity_mask(
    observation: Any, builder: TerrainObservationBuilder, state_id: int, output: Path,
) -> str:
    mask = observation.validity_channels[0]
    figure, ax = plt.subplots(figsize=(7.0, 5.8))
    image = _imshow_local(ax, mask, builder.config, cmap="gray_r", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=ax, ticks=(0, 1), label="0 outside domain, 1 valid")
    ax.set_title(
        f"Explicit boundary validity — state {state_id}\n"
        f"valid={int(mask.sum())}, padded={int(mask.size - mask.sum())} cells"
    )
    return _save(figure, output)


def figure_component_summary(observation: Any, builder: TerrainObservationBuilder, output: Path) -> str:
    tensor = builder.to_tensor_ready(observation)
    rows = [
        ("ego", ", ".join(EGO_FEATURE_NAMES), str(observation.ego_features.shape)),
        ("goal", ", ".join(GOAL_FEATURE_NAMES), str(observation.goal_features.shape)),
        ("terrain", ", ".join(TERRAIN_CHANNEL_NAMES), str(observation.terrain_channels.shape)),
        ("hazard", ", ".join(HAZARD_CHANNEL_NAMES), str(observation.hazard_channels.shape)),
        ("validity", ", ".join(VALIDITY_CHANNEL_NAMES), str(observation.validity_channels.shape)),
        ("tensor scalar", "ego + goal", str(tensor.scalar.shape)),
        ("tensor spatial", "terrain + hazard + validity", str(tensor.spatial.shape)),
    ]
    figure, ax = plt.subplots(figsize=(13.2, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=("component", "ordered features/channels", "shape"),
        cellLoc="left", colLoc="left", loc="center", colWidths=(0.15, 0.68, 0.17),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    ax.set_title("Phase 16.2 structured observation and tensor-ready contract", pad=18)
    return _save(figure, output)


def figure_raw_vs_normalized(observation: Any, builder: TerrainObservationBuilder, output: Path) -> str:
    tensor = builder.to_tensor_ready(observation)
    raw_scalar = np.concatenate((observation.ego_features, observation.goal_features))
    scalar_names = list(EGO_FEATURE_NAMES + GOAL_FEATURE_NAMES)
    raw_spatial = np.concatenate((
        observation.terrain_channels, observation.hazard_channels,
        observation.validity_channels,
    ))
    spatial_names = list(TERRAIN_CHANNEL_NAMES + HAZARD_CHANNEL_NAMES + VALIDITY_CHANNEL_NAMES)
    raw_min = np.asarray([np.nanmin(channel) for channel in raw_spatial])
    raw_max = np.asarray([np.nanmax(channel) for channel in raw_spatial])
    norm_min = tensor.spatial.reshape(tensor.spatial.shape[0], -1).min(axis=1)
    norm_max = tensor.spatial.reshape(tensor.spatial.shape[0], -1).max(axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    indices = np.arange(len(raw_scalar))
    axes[0, 0].bar(indices, raw_scalar, color="#0072B2")
    axes[0, 0].set_xticks(indices, scalar_names, rotation=35, ha="right", fontsize=8)
    axes[0, 0].set_title("Raw scalar values [physical units or bounded angle]")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 1].bar(indices, tensor.scalar, color="#009E73")
    axes[0, 1].set_xticks(indices, scalar_names, rotation=35, ha="right", fontsize=8)
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_title("Deterministically scaled scalar values")
    axes[0, 1].grid(axis="y", alpha=0.25)
    channel_indices = np.arange(len(spatial_names))
    axes[1, 0].vlines(channel_indices, raw_min, raw_max, linewidth=5, color="#0072B2")
    axes[1, 0].scatter(channel_indices, raw_min, color="black", s=12)
    axes[1, 0].scatter(channel_indices, raw_max, color="black", s=12)
    axes[1, 0].set_xticks(channel_indices, spatial_names, rotation=35, ha="right", fontsize=8)
    axes[1, 0].set_title("Raw spatial valid-cell ranges")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 1].vlines(channel_indices, norm_min, norm_max, linewidth=5, color="#009E73")
    axes[1, 1].scatter(channel_indices, norm_min, color="black", s=12)
    axes[1, 1].scatter(channel_indices, norm_max, color="black", s=12)
    axes[1, 1].set_xticks(channel_indices, spatial_names, rotation=35, ha="right", fontsize=8)
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set_title("Tensor-ready ranges (padding is zero)")
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle("Raw vs deterministic physical normalization (no fitted statistics)")
    return _save(figure, output)


def figure_translation_sanity(
    problem: AttackerBRProblem,
    builder: TerrainObservationBuilder,
    first_id: int,
    second_id: int,
    output: Path,
) -> str:
    first = builder.build(first_id)
    second = builder.build(second_id)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    _add_terrain_top(axes[0], problem)
    for state_id, color, label in (
        (first_id, "#0072B2", "state A"), (second_id, "#D55E00", "state B"),
    ):
        position = problem.position_map(state_id)
        axes[0].scatter(position[0], position[1], color=color, s=55, label=label)
    axes[0].legend()
    axes[0].set_title("Two global placements")
    for ax, observation, label in zip(axes[1:], (first, second), ("A", "B")):
        image = _imshow_local(ax, observation.terrain_channels[0], builder.config, cmap="terrain")
        ax.set_title(f"State {label}: local terrain clearance")
        figure.colorbar(image, ax=ax, fraction=0.046, label="m")
    figure.suptitle(
        "Relative-coordinate geometry sanity check (development visualization; not generalization evidence)"
    )
    return _save(figure, output)


def _boundary_state(problem: AttackerBRProblem) -> int:
    reachable = np.flatnonzero(problem.scene.graph.node_mask)
    return min((int(value) for value in reachable), key=lambda value: problem.position_map(value)[0])


def _comparison_state(problem: AttackerBRProblem, state_id: int) -> int:
    base = problem.state(state_id)
    reachable = np.flatnonzero(problem.scene.graph.node_mask)
    candidates = [
        int(value) for value in reachable
        if problem.state(int(value)).heading_bin == base.heading_bin
        and problem.state(int(value)).altitude_index == base.altitude_index
        and int(value) != state_id
    ]
    return max(
        candidates,
        key=lambda value: np.linalg.norm(
            problem.position_map(value)[:2] - problem.position_map(state_id)[:2]
        ),
    )


def run_validation(
    problem: AttackerBRProblem,
    builder: TerrainObservationBuilder,
    state_id: int,
    boundary_id: int,
    oracle: Any,
) -> dict[str, Any]:
    before_successors = tuple(problem.successors(state_id))
    trajectory = tuple(int(value) for value in oracle.glide_state_ids)
    before_terminal = tuple(problem.is_terminal(value) for value in trajectory)
    before_objective = problem.attacker_objective(int(oracle.switching_state_id), trajectory)
    observation = builder.build(state_id)
    repeat = builder.build(state_id)
    tensor = builder.to_tensor_ready(observation)

    deterministic = all(np.array_equal(a, b, equal_nan=True) for a, b in (
        (observation.ego_features, repeat.ego_features),
        (observation.goal_features, repeat.goal_features),
        (observation.terrain_channels, repeat.terrain_channels),
        (observation.hazard_channels, repeat.hazard_channels),
        (observation.validity_channels, repeat.validity_channels),
    ))
    expected_shapes = builder.tensor_ready_shapes
    shapes_pass = (
        tensor.scalar.shape == expected_shapes["scalar"]
        and tensor.spatial.shape == expected_shapes["spatial"]
    )
    forbidden = {
        "terrain_id", "terrain_name", "terrain_filename", "terrain_category",
        "scenario_id", "scenario_name", "defender_id", "r_neighbor",
        "spatial_resolution_m", "heading_spacing_deg",
    }
    no_identifier_leak = forbidden.isdisjoint(observation.metadata)

    positions = builder.sample_positions_map(state_id)
    valid = observation.validity_channels[0].astype(bool)
    query_errors: list[float] = []
    for row, column in zip(*np.nonzero(valid)):
        if (row + column) % 10:
            continue
        origin_z = max(problem.grid.maximum_altitude_map, problem.scene.terrain.maximum_height) + 1.0
        hit = problem.scene.terrain.first_ray_hit(
            np.asarray([positions[row, column, 0], positions[row, column, 1], origin_z]),
            np.asarray([0.0, 0.0, -1.0]),
        )
        expected = problem.scene.config.physical_scale.distance_m(
            positions[row, column, 2] - hit.point[2],
        )
        query_errors.append(abs(expected - observation.terrain_channels[0, row, column]))

    hazard_errors: list[float] = []
    edges = problem.successors(state_id)[: min(4, len(problem.successors(state_id)))]
    sample_cells = [(10, 10), (8, 10), (10, 8)]
    for edge in edges:
        velocity = problem.scene.config.physical_scale.position_m(
            problem.position_map(edge.target_id) - problem.position_map(state_id)
        ) / edge.duration_s
        for row, column in sample_cells:
            if not valid[row, column]:
                continue
            expected = problem.mdp.hazard_field.evaluate_rate(
                positions[row, column], velocity, 0.0,
            ).total_rate_per_s
            reconstructed = builder.reconstruct_hazard_rate(
                observation, row, column, velocity,
            )
            hazard_errors.append(abs(expected - reconstructed))

    boundary = builder.build(boundary_id)
    boundary_mask = boundary.validity_channels[0].astype(bool)
    boundary_pass = bool(
        np.any(boundary_mask) and np.any(~boundary_mask)
        and np.all(np.isnan(boundary.terrain_channels[:, ~boundary_mask]))
        and np.all(np.isnan(boundary.hazard_channels[:, ~boundary_mask]))
    )

    base = problem.state(state_id)
    angles = np.asarray([problem.grid.heading_rad(i) for i in range(problem.grid.heading_bin_count)])
    wrap_index = int(np.argmax(np.abs(np.diff(angles))))
    state_type = type(base)
    first_id = problem.state_id(state_type(base.x_index, base.y_index, base.altitude_index, wrap_index))
    second_id = problem.state_id(state_type(base.x_index, base.y_index, base.altitude_index, wrap_index + 1))
    encoded_gap = float(np.linalg.norm(
        builder.build(first_id).ego_features[2:] - builder.build(second_id).ego_features[2:]
    ))
    raw_angle_gap = float(abs(angles[wrap_index + 1] - angles[wrap_index]))
    angle_wrap_pass = encoded_gap < 0.2 and raw_angle_gap > np.pi

    serialized = json.loads(json.dumps(builder.config.as_dict()))
    restored_builder = TerrainObservationBuilder(problem, ObservationConfig.from_dict(serialized))
    restored = restored_builder.build(state_id)
    serialization_pass = bool(
        np.array_equal(observation.terrain_channels, restored.terrain_channels, equal_nan=True)
        and np.array_equal(observation.hazard_channels, restored.hazard_channels, equal_nan=True)
    )
    legacy_pass = bool(
        before_successors == tuple(problem.successors(state_id))
        and before_terminal == tuple(problem.is_terminal(value) for value in trajectory)
        and before_objective == problem.attacker_objective(int(oracle.switching_state_id), trajectory)
    )
    checks = {
        "determinism": bool(deterministic),
        "fixed_tensor_shapes": bool(shapes_pass),
        "no_terrain_or_scenario_identifier_leak": bool(no_identifier_leak),
        "terrain_query_parity": bool(max(query_errors, default=0.0) <= TOLERANCE),
        "hazard_rate_reconstruction_parity": bool(max(hazard_errors, default=0.0) <= TOLERANCE),
        "explicit_boundary_mask": bool(boundary_pass),
        "angle_wrap_encoding": bool(angle_wrap_pass),
        "configuration_round_trip": bool(serialization_pass),
        "phase1_behavior_preserved": bool(legacy_pass),
    }
    return {
        "tolerance": TOLERANCE,
        "checks": checks,
        "passed": all(checks.values()),
        "maximum_terrain_query_absolute_error_m": float(max(query_errors, default=0.0)),
        "terrain_queries_checked": len(query_errors),
        "maximum_hazard_rate_reconstruction_absolute_error_per_s": float(max(hazard_errors, default=0.0)),
        "hazard_reconstructions_checked": len(hazard_errors),
        "boundary_state_id": boundary_id,
        "boundary_valid_cells": int(boundary_mask.sum()),
        "boundary_padded_cells": int(boundary_mask.size - boundary_mask.sum()),
        "raw_angle_wrap_gap_rad": raw_angle_gap,
        "encoded_angle_wrap_gap": encoded_gap,
        "legacy_attacker_objective": float(before_objective),
    }


def run_phase2() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0_manifest = _load_json(PHASE0_DIR / "phase0_discovery_manifest.json")
    phase1_manifest = _load_json(PHASE1_DIR / "phase1_attacker_br_interface_manifest.json")
    condition = _condition_from_phase0(phase0_manifest)
    scene = build_scene(condition)
    problem = AttackerBRProblem(scene, default_sensor(scene))
    oracle = exact_best_response(scene, problem.sensor_map, keep_solution=True)
    if not oracle.feasible:
        raise RuntimeError("canonical Phase 0/1 Bellman response is infeasible")
    state_id = int(phase1_manifest["representative_state"]["state_id"])
    boundary_id = _boundary_state(problem)
    comparison_id = _comparison_state(problem, state_id)
    config = ObservationConfig(
        local_window_extent_m=2000.0,
        local_window_shape_cells=(21, 21),
        local_window_orientation="heading_aligned",
    )
    builder = TerrainObservationBuilder(problem, config)
    observation = builder.build(state_id)
    boundary_observation = builder.build(boundary_id)
    validation = run_validation(problem, builder, state_id, boundary_id, oracle)
    if not validation["passed"]:
        failed = [name for name, passed in validation["checks"].items() if not passed]
        raise RuntimeError(f"Phase 2 validation failed: {failed}")

    visualizations = [
        figure_global_to_local(problem, builder, state_id, OUTPUT_DIR / "phase2_global_to_local.png"),
        figure_terrain_channel(observation, builder, OUTPUT_DIR / "phase2_local_terrain_channel.png"),
        figure_hazard_channels(observation, builder, OUTPUT_DIR / "phase2_local_hazard_channels.png"),
        figure_validity_mask(boundary_observation, builder, boundary_id, OUTPUT_DIR / "phase2_validity_mask.png"),
        figure_component_summary(observation, builder, OUTPUT_DIR / "phase2_observation_component_summary.png"),
        figure_raw_vs_normalized(observation, builder, OUTPUT_DIR / "phase2_raw_vs_normalized.png"),
        figure_translation_sanity(
            problem, builder, state_id, comparison_id,
            OUTPUT_DIR / "phase2_translation_geometry_sanity.png",
        ),
    ]
    manifest = {
        "phase": 2,
        "stage": "16.2",
        "purpose": "terrain_aware_dqn_state_representation",
        "observation_builder_source": "3D_RL_DQN/terrain_observation.py:TerrainObservationBuilder",
        "canonical_state_source": "3D_0827/bellman_state.py:BellmanState(x_index, y_index, altitude_index, heading_bin)",
        "canonical_state_facts": {
            "persistent_components": ["x_index", "y_index", "altitude_index", "heading_bin"],
            "gamma_glide_slope": "edge-derived; not persistent state",
            "speed": "fixed glider parameter used by transitions; not persistent state",
            "motion_mode": "absent",
            "representative_state_id": state_id,
        },
        "ego_feature_definition": {
            "ordered_names": list(EGO_FEATURE_NAMES),
            "units": ["m", "m", "dimensionless", "dimensionless"],
            "heading_encoding": "sin/cos to preserve periodic wrap continuity",
        },
        "goal_feature_definition": {
            "ordered_names": list(GOAL_FEATURE_NAMES),
            "units": ["m", "m", "m", "m"],
            "frame": "attacker-centered and heading-aligned",
            "terminal_source": "scene.config.goal plus glider.goal_tolerance_m; terminal is inclusive 3D ball",
        },
        "terrain_source": "3D_0827/map_geometry.py:TerrainModel.surface_meshes/first_ray_hit",
        "terrain_channel_definition": {
            "ordered_names": list(TERRAIN_CHANNEL_NAMES),
            "definition": "current attacker altitude minus highest terrain surface sampled by authoritative vertical mesh ray",
            "units": ["m"],
            "cube_hard_coded": False,
        },
        "hazard_source": "3D_0827/detection_hazard.py:GlideDetectionHazardModel.evaluate_rate",
        "hazard_dependencies": [
            "sensor-to-query LOS visibility", "sensor range", "query position/altitude",
            "candidate action velocity", "radial velocity", "speed/aspect",
        ],
        "hazard_channel_definition": {
            "ordered_names": list(HAZARD_CHANNEL_NAMES),
            "encoding": "five position-dependent sufficient statistics; candidate action velocity reconstructs exact instantaneous rate",
            "decision": "code-derived faithful encoding (Decision C)",
            "authoritative_rate_reconstruction_validated": True,
        },
        "validity_channel_definition": {
            "ordered_names": list(VALIDITY_CHANNEL_NAMES),
            "meaning": "1 inside Phase 1 horizontal attacker domain; 0 for fixed-window padding outside it",
            "raw_unavailable_values": "NaN plus explicit mask",
            "tensor_padding_values": 0.0,
        },
        "local_window_extent_m": config.local_window_extent_m,
        "local_window_shape_cells": list(config.local_window_shape_cells),
        "local_window_sample_spacing_m": list(config.sample_spacing_m),
        "local_window_orientation": config.local_window_orientation,
        "new_design_choices": {
            "decision_B": {
                "status": "explicitly_confirmed_by_user_during_phase_16_2",
                "extent_m": 2000.0,
                "shape_cells": [21, 21],
                "sample_spacing_m": [100.0, 100.0],
                "orientation": "heading_aligned",
            },
            "hazard_channels": "user accepted the code-derived five-channel sufficient-statistic proposal",
        },
        "normalization_definition": {
            "type": "deterministic physical scaling only",
            "ego_altitude_and_clearance": "divide by canonical vertical domain span and clip [-1,1]",
            "goal_horizontal": "divide by horizontal domain diagonal and clip [-1,1]",
            "goal_vertical": "divide by vertical domain span and clip [-1,1]",
            "goal_distance_minus_tolerance": "divide by 3D domain diagonal and clip [-1,1]",
            "terrain_clearance": "divide by vertical domain span and clip [-1,1]",
            "hazard_sufficient_statistics": "already bounded; unchanged",
            "invalid_spatial_cells": "zeroed, with explicit validity channel retained",
            "dataset_fitted_statistics": "none in Phase 2; deferred to Phase 4 training terrains if later required",
        },
        "raw_observation_schema": observation.raw_schema(),
        "tensor_ready_shape": {
            key: list(value) for key, value in builder.tensor_ready_shapes.items()
        },
        "discretization_scope_decision": {
            "decision_A": "condition-specific DQN for each spatial/angular discretization",
            "neural_state_excludes": ["dx", "dy", "dh", "dpsi", "dgamma", "Local-SSE r_neighbor"],
            "future_only": "cross-discretization DQN (A2)",
        },
        "repository_sources_inspected": [
            "3D_RL_DQN/attacker_br_problem.py",
            "3D_0827/bellman_state.py",
            "3D_0827/bellman_geometry.py",
            "3D_0827/map_geometry.py",
            "3D_0827/detection_hazard.py",
            "3D_0827/edge_hazard.py",
            "3D_0827/P1b_RL_approximation.py",
            "3D_0827/P1b_condition.py",
            "3D_0827/glider_gym_env.py",
        ],
        "validation_test_summary": validation,
        "visualization_outputs": visualizations,
        "visualization_scope": "canonical cube development validation only; no terrain-generalization claim",
        "unresolved_design_items": [],
        "implementation_restrictions_verified": {
            "dqn_network_added": False,
            "training_or_replay_added": False,
            "bellman_or_tabular_behavior_modified": False,
            "local_sse_modified": False,
            "torch_or_gymnasium_dependency": False,
        },
        "files_changed": [
            "3D_RL_DQN/terrain_observation.py",
            "3D_RL_DQN/test_terrain_observation.py",
            "3D_RL_DQN/phase2_validation.py",
            "3D_RL_DQN/README.md",
            "3D_0827/docs/stage16_dqn_terrain_generalization_plan.md",
        ],
    }
    output = OUTPUT_DIR / MANIFEST_NAME
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": _relative(output),
        "validation_passed": validation["passed"],
        "tensor_ready_shape": manifest["tensor_ready_shape"],
        "visualization_count": len(visualizations),
    }, indent=2))
    return output


if __name__ == "__main__":
    run_phase2()
