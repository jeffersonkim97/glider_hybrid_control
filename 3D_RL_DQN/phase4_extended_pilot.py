"""Generate and smoke-check the extended Phase 16.4 random-box pilot bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import project_paths
from extended_failure_env import ExpandedFailureAttackerBRProblem
from multiscale_observation import (
    MultiscaleObservationConfig,
    MultiscaleTerrainObservationBuilder,
)
from phase4_early_stopping import EarlyStoppingConfig
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    VALIDATION_EXTENDED,
    ExtendedPilotConfig,
    build_extended_scenario_bank,
    build_random_box_terrain,
    extended_dataset_manifest,
)
from phase4_extended_scene import build_extended_scene


OUTPUT_ROOT = (
    project_paths.CORE_DIR / "figure" / "phase_4_extended_random_boxes" / "pilot_v1"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _circle(
    center: tuple[float, float, float], radius_map: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle = np.linspace(0.0, 2.0 * np.pi, 101)
    return (
        center[0] + radius_map * np.cos(angle),
        center[1] + radius_map * np.sin(angle),
        np.full_like(angle, center[2] + 0.03),
    )


def _terrain_bank_figure(
    splits: dict[str, tuple[Any, ...]],
    terrains: dict[str, Any],
    config: ExtendedPilotConfig,
) -> go.Figure:
    ordered = [terrains[key] for key in sorted(terrains)]
    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)] for _ in range(2)],
        subplot_titles=[spec.terrain_instance_id for spec in ordered],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    all_scenarios = (*splits[TRAIN_EXTENDED], *splits[VALIDATION_EXTENDED])
    visibility_colors = {"low": "#2ca02c", "mixed": "#ffbf00", "high": "#d62728"}
    for index, terrain_spec in enumerate(ordered):
        row, column = divmod(index, 3)
        row += 1
        column += 1
        terrain = build_random_box_terrain(terrain_spec, config)
        for box_index, box in enumerate(terrain.obstacle_boxes()):
            mesh = box.mesh()
            figure.add_trace(go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.triangles[:, 0],
                j=mesh.triangles[:, 1],
                k=mesh.triangles[:, 2],
                color="#7f8c8d",
                opacity=0.70,
                flatshading=True,
                name="random box",
                legendgroup="boxes",
                showlegend=index == 0 and box_index == 0,
                hovertemplate="box<extra></extra>",
            ), row=row, col=column)
        cases = [
            spec for spec in all_scenarios
            if spec.terrain_category == terrain_spec.terrain_instance_id
        ]
        mission = cases[0]
        figure.add_trace(go.Scatter3d(
            x=[mission.start_map[0]], y=[mission.start_map[1]], z=[mission.start_map[2]],
            mode="markers", marker={"size": 6, "color": "#1f77b4", "symbol": "diamond"},
            name="sampled start", legendgroup="start", showlegend=index == 0,
        ), row=row, col=column)
        figure.add_trace(go.Scatter3d(
            x=[mission.goal_map[0]], y=[mission.goal_map[1]], z=[mission.goal_map[2]],
            mode="markers", marker={"size": 7, "color": "#9467bd", "symbol": "diamond"},
            name="sampled goal", legendgroup="goal", showlegend=index == 0,
        ), row=row, col=column)
        for label in ("low", "mixed", "high"):
            selected = [case for case in cases if case.visibility_class == label]
            figure.add_trace(go.Scatter3d(
                x=[case.sensor_map[0] for case in selected],
                y=[case.sensor_map[1] for case in selected],
                z=[case.sensor_map[2] + 0.03 for case in selected],
                mode="markers",
                marker={"size": 5, "color": visibility_colors[label]},
                name=f"sensor: {label}",
                legendgroup=f"sensor-{label}",
                showlegend=index == 0,
                text=[f"visibility={case.visibility_fraction:.3f}" for case in selected],
                hovertemplate="%{text}<extra></extra>",
            ), row=row, col=column)
        for center, radius, color, name, group in (
            (
                config.start_center_map, config.start_sampling_radius_map,
                "#1f77b4", "r_sp = 100 m", "start-region",
            ),
            (
                config.goal_center_map, config.goal_sampling_radius_map,
                "#9467bd", "r_gp = 100 m", "goal-region",
            ),
            (
                mission.goal_map,
                mission.goal_tolerance_m / config.meters_per_map_unit,
                "#e377c2", "terminal radius = 50 m", "terminal",
            ),
        ):
            x, y, z = _circle(center, radius)
            figure.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines",
                line={"width": 4, "color": color},
                name=name, legendgroup=group, showlegend=index == 0,
                hoverinfo="name",
            ), row=row, col=column)
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        figure.layout[scene_name].update(
            xaxis={"range": [-10, 10], "title": "x [map units]"},
            yaxis={"range": [-8, 8], "title": "y [map units]"},
            zaxis={"range": [0, 5.2], "title": "z [map units]"},
            aspectmode="manual",
            aspectratio={"x": 1.25, "y": 1.0, "z": 0.45},
        )
    figure.update_layout(
        title=(
            "Extended Phase 16.4 finite random-box scenario bank<br>"
            "<sup>Each panel is independently rotatable; green/amber/red sensors are "
            "low/mixed/high geometry-only visibility strata.</sup>"
        ),
        height=980,
        width=1500,
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        legend={"orientation": "h", "y": -0.02},
    )
    return figure


def run_extended_pilot(
    *, output_dir: Path = OUTPUT_ROOT, spatial_resolution_m: float = 100.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExtendedPilotConfig()
    manifest = extended_dataset_manifest(config)
    splits, terrains = build_extended_scenario_bank(config)
    manifest_path = output_dir / "extended_dataset_manifest.json"
    _write_json(manifest_path, manifest)

    figure_path = output_dir / "random_box_scenario_bank_3d.html"
    figure = _terrain_bank_figure(splits, terrains, config)
    figure.write_html(
        figure_path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )

    scenario = splits[TRAIN_EXTENDED][0]
    started = perf_counter()
    scene = build_extended_scene(
        scenario,
        terrains[scenario.terrain_category],
        spatial_resolution_m=float(spatial_resolution_m),
    )
    scene_runtime = perf_counter() - started
    problem = ExpandedFailureAttackerBRProblem(scene, scenario.sensor_map)
    started = perf_counter()
    starts = problem.switching_state_ids()
    switching_runtime = perf_counter() - started
    if not starts:
        raise RuntimeError("pilot smoke scenario has no fully occluded switching candidate")
    observation_config = MultiscaleObservationConfig()
    builder = MultiscaleTerrainObservationBuilder(problem, observation_config)
    tensor = builder.to_tensor_ready(builder.build(starts[0]))
    sampled_starts = starts[:20]
    expanded_edges = sum(len(problem.successors(state)) for state in sampled_starts)
    goal_only_edges = sum(len(problem.mdp.actions(state)) for state in sampled_starts)
    audit = {
        "schema": "p1b-phase4-extended-pilot-audit-v1",
        "status": "smoke_passed",
        "spatial_resolution_m": float(spatial_resolution_m),
        "scenario_id": scenario.scenario_id,
        "terrain_instance_id": scenario.terrain_category,
        "scene_build_runtime_sec": scene_runtime,
        "switching_build_runtime_sec": switching_runtime,
        "terminal_state_count": int(scene.graph.statistics.terminal_state_count),
        "goal_reachable_state_count": int(scene.sizes()["N_S_goal_reachable"]),
        "switching_candidates": {
            "input_before_powered_occlusion": int(
                problem._switching_candidates().timing[
                    "powered_occlusion_candidate_input"
                ]
            ),
            "retained_after_powered_occlusion": len(starts),
            "interior_occlusion_sample_spacing_m": (
                problem.powered_occlusion_spacing_m
            ),
        },
        "observation_config": observation_config.as_dict(),
        "observation_shapes": {
            key: list(value) for key, value in builder.tensor_ready_shapes.items()
        },
        "observation_value_range": [
            float(np.min(tensor.spatial)), float(np.max(tensor.spatial)),
        ],
        "failure_action_audit_first_20_switching_states": {
            "goal_reachable_edges": goal_only_edges,
            "expanded_physically_feasible_edges": expanded_edges,
            "additional_failure_branch_edges": expanded_edges - goal_only_edges,
        },
        "early_stopping": EarlyStoppingConfig().as_dict(),
        "artifacts": {
            "dataset_manifest": str(
                manifest_path.relative_to(project_paths.WORKSPACE_ROOT)
            ),
            "terrain_bank_3d_html": str(
                figure_path.relative_to(project_paths.WORKSPACE_ROOT)
            ),
        },
        "scope": (
            "dataset/scene/observation/failure-action smoke only; no DQN training was run"
        ),
    }
    audit_path = output_dir / "extended_pilot_smoke_audit.json"
    _write_json(audit_path, audit)
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-resolution-m", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    path = run_extended_pilot(
        output_dir=arguments.output_dir,
        spatial_resolution_m=arguments.spatial_resolution_m,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = ["OUTPUT_ROOT", "run_extended_pilot"]
