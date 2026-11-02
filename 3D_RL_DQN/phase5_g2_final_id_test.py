"""Open and evaluate the first sealed G2 final-ID random-box terrain.

The terrain, mission, and sensors are generated exactly once from predeclared
seed ranges.  The frozen episode-10,000 final checkpoints are evaluated without
training, checkpoint selection, or hyperparameter changes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

import project_paths
from dqn_model import DQNNetworkConfig, TerrainDQN
from los_geometry import LOSModel
from phase4_early_stopping import evaluate_final_acceptance
from phase4_extended_dataset import (
    ExtendedScenarioSpec,
    RandomBoxTerrainSpec,
    build_random_box_terrain,
)
from phase4_g2_dataset import (
    G2_FINAL_ID_TEST,
    G2_TRAIN,
    G2_VALIDATION,
    G2BenchmarkConfig,
    _build_terrain_scenarios,
    build_g2_scenario_bank,
    g2_dataset_manifest,
)
from phase4_g2_training import build_g2_runtimes
from phase4_multi_terrain import evaluate_scenarios


FINAL_ID_SCHEMA = "p1b-phase5-g2-final-id-test-v1"
FINAL_TERRAIN_INDEX = 0
FINAL_TERRAIN_COUNT = 4
FINAL_TERRAIN_SEED_BASE = 28_401
FINAL_MISSION_SEED_BASE = 28_501
FINAL_SENSOR_SEED_BASE = 28_601
SPATIAL_RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
HEADING_SPACING_DEG = 5.0

FINAL_TERRAIN_LABEL = f"final_id_{FINAL_TERRAIN_INDEX:03d}"
FINAL_ID_OUTPUT_BASE = (
    project_paths.DQN_DIR
    / "runs"
    / "phase5_g2_final_id_test"
)
OUTPUT_ROOT = FINAL_ID_OUTPUT_BASE / FINAL_TERRAIN_LABEL
MANIFEST_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_manifest.json"
TERRAIN_HTML_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_terrain_3d.html"
SUMMARY_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_summary.json"
RESULTS_CSV_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_results_table.csv"
RESULTS_MARKDOWN_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_results_table.md"
AGGREGATE_ROOT = FINAL_ID_OUTPUT_BASE / "aggregate_000_003"
AGGREGATE_JSON_PATH = AGGREGATE_ROOT / "final_id_aggregate_000_003.json"
AGGREGATE_CSV_PATH = AGGREGATE_ROOT / "final_id_aggregate_000_003.csv"
AGGREGATE_MARKDOWN_PATH = AGGREGATE_ROOT / "final_id_aggregate_000_003.md"
CHECKPOINT_ROOT = (
    project_paths.DQN_DIR
    / "checkpoints"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g2_generalization"
)


def configure_final_terrain(index: int) -> None:
    """Select one of the four predeclared final-ID terrains before any I/O."""

    global FINAL_TERRAIN_INDEX, FINAL_TERRAIN_LABEL, OUTPUT_ROOT
    global MANIFEST_PATH, TERRAIN_HTML_PATH, SUMMARY_PATH
    global RESULTS_CSV_PATH, RESULTS_MARKDOWN_PATH
    index = int(index)
    if not 0 <= index < FINAL_TERRAIN_COUNT:
        raise ValueError(
            f"terrain index must be in [0, {FINAL_TERRAIN_COUNT - 1}]"
        )
    FINAL_TERRAIN_INDEX = index
    FINAL_TERRAIN_LABEL = f"final_id_{index:03d}"
    OUTPUT_ROOT = FINAL_ID_OUTPUT_BASE / FINAL_TERRAIN_LABEL
    MANIFEST_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_manifest.json"
    TERRAIN_HTML_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_terrain_3d.html"
    SUMMARY_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_summary.json"
    RESULTS_CSV_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_results_table.csv"
    RESULTS_MARKDOWN_PATH = OUTPUT_ROOT / f"{FINAL_TERRAIN_LABEL}_results_table.md"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _condition_tag(dx_m: float) -> str:
    return f"dx{float(dx_m):g}m_dpsi{HEADING_SPACING_DEG:g}deg"


def _checkpoint_path(dx_m: float) -> Path:
    return (
        CHECKPOINT_ROOT
        / _condition_tag(dx_m)
        / "seed0_ep10000_v1"
        / "phase4_seed0_final.pt"
    )


def build_final_id_000() -> tuple[
    tuple[ExtendedScenarioSpec, ...], RandomBoxTerrainSpec, G2BenchmarkConfig,
]:
    config = G2BenchmarkConfig()
    scenarios, terrain = _build_terrain_scenarios(
        split=G2_FINAL_ID_TEST,
        split_name="final_id",
        index=FINAL_TERRAIN_INDEX,
        terrain_seed_base=FINAL_TERRAIN_SEED_BASE,
        mission_seed_base=FINAL_MISSION_SEED_BASE,
        sensor_seed_base=FINAL_SENSOR_SEED_BASE,
        resample_attempt=0,
        config=config,
    )
    if len(scenarios) != 4:
        raise RuntimeError(
            f"final ID terrain {FINAL_TERRAIN_INDEX:03d} must contain four "
            "sensor cases"
        )

    active_splits, active_terrains = build_g2_scenario_bank(config)
    active_hashes = {
        item.terrain_hash
        for split in (G2_TRAIN, G2_VALIDATION)
        for item in active_splits[split]
    }
    active_seeds = {
        item.generation_seed for item in active_terrains.values()
    }
    if terrain.terrain_hash in active_hashes:
        raise RuntimeError("final test terrain geometry overlaps train/validation")
    if terrain.generation_seed in active_seeds:
        raise RuntimeError("final test terrain seed overlaps train/validation")
    return tuple(scenarios), terrain, config


def freeze_final_id_000() -> tuple[
    dict[str, Any], tuple[ExtendedScenarioSpec, ...], RandomBoxTerrainSpec,
    G2BenchmarkConfig,
]:
    scenarios, terrain, config = build_final_id_000()
    training_manifest = g2_dataset_manifest(config)
    checkpoints: dict[str, Any] = {}
    for dx_m in SPATIAL_RESOLUTIONS_M:
        path = _checkpoint_path(dx_m)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen final checkpoint: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "final_training_state":
            raise RuntimeError(f"checkpoint is not a final training state: {path}")
        if int(payload.get("episode", -1)) != 10_000:
            raise RuntimeError(f"checkpoint is not episode 10,000: {path}")
        if payload.get("dataset_manifest_sha256") != training_manifest["manifest_sha256"]:
            raise RuntimeError(f"checkpoint training dataset hash drifted: {path}")
        checkpoints[_condition_tag(dx_m)] = {
            "path": path.relative_to(project_paths.WORKSPACE_ROOT).as_posix(),
            "sha256": _file_sha256(path),
            "kind": payload["kind"],
            "episode": int(payload["episode"]),
            "training_dataset_manifest_sha256": payload["dataset_manifest_sha256"],
        }
        del payload

    manifest_without_hash = {
        "schema": FINAL_ID_SCHEMA,
        "status": "frozen_before_evaluation",
        "test_role": "reserved unseen in-distribution generalization probe",
        "training_dataset_manifest_sha256": training_manifest["manifest_sha256"],
        "split": G2_FINAL_ID_TEST,
        "seed_contract": {
            "terrain_seed_base": FINAL_TERRAIN_SEED_BASE,
            "mission_seed_base": FINAL_MISSION_SEED_BASE,
            "sensor_seed_base": FINAL_SENSOR_SEED_BASE,
            "terrain_index": FINAL_TERRAIN_INDEX,
            "resample_attempt": 0,
            "selection_policy": "generated_once_without_visual_or_performance_selection",
        },
        "terrain": terrain.as_dict(),
        "scenarios": [scenario.as_dict() for scenario in scenarios],
        "conditions": [
            {
                "spatial_resolution_m": dx_m,
                "heading_spacing_deg": HEADING_SPACING_DEG,
            }
            for dx_m in SPATIAL_RESOLUTIONS_M
        ],
        "checkpoints": checkpoints,
        "acceptance_rule": {
            "scope": "descriptive for this four-case probe",
            "success_rate_minimum": 1.0,
            "median_relative_error_maximum": 0.10,
            "p90_relative_error_maximum": 0.30,
            "p90_method": "higher",
            "maximum_relative_error_policy": "report_only",
            "formal_conclusion_requires": "four final-ID terrains / sixteen cases",
        },
        "prohibited_uses": [
            "training", "checkpoint selection", "early stopping",
            "hyperparameter tuning", "terrain reselection based on results",
        ],
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": sha256(
            _canonical_json(manifest_without_hash).encode("utf-8")
        ).hexdigest(),
    }
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("frozen final-ID manifest differs from regenerated value")
    else:
        _write_json(MANIFEST_PATH, manifest)
    return manifest, scenarios, terrain, config


def _add_terrain(figure: go.Figure, terrain: Any, *, row: int | None = None,
                 col: int | None = None, showlegend: bool = True,
                 meters_per_unit: float = 100.0) -> None:
    for index, box in enumerate(terrain.obstacle_boxes()):
        mesh = box.mesh()
        vertices = mesh.vertices * meters_per_unit
        trace = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=mesh.triangles[:, 0], j=mesh.triangles[:, 1],
            k=mesh.triangles[:, 2], color="#7f8c8d", opacity=0.72,
            flatshading=True, name="terrain boxes", legendgroup="terrain",
            showlegend=showlegend and index == 0,
            hovertemplate=f"box {index}<extra></extra>",
        )
        if row is None:
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=col)


def plot_final_id_000_terrain(
    manifest: dict[str, Any], scenarios: tuple[ExtendedScenarioSpec, ...],
    terrain_spec: RandomBoxTerrainSpec, config: G2BenchmarkConfig,
) -> Path:
    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    scale = float(config.pilot.meters_per_map_unit)
    figure = go.Figure()
    _add_terrain(figure, terrain, meters_per_unit=scale)
    mission = scenarios[0]
    start = np.asarray(mission.start_map) * scale
    goal = np.asarray(mission.goal_map) * scale
    figure.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2] + 4.0],
        mode="markers+text", text=["START"], textposition="top center",
        marker={"size": 8, "color": "#0072B2", "symbol": "diamond"},
        name="start", hovertemplate="START<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
    ))
    figure.add_trace(go.Scatter3d(
        x=[goal[0]], y=[goal[1]], z=[goal[2] + 4.0],
        mode="markers+text", text=["GOAL"], textposition="top center",
        marker={"size": 9, "color": "#9467bd", "symbol": "diamond"},
        name="goal", hovertemplate="GOAL<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>",
    ))
    colors = {"low": "#2ca02c", "mixed": "#ffbf00", "high": "#d62728"}
    for label in ("low", "mixed", "high"):
        selected = [case for case in scenarios if case.visibility_class == label]
        figure.add_trace(go.Scatter3d(
            x=[case.sensor_map[0] * scale for case in selected],
            y=[case.sensor_map[1] * scale for case in selected],
            z=[case.sensor_map[2] * scale + 4.0 for case in selected],
            mode="markers+text",
            text=[label.upper() if len(selected) == 1 else f"{label.upper()} {i + 1}"
                  for i in range(len(selected))],
            textposition="top center", marker={"size": 7, "color": colors[label]},
            name=f"sensor: {label}",
            customdata=[[case.visibility_fraction, case.scenario_id] for case in selected],
            hovertemplate=(
                "%{text} SENSOR<br>x=%{x:.1f} m<br>y=%{y:.1f} m"
                "<br>visibility=%{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"
            ),
        ))
    x0, x1, y0, y1 = config.pilot.map_bounds
    figure.update_layout(
        title=(
            f"Frozen unseen G2 {FINAL_TERRAIN_LABEL.replace('_', '-')}<br>"
            f"<sup>seed={terrain_spec.generation_seed}; terrain hash="
            f"{terrain_spec.terrain_hash[:12]}; manifest="
            f"{manifest['manifest_sha256'][:12]}</sup>"
        ),
        width=1150, height=820, margin={"l": 0, "r": 0, "t": 90, "b": 0},
        legend={"orientation": "h", "y": 0.01, "x": 0.02},
        scene={
            "xaxis": {"range": [x0 * scale, x1 * scale], "title": "x [m]"},
            "yaxis": {"range": [y0 * scale, y1 * scale], "title": "y [m]"},
            "zaxis": {"range": [0.0, 520.0], "title": "z [m]"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.25, "y": 1.0, "z": 0.45},
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.0}},
        },
    )
    figure.write_html(
        TERRAIN_HTML_PATH, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return TERRAIN_HTML_PATH


def _trajectory_positions(context: Any, state_ids: Iterable[int]) -> np.ndarray:
    values = [context.problem.position_map(int(state_id)) for state_id in state_ids]
    if not values:
        return np.empty((0, 3), dtype=float)
    return np.asarray(values, dtype=float) * float(
        context.problem.scene.config.physical_scale.meters_per_map_unit
    )


def _sensor_case_labels(
    scenarios: tuple[ExtendedScenarioSpec, ...],
) -> dict[str, tuple[str, str]]:
    """Return stable filename/display labels for the four sensor cases."""

    mixed_index = 0
    labels: dict[str, tuple[str, str]] = {}
    for scenario in scenarios:
        if scenario.visibility_class == "mixed":
            suffix = "a" if mixed_index == 0 else "b"
            labels[scenario.scenario_id] = (
                f"mixed_{suffix}", f"Mixed {suffix.upper()}",
            )
            mixed_index += 1
        else:
            labels[scenario.scenario_id] = (
                scenario.visibility_class,
                scenario.visibility_class.capitalize(),
            )
    if mixed_index != 2:
        raise RuntimeError("final ID terrain must contain Mixed A and Mixed B")
    return labels


def plot_single_sensor_trajectory(
    *, dx_m: float, context: Any | None, record: dict[str, Any] | None,
    scenario: ExtendedScenarioSpec, sensor_display_name: str,
    terrain_spec: RandomBoxTerrainSpec, config: G2BenchmarkConfig,
    shared_scene: Any, output: Path,
) -> Path:
    """Render one interactive audit plot for one dx and one sensor case."""

    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    scale = float(config.pilot.meters_per_map_unit)
    figure = go.Figure()
    _add_terrain(figure, terrain, meters_per_unit=scale)

    mission = shared_scene.mission_for(scenario.sensor_map)
    los_model = LOSModel(shared_scene.terrain)
    contour = los_model.trace_tangent_contour(
        mission.sensor,
        probe_grid_size=shared_scene.config.discretization.los_probe_grid_size,
        boundary_refinement_steps=(
            shared_scene.config.discretization.los_boundary_refinement_steps
        ),
    )
    los_mesh = los_model.build_tangent_surface(
        contour, display_extension_factor=2.2,
    ).display_mesh()
    los_vertices = np.asarray(los_mesh.vertices, dtype=float) * scale
    los_triangles = np.asarray(los_mesh.triangles, dtype=np.int64)
    figure.add_trace(go.Mesh3d(
        x=los_vertices[:, 0], y=los_vertices[:, 1], z=los_vertices[:, 2],
        i=los_triangles[:, 0], j=los_triangles[:, 1],
        k=los_triangles[:, 2], color="#56B4E9", opacity=0.18,
        flatshading=True, name="LOS tangent surface",
        hovertemplate="LOS tangent surface<extra></extra>",
    ))

    start = np.asarray(scenario.start_map, dtype=float) * scale
    goal = np.asarray(scenario.goal_map, dtype=float) * scale
    sensor = np.asarray(scenario.sensor_map, dtype=float) * scale
    for name, position, color, symbol in (
        ("START", start, "black", "diamond"),
        ("GOAL", goal, "#9467bd", "diamond"),
        (f"{sensor_display_name.upper()} SENSOR", sensor, "#d62728", "circle"),
    ):
        figure.add_trace(go.Scatter3d(
            x=[position[0]], y=[position[1]], z=[position[2]],
            mode="markers+text", text=[name], textposition="top center",
            marker={"size": 7, "color": color, "symbol": symbol},
            name=name.lower(),
            hovertemplate=(
                f"{name}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m"
                "<br>z=%{z:.1f} m<extra></extra>"
            ),
        ))

    status_line: str
    if context is not None and record is not None:
        for method, state_ids, color, symbol in (
            ("Bellman", record["bellman_trajectory"], "#D55E00", "circle"),
            ("DQN", record["dqn"]["trajectory"], "#0072B2", "square"),
        ):
            positions = _trajectory_positions(context, state_ids)
            if not len(positions):
                continue
            switch = positions[0]
            figure.add_trace(go.Scatter3d(
                x=[start[0], switch[0]], y=[start[1], switch[1]],
                z=[start[2], switch[2]], mode="lines",
                line={"color": color, "width": 6, "dash": "dot"},
                name=f"{method} powered segment",
                hovertemplate=f"{method} powered segment<extra></extra>",
            ))
            figure.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="lines+markers", line={"color": color, "width": 6},
                marker={"size": 3, "color": color},
                name=f"{method} glide trajectory",
                hovertemplate=(
                    f"{method}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m"
                    "<br>z=%{z:.1f} m<extra></extra>"
                ),
            ))
            figure.add_trace(go.Scatter3d(
                x=[switch[0]], y=[switch[1]], z=[switch[2]],
                mode="markers+text", text=[f"{method} SWITCH"],
                textposition="top center",
                marker={
                    "size": 9, "color": color, "symbol": symbol,
                    "line": {"color": "white", "width": 1},
                },
                name=f"{method} switching point",
                hovertemplate=(
                    f"{method} switching point<br>state={int(state_ids[0])}"
                    "<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m"
                    "<extra></extra>"
                ),
            ))
            figure.add_trace(go.Scatter3d(
                x=[sensor[0], switch[0]], y=[sensor[1], switch[1]],
                z=[sensor[2], switch[2]], mode="lines",
                line={"color": color, "width": 2, "dash": "dash"},
                name=f"{method} sensor-to-switch tangent ray",
                hoverinfo="skip",
            ))
        status_line = (
            f"Bellman J_A={record['bellman_J_A']:.6f}; "
            f"DQN J_A={record['dqn']['attacker_objective']:.6f}; "
            f"relative error={100.0 * record['relative_error']:.2f}%"
        )
    else:
        status_line = (
            "MISSION INFEASIBLE: no powered-occluded, goal-reachable "
            "switching candidate; therefore no Bellman/DQN trajectory or "
            "switching point exists"
        )
        figure.add_annotation(
            text=status_line, x=0.5, y=0.98, xref="paper", yref="paper",
            showarrow=False, bgcolor="rgba(255,235,235,0.92)",
            bordercolor="#d62728", borderwidth=1, font={"color": "#a00000"},
        )

    x0, x1, y0, y1 = config.pilot.map_bounds
    figure.update_layout(
        title=(
            f"Unseen {FINAL_TERRAIN_LABEL.replace('_', '-')} — "
            f"{sensor_display_name} sensor — "
            f"dx={dx_m:g} m<br><sup>visibility="
            f"{scenario.visibility_fraction:.3f}; {status_line}</sup>"
        ),
        width=1250, height=900, margin={"l": 0, "r": 0, "t": 100, "b": 0},
        legend={"orientation": "h", "y": 0.01, "x": 0.01},
        scene={
            "xaxis": {"range": [x0 * scale, x1 * scale], "title": "x [m]"},
            "yaxis": {"range": [y0 * scale, y1 * scale], "title": "y [m]"},
            "zaxis": {"range": [0.0, 520.0], "title": "z [m]"},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.25, "y": 1.0, "z": 0.45},
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.0}},
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return output


def plot_trajectory_comparison(
    *, dx_m: float, contexts: list[Any], records: list[dict[str, Any]],
    scenarios: tuple[ExtendedScenarioSpec, ...],
    terrain_spec: RandomBoxTerrainSpec, config: G2BenchmarkConfig,
    output: Path,
) -> Path:
    records_by_id = {record["scenario_id"]: record for record in records}
    contexts_by_id = {context.spec.scenario_id: context for context in contexts}
    titles = [
        (
            f"{scenario.visibility_class.upper()}: "
            f"visibility={scenario.visibility_fraction:.3f}"
            + (
                ""
                if scenario.scenario_id in records_by_id
                else " | mission infeasible"
            )
        )
        for scenario in scenarios
    ]
    figure = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles,
        horizontal_spacing=0.02, vertical_spacing=0.06,
    )
    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    scale = float(config.pilot.meters_per_map_unit)
    if not contexts:
        raise RuntimeError("trajectory plotting requires at least one feasible context")
    shared_scene = contexts[0].problem.scene
    los_model = LOSModel(shared_scene.terrain)
    for panel, scenario in enumerate(scenarios):
        row, col = divmod(panel, 2)
        row += 1
        col += 1
        context = contexts_by_id.get(scenario.scenario_id)
        record = records_by_id.get(scenario.scenario_id)
        _add_terrain(
            figure, terrain, row=row, col=col,
            showlegend=panel == 0, meters_per_unit=scale,
        )
        mission = shared_scene.mission_for(scenario.sensor_map)
        contour = los_model.trace_tangent_contour(
            mission.sensor,
            probe_grid_size=(
                shared_scene.config.discretization.los_probe_grid_size
            ),
            boundary_refinement_steps=(
                shared_scene.config.discretization.los_boundary_refinement_steps
            ),
        )
        los_mesh = los_model.build_tangent_surface(
            contour, display_extension_factor=2.2,
        ).display_mesh()
        los_vertices = np.asarray(los_mesh.vertices, dtype=float) * scale
        los_triangles = np.asarray(los_mesh.triangles, dtype=np.int64)
        figure.add_trace(go.Mesh3d(
            x=los_vertices[:, 0], y=los_vertices[:, 1], z=los_vertices[:, 2],
            i=los_triangles[:, 0], j=los_triangles[:, 1],
            k=los_triangles[:, 2], color="#56B4E9", opacity=0.16,
            flatshading=True, name="LOS tangent surface",
            legendgroup="los-surface", showlegend=panel == 0,
            hovertemplate="LOS tangent surface<extra></extra>",
        ), row=row, col=col)
        start = np.asarray(scenario.start_map) * scale
        goal = np.asarray(scenario.goal_map) * scale
        sensor = np.asarray(scenario.sensor_map) * scale
        figure.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]], mode="markers+text",
            marker={"size": 5, "color": "black", "symbol": "diamond"},
            text=["START"], textposition="top center",
            name="start", legendgroup="start", showlegend=panel == 0,
        ), row=row, col=col)
        figure.add_trace(go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]], mode="markers+text",
            marker={"size": 6, "color": "#9467bd", "symbol": "diamond"},
            text=["GOAL"], textposition="top center",
            name="goal", legendgroup="goal", showlegend=panel == 0,
        ), row=row, col=col)
        figure.add_trace(go.Scatter3d(
            x=[sensor[0]], y=[sensor[1]], z=[sensor[2]], mode="markers+text",
            marker={"size": 5, "color": "#d62728"}, name="sensor",
            text=["SENSOR"], textposition="top center",
            legendgroup="sensor", showlegend=panel == 0,
        ), row=row, col=col)
        if context is not None and record is not None:
            for method, state_ids, color, symbol in (
                ("Bellman", record["bellman_trajectory"], "#D55E00", "circle"),
                ("DQN", record["dqn"]["trajectory"], "#0072B2", "square"),
            ):
                positions = _trajectory_positions(context, state_ids)
                if not len(positions):
                    continue
                switch = positions[0]
                figure.add_trace(go.Scatter3d(
                    x=[start[0], switch[0]],
                    y=[start[1], switch[1]],
                    z=[start[2], switch[2]],
                    mode="lines", line={"color": color, "width": 5, "dash": "dot"},
                    name=f"{method} powered segment", legendgroup=method,
                    showlegend=panel == 0,
                    hovertemplate=f"{method} powered segment<extra></extra>",
                ), row=row, col=col)
                figure.add_trace(go.Scatter3d(
                    x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                    mode="lines+markers",
                    line={"color": color, "width": 5},
                    marker={"size": 2, "color": color},
                    name=f"{method} glide trajectory", legendgroup=method,
                    showlegend=panel == 0,
                    hovertemplate=(
                        f"{method}<br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m"
                        "<br>z=%{z:.1f} m<extra></extra>"
                    ),
                ), row=row, col=col)
                figure.add_trace(go.Scatter3d(
                    x=[switch[0]], y=[switch[1]], z=[switch[2]],
                    mode="markers+text", text=[f"{method[0]} SWITCH"],
                    textposition="top center",
                    marker={
                        "size": 8, "color": color, "symbol": symbol,
                        "line": {"color": "white", "width": 1},
                    },
                    name=f"{method} switching point", legendgroup=method,
                    showlegend=panel == 0,
                    hovertemplate=(
                        f"{method} switching point<br>state={int(state_ids[0])}"
                        "<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m"
                        "<extra></extra>"
                    ),
                ), row=row, col=col)
                figure.add_trace(go.Scatter3d(
                    x=[sensor[0], switch[0]],
                    y=[sensor[1], switch[1]],
                    z=[sensor[2], switch[2]], mode="lines",
                    line={"color": color, "width": 2, "dash": "dash"},
                    name=f"{method} tangent ray", legendgroup=method,
                    showlegend=False, hoverinfo="skip",
                ), row=row, col=col)
        scene_name = "scene" if panel == 0 else f"scene{panel + 1}"
        x0, x1, y0, y1 = config.pilot.map_bounds
        figure.layout[scene_name].update(
            xaxis={"range": [x0 * scale, x1 * scale], "title": "x [m]"},
            yaxis={"range": [y0 * scale, y1 * scale], "title": "y [m]"},
            zaxis={"range": [0.0, 520.0], "title": "z [m]"},
            aspectmode="manual", aspectratio={"x": 1.25, "y": 1.0, "z": 0.45},
        )
    figure.update_layout(
        title=(
            f"Unseen {FINAL_TERRAIN_LABEL.replace('_', '-')}: Bellman vs DQN "
            f"trajectories, dx={dx_m:g} m"
        ),
        width=1550, height=1100, margin={"l": 5, "r": 5, "t": 85, "b": 5},
        legend={"orientation": "h", "y": -0.02},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return output


def _load_network(
    checkpoint: Path, device: torch.device,
) -> tuple[TerrainDQN, dict[str, Any], float]:
    started = perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network = TerrainDQN(DQNNetworkConfig.from_dict(payload["network_config"]))
    network.load_state_dict(payload["model_state_dict"])
    network.to(device)
    network.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return network, payload, perf_counter() - started


def _apply_policy_aggregation(
    result: dict[str, Any], *, generated_case_count: int,
) -> dict[str, Any]:
    """Separate mission feasibility from policy success on evaluable cases."""

    records = list(result["records"])
    successful = [record for record in records if record["dqn"]["success"]]
    errors = [float(record["relative_error"]) for record in successful]
    acceptance = evaluate_final_acceptance(
        success_count=len(successful), case_count=len(records),
        successful_relative_errors=errors,
    )
    result["acceptance"] = asdict(acceptance)
    result["metrics"] = {
        "generated_case_count": int(generated_case_count),
        "mission_feasible_case_count": len(records),
        "mission_feasibility_rate": len(records) / generated_case_count,
        "mission_failure_count": int(generated_case_count - len(records)),
        "policy_successful_case_count": len(successful),
        "policy_success_rate_on_feasible_missions": (
            len(successful) / len(records) if records else 0.0
        ),
        "median_relative_error": (
            None if not errors else float(median(errors))
        ),
        "p90_relative_error_higher": (
            None if not errors
            else float(np.percentile(errors, 90.0, method="higher"))
        ),
        "maximum_relative_error": (
            None if not errors else float(max(errors))
        ),
    }
    return result


def evaluate_final_id_000() -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("final-ID evaluation requires CUDA")
    manifest, scenarios, terrain_spec, config = freeze_final_id_000()
    device = torch.device("cuda")
    terrain_specs = {terrain_spec.terrain_instance_id: terrain_spec}
    condition_summaries: list[dict[str, Any]] = []

    for dx_m in SPATIAL_RESOLUTIONS_M:
        tag = _condition_tag(dx_m)
        condition_dir = OUTPUT_ROOT / tag
        result_path = condition_dir / "evaluation.json"
        trajectory_path = condition_dir / "trajectory_comparison_3d.html"
        if result_path.exists() and trajectory_path.exists():
            print(f"Final-ID {tag}: already complete", flush=True)
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            _apply_policy_aggregation(
                existing, generated_case_count=len(scenarios),
            )
            _write_json(result_path, existing)
            condition_summaries.append(existing)
            continue

        print(f"Final-ID {tag}: building Bellman references", flush=True)
        setup_started = perf_counter()
        contexts, mission_failures, runtime_metadata = build_g2_runtimes(
            scenarios,
            terrain_specs=terrain_specs,
            benchmark_config=config,
            spatial_resolution_m=dx_m,
            heading_spacing_deg=HEADING_SPACING_DEG,
            dataset_hash=manifest["manifest_sha256"],
            reference_path=condition_dir / "bellman_references_v1.json",
            observation_device=device,
        )
        setup_wall_sec = perf_counter() - setup_started
        checkpoint = _checkpoint_path(dx_m)
        network, payload, checkpoint_load_sec = _load_network(checkpoint, device)
        if runtime_metadata["compatibility"] != payload["compatibility"]:
            raise RuntimeError(f"test/checkpoint compatibility mismatch at {tag}")

        print(f"Final-ID {tag}: evaluating frozen DQN", flush=True)
        evaluation_started = perf_counter()
        records = evaluate_scenarios(
            network, contexts, device, clear_after_each=True,
            isolate_cache_context_ids=frozenset(map(id, contexts)),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dqn_evaluation_wall_sec = perf_counter() - evaluation_started
        result = {
            "schema": FINAL_ID_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "condition": {
                "spatial_resolution_m": dx_m,
                "heading_spacing_deg": HEADING_SPACING_DEG,
            },
            "checkpoint": manifest["checkpoints"][tag],
            "runtime_metadata": runtime_metadata,
            "mission_failures": mission_failures,
            "timing_sec": {
                "runtime_and_bellman_setup_wall": setup_wall_sec,
                "bellman_reference_runtime_sum": sum(
                    float(record["bellman_runtime_sec"]) for record in records
                ),
                "checkpoint_load": checkpoint_load_sec,
                "dqn_evaluation_wall": dqn_evaluation_wall_sec,
                "dqn_inference_runtime_sum": sum(
                    float(record["dqn"]["inference_runtime_sec"])
                    for record in records
                ),
            },
            "records": records,
            "trajectory_html": trajectory_path.relative_to(
                project_paths.WORKSPACE_ROOT
            ).as_posix(),
        }
        _apply_policy_aggregation(result, generated_case_count=len(scenarios))
        _write_json(result_path, result)
        plot_trajectory_comparison(
            dx_m=dx_m, contexts=contexts, records=records,
            scenarios=scenarios,
            terrain_spec=terrain_spec, config=config, output=trajectory_path,
        )
        condition_summaries.append(result)
        print(
            f"Final-ID {tag}: policy_success="
            f"{result['metrics']['policy_successful_case_count']}/"
            f"{result['metrics']['mission_feasible_case_count']} "
            f"mission_feasible={result['metrics']['mission_feasible_case_count']}/"
            f"{result['metrics']['generated_case_count']} "
            f"median={result['metrics']['median_relative_error']} "
            f"p90={result['metrics']['p90_relative_error_higher']}",
            flush=True,
        )
        del network, payload, contexts
        torch.cuda.empty_cache()

    summary = {
        "schema": FINAL_ID_SCHEMA,
        "status": "complete",
        "manifest_sha256": manifest["manifest_sha256"],
        "terrain_html": TERRAIN_HTML_PATH.relative_to(
            project_paths.WORKSPACE_ROOT
        ).as_posix(),
        "conditions": condition_summaries,
        "interpretation_scope": (
            "one four-case unseen-ID probe; formal ID conclusion requires "
            "all four reserved terrains / sixteen cases"
        ),
    }
    _write_json(SUMMARY_PATH, summary)
    return SUMMARY_PATH


def _assert_same_policy_outputs(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]],
) -> None:
    expected_by_id = {item["scenario_id"]: item for item in expected}
    actual_by_id = {item["scenario_id"]: item for item in actual}
    if set(expected_by_id) != set(actual_by_id):
        raise RuntimeError("warm timing changed the evaluated scenario set")
    for scenario_id, expected_record in expected_by_id.items():
        actual_record = actual_by_id[scenario_id]
        expected_dqn = expected_record["dqn"]
        actual_dqn = actual_record["dqn"]
        if (
            expected_dqn["success"] != actual_dqn["success"]
            or expected_dqn["trajectory"] != actual_dqn["trajectory"]
            or expected_dqn["switching_state_id"]
            != actual_dqn["switching_state_id"]
        ):
            raise RuntimeError(f"warm timing changed policy output: {scenario_id}")
        expected_objective = expected_dqn["attacker_objective"]
        actual_objective = actual_dqn["attacker_objective"]
        if expected_objective is None or actual_objective is None:
            if expected_objective != actual_objective:
                raise RuntimeError(f"warm timing changed objective: {scenario_id}")
        elif not np.isclose(
            float(expected_objective), float(actual_objective),
            rtol=0.0, atol=1.0e-12,
        ):
            raise RuntimeError(f"warm timing changed objective: {scenario_id}")


def retime_final_id_000_warm() -> Path:
    """Measure steady-state DQN evaluation after excluding one-time overhead."""

    if not torch.cuda.is_available():
        raise RuntimeError("warm DQN timing requires CUDA")
    manifest, scenarios, terrain_spec, config = freeze_final_id_000()
    device = torch.device("cuda")
    terrain_specs = {terrain_spec.terrain_instance_id: terrain_spec}
    for dx_m in SPATIAL_RESOLUTIONS_M:
        tag = _condition_tag(dx_m)
        condition_dir = OUTPUT_ROOT / tag
        result_path = condition_dir / "evaluation.json"
        if not result_path.exists():
            raise FileNotFoundError(f"evaluate before warm retiming: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"Final-ID {tag}: preparing warm steady-state timing", flush=True)
        contexts, mission_failures, runtime_metadata = build_g2_runtimes(
            scenarios,
            terrain_specs=terrain_specs,
            benchmark_config=config,
            spatial_resolution_m=dx_m,
            heading_spacing_deg=HEADING_SPACING_DEG,
            dataset_hash=manifest["manifest_sha256"],
            reference_path=condition_dir / "bellman_references_v1.json",
            observation_device=device,
        )
        if mission_failures != result["mission_failures"]:
            raise RuntimeError(f"mission feasibility changed at {tag}")
        network, payload, checkpoint_load_sec = _load_network(
            _checkpoint_path(dx_m), device,
        )
        if runtime_metadata["compatibility"] != payload["compatibility"]:
            raise RuntimeError(f"test/checkpoint compatibility mismatch at {tag}")

        warmup_started = perf_counter()
        warmup_records = evaluate_scenarios(
            network, contexts, device, clear_after_each=False,
        )
        torch.cuda.synchronize(device)
        warmup_wall_sec = perf_counter() - warmup_started
        _assert_same_policy_outputs(result["records"], warmup_records)

        timed_started = perf_counter()
        timed_records = evaluate_scenarios(
            network, contexts, device, clear_after_each=False,
        )
        torch.cuda.synchronize(device)
        steady_state_wall_sec = perf_counter() - timed_started
        _assert_same_policy_outputs(result["records"], timed_records)
        result["timing_sec"]["checkpoint_load_excluded"] = checkpoint_load_sec
        result["timing_sec"]["dqn_cold_evaluation_wall_excluded"] = result[
            "timing_sec"
        ]["dqn_evaluation_wall"]
        result["timing_sec"]["dqn_cold_inference_runtime_sum_excluded"] = result[
            "timing_sec"
        ]["dqn_inference_runtime_sum"]
        result["timing_sec"]["dqn_warmup_wall_excluded"] = warmup_wall_sec
        result["timing_sec"][
            "dqn_warm_steady_state_evaluation_wall"
        ] = steady_state_wall_sec
        result["timing_sec"][
            "dqn_warm_steady_state_inference_runtime_sum"
        ] = sum(
            float(record["dqn"]["inference_runtime_sec"])
            for record in timed_records
        )
        result["timing_policy"] = {
            "reported_dqn_computation_time": (
                "dqn_warm_steady_state_evaluation_wall"
            ),
            "excluded": [
                "scene_graph_build", "Bellman reference construction",
                "checkpoint load", "CUDA first-use initialization",
                "observation/action/cache warm-up",
            ],
            "included": (
                "steady-state greedy policy rollout over every admissible "
                "switching candidate and final authoritative trajectory scoring"
            ),
        }
        _write_json(result_path, result)
        print(
            f"Final-ID {tag}: warm DQN wall={steady_state_wall_sec:.6f} s "
            f"(excluded warmup={warmup_wall_sec:.6f} s)",
            flush=True,
        )
        del contexts, network, payload
        torch.cuda.empty_cache()

    return evaluate_final_id_000()


def render_final_id_000_trajectory_plots() -> list[Path]:
    """Render combined and per-sensor interactive trajectory audit plots."""

    if not torch.cuda.is_available():
        raise RuntimeError("trajectory rendering requires CUDA runtime construction")
    manifest, scenarios, terrain_spec, config = freeze_final_id_000()
    device = torch.device("cuda")
    terrain_specs = {terrain_spec.terrain_instance_id: terrain_spec}
    case_labels = _sensor_case_labels(scenarios)
    outputs: list[Path] = []
    for dx_m in SPATIAL_RESOLUTIONS_M:
        tag = _condition_tag(dx_m)
        condition_dir = OUTPUT_ROOT / tag
        result_path = condition_dir / "evaluation.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing evaluation result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"Final-ID {tag}: rendering full 3D audit", flush=True)
        contexts, mission_failures, _ = build_g2_runtimes(
            scenarios,
            terrain_specs=terrain_specs,
            benchmark_config=config,
            spatial_resolution_m=dx_m,
            heading_spacing_deg=HEADING_SPACING_DEG,
            dataset_hash=manifest["manifest_sha256"],
            reference_path=condition_dir / "bellman_references_v1.json",
            observation_device=device,
        )
        if mission_failures != result["mission_failures"]:
            raise RuntimeError(f"mission feasibility changed at {tag}")
        output = condition_dir / "trajectory_comparison_3d.html"
        plot_trajectory_comparison(
            dx_m=dx_m, contexts=contexts, records=result["records"],
            scenarios=scenarios, terrain_spec=terrain_spec, config=config,
            output=output,
        )
        outputs.append(output)
        records_by_id = {
            record["scenario_id"]: record for record in result["records"]
        }
        contexts_by_id = {
            context.spec.scenario_id: context for context in contexts
        }
        shared_scene = contexts[0].problem.scene
        sensor_outputs: dict[str, str] = {}
        for scenario in scenarios:
            filename_label, display_label = case_labels[scenario.scenario_id]
            sensor_output = condition_dir / f"trajectory_{filename_label}_3d.html"
            plot_single_sensor_trajectory(
                dx_m=dx_m,
                context=contexts_by_id.get(scenario.scenario_id),
                record=records_by_id.get(scenario.scenario_id),
                scenario=scenario,
                sensor_display_name=display_label,
                terrain_spec=terrain_spec,
                config=config,
                shared_scene=shared_scene,
                output=sensor_output,
            )
            outputs.append(sensor_output)
            sensor_outputs[filename_label] = sensor_output.relative_to(
                project_paths.WORKSPACE_ROOT
            ).as_posix()
        result["trajectory_html_by_sensor"] = sensor_outputs
        _write_json(result_path, result)
        del contexts
        torch.cuda.empty_cache()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for condition in summary["conditions"]:
        tag = _condition_tag(
            float(condition["condition"]["spatial_resolution_m"])
        )
        refreshed = json.loads(
            (OUTPUT_ROOT / tag / "evaluation.json").read_text(encoding="utf-8")
        )
        condition["trajectory_html_by_sensor"] = refreshed[
            "trajectory_html_by_sensor"
        ]
    _write_json(SUMMARY_PATH, summary)
    return outputs


def write_final_id_000_result_tables() -> tuple[Path, Path]:
    """Write aggregate CSV and a human-readable aggregate/per-case table."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("final-ID summary does not exist")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    csv_rows: list[dict[str, Any]] = []
    for condition in summary["conditions"]:
        dx_m = float(condition["condition"]["spatial_resolution_m"])
        timing = condition["timing_sec"]
        dqn_time = float(timing["dqn_warm_steady_state_evaluation_wall"])
        bellman_time = float(timing["bellman_reference_runtime_sum"])
        metrics = condition["metrics"]
        csv_rows.append({
            "dx_m": dx_m,
            "mission_feasible": (
                f"{metrics['mission_feasible_case_count']}/"
                f"{metrics['generated_case_count']}"
            ),
            "policy_success_on_feasible": (
                f"{metrics['policy_successful_case_count']}/"
                f"{metrics['mission_feasible_case_count']}"
            ),
            "median_relative_error_pct": 100.0 * float(
                metrics["median_relative_error"]
            ),
            "p90_relative_error_pct": 100.0 * float(
                metrics["p90_relative_error_higher"]
            ),
            "maximum_relative_error_pct": 100.0 * float(
                metrics["maximum_relative_error"]
            ),
            "bellman_solve_sec": bellman_time,
            "dqn_warm_computation_sec": dqn_time,
            "dqn_speedup": bellman_time / dqn_time,
            "accepted": bool(condition["acceptance"]["accepted"]),
        })
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        f"# G2 {FINAL_TERRAIN_LABEL.replace('_', '-')} Results",
        "",
        "DQN time is warm steady-state computation time; setup, checkpoint load, "
        "CUDA initialization, and cache warm-up are excluded.",
        "",
        "| dx | Mission feasible | Policy success | Median error | P90 error | "
        "Max error | Bellman | DQN warm | Speedup | Accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['dx_m']:g} m | {row['mission_feasible']} | "
            f"{row['policy_success_on_feasible']} | "
            f"{row['median_relative_error_pct']:.2f}% | "
            f"{row['p90_relative_error_pct']:.2f}% | "
            f"{row['maximum_relative_error_pct']:.2f}% | "
            f"{row['bellman_solve_sec']:.3f} s | "
            f"{row['dqn_warm_computation_sec']:.3f} s | "
            f"{row['dqn_speedup']:.2f}x | "
            f"{'yes' if row['accepted'] else 'no'} |"
        )
    lines.extend([
        "", "## Per-case accuracy", "",
        "| dx | Case | Bellman J_A | DQN J_A | Relative error | Status |",
        "|---:|:---|---:|---:|---:|:---|",
    ])
    for condition in summary["conditions"]:
        dx_m = float(condition["condition"]["spatial_resolution_m"])
        case_labels = _sensor_case_labels(build_final_id_000()[0])
        for record in condition["records"]:
            display = case_labels[record["scenario_id"]][1]
            lines.append(
                f"| {dx_m:g} m | {display} | {record['bellman_J_A']:.6f} | "
                f"{record['dqn']['attacker_objective']:.6f} | "
                f"{100.0 * record['relative_error']:.2f}% | success |"
            )
        for failure in condition["mission_failures"]:
            lines.append(
                f"| {dx_m:g} m | {failure['visibility_class'].capitalize()} | "
                "-- | -- | -- | mission infeasible |"
            )
    RESULTS_MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RESULTS_CSV_PATH, RESULTS_MARKDOWN_PATH


def write_final_id_aggregate() -> tuple[Path, Path, Path]:
    """Aggregate all four sealed final-ID terrains without rerunning solvers."""

    summaries: dict[int, dict[str, Any]] = {}
    for index in range(FINAL_TERRAIN_COUNT):
        label = f"final_id_{index:03d}"
        path = FINAL_ID_OUTPUT_BASE / label / f"{label}_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing final-ID summary: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") != FINAL_ID_SCHEMA:
            raise RuntimeError(f"unexpected final-ID summary schema: {path}")
        summaries[index] = summary

    aggregate_rows: list[dict[str, Any]] = []
    terrain_rows: list[dict[str, Any]] = []
    for dx_m in SPATIAL_RESOLUTIONS_M:
        conditions: list[tuple[int, dict[str, Any]]] = []
        for index, summary in summaries.items():
            matches = [
                item for item in summary["conditions"]
                if float(item["condition"]["spatial_resolution_m"])
                == float(dx_m)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one dx={dx_m:g} condition for terrain {index:03d}"
                )
            conditions.append((index, matches[0]))

        generated = sum(
            int(item["metrics"]["generated_case_count"])
            for _, item in conditions
        )
        feasible = sum(
            int(item["metrics"]["mission_feasible_case_count"])
            for _, item in conditions
        )
        successful = sum(
            int(item["metrics"]["policy_successful_case_count"])
            for _, item in conditions
        )
        errors = [
            float(record["relative_error"])
            for _, item in conditions
            for record in item["records"]
            if record["dqn"]["success"]
        ]
        acceptance = evaluate_final_acceptance(
            success_count=successful,
            case_count=feasible,
            successful_relative_errors=errors,
        )
        bellman_times = [
            float(item["timing_sec"]["bellman_reference_runtime_sum"])
            for _, item in conditions
        ]
        dqn_times = [
            float(item["timing_sec"]["dqn_warm_steady_state_evaluation_wall"])
            for _, item in conditions
        ]
        candidate_count = sum(
            int(record["dqn"]["candidate_count"])
            for _, item in conditions
            for record in item["records"]
        )
        bellman_total = float(sum(bellman_times))
        dqn_total = float(sum(dqn_times))
        row = {
            "dx_m": float(dx_m),
            "terrain_count": len(conditions),
            "generated_case_count": generated,
            "mission_feasible_case_count": feasible,
            "mission_feasibility_rate": feasible / generated,
            "policy_successful_case_count": successful,
            "policy_success_rate_on_feasible": successful / feasible,
            "median_relative_error": float(median(errors)),
            "p90_relative_error_higher": float(
                np.percentile(errors, 90.0, method="higher")
            ),
            "maximum_relative_error": float(max(errors)),
            "switching_candidate_count_total": candidate_count,
            "bellman_runtime_total_sec": bellman_total,
            "dqn_warm_runtime_total_sec": dqn_total,
            "dqn_warm_msec_per_candidate": (
                1000.0 * dqn_total / candidate_count
            ),
            "dqn_speedup_for_full_bank": bellman_total / dqn_total,
            "dqn_warm_runtime_min_sec": float(min(dqn_times)),
            "dqn_warm_runtime_max_sec": float(max(dqn_times)),
            "terrain_acceptance_count": sum(
                bool(item["acceptance"]["accepted"])
                for _, item in conditions
            ),
            "aggregate_accepted": bool(acceptance.accepted),
        }
        aggregate_rows.append(row)
        for index, item in conditions:
            terrain_rows.append({
                "terrain_index": index,
                "dx_m": float(dx_m),
                "mission_feasible_case_count": int(
                    item["metrics"]["mission_feasible_case_count"]
                ),
                "generated_case_count": int(
                    item["metrics"]["generated_case_count"]
                ),
                "policy_successful_case_count": int(
                    item["metrics"]["policy_successful_case_count"]
                ),
                "median_relative_error": float(
                    item["metrics"]["median_relative_error"]
                ),
                "p90_relative_error_higher": float(
                    item["metrics"]["p90_relative_error_higher"]
                ),
                "switching_candidate_count": sum(
                    int(record["dqn"]["candidate_count"])
                    for record in item["records"]
                ),
                "bellman_runtime_sec": float(
                    item["timing_sec"]["bellman_reference_runtime_sum"]
                ),
                "dqn_warm_runtime_sec": float(
                    item["timing_sec"]["dqn_warm_steady_state_evaluation_wall"]
                ),
                "accepted": bool(item["acceptance"]["accepted"]),
            })

    result = {
        "schema": "p1b-phase5-g2-final-id-aggregate-v1",
        "status": "complete",
        "terrain_indices": list(range(FINAL_TERRAIN_COUNT)),
        "spatial_resolutions_m": list(SPATIAL_RESOLUTIONS_M),
        "checkpoint_seed": 0,
        "checkpoint_episode": 10_000,
        "timing_policy": {
            "dqn": "warm steady-state wall time summed over four terrains",
            "excluded": [
                "scene_graph_build", "Bellman reference construction",
                "checkpoint load", "CUDA first-use initialization",
                "observation/action/cache warm-up",
            ],
            "training_cost": "reported separately; not included here",
        },
        "acceptance_rule": {
            "policy_success_on_feasible": 1.0,
            "median_relative_error_maximum": 0.10,
            "p90_relative_error_maximum": 0.30,
            "p90_method": "higher",
        },
        "aggregate_by_resolution": aggregate_rows,
        "per_terrain_by_resolution": terrain_rows,
    }
    AGGREGATE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(AGGREGATE_JSON_PATH, result)
    with AGGREGATE_CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    lines = [
        "# G2 Final-ID Aggregate: Terrains 000-003",
        "",
        "The bank contains four sealed unseen random-box terrains and sixteen "
        "generated sensor scenarios per resolution. DQN time is warm "
        "steady-state time; setup, checkpoint load, CUDA initialization, and "
        "cache warm-up are excluded.",
        "",
        "| dx | Feasible | Policy success | Median error | P90 error | "
        "Max error | Candidates | Bellman total | DQN warm total | Speedup | "
        "Terrain passes | Bank pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['dx_m']:g} m | "
            f"{row['mission_feasible_case_count']}/"
            f"{row['generated_case_count']} | "
            f"{row['policy_successful_case_count']}/"
            f"{row['mission_feasible_case_count']} | "
            f"{100.0 * row['median_relative_error']:.2f}% | "
            f"{100.0 * row['p90_relative_error_higher']:.2f}% | "
            f"{100.0 * row['maximum_relative_error']:.2f}% | "
            f"{row['switching_candidate_count_total']} | "
            f"{row['bellman_runtime_total_sec']:.3f} s | "
            f"{row['dqn_warm_runtime_total_sec']:.3f} s | "
            f"{row['dqn_speedup_for_full_bank']:.2f}x | "
            f"{row['terrain_acceptance_count']}/{row['terrain_count']} | "
            f"{'yes' if row['aggregate_accepted'] else 'no'} |"
        )
    lines.extend([
        "", "## Per-terrain diagnostic", "",
        "| Terrain | dx | Feasible | Success | Median error | P90 error | "
        "Candidates | Bellman | DQN warm | Pass |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in sorted(
        terrain_rows, key=lambda item: (item["terrain_index"], -item["dx_m"]),
    ):
        lines.append(
            f"| {row['terrain_index']:03d} | {row['dx_m']:g} m | "
            f"{row['mission_feasible_case_count']}/"
            f"{row['generated_case_count']} | "
            f"{row['policy_successful_case_count']}/"
            f"{row['mission_feasible_case_count']} | "
            f"{100.0 * row['median_relative_error']:.2f}% | "
            f"{100.0 * row['p90_relative_error_higher']:.2f}% | "
            f"{row['switching_candidate_count']} | "
            f"{row['bellman_runtime_sec']:.3f} s | "
            f"{row['dqn_warm_runtime_sec']:.3f} s | "
            f"{'yes' if row['accepted'] else 'no'} |"
        )
    AGGREGATE_MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    return AGGREGATE_JSON_PATH, AGGREGATE_CSV_PATH, AGGREGATE_MARKDOWN_PATH


def prepare_final_id_000() -> Path:
    manifest, scenarios, terrain, config = freeze_final_id_000()
    return plot_final_id_000_terrain(manifest, scenarios, terrain, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--terrain-index", type=int, default=0,
        help="reserved final-ID terrain index (0, 1, 2, or 3)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--retime-warm", action="store_true")
    parser.add_argument("--render-trajectories", action="store_true")
    parser.add_argument("--write-tables", action="store_true")
    parser.add_argument("--write-aggregate", action="store_true")
    arguments = parser.parse_args()
    if arguments.write_aggregate:
        for output in write_final_id_aggregate():
            print(output, flush=True)
        return
    configure_final_terrain(arguments.terrain_index)
    terrain_path = prepare_final_id_000()
    print(terrain_path, flush=True)
    if arguments.render_trajectories:
        for output in render_final_id_000_trajectory_plots():
            print(output, flush=True)
    elif arguments.write_tables:
        for output in write_final_id_000_result_tables():
            print(output, flush=True)
    elif arguments.retime_warm:
        print(retime_final_id_000_warm(), flush=True)
    elif arguments.evaluate:
        print(evaluate_final_id_000(), flush=True)
    elif not arguments.prepare_only:
        print("Prepared only; pass --evaluate to run Bellman/DQN tests.", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "build_final_id_000", "evaluate_final_id_000", "freeze_final_id_000",
    "configure_final_terrain",
    "plot_final_id_000_terrain", "prepare_final_id_000",
    "render_final_id_000_trajectory_plots", "retime_final_id_000_warm",
    "write_final_id_000_result_tables", "write_final_id_aggregate",
]
