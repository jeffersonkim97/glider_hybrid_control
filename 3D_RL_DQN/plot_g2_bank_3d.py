"""Interactive 3D audit gallery for the frozen G2 train/validation bank."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import project_paths
from phase4_extended_dataset import build_random_box_terrain
from phase4_g2_dataset import (
    G2_TRAIN,
    G2_VALIDATION,
    G2BenchmarkConfig,
    build_g2_scenario_bank,
    g2_dataset_manifest,
)


OUTPUT_PATH = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "g2_benchmark_bank"
    / "design_v1"
    / "g2_train_validation_bank_3d.html"
)
TRAINING_MAP_OUTPUT_DIR = (
    project_paths.DQN_DIR
    / "runs"
    / "phase4_extended_random_boxes"
    / "training_v2"
    / "g2_generalization"
    / "training_terrain_maps_3d"
)


def plot_g2_bank(output: Path = OUTPUT_PATH) -> Path:
    config = G2BenchmarkConfig()
    splits, terrains = build_g2_scenario_bank(config)
    ordered_ids = [
        key for split in (G2_TRAIN, G2_VALIDATION)
        for key in sorted(
            terrain_id for terrain_id, terrain in terrains.items()
            if terrain.split == split
        )
    ]
    figure = make_subplots(
        rows=3,
        cols=4,
        specs=[[{"type": "scene"} for _ in range(4)] for _ in range(3)],
        subplot_titles=ordered_ids,
        horizontal_spacing=0.015,
        vertical_spacing=0.045,
    )
    all_scenarios = (*splits[G2_TRAIN], *splits[G2_VALIDATION])
    sensor_colors = {"low": "#2ca02c", "mixed": "#ffbf00", "high": "#d62728"}
    for panel, terrain_id in enumerate(ordered_ids):
        row, column = divmod(panel, 4)
        row += 1
        column += 1
        terrain_spec = terrains[terrain_id]
        terrain = build_random_box_terrain(terrain_spec, config.pilot)
        cases = [
            item for item in all_scenarios
            if item.terrain_category == terrain_id
        ]
        mission = cases[0]
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
                opacity=0.72,
                flatshading=True,
                name="random box",
                legendgroup="terrain",
                showlegend=panel == 0 and box_index == 0,
                hovertemplate=f"{terrain_id}<extra></extra>",
            ), row=row, col=column)
        figure.add_trace(go.Scatter3d(
            x=[mission.start_map[0]],
            y=[mission.start_map[1]],
            z=[mission.start_map[2] + 0.04],
            mode="markers",
            marker={"size": 5, "color": "#1f77b4", "symbol": "diamond"},
            name="start",
            legendgroup="start",
            showlegend=panel == 0,
        ), row=row, col=column)
        figure.add_trace(go.Scatter3d(
            x=[mission.goal_map[0]],
            y=[mission.goal_map[1]],
            z=[mission.goal_map[2] + 0.04],
            mode="markers",
            marker={"size": 6, "color": "#9467bd", "symbol": "diamond"},
            name="goal",
            legendgroup="goal",
            showlegend=panel == 0,
        ), row=row, col=column)
        for label in ("low", "mixed", "high"):
            selected = [case for case in cases if case.visibility_class == label]
            figure.add_trace(go.Scatter3d(
                x=[case.sensor_map[0] for case in selected],
                y=[case.sensor_map[1] for case in selected],
                z=[case.sensor_map[2] + 0.04 for case in selected],
                mode="markers",
                marker={"size": 4, "color": sensor_colors[label]},
                name=f"sensor: {label}",
                legendgroup=f"sensor-{label}",
                showlegend=panel == 0,
                text=[
                    f"visibility={case.visibility_fraction:.3f}"
                    for case in selected
                ],
                hovertemplate="%{text}<extra></extra>",
            ), row=row, col=column)
        scene_name = "scene" if panel == 0 else f"scene{panel + 1}"
        figure.layout[scene_name].update(
            xaxis={"range": [-10, 10], "title": "x"},
            yaxis={"range": [-8, 8], "title": "y"},
            zaxis={"range": [0, 5.2], "title": "z"},
            aspectmode="manual",
            aspectratio={"x": 1.25, "y": 1.0, "z": 0.45},
        )
    figure.update_layout(
        title=(
            "G2 frozen random-box train/validation bank<br>"
            "<sup>Rows 1–2: training (8 terrains); row 3: held-out validation "
            "(4 terrains). Final test is not instantiated.</sup>"
        ),
        width=1700,
        height=1320,
        margin={"l": 5, "r": 5, "t": 95, "b": 5},
        legend={"orientation": "h", "y": -0.015},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    return output


def plot_g2_training_terrain_maps(
    output_dir: Path = TRAINING_MAP_OUTPUT_DIR,
) -> list[Path]:
    """Write one self-contained interactive HTML map per G2 training terrain."""

    config = G2BenchmarkConfig()
    splits, terrain_specs = build_g2_scenario_bank(config)
    dataset_hash = g2_dataset_manifest(config)["manifest_sha256"]
    meters_per_unit = float(config.pilot.meters_per_map_unit)
    x_min, x_max, y_min, y_max = config.pilot.map_bounds
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    sensor_colors = {"low": "#2ca02c", "mixed": "#ffbf00", "high": "#d62728"}

    terrain_ids = sorted(
        terrain_id
        for terrain_id, terrain_spec in terrain_specs.items()
        if terrain_spec.split == G2_TRAIN
    )
    for terrain_id in terrain_ids:
        terrain_spec = terrain_specs[terrain_id]
        terrain = build_random_box_terrain(terrain_spec, config.pilot)
        cases = sorted(
            (
                case
                for case in splits[G2_TRAIN]
                if case.terrain_category == terrain_id
            ),
            key=lambda case: (
                ("low", "mixed", "high").index(case.visibility_class),
                case.scenario_id,
            ),
        )
        if len(cases) != 4:
            raise RuntimeError(f"Expected four training cases for {terrain_id}")
        mission = cases[0]
        figure = go.Figure()

        for box_index, box in enumerate(terrain.obstacle_boxes()):
            mesh = box.mesh()
            vertices_m = mesh.vertices * meters_per_unit
            figure.add_trace(go.Mesh3d(
                x=vertices_m[:, 0],
                y=vertices_m[:, 1],
                z=vertices_m[:, 2],
                i=mesh.triangles[:, 0],
                j=mesh.triangles[:, 1],
                k=mesh.triangles[:, 2],
                color="#7f8c8d",
                opacity=0.72,
                flatshading=True,
                name="terrain boxes",
                legendgroup="terrain",
                showlegend=box_index == 0,
                hovertemplate=(
                    f"{terrain_id}<br>box {box_index}<extra></extra>"
                ),
            ))

        figure.add_trace(go.Mesh3d(
            x=[x_min * meters_per_unit, x_max * meters_per_unit,
               x_max * meters_per_unit, x_min * meters_per_unit],
            y=[y_min * meters_per_unit, y_min * meters_per_unit,
               y_max * meters_per_unit, y_max * meters_per_unit],
            z=[0.0, 0.0, 0.0, 0.0],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#d9e6f2", opacity=0.18, flatshading=True,
            name="ground plane", hoverinfo="skip", showlegend=True,
        ))

        start_m = [value * meters_per_unit for value in mission.start_map]
        goal_m = [value * meters_per_unit for value in mission.goal_map]
        figure.add_trace(go.Scatter3d(
            x=[start_m[0]], y=[start_m[1]], z=[start_m[2] + 4.0],
            mode="markers+text",
            marker={"size": 8, "color": "#0072B2", "symbol": "diamond"},
            text=["START"], textposition="top center",
            name="start",
            hovertemplate=(
                "START<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m"
                "<extra></extra>"
            ),
        ))
        figure.add_trace(go.Scatter3d(
            x=[goal_m[0]], y=[goal_m[1]], z=[goal_m[2] + 4.0],
            mode="markers+text",
            marker={"size": 9, "color": "#9467bd", "symbol": "diamond"},
            text=["GOAL"], textposition="top center",
            name="goal",
            hovertemplate=(
                "GOAL<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m"
                "<extra></extra>"
            ),
        ))

        for visibility_class in ("low", "mixed", "high"):
            selected = [
                case for case in cases
                if case.visibility_class == visibility_class
            ]
            labels = [
                (
                    visibility_class.upper()
                    if len(selected) == 1
                    else f"{visibility_class.upper()} {index + 1}"
                )
                for index in range(len(selected))
            ]
            figure.add_trace(go.Scatter3d(
                x=[case.sensor_map[0] * meters_per_unit for case in selected],
                y=[case.sensor_map[1] * meters_per_unit for case in selected],
                z=[case.sensor_map[2] * meters_per_unit + 4.0 for case in selected],
                mode="markers+text",
                marker={"size": 7, "color": sensor_colors[visibility_class]},
                text=labels,
                textposition="top center",
                name=f"sensor: {visibility_class}",
                customdata=[
                    [case.visibility_fraction, case.scenario_id]
                    for case in selected
                ],
                hovertemplate=(
                    "%{text} SENSOR<br>x=%{x:.1f} m<br>y=%{y:.1f} m"
                    "<br>z=%{z:.1f} m<br>visibility=%{customdata[0]:.3f}"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            ))

        figure.update_layout(
            title=(
                f"G2 training terrain: {terrain_id}<br>"
                f"<sup>terrain seed={terrain_spec.generation_seed}; "
                f"dataset={dataset_hash[:12]}; one mission, four sensor scenarios"
                "</sup>"
            ),
            width=1150,
            height=820,
            margin={"l": 0, "r": 0, "t": 90, "b": 0},
            legend={"orientation": "h", "y": 0.01, "x": 0.02},
            scene={
                "xaxis": {
                    "range": [x_min * meters_per_unit, x_max * meters_per_unit],
                    "title": "x [m]",
                },
                "yaxis": {
                    "range": [y_min * meters_per_unit, y_max * meters_per_unit],
                    "title": "y [m]",
                },
                "zaxis": {"range": [0.0, 520.0], "title": "z [m]"},
                "aspectmode": "manual",
                "aspectratio": {"x": 1.25, "y": 1.0, "z": 0.45},
                "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.0}},
            },
        )
        output = output_dir / f"{terrain_id}_start_goal_sensors_3d.html"
        figure.write_html(
            output,
            include_plotlyjs=True,
            full_html=True,
            config={"responsive": True, "displaylogo": False},
        )
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    print(plot_g2_bank())
    for training_map in plot_g2_training_terrain_maps():
        print(training_map)


__all__ = ["plot_g2_bank", "plot_g2_training_terrain_maps"]
