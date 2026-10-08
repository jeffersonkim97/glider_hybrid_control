"""Create terrain-only interactive 3D plots for Phase 16.4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import project_paths
from phase4_dataset import TRAIN_TERRAINS
from terrain_catalog import TERRAIN_LABELS, build_terrain


STEPPED_PYRAMID = "stepped_pyramid"
OUTPUT_DIR = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_multi_terrain_dqn"
    / "terrain_only_3d"
)


def add_mesh(
    figure: go.Figure,
    mesh: object,
    *,
    name: str,
    color: str,
    opacity: float,
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={
            "ambient": 0.55,
            "diffuse": 0.75,
            "specular": 0.15,
            "roughness": 0.8,
        },
        lightposition={"x": 100, "y": -120, "z": 180},
        hovertemplate=f"{name}<extra></extra>",
    ))


def write_plot(category: str, split: str) -> Path:
    terrain = build_terrain(category)
    figure = go.Figure()
    add_mesh(
        figure,
        terrain.ground_mesh(),
        name="ground plane",
        color="#DDEBF7",
        opacity=0.65,
    )
    add_mesh(
        figure,
        terrain.obstacle_mesh(),
        name=TERRAIN_LABELS[category],
        color="#707070" if category != STEPPED_PYRAMID else "#8C6D4F",
        opacity=0.96,
    )

    split_label = (
        "DQN training terrain"
        if split == "TRAIN_SIMPLE"
        else "Held-out validation terrain"
    )
    figure.update_layout(
        title={
            "text": f"{TERRAIN_LABELS[category]}<br><sup>{split_label}</sup>",
            "x": 0.5,
        },
        scene={
            "xaxis": {
                "title": "x [map unit]",
                "range": [terrain.bounds.x_min, terrain.bounds.x_max],
            },
            "yaxis": {
                "title": "y [map unit]",
                "range": [terrain.bounds.y_min, terrain.bounds.y_max],
            },
            "zaxis": {
                "title": "z [map unit]",
                "range": [terrain.ground_z, 5.0],
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 0.8, "z": 0.42},
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.15}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
        template="plotly_white",
        uirevision=f"terrain-only-{category}",
    )
    output = OUTPUT_DIR / f"terrain_only_{category}_3d.html"
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        auto_open=False,
    )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for category in (*TRAIN_TERRAINS, STEPPED_PYRAMID):
        split = "TRAIN_SIMPLE" if category in TRAIN_TERRAINS else "VALIDATION_SIMPLE"
        output = write_plot(category, split)
        records.append({
            "terrain_category": category,
            "dataset_split": split,
            "training_used": category in TRAIN_TERRAINS,
            "output_html": str(output).replace("\\", "/"),
        })
        print(output)

    manifest = {
        "purpose": "terrain-only interactive 3D plots for Phase 16.4",
        "training_terrains": list(TRAIN_TERRAINS),
        "stepped_pyramid_role": "held-out VALIDATION_SIMPLE terrain",
        "plots": records,
    }
    manifest_path = OUTPUT_DIR / "terrain_only_3d_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
