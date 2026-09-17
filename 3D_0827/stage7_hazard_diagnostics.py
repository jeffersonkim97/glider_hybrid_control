"""Generate Stage-7 LOS-gated edge-hazard audit artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from bellman_geometry import GlideEdge
from bellman_state import BellmanState, BellmanStateGrid
from detection_hazard import (
    GlideDetectionHazardModel,
    HazardRateEvaluation,
    evaluate_attacker_hazard_time_objective,
    hazard_to_detection_probability,
)
from edge_hazard import integrate_edge_hazard, precompute_edge_hazards
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from map_geometry import MapBounds
from scenario import Point3D
from terrain_catalog import build_terrain
from visualization import plot_edge_hazard_audit


DEFAULT_STAGE7_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_7_detection_hazard"
)
DEFAULT_HAZARD_BOUNDS = MapBounds(-8.0, 8.0, -8.0, 8.0)


@dataclass(frozen=True)
class SmoothQuadraticHazardField:
    """Analytical lambda=x^2 field for the convergence artifact."""

    def evaluate_rate(
        self,
        position_map: np.ndarray,
        velocity_mps: np.ndarray,
        time_s: float,
    ) -> HazardRateEvaluation:
        rate = float(position_map[0] ** 2)
        return HazardRateEvaluation(
            visible=True,
            sensor_range_m=0.0,
            radial_velocity_mps=0.0,
            cosine_aspect=0.0,
            radar_cross_section=0.0,
            radar_rate_per_s=rate,
            doppler_rate_per_s=0.0,
            total_rate_per_s=rate,
        )


def _state_at(
    grid: BellmanStateGrid,
    x: float,
    y: float,
    altitude: float,
    heading_bin: int,
) -> BellmanState:
    return BellmanState(
        int(round((x - grid.bounds.x_min) / grid.horizontal_spacing_map)),
        int(round((y - grid.bounds.y_min) / grid.horizontal_spacing_map)),
        int(round(
            (altitude - grid.minimum_altitude_map)
            / grid.altitude_spacing_map
        )),
        heading_bin,
    )


def _edge_between(
    grid: BellmanStateGrid,
    source_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    heading_bin: int,
) -> GlideEdge:
    source = _state_at(grid, *source_xyz, heading_bin)
    target = _state_at(grid, *target_xyz, heading_bin)
    source_position = grid.position_map(source)
    target_position = grid.position_map(target)
    horizontal_distance_m = DEFAULT_PHYSICAL_SCALE.distance_m(float(
        np.linalg.norm(target_position[:2] - source_position[:2])
    ))
    duration_s = horizontal_distance_m / DEFAULT_GLIDER.best_glide_speed_mps
    return GlideEdge(
        source_id=grid.encode(source),
        target_id=grid.encode(target),
        source_state=source,
        target_state=target,
        horizontal_distance_m=horizontal_distance_m,
        altitude_loss_m=DEFAULT_PHYSICAL_SCALE.distance_m(
            float(source_position[2] - target_position[2])
        ),
        duration_s=duration_s,
        heading_change_rad=0.0,
    )


def _write_sample_table(path: Path, result: Any) -> None:
    columns = (
        "sample_index",
        "time_fraction",
        "time_s",
        "x_map",
        "y_map",
        "z_map",
        "los_visible",
        "sensor_range_m",
        "radial_velocity_mps",
        "radar_cross_section",
        "radar_rate_per_s",
        "doppler_rate_per_s",
        "total_rate_per_s",
        "quadrature_weight_s",
        "hazard_contribution",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sample in result.samples:
            writer.writerow({
                "sample_index": sample.sample_index,
                "time_fraction": sample.time_fraction,
                "time_s": sample.time_s,
                "x_map": sample.position_map[0],
                "y_map": sample.position_map[1],
                "z_map": sample.position_map[2],
                "los_visible": sample.visible,
                "sensor_range_m": sample.sensor_range_m,
                "radial_velocity_mps": sample.radial_velocity_mps,
                "radar_cross_section": sample.radar_cross_section,
                "radar_rate_per_s": sample.radar_rate_per_s,
                "doppler_rate_per_s": sample.doppler_rate_per_s,
                "total_rate_per_s": sample.total_rate_per_s,
                "quadrature_weight_s": sample.quadrature_weight_s,
                "hazard_contribution": sample.hazard_contribution,
            })


def _print_sample_table(result: Any) -> None:
    print("sample  fraction  visible  lambda[1/s]    weight[s]      dH")
    for sample in result.samples:
        print(
            f"{sample.sample_index:>6d}  {sample.time_fraction:>8.5f}  "
            f"{str(sample.visible):>7s}  {sample.total_rate_per_s:>12.6e}  "
            f"{sample.quadrature_weight_s:>10.6f}  "
            f"{sample.hazard_contribution:>12.6e}"
        )


def generate_stage7_artifacts(
    output_directory: Path = DEFAULT_STAGE7_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Write the crossing-edge figure, audit table, convergence, and summary."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    grid = BellmanStateGrid(
        bounds=DEFAULT_HAZARD_BOUNDS,
        minimum_altitude_map=0.0,
        maximum_altitude_map=2.0,
        altitude_spacing_map=0.1,
    )
    terrain = build_terrain("centered_cube", bounds=DEFAULT_HAZARD_BOUNDS)
    sensor = Point3D(5.0, 0.0, 0.0)
    hazard_model = GlideDetectionHazardModel(terrain, sensor)
    occluded_edge = _edge_between(
        grid,
        (-3.0, 3.0, 1.0),
        (-3.0, 4.0, 0.9),
        2,
    )
    crossing_edge = _edge_between(
        grid,
        (-3.0, 4.0, 1.0),
        (-3.0, 6.0, 0.8),
        2,
    )
    visible_edge = _edge_between(
        grid,
        (3.0, 0.0, 1.0),
        (4.0, 0.0, 0.9),
        0,
    )
    results = {
        "fully_occluded": integrate_edge_hazard(
            occluded_edge,
            grid,
            hazard_model,
            quadrature_resolution=33,
        ),
        "boundary_crossing": integrate_edge_hazard(
            crossing_edge,
            grid,
            hazard_model,
            quadrature_resolution=33,
        ),
        "fully_visible": integrate_edge_hazard(
            visible_edge,
            grid,
            hazard_model,
            quadrature_resolution=33,
        ),
    }
    crossing_result = results["boundary_crossing"]
    if not (
        results["fully_occluded"].hazard
        < crossing_result.hazard
        < results["fully_visible"].hazard
    ):
        raise RuntimeError("LOS-gated diagnostic hazard ordering failed")

    figure = plot_edge_hazard_audit(terrain, sensor, crossing_result)
    figure.write_html(
        output_directory / "los_gated_edge_quadrature.html",
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
    )
    _write_sample_table(
        output_directory / "selected_edge_quadrature_samples.csv",
        crossing_result,
    )

    convergence_grid = BellmanStateGrid(
        bounds=MapBounds(0.0, 1.0, 0.0, 1.0),
        minimum_altitude_map=0.0,
        maximum_altitude_map=0.1,
        altitude_spacing_map=0.1,
    )
    convergence_edge = _edge_between(
        convergence_grid,
        (0.0, 0.0, 0.1),
        (1.0, 0.0, 0.0),
        0,
    )
    analytical_hazard = convergence_edge.duration_s / 3.0
    convergence_rows = []
    for resolution in (4, 8, 16, 32):
        result = integrate_edge_hazard(
            convergence_edge,
            convergence_grid,
            SmoothQuadraticHazardField(),
            quadrature_resolution=resolution,
        )
        convergence_rows.append({
            "quadrature_resolution": resolution,
            "hazard": result.hazard,
            "absolute_error": abs(result.hazard - analytical_hazard),
        })
    convergence = {
        "field": "lambda(x)=x^2 along x in [0,1]",
        "analytical_hazard": analytical_hazard,
        "rows": convergence_rows,
    }
    (output_directory / "quadrature_convergence.json").write_text(
        json.dumps(convergence, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    timing = precompute_edge_hazards(
        ((occluded_edge, crossing_edge, visible_edge),),
        grid,
        hazard_model,
        quadrature_resolution=33,
    )
    objective = evaluate_attacker_hazard_time_objective(
        crossing_result.hazard,
        crossing_result.duration_s,
    )
    summary = {
        "sensor_map": sensor.as_array().tolist(),
        "terrain": "centered_cube",
        "hazard_model": {
            "range_floor_m": hazard_model.parameters.range_floor_m,
            "radar_coefficient": hazard_model.parameters.radar_coefficient,
            "doppler_coefficient": hazard_model.parameters.doppler_coefficient,
            "rcs_min": hazard_model.parameters.rcs_min,
            "rcs_max": hazard_model.parameters.rcs_max,
            "los_query": "TerrainModel.segment_intersects_solid",
            "glide_acoustic_hazard": "excluded",
        },
        "edge_cases": {
            name: {
                "duration_s": result.duration_s,
                "hazard": result.hazard,
                "pod": hazard_to_detection_probability(result.hazard),
                "quadrature_resolution": result.quadrature_resolution,
                "visible_sample_count": result.visible_sample_count,
                "occluded_sample_count": result.occluded_sample_count,
            }
            for name, result in results.items()
        },
        "hazard_ordering": (
            "fully_occluded < boundary_crossing < fully_visible"
        ),
        "selected_edge_objective_decomposition": asdict(objective),
        "runtime": {
            "timing": asdict(timing.timing),
            "visibility_sample_count": timing.visibility_sample_count,
        },
    }
    (output_directory / "detection_hazard_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _print_sample_table(crossing_result)
    print(
        "Stage 7 hazard: "
        f"occluded={results['fully_occluded'].hazard:.8e}, "
        f"crossing={crossing_result.hazard:.8e}, "
        f"visible={results['fully_visible'].hazard:.8e}, "
        f"samples={crossing_result.quadrature_resolution}"
    )
    return summary


if __name__ == "__main__":
    generate_stage7_artifacts()
    print(
        "Generated Stage-7 diagnostics in "
        f"{DEFAULT_STAGE7_FIGURE_DIRECTORY}"
    )
