"""Fresh-process Stage-14.3 worker with explicit realized-grid metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attacker_best_response import DEFAULT_GRAPH_BOUNDS
from stage11_notebook_support import write_json
from stage14_2c_worker import _build_case, run_worker_case
from stage14_3_contract import STAGE14_3_SCHEMA_VERSION


def _axis_record(values: np.ndarray, requested_spacing_map: float) -> dict:
    realized = float(values[1] - values[0]) if len(values) > 1 else None
    return {
        "count": int(len(values)),
        "minimum_map": float(values[0]),
        "maximum_map": float(values[-1]),
        "requested_spacing_map": float(requested_spacing_map),
        "realized_spacing_map": realized,
        "coordinates_map": list(map(float, values)),
    }


def run_worker(configuration: dict) -> dict:
    payload = run_worker_case(configuration)
    stage_config, _candidates, _initial_id = _build_case(configuration)
    discretization = stage_config.discretization
    grid = discretization.build_bellman_grid(DEFAULT_GRAPH_BOUNDS)
    payload["stage14_3_schema_version"] = STAGE14_3_SCHEMA_VERSION
    payload["realized_grid"] = {
        "requested_spatial_resolution_m": float(
            configuration["parameters"]["spatial_resolution_m"]
        ),
        "meters_per_map_unit": float(stage_config.physical_scale.meters_per_map_unit),
        "x": _axis_record(grid.x_coordinates, discretization.horizontal_spacing_map),
        "y": _axis_record(grid.y_coordinates, discretization.horizontal_spacing_map),
        "altitude": _axis_record(
            grid.altitude_coordinates, discretization.altitude_spacing_map
        ),
        "heading_bin_count": int(grid.heading_bin_count),
        "heading_spacing_deg": 360.0 / float(grid.heading_bin_count),
        "motion_primitive_radius": int(grid.motion_primitive_radius),
        "motion_primitive_step_cells": int(grid.motion_primitive_step_cells),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text(encoding="utf-8"))
    payload = run_worker(configuration)
    write_json(arguments.output, payload)
    print(f"{configuration['repetition_id']}: {payload['status']}")


if __name__ == "__main__":
    main()
