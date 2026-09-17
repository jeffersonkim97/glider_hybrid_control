"""Fresh-process Stage-14.4 worker with realized heading-grid metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attacker_best_response import DEFAULT_GRAPH_BOUNDS
from stage11_notebook_support import write_json
from stage14_2c_worker import _build_case, run_worker_case
from stage14_4_contract import STAGE14_4_SCHEMA_VERSION


def run_worker(configuration: dict) -> dict:
    payload = run_worker_case(configuration)
    stage_config, _candidates, _initial_id = _build_case(configuration)
    grid = stage_config.discretization.build_bellman_grid(DEFAULT_GRAPH_BOUNDS)
    realized_spacing_deg = 360.0 / float(grid.heading_bin_count)
    payload["stage14_4_schema_version"] = STAGE14_4_SCHEMA_VERSION
    realized_heading_grid = {
        "requested_heading_spacing_deg": float(
            configuration["parameters"]["heading_spacing_deg"]
        ),
        "heading_bin_count": int(grid.heading_bin_count),
        "realized_heading_spacing_deg": realized_spacing_deg,
        "maximum_heading_quantization_error_deg": realized_spacing_deg / 2.0,
        "period_deg": 360.0,
    }
    payload["realized_heading_grid"] = realized_heading_grid
    payload["realized_grid"] = {"heading": realized_heading_grid}
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
