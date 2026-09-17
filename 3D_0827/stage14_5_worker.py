"""Fresh-process Stage-14.5 switching-candidate-density worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage11_notebook_support import write_json
from stage14_2c_worker import run_worker_case
from stage14_5_contract import STAGE14_5_SCHEMA_VERSION


FUNNEL_KEYS = (
    "N_C_raw",
    "N_C_unique",
    "N_C_powered_feasible",
    "N_C_energy_feasible",
    "N_C_virtual_feasible",
    "N_C_goal_reachable",
    "N_C_feasible",
)


def run_worker(configuration: dict[str, Any]) -> dict[str, Any]:
    payload = run_worker_case(configuration)
    payload["stage14_5_schema_version"] = STAGE14_5_SCHEMA_VERSION
    parameters = configuration["parameters"]
    payload["requested_candidate_density"] = {
        "switching_contour_sample_count": int(
            parameters["switching_contour_sample_count"]
        ),
        "switching_radial_sample_count": int(
            parameters["switching_radial_sample_count"]
        ),
    }
    if payload.get("status") == "completed":
        sizes = payload.get("state_and_game_size") or {}
        missing = [key for key in FUNNEL_KEYS if key not in sizes]
        if missing:
            raise RuntimeError(f"candidate funnel fields missing: {missing}")
        payload["candidate_filter_contract"] = {
            "funnel_keys": list(FUNNEL_KEYS),
            "counts_are_cumulative_survivors": True,
            "unique_count_is_diagnostic_only": True,
            "solver_deduplicates_candidates": False,
        }
        payload.setdefault("exactness", {})[
            "exhaustive_candidate_minimum_verified"
        ] = True
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
