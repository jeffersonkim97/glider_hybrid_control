"""Fresh-process dispatch worker for Stage 14.2 benchmark repetitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep
from typing import Any

from stage11_notebook_support import write_json
from stage14_1_profile_worker import (
    run_canonical_profile,
    run_infeasible_profile,
)


def run_worker_case(configuration: dict[str, Any]) -> dict[str, Any]:
    """Execute one declared fixture without changing exact-solver behavior."""
    mode = str(configuration["worker_mode"])
    if mode == "canonical":
        payload = run_canonical_profile()
    elif mode == "infeasible":
        payload = run_infeasible_profile()
    elif mode == "controlled_timeout":
        delay_s = float(configuration["parameters"]["delay_s"])
        if delay_s <= 0.0:
            raise ValueError("controlled timeout delay_s must be positive")
        sleep(delay_s)
        payload = {
            "status": "completed",
            "case_id": "controlled_timeout_unexpected_completion",
            "fixture_delay_s": delay_s,
        }
    else:
        raise ValueError(f"unsupported Stage-14.2 worker_mode: {mode!r}")
    payload["stage14_2_case_id"] = configuration["case_id"]
    payload["stage14_2_repetition_id"] = configuration["repetition_id"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text(encoding="utf-8"))
    payload = run_worker_case(configuration)
    write_json(arguments.output, payload)
    print(f"{configuration['repetition_id']}: {payload['status']}")


if __name__ == "__main__":
    main()
