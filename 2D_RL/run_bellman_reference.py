"""Command-line entry point for the exact 2D Bellman RL reference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = THIS_DIRECTORY.parent
for path in (REPOSITORY_ROOT, THIS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl2d.bellman_reference import generate_bellman_reference
from rl2d.reference_configuration import (
    REFERENCE_SENSOR_Z,
    build_reference_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the authoritative 2D Bellman reference for Discrete SAC."
    )
    parser.add_argument("--sensor-z", type=float, default=REFERENCE_SENSOR_Z)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIRECTORY / "results" / "bellman_reference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = build_reference_configuration(REPOSITORY_ROOT, args.sensor_z)
    try:
        result = generate_bellman_reference(reference, args.output_dir)
    finally:
        logging_utilities = reference.configuration_bundle["primary_result"][
            "logging_utilities"
        ]
        logging_utilities["close_logger"](logging_utilities["logger"])

    summary = result["summary"]
    mission = summary["mission"]
    validation = summary["validation"]
    print(f"Reference: {result['artifact_path']}")
    print(f"Summary:   {result['summary_path']}")
    print(f"Config SHA-256: {summary['configuration_sha256']}")
    print(
        "Mission: "
        f"cost={mission['mission_objective']:.12g}, "
        f"PoD={mission['mission_pod']:.12g}, "
        f"time={mission['mission_time']:.12g} s"
    )
    print(
        "Validation: "
        f"passed={validation['passed']}, "
        "value max diff="
        f"{validation['metrics']['production_export_value_maximum_difference']:.3e}, "
        "Bellman max residual="
        f"{validation['metrics']['maximum_bellman_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
