"""Generate the Step-1 visual verification sheet from saved artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = THIS_DIRECTORY.parent
for path in (REPOSITORY_ROOT, THIS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl2d.reference_visualization import create_reference_verification_figure


def parse_args() -> argparse.Namespace:
    default_results = THIS_DIRECTORY / "results" / "bellman_reference"
    parser = argparse.ArgumentParser(description="Visualize the exact Bellman reference.")
    parser.add_argument(
        "--artifact", type=Path, default=default_results / "bellman_reference.npz"
    )
    parser.add_argument(
        "--summary", type=Path,
        default=default_results / "bellman_reference_summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=THIS_DIRECTORY / "figures" / "bellman_reference_verification.png",
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = create_reference_verification_figure(
        args.artifact, args.summary, args.output, dpi=args.dpi
    )
    print(output)


if __name__ == "__main__":
    main()

