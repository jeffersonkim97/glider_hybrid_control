"""Build and verify the exact Bellman-to-RL transition contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


THIS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = THIS_DIRECTORY.parent
for path in (REPOSITORY_ROOT, THIS_DIRECTORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl2d.transition_contract import (
    TRANSITION_SCHEMA_VERSION,
    ExactTransitionModel,
    validate_transition_contract,
)
from rl2d.transition_visualization import create_transition_verification_figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _representative_records(
    model: ExactTransitionModel, arrays: Any
) -> list[dict[str, Any]]:
    path = np.asarray(arrays["optimal_glide_trajectory"])
    records = []
    used: set[tuple[int, int]] = set()
    for fraction in (0.2, 0.5, 0.8):
        point = path[int(fraction * (path.shape[0] - 1))]
        zi = int(np.argmin(np.abs(model.z_grid - point[0])))
        hi = int(np.argmin(np.abs(model.h_grid - point[1])))
        if (zi, hi) in used or model.reference_policy[zi, hi] < 0:
            continue
        used.add((zi, hi))
        ai = int(model.reference_policy[zi, hi])
        record = model.transition((zi, hi), ai).as_dict()
        record["normalized_state"] = model.normalize_state(model.state((zi, hi)))
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    reference = THIS_DIRECTORY / "results" / "bellman_reference"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=reference / "bellman_reference.npz"
    )
    parser.add_argument(
        "--reference-summary", type=Path,
        default=reference / "bellman_reference_summary.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=THIS_DIRECTORY / "results" / "transition_contract" / "transition_contract_summary.json",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=THIS_DIRECTORY / "figures" / "transition_contract_verification.png",
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact.resolve()
    reference_summary = json.loads(
        args.reference_summary.read_text(encoding="utf-8")
    )
    artifact_hash = _sha256(artifact)
    if artifact_hash != reference_summary["artifact"]["sha256"]:
        raise RuntimeError("Bellman artifact hash does not match its reference summary.")

    with np.load(artifact) as arrays:
        model = ExactTransitionModel(arrays)
        validation = validate_transition_contract(model)
        if not validation["passed"]:
            raise RuntimeError(
                f"Transition contract validation failed: {validation['failed_checks']}"
            )
        figure_path = create_transition_verification_figure(
            model, arrays, validation, args.figure, dpi=args.dpi
        )
        representative_records = _representative_records(model, arrays)

    summary = {
        "schema_name": "Exact2DBellmanRLTransitionContract",
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "configuration_sha256": reference_summary["configuration_sha256"],
        "source_bellman_artifact": {
            "file": str(artifact),
            "sha256": artifact_hash,
        },
        "state_contract": {
            "axis_order": ["z", "h"],
            "shape": list(model.state_shape),
            "bounds": model.observation_bounds,
            "normalization": "min-max to [0,1] on each spatial axis",
        },
        "action_contract": {
            "type": "masked_discrete",
            "count": model.action_count,
            "fields": [
                "forward_cells", "descent_cells", "speed", "gamma", "duration"
            ],
        },
        "transition_tuple": [
            "state", "action_index", "edge_cost", "next_state", "done", "feasible"
        ],
        "reward_definition": "reward = -edge_cost",
        "infeasible_action_rule": (
            "action mask blocks selection; direct query returns feasible=false, "
            "edge_cost=+inf, reward=-inf, next_state=null"
        ),
        "terminal_rule": (
            "goal-intersecting physical edge returns its continuous truncated "
            "endpoint, done=true, and no discrete next_state_index"
        ),
        "validation": {
            key: value
            for key, value in validation.items()
            if key not in {
                "reconstructed_value", "reconstructed_policy", "value_error",
                "policy_match_map"
            }
        },
        "representative_greedy_transitions": representative_records,
        "verification_figure": str(figure_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_native(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    metrics = summary["validation"]["metrics"]
    print(args.output.resolve())
    print(figure_path)
    print(
        f"states={metrics['state_count']}, actions={metrics['action_count']}, "
        f"feasible={metrics['feasible_transition_count']}, "
        f"terminal={metrics['terminal_transition_count']}"
    )
    print(
        f"value_max_error={metrics['maximum_value_error']:.3e}, "
        f"policy_agreement={100.0 * metrics['greedy_policy_agreement']:.6f}%, "
        f"passed={summary['validation']['passed']}"
    )


if __name__ == "__main__":
    main()

