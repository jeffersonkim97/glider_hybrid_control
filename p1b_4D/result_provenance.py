"""Reproducibility metadata shared by paper-facing result writers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROVENANCE_SCHEMA_VERSION = "1.0.0"
PROVENANCE_CONFIG_KEYS = (
    "environment_config",
    "vehicle_config",
    "sensor_config",
    "cost_config",
    "bellman_config",
    "attacker_solver_config",
    "defender_config",
    "validation_config",
    "global_random_seed",
)


def build_result_provenance(
    configuration_bundle: dict[str, Any],
    *,
    script_identifier: str,
    continuous_validation: dict[str, Any] | None = None,
    script_version: str = "1.0.0",
) -> dict[str, Any]:
    """Build a stable configuration fingerprint and execution provenance."""
    if not script_identifier:
        raise ValueError("script_identifier must be nonempty")
    primary = configuration_bundle["primary_result"]
    snapshot = {
        key: primary[key]
        for key in PROVENANCE_CONFIG_KEYS
        if key in primary
    }
    canonical_snapshot = _json_native(snapshot)
    encoded = json.dumps(
        canonical_snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    configuration_hash = hashlib.sha256(encoded).hexdigest()

    paths = primary["project_paths"]
    project_root = Path(paths.project_root).resolve()
    source_control = _source_control_state(project_root)
    grid = primary["environment_config"]["grid"]
    bellman_search = primary["bellman_config"]["search_options"]
    tolerances = primary["validation_config"]

    return {
        "schema_name": "ResultProvenance",
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_identifier": script_identifier,
        "script_version": script_version,
        "source_control": source_control,
        "configuration_hash_sha256": configuration_hash,
        "transition_model": primary["attacker_solver_config"]["transition_model"],
        "resolution": {
            "spatial": {
                "z_count": int(grid["z_count"]),
                "h_count": int(grid["h_count"]),
                "z_spacing": float(grid["z_spacing"]),
                "h_spacing": float(grid["h_spacing"]),
            },
            "action": {
                "v_count": int(grid["v_count"]),
                "gamma_count": int(grid["gamma_count"]),
            },
        },
        "numerical_validation": {
            "segment_check_count": int(bellman_search["segment_check_count"]),
            "terrain_tolerance": float(tolerances["terrain_tolerance"]),
            "los_tolerance": float(tolerances["los_tolerance"]),
            "goal_radius": float(tolerances["goal_radius"]),
            "dynamic_tolerance": float(tolerances["dynamic_tolerance"]),
            "solver_tolerance": float(tolerances["solver_tolerance"]),
            "objective_tolerance": float(tolerances["objective_tolerance"]),
        },
        "continuous_validation": _continuous_validation_summary(
            continuous_validation
        ),
        "random_seed": int(primary["global_random_seed"]),
    }


def provenance_from_evaluation(
    configuration_bundle: dict[str, Any],
    evaluation_result: dict[str, Any],
    *,
    script_identifier: str,
    script_version: str = "1.0.0",
) -> dict[str, Any]:
    """Build provenance using an evaluation's continuous-replay result."""
    best = evaluation_result["primary_result"]["best_found_attacker_response"]
    return build_result_provenance(
        configuration_bundle,
        script_identifier=script_identifier,
        script_version=script_version,
        continuous_validation=best.get("continuous_replay_validation"),
    )


def _source_control_state(project_root: Path) -> dict[str, Any]:
    commit = _git_output(project_root, "rev-parse", "HEAD")
    status = _git_output(
        project_root, "status", "--porcelain", "--untracked-files=normal"
    )
    dirty = None if status is None else bool(status.strip())
    if commit is None:
        identifier = "git-unavailable"
    elif dirty is None:
        identifier = f"{commit}-dirty-unknown"
    else:
        identifier = f"{commit}-dirty" if dirty else commit
    return {
        "source_commit": commit,
        "working_tree_dirty": dirty,
        "working_tree_identifier": identifier,
    }


def _git_output(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _continuous_validation_summary(
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    if validation is None:
        return {
            "checked": False,
            "feasible": None,
            "violation": None,
            "reached_goal": None,
            "goal_miss": None,
        }
    return {
        "checked": True,
        "feasible": bool(validation["feasible"]),
        "violation": validation.get("violation"),
        "reached_goal": bool(validation["reached_goal"]),
        "goal_miss": float(validation["goal_miss"]),
    }


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
