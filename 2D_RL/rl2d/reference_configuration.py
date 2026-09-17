"""Frozen configuration contract for the first 2D RL reference case.

The numerical configuration is owned by :mod:`p1b_4D.configuration`.  This
module deliberately imports that configuration instead of copying its values.
It only validates the assumptions that identify the first RL benchmark and
computes a stable hash of the numerical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from p1b_4D.configuration import build_configuration_bundle


REFERENCE_SCENARIO_ID = "canonical_single_hill_default"
REFERENCE_SENSOR_Z = 4000.0
REFERENCE_TRANSITION_MODEL = "successor_grid_physical_edge"

_NUMERICAL_CONFIG_KEYS = (
    "environment_config",
    "vehicle_config",
    "sensor_config",
    "cost_config",
    "bellman_config",
    "attacker_solver_config",
    "defender_config",
    "validation_config",
)


@dataclass(frozen=True)
class ReferenceConfiguration:
    """Validated handle to the authoritative 2D configuration."""

    scenario_id: str
    sensor_z: float
    configuration_bundle: dict[str, Any]
    configuration_hash: str
    configuration_snapshot: dict[str, Any]


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def numerical_configuration_snapshot(
    configuration_bundle: dict[str, Any], sensor_z: float
) -> dict[str, Any]:
    """Return only inputs that can change the reference computation."""

    primary = configuration_bundle["primary_result"]
    return {
        "scenario_id": REFERENCE_SCENARIO_ID,
        "sensor_z": float(sensor_z),
        **{
            key: _json_native(primary[key])
            for key in _NUMERICAL_CONFIG_KEYS
        },
    }


def configuration_sha256(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reference_configuration(
    repository_root: Path, sensor_z: float = REFERENCE_SENSOR_Z
) -> ReferenceConfiguration:
    """Build and validate the unmodified production configuration."""

    repository_root = Path(repository_root).resolve()
    bundle = build_configuration_bundle(repository_root)
    if not bundle["status"]["success"]:
        raise RuntimeError(bundle["status"]["message"])

    primary = bundle["primary_result"]
    model = primary["attacker_solver_config"]["transition_model"]
    if model != REFERENCE_TRANSITION_MODEL:
        raise RuntimeError(
            "The RL reference must mirror the production physical-successor "
            f"solver, but the active model is {model!r}."
        )

    terrain = primary["environment_config"]["terrain"]
    hills = terrain["hills"]
    if len(hills) != 1:
        raise RuntimeError(
            f"Reference scenario requires one hill; configuration has {len(hills)}."
        )

    bounds = primary["defender_config"]["continuous_search_bounds"]
    if not bounds["z_sensor_min"] <= sensor_z <= bounds["z_sensor_max"]:
        raise ValueError("Reference sensor position lies outside Defender bounds.")

    snapshot = numerical_configuration_snapshot(bundle, sensor_z)
    return ReferenceConfiguration(
        scenario_id=REFERENCE_SCENARIO_ID,
        sensor_z=float(sensor_z),
        configuration_bundle=bundle,
        configuration_hash=configuration_sha256(snapshot),
        configuration_snapshot=snapshot,
    )

