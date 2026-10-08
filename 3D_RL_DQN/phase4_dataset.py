"""Approved Phase 16.4 simple-terrain and sensing-scenario manifest.

The manifest is deliberately data-only.  Terrain/scenario identifiers are
provenance and replay diagnostics; they are never appended to a DQN observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

import project_paths  # noqa: F401
from terrain_catalog import build_terrain


DATASET_SCHEMA = "p1b-phase4-dataset-v1"
TRAIN_SIMPLE = "TRAIN_SIMPLE"
VALIDATION_SIMPLE = "VALIDATION_SIMPLE"
RESERVED_PHASE5_TEST = "RESERVED_PHASE5_TEST"

TRAIN_TERRAINS = (
    "centered_cube",
    "centered_cube_half_height",
    "offset_cube_left",
)
VALIDATION_TERRAINS = (
    "offset_cube_right",
    "stepped_pyramid",
)

# User-approved common 25 m-lattice sensing positions.  They span the shared
# obstacle-free Defender region while retaining the legacy (5, 0, 0) position.
SENSOR_X_MAP = (3.25, 5.0, 6.75)
SENSOR_Y_MAP = (-3.75, 0.0, 3.75)
SENSOR_Z_MAP = 0.0


@dataclass(frozen=True)
class Phase4ScenarioSpec:
    scenario_id: str
    split: str
    terrain_category: str
    sensor_map: tuple[float, float, float]
    terrain_hash: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sensor_map"] = list(self.sensor_map)
        return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def terrain_definition(category: str) -> dict[str, Any]:
    terrain = build_terrain(category)
    return {
        "category": str(category),
        "bounds": asdict(terrain.bounds),
        "ground_z": float(terrain.ground_z),
        "boxes": [asdict(box) for box in terrain.obstacle_boxes()],
    }


def terrain_definition_hash(category: str) -> str:
    encoded = _canonical_json(terrain_definition(category)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coordinate_tag(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    magnitude = f"{abs(float(value)):.2f}".replace(".", "p")
    return f"{sign}{magnitude}"


def scenario_id(
    split: str, terrain_category: str, sensor_map: tuple[float, float, float],
) -> str:
    x, y, z = sensor_map
    return (
        f"{split.lower()}__{terrain_category}__"
        f"x{_coordinate_tag(x)}_y{_coordinate_tag(y)}_z{_coordinate_tag(z)}"
    )


def _specs(split: str, terrains: Iterable[str]) -> tuple[Phase4ScenarioSpec, ...]:
    result: list[Phase4ScenarioSpec] = []
    for terrain_category in terrains:
        digest = terrain_definition_hash(terrain_category)
        for x_map in SENSOR_X_MAP:
            for y_map in SENSOR_Y_MAP:
                sensor = (float(x_map), float(y_map), float(SENSOR_Z_MAP))
                result.append(Phase4ScenarioSpec(
                    scenario_id=scenario_id(split, terrain_category, sensor),
                    split=split,
                    terrain_category=terrain_category,
                    sensor_map=sensor,
                    terrain_hash=digest,
                ))
    return tuple(result)


def approved_phase4_splits() -> dict[str, tuple[Phase4ScenarioSpec, ...]]:
    splits = {
        TRAIN_SIMPLE: _specs(TRAIN_SIMPLE, TRAIN_TERRAINS),
        VALIDATION_SIMPLE: _specs(VALIDATION_SIMPLE, VALIDATION_TERRAINS),
        RESERVED_PHASE5_TEST: (),
    }
    validate_split_integrity(splits)
    return splits


def validate_split_integrity(
    splits: dict[str, tuple[Phase4ScenarioSpec, ...]],
) -> None:
    expected = {TRAIN_SIMPLE, VALIDATION_SIMPLE, RESERVED_PHASE5_TEST}
    if set(splits) != expected:
        raise ValueError(f"dataset splits must be exactly {sorted(expected)}")
    seen_ids: dict[str, str] = {}
    seen_cases: dict[tuple[str, tuple[float, float, float]], str] = {}
    for split, specs in splits.items():
        for spec in specs:
            if spec.split != split:
                raise ValueError(f"scenario {spec.scenario_id} has the wrong split")
            previous = seen_ids.setdefault(spec.scenario_id, split)
            if previous != split:
                raise ValueError(f"scenario ID leakage: {spec.scenario_id}")
            case = (spec.terrain_hash, spec.sensor_map)
            previous_case = seen_cases.setdefault(case, split)
            if previous_case != split:
                raise ValueError(
                    f"terrain/sensor leakage between {previous_case} and {split}: {case}"
                )
    if not splits[TRAIN_SIMPLE] or not splits[VALIDATION_SIMPLE]:
        raise ValueError("training and validation splits must both be nonempty")


def approved_dataset_manifest() -> dict[str, Any]:
    splits = approved_phase4_splits()
    terrain_ids = tuple(dict.fromkeys(TRAIN_TERRAINS + VALIDATION_TERRAINS))
    manifest = {
        "schema": DATASET_SCHEMA,
        "design_status": "user_approved_2026-10-03",
        "terrain_definitions": {
            category: {
                **terrain_definition(category),
                "sha256": terrain_definition_hash(category),
            }
            for category in terrain_ids
        },
        "sensor_distribution": {
            "x_map": list(SENSOR_X_MAP),
            "y_map": list(SENSOR_Y_MAP),
            "z_map": SENSOR_Z_MAP,
            "sampling": "uniform terrain, then uniform sensor position",
            "neural_input_policy": (
                "scenario and terrain IDs are metadata only; sensing geometry enters "
                "through the approved Phase 16.2 hazard channels"
            ),
        },
        "splits": {
            name: [spec.as_dict() for spec in specs]
            for name, specs in splits.items()
        },
        "counts": {
            "training_terrains": len(TRAIN_TERRAINS),
            "validation_terrains": len(VALIDATION_TERRAINS),
            "sensors_per_terrain": len(SENSOR_X_MAP) * len(SENSOR_Y_MAP),
            "training_scenarios": len(splits[TRAIN_SIMPLE]),
            "validation_scenarios": len(splits[VALIDATION_SIMPLE]),
            "reserved_phase5_scenarios": 0,
        },
        "leakage_check": {
            "passed": True,
            "identity": "terrain definition hash plus exact sensor position",
        },
        "fixed_non_target_variables": [
            "attacker start", "goal", "glider physics", "objective",
            "detection model", "terminal rule", "transition feasibility",
        ],
        "phase5_policy": (
            "namespace reserved but empty; complex unseen terrain composition requires "
            "a separate Phase 16.5 approval"
        ),
    }
    encoded = _canonical_json(manifest).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def specs_from_manifest(
    manifest: dict[str, Any], split: str,
) -> tuple[Phase4ScenarioSpec, ...]:
    values = manifest["splits"][split]
    return tuple(Phase4ScenarioSpec(
        scenario_id=str(item["scenario_id"]),
        split=str(item["split"]),
        terrain_category=str(item["terrain_category"]),
        sensor_map=tuple(float(value) for value in item["sensor_map"]),
        terrain_hash=str(item["terrain_hash"]),
    ) for item in values)


def common_sensor_is_outside_terrain(spec: Phase4ScenarioSpec) -> bool:
    terrain = build_terrain(spec.terrain_category)
    point = np.asarray(spec.sensor_map, dtype=float)
    return not terrain.contains_solid(point)


__all__ = [
    "DATASET_SCHEMA", "RESERVED_PHASE5_TEST", "SENSOR_X_MAP", "SENSOR_Y_MAP",
    "SENSOR_Z_MAP", "TRAIN_SIMPLE", "TRAIN_TERRAINS", "VALIDATION_SIMPLE",
    "VALIDATION_TERRAINS", "Phase4ScenarioSpec", "approved_dataset_manifest",
    "approved_phase4_splits", "common_sensor_is_outside_terrain", "scenario_id",
    "specs_from_manifest", "terrain_definition", "terrain_definition_hash",
    "validate_split_integrity",
]
