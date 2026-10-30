"""Frozen train/validation bank and sealed-test contract for G2.

G2 deliberately uses explicit, split-specific seed ranges.  Unlike the G1
pilot, increasing the number of training terrains cannot silently reassign a
former validation terrain into training.  Final ID/OOD test geometry is not
instantiated until the model and computational condition are frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import project_paths
from phase4_extended_dataset import (
    TRAIN_EXTENDED,
    VALIDATION_EXTENDED,
    VISIBILITY_CLASSES,
    ExtendedPilotConfig,
    ExtendedScenarioSpec,
    RandomBoxTerrainSpec,
    _common_lattice_goal_candidates,
    _generate_boxes,
    _sample_horizontal_disk,
    _scenario_id,
    _sensor_strata,
    _terrain_hash,
    build_random_box_terrain,
)


G2_DATASET_SCHEMA = "p1b-phase4-g2-random-box-benchmark-v2"
G2_TRAIN = TRAIN_EXTENDED
G2_VALIDATION = VALIDATION_EXTENDED
G2_FINAL_ID_TEST = "FINAL_TEST_RANDOM_BOX_ID"
G2_FINAL_OOD_TEST = "FINAL_TEST_OOD"
G2_FEASIBILITY_RESOLUTIONS_M = (100.0, 50.0, 25.0, 10.0)
G2_RESAMPLE_SEED_STRIDE = 100_000


@dataclass(frozen=True)
class G2BenchmarkConfig:
    pilot: ExtendedPilotConfig = field(default_factory=lambda: ExtendedPilotConfig(
        train_terrain_count=8,
        validation_terrain_count=4,
    ))
    training_terrain_count: int = 8
    validation_terrain_count: int = 4
    training_terrain_seed_base: int = 26_401
    validation_terrain_seed_base: int = 27_401
    training_mission_seed_base: int = 26_501
    validation_mission_seed_base: int = 27_501
    training_sensor_seed_base: int = 26_601
    validation_sensor_seed_base: int = 27_601
    final_id_test_terrain_count: int = 4
    mission_pairs_per_terrain: int = 1
    scenarios_per_mission: int = 4
    # These indices are frozen only after the multi-resolution feasibility
    # audit.  Terrain seeds and geometry never change; an index changes only
    # the mission and sensor RNG streams for that terrain.
    training_resample_attempts: tuple[int, ...] = (10, 0, 7, 0, 0, 0, 3, 3)
    validation_resample_attempts: tuple[int, ...] = (1, 0, 0, 0)

    def __post_init__(self) -> None:
        if self.training_terrain_count != 8:
            raise ValueError("G2 v1 freezes exactly eight training terrains")
        if self.validation_terrain_count != 4:
            raise ValueError("G2 v1 freezes exactly four validation terrains")
        if self.final_id_test_terrain_count != 4:
            raise ValueError("G2 v1 reserves exactly four final ID-test terrains")
        if self.mission_pairs_per_terrain != 1:
            raise ValueError("G2 v1 freezes one mission pair per terrain")
        if self.scenarios_per_mission != sum(self.pilot.visibility_mix):
            raise ValueError("scenario count must match the approved 25/50/25 mix")
        if len(self.training_resample_attempts) != self.training_terrain_count:
            raise ValueError("one training resample attempt is required per terrain")
        if len(self.validation_resample_attempts) != self.validation_terrain_count:
            raise ValueError("one validation resample attempt is required per terrain")
        if any(
            int(value) < 0
            for value in (
                *self.training_resample_attempts,
                *self.validation_resample_attempts,
            )
        ):
            raise ValueError("G2 resample attempts must be nonnegative")
        ranges = (
            range(
                self.training_terrain_seed_base,
                self.training_terrain_seed_base + self.training_terrain_count,
            ),
            range(
                self.validation_terrain_seed_base,
                self.validation_terrain_seed_base + self.validation_terrain_count,
            ),
        )
        if set(ranges[0]) & set(ranges[1]):
            raise ValueError("G2 terrain seed ranges overlap")

    def as_dict(self) -> dict[str, Any]:
        physical_contract = self.pilot.as_dict()
        ignored_g1_bank_fields = {
            key: physical_contract.pop(key)
            for key in (
                "train_terrain_count", "validation_terrain_count",
                "terrain_seed", "mission_seed", "sensor_seed",
            )
        }
        return {
            **asdict(self),
            "pilot": physical_contract,
            "ignored_extended_pilot_bank_fields": {
                **ignored_g1_bank_fields,
                "reason": (
                    "G2 uses the explicit split-specific counts and seed bases "
                    "stored at the top level"
                ),
            },
            "seed_policy": (
                "explicit split-specific bases; split membership is invariant to "
                "the other split's size"
            ),
            "mission_sensor_resampling": {
                "terrain_geometry_changes": False,
                "seed_stride": G2_RESAMPLE_SEED_STRIDE,
                "acceptance_rule": (
                    "all four low/mixed/mixed/high scenarios have at least one "
                    "powered-occluded, goal-reachable switching candidate at "
                    "every frozen feasibility resolution"
                ),
                "feasibility_resolutions_m": list(G2_FEASIBILITY_RESOLUTIONS_M),
            },
            "parameter_origin": (
                "approved_G2_plan_2026-10-27_and_feasibility_resampling_"
                "approved_2026-10-28"
            ),
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _build_terrain_scenarios(
    *,
    split: str,
    split_name: str,
    index: int,
    terrain_seed_base: int,
    mission_seed_base: int,
    sensor_seed_base: int,
    resample_attempt: int,
    config: G2BenchmarkConfig,
) -> tuple[list[ExtendedScenarioSpec], RandomBoxTerrainSpec]:
    """Build one terrain's four scenarios for a deterministic attempt index."""

    goal_candidates = _common_lattice_goal_candidates(config.pilot)
    attempt = int(resample_attempt)
    terrain_seed = int(terrain_seed_base + index)
    mission_seed = int(
        mission_seed_base + index + attempt * G2_RESAMPLE_SEED_STRIDE
    )
    sensor_seed = int(
        sensor_seed_base + index + attempt * G2_RESAMPLE_SEED_STRIDE
    )
    mission_rng = np.random.default_rng(mission_seed)
    sensor_rng = np.random.default_rng(sensor_seed)
    boxes = _generate_boxes(terrain_seed, config.pilot)
    terrain_id = f"g2_random_boxes_{split_name}_{index:03d}"
    terrain_hash = _terrain_hash(boxes)
    terrain_spec = RandomBoxTerrainSpec(
        terrain_instance_id=terrain_id,
        split=split,
        generation_seed=terrain_seed,
        boxes=boxes,
        terrain_hash=terrain_hash,
    )
    terrain = build_random_box_terrain(terrain_spec, config.pilot)
    start = _sample_horizontal_disk(
        mission_rng,
        config.pilot.start_center_map,
        config.pilot.start_sampling_radius_map,
        config.pilot.graph_bounds,
    )
    goal = goal_candidates[int(mission_rng.integers(len(goal_candidates)))]
    if terrain.contains_solid(np.asarray(start)):
        raise RuntimeError(f"G2 terrain overlaps sampled start: {terrain_id}")
    if terrain.contains_solid(np.asarray(goal)):
        raise RuntimeError(f"G2 terrain overlaps sampled goal: {terrain_id}")
    selected_sensors = _sensor_strata(terrain, sensor_rng, config.pilot)
    scenarios: list[ExtendedScenarioSpec] = []
    for sensor_label, (sensor, visibility) in selected_sensors.items():
        visibility_class = sensor_label.rsplit("_", 1)[0]
        scenarios.append(ExtendedScenarioSpec(
            scenario_id=_scenario_id(
                split, terrain_id, start, goal, sensor, visibility_class,
            ),
            split=split,
            terrain_category=terrain_id,
            terrain_hash=terrain_hash,
            start_map=start,
            goal_map=goal,
            goal_tolerance_m=config.pilot.goal_terminal_radius_m,
            sensor_map=sensor,
            visibility_class=visibility_class,
            visibility_fraction=visibility,
        ))
    return scenarios, terrain_spec


def _build_split(
    *,
    split: str,
    split_name: str,
    count: int,
    terrain_seed_base: int,
    mission_seed_base: int,
    sensor_seed_base: int,
    resample_attempts: tuple[int, ...],
    config: G2BenchmarkConfig,
) -> tuple[list[ExtendedScenarioSpec], dict[str, RandomBoxTerrainSpec]]:
    scenarios: list[ExtendedScenarioSpec] = []
    terrains: dict[str, RandomBoxTerrainSpec] = {}
    for index in range(int(count)):
        selected, terrain = _build_terrain_scenarios(
            split=split,
            split_name=split_name,
            index=index,
            terrain_seed_base=terrain_seed_base,
            mission_seed_base=mission_seed_base,
            sensor_seed_base=sensor_seed_base,
            resample_attempt=int(resample_attempts[index]),
            config=config,
        )
        scenarios.extend(selected)
        terrains[terrain.terrain_instance_id] = terrain
    return scenarios, terrains


def build_g2_scenario_bank(
    config: G2BenchmarkConfig = G2BenchmarkConfig(),
) -> tuple[
    dict[str, tuple[ExtendedScenarioSpec, ...]],
    dict[str, RandomBoxTerrainSpec],
]:
    train, train_terrains = _build_split(
        split=G2_TRAIN,
        split_name="train",
        count=config.training_terrain_count,
        terrain_seed_base=config.training_terrain_seed_base,
        mission_seed_base=config.training_mission_seed_base,
        sensor_seed_base=config.training_sensor_seed_base,
        resample_attempts=config.training_resample_attempts,
        config=config,
    )
    validation, validation_terrains = _build_split(
        split=G2_VALIDATION,
        split_name="validation",
        count=config.validation_terrain_count,
        terrain_seed_base=config.validation_terrain_seed_base,
        mission_seed_base=config.validation_mission_seed_base,
        sensor_seed_base=config.validation_sensor_seed_base,
        resample_attempts=config.validation_resample_attempts,
        config=config,
    )
    splits = {
        G2_TRAIN: tuple(train),
        G2_VALIDATION: tuple(validation),
    }
    terrains = {**train_terrains, **validation_terrains}
    validate_g2_bank(splits, terrains, config)
    return splits, terrains


def validate_g2_bank(
    splits: dict[str, tuple[ExtendedScenarioSpec, ...]],
    terrains: dict[str, RandomBoxTerrainSpec],
    config: G2BenchmarkConfig,
) -> None:
    if set(splits) != {G2_TRAIN, G2_VALIDATION}:
        raise ValueError("G2 instantiated bank contains train and validation only")
    expected_counts = {
        G2_TRAIN: config.training_terrain_count,
        G2_VALIDATION: config.validation_terrain_count,
    }
    all_ids: set[str] = set()
    split_hashes: dict[str, set[str]] = {}
    split_seeds: dict[str, set[int]] = {}
    for split, terrain_count in expected_counts.items():
        values = splits[split]
        if len(values) != terrain_count * config.scenarios_per_mission:
            raise ValueError(f"G2 {split} scenario count drifted")
        terrain_ids = {spec.terrain_category for spec in values}
        if len(terrain_ids) != terrain_count:
            raise ValueError(f"G2 {split} terrain count drifted")
        class_counts = {
            label: sum(spec.visibility_class == label for spec in values)
            for label in VISIBILITY_CLASSES
        }
        expected_classes = dict(zip(
            VISIBILITY_CLASSES,
            np.asarray(config.pilot.visibility_mix) * terrain_count,
        ))
        if class_counts != expected_classes:
            raise ValueError(f"G2 {split} visibility mix drifted")
        split_hashes[split] = set()
        split_seeds[split] = set()
        for spec in values:
            if spec.scenario_id in all_ids:
                raise ValueError("duplicate G2 scenario ID")
            all_ids.add(spec.scenario_id)
            terrain = terrains[spec.terrain_category]
            if terrain.split != split or terrain.terrain_hash != spec.terrain_hash:
                raise ValueError("G2 terrain/scenario provenance mismatch")
            split_hashes[split].add(spec.terrain_hash)
            split_seeds[split].add(terrain.generation_seed)
            start_distance = np.linalg.norm(
                np.asarray(spec.start_map)
                - np.asarray(config.pilot.start_center_map)
            ) * config.pilot.meters_per_map_unit
            goal_distance = np.linalg.norm(
                np.asarray(spec.goal_map)
                - np.asarray(config.pilot.goal_center_map)
            ) * config.pilot.meters_per_map_unit
            if start_distance > config.pilot.start_sampling_radius_m + 1.0e-9:
                raise ValueError("G2 start lies outside r_sp")
            if goal_distance > config.pilot.goal_sampling_radius_m + 1.0e-9:
                raise ValueError("G2 goal lies outside r_gp")
    if split_hashes[G2_TRAIN] & split_hashes[G2_VALIDATION]:
        raise ValueError("G2 terrain geometry leaked between train and validation")
    if split_seeds[G2_TRAIN] & split_seeds[G2_VALIDATION]:
        raise ValueError("G2 terrain seeds leaked between train and validation")
    g1_seed_range = set(range(16_401, 16_407))
    if (split_seeds[G2_TRAIN] | split_seeds[G2_VALIDATION]) & g1_seed_range:
        raise ValueError("G2 terrain seeds overlap the G1 pilot")


def g2_dataset_manifest(
    config: G2BenchmarkConfig = G2BenchmarkConfig(),
) -> dict[str, Any]:
    splits, terrains = build_g2_scenario_bank(config)
    manifest: dict[str, Any] = {
        "schema": G2_DATASET_SCHEMA,
        "status": "train_validation_frozen_final_test_sealed",
        "config": config.as_dict(),
        "split_policy": {
            "training": "policy optimization only",
            "validation": "checkpoint selection and stopping only",
            "final_test": "not instantiated until model and condition freeze",
        },
        "counts": {
            "training_terrains": config.training_terrain_count,
            "validation_terrains": config.validation_terrain_count,
            "training_scenarios": len(splits[G2_TRAIN]),
            "validation_scenarios": len(splits[G2_VALIDATION]),
            "reserved_final_id_test_terrains": config.final_id_test_terrain_count,
        },
        "splits": {
            split: [spec.as_dict() for spec in values]
            for split, values in splits.items()
        },
        "terrains": {
            key: terrains[key].as_dict() for key in sorted(terrains)
        },
        "sealed_final_test_contract": {
            "instantiated": False,
            "random_box_id": {
                "terrain_count": config.final_id_test_terrain_count,
                "seed_status": "not_generated",
                "geometry_status": "not_generated",
            },
            "ood_families": [
                "corridor_or_wall_procedural",
                "stepped_pyramid",
                "urban_mesh",
                "mountain_height_map",
            ],
            "opening_condition": (
                "DQN algorithm, training bank, stopping rule, computational "
                "condition, and selected checkpoint are frozen"
            ),
            "prohibited_uses": [
                "training", "checkpoint_selection", "early_stopping",
                "hyperparameter_tuning",
            ],
        },
    }
    digest_payload = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json(digest_payload).encode("utf-8")
    ).hexdigest()
    return manifest


def write_g2_design_artifacts(
    output_dir: Path,
    config: G2BenchmarkConfig = G2BenchmarkConfig(),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = g2_dataset_manifest(config)
    manifest_path = output_dir / "g2_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    splits, terrains = build_g2_scenario_bank(config)
    audit = {
        "schema": "p1b-phase4-g2-bank-audit-v1",
        "status": "passed",
        "manifest_sha256": manifest["manifest_sha256"],
        "training_terrain_count": len({
            item.terrain_category for item in splits[G2_TRAIN]
        }),
        "validation_terrain_count": len({
            item.terrain_category for item in splits[G2_VALIDATION]
        }),
        "training_scenario_count": len(splits[G2_TRAIN]),
        "validation_scenario_count": len(splits[G2_VALIDATION]),
        "training_visibility_counts": {
            label: sum(item.visibility_class == label for item in splits[G2_TRAIN])
            for label in VISIBILITY_CLASSES
        },
        "validation_visibility_counts": {
            label: sum(
                item.visibility_class == label for item in splits[G2_VALIDATION]
            )
            for label in VISIBILITY_CLASSES
        },
        "train_validation_hash_overlap": len(
            {item.terrain_hash for item in splits[G2_TRAIN]}
            & {item.terrain_hash for item in splits[G2_VALIDATION]}
        ),
        "g1_seed_overlap": len(
            {terrain.generation_seed for terrain in terrains.values()}
            & set(range(16_401, 16_407))
        ),
        "independent_start_count": len({
            item.start_map for values in splits.values() for item in values
        }),
        "sampled_goal_count": len({
            item.goal_map for values in splits.values() for item in values
        }),
        "sealed_final_test_instantiated": False,
        "scope": "dataset contract only; no Bellman solve and no DQN training",
    }
    audit_path = output_dir / "g2_dataset_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path


def main() -> None:
    output = (
        project_paths.CORE_DIR
        / "figure"
        / "phase_4_extended_random_boxes"
        / "training_v2"
        / "g2_benchmark_bank"
        / "design_v1"
    )
    print(write_g2_design_artifacts(output))


if __name__ == "__main__":
    main()


__all__ = [
    "G2BenchmarkConfig", "G2_DATASET_SCHEMA", "G2_FEASIBILITY_RESOLUTIONS_M",
    "G2_FINAL_ID_TEST", "G2_FINAL_OOD_TEST", "G2_TRAIN", "G2_VALIDATION",
    "G2_RESAMPLE_SEED_STRIDE", "_build_terrain_scenarios",
    "build_g2_scenario_bank", "g2_dataset_manifest", "validate_g2_bank",
    "write_g2_design_artifacts",
]
