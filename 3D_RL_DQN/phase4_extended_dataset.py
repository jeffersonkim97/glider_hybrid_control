"""Deterministic random-box scenario bank for the extended Phase 16.4 pilot.

The original Phase 16.4 dataset is frozen.  This module owns a separate schema
for the user-approved start, goal, sensor, and random-terrain variation.  Every
sample needed to reconstruct a scenario is stored in the manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import cos, pi, sin
from typing import Any, Iterable

import numpy as np

import project_paths  # noqa: F401
from map_geometry import BoxObstacle, CompositeTerrainMap, MapBounds


EXTENDED_DATASET_SCHEMA = "p1b-phase4-extended-random-box-v1"
TRAIN_EXTENDED = "TRAIN_SIMPLE"
VALIDATION_EXTENDED = "VALIDATION_SIMPLE"
VISIBILITY_CLASSES = ("low", "mixed", "high")


@dataclass(frozen=True)
class ExtendedPilotConfig:
    """User-approved values plus explicit, versioned pilot sizing choices."""

    map_bounds: tuple[float, float, float, float] = (-10.0, 10.0, -8.0, 8.0)
    graph_bounds: tuple[float, float, float, float] = (-8.0, 8.0, -4.0, 4.0)
    meters_per_map_unit: float = 100.0
    start_center_map: tuple[float, float, float] = (-8.0, 0.0, 0.0)
    start_sampling_radius_m: float = 100.0
    goal_center_map: tuple[float, float, float] = (8.0, 0.0, 0.0)
    goal_sampling_radius_m: float = 100.0
    goal_terminal_radius_m: float = 50.0
    sensor_x_range_map: tuple[float, float] = (3.25, 6.75)
    sensor_y_range_map: tuple[float, float] = (-3.75, 3.75)
    sensor_z_map: float = 0.0
    visibility_mix: tuple[int, int, int] = (1, 2, 1)
    sensor_candidate_count: int = 96
    train_terrain_count: int = 4
    validation_terrain_count: int = 2
    terrain_seed: int = 16_401
    mission_seed: int = 16_402
    sensor_seed: int = 16_403
    minimum_box_gap_m: float = 25.0
    extra_box_count_range: tuple[int, int] = (1, 4)

    def __post_init__(self) -> None:
        positive = (
            self.meters_per_map_unit,
            self.start_sampling_radius_m,
            self.goal_sampling_radius_m,
            self.goal_terminal_radius_m,
            self.minimum_box_gap_m,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("physical radii, scale, and box gap must be positive")
        if self.visibility_mix != (1, 2, 1):
            raise ValueError("the approved visibility mix is exactly 25/50/25")
        if self.sensor_candidate_count < 16:
            raise ValueError("sensor_candidate_count must support visibility strata")
        if min(self.train_terrain_count, self.validation_terrain_count) < 1:
            raise ValueError("both terrain splits must be nonempty")
        lower, upper = self.extra_box_count_range
        if lower < 0 or upper < lower:
            raise ValueError("invalid extra_box_count_range")

    @property
    def start_sampling_radius_map(self) -> float:
        return self.start_sampling_radius_m / self.meters_per_map_unit

    @property
    def goal_sampling_radius_map(self) -> float:
        return self.goal_sampling_radius_m / self.meters_per_map_unit

    @property
    def minimum_box_gap_map(self) -> float:
        return self.minimum_box_gap_m / self.meters_per_map_unit

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameter_origin"] = {
            "start_sampling_radius_m": "user_approved_2026-10-19",
            "goal_sampling_radius_m": "user_approved_2026-10-19",
            "goal_terminal_radius_m": "user_approved_2026-10-19",
            "visibility_mix": "user_approved_2026-10-19",
            "terrain_family": "user_approved_random_boxes_2026-10-19",
            "finite_scenario_bank": "user_approved_2026-10-19",
            "bank_and_box_counts": "versioned_pilot_engineering_choice",
        }
        return result


@dataclass(frozen=True)
class RandomBoxTerrainSpec:
    terrain_instance_id: str
    split: str
    generation_seed: int
    boxes: tuple[BoxObstacle, ...]
    terrain_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "terrain_instance_id": self.terrain_instance_id,
            "split": self.split,
            "generation_seed": self.generation_seed,
            "boxes": [asdict(box) for box in self.boxes],
            "terrain_hash": self.terrain_hash,
        }


@dataclass(frozen=True)
class ExtendedScenarioSpec:
    scenario_id: str
    split: str
    terrain_category: str
    terrain_hash: str
    start_map: tuple[float, float, float]
    goal_map: tuple[float, float, float]
    goal_tolerance_m: float
    sensor_map: tuple[float, float, float]
    visibility_class: str
    visibility_fraction: float

    def __post_init__(self) -> None:
        if self.visibility_class not in VISIBILITY_CLASSES:
            raise ValueError("unknown visibility class")

    @property
    def terrain_instance_id(self) -> str:
        return self.terrain_category

    @property
    def graph_cache_key(self) -> str:
        payload = {
            "terrain_hash": self.terrain_hash,
            "goal_map": list(self.goal_map),
            "goal_tolerance_m": self.goal_tolerance_m,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start_map"] = list(self.start_map)
        result["goal_map"] = list(self.goal_map)
        result["sensor_map"] = list(self.sensor_map)
        result["graph_cache_key"] = self.graph_cache_key
        return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _terrain_payload(boxes: Iterable[BoxObstacle]) -> list[dict[str, float]]:
    return [asdict(box) for box in boxes]


def _terrain_hash(boxes: Iterable[BoxObstacle]) -> str:
    return hashlib.sha256(
        _canonical_json(_terrain_payload(boxes)).encode("utf-8")
    ).hexdigest()


def _footprints_separated(
    candidate: BoxObstacle, existing: Iterable[BoxObstacle], gap_map: float,
) -> bool:
    cx0, cx1 = candidate.x_limits
    cy0, cy1 = candidate.y_limits
    for box in existing:
        bx0, bx1 = box.x_limits
        by0, by1 = box.y_limits
        separated = (
            cx1 + gap_map <= bx0 or bx1 + gap_map <= cx0
            or cy1 + gap_map <= by0 or by1 + gap_map <= cy0
        )
        if not separated:
            return False
    return True


def _generate_boxes(seed: int, config: ExtendedPilotConfig) -> tuple[BoxObstacle, ...]:
    rng = np.random.default_rng(int(seed))
    # One randomized central occluder preserves the intended terrain-hidden
    # powered leg.  Remaining boxes supply topology and geometry variation.
    boxes: list[BoxObstacle] = [
        BoxObstacle(
            center_x=float(rng.uniform(-0.5, 0.5)),
            center_y=float(rng.uniform(-0.75, 0.75)),
            width_x=float(rng.uniform(2.0, 3.5)),
            width_y=float(rng.uniform(2.0, 3.5)),
            height=float(rng.uniform(2.0, 4.0)),
        )
    ]
    lower, upper = config.extra_box_count_range
    requested = int(rng.integers(lower, upper + 1))
    attempts = 0
    while len(boxes) < requested + 1 and attempts < 500:
        attempts += 1
        candidate = BoxObstacle(
            center_x=float(rng.uniform(-4.5, 3.0)),
            center_y=float(rng.uniform(-3.25, 3.25)),
            width_x=float(rng.uniform(0.75, 2.25)),
            width_y=float(rng.uniform(0.75, 2.25)),
            height=float(rng.uniform(1.0, 4.0)),
        )
        x0, x1 = candidate.x_limits
        y0, y1 = candidate.y_limits
        if not (-7.5 <= x0 < x1 <= 7.0 and -3.75 <= y0 < y1 <= 3.75):
            continue
        if _footprints_separated(candidate, boxes, config.minimum_box_gap_map):
            boxes.append(candidate)
    if len(boxes) != requested + 1:
        raise RuntimeError("could not place the requested separated random boxes")
    return tuple(boxes)


def build_random_box_terrain(
    spec: RandomBoxTerrainSpec,
    config: ExtendedPilotConfig = ExtendedPilotConfig(),
) -> CompositeTerrainMap:
    bounds = MapBounds(*config.map_bounds)
    terrain = CompositeTerrainMap(
        bounds=bounds,
        boxes=spec.boxes,
        category_id=spec.terrain_instance_id,
    )
    if _terrain_hash(terrain.obstacle_boxes()) != spec.terrain_hash:
        raise RuntimeError("terrain reconstruction hash mismatch")
    return terrain


def _sample_horizontal_disk(
    rng: np.random.Generator,
    center: tuple[float, float, float],
    radius_map: float,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    x_min, x_max, y_min, y_max = bounds
    for _ in range(10_000):
        radius = radius_map * float(np.sqrt(rng.random()))
        angle = float(rng.uniform(0.0, 2.0 * pi))
        point = (
            float(center[0] + radius * cos(angle)),
            float(center[1] + radius * sin(angle)),
            float(center[2]),
        )
        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
            return point
    raise RuntimeError("could not sample a point inside the bounded disk")


def _common_lattice_goal_candidates(
    config: ExtendedPilotConfig,
) -> tuple[tuple[float, float, float], ...]:
    x_min, x_max, y_min, y_max = config.graph_bounds
    center = np.asarray(config.goal_center_map, dtype=float)
    radius = config.goal_sampling_radius_map
    # Integer map coordinates are shared by every planned 100/50/25/10/5 m grid.
    candidates: list[tuple[float, float, float]] = []
    for x in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
        for y in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
            point = np.asarray([float(x), float(y), center[2]], dtype=float)
            if np.linalg.norm(point - center) <= radius + 1.0e-12:
                candidates.append(tuple(float(value) for value in point))
    if not candidates:
        raise RuntimeError("goal sampling disk contains no common lattice point")
    return tuple(sorted(candidates))


def _point_in_box_footprint(point: tuple[float, float, float], box: BoxObstacle) -> bool:
    x0, x1 = box.x_limits
    y0, y1 = box.y_limits
    return x0 < point[0] < x1 and y0 < point[1] < y1


def _visibility_probes(
    terrain: CompositeTerrainMap,
    config: ExtendedPilotConfig,
) -> np.ndarray:
    x_min, x_max, y_min, y_max = config.graph_bounds
    values = np.asarray([
        (x, y, z)
        for x in np.linspace(x_min + 0.5, x_max - 0.5, 9)
        for y in np.linspace(y_min + 0.25, y_max - 0.25, 5)
        for z in (0.5, 2.5, 4.5)
    ], dtype=float)
    return values[~terrain.contains_solid_many(values)]


def visibility_fraction(
    terrain: CompositeTerrainMap,
    sensor_map: tuple[float, float, float],
    probes: np.ndarray,
) -> float:
    sensors = np.repeat(np.asarray(sensor_map, dtype=float)[None, :], len(probes), axis=0)
    blocked = terrain.segments_intersect_solid_many(probes, sensors)
    return float(np.mean(~blocked)) if len(probes) else 0.0


def _sensor_strata(
    terrain: CompositeTerrainMap,
    rng: np.random.Generator,
    config: ExtendedPilotConfig,
) -> dict[str, tuple[tuple[float, float, float], float]]:
    candidates: list[tuple[tuple[float, float, float], float]] = []
    probes = _visibility_probes(terrain, config)
    terrain_maximum_x = max(box.x_limits[1] for box in terrain.obstacle_boxes())
    sensor_x_min = max(
        float(config.sensor_x_range_map[0]),
        float(terrain_maximum_x + config.minimum_box_gap_map),
    )
    sensor_x_max = float(config.sensor_x_range_map[1])
    if sensor_x_min >= sensor_x_max:
        raise RuntimeError("random terrain leaves no sensor region beyond its far edge")
    while len(candidates) < config.sensor_candidate_count:
        point = (
            float(rng.uniform(sensor_x_min, sensor_x_max)),
            float(rng.uniform(*config.sensor_y_range_map)),
            float(config.sensor_z_map),
        )
        if any(_point_in_box_footprint(point, box) for box in terrain.obstacle_boxes()):
            continue
        score = visibility_fraction(terrain, point, probes)
        candidates.append((point, score))
    candidates.sort(key=lambda item: (item[1], item[0]))
    quartile = len(candidates) // 4
    pools = {
        "low": candidates[:quartile],
        "mixed": candidates[quartile:3 * quartile],
        "high": candidates[3 * quartile:],
    }
    selected: dict[str, tuple[tuple[float, float, float], float]] = {}
    counts = dict(zip(VISIBILITY_CLASSES, config.visibility_mix))
    for label in VISIBILITY_CLASSES:
        pool = pools[label]
        choices = rng.choice(len(pool), size=counts[label], replace=False)
        for ordinal, index in enumerate(np.atleast_1d(choices)):
            selected[f"{label}_{ordinal}"] = pool[int(index)]
    return selected


def _scenario_id(
    split: str, terrain_id: str, start: tuple[float, float, float],
    goal: tuple[float, float, float], sensor: tuple[float, float, float],
    visibility_class: str,
) -> str:
    payload = {
        "split": split,
        "terrain": terrain_id,
        "start": start,
        "goal": goal,
        "sensor": sensor,
        "visibility_class": visibility_class,
    }
    tag = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{split.lower()}__{terrain_id}__{visibility_class}__{tag}"


def build_extended_scenario_bank(
    config: ExtendedPilotConfig = ExtendedPilotConfig(),
) -> tuple[dict[str, tuple[ExtendedScenarioSpec, ...]], dict[str, RandomBoxTerrainSpec]]:
    terrain_specs: dict[str, RandomBoxTerrainSpec] = {}
    scenarios = {TRAIN_EXTENDED: [], VALIDATION_EXTENDED: []}
    goal_candidates = _common_lattice_goal_candidates(config)
    split_counts = (
        (TRAIN_EXTENDED, config.train_terrain_count),
        (VALIDATION_EXTENDED, config.validation_terrain_count),
    )
    ordinal = 0
    for split, count in split_counts:
        for split_index in range(count):
            terrain_seed = config.terrain_seed + ordinal
            mission_rng = np.random.default_rng(config.mission_seed + ordinal)
            sensor_rng = np.random.default_rng(config.sensor_seed + ordinal)
            boxes = _generate_boxes(terrain_seed, config)
            terrain_id = f"random_boxes_{split.lower()}_{split_index:03d}"
            digest = _terrain_hash(boxes)
            terrain_spec = RandomBoxTerrainSpec(
                terrain_instance_id=terrain_id,
                split=split,
                generation_seed=terrain_seed,
                boxes=boxes,
                terrain_hash=digest,
            )
            terrain_specs[terrain_id] = terrain_spec
            terrain = build_random_box_terrain(terrain_spec, config)
            start = _sample_horizontal_disk(
                mission_rng,
                config.start_center_map,
                config.start_sampling_radius_map,
                config.graph_bounds,
            )
            goal = goal_candidates[int(mission_rng.integers(len(goal_candidates)))]
            if terrain.contains_solid(np.asarray(start)) or terrain.contains_solid(np.asarray(goal)):
                raise RuntimeError("generated terrain overlaps the protected mission endpoints")
            for label, (sensor, score) in _sensor_strata(
                terrain, sensor_rng, config,
            ).items():
                visibility_class = label.rsplit("_", 1)[0]
                scenarios[split].append(ExtendedScenarioSpec(
                    scenario_id=_scenario_id(
                        split, terrain_id, start, goal, sensor, visibility_class,
                    ),
                    split=split,
                    terrain_category=terrain_id,
                    terrain_hash=digest,
                    start_map=start,
                    goal_map=goal,
                    goal_tolerance_m=config.goal_terminal_radius_m,
                    sensor_map=sensor,
                    visibility_class=visibility_class,
                    visibility_fraction=score,
                ))
            ordinal += 1
    frozen = {key: tuple(value) for key, value in scenarios.items()}
    validate_extended_bank(frozen, terrain_specs, config)
    return frozen, terrain_specs


def validate_extended_bank(
    splits: dict[str, tuple[ExtendedScenarioSpec, ...]],
    terrains: dict[str, RandomBoxTerrainSpec],
    config: ExtendedPilotConfig,
) -> None:
    if set(splits) != {TRAIN_EXTENDED, VALIDATION_EXTENDED}:
        raise ValueError("extended bank requires train and validation splits")
    ids: set[str] = set()
    split_hashes: dict[str, set[str]] = {}
    for split, specs in splits.items():
        split_hashes[split] = set()
        class_counts = {label: 0 for label in VISIBILITY_CLASSES}
        for spec in specs:
            if spec.scenario_id in ids:
                raise ValueError("duplicate extended scenario ID")
            ids.add(spec.scenario_id)
            if spec.split != split:
                raise ValueError("extended scenario split mismatch")
            terrain = terrains[spec.terrain_category]
            if terrain.split != split or terrain.terrain_hash != spec.terrain_hash:
                raise ValueError("terrain/scenario provenance mismatch")
            split_hashes[split].add(spec.terrain_hash)
            class_counts[spec.visibility_class] += 1
            start_distance = np.linalg.norm(
                np.asarray(spec.start_map) - np.asarray(config.start_center_map)
            ) * config.meters_per_map_unit
            goal_distance = np.linalg.norm(
                np.asarray(spec.goal_map) - np.asarray(config.goal_center_map)
            ) * config.meters_per_map_unit
            if start_distance > config.start_sampling_radius_m + 1.0e-9:
                raise ValueError("start lies outside r_sp")
            if goal_distance > config.goal_sampling_radius_m + 1.0e-9:
                raise ValueError("goal lies outside r_gp")
        ratio = np.asarray([class_counts[label] for label in VISIBILITY_CLASSES])
        expected = np.asarray(config.visibility_mix)
        if not np.all(ratio * expected.sum() == expected * ratio.sum()):
            raise ValueError("visibility classes do not match the approved 25/50/25 mix")
    if split_hashes[TRAIN_EXTENDED] & split_hashes[VALIDATION_EXTENDED]:
        raise ValueError("terrain geometry leaked between train and validation")


def extended_dataset_manifest(
    config: ExtendedPilotConfig = ExtendedPilotConfig(),
) -> dict[str, Any]:
    splits, terrains = build_extended_scenario_bank(config)
    manifest: dict[str, Any] = {
        "schema": EXTENDED_DATASET_SCHEMA,
        "design_status": "user_approved_2026-10-19",
        "config": config.as_dict(),
        "terrain_family": "random axis-aligned boxes with one randomized central occluder",
        "terrain_instances": {
            key: terrains[key].as_dict() for key in sorted(terrains)
        },
        "splits": {
            key: [spec.as_dict() for spec in splits[key]] for key in sorted(splits)
        },
        "counts": {
            "training_terrains": config.train_terrain_count,
            "validation_terrains": config.validation_terrain_count,
            "sensors_per_terrain": sum(config.visibility_mix),
            "training_scenarios": len(splits[TRAIN_EXTENDED]),
            "validation_scenarios": len(splits[VALIDATION_EXTENDED]),
        },
        "sampling_contract": {
            "start": "uniform-area sample in bounded horizontal r_sp disk; fixed altitude",
            "goal": (
                "uniform sample from map-unit lattice points shared by all planned spatial "
                "resolutions inside the r_gp disk; fixed altitude"
            ),
            "sensor": (
                "continuous candidate pool ranked by geometry-only visible-airspace fraction; "
                "sampled 25% low, 50% mixed, 25% high"
            ),
            "graph_cache_key": "terrain hash + exact goal point + terminal radius",
        },
        "split_contract": "terrain generation seeds and geometry hashes are disjoint",
    }
    encoded = _canonical_json(manifest).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def specs_from_extended_manifest(
    manifest: dict[str, Any], split: str,
) -> tuple[ExtendedScenarioSpec, ...]:
    return tuple(ExtendedScenarioSpec(
        scenario_id=str(item["scenario_id"]),
        split=str(item["split"]),
        terrain_category=str(item["terrain_category"]),
        terrain_hash=str(item["terrain_hash"]),
        start_map=tuple(float(value) for value in item["start_map"]),
        goal_map=tuple(float(value) for value in item["goal_map"]),
        goal_tolerance_m=float(item["goal_tolerance_m"]),
        sensor_map=tuple(float(value) for value in item["sensor_map"]),
        visibility_class=str(item["visibility_class"]),
        visibility_fraction=float(item["visibility_fraction"]),
    ) for item in manifest["splits"][split])


def terrain_specs_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, RandomBoxTerrainSpec]:
    result: dict[str, RandomBoxTerrainSpec] = {}
    for key, item in manifest["terrain_instances"].items():
        result[key] = RandomBoxTerrainSpec(
            terrain_instance_id=str(item["terrain_instance_id"]),
            split=str(item["split"]),
            generation_seed=int(item["generation_seed"]),
            boxes=tuple(BoxObstacle(**box) for box in item["boxes"]),
            terrain_hash=str(item["terrain_hash"]),
        )
    return result


__all__ = [
    "EXTENDED_DATASET_SCHEMA", "TRAIN_EXTENDED", "VALIDATION_EXTENDED",
    "VISIBILITY_CLASSES", "ExtendedPilotConfig", "ExtendedScenarioSpec",
    "RandomBoxTerrainSpec", "build_extended_scenario_bank",
    "build_random_box_terrain", "extended_dataset_manifest",
    "specs_from_extended_manifest", "terrain_specs_from_manifest",
    "validate_extended_bank", "visibility_fraction",
]
