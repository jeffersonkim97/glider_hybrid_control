"""Search and audit G2 mission/sensor attempts across planned resolutions.

Terrain geometry is fixed by the G2 terrain seeds.  This tool changes only the
mission and sensor RNG attempt for each terrain and accepts an attempt exactly
when all four visibility-stratified scenarios expose at least one powered-
occluded, goal-reachable switching candidate at 100/50/25/10 m.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import project_paths
from phase4_extended_scene import build_extended_scene
from phase4_g2_dataset import (
    G2_FEASIBILITY_RESOLUTIONS_M,
    G2_TRAIN,
    G2_VALIDATION,
    G2BenchmarkConfig,
    _build_terrain_scenarios,
)
from phase4_v2_run_a import RunAGoalReachableProblem


OUTPUT_DIR = (
    project_paths.CORE_DIR
    / "figure"
    / "phase_4_extended_random_boxes"
    / "training_v2"
    / "g2_benchmark_bank"
    / "feasibility_freeze_v2"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _candidate(
    *,
    split: str,
    split_name: str,
    index: int,
    attempt: int,
    config: G2BenchmarkConfig,
) -> tuple[list[Any], Any]:
    if split == G2_TRAIN:
        terrain_seed_base = config.training_terrain_seed_base
        mission_seed_base = config.training_mission_seed_base
        sensor_seed_base = config.training_sensor_seed_base
    elif split == G2_VALIDATION:
        terrain_seed_base = config.validation_terrain_seed_base
        mission_seed_base = config.validation_mission_seed_base
        sensor_seed_base = config.validation_sensor_seed_base
    else:
        raise ValueError(split)
    return _build_terrain_scenarios(
        split=split,
        split_name=split_name,
        index=index,
        terrain_seed_base=terrain_seed_base,
        mission_seed_base=mission_seed_base,
        sensor_seed_base=sensor_seed_base,
        resample_attempt=attempt,
        config=config,
    )


def _audit_attempt(
    specs: list[Any],
    terrain_spec: Any,
    config: G2BenchmarkConfig,
) -> tuple[bool, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for resolution in G2_FEASIBILITY_RESOLUTIONS_M:
        started = perf_counter()
        scene = build_extended_scene(
            specs[0],
            terrain_spec,
            spatial_resolution_m=float(resolution),
            heading_spacing_deg=5.0,
            pilot_config=config.pilot,
        )
        candidate_counts: list[int] = []
        for spec in specs:
            problem = RunAGoalReachableProblem(scene, spec.sensor_map)
            candidate_counts.append(len(problem.switching_state_ids()))
            del problem
        records.append({
            "spatial_resolution_m": float(resolution),
            "candidate_counts": candidate_counts,
            "all_four_feasible": all(value > 0 for value in candidate_counts),
            "runtime_sec": perf_counter() - started,
            "scene_build_timing_sec": dict(scene.build_timing),
        })
        accepted = records[-1]["all_four_feasible"]
        del scene
        gc.collect()
        # Coarse failure rejects the attempt immediately.  This avoids building
        # large fine-resolution graphs for a mission that cannot be accepted.
        if not accepted:
            return False, records
    return True, records


def freeze_feasible_attempts(*, max_attempts: int = 100) -> Path:
    config = G2BenchmarkConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    accepted: dict[str, list[int]] = {G2_TRAIN: [], G2_VALIDATION: []}
    terrain_records: list[dict[str, Any]] = []
    rejected_attempts: list[dict[str, Any]] = []

    for split, split_name, count in (
        (G2_TRAIN, "train", config.training_terrain_count),
        (G2_VALIDATION, "validation", config.validation_terrain_count),
    ):
        for index in range(count):
            selected: dict[str, Any] | None = None
            for attempt in range(int(max_attempts)):
                try:
                    specs, terrain_spec = _candidate(
                        split=split,
                        split_name=split_name,
                        index=index,
                        attempt=attempt,
                        config=config,
                    )
                    feasible, condition_records = _audit_attempt(
                        specs, terrain_spec, config,
                    )
                except RuntimeError as error:
                    rejected_attempts.append({
                        "split": split,
                        "terrain_index": index,
                        "attempt": attempt,
                        "reason": str(error),
                    })
                    continue
                if not feasible:
                    rejected_attempts.append({
                        "split": split,
                        "terrain_index": index,
                        "attempt": attempt,
                        "reason": "one_or_more_scenarios_have_no_switching_candidate",
                        "conditions_checked": condition_records,
                    })
                    print(
                        f"G2 feasibility reject {split} terrain {index} "
                        f"attempt {attempt}",
                        flush=True,
                    )
                    continue
                selected = {
                    "split": split,
                    "terrain_index": index,
                    "terrain_id": terrain_spec.terrain_instance_id,
                    "terrain_seed": terrain_spec.generation_seed,
                    "terrain_hash": terrain_spec.terrain_hash,
                    "accepted_attempt": attempt,
                    "scenario_ids": [spec.scenario_id for spec in specs],
                    "start_map": list(specs[0].start_map),
                    "goal_map": list(specs[0].goal_map),
                    "visibility_classes": [
                        spec.visibility_class for spec in specs
                    ],
                    "conditions": condition_records,
                }
                break
            if selected is None:
                raise RuntimeError(
                    f"no feasible G2 attempt for {split} terrain {index} "
                    f"within {max_attempts} attempts"
                )
            accepted[split].append(int(selected["accepted_attempt"]))
            terrain_records.append(selected)
            print(
                f"G2 feasibility accepted {split} terrain {index}: "
                f"attempt {selected['accepted_attempt']}",
                flush=True,
            )
            _write_json(OUTPUT_DIR / "g2_feasibility_search_progress.json", {
                "status": "running",
                "accepted_attempts": accepted,
                "terrain_records": terrain_records,
                "rejected_attempt_count": len(rejected_attempts),
                "elapsed_sec": perf_counter() - started,
            })

    result = {
        "schema": "p1b-phase4-g2-multiresolution-feasibility-freeze-v1",
        "status": "passed",
        "criterion": (
            "each of four visibility-stratified scenarios has at least one "
            "powered-occluded, goal-reachable switching candidate"
        ),
        "spatial_resolutions_m": list(G2_FEASIBILITY_RESOLUTIONS_M),
        "heading_spacing_deg": 5.0,
        "terrain_geometry_frozen": True,
        "training_resample_attempts": accepted[G2_TRAIN],
        "validation_resample_attempts": accepted[G2_VALIDATION],
        "terrain_records": terrain_records,
        "rejected_attempt_count": len(rejected_attempts),
        "rejected_attempts": rejected_attempts,
        "runtime_sec": perf_counter() - started,
        "base_config": asdict(config),
        "next_action": (
            "copy the accepted attempt tuples into G2BenchmarkConfig defaults, "
            "regenerate the manifest, then recompute Bellman references"
        ),
    }
    result_path = OUTPUT_DIR / "g2_feasibility_freeze.json"
    _write_json(result_path, result)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=100)
    arguments = parser.parse_args()
    print(freeze_feasible_attempts(max_attempts=arguments.max_attempts))


if __name__ == "__main__":
    main()


__all__ = ["freeze_feasible_attempts"]
