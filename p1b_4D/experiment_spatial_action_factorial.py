"""P4 two-hill spatial/speed/successor-stencil factorial experiment.

The two physical P2 sensor candidates are fixed.  The follower is evaluated
for every combination of:

* spatial grid: native or refined;
* speed grid: 5 or 9 nested speeds;
* successor stencil: 3x8 or nested 6x16 cell offsets.

``gamma_count`` is deliberately not treated as an action-resolution factor
because the physical successor solver derives flight-path angles from spatial
successor offsets.  Standard-action native/refined records are reused from P3.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from p1b_4D.experiment_multiterrain_baselines import (
    RESULT_PATH as P2_RESULT_PATH,
    build_terrain_config,
)
from p1b_4D.experiment_p2_selected_ranking_stability import (
    RESULT_PATH as P3_RESULT_PATH,
)
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.result_provenance import (
    build_result_provenance,
    provenance_from_evaluation,
)
from p1b_4D.stackelberg_solver import evaluate_defender_position


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "spatial_action_factorial"
CHECKPOINT_PATH = OUTPUT_DIR / "factorial_checkpoint.json"
RESULT_PATH = OUTPUT_DIR / "factorial_results.json"
ANALYSIS_PATH = OUTPUT_DIR / "factorial_analysis.json"
SCRIPT_IDENTIFIER = "p1b_4D/experiment_spatial_action_factorial.py"
TERRAIN_NAME = "two_hill"
CANDIDATES = ("coverage_only", "stackelberg")
SPATIAL_TIERS = ("native", "refined")
SPEED_TIERS = ("v5", "v9")
STENCIL_TIERS = ("stencil_3x8", "stencil_6x16")
SPATIAL_COUNTS = {"native": (161, 101), "refined": (321, 201)}
SPEED_COUNTS = {"v5": 5, "v9": 9}
STENCILS = {
    "stencil_3x8": (3, 8),
    "stencil_6x16": (6, 16),
}


def build_factorial_config(
    spatial_tier: str,
    speed_tier: str,
    stencil_tier: str,
    project_root: Path,
) -> dict[str, Any]:
    configuration_bundle = deepcopy(
        build_terrain_config(TERRAIN_NAME, project_root)
    )
    primary = configuration_bundle["primary_result"]
    grid = primary["environment_config"]["grid"]
    z_count, h_count = SPATIAL_COUNTS[spatial_tier]
    grid["z_count"] = z_count
    grid["h_count"] = h_count
    grid["z_spacing"] = (grid["z_max"] - grid["z_min"]) / (z_count - 1)
    grid["h_spacing"] = (grid["h_max"] - grid["h_min"]) / (h_count - 1)

    speed_count = SPEED_COUNTS[speed_tier]
    grid["v_count"] = speed_count
    primary["vehicle_config"]["glide_speed_count"] = speed_count

    forward_cells, descent_cells = STENCILS[stencil_tier]
    successor = primary["attacker_solver_config"]["successor_grid"]
    successor["maximum_forward_cells"] = forward_cells
    successor["maximum_descent_cells"] = descent_cells
    successor["virtual_switch_maximum_forward_cells"] = forward_cells
    successor["virtual_switch_maximum_descent_cells"] = descent_cells
    search = primary["bellman_config"]["search_options"]
    search["max_forward_cells"] = forward_cells
    search["max_descent_cells"] = descent_cells
    return configuration_bundle


def _summarize(
    result: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    objective = primary["objective_breakdown"]
    pipeline = primary["attacker_best_response_bundle"]["primary_result"]
    bellman = pipeline["bellman_candidate_bundle"]
    graph_metadata = bellman.get("metadata", {}).get("graph_metadata", {})
    return {
        "status_success": bool(result["status"]["success"]),
        "mission_cost": float(best["mission_cost"]),
        "mission_pod": float(best["mission_pod"]),
        "mission_time": float(best["mission_time"]),
        "defender_objective": float(primary["defender_objective"]),
        "defender_pod_normalized": float(objective["defender_pod_normalized"]),
        "coverage_area_normalized": float(objective["coverage_area_normalized"]),
        "switching_point": np.asarray(best["switching_point"], dtype=float).tolist(),
        "continuous_feasible": bool(replay["feasible"]),
        "continuous_reached_goal": bool(replay["reached_goal"]),
        "continuous_violation": replay["violation"],
        "continuous_goal_miss": float(replay["goal_miss"]),
        "graph_edge_count": graph_metadata.get("edge_count"),
        "graph_state_action_count": graph_metadata.get("state_action_count"),
        "provenance": provenance_from_evaluation(
            configuration_bundle,
            result,
            script_identifier=SCRIPT_IDENTIFIER,
        ),
    }


def _reused_standard_records(
    p2_results: dict[str, Any],
    p3_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    p3_lookup = {
        (record["resolution"], record["candidate"]): record
        for record in p3_records
        if record.get("terrain") == TERRAIN_NAME
    }
    records = []
    for spatial_tier in SPATIAL_TIERS:
        for candidate in CANDIDATES:
            source = deepcopy(p3_lookup[(spatial_tier, candidate)])
            source.update(
                {
                    "terrain": TERRAIN_NAME,
                    "spatial_tier": spatial_tier,
                    "speed_tier": "v5",
                    "stencil_tier": "stencil_3x8",
                    "candidate": candidate,
                    "z_sensor": float(
                        p2_results[TERRAIN_NAME][candidate]["z_sensor_selected"]
                    ),
                    "source": "reused_p3_standard_action_record",
                }
            )
            records.append(source)
    return records


def _cell_key(spatial: str, speed: str, stencil: str) -> str:
    return f"{spatial}|{speed}|{stencil}"


def analyze_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = {
        (
            record["spatial_tier"],
            record["speed_tier"],
            record["stencil_tier"],
            record["candidate"],
        ): record
        for record in records
        if record.get("status_success", False)
    }
    required = [
        (spatial, speed, stencil, candidate)
        for spatial in SPATIAL_TIERS
        for speed in SPEED_TIERS
        for stencil in STENCIL_TIERS
        for candidate in CANDIDATES
    ]
    missing = [list(key) for key in required if key not in successful]
    if missing:
        return {
            "schema_name": "SpatialActionFactorialAnalysis",
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "missing_or_failed_cases": missing,
        }

    objectives = {
        candidate: {
            _cell_key(spatial, speed, stencil): successful[
                (spatial, speed, stencil, candidate)
            ]["defender_objective"]
            for spatial in SPATIAL_TIERS
            for speed in SPEED_TIERS
            for stencil in STENCIL_TIERS
        }
        for candidate in CANDIDATES
    }
    response_metrics = {
        metric: {
            candidate: {
                _cell_key(spatial, speed, stencil): float(
                    successful[(spatial, speed, stencil, candidate)][metric]
                    if metric != "switching_z"
                    else successful[(spatial, speed, stencil, candidate)][
                        "switching_point"
                    ][0]
                )
                for spatial in SPATIAL_TIERS
                for speed in SPEED_TIERS
                for stencil in STENCIL_TIERS
            }
            for candidate in CANDIDATES
        }
        for metric in ("mission_pod", "mission_time", "switching_z")
    }
    response_metrics["switching_h"] = {
        candidate: {
            _cell_key(spatial, speed, stencil): float(
                successful[(spatial, speed, stencil, candidate)][
                    "switching_point"
                ][1]
            )
            for spatial in SPATIAL_TIERS
            for speed in SPEED_TIERS
            for stencil in STENCIL_TIERS
        }
        for candidate in CANDIDATES
    }
    margins = {
        _cell_key(spatial, speed, stencil): float(
            successful[(spatial, speed, stencil, "stackelberg")][
                "defender_objective"
            ]
            - successful[(spatial, speed, stencil, "coverage_only")][
                "defender_objective"
            ]
        )
        for spatial in SPATIAL_TIERS
        for speed in SPEED_TIERS
        for stencil in STENCIL_TIERS
    }

    effects: dict[str, Any] = {}
    for candidate in CANDIDATES:
        value = objectives[candidate]
        spatial_effects = {
            f"{speed}|{stencil}": float(
                value[_cell_key("refined", speed, stencil)]
                - value[_cell_key("native", speed, stencil)]
            )
            for speed in SPEED_TIERS
            for stencil in STENCIL_TIERS
        }
        speed_effects = {
            f"{spatial}|{stencil}": float(
                value[_cell_key(spatial, "v9", stencil)]
                - value[_cell_key(spatial, "v5", stencil)]
            )
            for spatial in SPATIAL_TIERS
            for stencil in STENCIL_TIERS
        }
        stencil_effects = {
            f"{spatial}|{speed}": float(
                value[_cell_key(spatial, speed, "stencil_6x16")]
                - value[_cell_key(spatial, speed, "stencil_3x8")]
            )
            for spatial in SPATIAL_TIERS
            for speed in SPEED_TIERS
        }
        spatial_speed_interaction = {
            stencil: float(
                (
                    value[_cell_key("refined", "v9", stencil)]
                    - value[_cell_key("refined", "v5", stencil)]
                )
                - (
                    value[_cell_key("native", "v9", stencil)]
                    - value[_cell_key("native", "v5", stencil)]
                )
            )
            for stencil in STENCIL_TIERS
        }
        spatial_stencil_interaction = {
            speed: float(
                (
                    value[_cell_key("refined", speed, "stencil_6x16")]
                    - value[_cell_key("refined", speed, "stencil_3x8")]
                )
                - (
                    value[_cell_key("native", speed, "stencil_6x16")]
                    - value[_cell_key("native", speed, "stencil_3x8")]
                )
            )
            for speed in SPEED_TIERS
        }
        speed_stencil_interaction = {
            spatial: float(
                (
                    value[_cell_key(spatial, "v9", "stencil_6x16")]
                    - value[_cell_key(spatial, "v5", "stencil_6x16")]
                )
                - (
                    value[_cell_key(spatial, "v9", "stencil_3x8")]
                    - value[_cell_key(spatial, "v5", "stencil_3x8")]
                )
            )
            for spatial in SPATIAL_TIERS
        }
        three_way_interaction = float(
            spatial_speed_interaction["stencil_6x16"]
            - spatial_speed_interaction["stencil_3x8"]
        )
        effects[candidate] = {
            "spatial_effects_refined_minus_native": spatial_effects,
            "speed_effects_v9_minus_v5": speed_effects,
            "stencil_effects_6x16_minus_3x8": stencil_effects,
            "spatial_speed_interaction": spatial_speed_interaction,
            "spatial_stencil_interaction": spatial_stencil_interaction,
            "speed_stencil_interaction": speed_stencil_interaction,
            "three_way_interaction": three_way_interaction,
        }

    margin_signs = {
        0 if margin == 0.0 else int(np.sign(margin)) for margin in margins.values()
    }
    return {
        "schema_name": "SpatialActionFactorialAnalysis",
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "terrain": TERRAIN_NAME,
        "factor_levels": {
            "spatial": list(SPATIAL_TIERS),
            "speed": {tier: SPEED_COUNTS[tier] for tier in SPEED_TIERS},
            "successor_stencil": {
                tier: list(STENCILS[tier]) for tier in STENCIL_TIERS
            },
        },
        "stencil_factor_interpretation": (
            "nested successor action-set expansion; increasing cell-offset "
            "bounds also increases maximum physical edge span at a fixed "
            "spatial grid"
        ),
        "defender_objectives": objectives,
        "response_metrics": response_metrics,
        "stackelberg_minus_coverage_margins": margins,
        "margin_sign_stable_across_all_cells": len(margin_signs) == 1,
        "margin_minimum": float(min(margins.values())),
        "margin_maximum": float(max(margins.values())),
        "factor_effects_and_interactions": effects,
        "all_continuous_replays_feasible": all(
            record["continuous_feasible"] for record in successful.values()
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _load_records(
    restart: bool,
    reused_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if restart or not CHECKPOINT_PATH.exists():
        return reused_records
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    retained = [
        record
        for record in records
        if not (
            record.get("speed_tier") == "v5"
            and record.get("stencil_tier") == "stencil_3x8"
        )
    ]
    return retained + reused_records


def _save(records: list[dict[str, Any]]) -> None:
    records.sort(
        key=lambda record: (
            SPATIAL_TIERS.index(record["spatial_tier"]),
            SPEED_TIERS.index(record["speed_tier"]),
            STENCIL_TIERS.index(record["stencil_tier"]),
            record["candidate"],
        )
    )
    _write_json(CHECKPOINT_PATH, records)
    _write_json(RESULT_PATH, records)
    _write_json(ANALYSIS_PATH, analyze_results(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with P2_RESULT_PATH.open("r", encoding="utf-8") as handle:
        p2_results = json.load(handle)
    with P3_RESULT_PATH.open("r", encoding="utf-8") as handle:
        p3_records = json.load(handle)
    reused_records = _reused_standard_records(p2_results, p3_records)
    records = _load_records(args.restart, reused_records)
    completed = {
        (
            record["spatial_tier"],
            record["speed_tier"],
            record["stencil_tier"],
            record["candidate"],
        )
        for record in records
        if record.get("status_success", False)
    }

    for spatial_tier in SPATIAL_TIERS:
        for speed_tier in SPEED_TIERS:
            for stencil_tier in STENCIL_TIERS:
                if speed_tier == "v5" and stencil_tier == "stencil_3x8":
                    continue
                for candidate in CANDIDATES:
                    key = (spatial_tier, speed_tier, stencil_tier, candidate)
                    if key in completed:
                        print(f"SKIP completed {key}", flush=True)
                        continue
                    z_sensor = float(
                        p2_results[TERRAIN_NAME][candidate]["z_sensor_selected"]
                    )
                    run_id = "-".join(key)
                    run_root = OUTPUT_DIR / spatial_tier / speed_tier / stencil_tier / candidate
                    configuration_bundle = build_factorial_config(
                        spatial_tier, speed_tier, stencil_tier, run_root
                    )
                    primary = configuration_bundle["primary_result"]
                    logger = primary["logging_utilities"]["logger"]
                    print(f"START {run_id} z={z_sensor:.6f}", flush=True)
                    start = time.perf_counter()
                    try:
                        result = evaluate_defender_position(
                            z_sensor, configuration_bundle, run_id
                        )
                        elapsed = time.perf_counter() - start
                        summary = _summarize(result, configuration_bundle)
                        summary.update(
                            {
                                "terrain": TERRAIN_NAME,
                                "spatial_tier": spatial_tier,
                                "speed_tier": speed_tier,
                                "stencil_tier": stencil_tier,
                                "candidate": candidate,
                                "z_sensor": z_sensor,
                                "elapsed_seconds": elapsed,
                                "source": "new_factorial_follower_evaluation",
                            }
                        )
                        records = [
                            record
                            for record in records
                            if (
                                record.get("spatial_tier"),
                                record.get("speed_tier"),
                                record.get("stencil_tier"),
                                record.get("candidate"),
                            )
                            != key
                        ]
                        records.append(summary)
                        completed.add(key)
                        print(
                            f"DONE {run_id} elapsed={elapsed:.1f}s "
                            f"J_D={summary['defender_objective']:.8f} "
                            f"edges={summary['graph_edge_count']} "
                            f"feasible={summary['continuous_feasible']}",
                            flush=True,
                        )
                    except Exception:
                        elapsed = time.perf_counter() - start
                        error = traceback.format_exc()
                        print(f"FAILED {run_id}\n{error}", flush=True)
                        records.append(
                            {
                                "terrain": TERRAIN_NAME,
                                "spatial_tier": spatial_tier,
                                "speed_tier": speed_tier,
                                "stencil_tier": stencil_tier,
                                "candidate": candidate,
                                "z_sensor": z_sensor,
                                "status_success": False,
                                "elapsed_seconds": elapsed,
                                "error": error,
                                "provenance": build_result_provenance(
                                    configuration_bundle,
                                    script_identifier=SCRIPT_IDENTIFIER,
                                ),
                            }
                        )
                    finally:
                        close_phase_logger(logger)
                        _save(records)

    analysis = analyze_results(records)
    _save(records)
    print(f"Results: {RESULT_PATH}", flush=True)
    print(f"Analysis: {ANALYSIS_PATH}", flush=True)
    print(
        f"complete={analysis['complete']} "
        f"all_feasible={analysis.get('all_continuous_replays_feasible')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
