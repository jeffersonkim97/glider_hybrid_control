"""P3: cross-resolution ranking stability for the final P2 candidates.

The coverage-only and Stackelberg sensor positions selected by the refined P2
run are held fixed in physical space.  Only the follower grid changes across
coarse, native, and refined resolutions; no outer sensor optimization is
repeated.  Refined evaluations are reused from the authoritative P2 artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from p1b_4D.experiment_multiterrain_baselines import (
    RESULT_PATH as P2_RESULT_PATH,
    TERRAINS,
    build_terrain_config,
)
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.result_provenance import (
    build_result_provenance,
    provenance_from_evaluation,
)
from p1b_4D.stackelberg_solver import evaluate_defender_position


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "p2_selected_ranking_stability"
CHECKPOINT_PATH = OUTPUT_DIR / "ranking_stability_checkpoint.json"
RESULT_PATH = OUTPUT_DIR / "ranking_stability_results.json"
ANALYSIS_PATH = OUTPUT_DIR / "ranking_stability_analysis.json"
SCRIPT_IDENTIFIER = "p1b_4D/experiment_p2_selected_ranking_stability.py"
RESOLUTION_ORDER = ("coarse", "native", "refined")
CANDIDATES = ("coverage_only", "stackelberg")

# Counts preserve the refined/native 2:1 interval ratio exactly.  Valley's
# coarse z count is the nearest nested-style integer tier because 233 native
# z intervals cannot be divided by two exactly.
RESOLUTION_COUNTS = {
    "single_hill": {
        "coarse": (161, 101),
        "native": (321, 201),
        "refined": (641, 401),
    },
    "two_hill": {
        "coarse": (81, 51),
        "native": (161, 101),
        "refined": (321, 201),
    },
    "goal_in_valley": {
        "coarse": (117, 51),
        "native": (234, 101),
        "refined": (467, 201),
    },
}


def build_resolution_config(
    terrain_name: str,
    resolution_name: str,
    project_root: Path,
) -> dict[str, Any]:
    if resolution_name not in RESOLUTION_ORDER:
        raise ValueError(f"Unknown resolution: {resolution_name}")
    configuration_bundle = deepcopy(
        build_terrain_config(terrain_name, project_root)
    )
    z_count, h_count = RESOLUTION_COUNTS[terrain_name][resolution_name]
    grid = configuration_bundle["primary_result"]["environment_config"]["grid"]
    grid["z_count"] = z_count
    grid["h_count"] = h_count
    grid["z_spacing"] = (grid["z_max"] - grid["z_min"]) / (z_count - 1)
    grid["h_spacing"] = (grid["h_max"] - grid["h_min"]) / (h_count - 1)
    return configuration_bundle


def _summarize_evaluation(
    result: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    objective = primary["objective_breakdown"]
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
        "transition_model": configuration_bundle["primary_result"][
            "attacker_solver_config"
        ]["transition_model"],
        "provenance": provenance_from_evaluation(
            configuration_bundle,
            result,
            script_identifier=SCRIPT_IDENTIFIER,
        ),
    }


def _refined_records_from_p2(p2_results: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for terrain_name in TERRAINS:
        for candidate in CANDIDATES:
            source = deepcopy(p2_results[terrain_name][candidate])
            source.update(
                {
                    "terrain": terrain_name,
                    "resolution": "refined",
                    "candidate": candidate,
                    "z_sensor": float(source["z_sensor_selected"]),
                    "source": "reused_authoritative_refined_p2_result",
                }
            )
            records.append(source)
    return records


def analyze_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = {
        (record["terrain"], record["resolution"], record["candidate"]): record
        for record in records
        if record.get("status_success", False)
    }
    terrain_analyses = {}
    for terrain_name in TERRAINS:
        required = [
            (terrain_name, resolution, candidate)
            for resolution in RESOLUTION_ORDER
            for candidate in CANDIDATES
        ]
        missing = [list(key) for key in required if key not in successful]
        if missing:
            terrain_analyses[terrain_name] = {
                "complete": False,
                "missing_or_failed_cases": missing,
            }
            continue

        values = {
            resolution: {
                candidate: successful[(terrain_name, resolution, candidate)]
                for candidate in CANDIDATES
            }
            for resolution in RESOLUTION_ORDER
        }
        margins = {
            resolution: float(
                values[resolution]["stackelberg"]["defender_objective"]
                - values[resolution]["coverage_only"]["defender_objective"]
            )
            for resolution in RESOLUTION_ORDER
        }
        rankings = {
            resolution: sorted(
                CANDIDATES,
                key=lambda candidate: -values[resolution][candidate][
                    "defender_objective"
                ],
            )
            for resolution in RESOLUTION_ORDER
        }
        objective_changes = {
            candidate: {
                "coarse_to_native": float(
                    values["native"][candidate]["defender_objective"]
                    - values["coarse"][candidate]["defender_objective"]
                ),
                "native_to_refined": float(
                    values["refined"][candidate]["defender_objective"]
                    - values["native"][candidate]["defender_objective"]
                ),
            }
            for candidate in CANDIDATES
        }
        native_refined_shift = max(
            abs(changes["native_to_refined"])
            for changes in objective_changes.values()
        )
        signs = {
            0 if margin == 0.0 else int(math.copysign(1.0, margin))
            for margin in margins.values()
        }
        ranking_sign_stable = len(signs) == 1
        refined_margin_resolved = (
            abs(margins["refined"]) > native_refined_shift
        )
        z_coverage = values["refined"]["coverage_only"]["z_sensor"]
        z_stackelberg = values["refined"]["stackelberg"]["z_sensor"]
        sensor_separation = abs(z_stackelberg - z_coverage)
        if sensor_separation <= 1.0:
            classification = "numerical_tie_colocated_candidates"
        elif ranking_sign_stable and refined_margin_resolved:
            classification = "stable_resolved_stackelberg_advantage"
        elif ranking_sign_stable:
            classification = "stable_but_below_resolution_shift"
        elif abs(margins["refined"]) <= native_refined_shift:
            classification = "ranking_reversal_within_resolution_uncertainty"
        else:
            classification = "material_resolution_sensitive_ranking"

        switching_changes = {}
        for candidate in CANDIDATES:
            native_switch = np.asarray(
                values["native"][candidate]["switching_point"], dtype=float
            )
            refined_switch = np.asarray(
                values["refined"][candidate]["switching_point"], dtype=float
            )
            switching_changes[candidate] = float(
                np.linalg.norm(refined_switch - native_switch)
            )

        terrain_analyses[terrain_name] = {
            "complete": True,
            "fixed_sensor_positions": {
                candidate: values["refined"][candidate]["z_sensor"]
                for candidate in CANDIDATES
            },
            "sensor_separation": sensor_separation,
            "defender_objectives": {
                resolution: {
                    candidate: values[resolution][candidate]["defender_objective"]
                    for candidate in CANDIDATES
                }
                for resolution in RESOLUTION_ORDER
            },
            "stackelberg_minus_coverage_margin": margins,
            "rankings": rankings,
            "ranking_sign_stable": ranking_sign_stable,
            "objective_changes": objective_changes,
            "maximum_native_to_refined_candidate_shift": native_refined_shift,
            "refined_margin_exceeds_native_to_refined_shift": (
                refined_margin_resolved
            ),
            "native_to_refined_switching_displacement": switching_changes,
            "all_continuous_replays_feasible": all(
                values[resolution][candidate]["continuous_feasible"]
                for resolution in RESOLUTION_ORDER
                for candidate in CANDIDATES
            ),
            "classification": classification,
            "paper_claim_supported": (
                classification == "stable_resolved_stackelberg_advantage"
            ),
        }

    complete = all(item.get("complete", False) for item in terrain_analyses.values())
    return {
        "schema_name": "P2SelectedRankingStabilityAnalysis",
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "all_continuous_replays_feasible": complete
        and all(
            item["all_continuous_replays_feasible"]
            for item in terrain_analyses.values()
        ),
        "terrain_analyses": terrain_analyses,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _load_records(restart: bool, p2_results: dict[str, Any]) -> list[dict[str, Any]]:
    refined_records = _refined_records_from_p2(p2_results)
    if restart or not CHECKPOINT_PATH.exists():
        return refined_records
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    retained = [record for record in records if record.get("resolution") != "refined"]
    return retained + refined_records


def _save(records: list[dict[str, Any]]) -> None:
    records.sort(
        key=lambda record: (
            record.get("terrain", ""),
            RESOLUTION_ORDER.index(record.get("resolution", "coarse")),
            record.get("candidate", ""),
        )
    )
    _write_json(CHECKPOINT_PATH, records)
    _write_json(RESULT_PATH, records)
    _write_json(ANALYSIS_PATH, analyze_results(records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrains",
        nargs="+",
        choices=tuple(TERRAINS),
        default=list(TERRAINS),
    )
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with P2_RESULT_PATH.open("r", encoding="utf-8") as handle:
        p2_results = json.load(handle)
    records = _load_records(args.restart, p2_results)
    completed = {
        (record["terrain"], record["resolution"], record["candidate"])
        for record in records
        if record.get("status_success", False)
    }

    for terrain_name in args.terrains:
        for resolution_name in ("coarse", "native"):
            for candidate in CANDIDATES:
                key = (terrain_name, resolution_name, candidate)
                if key in completed:
                    print(f"SKIP completed {key}", flush=True)
                    continue
                z_sensor = float(p2_results[terrain_name][candidate]["z_sensor_selected"])
                run_id = f"{terrain_name}-{resolution_name}-{candidate}"
                run_root = OUTPUT_DIR / terrain_name / resolution_name / candidate
                configuration_bundle = build_resolution_config(
                    terrain_name, resolution_name, run_root
                )
                primary = configuration_bundle["primary_result"]
                logger = primary["logging_utilities"]["logger"]
                grid = primary["environment_config"]["grid"]
                print(
                    f"START {run_id} z={z_sensor:.6f} "
                    f"grid={grid['z_count']}x{grid['h_count']}",
                    flush=True,
                )
                start = time.perf_counter()
                try:
                    result = evaluate_defender_position(
                        z_sensor, configuration_bundle, run_id
                    )
                    elapsed = time.perf_counter() - start
                    summary = _summarize_evaluation(result, configuration_bundle)
                    summary.update(
                        {
                            "terrain": terrain_name,
                            "resolution": resolution_name,
                            "candidate": candidate,
                            "z_sensor": z_sensor,
                            "z_sensor_selected": z_sensor,
                            "elapsed_seconds": elapsed,
                            "source": "new_fixed_candidate_follower_evaluation",
                        }
                    )
                    records = [
                        record
                        for record in records
                        if (
                            record.get("terrain"),
                            record.get("resolution"),
                            record.get("candidate"),
                        )
                        != key
                    ]
                    records.append(summary)
                    completed.add(key)
                    print(
                        f"DONE {run_id} elapsed={elapsed:.1f}s "
                        f"J_D={summary['defender_objective']:.8f} "
                        f"feasible={summary['continuous_feasible']}",
                        flush=True,
                    )
                except Exception:
                    elapsed = time.perf_counter() - start
                    error = traceback.format_exc()
                    print(f"FAILED {run_id}\n{error}", flush=True)
                    records.append(
                        {
                            "terrain": terrain_name,
                            "resolution": resolution_name,
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
        f"all_feasible={analysis['all_continuous_replays_feasible']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
