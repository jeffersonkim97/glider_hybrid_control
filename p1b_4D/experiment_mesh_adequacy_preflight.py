"""Preflight mesh-adequacy gate for the corrected multi-terrain baselines.

The experiment fixes physically identical sensor positions and evaluates each
one with the native P2 spatial grid and a grid with both spatial intervals
halved.  The speed grid and successor-cell offsets are held fixed.  Refining
both spatial axes by the same factor preserves the successor-edge direction
set while reducing the physical edge length.

Run from the repository root::

    python -m p1b_4D.experiment_mesh_adequacy_preflight

Results are checkpointed after every evaluation, so an interrupted run can be
continued with the same command.  Use ``--restart`` to discard only this
experiment's checkpoint and recompute all requested cases.
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
OUTPUT_DIR = REPO_ROOT / "results" / "mesh_adequacy_preflight"
CHECKPOINT_PATH = OUTPUT_DIR / "mesh_adequacy_results.json"
ANALYSIS_PATH = OUTPUT_DIR / "mesh_adequacy_analysis.json"
SCRIPT_IDENTIFIER = "p1b_4D/experiment_mesh_adequacy_preflight.py"
REFINEMENT_FACTOR = 2
RESOLUTIONS = ("native", "refined_2x")

# These are fixed physical positions, not results reused as evidence.  The
# single-hill Stackelberg point comes from the 2026-07-28 successor-grid
# notebook run.  The remaining non-fixed positions come from the prior
# baseline solely to place the preflight near the paper-critical alternatives.
SENSOR_CANDIDATES: dict[str, dict[str, float]] = {
    "single_hill": {
        "fixed": 4000.0,
        "coverage_only": 4498.799725651577,
        "stackelberg": 4490.264441396129,
    },
    "two_hill": {
        "fixed": 1750.0,
        "coverage_only": 1974.691358024691,
        "stackelberg": 1980.452674897119,
    },
    "goal_in_valley": {
        "fixed": 1950.0,
        "coverage_only": 2198.971193415638,
        "stackelberg": 1705.8299039780522,
    },
}


def refined_count(count: int, factor: int = REFINEMENT_FACTOR) -> int:
    """Return a point count that divides every original interval by factor."""
    if count < 2 or factor < 1:
        raise ValueError("count must be >= 2 and factor must be >= 1")
    return (int(count) - 1) * int(factor) + 1


def build_preflight_config(
    terrain_name: str,
    resolution_name: str,
    project_root: Path,
) -> dict[str, Any]:
    """Build a native or proportionally refined successor-grid configuration."""
    if resolution_name not in RESOLUTIONS:
        raise ValueError(f"Unknown resolution: {resolution_name}")
    configuration_bundle = build_terrain_config(terrain_name, project_root)
    configuration_bundle = deepcopy(configuration_bundle)
    grid = configuration_bundle["primary_result"]["environment_config"]["grid"]
    # ``build_terrain_config`` now carries the adopted refined P2 grids.  The
    # preflight's native tier is reconstructed by taking every second refined
    # interval; the refined_2x tier is the authoritative P2 grid unchanged.
    if resolution_name == "native":
        grid["z_count"] = (int(grid["z_count"]) - 1) // REFINEMENT_FACTOR + 1
        grid["h_count"] = (int(grid["h_count"]) - 1) // REFINEMENT_FACTOR + 1
        grid["z_spacing"] = (grid["z_max"] - grid["z_min"]) / (
            grid["z_count"] - 1
        )
        grid["h_spacing"] = (grid["h_max"] - grid["h_min"]) / (
            grid["h_count"] - 1
        )
    return configuration_bundle


def successor_direction_signature(configuration_bundle: dict[str, Any]) -> list[float]:
    """Return the geometric successor directions before flight-limit filtering."""
    primary = configuration_bundle["primary_result"]
    grid = primary["environment_config"]["grid"]
    options = primary["attacker_solver_config"]["successor_grid"]
    dz = float(grid["z_spacing"])
    dh = float(grid["h_spacing"])
    return [
        math.atan2(-descent * dh, forward * dz)
        for forward in range(1, int(options["maximum_forward_cells"]) + 1)
        for descent in range(1, int(options["maximum_descent_cells"]) + 1)
    ]


def summarize_evaluation(
    result: dict[str, Any],
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    trajectory = np.asarray(best["trajectory"], dtype=float)
    differences = np.diff(trajectory, axis=0)
    gamma_profile = np.asarray(best["gamma_profile"], dtype=float)
    return {
        "status_success": bool(result["status"]["success"]),
        "mission_cost": float(best["mission_cost"]),
        "mission_pod": float(best["mission_pod"]),
        "mission_time": float(best["mission_time"]),
        "defender_objective": float(primary["defender_objective"]),
        "switching_point": np.asarray(best["switching_point"], dtype=float).tolist(),
        "trajectory": trajectory.tolist(),
        "path_node_count": int(trajectory.shape[0]),
        "glide_path_length": float(np.sum(np.linalg.norm(differences, axis=1))),
        "unique_glide_direction_count": int(
            np.unique(np.round(gamma_profile, decimals=12)).size
        ),
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


def _path_comparison(native: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    path_a = np.asarray(native["trajectory"], dtype=float)
    path_b = np.asarray(refined["trajectory"], dtype=float)
    z_start = max(float(path_a[0, 0]), float(path_b[0, 0]))
    z_end = min(float(path_a[-1, 0]), float(path_b[-1, 0]))
    if z_end <= z_start:
        return {
            "common_z_interval": [z_start, z_end],
            "altitude_rmse": None,
            "maximum_altitude_difference": None,
        }
    sample_z = np.linspace(z_start, z_end, 401)
    h_a = np.interp(sample_z, path_a[:, 0], path_a[:, 1])
    h_b = np.interp(sample_z, path_b[:, 0], path_b[:, 1])
    difference = h_b - h_a
    return {
        "common_z_interval": [z_start, z_end],
        "altitude_rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_altitude_difference": float(np.max(np.abs(difference))),
    }


def analyze_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare rankings and response changes for every complete terrain pair."""
    successful = {
        (record["terrain"], record["resolution"], record["candidate"]): record
        for record in records
        if record.get("status_success", False)
    }
    terrain_analyses: dict[str, Any] = {}
    for terrain_name, candidates in SENSOR_CANDIDATES.items():
        required = [
            (terrain_name, resolution, candidate)
            for resolution in RESOLUTIONS
            for candidate in candidates
        ]
        missing = [list(key) for key in required if key not in successful]
        if missing:
            terrain_analyses[terrain_name] = {
                "complete": False,
                "missing_or_failed_cases": missing,
            }
            continue

        by_resolution = {
            resolution: {
                candidate: successful[(terrain_name, resolution, candidate)]
                for candidate in candidates
            }
            for resolution in RESOLUTIONS
        }
        rankings = {
            resolution: sorted(
                candidates,
                key=lambda candidate: (
                    -by_resolution[resolution][candidate]["defender_objective"],
                    candidate,
                ),
            )
            for resolution in RESOLUTIONS
        }
        candidate_changes: dict[str, Any] = {}
        for candidate in candidates:
            native = by_resolution["native"][candidate]
            refined = by_resolution["refined_2x"][candidate]
            switch_delta = np.asarray(refined["switching_point"]) - np.asarray(
                native["switching_point"]
            )
            candidate_changes[candidate] = {
                "defender_objective_change": float(
                    refined["defender_objective"] - native["defender_objective"]
                ),
                "mission_cost_change": float(
                    refined["mission_cost"] - native["mission_cost"]
                ),
                "mission_pod_change": float(
                    refined["mission_pod"] - native["mission_pod"]
                ),
                "mission_time_change": float(
                    refined["mission_time"] - native["mission_time"]
                ),
                "switching_point_displacement": float(np.linalg.norm(switch_delta)),
                "path_node_count_native": native["path_node_count"],
                "path_node_count_refined": refined["path_node_count"],
                "path_comparison": _path_comparison(native, refined),
                "continuous_feasible_both": bool(
                    native["continuous_feasible"] and refined["continuous_feasible"]
                ),
            }

        critical_margins = {}
        for resolution in RESOLUTIONS:
            critical_margins[resolution] = float(
                by_resolution[resolution]["stackelberg"]["defender_objective"]
                - by_resolution[resolution]["coverage_only"]["defender_objective"]
            )
        objective_uncertainty = max(
            abs(change["defender_objective_change"])
            for change in candidate_changes.values()
        )
        critical_sign_stable = (
            critical_margins["native"] == 0.0
            or critical_margins["refined_2x"] == 0.0
            or math.copysign(1.0, critical_margins["native"])
            == math.copysign(1.0, critical_margins["refined_2x"])
        )
        critical_margin_resolved = bool(
            min(abs(value) for value in critical_margins.values())
            > objective_uncertainty
        )
        terrain_analyses[terrain_name] = {
            "complete": True,
            "rankings": rankings,
            "top_candidate_stable": rankings["native"][0]
            == rankings["refined_2x"][0],
            "complete_ranking_stable": rankings["native"]
            == rankings["refined_2x"],
            "critical_stackelberg_minus_coverage_margin": critical_margins,
            "critical_margin_sign_stable": critical_sign_stable,
            "maximum_candidate_objective_resolution_shift": objective_uncertainty,
            "critical_margin_exceeds_resolution_shift": critical_margin_resolved,
            "critical_pair_interpretation": (
                "resolved_ordering"
                if critical_sign_stable and critical_margin_resolved
                else "numerically_unresolved_tie"
                if not critical_margin_resolved
                else "resolution_sensitive_ordering"
            ),
            "all_continuous_replays_feasible": all(
                record["continuous_feasible"]
                for resolution_records in by_resolution.values()
                for record in resolution_records.values()
            ),
            "candidate_changes": candidate_changes,
        }

    complete = all(item.get("complete", False) for item in terrain_analyses.values())
    all_feasible = complete and all(
        item["all_continuous_replays_feasible"]
        for item in terrain_analyses.values()
    )
    no_material_instability = complete and all(
        item["critical_margin_sign_stable"]
        or item["critical_pair_interpretation"] == "numerically_unresolved_tie"
        for item in terrain_analyses.values()
    )
    native_resolves_all_critical_orderings = complete and all(
        item["critical_margin_sign_stable"]
        and item["critical_margin_exceeds_resolution_shift"]
        for item in terrain_analyses.values()
    )
    return {
        "schema_name": "MeshAdequacyPreflightAnalysis",
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "all_continuous_replays_feasible": all_feasible,
        "no_material_critical_ranking_instability": no_material_instability,
        "native_grid_resolves_all_critical_orderings": (
            native_resolves_all_critical_orderings
        ),
        "p2_grid_decision": (
            "retain_native"
            if native_resolves_all_critical_orderings
            else "promote_refined_2x_and_report_unresolved_near_ties"
        ),
        "terrain_analyses": terrain_analyses,
    }


def _load_checkpoint(restart: bool) -> list[dict[str, Any]]:
    if restart or not CHECKPOINT_PATH.exists():
        return []
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise ValueError(f"Invalid checkpoint format: {CHECKPOINT_PATH}")
    return loaded


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrains",
        nargs="+",
        choices=tuple(SENSOR_CANDIDATES),
        default=list(SENSOR_CANDIDATES),
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        choices=RESOLUTIONS,
        default=list(RESOLUTIONS),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore the existing checkpoint for the selected run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_checkpoint(args.restart)
    completed_keys = {
        (record.get("terrain"), record.get("resolution"), record.get("candidate"))
        for record in records
        if record.get("status_success", False)
    }

    for terrain_name in args.terrains:
        for resolution_name in args.resolutions:
            for candidate_name, z_sensor in SENSOR_CANDIDATES[terrain_name].items():
                key = (terrain_name, resolution_name, candidate_name)
                if key in completed_keys:
                    print(f"SKIP completed {key}", flush=True)
                    continue
                run_id = f"{terrain_name}-{resolution_name}-{candidate_name}"
                run_root = OUTPUT_DIR / terrain_name / resolution_name / candidate_name
                configuration_bundle = build_preflight_config(
                    terrain_name, resolution_name, run_root
                )
                primary = configuration_bundle["primary_result"]
                logger = primary["logging_utilities"]["logger"]
                grid = primary["environment_config"]["grid"]
                started = datetime.now(timezone.utc)
                print(
                    f"[{started.isoformat()}] START {run_id} "
                    f"z_sensor={z_sensor:.6f} grid={grid['z_count']}x{grid['h_count']} "
                    f"spacing=({grid['z_spacing']:.6f},{grid['h_spacing']:.6f})",
                    flush=True,
                )
                start = time.perf_counter()
                try:
                    result = evaluate_defender_position(
                        z_sensor, configuration_bundle, run_id
                    )
                    elapsed = time.perf_counter() - start
                    summary = summarize_evaluation(result, configuration_bundle)
                    summary.update(
                        {
                            "terrain": terrain_name,
                            "resolution": resolution_name,
                            "candidate": candidate_name,
                            "z_sensor": float(z_sensor),
                            "elapsed_seconds": float(elapsed),
                            "grid": {
                                "z_count": int(grid["z_count"]),
                                "h_count": int(grid["h_count"]),
                                "z_spacing": float(grid["z_spacing"]),
                                "h_spacing": float(grid["h_spacing"]),
                            },
                            "successor_direction_signature": successor_direction_signature(
                                configuration_bundle
                            ),
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
                    completed_keys.add(key)
                    print(
                        f"[{run_id}] DONE elapsed={elapsed:.1f}s "
                        f"J_D={summary['defender_objective']:.8f} "
                        f"J_A={summary['mission_cost']:.8f} "
                        f"feasible={summary['continuous_feasible']}",
                        flush=True,
                    )
                except Exception:
                    elapsed = time.perf_counter() - start
                    error = traceback.format_exc()
                    print(f"[{run_id}] FAILED after {elapsed:.1f}s\n{error}", flush=True)
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
                    records.append(
                        {
                            "terrain": terrain_name,
                            "resolution": resolution_name,
                            "candidate": candidate_name,
                            "z_sensor": float(z_sensor),
                            "status_success": False,
                            "elapsed_seconds": float(elapsed),
                            "error": error,
                            "provenance": build_result_provenance(
                                configuration_bundle,
                                script_identifier=SCRIPT_IDENTIFIER,
                            ),
                        }
                    )
                finally:
                    close_phase_logger(logger)
                    records.sort(
                        key=lambda record: (
                            record.get("terrain", ""),
                            record.get("resolution", ""),
                            record.get("candidate", ""),
                        )
                    )
                    _write_json(CHECKPOINT_PATH, records)
                    _write_json(ANALYSIS_PATH, analyze_results(records))

    analysis = analyze_results(records)
    _write_json(ANALYSIS_PATH, analysis)
    print(f"Results: {CHECKPOINT_PATH}", flush=True)
    print(f"Analysis: {ANALYSIS_PATH}", flush=True)
    print(
        f"complete={analysis['complete']} "
        f"all_continuous_feasible={analysis['all_continuous_replays_feasible']} "
        f"p2_grid_decision={analysis['p2_grid_decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
