"""Plan item 4 runner (p1b_roadmap_0727.md): 3 terrains x 4 sensor-
selection baselines, each baseline re-evaluated by the SAME authoritative
evaluate_defender_position.

Run from the repo root: python -m p1b_4D.experiment_multiterrain_baselines
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import p1b_4D.experiment_strategic_baselines as sb
from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.stackelberg_solver import evaluate_defender_position
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.result_provenance import (
    build_result_provenance,
    provenance_from_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "results" / "multiterrain_strategic_baselines_refined"
CHECKPOINT_PATH = OUTPUT_DIR / "multiterrain_baseline_checkpoint.json"
RESULT_PATH = OUTPUT_DIR / "multiterrain_baseline_results.json"
ANALYSIS_PATH = OUTPUT_DIR / "multiterrain_baseline_analysis.json"
PAPER_TRANSITION_MODEL = "successor_grid_physical_edge"

# Terrain params confirmed from each scenario's existing geometry_bundle.json
# export (results_scaled_singlehill_wcov05 / results_two_hill /
# results_goal_in_valley). Resolution for two-hill/valley reuses each
# terrain's own already-validated native grid. single_hill's grid was
# corrected 2026-07-28 to the actual production resolution ("fine" =
# current p1b_4D.configuration default) after the grid-snapping bias
# investigation (see experiment_grid_snapping_bias.py and the roadmap's
# item 1 correction) showed "medium" carries a directional undercounting
# bias -- results generated with the earlier medium-resolution single_hill
# config should be treated as superseded, not authoritative.  The grids below
# are the 2x proportional spatial refinement selected by the 2026-07-28 mesh-
# adequacy preflight.  Both dz and dh are halved, preserving the successor-
# edge direction set.
TERRAINS = {
    "single_hill": {
        "z_min": 0.0, "z_max": 5500.0,
        "hills": ({"z_ridge": 2500.0, "h_ridge": 200.0, "width": 400.0},),
        "z_goal": 5000.0,
        "grid": {"z_count": 641, "h_count": 401, "v_count": 5, "gamma_count": 20},
        "h_max": 400.0,
        "sensor_bounds": (2750.0, 4500.0),
        "default_z_sensor": 4000.0,
    },
    "two_hill": {
        "z_min": 0.0, "z_max": 2750.0,
        "hills": (
            {"z_ridge": 1000.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 2000.0, "h_ridge": 50.0, "width": 100.0},
        ),
        "z_goal": 2500.0,
        "grid": {"z_count": 321, "h_count": 201, "v_count": 5, "gamma_count": 20},
        "h_max": 200.0,
        "sensor_bounds": (1300.0, 2200.0),
        "default_z_sensor": 1750.0,
    },
    "goal_in_valley": {
        "z_min": 0.0, "z_max": 4000.0,
        "hills": (
            {"z_ridge": 1500.0, "h_ridge": 100.0, "width": 100.0},
            {"z_ridge": 3500.0, "h_ridge": 100.0, "width": 100.0},
        ),
        "z_goal": 2500.0,  # valley floor, between the two hills
        "grid": {"z_count": 467, "h_count": 201, "v_count": 5, "gamma_count": 20},
        "h_max": 200.0,
        "sensor_bounds": (1700.0, 2200.0),
        "default_z_sensor": 1950.0,
    },
}


def build_terrain_config(terrain_name: str, project_root: Path) -> dict:
    params = TERRAINS[terrain_name]
    cb = build_configuration_bundle(project_root)
    cb = deepcopy(cb)
    cb["primary_result"]["attacker_solver_config"][
        "transition_model"
    ] = PAPER_TRANSITION_MODEL
    env = cb["primary_result"]["environment_config"]
    env["z_start"] = 0.0
    env["z_goal"] = params["z_goal"]
    env["terrain"] = {"z_min": params["z_min"], "z_max": params["z_max"], "hills": params["hills"]}
    grid_params = params["grid"]
    z_min, z_max = params["z_min"], params["z_max"]
    h_min, h_max = 0.0, params["h_max"]
    env["grid"] = {
        "z_min": z_min, "z_max": z_max, "z_count": grid_params["z_count"],
        "z_spacing": (z_max - z_min) / (grid_params["z_count"] - 1),
        "h_min": h_min, "h_max": h_max, "h_count": grid_params["h_count"],
        "h_spacing": (h_max - h_min) / (grid_params["h_count"] - 1),
        "v_count": grid_params["v_count"], "gamma_count": grid_params["gamma_count"],
        "axis_order_4d": ("z", "h", "v", "gamma"),
    }
    env["airspace"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max}
    env["simulation"] = {"z_min": z_min, "z_max": z_max, "h_min": h_min, "h_max": h_max, "max_path_steps": 1000}
    vehicle = cb["primary_result"]["vehicle_config"]
    vehicle["glide_speed_count"] = grid_params["v_count"]
    vehicle["gamma_count"] = grid_params["gamma_count"]
    sensor = cb["primary_result"]["sensor_config"]
    sensor["default_z_sensor"] = params["default_z_sensor"]
    costs = cb["primary_result"]["cost_config"]["attacker"]
    costs["w_pod"] = 0.5
    costs["w_time"] = 0.5
    costs["normalization"]["time"]["reference_seconds"] = params["z_goal"] / vehicle["glide_speed_max"]
    defender = cb["primary_result"]["defender_config"]
    z_sensor_min, z_sensor_max = params["sensor_bounds"]
    defender["continuous_search_bounds"] = {"z_sensor_min": z_sensor_min, "z_sensor_max": z_sensor_max}
    return cb


def evaluate_baseline(configuration_bundle: dict, z_sensor: float, run_id: str) -> dict:
    result = evaluate_defender_position(z_sensor, configuration_bundle, run_id)
    primary = result["primary_result"]
    best = primary["best_found_attacker_response"]
    replay = best["continuous_replay_validation"]
    return {
        "z_sensor_selected": z_sensor,
        "status_success": result["status"]["success"],
        "mission_cost": best["mission_cost"],
        "mission_pod": best["mission_pod"],
        "mission_time": best["mission_time"],
        "defender_objective": primary["defender_objective"],
        "switching_point": list(np.asarray(best["switching_point"]).tolist()),
        "transition_model": configuration_bundle["primary_result"][
            "attacker_solver_config"
        ]["transition_model"],
        "continuous_feasible": replay["feasible"],
        "continuous_violation": replay["violation"],
        "continuous_goal_miss": replay["goal_miss"],
        "provenance": provenance_from_evaluation(
            configuration_bundle,
            result,
            script_identifier="p1b_4D/experiment_multiterrain_baselines.py",
        ),
    }


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _new_checkpoint() -> dict:
    return {
        "schema_name": "RefinedMultiTerrainBaselineCheckpoint",
        "schema_version": "1.0.0",
        "transition_model": PAPER_TRANSITION_MODEL,
        "spatial_grid_tier": "refined_2x",
        "terrains": {},
    }


def _load_checkpoint(restart: bool) -> dict:
    if restart or not CHECKPOINT_PATH.exists():
        return _new_checkpoint()
    with CHECKPOINT_PATH.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if checkpoint.get("schema_name") != "RefinedMultiTerrainBaselineCheckpoint":
        raise ValueError(f"Unexpected checkpoint schema: {CHECKPOINT_PATH}")
    return checkpoint


def _paper_results(checkpoint: dict) -> dict[str, dict[str, dict]]:
    return {
        terrain_name: terrain_state.get("evaluations", {})
        for terrain_name, terrain_state in checkpoint["terrains"].items()
    }


def _save_checkpoint(checkpoint: dict) -> None:
    _write_json(CHECKPOINT_PATH, checkpoint)
    _write_json(RESULT_PATH, _paper_results(checkpoint))
    _write_json(ANALYSIS_PATH, _analyze_checkpoint(checkpoint))


def _analyze_checkpoint(checkpoint: dict) -> dict:
    analyses = {}
    all_complete = True
    all_feasible = True
    for terrain_name in TERRAINS:
        terrain_state = checkpoint["terrains"].get(terrain_name, {})
        evaluations = terrain_state.get("evaluations", {})
        complete = all(
            name in evaluations and evaluations[name].get("status_success", False)
            for name in ("fixed", "coverage_only", "nominal_path", "stackelberg")
        )
        all_complete &= complete
        if not complete:
            analyses[terrain_name] = {"complete": False}
            all_feasible = False
            continue
        feasible = all(item["continuous_feasible"] for item in evaluations.values())
        all_feasible &= feasible
        stackelberg = evaluations["stackelberg"]
        coverage = evaluations["coverage_only"]
        nominal = evaluations["nominal_path"]
        fixed = evaluations["fixed"]
        hashes = {
            item["provenance"]["configuration_hash_sha256"]
            for item in evaluations.values()
        }
        analyses[terrain_name] = {
            "complete": True,
            "all_continuous_feasible": feasible,
            "strategy_ranking": sorted(
                evaluations,
                key=lambda name: -evaluations[name]["defender_objective"],
            ),
            "selected_positions": {
                name: item["z_sensor_selected"] for name, item in evaluations.items()
            },
            "defender_objectives": {
                name: item["defender_objective"] for name, item in evaluations.items()
            },
            "stackelberg_minus_coverage": (
                stackelberg["defender_objective"] - coverage["defender_objective"]
            ),
            "stackelberg_minus_nominal": (
                stackelberg["defender_objective"] - nominal["defender_objective"]
            ),
            "stackelberg_minus_fixed": (
                stackelberg["defender_objective"] - fixed["defender_objective"]
            ),
            "stackelberg_search_evaluation_count": len(
                terrain_state.get("stackelberg_search", {}).get("evaluations", [])
            ),
            "configuration_hash_consistent": len(hashes) == 1,
            "configuration_hash_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
            "resolution": stackelberg["provenance"]["resolution"],
            "selection_reconciliation": terrain_state.get(
                "selection_reconciliation"
            ),
        }
    return {
        "schema_name": "RefinedMultiTerrainBaselineAnalysis",
        "schema_version": "1.0.0",
        "all_terrains_complete": all_complete,
        "all_continuous_replays_feasible": all_feasible,
        "terrains": analyses,
    }


def _reconcile_stackelberg_dominance(terrain_state: dict) -> None:
    """Ensure the reported Stackelberg choice dominates evaluated baselines."""
    evaluations = terrain_state["evaluations"]
    if not all(
        name in evaluations
        and evaluations[name].get("status_success", False)
        for name in ("fixed", "coverage_only", "nominal_path", "stackelberg")
    ):
        return
    best_name = max(
        evaluations,
        key=lambda name: evaluations[name]["defender_objective"],
    )
    if (
        best_name == "stackelberg"
        or evaluations[best_name]["defender_objective"]
        <= evaluations["stackelberg"]["defender_objective"] + 1e-12
    ):
        return
    direct_evaluation = deepcopy(evaluations["stackelberg"])
    promoted_evaluation = deepcopy(evaluations[best_name])
    terrain_state["selection_reconciliation"] = {
        "reason": "explicitly_evaluated_baseline_candidate_dominates_direct_result",
        "direct_z_sensor": direct_evaluation["z_sensor_selected"],
        "direct_defender_objective": direct_evaluation["defender_objective"],
        "promoted_from": best_name,
        "promoted_z_sensor": promoted_evaluation["z_sensor_selected"],
        "promoted_defender_objective": promoted_evaluation["defender_objective"],
        "objective_improvement": (
            promoted_evaluation["defender_objective"]
            - direct_evaluation["defender_objective"]
        ),
    }
    terrain_state["selected_positions"]["stackelberg"] = promoted_evaluation[
        "z_sensor_selected"
    ]
    evaluations["stackelberg"] = promoted_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terrains",
        nargs="+",
        choices=tuple(TERRAINS),
        default=list(TERRAINS),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard this refined experiment's checkpoint and start again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(args.restart)

    for terrain_name in args.terrains:
        params = TERRAINS[terrain_name]
        bounds = params["sensor_bounds"]
        terrain_state = checkpoint["terrains"].setdefault(
            terrain_name,
            {
                "selected_positions": {},
                "stackelberg_search": {"objective_by_z": {}, "evaluations": []},
                "evaluations": {},
            },
        )
        print(f"\n=== TERRAIN: {terrain_name} ===", flush=True)

        cb = build_terrain_config(terrain_name, OUTPUT_DIR / terrain_name)
        logger = cb["primary_result"]["logging_utilities"]["logger"]
        try:
            selected_positions = terrain_state["selected_positions"]

            if "fixed" not in selected_positions:
                t0 = time.perf_counter()
                selected_positions["fixed"] = sb.select_fixed_sensor(cb)
                _save_checkpoint(checkpoint)
                print(f"  [{terrain_name}] fixed selected: {selected_positions['fixed']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            else:
                print(f"  [{terrain_name}] fixed resumed: {selected_positions['fixed']:.2f}", flush=True)

            if "coverage_only" not in selected_positions:
                t0 = time.perf_counter()
                selected_positions["coverage_only"] = sb.select_coverage_only_sensor(cb, bounds)
                _save_checkpoint(checkpoint)
                print(f"  [{terrain_name}] coverage_only selected: {selected_positions['coverage_only']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            else:
                print(f"  [{terrain_name}] coverage_only resumed: {selected_positions['coverage_only']:.2f}", flush=True)

            if "nominal_path" not in selected_positions:
                t0 = time.perf_counter()
                reference_z = 0.5 * (bounds[0] + bounds[1])
                nominal_response = sb.compute_nominal_attacker_path(cb, reference_z)
                selected_positions["nominal_path"] = sb.select_nominal_path_optimal_sensor(cb, bounds, nominal_response)
                _save_checkpoint(checkpoint)
                print(f"  [{terrain_name}] nominal_path selected: {selected_positions['nominal_path']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            else:
                print(f"  [{terrain_name}] nominal_path resumed: {selected_positions['nominal_path']:.2f}", flush=True)

            if "stackelberg" not in selected_positions:
                t0 = time.perf_counter()
                search_state = terrain_state["stackelberg_search"]

                def checkpoint_outer_evaluation(z_sensor: float, objective: float) -> None:
                    search_state["evaluations"].append(
                        {"z_sensor": float(z_sensor), "defender_objective": float(objective)}
                    )
                    _save_checkpoint(checkpoint)
                    print(
                        f"  [{terrain_name}] Stackelberg eval "
                        f"{len(search_state['evaluations']):02d}: z={z_sensor:.6f} "
                        f"J_D={objective:.8f}",
                        flush=True,
                    )

                selected_positions["stackelberg"] = sb.select_stackelberg_optimal_sensor(
                    cb,
                    bounds,
                    evaluation_cache=search_state["objective_by_z"],
                    on_evaluation=checkpoint_outer_evaluation,
                    additional_candidates=tuple(
                        selected_positions[name]
                        for name in ("fixed", "coverage_only", "nominal_path")
                    ),
                )
                _save_checkpoint(checkpoint)
                print(f"  [{terrain_name}] stackelberg selected: {selected_positions['stackelberg']:.2f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            else:
                print(f"  [{terrain_name}] stackelberg resumed: {selected_positions['stackelberg']:.2f}", flush=True)

            for baseline_name, z_sensor in selected_positions.items():
                if (
                    baseline_name in terrain_state["evaluations"]
                    and terrain_state["evaluations"][baseline_name].get(
                        "status_success", False
                    )
                ):
                    print(f"  [{terrain_name}-{baseline_name}] EVAL resumed", flush=True)
                    continue
                run_id = f"{terrain_name}-{baseline_name}"
                try:
                    evaluation = evaluate_baseline(cb, z_sensor, run_id)
                    terrain_state["evaluations"][baseline_name] = evaluation
                    _save_checkpoint(checkpoint)
                    print(
                        f"  [{run_id}] EVAL z={z_sensor:.2f} "
                        f"mission_pod={evaluation['mission_pod']:.4f} "
                        f"defender_objective={evaluation['defender_objective']:.4f}",
                        flush=True,
                    )
                except Exception:
                    print(f"  [{run_id}] EVAL FAILED:", flush=True)
                    traceback.print_exc()
                    terrain_state["evaluations"][baseline_name] = {
                        "status_success": False,
                        "error": traceback.format_exc(),
                        "provenance": build_result_provenance(
                            cb,
                            script_identifier=(
                                "p1b_4D/experiment_multiterrain_baselines.py"
                            ),
                        ),
                    }
                    _save_checkpoint(checkpoint)
            _reconcile_stackelberg_dominance(terrain_state)
            _save_checkpoint(checkpoint)
        except Exception:
            print(f"  [{terrain_name}] TERRAIN-LEVEL FAILURE:", flush=True)
            traceback.print_exc()
        finally:
            close_phase_logger(logger)

        _save_checkpoint(checkpoint)

    print(f"\nRequested terrains complete. Results saved to {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
