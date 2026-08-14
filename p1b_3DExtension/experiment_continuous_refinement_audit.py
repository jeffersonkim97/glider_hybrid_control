"""Mesh and multi-start audit for the continuous 3-DOF refinement.

This experiment deliberately does not import or use the legacy 6D-to-3D
projection.  It perturbs only the direct multiple-shooting mesh and geometric
initial guess used by the downstream continuous solver.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_trajectory_refinement import (
    REPO_ROOT,
    _dense_validate,
    solve_continuous_refinement,
)


OUTPUT_DIR = REPO_ROOT / "results" / "extreme_ridge_275_continuous_audit"
MESHES = (35, 50, 70)
INITIAL_GAMMAS_DEG = (5.0, 8.0, 11.0)
INITIAL_TOPOLOGIES = ("south", "north")
COMPARISON_SAMPLES = 401
MAXIMUM_CPU_TIME_S = 45.0


def _audit_cases() -> list[tuple[int, str, float]]:
    """Separate mesh sensitivity from initialization-basin sensitivity."""
    cases = [(mesh, "south", 8.0) for mesh in MESHES]
    cases.extend(
        (50, topology, gamma)
        for topology in INITIAL_TOPOLOGIES
        for gamma in INITIAL_GAMMAS_DEG
        if (topology, gamma) != ("south", 8.0)
    )
    return cases


def _mission_curve(result: dict[str, Any], validation: dict[str, Any]) -> np.ndarray:
    powered_fraction = np.linspace(0.0, 1.0, 301)
    powered_time = powered_fraction * result["powered_time"]
    powered_position = (
        result["launch"][None, :]
        + powered_fraction[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    glide_time = np.asarray(validation["dense_time"], dtype=float)
    glide_position = np.asarray(validation["dense_states"][:3], dtype=float).T
    time = np.concatenate((powered_time, glide_time[1:]))
    position = np.vstack((powered_position, glide_position[1:]))
    normalized_time = time / time[-1]
    query = np.linspace(0.0, 1.0, COMPARISON_SAMPLES)
    return np.column_stack([
        np.interp(query, normalized_time, position[:, dimension])
        for dimension in range(3)
    ])


def _record_success(
    result: dict[str, Any], validation: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "success": True,
        "validation_passed": bool(validation["passed"]),
        "interval_count": int(result["controls"].shape[1]),
        "initial_gamma_deg": result["initial_gamma_deg"],
        "initial_topology": result["initial_topology"],
        "elapsed_seconds": result["elapsed_seconds"],
        "solver_return_status": result["solver_stats"].get("return_status", ""),
        "physical_objective": result["physical_objective"],
        "mission_pod": validation["mission_pod"],
        "mission_time_s": result["powered_time"] + result["glide_time"],
        "powered_time_s": result["powered_time"],
        "glide_time_s": result["glide_time"],
        "powered_gamma_deg": float(np.rad2deg(result["powered_gamma"])),
        "powered_heading_deg": float(np.rad2deg(result["powered_heading"])),
        "switch_position_m": np.asarray(result["switch_state"][:3]).tolist(),
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "post_switch_altitude_gain_m": validation["post_switch_altitude_gain_m"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_bank_deg": validation["maximum_bank_deg"],
        "maximum_roll_rate_deg_s": validation["maximum_roll_rate_deg_s"],
        "goal_error_m": validation["goal_error_m"],
        "validation_checks": validation["checks"],
    }


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-12)


def _summarize(runs: list[dict[str, Any]], curves: dict[str, np.ndarray]) -> dict[str, Any]:
    feasible = [
        run for run in runs
        if run.get("success") and run.get("validation_passed")
    ]
    if not feasible:
        return {
            "passed": False,
            "reason": "No densely validated solution was found.",
            "run_count": len(runs),
            "feasible_count": 0,
        }
    best_by_mesh = {}
    for mesh in MESHES:
        candidates = [run for run in feasible if run["interval_count"] == mesh]
        if candidates:
            best_by_mesh[str(mesh)] = min(
                candidates, key=lambda item: item["physical_objective"]
            )
    reference = best_by_mesh.get(str(max(MESHES)))
    if reference is None:
        reference = min(feasible, key=lambda item: item["physical_objective"])
    reference_curve = curves[reference["run_id"]]
    mesh_comparison = {}
    thresholds = {
        "objective_relative": 0.01,
        "pod_absolute": 0.0002,
        "time_relative": 0.02,
        "switch_distance_m": 50.0,
        "trajectory_rms_m": 25.0,
    }
    for mesh, run in best_by_mesh.items():
        curve = curves[run["run_id"]]
        differences = {
            "objective_relative": _relative_difference(
                run["physical_objective"], reference["physical_objective"]
            ),
            "pod_absolute": abs(run["mission_pod"] - reference["mission_pod"]),
            "time_relative": _relative_difference(
                run["mission_time_s"], reference["mission_time_s"]
            ),
            "switch_distance_m": float(np.linalg.norm(
                np.asarray(run["switch_position_m"])
                - np.asarray(reference["switch_position_m"])
            )),
            "trajectory_rms_m": float(np.sqrt(np.mean(np.sum(
                (curve - reference_curve) ** 2, axis=1
            )))),
        }
        mesh_comparison[mesh] = {
            "run_id": run["run_id"],
            "differences_from_finest": differences,
            "within_thresholds": {
                name: value <= thresholds[name]
                for name, value in differences.items()
            },
        }
    finest_comparison = mesh_comparison.get(str(max(MESHES)), {})
    # The finest mesh is the reference by construction.  Mesh stability is
    # judged on the next-finer practical mesh (N=50) against that reference;
    # N=35 is retained as an intentionally coarser diagnostic.
    practical = mesh_comparison.get("50", finest_comparison)
    mesh_stable = bool(practical) and all(practical["within_thresholds"].values())
    basin_groups = {}
    for mesh in MESHES:
        candidates = [run for run in feasible if run["interval_count"] == mesh]
        if not candidates:
            continue
        objectives = np.asarray([run["physical_objective"] for run in candidates])
        basin_groups[str(mesh)] = {
            "feasible_starts": len(candidates),
            "attempted_starts": sum(
                run["interval_count"] == mesh for run in runs
            ),
            "objective_min": float(np.min(objectives)),
            "objective_max": float(np.max(objectives)),
            "objective_relative_spread": float(
                (np.max(objectives) - np.min(objectives)) / np.min(objectives)
            ),
            "converged_powered_heading_deg": [
                run["powered_heading_deg"] for run in candidates
            ],
        }
    representative_starts = [run for run in runs if run["interval_count"] == 50]
    representative_feasible = [
        run for run in representative_starts
        if run.get("success") and run.get("validation_passed")
    ]
    all_starts_feasible = len(representative_feasible) == len(representative_starts)
    return {
        "passed": mesh_stable and all_starts_feasible,
        "mesh_stable_N50_vs_N70": mesh_stable,
        "all_starts_feasible": all_starts_feasible,
        "run_count": len(runs),
        "feasible_count": len(feasible),
        "thresholds": thresholds,
        "reference_run_id": reference["run_id"],
        "reference_solution": reference,
        "best_by_mesh": best_by_mesh,
        "mesh_comparison": mesh_comparison,
        "initialization_basin_summary": basin_groups,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {}
    for mesh, topology, gamma_deg in _audit_cases():
        run_id = f"N{mesh}_{topology}_gamma{gamma_deg:g}"
        print(f"START {run_id}", flush=True)
        try:
            result = solve_continuous_refinement(
                interval_count=mesh,
                initial_gamma_deg=gamma_deg,
                initial_topology=topology,
                maximum_cpu_time_s=(
                    15.0 if topology == "north" else MAXIMUM_CPU_TIME_S
                ),
            )
            validation = _dense_validate(result)
            record = _record_success(result, validation, run_id)
            runs.append(record)
            curves[run_id] = _mission_curve(result, validation)
            print(
                f"DONE {run_id} objective={record['physical_objective']:.9f} "
                f"PoD={100.0 * record['mission_pod']:.5f}% "
                f"valid={record['validation_passed']}",
                flush=True,
            )
        except Exception as exc:  # retain failures as audit evidence
            runs.append({
                "run_id": run_id,
                "success": False,
                "validation_passed": False,
                "interval_count": mesh,
                "initial_gamma_deg": gamma_deg,
                "initial_topology": topology,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"FAILED {run_id}: {type(exc).__name__}: {exc}", flush=True)
    summary = _summarize(runs, curves)
    payload = {"summary": summary, "runs": runs}
    with (OUTPUT_DIR / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if curves:
        np.savez_compressed(OUTPUT_DIR / "normalized_mission_curves.npz", **curves)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
