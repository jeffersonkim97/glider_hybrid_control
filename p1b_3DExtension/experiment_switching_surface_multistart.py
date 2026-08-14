"""LOS-independent continuous switching-surface multistart audit.

The legacy discrete shadow solution remains one initializer.  Additional
powered time, climb-angle, and heading-topology seeds are defined directly in
continuous physical variables and are not restricted to an LOS boundary.
Visible powered flight is charged acoustic + radar + Doppler hazard.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .continuous_flight_dynamics import powered_switch_state
from .continuous_trajectory_refinement import (
    OUTPUT_DIR as CONTINUOUS_REFERENCE_DIR,
    REPO_ROOT,
    _dense_validate,
    solve_continuous_refinement,
)


OUTPUT_DIR = REPO_ROOT / "results" / "switching_surface_multistart"
RESAMPLE_COUNT = 401
CASES = (
    {
        "run_id": "legacy_shadow_seed",
        "source": "legacy_discrete_shadow",
        "topology": "south", "gamma_deg": 8.0, "powered_time_s": 52.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "south_low_near",
        "source": "physical_switching_surface",
        "topology": "south", "gamma_deg": 5.0, "powered_time_s": 40.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "south_low_far",
        "source": "physical_switching_surface",
        "topology": "south", "gamma_deg": 5.0, "powered_time_s": 70.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "south_high_near",
        "source": "physical_switching_surface",
        "topology": "south", "gamma_deg": 11.0, "powered_time_s": 40.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "south_high_far",
        "source": "physical_switching_surface",
        "topology": "south", "gamma_deg": 11.0, "powered_time_s": 50.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "center_low",
        "source": "physical_switching_surface",
        "topology": "center", "gamma_deg": 5.0, "powered_time_s": 40.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "center_mid",
        "source": "physical_switching_surface",
        "topology": "center", "gamma_deg": 8.0, "powered_time_s": 55.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "center_high",
        "source": "physical_switching_surface",
        "topology": "center", "gamma_deg": 11.0, "powered_time_s": 45.0,
        "initialization_source": "discrete", "cpu_limit_s": 45.0,
    },
    {
        "run_id": "north_mirrored_feasible_seed",
        "source": "continuous_mirrored_topology",
        "topology": "north", "gamma_deg": 8.0, "powered_time_s": 52.0,
        "initialization_source": "continuous_solution", "cpu_limit_s": 60.0,
    },
)


def _initial_heading(case: dict[str, Any], discrete_switch: np.ndarray) -> float:
    if case["topology"] == "center":
        return 0.0
    sign = -1.0 if case["topology"] == "south" else 1.0
    return sign * abs(float(np.arctan2(discrete_switch[1], discrete_switch[0])))


def _curve(result: dict[str, Any], validation: dict[str, Any]) -> np.ndarray:
    powered_fraction = np.linspace(0.0, 1.0, 301)
    powered_time = powered_fraction * result["powered_time"]
    powered_position = (
        result["launch"][None, :]
        + powered_fraction[:, None]
        * (np.asarray(result["switch_state"][:3]) - result["launch"])[None, :]
    )
    glide_time = np.asarray(validation["dense_time"])
    glide_position = np.asarray(validation["dense_states"][:3]).T
    times = np.concatenate((powered_time, glide_time[1:]))
    positions = np.vstack((powered_position, glide_position[1:]))
    normalized = times / times[-1]
    query = np.linspace(0.0, 1.0, RESAMPLE_COUNT)
    return np.column_stack([
        np.interp(query, normalized, positions[:, dimension])
        for dimension in range(3)
    ])


def _record(
    case: dict[str, Any], result: dict[str, Any], validation: dict[str, Any],
    initial_switch: np.ndarray,
) -> dict[str, Any]:
    return {
        **case,
        "solver_return_status": result["solver_stats"].get("return_status", ""),
        "solver_elapsed_seconds": result["elapsed_seconds"],
        "dense_validation_passed": validation["passed"],
        "validation_checks": validation["checks"],
        "initial_switch_position_m": initial_switch.tolist(),
        "optimized_switch_position_m": np.asarray(result["switch_state"][:3]).tolist(),
        "optimized_powered_gamma_deg": float(np.rad2deg(result["powered_gamma"])),
        "optimized_powered_heading_deg": float(np.rad2deg(result["powered_heading"])),
        "optimized_powered_time_s": result["powered_time"],
        "glide_time_s": result["glide_time"],
        "mission_time_s": result["powered_time"] + result["glide_time"],
        "physical_objective": result["physical_objective"],
        "mission_pod": validation["mission_pod"],
        "powered_pod": float(1.0 - np.exp(-validation["powered_hazard"])),
        "glide_only_pod": float(1.0 - np.exp(-validation["glide_hazard"])),
        "powered_visible_fraction": validation["powered_visible_fraction"],
        "maximum_powered_los_visibility": validation["maximum_powered_los_visibility"],
        "minimum_terrain_clearance_m": validation["minimum_terrain_clearance_m"],
        "maximum_altitude_m": validation["maximum_altitude_m"],
        "post_switch_altitude_gain_m": validation["post_switch_altitude_gain_m"],
        "goal_error_m": validation["goal_error_m"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(
        REPO_ROOT / "results" / "extreme_ridge_275_fine" / "trajectory_data.npz"
    ) as handle:
        discrete_switch = np.asarray(handle["switching_point"], dtype=float)
    records = []
    curves = {}
    for case in CASES:
        heading = _initial_heading(case, discrete_switch)
        # The initial-switch marker is a diagnostic of the physical seed;
        # the continuous-solution north initializer uses its saved feasible
        # powered state internally but has the same nominal topology.
        initial_switch = powered_switch_state(
            np.zeros(3), 21.0, case["powered_time_s"],
            np.deg2rad(case["gamma_deg"]), heading,
        )[:3]
        print(f"START {case['run_id']}", flush=True)
        try:
            result = solve_continuous_refinement(
                interval_count=50,
                initial_gamma_deg=case["gamma_deg"],
                initial_topology=case["topology"],
                maximum_cpu_time_s=case["cpu_limit_s"],
                initial_powered_time_s=case["powered_time_s"],
                initialization_source=case["initialization_source"],
                continuous_warm_start_dir=CONTINUOUS_REFERENCE_DIR,
                accept_limited_solution=True,
            )
            validation = _dense_validate(result)
            record = _record(case, result, validation, initial_switch)
            records.append(record)
            curves[case["run_id"]] = _curve(result, validation)
            print(
                f"DONE {case['run_id']} valid={validation['passed']} "
                f"objective={record['physical_objective']:.9f} "
                f"PoD={100.0 * record['mission_pod']:.4f}%",
                flush=True,
            )
        except Exception as exc:
            records.append({
                **case,
                "dense_validation_passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "initial_switch_position_m": initial_switch.tolist(),
            })
            print(f"FAILED {case['run_id']}: {exc}", flush=True)

    feasible = [record for record in records if record.get("dense_validation_passed")]
    if not feasible:
        raise RuntimeError("No multistart solution passed dense validation")
    best = min(feasible, key=lambda record: record["physical_objective"])
    best_switch = np.asarray(best["optimized_switch_position_m"])
    best_curve = curves[best["run_id"]]
    for record in feasible:
        record["objective_gap_from_best"] = (
            record["physical_objective"] - best["physical_objective"]
        )
        record["switch_distance_from_best_m"] = float(np.linalg.norm(
            np.asarray(record["optimized_switch_position_m"]) - best_switch
        ))
        record["trajectory_rms_from_best_m"] = float(np.sqrt(np.mean(np.sum(
            (curves[record["run_id"]] - best_curve) ** 2, axis=1
        ))))
    surface_feasible = [
        record for record in feasible
        if record["source"] == "physical_switching_surface"
    ]
    payload = {
        "status_success": True,
        "model_change": (
            "visible powered flight charged acoustic + radar + Doppler hazard; "
            "LOS is no longer a hard switching constraint"
        ),
        "legacy_shadow_seed_retained": True,
        "projection_6d_to_3d_modified": False,
        "projection_used": False,
        "attempt_count": len(records),
        "dense_feasible_count": len(feasible),
        "physical_surface_feasible_count": len(surface_feasible),
        "best_run_id": best["run_id"],
        "best_solution": best,
        "physical_surface_converged_to_legacy_basin": bool(
            surface_feasible
            and max(record["switch_distance_from_best_m"] for record in surface_feasible)
            < 1.0e-3
        ),
        "global_optimality_claimed": False,
        "records": records,
    }
    with (OUTPUT_DIR / "multistart_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2)
    np.savez_compressed(OUTPUT_DIR / "normalized_trajectories.npz", **curves)
    print(json.dumps({
        key: payload[key] for key in (
            "attempt_count", "dense_feasible_count",
            "physical_surface_feasible_count", "best_run_id",
            "physical_surface_converged_to_legacy_basin",
            "global_optimality_claimed",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
