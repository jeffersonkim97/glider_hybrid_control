"""Persistence for a completed continuous Stackelberg solution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_stackelberg_solution_bundle(
    stackelberg_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "stackelberg_solution_bundle",
) -> dict[str, Any]:
    """Export final objectives/metadata to JSON and numerical results to NPZ."""
    _require_successful(stackelberg_bundle, "stackelberg_bundle")
    _require_successful(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a portable filename stem")
    result = stackelberg_bundle["primary_result"]
    solution = result["final_stackelberg_solution"]
    attacker = solution["optimal_attacker_strategy"]
    summaries = result["outer_evaluation_summaries"]
    arrays = {
        "optimal_trajectory": np.asarray(solution["optimal_glide_trajectory"]),
        "optimal_switching_point": np.asarray(solution["optimal_switching_point"]),
        "optimal_sensor_position": np.asarray(solution["optimal_sensor_position"]),
        "optimal_velocity_profile": np.asarray(attacker["speed_profile"]),
        "optimal_gamma_profile": np.asarray(attacker["gamma_profile"]),
        "coverage_los_mask": np.asarray(solution["coverage_maps"]["los_mask"]),
        "coverage_occlusion_mask": np.asarray(solution["coverage_maps"]["occlusion_mask"]),
        "coverage_terrain_mask": np.asarray(solution["coverage_maps"]["terrain_mask"]),
        "outer_z_sensor": np.asarray([item["z_sensor"] for item in summaries]),
        "outer_h_sensor": np.asarray([item["h_sensor"] for item in summaries]),
        "outer_defender_objective": np.asarray([item["defender_objective"] for item in summaries]),
        "outer_attacker_objective": np.asarray([item["attacker_objective"] for item in summaries]),
        "outer_mission_pod": np.asarray([item["mission_pod"] for item in summaries]),
        "outer_coverage_normalized": np.asarray([item["coverage_area_normalized"] for item in summaries]),
        "constraint_goal_error": np.asarray(attacker["constraint_residuals"]["goal_error"]),
    }
    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    array_manifest = {
        name: {"npz_key": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    config = configuration_bundle["primary_result"]
    manifest = {
        "bundle_id": "continuous-stackelberg-solution-v1",
        "bundle_type": "StackelbergSolutionBundle",
        "schema_name": stackelberg_bundle["metadata"]["schema_name"],
        "schema_version": stackelberg_bundle["metadata"]["schema_version"],
        "producer_phase": 9,
        "producer_module": "p1b_4D.stackelberg_io",
        "configuration": {
            "environment_config": config["environment_config"],
            "sensor_config": config["sensor_config"],
            "cost_config": config["cost_config"],
            "defender_config": config["defender_config"],
        },
        "final_objectives": {
            "attacker_objective": solution["attacker_objective"],
            "defender_objective": solution["defender_objective"],
            "objective_breakdown": solution["objective_breakdown"],
        },
        "final_solution": {
            "optimal_z_sensor": solution["optimal_z_sensor"],
            "optimal_h_sensor": solution["optimal_h_sensor"],
            "mission_pod": solution["mission_pod"],
            "coverage_area": solution["coverage_area"],
            "coverage_area_normalized": solution["coverage_area_normalized"],
            "optimal_switching_point": solution["optimal_switching_point"],
        },
        "solver_metadata": {
            "outer": result["outer_optimizer_result"],
            "attacker": {
                "solution_method": "bellman_dynamic_programming",
                "success": attacker["validation"]["passed"],
            },
            "stackelberg": stackelberg_bundle["metadata"],
        },
        "validation": stackelberg_bundle["validation"],
        "summary_statistics": {
            "outer_evaluation_count": len(summaries),
            "fresh_nested_solve_count": sum(item["fresh_nested_attacker_solve"] for item in summaries),
        },
        "payloads": {"npz": {"path": f"../npz/{npz_path.name}", "arrays": array_manifest}},
        "paths": {"json": json_path.name, "npz": f"../npz/{npz_path.name}"},
    }
    write_json_npz(json_path, npz_path, manifest, arrays)
    passed = json_path.is_file() and npz_path.is_file()
    return {
        "primary_result": {"bundle_id": manifest["bundle_id"], "json_path": json_path, "npz_path": npz_path, "array_manifest": array_manifest},
        "validation": {"passed": passed, "checks": {"json_exists": json_path.is_file(), "npz_exists": npz_path.is_file()}, "metrics": {"array_count": len(arrays)}, "summary": "Stackelberg Solution Bundle export completed"},
        "metadata": {"schema_name": "ExportManifest", "schema_version": "1.0.0"},
        "status": {"success": passed, "code": "OK" if passed else "STACKELBERG_EXPORT_FAILED", "message": "Stackelberg Solution Bundle exported" if passed else "Export failed", "warnings": [], "failed_checks": [] if passed else ["bundle_files_exist"]},
    }


def import_stackelberg_solution_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate a persisted Stackelberg solution."""
    path = Path(json_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "StackelbergSolutionBundle":
        raise ValueError("Manifest is not a StackelbergSolutionBundle")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    arrays: dict[str, np.ndarray] = {}
    failures: list[str] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, specification in manifest["payloads"]["npz"]["arrays"].items():
            if name not in payload:
                failures.append(f"missing_array:{name}")
                continue
            value = np.array(payload[name], copy=True)
            if list(value.shape) != specification["shape"] or str(value.dtype) != specification["dtype"]:
                failures.append(f"manifest_mismatch:{name}")
            value.setflags(write=False)
            arrays[name] = value
    passed = not failures
    return {
        "primary_result": {"manifest": manifest, "arrays": arrays},
        "validation": {"passed": passed, "checks": {"payload_matches_manifest": passed}, "metrics": {"loaded_array_count": len(arrays)}, "summary": "Stackelberg Solution Bundle import passed" if passed else str(failures)},
        "metadata": {"schema_name": manifest["schema_name"], "schema_version": manifest["schema_version"]},
        "status": {"success": passed, "code": "OK" if passed else "STACKELBERG_IMPORT_INVALID", "message": "Stackelberg Solution Bundle imported" if passed else "Import failed", "warnings": [], "failed_checks": failures},
    }


def _require_successful(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
