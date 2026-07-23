"""Persistence for every feasible NLP solution and the best-found response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_attacker_nlp_bundle(
    nlp_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "attacker_nlp_bundle",
) -> dict[str, Any]:
    """Export solver metadata to JSON and solution arrays to NPZ."""
    _require_successful(nlp_bundle, "nlp_bundle")
    _require_successful(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a portable filename stem")
    result = nlp_bundle["primary_result"]
    solutions = result["feasible_solutions"]
    best = result["best_found_attacker_response"]
    trajectories, trajectory_offsets = _pack([item["trajectory"] for item in solutions], 2)
    velocities, velocity_offsets = _pack([item["velocity_profile"] for item in solutions], 1)
    gammas, gamma_offsets = _pack([item["gamma_profile"] for item in solutions], 1)
    interval_times, interval_time_offsets = _pack(
        [item["interval_time_profile"] for item in solutions], 1
    )
    dynamic_z, dynamic_offsets = _pack([item["constraint_residuals"]["dynamic_z_residual"] for item in solutions], 1)
    dynamic_h, dynamic_h_offsets = _pack([item["constraint_residuals"]["dynamic_h_residual"] for item in solutions], 1)
    arrays = {
        "switching_points": np.asarray([item["switching_point"] for item in solutions]).reshape(-1, 2),
        "trajectory_points": trajectories,
        "trajectory_offsets": trajectory_offsets,
        "velocity_profiles": velocities,
        "velocity_offsets": velocity_offsets,
        "gamma_profiles": gammas,
        "gamma_offsets": gamma_offsets,
        "interval_time_profiles": interval_times,
        "interval_time_offsets": interval_time_offsets,
        "dynamic_z_residuals": dynamic_z,
        "dynamic_residual_offsets": dynamic_offsets,
        "dynamic_h_residuals": dynamic_h,
        "dynamic_h_residual_offsets": dynamic_h_offsets,
        "mission_objectives": np.asarray([item["mission_objective"] for item in solutions]),
        "mission_pods": np.asarray([item["mission_pod"] for item in solutions]),
        "mission_times": np.asarray([item["mission_time"] for item in solutions]),
        "best_switching_point": np.asarray(best["switching_point"]),
        "best_trajectory": np.asarray(best["trajectory"]),
        "best_velocity_profile": np.asarray(best["velocity_profile"]),
        "best_gamma_profile": np.asarray(best["gamma_profile"]),
        "best_interval_time_profile": np.asarray(
            best["interval_time_profile"]
        ),
        "best_goal_error": np.asarray(best["constraint_residuals"]["goal_error"]),
        "best_dynamic_z_residual": np.asarray(best["constraint_residuals"]["dynamic_z_residual"]),
        "best_dynamic_h_residual": np.asarray(best["constraint_residuals"]["dynamic_h_residual"]),
    }
    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    array_manifest = {
        name: {"npz_key": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    manifest = {
        "bundle_id": "attacker-nlp-best-found-response-v1",
        "bundle_type": "AttackerNLPBundle",
        "schema_name": nlp_bundle["metadata"]["schema_name"],
        "schema_version": nlp_bundle["metadata"]["schema_version"],
        "producer_phase": 8,
        "producer_module": "p1b_4D.attacker_nlp_io",
        "metadata": nlp_bundle["metadata"],
        "solver_metadata": [_solution_metadata(item) for item in solutions],
        "solver_attempts": result["solver_attempts"],
        "objective_summary": nlp_bundle["validation"]["metrics"],
        "best_found_attacker_response": _solution_metadata(best),
        "validation": nlp_bundle["validation"],
        "global_optimum_claim": False,
        "only_attacker_solution_for_defender": True,
        "payloads": {"npz": {"path": f"../npz/{npz_path.name}", "arrays": array_manifest}},
        "paths": {"json": json_path.name, "npz": f"../npz/{npz_path.name}"},
    }
    write_json_npz(json_path, npz_path, manifest, arrays)
    passed = json_path.is_file() and npz_path.is_file()
    return {
        "primary_result": {"bundle_id": manifest["bundle_id"], "json_path": json_path, "npz_path": npz_path, "array_manifest": array_manifest},
        "validation": {"passed": passed, "checks": {"json_exists": json_path.is_file(), "npz_exists": npz_path.is_file()}, "metrics": {"feasible_solution_count": len(solutions), "array_count": len(arrays)}, "summary": "Attacker NLP Bundle export completed"},
        "metadata": {"schema_name": "ExportManifest", "schema_version": "1.0.0"},
        "status": {"success": passed, "code": "OK" if passed else "ATTACKER_NLP_EXPORT_FAILED", "message": "Attacker NLP Bundle exported" if passed else "Export failed", "warnings": [], "failed_checks": [] if passed else ["bundle_files_exist"]},
    }


def import_attacker_nlp_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate a persisted Attacker NLP bundle."""
    path = Path(json_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "AttackerNLPBundle":
        raise ValueError("Manifest is not an AttackerNLPBundle")
    if manifest.get("global_optimum_claim", True):
        raise ValueError("Attacker NLP bundle must not claim a global optimum")
    if not manifest.get("only_attacker_solution_for_defender", False):
        raise ValueError("Bundle lacks Defender-consumer authority marker")
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
        "validation": {"passed": passed, "checks": {"payload_matches_manifest": passed}, "metrics": {"loaded_array_count": len(arrays)}, "summary": "Attacker NLP Bundle import passed" if passed else str(failures)},
        "metadata": {"schema_name": manifest["schema_name"], "schema_version": manifest["schema_version"]},
        "status": {"success": passed, "code": "OK" if passed else "ATTACKER_NLP_IMPORT_INVALID", "message": "Attacker NLP Bundle imported" if passed else "Import failed", "warnings": [], "failed_checks": failures},
    }


def _solution_metadata(solution: dict[str, Any]) -> dict[str, Any]:
    return {key: solution[key] for key in (
        "solution_id", "source_candidate_id", "source_rank", "switching_point",
        "powered_time", "glide_time", "mission_time", "mission_pod",
        "pod_normalized", "time_normalized", "mission_objective",
        "solver_status", "validation", "metadata",
    )}


def _pack(arrays: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(value) for value in arrays], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    if not arrays:
        return (np.empty((0, width)) if width > 1 else np.empty(0)), offsets
    return np.concatenate(arrays, axis=0), offsets


def _require_successful(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items() if key != "solution"}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
