"""JSON and NPZ persistence for the filtered Top-K Bellman candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_filtered_bellman_bundle(
    filtered_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "filtered_bellman_bundle",
) -> dict[str, Any]:
    """Export ranking metadata and packed Top-K warm-start arrays."""
    _require_successful(filtered_bundle, "filtered_bundle")
    _require_successful(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a portable filename stem")
    result = filtered_bundle["primary_result"]
    candidates = result["candidates"]
    trajectories, trajectory_offsets = _pack([item["trajectory"] for item in candidates], 2)
    speeds, profile_offsets = _pack([item["speed_profile"] for item in candidates], 1)
    gammas, gamma_offsets = _pack([item["gamma_profile"] for item in candidates], 1)
    arrays = {
        "switching_points": np.asarray([item["switching_point"] for item in candidates], dtype=float).reshape(-1, 2),
        "trajectory_points": trajectories,
        "trajectory_offsets": trajectory_offsets,
        "speed_profiles": speeds,
        "gamma_profiles": gammas,
        "profile_offsets": profile_offsets,
        "gamma_offsets": gamma_offsets,
        "mission_costs": np.asarray([item["mission_cost"] for item in candidates]),
        "mission_pods": np.asarray([item["mission_pod"] for item in candidates]),
        "mission_times": np.asarray([item["mission_time"] for item in candidates]),
        "ranks": np.asarray([item["rank"] for item in candidates], dtype=np.int64),
    }
    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    array_manifest = {
        name: {"npz_key": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    manifest = {
        "bundle_id": "filtered-bellman-top-k-attacker-pod-time-v1",
        "bundle_type": "FilteredBellmanBundle",
        "schema_name": filtered_bundle["metadata"]["schema_name"],
        "schema_version": filtered_bundle["metadata"]["schema_version"],
        "producer_phase": 7,
        "producer_module": "p1b_4D.candidate_filtering_io",
        "metadata": filtered_bundle["metadata"],
        "ranking": result["ranking"],
        "candidate_metadata": [_candidate_metadata(item) for item in candidates],
        "duplicate_records": result["duplicate_records"],
        "objective_summary": filtered_bundle["validation"]["metrics"],
        "validation": filtered_bundle["validation"],
        "only_attacker_nlp_warm_start_source": True,
        "payloads": {"npz": {"path": f"../npz/{npz_path.name}", "arrays": array_manifest}},
        "paths": {"json": json_path.name, "npz": f"../npz/{npz_path.name}"},
    }
    write_json_npz(json_path, npz_path, manifest, arrays)
    passed = json_path.is_file() and npz_path.is_file()
    return {
        "primary_result": {"bundle_id": manifest["bundle_id"], "json_path": json_path, "npz_path": npz_path, "array_manifest": array_manifest},
        "validation": {"passed": passed, "checks": {"json_exists": json_path.is_file(), "npz_exists": npz_path.is_file()}, "metrics": {"candidate_count": len(candidates)}, "summary": "Filtered Bellman Bundle export completed"},
        "metadata": {"schema_name": "ExportManifest", "schema_version": "1.0.0"},
        "status": {"success": passed, "code": "OK" if passed else "FILTERED_EXPORT_FAILED", "message": "Filtered Bellman Bundle exported" if passed else "Export failed", "warnings": [], "failed_checks": [] if passed else ["bundle_files_exist"]},
    }


def import_filtered_bellman_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate the filtered Bellman JSON/NPZ pair."""
    path = Path(json_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "FilteredBellmanBundle":
        raise ValueError("Manifest is not a FilteredBellmanBundle")
    if not manifest.get("only_attacker_nlp_warm_start_source", False):
        raise ValueError("Filtered bundle lacks the exclusive warm-start marker")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    failures: list[str] = []
    arrays: dict[str, np.ndarray] = {}
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
    count = len(manifest["candidate_metadata"])
    if arrays.get("trajectory_offsets", np.empty(0)).size != count + 1:
        failures.append("trajectory_offset_count")
    if arrays.get("profile_offsets", np.empty(0)).size != count + 1:
        failures.append("profile_offset_count")
    passed = not failures
    return {
        "primary_result": {"manifest": manifest, "arrays": arrays},
        "validation": {"passed": passed, "checks": {"payload_matches_manifest": passed}, "metrics": {"candidate_count": count}, "summary": "Filtered Bellman Bundle import passed" if passed else str(failures)},
        "metadata": {"schema_name": manifest["schema_name"], "schema_version": manifest["schema_version"]},
        "status": {"success": passed, "code": "OK" if passed else "FILTERED_IMPORT_INVALID", "message": "Filtered Bellman Bundle imported" if passed else "Import failed", "warnings": [], "failed_checks": failures},
    }


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate[key] for key in (
        "candidate_id", "rank", "switching_point", "mission_cost", "mission_pod",
        "mission_time", "metadata", "validation",
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
