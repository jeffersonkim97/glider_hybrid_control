"""Persistence for the unfiltered Phase 6 Bellman candidate set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_bellman_candidate_bundle(
    bellman_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "bellman_candidate_bundle",
) -> dict[str, Any]:
    """Export candidate metadata to JSON and packed numeric profiles to NPZ."""
    _require_successful_bundle(bellman_bundle, "bellman_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a portable filename stem")
    result = bellman_bundle["primary_result"]
    if result["filtering_applied"] or result["ranking_applied"]:
        raise ValueError("Phase 6 export accepts only unfiltered, unranked sets")

    candidates = result["candidates"]
    trajectories, trajectory_offsets = _pack(
        [candidate["trajectory"] for candidate in candidates], 2
    )
    speeds, profile_offsets = _pack(
        [candidate["speed_profile"] for candidate in candidates], 1
    )
    gammas, gamma_offsets = _pack(
        [candidate["gamma_profile"] for candidate in candidates], 1
    )
    powered_paths, powered_offsets = _pack(
        [candidate["powered_path"] for candidate in candidates], 2
    )
    arrays = {
        "switching_points": np.asarray(
            [candidate["switching_point"] for candidate in candidates],
            dtype=float,
        ).reshape(-1, 2),
        "trajectory_points": trajectories,
        "trajectory_offsets": trajectory_offsets,
        "speed_profiles": speeds,
        "gamma_profiles": gammas,
        "profile_offsets": profile_offsets,
        "gamma_offsets": gamma_offsets,
        "powered_path_points": powered_paths,
        "powered_path_offsets": powered_offsets,
        "mission_costs": np.asarray(
            [candidate["mission_cost"] for candidate in candidates]
        ),
        "powered_times": np.asarray(
            [candidate["powered_time"] for candidate in candidates]
        ),
        "glide_times": np.asarray(
            [candidate["glide_time"] for candidate in candidates]
        ),
        "mission_pods": np.asarray(
            [candidate["mission_pod"] for candidate in candidates]
        ),
    }
    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    array_manifest = {
        name: {"npz_key": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    candidate_metadata = [
        {
            "candidate_id": candidate["candidate_id"],
            "start_id": candidate["start_id"],
            "switching_point": candidate["switching_point"],
            "mission_cost": candidate["mission_cost"],
            "powered_time": candidate["powered_time"],
            "glide_time": candidate["glide_time"],
            "mission_time": candidate["mission_time"],
            "mission_pod": candidate["mission_pod"],
            "objective_breakdown": candidate["objective_breakdown"],
            "hazard_breakdown": candidate["hazard_breakdown"],
            "metadata": candidate["metadata"],
            "validation": candidate["validation"],
        }
        for candidate in candidates
    ]
    manifest = {
        "bundle_id": "bellman-candidates-attacker-pod-time-v1",
        "bundle_type": "BellmanCandidateBundle",
        "schema_name": bellman_bundle["metadata"]["schema_name"],
        "schema_version": bellman_bundle["metadata"]["schema_version"],
        "producer_phase": 6,
        "producer_module": "p1b_4D.bellman_io",
        "metadata": bellman_bundle["metadata"],
        "candidate_metadata": candidate_metadata,
        "start_attempts": result["start_attempts"],
        "summary_statistics": bellman_bundle["validation"]["metrics"],
        "validation": bellman_bundle["validation"],
        "filtering_applied": False,
        "ranking_applied": False,
        "payloads": {"npz": {"path": f"../npz/{npz_path.name}", "arrays": array_manifest}},
        "paths": {"json": json_path.name, "npz": f"../npz/{npz_path.name}"},
    }
    write_json_npz(json_path, npz_path, manifest, arrays)
    passed = json_path.is_file() and npz_path.is_file()
    return {
        "primary_result": {
            "bundle_id": manifest["bundle_id"],
            "json_path": json_path,
            "npz_path": npz_path,
            "array_manifest": array_manifest,
        },
        "validation": {
            "passed": passed,
            "checks": {"json_exists": json_path.is_file(), "npz_exists": npz_path.is_file()},
            "metrics": {"candidate_count": len(candidates), "array_count": len(arrays)},
            "summary": "Bellman Candidate Bundle export completed",
        },
        "metadata": {"schema_name": "ExportManifest", "schema_version": "1.0.0"},
        "status": {
            "success": passed,
            "code": "OK" if passed else "BELLMAN_EXPORT_FAILED",
            "message": "Bellman Candidate Bundle exported" if passed else "Export failed",
            "warnings": [],
            "failed_checks": [] if passed else ["bundle_files_exist"],
        },
    }


def import_bellman_candidate_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate the packed Bellman candidate export."""
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bellman JSON manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "BellmanCandidateBundle":
        raise ValueError("Manifest is not a BellmanCandidateBundle")
    if manifest.get("filtering_applied") or manifest.get("ranking_applied"):
        raise ValueError("Phase 6 manifest must be unfiltered and unranked")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    declared = manifest["payloads"]["npz"]["arrays"]
    arrays: dict[str, np.ndarray] = {}
    failures: list[str] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, specification in declared.items():
            if name not in payload:
                failures.append(f"missing_array:{name}")
                continue
            value = np.array(payload[name], copy=True)
            if list(value.shape) != specification["shape"]:
                failures.append(f"shape_mismatch:{name}")
            if str(value.dtype) != specification["dtype"]:
                failures.append(f"dtype_mismatch:{name}")
            value.setflags(write=False)
            arrays[name] = value
    candidate_count = len(manifest["candidate_metadata"])
    offsets_valid = all(
        arrays[name].size == candidate_count + 1
        for name in ("trajectory_offsets", "profile_offsets", "gamma_offsets", "powered_path_offsets")
    )
    if not offsets_valid:
        failures.append("offset_count_mismatch")
    passed = not failures
    return {
        "primary_result": {"manifest": manifest, "arrays": arrays},
        "validation": {
            "passed": passed,
            "checks": {"payload_matches_manifest": passed, "offsets_valid": offsets_valid},
            "metrics": {"candidate_count": candidate_count, "loaded_array_count": len(arrays)},
            "summary": "Bellman Candidate Bundle import passed" if passed else str(failures),
        },
        "metadata": {"schema_name": manifest["schema_name"], "schema_version": manifest["schema_version"]},
        "status": {
            "success": passed,
            "code": "OK" if passed else "BELLMAN_IMPORT_INVALID",
            "message": "Bellman Candidate Bundle imported" if passed else "Import failed",
            "warnings": [],
            "failed_checks": failures,
        },
    }


def _pack(arrays: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([np.asarray(array).shape[0] for array in arrays], dtype=np.int64)
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths)))
    if not arrays:
        return np.empty((0, width)) if width > 1 else np.empty(0), offsets
    return np.concatenate(arrays, axis=0), offsets


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
