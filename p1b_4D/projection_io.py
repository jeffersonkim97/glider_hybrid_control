"""Dedicated Projected Cost Bundle export and import functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_projected_cost_bundle(
    projection_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "projected_cost_bundle",
) -> dict[str, Any]:
    """Serialize projected cost, diagnostic controls, indices, and mask.

    Inputs
    ------
    projection_bundle:
        Successful visualization-only Projection2DResult.
    configuration_bundle:
        Successful Phase 1 bundle providing centralized paths.
    bundle_name:
        Portable filename stem.

    Outputs
    -------
    dict
        Universal ExportManifest envelope.

    Assumptions
    -----------
    Projection validation passed and visualization_only is true.

    Notes
    -----
    File writing is this function's only responsibility.
    """
    _require_successful_bundle(projection_bundle, "projection_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if not projection_bundle["metadata"].get("visualization_only", False):
        raise ValueError("Projection Bundle must be marked visualization_only")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a non-empty portable filename stem")

    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    result = projection_bundle["primary_result"]
    arrays = {
        "projected_cost": result["projected_cost"],
        "optimal_velocity": result["optimal_velocity"],
        "optimal_gamma": result["optimal_gamma"],
        "optimal_velocity_index": result["optimal_velocity_index"],
        "optimal_gamma_index": result["optimal_gamma_index"],
        "projection_mask": result["projection_mask"],
        "z_grid": result["grids"]["z"],
        "h_grid": result["grids"]["h"],
    }
    array_manifest = {
        name: {
            "npz_key": name,
            "dtype": str(np.asarray(values).dtype),
            "shape": list(np.asarray(values).shape),
        }
        for name, values in arrays.items()
    }
    config = configuration_bundle["primary_result"]
    manifest = {
        "bundle_id": (
            "projected-cost-"
            f"{projection_bundle['metadata']['source_attacker_objective_id']}-v1"
        ),
        "bundle_type": "ProjectedCostBundle",
        "schema_name": projection_bundle["metadata"]["schema_name"],
        "schema_version": projection_bundle["metadata"]["schema_version"],
        "producer_phase": 5,
        "producer_module": "p1b_4D.projection_io",
        "metadata": projection_bundle["metadata"],
        "configuration": {
            "environment_config": config["environment_config"],
            "vehicle_config": config["vehicle_config"],
            "cost_config": config["cost_config"],
        },
        "projection_statistics": result["projection_metadata"],
        "validation": projection_bundle["validation"],
        "status": projection_bundle["status"],
        "visualization_only": True,
        "bellman_policy_input": False,
        "payloads": {
            "npz": {
                "path": f"../npz/{npz_path.name}",
                "arrays": array_manifest,
            }
        },
        "paths": {
            "json": json_path.name,
            "npz": f"../npz/{npz_path.name}",
        },
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
            "checks": {
                "json_exists": json_path.is_file(),
                "npz_exists": npz_path.is_file(),
                "visualization_only": manifest["visualization_only"],
                "bellman_policy_input_false": not manifest[
                    "bellman_policy_input"
                ],
            },
            "metrics": {"array_count": len(arrays)},
            "summary": "Projected Cost Bundle export completed",
        },
        "metadata": {
            "schema_name": "ExportManifest",
            "schema_version": "1.0.0",
            "source_bundle_id": manifest["bundle_id"],
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "PROJECTION_EXPORT_FAILED",
            "message": (
                "Projected Cost Bundle exported"
                if passed
                else "Projected Cost Bundle export failed"
            ),
            "warnings": [],
            "failed_checks": [] if passed else ["bundle_files_exist"],
        },
    }


def import_projected_cost_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate a persisted visualization-only projection bundle."""
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Projection JSON manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "ProjectedCostBundle":
        raise ValueError("Manifest is not a ProjectedCostBundle")
    if not manifest.get("visualization_only", False):
        raise ValueError("ProjectedCostBundle lacks visualization-only marker")
    if manifest.get("bellman_policy_input", True):
        raise ValueError("ProjectedCostBundle incorrectly permits Bellman use")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Projection NPZ payload not found: {npz_path}")

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
    passed = not failures
    return {
        "primary_result": {"manifest": manifest, "arrays": arrays},
        "validation": {
            "passed": passed,
            "checks": {
                "payload_matches_manifest": passed,
                "visualization_only": True,
            },
            "metrics": {"loaded_array_count": len(arrays)},
            "summary": (
                "Projected Cost Bundle import passed"
                if passed
                else f"Projected Cost Bundle import failures: {failures}"
            ),
        },
        "metadata": {
            "schema_name": manifest["schema_name"],
            "schema_version": manifest["schema_version"],
            "json_path": str(path),
            "npz_path": str(npz_path),
            "visualization_only": True,
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "PROJECTION_IMPORT_INVALID",
            "message": (
                "Projected Cost Bundle imported" if passed else "Import failed"
            ),
            "warnings": [],
            "failed_checks": failures,
        },
    }


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict):
        raise TypeError(f"{name} must be a dictionary")
    if not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must have successful status")


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
