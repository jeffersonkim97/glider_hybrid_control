"""Dedicated Geometry Bundle export and import functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_geometry_bundle(
    geometry_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "geometry_bundle",
) -> dict[str, Any]:
    """Serialize a validated Geometry Bundle to JSON and NPZ.

    Inputs
    ------
    geometry_bundle:
        Successful result returned by build_geometry_bundle.
    configuration_bundle:
        Successful Phase 1 bundle providing centralized project paths.
    bundle_name:
        Portable filename stem.

    Outputs
    -------
    dict
        Export status, paths, schema, array manifest, and source metadata.

    Assumptions
    -----------
    Geometry arrays are complete and validation passed.

    Notes
    -----
    File writing is this function's only responsibility. NPZ is completed
    before the authoritative JSON manifest is written.
    """
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a non-empty portable filename stem")

    project_paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = project_paths.json_dir / f"{bundle_name}.json"
    npz_path = project_paths.npz_dir / f"{bundle_name}.npz"
    result = geometry_bundle["primary_result"]
    terrain = result["terrain_arrays"]
    los_geometry = result["los_geometry"]
    masks = result["los_masks"]

    arrays = {
        "terrain_z": terrain["z"],
        "terrain_height": terrain["height"],
        "terrain_gradient": terrain["gradient"],
        "terrain_curvature": terrain["curvature"],
        "airspace_h_grid": terrain["h_grid"],
        "terrain_mask": masks["terrain_mask"],
        "los_mask": masks["los_mask"],
        "occlusion_mask": masks["occlusion_mask"],
        "non_visible_airspace_mask": masks["non_visible_airspace_mask"],
        "sensor_position": result["sensor_position"],
        "goal_position": result["goal_position"],
        "los_tangent_point": los_geometry["tangent_point"],
        "los_tangent_line_height": los_geometry["tangent_line_height"],
        "los_boundary": los_geometry["los_boundary"],
    }

    array_manifest = {
        key: {
            "npz_key": key,
            "dtype": str(np.asarray(value).dtype),
            "shape": list(np.asarray(value).shape),
        }
        for key, value in arrays.items()
    }
    manifest = {
        "bundle_id": (
            f"geometry-zsensor-{float(result['sensor_position'][0]):.10g}-v1"
        ),
        "bundle_type": "GeometryBundle",
        "schema_name": geometry_bundle["metadata"]["schema_name"],
        "schema_version": geometry_bundle["metadata"]["schema_version"],
        "producer_phase": 2,
        "producer_module": "p1b_4D.geometry_io",
        "configuration": {
            "schema_version": configuration_bundle["metadata"]["schema_version"],
            "global_random_seed": configuration_bundle["primary_result"][
                "global_random_seed"
            ],
            "environment_config": configuration_bundle["primary_result"][
                "environment_config"
            ],
            "sensor_config": configuration_bundle["primary_result"][
                "sensor_config"
            ],
            "validation_config": configuration_bundle["primary_result"][
                "validation_config"
            ],
        },
        "metadata": geometry_bundle["metadata"],
        "dimensions": geometry_bundle["metadata"]["dimensions"],
        "validation": geometry_bundle["validation"],
        "status": geometry_bundle["status"],
        "geometry": {
            "sensor_position": result["sensor_position"].tolist(),
            "goal_position": result["goal_position"].tolist(),
            "los_tangent_point": los_geometry["tangent_point"].tolist(),
            "los_tangent_slope": los_geometry["tangent_slope"],
            "los_tangent_intercept": los_geometry["tangent_intercept"],
            "los_tangent_residual": los_geometry["tangent_residual"],
            "coverage": result["coverage"],
        },
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
    return {
        "primary_result": {
            "bundle_id": manifest["bundle_id"],
            "json_path": json_path,
            "npz_path": npz_path,
            "array_manifest": array_manifest,
        },
        "validation": {
            "passed": json_path.is_file() and npz_path.is_file(),
            "checks": {
                "json_exists": json_path.is_file(),
                "npz_exists": npz_path.is_file(),
                "array_count": len(array_manifest) == len(arrays),
            },
            "metrics": {"array_count": len(arrays)},
            "summary": "Geometry Bundle export completed",
        },
        "metadata": {
            "schema_name": "ExportManifest",
            "schema_version": "1.0.0",
            "source_bundle_id": manifest["bundle_id"],
        },
        "status": {
            "success": json_path.is_file() and npz_path.is_file(),
            "code": "OK",
            "message": "Geometry Bundle exported",
            "warnings": [],
            "failed_checks": [],
        },
    }


def import_geometry_bundle(json_path: Path) -> dict[str, Any]:
    """Load and validate a persisted Geometry Bundle without recomputation.

    Inputs
    ------
    json_path:
        Path to the authoritative JSON manifest.

    Outputs
    -------
    dict
        Imported manifest and named NPZ arrays in a universal result envelope.

    Assumptions
    -----------
    The NPZ reference resolves relative to the JSON directory.

    Notes
    -----
    This function performs storage validation only and never builds geometry.
    """
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Geometry JSON manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "GeometryBundle":
        raise ValueError("Manifest is not a GeometryBundle")
    relative_npz = manifest["payloads"]["npz"]["path"]
    npz_path = (path.parent / relative_npz).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Geometry NPZ payload not found: {npz_path}")

    expected = manifest["payloads"]["npz"]["arrays"]
    arrays: dict[str, np.ndarray] = {}
    failed_checks: list[str] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, specification in expected.items():
            if name not in payload:
                failed_checks.append(f"missing_array:{name}")
                continue
            value = np.array(payload[name], copy=True)
            if list(value.shape) != specification["shape"]:
                failed_checks.append(f"shape_mismatch:{name}")
            if str(value.dtype) != specification["dtype"]:
                failed_checks.append(f"dtype_mismatch:{name}")
            value.setflags(write=False)
            arrays[name] = value
    passed = not failed_checks
    return {
        "primary_result": {"manifest": manifest, "arrays": arrays},
        "validation": {
            "passed": passed,
            "checks": {"payload_matches_manifest": passed},
            "metrics": {"loaded_array_count": len(arrays)},
            "summary": (
                "Geometry Bundle import passed"
                if passed
                else f"Geometry Bundle import failures: {failed_checks}"
            ),
        },
        "metadata": {
            "schema_name": manifest["schema_name"],
            "schema_version": manifest["schema_version"],
            "json_path": str(path),
            "npz_path": str(npz_path),
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "GEOMETRY_IMPORT_INVALID",
            "message": "Geometry Bundle imported" if passed else "Import failed",
            "warnings": [],
            "failed_checks": failed_checks,
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
