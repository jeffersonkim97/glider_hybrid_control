"""Dedicated Detection Bundle export and import functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_detection_bundle(
    detection_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    bundle_name: str = "detection_bundle",
) -> dict[str, Any]:
    """Serialize symbolic metadata and validation samples to JSON and NPZ.

    Inputs
    ------
    detection_bundle:
        Successful Phase 3 symbolic DetectionBundle.
    configuration_bundle:
        Successful Phase 1 bundle providing configuration and paths.
    geometry_bundle:
        Successful Phase 2 bundle providing LOS provenance.
    bundle_name:
        Portable filename stem.

    Outputs
    -------
    dict
        Universal ExportManifest envelope.

    Assumptions
    -----------
    CasADi Functions remain in memory and are described, not serialized as
    executable objects.

    Notes
    -----
    File writing is this function's only responsibility.
    """
    _require_successful_bundle(detection_bundle, "detection_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    _require_successful_bundle(geometry_bundle, "geometry_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a non-empty portable filename stem")

    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    detection_result = detection_bundle["primary_result"]
    geometry_result = geometry_bundle["primary_result"]
    sample_outputs = np.asarray(
        detection_bundle["validation"]["metrics"]["sample_outputs"],
        dtype=float,
    )
    tangent = geometry_result["los_geometry"]
    arrays = {
        "validation_sample_outputs": sample_outputs,
        "sensor_position": geometry_result["sensor_position"],
        "goal_position": geometry_result["goal_position"],
        "los_tangent_point": tangent["tangent_point"],
        "los_parameters": np.array(
            [
                tangent["tangent_point"][0],
                tangent["tangent_slope"],
                tangent["tangent_intercept"],
            ],
            dtype=float,
        ),
    }

    array_manifest = {
        name: {
            "npz_key": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for name, value in arrays.items()
    }
    validation_without_array = {
        **detection_bundle["validation"],
        "metrics": {
            key: value
            for key, value in detection_bundle["validation"]["metrics"].items()
            if key != "sample_outputs"
        },
    }
    manifest = {
        "bundle_id": (
            "detection-"
            f"{detection_bundle['metadata']['attacker_objective_id']}-v1"
        ),
        "bundle_type": "DetectionBundle",
        "schema_name": detection_bundle["metadata"]["schema_name"],
        "schema_version": detection_bundle["metadata"]["schema_version"],
        "producer_phase": 3,
        "producer_module": "p1b_4D.detection_io",
        "symbolic_metadata": {
            "casadi_version": detection_bundle["metadata"]["casadi_version"],
            "standard_symbol_order": detection_bundle["metadata"][
                "standard_symbol_order"
            ],
            "function_metadata": detection_result["function_metadata"],
            "detection_components": detection_result["detection_components"],
        },
        "configuration": {
            "environment_config": configuration_bundle["primary_result"][
                "environment_config"
            ],
            "sensor_config": configuration_bundle["primary_result"][
                "sensor_config"
            ],
            "vehicle_config": configuration_bundle["primary_result"][
                "vehicle_config"
            ],
            "cost_config": configuration_bundle["primary_result"]["cost_config"],
            "validation_config": configuration_bundle["primary_result"][
                "validation_config"
            ],
        },
        "geometry_reference": {
            "schema_version": geometry_bundle["metadata"]["schema_version"],
            "z_sensor": geometry_bundle["metadata"]["z_sensor"],
            "tangent_point": tangent["tangent_point"].tolist(),
            "tangent_slope": tangent["tangent_slope"],
            "tangent_intercept": tangent["tangent_intercept"],
            "normalized_coverage_area": geometry_result["coverage"][
                "normalized_coverage_area"
            ],
        },
        "metadata": detection_bundle["metadata"],
        "validation": validation_without_array,
        "status": detection_bundle["status"],
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
                "function_metadata_present": bool(
                    detection_result["function_metadata"]
                ),
            },
            "metrics": {
                "array_count": len(arrays),
                "function_count": len(detection_result["functions"]),
            },
            "summary": "Detection Bundle export completed",
        },
        "metadata": {
            "schema_name": "ExportManifest",
            "schema_version": "1.0.0",
            "source_bundle_id": manifest["bundle_id"],
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "DETECTION_EXPORT_FAILED",
            "message": (
                "Detection Bundle exported"
                if passed
                else "Detection Bundle export failed"
            ),
            "warnings": [],
            "failed_checks": [] if passed else ["bundle_files_exist"],
        },
    }


def import_detection_bundle(json_path: Path) -> dict[str, Any]:
    """Load Detection Bundle metadata and NPZ validation arrays.

    Inputs
    ------
    json_path:
        Path to the authoritative Detection Bundle JSON manifest.

    Outputs
    -------
    dict
        Imported manifest and arrays with schema validation.

    Assumptions
    -----------
    Executable CasADi functions are reconstructed only by the authoritative
    detection module, never by this storage importer.

    Notes
    -----
    This function performs no detection computation.
    """
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Detection JSON manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "DetectionBundle":
        raise ValueError("Manifest is not a DetectionBundle")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Detection NPZ payload not found: {npz_path}")

    expected = manifest["payloads"]["npz"]["arrays"]
    arrays: dict[str, np.ndarray] = {}
    failures: list[str] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, specification in expected.items():
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
            "checks": {"payload_matches_manifest": passed},
            "metrics": {"loaded_array_count": len(arrays)},
            "summary": (
                "Detection Bundle import passed"
                if passed
                else f"Detection Bundle import failures: {failures}"
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
            "code": "OK" if passed else "DETECTION_IMPORT_INVALID",
            "message": "Detection Bundle imported" if passed else "Import failed",
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
