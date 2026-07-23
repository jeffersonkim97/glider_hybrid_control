"""Dedicated 4D Stage Cost Bundle export and import functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .result_export import write_json_npz


def export_stage_cost_4d_bundle(
    stage_cost_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    bundle_name: str = "stage_cost_4d_bundle",
) -> dict[str, Any]:
    """Serialize validated 4D stage costs, masks, components, and grids.

    Inputs
    ------
    stage_cost_bundle:
        Successful StageCost4DResult.
    configuration_bundle:
        Successful Phase 1 bundle providing centralized project paths.
    bundle_name:
        Portable filename stem.

    Outputs
    -------
    dict
        Universal ExportManifest envelope.

    Assumptions
    -----------
    Stage-cost validation passed and arrays share the declared 4D shape.

    Notes
    -----
    File writing is this function's only responsibility.
    """
    _require_successful_bundle(stage_cost_bundle, "stage_cost_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if not bundle_name or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a non-empty portable filename stem")

    paths = configuration_bundle["primary_result"]["project_paths"]
    json_path = paths.json_dir / f"{bundle_name}.json"
    npz_path = paths.npz_dir / f"{bundle_name}.npz"
    result = stage_cost_bundle["primary_result"]
    arrays: dict[str, np.ndarray] = {
        "j4d": result["j4d"],
        "powered_stage_cost_4d": result["powered_stage_cost_4d"],
        "feasible_mask": result["feasible_mask"],
    }
    arrays.update(
        {
            f"component__{name}": values
            for name, values in result["component_maps"].items()
        }
    )
    arrays.update(
        {
            f"mask__{name}": values
            for name, values in result["validity_masks"].items()
        }
    )
    arrays.update(
        {f"grid__{name}": values for name, values in result["grids"].items()}
    )

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
            f"stage-cost-4d-{stage_cost_bundle['metadata']['attacker_objective_id']}-v1"
        ),
        "bundle_type": "StageCost4DBundle",
        "schema_name": stage_cost_bundle["metadata"]["schema_name"],
        "schema_version": stage_cost_bundle["metadata"]["schema_version"],
        "producer_phase": 4,
        "producer_module": "p1b_4D.stage_cost_io",
        "metadata": stage_cost_bundle["metadata"],
        "grid_information": result["grid_metadata"],
        "configuration": {
            "environment_config": config["environment_config"],
            "vehicle_config": config["vehicle_config"],
            "sensor_config": config["sensor_config"],
            "cost_config": config["cost_config"],
            "validation_config": config["validation_config"],
        },
        "validation": stage_cost_bundle["validation"],
        "objective_summary": {
            "minimum_cost": stage_cost_bundle["validation"]["metrics"][
                "minimum_cost"
            ],
            "maximum_cost": stage_cost_bundle["validation"]["metrics"][
                "maximum_cost"
            ],
            "invalid_state_count": stage_cost_bundle["validation"]["metrics"][
                "invalid_state_count"
            ],
            "feasible_state_count": stage_cost_bundle["validation"]["metrics"][
                "feasible_state_count"
            ],
            "total_state_count": stage_cost_bundle["validation"]["metrics"][
                "total_state_count"
            ],
        },
        "status": stage_cost_bundle["status"],
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
                "j4d_present": "j4d" in array_manifest,
                "feasible_mask_present": "feasible_mask" in array_manifest,
            },
            "metrics": {"array_count": len(arrays)},
            "summary": "4D Stage Cost Bundle export completed",
        },
        "metadata": {
            "schema_name": "ExportManifest",
            "schema_version": "1.0.0",
            "source_bundle_id": manifest["bundle_id"],
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "STAGE_COST_EXPORT_FAILED",
            "message": (
                "4D Stage Cost Bundle exported"
                if passed
                else "4D Stage Cost Bundle export failed"
            ),
            "warnings": [],
            "failed_checks": [] if passed else ["bundle_files_exist"],
        },
    }


def import_stage_cost_4d_bundle(
    json_path: Path,
    array_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load selected persisted 4D arrays without recomputing stage costs.

    Inputs
    ------
    json_path:
        Authoritative JSON manifest path.
    array_names:
        Optional NPZ keys. When omitted, all declared arrays are loaded.

    Outputs
    -------
    dict
        Imported manifest and validated arrays.

    Assumptions
    -----------
    Payload paths are relative to the JSON manifest directory.

    Notes
    -----
    This importer performs no detection or stage-cost computation.
    """
    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-cost JSON manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bundle_type") != "StageCost4DBundle":
        raise ValueError("Manifest is not a StageCost4DBundle")
    npz_path = (path.parent / manifest["payloads"]["npz"]["path"]).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"Stage-cost NPZ payload not found: {npz_path}")

    declared = manifest["payloads"]["npz"]["arrays"]
    selected = tuple(declared) if array_names is None else array_names
    unknown = sorted(set(selected) - set(declared))
    if unknown:
        raise KeyError(f"Undeclared stage-cost arrays requested: {unknown}")
    arrays: dict[str, np.ndarray] = {}
    failures: list[str] = []
    with np.load(npz_path, allow_pickle=False) as payload:
        for name in selected:
            specification = declared[name]
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
                "4D Stage Cost Bundle import passed"
                if passed
                else f"4D Stage Cost Bundle import failures: {failures}"
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
            "code": "OK" if passed else "STAGE_COST_IMPORT_INVALID",
            "message": (
                "4D Stage Cost Bundle imported" if passed else "Import failed"
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
