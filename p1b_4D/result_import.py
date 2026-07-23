"""Standardized read-only import utilities for exported result collections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def import_result_collection(manifest_path: Path) -> dict[str, Any]:
    """Load every exported JSON+NPZ pair without invoking computation modules."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Result collection manifest not found: {path}")
    collection = json.loads(path.read_text(encoding="utf-8"))
    bundles: dict[str, dict[str, Any]] = {}
    missing = list(collection.get("missing_bundles", []))
    failures: list[str] = []
    for item in collection["bundle_exports"]:
        name = item["bundle_name"]
        if item.get("status") != "exported":
            continue
        json_path = Path(item["json_path"])
        npz_path = Path(item["npz_path"])
        if not json_path.is_file() or not npz_path.is_file():
            failures.append(f"missing_files:{name}")
            continue
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        arrays: dict[str, np.ndarray] = {}
        with np.load(npz_path, allow_pickle=False) as payload:
            for array_name, declaration in manifest["array_references"].items():
                if array_name not in payload:
                    failures.append(f"missing_array:{name}:{array_name}")
                    continue
                value = np.array(payload[array_name], copy=True)
                if list(value.shape) != declaration["shape"] or str(value.dtype) != declaration["dtype"]:
                    failures.append(f"array_mismatch:{name}:{array_name}")
                value.setflags(write=False)
                arrays[array_name] = value
        bundles[name] = {"manifest": manifest, "arrays": arrays}
    passed = not failures
    return {
        "primary_result": {
            "collection_manifest": collection,
            "bundles": bundles,
            "missing_bundles": tuple(missing),
        },
        "validation": {
            "passed": passed,
            "checks": {
                "manifest_readable": True,
                "available_bundle_files_readable": passed,
                "array_metadata_consistent": passed,
            },
            "metrics": {
                "loaded_bundle_count": len(bundles),
                "missing_bundle_count": len(missing),
            },
            "warnings": [f"Missing exported bundle: {name}" for name in missing],
            "failed_checks": failures,
            "summary": "Standard result import passed" if passed else f"Import failed: {failures}",
        },
        "metadata": {
            "schema_name": "ImportedResultCollection",
            "schema_version": "1.0.0",
            "source_manifest": str(path),
            "read_only": True,
        },
        "status": {
            "success": passed,
            "code": "OK" if passed else "RESULT_IMPORT_INVALID",
            "message": "Available standardized bundles imported" if passed else "Import failed",
            "warnings": [f"Missing exported bundle: {name}" for name in missing],
            "failed_checks": failures,
        },
    }
