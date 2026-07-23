"""Phase 10 sole active writer for standardized computational result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


STANDARD_DIRECTORIES = (
    "geometry", "cost", "bellman", "stackelberg", "metadata", "logs"
)


def write_json_npz(
    json_path: Path,
    npz_path: Path,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Central low-level write primitive used by all compatibility adapters."""
    with Path(npz_path).open("wb") as payload:
        np.savez_compressed(payload, **arrays)
    Path(json_path).write_text(
        json.dumps(_json_native(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_all_results(
    configuration_bundle: dict[str, Any],
    geometry_bundle: dict[str, Any],
    detection_bundle: dict[str, Any],
    stage_cost_4d_bundle: dict[str, Any],
    projected_cost_bundle: dict[str, Any],
    bellman_candidate_bundle: dict[str, Any],
    bellman_response_bundle: dict[str, Any],
    stackelberg_solution_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write every available result through one standardized JSON+NPZ system."""
    started = perf_counter()
    _require_successful(configuration_bundle, "configuration_bundle")
    supplied = {
        "geometry": geometry_bundle,
        "detection": detection_bundle,
        "stage_cost": stage_cost_4d_bundle,
        "projected_cost": projected_cost_bundle,
        "bellman": bellman_candidate_bundle,
        "bellman_response": bellman_response_bundle,
        "stackelberg": stackelberg_solution_bundle,
    }
    for name, bundle in supplied.items():
        if bundle is not None:
            _require_successful(bundle, f"{name}_bundle")
    root = configuration_bundle["primary_result"]["project_paths"].results_dir
    directories = {name: root / name for name in STANDARD_DIRECTORIES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    specifications = (
        ("geometry", "geometry", geometry_bundle, _geometry_arrays),
        ("detection", "geometry", detection_bundle, lambda _: _detection_arrays(geometry_bundle)),
        ("stage_cost", "cost", stage_cost_4d_bundle, _stage_arrays),
        ("projected_cost", "cost", projected_cost_bundle, _projection_arrays),
        ("bellman", "bellman", bellman_candidate_bundle, _bellman_arrays),
        ("bellman_response", "bellman", bellman_response_bundle, _bellman_response_arrays),
        ("stackelberg", "stackelberg", stackelberg_solution_bundle, _stackelberg_arrays),
    )
    exports: list[dict[str, Any]] = []
    missing: list[str] = []
    for bundle_name, category, bundle, array_builder in specifications:
        if bundle is None:
            missing.append(bundle_name)
            exports.append({"bundle_name": bundle_name, "status": "missing_input"})
            continue
        arrays = array_builder(bundle)
        exports.append(_write_standard_bundle(
            bundle_name, category, bundle, arrays,
            configuration_bundle, directories[category],
        ))
    master = {
        "schema_name": "StandardResultCollection",
        "schema_version": "1.0.0",
        "producer_phase": 10,
        "standard_directories": {name: str(path.relative_to(root)) for name, path in directories.items()},
        "bundle_exports": exports,
        "missing_bundles": missing,
        "complete": not missing,
        "future_visualization_input_policy": "exported_files_only",
        "execution_time_seconds": perf_counter() - started,
    }
    master_path = directories["metadata"] / "result_collection_manifest.json"
    master_path.write_text(json.dumps(_json_native(master), indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_export_collection(root, directories, exports, master_path, missing)
    generated_files = tuple(
        path for item in exports if item.get("status") == "exported"
        for path in (Path(item["json_path"]), Path(item["npz_path"]))
    ) + (master_path,)
    return {
        "primary_result": {
            "export_status": "complete" if not missing else "incomplete",
            "exports": tuple(exports),
            "generated_files": generated_files,
            "missing_bundles": tuple(missing),
            "master_manifest_path": master_path,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "ResultExportStatus",
            "schema_version": "1.0.0",
            "producer_phase": 10,
            "producer_module": "p1b_4D.result_export",
            "sole_active_disk_writer": True,
            "execution_time_seconds": perf_counter() - started,
        },
        "status": {
            "success": validation["passed"] and not missing,
            "code": "OK" if validation["passed"] and not missing else ("EXPORT_INPUT_INCOMPLETE" if missing else "EXPORT_VALIDATION_FAILED"),
            "message": "All result bundles exported" if not missing else f"Available bundles exported; missing: {missing}",
            "warnings": [f"Missing computational bundle: {name}" for name in missing],
            "failed_checks": validation["failed_checks"],
        },
    }


def validate_export_collection(
    root: Path,
    directories: dict[str, Path],
    exports: list[dict[str, Any]],
    master_path: Path,
    missing: list[str],
) -> dict[str, Any]:
    """Read every generated JSON and NPZ and verify declared array metadata."""
    failures: list[str] = []
    checks = {
        "directories_exist": all(path.is_dir() for path in directories.values()),
        "master_json_readable": False,
        "bundle_files_readable": True,
        "metadata_consistent": True,
        "array_dimensions_consistent": True,
    }
    try:
        json.loads(master_path.read_text(encoding="utf-8"))
        checks["master_json_readable"] = True
    except (OSError, json.JSONDecodeError):
        failures.append("master_json_readable")
    for item in exports:
        if item.get("status") != "exported":
            continue
        try:
            manifest = json.loads(Path(item["json_path"]).read_text(encoding="utf-8"))
            if manifest["bundle_name"] != item["bundle_name"]:
                checks["metadata_consistent"] = False
            with np.load(item["npz_path"], allow_pickle=False) as payload:
                for name, declaration in manifest["array_references"].items():
                    if name not in payload or list(payload[name].shape) != declaration["shape"] or str(payload[name].dtype) != declaration["dtype"]:
                        checks["array_dimensions_consistent"] = False
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            checks["bundle_files_readable"] = False
    for name, passed in checks.items():
        if not passed and name not in failures:
            failures.append(name)
    return {
        "passed": not failures,
        "checks": checks,
        "metrics": {
            "standard_directory_count": len(directories),
            "exported_bundle_count": sum(item.get("status") == "exported" for item in exports),
            "missing_bundle_count": len(missing),
        },
        "warnings": [f"Collection is incomplete: {missing}"] if missing else [],
        "failed_checks": failures,
        "summary": "Standard export validation passed" if not failures else f"Export validation failed: {failures}",
    }


def _write_standard_bundle(
    name: str, category: str, bundle: dict[str, Any], arrays: dict[str, np.ndarray],
    configuration_bundle: dict[str, Any], directory: Path,
) -> dict[str, Any]:
    started = perf_counter()
    json_path = directory / f"{name}_bundle.json"
    npz_path = directory / f"{name}_bundle.npz"
    normalized_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    references = {
        key: {"npz_file": npz_path.name, "npz_key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in normalized_arrays.items()
    }
    manifest = {
        "bundle_name": name,
        "category": category,
        "schema_name": bundle["metadata"]["schema_name"],
        "schema_version": bundle["metadata"]["schema_version"],
        "configuration": _configuration_snapshot(configuration_bundle),
        "metadata": bundle["metadata"],
        "validation": bundle["validation"],
        "summary_statistics": bundle["validation"].get("metrics", {}),
        "solver_information": _solver_information(bundle),
        "objective_values": _objective_values(bundle),
        "execution_time": {
            "computation_seconds": bundle["metadata"].get("execution_time_seconds"),
            "export_seconds": perf_counter() - started,
        },
        "array_references": references,
        "large_arrays_embedded_in_json": False,
    }
    write_json_npz(json_path, npz_path, manifest, normalized_arrays)
    return {
        "bundle_name": name,
        "category": category,
        "status": "exported",
        "json_path": str(json_path),
        "npz_path": str(npz_path),
        "array_count": len(arrays),
        "export_seconds": perf_counter() - started,
    }


def _geometry_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    primary = bundle["primary_result"]
    arrays = {f"terrain_{key}": value for key, value in primary["terrain_arrays"].items()}
    arrays.update({f"los_{key}": value for key, value in primary["los_masks"].items()})
    arrays.update({"sensor_position": primary["sensor_position"], "goal_position": primary["goal_position"], "tangent_point": primary["los_geometry"]["tangent_point"], "tangent_line_height": primary["los_geometry"]["tangent_line_height"]})
    return arrays


def _detection_arrays(geometry_bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    geometry = geometry_bundle["primary_result"]
    return {"sensor_position": geometry["sensor_position"], "goal_position": geometry["goal_position"], "tangent_point": geometry["los_geometry"]["tangent_point"]}


def _stage_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    primary = bundle["primary_result"]
    arrays = {"j4d": primary["j4d"], "powered_stage_cost_4d": primary["powered_stage_cost_4d"], "feasible_mask": primary["feasible_mask"]}
    arrays.update({f"component_{key}": value for key, value in primary["component_maps"].items()})
    arrays.update({f"mask_{key}": value for key, value in primary["validity_masks"].items()})
    arrays.update({f"grid_{key}": value for key, value in primary["grids"].items()})
    return arrays


def _projection_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    primary = bundle["primary_result"]
    return {key: primary[key] for key in ("projected_cost", "optimal_velocity", "optimal_gamma", "optimal_velocity_index", "optimal_gamma_index", "projection_mask")}


def _bellman_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    primary = bundle["primary_result"]
    arrays = _candidate_arrays(primary["candidates"], "speed_profile")
    arrays.update({
        f"cost_to_go_{ordering}": values
        for ordering, values in primary["cost_to_go_maps"].items()
    })
    arrays.update({
        f"pod_to_go_{ordering}": values
        for ordering, values in primary["pod_to_go_maps"].items()
    })
    arrays["cost_to_go"] = primary["cost_to_go_maps"][
        primary["cost_to_go_primary_ordering"]
    ]
    arrays["pod_to_go"] = primary["pod_to_go_maps"][
        primary["cost_to_go_primary_ordering"]
    ]
    return arrays


def _candidate_arrays(candidates: tuple[dict[str, Any], ...], speed_key: str) -> dict[str, np.ndarray]:
    trajectories, trajectory_offsets = _pack([item["trajectory"] for item in candidates], 2)
    speeds, speed_offsets = _pack([item[speed_key] for item in candidates], 1)
    gammas, gamma_offsets = _pack([item["gamma_profile"] for item in candidates], 1)
    powered_paths, powered_path_offsets = _pack([item["powered_path"] for item in candidates], 2)
    return {"switching_points": np.asarray([item["switching_point"] for item in candidates]).reshape(-1, 2), "trajectory_points": trajectories, "trajectory_offsets": trajectory_offsets, "velocity_profiles": speeds, "velocity_offsets": speed_offsets, "gamma_profiles": gammas, "gamma_offsets": gamma_offsets, "powered_path_points": powered_paths, "powered_path_offsets": powered_path_offsets, "mission_costs": np.asarray([item["mission_cost"] for item in candidates])}


def _bellman_response_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    primary = bundle["primary_result"]
    return {
        "switching_point": np.asarray(primary["switching_point"]),
        "trajectory": np.asarray(primary["trajectory"]),
        "powered_path": np.asarray(primary["powered_path"]),
        "speed_profile": np.asarray(primary["speed_profile"]),
        "gamma_profile": np.asarray(primary["gamma_profile"]),
        "mission_cost": np.asarray(primary["mission_cost"]),
        "mission_pod": np.asarray(primary["mission_pod"]),
        "mission_time": np.asarray(primary["mission_time"]),
        "goal_error": np.asarray(primary["constraint_residuals"]["goal_error"]),
    }


def _stackelberg_arrays(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    result = bundle["primary_result"]
    solution = result["final_stackelberg_solution"]
    visual = solution["visualization_payload"]
    summaries = result["outer_evaluation_summaries"]
    optimizer = result["outer_optimizer_result"]
    coarse = optimizer["coarse_sweep_results"]
    history = optimizer["brent_optimization_history"]
    return {
        "optimal_trajectory": solution["optimal_glide_trajectory"],
        "optimal_powered_path": solution["optimal_attacker_strategy"]["powered_path"],
        "optimal_switching_point": solution["optimal_switching_point"],
        "optimal_sensor_position": solution["optimal_sensor_position"],
        "optimal_velocity_profile": solution["optimal_attacker_strategy"]["speed_profile"],
        "optimal_gamma_profile": solution["optimal_attacker_strategy"]["gamma_profile"],
        "final_cost_to_go": visual["cost_to_go"],
        "final_pod_to_go": visual["pod_to_go"],
        "final_terrain_z": visual["terrain_z"],
        "final_terrain_height": visual["terrain_height"],
        "final_terrain_h_grid": visual["terrain_h_grid"],
        "final_terrain_mask": visual["terrain_mask"],
        "final_los_mask": visual["los_mask"],
        "final_occlusion_mask": visual["occlusion_mask"],
        "final_sensor_position": visual["sensor_position"],
        "final_goal_position": visual["goal_position"],
        "final_tangent_point": visual["tangent_point"],
        "final_tangent_line_height": visual["tangent_line_height"],
        "coverage_los_mask": solution["coverage_maps"]["los_mask"],
        "coverage_occlusion_mask": solution["coverage_maps"]["occlusion_mask"],
        "coverage_terrain_mask": solution["coverage_maps"]["terrain_mask"],
        "outer_z_sensor": np.asarray([item["z_sensor"] for item in summaries]),
        "outer_defender_objective": np.asarray([item["defender_objective"] for item in summaries]),
        "outer_attacker_objective": np.asarray([item["attacker_objective"] for item in summaries]),
        "outer_mission_pod": np.asarray([item["mission_pod"] for item in summaries]),
        "outer_defender_pod_normalized": np.asarray([item["defender_pod_normalized"] for item in summaries]),
        "outer_coverage_normalized": np.asarray([item["coverage_area_normalized"] for item in summaries]),
        "coarse_z_sensor": np.asarray([item["z_sensor"] for item in coarse]),
        "coarse_defender_objective": np.asarray([item["defender_objective"] for item in coarse]),
        "coarse_coverage_area": np.asarray([item["coverage_area"] for item in coarse]),
        "coarse_mission_pod": np.asarray([item["mission_pod"] for item in coarse]),
        "brent_z_sensor": np.asarray([item["z_sensor"] for item in history]),
        "brent_defender_objective": np.asarray([item["defender_objective"] for item in history]),
        "brent_coverage_area": np.asarray([item["coverage_area"] for item in history]),
        "brent_mission_pod": np.asarray([item["mission_pod"] for item in history]),
    }


def _configuration_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    primary = bundle["primary_result"]
    return {key: primary[key] for key in ("environment_config", "vehicle_config", "sensor_config", "cost_config", "bellman_config", "defender_config", "plot_config", "validation_config")}


def _solver_information(bundle: dict[str, Any]) -> Any:
    primary = bundle["primary_result"]
    return primary.get("solver_attempts", primary.get("outer_optimizer_result", None))


def _objective_values(bundle: dict[str, Any]) -> dict[str, Any]:
    primary = bundle["primary_result"]
    if "mission_objective" in primary:
        return {key: primary[key] for key in ("mission_objective", "mission_pod", "mission_time")}
    best = primary.get("best_found_attacker_response")
    if best is not None:
        return {key: best[key] for key in ("mission_objective", "mission_pod", "mission_time")}
    final = primary.get("final_stackelberg_solution")
    if final is not None:
        return {"attacker_objective": final["attacker_objective"], "defender_objective": final["defender_objective"], "mission_pod": final["mission_pod"]}
    return {}


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
        return {str(key): _json_native(item) for key, item in value.items() if not isinstance(item, (np.ndarray,))}
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and callable(value.name):
        return {"nonserializable_type": type(value).__name__, "name": value.name()}
    return value
