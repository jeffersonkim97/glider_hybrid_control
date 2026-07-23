"""DEPRECATED / EXPERIMENTAL: Bellman candidate filtering and Top-K ranking.

Existed solely to prepare NLP multistart warm starts. Disconnected from the
authoritative solver now that `bellman.select_authoritative_bellman_response`
selects the Attacker best response directly from the complete, unfiltered
Bellman candidate set. Retained only alongside the deprecated `attacker_nlp`
comparison module.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def filter_bellman_candidates(
    bellman_candidate_bundle: dict[str, Any],
    configuration_bundle: dict[str, Any],
    validation_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Return objective-ranked Top-K representatives of unique path topologies.

    The function only compares, filters, ranks, and selects existing Phase 6
    candidates. It never changes a trajectory or control profile.
    """
    _require_successful_bundle(bellman_candidate_bundle, "bellman_candidate_bundle")
    _require_successful_bundle(configuration_bundle, "configuration_bundle")
    if not isinstance(validation_bundle, dict) or not validation_bundle.get("passed", False):
        raise ValueError("validation_bundle must be a passed Configuration validation result")
    source = bellman_candidate_bundle["primary_result"]
    if source.get("filtering_applied") or source.get("ranking_applied"):
        raise ValueError("Input must be the complete unfiltered Phase 6 candidate set")
    candidates = tuple(source["candidates"])
    bellman_config = configuration_bundle["primary_result"]["bellman_config"]
    thresholds = bellman_config["duplicate_threshold"]
    top_k = bellman_config["top_k"]
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("bellman_config.top_k must be a positive integer")

    ordered = sorted(candidates, key=lambda candidate: (candidate["mission_cost"], candidate["candidate_id"]))
    representatives: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    for candidate in ordered:
        matched = None
        matched_metrics = None
        for representative in representatives:
            metrics = candidate_similarity(candidate, representative, thresholds)
            if metrics["is_duplicate"]:
                matched = representative
                matched_metrics = metrics
                break
        if matched is None:
            representatives.append(candidate)
        else:
            duplicate_records.append(
                {
                    "removed_candidate_id": candidate["candidate_id"],
                    "representative_candidate_id": matched["candidate_id"],
                    "similarity_metrics": matched_metrics,
                }
            )

    ranked_unique = sorted(
        representatives,
        key=lambda candidate: (candidate["mission_cost"], candidate["candidate_id"]),
    )
    retained = tuple(
        _ranked_candidate(candidate, rank)
        for rank, candidate in enumerate(ranked_unique[:top_k], start=1)
    )
    validation = validate_filtered_candidate_set(
        retained,
        ranked_unique,
        duplicate_records,
        thresholds,
        top_k,
        configuration_bundle,
    )
    return {
        "primary_result": {
            "candidates": retained,
            "ranking": tuple(
                {
                    "rank": rank,
                    "candidate_id": candidate["candidate_id"],
                    "mission_cost": candidate["mission_cost"],
                    "selected_top_k": rank <= top_k,
                }
                for rank, candidate in enumerate(ranked_unique, start=1)
            ),
            "duplicate_records": tuple(duplicate_records),
            "source_candidate_count": len(candidates),
            "unique_candidate_count": len(ranked_unique),
            "selected_candidate_count": len(retained),
            "top_k": top_k,
            "filtering_applied": True,
            "ranking_applied": True,
            "trajectory_refinement_applied": False,
        },
        "validation": validation,
        "metadata": {
            "schema_name": "FilteredBellmanCandidateSet",
            "schema_version": "1.0.0",
            "producer_phase": 7,
            "producer_module": "p1b_4D.candidate_filtering",
            "source_schema": bellman_candidate_bundle["metadata"]["schema_name"],
            "attacker_objective_id": bellman_candidate_bundle["metadata"]["attacker_objective_id"],
            "ranking_key": "mission_cost",
            "ranking_direction": "ascending",
            "duplicate_threshold": thresholds,
            "only_attacker_nlp_warm_start_source": True,
            "trajectory_refinement_applied": False,
            "is_final_attacker_solution": False,
        },
        "status": {
            "success": validation["passed"],
            "code": "OK" if validation["passed"] else "FILTERED_CANDIDATES_INVALID",
            "message": validation["summary"],
            "warnings": validation["warnings"],
            "failed_checks": validation["failed_checks"],
        },
    }


def candidate_similarity(
    first: dict[str, Any], second: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the configured switching, shape, cost, and length criteria."""
    first_path = np.asarray(first["trajectory"], dtype=float)
    second_path = np.asarray(second["trajectory"], dtype=float)
    first_resampled = _resample_path(first_path, thresholds["resample_point_count"])
    second_resampled = _resample_path(second_path, thresholds["resample_point_count"])
    switching_distance = float(np.linalg.norm(
        np.asarray(first["switching_point"]) - np.asarray(second["switching_point"])
    ))
    trajectory_rms_distance = float(np.sqrt(np.mean(np.sum(
        (first_resampled - second_resampled) ** 2, axis=1
    ))))
    first_length = _path_length(first_path)
    second_length = _path_length(second_path)
    denominator = max(first_length, second_length, np.finfo(float).eps)
    relative_length_difference = abs(first_length - second_length) / denominator
    mission_cost_difference = abs(first["mission_cost"] - second["mission_cost"])
    criteria = {
        "switching_point": switching_distance <= thresholds["switching_distance"],
        "trajectory_shape": trajectory_rms_distance <= thresholds["trajectory_rms_distance"],
        "mission_cost": mission_cost_difference <= thresholds["mission_cost_difference"],
        "path_length": relative_length_difference <= thresholds["path_length_relative_difference"],
    }
    return {
        "is_duplicate": all(criteria.values()),
        "criteria": criteria,
        "switching_distance": switching_distance,
        "trajectory_rms_distance": trajectory_rms_distance,
        "mission_cost_difference": mission_cost_difference,
        "path_length_relative_difference": relative_length_difference,
    }


def validate_filtered_candidate_set(
    retained: tuple[dict[str, Any], ...],
    ranked_unique: list[dict[str, Any]],
    duplicate_records: list[dict[str, Any]],
    thresholds: dict[str, Any],
    top_k: int,
    configuration_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Validate unique topology, objective order, ranks, switches, and goals."""
    environment = configuration_bundle["primary_result"]["environment_config"]
    validation_config = configuration_bundle["primary_result"]["validation_config"]
    ids = [candidate["candidate_id"] for candidate in retained]
    costs = [candidate["mission_cost"] for candidate in retained]
    topology_unique = all(
        not candidate_similarity(retained[i], retained[j], thresholds)["is_duplicate"]
        for i in range(len(retained))
        for j in range(i + 1, len(retained))
    )
    goal = np.array([environment["z_goal"], environment["h_goal"]])
    checks = {
        "candidate_ids_unique": len(ids) == len(set(ids)),
        "unique_path_topology": topology_unique,
        "objective_ordering": costs == sorted(costs),
        "rank_consistency": [candidate["rank"] for candidate in retained] == list(range(1, len(retained) + 1)),
        "top_k_limit": len(retained) == min(top_k, len(ranked_unique)),
        "valid_switching_points": all(candidate["validation"]["checks"]["switching_consistency"] for candidate in retained),
        "goal_convergence": all(np.allclose(
            candidate["trajectory"][-1], goal, rtol=0.0,
            atol=max(validation_config["goal_tolerance_z"], validation_config["goal_tolerance_h"]),
        ) for candidate in retained),
        "source_candidates_valid": all(candidate["validation"]["passed"] for candidate in retained),
        "no_refinement": all(candidate["metadata"]["trajectory_refinement_applied"] is False for candidate in retained),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    warnings = []
    if len(ranked_unique) < top_k:
        warnings.append(f"Only {len(ranked_unique)} unique candidates were available for Top-{top_k}")
    return {
        "passed": not failed_checks,
        "checks": checks,
        "metrics": {
            "retained_candidate_count": len(retained),
            "unique_candidate_count": len(ranked_unique),
            "duplicate_candidate_count": len(duplicate_records),
            "minimum_selected_cost": float(min(costs)) if costs else np.inf,
            "maximum_selected_cost": float(max(costs)) if costs else np.inf,
        },
        "tolerances": thresholds,
        "warnings": warnings,
        "failed_checks": failed_checks,
        "summary": "Phase 7 Bellman filtering validation passed" if not failed_checks else f"Filtering failed checks: {failed_checks}",
    }


def _ranked_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    """Attach ranking metadata while preserving every source numeric array."""
    result = dict(candidate)
    result["rank"] = rank
    result["metadata"] = dict(candidate["metadata"])
    result["metadata"].update({
        "source_candidate_id": candidate["candidate_id"],
        "rank": rank,
        "selected_for_attacker_nlp": True,
        "trajectory_refinement_applied": False,
    })
    return result


def _resample_path(path: np.ndarray, count: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] == 0.0:
        return np.repeat(path[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(targets, cumulative, path[:, dimension]) for dimension in range(2)
    ])


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _require_successful_bundle(bundle: Any, name: str) -> None:
    if not isinstance(bundle, dict) or not bundle.get("status", {}).get("success", False):
        raise ValueError(f"{name} must be a successful bundle")
