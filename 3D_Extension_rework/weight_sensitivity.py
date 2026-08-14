"""Defender-weight sensitivity using already-solved Attacker responses."""

from __future__ import annotations

from typing import Any

import numpy as np


def defender_score(
    pod_normalized: float,
    coverage_normalized: float,
    coverage_weight: float,
    weight_sum: float = 1.0,
) -> dict[str, float]:
    """Return complementary weights, weighted components, and their sum."""
    coverage_weight = float(coverage_weight)
    pod_weight = float(weight_sum) - coverage_weight
    if pod_weight < 0.0 or coverage_weight < 0.0:
        raise ValueError("Defender weights must be nonnegative")
    weighted_pod = pod_weight * float(pod_normalized)
    weighted_coverage = coverage_weight * float(coverage_normalized)
    return {
        "w_pod": pod_weight,
        "w_coverage": coverage_weight,
        "weighted_pod": weighted_pod,
        "weighted_coverage": weighted_coverage,
        "defender_objective": weighted_pod + weighted_coverage,
    }


def _candidate_key(candidate: dict[str, Any], coverage_weight: float) -> tuple:
    score = defender_score(
        candidate["defender_pod_normalized"],
        candidate["coverage_volume_normalized"],
        coverage_weight,
    )["defender_objective"]
    sensor = candidate["sensor_position_m"]
    return score, -float(sensor[0]), -float(sensor[1])


def select_cached_candidate(
    candidates: list[dict[str, Any]], coverage_weight: float,
) -> dict[str, Any]:
    """Select the best feasible cached response at one Defender weight."""
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if not feasible:
        raise ValueError("No feasible cached Defender candidates are available")
    selected = max(feasible, key=lambda item: _candidate_key(item, coverage_weight))
    score = defender_score(
        selected["defender_pod_normalized"],
        selected["coverage_volume_normalized"],
        coverage_weight,
    )
    return {
        "sensor_position_m": list(selected["sensor_position_m"]),
        **score,
        "pod_normalized": float(selected["defender_pod_normalized"]),
        "mission_pod": float(selected["mission_pod"]),
        "coverage_volume_normalized": float(selected["coverage_volume_normalized"]),
        "attacker_objective": float(selected["attacker_objective"]),
        "switching_point_m": list(selected["switching_point_m"]),
    }


def _find_reference(
    candidates: list[dict[str, Any]], reference_xy: tuple[float, float],
) -> dict[str, Any]:
    matches = [
        candidate for candidate in candidates
        if candidate["feasible"] and np.allclose(
            candidate["sensor_position_m"][:2], reference_xy,
            rtol=0.0, atol=1.0e-9,
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one feasible reference sensor, found {len(matches)}")
    return matches[0]


def _selection_segments(
    candidates: list[dict[str, Any]], bounds: tuple[float, float],
) -> list[dict[str, Any]]:
    """Build the exact upper envelope of affine cached-candidate scores."""
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    breakpoints = [float(bounds[0]), float(bounds[1])]
    for index, first in enumerate(feasible):
        p_first = float(first["defender_pod_normalized"])
        slope_first = float(first["coverage_volume_normalized"]) - p_first
        for second in feasible[index + 1:]:
            p_second = float(second["defender_pod_normalized"])
            slope_second = float(second["coverage_volume_normalized"]) - p_second
            denominator = slope_first - slope_second
            if abs(denominator) <= 1.0e-15:
                continue
            crossing = (p_second - p_first) / denominator
            if bounds[0] < crossing < bounds[1]:
                breakpoints.append(float(crossing))
    breakpoints = sorted(set(round(value, 14) for value in breakpoints))
    segments: list[dict[str, Any]] = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        if end - start <= 1.0e-13:
            continue
        selected = select_cached_candidate(feasible, 0.5 * (start + end))
        sensor = selected["sensor_position_m"]
        identity = (float(sensor[0]), float(sensor[1]))
        if segments and segments[-1]["identity"] == identity:
            segments[-1]["coverage_weight_end"] = end
        else:
            segments.append({
                "coverage_weight_start": start,
                "coverage_weight_end": end,
                "identity": identity,
                "sensor_position_m": sensor,
            })
    for segment in segments:
        segment.pop("identity")
    return segments


def analyze_weight_sensitivity(
    stackelberg_solution: dict[str, Any], options: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(stackelberg_solution["search"]["evaluations"])
    reference_xy = tuple(float(value) for value in options["reference_sensor_xy_m"])
    reference = _find_reference(candidates, reference_xy)
    weight_sum = float(options["weight_sum"])
    plot_bounds = tuple(float(value) for value in options["coverage_weight_plot_bounds"])
    dense_weights = np.linspace(
        plot_bounds[0], plot_bounds[1], int(options["dense_plot_count"]),
    )

    fixed_dense = []
    selected_dense = []
    for coverage_weight in dense_weights:
        fixed_dense.append(defender_score(
            reference["defender_pod_normalized"],
            reference["coverage_volume_normalized"],
            coverage_weight,
            weight_sum,
        ))
        selected_dense.append(select_cached_candidate(candidates, coverage_weight))

    reported = []
    for coverage_weight in options["coverage_weights"]:
        coverage_weight = float(coverage_weight)
        reported.append({
            "w_coverage": coverage_weight,
            "fixed_reference": defender_score(
                reference["defender_pod_normalized"],
                reference["coverage_volume_normalized"],
                coverage_weight,
                weight_sum,
            ),
            "cached_candidate_selection": select_cached_candidate(
                candidates, coverage_weight,
            ),
        })

    return {
        "reference_sensor": {
            "sensor_position_m": list(reference["sensor_position_m"]),
            "pod_normalized": float(reference["defender_pod_normalized"]),
            "mission_pod": float(reference["mission_pod"]),
            "coverage_volume_normalized": float(reference["coverage_volume_normalized"]),
            "attacker_objective": float(reference["attacker_objective"]),
        },
        "reported_weights": reported,
        "selection_segments": _selection_segments(candidates, plot_bounds),
        "dense": {
            "coverage_weights": dense_weights.tolist(),
            "fixed_reference": fixed_dense,
            "cached_candidate_selection": selected_dense,
        },
        "candidate_tradeoff": [
            {
                "sensor_position_m": list(item["sensor_position_m"]),
                "pod_normalized": float(item["defender_pod_normalized"]),
                "mission_pod": float(item["mission_pod"]),
                "coverage_volume_normalized": float(item["coverage_volume_normalized"]),
            }
            for item in candidates if item["feasible"]
        ],
        "metadata": {
            "weight_constraint": "w_pod + w_coverage = 1",
            "attacker_responses_recomputed": False,
            "selection_scope": "best among cached feasible Stage-8 sensor candidates",
            "candidate_count": len(candidates),
            "feasible_candidate_count": sum(bool(item["feasible"]) for item in candidates),
            "warning": (
                "Candidate rescoring is exact for cached responses but is not a "
                "fresh continuous outer search for every weight."
            ),
        },
    }
