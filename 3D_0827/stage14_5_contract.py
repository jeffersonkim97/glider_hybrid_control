"""Immutable one-variable switching-candidate-density contract for Stage 14.5."""

from __future__ import annotations

from typing import Any, Iterable

from stage14_2c_contract import (
    MEMORY_LIMIT_BYTES,
    REPETITIONS,
    SAMPLING_INTERVAL_S,
    stage14_2c_cases,
)
from stage14_benchmark_runner import BenchmarkCase
from defender_grid_config import DEFENDER_ACTION_COUNT


STAGE14_5_SCHEMA_VERSION = "stage14.5-v1"
CONTOUR_SAMPLE_COUNTS = (6, 9, 12)
RADIAL_SAMPLE_COUNT = 8
CASE_TIMEOUT_S = 600.0


def _anchor_parameters() -> tuple[tuple[str, Any], ...]:
    return next(
        case.parameters
        for case in stage14_2c_cases(include_global_oracle=False)
        if case.name == "local_sse_canonical_anchor"
    )


def _parameters_at_density(contour_sample_count: int) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (
            name,
            int(contour_sample_count)
            if name == "switching_contour_sample_count"
            else value,
        )
        for name, value in _anchor_parameters()
    )


def stage14_5_cases() -> tuple[BenchmarkCase, ...]:
    """Three monotonically increasing densities for local SSE and its oracle."""
    return tuple(
        BenchmarkCase(
            experiment_type=f"stage14_5_{variant}_switching_density",
            name=f"{variant}_switch_contour_{contour_count}",
            worker_mode=variant,
            sweep_variable="switching_contour_sample_count",
            parameters=_parameters_at_density(contour_count),
            repetitions=REPETITIONS,
            timeout_s=CASE_TIMEOUT_S,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            sampling_interval_s=SAMPLING_INTERVAL_S,
        )
        for contour_count in CONTOUR_SAMPLE_COUNTS
        for variant in ("local_sse", "global_oracle")
    )


def configuration_audit(cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    cases = tuple(cases)
    anchor = dict(_anchor_parameters())
    fixed = {
        name: value for name, value in anchor.items()
        if name != "switching_contour_sample_count"
    }
    rows = []
    for case in cases:
        parameters = case.parameter_dict
        contour_count = int(parameters["switching_contour_sample_count"])
        radial_count = int(parameters["switching_radial_sample_count"])
        rows.append({
            "case_id": case.case_id,
            "name": case.name,
            "algorithm_variant": case.worker_mode,
            "requested_contour_sample_count": contour_count,
            "fixed_radial_sample_count": radial_count,
            "expected_raw_candidate_count": contour_count * radial_count,
            "declared_sweep_variable": case.sweep_variable,
            "fixed_parameter_differences": sorted(
                name for name, value in fixed.items()
                if parameters[name] != value
            ),
            "parameters": parameters,
            "repetitions": case.repetitions,
            "timeout_s": case.timeout_s,
            "memory_limit_bytes": case.memory_limit_bytes,
        })
    by_variant = {
        variant: sorted(
            int(case.parameter_dict["switching_contour_sample_count"])
            for case in cases if case.worker_mode == variant
        )
        for variant in ("local_sse", "global_oracle")
    }
    checks = {
        "only_switching_candidate_density_varies": all(
            not row["fixed_parameter_differences"] for row in rows
        ),
        "declared_sweep_variable_is_contour_density": all(
            case.sweep_variable == "switching_contour_sample_count"
            for case in cases
        ),
        "monotonically_increasing_densities_present": all(
            values == list(CONTOUR_SAMPLE_COUNTS)
            for values in by_variant.values()
        ),
        "radial_sampling_fixed": all(
            int(case.parameter_dict["switching_radial_sample_count"])
            == RADIAL_SAMPLE_COUNT
            for case in cases
        ),
        "spatial_heading_quadrature_defender_fixed": all(
            float(case.parameter_dict["spatial_resolution_m"]) == 25.0
            and float(case.parameter_dict["heading_spacing_deg"]) == 5.0
            and int(case.parameter_dict["defender_action_count"])
            == DEFENDER_ACTION_COUNT
            and int(case.parameter_dict["r_neighbor"]) == 1
            for case in cases
        ),
        "terrain_and_physical_mission_fixed": all(
            case.parameter_dict["terrain_category"] == "centered_cube"
            for case in cases
        ),
        "exact_non_rl_single_start": all(
            not case.parameter_dict["approximate_planner"]
            and not case.parameter_dict["reinforcement_learning"]
            and not case.parameter_dict["multi_start"]
            for case in cases
        ),
        "three_isolated_process_repetitions": all(
            case.repetitions == REPETITIONS for case in cases
        ),
    }
    return {
        "schema_version": STAGE14_5_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "contour_sample_counts": list(CONTOUR_SAMPLE_COUNTS),
        "radial_sample_count": RADIAL_SAMPLE_COUNT,
        "expected_raw_candidate_counts": [
            count * RADIAL_SAMPLE_COUNT for count in CONTOUR_SAMPLE_COUNTS
        ],
        "candidate_density_rule": (
            "vary contour sample count only; radial sample count remains fixed"
        ),
        "cases": rows,
    }


__all__ = [
    "CASE_TIMEOUT_S", "CONTOUR_SAMPLE_COUNTS", "RADIAL_SAMPLE_COUNT",
    "STAGE14_5_SCHEMA_VERSION", "configuration_audit", "stage14_5_cases",
]
