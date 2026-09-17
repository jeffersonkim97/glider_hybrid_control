"""Immutable one-factor-at-a-time benchmark contract for Stage 14.2C."""

from __future__ import annotations

from typing import Any

from defender_grid_config import (
    DEFENDER_ACTION_COUNT,
    DEFENDER_X_MAX_MAP,
    DEFENDER_X_MIN_MAP,
)
from stage14_benchmark_runner import BenchmarkCase


STAGE14_2C_SCHEMA_VERSION = "stage14.2C-v2"
REPETITIONS = 3
TIMEOUT_S = 300.0
MEMORY_LIMIT_BYTES = 4 * 1024**3
SAMPLING_INTERVAL_S = 0.02


def _parameters(
    *,
    spatial_resolution_m: float = 25.0,
    heading_spacing_deg: float = 5.0,
    switching_contour_sample_count: int = 12,
    defender_grid_policy: str = "canonical_six",
    defender_action_count: int = DEFENDER_ACTION_COUNT,
    baseline_anchor: int = 0,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("baseline_anchor", baseline_anchor),
        ("spatial_resolution_m", spatial_resolution_m),
        ("heading_spacing_deg", heading_spacing_deg),
        ("switching_contour_sample_count", switching_contour_sample_count),
        ("switching_radial_sample_count", 8),
        ("defender_grid_policy", defender_grid_policy),
        ("defender_action_count", defender_action_count),
        ("defender_domain_x_min", DEFENDER_X_MIN_MAP),
        ("defender_domain_x_max", DEFENDER_X_MAX_MAP),
        ("initial_defender_action_id", 0),
        ("r_neighbor", 1),
        ("terrain_category", "centered_cube"),
        ("approximate_planner", False),
        ("reinforcement_learning", False),
        ("multi_start", False),
    )


def _case(
    name: str,
    sweep_group: str,
    sweep_variable: str,
    *,
    algorithm_variant: str,
    parameters: tuple[tuple[str, Any], ...],
) -> BenchmarkCase:
    return BenchmarkCase(
        experiment_type=f"stage14_2c_{algorithm_variant}_{sweep_group}",
        name=f"{algorithm_variant}_{name}",
        worker_mode=algorithm_variant,
        sweep_variable=sweep_variable,
        parameters=parameters,
        repetitions=REPETITIONS,
        timeout_s=TIMEOUT_S,
        memory_limit_bytes=MEMORY_LIMIT_BYTES,
        sampling_interval_s=SAMPLING_INTERVAL_S,
    )


def _physical_profiles() -> tuple[tuple[str, str, str, tuple[tuple[str, Any], ...]], ...]:
    """Canonical anchor plus two values for each physical/numerical sweep."""
    return (
        ("canonical_anchor", "baseline", "baseline_anchor", _parameters()),
        ("spatial_100m", "spatial", "spatial_resolution_m", _parameters(
            spatial_resolution_m=100.0,
        )),
        ("spatial_50m", "spatial", "spatial_resolution_m", _parameters(
            spatial_resolution_m=50.0,
        )),
        ("heading_15deg", "heading", "heading_spacing_deg", _parameters(
            heading_spacing_deg=15.0,
        )),
        ("heading_10deg", "heading", "heading_spacing_deg", _parameters(
            heading_spacing_deg=10.0,
        )),
        ("switch_contour_6", "switching", "switching_contour_sample_count", _parameters(
            switching_contour_sample_count=6,
        )),
        ("switch_contour_9", "switching", "switching_contour_sample_count", _parameters(
            switching_contour_sample_count=9,
        )),
    )


def _defender_profiles() -> tuple[tuple[str, str, str, tuple[tuple[str, Any], ...]], ...]:
    return tuple(
        (
            f"defender_count_{count}",
            "defender",
            "defender_action_count",
            _parameters(
                defender_grid_policy="uniform_fixed_domain",
                defender_action_count=count,
            ),
        )
        for count in (2, 3, 5)
    )


def stage14_2c_cases(
    *,
    include_global_oracle: bool = True,
) -> tuple[BenchmarkCase, ...]:
    profiles = _physical_profiles() + _defender_profiles()
    local = tuple(
        _case(
            name, group, variable,
            algorithm_variant="local_sse",
            parameters=parameters,
        )
        for name, group, variable, parameters in profiles
    )
    if not include_global_oracle:
        return local
    oracle = tuple(
        _case(
            name, group, variable,
            algorithm_variant="global_oracle",
            parameters=parameters,
        )
        for name, group, variable, parameters in profiles
    )
    return local + oracle


def configuration_audit(cases: tuple[BenchmarkCase, ...]) -> dict[str, Any]:
    """Prove fixed terrain/policies and one declared sweep variable per case."""
    rows = []
    for case in cases:
        parameters = case.parameter_dict
        rows.append({
            "case_id": case.case_id,
            "name": case.name,
            "algorithm_variant": case.worker_mode,
            "sweep_group": case.experiment_type.rsplit("_", 1)[-1],
            "declared_sweep_variable": case.sweep_variable,
            "declared_sweep_value": parameters[case.sweep_variable],
            "parameters": parameters,
            "repetitions": case.repetitions,
            "timeout_s": case.timeout_s,
            "memory_limit_bytes": case.memory_limit_bytes,
        })
    checks = {
        "all_centered_cube": all(
            row["parameters"]["terrain_category"] == "centered_cube" for row in rows
        ),
        "all_single_start": all(
            row["parameters"]["multi_start"] is False for row in rows
        ),
        "all_exact_no_rl": all(
            row["parameters"]["approximate_planner"] is False
            and row["parameters"]["reinforcement_learning"] is False
            for row in rows
        ),
        "all_r_neighbor_one": all(
            row["parameters"]["r_neighbor"] == 1 for row in rows
        ),
        "all_three_cold_repetitions": all(case.repetitions == 3 for case in cases),
        "declared_variable_present_once": all(
            sum(name == case.sweep_variable for name, _ in case.parameters) == 1
            for case in cases
        ),
    }
    return {
        "schema_version": STAGE14_2C_SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "cases": rows,
        "defender_sweep_policy": (
            "uniform samples over the fixed ordered x-domain [5.0, 10.0]; "
            "only N_D changes and r_neighbor remains one grid-index length"
        ),
    }


__all__ = [
    "MEMORY_LIMIT_BYTES", "REPETITIONS", "SAMPLING_INTERVAL_S",
    "STAGE14_2C_SCHEMA_VERSION", "TIMEOUT_S", "configuration_audit",
    "stage14_2c_cases",
]
