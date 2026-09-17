"""Immutable one-variable spatial-resolution contract for Stage 14.3."""

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


STAGE14_3_SCHEMA_VERSION = "stage14.3-v2"
# Exact isotropic spacings for the fixed 1600 x 800 x 500 m domain.
# Delta=100/n m makes every axis interval count integral for n=1,...,20.
SPATIAL_RESOLUTIONS_M = tuple(100.0 / n for n in range(1, 21))
BASELINE_RESOLUTION_M = 25.0
FINER_RESOLUTION_M = 20.0
SECOND_FINER_RESOLUTION_M = 12.5
NEW_CASE_TIMEOUT_S = 600.0


def _profile(case: BenchmarkCase) -> str:
    prefix = f"{case.worker_mode}_"
    return case.name.removeprefix(prefix)


def reused_stage14_2c_cases() -> tuple[BenchmarkCase, ...]:
    """Return the byte-identical 100/50/25 m cases measured in Stage 14.2C."""
    selected_profiles = {"spatial_100m", "spatial_50m", "canonical_anchor"}
    return tuple(
        case for case in stage14_2c_cases()
        if _profile(case) in selected_profiles
    )


def _parameters_at_resolution(resolution_m: float) -> tuple[tuple[str, Any], ...]:
    anchor = next(
        case for case in stage14_2c_cases(include_global_oracle=False)
        if _profile(case) == "canonical_anchor"
    )
    return tuple(
        (name, resolution_m if name == "spatial_resolution_m" else value)
        for name, value in anchor.parameters
    )


def new_stage14_3_cases() -> tuple[BenchmarkCase, ...]:
    """Declare every non-reused point, ordered from coarse toward fine."""
    reused_resolutions = {100.0, 50.0, 25.0}
    return tuple(
        BenchmarkCase(
            experiment_type=f"stage14_3_{variant}_spatial",
            name=f"{variant}_spatial_{str(resolution).replace('.', 'p')}m",
            worker_mode=variant,
            sweep_variable="spatial_resolution_m",
            parameters=_parameters_at_resolution(resolution),
            repetitions=REPETITIONS,
            timeout_s=NEW_CASE_TIMEOUT_S,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            sampling_interval_s=SAMPLING_INTERVAL_S,
        )
        for resolution in SPATIAL_RESOLUTIONS_M
        if resolution not in reused_resolutions
        for variant in ("local_sse", "global_oracle")
    )


def stage14_3_cases() -> tuple[BenchmarkCase, ...]:
    return reused_stage14_2c_cases() + new_stage14_3_cases()


def configuration_audit(cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    cases = tuple(cases)
    anchor = next(
        case for case in cases
        if case.worker_mode == "local_sse"
        and float(case.parameter_dict["spatial_resolution_m"]) == BASELINE_RESOLUTION_M
    )
    fixed = {
        name: value for name, value in anchor.parameters
        if name != "spatial_resolution_m"
    }
    rows = []
    for case in cases:
        parameters = case.parameter_dict
        differences = sorted(
            name for name, value in fixed.items() if parameters[name] != value
        )
        rows.append({
            "case_id": case.case_id,
            "name": case.name,
            "algorithm_variant": case.worker_mode,
            "resolution_m": parameters["spatial_resolution_m"],
            "declared_sweep_variable": case.sweep_variable,
            "stage14_3_sweep_variable": "spatial_resolution_m",
            "fixed_parameter_differences": differences,
            "parameters": parameters,
            "repetitions": case.repetitions,
            "timeout_s": case.timeout_s,
            "memory_limit_bytes": case.memory_limit_bytes,
            "source": (
                "reused_stage14_2c_cold_process_rows"
                if case in reused_stage14_2c_cases()
                else "new_stage14_3_cold_process_rows"
            ),
        })
    by_variant = {
        variant: sorted(
            float(case.parameter_dict["spatial_resolution_m"])
            for case in cases if case.worker_mode == variant
        )
        for variant in ("local_sse", "global_oracle")
    }
    checks = {
        "only_spatial_resolution_varies": all(
            not row["fixed_parameter_differences"] for row in rows
        ),
        "source_declaration_compatible_with_spatial_sweep": all(
            case.sweep_variable == "spatial_resolution_m"
            or (
                case.sweep_variable == "baseline_anchor"
                and float(case.parameter_dict["spatial_resolution_m"])
                == BASELINE_RESOLUTION_M
            )
            for case in cases
        ),
        "coarse_baseline_finer_present": all(
            values == sorted(SPATIAL_RESOLUTIONS_M) for values in by_variant.values()
        ),
        "terrain_fixed_centered_cube": all(
            case.parameter_dict["terrain_category"] == "centered_cube" for case in cases
        ),
        "heading_fixed_five_degrees": all(
            float(case.parameter_dict["heading_spacing_deg"]) == 5.0 for case in cases
        ),
        "switching_and_defender_controls_fixed": all(
            int(case.parameter_dict["switching_contour_sample_count"]) == 12
            and int(case.parameter_dict["switching_radial_sample_count"]) == 8
            and int(case.parameter_dict["defender_action_count"])
            == DEFENDER_ACTION_COUNT
            and int(case.parameter_dict["r_neighbor"]) == 1
            for case in cases
        ),
        "exact_non_rl_single_start": all(
            not case.parameter_dict["approximate_planner"]
            and not case.parameter_dict["reinforcement_learning"]
            and not case.parameter_dict["multi_start"]
            for case in cases
        ),
        "three_cold_repetitions": all(case.repetitions == 3 for case in cases),
    }
    return {
        "schema_version": STAGE14_3_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "resolutions_m": list(SPATIAL_RESOLUTIONS_M),
        "baseline_resolution_m": BASELINE_RESOLUTION_M,
        "finer_resolution_m": FINER_RESOLUTION_M,
        "cases": rows,
    }


__all__ = [
    "BASELINE_RESOLUTION_M", "FINER_RESOLUTION_M", "SECOND_FINER_RESOLUTION_M",
    "SPATIAL_RESOLUTIONS_M",
    "STAGE14_3_SCHEMA_VERSION", "configuration_audit", "new_stage14_3_cases",
    "reused_stage14_2c_cases", "stage14_3_cases",
]
