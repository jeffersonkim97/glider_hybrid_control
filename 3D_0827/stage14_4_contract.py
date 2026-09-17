"""Immutable one-variable heading-resolution contract for Stage 14.4."""

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


STAGE14_4_SCHEMA_VERSION = "stage14.4-v2"
HEADING_SPACINGS_DEG = (
    45.0, 36.0, 30.0, 22.5, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0,
    7.5, 6.0, 5.0, 4.5, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0,
)
BASELINE_HEADING_SPACING_DEG = 5.0
FINE_HEADING_SPACING_DEG = 1.0
NEW_CASE_TIMEOUT_S = 900.0


def _profile(case: BenchmarkCase) -> str:
    return case.name.removeprefix(f"{case.worker_mode}_")


def reused_stage14_2c_cases() -> tuple[BenchmarkCase, ...]:
    """Return byte-identical 15/10/5-degree cases measured in Stage 14.2C."""
    selected_profiles = {"heading_15deg", "heading_10deg", "canonical_anchor"}
    return tuple(
        case for case in stage14_2c_cases()
        if _profile(case) in selected_profiles
    )


def _parameters_at_heading(heading_spacing_deg: float) -> tuple[tuple[str, Any], ...]:
    anchor = next(
        case for case in stage14_2c_cases(include_global_oracle=False)
        if _profile(case) == "canonical_anchor"
    )
    return tuple(
        (name, heading_spacing_deg if name == "heading_spacing_deg" else value)
        for name, value in anchor.parameters
    )


def new_stage14_4_cases() -> tuple[BenchmarkCase, ...]:
    reused_headings = {15.0, 10.0, 5.0}
    return tuple(
        BenchmarkCase(
            experiment_type=f"stage14_4_{variant}_heading",
            name=(
                f"{variant}_heading_"
                f"{str(heading_spacing).replace('.', 'p')}deg"
            ),
            worker_mode=variant,
            sweep_variable="heading_spacing_deg",
            parameters=_parameters_at_heading(heading_spacing),
            repetitions=REPETITIONS,
            timeout_s=NEW_CASE_TIMEOUT_S,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            sampling_interval_s=SAMPLING_INTERVAL_S,
        )
        for heading_spacing in HEADING_SPACINGS_DEG
        if heading_spacing not in reused_headings
        for variant in ("local_sse", "global_oracle")
    )


def stage14_4_cases() -> tuple[BenchmarkCase, ...]:
    return reused_stage14_2c_cases() + new_stage14_4_cases()


def configuration_audit(cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    cases = tuple(cases)
    anchor = next(
        case for case in cases
        if case.worker_mode == "local_sse"
        and float(case.parameter_dict["heading_spacing_deg"])
        == BASELINE_HEADING_SPACING_DEG
    )
    fixed = {
        name: value for name, value in anchor.parameters
        if name != "heading_spacing_deg"
    }
    reused_ids = {case.case_id for case in reused_stage14_2c_cases()}
    rows = []
    for case in cases:
        parameters = case.parameter_dict
        rows.append({
            "case_id": case.case_id,
            "name": case.name,
            "algorithm_variant": case.worker_mode,
            "heading_spacing_deg": parameters["heading_spacing_deg"],
            "declared_sweep_variable": case.sweep_variable,
            "fixed_parameter_differences": sorted(
                name for name, value in fixed.items() if parameters[name] != value
            ),
            "parameters": parameters,
            "repetitions": case.repetitions,
            "timeout_s": case.timeout_s,
            "memory_limit_bytes": case.memory_limit_bytes,
            "source": (
                "reused_stage14_2c_isolated_process_rows"
                if case.case_id in reused_ids
                else "new_stage14_4_isolated_process_rows"
            ),
        })
    by_variant = {
        variant: sorted(
            float(case.parameter_dict["heading_spacing_deg"])
            for case in cases if case.worker_mode == variant
        )
        for variant in ("local_sse", "global_oracle")
    }
    checks = {
        "only_heading_resolution_varies": all(
            not row["fixed_parameter_differences"] for row in rows
        ),
        "source_declaration_compatible_with_heading_sweep": all(
            case.sweep_variable == "heading_spacing_deg"
            or (
                case.sweep_variable == "baseline_anchor"
                and float(case.parameter_dict["heading_spacing_deg"])
                == BASELINE_HEADING_SPACING_DEG
            )
            for case in cases
        ),
        "required_heading_spacings_present": all(
            values == sorted(HEADING_SPACINGS_DEG) for values in by_variant.values()
        ),
        "spatial_resolution_fixed_25m": all(
            float(case.parameter_dict["spatial_resolution_m"]) == 25.0
            for case in cases
        ),
        "terrain_fixed_centered_cube": all(
            case.parameter_dict["terrain_category"] == "centered_cube"
            for case in cases
        ),
        "switching_quadrature_defender_controls_fixed": all(
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
        "three_isolated_process_repetitions": all(
            case.repetitions == REPETITIONS for case in cases
        ),
    }
    return {
        "schema_version": STAGE14_4_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "heading_spacings_deg": list(HEADING_SPACINGS_DEG),
        "baseline_heading_spacing_deg": BASELINE_HEADING_SPACING_DEG,
        "fine_heading_spacing_deg": FINE_HEADING_SPACING_DEG,
        "cases": rows,
    }


__all__ = [
    "BASELINE_HEADING_SPACING_DEG", "FINE_HEADING_SPACING_DEG",
    "HEADING_SPACINGS_DEG", "STAGE14_4_SCHEMA_VERSION",
    "configuration_audit", "new_stage14_4_cases",
    "reused_stage14_2c_cases", "stage14_4_cases",
]
