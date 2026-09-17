"""Immutable one-variable local-neighborhood-radius contract for Stage 14.6."""

from __future__ import annotations

from typing import Any, Iterable

from defender_grid_config import (
    CANONICAL_DEFENDER_X_MAP,
    DEFENDER_ACTION_COUNT,
    DEFENDER_X_MAX_MAP,
    DEFENDER_X_MIN_MAP,
)
from stage14_2c_contract import (
    MEMORY_LIMIT_BYTES,
    REPETITIONS,
    SAMPLING_INTERVAL_S,
    stage14_2c_cases,
)
from stage14_benchmark_runner import BenchmarkCase


STAGE14_6_SCHEMA_VERSION = "stage14.6-v2"
DEFENDER_X_MIN = DEFENDER_X_MIN_MAP
DEFENDER_X_MAX = DEFENDER_X_MAX_MAP
DEFENDER_X_MAP = CANONICAL_DEFENDER_X_MAP
INITIAL_DEFENDER_ACTION_ID = 0
R_NEIGHBOR_VALUES = (1, 2, 3, 5)
FULL_RADIUS = DEFENDER_ACTION_COUNT - 1
CASE_TIMEOUT_S = 600.0


def _anchor_parameters() -> dict[str, Any]:
    parameters = next(
        case.parameter_dict
        for case in stage14_2c_cases(include_global_oracle=False)
        if case.name == "local_sse_canonical_anchor"
    )
    return dict(parameters)


def _parameters(radius: int) -> tuple[tuple[str, Any], ...]:
    parameters = _anchor_parameters()
    parameters.update({
        "defender_grid_policy": "uniform_fixed_domain",
        "defender_action_count": DEFENDER_ACTION_COUNT,
        "defender_domain_x_min": DEFENDER_X_MIN,
        "defender_domain_x_max": DEFENDER_X_MAX,
        "initial_defender_action_id": INITIAL_DEFENDER_ACTION_ID,
        "r_neighbor": int(radius),
    })
    return tuple(parameters.items())


def stage14_6_cases(*, include_global_reference: bool = True) -> tuple[BenchmarkCase, ...]:
    local = tuple(
        BenchmarkCase(
            experiment_type="stage14_6_local_sse_neighbor_radius",
            name=f"local_sse_r_neighbor_{radius}",
            worker_mode="local_sse",
            sweep_variable="r_neighbor",
            parameters=_parameters(radius),
            repetitions=REPETITIONS,
            timeout_s=CASE_TIMEOUT_S,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            sampling_interval_s=SAMPLING_INTERVAL_S,
        )
        for radius in R_NEIGHBOR_VALUES
    )
    if not include_global_reference:
        return local
    reference = BenchmarkCase(
        experiment_type="stage14_6_exact_global_full_radius_reference",
        name="global_oracle_r_neighbor_8_reference",
        worker_mode="global_oracle",
        sweep_variable="r_neighbor",
        parameters=_parameters(FULL_RADIUS),
        repetitions=REPETITIONS,
        timeout_s=CASE_TIMEOUT_S,
        memory_limit_bytes=MEMORY_LIMIT_BYTES,
        sampling_interval_s=SAMPLING_INTERVAL_S,
    )
    return local + (reference,)


def configuration_audit(cases: Iterable[BenchmarkCase]) -> dict[str, Any]:
    cases = tuple(cases)
    local = tuple(case for case in cases if case.worker_mode == "local_sse")
    oracle = tuple(case for case in cases if case.worker_mode == "global_oracle")
    fixed_names = tuple(
        name for name in local[0].parameter_dict if name != "r_neighbor"
    ) if local else ()
    checks = {
        "four_declared_local_radii": tuple(
            case.parameter_dict["r_neighbor"] for case in local
        ) == R_NEIGHBOR_VALUES,
        "only_r_neighbor_varies_across_local_cases": bool(local) and all(
            all(
                case.parameter_dict[name] == local[0].parameter_dict[name]
                for name in fixed_names
            )
            for case in local
        ),
        "fixed_six_action_uniform_grid": all(
            case.parameter_dict["defender_grid_policy"] == "uniform_fixed_domain"
            and case.parameter_dict["defender_action_count"] == DEFENDER_ACTION_COUNT
            and case.parameter_dict["defender_domain_x_min"] == DEFENDER_X_MIN
            and case.parameter_dict["defender_domain_x_max"] == DEFENDER_X_MAX
            for case in cases
        ),
        "fixed_start_action_zero": all(
            case.parameter_dict["initial_defender_action_id"]
            == INITIAL_DEFENDER_ACTION_ID for case in cases
        ),
        "full_radius_equals_n_d_minus_one": FULL_RADIUS == DEFENDER_ACTION_COUNT - 1
        and R_NEIGHBOR_VALUES[-1] == FULL_RADIUS,
        "single_matching_global_reference": len(oracle) == 1 and bool(local)
        and oracle[0].parameter_dict == local[-1].parameter_dict,
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
        "schema_version": STAGE14_6_SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "r_neighbor_values": list(R_NEIGHBOR_VALUES),
        "full_radius": FULL_RADIUS,
        "defender_action_count": DEFENDER_ACTION_COUNT,
        "defender_action_grid": [
            {"action_id": action_id, "sensor_position_map": [x, 0.0, 0.0]}
            for action_id, x in enumerate(DEFENDER_X_MAP)
        ],
        "initial_defender_action_id": INITIAL_DEFENDER_ACTION_ID,
        "neighborhood_rule": "0 < abs(j - i) <= r_neighbor",
        "cases": [case.as_configuration() | {"case_id": case.case_id} for case in cases],
    }


__all__ = [
    "CASE_TIMEOUT_S", "DEFENDER_ACTION_COUNT", "DEFENDER_X_MAP", "FULL_RADIUS",
    "INITIAL_DEFENDER_ACTION_ID", "R_NEIGHBOR_VALUES", "STAGE14_6_SCHEMA_VERSION",
    "configuration_audit", "stage14_6_cases",
]
