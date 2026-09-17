"""Fresh-process worker used only by the Stage-14.1 profiling diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from discretization_config import DiscretizationConfig
from stage11_config import Stage11Config
from stage11_notebook_support import write_json
from stage14_benchmark_contract import (
    CANONICAL_DEFENDER_X_MAP,
    canonical_attacker_kwargs,
    canonical_stage14_config,
    environment_manifest,
    frozen_reference_from_run,
    validate_frozen_reference,
)
from stage14_profiling import PROFILE_SCHEMA_VERSION, decompose_exact_sse_timing
from stackelberg_solver import generate_defender_line_actions, run_finite_stackelberg
from stackelberg_validation import validate_selected_stackelberg_trajectory
from terrain_catalog import build_terrain


ROOT = Path(__file__).resolve().parent


def run_canonical_profile() -> dict[str, Any]:
    config = canonical_stage14_config()
    terrain = build_terrain("centered_cube")
    run = run_finite_stackelberg(
        generate_defender_line_actions(CANONICAL_DEFENDER_X_MAP),
        config.attacker_initial_condition,
        terrain=terrain,
        attacker_kwargs=canonical_attacker_kwargs(config),
        reuse_reachability_graph=True,
    )
    validation_started = perf_counter()
    audit = validate_selected_stackelberg_trajectory(run, config)
    validation_s = perf_counter() - validation_started
    environment = environment_manifest(ROOT.parent)
    reference = frozen_reference_from_run(run, audit, environment)
    regression = validate_frozen_reference(reference)
    if not regression["passed"] or not audit.report.passed:
        raise RuntimeError("profiled canonical solution changed the exact SSE result")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "case_id": "canonical_centered_cube_25m_25m_5deg",
        "status": "completed",
        "terrain_category": "centered_cube",
        "timing": decompose_exact_sse_timing(
            run,
            validation_time_s=validation_s,
        ),
        "state_and_game_size": reference["state_and_game_size"],
        "solution_identity": reference["solution_identity"],
        "independent_replay": reference["independent_replay"],
        "frozen_solution_regression": regression,
        "solver_source_fingerprint": reference["solver_source_fingerprint"],
    }


def _infeasible_config() -> Stage11Config:
    """Small centered-cube fixture with two sampled switches and no feasible BR."""
    resolution = DiscretizationConfig(
        horizontal_spacing_map=4.0,
        altitude_spacing_map=1.0,
        heading_bin_count=8,
        motion_primitive_radius=1,
        switching_contour_sample_count=2,
        switching_radial_min=0.5,
        switching_radial_max=0.5,
        switching_radial_sample_count=1,
    )
    return replace(
        Stage11Config(),
        terrain_category="centered_cube",
        discretization=resolution,
    )


def run_infeasible_profile() -> dict[str, Any]:
    config = _infeasible_config()
    started = perf_counter()
    try:
        run_finite_stackelberg(
            generate_defender_line_actions((5.0,)),
            config.attacker_initial_condition,
            terrain=build_terrain("centered_cube"),
            attacker_kwargs=canonical_attacker_kwargs(config),
            reuse_reachability_graph=True,
        )
    except RuntimeError as error:
        elapsed = perf_counter() - started
        if str(error) != "no Defender action has a feasible Attacker response":
            raise
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "case_id": "explicit_infeasible_centered_cube_small",
            "status": "model_infeasible",
            "terrain_category": "centered_cube",
            "T_SSE_attempt_s": elapsed,
            "component_timing_available": False,
            "component_timing_unavailable_reason": (
                "The existing exact solver raises after confirming that all Defender "
                "actions have infeasible Attacker responses and does not return its "
                "partial FiniteStackelbergRun. Stage 14.1 preserves that behavior."
            ),
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "configuration": {
                "terrain_category": config.terrain_category,
                "discretization": config.discretization.as_dict(),
                "defender_x_map": [5.0],
                "vehicle_and_objectives_unchanged": True,
            },
        }
    raise RuntimeError("explicit infeasible fixture unexpectedly became feasible")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("canonical", "infeasible"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = (
        run_canonical_profile()
        if arguments.case == "canonical"
        else run_infeasible_profile()
    )
    write_json(arguments.output, payload)
    print(f"{payload['case_id']}: {payload['status']}")


if __name__ == "__main__":
    main()
