"""Generate the Stage-12 finite Stackelberg numerical/visual gate artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from defender_grid_config import CANONICAL_DEFENDER_X_MAP
from stage11_config import Stage11Config
from stage11_notebook_support import save_figure, write_json
from stackelberg_solver import (
    FiniteStackelbergRun,
    generate_defender_line_actions,
    run_finite_stackelberg,
)
from stackelberg_validation import (
    selected_attacker_run,
    validate_selected_stackelberg_trajectory,
)
from terrain_catalog import build_terrain
from trajectory_validation import (
    TrajectoryReplayAudit,
    snapshot_selected_trajectory,
    validate_trajectory_replay,
)
from visualization import (
    plot_defender_payoff_comparison,
    plot_finite_stackelberg_solution,
    plot_trajectory_validation,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE12_FIGURE_DIRECTORY = ROOT / "figure" / "stage_12_finite_stackelberg"
CANONICAL_DEFENDER_X = CANONICAL_DEFENDER_X_MAP


def _attacker_kwargs(config: Stage11Config) -> dict[str, object]:
    discretization = config.discretization
    return {
        "bellman_grid": discretization.build_bellman_grid(config.graph_bounds),
        "contour_sample_count": discretization.switching_contour_sample_count,
        "radial_scales": discretization.radial_scales,
        "quadrature_resolution": discretization.hazard_quadrature_resolution,
        "los_probe_grid_size": discretization.los_probe_grid_size,
        "los_boundary_refinement_steps": discretization.los_boundary_refinement_steps,
        "los_display_extension_factor": discretization.los_display_extension_factor,
        "visualization_ray_count": discretization.visualization_ray_count,
        "parameters": config.glider,
        "physical_scale": config.physical_scale,
        "detection_parameters": config.detection,
        "objective_parameters": config.attacker_objective,
    }


def _validate_selected(
    stackelberg_run: FiniteStackelbergRun,
    config: Stage11Config,
) -> TrajectoryReplayAudit:
    return validate_selected_stackelberg_trajectory(stackelberg_run, config)


def _evaluation_row(evaluation: Any) -> dict[str, Any]:
    sensor = evaluation.candidate.action.sensor_position_map
    follower = evaluation.sse_follower_result
    outcome = evaluation.outcome
    return {
        "defender_action_id": evaluation.candidate.action_id,
        "sensor_x_map": float(sensor[0]),
        "sensor_y_map": float(sensor[1]),
        "sensor_z_map": float(sensor[2]),
        "feasible": evaluation.feasible,
        "attacker_candidate_id": None if follower is None else follower.candidate_id,
        "attacker_cooptimal_candidate_ids": list(
            evaluation.attacker_run.cooptimal_candidate_ids
        ),
        "attacker_objective": None if outcome is None else outcome.attacker_payoff,
        "mission_time_s": None if follower is None else follower.mission_time_s,
        "cumulative_hazard": None if follower is None else follower.cumulative_hazard,
        "detection_probability": None if outcome is None else outcome.defender_payoff,
        "attacker_br_runtime_s": evaluation.attacker_br_runtime_s,
        "infeasibility_reason": evaluation.infeasibility_reason,
    }


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["attacker_cooptimal_candidate_ids"] = "|".join(map(
                str, row["attacker_cooptimal_candidate_ids"],
            ))
            writer.writerow(serialized)


def run_stage12_diagnostics(
    output_directory: Path = DEFAULT_STAGE12_FIGURE_DIRECTORY,
    *,
    repeat_for_determinism: bool = True,
) -> tuple[FiniteStackelbergRun, TrajectoryReplayAudit, dict[str, Any]]:
    """Run the canonical 3-sensor game and persist every gate artifact."""
    output_directory.mkdir(parents=True, exist_ok=True)
    config = Stage11Config()
    terrain = build_terrain(config.terrain_category)
    candidates = generate_defender_line_actions(CANONICAL_DEFENDER_X)
    kwargs = _attacker_kwargs(config)
    first = run_finite_stackelberg(
        candidates,
        config.attacker_initial_condition,
        terrain=terrain,
        attacker_kwargs=kwargs,
    )
    selected_audit = _validate_selected(first, config)
    if not selected_audit.report.passed:
        raise RuntimeError("selected Stackelberg trajectory failed independent replay")

    repeated: FiniteStackelbergRun | None = None
    deterministic = True
    if repeat_for_determinism:
        repeated = run_finite_stackelberg(
            candidates,
            config.attacker_initial_condition,
            terrain=terrain,
            attacker_kwargs=_attacker_kwargs(config),
        )
        first_selected = first.selected_evaluation
        repeated_selected = repeated.selected_evaluation
        deterministic = bool(
            first_selected.candidate.action_id == repeated_selected.candidate.action_id
            and first_selected.sse_follower_result is not None
            and repeated_selected.sse_follower_result is not None
            and first_selected.sse_follower_result.candidate_id
            == repeated_selected.sse_follower_result.candidate_id
            and np.isclose(
                first.outcome.attacker_payoff,
                repeated.outcome.attacker_payoff,
                rtol=0.0, atol=1.0e-12,
            )
            and np.isclose(
                first.outcome.defender_payoff,
                repeated.outcome.defender_payoff,
                rtol=0.0, atol=1.0e-12,
            )
        )
        if not deterministic:
            raise RuntimeError("repeated finite Stackelberg run was not deterministic")

    rows = tuple(_evaluation_row(item) for item in first.evaluations)
    feasible_payoffs = tuple(
        float(row["detection_probability"])
        for row in rows if row["feasible"]
    )
    if not np.isclose(
        first.outcome.defender_payoff,
        max(feasible_payoffs),
        rtol=0.0,
        atol=first.leader_payoff_tolerance,
    ):
        raise RuntimeError("selected Defender action is not the literal maximum")

    solution_figure = plot_finite_stackelberg_solution(first)
    comparison_figure = plot_defender_payoff_comparison(first)
    selected_run = selected_attacker_run(first)
    snapshot = snapshot_selected_trajectory(
        selected_run,
        hazard_quadrature_resolution=(
            config.discretization.hazard_quadrature_resolution
        ),
    )
    validation_figure = plot_trajectory_validation(
        selected_run.terrain,
        selected_run.mission_points,
        snapshot,
        selected_audit,
        goal_tolerance_map=(
            config.glider.goal_tolerance_m
            / config.physical_scale.meters_per_map_unit
        ),
    )
    save_figure(solution_figure, output_directory, "finite_stackelberg_solution.html")
    save_figure(comparison_figure, output_directory, "defender_payoff_comparison.html")
    save_figure(validation_figure, output_directory, "selected_trajectory_validation.html")
    _write_csv(output_directory / "defender_results.csv", rows)

    selected = first.selected_evaluation
    follower = selected.sse_follower_result
    if follower is None:
        raise RuntimeError("selected result unexpectedly disappeared")
    summary = {
        "stage": 12,
        "gate_passed": True,
        "equilibrium_scope": first.equilibrium_scope,
        "payoff_convention": {
            "attacker": "minimize attacker_hazard_time_v2",
            "defender": "maximize detection probability",
            "follower_tie": "maximize Defender PoD, then lowest candidate ID",
            "leader_tie": "stable Defender action order",
        },
        "configuration": config.as_dict(),
        "defender_actions": rows,
        "selected": {
            "defender_action_id": selected.candidate.action_id,
            "sensor_position_map": selected.candidate.action.sensor_position_map,
            "attacker_candidate_id": follower.candidate_id,
            "attacker_cooptimal_candidate_ids": (
                selected.attacker_run.cooptimal_candidate_ids
            ),
            "attacker_objective": first.outcome.attacker_payoff,
            "defender_payoff_pod": first.outcome.defender_payoff,
            "mission_time_s": follower.mission_time_s,
            "cumulative_hazard": follower.cumulative_hazard,
        },
        "literal_exhaustive_check": {
            "all_defender_actions_evaluated": len(first.evaluations) == len(candidates),
            "selected_payoff_equals_maximum": True,
        },
        "determinism": {
            "repeated": repeat_for_determinism,
            "passed": deterministic,
            "second_run_total_s": None if repeated is None else repeated.timing.total_s,
        },
        "validation": {
            "report": asdict(selected_audit.report),
            "recomputed_mission_time_s": selected_audit.recomputed_mission_time_s,
            "recomputed_cumulative_hazard": selected_audit.recomputed_cumulative_hazard,
            "recomputed_detection_probability": selected_audit.recomputed_detection_probability,
            "energy_margin_m": selected_audit.energy_margin_m,
            "goal_distance_m": selected_audit.goal_distance_m,
        },
        "timing": {
            "per_defender_action_s": first.timing.per_action_s,
            "total_stackelberg_s": first.timing.total_s,
        },
    }
    write_json(output_directory / "stackelberg_summary.json", summary)
    write_json(output_directory / "selected_trajectory_validation.json", summary["validation"])
    write_json(output_directory / "timing_report.json", summary["timing"])

    print("Stage 12 finite Stackelberg result")
    for row in rows:
        attacker_label = (
            "infeasible"
            if row["attacker_objective"] is None
            else f"J_A={row['attacker_objective']:.9f}, "
            f"PoD={row['detection_probability']:.9f}"
        )
        print(
            f"D{row['defender_action_id']}: sensor=({row['sensor_x_map']:.1f}, "
            f"{row['sensor_y_map']:.1f}, {row['sensor_z_map']:.1f}), "
            f"A={row['attacker_candidate_id']}, {attacker_label}, "
            f"runtime={row['attacker_br_runtime_s']:.3f}s"
        )
    print(
        f"Selected D{selected.candidate.action_id}, A{follower.candidate_id}, "
        f"PoD={first.outcome.defender_payoff:.9f}, "
        f"validation={'PASS' if selected_audit.report.passed else 'FAIL'}, "
        f"determinism={'PASS' if deterministic else 'FAIL'}"
    )
    return first, selected_audit, summary


if __name__ == "__main__":
    run_stage12_diagnostics()
