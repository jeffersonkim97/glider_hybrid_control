"""Generate Stage-9 exhaustive best-response and resolution artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from attacker_best_response import AttackerBestResponseRun, run_attacker_best_response
from game_types import AttackerInitialCondition, DefenderAction
from scenario import Point3D
from visualization import (
    plot_attacker_best_response,
    plot_candidate_objective_diagnostics,
)


DEFAULT_STAGE9_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_9_attacker_best_response"
)


def _canonical_inputs() -> tuple[DefenderAction, AttackerInitialCondition]:
    return (
        DefenderAction(np.array([5.0, 0.0, 0.0])),
        AttackerInitialCondition(
            start=Point3D(-8.0, 0.0, 0.0),
            goal=Point3D(8.0, 0.0, 0.0),
        ),
    )


def _write_candidate_table(path: Path, run: AttackerBestResponseRun) -> None:
    columns = (
        "candidate_id", "x_map", "y_map", "z_map", "powered_feasible",
        "energy_feasible", "virtual_feasible", "goal_reachable", "feasible",
        "mission_time_s", "cumulative_hazard", "detection_probability",
        "objective", "status_category", "infeasibility_reason",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in run.candidate_results:
            writer.writerow({
                "candidate_id": result.candidate_id,
                "x_map": result.position_map[0],
                "y_map": result.position_map[1],
                "z_map": result.position_map[2],
                "powered_feasible": result.powered_feasible,
                "energy_feasible": result.energy_feasible,
                "virtual_feasible": result.virtual_feasible,
                "goal_reachable": result.goal_reachable,
                "feasible": result.feasible,
                "mission_time_s": result.mission_time_s,
                "cumulative_hazard": result.cumulative_hazard,
                "detection_probability": result.detection_probability,
                "objective": result.objective,
                "status_category": result.status_category,
                "infeasibility_reason": result.infeasibility_reason,
            })


def _resolution_row(run: AttackerBestResponseRun, contour_count: int) -> dict[str, Any]:
    selected = run.selected_result
    return {
        "contour_sample_count": contour_count,
        "radial_scale_count": 8,
        "candidate_count": run.metrics.number_of_candidates,
        "best_candidate_id": None if selected is None else selected.candidate_id,
        "best_x_map": None if selected is None else selected.position_map[0],
        "best_y_map": None if selected is None else selected.position_map[1],
        "best_z_map": None if selected is None else selected.position_map[2],
        "best_objective": run.response.objective,
        "total_runtime_s": run.metrics.timing.total_attacker_br_s,
        "number_feasible": run.metrics.number_feasible,
    }


def _write_resolution_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_stage9_artifacts(
    output_directory: Path = DEFAULT_STAGE9_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Run 96- and 128-candidate exhaustive studies and write audit artifacts."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    action, scenario = _canonical_inputs()
    canonical = run_attacker_best_response(
        action, scenario, contour_sample_count=12, quadrature_resolution=8,
    )
    sensitivity = run_attacker_best_response(
        action, scenario, contour_sample_count=16, quadrature_resolution=8,
    )
    if canonical.selected_result is None:
        raise RuntimeError("canonical Stage-9 exhaustive response is infeasible")

    overview = plot_attacker_best_response(canonical)
    overview.write_html(
        output_directory / "exhaustive_attacker_best_response.html",
        include_plotlyjs="directory", full_html=True, auto_open=False,
    )
    objective = plot_candidate_objective_diagnostics(canonical)
    objective.write_html(
        output_directory / "candidate_objective_diagnostics.html",
        include_plotlyjs="directory", full_html=True, auto_open=False,
    )
    _write_candidate_table(
        output_directory / "candidate_results.csv", canonical,
    )
    resolution_rows = [
        _resolution_row(canonical, 12),
        _resolution_row(sensitivity, 16),
    ]
    _write_resolution_table(
        output_directory / "candidate_resolution_sensitivity.csv",
        resolution_rows,
    )

    selected = canonical.selected_result
    feasible_objectives = [
        float(result.objective)
        for result in canonical.candidate_results
        if result.feasible
    ]
    minimum_objective = min(feasible_objectives)
    exactness_error = abs(float(selected.objective) - minimum_objective)
    if exactness_error > 1.0e-12:
        raise RuntimeError("artifact selected candidate is not exhaustive minimum")
    categories = Counter(
        result.status_category for result in canonical.candidate_results
    )
    summary = {
        "architecture": {
            "candidate_enumeration": "literal exhaustive result storage",
            "bellman": "one shared exact solve after literal equivalence proof",
            "shared_bellman_solve_count": canonical.metrics.shared_bellman_solve_count,
            "tie_break": canonical.tie_break_convention,
        },
        "selected": {
            "candidate_id": selected.candidate_id,
            "position_map": selected.position_map.tolist(),
            "objective": selected.objective,
            "mission_time_s": selected.mission_time_s,
            "cumulative_hazard": selected.cumulative_hazard,
            "detection_probability": selected.detection_probability,
            "cooptimal_candidate_ids": list(canonical.cooptimal_candidate_ids),
            "virtual_target_state_id": (
                selected.selected_option.connection.target_state_id
            ),
        },
        "exactness": {
            "minimum_stored_feasible_objective": minimum_objective,
            "selected_minus_minimum_absolute": exactness_error,
            "assertion_tolerance": 1.0e-12,
        },
        "counts": {
            **asdict(canonical.metrics),
            "status_categories": dict(sorted(categories.items())),
        },
        "runtime": asdict(canonical.metrics.timing),
        "resolution_sensitivity": resolution_rows,
        "objective_definition": {
            "hazard_weight": 0.5,
            "time_weight": 0.5,
            "hazard_reference": 1.0,
            "time_reference_s": 5000.0 / 22.6,
        },
    }
    # Avoid duplicating the nested timing dictionary under counts.
    summary["counts"].pop("timing", None)
    (output_directory / "stage9_attacker_best_response_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    (output_directory / "candidate_resolution_sensitivity.json").write_text(
        json.dumps(resolution_rows, indent=2, sort_keys=True), encoding="utf-8",
    )
    print("Stage 9 exhaustive Attacker best response")
    print(f"candidates:     {canonical.metrics.number_of_candidates}")
    print(f"feasible:       {canonical.metrics.number_feasible}")
    print(f"selected ID:    {selected.candidate_id}")
    print(f"co-optimal IDs: {canonical.cooptimal_candidate_ids}")
    print(f"objective:      {selected.objective:.9f}")
    print(f"time:           {selected.mission_time_s:.6f} s")
    print(f"hazard:         {selected.cumulative_hazard:.9f}")
    print(f"PoD:            {selected.detection_probability:.9f}")
    print(f"runtime:        {canonical.metrics.timing.total_attacker_br_s:.3f} s")
    return summary


if __name__ == "__main__":
    generate_stage9_artifacts()
    print(f"Generated Stage-9 diagnostics in {DEFAULT_STAGE9_FIGURE_DIRECTORY}")
