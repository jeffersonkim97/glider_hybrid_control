"""Generate Stage-10 independent replay reports and validation figures."""

from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from attacker_best_response import run_attacker_best_response
from detection_hazard import GlideDetectionHazardModel
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from game_types import AttackerInitialCondition, DefenderAction
from scenario import Point3D
from trajectory_validation import (
    DEFAULT_REPLAY_TOLERANCES,
    energy_margin_is_valid,
    goal_is_valid,
    snapshot_selected_trajectory,
    validate_trajectory_replay,
)
from visualization import plot_trajectory_validation


DEFAULT_STAGE10_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_10_independent_replay"
)


def _write_segment_table(path: Path, audit: Any) -> None:
    columns = (
        "segment_index", "phase", "start_x_map", "start_y_map", "start_z_map",
        "end_x_map", "end_y_map", "end_z_map", "duration_s", "hazard",
        "terrain_clear", "heading_change_rad", "allowed_heading_change_rad",
        "turn_valid", "altitude_loss_m", "required_altitude_loss_m",
        "altitude_valid",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for segment in audit.segments:
            writer.writerow({
                "segment_index": segment.segment_index,
                "phase": segment.phase,
                "start_x_map": segment.start_position_map[0],
                "start_y_map": segment.start_position_map[1],
                "start_z_map": segment.start_position_map[2],
                "end_x_map": segment.end_position_map[0],
                "end_y_map": segment.end_position_map[1],
                "end_z_map": segment.end_position_map[2],
                "duration_s": segment.duration_s,
                "hazard": segment.hazard,
                "terrain_clear": segment.terrain_clear,
                "heading_change_rad": segment.heading_change_rad,
                "allowed_heading_change_rad": segment.allowed_heading_change_rad,
                "turn_valid": segment.turn_valid,
                "altitude_loss_m": segment.altitude_loss_m,
                "required_altitude_loss_m": segment.required_altitude_loss_m,
                "altitude_valid": segment.altitude_valid,
            })


def _audit_summary(audit: Any) -> dict[str, Any]:
    return {
        "report": asdict(audit.report),
        "recomputed_mission_time_s": audit.recomputed_mission_time_s,
        "recomputed_cumulative_hazard": audit.recomputed_cumulative_hazard,
        "recomputed_detection_probability": audit.recomputed_detection_probability,
        "available_specific_height_m": audit.available_specific_height_m,
        "required_specific_height_m": audit.required_specific_height_m,
        "energy_margin_m": audit.energy_margin_m,
        "switching_heading_error_rad": audit.switching_heading_error_rad,
        "switching_speed_error_mps": audit.switching_speed_error_mps,
        "switching_energy_error_j": audit.switching_energy_error_j,
        "goal_distance_m": audit.goal_distance_m,
        "first_failure_segment_index": audit.first_failure_segment_index,
        "first_failure_position_map": (
            None if audit.first_failure_position_map is None
            else audit.first_failure_position_map.tolist()
        ),
    }


def generate_stage10_artifacts(
    output_directory: Path = DEFAULT_STAGE10_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Run valid and injected-failure replay cases and write all artifacts."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run = run_attacker_best_response(
        DefenderAction(np.array([5.0, 0.0, 0.0])),
        AttackerInitialCondition(
            Point3D(-8.0, 0.0, 0.0), Point3D(8.0, 0.0, 0.0),
        ),
    )
    snapshot = snapshot_selected_trajectory(run)
    hazard = GlideDetectionHazardModel(run.terrain, run.mission_points.sensor)

    def validate(candidate_snapshot: Any, *, parameters=DEFAULT_GLIDER) -> Any:
        return validate_trajectory_replay(
            candidate_snapshot,
            run.terrain,
            run.mission_points,
            run.tangent_contour,
            hazard,
            parameters=parameters,
        )

    valid = validate(snapshot)
    if not valid.report.passed:
        raise RuntimeError("canonical optimized trajectory failed independent replay")

    collision_positions = snapshot.discrete_positions_map.copy()
    collision_positions[0] = np.array([0.0, 0.0, 2.0])
    collision = validate(replace(
        snapshot, discrete_positions_map=collision_positions,
    ))
    turn_headings = list(snapshot.discrete_headings_rad)
    turn_headings[0] = np.pi
    turn = validate(replace(
        snapshot, discrete_headings_rad=tuple(turn_headings),
    ))
    time_failure = validate(replace(
        snapshot,
        stored_mission_time_s=snapshot.stored_mission_time_s + 1.0e-4,
    ))
    hazard_failure = validate(replace(
        snapshot,
        stored_cumulative_hazard=snapshot.stored_cumulative_hazard + 1.0e-5,
    ))
    margin = valid.energy_margin_m
    epsilon = DEFAULT_REPLAY_TOLERANCES.energy_height_m
    energy_within_parameters = replace(
        DEFAULT_GLIDER,
        switch_energy_loss_height_m=(
            DEFAULT_GLIDER.switch_energy_loss_height_m + margin + 0.5 * epsilon
        ),
    )
    energy_outside_parameters = replace(
        DEFAULT_GLIDER,
        switch_energy_loss_height_m=(
            DEFAULT_GLIDER.switch_energy_loss_height_m + margin + 2.0 * epsilon
        ),
    )
    energy_within = validate(snapshot, parameters=energy_within_parameters)
    energy_outside = validate(snapshot, parameters=energy_outside_parameters)

    expected_failures = (
        not collision.report.terrain_clear,
        not turn.report.turn_constraints_valid,
        not time_failure.report.time_consistent,
        not hazard_failure.report.hazard_consistent,
        energy_within.report.energy_valid,
        not energy_outside.report.energy_valid,
    )
    if not all(expected_failures):
        raise RuntimeError("one or more injected replay failures was not detected")

    tolerance_map = (
        DEFAULT_GLIDER.goal_tolerance_m
        / DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
    )
    plot_trajectory_validation(
        run.terrain, run.mission_points, snapshot, valid,
        goal_tolerance_map=tolerance_map,
    ).write_html(
        output_directory / "valid_selected_trajectory.html",
        include_plotlyjs="directory", full_html=True, auto_open=False,
    )
    plot_trajectory_validation(
        run.terrain, run.mission_points, snapshot, collision,
        goal_tolerance_map=tolerance_map,
    ).write_html(
        output_directory / "injected_collision_failure.html",
        include_plotlyjs="directory", full_html=True, auto_open=False,
    )
    _write_segment_table(
        output_directory / "independent_segment_replay.csv", valid,
    )
    (output_directory / "replay_tolerances.json").write_text(
        json.dumps({
            **asdict(DEFAULT_REPLAY_TOLERANCES),
            "configured_goal_tolerance_m": DEFAULT_GLIDER.goal_tolerance_m,
            "goal_tolerance_map": tolerance_map,
            "min_terrain_clearance_note": (
                "None: generic TerrainModel exposes collision queries but no signed distance"
            ),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    injections = {
        "R2_injected_terrain_collision": _audit_summary(collision),
        "R3_injected_turn_violation": _audit_summary(turn),
        "R4_injected_time_inconsistency": _audit_summary(time_failure),
        "R5_injected_hazard_inconsistency": _audit_summary(hazard_failure),
        "R6_goal_boundary": {
            "inside": goal_is_valid(DEFAULT_GLIDER.goal_tolerance_m - 1.0e-9),
            "boundary": goal_is_valid(DEFAULT_GLIDER.goal_tolerance_m),
            "tolerance_edge": goal_is_valid(DEFAULT_GLIDER.goal_tolerance_m + 1.0e-9),
            "outside": goal_is_valid(DEFAULT_GLIDER.goal_tolerance_m + 2.0e-9),
        },
        "R7_energy_boundary": {
            "within_margin_m": energy_within.energy_margin_m,
            "within_tolerance_valid": energy_margin_is_valid(energy_within.energy_margin_m),
            "outside_margin_m": energy_outside.energy_margin_m,
            "outside_tolerance_valid": energy_margin_is_valid(energy_outside.energy_margin_m),
        },
    }
    (output_directory / "injected_failure_reports.json").write_text(
        json.dumps(injections, indent=2, sort_keys=True), encoding="utf-8",
    )
    summary = {
        "candidate_id": snapshot.candidate_id,
        "optimizer": {
            "mission_time_s": snapshot.stored_mission_time_s,
            "cumulative_hazard": snapshot.stored_cumulative_hazard,
            "detection_probability": snapshot.stored_detection_probability,
        },
        "independent_replay": _audit_summary(valid),
        "independence": {
            "bellman_value_or_policy_used_during_validation": False,
            "mission_response_energy_certificate_reused": False,
            "edge_hazard_integrator_reused": False,
            "terrain_interface": "TerrainModel.segment_intersects_solid",
        },
    }
    (output_directory / "trajectory_validation_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    print("Stage 10 independent replay")
    print(f"candidate ID:  {snapshot.candidate_id}")
    print(f"passed:        {valid.report.passed}")
    print(f"time error:    {valid.report.time_error_s:.3e} s")
    print(f"hazard error:  {valid.report.hazard_error:.3e}")
    print(f"energy margin: {valid.energy_margin_m:.6f} m")
    print(f"goal distance: {valid.goal_distance_m:.6f} m")
    return summary


if __name__ == "__main__":
    generate_stage10_artifacts()
    print(f"Generated Stage-10 diagnostics in {DEFAULT_STAGE10_FIGURE_DIRECTORY}")
