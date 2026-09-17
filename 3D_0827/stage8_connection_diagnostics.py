"""Generate Stage-8 single-candidate integration and replay artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bellman_graph import build_bellman_graph, solve_unit_cost_reachability
from bellman_state import BellmanStateGrid
from candidate_energy import evaluate_switching_candidate
from detection_hazard import GlideDetectionHazardModel
from los_explorer_gui import compute_los_case
from map_geometry import MapBounds
from mission_response import SingleCandidateMissionResponse, solve_single_candidate_response
from switching_candidates import generate_switching_candidates
from visualization import plot_single_candidate_mission


DEFAULT_STAGE8_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_8_single_switch_connection"
)
DEFAULT_GRAPH_BOUNDS = MapBounds(-8.0, 8.0, -4.0, 4.0)
FIXED_CANDIDATE_ID = 25


def build_canonical_stage8_response() -> tuple[Any, Any, SingleCandidateMissionResponse]:
    """Build the one documented candidate without Stage-9 enumeration."""
    case = compute_los_case("centered_cube", 5.0, 0.0)
    candidate = generate_switching_candidates(case.tangent_contour)[FIXED_CANDIDATE_ID]
    evaluation = evaluate_switching_candidate(
        candidate,
        case.tangent_contour,
        case.terrain_map,
        case.mission_points,
    )
    graph = build_bellman_graph(
        BellmanStateGrid(DEFAULT_GRAPH_BOUNDS),
        case.terrain_map,
        case.mission_points.goal,
    )
    unit = solve_unit_cost_reachability(graph)
    response = solve_single_candidate_response(
        evaluation,
        graph,
        unit,
        case.mission_points,
        GlideDetectionHazardModel(case.terrain_map, case.mission_points.sensor),
        quadrature_resolution=8,
    )
    return case, evaluation, response


def _write_connection_table(path: Path, response: SingleCandidateMissionResponse) -> None:
    columns = (
        "candidate_id",
        "target_state_id",
        "target_x_map",
        "target_y_map",
        "target_z_map",
        "target_heading_bin",
        "projection_error_m",
        "altitude_mismatch_m",
        "heading_mismatch_rad",
        "chord_heading_mismatch_rad",
        "duration_s",
        "node_admissible",
        "bellman_reachable",
        "terrain_feasible",
        "altitude_feasible",
        "turn_feasible",
        "chord_heading_feasible",
        "powered_feasible",
        "coarse_energy_feasible",
        "virtual_feasible",
        "rejection_reasons",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in response.connections.proposals:
            writer.writerow({
                "candidate_id": item.candidate_id,
                "target_state_id": item.target_state_id,
                "target_x_map": item.target_position_map[0],
                "target_y_map": item.target_position_map[1],
                "target_z_map": item.target_position_map[2],
                "target_heading_bin": item.target_state.heading_bin,
                "projection_error_m": item.projection_error_m,
                "altitude_mismatch_m": item.altitude_loss_m,
                "heading_mismatch_rad": item.heading_mismatch_rad,
                "chord_heading_mismatch_rad": item.chord_heading_mismatch_rad,
                "duration_s": item.duration_s,
                "node_admissible": item.node_admissible,
                "bellman_reachable": item.bellman_reachable,
                "terrain_feasible": item.terrain_feasible,
                "altitude_feasible": item.altitude_feasible,
                "turn_feasible": item.turn_feasible,
                "chord_heading_feasible": item.chord_heading_feasible,
                "powered_feasible": item.powered_feasible,
                "coarse_energy_feasible": item.coarse_energy_feasible,
                "virtual_feasible": item.feasible,
                "rejection_reasons": "; ".join(item.rejection_reasons),
            })


def _write_replay_table(path: Path, response: SingleCandidateMissionResponse) -> None:
    if response.replay is None:
        raise RuntimeError("canonical Stage-8 response is infeasible")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("point_index", "incoming_phase", "x_map", "y_map", "z_map"),
        )
        writer.writeheader()
        for index, position in enumerate(response.replay.positions_map):
            writer.writerow({
                "point_index": index,
                "incoming_phase": (
                    "start" if index == 0 else response.replay.segment_labels[index - 1]
                ),
                "x_map": position[0],
                "y_map": position[1],
                "z_map": position[2],
            })


def generate_stage8_artifacts(
    output_directory: Path = DEFAULT_STAGE8_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Write the auditable figure, proposal table, replay table, and summary."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    case, evaluation, response = build_canonical_stage8_response()
    if response.selected_option is None or response.replay is None:
        raise RuntimeError("fixed Stage-8 candidate did not produce a mission response")
    option = response.selected_option

    figure = plot_single_candidate_mission(
        case.terrain_map,
        case.mission_points,
        case.los_surface,
        case.visualization_rays,
        response,
    )
    figure.write_html(
        output_directory / "fixed_candidate_mission_response.html",
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
    )
    _write_connection_table(
        output_directory / "virtual_connection_proposals.csv", response,
    )
    _write_replay_table(
        output_directory / "continuous_mission_replay.csv", response,
    )

    connection = option.connection
    objective = option.objective
    energy = option.energy
    summary: dict[str, Any] = {
        "scope": {
            "candidate_count_considered": response.candidate_count_considered,
            "candidate_optimized": False,
            "fixed_candidate_id": response.candidate_id,
            "terrain_category": "centered_cube",
            "sensor_position_map": case.mission_points.sensor.as_array().tolist(),
        },
        "continuous_switching_state": {
            "position_map": evaluation.switching_state.position_map.tolist(),
            "heading_rad": evaluation.switching_state.heading_rad,
            "speed_mps": float(
                (evaluation.switching_state.velocity_mps**2).sum() ** 0.5
            ),
            "total_mechanical_energy_j": (
                evaluation.switching_state.total_mechanical_energy_j
            ),
            "powered_path_length_m": evaluation.switching_state.powered_path_length_m,
        },
        "virtual_connection": {
            "proposal_count": len(response.connections.proposals),
            "feasible_proposal_count": len(response.connections.feasible_proposals),
            "selected_state_id": connection.target_state_id,
            "selected_state": {
                "x_index": connection.target_state.x_index,
                "y_index": connection.target_state.y_index,
                "altitude_index": connection.target_state.altitude_index,
                "heading_bin": connection.target_state.heading_bin,
            },
            "selected_position_map": connection.target_position_map.tolist(),
            "projection_error_m": connection.projection_error_m,
            "altitude_mismatch_m": connection.altitude_loss_m,
            "heading_mismatch_rad": connection.heading_mismatch_rad,
            "duration_s": connection.duration_s,
            "hazard": option.virtual_hazard.hazard,
            "feasible": connection.feasible,
        },
        "bellman": {
            "active_state_count": response.active_state_count,
            "hazard_edge_count": response.hazard_precomputation.timing.edge_count,
            "hazard_precompute_s": response.hazard_precomputation.timing.precompute_s,
            "glide_edge_count": len(option.glide_edges),
            "feasible": bool(response.bellman_solution.goal_reachable[
                connection.target_state_id
            ]),
        },
        "energy": {
            "available_specific_height_m": energy.available_specific_height_m,
            "required_specific_height_m": energy.required_specific_height_m,
            "margin_m": energy.margin_m,
            "margin_j": energy.margin_j,
            "feasible": energy.feasible,
        },
        "phase_objective": {
            phase.name: {
                "duration_s": phase.duration_s,
                "hazard": phase.hazard,
                "weighted_cost": phase.weighted_cost,
                "hazard_assumption": phase.hazard_assumption,
            }
            for phase in (option.powered_phase, option.virtual_phase, option.glide_phase)
        },
        "mission": {
            "total_time_s": objective.mission_time_s,
            "total_hazard": objective.mission_hazard,
            "pod": objective.mission_pod,
            "objective": objective.objective_value,
            "maximum_join_gap_m": response.replay.maximum_join_gap_m,
            "terminal_goal_distance_m": response.replay.terminal_goal_distance_m,
            "within_goal_tolerance": response.replay.within_goal_tolerance,
        },
        "powered_detection_assumption": response.powered_detection_assumption,
    }
    (output_directory / "stage8_single_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("Stage 8 fixed-candidate response")
    print(f"candidate ID:       {response.candidate_id}")
    print(f"powered feasible:   {connection.powered_feasible}")
    print(f"virtual feasible:   {connection.feasible}")
    print(f"Bellman feasible:   {summary['bellman']['feasible']}")
    print(f"total time:         {objective.mission_time_s:.6f} s")
    print(f"total hazard:       {objective.mission_hazard:.9f}")
    print(f"PoD:                {objective.mission_pod:.9f}")
    print(f"objective:          {objective.objective_value:.9f}")
    return summary


if __name__ == "__main__":
    generate_stage8_artifacts()
    print(f"Generated Stage-8 diagnostics in {DEFAULT_STAGE8_FIGURE_DIRECTORY}")
