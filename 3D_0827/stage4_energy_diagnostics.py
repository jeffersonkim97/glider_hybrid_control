"""Generate Stage-4 powered/energy candidate certification artifacts."""

from __future__ import annotations

from collections import Counter
import json
from math import asin, atan2
from pathlib import Path
from typing import Any

import numpy as np

from candidate_energy import evaluate_switching_candidates
from energy_model import DEFAULT_GLIDER, DEFAULT_PHYSICAL_SCALE
from los_explorer_gui import compute_los_case
from switching_candidates import generate_switching_candidates
from terrain_catalog import TERRAIN_LABELS
from visualization import (
    plot_candidate_energy_classification,
    plot_single_candidate_energy,
)


SELECTED_CANDIDATE_ID = 54
DEFAULT_STAGE4_FIGURE_DIRECTORY = (
    Path(__file__).resolve().parent / "figure" / "stage_4_powered_energy"
)


def _single_candidate_report(los_result: Any, evaluation: Any) -> dict[str, Any]:
    candidate = evaluation.candidate
    state = evaluation.switching_state
    start = los_result.mission_points.start.as_array()
    displacement_map = candidate.position_map - start
    distance_map = float(np.linalg.norm(displacement_map))
    direction = displacement_map / distance_map
    manual_velocity = DEFAULT_GLIDER.powered_speed_mps * direction
    manual_heading = atan2(float(direction[1]), float(direction[0]))
    manual_flight_path_angle = asin(float(direction[2]))
    altitude_m = candidate.position_map[2] * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
    manual_energy = DEFAULT_GLIDER.mass_kg * (
        DEFAULT_GLIDER.gravity_mps2 * altitude_m
        + 0.5 * DEFAULT_GLIDER.powered_speed_mps**2
    )
    return {
        "candidate_id": candidate.candidate_id,
        "position_map": candidate.position_map.tolist(),
        "contour_fraction": candidate.contour_fraction,
        "radial_scale": candidate.radial_scale,
        "surface_residual": candidate.surface_residual,
        "acoustically_neutralized": evaluation.acoustically_neutralized,
        "powered_feasible": evaluation.powered_feasible,
        "glide_reachable": evaluation.reachable,
        "infeasibility_reason": evaluation.infeasibility_reason,
        "powered_path_length_m": state.powered_path_length_m,
        "heading_rad": state.heading_rad,
        "flight_path_angle_rad": state.flight_path_angle_rad,
        "speed_mps": float(np.linalg.norm(state.velocity_mps)),
        "total_mechanical_energy_j": state.total_mechanical_energy_j,
        "manual_errors": {
            "path_length_m": abs(
                state.powered_path_length_m
                - distance_map * DEFAULT_PHYSICAL_SCALE.meters_per_map_unit
            ),
            "velocity_mps": float(np.linalg.norm(state.velocity_mps - manual_velocity)),
            "heading_rad": abs(state.heading_rad - manual_heading),
            "flight_path_angle_rad": abs(
                state.flight_path_angle_rad - manual_flight_path_angle
            ),
            "total_mechanical_energy_j": abs(
                state.total_mechanical_energy_j - manual_energy
            ),
        },
    }


def generate_stage4_artifacts(
    output_directory: Path = DEFAULT_STAGE4_FIGURE_DIRECTORY,
) -> dict[str, Any]:
    """Write one single-candidate and four all-candidate interactive figures."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {
        "physical_constants": {
            "aircraft_name": DEFAULT_GLIDER.aircraft_name,
            "powered_speed_mps": DEFAULT_GLIDER.powered_speed_mps,
            "best_glide_speed_mps": DEFAULT_GLIDER.best_glide_speed_mps,
            "best_glide_ratio": DEFAULT_GLIDER.best_glide_ratio,
            "maximum_bank_deg": DEFAULT_GLIDER.maximum_bank_deg,
            "goal_tolerance_m": DEFAULT_GLIDER.goal_tolerance_m,
            "switch_energy_loss_height_m": (
                DEFAULT_GLIDER.switch_energy_loss_height_m
            ),
            "meters_per_map_unit": DEFAULT_PHYSICAL_SCALE.meters_per_map_unit,
        },
        "terrain_cases": {},
    }

    centered_los_result = None
    centered_evaluations = None
    for terrain_name in TERRAIN_LABELS:
        los_result = compute_los_case(terrain_name, 5.0, 0.0)
        candidates = generate_switching_candidates(los_result.tangent_contour)
        evaluations = evaluate_switching_candidates(
            candidates,
            los_result.tangent_contour,
            los_result.terrain_map,
            los_result.mission_points,
        )
        counts = Counter(
            "acoustic_invalid"
            if not result.acoustically_neutralized
            else "powered_infeasible"
            if not result.powered_feasible
            else "reachable"
            if result.reachable
            else "unreachable"
            for result in evaluations
        )
        feasible_margins = np.asarray([
            result.energy_margin_j
            for result in evaluations
            if result.powered_feasible
        ])
        reasons = Counter(
            result.infeasibility_reason or "none"
            for result in evaluations
        )
        summaries["terrain_cases"][terrain_name] = {
            "candidate_count": len(evaluations),
            "classification_counts": dict(sorted(counts.items())),
            "acoustically_neutralized_count": sum(
                result.acoustically_neutralized for result in evaluations
            ),
            "finite_energy_margin_range_j": [
                float(np.min(feasible_margins)),
                float(np.max(feasible_margins)),
            ],
            "infeasibility_reasons": dict(sorted(reasons.items())),
            "contains_nan": any(
                np.isnan(value)
                for result in evaluations
                for value in (
                    result.switching_state.total_mechanical_energy_j,
                    result.glide_result.available_energy_j,
                    result.glide_result.required_energy_j,
                    result.glide_result.energy_margin_j,
                    result.glide_result.equivalent_height_margin_m,
                )
            ),
        }

        figure = plot_candidate_energy_classification(
            los_result.terrain_map,
            los_result.mission_points,
            los_result.los_surface,
            los_result.visualization_rays,
            evaluations,
        )
        figure.update_layout(title={
            "text": (
                f"Stage 4B candidate certification: {TERRAIN_LABELS[terrain_name]}"
                f"<br><sup>reachable={counts['reachable']}, "
                f"unreachable={counts['unreachable']}, "
                f"powered infeasible={counts['powered_infeasible']}</sup>"
            ),
            "x": 0.5,
        })
        figure.write_html(
            output_directory / f"{terrain_name}_candidate_energy.html",
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
        )
        if terrain_name == "centered_cube":
            centered_los_result = los_result
            centered_evaluations = evaluations

    assert centered_los_result is not None
    assert centered_evaluations is not None
    selected = centered_evaluations[SELECTED_CANDIDATE_ID]
    summaries["selected_candidate"] = _single_candidate_report(
        centered_los_result,
        selected,
    )
    single_figure = plot_single_candidate_energy(
        centered_los_result.terrain_map,
        centered_los_result.mission_points,
        centered_los_result.los_surface,
        centered_los_result.visualization_rays,
        selected,
    )
    single_figure.write_html(
        output_directory / "centered_cube_candidate_54_powered_path.html",
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
    )

    (output_directory / "candidate_energy_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = summaries["selected_candidate"]
    print(
        "Stage 4A validation: "
        f"candidate={report['candidate_id']}, "
        f"path={report['powered_path_length_m']:.6f} m, "
        f"heading={np.degrees(report['heading_rad']):.6f} deg, "
        f"speed={report['speed_mps']:.6f} m/s, "
        f"energy={report['total_mechanical_energy_j']:.6f} J, "
        f"powered_feasible={report['powered_feasible']}, "
        f"glide_reachable={report['glide_reachable']}"
    )
    return summaries


if __name__ == "__main__":
    generated = generate_stage4_artifacts()
    print(
        "Generated Stage-4 diagnostics in "
        f"{DEFAULT_STAGE4_FIGURE_DIRECTORY}"
    )
