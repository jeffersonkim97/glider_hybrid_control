"""Freeze the B4 production lattice from completed B2/B3 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import (
    DIRECTION_B_GRID_COUNTS,
    DIRECTION_B_PRODUCTION_ACTION_FAMILY,
    DIRECTION_B_PRODUCTION_CONFIGURATION_ID,
    DIRECTION_B_PRODUCTION_EVALUATOR_SAMPLE_COUNT,
    DIRECTION_B_PRODUCTION_LEVEL,
    DIRECTION_B_PRODUCTION_PLANNING_QUADRATURE_COUNT,
    DIRECTION_B_PRODUCTION_SPEED_FAMILY,
    build_direction_b_production_configuration,
    construct_direction_b_grids,
    direction_b_physical_envelope,
)
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    B2_SENSOR_CANDIDATES,
    build_two_hill_configuration,
)
from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    B3_TERRAIN_SPECIFICATIONS,
    build_b3_physical_configuration,
)
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.successor_grid_solver import regular_action_offsets


def freeze_b4_production_lattice(
    project_root: Path,
    b2_result_path: Path,
    b3_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate B2/B3 evidence and write the deterministic B4 manifest."""
    b2 = json.loads(b2_result_path.read_text(encoding="utf-8"))
    b3 = json.loads(b3_result_path.read_text(encoding="utf-8"))
    evidence = _select_global_settings(b2, b3)

    base = build_configuration_bundle(project_root)
    try:
        summaries = {}
        for terrain_name in DIRECTION_B_GRID_COUNTS:
            physical = _representative_physical_bundle(
                base, terrain_name
            )
            production = build_direction_b_production_configuration(
                physical, terrain_name
            )
            summaries[terrain_name] = _configuration_summary(production)
    finally:
        close_phase_logger(
            base["primary_result"]["logging_utilities"]["logger"]
        )

    settings_match_factory = (
        evidence["selected"]["level"] == DIRECTION_B_PRODUCTION_LEVEL
        and evidence["selected"]["action_family"]
        == DIRECTION_B_PRODUCTION_ACTION_FAMILY
        and evidence["selected"]["speed_family"]
        == DIRECTION_B_PRODUCTION_SPEED_FAMILY
        and evidence["selected"]["planning_quadrature_count"]
        == DIRECTION_B_PRODUCTION_PLANNING_QUADRATURE_COUNT
        and evidence["selected"]["evaluator_sample_count"]
        == DIRECTION_B_PRODUCTION_EVALUATOR_SAMPLE_COUNT
    )
    acceptance = {
        "b2_complete": b2.get("status") == "complete",
        "b3_complete": b3.get("status") == "complete",
        "b2_case_count_is_18": b2.get("case_count") == 18,
        "b3_case_count_is_36": b3.get("case_count") == 36,
        "b2_evaluator_gate_passed": bool(
            b2.get("common_evaluator_gate", {}).get("passed")
        ),
        "b3_evaluator_gate_passed": bool(
            b3.get("common_evaluator_gate", {}).get("passed")
        ),
        "selected_settings_match_production_factory": settings_match_factory,
        "all_terrain_factories_validated": len(summaries) == 3,
    }
    manifest = {
        "schema_name": "DirectionBProductionLatticeFreeze",
        "schema_version": "1.0.0",
        "status": (
            "complete" if all(acceptance.values())
            else "acceptance_gate_failed"
        ),
        "production_configuration_id": (
            DIRECTION_B_PRODUCTION_CONFIGURATION_ID
        ),
        "selected_settings": evidence["selected"],
        "intuitive_description": {
            "position_and_direction_resolution": (
                "finest tested nested position grid and enriched movement set"
            ),
            "speed_choice": "nine candidate glide speeds",
            "planning_edge_cost": (
                "nine samples along each physical movement segment"
            ),
            "continuous_replay": (
                "1025 samples along each selected physical segment"
            ),
        },
        "selection_evidence": evidence["terrain_evidence"],
        "terrain_configurations": summaries,
        "acceptance_gates": acceptance,
        "source_results": {
            "b2": str(b2_result_path.as_posix()),
            "b3": str(b3_result_path.as_posix()),
        },
        "scope": {
            "intended_use": "finite_c_lite_leader_follower_game",
            "finite_follower_lattice_frozen": True,
            "continuous_follower_optimum_claimed": False,
            "continuous_leader_optimum_claimed": False,
            "outer_sensor_candidates_must_be_reenumerated": True,
            "p2_sensor_positions_reused_as_c_lite_optima": False,
        },
    }
    manifest["manifest_sha256"] = _sha256({
        "production_configuration_id": manifest[
            "production_configuration_id"
        ],
        "selected_settings": manifest["selected_settings"],
        "terrain_configurations": manifest["terrain_configurations"],
        "scope": manifest["scope"],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _select_global_settings(b2, b3):
    terrain_evidence = {
        "two_hill": _terrain_evidence(
            b2["analysis"],
            speed_key="production_speed_family",
            quadrature_key="production_planning_quadrature_count",
        )
    }
    for terrain_name in ("single_hill", "goal_in_valley"):
        terrain_evidence[terrain_name] = _terrain_evidence(
            b3["analysis"][terrain_name],
            speed_key="production_speed_family_if_terrain_only",
            quadrature_key=(
                "production_planning_quadrature_count_if_terrain_only"
            ),
        )
    speed_family = (
        "V9" if any(
            evidence["terrain_only_speed_choice"] == "V9"
            for evidence in terrain_evidence.values()
        ) else "V5"
    )
    quadrature = (
        17 if any(
            evidence["terrain_only_quadrature_choice"] == 17
            for evidence in terrain_evidence.values()
        ) else 9
    )
    evaluator_sample_count = max(
        int(b2["global_common_evaluator_sample_count"]),
        int(b3["global_common_evaluator_sample_count"]),
    )
    return {
        "selected": {
            "level": 2,
            "action_family": "enriched",
            "speed_family": speed_family,
            "speed_count": int(speed_family[1:]),
            "planning_quadrature_count": quadrature,
            "evaluator_sample_count": evaluator_sample_count,
            "endpoint_snapping": False,
            "transition_model": "successor_grid_physical_edge",
        },
        "terrain_evidence": terrain_evidence,
    }


def _terrain_evidence(analysis, *, speed_key, quadrature_key):
    speed_values = [
        float(value)
        for by_level in analysis["speed_sensitivity"].values()
        for value in by_level.values()
        if value is not None
    ]
    quadrature_values = [
        float(value)
        for value in analysis["quadrature_sensitivity"].values()
        if value is not None
    ]
    return {
        "maximum_speed_sensitivity": max(speed_values),
        "maximum_quadrature_sensitivity": max(quadrature_values),
        "selection_tolerance": float(analysis["tau_b"]),
        "terrain_only_speed_choice": analysis[speed_key],
        "terrain_only_quadrature_choice": analysis[quadrature_key],
        "ranking_diagnostically_resolved": bool(
            analysis["ranking_diagnostically_resolved"]
        ),
    }


def _representative_physical_bundle(base, terrain_name):
    if terrain_name == "two_hill":
        return build_two_hill_configuration(
            base, B2_SENSOR_CANDIDATES["coverage"]
        )
    specification = B3_TERRAIN_SPECIFICATIONS[terrain_name]
    return build_b3_physical_configuration(
        base,
        terrain_name,
        specification["sensor_candidates"]["coverage"],
    )


def _configuration_summary(bundle):
    configs = bundle["primary_result"]
    protocol = configs["direction_b_protocol"]
    grid = configs["environment_config"]["grid"]
    options = configs["attacker_solver_config"]["successor_grid"]
    speeds = construct_direction_b_grids(bundle)["v"]
    displacement_count = len(regular_action_offsets(options))
    summary = {
        "configuration_id": protocol["production_configuration_id"],
        "grid": {
            "z_count": int(grid["z_count"]),
            "h_count": int(grid["h_count"]),
            "position_node_count": int(grid["z_count"] * grid["h_count"]),
            "z_spacing_m": float(grid["z_spacing"]),
            "h_spacing_m": float(grid["h_spacing"]),
        },
        "movement": {
            "displacement_direction_count": displacement_count,
            "speed_count": int(speeds.size),
            "regular_action_count": int(displacement_count * speeds.size),
            "speed_values_mps": speeds,
            "maximum_physical_envelope_m": (
                direction_b_physical_envelope(bundle)
            ),
        },
        "planning_quadrature_count": int(
            protocol["planning_quadrature_count"]
        ),
        "continuous_replay_sample_count": int(
            protocol["common_evaluator_sample_count"]
        ),
        "transition_model": configs["attacker_solver_config"][
            "transition_model"
        ],
        "endpoint_snapping": bool(protocol["endpoint_snapping"]),
        "production_frozen": bool(protocol["production_frozen"]),
    }
    summary["configuration_sha256"] = _sha256(summary)
    return summary


def _sha256(value):
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b2-result",
        type=Path,
        default=Path(
            "results/direction_b/b2_two_hill_nested_consistency.json"
        ),
    )
    parser.add_argument(
        "--b3-result",
        type=Path,
        default=Path(
            "results/direction_b/b3_multiterrain_nested_consistency.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/direction_b/b4_production_lattice_freeze.json"
        ),
    )
    arguments = parser.parse_args()
    result = freeze_b4_production_lattice(
        Path.cwd(), arguments.b2_result, arguments.b3_result, arguments.output
    )
    print(json.dumps({
        "status": result["status"],
        "production_configuration_id": result[
            "production_configuration_id"
        ],
        "selected_settings": result["selected_settings"],
        "manifest_sha256": result["manifest_sha256"],
        "output": str(arguments.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
