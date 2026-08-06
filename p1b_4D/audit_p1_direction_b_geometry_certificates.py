"""Audit stored Direction-B selected policies with the P1 exact certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from p1b_4D.configuration import build_configuration_bundle
from p1b_4D.direction_b_discretization import build_direction_b_configuration
from p1b_4D.experiment_b2_two_hill_nested_consistency import (
    build_two_hill_configuration,
)
from p1b_4D.experiment_b3_multiterrain_nested_consistency import (
    build_b3_physical_configuration,
)
from p1b_4D.geometry import build_geometry_bundle
from p1b_4D.phase_logging import close_phase_logger
from p1b_4D.segment_feasibility import certify_straight_segment_geometry


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_B2_PATH = REPO_ROOT / "results" / "direction_b" / "b2_two_hill_nested_consistency.json"
DEFAULT_B3_PATH = REPO_ROOT / "results" / "direction_b" / "b3_multiterrain_nested_consistency.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "direction_b" / "p1_exact_geometry_audit.json"


def audit_direction_b_geometry(
    project_root: Path,
    b2_path: Path = DEFAULT_B2_PATH,
    b3_path: Path = DEFAULT_B3_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    b2 = _load_json(b2_path)
    b3 = _load_json(b3_path)
    base = build_configuration_bundle(project_root)
    logger = base["primary_result"]["logging_utilities"]["logger"]
    records: list[dict[str, Any]] = []
    geometry_cache: dict[tuple[str, float], tuple[dict, dict]] = {}
    try:
        for source_name, payload in (("B2", b2), ("B3", b3)):
            for case in payload["cases"]:
                identity = {
                    "source": source_name,
                    "terrain_name": case.get("terrain_name", "two_hill"),
                    "sensor_name": case["sensor_name"],
                    "sensor_z": float(case["sensor_z"]),
                    "case_id": case["case_id"],
                    "original_status": case["status"],
                }
                if case["status"] != "feasible":
                    records.append({
                        **identity,
                        "audit_status": "not_applicable_no_selected_policy",
                    })
                    continue
                cache_key = (identity["terrain_name"], identity["sensor_z"])
                if cache_key not in geometry_cache:
                    physical = (
                        build_two_hill_configuration(base, identity["sensor_z"])
                        if identity["terrain_name"] == "two_hill"
                        else build_b3_physical_configuration(
                            base, identity["terrain_name"], identity["sensor_z"]
                        )
                    )
                    # Direction-B fixes the physical geometry reference at the
                    # terrain's L2 grid.  Geometry is therefore independent of
                    # the audited case's planning level/action/speed family.
                    geometry_configuration = build_direction_b_configuration(
                        physical,
                        identity["terrain_name"],
                        2,
                        action_family="enriched",
                        speed_family="V5",
                        edge_quadrature_count=9,
                    )
                    geometry_cache[cache_key] = (
                        geometry_configuration,
                        build_geometry_bundle(geometry_configuration),
                    )
                configuration, geometry_bundle = geometry_cache[cache_key]
                records.append(_audit_case(identity, case, configuration, geometry_bundle))
    finally:
        close_phase_logger(logger)

    applicable = [
        record for record in records
        if record["audit_status"] != "not_applicable_no_selected_policy"
    ]
    failed = [record for record in applicable if not record["passed"]]
    result = {
        "schema_name": "P1DirectionBExactGeometryAudit",
        "schema_version": "1.0.0",
        "certificate": (
            "piecewise_cubic_terrain_and_piecewise_linear_los_global_minimum"
        ),
        "source_results": {
            "b2": str(b2_path),
            "b3": str(b3_path),
        },
        "summary": {
            "stored_case_count": len(records),
            "selected_policy_count": len(applicable),
            "not_applicable_infeasible_case_count": len(records) - len(applicable),
            "certified_policy_count": len(applicable) - len(failed),
            "failed_policy_count": len(failed),
            "all_selected_policies_certified": not failed,
        },
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def _audit_case(identity, case, configuration_bundle, geometry_bundle):
    configs = configuration_bundle["primary_result"]
    geometry = geometry_bundle["primary_result"]
    environment = configs["environment_config"]
    validation = configs["validation_config"]
    planning = case["planning"]
    switching_point = np.asarray(planning["switching_point"], dtype=float)
    trajectory = np.asarray(planning["trajectory"], dtype=float)
    launch = np.array([environment["z_start"], environment["h_start"]], dtype=float)
    powered = certify_straight_segment_geometry(
        launch,
        switching_point,
        geometry["terrain_model"],
        geometry["los_geometry"],
        float(geometry["sensor_position"][0]),
        environment["airspace"],
        terrain_tolerance=float(validation["terrain_tolerance"]),
        los_requirement="occluded",
        los_tolerance=float(validation["los_tolerance"]),
    )
    glide = [
        certify_straight_segment_geometry(
            trajectory[index],
            trajectory[index + 1],
            geometry["terrain_model"],
            geometry["los_geometry"],
            float(geometry["sensor_position"][0]),
            environment["airspace"],
            terrain_tolerance=float(validation["terrain_tolerance"]),
            los_requirement="visible",
        )
        for index in range(trajectory.shape[0] - 1)
    ]
    passed = bool(powered["passed"] and all(edge["passed"] for edge in glide))
    minimum_glide_terrain = min(
        edge["minimum_terrain_margin"] for edge in glide
    )
    minimum_glide_los = min(edge["minimum_los_margin"] for edge in glide)
    failed_edges = [index for index, edge in enumerate(glide) if not edge["passed"]]
    return {
        **identity,
        "audit_status": "certified" if passed else "failed",
        "passed": passed,
        "powered_passed": bool(powered["passed"]),
        "glide_edge_count": len(glide),
        "failed_glide_edge_indices": failed_edges,
        "minimum_powered_terrain_margin": float(powered["minimum_terrain_margin"]),
        "minimum_powered_occlusion_margin": float(powered["minimum_los_margin"]),
        "minimum_glide_terrain_margin": float(minimum_glide_terrain),
        "minimum_glide_los_margin": float(minimum_glide_los),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--b2", type=Path, default=DEFAULT_B2_PATH)
    parser.add_argument("--b3", type=Path, default=DEFAULT_B3_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    result = audit_direction_b_geometry(
        arguments.project_root, arguments.b2, arguments.b3, arguments.output
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
