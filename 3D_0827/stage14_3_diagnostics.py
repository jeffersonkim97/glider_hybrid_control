"""Stage 14.3 controlled spatial-resolution sweep and exit gate."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from attacker_best_response import DEFAULT_GRAPH_BOUNDS
from stage11_notebook_support import save_figure_png, write_json
from stage14_2c_worker import _build_case
from stage14_3_contract import (
    BASELINE_RESOLUTION_M,
    FINER_RESOLUTION_M,
    SPATIAL_RESOLUTIONS_M,
    STAGE14_3_SCHEMA_VERSION,
    configuration_audit,
    new_stage14_3_cases,
    reused_stage14_2c_cases,
    stage14_3_cases,
)
from stage14_benchmark_contract import (
    DEFAULT_STAGE14_TOLERANCES,
    environment_manifest,
)
from stage14_benchmark_runner import (
    BenchmarkCase,
    numeric_statistics,
    run_benchmark_cases,
)
from stage14_resolution_figures import build_resolution_figure_set


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_3_spatial_resolution"
STAGE14_2C_OUTPUT = ROOT / "figure" / "stage_14_2c_local_scaling"
FROZEN_REFERENCE = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
)
WORKER = ROOT / "stage14_3_worker.py"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _axis_record(values: np.ndarray, requested_spacing_map: float) -> dict[str, Any]:
    return {
        "count": int(len(values)),
        "minimum_map": float(values[0]),
        "maximum_map": float(values[-1]),
        "requested_spacing_map": float(requested_spacing_map),
        "realized_spacing_map": float(values[1] - values[0]),
        "coordinates_map": list(map(float, values)),
    }


def _realized_grid(case: BenchmarkCase) -> dict[str, Any]:
    configuration = case.as_configuration() | {
        "case_id": case.case_id,
        "repetition_id": case.repetition_id(0),
    }
    stage_config, _candidates, _initial = _build_case(configuration)
    discretization = stage_config.discretization
    grid = discretization.build_bellman_grid(DEFAULT_GRAPH_BOUNDS)
    return {
        "requested_spatial_resolution_m": float(
            case.parameter_dict["spatial_resolution_m"]
        ),
        "meters_per_map_unit": float(stage_config.physical_scale.meters_per_map_unit),
        "x": _axis_record(grid.x_coordinates, discretization.horizontal_spacing_map),
        "y": _axis_record(grid.y_coordinates, discretization.horizontal_spacing_map),
        "altitude": _axis_record(
            grid.altitude_coordinates, discretization.altitude_spacing_map
        ),
        "heading_bin_count": int(grid.heading_bin_count),
        "heading_spacing_deg": 360.0 / float(grid.heading_bin_count),
        "motion_primitive_radius": int(grid.motion_primitive_radius),
        "motion_primitive_step_cells": int(grid.motion_primitive_step_cells),
    }


def _profile(case: BenchmarkCase) -> str:
    resolution = str(case.parameter_dict["spatial_resolution_m"]).replace(".", "p")
    return f"spatial_{resolution}m"


def _metadata(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    reused = {case.case_id for case in reused_stage14_2c_cases()}
    return {
        case.case_id: {
            "profile": _profile(case),
            "algorithm_variant": case.worker_mode,
            "resolution_m": float(case.parameter_dict["spatial_resolution_m"]),
            "source": (
                "reused_stage14_2c_cold_process_rows"
                if case.case_id in reused else "new_stage14_3_cold_process_rows"
            ),
            "realized_grid": _realized_grid(case),
        }
        for case in cases
    }


def _completed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _runtime(row: dict[str, Any], key: str = "T_SSE_s") -> float | None:
    value = ((row.get("timing") or {}).get("totals") or {}).get(key)
    return None if value is None else float(value)


def _summaries(
    rows: list[dict[str, Any]], cases: tuple[BenchmarkCase, ...],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    result = []
    for case in cases:
        case_rows = sorted(grouped.get(case.case_id, []), key=lambda row: row["repetition_id"])
        complete = _completed(case_rows)
        components = sorted({
            key for row in complete
            for key in ((row.get("timing") or {}).get("totals") or {})
        })
        identities = [
            (
                row["selected_defender_action_id"],
                row["selected_attacker_candidate_id"],
                row["J_A"], row["J_D"],
                (row.get("trajectory_identity") or {}).get("sha256"),
            )
            for row in complete
        ]
        result.append({
            "case_id": case.case_id,
            **metadata[case.case_id],
            "configuration": case.as_configuration(),
            "attempted_repetitions": len(case_rows),
            "successful_repetitions": len(complete),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "T_SSE_statistics_s": numeric_statistics([
                value for row in complete if (value := _runtime(row)) is not None
            ]),
            "runtime_component_statistics_s": {
                key: numeric_statistics([
                    value for row in complete if (value := _runtime(row, key)) is not None
                ])
                for key in components
            },
            "peak_rss_statistics_bytes": numeric_statistics([
                row["memory"]["peak_rss_bytes"] for row in complete
            ]),
            "J_A_statistics": numeric_statistics([row["J_A"] for row in complete]),
            "J_D_statistics": numeric_statistics([row["J_D"] for row in complete]),
            "state_and_game_size": None if not complete else complete[0]["state_and_game_size"],
            "solution_identities": identities,
            "deterministic_solution_identity": bool(identities and len(set(identities)) == 1),
            "all_completed_replays_pass": bool(
                complete and all(row["independent_replay_passed"] is True for row in complete)
            ),
            "all_completed_exactness_checks_pass": bool(
                complete and all(all((row.get("exactness") or {}).values()) for row in complete)
            ),
        })
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_raw_csv(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]],
) -> None:
    fields = (
        "profile", "algorithm_variant", "source", "resolution_m", "case_id",
        "repetition_id", "status", "failure_category", "failure_message",
        "T_SSE_s", "T_Bellman_s", "worker_wall_s", "peak_rss_bytes",
        "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
        "N_S_goal_reachable", "N_S_active", "N_E", "B", "Q", "N_C", "N_D",
        "J_A", "J_D", "selected_defender_action_id",
        "selected_attacker_candidate_id", "independent_replay_passed",
        "realized_grid_json", "configuration_json", "timing_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = metadata[row["case_id"]]
            counts = row.get("state_and_game_size") or {}
            writer.writerow({
                **{name: meta[name] for name in (
                    "profile", "algorithm_variant", "source", "resolution_m"
                )},
                "case_id": row["case_id"], "repetition_id": row["repetition_id"],
                "status": row["status"], "failure_category": row["failure_category"],
                "failure_message": row["failure_message"],
                "T_SSE_s": _runtime(row), "T_Bellman_s": _runtime(row, "T_Bellman_s"),
                "worker_wall_s": row["worker_wall_s"],
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
                **{name: counts.get(name) for name in (
                    "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
                    "N_S_goal_reachable", "N_S_active", "N_E", "B", "Q", "N_C", "N_D",
                )},
                "J_A": row["J_A"], "J_D": row["J_D"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "independent_replay_passed": row["independent_replay_passed"],
                "realized_grid_json": json.dumps(meta["realized_grid"], sort_keys=True),
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
                "timing_json": json.dumps(row.get("timing"), sort_keys=True),
            })


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "profile", "algorithm_variant", "source", "resolution_m", "status_counts",
        "successful_repetitions", "T_SSE_median_s", "T_Bellman_median_s",
        "peak_rss_median_bytes", "N_S_cart", "N_S_active", "N_E", "J_A", "J_D",
        "deterministic_solution_identity", "all_completed_replays_pass",
        "all_completed_exactness_checks_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            counts = summary["state_and_game_size"] or {}
            components = summary["runtime_component_statistics_s"]
            writer.writerow({
                "profile": summary["profile"],
                "algorithm_variant": summary["algorithm_variant"],
                "source": summary["source"], "resolution_m": summary["resolution_m"],
                "status_counts": json.dumps(summary["status_counts"], sort_keys=True),
                "successful_repetitions": summary["successful_repetitions"],
                "T_SSE_median_s": None if summary["T_SSE_statistics_s"] is None else summary["T_SSE_statistics_s"]["median"],
                "T_Bellman_median_s": None if components.get("T_Bellman_s") is None else components["T_Bellman_s"]["median"],
                "peak_rss_median_bytes": None if summary["peak_rss_statistics_bytes"] is None else summary["peak_rss_statistics_bytes"]["median"],
                **{name: counts.get(name) for name in ("N_S_cart", "N_S_active", "N_E")},
                "J_A": None if summary["J_A_statistics"] is None else summary["J_A_statistics"]["median"],
                "J_D": None if summary["J_D_statistics"] is None else summary["J_D_statistics"]["median"],
                "deterministic_solution_identity": summary["deterministic_solution_identity"],
                "all_completed_replays_pass": summary["all_completed_replays_pass"],
                "all_completed_exactness_checks_pass": summary["all_completed_exactness_checks_pass"],
            })


def _figures(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> list[str]:
    return build_resolution_figure_set(
        rows,
        metadata,
        x_key="resolution_m",
        x_title="requested Delta x = Delta y = Delta h [m]",
        sweep_label="Spatial sweep",
        output=output,
        variants=("local_sse",),
        clean_performance_display=True,
        include_pod_figure=True,
        focused_runtime_display=True,
    )

def regenerate_stage14_3_figures_from_saved_data(
    output_directory: Path = OUTPUT,
) -> list[str]:
    """Re-render the Stage-14.3 report without invoking any benchmark worker."""
    rows = _read_json(output_directory / "raw_spatial_repetitions.json")
    metadata = _metadata(stage14_3_cases())
    figures = _figures(rows, metadata, output_directory)
    summary_path = output_directory / "stage14_3_summary.json"
    summary = _read_json(summary_path)
    summary["figure_files"] = figures
    summary["figure_methodology"] = {
        "controlled_x_axis": "requested spatial resolution Delta x = Delta y = Delta h [m]",
        "individual_measurement_label": "Individual isolated-process runs (n=3)",
        "process_isolation_boundary": (
            "fresh Python worker per repetition; operating-system and hardware caches "
            "are not asserted to be cold"
        ),
        "combined_state_edge_metric_used": False,
        "shared_resolution_figure_schema": True,
        "primary_algorithm": "local_sse",
        "reference_algorithm": "global_oracle",
        "displayed_algorithms": ["local_sse"],
        "oracle_measurements_retained": True,
        "performance_legend_algorithm_prefix": False,
        "individual_runtime_memory_points_displayed": False,
        "pod_figure": "04_1_detection_probability.png",
    }
    write_json(summary_path, summary)
    print(
        "Stage 14.3 figures regenerated from saved measurements only; "
        "no benchmark worker was invoked."
    )
    return figures


def run_stage14_3_diagnostics(
    output_directory: Path = OUTPUT, *, force: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = stage14_3_cases()
    audit = configuration_audit(cases)
    if not audit["passed"]:
        raise RuntimeError("Stage-14.3 configuration audit failed")
    metadata = _metadata(cases)
    environment = environment_manifest(ROOT.parent)
    new_directory = output_directory / "new_finer_measurements"
    prior_manifest_path = output_directory / "source_reuse_manifest.json"
    prior_manifest = (
        _read_json(prior_manifest_path) if prior_manifest_path.exists() else {}
    )
    resume_full_fresh = (
        not force
        and prior_manifest.get("mode") in {"full_fresh", "resume_full_fresh"}
        and (new_directory / "raw_repetitions.jsonl").exists()
    )
    full_fresh_measurement = force or resume_full_fresh
    if full_fresh_measurement:
        # A notebook recompute must be independent of Stage-14.2C caches:
        # measure all 20 resolutions, both algorithms, and all repetitions.
        # A subsequent non-forced call resumes that same cache instead of
        # silently switching back to the older Stage-14.2C reuse contract.
        measurement_cases = cases
        reused_rows: list[dict[str, Any]] = []
    else:
        source_rows = _read_json(STAGE14_2C_OUTPUT / "raw_repetitions.json")
        reused_ids = {case.case_id for case in reused_stage14_2c_cases()}
        reused_rows = [row for row in source_rows if row["case_id"] in reused_ids]
        if len(reused_rows) != 18 or any(
            row["status"] != "completed" for row in reused_rows
        ):
            raise RuntimeError("Stage-14.2C reusable spatial rows are incomplete")
        measurement_cases = new_stage14_3_cases()
    first = run_benchmark_cases(
        measurement_cases, output_directory=new_directory,
        environment_manifest_id=environment["environment_manifest_id"], force=force,
        worker_path=WORKER,
    )
    second = run_benchmark_cases(
        measurement_cases, output_directory=new_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        worker_path=WORKER,
    )
    new_case_ids = {case.case_id for case in measurement_cases}
    new_rows_by_id: dict[str, dict[str, Any]] = {}
    for row in second["rows"]:
        if row["case_id"] not in new_case_ids:
            continue
        new_rows_by_id.setdefault(row["repetition_id"], row)
    new_rows = list(new_rows_by_id.values())
    rows = sorted(reused_rows + new_rows, key=lambda row: (row["case_id"], row["repetition_index"]))
    for row in rows:
        row["algorithm_variant"] = row.get("algorithm_variant") or row["configuration"]["worker_mode"]
        row["stage14_3_source"] = (
            "fresh_stage14_3_isolated_process_rows"
            if full_fresh_measurement else metadata[row["case_id"]]["source"]
        )
        row["realized_grid"] = row.get("realized_grid") or metadata[row["case_id"]]["realized_grid"]
    summaries = _summaries(rows, cases, metadata)
    figures = _figures(rows, metadata, output_directory)
    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "spatial_sweep_configuration.json", audit)
    write_json(output_directory / "source_reuse_manifest.json", {
        "mode": (
            "full_fresh" if force
            else "resume_full_fresh" if resume_full_fresh
            else "reuse_stage14_2c"
        ),
        "source_stage": None if full_fresh_measurement else "14.2C",
        "source_artifact": (
            None
            if full_fresh_measurement
            else str(STAGE14_2C_OUTPUT / "raw_repetitions.json")
        ),
        "reused_repetition_ids": [row["repetition_id"] for row in reused_rows],
        "new_repetition_ids": [row["repetition_id"] for row in new_rows],
        "new_first_invocation_attempted": first["attempted_repetition_ids"],
        "new_second_invocation_skipped": second["skipped_repetition_ids"],
    })
    write_json(output_directory / "raw_spatial_repetitions.json", rows)
    _write_jsonl(output_directory / "raw_spatial_repetitions.jsonl", rows)
    _write_raw_csv(output_directory / "raw_spatial_repetitions.csv", rows, metadata)
    write_json(output_directory / "spatial_sweep_summaries.json", summaries)
    _write_summary_csv(output_directory / "spatial_sweep_summaries.csv", summaries)

    oracle_summaries = [row for row in summaries if row["algorithm_variant"] == "global_oracle"]
    oracle_by_resolution = {row["resolution_m"]: row for row in oracle_summaries}
    baseline_counts = oracle_by_resolution.get(25.0, {}).get("state_and_game_size")
    twenty_counts = oracle_by_resolution.get(20.0, {}).get("state_and_game_size")
    topology_counts_available = bool(baseline_counts and twenty_counts)
    topology_shift = {
        "baseline_resolution_m": 25.0,
        "finer_resolution_m": 20.0,
        "baseline_N_S_cart": (
            baseline_counts["N_S_cart"] if topology_counts_available else None
        ),
        "finer_N_S_cart": (
            twenty_counts["N_S_cart"] if topology_counts_available else None
        ),
        "baseline_N_S_active": (
            baseline_counts["N_S_active"] if topology_counts_available else None
        ),
        "finer_N_S_active": (
            twenty_counts["N_S_active"] if topology_counts_available else None
        ),
        "baseline_N_E": (
            baseline_counts["N_E"] if topology_counts_available else None
        ),
        "finer_N_E": (
            twenty_counts["N_E"] if topology_counts_available else None
        ),
        "active_state_ratio_finer_over_baseline": (
            twenty_counts["N_S_active"] / baseline_counts["N_S_active"]
            if topology_counts_available else None
        ),
        "edge_ratio_finer_over_baseline": (
            twenty_counts["N_E"] / baseline_counts["N_E"]
            if topology_counts_available else None
        ),
        "interpretation": (
            "the realized reachable/active graph changes non-monotonically with spacing; "
            "the measured curve is not evidence of a monotone asymptotic scaling law"
            if topology_counts_available else
            "the 25 m and/or 20 m global-oracle result did not complete; "
            "the topology ratio is unavailable and the failure remains explicit"
        ),
    }
    completed_rows = _completed(rows)
    finer_oracle = [
        row for row in rows
        if row["algorithm_variant"] == "global_oracle"
        and metadata[row["case_id"]]["resolution_m"] < BASELINE_RESOLUTION_M
    ]
    finer_completed_resolutions = sorted({
        metadata[row["case_id"]]["resolution_m"] for row in _completed(finer_oracle)
    })
    finer_limit_resolutions = sorted({
        metadata[row["case_id"]]["resolution_m"] for row in finer_oracle
        if row["status"] in {"timeout", "memory_limit"}
    })
    finer_complete = bool(finer_completed_resolutions)
    finer_limit = bool(finer_limit_resolutions)
    frozen = _read_json(FROZEN_REFERENCE)
    expected = frozen["solution_identity"]
    baseline_complete = [
        row for row in completed_rows
        if metadata[row["case_id"]]["resolution_m"] == BASELINE_RESOLUTION_M
    ]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    baseline_regression = bool(baseline_complete and all(
        row["selected_defender_action_id"] == expected["selected_defender_action_id"]
        and row["selected_attacker_candidate_id"] == expected["selected_attacker_candidate_id"]
        and abs(float(row["J_A"]) - float(expected["attacker_objective"])) <= tolerance.attacker_objective_abs
        and abs(float(row["J_D"]) - float(expected["defender_objective_pod"])) <= tolerance.defender_objective_abs
        for row in baseline_complete
    ))
    checks = {
        "only_spatial_resolution_varied": audit["checks"]["only_spatial_resolution_varies"],
        "coarse_baseline_and_finer_declared": audit["checks"]["coarse_baseline_finer_present"],
        "three_repetitions_per_configuration_recorded": len(rows) == sum(
            case.repetitions for case in cases
        ),
        "reused_stage14_2c_rows_complete": (
            len(reused_rows) == 0
            if full_fresh_measurement
            else len(reused_rows) == 18
        ),
        "new_runner_resume_deterministic": (
            not second["attempted_repetition_ids"]
            and len(second["skipped_repetition_ids"]) == sum(
                case.repetitions for case in measurement_cases
            )
        ),
        "finer_completed_or_explicit_computational_limit": finer_complete or finer_limit,
        "every_completed_replay_passes": bool(
            completed_rows and all(row["independent_replay_passed"] is True for row in completed_rows)
        ),
        "every_completed_exactness_check_passes": bool(
            completed_rows and all(all((row.get("exactness") or {}).values()) for row in completed_rows)
        ),
        "completed_solution_identities_deterministic": all(
            summary["deterministic_solution_identity"]
            for summary in summaries if summary["successful_repetitions"]
        ),
        "realized_axis_grids_recorded": all(
            row.get("realized_grid") and row["realized_grid"]["x"]["coordinates_map"]
            for row in rows
        ),
        "canonical_frozen_regression_passes": baseline_regression,
        "solver_source_fingerprint_unchanged": all(
            row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"]
            for row in completed_rows
        ),
        "global_solver_is_oracle_only": all(
            row["algorithm_variant"] != "global_oracle"
            or (row.get("oracle_metadata") or {}).get("role") == "tractable finite global oracle/reference only"
            for row in completed_rows
        ),
        "png_figures_exist": all((output_directory / name).is_file() for name in figures),
    }
    gate_passed = all(checks.values())
    validation = {
        "schema_version": STAGE14_3_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "completed_global_oracle_resolutions_m": sorted(
            summary["resolution_m"] for summary in oracle_summaries
            if summary["successful_repetitions"] == 3
        ),
        "finer_resolution_outcome": {
            "attempted_resolutions_m": sorted({
                metadata[row["case_id"]]["resolution_m"] for row in finer_oracle
            }),
            "global_oracle_status_counts": dict(sorted(Counter(row["status"] for row in finer_oracle).items())),
            "local_sse_status_counts": dict(sorted(Counter(
                row["status"] for row in rows
                if row["algorithm_variant"] == "local_sse"
                and metadata[row["case_id"]]["resolution_m"] < BASELINE_RESOLUTION_M
            ).items())),
            "completed": finer_complete,
            "computational_limit": finer_limit,
            "completed_resolutions_m": finer_completed_resolutions,
            "computational_limit_resolutions_m": finer_limit_resolutions,
        },
        "global_oracle_role": "tractable finite reference only",
        "power_law_fit_performed": False,
        "objective_convergence_established": False,
        "nonmonotonic_active_graph_warning": topology_shift,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_3_validation_report.json", validation)
    summary = {
        "stage": "14.3", "schema_version": STAGE14_3_SCHEMA_VERSION,
        "gate_passed": gate_passed, "resolutions_m": list(SPATIAL_RESOLUTIONS_M),
        "reused_repetition_count": len(reused_rows),
        "new_repetition_count": len(new_rows),
        "total_repetition_count": len(rows),
        "figure_files": figures, "gate_checks": checks,
        "figure_methodology": {
            "controlled_x_axis": "requested spatial resolution Delta x = Delta y = Delta h [m]",
            "individual_measurement_label": "Individual isolated-process runs (n=3)",
            "combined_state_edge_metric_used": False,
            "shared_resolution_figure_schema": True,
            "primary_algorithm": "local_sse",
            "reference_algorithm": "global_oracle",
            "displayed_algorithms": ["local_sse"],
            "oracle_measurements_retained": True,
            "performance_legend_algorithm_prefix": False,
            "individual_runtime_memory_points_displayed": False,
            "pod_figure": "04_1_detection_probability.png",
        },
        "finer_resolution_outcome": validation["finer_resolution_outcome"],
        "scientific_interpretation": {
            "power_law_fit_performed": False,
            "objective_convergence_established": False,
            "nonmonotonic_active_graph_warning": topology_shift,
        },
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_3_summary.json", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.3 exit gate failed: {failed}")
    print(
        "Stage 14.3 spatial-resolution sweep: PASS; "
        f"rows={len(rows)}, completed oracle resolutions="
        f"{validation['completed_global_oracle_resolutions_m']}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_3_diagnostics()
