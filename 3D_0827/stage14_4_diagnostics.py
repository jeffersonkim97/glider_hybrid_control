"""Stage 14.4 controlled heading-resolution sweep and exit gate."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from attacker_best_response import DEFAULT_GRAPH_BOUNDS
from stage11_notebook_support import write_json
from stage14_2c_worker import _build_case
from stage14_4_contract import (
    BASELINE_HEADING_SPACING_DEG,
    FINE_HEADING_SPACING_DEG,
    HEADING_SPACINGS_DEG,
    STAGE14_4_SCHEMA_VERSION,
    configuration_audit,
    new_stage14_4_cases,
    reused_stage14_2c_cases,
    stage14_4_cases,
)
from stage14_benchmark_contract import DEFAULT_STAGE14_TOLERANCES, environment_manifest
from stage14_benchmark_runner import BenchmarkCase, numeric_statistics, run_benchmark_cases
from stage14_resolution_figures import build_resolution_figure_set


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_4_heading_resolution"
STAGE14_2C_OUTPUT = ROOT / "figure" / "stage_14_2c_local_scaling"
FROZEN_REFERENCE = ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
WORKER = ROOT / "stage14_4_worker.py"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile(case: BenchmarkCase) -> str:
    return case.name.removeprefix(f"{case.worker_mode}_")


def _realized_heading(case: BenchmarkCase) -> dict[str, Any]:
    configuration = case.as_configuration() | {
        "case_id": case.case_id,
        "repetition_id": case.repetition_id(0),
    }
    stage_config, _candidates, _initial = _build_case(configuration)
    grid = stage_config.discretization.build_bellman_grid(DEFAULT_GRAPH_BOUNDS)
    realized_spacing = 360.0 / float(grid.heading_bin_count)
    return {
        "requested_heading_spacing_deg": float(
            case.parameter_dict["heading_spacing_deg"]
        ),
        "heading_bin_count": int(grid.heading_bin_count),
        "realized_heading_spacing_deg": realized_spacing,
        "maximum_heading_quantization_error_deg": realized_spacing / 2.0,
        "period_deg": 360.0,
    }


def _metadata(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    reused_ids = {case.case_id for case in reused_stage14_2c_cases()}
    return {
        case.case_id: {
            "profile": _profile(case),
            "algorithm_variant": case.worker_mode,
            "heading_spacing_deg": float(case.parameter_dict["heading_spacing_deg"]),
            "source": (
                "reused_stage14_2c_isolated_process_rows"
                if case.case_id in reused_ids
                else "new_stage14_4_isolated_process_rows"
            ),
            "realized_heading_grid": _realized_heading(case),
        }
        for case in cases
    }


def _runtime(row: dict[str, Any], key: str = "T_SSE_s") -> float | None:
    value = ((row.get("timing") or {}).get("totals") or {}).get(key)
    return None if value is None else float(value)


def _completed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _summaries(
    rows: list[dict[str, Any]], cases: tuple[BenchmarkCase, ...],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    summaries = []
    for case in cases:
        case_rows = sorted(grouped.get(case.case_id, []), key=lambda item: item["repetition_id"])
        complete = _completed(case_rows)
        component_names = sorted({
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
        summaries.append({
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
                for key in component_names
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
    return summaries


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> None:
    fields = (
        "profile", "algorithm_variant", "source", "requested_heading_spacing_deg",
        "realized_heading_spacing_deg", "N_psi", "maximum_heading_quantization_error_deg",
        "case_id", "repetition_id", "status", "failure_category", "failure_message",
        "T_SSE_s", "T_Bellman_s", "peak_rss_bytes", "N_S_cart", "N_S_active", "N_E",
        "J_A", "J_D", "selected_defender_action_id", "selected_attacker_candidate_id",
        "independent_replay_passed", "configuration_json", "timing_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = metadata[row["case_id"]]
            heading = meta["realized_heading_grid"]
            counts = row.get("state_and_game_size") or {}
            writer.writerow({
                "profile": meta["profile"],
                "algorithm_variant": meta["algorithm_variant"],
                "source": meta["source"],
                "requested_heading_spacing_deg": heading["requested_heading_spacing_deg"],
                "realized_heading_spacing_deg": heading["realized_heading_spacing_deg"],
                "N_psi": heading["heading_bin_count"],
                "maximum_heading_quantization_error_deg": heading["maximum_heading_quantization_error_deg"],
                "case_id": row["case_id"], "repetition_id": row["repetition_id"],
                "status": row["status"], "failure_category": row["failure_category"],
                "failure_message": row["failure_message"],
                "T_SSE_s": _runtime(row), "T_Bellman_s": _runtime(row, "T_Bellman_s"),
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
                "N_S_cart": counts.get("N_S_cart"), "N_S_active": counts.get("N_S_active"),
                "N_E": counts.get("N_E"), "J_A": row["J_A"], "J_D": row["J_D"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "independent_replay_passed": row["independent_replay_passed"],
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
                "timing_json": json.dumps(row.get("timing"), sort_keys=True),
            })


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "profile", "algorithm_variant", "source", "requested_heading_spacing_deg",
        "realized_heading_spacing_deg", "N_psi", "maximum_heading_quantization_error_deg",
        "status_counts", "successful_repetitions", "T_SSE_median_s", "T_Bellman_median_s",
        "peak_rss_median_bytes", "N_S_cart", "N_S_active", "N_E", "J_A", "J_D",
        "deterministic_solution_identity", "all_completed_replays_pass",
        "all_completed_exactness_checks_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            heading = summary["realized_heading_grid"]
            counts = summary["state_and_game_size"] or {}
            components = summary["runtime_component_statistics_s"]
            writer.writerow({
                "profile": summary["profile"], "algorithm_variant": summary["algorithm_variant"],
                "source": summary["source"],
                "requested_heading_spacing_deg": heading["requested_heading_spacing_deg"],
                "realized_heading_spacing_deg": heading["realized_heading_spacing_deg"],
                "N_psi": heading["heading_bin_count"],
                "maximum_heading_quantization_error_deg": heading["maximum_heading_quantization_error_deg"],
                "status_counts": json.dumps(summary["status_counts"], sort_keys=True),
                "successful_repetitions": summary["successful_repetitions"],
                "T_SSE_median_s": None if summary["T_SSE_statistics_s"] is None else summary["T_SSE_statistics_s"]["median"],
                "T_Bellman_median_s": None if components.get("T_Bellman_s") is None else components["T_Bellman_s"]["median"],
                "peak_rss_median_bytes": None if summary["peak_rss_statistics_bytes"] is None else summary["peak_rss_statistics_bytes"]["median"],
                "N_S_cart": counts.get("N_S_cart"), "N_S_active": counts.get("N_S_active"),
                "N_E": counts.get("N_E"),
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
        x_key="heading_spacing_deg",
        x_title="requested heading resolution Delta psi [deg]",
        sweep_label="Heading sweep",
        output=output,
        variants=("local_sse",),
        clean_performance_display=True,
        include_pod_figure=True,
        focused_runtime_display=True,
    )


def regenerate_stage14_4_figures_from_saved_data(
    output_directory: Path = OUTPUT,
) -> list[str]:
    """Re-render the Stage-14.4 report without invoking benchmark workers."""
    rows = _read_json(output_directory / "raw_heading_repetitions.json")
    metadata = _metadata(stage14_4_cases())
    figures = _figures(rows, metadata, output_directory)
    summary_path = output_directory / "stage14_4_summary.json"
    summary = _read_json(summary_path)
    summary["figure_files"] = figures
    summary["figure_methodology"] = {
        "controlled_x_axis": "requested heading resolution Delta psi [deg]",
        "realized_heading_bins_location": "summary table and machine-readable metadata only",
        "individual_measurement_label": "Individual isolated-process runs (n=3)",
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
        "Stage 14.4 figures regenerated from saved measurements only; "
        "no benchmark worker was invoked."
    )
    return figures

def run_stage14_4_diagnostics(
    output_directory: Path = OUTPUT, *, force: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = stage14_4_cases()
    audit = configuration_audit(cases)
    if not audit["passed"]:
        raise RuntimeError("Stage-14.4 configuration audit failed")
    metadata = _metadata(cases)
    environment = environment_manifest(ROOT.parent)
    new_directory = output_directory / "new_fine_measurements"
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
        # Full notebook recomputation measures this complete stage locally;
        # it does not silently reuse Stage-14.2C timing rows.
        measurement_cases = cases
        reused_rows: list[dict[str, Any]] = []
    else:
        source_rows = _read_json(STAGE14_2C_OUTPUT / "raw_repetitions.json")
        reused_ids = {case.case_id for case in reused_stage14_2c_cases()}
        reused_rows = [row for row in source_rows if row["case_id"] in reused_ids]
        if len(reused_rows) != 18 or any(
            row["status"] != "completed" for row in reused_rows
        ):
            raise RuntimeError("Stage-14.2C reusable heading rows are incomplete")
        measurement_cases = new_stage14_4_cases()
    first = run_benchmark_cases(
        measurement_cases, output_directory=new_directory,
        environment_manifest_id=environment["environment_manifest_id"], force=force,
        worker_path=WORKER,
    )
    second = run_benchmark_cases(
        measurement_cases, output_directory=new_directory,
        environment_manifest_id=environment["environment_manifest_id"], worker_path=WORKER,
    )
    new_ids = {case.case_id for case in measurement_cases}
    new_rows = [row for row in second["rows"] if row["case_id"] in new_ids]
    rows = sorted(reused_rows + new_rows, key=lambda row: (row["case_id"], row["repetition_index"]))
    for row in rows:
        row["algorithm_variant"] = row.get("algorithm_variant") or row["configuration"]["worker_mode"]
        row["stage14_4_source"] = (
            "fresh_stage14_4_isolated_process_rows"
            if full_fresh_measurement else metadata[row["case_id"]]["source"]
        )
        row["realized_heading_grid"] = metadata[row["case_id"]]["realized_heading_grid"]

    summaries = _summaries(rows, cases, metadata)
    figures = _figures(rows, metadata, output_directory)
    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "heading_sweep_configuration.json", audit)
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
    write_json(output_directory / "raw_heading_repetitions.json", rows)
    _write_jsonl(output_directory / "raw_heading_repetitions.jsonl", rows)
    _write_csv(output_directory / "raw_heading_repetitions.csv", rows, metadata)
    write_json(output_directory / "heading_sweep_summaries.json", summaries)
    _write_summary_csv(output_directory / "heading_sweep_summaries.csv", summaries)

    completed = _completed(rows)
    local_rows = [row for row in rows if row["algorithm_variant"] == "local_sse"]
    fine_local = [
        row for row in local_rows
        if metadata[row["case_id"]]["heading_spacing_deg"] == FINE_HEADING_SPACING_DEG
    ]
    fine_local_completed = all(row["status"] == "completed" for row in fine_local)
    fine_local_limit = bool(fine_local) and all(
        row["status"] in {"timeout", "memory_limit"} for row in fine_local
    )
    completed_local_headings = sorted({
        metadata[row["case_id"]]["heading_spacing_deg"]
        for row in completed if row["algorithm_variant"] == "local_sse"
    })
    completed_oracle_headings = sorted({
        metadata[row["case_id"]]["heading_spacing_deg"]
        for row in completed if row["algorithm_variant"] == "global_oracle"
    })
    frozen = _read_json(FROZEN_REFERENCE)
    expected = frozen["solution_identity"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    baseline = [
        row for row in completed
        if metadata[row["case_id"]]["heading_spacing_deg"] == BASELINE_HEADING_SPACING_DEG
    ]
    baseline_regression = bool(baseline and all(
        row["selected_defender_action_id"] == expected["selected_defender_action_id"]
        and row["selected_attacker_candidate_id"] == expected["selected_attacker_candidate_id"]
        and abs(float(row["J_A"]) - float(expected["attacker_objective"])) <= tolerance.attacker_objective_abs
        and abs(float(row["J_D"]) - float(expected["defender_objective_pod"])) <= tolerance.defender_objective_abs
        for row in baseline
    ))
    expected_npsi = {
        spacing: int(round(360.0 / spacing)) for spacing in HEADING_SPACINGS_DEG
    }
    checks = {
        "only_heading_resolution_varied": audit["checks"]["only_heading_resolution_varies"],
        "required_heading_spacings_declared": audit["checks"]["required_heading_spacings_present"],
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
        "fine_local_completed_or_explicit_computational_limit": fine_local_completed or fine_local_limit,
        "realized_heading_metadata_recorded": all(
            row.get("realized_heading_grid") for row in rows
        ),
        "realized_heading_bins_match_requested": all(
            metadata[row["case_id"]]["realized_heading_grid"]["heading_bin_count"]
            == expected_npsi[metadata[row["case_id"]]["heading_spacing_deg"]]
            for row in rows
        ),
        "every_completed_replay_passes": bool(
            completed and all(row["independent_replay_passed"] is True for row in completed)
        ),
        "every_completed_exactness_check_passes": bool(
            completed and all(all((row.get("exactness") or {}).values()) for row in completed)
        ),
        "every_completed_local_result_is_certified": all(
            (row.get("local_search") or {}).get("local_sse_verified") is True
            for row in completed if row["algorithm_variant"] == "local_sse"
        ),
        "completed_solution_identities_deterministic": all(
            summary["deterministic_solution_identity"]
            for summary in summaries if summary["successful_repetitions"]
        ),
        "canonical_frozen_regression_passes": baseline_regression,
        "solver_source_fingerprint_unchanged": all(
            row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"]
            for row in completed
        ),
        "global_solver_is_oracle_only": all(
            row["algorithm_variant"] != "global_oracle"
            or (row.get("oracle_metadata") or {}).get("role")
            == "tractable finite global oracle/reference only"
            for row in completed
        ),
        "png_figures_exist": all((output_directory / name).is_file() for name in figures),
    }
    gate_passed = all(checks.values())
    validation = {
        "schema_version": STAGE14_4_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "completed_local_sse_heading_spacings_deg": completed_local_headings,
        "completed_global_oracle_heading_spacings_deg": completed_oracle_headings,
        "fine_heading_outcome": {
            "heading_spacing_deg": FINE_HEADING_SPACING_DEG,
            "local_status_counts": dict(sorted(Counter(row["status"] for row in fine_local).items())),
            "completed": fine_local_completed,
            "computational_limit": fine_local_limit,
        },
        "primary_research_object": "certified local SSE with exact finite Attacker responses",
        "global_oracle_role": "tractable finite reference only",
        "power_law_fit_performed": False,
        "objective_convergence_established": False,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_4_validation_report.json", validation)
    summary = {
        "stage": "14.4", "schema_version": STAGE14_4_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "heading_spacings_deg": list(HEADING_SPACINGS_DEG),
        "reused_repetition_count": len(reused_rows),
        "new_repetition_count": len(new_rows),
        "total_repetition_count": len(rows),
        "figure_files": figures,
        "figure_methodology": {
            "controlled_x_axis": "requested heading resolution Delta psi [deg]",
            "realized_heading_bins_location": "summary table and machine-readable metadata only",
            "individual_measurement_label": "Individual isolated-process runs (n=3)",
            "shared_resolution_figure_schema": True,
            "primary_algorithm": "local_sse",
            "reference_algorithm": "global_oracle",
            "displayed_algorithms": ["local_sse"],
            "oracle_measurements_retained": True,
            "performance_legend_algorithm_prefix": False,
            "individual_runtime_memory_points_displayed": False,
            "pod_figure": "04_1_detection_probability.png",
        },
        "gate_checks": checks,
        "fine_heading_outcome": validation["fine_heading_outcome"],
        "scientific_interpretation": {
            "power_law_fit_performed": False,
            "objective_convergence_established": False,
        },
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_4_summary.json", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.4 exit gate failed: {failed}")
    print(
        "Stage 14.4 heading-resolution sweep: PASS; "
        f"rows={len(rows)}, completed local headings={completed_local_headings}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_4_diagnostics()
