"""Stage 14.2 smoke benchmark, artifacts, and repetition-stability diagnostic."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_benchmark_runner import (
    BenchmarkCase,
    RUNNER_SCHEMA_VERSION,
    run_benchmark_cases,
    summarize_repetitions,
    write_raw_csv,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2_benchmark_runner"
ENVIRONMENT_MANIFEST = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "environment_manifest.json"
)
FROZEN_REFERENCE = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
)


def stage14_2_smoke_cases() -> tuple[BenchmarkCase, ...]:
    """Small matrix: repeated canonical, infeasible, and controlled timeout."""
    return (
        BenchmarkCase(
            experiment_type="stage14_2_smoke",
            name="canonical_centered_cube_25m_25m_5deg",
            worker_mode="canonical",
            sweep_variable="horizontal_spacing_m",
            parameters=(
                ("horizontal_spacing_m", 25.0),
                ("altitude_spacing_m", 25.0),
                ("heading_spacing_deg", 5.0),
                ("terrain_category", "centered_cube"),
            ),
            repetitions=3,
            timeout_s=300.0,
        ),
        BenchmarkCase(
            experiment_type="stage14_2_smoke",
            name="explicit_infeasible_centered_cube_small",
            worker_mode="infeasible",
            sweep_variable="switching_contour_sample_count",
            parameters=(
                ("switching_contour_sample_count", 2),
                ("terrain_category", "centered_cube"),
                ("vehicle_and_objectives_unchanged", True),
            ),
            repetitions=1,
            timeout_s=30.0,
        ),
        BenchmarkCase(
            experiment_type="stage14_2_smoke",
            name="controlled_timeout_fixture",
            worker_mode="controlled_timeout",
            sweep_variable="delay_s",
            parameters=(("delay_s", 2.0),),
            repetitions=1,
            timeout_s=0.25,
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration_table(cases: tuple[BenchmarkCase, ...]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "name": case.name,
            "worker_mode": case.worker_mode,
            "declared_sweep_variable": case.sweep_variable,
            "declared_sweep_value": case.parameter_dict[case.sweep_variable],
            "repetitions": case.repetitions,
            "timeout_s": case.timeout_s,
            "memory_limit_bytes": case.memory_limit_bytes,
            "process_semantics": "fresh cold process per repetition",
            "parameters": case.parameter_dict,
        }
        for case in cases
    ]


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "case_id", "attempted_count", "status_counts_json", "T_SSE_count",
        "T_SSE_median_s", "T_SSE_mean_s", "T_SSE_std_sample_s",
        "T_SSE_minimum_s", "T_SSE_maximum_s", "configuration_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            stats = summary["T_SSE_statistics_s"] or {}
            writer.writerow({
                "case_id": summary["case_id"],
                "attempted_count": summary["attempted_count"],
                "status_counts_json": json.dumps(summary["status_counts"], sort_keys=True),
                "T_SSE_count": stats.get("count", 0),
                "T_SSE_median_s": stats.get("median"),
                "T_SSE_mean_s": stats.get("mean"),
                "T_SSE_std_sample_s": stats.get("standard_deviation_sample"),
                "T_SSE_minimum_s": stats.get("minimum"),
                "T_SSE_maximum_s": stats.get("maximum"),
                "configuration_json": json.dumps(
                    summary["configuration"], sort_keys=True, separators=(",", ":")
                ),
            })


def _repetition_figure(
    canonical_rows: list[dict[str, Any]],
    canonical_summary: dict[str, Any],
) -> go.Figure:
    values = [row["timing"]["totals"]["T_SSE_s"] for row in canonical_rows]
    indices = [row["repetition_index"] for row in canonical_rows]
    stats = canonical_summary["T_SSE_statistics_s"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=indices,
        y=values,
        mode="markers+text",
        text=[f"{value:.3f} s" for value in values],
        textposition="top center",
        marker={"size": 12, "color": "#4472C4"},
        name="fresh-process repetition",
    ))
    figure.add_hline(
        y=stats["median"], line_dash="dash", line_color="#C00000",
        annotation_text=f"median {stats['median']:.3f} s",
    )
    figure.add_trace(go.Scatter(
        x=[-0.25, len(values) - 0.75],
        y=[stats["minimum"], stats["maximum"]],
        mode="lines+markers",
        line={"color": "#70AD47", "width": 5},
        marker={"size": 9},
        name="min–max range",
    ))
    figure.update_layout(
        title="Stage 14.2 canonical exact-SSE repetition stability (no scaling fit)",
        xaxis={"title": "Cold-process repetition index", "dtick": 1},
        yaxis={"title": "T_SSE [s]"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 110},
    )
    return figure


def _validate_exact_repetitions(
    canonical_rows: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    expected = frozen["solution_identity"]
    checks: list[dict[str, Any]] = []
    for row in canonical_rows:
        identity_match = bool(
            np.isclose(row["J_A"], expected["attacker_objective"], atol=1.0e-12, rtol=0.0)
            and np.isclose(row["J_D"], expected["defender_objective_pod"], atol=1.0e-12, rtol=0.0)
            and row["selected_defender_action_id"]
            == expected["selected_defender_action_id"]
            and row["selected_attacker_candidate_id"]
            == expected["selected_attacker_candidate_id"]
            and row["trajectory_identity"]["sha256"]
            == expected["trajectory_identity"]["sha256"]
            and row["tie_breaking_result"]["leader_cooptimal_action_ids"]
            == expected["leader_cooptimal_action_ids"]
            and row["tie_breaking_result"]["attacker_objective_cooptimal_candidate_ids"]
            == expected["attacker_objective_cooptimal_candidate_ids"]
            and row["independent_replay_passed"] is True
        )
        checks.append({"repetition_id": row["repetition_id"], "passed": identity_match})
    return {"passed": bool(checks and all(item["passed"] for item in checks)), "checks": checks}


def run_stage14_2_diagnostics(
    output_directory: Path = OUTPUT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run/resume the Stage-14.2 smoke matrix and validate its artifact contract."""
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = _read_json(ENVIRONMENT_MANIFEST)
    frozen = _read_json(FROZEN_REFERENCE)
    cases = stage14_2_smoke_cases()
    write_json(output_directory / "smoke_case_configurations.json", {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "exactly_one_declared_sweep_variable_per_case": True,
        "cases": _configuration_table(cases),
    })
    first = run_benchmark_cases(
        cases,
        output_directory=output_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        force=force,
    )
    second = run_benchmark_cases(
        cases,
        output_directory=output_directory,
        environment_manifest_id=environment["environment_manifest_id"],
    )
    rows = second["rows"]
    summaries = summarize_repetitions(rows)
    write_json(output_directory / "raw_repetitions.json", rows)
    write_raw_csv(output_directory / "raw_repetitions.csv", rows)
    write_json(output_directory / "configuration_summaries.json", summaries)
    _write_summary_csv(output_directory / "configuration_summaries.csv", summaries)
    expected_attempts = sum(case.repetitions for case in cases)
    resume_valid = bool(
        len(rows) == expected_attempts
        and not second["attempted_repetition_ids"]
        and len(second["skipped_repetition_ids"]) == expected_attempts
        and len({row["repetition_id"] for row in rows}) == expected_attempts
    )
    resume_report = {
        "first_invocation_attempted": first["attempted_repetition_ids"],
        "first_invocation_skipped": first["skipped_repetition_ids"],
        "second_invocation_attempted": second["attempted_repetition_ids"],
        "second_invocation_skipped": second["skipped_repetition_ids"],
        "deterministic_resume_passed": resume_valid,
    }
    write_json(output_directory / "resume_report.json", resume_report)
    canonical_case = cases[0]
    canonical_rows = sorted(
        [row for row in rows if row["case_id"] == canonical_case.case_id],
        key=lambda row: row["repetition_index"],
    )
    canonical_summary = next(
        summary for summary in summaries if summary["case_id"] == canonical_case.case_id
    )
    exact_regression = _validate_exact_repetitions(canonical_rows, frozen)
    statuses = Counter(row["status"] for row in rows)
    smoke_matrix_valid = bool(
        statuses == Counter({"completed": 3, "model_infeasible": 1, "timeout": 1})
        and all(row["environment_manifest_id"] == environment["environment_manifest_id"] for row in rows)
        and all(row["process_semantics"] == "fresh cold process" for row in rows)
    )
    statistics_reproduced = bool(
        canonical_summary["T_SSE_statistics_s"] is not None
        and canonical_summary["T_SSE_statistics_s"]["count"] == 3
        and np.isclose(
            canonical_summary["T_SSE_statistics_s"]["median"],
            np.median([row["timing"]["totals"]["T_SSE_s"] for row in canonical_rows]),
            atol=1.0e-12,
            rtol=0.0,
        )
    )
    gate_passed = bool(
        exact_regression["passed"]
        and smoke_matrix_valid
        and statistics_reproduced
        and resume_valid
    )
    if not gate_passed:
        raise RuntimeError("Stage-14.2 benchmark-runner gate failed")
    save_figure_png(
        _repetition_figure(canonical_rows, canonical_summary),
        output_directory,
        "canonical_repetition_stability.png",
    )
    result = {
        "stage": "14.2",
        "schema_version": RUNNER_SCHEMA_VERSION,
        "gate_passed": True,
        "full_scaling_sweep_performed": False,
        "scaling_law_fitted": False,
        "environment_manifest_id": environment["environment_manifest_id"],
        "attempted_case_count": len(cases),
        "attempted_repetition_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "canonical_case_id": canonical_case.case_id,
        "canonical_repetition_statistics": canonical_summary["T_SSE_statistics_s"],
        "exact_solution_regression": exact_regression,
        "all_attempts_recorded": smoke_matrix_valid,
        "summary_statistics_reproduced": statistics_reproduced,
        "resume_behavior_deterministic": resume_valid,
        "resume_report": resume_report,
    }
    write_json(output_directory / "stage14_2_summary.json", result)
    stats = result["canonical_repetition_statistics"]
    print("Stage 14.2 reproducible benchmark runner: PASS")
    print(
        f"canonical n={stats['count']}; median={stats['median']:.6f} s; "
        f"range=[{stats['minimum']:.6f}, {stats['maximum']:.6f}] s"
    )
    print(f"statuses={result['status_counts']}; resume=PASS")
    return result


if __name__ == "__main__":
    run_stage14_2_diagnostics()
