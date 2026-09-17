"""Stage 14.6 local-neighborhood-radius sweep, figures, and exit gate."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_6_contract import (
    DEFENDER_ACTION_COUNT,
    DEFENDER_X_MAP,
    FULL_RADIUS,
    R_NEIGHBOR_VALUES,
    STAGE14_6_SCHEMA_VERSION,
    configuration_audit,
    stage14_6_cases,
)
from stage14_benchmark_contract import DEFAULT_STAGE14_TOLERANCES, environment_manifest
from stage14_benchmark_runner import BenchmarkCase, numeric_statistics, run_benchmark_cases
from stage14_resolution_figures import build_resolution_figure_set


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_6_neighbor_radius"
FROZEN_REFERENCE = ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
WORKER = ROOT / "stage14_6_worker.py"


def _metadata(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {
            "profile": case.name,
            "algorithm_variant": case.worker_mode,
            "r_neighbor": int(case.parameter_dict["r_neighbor"]),
            "source": "new_stage14_6_isolated_process_rows",
        }
        for case in cases
    }


def _runtime(row: dict[str, Any], key: str) -> float | None:
    value = ((row.get("timing") or {}).get("totals") or {}).get(key)
    return None if value is None else float(value)


def _local_audit(row: dict[str, Any]) -> dict[str, Any] | None:
    return ((row.get("local_search") or {}).get("radius_sweep_audit"))


def _stats(values: Iterable[float | int | None]) -> dict[str, Any] | None:
    return numeric_statistics([float(value) for value in values if value is not None])


def _summaries(
    rows: list[dict[str, Any]], cases: tuple[BenchmarkCase, ...],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    summaries = []
    for case in cases:
        case_rows = sorted(grouped.get(case.case_id, []), key=lambda row: row["repetition_id"])
        complete = [row for row in case_rows if row["status"] == "completed"]
        audits = [audit for row in complete if (audit := _local_audit(row)) is not None]
        identities = [
            (
                row["selected_defender_action_id"], row["selected_attacker_candidate_id"],
                row["J_A"], row["J_D"], (row.get("trajectory_identity") or {}).get("sha256"),
            )
            for row in complete
        ]
        runtime_keys = sorted({
            key for row in complete for key in ((row.get("timing") or {}).get("totals") or {})
        })
        summaries.append({
            "case_id": case.case_id,
            **metadata[case.case_id],
            "configuration": case.as_configuration(),
            "attempted_repetitions": len(case_rows),
            "successful_repetitions": len(complete),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "runtime_statistics_s": {
                key: _stats(_runtime(row, key) for row in complete) for key in runtime_keys
            },
            "peak_rss_statistics_bytes": _stats(
                row["memory"]["peak_rss_bytes"] for row in complete
            ),
            "unique_evaluation_statistics": _stats(
                audit["unique_exact_attacker_br_evaluations"] for audit in audits
            ),
            "raw_request_statistics": _stats(
                audit["raw_evaluation_requests"] for audit in audits
            ),
            "cache_hit_statistics": _stats(audit["cache_hits"] for audit in audits),
            "iteration_statistics": _stats(
                audit["local_search_iterations"] for audit in audits
            ),
            "summed_br_statistics_s": _stats(
                audit["summed_per_action_attacker_br_runtime_s"] for audit in audits
            ),
            "shared_graph_statistics_s": _stats(
                audit["shared_graph_runtime_s"] for audit in audits
            ),
            "local_overhead_statistics_s": _stats(
                audit["local_search_overhead_s"] for audit in audits
            ),
            "mean_neighborhood_statistics": _stats(
                sum(item["neighborhood_size"] for item in audit["per_iteration"])
                / len(audit["per_iteration"])
                for audit in audits if audit["per_iteration"]
            ),
            "max_neighborhood_statistics": _stats(
                max(item["neighborhood_size"] for item in audit["per_iteration"])
                for audit in audits if audit["per_iteration"]
            ),
            "J_A_statistics": _stats(row["J_A"] for row in complete),
            "J_D_statistics": _stats(row["J_D"] for row in complete),
            "selected_defender_action_ids": [row["selected_defender_action_id"] for row in complete],
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


def _local_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [item for item in summaries if item["algorithm_variant"] == "local_sse"],
        key=lambda item: item["r_neighbor"],
    )


def _median(item: dict[str, Any], key: str) -> float | None:
    value = item.get(key)
    return None if value is None else float(value["median"])


def _base_layout(title: str, y_title: str) -> dict[str, Any]:
    return {
        "title": title,
        "xaxis_title": "local-neighborhood radius r_neighbor [Defender grid lengths]",
        "yaxis_title": y_title,
        "template": "plotly_white",
    }


def _add_global_endpoint(
    figure: go.Figure, summaries: list[dict[str, Any]], extractor, *,
    name: str, yaxis: str | None = None, color: str = "#C00000",
    symbol: str = "diamond",
) -> None:
    oracle = next(
        (item for item in summaries if item["algorithm_variant"] == "global_oracle"), None
    )
    if oracle is None:
        return
    value = extractor(oracle)
    if value is None:
        return
    kwargs = {} if yaxis is None else {"yaxis": yaxis}
    figure.add_trace(go.Scatter(
        x=[FULL_RADIUS], y=[value], mode="markers", name=name,
        marker={"color": color, "symbol": symbol, "size": 12}, **kwargs,
    ))


def _legacy_figures(summaries: list[dict[str, Any]], output: Path) -> list[str]:
    local = _local_summaries(summaries)
    radii = [item["r_neighbor"] for item in local]
    names: list[str] = []

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=radii, y=[_median(item["runtime_statistics_s"], "T_SSE_s") for item in local],
        mode="lines+markers", name="Local SSE median total runtime",
        line={"color": "#4472C4", "width": 3}, marker={"size": 9},
    ))
    _add_global_endpoint(
        figure, summaries, lambda item: _median(item["runtime_statistics_s"], "T_SSE_s"),
        name="Exact global reference (full radius)",
    )
    figure.update_layout(**_base_layout(
        "Neighborhood-radius sweep: total runtime", "median runtime [s]",
    ))
    names.append("01_total_local_sse_runtime.png")
    save_figure_png(figure, output, names[-1])

    figure = go.Figure()
    for key, label, color, dash in (
        ("summed_br_statistics_s", "Summed exact Attacker BR", "#4472C4", "solid"),
        ("shared_graph_statistics_s", "Shared graph", "#70AD47", "dash"),
        ("local_overhead_statistics_s", "Local-search/validation overhead", "#ED7D31", "dot"),
    ):
        figure.add_trace(go.Scatter(
            x=radii, y=[_median(item, key) for item in local], mode="lines+markers",
            name=label, line={"color": color, "dash": dash, "width": 3},
            marker={"size": 9},
        ))
    figure.update_layout(**_base_layout(
        "Neighborhood-radius sweep: realized runtime decomposition", "median runtime [s]",
    ))
    names.append("02_attacker_br_and_overhead.png")
    save_figure_png(figure, output, names[-1])

    figure = go.Figure()
    for key, label, color, dash in (
        ("unique_evaluation_statistics", "Unique exact BR evaluations", "#4472C4", "solid"),
        ("raw_request_statistics", "Raw evaluation requests", "#ED7D31", "dash"),
        ("cache_hit_statistics", "Cache hits", "#70AD47", "dot"),
    ):
        figure.add_trace(go.Scatter(
            x=radii, y=[_median(item, key) for item in local], mode="lines+markers",
            name=label, line={"color": color, "dash": dash, "width": 3}, marker={"size": 9},
        ))
    figure.add_trace(go.Scatter(
        x=radii, y=[_median(item, "iteration_statistics") for item in local],
        mode="lines+markers", name="Local-search iterations K",
        line={"color": "#A64D79", "dash": "dashdot", "width": 3},
        marker={"symbol": "diamond", "size": 9}, yaxis="y2",
    ))
    layout = _base_layout(
        "Neighborhood-radius sweep: realized Defender-search work", "median action count",
    )
    layout["yaxis2"] = {
        "title": "median local-search iterations K", "overlaying": "y",
        "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    names.append("03_evaluations_and_iterations.png")
    save_figure_png(figure, output, names[-1])

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=radii, y=[_median(item, "mean_neighborhood_statistics") for item in local],
        mode="lines+markers", name="Mean realized neighborhood size",
        line={"color": "#4472C4", "width": 3}, marker={"size": 9},
    ))
    figure.add_trace(go.Scatter(
        x=radii, y=[_median(item, "max_neighborhood_statistics") for item in local],
        mode="lines+markers", name="Maximum realized neighborhood size",
        line={"color": "#ED7D31", "dash": "dash", "width": 3}, marker={"size": 9},
    ))
    figure.add_trace(go.Scatter(
        x=radii, y=[min(2 * radius, DEFENDER_ACTION_COUNT - 1) for radius in radii],
        mode="lines+markers", name="Interior-grid upper reference min(2r, N_D-1)",
        line={"color": "#777777", "dash": "dot", "width": 2.5}, marker={"size": 7},
    ))
    figure.update_layout(**_base_layout(
        "Neighborhood-radius sweep: realized neighborhood width", "neighbor action count",
    ))
    names.append("04_neighborhood_width.png")
    save_figure_png(figure, output, names[-1])

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=radii,
        y=[_median(item, "peak_rss_statistics_bytes") / 1024.0**2 for item in local],
        mode="lines+markers", name="Local SSE median peak RSS",
        line={"color": "#4472C4", "width": 3}, marker={"size": 9},
    ))
    _add_global_endpoint(
        figure, summaries,
        lambda item: _median(item, "peak_rss_statistics_bytes") / 1024.0**2,
        name="Exact global reference peak RSS",
    )
    figure.update_layout(**_base_layout(
        "Neighborhood-radius sweep: peak process-tree memory", "median peak RSS [MiB]",
    ))
    names.append("05_peak_memory.png")
    save_figure_png(figure, output, names[-1])

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=radii,
        y=[item["selected_defender_action_ids"][0] for item in local],
        mode="lines+markers", name="Selected local Defender action ID",
        line={"color": "#4472C4", "width": 3}, marker={"size": 9},
    ))
    figure.add_trace(go.Scatter(
        x=radii, y=[_median(item, "J_D_statistics") for item in local],
        mode="lines+markers", name="Local Defender J_D",
        line={"color": "#70AD47", "dash": "dash", "width": 3},
        marker={"size": 9}, yaxis="y2",
    ))
    _add_global_endpoint(
        figure, summaries,
        lambda item: float(item["selected_defender_action_ids"][0]),
        name="Exact global selected action",
    )
    _add_global_endpoint(
        figure, summaries, lambda item: _median(item, "J_D_statistics"),
        name="Exact global J_D", yaxis="y2", color="#A64D79", symbol="x",
    )
    layout = _base_layout(
        "Neighborhood-radius sweep: selected equilibrium and Defender payoff",
        "selected Defender action ID",
    )
    layout["yaxis2"] = {
        "title": "Defender objective J_D [PoD]", "overlaying": "y",
        "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    names.append("06_selected_action_and_defender_payoff.png")
    save_figure_png(figure, output, names[-1])
    return names


def _figures(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> list[str]:
    return build_resolution_figure_set(
        rows,
        metadata,
        x_key="r_neighbor",
        x_title="Neighbor radius r_neighbor [grid steps]",
        sweep_label="Neighborhood-radius sweep",
        output=output,
        performance_local_only=True,
        focused_runtime_display=True,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> None:
    fields = (
        "algorithm_variant", "r_neighbor", "repetition_id", "status", "T_SSE_s",
        "summed_T_BR_s", "shared_graph_s", "local_overhead_s", "unique_evaluations",
        "raw_requests", "cache_hits", "iterations", "selected_defender_action_id",
        "J_A", "J_D", "peak_rss_bytes", "configuration_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            audit = _local_audit(row) or {}
            writer.writerow({
                "algorithm_variant": metadata[row["case_id"]]["algorithm_variant"],
                "r_neighbor": metadata[row["case_id"]]["r_neighbor"],
                "repetition_id": row["repetition_id"], "status": row["status"],
                "T_SSE_s": _runtime(row, "T_SSE_s"),
                "summed_T_BR_s": audit.get("summed_per_action_attacker_br_runtime_s"),
                "shared_graph_s": audit.get("shared_graph_runtime_s"),
                "local_overhead_s": audit.get("local_search_overhead_s"),
                "unique_evaluations": audit.get("unique_exact_attacker_br_evaluations"),
                "raw_requests": audit.get("raw_evaluation_requests"),
                "cache_hits": audit.get("cache_hits"),
                "iterations": audit.get("local_search_iterations"),
                "selected_defender_action_id": row["selected_defender_action_id"],
                "J_A": row["J_A"], "J_D": row["J_D"],
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
            })


def _full_radius_equivalence(rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    local = [
        row for row in rows if row["status"] == "completed"
        and metadata[row["case_id"]]["algorithm_variant"] == "local_sse"
        and metadata[row["case_id"]]["r_neighbor"] == FULL_RADIUS
    ]
    oracle = [
        row for row in rows if row["status"] == "completed"
        and metadata[row["case_id"]]["algorithm_variant"] == "global_oracle"
    ]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    comparisons = []
    for left in local:
        for right in oracle:
            checks = {
                "selected_defender_action_id": left["selected_defender_action_id"] == right["selected_defender_action_id"],
                "selected_attacker_candidate_id": left["selected_attacker_candidate_id"] == right["selected_attacker_candidate_id"],
                "trajectory_sha256": (left.get("trajectory_identity") or {}).get("sha256") == (right.get("trajectory_identity") or {}).get("sha256"),
                "J_A": abs(float(left["J_A"]) - float(right["J_A"])) <= tolerance.attacker_objective_abs,
                "J_D": abs(float(left["J_D"]) - float(right["J_D"])) <= tolerance.defender_objective_abs,
            }
            comparisons.append({
                "local_repetition_id": left["repetition_id"],
                "global_repetition_id": right["repetition_id"],
                "checks": checks, "passed": all(checks.values()),
            })
    return {
        "passed": bool(comparisons and all(item["passed"] for item in comparisons)),
        "local_completed": len(local), "global_completed": len(oracle),
        "cross_product_comparisons": comparisons,
    }


def run_stage14_6_diagnostics(
    output_directory: Path = OUTPUT, *, force: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = stage14_6_cases()
    contract = configuration_audit(cases)
    if not contract["passed"]:
        raise RuntimeError("Stage-14.6 configuration audit failed")
    metadata = _metadata(cases)
    environment = environment_manifest(ROOT.parent)
    measurements = output_directory / "measurements"
    first = run_benchmark_cases(
        cases, output_directory=measurements,
        environment_manifest_id=environment["environment_manifest_id"],
        force=force, worker_path=WORKER,
    )
    second = run_benchmark_cases(
        cases, output_directory=measurements,
        environment_manifest_id=environment["environment_manifest_id"],
        worker_path=WORKER,
    )
    case_ids = {case.case_id for case in cases}
    rows = sorted(
        [row for row in second["rows"] if row["case_id"] in case_ids],
        key=lambda row: (metadata[row["case_id"]]["r_neighbor"], row["algorithm_variant"], row["repetition_index"]),
    )
    for row in rows:
        row["stage14_6_source"] = metadata[row["case_id"]]["source"]
        row["process_semantics"] = (
            "fresh isolated Python worker; OS and hardware cache state uncontrolled"
        )
    summaries = _summaries(rows, cases, metadata)
    figures = _figures(rows, metadata, output_directory)
    complete = [row for row in rows if row["status"] == "completed"]
    local = [row for row in complete if row["algorithm_variant"] == "local_sse"]
    oracle = [row for row in complete if row["algorithm_variant"] == "global_oracle"]
    equivalence = _full_radius_equivalence(rows, metadata)
    frozen = json.loads(FROZEN_REFERENCE.read_text(encoding="utf-8"))
    checks = {
        "configuration_contract_passes": contract["passed"],
        "only_r_neighbor_varied_in_local_sweep": contract["checks"]["only_r_neighbor_varies_across_local_cases"],
        "fixed_defender_grid_start_and_attacker_settings": bool(
            contract["checks"]["fixed_six_action_uniform_grid"]
            and contract["checks"]["fixed_start_action_zero"]
            and contract["checks"]["exact_non_rl_single_start"]
        ),
        "three_repetitions_per_configuration_recorded": len(rows) == len(cases) * 3,
        "all_measurements_completed": len(complete) == len(rows),
        "runner_resume_deterministic": not second["attempted_repetition_ids"]
        and len(second["skipped_repetition_ids"]) == len(cases) * 3,
        "all_realized_neighborhoods_match_rule": bool(local and all(
            _local_audit(row)["neighborhood_rule_passed"] for row in local
        )),
        "all_evaluation_accounting_reconciles": bool(local and all(
            _local_audit(row)["request_accounting_passed"]
            and _local_audit(row)["reported_cache_accounting_passed"]
            and _local_audit(row)["unique_record_accounting_passed"]
            for row in local
        )),
        "all_local_exactness_and_strong_ties_verified": bool(local and all(
            (row.get("exactness") or {}).get("all_evaluated_attacker_responses_exact") is True
            and (row.get("exactness") or {}).get("strong_follower_tie_break_verified") is True
            and (row.get("local_search") or {}).get("local_sse_verified") is True
            for row in local
        )),
        "all_independent_replays_pass": bool(complete and all(
            row["independent_replay_passed"] is True for row in complete
        )),
        "full_radius_covers_complete_grid": bool(local and all(
            _local_audit(row)["full_grid_covered"] for row in local
            if metadata[row["case_id"]]["r_neighbor"] == FULL_RADIUS
        )),
        "full_radius_matches_exact_global": equivalence["passed"],
        "global_reference_enumerates_complete_grid": bool(oracle and all(
            (row.get("oracle_metadata") or {}).get("stage14_6_full_radius_reference", {}).get("complete_action_grid_covered") is True
            for row in oracle
        )),
        "deterministic_solution_per_configuration": all(
            item["deterministic_solution_identity"] for item in summaries
        ),
        "solver_source_fingerprint_unchanged": bool(complete and all(
            row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"]
            for row in complete
        )),
        "six_static_png_figures_exist": len(figures) == 6 and all(
            (output_directory / name).is_file()
            and (output_directory / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            for name in figures
        ),
    }
    gate_passed = all(checks.values())
    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "radius_sweep_configuration.json", contract)
    write_json(output_directory / "runner_resume_report.json", {
        "first_invocation_attempted": first["attempted_repetition_ids"],
        "first_invocation_skipped": first["skipped_repetition_ids"],
        "second_invocation_attempted": second["attempted_repetition_ids"],
        "second_invocation_skipped": second["skipped_repetition_ids"],
    })
    write_json(output_directory / "raw_radius_repetitions.json", rows)
    _write_jsonl(output_directory / "raw_radius_repetitions.jsonl", rows)
    _write_csv(output_directory / "raw_radius_repetitions.csv", rows, metadata)
    write_json(output_directory / "radius_sweep_summaries.json", summaries)
    write_json(output_directory / "full_radius_global_equivalence.json", equivalence)
    validation = {
        "schema_version": STAGE14_6_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "runtime_interpretation": (
            "No monotonicity is assumed: larger radius widens each neighborhood but may reduce K; "
            "interpret total time using unique exact BR evaluations and measured overhead."
        ),
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_6_validation_report.json", validation)
    summary = {
        "stage": "14.6", "schema_version": STAGE14_6_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "r_neighbor_values": list(R_NEIGHBOR_VALUES),
        "defender_action_count": DEFENDER_ACTION_COUNT,
        "defender_x_map": list(DEFENDER_X_MAP),
        "full_radius": FULL_RADIUS,
        "total_repetition_count": len(rows),
        "figure_files": figures,
        "full_radius_global_equivalence_passed": equivalence["passed"],
        "gate_checks": checks,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_6_summary.json", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.6 exit gate failed: {failed}")
    print(
        "Stage 14.6 neighborhood-radius sweep: PASS; "
        f"rows={len(rows)}, radii={list(R_NEIGHBOR_VALUES)}"
    )
    return summary


def regenerate_stage14_6_figures_from_saved_data(
    output_directory: Path = OUTPUT,
) -> list[str]:
    rows = json.loads(
        (output_directory / "raw_radius_repetitions.json").read_text(encoding="utf-8")
    )
    # Saved measurements may predate the current six-action grid. Keep their
    # own case IDs and radius values; re-rendering must not relabel old data.
    metadata = {
        row["case_id"]: {"r_neighbor": int(row["configuration"]["parameters"]["r_neighbor"])}
        for row in rows
    }
    figures = _figures(rows, metadata, output_directory)
    summary_path = output_directory / "stage14_6_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["figure_files"] = figures
    summary["figure_methodology"] = {
        "controlled_x_axis": "r_neighbor [grid steps]",
        "performance_displayed_algorithms": ["local_sse"],
        "performance_legend_algorithm_prefix": False,
        "other_solver_control_displayed": False,
        "saved_measurement_configuration_preserved": True,
        "shared_performance_figure_schema": True,
        "search_work_diagnostics_location": "raw/tabular audit artifacts only",
        "primary_algorithm": "local_sse",
        "reference_algorithm": "global_oracle",
    }
    write_json(summary_path, summary)
    return figures


if __name__ == "__main__":
    run_stage14_6_diagnostics()
