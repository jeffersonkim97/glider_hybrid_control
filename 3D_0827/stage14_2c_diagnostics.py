"""Run/resume Stage 14.2C scaling, oracle comparison, and diagnostics."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

import numpy as np
import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_2c_contract import (
    STAGE14_2C_SCHEMA_VERSION,
    configuration_audit,
    stage14_2c_cases,
)
from stage14_benchmark_contract import environment_manifest
from stage14_benchmark_runner import (
    BenchmarkCase,
    numeric_statistics,
    run_benchmark_cases,
    summarize_repetitions,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2c_local_scaling"
FROZEN_REFERENCE = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
)
WORKER = ROOT / "stage14_2c_worker.py"
NUMERICAL_TOLERANCE = 1.0e-12


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_metadata(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for case in cases:
        variant = case.worker_mode
        prefix = f"{variant}_"
        profile = case.name[len(prefix):] if case.name.startswith(prefix) else case.name
        group_prefix = f"stage14_2c_{variant}_"
        group = case.experiment_type.removeprefix(group_prefix)
        metadata[case.case_id] = {
            "profile": profile,
            "algorithm_variant": variant,
            "sweep_group": group,
            "sweep_variable": case.sweep_variable,
            "sweep_value": case.parameter_dict[case.sweep_variable],
        }
    return metadata


def _variant(row: dict[str, Any]) -> str:
    return str(row.get("algorithm_variant") or row["configuration"]["worker_mode"])


def _total_runtime(row: dict[str, Any]) -> float | None:
    timing = row.get("timing") or {}
    value = (timing.get("totals") or {}).get("T_SSE_s")
    return None if value is None else float(value)


def _completed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _statistics_for_path(
    rows: Iterable[dict[str, Any]],
    extractor: Callable[[dict[str, Any]], float | int | None],
) -> dict[str, Any] | None:
    values = [extractor(row) for row in rows]
    return numeric_statistics([float(value) for value in values if value is not None])


def _configuration_summaries(
    rows: list[dict[str, Any]],
    cases: tuple[BenchmarkCase, ...],
) -> list[dict[str, Any]]:
    case_map = {case.case_id: case for case in cases}
    metadata = _case_metadata(cases)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_rows = sorted(grouped.get(case.case_id, []), key=lambda row: row["repetition_id"])
        complete = _completed(case_rows)
        meta = metadata[case.case_id]
        first = complete[0] if complete else None
        counts = None if first is None else first["state_and_game_size"]
        runtime_components: dict[str, Any] = {}
        component_names = sorted({
            name
            for row in complete
            for name in ((row.get("timing") or {}).get("totals") or {})
        })
        for name in component_names:
            runtime_components[name] = _statistics_for_path(
                complete,
                lambda row, key=name: ((row.get("timing") or {}).get("totals") or {}).get(key),
            )
        local_metrics: dict[str, Any] | None = None
        local_outcomes: dict[str, Any] | None = None
        if meta["algorithm_variant"] == "local_sse" and complete:
            local_metrics = {
                name: _statistics_for_path(
                    complete,
                    lambda row, key=name: (row.get("local_search") or {}).get(key),
                )
                for name in (
                    "unique_defender_evaluations", "local_search_iterations",
                    "mean_neighborhood_size", "max_neighborhood_size",
                    "feasible_neighbor_comparisons", "cached_evaluation_reuses",
                    "local_optimality_margin", "search_path_length",
                )
            }
            local_outcomes = {
                "termination_status_counts": dict(sorted(Counter(
                    row["local_search"]["termination_status"] for row in complete
                ).items())),
                "isolated_feasible_local_sse_repetitions": sum(
                    bool(row["local_search"]["isolated_feasible_local_solution"])
                    for row in complete
                ),
                "initial_defender_action_ids": sorted({
                    int(row["local_search"]["initial_defender_action_id"])
                    for row in complete
                }),
                "final_local_sse_action_ids": sorted({
                    int(row["local_search"]["final_local_sse_action_id"])
                    for row in complete
                }),
            }
        identities = [
            {
                "repetition_id": row["repetition_id"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "J_A": row["J_A"],
                "J_D": row["J_D"],
                "trajectory_sha256": (row.get("trajectory_identity") or {}).get("sha256"),
            }
            for row in complete
        ]
        deterministic = bool(
            identities and len({
                (
                    identity["selected_defender_action_id"],
                    identity["selected_attacker_candidate_id"],
                    identity["J_A"], identity["J_D"], identity["trajectory_sha256"],
                )
                for identity in identities
            }) == 1
        )
        summaries.append({
            "case_id": case.case_id,
            **meta,
            "configuration": case.as_configuration(),
            "attempted_repetitions": len(case_rows),
            "successful_repetitions": len(complete),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "runtime_statistics_s": _statistics_for_path(complete, _total_runtime),
            "worker_wall_statistics_s": _statistics_for_path(
                complete, lambda row: row["worker_wall_s"]
            ),
            "peak_rss_statistics_bytes": _statistics_for_path(
                complete, lambda row: row["memory"]["peak_rss_bytes"]
            ),
            "delta_peak_rss_statistics_bytes": _statistics_for_path(
                complete, lambda row: row["memory"]["delta_peak_rss_bytes"]
            ),
            "J_A_statistics": _statistics_for_path(complete, lambda row: row["J_A"]),
            "J_D_statistics": _statistics_for_path(complete, lambda row: row["J_D"]),
            "runtime_component_statistics_s": runtime_components,
            "local_search_statistics": local_metrics,
            "local_search_outcomes": local_outcomes,
            "state_and_game_size": counts,
            "exact_solution_identities": identities,
            "deterministic_solution_identity": deterministic,
            "all_completed_replays_pass": bool(
                complete and all(row["independent_replay_passed"] is True for row in complete)
            ),
            "all_completed_exactness_checks_pass": bool(
                complete and all(all((row.get("exactness") or {}).values()) for row in complete)
            ),
        })
    return summaries


def _oracle_comparisons(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (summary["profile"], summary["algorithm_variant"]): summary
        for summary in summaries
    }
    profiles = sorted({summary["profile"] for summary in summaries})
    comparisons: list[dict[str, Any]] = []
    for profile in profiles:
        local = by_key[(profile, "local_sse")]
        oracle = by_key[(profile, "global_oracle")]
        local_complete = local["successful_repetitions"] == local["attempted_repetitions"] == 3
        oracle_complete = oracle["successful_repetitions"] == oracle["attempted_repetitions"] == 3
        if not local_complete:
            status = "local_baseline_unavailable"
            reason = json.dumps(local["status_counts"], sort_keys=True)
        elif not oracle_complete:
            status = "global_oracle_unavailable"
            reason = json.dumps(oracle["status_counts"], sort_keys=True)
        else:
            status = "completed"
            reason = None
        if status == "completed":
            local_time = float(local["runtime_statistics_s"]["median"])
            global_time = float(oracle["runtime_statistics_s"]["median"])
            local_jd = float(local["J_D_statistics"]["median"])
            global_jd = float(oracle["J_D_statistics"]["median"])
            gap = global_jd - local_jd
            delta_time = global_time - local_time
            ratio = global_time / local_time if local_time > 0.0 else None
        else:
            local_time = global_time = local_jd = global_jd = None
            gap = delta_time = ratio = None
        comparisons.append({
            "profile": profile,
            "sweep_group": local["sweep_group"],
            "sweep_variable": local["sweep_variable"],
            "sweep_value": local["sweep_value"],
            "status": status,
            "unavailable_reason": reason,
            "T_local_median_s": local_time,
            "T_global_median_s": global_time,
            "delta_T_s": delta_time,
            "runtime_ratio_global_over_local": ratio,
            "J_D_local": local_jd,
            "J_D_global": global_jd,
            "delta_J_D": gap,
            "local_defender_action": (
                None if not local["exact_solution_identities"]
                else local["exact_solution_identities"][0]["selected_defender_action_id"]
            ),
            "global_defender_action": (
                None if not oracle["exact_solution_identities"]
                else oracle["exact_solution_identities"][0]["selected_defender_action_id"]
            ),
            "global_oracle_role": "finite exact reference only",
            "gap_interpretation": (
                None if gap is None
                else "empirical gap on this tested finite configuration; no error bound"
            ),
        })
    return comparisons


def _write_raw_csv(
    path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
) -> None:
    fields = (
        "profile", "algorithm_variant", "sweep_group", "case_id", "repetition_id",
        "repetition_index", "sweep_variable", "sweep_value", "status",
        "failure_category", "failure_message", "worker_wall_s", "T_solver_s",
        "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
        "N_S_goal_reachable", "N_S_active", "N_E", "N_C", "N_D", "N_eval",
        "B", "Q", "start_rss_bytes", "peak_rss_bytes", "delta_peak_rss_bytes",
        "J_A", "J_D", "selected_defender_action_id", "selected_attacker_candidate_id",
        "local_iterations", "local_margin", "mean_neighborhood_size",
        "max_neighborhood_size", "cache_reuses", "local_sse_verified",
        "attacker_exact", "strong_tie_verified", "independent_replay_passed",
        "started_at_utc", "ended_at_utc", "environment_manifest_id",
        "configuration_json", "timing_json", "local_search_json", "exactness_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = metadata[row["case_id"]]
            counts = row.get("state_and_game_size") or {}
            local = row.get("local_search") or {}
            exactness = row.get("exactness") or {}
            writer.writerow({
                "profile": meta["profile"],
                "algorithm_variant": _variant(row),
                "sweep_group": meta["sweep_group"],
                "case_id": row["case_id"],
                "repetition_id": row["repetition_id"],
                "repetition_index": row["repetition_index"],
                "sweep_variable": meta["sweep_variable"],
                "sweep_value": meta["sweep_value"],
                "status": row["status"],
                "failure_category": row["failure_category"],
                "failure_message": row["failure_message"],
                "worker_wall_s": row["worker_wall_s"],
                "T_solver_s": _total_runtime(row),
                **{name: counts.get(name) for name in (
                    "N_x", "N_y", "N_h", "N_psi", "N_S_cart", "N_S_admissible",
                    "N_S_goal_reachable", "N_S_active", "N_E", "N_C", "N_D",
                    "N_eval", "B", "Q",
                )},
                "start_rss_bytes": row["memory"]["start_rss_bytes"],
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
                "delta_peak_rss_bytes": row["memory"]["delta_peak_rss_bytes"],
                "J_A": row["J_A"], "J_D": row["J_D"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "local_iterations": local.get("local_search_iterations"),
                "local_margin": local.get("local_optimality_margin"),
                "mean_neighborhood_size": local.get("mean_neighborhood_size"),
                "max_neighborhood_size": local.get("max_neighborhood_size"),
                "cache_reuses": local.get("cached_evaluation_reuses"),
                "local_sse_verified": local.get("local_sse_verified"),
                "attacker_exact": exactness.get("all_evaluated_attacker_responses_exact"),
                "strong_tie_verified": exactness.get("strong_follower_tie_break_verified"),
                "independent_replay_passed": row["independent_replay_passed"],
                "started_at_utc": row["started_at_utc"],
                "ended_at_utc": row["ended_at_utc"],
                "environment_manifest_id": row["environment_manifest_id"],
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
                "timing_json": json.dumps(row.get("timing"), sort_keys=True),
                "local_search_json": json.dumps(row.get("local_search"), sort_keys=True),
                "exactness_json": json.dumps(row.get("exactness"), sort_keys=True),
            })


def _write_table_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list)) else value
                )
                for key, value in record.items()
            })


def _local_rows_for_profiles(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], profiles: set[str],
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["status"] == "completed"
        and _variant(row) == "local_sse"
        and metadata[row["case_id"]]["profile"] in profiles
    ]


def _measured_scaling_figure(
    rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    profiles: set[str],
    x_value: Callable[[dict[str, Any]], float],
    x_title: str,
    title: str,
    filename: str,
    output_directory: Path,
) -> None:
    selected = _local_rows_for_profiles(rows, metadata, profiles)
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[x_value(row) for row in selected],
        y=[_total_runtime(row) for row in selected],
        mode="markers",
        name="raw cold-process repetitions",
        marker={"size": 8, "opacity": 0.55, "color": "#4472C4"},
        text=[metadata[row["case_id"]]["profile"] for row in selected],
        hovertemplate="%{text}<br>x=%{x}<br>T_local=%{y:.3f}s<extra></extra>",
    ))
    grouped: dict[float, list[float]] = {}
    for row in selected:
        grouped.setdefault(x_value(row), []).append(float(_total_runtime(row)))
    ordered = sorted(grouped)
    figure.add_trace(go.Scatter(
        x=ordered, y=[median(grouped[value]) for value in ordered],
        mode="lines+markers", name="median",
        line={"color": "#C00000", "width": 2}, marker={"size": 10},
    ))
    figure.update_layout(
        title=title + "<br><sup>measured points; no asymptotic Big-O claim</sup>",
        xaxis_title=x_title, yaxis_title="T_local [s]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 115},
    )
    save_figure_png(figure, output_directory, filename)


def _evaluated_vs_total_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> None:
    selected = _local_rows_for_profiles(
        rows, metadata,
        {"defender_count_2", "defender_count_3", "defender_count_5"},
    )
    figure = go.Figure(go.Scatter(
        x=[row["state_and_game_size"]["N_D"] for row in selected],
        y=[row["local_search"]["unique_defender_evaluations"] for row in selected],
        mode="markers", name="raw repetitions", marker={"size": 10, "opacity": 0.6},
        text=[metadata[row["case_id"]]["profile"] for row in selected],
        hovertemplate="%{text}<br>N_D=%{x}<br>N_eval_local=%{y}<extra></extra>",
    ))
    maximum = max(row["state_and_game_size"]["N_D"] for row in selected)
    figure.add_trace(go.Scatter(
        x=[0, maximum], y=[0, maximum], mode="lines", name="N_eval = N_D",
        line={"dash": "dash", "color": "#A5A5A5"},
    ))
    figure.update_layout(
        title="Locally evaluated Defender actions versus configured action count",
        xaxis_title="Total N_D", yaxis_title="N_eval_local",
    )
    save_figure_png(figure, output, "05_local_evaluated_vs_total_defenders.png")


def _local_global_runtime_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> None:
    figure = go.Figure()
    for variant, color in (("local_sse", "#4472C4"), ("global_oracle", "#ED7D31")):
        selected = [row for row in rows if row["status"] == "completed" and _variant(row) == variant]
        figure.add_trace(go.Scatter(
            x=[metadata[row["case_id"]]["profile"] for row in selected],
            y=[_total_runtime(row) for row in selected], mode="markers",
            name=f"{variant} raw repetitions", marker={"size": 8, "opacity": 0.55, "color": color},
        ))
    figure.update_layout(
        title="Local-SSE versus exact global-oracle runtime — raw repetitions",
        xaxis={"title": "Matched configuration", "tickangle": -35},
        yaxis_title="Solver wall time [s]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"b": 150, "t": 110},
    )
    save_figure_png(figure, output, "06_local_vs_global_runtime.png")


def _gap_figures(comparisons: list[dict[str, Any]], output: Path) -> None:
    completed = [row for row in comparisons if row["status"] == "completed"]
    gap = go.Figure(go.Bar(
        x=[row["profile"] for row in completed],
        y=[row["delta_J_D"] for row in completed],
        name="measured median gap", marker_color="#70AD47",
    ))
    gap.add_hline(y=0.0, line_dash="dash", line_color="#555555")
    gap.update_layout(
        title="Defender optimality gap on oracle-tractable finite cases",
        xaxis={"title": "Configuration", "tickangle": -35},
        yaxis_title="ΔJ_D = J_D(global) − J_D(local)", margin={"b": 150},
    )
    save_figure_png(gap, output, "07_defender_optimality_gap.png")
    tradeoff = go.Figure(go.Scatter(
        x=[row["delta_J_D"] for row in completed],
        y=[row["runtime_ratio_global_over_local"] for row in completed],
        mode="markers+text", text=[row["profile"] for row in completed],
        textposition="top center", marker={"size": 11, "color": "#7030A0"},
        name="tested finite configurations",
    ))
    tradeoff.update_layout(
        title="Measured runtime ratio versus empirical Defender gap",
        xaxis_title="ΔJ_D", yaxis_title="T_global / T_local",
    )
    save_figure_png(tradeoff, output, "08_speedup_vs_optimality_gap.png")


def _memory_figure(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [row for row in rows if row["status"] == "completed" and _variant(row) == "local_sse"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[row["state_and_game_size"]["N_S_active"] for row in selected],
        y=[row["memory"]["peak_rss_bytes"] / 1024**2 for row in selected],
        mode="markers", name="peak RSS vs active states", marker={"size": 8, "opacity": 0.6},
    ))
    figure.add_trace(go.Scatter(
        x=[row["state_and_game_size"]["N_E"] for row in selected],
        y=[row["memory"]["peak_rss_bytes"] / 1024**2 for row in selected],
        mode="markers", name="peak RSS vs active edges", marker={"size": 8, "opacity": 0.6},
        xaxis="x2",
    ))
    figure.update_layout(
        title="Local solver peak process-tree RSS versus active state/edge count",
        xaxis={"title": "N_S_active"},
        xaxis2={"title": "N_E", "overlaying": "x", "side": "top"},
        yaxis_title="Peak RSS [MiB]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.08},
        margin={"t": 125},
    )
    save_figure_png(figure, output, "09_peak_memory_vs_state_edge_count.png")


def _decomposition_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> None:
    representatives = ("canonical_anchor", "spatial_100m", "defender_count_5")
    keys = (
        "T_graph_s", "T_LOS_s", "T_switch_s", "T_hazard_s", "T_Bellman_s",
        "T_attacker_residual_s", "T_local_search_control_s", "T_validation_s",
        "T_wrapper_residual_s",
    )
    figure = go.Figure()
    for key in keys:
        values = []
        for profile in representatives:
            selected = _local_rows_for_profiles(rows, metadata, {profile})
            component = [
                float(row["timing"]["totals"].get(key, 0.0)) for row in selected
            ]
            values.append(median(component) if component else 0.0)
        figure.add_trace(go.Bar(x=list(representatives), y=values, name=key))
    figure.update_layout(
        barmode="stack", title="Representative local-SSE median runtime decomposition",
        xaxis_title="Configuration", yaxis_title="Component wall time [s]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 135},
    )
    save_figure_png(figure, output, "10_representative_runtime_decomposition.png")


def _generate_figures(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    output: Path,
) -> list[str]:
    baseline = {"canonical_anchor"}
    _measured_scaling_figure(
        rows, metadata, baseline | {"spatial_100m", "spatial_50m"},
        lambda row: float(row["state_and_game_size"]["N_S_active"]),
        "N_S_active", "Local-SSE runtime versus Bellman active-state count",
        "01_local_runtime_vs_state_count.png", output,
    )
    _measured_scaling_figure(
        rows, metadata, baseline | {"heading_15deg", "heading_10deg"},
        lambda row: float(row["state_and_game_size"]["N_psi"]),
        "Heading-bin count N_psi", "Local-SSE runtime versus heading-bin count",
        "02_local_runtime_vs_heading_bins.png", output,
    )
    _measured_scaling_figure(
        rows, metadata, baseline | {"switch_contour_6", "switch_contour_9"},
        lambda row: float(row["state_and_game_size"]["N_C"]),
        "Switching candidate count N_C", "Local-SSE runtime versus switching-candidate count",
        "03_local_runtime_vs_switch_candidates.png", output,
    )
    _measured_scaling_figure(
        rows, metadata, {"defender_count_2", "defender_count_3", "defender_count_5"},
        lambda row: float(row["state_and_game_size"]["N_D"]),
        "Configured Defender action count N_D", "Local-SSE runtime versus Defender action-set size",
        "04_local_runtime_vs_defender_count.png", output,
    )
    _evaluated_vs_total_figure(rows, metadata, output)
    _local_global_runtime_figure(rows, metadata, output)
    _gap_figures(comparisons, output)
    _memory_figure(rows, output)
    _decomposition_figure(rows, metadata, output)
    return [
        "01_local_runtime_vs_state_count.png",
        "02_local_runtime_vs_heading_bins.png",
        "03_local_runtime_vs_switch_candidates.png",
        "04_local_runtime_vs_defender_count.png",
        "05_local_evaluated_vs_total_defenders.png",
        "06_local_vs_global_runtime.png",
        "07_defender_optimality_gap.png",
        "08_speedup_vs_optimality_gap.png",
        "09_peak_memory_vs_state_edge_count.png",
        "10_representative_runtime_decomposition.png",
    ]


def run_stage14_2c_diagnostics(
    output_directory: Path = OUTPUT,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run/resume all declared cold-process repetitions and enforce the exit gate."""
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = stage14_2c_cases()
    audit = configuration_audit(cases)
    if not audit["passed"]:
        raise RuntimeError("Stage-14.2C configuration audit failed")
    environment = environment_manifest(ROOT.parent)
    frozen = _read_json(FROZEN_REFERENCE)
    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "benchmark_configurations.json", audit)
    first = run_benchmark_cases(
        cases,
        output_directory=output_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        force=force,
        worker_path=WORKER,
    )
    second = run_benchmark_cases(
        cases,
        output_directory=output_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        worker_path=WORKER,
    )
    current_case_ids = {case.case_id for case in cases}
    rows = [
        row for row in second["rows"] if row["case_id"] in current_case_ids
    ]
    metadata = _case_metadata(cases)
    summaries = _configuration_summaries(rows, cases)
    comparisons = _oracle_comparisons(summaries)
    local_summaries = [
        summary for summary in summaries
        if summary["algorithm_variant"] == "local_sse"
    ]
    write_json(output_directory / "raw_repetitions.json", rows)
    _write_raw_csv(output_directory / "raw_repetitions.csv", rows, metadata)
    write_json(output_directory / "configuration_summaries.json", summaries)
    _write_table_csv(output_directory / "configuration_summaries.csv", summaries)
    write_json(output_directory / "local_scaling_summaries.json", local_summaries)
    _write_table_csv(output_directory / "local_scaling_summaries.csv", local_summaries)
    write_json(output_directory / "local_vs_global_oracle.json", comparisons)
    _write_table_csv(output_directory / "local_vs_global_oracle.csv", comparisons)
    figures = _generate_figures(rows, comparisons, metadata, output_directory)

    expected_rows = sum(case.repetitions for case in cases)
    local_rows = [row for row in rows if _variant(row) == "local_sse"]
    completed_local = _completed(local_rows)
    completed_oracle = _completed(
        row for row in rows if _variant(row) == "global_oracle"
    )
    successful_comparisons = [row for row in comparisons if row["status"] == "completed"]
    unavailable_comparisons = [
        row for row in comparisons if row["status"] == "global_oracle_unavailable"
    ]
    gap_only_on_success = all(
        (row["status"] == "completed") == (row["delta_J_D"] is not None)
        for row in comparisons
    )
    checks = {
        "same_reproducible_runner_and_resume": (
            len(rows) == expected_rows
            and len({row["repetition_id"] for row in rows}) == expected_rows
            and not second["attempted_repetition_ids"]
            and len(second["skipped_repetition_ids"]) == expected_rows
        ),
        "one_variable_at_a_time": audit["passed"],
        "terrain_fixed_centered_cube": all(
            case.parameter_dict["terrain_category"] == "centered_cube" for case in cases
        ),
        "all_local_repetitions_completed": len(completed_local) == 30,
        "every_local_certificate_verified": bool(
            completed_local and all(
                row["local_search"]["local_sse_verified"]
                and row["exactness"]["all_required_final_neighbors_evaluated"]
                for row in completed_local
            )
        ),
        "every_evaluated_attacker_response_exact": bool(
            completed_local and all(
                row["exactness"]["all_evaluated_attacker_responses_exact"]
                and row["exactness"]["strong_follower_tie_break_verified"]
                for row in completed_local
            )
        ),
        "global_solver_only_used_as_oracle": bool(
            completed_oracle and all(
                row["oracle_metadata"]["role"]
                == "tractable finite global oracle/reference only"
                for row in completed_oracle
            )
        ),
        "gap_only_when_oracle_succeeds": gap_only_on_success,
        "completed_gaps_nonnegative_within_tolerance": bool(
            successful_comparisons and all(
                row["delta_J_D"] >= -NUMERICAL_TOLERANCE
                for row in successful_comparisons
            )
        ),
        "plots_generated_from_stored_raw_data": all(
            (output_directory / name).is_file() for name in figures
        ),
        "no_rl_or_approximate_planner": all(
            not case.parameter_dict["reinforcement_learning"]
            and not case.parameter_dict["approximate_planner"]
            for case in cases
        ),
        "multi_start_deferred": all(
            not case.parameter_dict["multi_start"] for case in cases
        ),
        "global_solver_fingerprint_unchanged": all(
            row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"]
            for row in _completed(rows)
        ),
        "all_completed_replays_pass": all(
            row["independent_replay_passed"] is True for row in _completed(rows)
        ),
    }
    gate_passed = all(checks.values())
    validation = {
        "schema_version": STAGE14_2C_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "expected_repetitions": expected_rows,
        "observed_repetitions": len(rows),
        "local_completed": len(completed_local),
        "global_oracle_completed": len(completed_oracle),
        "successful_oracle_comparisons": len(successful_comparisons),
        "global_oracle_unavailable": unavailable_comparisons,
        "isolated_local_sse_repetitions": sum(
            bool(row["local_search"]["isolated_feasible_local_solution"])
            for row in completed_local
        ),
        "isolated_local_sse_profiles": sorted({
            metadata[row["case_id"]]["profile"] for row in completed_local
            if row["local_search"]["isolated_feasible_local_solution"]
        }),
        "first_invocation_attempted": first["attempted_repetition_ids"],
        "second_invocation_skipped": second["skipped_repetition_ids"],
        "power_law_fit_performed": False,
        "power_law_fit_reason": (
            "three measured configurations per sweep are retained as empirical "
            "points and are insufficient for a robust asymptotic claim"
        ),
        "claim_separation": {
            "local_sse_certification": True,
            "local_sse_computational_scaling": True,
            "global_oracle_gap_only_on_completed_cases": True,
            "bounded_global_approximation_claimed": False,
        },
    }
    write_json(output_directory / "regression_validation_report.json", validation)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.2C exit gate failed: {failed}")
    summary = {
        "stage": "14.2C",
        "schema_version": STAGE14_2C_SCHEMA_VERSION,
        "gate_passed": True,
        "local_case_count": 10,
        "global_oracle_case_count": 10,
        "repetitions_per_case": 3,
        "total_repetition_rows": len(rows),
        "local_completed_repetitions": len(completed_local),
        "global_oracle_completed_repetitions": len(completed_oracle),
        "successful_oracle_comparisons": len(successful_comparisons),
        "global_oracle_unavailable_count": len(unavailable_comparisons),
        "isolated_local_sse_repetition_count": validation[
            "isolated_local_sse_repetitions"
        ],
        "isolated_local_sse_profiles": validation["isolated_local_sse_profiles"],
        "figure_files": figures,
        "gate_checks": checks,
        "interpretation": {
            "primary_object": "single-start local finite-grid SSE",
            "oracle_role": "exact finite global reference on tractable cases only",
            "gap_is_empirical_not_bounded": True,
            "multi_start_performed": False,
            "rl_or_approximation_used": False,
        },
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_2c_summary.json", summary)
    print(
        "Stage 14.2C local scaling/global oracle: PASS; "
        f"rows={len(rows)}, comparisons={len(successful_comparisons)}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_2c_diagnostics()
