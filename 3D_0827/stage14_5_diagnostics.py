"""Stage 14.5 switching-candidate-density sweep, figures, and exit gate."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_5_contract import (
    CONTOUR_SAMPLE_COUNTS,
    RADIAL_SAMPLE_COUNT,
    STAGE14_5_SCHEMA_VERSION,
    configuration_audit,
    stage14_5_cases,
)
from stage14_benchmark_contract import DEFAULT_STAGE14_TOLERANCES, environment_manifest
from stage14_benchmark_runner import BenchmarkCase, numeric_statistics, run_benchmark_cases
from stage14_resolution_figures import build_resolution_figure_set


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_5_switching_candidates"
FROZEN_REFERENCE = ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
WORKER = ROOT / "stage14_5_worker.py"
FUNNEL_KEYS = (
    "N_C_raw", "N_C_unique", "N_C_powered_feasible",
    "N_C_energy_feasible", "N_C_virtual_feasible",
    "N_C_goal_reachable", "N_C_feasible",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {
            "profile": case.name.removeprefix(f"{case.worker_mode}_"),
            "algorithm_variant": case.worker_mode,
            "requested_contour_sample_count": int(
                case.parameter_dict["switching_contour_sample_count"]
            ),
            "contour_spacing": 1.0 / float(
                case.parameter_dict["switching_contour_sample_count"]
            ),
            "fixed_radial_sample_count": int(
                case.parameter_dict["switching_radial_sample_count"]
            ),
            "expected_raw_candidate_count": int(
                case.parameter_dict["switching_contour_sample_count"]
                * case.parameter_dict["switching_radial_sample_count"]
            ),
            "source": "new_stage14_5_isolated_process_rows",
        }
        for case in cases
    }


def _runtime(row: dict[str, Any], key: str) -> float | None:
    value = ((row.get("timing") or {}).get("totals") or {}).get(key)
    return None if value is None else float(value)


def _completed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == "completed"]


def _summaries(
    rows: list[dict[str, Any]],
    cases: tuple[BenchmarkCase, ...],
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["case_id"], []).append(row)
    summaries = []
    for case in cases:
        case_rows = sorted(
            grouped.get(case.case_id, []), key=lambda row: row["repetition_id"]
        )
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
        first_counts = None if not complete else complete[0]["state_and_game_size"]
        funnel_stable = bool(complete and all(
            all(row["state_and_game_size"][key] == first_counts[key] for key in FUNNEL_KEYS)
            for row in complete
        ))
        summaries.append({
            "case_id": case.case_id,
            **metadata[case.case_id],
            "configuration": case.as_configuration(),
            "attempted_repetitions": len(case_rows),
            "successful_repetitions": len(complete),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "runtime_statistics_s": {
                key: numeric_statistics([
                    value for row in complete
                    if (value := _runtime(row, key)) is not None
                ])
                for key in component_names
            },
            "peak_rss_statistics_bytes": numeric_statistics([
                row["memory"]["peak_rss_bytes"] for row in complete
            ]),
            "J_A_statistics": numeric_statistics([row["J_A"] for row in complete]),
            "J_D_statistics": numeric_statistics([row["J_D"] for row in complete]),
            "state_and_game_size": first_counts,
            "candidate_funnel_stable_across_repetitions": funnel_stable,
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


def _write_raw_csv(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]],
) -> None:
    fields = (
        "profile", "algorithm_variant", "source", "requested_contour_sample_count",
        "fixed_radial_sample_count", "case_id", "repetition_id", "status",
        "failure_category", "failure_message", "T_switch_s", "T_Bellman_s",
        "T_attacker_BR_s", "T_SSE_s", "peak_rss_bytes", *FUNNEL_KEYS,
        "J_A", "J_D", "selected_defender_action_id", "selected_attacker_candidate_id",
        "independent_replay_passed", "configuration_json", "timing_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            meta = metadata[row["case_id"]]
            counts = row.get("state_and_game_size") or {}
            writer.writerow({
                "profile": meta["profile"],
                "algorithm_variant": meta["algorithm_variant"],
                "source": meta["source"],
                "requested_contour_sample_count": meta["requested_contour_sample_count"],
                "fixed_radial_sample_count": meta["fixed_radial_sample_count"],
                "case_id": row["case_id"], "repetition_id": row["repetition_id"],
                "status": row["status"], "failure_category": row["failure_category"],
                "failure_message": row["failure_message"],
                "T_switch_s": _runtime(row, "T_switch_s"),
                "T_Bellman_s": _runtime(row, "T_Bellman_s"),
                "T_attacker_BR_s": _runtime(row, "T_attacker_BR_s"),
                "T_SSE_s": _runtime(row, "T_SSE_s"),
                "peak_rss_bytes": row["memory"]["peak_rss_bytes"],
                **{key: counts.get(key) for key in FUNNEL_KEYS},
                "J_A": row["J_A"], "J_D": row["J_D"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "independent_replay_passed": row["independent_replay_passed"],
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
                "timing_json": json.dumps(row.get("timing"), sort_keys=True),
            })


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "profile", "algorithm_variant", "requested_contour_sample_count",
        "fixed_radial_sample_count", "status_counts", "successful_repetitions",
        "T_switch_median_s", "T_Bellman_median_s", "T_attacker_BR_median_s",
        "T_SSE_median_s", "peak_rss_median_bytes", *FUNNEL_KEYS,
        "J_A", "J_D", "deterministic_solution_identity",
        "candidate_funnel_stable_across_repetitions",
        "all_completed_replays_pass", "all_completed_exactness_checks_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            runtime = summary["runtime_statistics_s"]
            counts = summary["state_and_game_size"] or {}
            writer.writerow({
                "profile": summary["profile"],
                "algorithm_variant": summary["algorithm_variant"],
                "requested_contour_sample_count": summary["requested_contour_sample_count"],
                "fixed_radial_sample_count": summary["fixed_radial_sample_count"],
                "status_counts": json.dumps(summary["status_counts"], sort_keys=True),
                "successful_repetitions": summary["successful_repetitions"],
                **{
                    f"{key.removesuffix('_s')}_median_s": (
                        None if runtime.get(key) is None else runtime[key]["median"]
                    )
                    for key in ("T_switch_s", "T_Bellman_s", "T_attacker_BR_s", "T_SSE_s")
                },
                "peak_rss_median_bytes": (
                    None if summary["peak_rss_statistics_bytes"] is None
                    else summary["peak_rss_statistics_bytes"]["median"]
                ),
                **{key: counts.get(key) for key in FUNNEL_KEYS},
                "J_A": None if summary["J_A_statistics"] is None else summary["J_A_statistics"]["median"],
                "J_D": None if summary["J_D_statistics"] is None else summary["J_D_statistics"]["median"],
                "deterministic_solution_identity": summary["deterministic_solution_identity"],
                "candidate_funnel_stable_across_repetitions": summary["candidate_funnel_stable_across_repetitions"],
                "all_completed_replays_pass": summary["all_completed_replays_pass"],
                "all_completed_exactness_checks_pass": summary["all_completed_exactness_checks_pass"],
            })


def _completed_series(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], variant: str,
) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row["status"] == "completed" and metadata[row["case_id"]]["algorithm_variant"] == variant],
        key=lambda row: metadata[row["case_id"]]["expected_raw_candidate_count"],
    )


def _median_points(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]],
    variant: str, extractor,
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for raw_count in sorted({
        meta["expected_raw_candidate_count"] for meta in metadata.values()
    }):
        values = [
            extractor(row) for row in rows
            if row["status"] == "completed"
            and metadata[row["case_id"]]["algorithm_variant"] == variant
            and metadata[row["case_id"]]["expected_raw_candidate_count"] == raw_count
        ]
        values = [float(value) for value in values if value is not None]
        if values:
            stats = numeric_statistics(values)
            xs.append(raw_count)
            ys.append(float(stats["median"]))
    return xs, ys


def _base_layout(title: str, y_title: str) -> dict[str, Any]:
    return {
        "title": title,
        "xaxis_title": "raw switching-candidate count N_C",
        "yaxis_title": y_title,
        "template": "plotly_white",
        "legend": {"x": 1.02, "y": 1.0},
        "margin": {"r": 330},
    }


def _runtime_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> str:
    figure = go.Figure()
    styles = {
        ("local_sse", "T_SSE_s"): ("#4472C4", "solid", "circle", "Local SSE T_SSE"),
        ("local_sse", "T_switch_s"): ("#70AD47", "solid", "square", "Local SSE T_switch"),
        ("global_oracle", "T_SSE_s"): ("#ED7D31", "dash", "diamond", "Global oracle T_SSE"),
        ("global_oracle", "T_switch_s"): ("#A64D79", "dash", "x", "Global oracle T_switch"),
    }
    for (variant, key), (color, dash, marker, label) in styles.items():
        x, y = _median_points(rows, metadata, variant, lambda row, k=key: _runtime(row, k))
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=f"{label} median",
            line={"color": color, "dash": dash, "width": 3},
            marker={"color": color, "symbol": marker, "size": 10},
            yaxis="y2" if key == "T_switch_s" else "y",
        ))
    layout = _base_layout(
        "Switching-density sweep: switching and total SSE runtime",
        "median T_SSE [s]",
    )
    layout["yaxis2"] = {
        "title": "median T_switch [s]",
        "overlaying": "y", "side": "right", "showgrid": False,
    }
    figure.update_layout(**layout)
    name = "01_switch_and_total_sse_runtime.png"
    save_figure_png(figure, output, name)
    return name


def _single_metric_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
    *, key: str, title: str, y_title: str, filename: str,
) -> str:
    figure = go.Figure()
    for variant, color, dash, marker, label in (
        ("local_sse", "#4472C4", "solid", "circle", "Local SSE"),
        ("global_oracle", "#ED7D31", "dash", "diamond", "Global oracle"),
    ):
        x, y = _median_points(rows, metadata, variant, lambda row, k=key: _runtime(row, k))
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=f"{label} median",
            line={"color": color, "dash": dash, "width": 3},
            marker={"color": color, "symbol": marker, "size": 10},
        ))
    figure.update_layout(**_base_layout(title, y_title))
    save_figure_png(figure, output, filename)
    return filename


def _funnel_figure(
    summaries: list[dict[str, Any]], output: Path,
) -> str:
    local = sorted(
        [item for item in summaries if item["algorithm_variant"] == "local_sse"],
        key=lambda item: item["expected_raw_candidate_count"],
    )
    figure = go.Figure()
    styles = (
        ("N_C_raw", "Raw generated", "#4472C4", "circle", "solid"),
        ("N_C_unique", "Unique physical positions", "#5B9BD5", "diamond", "dash"),
        ("N_C_powered_feasible", "Powered feasible", "#70AD47", "square", "solid"),
        ("N_C_energy_feasible", "Energy feasible", "#FFC000", "triangle-up", "dash"),
        ("N_C_virtual_feasible", "Virtual-connection feasible", "#ED7D31", "x", "dot"),
        ("N_C_goal_reachable", "Goal reachable", "#A64D79", "cross", "dashdot"),
        ("N_C_feasible", "Complete mission feasible", "#C00000", "star", "solid"),
    )
    for key, label, color, marker, dash in styles:
        figure.add_trace(go.Scatter(
            x=[item["expected_raw_candidate_count"] for item in local],
            y=[item["state_and_game_size"][key] for item in local],
            mode="lines+markers", name=label,
            line={"color": color, "dash": dash, "width": 2.5},
            marker={"color": color, "symbol": marker, "size": 9},
        ))
    figure.update_layout(**_base_layout(
        "Switching-density sweep: Local-SSE candidate filter funnel",
        "candidate count",
    ))
    name = "03_candidate_filter_funnel.png"
    save_figure_png(figure, output, name)
    return name


def _memory_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> str:
    figure = go.Figure()
    for variant, color, dash, marker, label in (
        ("local_sse", "#4472C4", "solid", "circle", "Local SSE"),
        ("global_oracle", "#ED7D31", "dash", "diamond", "Global oracle"),
    ):
        x, y = _median_points(
            rows, metadata, variant,
            lambda row: float(row["memory"]["peak_rss_bytes"]) / 1024.0**2,
        )
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=f"{label} median peak RSS",
            line={"color": color, "dash": dash, "width": 3},
            marker={"color": color, "symbol": marker, "size": 10},
        ))
    figure.update_layout(**_base_layout(
        "Switching-density sweep: peak process-tree memory", "peak RSS [MiB]",
    ))
    name = "04_peak_memory.png"
    save_figure_png(figure, output, name)
    return name


def _objective_figure(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], output: Path,
) -> str:
    figure = go.Figure()
    styles = {
        ("local_sse", "J_A"): ("#4472C4", "solid", "circle", "Local SSE Attacker J_A"),
        ("local_sse", "J_D"): ("#70AD47", "solid", "square", "Local SSE Defender J_D"),
        ("global_oracle", "J_A"): ("#ED7D31", "dash", "diamond", "Global oracle Attacker J_A"),
        ("global_oracle", "J_D"): ("#A64D79", "dash", "x", "Global oracle Defender J_D"),
    }
    for (variant, key), (color, dash, marker, label) in styles.items():
        x, y = _median_points(rows, metadata, variant, lambda row, k=key: row.get(k))
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=label,
            line={"color": color, "dash": dash, "width": 3},
            marker={"color": color, "symbol": marker, "size": 10},
        ))
    figure.update_layout(**_base_layout(
        "Switching-density sweep: objective convergence diagnostic",
        "median objective value",
    ))
    name = "05_objectives.png"
    save_figure_png(figure, output, name)
    return name


def _figures(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]], output: Path,
) -> list[str]:
    del summaries
    return build_resolution_figure_set(
        rows,
        metadata,
        x_key="expected_raw_candidate_count",
        x_title="Sampled candidates [count, before feasibility filtering]",
        sweep_label="Switching-contour sweep",
        output=output,
        performance_local_only=True,
        focused_runtime_display=True,
    )


def regenerate_stage14_5_figures_from_saved_data(
    output_directory: Path = OUTPUT,
) -> list[str]:
    """Re-render the common Stage-14.5 report without benchmark workers."""
    rows = _read_json(output_directory / "raw_candidate_repetitions.json")
    metadata = _metadata(stage14_5_cases())
    figures = _figures(rows, [], metadata, output_directory)
    summary_path = output_directory / "stage14_5_summary.json"
    summary = _read_json(summary_path)
    summary["figure_files"] = figures
    summary["figure_methodology"] = {
        "controlled_x_axis": "sampled candidates before feasibility filtering",
        "candidate_count_definition": "contour ray samples times radial samples (fixed at 8)",
        "performance_displayed_algorithms": ["local_sse"],
        "performance_legend_algorithm_prefix": False,
        "other_solver_control_displayed": False,
        "shared_performance_figure_schema": True,
        "candidate_filter_counts_location": "raw/tabular audit artifacts only",
        "primary_algorithm": "local_sse",
        "reference_algorithm": "global_oracle",
    }
    write_json(summary_path, summary)
    return figures


def run_stage14_5_diagnostics(
    output_directory: Path = OUTPUT, *, force: bool = False,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = stage14_5_cases()
    audit = configuration_audit(cases)
    if not audit["passed"]:
        raise RuntimeError("Stage-14.5 configuration audit failed")
    metadata = _metadata(cases)
    environment = environment_manifest(ROOT.parent)
    measurement_directory = output_directory / "measurements"
    first = run_benchmark_cases(
        cases, output_directory=measurement_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        force=force, worker_path=WORKER,
    )
    second = run_benchmark_cases(
        cases, output_directory=measurement_directory,
        environment_manifest_id=environment["environment_manifest_id"],
        worker_path=WORKER,
    )
    case_ids = {case.case_id for case in cases}
    rows = sorted(
        [row for row in second["rows"] if row["case_id"] in case_ids],
        key=lambda row: (row["case_id"], row["repetition_index"]),
    )
    for row in rows:
        row["algorithm_variant"] = row.get("algorithm_variant") or row["configuration"]["worker_mode"]
        row["stage14_5_source"] = metadata[row["case_id"]]["source"]
        row["process_semantics"] = (
            "fresh isolated Python worker; OS and hardware cache state uncontrolled"
        )
    summaries = _summaries(rows, cases, metadata)
    figures = _figures(rows, summaries, metadata, output_directory)

    write_json(output_directory / "environment_manifest.json", environment)
    write_json(output_directory / "candidate_sweep_configuration.json", audit)
    write_json(output_directory / "runner_resume_report.json", {
        "first_invocation_attempted": first["attempted_repetition_ids"],
        "first_invocation_skipped": first["skipped_repetition_ids"],
        "second_invocation_attempted": second["attempted_repetition_ids"],
        "second_invocation_skipped": second["skipped_repetition_ids"],
    })
    write_json(output_directory / "raw_candidate_repetitions.json", rows)
    _write_jsonl(output_directory / "raw_candidate_repetitions.jsonl", rows)
    _write_raw_csv(output_directory / "raw_candidate_repetitions.csv", rows, metadata)
    write_json(output_directory / "candidate_sweep_summaries.json", summaries)
    _write_summary_csv(output_directory / "candidate_sweep_summaries.csv", summaries)

    complete = _completed(rows)
    frozen = _read_json(FROZEN_REFERENCE)
    expected = frozen["solution_identity"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    baseline = [
        row for row in complete
        if metadata[row["case_id"]]["requested_contour_sample_count"] == 12
    ]
    baseline_regression = bool(baseline and all(
        row["selected_defender_action_id"] == expected["selected_defender_action_id"]
        and row["selected_attacker_candidate_id"] == expected["selected_attacker_candidate_id"]
        and abs(float(row["J_A"]) - float(expected["attacker_objective"])) <= tolerance.attacker_objective_abs
        and abs(float(row["J_D"]) - float(expected["defender_objective_pod"])) <= tolerance.defender_objective_abs
        for row in baseline
    ))
    def funnel_valid(row: dict[str, Any]) -> bool:
        counts = row.get("state_and_game_size") or {}
        if any(key not in counts for key in FUNNEL_KEYS):
            return False
        chain = [
            counts["N_C_raw"], counts["N_C_powered_feasible"],
            counts["N_C_energy_feasible"], counts["N_C_virtual_feasible"],
            counts["N_C_goal_reachable"], counts["N_C_feasible"],
        ]
        return (
            counts["N_C_unique"] <= counts["N_C_raw"]
            and all(left >= right >= 0 for left, right in zip(chain, chain[1:]))
        )
    local_summaries = [item for item in summaries if item["algorithm_variant"] == "local_sse"]
    bellman_medians = {
        item["state_and_game_size"]["N_C_raw"]:
        item["runtime_statistics_s"]["T_Bellman_s"]["median"]
        for item in local_summaries
    }
    bellman_min = min(bellman_medians.values())
    bellman_max = max(bellman_medians.values())
    bellman_relative_span = (
        None if bellman_min <= 0.0 else (bellman_max - bellman_min) / bellman_min
    )
    checks = {
        "only_switching_candidate_density_varied": audit["checks"]["only_switching_candidate_density_varies"],
        "monotonic_requested_densities_recorded": audit["checks"]["monotonically_increasing_densities_present"],
        "three_repetitions_per_configuration_recorded": len(rows) == len(cases) * 3,
        "all_measurements_completed": len(complete) == len(rows),
        "runner_resume_deterministic": (
            not second["attempted_repetition_ids"]
            and len(second["skipped_repetition_ids"]) == len(cases) * 3
        ),
        "raw_and_filtered_candidate_counts_recorded": all(funnel_valid(row) for row in complete),
        "raw_counts_match_requested_density": all(
            row["state_and_game_size"]["N_C_raw"]
            == metadata[row["case_id"]]["expected_raw_candidate_count"]
            for row in complete
        ),
        "candidate_funnel_deterministic_per_configuration": all(
            item["candidate_funnel_stable_across_repetitions"] for item in summaries
        ),
        "every_completed_replay_passes": bool(
            complete and all(row["independent_replay_passed"] is True for row in complete)
        ),
        "exhaustive_candidate_minimum_and_ties_verified": bool(
            complete and all(
                (row.get("exactness") or {}).get("exhaustive_candidate_minimum_verified") is True
                and (row.get("exactness") or {}).get("strong_follower_tie_break_verified") is True
                for row in complete
            )
        ),
        "every_completed_exactness_check_passes": bool(
            complete and all(all((row.get("exactness") or {}).values()) for row in complete)
        ),
        "every_local_result_is_certified": all(
            (row.get("local_search") or {}).get("local_sse_verified") is True
            for row in complete if row["algorithm_variant"] == "local_sse"
        ),
        "deterministic_solution_identity_per_configuration": all(
            item["deterministic_solution_identity"] for item in summaries
        ),
        "canonical_frozen_regression_passes": baseline_regression,
        "solver_source_fingerprint_unchanged": all(
            row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"]
            for row in complete
        ),
        "global_solver_is_oracle_only": all(
            row["algorithm_variant"] != "global_oracle"
            or (row.get("oracle_metadata") or {}).get("role")
            == "tractable finite global oracle/reference only"
            for row in complete
        ),
        "png_figures_exist": all((output_directory / name).is_file() for name in figures),
    }
    gate_passed = all(checks.values())
    validation = {
        "schema_version": STAGE14_5_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "candidate_funnel_keys": list(FUNNEL_KEYS),
        "candidate_funnel_semantics": (
            "cumulative solver survivors; N_C_unique is a diagnostic physical-position count "
            "and is not a solver deduplication stage"
        ),
        "bellman_runtime_architecture_measurement": {
            "local_median_T_Bellman_s_by_raw_N_C": bellman_medians,
            "relative_span_over_minimum": bellman_relative_span,
            "interpretation": (
                "measured shared-Bellman behavior only; no monotonic trend is imposed"
            ),
        },
        "objective_convergence_established": False,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_5_validation_report.json", validation)
    summary = {
        "stage": "14.5",
        "schema_version": STAGE14_5_SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "contour_sample_counts": list(CONTOUR_SAMPLE_COUNTS),
        "fixed_radial_sample_count": RADIAL_SAMPLE_COUNT,
        "raw_candidate_counts": [count * RADIAL_SAMPLE_COUNT for count in CONTOUR_SAMPLE_COUNTS],
        "total_repetition_count": len(rows),
        "figure_files": figures,
        "bellman_runtime_architecture_measurement": validation["bellman_runtime_architecture_measurement"],
        "gate_checks": checks,
        "objective_convergence_established": False,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_5_summary.json", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage-14.5 exit gate failed: {failed}")
    print(
        "Stage 14.5 switching-candidate-density sweep: PASS; "
        f"rows={len(rows)}, raw N_C={summary['raw_candidate_counts']}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_5_diagnostics()
