"""Canonical Stage 14.2B local-SSE execution, certification, and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_benchmark_contract import (
    DEFAULT_STAGE14_TOLERANCES,
    environment_manifest,
)
from stage14_benchmark_runner import run_limited_subprocess


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_2b_local_sse_search"
FROZEN_STAGE14 = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_canonical_worker(output_path: Path) -> dict[str, Any]:
    process = run_limited_subprocess(
        (
            sys.executable,
            str(ROOT / "stage14_2b_worker.py"),
            "--output",
            str(output_path),
        ),
        cwd=ROOT,
        timeout_s=1800.0,
        memory_limit_bytes=4 * 1024**3,
        sampling_interval_s=0.02,
    )
    process_report = {
        "returncode": process.returncode,
        "limit_status": process.limit_status,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "wall_s": process.wall_s,
        "start_rss_bytes": process.start_rss_bytes,
        "peak_rss_bytes": process.peak_rss_bytes,
        "delta_peak_rss_bytes": process.delta_peak_rss_bytes,
        "sampling_interval_s": process.sampling_interval_s,
        "memory_semantics": "peak RSS of fresh worker process tree",
    }
    if process.limit_status is not None:
        raise RuntimeError(f"canonical worker hit {process.limit_status}")
    if process.returncode != 0:
        raise RuntimeError(
            "canonical worker failed: " + (process.stderr.strip() or "unknown failure")
        )
    if not output_path.exists():
        raise RuntimeError("canonical worker returned without its JSON result")
    return process_report


def _diagnostic_figure(payload: dict[str, Any]) -> go.Figure:
    records = payload["exact_evaluation_records"]
    search = payload["search_result"]
    value_by_id = {
        int(record["action_id"]): float(record["local_evaluation"]["defender_value"])
        for record in records
        if record["local_evaluation"]["status"] == "feasible"
    }
    x_by_id = {
        int(record["action_id"]): float(record["sensor_position_map"][0])
        for record in records
    }
    evaluated = [int(value) for value in search["evaluated_defender_actions"]]
    visited = [int(value) for value in search["visited_defender_actions"]]
    final_id = int(search["final_local_sse_action_id"])
    final_iteration = search["iterations"][-1]
    final_neighbors = [int(value) for value in final_iteration["neighbor_action_ids"]]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[x_by_id[action_id] for action_id in evaluated],
        y=[value_by_id[action_id] for action_id in evaluated],
        mode="markers+text",
        text=[f"D{action_id}" for action_id in evaluated],
        textposition="top center",
        name="Actually evaluated",
        marker={"size": 11, "color": "#4472C4"},
        customdata=evaluated,
        hovertemplate="D%{customdata}<br>x=%{x:.2f}<br>V_D=%{y:.9f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[x_by_id[action_id] for action_id in visited],
        y=[value_by_id[action_id] for action_id in visited],
        mode="lines+markers",
        name="Visited sequence",
        line={"color": "#ED7D31", "width": 3},
        marker={"size": 8},
        customdata=list(range(len(visited))),
        hovertemplate="visit %{customdata}<br>x=%{x:.2f}<br>V_D=%{y:.9f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[x_by_id[action_id] for action_id in final_neighbors],
        y=[value_by_id[action_id] for action_id in final_neighbors],
        mode="markers",
        name="Final required neighborhood",
        marker={"size": 18, "symbol": "circle-open", "color": "#A5A5A5"},
        customdata=final_neighbors,
        hovertemplate="neighbor D%{customdata}<br>x=%{x:.2f}<br>V_D=%{y:.9f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[x_by_id[final_id]], y=[value_by_id[final_id]], mode="markers",
        name="Certified local SSE",
        marker={"size": 20, "symbol": "star", "color": "#70AD47"},
        customdata=[final_id],
        hovertemplate="local SSE D%{customdata}<br>x=%{x:.2f}<br>V_D=%{y:.9f}<extra></extra>",
    ))
    figure.update_layout(
        title=(
            "Stage 14.2B canonical local-SSE search<br>"
            "<sup>Only exact-evaluated Defender payoffs are displayed; no global claim</sup>"
        ),
        xaxis={"title": "Defender sensor x position [map units]"},
        yaxis={"title": "Induced Defender value V_D (PoD)", "range": [0.0, 1.0]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 120},
    )
    return figure


def _validate(payload: dict[str, Any], frozen: dict[str, Any]) -> dict[str, bool]:
    search = payload["search_result"]
    records = payload["exact_evaluation_records"]
    iterations = search["iterations"]
    final_iteration = iterations[-1]
    evaluated = {int(value) for value in search["evaluated_defender_actions"]}
    required_final = {int(value) for value in final_iteration["neighbor_action_ids"]}
    identity = frozen["solution_identity"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    current_environment = environment_manifest(ROOT.parent)
    final_identity = payload["final_trajectory_identity"]
    return {
        "canonical_centered_cube_certified": (
            payload["configuration"]["terrain_category"] == "centered_cube"
            and search["termination_status"] == "certified_local_sse"
            and bool(search["local_sse_verified"])
        ),
        "all_evaluated_actions_use_exact_attacker_br": (
            bool(search["attacker_exactness_verified"])
            and all(record["exact_exhaustive_minimum_verified"] for record in records)
        ),
        "strong_follower_tie_preserved": (
            bool(search["strong_tie_break_verified"])
            and all(record["strong_follower_tie_audit_passed"] for record in records)
        ),
        "all_final_required_neighbors_evaluated": required_final <= evaluated,
        "final_neighbor_comparisons_known": (
            not final_iteration["unknown_neighbor_ids"]
            and not final_iteration["local_verification"]["unknown_neighbor_diagnostics"]
        ),
        "local_margin_satisfies_tolerance": (
            float(search["local_optimality_margin"]) >= -1.0e-12
        ),
        "independent_replay_passed": bool(search["independent_replay_status"]),
        "not_mislabeled_global": (
            search["global_optimality_evaluated"] is False
            and search["global_optimal"] is None
            and "not a global" in search["solution_scope"]
        ),
        "frozen_canonical_action_matches_as_regression_only": (
            int(search["final_local_sse_action_id"])
            == int(identity["selected_defender_action_id"])
            and int(search["final_attacker_response_id"])
            == int(identity["selected_attacker_candidate_id"])
        ),
        "frozen_canonical_payoffs_within_tolerance": (
            abs(float(search["final_J_A"]) - float(identity["attacker_objective"]))
            <= tolerance.attacker_objective_abs
            and abs(float(search["final_J_D"]) - float(identity["defender_objective_pod"]))
            <= tolerance.defender_objective_abs
        ),
        "frozen_trajectory_identity_matches": (
            final_identity["sha256"] == identity["trajectory_identity"]["sha256"]
        ),
        "global_solver_source_unchanged": (
            current_environment["software_revision"]["solver_source"]["aggregate_sha256"]
            == frozen["solver_source_fingerprint"]
        ),
    }


def run_stage14_2b_diagnostics(
    output_directory: Path = OUTPUT,
    *,
    recompute: bool = True,
) -> dict[str, Any]:
    """Run or inspect the canonical local search and emit the 14.2B gate artifacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    worker_path = output_directory / "canonical_worker_result.json"
    process_report: dict[str, Any] | None = None
    if recompute or not worker_path.exists():
        process_report = _run_canonical_worker(worker_path)
    else:
        runtime_path = output_directory / "runtime_memory_summary.json"
        if runtime_path.exists():
            saved_process = _read_json(runtime_path).get("process_measurement")
            if isinstance(saved_process, dict):
                process_report = saved_process
    payload = _read_json(worker_path)
    frozen = _read_json(FROZEN_STAGE14)
    search = payload["search_result"]
    if process_report is None:
        process_report = {
            "status": "reused_existing_worker_artifact",
            "T_local_s": search["runtime_decomposition_s"]["T_local_s"],
            "peak_rss_bytes": None,
            "delta_peak_rss_bytes": None,
        }
    search["peak_memory"] = {
        "peak_rss_bytes": process_report.get("peak_rss_bytes"),
        "delta_peak_rss_bytes": process_report.get("delta_peak_rss_bytes"),
        "semantics": process_report.get(
            "memory_semantics", "not measured while reusing existing artifact"
        ),
    }
    gates = _validate(payload, frozen)
    gate_passed = all(gates.values())
    if not gate_passed:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Stage-14.2B canonical gate failed: {failed}")

    write_json(output_directory / "canonical_local_sse_result.json", search)
    write_json(output_directory / "local_search_history.json", {
        "visited_defender_actions": search["visited_defender_actions"],
        "evaluated_defender_actions": search["evaluated_defender_actions"],
        "iterations": search["iterations"],
        "exact_evaluation_records": payload["exact_evaluation_records"],
    })
    write_json(output_directory / "runtime_memory_summary.json", {
        "runtime_decomposition_s": search["runtime_decomposition_s"],
        "process_measurement": process_report,
        "state_and_search_size": payload["state_and_search_size"],
    })
    save_figure_png(
        _diagnostic_figure(payload), output_directory,
        "local_search_diagnostic.png",
    )
    summary = {
        "stage": "14.2B",
        "gate_passed": True,
        "termination_status": search["termination_status"],
        "solution_scope": search["solution_scope"],
        "initial_defender_action_id": search["initial_defender_action_id"],
        "final_local_sse_action_id": search["final_local_sse_action_id"],
        "final_attacker_response_id": search["final_attacker_response_id"],
        "final_J_A": search["final_J_A"],
        "final_J_D": search["final_J_D"],
        "visited_defender_actions": search["visited_defender_actions"],
        "evaluated_defender_actions": search["evaluated_defender_actions"],
        "unique_defender_evaluations": search["unique_defender_evaluations"],
        "local_search_iterations": search["local_search_iterations"],
        "local_optimality_margin": search["local_optimality_margin"],
        "runtime_decomposition_s": search["runtime_decomposition_s"],
        "peak_memory": search["peak_memory"],
        "gate_checks": gates,
        "global_optimality_evaluated": False,
        "global_solver_modified": False,
        "next_stage_started": False,
    }
    write_json(output_directory / "stage14_2b_summary.json", summary)
    print(
        "Stage 14.2B canonical local SSE: PASS; "
        f"D{search['initial_defender_action_id']} -> "
        f"D{search['final_local_sse_action_id']}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_2b_diagnostics()
