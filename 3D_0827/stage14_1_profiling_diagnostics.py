"""Execute the Stage-14.1 canonical timing and process-tree RSS gate."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go

from stage11_notebook_support import save_figure_png, write_json
from stage14_benchmark_contract import DEFAULT_STAGE14_TOLERANCES
from stage14_profiling import (
    COUPLING_NOTES,
    PROFILE_SCHEMA_VERSION,
    TIMING_DEFINITIONS,
    run_profiled_subprocess,
    subprocess_result_dict,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "figure" / "stage_14_1_runtime_profiling"
STAGE14_0_REFERENCE = (
    ROOT / "figure" / "stage_14_0_benchmark_contract" / "frozen_reference.json"
)
WORKER = ROOT / "stage14_1_profile_worker.py"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return payload


def _run_worker(case: str, raw_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = run_profiled_subprocess(
        (sys.executable, str(WORKER), "--case", case, "--output", str(raw_output)),
        cwd=ROOT,
        sampling_interval_s=0.01,
    )
    monitor = subprocess_result_dict(result)
    if result.returncode != 0:
        raise RuntimeError(
            f"Stage-14.1 {case} worker failed with {result.returncode}: {result.stderr}"
        )
    payload = _read_json(raw_output)
    payload["process_tree_memory"] = monitor["memory"]
    payload["worker_monitor_wall_s"] = monitor["monitor_wall_s"]
    return payload, monitor


def _identity_regression(
    profiled: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    current = profiled["solution_identity"]
    expected = frozen["solution_identity"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    checks = {
        "selected_defender": current["selected_defender_action_id"] == expected["selected_defender_action_id"],
        "selected_attacker": current["selected_attacker_candidate_id"] == expected["selected_attacker_candidate_id"],
        "attacker_cooptimal_set": current["attacker_objective_cooptimal_candidate_ids"] == expected["attacker_objective_cooptimal_candidate_ids"],
        "leader_cooptimal_set": current["leader_cooptimal_action_ids"] == expected["leader_cooptimal_action_ids"],
        "sensor_position": bool(np.allclose(
            current["selected_sensor_position_map"], expected["selected_sensor_position_map"],
            rtol=0.0, atol=tolerance.sensor_position_map_abs,
        )),
        "attacker_objective": bool(np.isclose(
            current["attacker_objective"], expected["attacker_objective"],
            rtol=0.0, atol=tolerance.attacker_objective_abs,
        )),
        "defender_objective": bool(np.isclose(
            current["defender_objective_pod"], expected["defender_objective_pod"],
            rtol=0.0, atol=tolerance.defender_objective_abs,
        )),
        "mission_time": bool(np.isclose(
            current["mission_time_s"], expected["mission_time_s"],
            rtol=0.0, atol=tolerance.mission_time_s_abs,
        )),
        "cumulative_hazard": bool(np.isclose(
            current["cumulative_hazard"], expected["cumulative_hazard"],
            rtol=0.0, atol=tolerance.cumulative_hazard_abs,
        )),
        "detection_probability": bool(np.isclose(
            current["detection_probability"], expected["detection_probability"],
            rtol=0.0, atol=tolerance.detection_probability_abs,
        )),
        "trajectory_identity": current["trajectory_identity"]["sha256"] == expected["trajectory_identity"]["sha256"],
        "state_and_game_size": profiled["state_and_game_size"] == frozen["state_and_game_size"],
        "solver_source_fingerprint": profiled["solver_source_fingerprint"] == frozen["solver_source_fingerprint"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _runtime_figure(profile: dict[str, Any]) -> go.Figure:
    totals = profile["timing"]["totals"]
    components = (
        ("Graph", "T_graph_s", "#4c78a8"),
        ("LOS", "T_LOS_s", "#f58518"),
        ("Switch", "T_switch_s", "#eeca3b"),
        ("Hazard", "T_hazard_s", "#54a24b"),
        ("Bellman", "T_Bellman_s", "#b279a2"),
        ("Attacker residual", "T_attacker_unclassified_s", "#ff9da6"),
        ("Defender overhead", "T_defender_overhead_s", "#9d755d"),
    )
    figure = go.Figure()
    for label, key, color in components:
        figure.add_trace(go.Bar(
            x=["Canonical exact SSE"], y=[totals[key]], name=label,
            marker_color=color,
            hovertemplate=f"{label}: %{{y:.6f}} s<extra></extra>",
        ))
    figure.add_trace(go.Scatter(
        x=["Canonical exact SSE"], y=[totals["T_SSE_s"]],
        mode="markers", name="Measured T_SSE",
        marker={"symbol": "line-ew", "size": 24, "color": "black", "line": {"width": 3}},
        hovertemplate="T_SSE=%{y:.6f} s<extra></extra>",
    ))
    figure.update_layout(
        title=(
            "Stage 14.1 Canonical Exact-SSE Runtime Decomposition"
            f"<br><sup>Independent replay: {totals['T_validation_s']:.6f} s "
            "(measured separately, not included in T_SSE)</sup>"
        ),
        barmode="stack", template="plotly_white", height=620, width=1050,
        yaxis_title="wall time [s]",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.18},
        margin={"l": 80, "r": 40, "t": 100, "b": 130},
    )
    return figure


def _memory_figure(profile: dict[str, Any]) -> go.Figure:
    memory = profile["process_tree_memory"]
    labels = ("Worker start RSS", "Worker peak RSS", "Peak increase")
    bytes_values = (
        memory["start_rss_bytes"], memory["peak_rss_bytes"],
        memory["delta_peak_rss_bytes"],
    )
    mib = [value / 1024.0**2 for value in bytes_values]
    figure = go.Figure(go.Bar(
        x=list(labels), y=mib,
        marker_color=["#4c78a8", "#e45756", "#72b7b2"],
        text=[f"{value:.1f} MiB" for value in mib], textposition="outside",
        customdata=np.asarray(bytes_values)[:, None],
        hovertemplate="%{x}<br>%{y:.3f} MiB<br>%{customdata[0]:,} bytes<extra></extra>",
    ))
    figure.update_layout(
        title=(
            "Stage 14.1 Canonical Worker Process-Tree Memory"
            "<br><sup>10 ms sampled RSS; NumPy/native allocations and child processes included</sup>"
        ),
        template="plotly_white", height=600, width=950,
        yaxis_title="resident memory [MiB]", showlegend=False,
        margin={"l": 80, "r": 40, "t": 100, "b": 80},
    )
    return figure


def _write_timing_csv(path: Path, profile: dict[str, Any]) -> None:
    totals = profile["timing"]["totals"]
    sizes = profile["state_and_game_size"]
    identity = profile["solution_identity"]
    memory = profile["process_tree_memory"]
    row = {
        "schema_version": profile["schema_version"],
        "case_id": profile["case_id"],
        "status": profile["status"],
        **totals,
        "start_rss_bytes": memory["start_rss_bytes"],
        "peak_rss_bytes": memory["peak_rss_bytes"],
        "delta_peak_rss_bytes": memory["delta_peak_rss_bytes"],
        "N_S_cart": sizes["N_S_cart"],
        "N_S_admissible": sizes["N_S_admissible"],
        "N_S_goal_reachable": sizes["N_S_goal_reachable"],
        "N_S_active": sizes["N_S_active"],
        "N_E": sizes["N_E"],
        "N_C_raw": sizes["N_C_raw"],
        "N_D": sizes["N_D"],
        "B": sizes["B"],
        "Q": sizes["Q"],
        "selected_defender_action_id": identity["selected_defender_action_id"],
        "selected_attacker_candidate_id": identity["selected_attacker_candidate_id"],
        "attacker_objective": identity["attacker_objective"],
        "defender_objective_pod": identity["defender_objective_pod"],
        "validation_passed": profile["independent_replay"]["passed"],
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def run_stage14_1_diagnostics(
    output_directory: Path = OUTPUT,
) -> dict[str, Any]:
    """Profile canonical and infeasible fresh workers, validate, and persist."""
    output_directory.mkdir(parents=True, exist_ok=True)
    frozen = _read_json(STAGE14_0_REFERENCE)
    canonical_raw = output_directory / "canonical_worker_raw.json"
    infeasible_raw = output_directory / "infeasible_worker_raw.json"
    canonical, canonical_monitor = _run_worker("canonical", canonical_raw)
    infeasible, infeasible_monitor = _run_worker("infeasible", infeasible_raw)
    regression = _identity_regression(canonical, frozen)
    if not regression["passed"]:
        raise RuntimeError("profiled and unprofiled exact SSE identities differ")
    timing = canonical["timing"]
    totals = timing["totals"]
    memory = canonical["process_tree_memory"]
    timing_fields_valid = all(
        np.isfinite(value) and value >= 0.0 for value in totals.values()
    )
    accounting_valid = abs(timing["T_SSE_reconciliation_error_s"]) <= 1.0e-9
    memory_valid = (
        memory["start_rss_bytes"] > 0
        and memory["peak_rss_bytes"] >= memory["start_rss_bytes"]
        and memory["delta_peak_rss_bytes"]
        == memory["peak_rss_bytes"] - memory["start_rss_bytes"]
        and memory["native_allocations_included"]
        and memory["child_processes_included"]
    )
    infeasible_valid = bool(
        infeasible["status"] == "model_infeasible"
        and infeasible["terrain_category"] == "centered_cube"
        and infeasible["T_SSE_attempt_s"] >= 0.0
        and not infeasible["component_timing_available"]
    )
    gate_passed = bool(
        timing_fields_valid
        and accounting_valid
        and memory_valid
        and regression["passed"]
        and canonical["independent_replay"]["passed"]
        and infeasible_valid
    )
    if not gate_passed:
        raise RuntimeError("Stage-14.1 profiling gate failed")

    canonical["profiled_vs_unprofiled_regression"] = regression
    canonical["instrumentation_gate"] = {
        "passed": True,
        "timing_fields_valid": timing_fields_valid,
        "accounting_valid": accounting_valid,
        "memory_valid": memory_valid,
        "infeasible_fixture_valid": infeasible_valid,
    }
    write_json(output_directory / "canonical_profile.json", canonical)
    write_json(output_directory / "infeasible_profile.json", infeasible)
    write_json(output_directory / "canonical_worker_monitor.json", canonical_monitor)
    write_json(output_directory / "infeasible_worker_monitor.json", infeasible_monitor)
    write_json(output_directory / "profiling_contract.json", {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "timing_definitions": TIMING_DEFINITIONS,
        "coupling_notes": COUPLING_NOTES,
        "memory_definition": {
            "primary_metric": "sampled peak worker process-tree RSS",
            "sampling_interval_s": memory["sampling_interval_s"],
            "includes_numpy_native_allocations": True,
            "includes_child_processes": True,
            "tracemalloc_is_primary": False,
        },
    })
    _write_timing_csv(output_directory / "canonical_timing.csv", canonical)
    save_figure_png(
        _runtime_figure(canonical), output_directory,
        "canonical_runtime_decomposition.png",
    )
    save_figure_png(
        _memory_figure(canonical), output_directory,
        "canonical_memory_summary.png",
    )
    summary = {
        "stage": "14.1",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "gate_passed": True,
        "canonical_status": canonical["status"],
        "infeasible_fixture_status": infeasible["status"],
        "profiled_vs_unprofiled_match": regression["passed"],
        "independent_replay_passed": canonical["independent_replay"]["passed"],
        "timing": timing,
        "process_tree_memory": memory,
        "solution_identity": canonical["solution_identity"],
        "state_and_game_size": canonical["state_and_game_size"],
        "instrumentation_gate": canonical["instrumentation_gate"],
        "scaling_claim_made": False,
        "sweep_runner_added": False,
    }
    write_json(output_directory / "stage14_1_summary.json", summary)
    print("Stage 14.1 reliable runtime/memory instrumentation: PASS")
    print(
        f"T_SSE={totals['T_SSE_s']:.6f} s; "
        f"peak RSS={memory['peak_rss_bytes'] / 1024.0**2:.1f} MiB; "
        f"delta={memory['delta_peak_rss_bytes'] / 1024.0**2:.1f} MiB"
    )
    return summary


if __name__ == "__main__":
    run_stage14_1_diagnostics()
