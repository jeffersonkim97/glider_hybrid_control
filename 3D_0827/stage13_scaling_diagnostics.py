"""Execute the Stage-13 measured scaling-preparation smoke suite."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import plotly.graph_objects as go

from scaling_benchmark import (
    ProcessPeakMemorySampler,
    ScalingBenchmarkRecord,
    RepeatMeasurement,
    ScalingRunConfig,
    _attacker_kwargs,
    differing_fields,
    run_scaling_benchmark,
)
from attacker_best_response import run_attacker_best_response
from stackelberg_solver import generate_defender_line_actions
from stage11_notebook_support import save_figure, write_json
from terrain_catalog import build_terrain


ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE13_FIGURE_DIRECTORY = ROOT / "figure" / "stage_13_scaling_preparation"


def _record_from_dict(raw: dict[str, Any]) -> ScalingBenchmarkRecord:
    copied = dict(raw)
    copied["repeats"] = tuple(
        RepeatMeasurement(**item) for item in copied["repeats"]
    )
    return ScalingBenchmarkRecord(**copied)


def benchmark_profiles() -> tuple[ScalingRunConfig, ...]:
    baseline = ScalingRunConfig(
        run_id="baseline_repeat",
        sweep_variable="baseline",
        repeat_count=2,
    )
    resolution = baseline.discretization
    return (
        baseline,
        ScalingRunConfig(
            run_id="spatial_fine_50m_5m",
            sweep_variable="spatial_resolution",
            discretization=replace(
                resolution,
                horizontal_spacing_map=0.5,
                altitude_spacing_map=0.05,
                motion_primitive_step_cells=2,
            ),
        ),
        ScalingRunConfig(
            run_id="heading_12",
            sweep_variable="heading_resolution",
            discretization=replace(
                resolution,
                heading_bin_count=12,
                motion_primitive_radius=2,
            ),
        ),
        ScalingRunConfig(
            run_id="switch_candidates_128",
            sweep_variable="switch_candidate_count",
            discretization=replace(
                resolution,
                switching_contour_sample_count=16,
            ),
        ),
        ScalingRunConfig(
            run_id="defender_actions_2",
            sweep_variable="defender_action_count",
            defender_x_map=(5.0, 7.0),
        ),
        ScalingRunConfig(
            run_id="terrain_stepped_pyramid",
            sweep_variable="terrain_complexity",
            terrain_category="stepped_pyramid",
        ),
        ScalingRunConfig(
            run_id="terrain_offset_cube_left",
            sweep_variable="terrain_complexity",
            terrain_category="offset_cube_left",
        ),
    )


def _write_records_csv(
    path: Path,
    records: tuple[ScalingBenchmarkRecord, ...],
) -> None:
    field_names = tuple(
        field.name for field in fields(ScalingBenchmarkRecord)
        if field.name not in ("repeats", "configuration")
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_dict(include_repeats=False))


def _runtime_vs_state_count(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=[record.state_count for record in records],
        y=[record.t_attacker_br_s for record in records],
        mode="markers+text",
        text=[record.run_id for record in records],
        textposition="top center",
        marker={"size": 11, "color": [record.heading_bins for record in records], "colorscale": "Viridis", "colorbar": {"title": "heading bins"}},
        hovertemplate="%{text}<br>states=%{x:,}<br>T_attacker_BR=%{y:.3f}s<extra></extra>",
    ))
    figure.update_layout(
        title="Stage 13 Smoke: State Count vs Exact Attacker-BR Runtime",
        xaxis_title="admissible Bellman states",
        yaxis_title="median exact attacker-BR runtime [s]",
        template="plotly_white",
    )
    return figure


def _objective_convergence(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    spatial_ids = {"spatial_fine_50m_5m", "baseline_repeat"}
    spatial = tuple(record for record in records if record.run_id in spatial_ids)
    spatial = tuple(sorted(spatial, key=lambda record: record.grid_dx, reverse=True))
    figure = go.Figure(go.Scatter(
        x=[record.grid_dx * 100.0 for record in spatial],
        y=[record.attacker_objective for record in spatial],
        mode="lines+markers+text",
        text=[record.run_id for record in spatial],
        textposition="top center",
        marker={"size": 11},
        hovertemplate="%{text}<br>xy spacing=%{x:.1f}m<br>J_A=%{y:.9f}<extra></extra>",
    ))
    figure.update_layout(
        title="Stage 13 Smoke: Spatial Resolution vs Attacker Objective",
        xaxis_title="x/y spacing [m] (smaller is finer)",
        yaxis_title="exact finite attacker objective",
        template="plotly_white",
    )
    return figure


def _memory_vs_state_count(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=[record.state_count for record in records],
        y=[record.peak_memory_delta_bytes / 2**20 for record in records],
        mode="markers+text",
        text=[record.run_id for record in records],
        textposition="top center",
        marker={"size": 11, "color": "#7c3aed"},
        hovertemplate="%{text}<br>states=%{x:,}<br>incremental peak RSS=%{y:.2f} MiB<extra></extra>",
    ))
    figure.update_layout(
        title="Stage 13 Smoke: State Count vs Incremental Peak RSS",
        xaxis_title="admissible Bellman states",
        yaxis_title="sampled peak RSS minus run-start RSS [MiB]",
        template="plotly_white",
    )
    return figure


def _runtime_decomposition(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    figure = go.Figure()
    for attribute, label, color in (
        ("t_los_s", "LOS", "#06b6d4"),
        ("t_graph_build_s", "graph + unit reachability", "#64748b"),
        ("t_hazard_s", "hazard precompute", "#ef4444"),
        ("t_bellman_s", "additive Bellman recursion", "#22c55e"),
    ):
        figure.add_trace(go.Bar(
            x=[record.run_id for record in records],
            y=[getattr(record, attribute) for record in records],
            name=label,
        ))
    figure.update_layout(
        title="Stage 13 Smoke: Measured Runtime Decomposition",
        xaxis_title="benchmark run",
        yaxis_title="median wall time [s]",
        barmode="stack",
        template="plotly_white",
    )
    return figure


def _heading_runtime(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    selected = tuple(
        record for record in records
        if record.run_id in ("baseline_repeat", "heading_12")
    )
    selected = tuple(sorted(selected, key=lambda record: record.heading_bins))
    figure = go.Figure(go.Scatter(
        x=[record.heading_bins for record in selected],
        y=[record.t_attacker_br_s for record in selected],
        mode="lines+markers+text",
        text=[record.run_id for record in selected],
        textposition="top center",
        marker={"size": 11},
    ))
    figure.update_layout(
        title="Stage 13 Smoke: Heading Resolution vs Exact Runtime",
        xaxis_title="heading bins",
        yaxis_title="median exact attacker-BR runtime [s]",
        template="plotly_white",
    )
    return figure


def _defender_runtime(records: tuple[ScalingBenchmarkRecord, ...]) -> go.Figure:
    selected = tuple(
        record for record in records
        if record.run_id in ("baseline_repeat", "defender_actions_2")
    )
    selected = tuple(sorted(selected, key=lambda record: record.defender_action_count))
    figure = go.Figure(go.Scatter(
        x=[record.defender_action_count for record in selected],
        y=[record.t_sse_s for record in selected],
        mode="lines+markers+text",
        text=[record.run_id for record in selected],
        textposition="top center",
        marker={"size": 11},
    ))
    figure.update_layout(
        title="Stage 13 Smoke: Defender Action Count vs Exact SSE Runtime",
        xaxis_title="enumerated defender actions",
        yaxis_title="median exact SSE runtime [s]",
        template="plotly_white",
    )
    return figure


def _diagnose_exact_infeasibility(profile: ScalingRunConfig) -> dict[str, Any]:
    """Exhaustively certify a profile whose finite game has no follower path."""
    config = profile.stage11_config()
    candidates = generate_defender_line_actions(
        profile.defender_x_map,
        y_map=profile.defender_y_map,
        z_map=profile.defender_z_map,
    )
    start = perf_counter()
    action_rows: list[dict[str, Any]] = []
    with ProcessPeakMemorySampler() as memory:
        for candidate in candidates:
            run = run_attacker_best_response(
                candidate.action,
                config.attacker_initial_condition,
                terrain=build_terrain(profile.terrain_category),
                **_attacker_kwargs(config),
            )
            if run.selected_result is not None:
                raise RuntimeError(
                    "infeasibility diagnostic found a feasible attacker response"
                )
            status_counts = Counter(
                result.status_category for result in run.candidate_results
            )
            reason_counts = Counter(
                result.infeasibility_reason or "unspecified"
                for result in run.candidate_results
            )
            action_rows.append({
                "defender_action_id": candidate.action_id,
                "sensor_position_map": candidate.action.sensor_position_map.tolist(),
                "candidate_count": len(run.candidate_results),
                "candidate_ids_unique": len({
                    item.candidate_id for item in run.candidate_results
                }) == len(run.candidate_results),
                "feasible_candidate_count": run.metrics.number_feasible,
                "status_counts": dict(sorted(status_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "state_count": run.graph.statistics.state_count,
                "edge_count": run.graph.statistics.valid_edge_count,
                "T_LOS_s": run.metrics.timing.los_s,
                "T_graph_build_s": run.metrics.timing.graph_build_s,
                "T_hazard_s": run.metrics.timing.hazard_precompute_s,
                "T_Bellman_s": run.metrics.timing.bellman_solve_s,
                "T_attacker_BR_s": run.metrics.timing.total_attacker_br_s,
            })
    return {
        "run_id": profile.run_id,
        "sweep_variable": profile.sweep_variable,
        "terrain": profile.terrain_category,
        "status": "exact_finite_infeasible",
        "category": "no feasible attacker response",
        "validation_passed": True,
        "trajectory_replay_applicable": False,
        "exactness_certificate": (
            "every configured defender action and every generated attacker "
            "switching candidate was evaluated; none produced a feasible path"
        ),
        "defender_action_count": len(candidates),
        "attacker_objective": None,
        "defender_payoff": None,
        "T_total_attempt_s": perf_counter() - start,
        "process_rss_start_bytes": memory.start_rss,
        "process_rss_peak_bytes": memory.peak_rss,
        "process_rss_delta_peak_bytes": memory.delta_peak_bytes,
        "memory_measurement_method": ProcessPeakMemorySampler.method,
        "actions": action_rows,
        "configuration": profile.as_dict(),
    }


def _run_profile_worker(profile: ScalingRunConfig, output_path: Path) -> None:
    """Run one profile in a fresh process so RSS is not cross-run polluted."""
    try:
        record, _ = run_scaling_benchmark(profile)
    except RuntimeError as error:
        if str(error) != "no Defender action has a feasible Attacker response":
            raise
        failure = _diagnose_exact_infeasibility(profile)
        write_json(output_path, {"status": "infeasible", "failure": failure})
        return
    write_json(
        output_path,
        {"status": "complete", "record": record.as_dict()},
    )


def _execute_profile_isolated(
    profile: ScalingRunConfig,
    output_directory: Path,
) -> dict[str, Any]:
    worker_path = output_directory / f".{profile.run_id}.worker.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--profile-worker",
            profile.run_id,
            str(worker_path),
        ],
        cwd=str(ROOT),
        check=False,
    )
    if completed.returncode != 0 or not worker_path.is_file():
        raise RuntimeError(
            f"isolated benchmark worker failed for {profile.run_id}"
        )
    payload = json.loads(worker_path.read_text(encoding="utf-8"))
    worker_path.unlink()
    return payload


def run_stage13_diagnostics(
    output_directory: Path = DEFAULT_STAGE13_FIGURE_DIRECTORY,
) -> tuple[ScalingBenchmarkRecord, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    profiles = benchmark_profiles()
    baseline = profiles[0]
    allowed_changes = {
        "spatial_fine_50m_5m": {
            "discretization.horizontal_spacing_map",
            "discretization.altitude_spacing_map",
            "discretization.motion_primitive_step_cells",
        },
        "heading_12": {
            "discretization.heading_bin_count",
            "discretization.motion_primitive_radius",
        },
        "switch_candidates_128": {
            "discretization.switching_contour_sample_count",
        },
        "defender_actions_2": {"defender_x_map"},
        "terrain_stepped_pyramid": {"terrain_category"},
        "terrain_offset_cube_left": {"terrain_category"},
    }
    for profile in profiles[1:]:
        actual = set(differing_fields(baseline, profile))
        if actual != allowed_changes[profile.run_id]:
            raise RuntimeError(
                f"{profile.run_id} changed unexpected fields: {sorted(actual)}"
            )

    partial = output_directory / "benchmark_results.partial.json"
    failure_partial = output_directory / "benchmark_failures.partial.json"
    records: list[ScalingBenchmarkRecord] = []
    if partial.exists():
        raw_records = json.loads(partial.read_text(encoding="utf-8"))
        records.extend(_record_from_dict(raw) for raw in raw_records)
    failures: list[dict[str, Any]] = (
        json.loads(failure_partial.read_text(encoding="utf-8"))
        if failure_partial.exists()
        else []
    )
    completed_by_id = {record.run_id: record for record in records}
    failed_by_id = {item["run_id"]: item for item in failures}
    for profile in profiles:
        checkpoint = completed_by_id.get(profile.run_id)
        if checkpoint is not None:
            if checkpoint.configuration != profile.as_dict():
                raise RuntimeError(
                    f"checkpoint configuration mismatch for {profile.run_id}"
                )
            print(f"Reusing validated checkpoint {profile.run_id}", flush=True)
            continue
        failure_checkpoint = failed_by_id.get(profile.run_id)
        if failure_checkpoint is not None:
            if failure_checkpoint["configuration"] != profile.as_dict():
                raise RuntimeError(
                    f"failure checkpoint configuration mismatch for {profile.run_id}"
                )
            print(f"Reusing infeasibility certificate {profile.run_id}", flush=True)
            continue
        print(
            f"Running {profile.run_id}: sweep={profile.sweep_variable}, "
            f"repeats={profile.repeat_count}, defenders={len(profile.defender_x_map)}",
            flush=True,
        )
        payload = _execute_profile_isolated(profile, output_directory)
        if payload["status"] == "infeasible":
            failure = payload["failure"]
            if not failure["validation_passed"]:
                raise RuntimeError(
                    f"infeasibility certificate failed for {profile.run_id}"
                )
            failures.append(failure)
            failed_by_id[profile.run_id] = failure
            write_json(failure_partial, failures)
            print(
                f"  EXACT INFEASIBLE actions={failure['defender_action_count']}, "
                f"T_attempt={failure['T_total_attempt_s']:.3f}s",
                flush=True,
            )
            continue
        if payload["status"] != "complete":
            raise RuntimeError(f"unknown worker status for {profile.run_id}")
        record = _record_from_dict(payload["record"])
        if not (
            record.validation_passed
            and record.deterministic
            and record.exhaustive_attacker_verified
            and record.exhaustive_defender_verified
        ):
            raise RuntimeError(f"benchmark gate failed for {record.run_id}")
        records.append(record)
        write_json(
            output_directory / "benchmark_results.partial.json",
            [item.as_dict() for item in records],
        )
        print(
            f"  PASS states={record.state_count:,}, edges={record.edge_count:,}, "
            f"T_SSE={record.t_sse_s:.3f}s, peak={record.peak_memory_bytes / 2**20:.1f}MiB, "
            f"J_A={record.attacker_objective:.9f}",
            flush=True,
        )

    result = tuple(records)
    _write_records_csv(output_directory / "benchmark_results.csv", result)
    write_json(
        output_directory / "benchmark_results.json",
        [record.as_dict() for record in result],
    )
    write_json(output_directory / "benchmark_failures.json", failures)
    write_json(
        output_directory / "benchmark_attempts.json",
        [
            {"run_id": record.run_id, "status": "complete", "record": record.as_dict()}
            for record in result
        ] + [
            {"run_id": failure["run_id"], "status": "exact_finite_infeasible", "failure": failure}
            for failure in failures
        ],
    )
    write_json(
        output_directory / "benchmark_configurations.json",
        [profile.as_dict() for profile in profiles],
    )
    baseline_record = result[0]
    write_json(
        output_directory / "repeated_timing_summary.json",
        {
            "run_id": baseline_record.run_id,
            "repeat_count": baseline_record.repeat_count,
            "runtime_min_s": baseline_record.runtime_min_s,
            "runtime_median_s": baseline_record.runtime_median_s,
            "runtime_max_s": baseline_record.runtime_max_s,
            "runtime_spread_s": baseline_record.runtime_spread_s,
            "deterministic": baseline_record.deterministic,
        },
    )

    figures = {
        "runtime_vs_state_count.html": _runtime_vs_state_count(result),
        "resolution_vs_objective.html": _objective_convergence(result),
        "memory_vs_state_count.html": _memory_vs_state_count(result),
        "runtime_decomposition.html": _runtime_decomposition(result),
        "heading_vs_runtime.html": _heading_runtime(result),
        "defender_count_vs_runtime.html": _defender_runtime(result),
    }
    for filename, figure in figures.items():
        save_figure(figure, output_directory, filename)

    summary: dict[str, Any] = {
        "stage": 13,
        "gate_passed": True,
        "scope": (
            "measured smoke data validating the scaling framework; not a "
            "production convergence conclusion and not an RL comparison"
        ),
        "profile_count": len(profiles),
        "valid_result_count": len(result),
        "exact_infeasible_count": len(failures),
        "all_validation_passed": all(item.validation_passed for item in result),
        "all_exactness_checks_passed": all(
            item.exhaustive_attacker_verified and item.exhaustive_defender_verified
            for item in result
        ),
        "repeated_baseline_deterministic": baseline_record.deterministic,
        "memory_measurement_method": baseline_record.memory_measurement_method,
        "memory_comparison_metric": (
            "peak_memory_delta_bytes (sampled peak RSS minus worker start RSS); "
            "absolute process peak remains stored as peak_memory_bytes"
        ),
        "run_ids": [profile.run_id for profile in profiles],
        "terrain_categories_attempted": sorted({
            profile.terrain_category for profile in profiles
        }),
        "terrain_categories_with_feasible_results": sorted({
            item.terrain for item in result
        }),
        "spatial_motion_horizon_policy": (
            "motion_primitive_step_cells changes inversely with horizontal "
            "spacing so the spatial sweep preserves the 1-map-unit/100-m "
            "canonical transition horizon"
        ),
        "virtual_connection_regression_fixes": [
            "turn-infeasible zero-length identity proposals use only corners "
            "of cells incident to the exact lattice intersection",
            "virtual chord-heading compatibility is rejected in the solver "
            "at the same 1e-8-rad tolerance used by independent replay",
        ],
        "notebook_policy": "Stage-11 and original GUI notebooks remain frozen",
        "notebook_sha256": {
            "3D_bellman_0827.ipynb": hashlib.sha256(
                (ROOT / "3D_bellman_0827.ipynb").read_bytes()
            ).hexdigest(),
            "3D_Attacker_Bellman_Validated.ipynb": hashlib.sha256(
                (ROOT / "3D_Attacker_Bellman_Validated.ipynb").read_bytes()
            ).hexdigest(),
        },
        "rl_implemented": False,
    }
    write_json(output_directory / "stage13_summary.json", summary)
    write_json(
        output_directory / "rejected_preliminary_runs.json",
        [
            {
                "run_id": "spatial_coarse_200m_20m_pre_fix",
                "recorded_as_benchmark": False,
                "category": "virtual connection infeasible",
                "reason": (
                    "independent replay rejected a selected virtual chord "
                    "whose direction did not match its target heading bin"
                ),
            },
            {
                "run_id": "spatial_fine_50m_10m_pre_fix",
                "recorded_as_benchmark": False,
                "category": "goal unreachable",
                "reason": (
                    "halving horizontal spacing alone shortened the physical "
                    "motion horizon and made 45-degree turns unreachable"
                ),
            },
            {
                "run_id": "spatial_fine_50m_5m_pre_fix",
                "recorded_as_benchmark": False,
                "category": "virtual connection infeasible",
                "reason": (
                    "an exact x-y lattice switching point generated only a "
                    "turn-infeasible zero-length identity proposal"
                ),
            },
        ],
    )
    if partial.exists():
        partial.unlink()
    if failure_partial.exists():
        failure_partial.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-worker")
    parser.add_argument("worker_output", nargs="?")
    arguments = parser.parse_args()
    if arguments.profile_worker is None:
        run_stage13_diagnostics()
    else:
        profiles_by_id = {item.run_id: item for item in benchmark_profiles()}
        try:
            selected_profile = profiles_by_id[arguments.profile_worker]
        except KeyError as error:
            raise SystemExit(f"unknown profile {arguments.profile_worker!r}") from error
        if arguments.worker_output is None:
            raise SystemExit("worker_output is required with --profile-worker")
        _run_profile_worker(selected_profile, Path(arguments.worker_output))
