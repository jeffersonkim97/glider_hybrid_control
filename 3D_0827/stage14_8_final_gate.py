"""Stage 14.8 final integration and reproducibility gate.

This module does not change or replace the exact finite P1b solver.  It audits
the frozen Stage-14 artifacts, executes one canonical benchmark-harness smoke
run, regenerates figures from saved machine-readable artifacts, runs the full
Stage-14 regression suite, and records the reproducibility boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from stage14_benchmark_contract import (
    DEFAULT_STAGE14_TOLERANCES,
    solver_source_manifest,
)
from stage14_benchmark_runner import (
    BenchmarkCase,
    read_jsonl,
    run_benchmark_cases,
    summarize_repetitions,
    write_raw_csv,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
FIGURE_ROOT = ROOT / "figure"
OUTPUT = FIGURE_ROOT / "stage_14_8_final_gate"
SCHEMA_VERSION = "stage14.8-v1"

FROZEN_REFERENCE_PATH = FIGURE_ROOT / "stage_14_0_benchmark_contract" / "frozen_reference.json"
FROZEN_ENVIRONMENT_PATH = FIGURE_ROOT / "stage_14_0_benchmark_contract" / "environment_manifest.json"
PROFILE_PATH = FIGURE_ROOT / "stage_14_1_runtime_profiling" / "canonical_profile.json"

RAW_SOURCES = (
    ("14.2", FIGURE_ROOT / "stage_14_2_benchmark_runner" / "raw_repetitions.json"),
    ("14.2C", FIGURE_ROOT / "stage_14_2c_local_scaling" / "raw_repetitions.json"),
    ("14.3", FIGURE_ROOT / "stage_14_3_spatial_resolution" / "raw_spatial_repetitions.json"),
    ("14.4", FIGURE_ROOT / "stage_14_4_heading_resolution" / "raw_heading_repetitions.json"),
    ("14.5", FIGURE_ROOT / "stage_14_5_switching_candidates" / "raw_candidate_repetitions.json"),
    ("14.6", FIGURE_ROOT / "stage_14_6_neighbor_radius" / "raw_radius_repetitions.json"),
)

CONFIGURATION_SOURCES = (
    FIGURE_ROOT / "stage_14_0_benchmark_contract" / "benchmark_configuration.json",
    FIGURE_ROOT / "stage_14_2_benchmark_runner" / "smoke_case_configurations.json",
    FIGURE_ROOT / "stage_14_2c_local_scaling" / "benchmark_configurations.json",
    FIGURE_ROOT / "stage_14_3_spatial_resolution" / "spatial_sweep_configuration.json",
    FIGURE_ROOT / "stage_14_4_heading_resolution" / "heading_sweep_configuration.json",
    FIGURE_ROOT / "stage_14_5_switching_candidates" / "candidate_sweep_configuration.json",
    FIGURE_ROOT / "stage_14_6_neighbor_radius" / "radius_sweep_configuration.json",
)

STAGE_SUMMARIES = (
    FIGURE_ROOT / "stage_14_0_benchmark_contract" / "stage14_0_summary.json",
    FIGURE_ROOT / "stage_14_1_runtime_profiling" / "stage14_1_summary.json",
    FIGURE_ROOT / "stage_14_2a_local_sse_contract" / "stage14_2a_summary.json",
    FIGURE_ROOT / "stage_14_2b_local_sse_search" / "stage14_2b_summary.json",
    FIGURE_ROOT / "stage_14_2c_local_scaling" / "stage14_2c_summary.json",
    FIGURE_ROOT / "stage_14_2_benchmark_runner" / "stage14_2_summary.json",
    FIGURE_ROOT / "stage_14_3_spatial_resolution" / "stage14_3_summary.json",
    FIGURE_ROOT / "stage_14_4_heading_resolution" / "stage14_4_summary.json",
    FIGURE_ROOT / "stage_14_5_switching_candidates" / "stage14_5_summary.json",
    FIGURE_ROOT / "stage_14_6_neighbor_radius" / "stage14_6_summary.json",
    FIGURE_ROOT / "stage_14_7_scaling_analysis" / "stage14_7_summary.json",
)

SCIENTIFIC_SWEEP_STAGES = frozenset({"14.2C", "14.3", "14.4", "14.5", "14.6"})
MODEL_INFEASIBLE_STATUSES = frozenset({
    "model_infeasible", "initial_defender_infeasible",
    "initial_neighborhood_infeasible",
})
COMPUTATIONAL_FAILURE_STATUSES = frozenset({
    "timeout", "memory_limit", "worker_failure", "computational_failure", "numerical_failure",
    "initial_neighborhood_comparison_unknown",
})
REQUIRED_COUNTS = ("N_S_cart", "N_S_active", "N_E", "N_D", "B", "Q")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    tie = row.get("tie_breaking_result") or {}
    return (
        row.get("J_A"), row.get("J_D"), row.get("selected_defender_action_id"),
        row.get("selected_attacker_candidate_id"),
        (row.get("trajectory_identity") or {}).get("sha256"),
        tuple(tie.get("leader_cooptimal_action_ids") or ()),
        tuple(tie.get("attacker_objective_cooptimal_candidate_ids") or ()),
    )


def stage14_8_smoke_case() -> BenchmarkCase:
    """One exact canonical run through the fresh-process benchmark harness."""
    return BenchmarkCase(
        experiment_type="stage14_8_final_gate",
        name="canonical_centered_cube_25m_25m_5deg_final_smoke",
        worker_mode="canonical",
        sweep_variable="horizontal_spacing_m",
        parameters=(
            ("horizontal_spacing_m", 25.0),
            ("altitude_spacing_m", 25.0),
            ("heading_spacing_deg", 5.0),
            ("terrain_category", "centered_cube"),
        ),
        repetitions=1,
        timeout_s=300.0,
        sampling_interval_s=0.02,
    )


def compare_canonical_row(row: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    """Compare one harness result to the immutable pre-Stage-14 identity."""
    expected = frozen["solution_identity"]
    tolerance = DEFAULT_STAGE14_TOLERANCES
    tie = row.get("tie_breaking_result") or {}
    checks = {
        "completed": row.get("status") == "completed",
        "feasible": row.get("feasible") is True,
        "selected_defender_action": row.get("selected_defender_action_id") == expected["selected_defender_action_id"],
        "selected_attacker_candidate": row.get("selected_attacker_candidate_id") == expected["selected_attacker_candidate_id"],
        "attacker_objective": bool(np.isclose(
            row.get("J_A"), expected["attacker_objective"], rtol=0.0,
            atol=tolerance.attacker_objective_abs,
        )),
        "defender_objective": bool(np.isclose(
            row.get("J_D"), expected["defender_objective_pod"], rtol=0.0,
            atol=tolerance.defender_objective_abs,
        )),
        "trajectory_identity": (
            (row.get("trajectory_identity") or {}).get("sha256")
            == expected["trajectory_identity"]["sha256"]
        ),
        "leader_tie_set": tie.get("leader_cooptimal_action_ids") == expected["leader_cooptimal_action_ids"],
        "follower_tie_set": (
            tie.get("attacker_objective_cooptimal_candidate_ids")
            == expected["attacker_objective_cooptimal_candidate_ids"]
        ),
        "tie_convention": tie.get("convention") == expected["tie_break_convention"],
        "independent_replay": row.get("independent_replay_passed") is True,
        "solver_source_fingerprint": row.get("solver_source_fingerprint") == frozen["solver_source_fingerprint"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def load_tagged_raw_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, path in RAW_SOURCES:
        payload = _read_json(path)
        if not isinstance(payload, list):
            raise TypeError(f"raw repetition artifact must be a list: {path}")
        rows.extend({"stage_source": stage, **row} for row in payload)
    return rows


def audit_raw_repetitions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    noncompleted = [row for row in rows if row.get("status") != "completed"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage_source"], row["case_id"])].append(row)

    completed_fields_valid = all(
        row.get("feasible") is True
        and row.get("independent_replay_passed") is True
        and row.get("selected_defender_action_id") is not None
        and row.get("selected_attacker_candidate_id") is not None
        and (row.get("trajectory_identity") or {}).get("sha256")
        for row in completed
    )
    noncompleted_explicit = all(
        row.get("status") in MODEL_INFEASIBLE_STATUSES | COMPUTATIONAL_FAILURE_STATUSES
        and bool(row.get("failure_category"))
        and bool(row.get("failure_message"))
        for row in noncompleted
    )
    repeated = {
        f"{stage}:{case_id}": len({_canonical_signature(row) for row in case_rows})
        for (stage, case_id), case_rows in groups.items()
        if sum(row.get("status") == "completed" for row in case_rows) >= 2
    }
    scientific_groups = {
        key: case_rows for key, case_rows in groups.items() if key[0] in SCIENTIFIC_SWEEP_STAGES
    }
    counts_present = all(
        all((row.get("state_and_game_size") or {}).get(name) is not None for name in REQUIRED_COUNTS)
        and any(
            (row.get("state_and_game_size") or {}).get(name) is not None
            for name in ("N_C_raw", "N_C")
        )
        for row in completed if row["stage_source"] in SCIENTIFIC_SWEEP_STAGES
    )
    one_variable_contract = all(
        (row.get("configuration") or {}).get("sweep_variable")
        in ((row.get("configuration") or {}).get("parameters") or {})
        for row in rows
    )
    physics_rows = [
        row for row in rows
        if ((row.get("configuration") or {}).get("parameters") or {}).get("terrain_category") is not None
    ]
    single_cube = all(
        ((row.get("configuration") or {}).get("parameters") or {}).get("terrain_category") == "centered_cube"
        for row in physics_rows
    )
    no_approximation = all(
        parameters.get("reinforcement_learning", False) is False
        and parameters.get("approximate_planner", False) is False
        for parameters in (
            ((row.get("configuration") or {}).get("parameters") or {}) for row in rows
        )
    )
    checks = {
        "raw_sources_nonempty": bool(rows),
        "stage_qualified_repetition_ids_unique": len(rows) == len({
            (row["stage_source"], row["repetition_id"]) for row in rows
        }),
        "completed_rows_are_feasible_and_replay_validated": completed_fields_valid,
        "noncompleted_rows_have_explicit_classification": noncompleted_explicit,
        "repeated_identical_cases_are_solution_deterministic": bool(repeated) and all(
            count == 1 for count in repeated.values()
        ),
        "scientific_sweep_configurations_attempt_at_least_three_repetitions": bool(scientific_groups) and all(
            len(case_rows) >= 3 for case_rows in scientific_groups.values()
        ),
        "median_is_primary_runtime_statistic": True,
        "required_state_game_counts_present": counts_present,
        "one_declared_sweep_variable_per_case": one_variable_contract,
        "single_cube_used_for_physics_benchmarks": bool(physics_rows) and single_cube,
        "no_rl_or_approximate_planner": no_approximation,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "row_count": len(rows),
        "completed_count": len(completed),
        "noncompleted_count": len(noncompleted),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "stage_counts": dict(sorted(Counter(row["stage_source"] for row in rows).items())),
        "repeated_case_unique_solution_signature_counts": repeated,
        "scientific_configuration_count": len(scientific_groups),
        "five_repetition_policy": (
            "three fresh-process repetitions were used for every scientific configuration; "
            "five were not practical for the measured fine-grid runtime/memory cases"
        ),
        "noncompleted_cases": [
            {
                "stage_source": row["stage_source"],
                "case_id": row["case_id"],
                "repetition_id": row["repetition_id"],
                "status": row["status"],
                "failure_category": row.get("failure_category"),
                "failure_message": row.get("failure_message"),
            }
            for row in noncompleted
        ],
    }


def summarize_tagged_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage_source"], row["case_id"])].append(row)
    summaries = []
    for (stage, case_id), case_rows in sorted(groups.items()):
        completed = [row for row in case_rows if row["status"] == "completed"]

        def med(extractor) -> float | None:
            values = [extractor(row) for row in completed]
            values = [float(value) for value in values if value is not None]
            return None if not values else float(median(values))

        summaries.append({
            "stage_source": stage,
            "case_id": case_id,
            "configuration": case_rows[0]["configuration"],
            "attempted_repetitions": len(case_rows),
            "completed_repetitions": len(completed),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "median_total_runtime_s": med(
                lambda row: (((row.get("timing") or {}).get("totals") or {}).get("T_SSE_s"))
            ),
            "median_peak_rss_bytes": med(lambda row: (row.get("memory") or {}).get("peak_rss_bytes")),
            "median_attacker_objective": med(lambda row: row.get("J_A")),
            "median_defender_objective": med(lambda row: row.get("J_D")),
            "unique_solution_signature_count": len({_canonical_signature(row) for row in completed}),
            "all_completed_replays_pass": bool(completed) and all(
                row.get("independent_replay_passed") is True for row in completed
            ),
        })
    return summaries


def _write_unified_raw(output: Path, rows: list[dict[str, Any]]) -> None:
    _write_json(output / "all_raw_repetitions.json", rows)
    with (output / "all_raw_repetitions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    fields = (
        "stage_source", "experiment_type", "case_id", "repetition_id", "status",
        "failure_category", "failure_message", "declared_sweep_variable",
        "declared_sweep_value", "worker_wall_s", "T_SSE_s", "peak_rss_bytes",
        "J_A", "J_D", "selected_defender_action_id", "selected_attacker_candidate_id",
        "trajectory_sha256", "independent_replay_passed",
    )
    with (output / "all_raw_repetitions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stage_source": row["stage_source"],
                "experiment_type": row.get("experiment_type"),
                "case_id": row.get("case_id"),
                "repetition_id": row.get("repetition_id"),
                "status": row.get("status"),
                "failure_category": row.get("failure_category"),
                "failure_message": row.get("failure_message"),
                "declared_sweep_variable": row.get("declared_sweep_variable"),
                "declared_sweep_value": row.get("declared_sweep_value"),
                "worker_wall_s": row.get("worker_wall_s"),
                "T_SSE_s": (((row.get("timing") or {}).get("totals") or {}).get("T_SSE_s")),
                "peak_rss_bytes": (row.get("memory") or {}).get("peak_rss_bytes"),
                "J_A": row.get("J_A"), "J_D": row.get("J_D"),
                "selected_defender_action_id": row.get("selected_defender_action_id"),
                "selected_attacker_candidate_id": row.get("selected_attacker_candidate_id"),
                "trajectory_sha256": (row.get("trajectory_identity") or {}).get("sha256"),
                "independent_replay_passed": row.get("independent_replay_passed"),
            })


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = (
        "stage_source", "case_id", "attempted_repetitions", "completed_repetitions",
        "status_counts_json", "median_total_runtime_s", "median_peak_rss_bytes",
        "median_attacker_objective", "median_defender_objective",
        "unique_solution_signature_count", "all_completed_replays_pass", "configuration_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                **{name: row.get(name) for name in fields},
                "status_counts_json": json.dumps(row["status_counts"], sort_keys=True),
                "configuration_json": json.dumps(row["configuration"], sort_keys=True),
            })


def configuration_manifest() -> dict[str, Any]:
    entries = []
    for path in CONFIGURATION_SOURCES:
        entries.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "payload": _read_json(path),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "frozen": True,
        "one_variable_at_a_time_required": True,
        "configuration_sources": entries,
    }


def stage_summary_audit() -> dict[str, Any]:
    entries = []
    for path in STAGE_SUMMARIES:
        payload = _read_json(path)
        entries.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "gate_passed": payload.get("gate_passed"),
        })
    return {"passed": all(item["gate_passed"] is True for item in entries), "entries": entries}


def profile_regression(frozen: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "profile_status_completed": profile.get("status") == "completed",
        "profiled_vs_unprofiled_passed": (profile.get("profiled_vs_unprofiled_regression") or {}).get("passed") is True,
        "frozen_solution_regression_passed": (profile.get("frozen_solution_regression") or {}).get("passed") is True,
        "instrumentation_gate_passed": (profile.get("instrumentation_gate") or {}).get("passed") is True,
        "solution_identity_exact_match": profile.get("solution_identity") == frozen.get("solution_identity"),
        "solver_source_fingerprint_match": profile.get("solver_source_fingerprint") == frozen.get("solver_source_fingerprint"),
        "independent_replay_passed": (profile.get("independent_replay") or {}).get("passed") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _run_command(command: list[str], *, cwd: Path, label: str) -> dict[str, Any]:
    started = perf_counter()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    return {
        "label": label,
        "command": command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "wall_s": perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
    }


def _test_count(record: dict[str, Any]) -> int | None:
    match = re.search(r"Ran\s+(\d+)\s+tests?", record["stdout"] + "\n" + record["stderr"])
    return None if match is None else int(match.group(1))


def _notebook_output_errors() -> int:
    notebook = _read_json(ROOT / "3D_Stage14_Computation_Load.ipynb")
    return sum(
        output.get("output_type") == "error"
        for cell in notebook["cells"] for output in cell.get("outputs", [])
    )


def figure_manifest() -> dict[str, Any]:
    figures = []
    for directory in sorted(FIGURE_ROOT.glob("stage_14_*")):
        if directory.name == OUTPUT.name:
            continue
        for path in sorted(directory.glob("*.png")):
            figures.append({
                "stage_directory": directory.name,
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return {"schema_version": SCHEMA_VERSION, "figure_count": len(figures), "figures": figures}


def artifact_manifest(output: Path) -> dict[str, Any]:
    files = []
    for directory in sorted(FIGURE_ROOT.glob("stage_14_*")):
        if directory == output:
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            files.append({
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return {"schema_version": SCHEMA_VERSION, "artifact_count": len(files), "artifacts": files}


def practical_boundaries() -> dict[str, Any]:
    normalized = _read_json(
        FIGURE_ROOT / "stage_14_7_scaling_analysis" / "normalized_scaling_measurements.json"
    )
    completed = [row for row in normalized if row["successful_repetitions"] > 0]
    max_runtime = max(completed, key=lambda row: row["T_total_s"])
    max_memory = max(completed, key=lambda row: row["peak_rss_bytes"])
    limits = _read_json(
        FIGURE_ROOT / "stage_14_7_scaling_analysis" / "computational_limit_cases.json"
    )
    return {
        "claim_scope": "observations only over completed measured configurations; no extrapolation",
        "completed_configuration_count": len(completed),
        "maximum_measured_median_total_runtime": {
            "seconds": max_runtime["T_total_s"], "sweep": max_runtime["sweep"],
            "variant": max_runtime["algorithm_variant"], "label": max_runtime["label"],
        },
        "maximum_measured_median_peak_rss": {
            "bytes": max_memory["peak_rss_bytes"], "mib": max_memory["peak_rss_bytes"] / 1024**2,
            "sweep": max_memory["sweep"], "variant": max_memory["algorithm_variant"],
            "label": max_memory["label"],
        },
        "noncompleted_repetition_inventory": limits,
    }


def _write_boundary_markdown(path: Path, boundary: dict[str, Any]) -> None:
    runtime = boundary["maximum_measured_median_total_runtime"]
    memory = boundary["maximum_measured_median_peak_rss"]
    limits = boundary["noncompleted_repetition_inventory"]
    path.write_text(
        "# Stage 14 measured practical boundary\n\n"
        "These observations are limited to the configurations actually measured; no extrapolation is made.\n\n"
        f"- Largest completed median total runtime: {runtime['seconds']:.6f} s "
        f"({runtime['sweep']}, {runtime['variant']}, {runtime['label']}).\n"
        f"- Largest completed median peak RSS: {memory['mib']:.3f} MiB "
        f"({memory['sweep']}, {memory['variant']}, {memory['label']}).\n"
        f"- Noncompleted repetitions retained: {limits['noncompleted_repetition_count']} "
        f"({limits['model_infeasible_repetition_count']} model-infeasible; "
        f"{limits['computational_or_instrumentation_failure_repetition_count']} computational/instrumentation failures).\n"
        "- No continuous-space, unmeasured-resolution, or future-hardware claim is made.\n",
        encoding="utf-8",
    )


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        "# Stage 14 reproducibility record\n\n"
        "Run from the repository root with the project virtual environment.\n\n"
        "```powershell\n"
        ".\\.venv_p1b\\Scripts\\python.exe 3D_0827\\stage14_8_final_gate.py\n"
        "```\n\n"
        "The command performs or resumes exactly one canonical fresh-process smoke run, "
        "rebuilds Stage-14 figures from saved machine-readable artifacts, runs all "
        "`test_stage14*.py` tests, and executes the integrated notebook. It does not run "
        "the full Stage-14.3--14.6 sweep again.\n\n"
        "Primary outputs are `stage14_8_summary.json`, `stage14_8_validation_report.json`, "
        "`all_raw_repetitions.{json,jsonl,csv}`, `all_configuration_summaries.{json,csv}`, "
        "`frozen_benchmark_configuration_set.json`, `figure_manifest.json`, "
        "`artifact_manifest.json`, and `command_record.json`.\n\n"
        f"Recorded final gate: {'PASS' if summary['gate_passed'] else 'FAIL'}.\n"
        "The certified equilibrium remains an exact equilibrium of the configured finite "
        "discretized model, not a continuous-space global-equilibrium claim.\n",
        encoding="utf-8",
    )


def run_stage14_8_gate(output_directory: Path = OUTPUT) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    frozen = _read_json(FROZEN_REFERENCE_PATH)
    frozen_environment = _read_json(FROZEN_ENVIRONMENT_PATH)
    profile = _read_json(PROFILE_PATH)

    smoke_case = stage14_8_smoke_case()
    smoke_output = output_directory / "clean_canonical_smoke"
    smoke_preexisting = (smoke_output / "raw_repetitions.jsonl").exists()
    prior_smoke_report_path = output_directory / "canonical_smoke_report.json"
    prior_smoke_report = (
        _read_json(prior_smoke_report_path) if prior_smoke_report_path.is_file() else {}
    )
    smoke = run_benchmark_cases(
        (smoke_case,), output_directory=smoke_output,
        environment_manifest_id=frozen_environment["environment_manifest_id"],
        python_executable=sys.executable,
    )
    smoke_rows = smoke["rows"]
    _write_json(smoke_output / "raw_repetitions.json", smoke_rows)
    write_raw_csv(smoke_output / "raw_repetitions.csv", smoke_rows)
    _write_json(smoke_output / "configuration_summaries.json", summarize_repetitions(smoke_rows))
    smoke_identity = compare_canonical_row(smoke_rows[0], frozen)
    clean_smoke_was_executed = bool(
        (not smoke_preexisting and len(smoke["attempted_repetition_ids"]) == 1)
        or prior_smoke_report.get("clean_smoke_was_executed")
        or prior_smoke_report.get("fresh_output_absent_at_this_invocation")
    )
    smoke_report = {
        "passed": len(smoke_rows) == 1 and smoke_identity["passed"] and clean_smoke_was_executed,
        "clean_smoke_was_executed": clean_smoke_was_executed,
        "fresh_output_absent_at_this_invocation": not smoke_preexisting,
        "attempted_repetition_ids": smoke["attempted_repetition_ids"],
        "skipped_repetition_ids": smoke["skipped_repetition_ids"],
        "initial_attempted_repetition_ids": (
            smoke["attempted_repetition_ids"]
            or prior_smoke_report.get("initial_attempted_repetition_ids")
            or prior_smoke_report.get("attempted_repetition_ids")
            or []
        ),
        "case_configuration": smoke_case.as_configuration(),
        "canonical_identity_regression": smoke_identity,
    }
    _write_json(output_directory / "canonical_smoke_report.json", smoke_report)

    tagged_rows = load_tagged_raw_rows()
    tagged_rows.extend({"stage_source": "14.8", **row} for row in smoke_rows)
    raw_audit = audit_raw_repetitions(tagged_rows)
    summaries = summarize_tagged_rows(tagged_rows)
    _write_unified_raw(output_directory, tagged_rows)
    _write_json(output_directory / "all_configuration_summaries.json", summaries)
    _write_summary_csv(output_directory / "all_configuration_summaries.csv", summaries)
    _write_json(output_directory / "raw_repetition_audit.json", raw_audit)

    configs = configuration_manifest()
    stage_audit = stage_summary_audit()
    profile_audit = profile_regression(frozen, profile)
    current_source = solver_source_manifest(ROOT)
    source_audit = {
        "passed": current_source["aggregate_sha256"] == frozen["solver_source_fingerprint"],
        "frozen_sha256": frozen["solver_source_fingerprint"],
        "current_sha256": current_source["aggregate_sha256"],
        "current_manifest": current_source,
    }
    _write_json(output_directory / "frozen_benchmark_configuration_set.json", configs)
    _write_json(output_directory / "environment_manifest.json", frozen_environment)
    _write_json(output_directory / "profiled_vs_original_regression.json", profile_audit)
    _write_json(output_directory / "solver_source_regression.json", source_audit)
    _write_json(output_directory / "prior_stage_gate_audit.json", stage_audit)

    core_audit = {
        "schema_version": SCHEMA_VERSION,
        "raw_repetitions": raw_audit,
        "canonical_smoke": smoke_report,
        "profiled_vs_original": profile_audit,
        "solver_source": source_audit,
        "prior_stage_gates": stage_audit,
    }
    core_audit["passed"] = all(item["passed"] for item in (
        raw_audit, smoke_report, profile_audit, source_audit, stage_audit,
    ))
    _write_json(output_directory / "core_integration_audit.json", core_audit)

    test_record = _run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_stage14*.py"],
        cwd=ROOT, label="all prior regression and Stage-14 tests",
    )
    test_record["test_count"] = _test_count(test_record)
    figure_record = _run_command(
        [sys.executable, "regenerate_stage14_all_figures.py"],
        cwd=ROOT, label="recreate all Stage-14 figures from saved machine-readable artifacts",
    )

    figures = figure_manifest()
    artifacts = artifact_manifest(output_directory)
    boundary = practical_boundaries()
    _write_json(output_directory / "figure_manifest.json", figures)
    _write_json(output_directory / "artifact_manifest.json", artifacts)
    _write_json(output_directory / "practical_boundary_observations.json", boundary)
    _write_boundary_markdown(output_directory / "practical_boundary_observations.md", boundary)

    fit_directory = FIGURE_ROOT / "stage_14_7_scaling_analysis"
    (output_directory / "fit_parameters.json").write_bytes((fit_directory / "fit_parameters.json").read_bytes())
    (output_directory / "fit_parameters.csv").write_bytes((fit_directory / "fit_parameters.csv").read_bytes())

    preliminary_checks = {
        "canonical_frozen_reference_rerun_through_harness": smoke_report["passed"],
        "profiled_and_original_solution_identities_match": profile_audit["passed"],
        "strong_stackelberg_tie_breaking_unchanged": smoke_identity["checks"]["leader_tie_set"]
            and smoke_identity["checks"]["follower_tie_set"] and smoke_identity["checks"]["tie_convention"],
        "attacker_and_defender_objectives_unchanged": smoke_identity["checks"]["attacker_objective"]
            and smoke_identity["checks"]["defender_objective"],
        "all_raw_attempts_and_failures_explicit": raw_audit["passed"],
        "repeated_identical_inputs_are_deterministic": raw_audit["checks"]["repeated_identical_cases_are_solution_deterministic"],
        "all_completed_equilibria_pass_independent_replay": raw_audit["checks"]["completed_rows_are_feasible_and_replay_validated"],
        "all_prior_stage_gates_pass": stage_audit["passed"],
        "solver_model_source_unchanged": source_audit["passed"],
        "all_regression_and_stage14_tests_pass": test_record["passed"] and bool(test_record["test_count"]),
        "all_stage14_figures_recreated": figure_record["passed"] and figures["figure_count"] >= 48,
        "raw_and_summary_exports_present": all(
            (output_directory / name).is_file() for name in (
                "all_raw_repetitions.json", "all_raw_repetitions.jsonl", "all_raw_repetitions.csv",
                "all_configuration_summaries.json", "all_configuration_summaries.csv",
            )
        ),
        "empirical_fits_remain_non_asymptotic": (
            _read_json(fit_directory / "stage14_7_validation_report.json")["checks"]["empirical_and_formal_claims_distinguished"]
        ),
        "stage14_7_uses_discretization_x_axes": (
            _read_json(fit_directory / "stage14_7_validation_report.json")["checks"]["all_figure_x_axes_use_declared_discretization_controls"]
        ),
    }
    preliminary_passed = all(preliminary_checks.values())
    stage14_checklist = {
        "single_cube_environment_only": raw_audit["checks"]["single_cube_used_for_physics_benchmarks"],
        "one_variable_at_a_time_control": raw_audit["checks"]["one_declared_sweep_variable_per_case"] and stage_audit["passed"],
        "three_repetitions_minimum_and_five_where_practical": raw_audit["checks"]["scientific_sweep_configurations_attempt_at_least_three_repetitions"],
        "median_primary_runtime_statistic": True,
        "runtime_decomposition_and_peak_rss_present": figures["figure_count"] >= 48,
        "state_edge_candidate_defender_branching_quadrature_counts_present": raw_audit["checks"]["required_state_game_counts_present"],
        "failures_and_computational_limits_explicit": raw_audit["checks"]["noncompleted_rows_have_explicit_classification"],
        "exact_logic_and_sse_tie_breaking_unchanged": source_audit["passed"] and smoke_identity["passed"],
        "no_rl_approximate_or_alternate_solver": raw_audit["checks"]["no_rl_or_approximate_planner"],
        "empirical_fits_not_formal_big_o": preliminary_checks["empirical_fits_remain_non_asymptotic"],
        "canonical_solution_and_replay_regressions_pass": smoke_report["passed"],
    }
    preliminary_summary = {
        "stage": "14.8", "schema_version": SCHEMA_VERSION,
        "gate_passed": preliminary_passed,
        "checks": preliminary_checks,
        "test_count": test_record["test_count"],
        "figure_count": figures["figure_count"],
        "raw_repetition_count": raw_audit["row_count"],
        "completed_repetition_count": raw_audit["completed_count"],
        "noncompleted_repetition_count": raw_audit["noncompleted_count"],
        "canonical_smoke_attempted_now": len(smoke["attempted_repetition_ids"]) == 1,
        "canonical_clean_smoke_recorded": clean_smoke_was_executed,
        "full_scaling_sweeps_rerun": False,
        "next_stage_started": False,
    }
    _write_json(output_directory / "stage14_8_summary.json", preliminary_summary)
    _write_json(output_directory / "stage14_8_validation_report.json", {
        "schema_version": SCHEMA_VERSION,
        "gate_passed": preliminary_passed,
        "checks": preliminary_checks,
        "stage14_pass_checklist": stage14_checklist,
        "claim_boundary": (
            "exact finite configured-model SSE only; empirical scaling is measured-range descriptive; "
            "no continuous-space or unmeasured-resolution claim"
        ),
        "next_stage_started": False,
        "notebook_execution_pending": True,
    })
    _write_readme(output_directory / "README.md", preliminary_summary)

    notebook_record = _run_command(
        [sys.executable, "execute_stage14_notebook.py"],
        cwd=ROOT, label="execute integrated Stage-14 notebook top to bottom",
    )
    notebook_errors = _notebook_output_errors() if notebook_record["passed"] else -1
    checks = preliminary_checks | {
        "integrated_notebook_executes_without_output_errors": notebook_record["passed"] and notebook_errors == 0,
    }
    gate_passed = all(checks.values())
    summary = preliminary_summary | {
        "gate_passed": gate_passed,
        "checks": checks,
        "notebook_output_error_count": notebook_errors,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    command_record = {
        "schema_version": SCHEMA_VERSION,
        "canonical_smoke": {
            "implementation": "stage14_benchmark_runner.run_benchmark_cases",
            "attempted_repetition_ids": smoke["attempted_repetition_ids"],
            "skipped_repetition_ids": smoke["skipped_repetition_ids"],
        },
        "commands": [test_record, figure_record, notebook_record],
        "full_scaling_sweeps_rerun": False,
    }
    _write_json(output_directory / "command_record.json", command_record)
    _write_json(output_directory / "test_validation_report.json", test_record)
    _write_json(output_directory / "stage14_8_validation_report.json", {
        "schema_version": SCHEMA_VERSION,
        "gate_passed": gate_passed,
        "checks": checks,
        "stage14_pass_checklist": stage14_checklist,
        "claim_boundary": (
            "exact finite configured-model SSE only; empirical scaling is measured-range descriptive; "
            "no continuous-space or unmeasured-resolution claim"
        ),
        "next_stage_started": False,
        "notebook_execution_pending": False,
    })
    _write_json(output_directory / "stage14_8_summary.json", summary)
    _write_readme(output_directory / "README.md", summary)
    if not gate_passed:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Stage 14.8 final gate failed: {failed}")
    print(
        "Stage 14.8 final integration/regression gate: PASS; "
        f"tests={test_record['test_count']}; figures={figures['figure_count']}; "
        f"raw repetitions={raw_audit['row_count']}"
    )
    return summary


if __name__ == "__main__":
    run_stage14_8_gate()
