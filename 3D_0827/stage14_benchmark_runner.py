"""Append-safe, fresh-process benchmark harness for Stage 14.2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter, sleep
from typing import Any, Iterable, Mapping, Sequence

import psutil


RUNNER_SCHEMA_VERSION = "stage14.2-v1"
TERMINAL_STATUSES = frozenset({
    "completed", "model_infeasible", "timeout", "memory_limit", "worker_failure",
    "initial_defender_infeasible", "numerical_failure", "computational_failure",
    "initial_neighborhood_infeasible", "initial_neighborhood_comparison_unknown",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BenchmarkCase:
    """Immutable case declaration with one explicit sweep variable."""

    experiment_type: str
    name: str
    worker_mode: str
    sweep_variable: str
    parameters: tuple[tuple[str, Any], ...]
    repetitions: int
    timeout_s: float
    memory_limit_bytes: int | None = None
    sampling_interval_s: float = 0.01

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.parameters)
        if not self.experiment_type or not self.name or not self.worker_mode:
            raise ValueError("experiment_type, name, and worker_mode are required")
        if len(set(names)) != len(names):
            raise ValueError("case parameter names must be unique")
        if self.sweep_variable not in names:
            raise ValueError("exactly one declared sweep_variable must name a parameter")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if self.memory_limit_bytes is not None and self.memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive when provided")
        if not math.isfinite(self.sampling_interval_s) or self.sampling_interval_s <= 0.0:
            raise ValueError("sampling_interval_s must be finite and positive")
        _canonical_json(self.as_configuration())

    @property
    def parameter_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    def as_configuration(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "name": self.name,
            "worker_mode": self.worker_mode,
            "sweep_variable": self.sweep_variable,
            "parameters": self.parameter_dict,
            "timeout_s": self.timeout_s,
            "memory_limit_bytes": self.memory_limit_bytes,
            "sampling_interval_s": self.sampling_interval_s,
            "process_semantics": "fresh cold process per repetition",
        }

    @property
    def case_id(self) -> str:
        digest = hashlib.sha256(
            _canonical_json(self.as_configuration()).encode("utf-8")
        ).hexdigest()[:12]
        slug = "".join(character if character.isalnum() else "-" for character in self.name)
        slug = "-".join(filter(None, slug.lower().split("-")))
        return f"{slug}-{digest}"

    def repetition_id(self, repetition_index: int) -> str:
        if repetition_index < 0 or repetition_index >= self.repetitions:
            raise IndexError("repetition index is outside the declared case")
        return f"{self.case_id}-r{repetition_index:03d}"


@dataclass(frozen=True)
class LimitedProcessResult:
    returncode: int | None
    limit_status: str | None
    stdout: str
    stderr: str
    wall_s: float
    start_rss_bytes: int
    peak_rss_bytes: int
    delta_peak_rss_bytes: int
    sampling_interval_s: float


def _process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _terminate_process_tree(child: subprocess.Popen[str]) -> None:
    """Terminate only the worker process tree launched by this runner."""
    try:
        parent = psutil.Process(child.pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        processes = []
    for process in reversed(processes):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(processes, timeout=2.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        child.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run_limited_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    memory_limit_bytes: int | None,
    sampling_interval_s: float,
) -> LimitedProcessResult:
    """Run one cold worker with parent-enforced wall-time and RSS ceilings."""
    started = perf_counter()
    child = subprocess.Popen(
        tuple(command), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    process = psutil.Process(child.pid)
    start_rss = 0
    peak_rss = 0
    limit_status: str | None = None
    while child.poll() is None:
        rss = _process_tree_rss(process)
        if rss > 0 and start_rss == 0:
            start_rss = rss
        peak_rss = max(peak_rss, rss)
        elapsed = perf_counter() - started
        if memory_limit_bytes is not None and rss > memory_limit_bytes:
            limit_status = "memory_limit"
            _terminate_process_tree(child)
            break
        if elapsed > timeout_s:
            limit_status = "timeout"
            _terminate_process_tree(child)
            break
        sleep(sampling_interval_s)
    rss = _process_tree_rss(process)
    if rss > 0 and start_rss == 0:
        start_rss = rss
    peak_rss = max(peak_rss, rss, start_rss)
    stdout, stderr = child.communicate()
    return LimitedProcessResult(
        returncode=child.returncode,
        limit_status=limit_status,
        stdout=stdout,
        stderr=stderr,
        wall_s=perf_counter() - started,
        start_rss_bytes=start_rss,
        peak_rss_bytes=peak_rss,
        delta_peak_rss_bytes=peak_rss - start_rss,
        sampling_interval_s=sampling_interval_s,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(dict(row)) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL file while preserving unrelated cases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    encoded = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_worker_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_attempt(
    case: BenchmarkCase,
    repetition_index: int,
    environment_manifest_id: str,
    process_result: LimitedProcessResult,
    payload: dict[str, Any] | None,
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    if process_result.limit_status is not None:
        status = process_result.limit_status
        failure_category = status
        failure_message = (
            f"parent-enforced {status}: timeout_s={case.timeout_s}, "
            f"memory_limit_bytes={case.memory_limit_bytes}"
        )
    elif process_result.returncode != 0:
        status = "worker_failure"
        failure_category = "nonzero_worker_exit"
        failure_message = process_result.stderr.strip() or "worker returned nonzero"
    elif payload is None:
        status = "worker_failure"
        failure_category = "missing_worker_payload"
        failure_message = "worker exited without a result payload"
    else:
        status = str(payload.get("status", "worker_failure"))
        failure_category = payload.get("failure_type")
        failure_message = payload.get("failure_message")
    solution = (payload or {}).get("solution_identity") or {}
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "experiment_type": case.experiment_type,
        "case_id": case.case_id,
        "repetition_id": case.repetition_id(repetition_index),
        "repetition_index": repetition_index,
        "configuration": case.as_configuration(),
        "declared_sweep_variable": case.sweep_variable,
        "declared_sweep_value": case.parameter_dict[case.sweep_variable],
        "process_semantics": "fresh cold process",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "environment_manifest_id": environment_manifest_id,
        "status": status,
        "failure_category": failure_category,
        "failure_message": failure_message,
        "worker_returncode": process_result.returncode,
        "worker_wall_s": process_result.wall_s,
        "timing": (payload or {}).get("timing"),
        "T_SSE_attempt_s": (payload or {}).get("T_SSE_attempt_s"),
        "state_and_game_size": (payload or {}).get("state_and_game_size"),
        "memory": {
            "method": "psutil sampled worker process-tree RSS",
            "start_rss_bytes": process_result.start_rss_bytes,
            "peak_rss_bytes": process_result.peak_rss_bytes,
            "delta_peak_rss_bytes": process_result.delta_peak_rss_bytes,
            "sampling_interval_s": process_result.sampling_interval_s,
            "native_allocations_included": True,
            "child_processes_included": True,
        },
        "J_A": solution.get("attacker_objective"),
        "J_D": solution.get("defender_objective_pod"),
        "feasible": solution.get("feasible", status == "completed"),
        "selected_defender_action_id": solution.get("selected_defender_action_id"),
        "selected_attacker_candidate_id": solution.get("selected_attacker_candidate_id"),
        "trajectory_identity": solution.get("trajectory_identity"),
        "tie_breaking_result": {
            "leader_cooptimal_action_ids": solution.get("leader_cooptimal_action_ids"),
            "attacker_objective_cooptimal_candidate_ids": solution.get(
                "attacker_objective_cooptimal_candidate_ids"
            ),
            "convention": solution.get("tie_break_convention"),
        },
        "independent_replay_passed": (payload or {}).get(
            "independent_replay", {}
        ).get("passed"),
        "solver_source_fingerprint": (payload or {}).get("solver_source_fingerprint"),
        "algorithm_variant": (payload or {}).get("algorithm_variant"),
        "local_search": (payload or {}).get("local_search"),
        "exactness": (payload or {}).get("exactness"),
        "oracle_metadata": (payload or {}).get("oracle_metadata"),
        "worker_configuration": (payload or {}).get("complete_configuration"),
        "realized_grid": (payload or {}).get("realized_grid"),
        "worker_stdout": process_result.stdout.strip(),
        "worker_stderr": process_result.stderr.strip(),
    }


def run_benchmark_cases(
    cases: Iterable[BenchmarkCase],
    *,
    output_directory: Path,
    environment_manifest_id: str,
    force: bool = False,
    python_executable: str = sys.executable,
    worker_path: Path | None = None,
) -> dict[str, Any]:
    """Run missing repetitions and append each terminal attempt atomically."""
    cases = tuple(cases)
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_path = output_directory / "raw_repetitions.jsonl"
    existing = read_jsonl(raw_path)
    requested_ids = {
        case.repetition_id(repetition_index)
        for case in cases
        for repetition_index in range(case.repetitions)
    }
    if force:
        # A forced run means fresh measurements, not duplicate JSONL rows.
        # Remove only the requested cases; unrelated benchmark rows survive.
        existing = [
            row for row in existing if row.get("repetition_id") not in requested_ids
        ]
        _replace_jsonl(raw_path, existing)
    completed_ids = {
        row["repetition_id"] for row in existing if row.get("status") in TERMINAL_STATUSES
    }
    worker = worker_path or Path(__file__).with_name("stage14_2_worker.py")
    worker_directory = output_directory / "worker_payloads"
    worker_directory.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []
    skipped: list[str] = []
    for case in cases:
        for repetition_index in range(case.repetitions):
            repetition_id = case.repetition_id(repetition_index)
            if repetition_id in completed_ids and not force:
                skipped.append(repetition_id)
                continue
            config_path = worker_directory / f"{repetition_id}.configuration.json"
            result_path = worker_directory / f"{repetition_id}.result.json"
            configuration = case.as_configuration() | {
                "case_id": case.case_id,
                "repetition_id": repetition_id,
            }
            config_path.write_text(
                json.dumps(configuration, indent=2, sort_keys=True), encoding="utf-8"
            )
            if result_path.exists():
                result_path.unlink()
            started_at = _utc_now()
            result = run_limited_subprocess(
                (
                    python_executable, str(worker), "--configuration", str(config_path),
                    "--output", str(result_path),
                ),
                cwd=Path(__file__).resolve().parent,
                timeout_s=case.timeout_s,
                memory_limit_bytes=case.memory_limit_bytes,
                sampling_interval_s=case.sampling_interval_s,
            )
            ended_at = _utc_now()
            payload = _load_worker_payload(result_path)
            row = _row_from_attempt(
                case, repetition_index, environment_manifest_id, result, payload,
                started_at, ended_at,
            )
            append_jsonl(raw_path, row)
            attempted.append(repetition_id)
            completed_ids.add(repetition_id)
    return {
        "attempted_repetition_ids": attempted,
        "skipped_repetition_ids": skipped,
        "raw_jsonl": str(raw_path),
        "rows": read_jsonl(raw_path),
    }


def numeric_statistics(values: Sequence[float]) -> dict[str, Any] | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return {
        "count": len(clean),
        "individual": clean,
        "median": median(clean),
        "mean": mean(clean),
        "standard_deviation_sample": stdev(clean) if len(clean) >= 2 else 0.0,
        "minimum": min(clean),
        "maximum": max(clean),
    }


def summarize_repetitions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        case_rows = sorted(grouped[case_id], key=lambda row: row["repetition_id"])
        completed = [row for row in case_rows if row["status"] == "completed"]
        sse_values = [
            row["timing"]["totals"]["T_SSE_s"]
            for row in completed if row.get("timing") is not None
        ]
        summaries.append({
            "case_id": case_id,
            "experiment_type": case_rows[0]["experiment_type"],
            "configuration": case_rows[0]["configuration"],
            "attempted_count": len(case_rows),
            "status_counts": dict(sorted(Counter(row["status"] for row in case_rows).items())),
            "worker_wall_statistics_s": numeric_statistics([
                row["worker_wall_s"] for row in case_rows
            ]),
            "T_SSE_statistics_s": numeric_statistics(sse_values),
            "exact_solution_identities": [
                {
                    "repetition_id": row["repetition_id"],
                    "J_A": row["J_A"],
                    "J_D": row["J_D"],
                    "selected_defender_action_id": row["selected_defender_action_id"],
                    "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                    "trajectory_sha256": (row.get("trajectory_identity") or {}).get("sha256"),
                    "independent_replay_passed": row["independent_replay_passed"],
                }
                for row in completed
            ],
        })
    return summaries


def write_raw_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "experiment_type", "case_id", "repetition_id", "repetition_index",
        "declared_sweep_variable", "declared_sweep_value", "status",
        "failure_category", "failure_message", "started_at_utc", "ended_at_utc",
        "environment_manifest_id", "worker_wall_s", "T_SSE_s",
        "start_rss_bytes", "peak_rss_bytes", "delta_peak_rss_bytes",
        "J_A", "J_D", "feasible", "selected_defender_action_id",
        "selected_attacker_candidate_id", "trajectory_sha256",
        "independent_replay_passed", "configuration_json", "counts_json",
        "timing_json", "tie_breaking_json",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            timing = row.get("timing") or {}
            memory = row["memory"]
            writer.writerow({
                "experiment_type": row["experiment_type"],
                "case_id": row["case_id"],
                "repetition_id": row["repetition_id"],
                "repetition_index": row["repetition_index"],
                "declared_sweep_variable": row["declared_sweep_variable"],
                "declared_sweep_value": row["declared_sweep_value"],
                "status": row["status"],
                "failure_category": row["failure_category"],
                "failure_message": row["failure_message"],
                "started_at_utc": row["started_at_utc"],
                "ended_at_utc": row["ended_at_utc"],
                "environment_manifest_id": row["environment_manifest_id"],
                "worker_wall_s": row["worker_wall_s"],
                "T_SSE_s": (timing.get("totals") or {}).get("T_SSE_s"),
                "start_rss_bytes": memory["start_rss_bytes"],
                "peak_rss_bytes": memory["peak_rss_bytes"],
                "delta_peak_rss_bytes": memory["delta_peak_rss_bytes"],
                "J_A": row["J_A"], "J_D": row["J_D"],
                "feasible": row["feasible"],
                "selected_defender_action_id": row["selected_defender_action_id"],
                "selected_attacker_candidate_id": row["selected_attacker_candidate_id"],
                "trajectory_sha256": (row.get("trajectory_identity") or {}).get("sha256"),
                "independent_replay_passed": row["independent_replay_passed"],
                "configuration_json": _canonical_json(row["configuration"]),
                "counts_json": _canonical_json(row["state_and_game_size"]),
                "timing_json": _canonical_json(row["timing"]),
                "tie_breaking_json": _canonical_json(row["tie_breaking_result"]),
            })


__all__ = [
    "BenchmarkCase", "LimitedProcessResult", "RUNNER_SCHEMA_VERSION",
    "TERMINAL_STATUSES", "append_jsonl", "numeric_statistics", "read_jsonl",
    "run_benchmark_cases", "run_limited_subprocess", "summarize_repetitions",
    "write_raw_csv",
]
